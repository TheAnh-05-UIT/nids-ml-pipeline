"""Store stable timestamp/ordinal processing references in SQLite.
Lưu processing references theo timestamp/ordinal ổn định bằng SQLite.
"""
from contextlib import closing
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import shutil
import sqlite3
import time

from nids_ml.flows.features import CaptureContext
from nids_ml.labeling.contracts import canonical_bytes, strict_json
from nids_ml.ordering.contracts import (FORMAT, VERSION, OrderingError, OrderingLimitError,
    OrderingLimits, packet_bytes, validate_source, validate_view_manifest)
from nids_ml.pcap.decoder import file_sha256, file_signature, regular_file

SELECT_ORDER = 'SELECT timestamp_ns,source_ordinal,packet_sha256 FROM packets ORDER BY timestamp_ns,source_ordinal'


@dataclass(frozen=True)
class ProcessingRecord:
    processing_index: int
    source_packet_ordinal: int
    timestamp_ns: int
    source_packet_sha256: str
    capture: CaptureContext

    def to_dict(self):
        return dict(processing_index=self.processing_index, source_packet_ordinal=self.source_packet_ordinal,
                    timestamp_ns=self.timestamp_ns, source_packet_sha256=self.source_packet_sha256,
                    capture=self.capture.to_dict())


def _new_json(path, value):
    with path.open('xb') as file:
        file.write(canonical_bytes(value)+b'\n')


def _fresh_directory(path):
    path = Path(path).absolute()
    if path.exists() or path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise OrderingError('FRESH_INDEPENDENT_OUTPUT_DIRECTORY_REQUIRED')
    if not path.parent.is_dir():
        raise OrderingError('EXISTING_OUTPUT_PARENT_REQUIRED')
    return path


def _processing_digest(source):
    value = hashlib.sha256(b'nids-processing-order/1\x00')
    value.update(hashlib.sha256(canonical_bytes(source)).digest())
    return value


def _ordered_records(db, source):
    context = CaptureContext.from_dict(source['capture'])
    for index, (timestamp, ordinal, digest) in enumerate(db.execute(SELECT_ORDER), 1):
        if (type(timestamp) is not int or not 0<=timestamp<=2**63-1
                or type(ordinal) is not int or not 1<=ordinal<=2**63-1
                or type(digest) is not bytes or len(digest)!=32):
            raise OrderingError('INVALID_INDEX_RECORD')
        yield ProcessingRecord(index, ordinal, timestamp, digest.hex(), context)


def _summarize(db, source):
    digest = _processing_digest(source)
    count = moved = 0
    first = last = None
    previous = None
    for record in _ordered_records(db, source):
        key = (record.timestamp_ns, record.source_packet_ordinal)
        if previous is not None and key<=previous:
            raise OrderingError('INDEX_ORDER_NOT_STRICT')
        previous = key
        if first is None:
            first = record.timestamp_ns
        last = record.timestamp_ns
        count += 1
        if count>source['expected_packets']:
            raise OrderingError('INDEX_EXCEEDS_DECLARED_COUNT')
        moved += record.source_packet_ordinal != source['source_first_ordinal']+record.processing_index-1
        digest.update(canonical_bytes(record.to_dict())+b'\n')
    if count!=source['expected_packets']:
        raise OrderingError('INDEX_RECORD_COUNT_MISMATCH')
    bounds = db.execute('SELECT min(source_ordinal),max(source_ordinal) FROM packets').fetchone()
    if bounds != (source['source_first_ordinal'], source['source_first_ordinal']+count-1):
        raise OrderingError('INDEX_SOURCE_ORDINAL_RANGE')
    return dict(record_count=count, processing_order_sha256=digest.hexdigest(),
                moved_records=moved, min_timestamp_ns=first, max_timestamp_ns=last)


def _source_stats(db, source):
    """Kiểm lại thứ tự nguồn và digest từ index mà không giữ tập ordinals trong RAM."""
    digest = hashlib.sha256(b'nids-source-order/1\x00')
    count = regressions = max_regression = 0
    previous = None
    for timestamp, ordinal, packet_digest in db.execute(
            'SELECT timestamp_ns,source_ordinal,packet_sha256 FROM packets ORDER BY source_ordinal'):
        if ordinal!=source['source_first_ordinal']+count:
            raise OrderingError('INDEX_SOURCE_ORDINAL_GAP_OR_DUPLICATE')
        count += 1
        if count>source['expected_packets']:
            raise OrderingError('INDEX_SOURCE_COUNT_EXCEEDED')
        digest.update(packet_digest)
        if previous is not None and timestamp<previous:
            regressions += 1
            max_regression = max(max_regression, previous-timestamp)
        previous = timestamp
    if count!=source['expected_packets'] or digest.hexdigest()!=source['expected_source_order_sha256']:
        raise OrderingError('INDEX_SOURCE_DIGEST_OR_COUNT_MISMATCH')
    return dict(source_timestamp_regressions=regressions, max_regression_ns=max_regression)


def build_processing_view(packets, source, output, *, limits=None):
    """Hoàn tất index trước khi publish manifest; partial/failed artifacts được giữ riêng."""
    source = validate_source(source)
    limits = (OrderingLimits() if limits is None else limits)
    if not isinstance(limits, OrderingLimits):
        raise OrderingError('ORDERING_LIMITS_REQUIRED')
    limits.validate()
    if source['expected_packets']>limits.max_records:
        raise OrderingLimitError('EXPECTED_COUNT_EXCEEDS_BUDGET')
    output = _fresh_directory(output)
    if shutil.disk_usage(output.parent).free<limits.min_free_bytes:
        raise OrderingLimitError('DISK_HEADROOM')
    output.mkdir(exist_ok=False)
    _new_json(output/'plan.json', dict(source=source, limits=asdict(limits), status='BUILDING_SYNTHETIC_VIEW'))
    started = time.monotonic()
    path = output/'index.sqlite'
    count = regressions = max_regression = 0
    previous_timestamp = None
    source_digest = hashlib.sha256(b'nids-source-order/1\x00')

    def budget():
        if time.monotonic()-started>limits.timeout_seconds:
            raise OrderingLimitError('ORDERING_TIMEOUT')
        if sum(p.stat().st_size for p in output.iterdir() if p.is_file())>limits.scratch_bytes:
            raise OrderingLimitError('SCRATCH_ABORT_THRESHOLD')
        if shutil.disk_usage(output).free<limits.min_free_bytes:
            raise OrderingLimitError('DISK_HEADROOM')

    try:
        with closing(sqlite3.connect(path)) as db:
            db.execute('PRAGMA page_size=4096')
            db.execute('PRAGMA max_page_count='+str(limits.sqlite_bytes//4096))
            db.execute('PRAGMA cache_size=-'+str(limits.cache_kib))
            db.execute('PRAGMA journal_mode=DELETE')
            db.execute('PRAGMA temp_store=FILE')
            db.set_progress_handler(lambda: int(time.monotonic()-started>limits.timeout_seconds),10000)
            # PK là ordering index tăng dần; không cần sort toàn bộ rows vào RAM/temp.
            db.execute('CREATE TABLE packets(timestamp_ns INTEGER NOT NULL,source_ordinal INTEGER NOT NULL UNIQUE,'
                       'packet_sha256 BLOB NOT NULL,PRIMARY KEY(timestamp_ns,source_ordinal)) WITHOUT ROWID')
            db.commit()
            for packet in packets:
                count += 1
                if count>limits.max_records or count>source['expected_packets']:
                    raise OrderingLimitError('EXTRA_PACKET_OR_RECORD_LIMIT')
                data = packet_bytes(packet)
                if packet['capture']!=source['capture']:
                    raise OrderingError('CAPTURE_RUN_SEGMENT_MISMATCH')
                if packet['ordinal']!=source['source_first_ordinal']+count-1:
                    raise OrderingError('SOURCE_ORDINAL_MISSING_DUPLICATE_OR_REGRESSION')
                timestamp = packet['timestamp_ns']
                if previous_timestamp is not None and timestamp<previous_timestamp:
                    regressions += 1
                    max_regression = max(max_regression, previous_timestamp-timestamp)
                previous_timestamp = timestamp
                packet_digest = hashlib.sha256(data).digest()
                source_digest.update(packet_digest)
                db.execute('INSERT INTO packets VALUES(?,?,?)', (timestamp, packet['ordinal'], packet_digest))
                if count%limits.commit_every==0:
                    db.commit()
                    budget()
            if count!=source['expected_packets']:
                raise OrderingError('SOURCE_INCOMPLETE_OR_PACKET_MISSING')
            if source_digest.hexdigest()!=source['expected_source_order_sha256']:
                raise OrderingError('SOURCE_ORDER_DIGEST_MISMATCH')
            db.commit()
            summary = _summarize(db, source)
            budget()
        if path.stat().st_size>limits.sqlite_bytes:
            raise OrderingLimitError('SQLITE_SIZE_LIMIT')
        manifest = dict(format=FORMAT, schema_version=VERSION, status='COMPLETE_SYNTHETIC_VIEW',
            source=source, record_count=count, processing_order_sha256=summary['processing_order_sha256'],
            index=dict(filename='index.sqlite', bytes=path.stat().st_size, sha256=file_sha256(path)),
            limits=asdict(limits), stats=dict(source_timestamp_regressions=regressions,
                max_regression_ns=max_regression, moved_records=summary['moved_records'],
                min_timestamp_ns=summary['min_timestamp_ns'], max_timestamp_ns=summary['max_timestamp_ns']),
            labels='NOT_EVALUATED', flow_integration='NOT_ACTIVE', real_data_export='BLOCKED')
        validate_view_manifest(manifest)
        _new_json(output/'manifest.json', manifest)
        return manifest
    except Exception as error:
        _new_json(output/'FAILED.json', dict(status='FAILED_RETAINED', error=str(error),
                  received_packets=count, real_data_export='BLOCKED', automatic_cleanup=False))
        if isinstance(error, sqlite3.Error):
            raise OrderingError('SQLITE_ORDERING_FAILED:'+str(error)) from error
        raise


def _read_manifest(output):
    output = Path(output).absolute()
    if (output/'FAILED.json').exists():
        raise OrderingError('FAILED_VIEW_IS_NOT_READABLE')
    manifest_file = regular_file(output/'manifest.json')
    with manifest_file.open('rb') as stream:
        manifest = validate_view_manifest(strict_json(stream.read(1048577)))
    path = regular_file(output/manifest['index']['filename'])
    if path.stat().st_size!=manifest['index']['bytes'] or path.stat().st_size>manifest['limits']['sqlite_bytes']:
        raise OrderingError('INDEX_SIZE_MISMATCH')
    if file_sha256(path)!=manifest['index']['sha256']:
        raise OrderingError('INDEX_HASH_MISMATCH')
    return manifest, path


def _open_readonly(path, cache_kib):
    db = sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1', uri=True)
    db.execute('PRAGMA query_only=ON')
    db.execute('PRAGMA trusted_schema=OFF')
    db.execute('PRAGMA cache_size=-'+str(cache_kib))
    kind = db.execute("SELECT type FROM sqlite_master WHERE name='packets'").fetchone()
    if kind!=('table',):
        db.close()
        raise OrderingError('PACKETS_TABLE_REQUIRED')
    return db


def validate_processing_view(output):
    """Đọc hết index để kiểm semantic digest; không decode PCAP hoặc load ML."""
    manifest, path = _read_manifest(output)
    signature = file_signature(path)
    with closing(_open_readonly(path, manifest['limits']['cache_kib'])) as db:
        started = time.monotonic()
        db.set_progress_handler(lambda: int(time.monotonic()-started>manifest['limits']['timeout_seconds']),10000)
        summary = _summarize(db, manifest['source'])
        original_stats = _source_stats(db, manifest['source'])
    if (summary['processing_order_sha256']!=manifest['processing_order_sha256']
            or summary['moved_records']!=manifest['stats']['moved_records']
            or summary['min_timestamp_ns']!=manifest['stats']['min_timestamp_ns']
            or summary['max_timestamp_ns']!=manifest['stats']['max_timestamp_ns']
            or any(manifest['stats'][k]!=v for k,v in original_stats.items())):
        raise OrderingError('PROCESSING_VIEW_DIGEST_OR_STATS_MISMATCH')
    if file_signature(path)!=signature:
        raise OrderingError('INDEX_CHANGED_DURING_VALIDATION')
    return manifest


def iter_processing_view(output):
    """Validate toàn index trước khi yield references; caller phải consume hết stream."""
    manifest = validate_processing_view(output)
    path = regular_file(Path(output).absolute()/'index.sqlite')
    signature = file_signature(path)
    with closing(_open_readonly(path, manifest['limits']['cache_kib'])) as db:
        yield from _ordered_records(db, manifest['source'])
    if file_signature(path)!=signature:
        raise OrderingError('INDEX_CHANGED_DURING_ITERATION')
