"""Index real PCAP records in stable order and seek original bytes without rewriting them.
Lập index thứ tự ổn định cho PCAP thật, tra bytes gốc; không sửa raw hoặc gán nhãn.
"""
from contextlib import closing
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import struct
import time

from nids_ml.flows.features import CaptureContext
from nids_ml.labeling.contracts import canonical_bytes, strict_json
from nids_ml.ordering.contracts import OrderingError, OrderingLimitError
from nids_ml.ordering.view import _fresh_directory, _new_json
from nids_ml.pcap.decoder import file_sha256, file_signature, regular_file
from nids_ml.pcap.metadata import scan_record_headers

FORMAT='nids-pcap-record-processing-view/1.0.0'
COLUMNS='timestamp_ns,source_ordinal,record_offset,captured_length,original_length,raw_record_sha256'


def policy():
    return dict(id='stable-raw-pcap-record-view',version='1.0.0',
        sort_keys=['timestamp_ns','source_ordinal'],raw_mutation=False,timestamp_correction=False,
        hash_domain='SHA256(original_16_byte_record_header_plus_captured_bytes)',
        labels='NOT_EVALUATED',flow_integration='NOT_ACTIVE',HDF5_export='BLOCKED')


@dataclass(frozen=True)
class PcapViewLimits:
    max_records:int=250000
    sqlite_bytes:int=64*1048576
    cache_kib:int=1024
    scratch_bytes:int=160*1048576
    min_free_bytes:int=2*1024**3
    timeout_seconds:int=300
    commit_every:int=256

    def validate(self):
        bounds=dict(max_records=(1,250000),sqlite_bytes=(32768,128*1048576),cache_kib=(64,8192),
            scratch_bytes=(131072,320*1048576),min_free_bytes=(2*1024**3,4*1024**3),
            timeout_seconds=(1,300),commit_every=(1,512))
        for name,(low,high) in bounds.items():
            if type(getattr(self,name)) is not int or not low<=getattr(self,name)<=high:
                raise OrderingLimitError('PCAP_VIEW_LIMIT:'+name)
        if self.sqlite_bytes%4096 or self.scratch_bytes<2*self.sqlite_bytes+65536:
            raise OrderingLimitError('PCAP_VIEW_SQLITE_OR_SCRATCH_BUDGET')
        return self


def validate_source(source):
    if type(source) is not dict or set(source)!={'capture','path','size_bytes','expected_records'}:
        raise OrderingError('PCAP_VIEW_SOURCE_KEYS')
    CaptureContext.from_dict(source['capture'])
    relative=source['path']
    if (type(relative) is not str or '\\' in relative or ':' in relative
            or PurePosixPath(relative).is_absolute() or '..' in PurePosixPath(relative).parts
            or PurePosixPath(relative).parts[:2]!=('data','raw') or not relative.endswith('.pcap')):
        raise OrderingError('PCAP_VIEW_SOURCE_PATH')
    if (type(source['size_bytes']) is not int or not 24<=source['size_bytes']<=1024**3
            or type(source['expected_records']) is not int or not 1<=source['expected_records']<=250000):
        raise OrderingError('PCAP_VIEW_SOURCE_SIZE_OR_COUNT')
    return strict_json(canonical_bytes(source))


@dataclass(frozen=True)
class PcapRecordReference:
    processing_index:int
    source_ordinal:int
    timestamp_ns:int
    record_offset:int
    captured_length:int
    original_length:int
    raw_record_sha256:str
    capture:dict

    def to_dict(self):
        return asdict(self)


def _records(db,source,*,source_order=False):
    order='source_ordinal' if source_order else 'timestamp_ns,source_ordinal'
    for number,row in enumerate(db.execute('SELECT '+COLUMNS+',processing_index FROM packets ORDER BY '+order),1):
        timestamp,ordinal,offset,caplen,wirelen,digest,stored_index=row
        if (any(type(value) is not int for value in row[:5])
                or not 0<=timestamp<=2**63-1 or not 1<=ordinal<=source['expected_records']
                or not 24<=offset<source['size_bytes'] or not 0<=caplen<=1048576
                or not caplen<=wirelen<=2**32-1 or offset+16+caplen>source['size_bytes']
                or type(digest) is not bytes or len(digest)!=32
                or type(stored_index) is not int or not 1<=stored_index<=source['expected_records']
                or (not source_order and stored_index!=number)):
            raise OrderingError('PCAP_INDEX_RECORD_RANGE_OR_TYPE')
        yield PcapRecordReference(number,ordinal,timestamp,offset,caplen,wirelen,digest.hex(),source['capture'])


def _summary(db,source):
    source_digest=hashlib.sha256(b'nids-pcap-record-source-order/1\0')
    previous=None;regressions=max_regression=count=0;offset=24
    for record in _records(db,source,source_order=True):
        count+=1
        if record.source_ordinal!=count or record.record_offset!=offset:
            raise OrderingError('PCAP_INDEX_SOURCE_GAP_DUPLICATE_OR_OFFSET')
        offset+=16+record.captured_length
        if previous is not None and record.timestamp_ns<previous:
            regressions+=1;max_regression=max(max_regression,previous-record.timestamp_ns)
        previous=record.timestamp_ns
        source_digest.update(bytes.fromhex(record.raw_record_sha256))
    if count!=source['expected_records'] or offset!=source['size_bytes']:
        raise OrderingError('PCAP_INDEX_INCOMPLETE_SOURCE')
    ordered=hashlib.sha256(b'nids-pcap-record-processing-order/1\0'+canonical_bytes(source)+canonical_bytes(policy()))
    previous=None;moved=number=0;minimum=maximum=None
    for record in _records(db,source):
        number+=1;key=(record.timestamp_ns,record.source_ordinal)
        if previous is not None and key<=previous:raise OrderingError('PCAP_PROCESSING_ORDER')
        previous=key;moved+=record.processing_index!=record.source_ordinal
        if minimum is None:minimum=record.timestamp_ns
        maximum=record.timestamp_ns
        ordered.update(canonical_bytes(record.to_dict())+b'\n')
    if number!=count:raise OrderingError('PCAP_PROCESSING_COUNT')
    return dict(record_count=count,source_order_sha256=source_digest.hexdigest(),
        processing_order_sha256=ordered.hexdigest(),source_timestamp_regressions=regressions,
        max_regression_ns=max_regression,moved_records=moved,min_timestamp_ns=minimum,max_timestamp_ns=maximum)


def _readonly(path,limits):
    db=sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True)
    try:
        db.execute('PRAGMA query_only=ON');db.execute('PRAGMA trusted_schema=OFF')
        db.execute('PRAGMA cache_size=-'+str(limits.cache_kib))
        if db.execute("SELECT type FROM sqlite_master WHERE name='packets'").fetchone()!=('table',):
            raise OrderingError('PCAP_PACKETS_TABLE_REQUIRED')
        started=time.monotonic()
        db.set_progress_handler(lambda:int(time.monotonic()-started>limits.timeout_seconds),10000)
        return db
    except BaseException:
        db.close();raise


def build_pcap_view(source,workspace,output,*,limits=None,progress=None):
    """Publish only after EOF/hash/count agree; keep failed work, never overwrite.
    Chỉ publish khi EOF/hash/count đúng; giữ output lỗi và không ghi đè.
    """
    source=validate_source(source);limits=(PcapViewLimits() if limits is None else limits).validate()
    if source['expected_records']>limits.max_records:raise OrderingLimitError('PCAP_VIEW_EXPECTED_COUNT_BUDGET')
    workspace=Path(workspace).resolve();path=regular_file(workspace/source['path'])
    output=_fresh_directory(output)
    if not output.resolve().is_relative_to(workspace/'artifacts/runs'):
        raise OrderingError('PCAP_VIEW_OUTPUT_UNDER_ARTIFACTS_RUNS_REQUIRED')
    if path.stat().st_size!=source['size_bytes']:raise OrderingError('PCAP_VIEW_SOURCE_SIZE_MISMATCH')
    if shutil.disk_usage(output.parent).free<limits.scratch_bytes+limits.min_free_bytes:
        raise OrderingLimitError('PCAP_VIEW_DISK_RESERVE_PLUS_SCRATCH')
    signature=file_signature(path);output.mkdir()
    _new_json(output/'plan.json',dict(format=FORMAT,source=source,policy=policy(),limits=asdict(limits)))
    started=time.monotonic();db=None

    def budget():
        if time.monotonic()-started>limits.timeout_seconds:raise OrderingLimitError('PCAP_VIEW_TIMEOUT')
        if sum(item.stat().st_size for item in output.iterdir() if item.is_file())>limits.scratch_bytes:
            raise OrderingLimitError('PCAP_VIEW_SCRATCH_LIMIT')
        if shutil.disk_usage(output).free<limits.min_free_bytes:raise OrderingLimitError('PCAP_VIEW_DISK_RESERVE')

    try:
        index=output/'index.sqlite';db=sqlite3.connect(index)
        db.execute('PRAGMA page_size=4096');db.execute('PRAGMA max_page_count='+str(limits.sqlite_bytes//4096))
        db.execute('PRAGMA cache_size=-'+str(limits.cache_kib));db.execute('PRAGMA journal_mode=DELETE')
        db.execute('PRAGMA temp_store=MEMORY')
        db.set_progress_handler(lambda:int(time.monotonic()-started>limits.timeout_seconds),10000)
        db.execute('CREATE TABLE packets(timestamp_ns INTEGER NOT NULL,source_ordinal INTEGER NOT NULL UNIQUE,'
            'record_offset INTEGER NOT NULL,captured_length INTEGER NOT NULL,original_length INTEGER NOT NULL,'
            'raw_record_sha256 BLOB NOT NULL,processing_index INTEGER NOT NULL DEFAULT 0,'
            'PRIMARY KEY(timestamp_ns,source_ordinal)) WITHOUT ROWID')

        def sink(item):
            ordinal=item['source_ordinal']
            if ordinal>limits.max_records or ordinal>source['expected_records']:
                raise OrderingLimitError('PCAP_VIEW_EXTRA_RECORD_OR_LIMIT')
            db.execute('INSERT INTO packets('+COLUMNS+') VALUES(?,?,?,?,?,?)',
                (item['timestamp_ns'],ordinal,item['record_offset'],item['captured_length'],
                 item['original_length'],bytes.fromhex(item['raw_record_sha256'])))
            if ordinal%limits.commit_every==0:db.commit();budget()

        metadata=scan_record_headers(path,expected_sha256=source['capture']['capture_sha256'],
            expected_size=source['size_bytes'],record_sink=sink,progress=progress)
        if metadata['records']!=source['expected_records']:raise OrderingError('PCAP_VIEW_EXPECTED_COUNT_MISMATCH')
        db.commit()
        # Store rank once; lookup stays O(log N), not a COUNT scan per packet.
        # Ghi processing_index một lần; lookup không quét COUNT toàn index mỗi packet.
        for number,(ordinal,) in enumerate(db.execute('SELECT source_ordinal FROM packets ORDER BY timestamp_ns,source_ordinal'),1):
            db.execute('UPDATE packets SET processing_index=? WHERE source_ordinal=?',(number,ordinal))
            if number%limits.commit_every==0:budget()
        db.commit();summary=_summary(db,source);budget();db.close();db=None
        if index.stat().st_size>limits.sqlite_bytes:raise OrderingLimitError('PCAP_VIEW_SQLITE_SIZE')
        if (file_signature(path)!=signature or file_sha256(path)!=source['capture']['capture_sha256']
                or file_signature(path)!=signature):raise OrderingError('PCAP_CHANGED_DURING_VIEW')
        manifest=dict(format=FORMAT,status='COMPLETE_PCAP_RECORD_VIEW',source=source,policy=policy(),
            limits=asdict(limits),summary=summary,index=dict(filename='index.sqlite',bytes=index.stat().st_size,
            sha256=file_sha256(index)),labels='NOT_EVALUATED',flow_integration='NOT_ACTIVE',HDF5_export='BLOCKED')
        _new_json(output/'manifest.json',manifest)
        return manifest
    except Exception as error:
        _new_json(output/'FAILED.json',dict(status='FAILED_RETAINED',error=str(error),automatic_cleanup=False,
                                           labels_assigned=False,training=False))
        raise
    finally:
        if db is not None:db.close()


def validate_pcap_view(output):
    output=Path(output).absolute()
    if (output/'FAILED.json').exists():raise OrderingError('FAILED_PCAP_VIEW')
    manifest_path=regular_file(output/'manifest.json')
    if manifest_path.stat().st_size>1048576:raise OrderingError('PCAP_MANIFEST_BYTE_LIMIT')
    manifest=strict_json(manifest_path.read_bytes())
    keys={'format','status','source','policy','limits','summary','index','labels','flow_integration','HDF5_export'}
    if (type(manifest) is not dict or set(manifest)!=keys or manifest['format']!=FORMAT
            or manifest['status']!='COMPLETE_PCAP_RECORD_VIEW' or manifest['policy']!=policy()
            or manifest['labels']!='NOT_EVALUATED' or manifest['flow_integration']!='NOT_ACTIVE'
            or manifest['HDF5_export']!='BLOCKED'):
        raise OrderingError('PCAP_VIEW_MANIFEST_CONTRACT')
    source=validate_source(manifest['source']);limits=PcapViewLimits(**manifest['limits']).validate()
    if source['expected_records']>limits.max_records:raise OrderingLimitError('PCAP_VIEW_DECLARED_COUNT_LIMIT')
    identity=manifest['index']
    if (type(identity) is not dict or set(identity)!={'filename','bytes','sha256'}
            or identity['filename']!='index.sqlite' or type(identity['bytes']) is not int
            or not 1<=identity['bytes']<=limits.sqlite_bytes or type(identity['sha256']) is not str
            or not re.fullmatch('[a-f0-9]{64}',identity['sha256'])):
        raise OrderingError('PCAP_INDEX_MANIFEST')
    path=regular_file(output/'index.sqlite');signature=file_signature(path)
    if path.stat().st_size!=identity['bytes'] or file_sha256(path)!=identity['sha256']:
        raise OrderingError('PCAP_INDEX_HASH_OR_SIZE')
    with closing(_readonly(path,limits)) as db:summary=_summary(db,source)
    if summary!=manifest['summary'] or file_signature(path)!=signature:
        raise OrderingError('PCAP_INDEX_SEMANTIC_DIGEST_OR_MUTATION')
    return manifest


def iter_pcap_view(output):
    """Yield raw-record references; incompatible with synthetic flow/HDF5 contracts.
    Trả references record gốc; không giả dạng input synthetic cho flow/HDF5.
    """
    manifest=validate_pcap_view(output);path=regular_file(Path(output)/'index.sqlite')
    signature=file_signature(path)
    with closing(_readonly(path,PcapViewLimits(**manifest['limits']))) as db:
        yield from _records(db,manifest['source'])
    if file_signature(path)!=signature:raise OrderingError('PCAP_INDEX_CHANGED_DURING_READ')


class PcapRecordLookup:
    """Read at most one original record into memory; never decode records in isolation.
    Tra tối đa một record gốc trong RAM; không decode lẻ làm mất TCP state.
    """
    def __init__(self,output,workspace):
        self.manifest=validate_pcap_view(output);self.source=self.manifest['source']
        self.path=regular_file(Path(workspace)/self.source['path'])
        self.signature=file_signature(self.path)
        if (self.signature[2]!=self.source['size_bytes']
                or file_sha256(self.path)!=self.source['capture']['capture_sha256']
                or file_signature(self.path)!=self.signature):
            raise OrderingError('PCAP_LOOKUP_SOURCE_IDENTITY')
        self.index_path=regular_file(Path(output)/'index.sqlite');self.index_signature=file_signature(self.index_path)
        self.db=_readonly(self.index_path,PcapViewLimits(**self.manifest['limits']))
        self.stream=None;self.closed=False
        try:
            self.stream=self.path.open('rb')
            magic=self.stream.read(4)
            formats={b'\xd4\xc3\xb2\xa1':('<',1000),b'\xa1\xb2\xc3\xd4':('>',1000),
                     b'\x4d\x3c\xb2\xa1':('<',1),b'\xa1\xb2\x3c\x4d':('>',1)}
            self.endian,self.multiplier=formats[magic]
        except BaseException:
            if self.stream is not None:self.stream.close()
            self.db.close();raise

    def __enter__(self):return self
    def __exit__(self,*_):self.close()

    def __call__(self,reference):
        if self.closed or type(reference) is not PcapRecordReference:
            raise OrderingError('OPEN_PCAP_LOOKUP_AND_TYPED_REFERENCE_REQUIRED')
        if (any(type(getattr(reference,key)) is not int for key in
                ('processing_index','source_ordinal','timestamp_ns','record_offset','captured_length','original_length'))
                or type(reference.raw_record_sha256) is not str
                or not re.fullmatch('[a-f0-9]{64}',reference.raw_record_sha256)):
            raise OrderingError('PCAP_LOOKUP_REFERENCE_TYPE')
        row=self.db.execute('SELECT '+COLUMNS+',processing_index FROM packets WHERE source_ordinal=?',(reference.source_ordinal,)).fetchone()
        expected=(reference.timestamp_ns,reference.source_ordinal,reference.record_offset,reference.captured_length,
                  reference.original_length,bytes.fromhex(reference.raw_record_sha256),reference.processing_index)
        if row!=expected or reference.capture!=self.source['capture']:
            raise OrderingError('PCAP_LOOKUP_REFERENCE_BINDING')
        if file_signature(self.path)!=self.signature:raise OrderingError('PCAP_CHANGED_BEFORE_LOOKUP')
        if file_signature(self.index_path)!=self.index_signature:raise OrderingError('PCAP_INDEX_CHANGED_BEFORE_LOOKUP')
        self.stream.seek(reference.record_offset);raw=self.stream.read(16+reference.captured_length)
        if hashlib.sha256(raw).hexdigest()!=reference.raw_record_sha256:
            raise OrderingError('PCAP_RECORD_HASH_MISMATCH')
        seconds,fraction,caplen,wirelen=struct.unpack(self.endian+'IIII',raw[:16])
        if (seconds*1000000000+fraction*self.multiplier!=reference.timestamp_ns
                or caplen!=reference.captured_length or wirelen!=reference.original_length):
            raise OrderingError('PCAP_RECORD_HEADER_BINDING')
        return raw[16:]

    def close(self):
        if not self.closed:
            self.closed=True;self.stream.close();self.db.close()
            if (file_signature(self.path)!=self.signature
                    or file_sha256(self.path)!=self.source['capture']['capture_sha256']
                    or file_signature(self.path)!=self.signature):
                raise OrderingError('PCAP_CHANGED_DURING_LOOKUP')
            if (file_signature(self.index_path)!=self.index_signature
                    or file_sha256(self.index_path)!=self.manifest['index']['sha256']):
                raise OrderingError('PCAP_INDEX_CHANGED_DURING_LOOKUP')
