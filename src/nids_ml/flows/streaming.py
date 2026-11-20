"""Bounded reordering and one-window disk spooling for validated LUCID features.
Sắp thứ tự có giới hạn, lưu một window trên đĩa; tái dùng lõi LUCID đã kiểm.

Internal feature-stream core, not a PCAP decoder or supervised exporter. All
outputs remain UNKNOWN and provisional until the caller drains completion.
Lõi nhận FeaturePacket đã kiểm; chưa decode PCAP hoặc xuất dữ liệu có nhãn.
"""
from dataclasses import dataclass
import heapq
import ipaddress
import json
from pathlib import Path
import re
import shutil
import sqlite3
import struct
import time

from nids_ml.flows.extractor import FlowExtractor, FlowTensor, _FlowState
from nids_ml.flows.features import (
    CaptureContext, ExtractionError, FeaturePacket, ResourceLimitError,
    configuration, normalize_row,
)
from nids_ml.labeling.contracts import canonical_bytes

WINDOW_NS = 10_000_000_000
# Conservative reservation for one validated feature, heap tuple and shared fields.
# Dự trù bảo thủ cho FeaturePacket và heap; không phải hard OS RSS limit.
HEAP_PACKET_RESERVATION = 8192
VECTOR = struct.Struct('<10Q')


@dataclass(frozen=True)
class SourceRecord:
    """Keep every source position, including outside-scope and unsafe packets.
    Giữ mọi vị trí nguồn, kể cả packet ngoài scope hoặc cần quarantine.
    """
    ordinal: int
    timestamp_ns: int
    disposition: str
    feature: FeaturePacket | None = None


@dataclass(frozen=True)
class StreamingLimits:
    max_input_packets: int = 30_000_000
    max_heap_packets: int = 4096
    sqlite_bytes: int = 256 * 1024**2
    min_free_bytes: int = 256 * 1024**2
    max_window_packets: int = 2_000_000
    timeout_seconds: int = 7200

    def validate(self):
        bounds = dict(max_input_packets=(1, 30_000_000), max_heap_packets=(1, 8192),
                      sqlite_bytes=(65536, 512*1024**2), min_free_bytes=(1048576, 4*1024**3),
                      max_window_packets=(1, 5_000_000), timeout_seconds=(1, 14400))
        for field, (low, high) in bounds.items():
            value = getattr(self, field)
            if type(value) is not int or not low <= value <= high:
                raise ExtractionError('STREAM_LIMIT:' + field)
        if self.sqlite_bytes % 4096:
            raise ExtractionError('STREAM_SQLITE_PAGE_BUDGET')
        return self


class WindowSpool:
    """One capture and one run per instance; no resume or numeric label API.
    Mỗi instance xử lý một capture/run; không resume hoặc tự gán nhãn số.
    """

    def __init__(self, context, directory, *, max_lateness_ns, input_kind='decoded_pcap', limits=None,
                 resource_guard=None):
        if not isinstance(context, CaptureContext):
            raise ExtractionError('STREAM_CONTEXT_REQUIRED')
        self.context = CaptureContext.from_dict(context.to_dict())
        self.config = configuration(input_kind=input_kind)
        if type(max_lateness_ns) is not int or not 0 <= max_lateness_ns <= 60*10**9:
            raise ExtractionError('STREAM_LATENESS_BUDGET')
        self.lateness = max_lateness_ns
        self.limits = (StreamingLimits() if limits is None else limits).validate()
        self.directory = Path(directory)
        if resource_guard is not None and not callable(resource_guard):
            raise ExtractionError('STREAM_RESOURCE_GUARD_REQUIRED')
        self.resource_guard = resource_guard
        self.status = 'NEW'
        self.stats = dict(input_packets=0, tensors=0, windows=0, peak_heap_packets=0,
                          peak_window_packets=0, peak_sqlite_bytes=0, max_observed_lateness_ns=0,
                          overflow_packets=0, unknown_protocol_token_packets=0,
                          feature_packets=0, outside_packets=0, quarantined_packets=0,
                          quarantined_windows=0, quarantined_flow_windows=0)
        self.full_source = False

    def _check_packet(self, packet):
        # Caller must use extract_features/map_record; the dataclass is internal.
        # Caller dùng extract_features/map_record; kiểm thêm identity và bounds tại đây.
        if type(packet) is not FeaturePacket or packet.context != self.context:
            raise ExtractionError('STREAM_FEATURE_CONTEXT')
        if (type(packet.ordinal) is not int or not 1 <= packet.ordinal < 2**63
                or type(packet.timestamp_ns) is not int or not 0 <= packet.timestamp_ns < 2**63):
            raise ExtractionError('STREAM_POSITION')
        key = packet.canonical_tuple
        if (type(key) is not tuple or len(key) != 5 or len(canonical_bytes(key)) > 256
                or packet.forward_tuple not in (key, (key[2], key[3], key[0], key[1], key[4]))):
            raise ExtractionError('STREAM_FLOW_IDENTITY')
        if (any(type(key[i]) is not str for i in (0, 2))
                or any(type(key[i]) is not int or not 0 <= key[i] <= 65535 for i in (1, 3))
                or type(key[4]) is not int or key[4] not in (1, 6, 17)
                or any(ipaddress.ip_address(key[i]).version != 4 for i in (0, 2))
                or (key[0], key[1]) > (key[2], key[3])):
            raise ExtractionError('STREAM_CANONICAL_IPV4_TUPLE')
        if (type(packet.packet_sha256) is not str
                or re.fullmatch('[a-f0-9]{64}', packet.packet_sha256) is None
                or type(packet.features) is not tuple or len(packet.features) != 11
                or packet.features[0] != packet.timestamp_ns/1e9
                or any(type(v) is not int for v in packet.features[1:])
                or type(packet.unknown_protocol_tokens) is not tuple
                or len(packet.unknown_protocol_tokens) > 64
                or any(type(t) is not str or len(t) > 256 for t in packet.unknown_protocol_tokens)
                or sum(len(t) for t in packet.unknown_protocol_tokens) > 256
                or (packet.icmp_identifier is not None and
                    (type(packet.icmp_identifier) is not int or not 0 <= packet.icmp_identifier <= 65535))):
            raise ExtractionError('STREAM_FEATURE_RECORD')
        normalize_row(packet.features, 0)

    def _guard(self, db):
        if self.resource_guard is not None:
            self.resource_guard()
        if time.monotonic() - self.started > self.limits.timeout_seconds:
            raise ResourceLimitError('STREAM_TIMEOUT')
        if shutil.disk_usage(self.directory).free < self.limits.min_free_bytes:
            raise ResourceLimitError('STREAM_DISK_RESERVE')
        size = db.execute('PRAGMA page_count').fetchone()[0] * 4096
        self.stats['peak_sqlite_bytes'] = max(self.stats['peak_sqlite_bytes'], size)
        if size > self.limits.sqlite_bytes:
            raise ResourceLimitError('STREAM_SQLITE_BUDGET')

    def _flush(self, db, anchor, window, complete, poison=False):
        db.commit()
        self._guard(db)
        engine = FlowExtractor(self.context, self.config)
        if poison:
            self.stats['quarantined_windows'] += 1
            self.stats['quarantined_flow_windows'] += db.execute('SELECT count(*) FROM flows').fetchone()[0]
        flows = () if poison else db.execute('SELECT flow FROM flows ORDER BY first_ts,first_ord')
        for (flow,) in flows:
            key = tuple(json.loads(flow))
            state = None
            for ts, ordinal, direction, values, sha, icmp in db.execute(
                    'SELECT ts,ord,direction,f,sha,icmp FROM packets WHERE flow=? ORDER BY ts,ord', (flow,)):
                forward = key if direction == 0 else (key[2], key[3], key[0], key[1], key[4])
                packet = FeaturePacket(self.context, ordinal, ts, forward, key,
                                       (ts/1e9, *VECTOR.unpack(values)), sha.hex(), (), icmp)
                if state is None:
                    state = _FlowState(packet, 20)
                state.add(packet, 20)
                if state.count % 256 == 0:
                    self._guard(db)
            engine._flows[key] = state
            tensor = next(engine._flush(anchor, window, complete))
            metadata = tensor.metadata
            metadata['processing_view'] = dict(
                contract='nids-window-spool/1.0.0', order='STABLE_TIMESTAMP_THEN_ORIGINAL_ORDINAL',
                source_ordinals='PRESERVED_NONMONOTONIC', max_lateness_ns=self.lateness,
                scope='ALL_DECODED_SOURCE_RECORDS' if self.full_source else 'ELIGIBLE_FEATURE_STREAM_ONLY_NOT_FULL_CAPTURE',
                quarantine_policy='EXCLUDE_ENTIRE_GLOBAL_WINDOW' if self.full_source else 'NOT_EVALUATED',
                labels='UNKNOWN_ONLY', provisional_until_stream_completion=True)
            self.stats['tensors'] += 1
            self.stats['overflow_packets'] += max(0, state.count-20)
            if self.stats['tensors'] % 256 == 0:
                self._guard(db)
            yield FlowTensor(tensor.shape, tensor.tensor_bytes, canonical_bytes(metadata))
        # Reuse SQLite pages; never delete caller data or an earlier run's cache.
        # Tái dùng pages của spool riêng; không xóa dữ liệu hoặc cache của run khác.
        db.execute('DELETE FROM packets')
        db.execute('DELETE FROM flows')
        db.commit()

    def run(self, packets, *, input_complete):
        return self._run(packets, input_complete=input_complete, full_source=False)

    def run_records(self, records, *, input_complete):
        """Use all decoded ordinals for anchors and conservative window quarantine.
        Dùng đủ ordinal để đặt anchor và loại toàn window có packet không an toàn.
        """
        return self._run(records, input_complete=input_complete, full_source=True)

    def _run(self, packets, *, input_complete, full_source):
        if self.status != 'NEW' or type(input_complete) is not bool:
            raise ExtractionError('STREAM_SINGLE_USE_AND_COMPLETENESS_REQUIRED')
        self.status = 'RUNNING'
        self.full_source = full_source
        self.started = time.monotonic()
        db = None
        owned = False
        heap = []
        high = previous_ordinal = None
        anchor = None
        window = window_packets = 0
        poison = False

        def store(event):
            nonlocal anchor, window, window_packets, poison
            if anchor is not None and event.timestamp_ns > anchor + WINDOW_NS:
                yield from self._flush(db, anchor, window, True, poison)
                window += 1
                anchor = None
                window_packets = 0
                poison = False
            if anchor is None:
                anchor = event.timestamp_ns
                self.stats['windows'] += 1
            if window_packets >= self.limits.max_window_packets:
                raise ResourceLimitError('STREAM_WINDOW_PACKET_BUDGET')
            poison |= event.disposition == 'QUARANTINE'
            packet = event.feature
            if packet is not None:
                flow = canonical_bytes(packet.canonical_tuple).decode('ascii')
                db.execute('INSERT OR IGNORE INTO flows VALUES (?,?,?)',
                           (flow, packet.timestamp_ns, packet.ordinal))
                db.execute('INSERT INTO packets VALUES (?,?,?,?,?,?,?)', (
                    packet.ordinal, packet.timestamp_ns, flow, int(packet.forward_tuple != packet.canonical_tuple),
                    VECTOR.pack(*packet.features[1:]), bytes.fromhex(packet.packet_sha256), packet.icmp_identifier))
            window_packets += 1
            self.stats['peak_window_packets'] = max(self.stats['peak_window_packets'], window_packets)
            if window_packets % 256 == 0:
                self._guard(db)

        try:
            if shutil.disk_usage(self.directory.parent).free < self.limits.min_free_bytes + self.limits.sqlite_bytes:
                raise ResourceLimitError('STREAM_START_DISK_RESERVE')
            self.directory.mkdir(exist_ok=False)
            owned = True
            db = sqlite3.connect(self.directory/'window.sqlite')
            db.execute('PRAGMA page_size=4096')
            db.execute('PRAGMA max_page_count=' + str(self.limits.sqlite_bytes//4096))
            db.execute('PRAGMA cache_size=-1024')
            db.execute('PRAGMA mmap_size=0')
            db.execute('PRAGMA temp_store=FILE')
            # Disposable scratch only: a crash is FAILED, never recover/accept it.
            # Chỉ là scratch: crash phải thất bại, không phục hồi rồi nhận là COMPLETE.
            db.execute('PRAGMA journal_mode=OFF')
            db.execute('CREATE TABLE flows(flow TEXT PRIMARY KEY,first_ts INTEGER,first_ord INTEGER)')
            db.execute('CREATE INDEX flow_order ON flows(first_ts,first_ord)')
            db.execute('CREATE TABLE packets(ord INTEGER PRIMARY KEY,ts INTEGER,flow TEXT,direction INTEGER,f BLOB,sha BLOB,icmp INTEGER)')
            db.execute('CREATE INDEX packet_order ON packets(flow,ts,ord)')
            for item in packets:
                if full_source:
                    event = item
                    if (type(event) is not SourceRecord or type(event.ordinal) is not int
                            or not 1 <= event.ordinal < 2**63 or type(event.timestamp_ns) is not int
                            or not 0 <= event.timestamp_ns < 2**63
                            or event.disposition not in ('FEATURE', 'OUTSIDE', 'QUARANTINE')
                            or (event.feature is not None) != (event.disposition == 'FEATURE')):
                        raise ExtractionError('STREAM_SOURCE_RECORD')
                    if event.ordinal != (1 if previous_ordinal is None else previous_ordinal+1):
                        raise ExtractionError('STREAM_SOURCE_ORDINAL_GAP')
                else:
                    event = SourceRecord(item.ordinal, item.timestamp_ns, 'FEATURE', item) if type(item) is FeaturePacket else None
                    if event is None:
                        raise ExtractionError('STREAM_FEATURE_CONTEXT')
                packet = event.feature
                if packet is not None:
                    self._check_packet(packet)
                    if (event.ordinal, event.timestamp_ns) != (packet.ordinal, packet.timestamp_ns):
                        raise ExtractionError('STREAM_SOURCE_FEATURE_POSITION')
                if previous_ordinal is not None and event.ordinal <= previous_ordinal:
                    raise ExtractionError('STREAM_SOURCE_ORDINAL_REGRESSION')
                if self.stats['input_packets'] >= self.limits.max_input_packets:
                    raise ResourceLimitError('STREAM_INPUT_PACKET_BUDGET')
                high = event.timestamp_ns if high is None else max(high, event.timestamp_ns)
                late = high-event.timestamp_ns
                self.stats['max_observed_lateness_ns'] = max(self.stats['max_observed_lateness_ns'], late)
                if late > self.lateness:
                    raise ExtractionError('STREAM_LATENESS_EXCEEDED')
                if len(heap) >= self.limits.max_heap_packets:
                    raise ResourceLimitError('STREAM_HEAP_PACKET_BUDGET')
                heapq.heappush(heap, (event.timestamp_ns, event.ordinal, event))
                previous_ordinal = event.ordinal
                self.stats['input_packets'] += 1
                self.stats['feature_packets'] += packet is not None
                self.stats['outside_packets'] += event.disposition == 'OUTSIDE'
                self.stats['quarantined_packets'] += event.disposition == 'QUARANTINE'
                self.stats['unknown_protocol_token_packets'] += bool(packet and packet.unknown_protocol_tokens)
                self.stats['peak_heap_packets'] = max(self.stats['peak_heap_packets'], len(heap))
                # Strict comparison retains equal timestamps for stable future ties.
                # Giữ timestamp bằng watermark để thứ tự ordinal của ties ổn định.
                while heap and heap[0][0] < high-self.lateness:
                    yield from store(heapq.heappop(heap)[2])
                if self.stats['input_packets'] % 256 == 0:
                    self._guard(db)
            while heap:
                yield from store(heapq.heappop(heap)[2])
            if anchor is not None:
                yield from self._flush(db, anchor, window, input_complete, poison)
            self._guard(db)
            self.status = 'COMPLETE' if input_complete else 'PARTIAL'
        except BaseException as error:
            self.status = 'ABORTED' if isinstance(error, GeneratorExit) else 'FAILED'
            if owned:
                (self.directory/'FAILED.json').write_bytes(canonical_bytes(dict(
                    status=self.status, error=str(error) or type(error).__name__, stats=self.stats,
                    automatic_cleanup=False, training_ready=False)))
            raise
        finally:
            if db is not None:
                db.close()
            heap.clear()
