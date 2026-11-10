"""Scan classic-PCAP record headers, checksums and timestamp order.
Quét record headers PCAP classic, checksum và thứ tự timestamp.
"""
import hashlib
from pathlib import Path
import struct
import time

from nids_ml.alignment.candidates import AlignmentError
from nids_ml.pcap.decoder import file_signature, regular_file


def scan_record_headers(path, *, expected_sha256, expected_size,
                        max_records=3500000, timeout_seconds=300, progress=None, record_sink=None):
    """Giữ source ordinal/timestamp; đếm regressions, không sort hoặc sửa raw."""
    if max_records != 3500000 or timeout_seconds != 300:
        raise AlignmentError('UNAPPROVED_METADATA_SCAN_BUDGET')
    if record_sink is not None and not callable(record_sink):
        raise AlignmentError('CALLABLE_RECORD_SINK_REQUIRED')
    path = regular_file(Path(path))
    before = file_signature(path)
    if before[2] != expected_size or expected_size > 1024**3:
        raise AlignmentError('PCAP_SIZE_MISMATCH_OR_BUDGET')
    digest = hashlib.sha256()
    started = time.monotonic()
    count = regressions = max_regression = truncated = 0
    previous = first = minimum = maximum = None
    examples = []
    formats = {b'\xd4\xc3\xb2\xa1': ('<', 1000, 1000000),
               b'\xa1\xb2\xc3\xd4': ('>', 1000, 1000000),
               b'\x4d\x3c\xb2\xa1': ('<', 1, 1000000000),
               b'\xa1\xb2\x3c\x4d': ('>', 1, 1000000000)}
    with path.open('rb', buffering=1048576) as stream:
        header = stream.read(24)
        if len(header) != 24 or header[:4] not in formats:
            raise AlignmentError('CLASSIC_PCAP_HEADER_REQUIRED')
        endian, multiplier, denominator = formats[header[:4]]
        major, minor, _, _, snaplen, linktype = struct.unpack(endian + 'HHiIII', header[4:])
        if (major, minor) != (2, 4) or not 1 <= snaplen <= 1048576:
            raise AlignmentError('UNSUPPORTED_PCAP_VERSION_OR_SNAPLEN')
        digest.update(header)
        while True:
            offset=stream.tell()
            raw = stream.read(16)
            if not raw:
                break
            if len(raw) != 16:
                raise AlignmentError('TRUNCATED_RECORD_HEADER')
            seconds, fraction, caplen, wirelen = struct.unpack(endian + 'IIII', raw)
            if fraction >= denominator or caplen > snaplen or caplen > wirelen:
                raise AlignmentError('INVALID_PCAP_RECORD_LENGTH_OR_CLOCK')
            payload = stream.read(caplen)
            if len(payload) != caplen:
                raise AlignmentError('TRUNCATED_RECORD_PAYLOAD')
            digest.update(raw)
            digest.update(payload)
            current = seconds * 1000000000 + fraction * multiplier
            count += 1
            if count > max_records:
                raise AlignmentError('PCAP_RECORD_CAP')
            if record_sink is not None:
                record_sink(dict(source_ordinal=count,timestamp_ns=current,record_offset=offset,
                    captured_length=caplen,original_length=wirelen,
                    raw_record_sha256=hashlib.sha256(raw+payload).hexdigest()))
            if first is None:
                first = minimum = maximum = current
            if previous is not None and current < previous:
                regressions += 1
                max_regression = max(max_regression, previous - current)
                if len(examples) < 8:
                    examples.append(dict(source_ordinal=count, delta_ns=current - previous))
            previous = current
            minimum, maximum = min(minimum, current), max(maximum, current)
            truncated += caplen < wirelen
            if count % 10000 == 0 and time.monotonic() - started > timeout_seconds:
                raise AlignmentError('PCAP_SCAN_TIMEOUT')
            if progress and count % 250000 == 0:
                progress('pcap_headers', count)
    if digest.hexdigest() != expected_sha256 or file_signature(path) != before:
        raise AlignmentError('PCAP_IDENTITY_CHANGED_OR_HASH_MISMATCH')
    if not count:
        raise AlignmentError('EMPTY_PCAP')
    return dict(scope='FULL_RECORD_HEADERS_AND_SHA256_NO_PAYLOAD_DECODING', full_scan=True,
                records=count, sha256=digest.hexdigest(), timestamp_regressions=regressions,
                max_regression_ns=max_regression, regression_examples=examples,
                first_timestamp_ns=first, last_timestamp_ns=previous,
                min_timestamp_ns=minimum, max_timestamp_ns=maximum,
                observed_span_ns=maximum-minimum, linktype=linktype,
                capture_truncated_records=truncated, sorting_performed=False,
                timestamp_edits_performed=False, elapsed_seconds=time.monotonic()-started)
