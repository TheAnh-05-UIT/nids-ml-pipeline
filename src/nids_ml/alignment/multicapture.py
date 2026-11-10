"""Audit CSV quality and prefix alignment across multiple captures.
Audit chất lượng CSV và prefix mapping của nhiều capture.
"""
from collections import Counter
import csv
from dataclasses import replace
import hashlib
import ipaddress
import time

from nids_ml.alignment.candidates import (AlignmentError, candidate_join,
    clock_hypothesis, csv_clock, make_record, numeric_token, time_witness_join)
from nids_ml.eda.csv_eda import BoundedLines
from nids_ml.pcap.decoder import file_sha256, file_signature, regular_file


def scan_csv_quality(path, *, expected_sha256, expected_size, progress=None):
    """Thống kê toàn CSV theo header nguyên bản; không dịch cột khi thấy bất thường."""
    path = regular_file(path)
    before = file_signature(path)
    if before[2] != expected_size or expected_size > 1024**3:
        raise AlignmentError('CSV_SIZE_MISMATCH_OR_BUDGET')
    if file_sha256(path) != expected_sha256 or file_signature(path) != before:
        raise AlignmentError('CSV_IDENTITY_MISMATCH')
    started = time.monotonic()
    widths, labels, years = Counter(), Counter(), Counter()
    rows = valid_clock = invalid_clock = regressions = 0
    previous = None
    prefix, examples = [], []
    prefix_chars = 0
    old_limit = csv.field_size_limit(1048576)
    try:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            lines = BoundedLines(stream, 1048576)
            reader = csv.reader(lines, strict=True)
            header = lines.record(reader)
            if not header or len(header) > 256 or len(set(header)) != len(header):
                raise AlignmentError('INVALID_CSV_HEADER')
            if not {'frame.time', 'Attack_label', 'Attack_type'} <= set(header):
                raise AlignmentError('MISSING_AUDIT_COLUMNS')
            clock_at, label_at, type_at = (header.index(k) for k in ('frame.time', 'Attack_label', 'Attack_type'))
            while True:
                try:
                    values = lines.record(reader)
                except StopIteration:
                    break
                rows += 1
                if rows > 3500000:
                    raise AlignmentError('CSV_RECORD_CAP')
                if len(values) > 256:
                    raise AlignmentError('CSV_WIDTH_BUDGET')
                widths[len(values)] += 1
                if rows <= 1024:
                    prefix_chars += lines.used
                    if prefix_chars > 4194304:
                        raise AlignmentError('CSV_PREFIX_CHARACTER_BUDGET')
                    # Không zip/cắt thừa cột để biến record lỗi thành hợp lệ.
                    prefix.append(dict(zip(header, values)) if len(values) == len(header) else {})
                if len(values) == len(header):
                    a, b = values[label_at], values[type_at]
                    if max(len(a), len(b)) > 128 or (a, b) not in labels and len(labels) >= 64:
                        raise AlignmentError('CSV_LABEL_TOKEN_OR_CATEGORY_BUDGET')
                    labels[(a, b)] += 1
                    year, clock = csv_clock(values[clock_at])
                    if clock is None:
                        invalid_clock += 1
                        if len(examples) < 8:
                            examples.append(dict(source_csv_ordinal=rows,
                                                 frame_time_token=values[clock_at][:100]))
                    else:
                        valid_clock += 1
                        years[year] += 1
                        if previous is not None and clock < previous:
                            regressions += 1
                        previous = clock
                if rows % 10000 == 0 and time.monotonic() - started > 300:
                    raise AlignmentError('CSV_SCAN_TIMEOUT')
                if progress and rows % 250000 == 0:
                    progress('csv_quality', rows)
    finally:
        csv.field_size_limit(old_limit)
    if file_signature(path) != before:
        raise AlignmentError('CSV_CHANGED_DURING_SCAN')
    return dict(scope='FULL_CSV_BY_UNMODIFIED_HEADER', full_scan=True, rows=rows,
                columns=len(header), header=header,
                header_sha256=hashlib.sha256('\n'.join(header).encode()).hexdigest(),
                width_counts={str(k): v for k, v in sorted(widths.items())},
                wrong_width_rows=rows-widths[len(header)],
                valid_frame_time_rows=valid_clock, invalid_frame_time_rows=invalid_clock,
                clock_year_counts={str(k): v for k, v in sorted(years.items())},
                time_of_day_regressions=regressions,
                clock_note='CSV lacks full calendar date/timezone; TOD regressions are not inferred attack boundaries',
                invalid_clock_examples=examples,
                dataset_provided_label_counts=[dict(Attack_label=a, Attack_type=b, rows=n)
                                              for (a, b), n in sorted(labels.items())],
                prefix_rows=len(prefix), prefix_chars=prefix_chars,
                column_repair_performed=False, labels_verified=False,
                elapsed_seconds=time.monotonic()-started), prefix


def diagnostic_record(row, *, kind, source_sha256, ordinal):
    """Khóa TCP cũ và khóa ICMP yếu hơn; không suy protocol từ nhãn/filename."""
    if kind not in ('pcap', 'csv') or type(row) is not dict:
        raise AlignmentError('INVALID_DIAGNOSTIC_INPUT')
    # Dùng validation identity/ordinal đã có; không lấy label fields làm key.
    base = make_record(row, kind=kind, source_sha256=source_sha256, ordinal=ordinal)
    if kind == 'pcap':
        try:
            protocol = numeric_token(row.get('ip.proto'))
            fragment = numeric_token(row.get('ip.frag_offset', '0'))
            flag = row.get('ip.flags.mf', '0')
            # Wireshark fields biểu diễn boolean bằng True/False, khác numeric flags.
            more = int(flag == 'True') if flag in ('True', 'False') else numeric_token(flag)
        except AlignmentError:
            return replace(base, key=None, reason='MISSING_MULTIPLE_PROTOCOL_OR_FRAGMENT_FIELDS')
        if fragment or more:
            return replace(base, key=None, reason='FRAGMENT_NOT_SUPPORTED_FOR_MAPPING')
        if protocol == 6:
            return replace(base, key=('TCPv4', *base.key) if base.key else None)
        if protocol != 1:
            return replace(base, key=None, reason='UDP_OR_OTHER_PROTOCOL_NO_APPROVED_SHARED_KEY')
    else:
        # Cổng hoặc checksum/sequence chỉ kích hoạt profile kiểm tra khả thi,
        # không chứng minh protocol hay độ tin cậy của dòng CSV.
        try:
            tcp_present = bool(numeric_token(row.get('tcp.srcport'), 16)
                               or numeric_token(row.get('tcp.dstport'), 16))
            icmp_present = bool(numeric_token(row.get('icmp.checksum'), 16)
                                or numeric_token(row.get('icmp.seq_le'), 16))
        except AlignmentError:
            return replace(base, key=None, reason='CSV_PROTOCOL_FIELDS_INVALID')
        if tcp_present and icmp_present:
            return replace(base, key=None, reason='CSV_MULTIPLE_PROTOCOL_CANDIDATES')
        if tcp_present:
            return replace(base, key=('TCPv4', *base.key) if base.key else None)
        if not icmp_present:
            return replace(base, key=None, reason='CSV_PROTOCOL_NOT_ESTABLISHED')
    try:
        source_key = 'ip.src_host' if kind == 'csv' else 'ip.src'
        destination_key = 'ip.dst_host' if kind == 'csv' else 'ip.dst'
        addresses = [ipaddress.ip_address(row.get(k, '').strip()) for k in (source_key, destination_key)]
        if any(a.version != 4 for a in addresses):
            raise AlignmentError('ONLY_IPV4')
        checksum = numeric_token(row.get('icmp.checksum'), 16)
        sequence = numeric_token(row.get('icmp.seq_le'), 16)
    except (ValueError, TypeError, AttributeError):
        return replace(base, key=None, reason='ICMP_SHARED_FIELDS_UNAVAILABLE')
    key = ('ICMPv4_DIAGNOSTIC', *(str(a) for a in addresses), checksum, sequence)
    return replace(base, key=key, reason=None)


def audit_prefix(scalars, csv_rows, *, pcap_sha256, csv_sha256):
    """Mapping giới hạn prefix; không nhận là uniqueness toàn capture hay ground truth."""
    left = [diagnostic_record(r, kind='pcap', source_sha256=pcap_sha256, ordinal=i)
            for i, r in enumerate(scalars, 1)]
    right = [diagnostic_record(r, kind='csv', source_sha256=csv_sha256, ordinal=i)
             for i, r in enumerate(csv_rows, 1)]
    joined = candidate_join(left, right)
    joined['key_version'] = 'multi-protocol-prefix-diagnostic/1.0.0'
    joined['key_fields'] = dict(TCPv4=['IPv4 src/dst', 'TCP src/dst ports', 'len', 'flags', 'ack_raw', 'checksum'],
                                ICMPv4_DIAGNOSTIC=['IPv4 src/dst', 'icmp.checksum', 'icmp.seq_le'])
    clock = clock_hypothesis(left, right, joined)
    witnessed = time_witness_join(left, right, clock)
    witnessed['key_version'] = joined['key_version']
    return dict(content=joined, clock_diagnostic=clock, time_witness=witnessed,
                pcap_reasons=dict(Counter(r.reason or 'KEY_AVAILABLE' for r in left)),
                csv_reasons=dict(Counter(r.reason or 'KEY_AVAILABLE' for r in right)),
                pcap_protocol_counts=dict(Counter(r.get('ip.proto', '') for r in scalars)),
                ICMP_limitations='CSV lacks type/code/identifier: diagnostic candidate only, not full packet identity',
                UDP_status='BLOCKED_NO_DIRECTIONAL_PORT_PAIR_IN_CSV_SCHEMA',
                global_PCAP_uniqueness='NOT_VERIFIED', global_CSV_uniqueness='NOT_VERIFIED',
                label_assignment_performed=False, supervised_eligible=False)
