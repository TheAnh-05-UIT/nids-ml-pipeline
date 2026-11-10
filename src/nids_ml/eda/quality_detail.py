"""Count label spelling, row-order and port-value quality issues.
Đếm vấn đề spelling nhãn, thứ tự dòng và giá trị port.

Không huấn luyện rule/model; chỉ đếm contingency tables để xem xét leakage.
Không giữ bản ghi hoặc tự sửa giá trị, kể cả khi phát hiện giá trị đáng ngờ.
"""
from collections import Counter, defaultdict
import csv
import ipaddress
import math
from pathlib import Path

from nids_ml.eda.csv_eda import BoundedLines
from nids_ml.data.inventory import fingerprint

FORMAT_COLUMNS = ('dns.qry.name.len', 'mqtt.conack.flags', 'mqtt.protoname', 'mqtt.topic')
PORT_COLUMNS = ('tcp.srcport', 'tcp.dstport')


def port_kind(raw):
    try:
        value = float(raw.strip())
    except ValueError:
        return 'non_numeric_text'
    if not math.isfinite(value):
        return 'nonfinite'
    if not value.is_integer():
        return 'noninteger'
    return 'in_range_integer' if 0 <= value <= 65535 else 'out_of_range'


def audit_quality(path):
    """Counters bị chặn ở 1.000 label tuples; mỗi record tối đa 1 Mi ký tự."""
    path = Path(path)
    identity = fingerprint(path)
    counters = {name: defaultdict(Counter) for name in FORMAT_COLUMNS}
    ports = {name: defaultdict(Counter) for name in PORT_COLUMNS}
    time_kinds, joint, order = defaultdict(Counter), Counter(), {}
    previous, current_run, max_run, total_runs, rows, mismatch = None, 0, 0, 0, 0, 0
    with path.open(encoding='utf-8-sig', newline='') as stream:
        lines = BoundedLines(stream, 1048576)
        reader = csv.reader(lines, strict=True)
        try:
            header = lines.record(reader)
        except StopIteration:
            raise ValueError('Empty CSV') from None
        required = (*FORMAT_COLUMNS, *PORT_COLUMNS, 'Attack_label', 'Attack_type', 'frame.time')
        if len(set(header)) != len(header) or any(name not in header for name in required):
            raise ValueError('Required distinct audit columns missing')
        ids = {name: header.index(name) for name in required}
        while True:
            try:
                row = lines.record(reader)
            except StopIteration:
                break
            if len(row) != len(header):
                raise ValueError('Record width mismatch; no row skipped')
            rows += 1
            binary, label = row[ids['Attack_label']], row[ids['Attack_type']]
            if len(label) > 256 or len(binary) > 256:
                raise ValueError('Label size limit exceeded')
            joint[binary, label] += 1
            if len(joint) > 1000:
                raise ValueError('Label tuple budget exceeded')
            mismatch += int(binary != ('0' if label == 'Normal' else '1'))
            for name in FORMAT_COLUMNS:
                raw = row[ids[name]]
                # Chỉ ba buckets, không giữ vocabulary/raw payload không giới hạn.
                counters[name][raw if raw in ('0', '0.0') else 'OTHER'][label] += 1
            for name in PORT_COLUMNS:
                ports[name][port_kind(row[ids[name]])][label] += 1
            value = row[ids['frame.time']].strip()
            try:
                is_ip = len(value) <= 64 and bool(ipaddress.ip_address(value))
            except ValueError:
                is_ip = False
            try:
                numeric = math.isfinite(float(value))
            except ValueError:
                numeric = False
            time_kinds['ip_literal' if is_ip else 'finite_numeric' if numeric else 'other_text'][label] += 1
            group = order.setdefault(label, {'count': 0, 'first_data_row': rows,
                                            'last_data_row': rows, 'runs': 0})
            group['count'] += 1
            group['last_data_row'] = rows
            if label != previous:
                total_runs += 1
                group['runs'] += 1
                current_run = 0
            current_run += 1
            max_run = max(max_run, current_run)
            previous = label
    if fingerprint(path) != identity:
        raise ValueError('Input changed during audit')

    def table(groups):
        return {kind: {'count': sum(counts.values()), 'by_attack_type': dict(sorted(counts.items()))}
                for kind, counts in sorted(groups.items())}

    return {'schema_version': '1.0.0', 'input': identity, 'row_count': rows,
            'scope': 'full_scan_exact_text_contingency_not_model_performance',
            'formatting': {name: table(groups) for name, groups in counters.items()},
            'label_pairs': [{'Attack_label': binary, 'Attack_type': label, 'count': count}
                            for (binary, label), count in sorted(joint.items())],
            'binary_vs_Normal_rule_mismatch_count': mismatch,
            'ports': {name: table(groups) for name, groups in ports.items()},
            'frame_time_lexical_kinds': table(time_kinds),
            'row_order': {'scope': 'CSV order, not verified capture chronology',
                          'total_label_runs': total_runs, 'longest_label_run': max_run,
                          'by_attack_type': dict(sorted(order.items()))},
            'limits': {'max_record_chars': 1048576, 'max_label_pairs': 1000,
                       'zero_buckets': 'Exact raw 0 / 0.0 / OTHER; no normalization',
                       'time_kinds': 'Lexical anomaly review only; not date parsing or schema repair'},
            'preprocessing_split_training': 'NOT RUN'}
