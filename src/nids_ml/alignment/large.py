"""Define bounded large-capture mapping and dataset-provided binary NIDS annotations.
Mapping capture lớn bằng khóa gọn; nhãn CSV duy nhất có lineage, UNKNOWN không thành Normal.
"""
from dataclasses import asdict, dataclass, replace
import ipaddress
import re
import struct

from .candidates import AlignmentError
from .full import FullAuditLimits
from .multicapture import diagnostic_record

POLICY = 'DATASET_PROVIDED_MAPPED_NORMAL_ATTACK/1.0.0'
MANIFEST = 'nids-multi-attack-mapping/1.0.0'
KEY_PROFILE = 'compact-shared-header-time/1.0.0'
ATTACK_TYPES = ('Backdoor', 'DDoS_HTTP', 'DDoS_ICMP', 'DDoS_TCP', 'DDoS_UDP',
                'OS_Fingerprinting', 'Fingerprinting', 'MITM', 'Password', 'Port_Scanning',
                'Ransomware', 'SQL_injection', 'Uploading', 'Vulnerability_scanner', 'XSS')


def require(condition, reason):
    if not condition:
        raise AlignmentError(reason)


@dataclass(frozen=True)
class LargeAuditLimits(FullAuditLimits):
    """One explicit bounded profile, separate from accepted 250k legacy limits.
    Profile mới có budget cố định; giữ nguyên giới hạn của CLI/pilot cũ.
    """
    record_cap: int = 3500000
    sqlite_mib: int = 1536
    scratch_mib: int = 1536
    timeout_seconds: int = 7200
    min_free_mib: int = 1024
    stdout_mib: int = 1024

    def validate(self):
        expected = LargeAuditLimits()
        for name in self.__dataclass_fields__:
            require(type(getattr(self, name)) is int and getattr(self, name) == getattr(expected, name),
                    'UNAPPROVED_LARGE_AUDIT_LIMIT:' + name)
        return self


def validate_large_config(config):
    required = {'schema_version', 'kind', 'case_id', 'key_profile', 'protocol', 'label_policy',
                'limits', 'pcap', 'csv', 'toolchain', 'expected_pcap_records', 'expected_csv_rows'}
    require(type(config) is dict and set(config) == required, 'LARGE_CONFIG_KEYS')
    require(config['schema_version'] == '1.2.0' and config['kind'] == 'full_pair_mapping_audit',
            'LARGE_CONFIG_VERSION')
    require(config['key_profile'] == KEY_PROFILE and config['label_policy'] == POLICY, 'LARGE_MAPPING_POLICY')
    require(config['protocol'] in ('TCP', 'ICMP'), 'PROTOCOL_NOT_APPROVED_UDP_ARP_BLOCKED')
    require(isinstance(config['case_id'], str) and
            re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', config['case_id']), 'LARGE_CASE_ID')
    limits = LargeAuditLimits().validate()
    require(config['limits'] == asdict(limits) and all(type(v) is int for v in config['limits'].values()),
            'UNAPPROVED_LARGE_AUDIT_LIMITS')
    for name in ('expected_pcap_records', 'expected_csv_rows'):
        require(type(config[name]) is int and 1 <= config[name] < limits.record_cap,
                'EXACT_LARGE_SOURCE_COUNT_REQUIRED:' + name)
    for kind in ('pcap', 'csv'):
        identity = config[kind]
        require(type(identity) is dict and set(identity) == {'path', 'sha256', 'size_bytes'}, 'SOURCE_MANIFEST_KEYS')
        require(type(identity['path']) is str and type(identity['sha256']) is str and
                re.fullmatch('[a-f0-9]{64}', identity['sha256']) and
                type(identity['size_bytes']) is int and 0 < identity['size_bytes'] <= 1024**3,
                'LARGE_SOURCE_IDENTITY')
    tool = config['toolchain']
    require(type(tool) is dict and set(tool) == {'tshark_path', 'tshark_sha256'} and
            type(tool['tshark_path']) is str and bool(tool['tshark_path']) and
            type(tool['tshark_sha256']) is str and re.fullmatch('[a-f0-9]{64}', tool['tshark_sha256']),
            'LARGE_TOOLCHAIN_IDENTITY')
    return limits


def select_case(manifest, case_id, toolchain):
    """Select one explicit source pair; filenames/labels never select packet candidates.
    Chọn cặp nguồn theo manifest; không suy protocol hoặc nhãn packet từ tên file.
    """
    require(type(manifest) is dict and set(manifest) == {'format', 'label_policy', 'cases'}, 'MANIFEST_KEYS')
    require(manifest['format'] == MANIFEST and manifest['label_policy'] == POLICY, 'MANIFEST_VERSION_POLICY')
    cases = manifest['cases']
    require(type(cases) is list and 1 <= len(cases) <= 24, 'MANIFEST_CASE_COUNT')
    require(all(type(c) is dict and type(c.get('case_id')) is str for c in cases), 'MANIFEST_CASE_SCHEMA')
    require(len({c['case_id'] for c in cases}) == len(cases), 'DUPLICATE_CASE_ID')
    selected = [c for c in cases if c['case_id'] == case_id]
    require(len(selected) == 1, 'CASE_NOT_FOUND')
    case = selected[0]
    keys = {'case_id', 'enabled', 'block_reason', 'protocol', 'pcap', 'csv',
            'expected_pcap_records', 'expected_csv_rows'}
    require(set(case) == keys and type(case['enabled']) is bool, 'MANIFEST_CASE_KEYS')
    require(case['enabled'], 'CASE_BLOCKED:' + str(case['block_reason']))
    require(case['block_reason'] is None, 'ENABLED_CASE_HAS_BLOCK_REASON')
    config = {k: v for k, v in case.items() if k not in ('enabled', 'block_reason')}
    config.update(schema_version='1.2.0', kind='full_pair_mapping_audit', key_profile=KEY_PROFILE,
                  label_policy=POLICY, limits=asdict(LargeAuditLimits()), toolchain=toolchain)
    validate_large_config(config)
    return config


def large_record(row, *, protocol, **kwargs):
    require(protocol in ('TCP', 'ICMP'), 'PROTOCOL_NOT_APPROVED_UDP_ARP_BLOCKED')
    record = diagnostic_record(row, **kwargs)
    allowed = 'TCPv4' if protocol == 'TCP' else 'ICMPv4_DIAGNOSTIC'
    if record.key is not None and record.key[0] != allowed:
        return replace(record, key=None, reason='OUTSIDE_CONFIGURED_MAPPING_PROTOCOL')
    return record


def compact_key_storage(record):
    """Pack exact shared fields rather than hash+JSON; no hash collision matching.
    Đóng gói nguyên giá trị shared key, không dùng hash đơn lẻ làm bằng chứng match.
    """
    if record.key is None:
        return None, None
    tag, source, destination, *numbers = record.key
    source = ipaddress.IPv4Address(source).packed
    destination = ipaddress.IPv4Address(destination).packed
    if tag == 'TCPv4':
        require(len(numbers) == 6, 'TCP_COMPACT_KEY_FIELDS')
        raw = struct.pack('>B4s4sHHHHIH', 6, source, destination, *numbers)
    elif tag == 'ICMPv4_DIAGNOSTIC':
        require(len(numbers) == 2, 'ICMP_COMPACT_KEY_FIELDS')
        raw = struct.pack('>B4s4sHH', 1, source, destination, *numbers)
    else:
        raise AlignmentError('UNKNOWN_COMPACT_KEY_PROFILE')
    return raw, ''


def binary_target(label, attack_type):
    """Keep source taxonomy, accept only consistent dataset-provided labels.
    Chỉ nhận nhãn 0/1 nhất quán; không chuyển nhãn thiếu/sai thành Normal.
    """
    if label == '0' and attack_type == 'Normal':
        return 0
    if label == '1' and attack_type in ATTACK_TYPES:
        return 1
    return None


def mapping_contract():
    """Separate active packet annotations from not-yet-implemented flow supervision.
    Phân biệt nhãn packet NIDS đang áp dụng và bước gán nhãn flow chưa thực hiện.
    """
    return dict(contract_id=POLICY, activation='ACTIVE_DATASET_PROVIDED_PACKET_MAPPING',
        target=dict(normal=0, attack=1, unknown=None), positive_types=list(ATTACK_TYPES),
        negative=dict(Attack_label='0',Attack_type='Normal'), unmatched='UNKNOWN', ambiguous='UNKNOWN',
        conflicting_source_label='UNKNOWN', global_one_to_one=True,
        clock='EXACT_OBSERVED_MODULO_DAY_WITNESS; single UTC day; not trusted timezone conversion',
        labels_used_for_candidate_selection=False, labels_are_independent_ground_truth=False,
        flow_supervision='NOT_IMPLEMENTED_BY_THIS_MAPPING_TASK',
        full_window=dict(all_real_0=0,all_real_1=1,contains_unknown='UNKNOWN',
            contains_0_and_1='MIXED',padding_is_packet=False,overflow_packets_included=True,majority_vote=False),
        ordering=dict(preserve_raw_source=True,source_timestamp_edits=False,sorting_performed=False,
                      timestamp_regression='REQUIRES_APPROVED_PROCESSING_VIEW_BEFORE_FLOW_EXTRACTION'),
        training=False)


def install_label_view(db):
    """Read-only annotation projection; every source packet remains represented.
    View nhãn giữ mọi packet, chỉ unique content+time và nhãn hợp lệ mới có y=0/1.
    """
    # Fixed vocabulary is code-owned; raw CSV strings are SQL values, never SQL code.
    # Vocabulary cố định từ source; nội dung CSV không được nối vào câu lệnh SQL.
    types = ','.join("'" + value + "'" for value in ATTACK_TYPES)
    eligible = "w.status='UNIQUE_CONTENT_AND_TIME_CANDIDATE'"
    label = ("CASE WHEN " + eligible + " AND c.alabel='0' AND c.atype='Normal' THEN 0 "
             "WHEN " + eligible + " AND c.alabel='1' AND c.atype IN (" + types + ") THEN 1 ELSE NULL END")
    db.execute('''CREATE VIEW mapped_labels AS
        SELECT p.ord source_ordinal,p.ts timestamp_ns,w.cord csv_ordinal,w.status mapping_status,
               ''' + label + ''' y,c.atype source_attack_type,
               CASE WHEN c.atype='OS_Fingerprinting' THEN 'Fingerprinting' ELSE c.atype END attack_type
        FROM p JOIN witnessed w ON w.pord=p.ord LEFT JOIN c ON c.ord=w.cord''')
    db.commit()
    counts = [dict(y=y, attack_type=kind, packets=n) for y, kind, n in db.execute(
        'SELECT y,CASE WHEN y IS NOT NULL THEN attack_type ELSE NULL END,count(*) '
        'FROM mapped_labels GROUP BY y,CASE WHEN y IS NOT NULL THEN attack_type ELSE NULL END')]
    total = db.execute('SELECT count(*) FROM mapped_labels').fetchone()[0]
    known = db.execute('SELECT count(*) FROM mapped_labels WHERE y IS NOT NULL').fetchone()[0]
    require(total == db.execute('SELECT count(*) FROM p').fetchone()[0], 'LABEL_DENOMINATOR_MISMATCH')
    return dict(policy=POLICY, label_provenance='DATASET_PROVIDED_MAPPED',
                readiness='MAPPED_PACKETS_AVAILABLE' if known else 'BLOCKED_NO_MAPPED_LABELS',
                independent_ground_truth_verified=False, packet_count=total, mapped_packets=known,
                unknown_packets=total-known, counts=counts, sqlite_view='mapped_labels',
                inference='0=Normal,1=Attack,NULL=UNKNOWN; family remains metadata, not CNN input',
                filename_labels_used=False, labels_used_for_candidate_selection=False)
