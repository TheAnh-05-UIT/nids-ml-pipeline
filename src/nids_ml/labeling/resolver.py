"""Resolve synthetic packet labels and strictly aggregate complete windows.
Giải nhãn packet synthetic và tổng hợp window đầy đủ theo rule nghiêm ngặt.
"""
from collections import Counter
from dataclasses import dataclass
from enum import Enum
import ipaddress

from nids_ml.labeling.contracts import (PACKET_SCHEMA, RULE_SCHEMA, ContractError,
    canonical_bytes, digest_bytes, rule_digest, strict_json, structural)


class Label(Enum):
    VERIFIED_BENIGN = 0
    VERIFIED_DDOS = 1
    UNKNOWN = 'UNKNOWN'
    MIXED = 'MIXED'


@dataclass(frozen=True)
class PacketDecision:
    label: Label
    reasons: tuple[str, ...]
    rule_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    identity: tuple = ()
    ordinal: int = 0
    timestamp_ns: int = 0
    is_padding: bool = False

    def __post_init__(self):
        if not isinstance(self.label, Label) or type(self.is_padding) is not bool:
            raise ContractError('Expected typed label and padding flag')
        if self.is_padding and (self.label is not Label.UNKNOWN or self.identity or self.ordinal):
            raise ContractError('Padding cannot carry a real packet label or identity')

    @classmethod
    def padding(cls):
        return cls(Label.UNKNOWN, ('PADDING_NOT_A_PACKET',), is_padding=True)

    def numeric_label(self):
        if self.is_padding or self.label in (Label.UNKNOWN, Label.MIXED):
            raise ContractError('Unverified or padding label cannot become a numeric target')
        return self.label.value

    def to_dict(self):
        return dict(label=self.label.value, reasons=list(self.reasons), rule_ids=list(self.rule_ids),
                    evidence_ids=list(self.evidence_ids), ordinal=self.ordinal,
                    timestamp_ns=self.timestamp_ns, is_padding=self.is_padding,
                    flow_identity_sha256=digest_bytes(canonical_bytes(self.identity)) if self.identity else None)


def address(value):
    try:
        return ipaddress.ip_address(value)
    except ValueError as error:
        raise ContractError('Expected a literal IP address') from error


class LabelResolver:
    """Chỉ chấp nhận approval synthetic có digest/evidence nhất quán, không có real mode."""

    def __init__(self, rule_set, evidence_documents, approval_document):
        # Sao chép qua JSON trước khi validate; caller không thể sửa rules sau đó.
        data = strict_json(canonical_bytes(rule_set))
        structural(data, RULE_SCHEMA)
        if data['rule_set_sha256'] != rule_digest(data):
            raise ContractError('Rule set digest mismatch')
        approval = data['approval']
        if type(approval_document) is not bytes or digest_bytes(approval_document) != approval['record_sha256']:
            raise ContractError('Missing or mismatched approval document')
        record = strict_json(approval_document)
        expected = {k:v for k,v in approval.items() if k!='record_sha256'}
        if record != expected or approval['approved_rule_set_sha256'] != data['rule_set_sha256']:
            raise ContractError('Approval does not bind this rule set')
        evidences = {x['evidence_id']:x for x in data['evidence']}
        if len(evidences)!=len(data['evidence']) or type(evidence_documents) is not dict or set(evidence_documents)!=set(evidences):
            raise ContractError('Missing, duplicate or extra evidence documents')
        for key, item in evidences.items():
            blob = evidence_documents[key]
            if type(blob) is not bytes or not blob or len(blob)>1048576 or digest_bytes(blob)!=item['sha256']:
                raise ContractError('Evidence bytes/hash mismatch')
        segments = {x['segment_id']:x for x in data['capture']['segments']}
        if len(segments)!=len(data['capture']['segments']):
            raise ContractError('Duplicate segment ID')
        ordered = sorted(segments.values(),key=lambda x:x['ordinal_start'])
        end = 0
        for segment in ordered:
            if not end < segment['ordinal_start'] <= segment['ordinal_end']:
                raise ContractError('Overlapping or reversed capture segments')
            end = segment['ordinal_end']
        if len({x['rule_id'] for x in data['rules']})!=len(data['rules']):
            raise ContractError('Duplicate rule ID')
        for rule in data['rules']:
            self._validate_rule(rule, segments, evidences)
        self._data = data
        self._segments = segments

    @staticmethod
    def _validate_rule(rule, segments, evidences):
        segment = segments.get(rule['segment_id'])
        if not segment or rule['run_id']!=segment['run_id']:
            raise ContractError('Unknown segment/run identity')
        if rule['interval']['start_ns']>=rule['interval']['end_ns']:
            raise ContractError('Empty or reversed interval')
        refs = rule['evidence_ids']+rule['response_evidence_ids']
        if any(ref not in evidences for ref in refs):
            raise ContractError('Unknown evidence reference')
        for key in ('source_ip','target_ip'):
            ip = address(rule[key])
            if ip.version!=4 or ip.is_multicast or ip.is_unspecified or str(ip)!=rule[key]:
                raise ContractError('Only canonical unicast IPv4 role addresses supported')
        if rule['source_ip']==rule['target_ip']:
            raise ContractError('Source and target roles must differ')
        if rule['label']==0:
            if (rule['source_role'],rule['target_role'],rule['scenario'],rule['predicate']) != ('benign','benign','BENIGN','BENIGN_EVENT'):
                raise ContractError('Benign requires positive benign event/role evidence')
        else:
            expected = {'TCP_SYN':('TCP','TCP_SYN'),'UDP':('UDP','UDP_EVENT'),
                        'HTTP':('TCP','HTTP_GET'),'ICMP':('ICMP','ICMP_ECHO_REQUEST')}
            if (rule['source_role'],rule['target_role'])!=('attacker','victim') or expected.get(rule['scenario'])!=(rule['protocol'],rule['predicate']):
                raise ContractError('DDoS role/protocol/predicate mismatch')
        if rule['protocol']=='ICMP' and (rule['source_ports']!=[0] or rule['target_ports']!=[0]):
            raise ContractError('ICMP must use zero transport ports')
        if rule['response_policy']=='VERIFIED_SAME_LABEL':
            if rule['direction']!='BIDIRECTIONAL' or not rule['response_evidence_ids']:
                raise ContractError('Reverse labels require explicit response evidence')
        elif rule['response_evidence_ids']:
            raise ContractError('Response evidence with no active response policy')

    @property
    def rule_set_sha256(self):
        return self._data['rule_set_sha256']

    def resolve(self, packet):
        structural(packet, PACKET_SCHEMA)
        capture = self._data['capture']
        if (packet['capture_sha256'],packet['capture_size_bytes'],packet['capture_id'])!=(capture['sha256'],capture['size_bytes'],capture['capture_id']):
            raise ContractError('Capture identity mismatch')
        segment = self._segments.get(packet['segment_id'])
        if not segment or segment['run_id']!=packet['run_id'] or not segment['ordinal_start']<=packet['ordinal']<=segment['ordinal_end']:
            raise ContractError('Packet segment/run/ordinal mismatch')
        src,dst=address(packet['src_ip']),address(packet['dst_ip'])
        endpoints = tuple(sorted(((str(src),packet['src_port']),(str(dst),packet['dst_port']))))
        identity=(capture['sha256'],packet['capture_id'],packet['segment_id'],packet['run_id'],packet['protocol'],endpoints,
                  packet['icmp_identifier'] if packet['protocol']=='ICMP' else None,self.rule_set_sha256)
        time = packet['timestamp_ns']+segment['clock']['offset_ns']
        if not 0<=time<=2**63-1:
            raise ContractError('Corrected timestamp outside supported range')
        def decision(label, reasons, rules=(), refs=()):
            return PacketDecision(label,tuple(sorted(reasons)),tuple(sorted(rules)),tuple(sorted(refs)),
                                  identity,packet['ordinal'],time)
        if src.version!=4 or dst.version!=4 or packet['protocol'] not in ('TCP','UDP','ICMP'):
            return decision(Label.UNKNOWN,('UNSUPPORTED_PROTOCOL',))
        if packet['protocol']=='ICMP' and (packet['src_port']!=0 or packet['dst_port']!=0):
            raise ContractError('ICMP packet must have zero transport ports')
        if packet['protocol']!='TCP' and (packet['tcp_syn'] or packet['tcp_ack'] or packet['application']!='NONE' or packet['http_method']!='NONE'):
            raise ContractError('TCP/application metadata on a non-TCP packet')
        if packet['protocol']!='ICMP' and any(packet[k]!=0 for k in ('icmp_type','icmp_code','icmp_identifier')):
            raise ContractError('ICMP metadata on a non-ICMP packet')
        if packet['http_method']!='NONE' and packet['application']!='HTTP':
            raise ContractError('HTTP method without HTTP application evidence')
        labels, matched, refs = set(),set(),set()
        uncertain=False
        for rule in self._data['rules']:
            if rule['segment_id']!=packet['segment_id'] or rule['protocol']!=packet['protocol']:
                continue
            forward=(str(src)==rule['source_ip'] and str(dst)==rule['target_ip'] and
                     packet['src_port'] in rule['source_ports'] and packet['dst_port'] in rule['target_ports'])
            reverse=(str(dst)==rule['source_ip'] and str(src)==rule['target_ip'] and
                     packet['dst_port'] in rule['source_ports'] and packet['src_port'] in rule['target_ports'])
            if forward:
                predicate=rule['predicate']
                valid = (predicate in ('BENIGN_EVENT','UDP_EVENT') or
                         predicate=='TCP_SYN' and packet['tcp_syn'] and not packet['tcp_ack'] or
                         predicate=='HTTP_GET' and packet['application']=='HTTP' and packet['http_method']=='GET' or
                         predicate=='ICMP_ECHO_REQUEST' and packet['icmp_type']==8 and packet['icmp_code']==0)
                if not valid:
                    continue
            elif not (reverse and rule['direction']=='BIDIRECTIONAL' and rule['response_policy']=='VERIFIED_SAME_LABEL'):
                continue
            lo,hi=time-segment['clock']['uncertainty_ns'],time+segment['clock']['uncertainty_ns']
            start,end=rule['interval']['start_ns'],rule['interval']['end_ns']
            if hi<start or lo>=end:
                continue
            matched.add(rule['rule_id']);refs.update(rule['evidence_ids'])
            if reverse:
                refs.update(rule['response_evidence_ids'])
            if lo<start or hi>=end:
                uncertain=True
            else:
                labels.add(Label(rule['label']))
        if len(labels)>1:
            return decision(Label.MIXED,('CONFLICTING_VERIFIED_RULES',),matched,refs)
        if uncertain:
            return decision(Label.UNKNOWN,('CLOCK_BOUNDARY_UNCERTAINTY',),matched,refs)
        if labels:
            return decision(next(iter(labels)),('VERIFIED_SYNTHETIC_RULE',),matched,refs)
        return decision(Label.UNKNOWN,('NO_VERIFIED_MATCH',))


def strict_label(counts):
    if not sum(counts.values()):
        return None
    if counts[Label.MIXED] or counts[Label.VERIFIED_BENIGN] and counts[Label.VERIFIED_DDOS]:
        return Label.MIXED
    if counts[Label.UNKNOWN]:
        return Label.UNKNOWN
    return Label.VERIFIED_DDOS if counts[Label.VERIFIED_DDOS] else Label.VERIFIED_BENIGN


@dataclass(frozen=True)
class WindowDecision:
    full_window_label: Label | None
    retained_fragment_label: Label | None
    status: str
    reasons: tuple[str,...]
    real_packet_count: int
    retained_packet_count: int
    padding_count: int
    full_window_complete: bool

    def numeric_label(self):
        if (self.status!='ACCEPT' or self.full_window_label not in (Label.VERIFIED_BENIGN,Label.VERIFIED_DDOS)
                or self.retained_fragment_label is not self.full_window_label or not self.full_window_complete
                or self.real_packet_count<1 or self.reasons):
            raise ContractError('Quarantined/empty window cannot become a numeric target')
        return self.full_window_label.value

    def to_dict(self):
        return dict(full_window_label=self.full_window_label.value if self.full_window_label is not None else None,
                    retained_fragment_label=self.retained_fragment_label.value if self.retained_fragment_label is not None else None,
                    status=self.status,reasons=list(self.reasons),real_packet_count=self.real_packet_count,
                    retained_packet_count=self.retained_packet_count,padding_count=self.padding_count,
                    full_window_complete=self.full_window_complete)


def aggregate_window(decisions, *, full_window_complete, max_packets=20):
    """Chỉ tổng hợp một window được caller cung cấp; không chia flow/time windows."""
    if type(full_window_complete) is not bool or type(max_packets) is not int or max_packets!=20:
        raise ContractError('Explicit completeness and n=20 required')
    full, retained = Counter(),Counter()
    real=padding=0
    identity=previous=None
    regression=retained_regression=False
    for index,item in enumerate(decisions):
        if index>=100000:
            raise ContractError('Window decision budget exceeded')
        if not isinstance(item,PacketDecision):
            raise ContractError('Expected PacketDecision objects')
        if item.is_padding:
            padding+=1
            continue
        if padding:
            raise ContractError('Padding may only follow real packets')
        if not item.identity or type(item.ordinal) is not int or item.ordinal<1 or type(item.timestamp_ns) is not int:
            raise ContractError('Missing real packet identity/order')
        if identity is None:
            identity=item.identity
        elif item.identity!=identity:
            raise ContractError('Capture/segment/run/bidirectional flow mismatch')
        if previous:
            if item.ordinal<=previous.ordinal:
                raise ContractError('Duplicate or regressing packet ordinals')
            if item.timestamp_ns<previous.timestamp_ns:
                regression=True
                retained_regression |= real<max_packets
        full[item.label]+=1
        if real<max_packets:
            retained[item.label]+=1
        real+=1;previous=item
    if not real:
        return WindowDecision(None,None,'REJECT_EMPTY',('NO_REAL_PACKETS',),0,0,padding,full_window_complete)
    full_label,retained_label=strict_label(full),strict_label(retained)
    reasons=[]
    if not full_window_complete:
        reasons.append('UNEXAMINED_PACKETS')
        if full_label is not Label.MIXED:
            full_label=Label.UNKNOWN
    if regression:
        reasons.append('TIMESTAMP_REGRESSION')
        if full_label is not Label.MIXED:
            full_label=Label.UNKNOWN
    if retained_regression and retained_label is not Label.MIXED:
        retained_label=Label.UNKNOWN
    if full_label!=retained_label:
        reasons.append('TRUNCATION_LABEL_SCOPE_MISMATCH')
    if full_label in (Label.UNKNOWN,Label.MIXED):
        reasons.append('UNVERIFIED_OR_MIXED_WINDOW')
    eligible=(not reasons and full_label in (Label.VERIFIED_BENIGN,Label.VERIFIED_DDOS))
    return WindowDecision(full_label,retained_label,'ACCEPT' if eligible else 'QUARANTINE',
                          tuple(sorted(reasons)),real,min(real,max_packets),padding,full_window_complete)
