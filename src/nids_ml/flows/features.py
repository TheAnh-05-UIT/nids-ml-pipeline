"""Validate packet metadata and derive the fixed 11-feature flow representation.
Kiểm metadata packet và tạo biểu diễn flow cố định 11 features.
"""
from dataclasses import dataclass
from copy import deepcopy
import hashlib
import ipaddress
import math
import re

from nids_ml.labeling.contracts import canonical_bytes, integer, obj, structural, text

EXTRACTOR_VERSION='lucid-flow-core/1.0.0'
FEATURE_ORDER=('Time','Packet Length','Higher Protocol','IP Flags','Protocols','TCP Length',
               'TCP Ack','TCP Flags','TCP Window Size','UDP Length','ICMP Type')
# CountVectorizer mặc định sắp lexical; tcp đứng trước telnet, khác list gốc.
PROTOCOL_VOCABULARY=('arp','data','dns','ftp','http','icmp','ip','ssdp','ssl','tcp','telnet','udp')
PROTOCOL_BITS={word:1<<i for i,word in enumerate(PROTOCOL_VOCABULARY)}
TOKEN_PATTERN=r'(?u)\b\w\w+\b'
PROFILES={'flow-cnn-n20-t10':20,'paper-table3-n10':10}
MAXIMA=(10,65536,4294967296,65536,4096,65536,4294967296,65536,65536,65536,256)


class ExtractionError(ValueError):
    """Lỗi fail-closed; không chuyển packet lỗi thành benign hoặc vector zeros."""


class UnsupportedPacketError(ExtractionError):
    """Packet ngoài eligibility profile hiện tại."""


class ResourceLimitError(ExtractionError):
    """Vượt resource contract; không evict flow âm thầm."""


IDENTIFIER=dict(type='string',minLength=1,maxLength=64,pattern=r'^[a-z0-9][a-z0-9._-]*$')
SHA=dict(type='string',minLength=64,maxLength=64,pattern=r'^[a-f0-9]{64}$')
CONTEXT_SCHEMA=obj(dict(capture_id=IDENTIFIER,capture_sha256=SHA,run_id=IDENTIFIER,segment_id=IDENTIFIER))


def nullable(properties):
    schema=obj(properties)
    schema['type']=['object','null']
    return schema


PACKET_SCHEMA=dict(**{'$schema':'https://json-schema.org/draft/2020-12/schema'},
    title='Decoded synthetic IPv4 packet metadata; no raw capture/labels',
    **obj(dict(schema_version=dict(type='string',const='1.0.0'),input_kind=dict(type='string',const='synthetic_fixture'),
        capture=CONTEXT_SCHEMA,ordinal=integer(1,2**64-1),timestamp_ns=integer(0,2**63-1),
        ip_version=dict(type='integer',enum=[0,4,6]),src_ip=dict(type=['string','null'],maxLength=64),
        dst_ip=dict(type=['string','null'],maxLength=64),ip_protocol=integer(0,255),ip_length=integer(0,65535),
        ip_flags=integer(0,7),fragmented=dict(type='boolean'),highest_layer=dict(type='string',minLength=1,maxLength=64),
        protocol_stack=dict(type='string',minLength=1,maxLength=256),
        tcp=nullable(dict(src_port=integer(0,65535),dst_port=integer(0,65535),length=integer(0,65535),
            ack=integer(0,2**32-1),flags=integer(0,65535),window_size=integer(0,65535))),
        udp=nullable(dict(src_port=integer(0,65535),dst_port=integer(0,65535),length=integer(8,65535))),
        icmp=nullable(dict(type=integer(0,255),code=integer(0,255),identifier=integer(0,65535))))))


def checked(value,schema):
    try:structural(value,schema)
    except (ValueError,TypeError,RecursionError) as error:raise ExtractionError(str(error)) from error


@dataclass(frozen=True)
class CaptureContext:
    capture_id:str
    capture_sha256:str
    run_id:str
    segment_id:str

    @classmethod
    def from_dict(cls,value):
        checked(value,CONTEXT_SCHEMA)
        if any(re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,63}',value[k]) is None
               for k in ('capture_id','run_id','segment_id')):
            raise ExtractionError('Invalid capture-local identifier')
        return cls(**value)

    def to_dict(self):
        return dict(capture_id=self.capture_id,capture_sha256=self.capture_sha256,run_id=self.run_id,segment_id=self.segment_id)


def packet_schema(input_kind='synthetic_fixture'):
    """Giữ schema D1 bất biến; metadata PCAP có scope/version riêng từ D2B."""
    if input_kind == 'synthetic_fixture':
        return PACKET_SCHEMA
    if input_kind != 'decoded_pcap':
        raise ExtractionError('Unsupported packet input scope')
    schema = deepcopy(PACKET_SCHEMA)
    schema['title'] = 'Decoded PCAP IPv4 metadata; no supervised labels'
    schema['properties']['schema_version']['const'] = '1.1.0'
    schema['properties']['input_kind']['const'] = 'decoded_pcap'
    return schema


def configuration(profile='flow-cnn-n20-t10', *, input_kind='synthetic_fixture'):
    if type(profile) is not str or profile not in PROFILES:raise ExtractionError('Unsupported extraction profile')
    packet_schema(input_kind)
    result = dict(schema_version='1.0.0',extractor_version=EXTRACTOR_VERSION,profile=profile,
        input_kind='synthetic_fixture',feature_schema_version='lucid-flow-11/1.0.0-candidate',
        feature_order=list(FEATURE_ORDER),tensor_shape=[PROFILES[profile],11,1],tensor_dtype='<f4',
        mode='lucid_compatible_eligible_ipv4_v1',window_ns=10000000000,
        window_boundary='STRICT_GREATER_THAN_ANCHOR_PLUS_T',packet_order='CAPTURE_ORDINAL',
        regression_policy='FAIL_CLOSED',unsupported_policy='FAIL_CLOSED',label_policy='UNKNOWN_ONLY',
        protocol_vocabulary=dict(PROTOCOL_BITS),protocol_token_pattern=TOKEN_PATTERN,
        highest_layer_encoding='SHA256_UTF8_CASE_SENSITIVE_MOD_100000000',
        normalization_version='lucid-fixed-nominal/1.0.0',normalization_maxima=list(MAXIMA),
        normalization_fit='NONE',padding='ZERO_AFTER_NORMALIZATION',
        hardening='integer_ns; strict eligibility/ranges/finite; no silent nan_to_num or sorting')
    if input_kind == 'decoded_pcap':
        result.update(schema_version='1.1.0', extractor_version='lucid-flow-core/1.1.0',
                      input_kind='decoded_pcap', decoder_contract_version='pcap-decoder/1.0.0',
                      max_active_flows_hard=256)
    return result


def validate_configuration(value):
    if type(value) is not dict:raise ExtractionError('Expected extraction configuration')
    expected=configuration(value.get('profile'), input_kind=value.get('input_kind'))
    if canonical_bytes(value)!=canonical_bytes(expected):raise ExtractionError('Feature/flow configuration mismatch')
    return expected


def encode_protocol_stack(value):
    if type(value) is not str or not 1<=len(value)<=256:raise ExtractionError('Invalid protocol stack')
    tokens=set(re.findall(TOKEN_PATTERN,value.lower()))
    return sum(PROTOCOL_BITS[t] for t in tokens if t in PROTOCOL_BITS),tuple(sorted(tokens-set(PROTOCOL_BITS)))


@dataclass(frozen=True)
class FeaturePacket:
    context:CaptureContext
    ordinal:int
    timestamp_ns:int
    forward_tuple:tuple
    canonical_tuple:tuple
    features:tuple
    packet_sha256:str
    unknown_protocol_tokens:tuple[str,...]
    icmp_identifier:int|None


def extract_features(packet, *, input_kind='synthetic_fixture'):
    checked(packet,packet_schema(input_kind))
    context=CaptureContext.from_dict(packet['capture'])
    if packet['ip_version']!=4 or packet['ip_protocol'] not in (1,6,17) or packet['fragmented']:
        raise UnsupportedPacketError('Only decoded, unfragmented IPv4 TCP/UDP/ICMP supported')
    addresses=[]
    for field in ('src_ip','dst_ip'):
        try:ip=ipaddress.ip_address(packet[field])
        except (ValueError,TypeError) as error:raise ExtractionError('Expected literal IPv4 endpoints') from error
        if ip.version!=4 or str(ip)!=packet[field]:raise ExtractionError('Noncanonical IPv4 endpoint')
        addresses.append(str(ip))
    if len(packet['highest_layer'].encode('utf-8'))>128:raise ExtractionError('Higher protocol name too large')
    protocol=packet['ip_protocol'];kind={6:'tcp',17:'udp',1:'icmp'}[protocol]
    if packet[kind] is None or any(packet[k] is not None for k in ('tcp','udp','icmp') if k!=kind):
        raise ExtractionError('Missing or inconsistent transport fields')
    if packet['ip_length']<(40 if protocol==6 else 28):raise ExtractionError('IP length below minimum headers')
    transport=packet[kind];src_port=dst_port=0
    tcp_values=[0,0,0,0];udp_length=icmp_type=0;icmp_id=None
    if protocol in (6,17):
        src_port,dst_port=transport['src_port'],transport['dst_port']
    if protocol==6:
        if transport['length']>packet['ip_length']-40:raise ExtractionError('TCP payload exceeds IP length')
        tcp_values=[transport['length'],transport['ack'],transport['flags'],transport['window_size']]
    elif protocol==17:
        if transport['length']>packet['ip_length']-20:raise ExtractionError('UDP length exceeds IP payload')
        udp_length=transport['length']
    else:
        icmp_type,icmp_id=transport['type'],transport['identifier']
    bits,unknown=encode_protocol_stack(packet['protocol_stack'])
    higher=int(hashlib.sha256(packet['highest_layer'].encode('utf-8')).hexdigest(),16)%100000000
    values=(packet['timestamp_ns']/1e9,packet['ip_length'],higher,packet['ip_flags'],bits,*tcp_values,udp_length,icmp_type)
    if len(values)!=11 or not all(math.isfinite(v) for v in values):raise ExtractionError('Invalid 11-feature vector')
    forward=(addresses[0],src_port,addresses[1],dst_port,protocol)
    ends=sorted(((addresses[0],src_port),(addresses[1],dst_port)))
    key=(ends[0][0],ends[0][1],ends[1][0],ends[1][1],protocol)
    return FeaturePacket(context,packet['ordinal'],packet['timestamp_ns'],forward,key,values,
        hashlib.sha256(canonical_bytes(packet)).hexdigest(),unknown,icmp_id)


def normalize_row(values,relative_ns):
    if type(values) not in (list,tuple) or type(relative_ns) is not int or not 0<=relative_ns<=10000000000 or len(values)!=11:
        raise ExtractionError('Invalid relative timestamp or feature count')
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values):
        raise ExtractionError('Nonfinite or invalid raw feature')
    raw=(relative_ns/1e9,*values[1:])
    if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=maximum for v,maximum in zip(raw,MAXIMA)):
        raise ExtractionError('Feature outside fixed nominal bounds')
    return tuple(float(v/maximum) for v,maximum in zip(raw,MAXIMA))
