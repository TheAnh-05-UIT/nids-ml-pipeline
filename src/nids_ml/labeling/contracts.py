"""Validate synthetic label-rule schemas, evidence and approval bindings.
Kiểm schema rule synthetic và ràng buộc evidence/approval.
"""
import hashlib
import json

from nids_ml.validation.contracts import validate_schema

SCHEMA_VERSION = '1.0.0'
RESOLVER_VERSION = 'flow-label-core/0.1.0'
MAX_JSON_BYTES = 1048576


class ContractError(ValueError):
    """Dừng an toàn khi contract, identity hoặc approval không hợp lệ."""


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode('utf-8')


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def rule_digest(value):
    # Approval nằm ngoài nội dung được duyệt để không tạo vòng hash tự tham chiếu.
    return digest_bytes(canonical_bytes({k:v for k,v in value.items()
                                        if k not in ('approval','rule_set_sha256')}))


def strict_json(data):
    if type(data) is not bytes or len(data) > MAX_JSON_BYTES:
        raise ContractError('JSON must be bytes, at most 1 MiB')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ContractError('Duplicate JSON key')
            result[key] = value
        return result
    def reject(_):
        raise ContractError('Non-finite JSON value')
    try:
        return json.loads(data.decode('utf-8'), object_pairs_hook=unique, parse_constant=reject)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ContractError('Invalid strict JSON') from error


def obj(properties):
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


def text(pattern=None):
    result = dict(type='string', minLength=1, maxLength=256)
    if pattern:
        result['pattern'] = pattern
    return result


def integer(low=0, high=2**63-1):
    return dict(type='integer', minimum=low, maximum=high)


def choices(*values):
    return dict(type='string', enum=list(values))


def items(schema, minimum=1, maximum=128):
    return dict(type='array', items=schema, minItems=minimum, maxItems=maximum, uniqueItems=True)


ID = text(r'^[a-z0-9][a-z0-9._-]{0,63}$')
SHA = dict(type='string', pattern=r'^[a-f0-9]{64}$', minLength=64, maxLength=64)
INTERVAL = obj(dict(start_ns=integer(), end_ns=integer()))
EVIDENCE = obj(dict(evidence_id=ID, sha256=SHA, reference=text(r'^synthetic://'),
                    anchor=text(), curator=text(), scope=choices('synthetic_fixture')))
SEGMENT = obj(dict(segment_id=ID, run_id=ID, ordinal_start=integer(1), ordinal_end=integer(1),
    clock=obj(dict(basis=choices('synthetic_epoch_ns'), offset_ns=integer(-(2**62),2**62),
                   uncertainty_ns=integer(0,10**12)))))
RULE = obj(dict(rule_id=ID, segment_id=ID, run_id=ID,
    label=dict(type='integer',enum=[0,1]),
    scenario=choices('BENIGN','TCP_SYN','UDP','HTTP','ICMP'),
    evidence_ids=items(ID), response_evidence_ids=items(ID,0), interval=INTERVAL,
    source_ip=text(), target_ip=text(), source_role=choices('benign','attacker'),
    target_role=choices('benign','victim'), protocol=choices('TCP','UDP','ICMP'),
    source_ports=items(integer(0,65535),1,64), target_ports=items(integer(0,65535),1,64),
    predicate=choices('BENIGN_EVENT','TCP_SYN','UDP_EVENT','HTTP_GET','ICMP_ECHO_REQUEST'),
    direction=choices('FORWARD','BIDIRECTIONAL'),
    response_policy=choices('UNKNOWN','VERIFIED_SAME_LABEL'),
    address_assumption=choices('STABLE_UNTRANSLATED_UNSPOOFED')))
APPROVAL = obj(dict(status=choices('SYNTHETIC_TEST_ONLY'), scope=choices('synthetic_fixture'),
    record_id=ID, reviewer=text(), approved_rule_set_sha256=SHA, record_sha256=SHA))
RULE_SCHEMA = dict(
    **{'$schema':'https://json-schema.org/draft/2020-12/schema'},
    title='Synthetic flow label rule set; not a real-data approval',
    **obj(dict(schema_version=dict(type='string',const=SCHEMA_VERSION),
        resolver_version=dict(type='string',const=RESOLVER_VERSION), rule_set_id=ID,
        input_kind=choices('synthetic_fixture'), label_target=choices('packet'), rule_set_sha256=SHA,
        capture=obj(dict(capture_id=ID, sha256=SHA, size_bytes=integer(1),
            source_revision=text(), source_member=text(), segments=items(SEGMENT))),
        evidence=items(EVIDENCE), approval=APPROVAL,
        policies=obj(dict(unmatched=choices('UNKNOWN'), unknown=choices('QUARANTINE'),
            mixed=choices('QUARANTINE'), ordering=choices('REJECT_REGRESSION'),
            max_packets=integer(20,20))), rules=items(RULE))))
PACKET_SCHEMA = obj(dict(capture_id=ID, capture_sha256=SHA, capture_size_bytes=integer(1),
    segment_id=ID, run_id=ID, ordinal=integer(1), timestamp_ns=integer(),
    src_ip=text(), dst_ip=text(), protocol=text(r'^[A-Z0-9_]{1,16}$'),
    src_port=integer(0,65535), dst_port=integer(0,65535), tcp_syn=dict(type='boolean'),
    tcp_ack=dict(type='boolean'), application=choices('NONE','HTTP'),
    http_method=choices('NONE','GET','POST'), icmp_type=integer(0,255),
    icmp_code=integer(0,255), icmp_identifier=integer(0,65535)))


def structural(value, schema):
    try:
        validate_schema(value, schema)
    except (ValueError, TypeError, RecursionError) as error:
        raise ContractError(str(error)) from error
