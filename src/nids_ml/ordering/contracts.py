"""Validate ordering policies, source identity, digests and resource budgets.
Kiểm ordering policy, source identity, digest và budget tài nguyên.
"""
from dataclasses import dataclass
import hashlib

from nids_ml.flows.features import CaptureContext, CONTEXT_SCHEMA, packet_schema
from nids_ml.labeling.contracts import canonical_bytes, integer, obj, SHA, structural, strict_json

VERSION = '1.0.0'
FORMAT = 'nids-packet-processing-view'
MAX_PACKET_BYTES = 16384


class OrderingError(ValueError):
    """Lỗi contract/lineage/completion: không xuất processing view hợp lệ giả."""


class OrderingLimitError(OrderingError):
    """Vượt budget thì dừng; không tự nới giới hạn hoặc bỏ packet."""


def ordering_policy():
    return dict(id='stable-packet-processing-view', version=VERSION,
                scope='SYNTHETIC_TEST_ONLY', sort_keys=['timestamp_ns', 'source_packet_ordinal'],
                tie_breaker='SOURCE_ORDINAL_ASCENDING', raw_mutation=False,
                timestamp_correction=False, default_flow_extractor='UNCHANGED_CAPTURE_ORDER',
                real_data_enabled=False)


SOURCE_SCHEMA = dict(**{'$schema': 'https://json-schema.org/draft/2020-12/schema'},
    title='Synthetic packet ordering source contract; not real PCAP approval',
    **obj(dict(schema_version=dict(type='string', const=VERSION),
        input_kind=dict(type='string', const='synthetic_fixture'),
        capture=CONTEXT_SCHEMA, synthetic_capture_text=dict(type='string', minLength=1, maxLength=4096),
        source_first_ordinal=integer(1), expected_packets=integer(1,3500000),
        expected_source_order_sha256=SHA, input_complete=dict(type='boolean', const=True),
        ordering_policy=dict(type='object', const=ordering_policy()))))


@dataclass(frozen=True)
class OrderingLimits:
    max_records: int = 100000
    sqlite_bytes: int = 64 * 1048576
    cache_kib: int = 1024
    scratch_bytes: int = 160 * 1048576
    min_free_bytes: int = 16 * 1048576
    timeout_seconds: int = 120
    commit_every: int = 128

    def validate(self):
        bounds = dict(max_records=(1,3500000), sqlite_bytes=(32768,512*1048576),
                      cache_kib=(64,8192), scratch_bytes=(131072,1536*1048576),
                      min_free_bytes=(1048576,1024*1048576), timeout_seconds=(1,300),
                      commit_every=(1,512))
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise OrderingLimitError('INVALID_ORDERING_LIMIT:'+name)
        if self.sqlite_bytes % 4096 or self.scratch_bytes < 2*self.sqlite_bytes+65536:
            raise OrderingLimitError('SQLITE_PAGE_OR_SCRATCH_BUDGET')
        return self


def checked(value, schema):
    try:
        structural(value, schema)
    except (TypeError, ValueError, RecursionError) as error:
        raise OrderingError('ORDERING_SCHEMA:'+str(error)) from error


def validate_source(value):
    checked(value, SOURCE_SCHEMA)
    if canonical_bytes(value['ordering_policy']) != canonical_bytes(ordering_policy()):
        raise OrderingError('ORDERING_POLICY_MISMATCH')
    context = CaptureContext.from_dict(value['capture'])
    text = value['synthetic_capture_text'].encode('utf-8')
    if len(text)>4096 or hashlib.sha256(text).hexdigest()!=context.capture_sha256:
        raise OrderingError('SYNTHETIC_LITERAL_CAPTURE_HASH_MISMATCH')
    if value['source_first_ordinal']+value['expected_packets']-1 > 2**63-1:
        raise OrderingError('SOURCE_ORDINAL_SQLITE_RANGE')
    return strict_json(canonical_bytes(value))


def packet_bytes(packet):
    checked(packet, packet_schema('synthetic_fixture'))
    data = canonical_bytes(packet)
    if len(data)>MAX_PACKET_BYTES:
        raise OrderingLimitError('PACKET_METADATA_BYTES')
    return data


def source_order_digest(packets):
    """Digest chuỗi source packet hashes, không sort hoặc lấy nhãn làm key."""
    digest = hashlib.sha256(b'nids-source-order/1\x00')
    for packet in packets:
        digest.update(hashlib.sha256(packet_bytes(packet)).digest())
    return digest.hexdigest()


def make_source_manifest(context, *, synthetic_capture_text, expected_packets,
                         expected_source_order_sha256, source_first_ordinal=1,
                         input_complete=True, input_kind='synthetic_fixture'):
    if not isinstance(context, CaptureContext):
        raise OrderingError('CAPTURE_CONTEXT_REQUIRED')
    return validate_source(dict(schema_version=VERSION, input_kind=input_kind,
        capture=context.to_dict(), synthetic_capture_text=synthetic_capture_text,
        source_first_ordinal=source_first_ordinal, expected_packets=expected_packets,
        expected_source_order_sha256=expected_source_order_sha256,
        input_complete=input_complete, ordering_policy=ordering_policy()))


LIMITS_SCHEMA = obj({name: integer() for name in OrderingLimits.__dataclass_fields__})
VIEW_SCHEMA = dict(**{'$schema': 'https://json-schema.org/draft/2020-12/schema'},
    title='Completed synthetic stable ordering index; not a flow tensor dataset',
    **obj(dict(format=dict(type='string', const=FORMAT), schema_version=dict(type='string', const=VERSION),
        status=dict(type='string', const='COMPLETE_SYNTHETIC_VIEW'), source=SOURCE_SCHEMA,
        record_count=integer(1,3500000), processing_order_sha256=SHA,
        index=obj(dict(filename=dict(type='string', const='index.sqlite'), bytes=integer(1), sha256=SHA)),
        limits=LIMITS_SCHEMA, stats=obj(dict(source_timestamp_regressions=integer(),
            max_regression_ns=integer(), moved_records=integer(), min_timestamp_ns=integer(),
            max_timestamp_ns=integer())),
        labels=dict(type='string', const='NOT_EVALUATED'),
        flow_integration=dict(type='string', const='NOT_ACTIVE'),
        real_data_export=dict(type='string', const='BLOCKED'))))


def validate_view_manifest(value):
    checked(value, VIEW_SCHEMA)
    validate_source(value['source'])
    limits = OrderingLimits(**value['limits']).validate()
    if value['record_count'] != value['source']['expected_packets']:
        raise OrderingError('VIEW_SOURCE_COUNT_MISMATCH')
    if value['record_count'] > limits.max_records:
        raise OrderingLimitError('VIEW_COUNT_EXCEEDS_DECLARED_BUDGET')
    return strict_json(canonical_bytes(value))
