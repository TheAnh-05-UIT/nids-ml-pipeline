"""Define and validate versioned ordered-flow configuration and metadata.
Định nghĩa và kiểm config/metadata ordered-flow theo version.
"""
import hashlib
import ipaddress
import math
import struct

from nids_ml.flows.features import (CONTEXT_SCHEMA, ExtractionError, PROFILES,
    configuration as capture_configuration)
from nids_ml.labeling.contracts import canonical_bytes, integer, obj, SHA, structural, strict_json

VERSION='2.0.0'
EXTRACTOR_VERSION='lucid-ordered-flow/2.0.0'


def configuration(profile='flow-cnn-n20-t10'):
    value=capture_configuration(profile)
    value.update(schema_version=VERSION, extractor_version=EXTRACTOR_VERSION,
        mode='ordered_synthetic_eligible_ipv4_v1', packet_order='VERIFIED_PROCESSING_INDEX',
        ordering_contract_version='1.0.0', source_ordinal_policy='PRESERVE_ORIGINAL_NONMONOTONIC',
        regression_policy='REJECT_PROCESSING_KEY_OR_INDEX_REGRESSION',
        hardening='verify view digest and source packet hashes; no ordinal/timestamp edits',
        real_data_export='BLOCKED', HDF5_integration='NOT_ACTIVE')
    return value


def validate_configuration(value):
    if type(value) is not dict:
        raise ExtractionError('ORDERED_CONFIGURATION_REQUIRED')
    expected=configuration(value.get('profile'))
    if canonical_bytes(value)!=canonical_bytes(expected):
        raise ExtractionError('ORDERED_CONFIGURATION_MISMATCH')
    return strict_json(canonical_bytes(expected))


CONFIG_SCHEMA=dict(**{'$schema':'https://json-schema.org/draft/2020-12/schema'},
    title='Ordered-flow synthetic configuration v2; capture-order defaults unchanged',
    type='object',enum=[configuration(p) for p in sorted(PROFILES)])

def _const(value):
    return dict(type='boolean' if type(value) is bool else 'string',const=value)


METADATA_SCHEMA=dict(**{'$schema':'https://json-schema.org/draft/2020-12/schema'},
    title='Ordered-flow synthetic metadata v2; original ordinals and processing indices are distinct',
    **obj(dict(schema_version=_const(VERSION),extractor_version=_const(EXTRACTOR_VERSION),
        feature_schema_version=_const('lucid-flow-11/1.0.0-candidate'),
        profile=dict(type='string',enum=sorted(PROFILES)),
        normalization_version=_const('lucid-fixed-nominal/1.0.0'),configuration_sha256=SHA,
        capture=CONTEXT_SCHEMA,flow_tuple=dict(type='array',minItems=5,maxItems=5),
        first_observed_tuple=dict(type='array',minItems=5,maxItems=5),
        window_index=integer(),anchor_timestamp_ns=integer(),nominal_end_timestamp_ns=integer(0,2**64-1),
        window_boundary=_const('STRICT_GREATER_THAN_ANCHOR_PLUS_T'),full_window_complete=_const(True),
        first_ordinal=integer(1),last_ordinal=integer(1),first_timestamp_ns=integer(),last_timestamp_ns=integer(),
        real_packet_count=integer(1,1000000),retained_packet_count=integer(1,20),overflow_packet_count=integer(0,1000000),
        retained_ordinals=dict(type='array',items=integer(1),minItems=1,maxItems=20),
        real_packet_mask=dict(type='array',items=dict(type='boolean'),minItems=10,maxItems=20),
        forward_packet_count=integer(0,1000000),reverse_packet_count=integer(0,1000000),
        full_packet_digest=SHA,overflow_packet_digest=SHA,icmp_identifier_changed=dict(type='boolean'),
        full_window_label=_const('UNKNOWN'),retained_fragment_label=_const('UNKNOWN'),
        supervised_eligible=_const(False),label_coverage=_const('NOT_EVALUATED'),
        overflow_metadata=_const('COUNTS_BOUNDS_DIGEST_PLUS_STREAMED_RECEIPTS'),stream_completion_required=_const(True),
        input_kind=_const('synthetic_fixture'),ordering_contract_version=_const('1.0.0'),
        processing_view_sha256=SHA,packet_digest_order=_const('PROCESSING_INDEX'),
        first_processing_index=integer(1),last_processing_index=integer(1),
        retained_processing_indices=dict(type='array',items=integer(1),minItems=1,maxItems=20),
        min_source_ordinal=integer(1),max_source_ordinal=integer(1),
        full_processing_digest=SHA,overflow_processing_digest=SHA,
        real_data_export=_const('BLOCKED'),HDF5_integration=_const('NOT_ACTIVE'))))


def validate_metadata(value, config, view):
    """Kiểm invariants mới; source ordinals có thể giảm, processing indices thì không."""
    try:structural(value,METADATA_SCHEMA)
    except (ValueError,TypeError) as error:raise ExtractionError('ORDERED_METADATA_SCHEMA:'+str(error)) from error
    config=validate_configuration(config)
    source=view['source'];start=source['source_first_ordinal'];end=start+view['record_count']-1
    n=config['tensor_shape'][0];count=value['retained_packet_count'];real=value['real_packet_count']
    def require(ok,reason):
        if not ok:raise ExtractionError(reason)
    require(value['capture']==source['capture'],'ORDERED_CAPTURE_MISMATCH')
    require(value['profile']==config['profile'] and value['configuration_sha256']==hashlib.sha256(canonical_bytes(config)).hexdigest(),
            'ORDERED_CONFIG_BINDING')
    require(value['processing_view_sha256']==view['processing_order_sha256'],'ORDERED_VIEW_BINDING')
    require(count==min(n,real) and value['overflow_packet_count']==real-count,'ORDERED_PACKET_COUNTS')
    require(value['forward_packet_count']+value['reverse_packet_count']==real,'ORDERED_DIRECTION_COUNTS')
    ordinals=value['retained_ordinals'];indices=value['retained_processing_indices']
    require(len(ordinals)==len(indices)==count and len(set(ordinals))==count,'ORDERED_RETAINED_LINEAGE')
    require(all(start<=v<=end for v in ordinals),'SOURCE_ORDINAL_RANGE')
    require(all(a<b for a,b in zip(indices,indices[1:])),'PROCESSING_INDEX_REGRESSION')
    require(value['first_processing_index']==indices[0]<=indices[-1]<=value['last_processing_index']<=view['record_count'],
            'PROCESSING_INDEX_BOUNDS')
    require(real<=view['record_count'] and value['last_processing_index']-value['first_processing_index']+1>=real,
            'PROCESSING_COUNT_BOUNDS')
    require(value['first_ordinal']==ordinals[0] and
            start<=value['min_source_ordinal']<=min(ordinals)<=max(ordinals)<=value['max_source_ordinal']<=end,
            'SOURCE_LINEAGE_BOUNDS')
    require(value['min_source_ordinal']<=value['last_ordinal']<=value['max_source_ordinal'],'SOURCE_LAST_ORDINAL')
    if real==count:
        require(indices[-1]==value['last_processing_index'] and ordinals[-1]==value['last_ordinal'],
                'COMPLETE_RETAINED_BOUNDS')
    require(value['real_packet_mask']==[i<count for i in range(n)],'ORDERED_PADDING_MASK')
    require(value['nominal_end_timestamp_ns']==value['anchor_timestamp_ns']+config['window_ns'] and
            value['anchor_timestamp_ns']<=value['first_timestamp_ns']<=value['last_timestamp_ns']<=value['nominal_end_timestamp_ns'],
            'ORDERED_WINDOW_BOUNDS')
    if not value['overflow_packet_count']:
        require(value['overflow_packet_digest']==hashlib.sha256(b'').hexdigest() and
                value['overflow_processing_digest']==hashlib.sha256(b'').hexdigest(),'EMPTY_OVERFLOW_DIGEST')
    first=value['first_observed_tuple']
    try:
        require(all(type(first[i]) is str and str(ipaddress.IPv4Address(first[i]))==first[i] for i in (0,2)),
                'ORDERED_FLOW_IPS')
        require(all(type(first[i]) is int and 0<=first[i]<=65535 for i in (1,3)) and
                type(first[4]) is int and first[4] in (1,6,17),'ORDERED_FLOW_PORTS_PROTOCOL')
    except (ValueError,TypeError) as error:raise ExtractionError('ORDERED_FLOW_IDENTITY') from error
    if first[4]==1:require(first[1]==first[3]==0,'ORDERED_ICMP_PORTS')
    ends=sorted(((first[0],first[1]),(first[2],first[3])))
    require(canonical_bytes(value['flow_tuple'])==canonical_bytes(
        [ends[0][0],ends[0][1],ends[1][0],ends[1][1],first[4]]),'ORDERED_CANONICAL_FLOW')
    return value


def validate_tensor_bytes(shape, data, metadata, config, view):
    validate_metadata(metadata,config,view)
    n=config['tensor_shape'][0];count=metadata['retained_packet_count']
    if (type(shape) is not tuple or any(type(v) is not int for v in shape)
            or shape!=(n,11,1) or type(data) is not bytes or len(data)!=n*44):
        raise ExtractionError('ORDERED_TENSOR_SHAPE_OR_BYTES')
    values=[v[0] for v in struct.iter_unpack('<f',data)]
    if not all(math.isfinite(v) and 0<=v<=1 for v in values) or any(values[count*11:]):
        raise ExtractionError('ORDERED_NONFINITE_NORMALIZATION_OR_PADDING')
    times=values[:count*11:11]
    if times[0]!=0 or any(a>b for a,b in zip(times,times[1:])):
        raise ExtractionError('ORDERED_TENSOR_TIME')
