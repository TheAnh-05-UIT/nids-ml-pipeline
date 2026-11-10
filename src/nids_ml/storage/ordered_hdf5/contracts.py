"""Bind ordered tensors, processing views and exclusion-only label states.
Ràng buộc ordered tensor, processing view và nhãn chỉ dùng để loại mẫu.
"""
from dataclasses import dataclass
import hashlib

from nids_ml.flows.ordered import OrderedFlowTensor
from nids_ml.flows.ordered_contracts import configuration, CONFIG_SCHEMA, validate_configuration
from nids_ml.labeling.contracts import canonical_bytes, strict_json, obj, structural
from nids_ml.ordering.contracts import VIEW_SCHEMA, validate_view_manifest
from nids_ml.storage.hdf5.schema import ExportError

FORMAT='nids-ordered-flow-hdf5'
VERSION='1.0.0'
JSON_BYTES=16384
MAX_SAMPLES=4096
MAX_RECEIPTS=10000
MAX_EVENTS=MAX_SAMPLES+MAX_RECEIPTS+1
CHUNK=16
DTYPES={'X':'<f4','y':'i1','supervised_mask':'?','label_state':'S8',
        'metadata_json':'S16384','sample_sha256':'S64','events_json':'S16384'}
ATTRS={'format':'S32','schema_version':'S16','status':'S16',
       'sample_count':'<u8','event_count':'<u8','content_sha256':'S64'}
STATES=('UNKNOWN','MIXED')

def require(ok,reason):
    if not ok:raise ExportError(reason)

def sha(data):return hashlib.sha256(data).hexdigest()

def bounded_json(value):
    data=canonical_bytes(value)
    require(len(data)<=JSON_BYTES,'ORDERED_JSON_SIZE_LIMIT')
    return data

def read_json(data):
    require(type(data) is bytes and 0<len(data)<=JSON_BYTES,'ORDERED_JSON_SIZE_LIMIT')
    value=strict_json(data)
    require(canonical_bytes(value)==data,'NONCANONICAL_ORDERED_JSON')
    return value

@dataclass(frozen=True)
class OrderedHdf5Sample:
    tensor: OrderedFlowTensor
    # Chỉ cho phép đánh dấu loại khỏi supervised; đây không phải ground truth.
    label_state: str='UNKNOWN'

def make_manifest(view,profile='flow-cnn-n20-t10'):
    value=dict(format=FORMAT,schema_version=VERSION,input_kind='synthetic_fixture',
        processing_view=validate_view_manifest(view),extraction_configuration=configuration(profile),
        storage_integration='ORDERED_SYNTHETIC_ONLY',
        upstream_marker_policy='PRESERVE_D2G_D2H_NOT_ACTIVE_MARKERS_AS_PROVENANCE',
        label_policy='EXCLUSION_ONLY_NOT_GROUND_TRUTH',allowed_states=list(STATES),
        y_sentinel=-1,supervised_eligible=False,real_data_export='BLOCKED',native_ARM64='NOT_VERIFIED')
    bounded_json(value)
    return strict_json(canonical_bytes(value))

MANIFEST_SCHEMA=dict(**{'$schema':'https://json-schema.org/draft/2020-12/schema'},
    title='Ordered-HDF5 synthetic manifest; no approval for real PCAP or labels',
    **obj(dict(format=dict(const=FORMAT),schema_version=dict(const=VERSION),
        input_kind=dict(const='synthetic_fixture'),processing_view=VIEW_SCHEMA,
        extraction_configuration=CONFIG_SCHEMA,storage_integration=dict(const='ORDERED_SYNTHETIC_ONLY'),
        upstream_marker_policy=dict(const='PRESERVE_D2G_D2H_NOT_ACTIVE_MARKERS_AS_PROVENANCE'),
        label_policy=dict(const='EXCLUSION_ONLY_NOT_GROUND_TRUTH'),allowed_states=dict(const=list(STATES)),
        y_sentinel=dict(const=-1),supervised_eligible=dict(const=False),
        real_data_export=dict(const='BLOCKED'),native_ARM64=dict(const='NOT_VERIFIED'))))

def validate_manifest(value):
    require(type(value) is dict,'ORDERED_MANIFEST_REQUIRED')
    structural(value,MANIFEST_SCHEMA)
    validate_configuration(value['extraction_configuration'])
    expected=make_manifest(value['processing_view'],value['extraction_configuration']['profile'])
    require(canonical_bytes(value)==canonical_bytes(expected),'ORDERED_MANIFEST_MISMATCH')
    return expected

def sample_hash(tensor,state):
    require(type(state) is str and state in STATES,'SUPERVISED_OR_UNKNOWN_LABEL_POLICY')
    return sha(bounded_json(dict(metadata=tensor.metadata,tensor_sha256=sha(tensor.tensor_bytes),
                                label_state=state,y=-1,supervised_mask=False)))

def content_hash(manifest,events_digest,count,event_count):
    return sha(bounded_json(dict(manifest_sha256=sha(bounded_json(manifest)),
        events_sha256=events_digest,sample_count=count,event_count=event_count)))
