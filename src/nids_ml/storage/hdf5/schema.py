"""Define synthetic capture-order HDF5 types, manifests and runtime pins.
Định nghĩa types/manifest HDF5 capture-order synthetic và runtime pins.
"""
from dataclasses import dataclass, field
import hashlib
import sys

from nids_ml.flows.extractor import FlowTensor
from nids_ml.flows.features import CaptureContext, configuration
from nids_ml.labeling.contracts import canonical_bytes
from nids_ml.labeling.resolver import WindowDecision

FORMAT='nids-flow-hdf5'
VERSION='1.0.0'
MAX_SAMPLES=4096
MAX_FILE_BYTES=128*1048576
JSON_BYTES=16384
LABEL_BYTES=4096
CHUNK_ROWS=32
STATES=('VERIFIED_BENIGN','VERIFIED_DDOS','UNKNOWN','MIXED','OUT_OF_SCOPE','DATASET_NORMAL','DATASET_DDOS')
LABEL_POLICY={'id':'flow-label-core','version':'0.1.0','scope':'SYNTHETIC_TEST_ONLY'}
UNKNOWN_POLICY={'id':None,'version':None,'scope':'NOT_PROVIDED'}
ORDER_POLICY={'id':'capture-order','version':'1.0.0','scope':'SYNTHETIC_TEST_ONLY','processing_view':'NOT_ACTIVE'}
ROW_DTYPES={'X':'<f4','y':'i1','label_state':'S24','supervised_mask':'?',
            'dataset_proposed_y':'i1','metadata_json':'S16384','labels_json':'S4096','sample_sha256':'S64'}
ROOT_ATTRS={'format':'S32','schema_version':'S16','status':'S16','sample_count':'<u8','content_sha256':'S64'}


class ExportError(ValueError):
    """Artifact/label/ordering lỗi thì fail closed, không tự sửa dữ liệu."""


def digest(value):
    return hashlib.sha256(value).hexdigest()


def backend():
    if sys.version_info[:2]!=(3,11):
        raise RuntimeError('HDF5 validated runtime requires Python 3.11; no auto-install')
    try:
        import numpy as np
        import h5py
    except ImportError as error:
        raise RuntimeError('Install the declared hdf5 extra in an approved environment; no auto-install') from error
    if (np.__version__,h5py.__version__)!=('1.26.4','3.16.0'):
        raise RuntimeError('Unvalidated HDF5 runtime: expected NumPy 1.26.4 / h5py 3.16.0')
    return np,h5py


@dataclass(frozen=True)
class LabelRecord:
    state:str='UNKNOWN'
    y:int=-1
    supervised_mask:bool=False
    dataset_proposed_y:int=-1
    decision:WindowDecision|None=None
    target_metadata_sha256:str|None=None
    evidence_sha256:str|None=None


@dataclass(frozen=True)
class FlowSample:
    tensor:FlowTensor
    label:LabelRecord=field(default_factory=LabelRecord)
    extraction_receipt_sha256:str|None=None
    extraction_manifest_sha256:str|None=None
    processing_indices:tuple|None=None


def _manifest(context,profile,label_policy):
    cfg=configuration(profile)
    return dict(format=FORMAT,schema_version=VERSION,input_kind='synthetic_fixture',
        source_capture=context.to_dict(),synthetic_capture_sha256=context.capture_sha256,
        source_evidence_kind='LITERAL_SYNTHETIC_TEXT_SHA256_CHECKED',
        extraction_configuration=cfg,label_policy=label_policy,ordering_policy=dict(ORDER_POLICY),
        real_data_export='BLOCKED',label_approval='BLOCKED_FOR_REAL_DATA',
        ordering_approval='BLOCKED_FOR_REAL_DATA',native_ARM64='NOT_RUN',
        missing_evidence='null means NOT_PROVIDED; hashes are consistency, not authority proof',
        y_sentinel=-1,empty_policy='REJECT_EMPTY',processing_view='NOT_ACTIVE')


def make_manifest(context,*,synthetic_capture_text,profile='flow-cnn-n20-t10',
                  input_kind='synthetic_fixture',label_policy=None):
    # Không có option bật real mode hoặc tự đọc một đường dẫn PCAP/CSV.
    if input_kind!='synthetic_fixture':raise ExportError('REAL_DATA_EXPORT_BLOCKED')
    if not isinstance(context,CaptureContext):raise ExportError('CAPTURE_CONTEXT_REQUIRED')
    context=CaptureContext.from_dict(context.to_dict())
    if type(synthetic_capture_text) is not str or not 1<=len(synthetic_capture_text.encode('utf-8'))<=4096:
        raise ExportError('BOUNDED_SYNTHETIC_LITERAL_REQUIRED')
    if digest(synthetic_capture_text.encode('utf-8'))!=context.capture_sha256:
        raise ExportError('SYNTHETIC_CAPTURE_HASH_MISMATCH')
    policy=dict(UNKNOWN_POLICY) if label_policy is None else label_policy
    if canonical_bytes(policy) not in (canonical_bytes(UNKNOWN_POLICY),canonical_bytes(LABEL_POLICY)):
        raise ExportError('SYNTHETIC_LABEL_POLICY_REQUIRED')
    value=_manifest(context,profile,policy)
    # Snapshot tránh caller sửa dictionary policy sau khi tạo manifest.
    import json
    return json.loads(canonical_bytes(value))


def validate_manifest(value):
    if type(value) is not dict or value.get('input_kind')!='synthetic_fixture':
        raise ExportError('REAL_DATA_EXPORT_BLOCKED')
    try:
        context=CaptureContext.from_dict(value['source_capture'])
        profile=value['extraction_configuration']['profile']
        policy=value['label_policy']
        if canonical_bytes(policy) not in (canonical_bytes(UNKNOWN_POLICY),canonical_bytes(LABEL_POLICY)):
            raise ExportError('LABEL_POLICY_MISMATCH')
        expected=_manifest(context,profile,policy)
        if canonical_bytes(value)!=canonical_bytes(expected):raise ExportError('MANIFEST_SCHEMA_OR_POLICY_MISMATCH')
        if len(canonical_bytes(value))>JSON_BYTES:raise ExportError('MANIFEST_SIZE')
    except (KeyError,TypeError,ValueError) as error:
        if isinstance(error,ExportError):raise
        raise ExportError('MANIFEST_INVALID') from error
    return expected


def content_digest(manifest,count,sample_chain_digest):
    return digest(canonical_bytes(dict(manifest=manifest,sample_count=count,sample_chain_sha256=sample_chain_digest)))
