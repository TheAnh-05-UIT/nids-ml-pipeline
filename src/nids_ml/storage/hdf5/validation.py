"""Check tensor, metadata and label safety separately from HDF5 I/O.
Kiểm tensor, metadata và an toàn nhãn, tách khỏi HDF5 I/O.
"""
import ipaddress
import json
import re

from nids_ml.flows.extractor import FlowTensor
from nids_ml.labeling.contracts import canonical_bytes,strict_json
from nids_ml.labeling.resolver import Label,WindowDecision
from .schema import (ExportError,FlowSample,LabelRecord,JSON_BYTES,LABEL_BYTES,STATES,
                     LABEL_POLICY,backend,digest)

FLOW_FIELDS=set(('schema_version extractor_version feature_schema_version profile normalization_version '
    'configuration_sha256 capture flow_tuple first_observed_tuple window_index anchor_timestamp_ns '
    'nominal_end_timestamp_ns window_boundary full_window_complete first_ordinal last_ordinal '
    'first_timestamp_ns last_timestamp_ns real_packet_count retained_packet_count overflow_packet_count '
    'retained_ordinals real_packet_mask forward_packet_count reverse_packet_count full_packet_digest '
    'overflow_packet_digest icmp_identifier_changed full_window_label retained_fragment_label '
    'supervised_eligible label_coverage overflow_metadata stream_completion_required').split())
META_FIELDS={'flow','flow_id','processing_indices','processing_view','extraction_receipt_sha256','extraction_manifest_sha256'}
LABEL_FIELDS={'state','y','supervised_mask','dataset_proposed_y','decision','target_metadata_sha256','evidence_sha256'}
DECISION_FIELDS={'full_window_label','retained_fragment_label','status','reasons','real_packet_count',
                 'retained_packet_count','padding_count','full_window_complete'}
TEST_NETS=tuple(ipaddress.ip_network(x) for x in ('192.0.2.0/24','198.51.100.0/24','203.0.113.0/24'))


def require(condition,reason):
    if not condition:raise ExportError(reason)


def sha_or_none(value):
    require(value is None or type(value) is str and re.fullmatch('[a-f0-9]{64}',value) is not None,'INVALID_LINEAGE_SHA')


def read_json(raw,limit):
    require(type(raw) is bytes and len(raw)<=limit,'JSON_SIZE_OR_TYPE')
    try:value=strict_json(raw)
    except (ValueError,TypeError) as error:raise ExportError('INVALID_JSON') from error
    require(canonical_bytes(value)==raw,'NONCANONICAL_JSON')
    return value


def integer(value,minimum=0,maximum=2**63-1):
    require(type(value) is int and minimum<=value<=maximum,'INVALID_INTEGER_METADATA')


def flow_id(meta):
    return digest(canonical_bytes(dict(capture=meta['capture'],flow_tuple=meta['flow_tuple'],
        window_index=meta['window_index'],anchor_timestamp_ns=meta['anchor_timestamp_ns'])))


def validate_flow(tensor,manifest):
    require(isinstance(tensor,FlowTensor),'FLOW_TENSOR_REQUIRED')
    cfg=manifest['extraction_configuration'];n=cfg['tensor_shape'][0]
    require(type(tensor.shape) is tuple and all(type(x) is int for x in tensor.shape)
            and tensor.shape==(n,11,1),'TENSOR_SHAPE')
    require(type(tensor.tensor_bytes) is bytes and len(tensor.tensor_bytes)==n*11*4,'TENSOR_BYTES_OR_DTYPE')
    meta=read_json(tensor.metadata_json,JSON_BYTES)
    require(type(meta) is dict and set(meta)==FLOW_FIELDS,'D1_METADATA_FIELDS_OR_REAL_DATA_BLOCKED')
    require(meta['schema_version']=='1.0.0','D1_METADATA_VERSION_REQUIRED')
    require(meta['capture']==manifest['source_capture'],'CAPTURE_IDENTITY_MISMATCH')
    for name in ('extractor_version','feature_schema_version','profile','normalization_version','window_boundary'):
        require(meta[name]==cfg[name],'FEATURE_OR_EXTRACTOR_METADATA_MISMATCH')
    require(meta['configuration_sha256']==digest(canonical_bytes(cfg)),'CONFIGURATION_SHA_MISMATCH')
    for name in ('full_window_complete','icmp_identifier_changed','supervised_eligible','stream_completion_required'):
        require(type(meta[name]) is bool,'METADATA_BOOLEAN_REQUIRED')
    require(meta['full_window_label']=='UNKNOWN' and meta['retained_fragment_label']=='UNKNOWN'
            and meta['supervised_eligible'] is False and meta['label_coverage']=='NOT_EVALUATED',
            'D1_UNKNOWN_ONLY_METADATA_REQUIRED')
    require(meta['stream_completion_required'] is True and
            meta['overflow_metadata']=='COUNTS_BOUNDS_DIGEST_PLUS_STREAMED_RECEIPTS','D1_RECEIPT_CONTRACT')
    for name in ('window_index','anchor_timestamp_ns','nominal_end_timestamp_ns','first_timestamp_ns','last_timestamp_ns'):
        integer(meta[name])
    for name in ('first_ordinal','last_ordinal'):integer(meta[name],1,2**64-1)
    for name in ('real_packet_count','retained_packet_count','overflow_packet_count','forward_packet_count','reverse_packet_count'):
        integer(meta[name],0,1000000)
    real=meta['real_packet_count'];count=meta['retained_packet_count']
    require(real>=1 and count==min(n,real) and meta['overflow_packet_count']==real-count,'PACKET_COUNTS')
    require(meta['forward_packet_count']+meta['reverse_packet_count']==real,'DIRECTION_COUNTS')
    require(meta['nominal_end_timestamp_ns']==meta['anchor_timestamp_ns']+cfg['window_ns'],'WINDOW_DURATION')
    require(meta['anchor_timestamp_ns']<=meta['first_timestamp_ns']<=meta['last_timestamp_ns']<=meta['nominal_end_timestamp_ns'],
            'TIMESTAMP_REGRESSION_OR_WINDOW_BOUNDS')
    ordinals=meta['retained_ordinals']
    require(type(ordinals) is list and len(ordinals)==count,'RETAINED_ORDINALS')
    for value in ordinals:integer(value,1,2**64-1)
    require(all(a<b for a,b in zip(ordinals,ordinals[1:])) and ordinals[0]==meta['first_ordinal']
            and ordinals[-1]<=meta['last_ordinal'],'SOURCE_ORDINAL_REGRESSION')
    if real==count:require(ordinals[-1]==meta['last_ordinal'],'SOURCE_ORDINAL_BOUNDS')
    mask=meta['real_packet_mask']
    require(type(mask) is list and all(type(x) is bool for x in mask)
            and mask==[i<count for i in range(n)],'PADDING_MASK')
    for name in ('full_packet_digest','overflow_packet_digest'):
        sha_or_none(meta[name]);require(meta[name] is not None,'MISSING_PACKET_DIGEST')
    if not meta['overflow_packet_count']:require(meta['overflow_packet_digest']==digest(b''),'EMPTY_OVERFLOW_DIGEST')
    for name in ('flow_tuple','first_observed_tuple'):
        item=meta[name]
        require(type(item) is list and len(item)==5,'FIVE_TUPLE')
        for position in (0,2):
            try:address=ipaddress.ip_address(item[position])
            except (TypeError,ValueError) as error:raise ExportError('SYNTHETIC_ENDPOINT_REQUIRED') from error
            require(address.version==4 and any(address in network for network in TEST_NETS),'SYNTHETIC_TEST_NET_ONLY')
        for position in (1,3):integer(item[position],0,65535)
        require(type(item[4]) is int and item[4] in (1,6,17),'FLOW_PROTOCOL')
        if item[4]==1:require(item[1]==item[3]==0,'ICMP_PORTS')
    first=meta['first_observed_tuple'];ends=sorted(((first[0],first[1]),(first[2],first[3])))
    require(meta['flow_tuple']==[ends[0][0],ends[0][1],ends[1][0],ends[1][1],first[4]],'CANONICAL_FLOW_IDENTITY')
    np,_=backend();array=np.frombuffer(tensor.tensor_bytes,dtype='<f4').reshape(tensor.shape)
    require(bool(np.isfinite(array).all()) and bool(((array>=0)&(array<=1)).all()),'NONFINITE_OR_NONNORMALIZED_TENSOR')
    require(bool((array[count:]==0).all()),'NONZERO_PADDING')
    times=array[:count,0,0]
    require(times[0]==0 and bool((times[1:]>=times[:-1]).all()),'TENSOR_TIME_REGRESSION')
    return array,meta


def decision_dict(decision):
    if decision is None:return None
    require(isinstance(decision,WindowDecision),'B2A_WINDOW_DECISION_REQUIRED')
    return decision.to_dict()


def label_dict(label):
    require(isinstance(label,LabelRecord),'LABEL_RECORD_REQUIRED')
    return dict(state=label.state,y=label.y,supervised_mask=label.supervised_mask,
        dataset_proposed_y=label.dataset_proposed_y,decision=decision_dict(label.decision),
        target_metadata_sha256=label.target_metadata_sha256,evidence_sha256=label.evidence_sha256)


def label_from_dict(value):
    require(type(value) is dict and set(value)==LABEL_FIELDS,'LABEL_FIELDS')
    data=dict(value);raw=data.pop('decision')
    if raw is not None:
        require(type(raw) is dict and set(raw)==DECISION_FIELDS,'DECISION_FIELDS')
        for name in ('full_window_label','retained_fragment_label'):
            label=raw[name]
            require(label is None or type(label) is int and label in (0,1) or type(label) is str and label in ('UNKNOWN','MIXED'),
                    'DECISION_LABEL')
        require(type(raw['reasons']) is list and all(type(x) is str and len(x)<=128 for x in raw['reasons']), 'DECISION_REASONS')
        require(len(raw['reasons'])<=16 and raw['status'] in ('ACCEPT','QUARANTINE','REJECT_EMPTY'),'DECISION_STATUS')
        for name in ('real_packet_count','retained_packet_count','padding_count'):integer(raw[name],0,100000)
        require(type(raw['full_window_complete']) is bool,'DECISION_COMPLETENESS')
        raw={**raw,'full_window_label':Label(raw['full_window_label']) if raw['full_window_label'] is not None else None,
             'retained_fragment_label':Label(raw['retained_fragment_label']) if raw['retained_fragment_label'] is not None else None,
             'reasons':tuple(raw['reasons'])}
        raw=WindowDecision(**raw)
    return LabelRecord(**data,decision=raw)


def validate_label(label,tensor,meta,manifest):
    value=label_dict(label)
    # Writer cũng qua parser để không nhận bool như target integer hoặc object lạ.
    parsed=label_from_dict(value)
    require(type(label.state) is str and label.state in STATES,'LABEL_STATE')
    require(type(label.y) is int and label.y in (-1,0,1),'Y_TARGET')
    require(type(label.dataset_proposed_y) is int and label.dataset_proposed_y in (-1,0,1),'DATASET_PROPOSED_TARGET')
    require(type(label.supervised_mask) is bool,'SUPERVISED_BOOLEAN')
    for item in (label.target_metadata_sha256,label.evidence_sha256):sha_or_none(item)
    verified=label.state in ('VERIFIED_BENIGN','VERIFIED_DDOS')
    if verified:
        require(label.supervised_mask is True and label.y==int(label.state=='VERIFIED_DDOS'),'VERIFIED_TARGET_MASK')
        require(manifest['label_policy']==LABEL_POLICY,'VERIFIED_SYNTHETIC_POLICY_REQUIRED')
        require(label.dataset_proposed_y==-1,'VERIFIED_IS_NOT_DATASET_PROPOSAL')
        require(label.target_metadata_sha256==digest(tensor.metadata_json) and label.evidence_sha256 is not None,'VERIFIED_EVIDENCE_BINDING')
        require(parsed.decision is not None,'VERIFIED_DECISION_MISSING')
        try:target=parsed.decision.numeric_label()
        except ValueError as error:raise ExportError('B2A_DECISION_NOT_ELIGIBLE') from error
        require(target==label.y and parsed.decision.real_packet_count==meta['real_packet_count']
            and parsed.decision.retained_packet_count==meta['retained_packet_count']
            and parsed.decision.full_window_complete is meta['full_window_complete']
            and parsed.decision.padding_count in (0,len(meta['real_packet_mask'])-meta['retained_packet_count']),
            'DECISION_FLOW_COUNTS_OR_COMPLETENESS')
    else:
        require(label.supervised_mask is False and label.y==-1,'UNVERIFIED_MUST_BE_MASKED')
        if parsed.decision is not None:
            require(label.state in ('UNKNOWN','MIXED') and parsed.decision.full_window_label is Label(label.state),
                    'UNVERIFIED_DECISION_STATE')
        if label.state=='DATASET_NORMAL':require(label.dataset_proposed_y in (-1,0),'NORMAL_PROPOSAL')
        if label.state=='DATASET_DDOS':require(label.dataset_proposed_y in (-1,1),'DDOS_PROPOSAL')
    require(len(canonical_bytes(value))<=LABEL_BYTES,'LABEL_JSON_SIZE')
    return value


def validate_sample(sample,manifest):
    require(isinstance(sample,FlowSample),'FLOW_SAMPLE_REQUIRED')
    require(sample.processing_indices is None,'PROCESSING_VIEW_NOT_ACTIVE')
    array,meta=validate_flow(sample.tensor,manifest)
    for value in (sample.extraction_receipt_sha256,sample.extraction_manifest_sha256):sha_or_none(value)
    labels=validate_label(sample.label,sample.tensor,meta,manifest)
    metadata=dict(flow=meta,flow_id=flow_id(meta),processing_indices=None,processing_view='NOT_ACTIVE',
                  extraction_receipt_sha256=sample.extraction_receipt_sha256,
                  extraction_manifest_sha256=sample.extraction_manifest_sha256)
    raw=canonical_bytes(metadata);require(len(raw)<=JSON_BYTES,'METADATA_JSON_SIZE')
    sample_hash=digest(canonical_bytes(dict(tensor_sha256=digest(sample.tensor.tensor_bytes),
        metadata_sha256=digest(raw),labels=labels)))
    return array,metadata,labels,sample_hash


def sample_from_storage(array,metadata,labels):
    require(type(metadata) is dict and set(metadata)==META_FIELDS,'METADATA_ENVELOPE_FIELDS')
    require(metadata['processing_indices'] is None and metadata['processing_view']=='NOT_ACTIVE','PROCESSING_VIEW_NOT_ACTIVE')
    require(metadata['flow_id']==flow_id(metadata['flow']),'FLOW_ID_MISMATCH')
    return FlowSample(FlowTensor(tuple(array.shape),array.tobytes(order='C'),canonical_bytes(metadata['flow'])),
        label_from_dict(labels),metadata['extraction_receipt_sha256'],metadata['extraction_manifest_sha256'])
