"""Build synthetic flow and label fixtures for HDF5 tests.
Tạo fixture flow và nhãn synthetic để test HDF5.
"""
import hashlib

from nids_ml.flows.extractor import FlowExtractor,FlowTensor
from nids_ml.flows.features import CaptureContext,configuration
from nids_ml.labeling.resolver import Label,PacketDecision,aggregate_window
from nids_ml.storage.hdf5 import FlowSample,LabelRecord,make_manifest
from nids_ml.storage.hdf5.schema import LABEL_POLICY
from flow_extraction_fixtures import CAPTURE_TEXT,context,packet

EVIDENCE=b'Synthetic HDF5 label assertion only; not approval for Edge-IIoTset.'
EVIDENCE_SHA=hashlib.sha256(EVIDENCE).hexdigest()


def flows(profile='flow-cnn-n20-t10',count=7,packets_per_flow=None):
    raw=[];ordinal=0
    for flow in range(count):
        for _ in range(flow+1 if packets_per_flow is None else packets_per_flow):
            ordinal+=1
            raw.append(packet(flow=flow,ordinal=ordinal,timestamp_ns=ordinal*1000))
    core=FlowExtractor(CaptureContext.from_dict(context()),configuration(profile))
    return [event for event in core.run(raw,input_complete=True) if isinstance(event,FlowTensor)]


def manifest(profile='flow-cnn-n20-t10',verified=False):
    return make_manifest(CaptureContext.from_dict(context()),synthetic_capture_text=CAPTURE_TEXT,
                         profile=profile,label_policy=LABEL_POLICY if verified else None)


def verified_label(flow,target):
    meta=flow.metadata
    # Fixture generator tạo các packets liên tiếp, không áp dụng giả thiết này cho PCAP thật.
    decisions=[PacketDecision(Label(target),('SYNTHETIC_FIXTURE_ASSERTION',),
        identity=(meta['capture']['capture_sha256'],tuple(meta['flow_tuple'])),
        ordinal=meta['first_ordinal']+index,timestamp_ns=meta['first_timestamp_ns']+index*1000)
        for index in range(meta['real_packet_count'])]
    decision=aggregate_window(decisions,full_window_complete=True)
    return LabelRecord('VERIFIED_DDOS' if target else 'VERIFIED_BENIGN',target,True,
        decision=decision,target_metadata_sha256=hashlib.sha256(flow.metadata_json).hexdigest(),evidence_sha256=EVIDENCE_SHA)


def mixed_samples():
    items=flows()
    labels=[verified_label(items[0],0),verified_label(items[1],1),LabelRecord(),LabelRecord('MIXED'),
            LabelRecord('OUT_OF_SCOPE'),LabelRecord('DATASET_NORMAL',dataset_proposed_y=0),
            LabelRecord('DATASET_DDOS',dataset_proposed_y=1)]
    return [FlowSample(flow,label) for flow,label in zip(items,labels)]
