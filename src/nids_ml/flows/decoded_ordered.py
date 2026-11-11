"""Reuse ordered flow windows for verified decoded-metadata spools; labels stay unknown.
Dùng lại ordered-flow core cho spool metadata đã kiểm; nhãn vẫn UNKNOWN, chưa HDF5.
"""
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import hashlib
import json

from nids_ml.flows.features import ExtractionError,extract_features,configuration as capture_configuration
from nids_ml.flows.ordered import OrderedFlowExtractor
from nids_ml.flows.ordered_contracts import (METADATA_SCHEMA,validate_metadata_invariants,validate_tensor_layout)
from nids_ml.labeling.contracts import canonical_bytes,strict_json,structural
from nids_ml.ordering.decoded import decoded_packet_bytes,processing_digest,validate_flow_view

VERSION='3.0.0'


def configuration(profile='flow-cnn-n20-t10'):
    value=capture_configuration(profile,input_kind='decoded_pcap')
    value.update(schema_version=VERSION,extractor_version='lucid-ordered-decoded-flow/3.0.0',
        mode='ordered_decoded_eligible_ipv4_v1',packet_order='VERIFIED_DECODED_PROCESSING_INDEX',
        ordering_contract_version='decoded-processing-view/1.0.0',
        source_ordinal_policy='PRESERVE_ORIGINAL_NONMONOTONIC',
        regression_policy='REJECT_PROCESSING_KEY_OR_INDEX_REGRESSION',
        unsupported_policy='FAIL_CLOSED_ON_ANY_REJECTED_RECORD',
        hardening='sequential source decode; verify spool, raw binding and processing digest',
        real_data_export='BLOCKED',HDF5_integration='NOT_ACTIVE')
    return value


def validate_configuration(value):
    if type(value) is not dict or canonical_bytes(value)!=canonical_bytes(configuration(value.get('profile'))):
        raise ExtractionError('DECODED_ORDERED_CONFIGURATION_MISMATCH')
    return strict_json(canonical_bytes(value))


def validate_tensor_bytes(shape,data,metadata,config,view):
    schema=deepcopy(METADATA_SCHEMA)
    schema['title']='Ordered decoded-PCAP metadata v3; UNKNOWN-only diagnostics'
    for key,value in dict(schema_version=VERSION,extractor_version='lucid-ordered-decoded-flow/3.0.0',
            input_kind='decoded_pcap',ordering_contract_version='decoded-processing-view/1.0.0').items():
        schema['properties'][key]['const']=value
    schema['properties']['decoder_contract_version']=dict(type='string',const='pcap-decoder/1.0.0')
    schema['required'].append('decoder_contract_version')
    structural(metadata,schema);config=validate_configuration(config);validate_flow_view(view)
    validate_metadata_invariants(metadata,config,view)
    validate_tensor_layout(shape,data,metadata,config)


@dataclass(frozen=True)
class DecodedOrderedReceipt:
    receipt:object
    processing_index:int
    processing_view_sha256:str

    def to_dict(self):
        value=self.receipt.to_dict()
        value.update(kind='decoded_ordered_packet_receipt',schema_version=VERSION,
            source_packet_ordinal=value.pop('ordinal'),source_packet_sha256=value.pop('packet_sha256'),
            processing_index=self.processing_index,processing_view_sha256=self.processing_view_sha256,
            input_kind='decoded_pcap')
        return value


@dataclass(frozen=True)
class DecodedOrderedTensor:
    shape:tuple
    tensor_bytes:bytes
    metadata_json:bytes

    @property
    def metadata(self):return json.loads(self.metadata_json)

    def to_dict(self):
        return dict(kind='decoded_ordered_flow_tensor',schema_version=VERSION,shape=list(self.shape),dtype='<f4',
                    tensor_sha256=hashlib.sha256(self.tensor_bytes).hexdigest(),metadata=self.metadata)


@dataclass(frozen=True)
class DecodedExtractionComplete:
    stats_json:bytes

    def to_dict(self):
        return dict(kind='decoded_ordered_stream_complete',schema_version=VERSION,
            status='COMPLETE_UNKNOWN_DECODED_FLOW',stats=json.loads(self.stats_json),
            labels='UNKNOWN_ONLY',HDF5_integration='NOT_ACTIVE',training=False)


class DecodedOrderedFlowExtractor(OrderedFlowExtractor):
    """A distinct scope and event format; old synthetic/HDF5 contracts stay closed.
    Scope/event format riêng; không mở gate synthetic hoặc HDF5 cũ.
    """
    _validate_view=staticmethod(validate_flow_view)
    _configuration=staticmethod(configuration)
    _validate_configuration=staticmethod(validate_configuration)
    _packet_bytes=staticmethod(decoded_packet_bytes)
    _extract_features=staticmethod(partial(extract_features,input_kind='decoded_pcap'))
    _validate_tensor=staticmethod(validate_tensor_bytes)
    _digest_seed=staticmethod(processing_digest)
    _receipt_event=DecodedOrderedReceipt
    _tensor_event=DecodedOrderedTensor
    _completion_event=DecodedExtractionComplete
    _input_kind='decoded_pcap'
