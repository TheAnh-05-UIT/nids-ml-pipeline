"""Extract synthetic ordered flows while preserving original packet identities.
Trích ordered flow synthetic, giữ nguyên định danh packet gốc.
"""
from dataclasses import dataclass
import hashlib
import json
import re
import struct

from nids_ml.flows.extractor import BASE_RESERVATION, RECORD, FlowExtractor, PacketReceipt, _FlowState
from nids_ml.flows.features import (CaptureContext, ExtractionError, ResourceLimitError,
    configuration as capture_configuration, extract_features)
from nids_ml.flows.ordered_contracts import (VERSION, configuration, validate_configuration,
    validate_tensor_bytes)
from nids_ml.labeling.contracts import canonical_bytes, strict_json
from nids_ml.ordering.contracts import packet_bytes, validate_view_manifest
from nids_ml.ordering.view import ProcessingRecord, _processing_digest

INDEX=struct.Struct('<Q')
LINEAGE_OVERHEAD_RESERVATION=2048


@dataclass(frozen=True)
class OrderedPacketReceipt:
    receipt: PacketReceipt
    processing_index: int
    processing_view_sha256: str

    def to_dict(self):
        value=self.receipt.to_dict()
        value.update(kind='ordered_packet_receipt',schema_version=VERSION,
            source_packet_ordinal=value.pop('ordinal'),source_packet_sha256=value.pop('packet_sha256'),
            processing_index=self.processing_index,processing_view_sha256=self.processing_view_sha256)
        return value


@dataclass(frozen=True)
class OrderedFlowTensor:
    shape: tuple
    tensor_bytes: bytes
    metadata_json: bytes

    @property
    def metadata(self):return json.loads(self.metadata_json)

    def to_dict(self):
        return dict(kind='ordered_flow_tensor',schema_version=VERSION,shape=list(self.shape),dtype='<f4',
                    tensor_sha256=hashlib.sha256(self.tensor_bytes).hexdigest(),metadata=self.metadata)


@dataclass(frozen=True)
class OrderedExtractionComplete:
    stats_json: bytes

    def to_dict(self):
        return dict(kind='ordered_stream_complete',schema_version=VERSION,status='COMPLETE_SYNTHETIC_ONLY',
                    stats=json.loads(self.stats_json),real_data_export='BLOCKED',HDF5_integration='NOT_ACTIVE')


class _LineageState:
    def __init__(self,reference,n):
        self.buffer=bytearray(n*INDEX.size);self.count=0
        self.first_index=self.last_index=reference.processing_index
        self.minimum=self.maximum=reference.source_packet_ordinal
        self.full=hashlib.sha256();self.overflow=hashlib.sha256()

    def add(self,reference,n):
        data=canonical_bytes(reference.to_dict())+b'\n'
        self.full.update(data)
        if self.count<n:INDEX.pack_into(self.buffer,self.count*INDEX.size,reference.processing_index)
        else:self.overflow.update(data)
        self.count+=1;self.last_index=reference.processing_index
        self.minimum=min(self.minimum,reference.source_packet_ordinal)
        self.maximum=max(self.maximum,reference.source_packet_ordinal)


class OrderedFlowExtractor(FlowExtractor):
    """Một completed view, một callback lookup; outputs chỉ commit sau ordered_stream_complete."""
    _validate_view=staticmethod(validate_view_manifest)
    _configuration=staticmethod(configuration)
    _validate_configuration=staticmethod(validate_configuration)
    _packet_bytes=staticmethod(packet_bytes)
    _extract_features=staticmethod(extract_features)
    _validate_tensor=staticmethod(validate_tensor_bytes)
    _digest_seed=staticmethod(_processing_digest)
    _receipt_event=OrderedPacketReceipt
    _tensor_event=OrderedFlowTensor
    _completion_event=OrderedExtractionComplete
    _input_kind='synthetic_fixture'

    def __init__(self,view,config=None,limits=None):
        self.view=self._validate_view(view)
        self.ordered_config=self._validate_configuration(self._configuration() if config is None else config)
        context=CaptureContext.from_dict(self.view['source']['capture'])
        # Config capture-order chỉ dùng nội bộ cho features/buffer/normalization; không gọi run cũ.
        super().__init__(context,capture_configuration(self.ordered_config['profile'],input_kind=self._input_kind),limits)
        if self.view['record_count']>self.limits.max_input_packets:
            raise ResourceLimitError('ORDERED_VIEW_EXCEEDS_INPUT_LIMIT')
        self.flow_reservation+=LINEAGE_OVERHEAD_RESERVATION+self.n*INDEX.size
        self._lineage={}
        self._stats['peak_lineage_buffer_bytes']=0

    @property
    def stats(self):
        return dict(super().stats,lineage_buffer_bytes=len(self._lineage)*self.n*INDEX.size,
                    processing_view_sha256=self.view['processing_order_sha256'],
                    extractor_version=self.ordered_config['extractor_version'])

    def _flush(self,anchor,index,complete):
        for tensor in super()._flush(anchor,index,complete):
            metadata=tensor.metadata
            lineage=self._lineage.pop(tuple(metadata['flow_tuple']))
            count=metadata['retained_packet_count']
            metadata.update(schema_version=self.ordered_config['schema_version'],extractor_version=self.ordered_config['extractor_version'],
                configuration_sha256=hashlib.sha256(canonical_bytes(self.ordered_config)).hexdigest(),
                input_kind=self._input_kind,ordering_contract_version=self.ordered_config['ordering_contract_version'],
                processing_view_sha256=self.view['processing_order_sha256'],packet_digest_order='PROCESSING_INDEX',
                first_processing_index=lineage.first_index,last_processing_index=lineage.last_index,
                retained_processing_indices=[INDEX.unpack_from(lineage.buffer,i*INDEX.size)[0] for i in range(count)],
                min_source_ordinal=lineage.minimum,max_source_ordinal=lineage.maximum,
                full_processing_digest=lineage.full.hexdigest(),overflow_processing_digest=lineage.overflow.hexdigest(),
                real_data_export='BLOCKED',HDF5_integration='NOT_ACTIVE')
            self._validate_tensor(tensor.shape,tensor.tensor_bytes,metadata,self.ordered_config,self.view)
            yield self._tensor_event(tensor.shape,tensor.tensor_bytes,canonical_bytes(metadata))

    def _check_reference(self,reference,index,previous):
        if type(reference) is not ProcessingRecord or type(reference.processing_index) is not int or reference.processing_index!=index:
            raise ExtractionError('PROCESSING_INDEX_GAP_DUPLICATE_OR_TYPE')
        source=self.view['source'];start=source['source_first_ordinal'];end=start+self.view['record_count']-1
        if (type(reference.source_packet_ordinal) is not int or not start<=reference.source_packet_ordinal<=end
                or type(reference.timestamp_ns) is not int or not 0<=reference.timestamp_ns<=2**63-1
                or reference.capture!=self.context or type(reference.source_packet_sha256) is not str
                or re.fullmatch('[a-f0-9]{64}',reference.source_packet_sha256) is None):
            raise ExtractionError('PROCESSING_REFERENCE_IDENTITY_OR_RANGE')
        key=(reference.timestamp_ns,reference.source_packet_ordinal)
        if previous is not None and key<=previous:
            raise ExtractionError('PROCESSING_TIMESTAMP_OR_TIE_ORDER_REGRESSION')
        return key

    def run(self,references,packet_lookup,*,input_complete):
        if self.status!='NEW':raise ExtractionError('ORDERED_EXTRACTOR_CANNOT_REPLAY')
        self.status='RUNNING';anchor=None;index=window_count=0;previous=None
        iterator=None;digest=self._digest_seed(self.view['source'])
        try:
            if input_complete is not True or not callable(packet_lookup):
                raise ExtractionError('COMPLETE_VIEW_AND_CALLABLE_LOOKUP_REQUIRED')
            iterator=iter(references)
            for reference in iterator:
                self._stats['input_packets']+=1
                number=self._stats['input_packets']
                if number>self.view['record_count'] or number>self.limits.max_input_packets:
                    raise ResourceLimitError('EXTRA_REFERENCE_OR_INPUT_LIMIT')
                previous=self._check_reference(reference,number,previous)
                # Snapshot metadata để callback không thể sửa object đã được kiểm hash sau đó.
                raw=strict_json(self._packet_bytes(packet_lookup(reference.source_packet_ordinal)))
                if (raw['ordinal']!=reference.source_packet_ordinal or raw['timestamp_ns']!=reference.timestamp_ns
                        or raw['capture']!=self.context.to_dict()
                        or hashlib.sha256(canonical_bytes(raw)).hexdigest()!=reference.source_packet_sha256):
                    raise ExtractionError('SOURCE_PACKET_HASH_CONTEXT_TIMESTAMP_OR_ORDINAL_MISMATCH')
                packet=self._extract_features(raw)
                digest.update(canonical_bytes(reference.to_dict())+b'\n')
                if anchor is None:
                    anchor=packet.timestamp_ns;self._stats['windows']=1
                elif packet.timestamp_ns>anchor+self.ordered_config['window_ns']:
                    yield from self._flush(anchor,index,True)
                    anchor=packet.timestamp_ns;index+=1;window_count=0;self._stats['windows']+=1
                if window_count>=self.limits.max_packets_per_window:
                    raise ResourceLimitError('ORDERED_FULL_WINDOW_PACKET_LIMIT')
                key=packet.canonical_tuple
                if key not in self._flows:
                    projected=BASE_RESERVATION+(len(self._flows)+1)*self.flow_reservation
                    if len(self._flows)>=self.limits.max_active_flows:
                        raise ResourceLimitError('ORDERED_ACTIVE_FLOW_LIMIT')
                    if projected>self.limits.memory_budget_bytes:
                        raise ResourceLimitError('ORDERED_ACCOUNTED_MEMORY_LIMIT')
                    self._flows[key]=_FlowState(packet,self.n)
                    self._lineage[key]=_LineageState(reference,self.n)
                    self._stats['peak_active_flows']=max(self._stats['peak_active_flows'],len(self._flows))
                    self._stats['peak_retained_buffer_bytes']=max(self._stats['peak_retained_buffer_bytes'],len(self._flows)*self.n*RECORD.size)
                    self._stats['peak_accounted_state_bytes']=max(self._stats['peak_accounted_state_bytes'],projected)
                    self._stats['peak_lineage_buffer_bytes']=max(self._stats['peak_lineage_buffer_bytes'],len(self._lineage)*self.n*INDEX.size)
                retained=self._flows[key].add(packet,self.n)
                self._lineage[key].add(reference,self.n)
                self._stats['overflow_packets']+=not retained
                self._stats['unknown_protocol_token_packets']+=bool(packet.unknown_protocol_tokens)
                self._stats['accepted_packets']+=1;self._stats['emitted_receipts']+=1
                window_count+=1
                receipt=PacketReceipt(packet.context,key,index,packet.ordinal,packet.timestamp_ns,
                                      packet.packet_sha256,retained,packet.unknown_protocol_tokens)
                yield self._receipt_event(receipt,reference.processing_index,self.view['processing_order_sha256'])
            if self._stats['input_packets']!=self.view['record_count']:
                raise ExtractionError('MISSING_PROCESSING_REFERENCE')
            if digest.hexdigest()!=self.view['processing_order_sha256']:
                raise ExtractionError('PROCESSING_VIEW_DIGEST_MISMATCH')
            if anchor is not None:yield from self._flush(anchor,index,True)
            self.status='COMPLETE'
            yield self._completion_event(canonical_bytes(self.stats))
        except GeneratorExit:
            if self.status=='RUNNING':self.status='ABORTED'
            raise
        except Exception as error:
            self.status='FAILED';self.failure=dict(type=type(error).__name__,message=str(error));raise
        finally:
            if self.status=='RUNNING':self.status='ABORTED'
            self._flows.clear();self._lineage.clear()
            if iterator is not None and hasattr(iterator,'close'):iterator.close()
