"""Build bounded flow tensors while preserving full-window packet receipts.
Tạo flow tensor có giới hạn, giữ receipts của full window.
"""
from dataclasses import dataclass
import hashlib
import json
import struct

from nids_ml.flows.features import (CaptureContext,ExtractionError,ResourceLimitError,configuration,
                            extract_features,normalize_row,validate_configuration)
from nids_ml.labeling.contracts import canonical_bytes

RECORD=struct.Struct('<12Q')  # timestamp ns, ordinal, rồi 10 numeric feature values.
BASE_RESERVATION=8192
FLOW_OVERHEAD_RESERVATION=4096


@dataclass(frozen=True)
class ResourceLimits:
    max_active_flows:int=32
    max_packets_per_window:int=100000
    max_input_packets:int=1000000
    memory_budget_bytes:int=1048576

    def validate(self,*,decoded_pcap=False):
        if type(decoded_pcap) is not bool:raise ExtractionError('Explicit decoded scope required')
        for name,maximum in [('max_active_flows',256 if decoded_pcap else 128),('max_packets_per_window',1000000),
                             ('max_input_packets',1000000),('memory_budget_bytes',67108864)]:
            value=getattr(self,name)
            if type(value) is not int or not 1<=value<=maximum:raise ExtractionError('Invalid resource limit: '+name)
        if self.memory_budget_bytes<BASE_RESERVATION:raise ExtractionError('Memory budget below base reservation')
        return self


@dataclass(frozen=True)
class PacketReceipt:
    context:CaptureContext
    flow_tuple:tuple
    window_index:int
    ordinal:int
    timestamp_ns:int
    packet_sha256:str
    retained:bool
    unknown_protocol_tokens:tuple

    def to_dict(self):
        return dict(kind='packet_receipt',capture=self.context.to_dict(),flow_tuple=list(self.flow_tuple),
            window_index=self.window_index,ordinal=self.ordinal,timestamp_ns=self.timestamp_ns,
            packet_sha256=self.packet_sha256,retained=self.retained,
            unknown_protocol_tokens=list(self.unknown_protocol_tokens),label_status='UNKNOWN')


@dataclass(frozen=True)
class FlowTensor:
    shape:tuple
    tensor_bytes:bytes
    metadata_json:bytes

    @property
    def metadata(self):return json.loads(self.metadata_json)

    def to_dict(self):
        return dict(kind='flow_tensor',shape=list(self.shape),dtype='<f4',
                    tensor_sha256=hashlib.sha256(self.tensor_bytes).hexdigest(),metadata=self.metadata)

    def to_numpy(self,*,batch=True):
        if type(batch) is not bool:raise ExtractionError('Expected boolean batch flag')
        try:import numpy as np
        except ImportError as error:raise RuntimeError('NumPy required only for tensor view; no auto-install') from error
        array=np.frombuffer(self.tensor_bytes,dtype='<f4').reshape(self.shape)
        return array.reshape((1,*self.shape)).copy() if batch else array.copy()


@dataclass(frozen=True)
class ExtractionComplete:
    status:str
    stats_json:bytes

    def to_dict(self):return dict(kind='stream_complete',status=self.status,stats=json.loads(self.stats_json))


class _FlowState:
    def __init__(self,packet,n):
        self.buffer=bytearray(n*RECORD.size)
        self.first_tuple=packet.forward_tuple
        self.first_timestamp=packet.timestamp_ns
        self.first_ordinal=packet.ordinal
        self.last_timestamp=packet.timestamp_ns
        self.last_ordinal=packet.ordinal
        self.count=0
        self.forward_count=0
        self.reverse_count=0
        self.full_digest=hashlib.sha256()
        self.overflow_digest=hashlib.sha256()
        self.icmp_identifier=packet.icmp_identifier
        self.icmp_identifier_changed=False

    def add(self,packet,n):
        retained=self.count<n
        self.full_digest.update(bytes.fromhex(packet.packet_sha256))
        if retained:
            RECORD.pack_into(self.buffer,self.count*RECORD.size,packet.timestamp_ns,packet.ordinal,*packet.features[1:])
        else:self.overflow_digest.update(bytes.fromhex(packet.packet_sha256))
        self.count+=1
        self.last_timestamp=packet.timestamp_ns;self.last_ordinal=packet.ordinal
        if packet.forward_tuple==self.first_tuple:self.forward_count+=1
        else:self.reverse_count+=1
        self.icmp_identifier_changed |= packet.icmp_identifier!=self.icmp_identifier
        return retained


class FlowExtractor:
    """Một capture/run/segment trên một instance; caller phải drain events, không queue toàn dataset."""

    def __init__(self,context,config=None,limits=None):
        if not isinstance(context,CaptureContext):raise ExtractionError('Expected CaptureContext')
        self.context=CaptureContext.from_dict(context.to_dict())
        self.config=validate_configuration(configuration() if config is None else config)
        if limits is not None and not isinstance(limits,ResourceLimits):raise ExtractionError('Expected ResourceLimits')
        self.limits=(ResourceLimits() if limits is None else limits).validate(decoded_pcap=self.config['input_kind']=='decoded_pcap')
        self.n=self.config['tensor_shape'][0]
        self.flow_reservation=FLOW_OVERHEAD_RESERVATION+self.n*RECORD.size
        self._flows={}
        self.status='NEW'
        self.failure=None
        self._stats=dict(input_packets=0,accepted_packets=0,emitted_tensors=0,emitted_receipts=0,
            windows=0,overflow_packets=0,unknown_protocol_token_packets=0,peak_active_flows=0,
            peak_retained_buffer_bytes=0,peak_accounted_state_bytes=BASE_RESERVATION)

    @property
    def stats(self):
        return dict(self._stats,status=self.status,active_flows=len(self._flows),
                    retained_buffer_bytes=len(self._flows)*self.n*RECORD.size,
                    accounted_state_bytes=BASE_RESERVATION+len(self._flows)*self.flow_reservation)

    def _flush(self,anchor,index,complete):
        # Dict insertion order = first-observed flow order; không sort timestamps.
        while self._flows:
            key=next(iter(self._flows));state=self._flows.pop(key)
            count=min(state.count,self.n);output=bytearray(self.n*11*4);ordinals=[]
            for i in range(count):
                timestamp,ordinal,*values=RECORD.unpack_from(state.buffer,i*RECORD.size)
                row=normalize_row((timestamp/1e9,*values),timestamp-state.first_timestamp)
                struct.pack_into('<11f',output,i*44,*row)
                ordinals.append(ordinal)
            overflow=state.count-count
            metadata=dict(schema_version='1.0.0',extractor_version=self.config['extractor_version'],
                feature_schema_version=self.config['feature_schema_version'],profile=self.config['profile'],
                normalization_version=self.config['normalization_version'],configuration_sha256=hashlib.sha256(canonical_bytes(self.config)).hexdigest(),
                capture=self.context.to_dict(),flow_tuple=list(key),first_observed_tuple=list(state.first_tuple),
                window_index=index,anchor_timestamp_ns=anchor,nominal_end_timestamp_ns=anchor+10000000000,
                window_boundary='STRICT_GREATER_THAN_ANCHOR_PLUS_T',full_window_complete=complete,
                first_ordinal=state.first_ordinal,last_ordinal=state.last_ordinal,
                first_timestamp_ns=state.first_timestamp,last_timestamp_ns=state.last_timestamp,
                real_packet_count=state.count,retained_packet_count=count,overflow_packet_count=overflow,
                retained_ordinals=ordinals,real_packet_mask=[i<count for i in range(self.n)],
                forward_packet_count=state.forward_count,reverse_packet_count=state.reverse_count,
                full_packet_digest=state.full_digest.hexdigest(),overflow_packet_digest=state.overflow_digest.hexdigest(),
                icmp_identifier_changed=state.icmp_identifier_changed,
                full_window_label='UNKNOWN',retained_fragment_label='UNKNOWN',supervised_eligible=False,
                label_coverage='NOT_EVALUATED',overflow_metadata='COUNTS_BOUNDS_DIGEST_PLUS_STREAMED_RECEIPTS',
                stream_completion_required=True)
            if self.config['input_kind'] == 'decoded_pcap':
                metadata.update(schema_version='1.1.0', input_kind='decoded_pcap',
                                decoder_contract_version=self.config['decoder_contract_version'])
            self._stats['emitted_tensors']+=1
            yield FlowTensor((self.n,11,1),bytes(output),canonical_bytes(metadata))

    def run(self,packets,*,input_complete):
        if self.status!='NEW':raise ExtractionError('Extractor instances cannot resume/replay')
        if type(input_complete) is not bool:raise ExtractionError('Explicit input completeness required')
        self.status='RUNNING';anchor=None;index=0;window_count=0;previous=None
        try:
            for raw in packets:
                if self._stats['input_packets']>=self.limits.max_input_packets:
                    raise ResourceLimitError('Input packet limit exceeded')
                self._stats['input_packets']+=1
                packet=extract_features(raw, input_kind=self.config['input_kind'])
                if packet.context!=self.context:raise ExtractionError('Capture/run/segment mismatch')
                if previous is not None:
                    if packet.ordinal<=previous[0]:raise ExtractionError('Packet ordinal duplicate/regression')
                    if packet.timestamp_ns<previous[1]:raise ExtractionError('Timestamp regression; no implicit sort')
                if anchor is None:
                    anchor=packet.timestamp_ns;self._stats['windows']=1
                elif packet.timestamp_ns>anchor+10000000000:
                    yield from self._flush(anchor,index,True)
                    anchor=packet.timestamp_ns;index+=1;window_count=0;self._stats['windows']+=1
                if window_count>=self.limits.max_packets_per_window:
                    raise ResourceLimitError('Full window packet limit exceeded')
                key=packet.canonical_tuple
                if key not in self._flows:
                    projected=BASE_RESERVATION+(len(self._flows)+1)*self.flow_reservation
                    if len(self._flows)>=self.limits.max_active_flows:raise ResourceLimitError('Active flow limit exceeded')
                    if projected>self.limits.memory_budget_bytes:raise ResourceLimitError('Accounted state memory budget exceeded')
                    self._flows[key]=_FlowState(packet,self.n)
                    self._stats['peak_active_flows']=max(self._stats['peak_active_flows'],len(self._flows))
                    self._stats['peak_retained_buffer_bytes']=max(self._stats['peak_retained_buffer_bytes'],len(self._flows)*self.n*RECORD.size)
                    self._stats['peak_accounted_state_bytes']=max(self._stats['peak_accounted_state_bytes'],projected)
                retained=self._flows[key].add(packet,self.n)
                self._stats['overflow_packets']+=not retained
                self._stats['unknown_protocol_token_packets']+=bool(packet.unknown_protocol_tokens)
                self._stats['accepted_packets']+=1;self._stats['emitted_receipts']+=1
                window_count+=1;previous=(packet.ordinal,packet.timestamp_ns)
                yield PacketReceipt(packet.context,key,index,packet.ordinal,packet.timestamp_ns,
                                    packet.packet_sha256,retained,packet.unknown_protocol_tokens)
            if anchor is not None:yield from self._flush(anchor,index,input_complete)
            self.status='COMPLETE' if input_complete else 'PARTIAL'
            yield ExtractionComplete(self.status,canonical_bytes(self.stats))
        except GeneratorExit:
            if self.status=='RUNNING':self.status='ABORTED'
            raise
        except Exception as error:
            self.status='FAILED';self.failure=dict(type=type(error).__name__,message=str(error));raise
        finally:
            if self.status=='RUNNING':self.status='ABORTED'
            self._flows.clear()
