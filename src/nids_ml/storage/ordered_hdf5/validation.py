"""Match complete receipts, tensors and completion statistics before publication.
Đối chiếu đầy đủ receipts, tensor và completion stats trước công bố.
"""
import hashlib
import ipaddress
import re

from nids_ml.flows.ordered import OrderedFlowTensor
from nids_ml.flows.ordered_contracts import validate_tensor_bytes
from nids_ml.labeling.contracts import canonical_bytes
from .contracts import (MAX_RECEIPTS, MAX_SAMPLES, bounded_json, require,
                        sample_hash, validate_manifest)

RECEIPT_KEYS=set(('kind capture flow_tuple window_index timestamp_ns retained unknown_protocol_tokens '
    'label_status schema_version source_packet_ordinal source_packet_sha256 processing_index processing_view_sha256').split())
COMPLETION_KEYS={'kind','schema_version','status','stats','real_data_export','HDF5_integration'}

class Ledger:
    """Tối đa 128 flows đang chờ, 10k source hashes, 4096 flow IDs; không giữ mọi packet."""
    def __init__(self,manifest):
        self.manifest=validate_manifest(manifest)
        self.view=self.manifest['processing_view'];self.config=self.manifest['extraction_configuration']
        require(self.view['record_count']<=MAX_RECEIPTS,'ORDERED_RECEIPT_LIMIT')
        self.n=self.config['tensor_shape'][0];self.count=self.samples=self.overflow=self.windows=0
        self.anchor=None;self.window=0;self.previous=None;self.pending={};self.seen=set();self.sources={}
        self.complete=False;self.digest=hashlib.sha256(b'nids-processing-order/1\0')
        self.digest.update(hashlib.sha256(canonical_bytes(self.view['source'])).digest())

    def receipt(self,value):
        bounded_json(value)
        require(not self.complete and type(value) is dict and set(value)==RECEIPT_KEYS,'ORDERED_RECEIPT_SCHEMA_OR_AFTER_COMPLETE')
        require(value['kind']=='ordered_packet_receipt' and value['schema_version']=='2.0.0'
                and value['label_status']=='UNKNOWN','ORDERED_RECEIPT_VERSION_OR_LABEL')
        require(value['capture']==self.view['source']['capture'] and
                value['processing_view_sha256']==self.view['processing_order_sha256'],'RECEIPT_CONTEXT_OR_VIEW')
        index=value['processing_index'];ordinal=value['source_packet_ordinal'];time=value['timestamp_ns']
        require(type(index) is int and index==self.count+1,'RECEIPT_PROCESSING_INDEX')
        first=self.view['source']['source_first_ordinal']
        require(type(ordinal) is int and first<=ordinal<first+self.view['record_count']
                and ordinal not in self.sources,'RECEIPT_SOURCE_ORDINAL')
        require(type(time) is int and 0<=time<=2**63-1,'RECEIPT_TIMESTAMP')
        key=(time,ordinal)
        require(self.previous is None or key>self.previous,'RECEIPT_PROCESSING_ORDER')
        require(type(value['source_packet_sha256']) is str and
                re.fullmatch('[a-f0-9]{64}',value['source_packet_sha256']) is not None,'RECEIPT_PACKET_HASH')
        flow=value['flow_tuple']
        require(type(flow) is list and len(flow)==5 and all(type(flow[i]) is str for i in (0,2))
                and all(type(flow[i]) is int and 0<=flow[i]<=65535 for i in (1,3))
                and type(flow[4]) is int and flow[4] in (1,6,17),'RECEIPT_FLOW')
        require(all(str(ipaddress.IPv4Address(flow[i]))==flow[i] for i in (0,2))
                and (flow[0],flow[1])<=(flow[2],flow[3]),'RECEIPT_CANONICAL_FLOW')
        require(flow[4]!=1 or flow[1]==flow[3]==0,'RECEIPT_ICMP_PORTS')
        require(type(value['retained']) is bool and type(value['unknown_protocol_tokens']) is list
                and all(type(v) is str for v in value['unknown_protocol_tokens']),'RECEIPT_FIELD_TYPES')
        if self.anchor is None:self.anchor=time;self.windows=1
        elif time>self.anchor+self.config['window_ns']:
            require(not self.pending,'PREVIOUS_WINDOW_TENSORS_MISSING')
            self.anchor=time;self.window+=1;self.windows+=1
        require(type(value['window_index']) is int and value['window_index']==self.window,'RECEIPT_WINDOW_INDEX')
        group=(self.window,*flow)
        require(group not in self.seen,'RECEIPT_AFTER_FLOW_TENSOR')
        if group not in self.pending:
            require(len(self.pending)<128,'ORDERED_PENDING_FLOW_LIMIT')
            self.pending[group]=dict(count=0,ordinals=[],indices=[],first=ordinal,last=ordinal,
                min=ordinal,max=ordinal,first_time=time,last_time=time,first_index=index,last_index=index,
                full=hashlib.sha256(),overflow=hashlib.sha256(),refs=hashlib.sha256(),overflow_refs=hashlib.sha256())
        state=self.pending[group]
        retained=state['count']<self.n
        require(value['retained'] is retained,'RECEIPT_RETAINED_MASK')
        reference={k:value[k] for k in ('processing_index','source_packet_ordinal','timestamp_ns','source_packet_sha256','capture')}
        refbytes=canonical_bytes(reference)+b'\n';packet_hash=bytes.fromhex(value['source_packet_sha256'])
        state['full'].update(packet_hash);state['refs'].update(refbytes)
        if retained:state['ordinals'].append(ordinal);state['indices'].append(index)
        else:state['overflow'].update(packet_hash);state['overflow_refs'].update(refbytes);self.overflow+=1
        state['count']+=1;state['last']=ordinal;state['last_time']=time;state['last_index']=index
        state['min']=min(state['min'],ordinal);state['max']=max(state['max'],ordinal)
        self.count+=1;require(self.count<=MAX_RECEIPTS and self.count<=self.view['record_count'],'ORDERED_RECEIPT_LIMIT')
        self.sources[ordinal]=packet_hash;self.digest.update(refbytes);self.previous=key

    def tensor(self,tensor,state):
        require(not self.complete and type(tensor) is OrderedFlowTensor,'ORDERED_TENSOR_TYPE_OR_AFTER_COMPLETE')
        meta=tensor.metadata;bounded_json(meta)
        validate_tensor_bytes(tensor.shape,tensor.tensor_bytes,meta,self.config,self.view)
        group=(meta['window_index'],*meta['flow_tuple'])
        require(group in self.pending and group not in self.seen,'TENSOR_MISSING_RECEIPTS_OR_DUPLICATE')
        s=self.pending[group]
        expected=dict(real_packet_count=s['count'],retained_ordinals=s['ordinals'],retained_processing_indices=s['indices'],
            first_ordinal=s['first'],last_ordinal=s['last'],min_source_ordinal=s['min'],max_source_ordinal=s['max'],
            first_timestamp_ns=s['first_time'],last_timestamp_ns=s['last_time'],anchor_timestamp_ns=self.anchor,
            first_processing_index=s['first_index'],last_processing_index=s['last_index'],
            full_packet_digest=s['full'].hexdigest(),overflow_packet_digest=s['overflow'].hexdigest(),
            full_processing_digest=s['refs'].hexdigest(),overflow_processing_digest=s['overflow_refs'].hexdigest())
        require(all(canonical_bytes(meta[k])==canonical_bytes(v) for k,v in expected.items()),'TENSOR_RECEIPT_LINEAGE_MISMATCH')
        result=sample_hash(tensor,state)
        del self.pending[group];self.seen.add(group);self.samples+=1
        require(self.samples<=MAX_SAMPLES,'ORDERED_SAMPLE_LIMIT')
        return result

    def finish(self,value):
        bounded_json(value)
        require(not self.complete and type(value) is dict and set(value)==COMPLETION_KEYS,'COMPLETION_SCHEMA_OR_DUPLICATE')
        require(value['kind']=='ordered_stream_complete' and value['schema_version']=='2.0.0'
            and value['status']=='COMPLETE_SYNTHETIC_ONLY' and value['real_data_export']=='BLOCKED'
            and value['HDF5_integration']=='NOT_ACTIVE','COMPLETION_SCOPE')
        require(not self.pending and self.samples>0 and self.count==self.view['record_count'],'INCOMPLETE_FLOW_STREAM')
        require(self.digest.hexdigest()==self.view['processing_order_sha256'],'RECEIPT_VIEW_DIGEST_MISMATCH')
        source_digest=hashlib.sha256(b'nids-source-order/1\0')
        for ordinal in sorted(self.sources):source_digest.update(self.sources[ordinal])
        require(source_digest.hexdigest()==self.view['source']['expected_source_order_sha256'],'RECEIPT_SOURCE_DIGEST_MISMATCH')
        stats=value['stats'];require(type(stats) is dict,'COMPLETION_STATS')
        expected=dict(input_packets=self.count,accepted_packets=self.count,emitted_receipts=self.count,
            emitted_tensors=self.samples,overflow_packets=self.overflow,windows=self.windows,
            active_flows=0,retained_buffer_bytes=0,lineage_buffer_bytes=0,status='COMPLETE',
            extractor_version=self.config['extractor_version'],processing_view_sha256=self.view['processing_order_sha256'])
        require(all(k in stats and canonical_bytes(stats[k])==canonical_bytes(v) for k,v in expected.items()),'COMPLETION_STATS_MISMATCH')
        self.complete=True
