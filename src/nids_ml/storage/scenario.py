"""Bound a binary tensor pipe to an approved scenario and write HDF5 shards.
Ràng buộc pipe tensor với scenario đã duyệt, ghi HDF5 theo batch có giới hạn.
"""
from collections import Counter
from contextlib import closing
import hashlib
from pathlib import Path
import shutil
import struct

from nids_ml.flows.extractor import FlowTensor
from nids_ml.flows.features import configuration
from nids_ml.flows.mapped import read_manifest,require
from nids_ml.labeling.contracts import canonical_bytes,strict_json
from nids_ml.labeling.scenario import PROVENANCE,HEX,label_tensor,load_policy,select_source
from nids_ml.pcap.decoder import file_sha256
from nids_ml.storage.shards import FlowShardReader,ShardWriter

FORMAT='nids-scenario-flow-shards/1.0.0'
PIPE_FORMAT='nids-scenario-tensor-pipe/1.0.0'
MAX_FRAME=16384


def pipe_header(policy,policy_sha,rule):
    return dict(format=PIPE_FORMAT,policy_sha256=policy_sha,dataset_id=policy['dataset_id'],source=rule,
                label_provenance=PROVENANCE,independent_ground_truth_verified=False,
                feature_configuration=configuration(input_kind='decoded_pcap'))


def write_frame(stream,kind,payload):
    require(kind in (b'H',b'T',b'E') and type(payload) is bytes,'SCENARIO_FRAME_TYPE')
    data=kind+payload
    require(1<=len(data)<=MAX_FRAME,'SCENARIO_FRAME_BYTE_CAP')
    stream.write(struct.pack('<I',len(data)));stream.write(data)
    return data


def read_frame(stream):
    prefix=bytearray()
    while len(prefix)<4:
        chunk=stream.read(4-len(prefix))
        require(bool(chunk),'INCOMPLETE_SCENARIO_STREAM')
        prefix.extend(chunk)
    length=struct.unpack('<I',prefix)[0]
    require(1<=length<=MAX_FRAME,'SCENARIO_FRAME_BYTE_CAP')
    data=bytearray()
    while len(data)<length:
        chunk=stream.read(length-len(data))
        require(bool(chunk),'INCOMPLETE_SCENARIO_FRAME')
        data.extend(chunk)
    return bytes(data[:1]),bytes(data[1:])


def validate_completion(trailer,policy_sha,rule,total,frames_sha=None):
    require(trailer['format']==PIPE_FORMAT and trailer['policy_sha256']==policy_sha
            and trailer['source_id']==rule['source_id'] and trailer['source_complete'] is True
            and type(trailer['tensors']) is int and trailer['tensors']==total,
            'SCENARIO_COMPLETION_BINDING')
    if frames_sha is not None:
        require(trailer['tensor_frames_sha256']==frames_sha,'SCENARIO_COMPLETION_DIGEST')
    decoder=trailer['decoder'];stats=trailer['stream']
    transport=trailer.get('source_identity',dict(compression='none',sha256=rule['sha256'],size_bytes=rule['size_bytes'],
        logical_sha256=rule['sha256'],logical_size_bytes=rule['size_bytes'],logical_verified=True))
    require(transport['compression'] in ('none','gzip') and transport['logical_verified'] is True
            and transport['logical_sha256']==rule['sha256'] and transport['logical_size_bytes']==rule['size_bytes']
            and type(transport['sha256']) is str and HEX.fullmatch(transport['sha256'])
            and type(transport['size_bytes']) is int and 1<=transport['size_bytes']<=rule['size_bytes']+1048576,
            'SCENARIO_LOGICAL_SOURCE_BINDING')
    if transport['compression']=='none':
        require(transport['sha256']==rule['sha256'] and transport['size_bytes']==rule['size_bytes'],
                'SCENARIO_RAW_SOURCE_BINDING')
    require(decoder['stats']['source_complete'] is True
            and decoder['source_sha256']==transport['sha256']
            and type(decoder['child_returncode']) is int and decoder['child_returncode']==0
            and type(decoder['stats']['decoded_records']) is int and type(stats['input_packets']) is int
            and decoder['stats']['decoded_records']==stats['input_packets']==rule['records']
            and stats['status']=='COMPLETE' and stats['tensors']==total
            and stats['input_packets']==sum(stats[k] for k in ('feature_packets','outside_packets','quarantined_packets')),
            'SCENARIO_WHOLE_CAPTURE_ACCOUNTING')


class ScenarioWriter(ShardWriter):
    format=FORMAT
    provenance=PROVENANCE
    maximum_shards=8192

    def __init__(self,directory,max_output_mib):
        super().__init__(directory)
        require(type(max_output_mib) is int and 16<=max_output_mib<=2048,'SCENARIO_OUTPUT_BUDGET')
        self.maximum=max_output_mib*1048576

    def guard(self,*,before_batch=False):
        size=sum(s['bytes'] for s in self.shards)
        if self.handle is not None:
            self.handle.flush();size+=self.path.stat().st_size
        require(size+(1048576 if before_batch else 0)<=self.maximum,'SCENARIO_OUTPUT_BYTE_CAP')
        require(shutil.disk_usage(self.directory).free>=1024**3,'SCENARIO_OUTPUT_DISK_RESERVE')

    def flush(self):
        if not self.buffer:return
        self.guard(before_batch=True)
        super().flush()
        self.guard()


def export_scenario_pipe(stream,policy_path,policy_sha,source_id,output,*,max_output_mib=1024,progress=None):
    """Only a full verified EOF trailer can publish a complete HDF5 manifest.
    Chỉ công bố manifest sau trailer chứng minh EOF toàn nguồn; không nhận prefix.
    """
    policy,digest=load_policy(policy_path,policy_sha);rule=select_source(policy,source_id)
    output=Path(output).absolute()
    require(not output.exists() and not output.is_symlink() and output.parent.is_dir()
            and not any(p.is_symlink() for p in output.parents),'FRESH_SCENARIO_OUTPUT_REQUIRED')
    writer=ScenarioWriter(output,max_output_mib)
    require(shutil.disk_usage(output.parent).free>=writer.maximum+1024**3,'SCENARIO_OUTPUT_START_RESERVE')
    kind,payload=read_frame(stream)
    header=pipe_header(policy,digest,rule)
    require(kind==b'H' and strict_json(payload)==header,'SCENARIO_PIPE_HEADER_BINDING')
    output.mkdir();counts=Counter();families=Counter();running=hashlib.sha256();total=0
    try:
        while True:
            kind,payload=read_frame(stream)
            if kind==b'E':
                trailer=strict_json(payload);break
            require(kind==b'T' and len(payload)>880,'SCENARIO_TENSOR_FRAME')
            metadata=strict_json(payload[880:])
            tensor=FlowTensor((20,11,1),payload[:880],canonical_bytes(metadata['core']))
            require(metadata==label_tensor(tensor,policy,digest,rule),'SCENARIO_METADATA_BINDING')
            total+=1;require(total<=rule['records'],'SCENARIO_TENSOR_COUNT_CAP')
            running.update(b'T'+payload)
            writer.add(tensor,metadata);counts[str(metadata['y'])]+=1;families[metadata['attack_type']]+=1
            if progress and total%4096==0:progress('scenario_hdf5',total)
        require(stream.read(1)==b'','SCENARIO_TRAILING_BYTES')
        validate_completion(trailer,digest,rule,total,running.hexdigest())
        writer.flush();writer.seal();writer.guard();require(writer.count>0,'NO_ELIGIBLE_SCENARIO_FLOWS')
        result=dict(format=FORMAT,status='COMPLETE_SCENARIO_FLOW_SHARDS',samples=writer.count,
            shape=[writer.count,20,11,1],shards=writer.shards,label_counts=dict(counts),
            attack_type_counts=dict(families),label_provenance=PROVENANCE,
            independent_ground_truth_verified=False,policy_sha256=digest,source_id=source_id,
            source=rule,dataset_id=policy['dataset_id'],
            feature_configuration=configuration(input_kind='decoded_pcap'),
            split='PUBLISHER_'+rule['publisher_split'].upper(),publisher_split=rule['publisher_split'],
            split_reassigned=False,training_ready=False,training=False,
            required_before_training='GLOBAL_PARENT_CONNECTION_AND_EXACT_TENSOR_DEDUP_AUDIT; VALIDATION_FROM_TRAIN_ONLY',
            label_policy='APPROVED_CAPTURE_SCENARIO; ALL_REAL_PACKETS; QUARANTINED_WINDOWS_EXCLUDED',
            stream_completion=trailer)
        with (output/'manifest.json').open('xb') as f:f.write(canonical_bytes(result))
        return result
    except BaseException as error:
        writer.close_failed()
        with (output/'FAILED.json').open('xb') as f:f.write(canonical_bytes(dict(
            status='FAILED_RETAINED',error=str(error) or type(error).__name__,automatic_cleanup=False,training=False)))
        raise


class ScenarioShardReader(FlowShardReader):
    format=FORMAT
    provenance=PROVENANCE
    completion_status='COMPLETE_SCENARIO_FLOW_SHARDS'
    maximum_shards=8192

    def __init__(self,directory,policy_path,policy_sha):
        self.policy,self.policy_sha=load_policy(policy_path,policy_sha)
        super().__init__(directory)

    def _manifest_scope(self,m):
        rule=select_source(self.policy,m['source_id']);self.rule=rule
        require(m['source']==rule and m['policy_sha256']==self.policy_sha
                and m['dataset_id']==self.policy['dataset_id'] and m['split']=='PUBLISHER_'+rule['publisher_split'].upper()
                and m['publisher_split']==rule['publisher_split'] and m['split_reassigned'] is False
                and m['training_ready'] is False
                and type(m['samples']) is int and 1<=m['samples']<=rule['records']
                and m['feature_configuration']==configuration(input_kind='decoded_pcap')
                and m['label_counts']=={str(rule['y']):m['samples']}
                and m['attack_type_counts']=={rule['attack_type']:m['samples']},'SCENARIO_MANIFEST_SCOPE')
        validate_completion(m['stream_completion'],self.policy_sha,rule,m['samples'])

    def batches(self,batch_size=64,*,metadata=False):
        require(type(metadata) is bool,'SCENARIO_METADATA_FLAG')
        running=hashlib.sha256();total=0
        with closing(super().batches(batch_size,metadata=True)) as batches:
            for batch in batches:
                for i,encoded in enumerate(batch['metadata']):
                    meta=strict_json(encoded.tobytes())
                    tensor=FlowTensor((20,11,1),batch['X'][i].astype('<f4',copy=False).tobytes(),canonical_bytes(meta['core']))
                    require(meta==label_tensor(tensor,self.policy,self.policy_sha,self.rule),'SCENARIO_ROW_LINEAGE')
                    running.update(b'T'+tensor.tensor_bytes+canonical_bytes(meta));total+=1
                if not metadata:batch.pop('metadata')
                yield batch
        validate_completion(self.manifest['stream_completion'],self.policy_sha,self.rule,total,running.hexdigest())
