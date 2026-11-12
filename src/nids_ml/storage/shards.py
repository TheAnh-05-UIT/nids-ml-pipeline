"""Write fixed-size LUCID HDF5 shards and read bounded batches without TensorFlow.
Ghi HDF5 theo shard, đọc từng batch; không nạp toàn dataset vào RAM hoặc tự chia tập.
"""
from collections import Counter
import hashlib
from pathlib import Path

from nids_ml.flows.mapped import (iter_tensors, new_output, read_manifest, require, PROVENANCE)
from nids_ml.labeling.contracts import canonical_bytes, strict_json
from nids_ml.ordering.view import _new_json
from nids_ml.pcap.decoder import regular_file, file_sha256, file_signature

FORMAT='nids-mapped-flow-shards/1.0.0'
SHARD_ROWS=4096
WRITE_BATCH=64
METADATA_BYTES=8192
LAYOUT={'X':('<f4',(20,11,1)), 'y':('i1',()), 'group_id':('S64',()),
        'attack_type':('S64',()), 'tensor_sha256':('S64',()), 'metadata':('S8192',())}


class ShardWriter:
    def __init__(self,directory):
        self.directory=Path(directory); self.handle=None; self.buffer=[]; self.shards=[]; self.count=0; self.row=0

    def add(self,tensor,metadata):
        require(tensor.shape==(20,11,1) and len(tensor.tensor_bytes)==880,'FLOW_TENSOR_SHAPE')
        require(metadata['y'] in (0,1) and type(metadata['y']) is int,'NUMERIC_FLOW_LABEL')
        encoded=canonical_bytes(metadata); require(len(encoded)<=METADATA_BYTES,'FLOW_METADATA_BYTE_CAP')
        self.buffer.append((tensor.tensor_bytes,metadata,encoded))
        if len(self.buffer)==WRITE_BATCH: self.flush()

    def flush(self):
        if not self.buffer: return
        import h5py
        import numpy as np
        if self.handle is None:
            require(len(self.shards)<2048,'SHARD_COUNT_CAP')
            self.path=self.directory/('flows-%05d.h5'%len(self.shards))
            self.handle=h5py.File(self.path,'x'); self.row=0
            self.handle.attrs['format']=FORMAT; self.handle.attrs['label_provenance']=PROVENANCE
            self.handle.attrs['independent_ground_truth_verified']=False
            for name,(dtype,tail) in LAYOUT.items():
                self.handle.create_dataset(name,shape=(0,*tail),maxshape=(SHARD_ROWS,*tail),dtype=dtype,
                    chunks=(WRITE_BATCH,*tail),compression='gzip',compression_opts=1,fletcher32=True,track_times=False)
        batch=self.buffer; self.buffer=[]
        require(self.row+len(batch)<=SHARD_ROWS,'SHARD_ROW_LIMIT')
        for name,(_,tail) in LAYOUT.items(): self.handle[name].resize((self.row+len(batch),*tail))
        end=self.row+len(batch)
        x=np.stack([np.frombuffer(item[0],dtype='<f4').reshape(20,11,1) for item in batch])
        require(np.isfinite(x).all() and ((x>=0)&(x<=1)).all(),'INVALID_NORMALIZED_FLOW_TENSOR')
        self.handle['X'][self.row:end]=x
        for name in ('y','group_id','attack_type','tensor_sha256'):
            values=[item[1][name] if name=='y' else item[1][name].encode('ascii') for item in batch]
            self.handle[name][self.row:end]=values
        self.handle['metadata'][self.row:end]=[item[2] for item in batch]
        self.row=end; self.count+=len(batch)
        if self.row==SHARD_ROWS: self.seal()

    def seal(self):
        if self.handle is not None:
            self.handle.close(); self.handle=None
            self.shards.append(dict(filename=self.path.name,samples=self.row,
                bytes=self.path.stat().st_size,sha256=file_sha256(self.path)))

    def close_failed(self):
        if self.handle is not None: self.handle.close(); self.handle=None


def export_shards(caches,workspace,output,progress=None):
    caches=[Path(path).absolute() for path in caches]
    require(1<=len(caches)<=24 and len(set(caches))==len(caches),'ONE_TO_24_DISTINCT_CACHES_REQUIRED')
    reports=[read_manifest(path) for path in caches]
    hashes=[r['context']['capture_sha256'] for r in reports]
    require(len(set(hashes))==len(hashes),'DUPLICATE_CAPTURE_NOT_ALLOWED')
    # Canonical ordering makes the same cache set produce the same shards.
    # Thứ tự checksum cố định giúp input list đảo thứ tự vẫn sinh nội dung giống nhau.
    caches=[path for _,path in sorted(zip(hashes,caches))]
    output,budget=new_output(output,workspace); writer=ShardWriter(output)
    stats=Counter(); labels=Counter(); families=Counter(); sources=[]
    try:
        for cache in caches:
            report=read_manifest(cache); current=Counter()
            for tensor,metadata in iter_tensors(cache,current):
                writer.add(tensor,metadata); labels[str(metadata['y'])]+=1; families[metadata['attack_type']]+=1
                if sum(labels.values())%256==0:
                    budget.check()
                    if progress: progress('export',sum(labels.values()))
            stats.update(current)
            sources.append(dict(context=report['context'],index=report['index'],counts=dict(current),
                                source=report['source'],mapping_index_sha256=report['mapping_index_sha256']))
        writer.flush(); writer.seal(); require(writer.count>0,'NO_ELIGIBLE_FLOW_WINDOWS')
        budget.check(force=True)
        result=dict(format=FORMAT,status='COMPLETE_MAPPED_FLOW_SHARDS',samples=writer.count,
            shape=[writer.count,20,11,1],shards=writer.shards,counts=dict(stats),label_counts=dict(labels),
            attack_type_counts=dict(families),sources=sources,label_provenance=PROVENANCE,
            independent_ground_truth_verified=False,feature_configuration=reports[0]['configuration'],
            ordering='SOURCE_CONTIGUOUS_DECODE_THEN_STABLE_TIMESTAMP_ORDINAL_VIEW',
            label_policy='ALL_REAL_PACKETS_INCLUDING_OVERFLOW; UNKNOWN/MIXED/QUARANTINE_EXCLUDED',
            split='UNASSIGNED',exact_tensor_dedup='NOT_APPLIED',training_ready=False,training=False,
            readiness='REQUIRES_GLOBAL_CONNECTION_SPLIT_AND_EXACT_TENSOR_DEDUP_BEFORE_TRAINING',
            resources=dict(peak_scratch_bytes=budget.peak_scratch,peak_python_rss_bytes=budget.peak_python_rss),
            native_ARM64='NOT VERIFIED')
        _new_json(output/'manifest.json',result)
        return result
    except Exception as error:
        writer.close_failed()
        _new_json(output/'FAILED.json',dict(status='FAILED_RETAINED',error=str(error),automatic_cleanup=False))
        raise


class FlowShardReader:
    """Verify all shard identities, then yield a maximum of 256 rows per batch.
    Kiểm SHA/layout trước khi đọc; drain/close generator để đóng HDF5 khi dừng sớm.
    """
    def __init__(self,directory):
        import h5py
        import numpy as np
        self.directory=Path(directory).absolute(); self.manifest=read_manifest(self.directory)
        m=self.manifest
        require(m['format']==FORMAT and m['status']=='COMPLETE_MAPPED_FLOW_SHARDS','SHARD_MANIFEST_FORMAT')
        require(m['label_provenance']==PROVENANCE and m['independent_ground_truth_verified'] is False,'SHARD_PROVENANCE')
        require(m['split']=='UNASSIGNED' and m['training_ready'] is False,'UNEXPECTED_TRAINING_READINESS')
        require(type(m['shards']) is list and 1<=len(m['shards'])<=2048,'SHARD_LIST_CAP')
        self.identities=[]; total=0
        for number,entry in enumerate(m['shards']):
            require(entry['filename']=='flows-%05d.h5'%number,'SHARD_FILENAME_OR_ORDER')
            require(type(entry['samples']) is int and 1<=entry['samples']<=SHARD_ROWS,'SHARD_SAMPLE_CAP')
            path=regular_file(self.directory/entry['filename'])
            require(path.stat().st_size==entry['bytes'] and path.stat().st_size<=64*1048576
                    and file_sha256(path)==entry['sha256'],'SHARD_SHA256_OR_SIZE')
            with h5py.File(path,'r') as handle:
                require(set(handle)==set(LAYOUT) and handle.attrs.get('format')==FORMAT
                        and handle.attrs.get('label_provenance')==PROVENANCE
                        and handle.attrs.get('independent_ground_truth_verified')==False,'SHARD_LAYOUT_OR_ATTRIBUTES')
                for name,(dtype,tail) in LAYOUT.items():
                    require(isinstance(handle.get(name,getlink=True),h5py.HardLink),'UNSAFE_HDF5_LINK')
                    item=handle[name]
                    require(isinstance(item,h5py.Dataset) and not item.is_virtual and not item.external
                            and item.shape==(entry['samples'],*tail) and item.dtype==np.dtype(dtype),'SHARD_DATASET_SCHEMA')
            self.identities.append((path,file_signature(path),entry)); total+=entry['samples']
        require(type(m['samples']) is int and total==m['samples'] and m['shape']==[total,20,11,1],'SHARD_TOTAL_COUNT')

    def batches(self,batch_size=64,*,metadata=False):
        """Use contextlib.closing(reader.batches(...)) when stopping before EOF.
        Nếu dừng trước EOF, dùng contextlib.closing để giải phóng file ngay.
        """
        import h5py
        import numpy as np
        require(type(batch_size) is int and 1<=batch_size<=256 and type(metadata) is bool,'BATCH_SIZE_1_TO_256')
        for path,signature,entry in self.identities:
            require(file_signature(path)==signature and file_sha256(path)==entry['sha256'],'SHARD_CHANGED_BEFORE_READ')
            with h5py.File(path,'r') as handle:
                for start in range(0,entry['samples'],batch_size):
                    end=min(start+batch_size,entry['samples'])
                    batch={name:handle[name][start:end] for name in LAYOUT if metadata or name!='metadata'}
                    require(np.isfinite(batch['X']).all() and ((batch['X']>=0)&(batch['X']<=1)).all()
                            and np.isin(batch['y'],[0,1]).all(),'INVALID_FLOW_BATCH')
                    for i in range(end-start):
                        require(hashlib.sha256(batch['X'][i].astype('<f4',copy=False).tobytes()).hexdigest()
                                ==batch['tensor_sha256'][i].decode(),'TENSOR_SHA256_MISMATCH')
                        if metadata:
                            item=strict_json(batch['metadata'][i].tobytes())
                            require(item['y']==int(batch['y'][i]) and item['group_id']==batch['group_id'][i].decode()
                                    and item['attack_type']==batch['attack_type'][i].decode()
                                    and item['tensor_sha256']==batch['tensor_sha256'][i].decode()
                                    and item['label_provenance']==PROVENANCE
                                    and item['independent_ground_truth_verified'] is False,'BATCH_METADATA_MISMATCH')
                    yield batch
            require(file_signature(path)==signature and file_sha256(path)==entry['sha256'],'SHARD_CHANGED_DURING_READ')
