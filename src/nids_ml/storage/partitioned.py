"""Read catalog-selected HDF5 rows in bounded, reproducibly shuffled batches.
Đọc đúng split/client từ catalog; RAM/file handles có giới hạn, không load toàn dataset.
"""
from collections import OrderedDict,defaultdict
from contextlib import contextmanager,closing
import hashlib
from pathlib import Path

from nids_ml.data.catalog import FORMAT,SPLITS,identity,audit,class_counts
from nids_ml.flows.mapped import read_manifest,readonly,require,PROVENANCE
from nids_ml.pcap.decoder import regular_file,file_sha256,file_signature
from nids_ml.storage.shards import FlowShardReader


class PartitionReader:
    def __init__(self,directory,workspace=None,*,sources=None,scratch=None,budget=None):
        self.directory=Path(directory).absolute();self.manifest=read_manifest(self.directory)
        self.workspace=Path(workspace or Path(__file__).resolve().parents[3]).resolve()
        m=self.manifest
        require(m['format']==FORMAT and m['status']=='COMPLETE_CONNECTION_CATALOG'
            and m['training_ready'] is True and m['label_provenance']==PROVENANCE
            and m['independent_ground_truth_verified'] is False,'READY_NIDS_CATALOG_REQUIRED')
        require(m['catalog_id']==identity(m) and m['index']['filename']=='catalog.sqlite','CATALOG_IDENTITY')
        self.index=regular_file(self.directory/'catalog.sqlite');self.signature=file_signature(self.index)
        require(self.index.stat().st_size==m['index']['bytes']<=1536*1048576
                and file_sha256(self.index)==m['index']['sha256'],'CATALOG_INDEX_SHA256')
        require(1<=len(m['sources'])<=24,'CATALOG_SOURCE_CAP')
        self.inputs={s['dataset_id']:s for s in m['sources']}
        require(len(self.inputs)==len(m['sources']),'DUPLICATE_CATALOG_SOURCE')
        self.overrides={} if sources is None else dict(sources)
        self.scratch=None if scratch is None else Path(scratch).absolute()
        self.budget=budget
        require(set(self.overrides)<=set(self.inputs),'UNKNOWN_SOURCE_OVERRIDE')
        self.readers={}
        with readonly(self.index) as db:
            audit(db)
            require({s:class_counts(db,s) for s in SPLITS}==m['split_counts'],'CATALOG_GLOBAL_COUNTS')
            require({str(c):{s:class_counts(db,s,c) for s in SPLITS} for c in (0,1)}==m['client_counts'],
                    'CATALOG_CLIENT_COUNTS')
            require(db.execute('SELECT count(*) FROM samples WHERE eligible=1').fetchone()[0]
                    ==m['statistics']['retained_samples'],'CATALOG_RETAINED_COUNT')

    def selection(self,split,client):
        require(split in SPLITS and (client is None or type(client) is int and client in (0,1)),'SPLIT_OR_CLIENT')
        where='eligible=1 AND split=?';args=[split]
        if client is not None:
            where+=' AND client=?' if split=='train' else ' AND (owner IS NULL OR owner=?)'
            args.append(client)
        return where,args

    def counts(self,split,client=None):
        self.selection(split,client)
        return self.manifest['split_counts'][split] if client is None else self.manifest['client_counts'][str(client)][split]

    def dataset(self,dataset_id):
        if dataset_id not in self.readers:
            source=self.inputs[dataset_id]
            path=Path(self.overrides.get(dataset_id,source['path_hint']))
            if not path.is_absolute():path=self.workspace/path
            require(path.is_dir(),'LOCAL_DATASET_MISSING:'+dataset_id)
            require(file_sha256(regular_file(path/'manifest.json'))==dataset_id,'LOCAL_DATASET_IDENTITY')
            reader=FlowShardReader(path)
            require(reader.manifest['shards']==source['shards'],'LOCAL_SHARD_IDENTITY')
            self.readers[dataset_id]=reader
        return self.readers[dataset_id]

    def preflight(self,split,client=None):
        where,args=self.selection(split,client)
        with readonly(self.index) as db:
            for (dataset_id,) in db.execute('SELECT DISTINCT dataset FROM samples WHERE '+where,args):
                self.dataset(dataset_id)

    @contextmanager
    def batch_index(self):
        with readonly(self.index) as db:
            previous=db.execute('PRAGMA temp_store_directory').fetchone()
            try:
                db.execute('PRAGMA temp_store=FILE')
                if self.scratch is not None:
                    db.execute('PRAGMA temp_store_directory='+db.execute('SELECT quote(?)',(str(self.scratch),)).fetchone()[0])
                if self.budget is not None:db.set_progress_handler(self.budget.sqlite_progress,20000)
                yield db
            finally:
                db.set_progress_handler(None,0)
                if self.scratch is not None:
                    db.execute('PRAGMA temp_store_directory='+db.execute('SELECT quote(?)',(previous[0] if previous else '',)).fetchone()[0])

    def batches(self,split,client=None,*,batch_size=64,epoch=None):
        import h5py
        import numpy as np
        require(type(batch_size) is int and 1<=batch_size<=128,'BATCH_SIZE_1_TO_128')
        require(epoch is None or type(epoch) is int and 0<=epoch<=1000,'SHUFFLE_EPOCH')
        require(epoch is None or self.scratch is not None and self.scratch.is_dir()
                and not self.scratch.is_symlink() and not any(p.is_symlink() for p in self.scratch.parents),
                'SHUFFLE_REQUIRES_OWN_SCRATCH_DIRECTORY')
        where,args=self.selection(split,client);self.preflight(split,client)
        require(file_signature(self.index)==self.signature and file_sha256(self.index)==self.manifest['index']['sha256'],
                'CATALOG_CHANGED_BEFORE_READ')
        handles=OrderedDict();used=set()
        try:
            with self.batch_index() as db:
                # The Python callback only hashes trusted scalar IDs; no executable input.
                # Chỉ hash ID/seed/epoch; SQLite không thực thi code lấy từ metadata.
                if epoch is not None:
                    prefix=str(self.manifest['seed'])+':'+str(epoch)+':'
                    db.create_function('nids_order',1,lambda value:hashlib.sha256((prefix+value).encode()).digest())
                order='nids_order(sha),id' if epoch is not None else 'dataset,shard,row_no'
                index_hint='' if epoch is not None else ' INDEXED BY location'
                with closing(db.execute('SELECT id,dataset,shard,row_no,sha,group_id,y,family FROM samples'+index_hint+' WHERE '+where+' ORDER BY '+order,args)) as cursor:
                    while True:
                        rows=cursor.fetchmany(batch_size)
                        if not rows:break
                        x=np.empty((len(rows),20,11,1),dtype='<f4');locations=defaultdict(list)
                        for position,row in enumerate(rows):locations[row[1],row[2]].append((row[3],position,row))
                        for key,selected in locations.items():
                            dataset_id,shard=key;reader=self.dataset(dataset_id);used.add(dataset_id)
                            require(type(shard) is int and 0<=shard<len(reader.identities),'CATALOG_SHARD_NUMBER')
                            path,signature,entry=reader.identities[shard]
                            require(file_signature(path)==signature,'SHARD_CHANGED_BEFORE_BATCH')
                            if key not in handles:
                                if len(handles)==8:handles.popitem(last=False)[1].close()
                                handles[key]=h5py.File(path,'r',rdcc_nbytes=1048576)
                            handles.move_to_end(key);handle=handles[key]
                            ordered=sorted(selected);indices=[item[0] for item in ordered]
                            require(len(set(indices))==len(indices) and 0<=indices[0]<=indices[-1]<entry['samples'],'CATALOG_LOCATION')
                            values=handle['X'][indices]
                            for value,(_,position,row) in zip(values,ordered):
                                y=int(handle['y'][row[3]]);group=handle['group_id'][row[3]].decode()
                                sha=handle['tensor_sha256'][row[3]].decode();family=handle['attack_type'][row[3]].decode()
                                require(y==row[6] and group==row[5] and sha==row[4] and family==row[7],
                                        'CATALOG_ROW_VS_SHARD_MISMATCH')
                                require(np.isfinite(value).all() and ((value>=0)&(value<=1)).all()
                                    and hashlib.sha256(value.astype('<f4',copy=False).tobytes()).hexdigest()==sha,'INVALID_BATCH_TENSOR')
                                x[position]=value
                        yield dict(X=x,y=np.asarray([r[6] for r in rows],dtype='<f4').reshape(-1,1),
                            sample_id=[r[0] for r in rows],sha=[r[4] for r in rows],group_id=[r[5] for r in rows],
                            attack_type=[r[7] for r in rows])
            require(file_signature(self.index)==self.signature and file_sha256(self.index)==self.manifest['index']['sha256'],
                    'CATALOG_CHANGED_DURING_READ')
            for dataset_id in used:
                reader=self.readers[dataset_id]
                require(file_sha256(reader.directory/'manifest.json')==dataset_id,'SOURCE_MANIFEST_CHANGED')
                for path,signature,entry in reader.identities:
                    require(file_signature(path)==signature and file_sha256(path)==entry['sha256'],'SOURCE_SHARD_CHANGED')
        finally:
            for handle in handles.values():handle.close()
