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
    format=FORMAT
    provenance=PROVENANCE
    completion_status='COMPLETE_CONNECTION_CATALOG'
    catalog_identity=staticmethod(identity)
    catalog_audit=staticmethod(audit)
    extra_index_columns=''

    def validate_selection(self,handle,rows,split):
        """Optional artifact-specific metadata checks before a batch is yielded."""
        pass

    def manifest_scope(self,m):
        pass

    def source_reader(self,path):
        return FlowShardReader(path)

    def __init__(self,directory,workspace=None,*,sources=None,scratch=None,budget=None):
        self.directory=Path(directory).absolute();self.manifest=read_manifest(self.directory)
        self.workspace=Path(workspace or Path(__file__).resolve().parents[3]).resolve()
        m=self.manifest
        require(m['format']==self.format and m['status']==self.completion_status
            and m['training_ready'] is True and m['label_provenance']==self.provenance
            and m['independent_ground_truth_verified'] is False,'READY_NIDS_CATALOG_REQUIRED')
        self.manifest_scope(m)
        require(m['catalog_id']==self.catalog_identity(m) and m['index']['filename']=='catalog.sqlite','CATALOG_IDENTITY')
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
            self.catalog_audit(db)
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
            reader=self.source_reader(path)
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
        # Skip multi-hundred-MiB full re-hashing per batch pass when file signatures are unchanged.
        large_index=self.signature[2]>16*1048576
        require(file_signature(self.index)==self.signature
                and (large_index or file_sha256(self.index)==self.manifest['index']['sha256']),
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
                with closing(db.execute('SELECT id,dataset,shard,row_no,sha,group_id,y,family'+self.extra_index_columns+' FROM samples'+index_hint+' WHERE '+where+' ORDER BY '+order,args)) as cursor:
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
                            self.validate_selection(handle,[item[2] for item in ordered],split)
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
            require(file_signature(self.index)==self.signature
                    and (large_index or file_sha256(self.index)==self.manifest['index']['sha256']),
                    'CATALOG_CHANGED_DURING_READ')
            for dataset_id in used:
                reader=self.readers[dataset_id]
                require(file_sha256(reader.directory/'manifest.json')==dataset_id,'SOURCE_MANIFEST_CHANGED')
                large_dataset=len(reader.identities)>4
                for path,signature,entry in reader.identities:
                    require(file_signature(path)==signature
                            and (large_dataset or file_sha256(path)==entry['sha256']),'SOURCE_SHARD_CHANGED')
        finally:
            for handle in handles.values():handle.close()


class ScenarioPartitionReader(PartitionReader):
    """Read the same bounded batch protocol with explicit approved-scenario scope.
    Đọc batch cùng giao diện, nhưng bắt buộc policy/provenance scenario riêng.
    """

    def __init__(self,directory,workspace=None,*,policy_path,policy_sha,**kwargs):
        from nids_ml.data.scenario_catalog import FORMAT as SCENARIO_FORMAT,identity as scenario_identity
        from nids_ml.labeling.scenario import load_policy,PROVENANCE as SCENARIO_PROVENANCE
        self.policy,self.policy_sha=load_policy(policy_path,policy_sha)
        self.policy_path=policy_path
        self.format=SCENARIO_FORMAT;self.provenance=SCENARIO_PROVENANCE
        self.completion_status='COMPLETE_SCENARIO_CATALOG'
        self.catalog_identity=scenario_identity;self.catalog_audit=self.audit_database
        super().__init__(directory,workspace,**kwargs)

    def manifest_scope(self,m):
        from nids_ml.data.scenario_catalog import POLICY
        from nids_ml.labeling.scenario import FAMILIES
        require(len(self.policy['sources'])==6 and {(r['attack_type'],r['publisher_split']) for r in self.policy['sources']}
                =={(family,split) for family in FAMILIES for split in ('train','test')},'SCENARIO_SIX_SOURCE_SPLIT_SCOPE')
        require(m['policy_sha256']==self.policy_sha and m['split_policy']==POLICY
                and {s['source_id'] for s in m['sources']}=={s['source_id'] for s in self.policy['sources']}
                and len(m['sources'])==len(self.policy['sources'])
                and all(s['owner'] is None for s in m['sources']),'SCENARIO_CATALOG_POLICY_SCOPE')
        rules={r['source_id']:r for r in self.policy['sources']}
        require(all(s['publisher_split']==rules[s['source_id']]['publisher_split'] for s in m['sources']),
                'SCENARIO_CATALOG_SOURCE_SPLIT')

    def audit_database(self,db):
        from nids_ml.data.scenario_catalog import audit_catalog
        audit_catalog(db)
        require({r[0] for r in db.execute('SELECT DISTINCT dataset FROM samples')}<=set(self.inputs),
                'SCENARIO_CATALOG_UNKNOWN_DATASET')
        for dataset_id,source in self.inputs.items():
            require(db.execute('SELECT count(*) FROM samples WHERE dataset=? AND publisher!=?',
                (dataset_id,source['publisher_split'])).fetchone()[0]==0,'SCENARIO_CATALOG_SOURCE_MEMBERSHIP')
        families=[dict(split=s,family=f,samples=n) for s,f,n in db.execute(
            'SELECT split,family,count(*) FROM samples GROUP BY split,family ORDER BY split,family')]
        require(families==self.manifest['attack_type_counts'],'SCENARIO_CATALOG_FAMILY_COUNTS')

    def source_reader(self,path):
        from nids_ml.storage.scenario import ScenarioShardReader
        return ScenarioShardReader(path,self.policy_path,self.policy_sha)

    def dataset(self,dataset_id):
        reader=super().dataset(dataset_id);source=self.inputs[dataset_id]
        require(reader.manifest['source_id']==source['source_id']
                and reader.manifest['publisher_split']==source['publisher_split'],'SCENARIO_LOCAL_SOURCE_LINEAGE')
        return reader


class TemporalScenarioPartitionReader(ScenarioPartitionReader):
    """Explicit temporal scope; never accepted by connection-only entrypoints."""
    extra_index_columns=',anchor_ns,end_ns,first_ns,last_ns,publisher'

    def __init__(self,directory,workspace=None,*,policy_path,policy_sha,**kwargs):
        from nids_ml.data.temporal_catalog import FORMAT,identity
        from nids_ml.labeling.scenario import load_policy,PROVENANCE
        self.policy,self.policy_sha=load_policy(policy_path,policy_sha)
        self.policy_path=policy_path;self.format=FORMAT;self.provenance=PROVENANCE
        self.completion_status='COMPLETE_TEMPORAL_SCENARIO_CATALOG'
        self.catalog_identity=identity;self.catalog_audit=self.audit_database
        PartitionReader.__init__(self,directory,workspace,**kwargs)

    def manifest_scope(self,m):
        from nids_ml.data.temporal_catalog import POLICY,SCOPE,parameters,VALIDATION_PERCENT
        from nids_ml.labeling.scenario import FAMILIES
        require(len(self.policy['sources'])==6 and {(s['attack_type'],s['publisher_split']) for s in self.policy['sources']}
                =={(f,p) for f in FAMILIES for p in ('train','test')},'TEMPORAL_POLICY_SIX_SOURCES')
        self.rules={r['source_id']:r for r in self.policy['sources']}
        require(m['policy_sha256']==self.policy_sha and m['split_policy']==POLICY and m['scope']==SCOPE
                and m['parent_connection_independence'] is False and type(m['seed']) is int
                and 0<=m['seed']<2**31-1,'TEMPORAL_MANIFEST_SCOPE')
        t=m['temporal'];parameters(self.policy,t['gap_seconds'])
        require(t['validation_percent']==VALIDATION_PERCENT and set(t)=={'gap_seconds','validation_percent','bounds'}
                and set(t['bounds'])==set(FAMILIES),'TEMPORAL_PARAMETERS')
        for bound in t['bounds'].values():
            require(set(bound)=={'train_start_ns','train_end_ns','train_last_packet_ns','validation_start_ns','test_start_ns'}
                    and all(type(v) is int and 0<=v<2**63 for v in bound.values()),'TEMPORAL_MANIFEST_BOUNDS')
            require(bound['validation_start_ns']==bound['train_start_ns']+
                (bound['train_end_ns']-bound['train_start_ns'])*(100-VALIDATION_PERCENT)//100,
                'TEMPORAL_VALIDATION_CUTOFF')
        require(len(m['sources'])==6 and {s['source_id'] for s in m['sources']}==set(self.rules)
                and all(s['owner'] is None and s['publisher_split']==self.rules[s['source_id']]['publisher_split']
                        for s in m['sources']),'TEMPORAL_SOURCE_SCOPE')

    def audit_database(self,db):
        from nids_ml.data.temporal_catalog import audit_catalog
        audit_catalog(db,self.manifest['temporal'])
        require({r[0] for r in db.execute('SELECT DISTINCT dataset FROM samples')}<=set(self.inputs),
                'TEMPORAL_UNKNOWN_DATASET')
        for dataset_id,source in self.inputs.items():
            rule=self.rules[source['source_id']]
            require(not db.execute('SELECT 1 FROM samples WHERE dataset=? AND (publisher!=? OR family!=? OR y!=?) LIMIT 1',
                (dataset_id,rule['publisher_split'],rule['attack_type'],rule['y'])).fetchone(),'TEMPORAL_SOURCE_MEMBERSHIP')
        families=[dict(split=s,family=f,samples=n) for s,f,n in db.execute(
            'SELECT split,family,count(*) FROM samples GROUP BY split,family ORDER BY split,family')]
        require(families==self.manifest['attack_type_counts'],'TEMPORAL_FAMILY_COUNTS')

    def validate_selection(self,handle,rows,split):
        from nids_ml.data.temporal_catalog import row_split
        from nids_ml.labeling.contracts import strict_json
        temporal=self.manifest['temporal'];gap=temporal['gap_seconds']*10**9
        encoded=handle['metadata'][[row[3] for row in rows]]
        for row,data in zip(rows,encoded):
            meta=strict_json(data.tobytes());core=meta['core']
            rule=self.rules[self.inputs[row[1]]['source_id']]
            times=tuple(core[k] for k in ('anchor_timestamp_ns','nominal_end_timestamp_ns','first_timestamp_ns','last_timestamp_ns'))
            require(tuple(row[8:12])==times and row[12]==rule['publisher_split']
                    and core['capture']['capture_sha256']==rule['sha256']
                    and meta['source_id']==rule['source_id'] and meta['publisher_split']==rule['publisher_split']
                    and meta['policy_sha256']==self.policy_sha and meta['group_id']==row[5]
                    and meta['tensor_sha256']==row[4] and meta['y']==row[6] and meta['attack_type']==row[7],
                    'TEMPORAL_ROW_METADATA_LINEAGE')
            require(row_split(rule['publisher_split'],times[0],times[1],temporal['bounds'][rule['attack_type']],gap)==split,
                    'TEMPORAL_ACTUAL_WINDOW_SPLIT')
