"""Read compact full-training references without replicating parent SQLite indexes."""
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3

from nids_ml.flows.mapped import read_manifest,require
from nids_ml.pcap.decoder import regular_file,file_sha256,file_signature
from nids_ml.storage.partitioned import PartitionReader
from nids_ml.storage.scenario import ScenarioShardReader
from nids_ml.storage.shards import FlowShardReader
from nids_ml.training.full_plan import FORMAT,POLICY,REF,FAMILIES,identity,schedule


class PlanBank:
    def __init__(self,directory,workspace,client,*,policy_path,policy_sha,sources=None):
        from nids_ml.labeling.scenario import load_policy
        self.directory=Path(directory).absolute();self.manifest=read_manifest(self.directory)
        m=self.manifest
        require(m['format']==FORMAT and m['policy']==POLICY and m['training_ready'] is True
                and m['training'] is False and m['plan_id']==identity(m),'FULL_PLAN_IDENTITY')
        require(type(client) is int and client in (0,1) and set(m['clients'])=={'0','1'},'FULL_PLAN_CLIENT')
        self.client=client;self.workspace=Path(workspace).resolve();self.policy_path=policy_path
        self.policy,self.policy_sha=load_policy(policy_path,policy_sha)
        require(len(m['parents'])==2 and [p['kind'] for p in m['parents']]==['mapped-edge','temporal-mqtt']
                and m['parents'][1]['policy_sha256']==self.policy_sha,'FULL_PARENT_PROVENANCE')
        require(m['target']=={'0':'Normal','1':'DDoS'} and m['label_provenance']=='MIXED_EXPLICIT_PER_SOURCE'
                and m['independent_ground_truth_verified'] is False,'FULL_PLAN_TARGET')
        count_map={c:p['train_counts'] for c,p in m['clients'].items()}
        n,rounds=schedule(count_map,m['subset_cap'],m['passes'])
        require(n==m['rounds_per_pass'] and len(rounds)==m['total_rounds'],'FULL_SCHEDULE_IDENTITY')
        self.schedule=rounds
        require(1<=len(m['sources'])<=24 and len({s['dataset_id'] for s in m['sources']})==len(m['sources']),'FULL_SOURCES')
        require(set(m['present_families'])<=set(FAMILIES) and set(m['missing_requested_families'])==set(FAMILIES)-set(m['present_families']),
                'FULL_FAMILY_SCOPE')
        self.inputs={s['dataset_id']:s for s in m['sources']};self.readers={}
        self.overrides={} if sources is None else dict(sources)
        require(set(self.overrides)<=set(self.inputs),'FULL_SOURCE_OVERRIDE')
        p=m['clients'][str(client)];require(p['filename']==f'client-{client}.refs','FULL_REFERENCE_NAME')
        self.path=regular_file(self.directory/p['filename']);self.signature=file_signature(self.path)
        require(self.path.stat().st_size==p['bytes']<=512*1048576 and file_sha256(self.path)==p['sha256'],'FULL_REFERENCE_SHA')
        require(len(p['rounds'])==n,'FULL_ROUND_ENTRY_COUNT')
        end=0
        for slot,entry in enumerate(p['rounds']):
            expected={y:part['samples'] for y,part in rounds[slot]['clients'][str(client)].items()}
            require(entry['class_counts']==expected and entry['samples']==sum(expected.values()) and entry['offset']==end,'FULL_TRAIN_ENTRY')
            end+=entry['samples']*REF.size
        for split in ('validation','test'):
            entry=p['evaluation'][split]
            require(entry['offset']==end and sum(entry['class_counts'].values())==entry['samples']
                    and 0<=entry['samples']<=512000,'FULL_EVALUATION_ENTRY')
            end+=entry['samples']*REF.size
        require(end==p['bytes'],'FULL_REFERENCE_ACCOUNTING')
        require(all(type(i) is int and 0<=i<len(m['sources']) for i in p['sources']),'FULL_SOURCE_INDICES')
        self.entry=p
        self.index=regular_file(self.directory/'manifest.json');self.index_sha=file_sha256(self.index)
        self.index_signature=file_signature(self.index)

    def dataset(self,source_index):
        source=self.manifest['sources'][source_index];key=source['dataset_id']
        if key not in self.readers:
            path=Path(self.overrides.get(key,source['path_hint']))
            if not path.is_absolute():path=self.workspace/path
            require(path.is_dir() and file_sha256(regular_file(path/'manifest.json'))==key,'FULL_LOCAL_SOURCE_IDENTITY')
            parent=source['parent'];require(parent in (0,1),'FULL_SOURCE_PARENT')
            reader=FlowShardReader(path) if parent==0 else ScenarioShardReader(path,self.policy_path,self.policy_sha)
            require(reader.manifest['shards']==source['shards'] and reader.manifest['label_provenance']==source['label_provenance'],
                    'FULL_LOCAL_SOURCE_PROVENANCE')
            if parent==1:
                require(reader.manifest['source_id']==source['source_id'] and reader.manifest['publisher_split']==source['publisher_split'],
                        'FULL_LOCAL_SCENARIO_SOURCE')
            self.readers[key]=reader
        return self.readers[key]

    def preflight(self):
        for i in self.entry['sources']:self.dataset(i)

    def reader(self,split,*,round_id=None,scratch=None,budget=None,progress=None):
        return PlannedReader(self,split,round_id=round_id,scratch=scratch,budget=budget,progress=progress)


class PlannedReader(PartitionReader):
    def __init__(self,bank,split,*,round_id=None,scratch=None,budget=None,progress=None):
        import h5py
        require(split in ('train','validation','test'),'FULL_PLAN_SPLIT')
        if split=='train':
            require(type(round_id) is int and 1<=round_id<=bank.manifest['total_rounds'],'FULL_PLAN_ROUND')
            slot=(round_id-1)%bank.manifest['rounds_per_pass'];entry=bank.entry['rounds'][slot]
        else:
            require(round_id is None,'FULL_EVALUATION_NO_ROUND');entry=bank.entry['evaluation'][split]
        self.bank=bank;self.client=bank.client;self.round_id=round_id;self.split=split;self.entry=entry
        self.workspace=bank.workspace;self.directory=bank.directory;self.index=bank.index;self.signature=bank.index_signature
        self.inputs=bank.inputs;self.readers=bank.readers;self.overrides=bank.overrides
        self.scratch=None if scratch is None else Path(scratch).absolute();self.budget=budget;self.progress=progress
        require(progress is None or callable(progress),'FULL_READER_PROGRESS_CALLBACK')
        self.manifest=dict(bank.manifest,catalog_id=bank.manifest['plan_id'],
                           index=dict(sha256=bank.index_sha),seed=bank.manifest['seed']+(round_id or 0))
        self.source_reader=None
        require(file_signature(bank.path)==bank.signature,'FULL_REFERENCE_CHANGED')
        with bank.path.open('rb') as stream:
            stream.seek(entry['offset']);raw=stream.read(entry['samples']*REF.size)
        require(len(raw)==entry['samples']*REF.size and hashlib.sha256(raw).hexdigest()==entry['sha256']
                and file_signature(bank.path)==bank.signature,'FULL_REFERENCE_SEGMENT_SHA')
        refs=defaultdict(list)
        for sid,source,shard,row in REF.iter_unpack(raw):
            require(source in bank.entry['sources'] and sid//30000001==bank.manifest['sources'][source]['parent'],
                    'FULL_REFERENCE_PARENT_OR_SOURCE')
            refs[source,shard].append((row,sid))
        self.db=sqlite3.connect(':memory:')
        try:
            self.db.execute('CREATE TABLE samples(id INTEGER PRIMARY KEY,dataset TEXT,shard INTEGER,row_no INTEGER,sha TEXT,'
                'group_id TEXT,y INTEGER,family TEXT,owner INTEGER,eligible INTEGER DEFAULT 1,split TEXT,client INTEGER)')
            observed=defaultdict(int);indexed=0;reported=0
            for (source,shard),locations in refs.items():
                reader=bank.dataset(source);require(0<=shard<len(reader.identities),'FULL_REFERENCE_SHARD')
                path,signature,item=reader.identities[shard]
                require(file_signature(path)==signature,'FULL_SOURCE_CHANGED')
                locations.sort();indices=[r for r,_ in locations]
                require(len(set(indices))==len(indices) and indices and 0<=indices[0]<=indices[-1]<item['samples'],'FULL_REFERENCE_ROWS')
                with h5py.File(path,'r') as h:
                    fields={name:h[name][indices] for name in ('y','group_id','tensor_sha256','attack_type')}
                for i,(row,sid) in enumerate(locations):
                    y=int(fields['y'][i]);family=fields['attack_type'][i].decode()
                    require(family in FAMILIES and y==int(family!='Normal'),'FULL_ROW_TARGET')
                    observed[str(y)]+=1
                    self.db.execute('INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,NULL,1,?,?)',
                        (sid,bank.manifest['sources'][source]['dataset_id'],shard,row,fields['tensor_sha256'][i].decode(),
                         fields['group_id'][i].decode(),y,family,split,bank.client if split=='train' else None))
                    indexed+=1
                if self.budget:self.budget.check()
                if progress and (indexed-reported>=8192 or indexed==entry['samples']):
                    progress(dict(stage='plan_reference_index',split=split,client=bank.client,samples=indexed,total=entry['samples']))
                    reported=indexed
            self.db.execute('CREATE UNIQUE INDEX location ON samples(dataset,shard,row_no)');self.db.commit()
            require(dict(observed)==entry['class_counts'],'FULL_REFERENCE_CLASS_COUNTS')
        except BaseException:
            self.db.close();raise

    def dataset(self,dataset_id):
        index=next(i for i,s in enumerate(self.bank.manifest['sources']) if s['dataset_id']==dataset_id)
        return self.bank.dataset(index)

    def counts(self,split,client=None):
        require(split==self.split and client in (None,self.client),'FULL_READER_BOUND_SELECTION')
        return self.entry['class_counts']

    def preflight(self,split,client=None):
        self.counts(split,client)

    @contextmanager
    def batch_index(self):
        if self.budget:self.db.set_progress_handler(self.budget.sqlite_progress,20000)
        try:yield self.db
        finally:self.db.set_progress_handler(None,0)

    def validate_selection(self,handle,rows,split):
        from nids_ml.data.temporal_catalog import row_split
        from nids_ml.labeling.contracts import canonical_bytes
        metadata=handle['metadata'][[r[3] for r in rows]]
        for row,encoded in zip(rows,metadata):
            meta=json.loads(encoded.tobytes());source=self.inputs[row[1]];core=meta['core']
            require(meta['group_id']==row[5] and meta['tensor_sha256']==row[4] and meta['y']==row[6]
                    and meta['attack_type']==row[7] and meta['label_provenance']==source['label_provenance']
                    and meta['independent_ground_truth_verified'] is False,'FULL_ROW_METADATA')
            if source['parent']==1:
                temporal=self.bank.manifest['parents'][1]['temporal']
                require(meta['policy_sha256']==self.bank.policy_sha and meta['source_id']==source['source_id']
                        and meta['publisher_split']==source['publisher_split'],'FULL_SCENARIO_ROW_SOURCE')
                actual=row_split(source['publisher_split'],core['anchor_timestamp_ns'],core['nominal_end_timestamp_ns'],
                                 temporal['bounds'][row[7]],temporal['gap_seconds']*10**9)
                require(actual==split,'FULL_TEMPORAL_ROW_MEMBERSHIP')
            else:
                expected=hashlib.sha256((core['capture']['capture_sha256']+canonical_bytes(core['flow_tuple']).decode()).encode()).hexdigest()
                require(row[5]==expected and core['full_window_complete'] is True,'FULL_MAPPED_ROW_LINEAGE')

    def report(self):
        return dict(plan_id=self.bank.manifest['plan_id'],client=self.client,round_id=self.round_id,
            samples=self.entry['samples'],class_counts=self.entry['class_counts'],
            reference_sha256=self.entry['sha256'],reference_offset=self.entry['offset'],
            policy=POLICY,pool_samples=sum(self.bank.entry['train_counts'].values()))

    def __len__(self):return self.entry['samples']
    def close(self):self.db.close()
