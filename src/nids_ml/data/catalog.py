"""Build a disk-backed global deduplication and connection split catalog.
Lập catalog toàn cục trên đĩa: chống trùng tensor, chia theo connection và giao client.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from nids_ml.flows.features import configuration
from nids_ml.flows.mapped import new_output,read_manifest,readonly,require,PROVENANCE
from nids_ml.alignment.large import ATTACK_TYPES
from nids_ml.labeling.contracts import canonical_bytes
from nids_ml.ordering.view import _new_json
from nids_ml.pcap.decoder import regular_file,file_sha256
from nids_ml.storage.shards import FlowShardReader

FORMAT='nids-connection-catalog/1.0.0'
SPLITS=('train','validation','test')
HEX=re.compile('[a-f0-9]{64}')


def identity(manifest):
    value=dict(format=FORMAT,seed=manifest['seed'],index_sha256=manifest['index']['sha256'],
        sources=[dict(dataset_id=s['dataset_id'],owner=s['owner']) for s in manifest['sources']],
        policy='EXCLUDE_CROSS_GROUP_OR_LABEL_DUPLICATES; KEEP_ONE_WITHIN_GROUP; GROUP_HOLDOUT_70_15_15')
    return 'sha256:'+hashlib.sha256(canonical_bytes(value)).hexdigest()


def class_counts(db,split,client=None):
    where='eligible=1 AND split=?'; args=[split]
    if client is not None:
        where+=' AND client=?' if split=='train' else ' AND (owner IS NULL OR owner=?)'
        args.append(client)
    return {str(y):n for y,n in db.execute('SELECT y,count(*) FROM samples WHERE '+where+' GROUP BY y',args)}


def audit(db):
    require(db.execute("SELECT count(*) FROM samples WHERE eligible=1 AND (split NOT IN ('train','validation','test') "
        "OR split IS NULL OR (split='train' AND (client IS NULL OR client NOT IN (0,1))) "
        "OR (split!='train' AND client IS NOT NULL))").fetchone()[0]==0,'INVALID_SPLIT_ASSIGNMENT')
    require(not db.execute('SELECT sha FROM samples WHERE eligible=1 GROUP BY sha HAVING count(*)>1 LIMIT 1').fetchone(),
            'EXACT_TENSOR_DUPLICATE_RETAINED')
    require(not db.execute('SELECT group_id FROM samples WHERE eligible=1 GROUP BY group_id '
        'HAVING count(DISTINCT split)>1 OR count(DISTINCT coalesce(client,-1))>1 LIMIT 1').fetchone(),'CONNECTION_LEAKAGE')
    require(db.execute('SELECT count(*) FROM samples WHERE eligible=1 AND split=\'train\' '
        'AND owner IS NOT NULL AND owner!=client').fetchone()[0]==0,'CLIENT_PLACEMENT_VIOLATION')
    for split in SPLITS:
        require(set(class_counts(db,split))=={'0','1'},'GLOBAL_HOLDOUT_MISSING_CLASS:'+split)
        for client in (0,1):
            require(set(class_counts(db,split,client))=={'0','1'},'CLIENT_OR_LOCAL_HOLDOUT_MISSING_CLASS:'+str(client)+':'+split)


def build_catalog(inputs,workspace,output,*,seed=42,progress=None):
    """inputs are (dataset directory, owner=None/0/1); shared data must exist on both nodes.
    Owner là vị trí dữ liệu, không phải nhãn; không tự đọc dữ liệu từ máy khác.
    """
    require(type(seed) is int and 0<=seed<2**31-1,'CATALOG_SEED')
    require(1<=len(inputs)<=24,'ONE_TO_24_INPUT_DATASETS')
    declared=[]
    for directory,owner in inputs:
        require(owner is None or type(owner) is int and owner in (0,1),'INPUT_OWNER')
        path=regular_file(Path(directory)/'manifest.json')
        declared.append((file_sha256(path),Path(directory).absolute(),owner))
    require(len({d[0] for d in declared})==len(declared),'DUPLICATE_INPUT_DATASET')
    output,budget=new_output(output,workspace);db=None;previous=None
    try:
        db=sqlite3.connect(output/'catalog.sqlite')
        # Register priority UDF on scalar group_id values to avoid a row-by-row Python update pass.
        db.create_function('nids_priority',1,lambda group:hashlib.sha256((str(seed)+':'+group).encode()).hexdigest())
        db.executescript('PRAGMA page_size=4096; PRAGMA max_page_count=393216; PRAGMA cache_size=-16384; '
            'PRAGMA journal_mode=DELETE; PRAGMA temp_store=FILE; '
            'CREATE TABLE samples(id INTEGER PRIMARY KEY,dataset TEXT,shard INTEGER,row_no INTEGER,sha TEXT,'
            'group_id TEXT,y INTEGER,family TEXT,owner INTEGER,eligible INTEGER DEFAULT 0,split TEXT,client INTEGER);')
        previous=db.execute('PRAGMA temp_store_directory').fetchone()
        db.execute('PRAGMA temp_store_directory='+db.execute('SELECT quote(?)',(str(output),)).fetchone()[0])
        db.set_progress_handler(budget.sqlite_progress,20000)
        sources=[];total=0
        for dataset_id,directory,owner in sorted(declared,key=lambda item:item[0]):
            reader=FlowShardReader(directory);m=reader.manifest
            require(canonical_bytes(m['feature_configuration'])==canonical_bytes(configuration(input_kind='decoded_pcap')),
                    'FEATURE_CONFIGURATION_MISMATCH')
            captures={s['context']['capture_sha256'] for s in m['sources']}
            require(sum(len(s['shards']) for s in sources)+len(m['shards'])<=2048,'GLOBAL_SHARD_COUNT_CAP')
            shard=0;row=0;counts=Counter();families_observed=Counter()
            for batch in reader.batches(256,metadata=True):
                for i,y in enumerate(batch['y']):
                    meta=json.loads(batch['metadata'][i].tobytes());group=batch['group_id'][i].decode()
                    sha=batch['tensor_sha256'][i].decode();family=batch['attack_type'][i].decode();y=int(y)
                    require(HEX.fullmatch(group) and HEX.fullmatch(sha),'SAMPLE_HASH_ENCODING')
                    core=meta['core'];capture=core['capture']['capture_sha256']
                    require(capture in captures and core['full_window_complete'] is True
                            and type(meta['y']) is int and meta['y']==y,'UNVERIFIED_SAMPLE_LINEAGE')
                    expected=hashlib.sha256((capture+canonical_bytes(core['flow_tuple']).decode()).encode()).hexdigest()
                    require(group==expected,'CONNECTION_GROUP_IDENTITY')
                    require((y==0 and family=='Normal') or (y==1 and family in (*ATTACK_TYPES,'MULTI_ATTACK')),'NIDS_LABEL_FAMILY')
                    if family=='MULTI_ATTACK':
                        require(type(meta['attack_types']) is list and 2<=len(meta['attack_types'])<=14
                                and all(v in ATTACK_TYPES for v in meta['attack_types']),'MULTI_ATTACK_LINEAGE')
                    db.execute('INSERT INTO samples(dataset,shard,row_no,sha,group_id,y,family,owner) VALUES(?,?,?,?,?,?,?,?)',
                               (dataset_id,shard,row,sha,group,y,family,owner))
                    counts[str(y)]+=1;families_observed[family]+=1;total+=1;row+=1
                    require(total<=8000000,'CATALOG_SAMPLE_CAP')
                    if row==m['shards'][shard]['samples']:shard+=1;row=0
                    if total%4096==0:
                        db.commit();budget.check()
                        if progress:progress('catalog_ingest',total)
            db.commit();budget.check()
            require(dict(counts)==m['label_counts'] and shard==len(m['shards']) and row==0,'INPUT_OBSERVED_COUNTS')
            require(dict(families_observed)==m['attack_type_counts'],'INPUT_FAMILY_COUNTS')
            require(file_sha256(directory/'manifest.json')==dataset_id,'INPUT_MANIFEST_CHANGED')
            try:hint=directory.relative_to(Path(workspace).resolve()).as_posix()
            except ValueError:hint=str(directory)
            sources.append(dict(dataset_id=dataset_id,owner=owner,path_hint=hint,shards=m['shards']))
        db.execute('CREATE INDEX by_sha ON samples(sha)')
        excluded_patterns=db.execute('SELECT count(*) FROM (SELECT 1 FROM samples INDEXED BY by_sha '
            'GROUP BY sha HAVING count(DISTINCT group_id)>1 OR count(DISTINCT y)>1)').fetchone()[0]
        db.execute('UPDATE samples SET eligible=1 WHERE id IN (SELECT min(id) FROM samples INDEXED BY by_sha '
            'GROUP BY sha HAVING count(DISTINCT group_id)=1 AND count(DISTINCT y)=1)')
        db.commit();db.execute('DROP INDEX by_sha');db.commit()
        db.execute('CREATE INDEX by_group ON samples(group_id)')
        require(not db.execute('SELECT 1 FROM samples INDEXED BY by_group GROUP BY group_id '
            'HAVING count(DISTINCT coalesce(owner,-1))>1 LIMIT 1').fetchone(),'GROUP_OWNER_CONFLICT')
        excluded_mixed=db.execute('SELECT count(*) FROM (SELECT 1 FROM samples INDEXED BY by_group '
            'GROUP BY group_id HAVING count(DISTINCT y)>1)').fetchone()[0]
        if excluded_mixed:
            db.execute('UPDATE samples SET eligible=0 WHERE eligible=1 AND group_id IN '
                '(SELECT group_id FROM samples INDEXED BY by_group GROUP BY group_id HAVING count(DISTINCT y)>1)')
        db.commit();db.execute('DROP INDEX by_group');db.commit()
        distinct_families=[f for (f,) in db.execute('SELECT DISTINCT family FROM samples ORDER BY family')]
        db.execute('DELETE FROM samples WHERE eligible=0');db.commit();db.execute('VACUUM')
        db.executescript('CREATE TABLE groups AS SELECT group_id,min(y) y,min(owner) owner,count(*) n,'
            'nids_priority(group_id) priority,NULL split,NULL client FROM samples WHERE eligible=1 GROUP BY group_id; '
            'CREATE UNIQUE INDEX group_key ON groups(group_id);')
        db.commit();assigned=Counter()
        for owner in (0,1,None):
            for y in (0,1):
                count=db.execute('SELECT count(*) FROM groups WHERE y=? AND owner IS ?',(y,owner)).fetchone()[0]
                if count==0:continue
                require(count>=(4 if owner is None else 3),'INSUFFICIENT_CONNECTION_GROUPS:'+str(y)+':'+str(owner))
                hold=max(1,count*15//100);train=count-2*hold;batch_updates=[]
                for number,(group,n) in enumerate(db.execute('SELECT group_id,n FROM groups WHERE y=? AND owner IS ? '
                    'ORDER BY priority,group_id',(y,owner))):
                    split='train' if number<train else 'validation' if number<train+hold else 'test'
                    client=None
                    if split=='train':
                        client=owner if owner is not None else min((0,1),key=lambda c:(assigned[c,y],c))
                        assigned[client,y]+=n
                    batch_updates.append((split,client,group))
                    if len(batch_updates)==2048:
                        db.executemany('UPDATE groups SET split=?,client=? WHERE group_id=?',batch_updates)
                        batch_updates=[]
                if batch_updates:
                    db.executemany('UPDATE groups SET split=?,client=? WHERE group_id=?',batch_updates)
        group_count=db.execute('SELECT count(*) FROM groups').fetchone()[0]
        db.execute('UPDATE samples SET split=(SELECT split FROM groups WHERE groups.group_id=samples.group_id),'
                   'client=(SELECT client FROM groups WHERE groups.group_id=samples.group_id) WHERE eligible=1')
        db.commit();db.execute('DROP TABLE groups');db.commit()
        db.executescript('CREATE INDEX by_sha ON samples(eligible,sha); '
            'CREATE INDEX by_group ON samples(eligible,group_id,split,client); '
            'CREATE INDEX selection ON samples(eligible,split,client,sha); '
            'CREATE UNIQUE INDEX location ON samples(dataset,shard,row_no);')
        db.commit();audit(db);budget.check(force=True)
        counts={s:class_counts(db,s) for s in SPLITS}
        clients={str(c):{s:class_counts(db,s,c) for s in SPLITS} for c in (0,1)}
        families=[dict(split=s,family=f,samples=n) for s,f,n in db.execute(
            'SELECT split,family,count(*) FROM samples WHERE eligible=1 GROUP BY split,family ORDER BY split,family')]
        family_coverage={f:{s:0 for s in SPLITS} for f in distinct_families}
        for row in families:family_coverage[row['family']][row['split']]=row['samples']
        stats=dict(input_samples=total,retained_samples=db.execute('SELECT count(*) FROM samples WHERE eligible=1').fetchone()[0],
            excluded_cross_group_or_conflicting_tensor_patterns=excluded_patterns,
            excluded_mixed_label_groups=excluded_mixed,groups=group_count)
        stats['excluded_samples']=total-stats['retained_samples']
        db.set_progress_handler(None,0)
        db.execute('PRAGMA temp_store_directory='+db.execute('SELECT quote(?)',(previous[0] if previous else '',)).fetchone()[0])
        db.close();db=None
        index=output/'catalog.sqlite'
        result=dict(format=FORMAT,status='COMPLETE_CONNECTION_CATALOG',seed=seed,sources=sources,
            index=dict(filename=index.name,bytes=index.stat().st_size,sha256=file_sha256(index)),statistics=stats,
            split_counts=counts,client_counts=clients,attack_type_counts=families,family_coverage=family_coverage,
            label_provenance=PROVENANCE,independent_ground_truth_verified=False,training_ready=True,
            policy='EXCLUDE_CROSS_GROUP_OR_LABEL_DUPLICATES; KEEP_ONE_WITHIN_GROUP; GROUP_HOLDOUT_70_15_15',
            scope='WITHIN_CAPTURE_CONNECTION_HOLDOUT; not unseen-capture/attack generalization',
            owner_contract='0/1=local; null=shared available on both; validation/test shared rows must not be counted twice',
            native_ARM64='NOT VERIFIED',training=False)
        result['catalog_id']=identity(result);_new_json(output/'manifest.json',result)
        return result
    except Exception as error:
        _new_json(output/'FAILED.json',dict(status='FAILED_RETAINED',error=str(error),automatic_cleanup=False));raise
    finally:
        if db is not None:
            try:db.execute('PRAGMA temp_store_directory='+db.execute('SELECT quote(?)',(previous[0] if previous else '',)).fetchone()[0])
            finally:db.close()
