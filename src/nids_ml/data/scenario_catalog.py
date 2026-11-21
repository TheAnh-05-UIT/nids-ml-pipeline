"""Build official-split scenario catalogs with conservative global leakage guards.
Tạo catalog scenario giữ split gốc, kiểm trùng và overlap toàn cục theo connection.
"""
from collections import Counter,OrderedDict
from contextlib import closing
import hashlib
from pathlib import Path
import sqlite3

from nids_ml.data.catalog import SPLITS,audit,class_counts
from nids_ml.flows.mapped import new_output,require
from nids_ml.labeling.contracts import canonical_bytes
from nids_ml.labeling.scenario import PROVENANCE,FAMILIES,load_policy
from nids_ml.pcap.decoder import file_sha256
from nids_ml.storage.scenario import ScenarioShardReader

FORMAT='nids-scenario-connection-catalog/1.0.0'
POLICY=('KEEP_PUBLISHER_TEST_MEMBERSHIP; EXCLUDE_AMBIGUOUS_TENSOR_PATTERNS_AND_MIXED_GROUPS; '
        'EXCLUDE_TRAIN_GROUPS_SEEN_IN_PUBLISHER_TEST; GROUP_VALIDATION_FROM_PUBLISHER_TRAIN_15_PERCENT')
_INGEST_COMMIT_ROWS=8192
_INGEST_PROGRESS_ROWS=4096


def identity(manifest):
    value=dict(format=FORMAT,seed=manifest['seed'],policy_sha256=manifest['policy_sha256'],
        index_sha256=manifest['index']['sha256'],sources=[dict(dataset_id=s['dataset_id'],source_id=s['source_id'])
        for s in manifest['sources']],split_policy=POLICY)
    return 'sha256:'+hashlib.sha256(canonical_bytes(value)).hexdigest()


def audit_catalog(db):
    audit(db)
    require(db.execute("SELECT count(*) FROM samples WHERE eligible=1 AND "
        "((split='test' AND publisher!='test') OR (split!='test' AND publisher!='train'))").fetchone()[0]==0,
        'SCENARIO_OFFICIAL_SPLIT_VIOLATION')
    require(not db.execute('SELECT group_id FROM samples WHERE eligible=1 GROUP BY group_id '
        'HAVING count(DISTINCT y)>1 LIMIT 1').fetchone(),'SCENARIO_MIXED_LABEL_GROUP_RETAINED')
    for split in SPLITS:
        families={r[0] for r in db.execute('SELECT DISTINCT family FROM samples WHERE eligible=1 AND split=?',(split,))}
        require(families==set(FAMILIES),'SCENARIO_FAMILY_COVERAGE:'+split)


def build_scenario_catalog(directories,workspace,output,*,policy_path,policy_sha,seed=42,progress=None):
    """All approved sources required; never replace missing test with training rows.
    Phải đủ source đã duyệt; không lấy train bù vào test bị thiếu.
    """
    require(type(seed) is int and 0<=seed<2**31-1,'SCENARIO_CATALOG_SEED')
    policy,digest=load_policy(policy_path,policy_sha)
    require(len(policy['sources'])==6 and {(r['attack_type'],r['publisher_split']) for r in policy['sources']}
        =={(family,split) for family in FAMILIES for split in ('train','test')},'SCENARIO_SIX_SOURCE_SPLIT_SCOPE')
    require(1<=len(directories)<=6,'SCENARIO_CATALOG_SOURCE_COUNT')
    require(sum(r['records'] for r in policy['sources'])<=30000000,'SCENARIO_DECLARED_RECORD_CAP')
    readers=[];seen=set()
    for directory in directories:
        reader=ScenarioShardReader(directory,policy_path,digest);m=reader.manifest
        require(m['source_id'] not in seen,'DUPLICATE_SCENARIO_SOURCE')
        seen.add(m['source_id']);readers.append((file_sha256(reader.directory/'manifest.json'),reader))
    missing={r['source_id'] for r in policy['sources']}-seen
    require(not missing,'SCENARIO_POLICY_SOURCES_MISSING:'+','.join(sorted(missing)))
    require({r['attack_type'] for r in policy['sources']}==set(FAMILIES),'SCENARIO_POLICY_FAMILY_COVERAGE')
    output,budget=new_output(output,workspace);db=None;previous=None;total=0;sources=[];stats={}
    try:
        db=sqlite3.connect(output/'catalog.sqlite')
        db.executescript('PRAGMA page_size=4096; PRAGMA max_page_count=393216; PRAGMA cache_size=-16384; '
            'PRAGMA journal_mode=DELETE; PRAGMA temp_store=FILE; '
            'CREATE TABLE samples(id INTEGER PRIMARY KEY,dataset TEXT,shard INTEGER,row_no INTEGER,sha TEXT,'
            'group_id TEXT,y INTEGER,family TEXT,owner INTEGER,eligible INTEGER DEFAULT 1,split TEXT,client INTEGER,publisher TEXT); '
            'CREATE UNIQUE INDEX unique_pattern ON samples(sha,group_id,y,publisher); '
            'CREATE TABLE group_seen(g TEXT,p TEXT,y INTEGER,PRIMARY KEY(g,p,y)) WITHOUT ROWID;')
        previous=db.execute('PRAGMA temp_store_directory').fetchone()
        db.execute('PRAGMA temp_store_directory='+db.execute('SELECT quote(?)',(str(output),)).fetchone()[0])
        db.set_progress_handler(budget.sqlite_progress,20000)
        hot=OrderedDict();pending_commit=0;reported=0
        for dataset_id,reader in sorted(readers,key=lambda item:item[0]):
            m=reader.manifest;shard=row=observed=0;labels=Counter()
            with closing(reader.batches(256,metadata=True)) as batches:
                for batch in batches:
                    for i,label in enumerate(batch['y']):
                        total+=1;observed+=1;require(total<=30000000,'SCENARIO_CATALOG_SAMPLE_CAP')
                        y=int(label);group=batch['group_id'][i].decode();sha=batch['tensor_sha256'][i].decode()
                        family=batch['attack_type'][i].decode();publisher=m['publisher_split']
                        labels[str(y)]+=1
                        db.execute('INSERT OR IGNORE INTO group_seen VALUES(?,?,?)',(group,publisher,y))
                        key=(sha,group,y,publisher)
                        if key not in hot:
                            db.execute('INSERT OR IGNORE INTO samples(id,dataset,shard,row_no,sha,group_id,y,family,owner,publisher) '
                                'VALUES(?,?,?,?,?,?,?,?,NULL,?)',(total,dataset_id,shard,row,sha,group,y,family,publisher))
                            hot[key]=None
                            if len(hot)>32768:hot.popitem(last=False)
                        else:hot.move_to_end(key)
                        row+=1
                        if row==m['shards'][shard]['samples']:shard+=1;row=0
                    # Bound transactions, but keep resource checks on every read batch.
                    # Gom commit để giảm fsync thẻ nhớ; vẫn kiểm tài nguyên mỗi batch.
                    pending_commit+=len(batch['y'])
                    if pending_commit>=_INGEST_COMMIT_ROWS:
                        db.commit();pending_commit=0
                    budget.check()
                    if progress and total-reported>=_INGEST_PROGRESS_ROWS:
                        progress('scenario_catalog_ingest',total);reported=total
            require(observed==m['samples'] and dict(labels)==m['label_counts'],'SCENARIO_CATALOG_INPUT_COUNTS')
            require(shard==len(m['shards']) and row==0 and file_sha256(reader.directory/'manifest.json')==dataset_id,
                    'SCENARIO_CATALOG_SOURCE_CHANGED')
            db.commit();pending_commit=0;budget.check()
            if progress:
                progress('scenario_catalog_source_complete',total);reported=total
            try:hint=reader.directory.relative_to(Path(workspace).resolve()).as_posix()
            except ValueError:hint=str(reader.directory)
            sources.append(dict(dataset_id=dataset_id,source_id=m['source_id'],publisher_split=m['publisher_split'],
                                owner=None,path_hint=hint,shards=m['shards']))
        hot.clear();db.commit()
        before=db.execute('SELECT count(*) FROM samples').fetchone()[0]
        stats['input_samples']=total;stats['collapsed_identical_within_group_split']=total-before
        stats['excluded_ambiguous_tensor_patterns']=db.execute('SELECT count(*) FROM (SELECT sha FROM samples '
            'GROUP BY sha HAVING count(DISTINCT group_id)>1 OR count(DISTINCT y)>1)').fetchone()[0]
        db.execute('DELETE FROM samples WHERE sha IN (SELECT sha FROM samples GROUP BY sha '
            'HAVING count(DISTINCT group_id)>1 OR count(DISTINCT y)>1)')
        after=db.execute('SELECT count(*) FROM samples').fetchone()[0]
        stats['excluded_pattern_rows']=before-after;before=after
        stats['excluded_mixed_label_groups']=db.execute('SELECT count(*) FROM (SELECT g FROM group_seen '
            'GROUP BY g HAVING min(y)!=max(y))').fetchone()[0]
        db.execute('DELETE FROM samples WHERE group_id IN (SELECT g FROM group_seen GROUP BY g HAVING min(y)!=max(y))')
        after=db.execute('SELECT count(*) FROM samples').fetchone()[0]
        stats['excluded_mixed_group_rows']=before-after;before=after
        db.execute("DELETE FROM samples WHERE publisher='train' AND group_id IN (SELECT g FROM group_seen WHERE p='test')")
        after=db.execute('SELECT count(*) FROM samples').fetchone()[0]
        stats['excluded_train_rows_parent_seen_in_test']=before-after
        db.commit();db.execute('DROP INDEX unique_pattern');db.execute('DROP TABLE group_seen');db.commit()
        db.create_function('nids_priority',1,lambda g:hashlib.sha256((str(seed)+':'+g).encode()).hexdigest())
        db.executescript('CREATE TABLE groups AS SELECT group_id,min(y) y,'
            "CASE WHEN min(family)=max(family) THEN min(family) ELSE 'MULTI_FAMILY' END family,count(*) n,"
            "nids_priority(group_id) priority,NULL split,NULL client FROM samples WHERE publisher='train' GROUP BY group_id; "
            'CREATE UNIQUE INDEX group_key ON groups(group_id);')
        assigned=Counter();strata=list(db.execute('SELECT y,family,count(*) FROM groups GROUP BY y,family ORDER BY y,family'))
        for y,family,number in strata:
            require(number>=2,'SCENARIO_TOO_FEW_VALIDATION_GROUPS:'+family)
            hold=max(1,number*15//100);updates=[]
            for position,(group,n) in enumerate(db.execute('SELECT group_id,n FROM groups WHERE y=? AND family=? '
                    'ORDER BY priority,group_id',(y,family))):
                split='validation' if position<hold else 'train';client=None
                if split=='train':
                    client=min((0,1),key=lambda c:(assigned[c,family],assigned[c,y],assigned[c,'all'],c))
                    assigned[client,family]+=n;assigned[client,y]+=n;assigned[client,'all']+=n
                updates.append((split,client,group))
                if len(updates)==2048:
                    db.executemany('UPDATE groups SET split=?,client=? WHERE group_id=?',updates);updates=[]
            if updates:db.executemany('UPDATE groups SET split=?,client=? WHERE group_id=?',updates)
        db.execute("UPDATE samples SET split='test',client=NULL WHERE publisher='test'")
        db.execute("UPDATE samples SET split=(SELECT split FROM groups WHERE groups.group_id=samples.group_id),"
            "client=(SELECT client FROM groups WHERE groups.group_id=samples.group_id) WHERE publisher='train'")
        db.commit();db.execute('DROP TABLE groups');db.commit()
        db.executescript('CREATE INDEX by_sha ON samples(eligible,sha); CREATE INDEX by_group ON samples(eligible,group_id,split,client); '
            'CREATE INDEX selection ON samples(eligible,split,client,sha); CREATE UNIQUE INDEX location ON samples(dataset,shard,row_no);')
        db.commit();audit_catalog(db)
        require(db.execute('PRAGMA quick_check').fetchall()==[('ok',)],'SCENARIO_SQLITE_INTEGRITY')
        stats['retained_samples']=db.execute('SELECT count(*) FROM samples').fetchone()[0]
        stats['excluded_samples']=total-stats['retained_samples']
        counts={s:class_counts(db,s) for s in SPLITS}
        clients={str(c):{s:class_counts(db,s,c) for s in SPLITS} for c in (0,1)}
        families=[dict(split=s,family=f,samples=n) for s,f,n in db.execute(
            'SELECT split,family,count(*) FROM samples GROUP BY split,family ORDER BY split,family')]
        budget.check(force=True);db.set_progress_handler(None,0)
        db.execute('PRAGMA temp_store_directory='+db.execute('SELECT quote(?)',(previous[0] if previous else '',)).fetchone()[0])
        db.close();db=None;index=output/'catalog.sqlite'
        result=dict(format=FORMAT,status='COMPLETE_SCENARIO_CATALOG',seed=seed,policy_sha256=digest,sources=sources,
            index=dict(filename=index.name,bytes=index.stat().st_size,sha256=file_sha256(index)),statistics=stats,
            split_counts=counts,client_counts=clients,attack_type_counts=families,label_provenance=PROVENANCE,
            independent_ground_truth_verified=False,training_ready=True,training=False,split_policy=POLICY,
            scope='PUBLISHER_TEST_RETAINED_AFTER_DECLARED_EXCLUSIONS; VALIDATION_ONLY_FROM_TRAIN; CONSERVATIVE_PARENT_GROUP_HOLDOUT',
            source_coverage='ALL_APPROVED_SOURCES_REQUIRED',owner_contract='All sources shared; local copies/overrides required before client training',
            runtime_adapter='SCENARIO_PARTITION_READER; existing mapped-only training/Swarm entrypoints do not accept this catalog',
            native_ARM64='NOT VERIFIED')
        result['catalog_id']=identity(result)
        encoded=canonical_bytes(result)
        require(len(encoded)<=1048576,'SCENARIO_CATALOG_MANIFEST_BYTE_CAP')
        with (output/'manifest.json').open('xb') as f:f.write(encoded)
        return result
    except BaseException as error:
        with (output/'FAILED.json').open('xb') as f:f.write(canonical_bytes(dict(status='FAILED_RETAINED',
            error=str(error) or type(error).__name__,statistics=stats,training=False,automatic_cleanup=False)))
        raise
    finally:
        if db is not None:
            db.set_progress_handler(None,0)
            try:db.execute('PRAGMA temp_store_directory='+db.execute('SELECT quote(?)',(previous[0] if previous else '',)).fetchone()[0])
            finally:db.close()
