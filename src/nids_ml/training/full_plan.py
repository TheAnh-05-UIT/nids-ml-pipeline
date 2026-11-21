"""Compile immutable parent catalogs into small, disjoint full-pool references.

No HDF5 copies or new large persistent catalog. Parent split/client assignments
stay fixed; all cross-parent tensor duplicates are purged before plan generation.
Shared validation/test rows are evaluated by client 0 only, local rows by owner.
"""
from collections import Counter
import hashlib
import math
from pathlib import Path
import sqlite3
import struct

from nids_ml.flows.mapped import new_output, readonly, read_manifest, require
from nids_ml.labeling.contracts import canonical_bytes
from nids_ml.pcap.decoder import file_sha256
from nids_ml.storage.partitioned import PartitionReader, TemporalScenarioPartitionReader

FORMAT='nids-full-pool-reference-plan/1.0.0'
POLICY='KEEP_PARENT_SPLITS_AND_CLIENTS; PURGE_ALL_CROSS_PARENT_TENSOR_DUPLICATES; STRATIFIED_COMPLETE_POOL; SHARD_LOCAL_ORDER'
FAMILIES=('Normal','DDoS_HTTP','DDoS_TCP','DDoS_UDP','DDoS_ICMP',
          'MQTT_DDoS_Connect_Flood','MQTT_DDoS_Publish_Flood')
REF=struct.Struct('<IBHH')
MAX_ROUNDS=4096


def identity(m):
    value={k:v for k,v in m.items() if k!='plan_id'}
    return 'sha256:'+hashlib.sha256(canonical_bytes(value)).hexdigest()


def span(n, rounds, slot):
    """Complete balanced contiguous slices; no wrap, padding, replay or omitted tail."""
    q,r=divmod(n,rounds)
    return slot*q+min(slot,r),q+int(slot<r)


def schedule(counts, cap=4096, passes=1):
    require(type(cap) is int and 2<=cap<=4096 and type(passes) is int and 1<=passes<=10,'FULL_SCHEDULE_BUDGET')
    require(set(counts)=={'0','1'} and all(set(v)=={'0','1'} and
        all(type(n) is int and n>0 for n in v.values()) for v in counts.values()),'FULL_CLIENT_CLASSES')
    rounds=max(math.ceil(sum(v.values())/cap) for v in counts.values())
    while any(sum(math.ceil(n/rounds) for n in v.values())>cap for v in counts.values()):rounds+=1
    require(rounds*passes<=MAX_ROUNDS and all(min(v.values())>=rounds for v in counts.values()),
            'FULL_ROUND_COVERAGE_OR_RARE_CLASS_BUDGET')
    result=[]
    for epoch in range(passes):
        for slot in range(rounds):
            peers={c:{y:dict(zip(('start','samples'),span(n,rounds,slot))) for y,n in ys.items()}
                   for c,ys in counts.items()}
            require(all(2<=sum(v['samples'] for v in p.values())<=cap for p in peers.values()),'FULL_ROUND_SAMPLE_CAP')
            result.append(dict(round_id=len(result)+1,pass_id=epoch+1,slot=slot,clients=peers))
    return rounds,result


def compile_plan(edge_catalog,mqtt_catalog,workspace,output,*,policy_path,policy_sha,
                 cap=4096,passes=1,seed=42,progress=None):
    require(type(seed) is int and 0<=seed<2**31-1,'FULL_PLAN_SEED')
    # Existing catalog audits remain authoritative; missing/raw sources cannot be substituted.
    if progress:progress('full_plan_parent_audit_started',0)
    edge=PartitionReader(edge_catalog,workspace)
    if progress:progress('full_plan_edge_audit_complete',0)
    mqtt=TemporalScenarioPartitionReader(mqtt_catalog,workspace,policy_path=policy_path,policy_sha=policy_sha)
    if progress:progress('full_plan_mqtt_audit_complete',0)
    parents=[edge,mqtt]
    output,budget=new_output(output,workspace);db=None;previous=None;sources=[];descriptors=[];total=0;stats={}
    def emit(stage,n):
        budget.check()
        if progress:progress(stage,n)
    try:
        db=sqlite3.connect(output/'selection.sqlite')
        db.executescript('PRAGMA page_size=4096; PRAGMA max_page_count=262144; PRAGMA cache_size=-16384; '
            'PRAGMA journal_mode=DELETE; PRAGMA temp_store=FILE; '
            'CREATE TABLE rows(id INTEGER PRIMARY KEY,sha BLOB,g BLOB,y INTEGER,family TEXT,split TEXT,'
            'client INTEGER,source INTEGER,shard INTEGER,row_no INTEGER,block BLOB,rank INTEGER);')
        previous=db.execute('PRAGMA temp_store_directory').fetchone()
        db.execute('PRAGMA temp_store_directory='+db.execute('SELECT quote(?)',(str(output),)).fetchone()[0])
        db.set_progress_handler(budget.sqlite_progress,20000)
        source_ids={};blocks={};reported=0
        for parent_no,parent in enumerate(parents):
            manifest=parent.manifest
            try:hint=parent.directory.relative_to(Path(workspace).resolve()).as_posix()
            except ValueError:hint=str(parent.directory)
            descriptor=dict(kind='mapped-edge' if parent_no==0 else 'temporal-mqtt',
                catalog_id=manifest['catalog_id'],manifest_sha256=file_sha256(parent.directory/'manifest.json'),
                index=manifest['index'],path_hint=hint,scope=manifest['scope'],label_provenance=manifest['label_provenance'])
            if parent_no==1:descriptor.update(temporal=manifest['temporal'],policy_sha256=policy_sha)
            descriptors.append(descriptor)
            with readonly(parent.index) as original:
                for row in original.execute('SELECT id,dataset,shard,row_no,sha,group_id,y,family,split,client,owner '
                        'FROM samples WHERE eligible=1 ORDER BY id'):
                    sid,dataset,shard,position,sha,group,y,family,split,client,owner=row
                    if family not in FAMILIES:
                        stats['excluded_non_ddos_parent_rows']=stats.get('excluded_non_ddos_parent_rows',0)+1
                        continue
                    require(split in ('train','validation','test') and y==int(family!='Normal'),'FULL_PARENT_LABEL_SPLIT')
                    key=(parent_no,dataset)
                    if key not in source_ids:
                        require(len(sources)<24,'FULL_SOURCE_CAP')
                        source_ids[key]=len(sources)
                        item=dict(parent=parent_no,**parent.inputs[dataset])
                        item['label_provenance']=manifest['label_provenance'];sources.append(item)
                    source=source_ids[key]
                    if split!='train':client=0 if owner is None else owner
                    require(type(client) is int and client in (0,1) and type(sid) is int
                            and 0<sid<=30000000 and 0<=shard<8192 and 0<=position<4096,'FULL_REFERENCE_FIELDS')
                    block_key=(source,shard)
                    if block_key not in blocks:
                        blocks[block_key]=hashlib.sha256(f'{seed}:{dataset}:{shard}'.encode()).digest()[:8]
                    global_id=parent_no*30000001+sid
                    db.execute('INSERT INTO rows VALUES(?,?,?,?,?,?,?,?,?,?,?,NULL)',
                        (global_id,bytes.fromhex(sha),bytes.fromhex(group),y,family,split,client,source,shard,position,blocks[block_key]))
                    total+=1
                    if total%8192==0:
                        db.commit();emit('full_plan_catalog_scan',total);reported=total
            db.commit();emit('full_plan_parent_complete',total)
            require(file_sha256(parent.index)==manifest['index']['sha256'],'FULL_PARENT_INDEX_CHANGED')
        stats['input_selected_rows']=total
        db.execute('CREATE INDEX by_sha ON rows(sha)');db.commit()
        db.execute('DELETE FROM rows WHERE sha IN (SELECT sha FROM rows GROUP BY sha HAVING count(*)>1)')
        retained=db.execute('SELECT count(*) FROM rows').fetchone()[0]
        stats['excluded_cross_parent_duplicate_rows']=total-retained
        require(not db.execute('SELECT sha FROM rows GROUP BY sha HAVING count(*)>1 LIMIT 1').fetchone(),'FULL_GLOBAL_TENSOR_OVERLAP')
        # No parent connection can be assigned to two training clients.
        require(not db.execute("SELECT g FROM rows WHERE split='train' GROUP BY g HAVING count(DISTINCT client)>1 LIMIT 1").fetchone(),
                'FULL_TRAIN_GROUP_PLACEMENT')
        counts={str(c):{str(y):n for y,n in db.execute("SELECT y,count(*) FROM rows WHERE split='train' AND client=? GROUP BY y",(c,))}
                for c in (0,1)}
        rounds_per_pass,rounds=schedule(counts,cap,passes)
        db.execute('CREATE TABLE ranked AS SELECT id,row_number() OVER (PARTITION BY client,y ORDER BY block,source,shard,row_no,id)-1 n '
                   "FROM rows WHERE split='train'")
        db.execute('CREATE UNIQUE INDEX ranked_id ON ranked(id)')
        db.execute("UPDATE rows SET rank=(SELECT n FROM ranked WHERE ranked.id=rows.id) WHERE split='train'")
        db.commit();db.execute('DROP TABLE ranked');db.commit()
        db.execute('CREATE INDEX selected ON rows(split,client,y,rank)');db.commit()
        all_counts={s:{str(y):n for y,n in db.execute('SELECT y,count(*) FROM rows WHERE split=? GROUP BY y',(s,))}
                    for s in ('train','validation','test')}
        families=[dict(split=s,family=f,samples=n) for s,f,n in db.execute('SELECT split,family,count(*) FROM rows GROUP BY split,family ORDER BY split,family')]
        require(all(set(v)=={'0','1'} for v in all_counts.values()),'FULL_GLOBAL_SPLIT_CLASSES')
        present=sorted({r['family'] for r in families})
        require('DDoS_TCP' in present and 'DDoS_ICMP' in present and all(f in present for f in FAMILIES[-2:]),'FULL_REQUIRED_READY_SOURCES')
        clients={}
        for client in (0,1):
            path=output/f'client-{client}.refs';entries=[];position=0
            with path.open('xb') as stream:
                # Store each pass once; later passes reference the same complete slices.
                for record in rounds[:rounds_per_pass]:
                    rows=[]
                    for y,part in record['clients'][str(client)].items():
                        rows.extend(db.execute("SELECT id,source,shard,row_no FROM rows WHERE split='train' AND client=? AND y=? AND rank>=? AND rank<?",
                            (client,int(y),part['start'],part['start']+part['samples'])))
                    rows.sort(key=lambda r:(r[1],r[2],r[3]))
                    raw=b''.join(REF.pack(*r) for r in rows);stream.write(raw)
                    entries.append(dict(offset=position,samples=len(rows),sha256=hashlib.sha256(raw).hexdigest(),
                                        class_counts={y:p['samples'] for y,p in record['clients'][str(client)].items()}))
                    position+=len(raw);emit('full_plan_round_references',record['round_id'])
                evaluation={}
                for split in ('validation','test'):
                    start=position;h=hashlib.sha256();n=0;labels=Counter()
                    for sid,source,shard,row_no,y in db.execute('SELECT id,source,shard,row_no,y FROM rows WHERE split=? AND client=? ORDER BY source,shard,row_no',(split,client)):
                        raw=REF.pack(sid,source,shard,row_no);stream.write(raw);h.update(raw);n+=1;position+=REF.size;labels[str(y)]+=1
                    evaluation[split]=dict(offset=start,samples=n,sha256=h.hexdigest(),class_counts=dict(labels))
            require(sum(e['samples'] for e in entries)==sum(counts[str(client)].values()),'FULL_REFERENCE_POOL_ACCOUNTING')
            clients[str(client)]=dict(filename=path.name,bytes=path.stat().st_size,sha256=file_sha256(path),train_counts=counts[str(client)],
                rounds=entries,evaluation=evaluation,
                sources=[n for (n,) in db.execute('SELECT DISTINCT source FROM rows WHERE client=? ORDER BY source',(client,))])
        stats.update(retained_samples=retained)
        db.set_progress_handler(None,0)
        db.execute('PRAGMA temp_store_directory='+db.execute('SELECT quote(?)',(previous[0] if previous else '',)).fetchone()[0])
        db.close();db=None
        (output/'selection.sqlite').unlink()  # Only this newly created, private intermediate.
        result=dict(format=FORMAT,policy=POLICY,seed=seed,parents=descriptors,sources=sources,clients=clients,
            rounds_per_pass=rounds_per_pass,passes=passes,total_rounds=len(rounds),subset_cap=cap,
            statistics=stats,split_counts=all_counts,attack_type_counts=families,
            family_coverage={f:{s:sum(r['samples'] for r in families if r['family']==f and r['split']==s)
                                for s in ('train','validation','test')} for f in present},
            target={'0':'Normal','1':'DDoS'},label_provenance='MIXED_EXPLICIT_PER_SOURCE',
            independent_ground_truth_verified=False,requested_families=list(FAMILIES),present_families=present,
            missing_requested_families=sorted(set(FAMILIES)-set(present)),
            evaluation_placement='SHARED_ROWS_CLIENT_0_ONLY; LOCAL_ROWS_ORIGINAL_OWNER; SUM_DISJOINT_METRICS_ONCE',
            training=False,training_ready=True,scope='EDGE_PARENT_CONNECTION_HOLDOUT_PLUS_MQTT_PER_SCENARIO_TEMPORAL_HOLDOUT; NOT_IDENTICAL_PROTOCOLS')
        result['plan_id']=identity(result)
        raw=canonical_bytes(result);require(len(raw)<=2*1048576,'FULL_PLAN_MANIFEST_CAP')
        with (output/'manifest.json').open('xb') as stream:stream.write(raw)
        return result
    except BaseException as error:
        with (output/'FAILED.json').open('xb') as stream:
            stream.write(canonical_bytes(dict(status='FAILED_RETAINED',error=str(error),statistics=stats,training=False,automatic_cleanup=False)))
        raise
    finally:
        if db is not None:
            db.set_progress_handler(None,0)
            try:db.execute('PRAGMA temp_store_directory='+db.execute('SELECT quote(?)',(previous[0] if previous else '',)).fetchone()[0])
            finally:db.close()
