"""Audit official scenario splits using parent connections and exact tensor hashes.
Kiểm split scenario theo connection của dataset cha và checksum tensor chính xác.
"""
from collections import Counter
from contextlib import closing
from pathlib import Path
import shutil
import sqlite3
import os
import time

from nids_ml.flows.mapped import require
from nids_ml.labeling.contracts import canonical_bytes
from nids_ml.labeling.scenario import load_policy
from nids_ml.pcap.decoder import file_sha256,process_rss
from nids_ml.storage.scenario import ScenarioShardReader


def audit_scenario_sources(directories,policy_path,policy_sha,output,*,sqlite_mib=512,progress=None):
    policy,digest=load_policy(policy_path,policy_sha)
    require(type(sqlite_mib) is int and 16<=sqlite_mib<=1536,'SCENARIO_AUDIT_SQLITE_BUDGET')
    require(1<=len(directories)<=6,'SCENARIO_AUDIT_SOURCE_COUNT')
    output=Path(output).absolute()
    require(not output.exists() and not output.is_symlink() and output.parent.is_dir()
            and not any(p.is_symlink() for p in output.parents),'FRESH_SCENARIO_AUDIT_REQUIRED')
    require(shutil.disk_usage(output.parent).free>=(sqlite_mib+1024)*1048576,'SCENARIO_AUDIT_START_RESERVE')
    output.mkdir();db=None;previous=None;seen=set();sources=[];counts=Counter();rows=0
    started=time.monotonic();last_guard=0.;guard_error=[]
    def guard():
        nonlocal last_guard
        now=time.monotonic()
        if now-last_guard<.25:return 0
        last_guard=now
        try:
            require(now-started<=7200,'SCENARIO_AUDIT_TIMEOUT')
            require(shutil.disk_usage(output).free>=1024**3,'SCENARIO_AUDIT_DISK_RESERVE')
            rss=process_rss(os.getpid())
            require(rss is not None and rss<=512*1048576,'SCENARIO_AUDIT_RSS_LIMIT_OR_UNAVAILABLE')
        except (ValueError,OSError) as error:
            guard_error.append(str(error));return 1
        return 0
    try:
        db=sqlite3.connect(output/'audit.sqlite')
        db.execute('PRAGMA page_size=4096');db.execute('PRAGMA max_page_count='+str(sqlite_mib*256))
        db.execute('PRAGMA cache_size=-8192');db.execute('PRAGMA journal_mode=DELETE')
        db.execute('PRAGMA temp_store=FILE')
        previous=db.execute('PRAGMA temp_store_directory').fetchone()
        db.execute('PRAGMA temp_store_directory='+db.execute('SELECT quote(?)',(str(output),)).fetchone()[0])
        db.execute('CREATE TABLE patterns(sha BLOB,g BLOB,y INTEGER,s INTEGER,PRIMARY KEY(sha,g,y,s)) WITHOUT ROWID')
        db.set_progress_handler(guard,20000)
        for directory in directories:
            reader=ScenarioShardReader(directory,policy_path,digest);m=reader.manifest
            require(m['source_id'] not in seen,'DUPLICATE_SCENARIO_CAPTURE')
            seen.add(m['source_id']);observed=0
            for batch in reader.batches(256):
                for sha,group,y in zip(batch['tensor_sha256'],batch['group_id'],batch['y']):
                    rows+=1;observed+=1;require(rows<=30000000,'SCENARIO_AUDIT_SAMPLE_CAP')
                    s=int(m['publisher_split']=='test');counts[m['publisher_split']+':'+str(int(y))]+=1
                    db.execute('INSERT OR IGNORE INTO patterns VALUES(?,?,?,?)',
                        (bytes.fromhex(sha.decode()),bytes.fromhex(group.decode()),int(y),s))
                db.commit()
                require(shutil.disk_usage(output).free>=1024**3,'SCENARIO_AUDIT_DISK_RESERVE')
                if progress and rows%4096==0:progress('scenario_split_audit',rows)
            require(observed==m['samples'],'SCENARIO_AUDIT_OBSERVED_COUNT')
            sources.append(dict(source_id=m['source_id'],manifest_sha256=file_sha256(Path(directory)/'manifest.json'),
                                publisher_split=m['publisher_split'],samples=observed))
        db.commit()
        mixed=db.execute('SELECT count(*) FROM (SELECT sha FROM patterns GROUP BY sha HAVING min(y)!=max(y))').fetchone()[0]
        tensor_overlap=db.execute('SELECT count(*) FROM (SELECT sha FROM patterns GROUP BY sha HAVING min(s)!=max(s))').fetchone()[0]
        db.execute('CREATE INDEX parent_groups ON patterns(g,s)');db.commit()
        group_overlap=db.execute('SELECT count(*) FROM (SELECT g FROM patterns GROUP BY g HAVING min(s)!=max(s))').fetchone()[0]
        all_sources=seen=={rule['source_id'] for rule in policy['sources']}
        both_classes=all(counts[s+':'+str(y)]>0 for s in ('train','test') for y in (0,1))
        result=dict(format='nids-scenario-split-audit/1.0.0',status='COMPLETE_SCENARIO_SPLIT_AUDIT',
            policy_sha256=digest,sources=sources,all_policy_sources_present=all_sources,
            observed_counts=dict(counts),samples=rows,unique_pattern_group_label_split_rows=db.execute('SELECT count(*) FROM patterns').fetchone()[0],
            cross_official_split_tensor_patterns=tensor_overlap,cross_official_split_parent_groups=group_overlap,
            conflicting_label_patterns=mixed,both_classes_in_official_train_and_test=both_classes,
            decision='NEEDS_GLOBAL_CATALOG_AND_VALIDATION_FROM_TRAIN',training_ready=False,training=False,
            exclusions_required=dict(parent_groups=group_overlap,tensor_patterns=tensor_overlap,mixed_labels=mixed),
            policy='Do not reassign official test. Remove conflicting/overlapping training groups in a reviewed catalog; never train directly from these shards.')
        (output/'report.json').write_bytes(canonical_bytes(result))
        return result
    except BaseException as error:
        (output/'FAILED.json').write_bytes(canonical_bytes(dict(status='FAILED_RETAINED',
            error=guard_error[0] if guard_error else str(error),automatic_cleanup=False)))
        raise
    finally:
        if db is not None:
            db.set_progress_handler(None,0)
            if previous:db.execute('PRAGMA temp_store_directory='+db.execute('SELECT quote(?)',(previous[0],)).fetchone()[0])
            else:db.execute("PRAGMA temp_store_directory=''")
            db.close()
