"""Validate scenario splits using tiny synthetic flow sources, never real MQTT metrics.
Kiểm split scenario bằng flow giả nhỏ; không dùng số liệu fixture cho dataset thật.
"""
from contextlib import closing
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tests'))
from flow_extraction_fixtures import packet
from nids_ml.flows.features import CaptureContext,extract_features
from nids_ml.flows.streaming import SourceRecord,WindowSpool
from nids_ml.labeling.contracts import canonical_bytes
from nids_ml.labeling.scenario import PROVENANCE,FAMILIES,label_tensor
from nids_ml.storage.scenario import PIPE_FORMAT,pipe_header,write_frame,export_scenario_pipe
from nids_ml.data.scenario_catalog import build_scenario_catalog,identity
from nids_ml.storage.partitioned import PartitionReader,ScenarioPartitionReader
from nids_ml.pcap.decoder import file_sha256


class ScenarioCatalogTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.work=Path(self.tmp.name);(self.work/'artifacts/runs').mkdir(parents=True)
        mock=patch('shutil.disk_usage',return_value=SimpleNamespace(free=10*1024**3))
        mock.start();self.addCleanup(mock.stop)
        mock=patch('nids_ml.alignment.full.process_rss',return_value=1024)
        mock.start();self.addCleanup(mock.stop)

    def sources(self,mode='base'):
        self.policy=dict(format='nids-scenario-label-policy/1.0.0',dataset_id='synthetic-catalog-only',
            label_provenance=PROVENANCE,independent_ground_truth_verified=False,
            approval=dict(status='APPROVED_BY_OPERATOR',record_id='SYNTHETIC_NOT_REAL_APPROVAL',
                          evidence='Only synthetic unit-test fixtures, no real packet annotations.'),sources=[])
        rows={}
        for f,family in enumerate(FAMILIES):
            for split in ('train','test'):
                source=f'family-{f}-{split}';base=100+f*100+(1000 if split=='test' else 0)
                values=[(base+i,base+i,0) for i in range(6)]
                rows[source]=values
        if mode=='exclusions':
            rows['family-0-train'] += [(100,100,1),(150,101,1),(1100,999,1),(999,998,1)]
            rows['family-1-train'] += [(999,997,1)]
        if mode=='overlap':rows['family-0-train']=[(1100+i,9900+i,0) for i in range(6)]
        if mode=='one-group':rows['family-0-train']=rows['family-0-train'][:1]
        for source,values in rows.items():
            f=int(source.split('-')[1]);split=source.split('-')[2]
            self.policy['sources'].append(dict(source_id=source,file_name=source+'.pcap',
                sha256=hashlib.sha256(canonical_bytes(dict(source=source,rows=values))).hexdigest(),
                size_bytes=1000,records=len(values),publisher_split=split,y=int(f>0),
                attack_type=FAMILIES[f],max_lateness_ns=0))
        self.policy_path=self.work/'policy.json';self.policy_path.write_bytes(canonical_bytes(self.policy))
        self.policy_sha=file_sha256(self.policy_path);directories=[]
        for rule in self.policy['sources']:
            source=rule['source_id'];ctx=CaptureContext('capture-'+source,rule['sha256'],'synthetic-run','whole-capture')
            events=[]
            for ordinal,(flow,ack,window) in enumerate(rows[source],1):
                value=packet(ordinal=ordinal,flow=flow,timestamp_ns=window*11*10**9,capture=ctx.to_dict())
                value.update(schema_version='1.1.0',input_kind='decoded_pcap');value['tcp']['ack']=ack
                feature=extract_features(value,input_kind='decoded_pcap')
                events.append(SourceRecord(ordinal,value['timestamp_ns'],'FEATURE',feature))
            engine=WindowSpool(ctx,self.work/(source+'-spool'),max_lateness_ns=0)
            stream=io.BytesIO();write_frame(stream,b'H',canonical_bytes(pipe_header(self.policy,self.policy_sha,rule)))
            digest=hashlib.sha256();count=0
            for tensor in engine.run_records(events,input_complete=True):
                payload=tensor.tensor_bytes+canonical_bytes(label_tensor(tensor,self.policy,self.policy_sha,rule))
                digest.update(write_frame(stream,b'T',payload));count+=1
            footer=dict(format=PIPE_FORMAT,policy_sha256=self.policy_sha,source_id=source,source_complete=True,
                tensors=count,tensor_frames_sha256=digest.hexdigest(),stream=dict(engine.stats,status=engine.status),
                decoder=dict(source_sha256=rule['sha256'],child_returncode=0,
                             stats=dict(source_complete=True,decoded_records=rule['records'])))
            write_frame(stream,b'E',canonical_bytes(footer));stream.seek(0)
            output=self.work/'artifacts/runs'/source
            export_scenario_pipe(stream,self.policy_path,self.policy_sha,source,output,max_output_mib=16)
            directories.append(output)
        self.directories=directories

    def build(self,name='catalog',directories=None,seed=42):
        return build_scenario_catalog(self.directories if directories is None else directories,
            self.work,self.work/'artifacts/runs'/name,policy_path=self.policy_path,policy_sha=self.policy_sha,seed=seed)

    def reader(self,name='catalog',**kwargs):
        return ScenarioPartitionReader(self.work/'artifacts/runs'/name,self.work,
            policy_path=self.policy_path,policy_sha=self.policy_sha,**kwargs)

    def test_official_test_validation_and_two_client_membership(self):
        self.sources();m=self.build();reader=self.reader()
        self.assertEqual(m['statistics']['input_samples'],36)
        self.assertEqual(m['split_counts'],dict(train={'0':5,'1':10},validation={'0':1,'1':2},test={'0':6,'1':12}))
        with closing(sqlite3.connect(reader.index)) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM samples WHERE (split='test')!=(publisher='test')").fetchone()[0],0)
            self.assertFalse(db.execute('SELECT group_id FROM samples GROUP BY group_id HAVING count(DISTINCT split)>1').fetchall())
        for client in (None,0,1):
            for split in ('train','validation','test'):
                with closing(reader.batches(split,client,batch_size=2)) as batches:
                    observed=[int(y) for batch in batches for y in batch['y'][:,0]]
                self.assertEqual({str(y):observed.count(y) for y in set(observed)},reader.counts(split,client))
        with self.assertRaisesRegex(ValueError,'READY_NIDS_CATALOG_REQUIRED'):PartitionReader(reader.directory,self.work)

    def test_reverse_input_order_is_deterministic_and_no_overwrite(self):
        self.sources();first=self.build();second=self.build('reverse',list(reversed(self.directories)))
        self.assertEqual(first['catalog_id'],second['catalog_id'])
        with self.assertRaises(ValueError):self.build()

    def test_larger_transactions_preserve_membership_and_batch_resource_checks(self):
        self.sources('exclusions')
        from nids_ml.data import scenario_catalog as module
        from nids_ml.alignment.full import Budget
        connect=sqlite3.connect;reader_type=module.ScenarioShardReader;check=Budget.check
        def small_batches(*args):
            reader=reader_type(*args);batches=reader.batches
            reader.batches=lambda size=256,metadata=False:batches(2,metadata=metadata)
            return reader
        outputs=[];commits=[]
        for limit,name in ((1,'per-batch'),(8,'grouped-commit')):
            statements=[]
            def traced(*args,**kwargs):
                db=connect(*args,**kwargs);db.set_trace_callback(statements.append)
                self.assertEqual(db.execute('PRAGMA synchronous').fetchone()[0],2)
                self.assertEqual(db.execute('PRAGMA journal_mode').fetchone()[0],'delete')
                return db
            with patch.object(module,'_INGEST_COMMIT_ROWS',limit), \
                 patch.object(module,'ScenarioShardReader',side_effect=small_batches), \
                 patch.object(module.sqlite3,'connect',side_effect=traced), \
                 patch.object(Budget,'check',autospec=True,side_effect=check) as checked:
                result=self.build(name)
                self.assertGreaterEqual(checked.call_count,18)
            with closing(connect(self.work/'artifacts/runs'/name/'catalog.sqlite')) as db:
                rows=db.execute('SELECT id,dataset,shard,row_no,sha,group_id,y,family,publisher,split,client '
                                'FROM samples ORDER BY id').fetchall()
            outputs.append((result['statistics'],result['split_counts'],result['client_counts'],rows))
            commits.append(sum(s=='COMMIT' for s in statements))
        self.assertEqual(outputs[0],outputs[1])
        self.assertLess(commits[1],commits[0])

    def test_progress_crosses_source_boundaries_without_exact_modulo_match(self):
        self.sources();events=[]
        with patch('nids_ml.data.scenario_catalog._INGEST_PROGRESS_ROWS',5):
            build_scenario_catalog(self.directories,self.work,self.work/'artifacts/runs/progress',
                policy_path=self.policy_path,policy_sha=self.policy_sha,
                progress=lambda stage,count:events.append((stage,count)))
        ingested=[n for stage,n in events if stage=='scenario_catalog_ingest']
        completed=[n for stage,n in events if stage=='scenario_catalog_source_complete']
        self.assertEqual(ingested,[6,12,18,24,30,36])
        self.assertEqual(completed,[6,12,18,24,30,36])

    def test_duplicate_patterns_mixed_groups_and_test_overlap_excluded(self):
        self.sources('exclusions');m=self.build();s=m['statistics']
        self.assertEqual(s['collapsed_identical_within_group_split'],1)
        self.assertEqual(s['excluded_pattern_rows'],2)
        self.assertEqual(s['excluded_mixed_group_rows'],2)
        self.assertEqual(s['excluded_train_rows_parent_seen_in_test'],1)
        self.assertEqual(m['split_counts']['test'],{'0':6,'1':12})

    def test_missing_duplicate_sources_and_wrong_policy_fail_before_output(self):
        self.sources()
        for values,reason in [(self.directories[:-1],'SOURCES_MISSING'),
                              ([*self.directories[:-1],self.directories[0]],'DUPLICATE_SCENARIO_SOURCE')]:
            with self.assertRaisesRegex(ValueError,reason):self.build(directories=values)
            self.assertFalse((self.work/'artifacts/runs/catalog').exists())
        with self.assertRaisesRegex(ValueError,'POLICY_SHA256'):
            build_scenario_catalog(self.directories,self.work,self.work/'artifacts/runs/nope',
                                   policy_path=self.policy_path,policy_sha='0'*64)

    def test_all_benign_train_overlap_fails_without_publishing_ready_catalog(self):
        self.sources('overlap')
        with self.assertRaisesRegex(ValueError,'MISSING_CLASS'):self.build()
        output=self.work/'artifacts/runs/catalog'
        self.assertFalse((output/'manifest.json').exists())
        self.assertEqual(json.loads((output/'FAILED.json').read_bytes())['statistics']['excluded_train_rows_parent_seen_in_test'],6)

    def test_insufficient_validation_groups_fail_closed(self):
        self.sources('one-group')
        with self.assertRaisesRegex(ValueError,'TOO_FEW_VALIDATION_GROUPS'):self.build()

    def test_seeded_shuffle_bounded_batches_and_early_close(self):
        self.sources();self.build();scratch=self.work/'shuffle';scratch.mkdir()
        reader=self.reader(scratch=scratch)
        def ids(epoch):
            with closing(reader.batches('train',batch_size=3,epoch=epoch)) as batches:
                result=[]
                for batch in batches:
                    self.assertLessEqual(len(batch['X']),3);result.extend(batch['sample_id'])
                return result
        self.assertEqual(ids(0),ids(0));self.assertEqual(set(ids(0)),set(ids(1)))
        self.assertNotEqual(ids(0),ids(1))
        batches=reader.batches('train',batch_size=1);next(batches);batches.close()
        self.assertEqual(len(ids(0)),15)

    def test_forged_publisher_membership_rejected_even_with_refreshed_hash(self):
        self.sources();self.build();output=self.work/'artifacts/runs/catalog'
        with closing(sqlite3.connect(output/'catalog.sqlite')) as db:
            db.execute("UPDATE samples SET publisher='train' WHERE split='test'")
            db.commit()
        p=output/'manifest.json';m=json.loads(p.read_bytes());m['index']['sha256']=file_sha256(output/'catalog.sqlite')
        m['catalog_id']=identity(m);p.write_bytes(canonical_bytes(m))
        with self.assertRaisesRegex(ValueError,'OFFICIAL_SPLIT_VIOLATION'):self.reader()

    def test_portable_override_and_wrong_source_identity(self):
        self.sources();m=self.build()
        overrides={s['dataset_id']:self.work/s['path_hint'] for s in m['sources']}
        reader=self.reader(sources=overrides)
        self.assertEqual(sum(len(b['X']) for b in reader.batches('test')),18)
        key=next(iter(overrides));overrides[key]=self.work/'absent'
        bad=self.reader(sources=overrides)
        with self.assertRaisesRegex(ValueError,'LOCAL_DATASET_MISSING'):bad.dataset(key)

    def test_cli_inspection_and_incomplete_catalog(self):
        self.sources();self.build()
        common=[sys.executable,'-X','utf8','-B',str(ROOT/'scripts/run_training.py')]
        args=['--workspace',str(self.work),'--policy',str(self.policy_path),'--policy-sha256',self.policy_sha]
        result=subprocess.run([*common,'scenario-inspect','--catalog',str(self.work/'artifacts/runs/catalog'),*args],
                              capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'],'PASS_SCENARIO_CATALOG_INSPECTION')
        result=subprocess.run([*common,'scenario-catalog','--dataset',str(self.directories[0]),
            '--output',str(self.work/'artifacts/runs/incomplete'),*args],capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,2);self.assertIn('SOURCES_MISSING',result.stdout)

    def test_policy_scope_invalid_seed_batch_and_changed_source_fail_closed(self):
        self.sources();self.build()
        with self.assertRaisesRegex(ValueError,'CATALOG_SEED'):self.build('bad-seed',seed=True)
        reader=self.reader()
        with self.assertRaisesRegex(ValueError,'BATCH_SIZE'):next(reader.batches('train',batch_size=129))
        reader.preflight('test')
        dataset_id=next(iter(reader.readers));source=reader.readers[dataset_id]
        path=source.identities[0][0]
        with path.open('ab') as f:f.write(b'SYNTHETIC_TAMPER')
        with self.assertRaisesRegex(ValueError,'SHARD_CHANGED'):list(reader.batches('test'))
        policy=json.loads(self.policy_path.read_bytes());policy['sources']=policy['sources'][:-1]
        self.policy_path.write_bytes(canonical_bytes(policy));digest=file_sha256(self.policy_path)
        with self.assertRaisesRegex(ValueError,'SIX_SOURCE_SPLIT_SCOPE'):
            build_scenario_catalog([],self.work,self.work/'artifacts/runs/five',
                                   policy_path=self.policy_path,policy_sha=digest)


if __name__=='__main__':unittest.main()
