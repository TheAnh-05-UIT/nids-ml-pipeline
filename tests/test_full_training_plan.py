"""Small synthetic mixed-provenance plans; never dataset or board training results."""
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
import test_partitioned_training as mapped
import test_temporal_catalog as temporal
from nids_ml.labeling.contracts import canonical_bytes
from nids_ml.storage.planned import PlanBank
from nids_ml.training.full_plan import compile_plan,schedule,span,REF


class FullPlanTests(unittest.TestCase):
    def setUp(self):
        import h5py
        self.edge=mapped.PartitionTests();self.edge.setUp();self.addCleanup(self.edge.tearDown)
        self.work=self.edge.work
        dataset=self.edge.dataset('edge',shift=50000)
        path=dataset/'flows-00000.h5'
        with h5py.File(path,'r+') as h:
            for i in range(len(h['y'])):
                if h['y'][i]:
                    family='DDoS_TCP' if i%2 else 'DDoS_ICMP'
                    meta=json.loads(h['metadata'][i].tobytes());meta['attack_type']=family;meta['attack_types']=[family]
                    h['metadata'][i]=canonical_bytes(meta);h['attack_type'][i]=family.encode()
            counts=Counter(v.decode() for v in h['attack_type'][:])
        self.edge.fx.refresh_shard(dataset)
        m=json.loads((dataset/'manifest.json').read_bytes());m['attack_type_counts']=dict(counts)
        (dataset/'manifest.json').write_bytes(canonical_bytes(m))
        self.edge_path,self.edge_manifest=self.edge.catalog([(dataset,None)],'edge-catalog')
        self.mqtt=temporal.TemporalTests();self.mqtt.setUp();self.addCleanup(self.mqtt.doCleanups)
        self.mqtt.work=self.work;self.mqtt.fixture.work=self.work
        self.mqtt.sources();self.mqtt_manifest=self.mqtt.build('mqtt-catalog')
        self.mqtt_path=self.work/'artifacts/runs/mqtt-catalog'
        self.output=self.work/'artifacts/runs/full-plan'

    def compile(self,**kwargs):
        return compile_plan(self.edge_path,self.mqtt_path,self.work,self.output,
            policy_path=self.mqtt.policy_path,policy_sha=self.mqtt.policy_sha,cap=8,**kwargs)

    def bank(self,client):
        return PlanBank(self.output,self.work,client,policy_path=self.mqtt.policy_path,policy_sha=self.mqtt.policy_sha)

    def test_small_plan_preserves_splits_and_covers_every_sample_exactly_once(self):
        m=self.compile();self.assertFalse((self.output/'selection.sqlite').exists())
        self.assertEqual(m['missing_requested_families'],['DDoS_HTTP','DDoS_UDP'])
        self.assertEqual(m['statistics']['excluded_cross_parent_duplicate_rows'],0)
        expected={s:sum(self.edge_manifest['split_counts'][s].values())+sum(self.mqtt_manifest['split_counts'][s].values())
                  for s in ('train','validation','test')}
        self.assertEqual({s:sum(v.values()) for s,v in m['split_counts'].items()},expected)
        sets=[];evaluation=Counter()
        for client in (0,1):
            bank=self.bank(client);seen=[]
            for rid in range(1,m['total_rounds']+1):
                with closing(bank.reader('train',round_id=rid,scratch=self.output)) as reader:
                    self.assertTrue(2<=len(reader)<=8)
                    for batch in reader.batches('train',client,batch_size=3,epoch=0):seen.extend(batch['sample_id'])
            self.assertEqual(len(seen),len(set(seen)))
            self.assertEqual(len(seen),sum(bank.entry['train_counts'].values()));sets.append(set(seen))
            for split in ('validation','test'):
                with closing(bank.reader(split)) as reader:
                    evaluation[split]+=sum(len(b['X']) for b in reader.batches(split,client,batch_size=4))
        self.assertFalse(sets[0]&sets[1])
        self.assertEqual(dict(evaluation),{s:expected[s] for s in ('validation','test')})
        self.assertLess(sum(p.stat().st_size for p in self.output.glob('*.refs')),10000)

    def test_multiple_passes_reuse_complete_references_with_explicit_pass_count(self):
        m=self.compile(passes=3);self.assertEqual(m['total_rounds'],3*m['rounds_per_pass'])
        bank=self.bank(0)
        with closing(bank.reader('train',round_id=1)) as a,closing(bank.reader('train',round_id=1+m['rounds_per_pass'])) as b:
            self.assertEqual(a.report()['reference_sha256'],b.report()['reference_sha256'])

    def test_reference_tamper_and_wrong_policy_fail_closed(self):
        m=self.compile();p=self.output/m['clients']['0']['filename']
        with p.open('ab') as f:f.write(b'x')
        with self.assertRaisesRegex(ValueError,'FULL_REFERENCE_SHA'):self.bank(0)
        with self.assertRaises(ValueError):
            PlanBank(self.output,self.work,1,policy_path=self.mqtt.policy_path,policy_sha='0'*64)

    def test_no_overwrite_and_source_preflight_hashes_real_hdf5(self):
        m=self.compile()
        with self.assertRaises(ValueError):self.compile()
        bank=self.bank(0);bank.preflight()
        reader=next(iter(bank.readers.values()));path=reader.identities[0][0]
        with path.open('ab') as f:f.write(b'fixture tamper')
        with self.assertRaises(ValueError):
            with closing(bank.reader('train',round_id=1)) as r:list(r.batches('train',0))

    def test_all_cross_parent_duplicate_rows_are_excluded(self):
        import sqlite3
        from nids_ml.data.temporal_catalog import identity
        from nids_ml.pcap.decoder import file_sha256
        with closing(sqlite3.connect(self.edge_path/'catalog.sqlite')) as db:
            digest=db.execute("SELECT sha FROM samples WHERE eligible=1 AND split='train' AND y=0 LIMIT 1").fetchone()[0]
        with closing(sqlite3.connect(self.mqtt_path/'catalog.sqlite')) as db:
            sid=db.execute("SELECT id FROM samples WHERE eligible=1 AND split='train' AND y=0 LIMIT 1").fetchone()[0]
            db.execute('UPDATE samples SET sha=? WHERE id=?',(digest,sid));db.commit()
        # Only this synthetic catalog is modified to exercise the cross-parent collision guard.
        p=self.mqtt_path/'manifest.json';m=json.loads(p.read_bytes())
        m['index']['sha256']=file_sha256(self.mqtt_path/'catalog.sqlite');m['catalog_id']=identity(m)
        p.write_bytes(canonical_bytes(m))
        plan=self.compile()
        self.assertEqual(plan['statistics']['excluded_cross_parent_duplicate_rows'],2)
        self.assertEqual(plan['statistics']['retained_samples'],plan['statistics']['input_selected_rows']-2)


class ScheduleTests(unittest.TestCase):
    def test_616446_per_node_has_151_rounds_and_no_tail_drop(self):
        counts={'0':{'0':76676,'1':539770},'1':{'0':76676,'1':539770}}
        n,rows=schedule(counts);self.assertEqual(n,151)
        for client in counts:
            for y,total in counts[client].items():
                end=0
                for row in rows:
                    p=row['clients'][client][y];self.assertEqual(p['start'],end);end+=p['samples']
                    self.assertLessEqual(sum(v['samples'] for v in row['clients'][client].values()),4096)
                self.assertEqual(end,total)

    def test_tiny_balanced_cap_two_and_rare_class_failure(self):
        n,_=schedule({'0':{'0':2,'1':2},'1':{'0':2,'1':2}},cap=2)
        self.assertEqual(n,2)
        with self.assertRaisesRegex(ValueError,'RARE_CLASS'):
            schedule({'0':{'0':1,'1':10000},'1':{'0':1,'1':10000}})


if __name__=='__main__':unittest.main()
