"""Exercise global dedup/group splits, placement and streamed CNN learning on fixtures.
Kiểm synthetic có kiểm soát; không dùng fixture metrics làm kết quả dataset thật.
"""
from collections import Counter
from contextlib import closing,redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
import test_flow_shards as fixtures
from test_demo_pipeline import FixtureDecoder
from nids_ml.data.catalog import build_catalog,identity,audit
from nids_ml.storage.partitioned import PartitionReader
from nids_ml.training.batched import (BinaryMetrics,StreamTrainer,model_identity,save_checkpoint,
    restore_checkpoint,train_catalog)
from nids_ml.adapters.swarm.contract import CNNTemplate
from nids_ml.pcap.decoder import file_sha256
from nids_ml.cli.large_training import main


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.fx=fixtures.FlowShardTests();self.fx.setUp();self.work=self.fx.work
    def tearDown(self):self.fx.tearDown()
    def dataset(self,name='shared',shift=0):
        caches=[]
        for suffix,label in (('normal',0),('attack',1)):
            mapping=self.fx.mapping(name+'-'+suffix,label)
            if shift:
                path=str(self.work/'data/raw'/(name+'-'+suffix+'.pcap'))
                FixtureDecoder.records_by_path[path]=[replace(record,fields=tuple(replace(field,
                    show=str(int(field.show)+shift) if field.name=='tcp.ack' else field.show)
                    for field in record.fields)) for record in FixtureDecoder.records_by_path[path]]
            cache,_=self.fx.cache(mapping,name+'-'+suffix+'-cache');caches.append(cache)
        return self.fx.export(caches,name+'-dataset')[0]
    def catalog(self,inputs,name='catalog'):
        out=self.work/'artifacts/runs'/name
        result=build_catalog(inputs,self.work,out)
        return out,result
    def sql(self,catalog,sql):
        with closing(sqlite3.connect(catalog/'catalog.sqlite')) as db:return db.execute(sql).fetchall()
    def refresh_catalog(self,path):
        m=json.loads((path/'manifest.json').read_text());m['index'].update(
            bytes=(path/'catalog.sqlite').stat().st_size,sha256=file_sha256(path/'catalog.sqlite'))
        m['catalog_id']=identity(m);(path/'manifest.json').write_text(json.dumps(m))
    def alter_sample(self,dataset,row,*,copy_x=None,label=None,group=None):
        import h5py
        with h5py.File(dataset/'flows-00000.h5','r+') as h:
            meta=json.loads(h['metadata'][row].tobytes())
            if copy_x is not None:h['X'][row]=copy_x
            if label is not None:
                h['y'][row]=label;meta['y']=label
                meta['attack_type']='Normal' if label==0 else 'Backdoor';meta['attack_types']=[meta['attack_type']]
                h['attack_type'][row]=meta['attack_type'].encode()
            if group is not None:
                meta['core']['flow_tuple']=group
                gid=hashlib.sha256((meta['core']['capture']['capture_sha256']+
                    json.dumps(group,separators=(',',':'))).encode()).hexdigest()
                meta['group_id']=gid;h['group_id'][row]=gid.encode()
            sha=hashlib.sha256(h['X'][row].astype('<f4',copy=False).tobytes()).hexdigest()
            meta['tensor_sha256']=sha;h['tensor_sha256'][row]=sha.encode()
            h['metadata'][row]=json.dumps(meta,sort_keys=True,separators=(',',':')).encode()
        self.fx.refresh_shard(dataset)
        if label is not None:
            with h5py.File(dataset/'flows-00000.h5','r') as h:
                counts=Counter(str(int(y)) for y in h['y'][:]);families=Counter(f.decode() for f in h['attack_type'][:])
            m=json.loads((dataset/'manifest.json').read_text());m['label_counts']=dict(counts);m['attack_type_counts']=dict(families)
            (dataset/'manifest.json').write_text(json.dumps(m))

    def test_connection_split_both_clients_both_classes_and_stream_counts(self):
        path,m=self.catalog([(self.dataset(),None)]);reader=PartitionReader(path,self.work,scratch=path)
        self.assertTrue(m['training_ready']);self.assertEqual(m['statistics']['retained_samples'],48)
        all_ids=set();train_groups=[]
        for client in (0,1):
            ids=set();groups=set()
            for batch in reader.batches('train',client,batch_size=5,epoch=0):
                self.assertLessEqual(len(batch['y']),5);ids.update(batch['sample_id']);groups.update(batch['group_id'])
            self.assertFalse(all_ids&ids);all_ids|=ids;train_groups.append(groups)
            self.assertEqual(set(reader.counts('train',client)),{'0','1'})
        self.assertFalse(train_groups[0]&train_groups[1])
        folds={s:{g for b in reader.batches(s,batch_size=7) for g in b['group_id']} for s in ('train','validation','test')}
        self.assertFalse(folds['train']&folds['test']);self.assertFalse(folds['train']&folds['validation'])
        self.assertFalse(folds['validation']&folds['test'])

    def test_deterministic_catalog_and_epoch_shuffle_preserve_membership(self):
        a=self.dataset();b=self.dataset('other',shift=10000)
        first,m=self.catalog([(a,None),(b,None)],'first');second,n=self.catalog([(b,None),(a,None)],'second')
        self.assertEqual(m['catalog_id'],n['catalog_id']);self.assertEqual(m['index'],n['index'])
        reader=PartitionReader(first,self.work,scratch=first)
        order=lambda epoch:[v for batch in reader.batches('train',0,epoch=epoch,batch_size=4) for v in batch['sample_id']]
        self.assertEqual(order(0),order(0));self.assertNotEqual(order(0),order(1));self.assertEqual(set(order(0)),set(order(1)))

    def test_global_cross_group_label_duplicates_are_excluded_before_split(self):
        import h5py
        dataset=self.dataset()
        with h5py.File(dataset/'flows-00000.h5','r') as h:
            first=next(i for i in range(len(h['y'])) if h['y'][i]==0)
            other=next(i for i in range(len(h['y'])) if h['y'][i]==1)
            value=h['X'][first]
        self.alter_sample(dataset,other,copy_x=value)
        path,m=self.catalog([(dataset,None)])
        self.assertEqual(m['statistics']['excluded_cross_group_or_conflicting_tensor_patterns'],1)
        self.assertGreaterEqual(m['statistics']['excluded_samples'],2)
        sha=hashlib.sha256(value.astype('<f4',copy=False).tobytes()).hexdigest()
        self.assertEqual(self.sql(path,"SELECT count(*) FROM samples WHERE eligible=1 AND sha='"+sha+"'"),[(0,)])

    def test_within_connection_duplicates_keep_one_and_mixed_groups_are_excluded(self):
        import h5py
        dataset=self.dataset()
        with h5py.File(dataset/'flows-00000.h5','r') as h:
            groups={}
            for i,g in enumerate(h['group_id'][:]):groups.setdefault(g.decode(),[]).append(i)
            indices=next(v for v in groups.values() if int(h['y'][v[0]])==0)
            value=h['X'][indices[0]]
        self.alter_sample(dataset,indices[1],copy_x=value)
        first,m=self.catalog([(dataset,None)],'dedup')
        self.assertEqual(m['statistics']['excluded_samples'],1)
        self.alter_sample(dataset,indices[2],label=1)
        second,n=self.catalog([(dataset,None)],'mixed')
        self.assertEqual(n['statistics']['excluded_mixed_label_groups'],1)
        self.assertGreaterEqual(n['statistics']['excluded_samples'],3)

    def test_insufficient_groups_does_not_fallback_to_row_split(self):
        import h5py
        dataset=self.dataset()
        with h5py.File(dataset/'flows-00000.h5','r') as h:
            rows=[i for i,y in enumerate(h['y'][:]) if int(y)==0]
            group=json.loads(h['metadata'][rows[0]].tobytes())['core']['flow_tuple']
        for row in rows:self.alter_sample(dataset,row,group=group)
        with self.assertRaisesRegex(ValueError,'INSUFFICIENT_CONNECTION_GROUPS'):self.catalog([(dataset,None)])
        out=self.work/'artifacts/runs/catalog';self.assertTrue((out/'FAILED.json').exists());self.assertFalse((out/'manifest.json').exists())

    def test_local_owners_are_respected_and_missing_remote_source_is_not_read(self):
        a=self.dataset('node0');b=self.dataset('node1',shift=10000)
        path,m=self.catalog([(a,0),(b,1)])
        reader=PartitionReader(path,self.work,scratch=path)
        ids0={v for batch in reader.batches('train',0) for v in batch['sha']}
        self.assertTrue(ids0)
        b_id=file_sha256(b/'manifest.json')
        limited=PartitionReader(path,self.work,sources={b_id:self.work/'not-present'})
        self.assertTrue(list(limited.batches('train',0)))
        with self.assertRaisesRegex(ValueError,'LOCAL_DATASET_MISSING'):limited.preflight('train',1)

    def test_owner_conflict_duplicate_inputs_invalid_owner_and_bad_source(self):
        dataset=self.dataset()
        with self.assertRaisesRegex(ValueError,'DUPLICATE_INPUT'):self.catalog([(dataset,None),(dataset,None)])
        with self.assertRaisesRegex(ValueError,'INPUT_OWNER'):self.catalog([(dataset,True)])
        copy=self.work/'artifacts/runs/copy';shutil.copytree(dataset,copy)
        m=json.loads((copy/'manifest.json').read_text());m['resources']['fixture_only']='different-manifest'
        (copy/'manifest.json').write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'GROUP_OWNER_CONFLICT'):self.catalog([(dataset,None),(copy,0)],'conflict')
        with self.assertRaises(ValueError):PartitionReader(self.work/'missing',self.work)

    def test_catalog_tampering_and_claimed_counts_fail_closed(self):
        dataset=self.dataset();path,m=self.catalog([(dataset,None)])
        with closing(sqlite3.connect(path/'catalog.sqlite')) as db:
            group=db.execute("SELECT group_id FROM samples WHERE split='train' GROUP BY group_id HAVING count(*)>1 LIMIT 1").fetchone()[0]
            db.execute("UPDATE samples SET split='test',client=NULL WHERE id=(SELECT min(id) FROM samples WHERE group_id=?)",(group,));db.commit()
        with self.assertRaisesRegex(ValueError,'INDEX_SHA256'):PartitionReader(path,self.work)
        self.refresh_catalog(path)
        with self.assertRaisesRegex(ValueError,'CONNECTION_LEAKAGE'):PartitionReader(path,self.work)

    def test_local_source_override_catalog_row_validation_and_budget_guards(self):
        dataset=self.dataset();path,m=self.catalog([(dataset,None)])
        source_id=m['sources'][0]['dataset_id']
        reader=PartitionReader(path,self.work,sources={source_id:dataset})
        self.assertTrue(list(reader.batches('test',batch_size=3)))
        with self.assertRaises(ValueError):PartitionReader(path,self.work,sources={'unknown':dataset})
        for size in (True,0,129):
            with self.assertRaises(ValueError):list(reader.batches('train',batch_size=size))
        with self.assertRaisesRegex(ValueError,'SCRATCH'):list(reader.batches('train',epoch=0))
        with closing(sqlite3.connect(path/'catalog.sqlite')) as db:
            db.execute('UPDATE samples SET row_no=-1 WHERE id=(SELECT min(id) FROM samples WHERE eligible=1)');db.commit()
        self.refresh_catalog(path);reader=PartitionReader(path,self.work)
        with self.assertRaisesRegex(ValueError,'LOCATION'):list(reader.batches('train'))

    def test_shuffle_early_close_restores_temp_directory_and_releases_files(self):
        dataset=self.dataset();path,_=self.catalog([(dataset,None)])
        reader=PartitionReader(path,self.work,scratch=path)
        with closing(sqlite3.connect(':memory:')) as db:before=db.execute('PRAGMA temp_store_directory').fetchone()
        batches=reader.batches('train',epoch=0);next(batches);batches.close()
        with closing(sqlite3.connect(':memory:')) as db:after=db.execute('PRAGMA temp_store_directory').fetchone()
        self.assertEqual(before,after)

    def test_cli_inspect_counts_are_read_from_selected_rows(self):
        path,m=self.catalog([(self.dataset(),None)])
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(['inspect','--catalog',str(path),'--workspace',str(self.work),'--client','0']),0)
        result=json.loads(out.getvalue());self.assertEqual(result['observed'],m['client_counts']['0'])

    def test_missing_local_data_stops_before_creating_a_trainer_and_no_overwrite(self):
        dataset=self.dataset();path,m=self.catalog([(dataset,None)])
        with self.assertRaises(ValueError):self.catalog([(dataset,None)])
        missing={m['sources'][0]['dataset_id']:self.work/'missing-data'}
        with patch('nids_ml.training.batched.StreamTrainer') as trainer:
            with self.assertRaisesRegex(ValueError,'LOCAL_DATASET_MISSING'):
                train_catalog(path,self.work,self.work/'artifacts/runs/failed-train',sources=missing)
            trainer.assert_not_called()
        self.assertTrue((self.work/'artifacts/runs/failed-train/FAILED.json').exists())

    def test_group_lineage_and_feature_contract_cannot_be_forged(self):
        import h5py
        dataset=self.dataset()
        with h5py.File(dataset/'flows-00000.h5','r+') as h:
            meta=json.loads(h['metadata'][0].tobytes());meta['core']['flow_tuple'][1]+=1
            h['metadata'][0]=json.dumps(meta,separators=(',',':')).encode()
        self.fx.refresh_shard(dataset)
        with self.assertRaisesRegex(ValueError,'CONNECTION_GROUP_IDENTITY'):self.catalog([(dataset,None)],'bad-group')
        m=json.loads((dataset/'manifest.json').read_text());m['feature_configuration']['window_seconds']=True
        (dataset/'manifest.json').write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'FEATURE_CONFIGURATION'):self.catalog([(dataset,None)],'bad-feature')

    def test_streamed_metrics_and_per_family_missing_data_semantics(self):
        import numpy as np
        from nids_ml.training.demo import metrics
        y=np.asarray([0,1,0,1]);p=np.asarray([.1,.9,.7,.3],dtype='float32')
        total=BinaryMetrics();total.update(y[:2],p[:2]);total.update(y[2:],p[2:]);result=total.report();expected=metrics(y,p)
        for key in ('loss','accuracy','precision','recall','f1','false_positive_rate'):
            self.assertAlmostEqual(result[key],expected[key],places=7)
        self.assertEqual(BinaryMetrics().report(),dict(status='NOT_EVALUATED',samples=0))
        normal=BinaryMetrics();normal.update([0],[.1]);self.assertIsNone(normal.report()['recall'])

    def test_batch_training_checkpoint_and_new_nids_identity(self):
        path,m=self.catalog([(self.dataset(),None)])
        output=self.work/'artifacts/runs/training'
        result=train_catalog(path,self.work,output,client=0,epochs=1,batch_size=5)
        self.assertEqual(result['status'],'PASS_NIDS_BATCH_TRAINING')
        self.assertEqual(result['checkpoint_restore_max_prediction_error'],0.)
        self.assertNotEqual(model_identity(),CNNTemplate().identity)
        self.assertEqual(set(result['test']['per_family']),set(m['family_coverage']))
        with self.assertRaises(ValueError):restore_checkpoint(output,expected_catalog_id='wrong')
        with (output/'weights.bin').open('r+b') as stream:stream.write(b'\xff'*4)
        with self.assertRaisesRegex(ValueError,'SHA256'):restore_checkpoint(output)

    def test_two_local_clients_two_numeric_rounds_use_applied_global_weights(self):
        import numpy as np
        path,m=self.catalog([(self.dataset(),None)]);reader=PartitionReader(path,self.work,scratch=path)
        trainers=[StreamTrainer(42),StreamTrainer(42)]
        initial=trainers[0].get_weights()
        for trainer in trainers:trainer.set_weights(initial)
        for epoch in (0,1):
            base=[v.copy() for v in trainers[0].get_weights()]
            reports=[t.train_epoch(reader,c,batch_size=5,epoch=epoch) for c,t in enumerate(trainers)]
            expected=hashlib.sha256(b''.join(v.tobytes() for v in base)).hexdigest()
            self.assertTrue(all(r['initial_tensor_sha256']==expected for r in reports))
            samples=[r['samples'] for r in reports];total=sum(samples)
            globals=[sum(t.get_weights()[i].astype('float64')*n for t,n in zip(trainers,samples))/total for i in range(4)]
            for trainer in trainers:
                trainer.set_weights([v.astype('float32') for v in globals]);self.assertEqual(trainer.steps,0)
            for left,right in zip(trainers[0].get_weights(),trainers[1].get_weights()):np.testing.assert_array_equal(left,right)

    def test_weight_dtype_nonfinite_and_pilot_identity_are_rejected(self):
        import numpy as np
        trainer=StreamTrainer();weights=list(trainer.get_weights());weights[0]=weights[0].astype('float64')
        with self.assertRaises(ValueError):trainer.set_weights(weights)
        path=self.work/'artifacts/runs/checkpoint';path.mkdir()
        save_checkpoint(trainer,path,'sha256:'+'a'*64)
        m=json.loads((path/'checkpoint.json').read_text());m['model_id']=CNNTemplate().identity
        (path/'checkpoint.json').write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'NIDS_CHECKPOINT_MODEL'):restore_checkpoint(path)
        m['model_id']=model_identity()
        binary=path/'weights.bin';raw=binary.read_bytes();raw=np.asarray([np.nan],dtype='<f4').tobytes()+raw[4:]
        binary.write_bytes(raw);m['weights_sha256']=hashlib.sha256(raw).hexdigest();(path/'checkpoint.json').write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'WEIGHT_SHAPE_DTYPE'):restore_checkpoint(path)


if __name__=='__main__':unittest.main()
