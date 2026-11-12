"""Validate catalog-bound codecs, rotating subsets and batch SDK failure guards.
Kiểm synthetic riêng; không coi fixture metrics là kết quả Edge-IIoTset thật.
"""
import base64
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
for key in ('NIDS_SL_SOURCE','NIDS_SWOP_SOURCE'):
    if os.environ.get(key):sys.path.insert(0,os.environ[key])
import numpy as np
import test_partitioned_training as fixtures
from nids_ml.adapters.swarm.batch_data import CatalogTemplate,RoundReader,SUBSET_POLICY
from nids_ml.adapters.swarm.batch_job import bootstrap,load_batch_job,main
from nids_ml.adapters.swarm.batch_client import BatchTrainer,RoundBudget,run_batch_client
from nids_ml.adapters.swarm.contract import CNNTemplate,context,encode_weights,validate_artifact,sha,write_json
from nids_ml.training.batched import model_identity,TrainingLimits


def weights(value=0.):
    return tuple(np.full(shape,value,dtype='<f4') for _,shape in CNNTemplate.tensors)


class BatchContractTests(unittest.TestCase):
    def setUp(self):
        self.template=CatalogTemplate('sha256:'+'1'*64)

    def test_catalog_identity_distinct_from_pilot_and_local_architecture(self):
        self.assertNotEqual(self.template.identity,CNNTemplate().identity)
        self.assertNotEqual(self.template.identity,model_identity())
        self.assertNotEqual(self.template.identity,CatalogTemplate('sha256:'+'2'*64).identity)

    def test_unchanged_sl_and_user_tf_wire_interop(self):
        from user_tf.weights import serialize,deserialize
        from sl_runtime.model_data import decode,encode,fedavg
        ctx=context('epoch',1,'sl-a','local')
        raw=encode_weights(weights(.2),3,ctx,template=self.template)
        self.assertEqual(raw,serialize(self.template,weights(.2),3,ctx))
        bundle=deserialize(raw,self.template,ctx)
        np.testing.assert_array_equal(bundle.weights[0],weights(.2)[0])
        a=decode(encode_weights(weights(0),template=self.template),self.template)
        b=decode(encode_weights(weights(2),template=self.template),self.template)
        result=fedavg([(a,1),(b,3)],self.template)
        self.assertTrue(all(v==1.5 for tensor in result for v in tensor))
        validate_artifact(encode(result,self.template),template=self.template)

    def test_wrong_catalog_pilot_shape_context_and_integrity_rejected(self):
        ctx=context('epoch',1,'sl-a','local')
        raw=encode_weights(weights(),1,ctx,template=self.template)
        for template in (None,CatalogTemplate('sha256:'+'2'*64)):
            with self.assertRaisesRegex(ValueError,'MODEL_MISMATCH'):
                validate_artifact(raw,ctx,template=template)
        for bad in (raw[:-1],raw+b'x',raw[:-1]+b'x'):
            with self.assertRaises(ValueError):validate_artifact(bad,ctx,template=self.template)
        with self.assertRaises(ValueError):validate_artifact(raw,context('other',1,'sl-a','local'),template=self.template)
        with self.assertRaises(ValueError):encode_weights([np.zeros((33,64),np.float32),*weights()[1:]],template=self.template)

    def test_invalid_catalog_and_nonfinite_weights_fail_closed(self):
        for catalog in (None,True,'1'*64,'sha256:'+'A'*64):
            with self.assertRaises(ValueError):CatalogTemplate(catalog)
        with self.assertRaises(ValueError):encode_weights(weights(float('nan')),template=self.template)

    def test_service_sample_cap_not_changed_and_duplicate_stale_rejected(self):
        from sl_runtime.exchange_state import ExchangeState
        from sl_runtime.model_data import ExchangeError
        state=ExchangeState('sl-a',('sl-a','sl-b'),encode_weights(weights(),template=self.template),
            'nids-unit',self.template,10)
        state.observe(SimpleNamespace(round_id=1,instance_id='epoch',participants=('sl-a','sl-b'),
            status='waiting',ready_peers=()),'epoch',{'sl-a','sl-b'})
        raw=encode_weights(weights(.1),1,context('epoch',1,'sl-a','local'),template=self.template)
        message=dict(state.common(),producer_peer_id='sl-a',samples=4096,weights_sha256=sha(raw),weights_b64=base64.b64encode(raw).decode())
        with self.assertRaisesRegex(ExchangeError,'INVALID_SAMPLE_COUNT'):state.update(dict(message,samples=4097),True)
        state.update(message,True)
        with self.assertRaisesRegex(ExchangeError,'DUPLICATE_UPDATE'):state.update(message,True)
        with self.assertRaisesRegex(ExchangeError,'STALE_ROUND'):state.update(dict(message,round_id=2),True)


class BatchReaderTests(unittest.TestCase):
    def setUp(self):
        self.fx=fixtures.PartitionTests();self.fx.setUp()
        self.work=self.fx.work
        self.catalog,self.manifest=self.fx.catalog([(self.fx.dataset(),None)])
    def tearDown(self):self.fx.tearDown()
    def reader(self,client=0,rid=1,cap=12):
        return RoundReader(self.catalog,self.work,client,rid,cap,scratch=self.catalog)
    def setup_job(self):
        output=self.work/'artifacts/runs/job'
        report=bootstrap(self.catalog,self.work,output,{'sn':'http://127.0.0.1:1','sl-a':'http://127.0.0.1:2',
            'sl-b':'http://127.0.0.1:3','controller':'http://127.0.0.1:4'},cap=12)
        return output,report,load_batch_job(output/'job.json')

    def test_rotation_no_duplicates_complete_union_for_small_pool(self):
        for client in (0,1):
            a,b=self.reader(client,1),self.reader(client,2)
            ids=a.selected_ids();other=b.selected_ids()
            self.assertEqual(len(ids),len(set(ids)))
            self.assertEqual(len(other),len(set(other)))
            self.assertEqual(len(set(ids)|set(other)),a.pool)
            self.assertNotEqual(ids,other)
            self.assertFalse(a.report()['full_pool_used_this_round'])

    def test_selection_independent_of_batch_and_shuffle(self):
        reader=self.reader()
        order=lambda size,epoch:[i for b in reader.batches('train',0,batch_size=size,epoch=epoch) for i in b['sample_id']]
        self.assertEqual(set(order(3,0)),set(order(7,1)))
        self.assertEqual(order(3,0),order(7,0))
        self.assertEqual(reader.report(),self.reader().report())

    def test_validation_test_unchanged_and_never_selected_for_train(self):
        reader=self.reader();train=set(reader.selected_ids())
        for split in ('validation','test'):
            ids=[i for b in reader.batches(split,0,batch_size=3) for i in b['sample_id']]
            self.assertFalse(train&set(ids));self.assertEqual(len(ids),sum(reader.counts(split,0).values()))

    def test_cap_larger_than_pool_uses_all_and_types_fail_closed(self):
        a,b=self.reader(cap=4096),self.reader(rid=2,cap=4096)
        self.assertEqual(a.selected_ids(),b.selected_ids());self.assertTrue(a.report()['full_pool_used_this_round'])
        for cap in (True,1,4097):
            with self.assertRaises(ValueError):self.reader(cap=cap)
        with self.assertRaises(ValueError):self.reader(rid=True)

    def test_stream_true_counts_and_two_classes(self):
        reader=self.reader();counts={'0':0,'1':0}
        for batch in reader.batches('train',0,batch_size=3):
            self.assertLessEqual(len(batch['y']),3)
            for y in batch['y'][:,0]:counts[str(int(y))]+=1
        self.assertEqual(counts,reader.counts('train',0));self.assertEqual(sum(counts.values()),12)
        with self.assertRaises(ValueError):reader.counts('train',1)

    def test_single_class_subset_rejected_instead_of_silent_resampling(self):
        with patch.object(RoundReader,'selection',return_value=("eligible=1 AND split='train' AND y=0",[])):
            with self.assertRaisesRegex(ValueError,'BOTH_CLASSES'):self.reader()

    def test_bound_job_bootstrap_and_offline_roles(self):
        output,report,cfg=self.setup_job()
        self.assertEqual(report['catalog_id'],self.manifest['catalog_id'])
        self.assertEqual(cfg['samples'],{'sl-a':12,'sl-b':12})
        for role,extra in [('validate',[]),('sl',['--peer-id','sl-a','--port','9000']),
            ('client',['--peer-id','sl-a','--catalog','not-read-in-dry-run','--output',str(output/'client')]),
            ('controller',['--port','9001','--output',str(output/'controller')])]:
            with patch('http.client.HTTPConnection') as http,contextlib.redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(main([role,'--config',str(output/'job.json'),*extra]),0)
                self.assertIn(json.loads(stream.getvalue())['status'],('VALID','DRY_RUN'))
            http.assert_not_called()
        self.assertFalse((output/'controller').exists());self.assertFalse((output/'client').exists())
        with self.assertRaises(ValueError):bootstrap(self.catalog,self.work,output,{},cap=12)

    def test_job_catalog_count_timing_and_version_rejections(self):
        output,_,cfg=self.setup_job()
        original=json.loads((output/'job.json').read_text())
        for change in ({'catalog_id':'sha256:'+'2'*64},{'model_id':CNNTemplate().identity},
            {'samples':{'sl-a':13,'sl-b':12}},{'subset_cap':4097},{'epochs':True},
            {'round_ids':[True,2]},{'round_timeout':float('inf')},{'extra':1},{'format':'old'}):
            path=output/'invalid.json';path.write_text(json.dumps(dict(original,**change)),encoding='utf-8')
            with self.subTest(change=change),self.assertRaises(ValueError):load_batch_job(path)

    def test_missing_catalog_and_wrong_job_block_before_network(self):
        output,_,cfg=self.setup_job();bad=copy.deepcopy(cfg);bad['catalog_id']='sha256:'+'2'*64
        with patch('http.client.HTTPConnection') as network:
            with self.assertRaisesRegex(ValueError,'CATALOG_JOB_SAMPLE'):run_batch_client(bad,'sl-a',self.catalog,
                output/'client',workspace=self.work)
        network.assert_not_called();self.assertFalse((output/'client').exists())

    def test_sdk_batch_train_weights_change_and_reset(self):
        reader=self.reader();target=self.work/'artifacts/runs/trainer';target.mkdir()
        trainer=BatchTrainer(CatalogTemplate(self.manifest['catalog_id']),target)
        trainer.reader=reader;trainer.transfer_context=context('epoch',1,'sl-a','local')
        initial=trainer.get_weights();result=trainer.train(reader,None,batch_size=5)
        self.assertEqual(result['samples'],12);self.assertEqual(result['steps'],3)
        self.assertTrue(any(not np.array_equal(a,b) for a,b in zip(initial,trainer.get_weights())))
        validate_artifact((target/'round-1-local.iutfw').read_bytes(),trainer.transfer_context,template=trainer.spec)
        trainer.set_weights(initial,0);self.assertEqual(trainer.training_step,0)
        with self.assertRaises(ValueError):trainer.set_weights(initial,1)
        with self.assertRaises(ValueError):trainer.train(np.zeros((12,20,11,1),np.float32),None)

    def test_deadline_aborts_local_training_without_upload(self):
        budget=RoundBudget(self.catalog,TrainingLimits());budget.deadline=time.monotonic()-1
        with self.assertRaisesRegex(ValueError,'LOCAL_TRAINING_DEADLINE'):budget.check()

    def test_epoch_change_blocks_training_and_receipt(self):
        output,_,cfg=self.setup_job()
        good={'service':'independent-sn','api_version':'1','instance_id':'epoch'}
        changed=dict(good,instance_id='changed')
        with patch('nids_ml.adapters.swarm.batch_client.http_request',side_effect=[(200,good),(200,changed)]),\
             patch('user_tf.exchange.MLExchangeClient.train_round') as training,\
             patch('nids_ml.adapters.swarm.batch_client.send_receipt') as receipt:
            with self.assertRaisesRegex(ValueError,'SN_EPOCH_CHANGED'):
                run_batch_client(cfg,'sl-a',self.catalog,output/'client',workspace=self.work)
        training.assert_not_called();receipt.assert_not_called()
        self.assertEqual(json.loads((output/'client/FAILED.json').read_text())['completed_rounds'],0)

    def test_uncertain_update_never_retries_training_or_sends_receipt(self):
        from user_tf.exchange import ModelExchangeError
        output,_,cfg=self.setup_job()
        health={'service':'independent-sn','api_version':'1','instance_id':'epoch'}
        with patch('nids_ml.adapters.swarm.batch_client.http_request',return_value=(200,health)),\
             patch('user_tf.exchange.MLExchangeClient.session',return_value={'round':{'phase':'collecting'}}),\
             patch('user_tf.exchange.MLExchangeClient.train_round',side_effect=ModelExchangeError('acceptance unknown')) as training,\
             patch('nids_ml.adapters.swarm.batch_client.send_receipt') as receipt:
            with self.assertRaisesRegex(ModelExchangeError,'acceptance unknown'):
                run_batch_client(cfg,'sl-a',self.catalog,output/'client',workspace=self.work)
        self.assertEqual(training.call_count,1);receipt.assert_not_called()
        self.assertFalse(json.loads((output/'client/FAILED.json').read_text())['automatic_retry'])

    def test_round_rotates_at_production_4096_cap(self):
        # Synthetic index-only population; not PCAP, not dataset metrics.
        # Chỉ kiểm SQL trên 5523 IDs giả, không tạo ground truth dữ liệu thật.
        with contextlib.closing(sqlite3.connect(self.catalog/'catalog.sqlite')) as db:
            db.execute('DELETE FROM samples')
            db.executemany('INSERT INTO samples(id,sha,y,eligible,split,client) VALUES(?,?,?,1,\'train\',0)',
                [(i,f'{i:064x}',i%2) for i in range(1,5524)])
            db.commit()
        self.fx.refresh_catalog(self.catalog)
        # Base catalog audits cannot accept the synthetic index-only fixture.
        # Mock only constructor metadata, then exercise real SQL membership/counts.
        from nids_ml.storage.partitioned import PartitionReader
        def init(reader,*args,**kwargs):
            reader.index=self.catalog/'catalog.sqlite';reader.manifest={'client_counts':{'0':{'train':{'0':2761,'1':2762}}}}
        with patch.object(PartitionReader,'__init__',init):
            first,second=self.reader(cap=4096),self.reader(rid=2,cap=4096)
            self.assertEqual(len(first),4096);self.assertEqual(len(second),4096)
            self.assertEqual(len(set(first.selected_ids())|set(second.selected_ids())),5523)
            self.assertEqual(len(set(first.selected_ids())&set(second.selected_ids())),2669)


if __name__=='__main__':unittest.main()
