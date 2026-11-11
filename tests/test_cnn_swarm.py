"""Check CNN wire compatibility, trainer safety and controller/SL failure semantics with small fixtures.
Kiểm contract, trainer và guards bằng fixtures nhỏ; không dùng số liệu giả làm kết quả dataset.
"""
import base64
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import contextlib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
for variable in ('NIDS_SL_SOURCE','NIDS_SWOP_SOURCE'):
    if os.environ.get(variable):sys.path.insert(0,str(Path(os.environ[variable]).resolve()))
import numpy as np
from nids_ml.adapters.swarm.contract import (CNNTemplate,context,encode_weights,fresh_output,sha,
    validate_artifact,write_json)
from nids_ml.adapters.swarm.job import load_job,main,http_request

SL_AVAILABLE=importlib.util.find_spec('sl_runtime') is not None
SWOP_AVAILABLE=importlib.util.find_spec('swop') is not None


def weights(value=0.):
    return tuple(np.full(shape,value,dtype='<f4') for _,shape in CNNTemplate.tensors)


def edit_header(raw,edit,blobs=None):
    size=struct.unpack('>I',raw[8:12])[0];header=json.loads(raw[12:12+size])
    body=raw[12+size:] if blobs is None else blobs
    edit(header)
    data=json.dumps(header,sort_keys=True,separators=(',',':')).encode()
    return raw[:8]+struct.pack('>I',len(data))+data+body


class CNNContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='cnn-swarm-unit-',dir=ROOT/'artifacts/runs')
        self.directory=Path(self.tmp.name);self.raw=encode_weights(weights())
    def tearDown(self):
        assert self.directory.resolve().parent==(ROOT/'artifacts/runs').resolve()
        self.tmp.cleanup()
    def config(self):
        (self.directory/'initial.iutfw').write_bytes(self.raw)
        cfg=dict(format='nids-cnn-swarm-job/1.0.0',run_id='unit',job_id='cnn-unit',sn_url='http://127.0.0.1:1',
            peers={'sl-a':'http://127.0.0.1:2','sl-b':'http://127.0.0.1:3'},controller_url='http://127.0.0.1:4',
            initial_weights='initial.iutfw',initial_sha256=sha(self.raw),dataset_sha256='0'*64,
            samples={'sl-a':8,'sl-b':8},round_ids=[1,2],epochs=1,batch_size=16,
            request_timeout=.5,poll_interval=.1,round_timeout=10,startup_timeout=10)
        write_json(self.directory/'job.json',cfg);return cfg

    def test_fixed_identity_registry_and_roundtrip(self):
        self.assertEqual(sum(np.prod(shape) for _,shape in CNNTemplate.tensors),2241)
        a=validate_artifact(self.raw);b=validate_artifact(encode_weights(weights()))
        self.assertEqual(a,b);self.assertLess(len(self.raw),65536)
        self.assertEqual(a[2],sha(b''.join(w.tobytes() for w in weights())))

    def test_unmodified_user_tf_codec_interoperates(self):
        from user_tf.weights import serialize,deserialize
        from user_tf.spec import ModelSpec
        ctx=context('epoch',1,'sl-a','local');arrays=weights(.25)
        ours=encode_weights(arrays,3,ctx)
        self.assertEqual(ours,serialize(CNNTemplate(),arrays,3,ctx))
        bundle=deserialize(ours,CNNTemplate(),ctx)
        for a,b in zip(bundle.weights,arrays):np.testing.assert_array_equal(a,b)
        self.assertNotEqual(CNNTemplate().identity,ModelSpec().identity)

    def test_model_shape_dtype_and_version_mismatches_rejected(self):
        changes=[lambda h:h.update(model_identity='sha256:'+'0'*64),
            lambda h:h['model_spec'].update(architecture='tiny-regressor'),
            lambda h:h['model_spec'].update(packets=10),lambda h:h.update(format_version=True),
            lambda h:h['tensors'][0].update(shape=[33,64]),
            lambda h:h['tensors'][0].update(dtype='<f8'),lambda h:h['tensors'][0].update(nbytes=1)]
        for change in changes:
            with self.subTest(change=change),self.assertRaises(ValueError):validate_artifact(edit_header(self.raw,change))

    def test_integrity_truncation_trailing_bytes_and_nonfinite_rejected(self):
        for raw in (self.raw[:-1],self.raw+b'x',b'pickle',b'x'*65537):
            with self.assertRaises(ValueError):validate_artifact(raw)
        header,blobs,_=validate_artifact(self.raw)
        bad=struct.pack('<f',float('nan'))+blobs[0][4:]
        raw=edit_header(self.raw,lambda h:h['tensors'][0].update(sha256=sha(bad)),bad+b''.join(blobs[1:]))
        with self.assertRaisesRegex(ValueError,'NONFINITE'):validate_artifact(raw)

    def test_context_and_round_identity_fail_closed(self):
        ctx=context('epoch',1,'sl-a','local');raw=encode_weights(weights(),1,ctx)
        validate_artifact(raw,ctx)
        for wrong in (None,context('other',1,'sl-a','local'),context('epoch',2,'sl-a','local'),context('epoch',1,'sl-b','global')):
            with self.assertRaises(ValueError):validate_artifact(raw,wrong)
        for rid in (True,0,3):
            with self.assertRaises(ValueError):context('epoch',rid,'sl-a','local')
        with self.assertRaisesRegex(ValueError,'GLOBAL_TRAINING_STEP'):
            encode_weights(weights(),1,context('epoch',1,'sl-a','global'))

    def test_no_coercion_of_float64_or_wrong_weight_shapes(self):
        for arrays in (tuple(w.astype('float64') for w in weights()),weights()[:3],
                       (np.zeros((33,64),np.float32),*weights()[1:])):
            with self.assertRaises(ValueError):encode_weights(arrays)

    def test_job_validation_and_malformed_fields(self):
        cfg=self.config();self.assertEqual(load_job(self.directory/'job.json')['job_id'],'cnn-unit')
        for changes in ({'job_id':'old-live-job'},{'round_ids':[2,3]},{'epochs':True},
                {'samples':{'sl-a':8}},{'initial_sha256':'0'*64},{'extra':1},
                {'controller_url':'http://user:secret@host'},{'round_timeout':float('inf')}):
            path=self.directory/'invalid.json';path.write_text(json.dumps(dict(cfg,**changes)),encoding='utf-8')
            with self.subTest(changes=changes),self.assertRaises(ValueError):load_job(path)

    def test_all_network_cli_roles_default_to_offline_dry_run(self):
        self.config()
        for role,extra in [('sl',['--peer-id','sl-a','--port','9000']),
            ('client',['--peer-id','sl-a','--dataset','unused','--output',str(self.directory/'client')]),
            ('controller',['--port','9001','--output',str(self.directory/'controller')])]:
            with patch('http.client.HTTPConnection') as network,contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main([role,'--config',str(self.directory/'job.json'),*extra]),0)
            self.assertEqual(json.loads(output.getvalue())['status'],'DRY_RUN');network.assert_not_called()
        self.assertFalse((self.directory/'controller').exists());self.assertFalse((self.directory/'client').exists())

    def test_connection_error_never_implies_success(self):
        with patch('http.client.HTTPConnection') as connection:
            connection.return_value.request.side_effect=ConnectionRefusedError()
            with self.assertRaises(OSError):http_request('POST','http://127.0.0.1:9','/v1/rounds',{})

    def test_identical_receipt_retry_does_not_repeat_training_or_upload(self):
        from nids_ml.adapters.swarm.client import send_receipt
        cfg=dict(controller_url='http://127.0.0.1:1',request_timeout=.1,poll_interval=.1)
        calls=[];body={'round_id':1,'peer_id':'sl-a'}
        def request(method,url,path,receipt,timeout):
            calls.append((method,path,dict(receipt)))
            if len(calls)==1:raise OSError('ACK lost after acceptance')
            return 200,{'replayed':True}
        send_receipt(cfg,body,request,lambda _:None)
        self.assertEqual(len(calls),2);self.assertEqual(calls[0],calls[1])
        self.assertEqual(calls[0][1],'/v1/controller/receipts')

    def test_receipt_failures_are_bounded_and_rejection_is_not_retried(self):
        from nids_ml.adapters.swarm.client import send_receipt
        cfg=dict(controller_url='http://127.0.0.1:1',request_timeout=.1,poll_interval=.1)
        with patch('nids_ml.adapters.swarm.job.http_request',side_effect=OSError('offline')) as request:
            with self.assertRaises(OSError):send_receipt(cfg,{},sleep=lambda _:None)
            self.assertEqual(request.call_count,3)
        with patch('nids_ml.adapters.swarm.job.http_request',return_value=(409,{})) as request:
            with self.assertRaisesRegex(ValueError,'RECEIPT_REJECTED'):send_receipt(cfg,{},sleep=lambda _:None)
            self.assertEqual(request.call_count,1)

    def test_fresh_output_refuses_existing_directory(self):
        with self.assertRaises((ValueError,OSError)):fresh_output(self.directory)

    def test_cnn_training_and_optimizer_reset(self):
        from nids_ml.adapters.swarm.client import CNNTrainer
        trainer=CNNTrainer(self.directory);trainer.round_id=1;trainer.transfer_context=context('epoch',1,'sl-a','local')
        x=np.random.default_rng(42).uniform(0,.1,(16,20,11,1)).astype('float32')
        y=(np.arange(16)%2).astype('float32').reshape(-1,1);x+=y.reshape(-1,1,1,1)*np.float32(.9)
        before=trainer.get_weights();result=trainer.train(x,y,epochs=1,batch_size=8)
        self.assertEqual(result['samples'],16);self.assertEqual(trainer.training_step,2)
        self.assertTrue(any(not np.array_equal(a,b) for a,b in zip(before,trainer.get_weights())))
        trainer.set_weights(before,0);self.assertEqual(trainer.training_step,0)
        for a,b in zip(before,trainer.get_weights()):np.testing.assert_array_equal(a,b)
        validate_artifact((self.directory/'round-1-local.iutfw').read_bytes(),trainer.transfer_context)
        with self.assertRaises(ValueError):trainer.set_weights(before,1)
        with self.assertRaises(ValueError):trainer.train(x,np.full((16,1),-1,np.float32))
        with self.assertRaisesRegex(ValueError,'MINIBATCH_STEP_BUDGET'):
            trainer.train(np.zeros((512,20,11,1),np.float32),
                          (np.arange(512)%2).astype('float32').reshape(-1,1),epochs=3,batch_size=1)

    def exchange_fixture(self,wrong_final=False):
        control=self.directory/'controller';control.mkdir()
        directories={peer:self.directory/peer for peer in ('sl-a','sl-b')}
        for directory in directories.values():directory.mkdir()
        records={peer:[] for peer in directories};decisions=[]
        base=encode_weights(weights(.25));base_tensor=validate_artifact(base)[2]
        for rid,a,b,expected in [(1,0.,2.,1.5),(2,2.5,4.5,4.)]:
            aggregator='sl-a' if rid==1 else 'sl-b'
            raw_local={p:encode_weights(weights(value),1,context('epoch',rid,p,'local'))
                       for p,value in [('sl-a',a),('sl-b',b)]}
            contributions={p:dict(samples=1 if p=='sl-a' else 3,weights_sha256=sha(raw)) for p,raw in raw_local.items()}
            global_raw=encode_weights(weights(5. if wrong_final and rid==2 else expected),0,
                                      context('epoch',rid,aggregator,'global'))
            global_tensor=validate_artifact(global_raw,context('epoch',rid,aggregator,'global'))[2]
            message=dict(protocol_version=1,job_id='cnn-unit',sn_instance_id='epoch',round_id=rid,
                model_id=CNNTemplate().identity,base_version=rid-1,base_sha256=sha(base),producer_peer_id=aggregator,
                global_version=rid,weights_sha256=sha(global_raw),weights_b64=base64.b64encode(global_raw).decode(),
                contributions=contributions)
            for peer,directory in directories.items():
                (directory/f'round-{rid}-local.iutfw').write_bytes(raw_local[peer])
                (directory/f'round-{rid}-global.iutfw').write_bytes(global_raw)
                write_json(directory/f'round-{rid}-global-message.json',message)
                records[peer].append(dict(peer_id=peer,round_id=rid,aggregator=aggregator,base_version=rid-1,
                    base_sha256=sha(base),global_version=rid,local_sha256=sha(raw_local[peer]),global_sha256=sha(global_raw),
                    global_tensor_sha256=global_tensor,base_tensor_sha256=base_tensor,
                    training_start_tensor_sha256=base_tensor,samples=contributions[peer]['samples']))
            decisions.append(dict(round_id=rid,base_sha256=sha(base),global_sha256=sha(global_raw),
                                  client_tensor_sha256=global_tensor,clients_completed=['sl-a','sl-b']))
            base=global_raw;base_tensor=global_tensor
        write_json(control/'result.json',dict(status='PASS',rounds=decisions,model_id=CNNTemplate().identity,
                                              job_id='cnn-unit',instance_id='epoch'))
        for peer,directory in directories.items():
            write_json(directory/'result.json',dict(status='PASS_CNN_SWARM_CLIENT',peer_id=peer,
                model_id=CNNTemplate().identity,rounds=records[peer],dataset_sha256='0'*64,
                label_provenance='DATASET_PROVIDED_MAPPED',independent_ground_truth_verified=False))
        return control,directories['sl-a'],directories['sl-b']

    def test_independent_replay_known_unequal_sample_counts(self):
        from nids_ml.adapters.swarm.verify import verify_exchange
        result=verify_exchange(*self.exchange_fixture())
        self.assertEqual(result['status'],'PASS_CNN_FEDAVG_REPLAY')
        self.assertEqual([r['fedavg_max_absolute_error'] for r in result['rounds']],[0.,0.])

    def test_coherently_rehashed_wrong_average_is_rejected(self):
        from nids_ml.adapters.swarm.verify import verify_exchange
        with self.assertRaisesRegex(ValueError,'FEDAVG_ARITHMETIC_MISMATCH'):
            verify_exchange(*self.exchange_fixture(wrong_final=True))

    def test_outer_global_round_context_is_validated(self):
        from nids_ml.adapters.swarm.verify import verify_exchange
        paths=self.exchange_fixture();path=paths[1]/'round-2-global-message.json'
        value=json.loads(path.read_text());value['round_id']=1;path.write_text(json.dumps(value),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'GLOBAL_MESSAGE_CONTEXT'):verify_exchange(*paths)


@unittest.skipUnless(SL_AVAILABLE,'Set NIDS_SL_SOURCE to the existing SL SDK source for interoperability checks')
class ExistingSLCompatibilityTests(unittest.TestCase):
    def state(self):
        from sl_runtime.exchange_state import ExchangeState
        state=ExchangeState('sl-a',('sl-a','sl-b'),encode_weights(weights()),'cnn-unit',CNNTemplate(),.1,lambda:self.now)
        snapshot=SimpleNamespace(round_id=1,instance_id='epoch',participants=('sl-a','sl-b'),status='waiting',ready_peers=())
        state.observe(snapshot,'epoch',{'sl-a','sl-b'})
        return state
    def setUp(self):self.now=0.
    def message(self,state):
        raw=encode_weights(weights(.1),1,context('epoch',1,'sl-a','local'))
        return dict(state.common(),producer_peer_id='sl-a',samples=3,weights_sha256=sha(raw),weights_b64=base64.b64encode(raw).decode())
    def test_unmodified_sl_codec_and_weighted_fedavg_arithmetic(self):
        from sl_runtime.model_data import decode,encode,fedavg
        spec=CNNTemplate();a=decode(encode_weights(weights(0.)),spec);b=decode(encode_weights(weights(2.)),spec)
        result=fedavg([(a,1),(b,3)],spec)
        for tensor in result:self.assertTrue(all(v==1.5 for v in tensor))
        validate_artifact(encode(result,spec))
    def test_duplicate_stale_and_model_mismatch_rejected(self):
        from sl_runtime.model_data import ExchangeError
        state=self.state();message=self.message(state);state.update(message,local=True)
        with self.assertRaisesRegex(ExchangeError,'DUPLICATE_UPDATE'):state.update(message,local=True)
        with self.assertRaisesRegex(ExchangeError,'STALE_ROUND'):state.update(dict(message,round_id=2),local=True)
        with self.assertRaisesRegex(ExchangeError,'ROUND_MODEL'):state.update(dict(message,model_id='wrong'),local=True)
    def test_missing_peer_timeout_does_not_create_global_or_commit(self):
        from sl_runtime.model_data import ExchangeError
        state=self.state();state.update(self.message(state),local=True);self.now=.2;state.expire()
        self.assertEqual(state.round.phase,'failed');self.assertIsNone(state.round.pending)
        self.assertEqual(state.committed['version'],0)
        with self.assertRaises(ExchangeError):state.global_model(1)


@unittest.skipUnless(SWOP_AVAILABLE,'Set NIDS_SWOP_SOURCE to the existing SWOP SDK source for controller tests')
class CNNControllerTests(unittest.TestCase):
    def scenario(self,mode=None):
        from nids_ml.adapters.swarm.controller import CNNController,ControllerConfig
        from swop.controller_io import TransportError,HTTPError
        from swop.controller_receipts import ReceiptStore
        class Log:
            def __init__(self):self.items=[]
            def event(self,event,**values):self.items.append(dict(event=event,**values))
        clock=SimpleNamespace(now=0.)
        def sleep(value):clock.now+=max(.001,value)
        initial=encode_weights(weights())
        cfg=ControllerConfig('http://127.0.0.1:1',{'sl-a':'http://127.0.0.1:2','sl-b':'http://127.0.0.1:3'},
            {'sl-a':8,'sl-b':8},'cnn-unit',CNNTemplate().identity,sha(initial),'unit',
            startup_timeout=.3,round_timeout=.3,poll_interval=.01,request_timeout=.05)
        log=Log();store=ReceiptStore(cfg,log)
        class HTTP:
            def __init__(self):
                self.rid=0;self.posts=0;self.get_failures=1 if mode=='get_retry' else 0
                self.raw={0:initial,1:encode_weights(weights(.1),0,context('epoch',1,'sl-a','global')),
                          2:encode_weights(weights(.2),0,context('epoch',2,'sl-b','global'))}
            def snapshot(self,status='synchronized'):
                return dict(instance_id='epoch',round_id=self.rid,status=status,participants=['sl-a','sl-b'],
                    ready_peers=['sl-a','sl-b'] if status=='synchronized' else [],
                    pending_peers=[] if status=='synchronized' else ['sl-a','sl-b'])
            def session(self,peer):
                rid=self.rid;raw=self.raw[rid];aggregate=None if rid==0 else ('sl-a' if rid==1 else 'sl-b')
                value=dict(protocol_version=1,job_id=cfg.job,sn_instance_id='epoch',peer_id=peer,
                    participants=['sl-a','sl-b'],model_id='wrong' if mode=='model_mismatch' else cfg.model_id,
                    committed=dict(version=rid,round_id=rid or None,aggregator=aggregate,
                                   weights_sha256=sha(raw),weights_b64=base64.b64encode(raw).decode()),round=None)
                if rid:
                    base=sha(self.raw[rid-1])
                    value['round']=dict(protocol_version=1,job_id=cfg.job,sn_instance_id='epoch',round_id=rid,
                        model_id=cfg.model_id,base_version=rid-1,base_sha256=base,aggregator=aggregate,
                        phase='completed',error=None,received_peers=['sl-a','sl-b'],applied=True,global_sha256=sha(raw))
                    previous_context=None if rid==1 else context('epoch',1,'sl-a','global')
                    start=validate_artifact(self.raw[rid-1],previous_context)[2]
                    if mode=='wrong_carry' and rid==2:start=validate_artifact(initial)[2]
                    receipt=dict(protocol_version=1,run_id=cfg.run_id,job_id=cfg.job,sn_instance_id='epoch',peer_id=peer,
                        model_id=cfg.model_id,round_id=rid,base_version=rid-1,base_sha256=base,global_version=rid,
                        global_sha256=sha(raw),base_tensor_sha256=start,training_start_tensor_sha256=start,
                        global_tensor_sha256=validate_artifact(raw,context('epoch',rid,aggregate,'global'))[2],samples=8)
                    if not (mode=='missing_receipt' and peer=='sl-b'):store.accept(receipt)
                return value
            def request(self,method,url,path,body=None,timeout=1):
                if method=='GET' and self.get_failures:
                    self.get_failures-=1;raise TransportError('temporary read failure')
                if path=='/health':
                    return 200,dict(service='independent-sn',api_version='1',status='ok',storage='memory',
                                    instance_id='changed' if mode=='epoch_change' and self.rid else 'epoch')
                if path.startswith('/v1/peers'):
                    return 200,dict(instance_id='epoch',peers=[dict(peer_id=p,endpoint=u,alive=True) for p,u in cfg.peers.items()])
                if path=='/v1/ml/session':return 200,self.session(next(p for p,u in cfg.peers.items() if u==url))
                if method=='POST':
                    self.posts+=1;self.rid+=1
                    if mode=='lost_post':raise TransportError('create response lost')
                    return 201,self.snapshot('waiting')
                if not self.rid:raise HTTPError(404,'round_not_found')
                return 200,self.snapshot()
        http=HTTP();controller=CNNController(cfg,log,store,http,lambda:clock.now,sleep)
        return controller,http,store,log

    def test_two_rounds_and_transient_read_retry(self):
        controller,http,store,log=self.scenario('get_retry')
        result=controller.run()
        self.assertEqual(result['status'],'PASS');self.assertEqual(http.posts,2)
        self.assertEqual([r['aggregator'] for r in result['rounds']],['sl-a','sl-b'])
        self.assertTrue(any(item['event']=='GET_RETRY' for item in log.items))

    def test_ambiguous_create_response_is_never_retried(self):
        from swop.controller_io import ControllerError
        controller,http,_,_=self.scenario('lost_post')
        with self.assertRaisesRegex(ControllerError,'UNCERTAIN_NO_RETRY'):controller.run()
        self.assertEqual(http.posts,1);self.assertEqual(controller.records,[])

    def test_missing_completion_receipt_prevents_next_round(self):
        from swop.controller_io import ControllerError
        controller,http,_,_=self.scenario('missing_receipt')
        with self.assertRaisesRegex(ControllerError,'TIMEOUT'):controller.run()
        self.assertEqual(http.posts,1);self.assertEqual(controller.records,[])

    def test_model_mismatch_and_epoch_change_fail_closed(self):
        from swop.controller_io import ControllerError
        for mode,error in [('model_mismatch','SL_SESSION_IDENTITY'),('epoch_change','SN_EPOCH_CHANGED')]:
            controller,http,_,_=self.scenario(mode)
            with self.subTest(mode=mode),self.assertRaisesRegex(ControllerError,error):controller.run()
            self.assertEqual(controller.records,[])

    def test_wrong_previous_global_is_rejected_in_round_two(self):
        from swop.controller_io import ControllerError
        controller,http,_,_=self.scenario('wrong_carry')
        with self.assertRaisesRegex(ControllerError,'CARRY_OVER_MISMATCH'):controller.run()
        self.assertEqual(len(controller.records),1)

    def test_identical_receipt_is_idempotent_and_conflicting_receipt_rejected(self):
        from swop.controller_io import ControllerError
        controller,http,store,_=self.scenario()
        controller.run();receipt=store.for_round(2)['sl-a']
        self.assertTrue(store.accept(receipt)['replayed'])
        with self.assertRaises(ControllerError):store.accept(dict(receipt,global_tensor_sha256='0'*64))

    def test_missing_peer_times_out_without_creating_round(self):
        from nids_ml.adapters.swarm.controller import CNNController,ControllerConfig
        from swop.controller_io import ControllerError,HTTPError
        from swop.controller_receipts import ReceiptStore
        class Log:
            def event(self,*args,**kwargs):pass
        clock=SimpleNamespace(now=0.)
        def sleep(value):clock.now+=max(.001,value)
        cfg=ControllerConfig('http://127.0.0.1:1',{'sl-a':'http://127.0.0.1:2','sl-b':'http://127.0.0.1:3'},
            {'sl-a':8,'sl-b':8},'cnn-unit',CNNTemplate().identity,sha(encode_weights(weights())),'unit',
            startup_timeout=.2,round_timeout=.2,poll_interval=.01,request_timeout=.05)
        class HTTP:
            posts=0
            def request(self,method,url,path,body=None,timeout=1):
                if method=='POST':self.posts+=1;raise AssertionError('Must not create a round')
                if path=='/health':return 200,dict(service='independent-sn',api_version='1',status='ok',storage='memory',instance_id='epoch')
                if path=='/v1/rounds/current':raise HTTPError(404,'round_not_found')
                return 200,dict(instance_id='epoch',peers=[dict(peer_id='sl-a',endpoint=cfg.peers['sl-a'],alive=True)])
        http=HTTP();log=Log();store=ReceiptStore(cfg,log)
        with self.assertRaisesRegex(ControllerError,'TIMEOUT'):
            CNNController(cfg,log,store,http,lambda:clock.now,sleep).run()
        self.assertEqual(http.posts,0)


if __name__=='__main__':unittest.main()
