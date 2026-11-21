"""Run full-pool numeric exchange using unchanged SN/SL/User-TF services."""
from contextlib import closing
from dataclasses import asdict,dataclass
import hashlib
import json
from pathlib import Path
import re
import threading
import time
from types import SimpleNamespace

from .contract import encode_weights,fresh_output,read_artifact,sha,validate_artifact,write_json,require,PARTICIPANTS
from .full_job import FullTemplate,full_context,samples_for
from .job import http_request
from .client import send_receipt
from .batch_client import RoundBudget
from nids_ml.storage.planned import PlanBank,PlannedReader
from nids_ml.training.batched import StreamTrainer,TrainingLimits,evaluate
from nids_ml.pcap.decoder import regular_file


@dataclass(frozen=True)
class FullTrainingLimits(TrainingLimits):
    timeout_seconds:int=24*3600
    def validate(self):
        expected=FullTrainingLimits()
        require(all(type(getattr(self,k)) is int and getattr(self,k)==getattr(expected,k)
                    for k in self.__dataclass_fields__),'FULL_TRAINING_RESOURCE_PROFILE')
        return self


class FullTrainer:
    def __init__(self,spec,output,seed):
        self.spec=spec;self.output=output;self.trainer=StreamTrainer(seed)
        self.reader=None;self.transfer_context=None
    @property
    def training_step(self):return self.trainer.steps
    def get_weights(self):return self.trainer.get_weights()
    def set_weights(self,arrays,training_step=0):
        require(training_step==0,'FULL_GLOBAL_OPTIMIZER_RESET');encode_weights(arrays,template=self.spec)
        self.trainer.set_weights(arrays)
    def train(self,x,y,epochs=1,batch_size=128):
        require(x is self.reader and isinstance(x,PlannedReader) and y is None and epochs==1,'FULL_ONE_ROUND_READER')
        require(x.bank.manifest['plan_id']==self.spec.plan_id,'FULL_TRAIN_PLAN_ID')
        update=self.trainer.train_epoch(x,x.client,batch_size=batch_size,epoch=0)
        raw=encode_weights(self.get_weights(),self.training_step,self.transfer_context,template=self.spec)
        with (self.output/f'round-{x.round_id}-local.iutfw').open('xb') as stream:stream.write(raw)
        return dict(samples=len(x),epochs=1,updates=[update],subset=x.report(),optimizer='Adam reset after global application')


def run_controller(cfg,output,bind,port):
    from swop.controller import ControllerConfig
    from swop.controller_io import Journal,ensure
    from swop.controller_receipts import ReceiptStore,ReceiptServer,FIELDS
    from .controller import CNNController
    target=fresh_output(output);target.mkdir();journal=Journal(target/'events.jsonl')
    # The larger orchestration profile lives here; external SWOP config/source is unchanged.
    probe=ControllerConfig(cfg['sn_url'],cfg['peers'],samples_for(cfg,1),cfg['job_id'],cfg['model_id'],
        cfg['initial_sha256'],cfg['run_id'],(1,),cfg['startup_timeout'],cfg['round_timeout'],
        cfg['poll_interval'],cfg['request_timeout'],2)
    config=SimpleNamespace(**dict(asdict(probe),round_ids=tuple(range(1,cfg['total_rounds']+1))))
    class FullReceipts(ReceiptStore):
        def accept(self,body):
            with self.lock:
                ensure(type(body) is dict and set(body)==FIELDS,'RECEIPT_SCHEMA')
                rid=body['round_id'];peer=body['peer_id']
                ensure(type(rid) is int and rid in self.intents and type(peer) is str and peer in config.peers,'RECEIPT_ROUND_PEER')
                ensure(type(body['protocol_version']) is int and body['protocol_version']==1 and body['run_id']==config.run_id
                    and body['job_id']==config.job and body['model_id']==config.model_id
                    and body['sn_instance_id']==self.epoch,'RECEIPT_IDENTITY')
                ensure(type(body['samples']) is int and body['samples']==samples_for(cfg,rid)[peer],'RECEIPT_SAMPLES')
                ensure(type(body['global_version']) is int and body['global_version']==rid
                    and type(body['base_version']) is int and body['base_version']==rid-1,'RECEIPT_VERSION')
                for key in ('base_sha256','global_sha256','base_tensor_sha256','training_start_tensor_sha256','global_tensor_sha256'):
                    ensure(type(body[key]) is str and re.fullmatch('[a-f0-9]{64}',body[key]),'RECEIPT_DIGEST')
                ensure(body['base_tensor_sha256']==body['training_start_tensor_sha256'],'RECEIPT_BASE_NOT_APPLIED')
                key=rid,peer
                if key in self.items:
                    ensure(self.items[key]==body,'CONFLICTING_RECEIPT');return dict(protocol_version=1,replayed=True)
                self.journal.event('CLIENT_COMPLETED',receipt=body);self.items[key]=dict(body)
                return dict(protocol_version=1,replayed=False)
    class FullController(CNNController):
        maximum_rounds=cfg['total_rounds']
        transfer_context=staticmethod(full_context)
    server=None
    try:
        receipts=FullReceipts(config,journal)
        server=ReceiptServer(receipts,(bind,port),cfg['request_timeout']).start()
        write_json(target/'ready.json',dict(listener=server.url,model_id=cfg['model_id']))
        result=FullController(config,journal,receipts,template=FullTemplate(cfg['plan_id'])).run()
        # Keep the final durable receipt response alive until both clients read it.
        time.sleep(3*cfg['request_timeout']+2*cfg['poll_interval']+.5)
        result.update(plan_id=cfg['plan_id'],model_id=cfg['model_id'],job_id=cfg['job_id'],run_id=cfg['run_id'],
                      label_provenance='MIXED_EXPLICIT_PER_SOURCE',independent_ground_truth_verified=False)
        write_json(target/'result.json',result);return result
    except BaseException as error:
        write_json(target/'FAILED.json',dict(error=str(error),automatic_retry=False));raise
    finally:
        if server:server.close()
        journal.close()


def run_client(cfg,peer,plan,output,*,workspace,policy_path,policy_sha,sources=None):
    from user_tf.exchange import MLExchangeClient,ModelExchangeError
    from nids_ml.models.flow_cnn import runtime_info
    client_id=PARTICIPANTS.index(peer)
    bank=PlanBank(plan,workspace,client_id,policy_path=policy_path,policy_sha=policy_sha,sources=sources)
    m=bank.manifest
    require(m['plan_id']==cfg['plan_id'] and m['clients'][str(client_id)]['train_counts']==cfg['pool_counts'][str(client_id)]
            and m['total_rounds']==cfg['total_rounds'],'FULL_PLAN_JOB_BINDING')
    require(not m['missing_requested_families'] or cfg['missing_families_accepted'],'FULL_FAMILIES_NOT_ACCEPTED')
    bank.preflight()
    target=fresh_output(output);target.mkdir();budget=RoundBudget(target,FullTrainingLimits());completed=0;passes=[]
    try:
        trainer=FullTrainer(FullTemplate(cfg['plan_id']),target,cfg['seed'])
        deadline=time.monotonic()+cfg['startup_timeout']
        while True:
            try:
                _,health=http_request('GET',cfg['sn_url'],'/health',timeout=cfg['request_timeout'])
                require(health.get('service')=='independent-sn' and health.get('api_version')=='1','FULL_SN_IDENTITY')
                epoch=health['instance_id'];break
            except OSError:
                require(time.monotonic()<deadline,'FULL_STARTUP_TIMEOUT');time.sleep(cfg['poll_interval'])
        class PacedClient(MLExchangeClient):
            next_session=0.
            def session(self,*args):
                time.sleep(max(0.,self.next_session-time.monotonic()));self.next_session=time.monotonic()+cfg['poll_interval']
                return super().session(*args)
        sdk=PacedClient(cfg['peers'][peer],peer,PARTICIPANTS,cfg['job_id'],epoch,request_timeout=cfg['request_timeout'])
        for rid in range(1,cfg['total_rounds']+1):
            deadline=time.monotonic()+cfg['round_timeout']
            while True:
                require(time.monotonic()<deadline,'FULL_ROUND_WAIT_TIMEOUT')
                _,health=http_request('GET',cfg['sn_url'],'/health',timeout=cfg['request_timeout'])
                require(health.get('instance_id')==epoch,'FULL_SN_EPOCH_CHANGED')
                try:
                    session=sdk.session(trainer.spec,rid)
                    require(session['round']['phase']=='collecting','FULL_ROUND_NOT_COLLECTING');break
                except ModelExchangeError as error:
                    if not any(code in str(error) for code in ('Stale round','SN_UNAVAILABLE')):raise
                time.sleep(cfg['poll_interval'])
            budget.deadline=time.monotonic()+cfg['round_timeout']
            with closing(bank.reader('train',round_id=rid,scratch=target,budget=budget)) as reader:
                require(len(reader)==samples_for(cfg,rid)[peer],'FULL_ROUND_CONTRIBUTION_COUNT')
                trainer.reader=reader;trainer.transfer_context=full_context(epoch,rid,peer,'local')
                result=sdk.train_round(trainer,reader,None,rid,epochs=1,batch_size=cfg['batch_size'],
                                      timeout=cfg['round_timeout'],checkpoint_dir=None)
            raw=encode_weights(trainer.get_weights(),0,full_context(epoch,rid,result['aggregator'],'global'),template=trainer.spec)
            require(sha(raw)==result['global_sha256'],'FULL_GLOBAL_RECONSTRUCTION')
            with (target/f'round-{rid}-global.iutfw').open('xb') as stream:stream.write(raw)
            message=sdk.request('GET',f'/v1/ml/global/{rid}')
            require(message['weights_sha256']==sha(raw) and set(message['contributions'])==set(PARTICIPANTS)
                    and all(message['contributions'][p]['samples']==samples_for(cfg,rid)[p] for p in PARTICIPANTS),
                    'FULL_GLOBAL_CONTRIBUTIONS')
            write_json(target/f'round-{rid}-global-message.json',message)
            if rid%m['rounds_per_pass']==0:
                with closing(bank.reader('validation',scratch=target,budget=budget,
                        progress=lambda event:print(json.dumps(dict(event,peer_id=peer)),flush=True))) as validation:
                    metrics=evaluate(trainer.trainer,validation,'validation',client_id,batch_size=cfg['batch_size'])
                entry=dict(pass_id=rid//m['rounds_per_pass'],round_id=rid,global_sha256=sha(raw),validation=metrics)
                write_json(target/f'pass-{entry["pass_id"]}-validation.json',entry);passes.append(entry)
            write_json(target/f'round-{rid}.json',result)
            receipt={k:result[k] for k in ('round_id','base_version','base_sha256','global_version','global_sha256',
                'base_tensor_sha256','training_start_tensor_sha256','global_tensor_sha256','samples')}
            receipt.update(protocol_version=1,run_id=cfg['run_id'],job_id=cfg['job_id'],sn_instance_id=epoch,
                           peer_id=peer,model_id=trainer.spec.identity)
            send_receipt(cfg,receipt);completed=rid
            print(json.dumps(dict(stage='full_pool_round_completed',peer=peer,round_id=rid,total_rounds=cfg['total_rounds'],
                samples=result['samples'],global_sha256=sha(raw))),flush=True)
        budget.deadline=None
        final=encode_weights(trainer.get_weights(),template=trainer.spec)
        with (target/'final.iutfw').open('xb') as stream:stream.write(final)
        checkpoint=dict(format='nids-full-plan-cnn-checkpoint/1.0.0',plan_id=m['plan_id'],model_id=trainer.spec.identity,
            weights_sha256=sha(final),round_id=completed,seed=cfg['seed'],optimizer_state='NOT_SAVED; reset at each global',
            label_provenance=m['label_provenance'],scope=m['scope'],independent_ground_truth_verified=False)
        write_json(target/'checkpoint.json',checkpoint)
        report=dict(status='PASS_FULL_POOL_SWARM_CLIENT',peer_id=peer,client=client_id,plan_id=m['plan_id'],
            model_id=trainer.spec.identity,job_id=cfg['job_id'],run_id=cfg['run_id'],completed_rounds=completed,
            passes=m['passes'],pool_samples=sum(bank.entry['train_counts'].values()),
            training_samples_seen=sum(bank.entry['train_counts'].values())*m['passes'],full_pool_seen_every_pass=True,
            validation=passes,test=dict(status='NOT_EVALUATED; frozen checkpoint requires separate explicit test evaluation',samples=0),
            final_weights_sha256=sha(final),runtime=runtime_info(),label_provenance=m['label_provenance'],
            source_label_provenance={s['dataset_id']:s['label_provenance'] for s in m['sources']},
            missing_requested_families=m['missing_requested_families'],scope=m['scope'],independent_ground_truth_verified=False,
            resources=dict(peak_python_rss_bytes=budget.peak_python_rss,peak_scratch_bytes=budget.peak_scratch))
        write_json(target/'result.json',report);return report
    except BaseException as error:
        write_json(target/'FAILED.json',dict(error=str(error),completed_rounds=completed,automatic_retry=False));raise


def evaluate_checkpoint(plan,workspace,client,checkpoint,output,*,policy_path,policy_sha,sources=None,
                        split='validation',threshold=.5,evaluate_test_approved=False):
    from nids_ml.evaluation.thresholds import validate_threshold
    require(split in ('validation','test') and (split!='test' or evaluate_test_approved),'FULL_TEST_APPROVAL_REQUIRED')
    threshold=validate_threshold(threshold)
    bank=PlanBank(plan,workspace,client,policy_path=policy_path,policy_sha=policy_sha,sources=sources)
    spec=FullTemplate(bank.manifest['plan_id']);checkpoint=Path(checkpoint)
    value=json.loads(regular_file(checkpoint/'checkpoint.json').read_bytes())
    require(value['format']=='nids-full-plan-cnn-checkpoint/1.0.0' and value['plan_id']==spec.plan_id
            and value['model_id']==spec.identity,'FULL_CHECKPOINT_BINDING')
    raw=read_artifact(checkpoint/'final.iutfw',template=spec)
    require(sha(raw)==value['weights_sha256'],'FULL_CHECKPOINT_SHA')
    _,blobs,_=validate_artifact(raw,template=spec)
    target=fresh_output(output);target.mkdir();trainer=StreamTrainer(value['seed'])
    weights=[trainer.np.frombuffer(blob,dtype='<f4').reshape(shape).copy() for blob,(_,shape) in zip(blobs,spec.tensors)]
    trainer.set_weights(weights);before=hashlib.sha256(b''.join(v.tobytes() for v in trainer.get_weights())).hexdigest()
    with closing(bank.reader(split,progress=lambda event:print(json.dumps(event),flush=True))) as reader:
        metrics=evaluate(trainer,reader,split,client,threshold=threshold)
    require(before==hashlib.sha256(b''.join(v.tobytes() for v in trainer.get_weights())).hexdigest(),'FULL_EVAL_CHANGED_WEIGHTS')
    report=dict(status='PASS_FULL_PLAN_FROZEN_EVALUATION',plan_id=spec.plan_id,model_id=spec.identity,client=client,
        split=split,threshold=threshold,weights_sha256=sha(raw),metrics=metrics,training=False,
        scope=bank.manifest['scope'],evaluation_placement=bank.manifest['evaluation_placement'])
    write_json(target/'metrics.json',report);return report
