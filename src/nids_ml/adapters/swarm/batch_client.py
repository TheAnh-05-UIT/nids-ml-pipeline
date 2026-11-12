"""Train rotating catalog subsets through the unchanged User-TF exchange SDK.
Train theo batch, trao weights qua SL; xác nhận apply/checkpoint trước completion receipt.
"""
import json
import time

from .batch_data import CatalogTemplate, RoundReader, SUBSET_POLICY
from .client import send_receipt
from .contract import context, encode_weights, fresh_output, read_artifact, require, sha, write_json
from .job import http_request
from nids_ml.alignment.full import Budget
from nids_ml.training.batched import StreamTrainer, TrainingLimits, evaluate, save_checkpoint, restore_checkpoint


class RoundBudget(Budget):
    deadline = None

    def check(self, *args, **kwargs):
        require(self.deadline is None or time.monotonic()<self.deadline, 'LOCAL_TRAINING_DEADLINE')
        return super().check(*args, **kwargs)


class BatchTrainer:
    """Duck-typed SDK trainer: len(reader) is the true round contribution count.
    SDK nhận reader thay ndarray; không giả số mẫu hoặc load toàn pool vào RAM.
    """
    def __init__(self, template, output):
        self.spec,self.output=template,output
        self.trainer=StreamTrainer()
        self.reader=None
        self.transfer_context=None

    @property
    def training_step(self):
        return self.trainer.steps

    def get_weights(self):
        return self.trainer.get_weights()

    def set_weights(self, arrays, training_step=0):
        require(type(training_step) is int and training_step==0,'GLOBAL_OPTIMIZER_STEP_MUST_RESET')
        encode_weights(arrays,template=self.spec)
        self.trainer.set_weights(arrays)

    def train(self, x, y, epochs=1, batch_size=64):
        require(x is self.reader and y is None and isinstance(x,RoundReader), 'BOUND_CATALOG_READER_REQUIRED')
        require(x.manifest['catalog_id']==self.spec.catalog_id and self.transfer_context is not None,
                'TRAIN_CATALOG_CONTEXT')
        require(type(epochs) is int and 1<=epochs<=3 and type(batch_size) is int
                and 1<=batch_size<=128,'TRAINING_BUDGET')
        updates=[]
        for epoch in range(epochs):
            updates.append(self.trainer.train_epoch(x,x.client,batch_size=batch_size,
                epoch=(x.round_id-1)*epochs+epoch))
        raw=encode_weights(self.get_weights(),self.training_step,self.transfer_context,template=self.spec)
        with (self.output/f'round-{x.round_id}-local.iutfw').open('xb') as stream:
            stream.write(raw)
        return dict(samples=len(x),epochs=epochs,steps=self.training_step,updates=updates,
                    subset=x.report(),optimizer='Adam reset on every global application')


def run_batch_client(cfg, peer, catalog, output, *, workspace, sources=None):
    from user_tf.exchange import MLExchangeClient, ModelExchangeError
    from nids_ml.models.flow_cnn import runtime_info

    require(peer in cfg['peers'],'PEER_IDENTITY')
    index=sorted(cfg['peers']).index(peer)
    # Validate both round memberships and all local inputs before network/training.
    # Kiểm catalog, subset và đủ data trước khi đăng ký completion hoặc upload.
    readers=[RoundReader(catalog,workspace,index,rid,cfg['subset_cap'],sources=sources) for rid in (1,2)]
    for reader in readers:
        require(reader.manifest['catalog_id']==cfg['catalog_id'] and reader.pool==cfg['pool_samples'][peer]
                and len(reader)==cfg['samples'][peer], 'CATALOG_JOB_SAMPLE_MISMATCH')
        for split in ('train','validation','test'):reader.preflight(split,index)
    target=fresh_output(output);target.mkdir()
    budget=RoundBudget(target,TrainingLimits());records=[];used=set()
    try:
        trainer=BatchTrainer(CatalogTemplate(cfg['catalog_id']),target)
        deadline=time.monotonic()+cfg['startup_timeout']
        while True:
            try:
                _,health=http_request('GET',cfg['sn_url'],'/health',timeout=cfg['request_timeout'])
                require(health.get('service')=='independent-sn' and health.get('api_version')=='1','SN_HEALTH')
                epoch=health['instance_id'];break
            except OSError:
                require(time.monotonic()<deadline,'STARTUP_TIMEOUT');time.sleep(cfg['poll_interval'])

        class PacedClient(MLExchangeClient):
            next_session=0.
            def session(self,*args):
                time.sleep(max(0.,self.next_session-time.monotonic()))
                self.next_session=time.monotonic()+cfg['poll_interval']
                return super().session(*args)

        client=PacedClient(cfg['peers'][peer],peer,sorted(cfg['peers']),cfg['job_id'],epoch,
                           request_timeout=cfg['request_timeout'])
        for rid,reader in enumerate(readers,1):
            deadline=time.monotonic()+cfg['round_timeout']
            while True:
                require(time.monotonic()<deadline,'ROUND_WAIT_TIMEOUT')
                _,health=http_request('GET',cfg['sn_url'],'/health',timeout=cfg['request_timeout'])
                require(health.get('instance_id')==epoch,'SN_EPOCH_CHANGED')
                try:
                    session=client.session(trainer.spec,rid)
                    require(session['round']['phase']=='collecting','ROUND_NOT_COLLECTING');break
                except ModelExchangeError as error:
                    if not any(code in str(error) for code in ('Stale round','SN_UNAVAILABLE')):raise
                time.sleep(cfg['poll_interval'])
            reader.scratch=target;reader.budget=budget;budget.deadline=time.monotonic()+cfg['round_timeout']
            trainer.reader=reader;trainer.transfer_context=context(epoch,rid,peer,'local')
            result=client.train_round(trainer,reader,None,rid,epochs=cfg['epochs'],batch_size=cfg['batch_size'],
                                      timeout=cfg['round_timeout'],checkpoint_dir=None)
            raw=encode_weights(trainer.get_weights(),0,context(epoch,rid,result['aggregator'],'global'),template=trainer.spec)
            require(sha(raw)==result['global_sha256'],'GLOBAL_ARTIFACT_RECONSTRUCTION')
            with (target/f'round-{rid}-global.iutfw').open('xb') as stream:stream.write(raw)
            message=client.request('GET',f'/v1/ml/global/{rid}')
            require(message['weights_sha256']==sha(raw) and set(message['contributions'])==set(cfg['peers']) and all(
                message['contributions'][p]['samples']==cfg['samples'][p] for p in cfg['peers']), 'GLOBAL_CONTRIBUTION_COUNTS')
            write_json(target/f'round-{rid}-global-message.json',message)
            checkpoint_dir=target/f'round-{rid}-checkpoint';checkpoint_dir.mkdir()
            save_checkpoint(trainer.trainer,checkpoint_dir,cfg['catalog_id'])
            probe,_=restore_checkpoint(checkpoint_dir,expected_catalog_id=cfg['catalog_id'])
            error=0.
            for batch in reader.batches('validation',index,batch_size=cfg['batch_size']):
                error=max(error,float(trainer.trainer.np.max(trainer.trainer.np.abs(
                    trainer.trainer.predict(batch['X'])-probe.predict(batch['X'])))))
            require(error<=1e-7,'CHECKPOINT_RESTORE')
            record=dict(result,validation=evaluate(trainer.trainer,reader,'validation',index,
                        batch_size=cfg['batch_size']),checkpoint_restore_max_error=error)
            write_json(target/f'round-{rid}.json',record);records.append(record)
            used.update(result['training']['subset']['sample_ids'])
            receipt={k:result[k] for k in ('round_id','base_version','base_sha256','global_version','global_sha256',
                     'base_tensor_sha256','training_start_tensor_sha256','global_tensor_sha256','samples')}
            receipt.update(protocol_version=1,run_id=cfg['run_id'],job_id=cfg['job_id'],sn_instance_id=epoch,
                           peer_id=peer,model_id=trainer.spec.identity)
            send_receipt(cfg,receipt)
            print(json.dumps(dict(event='NIDS_BATCH_ROUND_COMPLETED',peer=peer,round_id=rid,
                samples=result['samples'],global_sha256=sha(raw))),flush=True)
        budget.deadline=None
        with (target/'test-predictions.csv').open('x',encoding='utf-8',newline='') as stream:
            test=evaluate(trainer.trainer,readers[-1],'test',index,batch_size=cfg['batch_size'],predictions=stream)
        report=dict(status='PASS_NIDS_BATCH_SWARM_CLIENT',peer_id=peer,client=index,job_id=cfg['job_id'],
            run_id=cfg['run_id'],catalog_id=cfg['catalog_id'],model_id=trainer.spec.identity,
            pool_samples=readers[0].pool,per_round_samples=cfg['samples'][peer],unique_training_samples=len(used),
            full_pool_seen_across_rounds=len(used)==readers[0].pool,subset_policy=SUBSET_POLICY,
            runtime=runtime_info(),rounds=records,test=test,label_provenance='DATASET_PROVIDED_MAPPED',
            independent_ground_truth_verified=False,native_ARM64='REQUIRES_HOST_EVIDENCE',
            resources=dict(peak_python_rss_bytes=budget.peak_python_rss,peak_scratch_bytes=budget.peak_scratch))
        write_json(target/'result.json',report);return report
    except Exception as error:
        write_json(target/'FAILED.json',dict(error=str(error),completed_rounds=len(records),automatic_retry=False));raise
