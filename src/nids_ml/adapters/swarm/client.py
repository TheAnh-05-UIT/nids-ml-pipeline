"""Adapt the existing exchange SDK to the fixed CNN and mapped HDF5 client partitions.
Kết nối CNN/HDF5 với SDK exchange có sẵn; không sửa User-TF hoặc tự retry model upload.
"""
import json
from pathlib import Path
import time

from .contract import (CNNTemplate, context, encode_weights, fresh_output, read_artifact,
                       require, sha, validate_artifact, write_json)


def send_receipt(cfg,receipt,request=None,sleep=time.sleep):
    from .job import http_request
    request=request or http_request
    for attempt in range(3):
        try:
            status,_=request('POST',cfg['controller_url'],'/v1/controller/receipts',receipt,cfg['request_timeout'])
            require(status==200,'RECEIPT_REJECTED')
            return
        except OSError:
            if attempt==2:raise
            sleep(cfg['poll_interval'])


class CNNTrainer:
    def __init__(self, output, seed=42):
        from nids_ml.models.flow_cnn import backend, build_model, configuration
        self.np, self.tf = backend()
        self.spec = CNNTemplate()
        self.config = configuration(seed=seed)
        self.model = build_model(self.config)
        self.training_step = 0
        self.round_id = 0
        self.transfer_context = None
        self.output = Path(output)

    def get_weights(self):
        weights = self.model.get_weights()
        encode_weights(weights)
        return tuple(weights)

    def set_weights(self, arrays, training_step=0):
        require(type(training_step) is int and training_step == 0, 'GLOBAL_OPTIMIZER_STEP_MUST_RESET')
        encode_weights(arrays)
        self.model.set_weights(list(arrays))
        ref = self.config['training_reference']
        self.model.optimizer = self.tf.keras.optimizers.Adam(learning_rate=ref['learning_rate'],
            beta_1=ref['adam_beta_1'], beta_2=ref['adam_beta_2'], epsilon=ref['adam_epsilon'])
        self.training_step = 0

    def predict(self, x):
        from nids_ml.models.flow_cnn import predict_flow
        np = self.np
        return np.concatenate([predict_flow(self.model, x[i:i+128], self.config) for i in range(0,len(x),128)])

    def train(self, x, y, epochs=1, batch_size=64):
        from nids_ml.models.flow_cnn import train_step
        from nids_ml.training.demo import metrics
        np = self.np
        require(type(epochs) is int and 1 <= epochs <= 3 and type(batch_size) is int and 1 <= batch_size <= 128, 'TRAINING_BUDGET')
        require(isinstance(x,np.ndarray) and x.dtype==np.float32 and x.shape[1:]==(20,11,1)
                and 2 <= len(x) <= 4096 and np.isfinite(x).all() and ((x>=0)&(x<=1)).all(), 'TRAINING_INPUT')
        require(y.dtype==np.float32 and y.shape==(len(x),1) and set(y.reshape(-1))=={0.,1.}, 'TRAINING_LABELS')
        require(((len(x)+batch_size-1)//batch_size)*epochs<=1024,'MINIBATCH_STEP_BUDGET')
        require(self.transfer_context is not None and self.round_id in (1,2), 'ROUND_CONTEXT_REQUIRED')
        weights = {label:len(x)/(2*int((y==label).sum())) for label in (0,1)}
        before = metrics(y, self.predict(x))
        for epoch in range(epochs):
            order = np.random.default_rng(42 + 100*self.round_id + epoch).permutation(len(x))
            for start in range(0,len(x),batch_size):
                indices = order[start:start+batch_size]
                sample_weights = np.asarray([weights[int(v)] for v in y[indices,0]],dtype='float32')
                train_step(self.model,x[indices],y[indices],self.config,sample_weights=sample_weights)
                self.training_step += 1
        raw = encode_weights(self.get_weights(), self.training_step, self.transfer_context)
        with (self.output/f'round-{self.round_id}-local.iutfw').open('xb') as stream:
            stream.write(raw)
        return dict(samples=len(x),epochs=epochs,steps=self.training_step,loss_before=before['loss'],
                    loss_after=metrics(y,self.predict(x))['loss'],optimizer='Adam reset at global application')


def run_client(cfg, peer, dataset, output):
    from user_tf.exchange import MLExchangeClient, ModelExchangeError
    from nids_ml.data.demo import read_dataset
    from nids_ml.models.flow_cnn import runtime_info
    from nids_ml.training.demo import metrics
    from .job import http_request

    class PacedClient(MLExchangeClient):
        next_session = 0.
        def session(self,*args):
            time.sleep(max(0.,self.next_session-time.monotonic()))
            self.next_session=time.monotonic()+cfg['poll_interval']
            return super().session(*args)

    manifest,x,y,splits,clients,metadata = read_dataset(dataset)
    require(manifest['hdf5']['sha256']==cfg['dataset_sha256'], 'DATASET_IDENTITY')
    require(peer in cfg['peers'], 'PEER_IDENTITY')
    client_index = sorted(cfg['peers']).index(peer)
    selected = (splits==0)&(clients==client_index)
    require(int(selected.sum())==cfg['samples'][peer], 'CLIENT_SAMPLE_COUNT')
    xtrain=x[selected]; ytrain=y[selected].astype('float32').reshape(-1,1)
    target=fresh_output(output); target.mkdir()
    trainer=CNNTrainer(target)
    deadline=time.monotonic()+cfg['startup_timeout']
    while True:
        try:
            _,health=http_request('GET',cfg['sn_url'],'/health',timeout=cfg['request_timeout'])
            require(health.get('service')=='independent-sn' and health.get('api_version')=='1', 'SN_HEALTH')
            epoch=health['instance_id'];break
        except OSError:
            require(time.monotonic()<deadline,'STARTUP_TIMEOUT');time.sleep(cfg['poll_interval'])
    client=PacedClient(cfg['peers'][peer],peer,sorted(cfg['peers']),cfg['job_id'],epoch,
                       request_timeout=cfg['request_timeout'])
    records=[]
    try:
        for rid in (1,2):
            deadline=time.monotonic()+cfg['round_timeout']
            while True:
                require(time.monotonic()<deadline,'ROUND_WAIT_TIMEOUT')
                _,health=http_request('GET',cfg['sn_url'],'/health',timeout=cfg['request_timeout'])
                require(health.get('instance_id')==epoch,'SN_EPOCH_CHANGED')
                try:
                    current=client.session(trainer.spec,rid)
                    require(current['round']['phase']=='collecting','ROUND_NOT_COLLECTING')
                    break
                except ModelExchangeError as error:
                    if not any(code in str(error) for code in ('Stale round','SN_UNAVAILABLE')):
                        raise
                time.sleep(cfg['poll_interval'])
            trainer.round_id=rid;trainer.transfer_context=context(epoch,rid,peer,'local')
            result=client.train_round(trainer,xtrain,ytrain,rid,epochs=cfg['epochs'],batch_size=cfg['batch_size'],
                                      timeout=cfg['round_timeout'],checkpoint_dir=None)
            raw=encode_weights(trainer.get_weights(),0,context(epoch,rid,result['aggregator'],'global'))
            require(sha(raw)==result['global_sha256'],'GLOBAL_ARTIFACT_RECONSTRUCTION')
            global_path=target/f'round-{rid}-global.iutfw'
            with global_path.open('xb') as stream:stream.write(raw)
            message=client.request('GET',f'/v1/ml/global/{rid}')
            require(message['weights_sha256']==sha(raw),'GLOBAL_MESSAGE_CHANGED')
            require(set(message['contributions'])==set(cfg['peers']) and all(
                message['contributions'][p]['samples']==cfg['samples'][p] for p in cfg['peers']),
                'GLOBAL_CONTRIBUTION_COUNTS')
            write_json(target/f'round-{rid}-global-message.json',message)
            original=trainer.predict(x[splits==1])
            restored_raw=read_artifact(global_path,context(epoch,rid,result['aggregator'],'global'))
            _,blobs,_=validate_artifact(restored_raw,context(epoch,rid,result['aggregator'],'global'))
            restored=[trainer.np.frombuffer(blob,dtype='<f4').reshape(shape).copy()
                      for blob,(_,shape) in zip(blobs,trainer.spec.tensors)]
            probe=CNNTrainer(target)
            probe.set_weights(restored,0)
            error=float(trainer.np.max(trainer.np.abs(original-probe.predict(x[splits==1]))))
            require(error<=1e-7,'CHECKPOINT_RESTORE')
            record=dict(result,validation=metrics(y[splits==1],original),checkpoint_restore_max_error=error)
            write_json(target/f'round-{rid}.json',record);records.append(record)
            receipt={key:result[key] for key in ('round_id','base_version','base_sha256','global_version',
                     'global_sha256','base_tensor_sha256','training_start_tensor_sha256','global_tensor_sha256','samples')}
            receipt.update(protocol_version=1,run_id=cfg['run_id'],job_id=cfg['job_id'],sn_instance_id=epoch,
                           peer_id=peer,model_id=trainer.spec.identity)
            # Only identical completion receipts may retry; training/model uploads never retry.
            # Chỉ retry receipt giống hệt; không tự train hoặc upload lại weights.
            send_receipt(cfg,receipt)
            print(json.dumps(dict(event='CNN_ROUND_COMPLETED',peer=peer,round_id=rid,global_sha256=sha(raw))),flush=True)
        report=dict(status='PASS_CNN_SWARM_CLIENT',peer_id=peer,client=client_index,training_samples=len(xtrain),
                    dataset_sha256=cfg['dataset_sha256'],model_id=trainer.spec.identity,runtime=runtime_info(),rounds=records,
                    test=metrics(y[splits==2],trainer.predict(x[splits==2])),label_provenance=manifest['label_provenance'],
                    independent_ground_truth_verified=False,native_ARM64='REQUIRES_HOST_EVIDENCE')
        write_json(target/'result.json',report);return report
    except Exception as error:
        write_json(target/'FAILED.json',dict(error=str(error),completed_rounds=len(records),automatic_retry=False));raise
