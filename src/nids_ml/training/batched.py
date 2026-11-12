"""Train/evaluate the fixed CNN from a deduplicated catalog using bounded batches.
Train/evaluate CNN theo batch, giữ contract Normal/Attack riêng và checkpoint không pickle.
"""
from collections import Counter
from dataclasses import dataclass
import csv
import hashlib
import os
from pathlib import Path

from nids_ml.alignment.full import Budget
from nids_ml.flows.mapped import FlowLimits,new_output,require,PROVENANCE
from nids_ml.labeling.contracts import canonical_bytes,strict_json
from nids_ml.ordering.view import _new_json
from nids_ml.pcap.decoder import regular_file,file_sha256


def model_description():
    # Reuse geometry only: the pilot DDoS label mapping is not the new task contract.
    # Chỉ dùng lại kiến trúc; Normal/Attack có identity riêng, không tái dùng model ID pilot.
    from nids_ml.adapters.swarm.contract import CNNTemplate
    return dict(format='nids-binary-flow-cnn/1.0.0',architecture=CNNTemplate().description(),
        target={'0':'Normal','1':'Attack'},label_provenance=PROVENANCE,
        normalization='lucid-fixed-nominal/1.0.0',unknown_mixed='EXCLUDED_BEFORE_CATALOG')


def model_identity():return 'sha256:'+hashlib.sha256(canonical_bytes(model_description())).hexdigest()


@dataclass(frozen=True)
class TrainingLimits(FlowLimits):
    python_rss_mib:int=768
    def validate(self):
        expected=TrainingLimits()
        require(all(type(getattr(self,k)) is int and getattr(self,k)==getattr(expected,k)
                    for k in self.__dataclass_fields__),'UNAPPROVED_TRAINING_BUDGET')
        return self


class BinaryMetrics:
    def __init__(self):self.n=self.tp=self.tn=self.fp=self.fn=0;self.loss=0.
    def update(self,y,probability):
        import numpy as np
        y=np.asarray(y).reshape(-1);p=np.asarray(probability).reshape(-1).astype('float64')
        require(len(y)==len(p)>0 and np.isin(y,[0,1]).all() and np.isfinite(p).all()
                and ((p>=0)&(p<=1)).all(),'INVALID_EVALUATION_BATCH')
        predicted=p>=.5;self.n+=len(y)
        self.tp+=int(((y==1)&predicted).sum());self.tn+=int(((y==0)&~predicted).sum())
        self.fp+=int(((y==0)&predicted).sum());self.fn+=int(((y==1)&~predicted).sum())
        p=np.clip(p,1e-7,1-1e-7);self.loss+=float(-(y*np.log(p)+(1-y)*np.log(1-p)).sum())
    def report(self):
        if not self.n:return dict(status='NOT_EVALUATED',samples=0)
        return dict(samples=self.n,loss=self.loss/self.n,accuracy=(self.tp+self.tn)/self.n,
            precision=self.tp/(self.tp+self.fp) if self.tp+self.fp else None,
            recall=self.tp/(self.tp+self.fn) if self.tp+self.fn else None,
            f1=2*self.tp/(2*self.tp+self.fp+self.fn) if 2*self.tp+self.fp+self.fn else None,
            false_positive_rate=self.fp/(self.fp+self.tn) if self.fp+self.tn else None,
            confusion_matrix=[[self.tn,self.fp],[self.fn,self.tp]],threshold=.5)


class StreamTrainer:
    def __init__(self,seed=42):
        require(type(seed) is int and 0<=seed<2**31-1,'TRAINING_SEED')
        for name,value in {'CUDA_VISIBLE_DEVICES':'-1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
            'TF_NUM_INTRAOP_THREADS':'1','TF_NUM_INTEROP_THREADS':'1','TF_CPP_MIN_LOG_LEVEL':'2'}.items():
            os.environ.setdefault(name,value)
        from nids_ml.models.flow_cnn import backend,build_model,configuration
        self.np,self.tf=backend();self.seed=seed
        try:
            self.tf.config.threading.set_intra_op_parallelism_threads(1)
            self.tf.config.threading.set_inter_op_parallelism_threads(1)
        except RuntimeError:pass
        self.tf.keras.utils.set_random_seed(seed)
        try:self.tf.config.experimental.enable_op_determinism()
        except AttributeError:pass
        self.config=configuration(seed=seed);self.model=build_model(self.config);self.steps=0
        self.model_id=model_identity()

    def get_weights(self):
        from nids_ml.adapters.swarm.contract import CNNTemplate
        values=tuple(self.model.get_weights())
        require(len(values)==4 and all(v.shape==shape and v.dtype==self.np.float32 and self.np.isfinite(v).all()
                for v,(_,shape) in zip(values,CNNTemplate.tensors)),'NIDS_WEIGHT_SHAPE_DTYPE')
        return values

    def set_weights(self,values):
        from nids_ml.adapters.swarm.contract import CNNTemplate
        require(isinstance(values,(list,tuple)) and len(values)==4 and all(
            isinstance(v,self.np.ndarray) and v.shape==shape and v.dtype==self.np.float32 and self.np.isfinite(v).all()
            for v,(_,shape) in zip(values,CNNTemplate.tensors)),'NIDS_WEIGHT_SHAPE_DTYPE')
        self.model.set_weights(list(values))
        self.model.optimizer=self.tf.keras.optimizers.Adam(learning_rate=.001,beta_1=.9,beta_2=.999,epsilon=1e-7)
        self.steps=0

    def predict(self,x):
        from nids_ml.models.flow_cnn import predict_flow
        return predict_flow(self.model,x,self.config)

    def train_epoch(self,reader,client=None,*,batch_size=64,epoch=0):
        from nids_ml.models.flow_cnn import train_step
        counts=reader.counts('train',client);total=sum(counts.values())
        require(set(counts)=={'0','1'} and all(type(v) is int and v>0 for v in counts.values()),'TRAIN_CLIENT_MISSING_CLASS')
        weights={int(y):total/(2*n) for y,n in counts.items()};seen=0
        initial=hashlib.sha256(b''.join(v.tobytes() for v in self.get_weights())).hexdigest()
        for batch in reader.batches('train',client,batch_size=batch_size,epoch=epoch):
            sample_weights=self.np.asarray([weights[int(y)] for y in batch['y'][:,0]],dtype='float32')
            train_step(self.model,batch['X'],batch['y'],self.config,sample_weights=sample_weights)
            self.steps+=1;seen+=len(batch['y'])
            if reader.budget is not None:reader.budget.check()
        require(seen==total,'TRAIN_STREAM_SAMPLE_COUNT')
        return dict(samples=seen,class_weights=weights,optimizer_steps=self.steps,
            initial_tensor_sha256=initial,trained_tensor_sha256=hashlib.sha256(b''.join(v.tobytes() for v in self.get_weights())).hexdigest())


def evaluate(trainer,reader,split,client=None,*,batch_size=64,predictions=None):
    total=BinaryMetrics();families={f:BinaryMetrics() for f in reader.manifest['family_coverage']}
    writer=None
    if predictions is not None:
        writer=csv.writer(predictions);writer.writerow(('sample_id','y','probability','attack_type','group_id'))
    for batch in reader.batches(split,client,batch_size=batch_size):
        probability=trainer.predict(batch['X']).reshape(-1);total.update(batch['y'],probability)
        for family in set(batch['attack_type']):
            mask=trainer.np.asarray([f==family for f in batch['attack_type']])
            families[family].update(batch['y'][mask],probability[mask])
        if writer:
            for i,p in enumerate(probability):
                writer.writerow((batch['sample_id'][i],int(batch['y'][i,0]),float(p),batch['attack_type'][i],batch['group_id'][i]))
        if reader.budget is not None:reader.budget.check()
    require(total.n==sum(reader.counts(split,client).values()),'EVALUATION_STREAM_COUNT')
    return dict(overall=total.report(),per_family={f:value.report() for f,value in families.items()})


def save_checkpoint(trainer,directory,catalog_id):
    raw=b''.join(trainer.np.asarray(v,dtype='<f4').tobytes() for v in trainer.get_weights())
    require(len(raw)==8964,'CHECKPOINT_BYTE_COUNT')
    with (Path(directory)/'weights.bin').open('xb') as stream:stream.write(raw)
    metadata=dict(format='nids-catalog-cnn-checkpoint/1.0.0',model_id=model_identity(),
        model_description=model_description(),seed=trainer.seed,catalog_id=catalog_id,
        weights_sha256=hashlib.sha256(raw).hexdigest(),weights_bytes=8964,
        optimizer_state='NOT_SAVED; reset on restore/global application',label_provenance=PROVENANCE,
        independent_ground_truth_verified=False)
    _new_json(Path(directory)/'checkpoint.json',metadata);return metadata


def restore_checkpoint(directory,*,expected_catalog_id=None):
    path=Path(directory);require(not (path/'FAILED.json').exists(),'FAILED_CHECKPOINT')
    descriptor=regular_file(path/'checkpoint.json');require(descriptor.stat().st_size<=65536,'CHECKPOINT_METADATA_CAP')
    value=strict_json(descriptor.read_bytes())
    require(value['format']=='nids-catalog-cnn-checkpoint/1.0.0' and value['model_id']==model_identity()
            and canonical_bytes(value['model_description'])==canonical_bytes(model_description()) and value['weights_bytes']==8964
            and value['label_provenance']==PROVENANCE and value['independent_ground_truth_verified'] is False,'NIDS_CHECKPOINT_MODEL')
    require(expected_catalog_id is None or value['catalog_id']==expected_catalog_id,'CHECKPOINT_CATALOG_MISMATCH')
    binary=regular_file(path/'weights.bin');require(binary.stat().st_size==8964,'CHECKPOINT_BYTE_COUNT')
    raw=binary.read_bytes();require(hashlib.sha256(raw).hexdigest()==value['weights_sha256'],'CHECKPOINT_SHA256')
    trainer=StreamTrainer(value['seed']);values=[];offset=0
    from nids_ml.adapters.swarm.contract import CNNTemplate
    for _,shape in CNNTemplate.tensors:
        length=int(trainer.np.prod(shape))*4
        values.append(trainer.np.frombuffer(raw[offset:offset+length],dtype='<f4').reshape(shape).copy());offset+=length
    trainer.set_weights(values);return trainer,value


def train_catalog(catalog,workspace,output,*,client=None,epochs=1,batch_size=64,seed=42,sources=None):
    from nids_ml.storage.partitioned import PartitionReader
    from nids_ml.models.flow_cnn import runtime_info
    require(type(epochs) is int and 1<=epochs<=10 and type(batch_size) is int and 1<=batch_size<=128,'EPOCH_OR_BATCH_BUDGET')
    target,_=new_output(output,workspace);budget=Budget(target,TrainingLimits())
    try:
        reader=PartitionReader(catalog,workspace,sources=sources,scratch=target,budget=budget)
        # All required local data must be available before the first optimizer update.
        # Thiếu data tại node thì dừng trước train; không tự đọc/pull từ node còn lại.
        for split in ('train','validation','test'):reader.preflight(split,client)
        trainer=StreamTrainer(seed);budget.check(force=True)
        before=evaluate(trainer,reader,'train',client,batch_size=batch_size);history=[]
        for epoch in range(epochs):
            update=trainer.train_epoch(reader,client,batch_size=batch_size,epoch=epoch)
            tr=evaluate(trainer,reader,'train',client,batch_size=batch_size)
            val=evaluate(trainer,reader,'validation',client,batch_size=batch_size)
            history.append(dict(epoch=epoch+1,update=update,train=tr,validation=val))
            print(__import__('json').dumps(dict(stage='epoch',epoch=epoch+1,client=client,
                train_loss=tr['overall']['loss'],validation_loss=val['overall']['loss'])),flush=True)
        checkpoint=save_checkpoint(trainer,target,reader.manifest['catalog_id'])
        restored,_=restore_checkpoint(target,expected_catalog_id=reader.manifest['catalog_id'])
        error=0.
        for batch in reader.batches('validation',client,batch_size=batch_size):
            error=max(error,float(trainer.np.max(trainer.np.abs(trainer.predict(batch['X'])-restored.predict(batch['X'])))))
        require(error<=1e-7,'CHECKPOINT_PREDICTION_RESTORE')
        with (target/'test-predictions.csv').open('x',encoding='utf-8',newline='') as stream:
            test=evaluate(trainer,reader,'test',client,batch_size=batch_size,predictions=stream)
        result=dict(status='PASS_NIDS_BATCH_TRAINING',model_id=model_identity(),model_description=model_description(),
            catalog_id=reader.manifest['catalog_id'],client=client,epochs=epochs,batch_size=batch_size,seed=seed,
            before_train=before,history=history,test=test,checkpoint=checkpoint,
            checkpoint_restore_max_prediction_error=error,runtime=runtime_info(),
            label_provenance=PROVENANCE,independent_ground_truth_verified=False,
            scope=reader.manifest['scope'],distributed_training=False,native_ARM64='REQUIRES_HOST_EVIDENCE',
            resources=dict(peak_python_rss_bytes=budget.peak_python_rss,peak_scratch_bytes=budget.peak_scratch))
        _new_json(target/'metrics.json',result);return result
    except Exception as error:
        _new_json(target/'FAILED.json',dict(status='FAILED_RETAINED',error=str(error),automatic_cleanup=False));raise
