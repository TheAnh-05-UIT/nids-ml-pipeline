"""One demo CLI for mapped HDF5 preparation, CPU CNN training and real holdout metrics.
Một CLI chuẩn bị HDF5 có nhãn mapping, train CNN CPU và đánh giá holdout thực tế.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3

from nids_ml.data.demo import build_dataset,read_dataset,require,PROVENANCE
from nids_ml.labeling.contracts import strict_json
from nids_ml.ordering.view import _fresh_directory,_new_json
from nids_ml.pcap.decoder import regular_file


def metrics(y,probability):
    import numpy as np
    actual=np.asarray(y).reshape(-1);probability=np.asarray(probability).reshape(-1)
    require(len(actual)==len(probability)>0 and np.isfinite(probability).all(),'INVALID_EVALUATION_INPUT')
    require(np.isin(actual,[0,1]).all() and ((probability>=0)&(probability<=1)).all(),'INVALID_LABEL_OR_PROBABILITY')
    predicted=probability>=0.5
    tp=int(((actual==1)&predicted).sum());tn=int(((actual==0)&~predicted).sum())
    fp=int(((actual==0)&predicted).sum());fn=int(((actual==1)&~predicted).sum())
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
    p=np.clip(probability.astype('float64'),1e-7,1-1e-7)
    return dict(samples=len(actual),loss=float(-(actual*np.log(p)+(1-actual)*np.log(1-p)).mean()),
        accuracy=(tp+tn)/len(actual),precision=precision,recall=recall,
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
        false_positive_rate=fp/(fp+tn) if fp+tn else 0.,confusion_matrix=[[tn,fp],[fn,tp]],threshold=0.5)


def restore_checkpoint(directory):
    from nids_ml.models.flow_cnn import backend,build_model,validate_configuration
    from nids_ml.models.flow_checkpoint import SHAPES,TOTAL_BYTES
    path=Path(directory)
    require(not (path/'FAILED.json').exists(),'FAILED_CHECKPOINT')
    descriptor=regular_file(path/'checkpoint.json')
    require(descriptor.stat().st_size<=65536,'CHECKPOINT_MANIFEST_SIZE')
    manifest=strict_json(descriptor.read_bytes())
    require(manifest['format']=='mapped-demo-cnn-weights/1.0.0','CHECKPOINT_FORMAT')
    binary=regular_file(path/'weights.bin')
    require(binary.stat().st_size==TOTAL_BYTES,'CHECKPOINT_BYTE_COUNT')
    raw=binary.read_bytes()
    require(len(raw)==TOTAL_BYTES and hashlib.sha256(raw).hexdigest()==manifest['weights_sha256'],'CHECKPOINT_HASH')
    np,tf=backend();weights=[];offset=0
    for shape in SHAPES:
        size=int(np.prod(shape))*4;value=np.frombuffer(raw[offset:offset+size],dtype='<f4').reshape(shape)
        require(np.isfinite(value).all(),'NONFINITE_CHECKPOINT');weights.append(value.copy());offset+=size
    config=validate_configuration(manifest['model_config']);model=build_model(config,compile_model=False)
    model.set_weights(weights);return model,config


def train(dataset,output,*,epochs=3,batch_size=64,seed=42,client=None):
    require(type(epochs) is int and 1<=epochs<=10,'EPOCHS_1_TO_10')
    require(type(batch_size) is int and 1<=batch_size<=128,'BATCH_1_TO_128')
    require(type(seed) is int and 0<=seed<2**31,'INVALID_SEED')
    require(client is None or type(client) is int and client in (0,1),'INVALID_CLIENT')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2');os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
    os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('TF_NUM_INTRAOP_THREADS','1');os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
    from nids_ml.models.flow_cnn import backend,build_model,configuration,predict_flow,train_step,runtime_info
    np,tf=backend()
    try:tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:pass
    tf.keras.utils.set_random_seed(seed)
    try:tf.config.experimental.enable_op_determinism()
    except AttributeError:pass
    manifest,x,y,splits,clients,metadata=read_dataset(dataset)
    target=_fresh_directory(output)
    runs=Path(__file__).resolve().parents[3]/'artifacts/runs'
    require(target.resolve().is_relative_to(runs.resolve()),'TRAIN_OUTPUT_UNDER_ARTIFACTS_RUNS_REQUIRED')
    training=np.flatnonzero((splits==0)&((clients==client) if client is not None else True))
    require(set(y[training])=={0,1},'TRAIN_CLIENT_MISSING_CLASS')
    config=configuration(seed=seed);model=build_model(config);target.mkdir();history=[]
    class_weights={label:len(training)/(2*int((y[training]==label).sum())) for label in (0,1)}

    def predict(indices):
        result=[]
        for start in range(0,len(indices),128):result.extend(predict_flow(model,x[indices[start:start+128]],config).reshape(-1))
        return np.asarray(result,dtype='float32')

    try:
        for epoch in range(epochs):
            indices=training.copy();np.random.default_rng(seed+epoch).shuffle(indices)
            for start in range(0,len(indices),batch_size):
                batch=indices[start:start+batch_size]
                weights=np.asarray([class_weights[int(value)] for value in y[batch]],dtype='float32')
                train_step(model,x[batch],y[batch].astype('float32').reshape(-1,1),config,sample_weights=weights)
            tr=metrics(y[training],predict(training));validation=np.flatnonzero(splits==1)
            val=metrics(y[validation],predict(validation));history.append(dict(epoch=epoch+1,train=tr,validation=val))
            print(json.dumps(dict(stage='epoch',epoch=epoch+1,train_loss=tr['loss'],validation_loss=val['loss'])),flush=True)
        test=np.flatnonzero(splits==2);test_probability=predict(test);test_metrics=metrics(y[test],test_probability)
        weights=b''.join(np.asarray(value,dtype='<f4').tobytes() for value in model.get_weights())
        with (target/'weights.bin').open('xb') as stream:stream.write(weights)
        _new_json(target/'checkpoint.json',dict(format='mapped-demo-cnn-weights/1.0.0',model_config=config,
            weights_sha256=hashlib.sha256(weights).hexdigest(),weights_bytes=len(weights),
            optimizer_state='NOT_SAVED; reset on restore',dataset_sha256=manifest['hdf5']['sha256'],
            label_provenance=PROVENANCE,independent_ground_truth_verified=False))
        restored,_=restore_checkpoint(target)
        restored_probability=[]
        for start in range(0,len(test),128):restored_probability.extend(predict_flow(restored,x[test[start:start+128]],config).reshape(-1))
        max_error=float(np.max(np.abs(test_probability-np.asarray(restored_probability))))
        require(max_error<=1e-7,'CHECKPOINT_RESTORE_PREDICTION_MISMATCH')
        with (target/'test-predictions.csv').open('x',encoding='utf-8') as stream:
            stream.write('sample_index,y,probability,group\n')
            for index,probability in zip(test,test_probability):stream.write(f'{index},{y[index]},{float(probability)},{metadata[index]["group"]}\n')
        plot_error=None
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            figure,axes=plt.subplots(1,2,figsize=(9,3))
            axes[0].plot(range(1,epochs+1),[row['train']['loss'] for row in history],label='Train')
            axes[0].plot(range(1,epochs+1),[row['validation']['loss'] for row in history],label='Validation')
            axes[0].set(xlabel='Epoch',ylabel='Binary cross-entropy');axes[0].legend()
            image=axes[1].imshow(test_metrics['confusion_matrix'],cmap='Blues');axes[1].set(xlabel='Predicted',ylabel='Actual',xticks=[0,1],yticks=[0,1])
            for a in range(2):
                for b in range(2):axes[1].text(b,a,str(test_metrics['confusion_matrix'][a][b]),ha='center',va='center')
            figure.tight_layout();figure.savefig(target/'training-results.png',dpi=150);plt.close(figure)
        except ImportError:plot_error='matplotlib not installed; metrics/history remain available, no auto-install'
        report=dict(status='PASS_MAPPED_DEMO_BASELINE',runtime=runtime_info(),epochs=epochs,batch_size=batch_size,
            seed=seed,client=client,training_samples=len(training),class_weights=class_weights,history=history,
            test=test_metrics,checkpoint_restore_max_prediction_error=max_error,plot_note=plot_error,
            dataset_sha256=manifest['hdf5']['sha256'],label_provenance=PROVENANCE,
            independent_ground_truth_verified=False,evaluation_scope=manifest['scope'],limitation=manifest['limitation'],
            swarm_training=False,native_ARM64='REQUIRES_HOST_EVIDENCE')
        _new_json(target/'metrics.json',report);return report
    except Exception as error:
        _new_json(target/'FAILED.json',dict(status='FAILED_RETAINED',error=str(error),automatic_cleanup=False));raise


def main(argv=None):
    parser=argparse.ArgumentParser(description='Normal/HTTP demo: mapped labels -> n20 HDF5 -> CPU CNN baseline. No Docker/SSH.')
    commands=parser.add_subparsers(dest='command',required=True)
    prepare=commands.add_parser('prepare');prepare.add_argument('--normal-mapping',required=True);prepare.add_argument('--attack-mapping',required=True)
    prepare.add_argument('--tshark',required=True);prepare.add_argument('--tshark-sha256',required=True)
    prepare.add_argument('--output',required=True);prepare.add_argument('--max-per-class',type=int,default=1000)
    prepare.add_argument('--workspace',default=str(Path(__file__).resolve().parents[3]))
    training=commands.add_parser('train');training.add_argument('--dataset',required=True);training.add_argument('--output',required=True)
    training.add_argument('--epochs',type=int,default=3);training.add_argument('--batch-size',type=int,default=64)
    training.add_argument('--client',type=int,choices=(0,1))
    inspect=commands.add_parser('inspect');inspect.add_argument('--dataset',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='prepare':
            report=build_dataset(args.normal_mapping,args.attack_mapping,args.workspace,args.output,tshark=args.tshark,
                tshark_sha256=args.tshark_sha256,max_per_class=args.max_per_class,
                progress=lambda stage,records:print(json.dumps(dict(stage=stage,records=records)),flush=True))
        elif args.command=='inspect':report=read_dataset(args.dataset)[0]
        else:report=train(args.dataset,args.output,epochs=args.epochs,batch_size=args.batch_size,client=args.client)
        print(json.dumps(report,indent=2,allow_nan=False));return 0
    except (ValueError,TypeError,KeyError,OSError,RuntimeError,sqlite3.Error) as error:
        print(json.dumps(dict(status='FAIL',error=str(error),automatic_cleanup=False)));return 2


if __name__=='__main__':raise SystemExit(main())
