"""Define the Flow CNN and bounded synthetic forward/learning utilities.
Định nghĩa Flow CNN và tiện ích forward/learning synthetic nhỏ.
"""
from functools import lru_cache
import importlib.metadata
import json
import platform
import sys

ARCHITECTURE_VERSION = 'flow-cnn/1.0.0'
FEATURE_SCHEMA_VERSION = 'lucid-flow-11/1.0.0-candidate'
FEATURE_ORDER = ('Time','Packet Length','Higher Protocol','IP Flags','Protocols',
                 'TCP Length','TCP Ack','TCP Flags','TCP Window Size','UDP Length','ICMP Type')
PROFILES = {'flow-cnn-n20-t10':20,'paper-table3-n10':10}


class ModelContractError(ValueError):
    """Từ chối cấu hình, input hoặc artifact không đúng contract."""


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()


def training_reference():
    return dict(schema_version='1.0.0',optimizer='Adam',learning_rate=0.001,
        batch_size=128,epochs=10,loss='binary_crossentropy',label_mapping={'0':'Benign','1':'DDoS'},
        adam_beta_1=0.9,adam_beta_2=0.999,adam_epsilon=1e-7,adam_amsgrad=False,
        reference_values_status='VERIFIED_FROM_PAPER_TABLE12; Flow-only application PROPOSED_IMPLEMENTATION',
        adam_details_status='PROPOSED_IMPLEMENTATION; not fully specified in paper',
        distributed_optimizer_state='UNKNOWN; no OpenSwarm integration')


def configuration(profile='flow-cnn-n20-t10',seed=42):
    if profile not in PROFILES or type(seed) is not int or not 0<=seed<2**31-1:
        raise ModelContractError('Unsupported model profile or seed')
    return dict(schema_version='1.0.0',architecture_version=ARCHITECTURE_VERSION,
        profile=profile,input_shape=[PROFILES[profile],11,1],input_dtype='float32',layout='channels_last',
        feature_schema_version=FEATURE_SCHEMA_VERSION,feature_order=list(FEATURE_ORDER),window_seconds=10,
        convolution=dict(filters=64,kernel_size=[3,11],strides=[1,1],padding='valid',use_bias=True),
        activation='relu',pooling='GlobalMaxPooling2D',flatten=True,
        output=dict(units=1,activation='sigmoid'),seed=seed,
        initializer='GlorotUniform',bias_initializer='zeros',
        training_reference=training_reference(),
        provenance=dict(packet_count='VERIFIED_FROM_PAPER experiment n20; n10 Table3 comparison',
            layer_counts='VERIFIED_FROM_PAPER_TABLE3',
            kernel_activation_layout='VERIFIED_FROM_SOURCE_REFERENCE; PROPOSED_IMPLEMENTATION for this paper',
            seed_initializer_optimizer_details='PROPOSED_IMPLEMENTATION',
            normalization='UNKNOWN_FROM_PAPER; no transform implemented',
            exact_reproduction='NOT CLAIMED'))


def validate_configuration(value):
    if type(value) is not dict or type(value.get('profile')) is not str:
        raise ModelContractError('Expected a versioned model configuration')
    expected=configuration(value['profile'],value.get('seed'))
    # So canonical bytes để không coi bool là số nguyên hoặc bỏ qua unknown keys.
    if canonical(value)!=canonical(expected):
        raise ModelContractError('Model configuration differs from the fixed architecture/feature contract')
    return expected


@lru_cache(maxsize=1)
def backend():
    if sys.version_info[:2]!=(3,11):
        raise RuntimeError('Flow CNN validated runtime requires Python 3.11 and TensorFlow CPU 2.15.1')
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError('TensorFlow/NumPy unavailable; no automatic dependency installation') from error
    if (tf.__version__,np.__version__,importlib.metadata.version('keras'))!=('2.15.1','1.26.4','2.15.0'):
        raise RuntimeError('Unvalidated framework versions; expected TF 2.15.1 / NumPy 1.26.4 / Keras 2.15.0')
    return np,tf


def runtime_info():
    np,tf=backend()
    return dict(python=platform.python_version(),system=platform.system(),machine=platform.machine(),
                tensorflow=tf.__version__,keras=importlib.metadata.version('keras'),numpy=np.__version__,
                device='CPU',cross_platform_bitwise_reproducibility='NOT VERIFIED')


def build_model(config='flow-cnn-n20-t10',*,compile_model=True):
    if type(compile_model) is not bool:
        raise ModelContractError('compile_model must be boolean')
    config=configuration(config) if type(config) is str else validate_configuration(config)
    np,tf=backend()
    seed=config['seed']
    with tf.device('/CPU:0'):
        model=tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=tuple(config['input_shape']),dtype='float32',name='flow_input'),
            tf.keras.layers.Conv2D(64,(3,11),strides=(1,1),padding='valid',data_format='channels_last',
                use_bias=True,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                bias_initializer='zeros',dtype='float32',name='flow_conv'),
            tf.keras.layers.Activation('relu',dtype='float32',name='flow_relu'),
            tf.keras.layers.GlobalMaxPooling2D(data_format='channels_last',name='flow_pool'),
            tf.keras.layers.Flatten(data_format='channels_last',name='flow_flatten'),
            tf.keras.layers.Dense(1,activation='sigmoid',use_bias=True,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed+1),
                bias_initializer='zeros',dtype='float32',name='flow_output'),
        ],name=config['profile'].replace('-','_'))
        if compile_model:
            ref=config['training_reference']
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=ref['learning_rate'],
                beta_1=ref['adam_beta_1'],beta_2=ref['adam_beta_2'],epsilon=ref['adam_epsilon'],amsgrad=False),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False))
    inspect_model(model,config)
    return model


def inspect_model(model,config):
    config=validate_configuration(config)
    classes=['Conv2D','Activation','GlobalMaxPooling2D','Flatten','Dense']
    shapes=[[None,config['input_shape'][0]-2,1,64]]*2+[[None,64],[None,64],[None,1]]
    params=[2176,0,0,0,65]
    if ([type(layer).__name__ for layer in model.layers]!=classes or list(model.input_shape)!=[None,*config['input_shape']]
            or list(model.output_shape)!=[None,1] or model.count_params()!=2241):
        raise ModelContractError('Unexpected model architecture')
    actual_shapes=[list(layer.output_shape) for layer in model.layers]
    if actual_shapes!=shapes or [layer.count_params() for layer in model.layers]!=params:
        raise ModelContractError('Layer shape or parameter count mismatch')
    weight_shapes=[list(w.shape) for w in model.trainable_weights]
    if weight_shapes!=[[3,11,1,64],[64],[64,1],[1]] or model.non_trainable_weights:
        raise ModelContractError('Unexpected trainable tensor registry')
    # Config layers được so sánh với contract cố định, không load tùy ý custom layers.
    conv=model.layers[0].get_config();dense=model.layers[-1].get_config()
    if (tuple(conv['kernel_size'])!=(3,11) or tuple(conv['strides'])!=(1,1) or conv['padding']!='valid'
            or conv['activation']!='linear' or not conv['use_bias'] or conv['data_format']!='channels_last'
            or model.layers[1].get_config()['activation']!='relu' or dense['activation']!='sigmoid'
            or not dense['use_bias']):
        raise ModelContractError('Layer semantics mismatch')
    return dict(profile=config['profile'],input_shape=list(model.input_shape),output_shape=list(model.output_shape),
                layers=[dict(name=l.name,class_name=type(l).__name__,output_shape=s,parameters=p)
                        for l,s,p in zip(model.layers,actual_shapes,params)],
                trainable_parameters=2241,non_trainable_parameters=0,weight_shapes=weight_shapes)


def validate_inputs(values,config):
    config=validate_configuration(config);np,tf=backend()
    x=np.asarray(values)
    if x.dtype!=np.dtype('float32') or x.ndim!=4 or tuple(x.shape[1:])!=tuple(config['input_shape']):
        raise ModelContractError('Expected float32 batch with fixed packet/feature/channel axes')
    if not 1<=x.shape[0]<=128 or not np.isfinite(x).all():
        raise ModelContractError('Invalid batch size or nonfinite input')
    return x


def validate_labels(values,batch):
    np,tf=backend();y=np.asarray(values)
    if y.dtype!=np.dtype('float32') or y.shape!=(batch,1) or not np.isfinite(y).all() or not np.isin(y,[0.,1.]).all():
        raise ModelContractError('Labels must be finite float32 (B,1) with values exactly 0/1; UNKNOWN/MIXED forbidden')
    return y


def predict_flow(model,values,config):
    np,tf=backend();x=validate_inputs(values,config);inspect_model(model,config)
    with tf.device('/CPU:0'):
        prediction=model(tf.convert_to_tensor(x),training=False).numpy()
    if prediction.shape!=(len(x),1) or prediction.dtype!=np.float32 or not np.isfinite(prediction).all() or not ((prediction>=0)&(prediction<=1)).all():
        raise ModelContractError('Invalid model prediction')
    return prediction


def train_step(model,values,labels,config,*,sample_weights=None):
    np,tf=backend();x=validate_inputs(values,config);y=validate_labels(labels,len(x));inspect_model(model,config)
    if not hasattr(model,'optimizer') or model.optimizer is None:
        raise ModelContractError('Compile the model before a synthetic training step')
    with tf.device('/CPU:0'):
        with tf.GradientTape() as tape:
            predictions=model(tf.convert_to_tensor(x),training=True)
            if sample_weights is None:
                loss=tf.keras.losses.BinaryCrossentropy()(tf.convert_to_tensor(y),predictions)
            else:
                weights=np.asarray(sample_weights)
                if weights.dtype!=np.float32 or weights.shape!=(len(x),) or not np.isfinite(weights).all() or not (weights>0).all():
                    raise ModelContractError('Expected positive finite float32 sample weights (B,)')
                per_sample=tf.keras.losses.BinaryCrossentropy(reduction='none')(tf.convert_to_tensor(y),predictions)
                loss=tf.reduce_mean(per_sample*tf.convert_to_tensor(weights))
        gradients=tape.gradient(loss,model.trainable_weights)
        if not bool(tf.math.is_finite(loss).numpy()) or any(g is None or not bool(tf.reduce_all(tf.math.is_finite(g)).numpy()) for g in gradients):
            raise ModelContractError('Missing or nonfinite gradient/loss')
        norm=float(tf.linalg.global_norm(gradients).numpy())
        if not np.isfinite(norm):
            raise ModelContractError('Nonfinite gradient norm')
        model.optimizer.apply_gradients(zip(gradients,model.trainable_weights))
    if any(not np.isfinite(w.numpy()).all() for w in model.trainable_weights):
        raise ModelContractError('Nonfinite model weights after update')
    return dict(loss_before_update=float(loss.numpy()),gradient_norm=norm,gradient_shapes=[list(g.shape) for g in gradients])


def synthetic_batch(config,samples=16,seed=17):
    config=validate_configuration(config);np,tf=backend()
    if type(samples) is not int or not 2<=samples<=32 or type(seed) is not int or not 0<=seed<2**31:
        raise ModelContractError('Synthetic fixture budget/seed invalid')
    rng=np.random.default_rng(seed)
    labels=(np.arange(samples)%2).astype('float32').reshape(-1,1)
    noise=rng.uniform(0.,0.1,(samples,*config['input_shape'])).astype('float32')
    values=noise+labels.reshape(-1,1,1,1)*np.float32(0.9)
    return values,labels


def learning_sanity(config,steps=32):
    config=validate_configuration(config);np,tf=backend()
    if type(steps) is not int or not 1<=steps<=32:
        raise ModelContractError('Tiny synthetic test is limited to 32 optimizer steps')
    model=build_model(config);x,y=synthetic_batch(config)
    loss_fn=tf.keras.losses.BinaryCrossentropy()
    before=float(loss_fn(y,predict_flow(model,x,config)).numpy())
    last=None
    for _ in range(steps):last=train_step(model,x,y,config)
    after=float(loss_fn(y,predict_flow(model,x,config)).numpy())
    return model,dict(input_kind='synthetic_fixture',samples=len(x),steps=steps,
        batch_size=len(x),loss_before=before,loss_after=after,last_step=last,
        passed=bool(np.isfinite([before,after]).all() and after<before-0.01),
        dataset_result=False,seed=17)
