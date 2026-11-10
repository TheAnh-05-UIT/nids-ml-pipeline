"""Save and restore validated synthetic CNN weights without pickle.
Lưu và khôi phục trọng số CNN synthetic đã kiểm, không dùng pickle.
"""
import hashlib
from pathlib import Path
import stat

from nids_ml.labeling.contracts import strict_json
from nids_ml.models.flow_cnn import (ARCHITECTURE_VERSION,FEATURE_SCHEMA_VERSION,ModelContractError,
    backend,build_model,canonical,inspect_model,runtime_info,validate_configuration)

FORMAT_VERSION='synthetic-flow-cnn-checkpoint/1.0.0'
NAMES=('flow_conv/kernel','flow_conv/bias','flow_output/kernel','flow_output/bias')
SHAPES=([3,11,1,64],[64],[64,1],[1])
RUNS_ROOT=Path(__file__).resolve().parents[3]/'artifacts/runs'
TOTAL_BYTES=8964
FIELDS=('schema_version','format_version','architecture_version','feature_schema_version',
        'model_config','training_configuration','label_mapping','runtime','training_state',
        'checkpoint_semantics','weights_file','weights_sha256','weights_size_bytes','tensors','model_hash')


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def model_hash(config,weights_sha256):
    return sha(canonical(dict(format_version=FORMAT_VERSION,model_config=config,weights_sha256=weights_sha256)))


def safe_directory(path,*,new=False):
    candidate=Path(path).absolute()
    for parent in [candidate,*candidate.parents]:
        if parent.is_symlink():
            raise ModelContractError('Symlinks are not supported for checkpoint paths')
    path=candidate.resolve()
    if path==RUNS_ROOT.resolve() or not path.is_relative_to(RUNS_ROOT.resolve()):
        raise ModelContractError('Checkpoint must be under this workspace artifacts/runs')
    if new and path.exists():
        raise FileExistsError('Checkpoint output already exists; no overwrite')
    if not new and not path.is_dir():
        raise ModelContractError('Checkpoint directory not found')
    return path


def read_regular(path,limit):
    before=path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or before.st_size>limit:
        raise ModelContractError('Nonregular or oversized checkpoint member')
    with path.open('rb') as f:blob=f.read(limit+1)
    after=path.lstat()
    if len(blob)>limit or (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
        raise ModelContractError('Checkpoint changed during read')
    return blob


def save_checkpoint(model,config,directory):
    config=validate_configuration(config);inspect_model(model,config);np,tf=backend()
    path=safe_directory(directory,new=True)
    optimizer=getattr(model,'optimizer',None)
    steps=int(optimizer.iterations.numpy()) if optimizer is not None else 0
    if not 0<=steps<=32:
        raise ModelContractError('This checkpoint version is limited to tiny synthetic validation')
    tensors=[];parts=[];offset=0
    for name,shape,value in zip(NAMES,SHAPES,model.get_weights()):
        if value.dtype!=np.float32 or list(value.shape)!=shape or not np.isfinite(value).all():
            raise ModelContractError('Invalid tensor before checkpoint export')
        blob=np.asarray(value,dtype='<f4',order='C').tobytes(order='C')
        tensors.append(dict(name=name,shape=shape,dtype='<f4',offset=offset,size_bytes=len(blob),sha256=sha(blob)))
        parts.append(blob);offset+=len(blob)
    payload=b''.join(parts)
    if len(payload)!=TOTAL_BYTES or len(tensors)!=4:
        raise ModelContractError('Unexpected checkpoint payload size')
    manifest=dict(schema_version='1.0.0',format_version=FORMAT_VERSION,architecture_version=ARCHITECTURE_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,model_config=config,
        training_configuration=config['training_reference'],label_mapping={'0':'Benign','1':'DDoS'},runtime=runtime_info(),
        training_state=dict(input_kind='synthetic_fixture',optimizer_steps_observed=steps,
                            optimizer_state='NOT_SAVED',dataset_manifest=None),
        checkpoint_semantics='MODEL_WEIGHTS_ONLY; optimizer reset on load',
        weights_file='weights.bin',weights_sha256=sha(payload),weights_size_bytes=len(payload),tensors=tensors,
        model_hash=model_hash(config,sha(payload)))
    path.mkdir(exist_ok=False)
    # Chỉ ghi outputs mới. Nếu ghi lỗi, giữ phần dở để review, không cleanup tự động.
    with (path/'weights.bin').open('xb') as f:f.write(payload)
    with (path/'manifest.json').open('xb') as f:f.write(canonical(manifest)+b'\n')
    return manifest


def load_checkpoint(directory,*,expected_config=None):
    path=safe_directory(directory)
    if {p.name for p in path.iterdir()}!={'weights.bin','manifest.json'}:
        raise ModelContractError('Unexpected checkpoint directory members')
    manifest=strict_json(read_regular(path/'manifest.json',65536))
    if type(manifest) is not dict or set(manifest)!=set(FIELDS):
        raise ModelContractError('Unexpected checkpoint metadata fields')
    config=validate_configuration(manifest['model_config'])
    if expected_config is not None and canonical(config)!=canonical(validate_configuration(expected_config)):
        raise ModelContractError('Checkpoint model profile/config mismatch')
    constants=dict(schema_version='1.0.0',format_version=FORMAT_VERSION,architecture_version=ARCHITECTURE_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,training_configuration=config['training_reference'],
        label_mapping={'0':'Benign','1':'DDoS'},checkpoint_semantics='MODEL_WEIGHTS_ONLY; optimizer reset on load',
        weights_file='weights.bin',weights_size_bytes=TOTAL_BYTES)
    if any(canonical(manifest[key])!=canonical(value) for key,value in constants.items()):
        raise ModelContractError('Checkpoint version, labels, training or payload contract mismatch')
    state=manifest['training_state']
    if (type(state) is not dict or set(state)!={'input_kind','optimizer_steps_observed','optimizer_state','dataset_manifest'}
            or state['input_kind']!='synthetic_fixture' or state['optimizer_state']!='NOT_SAVED' or state['dataset_manifest'] is not None
            or type(state['optimizer_steps_observed']) is not int or not 0<=state['optimizer_steps_observed']<=32):
        raise ModelContractError('Only synthetic weights-only checkpoints supported')
    runtime=manifest['runtime'];current=runtime_info()
    if type(runtime) is not dict or set(runtime)!=set(current) or any(type(x) is not str for x in runtime.values()):
        raise ModelContractError('Invalid runtime compatibility metadata')
    if any(runtime[k]!=current[k] for k in ['tensorflow','keras','numpy','device']):
        raise ModelContractError('Unvalidated checkpoint framework versions')
    payload=read_regular(path/'weights.bin',TOTAL_BYTES)
    if len(payload)!=TOTAL_BYTES or sha(payload)!=manifest['weights_sha256'] or model_hash(config,sha(payload))!=manifest['model_hash']:
        raise ModelContractError('Checkpoint hash or size mismatch')
    np,tf=backend();weights=[];offset=0
    if type(manifest['tensors']) is not list or len(manifest['tensors'])!=4:
        raise ModelContractError('Unexpected tensor count')
    for tensor,name,shape in zip(manifest['tensors'],NAMES,SHAPES):
        size=4
        for dim in shape:size*=dim
        blob=payload[offset:offset+size]
        expected=dict(name=name,shape=shape,dtype='<f4',offset=offset,size_bytes=size,sha256=sha(blob))
        if canonical(tensor)!=canonical(expected):
            raise ModelContractError('Tensor identity, shape, dtype, offset or digest mismatch')
        value=np.frombuffer(blob,dtype='<f4').reshape(shape).copy()
        if not np.isfinite(value).all():
            raise ModelContractError('Nonfinite checkpoint tensor')
        weights.append(value);offset+=size
    model=build_model(config)
    model.set_weights(weights)
    inspect_model(model,config)
    return model,manifest


def checkpoint_schema():
    # Schema cấu trúc; semantic checks ở loader là bắt buộc, không thay bằng schema-only.
    props={key:dict(type='string') for key in FIELDS}
    for key in ['model_config','training_configuration','label_mapping','runtime','training_state']:
        props[key]=dict(type='object')
    props['weights_size_bytes']=dict(type='integer',const=TOTAL_BYTES)
    props['tensors']=dict(type='array',minItems=4,maxItems=4,items=dict(type='object'))
    for key in ['weights_sha256','model_hash']:props[key]=dict(type='string',pattern='^[a-f0-9]{64}$')
    props['format_version']=dict(type='string',const=FORMAT_VERSION)
    return dict(**{'$schema':'https://json-schema.org/draft/2020-12/schema'},
        title='Synthetic Flow CNN weights-only checkpoint; semantic validation required',
        type='object',properties=props,required=list(FIELDS),additionalProperties=False)
