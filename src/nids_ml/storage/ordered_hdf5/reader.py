"""Replay ordered-HDF5 evidence and validate bounded batch reads.
Replay evidence ordered-HDF5 và kiểm đọc batch có giới hạn.
"""
from collections import Counter
import hashlib

from nids_ml.flows.ordered import OrderedFlowTensor
from nids_ml.labeling.contracts import canonical_bytes
from nids_ml.storage.hdf5 import atomic
from nids_ml.storage.hdf5.schema import backend,ExportError
from nids_ml.storage.hdf5_layout import read_attributes,check_datasets,check_row
from .contracts import (FORMAT,VERSION,JSON_BYTES,DTYPES,ATTRS,CHUNK,MAX_SAMPLES,MAX_EVENTS,
    require,read_json,validate_manifest,content_hash,sample_hash)
from .validation import Ledger

def layout(h,status,np,h5py):
    attrs=read_attributes(h,ATTRS,np,require,'ORDERED_ROOT_ATTRIBUTES','ORDERED_ATTRIBUTE_DTYPE')
    require(attrs['format']==FORMAT and attrs['schema_version']==VERSION,'ORDERED_FORMAT_OR_VERSION')
    require(attrs['status']==status,'ORDERED_HDF5_NOT_COMPLETE')
    require(1<=attrs['sample_count']<=MAX_SAMPLES and
            1<=attrs['event_count']<=MAX_EVENTS,'ORDERED_ROW_COUNT_LIMIT')
    require(set(h)==set(DTYPES)|{'manifest_json'},'ORDERED_DATASET_NAMES')
    check_datasets(h,h5py,require,('ORDERED_EXTERNAL_OR_SOFT_LINK','ORDERED_UNSAFE_DATASET',
        'ORDERED_UNSAFE_DATASET','ORDERED_ALIASED_DATASET','ORDERED_UNSUPPORTED_FILTER'))
    data=h['manifest_json']
    require(data.shape==(1,) and data.dtype==np.dtype('S16384') and data.chunks is None
            and data.compression is None,'ORDERED_MANIFEST_LAYOUT')
    manifest=validate_manifest(read_json(bytes(data[0])))
    shape=tuple(manifest['extraction_configuration']['tensor_shape'])
    for name,dtype in DTYPES.items():
        data=h[name];tail=shape if name=='X' else ()
        count=attrs['event_count'] if name=='events_json' else attrs['sample_count']
        check_row(data,(count,*tail),CHUNK,dtype,np,require,
            ('ORDERED_DATASET_DTYPE_OR_SHAPE','ORDERED_CHUNK_LAYOUT','ORDERED_FILTER_LAYOUT'))
    return attrs,manifest

def row(h,index):
    metadata_raw=bytes(h['metadata_json'][index]);metadata=read_json(metadata_raw)
    array=h['X'][index];tensor=OrderedFlowTensor(tuple(array.shape),array.tobytes(),metadata_raw)
    state=bytes(h['label_state'][index]).decode('ascii')
    require(int(h['y'][index])==-1 and not bool(h['supervised_mask'][index]),'ORDERED_UNSAFE_SUPERVISED_LABEL')
    digest=sample_hash(tensor,state)
    require(bytes(h['sample_sha256'][index]).decode('ascii')==digest,'ORDERED_SAMPLE_HASH_MISMATCH')
    return tensor,state,digest,array

def _validate(path,*,expected_status='COMPLETE'):
    path=atomic.safe_path(path,existing=True);initial=atomic.signature(path);np,h5py=backend()
    states=Counter();chain=hashlib.sha256()
    try:
        with h5py.File(path,'r') as h:
            attrs,manifest=layout(h,expected_status,np,h5py);ledger=Ledger(manifest)
            # Mỗi lượt tối đa một fixed-size event và một tensor; không bulk-load file.
            for index in range(attrs['event_count']):
                require(atomic.signature(path)==initial,'ORDERED_FILE_CHANGED_DURING_READ')
                require(not ledger.complete,'EVENT_AFTER_COMPLETION')
                raw=bytes(h['events_json'][index]);value=read_json(raw);chain.update(raw+b'\n')
                require(type(value) is dict,'ORDERED_EVENT_OBJECT')
                if value.get('kind')=='ordered_packet_receipt':ledger.receipt(value)
                elif value.get('kind')=='stored_ordered_tensor':
                    require(set(value)=={'kind','row','metadata','label_state','sample_sha256'} and
                            type(value['row']) is int and value['row']==ledger.samples<attrs['sample_count'],
                            'ORDERED_TENSOR_EVENT_ROW')
                    tensor,state,digest,_=row(h,ledger.samples)
                    require(canonical_bytes(value['metadata'])==tensor.metadata_json and value['label_state']==state
                            and value['sample_sha256']==digest,'ORDERED_EVENT_ROW_MISMATCH')
                    require(ledger.tensor(tensor,state)==digest,'ORDERED_SAMPLE_HASH_MISMATCH');states[state]+=1
                elif value.get('kind')=='ordered_stream_complete':ledger.finish(value)
                else:raise ExportError('ORDERED_EVENT_KIND')
            require(ledger.complete and ledger.samples==attrs['sample_count'],'ORDERED_COMPLETION_REQUIRED')
            expected=content_hash(manifest,chain.hexdigest(),ledger.samples,attrs['event_count'])
            require(attrs['content_sha256']==expected,'ORDERED_CONTENT_HASH_MISMATCH')
        require(atomic.signature(path)==initial,'ORDERED_FILE_CHANGED_DURING_READ')
        actual=atomic.file_sha(path)
        require(atomic.signature(path)==initial,'ORDERED_FILE_CHANGED_DURING_HASH')
    except (OSError,RuntimeError,KeyError,TypeError,UnicodeError) as error:
        raise ExportError('ORDERED_HDF5_READ_OR_CORRUPTION:'+str(error)) from error
    return dict(status='PASS_ORDERED_SYNTHETIC_ONLY',format=FORMAT,schema_version=VERSION,
        manifest=manifest,sample_count=ledger.samples,event_count=attrs['event_count'],receipt_count=ledger.count,
        shape=[ledger.samples,*manifest['extraction_configuration']['tensor_shape']],dtype='float32',
        label_counts=dict(states),supervised_count=0,content_sha256=expected,file_sha256=actual,
        file_signature=initial,path=str(path),real_data_export='BLOCKED',training=False,native_ARM64='NOT_VERIFIED')

def validate_hdf5(path):return _validate(path)

def iter_batches(path,*,batch_size=16,supervised_only=False):
    require(type(batch_size) is int and 1<=batch_size<=128,'ORDERED_BATCH_SIZE')
    require(type(supervised_only) is bool,'ORDERED_SUPERVISED_FLAG')
    report=validate_hdf5(path);path=atomic.safe_path(path,existing=True);np,h5py=backend()
    initial=tuple(report['file_signature'])
    require(atomic.signature(path)==initial and atomic.file_sha(path)==report['file_sha256'],
            'ORDERED_FILE_CHANGED_BEFORE_BATCH_READ')
    with h5py.File(path,'r') as h:
        for start in range(0,report['sample_count'],batch_size):
            require(atomic.signature(path)==initial,'ORDERED_FILE_CHANGED_DURING_BATCH_READ')
            rows=[row(h,i) for i in range(start,min(start+batch_size,report['sample_count']))]
            if supervised_only:continue  # UNKNOWN/MIXED không trở thành nhãn 0/1.
            yield dict(X=np.stack([r[3] for r in rows]),y=np.full(len(rows),-1,dtype='i1'),
                supervised_mask=np.zeros(len(rows),dtype='?'),label_state=[r[1] for r in rows],
                metadata=[r[0].metadata for r in rows])
    require(atomic.signature(path)==initial and atomic.file_sha(path)==report['file_sha256'],
            'ORDERED_FILE_CHANGED_DURING_BATCH_READ')
