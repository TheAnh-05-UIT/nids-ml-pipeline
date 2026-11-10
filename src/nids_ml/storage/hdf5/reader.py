"""Validate capture-order HDF5 integrity and stream bounded batches.
Kiểm integrity HDF5 capture-order và đọc batches có giới hạn.
"""
from collections import Counter
import hashlib

from nids_ml.storage.hdf5_layout import read_attributes,check_datasets,check_row
from .atomic import file_sha,safe_path,signature
from .schema import (FORMAT,VERSION,MAX_SAMPLES,JSON_BYTES,LABEL_BYTES,CHUNK_ROWS,ROW_DTYPES,
                     ROOT_ATTRS,ExportError,backend,content_digest,validate_manifest)
from .validation import read_json,require,sample_from_storage,validate_sample


def _layout(handle,status):
    np,h5py=backend()
    attrs=read_attributes(handle,ROOT_ATTRS,np,require,'ROOT_ATTRIBUTES','ROOT_ATTRIBUTE_TYPE')
    require(attrs['format']==FORMAT and attrs['schema_version']==VERSION,'HDF5_SCHEMA_VERSION')
    require(attrs['status']==status,'HDF5_NOT_COMPLETE')
    count=attrs['sample_count'];require(1<=count<=MAX_SAMPLES,'HDF5_SAMPLE_COUNT_OR_EMPTY')
    require(set(handle)==set(ROW_DTYPES)|{'manifest_json'},'HDF5_DATASET_NAMES')
    check_datasets(handle,h5py,require,('EXTERNAL_OR_SOFT_LINK_BLOCKED','EXTERNAL_STORAGE_OR_GROUP_BLOCKED',
        'DATASET_ATTRIBUTES_NOT_ALLOWED','ALIASED_DATASET_BLOCKED','UNSUPPORTED_HDF5_FILTER'))
    manifest_data=handle['manifest_json']
    require(manifest_data.shape==(1,) and manifest_data.dtype==np.dtype('S'+str(JSON_BYTES))
            and manifest_data.chunks is None and manifest_data.compression is None,'MANIFEST_DATASET_LAYOUT')
    manifest=validate_manifest(read_json(bytes(manifest_data[0]),JSON_BYTES))
    n=manifest['extraction_configuration']['tensor_shape'][0]
    for name,dtype in ROW_DTYPES.items():
        data=handle[name];tail=(n,11,1) if name=='X' else ()
        check_row(data,(count,*tail),CHUNK_ROWS,dtype,np,require,
            ('HDF5_DATASET_SHAPE_OR_DTYPE','HDF5_CHUNK_LAYOUT','LOSSY_OR_UNSUPPORTED_FILTER'))
    return attrs,manifest


def _validated_rows(handle,manifest,start,end):
    # Không đọc VLEN, object array hoặc bulk metadata không giới hạn.
    arrays={name:handle[name][start:end] for name in ROW_DTYPES}
    for i in range(end-start):
        metadata=read_json(bytes(arrays['metadata_json'][i]),JSON_BYTES)
        labels=read_json(bytes(arrays['labels_json'][i]),LABEL_BYTES)
        sample=sample_from_storage(arrays['X'][i],metadata,labels)
        _,checked_meta,checked_label,expected_hash=validate_sample(sample,manifest)
        require(checked_meta==metadata and checked_label==labels,'NONCANONICAL_SAMPLE')
        require(int(arrays['y'][i])==labels['y'] and bool(arrays['supervised_mask'][i]) is labels['supervised_mask']
            and int(arrays['dataset_proposed_y'][i])==labels['dataset_proposed_y']
            and bytes(arrays['label_state'][i]).decode('ascii')==labels['state'],'LABEL_COLUMN_MISMATCH')
        require(bytes(arrays['sample_sha256'][i]).decode('ascii')==expected_hash,'SAMPLE_HASH_MISMATCH')
        yield arrays,i,metadata,labels,expected_hash


def _validate(path,*,expected_status='COMPLETE',batch_size=32):
    require(type(batch_size) is int and 1<=batch_size<=128,'BATCH_SIZE')
    path=safe_path(path,existing=True);initial=signature(path);np,h5py=backend()
    states=Counter();chain=hashlib.sha256();seen=set();supervised=0
    try:
        with h5py.File(path,'r') as handle:
            attrs,manifest=_layout(handle,expected_status)
            for start in range(0,attrs['sample_count'],batch_size):
                require(signature(path)==initial,'HDF5_CHANGED_DURING_READ')
                for _,_,meta,labels,sample_hash in _validated_rows(handle,manifest,start,min(start+batch_size,attrs['sample_count'])):
                    require(meta['flow_id'] not in seen,'DUPLICATE_FLOW_WINDOW');seen.add(meta['flow_id'])
                    chain.update(bytes.fromhex(sample_hash));states[labels['state']]+=1
                    supervised+=int(labels['supervised_mask'])
            expected=content_digest(manifest,attrs['sample_count'],chain.hexdigest())
            require(attrs['content_sha256']==expected,'CONTENT_HASH_MISMATCH')
        require(signature(path)==initial,'HDF5_CHANGED_DURING_READ')
        actual_sha=file_sha(path)
        require(signature(path)==initial,'HDF5_CHANGED_DURING_HASH')
    except (OSError,RuntimeError,KeyError,TypeError,UnicodeError) as error:
        raise ExportError('HDF5_READ_OR_CORRUPTION_ERROR:'+str(error)) from error
    return dict(status='PASS_SYNTHETIC_ONLY',format=FORMAT,schema_version=VERSION,
        manifest=manifest,sample_count=attrs['sample_count'],label_counts=dict(states),supervised_count=supervised,
        shape=[attrs['sample_count'],*manifest['extraction_configuration']['tensor_shape']],dtype='float32',
        content_sha256=expected,file_sha256=actual_sha,path=str(path),file_signature=initial,
        real_data_export='BLOCKED',native_ARM64='NOT_RUN',training=False)


def validate_hdf5(path,*,batch_size=32):
    """Validate toàn file trước khi caller sử dụng; không nhận option bỏ qua integrity."""
    return _validate(path,batch_size=batch_size)


def iter_batches(path,*,batch_size=32,supervised_only=False):
    """Batches theo append order; supervised_only chỉ dùng mask đã được kiểm toàn file."""
    require(type(supervised_only) is bool,'SUPERVISED_FILTER_BOOLEAN')
    report=validate_hdf5(path,batch_size=batch_size)
    np,h5py=backend();path=safe_path(path,existing=True)
    initial=tuple(report['file_signature']);require(signature(path)==initial,'HDF5_CHANGED_BEFORE_BATCH_READ')
    with h5py.File(path,'r') as handle:
        for start in range(0,report['sample_count'],batch_size):
            require(signature(path)==initial,'HDF5_CHANGED_DURING_BATCH_READ')
            rows=list(_validated_rows(handle,report['manifest'],start,min(start+batch_size,report['sample_count'])))
            selected=[(arrays,index,meta,labels) for arrays,index,meta,labels,_ in rows
                      if not supervised_only or labels['supervised_mask']]
            if not selected:continue
            yield dict(X=np.stack([a['X'][i] for a,i,_,_ in selected]),
                y=np.array([label['y'] for _,_,_,label in selected],dtype='i1'),
                supervised_mask=np.array([label['supervised_mask'] for _,_,_,label in selected],dtype='?'),
                label_state=[label['state'] for _,_,_,label in selected],
                dataset_proposed_y=np.array([label['dataset_proposed_y'] for _,_,_,label in selected],dtype='i1'),
                metadata=[meta for _,_,meta,_ in selected])
    require(signature(path)==initial,'HDF5_CHANGED_DURING_BATCH_READ')
