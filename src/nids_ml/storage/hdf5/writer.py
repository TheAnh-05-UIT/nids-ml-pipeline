"""Write, read back and atomically publish synthetic capture-order HDF5.
Ghi, đọc kiểm và công bố nguyên tử HDF5 capture-order synthetic.
"""
import hashlib

from nids_ml.labeling.contracts import canonical_bytes
from nids_ml.storage.hdf5_layout import create_layout,record_failure
from . import atomic
from .reader import _validate
from .schema import (FORMAT,VERSION,MAX_SAMPLES,MAX_FILE_BYTES,JSON_BYTES,ROW_DTYPES,ROOT_ATTRS,
                     CHUNK_ROWS,ExportError,backend,content_digest,validate_manifest)
from .validation import require,validate_sample


def _append(handle,index,array,meta,labels,sample_hash):
    values=dict(X=array,y=labels['y'],label_state=labels['state'].encode('ascii'),
        supervised_mask=labels['supervised_mask'],dataset_proposed_y=labels['dataset_proposed_y'],
        metadata_json=canonical_bytes(meta),labels_json=canonical_bytes(labels),sample_sha256=sample_hash.encode('ascii'))
    for name,value in values.items():
        data=handle[name];data.resize(index+1,axis=0);data[index]=value


def write_flows(path,samples,manifest,*,compression='gzip',max_samples=MAX_SAMPLES):
    """Không sort iterable; verified chỉ nhận decision synthetic có evidence binding."""
    manifest=validate_manifest(manifest)
    require(type(max_samples) is int and 1<=max_samples<=MAX_SAMPLES,'MAX_SAMPLES')
    require(compression in (None,'gzip'),'COMPRESSION_NOT_SUPPORTED')
    target=atomic.safe_path(path,existing=False);np,h5py=backend()
    stage=atomic.staging_path(target);seen=set();chain=hashlib.sha256();count=0
    try:
        with h5py.File(stage,'x') as handle:
            attrs=dict(format=FORMAT,schema_version=VERSION,status='STAGED',sample_count=0,content_sha256='')
            create_layout(handle,np,attributes=attrs,attribute_types=ROOT_ATTRS,manifest_bytes=canonical_bytes(manifest),
                manifest_size=JSON_BYTES,row_types=ROW_DTYPES,tensor_shape=manifest['extraction_configuration']['tensor_shape'],
                chunk_rows=CHUNK_ROWS,compression=compression)
            for sample in samples:
                if count>=max_samples:raise ExportError('SAMPLE_LIMIT')
                array,meta,labels,sample_hash=validate_sample(sample,manifest)
                require(meta['flow_id'] not in seen,'DUPLICATE_FLOW_WINDOW');seen.add(meta['flow_id'])
                _append(handle,count,array,meta,labels,sample_hash)
                chain.update(bytes.fromhex(sample_hash));count+=1
                if count%CHUNK_ROWS==0:
                    handle.flush()
                    require(stage.stat().st_size<=MAX_FILE_BYTES,'HDF5_FILE_SIZE_LIMIT')
            require(count>0,'REJECT_EMPTY')
            handle.attrs.modify('sample_count',np.array(count,dtype='<u8'))
            handle.attrs.modify('content_sha256',content_digest(manifest,count,chain.hexdigest()).encode('ascii'))
            handle.flush()
        report=_validate(stage,expected_status='STAGED')
        with h5py.File(stage,'r+') as handle:
            handle.attrs.modify('status',b'COMPLETE');handle.flush()
        atomic.publish(stage,target)
        report.update(path=str(target),file_sha256=atomic.file_sha(target),file_signature=atomic.signature(target))
        return report
    except Exception as error:
        record_failure(stage,h5py,error,output_claimed_complete=False,real_data_export=False)
        raise
