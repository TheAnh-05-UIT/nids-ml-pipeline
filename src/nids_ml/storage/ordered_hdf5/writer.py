"""Publish ordered-HDF5 only after the full event stream passes validation.
Chỉ công bố ordered-HDF5 khi toàn stream events đã được kiểm.
"""
import hashlib

from nids_ml.flows.ordered import OrderedFlowTensor, OrderedPacketReceipt, OrderedExtractionComplete
from nids_ml.storage.hdf5 import atomic
from nids_ml.storage.hdf5.schema import backend, MAX_FILE_BYTES, ExportError
from nids_ml.storage.hdf5_layout import create_layout,record_failure
from .contracts import (FORMAT,VERSION,JSON_BYTES,CHUNK,DTYPES,ATTRS,MAX_EVENTS,
    OrderedHdf5Sample,bounded_json,validate_manifest,content_hash,require)
from .validation import Ledger

def append(data,value):
    index=len(data);data.resize(index+1,axis=0);data[index]=value

def write_events(path,events,manifest,*,compression='gzip'):
    """Chỉ nhận stream đầy đủ; exception/thiếu completion không tạo output hoàn chỉnh."""
    manifest=validate_manifest(manifest);ledger=Ledger(manifest)
    require(compression in (None,'gzip'),'ORDERED_COMPRESSION')
    target=atomic.safe_path(path,existing=False);np,h5py=backend();stage=atomic.staging_path(target)
    count=0;chain=hashlib.sha256();iterator=None
    try:
        iterator=iter(events)
        with h5py.File(stage,'x') as h:
            attrs=dict(format=FORMAT,schema_version=VERSION,status='STAGED',sample_count=0,event_count=0,content_sha256='')
            create_layout(h,np,attributes=attrs,attribute_types=ATTRS,manifest_bytes=bounded_json(manifest),
                manifest_size=JSON_BYTES,row_types=DTYPES,tensor_shape=manifest['extraction_configuration']['tensor_shape'],
                chunk_rows=CHUNK,compression=compression)
            for event in iterator:
                require(not ledger.complete,'EVENT_AFTER_COMPLETION')
                if type(event) is OrderedPacketReceipt:
                    value=event.to_dict();ledger.receipt(value)
                elif type(event) in (OrderedFlowTensor,OrderedHdf5Sample):
                    sample=event if type(event) is OrderedHdf5Sample else OrderedHdf5Sample(event)
                    tensor=sample.tensor;sample_sha=ledger.tensor(tensor,sample.label_state)
                    value=dict(kind='stored_ordered_tensor',row=ledger.samples-1,metadata=tensor.metadata,
                               label_state=sample.label_state,sample_sha256=sample_sha)
                    data=dict(X=np.frombuffer(tensor.tensor_bytes,dtype='<f4').reshape(tensor.shape),
                        metadata_json=bounded_json(tensor.metadata),sample_sha256=sample_sha.encode('ascii'),
                        y=-1,supervised_mask=False,label_state=sample.label_state.encode('ascii'))
                    for key,item in data.items():append(h[key],item)
                elif type(event) is OrderedExtractionComplete:
                    value=event.to_dict();ledger.finish(value)
                else:raise ExportError('UNSUPPORTED_ORDERED_EVENT_TYPE')
                raw=bounded_json(value);append(h['events_json'],raw);chain.update(raw+b'\n');count+=1
                require(count<=MAX_EVENTS,'ORDERED_EVENT_LIMIT')
                if count%CHUNK==0:
                    h.flush();require(stage.stat().st_size<=MAX_FILE_BYTES,'ORDERED_FILE_SIZE_LIMIT')
            require(ledger.complete,'ORDERED_COMPLETION_REQUIRED')
            # Lỗi cleanup của producer cũng phải xảy ra trước publication.
            producer=iterator;iterator=None
            if hasattr(producer,'close'):producer.close()
            h.attrs.modify('sample_count',np.array(ledger.samples,dtype='<u8'))
            h.attrs.modify('event_count',np.array(count,dtype='<u8'))
            h.attrs.modify('content_sha256',content_hash(manifest,chain.hexdigest(),ledger.samples,count).encode('ascii'))
            h.flush()
        from .reader import _validate
        _validate(stage,expected_status='STAGED')
        with h5py.File(stage,'r+') as h:h.attrs.modify('status',b'COMPLETE');h.flush()
        # Kiểm lại cả cờ COMPLETE trước khi tạo tên output công khai.
        report=_validate(stage)
        atomic.publish(stage,target)
        report.update(path=str(target),file_sha256=atomic.file_sha(target),file_signature=atomic.signature(target))
        return report
    except Exception as error:
        record_failure(stage,h5py,error)
        raise
    finally:
        if iterator is not None and hasattr(iterator,'close'):iterator.close()
