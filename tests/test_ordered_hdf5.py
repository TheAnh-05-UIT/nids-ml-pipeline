"""Test ordered-HDF5 roundtrips, complete-stream checks and label exclusion.
Test ordered-HDF5: roundtrip, đủ stream và loại nhãn không hợp lệ.
"""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from flow_extraction_fixtures import packet
from test_ordered_flow import source
from nids_ml.cli.ordered_hdf5 import main
from nids_ml.flows.ordered import OrderedFlowExtractor,OrderedFlowTensor,OrderedPacketReceipt,OrderedExtractionComplete
from nids_ml.flows.ordered_contracts import configuration
from nids_ml.labeling.contracts import canonical_bytes
from nids_ml.ordering.lookup import SyntheticPacketLookup
from nids_ml.ordering.view import build_processing_view,iter_processing_view
from nids_ml.storage.hdf5 import validate_hdf5 as old_validate
from nids_ml.storage.hdf5.schema import ExportError,backend
from nids_ml.storage.ordered_hdf5 import OrderedHdf5Sample,make_manifest,write_events,validate_hdf5,iter_batches
from nids_ml.storage.ordered_hdf5.contracts import MANIFEST_SCHEMA,DTYPES,validate_manifest

class OrderedHdf5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.np,cls.h5py=backend()

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='ordered-hdf5-synthetic-',dir=ROOT/'artifacts/runs')
        self.home=Path(self.tmp.name);self.path=self.home/'flow.hdf5';self.serial=0
        self.packets=[packet(ordinal=1,timestamp_ns=10000000001),packet(ordinal=2,timestamp_ns=0),
            packet(ordinal=3,timestamp_ns=10000000000),packet(ordinal=4,timestamp_ns=2)]
        self.events,self.manifest=self.prepare(self.packets)

    def tearDown(self):self.tmp.cleanup()

    def prepare(self,packets,profile='flow-cnn-n20-t10'):
        self.serial+=1;where=self.home/('view-'+str(self.serial));src=source(packets)
        view=build_processing_view(packets,src,where);lookup=SyntheticPacketLookup(packets,src)
        events=list(OrderedFlowExtractor(view,configuration(profile)).run(iter_processing_view(where),lookup,input_complete=True))
        return events,make_manifest(view,profile)

    def write(self,events=None,manifest=None,**kwargs):
        return write_events(self.path,iter(self.events if events is None else events),
                            self.manifest if manifest is None else manifest,**kwargs)

    def reject_events(self,events,manifest=None):
        with self.assertRaises((ValueError,TypeError,RuntimeError)):self.write(events,manifest)
        self.assertFalse(self.path.exists())

    def replace_tensor(self,change):
        return [change(e) if type(e) is OrderedFlowTensor else e for e in self.events]

    def test_n20_roundtrip_exact_bytes_and_lineage(self):
        report=self.write();checked=validate_hdf5(self.path)
        original=[e for e in self.events if type(e) is OrderedFlowTensor]
        batches=list(iter_batches(self.path,batch_size=1))
        self.assertEqual(report['content_sha256'],checked['content_sha256'])
        self.assertEqual(report['shape'],[2,20,11,1]);self.assertEqual(report['receipt_count'],4)
        for tensor,batch in zip(original,batches):
            self.assertEqual(tensor.tensor_bytes,batch['X'][0].tobytes())
            self.assertEqual(tensor.metadata,batch['metadata'][0])
            self.assertEqual(batch['y'].tolist(),[-1]);self.assertEqual(batch['supervised_mask'].tolist(),[False])
        self.assertEqual(batches[0]['metadata'][0]['retained_ordinals'],[2,4,3])
        self.assertEqual(batches[0]['metadata'][0]['retained_processing_indices'],[1,2,3])

    def test_n10_roundtrip_exact_bytes_and_padding(self):
        events,manifest=self.prepare(self.packets,'paper-table3-n10');self.write(events,manifest)
        tensors=[e for e in events if type(e) is OrderedFlowTensor]
        batch=next(iter_batches(self.path));self.assertEqual(batch['X'].shape,(2,10,11,1))
        for i,tensor in enumerate(tensors):
            self.assertEqual(batch['X'][i].tobytes(),tensor.tensor_bytes)
            self.assertFalse(batch['X'][i,tensor.metadata['retained_packet_count']:].any())

    def test_overflow_receipts_and_full_window_hash_preserved(self):
        packets=[packet(ordinal=i,timestamp_ns=26-i) for i in range(1,26)]
        events,manifest=self.prepare(packets);self.write(events,manifest)
        meta=next(iter_batches(self.path))['metadata'][0]
        self.assertEqual(meta['real_packet_count'],25);self.assertEqual(meta['overflow_packet_count'],5)
        self.assertEqual(meta['retained_ordinals'],list(range(25,5,-1)))
        self.assertEqual(meta['retained_processing_indices'],list(range(1,21)))

    def test_multiple_bidirectional_flows(self):
        packets=[packet(ordinal=i,timestamp_ns=i,flow=i%3,reverse=i%2==0) for i in range(1,10)]
        events,manifest=self.prepare(packets);report=self.write(events,manifest)
        self.assertEqual(report['sample_count'],3);self.assertEqual(report['receipt_count'],9)

    def test_unknown_and_mixed_excluded_from_supervised(self):
        count=0;events=[]
        for event in self.events:
            if type(event) is OrderedFlowTensor:
                count+=1;event=OrderedHdf5Sample(event,'MIXED' if count==2 else 'UNKNOWN')
            events.append(event)
        report=self.write(events)
        self.assertEqual(report['label_counts'],{'UNKNOWN':1,'MIXED':1})
        self.assertEqual(report['supervised_count'],0)
        self.assertEqual(list(iter_batches(self.path,supervised_only=True)),[])
        self.assertEqual(next(iter_batches(self.path))['y'].tolist(),[-1,-1])

    def test_numeric_label_rejected(self):
        self.reject_events(self.replace_tensor(lambda t:OrderedHdf5Sample(t,0)))

    def test_verified_label_rejected_without_authority(self):
        self.reject_events(self.replace_tensor(lambda t:OrderedHdf5Sample(t,'VERIFIED_DDOS')))

    def test_missing_completion(self):self.reject_events(self.events[:-1])
    def test_empty_stream(self):self.reject_events([])
    def test_duplicate_completion(self):self.reject_events(self.events+[self.events[-1]])
    def test_event_after_completion(self):self.reject_events(self.events+[self.events[0]])
    def test_missing_receipt(self):self.reject_events(self.events[1:])
    def test_duplicate_receipt(self):self.reject_events([self.events[0],*self.events])
    def test_missing_tensor(self):
        dropped=False;events=[]
        for e in self.events:
            if type(e) is OrderedFlowTensor and not dropped:dropped=True;continue
            events.append(e)
        self.reject_events(events)

    def test_duplicate_tensor(self):
        events=[]
        for e in self.events:
            events.append(e)
            if type(e) is OrderedFlowTensor:events.append(e)
        self.reject_events(events)

    def test_forged_completion_counts(self):
        stats=json.loads(self.events[-1].stats_json);stats['emitted_tensors']+=1
        self.reject_events([*self.events[:-1],OrderedExtractionComplete(canonical_bytes(stats))])

    def test_wrong_receipt_source_hash(self):
        event=self.events[0];receipt=replace(event.receipt,packet_sha256='0'*64)
        self.reject_events([replace(event,receipt=receipt),*self.events[1:]])

    def test_wrong_receipt_processing_index(self):
        self.reject_events([replace(self.events[0],processing_index=2),*self.events[1:]])

    def test_wrong_metadata_lineage(self):
        self.reject_events(self.replace_tensor(lambda t:replace(t,metadata_json=canonical_bytes(
            {**t.metadata,'full_packet_digest':'0'*64}))))

    def test_wrong_shape(self):
        self.reject_events(self.replace_tensor(lambda t:replace(t,shape=(10,11,1))))

    def test_wrong_profile_manifest(self):
        other=make_manifest(self.manifest['processing_view'],'paper-table3-n10')
        self.reject_events(self.events,other)

    def test_nonfinite_or_nonzero_padding(self):
        for value,offset in ((float('nan'),0),(1.,20*44-4)):
            def bad(t):
                data=bytearray(t.tensor_bytes);struct.pack_into('<f',data,offset,value)
                return replace(t,tensor_bytes=bytes(data))
            self.reject_events(self.replace_tensor(bad))

    def test_generator_failure_after_provisional_tensor(self):
        def fail():
            for event in self.events:
                yield event
                if type(event) is OrderedFlowTensor:raise RuntimeError('synthetic late failure')
        with self.assertRaises(RuntimeError):self.write(fail())
        self.assertFalse(self.path.exists())
        markers=list(self.home.glob('.*.pending-*/FAILED.json'));self.assertEqual(len(markers),1)

    def test_generator_closed_on_reject(self):
        closed=[]
        def stream():
            try:yield object()
            finally:closed.append(True)
        self.reject_events(stream());self.assertEqual(closed,[True])

    def test_producer_close_failure_does_not_publish(self):
        class Producer:
            def __init__(self,events):self.events=iter(events)
            def __iter__(self):return self
            def __next__(self):return next(self.events)
            def close(self):raise RuntimeError('synthetic producer close failure')
        self.reject_events(Producer(self.events))

    def test_existing_output_never_overwritten(self):
        self.path.write_bytes(b'existing-data');before=self.path.read_bytes()
        with self.assertRaisesRegex(ExportError,'OUTPUT_EXISTS'):self.write()
        self.assertEqual(self.path.read_bytes(),before)

    def test_repeated_export_content_digest_deterministic(self):
        first=self.write();self.path=self.home/'second.hdf5';second=self.write()
        self.assertEqual(first['content_sha256'],second['content_sha256'])
        # Không cam kết byte-for-byte HDF5 giữa library/platform khác nhau.

    def test_no_compression_supported(self):
        self.write(compression=None);self.assertEqual(validate_hdf5(self.path)['sample_count'],2)

    def test_bad_compression_rejected_before_staging(self):
        with self.assertRaises(ExportError):self.write(compression='lzf')
        self.assertFalse(list(self.home.glob('.*.pending-*')))

    def test_manifest_fail_closed_fields(self):
        for key,value in (('input_kind','decoded_pcap'),('real_data_export','APPROVED'),
                          ('supervised_eligible',True),('schema_version','2.0.0')):
            bad=deepcopy(self.manifest);bad[key]=value
            with self.assertRaises(ValueError):validate_manifest(bad)

    def test_manifest_no_extra_fields(self):
        bad=deepcopy(self.manifest);bad['approved']=True
        with self.assertRaises(ValueError):validate_manifest(bad)

    def test_manifest_snapshot_and_schema_match(self):
        checked=validate_manifest(self.manifest);checked['allowed_states'].append('DDOS')
        self.assertEqual(self.manifest['allowed_states'],['UNKNOWN','MIXED'])
        declared=json.loads((ROOT/'contracts/ordered-flow-hdf5.schema.json').read_text(encoding='utf-8'))
        self.assertEqual(declared,MANIFEST_SCHEMA)

    def test_old_reader_rejects_new_format(self):
        self.write()
        with self.assertRaises(ValueError):old_validate(self.path)

    def test_new_reader_rejects_old_format_and_old_file_unchanged(self):
        from hdf5_fixtures import flows,manifest
        from nids_ml.storage.hdf5 import FlowSample,write_flows
        write_flows(self.path,[FlowSample(t) for t in flows(count=1)],manifest())
        before=self.path.read_bytes()
        with self.assertRaises(ValueError):validate_hdf5(self.path)
        self.assertEqual(self.path.read_bytes(),before)
        self.assertEqual(old_validate(self.path)['sample_count'],1)

    def test_invalid_batch_size_and_filter(self):
        for kwargs in ({'batch_size':0},{'batch_size':True},{'batch_size':129},{'supervised_only':1}):
            with self.assertRaises(ExportError):list(iter_batches(self.path,**kwargs))

    def test_raw_output_path_rejected_without_creation(self):
        with self.assertRaises(ExportError):write_events(ROOT/'data/raw/forbidden.hdf5',self.events,self.manifest)

    def test_corrupt_X_detected(self):
        self.write()
        with self.h5py.File(self.path,'r+') as h:h['X'][0,0,1,0]=0.5
        with self.assertRaises(ValueError):validate_hdf5(self.path)

    def test_tampered_labels_rejected(self):
        self.write()
        with self.h5py.File(self.path,'r+') as h:h['y'][0]=0;h['supervised_mask'][0]=True
        with self.assertRaisesRegex(ValueError,'UNSAFE_SUPERVISED'):validate_hdf5(self.path)

    def test_incomplete_file_rejected(self):
        self.write()
        with self.h5py.File(self.path,'r+') as h:h.attrs.modify('status',b'STAGED')
        with self.assertRaisesRegex(ValueError,'NOT_COMPLETE'):validate_hdf5(self.path)

    def test_tampered_root_hash_rejected(self):
        self.write()
        with self.h5py.File(self.path,'r+') as h:h.attrs.modify('content_sha256',b'0'*64)
        with self.assertRaisesRegex(ValueError,'CONTENT_HASH'):validate_hdf5(self.path)

    def test_extra_dataset_rejected(self):
        self.write()
        with self.h5py.File(self.path,'r+') as h:h.create_dataset('unexpected',data=[1])
        with self.assertRaisesRegex(ValueError,'DATASET_NAMES'):validate_hdf5(self.path)

    def test_soft_link_rejected(self):
        self.write()
        with self.h5py.File(self.path,'r+') as h:del h['y'];h['y']=self.h5py.SoftLink('/supervised_mask')
        with self.assertRaisesRegex(ValueError,'SOFT_LINK'):validate_hdf5(self.path)

    def test_truncated_file_rejected(self):
        self.path.write_bytes(b'\x89HDF\r\n\x1a\n')
        with self.assertRaises(ValueError):validate_hdf5(self.path)

    def test_vlen_rejected_before_read(self):
        self.write()
        with self.h5py.File(self.path,'r+') as h:
            del h['metadata_json'];h.create_dataset('metadata_json',data=['x','y'],dtype=self.h5py.string_dtype())
        with self.assertRaisesRegex(ValueError,'DTYPE_OR_SHAPE'):validate_hdf5(self.path)

    def test_wrong_stored_dtype_rejected(self):
        self.write()
        with self.h5py.File(self.path,'r+') as h:
            data=h['X'][:].astype('<f8');del h['X'];h.create_dataset('X',data=data)
        with self.assertRaisesRegex(ValueError,'DTYPE_OR_SHAPE'):validate_hdf5(self.path)

    def test_event_lineage_tamper_rejected(self):
        self.write()
        with self.h5py.File(self.path,'r+') as h:
            value=json.loads(bytes(h['events_json'][0]));value['processing_index']=2
            h['events_json'][0]=canonical_bytes(value)
        with self.assertRaises(ValueError):validate_hdf5(self.path)

    def test_receipt_budget_rejected_before_output(self):
        from nids_ml.storage.ordered_hdf5.validation import Ledger
        with patch('nids_ml.storage.ordered_hdf5.validation.MAX_RECEIPTS',3):
            with self.assertRaisesRegex(ValueError,'RECEIPT_LIMIT'):Ledger(self.manifest)

    def test_reader_does_not_modify_file(self):
        self.write();before=(self.path.stat().st_mtime_ns,hashlib.sha256(self.path.read_bytes()).hexdigest())
        list(iter_batches(self.path));validate_hdf5(self.path)
        self.assertEqual(before,(self.path.stat().st_mtime_ns,hashlib.sha256(self.path.read_bytes()).hexdigest()))

    def test_file_changed_between_batches_rejected(self):
        self.write();batches=iter_batches(self.path,batch_size=1);next(batches)
        # Chỉ thay synthetic signature; không chỉnh thời gian OS hoặc raw files.
        import os
        info=self.path.stat();os.utime(self.path,ns=(info.st_atime_ns,info.st_mtime_ns+1000000000))
        with self.assertRaises(ValueError):next(batches)

    def test_cli_validate_and_no_overwrite(self):
        self.write();out=io.StringIO()
        with redirect_stdout(out):code=main(['validate','--input',str(self.path)])
        self.assertEqual(code,0);self.assertEqual(json.loads(out.getvalue())['supervised_count'],0)

    def test_cli_has_no_real_export_mode(self):
        with redirect_stdout(io.StringIO()),patch('sys.stderr',io.StringIO()):
            with self.assertRaises(SystemExit):main(['export-pcap'])

if __name__=='__main__':unittest.main()
