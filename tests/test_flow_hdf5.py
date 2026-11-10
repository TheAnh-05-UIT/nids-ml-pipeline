"""Test capture-order HDF5 roundtrips, labels, corruption and publication safety.
Test HDF5 capture-order: roundtrip, nhãn, dữ liệu hỏng và publish an toàn.
"""
from copy import deepcopy
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nids_ml.cli.hdf5 import main
from nids_ml.flows.features import CaptureContext,FEATURE_ORDER
from nids_ml.labeling.contracts import canonical_bytes
from nids_ml.labeling.resolver import Label
from nids_ml.storage.hdf5 import ExportError,FlowSample,LabelRecord,make_manifest,write_flows,validate_hdf5,iter_batches
from nids_ml.storage.hdf5 import atomic
from nids_ml.storage.hdf5.schema import backend,MAX_SAMPLES,LABEL_POLICY
from hdf5_fixtures import EVIDENCE_SHA,flows,manifest,mixed_samples,verified_label
from flow_extraction_fixtures import CAPTURE_TEXT,context


class Hdf5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.np,cls.h5py=backend()

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='hdf5-synthetic-',dir=ROOT/'artifacts/runs')
        self.home=Path(self.tmp.name);self.path=self.home/'flows.hdf5'

    def tearDown(self):self.tmp.cleanup()

    def write_unknown(self,profile='flow-cnn-n20-t10',**kwargs):
        items=[FlowSample(flow) for flow in flows(profile)]
        return items,write_flows(self.path,iter(items),manifest(profile),**kwargs)

    def change_metadata(self,flow,**changes):
        return replace(flow,metadata_json=canonical_bytes({**flow.metadata,**changes}))

    def test_N_roundtrip_dtype_shape_and_feature_order(self):
        items,report=self.write_unknown()
        checked=validate_hdf5(self.path)
        batches=list(iter_batches(self.path,batch_size=3))
        X=self.np.concatenate([batch['X'] for batch in batches])
        self.assertEqual(X.shape,(7,20,11,1));self.assertEqual(X.dtype,self.np.float32)
        self.assertEqual(checked['manifest']['extraction_configuration']['feature_order'],list(FEATURE_ORDER))
        self.assertEqual(report['content_sha256'],checked['content_sha256'])
        self.assertEqual([len(batch['X']) for batch in batches],[3,3,1])
        for i,item in enumerate(items):self.assertEqual(X[i].tobytes(),item.tensor.tensor_bytes)

    def test_n10_roundtrip_and_padding_unchanged(self):
        items,_=self.write_unknown('paper-table3-n10')
        batch=next(iter_batches(self.path))
        self.assertEqual(batch['X'].shape,(7,10,11,1))
        for i,item in enumerate(items):
            count=item.tensor.metadata['retained_packet_count']
            self.assertTrue((batch['X'][i,count:]==0).all())
            self.assertEqual(batch['X'][i].tobytes(),item.tensor.tensor_bytes)

    def test_manifest_metadata_sha_receipt_exact_roundtrip(self):
        flow=flows(count=1)[0]
        receipt=hashlib.sha256(b'Synthetic extraction receipt').hexdigest()
        source_manifest=hashlib.sha256(b'Synthetic manifest').hexdigest()
        write_flows(self.path,[FlowSample(flow,extraction_receipt_sha256=receipt,
            extraction_manifest_sha256=source_manifest)],manifest())
        batch=next(iter_batches(self.path));meta=batch['metadata'][0]
        self.assertEqual(meta['flow'],flow.metadata)
        self.assertEqual(meta['flow']['capture']['capture_sha256'],hashlib.sha256(CAPTURE_TEXT.encode()).hexdigest())
        self.assertEqual(meta['extraction_receipt_sha256'],receipt)
        self.assertEqual(meta['extraction_manifest_sha256'],source_manifest)

    def test_missing_evidence_remains_null_not_fabricated(self):
        self.write_unknown();meta=next(iter_batches(self.path))['metadata'][0]
        self.assertIsNone(meta['extraction_receipt_sha256']);self.assertIsNone(meta['extraction_manifest_sha256'])
        self.assertIsNone(meta['processing_indices']);self.assertEqual(meta['processing_view'],'NOT_ACTIVE')

    def test_verified_benign_ddos_and_all_unverified_states(self):
        items=mixed_samples();report=write_flows(self.path,items,manifest(verified=True))
        batch=next(iter_batches(self.path))
        self.assertEqual(batch['y'].tolist(),[0,1,-1,-1,-1,-1,-1])
        self.assertEqual(batch['supervised_mask'].tolist(),[True,True,False,False,False,False,False])
        self.assertEqual(batch['dataset_proposed_y'].tolist(),[-1,-1,-1,-1,-1,0,1])
        self.assertEqual(batch['label_state'],[item.label.state for item in items])
        self.assertEqual(report['supervised_count'],2)

    def test_supervised_reader_never_promotes_proposals(self):
        write_flows(self.path,mixed_samples(),manifest(verified=True))
        batches=list(iter_batches(self.path,batch_size=1,supervised_only=True))
        self.assertEqual([batch['y'].item() for batch in batches],[0,1])
        self.assertEqual([batch['label_state'][0] for batch in batches],['VERIFIED_BENIGN','VERIFIED_DDOS'])

    def test_only_unknown_gives_empty_supervised_iterator(self):
        self.write_unknown();self.assertEqual(list(iter_batches(self.path,supervised_only=True)),[])

    def test_unverified_states_cannot_set_y_or_mask(self):
        flow=flows(count=1)[0]
        for state in ('UNKNOWN','MIXED','OUT_OF_SCOPE','DATASET_NORMAL','DATASET_DDOS'):
            for label in (LabelRecord(state,y=0),LabelRecord(state,supervised_mask=True)):
                with self.subTest(label=label),self.assertRaises(ExportError):
                    write_flows(self.home/(state+str(label.y)+str(label.supervised_mask)+'.h5'),[FlowSample(flow,label)],manifest())

    def test_verified_missing_policy_decision_or_binding_rejected(self):
        flow=flows(count=1)[0];good=verified_label(flow,0)
        for label in (replace(good,decision=None),replace(good,evidence_sha256=None),
                      replace(good,target_metadata_sha256='0'*64),replace(good,y=1),replace(good,supervised_mask=False)):
            with self.subTest(label=label),self.assertRaises(ExportError):
                write_flows(self.path,[FlowSample(flow,label)],manifest(verified=True))
        with self.assertRaisesRegex(ExportError,'POLICY'):
            write_flows(self.path,[FlowSample(flow,good)],manifest())

    def test_target_bool_and_invalid_values_fail(self):
        flow=flows(count=1)[0]
        for y in (True,False,None,2,-2,1.0):
            with self.subTest(y=y),self.assertRaises(ExportError):
                write_flows(self.path,[FlowSample(flow,LabelRecord(y=y))],manifest())

    def test_verified_full_window_count_completeness_mismatch_fail(self):
        flow=flows(count=1)[0];label=verified_label(flow,1)
        for decision in (replace(label.decision,real_packet_count=2),replace(label.decision,full_window_complete=False),
                         replace(label.decision,reasons=('UNEXAMINED_PACKETS',)),
                         replace(label.decision,full_window_label=Label.MIXED)):
            with self.subTest(decision=decision),self.assertRaises(ExportError):
                write_flows(self.path,[FlowSample(flow,replace(label,decision=decision))],manifest(verified=True))

    def test_n20_verified_overflow_uses_full_window_decision(self):
        flow=flows(count=1,packets_per_flow=21)[0]
        label=verified_label(flow,1)
        write_flows(self.path,[FlowSample(flow,label)],manifest(verified=True))
        self.assertEqual(validate_hdf5(self.path)['supervised_count'],1)
        meta=next(iter_batches(self.path))['metadata'][0]['flow']
        self.assertEqual((meta['real_packet_count'],meta['retained_packet_count'],meta['overflow_packet_count']),(21,20,1))

    def test_n10_overflow_not_supervised_by_n20_decision(self):
        flow=flows('paper-table3-n10',count=1,packets_per_flow=11)[0]
        with self.assertRaises(ExportError):
            write_flows(self.path,[FlowSample(flow,verified_label(flow,1))],manifest('paper-table3-n10',verified=True))

    def test_n10_short_verified_window_has_matching_decision_counts(self):
        flow=flows('paper-table3-n10',count=1,packets_per_flow=3)[0]
        write_flows(self.path,[FlowSample(flow,verified_label(flow,0))],manifest('paper-table3-n10',verified=True))
        self.assertEqual(validate_hdf5(self.path)['supervised_count'],1)

    def test_nonfinite_or_out_of_range_tensor_rejected(self):
        flow=flows(count=1)[0]
        for value in (float('nan'),float('inf'),float('-inf'),-0.1,1.1):
            X=flow.to_numpy(batch=False);X[0,1,0]=value
            with self.subTest(value=value),self.assertRaises(ExportError):
                write_flows(self.path,[FlowSample(replace(flow,tensor_bytes=X.tobytes()))],manifest())

    def test_wrong_shape_feature_count_dtype_rejected(self):
        flow=flows(count=1)[0]
        for bad in (replace(flow,shape=(20,12,1)),replace(flow,shape=(10,11,1)),replace(flow,shape=(20,11,True)),
                    replace(flow,tensor_bytes=flow.to_numpy(batch=False).astype('float64').tobytes())):
            with self.subTest(shape=bad.shape),self.assertRaises(ExportError):write_flows(self.path,[FlowSample(bad)],manifest())

    def test_nonzero_padding_is_rejected_not_repaired(self):
        flow=flows(count=1)[0];X=flow.to_numpy(batch=False);X[-1,1,0]=0.2
        with self.assertRaisesRegex(ExportError,'PADDING'):
            write_flows(self.path,[FlowSample(replace(flow,tensor_bytes=X.tobytes()))],manifest())

    def test_invalid_schema_or_feature_order_rejected(self):
        for version in ('2.0.0',None,1):
            value=manifest();value['schema_version']=version
            with self.subTest(version=version),self.assertRaises(ExportError):write_flows(self.path,[],value)
        value=manifest();value['extraction_configuration']['feature_order'].reverse()
        with self.assertRaises(ExportError):write_flows(self.path,[],value)

    def test_raw_and_processing_view_gate_fail_before_iterating(self):
        def forbidden():raise AssertionError('Input iterable must not be consumed');yield
        value=manifest();value['input_kind']='decoded_pcap'
        with self.assertRaisesRegex(ExportError,'REAL_DATA_EXPORT_BLOCKED'):write_flows(self.path,forbidden(),value)
        value=manifest();value['ordering_policy']['processing_view']='ACTIVE'
        with self.assertRaises(ExportError):write_flows(self.path,forbidden(),value)
        with self.assertRaises(ExportError):make_manifest(CaptureContext.from_dict(context()),synthetic_capture_text=CAPTURE_TEXT,input_kind='real_data')
        self.assertFalse(self.path.exists())

    def test_literal_hash_and_real_decoder_metadata_rejected(self):
        with self.assertRaises(ExportError):make_manifest(CaptureContext.from_dict(context()),synthetic_capture_text='wrong literal')
        flow=flows(count=1)[0]
        for bad in (self.change_metadata(flow,schema_version='1.1.0'),
                    self.change_metadata(flow,input_kind='decoded_pcap')):
            with self.assertRaises(ExportError):write_flows(self.path,[FlowSample(bad)],manifest())

    def test_unknown_metadata_fields_and_missing_fields_fail(self):
        flow=flows(count=1)[0];meta=flow.metadata;del meta['full_packet_digest']
        for bad in (replace(flow,metadata_json=canonical_bytes(meta)),self.change_metadata(flow,raw_pcap_bytes='forbidden')):
            with self.assertRaises(ExportError):write_flows(self.path,[FlowSample(bad)],manifest())

    def test_empty_dataset_rejected_without_completed_output(self):
        with self.assertRaisesRegex(ExportError,'REJECT_EMPTY'):write_flows(self.path,[],manifest())
        self.assertFalse(self.path.exists())
        failed=list(self.home.glob('*.pending-*'))
        self.assertTrue(failed)
        self.assertTrue(all((directory/'FAILED.json').is_file() for directory in failed))

    def test_existing_output_never_overwritten(self):
        self.path.write_bytes(b'operator-owned existing artifact')
        with self.assertRaisesRegex(ExportError,'OUTPUT_EXISTS'):write_flows(self.path,[],manifest())
        self.assertEqual(self.path.read_bytes(),b'operator-owned existing artifact')

    def test_mid_write_failure_closes_and_marks_staging_failed(self):
        flow=flows(count=1)[0]
        with patch('nids_ml.storage.hdf5.writer._append',side_effect=OSError('synthetic disk failure')):
            with self.assertRaises(OSError):write_flows(self.path,[FlowSample(flow)],manifest())
        self.assertFalse(self.path.exists())
        directory=next(self.home.glob('*.pending-*'))
        self.assertEqual(json.loads((directory/'FAILED.json').read_text())['status'],'FAILED_NOT_PUBLISHED')
        with self.h5py.File(directory/'payload.hdf5','r') as handle:self.assertEqual(bytes(handle.attrs['status']),b'FAILED')
        with self.assertRaises(ExportError):validate_hdf5(directory/'payload.hdf5')

    def test_failure_in_readback_validator_never_publishes(self):
        with patch('nids_ml.storage.hdf5.writer._validate',side_effect=ExportError('synthetic readback failure')):
            with self.assertRaises(ExportError):self.write_unknown()
        self.assertFalse(self.path.exists())

    def test_output_collision_during_atomic_publish_preserves_other_file(self):
        original=atomic.publish
        def collision(stage,target):
            target.write_bytes(b'new operator file');original(stage,target)
        with patch('nids_ml.storage.hdf5.writer.atomic.publish',side_effect=collision):
            with self.assertRaises(ExportError):self.write_unknown()
        self.assertEqual(self.path.read_bytes(),b'new operator file')

    def test_atomic_unlink_failure_rolls_back_only_new_output(self):
        original=Path.unlink
        def failure(path,*args,**kwargs):
            if path.name=='payload.hdf5':raise OSError('synthetic staging unlink failure')
            return original(path,*args,**kwargs)
        with patch('nids_ml.storage.hdf5.atomic.Path.unlink',new=failure):
            with self.assertRaises(ExportError):self.write_unknown()
        self.assertFalse(self.path.exists())
        self.assertTrue(list(self.home.glob('*.pending-*/FAILED.json')))

    def test_bad_lineage_or_cross_capture_identity_rejected(self):
        flow=flows(count=1)[0]
        changed=self.change_metadata(flow,capture={**flow.metadata['capture'],'capture_sha256':'0'*64})
        for sample in (FlowSample(flow,extraction_receipt_sha256='not-a-sha'),FlowSample(changed)):
            with self.assertRaises(ExportError):write_flows(self.path,[sample],manifest())

    def test_source_order_metadata_and_append_order_preserved(self):
        items=[FlowSample(flow) for flow in reversed(flows())]
        before=[item.tensor.metadata_json for item in items]
        write_flows(self.path,items,manifest())
        records=next(iter_batches(self.path))['metadata']
        self.assertEqual([record['flow'] for record in records],[item.tensor.metadata for item in items])
        self.assertEqual([item.tensor.metadata_json for item in items],before)
        self.assertTrue(all(record['processing_indices'] is None for record in records))

    def test_timestamp_or_ordinal_regression_rejected_without_sort(self):
        flow=flows(count=1,packets_per_flow=3)[0]
        for bad in (self.change_metadata(flow,last_timestamp_ns=flow.metadata['first_timestamp_ns']-1),
                    self.change_metadata(flow,retained_ordinals=[3,2,1])):
            original=bad.metadata_json
            with self.assertRaises(ExportError):write_flows(self.path,[FlowSample(bad)],manifest())
            self.assertEqual(bad.metadata_json,original)
        with self.assertRaisesRegex(ExportError,'PROCESSING_VIEW'):
            write_flows(self.path,[FlowSample(flow,processing_indices=(1,2,3))],manifest())

    def test_duplicate_flow_window_and_sample_limit_rejected(self):
        flow=flows(count=1)[0]
        with self.assertRaisesRegex(ExportError,'DUPLICATE_FLOW_WINDOW'):
            write_flows(self.path,[FlowSample(flow),FlowSample(flow)],manifest())
        with self.assertRaisesRegex(ExportError,'SAMPLE_LIMIT'):self.write_unknown(max_samples=1)

    def test_bad_batch_size_and_filter_flags_rejected(self):
        self.write_unknown()
        for size in (0,129,True,1.0):
            with self.subTest(size=size),self.assertRaises(ExportError):validate_hdf5(self.path,batch_size=size)
        with self.assertRaises(ExportError):list(iter_batches(self.path,supervised_only=1))

    def test_compression_none_and_gzip_preserve_bitwise_tensors(self):
        flow=flows(count=1)[0]
        for mode in (None,'gzip'):
            path=self.home/(str(mode)+'.hdf5')
            write_flows(path,[FlowSample(flow)],manifest(),compression=mode)
            self.assertEqual(next(iter_batches(path))['X'][0].tobytes(),flow.tensor_bytes)
        with self.assertRaises(ExportError):write_flows(self.path,[],manifest(),compression='lzf')

    def test_identical_inputs_have_same_logical_content_digest(self):
        items=mixed_samples();value=manifest(verified=True)
        first=write_flows(self.path,items,value)
        second=write_flows(self.home/'another.hdf5',iter(items),value,compression=None)
        self.assertEqual(first['content_sha256'],second['content_sha256'])
        self.assertEqual(value,manifest(verified=True))

    def test_reader_detects_tensor_corruption(self):
        self.write_unknown()
        with self.h5py.File(self.path,'r+') as handle:handle['X'][0,0,1,0]=0.5
        with self.assertRaisesRegex(ExportError,'HASH_MISMATCH'):validate_hdf5(self.path)

    def test_reader_detects_label_promotion_and_column_corruption(self):
        self.write_unknown()
        with self.h5py.File(self.path,'r+') as handle:handle['supervised_mask'][0]=True
        with self.assertRaisesRegex(ExportError,'LABEL_COLUMN_MISMATCH'):list(iter_batches(self.path,supervised_only=True))

    def test_reader_rejects_coherent_dataset_label_promotion_semantically(self):
        self.write_unknown()
        with self.h5py.File(self.path,'r+') as handle:
            value=json.loads(bytes(handle['labels_json'][0]))
            value.update(state='DATASET_DDOS',y=1,supervised_mask=True,dataset_proposed_y=1)
            handle['labels_json'][0]=canonical_bytes(value)
            handle['label_state'][0]=b'DATASET_DDOS';handle['y'][0]=1
            handle['supervised_mask'][0]=True;handle['dataset_proposed_y'][0]=1
        with self.assertRaisesRegex(ExportError,'UNVERIFIED_MUST_BE_MASKED'):validate_hdf5(self.path)

    def test_reader_rejects_schema_version_and_incomplete_state(self):
        self.write_unknown()
        with self.h5py.File(self.path,'r+') as handle:handle.attrs.modify('schema_version',b'9.0.0')
        with self.assertRaisesRegex(ExportError,'VERSION'):validate_hdf5(self.path)
        with self.h5py.File(self.path,'r+') as handle:
            handle.attrs.modify('schema_version',b'1.0.0');handle.attrs.modify('status',b'STAGED')
        with self.assertRaisesRegex(ExportError,'NOT_COMPLETE'):validate_hdf5(self.path)

    def test_reader_rejects_wrong_dataset_dtype_before_data(self):
        self.write_unknown()
        with self.h5py.File(self.path,'r+') as handle:
            values=handle['X'][:].astype('float64');del handle['X'];handle.create_dataset('X',data=values)
        with self.assertRaisesRegex(ExportError,'SHAPE_OR_DTYPE'):validate_hdf5(self.path)

    def test_reader_rejects_vlen_metadata(self):
        self.write_unknown()
        with self.h5py.File(self.path,'r+') as handle:
            del handle['metadata_json']
            handle.create_dataset('metadata_json',shape=(7,),dtype=self.h5py.string_dtype('utf-8'))
        with self.assertRaisesRegex(ExportError,'SHAPE_OR_DTYPE'):validate_hdf5(self.path)

    def test_reader_rejects_external_links_without_opening_target(self):
        self.write_unknown()
        with self.h5py.File(self.path,'r+') as handle:
            del handle['X'];handle['X']=self.h5py.ExternalLink('missing-not-opened.hdf5','X')
        with self.assertRaisesRegex(ExportError,'EXTERNAL_OR_SOFT'):validate_hdf5(self.path)

    def test_reader_rejects_too_many_rows_or_extra_payload(self):
        self.write_unknown()
        with self.h5py.File(self.path,'r+') as handle:handle.attrs.modify('sample_count',self.np.array(MAX_SAMPLES+1,dtype='<u8'))
        with self.assertRaisesRegex(ExportError,'SAMPLE_COUNT'):validate_hdf5(self.path)
        with self.h5py.File(self.path,'r+') as handle:
            handle.attrs.modify('sample_count',self.np.array(7,dtype='<u8'));handle.create_dataset('raw_pcap',data=[1])
        with self.assertRaisesRegex(ExportError,'DATASET_NAMES'):validate_hdf5(self.path)

    def test_ordered_content_digest_rejects_coherent_row_swap(self):
        self.write_unknown()
        with self.h5py.File(self.path,'r+') as handle:
            for name in ('X','y','label_state','supervised_mask','dataset_proposed_y','metadata_json','labels_json','sample_sha256'):
                left,right=handle[name][0],handle[name][1];handle[name][0]=right;handle[name][1]=left
        with self.assertRaisesRegex(ExportError,'CONTENT_HASH'):validate_hdf5(self.path)

    def test_path_guard_rejects_raw_or_arbitrary_output(self):
        with self.assertRaises(ExportError):write_flows(ROOT/'data/raw/not-created.hdf5',[],manifest())
        self.assertFalse((ROOT/'data/raw/not-created.hdf5').exists())

    def test_hdf5_batches_match_CNN_input_and_forward_without_training(self):
        from nids_ml.models.flow_cnn import configuration as model_configuration,build_model,predict_flow,validate_inputs
        for profile in ('flow-cnn-n20-t10','paper-table3-n10'):
            path=self.home/(profile+'.hdf5');items=[FlowSample(flow) for flow in flows(profile)]
            write_flows(path,items,manifest(profile));X=next(iter_batches(path))['X'];cfg=model_configuration(profile)
            validate_inputs(X,cfg);model=build_model(cfg,compile_model=False)
            prediction=predict_flow(model,X,cfg)
            self.assertEqual(prediction.shape,(7,1));self.assertTrue(self.np.isfinite(prediction).all())

    def test_cli_synthetic_export_inspect_and_no_reexport(self):
        out=io.StringIO()
        with redirect_stdout(out):
            code=main(['synthetic-export','--fixture',str(ROOT/'configs/flow-extraction-synthetic.example.json'),
                       '--output',str(self.path)])
        self.assertEqual(code,0,out.getvalue());self.assertEqual(json.loads(out.getvalue())['sample_count'],4)
        data=self.path.read_bytes()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(['inspect','--input',str(self.path)]),0)
            self.assertEqual(main(['synthetic-export','--fixture',str(ROOT/'configs/flow-extraction-synthetic.example.json'),
                                   '--output',str(self.path)]),2)
        self.assertEqual(self.path.read_bytes(),data)

    def test_cli_rejects_real_scope_fixture(self):
        value=json.loads((ROOT/'configs/flow-extraction-synthetic.example.json').read_text())
        value['kind']='real_dataset';path=self.home/'real.json';path.write_bytes(canonical_bytes(value))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(['synthetic-export','--fixture',str(path),'--output',str(self.path)]),2)
        self.assertFalse(self.path.exists())

    def test_runtime_version_mismatch_does_not_auto_install(self):
        with patch.object(self.h5py,'__version__','0.0.0'):
            with self.assertRaisesRegex(RuntimeError,'Unvalidated HDF5 runtime'):backend()


if __name__=='__main__':unittest.main()
