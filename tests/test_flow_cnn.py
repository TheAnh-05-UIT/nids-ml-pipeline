"""Test CNN contracts, forward passes, checkpoints and tiny synthetic learning.
Test contract CNN, forward, checkpoint và learning synthetic nhỏ.
"""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nids_ml.models.flow_cnn import (FEATURE_ORDER,PROFILES,ModelContractError,backend,build_model,
    canonical,configuration,inspect_model,learning_sanity,predict_flow,synthetic_batch,
    train_step,training_reference,validate_configuration,validate_inputs,validate_labels)
from nids_ml.models.flow_checkpoint import (RUNS_ROOT,TOTAL_BYTES,checkpoint_schema,load_checkpoint,
    model_hash,save_checkpoint)
from nids_ml.cli.flow_cnn import main


class ConfigurationTests(unittest.TestCase):
    def test_both_configurations_checked_in_match_runtime(self):
        for name in PROFILES:
            value=json.loads((ROOT/'configs'/f'{name}.json').read_text())
            self.assertEqual(value,configuration(name));self.assertEqual(validate_configuration(value),value)

    def test_feature_order_exact_eleven(self):
        self.assertEqual(FEATURE_ORDER,('Time','Packet Length','Higher Protocol','IP Flags','Protocols','TCP Length','TCP Ack','TCP Flags','TCP Window Size','UDP Length','ICMP Type'))

    def test_reference_training_config_and_provenance(self):
        value=json.loads((ROOT/'configs/flow-cnn-training-reference.json').read_text())
        self.assertEqual(value,training_reference())
        for key,want in [('optimizer','Adam'),('learning_rate',0.001),('batch_size',128),('epochs',10),('loss','binary_crossentropy')]:
            self.assertEqual(value[key],want)
        self.assertIn('PROPOSED_IMPLEMENTATION',value['adam_details_status'])

    def test_reject_unknown_configuration_and_seed(self):
        for value in ['unknown','n10']:
            with self.assertRaises(ModelContractError):configuration(value)
        for value in [True,-1,2**31,1.0]:
            with self.assertRaises(ModelContractError):configuration(seed=value)

    def test_reject_feature_reordering_architecture_version_or_extra_layer(self):
        for key,value in [('feature_order',list(reversed(FEATURE_ORDER))),('architecture_version','other'),('extra_layer','Dropout'),('input_shape',[20,61,1])]:
            cfg=configuration();cfg[key]=value
            with self.subTest(key=key),self.assertRaises(ModelContractError):validate_configuration(cfg)
        cfg=configuration();cfg['convolution']['kernel_size']=[1,11]
        with self.assertRaises(ModelContractError):validate_configuration(cfg)

    def test_checkpoint_schema_export(self):
        self.assertEqual(json.loads((ROOT/'contracts/flow-cnn-checkpoint.schema.json').read_text()),checkpoint_schema())


class FrameworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:cls.np,cls.tf=backend()
        except (ImportError,RuntimeError) as error:raise unittest.SkipTest('Framework unavailable: '+str(error))

    def setUp(self):
        self.config=configuration()

    def test_model_initialization_both_profiles(self):
        for name,n in PROFILES.items():
            cfg=configuration(name);m=build_model(cfg);report=inspect_model(m,cfg)
            self.assertEqual(m.input_shape,(None,n,11,1));self.assertEqual(m.output_shape,(None,1))
            self.assertEqual(report['layers'][0]['output_shape'],[None,n-2,1,64])

    def test_layer_and_trainable_parameter_counts(self):
        m=build_model(self.config)
        self.assertEqual([x.count_params() for x in m.layers],[2176,0,0,0,65])
        self.assertEqual(sum(int(self.np.prod(w.shape)) for w in m.trainable_weights),2241)
        self.assertEqual(m.non_trainable_weights,[])

    def test_forward_shape_probabilities_and_cpu(self):
        for name in PROFILES:
            cfg=configuration(name);m=build_model(cfg);x,y=synthetic_batch(cfg)
            prediction=predict_flow(m,x,cfg)
            self.assertEqual(prediction.shape,(16,1));self.assertEqual(prediction.dtype,self.np.float32)
            self.assertTrue(self.np.isfinite(prediction).all());self.assertTrue(((prediction>=0)&(prediction<=1)).all())
            self.assertTrue(all('CPU' in w.device for w in m.weights))

    def test_reference_batch128_forward(self):
        m=build_model(self.config);x,y=synthetic_batch(self.config)
        self.assertEqual(predict_flow(m,self.np.tile(x,(8,1,1,1)),self.config).shape,(128,1))

    def test_backward_finite_gradients_and_update_both_profiles(self):
        for name in PROFILES:
            cfg=configuration(name);m=build_model(cfg);x,y=synthetic_batch(cfg)
            before=m.get_weights();report=train_step(m,x,y,cfg)
            self.assertTrue(self.np.isfinite(report['loss_before_update']));self.assertGreater(report['gradient_norm'],0)
            self.assertEqual(report['gradient_shapes'],[[3,11,1,64],[64],[64,1],[1]])
            self.assertTrue(any(not self.np.array_equal(a,b) for a,b in zip(before,m.get_weights())))

    def test_tiny_synthetic_learning_both_profiles(self):
        for name in PROFILES:
            m,report=learning_sanity(configuration(name))
            self.assertTrue(report['passed'],report);self.assertLess(report['loss_after'],report['loss_before']-0.01)
            self.assertEqual(report['samples'],16);self.assertEqual(report['steps'],32);self.assertFalse(report['dataset_result'])

    def test_deterministic_initialization_same_seed(self):
        for name in PROFILES:
            a=build_model(configuration(name));b=build_model(configuration(name))
            for wa,wb in zip(a.get_weights(),b.get_weights()):self.np.testing.assert_array_equal(wa,wb)

    def test_distinct_seeds_change_initialization(self):
        a=build_model(configuration(seed=42));b=build_model(configuration(seed=43))
        self.assertTrue(any(not self.np.array_equal(wa,wb) for wa,wb in zip(a.get_weights(),b.get_weights())))

    def test_invalid_input_shape_dtype_nonfinite_or_batch(self):
        np=self.np
        cases=[np.zeros((2,20,11),np.float32),np.zeros((2,10,11,1),np.float32),np.zeros((2,20,12,1),np.float32),
               np.zeros((2,20,11,1),np.float64),np.zeros((2,20,11,1),np.int32),np.zeros((0,20,11,1),np.float32),
               np.zeros((129,20,11,1),np.float32),np.full((1,20,11,1),np.nan,np.float32),np.full((1,20,11,1),np.inf,np.float32)]
        for x in cases:
            with self.subTest(shape=x.shape,dtype=str(x.dtype)),self.assertRaises(ModelContractError):validate_inputs(x,self.config)

    def test_unknown_mixed_invalid_labels_rejected(self):
        np=self.np
        for y in [np.array([['UNKNOWN']]),np.array([['MIXED']]),np.array([[2]],np.float32),np.array([[-1]],np.float32),
                  np.array([[np.nan]],np.float32),np.array([0],np.float32),np.array([[0]],np.float64),np.array([[False]])]:
            with self.subTest(dtype=str(y.dtype)),self.assertRaises(ModelContractError):validate_labels(y,1)

    def test_wrong_profile_predict_and_uncompiled_training_rejected(self):
        m=build_model(self.config);x,y=synthetic_batch(configuration('paper-table3-n10'))
        with self.assertRaises(ModelContractError):predict_flow(m,x,self.config)
        m=build_model(self.config,compile_model=False);x,y=synthetic_batch(self.config)
        with self.assertRaises(ModelContractError):train_step(m,x,y,self.config)

    def test_synthetic_budget_rejected(self):
        for value in [0,33,True]:
            with self.assertRaises(ModelContractError):learning_sanity(self.config,value)
        with self.assertRaises(ModelContractError):synthetic_batch(self.config,samples=1000)

    def test_model_semantics_mutation_rejected(self):
        m=build_model(self.config);m.layers[1].activation=self.tf.keras.activations.sigmoid
        with self.assertRaises(ModelContractError):inspect_model(m,self.config)

    def test_checkpoint_roundtrip_predictions_and_optimizer_reset(self):
        for name in PROFILES:
            cfg=configuration(name);m=build_model(cfg);x,y=synthetic_batch(cfg);train_step(m,x,y,cfg)
            with tempfile.TemporaryDirectory(prefix='flow-cnn-',dir=RUNS_ROOT) as tmp:
                path=Path(tmp)/'checkpoint';manifest=save_checkpoint(m,cfg,path)
                restored,loaded=load_checkpoint(path,expected_config=cfg)
                self.np.testing.assert_allclose(predict_flow(m,x,cfg),predict_flow(restored,x,cfg),rtol=0,atol=1e-7)
                self.assertEqual(loaded,manifest);self.assertEqual((path/'weights.bin').stat().st_size,TOTAL_BYTES)
                self.assertEqual(loaded['training_state']['optimizer_steps_observed'],1)
                self.assertEqual(int(restored.optimizer.iterations.numpy()),0)

    def test_checkpoint_no_overwrite_and_no_outside_runs(self):
        m=build_model(self.config)
        with tempfile.TemporaryDirectory(prefix='flow-cnn-',dir=RUNS_ROOT) as tmp:
            path=Path(tmp)/'checkpoint';save_checkpoint(m,self.config,path);before=(path/'weights.bin').read_bytes()
            with self.assertRaises(FileExistsError):save_checkpoint(m,self.config,path)
            self.assertEqual((path/'weights.bin').read_bytes(),before)
        with self.assertRaises(ModelContractError):save_checkpoint(m,self.config,ROOT/'data/raw/forbidden-checkpoint')
        self.assertFalse((ROOT/'data/raw/forbidden-checkpoint').exists())

    def test_checkpoint_wrong_profile_rejected(self):
        m=build_model(self.config)
        with tempfile.TemporaryDirectory(prefix='flow-cnn-',dir=RUNS_ROOT) as tmp:
            path=Path(tmp)/'checkpoint';save_checkpoint(m,self.config,path)
            with self.assertRaises(ModelContractError):load_checkpoint(path,expected_config=configuration('paper-table3-n10'))

    def test_checkpoint_metadata_shape_dtype_version_or_hash_tampering(self):
        m=build_model(self.config)
        for key in ['dtype','shape','format_version','weights_file','weights_sha256','model_hash','label_mapping','runtime']:
            with self.subTest(key=key),tempfile.TemporaryDirectory(prefix='flow-cnn-',dir=RUNS_ROOT) as tmp:
                path=Path(tmp)/'checkpoint';meta=save_checkpoint(m,self.config,path)
                if key=='dtype':meta['tensors'][0]['dtype']='float64'
                elif key=='shape':meta['tensors'][0]['shape']=[1]
                elif key=='runtime':meta['runtime']['tensorflow']='99'
                elif key=='label_mapping':meta[key]={'0':'DDoS','1':'Benign'}
                else:meta[key]='bad'
                (path/'manifest.json').write_bytes(canonical(meta))
                with self.assertRaises(ModelContractError):load_checkpoint(path)

    def test_checkpoint_corrupted_payload_rejected(self):
        m=build_model(self.config)
        with tempfile.TemporaryDirectory(prefix='flow-cnn-',dir=RUNS_ROOT) as tmp:
            path=Path(tmp)/'checkpoint';save_checkpoint(m,self.config,path)
            payload=bytearray((path/'weights.bin').read_bytes());payload[0]^=1;(path/'weights.bin').write_bytes(payload)
            with self.assertRaises(ModelContractError):load_checkpoint(path)

    def test_checkpoint_nonfinite_rejected_even_with_updated_checksums(self):
        m=build_model(self.config)
        with tempfile.TemporaryDirectory(prefix='flow-cnn-',dir=RUNS_ROOT) as tmp:
            path=Path(tmp)/'checkpoint';meta=save_checkpoint(m,self.config,path)
            payload=bytearray((path/'weights.bin').read_bytes());payload[:4]=self.np.array([self.np.nan],dtype='<f4').tobytes()
            meta['weights_sha256']=hashlib.sha256(payload).hexdigest();meta['model_hash']=model_hash(self.config,meta['weights_sha256'])
            size=meta['tensors'][0]['size_bytes'];meta['tensors'][0]['sha256']=hashlib.sha256(payload[:size]).hexdigest()
            (path/'manifest.json').write_bytes(canonical(meta));(path/'weights.bin').write_bytes(payload)
            with self.assertRaisesRegex(ModelContractError,'Nonfinite'):load_checkpoint(path)

    def test_profile_part_of_model_hash(self):
        a=build_model(self.config);cfg=configuration('paper-table3-n10');b=build_model(cfg)
        for wa,wb in zip(a.get_weights(),b.get_weights()):self.np.testing.assert_array_equal(wa,wb)
        with tempfile.TemporaryDirectory(prefix='flow-cnn-',dir=RUNS_ROOT) as tmp:
            aa=save_checkpoint(a,self.config,Path(tmp)/'a');bb=save_checkpoint(b,cfg,Path(tmp)/'b')
            self.assertEqual(aa['weights_sha256'],bb['weights_sha256']);self.assertNotEqual(aa['model_hash'],bb['model_hash'])

    def test_cli_inspect_both_profiles(self):
        for name in PROFILES:
            out=io.StringIO()
            with redirect_stdout(out):code=main(['inspect','--profile',name])
            self.assertEqual(code,0);self.assertEqual(json.loads(out.getvalue())['model']['trainable_parameters'],2241)

    def test_cli_sanity_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix='flow-cnn-',dir=RUNS_ROOT) as tmp:
            out=io.StringIO()
            with redirect_stdout(out):code=main(['sanity','--checkpoint',str(Path(tmp)/'checkpoint')])
            self.assertEqual(code,0,out.getvalue());result=json.loads(out.getvalue())
            self.assertTrue(result['metrics']['passed']);self.assertLessEqual(result['reload_prediction_max_abs_error'],1e-7)


if __name__=='__main__':
    unittest.main()
