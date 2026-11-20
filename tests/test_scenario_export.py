"""Test approved-scenario pipes/HDF5 with synthetic captures and explicit fake approval.
Kiểm pipe/HDF5 bằng capture giả, approval fixture; không phải metrics MQTT thật.
"""
from contextlib import closing
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tests'))
from decoded_pcap_fixtures import make_capture,pdml,FakeDecoderProcess
from nids_ml.flows.pcap_stream import stream_scenario
from nids_ml.flows.streaming import StreamingLimits
from nids_ml.labeling.contracts import canonical_bytes
from nids_ml.labeling.scenario import load_policy,ScenarioStreamingLimits,PROVENANCE
from nids_ml.storage.scenario import export_scenario_pipe,read_frame,write_frame,ScenarioShardReader
from nids_ml.storage.shards import FlowShardReader
from nids_ml.data.scenario_audit import audit_scenario_sources


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.work=Path(self.tmp.name)
        for target,value in [('shutil.disk_usage',SimpleNamespace(free=10*1024**3)),
                             ('nids_ml.pcap.decoder.process_rss',1024),
                             ('nids_ml.pcap.whole.available_memory_bytes',10*1024**3)]:
            mock=patch(target,return_value=value);mock.start();self.addCleanup(mock.stop)
        self.fixture=make_capture(self.work/'fixture')
        self.policy=dict(format='nids-scenario-label-policy/1.0.0',dataset_id='synthetic-only',
            label_provenance=PROVENANCE,independent_ground_truth_verified=False,
            approval=dict(status='APPROVED_BY_OPERATOR',record_id='SYNTHETIC_FIXTURE_NOT_REAL_APPROVAL',
                          evidence='Synthetic software-validation fixture only; no real dataset approval.'),
            sources=[self.rule(self.fixture,'normal-test','test',0)])
        self.save_policy()

    def rule(self,f,source_id,split,y):
        return dict(source_id=source_id,file_name=source_id+'.pcap',sha256=f['descriptor']['capture']['capture_sha256'],
                    size_bytes=f['source'].stat().st_size,records=len(f['records']),publisher_split=split,y=y,
                    attack_type='Normal' if y==0 else 'MQTT_DDoS_Connect_Flood',max_lateness_ns=2*10**9)

    def save_policy(self):
        self.policy_path=self.work/'policy.json';self.policy_path.write_bytes(canonical_bytes(self.policy))
        self.policy_sha=hashlib.sha256(self.policy_path.read_bytes()).hexdigest()

    def produce(self,f=None,source_id='normal-test',name='producer',code=0):
        f=self.fixture if f is None else f
        pipe=io.BytesIO();child=FakeDecoderProcess(pdml(f['records']),code)
        with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=child):
            result=stream_scenario(f['source'],self.work/name,pipe,policy_path=self.policy_path,
                policy_sha=self.policy_sha,source_id=source_id,tshark=f['tool'],tshark_sha256=f['tool_sha'])
        return pipe.getvalue(),result

    def consume(self,data,name='hdf5',source_id='normal-test'):
        return export_scenario_pipe(io.BytesIO(data),self.policy_path,self.policy_sha,source_id,
                                    self.work/name,max_output_mib=16)

    def frames(self,data):
        stream=io.BytesIO(data);rows=[]
        while stream.tell()<len(data):rows.append(read_frame(stream))
        return rows

    def encode(self,rows):
        stream=io.BytesIO()
        for kind,payload in rows:write_frame(stream,kind,payload)
        return stream.getvalue()

    def test_full_roundtrip_preserves_official_split_and_weak_provenance(self):
        data,result=self.produce();m=self.consume(data)
        self.assertEqual(result['status'],'PASS_WHOLE_SCENARIO_STREAM')
        self.assertEqual(m['samples'],2)
        self.assertEqual(m['label_counts'],{'0':2})
        self.assertEqual(m['split'],'PUBLISHER_TEST')
        self.assertFalse(m['training_ready']);self.assertFalse(m['independent_ground_truth_verified'])
        reader=ScenarioShardReader(self.work/'hdf5',self.policy_path,self.policy_sha)
        with closing(reader.batches(metadata=True)) as batches:
            for batch in batches:
                for raw in batch['metadata']:
                    meta=json.loads(raw.tobytes())
                    self.assertEqual(meta['label_provenance'],PROVENANCE)
                    self.assertEqual(meta['core']['full_window_label'],'UNKNOWN')
                    self.assertFalse(meta['core']['supervised_eligible'])
        with self.assertRaisesRegex(ValueError,'SHARD_MANIFEST_FORMAT'):
            FlowShardReader(self.work/'hdf5')

    def test_positive_label_and_overflow_do_not_change_features(self):
        self.policy['sources'][0].update(y=1,attack_type='MQTT_DDoS_Connect_Flood')
        self.save_policy();data,_=self.produce();m=self.consume(data)
        self.assertEqual(m['label_counts'],{'1':2})
        self.assertEqual(m['attack_type_counts'],{'MQTT_DDoS_Connect_Flood':2})
        f=make_capture(self.work/'overflow',timestamps=tuple(i*1000 for i in range(25)))
        self.policy['sources']=[self.rule(f,'overflow-train','train',1)];self.save_policy()
        data,_=self.produce(f,'overflow-train','overflow-producer')
        self.consume(data,'overflow-hdf5','overflow-train')
        with closing(ScenarioShardReader(self.work/'overflow-hdf5',self.policy_path,self.policy_sha).batches(metadata=True)) as batches:
            meta=json.loads(next(batches)['metadata'][0].tobytes())
        self.assertEqual(meta['core']['real_packet_count'],25)
        self.assertEqual(meta['core']['retained_packet_count'],20)

    def test_policy_approval_hash_label_split_duplicates_and_source_binding_fail_closed(self):
        original=deepcopy(self.policy)
        changes=[lambda p:p['approval'].update(status='PENDING'),
                 lambda p:p['approval'].update(evidence=''),
                 lambda p:p['sources'][0].update(y=True),
                 lambda p:p['sources'][0].update(publisher_split='validation'),
                 lambda p:p['sources'][0].update(file_name='../capture.pcap'),
                 lambda p:p['sources'].append(deepcopy(p['sources'][0]))]
        for change in changes:
            self.policy=deepcopy(original);change(self.policy);self.save_policy()
            with self.assertRaises(ValueError):load_policy(self.policy_path,self.policy_sha)
        self.policy=original;self.save_policy()
        with self.assertRaisesRegex(ValueError,'POLICY_SHA256'):load_policy(self.policy_path,'0'*64)
        self.policy['sources'][0]['sha256']='a'*64;self.save_policy()
        with self.assertRaisesRegex(ValueError,'SOURCE_SHA256'):self.produce()
        self.assertFalse((self.work/'producer').exists())

    def test_missing_end_bad_digest_and_extra_bytes_never_publish_manifest(self):
        data,_=self.produce();frames=self.frames(data)
        footer=json.loads(frames[-1][1]);footer['tensor_frames_sha256']='0'*64
        for i,bad in enumerate([self.encode(frames[:-1]),self.encode(frames[:-1]+[(b'E',canonical_bytes(footer))]),data+b'X']):
            with self.subTest(i=i),self.assertRaises(ValueError):self.consume(bad,'bad'+str(i))
            self.assertFalse((self.work/('bad'+str(i))/'manifest.json').exists())
            self.assertTrue((self.work/('bad'+str(i))/'FAILED.json').exists())

    def test_wrong_header_source_label_group_partial_and_manifest_tamper_rejected(self):
        data,_=self.produce();frames=self.frames(data)
        for i,key in enumerate(('y','publisher_split','group_id','partial')):
            edited=list(frames);meta=json.loads(edited[1][1][880:])
            if key=='partial':meta['core']['full_window_complete']=False
            else:meta[key]={'y':1,'publisher_split':'train','group_id':'0'*64}[key]
            edited[1]=(b'T',edited[1][1][:880]+canonical_bytes(meta))
            with self.assertRaises(ValueError):self.consume(self.encode(edited),'tamper'+str(i))
        header=json.loads(frames[0][1]);header['source']['publisher_split']='train'
        with self.assertRaisesRegex(ValueError,'HEADER_BINDING'):
            self.consume(self.encode([(b'H',canonical_bytes(header)),*frames[1:]]),'header')
        self.consume(data)
        p=self.work/'hdf5/manifest.json';m=json.loads(p.read_bytes());m['publisher_split']='train';p.write_bytes(canonical_bytes(m))
        with self.assertRaisesRegex(ValueError,'MANIFEST_SCOPE'):
            ScenarioShardReader(self.work/'hdf5',self.policy_path,self.policy_sha)

    def test_child_nonzero_failure_cannot_complete_pipe(self):
        with self.assertRaisesRegex(ValueError,'NONZERO_EXIT'):
            self.produce(code=1)
        self.assertTrue((self.work/'producer/FAILED.json').exists())

    def test_gzip_transport_preserves_logical_sha_and_exact_tensor_frames(self):
        import gzip
        raw,_=self.produce(name='raw-producer')
        compressed=self.work/'capture.pcap.gz'
        compressed.write_bytes(gzip.compress(self.fixture['source'].read_bytes(),mtime=0))
        f=dict(self.fixture,source=compressed)
        data,_=self.produce(f,name='gzip-producer');m=self.consume(data,'gzip-hdf5')
        self.assertEqual([x for x in self.frames(raw) if x[0]==b'T'],[x for x in self.frames(data) if x[0]==b'T'])
        identity=m['stream_completion']['source_identity']
        self.assertEqual(identity['compression'],'gzip')
        self.assertEqual(identity['logical_sha256'],self.policy['sources'][0]['sha256'])
        self.assertNotEqual(identity['sha256'],identity['logical_sha256'])

    def test_bad_gzip_crc_wrong_logical_hash_and_expansion_fail_before_output(self):
        import gzip
        original=self.fixture['source'].read_bytes()
        changed=bytearray(original);changed[-1]^=1
        cases=[gzip.compress(bytes(changed)),gzip.compress(original+b'X'),gzip.compress(original)[:-3]]
        for i,value in enumerate(cases):
            p=self.work/('bad-gzip'+str(i));p.write_bytes(value)
            with self.subTest(i=i),self.assertRaises((ValueError,OSError,EOFError)):
                self.produce(dict(self.fixture,source=p),name='bad-gzip-producer'+str(i))
            self.assertFalse((self.work/('bad-gzip-producer'+str(i))).exists())

    def test_installed_tshark_reads_raw_and_gzip_with_identical_tensor_frames(self):
        import gzip
        import shutil
        tool=Path('C:/Program Files/Wireshark/tshark.exe') if sys.platform=='win32' else Path(shutil.which('tshark') or '/missing-tshark')
        if not tool.is_file():self.skipTest('Optional real TShark not installed')
        tool_sha=hashlib.sha256(tool.read_bytes()).hexdigest()
        outputs=[]
        compressed=self.work/'actual-tool.pcap.gz'
        compressed.write_bytes(gzip.compress(self.fixture['source'].read_bytes(),mtime=0))
        for i,source in enumerate((self.fixture['source'],compressed)):
            data=io.BytesIO()
            stream_scenario(source,self.work/('actual'+str(i)),data,policy_path=self.policy_path,
                policy_sha=self.policy_sha,source_id='normal-test',tshark=tool,tshark_sha256=tool_sha)
            outputs.append([frame for frame in self.frames(data.getvalue()) if frame[0]==b'T'])
        self.assertEqual(outputs[0],outputs[1])
        self.assertTrue(outputs[0])

    def test_window_quarantine_suppresses_all_flows_and_empty_export_rejected(self):
        original=self.fixture['records']
        self.fixture['records']=[r for r in original[:3]]
        self.policy['sources'][0]['records']=3;self.save_policy()
        r=self.fixture['records'][1]
        from dataclasses import replace
        self.fixture['records'][1]=replace(r,layers=r.layers+('_ws.malformed',),all_protocols=r.all_protocols+('_ws.malformed',))
        data,result=self.produce()
        self.assertEqual(result['tensors'],0)
        with self.assertRaisesRegex(ValueError,'NO_ELIGIBLE'):self.consume(data)

    def test_oversized_protocol_stack_quarantines_only_new_scenario_path(self):
        from dataclasses import replace
        from nids_ml.pcap.decoder import PDMLParser,ScenarioPDMLParser,DecodeLimits
        original=self.fixture['records'][1]
        fields=tuple(replace(f,show='eth:ip:tcp:'+('x509sat:'*72)) if f.name=='frame.protocols' else f for f in original.fields)
        self.fixture['records'][1]=replace(original,fields=fields)
        raw=pdml(self.fixture['records'])
        old=PDMLParser(DecodeLimits())
        with self.assertRaisesRegex(ValueError,'SELECTED_FIELD_SIZE_LIMIT'):
            list(old.feed(raw,final=True))
        parser=ScenarioPDMLParser(DecodeLimits())
        records=list(parser.feed(raw,final=True))
        self.assertEqual(records[1].quarantine_reason,'OVERSIZED_PROTOCOL_STACK')
        data,result=self.produce()
        self.assertEqual(result['tensors'],1)
        m=self.consume(data)
        self.assertEqual(m['stream_completion']['rejection_counts']['OVERSIZED_PROTOCOL_STACK'],1)
        self.assertEqual(m['stream_completion']['stream']['quarantined_windows'],1)

    def test_framing_caps_and_no_overwrite_or_low_disk(self):
        for data in (b'',b'\x01',struct.pack('<I',20000),struct.pack('<I',10)+b'H{}'):
            with self.assertRaises(ValueError):read_frame(io.BytesIO(data))
        data,_=self.produce();self.consume(data)
        before=(self.work/'hdf5/manifest.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'FRESH'):self.consume(data)
        self.assertEqual((self.work/'hdf5/manifest.json').read_bytes(),before)
        with patch('shutil.disk_usage',return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(ValueError,'START_RESERVE'):self.consume(data,'low-disk')
        self.assertFalse((self.work/'low-disk').exists())

    def test_short_pipe_reads_and_core_count_configuration_tampering(self):
        data,_=self.produce()
        class ShortReads(io.BytesIO):
            def read(self,n=-1):return super().read(min(n,2))
        r=export_scenario_pipe(ShortReads(data),self.policy_path,self.policy_sha,'normal-test',self.work/'short',max_output_mib=16)
        self.assertEqual(r['samples'],2)
        original=self.frames(data)
        for i,(key,value) in enumerate([('configuration_sha256','0'*64),('real_packet_count',999),
                                        ('real_packet_mask',[1]*20),('full_packet_digest','bad')]):
            frames=list(original);meta=json.loads(frames[1][1][880:]);meta['core'][key]=value
            frames[1]=(b'T',frames[1][1][:880]+canonical_bytes(meta))
            with self.assertRaises(ValueError):self.consume(self.encode(frames),'core-bad'+str(i))

    def test_actual_cli_inspects_synthetic_output_and_bad_input_returns_failure(self):
        import subprocess
        data,_=self.produce();self.consume(data)
        command=[sys.executable,'-B',str(ROOT/'scripts/run_flows.py'),'scenario-inspect',
                 '--policy',str(self.policy_path),'--policy-sha256',self.policy_sha,'--dataset',str(self.work/'hdf5')]
        r=subprocess.run(command,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertEqual(json.loads(r.stdout)['status'],'PASS_SCENARIO_SHARD_INSPECTION')
        command=[sys.executable,'-B',str(ROOT/'scripts/run_flows.py'),'scenario-export',
                 '--policy',str(self.policy_path),'--policy-sha256','0'*64,'--source-id','normal-test',
                 '--output',str(self.work/'bad-cli')]
        r=subprocess.run(command,input=b'',capture_output=True)
        self.assertEqual(r.returncode,2)
        self.assertEqual(json.loads(r.stdout)['status'],'FAIL_CLOSED')

    def test_host_preflight_uses_no_numpy_hdf5_or_tensorflow(self):
        import subprocess
        values=dict(repo=str(ROOT/'src'),source=str(self.fixture['source']),output=str(self.work/'stdlib'),
            policy=str(self.policy_path),policy_sha=self.policy_sha,tool=str(self.fixture['tool']),tool_sha=self.fixture['tool_sha'])
        code='''
import json,sys
from unittest.mock import patch
from types import SimpleNamespace
v=json.loads(sys.argv[1]);sys.path.insert(0,v['repo'])
from nids_ml.flows.pcap_stream import stream_scenario
with patch('nids_ml.pcap.whole.available_memory_bytes',return_value=10*1024**3), \
     patch('shutil.disk_usage',return_value=SimpleNamespace(free=10*1024**3)):
    r=stream_scenario(v['source'],v['output'],None,policy_path=v['policy'],policy_sha=v['policy_sha'],
        source_id='normal-test',tshark=v['tool'],tshark_sha256=v['tool_sha'],preflight_only=True)
assert r['process_started'] is False
assert not any(name in sys.modules for name in ('numpy','h5py','tensorflow'))
print('STDLIB_HOST_PREFLIGHT_PASS')
'''
        r=subprocess.run([sys.executable,'-B','-c',code,json.dumps(values)],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertIn('STDLIB_HOST_PREFLIGHT_PASS',r.stdout)

    def test_export_budget_is_separate_from_legacy(self):
        ScenarioStreamingLimits().validate()
        with self.assertRaises(ValueError):StreamingLimits(max_heap_packets=32768).validate()
        from dataclasses import replace
        with self.assertRaises(ValueError):replace(ScenarioStreamingLimits(),max_heap_packets=32769).validate()

    def test_audit_detects_cross_source_connection_tensor_and_label_overlap(self):
        other=make_capture(self.work/'other',timestamps=(4000000,2000000,2000000,13000000))
        self.policy['sources'].append(self.rule(other,'attack-train','train',1));self.save_policy()
        a,_=self.produce();self.consume(a,'test-data')
        b,_=self.produce(other,'attack-train','other-producer');self.consume(b,'train-data','attack-train')
        report=audit_scenario_sources([self.work/'test-data',self.work/'train-data'],
            self.policy_path,self.policy_sha,self.work/'audit',sqlite_mib=16)
        self.assertTrue(report['all_policy_sources_present'])
        self.assertGreater(report['cross_official_split_parent_groups'],0)
        self.assertGreater(report['cross_official_split_tensor_patterns'],0)
        self.assertGreater(report['conflicting_label_patterns'],0)
        self.assertFalse(report['training_ready'])


if __name__=='__main__':unittest.main()
