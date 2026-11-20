"""Validate the explicit Connect train budget with synthetic sources and children.
Kiểm profile Connect train bằng fixtures giả; không phải benchmark 17 triệu packet.
"""
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tests'))
import test_scenario_export as fixtures
from decoded_pcap_fixtures import FakeDecoderProcess,pdml
from nids_ml.pcap.decoder import DecoderError
from nids_ml.pcap.whole import StreamingDecodeLimits,StreamingLargeMemoryLimits,StreamingConnectTrainLimits
from nids_ml.flows.pcap_stream import ScenarioConnectTrainDecoder,stream_scenario
from nids_ml.labeling.scenario import ScenarioStreamingLimits,ConnectTrainStreamingLimits
from nids_ml.storage.scenario import read_frame


class ConnectTrainTests(unittest.TestCase):
    def setUp(self):
        self.case=fixtures.ScenarioTests('runTest');self.case.setUp();self.addCleanup(self.case.doCleanups)
        rule=self.case.policy['sources'][0]
        rule.update(source_id='connect-train',file_name='connect-train.pcap',publisher_split='train',
                    y=1,attack_type='MQTT_DDoS_Connect_Flood')
        self.case.save_policy();self.f=self.case.fixture

    def decoder(self):
        return ScenarioConnectTrainDecoder(self.f['source'],expected_records=4,
            expected_sha256=self.f['descriptor']['capture']['capture_sha256'],expected_size=self.f['source'].stat().st_size,
            tshark=self.f['tool'],tshark_sha256=self.f['tool_sha'])

    def produce(self,name='producer',profile='connect-train',code=0):
        output=io.BytesIO();child=FakeDecoderProcess(pdml(self.f['records']),code)
        with patch('nids_ml.pcap.whole.available_memory_bytes',return_value=16*1024**3), \
             patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=child):
            result=stream_scenario(self.f['source'],self.case.work/name,output,policy_path=self.case.policy_path,
                policy_sha=self.case.policy_sha,source_id='connect-train',tshark=self.f['tool'],
                tshark_sha256=self.f['tool_sha'],decoder_profile=profile)
        return output.getvalue(),result

    def test_fixed_10gib_18h_and_legacy_budgets_unchanged(self):
        new=StreamingConnectTrainLimits();new.validate()
        self.assertEqual(new.decoder_rss_bytes,10*1024**3);self.assertEqual(new.timeout_seconds,64800.)
        self.assertEqual(new.record_bytes,1048576);self.assertEqual(new.stdout_bytes,256*1024**3)
        for option in (dict(decoder_rss_bytes=11*1024**3),dict(timeout_seconds=64801.),dict(packet_count=True)):
            with self.assertRaises(DecoderError):replace(new,**option).validate()
        self.assertEqual(StreamingDecodeLimits().timeout_seconds,43200.)
        self.assertEqual(StreamingLargeMemoryLimits().decoder_rss_bytes,6*1024**3)

    def test_spool_128mib_18h_and_invalid_override_rejected(self):
        limits=ConnectTrainStreamingLimits();limits.validate()
        self.assertEqual(limits.sqlite_bytes,128*1048576);self.assertEqual(limits.timeout_seconds,64800)
        self.assertEqual(limits.min_free_bytes,1024**3)
        for options in (dict(sqlite_bytes=256*1048576),dict(timeout_seconds=64800.),dict(timeout_seconds=43200)):
            with self.assertRaises(ValueError):replace(limits,**options).validate()
        self.assertEqual(ScenarioStreamingLimits().sqlite_bytes,256*1048576)
        self.assertEqual(ScenarioStreamingLimits().timeout_seconds,43200)

    def test_start_below_14gib_or_missing_linux_probe_refused_before_child(self):
        for amount in (8*1024**3,14*1024**3-1):
            with patch('nids_ml.pcap.whole.available_memory_bytes',return_value=amount), \
                 patch('nids_ml.pcap.decoder.subprocess.Popen') as start:
                with self.assertRaisesRegex(DecoderError,'14_GIB'):self.decoder()
                start.assert_not_called()
        with patch('nids_ml.pcap.whole.available_memory_bytes',side_effect=DecoderError('LARGE_MEMORY_PROFILE_REQUIRES_LINUX_MEMORY_PROBE')):
            with self.assertRaisesRegex(DecoderError,'LINUX_MEMORY_PROBE'):self.decoder()

    def test_runtime_host_below_2gib_and_missing_rss_fail_closed(self):
        with patch('nids_ml.pcap.whole.available_memory_bytes',return_value=16*1024**3):decoder=self.decoder()
        decoder._last_host_memory_check=None
        with patch('nids_ml.pcap.whole.available_memory_bytes',return_value=1536*1048576):
            with self.assertRaisesRegex(DecoderError,'BELOW_2_GIB'):decoder._host_memory_guard()
        decoder.stats['memory_probe_available']=False
        with self.assertRaisesRegex(DecoderError,'RSS_PROBE_UNAVAILABLE'):decoder._host_memory_guard()

    def test_decoder_rss_and_deadline_guards_remain_active(self):
        with patch('nids_ml.pcap.whole.available_memory_bytes',return_value=16*1024**3):decoder=self.decoder()
        decoder.started=time.monotonic();decoder.process=FakeDecoderProcess(b'')
        with patch('nids_ml.pcap.decoder.process_rss',side_effect=lambda pid:1024 if pid==os.getpid() else 10*1024**3+1):
            with self.assertRaisesRegex(DecoderError,'DECODER_RSS_ABORT'):decoder._resources()
        decoder.started=time.monotonic()-64801
        with self.assertRaisesRegex(DecoderError,'DECODER_TIMEOUT'):decoder._resources()
        decoder.close()

    def test_full_synthetic_pipe_hdf5_and_profile_report(self):
        data,result=self.produce();self.assertTrue(result['source_complete'])
        m=self.case.consume(data,source_id='connect-train')
        policy=m['stream_completion']['decoder']['memory_policy']
        self.assertEqual(policy['profile'],'connect-train')
        self.assertEqual(policy['minimum_start_available_bytes'],14*1024**3)
        self.assertEqual(policy['minimum_runtime_available_bytes'],2*1024**3)
        self.assertEqual(m['publisher_split'],'train');self.assertEqual(m['label_counts'],{'1':2})
        self.assertFalse(m['training_ready'])

    def test_resource_profile_does_not_change_tensor_frames(self):
        outputs=[self.produce('producer-'+profile,profile)[0] for profile in ('standard','connect-train')]
        frames=[]
        for data in outputs:
            stream=io.BytesIO(data);selected=[]
            while stream.tell()<len(data):
                kind,payload=read_frame(stream)
                if kind==b'T':selected.append(payload)
            frames.append(selected)
        self.assertTrue(frames[0]);self.assertEqual(frames[0],frames[1])

    def test_other_sources_cannot_use_connect_train_profile(self):
        self.case.policy['sources'][0].update(publisher_split='test');self.case.save_policy()
        with self.assertRaisesRegex(ValueError,'CONNECT_TRAIN_PROFILE_SCOPE'):self.produce()
        self.assertFalse((self.case.work/'producer').exists())

    def test_nonzero_child_cannot_publish_complete_stream(self):
        with self.assertRaisesRegex(ValueError,'NONZERO_EXIT'):self.produce(code=1)
        self.assertTrue((self.case.work/'producer/FAILED.json').exists())


if __name__=='__main__':unittest.main()
