"""Check opt-in decoder RAM budgets with synthetic packets and memory probes.
Kiểm profile RAM bằng fixtures nhỏ; không đoán kết quả decode PCAP thật.
"""
from contextlib import redirect_stdout
from collections import Counter
from dataclasses import fields,replace
import io
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import mock_open,patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from decoded_pcap_fixtures import make_capture,pdml,FakeDecoderProcess
from nids_ml.pcap.decoder import DecoderError,file_sha256,PREFERENCES
from nids_ml.pcap.whole import (LargeFlowDecoder,LargeFlowDecodeLimits,
    LargeMemoryFlowDecoder,LargeMemoryFlowDecodeLimits,available_memory_bytes)
from nids_ml.flows.mapped import prepare_cache
from nids_ml.cli.flow_shards import main

GIB=1024**3


class HostMemoryProbeTests(unittest.TestCase):
    def test_linux_memavailable_units(self):
        with patch('nids_ml.pcap.whole.sys_platform_linux',return_value=True), \
             patch('builtins.open',mock_open(read_data='MemTotal: 16777216 kB\nMemAvailable: 8388608 kB\n')):
            self.assertEqual(available_memory_bytes(),8*GIB)

    def test_unavailable_malformed_or_duplicate_probes_fail_closed(self):
        with patch('nids_ml.pcap.whole.sys_platform_linux',return_value=False):
            with self.assertRaisesRegex(DecoderError,'REQUIRES_LINUX'):
                available_memory_bytes()
        for data in ('MemTotal: 123 kB\n','MemAvailable: nan kB\n','MemAvailable: -1 kB\n',
                     'MemAvailable: 100 MB\n','MemAvailable: 1 kB\nMemAvailable: 2 kB\n','x'*65537):
            with self.subTest(data=data[:30]),patch('nids_ml.pcap.whole.sys_platform_linux',return_value=True), \
                 patch('builtins.open',mock_open(read_data=data)):
                with self.assertRaisesRegex(DecoderError,'HOST_PROBE_UNAVAILABLE'):
                    available_memory_bytes()

    def test_missing_meminfo_is_not_treated_as_free_memory(self):
        with patch('nids_ml.pcap.whole.sys_platform_linux',return_value=True), \
             patch('builtins.open',side_effect=FileNotFoundError):
            with self.assertRaisesRegex(DecoderError,'HOST_PROBE_UNAVAILABLE'):
                available_memory_bytes()


class LargeMemoryDecoderTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='large-memory-synthetic-',dir=ROOT/'artifacts/runs')
        self.disk=patch('shutil.disk_usage',return_value=SimpleNamespace(free=10*GIB));self.disk.start()
        self.fixture=make_capture(Path(self.tmp.name))
        self.memory=patch('nids_ml.pcap.whole.available_memory_bytes',return_value=16*GIB);self.memory.start()
        self.rss=patch('nids_ml.pcap.decoder.process_rss',return_value=1024);self.rss.start()

    def tearDown(self):
        self.rss.stop();self.memory.stop();self.disk.stop();self.tmp.cleanup()

    def decoder(self,kind=LargeMemoryFlowDecoder):
        f=self.fixture
        return kind(f['source'],expected_records=4,expected_sha256=file_sha256(f['source']),
                    expected_size=f['source'].stat().st_size,tshark=f['tool'],tshark_sha256=f['tool_sha'])

    def run_decoder(self,decoder,raw=None):
        child=FakeDecoderProcess(pdml(self.fixture['records']) if raw is None else raw)
        with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=child):
            records=list(decoder.records())
        return records,child

    def test_only_fixed_child_rss_differs_from_standard_limits(self):
        standard=LargeFlowDecodeLimits();large=LargeMemoryFlowDecodeLimits()
        standard.validate();large.validate()
        self.assertEqual(standard.decoder_rss_bytes,1536*1048576)
        self.assertEqual(large.decoder_rss_bytes,6*GIB)
        for field in fields(standard):
            if field.name!='decoder_rss_bytes':
                self.assertEqual(getattr(standard,field.name),getattr(large,field.name))
        for limits in (replace(large,decoder_rss_bytes=8*GIB),replace(large,timeout_seconds=14400.),
                       replace(large,python_rss_bytes=GIB),replace(large,packet_count=True)):
            with self.assertRaises(DecoderError):limits.validate()

    def test_low_start_memory_stops_before_child_and_source_checks(self):
        with patch('nids_ml.pcap.whole.available_memory_bytes',return_value=8*GIB-1), \
             patch('nids_ml.pcap.decoder.subprocess.Popen') as start:
            with self.assertRaisesRegex(DecoderError,'REQUIRES_8_GIB'):
                self.decoder()
            start.assert_not_called()

    def test_default_decoder_does_not_require_large_memory_probe(self):
        with patch('nids_ml.pcap.whole.available_memory_bytes',side_effect=AssertionError('No opt-in')):
            records,_=self.run_decoder(self.decoder(LargeFlowDecoder))
        self.assertEqual(len(records),4)

    def test_two_profiles_keep_records_commands_tcp_state_and_eof_identical(self):
        standard=self.decoder(LargeFlowDecoder);large=self.decoder()
        a,_=self.run_decoder(standard);b,_=self.run_decoder(large)
        self.assertEqual(a,b);self.assertEqual(standard.command,large.command)
        self.assertNotIn('-M',large.command);self.assertNotIn('-Y',large.command)
        self.assertTrue(large.stats['source_complete'])
        self.assertEqual(large.report()['preferences'],PREFERENCES)
        self.assertEqual(large.report()['memory_policy']['profile'],'large-memory')

    def test_large_profile_accepts_child_rss_above_old_cap(self):
        decoder=self.decoder()
        with patch('nids_ml.pcap.decoder.process_rss',side_effect=lambda pid: 1024 if pid==os.getpid() else 2*GIB):
            records,_=self.run_decoder(decoder)
        self.assertEqual(len(records),4)
        self.assertEqual(decoder.stats['peak_decoder_rss_bytes'],2*GIB)

    def test_child_rss_above_new_cap_stops_only_owned_child(self):
        decoder=self.decoder();child=FakeDecoderProcess(pdml(self.fixture['records']))
        with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=child), \
             patch('nids_ml.pcap.decoder.process_rss',side_effect=lambda pid: 1024 if pid==os.getpid() else 6*GIB+1):
            with self.assertRaisesRegex(DecoderError,'DECODER_RSS_ABORT_THRESHOLD'):
                list(decoder.records())
        self.assertTrue(child.terminated);self.assertFalse(decoder.stats['source_complete'])

    def test_low_runtime_host_memory_stops_owned_child(self):
        decoder=self.decoder();child=FakeDecoderProcess(pdml(self.fixture['records']))
        with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=child), \
             patch('nids_ml.pcap.whole.available_memory_bytes',return_value=GIB-1):
            with self.assertRaisesRegex(DecoderError,'BELOW_1_GIB_RESERVE'):
                list(decoder.records())
        self.assertTrue(child.terminated)

    def test_missing_rss_probe_stops_large_profile(self):
        decoder=self.decoder();child=FakeDecoderProcess(pdml(self.fixture['records']))
        with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=child), \
             patch('nids_ml.pcap.decoder.process_rss',return_value=None):
            with self.assertRaisesRegex(DecoderError,'RSS_PROBE_UNAVAILABLE'):
                list(decoder.records())
        self.assertTrue(decoder.report()['child_exited'])

    def exit_during_rss_probe(self,*,code=0,raw=None):
        decoder=self.decoder()
        child=FakeDecoderProcess(pdml(self.fixture['records']) if raw is None else raw,exit_code=code)
        def rss(pid):
            if pid==os.getpid():return 1024
            child.returncode=code
            return None
        return decoder,child,rss

    def test_child_exit_between_poll_and_rss_does_not_fail_complete_capture(self):
        decoder,child,rss=self.exit_during_rss_probe()
        with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=child), \
             patch('nids_ml.pcap.decoder.process_rss',side_effect=rss):
            records=list(decoder.records())
        self.assertEqual(len(records),4)
        self.assertTrue(decoder.stats['source_complete'])
        self.assertTrue(decoder.stats['memory_probe_available'])
        self.assertFalse(child.terminated)

    def test_exiting_child_missing_rss_never_masks_nonzero_exit(self):
        decoder,child,rss=self.exit_during_rss_probe(code=1)
        with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=child), \
             patch('nids_ml.pcap.decoder.process_rss',side_effect=rss):
            with self.assertRaisesRegex(DecoderError,'TSHARK_NONZERO_EXIT'):
                list(decoder.records())
        self.assertFalse(decoder.stats['source_complete'])

    def test_exiting_child_missing_rss_never_masks_truncated_xml(self):
        decoder,child,rss=self.exit_during_rss_probe(raw=pdml(self.fixture['records'],close=False))
        with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=child), \
             patch('nids_ml.pcap.decoder.process_rss',side_effect=rss):
            with self.assertRaises(DecoderError) as error:
                list(decoder.records())
        self.assertNotIn('RSS_PROBE_UNAVAILABLE',str(error.exception))
        self.assertFalse(decoder.stats['source_complete'])

    def test_still_live_child_missing_rss_remains_fail_closed(self):
        decoder=self.decoder();child=FakeDecoderProcess(pdml(self.fixture['records']))
        original_wait=child.wait
        def wait(timeout=None):
            if child.returncode is None:
                raise subprocess.TimeoutExpired('SYNTHETIC_LIVE_CHILD',timeout)
            return original_wait(timeout)
        with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=child), \
             patch.object(child,'wait',side_effect=wait), \
             patch('nids_ml.pcap.decoder.process_rss',side_effect=lambda pid:1024 if pid==os.getpid() else None):
            with self.assertRaisesRegex(DecoderError,'RSS_PROBE_UNAVAILABLE'):
                list(decoder.records())
        self.assertTrue(child.terminated);self.assertFalse(decoder.stats['source_complete'])

    def test_live_python_missing_rss_is_not_ignored_after_child_exit(self):
        decoder=self.decoder();child=FakeDecoderProcess(pdml(self.fixture['records']))
        child.returncode=0
        with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=child), \
             patch('nids_ml.pcap.decoder.process_rss',return_value=None):
            with self.assertRaisesRegex(DecoderError,'RSS_PROBE_UNAVAILABLE'):
                list(decoder.records())
        self.assertFalse(decoder.stats['source_complete'])

    def test_rss_unavailable_while_child_finishes_wait_is_bounded(self):
        decoder,child,_=self.exit_during_rss_probe()
        def wait(timeout=None):
            if child.returncode is None:
                self.assertIsNotNone(timeout);self.assertLessEqual(timeout,.05)
            child.returncode=0
            return 0
        with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=child), \
             patch.object(child,'wait',side_effect=wait) as waiter, \
             patch('nids_ml.pcap.decoder.process_rss',side_effect=lambda pid:1024 if pid==os.getpid() else None):
            records=list(decoder.records())
        self.assertEqual(len(records),4);self.assertTrue(decoder.stats['source_complete'])
        self.assertGreaterEqual(waiter.call_count,1)
        self.assertLessEqual(waiter.call_args_list[0].kwargs['timeout'],.05)

    def test_real_child_exit_still_drains_buffered_pdml_and_validates_eof(self):
        decoder=self.decoder();raw=pdml(self.fixture['records'])
        original_popen=subprocess.Popen;children=[]
        def start(command,**kwargs):
            child=original_popen([sys.executable,'-X','utf8','-B','-c',
                'import sys;sys.stdout.buffer.write('+repr(raw)+')'],**kwargs)
            children.append(child)
            return child
        def rss(pid):
            if pid==os.getpid():return 1024
            children[0].wait(timeout=10)
            return None
        with patch('nids_ml.pcap.decoder.subprocess.Popen',side_effect=start), \
             patch('nids_ml.pcap.decoder.process_rss',side_effect=rss):
            records=list(decoder.records())
        self.assertEqual(len(children),1);self.assertEqual(children[0].returncode,0)
        self.assertEqual(len(records),4);self.assertTrue(decoder.stats['source_complete'])
        self.assertTrue(decoder.stats['memory_probe_available'])
        self.assertEqual(decoder.report()['child_returncode'],0)

    def test_host_reserve_is_sampled_once_per_second(self):
        decoder=self.decoder();decoder.started=100.;decoder.process=FakeDecoderProcess(b'')
        with patch('nids_ml.pcap.whole.available_memory_bytes',return_value=9*GIB) as probe, \
             patch('nids_ml.pcap.whole.time.monotonic',return_value=100.):
            decoder._resources();decoder._resources()
            self.assertEqual(probe.call_count,1)
        with patch('nids_ml.pcap.whole.available_memory_bytes',return_value=9*GIB) as probe, \
             patch('nids_ml.pcap.whole.time.monotonic',return_value=101.):
            decoder._resources();probe.assert_called_once()
        decoder.close()

    def test_large_profile_still_rejects_missing_eof(self):
        decoder=self.decoder()
        with self.assertRaises(DecoderError):
            self.run_decoder(decoder,pdml(self.fixture['records'][:-1]))
        self.assertFalse(decoder.stats['source_complete'])

    def test_cli_selects_profile_and_defaults_to_standard(self):
        for supplied,expected in (([], 'standard'),(['--decoder-profile','large-memory'],'large-memory')):
            with patch('nids_ml.flows.mapped.prepare_cache',return_value={'status':'MOCK_ONLY'}) as run, \
                 redirect_stdout(io.StringIO()):
                code=main(['decode','--mapping','m','--tshark','t','--tshark-sha256','a'*64,
                           '--output','o',*supplied])
            self.assertEqual(code,0);self.assertEqual(run.call_args.kwargs['decoder_profile'],expected)

    def test_invalid_profile_and_low_start_memory_create_no_cache(self):
        output=Path(self.tmp.name)/'artifacts/runs/new'
        with self.assertRaisesRegex(ValueError,'UNAPPROVED_FLOW_DECODER_PROFILE'):
            prepare_cache('unused',Path(self.tmp.name),output,tshark='unused',tshark_sha256='a'*64,
                          decoder_profile='unbounded')
        with patch('nids_ml.pcap.whole.available_memory_bytes',return_value=GIB):
            with self.assertRaisesRegex(DecoderError,'REQUIRES_8_GIB'):
                prepare_cache('unused',Path(self.tmp.name),output,tshark='unused',tshark_sha256='a'*64,
                              decoder_profile='large-memory')
        self.assertFalse(output.exists())

    def test_mapped_wrapper_records_profile_and_preserves_tensor_semantics(self):
        from test_flow_shards import FlowShardTests,FixtureDecoder
        from nids_ml.flows.mapped import iter_tensors
        fixture=FlowShardTests();fixture.setUp()
        try:
            mapping=fixture.mapping()
            old,_=fixture.cache(mapping,name='standard')
            new=fixture.work/'artifacts/runs/large'
            with patch('nids_ml.flows.mapped.LargeMemoryFlowDecoder',FixtureDecoder):
                result=prepare_cache(mapping,fixture.work,new,tshark=fixture.tool,
                                     tshark_sha256=file_sha256(fixture.tool),decoder_profile='large-memory')
            self.assertEqual(result['decoder_profile'],'large-memory')
            a=list(iter_tensors(old,Counter()));b=list(iter_tensors(new,Counter()))
            self.assertEqual(a,b)
        finally:
            fixture.tearDown()

    def test_mapping_failure_retains_profile_and_decoder_evidence(self):
        from test_flow_shards import FlowShardTests,FixtureDecoder
        fixture=FlowShardTests();fixture.setUp()
        try:
            mapping=fixture.mapping();output=fixture.work/'artifacts/runs/failed'
            with patch('nids_ml.flows.mapped.LargeMemoryFlowDecoder',FixtureDecoder), \
                 patch.object(FixtureDecoder,'records',side_effect=DecoderError('DECODER_RSS_ABORT_THRESHOLD')):
                with self.assertRaisesRegex(DecoderError,'DECODER_RSS_ABORT_THRESHOLD'):
                    prepare_cache(mapping,fixture.work,output,tshark=fixture.tool,
                                  tshark_sha256=file_sha256(fixture.tool),decoder_profile='large-memory')
            failed=json.loads((output/'FAILED.json').read_text())
            self.assertEqual(failed['decoder_profile'],'large-memory')
            self.assertEqual(failed['decoder']['status'],'SYNTHETIC_FIXTURE')
            self.assertFalse((output/'manifest.json').exists())
        finally:
            fixture.tearDown()


if __name__=='__main__':unittest.main(verbosity=2)
