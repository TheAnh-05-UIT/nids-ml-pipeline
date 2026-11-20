"""Validate decoder-to-window integration with small synthetic PCAP/PDML fixtures.
Kiểm tích hợp decoder/window bằng fixture nhỏ; không dùng nhãn dataset làm giả ground truth.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tests'))
from decoded_pcap_fixtures import make_capture,pdml,FakeDecoderProcess
from flow_extraction_fixtures import context,packet
from nids_ml.flows.features import CaptureContext,extract_features
from nids_ml.flows.streaming import SourceRecord,WindowSpool
from nids_ml.flows.pcap_stream import diagnose_stream
from nids_ml.pcap.decoder import DecoderError,DecodeLimits
from nids_ml.pcap.whole import (StreamingCaptureDecoder,StreamingLargeMemoryDecoder,
    StreamingDecodeLimits,StreamingLargeMemoryLimits,LargeFlowDecodeLimits)


class PcapStreamTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work=Path(self.tmp.name)
        for target,value in [('shutil.disk_usage',SimpleNamespace(free=10*1024**3)),
                             ('nids_ml.pcap.decoder.process_rss',1024),
                             ('nids_ml.pcap.whole.available_memory_bytes',10*1024**3)]:
            mocked=patch(target,return_value=value)
            mocked.start();self.addCleanup(mocked.stop)
        self.fixture=make_capture(self.work)

    def options(self):
        f=self.fixture
        return dict(expected_sha256=f['descriptor']['capture']['capture_sha256'],
                    expected_size=f['source'].stat().st_size,expected_records=4,
                    tshark=f['tool'],tshark_sha256=f['tool_sha'],max_lateness_ns=2*10**9,
                    source_split='test',prefix_records=None)

    def run_diagnostic(self,records=None,code=0,name='diagnostic',options=None,close=True):
        options=self.options() if options is None else options
        child=FakeDecoderProcess(pdml(self.fixture['records'] if records is None else records,close=close),code)
        with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=child) as start:
            result=diagnose_stream(self.fixture['source'],self.work/name,**options)
        return result,child,start

    def test_full_source_accounting_eof_and_contiguous_single_child(self):
        result,child,start=self.run_diagnostic()
        self.assertEqual(result['status'],'PASS_WHOLE_STREAM_DIAGNOSTIC')
        self.assertEqual(result['stream']['input_packets'],4)
        self.assertEqual(result['stream']['feature_packets'],4)
        self.assertTrue(result['decoder']['stats']['source_complete'])
        self.assertEqual(result['decoder']['capture_scope']['expected_source_records'],4)
        self.assertEqual(result['tensor_count'],2)
        self.assertFalse(result['training_ready'])
        self.assertFalse(result['hdf5_written'])
        command=start.call_args.args[0]
        self.assertEqual(command[command.index('-c')+1],'5')
        self.assertNotIn('-M',command);self.assertNotIn('-Y',command)
        self.assertIn('tcp.analyze_sequence_numbers:TRUE',command)
        self.assertEqual(start.call_count,1)
        self.assertFalse(child.terminated)
        self.assertFalse((self.work/'diagnostic/FAILED.json').exists())

    def test_prefix_is_partial_never_eof_or_training_ready(self):
        options=self.options();options['prefix_records']=2
        result,_,start=self.run_diagnostic(self.fixture['records'][:2],options=options)
        self.assertEqual(result['status'],'PASS_PREFIX_STREAM_DIAGNOSTIC')
        self.assertEqual(result['stream']['status'],'PARTIAL')
        self.assertFalse(result['decoder']['stats']['source_complete'])
        self.assertEqual(result['partial_window_tensors'],1)
        command=start.call_args.args[0]
        self.assertEqual(command[command.index('-c')+1],'2')
        self.assertEqual(result['source']['publisher_split_declared'],'test')
        self.assertFalse(result['source']['split_reassigned'])

    def test_outside_record_sets_global_anchor(self):
        f=make_capture(self.work/'outside',timestamps=(0,9000000,11000000),rejected=(1,))
        self.fixture=f
        options=self.options();options.update(expected_records=3,max_lateness_ns=0)
        result,_,_=self.run_diagnostic(options=options)
        self.assertEqual(result['stream']['outside_packets'],1)
        self.assertEqual(result['stream']['windows'],2)
        self.assertEqual(result['tensor_count'],2)

    def test_malformed_outside_and_missing_transport_quarantine_entire_window(self):
        original=self.fixture['records']
        cases=[replace(original[1],layers=original[1].layers+('_ws.malformed',),
                       all_protocols=original[1].all_protocols+('_ws.malformed',)),
               replace(original[1],fields=tuple(f for f in original[1].fields if f.name!='tcp.hdr_len')),
               replace(original[1],layers=('geninfo','frame','eth','arp','_ws.malformed'),
                       all_protocols=('geninfo','frame','eth','arp','_ws.malformed'))]
        for i,bad in enumerate(cases):
            with self.subTest(i=i):
                result,_,_=self.run_diagnostic([original[0],bad,*original[2:]],name='quarantine'+str(i))
                self.assertEqual(result['stream']['quarantined_packets'],1)
                self.assertEqual(result['stream']['quarantined_windows'],1)
                self.assertEqual(result['stream']['quarantined_flow_windows'],1)
                self.assertEqual(result['tensor_count'],1)

    def test_truncated_outside_capture_quarantined(self):
        original=self.fixture['records']
        bad=replace(original[1],fields=tuple(replace(f,show='1') if f.name=='frame.cap_len' else f for f in original[1].fields))
        result,_,_=self.run_diagnostic([original[0],bad,*original[2:]])
        self.assertEqual(result['rejection_counts']['TRUNCATED_CAPTURE'],1)
        self.assertEqual(result['stream']['quarantined_windows'],1)

    def test_missing_extra_nonzero_malformed_xml_do_not_publish_success(self):
        records=self.fixture['records']
        for i,(data,code,closed,count) in enumerate([
                (records[:-1],0,True,4),(records,0,True,3),(records,1,True,4),(records,0,False,4)]):
            with self.subTest(i=i):
                options=self.options();options['expected_records']=count
                with self.assertRaises(DecoderError):
                    self.run_diagnostic(data,code,name='failed'+str(i),options=options,close=closed)
                directory=self.work/('failed'+str(i))
                self.assertTrue((directory/'FAILED.json').is_file())
                self.assertFalse((directory/'report.json').exists())

    def test_wrong_hash_size_profile_split_rejected_before_child(self):
        for i,(key,value) in enumerate([('expected_sha256','0'*64),('expected_size',1),
                                       ('source_split','validation'),('decoder_profile','fast')]):
            options=self.options();options[key]=value
            with patch('nids_ml.pcap.decoder.subprocess.Popen') as start:
                with self.assertRaises((ValueError,TypeError)):
                    diagnose_stream(self.fixture['source'],self.work/('bad'+str(i)),**options)
                start.assert_not_called()
            self.assertFalse((self.work/('bad'+str(i))).exists())

    def test_wrong_prefix_budget_and_old_decoder_profiles_remain_restricted(self):
        options=self.options()
        for value in (True,0,4,100001):
            with self.assertRaises(DecoderError):
                StreamingCaptureDecoder(self.fixture['source'],prefix_records=value,
                    **{k:v for k,v in options.items() if k in (
                        'expected_sha256','expected_size','expected_records','tshark','tshark_sha256')})
        StreamingDecodeLimits(packet_count=30000001).validate()
        StreamingLargeMemoryLimits(packet_count=30000001).validate()
        with self.assertRaises(DecoderError):LargeFlowDecodeLimits(packet_count=3500002).validate()
        with self.assertRaises(DecoderError):DecodeLimits(packet_count=4097).validate()
        with self.assertRaises(DecoderError):replace(StreamingDecodeLimits(),stdout_bytes=2**50).validate()

    def test_missing_rss_live_child_fails_closed_and_stops_owned_child(self):
        with patch('nids_ml.pcap.decoder.process_rss',return_value=None), \
                patch.object(StreamingCaptureDecoder,'rss_exit_grace_seconds',0.):
            child=FakeDecoderProcess(pdml(self.fixture['records']))
            with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=child):
                with self.assertRaisesRegex(DecoderError,'RSS_PROBE_UNAVAILABLE'):
                    diagnose_stream(self.fixture['source'],self.work/'rss',**self.options())
            self.assertTrue(child.terminated)
            self.assertTrue((self.work/'rss/FAILED.json').exists())

    def test_large_memory_mode_requires_linux_memory_preflight(self):
        options=self.options();options['decoder_profile']='large-memory'
        with patch('nids_ml.pcap.whole.require_large_memory_start',side_effect=DecoderError('NEED_8_GIB')):
            with self.assertRaisesRegex(DecoderError,'NEED_8_GIB'):
                diagnose_stream(self.fixture['source'],self.work/'large',**options)
        self.assertFalse((self.work/'large').exists())

    def test_standard_linux_host_memory_start_and_runtime_guards(self):
        with patch('nids_ml.pcap.whole.sys_platform_linux',return_value=True), \
                patch('nids_ml.pcap.whole.available_memory_bytes',return_value=2*1024**3):
            with self.assertRaisesRegex(DecoderError,'3_GIB_AVAILABLE'):
                diagnose_stream(self.fixture['source'],self.work/'low-start',**self.options())
        with patch('nids_ml.pcap.whole.sys_platform_linux',return_value=True), \
                patch('nids_ml.pcap.whole.available_memory_bytes',side_effect=[4*1024**3,512*1048576]):
            with self.assertRaisesRegex(DecoderError,'BELOW_1_GIB'):
                self.run_diagnostic(name='low-runtime')
        self.assertFalse((self.work/'low-runtime/report.json').exists())

    def test_no_overwrite_determinism_and_lateness_failure(self):
        first,_,_=self.run_diagnostic(name='first')
        second,_,_=self.run_diagnostic(name='second')
        self.assertEqual(first['tensor_stream_sha256'],second['tensor_stream_sha256'])
        before=(self.work/'first/report.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'FRESH'):
            self.run_diagnostic(name='first')
        self.assertEqual((self.work/'first/report.json').read_bytes(),before)
        options=self.options();options['max_lateness_ns']=0
        with self.assertRaisesRegex(ValueError,'LATENESS_EXCEEDED'):
            self.run_diagnostic(name='late',options=options)
        self.assertFalse((self.work/'late/report.json').exists())

    def test_large_memory_profile_keeps_host_guard_and_new_capture_scope(self):
        options=self.options();options['decoder_profile']='large-memory'
        with patch('nids_ml.pcap.whole.require_large_memory_start',return_value=10*1024**3), \
                patch('nids_ml.pcap.whole.available_memory_bytes',return_value=9*1024**3):
            result,_,_=self.run_diagnostic(options=options)
        self.assertEqual(result['decoder']['limits']['decoder_rss_bytes'],6*1024**3)
        self.assertEqual(result['decoder']['memory_policy']['profile'],'large-memory')
        self.assertTrue(result['decoder']['capture_scope']['eof_verified'])

    def test_prefix_rehash_detects_mutation_with_preserved_size_and_mtime(self):
        import os
        options=self.options()
        decoder=StreamingCaptureDecoder(self.fixture['source'],prefix_records=2,
            **{k:v for k,v in options.items() if k in (
                'expected_sha256','expected_size','expected_records','tshark','tshark_sha256')})
        child=FakeDecoderProcess(pdml(self.fixture['records'][:2]))
        with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=child):
            events=decoder.records();next(events)
            path=self.fixture['source'];before=path.stat();data=bytearray(path.read_bytes())
            data[-1]^=1;path.write_bytes(data);os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
            with self.assertRaisesRegex(DecoderError,'STREAM_SOURCE_CHANGED'):
                list(events)
        self.assertFalse(decoder.stats['source_complete'])

    def test_preflight_creates_no_output_or_child_and_cli_failure_exit(self):
        options=self.options();options['preflight_only']=True
        with patch('nids_ml.pcap.decoder.subprocess.Popen') as start:
            result=diagnose_stream(self.fixture['source'],self.work/'preflight',**options)
            start.assert_not_called()
        self.assertEqual(result['status'],'PASS_READONLY_STREAM_PREFLIGHT')
        self.assertFalse((self.work/'preflight').exists())
        import subprocess
        command=[sys.executable,'-B',str(ROOT/'scripts/run_flows.py'),'diagnose-stream',
            '--pcap',str(self.fixture['source']),'--sha256','0'*64,'--size-bytes',str(self.fixture['source'].stat().st_size),
            '--expected-records','4','--tshark',str(self.fixture['tool']),'--tshark-sha256',self.fixture['tool_sha'],
            '--max-lateness-ns','0','--source-split','test','--output',str(self.work/'cli-bad')]
        result=subprocess.run(command,capture_output=True,text=True)
        self.assertEqual(result.returncode,2)
        self.assertEqual(json.loads(result.stdout)['status'],'FAIL_CLOSED')


class SourceRecordTests(unittest.TestCase):
    def setUp(self):
        disk=patch('shutil.disk_usage',return_value=SimpleNamespace(free=10*1024**3))
        disk.start();self.addCleanup(disk.stop)

    def test_quarantine_after_n20_still_excludes_flow_and_outside_updates_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx=CaptureContext.from_dict(context())
            records=[SourceRecord(1,0,'OUTSIDE')]
            for i in range(2,24):
                p=extract_features(packet(ordinal=i,timestamp_ns=i))
                records.append(SourceRecord(i,i,'FEATURE',p))
            records.append(SourceRecord(24,24,'QUARANTINE'))
            engine=WindowSpool(ctx,Path(tmp)/'spool',max_lateness_ns=0,input_kind='synthetic_fixture')
            self.assertEqual(list(engine.run_records(records,input_complete=True)),[])
            self.assertEqual(engine.stats['quarantined_flow_windows'],1)
            self.assertEqual(engine.stats['input_packets'],24)

    def test_record_gaps_mismatched_feature_and_resource_callback_rejected(self):
        ctx=CaptureContext.from_dict(context());p=extract_features(packet())
        for i,event in enumerate([SourceRecord(2,0,'OUTSIDE'),SourceRecord(1,1,'FEATURE',p),
                                  SourceRecord(1,0,'OUTSIDE',p)]):
            with self.subTest(i=i),tempfile.TemporaryDirectory() as tmp:
                engine=WindowSpool(ctx,Path(tmp)/'spool',max_lateness_ns=0,input_kind='synthetic_fixture')
                with self.assertRaises(ValueError):list(engine.run_records([event],input_complete=True))
        with tempfile.TemporaryDirectory() as tmp:
            def guard():raise RuntimeError('SYNTHETIC_RESOURCE_ABORT')
            engine=WindowSpool(ctx,Path(tmp)/'spool',max_lateness_ns=0,resource_guard=guard)
            with self.assertRaisesRegex(RuntimeError,'RESOURCE_ABORT'):
                list(engine.run_records([],input_complete=True))


if __name__=='__main__':unittest.main()
