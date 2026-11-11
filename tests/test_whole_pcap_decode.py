"""Test whole-capture EOF/count, mutation, resource and process-ownership guards.
Kiểm EOF/count toàn capture, mutation, RSS và ownership decoder bằng fixtures nhỏ.
"""
from dataclasses import replace
import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch,mock_open

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tests'))
from decoded_pcap_fixtures import make_capture,pdml,FakeDecoderProcess
from nids_ml.pcap.decoder import DecoderError,DecodeLimits,process_rss
from nids_ml.pcap.whole import WholeCaptureDecoder,WholeCaptureLimits


class WholeDecoderTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='whole-pcap-fixture-',dir=ROOT/'artifacts/runs')
        self.work=Path(self.tmp.name)
        self.disk=patch('shutil.disk_usage',return_value=SimpleNamespace(free=10*1024**3));self.disk.start()
        self.fixture=make_capture(self.work);self.rss=patch('nids_ml.pcap.decoder.process_rss',return_value=1024);self.rss.start()

    def tearDown(self):
        self.rss.stop();self.disk.stop()
        assert self.work.resolve().parent==(ROOT/'artifacts/runs').resolve()
        self.tmp.cleanup()

    def decoder(self,expected_records=4):
        f=self.fixture
        return WholeCaptureDecoder(f['source'],expected_records=expected_records,
            expected_sha256=f['descriptor']['capture']['capture_sha256'],expected_size=f['source'].stat().st_size,
            tshark=f['tool'],tshark_sha256=f['tool_sha'])

    def test_complete_eof_and_one_sequential_command(self):
        d=self.decoder();child=FakeDecoderProcess(pdml(self.fixture['records']))
        with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=child) as start:
            records=list(d.records())
        self.assertEqual([r.number('frame.number') for r in records],[1,2,3,4])
        self.assertEqual(d.command[d.command.index('-c')+1],'5')
        self.assertEqual(d.command[d.command.index('-r')+1],str(self.fixture['source']))
        start.assert_called_once();self.assertTrue(d.stats['source_complete'])
        self.assertEqual(d.status,'COMPLETE_WHOLE_CAPTURE');self.assertFalse(child.terminated)

    def test_missing_extra_nonzero_and_partial_xml_fail_closed(self):
        cases=[(pdml(self.fixture['records'][:-1]),0,4),
               (pdml(self.fixture['records']),0,3),
               (pdml(self.fixture['records']),1,4),
               (pdml(self.fixture['records'],close=False),0,4)]
        for raw,code,count in cases:
            d=self.decoder(count);child=FakeDecoderProcess(raw,code)
            with self.subTest(count=count,code=code),patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=child):
                with self.assertRaises(DecoderError):list(d.records())
            self.assertFalse(d.stats['source_complete']);self.assertEqual(d.status,'FAILED')
            self.assertTrue(d.report()['child_exited'])

    def test_abort_terminates_only_owned_child(self):
        d=self.decoder();child=FakeDecoderProcess(pdml(self.fixture['records']))
        with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=child):
            events=d.records();next(events);events.close()
        self.assertEqual(d.status,'ABORTED');self.assertFalse(d.stats['source_complete'])
        self.assertTrue(child.terminated)

    def test_same_signature_source_mutation_at_end_is_detected(self):
        d=self.decoder();child=FakeDecoderProcess(pdml(self.fixture['records']))
        with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=child):
            events=d.records();next(events)
            path=self.fixture['source'];before=path.stat();data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
            os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
            with self.assertRaisesRegex(DecoderError,'SOURCE_CHANGED'):list(events)
        self.assertEqual(d.status,'FAILED')

    def test_whole_limits_do_not_relax_legacy_prefix(self):
        WholeCaptureLimits(packet_count=250001).validate()
        with self.assertRaises(DecoderError):DecodeLimits(packet_count=4097).validate()
        for key,value in [('stdout_bytes',8*1024**3),('timeout_seconds',1800.0),('packet_count',True)]:
            with self.subTest(key=key),self.assertRaises(DecoderError):
                replace(WholeCaptureLimits(),**{key:value}).validate()

    def test_invalid_expected_count_and_rss_abort(self):
        for count in (True,0,250001):
            with self.assertRaises(DecoderError):self.decoder(count)
        d=self.decoder();child=FakeDecoderProcess(pdml(self.fixture['records']))
        with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=child),patch('nids_ml.pcap.decoder.process_rss',return_value=2**31):
            with self.assertRaisesRegex(DecoderError,'RSS_ABORT'):list(d.records())
        self.assertTrue(child.terminated)


class LinuxRssTests(unittest.TestCase):
    def test_linux_rss_units_and_pid(self):
        with patch('nids_ml.pcap.decoder.sys_platform_linux',return_value=True), \
             patch('builtins.open',mock_open(read_data='Name:\towned-child\nVmRSS:\t1234 kB\n')) as source:
            self.assertEqual(process_rss(123),1234*1024)
            source.assert_called_once_with('/proc/123/status',encoding='ascii')

    def test_missing_probe_and_invalid_pid_are_not_zero(self):
        with patch('nids_ml.pcap.decoder.sys_platform_linux',return_value=True),patch('builtins.open',side_effect=FileNotFoundError):
            self.assertIsNone(process_rss(123))
        for pid in (True,0,-1,'123'):
            self.assertIsNone(process_rss(pid))


if __name__=='__main__':unittest.main()
