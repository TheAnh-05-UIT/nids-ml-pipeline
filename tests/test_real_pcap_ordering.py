"""Test real-PCAP index contracts with tiny synthetic captures and corruption cases.
Kiểm contract index PCAP bằng capture giả nhỏ, thứ tự lùi và các trường hợp hỏng.
"""
from contextlib import redirect_stdout, closing
from dataclasses import replace
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import sqlite3
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tests'))
from pcap_fixtures import frame,pcap
from nids_ml.cli.pcap_ordering import main
from nids_ml.flows.features import CaptureContext
from nids_ml.flows.ordered import OrderedFlowExtractor
from nids_ml.ordering.contracts import OrderingError
from nids_ml.ordering.pcap import (PcapViewLimits,PcapRecordLookup,build_pcap_view,
    iter_pcap_view,validate_pcap_view)
from nids_ml.pcap.metadata import scan_record_headers


class PcapViewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='pcap-view-fixture-',dir=ROOT/'artifacts/runs')
        self.work=Path(self.tmp.name)
        (self.work/'data/raw').mkdir(parents=True);(self.work/'artifacts/runs').mkdir(parents=True)
        self.path=self.work/'data/raw/synthetic.pcap'
        self.frames=[frame(payload=bytes([i])) for i in range(4)]
        self.original=pcap(list(zip((3000000,1000000,1000000,2000000),self.frames)))
        self.path.write_bytes(self.original)
        self.source=dict(capture=CaptureContext('synthetic-pcap',hashlib.sha256(self.original).hexdigest(),
            'synthetic-run','synthetic-segment').to_dict(),path='data/raw/synthetic.pcap',
            size_bytes=len(self.original),expected_records=4)
        self.output=self.work/'artifacts/runs/view'
        self.disk=patch('shutil.disk_usage',return_value=SimpleNamespace(free=10*1024**3));self.disk.start()

    def tearDown(self):
        self.disk.stop()
        assert self.work.resolve().parent==(ROOT/'artifacts/runs').resolve()
        self.tmp.cleanup()

    def build(self,**kwargs):return build_pcap_view(self.source,self.work,self.output,**kwargs)

    def test_stable_sort_ties_all_ordinals_preserved(self):
        manifest=self.build();records=list(iter_pcap_view(self.output))
        self.assertEqual([r.source_ordinal for r in records],[2,3,4,1])
        self.assertEqual([r.processing_index for r in records],[1,2,3,4])
        self.assertEqual(manifest['summary']['source_timestamp_regressions'],1)
        self.assertEqual(manifest['summary']['max_regression_ns'],2000000000)
        self.assertEqual(manifest['summary']['moved_records'],4)
        self.assertEqual(self.path.read_bytes(),self.original)
        self.assertEqual(manifest['HDF5_export'],'BLOCKED')

    def test_deterministic_semantic_digest(self):
        first=self.build()
        second=build_pcap_view(self.source,self.work,self.output.with_name('second'))
        self.assertEqual(first['summary'],second['summary'])

    def test_lookup_reads_exact_original_payload_after_sort(self):
        self.build()
        with PcapRecordLookup(self.output,self.work) as lookup:
            for reference in iter_pcap_view(self.output):
                self.assertEqual(lookup(reference),self.frames[reference.source_ordinal-1])
        with self.assertRaises(OrderingError):lookup(list(iter_pcap_view(self.output))[0])

    def test_bad_reference_context_position_hash_and_bool_rejected(self):
        self.build();reference=next(iter_pcap_view(self.output))
        with PcapRecordLookup(self.output,self.work) as lookup:
            for altered in (replace(reference,processing_index=2),replace(reference,raw_record_sha256='a'*64),
                            replace(reference,record_offset=25),replace(reference,processing_index=True),
                            replace(reference,capture={})):
                with self.subTest(reference=altered),self.assertRaises(OrderingError):lookup(altered)

    def test_wrong_count_retains_failure_without_manifest(self):
        self.source['expected_records']=5
        with self.assertRaisesRegex(OrderingError,'EXPECTED_COUNT_MISMATCH'):self.build()
        self.assertTrue((self.output/'FAILED.json').exists())
        self.assertFalse((self.output/'manifest.json').exists())
        with self.assertRaisesRegex(OrderingError,'FAILED'):validate_pcap_view(self.output)

    def test_wrong_hash_no_completed_view(self):
        self.source['capture']['capture_sha256']='f'*64
        with self.assertRaises(ValueError):self.build()
        self.assertFalse((self.output/'manifest.json').exists())

    def test_no_overwrite_existing_output(self):
        self.build();before=(self.output/'manifest.json').read_bytes()
        with self.assertRaises(OrderingError):self.build()
        self.assertEqual((self.output/'manifest.json').read_bytes(),before)

    def test_disk_shortage_no_output(self):
        with patch('shutil.disk_usage',return_value=SimpleNamespace(free=2*1024**3)):
            with self.assertRaisesRegex(OrderingError,'DISK_RESERVE_PLUS_SCRATCH'):self.build()
        self.assertFalse(self.output.exists())

    def test_timeout_and_lost_headroom_keep_failed_artifacts(self):
        with patch('nids_ml.ordering.pcap.time.monotonic',side_effect=lambda:next(clock)):
            clock=itertools.count(0,2)
            with self.assertRaisesRegex(OrderingError,'TIMEOUT'):
                self.build(limits=replace(PcapViewLimits(),timeout_seconds=1))
        self.assertTrue((self.output/'FAILED.json').exists())
        self.assertFalse((self.output/'manifest.json').exists())
        self.output=self.output.with_name('lost-disk')
        with patch('shutil.disk_usage',side_effect=[SimpleNamespace(free=10*1024**3),SimpleNamespace(free=1)]):
            with self.assertRaisesRegex(OrderingError,'DISK_RESERVE'):self.build()
        self.assertTrue((self.output/'FAILED.json').exists())

    def test_truncated_file_and_extra_records_do_not_complete(self):
        self.source['expected_records']=3
        with self.assertRaisesRegex(OrderingError,'EXTRA_RECORD'):self.build()
        self.assertFalse((self.output/'manifest.json').exists())
        self.output=self.output.with_name('truncated')
        data=self.original[:-1];self.path.write_bytes(data)
        self.source.update(size_bytes=len(data),expected_records=4)
        self.source['capture']['capture_sha256']=hashlib.sha256(data).hexdigest()
        with self.assertRaisesRegex(ValueError,'TRUNCATED_RECORD_PAYLOAD'):self.build()
        self.assertFalse((self.output/'manifest.json').exists())

    def test_record_limit_cannot_silently_expand(self):
        with self.assertRaises(OrderingError):self.build(limits=replace(PcapViewLimits(),max_records=3))
        with self.assertRaises(OrderingError):replace(PcapViewLimits(),max_records=250001).validate()

    def test_traversal_and_wrong_source_size_rejected(self):
        for key,value in [('path','data/raw/../../outside.pcap'),('size_bytes',True),('expected_records',True)]:
            source=dict(self.source,**{key:value})
            with self.subTest(key=key),self.assertRaises(OrderingError):
                build_pcap_view(source,self.work,self.output)

    def test_index_file_and_semantic_tampering_rejected(self):
        manifest=self.build();path=self.output/'index.sqlite'
        with closing(sqlite3.connect(path)) as db:
            db.execute('UPDATE packets SET processing_index=1');db.commit()
        with self.assertRaisesRegex(OrderingError,'HASH_OR_SIZE'):validate_pcap_view(self.output)
        manifest['index']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        manifest['index']['bytes']=path.stat().st_size
        (self.output/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        with self.assertRaisesRegex(OrderingError,'RANGE_OR_TYPE'):validate_pcap_view(self.output)

    def test_source_mutation_preserving_size_mtime_detected_at_close(self):
        self.build();lookup=PcapRecordLookup(self.output,self.work)
        before=self.path.stat();changed=bytearray(self.original);changed[-1]^=1;self.path.write_bytes(changed)
        os.utime(self.path,ns=(before.st_atime_ns,before.st_mtime_ns))
        with self.assertRaisesRegex(OrderingError,'CHANGED_DURING_LOOKUP'):lookup.close()

    def test_unsupported_payloads_are_kept_not_filtered(self):
        self.original=pcap([(0,b'not-an-IP-packet'),(1,b'another-unsupported-payload')])
        self.path.write_bytes(self.original);self.source['size_bytes']=len(self.original)
        self.source['capture']['capture_sha256']=hashlib.sha256(self.original).hexdigest()
        self.source['expected_records']=2
        self.assertEqual(self.build()['summary']['record_count'],2)

    def test_old_synthetic_flow_gate_not_weakened(self):
        manifest=self.build()
        with self.assertRaises(ValueError):OrderedFlowExtractor(manifest)

    def test_record_callback_does_not_change_existing_scan_statistics(self):
        options=dict(expected_sha256=self.source['capture']['capture_sha256'],expected_size=len(self.original))
        old=scan_record_headers(self.path,**options);references=[]
        new=scan_record_headers(self.path,record_sink=references.append,**options)
        old.pop('elapsed_seconds');new.pop('elapsed_seconds')
        self.assertEqual(old,new);self.assertEqual(len(references),4)

    def test_big_endian_nanosecond_capture(self):
        data=struct.pack('>IHHIIII',0xa1b23c4d,2,4,0,0,262144,1)
        data+=struct.pack('>IIII',1,7,len(self.frames[0]),len(self.frames[0]))+self.frames[0]
        self.path.write_bytes(data);self.source.update(size_bytes=len(data),expected_records=1)
        self.source['capture']['capture_sha256']=hashlib.sha256(data).hexdigest()
        self.build();reference=next(iter_pcap_view(self.output))
        self.assertEqual(reference.timestamp_ns,1000000007)
        with PcapRecordLookup(self.output,self.work) as lookup:self.assertEqual(lookup(reference),self.frames[0])

    def test_cli_build_inspect_preview_and_invalid_limit(self):
        config=self.work/'plan.json';config.write_text(json.dumps(dict(format='nids-pcap-view-plan/1.0.0',source=self.source)))
        with redirect_stdout(io.StringIO()) as stream:
            code=main(['build','--config',str(config),'--workspace',str(self.work),'--output',str(self.output)])
        self.assertEqual(code,0);self.assertEqual(json.loads(stream.getvalue())['status'],'COMPLETE_PCAP_RECORD_VIEW')
        for args in (['inspect','--input',str(self.output)],['preview','--input',str(self.output),'--limit','2']):
            with redirect_stdout(io.StringIO()) as stream:code=main(args)
            self.assertEqual(code,0)
        self.assertEqual(len(json.loads(stream.getvalue())['records']),2)
        with redirect_stdout(io.StringIO()):code=main(['preview','--input',str(self.output),'--limit','65'])
        self.assertEqual(code,2)


if __name__=='__main__':unittest.main()
