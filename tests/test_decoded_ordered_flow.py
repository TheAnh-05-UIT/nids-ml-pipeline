"""Exercise sequential decode, SQLite lookup and ordered tensors using synthetic PCAP.
Kiểm decode tuần tự, lookup SQLite và ordered tensor bằng PCAP giả, không training.
"""
from contextlib import closing,redirect_stdout
from dataclasses import replace
import hashlib
import io
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
from decoded_pcap_fixtures import make_capture,pdml,FakeDecoderProcess
from nids_ml.ordering.decoded import (prepare_decoded_spool,validate_decoded_spool,DecodedPacketSpool,SpoolLimits)
from nids_ml.ordering.contracts import OrderingError
from nids_ml.flows.ordered import OrderedFlowExtractor
from nids_ml.flows.ordered_contracts import configuration as synthetic_config
from nids_ml.flows.decoded_ordered import (DecodedOrderedFlowExtractor,DecodedOrderedReceipt,
    DecodedOrderedTensor,DecodedExtractionComplete,configuration,validate_tensor_bytes)
from nids_ml.flows.extractor import ResourceLimits
from nids_ml.cli.ordered_pcap import extract_diagnostics,main
from nids_ml.labeling.contracts import canonical_bytes


class DecodedFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='decoded-flow-fixture-',dir=ROOT/'artifacts/runs')
        self.work=Path(self.tmp.name)
        self.disk=patch('shutil.disk_usage',return_value=SimpleNamespace(free=10*1024**3));self.disk.start()
        self.rss=patch('nids_ml.pcap.decoder.process_rss',return_value=1024);self.rss.start()
        self.fixture=make_capture(self.work)
        self.spool=self.work/'artifacts/runs/spool'

    def tearDown(self):
        self.rss.stop();self.disk.stop()
        assert Path(self.tmp.name).resolve().parent==(ROOT/'artifacts/runs').resolve()
        self.tmp.cleanup()

    def prepare(self,*,records=None,code=0,output=None,**kwargs):
        f=self.fixture;child=FakeDecoderProcess(pdml(f['records'] if records is None else records),code)
        with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=child):
            result=prepare_decoded_spool(f['view'],self.work,self.spool if output is None else output,
                tshark=f['tool'],tshark_sha256=f['tool_sha'],**kwargs)
        return result

    def events(self,profile='flow-cnn-n20-t10',**kwargs):
        with DecodedPacketSpool(self.spool) as reader:
            engine=DecodedOrderedFlowExtractor(reader.flow_view(),configuration(profile),kwargs.get('limits'))
            events=list(engine.run(reader.references(),reader.lookup,input_complete=True))
        return events

    def test_complete_spool_and_source_state_preserved_after_reorder(self):
        manifest=self.prepare();self.assertEqual(manifest['summary']['records'],4)
        self.assertEqual(manifest['summary']['rejected'],0)
        with DecodedPacketSpool(self.spool) as reader:
            references=list(reader.references())
            self.assertEqual([r.source_packet_ordinal for r in references],[2,3,1,4])
            self.assertEqual([reader.lookup(r.source_packet_ordinal)['tcp']['ack'] for r in references],[20,30,10,40])
        self.assertTrue(manifest['decoder']['stats']['source_complete'])

    def test_numeric_tensors_relative_time_padding_and_lineage(self):
        self.prepare();events=self.events()
        receipts=[e for e in events if type(e) is DecodedOrderedReceipt]
        tensors=[e for e in events if type(e) is DecodedOrderedTensor]
        self.assertEqual([e.receipt.ordinal for e in receipts],[2,3,1,4])
        self.assertEqual(len(tensors),2);self.assertIs(type(events[-1]),DecodedExtractionComplete)
        first=tensors[0];values=[v[0] for v in struct.iter_unpack('<f',first.tensor_bytes)]
        self.assertEqual(first.shape,(20,11,1));self.assertEqual(first.metadata['retained_ordinals'],[2,3,1])
        self.assertEqual(first.metadata['retained_processing_indices'],[1,2,3])
        self.assertEqual(values[0],0);self.assertEqual(values[11],0);self.assertAlmostEqual(values[22],0.2,places=6)
        self.assertAlmostEqual(values[6],20/4294967296,places=12)
        self.assertTrue(all(value==0 for value in values[33:]))
        self.assertEqual(first.metadata['min_source_ordinal'],1)
        self.assertEqual(first.metadata['input_kind'],'decoded_pcap');self.assertFalse(first.metadata['supervised_eligible'])

    def test_n10_n20_overflow_full_lineage_and_boundary(self):
        # A new independent fixture; not extra records invented in the original capture.
        other=self.work/'overflow';other.mkdir()
        self.fixture=make_capture(other,timestamps=tuple([0]*25+[10000000,10000001]))
        self.spool=other/'artifacts/runs/spool';self.work=other
        self.prepare()
        for profile,n,overflow in [('paper-table3-n10',10,16),('flow-cnn-n20-t10',20,6)]:
            events=self.events(profile)
            tensors=[e for e in events if type(e) is DecodedOrderedTensor]
            self.assertEqual(len(tensors),2)
            self.assertEqual(tensors[0].metadata['real_packet_count'],26)
            self.assertEqual(tensors[0].metadata['overflow_packet_count'],overflow)
            self.assertEqual(tensors[0].metadata['retained_processing_indices'],list(range(1,n+1)))
            self.assertEqual(tensors[1].metadata['first_processing_index'],27)
        self.work=self.work.parent

    def test_rejected_packet_retained_blocks_flow_and_output(self):
        value=self.fixture['records'][1]
        rejected=replace(value,layers=value.layers+('ipv6',),all_protocols=value.all_protocols+('ipv6',))
        records=list(self.fixture['records']);records[1]=rejected
        manifest=self.prepare(records=records)
        self.assertEqual(manifest['summary']['records'],4);self.assertEqual(manifest['summary']['rejected'],1)
        self.assertIn('NON_IPV4_OR_AMBIGUOUS_IP',manifest['summary']['rejection_counts'])
        self.assertIsNone(manifest['flow_view'])
        output=self.work/'artifacts/runs/blocked'
        with self.assertRaisesRegex(OrderingError,'FLOW_BLOCKED_ON_REJECTED_RECORDS'):
            extract_diagnostics(self.spool,self.work,output)
        self.assertFalse(output.exists())

    def test_raw_frame_timestamp_or_length_mismatch_fails_before_completion(self):
        value=self.fixture['records'][0]
        wrong=replace(value,fields=tuple(replace(field,show='55') if field.name=='frame.len' else field for field in value.fields))
        records=list(self.fixture['records']);records[0]=wrong
        with self.assertRaisesRegex(OrderingError,'FRAME_VS_RAW_REFERENCE'):self.prepare(records=records)
        self.assertTrue((self.spool/'FAILED.json').exists());self.assertFalse((self.spool/'manifest.json').exists())

    def test_missing_extra_or_child_error_retains_failed_work(self):
        for name,records,code in [('missing',self.fixture['records'][:-1],0),('child',self.fixture['records'],1)]:
            output=self.spool.with_name(name)
            with self.subTest(name=name),self.assertRaises(ValueError):self.prepare(records=records,code=code,output=output)
            self.assertFalse((output/'manifest.json').exists());self.assertTrue((output/'FAILED.json').exists())

    def test_no_overwrite_existing_spool(self):
        self.prepare();before=(self.spool/'manifest.json').read_bytes()
        with self.assertRaises(OrderingError):self.prepare()
        self.assertEqual((self.spool/'manifest.json').read_bytes(),before)

    def test_legacy_synthetic_view_and_config_not_accepted(self):
        manifest=self.prepare();view=manifest['flow_view']
        with self.assertRaises(ValueError):OrderedFlowExtractor(view)
        with self.assertRaises(ValueError):DecodedOrderedFlowExtractor(view,synthetic_config())
        with self.assertRaises(ValueError):DecodedOrderedFlowExtractor(self.fixture['raw_manifest'])

    def test_deterministic_ledgers_and_tensors_across_spools(self):
        first=self.prepare();before=self.events()
        second_path=self.spool.with_name('second');second=self.prepare(output=second_path)
        self.assertEqual(first['summary'],second['summary']);self.assertEqual(first['flow_view'],second['flow_view'])
        self.spool=second_path
        after=self.events()
        self.assertEqual([e.to_dict() for e in before],[e.to_dict() for e in after])

    def test_metadata_database_tampering_rejected(self):
        self.prepare();path=self.spool/'metadata.sqlite'
        with closing(sqlite3.connect(path)) as db:
            db.execute('UPDATE records SET processing_index=100 WHERE source_ordinal=1');db.commit()
        with self.assertRaisesRegex(OrderingError,'HASH_OR_SIZE'):validate_decoded_spool(self.spool)
        manifest=json.loads((self.spool/'manifest.json').read_text())
        manifest['sqlite'].update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
        (self.spool/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaises(OrderingError):validate_decoded_spool(self.spool)

    def test_lookup_bounds_and_missing_references_fail(self):
        self.prepare()
        with DecodedPacketSpool(self.spool) as reader:
            for ordinal in (True,0,999):
                with self.assertRaises(OrderingError):reader.lookup(ordinal)
            references=list(reader.references());engine=DecodedOrderedFlowExtractor(reader.flow_view())
            with self.assertRaisesRegex(ValueError,'MISSING_PROCESSING_REFERENCE'):
                list(engine.run(references[:-1],reader.lookup,input_complete=True))

    def test_lookup_tampered_metadata_and_early_consumer_abort(self):
        self.prepare()
        with DecodedPacketSpool(self.spool) as reader:
            engine=DecodedOrderedFlowExtractor(reader.flow_view());references=list(reader.references())
            def wrong(ordinal):
                packet=reader.lookup(ordinal);packet['timestamp_ns']+=1;return packet
            with self.assertRaisesRegex(ValueError,'SOURCE_PACKET_HASH_CONTEXT_TIMESTAMP_OR_ORDINAL'):
                list(engine.run(references,wrong,input_complete=True))
            engine=DecodedOrderedFlowExtractor(reader.flow_view());stream=engine.run(reader.references(),reader.lookup,input_complete=True)
            next(stream);stream.close();self.assertEqual(engine.status,'ABORTED')

    def test_resource_limit_never_evicts_flows(self):
        self.prepare()
        with DecodedPacketSpool(self.spool) as reader:
            engine=DecodedOrderedFlowExtractor(reader.flow_view(),limits=ResourceLimits(max_packets_per_window=1))
            with self.assertRaisesRegex(ValueError,'WINDOW_PACKET_LIMIT'):
                list(engine.run(reader.references(),reader.lookup,input_complete=True))
            self.assertEqual(engine.status,'FAILED')

    def test_diagnostic_binary_roundtrip_and_unknown_only(self):
        self.prepare();events=self.events();expected=[e for e in events if type(e) is DecodedOrderedTensor]
        output=self.work/'artifacts/runs/flow-diagnostics';report=extract_diagnostics(self.spool,self.work,output)
        self.assertEqual(report['tensor_count'],2);self.assertEqual(report['receipts'],4)
        self.assertEqual((output/'tensors.f32').read_bytes(),b''.join(e.tensor_bytes for e in expected))
        metadata=[json.loads(line) for line in (output/'tensors.jsonl').read_bytes().splitlines()]
        for index,item in enumerate(metadata):
            self.assertEqual(item['byte_offset'],index*20*11*4)
            self.assertEqual(item['tensor_sha256'],hashlib.sha256(expected[index].tensor_bytes).hexdigest())
        self.assertEqual(report['labels'],'UNKNOWN_ONLY');self.assertFalse(report['training'])
        with self.assertRaises(OrderingError):extract_diagnostics(self.spool,self.work,output)

    def test_disk_shortage_and_outside_output_rejected(self):
        with patch('shutil.disk_usage',return_value=SimpleNamespace(free=1)):
            with self.assertRaises(OrderingError):self.prepare()
        with self.assertRaises(OrderingError):self.prepare(output=self.fixture['view']/'child')

    def test_same_count_prefix_claim_and_oversized_sql_blob_rejected(self):
        self.prepare();manifest_path=self.spool/'manifest.json'
        original=json.loads(manifest_path.read_text())
        for malformed in ('same-count-prefix','missing-count-value'):
            forged=json.loads(json.dumps(original))
            if malformed=='same-count-prefix':forged['decoder']['limits']['packet_count']=4
            else:forged['decoder']['command']=['fake-tshark','-c']
            manifest_path.write_text(json.dumps(forged))
            with self.subTest(case=malformed),self.assertRaisesRegex(OrderingError,'BUDGET_OR_EOF_COMMAND'):
                validate_decoded_spool(self.spool)
        path=self.spool/'metadata.sqlite'
        with closing(sqlite3.connect(path)) as db:
            db.execute('UPDATE records SET packet=? WHERE source_ordinal=1',(b'x'*16385,));db.commit()
        original['sqlite'].update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
        manifest_path.write_text(json.dumps(original))
        with self.assertRaisesRegex(OrderingError,'ROW_BYTES'):validate_decoded_spool(self.spool)

    def test_source_change_after_decode_before_publication_rejected(self):
        from nids_ml.ordering.decoded import _summarize
        def change_after_summary(*args):
            result=_summarize(*args)
            path=self.fixture['source'];before=path.stat();data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
            os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
            return result
        with patch('nids_ml.ordering.decoded._summarize',side_effect=change_after_summary):
            with self.assertRaisesRegex(OrderingError,'SOURCE_CHANGED_BEFORE_SPOOL_PUBLICATION'):self.prepare()
        self.assertFalse((self.spool/'manifest.json').exists())

    def test_lost_headroom_and_incomplete_diagnostic_stream_retain_failure(self):
        with patch('shutil.disk_usage',side_effect=[SimpleNamespace(free=10*1024**3),SimpleNamespace(free=1)]):
            with self.assertRaisesRegex(OrderingError,'DISK_RESERVE'):self.prepare()
        self.assertFalse((self.spool/'manifest.json').exists())
        self.spool=self.spool.with_name('second');self.prepare();events=self.events()
        for name,items in [('missing-completion',events[:-1])]:
            output=self.work/'artifacts/runs'/name
            with patch.object(DecodedOrderedFlowExtractor,'run',return_value=iter(items)):
                with self.assertRaisesRegex(OrderingError,'DIAGNOSTIC_INCOMPLETE'):
                    extract_diagnostics(self.spool,self.work,output)
            self.assertTrue((output/'FAILED.json').exists());self.assertFalse((output/'manifest.json').exists())

    def test_producer_close_error_must_precede_publication(self):
        self.prepare();original=DecodedOrderedFlowExtractor.run
        class LateClose:
            def __init__(self,inner):self.inner=inner
            def __iter__(self):return self
            def __next__(self):return next(self.inner)
            def close(self):self.inner.close();raise RuntimeError('SYNTHETIC_LATE_CLOSE_FAILURE')
        def wrapped(engine,*args,**kwargs):return LateClose(original(engine,*args,**kwargs))
        output=self.work/'artifacts/runs/late-close'
        with patch.object(DecodedOrderedFlowExtractor,'run',new=wrapped):
            with self.assertRaisesRegex(RuntimeError,'LATE_CLOSE_FAILURE'):
                extract_diagnostics(self.spool,self.work,output)
        self.assertFalse((output/'manifest.json').exists());self.assertTrue((output/'FAILED.json').exists())

    def test_cli_prepare_inspect_extract(self):
        f=self.fixture;child=FakeDecoderProcess(pdml(f['records']))
        with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=child),redirect_stdout(io.StringIO()) as stream:
            code=main(['prepare','--view',str(f['view']),'--workspace',str(self.work),'--output',str(self.spool),
                '--tshark',str(f['tool']),'--tshark-sha256',f['tool_sha']])
        self.assertEqual(code,0);self.assertTrue(json.loads(stream.getvalue())['flow_ready'])
        with redirect_stdout(io.StringIO()):self.assertEqual(main(['inspect','--spool',str(self.spool)]),0)
        with redirect_stdout(io.StringIO()):
            code=main(['extract','--spool',str(self.spool),'--workspace',str(self.work),
                       '--output',str(self.work/'artifacts/runs/cli-flow')])
        self.assertEqual(code,0)


if __name__=='__main__':unittest.main()
