"""Test stable ordering, source lineage, completion and resource budgets.
Test ordering ổn định, lineage nguồn, completion và budget.
"""
from copy import deepcopy
from contextlib import closing
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import tracemalloc
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
from flow_extraction_fixtures import CAPTURE_TEXT, context, packet
from nids_ml.cli.ordering import main
from nids_ml.flows.features import CaptureContext, ExtractionError, configuration
from nids_ml.flows.extractor import FlowExtractor
from nids_ml.labeling.contracts import canonical_bytes
from nids_ml.ordering.contracts import (OrderingError, OrderingLimitError, OrderingLimits,
    SOURCE_SCHEMA, VIEW_SCHEMA, make_source_manifest, source_order_digest, validate_source)
from nids_ml.ordering.view import (SELECT_ORDER, build_processing_view,
    iter_processing_view, validate_processing_view)


def manifest(packets, *, first=1):
    return make_source_manifest(CaptureContext.from_dict(context()), synthetic_capture_text=CAPTURE_TEXT,
        expected_packets=len(packets), expected_source_order_sha256=source_order_digest(packets),
        source_first_ordinal=first)


class OrderingFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='packet-ordering-synthetic-', dir=ROOT/'artifacts/runs')
        self.base = Path(self.temp.name)
        self.output = self.base/'view'

    def tearDown(self): self.temp.cleanup()

    def build(self, packets, source=None, output=None, limits=None):
        return build_processing_view(iter(packets), manifest(packets) if source is None else source,
                                     self.output if output is None else output, limits=limits)

    def references(self): return list(iter_processing_view(self.output))


class OrderingTests(OrderingFixture):

    def test_sort_regression_stably_without_mutating_source(self):
        packets = [packet(ordinal=i, timestamp_ns=ts) for i,ts in enumerate([3,1,2],1)]
        before = canonical_bytes(packets)
        result = self.build(packets)
        refs = self.references()
        self.assertEqual([r.source_packet_ordinal for r in refs], [2,3,1])
        self.assertEqual([r.timestamp_ns for r in refs], [1,2,3])
        self.assertEqual([r.processing_index for r in refs], [1,2,3])
        self.assertEqual(canonical_bytes(packets), before)
        self.assertEqual(result['stats']['source_timestamp_regressions'], 1)
        self.assertEqual(result['stats']['moved_records'], 3)

    def test_equal_timestamp_tie_uses_original_ordinal(self):
        packets = [packet(ordinal=i, timestamp_ns=7) for i in range(1,5)]
        self.build(packets)
        self.assertEqual([r.source_packet_ordinal for r in self.references()], [1,2,3,4])

    def test_deterministic_records_and_semantic_digest(self):
        packets = [packet(ordinal=i,timestamp_ns=4-i) for i in range(1,4)]
        first = self.build(packets)
        second = self.build(packets, output=self.base/'view-2')
        self.assertEqual(first['processing_order_sha256'], second['processing_order_sha256'])
        self.assertEqual([r.to_dict() for r in iter_processing_view(self.base/'view-2')],
                         [r.to_dict() for r in self.references()])

    def test_packet_sha_and_capture_follow_source_identity(self):
        packets = [packet(ordinal=1,timestamp_ns=2),packet(ordinal=2,timestamp_ns=1,reverse=True)]
        self.build(packets)
        for r in self.references():
            original = packets[r.source_packet_ordinal-1]
            self.assertEqual(r.source_packet_sha256, hashlib.sha256(canonical_bytes(original)).hexdigest())
            self.assertEqual(r.capture.to_dict(), original['capture'])
            self.assertEqual(r.timestamp_ns, original['timestamp_ns'])

    def test_monotonic_input_identity_processing_view(self):
        packets = [packet(ordinal=i,timestamp_ns=i) for i in range(1,5)]
        self.assertEqual(self.build(packets)['stats']['moved_records'], 0)

    def test_non_one_source_start_is_preserved(self):
        packets = [packet(ordinal=100,timestamp_ns=2), packet(ordinal=101,timestamp_ns=1)]
        self.build(packets, manifest(packets,first=100))
        self.assertEqual([r.source_packet_ordinal for r in self.references()], [101,100])
        self.assertEqual([r.processing_index for r in self.references()], [1,2])

    def test_sqlite_signed_boundaries_without_float_rounding(self):
        first = 2**63-2
        packets = [packet(ordinal=first,timestamp_ns=2**63-1),
                   packet(ordinal=first+1,timestamp_ns=2**63-2)]
        self.build(packets, manifest(packets,first=first))
        self.assertEqual(self.references()[0].timestamp_ns, 2**63-2)

    def test_same_contents_different_source_ordinals_not_deduplicated(self):
        packets = [packet(ordinal=1),packet(ordinal=2)]
        self.build(packets)
        self.assertEqual(len(self.references()),2)

    def test_preserves_unsupported_metadata_for_later_eligibility_decision(self):
        p = packet(); p['fragmented'] = True
        self.build([p])
        self.assertEqual(len(self.references()),1)

    def test_no_labels_or_model_features_added_to_references(self):
        result = self.build([packet()])
        self.assertEqual(set(self.references()[0].to_dict()),
                         {'processing_index','source_packet_ordinal','timestamp_ns','source_packet_sha256','capture'})
        self.assertEqual(result['labels'],'NOT_EVALUATED')
        self.assertEqual(result['real_data_export'],'BLOCKED')
        self.assertEqual(result['flow_integration'],'NOT_ACTIVE')

    def test_first_n20_is_processing_order_and_overflow_is_not_lost(self):
        packets = [packet(ordinal=i,timestamp_ns=26-i) for i in range(1,26)]
        self.build(packets)
        refs = self.references()
        self.assertEqual([r.source_packet_ordinal for r in refs[:20]], list(range(25,5,-1)))
        self.assertEqual(len(refs),25)
        self.assertEqual(sorted(r.source_packet_ordinal for r in refs),list(range(1,26)))

    def test_exact_ten_second_boundary_timestamps_not_rounded(self):
        times = [10000000001,0,10000000000,10000000000,2]
        self.build([packet(ordinal=i,timestamp_ns=ts) for i,ts in enumerate(times,1)])
        self.assertEqual([r.timestamp_ns for r in self.references()], [0,2,10000000000,10000000000,10000000001])

    def test_existing_flow_extractor_guard_not_bypassed_by_renumbering(self):
        packets = [packet(ordinal=1,timestamp_ns=2),packet(ordinal=2,timestamp_ns=1)]
        self.build(packets)
        reordered = [packets[r.source_packet_ordinal-1] for r in self.references()]
        engine = FlowExtractor(CaptureContext.from_dict(context()), configuration())
        with self.assertRaisesRegex(ExtractionError,'ordinal'):
            list(engine.run(reordered,input_complete=True))
        self.assertEqual(engine.status,'FAILED')

    def test_existing_capture_order_rejects_timestamp_regression(self):
        engine = FlowExtractor(CaptureContext.from_dict(context()), configuration())
        with self.assertRaisesRegex(ExtractionError,'Timestamp regression'):
            list(engine.run([packet(ordinal=1,timestamp_ns=2),packet(ordinal=2,timestamp_ns=1)],input_complete=True))


class FailureTests(OrderingFixture):
    def test_missing_source_packet_rejected_and_failed_artifact_retained(self):
        packets=[packet(),packet(ordinal=2)]
        with self.assertRaises(OrderingError): self.build(packets[:1],manifest(packets))
        self.assertTrue((self.output/'FAILED.json').exists())
        self.assertFalse((self.output/'manifest.json').exists())
        with self.assertRaises(OrderingError): validate_processing_view(self.output)

    def test_extra_packet_rejected_not_truncated(self):
        packets=[packet(),packet(ordinal=2)]
        with self.assertRaises(OrderingLimitError): self.build(packets,manifest(packets[:1]))

    def test_gap_duplicate_and_regressed_source_ordinals_rejected(self):
        for index,ordinals in enumerate(([1,3],[1,1],[2,1])):
            packets=[packet(ordinal=n) for n in ordinals]
            with self.assertRaises(OrderingError): self.build(packets,output=self.base/str(index))

    def test_cross_capture_run_segment_or_hash_rejected(self):
        original=[packet(),packet(ordinal=2)]
        for field,value in [('capture_id','other'),('capture_sha256','a'*64),('run_id','other'),('segment_id','other')]:
            packets=deepcopy(original);packets[1]['capture'][field]=value
            with self.assertRaises(OrderingError): self.build(packets,manifest(original),self.base/field)

    def test_wrong_expected_digest_rejected(self):
        source=manifest([packet()]);source['expected_source_order_sha256']='a'*64
        with self.assertRaises(OrderingError): self.build([packet()],source)

    def test_generator_failure_cannot_publish_complete_view(self):
        def broken():
            yield packet()
            raise RuntimeError('synthetic generator failed')
        source=manifest([packet(),packet(ordinal=2)])
        with self.assertRaisesRegex(RuntimeError,'generator failed'):
            build_processing_view(broken(),source,self.output)
        self.assertTrue((self.output/'FAILED.json').exists())
        self.assertFalse((self.output/'manifest.json').exists())

    def test_negative_float_nan_bool_timestamp_and_ordinal_rejected(self):
        for i,(field,value) in enumerate([('timestamp_ns',-1),('timestamp_ns',1.0),('timestamp_ns',float('nan')),
                                         ('timestamp_ns',True),('ordinal',True)]):
            p=packet();p[field]=value
            with self.assertRaises(ValueError): self.build([p],manifest([packet()]),self.base/str(i))

    def test_label_field_not_allowed_inside_packet(self):
        p=packet();p['label']=1
        with self.assertRaises(OrderingError): self.build([p],manifest([packet()]))

    def test_incomplete_declaration_and_real_input_rejected_before_output(self):
        for field,value in [('input_complete',False),('input_kind','decoded_pcap')]:
            source=manifest([packet()]);source[field]=value
            with self.assertRaises(OrderingError): self.build([packet()],source)
            self.assertFalse(self.output.exists())

    def test_synthetic_literal_hash_must_match(self):
        source=manifest([packet()]);source['synthetic_capture_text']='changed'
        with self.assertRaises(OrderingError): self.build([packet()],source)

    def test_policy_change_and_numeric_false_not_accepted(self):
        for value in [True,0]:
            source=manifest([packet()]);source['ordering_policy']['timestamp_correction']=value
            with self.assertRaises(OrderingError): self.build([packet()],source)

    def test_existing_output_not_overwritten(self):
        self.build([packet()]);before=(self.output/'manifest.json').read_bytes()
        with self.assertRaises(OrderingError): self.build([packet()])
        self.assertEqual((self.output/'manifest.json').read_bytes(),before)

    def test_missing_completion_manifest_rejected(self):
        self.output.mkdir();(self.output/'plan.json').write_text('{}')
        with self.assertRaises(ValueError): validate_processing_view(self.output)

    def test_sqlite_tamper_rejected_by_hash(self):
        self.build([packet()]);path=self.output/'index.sqlite'
        with path.open('ab') as f:f.write(b'tampered')
        with self.assertRaisesRegex(OrderingError,'SIZE|Size|size|HASH'):
            validate_processing_view(self.output)

    def test_manifest_digest_tamper_rejected(self):
        self.build([packet()]);path=self.output/'manifest.json';value=json.loads(path.read_text())
        value['processing_order_sha256']='a'*64;path.write_bytes(canonical_bytes(value))
        with self.assertRaises(OrderingError): validate_processing_view(self.output)

    def test_source_regression_stats_tamper_rejected(self):
        self.build([packet()]);path=self.output/'manifest.json';value=json.loads(path.read_text())
        value['stats']['source_timestamp_regressions']=1;path.write_bytes(canonical_bytes(value))
        with self.assertRaises(OrderingError): validate_processing_view(self.output)


class ResourceAndContractTests(unittest.TestCase):
    memory_observation = None
    def test_invalid_limits_and_bool_values_fail_closed(self):
        for limits in [OrderingLimits(max_records=True),OrderingLimits(sqlite_bytes=32769),
                       OrderingLimits(cache_kib=8193),OrderingLimits(scratch_bytes=65536),
                       OrderingLimits(timeout_seconds=0)]:
            with self.assertRaises(OrderingLimitError): limits.validate()

    def test_record_count_budget_rejects_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'view'
            packets=[packet(),packet(ordinal=2)]
            with self.assertRaises(OrderingLimitError):
                build_processing_view(packets,manifest(packets),output,limits=OrderingLimits(max_records=1))
            self.assertFalse(output.exists())

    def test_sqlite_page_limit_failure_does_not_drop_records_or_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            packets=[packet(ordinal=i,timestamp_ns=1000-i) for i in range(1,1001)]
            output=Path(directory)/'view'
            limits=OrderingLimits(max_records=1000,sqlite_bytes=32768,scratch_bytes=131072,cache_kib=64)
            with self.assertRaises(OrderingError): build_processing_view(packets,manifest(packets),output,limits=limits)
            self.assertTrue((output/'FAILED.json').exists())
            self.assertFalse((output/'manifest.json').exists())

    def test_low_disk_headroom_rejected_before_directory_creation(self):
        with tempfile.TemporaryDirectory() as directory, patch('nids_ml.ordering.view.shutil.disk_usage') as usage:
            usage.return_value.free=0
            output=Path(directory)/'view'
            with self.assertRaises(OrderingLimitError): build_processing_view([packet()],manifest([packet()]),output)
            self.assertFalse(output.exists())

    def test_time_limit_failure_keeps_failed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'view'
            with patch('nids_ml.ordering.view.time.monotonic',side_effect=[0,1000]):
                with self.assertRaises(OrderingLimitError):
                    build_processing_view([packet()],manifest([packet()]),output)
            self.assertTrue((output/'FAILED.json').exists())

    def test_pk_satisfies_order_without_temporary_sort(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'view'
            build_processing_view([packet()],manifest([packet()]),output)
            with closing(sqlite3.connect(output/'index.sqlite')) as db:
                plan=db.execute('EXPLAIN QUERY PLAN '+SELECT_ORDER).fetchall()
            self.assertFalse(any('TEMP B-TREE' in row[-1].upper() for row in plan))

    def test_generator_memory_is_bounded_without_materializing_input(self):
        count=4096
        def packets():
            for i in range(1,count+1): yield packet(ordinal=i,timestamp_ns=count-i)
        source=make_source_manifest(CaptureContext.from_dict(context()),synthetic_capture_text=CAPTURE_TEXT,
            expected_packets=count,expected_source_order_sha256=source_order_digest(packets()))
        with tempfile.TemporaryDirectory() as directory:
            tracemalloc.start()
            try:
                result=build_processing_view(packets(),source,Path(directory)/'view')
                _,peak=tracemalloc.get_traced_memory()
            finally:tracemalloc.stop()
        type(self).memory_observation=dict(synthetic_records=count,python_tracemalloc_peak_bytes=peak,
            sqlite_index_bytes=result['index']['bytes'],sqlite_cache_kib=result['limits']['cache_kib'],
            total_RSS_measured=False, native_measurement=False)
        self.assertLess(peak,4*1048576)  # Python allocations, không phải tổng RSS/SQLite C cache.

    def test_schema_files_match_exported_contracts(self):
        for name,schema in [('packet-ordering-source.schema.json',SOURCE_SCHEMA),('packet-processing-view.schema.json',VIEW_SCHEMA)]:
            self.assertEqual(json.loads((ROOT/'contracts'/name).read_text()),schema)


class CLITests(unittest.TestCase):
    def test_build_inspect_preview_and_reject_existing(self):
        with tempfile.TemporaryDirectory(prefix='ordering-cli-synthetic-',dir=ROOT/'artifacts/runs') as directory:
            output=Path(directory)/'view'
            with patch('sys.stdout',io.StringIO()) as stdout:
                self.assertEqual(main(['build-synthetic','--fixture',str(ROOT/'configs/packet-ordering-synthetic.example.json'),'--output',str(output)]),0)
                result=json.loads(stdout.getvalue());self.assertEqual(result['record_count'],5)
            with patch('sys.stdout',io.StringIO()):
                self.assertEqual(main(['inspect','--input',str(output)]),0)
            with patch('sys.stdout',io.StringIO()) as stdout:
                self.assertEqual(main(['preview','--input',str(output),'--limit','3']),0)
                result=json.loads(stdout.getvalue())
                self.assertEqual([r['source_packet_ordinal'] for r in result['records']],[2,5,3])
                self.assertFalse(result['preview_complete'])
            with patch('sys.stdout',io.StringIO()):
                self.assertEqual(main(['build-synthetic','--fixture',str(ROOT/'configs/packet-ordering-synthetic.example.json'),'--output',str(output)]),2)

    def test_invalid_preview_limit_and_raw_input_rejected_without_read(self):
        with patch('sys.stdout',io.StringIO()):
            self.assertEqual(main(['preview','--input','missing','--limit','65']),2)
            self.assertEqual(main(['build-synthetic','--fixture',str(ROOT/'data/raw/not-read.json'),
                                  '--output',str(ROOT/'artifacts/runs/not-created')]),2)

    def test_help_works_with_windows_cp1252(self):
        stream=io.TextIOWrapper(io.BytesIO(),encoding='cp1252',write_through=True)
        with patch('sys.stdout',stream),self.assertRaises(SystemExit) as exit:
            main(['--help'])
        self.assertEqual(exit.exception.code,0)


if __name__=='__main__': unittest.main()
