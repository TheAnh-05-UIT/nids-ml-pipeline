"""Synthetic-only parity, ordering and failure tests for the window spool.
Kiểm lõi window spool bằng fixture giả; không phải kết quả dataset MQTT thật.
"""
from dataclasses import replace
from contextlib import closing
import hashlib
import json
from pathlib import Path
import random
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from nids_ml.flows.features import CaptureContext, ExtractionError, extract_features, normalize_row
from nids_ml.flows.extractor import FlowExtractor, FlowTensor
from nids_ml.flows.streaming import StreamingLimits, WindowSpool
from flow_extraction_fixtures import bundle, context, packet


class StreamingFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)
        self.context = CaptureContext.from_dict(context())

    def engine(self, name='spool', lateness=0, limits=None):
        return WindowSpool(self.context, self.parent/name, max_lateness_ns=lateness,
                           input_kind='synthetic_fixture', limits=limits)

    def run_packets(self, packets, **kwargs):
        engine = self.engine(**kwargs)
        result = list(engine.run((extract_features(p) for p in packets), input_complete=True))
        return engine, result

    def test_parity_with_accepted_core_bytes_counts_direction_and_padding(self):
        packets = bundle()['packets']
        old = [e for e in FlowExtractor(self.context).run(packets, input_complete=True) if isinstance(e, FlowTensor)]
        engine, result = self.run_packets(packets)
        self.assertEqual(engine.status, 'COMPLETE')
        self.assertEqual([t.tensor_bytes for t in result], [t.tensor_bytes for t in old])
        for actual, expected in zip(result, old):
            metadata = actual.metadata
            metadata.pop('processing_view')
            self.assertEqual(metadata, expected.metadata)

    def test_stable_reordering_matches_manual_golden_and_preserves_original_ordinals(self):
        raw = [packet(ordinal=1, timestamp_ns=200), packet(ordinal=2, timestamp_ns=100, reverse=True),
               packet(ordinal=3, timestamp_ns=200), packet(ordinal=4, timestamp_ns=300)]
        engine, result = self.run_packets(raw, lateness=100)
        features = sorted(map(extract_features, raw), key=lambda p: (p.timestamp_ns, p.ordinal))
        expected = b''.join(struct.pack('<11f', *normalize_row(p.features, p.timestamp_ns-100)) for p in features)
        self.assertEqual(result[0].tensor_bytes, expected + bytes((20-len(raw))*44))
        self.assertEqual(result[0].metadata['retained_ordinals'], [2, 1, 3, 4])
        self.assertEqual(result[0].metadata['forward_packet_count'], 1)
        self.assertEqual(engine.stats['max_observed_lateness_ns'], 100)

    def test_decoded_feature_scope_reuses_exact_core_and_upstream_failure_invalidates_stream(self):
        raw = []
        for value in bundle()['packets']:
            raw.append(dict(value, schema_version='1.1.0', input_kind='decoded_pcap'))
        from nids_ml.flows.features import configuration
        old = [e for e in FlowExtractor(self.context, configuration(input_kind='decoded_pcap')).run(
            raw, input_complete=True) if isinstance(e, FlowTensor)]
        engine = WindowSpool(self.context, self.parent/'decoded', max_lateness_ns=0)
        result = list(engine.run((extract_features(p, input_kind='decoded_pcap') for p in raw),
                                 input_complete=True))
        self.assertEqual([t.tensor_bytes for t in result], [t.tensor_bytes for t in old])
        self.assertEqual(result[0].metadata['input_kind'], 'decoded_pcap')
        failed = self.engine(name='upstream')
        def broken():
            yield extract_features(packet())
            raise RuntimeError('SYNTHETIC_DECODER_EOF_FAILURE')
        with self.assertRaisesRegex(RuntimeError, 'DECODER_EOF_FAILURE'):
            list(failed.run(broken(), input_complete=True))
        self.assertEqual(failed.status, 'FAILED')
        self.assertTrue((failed.directory/'FAILED.json').exists())

    def test_many_windows_reuse_disk_and_match_legacy(self):
        raw = [packet(ordinal=i+1, timestamp_ns=(i//8)*11*10**9+i%8, flow=i%4) for i in range(800)]
        old = [e for e in FlowExtractor(self.context).run(raw, input_complete=True) if isinstance(e, FlowTensor)]
        engine, result = self.run_packets(raw)
        self.assertEqual(engine.stats['windows'], 100)
        self.assertEqual(engine.stats['peak_window_packets'], 8)
        self.assertEqual([x.tensor_bytes for x in result], [x.tensor_bytes for x in old])
        self.assertLess(engine.stats['peak_sqlite_bytes'], 65536)
        import sqlite3
        with closing(sqlite3.connect(engine.directory/'window.sqlite')) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM packets').fetchone()[0], 0)

    def test_more_than_legacy_active_flow_cap_uses_disk(self):
        raw = [packet(ordinal=i+1, timestamp_ns=i, flow=i) for i in range(300)]
        engine, result = self.run_packets(raw)
        self.assertEqual(len(result), 300)
        self.assertEqual(engine.stats['peak_window_packets'], 300)
        self.assertLessEqual(engine.stats['peak_heap_packets'], 2)

    def test_overflow_counts_and_digest_cover_packets_after_n20(self):
        raw = [packet(ordinal=i+1, timestamp_ns=i) for i in range(25)]
        engine, result = self.run_packets(raw)
        metadata = result[0].metadata
        self.assertEqual((metadata['retained_packet_count'], metadata['overflow_packet_count']), (20, 5))
        self.assertEqual(metadata['last_ordinal'], 25)
        expected = hashlib.sha256(b''.join(bytes.fromhex(extract_features(p).packet_sha256) for p in raw))
        self.assertEqual(metadata['full_packet_digest'], expected.hexdigest())
        self.assertEqual(engine.stats['overflow_packets'], 5)
        self.assertEqual(metadata['full_window_label'], 'UNKNOWN')
        self.assertFalse(metadata['supervised_eligible'])

    def test_exact_ten_second_boundary_included_and_next_ns_opens_window(self):
        raw = [packet(ordinal=i+1, timestamp_ns=t) for i, t in enumerate([0, 10**10, 10**10+1])]
        _, result = self.run_packets(raw)
        self.assertEqual([t.metadata['real_packet_count'] for t in result], [2, 1])

    def test_large_benign_lateness_crossing_window_boundary_has_no_early_flush(self):
        raw = [packet(ordinal=1, timestamp_ns=13*10**9), packet(ordinal=2, timestamp_ns=0),
               packet(ordinal=3, timestamp_ns=10*10**9)]
        _, result = self.run_packets(raw, lateness=13*10**9)
        self.assertEqual([t.metadata['retained_ordinals'] for t in result], [[2, 3], [1]])

    def test_late_packet_after_earlier_output_fails_and_marks_all_output_provisional(self):
        raw = [packet(ordinal=i+1, timestamp_ns=t) for i, t in enumerate([0, 11*10**9, 22*10**9, 0])]
        engine = self.engine()
        stream = engine.run(map(extract_features, raw), input_complete=True)
        first = next(stream)
        self.assertTrue(first.metadata['processing_view']['provisional_until_stream_completion'])
        with self.assertRaisesRegex(ExtractionError, 'LATENESS_EXCEEDED'):
            list(stream)
        self.assertEqual(engine.status, 'FAILED')
        self.assertTrue((engine.directory/'FAILED.json').is_file())

    def test_heap_window_total_timeout_and_disk_bounds_fail_closed(self):
        cases = [
            ('heap', StreamingLimits(max_heap_packets=2), [0, 0, 0], 'HEAP_PACKET'),
            ('window', StreamingLimits(max_window_packets=2), [0, 1, 2], 'WINDOW_PACKET'),
            ('total', StreamingLimits(max_input_packets=2), [0, 1, 2], 'INPUT_PACKET'),
        ]
        for name, limits, times, error in cases:
            with self.subTest(name=name):
                raw = [packet(ordinal=i+1, timestamp_ns=t) for i, t in enumerate(times)]
                with self.assertRaisesRegex(ExtractionError, error):
                    self.run_packets(raw, name=name, limits=limits)
        engine = self.engine(name='timeout')
        def timed():
            engine.started -= 8000
            yield extract_features(packet())
        with self.assertRaisesRegex(ExtractionError, 'STREAM_TIMEOUT'):
            list(engine.run(timed(), input_complete=True))
        engine = self.engine(name='disk')
        with patch('nids_ml.flows.streaming.shutil.disk_usage', return_value=shutil_usage(0)):
            with self.assertRaisesRegex(ExtractionError, 'DISK_RESERVE'):
                list(engine.run([], input_complete=True))
        self.assertFalse(engine.directory.exists())

    def test_sqlite_page_cap_retains_failed_scratch(self):
        raw = (extract_features(packet(ordinal=i+1, timestamp_ns=i, flow=i)) for i in range(1000))
        engine = self.engine(limits=StreamingLimits(sqlite_bytes=65536))
        import sqlite3
        with self.assertRaises(sqlite3.DatabaseError):
            list(engine.run(raw, input_complete=True))
        self.assertTrue((engine.directory/'FAILED.json').exists())
        self.assertLessEqual((engine.directory/'window.sqlite').stat().st_size, 65536)

    def test_source_context_ordinal_and_malformed_feature_rejected(self):
        feature = extract_features(packet())
        bad = [replace(feature, context=replace(self.context, capture_id='other')),
               replace(feature, ordinal=True), replace(feature, timestamp_ns=-1),
               replace(feature, packet_sha256='bad'), replace(feature, features=(0.,)*11),
               replace(feature, unknown_protocol_tokens=('x'*257,))]
        for i, candidate in enumerate(bad):
            with self.subTest(i=i):
                engine = self.engine(name='bad'+str(i))
                with self.assertRaises((ValueError, TypeError)):
                    list(engine.run([candidate], input_complete=True))
        engine = self.engine(name='duplicate')
        with self.assertRaisesRegex(ExtractionError, 'SOURCE_ORDINAL'):
            list(engine.run([feature, feature], input_complete=True))

    def test_no_overwrite_no_replay_and_early_close(self):
        existing = self.parent/'existing'
        existing.mkdir()
        (existing/'keep.txt').write_text('keep')
        engine = self.engine(name='existing')
        with self.assertRaises(FileExistsError):
            list(engine.run([], input_complete=True))
        self.assertEqual((existing/'keep.txt').read_text(), 'keep')
        engine = self.engine()
        raw = [packet(ordinal=i+1, timestamp_ns=i*11*10**9) for i in range(4)]
        stream = engine.run(map(extract_features, raw), input_complete=True)
        next(stream)
        stream.close()
        self.assertEqual(engine.status, 'ABORTED')
        with self.assertRaisesRegex(ExtractionError, 'SINGLE_USE'):
            list(engine.run([], input_complete=True))

    def test_partial_empty_and_deterministic_random_order(self):
        engine = self.engine(name='partial')
        result = list(engine.run([extract_features(packet())], input_complete=False))
        self.assertEqual(engine.status, 'PARTIAL')
        self.assertFalse(result[0].metadata['full_window_complete'])
        empty = self.engine(name='empty')
        self.assertEqual(list(empty.run([], input_complete=True)), [])
        self.assertEqual(empty.status, 'COMPLETE')
        rng = random.Random(42)
        raw = [packet(ordinal=i+1, timestamp_ns=rng.randrange(1000), flow=i%5) for i in range(100)]
        _, a = self.run_packets(raw, name='a', lateness=1000)
        _, b = self.run_packets(raw, name='b', lateness=1000)
        self.assertEqual([(x.tensor_bytes, x.metadata_json) for x in a], [(x.tensor_bytes, x.metadata_json) for x in b])

    def test_limit_validation_rejects_bool_out_of_bounds_and_legacy_unchanged(self):
        for limits in [StreamingLimits(max_heap_packets=True), StreamingLimits(sqlite_bytes=65537),
                       StreamingLimits(max_input_packets=30_000_001)]:
            with self.assertRaises(ExtractionError):
                limits.validate()
        with self.assertRaises(ExtractionError):
            self.engine(lateness=True)
        from nids_ml.flows.extractor import ResourceLimits
        with self.assertRaises(ExtractionError):
            ResourceLimits(max_input_packets=30_000_000).validate()

    def test_memory_does_not_grow_with_total_stream_length(self):
        import tracemalloc
        feature = extract_features(packet())
        peaks = []
        for count in (1000, 10000):
            engine = self.engine(name='memory'+str(count))
            def source():
                for i in range(count):
                    ts = (i//100)*11*10**9 + i%100
                    yield replace(feature, ordinal=i+1, timestamp_ns=ts,
                                  features=(ts/1e9, *feature.features[1:]))
            tracemalloc.start()
            try:
                for tensor in engine.run(source(), input_complete=True):
                    self.assertEqual(tensor.shape, (20, 11, 1))
                peaks.append(tracemalloc.get_traced_memory()[1])
            finally:
                tracemalloc.stop()
            self.assertLessEqual(engine.stats['peak_heap_packets'], 2)
            self.assertEqual(engine.stats['peak_window_packets'], 100)
        self.assertLess(max(peaks), 2*1024**2)

    def test_cli_synthetic_stream_runs_and_rejects_reusing_scratch(self):
        import subprocess
        fixture = self.parent/'fixture.json'
        fixture.write_text(json.dumps(bundle()), encoding='utf-8')
        command = [sys.executable, '-B', str(ROOT/'scripts/validate_flow_extraction.py'),
                   'validate-stream', '--fixture', str(fixture), '--scratch', str(self.parent/'cli'),
                   '--max-lateness-ns', '0']
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['status'], 'PASS_SYNTHETIC_STREAM_ONLY')
        self.assertEqual(report['stats']['input_packets'], 5)
        self.assertFalse(report['training'])
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)['status'], 'FAIL_CLOSED')


def shutil_usage(free):
    from collections import namedtuple
    return namedtuple('usage', 'total used free')(10**12, 10**12-free, free)


if __name__ == '__main__':
    unittest.main()
