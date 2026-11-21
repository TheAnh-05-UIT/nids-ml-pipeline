"""Causal split regressions using synthetic HDF5, never real dataset metrics."""
from contextlib import closing
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'tests'))
import test_scenario_catalog as fixtures
from nids_ml.data.temporal_catalog import build_temporal_catalog, identity, row_split, POLICY
from nids_ml.flows.mapped import readonly
from nids_ml.labeling.contracts import canonical_bytes
from nids_ml.pcap.decoder import file_sha256
from nids_ml.storage.partitioned import TemporalScenarioPartitionReader, ScenarioPartitionReader, PartitionReader


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ScenarioCatalogTests()
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.work = self.fixture.work

    def sources(self, mode='base', *, adjust=None, test_start=10000, train_times=None):
        original = fixtures.packet
        original_extract = fixtures.extract_features
        def packet(**kwargs):
            source = kwargs['capture']['capture_id'].removeprefix('capture-')
            rule = next(r for r in self.fixture.policy['sources'] if r['source_id'] == source)
            i = kwargs['ordinal']; n = rule['records']
            if rule['publisher_split'] == 'test':
                seconds = test_start + (i - 1) * 100
            else:
                seconds = ((n + i) if i > n - 2 else i - 1) * 100
                if train_times:
                    seconds = train_times[i - 1]
            kwargs['timestamp_ns'] = seconds * 10**9
            return original(**kwargs)
        def extract(value, **kwargs):
            if adjust:
                adjust(value)
            return original_extract(value, **kwargs)
        with patch.object(fixtures, 'packet', side_effect=packet), patch.object(fixtures, 'extract_features', side_effect=extract):
            self.fixture.sources(mode)
        self.directories = self.fixture.directories
        self.policy_path = self.fixture.policy_path; self.policy_sha = self.fixture.policy_sha

    def build(self, name='temporal', **kwargs):
        return build_temporal_catalog(self.directories, self.work, self.work / 'artifacts/runs' / name,
            policy_path=self.policy_path, policy_sha=self.policy_sha, **kwargs)

    def reader(self, name='temporal', **kwargs):
        return TemporalScenarioPartitionReader(self.work / 'artifacts/runs' / name, self.work,
            policy_path=self.policy_path, policy_sha=self.policy_sha, **kwargs)

    def rewrite_index(self, sql, args=()):
        output = self.work / 'artifacts/runs/temporal'
        with closing(sqlite3.connect(output / 'catalog.sqlite')) as db:
            db.execute(sql, args); db.commit()
        path = output / 'manifest.json'; m = json.loads(path.read_bytes())
        m['index']['sha256'] = file_sha256(output / 'catalog.sqlite')
        m['catalog_id'] = identity(m); path.write_bytes(canonical_bytes(m))

    def test_causal_counts_shared_evaluation_and_actual_hdf5_lineage(self):
        self.sources(); m = self.build(); reader = self.reader()
        self.assertEqual(m['split_policy'], POLICY)
        self.assertFalse(m['parent_connection_independence'])
        self.assertEqual(m['split_counts'], dict(train={'0': 4, '1': 8}, validation={'0': 2, '1': 4}, test={'0': 6, '1': 12}))
        for client in (None, 0, 1):
            for split in ('train', 'validation', 'test'):
                with closing(reader.batches(split, client, batch_size=2)) as batches:
                    labels = [str(int(y)) for b in batches for y in b['y'][:, 0]]
                self.assertEqual({y: labels.count(y) for y in set(labels)}, reader.counts(split, client))
        self.assertEqual(m['statistics']['retained_samples'], 36)

    def test_shared_parents_allowed_between_splits_but_train_clients_disjoint(self):
        self.sources('overlap'); m = self.build()
        with readonly(self.reader().index) as db:
            self.assertTrue(db.execute('SELECT group_id FROM samples GROUP BY group_id HAVING count(DISTINCT split)>1').fetchall())
            self.assertFalse(db.execute("SELECT group_id FROM samples WHERE split='train' GROUP BY group_id HAVING count(DISTINCT client)>1").fetchall())
        self.assertEqual(m['split_counts']['train']['0'], 4)
        with self.assertRaises(ValueError):
            self.fixture.build('old-connection')

    def test_connection_readers_reject_temporal_format(self):
        self.sources(); self.build(); output = self.work / 'artifacts/runs/temporal'
        with self.assertRaisesRegex(ValueError, 'READY_NIDS_CATALOG_REQUIRED'):
            PartitionReader(output, self.work)
        with self.assertRaisesRegex(ValueError, 'READY_NIDS_CATALOG_REQUIRED'):
            ScenarioPartitionReader(output, self.work, policy_path=self.policy_path, policy_sha=self.policy_sha)

    def test_validation_embargo_is_applied_without_reassigning_test(self):
        self.sources(train_times=[0, 100, 200, 850, 900, 1000]); m = self.build()
        self.assertEqual(m['statistics']['excluded_time_embargo_rows'], 3)
        self.assertEqual(m['split_counts']['test'], {'0': 6, '1': 12})
        self.assertEqual(m['split_counts']['train'], {'0': 3, '1': 6})

    def test_test_embargo_and_causal_source_order(self):
        self.sources(test_start=1220); m = self.build()
        self.assertEqual(m['statistics']['excluded_time_embargo_rows'], 3)
        self.assertEqual(m['split_counts']['validation'], {'0': 1, '1': 2})

    def test_noncausal_captures_fail_and_keep_failed_artifact(self):
        self.sources(test_start=500)
        with self.assertRaisesRegex(ValueError, 'TEMPORAL_SOURCE_TIME_ORDER'):
            self.build()
        output = self.work / 'artifacts/runs/temporal'
        self.assertTrue((output / 'FAILED.json').exists())
        self.assertFalse((output / 'manifest.json').exists())

    def test_exact_duplicate_prefers_test_over_train_for_same_parent(self):
        def adjust(value):
            if value['capture']['capture_id'] == 'capture-family-0-test' and value['ordinal'] == 1:
                value['tcp']['ack'] = 9900
        self.sources('overlap', adjust=adjust); m = self.build()
        self.assertEqual(m['statistics']['excluded_exact_duplicate_rows'], 1)
        self.assertEqual(m['split_counts']['train']['0'], 3)
        self.assertEqual(m['split_counts']['test']['0'], 6)
        with readonly(self.reader().index) as db:
            self.assertFalse(db.execute('SELECT sha FROM samples GROUP BY sha HAVING count(*)>1').fetchall())

    def test_cross_parent_duplicate_patterns_are_excluded(self):
        def adjust(value):
            if value['capture']['capture_id'] == 'capture-family-0-test' and value['ordinal'] == 1:
                value['tcp']['ack'] = 100
        self.sources(adjust=adjust); m = self.build()
        self.assertEqual(m['statistics']['excluded_pattern_rows'], 2)
        self.assertEqual(m['split_counts']['test']['0'], 5)

    def test_mixed_parent_label_witness_is_not_hidden_by_pattern_filter(self):
        self.sources('exclusions'); m = self.build()
        self.assertGreater(m['statistics']['excluded_mixed_label_groups'], 0)
        with readonly(self.reader().index) as db:
            self.assertFalse(db.execute('SELECT group_id FROM samples GROUP BY group_id HAVING min(y)!=max(y)').fetchall())

    def test_gap_covers_lateness_and_identity_binds_protocol_parameters(self):
        self.sources(); m = self.build()
        changed = dict(m, temporal=dict(m['temporal'], gap_seconds=31))
        self.assertNotEqual(identity(m), identity(changed))
        from nids_ml.data.temporal_catalog import parameters
        policy = dict(sources=[dict(max_lateness_ns=13*10**9)])
        self.assertEqual(parameters(policy, 23)['gap_seconds'], 23)
        with self.assertRaisesRegex(ValueError, 'WINDOW_PLUS_LATENESS'):
            parameters(policy, 22)

    def test_embargo_boundary_uses_full_window_not_first_packet(self):
        bound = dict(validation_start_ns=100*10**9, test_start_ns=200*10**9)
        self.assertEqual(row_split('train', 60*10**9, 70*10**9, bound, 30*10**9), 'train')
        self.assertIsNone(row_split('train', 65*10**9, 75*10**9, bound, 30*10**9))
        self.assertEqual(row_split('train', 100*10**9, 110*10**9, bound, 30*10**9), 'validation')
        self.assertIsNone(row_split('train', 165*10**9, 175*10**9, bound, 30*10**9))

    def test_gap_seed_missing_sources_and_policy_fail_before_output(self):
        self.sources()
        for kwargs in (dict(gap_seconds=9), dict(gap_seconds=True), dict(seed=True), dict(gap_seconds=3601)):
            with self.assertRaises(ValueError):
                self.build(**kwargs)
            self.assertFalse((self.work / 'artifacts/runs/temporal').exists())
        self.directories = self.directories[:-1]
        with self.assertRaisesRegex(ValueError, 'SIX_SOURCES'):
            self.build()

    def test_deterministic_input_order_and_no_overwrite(self):
        self.sources(); first = self.build()
        self.directories.reverse(); second = self.build('reverse')
        self.assertEqual(first['catalog_id'], second['catalog_id'])
        with self.assertRaises(ValueError):
            self.build()

    def test_forged_split_and_timestamp_fail_with_refreshed_index_hash(self):
        self.sources(); self.build()
        self.rewrite_index("UPDATE samples SET split='train',client=0 WHERE id=(SELECT min(id) FROM samples WHERE split='validation')")
        with self.assertRaisesRegex(ValueError, 'TEMPORAL_EMBARGO_OR_CAUSAL_VIOLATION'):
            self.reader()

    def test_timestamp_index_must_match_original_hdf5(self):
        self.sources(); self.build()
        self.rewrite_index("UPDATE samples SET first_ns=first_ns+1,last_ns=last_ns+1 WHERE split='train'")
        reader = self.reader()
        with self.assertRaisesRegex(ValueError, 'TEMPORAL_ROW_METADATA_LINEAGE'):
            list(reader.batches('train'))

    def test_portable_reader_and_source_tamper_guard(self):
        self.sources(); m = self.build()
        sources = {s['dataset_id']: self.work / s['path_hint'] for s in m['sources']}
        reader = self.reader(sources=sources)
        self.assertEqual(sum(len(b['X']) for b in reader.batches('test')), 18)
        path = reader.dataset(next(iter(reader.readers))).identities[0][0]
        with path.open('ab') as stream:
            stream.write(b'tamper')
        with self.assertRaises(ValueError):
            list(reader.batches('test'))

    def test_cli_build_inspect_and_no_training(self):
        self.sources(); output = self.work / 'artifacts/runs/temporal'
        from nids_ml.cli.large_training import main
        common = ['--workspace', str(self.work), '--policy', str(self.policy_path), '--policy-sha256', self.policy_sha]
        captured = io.StringIO()
        with redirect_stdout(captured):
            status = main(['scenario-temporal-catalog', '--dataset', *map(str, self.directories),
                '--output', str(output), *common])
        self.assertEqual(status, 0, captured.getvalue())
        captured = io.StringIO()
        with redirect_stdout(captured):
            status = main(['scenario-temporal-inspect', '--catalog', str(output), *common])
        self.assertEqual(status, 0, captured.getvalue())
        report = json.loads(captured.getvalue())
        self.assertEqual(report['status'], 'PASS_TEMPORAL_SCENARIO_CATALOG_INSPECTION')
        self.assertFalse(report['training'])
        self.assertFalse(report['parent_connection_independence'])
        self.assertEqual(report['gap_seconds'], 30)


if __name__ == '__main__':
    unittest.main()
