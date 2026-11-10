"""Test bounded CSV EDA using small synthetic datasets.
Test EDA CSV có giới hạn bằng dữ liệu synthetic nhỏ.
"""
from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
from nids_ml.cli.eda import main
from nids_ml.validation.workspace import workspace_path
from nids_ml.eda.csv_eda import EdaOptions, analyze_csv, pearson
from nids_ml.data.inventory import discover, fingerprint


class SyntheticEdaTests(unittest.TestCase):
    def setUp(self):
        evidence = ROOT.parent / 'task-arm-010/evidence'
        self.tmp = tempfile.TemporaryDirectory(prefix='data001-fixture-', dir=evidence)
        self.base = Path(self.tmp.name)
        self.assertTrue(self.base.resolve().is_relative_to(evidence.resolve()))
        self.addCleanup(self.tmp.cleanup)
        self.ws = self.base / 'synthetic-workspace'
        for path in ('data/raw', 'artifacts/runs', 'configs'):
            (self.ws / path).mkdir(parents=True, exist_ok=True)
        cfg = {'schema_version': '1.0.0', 'workspace_root': '.', 'dataset': {'id': 'synthetic-fixture'},
               'paths': {'raw_data': 'data/raw', 'runs': 'artifacts/runs'}}
        (self.ws / 'configs/workspace.json').write_text(json.dumps(cfg))
        (self.ws / 'configs/eda.json').write_text(json.dumps({'schema_version': '1.0.0',
            'encoding': 'utf-8-sig', 'limits': {'sample_rows': 4, 'sample_cells': 128, 'temp_db_mib': 8}}))
    def write(self, body, name='synthetic.csv'):
        path = self.ws / 'data/raw' / name
        path.write_text(body, encoding='utf-8', newline='')
        return path
    def analyze(self, body, labels=('label',), **limits):
        path = self.write(body)
        return analyze_csv(path, labels, self.base, EdaOptions(**limits))
    def call(self, *args):
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = main(['--workspace', str(self.ws), *args])
        return code, json.loads(stream.getvalue())
    def fixture(self):
        return 'a,b,Attack_label,Attack_type\n1,2,0,toy_normal\n2,4,1,toy_attack\n2,4,1,toy_attack\n3,6,1,toy_attack\n'

    def test_counts_features_labels_duplicates_and_imbalance(self):
        report = self.analyze(self.fixture(), ('Attack_label', 'Attack_type'))
        self.assertEqual((report['row_count'], report['column_count']), (4, 4))
        self.assertEqual(report['raw_feature_columns'], ['a', 'b'])
        self.assertIsNone(report['selected_feature_columns'])
        self.assertEqual(report['duplicates']['duplicate_rows_beyond_first'], 1)
        self.assertEqual(report['labels']['Attack_type']['max_to_min_class_ratio'], 3.)
        self.assertEqual(report['correlation']['pairs'][0]['pearson'], 1.)

    def test_missing_nan_inf_and_overflow_spelling(self):
        report = self.analyze('a,b,label\n,NaN,normal\nNULL,Inf,attack\n-infinity,word,attack\n1e309,-Inf,attack\n')
        self.assertEqual(report['columns'][0]['cell_counts']['blank'], 1)
        self.assertEqual(report['columns'][0]['cell_counts']['null_token'], 1)
        self.assertEqual(report['columns'][0]['cell_counts']['infinity'], 2)
        self.assertEqual(report['columns'][1]['cell_counts']['nan'], 1)
        json.dumps(report, allow_nan=False)

    def test_conflicting_labels_for_identical_features(self):
        report = self.analyze('x,label\n1,A\n1,B\n1,A\n')
        self.assertEqual(report['duplicates']['duplicate_rows_beyond_first'], 1)
        self.assertEqual(report['duplicates']['feature_only_duplicates_beyond_first'], 2)
        self.assertEqual(report['duplicates']['conflicting_feature_groups'], 1)

    def test_other_label_like_columns_are_review_candidates(self):
        report = self.analyze('x,Attack_label,Attack_type\n1,0,toy_normal\n2,1,toy_attack\n', ('Attack_type',))
        candidate = next(row for row in report['leakage_candidates'] if row['column'] == 'Attack_label')
        self.assertIn('possible_undeclared_target_column; review_label_columns', candidate['reasons'])
        self.assertIsNone(report['selected_feature_columns'])

    def test_strong_sample_label_association_is_not_auto_selection(self):
        body = 'opaque_feature,label\n' + '0,0\n'*3 + '1,1\n'*3
        report = self.analyze(body)
        self.assertIn('strong_sample_label_association_not_proof', report['leakage_candidates'][0]['reasons'])
        self.assertIsNone(report['selected_feature_columns'])

    def test_zero_spelling_is_not_canonicalized(self):
        body = 'mqtt.topic,label\n0,normal\n0.0,attack\n0,normal\n0.0,attack\n0,normal\n0.0,attack\n'
        report = self.analyze(body)
        tokens = report['columns'][0]['distribution']['top_tokens']
        self.assertEqual({x['token_or_long_value_hash'] for x in tokens}, {'0', '0.0'})
        self.assertIn('both_zero_spellings_present_in_sample_not_proof_of_leakage', report['leakage_candidates'][0]['reasons'])

    def test_sampling_is_deterministic_and_cell_bounded(self):
        body = 'x,y,label\n' + ''.join(f'{i},{2*i},A\n' for i in range(70))
        first = self.analyze(body, sample_rows=30, sample_cells=12)
        second = self.analyze(body, sample_rows=30, sample_cells=12)
        self.assertEqual(first, second)
        self.assertEqual(first['sampling']['retained_rows'], 4)
        self.assertEqual(first['row_count'], 70)

    def test_input_is_not_changed(self):
        path = self.write('x,label\n1,A\n2,B\n')
        before = path.read_bytes()
        report = analyze_csv(path, ['label'], self.base)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(report['input']['sha256'], hashlib.sha256(before).hexdigest())

    def test_no_fixed_feature_count(self):
        report = self.analyze('a,b,c,d,label\n1,2,3,4,A\n')
        self.assertEqual(len(report['raw_feature_columns']), 4)

    def test_malformed_row_width_is_rejected(self):
        with self.assertRaises(ValueError):
            self.analyze('x,label\n1,A,extra\n')

    def test_duplicate_header_is_rejected(self):
        with self.assertRaises(ValueError):
            self.analyze('x,x,label\n1,2,A\n')

    def test_missing_requested_label_is_rejected(self):
        with self.assertRaises(ValueError):
            self.analyze('x,not_the_label\n1,A\n')

    def test_oversize_record_is_rejected(self):
        with self.assertRaises(ValueError):
            self.analyze('x,label\n' + 'a'*50 + ',A\n', max_record_chars=32)

    def test_multiline_csv_is_counted_as_records(self):
        report = self.analyze('x,label\n"hello\nworld",A\nx,B\n')
        self.assertEqual(report['row_count'], 2)

    def test_multiline_record_limit_is_not_only_line_limit(self):
        with self.assertRaises(ValueError):
            self.analyze('x,label\n"'+'a'*14+'\n'+'b'*14+'",A\n', max_record_chars=32)

    def test_malformed_quoting_is_rejected(self):
        with self.assertRaises(csv.Error):
            self.analyze('x,label\n"unfinished,A\n')

    def test_limits_are_validated(self):
        with self.assertRaises(ValueError):
            self.analyze('x,label\n1,A\n', sample_rows=0)

    def test_empty_file_is_not_zero_row_dataset(self):
        with self.assertRaises(ValueError):
            self.analyze('')

    def test_header_only_reports_empty_without_fake_constants(self):
        report = self.analyze('x,label\n')
        self.assertTrue(report['quality_flags']['empty_dataset'])
        self.assertIsNone(report['columns'][0]['constant_raw_text'])

    def test_large_numeric_values_do_not_emit_nonfinite_JSON(self):
        report = self.analyze('x,label\n1e308,A\n-1e308,A\n1e308,A\n')
        self.assertTrue(report['columns'][0]['numeric_moment_overflow'])
        self.assertIsNone(report['columns'][0]['numeric_mean'])
        json.dumps(report, allow_nan=False)

    def test_correlation_budget_and_constant_case(self):
        report = self.analyze('x,y,label\n1,2,0\n2,4,1\n3,6,0\n', correlation_features=1)
        self.assertEqual(report['correlation']['omitted_numeric_features'], ['y'])
        self.assertIsNone(pearson([(1., 2.), (1., 3.), (1., 4.)])['pearson'])

    def test_label_output_limit_is_explicit(self):
        report = self.analyze('x,label\n1,A\n2,B\n3,C\n', label_report_limit=1)
        self.assertFalse(report['labels']['label']['distribution_complete'])
        self.assertEqual(report['labels']['label']['omitted_count'], 2)

    def test_SQLite_disk_cap_fails_without_unbounded_state(self):
        body = 'x,label\n' + ''.join(f'{i},A\n' for i in range(18000))
        with self.assertRaises(sqlite3.Error):
            self.analyze(body, temp_db_mib=1, sample_rows=2)

    def test_discovery_empty_is_blocked_with_null_statistics(self):
        code, report = self.call('discover')
        self.assertEqual(code, 3)
        self.assertEqual(report['status'], 'BLOCKED_ON_DATASET')
        self.assertIsNone(report['dataset_statistics'])

    def test_inventory_hash_and_entry_budget(self):
        path = self.write('x,label\n1,A\n')
        report = discover(self.ws)
        self.assertEqual(report['files'][0]['sha256'], fingerprint(path)['sha256'])
        self.write('x,label\n2,B\n', 'second.csv')
        with self.assertRaises(ValueError):
            discover(self.ws, max_entries=1)

    def test_discovery_uses_configured_raw_directory(self):
        alternate = self.ws / 'data/alternate-raw'
        alternate.mkdir()
        (alternate / 'synthetic.csv').write_text('x,label\n1,A\n')
        cfg_path = self.ws / 'configs/workspace.json'
        cfg = json.loads(cfg_path.read_text())
        cfg['paths']['raw_data'] = 'data/alternate-raw'
        cfg_path.write_text(json.dumps(cfg))
        code, report = self.call('discover')
        self.assertEqual(code, 0)
        self.assertEqual(report['raw_directory'], 'data/alternate-raw')

    def test_source_credentials_are_not_reported_or_used(self):
        self.write('x,label\n1,A\n')
        code, report = self.call('analyze', '--input', 'data/raw/synthetic.csv', '--label-column', 'label',
            '--dataset-version', 'fixture-v1', '--source-uri', 'https://user:never-use-this@example.invalid/data',
            '--run-id', 'credential-fixture')
        self.assertEqual(code, 2)
        self.assertNotIn('never-use-this', json.dumps(report))
        self.assertFalse((self.ws / 'artifacts/runs/credential-fixture').exists())

    def test_path_escape_is_rejected(self):
        for value in ('../outside.csv', '/tmp/out.csv', 'C:/out.csv', 'data\\raw\\file.csv'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                workspace_path(self.ws, value)

    def test_cli_fixture_is_named_and_existing_run_is_preserved(self):
        self.write(self.fixture())
        args = ('analyze', '--input', 'data/raw/synthetic.csv', '--label-column', 'Attack_label',
                '--label-column', 'Attack_type', '--dataset-version', 'fixture-v1',
                '--source-uri', 'synthetic://unit-tests', '--run-id', 'fixture-run', '--input-kind', 'synthetic_fixture')
        code, _ = self.call(*args)
        self.assertEqual(code, 0)
        out = self.ws / 'artifacts/runs/fixture-run'
        report = json.loads((out / 'data_quality.json').read_text())
        self.assertEqual(report['input']['dataset_id'], 'synthetic-fixture')
        self.assertEqual(report['input']['input_kind'], 'synthetic_fixture')
        before = {p.name: p.read_bytes() for p in out.iterdir()}
        code, _ = self.call(*args)
        self.assertEqual(code, 2)
        self.assertEqual(before, {p.name: p.read_bytes() for p in out.iterdir()})

    def test_cli_parse_error_has_no_complete_marker(self):
        self.write('x,label\n1,A,extra\n')
        code, report = self.call('analyze', '--input', 'data/raw/synthetic.csv', '--label-column', 'label',
            '--dataset-version', 'fixture-v1', '--source-uri', 'synthetic://unit-tests',
            '--run-id', 'failed-fixture', '--input-kind', 'synthetic_fixture')
        self.assertEqual(code, 2)
        out = self.ws / 'artifacts/runs/failed-fixture'
        self.assertTrue((out / 'failure.json').exists())
        self.assertFalse((out / 'COMPLETE.json').exists())


if __name__ == '__main__':
    unittest.main()
