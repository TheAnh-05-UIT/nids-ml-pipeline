"""Test CSV spelling, label and port-quality counters with synthetic fixtures.
Test bộ đếm spelling, nhãn và chất lượng port bằng fixture synthetic.
"""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from nids_ml.eda.quality_detail import FORMAT_COLUMNS, audit_quality, port_kind


class QualityDetailTests(unittest.TestCase):
    def test_port_kinds(self):
        for raw in ('0', '65535', '80.0'):
            self.assertEqual(port_kind(raw), 'in_range_integer')
        for raw, kind in (('-1', 'out_of_range'), ('65536', 'out_of_range'),
                          ('1.5', 'noninteger'), ('NaN', 'nonfinite'), ('text', 'non_numeric_text')):
            self.assertEqual(port_kind(raw), kind)

    def test_full_counts_exact_spelling_and_input_unchanged(self):
        self.with_csv(False)

    def test_wrong_width_is_rejected(self):
        self.with_csv(True)

    def with_csv(self, wrong_width):
        parent = ROOT.parent / 'task-arm-010/evidence'
        with tempfile.TemporaryDirectory(prefix='detail-fixture-', dir=parent) as tmp:
            self.assertTrue(Path(tmp).resolve().is_relative_to(parent.resolve()))
            path = Path(tmp) / 'synthetic.csv'
            header = [*FORMAT_COLUMNS, 'tcp.srcport', 'tcp.dstport', 'Attack_label', 'Attack_type', 'frame.time']
            rows = [(['0'] * 4 + ['80', '443', '1', 'Attack', '1']),
                    (['0.0'] * 4 + ['text', '-1', '0', 'Normal', '192.0.2.1']),
                    (['other'] * 4 + ['5.5', 'Inf', '1', 'Normal', 'date text'])]
            with path.open('w', newline='', encoding='utf-8') as stream:
                csv.writer(stream).writerows([header] + rows + ([['bad']] if wrong_width else []))
            before = path.read_bytes()
            if wrong_width:
                with self.assertRaisesRegex(ValueError, 'width'):
                    audit_quality(path)
            else:
                r = audit_quality(path)
                self.assertEqual(r['row_count'], 3)
                self.assertEqual(r['formatting'][FORMAT_COLUMNS[0]]['0']['by_attack_type'], {'Attack': 1})
                self.assertEqual(r['formatting'][FORMAT_COLUMNS[0]]['0.0']['by_attack_type'], {'Normal': 1})
                self.assertEqual(r['binary_vs_Normal_rule_mismatch_count'], 1)
                self.assertEqual(r['row_order']['total_label_runs'], 2)
                self.assertEqual(r['row_order']['longest_label_run'], 2)
                self.assertEqual(r['frame_time_lexical_kinds']['ip_literal']['count'], 1)
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
