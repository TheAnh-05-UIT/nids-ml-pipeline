"""Test multi-capture quality, diagnostic matching and audit reuse.
Test chất lượng nhiều capture, diagnostic matching và tái dùng audit.
"""
import csv
import hashlib
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from nids_ml.alignment.candidates import AlignmentError
from nids_ml.alignment.io import pcap_scalar_prefix, AUDIT_SCALAR_FIELDS
from nids_ml.alignment.multicapture import audit_prefix, diagnostic_record, scan_csv_quality
from nids_ml.cli.multicapture import main, validate_config, run_audit
from nids_ml.pcap.metadata import scan_record_headers


def packet(i=1, protocol=6, kind='pcap'):
    row = dict(**{'frame.number': str(i), 'frame.time_epoch': f'{1609459200+i}.000000001',
        'ip.src': '192.0.2.1', 'ip.dst': '192.0.2.2', 'ip.proto': str(protocol),
        'ip.flags.mf': '0', 'ip.frag_offset': '0', 'tcp.srcport': str(40000+i),
        'tcp.dstport': '80', 'tcp.len': '0', 'tcp.flags': '0x0002',
        'tcp.ack_raw': '0', 'tcp.checksum': '0x1234', 'icmp.checksum': '0', 'icmp.seq_le': '0'})
    if protocol == 1:
        row.update({'tcp.srcport': '0', 'tcp.dstport': '0',
                    'icmp.checksum': '0x1234', 'icmp.seq_le': str(i)})
    if kind == 'csv':
        row['ip.src_host'] = row.pop('ip.src')
        row['ip.dst_host'] = row.pop('ip.dst')
        row['frame.time'] = f'2021 01:00:{i:02d}.000000001'
        row['Attack_label'], row['Attack_type'] = '1', 'DDoS_ICMP'
    return row


def record(row, kind='pcap', ordinal=1):
    return diagnostic_record(row, kind=kind, source_sha256=('a' if kind == 'pcap' else 'b')*64,
                             ordinal=ordinal)


class DiagnosticKeyTests(unittest.TestCase):
    def test_tcp_compatible_shared_values(self):
        self.assertEqual(record(packet()).key, record(packet(kind='csv'), 'csv').key)

    def test_icmp_shared_values(self):
        self.assertEqual(record(packet(protocol=1)).key,
                         record(packet(protocol=1, kind='csv'), 'csv').key)

    def test_labels_do_not_select_protocol_or_match(self):
        row = packet(protocol=1, kind='csv')
        first = record(row, 'csv')
        row.update(Attack_label='0', Attack_type='Normal')
        self.assertEqual(first, record(row, 'csv'))

    def test_tcp_checksum_change_changes_key(self):
        row = packet(); first = record(row)
        row['tcp.checksum'] = '0xabcd'
        self.assertNotEqual(first.key, record(row).key)

    def test_direction_is_not_reversed_to_force_match(self):
        row = packet(protocol=1)
        row['ip.src'], row['ip.dst'] = row['ip.dst'], row['ip.src']
        self.assertNotEqual(record(packet(protocol=1)).key, record(row).key)

    def test_udp_no_invented_directional_ports(self):
        self.assertIsNone(record(packet(protocol=17)).key)

    def test_fragments_not_matched(self):
        for field in ('ip.flags.mf', 'ip.frag_offset'):
            row = packet(); row[field] = '1'
            self.assertIsNone(record(row).key)

    def test_tshark_boolean_fragment_flag(self):
        for protocol in (6, 1):
            row = packet(protocol=protocol); row['ip.flags.mf'] = 'False'
            self.assertIsNotNone(record(row).key)
            row['ip.flags.mf'] = 'True'
            self.assertIsNone(record(row).key)

    def test_invalid_boolean_flag_fails_closed(self):
        row = packet(); row['ip.flags.mf'] = 'not-a-boolean'
        self.assertIsNone(record(row).key)

    def test_multi_protocol_ambiguous_csv(self):
        row = packet(kind='csv'); row['icmp.checksum'] = '1'
        self.assertIsNone(record(row, 'csv').key)

    def test_missing_or_nan_icmp_fields_fail_closed(self):
        for value in ('NaN', '', '0x10000', '1;2'):
            row = packet(protocol=1); row['icmp.checksum'] = value
            self.assertIsNone(record(row).key)

    def test_invalid_ip_not_repaired(self):
        row = packet(protocol=1, kind='csv'); row['ip.dst_host'] = '0'
        self.assertIsNone(record(row, 'csv').key)

    def test_invalid_kind_hash_and_ordinal_rejected(self):
        with self.assertRaises(AlignmentError): record(packet(), 'bad')
        with self.assertRaises(AlignmentError): record(packet(), ordinal=True)
        with self.assertRaises(AlignmentError):
            diagnostic_record(packet(), kind='pcap', source_sha256='invalid', ordinal=1)

    def test_duplicate_is_ambiguous_not_first_match(self):
        result = audit_prefix([packet(protocol=1)],
                              [packet(protocol=1, kind='csv'), packet(protocol=1, kind='csv')],
                              pcap_sha256='a'*64, csv_sha256='b'*64)
        self.assertEqual(result['content']['counts'], {'AMBIGUOUS': 1})
        self.assertFalse(result['supervised_eligible'])

    def test_unknowns_retained_and_results_deterministic(self):
        args = dict(pcap_sha256='a'*64, csv_sha256='b'*64)
        result = audit_prefix([packet(protocol=17)], [packet(kind='csv')], **args)
        self.assertEqual(result, audit_prefix([packet(protocol=17)], [packet(kind='csv')], **args))
        self.assertEqual(result['content']['counts'], {'UNSUPPORTED_OR_INVALID': 1})
        self.assertFalse(result['label_assignment_performed'])
        self.assertEqual(result['global_PCAP_uniqueness'], 'NOT_VERIFIED')

    def test_prefix_bounds_rejected(self):
        with self.assertRaises(AlignmentError):
            audit_prefix([packet()]*257, [], pcap_sha256='a'*64, csv_sha256='b'*64)

    def test_unapproved_scalar_fields_fail_before_decoder(self):
        with self.assertRaises(AlignmentError):
            pcap_scalar_prefix('missing', source_sha256='a'*64, size_bytes=1, tshark='missing',
                               tshark_sha256='b'*64, scratch='missing', fields=('tcp.payload',))


class CSVQualityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='csv-audit-synthetic-')
        self.path = Path(self.temp.name) / 'fixture.csv'

    def tearDown(self): self.temp.cleanup()

    def write(self, rows, header=('frame.time', 'Attack_label', 'Attack_type')):
        with self.path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f); writer.writerow(header); writer.writerows(rows)

    def scan(self):
        data = self.path.read_bytes()
        return scan_csv_quality(self.path, expected_sha256=hashlib.sha256(data).hexdigest(),
                                expected_size=len(data))

    def test_full_counts_labels_not_ground_truth(self):
        self.write([['2021 01:00:01', '0', 'Normal'], ['2021 01:00:02', '1', 'DDoS_TCP']])
        quality, prefix = self.scan()
        self.assertEqual(quality['rows'], 2)
        self.assertEqual(quality['valid_frame_time_rows'], 2)
        self.assertFalse(quality['labels_verified'])
        self.assertEqual(len(prefix), 2)

    def test_bad_width_not_silently_zipped(self):
        self.write([['2021 01:00:01', '0'], ['2021 01:00:02', '1', 'DDoS_TCP', 'extra']])
        quality, prefix = self.scan()
        self.assertEqual(quality['wrong_width_rows'], 2)
        self.assertEqual(prefix, [{}, {}])
        self.assertEqual(quality['dataset_provided_label_counts'], [])

    def test_invalid_time_not_shifted(self):
        self.write([['192.0.2.1', '1', 'DDoS_UDP']])
        quality, prefix = self.scan()
        self.assertEqual(quality['invalid_frame_time_rows'], 1)
        self.assertEqual(prefix[0]['frame.time'], '192.0.2.1')
        self.assertFalse(quality['column_repair_performed'])

    def test_regression_reported_without_sort(self):
        self.write([['2021 01:00:02', '0', 'Normal'], ['2021 01:00:01', '0', 'Normal']])
        quality, prefix = self.scan()
        self.assertEqual(quality['time_of_day_regressions'], 1)
        self.assertEqual(prefix[0]['frame.time'], '2021 01:00:02')

    def test_hash_mismatch_rejected(self):
        self.write([['2021 01:00:01', '0', 'Normal']])
        with self.assertRaises(AlignmentError):
            scan_csv_quality(self.path, expected_sha256='f'*64, expected_size=self.path.stat().st_size)

    def test_duplicate_header_rejected(self):
        self.write([], header=('frame.time', 'Attack_label', 'Attack_type', 'Attack_type'))
        with self.assertRaises(AlignmentError): self.scan()

    def test_multiline_csv_record_counted_once(self):
        self.write([['2021 01:00:01', '0', 'Normal', 'line1\nline2']],
                   header=('frame.time', 'Attack_label', 'Attack_type', 'payload_fixture'))
        quality, _ = self.scan()
        self.assertEqual(quality['rows'], 1)


class PCAPMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='pcap-audit-synthetic-')
        self.path = Path(self.temp.name) / 'fixture.pcap'

    def tearDown(self): self.temp.cleanup()

    def write(self, fractions=(1, 3, 2), endian='<', nanos=False):
        magic = ({('<', False): b'\xd4\xc3\xb2\xa1', ('>', False): b'\xa1\xb2\xc3\xd4',
                  ('<', True): b'\x4d\x3c\xb2\xa1', ('>', True): b'\xa1\xb2\x3c\x4d'}[(endian, nanos)])
        data = magic + struct.pack(endian+'HHiIII', 2, 4, 0, 0, 65535, 1)
        for fraction in fractions:
            data += struct.pack(endian+'IIII', 1609459200, fraction, 1, 1) + b'x'
        self.path.write_bytes(data)

    def scan(self):
        data = self.path.read_bytes()
        return scan_record_headers(self.path, expected_sha256=hashlib.sha256(data).hexdigest(),
                                   expected_size=len(data))

    def test_microsecond_regression_full_scan(self):
        self.write(); r = self.scan()
        self.assertEqual(r['records'], 3)
        self.assertEqual(r['timestamp_regressions'], 1)
        self.assertEqual(r['max_regression_ns'], 1000)
        self.assertFalse(r['sorting_performed'])

    def test_nanoseconds_big_endian(self):
        self.write(endian='>', nanos=True)
        self.assertEqual(self.scan()['max_regression_ns'], 1)

    def test_equal_timestamps_allowed(self):
        self.write(fractions=(1, 1, 2))
        self.assertEqual(self.scan()['timestamp_regressions'], 0)

    def test_truncated_payload_rejected(self):
        self.write(); self.path.write_bytes(self.path.read_bytes()[:-1])
        with self.assertRaises(AlignmentError): self.scan()

    def test_truncated_record_header_rejected(self):
        self.write(); self.path.write_bytes(self.path.read_bytes()+b'x')
        with self.assertRaises(AlignmentError): self.scan()

    def test_hash_mismatch_rejected(self):
        self.write()
        with self.assertRaises(AlignmentError):
            scan_record_headers(self.path, expected_sha256='f'*64, expected_size=self.path.stat().st_size)

    def test_no_unbounded_budget_override(self):
        with self.assertRaises(AlignmentError):
            scan_record_headers(self.path, expected_sha256='f'*64, expected_size=1, max_records=10**9)


class ConfigurationTests(unittest.TestCase):
    def config(self):
        import json
        return json.loads((ROOT/'configs/multicapture-audit-v1.json').read_text())

    def test_prepared_manifest_valid(self): validate_config(self.config())

    def test_cli_help_works_with_windows_legacy_stdout_encoding(self):
        import io
        stream = io.TextIOWrapper(io.BytesIO(), encoding='cp1252', write_through=True)
        with patch('sys.stdout', stream), self.assertRaises(SystemExit) as result:
            main(['--help'])
        self.assertEqual(result.exception.code, 0)

    def test_label_activation_rejected(self):
        config = self.config(); config['verified_label_assignment'] = True
        with self.assertRaises(AlignmentError): validate_config(config)

    def test_missing_case_or_larger_budget_rejected(self):
        for field in ('cases', 'bounds'):
            config = self.config()
            if field == 'cases': config['cases'].pop()
            else: config['bounds']['pcap_prefix'] = 100000
            with self.assertRaises(AlignmentError): validate_config(config)

    def test_duplicate_case_rejected(self):
        config = self.config(); config['cases'][1]['case_id'] = config['cases'][0]['case_id']
        with self.assertRaises(AlignmentError): validate_config(config)

    def test_existing_output_rejected_before_scan(self):
        with self.assertRaises(AlignmentError):
            run_audit(self.config(), ROOT, ROOT/'artifacts/runs')


class WorkflowTests(unittest.TestCase):
    """Audit fixtures thật qua filesystem; chỉ mock TShark child, không đọc dataset."""
    def setUp(self):
        import json
        self.temp = tempfile.TemporaryDirectory(prefix='multicapture-workflow-synthetic-')
        self.work = Path(self.temp.name)
        raw = self.work/'data/raw'; raw.mkdir(parents=True)
        (self.work/'artifacts/runs').mkdir(parents=True)
        pc = raw/'fixture.pcap'
        pc.write_bytes(b'\xd4\xc3\xb2\xa1'+struct.pack('<HHiIII',2,4,0,0,65535,1)
                       +struct.pack('<IIII',1609459200,1,1,1)+b'x')
        cs = raw/'fixture.csv'
        cs.write_text('frame.time,Attack_label,Attack_type\n2021 01:00:01,0,Normal\n', encoding='utf-8')
        tool = self.work/'synthetic-tool-placeholder'
        tool.write_bytes(b'never executed')
        self.config = json.loads((ROOT/'configs/multicapture-audit-v1.json').read_text())
        self.config['toolchain'] = dict(tshark_path=str(tool), tshark_sha256=hashlib.sha256(tool.read_bytes()).hexdigest())
        for i, case in enumerate(self.config['cases']):
            case['case_id'] = 'synthetic-'+str(i)
            for kind, path in [('pcap', pc), ('csv', cs)]:
                case[kind] = dict(path=path.relative_to(self.work).as_posix(),
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                  size_bytes=path.stat().st_size)
            case['prior_pcap_metadata'] = dict(records=1, timestamp_regressions=0)
        self.r1 = self.work/'artifacts/runs/r1'
        self.r2 = self.work/'artifacts/runs/r2'

    def tearDown(self): self.temp.cleanup()

    def run_fixture(self, output, reuse=None):
        with patch('nids_ml.cli.multicapture.pcap_scalar_prefix', return_value=([], {})):
            return run_audit(self.config, self.work, output, reuse_quality_from=reuse)

    def test_full_workflow_13_pairs_no_label_activation(self):
        result = self.run_fixture(self.r1)
        self.assertEqual(result['cases_completed'], 13)
        self.assertEqual(result['CSV_rows_full_scan'], 13)
        self.assertEqual(result['verified_labels_created'], 0)
        self.assertFalse(result['reused_full_scan_results'])

    def test_reuse_provenance_and_no_second_full_scan(self):
        import json
        self.run_fixture(self.r1)
        with patch('nids_ml.cli.multicapture.scan_csv_quality', side_effect=AssertionError('must reuse')):
            result = self.run_fixture(self.r2, self.r1)
        self.assertTrue(result['reused_full_scan_results'])
        case = json.loads((self.r2/'synthetic-0.json').read_text())
        self.assertEqual(case['reused_full_scan_evidence']['scope'], 'CSV_QUALITY_AND_PCAP_HEADERS_ONLY')
        self.assertEqual(case['reused_full_scan_evidence']['sha256'],
                         hashlib.sha256((self.r1/'synthetic-0.json').read_bytes()).hexdigest())

    def test_reuse_changed_source_manifest_rejected_before_output(self):
        import json
        self.run_fixture(self.r1)
        path = self.r1/'source-manifest.json'
        wrong = json.loads(path.read_text()); wrong['scope'] = 'different'
        path.write_text(json.dumps(wrong))
        with self.assertRaises(AlignmentError): self.run_fixture(self.r2, self.r1)
        self.assertFalse(self.r2.exists())

    def test_reuse_csv_tamper_is_rejected_and_failure_retained(self):
        self.run_fixture(self.r1)
        (self.work/'data/raw/fixture.csv').write_text('tampered')
        with self.assertRaises(ValueError): self.run_fixture(self.r2, self.r1)
        self.assertTrue((self.r2/'FAILED.json').is_file())

    def test_reuse_outside_workspace_rejected(self):
        with self.assertRaises(AlignmentError): self.run_fixture(self.r2, self.work.parent)
        self.assertFalse(self.r2.exists())


if __name__ == '__main__': unittest.main()
