"""Validate large TCP/ICMP mapping, compact keys and fail-closed binary annotations.
Kiểm bằng fixtures synthetic; các số đếm test không phải dữ liệu Edge-IIoTset thật.
"""
from contextlib import redirect_stdout, closing
from dataclasses import asdict, replace
import csv
from functools import partial
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
import test_tcp_full_pair_v2 as fixtures
from nids_ml.alignment.candidates import AlignmentError, make_record
from nids_ml.alignment.full import FullAuditLimits, Budget, open_index, ingest_csv, FieldStream, build_matches
from nids_ml.alignment.io import AUDIT_SCALAR_FIELDS
from test_full_alignment import FakeProcess
from nids_ml.alignment.large import (LargeAuditLimits, POLICY, MANIFEST, KEY_PROFILE,
    validate_large_config, select_case, large_record, compact_key_storage, binary_target, mapping_contract)
from nids_ml.cli.full_alignment import validate_audit_config, main


class LargeMappingTests(unittest.TestCase):
    def setUp(self):
        self.fx = fixtures.TcpPairV2Tests()
        self.fx.setUp()
        self.fx.config.update(schema_version='1.2.0', key_profile=KEY_PROFILE, protocol='TCP',
                              label_policy=POLICY, limits=asdict(LargeAuditLimits()))

    def tearDown(self):
        self.fx.tearDown()

    def report(self, name='result', child=None):
        return self.fx.run_audit(name, child)

    def query(self, name, sql):
        path = self.fx.work / 'artifacts/runs' / name / 'alignment.sqlite'
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as db:
            return db.execute(sql).fetchall()

    def manifest(self):
        c = self.fx.config
        case = {k: c[k] for k in ('case_id', 'protocol', 'pcap', 'csv', 'expected_pcap_records', 'expected_csv_rows')}
        case.update(enabled=True, block_reason=None)
        return dict(format=MANIFEST, label_policy=POLICY, cases=[case])

    def test_complete_compact_tcp_annotations_keep_index_hash(self):
        report = self.report()
        self.assertEqual(report['annotations']['mapped_packets'], 96)
        self.assertEqual(report['annotations']['unknown_packets'], 0)
        self.assertEqual(report['annotations']['counts'], [dict(y=0, attack_type='Normal', packets=96)])
        self.assertTrue(report['labels_assigned'])
        self.assertFalse(report['annotations']['independent_ground_truth_verified'])
        self.assertFalse(report['training'])
        self.assertEqual(self.query('result', 'SELECT DISTINCT length(kh),kj FROM p'), [(23, '')])
        index = self.fx.work / 'artifacts/runs/result/alignment.sqlite'
        self.assertEqual(hashlib.sha256(index.read_bytes()).hexdigest(), report['index_sha256'])
        self.assertEqual(report['decoder']['mapping_session_reset_packets'], 50000)
        self.assertEqual(report['decoder']['command'][report['decoder']['command'].index('-M')+1], '50000')
        self.assertEqual(report['contract'], mapping_contract())
        self.assertEqual(json.loads((index.parent/'label-ordering-contract.json').read_text()), report['contract'])
        self.assertEqual(report['contract']['target'], dict(normal=0,attack=1,unknown=None))
        self.assertIn('Backdoor', report['contract']['positive_types'])
        self.assertEqual(report['contract']['flow_supervision'], 'NOT_IMPLEMENTED_BY_THIS_MAPPING_TASK')

    def test_all_attack_families_are_binary_not_all_called_ddos(self):
        families = ('Backdoor', 'DDoS_HTTP', 'DDoS_ICMP', 'DDoS_TCP', 'DDoS_UDP', 'OS_Fingerprinting',
                    'MITM', 'Password', 'Port_Scanning', 'Ransomware', 'SQL_injection', 'Uploading',
                    'Vulnerability_scanner', 'XSS')
        for kind in families:
            self.assertEqual(binary_target('1', kind), 1)
        self.assertEqual(binary_target('0', 'Normal'), 0)
        for label, kind in [('1', 'Normal'), ('0', 'Backdoor'), ('', ''), (1, 'Backdoor'),
                            ('1', 'arbitrary'), ('0', None)]:
            self.assertIsNone(binary_target(label, kind))

    def test_compaction_preserves_matching_labels_and_duplicate_statistics(self):
        # Compare the old SQL algorithm and storage compaction on identical fixtures.
        # Đối chiếu cả nhãn, ambiguity và clock-blocked; chỉ bỏ bảng làm việc của index mới.
        def baseline(db, **_):
            result = build_matches(db)
            result['working_table_histograms'] = {
                name: dict(db.execute('SELECT n,count(*) FROM '+name+' WHERE n>1 GROUP BY n'))
                for name in ('pg', 'cg')}
            return result

        for scenario in ('unique', 'duplicate', 'clock-blocked'):
            with self.subTest(scenario=scenario):
                if scenario == 'duplicate':
                    self.fx.rows[1].update({k: v for k, v in self.fx.rows[0].items() if k != 'frame.number'})
                if scenario == 'clock-blocked':
                    self.fx.rows[-1]['frame.time_epoch'] = '1609459200'
                compact = self.report(scenario+'-compact')
                with patch('nids_ml.cli.full_alignment.build_matches', side_effect=baseline):
                    old = self.report(scenario+'-baseline')
                for field in ('mapping', 'annotations', 'duplicate_key_group_histograms'):
                    self.assertEqual(compact[field], old[field])
                for table in ('p', 'c', 'witnessed', 'mapped_labels'):
                    self.assertEqual(self.query(scenario+'-compact', 'SELECT * FROM '+table+' ORDER BY 1'),
                                     self.query(scenario+'-baseline', 'SELECT * FROM '+table+' ORDER BY 1'))
                self.assertEqual(self.query(scenario+'-compact',
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"),
                    [('c',), ('p',), ('witnessed',)])

    def test_family_labels_never_influence_candidate_matching(self):
        first = self.report('first')
        self.fx.write_csv('1', 'Backdoor')
        self.fx.config['csv'] = self.fx.identity(self.fx.cs)
        second = self.report('second')
        self.assertEqual(first['mapping'], second['mapping'])
        self.assertEqual(second['annotations']['counts'], [dict(y=1, attack_type='Backdoor', packets=96)])

    def test_inconsistent_labels_become_unknown_without_dropping_packets(self):
        self.fx.write_csv('0', 'Backdoor')
        self.fx.config['csv'] = self.fx.identity(self.fx.cs)
        report = self.report()
        self.assertEqual(report['mapping']['time_counts'], {'UNIQUE_CONTENT_AND_TIME_CANDIDATE': 96})
        self.assertEqual(report['annotations']['mapped_packets'], 0)
        self.assertEqual(report['annotations']['unknown_packets'], 96)
        self.assertEqual(self.query('result', 'SELECT count(*) FROM mapped_labels WHERE y IS NULL'), [(96,)])

    def test_fingerprinting_alias_keeps_original_source_taxonomy(self):
        self.fx.write_csv('1', 'OS_Fingerprinting')
        self.fx.config['csv'] = self.fx.identity(self.fx.cs)
        self.report()
        self.assertEqual(self.query('result', 'SELECT DISTINCT source_attack_type,attack_type,y FROM mapped_labels'),
                         [('OS_Fingerprinting', 'Fingerprinting', 1)])

    def test_complete_icmp_shared_key_and_clock_with_declared_limitations(self):
        self.fx.config['protocol'] = 'ICMP'
        for i, row in enumerate(self.fx.rows, 1):
            row.update({'ip.proto': '1', 'tcp.srcport': '0', 'tcp.dstport': '0',
                        'icmp.checksum': str(40000+i), 'icmp.seq_le': str(i), 'icmp.type': '8', 'icmp.code': '0'})
        self.fx.write_csv('1', 'DDoS_ICMP')
        self.fx.config['csv'] = self.fx.identity(self.fx.cs)
        report = self.report()
        self.assertEqual(report['annotations']['mapped_packets'], 96)
        self.assertEqual(report['annotations']['counts'], [dict(y=1, attack_type='DDoS_ICMP', packets=96)])
        self.assertEqual(self.query('result', 'SELECT DISTINCT length(kh) FROM p'), [(13,)])
        self.assertIn('lacks type/code/identifier', report['key_limitations'])

    def test_udp_fragments_and_unmatched_stay_in_denominator(self):
        self.fx.rows[0]['ip.proto'] = '17'
        self.fx.rows[1]['ip.flags.mf'] = 'True'
        self.fx.rows[2]['tcp.checksum'] = '65535'
        report = self.report()
        self.assertEqual(report['PCAP']['records'], 96)
        self.assertEqual(report['annotations']['packet_count'], 96)
        self.assertEqual(report['annotations']['mapped_packets'], 93)
        self.assertEqual(report['annotations']['unknown_packets'], 3)
        self.assertEqual(self.query('result', 'SELECT source_ordinal,y FROM mapped_labels WHERE source_ordinal<=3'),
                         [(1, None), (2, None), (3, None)])

    def test_duplicate_same_key_same_time_not_resolved_by_matching_label(self):
        self.fx.rows[1].update({k: v for k, v in self.fx.rows[0].items() if k != 'frame.number'})
        report = self.report()
        self.assertEqual(report['mapping']['time_counts'].get('AMBIGUOUS'), 2)
        self.assertIsNone(self.query('result', 'SELECT y FROM mapped_labels WHERE source_ordinal=1')[0][0])

    def test_clock_disagreement_keeps_all_labels_unknown(self):
        with self.fx.cs.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream); rows = list(reader); fields = reader.fieldnames
        rows[-1]['frame.time'] = '2021 01:01:37'
        with self.fx.cs.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        self.fx.config['csv'] = self.fx.identity(self.fx.cs)
        report = self.report()
        self.assertFalse(report['mapping']['clock']['time_witness_enabled'])
        self.assertEqual(report['annotations']['unknown_packets'], 96)

    def test_compact_key_preserves_direction_protocol_and_exact_field_values(self):
        row = dict(self.fx.rows[0])
        kwargs = dict(kind='pcap', source_sha256='a'*64, ordinal=250001, ordinal_ceiling=3500000, protocol='TCP')
        original = large_record(row, **kwargs)
        key = compact_key_storage(original)
        row['tcp.checksum'] = str(int(row['tcp.checksum']) ^ 1)
        self.assertNotEqual(compact_key_storage(large_record(row, **kwargs)), key)
        row = dict(self.fx.rows[0]); row['ip.src'], row['ip.dst'] = row['ip.dst'], row['ip.src']
        self.assertNotEqual(compact_key_storage(large_record(row, **kwargs)), key)
        self.assertEqual(compact_key_storage(replace(original, key=None)), (None, None))

    def test_large_profile_is_explicit_and_legacy_limits_remain_unchanged(self):
        self.assertEqual(validate_large_config(self.fx.config).record_cap, 3500000)
        self.assertEqual(FullAuditLimits().validate().record_cap, 250000)
        old = json.loads((ROOT/'configs/full-pair-heart-rate-arm64-v2.json').read_text())
        self.assertEqual(validate_audit_config(old)[0].record_cap, 250000)
        for field, value in [('record_cap', 4000000), ('python_rss_mib', True), ('scratch_mib', 999999)]:
            invalid = dict(self.fx.config, limits=dict(self.fx.config['limits'], **{field: value}))
            with self.assertRaises(AlignmentError): validate_large_config(invalid)
        for field, value in [('expected_csv_rows', True), ('expected_pcap_records', 3500000),
                             ('label_policy', 'verified'), ('protocol', 'UDP')]:
            with self.assertRaises(AlignmentError): validate_large_config(dict(self.fx.config, **{field: value}))
        with self.assertRaises(AlignmentError): make_record({}, source_sha256='a'*64, kind='csv', ordinal=250001, ordinal_ceiling=250000)

    def test_streaming_csv_passes_old_250k_boundary_without_changing_raw(self):
        path = self.fx.work/'data/raw/boundary.csv'
        fields = ['unused_'+str(i) for i in range(60)] + ['frame.time', 'Attack_label', 'Attack_type']
        line = ','.join(['0']*60 + ['2021 00:00:01', '1', 'Backdoor']) + '\n'
        with path.open('w', encoding='utf-8', newline='') as stream:
            stream.write(','.join(fields)+'\n')
            for _ in range(250001): stream.write(line)
        before = self.fx.identity(path)
        target = self.fx.work/'artifacts/runs/boundary'; target.mkdir()
        budget = Budget(target, LargeAuditLimits())
        db = open_index(target/'alignment.sqlite', budget)
        try:
            report = ingest_csv(db, path, before['sha256'], budget,
                record_factory=partial(large_record, protocol='TCP'), key_encoder=compact_key_storage)
            self.assertEqual(report['rows'], 250001)
            self.assertEqual(db.execute('SELECT max(ord) FROM c').fetchone()[0], 250001)
        finally: db.close()
        self.assertEqual(before, self.fx.identity(path))

    def test_manifest_blocks_unsupported_sources_and_rejects_duplicates(self):
        manifest = self.manifest()
        selected = select_case(manifest, self.fx.config['case_id'], self.fx.config['toolchain'])
        self.assertEqual(selected, self.fx.config)
        manifest['cases'][0].update(enabled=False, block_reason='CSV_CLOCK_INVALID')
        with self.assertRaisesRegex(AlignmentError, 'CASE_BLOCKED'): select_case(manifest, self.fx.config['case_id'], {})
        real = json.loads((ROOT/'configs/multi-attack-pcap-v1.json').read_text())
        for case in ('ddos-udp', 'mitm', 'normal-modbus', 'vulnerability-scanner', 'ransomware',
                     'normal-ir-receiver', 'normal-phvalue', 'normal-temperature-and-humidity', 'normal-water-level'):
            with self.assertRaisesRegex(AlignmentError, 'CASE_BLOCKED'): select_case(real, case, {})
        manifest['cases'].append(dict(manifest['cases'][0]))
        with self.assertRaisesRegex(AlignmentError, 'DUPLICATE_CASE'): select_case(manifest, self.fx.config['case_id'], {})

    def test_dry_run_no_child_no_outputs_and_ordinary_cli_executes_once(self):
        path = self.fx.work/'manifest.json'; path.write_text(json.dumps(self.manifest()))
        output = self.fx.work/'artifacts/runs/dry-run'
        args = ['--config', str(path), '--case', self.fx.config['case_id'], '--workspace', str(self.fx.work),
                '--tshark', str(self.fx.tool), '--tshark-sha256', self.fx.config['toolchain']['tshark_sha256'],
                '--output', str(output)]
        with patch('nids_ml.alignment.full.subprocess.Popen') as child, redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(main(args+['--dry-run']), 0)
            child.assert_not_called()
        self.assertFalse(output.exists())
        self.assertEqual(json.loads(stdout.getvalue())['status'], 'PASS_MAPPING_PREFLIGHT')
        with patch('nids_ml.alignment.full.subprocess.Popen', return_value=self.fx.child()) as child, redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(main(args), 0)
            child.assert_called_once()
        self.assertEqual(json.loads(stdout.getvalue().splitlines()[-1])['annotations']['mapped_packets'], 96)

    def test_failure_no_false_complete_or_automatic_replay(self):
        with self.assertRaises(AlignmentError): self.report(child=self.fx.child(code=1))
        out = self.fx.work/'artifacts/runs/result'
        self.assertTrue((out/'FAILED.json').exists())
        self.assertFalse((out/'coverage.json').exists())
        with self.assertRaises(AlignmentError): self.report()

    def test_dry_run_rejects_missing_parent_and_existing_output_without_child(self):
        config=self.fx.work/'config.json';config.write_text(json.dumps(self.fx.config))
        existing=self.fx.work/'artifacts/runs/existing';existing.mkdir()
        for output in (existing,self.fx.work/'artifacts/runs/missing/child'):
            with patch('nids_ml.alignment.full.subprocess.Popen') as child,redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(main(['--config',str(config),'--workspace',str(self.fx.work),
                    '--output',str(output),'--dry-run']),2)
                child.assert_not_called()
            self.assertEqual(json.loads(stdout.getvalue())['status'],'FAIL_CLOSED')

    def test_session_reset_preserves_global_ordinals_and_validates_local_counter(self):
        target=self.fx.work/'artifacts/runs/reset';target.mkdir()
        budget=Budget(target,LargeAuditLimits())
        text=io.StringIO();writer=csv.writer(text,delimiter='|',quoting=csv.QUOTE_ALL,lineterminator='\n')
        for ordinal in range(1,50003):
            writer.writerow([str((ordinal-1)%50000+1) if field=='frame.number' else '' for field in AUDIT_SCALAR_FIELDS])
        data=text.getvalue().encode()
        stream=FieldStream([],budget,fields=AUDIT_SCALAR_FIELDS,session_reset_packets=50000)
        with patch('nids_ml.alignment.full.subprocess.Popen',return_value=FakeProcess(data)):
            records=[(ordinal,row['frame.number']) for ordinal,row in stream.rows() if ordinal in (1,50000,50001,50002)]
        self.assertEqual(records,[(1,'1'),(50000,'50000'),(50001,'1'),(50002,'2')])
        self.assertTrue(stream.complete)
        old=FieldStream([],budget,fields=AUDIT_SCALAR_FIELDS)
        with patch('nids_ml.alignment.full.subprocess.Popen',return_value=FakeProcess(data)):
            with self.assertRaisesRegex(AlignmentError,'SCALAR_ORDINAL'):
                for _ in old.rows():pass
        with self.assertRaisesRegex(AlignmentError,'UNAPPROVED_MAPPING_SESSION_RESET'):
            FieldStream([],budget,fields=AUDIT_SCALAR_FIELDS,session_reset_packets=100000)


if __name__ == '__main__':
    unittest.main()
