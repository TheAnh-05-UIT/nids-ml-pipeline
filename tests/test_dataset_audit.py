"""Test semantics, overlap, feature-policy and report-integrity audits.
Test audit semantics, overlap, feature policy và integrity report.
"""
from contextlib import contextmanager
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'src'))
from nids_ml.validation.contracts import validate_schema,verify_manifest,read_published_report,require_verified_dnn
from nids_ml.eda.feature_policy import make_policy,validate_policy
from nids_ml.data.inventory import discover,fingerprint
from nids_ml.eda.overlap import row_identity,audit_overlap
from nids_ml.eda.semantics import DEFINITIONS,LABELS,value_kind,safe_example,payload_markers,audit_semantics,audit_label_integrity


@contextmanager
def scratch():
    parent=ROOT.parent/'task-arm-010/evidence'
    with tempfile.TemporaryDirectory(prefix='data001b-fixture-',dir=parent) as directory:
        path=Path(directory)
        if not path.resolve().is_relative_to(parent.resolve()):raise ValueError('Unsafe fixture directory')
        yield path


def csv_fixture(path,header,rows):
    with path.open('w',encoding='utf-8',newline='') as f:csv.writer(f).writerows([header,*rows])


class SemanticsTests(unittest.TestCase):
    def test_unsigned_bounds_and_precision(self):
        for raw,rule,want in [('65536','u16','out_of_range'),('-1','u16','negative'),
                              ('1.5','u16','noninteger'),('2','bit','out_of_range'),
                              ('18446744073709551615','u64','valid_integer'),
                              ('18446744073709551616','u64','out_of_range'),
                              ('NaN','u32','missing_token'),('Inf','u32','nonfinite'),
                              ('0x01','bit','valid_integer'),('6','mqtt_version','unexpected_protocol_level')]:
            with self.subTest(raw=raw):self.assertEqual(value_kind(raw,rule),want)

    def test_exact_zero_spelling(self):
        self.assertEqual(value_kind('0','u16'),'raw_zero')
        self.assertEqual(value_kind('0.0','u16'),'raw_zero_decimal')
        self.assertEqual(value_kind('','u16'),'blank')

    def test_time_formats_are_not_repaired(self):
        self.assertEqual(value_kind('192.0.2.1','time'),'ip_literal')
        self.assertEqual(value_kind('2021 01:02:03.123','time'),'year_time_without_month_day')
        self.assertEqual(value_kind('2021-01-02 01:02:03','time'),'calendar_timestamp_candidate_not_validated')

    def test_payload_examples_are_redacted(self):
        raw='secret fixture payload http://example.invalid/?password=fixture'
        example=safe_example(raw,'payload')
        self.assertNotIn(raw,json.dumps(example));self.assertTrue(example['raw_redacted'])
        self.assertEqual(example['sha256'],hashlib.sha256(raw.encode()).hexdigest())

    def test_payload_patterns_do_not_execute(self):
        self.assertIn('script_literal',payload_markers(b'<script>fixture</script>'.hex()))
        self.assertNotIn('script_literal',payload_markers('x'*1100+'<script>'))

    def test_schema_mismatch(self):
        with scratch() as tmp:
            p=tmp/'fixture.csv';csv_fixture(p,['bad'],[['0']])
            with self.assertRaisesRegex(ValueError,'schema'):audit_semantics(p,['expected'])

    def test_full_fixture_counters_and_determinism(self):
        header=[d[0] for d in DEFINITIONS]+list(LABELS)
        rows=[['0']*61+['0','Normal'],['0.0']*61+['1','Attack']]
        with scratch() as tmp:
            path=tmp/'synthetic.csv';csv_fixture(path,header,rows);before=fingerprint(path)
            a=audit_semantics(path,header);b=audit_semantics(path,header)
            self.assertEqual(a,b);self.assertEqual(a['row_count'],2);self.assertEqual(len(a['columns']),61)
            self.assertEqual(a['columns']['mqtt.topic']['kinds']['raw_zero']['by_class'],{'Normal':1})
            self.assertEqual(fingerprint(path),before)

    def test_nonfinite_representation_and_label_consistency(self):
        with scratch() as tmp:
            p=tmp/'fixture.csv';header=['Attack_label','Attack_type','tcp.options','tcp.payload','http.request.uri.query']
            csv_fixture(p,header,[['0','Normal','0','9'*400,'?'],['0','Attack','Inf','0','0']])
            report=audit_label_integrity(p,header)
            self.assertEqual(report['binary_vs_type_mismatches'],1)
            self.assertEqual(report['exceptional_cell_counts']['tcp.payload'],{'finite_literal_float64_overflow':1})
            self.assertEqual(report['exceptional_cell_counts']['tcp.options'],{'literal_nonfinite':1})


class OverlapTests(unittest.TestCase):
    def test_exact_and_decimal_identity(self):
        self.assertNotEqual(row_identity(['0']),row_identity(['0.0']))
        self.assertEqual(row_identity(['0'],True),row_identity(['0.0'],True))
        self.assertNotEqual(row_identity(['1234567890123456789012345678901'],True),row_identity(['1234567890123456789012345678902'],True))
        self.assertNotEqual(row_identity(['private text']),row_identity(['private text ']))

    def test_real_stream_fixture_membership(self):
        with scratch() as tmp:
            a,b=tmp/'a.csv',tmp/'b.csv';header=['value','Attack_label','Attack_type']
            csv_fixture(a,header,[['0','0','Normal'],['1','1','Attack']])
            csv_fixture(b,header,[['0.0','0','Normal'],['1','1','Attack'],['9','1','Attack']])
            report=audit_overlap(a,b,header,tmp)
            self.assertEqual(report['candidate_rows_matching_reference'],{'exact':1,'decimal_equivalent':2})
            self.assertEqual(report['unique_identities']['decimal_equivalent']['reference_unique_seen_in_candidate'],2)
            self.assertFalse(report['split_created'])

    def test_schema_mismatch_rejected(self):
        with scratch() as tmp:
            a,b=tmp/'a.csv',tmp/'b.csv';csv_fixture(a,['x'],[['0']]);csv_fixture(b,['y'],[['0']])
            with self.assertRaisesRegex(ValueError,'schema'):audit_overlap(a,b,['x'],tmp)

    def test_reference_budget_enforced(self):
        with scratch() as tmp:
            a=tmp/'a.csv';csv_fixture(a,['x'],[['0'],['1']])
            with self.assertRaisesRegex(ValueError,'budget'):audit_overlap(a,a,['x'],tmp,max_reference_rows=1)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.names=[d[0] for d in DEFINITIONS]
        eda={'columns':[{'name':n,'constant_raw_text':False} for n in self.names],
             'raw_feature_columns':self.names,'input':{'sha256':hashlib.sha256(b'synthetic fixture identity').hexdigest()}}
        detail={'columns':{n:{'kinds':{}} for n in self.names}}
        self.policy=make_policy(eda,detail)
        self.schema=json.loads((ROOT/'contracts/feature-policy.schema.json').read_text())

    def test_valid_policy_and_all_features(self):
        validate_policy(self.policy,self.schema,self.names)
        self.assertEqual([len(p['features']) for p in self.policy['policies']],[61]*3)

    def test_label_exclusion(self):
        self.policy['forbidden_labels']=['Attack_type']
        with self.assertRaises(ValueError):validate_policy(self.policy,self.schema,self.names)

    def test_bad_decision_is_rejected(self):
        self.policy['policies'][0]['features'][0]['decision']='TRUST_ME'
        with self.assertRaises(ValueError):validate_policy(self.policy,self.schema,self.names)

    def test_feature_duplicate_is_rejected(self):
        self.policy['policies'][0]['features'][0]['name']=self.names[1]
        with self.assertRaises(ValueError):validate_policy(self.policy,self.schema,self.names)

    def test_no_auto_application(self):
        self.policy['applied_to_data']=True
        with self.assertRaises(ValueError):validate_policy(self.policy,self.schema,self.names)

    def test_protocol_presence_is_reviewed_not_called_causal_leakage(self):
        eda={'columns':[{'name':n,'constant_raw_text':False} for n in self.names],
             'raw_feature_columns':self.names,'input':{'sha256':'a'*64}}
        detail={'columns':{n:{'kinds':{}} for n in self.names}}
        detail['columns']['mqtt.ver']['kinds']={'valid_integer':{'by_class':{'Normal':2},'count':2}}
        detail['columns']['tcp.len']['kinds']={'valid_integer':{'by_class':{'Normal':2,'Attack':2},'count':4}}
        policy=make_policy(eda,detail)
        choices={v['name']:v['decision'] for v in policy['policies'][0]['features']}
        self.assertEqual(choices['mqtt.ver'],'NEEDS_VERIFICATION')
        self.assertEqual(choices['tcp.len'],'INCLUDE')


class IntegrityTests(unittest.TestCase):
    def test_manifest_and_inventory_hash(self):
        with scratch() as tmp:
            (tmp/'contracts').mkdir();(tmp/'data/raw').mkdir(parents=True)
            p=tmp/'data/raw/fixture.csv';p.write_bytes(b'x\n1\n')
            manifest={'root_path':'data/raw','files':[{'path':'fixture.csv',**fingerprint(p)}]}
            schema={'type':'object','required':['root_path','files'],'properties':{'root_path':{'type':'string'},'files':{'type':'array'}}}
            (tmp/'contracts/dataset.schema.json').write_text(json.dumps(schema))
            (tmp/'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(verify_manifest(tmp,'manifest.json'),manifest)
            inv=discover(tmp);self.assertEqual(inv['file_count'],1);self.assertEqual(inv['files'][0]['sha256'],manifest['files'][0]['sha256'])
            p.write_bytes(b'x\n2\n')
            with self.assertRaisesRegex(ValueError,'identity'):verify_manifest(tmp,'manifest.json')

    def test_incomplete_or_corrupt_report_is_rejected(self):
        with scratch() as tmp:
            p=tmp/'report.json';p.write_text('{"synthetic":true}')
            marker={'publication_complete':True,'sha256':{'report.json':fingerprint(p)['sha256']}}
            (tmp/'COMPLETE.json').write_text(json.dumps(marker))
            self.assertEqual(read_published_report(tmp,'report.json'),{'synthetic':True})
            p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'SHA256'):read_published_report(tmp,'report.json')

    def test_missing_or_unverified_dnn_stops(self):
        actual={'size_bytes':1,'sha256':hashlib.sha256(b'x').hexdigest()}
        with self.assertRaises(ValueError):require_verified_dnn({},actual)
        with scratch() as tmp:
            with self.assertRaises(ValueError):fingerprint(tmp/'missing.csv')

    def test_unsupported_schema_is_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError,'Unsupported'):validate_schema({}, {'unknownConstraint':True})
        with self.assertRaises(ValueError):validate_schema({'extra':1},{'type':'object','additionalProperties':False})


if __name__=='__main__':unittest.main()
