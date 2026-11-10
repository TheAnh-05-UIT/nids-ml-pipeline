"""Test full-pair indexing, stream guards and inactive label policies.
Test index toàn cặp, stream guards và policy nhãn chưa kích hoạt.
"""
from dataclasses import replace
import csv
import hashlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nids_ml.alignment.candidates import AlignmentError,DAY_NS,HeaderRecord,make_record
from nids_ml.alignment.full import (Budget,FieldStream,FullAuditLimits,build_matches,
    ingest_csv,key_storage,open_index,verified_source)
from nids_ml.alignment.io import SCALAR_FIELDS
from nids_ml.cli.full_alignment import full_audit
from nids_ml.labeling.policy import dataset_label_decision,label_ordering_contract,ordering_gate

START=1609459200000000000
OFFSET=3600000000000


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='full-alignment-fixture-',dir=ROOT/'artifacts/runs')
        self.path=Path(self.tmp.name)
        self.budget=Budget(self.path,FullAuditLimits())
        # Tests SQL dùng fixture nhỏ, độc lập RSS của TensorFlow trong suite chung.
        # Resource guards được kiểm riêng và được bật trong CLI dữ liệu thật.
        self.budget.check=lambda force=False:None
        self.db=open_index(self.path/'index.sqlite',self.budget)

    def tearDown(self):
        self.db.close();self.tmp.cleanup()

    def add(self,i,*,key=None,pord=None,cord=None,ts=None,ctod=None,year=2021):
        key=key or ('192.0.2.1','192.0.2.2',1000+i,80,1,16,i,42)
        ts=START+i*1000000000 if ts is None else ts
        row=HeaderRecord('a'*64,i,key)
        kh,kj=key_storage(row)
        self.db.execute('INSERT INTO p VALUES(?,?,?,?,?,?,?)',
            (i if pord is None else pord,kh,kj,ts,ts%DAY_NS,i,None))
        self.db.execute('INSERT INTO c VALUES(?,?,?,?,?,?,?,?,?)',
            (i if cord is None else cord,kh,kj,(ts+OFFSET)%DAY_NS if ctod is None else ctod,year,i,None,'1','DDoS_HTTP'))

    def populate(self):
        for i in range(1,97):self.add(i)
        self.db.commit()

    def test_complete_unique_and_reordered_not_index_join(self):
        for i in range(1,97):self.add(i,cord=97-i)
        result=build_matches(self.db)
        self.assertEqual(result['content_counts'],{'UNIQUE_CONTENT_CANDIDATE':96})
        self.assertEqual(result['time_counts'],{'UNIQUE_CONTENT_AND_TIME_CANDIDATE':96})
        self.assertTrue(result['content_max_one_to_one']);self.assertTrue(result['witnessed_max_one_to_one'])
        self.assertEqual(result['clock']['calibration_pairs'],64)
        self.assertEqual(result['clock']['heldout_pairs'],32)
        self.assertEqual(self.db.execute('SELECT cord FROM witnessed WHERE pord=1').fetchone()[0],96)

    def test_global_duplicate_outside_prefix_removes_primary_uniqueness(self):
        self.populate()
        self.db.execute('INSERT INTO p SELECT 97,kh,kj,ts+1000000000,tod+1000000000,seq,reason FROM p WHERE ord=1')
        result=build_matches(self.db)
        self.assertEqual(result['content_counts']['AMBIGUOUS'],2)
        self.assertEqual(result['time_counts']['UNMATCHED'],1)
        self.assertEqual(self.db.execute('SELECT cord FROM content WHERE pord=1').fetchone()[0],None)

    def test_same_key_same_time_duplicates_never_choose_first(self):
        self.populate()
        self.db.execute('INSERT INTO c SELECT 97,kh,kj,tod,year,seq,reason,alabel,atype FROM c WHERE ord=1')
        result=build_matches(self.db)
        self.assertEqual(result['time_counts']['AMBIGUOUS'],1)
        self.assertEqual(result['witnessed_CSV_without_unique_pair'],2)

    def test_malicious_label_change_does_not_change_mapping(self):
        self.populate()
        self.db.execute("UPDATE c SET alabel='0',atype='Normal'")
        result=build_matches(self.db)
        self.assertEqual(result['time_counts'],{'UNIQUE_CONTENT_AND_TIME_CANDIDATE':96})
        self.assertNotIn('label',result)

    def test_unsupported_and_unmatched_stay_in_denominator(self):
        self.populate()
        self.db.execute("INSERT INTO p VALUES(97,NULL,NULL,?,?,NULL,'NON_TCP')",(START+97000000000,97000000000))
        self.db.execute('DELETE FROM c WHERE ord=96')
        result=build_matches(self.db)
        self.assertEqual(sum(result['content_counts'].values()),97)
        self.assertEqual(result['time_counts']['UNSUPPORTED_OR_INVALID'],1)
        self.assertEqual(result['time_counts']['UNMATCHED'],1)

    def test_clock_heldout_disagreement_blocks_witness(self):
        self.populate();self.db.execute('UPDATE c SET tod=tod+1 WHERE ord=96')
        result=build_matches(self.db)
        self.assertEqual(result['clock']['heldout_disagree'],1)
        self.assertFalse(result['clock']['time_witness_enabled'])
        self.assertEqual(result['time_counts'],{'BLOCKED_CLOCK_WITNESS':96})

    def test_missing_calibration_and_heldout_evidence_blocks(self):
        for i in range(1,11):self.add(i)
        result=build_matches(self.db)
        self.assertFalse(result['clock']['time_witness_enabled'])

    def test_multiday_capture_blocks_modulo_time_witness(self):
        self.populate();self.db.execute('UPDATE p SET ts=ts+? WHERE ord=96',(DAY_NS,))
        result=build_matches(self.db)
        self.assertEqual(result['clock']['pcap_UTC_days'],2)
        self.assertFalse(result['clock']['time_witness_enabled'])

    def test_csv_calendar_year_mismatch_blocks_witness(self):
        self.populate();self.db.execute('UPDATE c SET year=2022')
        result=build_matches(self.db)
        self.assertFalse(result['clock']['time_witness_enabled'])
        self.assertEqual(result['clock']['csv_observed_years'],[2022])

    def test_fingerprint_collision_needs_equal_canonical_key(self):
        self.populate()
        self.db.execute("UPDATE c SET kj='[0]' WHERE ord=96")
        result=build_matches(self.db)
        self.assertEqual(result['content_counts']['UNMATCHED'],1)

    def test_sqlite_page_cache_caps_and_no_overwrite(self):
        self.assertEqual(self.db.execute('PRAGMA max_page_count').fetchone()[0],131072)
        self.assertEqual(self.db.execute('PRAGMA cache_size').fetchone()[0],-16384)
        with self.assertRaisesRegex(AlignmentError,'INDEX_ALREADY_EXISTS'):
            open_index(self.path/'index.sqlite',self.budget)

    def test_csv_counts_labels_and_clock_regression_without_assignment(self):
        header=['frame.time','ip.src_host','ip.dst_host','tcp.srcport','tcp.dstport','tcp.len',
                'tcp.flags','tcp.ack_raw','tcp.checksum','Attack_label','Attack_type']
        header += ['unused_'+str(i) for i in range(63-len(header))]
        path=self.path/'fixture.csv'
        with path.open('w',encoding='utf-8',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=header);writer.writeheader()
            for i,time in enumerate(('2021 01:00:02','2021 01:00:01'),1):
                writer.writerow(dict(zip(header[:11],[time,'192.0.2.1','192.0.2.2',1000+i,80,1,16,i,42,'1','DDoS_HTTP'])))
        original=path.read_bytes()
        result=ingest_csv(self.db,path,hashlib.sha256(original).hexdigest(),self.budget)
        self.assertEqual(result['rows'],2);self.assertEqual(result['time_of_day_regressions'],1)
        self.assertTrue(result['full_scan']);self.assertEqual(path.read_bytes(),original)

    def test_bad_csv_header_rejected(self):
        path=self.path/'bad.csv';path.write_text('a,b\n1,2\n',encoding='utf-8')
        with self.assertRaisesRegex(AlignmentError,'FULL_CSV_HEADER'):
            ingest_csv(self.db,path,'a'*64,self.budget)


class GuardsTests(unittest.TestCase):
    def test_resource_profile_cannot_silently_expand(self):
        self.assertIsInstance(FullAuditLimits().validate(),FullAuditLimits)
        for key in FullAuditLimits.__dataclass_fields__:
            with self.subTest(key=key),self.assertRaises(AlignmentError):
                replace(FullAuditLimits(),**{key:getattr(FullAuditLimits(),key)+1}).validate()

    def test_existing_output_rejected_before_sources_read(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/runs') as tmp:
            with self.assertRaisesRegex(AlignmentError,'FRESH_OUTPUT'):
                full_audit({},ROOT,tmp)

    def test_source_sha_size_and_raw_root_enforced(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/runs') as tmp:
            work=Path(tmp);path=work/'data/raw/a.csv';path.parent.mkdir(parents=True);path.write_bytes(b'a,b\n1,2\n')
            manifest=dict(path='data/raw/a.csv',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),size_bytes=8)
            self.assertEqual(verified_source(manifest,work)[0],path)
            with self.assertRaisesRegex(AlignmentError,'SOURCE_IDENTITY'):
                verified_source({**manifest,'sha256':'0'*64},work)
            with self.assertRaisesRegex(AlignmentError,'SOURCE_RELATIVE_PATH'):
                verified_source({**manifest,'path':'../outside.csv'},work)

    def test_memory_and_disk_budget_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/runs') as tmp:
            budget=Budget(tmp,FullAuditLimits())
            with patch('nids_ml.alignment.full.shutil.disk_usage',return_value=SimpleNamespace(free=1)):
                with self.assertRaisesRegex(AlignmentError,'DISK_FREE'):budget.check(force=True)
            with patch('nids_ml.alignment.full.shutil.disk_usage',return_value=SimpleNamespace(free=4*1024**3)), \
                 patch('nids_ml.alignment.full.process_rss',return_value=257*1024**2):
                with self.assertRaisesRegex(AlignmentError,'PYTHON_RSS'):budget.check(force=True)

    def test_ordinal_profile_keeps_old_prefix_default(self):
        with self.assertRaisesRegex(AlignmentError,'ORDINAL_RANGE'):
            make_record({},source_sha256='a'*64,ordinal=100001,kind='csv')
        self.assertEqual(make_record({},source_sha256='a'*64,ordinal=250000,kind='csv',ordinal_ceiling=250000).ordinal,250000)
        with self.assertRaises(AlignmentError):make_record({},source_sha256='a'*64,ordinal=250001,kind='csv',ordinal_ceiling=250000)

    def test_wall_timeout_and_sqlite_interrupt_propagate(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/runs') as tmp:
            budget=Budget(tmp,FullAuditLimits())
            with patch('nids_ml.alignment.full.time.monotonic',return_value=budget.started+901):
                with self.assertRaisesRegex(AlignmentError,'FULL_AUDIT_TIMEOUT'):budget.check(force=True)
            with patch.object(budget,'check',side_effect=AlignmentError('synthetic abort')):
                self.assertEqual(budget.sqlite_progress(),1)
                self.assertEqual(budget.error,'synthetic abort')


class FakeProcess:
    def __init__(self,data,exit_code=0):
        self.stdout=io.BytesIO(data);self.stderr=io.BytesIO();self.pid=12345;self.returncode=exit_code
    def poll(self):return self.returncode
    def wait(self,timeout=None):return self.returncode


class StreamTests(unittest.TestCase):
    def stream(self,data,exit_code=0):
        budget=SimpleNamespace(limits=FullAuditLimits(),check=lambda:None)
        child=FakeProcess(data,exit_code)
        return FieldStream(['synthetic-child-not-executed'],budget),child

    def line(self,ordinal=1):
        values=['']*len(SCALAR_FIELDS);values[0]=str(ordinal);values[1]='1609459201'
        output=io.StringIO();csv.writer(output,delimiter='|',quoting=csv.QUOTE_ALL,lineterminator='\n').writerow(values)
        return output.getvalue().encode()

    def run_stream(self,stream,child):
        with patch('nids_ml.alignment.full.subprocess.Popen',return_value=child), \
             patch('nids_ml.alignment.full.process_rss',return_value=1024):
            return list(stream.rows())

    def test_stream_eof_and_child_closed_without_replay(self):
        stream,child=self.stream(self.line())
        self.assertEqual(len(self.run_stream(stream,child)),1)
        self.assertTrue(stream.complete);self.assertTrue(child.stdout.closed)
        self.assertTrue(all(not thread.is_alive() for thread in stream.threads))
        with self.assertRaisesRegex(AlignmentError,'NO_REPLAY'):list(stream.rows())

    def test_partial_line_wrong_ordinal_width_and_child_failure_rejected(self):
        for data,code in ((self.line()[:-1],0),(self.line(2),0),(b'"1"|"2"\n',0),(self.line(),1)):
            stream,child=self.stream(data,code)
            with self.subTest(data=data,exit_code=code),self.assertRaises(AlignmentError):
                self.run_stream(stream,child)
            self.assertFalse(stream.complete);self.assertTrue(child.stdout.closed)


class PolicyTests(unittest.TestCase):
    def test_no_actual_approval_means_unknown(self):
        decision=dataset_label_decision('1','DDoS_HTTP',mapping_state='GLOBAL_UNIQUE_CONTENT_AND_TIME_CANDIDATE')
        self.assertIsNone(decision['y']);self.assertFalse(decision['supervised_eligible'])
        for invalid in (1,None,'approved'):
            with self.assertRaises(AlignmentError):dataset_label_decision('0','Normal',mapping_state='',manifest_approved=invalid)

    def test_synthetic_approved_policy_targets_and_rejections(self):
        def decide(a,b,state='GLOBAL_UNIQUE_CONTENT_AND_TIME_CANDIDATE'):
            return dataset_label_decision(a,b,mapping_state=state,manifest_approved=True)
        self.assertEqual(decide('0','Normal')['y'],0)
        for attack in ('DDoS_HTTP','DDoS_TCP','DDoS_UDP','DDoS_ICMP'):self.assertEqual(decide('1',attack)['y'],1)
        for a,b in [('1','Normal'),('0','DDoS_HTTP'),('1','Backdoor'),('',''),(1,'DDoS_HTTP')]:
            self.assertIsNone(decide(a,b)['y'])
        for status in ('AMBIGUOUS','UNMATCHED','MIXED','UNKNOWN','UNIQUE_CONTENT_CANDIDATE'):
            self.assertIsNone(decide('1','DDoS_HTTP',status)['y'])

    def test_ordering_and_contract_never_implicitly_sort_or_activate(self):
        self.assertEqual(ordering_gate(0,source_complete=True)['status'],'PASS_CAPTURE_ORDER')
        self.assertEqual(ordering_gate(1,source_complete=True)['status'],'BLOCKED')
        self.assertEqual(ordering_gate(0,source_complete=False)['status'],'BLOCKED')
        for bad in (-1,True,1.5):
            with self.assertRaises(AlignmentError):ordering_gate(bad,source_complete=True)
        contract=label_ordering_contract()
        self.assertEqual(contract['activation'],'PROPOSED_NOT_ACTIVE')
        self.assertTrue(contract['full_window']['overflow_packets_included'])
        self.assertFalse(contract['full_window']['majority_vote'])
        self.assertFalse(contract['ordering']['implicit_sort'])
        self.assertFalse(contract['labels_assigned_by_this_task'])


if __name__=='__main__':unittest.main()
