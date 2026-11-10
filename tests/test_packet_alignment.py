"""Test label-blind matching, duplicates, ambiguity and bounded prefix I/O.
Test matching không dùng nhãn, duplicates, ambiguity và prefix I/O.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nids_ml.alignment.candidates import (AlignmentError,HeaderRecord,candidate_join,clock_hypothesis,
    csv_clock,make_record,numeric_token,time_witness_join)
from nids_ml.alignment.io import csv_prefix,pcap_scalar_prefix,SCALAR_FIELDS


def raw(ordinal=1,kind='pcap'):
    row={'frame.number':str(ordinal),'frame.time_epoch':str(ordinal)+'.000000001','ip.proto':'6',
         'ip.src':'192.0.2.1','ip.dst':'192.0.2.2','tcp.srcport':str(40000+ordinal),'tcp.dstport':'80',
         'tcp.len':'0','tcp.flags':'0x0002','tcp.ack_raw':'0','tcp.checksum':'0x1234','tcp.seq':'0'}
    if kind=='csv':
        row['ip.src_host']=row.pop('ip.src');row['ip.dst_host']=row.pop('ip.dst')
        row['frame.time']=f' 2021 01:00:{ordinal:02d}.000000001 '
        row['Attack_label']='1';row['Attack_type']='DDoS_HTTP'
    return row


def view(ordinal=1,kind='pcap'):
    return make_record(raw(ordinal,kind),source_sha256=('a' if kind=='pcap' else 'b')*64,ordinal=ordinal,kind=kind)


class NumericAndClockTests(unittest.TestCase):
    def test_exact_float_spelling_and_hex_equivalence(self):
        self.assertEqual(numeric_token('40001.0',16),40001)
        self.assertEqual(numeric_token('0X00FF',16),255)
        self.assertEqual(numeric_token('4294967295.0'),4294967295)
    def test_extreme_nonfinite_fractional_negative_and_overflow_rejected(self):
        for token in ('NaN','sNaN','inf','1.1','-1','4294967296','0x100000000','1e999999999','1e-99999','1;2'):
            with self.subTest(token=token),self.assertRaises(AlignmentError):numeric_token(token)
    def test_clock_preserves_missing_calendar(self):
        year,tod=csv_clock(' 2021 11:35:29.350784000 ')
        self.assertEqual(year,2021);self.assertEqual(tod,41729350784000)
        self.assertEqual(csv_clock('6.0'),(None,None))
    def test_clock_range_precision_and_no_rounding(self):
        for token in ('2021 24:00:00','2021 00:60:00','2021 00:00:60','2021 00:00:00.0000000001','2021-11-25T10:00:00Z'):
            with self.subTest(token=token):self.assertEqual(csv_clock(token),(None,None))


class JoinTests(unittest.TestCase):
    def test_unique_reordered_and_dropped_records_not_index_join(self):
        p=[view(1),view(2),view(3)]
        c=[replace(view(3,'csv'),ordinal=1),replace(view(1,'csv'),ordinal=2)]
        result=candidate_join(p,c)
        self.assertEqual(result['counts'],{'UNIQUE_CONTENT_CANDIDATE':2,'UNMATCHED_IN_SEARCH_PREFIX':1})
        self.assertEqual(result['rows'][0]['candidate_csv_ordinal'],2)
        self.assertEqual(result['rows'][2]['candidate_csv_ordinal'],1)
    def test_duplicate_CSV_is_ambiguous_even_same_label(self):
        p=[view()];c=[view(1,'csv'),replace(view(1,'csv'),ordinal=2)]
        self.assertEqual(candidate_join(p,c)['counts'],{'AMBIGUOUS':1})
    def test_duplicate_PCAP_competing_for_one_CSV_is_ambiguous(self):
        p=[view(),replace(view(),ordinal=2)];c=[view(1,'csv')]
        self.assertEqual(candidate_join(p,c)['counts'],{'AMBIGUOUS':2})
    def test_labels_absent_changed_or_conflicting_never_resolve_join(self):
        row=raw(kind='csv');a=make_record(row,source_sha256='b'*64,ordinal=1,kind='csv')
        row['Attack_label']='0';row['Attack_type']='Normal'
        b=make_record(row,source_sha256='b'*64,ordinal=1,kind='csv')
        del row['Attack_label'];del row['Attack_type']
        c=make_record(row,source_sha256='b'*64,ordinal=1,kind='csv')
        self.assertEqual(a,b);self.assertEqual(a,c)
        self.assertEqual(candidate_join([view()],[a]),candidate_join([view()],[b]))
    def test_checksum_disambiguates_repeated_SYN_ports(self):
        a=view();b=replace(view(),ordinal=2,key=(*view().key[:-1],0xabcd))
        c=[view(1,'csv'),replace(view(1,'csv'),ordinal=2,key=b.key)]
        self.assertEqual(candidate_join([a,b],c)['counts'],{'UNIQUE_CONTENT_CANDIDATE':2})
    def test_direction_is_packet_direction_not_bidirectional_flow(self):
        c=view(1,'csv');reverse=(c.key[1],c.key[0],c.key[3],c.key[2],*c.key[4:])
        self.assertEqual(candidate_join([view()],[replace(c,key=reverse)])['counts'],{'UNMATCHED_IN_SEARCH_PREFIX':1})
    def test_nonTCP_invalid_and_missing_key_stay_in_denominator(self):
        r=raw();r['ip.proto']='17';a=make_record(r,source_sha256='a'*64,ordinal=1,kind='pcap')
        r=raw(2);r['tcp.checksum']='missing';b=make_record(r,source_sha256='a'*64,ordinal=2,kind='pcap')
        result=candidate_join([a,b],[view(1,'csv')])
        self.assertEqual(result['counts'],{'UNSUPPORTED_OR_INVALID':2});self.assertEqual(result['source_records'],2)
    def test_bounds_duplicate_ordinals_and_cross_capture_fail_closed(self):
        for p in ([view(),view()],[view(),replace(view(),ordinal=2,source_sha256='c'*64)],
                  [replace(view(),ordinal=True)]):
            with self.subTest(p=p),self.assertRaises(AlignmentError):candidate_join(p,[])
        with self.assertRaises(AlignmentError):candidate_join([replace(view(),ordinal=i+1) for i in range(257)],[])
    def test_determinism_and_never_supervised_eligible(self):
        a=candidate_join([view()],[view(1,'csv')]);b=candidate_join([view()],[view(1,'csv')])
        self.assertEqual(json.dumps(a,sort_keys=True),json.dumps(b,sort_keys=True))
        self.assertFalse(a['label_assignment_performed']);self.assertFalse(a['rows'][0]['supervised_eligible'])
    def test_range_censoring_not_claimed_global_unmatched(self):
        r=candidate_join([view()],[view(1,'csv'),view(2,'csv')])
        self.assertTrue(r['prefix_censoring']);self.assertEqual(r['csv_search_rows_without_key_in_PCAP_prefix'],1)


class HypothesisTests(unittest.TestCase):
    def records(self):
        p=[];c=[]
        for i in range(1,97):
            x=replace(view(),ordinal=i,key=(*view().key[:2],40000+i,*view().key[3:]),timestamp_ns=i*1000000000,time_of_day_ns=i*1000000000)
            p.append(x);c.append(replace(x,source_sha256='b'*64,time_of_day_ns=(3600+i)*1000000000))
        return p,c
    def test_fixed_calibration_heldout_and_no_timestamp_edit(self):
        p,c=self.records();h=clock_hypothesis(p,c,candidate_join(p,c))
        self.assertEqual(h['calibration_pairs'],64);self.assertEqual(h['heldout_agree'],32)
        self.assertEqual(h['csv_minus_pcap_modulo_day_ns'],3600000000000)
        self.assertEqual(h['timezone'],'UNKNOWN');self.assertFalse(h['timestamp_edits_performed'])
    def test_inconsistent_calibration_no_best_fit_or_fallback(self):
        p,c=self.records();c[0]=replace(c[0],time_of_day_ns=c[0].time_of_day_ns+1)
        h=clock_hypothesis(p,c,candidate_join(p,c))
        self.assertIsNone(h['csv_minus_pcap_modulo_day_ns'])
    def test_heldout_disagreement_does_not_rewrite_or_promote_candidates(self):
        p,c=self.records();c[-1]=replace(c[-1],time_of_day_ns=0)
        j=candidate_join(p,c);h=clock_hypothesis(p,c,j)
        self.assertEqual(h['heldout_disagree'],1)
        self.assertTrue(all(r['label_status']=='UNKNOWN' for r in j['rows']))
    def test_exact_time_witness_resolves_duplicate_headers_without_labels(self):
        p,c=self.records()
        p[-1]=replace(p[-1],key=p[-2].key);c[-1]=replace(c[-1],key=c[-2].key)
        primary=candidate_join(p,c);self.assertEqual(primary['counts']['AMBIGUOUS'],2)
        h=clock_hypothesis(p,c,primary);result=time_witness_join(p,c,h)
        self.assertEqual(result['counts'],{'UNIQUE_CONTENT_AND_TIME_CANDIDATE':96})
        self.assertFalse(result['clock_conversion_verified']);self.assertFalse(result['label_assignment_performed'])
    def test_exact_duplicate_time_and_headers_remain_ambiguous(self):
        p,c=self.records();p[-1]=replace(p[-1],key=p[-2].key,time_of_day_ns=p[-2].time_of_day_ns)
        c[-1]=replace(c[-1],key=c[-2].key,time_of_day_ns=c[-2].time_of_day_ns)
        primary=candidate_join(p,c);h=clock_hypothesis(p,c,primary)
        self.assertEqual(time_witness_join(p,c,h)['counts']['AMBIGUOUS'],2)
    def test_failed_or_insufficient_clock_evidence_blocks_witness(self):
        p,c=self.records();h=clock_hypothesis(p,c,candidate_join(p,c))
        for patch_value in [{'heldout_disagree':1},{'heldout_pairs':0},{'heldout_pairs':15},{'calibration_pairs':0},{'labels_used':True}]:
            self.assertEqual(time_witness_join(p,c,{**h,**patch_value})['status'],'BLOCKED_ON_CLOCK_WITNESS')
    def test_no_nearest_timestamp_or_epsilon_repair(self):
        p,c=self.records();c[-1]=replace(c[-1],time_of_day_ns=c[-1].time_of_day_ns+1)
        # Dùng unchanged heldout proof từ sample control riêng; đích lệch1ns vẫn không match.
        clean_p,clean_c=self.records();h=clock_hypothesis(clean_p,clean_c,candidate_join(clean_p,clean_c))
        result=time_witness_join(p,c,h)
        self.assertEqual(result['counts']['UNMATCHED_WITHIN_SEARCH_PREFIX'],1)
    def test_two_PCAP_records_cannot_claim_one_CSV_timestamp(self):
        p,c=self.records();p[-1]=replace(p[-1],key=p[-2].key,time_of_day_ns=p[-2].time_of_day_ns)
        clean_p,clean_c=self.records();h=clock_hypothesis(clean_p,clean_c,candidate_join(clean_p,clean_c))
        result=time_witness_join(p,c,h)
        self.assertEqual(result['counts']['AMBIGUOUS'],2)


class BoundedCSVTests(unittest.TestCase):
    def test_identity_prefix_limit_and_source_unchanged(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/runs') as folder:
            path=Path(folder)/'sample.csv';data=b'a,b\n1,2\n3,4\n';path.write_bytes(data)
            rows,meta=csv_prefix(path,hashlib.sha256(data).hexdigest(),len(data),rows=1)
            self.assertEqual(len(rows),1);self.assertEqual(path.read_bytes(),data)
            with self.assertRaises(AlignmentError):csv_prefix(path,'0'*64,len(data))
            with self.assertRaises(AlignmentError):csv_prefix(path,hashlib.sha256(data).hexdigest(),len(data),rows=1025)
    def test_duplicate_header_and_wrong_width_fail(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/runs') as folder:
            for data in (b'a,a\n1,2\n',b'a,b\n1\n'):
                path=Path(folder)/'sample.csv';path.write_bytes(data)
                with self.assertRaises(AlignmentError):csv_prefix(path,hashlib.sha256(data).hexdigest(),len(data))


class ScalarIOTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts/runs');self.addCleanup(self.tmp.cleanup)
        self.folder=Path(self.tmp.name);self.source=self.folder/'source.pcap';self.source.write_bytes(b'a'*24)
        self.exe=self.folder/'tshark.test';self.exe.write_bytes(b'fake-test-executable')
        self.kwargs=dict(source_sha256=hashlib.sha256(self.source.read_bytes()).hexdigest(),size_bytes=24,
            tshark=self.exe,tshark_sha256=hashlib.sha256(self.exe.read_bytes()).hexdigest(),scratch=self.folder)
    def process(self,raw=None,stderr=b'',code=0,running=False):
        if raw is None:
            row=['1','1.000000001','192.0.2.1','192.0.2.2','6','40000','80','0x0002','0','0','0xabcd','0','1000','','','eth:ip:tcp']
            raw=('|'.join('"'+x+'"' for x in row)+'\n').encode()
        class Fake:
            pid=999999
            def __init__(self):self.returncode=None if running else code;self.terminated=False
            def poll(self):return self.returncode
            def wait(self,timeout=None):return self.returncode
            def terminate(self):self.terminated=True;self.returncode=-15
            def kill(self):self.returncode=-9
        fake=Fake()
        def factory(*args,**kwargs):
            kwargs['stdout'].write(raw);kwargs['stdout'].flush();kwargs['stderr'].write(stderr);kwargs['stderr'].flush()
            return fake
        return fake,factory
    def test_pinned_source_scalar_shape_and_owned_process_exit(self):
        fake,factory=self.process()
        with patch('nids_ml.alignment.io.subprocess.Popen',side_effect=factory):
            rows,meta=pcap_scalar_prefix(self.source,**self.kwargs)
        self.assertEqual(len(rows),1);self.assertEqual(set(rows[0]),set(SCALAR_FIELDS));self.assertTrue(meta['child_exited'])
        self.assertIn('-r',meta['command']);self.assertNotIn('-Y',meta['command']);self.assertFalse(fake.terminated)
    def test_nonzero_exit_and_output_limits_fail_closed(self):
        for args in (dict(code=1),dict(raw=b'x'*1048577),dict(stderr=b'x'*65537)):
            fake,factory=self.process(**args)
            with patch('nids_ml.alignment.io.subprocess.Popen',side_effect=factory),self.assertRaises(AlignmentError):
                pcap_scalar_prefix(self.source,**self.kwargs)
    def test_timeout_terminates_owned_process(self):
        fake,factory=self.process(running=True)
        with patch('nids_ml.alignment.io.subprocess.Popen',side_effect=factory),\
             patch('nids_ml.alignment.io.time.monotonic',side_effect=[0,31]),self.assertRaisesRegex(AlignmentError,'SCALAR_TIMEOUT'):
            pcap_scalar_prefix(self.source,**self.kwargs)
        self.assertTrue(fake.terminated)
    def test_scalar_ordinal_gap_and_invalid_budget_rejected(self):
        fake,factory=self.process(raw=b'"2"'+b'|""'*15+b'\n')
        with patch('nids_ml.alignment.io.subprocess.Popen',side_effect=factory),self.assertRaisesRegex(AlignmentError,'ORDINAL'):
            pcap_scalar_prefix(self.source,**self.kwargs)
        with self.assertRaises(AlignmentError):pcap_scalar_prefix(self.source,packets=257,**self.kwargs)


if __name__=='__main__':unittest.main()
