"""Validate complete TCP pair audits using synthetic CSV/PCAP and a fake TShark child.
Kiểm audit TCP toàn cặp bằng CSV/PCAP giả và TShark child mô phỏng, không đọc dataset thật.
"""
from contextlib import redirect_stdout
from dataclasses import asdict
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tests'))
from pcap_fixtures import frame,pcap
from test_full_alignment import FakeProcess
from nids_ml.alignment.full import FullAuditLimits,FieldStream
from nids_ml.alignment.io import AUDIT_SCALAR_FIELDS,SCALAR_FIELDS
from nids_ml.cli.full_alignment import full_audit,guarded_tcp_record,validate_audit_config,main
from nids_ml.alignment.candidates import AlignmentError


class TcpPairV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='tcp-v2-fixture-',dir=ROOT/'artifacts/runs')
        self.work=Path(self.tmp.name);(self.work/'data/raw').mkdir(parents=True)
        (self.work/'artifacts/runs').mkdir(parents=True)
        self.tool=self.work/'synthetic-tshark-not-executable';self.tool.write_bytes(b'FAKE TOOL; DO NOT EXECUTE')
        self.pc=self.work/'data/raw/example.pcap';self.cs=self.work/'data/raw/example.csv'
        self.rows=[];packets=[]
        for i in range(1,97):
            payload=frame(ack=2000+i)
            packets.append((1609459200000000+i*1000000,payload))
            self.rows.append(dict(zip(AUDIT_SCALAR_FIELDS,['']*len(AUDIT_SCALAR_FIELDS))))
            self.rows[-1].update({'frame.number':str(i),'frame.time_epoch':str(1609459200+i),
                'ip.src':'192.0.2.10','ip.dst':'192.0.2.20','ip.proto':'6','tcp.srcport':'40000',
                'tcp.dstport':'8080','tcp.flags':'0x0012','tcp.len':'0','tcp.ack_raw':str(2000+i),
                'tcp.checksum':str(int.from_bytes(payload[50:52],'big')),'tcp.seq':'1000',
                'ip.flags.mf':'False','ip.frag_offset':'0','icmp.checksum':'0','icmp.seq_le':'0'})
        self.pc.write_bytes(pcap(packets))
        self.write_csv()
        self.config=dict(schema_version='1.1.0',kind='full_pair_mapping_audit',case_id='synthetic-tcp-pair',
            key_profile='tcp-v4-fragment-guard/1.0.0',expected_pcap_records=96,expected_csv_rows=96,
            pcap=self.identity(self.pc),csv=self.identity(self.cs),limits=asdict(FullAuditLimits()),
            toolchain=dict(tshark_path=str(self.tool),tshark_sha256=hashlib.sha256(self.tool.read_bytes()).hexdigest()))
        self.disk=patch('shutil.disk_usage',return_value=SimpleNamespace(free=10*1024**3));self.disk.start()

    def tearDown(self):
        self.disk.stop()
        assert self.work.resolve().parent==(ROOT/'artifacts/runs').resolve()
        self.tmp.cleanup()

    def identity(self,path):
        return dict(path=path.relative_to(self.work).as_posix(),size_bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def write_csv(self,label='0',kind='Normal'):
        fields=['frame.time','ip.src_host','ip.dst_host','tcp.srcport','tcp.dstport','tcp.len',
                'tcp.flags','tcp.ack_raw','tcp.checksum','tcp.seq','icmp.checksum','icmp.seq_le',
                'Attack_label','Attack_type']
        fields+=['unused_'+str(i) for i in range(63-len(fields))]
        with self.cs.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for i,row in enumerate(self.rows,1):
                item={name:row.get(name,'') for name in fields}
                item.update({'frame.time':f'2021 01:{i//60:02d}:{i%60:02d}',
                    'ip.src_host':row['ip.src'],'ip.dst_host':row['ip.dst'],
                    'Attack_label':label,'Attack_type':kind})
                writer.writerow(item)

    def child(self,rows=None,code=0):
        stream=io.StringIO();writer=csv.writer(stream,delimiter='|',quoting=csv.QUOTE_ALL,lineterminator='\n')
        for row in self.rows if rows is None else rows:writer.writerow([row.get(key,'') for key in AUDIT_SCALAR_FIELDS])
        return FakeProcess(stream.getvalue().encode(),code)

    def run_audit(self,name='result',child=None):
        with patch('nids_ml.alignment.full.subprocess.Popen',return_value=self.child() if child is None else child):
            return full_audit(self.config,self.work,self.work/'artifacts/runs'/name)

    def test_whole_pair_eof_one_to_one_and_labels_not_imported(self):
        before=self.pc.read_bytes(),self.cs.read_bytes()
        report=self.run_audit()
        self.assertEqual(report['PCAP']['records'],96);self.assertEqual(report['CSV']['rows'],96)
        self.assertEqual(report['mapping']['content_counts'],{'UNIQUE_CONTENT_CANDIDATE':96})
        self.assertEqual(report['mapping']['time_counts'],{'UNIQUE_CONTENT_AND_TIME_CANDIDATE':96})
        self.assertTrue(report['mapping']['witnessed_max_one_to_one'])
        self.assertFalse(report['labels_assigned']);self.assertFalse(report['training'])
        self.assertEqual((self.pc.read_bytes(),self.cs.read_bytes()),before)

    def test_labels_never_change_candidate_selection(self):
        first=self.run_audit('first')['mapping']
        self.write_csv('1','DDoS_HTTP');self.config['csv']=self.identity(self.cs)
        second=self.run_audit('second')['mapping']
        self.assertEqual(first,second)

    def test_fragment_and_udp_retained_in_denominator(self):
        self.rows[0]['ip.flags.mf']='True';self.rows[1]['ip.proto']='17'
        report=self.run_audit()
        self.assertEqual(report['PCAP']['records'],96)
        self.assertEqual(report['mapping']['content_counts']['UNSUPPORTED_OR_INVALID'],2)
        self.assertEqual(report['mapping']['content_counts']['UNIQUE_CONTENT_CANDIDATE'],94)

    def test_ambiguity_not_resolved_by_matching_label(self):
        for key in AUDIT_SCALAR_FIELDS:
            if key not in ('frame.number','frame.time_epoch'):self.rows[1][key]=self.rows[0][key]
        self.write_csv();self.config['csv']=self.identity(self.cs)
        result=self.run_audit()['mapping']
        self.assertEqual(result['content_counts']['AMBIGUOUS'],2)

    def test_wrong_pcap_count_and_nonzero_child_do_not_publish_coverage(self):
        for name,child in [('missing-record',self.child(self.rows[:-1])),('child-error',self.child(code=1))]:
            with self.subTest(name=name),self.assertRaises(ValueError):self.run_audit(name,child)
            output=self.work/'artifacts/runs'/name
            self.assertTrue((output/'FAILED.json').exists());self.assertFalse((output/'coverage.json').exists())

    def test_wrong_csv_count_rejected_before_child_start(self):
        self.config['expected_csv_rows']=97
        with patch('nids_ml.alignment.full.subprocess.Popen') as child:
            with self.assertRaisesRegex(AlignmentError,'WHOLE_CSV_COUNT'):
                full_audit(self.config,self.work,self.work/'artifacts/runs/result')
            child.assert_not_called()

    def test_legacy_http_and_v2_config_guards(self):
        legacy=json.loads((ROOT/'configs/full-pair-http-v1.json').read_text())
        self.assertFalse(validate_audit_config(legacy)[1]);self.assertTrue(validate_audit_config(self.config)[1])
        for key,value in [('expected_pcap_records',True),('expected_csv_rows',250000),('key_profile','guess')]:
            with self.subTest(key=key),self.assertRaises(AlignmentError):
                validate_audit_config(dict(self.config,**{key:value}))

    def test_csv_protocol_conflicts_and_missing_ports_fail_closed(self):
        row=dict(self.rows[0]);row.update({'ip.src_host':row['ip.src'],'ip.dst_host':row['ip.dst']})
        kwargs=dict(kind='csv',source_sha256='a'*64,ordinal=170000,ordinal_ceiling=250000)
        self.assertIsNotNone(guarded_tcp_record(row,**kwargs).key)
        row['icmp.checksum']='1';self.assertIsNone(guarded_tcp_record(row,**kwargs).key)
        row.update({'icmp.checksum':'0','tcp.srcport':'0','tcp.dstport':'0'})
        self.assertIsNone(guarded_tcp_record(row,**kwargs).key)

    def test_field_profile_rejects_arbitrary_extension(self):
        with self.assertRaises(AlignmentError):FieldStream([],SimpleNamespace(),fields=SCALAR_FIELDS+('unapproved',))

    def test_cli_runs_synthetic_complete_pair(self):
        config=self.work/'config.json';config.write_text(json.dumps(self.config))
        with patch('nids_ml.alignment.full.subprocess.Popen',return_value=self.child()),redirect_stdout(io.StringIO()) as stream:
            code=main(['--config',str(config),'--workspace',str(self.work),'--output',str(self.work/'artifacts/runs/cli')])
        self.assertEqual(code,0)
        self.assertEqual(json.loads(stream.getvalue().splitlines()[-1])['status'],'COMPLETE_FULL_PAIR_AUDIT')


if __name__=='__main__':unittest.main()
