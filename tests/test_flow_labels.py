"""Test synthetic rule validation, packet decisions and strict window labels.
Test rule synthetic, quyết định packet và nhãn window nghiêm ngặt.
"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nids_ml.labeling.resolver import Label,LabelResolver,PacketDecision,WindowDecision,aggregate_window
from nids_ml.labeling.contracts import ContractError,RULE_SCHEMA,canonical_bytes,rule_digest,strict_json
from flow_label_fixtures import fixture,bundle,approve_synthetic


def resolver(rules,evidence):
    return LabelResolver(rules,evidence,approve_synthetic(rules))


def timed_packets(engine,packet,times):
    return [engine.resolve(dict(packet,ordinal=i+1,timestamp_ns=t)) for i,t in enumerate(times)]


def split_rules():
    rules,evidence,approval,packet=fixture()
    benign=deepcopy(rules['rules'][0])
    benign.update(rule_id='benign-event',label=0,scenario='BENIGN',source_role='benign',target_role='benign',predicate='BENIGN_EVENT')
    benign['interval']['end_ns']=150
    rules['rules'][0]['interval']['start_ns']=150
    rules['rules'].append(benign)
    return rules,evidence,packet


class ResolverTests(unittest.TestCase):
    def setUp(self):
        self.rules,self.evidence,self.approval,self.packet=fixture()
        self.engine=LabelResolver(self.rules,self.evidence,self.approval)

    def test_verified_ddos_and_numeric_label(self):
        d=self.engine.resolve(self.packet)
        self.assertIs(d.label,Label.VERIFIED_DDOS);self.assertEqual(d.numeric_label(),1)
        self.assertEqual(d.rule_ids,('rule-1',));self.assertEqual(d.evidence_ids,('fixture-evidence',))

    def test_verified_benign_requires_benign_rule(self):
        rules,evidence,approval,packet=fixture(0)
        d=LabelResolver(rules,evidence,approval).resolve(packet)
        self.assertIs(d.label,Label.VERIFIED_BENIGN);self.assertEqual(d.numeric_label(),0)

    def test_unmatched_not_benign(self):
        d=self.engine.resolve(dict(self.packet,src_ip='192.0.2.99'))
        self.assertIs(d.label,Label.UNKNOWN)
        with self.assertRaises(ContractError):d.numeric_label()

    def test_source_port_restriction(self):
        self.assertIs(self.engine.resolve(dict(self.packet,src_port=40001)).label,Label.UNKNOWN)

    def test_target_port_restriction(self):
        self.assertIs(self.engine.resolve(dict(self.packet,dst_port=8081)).label,Label.UNKNOWN)

    def test_reversed_traffic_unknown_by_default(self):
        p=dict(self.packet,src_ip=self.packet['dst_ip'],dst_ip=self.packet['src_ip'],src_port=8080,dst_port=40000)
        self.assertIs(self.engine.resolve(p).label,Label.UNKNOWN)

    def test_bidirectional_alone_does_not_label_responses(self):
        self.rules['rules'][0]['direction']='BIDIRECTIONAL'
        e=resolver(self.rules,self.evidence)
        self.assertIs(e.resolve(dict(self.packet,src_ip='192.0.2.20',dst_ip='192.0.2.10',src_port=8080,dst_port=40000)).label,Label.UNKNOWN)

    def test_explicit_verified_response_policy(self):
        rule=self.rules['rules'][0]
        rule.update(direction='BIDIRECTIONAL',response_policy='VERIFIED_SAME_LABEL',response_evidence_ids=['fixture-evidence'])
        e=resolver(self.rules,self.evidence)
        response=e.resolve(dict(self.packet,src_ip='192.0.2.20',dst_ip='192.0.2.10',src_port=8080,dst_port=40000,tcp_syn=False,tcp_ack=True))
        self.assertIs(response.label,Label.VERIFIED_DDOS)
        self.assertEqual(response.identity,e.resolve(self.packet).identity)

    def test_conflicting_verified_rules_yield_mixed(self):
        benign=deepcopy(self.rules['rules'][0]);benign.update(rule_id='conflict',label=0,scenario='BENIGN',predicate='BENIGN_EVENT',source_role='benign',target_role='benign')
        self.rules['rules'].append(benign)
        d=resolver(self.rules,self.evidence).resolve(self.packet)
        self.assertIs(d.label,Label.MIXED)
        with self.assertRaises(ContractError):d.numeric_label()

    def test_exact_start_inclusive_end_exclusive(self):
        for t,want in [(99,Label.UNKNOWN),(100,Label.VERIFIED_DDOS),(199,Label.VERIFIED_DDOS),(200,Label.UNKNOWN)]:
            with self.subTest(t=t):self.assertIs(self.engine.resolve(dict(self.packet,timestamp_ns=t)).label,want)

    def test_clock_uncertainty_at_both_boundaries(self):
        self.rules['capture']['segments'][0]['clock']['uncertainty_ns']=2
        e=resolver(self.rules,self.evidence)
        for t in [99,100,101,198,199,200,201]:
            with self.subTest(t=t):self.assertIs(e.resolve(dict(self.packet,timestamp_ns=t)).label,Label.UNKNOWN)
        self.assertIs(e.resolve(dict(self.packet,timestamp_ns=150)).label,Label.VERIFIED_DDOS)

    def test_clock_offset_applied(self):
        self.rules['capture']['segments'][0]['clock']['offset_ns']=100
        d=resolver(self.rules,self.evidence).resolve(dict(self.packet,timestamp_ns=50))
        self.assertEqual(d.timestamp_ns,150);self.assertEqual(d.label,Label.VERIFIED_DDOS)

    def test_unsupported_protocol_unknown(self):
        self.assertIs(self.engine.resolve(dict(self.packet,protocol='ARP')).label,Label.UNKNOWN)

    def test_ipv6_unknown(self):
        self.assertIs(self.engine.resolve(dict(self.packet,src_ip='2001:db8::1',dst_ip='2001:db8::2')).label,Label.UNKNOWN)

    def test_tcp_syn_ack_is_not_attack_syn(self):
        self.assertIs(self.engine.resolve(dict(self.packet,tcp_ack=True)).label,Label.UNKNOWN)

    def test_http_requires_application_and_method(self):
        rules,evidence,approval,p=fixture(scenario='HTTP');e=LabelResolver(rules,evidence,approval)
        self.assertIs(e.resolve(p).label,Label.VERIFIED_DDOS)
        self.assertIs(e.resolve(dict(p,http_method='POST')).label,Label.UNKNOWN)
        self.assertIs(e.resolve(dict(p,application='NONE',http_method='NONE')).label,Label.UNKNOWN)

    def test_udp_event_rule(self):
        r,e,a,p=fixture(scenario='UDP')
        self.assertIs(LabelResolver(r,e,a).resolve(p).label,Label.VERIFIED_DDOS)

    def test_icmp_request_not_reply(self):
        r,e,a,p=fixture(scenario='ICMP');engine=LabelResolver(r,e,a)
        self.assertIs(engine.resolve(p).label,Label.VERIFIED_DDOS)
        self.assertIs(engine.resolve(dict(p,icmp_type=0)).label,Label.UNKNOWN)

    def test_icmp_bad_code_and_identifier_flow_separation(self):
        r,e,a,p=fixture(scenario='ICMP');engine=LabelResolver(r,e,a)
        self.assertIs(engine.resolve(dict(p,icmp_code=1)).label,Label.UNKNOWN)
        with self.assertRaises(ContractError):
            aggregate_window([engine.resolve(p),engine.resolve(dict(p,ordinal=2,icmp_identifier=8))],full_window_complete=True)

    def test_inconsistent_transport_metadata_rejected(self):
        r,e,a,p=fixture(scenario='UDP');engine=LabelResolver(r,e,a)
        with self.assertRaises(ContractError):engine.resolve(dict(p,tcp_syn=True))
        with self.assertRaises(ContractError):self.engine.resolve(dict(self.packet,icmp_identifier=2))

    def test_bad_capture_hash_rejects_not_unknown_numeric(self):
        with self.assertRaisesRegex(ContractError,'Capture identity'):self.engine.resolve(dict(self.packet,capture_sha256='0'*64))

    def test_bad_capture_size_and_id(self):
        for key,value in [('capture_size_bytes',1),('capture_id','wrong-capture')]:
            with self.subTest(key=key),self.assertRaises(ContractError):self.engine.resolve(dict(self.packet,**{key:value}))

    def test_bad_segment_run_ordinal(self):
        for key,value in [('segment_id','missing'),('run_id','wrong-run'),('ordinal',10001)]:
            with self.subTest(key=key),self.assertRaises(ContractError):self.engine.resolve(dict(self.packet,**{key:value}))

    def test_invalid_fields_and_filename_volume_rejected(self):
        for change in [dict(ordinal=True),dict(timestamp_ns=150.0),dict(filename='DDoS.pcap'),dict(packet_count=999999),dict(src_ip='attacker.invalid')]:
            with self.subTest(change=change),self.assertRaises(ContractError):self.engine.resolve(dict(self.packet,**change))

    def test_rule_data_copied_and_results_deterministic(self):
        before=self.engine.resolve(self.packet)
        self.rules['rules'][0]['source_ip']='192.0.2.99'
        self.assertEqual(self.engine.resolve(self.packet),before)
        self.assertEqual(canonical_bytes(before.to_dict()),canonical_bytes(self.engine.resolve(self.packet).to_dict()))

    def test_metadata_output_has_no_cnn_features(self):
        d=self.engine.resolve(self.packet).to_dict()
        self.assertNotIn('features',d);self.assertNotIn('src_ip',d);self.assertNotIn('dst_ip',d)


class RuleValidationTests(unittest.TestCase):
    def test_missing_contract_fields_fail_closed(self):
        paths=[('schema_version',),('resolver_version',),('approval',),('evidence',),('capture','sha256'),
               ('capture','segments'),('rules',0,'target_ip'),('rules',0,'interval'),
               ('rules',0,'source_role'),('rules',0,'response_policy')]
        for path in paths:
            r,e,a,p=fixture();parent=r
            for key in path[:-1]:parent=parent[key]
            del parent[path[-1]]
            with self.subTest(path=path),self.assertRaises(ContractError):LabelResolver(r,e,a)

    def test_invalid_versions_scope_and_policies(self):
        for key,value in [('schema_version','9'),('resolver_version','other'),('input_kind','real_dataset'),('label_target','conversation')]:
            r,e,a,p=fixture();r[key]=value
            with self.subTest(key=key),self.assertRaises(ContractError):resolver(r,e)
        for key,value in [('unmatched','0'),('mixed','MAJORITY'),('ordering','SORT'),('max_packets',10)]:
            r,e,a,p=fixture();r['policies'][key]=value
            with self.subTest(key=key),self.assertRaises(ContractError):resolver(r,e)

    def test_rule_digest_and_approval_digest_binding(self):
        r,e,a,p=fixture();r['rules'][0]['target_ip']='192.0.2.30'
        with self.assertRaisesRegex(ContractError,'digest'):LabelResolver(r,e,a)
        r,e,a,p=fixture();r['rules'][0]['interval']['end_ns']=201;r['rule_set_sha256']=rule_digest(r)
        with self.assertRaisesRegex(ContractError,'Approval'):LabelResolver(r,e,a)

    def test_approval_missing_pending_wrong_scope_or_tampered(self):
        for key,value in [('status','PENDING'),('scope','dataset'),('record_sha256','0'*64)]:
            r,e,a,p=fixture();r['approval'][key]=value
            with self.subTest(key=key),self.assertRaises(ContractError):LabelResolver(r,e,a)
        r,e,a,p=fixture()
        with self.assertRaises(ContractError):LabelResolver(r,e,None)
        with self.assertRaises(ContractError):LabelResolver(r,e,a+b' ')

    def test_missing_or_mismatched_evidence(self):
        r,e,a,p=fixture()
        for docs in [{},{'fixture-evidence':b'wrong'},{'fixture-evidence':None}]:
            with self.subTest(docs=docs),self.assertRaises(ContractError):LabelResolver(r,docs,a)

    def test_unknown_evidence_reference(self):
        r,e,a,p=fixture();r['rules'][0]['evidence_ids']=['missing']
        with self.assertRaises(ContractError):resolver(r,e)

    def test_missing_response_evidence(self):
        r,e,a,p=fixture();r['rules'][0].update(direction='BIDIRECTIONAL',response_policy='VERIFIED_SAME_LABEL')
        with self.assertRaises(ContractError):resolver(r,e)

    def test_empty_reversed_or_float_intervals(self):
        for change in [dict(start_ns=200,end_ns=200),dict(start_ns=201,end_ns=200),dict(start_ns=100.0,end_ns=200)]:
            r,e,a,p=fixture();r['rules'][0]['interval']=change
            with self.subTest(change=change),self.assertRaises(ContractError):resolver(r,e)

    def test_wrong_roles_or_predicates(self):
        for change in [dict(source_role='benign'),dict(target_role='benign'),dict(scenario='BENIGN'),dict(protocol='UDP'),dict(predicate='BENIGN_EVENT'),dict(label=False)]:
            r,e,a,p=fixture();r['rules'][0].update(change)
            with self.subTest(change=change),self.assertRaises(ContractError):resolver(r,e)

    def test_invalid_ips_ports_spoofing_nat_clock(self):
        for change in [dict(source_ip='227.1.2.3'),dict(target_ip='bad'),dict(source_ports=[]),dict(target_ports=[65536]),dict(address_assumption='SPOOFED')]:
            r,e,a,p=fixture();r['rules'][0].update(change)
            with self.subTest(change=change),self.assertRaises(ContractError):resolver(r,e)
        for change in [dict(uncertainty_ns=-1),dict(basis='UTC_ASSUMED')]:
            r,e,a,p=fixture();r['capture']['segments'][0]['clock'].update(change)
            with self.subTest(change=change),self.assertRaises(ContractError):resolver(r,e)

    def test_duplicate_ids_and_overlapping_segments(self):
        for target in ['rules','evidence']:
            r,e,a,p=fixture();r[target].append(deepcopy(r[target][0]))
            with self.subTest(target=target),self.assertRaises(ContractError):resolver(r,e)
        r,e,a,p=fixture();seg=deepcopy(r['capture']['segments'][0]);seg['segment_id']='segment-2';r['capture']['segments'].append(seg)
        with self.assertRaises(ContractError):resolver(r,e)

    def test_rule_wrong_segment_or_run(self):
        for key in ['segment_id','run_id']:
            r,e,a,p=fixture();r['rules'][0][key]='missing'
            with self.subTest(key=key),self.assertRaises(ContractError):resolver(r,e)

    def test_segment_and_role_validity_not_global_ip_lookup(self):
        r,e,a,p=fixture();seg=deepcopy(r['capture']['segments'][0])
        r['capture']['segments'][0]['ordinal_end']=10
        seg.update(segment_id='segment-2',run_id='run-2',ordinal_start=11,ordinal_end=20)
        r['capture']['segments'].append(seg)
        d=resolver(r,e).resolve(dict(p,ordinal=11,segment_id='segment-2',run_id='run-2'))
        self.assertIs(d.label,Label.UNKNOWN)

    def test_strict_json_duplicate_nan_and_budget(self):
        for value in [b'{"a":1,"a":2}',b'{"a":NaN}',b'x'*1048577,b'\xff']:
            with self.subTest(size=len(value)),self.assertRaises(ContractError):strict_json(value)


class WindowTests(unittest.TestCase):
    def test_all_benign_or_ddos(self):
        for label in (0,1):
            r,e,a,p=fixture(label);engine=LabelResolver(r,e,a)
            w=aggregate_window(timed_packets(engine,p,[150,151]),full_window_complete=True)
            self.assertEqual(w.status,'ACCEPT');self.assertEqual(w.numeric_label(),label)

    def test_mixed_window_no_majority(self):
        r,e,p=split_rules();engine=resolver(r,e)
        w=aggregate_window(timed_packets(engine,p,[140]*19+[150]),full_window_complete=True)
        self.assertIs(w.full_window_label,Label.MIXED);self.assertEqual(w.status,'QUARANTINE')
        with self.assertRaises(ContractError):w.numeric_label()

    def test_unknown_window(self):
        r,e,a,p=fixture();engine=LabelResolver(r,e,a)
        w=aggregate_window(timed_packets(engine,p,[150,200]),full_window_complete=True)
        self.assertIs(w.full_window_label,Label.UNKNOWN)

    def test_explicit_mixed_packet_and_mixed_over_unknown(self):
        r,e,a,p=fixture();benign=deepcopy(r['rules'][0]);benign.update(rule_id='conflict',label=0,scenario='BENIGN',predicate='BENIGN_EVENT',source_role='benign',target_role='benign');r['rules'].append(benign)
        engine=resolver(r,e)
        w=aggregate_window(timed_packets(engine,p,[150,200]),full_window_complete=True)
        self.assertIs(w.full_window_label,Label.MIXED)

    def test_benign_ddos_unknown_priority_is_mixed(self):
        r,e,p=split_rules();engine=resolver(r,e)
        w=aggregate_window(timed_packets(engine,p,[140,150,200]),full_window_complete=True)
        self.assertIs(w.full_window_label,Label.MIXED)

    def test_empty_and_padding_only_reject_empty(self):
        for values in [[],[PacketDecision.padding()]*20]:
            w=aggregate_window(values,full_window_complete=True)
            self.assertEqual(w.status,'REJECT_EMPTY');self.assertIsNone(w.full_window_label)
            with self.assertRaises(ContractError):w.numeric_label()

    def test_padding_ignored(self):
        r,e,a,p=fixture(0);engine=LabelResolver(r,e,a)
        w=aggregate_window(timed_packets(engine,p,[150])+[PacketDecision.padding()]*19,full_window_complete=True)
        self.assertEqual(w.numeric_label(),0);self.assertEqual(w.padding_count,19);self.assertEqual(w.real_packet_count,1)

    def test_truncated_positive_packet_not_given_benign_label(self):
        r,e,p=split_rules();engine=resolver(r,e)
        w=aggregate_window(timed_packets(engine,p,list(range(100,120))+[150]),full_window_complete=True)
        self.assertIs(w.full_window_label,Label.MIXED);self.assertIs(w.retained_fragment_label,Label.VERIFIED_BENIGN)
        self.assertIn('TRUNCATION_LABEL_SCOPE_MISMATCH',w.reasons)

    def test_unknown_suffix_and_unexamined_suffix_quarantined(self):
        r,e,a,p=fixture();engine=LabelResolver(r,e,a)
        for times,complete in [(list(range(150,170))+[200],True),(list(range(150,170)),False)]:
            w=aggregate_window(timed_packets(engine,p,times),full_window_complete=complete)
            self.assertIs(w.full_window_label,Label.UNKNOWN);self.assertIs(w.retained_fragment_label,Label.VERIFIED_DDOS)
            self.assertEqual(w.status,'QUARANTINE')

    def test_homogeneous_known_overflow_accepts(self):
        r,e,a,p=fixture();engine=LabelResolver(r,e,a)
        w=aggregate_window(timed_packets(engine,p,list(range(150,171))),full_window_complete=True)
        self.assertEqual(w.numeric_label(),1);self.assertEqual(w.retained_packet_count,20);self.assertEqual(w.real_packet_count,21)

    def test_timestamp_regression_in_retained(self):
        r,e,a,p=fixture();engine=LabelResolver(r,e,a)
        w=aggregate_window(timed_packets(engine,p,[151,150]),full_window_complete=True)
        self.assertIs(w.full_window_label,Label.UNKNOWN);self.assertIs(w.retained_fragment_label,Label.UNKNOWN)
        self.assertIn('TIMESTAMP_REGRESSION',w.reasons)

    def test_timestamp_regression_after_retention(self):
        r,e,a,p=fixture();engine=LabelResolver(r,e,a)
        w=aggregate_window(timed_packets(engine,p,list(range(150,170))+[160]),full_window_complete=True)
        self.assertIs(w.full_window_label,Label.UNKNOWN);self.assertIs(w.retained_fragment_label,Label.VERIFIED_DDOS)

    def test_duplicate_ordinals_cross_flow_segment_or_rule_set_reject(self):
        r,e,a,p=fixture();engine=LabelResolver(r,e,a);one=engine.resolve(p)
        with self.assertRaises(ContractError):aggregate_window([one,one],full_window_complete=True)
        two=engine.resolve(dict(p,ordinal=2,src_port=40001))
        with self.assertRaises(ContractError):aggregate_window([one,two],full_window_complete=True)
        r['rule_set_id']='another-rule-set';other=resolver(r,e).resolve(dict(p,ordinal=2))
        with self.assertRaises(ContractError):aggregate_window([one,other],full_window_complete=True)

    def test_padding_interleaved_or_fake_numeric_labels_reject(self):
        r,e,a,p=fixture();engine=LabelResolver(r,e,a)
        with self.assertRaises(ContractError):aggregate_window([PacketDecision.padding(),engine.resolve(p)],full_window_complete=True)
        with self.assertRaises(ContractError):PacketDecision(0,())
        with self.assertRaises(ContractError):PacketDecision(Label.VERIFIED_BENIGN,(),is_padding=True)

    def test_completeness_and_n_are_explicit(self):
        for kwargs in [dict(full_window_complete=1),dict(full_window_complete=True,max_packets=10)]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ContractError):aggregate_window([],**kwargs)

    def test_stream_and_list_results_equal(self):
        r,e,a,p=fixture();engine=LabelResolver(r,e,a);items=timed_packets(engine,p,[150,151])
        self.assertEqual(aggregate_window(items,full_window_complete=True),aggregate_window(iter(items),full_window_complete=True))

    def test_numeric_label_rejects_forged_accept_unknown(self):
        w=WindowDecision(Label.UNKNOWN,Label.UNKNOWN,'ACCEPT',(),1,1,0,True)
        with self.assertRaises(ContractError):w.numeric_label()

    def test_window_input_budget(self):
        with self.assertRaisesRegex(ContractError,'budget'):
            aggregate_window((PacketDecision.padding() for _ in range(100001)),full_window_complete=True)


class CliTests(unittest.TestCase):
    def run_cli(self,*args):
        return subprocess.run([sys.executable,'-B',str(ROOT/'scripts/validate_flow_labels.py'),*map(str,args)],
                              cwd=ROOT,capture_output=True,text=True,timeout=15)

    def test_schema_export_matches_checked_in_schema(self):
        result=self.run_cli('schema');self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout),RULE_SCHEMA)
        check=self.run_cli('schema','--check',ROOT/'contracts/flow-label-rules.schema.json')
        self.assertEqual(check.returncode,0,check.stdout)

    def test_all_cli_commands_and_deterministic_output(self):
        path=ROOT/'configs/flow-labels-synthetic.example.json'
        before=path.read_bytes()
        for cmd in ['validate','decide','aggregate']:
            a=self.run_cli(cmd,'--fixture',path);b=self.run_cli(cmd,'--fixture',path)
            self.assertEqual(a.returncode,0,a.stdout+a.stderr);self.assertEqual(a.stdout,b.stdout)
            result=json.loads(a.stdout);self.assertFalse(result['real_data_ready'])
            if cmd=='aggregate':self.assertEqual(result['window']['status'],'ACCEPT')
        self.assertEqual(path.read_bytes(),before)

    def test_cli_missing_approval_and_real_scope_rejected(self):
        with tempfile.TemporaryDirectory(prefix='flow-label-',dir=ROOT/'artifacts/runs') as tmp:
            path=Path(tmp)/'fixture.json';value=bundle();del value['rule_set']['approval']
            path.write_bytes(canonical_bytes(value))
            result=self.run_cli('validate','--fixture',path)
            self.assertEqual(result.returncode,2);self.assertEqual(json.loads(result.stdout)['status'],'FAIL_CLOSED')
            value=bundle();value['kind']='real_dataset';path.write_bytes(canonical_bytes(value))
            self.assertEqual(self.run_cli('decide','--fixture',path).returncode,2)

    def test_cli_rejects_non_json_and_invalid_schema(self):
        with tempfile.TemporaryDirectory(prefix='flow-label-',dir=ROOT/'artifacts/runs') as tmp:
            p=Path(tmp)/'fixture.pcap';p.write_text('{}')
            self.assertEqual(self.run_cli('decide','--fixture',p).returncode,2)
            s=Path(tmp)/'schema.json';s.write_text('{}')
            self.assertEqual(self.run_cli('schema','--check',s).returncode,2)


if __name__=='__main__':
    unittest.main()
