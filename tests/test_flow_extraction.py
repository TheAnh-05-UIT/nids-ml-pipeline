"""Test packet features, flow windows, resource bounds and CNN input compatibility.
Test packet features, flow window, budget và tương thích input CNN.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import tracemalloc
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nids_ml.flows.features import (CaptureContext,ExtractionError,UnsupportedPacketError,ResourceLimitError,
    PACKET_SCHEMA,PROTOCOL_BITS,configuration,encode_protocol_stack,extract_features,normalize_row,validate_configuration)
from nids_ml.flows.extractor import (BASE_RESERVATION,FLOW_OVERHEAD_RESERVATION,RECORD,
    ExtractionComplete,FlowExtractor,FlowTensor,PacketReceipt,ResourceLimits)
from nids_ml.labeling.contracts import canonical_bytes
from flow_extraction_fixtures import context,packet,bundle


def engine(profile='flow-cnn-n20-t10',limits=None):
    return FlowExtractor(CaptureContext.from_dict(context()),configuration(profile),limits)


def tensors(packets,profile='flow-cnn-n20-t10',complete=True):
    e=engine(profile);events=list(e.run(packets,input_complete=complete))
    return e,[x for x in events if isinstance(x,FlowTensor)],events


class FeatureTests(unittest.TestCase):
    def test_tcp_exact_eleven_golden_features(self):
        p=extract_features(packet())
        self.assertEqual(p.features,(0.,60,78354535,2,576,20,123,18,4096,0,0))
        self.assertEqual(p.forward_tuple,('192.0.2.10',40000,'192.0.2.20',8080,6))

    def test_udp_and_icmp_golden_features(self):
        self.assertEqual(extract_features(packet(protocol=17)).features,(0.,52,63782488,2,2116,0,0,0,0,32,0))
        self.assertEqual(extract_features(packet(protocol=1)).features,(0.,64,97525412,2,96,0,0,0,0,0,8))

    def test_highest_layer_hash_preserves_case(self):
        p=packet();p['highest_layer']='tcp'
        self.assertEqual(extract_features(p).features[2],6923585)

    def test_protocol_vocabulary_lexical_not_original_list_order(self):
        self.assertEqual(PROTOCOL_BITS,{'arp':1,'data':2,'dns':4,'ftp':8,'http':16,'icmp':32,'ip':64,'ssdp':128,'ssl':256,'tcp':512,'telnet':1024,'udp':2048})
        self.assertEqual(encode_protocol_stack('IP:TCP:tcp:telnet'),(64+512+1024,()))

    def test_unknown_tokens_reported_without_tls_alias(self):
        bits,unknown=encode_protocol_stack('eth:IP:tcp:tls')
        self.assertEqual(bits,576);self.assertEqual(unknown,('eth','tls'))

    def test_reverse_tuple_same_canonical_identity(self):
        for proto in [6,17,1]:
            with self.subTest(proto=proto):
                a,b=extract_features(packet(proto)),extract_features(packet(proto,reverse=True))
                self.assertEqual(a.canonical_tuple,b.canonical_tuple)
                self.assertEqual(a.forward_tuple,(b.forward_tuple[2],b.forward_tuple[3],b.forward_tuple[0],b.forward_tuple[1],b.forward_tuple[4]))

    def test_metadata_not_input_channels(self):
        p=packet();q=deepcopy(p);q['src_ip']='198.51.100.3';q['tcp']['src_port']=1234
        self.assertEqual(extract_features(p).features,extract_features(q).features)
        for extra in ['label','filename','traffic_volume']:
            q=deepcopy(p);q[extra]='anything'
            with self.subTest(extra=extra),self.assertRaises(ExtractionError):extract_features(q)

    def test_missing_nonfinite_bool_out_of_range_fields(self):
        cases=[]
        for key in ['tcp','ip_length','timestamp_ns','highest_layer']:
            p=packet();del p[key];cases.append(p)
        for key,value in [('timestamp_ns',float('nan')),('timestamp_ns',float('inf')),('ordinal',True),('ip_length',65536),('ip_flags',8),('protocol_stack','x'*257),('src_ip','not-an-ip')]:
            p=packet();p[key]=value;cases.append(p)
        for key,value in [('ack',2**32),('window_size',65536),('src_port',-1),('length',21)]:
            p=packet();p['tcp'][key]=value;cases.append(p)
        for i,p in enumerate(cases):
            with self.subTest(case=i),self.assertRaises(ExtractionError):extract_features(p)

    def test_unsupported_protocol_ipv6_fragmented_fail_closed(self):
        for change in [dict(ip_protocol=47),dict(ip_version=6),dict(ip_version=0),dict(fragmented=True)]:
            p=packet();p.update(change)
            with self.subTest(change=change),self.assertRaises(UnsupportedPacketError):extract_features(p)

    def test_inconsistent_transport_lengths_and_real_scope_rejected(self):
        p=packet();p['udp']=packet(17)['udp']
        with self.assertRaises(ExtractionError):extract_features(p)
        p=packet(17);p['udp']['length']=33
        with self.assertRaises(ExtractionError):extract_features(p)
        p=packet();p['input_kind']='real_dataset'
        with self.assertRaises(ExtractionError):extract_features(p)

    def test_configuration_and_schema_parity(self):
        for profile,file in [('flow-cnn-n20-t10','flow-extraction-n20-t10.json'),('paper-table3-n10','flow-extraction-n10-t10.json')]:
            cfg=json.loads((ROOT/'configs'/file).read_text());self.assertEqual(validate_configuration(cfg),configuration(profile))
        self.assertEqual(json.loads((ROOT/'contracts/flow-packet.schema.json').read_text()),PACKET_SCHEMA)
        for key,value in [('feature_order',[]),('window_ns',1),('label_policy','FOLDER_LABELS'),('mode','unknown')]:
            cfg=configuration();cfg[key]=value
            with self.subTest(key=key),self.assertRaises(ExtractionError):validate_configuration(cfg)
        with self.assertRaises(ExtractionError):FlowExtractor(CaptureContext.from_dict(context()),{})


class WindowTests(unittest.TestCase):
    def test_short_flow_padding_relative_time_and_float32(self):
        p=packet(timestamp_ns=1700000000000000000);q=packet(ordinal=2,timestamp_ns=p['timestamp_ns']+2500000000)
        for profile,n in [('flow-cnn-n20-t10',20),('paper-table3-n10',10)]:
            e,windows,events=tensors([p,q],profile);w=windows[0]
            self.assertEqual(w.shape,(n,11,1));self.assertEqual(len(w.tensor_bytes),n*44)
            self.assertEqual(struct.unpack_from('<f',w.tensor_bytes,44)[0],0.25)
            self.assertEqual(w.tensor_bytes[88:],bytes((n-2)*44))
            self.assertEqual(w.metadata['real_packet_mask'],[True,True]+[False]*(n-2))
            self.assertFalse(w.metadata['supervised_eligible']);self.assertEqual(w.metadata['full_window_label'],'UNKNOWN')
            self.assertEqual(events[-1].status,'COMPLETE')

    def test_bidirectional_group_preserves_packet_order(self):
        _,windows,_=tensors([packet(),packet(ordinal=2,timestamp_ns=1,reverse=True)])
        self.assertEqual(len(windows),1);m=windows[0].metadata
        self.assertEqual(m['retained_ordinals'],[1,2]);self.assertEqual(m['forward_packet_count'],1);self.assertEqual(m['reverse_packet_count'],1)

    def test_window_exact_ten_seconds_inclusive(self):
        _,windows,_=tensors([packet(),packet(ordinal=2,timestamp_ns=10000000000),packet(ordinal=3,timestamp_ns=10000000001)])
        self.assertEqual([w.metadata['real_packet_count'] for w in windows],[2,1])
        self.assertEqual(struct.unpack_from('<f',windows[0].tensor_bytes,44)[0],1.0)

    def test_anchor_is_capture_global_not_per_flow(self):
        _,windows,_=tensors([packet(),packet(ordinal=2,timestamp_ns=9000000000,flow=1),packet(ordinal=3,timestamp_ns=11000000000,flow=1)])
        self.assertEqual([w.metadata['window_index'] for w in windows],[0,0,1])

    def test_long_gap_does_not_emit_empty_windows(self):
        _,windows,_=tensors([packet(),packet(ordinal=2,timestamp_ns=1000000000000)])
        self.assertEqual(len(windows),2);self.assertEqual([w.metadata['window_index'] for w in windows],[0,1])
        self.assertEqual(windows[1].metadata['anchor_timestamp_ns'],1000000000000)

    def test_first_seen_flow_order_deterministic_flushing(self):
        _,windows,_=tensors([packet(flow=2),packet(ordinal=2,flow=1),packet(ordinal=3,timestamp_ns=11000000000)])
        self.assertEqual([w.metadata['first_ordinal'] for w in windows],[1,2,3])

    def test_overflow_keeps_all_receipts_counts_and_digest(self):
        source=[packet(ordinal=i+1,timestamp_ns=i) for i in range(25)]
        for profile,n in [('flow-cnn-n20-t10',20),('paper-table3-n10',10)]:
            e,windows,events=tensors(source,profile);m=windows[0].metadata
            receipts=[x for x in events if isinstance(x,PacketReceipt)]
            self.assertEqual(len(receipts),25);self.assertEqual(sum(x.retained for x in receipts),n)
            self.assertEqual(m['real_packet_count'],25);self.assertEqual(m['overflow_packet_count'],25-n)
            self.assertEqual(m['last_ordinal'],25);self.assertEqual(m['retained_ordinals'],list(range(1,n+1)))
            self.assertEqual(e.stats['overflow_packets'],25-n)

    def test_overflow_content_changes_full_evidence_not_retained_tensor(self):
        a=[packet(ordinal=i+1,timestamp_ns=i) for i in range(21)];b=deepcopy(a);b[-1]['tcp']['ack']=999
        _,wa,_=tensors(a);_,wb,_=tensors(b)
        self.assertEqual(wa[0].tensor_bytes,wb[0].tensor_bytes)
        self.assertNotEqual(wa[0].metadata['full_packet_digest'],wb[0].metadata['full_packet_digest'])
        self.assertNotEqual(wa[0].metadata['overflow_packet_digest'],wb[0].metadata['overflow_packet_digest'])

    def test_regression_even_after_overflow_fails_no_completion(self):
        source=[packet(ordinal=i+1,timestamp_ns=i) for i in range(21)];source.append(packet(ordinal=22,timestamp_ns=1))
        e=engine();events=[]
        with self.assertRaisesRegex(ExtractionError,'Timestamp regression'):
            for event in e.run(source,input_complete=True):events.append(event)
        self.assertEqual(e.status,'FAILED');self.assertFalse(any(isinstance(x,ExtractionComplete) for x in events))
        self.assertEqual(e.stats['active_flows'],0)

    def test_global_regression_across_different_flow(self):
        with self.assertRaises(ExtractionError):tensors([packet(timestamp_ns=5),packet(ordinal=2,timestamp_ns=4,flow=1)])

    def test_duplicate_ordinal_and_capture_run_segment_mismatch(self):
        for change in [dict(ordinal=1),dict(capture=dict(context(),capture_id='other')),dict(capture=dict(context(),run_id='other')),dict(capture=dict(context(),segment_id='other'))]:
            p=packet(ordinal=2,timestamp_ns=1);p.update(change)
            with self.subTest(change=change),self.assertRaises(ExtractionError):tensors([packet(),p])

    def test_malformed_overflow_not_ignored(self):
        source=[packet(ordinal=i+1,timestamp_ns=i) for i in range(20)];bad=packet(ordinal=21,timestamp_ns=21);del bad['tcp']['ack'];source.append(bad)
        with self.assertRaises(ExtractionError):tensors(source)

    def test_empty_and_partial_input(self):
        e,windows,events=tensors([]);self.assertEqual(windows,[]);self.assertEqual(events[-1].stats_json,json.dumps(e.stats,sort_keys=True,separators=(',',':')).encode())
        e,windows,_=tensors([packet()],complete=False)
        self.assertEqual(e.status,'PARTIAL');self.assertFalse(windows[0].metadata['full_window_complete'])

    def test_iterator_closed_and_no_replay(self):
        e=engine();it=e.run([packet()],input_complete=True);next(it);it.close();self.assertEqual(e.status,'ABORTED')
        with self.assertRaises(ExtractionError):list(e.run([],input_complete=True))
        e=engine();it=e.run([],input_complete=True);self.assertEqual(next(it).status,'COMPLETE');it.close();self.assertEqual(e.status,'COMPLETE')

    def test_icmp_identifiers_do_not_silently_change_lucid_five_tuple(self):
        a=packet(1);b=packet(1,ordinal=2,timestamp_ns=1);b['icmp']['identifier']=9
        _,windows,_=tensors([a,b])
        self.assertEqual(len(windows),1);self.assertTrue(windows[0].metadata['icmp_identifier_changed'])
        self.assertFalse(windows[0].metadata['supervised_eligible'])

    def test_deterministic_json_and_tensor_bytes(self):
        source=bundle()['packets'];_,wa,ea=tensors(source);_,wb,eb=tensors(deepcopy(source))
        self.assertEqual([w.tensor_bytes for w in wa],[w.tensor_bytes for w in wb])
        self.assertEqual(canonical_bytes([x.to_dict() for x in ea]),canonical_bytes([x.to_dict() for x in eb]))


class ResourceNormalizationTests(unittest.TestCase):
    def test_active_flow_limit_fails_without_eviction(self):
        e=engine(limits=ResourceLimits(max_active_flows=1))
        with self.assertRaisesRegex(ResourceLimitError,'Active flow'):
            list(e.run([packet(),packet(ordinal=2,flow=1)],input_complete=True))
        self.assertEqual(e.stats['peak_active_flows'],1);self.assertEqual(e.status,'FAILED')

    def test_memory_budget_checked_before_allocation(self):
        budget=BASE_RESERVATION+FLOW_OVERHEAD_RESERVATION+20*RECORD.size-1
        e=engine(limits=ResourceLimits(memory_budget_bytes=budget))
        with self.assertRaisesRegex(ResourceLimitError,'memory budget'):list(e.run([packet()],input_complete=True))
        self.assertEqual(e.stats['peak_retained_buffer_bytes'],0)

    def test_window_and_input_limits_and_invalid_limits(self):
        for limits in [ResourceLimits(max_packets_per_window=2),ResourceLimits(max_input_packets=2)]:
            with self.subTest(limits=limits),self.assertRaises(ResourceLimitError):
                list(engine(limits=limits).run([packet(ordinal=i+1) for i in range(3)],input_complete=True))
        for limits in [ResourceLimits(max_active_flows=0),ResourceLimits(max_active_flows=True),ResourceLimits(memory_budget_bytes=1)]:
            with self.subTest(limits=limits),self.assertRaises(ExtractionError):engine(limits=limits)

    def test_new_window_releases_active_capacity(self):
        e=engine(limits=ResourceLimits(max_active_flows=1))
        list(e.run([packet(),packet(ordinal=2,timestamp_ns=11000000000,flow=1)],input_complete=True))
        self.assertEqual(e.status,'COMPLETE');self.assertEqual(e.stats['peak_active_flows'],1)

    def test_streaming_memory_bounded_for_many_packets(self):
        def source():
            ordinal=0
            for window in range(10):
                for i in range(200):
                    ordinal+=1;yield packet(ordinal=ordinal,timestamp_ns=window*11000000000+i,flow=i%8)
        e=engine(limits=ResourceLimits(max_active_flows=8));tracemalloc.start()
        try:
            count=sum(1 for _ in e.run(source(),input_complete=True));current,peak=tracemalloc.get_traced_memory()
        finally:tracemalloc.stop()
        self.assertEqual(count,2000+80+1);self.assertEqual(e.stats['peak_retained_buffer_bytes'],8*20*96)
        self.assertEqual(e.stats['active_flows'],0);self.assertLess(peak,2*1048576)

    def test_static_normalization_golden_and_no_fit(self):
        raw=extract_features(packet()).features;row=normalize_row(raw,2500000000)
        self.assertEqual(row[0],0.25);self.assertEqual(row[1],60/65536);self.assertEqual(row[2],78354535/4294967296)
        self.assertEqual(row[4],576/4096)
        self.assertEqual(normalize_row(raw,2500000000),row)

    def test_invalid_normalization_rejected_not_nan_to_num(self):
        raw=list(extract_features(packet()).features)
        for bad in [-1,10000000001,True]:
            with self.subTest(bad=bad),self.assertRaises(ExtractionError):normalize_row(raw,bad)
        for index,value in [(0,float('nan')),(1,float('inf')),(4,4097),(6,-1)]:
            values=raw.copy();values[index]=value
            with self.subTest(index=index),self.assertRaises(ExtractionError):normalize_row(values,0)


class ExtractionCliTests(unittest.TestCase):
    def run_cli(self,*args):
        return subprocess.run([sys.executable,'-B',str(ROOT/'scripts/validate_flow_extraction.py'),*map(str,args)],cwd=ROOT,capture_output=True,timeout=15)

    def test_actual_cli_profiles_deterministic_and_read_only(self):
        fixture=ROOT/'configs/flow-extraction-synthetic.example.json';before=fixture.read_bytes()
        for profile in ['flow-cnn-n20-t10','paper-table3-n10']:
            a=self.run_cli('validate','--fixture',fixture,'--profile',profile);b=self.run_cli('validate','--fixture',fixture,'--profile',profile)
            self.assertEqual(a.returncode,0,a.stdout+a.stderr);self.assertEqual(a.stdout,b.stdout)
            value=json.loads(a.stdout);self.assertFalse(value['training']);self.assertEqual(value['stats']['emitted_tensors'],4)
        self.assertEqual(fixture.read_bytes(),before)

    def test_schema_cli(self):
        result=self.run_cli('schema','--check',ROOT/'contracts/flow-packet.schema.json')
        self.assertEqual(result.returncode,0,result.stdout)

    def test_non_json_real_scope_and_bad_identity_cli_reject(self):
        with tempfile.TemporaryDirectory(prefix='flow-extract-',dir=ROOT/'artifacts/runs') as tmp:
            path=Path(tmp)/'fixture.json'
            for change in ['real','identity','packet']:
                value=bundle()
                if change=='real':value['kind']='real_dataset'
                elif change=='identity':value['capture']['capture_sha256']='0'*64
                else:del value['packets'][0]['tcp']['ack']
                path.write_bytes(canonical_bytes(value));r=self.run_cli('validate','--fixture',path)
                self.assertEqual(r.returncode,2,r.stdout)
            p=Path(tmp)/'not-a-PCAP.pcap';p.write_text('{}')
            self.assertEqual(self.run_cli('validate','--fixture',p).returncode,2)


class CnnInputCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nids_ml.models.flow_cnn import backend
        try:cls.np,cls.tf=backend()
        except (ImportError,RuntimeError) as error:raise unittest.SkipTest('C1 backend unavailable: '+str(error))

    def test_extracted_n10_n20_C1_validator_and_forward_no_training(self):
        from nids_ml.models.flow_cnn import build_model,configuration as model_config,predict_flow,validate_inputs
        for profile in ['flow-cnn-n20-t10','paper-table3-n10']:
            _,windows,_=tensors(bundle()['packets'],profile)
            model=build_model(model_config(profile),compile_model=False)
            for window in windows:
                x=window.to_numpy();self.assertEqual(x.dtype,self.np.float32)
                validate_inputs(x,model_config(profile));prediction=predict_flow(model,x,model_config(profile))
                self.assertEqual(prediction.shape,(1,1));self.assertTrue(self.np.isfinite(prediction).all())
                self.assertFalse(window.metadata['supervised_eligible'])


if __name__=='__main__':unittest.main()
