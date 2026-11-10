"""Test timestamps, PDML parsing and decoder guards with synthetic inputs.
Test timestamp, PDML parser và decoder guards bằng input synthetic.
"""
from dataclasses import replace
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nids_ml.flows.features import (CaptureContext,ExtractionError,configuration,extract_features,
                                       packet_schema,validate_configuration)
from nids_ml.flows.extractor import FlowExtractor,FlowTensor
from nids_ml.pcap.decoder import (DecoderError,IneligiblePacket,DecodeLimits,Field,PDMLRecord,
                                     PDMLParser,TSharkDecoder,epoch_ns,file_sha256,map_record)
from flow_extraction_fixtures import context,packet


def record(ordinal=1):
    values={'frame.number':str(ordinal),'frame.time_epoch':'1.000000001','frame.len':'54','frame.cap_len':'54',
        'frame.protocols':'eth:ethertype:ip:tcp','ip.version':'4','ip.src':'192.0.2.10','ip.dst':'192.0.2.20',
        'ip.proto':'6','ip.len':'40','ip.hdr_len':'20','ip.flags':'0x02','ip.flags.mf':'False','ip.frag_offset':'0',
        'tcp.srcport':'40000','tcp.dstport':'8080','tcp.len':'0','tcp.ack':'1','tcp.ack_raw':'2001',
        'tcp.flags':'0x0010','tcp.window_size_value':'4096','tcp.window_size':'32768','tcp.hdr_len':'20'}
    fields=[Field(k,v,('0' if k=='ip.flags.mf' else None),k.split('.')[0]) for k,v in values.items()]
    fields.append(Field('timestamp','calendar text','1.000000001','geninfo'))
    return PDMLRecord(('geninfo','frame','eth','ip','tcp'),('geninfo','frame','eth','ip','tcp'),tuple(fields))


def change(rec,**values):
    return replace(rec,fields=tuple(replace(f,show=values.get(f.name,f.show)) for f in rec.fields))


def xml(ordinal=1):
    rec=record(ordinal);parts=['<packet>']
    for proto in rec.layers:
        parts.append('<proto name="'+proto+'">')
        for field in rec.fields:
            if field.protocol==proto:
                attrs=' name="'+field.name+'" show="'+field.show+'"'
                if field.value is not None:attrs+=' value="'+field.value+'"'
                parts.append('<field'+attrs+'/>')
        parts.append('</proto>')
    return (''.join(parts)+'</packet>').encode()


class TimestampTests(unittest.TestCase):
    def test_integer_exact_ns_and_microsecond_precision(self):
        self.assertEqual(epoch_ns('1637836529.350784000'),1637836529350784000)
        self.assertEqual(epoch_ns('0.000000001'),1)
        self.assertEqual(epoch_ns('1.000000000000000000000000000'),1000000000)

    def test_reject_nonfinite_fraction_rounding_and_range(self):
        for value in ['nan','inf','-1','1e2',' 1','1.000000000000000000000000001',str(2**63),'1.0000000001',True]:
            with self.subTest(value=value),self.assertRaises(DecoderError):epoch_ns(value)


class ParserTests(unittest.TestCase):
    def parse(self,raw,limits=None,chunk=1):
        parser=PDMLParser(limits);records=[]
        for i in range(0,len(raw),chunk):records.extend(parser.feed(raw[i:i+chunk]))
        records.extend(parser.feed(b'',final=True))
        return parser,records

    def test_chunk_boundary_and_multiple_records(self):
        for chunk in [1,7,8192]:
            p,records=self.parse(b'<?xml version="1.0" encoding="utf-8"?><pdml>'+xml()+xml(2)+b'</pdml>',chunk=chunk)
            self.assertTrue(p.finished);self.assertEqual([r.number('frame.number') for r in records],[1,2])
            self.assertEqual(records[0].timestamp_ns(),1000000001)

    def test_reject_DTD_entity_and_non_utf8(self):
        for body in [b'<!DOCTYPE pdml [<!ENTITY x "abc">]><pdml/>',
                     b'<?xml version="1.0" encoding="iso-8859-1"?><pdml/>',
                     b'<pdml>\xff</pdml>',b'<pdml>&bad;</pdml>']:
            with self.subTest(body=body),self.assertRaises(DecoderError):self.parse(body,chunk=8192)

    def test_reject_truncated_wrong_root_text_and_nesting(self):
        for body in [b'<pdml>'+xml(),b'<notpdml/>',b'<pdml>text</pdml>',b'<pdml><other/></pdml>',
                     b'<pdml><packet><packet/></packet></pdml>',b'<pdml/></pdml>']:
            with self.subTest(body=body),self.assertRaises(DecoderError):self.parse(body,chunk=8192)

    def test_record_stdout_count_depth_field_limits(self):
        cases=[(b'<pdml>'+xml()+xml(2)+b'</pdml>',DecodeLimits(packet_count=1)),
               (b'<pdml>'+xml()+b'</pdml>',DecodeLimits(max_fields=1)),
               (b'<pdml>'+xml()+b'</pdml>',DecodeLimits(max_depth=2)),
               (b'<pdml><packet><proto name="data" '+b'x="'+b'a'*2000+b'"/></packet></pdml>',DecodeLimits(record_bytes=1024)),
               (b'<pdml>'+b' '*2000+b'</pdml>',DecodeLimits(stdout_bytes=1024))]
        for raw,limits in cases:
            with self.subTest(limits=limits),self.assertRaises(DecoderError):self.parse(raw,limits,chunk=128)

    def test_parser_final_and_feed_size_guards(self):
        parser=PDMLParser();list(parser.feed(b'<pdml/>',final=True))
        with self.assertRaises(DecoderError):list(parser.feed(b''))
        with self.assertRaises(DecoderError):list(PDMLParser().feed(b' '*8193))


class MappingTests(unittest.TestCase):
    def setUp(self):self.ctx=CaptureContext.from_dict(context())

    def test_scalar_mapping_ack_and_unscaled_window(self):
        p,meta=map_record(record(),self.ctx)
        self.assertEqual(p['input_kind'],'decoded_pcap');self.assertEqual(p['schema_version'],'1.1.0')
        self.assertEqual(p['tcp']['ack'],1);self.assertEqual(meta['tcp_ack_raw'],2001)
        self.assertEqual(p['tcp']['window_size'],4096);self.assertEqual(meta['tcp_window_scaled'],32768)
        self.assertEqual(p['ip_flags'],2);self.assertEqual(p['tcp']['flags'],16)
        self.assertFalse(meta['supervised_eligible']);self.assertEqual(meta['label'],'UNKNOWN')

    def test_missing_duplicate_and_bad_numeric_fields(self):
        rec=record()
        for modified in [replace(rec,fields=rec.fields+(rec.fields[0],)),
                         replace(rec,fields=tuple(replace(f,protocol='tcp') if f.name=='ip.src' else f for f in rec.fields)),
                         replace(rec,fields=tuple(f for f in rec.fields if f.name!='tcp.ack')),
                         change(rec,**{'tcp.len':'NaN'}),change(rec,**{'ip.flags.mf':'maybe'}),
                         change(rec,**{'frame.time_epoch':'1.000000002'})]:
            with self.subTest(modified=modified),self.assertRaises(DecoderError):map_record(modified,self.ctx)

    def test_fragment_nested_ip_truncation_and_wrong_layer_order(self):
        rec=record()
        for modified in [change(rec,**{'ip.frag_offset':'1'}),change(rec,**{'frame.cap_len':'10'}),
                         replace(rec,all_protocols=rec.all_protocols+('ip',)),
                         replace(rec,layers=('frame','geninfo','ip','tcp'))]:
            with self.subTest(modified=modified),self.assertRaises(DecoderError):map_record(modified,self.ctx)

    def test_actual_header_lengths_not_only_minimum_headers(self):
        for values in [{'ip.hdr_len':'24'},{'tcp.hdr_len':'24'},{'ip.hdr_len':'21'},{'tcp.hdr_len':'100'}]:
            with self.subTest(values=values),self.assertRaises(DecoderError):map_record(change(record(),**values),self.ctx)

    def test_data_wrapper_highest_layer_matches_PyShark_property(self):
        rec=record();rec=replace(rec,layers=rec.layers+('fake-field-wrapper',))
        p,_=map_record(rec,self.ctx);self.assertEqual(p['highest_layer'],'DATA')

    def test_scopes_separated_and_old_configuration_unchanged(self):
        old=configuration();self.assertEqual(old['schema_version'],'1.0.0')
        self.assertEqual(old['input_kind'],'synthetic_fixture')
        self.assertEqual(packet_schema()['properties']['input_kind']['const'],'synthetic_fixture')
        new=configuration(input_kind='decoded_pcap');self.assertEqual(validate_configuration(new),new)
        p,_=map_record(record(),self.ctx)
        with self.assertRaises(ExtractionError):extract_features(p)
        with self.assertRaises(ExtractionError):extract_features(packet(),input_kind='decoded_pcap')
        with self.assertRaises(ExtractionError):configuration(input_kind='anything')

    def test_real_scope_core_shapes_and_unknown_metadata(self):
        p,_=map_record(record(),self.ctx)
        for profile,n in [('flow-cnn-n20-t10',20),('paper-table3-n10',10)]:
            e=FlowExtractor(self.ctx,configuration(profile,input_kind='decoded_pcap'))
            tensors=[x for x in e.run([p],input_complete=False) if isinstance(x,FlowTensor)]
            self.assertEqual(tensors[0].shape,(n,11,1));self.assertEqual(len(tensors[0].tensor_bytes),n*44)
            self.assertEqual(tensors[0].metadata['input_kind'],'decoded_pcap')
            self.assertFalse(tensors[0].metadata['supervised_eligible'])
            self.assertEqual(e.status,'PARTIAL')

    def test_timestamp_regression_and_ordinal_guards_preserved(self):
        p,_=map_record(record(),self.ctx);q={**p,'ordinal':2,'timestamp_ns':p['timestamp_ns']-1}
        e=FlowExtractor(self.ctx,configuration(input_kind='decoded_pcap'))
        with self.assertRaises(ExtractionError):list(e.run([p,q],input_complete=False))
        self.assertEqual(e.status,'FAILED');self.assertEqual(e.stats['active_flows'],0)

    def test_udp_mapping_header_and_data_layer(self):
        rec=record();fields=tuple(f for f in rec.fields if not f.name.startswith('tcp.'))
        fields+=tuple(Field(k,v,None,'udp') for k,v in {'udp.srcport':'40000','udp.dstport':'8080','udp.length':'16'}.items())
        rec=replace(rec,layers=('geninfo','frame','eth','ip','udp','fake-field-wrapper'),
                    all_protocols=('geninfo','frame','eth','ip','udp'),fields=fields)
        rec=change(rec,**{'ip.proto':'17','ip.len':'36','frame.protocols':'eth:ip:udp:data'})
        p,_=map_record(rec,self.ctx);self.assertEqual(p['udp']['length'],16);self.assertEqual(p['highest_layer'],'DATA')
        self.assertIsNone(p['tcp']);self.assertIsNone(p['icmp'])
        with self.assertRaises(DecoderError):map_record(change(rec,**{'udp.length':'17'}),self.ctx)

    def test_icmp_identifier_and_reverse_identity_preserved(self):
        rec=record();fields=tuple(f for f in rec.fields if not f.name.startswith('tcp.'))
        fields+=tuple(Field(k,v,None,'icmp') for k,v in {'icmp.type':'8','icmp.code':'0','icmp.ident':'7'}.items())
        rec=replace(rec,layers=('geninfo','frame','eth','ip','icmp'),
                    all_protocols=('geninfo','frame','eth','ip','icmp'),fields=fields)
        rec=change(rec,**{'ip.proto':'1','ip.len':'36','frame.protocols':'eth:ip:icmp:data'})
        p,side=map_record(rec,self.ctx);self.assertEqual(side['icmp_identifier'],7)
        reverse=change(rec,**{'ip.src':'192.0.2.20','ip.dst':'192.0.2.10','icmp.type':'0'})
        q,_=map_record(reverse,self.ctx)
        self.assertEqual(extract_features(p,input_kind='decoded_pcap').canonical_tuple,
                         extract_features(q,input_kind='decoded_pcap').canonical_tuple)
        no_id=replace(rec,fields=tuple(f for f in rec.fields if f.name!='icmp.ident'))
        with self.assertRaisesRegex(IneligiblePacket,'ICMP_IDENTIFIER'):map_record(no_id,self.ctx)

    def test_explicit_flow_caps_fail_without_eviction_and_accept_with_budget(self):
        from nids_ml.flows.extractor import ResourceLimits
        p,_=map_record(record(),self.ctx)
        packets=[{**p,'ordinal':i+1,'tcp':{**p['tcp'],'src_port':40000+i}} for i in range(40)]
        default=FlowExtractor(self.ctx,configuration(input_kind='decoded_pcap'))
        with self.assertRaisesRegex(ExtractionError,'Active flow'):list(default.run(packets,input_complete=False))
        self.assertEqual(default.status,'FAILED')
        explicit=FlowExtractor(self.ctx,configuration(input_kind='decoded_pcap'),ResourceLimits(max_active_flows=128))
        output=list(explicit.run(packets,input_complete=False))
        self.assertEqual(sum(isinstance(x,FlowTensor) for x in output),40)
        self.assertLessEqual(explicit.stats['peak_accounted_state_bytes'],1048576)

    def test_invalid_cli_flow_caps_fail_before_dataset_open(self):
        from argparse import Namespace
        from nids_ml.cli.pcap import validate_prefix
        for value in [0,257,True]:
            with self.subTest(value=value),self.assertRaisesRegex(DecoderError,'ACTIVE_FLOW_LIMIT'):
                validate_prefix(Namespace(max_active_flows=value))

    def test_decoded_burst_budget_explicit_and_synthetic_hardcap_preserved(self):
        from nids_ml.flows.extractor import ResourceLimits
        p,_=map_record(record(),self.ctx)
        packets=[{**p,'ordinal':i+1,'tcp':{**p['tcp'],'src_port':40000+i}} for i in range(256)]
        with self.assertRaises(ExtractionError):FlowExtractor(self.ctx,configuration(),ResourceLimits(max_active_flows=256))
        with self.assertRaises(ExtractionError):ResourceLimits(max_active_flows=257).validate(decoded_pcap=True)
        small=FlowExtractor(self.ctx,configuration(input_kind='decoded_pcap'),ResourceLimits(max_active_flows=256))
        with self.assertRaisesRegex(ExtractionError,'memory budget'):list(small.run(packets,input_complete=False))
        large=FlowExtractor(self.ctx,configuration(input_kind='decoded_pcap'),
            ResourceLimits(max_active_flows=256,memory_budget_bytes=2097152))
        self.assertEqual(sum(isinstance(x,FlowTensor) for x in large.run(packets,input_complete=False)),256)
        self.assertLessEqual(large.stats['peak_accounted_state_bytes'],2097152)
        self.assertEqual(large.stats['active_flows'],0)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='d2b-session-',dir=ROOT/'artifacts/runs')
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.source=self.root/'sample.pcap';self.source.write_bytes(b'0'*24)
        self.exe=self.root/'tshark.test';self.exe.write_bytes(b'not executable')

    def decoder(self,**kwargs):
        return TSharkDecoder(self.source,expected_sha256=file_sha256(self.source),expected_size=24,
            tshark=self.exe,tshark_sha256=file_sha256(self.exe),**kwargs)

    def test_source_hash_size_and_executable_hash_fail_before_start(self):
        for change in [dict(expected_sha256='0'*64),dict(expected_size=25),dict(tshark_sha256='0'*64)]:
            params=dict(expected_sha256=file_sha256(self.source),expected_size=24,tshark=self.exe,tshark_sha256=file_sha256(self.exe))
            params.update(change)
            with self.assertRaises(DecoderError):TSharkDecoder(self.source,**params)

    def test_source_change_close_new_and_limit_validation(self):
        d=self.decoder();before=self.source.stat();self.source.write_bytes(b'1'*24)
        # Chỉ fixture synthetic: buộc mtime đổi, tránh phụ thuộc độ phân giải clock Windows.
        # Decoder hiện kiểm file signature; same-signature content tamper là gap riêng.
        os.utime(self.source,ns=(before.st_atime_ns,before.st_mtime_ns+1000000000))
        self.assertNotEqual(self.source.stat().st_mtime_ns,before.st_mtime_ns)
        with self.assertRaises(DecoderError):list(d.records())
        d=self.decoder();d.close();self.assertEqual(d.status,'CLOSED')
        with self.assertRaises(DecoderError):list(d.records())
        for limit in [DecodeLimits(packet_count=True),DecodeLimits(timeout_seconds=float('nan')),DecodeLimits(packet_count=4097)]:
            with self.assertRaises(DecoderError):limit.validate()

    def fake_process(self,raw,exit_code=0):
        class Fake:
            def __init__(self):
                self.stdout=BytesIO(raw);self.stderr=BytesIO(b'');self.returncode=None;self.pid=123456;self.terminated=False
            def poll(self):return self.returncode
            def wait(self,timeout=None):self.returncode=exit_code;return exit_code
            def terminate(self):self.terminated=True;self.returncode=-15
            def kill(self):self.returncode=-9
        return Fake()

    def test_owned_child_completed_and_no_replay(self):
        fake=self.fake_process(b'<pdml>'+xml()+b'</pdml>');d=self.decoder()
        with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=fake),patch('nids_ml.pcap.decoder.process_rss',return_value=1024):
            records=list(d.records())
        self.assertEqual(len(records),1);self.assertEqual(d.status,'COMPLETE_BOUNDED_PREFIX')
        self.assertFalse(fake.terminated);self.assertTrue(d.report()['child_exited'])
        with self.assertRaises(DecoderError):list(d.records())

    def test_nonzero_exit_truncated_and_duplicate_ordinals_fail(self):
        for raw,code in [(b'<pdml>'+xml()+b'</pdml>',1),(b'<pdml>'+xml(),0),
                         (b'<pdml>'+xml()+xml()+b'</pdml>',0)]:
            d=self.decoder();fake=self.fake_process(raw,code)
            with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=fake),patch('nids_ml.pcap.decoder.process_rss',return_value=1024):
                with self.assertRaises(DecoderError):list(d.records())
            self.assertEqual(d.status,'FAILED');self.assertTrue(d.report()['child_exited'])

    def test_aborted_consumer_terminates_only_owned_process(self):
        d=self.decoder();fake=self.fake_process(b'<pdml>'+xml()+xml(2)+b'</pdml>')
        with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=fake),patch('nids_ml.pcap.decoder.process_rss',return_value=1024):
            stream=d.records();next(stream);stream.close()
        self.assertEqual(d.status,'ABORTED');self.assertTrue(fake.terminated)

    def test_memory_abort_terminates_owned_child(self):
        d=self.decoder();fake=self.fake_process(b'<pdml/>')
        with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=fake),patch('nids_ml.pcap.decoder.process_rss',return_value=2**31):
            with self.assertRaisesRegex(DecoderError,'RSS_ABORT_THRESHOLD'):list(d.records())
        self.assertEqual(d.status,'FAILED');self.assertTrue(fake.terminated)

    def test_timeout_and_stderr_cap(self):
        d=self.decoder(limits=DecodeLimits(timeout_seconds=.001));fake=self.fake_process(b'<pdml/>')
        with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=fake),patch('nids_ml.pcap.decoder.process_rss',return_value=1024),patch('nids_ml.pcap.decoder.time.monotonic',side_effect=[0,1,1]):
            with self.assertRaisesRegex(DecoderError,'TIMEOUT'):list(d.records())
        self.assertTrue(fake.terminated)
        d=self.decoder(limits=DecodeLimits(stderr_bytes=128));fake=self.fake_process(b'<pdml/>')
        fake.stderr=BytesIO(b'x'*5000)
        with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=fake),patch('nids_ml.pcap.decoder.process_rss',return_value=1024):
            with self.assertRaisesRegex(DecoderError,'STDERR_BYTE_LIMIT'):list(d.records())
        self.assertLessEqual(len(d.stderr_tail),128)

    def test_child_wait_timeout_reports_structured_error_and_closes(self):
        d=self.decoder();fake=self.fake_process(b'<pdml>'+xml()+b'</pdml>')
        calls=[]
        def wait(timeout=None):
            calls.append(timeout)
            if len(calls)==1:raise subprocess.TimeoutExpired('owned-fake-child',timeout)
            return fake.returncode
        fake.wait=wait
        with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=fake),patch('nids_ml.pcap.decoder.process_rss',return_value=1024):
            with self.assertRaisesRegex(DecoderError,'DECODER_DRAIN_TIMEOUT'):list(d.records())
        self.assertEqual(d.status,'FAILED');self.assertTrue(fake.terminated)
        d=self.decoder();fake=self.fake_process(b'<pdml>'+xml()+b'</pdml>')
        def failed_wait(timeout=None):raise subprocess.TimeoutExpired('owned-fake-child',timeout)
        fake.wait=failed_wait
        with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=fake),patch('nids_ml.pcap.decoder.process_rss',return_value=1024):
            with self.assertRaisesRegex(DecoderError,'CHILD_CLOSE_FAILED'):list(d.records())
        self.assertEqual(d.status,'FAILED')


if __name__=='__main__':unittest.main()
