"""Test ordered-flow numerics, lineage, limits and failure handling.
Test số học ordered-flow, lineage, giới hạn và xử lý lỗi.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import tracemalloc
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from flow_extraction_fixtures import CAPTURE_TEXT, context, packet
from nids_ml.cli.ordered_flow import main
from nids_ml.flows.extractor import FlowExtractor, FlowTensor, ResourceLimits
from nids_ml.flows.features import CaptureContext, ExtractionError, ResourceLimitError, configuration as old_config
from nids_ml.flows.ordered import OrderedFlowExtractor, OrderedFlowTensor, OrderedPacketReceipt, OrderedExtractionComplete
from nids_ml.flows.ordered_contracts import (CONFIG_SCHEMA,METADATA_SCHEMA,configuration,validate_configuration,
                                           validate_metadata,validate_tensor_bytes)
from nids_ml.labeling.contracts import canonical_bytes
from nids_ml.ordering.contracts import OrderingError,make_source_manifest,source_order_digest
from nids_ml.ordering.lookup import SyntheticPacketLookup
from nids_ml.ordering.view import build_processing_view,iter_processing_view


def source(packets,first=1):
    return make_source_manifest(CaptureContext.from_dict(context()),synthetic_capture_text=CAPTURE_TEXT,
        expected_packets=len(packets),source_first_ordinal=first,expected_source_order_sha256=source_order_digest(packets))


class OrderedFixture(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='ordered-flow-synthetic-',dir=ROOT/'artifacts/runs')
        self.base=Path(self.temp.name);self.next_view=0

    def tearDown(self):self.temp.cleanup()

    def prepare(self,packets,profile='flow-cnn-n20-t10',limits=None):
        self.next_view+=1;path=self.base/('view-'+str(self.next_view))
        src=source(packets,first=packets[0]['ordinal'])
        view=build_processing_view(iter(packets),src,path)
        refs=list(iter_processing_view(path))
        lookup=SyntheticPacketLookup(packets,src)
        engine=OrderedFlowExtractor(view,configuration(profile),limits)
        return engine,refs,lookup,view,path

    def run_case(self,packets,profile='flow-cnn-n20-t10',limits=None):
        engine,refs,lookup,view,path=self.prepare(packets,profile,limits)
        events=list(engine.run(iter(refs),lookup,input_complete=True))
        return engine,[e for e in events if isinstance(e,OrderedFlowTensor)],events,view


class NumericAndLineageTests(OrderedFixture):
    def test_reordered_source_ordinals_and_separate_processing_indices(self):
        packets=[packet(ordinal=1,timestamp_ns=2),packet(ordinal=2,timestamp_ns=1)]
        before=canonical_bytes(packets)
        engine,flows,events,view=self.run_case(packets)
        meta=flows[0].metadata
        self.assertEqual(meta['retained_ordinals'],[2,1])
        self.assertEqual(meta['retained_processing_indices'],[1,2])
        self.assertEqual((meta['first_ordinal'],meta['last_ordinal']),(2,1))
        self.assertEqual((meta['min_source_ordinal'],meta['max_source_ordinal']),(1,2))
        self.assertEqual(meta['processing_view_sha256'],view['processing_order_sha256'])
        self.assertEqual(canonical_bytes(packets),before)
        self.assertEqual(engine.status,'COMPLETE')
        self.assertIsInstance(events[-1],OrderedExtractionComplete)

    def test_monotonic_input_matches_existing_tensor_bytes_for_both_profiles(self):
        packets=[packet(ordinal=i,timestamp_ns=i,reverse=i%2==0) for i in range(1,26)]
        packets.extend([packet(protocol=17,ordinal=26,timestamp_ns=10000000000),
                        packet(protocol=1,ordinal=27,timestamp_ns=10000000001)])
        for profile in ('flow-cnn-n20-t10','paper-table3-n10'):
            old=FlowExtractor(CaptureContext.from_dict(context()),old_config(profile))
            original=[e for e in old.run(packets,input_complete=True) if isinstance(e,FlowTensor)]
            _,ordered,_,_=self.run_case(packets,profile)
            self.assertEqual([(t.shape,t.tensor_bytes) for t in original],[(t.shape,t.tensor_bytes) for t in ordered])
            self.assertEqual([t.metadata['full_packet_digest'] for t in original],
                             [t.metadata['full_packet_digest'] for t in ordered])

    def test_exact_ten_seconds_in_same_window_and_plus_one_ns_starts_next(self):
        packets=[packet(ordinal=1,timestamp_ns=10000000001),packet(ordinal=2,timestamp_ns=0),
                 packet(ordinal=3,timestamp_ns=10000000000)]
        for profile in ('flow-cnn-n20-t10','paper-table3-n10'):
            _,flows,_,_=self.run_case(packets,profile)
            self.assertEqual([t.metadata['retained_ordinals'] for t in flows],[[2,3],[1]])
            self.assertEqual([t.metadata['retained_processing_indices'] for t in flows],[[1,2],[3]])
            self.assertEqual(struct.unpack_from('<f',flows[0].tensor_bytes,44)[0],1.)

    def test_global_anchor_not_per_flow_after_ordering(self):
        packets=[packet(ordinal=1,timestamp_ns=11000000000),packet(ordinal=2,timestamp_ns=0),
                 packet(ordinal=3,timestamp_ns=9000000000,flow=1),packet(ordinal=4,timestamp_ns=11000000000,flow=1)]
        _,flows,_,_=self.run_case(packets)
        self.assertEqual([t.metadata['window_index'] for t in flows],[0,0,1,1])
        self.assertEqual([t.metadata['anchor_timestamp_ns'] for t in flows],[0,0,11000000000,11000000000])
        self.assertEqual([t.metadata['retained_ordinals'] for t in flows],[[2],[3],[1],[4]])

    def test_ties_preserve_source_order_and_source_first_ordinal(self):
        packets=[packet(ordinal=101,timestamp_ns=2),packet(ordinal=102,timestamp_ns=0),
                 packet(ordinal=103,timestamp_ns=0)]
        _,flows,_,_=self.run_case(packets)
        self.assertEqual(flows[0].metadata['retained_ordinals'],[102,103,101])
        self.assertEqual(flows[0].metadata['retained_processing_indices'],[1,2,3])

    def test_reverse_packets_share_flow_and_directions_follow_first_observed(self):
        packets=[packet(ordinal=1,timestamp_ns=2),packet(ordinal=2,timestamp_ns=0,reverse=True),
                 packet(ordinal=3,timestamp_ns=1,reverse=True)]
        _,flows,_,_=self.run_case(packets)
        self.assertEqual(len(flows),1)
        self.assertEqual(flows[0].metadata['forward_packet_count'],2)
        self.assertEqual(flows[0].metadata['reverse_packet_count'],1)
        self.assertEqual(flows[0].metadata['retained_ordinals'],[2,3,1])

    def test_overflow_covers_all_packets_digests_and_receipts_n10_n20(self):
        packets=[packet(ordinal=i,timestamp_ns=26-i) for i in range(1,26)]
        for profile,n in [('flow-cnn-n20-t10',20),('paper-table3-n10',10)]:
            engine,refs,lookup,view,_=self.prepare(packets,profile)
            events=list(engine.run(refs,lookup,input_complete=True))
            tensor=next(e for e in events if isinstance(e,OrderedFlowTensor));meta=tensor.metadata
            receipts=[e for e in events if isinstance(e,OrderedPacketReceipt)]
            self.assertEqual(len(receipts),25)
            self.assertEqual(meta['real_packet_count'],25)
            self.assertEqual(meta['overflow_packet_count'],25-n)
            self.assertEqual(meta['retained_ordinals'],list(range(25,25-n,-1)))
            self.assertEqual(meta['retained_processing_indices'],list(range(1,n+1)))
            self.assertEqual(meta['full_packet_digest'],hashlib.sha256(b''.join(bytes.fromhex(r.source_packet_sha256) for r in refs)).hexdigest())
            self.assertEqual(meta['overflow_packet_digest'],hashlib.sha256(b''.join(bytes.fromhex(r.source_packet_sha256) for r in refs[n:])).hexdigest())
            self.assertEqual(meta['full_processing_digest'],hashlib.sha256(b''.join(canonical_bytes(r.to_dict())+b'\n' for r in refs)).hexdigest())
            self.assertEqual(meta['overflow_processing_digest'],hashlib.sha256(b''.join(canonical_bytes(r.to_dict())+b'\n' for r in refs[n:])).hexdigest())
            self.assertEqual(sum(r.receipt.retained for r in receipts),n)
            self.assertEqual(meta['full_window_label'],'UNKNOWN')
            self.assertFalse(meta['supervised_eligible'])

    def test_padding_and_eleven_feature_golden_values(self):
        for profile,n in [('flow-cnn-n20-t10',20),('paper-table3-n10',10)]:
            _,flows,_,_=self.run_case([packet()],profile)
            tensor=flows[0]
            self.assertEqual(tensor.shape,(n,11,1))
            self.assertEqual(tensor.tensor_bytes[44:],bytes((n-1)*44))
            self.assertEqual(struct.unpack_from('<11f',tensor.tensor_bytes),struct.unpack('<11f',
                struct.pack('<11f',0,60/65536,78354535/4294967296,2/65536,576/4096,
                            20/65536,123/4294967296,18/65536,4096/65536,0,0)))

    def test_udp_and_icmp_features_and_icmp_identifier_changes(self):
        packets=[packet(protocol=17,ordinal=1),packet(protocol=1,ordinal=2,timestamp_ns=1),
                 packet(protocol=1,ordinal=3,timestamp_ns=2)]
        packets[2]['icmp']['identifier']=8
        _,flows,_,_=self.run_case(packets)
        self.assertEqual([t.metadata['flow_tuple'][4] for t in flows],[17,1])
        self.assertEqual(struct.unpack_from('<f',flows[0].tensor_bytes,9*4)[0],32/65536)
        self.assertEqual(struct.unpack_from('<f',flows[1].tensor_bytes,10*4)[0],8/256)
        self.assertTrue(flows[1].metadata['icmp_identifier_changed'])

    def test_identity_does_not_become_cnn_channel(self):
        a=[packet(ordinal=1,timestamp_ns=1),packet(ordinal=2,timestamp_ns=0)]
        b=deepcopy(a)
        for p in b:p['src_ip']='198.51.100.3';p['tcp']['src_port']=1234
        _,fa,_,_=self.run_case(a);_,fb,_,_=self.run_case(b)
        self.assertEqual(fa[0].tensor_bytes,fb[0].tensor_bytes)
        self.assertNotEqual(fa[0].metadata['full_packet_digest'],fb[0].metadata['full_packet_digest'])

    def test_receipt_version_and_original_identity(self):
        _,_,events,view=self.run_case([packet(ordinal=1,timestamp_ns=2),packet(ordinal=2,timestamp_ns=0)])
        receipt=next(e for e in events if isinstance(e,OrderedPacketReceipt)).to_dict()
        self.assertEqual(receipt['schema_version'],'2.0.0')
        self.assertEqual(receipt['source_packet_ordinal'],2)
        self.assertEqual(receipt['processing_index'],1)
        self.assertNotIn('ordinal',receipt)
        self.assertEqual(receipt['label_status'],'UNKNOWN')
        self.assertEqual(receipt['processing_view_sha256'],view['processing_order_sha256'])

    def test_deterministic_complete_event_and_output(self):
        packets=[packet(ordinal=1,timestamp_ns=2),packet(ordinal=2,timestamp_ns=1)]
        a=self.run_case(packets)[2];b=self.run_case(packets)[2]
        self.assertEqual([e.to_dict() for e in a],[e.to_dict() for e in b])


class IntegrityFailureTests(OrderedFixture):
    def failed(self,engine,refs,lookup):
        events=[]
        with self.assertRaises((ValueError,KeyError,RuntimeError)):
            for e in engine.run(refs,lookup,input_complete=True):events.append(e)
        self.assertEqual(engine.status,'FAILED')
        self.assertFalse(any(isinstance(e,OrderedExtractionComplete) for e in events))
        self.assertEqual(engine.stats['active_flows'],0)
        self.assertEqual(engine.stats['lineage_buffer_bytes'],0)
        return events

    def test_hash_or_metadata_changes_rejected(self):
        for key,value in [('timestamp_ns',7),('ordinal',99),('src_ip','198.51.100.1')]:
            engine,refs,lookup,_,_=self.prepare([packet()])
            def changed(ordinal):
                p=lookup(ordinal);p[key]=value;return p
            self.failed(engine,refs,changed)

    def test_capture_run_segment_mismatch_in_lookup_rejected(self):
        for key in ('capture_id','capture_sha256','run_id','segment_id'):
            engine,refs,lookup,_,_=self.prepare([packet()])
            def changed(ordinal):
                p=lookup(ordinal);p['capture'][key]='a'*64 if key=='capture_sha256' else 'other';return p
            self.failed(engine,refs,changed)

    def test_reference_hash_context_or_index_tamper(self):
        for field,value in [('processing_index',True),('processing_index',2),('source_packet_sha256','a'*64),
                            ('source_packet_ordinal',99),('timestamp_ns',True),('capture',CaptureContext('other','a'*64,'r','s'))]:
            engine,refs,lookup,_,_=self.prepare([packet()])
            self.failed(engine,[replace(refs[0],**{field:value})],lookup)

    def test_duplicate_missing_and_extra_references(self):
        packets=[packet(),packet(ordinal=2,timestamp_ns=1)]
        for mode in ('duplicate','missing','extra'):
            engine,refs,lookup,_,_=self.prepare(packets)
            bad=[refs[0],refs[0]] if mode=='duplicate' else refs[:1] if mode=='missing' else refs+[refs[1]]
            self.failed(engine,bad,lookup)

    def test_processing_clock_or_tie_regression_rejected(self):
        for timestamps in ([0,1],[0,0]):
            engine,refs,lookup,_,_=self.prepare([packet(ordinal=i,timestamp_ns=t) for i,t in enumerate(timestamps,1)])
            self.failed(engine,[replace(refs[1],processing_index=1),replace(refs[0],processing_index=2)],lookup)

    def test_view_digest_mismatch_prevents_completion(self):
        engine,refs,lookup,view,_=self.prepare([packet()])
        wrong=deepcopy(view);wrong['processing_order_sha256']='a'*64
        self.failed(OrderedFlowExtractor(wrong),refs,lookup)

    def test_lookup_missing_packet_or_exception(self):
        for callback in (lambda ordinal:None,Mock(side_effect=RuntimeError('synthetic lookup error'))):
            engine,refs,_,_,_=self.prepare([packet()])
            self.failed(engine,refs,callback)

    def test_reference_generator_failure_prevents_completion(self):
        engine,refs,lookup,_,_=self.prepare([packet(),packet(ordinal=2,timestamp_ns=1)])
        def broken():
            yield refs[0]
            raise RuntimeError('synthetic reference stream error')
        self.failed(engine,broken(),lookup)

    def test_label_injection_rejected_not_used_for_training(self):
        engine,refs,lookup,_,_=self.prepare([packet()])
        def changed(ordinal):
            p=lookup(ordinal);p['label']=1;return p
        self.failed(engine,refs,changed)

    def test_fragmented_packet_fails_no_silent_skip(self):
        p=packet();p['fragmented']=True
        engine,refs,lookup,_,_=self.prepare([p])
        self.failed(engine,refs,lookup)

    def test_error_after_first_window_keeps_earlier_output_provisional(self):
        packets=[packet(ordinal=i,timestamp_ns=t) for i,t in enumerate([0,11000000000,22000000000],1)]
        engine,refs,lookup,_,_=self.prepare(packets)
        def changed(ordinal):
            p=lookup(ordinal)
            if ordinal==3:p['tcp']['length']=19
            return p
        events=self.failed(engine,refs,changed)
        emitted=[e for e in events if isinstance(e,OrderedFlowTensor)]
        self.assertEqual(len(emitted),1)
        self.assertTrue(emitted[0].metadata['stream_completion_required'])

    def test_partial_declaration_rejected_before_lookup(self):
        for complete in (False,1,None):
            engine,refs,_,_,_=self.prepare([packet()])
            call=Mock(side_effect=AssertionError('lookup must not be called'))
            with self.assertRaises(ExtractionError):list(engine.run(refs,call,input_complete=complete))
            call.assert_not_called()

    def test_replay_rejected_and_aborted_generator_closes_reference_stream(self):
        engine,refs,lookup,_,_=self.prepare([packet()])
        list(engine.run(refs,lookup,input_complete=True))
        with self.assertRaises(ExtractionError):list(engine.run(refs,lookup,input_complete=True))
        engine,refs,lookup,_,_=self.prepare([packet(),packet(ordinal=2,timestamp_ns=1)])
        closed=[]
        def stream():
            try:yield from refs
            finally:closed.append(True)
        events=engine.run(stream(),lookup,input_complete=True);next(events);events.close()
        self.assertEqual(engine.status,'ABORTED');self.assertEqual(closed,[True])
        self.assertEqual(engine.stats['lineage_buffer_bytes'],0)


class ContractAndResourceTests(OrderedFixture):
    memory_observation=None

    def test_new_configs_and_schemas_match_runtime(self):
        for profile,name in [('flow-cnn-n20-t10','ordered-flow-n20-t10.json'),('paper-table3-n10','ordered-flow-n10-t10.json')]:
            self.assertEqual(validate_configuration(json.loads((ROOT/'configs'/name).read_text())),configuration(profile))
        for name,value in [('ordered-flow-configuration.schema.json',CONFIG_SCHEMA),('ordered-flow-metadata.schema.json',METADATA_SCHEMA)]:
            self.assertEqual(json.loads((ROOT/'contracts'/name).read_text()),value)

    def test_old_capture_core_rejects_new_config_and_keeps_timestamp_guard(self):
        with self.assertRaises(ExtractionError):FlowExtractor(CaptureContext.from_dict(context()),configuration())
        old=FlowExtractor(CaptureContext.from_dict(context()),old_config())
        with self.assertRaisesRegex(ExtractionError,'Timestamp regression'):
            list(old.run([packet(ordinal=1,timestamp_ns=2),packet(ordinal=2,timestamp_ns=1)],input_complete=True))

    def test_old_hdf5_validator_rejects_ordered_type_before_any_io(self):
        from nids_ml.storage.hdf5.validation import validate_flow
        from nids_ml.storage.hdf5.schema import ExportError
        tensor=self.run_case([packet()])[1][0]
        self.assertNotIsInstance(tensor,FlowTensor)
        with self.assertRaisesRegex(ExportError,'FLOW_TENSOR_REQUIRED'):validate_flow(tensor,{})

    def test_profile_policy_or_feature_changes_rejected(self):
        for key,value in [('label_policy','AUTO_LABEL'),('packet_order','CAPTURE_ORDINAL'),('window_ns',1),('feature_order',[])]:
            config=configuration();config[key]=value
            with self.assertRaises(ExtractionError):validate_configuration(config)

    def test_tensor_and_sidecar_validation_reject_tamper(self):
        engine,flows,_,view=self.run_case([packet(),packet(ordinal=2,timestamp_ns=1)])
        tensor=flows[0]
        for field,value in [('retained_processing_indices',[2,1]),('retained_ordinals',[1,1]),
                            ('processing_view_sha256','a'*64),('full_window_label','VERIFIED_DDOS')]:
            meta=deepcopy(tensor.metadata);meta[field]=value
            with self.assertRaises(ExtractionError):validate_metadata(meta,engine.ordered_config,view)
        corrupt=bytearray(tensor.tensor_bytes);struct.pack_into('<f',corrupt,0,float('nan'))
        with self.assertRaises(ExtractionError):validate_tensor_bytes(tensor.shape,bytes(corrupt),tensor.metadata,engine.ordered_config,view)
        with self.assertRaises(ExtractionError):validate_tensor_bytes((20,11,True),tensor.tensor_bytes,tensor.metadata,engine.ordered_config,view)

    def test_active_flow_window_and_accounted_memory_limits(self):
        for limits,packets in [(ResourceLimits(max_active_flows=1),[packet(),packet(ordinal=2,timestamp_ns=1,flow=1)]),
                               (ResourceLimits(max_packets_per_window=1),[packet(),packet(ordinal=2,timestamp_ns=1)]),
                               (ResourceLimits(memory_budget_bytes=16384),[packet()])]:
            engine,refs,lookup,_,_=self.prepare(packets,limits=limits)
            with self.assertRaises(ResourceLimitError):list(engine.run(refs,lookup,input_complete=True))
            self.assertEqual(engine.status,'FAILED')

    def test_view_count_exceeding_input_limit_rejected_at_construction(self):
        _,_,_,view,_=self.prepare([packet(),packet(ordinal=2)])
        with self.assertRaises(ResourceLimitError):OrderedFlowExtractor(view,limits=ResourceLimits(max_input_packets=1))

    def test_lookup_is_snapshot_not_mutable_reference(self):
        packets=[packet()];lookup=SyntheticPacketLookup(packets,source(packets))
        original=lookup(1);packets[0]['tcp']['length']=19
        borrowed=lookup(1);borrowed['tcp']['length']=18
        self.assertEqual(lookup(1),original)

    def test_lookup_rejects_count_digest_context_and_budget(self):
        packets=[packet(),packet(ordinal=2)];src=source(packets)
        with self.assertRaises(OrderingError):SyntheticPacketLookup(packets[:1],src)
        src=deepcopy(src);src['expected_source_order_sha256']='a'*64
        with self.assertRaises(OrderingError):SyntheticPacketLookup(packets,src)
        with self.assertRaises(OrderingError):SyntheticPacketLookup([],source([packet()]))
        lookup=SyntheticPacketLookup([packet()],source([packet()]))
        with self.assertRaises(OrderingError):lookup(True)

    def test_streaming_resource_observation_without_full_event_list(self):
        count=1024
        def packets():
            for i in range(1,count+1):yield packet(ordinal=i,timestamp_ns=count-i)
        src=make_source_manifest(CaptureContext.from_dict(context()),synthetic_capture_text=CAPTURE_TEXT,
            expected_packets=count,expected_source_order_sha256=source_order_digest(packets()))
        path=self.base/'streaming-view';view=build_processing_view(packets(),src,path)
        engine=OrderedFlowExtractor(view)
        tracemalloc.start()
        try:
            events=0
            for event in engine.run(iter_processing_view(path),lambda i:packet(ordinal=i,timestamp_ns=count-i),input_complete=True):
                events+=1
            _,peak=tracemalloc.get_traced_memory()
        finally:tracemalloc.stop()
        self.assertEqual(events,count+2);self.assertEqual(engine.status,'COMPLETE')
        self.assertEqual(engine.stats['peak_lineage_buffer_bytes'],20*8)
        self.assertLess(peak,4*1048576)
        type(self).memory_observation=dict(synthetic_packets=count,python_tracemalloc_peak_bytes=peak,
            active_flow_accounted_peak_bytes=engine.stats['peak_accounted_state_bytes'],
            retained_feature_peak_bytes=engine.stats['peak_retained_buffer_bytes'],
            retained_lineage_peak_bytes=engine.stats['peak_lineage_buffer_bytes'],total_RSS_measured=False,native=False)


class CLITests(OrderedFixture):
    def test_actual_cli_entry_in_process_both_profiles(self):
        cfg=json.loads((ROOT/'configs/packet-ordering-synthetic.example.json').read_text())
        path=self.base/'cli-view';build_processing_view(cfg['packets'],cfg['source'],path)
        for profile in ('flow-cnn-n20-t10','paper-table3-n10'):
            with patch('sys.stdout',io.StringIO()) as output:
                result=main(['--view',str(path),'--fixture',str(ROOT/'configs/packet-ordering-synthetic.example.json'),'--profile',profile])
            self.assertEqual(result,0,output.getvalue())
            value=json.loads(output.getvalue())
            self.assertEqual(value['status'],'PASS_ORDERED_SYNTHETIC_ONLY')
            self.assertEqual(value['events'][-1]['kind'],'ordered_stream_complete')
            self.assertFalse(value['training'])

    def test_invalid_raw_path_or_view_rejected(self):
        with patch('sys.stdout',io.StringIO()):
            self.assertEqual(main(['--view',str(ROOT/'data/raw/not-read'),'--fixture','missing']),2)

    def test_help_in_windows_cp1252(self):
        stream=io.TextIOWrapper(io.BytesIO(),encoding='cp1252',write_through=True)
        with patch('sys.stdout',stream),self.assertRaises(SystemExit) as result:main(['--help'])
        self.assertEqual(result.exception.code,0)


if __name__=='__main__':unittest.main()
