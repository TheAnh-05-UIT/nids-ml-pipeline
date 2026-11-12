"""Validate mapped LUCID caches/shards with controlled synthetic packet records.
Kiểm bằng fixtures nhỏ; số mẫu test không phải thống kê Edge-IIoTset thật.
"""
from collections import Counter
from contextlib import redirect_stdout, closing
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from test_demo_pipeline import FixtureDecoder, mapping_fixture
from test_pcap_decode import record
from decoded_pcap_fixtures import make_capture, pdml, FakeDecoderProcess
from nids_ml.alignment.large import KEY_PROFILE, POLICY
from nids_ml.flows.mapped import prepare_cache,iter_tensors,FlowLimits,protocol_scope
from nids_ml.storage.shards import export_shards,FlowShardReader,ShardWriter,SHARD_ROWS
from nids_ml.pcap.decoder import file_sha256,DecoderError,DecodeLimits,Field,map_record
from nids_ml.flows.features import CaptureContext,extract_features
from nids_ml.pcap.whole import LargeFlowDecoder,LargeFlowDecodeLimits,WholeCaptureLimits
from nids_ml.cli.flow_shards import main


class FlowShardTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='flow-shards-',dir=ROOT/'artifacts/runs')
        self.work=Path(self.tmp.name); (self.work/'artifacts/runs').mkdir(parents=True)
        self.disk=patch('shutil.disk_usage',return_value=SimpleNamespace(free=10*1024**3)); self.disk.start()
        self.decoder=patch('nids_ml.flows.mapped.LargeFlowDecoder',FixtureDecoder); self.decoder.start()
        self.tool=self.work/'synthetic-tool'; self.tool.write_bytes(b'FAKE NOT EXECUTABLE')

    def tearDown(self):
        self.decoder.stop(); self.disk.stop()
        assert self.work.resolve().parent==(ROOT/'artifacts/runs').resolve()
        self.tmp.cleanup()

    def refresh_mapping(self,path):
        with closing(sqlite3.connect(path/'alignment.sqlite')) as db:
            if 'kh' not in [row[1] for row in db.execute('PRAGMA table_info(p)')]:
                db.execute('ALTER TABLE p ADD COLUMN kh BLOB')
                db.execute('UPDATE p SET kh=?',(b'\x06'+bytes(22),));db.commit()
        report=json.loads((path/'coverage.json').read_text())
        report.update(index_sha256=file_sha256(path/'alignment.sqlite'),key_profile=KEY_PROFILE,
                      annotations=dict(policy=POLICY))
        for identity in report['sources'].values():identity['path']=identity['path'].replace('\\','/')
        report['mapping']['clock']['pcap_UTC_days']=1
        (path/'coverage.json').write_text(json.dumps(report))

    def mapping(self,case='normal',label=0,**options):
        path=mapping_fixture(self.work,case,label,**options); self.refresh_mapping(path); return path

    def cache(self,mapping,name='cache'):
        out=self.work/'artifacts/runs'/name
        report=prepare_cache(mapping,self.work,out,tshark=self.tool,tshark_sha256=file_sha256(self.tool))
        return out,report

    def export(self,caches,name='dataset'):
        out=self.work/'artifacts/runs'/name
        report=export_shards(caches,self.work,out)
        return out,report

    def refresh_shard(self,out):
        manifest=json.loads((out/'manifest.json').read_text()); entry=manifest['shards'][0]
        path=out/entry['filename']; entry.update(bytes=path.stat().st_size,sha256=file_sha256(path))
        (out/'manifest.json').write_text(json.dumps(manifest))

    def test_full_roundtrip_and_real_window_metadata(self):
        normal,n=self.cache(self.mapping())
        attack,a=self.cache(self.mapping('attack',1),'attack-cache')
        out,report=self.export([attack,normal])
        self.assertEqual(report['samples'],48)
        self.assertEqual(report['label_counts'],{'0':24,'1':24})
        self.assertFalse(report['training_ready']); self.assertEqual(report['split'],'UNASSIGNED')
        rows=[]
        for batch in FlowShardReader(out).batches(7,metadata=True):
            self.assertLessEqual(len(batch['y']),7); rows.extend(batch['metadata'])
            self.assertEqual(batch['X'].shape[1:],(20,11,1))
        first=json.loads(rows[0].tobytes())
        self.assertEqual(first['core']['real_packet_count'],2)
        self.assertEqual(first['core']['retained_packet_count'],2)
        self.assertFalse(first['core']['supervised_eligible'])
        self.assertEqual(first['label_provenance'],'DATASET_PROVIDED_MAPPED')
        self.assertEqual(n['ordering'],'SOURCE_CONTIGUOUS_DECODE_THEN_STABLE_TIMESTAMP_ORDINAL_VIEW')
        self.assertEqual(n['counts']['input_packets'],48)

    def test_unknown_overflow_packet_excludes_full_window_flow(self):
        cache,_=self.cache(self.mapping(unknown_tail=True))
        stats=Counter(); items=list(iter_tensors(cache,stats))
        self.assertEqual(len(items),23); self.assertEqual(stats['excluded_unknown'],1)

    def test_mixed_overflow_is_not_majority_voting(self):
        mapping=self.mapping(unknown_tail=True)
        index=mapping/'alignment.sqlite'
        with closing(sqlite3.connect(index)) as db:
            ordinal=db.execute("SELECT pord FROM witnessed WHERE status='UNMATCHED'").fetchone()[0]
            db.execute("UPDATE witnessed SET status='UNIQUE_CONTENT_AND_TIME_CANDIDATE' WHERE pord=?",(ordinal,))
            db.execute("UPDATE c SET alabel='1',atype='Backdoor' WHERE ord=?",(ordinal,));db.commit()
        csv=mapping.parent/'data/raw/normal.csv'
        lines=csv.read_text().splitlines(); lines[ordinal]='1,Backdoor';csv.write_text('\n'.join(lines)+'\n')
        report=json.loads((mapping/'coverage.json').read_text());report['sources']['csv'].update(
            size_bytes=csv.stat().st_size,sha256=file_sha256(csv));(mapping/'coverage.json').write_text(json.dumps(report))
        self.refresh_mapping(mapping)
        cache,_=self.cache(mapping);stats=Counter();list(iter_tensors(cache,stats))
        self.assertEqual(stats['excluded_mixed'],1)

    def test_uncertain_packet_quarantines_all_flows_in_its_window(self):
        cache,_=self.cache(self.mapping(poison=True)); stats=Counter(); items=list(iter_tensors(cache,stats))
        self.assertEqual(stats['excluded_quarantine'],8);self.assertEqual(len(items),16)

    def test_cache_and_shard_output_are_deterministic_for_input_order(self):
        a,_=self.cache(self.mapping()); b,_=self.cache(self.mapping('attack',1),'b')
        x,first=self.export([a,b],'first')
        time.sleep(1.1)  # Ensure deterministic HDF5 bytes across different wall-clock seconds.
        y,second=self.export([b,a],'second')
        self.assertEqual(first['shards'],second['shards'])
        self.assertEqual(first['sources'],second['sources'])
        self.assertEqual(first['label_counts'],second['label_counts'])

    def test_icmp_features_and_identifier_are_preserved_and_udp_labels_stay_unknown(self):
        for protocol,name in ((1,'icmp'),(17,'udp')):
            mapping=self.mapping(name,1)
            source=self.work/'data/raw'/(name+'.pcap')
            original=FixtureDecoder.records_by_path[str(source)];changed=[]
            for record in original:
                fields=[replace(f,show=str(protocol) if f.name=='ip.proto' else
                                'eth:ethertype:ip:'+name if f.name=='frame.protocols' else
                                '28' if f.name=='ip.len' else f.show)
                        for f in record.fields if f.protocol!='tcp']
                values={'icmp.type':'8','icmp.code':'0','icmp.ident':'123'} if protocol==1 else {
                    'udp.srcport':'40000','udp.dstport':'8080','udp.length':'8'}
                fields += [Field(k,v,None,name) for k,v in values.items()]
                changed.append(replace(record,layers=('geninfo','frame','eth','ip',name),
                    all_protocols=('geninfo','frame','eth','ip',name),fields=tuple(fields)))
            FixtureDecoder.records_by_path[str(source)]=changed
            with closing(sqlite3.connect(mapping/'alignment.sqlite')) as db:
                db.execute('UPDATE p SET kh=?',(b'\x01'+bytes(12) if protocol==1 else None,))
                if protocol==17:db.execute("UPDATE witnessed SET status='UNSUPPORTED_OR_INVALID',cord=NULL")
                db.commit()
            self.refresh_mapping(mapping)
            cache,report=self.cache(mapping,name+'-cache'); stats=Counter(); items=list(iter_tensors(cache,stats))
            self.assertEqual(report['counts']['feature_packets'],48)
            if protocol==1:
                self.assertEqual(len(items),3)
                self.assertFalse(items[0][1]['core']['icmp_identifier_changed'])
                self.assertEqual(items[0][1]['core']['flow_tuple'][-1],1)
                self.assertEqual(items[0][1]['core']['retained_packet_count'],16)
            else:
                self.assertEqual(len(items),0);self.assertEqual(stats['excluded_unknown'],3)

    def test_protocol_mismatch_rejects_cross_protocol_annotation(self):
        mapping=self.mapping()
        with closing(sqlite3.connect(mapping/'alignment.sqlite')) as db:
            db.execute('UPDATE p SET kh=?',(b'\x01'+bytes(12),));db.commit()
        self.refresh_mapping(mapping)
        with self.assertRaisesRegex(ValueError,'MAPPING_PROTOCOL_IDENTITY'):self.cache(mapping)

    def test_packet_cache_tamper_is_rejected(self):
        cache,_=self.cache(self.mapping())
        with (cache/'packets.sqlite').open('ab') as stream:stream.write(b'tampered')
        with self.assertRaisesRegex(ValueError,'CACHE_SHA256'):list(iter_tensors(cache,Counter()))

    def test_processing_digest_indices_and_windows_checked_even_with_updated_index_hash(self):
        for name,sql in (('indices','UPDATE packets SET processing=0 WHERE ord=1'),
                         ('window','UPDATE windows SET anchor=anchor+1 WHERE id=0'),
                         ('poison','UPDATE windows SET poison=1 WHERE id=0')):
            mapping=self.mapping(name);cache,_=self.cache(mapping,name+'-cache')
            with closing(sqlite3.connect(cache/'packets.sqlite')) as db:db.execute(sql);db.commit()
            m=json.loads((cache/'manifest.json').read_text());m['index']['sha256']=file_sha256(cache/'packets.sqlite')
            (cache/'manifest.json').write_text(json.dumps(m))
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'INTEGRITY'):
                list(iter_tensors(cache,Counter()))

    def test_mapping_sha_and_original_csv_labels_are_checked(self):
        mapping=self.mapping();index=mapping/'alignment.sqlite'
        with closing(sqlite3.connect(index)) as db:
            db.execute("UPDATE c SET atype='Backdoor' WHERE ord=1");db.commit()
        with self.assertRaisesRegex(ValueError,'INDEX_SHA256'):self.cache(mapping,'bad-sha')
        self.refresh_mapping(mapping)
        with self.assertRaisesRegex(ValueError,'LABEL_IMPORT_MISMATCH'):self.cache(mapping,'bad-label')

    def test_mapping_ordinal_timestamp_and_one_to_one_fail_closed(self):
        for mode in ('timestamp','duplicate','missing'):
            with self.subTest(mode=mode):
                mapping=self.mapping(mode)
                with closing(sqlite3.connect(mapping/'alignment.sqlite')) as db:
                    if mode=='timestamp':db.execute('UPDATE p SET ts=ts+1 WHERE ord=1')
                    elif mode=='duplicate':db.execute('UPDATE witnessed SET cord=1 WHERE pord=2')
                    else:db.execute('DELETE FROM witnessed WHERE pord=1')
                    db.commit()
                self.refresh_mapping(mapping)
                with self.assertRaises((ValueError,sqlite3.IntegrityError)):
                    self.cache(mapping,mode+'-output')
                out=self.work/'artifacts/runs'/(mode+'-output')
                self.assertTrue((out/'FAILED.json').exists());self.assertFalse((out/'manifest.json').exists())

    def test_no_overwrite_no_low_disk_and_no_incomplete_source_success(self):
        mapping=self.mapping();cache,_=self.cache(mapping)
        with self.assertRaises(ValueError):self.cache(mapping)
        with patch('shutil.disk_usage',return_value=SimpleNamespace(free=1024)):
            with self.assertRaisesRegex(ValueError,'3_GIB'):self.cache(mapping,'low-disk')
        with patch.object(FixtureDecoder,'records',lambda self:iter([])):
            with self.assertRaisesRegex(ValueError,'FULL_CAPTURE'):self.cache(mapping,'incomplete')

    def test_duplicate_captures_and_failed_caches_rejected(self):
        cache,_=self.cache(self.mapping())
        with self.assertRaises(ValueError):self.export([cache,cache])
        (cache/'FAILED.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'FAILED_ARTIFACT'):self.export([cache])

    def test_writer_and_reader_cross_old_4000_sample_cap_with_bounded_slices(self):
        import h5py
        cache,_=self.cache(self.mapping());tensor,meta=next(iter_tensors(cache,Counter()))
        out=self.work/'artifacts/runs/many';out.mkdir();writer=ShardWriter(out)
        for _ in range(SHARD_ROWS+17):writer.add(tensor,meta)
        writer.flush();writer.seal()
        report=dict(format='nids-mapped-flow-shards/1.0.0',status='COMPLETE_MAPPED_FLOW_SHARDS',
            samples=writer.count,shape=[writer.count,20,11,1],shards=writer.shards,
            label_provenance='DATASET_PROVIDED_MAPPED',independent_ground_truth_verified=False,
            split='UNASSIGNED',training_ready=False)
        (out/'manifest.json').write_text(json.dumps(report))
        self.assertEqual([s['samples'] for s in writer.shards],[4096,17])
        original=h5py.Dataset.__getitem__;seen=[]
        def bounded(item,key):
            self.assertIsInstance(key,slice);self.assertLessEqual(key.stop-key.start,63)
            seen.append(key.stop-key.start);return original(item,key)
        reader=FlowShardReader(out)
        with patch.object(h5py.Dataset,'__getitem__',bounded):
            count=sum(len(batch['y']) for batch in reader.batches(63,metadata=True))
        self.assertEqual(count,4113);self.assertTrue(seen)

    def test_shard_hash_traversal_external_links_and_shape_rejected(self):
        import h5py
        cache,_=self.cache(self.mapping())
        for scenario in ('hash','path','external','shape'):
            out,_=self.export([cache],scenario)
            m=json.loads((out/'manifest.json').read_text());p=out/m['shards'][0]['filename']
            if scenario=='hash':
                with p.open('ab') as f:f.write(b'x')
            elif scenario=='path':
                m['shards'][0]['filename']='../flows-00000.h5';(out/'manifest.json').write_text(json.dumps(m))
            else:
                with h5py.File(p,'r+') as h:
                    del h['X']
                    if scenario=='external':h['X']=h5py.ExternalLink('other.h5','X')
                    else:h.create_dataset('X',shape=(24,10,11,1),dtype='<f4')
                self.refresh_shard(out)
            with self.subTest(scenario=scenario),self.assertRaises(ValueError):FlowShardReader(out)

    def test_batch_tampering_bad_labels_metadata_and_limits(self):
        import h5py
        cache,_=self.cache(self.mapping());out,_=self.export([cache])
        for size in (True,0,257):
            with self.assertRaises(ValueError):list(FlowShardReader(out).batches(size))
        for name,value in [('X',float('nan')),('y',2),('metadata',b'{}')]:
            target,_=self.export([cache],name)
            with h5py.File(target/'flows-00000.h5','r+') as h:
                if name=='X':h['X'][0,0,0,0]=value
                else:h[name][0]=value
            self.refresh_shard(target)
            with self.assertRaises((ValueError,KeyError)):
                list(FlowShardReader(target).batches(metadata=True))

    def test_cli_help_and_inspect_no_docker_or_training(self):
        cache,_=self.cache(self.mapping());out,_=self.export([cache])
        with redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(main(['inspect','--dataset',str(out),'--batch-size','5']),0)
        self.assertEqual(json.loads(stream.getvalue())['status'],'PASS_SHARD_INSPECTION')
        with redirect_stdout(io.StringIO()):self.assertEqual(main(['inspect','--dataset',str(out),'--batch-size','0']),2)

    def test_inspect_checks_actual_class_and_family_counts(self):
        cache,_=self.cache(self.mapping());out,_=self.export([cache])
        m=json.loads((out/'manifest.json').read_text());m['label_counts']={'1':m['samples']}
        (out/'manifest.json').write_text(json.dumps(m))
        with redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(main(['inspect','--dataset',str(out)]),2)
        self.assertIn('COUNTS_DISAGREE',json.loads(stream.getvalue())['error'])

    def test_all_unknown_cache_cannot_publish_an_empty_training_dataset(self):
        mapping=self.mapping()
        with closing(sqlite3.connect(mapping/'alignment.sqlite')) as db:
            db.execute("UPDATE witnessed SET status='UNMATCHED',cord=NULL");db.commit()
        self.refresh_mapping(mapping)
        cache,_=self.cache(mapping)
        with self.assertRaisesRegex(ValueError,'NO_ELIGIBLE'):self.export([cache])
        out=self.work/'artifacts/runs/dataset'
        self.assertTrue((out/'FAILED.json').exists());self.assertFalse((out/'manifest.json').exists())

    def test_early_batch_close_releases_hdf5_handle(self):
        import h5py
        cache,_=self.cache(self.mapping());out,_=self.export([cache]);reader=FlowShardReader(out)
        original=h5py.File;handles=[]
        def opened(*args,**kwargs):
            handle=original(*args,**kwargs);handles.append(handle);return handle
        with patch.object(h5py,'File',opened):
            iterator=reader.batches(1);next(iterator);self.assertTrue(handles[-1].id.valid)
            iterator.close();self.assertFalse(handles[-1].id.valid)


class LargeFlowDecoderTests(unittest.TestCase):
    def test_validated_feature_fast_path_matches_legacy_packet_conversion(self):
        context=CaptureContext('fixture','a'*64,'run','segment')
        packet,sidecar=map_record(record(),context)
        feature,fast_sidecar=map_record(record(),context,return_feature=True)
        self.assertEqual(feature,extract_features(packet,input_kind='decoded_pcap'))
        self.assertEqual(sidecar,fast_sidecar)
        with self.assertRaises(DecoderError):map_record(record(),context,return_feature='yes')

    def test_explicit_profile_keeps_legacy_limits_and_continuous_preferences(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/runs') as temp:
            fixture=make_capture(Path(temp))
            decoder=LargeFlowDecoder(fixture['source'],expected_records=4,
                expected_sha256=file_sha256(fixture['source']),expected_size=fixture['source'].stat().st_size,
                tshark=fixture['tool'],tshark_sha256=fixture['tool_sha'])
            child=FakeDecoderProcess(pdml(fixture['records']))
            with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=child),patch('nids_ml.pcap.decoder.process_rss',return_value=1024):
                self.assertEqual(len(list(decoder.records())),4)
            self.assertNotIn('-M',decoder.command);self.assertTrue(decoder.stats['source_complete'])
            self.assertEqual(decoder.command[decoder.command.index('-c')+1],'5')
        LargeFlowDecodeLimits(packet_count=3500001).validate()
        for limits in (WholeCaptureLimits(packet_count=250002),DecodeLimits(packet_count=4097),
                       replace(LargeFlowDecodeLimits(),decoder_rss_bytes=2**40),
                       LargeFlowDecodeLimits(packet_count=True)):
            with self.assertRaises(DecoderError):limits.validate()
        with self.assertRaises(ValueError):replace(FlowLimits(),sqlite_mib=99999).validate()


if __name__=='__main__':unittest.main()
