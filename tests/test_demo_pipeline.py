"""Exercise the mapped-label demo with synthetic captures, then a tiny CPU training run.
Kiểm thử bằng capture giả có kiểm soát; số liệu fixture không phải kết quả Edge-IIoTset.
"""
from dataclasses import replace
import contextlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tests'))
from test_pcap_decode import record
from nids_ml.data.demo import build_dataset,read_dataset,packet_label,split_groups,tcp_scope,PROVENANCE
from nids_ml.training.demo import metrics,train,restore_checkpoint,main
from nids_ml.pcap.decoder import file_sha256


class FixtureDecoder:
    records_by_path={}
    def __init__(self,source,**kwargs):
        self.source=Path(source);self.stats={'source_complete':False}
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):pass
    def records(self):
        yield from self.records_by_path[str(self.source)]
        self.stats['source_complete']=True
    def report(self):return dict(status='SYNTHETIC_FIXTURE',source_complete=self.stats['source_complete'])


def mapping_fixture(work,case,label,*,unknown_tail=False,duplicate=False,poison=False):
    raw=work/'data/raw';raw.mkdir(parents=True,exist_ok=True)
    source=raw/(case+'.pcap');source.write_bytes(('SYNTHETIC ONLY '+case).encode())
    csv=raw/(case+'.csv');csv.write_text('SYNTHETIC FIXTURE; NOT DATASET\n',encoding='utf-8')
    directory=work/case;directory.mkdir();db=sqlite3.connect(directory/'alignment.sqlite')
    db.execute('CREATE TABLE p(ord INTEGER PRIMARY KEY,ts INTEGER)')
    db.execute('CREATE TABLE c(ord INTEGER PRIMARY KEY,alabel TEXT,atype TEXT)')
    db.execute('CREATE TABLE witnessed(pord INTEGER PRIMARY KEY,status TEXT,cord INTEGER)')
    records=[]
    for window in (2,0,1):
        for group in range(8):
            count=22 if unknown_tail and window==2 and group==0 else 2
            for packet in range(count):
                ordinal=len(records)+1;ns=(1+11*window)*10**9+group*10000000+packet*100000
                stamp=f'{ns//10**9}.{ns%10**9:09d}'
                feature_group=0 if duplicate and group==1 else group
                values={'frame.time_epoch':stamp,'tcp.srcport':str(40000+group),
                    'tcp.ack':str(5000*label+100*feature_group+window*2+packet),
                    'tcp.ack_raw':str(8000*label+100*feature_group+window*2+packet)}
                value=record(ordinal)
                value=replace(value,fields=tuple(replace(field,show=values.get(field.name,field.show),
                    value=stamp if field.name=='timestamp' else field.value) for field in value.fields))
                if poison and group==0 and window==1 and packet==0:
                    value=replace(value,fields=tuple(field for field in value.fields if field.name!='ip.proto'))
                records.append(value)
                unknown=unknown_tail and group==0 and window==2 and packet==21
                db.execute('INSERT INTO p VALUES(?,?)',(ordinal,ns))
                db.execute('INSERT INTO c VALUES(?,?,?)',(ordinal,str(label),'Normal' if label==0 else 'DDoS_HTTP'))
                db.execute('INSERT INTO witnessed VALUES(?,?,?)',(ordinal,'UNMATCHED' if unknown else 'UNIQUE_CONTENT_AND_TIME_CANDIDATE',ordinal))
    db.commit();db.close();FixtureDecoder.records_by_path[str(source)]=records
    csv.write_text('Attack_label,Attack_type\n'+
        ''.join(f'{label},'+('Normal' if label==0 else 'DDoS_HTTP')+'\n' for _ in records),encoding='utf-8')
    report=dict(status='COMPLETE_FULL_PAIR_AUDIT',PCAP=dict(records=len(records)),CSV=dict(rows=len(records)),
        sources={name:dict(path=str(path.relative_to(work)),size_bytes=path.stat().st_size,sha256=file_sha256(path))
                 for name,path in [('pcap',source),('csv',csv)]},
        mapping=dict(clock=dict(time_witness_enabled=True,heldout_disagree=0),witnessed_max_one_to_one=True))
    (directory/'coverage.json').write_text(json.dumps(report),encoding='utf-8')
    return directory


class DemoPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='demo-test-',dir=ROOT/'artifacts/runs')
        self.work=Path(self.tmp.name);(self.work/'artifacts/runs').mkdir(parents=True)
        self.disk=patch('shutil.disk_usage',return_value=SimpleNamespace(free=10*1024**3));self.disk.start()
        self.decoder=patch('nids_ml.data.demo.DemoDecoder',FixtureDecoder);self.decoder.start()

    def tearDown(self):
        self.decoder.stop();self.disk.stop()
        assert self.work.resolve().parent==(ROOT/'artifacts/runs').resolve()
        self.tmp.cleanup()

    def prepare(self,**options):
        normal=mapping_fixture(self.work,'normal',0,**options)
        attack=mapping_fixture(self.work,'attack',1)
        output=self.work/'artifacts/runs/dataset'
        report=build_dataset(normal,attack,self.work,output,tshark='SYNTHETIC',tshark_sha256='0'*64)
        return output,report

    def refresh_hash(self,path):
        manifest=json.loads((path/'manifest.json').read_text())
        manifest['hdf5'].update(bytes=(path/'dataset.h5').stat().st_size,sha256=file_sha256(path/'dataset.h5'))
        (path/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')

    def test_label_policy_never_defaults_unknown_to_benign(self):
        self.assertEqual(packet_label('UNIQUE_CONTENT_AND_TIME_CANDIDATE','0','Normal',0),0)
        self.assertEqual(packet_label('UNIQUE_CONTENT_AND_TIME_CANDIDATE','1','DDoS_HTTP',1),1)
        for status,label,kind,expected in [('UNMATCHED','0','Normal',0),('AMBIGUOUS','1','DDoS_HTTP',1),
                ('UNIQUE_CONTENT_AND_TIME_CANDIDATE','1','Normal',1),('UNIQUE_CONTENT_AND_TIME_CANDIDATE','0','Normal',1)]:
            self.assertIsNone(packet_label(status,label,kind,expected))

    def test_group_split_is_deterministic_and_rejects_insufficient_groups(self):
        rows=[dict(group=f'{label}-{n}',y=label) for label in (0,1) for n in range(8)]
        self.assertEqual(split_groups(rows),split_groups(list(reversed(rows))))
        for label in (0,1):self.assertEqual({split_groups(rows)[r['group']] for r in rows if r['y']==label},{0,1,2})
        with self.assertRaisesRegex(ValueError,'THREE_INDEPENDENT'):split_groups(rows[:2]+rows[8:])
        with self.assertRaisesRegex(ValueError,'GROUP_LABEL_CONFLICT'):split_groups(rows+[dict(group='0-0',y=1)])

    def test_tcp_uncertainty_is_not_silently_discarded(self):
        value=record();self.assertEqual(tcp_scope(value),'TCP')
        missing=replace(value,fields=tuple(f for f in value.fields if f.name!='ip.proto'))
        self.assertEqual(tcp_scope(missing),'UNCERTAIN')
        arp=replace(value,all_protocols=('frame','eth','arp'));self.assertEqual(tcp_scope(arp),'OUTSIDE')

    def test_complete_hdf5_ordering_split_and_two_client_contract(self):
        import numpy as np
        output,report=self.prepare();manifest,x,y,splits,clients,metadata=read_dataset(output)
        self.assertEqual(report['samples'],48);self.assertEqual(x.shape,(48,20,11,1))
        self.assertEqual(report['label_provenance'],PROVENANCE)
        self.assertFalse(report['independent_ground_truth_verified'])
        for client in (0,1):self.assertEqual(set(y[(splits==0)&(clients==client)]),{0,1})
        self.assertTrue(np.isfinite(x).all());self.assertEqual(len({m['tensor_sha256'] for m in metadata}),48)
        self.assertTrue(all(m['retained_processing_indices']==sorted(m['retained_processing_indices']) for m in metadata))
        for group in {m['group'] for m in metadata}:
            self.assertEqual(len({int(splits[i]) for i,m in enumerate(metadata) if m['group']==group}),1)

    def test_unknown_overflow_packet_excludes_entire_flow_not_only_retained_twenty(self):
        output,report=self.prepare(unknown_tail=True)
        self.assertEqual(report['statistics']['flow_unknown'],1)
        self.assertEqual(report['samples'],47)
        self.assertEqual(report['sources'][0]['counts']['unknown_mapping_packets'],1)

    def test_uncertain_tcp_quarantines_whole_window(self):
        output,report=self.prepare(poison=True)
        self.assertEqual(report['statistics']['flow_quarantined'],8)
        self.assertEqual(report['samples'],40)

    def test_cross_connection_exact_duplicates_excluded_from_all_splits(self):
        output,report=self.prepare(duplicate=True)
        self.assertEqual(report['statistics']['excluded_cross_connection_or_label_patterns'],3)
        self.assertEqual(report['samples'],42);read_dataset(output)

    def test_no_overwrite_and_raw_source_integrity(self):
        output,report=self.prepare();before=(output/'manifest.json').read_bytes()
        with self.assertRaises((ValueError,OSError)):
            build_dataset(self.work/'normal',self.work/'attack',self.work,output,tshark='unused',tshark_sha256='0'*64)
        self.assertEqual((output/'manifest.json').read_bytes(),before)
        (self.work/'data/raw/normal.csv').write_text('TAMPERED')
        with self.assertRaisesRegex(ValueError,'MAPPING_SOURCE_HASH'):
            build_dataset(self.work/'normal',self.work/'attack',self.work,self.work/'artifacts/runs/tampered',tshark='unused',tshark_sha256='0'*64)

    def test_hdf5_tampering_and_unsafe_labels_rejected(self):
        import h5py
        output,_=self.prepare()
        with h5py.File(output/'dataset.h5','r+') as handle:handle['y'][0]=-1
        with self.assertRaisesRegex(ValueError,'HASH_MISMATCH'):read_dataset(output)
        self.refresh_hash(output)
        with self.assertRaisesRegex(ValueError,'INVALID_TARGET'):read_dataset(output)

    def test_mapping_label_tampering_rejected_against_actual_csv(self):
        normal=mapping_fixture(self.work,'normal',0);attack=mapping_fixture(self.work,'attack',1)
        with contextlib.closing(sqlite3.connect(normal/'alignment.sqlite')) as db:
            db.execute("UPDATE c SET alabel='1' WHERE ord=1");db.commit()
        with self.assertRaisesRegex(ValueError,'CSV_LABEL_VS_MAPPING_MISMATCH'):
            build_dataset(normal,attack,self.work,self.work/'artifacts/runs/dataset',tshark='unused',tshark_sha256='0'*64)

    def test_connection_leakage_rejected_even_with_new_manifest_hash(self):
        import h5py
        output,_=self.prepare();manifest,x,y,splits,clients,metadata=read_dataset(output)
        group=metadata[0]['group'];indices=[i for i,m in enumerate(metadata) if m['group']==group]
        with h5py.File(output/'dataset.h5','r+') as handle:
            handle['split'][indices[-1]]=(int(splits[0])+1)%3
            handle['client'][indices[-1]]=0 if (int(splits[0])+1)%3==0 else -1
        self.refresh_hash(output)
        with self.assertRaisesRegex(ValueError,'CONNECTION_GROUP_LEAKAGE'):read_dataset(output)

    def test_metrics_known_arithmetic_and_invalid_input(self):
        result=metrics([0,0,1,1],[0.1,0.7,0.9,0.2])
        self.assertEqual(result['confusion_matrix'],[[1,1],[1,1]])
        self.assertEqual(result['accuracy'],0.5);self.assertEqual(result['f1'],0.5)
        for y,p in [([],[]),([0],[float('nan')]),([2],[0.5]),([0],[2.])]:
            with self.assertRaises(ValueError):metrics(y,p)

    def test_cli_inspect_and_bounded_prepare_fail_closed(self):
        output,_=self.prepare()
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            self.assertEqual(main(['inspect','--dataset',str(output)]),0)
        self.assertEqual(json.loads(captured.getvalue())['samples'],48)
        for cap in (True,9,2001):
            with self.assertRaisesRegex(ValueError,'MAX_PER_CLASS'):
                build_dataset('unused','unused',self.work,self.work/'artifacts/runs/unused',
                    tshark='unused',tshark_sha256='0'*64,max_per_class=cap)

    @unittest.skipUnless(sys.platform=='win32','Windows-specific ctypes lifetime regression')
    def test_windows_rss_probe_reuses_ctypes_type_and_does_not_grow_unbounded(self):
        import os
        from nids_ml.pcap.decoder import process_rss,_windows_memory_api
        api=_windows_memory_api();before=process_rss(os.getpid())
        for _ in range(2000):self.assertIs(_windows_memory_api(),api);after=process_rss(os.getpid())
        self.assertIsNotNone(before);self.assertIsNotNone(after)
        self.assertLess(after-before,8*1048576)

    def test_expat_signed_32_bit_stream_offsets_preserve_record_guard(self):
        from nids_ml.pcap.decoder import PDMLParser,DecoderError
        parser=PDMLParser()
        for total,raw,expected in [(2**31+100,-2**31+50,2**31+50),
                (2**32+100,50,2**32+50),(2**32+100,2**32+50,2**32+50)]:
            parser.total=total;parser.parser=SimpleNamespace(CurrentByteIndex=raw)
            self.assertEqual(parser._position(),expected)
        parser.total=100;parser.parser=SimpleNamespace(CurrentByteIndex=-1)
        with self.assertRaises(DecoderError):parser._position()

    def test_tiny_cpu_training_checkpoint_and_weight_validation(self):
        import numpy as np
        from nids_ml.models.flow_cnn import configuration,build_model,train_step,ModelContractError
        output,_=self.prepare();target=self.work/'artifacts/runs/training'
        report=train(output,target,epochs=2,batch_size=16,client=0)
        self.assertEqual(report['status'],'PASS_MAPPED_DEMO_BASELINE')
        self.assertEqual(report['checkpoint_restore_max_prediction_error'],0.)
        self.assertFalse(report['swarm_training']);self.assertEqual(len(report['history']),2)
        self.assertTrue((target/'test-predictions.csv').is_file())
        cfg=configuration();model=build_model(cfg);x=np.zeros((2,20,11,1),dtype='float32');y=np.zeros((2,1),dtype='float32')
        for weights in (np.zeros(2,dtype='float32'),np.ones(2,dtype='float64'),np.ones(3,dtype='float32')):
            with self.assertRaises(ModelContractError):train_step(model,x,y,cfg,sample_weights=weights)
        raw=bytearray((target/'weights.bin').read_bytes());raw[0]^=1;(target/'weights.bin').write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'CHECKPOINT_HASH'):restore_checkpoint(target)


if __name__=='__main__':unittest.main()
