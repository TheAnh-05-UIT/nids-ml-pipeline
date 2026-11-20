"""Test opaque payload handling without weakening selected-field or XML guards.
Kiểm payload ngoài features bằng fixture giả; giữ guard metadata/XML và EOF.
"""
from contextlib import closing
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tests'))
from decoded_pcap_fixtures import make_capture,pdml,FakeDecoderProcess
from nids_ml.pcap.decoder import PDMLParser,ScenarioPDMLParser,DecodeLimits,map_record
from nids_ml.flows.features import CaptureContext
from nids_ml.flows.pcap_stream import stream_scenario
from nids_ml.labeling.contracts import canonical_bytes
from nids_ml.labeling.scenario import PROVENANCE
from nids_ml.storage.scenario import export_scenario_pipe,ScenarioShardReader,read_frame
from nids_ml.pcap.decoder import file_sha256


def parse(raw,parser=None):
    parser=ScenarioPDMLParser() if parser is None else parser
    rows=[]
    for start in range(0,len(raw),8192):rows.extend(parser.feed(raw[start:start+8192]))
    rows.extend(parser.feed(b'',final=True))
    return rows


def opaque(raw,count=157948,extra=b''):
    data=b'<proto name="fake-field-wrapper"><field name="data" value="'+b'a'*count+b'"'+extra+b'/></proto>'
    return raw.replace(b'</packet>',data+b'</packet>',1)


class OpaquePayloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.work=Path(self.tmp.name)
        mock=patch('shutil.disk_usage',return_value=SimpleNamespace(free=10*1024**3));mock.start();self.addCleanup(mock.stop)
        self.fixture=make_capture(self.work/'fixture')
        self.raw=pdml(self.fixture['records'])

    def test_real_failure_length_drops_only_unselected_value_and_keeps_all_packets(self):
        small=parse(opaque(self.raw,16));large=parse(opaque(self.raw))
        self.assertEqual(small,large)
        self.assertEqual([r.number('frame.number') for r in large],[1,2,3,4])
        self.assertTrue(all(r.quarantine_reason is None for r in large))
        self.assertFalse(any(f.name=='data' for r in large for f in r.fields))
        context=CaptureContext.from_dict(self.fixture['descriptor']['capture'])
        self.assertEqual(map_record(small[0],context,return_feature=True),map_record(large[0],context,return_feature=True))

    def test_legacy_parser_still_rejects_same_opaque_value(self):
        with self.assertRaisesRegex(ValueError,'XML_STRUCTURE_LIMIT'):parse(opaque(self.raw),PDMLParser())

    def test_attribute_length_boundary_and_split_chunks(self):
        for count in (65535,65536,65537,157948):
            with self.subTest(count=count):self.assertEqual(len(parse(opaque(self.raw,count))),4)

    def test_selected_numeric_field_is_not_sanitized(self):
        for count,error in ((513,'SELECTED_FIELD_SIZE_LIMIT'),(65537,'XML_STRUCTURE_LIMIT')):
            raw=self.raw.replace(b'name="tcp.ack"',b'value="'+b'a'*count+b'" name="tcp.ack"',1)
            with self.subTest(count=count),self.assertRaisesRegex(ValueError,error):parse(raw)

    def test_other_oversized_fields_and_other_data_attributes_still_fail(self):
        cases=[opaque(self.raw).replace(b'name="data"',b'name="ip.unselected"',1),
               opaque(self.raw,extra=b' show="'+b'x'*65537+b'"')]
        for raw in cases:
            with self.assertRaisesRegex(ValueError,'XML_STRUCTURE_LIMIT'):parse(raw)

    def test_attribute_count_and_depth_limits_still_fail(self):
        raw=opaque(self.raw,extra=b''.join(b' a'+str(i).encode()+b'="x"' for i in range(32)))
        with self.assertRaisesRegex(ValueError,'XML_STRUCTURE_LIMIT'):parse(raw)
        with self.assertRaisesRegex(ValueError,'XML_STRUCTURE_LIMIT'):
            parse(opaque(self.raw),ScenarioPDMLParser(DecodeLimits(max_depth=3)))

    def test_total_packet_bytes_and_field_count_still_fail(self):
        with self.assertRaisesRegex(ValueError,'RECORD.*BYTE_LIMIT'):parse(opaque(self.raw,1100000))
        with self.assertRaisesRegex(ValueError,'FIELD_COUNT_LIMIT'):
            parse(opaque(self.raw),ScenarioPDMLParser(DecodeLimits(max_fields=3)))

    def test_malformed_xml_and_entities_not_accepted(self):
        for raw in (opaque(self.raw)[:-9],b'<!DOCTYPE pdml [<!ENTITY e "x">]><pdml/>'):
            with self.assertRaises(ValueError):parse(raw)

    def test_scenario_pipe_hdf5_matches_small_payload_and_preserves_completion(self):
        f=self.fixture;capture=f['descriptor']['capture']
        policy=dict(format='nids-scenario-label-policy/1.0.0',dataset_id='synthetic-opaque-only',
            label_provenance=PROVENANCE,independent_ground_truth_verified=False,
            approval=dict(status='APPROVED_BY_OPERATOR',record_id='SYNTHETIC_NOT_REAL_APPROVAL',evidence='Synthetic unit-test only'),
            sources=[dict(source_id='normal-test',file_name='synthetic.pcap',sha256=capture['capture_sha256'],
                          size_bytes=f['source'].stat().st_size,records=4,publisher_split='test',y=0,
                          attack_type='Normal',max_lateness_ns=2*10**9)])
        policy_path=self.work/'policy.json';policy_path.write_bytes(canonical_bytes(policy));digest=file_sha256(policy_path)
        frames=[]
        for index,count in enumerate((16,157948)):
            stream=io.BytesIO();child=FakeDecoderProcess(opaque(self.raw,count))
            with patch('nids_ml.pcap.decoder.subprocess.Popen',return_value=child),patch('nids_ml.pcap.decoder.process_rss',return_value=1024):
                result=stream_scenario(f['source'],self.work/('producer'+str(index)),stream,
                    policy_path=policy_path,policy_sha=digest,source_id='normal-test',tshark=f['tool'],tshark_sha256=f['tool_sha'])
            self.assertTrue(result['source_complete']);stream.seek(0);selected=[]
            while stream.tell()<len(stream.getvalue()):
                kind,payload=read_frame(stream)
                if kind==b'T':selected.append(payload)
            frames.append(selected);stream.seek(0)
            directory=self.work/('hdf5'+str(index))
            manifest=export_scenario_pipe(stream,policy_path,digest,'normal-test',directory,max_output_mib=16)
            self.assertEqual(manifest['publisher_split'],'test')
            with closing(ScenarioShardReader(directory,policy_path,digest).batches()) as batches:
                self.assertEqual(sum(len(b['y']) for b in batches),manifest['samples'])
        self.assertEqual(frames[0],frames[1]);self.assertTrue(frames[0])


if __name__=='__main__':unittest.main()
