"""Small source-ordered PCAP/PDML fixtures and an owned fake decoder process.
Fixtures PCAP/PDML nhỏ theo thứ tự nguồn và decoder process mô phỏng, không dataset thật.
"""
from dataclasses import replace
import hashlib
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import quoteattr

from pcap_fixtures import frame,pcap
from test_pcap_decode import record
from nids_ml.flows.features import CaptureContext
from nids_ml.ordering.pcap import build_pcap_view


def make_capture(work,timestamps=(3000000,1000000,1000000,12000000),rejected=()):
    work=Path(work);(work/'data/raw').mkdir(parents=True);(work/'artifacts/runs').mkdir(parents=True)
    packets=[];records=[]
    for ordinal,microseconds in enumerate(timestamps,1):
        payload=frame(flags=0x10,ack=2000+ordinal)
        value=record(ordinal);stamp=f'{microseconds//1000000}.{microseconds%1000000:06d}000'
        values={'frame.time_epoch':stamp,'tcp.ack':str(ordinal*10),'tcp.ack_raw':str(2000+ordinal)}
        fields=tuple(replace(field,show=values.get(field.name,field.show),
            value=stamp if field.name=='timestamp' else field.value) for field in value.fields)
        value=replace(value,fields=fields)
        if ordinal in rejected:
            payload=b'\xff'*12+b'\x08\x06'+b'\0'*40
            value=replace(value,layers=('geninfo','frame','eth','arp'),all_protocols=('geninfo','frame','eth','arp'),
                          fields=tuple(field for field in fields if field.protocol in ('geninfo','frame')))
        packets.append((microseconds,payload));records.append(value)
    source=work/'data/raw/synthetic.pcap';data=pcap(packets);source.write_bytes(data)
    descriptor=dict(capture=CaptureContext('synthetic-decoded',hashlib.sha256(data).hexdigest(),
        'synthetic-run','synthetic-segment').to_dict(),path='data/raw/synthetic.pcap',
        size_bytes=len(data),expected_records=len(packets))
    view=work/'artifacts/runs/raw-view';manifest=build_pcap_view(descriptor,work,view)
    tool=work/'fake-tshark-not-executable';tool.write_bytes(b'FAKE TSHARK; NEVER EXECUTE')
    return dict(source=source,descriptor=descriptor,view=view,raw_manifest=manifest,tool=tool,
                tool_sha=hashlib.sha256(tool.read_bytes()).hexdigest(),records=records)


def pdml(records,*,close=True):
    parts=['<?xml version="1.0" encoding="utf-8"?><pdml>']
    for record in records:
        parts.append('<packet>')
        for protocol in record.layers:
            parts.append('<proto name='+quoteattr(protocol)+'>')
            for field in record.fields:
                if field.protocol!=protocol:continue
                attrs=' name='+quoteattr(field.name)
                if field.show is not None:attrs+=' show='+quoteattr(field.show)
                if field.value is not None:attrs+=' value='+quoteattr(field.value)
                parts.append('<field'+attrs+'/>')
            parts.append('</proto>')
        parts.append('</packet>')
    if close:parts.append('</pdml>')
    return ''.join(parts).encode()


class FakeDecoderProcess:
    def __init__(self,data,exit_code=0):
        self.stdout=BytesIO(data);self.stderr=BytesIO(b'');self.returncode=None
        self.pid=123456;self.exit_code=exit_code;self.terminated=False
    def poll(self):return self.returncode
    def wait(self,timeout=None):self.returncode=self.exit_code;return self.exit_code
    def terminate(self):self.terminated=True;self.returncode=-15
    def kill(self):self.terminated=True;self.returncode=-9
