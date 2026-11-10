"""Build small TEST-NET packet-metadata fixtures for flow tests.
Tạo fixture metadata packet TEST-NET nhỏ để test flow.
"""
import hashlib

CAPTURE_TEXT='ARM-011D1 synthetic packets, not a PCAP.'


def context():
    return dict(capture_id='synthetic-flow-capture',capture_sha256=hashlib.sha256(CAPTURE_TEXT.encode()).hexdigest(),
                run_id='synthetic-run-1',segment_id='segment-1')


def packet(protocol=6,ordinal=1,timestamp_ns=0,flow=0,reverse=False,capture=None):
    tcp=udp=icmp=None
    if protocol==6:
        tcp=dict(src_port=40000+flow,dst_port=8080,length=20,ack=123,flags=18,window_size=4096)
        length=60;highest='TCP';stack='eth:ethertype:ip:tcp'
    elif protocol==17:
        udp=dict(src_port=40000+flow,dst_port=53,length=32)
        length=52;highest='DNS';stack='eth:ethertype:ip:udp:dns'
    else:
        icmp=dict(type=8,code=0,identifier=7)
        length=64;highest='ICMP';stack='eth:ethertype:ip:icmp'
    src,dst='192.0.2.10','192.0.2.20'
    if reverse:
        src,dst=dst,src
        if tcp:tcp['src_port'],tcp['dst_port']=tcp['dst_port'],tcp['src_port']
        if udp:udp['src_port'],udp['dst_port']=udp['dst_port'],udp['src_port']
    return dict(schema_version='1.0.0',input_kind='synthetic_fixture',capture=context() if capture is None else capture,
        ordinal=ordinal,timestamp_ns=timestamp_ns,ip_version=4,src_ip=src,dst_ip=dst,ip_protocol=protocol,
        ip_length=length,ip_flags=2,fragmented=False,highest_layer=highest,protocol_stack=stack,tcp=tcp,udp=udp,icmp=icmp)


def bundle():
    return dict(kind='synthetic_fixture',capture_fixture_text=CAPTURE_TEXT,capture=context(),input_complete=True,
        packets=[packet(),packet(ordinal=2,timestamp_ns=2500000000,reverse=True),
                 packet(protocol=17,ordinal=3,timestamp_ns=3000000000),
                 packet(protocol=1,ordinal=4,timestamp_ns=10000000000),
                 packet(ordinal=5,timestamp_ns=10000000001)])
