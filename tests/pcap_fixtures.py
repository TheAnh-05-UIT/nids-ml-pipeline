"""Construct small synthetic Ethernet/IPv4 PCAP fixtures.
Tạo fixture PCAP Ethernet/IPv4 synthetic nhỏ.
"""
import socket
import struct


def checksum(data):
    if len(data)%2:data+=b'\0'
    value=sum(struct.unpack('!'+str(len(data)//2)+'H',data))
    while value>>16:value=(value&65535)+(value>>16)
    return (~value)&65535


def frame(protocol=6,*,reverse=False,flags=0x12,sequence=1000,ack=2001,window=4096,payload=b'',identifier=7,icmp_type=8,options=b''):
    src,dst=('192.0.2.10','192.0.2.20') if not reverse else ('192.0.2.20','192.0.2.10')
    source,target=(40000,8080) if not reverse else (8080,40000)
    if protocol==6:
        if len(options)%4 or len(options)>40:raise ValueError('TCP options phải đủ4-byte words, tối đa40bytes')
        transport=struct.pack('!HHIIBBHHH',source,target,sequence,ack,(5+len(options)//4)<<4,flags,window,0,0)+options+payload
        pseudo=socket.inet_aton(src)+socket.inet_aton(dst)+struct.pack('!BBH',0,6,len(transport))
        transport=transport[:16]+struct.pack('!H',checksum(pseudo+transport))+transport[18:]
    elif protocol==17:
        transport=struct.pack('!HHHH',source,target,8+len(payload),0)+payload
    else:
        transport=struct.pack('!BBHHH',icmp_type,0,0,identifier,1)+payload
        transport=transport[:2]+struct.pack('!H',checksum(transport))+transport[4:]
    ip=struct.pack('!BBHHHBBH4s4s',0x45,0,20+len(transport),1,0x4000,64,protocol,0,
                   socket.inet_aton(src),socket.inet_aton(dst))
    ip=ip[:10]+struct.pack('!H',checksum(ip))+ip[12:]
    return bytes.fromhex('0200000000020200000000010800')+ip+transport


def pcap(packets):
    result=bytearray(struct.pack('<IHHIIII',0xa1b2c3d4,2,4,0,0,262144,1))
    for timestamp_us,data in packets:
        seconds,microseconds=divmod(timestamp_us,1000000)
        result.extend(struct.pack('<IIII',seconds,microseconds,len(data),len(data)))
        result.extend(data)
    return bytes(result)


def example_pcap():
    # Handshake cho phép kiểm relative ACK khác raw ACK; ba transports đều hiện diện.
    return pcap([(0,frame(flags=2,ack=0,options=b'\x03\x03\x03\x01')),
                 (1000000,frame(reverse=True,flags=0x12,sequence=2000,ack=1001,options=b'\x03\x03\x03\x01')),
                 (2000000,frame(flags=0x10,sequence=1001,ack=2001,payload=b'abcd')),
                 (3000000,frame(17,payload=b'abcdefgh')),
                 (10000000,frame(1,payload=b'abcdefgh')),
                 (10000001,frame(1,reverse=True,identifier=7,icmp_type=0,payload=b'abcdefgh'))])
