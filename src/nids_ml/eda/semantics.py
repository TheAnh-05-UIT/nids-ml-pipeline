"""Audit feature meanings and label consistency without repairing raw data.
Audit ý nghĩa feature và tính nhất quán nhãn, không sửa raw data.

Các định nghĩa đối chiếu Wireshark Display Filter Reference. Dissector version,
export options và mapping từ packet/capture sang curated CSV vẫn chưa được cấp.
"""
from collections import Counter, defaultdict
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import ipaddress
import math
from pathlib import Path
import re

from nids_ml.eda.csv_eda import BoundedLines
from nids_ml.eda.csv_eda import classify
from nids_ml.data.inventory import fingerprint

# name, protocol, ý nghĩa kỳ vọng, rule, nhóm rủi ro.
DEFINITIONS = (
 ('frame.time','frame','Thời điểm capture của frame','time','identity'),
 ('ip.src_host','ip','Nguồn IP hoặc tên host đã resolve','host','identity'),
 ('ip.dst_host','ip','Đích IP hoặc tên host đã resolve','host','identity'),
 ('arp.dst.proto_ipv4','arp','Địa chỉ IPv4 target trong ARP','ipv4','identity'),
 ('arp.opcode','arp','Mã thao tác ARP','u16','header'),
 ('arp.hw.size','arp','Số byte địa chỉ phần cứng ARP','u8','header'),
 ('arp.src.proto_ipv4','arp','Địa chỉ IPv4 sender trong ARP','ipv4','identity'),
 ('icmp.checksum','icmp','Checksum ICMP, không phải nhãn checksum tốt/xấu','u16','session'),
 ('icmp.seq_le','icmp','Sequence ICMP nhìn theo little endian','u16','session'),
 ('icmp.transmit_timestamp','icmp','Timestamp trong ICMP, không phải capture time','u32','timing'),
 ('icmp.unused','icmp','Bytes trường unused của ICMP','payload','annotation'),
 ('http.file_data','http','Nội dung body HTTP','payload','payload'),
 ('http.content_length','http','Độ dài body được header HTTP công bố','u64','header'),
 ('http.request.uri.query','http','Query string URI, có thể chứa định danh hoặc nội dung tấn công','string','payload'),
 ('http.request.method','http','HTTP method','method','category'),
 ('http.referer','http','Referer URI do client cung cấp','string','payload'),
 ('http.request.full_uri','http','URI đầy đủ của HTTP request','string','payload'),
 ('http.request.version','http','Phiên bản giao thức trong request HTTP','http_version','category'),
 ('http.response','http','Chỉ báo response của dissector','bit','annotation'),
 ('http.tls_port','http','Expert annotation về HTTP rõ trên port dự kiến mã hóa; KHÔNG phải số port','annotation','annotation'),
 ('tcp.ack','tcp','ACK number, có thể relative theo cấu hình dissector','u32','session'),
 ('tcp.ack_raw','tcp','ACK number nguyên bản trên wire','u32','session'),
 ('tcp.checksum','tcp','Giá trị checksum TCP','u16','session'),
 ('tcp.connection.fin','tcp','Expert annotation kết thúc kết nối','bit','annotation'),
 ('tcp.connection.rst','tcp','Expert annotation reset kết nối','bit','annotation'),
 ('tcp.connection.syn','tcp','Expert annotation thiết lập kết nối','bit','annotation'),
 ('tcp.connection.synack','tcp','Expert annotation SYN+ACK','bit','annotation'),
 ('tcp.dstport','tcp','TCP destination port','u16','port'),
 ('tcp.flags','tcp','Bitmask flags TCP','u16','header'),
 ('tcp.flags.ack','tcp','Bit ACK trong TCP header','bit','header'),
 ('tcp.len','tcp','Độ dài TCP segment theo dissector','u32','header'),
 ('tcp.options','tcp','Bytes options TCP','payload','payload'),
 ('tcp.payload','tcp','Bytes payload TCP','payload','payload'),
 ('tcp.seq','tcp','Sequence number TCP, có thể relative','u32','session'),
 ('tcp.srcport','tcp','TCP source port','u16','port'),
 ('udp.port','udp','Port nguồn hoặc đích; direction/multiplicity cần export contract','u16','port'),
 ('udp.stream','udp','Stream index cục bộ của capture/dissector','u32','identity'),
 ('udp.time_delta','udp','Thời gian từ frame trước theo dissector; không phải timestamp tuyệt đối','nonnegative','timing'),
 ('dns.qry.name','dns','Tên DNS được truy vấn, không phải số liên tục','string','identity'),
 ('dns.qry.name.len','dns','Độ dài tên query DNS','u16','formatting'),
 ('dns.qry.qu','dns','Boolean QU question flag','bit','header'),
 ('dns.qry.type','dns','Mã type query DNS','u16','header'),
 ('dns.retransmission','dns','Boolean retransmission theo state của dissector','bit','annotation'),
 ('dns.retransmit_request','dns','Expert annotation query retransmission','bit','annotation'),
 ('dns.retransmit_request_in','dns','Số frame của query gốc; không phải counter general','u32','identity'),
 ('mqtt.conack.flags','mqtt','Bitmask CONNACK flags','u8','formatting'),
 ('mqtt.conflag.cleansess','mqtt','Boolean clean session','bit','header'),
 ('mqtt.conflags','mqtt','Bitmask CONNECT flags','u8','header'),
 ('mqtt.hdrflags','mqtt','Flags trong MQTT header','u8','header'),
 ('mqtt.len','mqtt','Độ dài message theo dissector','u64','header'),
 ('mqtt.msg_decoded_as','mqtt','Tên/chuỗi cơ chế decode message, không phải số measurement','string','annotation'),
 ('mqtt.msg','mqtt','Nội dung message MQTT','payload','payload'),
 ('mqtt.msgtype','mqtt','Loại MQTT control message','mqtt_type','header'),
 ('mqtt.proto_len','mqtt','Độ dài tên giao thức MQTT','u16','header'),
 ('mqtt.protoname','mqtt','Tên giao thức MQTT trong CONNECT','string','formatting'),
 ('mqtt.topic','mqtt','Tên topic; có thể nhận diện thiết bị hoặc kịch bản','string','formatting'),
 ('mqtt.topic_len','mqtt','Độ dài topic MQTT','u16','header'),
 ('mqtt.ver','mqtt','Protocol level MQTT, không phải phiên bản software','mqtt_version','header'),
 ('mbtcp.len','mbtcp','Length field trong MBAP','u16','header'),
 ('mbtcp.trans_id','mbtcp','Transaction identifier Modbus/TCP','u16','identity'),
 ('mbtcp.unit_id','mbtcp','Unit identifier trong MBAP, không tự đồng nhất physical client','u8','identity'),
)
LABELS = ('Attack_label', 'Attack_type')
METHODS = frozenset(('GET','HEAD','POST','PUT','DELETE','CONNECT','OPTIONS','TRACE','PATCH'))
ZERO = frozenset(('0','0.0'))


def source_url(protocol):
    return 'https://www.wireshark.org/docs/dfref/' + protocol[0] + '/' + protocol + '.html'


def value_kind(raw, rule):
    """Nhận dạng lexical; zero/sentinel không đồng nghĩa field protocol tồn tại."""
    text = raw.strip()
    if not text:
        return 'blank'
    if text.casefold() in ('na','n/a','null','none','?','nan','+nan','-nan'):
        return 'missing_token'
    if raw in ZERO:
        return 'raw_zero' if raw == '0' else 'raw_zero_decimal'
    if rule in ('u8','u16','u32','u64','bit','nonnegative','mqtt_type','mqtt_version'):
        try:
            if len(text)>128:return 'numeric_literal_too_long'
            number = Decimal(int(text,16)) if re.fullmatch(r'0[xX][0-9a-fA-F]{1,16}',text) else Decimal(text)
        except (ValueError, InvalidOperation):
            return 'non_numeric'
        if not number.is_finite():
            return 'nonfinite'
        if number < 0:
            return 'negative'
        if rule == 'nonnegative':
            return 'valid_nonnegative'
        if number!=number.to_integral_value():
            return 'noninteger'
        upper = {'u8':255,'u16':65535,'u32':4294967295,'u64':18446744073709551615,
                 'bit':1,'mqtt_type':15,'mqtt_version':255}[rule]
        if number > upper:
            return 'out_of_range'
        if rule == 'mqtt_version' and number not in (3,4,5):
            return 'unexpected_protocol_level'
        return 'valid_integer'
    if rule in ('ipv4','host','time'):
        try:
            if len(text)<=64:
                address=ipaddress.ip_address(text)
                return 'ip_literal' if rule != 'ipv4' or address.version==4 else 'unexpected_ipv6'
        except ValueError:
            pass
        if rule == 'time':
            if re.fullmatch(r'\d{4}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?',text):
                return 'year_time_without_month_day'
            if re.match(r'(?:[A-Za-z]{3}\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})[ T]',text):
                return 'calendar_timestamp_candidate_not_validated'
            try:
                float(text)
                return 'numeric_not_capture_timestamp'
            except ValueError:
                return 'unrecognized_time_text'
        return 'hostname_candidate' if rule=='host' and re.fullmatch(r'[A-Za-z0-9_.-]{1,253}',text) else 'non_address_text'
    if rule == 'method':
        return 'known_HTTP_method' if text in METHODS else 'unrecognized_method'
    if rule == 'http_version':
        return 'HTTP_version_candidate' if re.fullmatch(r'HTTP/\d(?:\.\d)?',text) else 'unrecognized_HTTP_version'
    if rule == 'payload':
        return 'hex_bytes_candidate' if re.fullmatch(r'[0-9A-Fa-f]{2}(?::?[0-9A-Fa-f]{2})*',text) else 'other_payload_representation'
    return 'nonzero_text_or_annotation'


def safe_example(raw, rule):
    # Không xuất raw URI/IP/payload/topic/timestamp, dù chúng nằm trong dataset công khai.
    if raw in ZERO or (len(raw)<=32 and re.fullmatch(r'[+-]?\d+(?:\.\d+)?',raw) and rule.startswith(('u','bit','mqtt_','nonnegative'))):
        return {'numeric_or_sentinel_text':raw}
    if (rule=='method' and raw in METHODS) or (rule=='http_version' and re.fullmatch(r'HTTP/\d(?:\.\d)?',raw)):
        return {'protocol_token':raw}
    return {'sha256':hashlib.sha256(raw.encode()).hexdigest(),'characters':len(raw),'raw_redacted':True}


def payload_markers(raw):
    """Chỉ kiểm tối đa 512 bytes đầu; pattern match không chứng minh maliciousness."""
    text=raw[:1024].replace(':','')
    if len(text)%2:
        text=text[:-1]
    try:
        content=bytes.fromhex(text)[:512].lower()
    except ValueError:
        content=raw[:512].encode('utf-8').lower()
    return [name for name,needle in (('HTTP_literal',b'http/'),('script_literal',b'<script'),
                                    ('SQL_literal',b'union select'),('shell_literal',b'/bin/sh')) if needle in content]


def audit_semantics(path, expected_header, focus=None):
    identity=fingerprint(path)
    definitions=[d for d in DEFINITIONS if focus is None or d[0] in focus]
    if focus and set(focus)!={d[0] for d in definitions}:
        raise ValueError('Unknown focus feature')
    counts={d[0]:defaultdict(Counter) for d in definitions}
    examples={d[0]:{} for d in definitions}; markers=defaultdict(Counter)
    rows=0; classes=set()
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        lines=BoundedLines(stream,1048576); reader=csv.reader(lines,strict=True)
        header=lines.record(reader)
        if header!=expected_header or len(set(header))!=len(header):
            raise ValueError('Exact source schema/order mismatch')
        ids=[(header.index(name),name,rule) for name,_,_,rule,_ in definitions]
        label_id=header.index('Attack_type')
        while True:
            try:row=lines.record(reader)
            except StopIteration:break
            if len(row)!=len(header):raise ValueError('CSV width mismatch')
            label=row[label_id]
            if len(label)>256:raise ValueError('Oversized label')
            classes.add(label)
            if len(classes)>1000:raise ValueError('Class budget exceeded')
            rows+=1
            for i,name,rule in ids:
                raw=row[i];kind=value_kind(raw,rule);counts[name][kind][label]+=1
                if kind not in examples[name] and len(examples[name])<12:
                    examples[name][kind]=safe_example(raw,rule)
                if name=='tcp.payload':
                    for marker in payload_markers(raw):markers[marker][label]+=1
    if fingerprint(path)!=identity:raise ValueError('Input changed during semantic audit')
    return {'schema_version':'1.0.0','input':identity,'row_count':rows,'scope':'full_scan_rule_counts; not cleaning or causal inference',
            'ruleset':'Wireshark field expectations; exact original exporter version/mapping unknown',
            'columns':{name:{'kinds':{kind:{'count':sum(c.values()),'by_class':dict(sorted(c.items()))} for kind,c in sorted(groups.items())},
                             'safe_examples':examples[name]} for name,groups in counts.items()},
            'payload_marker_counts_prefix_only':{key:dict(sorted(c.items())) for key,c in sorted(markers.items())},
            'raw_payload_in_report':False,'max_record_chars':1048576,'feature_transforms_applied':False}


def audit_label_integrity(path,expected_header):
    """Đối chiếu nhãn và phân biệt raw Inf với overflow parse của payload bytes."""
    from nids_ml.eda.overlap import stream_rows
    identity=fingerprint(path); pairs=Counter(); problematic=[]; rows=0; mismatch=0
    counts=defaultdict(Counter); last=None;runs=0
    ids={n:expected_header.index(n) for n in (*LABELS,'tcp.options','tcp.payload','http.request.uri.query')}
    for row in stream_rows(path,expected_header):
        rows+=1; binary,label=(row[ids[n]] for n in LABELS)
        if max(len(binary),len(label))>256:raise ValueError('Label token too long')
        pairs[binary,label]+=1
        if len(pairs)>1000:raise ValueError('Label pair cap exceeded')
        mismatch+=int(binary!=('0' if label=='Normal' else '1'))
        runs+=int(last!=label);last=label
        for name in ('tcp.options','tcp.payload','http.request.uri.query'):
            raw=row[ids[name]];kind,_=classify(raw)
            if kind in ('infinity','nan','null_token','blank'):
                lexical=('literal_nonfinite' if raw.strip().casefold() in ('inf','+inf','-inf','infinity','+infinity','-infinity','nan','+nan','-nan')
                         else 'finite_literal_float64_overflow' if kind=='infinity' else 'missing_token_by_config')
                counts[name][lexical]+=1
                if len(problematic)>=1000:raise ValueError('Exceptional-cell reporting cap exceeded')
                problematic.append({'column':name,'data_row':rows,'class':label,'parser_kind':kind,'lexical_explanation':lexical,
                                    'characters':len(raw),'sha256':hashlib.sha256(raw.encode()).hexdigest(),
                                    'safe_missing_token':raw if kind=='null_token' else None})
    if fingerprint(path)!=identity:raise ValueError('Input changed during audit')
    return {'input':identity,'row_count':rows,'scope':'full scan, exact raw label pairs; only three named columns rechecked for nonfinite/missing token explanation',
            'label_pairs':[{'Attack_label':a,'Attack_type':b,'count':v} for (a,b),v in sorted(pairs.items())],
            'binary_vs_type_mismatches':mismatch,'contiguous_class_runs':runs,
            'exceptional_cell_counts':{k:dict(v) for k,v in counts.items()},'exceptional_cells_redacted':problematic,
            'raw_content_executed_or_transformed':False}
