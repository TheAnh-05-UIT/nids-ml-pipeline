"""Stream and validate TShark PDML packet metadata under explicit budgets.
Đọc/kiểm metadata packet TShark PDML theo luồng và budget rõ ràng.

Mỗi source record được kiểm riêng; không filter/sort hay giữ cả capture trong RAM.
RSS thresholds là ngưỡng abort được lấy mẫu, không là OS hard memory limit.
"""
from collections import deque
from functools import lru_cache
from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time
from xml.parsers import expat

from nids_ml.flows.features import CaptureContext, ExtractionError, extract_features

DECODER_VERSION = 'pcap-decoder/1.0.0'
PREFERENCES = {'tcp.relative_sequence_numbers':'TRUE', 'tcp.analyze_sequence_numbers':'TRUE',
               'tcp.desegment_tcp_streams':'TRUE', 'tcp.reassemble_out_of_order':'FALSE',
               'ip.defragment':'TRUE'}
FIELD_NAMES = frozenset(('frame.number','frame.time_epoch','frame.len','frame.cap_len','frame.protocols',
    'ip.version','ip.src','ip.dst','ip.proto','ip.len','ip.flags','ip.flags.mf','ip.frag_offset',
    'ip.hdr_len','ip.reassembled_in','tcp.srcport','tcp.dstport','tcp.len','tcp.ack','tcp.ack_raw',
    'tcp.flags','tcp.window_size_value','tcp.window_size','tcp.hdr_len',
    'udp.srcport','udp.dstport','udp.length','icmp.type','icmp.code','icmp.ident','icmp.ident_le','icmp.seq',
    'num','timestamp','len','caplen'))


class DecoderError(ValueError):
    """Lỗi decoder/contract: không trả success cho output dở dang."""


class IneligiblePacket(DecoderError):
    """Packet được decode nhưng không thuộc eligibility của flow profile."""


def epoch_ns(value):
    """Đổi decimal epoch chính xác; không float hoặc làm tròn timestamp."""
    if type(value) is not str or len(value)>40 or not re.fullmatch(r'\d+(?:\.\d+)?',value):
        raise DecoderError('INVALID_TIMESTAMP')
    seconds,_,fraction=value.partition('.')
    if any(character!='0' for character in fraction[9:]):
        raise DecoderError('TIMESTAMP_PRECISION_OR_RANGE')
    result=int(seconds)*1000000000+int((fraction[:9]+'000000000')[:9])
    if not 0<=result<=2**63-1:
        raise DecoderError('TIMESTAMP_PRECISION_OR_RANGE')
    return result


def regular_file(path):
    path=Path(path).absolute()
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink!=1:
        raise DecoderError('INDEPENDENT_REGULAR_FILE_REQUIRED')
    # Không resolve symlink trong ancestor để đổi sang dataset khác âm thầm.
    if any(parent.is_symlink() for parent in path.parents):
        raise DecoderError('SYMLINK_ANCESTOR')
    return path


def file_sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def file_signature(path):
    stat=path.stat()
    return stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns


@dataclass(frozen=True)
class DecodeLimits:
    packet_count:int=256
    record_bytes:int=1048576
    stdout_bytes:int=16777216
    stderr_bytes:int=65536
    max_fields:int=8192
    max_depth:int=64
    timeout_seconds:float=30.0
    close_seconds:float=5.0
    python_rss_bytes:int=134217728
    decoder_rss_bytes:int=536870912

    def validate(self):
        for name,low,high in [('packet_count',1,4096),('record_bytes',1024,1048576),
                ('stdout_bytes',1024,67108864),('stderr_bytes',128,65536),('max_fields',1,8192),
                ('max_depth',2,64),('python_rss_bytes',1048576,2147483648),
                ('decoder_rss_bytes',1048576,2147483648)]:
            value=getattr(self,name)
            if type(value) is not int or not low<=value<=high:
                raise DecoderError('INVALID_LIMIT:'+name)
        for name,maximum in [('timeout_seconds',60),('close_seconds',5)]:
            value=getattr(self,name)
            if type(value) not in (int,float) or not 0<value<=maximum:
                raise DecoderError('INVALID_LIMIT:'+name)
        return self


@dataclass(frozen=True)
class Field:
    name:str
    show:str|None
    value:str|None
    protocol:str


@dataclass(frozen=True)
class PDMLRecord:
    layers:tuple
    all_protocols:tuple
    fields:tuple

    def field(self,name,*,protocol=None):
        values=[x for x in self.fields if x.name==name]
        expected_protocol=protocol or (name.split('.')[0] if '.' in name else None)
        if len(values)!=1 or (expected_protocol is not None and values[0].protocol!=expected_protocol):
            raise DecoderError('FIELD_MISSING_OR_AMBIGUOUS:'+name)
        return values[0]

    def number(self,name,*,base=10,protocol=None):
        field=self.field(name,protocol=protocol)
        value=field.show
        pattern=r'(?:0x)?[0-9a-fA-F]+' if base==16 else r'\d+'
        if value is None or len(value)>32 or re.fullmatch(pattern,value) is None:
            raise DecoderError('INVALID_NUMERIC_FIELD:'+name)
        return int(value,base)

    def boolean(self,name):
        field=self.field(name)
        if field.show not in ('True','False','1','0'):
            raise DecoderError('INVALID_BOOLEAN_FIELD:'+name)
        result=field.show in ('True','1')
        if field.value is not None and field.value not in ('1' if result else '0',):
            raise DecoderError('BOOLEAN_SHOW_RAW_MISMATCH:'+name)
        return result

    def timestamp_ns(self):
        field=self.field('frame.time_epoch')
        candidates=[v for v in (field.show,field.value) if v and re.fullmatch(r'\d+(?:\.\d+)?',v)]
        if not candidates:
            raise DecoderError('INVALID_TIMESTAMP')
        values={epoch_ns(v) for v in candidates}
        if len(values)!=1:
            raise DecoderError('AMBIGUOUS_TIMESTAMP')
        result=next(iter(values))
        geninfo=self.field('timestamp',protocol='geninfo')
        if geninfo.value is None or epoch_ns(geninfo.value)!=result:
            raise DecoderError('GENINFO_TIMESTAMP_MISMATCH')
        return result


class PDMLParser:
    """Expat strict events: không recover, DTD/entities hoặc whole-document tree."""

    def __init__(self,limits=None):
        self.limits=(limits or DecodeLimits()).validate()
        self.parser=expat.ParserCreate('UTF-8')
        self.parser.StartElementHandler=self._start
        self.parser.EndElementHandler=self._end
        self.parser.CharacterDataHandler=self._text
        self.parser.StartDoctypeDeclHandler=self._forbidden
        self.parser.EntityDeclHandler=self._forbidden
        self.parser.ExternalEntityRefHandler=self._forbidden
        self.parser.XmlDeclHandler=self._declaration
        self.parser.ProcessingInstructionHandler=self._instruction
        self.stack=[];self.protocol_stack=[];self.records=deque();self.layers=[];self.protocols=[];self.fields=[]
        self.total=0;self.count=0;self.field_count=0;self.packet_start=None;self.last_boundary=0
        self.closed_root=False;self.finished=False;self.peak_record_bytes=0

    def _forbidden(self,*_):
        raise DecoderError('DTD_OR_ENTITY_FORBIDDEN')

    def _position(self):
        # Windows Expat can expose a signed 32-bit byte index for a >2 GiB stream.
        # Khôi phục offset stream trên Windows; giữ nguyên mọi giới hạn record/stdout.
        index=self.parser.CurrentByteIndex
        if -2**31<=index<0:index+=2**32
        if self.total>=2**32 and 0<=index<2**32:
            index+=(self.total//2**32)*2**32
            if index>self.total:index-=2**32
        if not 0<=index<=self.total:raise DecoderError('INVALID_EXPAT_BYTE_INDEX')
        return index

    def _declaration(self,version,encoding,standalone):
        if version!='1.0' or (encoding and encoding.lower().replace('-','')!='utf8'):
            raise DecoderError('UNSUPPORTED_XML_DECLARATION')

    def _instruction(self,target,data):
        if target!='xml-stylesheet' or len(data)>1024:
            raise DecoderError('UNSUPPORTED_XML_INSTRUCTION')

    def _text(self,text):
        if text.strip():
            raise DecoderError('UNEXPECTED_XML_TEXT')

    def _start(self,name,attrs):
        depth=len(self.stack)
        if depth>=self.limits.max_depth or len(attrs)>32 or any(len(v)>65536 for v in attrs.values()):
            raise DecoderError('XML_STRUCTURE_LIMIT')
        if depth==0:
            if name!='pdml' or self.closed_root:
                raise DecoderError('PDML_ROOT_REQUIRED')
        elif depth==1:
            if name!='packet' or self.count>=self.limits.packet_count:
                raise DecoderError('PACKET_COUNT_OR_ROOT_CHILD')
            self.packet_start=self._position()
            self.layers=[];self.protocols=[];self.fields=[];self.field_count=0
        else:
            if name not in ('proto','field'):
                raise DecoderError('UNKNOWN_PDML_ELEMENT')
            self.field_count+=1
            if self.field_count>self.limits.max_fields:
                raise DecoderError('FIELD_COUNT_LIMIT')
            if name=='proto':
                protocol=attrs.get('name','')
                if not protocol or len(protocol)>64:
                    raise DecoderError('INVALID_LAYER_NAME')
                self.protocols.append(protocol)
                if depth==2:self.layers.append(protocol)
                self.protocol_stack.append(protocol)
            elif attrs.get('name') in FIELD_NAMES:
                if not self.protocol_stack:
                    raise DecoderError('FIELD_WITHOUT_LAYER')
                show,value=attrs.get('show'),attrs.get('value')
                if any(v is not None and len(v)>512 for v in (show,value)):
                    raise DecoderError('SELECTED_FIELD_SIZE_LIMIT')
                self.fields.append(Field(attrs['name'],show,value,self.protocol_stack[-1]))
        self.stack.append(name)

    def _end(self,name):
        if not self.stack or self.stack[-1]!=name:
            raise DecoderError('XML_NESTING')
        if name=='proto':self.protocol_stack.pop()
        if name=='packet':
            size=self._position()+len('</packet>')-self.packet_start
            if size>self.limits.record_bytes:raise DecoderError('RECORD_BYTE_LIMIT')
            self.peak_record_bytes=max(self.peak_record_bytes,size)
            self.records.append(PDMLRecord(tuple(self.layers),tuple(self.protocols),tuple(self.fields)))
            self.count+=1;self.packet_start=None;self.last_boundary=self._position()+len('</packet>')
        if name=='pdml':self.closed_root=True
        self.stack.pop()

    def feed(self,data,*,final=False):
        if self.finished or type(data) is not bytes or len(data)>8192:
            raise DecoderError('INVALID_PARSER_FEED')
        self.total+=len(data)
        if self.total>self.limits.stdout_bytes:raise DecoderError('STDOUT_BYTE_LIMIT')
        try:
            self.parser.Parse(data,final)
        except expat.ExpatError as error:
            raise DecoderError('MALFORMED_PDML') from error
        anchor=self.packet_start if self.packet_start is not None else self.last_boundary
        if self.total-anchor>self.limits.record_bytes:
            raise DecoderError('RECORD_OR_HEADER_BYTE_LIMIT')
        if final:
            self.finished=True
            if not self.closed_root or self.stack:raise DecoderError('INCOMPLETE_PDML')
        while self.records:yield self.records.popleft()


def map_record(record,context):
    """Ánh xạ packet thật sang scope decoded_pcap, giữ metadata ngoài features."""
    ordinal=record.number('frame.number');timestamp=record.timestamp_ns()
    if record.layers[:2]!=('geninfo','frame'):
        raise DecoderError('GENINFO_FRAME_LAYER_ORDER')
    if '_ws.malformed' in record.all_protocols:
        raise DecoderError('MALFORMED_PACKET')
    wire,captured=record.number('frame.len'),record.number('frame.cap_len')
    if wire!=captured:raise IneligiblePacket('TRUNCATED_CAPTURE')
    if record.all_protocols.count('ip')!=1 or 'ipv6' in record.all_protocols:
        raise IneligiblePacket('NON_IPV4_OR_AMBIGUOUS_IP')
    if record.number('ip.version')!=4:raise IneligiblePacket('NON_IPV4')
    proto=record.number('ip.proto')
    if proto not in (6,17,1):raise IneligiblePacket('UNSUPPORTED_IP_PROTOCOL')
    if record.boolean('ip.flags.mf') or record.number('ip.frag_offset') or any(x.name=='ip.reassembled_in' for x in record.fields):
        raise IneligiblePacket('FRAGMENT_OR_REASSEMBLY')
    transport={6:'tcp',17:'udp',1:'icmp'}[proto]
    if record.all_protocols.count(transport)!=1 or any(p in record.all_protocols for p in ('tcp','udp','icmp') if p!=transport):
        raise IneligiblePacket('AMBIGUOUS_TRANSPORT')
    ip_length,ihl=record.number('ip.len'),record.number('ip.hdr_len')
    if not 20<=ihl<=60 or ihl%4 or ip_length<ihl:
        raise DecoderError('INVALID_IPV4_HEADER_LENGTH')
    tcp=udp=icmp=None;sidecar=dict(source_ordinal=ordinal,timestamp_ns=timestamp,
        captured_bytes=captured,wire_bytes=wire,ipv4_header_bytes=ihl,ordered_layers=list(record.layers),
        label='UNKNOWN',supervised_eligible=False)
    if proto==6:
        header=record.number('tcp.hdr_len');length=record.number('tcp.len')
        if not 20<=header<=60 or header%4 or length!=ip_length-ihl-header:
            raise DecoderError('INVALID_TCP_HEADER_OR_PAYLOAD_LENGTH')
        tcp=dict(src_port=record.number('tcp.srcport'),dst_port=record.number('tcp.dstport'),
            length=length,ack=record.number('tcp.ack'),flags=record.number('tcp.flags',base=16),
            window_size=record.number('tcp.window_size_value'))
        sidecar.update(tcp_header_bytes=header,tcp_ack_raw=record.number('tcp.ack_raw'),
                       tcp_window_scaled=record.number('tcp.window_size'))
    elif proto==17:
        length=record.number('udp.length')
        if length!=ip_length-ihl:raise DecoderError('INVALID_UDP_LENGTH')
        udp=dict(src_port=record.number('udp.srcport'),dst_port=record.number('udp.dstport'),length=length)
    else:
        try:identifier=record.number('icmp.ident')
        except DecoderError as error:raise IneligiblePacket('ICMP_IDENTIFIER_UNAVAILABLE') from error
        icmp=dict(type=record.number('icmp.type'),code=record.number('icmp.code'),identifier=identifier)
        sidecar.update(icmp_identifier=identifier,icmp_type=icmp['type'],icmp_code=icmp['code'])
    packet=dict(schema_version='1.1.0',input_kind='decoded_pcap',capture=context.to_dict(),
        ordinal=ordinal,timestamp_ns=timestamp,ip_version=4,
        src_ip=record.field('ip.src').show,dst_ip=record.field('ip.dst').show,ip_protocol=proto,
        ip_length=ip_length,ip_flags=record.number('ip.flags',base=16),fragmented=False,
        # XmlLayer.layer_name của PyShark đổi fake-field-wrapper thành DATA.
        highest_layer=('DATA' if record.layers[-1]=='fake-field-wrapper' else record.layers[-1].upper()),
        protocol_stack=record.field('frame.protocols').show,
        tcp=tcp,udp=udp,icmp=icmp)
    feature=extract_features(packet,input_kind='decoded_pcap')
    sidecar.update(packet_sha256=feature.packet_sha256,hash_domain='canonical_D1_decoded_metadata_JSON')
    return packet,sidecar


def process_rss(pid):
    """Read RSS for an owned PID on Linux/Windows; unavailable is never zero RAM.
    Đọc RSS đúng PID trên Linux/Windows; không coi thiếu probe là RAM bằng 0.
    """
    if type(pid) is not int or pid<=0:return None
    if sys_platform_linux():
        try:
            with open('/proc/'+str(pid)+'/status',encoding='ascii') as stream:
                for line in stream:
                    match=re.fullmatch(r'VmRSS:\s+(\d+)\s+kB\s*',line)
                    if match:return int(match[1])*1024
        except (OSError,UnicodeError):pass
        return None
    if os.name!='nt':return None
    import ctypes
    Counters,kernel,psapi=_windows_memory_api()
    handle=kernel.OpenProcess(0x1000|0x10,False,pid)
    if not handle:return None
    try:
        counters=Counters();counters.cb=ctypes.sizeof(counters)
        return counters.WorkingSetSize if psapi.GetProcessMemoryInfo(handle,ctypes.byref(counters),counters.cb) else None
    finally:kernel.CloseHandle(handle)


@lru_cache(maxsize=1)
def _windows_memory_api():
    # Cache the ctypes type: POINTER(type) retains each new type globally.
    # Tái sử dụng type ctypes để việc đo RAM không tự gây tăng RAM qua pointer cache.
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD),
                  *[(name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize',
                    'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
                    'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    return Counters,kernel,psapi


def sys_platform_linux():
    import sys
    return sys.platform.startswith('linux')


class TSharkDecoder:
    """Session sở hữu đúng một child process; caller phải drain hoặc close."""

    protocol_tree_filter=None
    protocol_tree_filter_flag='-J'

    def __init__(self,source,*,expected_sha256,expected_size,tshark,tshark_sha256,limits=None):
        self.source=regular_file(source);self.executable=regular_file(tshark)
        self.limits=(limits or DecodeLimits()).validate()
        if type(expected_size) is not int or expected_size<24 or self.source.stat().st_size!=expected_size:
            raise DecoderError('SOURCE_SIZE_MISMATCH')
        for digest in (expected_sha256,tshark_sha256):
            if type(digest) is not str or re.fullmatch('[0-9a-f]{64}',digest) is None:
                raise DecoderError('EXPECTED_SHA256_REQUIRED')
        self.signature=file_signature(self.source)
        if file_sha256(self.source)!=expected_sha256:raise DecoderError('SOURCE_SHA256_MISMATCH')
        if file_signature(self.source)!=self.signature:raise DecoderError('SOURCE_CHANGED_DURING_HASH')
        if file_sha256(self.executable)!=tshark_sha256:raise DecoderError('EXECUTABLE_SHA256_MISMATCH')
        self.source_sha256=expected_sha256;self.executable_sha256=tshark_sha256
        self.process=None;self.status='NEW';self.stop=threading.Event();self.stdout_queue=queue.Queue(maxsize=2)
        self.stderr_tail=bytearray();self.stderr_total=0;self.stderr_digest=hashlib.sha256()
        self.reader_error=None;self.threads=[];self.command=[]
        self.stats=dict(decoded_records=0,stdout_bytes=0,peak_record_bytes=0,source_complete=False,
                        peak_python_rss_bytes=0,peak_decoder_rss_bytes=0,memory_probe_available=True)

    def __enter__(self):return self
    def __exit__(self,*_):self.close()

    def _stdout(self):
        try:
            while not self.stop.is_set():
                block=self.process.stdout.read(8192)
                while not self.stop.is_set():
                    try:self.stdout_queue.put(block,timeout=.1);break
                    except queue.Full:continue
                if not block:break
        except (OSError,ValueError) as error:
            if not self.stop.is_set():self.reader_error='STDOUT_READ_ERROR:'+type(error).__name__

    def _stderr(self):
        try:
            while not self.stop.is_set():
                block=self.process.stderr.read(4096)
                if not block:break
                self.stderr_total+=len(block);self.stderr_digest.update(block)
                self.stderr_tail.extend(block)
                del self.stderr_tail[:-self.limits.stderr_bytes]
                if self.stderr_total>self.limits.stderr_bytes:
                    self.reader_error='STDERR_BYTE_LIMIT';break
        except (OSError,ValueError) as error:
            if not self.stop.is_set():self.reader_error='STDERR_READ_ERROR:'+type(error).__name__

    def _resources(self):
        if time.monotonic()-self.started>self.limits.timeout_seconds:raise DecoderError('DECODER_TIMEOUT')
        if self.reader_error:raise DecoderError(self.reader_error)
        for name,pid,limit in [('python',os.getpid(),self.limits.python_rss_bytes),
                               ('decoder',self.process.pid,self.limits.decoder_rss_bytes)]:
            if name=='decoder' and self.process.poll() is not None:continue
            rss=process_rss(pid)
            if rss is None:self.stats['memory_probe_available']=False
            else:
                key='peak_'+name+'_rss_bytes';self.stats[key]=max(self.stats[key],rss)
                if rss>limit:raise DecoderError(name.upper()+'_RSS_ABORT_THRESHOLD')

    def records(self):
        if self.status!='NEW':raise DecoderError('DECODER_CANNOT_REPLAY')
        if file_signature(self.source)!=self.signature:raise DecoderError('SOURCE_CHANGED_BEFORE_DECODE')
        self.status='RUNNING';self.started=time.monotonic();parser=PDMLParser(self.limits)
        self.command=[str(self.executable),'-n','-l','-r',str(self.source),'-T','pdml',
                      '-c',str(self.limits.packet_count)]
        for key,value in PREFERENCES.items():self.command+=['-o',key+':'+value]
        if self.protocol_tree_filter is not None:self.command+=[self.protocol_tree_filter_flag,self.protocol_tree_filter]
        try:
            self.process=subprocess.Popen(self.command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                          stdin=subprocess.DEVNULL)
            self.threads=[threading.Thread(target=self._stdout,daemon=True),threading.Thread(target=self._stderr,daemon=True)]
            for thread in self.threads:thread.start()
            previous=0
            while True:
                self._resources()
                try:block=self.stdout_queue.get(timeout=.1)
                except queue.Empty:continue
                self.stats['stdout_bytes']+=len(block)
                for record in parser.feed(block,final=not block):
                    ordinal=record.number('frame.number')
                    if ordinal!=previous+1:raise DecoderError('SOURCE_ORDINAL_GAP_OR_DUPLICATE')
                    previous=ordinal;self.stats['decoded_records']+=1
                    yield record
                if not block:break
            try:
                self.process.wait(timeout=min(self.limits.close_seconds,max(.01,self.limits.timeout_seconds-(time.monotonic()-self.started))))
            except subprocess.TimeoutExpired as error:
                raise DecoderError('DECODER_DRAIN_TIMEOUT') from error
            for thread in self.threads:thread.join(timeout=self.limits.close_seconds)
            if any(t.is_alive() for t in self.threads):raise DecoderError('READER_CLOSE_TIMEOUT')
            self._resources()
            if self.process.returncode!=0:raise DecoderError('TSHARK_NONZERO_EXIT')
            if file_signature(self.source)!=self.signature:raise DecoderError('SOURCE_CHANGED_DURING_DECODE')
            self.stats['peak_record_bytes']=parser.peak_record_bytes
            self.status='COMPLETE_BOUNDED_PREFIX'
        except GeneratorExit:
            self.status='ABORTED';raise
        except Exception:
            self.status='FAILED';raise
        finally:
            self.stats['peak_record_bytes']=parser.peak_record_bytes
            self.stats['elapsed_seconds']=time.monotonic()-self.started
            self.close()

    def close(self):
        if self.status=='NEW':self.status='CLOSED'
        if self.status=='RUNNING':self.status='ABORTED'
        self.stop.set()
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:self.process.wait(timeout=self.limits.close_seconds)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    try:self.process.wait(timeout=self.limits.close_seconds)
                    except subprocess.TimeoutExpired as error:raise DecoderError('CHILD_CLOSE_FAILED') from error
            for pipe in (self.process.stdout,self.process.stderr):
                if pipe is not None:pipe.close()
            for thread in self.threads:thread.join(timeout=self.limits.close_seconds)
            if any(t.is_alive() for t in self.threads):raise DecoderError('READER_CLOSE_FAILED')

    def report(self):
        return dict(status=self.status,decoder_version=DECODER_VERSION,command=self.command,
                    limits=asdict(self.limits),
                    source_sha256=self.source_sha256,executable_sha256=self.executable_sha256,
                    preferences=dict(PREFERENCES),stats=dict(self.stats),
                    stderr_bytes=self.stderr_total,stderr_sha256=self.stderr_digest.hexdigest(),
                    stderr_tail=self.stderr_tail.decode('utf-8',errors='replace'),
                    child_exited=self.process is None or self.process.poll() is not None)
