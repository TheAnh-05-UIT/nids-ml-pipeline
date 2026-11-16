"""Decode one complete capture sequentially under a separate whole-capture budget.
Decode tuần tự một capture đầy đủ bằng budget riêng; giữ nguyên decoder prefix cũ.
"""
from dataclasses import dataclass
import time

from nids_ml.pcap.decoder import (DecodeLimits,DecoderError,TSharkDecoder,file_sha256,
                                 file_signature,FIELD_NAMES,sys_platform_linux)

LARGE_MEMORY_START_BYTES=8*1024**3
LARGE_MEMORY_RESERVE_BYTES=1024**3


def available_memory_bytes():
    """Read Linux MemAvailable; reject unsupported or missing memory probes.
    Đọc RAM khả dụng của host Linux; không đoán RAM khi thiếu thông tin.
    """
    if not sys_platform_linux():
        raise DecoderError('LARGE_MEMORY_PROFILE_REQUIRES_LINUX_MEMORY_PROBE')
    try:
        with open('/proc/meminfo',encoding='ascii') as stream:
            data=stream.read(65537)
        lines=[line.split() for line in data.splitlines() if line.startswith('MemAvailable:')]
        if len(data)>65536 or len(lines)!=1 or len(lines[0])!=3 or \
                not lines[0][1].isascii() or not lines[0][1].isdigit() or lines[0][2]!='kB':
            raise ValueError('Invalid MemAvailable')
        return int(lines[0][1])*1024
    except (OSError,UnicodeError,ValueError) as error:
        raise DecoderError('LARGE_MEMORY_HOST_PROBE_UNAVAILABLE') from error


def require_large_memory_start():
    available=available_memory_bytes()
    if available<LARGE_MEMORY_START_BYTES:
        raise DecoderError('LARGE_MEMORY_PROFILE_REQUIRES_8_GIB_AVAILABLE')
    return available


@dataclass(frozen=True)
class WholeCaptureLimits(DecodeLimits):
    stdout_bytes:int=4*1024**3
    timeout_seconds:float=900.0
    python_rss_bytes:int=256*1048576

    def validate(self):
        if type(self.packet_count) is not int or not 2<=self.packet_count<=250001:
            raise DecoderError('WHOLE_CAPTURE_RECORD_LIMIT')
        # Only the declared whole-capture profile expands the legacy prefix limits.
        # Chỉ profile whole-capture tường minh mới dùng budget khác prefix cũ.
        expected=WholeCaptureLimits()
        for name in self.__dataclass_fields__:
            if name=='packet_count':continue
            actual=getattr(self,name)
            if type(actual) is not type(getattr(expected,name)) or actual!=getattr(expected,name):
                raise DecoderError('UNAPPROVED_WHOLE_CAPTURE_LIMIT:'+name)
        return self


class WholeCaptureDecoder(TSharkDecoder):
    maximum_records=250000
    limits_type=WholeCaptureLimits

    def __init__(self,source,*,expected_records,**kwargs):
        if type(expected_records) is not int or not 1<=expected_records<=self.maximum_records:
            raise DecoderError('WHOLE_CAPTURE_EXPECTED_COUNT')
        if 'limits' in kwargs:raise DecoderError('WHOLE_CAPTURE_FIXED_LIMITS')
        self.expected_records=expected_records
        # One extra slot distinguishes actual EOF from a count-limited prefix.
        # Dư một slot để phân biệt EOF thật với việc cắt đúng packet count.
        super().__init__(source,limits=self.limits_type(packet_count=expected_records+1),**kwargs)

    def records(self):
        iterator=super().records()
        try:
            for record in iterator:
                if self.stats['decoded_records']>self.expected_records:
                    raise DecoderError('WHOLE_CAPTURE_EXTRA_RECORD')
                yield record
            if self.stats['decoded_records']!=self.expected_records:
                raise DecoderError('WHOLE_CAPTURE_MISSING_RECORD')
            if (file_signature(self.source)!=self.signature or file_sha256(self.source)!=self.source_sha256
                    or file_signature(self.source)!=self.signature):
                raise DecoderError('WHOLE_CAPTURE_SOURCE_CHANGED')
            if file_sha256(self.executable)!=self.executable_sha256:
                raise DecoderError('WHOLE_CAPTURE_EXECUTABLE_CHANGED')
            self.status='COMPLETE_WHOLE_CAPTURE';self.stats['source_complete']=True
        except GeneratorExit:
            iterator.close()
            self.status='ABORTED';self.stats['source_complete']=False;raise
        except Exception:
            iterator.close()
            self.status='FAILED';self.stats['source_complete']=False;raise
        finally:
            iterator.close()


@dataclass(frozen=True)
class LargeFlowDecodeLimits(WholeCaptureLimits):
    """Continuous decoding keeps TCP state; no session resets or packet filtering.
    Decode liên tục giữ TCP state, budget riêng không nới profile prefix/pilot cũ.
    """
    stdout_bytes:int=32*1024**3
    timeout_seconds:float=7200.0
    python_rss_bytes:int=384*1048576
    decoder_rss_bytes:int=1536*1048576

    def validate(self):
        if type(self.packet_count) is not int or not 2<=self.packet_count<=3500001:
            raise DecoderError('LARGE_FLOW_RECORD_LIMIT')
        expected=LargeFlowDecodeLimits()
        for name in self.__dataclass_fields__:
            if name!='packet_count' and (type(getattr(self,name)) is not type(getattr(expected,name))
                                        or getattr(self,name)!=getattr(expected,name)):
                raise DecoderError('UNAPPROVED_LARGE_FLOW_LIMIT:'+name)
        return self


class LargeFlowDecoder(WholeCaptureDecoder):
    """Filter PDML output trees, never filter input packets or reset TCP state.
    Chỉ giảm cây PDML xuất ra; giữ mọi ordinal và preferences của decoder đã kiểm.
    """
    maximum_records=3500000
    limits_type=LargeFlowDecodeLimits
    protocol_tree_filter_flag='-j'
    protocol_tree_filter=' '.join(['frame','ip','tcp','udp','icmp','_ws.malformed']+
        sorted(FIELD_NAMES-{'num','timestamp','len','caplen'}))


@dataclass(frozen=True)
class LargeMemoryFlowDecodeLimits(LargeFlowDecodeLimits):
    """Opt-in 6 GiB child RSS profile; all other fixed limits stay unchanged.
    Profile RAM lớn phải chọn rõ; không nới timeout, parser hoặc packet count.
    """
    decoder_rss_bytes:int=6*1024**3

    def validate(self):
        if type(self.packet_count) is not int or not 2<=self.packet_count<=3500001:
            raise DecoderError('LARGE_FLOW_RECORD_LIMIT')
        expected=LargeMemoryFlowDecodeLimits()
        for name in self.__dataclass_fields__:
            if name!='packet_count' and (type(getattr(self,name)) is not type(getattr(expected,name))
                                        or getattr(self,name)!=getattr(expected,name)):
                raise DecoderError('UNAPPROVED_LARGE_MEMORY_FLOW_LIMIT:'+name)
        return self


class LargeMemoryFlowDecoder(LargeFlowDecoder):
    """Keep continuous TCP state with bounded RSS and a sampled host RAM reserve.
    Giữ TCP state liên tục; dừng child của mình khi vượt RSS hoặc host thiếu RAM.
    """
    limits_type=LargeMemoryFlowDecodeLimits

    def __init__(self,*args,**kwargs):
        available=require_large_memory_start()
        super().__init__(*args,**kwargs)
        self.stats['minimum_host_available_bytes']=available
        self._last_host_memory_check=None

    def _resources(self):
        super()._resources()
        if not self.stats['memory_probe_available']:
            raise DecoderError('LARGE_MEMORY_RSS_PROBE_UNAVAILABLE')
        now=time.monotonic()
        if self._last_host_memory_check is None or now-self._last_host_memory_check>=1:
            self._last_host_memory_check=now
            available=available_memory_bytes()
            self.stats['minimum_host_available_bytes']=min(self.stats['minimum_host_available_bytes'],available)
            if available<LARGE_MEMORY_RESERVE_BYTES:
                raise DecoderError('HOST_AVAILABLE_MEMORY_BELOW_1_GIB_RESERVE')

    def report(self):
        result=super().report()
        result['memory_policy']=dict(profile='large-memory',minimum_start_available_bytes=LARGE_MEMORY_START_BYTES,
            minimum_runtime_available_bytes=LARGE_MEMORY_RESERVE_BYTES,host_probe_interval_seconds=1,
            sampled_abort_thresholds=True,hard_os_memory_limit=False)
        return result
