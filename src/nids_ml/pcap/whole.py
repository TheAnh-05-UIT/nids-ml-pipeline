"""Decode one complete capture sequentially under a separate whole-capture budget.
Decode tuần tự một capture đầy đủ bằng budget riêng; giữ nguyên decoder prefix cũ.
"""
from dataclasses import dataclass,replace
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
    rss_exit_grace_seconds=5.0

    def __init__(self,*args,**kwargs):
        available=require_large_memory_start()
        super().__init__(*args,**kwargs)
        self.stats['minimum_host_available_bytes']=available
        self._last_host_memory_check=None

    def _host_memory_guard(self):
        if not self.stats['memory_probe_available']:
            raise DecoderError('LARGE_MEMORY_RSS_PROBE_UNAVAILABLE')
        now=time.monotonic()
        if self._last_host_memory_check is None or now-self._last_host_memory_check>=1:
            self._last_host_memory_check=now
            available=available_memory_bytes()
            self.stats['minimum_host_available_bytes']=min(self.stats['minimum_host_available_bytes'],available)
            if available<LARGE_MEMORY_RESERVE_BYTES:
                raise DecoderError('HOST_AVAILABLE_MEMORY_BELOW_1_GIB_RESERVE')

    def _resources(self):
        super()._resources()
        self._host_memory_guard()

    def _rss_exit_wait_resources(self):
        super()._rss_exit_wait_resources()
        self._host_memory_guard()

    def report(self):
        result=super().report()
        result['memory_policy']=dict(profile='large-memory',minimum_start_available_bytes=LARGE_MEMORY_START_BYTES,
            minimum_runtime_available_bytes=LARGE_MEMORY_RESERVE_BYTES,host_probe_interval_seconds=1,
            rss_exit_grace_seconds=self.rss_exit_grace_seconds,
            sampled_abort_thresholds=True,hard_os_memory_limit=False)
        return result


@dataclass(frozen=True)
class StreamingDecodeLimits(LargeFlowDecodeLimits):
    """Explicit long-stream budget; stdout is piped, never stored as full PDML.
    Budget riêng cho stream dài; PDML đi qua pipe, không lưu toàn XML ra đĩa.
    """
    stdout_bytes:int=256*1024**3
    timeout_seconds:float=43200.0
    python_rss_bytes:int=512*1048576

    def validate(self):
        if type(self.packet_count) is not int or not 1<=self.packet_count<=30000001:
            raise DecoderError('STREAM_CAPTURE_RECORD_LIMIT')
        expected=type(self)()
        for name in self.__dataclass_fields__:
            if name!='packet_count' and (type(getattr(self,name)) is not type(getattr(expected,name))
                                        or getattr(self,name)!=getattr(expected,name)):
                raise DecoderError('UNAPPROVED_STREAM_CAPTURE_LIMIT:'+name)
        return self


@dataclass(frozen=True)
class StreamingLargeMemoryLimits(StreamingDecodeLimits):
    decoder_rss_bytes:int=6*1024**3


@dataclass(frozen=True)
class StreamingConnectTrainLimits(StreamingDecodeLimits):
    """Explicit budget for the 17-million-packet approved Connect train capture.
    Budget riêng Connect train lớn; không đổi các profile đã nghiệm thu.
    """
    decoder_rss_bytes:int=10*1024**3
    timeout_seconds:float=64800.0


class _StreamingScope:
    maximum_records=30000000
    rss_exit_grace_seconds=5.0

    def __init__(self,source,*,expected_records,prefix_records=None,**kwargs):
        if prefix_records is not None and (type(prefix_records) is not int
                or not 1<=prefix_records<=100000 or type(expected_records) is not int
                or not prefix_records<expected_records):
            raise DecoderError('STREAM_PREFIX_MUST_BE_SMALLER_THAN_CAPTURE_AND_AT_MOST_100000')
        self.prefix_records=prefix_records
        available=None
        if sys_platform_linux() and not isinstance(self,LargeMemoryFlowDecoder):
            available=available_memory_bytes()
            if available<3*1024**3:
                raise DecoderError('STREAM_STANDARD_REQUIRES_3_GIB_AVAILABLE')
        super().__init__(source,expected_records=expected_records,**kwargs)
        self._stream_host_probe=available is not None
        self._stream_last_host_check=None
        if available is not None:self.stats['minimum_host_available_bytes']=available
        if prefix_records is not None:
            self.limits=replace(self.limits,packet_count=prefix_records).validate()

    def _resources(self):
        super()._resources()
        if not self.stats['memory_probe_available']:
            raise DecoderError('STREAM_RSS_PROBE_UNAVAILABLE')
        self._stream_memory_guard()

    def _rss_exit_wait_resources(self):
        super()._rss_exit_wait_resources()
        self._stream_memory_guard()

    def _stream_memory_guard(self):
        if self._stream_host_probe:
            now=time.monotonic()
            if self._stream_last_host_check is None or now-self._stream_last_host_check>=1:
                self._stream_last_host_check=now
                available=available_memory_bytes()
                self.stats['minimum_host_available_bytes']=min(self.stats['minimum_host_available_bytes'],available)
                if available<1024**3:
                    raise DecoderError('STREAM_HOST_MEMORY_BELOW_1_GIB')

    def records(self):
        if self.prefix_records is None:
            yield from super().records()
            return
        iterator=TSharkDecoder.records(self)
        try:
            yield from iterator
            if self.stats['decoded_records']!=self.prefix_records:
                raise DecoderError('STREAM_PREFIX_COUNT_MISMATCH')
            if (file_signature(self.source)!=self.signature or file_sha256(self.source)!=self.source_sha256
                    or file_signature(self.source)!=self.signature):
                raise DecoderError('STREAM_SOURCE_CHANGED')
            if file_sha256(self.executable)!=self.executable_sha256:
                raise DecoderError('STREAM_EXECUTABLE_CHANGED')
            self.status='COMPLETE_DIAGNOSTIC_PREFIX'
            self.stats['source_complete']=False
        except GeneratorExit:
            self.status='ABORTED';self.stats['source_complete']=False;raise
        except Exception:
            self.status='FAILED';self.stats['source_complete']=False;raise
        finally:
            iterator.close()

    def report(self):
        result=super().report()
        result['capture_scope']=dict(expected_source_records=self.expected_records,
            requested_prefix_records=self.prefix_records,full_capture=self.prefix_records is None,
            eof_verified=self.stats['source_complete'])
        if 'memory_policy' not in result:
            result['memory_policy']=dict(profile='stream-standard',
                host_available_probe=self._stream_host_probe,
                minimum_start_available_bytes=3*1024**3 if self._stream_host_probe else None,
                minimum_runtime_available_bytes=1024**3 if self._stream_host_probe else None,
                sampled_abort_thresholds=True,hard_os_memory_limit=False)
        return result


class StreamingCaptureDecoder(_StreamingScope,LargeFlowDecoder):
    """One source-contiguous process for diagnostic prefixes or explicit full EOF.
    Một process liên tục: prefix diagnostic hoặc toàn capture với kiểm EOF.
    """
    limits_type=StreamingDecodeLimits


class StreamingLargeMemoryDecoder(_StreamingScope,LargeMemoryFlowDecoder):
    limits_type=StreamingLargeMemoryLimits


class StreamingConnectTrainDecoder(StreamingLargeMemoryDecoder):
    """Keep one continuous TCP state with a 14 GiB start/2 GiB runtime host reserve.
    Giữ TCP state liên tục; yêu cầu RAM host đủ và fail closed khi thiếu probe.
    """
    limits_type=StreamingConnectTrainLimits

    def __init__(self,*args,**kwargs):
        if available_memory_bytes()<14*1024**3:
            raise DecoderError('CONNECT_TRAIN_REQUIRES_14_GIB_AVAILABLE')
        super().__init__(*args,**kwargs)
        if self.stats['minimum_host_available_bytes']<14*1024**3:
            raise DecoderError('CONNECT_TRAIN_REQUIRES_14_GIB_AVAILABLE')

    def _host_memory_guard(self):
        super()._host_memory_guard()
        if self.stats['minimum_host_available_bytes']<2*1024**3:
            raise DecoderError('CONNECT_TRAIN_HOST_MEMORY_BELOW_2_GIB')

    def report(self):
        result=super().report()
        result['memory_policy'].update(profile='connect-train',minimum_start_available_bytes=14*1024**3,
            minimum_runtime_available_bytes=2*1024**3)
        return result
