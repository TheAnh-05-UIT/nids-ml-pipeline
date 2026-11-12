"""Decode one complete capture sequentially under a separate whole-capture budget.
Decode tuần tự một capture đầy đủ bằng budget riêng; giữ nguyên decoder prefix cũ.
"""
from dataclasses import dataclass

from nids_ml.pcap.decoder import DecodeLimits,DecoderError,TSharkDecoder,file_sha256,file_signature,FIELD_NAMES


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
