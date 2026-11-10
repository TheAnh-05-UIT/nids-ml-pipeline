"""Match packet-header candidates and evaluate clock witnesses without using labels.
Đối chiếu header và clock witness, không dùng nhãn làm khóa.
"""
from collections import Counter,defaultdict
from dataclasses import dataclass
from decimal import Decimal,InvalidOperation
import hashlib
import ipaddress
import re

from nids_ml.labeling.contracts import canonical_bytes

KEY_VERSION='tcp-wire-common/1.0.0'
KEY_FIELDS=('ip.src','ip.dst','tcp.srcport','tcp.dstport','tcp.len','tcp.flags','tcp.ack_raw','tcp.checksum')
CSV_IP_FIELDS={'ip.src':'ip.src_host','ip.dst':'ip.dst_host'}
DAY_NS=86400000000000


class AlignmentError(ValueError):
    """Input/identity không hợp lệ; không tự chọn candidate đầu để có match."""


def numeric_token(value,bits=32):
    if type(value) is not str or not 1<=len(value)<=128 or type(bits) is not int or bits not in (16,32):
        raise AlignmentError('INVALID_NUMERIC_TOKEN')
    value=value.strip()
    try:
        if re.fullmatch(r'0[xX][0-9a-fA-F]+',value):number=int(value,16)
        else:
            decimal=Decimal(value)
            if not decimal.is_finite() or not 0<=decimal<2**bits:
                raise AlignmentError('NUMERIC_RANGE')
            if decimal!=decimal.to_integral_value():raise AlignmentError('NONINTEGER_NUMERIC_TOKEN')
            number=int(decimal)
    except (InvalidOperation,ValueError,OverflowError) as error:
        raise AlignmentError('INVALID_NUMERIC_TOKEN') from error
    if not 0<=number<2**bits:raise AlignmentError('NUMERIC_RANGE')
    return number


def csv_clock(value):
    """Chỉ lấy năm/time-of-day hiện hữu; không bịa ngày/tháng/timezone bị mất."""
    if type(value) is not str or len(value)>100:return None,None
    match=re.fullmatch(r'\s*(\d{4})\s+(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?\s*',value)
    if not match:return None,None
    year,hour,minute,second=(int(match[i]) for i in range(1,5));fraction=match[5] or ''
    if not 1970<=year<=2262 or hour>23 or minute>59 or second>59 or any(c!='0' for c in fraction[9:]):return None,None
    ns=((hour*60+minute)*60+second)*1000000000+int((fraction[:9]+'000000000')[:9])
    return year,ns


@dataclass(frozen=True)
class HeaderRecord:
    source_sha256:str
    ordinal:int
    key:tuple|None
    reason:str|None=None
    timestamp_ns:int|None=None
    year:int|None=None
    time_of_day_ns:int|None=None
    relative_seq:int|None=None


def make_record(row,*,source_sha256,ordinal,kind,ordinal_ceiling=100000):
    if kind not in ('pcap','csv') or type(row) is not dict:raise AlignmentError('INPUT_KIND')
    if type(source_sha256) is not str or not re.fullmatch('[a-f0-9]{64}',source_sha256):raise AlignmentError('SOURCE_SHA_REQUIRED')
    if type(ordinal_ceiling) is not int or ordinal_ceiling not in (100000,250000):raise AlignmentError('ORDINAL_PROFILE')
    if type(ordinal) is not int or not 1<=ordinal<=ordinal_ceiling:raise AlignmentError('ORDINAL_RANGE')
    timestamp=None;year=tod=seq=None
    if kind=='pcap':
        from nids_ml.pcap.decoder import epoch_ns
        timestamp=epoch_ns(row.get('frame.time_epoch'))
        tod=timestamp%DAY_NS
        try:protocol=numeric_token(row.get('ip.proto'))
        except AlignmentError:
            return HeaderRecord(source_sha256,ordinal,None,'MISSING_OR_MULTIPLE_IP_PROTOCOL',timestamp_ns=timestamp,time_of_day_ns=tod)
        if protocol!=6:
            return HeaderRecord(source_sha256,ordinal,None,'NON_TCP',timestamp_ns=timestamp,time_of_day_ns=tod)
    else:year,tod=csv_clock(row.get('frame.time'))
    try:
        parts=[]
        for name in KEY_FIELDS:
            key=CSV_IP_FIELDS.get(name,name) if kind=='csv' else name
            value=row.get(key)
            if name.startswith('ip.'):
                if type(value) is not str:raise AlignmentError('MISSING_IP')
                address=ipaddress.ip_address(value.strip())
                if address.version!=4:raise AlignmentError('ONLY_IPV4')
                parts.append(str(address))
            else:parts.append(numeric_token(value,32 if name=='tcp.ack_raw' else 16))
        key=tuple(parts)
    except (ValueError,TypeError) as error:
        # Dòng vẫn được giữ trong denominator/lineage, không unmatched→benign.
        return HeaderRecord(source_sha256,ordinal,None,'KEY_UNAVAILABLE:'+str(error),timestamp,year,tod)
    try:seq=numeric_token(row.get('tcp.seq'))
    except AlignmentError:pass
    return HeaderRecord(source_sha256,ordinal,key,None,timestamp,year,tod,seq)


def validate_records(records,maximum):
    if type(records) not in (list,tuple) or len(records)>maximum:raise AlignmentError('RECORD_BUDGET')
    source=None;previous=0
    for row in records:
        if not isinstance(row,HeaderRecord):raise AlignmentError('TYPED_RECORD_REQUIRED')
        if type(row.ordinal) is not int or not 1<=row.ordinal<=100000 or not re.fullmatch('[a-f0-9]{64}',row.source_sha256):
            raise AlignmentError('RECORD_IDENTITY')
        if row.ordinal<=previous:raise AlignmentError('ORDINAL_DUPLICATE_OR_REGRESSION')
        previous=row.ordinal
        if source is None:source=row.source_sha256
        elif row.source_sha256!=source:raise AlignmentError('MIXED_SOURCE_FILES')


def candidate_join(pcap,csv):
    """One-to-one trong hai prefix; global uniqueness và labels chưa được xác minh."""
    validate_records(pcap,256);validate_records(csv,1024)
    csv_index=defaultdict(list);pcap_counts=Counter(row.key for row in pcap if row.key is not None)
    for row in csv:
        if row.key is not None:csv_index[row.key].append(row.ordinal)
    counts=Counter();details=[];used=set();referenced=set();offsets=Counter()
    for row in pcap:
        candidates=csv_index.get(row.key,[]) if row.key is not None else []
        referenced.update(candidates)
        if row.key is None:status='UNSUPPORTED_OR_INVALID'
        elif not candidates:status='UNMATCHED_IN_SEARCH_PREFIX'
        elif len(candidates)!=1 or pcap_counts[row.key]!=1:status='AMBIGUOUS'
        else:status='UNIQUE_CONTENT_CANDIDATE'
        item=dict(source_ordinal=row.ordinal,status=status,candidate_csv_ordinals=list(candidates),
            pcap_key_occurrences=pcap_counts[row.key] if row.key is not None else 0,
            fingerprint_sha256=hashlib.sha256(canonical_bytes(row.key)).hexdigest() if row.key is not None else None,
            reason=row.reason,label_status='UNKNOWN',supervised_eligible=False)
        if status=='UNIQUE_CONTENT_CANDIDATE':
            item['candidate_csv_ordinal']=candidates[0]
            item['csv_minus_pcap_ordinal']=candidates[0]-row.ordinal
            offsets[item['csv_minus_pcap_ordinal']]+=1;used.add(candidates[0])
        counts[status]+=1;details.append(item)
    return dict(key_version=KEY_VERSION,key_fields=list(KEY_FIELDS),status='CANDIDATES_ONLY_NOT_ANNOTATIONS',
        source_records=len(pcap),csv_search_rows=len(csv),counts=dict(counts),rows=details,
        ordinal_offset_counts={str(k):v for k,v in sorted(offsets.items())},
        csv_search_rows_without_unique_pair=len(csv)-len(used),
        csv_search_rows_without_key_in_PCAP_prefix=len(csv)-len(referenced),
        prefix_censoring=True,label_assignment_performed=False,
        labels_used_for_matching=False,global_uniqueness='NOT VERIFIED')


def clock_hypothesis(pcap,csv,joined,*,calibration_end=64,minimum=16):
    """Hypothesis fit trên prefix cố định, witness ở phần còn lại; không normalize dữ liệu."""
    if type(calibration_end) is not int or calibration_end!=64 or type(minimum) is not int or minimum!=16:
        raise AlignmentError('FIXED_CLOCK_DIAGNOSTIC_PROFILE_REQUIRED')
    left={x.ordinal:x for x in pcap};right={x.ordinal:x for x in csv}
    observations=[]
    for item in joined['rows']:
        if item['status']!='UNIQUE_CONTENT_CANDIDATE':continue
        p=left[item['source_ordinal']];c=right[item['candidate_csv_ordinal']]
        if p.time_of_day_ns is not None and c.time_of_day_ns is not None:
            observations.append((p.ordinal,(c.time_of_day_ns-p.time_of_day_ns)%DAY_NS))
    calibration=[delta for ordinal,delta in observations if ordinal<=calibration_end]
    heldout=[delta for ordinal,delta in observations if ordinal>calibration_end]
    values=set(calibration)
    offset=next(iter(values)) if len(calibration)>=minimum and len(values)==1 else None
    return dict(status='SINGLE_OFFSET_HYPOTHESIS' if offset is not None else 'NO_UNIQUE_CALIBRATION_HYPOTHESIS',
        calibration_source_ordinals=[1,calibration_end],calibration_pairs=len(calibration),
        heldout_pairs=len(heldout),csv_minus_pcap_modulo_day_ns=offset,
        heldout_agree=sum(x==offset for x in heldout) if offset is not None else 0,
        heldout_disagree=sum(x!=offset for x in heldout) if offset is not None else 0,
        timezone='UNKNOWN',missing_calendar_fields='NOT RECONSTRUCTED',timestamp_edits_performed=False,
        labels_used=False,meaning='Observed modulo-day relationship, not verified clock conversion')


def time_witness_join(pcap,csv,hypothesis):
    """Tier diagnostic riêng: exact modulo-day witness, không nearest/sort/labels."""
    validate_records(pcap,256);validate_records(csv,1024)
    if (hypothesis.get('status')!='SINGLE_OFFSET_HYPOTHESIS'
        or hypothesis.get('calibration_pairs',0)<16 or hypothesis.get('heldout_pairs',0)<16
        or hypothesis.get('heldout_disagree')!=0 or hypothesis.get('labels_used') is not False):
        return dict(status='BLOCKED_ON_CLOCK_WITNESS',label_assignment_performed=False,rows=[])
    offset=hypothesis.get('csv_minus_pcap_modulo_day_ns')
    if type(offset) is not int or not 0<=offset<DAY_NS:raise AlignmentError('INVALID_CLOCK_HYPOTHESIS')
    right=defaultdict(list)
    for row in csv:
        if row.key is not None and row.time_of_day_ns is not None:
            right[(row.key,row.time_of_day_ns)].append(row.ordinal)
    candidates={row.ordinal:(right.get((row.key,(row.time_of_day_ns+offset)%DAY_NS),[])
        if row.key is not None and row.time_of_day_ns is not None else []) for row in pcap}
    reverse=Counter(index for indices in candidates.values() for index in indices)
    counts=Counter();details=[]
    for row in pcap:
        indices=candidates[row.ordinal]
        if row.key is None:status='UNSUPPORTED_OR_INVALID'
        elif row.time_of_day_ns is None:status='CLOCK_UNAVAILABLE'
        elif not indices:status='UNMATCHED_WITHIN_SEARCH_PREFIX'
        elif len(indices)>1 or reverse[indices[0]]>1:status='AMBIGUOUS'
        else:status='UNIQUE_CONTENT_AND_TIME_CANDIDATE'
        counts[status]+=1
        details.append(dict(source_ordinal=row.ordinal,status=status,candidate_csv_ordinals=list(indices),
                            label_status='UNKNOWN',supervised_eligible=False))
    return dict(status='TIME_WITNESSED_CANDIDATES_ONLY',key_version=KEY_VERSION,
        counts=dict(counts),rows=details,clock_conversion_verified=False,labels_used=False,
        label_assignment_performed=False,timestamp_edits_performed=False,global_uniqueness='NOT VERIFIED',
        note='Exact observed modulo-day equality; not a trusted timestamp/label conversion')
