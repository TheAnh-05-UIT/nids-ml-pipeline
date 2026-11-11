"""Spool source-ordered decoded metadata and expose hash-bound processing references.
Lưu metadata decode theo nguồn, tra theo processing index; không sửa raw hoặc gán nhãn.
"""
from collections import Counter
from contextlib import closing
from dataclasses import asdict,dataclass
import hashlib
from pathlib import Path
import re
import shutil
import sqlite3
import struct
import time

from nids_ml.flows.features import CaptureContext,ExtractionError,extract_features
from nids_ml.labeling.contracts import canonical_bytes,strict_json
from nids_ml.ordering.contracts import OrderingError,OrderingLimitError,MAX_PACKET_BYTES
from nids_ml.ordering.pcap import (FORMAT as RAW_FORMAT,COLUMNS,policy as raw_policy,
    validate_source as validate_raw_source,validate_pcap_view)
from nids_ml.ordering.view import ProcessingRecord,_fresh_directory,_new_json
from nids_ml.pcap.decoder import (DecoderError,PREFERENCES,map_record,regular_file,
    file_sha256,file_signature)
from nids_ml.pcap.whole import WholeCaptureDecoder,WholeCaptureLimits

FORMAT='nids-decoded-pcap-spool/1.0.0'
FLOW_FORMAT='nids-decoded-processing-view/1.0.0'
ROW_COLUMNS='source_ordinal,processing_index,timestamp_ns,raw_sha,packet,packet_sha,sidecar,rejection'


@dataclass(frozen=True)
class SpoolLimits:
    max_records:int=250000
    sqlite_bytes:int=512*1048576
    cache_kib:int=1024
    scratch_bytes:int=1152*1048576
    min_free_bytes:int=2*1024**3
    timeout_seconds:int=1800
    commit_every:int=256

    def validate(self):
        bounds=dict(max_records=(1,250000),sqlite_bytes=(32768,512*1048576),cache_kib=(64,8192),
            scratch_bytes=(131072,1152*1048576),min_free_bytes=(2*1024**3,4*1024**3),
            timeout_seconds=(1,1800),commit_every=(1,512))
        for name,(low,high) in bounds.items():
            value=getattr(self,name)
            if type(value) is not int or not low<=value<=high:raise OrderingLimitError('DECODED_SPOOL_LIMIT:'+name)
        if self.sqlite_bytes%4096 or self.scratch_bytes<2*self.sqlite_bytes+65536:
            raise OrderingLimitError('DECODED_SPOOL_SQLITE_OR_SCRATCH_BUDGET')
        return self


def decoded_packet_bytes(packet):
    extract_features(packet,input_kind='decoded_pcap')
    data=canonical_bytes(packet)
    if len(data)>MAX_PACKET_BYTES:raise OrderingLimitError('DECODED_PACKET_METADATA_BYTES')
    return data


def processing_digest(source):
    return hashlib.sha256(b'nids-decoded-processing-order/1\0'+canonical_bytes(source))


def _flow_source(raw_view,decoder,ledger):
    source=raw_view['source']
    return dict(input_kind='decoded_pcap',capture=source['capture'],source_first_ordinal=1,
        expected_packets=source['expected_records'],raw_source=source,
        raw_view_manifest_sha256=hashlib.sha256(canonical_bytes(raw_view)).hexdigest(),
        raw_processing_order_sha256=raw_view['summary']['processing_order_sha256'],
        decoded_source_ledger_sha256=ledger,tshark_sha256=decoder['executable_sha256'],
        decoder_contract_version='pcap-decoder/1.0.0',preferences=dict(PREFERENCES))


def validate_flow_view(view):
    keys={'format','status','source','record_count','processing_order_sha256','labels','HDF5_integration'}
    if (type(view) is not dict or set(view)!=keys or view['format']!=FLOW_FORMAT
            or view['status']!='COMPLETE_DECODED_VIEW' or view['labels']!='UNKNOWN_ONLY'
            or view['HDF5_integration']!='NOT_ACTIVE'):
        raise OrderingError('DECODED_FLOW_VIEW_CONTRACT')
    source=view['source']
    required={'input_kind','capture','source_first_ordinal','expected_packets','raw_source',
              'raw_view_manifest_sha256','raw_processing_order_sha256','decoded_source_ledger_sha256',
              'tshark_sha256','decoder_contract_version','preferences'}
    if type(source) is not dict or set(source)!=required:raise OrderingError('DECODED_FLOW_SOURCE_KEYS')
    raw=validate_raw_source(source['raw_source'])
    if (source['input_kind']!='decoded_pcap' or source['capture']!=raw['capture']
            or type(source['source_first_ordinal']) is not int or source['source_first_ordinal']!=1
            or type(view['record_count']) is not int or view['record_count']!=raw['expected_records']
            or type(source['expected_packets']) is not int or source['expected_packets']!=view['record_count']
            or source['decoder_contract_version']!='pcap-decoder/1.0.0' or source['preferences']!=PREFERENCES):
        raise OrderingError('DECODED_FLOW_SOURCE_BINDING')
    for digest in [view['processing_order_sha256']]+[source[key] for key in
            ('raw_view_manifest_sha256','raw_processing_order_sha256','decoded_source_ledger_sha256','tshark_sha256')]:
        if type(digest) is not str or not re.fullmatch('[a-f0-9]{64}',digest):
            raise OrderingError('DECODED_FLOW_VIEW_HASH')
    return strict_json(canonical_bytes(view))


def _open_db(path,limits):
    db=sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True)
    try:
        db.execute('PRAGMA query_only=ON');db.execute('PRAGMA trusted_schema=OFF')
        db.execute('PRAGMA cache_size=-'+str(limits.cache_kib))
        started=time.monotonic()
        db.set_progress_handler(lambda:int(time.monotonic()-started>limits.timeout_seconds),10000)
        if db.execute("SELECT type FROM sqlite_master WHERE name='records'").fetchone()!=('table',):
            raise OrderingError('DECODED_RECORDS_TABLE_REQUIRED')
        # Check lengths before fetching blobs; a forged huge row must not enter RAM.
        # Kiểm length trước khi fetch blob, tránh nạp row giả quá lớn vào RAM.
        if db.execute('SELECT 1 FROM records WHERE length(packet)>? OR length(sidecar)>? LIMIT 1',
                      (MAX_PACKET_BYTES,MAX_PACKET_BYTES)).fetchone():
            raise OrderingLimitError('DECODED_SPOOL_ROW_BYTES')
        return db
    except BaseException:db.close();raise


def _decode_row(row,source):
    ordinal,index,timestamp,raw_sha,data,packet_sha,side_bytes,rejection=row
    if (any(type(v) is not int for v in (ordinal,index,timestamp))
            or not 1<=ordinal<=source['expected_records'] or not 1<=index<=source['expected_records']
            or not 0<=timestamp<=2**63-1 or type(raw_sha) is not bytes or len(raw_sha)!=32
            or type(side_bytes) is not bytes or len(side_bytes)>MAX_PACKET_BYTES):
        raise OrderingError('DECODED_ROW_RANGE_OR_TYPE')
    side=strict_json(side_bytes)
    if (type(side) is not dict or side.get('source_ordinal')!=ordinal or side.get('timestamp_ns')!=timestamp
            or side.get('label')!='UNKNOWN' or side.get('supervised_eligible') is not False):
        raise OrderingError('DECODED_SIDECAR_BINDING')
    for name in ('raw_record_offset','captured_bytes','wire_bytes'):
        if type(side.get(name)) is not int:raise OrderingError('DECODED_RAW_RECORD_LENGTH_OR_OFFSET')
    if not 0<=side['captured_bytes']<=1048576 or not side['captured_bytes']<=side['wire_bytes']<=2**32-1:
        raise OrderingError('DECODED_RAW_LENGTH_RANGE')
    packet=None
    if rejection is None:
        if type(data) is not bytes or len(data)>MAX_PACKET_BYTES or type(packet_sha) is not bytes or len(packet_sha)!=32:
            raise OrderingError('DECODED_METADATA_ROW')
        packet=strict_json(data)
        if decoded_packet_bytes(packet)!=data or hashlib.sha256(data).digest()!=packet_sha:
            raise OrderingError('DECODED_PACKET_HASH_OR_CANONICAL_BYTES')
        if (packet['ordinal']!=ordinal or packet['timestamp_ns']!=timestamp or packet['capture']!=source['capture']
                or side.get('packet_sha256')!=packet_sha.hex()):
            raise OrderingError('DECODED_PACKET_SOURCE_BINDING')
    elif (type(rejection) is not str or not 1<=len(rejection)<=256 or data is not None
            or packet_sha is not None or 'packet_sha256' in side):
        raise OrderingError('REJECTED_RECORD_MUST_NOT_MASQUERADE_AS_PACKET')
    envelope=dict(source_ordinal=ordinal,processing_index=index,timestamp_ns=timestamp,
        raw_record_sha256=raw_sha.hex(),packet_sha256=packet_sha.hex() if packet_sha else None,
        sidecar_sha256=hashlib.sha256(side_bytes).hexdigest(),rejection=rejection)
    return packet,side,hashlib.sha256(b'nids-decoded-record-envelope/1\0'+canonical_bytes(envelope)).digest()


def _summarize(db,raw_view,decoder):
    source=raw_view['source'];count=eligible=regressions=max_regression=moved=0
    reasons=Counter();digest=hashlib.sha256(b'nids-decoded-source-ledger/1\0');offset=24;previous=None
    for row in db.execute('SELECT '+ROW_COLUMNS+' FROM records ORDER BY source_ordinal'):
        packet,side,identity=_decode_row(row,source);ordinal,index,timestamp=row[:3];count+=1
        if ordinal!=count or side['raw_record_offset']!=offset:raise OrderingError('DECODED_SOURCE_GAP_OR_OFFSET')
        offset+=16+side['captured_bytes'];moved+=ordinal!=index
        if previous is not None and timestamp<previous:
            regressions+=1;max_regression=max(max_regression,previous-timestamp)
        previous=timestamp;digest.update(identity)
        if packet is not None:eligible+=1
        else:
            if row[-1] not in reasons and len(reasons)>=64:raise OrderingLimitError('DECODED_REASON_CATEGORY_LIMIT')
            reasons[row[-1]]+=1
    if count!=source['expected_records'] or offset!=source['size_bytes']:
        raise OrderingError('DECODED_SOURCE_COUNT_OR_SIZE')
    if (regressions!=raw_view['summary']['source_timestamp_regressions']
            or max_regression!=raw_view['summary']['max_regression_ns']
            or moved!=raw_view['summary']['moved_records']):
        raise OrderingError('DECODED_RAW_VIEW_ORDER_STATS')
    summary=dict(records=count,eligible=eligible,rejected=count-eligible,rejection_counts=dict(reasons),
                 source_ledger_sha256=digest.hexdigest())
    flow_source=_flow_source(raw_view,decoder,digest.hexdigest());processing=processing_digest(flow_source)
    context=CaptureContext.from_dict(source['capture']);previous=None;number=0
    for ordinal,index,timestamp,packet_sha in db.execute(
            'SELECT source_ordinal,processing_index,timestamp_ns,packet_sha FROM records ORDER BY processing_index'):
        number+=1;key=(timestamp,ordinal)
        if index!=number or previous is not None and key<=previous:raise OrderingError('DECODED_PROCESSING_GAP_OR_REGRESSION')
        previous=key
        if eligible==count:
            reference=ProcessingRecord(index,ordinal,timestamp,packet_sha.hex(),context)
            processing.update(canonical_bytes(reference.to_dict())+b'\n')
    if number!=count:raise OrderingError('DECODED_PROCESSING_COUNT')
    view=None
    if eligible==count:
        view=validate_flow_view(dict(format=FLOW_FORMAT,status='COMPLETE_DECODED_VIEW',source=flow_source,
            record_count=count,processing_order_sha256=processing.hexdigest(),labels='UNKNOWN_ONLY',HDF5_integration='NOT_ACTIVE'))
    return summary,view


def prepare_decoded_spool(raw_view_directory,workspace,output,*,tshark,tshark_sha256,limits=None,progress=None):
    raw_view=validate_pcap_view(raw_view_directory);source=raw_view['source']
    limits=(SpoolLimits() if limits is None else limits).validate()
    if source['expected_records']>limits.max_records:raise OrderingLimitError('DECODED_EXPECTED_COUNT_BUDGET')
    workspace=Path(workspace).resolve();output=_fresh_directory(output)
    if (not output.resolve().is_relative_to(workspace/'artifacts/runs')
            or output.resolve().is_relative_to(Path(raw_view_directory).resolve())):
        raise OrderingError('DECODED_FRESH_INDEPENDENT_OUTPUT_UNDER_ARTIFACTS_RUNS_REQUIRED')
    if shutil.disk_usage(output.parent).free<limits.min_free_bytes+limits.scratch_bytes:
        raise OrderingLimitError('DECODED_DISK_RESERVE_PLUS_SCRATCH')
    source_path=regular_file(workspace/source['path'])
    decoder=WholeCaptureDecoder(source_path,expected_records=source['expected_records'],
        expected_sha256=source['capture']['capture_sha256'],expected_size=source['size_bytes'],
        tshark=tshark,tshark_sha256=tshark_sha256)
    raw_index=regular_file(Path(raw_view_directory)/'index.sqlite');raw_signature=file_signature(raw_index)
    output.mkdir();_new_json(output/'plan.json',dict(format=FORMAT,source=source,limits=asdict(limits),
        raw_view_manifest_sha256=hashlib.sha256(canonical_bytes(raw_view)).hexdigest(),tshark_sha256=tshark_sha256,
        rejection_policy='FAIL_CLOSED',training=False))
    started=time.monotonic();db=None;raw_db=None;received=0

    def budget():
        if time.monotonic()-started>limits.timeout_seconds:raise OrderingLimitError('DECODED_SPOOL_TIMEOUT')
        if sum(p.stat().st_size for p in output.iterdir() if p.is_file())>limits.scratch_bytes:
            raise OrderingLimitError('DECODED_SPOOL_SCRATCH_LIMIT')
        if shutil.disk_usage(output).free<limits.min_free_bytes:raise OrderingLimitError('DECODED_DISK_RESERVE')

    try:
        db=sqlite3.connect(output/'metadata.sqlite')
        db.execute('PRAGMA page_size=4096');db.execute('PRAGMA max_page_count='+str(limits.sqlite_bytes//4096))
        db.execute('PRAGMA cache_size=-'+str(limits.cache_kib));db.execute('PRAGMA journal_mode=DELETE')
        db.execute('PRAGMA temp_store=MEMORY')
        db.set_progress_handler(lambda:int(time.monotonic()-started>limits.timeout_seconds),10000)
        db.execute('CREATE TABLE records(source_ordinal INTEGER PRIMARY KEY,processing_index INTEGER NOT NULL UNIQUE,'
            'timestamp_ns INTEGER NOT NULL,raw_sha BLOB NOT NULL,packet BLOB,packet_sha BLOB,sidecar BLOB NOT NULL,rejection TEXT)')
        raw_db=sqlite3.connect(raw_index.as_uri()+'?mode=ro&immutable=1',uri=True)
        raw_db.execute('PRAGMA query_only=ON');raw_db.execute('PRAGMA trusted_schema=OFF')
        raw_db.execute('PRAGMA cache_size=-'+str(limits.cache_kib))
        raw_rows=iter(raw_db.execute('SELECT '+COLUMNS+',processing_index FROM packets ORDER BY source_ordinal'))
        context=CaptureContext.from_dict(source['capture'])
        with source_path.open('rb') as raw_stream,decoder:
            header=raw_stream.read(24)
            formats={b'\xd4\xc3\xb2\xa1':('<',1000),b'\xa1\xb2\xc3\xd4':('>',1000),
                     b'\x4d\x3c\xb2\xa1':('<',1),b'\xa1\xb2\x3c\x4d':('>',1)}
            if header[:4] not in formats:raise OrderingError('DECODED_CLASSIC_PCAP_REQUIRED')
            endian,multiplier=formats[header[:4]]
            for record in decoder.records():
                received+=1;raw_row=next(raw_rows,None)
                if raw_row is None:raise OrderingError('DECODED_EXTRA_RECORD')
                timestamp,ordinal,offset,caplen,wirelen,raw_sha,index=raw_row
                if (ordinal!=received or record.number('frame.number')!=ordinal or record.timestamp_ns()!=timestamp
                        or record.number('frame.cap_len')!=caplen or record.number('frame.len')!=wirelen
                        or raw_stream.tell()!=offset):
                    raise OrderingError('DECODED_FRAME_VS_RAW_REFERENCE')
                raw_header=raw_stream.read(16);payload=raw_stream.read(caplen)
                if len(payload)!=caplen or hashlib.sha256(raw_header+payload).digest()!=raw_sha:
                    raise OrderingError('DECODED_RAW_RECORD_HASH')
                seconds,fraction,actual_cap,actual_wire=struct.unpack(endian+'IIII',raw_header)
                if (seconds*1000000000+fraction*multiplier,actual_cap,actual_wire)!=(timestamp,caplen,wirelen):
                    raise OrderingError('DECODED_RAW_HEADER_VS_INDEX')
                packet_data=packet_sha=None;rejection=None
                try:
                    packet,side=map_record(record,context);packet_data=decoded_packet_bytes(packet)
                    packet_sha=hashlib.sha256(packet_data).digest()
                except (DecoderError,ExtractionError) as error:
                    rejection=str(error)
                    if not 1<=len(rejection)<=256:raise OrderingError('DECODED_REJECTION_REASON_LIMIT')
                    side=dict(source_ordinal=ordinal,timestamp_ns=timestamp,captured_bytes=caplen,
                              wire_bytes=wirelen,label='UNKNOWN',supervised_eligible=False)
                side['raw_record_offset']=offset;side_bytes=canonical_bytes(side)
                row=(ordinal,index,timestamp,raw_sha,packet_data,packet_sha,side_bytes,rejection)
                _decode_row(row,source);db.execute('INSERT INTO records VALUES(?,?,?,?,?,?,?,?)',row)
                if received%limits.commit_every==0:db.commit();budget()
                if progress and received%10000==0:progress('decoded_source_records',received)
            if next(raw_rows,None) is not None or raw_stream.read(1):raise OrderingError('DECODED_SOURCE_NOT_EXHAUSTED')
        if not decoder.stats['source_complete']:raise OrderingError('DECODED_EOF_NOT_CONFIRMED')
        db.commit();decoder_report=decoder.report();summary,view=_summarize(db,raw_view,decoder_report)
        budget();db.close();db=None
        if (file_signature(raw_index)!=raw_signature or file_sha256(raw_index)!=raw_view['index']['sha256']):
            raise OrderingError('RAW_VIEW_CHANGED_DURING_DECODE')
        if (file_signature(source_path)!=decoder.signature or file_sha256(source_path)!=decoder.source_sha256
                or file_signature(source_path)!=decoder.signature):
            raise OrderingError('RAW_SOURCE_CHANGED_BEFORE_SPOOL_PUBLICATION')
        path=output/'metadata.sqlite'
        if path.stat().st_size>limits.sqlite_bytes:raise OrderingLimitError('DECODED_SQLITE_SIZE')
        manifest=dict(format=FORMAT,status='COMPLETE_DECODED_SPOOL',raw_view=raw_view,limits=asdict(limits),
            decoder=decoder_report,summary=summary,flow_view=view,
            sqlite=dict(filename='metadata.sqlite',bytes=path.stat().st_size,sha256=file_sha256(path)),
            rejection_policy='FAIL_CLOSED',labels='UNKNOWN_ONLY',HDF5_export='BLOCKED',training=False)
        _new_json(output/'manifest.json',manifest)
        return manifest
    except Exception as error:
        _new_json(output/'FAILED.json',dict(status='FAILED_RETAINED',error=str(error),received_records=received,
                  decoder=decoder.report(),automatic_cleanup=False,training=False))
        raise
    finally:
        decoder.close()
        if db is not None:db.close()
        if raw_db is not None:raw_db.close()


def validate_decoded_spool(output):
    output=Path(output).absolute()
    if (output/'FAILED.json').exists():raise OrderingError('FAILED_DECODED_SPOOL')
    path=regular_file(output/'manifest.json')
    if path.stat().st_size>1048576:raise OrderingLimitError('DECODED_MANIFEST_BYTES')
    manifest=strict_json(path.read_bytes())
    keys={'format','status','raw_view','limits','decoder','summary','flow_view','sqlite',
          'rejection_policy','labels','HDF5_export','training'}
    if (type(manifest) is not dict or set(manifest)!=keys or manifest['format']!=FORMAT
            or manifest['status']!='COMPLETE_DECODED_SPOOL' or manifest['rejection_policy']!='FAIL_CLOSED'
            or manifest['labels']!='UNKNOWN_ONLY' or manifest['HDF5_export']!='BLOCKED' or manifest['training'] is not False):
        raise OrderingError('DECODED_SPOOL_MANIFEST')
    raw=manifest['raw_view'];source=validate_raw_source(raw['source']);limits=SpoolLimits(**manifest['limits']).validate()
    if (raw['format']!=RAW_FORMAT or raw['status']!='COMPLETE_PCAP_RECORD_VIEW'
            or canonical_bytes(raw['policy'])!=canonical_bytes(raw_policy())
            or raw['summary']['record_count']!=source['expected_records'] or source['expected_records']>limits.max_records):
        raise OrderingError('DECODED_RAW_VIEW_CONTRACT')
    decoder=manifest['decoder']
    if (decoder['status']!='COMPLETE_WHOLE_CAPTURE' or decoder['source_sha256']!=source['capture']['capture_sha256']
            or decoder['stats']['source_complete'] is not True or decoder['stats']['decoded_records']!=source['expected_records']
            or decoder['child_exited'] is not True or decoder['preferences']!=PREFERENCES):
        raise OrderingError('DECODED_WHOLE_CAPTURE_NOT_PROVEN')
    command=decoder.get('command')
    count_flag=command.index('-c') if type(command) is list and '-c' in command else -1
    if (decoder['decoder_version']!='pcap-decoder/1.0.0'
            or decoder['limits']!=asdict(WholeCaptureLimits(packet_count=source['expected_records']+1).validate())
            or type(decoder['executable_sha256']) is not str
            or not re.fullmatch('[a-f0-9]{64}',decoder['executable_sha256'])
            or count_flag<0 or count_flag+1>=len(command)
            or command[count_flag+1]!=str(source['expected_records']+1)):
        raise OrderingError('DECODED_DECODER_BUDGET_OR_EOF_COMMAND')
    identity=manifest['sqlite']
    if (type(identity) is not dict or set(identity)!={'filename','bytes','sha256'}
            or identity['filename']!='metadata.sqlite' or type(identity['bytes']) is not int
            or not 1<=identity['bytes']<=limits.sqlite_bytes or type(identity['sha256']) is not str
            or not re.fullmatch('[a-f0-9]{64}',identity['sha256'])):
        raise OrderingError('DECODED_SQLITE_MANIFEST')
    path=regular_file(output/'metadata.sqlite');signature=file_signature(path)
    if path.stat().st_size!=identity['bytes'] or file_sha256(path)!=identity['sha256']:
        raise OrderingError('DECODED_SQLITE_HASH_OR_SIZE')
    with closing(_open_db(path,limits)) as db:summary,view=_summarize(db,raw,decoder)
    if (summary!=manifest['summary'] or view!=manifest['flow_view'] or file_signature(path)!=signature):
        raise OrderingError('DECODED_SEMANTIC_LEDGER_OR_VIEW_MISMATCH')
    return manifest


class DecodedPacketSpool:
    def __init__(self,output):
        self.manifest=validate_decoded_spool(output);self.path=regular_file(Path(output)/'metadata.sqlite')
        self.signature=file_signature(self.path);self.db=_open_db(self.path,SpoolLimits(**self.manifest['limits']))
        self.closed=False

    def __enter__(self):return self
    def __exit__(self,*_):self.close()

    def flow_view(self):
        if self.closed:raise OrderingError('DECODED_SPOOL_CLOSED')
        if self.manifest['summary']['rejected']:
            raise OrderingError('FLOW_BLOCKED_ON_REJECTED_RECORDS; inspect rejection_counts; no implicit filtering')
        return validate_flow_view(self.manifest['flow_view'])

    def lookup(self,ordinal):
        if self.closed or type(ordinal) is not int:raise OrderingError('OPEN_DECODED_SPOOL_AND_INTEGER_ORDINAL_REQUIRED')
        if file_signature(self.path)!=self.signature:raise OrderingError('DECODED_SPOOL_CHANGED')
        row=self.db.execute('SELECT '+ROW_COLUMNS+' FROM records WHERE source_ordinal=?',(ordinal,)).fetchone()
        if row is None:raise OrderingError('DECODED_ORDINAL_NOT_FOUND')
        packet,_,_=_decode_row(row,self.manifest['raw_view']['source'])
        if packet is None:raise OrderingError('DECODED_RECORD_REJECTED')
        return packet

    def references(self):
        view=self.flow_view();digest=processing_digest(view['source']);context=CaptureContext.from_dict(view['source']['capture'])
        count=0
        for ordinal,index,timestamp,packet_sha in self.db.execute(
                'SELECT source_ordinal,processing_index,timestamp_ns,packet_sha FROM records ORDER BY processing_index'):
            count+=1;reference=ProcessingRecord(index,ordinal,timestamp,packet_sha.hex(),context)
            digest.update(canonical_bytes(reference.to_dict())+b'\n');yield reference
        if count!=view['record_count'] or digest.hexdigest()!=view['processing_order_sha256']:
            raise OrderingError('DECODED_REFERENCE_COMPLETION_MISMATCH')
        if file_signature(self.path)!=self.signature:raise OrderingError('DECODED_SPOOL_CHANGED')

    def close(self):
        if not self.closed:
            self.closed=True;self.db.close()
            if (file_signature(self.path)!=self.signature
                    or file_sha256(self.path)!=self.manifest['sqlite']['sha256']):
                raise OrderingError('DECODED_SPOOL_CHANGED_DURING_READ')
