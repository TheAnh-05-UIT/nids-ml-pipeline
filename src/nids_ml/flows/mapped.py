"""Decode mapped captures to a disk-backed, ordered LUCID feature cache.
Decode PCAP liên tục rồi tạo cache trên đĩa; không cần NumPy/h5py trên Python host.
"""
from collections import Counter
from contextlib import contextmanager, closing
import csv
from dataclasses import dataclass
from itertools import groupby
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import struct

from nids_ml.alignment.full import Budget, FullAuditLimits, verified_source
from nids_ml.alignment.large import KEY_PROFILE, POLICY, binary_target
from nids_ml.eda.csv_eda import BoundedLines
from nids_ml.flows.extractor import FlowExtractor, _FlowState
from nids_ml.flows.features import CaptureContext, FeaturePacket, configuration
from nids_ml.labeling.contracts import canonical_bytes, strict_json
from nids_ml.ordering.view import _new_json
from nids_ml.pcap.decoder import regular_file, file_sha256, file_signature, map_record, DecoderError
from nids_ml.pcap.whole import LargeFlowDecoder

CACHE_FORMAT='nids-mapped-flow-cache/1.0.0'
VECTOR=struct.Struct('<10Q')
PROVENANCE='DATASET_PROVIDED_MAPPED'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


@dataclass(frozen=True)
class FlowLimits(FullAuditLimits):
    record_cap:int=3500000
    sqlite_mib:int=1536
    scratch_mib:int=2048
    cache_mib:int=16
    min_free_mib:int=1024
    timeout_seconds:int=7200
    python_rss_mib:int=384

    def validate(self):
        expected=FlowLimits()
        require(all(type(getattr(self,k)) is int and getattr(self,k)==getattr(expected,k)
                    for k in self.__dataclass_fields__), 'UNAPPROVED_FLOW_BUDGET')
        return self


def new_output(output, workspace):
    output=Path(output).absolute()
    require(not output.exists() and not output.is_symlink() and output.parent.is_dir()
            and not any(p.is_symlink() for p in output.parents)
            and output.resolve().is_relative_to(Path(workspace).resolve()/'artifacts/runs'),
            'FRESH_OUTPUT_UNDER_ARTIFACTS_RUNS_REQUIRED')
    limits=FlowLimits()
    require(shutil.disk_usage(output.parent).free>=(limits.scratch_mib+limits.min_free_mib)*1048576,
            'NEED_3_GIB_FREE_AFTER_INPUTS')
    output.mkdir()
    return output, Budget(output,limits)


def read_manifest(directory):
    directory=Path(directory).absolute()
    require(not (directory/'FAILED.json').exists(),'FAILED_ARTIFACT')
    path=regular_file(directory/'manifest.json')
    require(path.stat().st_size<=1048576,'MANIFEST_BYTE_CAP')
    return strict_json(path.read_bytes())


@contextmanager
def readonly(path):
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        db.execute('PRAGMA trusted_schema=OFF')
        db.execute('PRAGMA cache_size=-16384')
        yield db


@contextmanager
def mapping_input(directory,workspace):
    directory=Path(directory).absolute()
    path=regular_file(directory/'coverage.json')
    require(path.stat().st_size<=1048576,'MAPPING_REPORT_BYTE_CAP')
    report=strict_json(path.read_bytes())
    require(not (directory/'FAILED.json').exists() and report['status']=='COMPLETE_FULL_PAIR_AUDIT'
            and report['key_profile']==KEY_PROFILE and report['annotations']['policy']==POLICY,
            'COMPLETE_012A_MAPPING_REQUIRED')
    clock=report['mapping']['clock']
    require(clock['time_witness_enabled'] is True and clock['heldout_disagree']==0
            and clock['pcap_UTC_days']==1 and report['mapping']['witnessed_max_one_to_one'] is True,
            'APPROVED_UNIQUE_TIME_WITNESS_REQUIRED')
    require(type(report['PCAP']['records']) is int and 1<=report['PCAP']['records']<=3500000,
            'FLOW_SOURCE_RECORD_CAP')
    require(type(report['CSV']['rows']) is int and 1<=report['CSV']['rows']<=3500000,'CSV_RECORD_CAP')
    sources={kind:verified_source(identity,workspace) for kind,identity in report['sources'].items()}
    require(set(sources)=={'pcap','csv'},'SOURCE_PAIR_REQUIRED')
    index=regular_file(directory/'alignment.sqlite'); signature=file_signature(index)
    require(file_sha256(index)==report['index_sha256'],'MAPPING_INDEX_SHA256')
    with readonly(index) as db:
        # Compare imported labels to the original CSV; never trust the annotation view alone.
        # Đối chiếu nhãn với CSV thật, không tin view có thể bị thay bằng nhãn mặc định.
        old=csv.field_size_limit(1048576)
        try:
            labels=iter(db.execute('SELECT ord,alabel,atype FROM c ORDER BY ord'))
            with sources['csv'][0].open(encoding='utf-8-sig',newline='') as stream:
                lines=BoundedLines(stream,1048576); reader=csv.reader(lines,strict=True)
                header=lines.record(reader)
                require(len(set(header))==len(header)<=256 and {'Attack_label','Attack_type'}<=set(header),'CSV_LABEL_HEADER')
                a,b=header.index('Attack_label'),header.index('Attack_type'); count=0
                while True:
                    try: values=lines.record(reader)
                    except StopIteration: break
                    count+=1
                    require(count<=3500000 and len(values)==len(header),'CSV_LABEL_WIDTH_OR_CAP')
                    require(next(labels,None)==(count,values[a],values[b]),'CSV_LABEL_IMPORT_MISMATCH')
                require(count==report['CSV']['rows'] and next(labels,None) is None,'CSV_LABEL_EOF_COUNT')
        finally:
            csv.field_size_limit(old)
        yield db,report
    require(file_signature(index)==signature and file_sha256(index)==report['index_sha256'],
            'MAPPING_CHANGED_DURING_EXTRACTION')
    for kind,(source,signature) in sources.items():
        require(file_signature(source)==signature and file_sha256(source)==report['sources'][kind]['sha256']
                and file_signature(source)==signature,'RAW_SOURCE_CHANGED_DURING_EXTRACTION')


def protocol_scope(record):
    protocols=record.all_protocols
    if protocols.count('ip')==0:
        return 'OUTSIDE' if 'arp' in protocols or 'ipv6' in protocols else 'UNCERTAIN'
    if protocols.count('ip')!=1 or 'ipv6' in protocols:
        return 'UNCERTAIN'
    try:
        return 'SUPPORTED' if record.number('ip.proto') in (1,6,17) else 'OUTSIDE'
    except DecoderError:
        return 'UNCERTAIN'


def prepare_cache(mapping,workspace,output,*,tshark,tshark_sha256,progress=None):
    workspace=Path(workspace).resolve()
    tool=regular_file(tshark)
    require(file_sha256(tool)==tshark_sha256,'TSHARK_IDENTITY_MISMATCH')
    output,budget=new_output(output,workspace)
    db=None; decoder=None; previous=None
    try:
        with mapping_input(mapping,workspace) as (mapped,report):
            pcap=report['sources']['pcap']; source=workspace/pcap['path']
            # Use a digest-derived capture ID, independent of untrusted filename spelling.
            # ID theo checksum; filename không tham gia suy nhãn hoặc feature.
            case_id='capture-'+pcap['sha256'][:16]
            context=CaptureContext(case_id,pcap['sha256'],'mapped-flows','complete-capture')
            db=sqlite3.connect(output/'packets.sqlite')
            db.executescript('PRAGMA page_size=4096; PRAGMA journal_mode=DELETE; PRAGMA temp_store=FILE; '
                'PRAGMA max_page_count=393216; PRAGMA cache_size=-16384; '
                'CREATE TABLE packets(ord INTEGER PRIMARY KEY,ts INTEGER,flow TEXT,direction INTEGER,f BLOB,'
                'sha BLOB,icmp INTEGER,tokens TEXT,y INTEGER,family TEXT,cord INTEGER UNIQUE,poison INTEGER,'
                'window_id INTEGER,processing INTEGER); '
                'CREATE TABLE windows(id INTEGER PRIMARY KEY,anchor INTEGER,poison INTEGER);')
            # Pin SQLite temp files to this new output, never a global OS directory.
            quoted=db.execute('SELECT quote(?)',(str(output),)).fetchone()[0]
            previous=db.execute('PRAGMA temp_store_directory').fetchone()
            db.execute('PRAGMA temp_store_directory='+quoted)
            db.set_progress_handler(budget.sqlite_progress,20000)
            stats=Counter()
            cursor=iter(mapped.execute('SELECT p.ord,p.ts,w.status,w.cord,c.alabel,c.atype,p.kh FROM p '
                'LEFT JOIN witnessed w ON w.pord=p.ord LEFT JOIN c ON c.ord=w.cord ORDER BY p.ord'))
            decoder=LargeFlowDecoder(source,expected_records=report['PCAP']['records'],
                expected_sha256=pcap['sha256'],expected_size=pcap['size_bytes'],tshark=tool,tshark_sha256=tshark_sha256)
            with decoder:
                for record in decoder.records():
                    ordinal=record.number('frame.number'); timestamp=record.timestamp_ns()
                    row=next(cursor,None)
                    require(row is not None and row[:2]==(ordinal,timestamp),'DECODE_MAPPING_PACKET_IDENTITY')
                    require(row[2] is not None,'MISSING_MAPPING_RECEIPT')
                    require(row[2]!='UNIQUE_CONTENT_AND_TIME_CANDIDATE' or
                            type(row[3]) is int and 1<=row[3]<=report['CSV']['rows'],'INVALID_CSV_RECEIPT')
                    flow=direction=values=sha=icmp=tokens=target=family=None; poison=0
                    scope=protocol_scope(record)
                    if scope=='OUTSIDE':
                        stats['outside_ipv4_tcp_udp_icmp']+=1
                    else:
                        try:
                            require(scope=='SUPPORTED','UNCERTAIN_IP_SCOPE')
                            feature,_=map_record(record,context,return_feature=True)
                        except (ValueError,TypeError,KeyError):
                            poison=1; stats['quarantined_packets']+=1
                        else:
                            if row[2]=='UNIQUE_CONTENT_AND_TIME_CANDIDATE':
                                require(isinstance(row[6],bytes) and
                                    ((len(row[6])==23 and row[6][0]==6) or (len(row[6])==13 and row[6][0]==1))
                                    and feature.canonical_tuple[-1]==row[6][0],'MAPPING_PROTOCOL_IDENTITY')
                            flow=canonical_bytes(feature.canonical_tuple).decode()
                            direction=int(feature.forward_tuple!=feature.canonical_tuple)
                            values=VECTOR.pack(*feature.features[1:]); sha=bytes.fromhex(feature.packet_sha256)
                            icmp=feature.icmp_identifier; tokens=canonical_bytes(feature.unknown_protocol_tokens).decode()
                            target=binary_target(row[4],row[5]) if row[2]=='UNIQUE_CONTENT_AND_TIME_CANDIDATE' else None
                            family=('Fingerprinting' if row[5]=='OS_Fingerprinting' else row[5]) if target is not None else None
                            stats['feature_packets']+=1
                            if target is None: stats['unknown_packet_labels']+=1
                    cord=row[3] if row[2]=='UNIQUE_CONTENT_AND_TIME_CANDIDATE' else None
                    db.execute('INSERT INTO packets VALUES(?,?,?,?,?,?,?,?,?,?,?,?,NULL,NULL)',
                        (ordinal,timestamp,flow,direction,values,sha,icmp,tokens,target,family,cord,poison))
                    stats['input_packets']+=1
                    if ordinal%2048==0: db.commit(); budget.check()
                    if progress and ordinal%20000==0: progress('decode',ordinal)
            require(decoder.stats['source_complete'] and stats['input_packets']==report['PCAP']['records']
                    and next(cursor,None) is None,'FULL_CAPTURE_MAPPING_EOF_REQUIRED')
            db.commit(); db.execute('CREATE INDEX time_order ON packets(ts,ord)')
            anchor=None; window=-1; poison=0; batch=[]; digest=hashlib.sha256()
            for processing,(timestamp,ordinal,bad) in enumerate(db.execute('SELECT ts,ord,poison FROM packets ORDER BY ts,ord'),1):
                if anchor is None or timestamp>anchor+10000000000:
                    if anchor is not None: db.execute('INSERT INTO windows VALUES(?,?,?)',(window,anchor,poison))
                    anchor=timestamp; window+=1; poison=0
                poison|=bad; batch.append((window,processing,ordinal)); digest.update(struct.pack('<QQ',timestamp,ordinal))
                if len(batch)==2048:
                    db.executemany('UPDATE packets SET window_id=?,processing=? WHERE ord=?',batch); batch=[]; budget.check()
            if batch: db.executemany('UPDATE packets SET window_id=?,processing=? WHERE ord=?',batch)
            require(anchor is not None,'EMPTY_CAPTURE')
            db.execute('INSERT INTO windows VALUES(?,?,?)',(window,anchor,poison)); db.commit()
            db.execute('CREATE INDEX flow_order ON packets(window_id,flow,ts,ord)'); db.commit()
            budget.check(force=True)
            result=dict(format=CACHE_FORMAT,status='COMPLETE_MAPPED_FLOW_CACHE',context=context.to_dict(),
                source=report['sources'],mapping_index_sha256=report['index_sha256'],
                label_provenance=PROVENANCE,independent_ground_truth_verified=False,
                configuration=configuration(input_kind='decoded_pcap'),counts=dict(stats),windows=window+1,
                ordering='SOURCE_CONTIGUOUS_DECODE_THEN_STABLE_TIMESTAMP_ORDINAL_VIEW',
                processing_digest=digest.hexdigest(),decoder=decoder.report(),
                resources=dict(peak_scratch_bytes=budget.peak_scratch,peak_python_rss_bytes=budget.peak_python_rss),
                training=False,native_ARM64='NOT VERIFIED')
            db.set_progress_handler(None,0)
            db.execute('PRAGMA temp_store_directory='+db.execute('SELECT quote(?)',(previous[0] if previous else '',)).fetchone()[0])
            db.close(); db=None
        index=output/'packets.sqlite'
        result['index']=dict(filename=index.name,bytes=index.stat().st_size,sha256=file_sha256(index))
        _new_json(output/'manifest.json',result)
        return result
    except Exception as error:
        _new_json(output/'FAILED.json',dict(status='FAILED_RETAINED',error=str(error),automatic_cleanup=False))
        raise
    finally:
        if decoder is not None: decoder.close()
        if db is not None:
            try: db.execute('PRAGMA temp_store_directory='+db.execute('SELECT quote(?)',(previous[0] if previous else '',)).fetchone()[0])
            finally: db.close()


def iter_tensors(cache,stats):
    """One flow state in RAM; inspect every overflow packet before deciding its label.
    Chỉ giữ một flow; packet vượt n20 vẫn tham gia quyết định UNKNOWN/MIXED.
    """
    cache=Path(cache).absolute(); report=read_manifest(cache)
    require(report['format']==CACHE_FORMAT and report['status']=='COMPLETE_MAPPED_FLOW_CACHE','FLOW_CACHE_FORMAT')
    require(report['label_provenance']==PROVENANCE and report['independent_ground_truth_verified'] is False,
            'FLOW_CACHE_LABEL_PROVENANCE')
    require(report['configuration']==configuration(input_kind='decoded_pcap'),'FLOW_FEATURE_CONFIGURATION')
    identity=report['index']; require(identity['filename']=='packets.sqlite','CACHE_INDEX_NAME')
    path=regular_file(cache/'packets.sqlite'); signature=file_signature(path)
    require(path.stat().st_size==identity['bytes'] and file_sha256(path)==identity['sha256'],'FLOW_CACHE_SHA256')
    context=CaptureContext.from_dict(report['context'])
    with readonly(path) as db:
        expected=report['counts']['input_packets']
        require(db.execute('SELECT count(*) FROM packets').fetchone()[0]==expected,'CACHE_PACKET_COUNT')
        digest=hashlib.sha256(); anchor=None; window=-1; poisoned=0; seen=0; stored_poison=0
        for timestamp,ordinal,wid,processing,bad,stored_anchor,window_poison in db.execute(
                'SELECT p.ts,p.ord,p.window_id,p.processing,p.poison,w.anchor,w.poison FROM packets p '
                'JOIN windows w ON w.id=p.window_id ORDER BY p.ts,p.ord'):
            seen+=1
            require(processing==seen and 1<=ordinal<=expected and bad in (0,1),'PROCESSING_VIEW_INTEGRITY')
            if anchor is None or timestamp>anchor+10000000000:
                if anchor is not None:require(poisoned==stored_poison,'WINDOW_QUARANTINE_INTEGRITY')
                anchor=timestamp;window+=1;poisoned=0
            require(wid==window and stored_anchor==anchor,'PROCESSING_WINDOW_INTEGRITY')
            poisoned|=bad;stored_poison=window_poison;digest.update(struct.pack('<QQ',timestamp,ordinal))
        require(seen==expected and poisoned==stored_poison and window+1==report['windows']
                and db.execute('SELECT count(*) FROM windows').fetchone()[0]==window+1
                and digest.hexdigest()==report['processing_digest'],'PROCESSING_VIEW_INTEGRITY')
        cursor=db.execute('SELECT p.window_id,p.flow,p.ts,p.ord,p.direction,p.f,p.sha,p.icmp,p.tokens,p.y,p.family,'
            'p.processing,w.anchor,w.poison FROM packets p JOIN windows w ON w.id=p.window_id '
            'WHERE p.flow IS NOT NULL ORDER BY p.window_id,p.flow,p.ts,p.ord')
        for (window,flow),rows in groupby(cursor,key=lambda row:row[:2]):
            key=tuple(json.loads(flow)); state=None; labels=set(); families=set(); unknown=False; poisoned=False
            indices=[]; first=None; last=None
            for row in rows:
                _,_,timestamp,ordinal,direction,values,sha,icmp,tokens,y,family,processing,anchor,bad=row
                require(direction in (0,1) and y in (None,0,1) and len(values)==80 and len(sha)==32,'CACHE_FEATURE_RECORD')
                forward=key if not direction else (key[2],key[3],key[0],key[1],key[4])
                packet=FeaturePacket(context,ordinal,timestamp,forward,key,
                    (timestamp/1e9,*VECTOR.unpack(values)),sha.hex(),tuple(json.loads(tokens)),icmp)
                if state is None: state=_FlowState(packet,20)
                if state.add(packet,20): indices.append(processing)
                unknown|=y is None; poisoned|=bool(bad)
                if y is not None: labels.add(y); families.add(family)
                first=ordinal if first is None else min(first,ordinal); last=ordinal if last is None else max(last,ordinal)
            stats['flow_windows']+=1
            if poisoned: stats['excluded_quarantine']+=1; continue
            if len(labels)>1: stats['excluded_mixed']+=1; continue
            if unknown: stats['excluded_unknown']+=1; continue
            require(len(labels)==1 and all(isinstance(f,str) for f in families),'FLOW_LABEL_COMPLETENESS')
            engine=FlowExtractor(context,configuration(input_kind='decoded_pcap')); engine._flows[key]=state
            tensor=next(engine._flush(anchor,window,True))
            group=hashlib.sha256((context.capture_sha256+flow).encode()).hexdigest()
            metadata=dict(core=tensor.metadata,group_id=group,y=next(iter(labels)),
                attack_type=next(iter(families)) if len(families)==1 else 'MULTI_ATTACK',attack_types=sorted(families),
                label_provenance=PROVENANCE,independent_ground_truth_verified=False,
                retained_processing_indices=indices,last_processing_index=processing,
                min_source_ordinal=first,max_source_ordinal=last,
                processing_view=dict(contract=CACHE_FORMAT,index_sha256=identity['sha256'],
                    processing_digest=report['processing_digest'],
                    order='STABLE_TIMESTAMP_THEN_ORIGINAL_ORDINAL',packet_digest_order='PROCESSING_INDEX',
                    source_ordinals='PRESERVED_NONMONOTONIC',
                    core_scope='REUSED_FEATURE_NORMALIZATION_AND_WINDOW_COUNTS; NOT_LEGACY_CAPTURE_ORDER_HDF5'),
                tensor_sha256=hashlib.sha256(tensor.tensor_bytes).hexdigest())
            stats['eligible_flow_windows']+=1
            yield tensor,metadata
    require(file_signature(path)==signature and file_sha256(path)==identity['sha256'],'CACHE_CHANGED_DURING_EXPORT')
