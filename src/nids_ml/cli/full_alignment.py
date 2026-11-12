"""Audit legacy pairs and map large TCP/ICMP pairs to dataset-provided NIDS annotations.
Giữ audit cũ; profile mới mapping TCP/ICMP lớn với nhãn CSV, chưa xuất flow hoặc training.
"""
import argparse
from dataclasses import asdict, replace
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys

from nids_ml.alignment.full import (Budget,FieldStream,FullAuditLimits,build_matches,ingest_csv,
    ingest_pcap,open_index,verified_source)
from nids_ml.alignment.io import SCALAR_FIELDS, AUDIT_SCALAR_FIELDS
from nids_ml.alignment.candidates import make_record
from nids_ml.alignment.multicapture import diagnostic_record
from nids_ml.labeling.policy import label_ordering_contract,ordering_gate
from nids_ml.labeling.fixtures import read_fixture_json
from nids_ml.alignment.candidates import AlignmentError
from nids_ml.pcap.decoder import PREFERENCES,file_sha256,file_signature,regular_file,process_rss


def save_new(path,value):
    with path.open('x',encoding='utf-8') as file:json.dump(value,file,indent=2,sort_keys=True,allow_nan=False)


def validate_audit_config(config):
    """Keep legacy HTTP v1 fixed; TCP v2 declares exact PCAP and CSV counts.
    Giữ profile HTTP v1; TCP v2 phải khai báo số records của cả hai nguồn.
    """
    if type(config) is dict and config.get('schema_version')=='1.2.0':
        from nids_ml.alignment.large import validate_large_config
        return validate_large_config(config),True
    limits=FullAuditLimits().validate()
    if type(config) is not dict or config.get('kind')!='full_pair_mapping_audit':
        raise AlignmentError('FULL_AUDIT_CONFIG')
    if config.get('limits')!=asdict(limits):raise AlignmentError('UNAPPROVED_FULL_AUDIT_LIMITS')
    version=config.get('schema_version')
    if version=='1.0.0':
        if config.get('expected_pcap_records')!=229142:raise AlignmentError('UNAPPROVED_HTTP_AUDIT_PROFILE')
        return limits,False
    if version!='1.1.0' or config.get('key_profile')!='tcp-v4-fragment-guard/1.0.0':
        raise AlignmentError('UNAPPROVED_TCP_AUDIT_PROFILE')
    required={'schema_version','kind','case_id','key_profile','limits','pcap','csv','toolchain',
              'expected_pcap_records','expected_csv_rows'}
    if set(config)!=required:raise AlignmentError('FULL_AUDIT_CONFIG_KEYS')
    for key in ('expected_pcap_records','expected_csv_rows'):
        if type(config[key]) is not int or not 1<=config[key]<limits.record_cap:
            raise AlignmentError('EXACT_SOURCE_COUNT_REQUIRED:'+key)
    for key in ('case_id',):
        if type(config[key]) is not str or not 1<=len(config[key])<=128:
            raise AlignmentError('FULL_AUDIT_CASE_ID')
    return limits,True


def guarded_tcp_record(row,**kwargs):
    """Exclude fragments and non-TCP candidates without dropping denominator rows.
    Giữ mẫu số, từ chối fragment và candidate không phải TCP.
    """
    record=diagnostic_record(row,**kwargs)
    if record.key is not None and record.key[0]!='TCPv4':
        return replace(record,key=None,reason='NON_TCP_NO_APPROVED_FULL_PAIR_KEY')
    return record


def full_audit(config,workspace,output,progress=None):
    workspace=Path(workspace).resolve();output=Path(output).absolute()
    if not output.resolve().is_relative_to(workspace/'artifacts/runs') or output.exists():
        raise AlignmentError('FRESH_OUTPUT_UNDER_ARTIFACTS_RUNS_REQUIRED')
    limits,guarded=validate_audit_config(config)
    fields=AUDIT_SCALAR_FIELDS if guarded else SCALAR_FIELDS
    record_factory=guarded_tcp_record if guarded else make_record
    ingestion_options={}
    large=config.get('schema_version')=='1.2.0'
    if large and (not output.parent.is_dir() or output.is_symlink() or any(p.is_symlink() for p in output.parents)):
        raise AlignmentError('FRESH_INDEPENDENT_OUTPUT_WITH_EXISTING_PARENT_REQUIRED')
    if large:
        from functools import partial
        from nids_ml.alignment.large import large_record,compact_key_storage
        record_factory=partial(large_record,protocol=config['protocol'])
        ingestion_options['key_encoder']=compact_key_storage
    free_bytes=shutil.disk_usage(workspace).free
    if free_bytes<limits.min_free_mib*1048576+limits.scratch_mib*1048576:
        raise AlignmentError('INSUFFICIENT_DISK_FOR_DECLARED_SCRATCH_AND_HEADROOM: available_bytes='+str(free_bytes))
    pc,pc_signature=verified_source(config['pcap'],workspace)
    cs,cs_signature=verified_source(config['csv'],workspace)
    executable=regular_file(config['toolchain']['tshark_path'])
    if file_sha256(executable)!=config['toolchain']['tshark_sha256']:raise AlignmentError('TSHARK_IDENTITY_MISMATCH')
    output.mkdir(parents=False,exist_ok=False)
    budget=Budget(output,limits);db=None;stream=None;previous_temp=None
    # Chỉ process SQLite mới này dùng temp directory riêng; không sửa OS/global config.
    previous_sqlite_temp=os.environ.get('SQLITE_TMPDIR');os.environ['SQLITE_TMPDIR']=str(output)
    try:
        save_new(output/'source-manifest.json',dict(config=config,observed_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        db=open_index(output/'alignment.sqlite',budget)
        previous=db.execute('PRAGMA temp_store_directory').fetchone()
        previous_temp=previous[0] if previous else ''
        quoted=db.execute('SELECT quote(?)',(str(output.resolve()),)).fetchone()[0]
        db.execute('PRAGMA temp_store_directory='+quoted)
        check_temp=db.execute('PRAGMA temp_store_directory').fetchone()
        if not check_temp or Path(check_temp[0]).resolve()!=output.resolve():raise AlignmentError('SQLITE_TEMP_DIRECTORY_NOT_CONFIRMED')
        if progress:progress('verified_sources_and_disk',0)
        csv_stats=ingest_csv(db,cs,config['csv']['sha256'],budget,progress,record_factory=record_factory,**ingestion_options)
        if guarded and csv_stats['rows']!=config['expected_csv_rows']:
            raise AlignmentError('WHOLE_CSV_COUNT_NOT_CONFIRMED')
        if progress:progress('csv_complete',csv_stats['rows'])
        scan_count=config['expected_pcap_records']+1 if large else limits.record_cap
        command=[str(executable),'-n','-l','-r',str(pc),'-c',str(scan_count),'-T','fields',
            '-E','separator=|','-E','quote=d','-E','occurrence=a','-E','aggregator=;']
        if large:command+=['-M','50000']
        for field in fields:command+=['-e',field]
        for name,value in PREFERENCES.items():command+=['-o',name+':'+value]
        stream=FieldStream(command,budget,fields=fields,session_reset_packets=50000 if large else None)
        pcap_stats=ingest_pcap(db,stream,config['pcap']['sha256'],budget,progress,record_factory=record_factory,**ingestion_options)
        if pcap_stats['records']!=config['expected_pcap_records'] or pcap_stats['records']>=limits.record_cap or not stream.complete:
            raise AlignmentError('WHOLE_CAPTURE_EOF_COUNT_NOT_CONFIRMED')
        if progress:progress('pcap_complete',pcap_stats['records'])
        matches=build_matches(db,compact=large)
        if progress:progress('global_mapping_complete',pcap_stats['records'])
        for path,signature,identity in ((pc,pc_signature,config['pcap']),(cs,cs_signature,config['csv'])):
            if file_signature(path)!=signature or file_sha256(path)!=identity['sha256'] or file_signature(path)!=signature:
                raise AlignmentError('SOURCE_CHANGED_AFTER_AUDIT')
        labels=[dict(Attack_label=a,Attack_type=b,rows=n) for a,b,n in db.execute('SELECT alabel,atype,count(*) FROM c GROUP BY alabel,atype')]
        ranges=db.execute('SELECT min(ts),max(ts) FROM p').fetchone()
        duplicates=matches.pop('working_table_histograms') if large else {
            name:dict(db.execute('SELECT n,count(*) FROM '+name+' WHERE n>1 GROUP BY n').fetchall()) for name in ('pg','cg')}
        ordering=ordering_gate(pcap_stats['timestamp_regressions'],source_complete=True)
        annotations=None
        if large:
            from nids_ml.alignment.large import install_label_view
            annotations=install_label_view(db)
            if db.execute('PRAGMA quick_check').fetchall()!=[('ok',)]:raise AlignmentError('SQLITE_INTEGRITY')
        budget.check(force=True)
        coverage=dict(status='COMPLETE_FULL_PAIR_AUDIT',observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            scope='Complete TCP candidate audit; no verified labels or Flow tensors' if guarded else
                  'HTTP pair only; TCP header keys, no labels assigned or Flow tensors',
            key_profile=config.get('key_profile','tcp-wire-common/1.0.0'),
            sources=dict(pcap=config['pcap'],csv=config['csv']),
            CSV=csv_stats,PCAP=pcap_stats,mapping=matches,source_timestamp_range_ns=list(ranges),
            dataset_provided_CSV_labels=labels,duplicate_key_group_histograms=duplicates,
            ordering_gate=ordering,contract=label_ordering_contract(),
            resources=dict(peak_scratch_bytes=budget.peak_scratch,peak_python_rss_bytes=budget.peak_python_rss,
                peak_tshark_rss_bytes=stream.peak_child_rss,elapsed_seconds=__import__('time').monotonic()-budget.started,
                sqlite_bytes=(output/'alignment.sqlite').stat().st_size,cache_MiB=limits.cache_mib,
                memory_probe_available=process_rss(os.getpid()) is not None,
                note='Sampled abort thresholds; not hard OS limits; RSS peaks are unavailable when probe is false'),
            decoder=dict(command=command,stdout_bytes=stream.total,stderr=stream.stderr.decode('utf-8',errors='replace'),
                         child_exited=stream.process.poll() is not None),
            labels_assigned=False,flow_tensors_created=False,training=False,native_ARM64='NOT VERIFIED')
        if large:
            from nids_ml.alignment.large import mapping_contract
            coverage.update(scope='Complete '+config['protocol']+' shared-header/time mapping with dataset-provided binary NIDS labels',
                annotations=annotations,labels_assigned=annotations['mapped_packets']>0,
                contract=mapping_contract(),
                evidence_tier='DATASET_PROVIDED_MAPPED; not independently verified ground truth',
                key_limitations='ICMP shared CSV key lacks type/code/identifier; TCP key also is not full packet identity; exact global uniqueness and clock witness required',
                index_sha256=file_sha256(output/'alignment.sqlite'))
            coverage['decoder'].update(mapping_session_reset_packets=50000,
                source_ordinal='Validated global input counter; TShark frame.number is session-local',
                extraction_scope='Raw shared-header mapping only; do not apply session reset to LUCID feature extraction')
        save_new(output/'coverage.json',coverage)
        save_new(output/'label-ordering-contract.json',coverage['contract'])
        return coverage
    except Exception as error:
        save_new(output/'FAILED.json',dict(status='FAILED_RETAINED_FOR_REVIEW',error=str(error),budget_error=budget.error,
            labels_assigned=False,training=False,no_retry_no_cleanup=True))
        raise
    finally:
        if stream is not None:stream.close()
        if db is not None:
            # Khôi phục SQLite temp preference trong process; không can thiệp file cũ.
            try:
                if previous_temp is not None:
                    prior=db.execute('SELECT quote(?)',(previous_temp,)).fetchone()[0]
                    db.execute('PRAGMA temp_store_directory='+prior)
            finally:db.close()
        if previous_sqlite_temp is None:os.environ.pop('SQLITE_TMPDIR',None)
        else:os.environ['SQLITE_TMPDIR']=previous_sqlite_temp


def main(argv=None):
    parser=argparse.ArgumentParser(description='Full CSV/PCAP candidate audit; no verified labels or training.')
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--workspace',default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument('--case',help='Select one capture from the versioned multi-attack manifest')
    parser.add_argument('--tshark',help='Explicit local executable, used only with --case')
    parser.add_argument('--tshark-sha256',help='Expected executable SHA-256, used only with --case')
    parser.add_argument('--dry-run',action='store_true',help='Validate multi-attack inputs/budgets without starting TShark or creating outputs')
    args=parser.parse_args(argv)
    try:
        def progress(stage,records):print(json.dumps(dict(stage=stage,records=records)),flush=True)
        config=read_fixture_json(args.config)
        if args.case is not None:
            from nids_ml.alignment.large import select_case
            if not args.tshark or not args.tshark_sha256:raise AlignmentError('EXPLICIT_TSHARK_AND_HASH_REQUIRED')
            config=select_case(config,args.case,dict(tshark_path=args.tshark,tshark_sha256=args.tshark_sha256))
        elif args.tshark is not None or args.tshark_sha256 is not None:
            raise AlignmentError('TOOLCHAIN_OVERRIDE_REQUIRES_CASE_MANIFEST')
        if args.dry_run:
            if config.get('schema_version')!='1.2.0':raise AlignmentError('DRY_RUN_REQUIRES_LARGE_PROFILE')
            limits,_=validate_audit_config(config)
            output=Path(args.output).absolute()
            if (output.exists() or output.is_symlink() or not output.parent.is_dir()
                or any(p.is_symlink() for p in output.parents)
                or not output.resolve().is_relative_to(Path(args.workspace).resolve()/'artifacts/runs')):
                raise AlignmentError('FRESH_OUTPUT_UNDER_ARTIFACTS_RUNS_REQUIRED')
            for kind in ('pcap','csv'):verified_source(config[kind],args.workspace)
            if file_sha256(regular_file(config['toolchain']['tshark_path']))!=config['toolchain']['tshark_sha256']:
                raise AlignmentError('TSHARK_IDENTITY_MISMATCH')
            available=shutil.disk_usage(args.workspace).free
            required=(limits.min_free_mib+limits.scratch_mib)*1048576
            if available<required:raise AlignmentError('INSUFFICIENT_DISK_FOR_DECLARED_SCRATCH_AND_HEADROOM')
            print(json.dumps(dict(status='PASS_MAPPING_PREFLIGHT',case_id=config['case_id'],protocol=config['protocol'],
                expected_pcap_records=config['expected_pcap_records'],expected_csv_rows=config['expected_csv_rows'],
                disk_available_bytes=available,disk_required_bytes=required,limits=config['limits'],
                child_started=False,outputs_created=False,training=False)),flush=True)
            return 0
        result=full_audit(config,args.workspace,args.output,progress)
        if config.get('schema_version')=='1.2.0':
            print(json.dumps(dict(status=result['status'],case_id=config['case_id'],PCAP=result['PCAP'],CSV=result['CSV'],
                mapping=result['mapping'],annotations=result['annotations'],resources=result['resources'],
                index_sha256=result['index_sha256'],ordering_gate=result['ordering_gate'],output=str(Path(args.output).resolve())),sort_keys=True))
            return 0
        print(json.dumps(dict(status=result['status'],PCAP=result['PCAP'],CSV=result['CSV'],mapping=result['mapping'],
            resources=result['resources'],ordering_gate=result['ordering_gate'],output=str(Path(args.output).resolve())),sort_keys=True))
        return 0
    except (ValueError,TypeError,OSError,sqlite3.Error,KeyError,RuntimeError) as error:
        print(json.dumps(dict(status='FAIL_CLOSED',error=str(error),training=False)),flush=True);return 2


if __name__=='__main__':raise SystemExit(main())
