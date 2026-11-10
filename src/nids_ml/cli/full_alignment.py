"""Audit full HTTP or versioned TCP pairs without importing training labels.
Audit toàn cặp HTTP hoặc TCP theo version; không nhập nhãn training.
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
        csv_stats=ingest_csv(db,cs,config['csv']['sha256'],budget,progress,record_factory=record_factory)
        if guarded and csv_stats['rows']!=config['expected_csv_rows']:
            raise AlignmentError('WHOLE_CSV_COUNT_NOT_CONFIRMED')
        if progress:progress('csv_complete',csv_stats['rows'])
        command=[str(executable),'-n','-l','-r',str(pc),'-c',str(limits.record_cap),'-T','fields',
            '-E','separator=|','-E','quote=d','-E','occurrence=a','-E','aggregator=;']
        for field in fields:command+=['-e',field]
        for name,value in PREFERENCES.items():command+=['-o',name+':'+value]
        stream=FieldStream(command,budget,fields=fields)
        pcap_stats=ingest_pcap(db,stream,config['pcap']['sha256'],budget,progress,record_factory=record_factory)
        if pcap_stats['records']!=config['expected_pcap_records'] or pcap_stats['records']>=limits.record_cap or not stream.complete:
            raise AlignmentError('WHOLE_CAPTURE_EOF_COUNT_NOT_CONFIRMED')
        if progress:progress('pcap_complete',pcap_stats['records'])
        matches=build_matches(db)
        if progress:progress('global_mapping_complete',pcap_stats['records'])
        for path,signature,identity in ((pc,pc_signature,config['pcap']),(cs,cs_signature,config['csv'])):
            if file_signature(path)!=signature or file_sha256(path)!=identity['sha256'] or file_signature(path)!=signature:
                raise AlignmentError('SOURCE_CHANGED_AFTER_AUDIT')
        labels=[dict(Attack_label=a,Attack_type=b,rows=n) for a,b,n in db.execute('SELECT alabel,atype,count(*) FROM c GROUP BY alabel,atype')]
        ranges=db.execute('SELECT min(ts),max(ts) FROM p').fetchone()
        duplicates={name:dict(db.execute('SELECT n,count(*) FROM '+name+' WHERE n>1 GROUP BY n').fetchall()) for name in ('pg','cg')}
        ordering=ordering_gate(pcap_stats['timestamp_regressions'],source_complete=True)
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
        save_new(output/'coverage.json',coverage)
        save_new(output/'label-ordering-contract.json',label_ordering_contract())
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
    args=parser.parse_args(argv)
    try:
        def progress(stage,records):print(json.dumps(dict(stage=stage,records=records)),flush=True)
        result=full_audit(read_fixture_json(args.config),args.workspace,args.output,progress)
        print(json.dumps(dict(status=result['status'],PCAP=result['PCAP'],CSV=result['CSV'],mapping=result['mapping'],
            resources=result['resources'],ordering_gate=result['ordering_gate'],output=str(Path(args.output).resolve())),sort_keys=True))
        return 0
    except (ValueError,TypeError,OSError,sqlite3.Error,KeyError,RuntimeError) as error:
        print(json.dumps(dict(status='FAIL_CLOSED',error=str(error),training=False)),flush=True);return 2


if __name__=='__main__':raise SystemExit(main())
