"""Run a bounded CSV/PCAP candidate-alignment audit from the CLI.
Chạy audit candidate CSV/PCAP có giới hạn qua CLI.
"""
import argparse
from collections import Counter
import datetime as dt
import json
from pathlib import Path

from nids_ml.alignment.io import csv_prefix,pcap_scalar_prefix
from nids_ml.alignment.candidates import AlignmentError,make_record,candidate_join,clock_hypothesis,time_witness_join
from nids_ml.labeling.fixtures import read_fixture_json


def workspace_source(workspace,value):
    if type(value) is not str or not value or '\\' in value:raise AlignmentError('POSIX_WORKSPACE_PATH_REQUIRED')
    relative=Path(value)
    path=workspace/relative
    if relative.is_absolute() or '..' in relative.parts or not path.resolve().is_relative_to(workspace/'data/raw'):
        raise AlignmentError('SOURCE_MUST_STAY_IN_RAW_WORKSPACE')
    return path


def run_alignment(config,workspace,scratch):
    if type(config) is not dict or set(config)!={'schema_version','kind','case_id','pcap','csv','toolchain','bounds'}:
        raise AlignmentError('ALIGNMENT_CONFIG_KEYS')
    if config['schema_version']!='1.0.0' or config['kind']!='bounded_packet_alignment_audit':raise AlignmentError('ALIGNMENT_CONFIG_VERSION')
    if config['bounds']!={'pcap_records':256,'csv_search_rows':1024,'calibration_end':64}:
        raise AlignmentError('EXACT_BOUNDED_AUDIT_PROFILE_REQUIRED')
    for name in ('pcap','csv'):
        if set(config[name])!={'path','sha256','size_bytes'}:raise AlignmentError('SOURCE_MANIFEST_KEYS')
    if set(config['toolchain'])!={'tshark_path','tshark_sha256'}:raise AlignmentError('TOOLCHAIN_KEYS')
    workspace=Path(workspace).resolve()
    pc=config['pcap'];cs=config['csv'];tools=config['toolchain']
    rows,source=csv_prefix(workspace_source(workspace,cs['path']),cs['sha256'],cs['size_bytes'],1024)
    scalars,decoder=pcap_scalar_prefix(workspace_source(workspace,pc['path']),source_sha256=pc['sha256'],
        size_bytes=pc['size_bytes'],tshark=tools['tshark_path'],tshark_sha256=tools['tshark_sha256'],scratch=scratch)
    left=[make_record(row,source_sha256=pc['sha256'],ordinal=index,kind='pcap') for index,row in enumerate(scalars,1)]
    right=[make_record(row,source_sha256=cs['sha256'],ordinal=index,kind='csv') for index,row in enumerate(rows,1)]
    joined=candidate_join(left,right)
    same=Counter()
    for p,c in zip(left,right):
        if p.key is None:same['UNSUPPORTED_OR_INVALID_PCAP']+=1
        elif c.key is not None and p.key[:7]==c.key[:7]:same['COMMON7_FIELDS_MATCH']+=1
        else:same['COMMON7_FIELDS_MISMATCH']+=1
    hypothesis=clock_hypothesis(left,right,joined)
    seq=Counter();index={row.ordinal:row for row in right}
    for item in joined['rows']:
        if item['status']=='UNIQUE_CONTENT_CANDIDATE':
            a=left[item['source_ordinal']-1];b=index[item['candidate_csv_ordinal']]
            seq['MATCH' if a.relative_seq is not None and a.relative_seq==b.relative_seq else 'MISSING_OR_MISMATCH']+=1
    labels=Counter((row.get('Attack_label','MISSING'),row.get('Attack_type','MISSING')) for row in rows)
    selected={n for n in (1,67,68,69,70,255,256) if n<=len(left)}
    samples=[dict(source_ordinal=n,pcap=scalars[n-1],same_ordinal_CSV={
        key:rows[n-1].get(key) for key in ('frame.time','ip.src_host','ip.dst_host','tcp.srcport','tcp.dstport',
            'tcp.len','tcp.flags','tcp.ack_raw','tcp.checksum','tcp.seq')},
        candidate_csv_ordinals=joined['rows'][n-1]['candidate_csv_ordinals']) for n in sorted(selected)]
    return dict(status='PASS_BOUNDED_ALIGNMENT_AUDIT',observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        case_id=config['case_id'],sources=dict(pcap=pc,csv=cs),bounds=config['bounds'],decoder=decoder,
        csv_prefix=source,alignment=joined,same_ordinal_common7=dict(same),clock_diagnostic=hypothesis,
        time_witness_alignment=time_witness_join(left,right,hypothesis),
        relative_sequence_witness=dict(seq),selected_evidence_samples=samples,
        dataset_provided_label_counts_CSV_search_prefix=[dict(Attack_label=a,Attack_type=b,count=n) for (a,b),n in sorted(labels.items())],
        verified_ground_truth='NOT VERIFIED',label_assignment_performed=False,timestamp_edits_performed=False,
        model_features_modified=False,flow_tensors_generated=False,training=False,
        note='Candidate one-to-one within declared prefixes; no claims of whole-capture or label correctness')


def main(argv=None):
    parser=argparse.ArgumentParser(description='Bounded CSV/PCAP candidate alignment, no label assignment or training.')
    parser.add_argument('--config',required=True)
    parser.add_argument('--workspace',default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument('--scratch',required=True)
    args=parser.parse_args(argv)
    try:
        result=run_alignment(read_fixture_json(args.config),args.workspace,args.scratch)
        print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False));return 0
    except (ValueError,TypeError,OSError,KeyError,RuntimeError) as error:
        print(json.dumps(dict(status='FAIL_CLOSED',error=str(error),training=False)));return 2


if __name__=='__main__':raise SystemExit(main())
