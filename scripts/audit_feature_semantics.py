"""Audit feature semantics or dataset overlap without network access.
Audit semantics feature hoặc overlap dataset, không dùng mạng.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nids_ml.validation.workspace import workspace_path,write_json
from nids_ml.eda.semantics import audit_semantics,audit_label_integrity
from nids_ml.eda.overlap import audit_overlap


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True);parser.add_argument('--reference',required=True)
    parser.add_argument('--run-id',required=True);parser.add_argument('--focus',nargs='+')
    parser.add_argument('--overlap',action='store_true')
    parser.add_argument('--labels-and-nonfinite',action='store_true')
    args=parser.parse_args()
    import re
    if not re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,127}',args.run_id):parser.error('Unsafe run ID')
    paths=[workspace_path(ROOT,p) for p in (args.input,args.reference)]
    if any(not p.is_relative_to(ROOT/'data/raw') or p.suffix.lower()!='.csv' for p in paths):parser.error('Only raw CSV allowed')
    out=ROOT/'artifacts/runs'/args.run_id;out.mkdir(exist_ok=False)
    try:
        with paths[1].open(encoding='utf-8-sig',newline='') as f:header=next(csv.reader(f))
        if args.overlap and args.labels_and_nonfinite:raise ValueError('Select one audit mode')
        report=(audit_overlap(paths[1],paths[0],header,out) if args.overlap else audit_label_integrity(paths[0],header)
                if args.labels_and_nonfinite else audit_semantics(paths[0],header,args.focus))
        report['input_path']=paths[0].relative_to(ROOT).as_posix()
        write_json(out/'audit.json',report)
        write_json(out/'COMPLETE.json',{'publication_complete':True,'sha256':{'audit.json':hashlib.sha256((out/'audit.json').read_bytes()).hexdigest()}})
        print(json.dumps({'status':'AUDIT_COMPLETE','run':args.run_id,'rows':report.get('row_count',report.get('candidate_rows'))}))
        return 0
    except Exception as exc:
        write_json(out/'failure.json',{'status':'FAIL','error_type':type(exc).__name__,'error':str(exc)[:200],'publication_complete':False})
        print(json.dumps({'status':'FAIL','error':str(exc)[:200]}));return 2


if __name__=='__main__':raise SystemExit(main())
