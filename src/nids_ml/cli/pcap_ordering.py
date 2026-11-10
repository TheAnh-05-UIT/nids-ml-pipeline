"""Build, inspect and preview raw-PCAP processing indexes without training exports.
Tạo, kiểm và xem trước index PCAP gốc; không xuất dataset hoặc training.
"""
import argparse
import json
from pathlib import Path
import sqlite3

from nids_ml.labeling.fixtures import read_fixture_json
from nids_ml.ordering.pcap import build_pcap_view,validate_pcap_view,iter_pcap_view


def main(argv=None):
    parser=argparse.ArgumentParser(description='Raw PCAP ordering references only; no labels, HDF5, CNN or Docker.')
    commands=parser.add_subparsers(dest='command',required=True)
    build=commands.add_parser('build')
    build.add_argument('--config',required=True);build.add_argument('--output',required=True)
    build.add_argument('--workspace',default=str(Path(__file__).resolve().parents[3]))
    inspect=commands.add_parser('inspect');inspect.add_argument('--input',required=True)
    preview=commands.add_parser('preview');preview.add_argument('--input',required=True)
    preview.add_argument('--limit',type=int,default=5)
    args=parser.parse_args(argv)
    try:
        if args.command=='build':
            config=read_fixture_json(args.config)
            if (type(config) is not dict or set(config)!={'format','source'}
                    or config['format']!='nids-pcap-view-plan/1.0.0'):
                raise ValueError('PCAP_VIEW_PLAN_CONTRACT')
            result=build_pcap_view(config['source'],args.workspace,args.output,
                progress=lambda stage,count:print(json.dumps(dict(stage=stage,records=count)),flush=True))
        elif args.command=='inspect':result=validate_pcap_view(args.input)
        else:
            if not 1<=args.limit<=64:raise ValueError('PREVIEW_LIMIT_1_TO_64')
            records=iter_pcap_view(args.input);items=[]
            try:
                for record in records:
                    items.append(record.to_dict())
                    if len(items)==args.limit:break
            finally:records.close()
            result=dict(status='PCAP_REFERENCE_PREVIEW',records=items,labels_assigned=False,
                        flow_integration='NOT_ACTIVE',HDF5_export='BLOCKED')
        print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False));return 0
    except (ValueError,TypeError,KeyError,OSError,sqlite3.Error,RuntimeError) as error:
        print(json.dumps(dict(status='FAIL_CLOSED',error=str(error),automatic_cleanup=False,training=False)));return 2


if __name__=='__main__':raise SystemExit(main())
