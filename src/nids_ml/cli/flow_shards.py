"""Run host decoding, shard export or bounded inspection as separate explicit steps.
CLI decode trên host, export trong runtime có h5py, inspect theo batch; không train/deploy.
"""
import argparse
from collections import Counter
import json
import sqlite3
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description='Mapped PCAP -> LUCID cache -> HDF5 shards. No training, Docker or SSH.')
    sub=parser.add_subparsers(dest='command',required=True)
    decode=sub.add_parser('decode',help='Host Python + TShark, continuous source-order decoding')
    decode.add_argument('--mapping',required=True); decode.add_argument('--tshark',required=True)
    decode.add_argument('--tshark-sha256',required=True)
    decode.add_argument('--decoder-profile',choices=('standard','large-memory'),default='standard',
        help='standard: 1.5 GiB TShark RSS; large-memory: Linux, 8 GiB available before start, 6 GiB RSS cap')
    export=sub.add_parser('export',help='NumPy/h5py runtime, bounded shard writer')
    export.add_argument('--cache',nargs='+',required=True)
    for cmd in (decode,export):
        cmd.add_argument('--output',required=True)
        cmd.add_argument('--workspace',default=str(Path(__file__).resolve().parents[3]))
    inspect=sub.add_parser('inspect',help='Verify all hashes/layouts and read every row in bounded batches')
    inspect.add_argument('--dataset',required=True); inspect.add_argument('--batch-size',type=int,default=64)
    diagnostic=sub.add_parser('diagnose-stream',help='PCAP -> rolling flow digest/counts; UNKNOWN only, no HDF5/training')
    diagnostic.add_argument('--pcap',required=True)
    diagnostic.add_argument('--sha256',required=True)
    diagnostic.add_argument('--size-bytes',required=True,type=int)
    diagnostic.add_argument('--expected-records',required=True,type=int)
    diagnostic.add_argument('--tshark',required=True)
    diagnostic.add_argument('--tshark-sha256',required=True)
    diagnostic.add_argument('--max-lateness-ns',required=True,type=int)
    diagnostic.add_argument('--source-split',required=True,choices=('train','test'))
    mode=diagnostic.add_mutually_exclusive_group()
    mode.add_argument('--packets',type=int,default=10000,help='Diagnostic prefix; maximum 100000, default 10000')
    mode.add_argument('--whole-capture',action='store_true',help='Explicit full EOF diagnostic; native full-size RAM feasibility not established')
    diagnostic.add_argument('--decoder-profile',choices=('standard','large-memory'),default='standard')
    diagnostic.add_argument('--output',required=True)
    diagnostic.add_argument('--preflight-only',action='store_true',help='Read/hash inputs and validate budgets; no process/output')
    args=parser.parse_args(argv)
    def progress(stage,records): print(json.dumps(dict(stage=stage,records=records)),flush=True)
    try:
        if args.command=='diagnose-stream':
            from nids_ml.flows.pcap_stream import diagnose_stream
            result=diagnose_stream(args.pcap,args.output,expected_sha256=args.sha256,
                expected_size=args.size_bytes,expected_records=args.expected_records,
                tshark=args.tshark,tshark_sha256=args.tshark_sha256,
                max_lateness_ns=args.max_lateness_ns,source_split=args.source_split,
                prefix_records=None if args.whole_capture else args.packets,
                decoder_profile=args.decoder_profile,progress=progress,preflight_only=args.preflight_only)
        elif args.command=='decode':
            from nids_ml.flows.mapped import prepare_cache
            result=prepare_cache(args.mapping,args.workspace,args.output,tshark=args.tshark,
                                 tshark_sha256=args.tshark_sha256,progress=progress,decoder_profile=args.decoder_profile)
        elif args.command=='export':
            from nids_ml.storage.shards import export_shards
            result=export_shards(args.cache,args.workspace,args.output,progress)
        else:
            from nids_ml.storage.shards import FlowShardReader
            reader=FlowShardReader(args.dataset); rows=0; labels=Counter(); families=Counter()
            for batch in reader.batches(args.batch_size,metadata=True):
                rows+=len(batch['y'])
                labels.update(str(int(y)) for y in batch['y'])
                families.update(value.decode('ascii') for value in batch['attack_type'])
            if dict(labels)!=reader.manifest['label_counts'] or dict(families)!=reader.manifest['attack_type_counts']:
                raise ValueError('OBSERVED_SHARD_COUNTS_DISAGREE_WITH_MANIFEST')
            result=dict(status='PASS_SHARD_INSPECTION',samples=rows,shape=reader.manifest['shape'],
                label_counts=reader.manifest['label_counts'],attack_type_counts=reader.manifest['attack_type_counts'],
                split=reader.manifest['split'],training_ready=False,training=False)
        print(json.dumps(result,sort_keys=True)); return 0
    except (ValueError,TypeError,KeyError,OSError,RuntimeError,ImportError,sqlite3.Error) as error:
        print(json.dumps(dict(status='FAIL_CLOSED',error=str(error),training=False))); return 2


if __name__=='__main__': raise SystemExit(main())
