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
    export=sub.add_parser('export',help='NumPy/h5py runtime, bounded shard writer')
    export.add_argument('--cache',nargs='+',required=True)
    for cmd in (decode,export):
        cmd.add_argument('--output',required=True)
        cmd.add_argument('--workspace',default=str(Path(__file__).resolve().parents[3]))
    inspect=sub.add_parser('inspect',help='Verify all hashes/layouts and read every row in bounded batches')
    inspect.add_argument('--dataset',required=True); inspect.add_argument('--batch-size',type=int,default=64)
    args=parser.parse_args(argv)
    def progress(stage,records): print(json.dumps(dict(stage=stage,records=records)),flush=True)
    try:
        if args.command=='decode':
            from nids_ml.flows.mapped import prepare_cache
            result=prepare_cache(args.mapping,args.workspace,args.output,tshark=args.tshark,
                                 tshark_sha256=args.tshark_sha256,progress=progress)
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
