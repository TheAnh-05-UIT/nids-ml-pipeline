"""CLI for global catalogs, read-only inspection and local batch training.
CLI tạo catalog/split, kiểm dữ liệu và train cục bộ theo batch; không Docker hoặc network.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3


def main(argv=None):
    parser=argparse.ArgumentParser(description='NIDS dedup -> connection split -> batch CPU training. No Docker/SSH/Swarm runtime.')
    commands=parser.add_subparsers(dest='command',required=True)
    prepare=commands.add_parser('catalog')
    prepare.add_argument('--dataset',nargs='+',default=[],help='Shared sources available on both clients')
    prepare.add_argument('--local-0',nargs='+',default=[]);prepare.add_argument('--local-1',nargs='+',default=[])
    prepare.add_argument('--seed',type=int,default=42);prepare.add_argument('--output',required=True)
    inspect=commands.add_parser('inspect');inspect.add_argument('--catalog',required=True)
    training=commands.add_parser('train');training.add_argument('--catalog',required=True)
    training.add_argument('--output',required=True);training.add_argument('--epochs',type=int,default=1)
    training.add_argument('--seed',type=int,default=42)
    for command in (inspect,training):
        command.add_argument('--client',type=int,choices=(0,1))
        command.add_argument('--batch-size',type=int,default=64)
        command.add_argument('--source',action='append',default=[],help='Portable override: DATASET_SHA256=LOCAL_DIRECTORY')
    for command in (prepare,inspect,training):
        command.add_argument('--workspace',default=str(Path(__file__).resolve().parents[3]))
    args=parser.parse_args(argv)
    try:
        sources={}
        for value in getattr(args,'source',[]):
            key,separator,path=value.partition('=')
            if not separator or not path or key in sources:raise ValueError('INVALID_OR_DUPLICATE_SOURCE_OVERRIDE')
            sources[key]=path
        if args.command=='catalog':
            from nids_ml.data.catalog import build_catalog
            inputs=[(p,None) for p in args.dataset]+[(p,0) for p in args.local_0]+[(p,1) for p in args.local_1]
            result=build_catalog(inputs,args.workspace,args.output,seed=args.seed,
                progress=lambda stage,records:print(json.dumps(dict(stage=stage,records=records)),flush=True))
        elif args.command=='train':
            from nids_ml.training.batched import train_catalog
            result=train_catalog(args.catalog,args.workspace,args.output,client=args.client,epochs=args.epochs,
                batch_size=args.batch_size,seed=args.seed,sources=sources)
        else:
            from nids_ml.storage.partitioned import PartitionReader
            reader=PartitionReader(args.catalog,args.workspace,sources=sources);observed={}
            for split in ('train','validation','test'):
                counts=Counter()
                for batch in reader.batches(split,args.client,batch_size=args.batch_size):
                    counts.update(str(int(y)) for y in batch['y'][:,0])
                if dict(counts)!=reader.counts(split,args.client):raise ValueError('INSPECT_OBSERVED_COUNTS')
                observed[split]=dict(counts)
            result=dict(status='PASS_PARTITION_INSPECTION',catalog_id=reader.manifest['catalog_id'],
                client=args.client,observed=observed,training=False,native_ARM64='NOT VERIFIED')
        print(json.dumps(result,sort_keys=True,allow_nan=False));return 0
    except (ValueError,TypeError,KeyError,OSError,RuntimeError,ImportError,sqlite3.Error) as error:
        print(json.dumps(dict(status='FAIL_CLOSED',error=str(error),automatic_cleanup=False)));return 2


if __name__=='__main__':raise SystemExit(main())
