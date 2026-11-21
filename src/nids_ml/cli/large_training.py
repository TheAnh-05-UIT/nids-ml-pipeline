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
    scenario_catalog=commands.add_parser('scenario-catalog',help='Official test retained, validation only from train; all approved scenario sources required')
    scenario_catalog.add_argument('--dataset',nargs='+',required=True)
    scenario_catalog.add_argument('--seed',type=int,default=42);scenario_catalog.add_argument('--output',required=True)
    scenario_inspect=commands.add_parser('scenario-inspect',help='Read-only inspection of scenario catalog batches; no model training')
    scenario_inspect.add_argument('--catalog',required=True)
    temporal_catalog=commands.add_parser('scenario-temporal-catalog',help='Per-scenario future traffic; shared parent keys allowed, time embargo and exact tensor dedup required')
    temporal_catalog.add_argument('--dataset',nargs='+',required=True)
    temporal_catalog.add_argument('--seed',type=int,default=42);temporal_catalog.add_argument('--output',required=True)
    temporal_catalog.add_argument('--gap-seconds',type=int,default=30)
    temporal_inspect=commands.add_parser('scenario-temporal-inspect',help='Inspect temporal catalog and HDF5 time lineage; no model training')
    temporal_inspect.add_argument('--catalog',required=True)
    for command in (scenario_catalog,scenario_inspect,temporal_catalog,temporal_inspect):
        command.add_argument('--policy',required=True);command.add_argument('--policy-sha256',required=True)
    training=commands.add_parser('train');training.add_argument('--catalog',required=True)
    training.add_argument('--output',required=True);training.add_argument('--epochs',type=int,default=1)
    training.add_argument('--seed',type=int,default=42)
    training.add_argument('--skip-test',action='store_true',help='Train/validation only; no test tensor evaluation or test predictions')
    training.add_argument('--threshold',type=float,default=.5,help='Evaluation decision threshold; does not affect training loss')
    calibrate=commands.add_parser('calibrate',help='Choose a frozen-checkpoint threshold using validation only')
    calibration_commands=[calibrate]
    frozen=commands.add_parser('evaluate',help='Evaluate a frozen checkpoint; default split is validation')
    calibration_commands.append(frozen)
    for command in calibration_commands:
        command.add_argument('--catalog',required=True)
        command.add_argument('--checkpoint',required=True)
        command.add_argument('--output',required=True)
    calibrate.add_argument('--max-fpr',type=float,default=.01)
    frozen.add_argument('--split',choices=('validation','test'),default='validation')
    frozen.add_argument('--evaluate-test-approved',action='store_true')
    choice=frozen.add_mutually_exclusive_group()
    choice.add_argument('--threshold',type=float)
    choice.add_argument('--threshold-artifact')
    for command in (inspect,training,scenario_inspect,temporal_inspect,*calibration_commands):
        command.add_argument('--client',type=int,choices=(0,1))
        command.add_argument('--batch-size',type=int,default=64)
        command.add_argument('--source',action='append',default=[],help='Portable override: DATASET_SHA256=LOCAL_DIRECTORY')
    for command in (prepare,inspect,training,scenario_catalog,scenario_inspect,temporal_catalog,temporal_inspect,*calibration_commands):
        command.add_argument('--workspace',default=str(Path(__file__).resolve().parents[3]))
    args=parser.parse_args(argv)
    try:
        sources={}
        for value in getattr(args,'source',[]):
            key,separator,path=value.partition('=')
            if not separator or not path or key in sources:raise ValueError('INVALID_OR_DUPLICATE_SOURCE_OVERRIDE')
            sources[key]=path
        if args.command=='scenario-temporal-catalog':
            from nids_ml.data.temporal_catalog import build_temporal_catalog
            result=build_temporal_catalog(args.dataset,args.workspace,args.output,
                policy_path=args.policy,policy_sha=args.policy_sha256,seed=args.seed,gap_seconds=args.gap_seconds,
                progress=lambda stage,records:print(json.dumps(dict(stage=stage,records=records)),flush=True))
        elif args.command=='scenario-catalog':
            from nids_ml.data.scenario_catalog import build_scenario_catalog
            result=build_scenario_catalog(args.dataset,args.workspace,args.output,
                policy_path=args.policy,policy_sha=args.policy_sha256,seed=args.seed,
                progress=lambda stage,records:print(json.dumps(dict(stage=stage,records=records)),flush=True))
        elif args.command=='catalog':
            from nids_ml.data.catalog import build_catalog
            inputs=[(p,None) for p in args.dataset]+[(p,0) for p in args.local_0]+[(p,1) for p in args.local_1]
            result=build_catalog(inputs,args.workspace,args.output,seed=args.seed,
                progress=lambda stage,records:print(json.dumps(dict(stage=stage,records=records)),flush=True))
        elif args.command=='train':
            from nids_ml.training.batched import train_catalog
            result=train_catalog(args.catalog,args.workspace,args.output,client=args.client,epochs=args.epochs,
                batch_size=args.batch_size,seed=args.seed,sources=sources,
                skip_test=args.skip_test,threshold=args.threshold)
        elif args.command in ('calibrate','evaluate'):
            from nids_ml.evaluation.thresholds import calibrate_checkpoint,evaluate_checkpoint
            common=dict(client=args.client,batch_size=args.batch_size,sources=sources)
            if args.command=='calibrate':
                result=calibrate_checkpoint(args.checkpoint,args.catalog,args.workspace,args.output,
                    max_fpr=args.max_fpr,**common)
            else:
                result=evaluate_checkpoint(args.checkpoint,args.catalog,args.workspace,args.output,
                    split=args.split,threshold=.5 if args.threshold is None else args.threshold,
                    threshold_artifact=args.threshold_artifact,evaluate_test_approved=args.evaluate_test_approved,**common)
        else:
            from nids_ml.storage.partitioned import PartitionReader,ScenarioPartitionReader,TemporalScenarioPartitionReader
            if args.command in ('scenario-inspect','scenario-temporal-inspect'):
                reader_type=TemporalScenarioPartitionReader if args.command=='scenario-temporal-inspect' else ScenarioPartitionReader
                reader=reader_type(args.catalog,args.workspace,sources=sources,
                                               policy_path=args.policy,policy_sha=args.policy_sha256)
            else:reader=PartitionReader(args.catalog,args.workspace,sources=sources)
            observed={}
            for split in ('train','validation','test'):
                counts=Counter()
                for batch in reader.batches(split,args.client,batch_size=args.batch_size):
                    counts.update(str(int(y)) for y in batch['y'][:,0])
                if dict(counts)!=reader.counts(split,args.client):raise ValueError('INSPECT_OBSERVED_COUNTS')
                observed[split]=dict(counts)
            status={'scenario-inspect':'PASS_SCENARIO_CATALOG_INSPECTION','scenario-temporal-inspect':'PASS_TEMPORAL_SCENARIO_CATALOG_INSPECTION'}.get(args.command,'PASS_PARTITION_INSPECTION')
            result=dict(status=status,catalog_id=reader.manifest['catalog_id'],
                client=args.client,observed=observed,training=False,native_ARM64='NOT VERIFIED')
            if args.command=='scenario-temporal-inspect':
                result.update(scope=reader.manifest['scope'],parent_connection_independence=False,
                              gap_seconds=reader.manifest['temporal']['gap_seconds'])
        print(json.dumps(result,sort_keys=True,allow_nan=False));return 0
    except (ValueError,TypeError,KeyError,OSError,RuntimeError,ImportError,sqlite3.Error) as error:
        print(json.dumps(dict(status='FAIL_CLOSED',error=str(error),automatic_cleanup=False)));return 2


if __name__=='__main__':raise SystemExit(main())
