"""Inspect the CNN or run a tiny synthetic learning/checkpoint sanity test.
Xem CNN hoặc chạy sanity learning/checkpoint synthetic nhỏ.
"""
import argparse
import json
from pathlib import Path

from nids_ml.labeling.contracts import strict_json
from nids_ml.models.flow_cnn import (PROFILES,backend,build_model,configuration,inspect_model,learning_sanity,
                       predict_flow,runtime_info,synthetic_batch,validate_configuration)
from nids_ml.models.flow_checkpoint import load_checkpoint,save_checkpoint


def main(argv=None):
    parser=argparse.ArgumentParser(description='Flow CNN synthetic validation only; no PCAP/dataset/training integration.')
    sub=parser.add_subparsers(dest='command',required=True)
    for cmd in ['inspect','sanity']:
        p=sub.add_parser(cmd);p.add_argument('--profile',choices=sorted(PROFILES),default='flow-cnn-n20-t10')
        p.add_argument('--config')
        if cmd=='sanity':p.add_argument('--checkpoint',required=True)
    args=parser.parse_args(argv)
    try:
        config=configuration(args.profile)
        if args.config:
            path=Path(args.config)
            if path.is_symlink() or path.suffix!='.json':raise ValueError('Use a regular model JSON config')
            with path.open('rb') as f:data=f.read(65537)
            if len(data)>65536:raise ValueError('Config exceeds 64 KiB')
            config=validate_configuration(strict_json(data))
            if config['profile']!=args.profile:raise ValueError('CLI profile/config mismatch')
        if args.command=='inspect':
            model=build_model(config)
            result=dict(status='PASS',input_kind='synthetic_model_inspection',model=inspect_model(model,config),runtime=runtime_info())
        else:
            model,metrics=learning_sanity(config)
            if not metrics['passed']:raise ValueError('Tiny synthetic learning criterion not met')
            manifest=save_checkpoint(model,config,args.checkpoint)
            restored,restored_manifest=load_checkpoint(args.checkpoint,expected_config=config)
            np,tf=backend()
            x,y=synthetic_batch(config)
            error=float(np.max(np.abs(predict_flow(model,x,config)-predict_flow(restored,x,config))))
            if error>1e-7:raise ValueError('Reload prediction mismatch')
            result=dict(status='PASS_SYNTHETIC_ONLY',input_kind='synthetic_fixture',metrics=metrics,
                checkpoint_model_hash=manifest['model_hash'],reload_prediction_max_abs_error=error,
                optimizer_restored=False,model=inspect_model(model,config),runtime=runtime_info())
        print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False))
        return 0
    except (ValueError,TypeError,KeyError,OSError,RuntimeError) as error:
        print(json.dumps(dict(status='BLOCKED_OR_INVALID',error=str(error)),sort_keys=True))
        return 2


if __name__=='__main__':
    raise SystemExit(main())
