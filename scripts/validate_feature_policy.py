"""Check a proposed feature policy against its local EDA evidence.
Kiểm feature policy đề xuất với evidence EDA local tương ứng.
"""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nids_ml.validation.workspace import workspace_path
from nids_ml.validation.contracts import read_published_report
from nids_ml.eda.feature_policy import validate_policy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',default='configs/feature-policy-ml-v5-candidates.json')
    args=parser.parse_args()
    try:
        policy=json.loads(workspace_path(ROOT,args.policy).read_text(encoding='utf-8'))
        schema=json.loads((ROOT/'contracts/feature-policy.schema.json').read_text(encoding='utf-8'))
        eda=read_published_report(ROOT/'artifacts/runs/data001-kaggle-v5-ml-20261106','data_quality.json')
        validate_policy(policy,schema,eda['raw_feature_columns'])
        if policy['dataset_sha256']!=eda['input']['sha256']:raise ValueError('Wrong dataset identity')
        print(json.dumps({'status':'PASS','policies':len(policy['policies']),'features_per_policy':len(policy['raw_feature_order']),
                          'scope':'Supported local schema keywords + exact feature coverage, labels, dataset identity',
                          'applied_to_data':False}))
        return 0
    except (ValueError,KeyError,OSError,TypeError) as exc:
        print(json.dumps({'status':'FAIL','error':str(exc)[:250]}));return 2


if __name__=='__main__':raise SystemExit(main())
