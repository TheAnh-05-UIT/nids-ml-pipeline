"""Validate synthetic label rules, decisions and window aggregation.
Kiểm rule nhãn synthetic, quyết định packet và tổng hợp window.
"""
import argparse
import json

from nids_ml.labeling.fixtures import read_fixture_json
from nids_ml.labeling.resolver import LabelResolver, PacketDecision, aggregate_window
from nids_ml.labeling.contracts import (RULE_SCHEMA, ContractError, canonical_bytes,
                                      digest_bytes)


def load_fixture(path):
    data=read_fixture_json(path)
    expected={'kind','capture_fixture_text','rule_set','evidence_documents','approval_document','packets','full_window_complete'}
    if type(data) is not dict or set(data)!=expected or data['kind']!='synthetic_fixture':
        raise ContractError('A complete synthetic fixture bundle is required')
    if type(data['capture_fixture_text']) is not str or type(data['evidence_documents']) is not dict or type(data['approval_document']) is not str:
        raise ContractError('Expected synthetic document strings')
    if type(data['packets']) is not list or len(data['packets'])>4096 or type(data['full_window_complete']) is not bool:
        raise ContractError('Invalid/budget-exceeding packet fixture list')
    if any(type(v) is not str for v in data['evidence_documents'].values()):
        raise ContractError('Expected evidence document strings')
    resolver=LabelResolver(data['rule_set'],{k:v.encode('utf-8') for k,v in data['evidence_documents'].items()},
                           data['approval_document'].encode('utf-8'))
    capture=data['rule_set']['capture'];content=data['capture_fixture_text'].encode('utf-8')
    if capture['sha256']!=digest_bytes(content) or capture['size_bytes']!=len(content):
        raise ContractError('Synthetic capture fixture content mismatch')
    return data,resolver


def main(argv=None):
    parser=argparse.ArgumentParser(description='Synthetic label contract validation only; no PCAP input or real labeling.')
    sub=parser.add_subparsers(dest='command',required=True)
    schema=sub.add_parser('schema',help='Show schema or check a checked-in schema copy')
    schema.add_argument('--check')
    for name in ('validate','decide','aggregate'):
        p=sub.add_parser(name)
        p.add_argument('--fixture',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='schema':
            if args.check:
                if read_fixture_json(args.check)!=RULE_SCHEMA:
                    raise ContractError('Schema differs from the runtime contract')
                result=dict(status='PASS',schema_sha256=digest_bytes(canonical_bytes(RULE_SCHEMA)))
            else:
                result=RULE_SCHEMA
        else:
            data,resolver=load_fixture(args.fixture)
            # Validate cũng kiểm toàn bộ synthetic packet decisions, không chỉ rule syntax.
            decisions=[PacketDecision.padding() if p=={'padding':True} else resolver.resolve(p) for p in data['packets']]
            result=dict(status='PASS_SYNTHETIC_ONLY',input_kind='synthetic_fixture',
                approval_status='SYNTHETIC_TEST_ONLY',rule_set_sha256=resolver.rule_set_sha256,
                rule_count=len(data['rule_set']['rules']),packet_count=len(decisions),real_data_ready=False)
            if args.command=='decide':
                result['decisions']=[p.to_dict() for p in decisions]
            elif args.command=='aggregate':
                result['window']=aggregate_window(decisions,full_window_complete=data['full_window_complete']).to_dict()
            if args.command=='validate':
                aggregate_window(decisions,full_window_complete=data['full_window_complete'])
        print(json.dumps(result,sort_keys=True,indent=2,ensure_ascii=True,allow_nan=False))
        return 0
    except (ValueError,TypeError,KeyError,OSError,RecursionError) as error:
        print(json.dumps(dict(status='FAIL_CLOSED',error=str(error)),sort_keys=True,ensure_ascii=True))
        return 2


if __name__=='__main__':
    raise SystemExit(main())
