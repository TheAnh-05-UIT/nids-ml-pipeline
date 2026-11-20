"""Validate synthetic packet fixtures and capture-order flow extraction.
Kiểm fixture packet synthetic và trích flow theo capture order.
"""
import argparse
import hashlib
import json
import sqlite3

from nids_ml.flows.features import CaptureContext,PACKET_SCHEMA,PROFILES,configuration,validate_configuration,extract_features
from nids_ml.flows.extractor import FlowExtractor
from nids_ml.labeling.fixtures import read_fixture_json


def main(argv=None):
    parser=argparse.ArgumentParser(description='Synthetic packet-to-flow validation only; no PCAP input/training.')
    sub=parser.add_subparsers(dest='command',required=True)
    schema=sub.add_parser('schema');schema.add_argument('--check')
    validate=sub.add_parser('validate');validate.add_argument('--fixture',required=True)
    validate.add_argument('--profile',choices=sorted(PROFILES),default='flow-cnn-n20-t10')
    validate.add_argument('--config')
    stream=sub.add_parser('validate-stream',help='Synthetic-only rolling-window core; no real PCAP labels/export')
    stream.add_argument('--fixture',required=True)
    stream.add_argument('--scratch',required=True,help='New scratch directory; never overwritten')
    stream.add_argument('--max-lateness-ns',required=True,type=int)
    stream.set_defaults(profile='flow-cnn-n20-t10',config=None)
    args=parser.parse_args(argv)
    try:
        if args.command=='schema':
            if args.check:
                if read_fixture_json(args.check)!=PACKET_SCHEMA:raise ValueError('Packet schema differs from runtime')
                output=dict(status='PASS_SCHEMA')
            else:output=PACKET_SCHEMA
        else:
            fixture=read_fixture_json(args.fixture)
            fields={'kind','capture_fixture_text','capture','input_complete','packets'}
            if type(fixture) is not dict or set(fixture)!=fields or fixture['kind']!='synthetic_fixture':
                raise ValueError('Synthetic fixture envelope required')
            if type(fixture['packets']) is not list or len(fixture['packets'])>128 or type(fixture['capture_fixture_text']) is not str:
                raise ValueError('Fixture packet/string budget invalid')
            ctx=CaptureContext.from_dict(fixture['capture'])
            if hashlib.sha256(fixture['capture_fixture_text'].encode()).hexdigest()!=ctx.capture_sha256:
                raise ValueError('Synthetic capture identity mismatch')
            config=configuration(args.profile)
            if args.config:
                config=validate_configuration(read_fixture_json(args.config))
                if config['profile']!=args.profile:raise ValueError('Profile/config mismatch')
            if args.command=='validate-stream':
                from nids_ml.flows.streaming import WindowSpool
                extractor=WindowSpool(ctx,args.scratch,max_lateness_ns=args.max_lateness_ns,
                                      input_kind='synthetic_fixture')
                events=[event.to_dict() for event in extractor.run(map(extract_features,fixture['packets']),
                                                                  input_complete=fixture['input_complete'])]
                output=dict(status='PASS_SYNTHETIC_STREAM_ONLY',events=events,
                            stats=dict(extractor.stats,status=extractor.status),
                            real_PCAP_integration='NOT IMPLEMENTED IN STREAMING CORE',
                            real_labels='BLOCKED',training=False)
            else:
                extractor=FlowExtractor(ctx,config)
                events=[event.to_dict() for event in extractor.run(fixture['packets'],input_complete=fixture['input_complete'])]
                output=dict(status='PASS_SYNTHETIC_ONLY',events=events,stats=extractor.stats,
                            real_PCAP_decoding='NOT IMPLEMENTED',real_labels='BLOCKED',training=False)
        print(json.dumps(output,sort_keys=True,indent=2,allow_nan=False))
        return 0
    except (ValueError,TypeError,OSError,KeyError,RecursionError,sqlite3.Error) as error:
        print(json.dumps(dict(status='FAIL_CLOSED',error=str(error)),sort_keys=True))
        return 2


if __name__=='__main__':raise SystemExit(main())
