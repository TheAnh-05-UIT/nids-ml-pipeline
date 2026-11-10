"""Build, inspect or preview synthetic packet processing views.
Tạo, kiểm hoặc xem trước processing view packet synthetic.
"""
import argparse
import json
from pathlib import Path

from nids_ml.labeling.fixtures import read_fixture_json
from nids_ml.ordering.contracts import OrderingError, validate_source
from nids_ml.ordering.view import build_processing_view, iter_processing_view, validate_processing_view


def read_fixture(path):
    path = Path(path).absolute()
    workspace = Path(__file__).resolve().parents[3]
    if path.resolve().is_relative_to((workspace/'data/raw').resolve()):
        raise OrderingError('RAW_DATA_NOT_ALLOWED_IN_SYNTHETIC_CLI')
    value = read_fixture_json(path)
    if type(value) is not dict or set(value) != {'kind','source','packets'}:
        raise OrderingError('SYNTHETIC_FIXTURE_KEYS')
    if value['kind']!='synthetic_fixture' or type(value['packets']) is not list or not 1<=len(value['packets'])<=1024:
        raise OrderingError('BOUNDED_SYNTHETIC_FIXTURE_REQUIRED')
    validate_source(value['source'])
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description='Synthetic stable packet ordering view; no PCAP, tensors or training.')
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('build-synthetic')
    create.add_argument('--fixture', required=True)
    create.add_argument('--output', required=True)
    inspect = sub.add_parser('inspect')
    inspect.add_argument('--input', required=True)
    preview = sub.add_parser('preview')
    preview.add_argument('--input', required=True)
    preview.add_argument('--limit', type=int, default=10)
    args = parser.parse_args(argv)
    try:
        if args.command=='build-synthetic':
            workspace = Path(__file__).resolve().parents[3]
            output = Path(args.output).absolute()
            if not output.resolve().is_relative_to(workspace/'artifacts/runs'):
                raise OrderingError('CLI_OUTPUT_MUST_STAY_UNDER_WORKSPACE_ARTIFACTS_RUNS')
            fixture = read_fixture(args.fixture)
            result = build_processing_view(iter(fixture['packets']), fixture['source'], output)
        elif args.command=='inspect':
            result = validate_processing_view(args.input)
        else:
            if not 1<=args.limit<=64:
                raise OrderingError('PREVIEW_LIMIT_1_TO_64')
            records = iter_processing_view(args.input)
            try:
                items = []
                for item in records:
                    items.append(item.to_dict())
                    if len(items)==args.limit:
                        break
            finally:
                records.close()
            result = dict(status='SYNTHETIC_PREVIEW', records=items,
                          preview_complete=False, real_data_export='BLOCKED')
        print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
        print(json.dumps(dict(status='FAIL_CLOSED', error=str(error), automatic_cleanup=False)))
        return 2


if __name__=='__main__': raise SystemExit(main())
