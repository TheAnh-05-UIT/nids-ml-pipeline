"""Discover local data or write bounded CSV EDA reports to a new run.
Kiểm kê dữ liệu local hoặc ghi EDA CSV vào run mới.
"""
import argparse
import csv
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import platform
import re
import sqlite3
from urllib.parse import urlsplit

from nids_ml.eda.csv_eda import EdaOptions, analyze_csv
from nids_ml.data.inventory import discover
from nids_ml.validation.workspace import workspace_path, write_json


def main(argv=None, default_workspace=None):
    parser = argparse.ArgumentParser(description='Bounded CSV EDA only; no download or ML training.')
    parser.add_argument('--workspace', default=str(default_workspace or Path.cwd()))
    sub = parser.add_subparsers(dest='command', required=True)
    inventory = sub.add_parser('discover')
    inventory.add_argument('--max-entries', type=int, default=10000)
    analyze = sub.add_parser('analyze')
    analyze.add_argument('--input', required=True, help='Path relative to workspace, under data/raw')
    analyze.add_argument('--label-column', action='append', required=True)
    analyze.add_argument('--dataset-version', required=True)
    analyze.add_argument('--source-uri', required=True, help='Canonical source page, no credentials or query tokens')
    analyze.add_argument('--run-id', required=True)
    analyze.add_argument('--input-kind', choices=('dataset', 'synthetic_fixture'), default='dataset')
    analyze.add_argument('--config', default='configs/eda.json')
    args = parser.parse_args(argv)
    output = None
    try:
        workspace = Path(args.workspace).resolve()
        cfg = json.loads((workspace / 'configs/workspace.json').read_text(encoding='utf-8'))
        if cfg['schema_version'] != '1.0.0' or cfg['workspace_root'] != '.':
            raise ValueError('Unsupported workspace contract')
        if args.command == 'discover':
            report = discover(workspace, args.max_entries, workspace_path(workspace, cfg['paths']['raw_data']))
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report['files'] else 3
        for value in (args.run_id, args.dataset_version):
            if not re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,127}', value):
                raise ValueError('run-id/version must be safe nonempty identifiers')
        source = urlsplit(args.source_uri)
        if (not source.scheme or source.username or source.password or source.query or source.fragment or
                (args.input_kind == 'dataset' and (source.scheme != 'https' or not source.hostname))):
            raise ValueError('Use canonical HTTPS source URI without credentials/query/fragment')
        raw_root = workspace_path(workspace, cfg['paths']['raw_data'])
        path = workspace_path(workspace, args.input)
        if not path.is_relative_to(raw_root) or path.suffix.lower() != '.csv':
            raise ValueError('Analyze supports raw CSV only, no archive/PCAP extraction')
        eda = json.loads(workspace_path(workspace, args.config).read_text(encoding='utf-8'))
        if eda['schema_version'] != '1.0.0':
            raise ValueError('Unsupported EDA config version')
        options = EdaOptions(**eda['limits'])
        options.validate()
        runs = workspace_path(workspace, cfg['paths']['runs'])
        if runs != workspace / 'artifacts/runs':
            raise ValueError('Expected canonical artifacts/runs path')
        # Check input trước khi tạo run; source file luôn chỉ đọc.
        if path.is_symlink() or not path.is_file():
            raise ValueError('CSV input not found or not a regular file')
        candidate = runs / args.run_id
        candidate.mkdir(exist_ok=False)
        output = candidate
        report = analyze_csv(path, args.label_column, output, options, eda['encoding'])
        report['input'].update(path=path.relative_to(workspace).as_posix(),
                               input_kind=args.input_kind, dataset_id='synthetic-fixture' if args.input_kind == 'synthetic_fixture' else cfg['dataset']['id'],
                               dataset_version=args.dataset_version, source_uri=args.source_uri,
                               provenance='Operator supplied; publisher revision/source linkage requires separate verification')
        report['options'] = asdict(options)
        report['runtime'] = {'python': platform.python_version(), 'sqlite': sqlite3.sqlite_version}
        features = {key: report[key] for key in ('schema_version', 'column_count', 'label_columns',
                    'raw_feature_columns', 'selected_feature_columns', 'feature_selection', 'columns',
                    'leakage_candidates', 'feature_label_associations')}
        features['input'] = report['input']
        write_json(output / 'data_quality.json', report)
        write_json(output / 'feature_inventory.json', features)
        write_json(output / 'COMPLETE.json', {'publication_complete': True,
            'input_kind': args.input_kind, 'sha256': {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                for name in ('data_quality.json', 'feature_inventory.json')}})
        print(json.dumps({'status': 'COMPLETE_LOCAL_SCAN_NOT_TASK_ACCEPTANCE', 'input_kind': args.input_kind,
                          'run_directory': str(output), 'row_count': report['row_count'],
                          'column_count': report['column_count'], 'sha256': report['input']['sha256'],
                          'preprocessing_split_training': 'NOT RUN'}, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError, csv.Error, sqlite3.Error) as exc:
        failure = {'status': 'FAIL', 'error_type': type(exc).__name__, 'error': str(exc)[:300],
                   'publication_complete': False, 'automatic_retry_cleanup': False}
        if output is not None and output.is_dir() and not (output / 'failure.json').exists():
            write_json(output / 'failure.json', failure)
        print(json.dumps(failure))
        return 2
