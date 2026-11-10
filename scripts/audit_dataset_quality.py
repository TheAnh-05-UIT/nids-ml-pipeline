"""Run a local CSV quality audit in a fresh output directory.
Chạy audit chất lượng CSV local trong output directory mới.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from nids_ml.validation.workspace import workspace_path, write_json
from nids_ml.eda.quality_detail import audit_quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    import re
    if not re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,127}', args.run_id):
        parser.error('Unsafe run ID')
    path = workspace_path(ROOT, args.input)
    if not path.is_relative_to(ROOT / 'data/raw') or path.suffix.lower() != '.csv':
        parser.error('Only raw CSV is supported')
    output = ROOT / 'artifacts/runs' / args.run_id
    output.mkdir(exist_ok=False)
    report = audit_quality(path)
    report['input']['path'] = path.relative_to(ROOT).as_posix()
    write_json(output / 'quality_detail.json', report)
    digest = hashlib.sha256((output / 'quality_detail.json').read_bytes()).hexdigest()
    write_json(output / 'COMPLETE.json', {'publication_complete': True,
        'sha256': {'quality_detail.json': digest}})
    print(json.dumps({'status': 'COMPLETE_QUALITY_DETAIL', 'rows': report['row_count'],
                      'directory': str(output), 'sha256': digest}, indent=2))


if __name__ == '__main__':
    main()
