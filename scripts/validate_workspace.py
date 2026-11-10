"""Check workspace structure, local schemas and lightweight package imports.
Kiểm cấu trúc workspace, schema local và import package nhẹ.
"""
import argparse
import ast
import importlib
import json
from pathlib import Path, PurePosixPath
import platform
import re
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    'nids_ml', 'nids_ml.data', 'nids_ml.preprocessing', 'nids_ml.models',
    'nids_ml.training', 'nids_ml.evaluation', 'nids_ml.adapters',
    'nids_ml.adapters.federated', 'nids_ml.adapters.swarm', 'nids_ml.continual',
    'nids_ml.eda', 'nids_ml.pcap', 'nids_ml.flows', 'nids_ml.labeling',
    'nids_ml.alignment', 'nids_ml.validation', 'nids_ml.cli',
    'nids_ml.storage', 'nids_ml.storage.hdf5',
)
SCHEMAS = ('common.schema.json', 'dataset.schema.json', 'artifact.schema.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    def unique(items):
        value = {}
        for key, item in items:
            require(key not in value, 'Duplicate JSON key: ' + key)
            value[key] = item
        return value
    def reject(value):
        raise ValueError('Non-finite JSON value')
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique, parse_constant=reject)


def relative_path(value):
    """Chỉ nhận paths nằm trong root; không tạo directories hoặc sửa symlinks."""
    require(isinstance(value, str) and value and '\\' not in value, 'Expected nonempty POSIX relative path')
    path = PurePosixPath(value)
    require(not path.is_absolute() and '..' not in path.parts and not re.match(r'^[A-Za-z]:', value),
            'Path escapes workspace: ' + value)
    resolved = (ROOT / value).resolve()
    require(resolved.is_relative_to(ROOT), 'Resolved path escapes workspace: ' + value)
    return resolved


def check_refs(schemas):
    references = []
    def walk(node):
        if isinstance(node, dict):
            if '$ref' in node:
                reference = node['$ref']
                filename, _, fragment = reference.partition('#')
                require(filename in schemas, 'Reference must resolve to a local known schema')
                target = schemas[filename]
                require(fragment.startswith('/'), 'Expected local JSON Pointer')
                for segment in fragment[1:].split('/'):
                    target = target[segment.replace('~1', '/').replace('~0', '~')]
                require(isinstance(target, dict), 'Schema reference target is not an object')
                references.append(reference)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    for schema in schemas.values():
        walk(schema)
    return len(references)


def validate(expect_empty=False):
    require(sys.version_info >= (3, 11), 'Python >=3.11 required')
    required_dirs = ('data/raw', 'data/interim', 'data/processed', 'data/splits',
                     'configs', 'contracts', 'artifacts/manifests', 'artifacts/runs',
                     'scripts', 'experiments', 'notebooks', 'tests')
    require(all((ROOT / name).is_dir() for name in required_dirs), 'Missing workspace directories')
    with (ROOT / 'pyproject.toml').open('rb') as stream:
        project = tomllib.load(stream)
    require(project['project']['name'] == 'nids-ml-pipeline', 'Wrong project name')
    require(project['project']['requires-python'] == '>=3.11', 'Unexpected Python baseline')
    require(project['build-system']['build-backend'] == 'setuptools.build_meta', 'Wrong build backend')
    require(project['tool']['setuptools']['package-dir'] == {'': 'src'}, 'Expected src layout')
    require(all((ROOT / 'src' / Path(*name.split('.')) / '__init__.py').is_file() for name in PACKAGES),
            'Missing package boundary')
    config = load_json(ROOT / 'configs/workspace.json')
    schemas = {name: load_json(ROOT / 'contracts' / name) for name in SCHEMAS}
    require(config['schema_version'] == schemas['common.schema.json']['$defs']['schemaVersion']['const'] == '1.0.0',
            'Contract version mismatch')
    require(config['workspace_root'] == '.', 'Config root must be the workspace itself')
    paths = config['paths']
    require(set(paths) == {'raw_data', 'interim_data', 'processed_data', 'splits', 'artifact_manifests', 'runs', 'experiments', 'notebooks'},
            'Config path roles mismatch')
    resolved = {name: relative_path(value) for name, value in paths.items()}
    require(all(path.is_dir() for path in resolved.values()), 'Missing configured directory')
    require(len(set(resolved.values())) == len(resolved), 'Configured storage paths overlap exactly')
    identifier = schemas['common.schema.json']['$defs']['identifier']['pattern']
    require(re.fullmatch(identifier, config['dataset']['id']) is not None, 'Invalid dataset ID')
    if config['dataset']['version'] is not None:
        require(re.fullmatch(identifier, config['dataset']['version']) is not None, 'Invalid dataset version')
    if config['dataset']['manifest_path'] is not None:
        require(relative_path(config['dataset']['manifest_path']).is_file(), 'Dataset manifest path does not exist')
    for schema in schemas.values():
        require(schema['$schema'] == 'https://json-schema.org/draft/2020-12/schema', 'Unexpected schema dialect')
    references = check_refs(schemas)
    path_pattern = schemas['common.schema.json']['$defs']['relativePath']['pattern']
    require(re.fullmatch(path_pattern, 'data/raw/dataset-v1/file.csv') is not None, 'Contract rejects relative path')
    for value in ('../escape', 'data/../escape', '/tmp/data', 'C:/dataset.csv', 'data\\file.csv'):
        require(re.fullmatch(path_pattern, value) is None, 'Contract accepts unsafe path: ' + value)
        try:
            relative_path(value)
        except ValueError:
            continue
        raise ValueError('Runtime path guard accepts unsafe path')
    sources = sorted((ROOT / 'src').rglob('*.py')) + sorted((ROOT / 'scripts').glob('*.py'))
    for path in sources:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    # Import khung package không được tạo pyc hoặc nạp framework/dataset.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / 'src'))
    for name in PACKAGES:
        module = importlib.import_module(name)
        require(Path(module.__file__).resolve().is_relative_to(ROOT / 'src'), 'Imported another package copy')
    heavy = sorted(name for name in ('tensorflow', 'torch', 'numpy', 'pandas', 'sklearn', 'h5py') if name in sys.modules)
    require(not heavy, 'Scaffold import loaded ML dependencies')
    if expect_empty:
        for base in (ROOT / 'data', ROOT / 'artifacts'):
            for path in base.rglob('*'):
                require(not path.is_symlink(), 'Unexpected symlink in initial storage')
                if path.is_file():
                    require(path.name == '.gitkeep' or path == base / 'README.md', 'Unexpected initial dataset/output file')
        require(config['dataset']['version'] is None and config['dataset']['manifest_path'] is None,
                'Initial workspace must not claim a dataset manifest/version')
    return {'status': 'PASS', 'scope': 'Workspace scaffold only', 'workspace_root': str(ROOT),
            'python': platform.python_version(), 'checks': {
                'required_directories': 'PASS', 'pyproject_src_layout': 'PASS', 'workspace_relative_paths': 'PASS',
                'contracts_JSON_and_local_refs': 'PASS', 'unsafe_path_rejection': 'PASS', 'Python_syntax': 'PASS',
                'package_import_without_ML': 'PASS', 'empty_storage': 'PASS' if expect_empty else 'NOT REQUESTED'},
            'packages_imported': len(PACKAGES), 'python_sources_parsed': len(sources),
            'local_schema_refs_checked': references, 'ML_frameworks_imported': heavy,
            'dataset_processing': 'NOT RUN', 'training': 'NOT RUN', 'model_training_tests': 'NOT RUN / NOT IMPLEMENTED',
            'EDA_fixture_tests': 'Run unittest separately; scaffold validator does not execute EDA tests',
            'full_JSON_Schema_instance_validation': 'NOT RUN by scaffold validator; dataset instances require separate contract/integrity checks'}


def main():
    parser = argparse.ArgumentParser(description='Read-only NIDS workspace scaffold validation.')
    parser.add_argument('--expect-empty', action='store_true', help='Check initial storage has only README/markers.')
    args = parser.parse_args()
    try:
        print(json.dumps(validate(args.expect_empty), indent=2))
    except (ValueError, KeyError, TypeError, OSError, SyntaxError) as exc:
        print(json.dumps({'status': 'FAIL', 'error': str(exc)}))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
