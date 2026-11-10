"""Inventory local raw files and compute streaming SHA-256 fingerprints.
Kiểm kê raw files local và tính fingerprint SHA-256 theo luồng.
"""
import hashlib
import os
from pathlib import Path


def fingerprint(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Input must be a regular file, not a symlink')
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('Input changed while hashing')
    return {'size_bytes': after.st_size, 'sha256': digest.hexdigest()}


def discover(workspace, max_entries=10000, raw_root=None):
    """Chỉ tìm trong data/raw, giới hạn entries để không quét thư mục vô hạn."""
    workspace = Path(workspace).resolve()
    raw = Path(raw_root) if raw_root is not None else workspace / 'data/raw'
    if raw.is_symlink() or not raw.is_dir() or not raw.resolve().is_relative_to(workspace):
        raise ValueError('Expected independent data/raw directory')
    if not 1 <= max_entries <= 100000:
        raise ValueError('max_entries must be 1..100000')
    pending, files, seen = [raw], [], 0
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                seen += 1
                if seen > max_entries:
                    raise ValueError('Inventory entry limit exceeded; no complete inventory claimed')
                if entry.is_symlink():
                    raise ValueError('Symlink in raw storage; review explicitly')
                path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False) and entry.name not in ('.gitkeep', 'README.md'):
                    files.append({'path': path.relative_to(workspace).as_posix(),
                                  'format': path.suffix.lower().lstrip('.') or 'unknown', **fingerprint(path)})
                elif not entry.is_file(follow_symlinks=False):
                    raise ValueError('Unsupported special file in raw storage')
    return {'schema_version': '1.0.0', 'status': 'DATA_FOUND' if files else 'BLOCKED_ON_DATASET',
            'scope': 'Local raw storage only; no network or archive extraction',
            'raw_directory': raw.relative_to(workspace).as_posix(),
            'files': sorted(files, key=lambda item: item['path']), 'file_count': len(files),
            'dataset_statistics': None}
