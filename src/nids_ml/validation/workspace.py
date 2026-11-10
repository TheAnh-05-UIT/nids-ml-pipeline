"""Enforce workspace-relative paths and exclusive JSON output creation.
Giữ paths trong workspace và ghi JSON mới không ghi đè.
"""
import json
from pathlib import Path
import re


def workspace_path(workspace, relative):
    if not relative or '\\' in relative or re.match(r'^[A-Za-z]:', relative):
        raise ValueError('Use a POSIX workspace-relative path')
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Path escapes workspace')
    current = workspace
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Symlink paths require explicit review')
    result = (workspace / path).resolve()
    if not result.is_relative_to(workspace):
        raise ValueError('Resolved path escapes workspace')
    return result


def write_json(path, data):
    # Exclusive create: không ghi đè report/run cũ, kể cả file đã có một phần.
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        stream.write('\n')
