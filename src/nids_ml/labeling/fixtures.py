"""Read bounded JSON fixtures with regular-file and identity checks.
Đọc fixture JSON có giới hạn, kiểm loại file và identity.
"""
from pathlib import Path
import stat
from nids_ml.labeling.contracts import MAX_JSON_BYTES, ContractError, strict_json


def read_fixture_json(path):
    path=Path(path)
    before=path.lstat()
    if path.suffix.lower()!='.json' or not stat.S_ISREG(before.st_mode) or before.st_nlink!=1:
        raise ContractError('Only independent regular .json fixtures are accepted')
    with path.open('rb') as stream:
        data=stream.read(MAX_JSON_BYTES+1)
    after=path.lstat()
    if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
        raise ContractError('Fixture changed during read')
    return strict_json(data)
