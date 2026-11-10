"""Validate artifact paths and publish new HDF5 files without overwriting.
Kiểm đường dẫn artifact và công bố HDF5 mới không ghi đè.
"""
import hashlib
import os
from pathlib import Path
import stat
import tempfile

from .schema import ExportError,MAX_FILE_BYTES

RUNS_ROOT=Path(__file__).resolve().parents[4]/'artifacts/runs'


def safe_path(path,*,existing):
    path=Path(path).absolute()
    if path.suffix.lower() not in ('.h5','.hdf5') or not path.resolve().is_relative_to(RUNS_ROOT.resolve()):
        raise ExportError('HDF5_PATH_UNDER_ARTIFACTS_RUNS_REQUIRED')
    for part in (path,*path.parents):
        if os.path.lexists(part):
            info=part.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',1024):
                raise ExportError('SYMLINK_OR_REPARSE_PATH')
    if not path.parent.is_dir():raise ExportError('EXISTING_OUTPUT_PARENT_REQUIRED')
    if existing:
        if not path.is_file() or path.stat().st_nlink!=1:raise ExportError('INDEPENDENT_REGULAR_HDF5_REQUIRED')
        if path.stat().st_size>MAX_FILE_BYTES:raise ExportError('HDF5_FILE_SIZE_LIMIT')
    elif os.path.lexists(path):raise ExportError('OUTPUT_EXISTS_NO_OVERWRITE')
    return path


def signature(path):
    item=path.stat()
    return item.st_dev,item.st_ino,item.st_size,item.st_mtime_ns,item.st_nlink


def file_sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()


def staging_path(target):
    target=safe_path(target,existing=False)
    directory=Path(tempfile.mkdtemp(prefix='.'+target.name+'.pending-',dir=target.parent))
    return directory/'payload.hdf5'


def publish(stage,target):
    # Hard-link creation là no-clobber trên cùng filesystem, không dùng replace().
    # Sau commit chỉ còn một link; không hỗ trợ filesystem không có hard links.
    linked=False
    try:
        os.link(stage,target)
        linked=True
        stage.unlink()
    except OSError as error:
        if linked and stage.exists() and target.exists():
            left,right=stage.stat(),target.stat()
            if (left.st_dev,left.st_ino)==(right.st_dev,right.st_ino):
                target.unlink()  # Chỉ rollback link mới của chính lần publish này.
        raise ExportError('ATOMIC_NO_CLOBBER_PUBLICATION_FAILED') from error
    try:stage.parent.rmdir()  # Chỉ thư mục staging riêng đã rỗng.
    except OSError:pass
