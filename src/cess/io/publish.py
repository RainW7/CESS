"""Atomically publish a new profile-library directory without replacing a path.

Only the final directory rename belongs here: the caller owns staging creation,
content validation, file flushing and cleanup. A successful return guarantees
atomic visibility with no replacement, not power-loss durability. Staging and
its existing parent must remain under the caller's control during publication.
"""
import ctypes
import errno
import os
from pathlib import Path
import stat
import sys


_PLATFORM = sys.platform
_RENAME_EXCL = 0x00000004  # macOS sys/stdio.h
_RENAME_NOREPLACE = 1      # Linux renameat2 UAPI
_AT_FDCWD = -100


def _publication_path(value, role):
    """Keep the final name unresolved so a symlink cannot redirect output."""
    raw = os.fspath(value)
    if (not isinstance(raw, str) or not raw or '\x00' in raw or
            raw.rstrip(os.sep).split(os.sep)[-1] in ('', '.', '..')):
        raise ValueError('{} must name a directory, not a root/current/parent path'.format(role))
    path = Path(raw)
    if '..' in path.parts:
        raise ValueError('{} must not contain a parent traversal (..)'.format(role))
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise NotADirectoryError(errno.ENOTDIR, 'Publication parent is not a directory', str(parent))
    return parent / path.name


def _rename_no_replace(staging, output):
    """Use native no-replace semantics or fail closed; never normal rename."""
    if _PLATFORM == 'darwin':
        symbol = 'renamex_np'
        argument_types = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        arguments = (os.fsencode(staging), os.fsencode(output), _RENAME_EXCL)
    elif _PLATFORM.startswith('linux'):
        symbol = 'renameat2'
        argument_types = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint]
        arguments = (_AT_FDCWD, os.fsencode(staging), _AT_FDCWD,
                     os.fsencode(output), _RENAME_NOREPLACE)
    else:
        raise OSError(errno.ENOTSUP, 'Atomic no-replace directory publication is unsupported on ' + _PLATFORM)

    library = ctypes.CDLL(None, use_errno=True)
    try:
        rename = getattr(library, symbol)
    except AttributeError:
        raise OSError(errno.ENOTSUP, 'Required atomic no-replace operation is unavailable: ' + symbol)
    rename.argtypes = argument_types
    rename.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = rename(*arguments)
    if result != 0:
        error_number = ctypes.get_errno() or errno.EIO
        raise OSError(error_number, 'Atomic no-replace publication failed: ' + os.strerror(error_number), str(output))


def publish_directory(staging, output):
    """Move a real staging directory to an absent sibling atomically.

    All forms of existing output, including dangling symlinks and empty
    directories, raise without replacement. Paths containing ``..`` or naming
    the root/current directory are rejected; source symlinks and non-directory
    sources are rejected. Parent aliases are resolved before requiring the same
    existing parent, while the final names are never resolved through symlinks.

    Native macOS ``renamex_np(RENAME_EXCL)`` or Linux libc
    ``renameat2(RENAME_NOREPLACE)`` arbitrates destination races. Missing native
    support or filesystem rejection raises with no weaker fallback. This helper
    performs no cleanup, content validation, fsync, or production-config change.
    On success it returns None and staging no longer exists under its old name.
    """
    staging = _publication_path(staging, 'staging')
    output = _publication_path(output, 'output')
    if staging.parent != output.parent:
        raise ValueError('Staging and output must be siblings in the same existing parent')
    if staging == output:
        raise ValueError('Staging and output must be different paths')

    source_stat = staging.lstat()
    if stat.S_ISLNK(source_stat.st_mode):
        raise ValueError('Staging must be a real directory, not a symlink')
    if not stat.S_ISDIR(source_stat.st_mode):
        raise NotADirectoryError(errno.ENOTDIR, 'Staging must be a directory', str(staging))
    if os.path.lexists(str(output)):
        raise FileExistsError(errno.EEXIST, 'Output already exists; publication never replaces it', str(output))
    _rename_no_replace(staging, output)
