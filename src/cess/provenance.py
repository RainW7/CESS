"""Shared identities for package source files, independent of cwd and shims.

The byte hashes are intentionally not cached: resume and HDF5 provenance must
agree on current bytes, including nested modules and package initializers.
Calling source_files/hash helpers performs filesystem I/O; importing does not
enumerate or read source/configuration/calibration files.
"""
import hashlib
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent


def source_files():
    """Return sorted absolute Paths of every Python source below ``cess``.

    Installed wheels include these .py files. Legacy compatibility wrappers,
    tests, private catalogues and working-directory files are not package code.
    """
    return tuple(sorted(path.resolve() for path in PACKAGE_ROOT.rglob('*.py') if path.is_file()))


def sha256_file(path):
    """SHA256 of current file bytes, with bounded read memory and no cache."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def code_sha256():
    """Return the absolute-path -> SHA256 map shared by runner and manifest."""
    return {str(path): sha256_file(path) for path in source_files()}
