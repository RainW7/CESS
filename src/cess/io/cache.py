"""One lazy per-process cache for trusted local calibration resources."""
from pathlib import Path
import pickle


_RESOURCE_CACHE = {}


def _resource_key(kind, path):
    path = Path(path).resolve()
    stat = path.stat()
    return kind, str(path), stat.st_size, stat.st_mtime_ns

def cached_table(path):
    """Load a FITS table on demand, once per unchanged resource per worker."""
    from astropy.table import Table
    key = _resource_key('table', path)
    if key not in _RESOURCE_CACHE:
        _RESOURCE_CACHE[key] = Table.read(str(path))
    return _RESOURCE_CACHE[key]

def cached_pickle(path):
    """Load a trusted local morphology library only when a task needs it."""
    key = _resource_key('pickle', path)
    if key not in _RESOURCE_CACHE:
        with open(str(path), 'rb') as handle:
            _RESOURCE_CACHE[key] = pickle.load(handle)
    return _RESOURCE_CACHE[key]

