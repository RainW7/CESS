"""Stable per-source random streams, independent of worker or chunk order.

The stream identity uses the input content hash, original source row, ID and
component. Relocating an unchanged input does not change the draws. This module
does not read files at import. Legacy NumPy consumers can use ``legacy_rng``;
that context is process-local and must not be used concurrently in threads.
"""
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import numbers
from pathlib import Path
import random

import numpy as np

STREAM_VERSION = "cess-source-sha256-randomstate-v1"


@lru_cache(maxsize=128)
def _file_digest(path, size, mtime_ns):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path):
    """SHA256 of bytes, cached per path/size/mtime; never a hash of the path."""
    path = Path(path).resolve()
    stat = path.stat()
    return _file_digest(str(path), stat.st_size, stat.st_mtime_ns)


def _identifier(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return {"bytes_hex": value.hex()}
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return str(value)
    raise ValueError("source_id must be a string, integer or bytes identifier")


def source_seed(master_seed, input_identity, source_id, row_index, component):
    """Derive 256 bits of seed material with canonical SHA256 encoding.

    Do not truncate to one 32-bit integer: large CESS catalogues would then
    produce many identical streams through birthday collisions.
    """
    if (not isinstance(master_seed, numbers.Integral) or isinstance(master_seed, bool)
            or master_seed < 0):
        raise ValueError("master_seed must be a non-negative integer")
    if (not isinstance(row_index, numbers.Integral) or isinstance(row_index, bool)
            or row_index < 0):
        raise ValueError("row_index must be the non-negative original input row")
    if not isinstance(input_identity, str) or not input_identity:
        raise ValueError("input_identity must be a non-empty stable string")
    if not isinstance(component, str) or not component:
        raise ValueError("component must be a non-empty string")
    encoded = json.dumps([STREAM_VERSION, int(master_seed), input_identity,
                          _identifier(source_id), int(row_index), component],
                         ensure_ascii=True, separators=(",", ":")).encode("ascii")
    return int.from_bytes(hashlib.sha256(encoded).digest(), "little")


def source_rng(master_seed, input_identity, source_id, row_index, component):
    """Return RandomState for a source/component, or None for legacy randomness."""
    if master_seed is None:
        return None
    seed = source_seed(master_seed, input_identity, source_id, row_index, component)
    words = np.frombuffer(seed.to_bytes(32, "little"), dtype="<u4").copy()
    return np.random.RandomState(words)


@contextmanager
def legacy_rng(rng):
    """Temporarily supply a stream to old np.random/Python-random call sites.

    An explicit RandomState advances with the NumPy calls made in this context;
    surrounding global RNG states are restored even on exceptions. ``None``
    leaves the normal global generators alone for API backward compatibility.
    """
    if rng is None:
        yield
        return
    if not isinstance(rng, np.random.RandomState):
        raise TypeError("legacy_rng requires RandomState or None")
    numpy_state, python_state = np.random.get_state(), random.getstate()
    state = rng.get_state()
    try:
        np.random.set_state(state)
        random.seed(int(state[1][0]))
        yield
    finally:
        rng.set_state(np.random.get_state())
        np.random.set_state(numpy_state)
        random.setstate(python_state)
