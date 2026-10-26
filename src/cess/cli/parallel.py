"""Native-library thread settings inherited before spawn imports NumPy.

These are library requests, not CPU affinity or a scheduler allocation. Keep
the environment active for the entire pool lifetime: an initializer alone is
too late when the spawned entry point imports h5py/NumPy before initialization.
"""
from contextlib import contextmanager
import os

THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')


@contextmanager
def worker_thread_environment(threads):
    """Temporarily set native thread requests for newly spawned workers."""
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError('threads_per_worker must be a positive integer')
    previous = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    try:
        for name in THREAD_VARIABLES:
            os.environ[name] = str(threads)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
