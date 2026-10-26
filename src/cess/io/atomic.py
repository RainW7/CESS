"""No-clobber HDF5 publication retaining the legacy hard-link contract."""
from contextlib import contextmanager
from pathlib import Path
import os
import tempfile
import h5py


def output_destination(output_dir, filename):
    """Resolve a new output file and refuse to overwrite any existing path."""
    destination = Path(output_dir) / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(str(destination)):
        raise FileExistsError('Output already exists: {}'.format(destination))
    return destination

@contextmanager
def atomic_hdf5(destination):
    """Publish a complete HDF5 file atomically, without replacing an existing file.

    The temporary file is in the destination directory. A hard-link publish
    supplies atomic no-clobber behavior, including concurrent writers.
    """
    destination = output_destination(Path(destination).parent, Path(destination).name)
    descriptor, temporary = tempfile.mkstemp(
        prefix='.' + destination.name + '.', suffix='.partial', dir=str(destination.parent))
    os.close(descriptor)
    try:
        with h5py.File(temporary, 'w') as handle:
            handle.attrs['output_product'] = str(destination)
            yield handle
            handle.flush()
        os.link(temporary, str(destination))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

