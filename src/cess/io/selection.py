"""Read-only source-row views and legacy output naming; no eager reads."""
from pathlib import Path
from collections.abc import Mapping
import h5py
import numpy as np
from cess.contracts import MGE_MORPHOLOGY_FIELDS


class _SelectedDataset:
    def __init__(self, dataset, selection):
        self.dataset, self.selection = dataset, selection
        self._data = None

    @property
    def data(self):
        if self._data is None:
            self._data = self.dataset[self.selection]
        return self._data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.selection.stop - self.selection.start

    def __array__(self, dtype=None, copy=None):
        """Honor NumPy's copy contract, including lazy HDF5 materialization.

        An uncached HDF5 selection requires a new array, so ``copy=False``
        cannot materialize it. Once cached, that option can return the same
        dtype without allocating. ``copy=True`` always copies the cached data.
        """
        if copy is not None and not isinstance(copy, (bool, np.bool_)):
            raise TypeError('copy must be True, False, or None')
        if copy is not None and not copy:
            if self._data is None:
                raise ValueError('copy=False cannot materialize an HDF5 selection')
            if dtype is not None and np.dtype(dtype) != self._data.dtype:
                raise ValueError('copy=False cannot convert the selection dtype')
            return self._data
        if copy:
            return np.array(self.data, dtype=dtype, copy=True)
        return np.asarray(self.data, dtype=dtype)

    @property
    def shape(self):
        return (len(self),) + self.dataset.shape[1:]

    @property
    def dtype(self):
        return self.dataset.dtype

    @property
    def attrs(self):
        return self.dataset.attrs

    @property
    def name(self):
        return self.dataset.name

    @property
    def file(self):
        return self.dataset.file

    @property
    def filename(self):
        return self.dataset.file.filename

    @property
    def ndim(self):
        return len(self.shape)

class SourceSelection(Mapping):
    """A read-only, lazy row view of a BayeSED or CESS HDF5 input.

    Only declared source-axis datasets are sliced; shared wavelengths and
    parameter names are never inferred from their coincidentally matching size.
    """
    def __init__(self, handle, source_start=0, source_stop=None, _root=None):
        if _root is None:
            if callable(getattr(handle, 'legacy_view', None)):
                handle = handle.legacy_view()
            elif isinstance(handle, h5py.File):
                schema = handle.attrs.get('schema_version', '')
                schema = schema.decode('utf-8') if isinstance(schema, bytes) else str(schema)
                if schema == '3.0':
                    from .products import ProductView
                    handle = ProductView(handle).legacy_view()
        self.handle = handle
        if _root is not None:
            self.root = _root
            return
        self.root = self
        self.total_sources = len(handle['ID'])
        source_stop = self.total_sources if source_stop is None else source_stop
        if (isinstance(source_start, (bool, np.bool_)) or isinstance(source_stop, (bool, np.bool_)) or
                not isinstance(source_start, (int, np.integer)) or
                not isinstance(source_stop, (int, np.integer)) or
                not 0 <= source_start <= source_stop <= self.total_sources):
            raise ValueError('Source bounds must satisfy 0 <= start <= stop <= input length')
        self.selection = slice(int(source_start), int(source_stop))
        self.source_rows = np.arange(source_start, source_stop, dtype=np.int64)
        self._cache = {}

    @staticmethod
    def _source_axis(name):
        parts = name.strip('/').split('/')
        if parts[0] in ('catalog', 'spectra'):
            parts = parts[1:]
        if not parts or parts[0] == 'lines':
            return False
        if len(parts) == 1:
            return parts[0] in ('ID', 'parameters', 'parameters_desi',
                                'parameters_grism', 'parameters_phot', 'zero_order_phot',
                                'parameters_elg', 'data_mask', 'input_quality_flags',
                                'input_row_indices', 'parent_row_indices')
        if parts[0] == 'best_fit':
            return parts[-1] == 'z' or parts[-1].startswith('spec_')
        if len(parts) == 2 and parts[0] == 'spectral_extraction':
            return parts[-1] in ('source_fraction', 'spatial_pixels', 'profile_height_sum', 'valid')
        if len(parts) == 2 and parts[0] == 'mge_morphology':
            return parts[-1] in MGE_MORPHOLOGY_FIELDS
        return (len(parts) == 2 and parts[0] in ('GU', 'GV', 'GI') and
                (parts[-1].startswith(('flux_', 'ferr', 'response_flux_ujy',
                                       'covariance_lag1',
                                       'source_electron_rate_pre_morph',
                                       'sky_rate_e_s_pixel', 'detector_model_mask',
                                       'calibration_support_mask')) or
                 parts[-1] in ('snr', 'snr_0th', 'snr_wce', 'spec_mask',
                               'spec_mask_0th', 'snr_mask', 'snr_mask_0th')))

    def __getitem__(self, key):
        value = self.handle[key]
        if isinstance(value, h5py.Group) or (isinstance(value, Mapping) and not hasattr(value, 'dtype')):
            return SourceSelection(value, _root=self.root)
        if not self._source_axis(value.name):
            return value
        if not value.shape or value.shape[0] != self.root.total_sources:
            raise ValueError('Source-axis dataset is not aligned with ID: ' + value.name)
        if value.name not in self.root._cache:
            self.root._cache[value.name] = _SelectedDataset(value, self.root.selection)
        return self.root._cache[value.name]

    def __iter__(self):
        return iter(self.handle)

    def __contains__(self, key):
        return key in self.handle

    def __len__(self):
        return len(self.handle)

    def visititems(self, function):
        return self.handle.visititems(function)

    @property
    def attrs(self):
        return self.handle.attrs

    @property
    def filename(self):
        return self.handle.file.filename

    @property
    def file(self):
        return self.handle.file

    @property
    def name(self):
        return self.handle.name

def simulation_filename(filename, selected, prefix='CSST_grism'):
    """Keep legacy whole-file names and disambiguate source-range products."""
    start, stop = selected.selection.start, selected.selection.stop
    stem = Path(filename).stem
    if start != 0 or stop != selected.total_sources:
        stem += '_rows{}-{}'.format(start, stop)
    return '{}_{}_{}.hdf5'.format(prefix, stem, stop - start)

