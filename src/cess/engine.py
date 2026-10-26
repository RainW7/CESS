"""Public emulator facade composed from the retained C13 method blocks.

Construction loads an independently validated configuration. Import itself does
not load configuration, catalogues, profile libraries or the optional renderer.
For dependency monkeypatches use the service module that owns the method; the
public class and instance method hooks remain shared across compatibility entry
points. Numerical methods live in spectra.operator, instrument.detector and
lines.detection; none imports this composition module.
"""
import os
import h5py
import numpy as np

from cess.config.loader import load_config
from cess.constants import c, c_aa, h
from cess.spectra.operator import SpectraMixin
from cess.instrument.detector import DetectorMixin
import cess.instrument.detector as _detector
from cess.lines.detection import LineDetectionMixin
# Read aliases retained for old imports; patch dependencies on their service.
from cess.lines.detection import (Spectrum1D, find_lines_derivative,
                                 fit_generic_continuum, linear_sum_assignment)
from cess.instrument.detector import hst_vband_bkg, longitudes, latitudes


_LEGACY_CONFIG_NAMES = frozenset(('arcsecperpix', 'dc_value', 'expnum', 'expt', 'gi_dlambda', 'gi_end_idx', 'gi_end_wl', 'gi_init_idx', 'gi_max_wl', 'gi_min_wl', 'gi_start_wl', 'gu_dlambda', 'gu_end_idx', 'gu_end_wl', 'gu_init_idx', 'gu_max_wl', 'gu_min_wl', 'gu_start_wl', 'gv_dlambda', 'gv_end_idx', 'gv_end_wl', 'gv_init_idx', 'gv_max_wl', 'gv_min_wl', 'gv_start_wl', 'radi'))


def __getattr__(name):
    if name == 'emulator_parameters':
        return load_config()
    if name == 'colarea':
        return np.pi * load_config()['radi'] ** 2
    if name in _LEGACY_CONFIG_NAMES:
        return load_config()[name]
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


class emulator(SpectraMixin, DetectorMixin, LineDetectionMixin):
    def __init__(self, SED_file_path = '', file_name = '', config=None):
        """Create an isolated simulator from a mapping, JSON path or default config.

        Configuration is loaded here, never at module import. Each instance owns
        an independent dict; legacy module constants do not control its physics.
        """
        self.config = load_config(config)
        self._legacy_background_warning_issued = False
        self.rootpath = self.config['_config_dir']
        self.sedpath = os.fspath(SED_file_path) if SED_file_path else self.config.get('input_path', '')
        # self.csstgu = S.FileBandpass(rootpath+'/throughput/GUdesi.fits')
        # self.csstgv = S.FileBandpass(rootpath+'/throughput/GVdesi.fits')
        # self.csstgi = S.FileBandpass(rootpath+'/throughput/GIdesi.fits')
        self.filename = file_name

    @property
    def colarea(self):
        """Geometric collecting area, cm**2, for this instance."""
        return np.pi * self.config['radi'] ** 2

    def read_hdf5(self):
        """
        Read input BayeSED hdf5 file.
        
        return: 
        ------
            cat - hdf5 file catalog - [<HDF5 file (mode r)>]
        """
        cat = h5py.File(os.path.join(self.sedpath, self.filename), 'r')
        return cat


if __name__ == 'cess.engine':
    _detector._bind_emulator_class(emulator)
