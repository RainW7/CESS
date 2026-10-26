"""Explicit task state for the package API; no module-level active configuration."""
import numpy as np

from cess.config.loader import load_config
from cess.pipeline.random import legacy_rng
from cess.pipeline import support


_OPERATION_NAMES = (
    'load_profile_libraries', 'cached_table', 'throughput_arrays',
    'throughput_interpolator', 'combine_band_masks', 'resolve_spectral_extraction',
    'measure_runner_band', 'write_vlen_rows', 'write_el_details',
    'write_spectral_extraction', 'write_mge_morphology', 'write_response_products',
    'write_spectral_covariance',
)


def legacy_axes(pixel_scale):
    """Original lookup axes, with the established rounding and iteration order."""
    return dict(
        n=np.array([.5,.6,.7,.8,.9,1.,1.1,1.2,1.3,1.4,1.5,1.6,1.8,2.,2.5,3.,3.5,4.,4.5,5.]),
        re_pixel=np.round(np.array([.3,.5,.7,.9,1.,1.2,1.4,1.6,1.8,2.,2.5,3.,3.5,4.5,5.,5.5,6.,6.5,7.,7.4])/pixel_scale),
        pa_deg=np.arange(0.,91.,10.), axis_ratio=np.arange(1.,11.)/10.)


class SimulationContext:
    """Own effective config, switches and empirical morphology for one task.

    Inputs are copied/validated; resources are loaded only by execution. Shared
    support caches contain calibration values, never an active task config.
    ``operations`` and ``morphology_model`` are explicit compatibility/testing
    injection points. Modern callers omit them and get isolated sampler state.
    """

    def __init__(self, config=None, options=None, operations=None, morphology_model=None):
        self.config = load_config(config)
        self.options = support.normalise_options(options)
        self.axes = legacy_axes(self.config['arcsecperpix'])
        if morphology_model is None:
            from cess.morphology.sampling import MorphologySampler
            morphology_model = MorphologySampler(self.config)
        self.morphology = morphology_model
        self.operations = {name: getattr(support, name) for name in _OPERATION_NAMES}
        self.operations.update(legacy_rng=legacy_rng,
                               configure_morphology=self.morphology.configure)
        if operations is not None:
            self.operations.update(operations)

    @property
    def colarea(self):
        return np.pi*self.config['radi']**2

    @property
    def delta_p_sys(self):
        return self.config['photometry_systematic_mag']
