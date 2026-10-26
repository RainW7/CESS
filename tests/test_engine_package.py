"""Canonical engine composition checks; no catalogue/profile resources needed."""
from copy import deepcopy
import importlib
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cess import engine
from cess.constants import c, c_aa, h
from cess.spectra.operator import SpectraMixin
from cess.instrument.detector import DetectorMixin
from cess.lines.detection import LineDetectionMixin


@pytest.fixture
def model(monkeypatch, tmp_path):
    # Explicit detector settings isolate method composition from bundled paths.
    config = dict(_config_dir=str(tmp_path), input_path=str(tmp_path), radi=100.,
        expt=150., expnum=4, dc_value=.02, read_noise=5., arcsecperpix=.074,
        background_mode='fixed_template', background_scale=1., background_v_mag=22.1,
        template_lsf={'mode': 'delta'}, template_lsf_broader_policy='error')
    for band, start, stop, lo, hi, delta in (
            ('gu', 2000., 4400., 2500., 4200., 15.),
            ('gv', 3600., 6800., 3800., 6600., 21.),
            ('gi', 5600., 12000., 5800., 10500., 33.)):
        config.update({band+'_start_wl': start, band+'_end_wl': stop,
            band+'_min_wl': lo, band+'_max_wl': hi, band+'_dlambda': delta,
            band+'_init_idx': 0, band+'_end_idx': 51})
    monkeypatch.setattr(engine, 'load_config',
                        lambda value=None: deepcopy(config if value is None else value))
    return engine.emulator(config=config)


def arguments():
    wave = np.linspace(4000., 6000., 51)
    return (0., 0., wave, np.full(51, 1.e-27), wave, np.zeros(51),
            np.full(51, .5), np.ones(3), np.array([.2, .6, .2]))


def assert_rng_equal(first, second):
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2:] == second[2:]


def test_composition_retains_public_and_static_method_ownership():
    assert engine.emulator.sls_wave_spec is SpectraMixin.sls_wave_spec
    assert engine.emulator.add_noise is DetectorMixin.add_noise
    assert engine.emulator.el_detect is LineDetectionMixin.el_detect
    assert isinstance(DetectorMixin.__dict__['_sample_wave_offset'], staticmethod)
    assert isinstance(SpectraMixin.__dict__['response_electron_rate'], staticmethod)
    assert isinstance(LineDetectionMixin.__dict__['_emission_peaks'], staticmethod)
    assert (c, c_aa, h) == (2.9979e8, 2.9979e18, 6.626e-27)
    assert (engine.c, engine.c_aa, engine.h) == (c, c_aa, h)


def test_import_has_no_configuration_resource_or_rng_work(tmp_path):
    code = r'''
import importlib
import pickle
import sys
import numpy as np
import h5py
from astropy.table import Table
import cess.config.loader as config
# Preload third-party dependencies before checking CESS composition work.
import cess.spectra.operator
import cess.instrument.detector
import cess.lines.detection
assert 'cess.engine' not in sys.modules
def forbidden(*args, **kwargs): raise AssertionError('unexpected I/O/config load')
config.load_config = forbidden
h5py.File = forbidden
Table.read = forbidden
pickle.load = forbidden
np.random.seed(9124)
before = np.random.get_state()
module = importlib.import_module('cess.engine')
after = np.random.get_state()
assert before[0] == after[0] and before[2:] == after[2:]
assert np.array_equal(before[1], after[1])
assert 'galsim' not in sys.modules
assert module.emulator.__name__ == 'emulator'
'''
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=str(tmp_path),
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
    assert result.returncode == 0, result.stderr.decode('utf-8', errors='replace')


def test_facade_keeps_lazy_legacy_attributes_and_open_handle_contract(model, tmp_path):
    assert engine.expt == 150.
    assert engine.colarea == model.colarea
    assert engine.emulator_parameters is not model.config
    import h5py
    filename = tmp_path / 'input.hdf5'
    with h5py.File(str(filename), 'w') as handle:
        handle['ID'] = [7]
    model.sedpath, model.filename = str(tmp_path), filename.name
    handle = model.read_hdf5()
    try:
        np.testing.assert_array_equal(handle['ID'][:], [7])
        assert handle.id.valid
    finally:
        handle.close()
    assert not handle.id.valid


@pytest.mark.parametrize('global_stream', [False, True])
def test_noise_realization_retains_poisson_normal_sequence(model, global_stream):
    args = arguments()
    rate = np.linspace(1.e-6, 3.e-6, 51)
    fraction, pixels, seed = .4, 3, 1214
    source = rate * model.colarea * model.config['expt'] * model.config['expnum']
    source *= fraction
    sky = .1 * model.config['expt'] * model.config['expnum'] * pixels
    dark = model.config['dc_value'] * model.config['expt'] * model.config['expnum'] * pixels
    read_variance = model.config['read_noise'] ** 2 * model.config['expnum'] * pixels
    expected_rng = np.random.RandomState(seed)
    expected = (expected_rng.poisson(source + sky + dark) - sky - dark +
                expected_rng.normal(0., np.sqrt(read_variance), 51))
    before = np.random.get_state()
    try:
        if global_stream:
            np.random.seed(seed)
            actual_rng = None
        else:
            actual_rng = np.random.RandomState(seed)
        result = model._noise_realization(*args, False, rng=actual_rng,
            source_rate_e_s_cm2=rate, sky_rate_e_s_pixel=.1,
            source_fraction=fraction, extraction_n_pixels=pixels)
        np.testing.assert_array_equal(result['observed'], expected)
        np.testing.assert_array_equal(result['model'], source)
        np.testing.assert_array_equal(result['variance'], source + sky + dark + read_variance)
        state = np.random.get_state() if actual_rng is None else actual_rng.get_state()
        assert_rng_equal(state, expected_rng.get_state())
    finally:
        np.random.set_state(before)


@pytest.mark.parametrize('wce', [False, True])
def test_standard_paired_paths_keep_shared_draws_and_instance_patch(model, monkeypatch, wce):
    monkeypatch.setattr(model, '_sample_wave_offset', lambda rng=None: .5)
    first, second = np.random.RandomState(15), np.random.RandomState(15)
    args = arguments() + ('GV', True, wce)
    standard = model.add_noise(*args, rng=first, return_covariance=True,
                              sky_rate_e_s_pixel=.1)
    paired = model.add_noise_zouhu(*args, rng=second, return_covariance=True,
                                  sky_rate_e_s_pixel=.1)
    for a, b in ((0, 2), (1, 4), (2, 6), (4, 9), (7, 12), (8, 13), (9, 15)):
        np.testing.assert_array_equal(standard[a], paired[b])
    assert standard[6] == paired[11] == (.5 if wce else 0.)
    assert_rng_equal(first.get_state(), second.get_state())


def test_static_covariance_keeps_public_class_patch_and_linear_matrix(model, monkeypatch):
    original = engine.emulator._resample_pixel_shift
    calls = []
    def capture(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(engine.emulator, '_resample_pixel_shift', staticmethod(capture))
    variance = np.array([1., 4., 9., 16.])
    result = model._resample_covariance_lag1(variance, .25)
    matrix = np.array([[0., 0., 0., 0.], [.25, .75, 0., 0.],
                       [0., .25, .75, 0.], [0., 0., .25, .75]])
    expected = np.diag(matrix @ np.diag(variance) @ matrix.T, 1).copy()
    expected[0] = np.nan
    np.testing.assert_array_equal(result, expected)
    assert calls == [True]


def test_line_dependency_patch_targets_detection_service(model, monkeypatch):
    from cess.lines import detection
    calls = []
    def continuum(spectrum):
        return lambda wave: np.ones(len(wave)) * u.uJy
    expected = QTable({'line_center': [5000.] * u.angstrom,
                       'line_type': ['emission'], 'line_center_index': [10]})
    def find(spectrum, flux_threshold):
        calls.append(flux_threshold)
        return expected
    monkeypatch.setattr(detection, 'fit_generic_continuum', continuum)
    monkeypatch.setattr(detection, 'find_lines_derivative', find)
    actual = model.el_detect(np.linspace(4000., 6000., 21), np.ones(21), 0, 21, 1.5)
    assert actual is expected
    assert calls == [1.5]


def test_spectral_methods_resolve_canonical_services_and_invalid_rows(model):
    wave = np.geomspace(100., 100000., 1001)
    flux = np.ones((2, wave.size)) * 1.e-27
    flux[1, 200] = np.nan
    output_wave, output, valid, invalid = model.kernels_convolve(
        wave, flux, np.array([.5, .5]), 'GV', .19)
    assert output.shape == (2, len(output_wave))
    np.testing.assert_array_equal(valid, [True, False])
    np.testing.assert_array_equal(invalid, [False, True])
    assert np.isfinite(output[0]).all() and np.isnan(output[1]).all()
