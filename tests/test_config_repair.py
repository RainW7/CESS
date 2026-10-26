"""Explicit configuration, import safety and calibration-cache regression tests."""

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import h5py
import numpy as np
from scipy.integrate import trapezoid
import pytest
from astropy.io import fits

CORE = Path(__file__).resolve().parents[1] / 'src'


def _module(name, filename):
    # File-spec tests deliberately create isolated state. Legacy entry files now
    # alias canonical modules; execute the owning source for this private probe.
    targets = {'cess_config.py': 'config/loader.py', 'main.py': 'engine.py',
               'morphology.py': 'morphology/empirical.py'}
    package_root = CORE.parent / 'src'
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    sys.path.insert(0, str(CORE))
    try:
        spec = importlib.util.spec_from_file_location(name, str(package_root / 'cess' / targets[filename]))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(CORE))
    return module


@pytest.fixture(scope='module')
def configuration():
    return _module('cess_config_tests', 'cess_config.py')


@pytest.fixture(scope='module')
def core():
    return _module('cess_main_config_tests', 'main.py')


@pytest.fixture
def baseline(configuration):
    return configuration.load_config(configuration.DEFAULT_CONFIG_PATH)


def test_imports_without_configuration_or_fits_from_arbitrary_cwd(tmp_path):
    code = '''
import sys
sys.path.insert(0, sys.argv[1])
from astropy.io import fits
import cess.config.loader as cess_config
def forbidden(*args, **kwargs):
    raise AssertionError('import attempted configuration/calibration I/O')
cess_config.load_config = forbidden
fits.open = forbidden
import cess.engine as main
import cess.morphology.empirical as morphology
assert 'radi' not in main.__dict__
assert morphology._config is None and morphology._gems_data is None
print('no configuration/calibration I/O')
'''
    env = dict(os.environ, CESS_CONFIG=str(tmp_path / 'missing.json'),
               MPLCONFIGDIR=str(tmp_path / 'mpl'), PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-B', '-c', code, str(CORE)],
                            cwd=str(tmp_path), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'no configuration/calibration I/O' in result.stdout


def test_relative_resources_resolve_against_json_and_copy_metadata(
        configuration, baseline, tmp_path, monkeypatch):
    folder = tmp_path / 'configuration'
    folder.mkdir()
    baseline.update(input_path='inputs', output_path='../outputs',
                    gems_path='cal/gems.fits', widthlib_pkl='cal/width.pkl')
    source = folder / 'custom.json'
    source.write_text(json.dumps(baseline))
    monkeypatch.chdir(str(tmp_path))
    loaded = configuration.load_config(source)
    assert loaded['input_path'] == str(folder / 'inputs')
    assert loaded['output_path'] == str(tmp_path / 'outputs')
    assert loaded['gems_path'] == str(folder / 'cal/gems.fits')
    assert loaded['widthlib_pkl'] == str(folder / 'cal/width.pkl')
    assert loaded['_config_path'] == str(source)
    assert loaded['_config_dir'] == str(folder)
    loaded['extra'] = {'values': [1, 2]}
    copied = configuration.validate_config(loaded)
    copied['extra']['values'][0] = 9
    assert loaded['extra']['values'][0] == 1
    assert not Path(loaded['gems_path']).exists()


def test_environment_explicit_default_precedence(configuration, baseline, tmp_path, monkeypatch):
    first, second = tmp_path / 'first.json', tmp_path / 'second.json'
    baseline['expt'] = 10.
    first.write_text(json.dumps(baseline))
    baseline['expt'] = 20.
    second.write_text(json.dumps(baseline))
    monkeypatch.setenv('CESS_CONFIG', str(first))
    assert configuration.load_config()['expt'] == 10.
    assert configuration.load_config(second)['expt'] == 20.
    monkeypatch.delenv('CESS_CONFIG')
    monkeypatch.chdir(str(tmp_path))
    assert configuration.load_config()['_config_path'] == str(configuration.DEFAULT_CONFIG_PATH)


@pytest.mark.parametrize('name,value', [
    ('expt', 0.), ('expt', np.nan), ('radi', -1.), ('expnum', 1.5),
    ('arcsecperpix', 0.), ('read_noise', -1.), ('dc_value', -1.),
    ('gu_min_wl', 10000.), ('gv_init_idx', -1), ('gi_end_idx', 0),
    ('gu_dlambda', 100000.), ('background_mode', 'invented_geometry'),
    ('background_scale', -1.), ('background_v_mag', np.inf),
])
def test_invalid_config_rejected(configuration, baseline, name, value):
    baseline[name] = value
    with pytest.raises(ValueError):
        configuration.validate_config(baseline)


def test_missing_fields_and_explicit_base_directory(configuration, baseline, tmp_path):
    baseline.pop('radi')
    with pytest.raises(ValueError, match='radi'):
        configuration.validate_config(baseline)
    baseline['radi'] = 100.
    baseline['gems_path'] = 'relative.fits'
    result = configuration.validate_config(baseline, base_dir=tmp_path)
    assert result['gems_path'] == str(tmp_path / 'relative.fits')


def test_instances_isolated_from_input_other_instances_and_module_globals(
        core, baseline, monkeypatch, tmp_path):
    first = deepcopy(baseline)
    first.update(radi=50., expt=60., expnum=2, arcsecperpix=.1, read_noise=3.,
                 dc_value=0., background_scale=0., extra={'nested': [1]})
    second = deepcopy(first)
    second.update(radi=100., expt=120., expnum=4, arcsecperpix=.05, read_noise=6.)
    a = core.emulator(config=first)
    source = tmp_path / 'second.json'
    source.write_text(json.dumps(second))
    b = core.emulator(config=source)
    first['expt'] = 999.
    first['extra']['nested'][0] = 8
    monkeypatch.setattr(core, 'expt', 999.)
    monkeypatch.setattr(core, 'colarea', 1.)
    assert a.config['expt'] == 60. and a.config['extra']['nested'] == [1]
    wave = np.linspace(4000., 6000., 201)
    args = (0., 0., wave, np.full(201, 1e-27), wave, np.zeros(201),
            np.ones(201), np.ones(3), np.ones(5), False)
    ra = a._noise_realization(*args, rng=np.random.RandomState(1))
    rb = b._noise_realization(*args, rng=np.random.RandomState(1))
    np.testing.assert_allclose(rb['model'], ra['model'] * 16.)
    np.testing.assert_allclose(ra['variance'] - ra['model'], 3. ** 2 * 2 * 3)
    np.testing.assert_allclose(rb['variance'] - rb['model'], 6. ** 2 * 4 * 3)
    template = np.geomspace(100., 100000., 2079)
    flux = np.ones((1, len(template)))
    aw = a.kernels_convolve(template, flux, [.1], 'GV', .19)[0]
    bw = b.kernels_convolve(template, flux, [.1], 'GV', .19)[0]
    assert len(bw) > len(aw)
    assert a.colarea == pytest.approx(np.pi * 50. ** 2)
    assert b.config['expt'] == 120.


def test_read_hdf5_directory_without_trailing_separator(core, baseline, tmp_path):
    with h5py.File(str(tmp_path / 'input.h5'), 'w') as file:
        file['ID'] = [7]
    e = core.emulator(str(tmp_path), 'input.h5', config=baseline)
    with e.read_hdf5() as file:
        np.testing.assert_array_equal(file['ID'][:], [7])


def _gems(path, n, ar):
    fits.BinTableHDU.from_columns([
        fits.Column(name='SCIENCE_FLAG', format='J', array=[1, 1]),
        fits.Column(name='FIT_MAG', format='D', array=[18., 18.]),
        fits.Column(name='FIT_N', format='D', array=[n, n]),
        fits.Column(name='SEX_AR', format='D', array=[ar, ar]),
    ]).writeto(str(path))


def test_gems_lazy_closed_and_cache_invalidated(baseline, tmp_path, monkeypatch):
    morphology = _module('cess_morph_config_tests', 'morphology.py')
    first, second = tmp_path / 'first.fits', tmp_path / 'second.fits'
    _gems(first, 1.2, .3)
    _gems(second, 4.4, .9)
    opened = []
    actual_open = morphology.fits.open
    def count_open(*args, **kwargs):
        opened.append(str(args[0]))
        return actual_open(*args, **kwargs)
    monkeypatch.setattr(morphology.fits, 'open', count_open)
    baseline['gems_path'] = str(first)
    before = np.random.get_state()
    morphology.configure(baseline)
    morphology.extraction_height([.2, .6, .2], .5)
    assert opened == []
    np.testing.assert_array_equal(np.random.get_state()[1], before[1])
    assert morphology.get_ar() == .3
    assert morphology.get_n_from_mag(18., 14., 21., .5, 5.) == 1.2
    assert opened == [str(first)] and morphology.gems_cat._file.closed
    assert morphology._n_distributions.cache_info().currsize == 1
    baseline['gems_path'] = str(second)
    result = morphology.configure(baseline)
    result['gems_path'] = 'not-the-configured-file'
    assert morphology._n_distributions.cache_info().currsize == 0
    assert opened == [str(first)]
    assert morphology.get_ar() == .9
    assert morphology.get_n_from_mag(18., 14., 21., .5, 5.) == 4.4
    assert opened == [str(first), str(second)]


def test_profile_rendering_does_not_load_gems(baseline, tmp_path):
    morphology = _module('cess_morph_profile_config_tests', 'morphology.py')
    baseline.update(gems_path=str(tmp_path / 'missing.fits'), arcsecperpix=.1)
    morphology.configure(baseline)
    width, height = morphology.get_2d_profile(1., 4., 0., .8)
    assert np.all(np.isfinite(width)) and np.sum(width) > 0
    assert np.all(np.isfinite(height)) and np.sum(height) > 0
    assert morphology._gems_data is None


def test_fixed_template_coordinate_independent_scaling(core, baseline):
    wave = np.linspace(4000., 6000., 101)
    flux, throughput = np.full(101, 1e-29), np.full(101, .5)
    baseline.update(background_mode='fixed_template', background_scale=3., background_v_mag=24.)
    e = core.emulator(config=baseline)
    expected = (trapezoid(throughput * flux * 3. / (core.h * wave), wave) *
                e.colarea * baseline['arcsecperpix'] ** 2)
    for ra, dec in [(0., 0.), (180., 90.), (np.nan, np.nan)]:
        assert e.bkg_per_pix_time(ra, dec, wave, flux, throughput) == pytest.approx(expected)
    assert e.match_bkg_mag(0., 0.) == 24.  # label does not renormalize the template
    baseline['background_scale'] = 0.
    assert core.emulator(config=baseline).bkg_per_pix_time(0., 0., wave, flux, throughput) == 0.


def test_legacy_background_only_explicit_and_warned(core, baseline):
    baseline.update(background_mode='legacy_radec', background_scale=1., background_v_mag=22.1)
    e = core.emulator(config=baseline)
    with pytest.warns(RuntimeWarning, match='legacy_radec'):
        assert e.match_bkg_mag(0., 0.) == 20.
    assert e._legacy_background_warning_issued
    with pytest.raises(ValueError, match='finite'):
        e.match_bkg_mag(np.nan, 0.)
