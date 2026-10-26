"""Synthetic correctness checks for the 2026-10-19 CESS core repair."""

import importlib.util
import os
from pathlib import Path
import sys
import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable


@pytest.fixture(scope='module')
def core():
    directory = Path(__file__).resolve().parents[1] / 'src'
    package_root = directory.parent / 'src'
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    previous = os.getcwd()
    sys.path.insert(0, str(directory))
    try:
        os.chdir(str(directory))
        spec = importlib.util.spec_from_file_location('cess_repaired_main', str(package_root / 'cess/engine.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.chdir(previous)
        sys.path.remove(str(directory))
    return module


def peaks(wavelengths, indices=None):
    wavelengths = np.asarray(wavelengths, dtype=float)
    if indices is None:
        indices = np.arange(len(wavelengths)) + 10
    return QTable({'line_center': wavelengths * u.AA,
                   'line_type': np.full(len(wavelengths), 'emission'),
                   'line_center_index': np.asarray(indices, dtype=int)})


@pytest.mark.parametrize('n', [0, 1, 2])
def test_convolution_zero_single_and_same_redshift(core, n):
    wave = np.geomspace(100., 100000., 2079)
    out_wave, flux, valid, invalid = core.emulator().kernels_convolve(
        wave, np.ones((n, len(wave))), np.full(n, .1), 'GV', .19)
    assert flux.shape == (n, len(out_wave))
    assert valid.shape == (n,)
    assert np.all(valid)
    assert np.all(np.isfinite(flux))
    np.testing.assert_array_equal(invalid, ~valid)


def test_convolution_invalid_rows_remain_aligned(core):
    wave = np.geomspace(100., 100000., 2079)
    flux = np.ones((4, len(wave)))
    flux[2, 10] = np.nan
    _, result, valid, invalid = core.emulator().kernels_convolve(
        wave, flux, np.array([.1, np.nan, .2, -1.]), 'GV', .19)
    np.testing.assert_array_equal(valid, [True, False, False, False])
    np.testing.assert_array_equal(invalid, ~valid)
    assert np.all(np.isnan(result[~valid]))
    assert np.all(np.isfinite(result[valid]))


def test_varsmooth_scalar_honors_xout_for_vectors_and_columns(core):
    x = np.linspace(4000., 6000., 257)
    y = np.exp(-.5 * ((x - 5000.) / 40.) ** 2)
    xout = np.linspace(4100., 5900., 193)
    e = core.emulator()
    result = e.varsmooth(x, y, 20., xout=xout)
    assert result.shape == (193,)
    np.testing.assert_allclose(result, np.interp(xout, x, e.varsmooth(x, y, 20.)))
    matrix = e.varsmooth(x, np.column_stack((y, y * 2)), 20., xout=xout)
    assert matrix.shape == (193, 2)
    np.testing.assert_allclose(matrix[:, 0], result)


def test_emission_truth_false_peaks_and_stable_empty_tables(core):
    e = core.emulator()
    truth, matched, details = e.el_success(
        [peaks([5001., 9000.]), peaks([])],
        [peaks([5000.]), peaks([6000.])], 50., return_details=True)
    assert truth.shape == (2,)
    assert matched.shape == (1,)
    np.testing.assert_array_equal(matched[0]['wave'], [5001.])
    np.testing.assert_array_equal(details['candidates'][0]['wave'], [5001., 9000.])
    np.testing.assert_array_equal(details['false_positive'][0]['wave'], [9000.])
    assert details['missed'][0]['id'] == 1
    assert details['pairs'].shape == (1,)
    for noisy, intrinsic in [([], []), ([peaks([])], [peaks([])]),
                             ([peaks([9000.])], [peaks([5000.])])]:
        a, b, d = e.el_success(noisy, intrinsic, 50., return_details=True)
        assert a.ndim == b.ndim == 1
        assert a.dtype.names and b.dtype.names
        assert b.shape == (0,)
        assert all(table.ndim == 1 and table.dtype.names for table in d.values())


def test_emission_maximum_cardinality_one_to_one_and_input_preserved(core):
    e = core.emulator()
    noisy = peaks([5030., 5070.])
    _, matched, details = e.el_success(
        [noisy], [peaks([5000., 5050.])], 35., return_details=True)
    assert matched[0]['detect_el_number'] == 2
    assert len(np.unique(details['pairs']['intrinsic_peak_id'])) == 2
    assert len(noisy) == 2
    _, matched = e.el_success([peaks([4990., 5010.])], [peaks([5000.])], 50.)
    assert matched[0]['detect_el_number'] == 1


def test_line_snr_uses_source_ids_and_clips_window_edges(core):
    e = core.emulator()
    _, matched = e.el_success([peaks([5001.], [0]), peaks([6001.], [149])],
                             [peaks([5000.]), peaks([6000.])], 50.)
    snr = np.vstack((np.full(150, 3.), np.full(150, 20.)))
    result = e.get_el_snr(snr, np.array([2, 2]), matched[::-1], 0, 4)
    assert result.shape == (2,) and result.dtype == object
    np.testing.assert_array_equal(result[0], [20.])
    np.testing.assert_array_equal(result[1], [3.])


def test_shift_identity_fractional_flux_variance_and_mask(core):
    e = core.emulator()
    impulse = np.zeros(21)
    impulse[10] = 1.
    variance = np.full(21, 4.)
    a, v, valid = e._resample_pixel_shift(impulse, 0., variance)
    np.testing.assert_array_equal(a, impulse)
    np.testing.assert_array_equal(v, variance)
    assert valid.all()
    a, v, valid = e._resample_pixel_shift(impulse, .5, variance)
    assert not valid[0] and np.isnan(a[0]) and np.isnan(v[0])
    assert np.nansum(a) == 1.
    np.testing.assert_array_equal(a[10:12], [.5, .5])
    np.testing.assert_array_equal(v[1:], np.full(20, 2.))
    off, shifted = e.wave_err(impulse, offset=.5)
    assert off == .5
    np.testing.assert_allclose(shifted, a, equal_nan=True)


def noise_arguments(n=30000, flux=0., height=9):
    wave = np.linspace(4000., 6000., n)
    return (0., 0., wave, np.full(n, flux), wave, np.zeros(n), np.ones(n),
            np.ones(height), np.array([.05, .2, .5, .2, .05]), 'GV')


def test_noise_realization_matches_reported_variance_and_preserves_negative(core):
    e = core.emulator()
    e.bkg_per_pix_time = lambda *args: 0.
    args = noise_arguments()
    result = e.add_noise(*args, False, False, rng=np.random.RandomState(20261019))
    wave = args[2]
    conversion = core.h * wave / (core.colarea * core.expt * core.expnum * np.gradient(wave))
    electrons = result[1] / conversion
    error = result[2] / conversion
    expected = (core.dc_value * core.expt * core.expnum * 9 + 25. * core.expnum * 9)
    assert abs(np.mean(electrons)) < 5 * np.sqrt(expected / len(wave))
    assert abs(np.var(electrons) / expected - 1) < .04
    np.testing.assert_allclose(error ** 2, expected)
    assert result[3] == pytest.approx(np.sqrt(expected))
    assert np.any(electrons < 0)


@pytest.mark.parametrize('morphology', [False, True])
@pytest.mark.parametrize('wce', [False, True])
def test_standard_and_paired_noise_share_correct_switch_semantics(core, monkeypatch, morphology, wce):
    e = core.emulator()
    e.bkg_per_pix_time = lambda *args: 0.
    monkeypatch.setattr(e, '_sample_wave_offset', lambda rng=None: .5)
    args = list(noise_arguments(n=401, flux=1e-27))
    args[3][200] = 1e-24
    standard = e.add_noise(*args, morphology, wce, rng=np.random.RandomState(83))
    paired = e.add_noise_zouhu(*args, morphology, wce, rng=np.random.RandomState(83))
    for a, b in [(standard[0], paired[2]), (standard[1], paired[4]),
                 (standard[2], paired[6]), (standard[4], paired[9])]:
        np.testing.assert_allclose(a, b, equal_nan=True)
    np.testing.assert_allclose(standard[1] / standard[2], standard[4], equal_nan=True)
    if wce:
        assert standard[6] == .5
        assert not standard[7][0] and not standard[8][0]
        assert np.isnan(standard[1][0])
    else:
        assert standard[6] == 0.
        np.testing.assert_array_equal(paired[1], paired[2])
        np.testing.assert_array_equal(paired[3], paired[4])
        np.testing.assert_array_equal(paired[5], paired[6])
        np.testing.assert_array_equal(paired[8], paired[9])
    if morphology:
        unbroadened = e.add_noise(*args, False, False, rng=np.random.RandomState(83))
        assert not np.allclose(paired[1], unbroadened[0])


def test_strong_source_not_clipped_and_zero_response_invalid(core):
    e = core.emulator()
    e.bkg_per_pix_time = lambda *args: 0.
    args = list(noise_arguments(n=401, flux=1e-22))
    args[6][0] = 0.
    result = e.add_noise(*args, False, False, rng=np.random.RandomState(8))
    assert np.nanmin(result[0]) > 90000
    assert np.isnan(result[1][0]) and np.isnan(result[2][0])
    assert not result[7][0] and not result[8][0]


def test_el_detection_ignores_nan_outside_requested_window(core):
    wave = np.linspace(4000., 6000., 501)
    flux = 1. + 5. * np.exp(-.5 * ((wave - 5000.) / 12.) ** 2)
    flux[[0, -1]] = np.nan
    lines = core.emulator().el_detect(wave, flux, 20, 480, 1.5)
    emission = lines[lines['line_type'] == 'emission']
    assert len(emission) == 1
    assert abs(emission['line_center'][0].to_value(u.AA) - 5000.) < 5.
    flux[250] = np.nan
    assert len(core.emulator().el_detect(wave, flux, 20, 480, 1.5)) == 0


def test_emission_type_bytes_and_empty_numeric_column(core):
    import warnings
    empty = QTable({'line_center': np.array([]) * u.AA,
                    'line_type': np.array([], dtype=float),
                    'line_center_index': np.array([], dtype=int)})
    encoded = QTable({'line_center': np.array([5000.]) * u.AA,
                      'line_type': np.array([b'emission'], dtype='S10'),
                      'line_center_index': np.array([10])})
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        _, matched = core.emulator().el_success([empty, encoded], [empty, encoded], 10.)
    assert len(matched) == 1 and matched[0]['id'] == 1
