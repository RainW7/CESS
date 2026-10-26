"""Independent matrix and sampling checks for registered spectral covariance."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cess.engine import emulator, h


def registration(n, offset):
    matrix = np.zeros((n, n))
    valid = np.zeros(n, dtype=bool)
    for j in range(n):
        x = j - offset
        if 0 <= x <= n - 1:
            lo = int(np.floor(x))
            fraction = x - lo
            matrix[j, lo] += 1. - fraction
            if fraction:
                matrix[j, lo + 1] += fraction
            valid[j] = True
    return matrix, valid


@pytest.mark.parametrize('offset', [0., -.5, .5, -1.25, 1.75, -2., 2., 40.])
def test_lag1_matches_independent_dense_response_matrix(offset):
    variance = np.linspace(.5, 8., 19) ** 2
    response, valid = registration(len(variance), offset)
    covariance = response.dot(np.diag(variance)).dot(response.T)
    actual = emulator._resample_covariance_lag1(variance, offset)
    pairs = valid[:-1] & valid[1:]
    np.testing.assert_allclose(actual[pairs], np.diag(covariance, 1)[pairs], rtol=2e-15)
    assert np.isnan(actual[~pairs]).all()
    # A scalar linear shift cannot connect bins separated by more than one.
    assert np.all(np.triu(covariance, 2) == 0.)


def test_masks_and_missing_variance_do_not_create_zero_covariance():
    variance = np.ones(11)
    variance[4] = np.nan
    mask = np.ones(11, dtype=bool)
    mask[8] = False
    for offset in (0., .5):
        _, _, good = emulator._resample_pixel_shift(np.zeros(11), offset, variance, mask)
        actual = emulator._resample_covariance_lag1(variance, offset, mask)
        np.testing.assert_array_equal(np.isfinite(actual), good[:-1] & good[1:])
        np.testing.assert_allclose(actual[np.isfinite(actual)], 0. if offset == 0 else .25)


def test_registered_window_variance_matches_monte_carlo():
    variance = np.linspace(1., 3., 20)
    response, _ = registration(20, .5)
    weights = np.zeros(20)
    weights[5:11] = 1.
    expected = np.sum(response.T.dot(weights) ** 2 * variance)
    rng = np.random.RandomState(20261020)
    sample = rng.normal(size=(60000, 20)) * np.sqrt(variance)
    measured = sample.dot(response.T).dot(weights)
    assert np.var(measured, ddof=1) == pytest.approx(expected, rel=.025)
    diagonal_only = np.sum(weights ** 2 * np.diag(response.dot(np.diag(variance)).dot(response.T)))
    assert expected > 1.7 * diagonal_only


@pytest.mark.parametrize('offset', [0., .5, -.7])
def test_calibrated_covariance_and_optional_api_preserve_rng_and_values(offset, monkeypatch):
    e = emulator()
    monkeypatch.setattr(e, '_sample_wave_offset', lambda rng=None: offset)
    n = 61
    wave = np.linspace(4000., 6000., n)
    tp = np.linspace(.1, .9, n)
    tp[0] = 0.
    args = (0., 0., wave, np.ones(n) * 1e-27, wave, np.zeros(n), tp,
            np.ones(3), np.ones(1), 'GV', False, True)
    kwargs = dict(source_rate_e_s_cm2=np.linspace(.001, .003, n), sky_rate_e_s_pixel=.01)
    rng_a, rng_b = np.random.RandomState(193), np.random.RandomState(193)
    original = e.add_noise(*args, rng=rng_a, **kwargs)
    extended = e.add_noise(*args, rng=rng_b, return_covariance=True, **kwargs)
    assert len(extended) == len(original) + 1
    for a, b in zip(original, extended):
        np.testing.assert_allclose(a, b, equal_nan=True, rtol=0, atol=0)
    assert rng_a.random_sample() == rng_b.random_sample()
    source = kwargs['source_rate_e_s_cm2'] * e.colarea * e.config['expt'] * e.config['expnum']
    base_var = source + (.01 + e.config['dc_value']) * e.config['expt'] * e.config['expnum'] * 3
    base_var += e.config['read_noise'] ** 2 * e.config['expnum'] * 3
    response, valid = registration(n, offset)
    conversion = h * wave / (e.colarea * e.config['expt'] * e.config['expnum'] * np.gradient(wave) * np.where(tp > 0, tp, 1.))
    matrix = response.dot(np.diag(base_var)).dot(response.T) * conversion[:, None] * conversion[None, :]
    valid &= tp > 0
    pairs = valid[:-1] & valid[1:]
    np.testing.assert_allclose(extended[-1][pairs], np.diag(matrix, 1)[pairs], rtol=3e-15)
    assert np.isnan(extended[-1][~pairs]).all()
    np.testing.assert_allclose(extended[2][valid] ** 2, np.diag(matrix)[valid], rtol=3e-15)


def test_paired_optional_covariances_and_shared_realization(monkeypatch):
    e = emulator()
    monkeypatch.setattr(e, '_sample_wave_offset', lambda rng=None: .5)
    n = 41
    wave = np.linspace(4000., 6000., n)
    args = (0., 0., wave, np.ones(n) * 1e-27, wave, np.zeros(n), np.ones(n),
            np.ones(2), np.ones(1), 'GV', False, True)
    a = e.add_noise_zouhu(*args, rng=np.random.RandomState(71), sky_rate_e_s_pixel=0.)
    b = e.add_noise_zouhu(*args, rng=np.random.RandomState(71), sky_rate_e_s_pixel=0., return_covariance=True)
    assert len(b) == len(a) + 2
    for old, new in zip(a, b):
        np.testing.assert_allclose(old, new, equal_nan=True, rtol=0, atol=0)
    np.testing.assert_array_equal(b[-2], np.zeros(n - 1))
    assert np.isnan(b[-1][0]) and np.all(b[-1][1:] > 0)


@pytest.mark.parametrize('variance,offset', [(np.ones((2, 3)), .5), (np.ones(3), np.nan)])
def test_bad_covariance_arguments(variance, offset):
    with pytest.raises(ValueError):
        emulator._resample_covariance_lag1(variance, offset)
