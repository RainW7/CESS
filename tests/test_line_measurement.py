"""Independent dense GLS and fixed-window sampling checks of line diagnostics."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cess.lines.measurement import measure_line_window


def dense_reference(wave, flux, error, center, width, side_width, gap=0, lag=None):
    # Independent dense covariance and explicit GLS projection, for small tests.
    edges = np.r_[wave[0] - (wave[1]-wave[0])/2.,
                  (wave[:-1]+wave[1:])/2., wave[-1] + (wave[-1]-wave[-2])/2.]
    start, stop = center-width//2, center-width//2+width
    line = np.arange(start, stop)
    side = np.r_[np.arange(start-gap-side_width, start-gap),
                 np.arange(stop+gap, stop+gap+side_width)]
    conversion = 2.99792458e18 * 1.e-29 / wave**2
    covariance = np.diag(error**2)
    if lag is not None:
        covariance += np.diag(lag, 1) + np.diag(lag, -1)
    covariance *= np.outer(conversion, conversion)
    design = np.column_stack([np.ones(len(wave)), wave-wave.mean()])
    side_inverse = np.linalg.inv(covariance[np.ix_(side, side)])
    projection = np.linalg.inv(design[side].T @ side_inverse @ design[side]) @ design[side].T @ side_inverse
    weights = np.zeros(len(wave))
    weights[line] = np.diff(edges)[line]
    weights[side] = -weights[line] @ design[line] @ projection
    estimate = weights @ (flux * conversion)
    variance = weights @ covariance @ weights
    return estimate, np.sqrt(variance), weights, covariance


def spectrum(n=31, nonuniform=False):
    wave = (4000. + np.cumsum(np.linspace(2., 7., n)) if nonuniform else
            np.linspace(4000., 4120., n))
    conversion = 2.99792458e18 * 1.e-29 / wave**2
    f_lambda = 2.e-18 + 3.e-21 * (wave-wave.mean())
    return wave, f_lambda/conversion, np.full(n, .2)


@pytest.mark.parametrize('gap', [0, 2])
@pytest.mark.parametrize('rho', [-.2, 0., .4])
@pytest.mark.parametrize('nonuniform', [False, True])
def test_dense_reference_matches_all_line_side_and_cross_covariance(gap, rho, nonuniform):
    wave, flux, error = spectrum(nonuniform=nonuniform)
    flux[13:18] += np.array([.5, 1., 2., 1., .5])
    error *= np.linspace(.8, 1.3, len(wave))
    lag = rho * error[:-1] * error[1:]
    expected_flux, expected_error, _, _ = dense_reference(wave, flux, error, 15, 5, 4, gap, lag)
    result = measure_line_window(wave, flux, error, 15, 5, 4, gap, lag, model_ujy=flux)
    assert result['valid'] and result['flag'] == 0 and result['reason'] == 'ok'
    assert result['flux'] == pytest.approx(expected_flux, rel=3.e-12)
    assert result['error'] == pytest.approx(expected_error, rel=3.e-13)
    assert result['snr'] == pytest.approx(expected_flux/expected_error, rel=3.e-12)
    assert result['model_flux'] == result['flux'] and result['model_snr'] == result['snr']
    assert result['nline'] == 5 and result['nside'] == 8
    assert (result['line_start'],result['line_stop']) == (13,18)
    assert (result['left_start'],result['left_stop']) == (9-gap,13-gap)
    assert (result['right_start'],result['right_stop']) == (18+gap,22+gap)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('slope', [0., 3.e-21])
@pytest.mark.parametrize('amplitude', [-5.e-19, 0., 5.e-19])
def test_affine_f_lambda_continuum_and_signed_box_line(slope, amplitude):
    wave, _, error = spectrum(nonuniform=True)
    conversion = 2.99792458e18 * 1.e-29 / wave**2
    f_lambda = 2.e-18 + slope*(wave-wave.mean())
    f_lambda[13:18] += amplitude
    result = measure_line_window(wave, f_lambda/conversion, error, 15, 5, 4)
    edge_left = (wave[12] + wave[13])/2.
    edge_right = (wave[17] + wave[18])/2.
    expected = amplitude*(edge_right-edge_left)
    assert result['valid']
    assert result['line_wave_min'] == edge_left
    assert result['line_wave_max'] == edge_right
    assert result['flux'] == pytest.approx(expected, rel=2.e-12, abs=2.e-31)
    if amplitude < 0:
        assert result['snr'] < 0
    assert result['model_flux'] is result['model_snr'] is None


def test_gap_zero_cross_covariance_changes_continuum_subtracted_variance():
    wave, flux, error = spectrum()
    lag = .4 * error[:-1]*error[1:]
    _, expected_error, weights, covariance = dense_reference(wave, flux, error, 15, 5, 3, 0, lag)
    start, stop = 13, 18
    naive = covariance.copy()
    for i, j in [(start-1, start), (stop-1, stop)]:
        naive[i,j] = naive[j,i] = 0.
    without_cross = weights @ naive @ weights
    result = measure_line_window(wave, flux, error, 15, 5, 3, covariance_lag1=lag)
    assert result['error'] == pytest.approx(expected_error, rel=2.e-13)
    assert abs(result['error']**2/without_cross-1.) > .05


def test_fixed_window_monte_carlo_recovers_reported_full_variance():
    wave, model, error = spectrum(n=25)
    model[10:15] += 2.
    lag = .35 * error[:-1]*error[1:]
    expected_flux, expected_error, _, _ = dense_reference(wave, model, error, 12, 5, 4, 0, lag)
    covariance = np.diag(error**2) + np.diag(lag, 1) + np.diag(lag, -1)
    rng = np.random.RandomState(20261020)
    draws = rng.multivariate_normal(model, covariance, size=3000)
    measurements = [measure_line_window(wave, draw, error, 12, 5, 4,
                                        covariance_lag1=lag) for draw in draws]
    assert all(result['valid'] for result in measurements)
    standardized = np.array([(result['flux']-expected_flux)/result['error'] for result in measurements])
    assert np.allclose([result['error'] for result in measurements], expected_error, rtol=2.e-13, atol=0.)
    assert abs(standardized.mean()) < .06
    assert .94 < standardized.var(ddof=1) < 1.06


def test_unused_gap_nan_mask_model_and_covariance_do_not_shrink_window():
    wave, flux, error = spectrum()
    model = flux.copy()
    mask = np.ones(len(wave), bool)
    lag = np.zeros(len(wave)-1)
    # Line=[13,18), sides=[8,11),[20,23), gaps=[11,13),[18,20).
    for array in (flux, error, model):
        array[[11,12,18,19]] = np.nan
    mask[[11,12,18,19]] = False
    lag[[10,11,12,17,18,19]] = np.nan
    result = measure_line_window(wave, flux, error, 15, 5, 3, 2, lag, mask, model)
    assert result['valid'] and result['nline'] == 5 and result['nside'] == 6


@pytest.mark.parametrize('damage,flag', [('mask',2), ('flux',4), ('model',4), ('error_zero',8),
    ('error_nan',8), ('cov_nan',16), ('cov_not_pd',32)])
def test_used_unavailable_samples_are_explicitly_invalid(damage, flag):
    wave, flux, error = spectrum()
    model, mask, lag = flux.copy(), np.ones(len(wave), bool), np.zeros(len(wave)-1)
    if damage == 'mask':
        mask[14] = False
    elif damage == 'flux':
        flux[14] = np.nan
    elif damage == 'model':
        model[10] = np.nan  # a used sideband sample
    elif damage == 'error_zero':
        error[14] = 0.
    elif damage == 'error_nan':
        error[14] = np.nan
    elif damage == 'cov_nan':
        lag[12] = np.nan  # gap=0 line/side cross term
    else:
        lag[14] = 2.*error[14]*error[15]
    result = measure_line_window(wave, flux, error, 15, 5, 3, covariance_lag1=lag,
                                 mask=mask, model_ujy=model)
    assert not result['valid'] and result['flag'] == flag
    assert result['flux'] is result['error'] is result['snr'] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('n,center', [(0,0), (1,0), (31,0), (31,30), (31,-1), (31,32)])
def test_empty_and_partial_edge_windows_return_invalid_without_clipping(n, center):
    wave, flux, error = spectrum(n=n) if n else (np.array([]),np.array([]),np.array([]))
    result = measure_line_window(wave, flux, error, center, 5, 3)
    assert not result['valid'] and result['flag'] == 1
    assert result['nline'] == 5 and result['nside'] == 6
    assert result['line_wave_min'] is None


@pytest.mark.parametrize('damage', ['wave_order', 'wave_nan', 'shape', 'fractional_center',
    'bad_width', 'bad_gap', 'cov_shape', 'mask_shape', 'mask_nan', 'model_shape'])
def test_malformed_inputs_raise_value_error(damage):
    wave, flux, error = spectrum()
    kwargs = dict(center_index=15, line_width_pixels=5, sideband_width_pixels=3)
    if damage == 'wave_order':
        wave[14] = wave[13]
    elif damage == 'wave_nan':
        wave[0] = np.nan
    elif damage == 'shape':
        flux = flux[:-1]
    elif damage == 'fractional_center':
        kwargs['center_index'] = 15.5
    elif damage == 'bad_width':
        kwargs['line_width_pixels'] = 0
    elif damage == 'bad_gap':
        kwargs['sideband_gap_pixels'] = -1
    elif damage == 'cov_shape':
        kwargs['covariance_lag1'] = np.zeros(len(wave))
    elif damage == 'mask_shape':
        kwargs['mask'] = np.ones(len(wave)-1, bool)
    elif damage == 'mask_nan':
        kwargs['mask'] = np.full(len(wave), np.nan)
    else:
        kwargs['model_ujy'] = flux[:-1]
    with pytest.raises(ValueError):
        measure_line_window(wave, flux, error, **kwargs)


def test_none_covariance_is_explicit_diagonal_assumption():
    wave, flux, error = spectrum()
    a = measure_line_window(wave, flux, error, 15, 4, 3)
    b = measure_line_window(wave, flux, error, 15, 4, 3, covariance_lag1=np.zeros(len(wave)-1))
    assert a['covariance_model'] == 'diagonal' and b['covariance_model'] == 'diagonal_plus_lag1'
    assert a['flux'] == b['flux'] and a['error'] == b['error']
