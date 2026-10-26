"""Analytic and small stochastic checks of the independent photometry module."""

from pathlib import Path
import sys

import numpy as np
from scipy.integrate import trapezoid
import pytest
from astropy import units as u

CORE = Path(__file__).resolve().parents[1] / 'CESS_v0.8.6'
sys.path.insert(0, str(CORE))
from cess_photometry import (measure_photometry, PLANCK_ERG_S,
                             FLAG_INCOMPLETE_COVERAGE, FLAG_NONPOSITIVE_OBSERVED)


class ExpectedCounts:
    """Evaluate calibration at the mean rather than adding a random residual."""
    def poisson(self, mean):
        return mean

    def normal(self, mean, sigma):
        return mean


class NoDraw:
    def poisson(self, mean):
        raise AssertionError('invalid measurements must not consume RNG')

    def normal(self, mean, sigma):
        raise AssertionError('invalid measurements must not consume RNG')


def photometry(flux=None, wave=None, response_wave=None, response=None, **kwargs):
    wave = np.asarray([3000., 4000., 5000., 6000., 7000.]) if wave is None else wave
    flux = np.full(len(wave), 1.e-29) if flux is None else flux
    response_wave = [4000., 6000.] if response_wave is None else response_wave
    response = [.5, .5] if response is None else response
    parameters = dict(area_cm2=10000., exposure_s=100., n_exposures=2,
                      aperture_fraction=.8, n_pixels=9., sky_rate_e_s_pixel=.1,
                      dark_current_e_s_pixel=.02, read_noise_e_pixel=5.,
                      rng=ExpectedCounts())
    parameters.update(kwargs)
    return measure_photometry(wave, flux, response_wave, response, **parameters)


def test_flat_fnu_ab_and_photon_electron_calibration():
    result = photometry()
    efficiency = .5 * np.log(6000. / 4000.)
    expected_counts = 10000. * 200. * .8 / PLANCK_ERG_S * 1.e-29 * efficiency
    assert result['valid']
    assert result['coverage_fraction'] == 1.
    assert result['model_flux_ujy'] == pytest.approx(1., rel=2.e-13)
    assert result['model_mag'] == pytest.approx(23.9, abs=1.e-12)
    assert result['response_efficiency'] == pytest.approx(efficiency, rel=2.e-12)
    assert result['source_counts_e'] == pytest.approx(expected_counts, rel=2.e-12)
    assert result['observed_flux_ujy'] == pytest.approx(1., rel=2.e-12)


def test_linear_spectrum_uses_response_weight_not_arithmetic_mean():
    wave = np.asarray([3000., 3900., 4100., 4300., 5700., 6100., 7000.])
    flux = 1.e-29 * (wave / 2000. - 1.)
    expected_fnu = 1.e-29 * ((6000. - 4000.) / (2000. * np.log(6000. / 4000.)) - 1.)
    result = photometry(wave=wave, flux=flux)
    assert result['model_flux_ujy'] == pytest.approx(expected_fnu / 1.e-29, rel=3.e-12)
    assert not np.isclose(result['model_flux_ujy'], np.mean(flux / 1.e-29), rtol=1.e-3)


def test_nonuniform_knots_preserve_sampled_narrow_line_and_filter_shape():
    wave = np.asarray([3500., 4000., 5100., 5199., 5200., 5201., 5400., 6500.])
    flux = np.zeros_like(wave)
    flux[4] = 5.e-27
    response_wave = np.asarray([4000., 4300., 5000., 5500., 6000.])
    response = np.asarray([0., .2, .6, .4, 0.])
    result = photometry(wave=wave, flux=flux, response_wave=response_wave, response=response)
    # Independent dense numerical integration of the explicitly sampled SED.
    fine = np.unique(np.r_[np.linspace(4000., 6000., 200001), wave[(wave >= 4000.) & (wave <= 6000.)]])
    f = np.interp(fine, wave, flux)
    transmission = np.interp(fine, response_wave, response)
    reference = trapezoid(f * transmission / fine, fine) / trapezoid(transmission / fine, fine)
    assert result['model_flux_ujy'] == pytest.approx(reference / 1.e-29, rel=2.e-7)
    assert result['source_counts_e'] > 0


def test_same_linear_sed_on_different_nonuniform_grids():
    first = np.asarray([3000., 4000., 6000., 7000.])
    second = np.asarray([3000., 3899., 4001., 4020., 4800., 6001., 7000.])
    a = photometry(wave=first, flux=first * 1.e-33)
    b = photometry(wave=second, flux=second * 1.e-33)
    assert a['model_flux_ujy'] == pytest.approx(b['model_flux_ujy'], rel=3.e-12)


def test_aperture_counts_and_total_flux_share_the_same_correction():
    a, b = photometry(aperture_fraction=.8), photometry(aperture_fraction=.4)
    assert b['source_counts_e'] == pytest.approx(.5 * a['source_counts_e'])
    assert a['model_flux_ujy'] == pytest.approx(b['model_flux_ujy'])
    assert a['observed_flux_ujy'] == pytest.approx(b['observed_flux_ujy'])
    assert b['flux_error_ujy'] > a['flux_error_ujy']


def test_exposure_splitting_and_read_noise_variance():
    kwargs = dict(n_pixels=12.5, sky_rate_e_s_pixel=.2, dark_current_e_s_pixel=.02,
                  read_noise_e_pixel=5.)
    a = photometry(exposure_s=150., n_exposures=4, **kwargs)
    b = photometry(exposure_s=75., n_exposures=8, **kwargs)
    assert a['total_exposure_s'] == 600.
    assert a['sky_counts_e'] == pytest.approx(12.5 * .2 * 600.)
    assert a['dark_counts_e'] == pytest.approx(12.5 * .02 * 600.)
    assert a['read_variance_e2'] == pytest.approx(12.5 * 4 * 25.)
    assert a['variance_e2'] == pytest.approx(a['source_counts_e'] + 1500. + 150. + 1250.)
    assert b['source_counts_e'] == pytest.approx(a['source_counts_e'])
    assert b['read_variance_e2'] == pytest.approx(2 * a['read_variance_e2'])
    assert b['expected_snr'] < a['expected_snr']


@pytest.mark.parametrize('wave', [[4500., 5000., 6500.], [1000., 1500.], [4000., 5000., 5500.]])
def test_incomplete_response_coverage_is_invalid_without_rng(wave):
    result = photometry(wave=wave, flux=np.ones(len(wave)) * 1.e-29, rng=NoDraw())
    assert not result['valid'] and not result['detected']
    assert result['flag'] == FLAG_INCOMPLETE_COVERAGE
    assert result['reason'] == 'incomplete_spectral_coverage'
    assert 0 <= result['coverage_fraction'] < 1
    assert np.isnan(result['model_flux_ujy'])
    assert np.isnan(result['observed_flux_ujy'])


def test_zero_response_tails_do_not_require_irrelevant_sed_coverage():
    result = photometry(wave=[4000., 5000., 6000.], flux=[1.e-29] * 3,
                        response_wave=[1000., 3000., 4000., 5000., 6000., 7000., 9000.],
                        response=[0., 0., 0., .5, 0., 0., 0.])
    assert result['valid'] and result['coverage_complete']


def test_negative_measured_flux_is_retained_without_magnitude():
    class ZeroPoisson(ExpectedCounts):
        def poisson(self, mean):
            return 0
    result = photometry(flux=np.zeros(5), area_cm2=1., exposure_s=1., n_exposures=1,
                        n_pixels=1., sky_rate_e_s_pixel=1., dark_current_e_s_pixel=0.,
                        read_noise_e_pixel=0., rng=ZeroPoisson())
    assert result['valid'] and not result['detected']
    assert result['observed_flux_ujy'] < 0 and result['snr'] < 0
    assert result['flag'] & FLAG_NONPOSITIVE_OBSERVED
    assert np.isnan(result['observed_mag']) and np.isnan(result['observed_mag_error'])
    assert result['model_flux_ujy'] == 0 and result['expected_snr'] == 0


def test_systematic_is_recorded_separately_and_does_not_draw_noise():
    first = photometry(rng=np.random.RandomState(13), systematic_mag=0.)
    second = photometry(rng=np.random.RandomState(13), systematic_mag=.1)
    assert first['observed_flux_ujy'] == second['observed_flux_ujy']
    assert first['flux_error_ujy'] == second['flux_error_ujy']
    assert second['systematic_flux_error_ujy'] == pytest.approx(np.log(10.) / 2.5 * .1)
    assert second['total_flux_error_ujy'] == pytest.approx(np.hypot(
        second['flux_error_ujy'], second['systematic_flux_error_ujy']))


def test_background_and_read_noise_realizations_are_unbiased_and_have_expected_variance():
    rng = np.random.RandomState(216)
    observations = [photometry(flux=np.zeros(5), exposure_s=1., n_exposures=1,
                              n_pixels=3., sky_rate_e_s_pixel=.1,
                              dark_current_e_s_pixel=0., read_noise_e_pixel=2., rng=rng)
                    for _ in range(2048)]
    counts = np.asarray([result['observed_counts_e'] for result in observations])
    # Independent statistical expectations: B=0.3 and Gaussian read variance=12.
    assert abs(counts.mean()) < .25
    assert counts.var(ddof=1) == pytest.approx(12.3, rel=.12)
    assert .4 < np.mean(counts < 0) < .6


def test_rng_reproducibility_and_detection_threshold():
    first = photometry(rng=np.random.RandomState(11), detection_snr=1000.)
    second = photometry(rng=np.random.RandomState(11), detection_snr=.001)
    assert first['observed_flux_ujy'] == second['observed_flux_ujy']
    assert not first['detected'] and second['detected']


def test_astropy_quantities_convert_and_wrong_units_fail():
    result = photometry(wave=np.asarray([300., 400., 500., 600., 700.]) * u.nm,
                        flux=np.ones(5) * u.uJy,
                        response_wave=np.asarray([400., 600.]) * u.nm,
                        response=np.asarray([.5, .5]) * u.dimensionless_unscaled,
                        area_cm2=1. * u.m ** 2, exposure_s=100. * u.s,
                        n_pixels=9. * u.pixel,
                        sky_rate_e_s_pixel=.1 * u.electron / (u.s * u.pixel),
                        dark_current_e_s_pixel=.02 * u.electron / (u.s * u.pixel),
                        read_noise_e_pixel=5. * u.electron)
    assert result['model_flux_ujy'] == pytest.approx(1.)
    with pytest.raises(ValueError, match='units'):
        photometry(flux=np.ones(5) * u.erg / (u.s * u.cm ** 2 * u.AA))
    with pytest.raises(ValueError, match='units'):
        photometry(exposure_s=100. * u.m)


@pytest.mark.parametrize('kwargs', [dict(n_exposures=1.5), dict(n_exposures=True),
                                   dict(n_pixels=0.), dict(aperture_fraction=1.1),
                                   dict(sky_rate_e_s_pixel=-1.), dict(response=[0., 0.]),
                                   dict(response=[-.1, .5]), dict(response=[1.1, .5]),
                                   dict(wave=[3000., 4000., 4000., 6000., 7000.])])
def test_invalid_detector_or_grid_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        photometry(**kwargs)


@pytest.mark.parametrize('flux,reason', [([1.e-29, 1.e-29, np.nan, 1.e-29, 1.e-29], 'nonfinite_model_flux'),
                                       ([1.e-29, 1.e-29, -1.e-30, 1.e-29, 1.e-29], 'negative_model_flux')])
def test_bad_model_is_flagged_not_clipped(flux, reason):
    result = photometry(flux=flux, rng=NoDraw())
    assert not result['valid'] and result['reason'] == reason
    assert np.isnan(result['observed_flux_ujy'])
