"""Independent photon-response checks; fnu units are arbitrary but consistent.

References integrate the two interpolated functions separately with scipy.quad.
They do not call the implementation's polynomial/basis integration routines.
The common 1/h, collecting area and exposure factors are intentionally omitted.
"""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr

CORE = Path(__file__).resolve().parents[1] / 'CESS_v0.8.6'
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from cess_spectral import convolve_to_bins, integrate_response, response_curve


def independent_edges(centers):
    centers = np.asarray(centers)
    return np.r_[centers[0] - .5 * (centers[1] - centers[0]),
                 .5 * (centers[:-1] + centers[1:]),
                 centers[-1] + .5 * (centers[-1] - centers[-2])]


def reference_integral(wave, fnu, response_wave, transmission, lower, upper,
                       probability=None, extra_knots=()):
    # Explicitly split at all discontinuities of the derivative so a narrow
    # response feature cannot be missed by an adaptive integrator's first nodes.
    knots = np.unique(np.r_[lower, upper, wave, response_wave, extra_knots])
    knots = knots[(knots >= lower) & (knots <= upper)]

    integrals = []
    for a, b in zip(knots[:-1], knots[1:]):
        source_a, source_b = np.interp([a, b], wave, fnu)
        response_a, response_b = np.interp([a, b], response_wave, transmission)

        def integrand(fraction):
            # Local coordinates retain precision for a 0.03 AA response slope
            # at 4005 AA; interpolating at rounded absolute quad nodes can
            # itself introduce relative errors of order 1e-11.
            wavelength = a + fraction * (b - a)
            value = ((source_a + fraction * (source_b - source_a))
                     * (response_a + fraction * (response_b - response_a))
                     / wavelength * (b - a))
            return value if probability is None else value * probability(wavelength)

        integrals.append(quad(integrand, 0., 1., epsabs=2.e-14, epsrel=2.e-12)[0])
    return sum(integrals)


def reference_bins(wave, fnu, response_wave, transmission, centers, sigma):
    edges = independent_edges(centers)
    result = []
    for a, b in zip(edges[:-1], edges[1:]):
        if not callable(sigma) and sigma == 0:
            photons = reference_integral(wave, fnu, response_wave, transmission, a, b)
        else:
            def probability(wavelength):
                width = sigma(wavelength) if callable(sigma) else sigma
                return ndtr((b - wavelength) / width) - ndtr((a - wavelength) / width)
            photons = reference_integral(
                wave, fnu, response_wave, transmission, wave[0], wave[-1],
                probability=probability, extra_knots=np.r_[edges, centers])
        result.append(photons)
    return np.asarray(result) * np.asarray(centers) / np.diff(edges)


def variable_sigma(wavelength):
    return 1.5 + .002 * (np.asarray(wavelength) - 4000.)


def test_response_curve_accepts_zero_and_returns_validated_arrays():
    wave, transmission = response_curve([3900, 4000, 4200], [0, 0, 0])
    np.testing.assert_array_equal(wave, [3900., 4000., 4200.])
    np.testing.assert_array_equal(transmission, [0., 0., 0.])


@pytest.mark.parametrize('wave,transmission', [
    ([3900., 4000.], [-.01, .5]),
    ([3900., 4000.], [.5, 1.01]),
    ([3900., 4000.], [.5, np.nan]),
    ([3900., 4000.], [.5, np.inf]),
    ([3900., 4000.], [.5]),
    ([3900., 4000.], [[.5, .6]]),
    ([4000., 3900.], [.5, .6]),
    ([3900., 3900.], [.5, .6]),
    ([0., 4000.], [.5, .6]),
    ([3900., np.nan], [.5, .6]),
    ([[3900., 4000.]], [.5, .6]),
])
def test_response_curve_rejects_invalid_values_without_clipping(wave, transmission):
    with pytest.raises(ValueError):
        response_curve(wave, transmission)


@pytest.mark.parametrize('transmission', [0., .3, 1.])
@pytest.mark.parametrize('sigma', [0., variable_sigma])
def test_constant_response_scales_the_existing_operator(transmission, sigma):
    wave = np.asarray([3800., 3950., 4000., 4070., 4160., 4300.])
    fnu = np.asarray([.3, 1.1, 2.4, .8, 1.7, .4])
    centers = np.asarray([3980., 4000., 4025., 4060., 4100.])
    plain, plain_diagnostics = convolve_to_bins(wave, fnu, centers, sigma, max_sigma=2.2)
    weighted, diagnostics = convolve_to_bins(
        wave, fnu, centers, sigma, max_sigma=2.2,
        response_wave=[3700., 4400.], response=[transmission, transmission])
    np.testing.assert_allclose(weighted, transmission * plain, rtol=2.e-13, atol=0.)
    assert diagnostics['response_weighted']
    assert diagnostics['response_coverage_complete']
    assert diagnostics['input_photon_integral'] == pytest.approx(
        transmission * plain_diagnostics['input_photon_integral'], rel=2.e-13, abs=0.)
    assert diagnostics['input_unweighted_photon_integral'] == pytest.approx(
        plain_diagnostics['input_photon_integral'], rel=2.e-13)


@pytest.mark.parametrize('sigma', [0., 1.7, variable_sigma])
def test_response_knots_between_sed_nodes_preserve_a_narrow_feature(sigma):
    wave = np.asarray([3800., 4000., 4100., 4300.])
    fnu = np.ones(len(wave))
    response_wave = np.asarray([3700., 4005.08, 4005.11, 4005.14, 4400.])
    transmission = np.asarray([0., 0., .9, 0., 0.])
    centers = np.asarray([3990., 4000., 4010., 4020.])
    assert np.interp(wave, response_wave, transmission).sum() == 0.
    output, _ = convolve_to_bins(wave, fnu, centers, sigma, max_sigma=2.2,
                                 response_wave=response_wave, response=transmission)
    reference = reference_bins(wave, fnu, response_wave, transmission, centers, sigma)
    assert output.max() > 0.
    np.testing.assert_allclose(output, reference, rtol=1.e-10, atol=1.e-15)


@pytest.mark.parametrize('sigma', [0., 2.])
def test_separately_linear_fnu_and_response_have_a_quadratic_product(sigma):
    wave = np.asarray([3900., 4000., 4100., 4200.])
    fnu = np.asarray([0., 0., 1., 1.])
    response_wave = np.asarray([3800., 4000., 4100., 4300.])
    transmission = np.asarray([1., 1., 0., 0.])
    centers = np.asarray([4010., 4040., 4080.])
    # Pre-multiplication at the SED knots yields zero everywhere, although the
    # true product is x*(1-x)>0 on the common 4000--4100 AA interval.
    np.testing.assert_array_equal(fnu * np.interp(wave, response_wave, transmission), 0.)
    output, _ = convolve_to_bins(wave, fnu, centers, sigma,
                                 response_wave=response_wave, response=transmission)
    reference = reference_bins(wave, fnu, response_wave, transmission, centers, sigma)
    assert np.all(output > 0.)
    np.testing.assert_allclose(output, reference, rtol=1.e-10, atol=1.e-13)


@pytest.mark.parametrize('sigma', [0., variable_sigma])
def test_nonuniform_bins_and_wavelength_dependent_response_against_quad(sigma):
    wave = np.asarray([3800., 3950., 4000., 4070., 4160., 4300.])
    fnu = np.asarray([.3, 1.1, 2.4, .8, 1.7, .4])
    response_wave = np.asarray([3700., 3977.3, 4007.9, 4050.1, 4400.])
    transmission = np.asarray([0., .1, .9, .2, 0.])
    centers = np.asarray([3980., 4000., 4025., 4060., 4100.])
    output, _ = convolve_to_bins(wave, fnu, centers, sigma, max_sigma=2.2,
                                 response_wave=response_wave, response=transmission)
    reference = reference_bins(wave, fnu, response_wave, transmission, centers, sigma)
    np.testing.assert_allclose(output, reference, rtol=1.e-10, atol=1.e-13)


@pytest.mark.parametrize('sigma', [0., 2.])
def test_response_coverage_uses_full_bin_edges_and_gaussian_support(sigma):
    wave = np.asarray([3800., 4300.])
    centers = np.asarray([4000., 4020., 4050.])
    edges = independent_edges(centers)
    lower, upper = edges[0] - 8. * sigma, edges[-1] + 8. * sigma
    output, _ = convolve_to_bins(wave, [1., 1.], centers, sigma,
                                 response_wave=[lower, upper], response=[.5, .5])
    assert np.isfinite(output).all()
    for incomplete in ([lower + .01, upper], [lower, upper - .01]):
        # A zero endpoint does not license extrapolation; allowing a partial
        # source SED does not waive calibration-curve coverage either.
        with pytest.raises(ValueError, match='response coverage'):
            convolve_to_bins(wave, [1., 1.], centers, sigma, require_full_coverage=False,
                             response_wave=incomplete, response=[0., 0.])


@pytest.mark.parametrize('kwargs', [{'response_wave': [3800., 4300.]},
                                    {'response': [.5, .5]}])
def test_response_arguments_must_be_supplied_together(kwargs):
    with pytest.raises(ValueError, match='supplied together'):
        convolve_to_bins([3800., 4300.], [1., 1.], [4000., 4020.], 0., **kwargs)


@pytest.mark.parametrize('sigma', [0., variable_sigma])
def test_explicit_none_response_preserves_c1_values_and_diagnostics(sigma):
    args = ([3800., 4000., 4100., 4300.], [1., 2., .5, 1.],
            [3980., 4010., 4060., 4110.], sigma)
    original, first = convolve_to_bins(*args, max_sigma=2.2)
    explicit, second = convolve_to_bins(*args, max_sigma=2.2,
                                       response_wave=None, response=None)
    np.testing.assert_array_equal(original, explicit)
    assert not second['response_weighted']
    for key in ('input_photon_integral', 'output_photon_integral', 'subinterval_count'):
        assert first[key] == second[key]


def test_integrate_response_matches_quad_with_disjoint_knots_and_quadratic_product():
    wave = np.asarray([3900., 4000., 4100., 4200.])
    fnu = np.asarray([0., 0., 1., 1.])
    response_wave = np.asarray([3900., 4000., 4031.17, 4100., 4200.])
    transmission = np.asarray([1., 1., .25, 0., 0.])
    measured = integrate_response(wave, fnu, response_wave, transmission)
    reference = reference_integral(wave, fnu, response_wave, transmission,
                                   response_wave[0], response_wave[-1])
    assert measured > 0.
    assert measured == pytest.approx(reference, rel=2.e-12, abs=0.)


def test_integrate_response_catches_a_feature_between_source_nodes():
    wave = np.asarray([3800., 4300.])
    response_wave = np.asarray([3800., 4005.08, 4005.11, 4005.14, 4300.])
    transmission = np.asarray([0., 0., .9, 0., 0.])
    measured = integrate_response(wave, [2., 2.], response_wave, transmission)
    reference = reference_integral(wave, [2., 2.], response_wave, transmission, 3800., 4300.)
    assert measured > 0.
    assert measured == pytest.approx(reference, rel=2.e-12, abs=0.)


@pytest.mark.parametrize('transmission', [0., .3, 1.])
def test_integrate_response_flat_fnu_has_an_analytic_full_band_integral(transmission):
    measured = integrate_response([3800., 4300.], [2., 2.],
                                   [3900., 3950., 4100., 4200.], [transmission] * 4)
    expected = 2. * transmission * np.log(4200. / 3900.)
    assert measured == pytest.approx(expected, rel=2.e-12, abs=0.)


@pytest.mark.parametrize('source_wave', [[3900.01, 4200.], [3900., 4199.99]])
def test_integrate_response_requires_coverage_even_for_zero_endpoints(source_wave):
    with pytest.raises(ValueError, match='coverage'):
        integrate_response(source_wave, [1., 1.], [3900., 4000., 4200.], [0., 1., 0.])


@pytest.mark.parametrize('source_wave,fnu', [([3900., 4200.], [1.]),
                                             ([3900., 4200.], [1., np.nan])])
def test_integrate_response_rejects_invalid_source_arrays(source_wave, fnu):
    with pytest.raises(ValueError, match='fnu'):
        integrate_response(source_wave, fnu, [3900., 4200.], [.5, .5])
