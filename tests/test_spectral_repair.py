"""Independent photon/bin tests for the conservative one-dimensional operator.

These synthetic spectra explicitly declare delta or a known template LSF.
They do not assign an LSF to the real BayeSED catalogue. All tests are small;
the Gaussian is finely sampled near the line, with coarse zero-flux padding.
"""

import importlib.util
from pathlib import Path
import sys
import warnings
import numpy as np
from scipy.integrate import trapezoid
import pytest
from scipy.integrate import quad
from scipy.special import ndtr


CORE = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(CORE.parent / 'src'))
C_AA_S = 2.99792458e18
FWHM_SIGMA = 2.3548200450309493
LINE_ENERGY = 1.e-16


@pytest.fixture(scope='module')
def core():
    spec = importlib.util.spec_from_file_location('cess_spectral_repair_main', str(CORE.parent / 'src/cess/engine.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def simulator(core, template=None, policy='error'):
    from cess.config.loader import load_config
    config = load_config()
    config['template_lsf'] = {'mode': 'delta'} if template is None else template
    config['template_lsf_broader_policy'] = policy
    return core.emulator(config=config)


def edges_from_centers(centers):
    # Independent test definition: midpoint edges and half-step outer edges.
    return np.r_[centers[0] - .5 * (centers[1] - centers[0]),
                 .5 * (centers[:-1] + centers[1:]),
                 centers[-1] + .5 * (centers[-1] - centers[-2])]


def old_output_centers(config, band):
    prefix = band.lower()
    n_coarse = int((config[prefix + '_end_wl'] - config[prefix + '_start_wl']) /
                   config[prefix + '_dlambda']) + 1
    n_out = round(n_coarse * config[prefix + '_ree80'] / config['arcsecperpix'])
    return np.linspace(config[prefix + '_min_wl'], config[prefix + '_max_wl'], n_out)


def gaussian_input(center, sigma, redshift, samples_per_sigma=50):
    """Known Gaussian in observed f_lambda; no luminosity/redshift rescaling."""
    fine = np.linspace(center - 12. * sigma, center + 12. * sigma,
                       24 * samples_per_sigma + 1)
    padding = np.linspace(500., 14000., 136)
    padding = padding[(padding < fine[0]) | (padding > fine[-1])]
    observed = np.sort(np.r_[padding, fine])
    flam = LINE_ENERGY / (np.sqrt(2. * np.pi) * sigma) * np.exp(
        -.5 * ((observed - center) / sigma) ** 2)
    fnu = flam * observed ** 2 / C_AA_S
    # Independently denser analytic truth; not the later coarse source samples.
    dense = np.linspace(center - 12. * sigma, center + 12. * sigma, 4801)
    dense_flam = LINE_ENERGY / (np.sqrt(2. * np.pi) * sigma) * np.exp(
        -.5 * ((dense - center) / sigma) ** 2)
    dense_q = dense_flam * dense / C_AA_S
    truth = trapezoid(dense_q, dense)
    assert truth == pytest.approx(LINE_ENERGY * center / C_AA_S, rel=2.e-13)
    return observed / (1. + redshift), fnu, truth


def photon_bins(centers, flux):
    return flux * np.diff(edges_from_centers(centers)) / centers


@pytest.mark.parametrize('band', ['GU', 'GV', 'GI'])
@pytest.mark.parametrize('width_in_coarse_steps', [.05, .5, 2.])
@pytest.mark.parametrize('phase', [0., .25, .5, .75])
@pytest.mark.parametrize('redshift', [.1, 1.])
def test_gaussian_photons_conserved_across_band_phase_width_and_redshift(
        core, band, width_in_coarse_steps, phase, redshift):
    e = simulator(core)
    config = e.config
    prefix = band.lower()
    coarse_step = config[prefix + '_dlambda']
    midpoint = .5 * (config[prefix + '_min_wl'] + config[prefix + '_max_wl'])
    coarse_center = config[prefix + '_start_wl'] + round(
        (midpoint - config[prefix + '_start_wl']) / coarse_step) * coarse_step
    center = coarse_center + phase * coarse_step
    rest, fnu, truth = gaussian_input(center, width_in_coarse_steps * coarse_step, redshift)
    result = e.kernels_convolve(rest, fnu[None, :], np.asarray([redshift]),
                                band, config[prefix + '_ree80'])
    assert len(result) == 4
    output_wave, output, valid, invalid = result
    np.testing.assert_array_equal(output_wave, old_output_centers(config, band))
    assert output.shape == (1, len(output_wave))
    np.testing.assert_array_equal(valid, [True])
    np.testing.assert_array_equal(invalid, [False])
    assert np.min(output) >= 0.
    assert photon_bins(output_wave, output[0]).sum() == pytest.approx(truth, rel=2.e-6)


def reference_bin_fractions(edges, center, input_sigma, additional_sigma):
    """Independent normalized analytic input and continuous Gaussian CDF/quad."""
    fractions = []
    norm = np.sqrt(2. * np.pi) * input_sigma
    for left, right in zip(edges[:-1], edges[1:]):
        def integrand(wavelength):
            # q=fnu/lambda for a Gaussian f_lambda has an extra wavelength.
            density = wavelength / (center * norm) * np.exp(
                -.5 * ((wavelength - center) / input_sigma) ** 2)
            sigma = additional_sigma(wavelength) if callable(additional_sigma) else additional_sigma
            probability = (ndtr((right - wavelength) / sigma) -
                           ndtr((left - wavelength) / sigma))
            return density * probability
        value, _ = quad(integrand, center - 12. * input_sigma,
                        center + 12. * input_sigma, epsabs=2.e-12, epsrel=2.e-11,
                        points=[center], limit=120)
        fractions.append(value)
    return np.asarray(fractions)


def test_narrow_line_shape_and_centroid_against_independent_quad(core):
    e = simulator(core)
    redshift, center, sigma_line = .5, 5203.1, .9
    rest, fnu, truth = gaussian_input(center, sigma_line, redshift, samples_per_sigma=100)
    wave, result, valid, _ = e.kernels_convolve(rest, fnu[None, :], np.asarray([redshift]),
                                               'GV', e.config['gv_ree80'])
    assert valid[0]
    expected = reference_bin_fractions(edges_from_centers(wave), center, sigma_line,
                                       e.config['gv_dlambda'] / FWHM_SIGMA)
    actual = photon_bins(wave, result[0]) / truth
    np.testing.assert_allclose(actual, expected, rtol=5.e-6, atol=3.e-9)
    actual_centroid = np.sum(wave * actual) / np.sum(actual)
    expected_centroid = np.sum(wave * expected) / np.sum(expected)
    assert actual_centroid == pytest.approx(expected_centroid, abs=2.e-4)


def test_out_of_band_padding_does_not_change_flux_or_lsf(core):
    e = simulator(core)
    redshift = .5
    rest, flux, _ = gaussian_input(8141.3, 7., redshift)
    first = e.kernels_convolve(rest, flux[None, :], np.asarray([redshift]), 'GI', e.config['gi_ree80'])
    padded_wave = np.r_[1., 2., rest]
    padded_flux = np.r_[0., 0., flux]
    second = e.kernels_convolve(padded_wave, padded_flux[None, :], np.asarray([redshift]),
                                'GI', e.config['gi_ree80'])
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_allclose(first[1], second[1], rtol=1.e-10, atol=1.e-300)


def test_zero_sigma_conservative_piecewise_linear_fnu_resampling():
    from cess.spectra.integration import convolve_to_bins
    centers = np.asarray([4000., 4100., 4300., 4600.])
    edges = edges_from_centers(centers)
    wave = np.asarray([3800., 3900., 4050., 4210., 4500., 4800.])
    # Exactly linear fnu, so its integral divided by lambda is analytic.
    constant, slope = 2.e-29, 1.e-33
    fnu = constant + slope * wave
    output, _ = convolve_to_bins(wave, fnu, centers, 0.)
    expected_integrals = constant * np.log(edges[1:] / edges[:-1]) + slope * np.diff(edges)
    np.testing.assert_allclose(photon_bins(centers, output), expected_integrals, rtol=2.e-12)


def test_zero_sigma_preserves_weak_continuum_after_strong_line():
    from cess.spectra.integration import convolve_to_bins
    wave = np.asarray([3500., 3999., 4000., 4001., 4002., 4500., 5000., 5500., 6500.])
    fnu = np.full(len(wave), 1.e-20)
    fnu[2] = 1.e12
    centers = np.arange(3800., 6200.1, 10.)
    output, _ = convolve_to_bins(wave, fnu, centers, 0.)
    edges = edges_from_centers(centers)
    widths = np.diff(edges)
    weak = centers >= 4200.
    # Each remote bin contains only flat fnu. Its local analytic photon
    # integral must survive, even when a preceding line is 1e32 times brighter.
    expected = (1.e-20 * centers[weak] / widths[weak]
                * np.log1p(widths[weak] / edges[:-1][weak]))
    assert np.all(output[weak] > 0.)
    np.testing.assert_allclose(output[weak], expected, rtol=2.e-12, atol=0.)


def test_resolving_power_zero_crossing_is_stable_and_bounds_subdivision():
    from cess.spectra.integration import convolve_to_bins
    coarse = np.asarray([5500., 9700., 9800., 9899.,
                         9900. - 1.e-8, 9900. + 1.e-8, 9950., 10700.])
    fine = np.unique(np.r_[coarse, np.linspace(9700., 10000., 3001)])
    centers = np.linspace(5800., 10500., 498)

    def sigma(wavelength):
        return np.sqrt(np.maximum(33. ** 2 - (wavelength / 300.) ** 2, 0.)) / FWHM_SIGMA

    outputs = []
    for native_wave in (coarse, fine):
        # Both native grids represent exactly the same flat fnu. The square
        # root reaches zero at 9900 AA without requiring a singular mesh.
        output, diagnostics = convolve_to_bins(
            native_wave, np.ones(len(native_wave)), centers, sigma,
            max_sigma=33. / FWHM_SIGMA, max_subintervals=10000)
        assert np.isfinite(output).all()
        assert 0 < diagnostics['subinterval_count'] <= 10000
        outputs.append(output)
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=1.e-10, atol=1.e-12)


def test_partial_coverage_invalidates_row_without_extrapolating(core):
    e = simulator(core)
    rest = np.linspace(1500., 6200., 101)
    output_wave, output, valid, invalid = e.kernels_convolve(
        rest, np.ones((2, len(rest))), np.asarray([.1, .01]), 'GV', e.config['gv_ree80'])
    np.testing.assert_array_equal(valid, [True, False])
    np.testing.assert_array_equal(invalid, [False, True])
    assert np.isfinite(output[0]).all()
    assert np.isnan(output[1]).all()


def test_invalid_source_rows_stay_aligned_and_empty_batch(core):
    e = simulator(core)
    rest = np.linspace(500., 14000., 136)
    flux = np.ones((4, len(rest)))
    flux[1, 30] = np.nan
    flux[2, 50] = np.inf
    redshift = np.asarray([.1, .2, .3, -.1])
    wave, output, valid, invalid = e.kernels_convolve(rest, flux, redshift, 'GV', e.config['gv_ree80'])
    np.testing.assert_array_equal(valid, [True, False, False, False])
    np.testing.assert_array_equal(invalid, ~valid)
    assert np.isnan(output[~valid]).all()
    _, empty, a, b = e.kernels_convolve(rest, np.ones((0, len(rest))), np.zeros(0),
                                       'GV', e.config['gv_ree80'])
    assert empty.shape == (0, len(wave)) and a.shape == b.shape == (0,)


@pytest.mark.parametrize('redshift', [.1, 1.])
def test_known_rest_fwhm_uses_observed_redshift_scaling(core, redshift):
    rest_fwhm = 8.
    e = simulator(core, template={'mode': 'fwhm_rest', 'value': rest_fwhm})
    template_observed = rest_fwhm * (1. + redshift)
    input_sigma = template_observed / FWHM_SIGMA
    center = 5201.2
    rest, fnu, truth = gaussian_input(center, input_sigma, redshift, samples_per_sigma=100)
    wave, output, valid, _ = e.kernels_convolve(rest, fnu[None, :], np.asarray([redshift]),
                                               'GV', e.config['gv_ree80'])
    assert valid[0]
    additional_sigma = np.sqrt(e.config['gv_dlambda'] ** 2 - template_observed ** 2) / FWHM_SIGMA
    reference = reference_bin_fractions(edges_from_centers(wave), center, input_sigma,
                                        additional_sigma)
    np.testing.assert_allclose(photon_bins(wave, output[0]) / truth, reference,
                               rtol=5.e-5, atol=1.e-8)


def test_overbroad_template_error_or_explicit_preserve_warning(core):
    redshift = .5
    config = {'mode': 'fwhm_rest', 'value': 100.}
    e = simulator(core, template=config, policy='error')
    centers = old_output_centers(e.config, 'GV')
    center = centers[np.argmin(abs(centers - 5200.))]
    input_sigma = config['value'] * (1. + redshift) / FWHM_SIGMA
    rest, flux, truth = gaussian_input(center, input_sigma, redshift, samples_per_sigma=100)
    with pytest.raises(ValueError, match='broader'):
        e.kernels_convolve(rest, flux[None, :], np.asarray([redshift]), 'GV', e.config['gv_ree80'])
    e = simulator(core, template=config, policy='preserve')
    with pytest.warns(RuntimeWarning, match='broader'):
        wave, output, valid, _ = e.kernels_convolve(rest, flux[None, :], np.asarray([redshift]),
                                                   'GV', e.config['gv_ree80'])
    assert valid[0]
    fractions = photon_bins(wave, output[0]) / truth
    assert fractions.sum() == pytest.approx(1., rel=2.e-6)
    # Analytic integral of q proportional to lambda times a Gaussian f_lambda.
    edges = edges_from_centers(wave)
    a, b = (edges[:-1] - center) / input_sigma, (edges[1:] - center) / input_sigma
    normal_a = np.exp(-.5 * a ** 2) / np.sqrt(2. * np.pi)
    normal_b = np.exp(-.5 * b ** 2) / np.sqrt(2. * np.pi)
    reference = ndtr(b) - ndtr(a) + input_sigma / center * (normal_a - normal_b)
    np.testing.assert_allclose(fractions, reference, rtol=5.e-5, atol=1.e-8)


def test_known_resolving_power_lambda_dependent_kernel_against_quad(core):
    resolving_power = 1000.
    e = simulator(core, template={'mode': 'resolving_power', 'value': resolving_power})
    center, redshift = 5202.7, .6
    input_sigma = center / resolving_power / FWHM_SIGMA
    rest, fnu, truth = gaussian_input(center, input_sigma, redshift, samples_per_sigma=100)
    wave, output, valid, _ = e.kernels_convolve(rest, fnu[None, :], np.asarray([redshift]),
                                               'GV', e.config['gv_ree80'])
    assert valid[0]
    target_fwhm = e.config['gv_dlambda']
    def additional_sigma(wavelength):
        return np.sqrt(target_fwhm ** 2 - (wavelength / resolving_power) ** 2) / FWHM_SIGMA
    reference = reference_bin_fractions(edges_from_centers(wave), center, input_sigma,
                                        additional_sigma)
    np.testing.assert_allclose(photon_bins(wave, output[0]) / truth, reference,
                               rtol=5.e-6, atol=3.e-9)


def test_unknown_lsf_warns_once_and_padding_does_not_supply_a_resolution(core):
    e = simulator(core, template={'mode': 'unknown'})
    redshift = .5
    rest, fnu, _ = gaussian_input(8141.3, 7., redshift)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        first = e.kernels_convolve(rest, fnu[None, :], np.asarray([redshift]),
                                   'GI', e.config['gi_ree80'])
        second = e.kernels_convolve(np.r_[1., 2., rest], np.r_[0., 0., fnu][None, :],
                                    np.asarray([redshift]), 'GI', e.config['gi_ree80'])
    unknown = [warning for warning in captured if 'Template LSF shape is unknown' in str(warning.message)]
    assert len(unknown) == 1 and unknown[0].category is RuntimeWarning
    np.testing.assert_allclose(first[1], second[1], rtol=1.e-10, atol=1.e-300)
    declared_delta = simulator(core)
    third = declared_delta.kernels_convolve(rest, fnu[None, :], np.asarray([redshift]),
                                            'GI', declared_delta.config['gi_ree80'])
    # Unknown is explicitly full additional broadening, not a sampling-based R.
    np.testing.assert_allclose(first[1], third[1], rtol=1.e-10, atol=1.e-300)
