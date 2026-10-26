"""Experimental positive Gaussian-mixture marginals of a Sersic source.

This approximates the source; it does not alter a Sersic index or clip FFT
tails. Numerical L1 diagnostics are estimates, not certified bounds. Output
bin probabilities retain finite-window losses. No production entry uses this
module. Import has no configuration/data I/O and no random draws.
"""
from copy import deepcopy
from functools import lru_cache
import math
from numbers import Integral, Real
import platform
import sys
import warnings

import numpy as np
import scipy
from scipy.optimize import nnls
from scipy.special import gammainc, gammaincc, gammaincinv, gammainccinv

from cess.morphology.reference import sersic_pixel_reference
from cess.spectra.numerics import trapezoid


MGE_SCHEMA = 'cess.positive_sersic_marginals'
MGE_REVISION = '20261022-C10'


class MixtureConvergenceError(RuntimeError):
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = {} if diagnostics is None else diagnostics


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(name + ' must be a positive real scalar, excluding booleans')
    try:
        value = float(value)
    except (ValueError, OverflowError):
        raise ValueError(name + ' must be representable and finite')
    if not math.isfinite(value) or value <= 0.:
        raise ValueError(name + ' must be positive and finite')
    return value


def _source_log_density(log_radius, n, b):
    """Unit-Re source probability per d(log r), not surface brightness."""
    return np.exp(2.*n*math.log(b)-math.log(n)-math.lgamma(2.*n)
                  +2.*log_radius-b*np.exp(log_radius/n))


def _component_log_density(log_radius, log_sigmas):
    # Unit-flux 2D Gaussian: dP/dlog(r)=(r/sigma)^2 exp[-r^2/(2sigma^2)].
    z = 2.*(np.asarray(log_radius)[..., None]-log_sigmas)
    return np.exp(z-.5*np.exp(z))


@lru_cache(maxsize=32)
def _fit(n, tolerance, tail, maximum):
    b = float(gammaincinv(2.*n, .5))
    lower_t = float(gammaincinv(2.*n, tail))
    upper_t = float(gammainccinv(2.*n, tail))
    lower, upper = n*math.log(lower_t/b), n*math.log(upper_t/b)
    source_omitted = float(gammainc(2.*n, lower_t)+gammaincc(2.*n, upper_t))
    radii = np.exp([lower, upper])
    training = np.linspace(lower, upper, 2049)
    desired = np.append(_source_log_density(training, n, b), 10.)
    steps = sorted(set([step for step in (32, 64, 128, 160) if step <= maximum]+[maximum]))
    attempts = []
    for count in steps:
        # Include the exact n=0.5 width; all other widths are logarithmically
        # spaced. The unit-total row constrains the whole source, not a window.
        logs = np.unique(np.append(np.linspace(lower-math.log(4.), upper, count),
                                    -.5*math.log(2.*math.log(2.))))
        matrix = np.vstack((_component_log_density(training, logs),
                            np.ones((1, logs.size))*10.))
        attempt = {'width_grid_count': count, 'candidate_components': int(logs.size),
                   'nnls_max_iterations': int(20*logs.size)}
        attempts.append(attempt)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                coefficients, residual = nnls(matrix, desired, maxiter=20*logs.size)
            except Exception as error:
                attempt.update(status='failed', reason=str(error))
                raise MixtureConvergenceError('Positive source fit failed; no fallback',
                                               {'attempts': attempts}) from error
        if caught:
            attempt['warnings'] = [str(item.message) for item in caught]
            raise MixtureConvergenceError('Positive source fit emitted a warning', {'attempts': attempts})
        total = float(np.sum(coefficients, dtype=np.float64))
        if (not np.all(np.isfinite(coefficients)) or np.any(coefficients < 0.) or
                not math.isfinite(total) or total <= 0. or not math.isfinite(residual)):
            raise MixtureConvergenceError('Invalid NNLS source coefficients; no clipping', {'attempts': attempts})
        active = coefficients > 0.
        sigmas, weights = np.exp(logs[active]), coefficients[active]/total
        estimates = []
        for points in (4097, 16385):
            grid = np.linspace(lower, upper, points)
            mixture = _component_log_density(grid, np.log(sigmas)).dot(weights)
            estimates.append(float(trapezoid(np.abs(_source_log_density(grid, n, b)-mixture), grid)))
        mixture_outside = float(np.dot(weights, -np.expm1(-.5*(radii[0]/sigmas)**2)
                                       +np.exp(-.5*(radii[1]/sigmas)**2)))
        estimated_l1 = estimates[-1]+source_omitted+mixture_outside
        refinement = abs(estimates[-1]-estimates[0])
        attempt.update(status='evaluated', active_components=int(weights.size),
            nnls_residual_norm=float(residual), raw_coefficient_sum=total,
            radial_L1_grid_estimates=estimates, radial_L1_refinement_difference=refinement,
            source_outside_fit_domain_probability=source_omitted,
            mixture_outside_fit_domain_probability=mixture_outside,
            estimated_radial_L1_including_outside=estimated_l1)
        if not all(math.isfinite(v) and v >= 0. for v in estimates+[mixture_outside, estimated_l1]):
            raise MixtureConvergenceError('Invalid radial probability diagnostics', {'attempts': attempts})
        if estimated_l1 <= tolerance and refinement <= tolerance*.1:
            weights.setflags(write=False)
            sigmas.setflags(write=False)
            metadata = {'method': 'NNLS positive Gaussians fit to circular source probability per log radius',
                'b_n': b, 'log_radius_over_re_fit_domain': [lower, upper],
                'fit_training_samples': 2049, 'L1_check_samples': [4097, 16385],
                'requested_estimated_L1_tolerance': tolerance,
                'requested_source_tail_each_end': tail,
                'maximum_width_grid_count': maximum, 'attempts': attempts,
                'estimated_radial_L1_including_outside': estimated_l1,
                'radial_L1_refinement_difference': refinement,
                'source_outside_fit_domain_probability': source_omitted,
                'mixture_outside_fit_domain_probability': mixture_outside,
                'source_mixture_normalization': 'positive NNLS coefficients divided by their full-source sum before validation; no detector window involved',
                'normalized_weight_sum': float(weights.sum()),
                'error_interpretation': 'radial L1 grid integrals/refinement are numerical estimates, not certified continuum bounds',
                'L1_transport': 'true source L1 bounds the L1 after equal area-preserving shear, normalized positive PSF convolution, projection and bin integration; sampled estimates are not that proof',
                'motivation_reference': 'https://arxiv.org/abs/astro-ph/0201430',
                'reference_scope': 'Gaussian expansions/NNLS motivation; this probability-density fit is independently constructed'}
            return sigmas, weights, metadata
    raise MixtureConvergenceError('Estimated source L1 target not achieved within the component budget',
                                    {'attempts': attempts, 'requested_estimated_L1_tolerance': tolerance})


def sersic_mge_marginals(n, re_pixel, pa_deg, axis_ratio,
                         psf_fwhm_arcsec, pixel_scale_arcsec,
                         x_edges_pixel, y_edges_pixel=None, *,
                         l1_tolerance=1.e-7, tail_probability=1.e-12,
                         max_components=160):
    """Return experimental, positive full-source bin probability marginals.

    Inputs have the same physical definitions as sersic_pixel_reference: Re is
    circularized half-light radius [pixel], PA image-x angle [degree], q=b/a,
    FWHM [arcsec], scale [arcsec/pixel], finite ordered edges centered at zero.
    Only 0.5<=n<=5 is implemented. Output bins are absolute probabilities and
    never window-renormalized. The source is an approximate positive mixture,
    not an exact Sersic renderer or a finite two-dimensional aperture model.

    l1_tolerance applies to an estimated global radial probability discrepancy
    including separately recorded outside-domain mass; it is not a certified
    bound or a relative tail accuracy target. NNLS uses at most 160 log-spaced
    widths plus the exact Gaussian-source width, with fixed training/check
    resource limits. Failure raises MixtureConvergenceError with diagnostics.
    Up to 32 small source fits are cached; returned arrays/metadata are owned.
    """
    n = _positive(n, 'n')
    radius = _positive(re_pixel, 're_pixel')
    tolerance = _positive(l1_tolerance, 'l1_tolerance')
    tail = _positive(tail_probability, 'tail_probability')
    if not .5 <= n <= 5.:
        raise ValueError('MGE source index n must lie in [0.5, 5]')
    if not 1.e-14 <= tail <= 1.e-6:
        raise ValueError('tail_probability must lie in [1e-14, 1e-6] per source radial end')
    if (isinstance(max_components, (bool, np.bool_)) or not isinstance(max_components, Integral)
            or not 16 <= max_components <= 160):
        raise ValueError('max_components must be an integer in [16, 160]')
    # Validate physical geometry/units/edges before spending the fit budget.
    geometry = sersic_pixel_reference(.5, radius, pa_deg, axis_ratio,
        psf_fwhm_arcsec, pixel_scale_arcsec, x_edges_pixel, y_edges_pixel)
    sigmas, weights, fit = _fit(n, tolerance, tail, int(max_components))
    px = np.zeros_like(geometry['x_probability'])
    py = np.zeros_like(geometry['y_probability'])
    outside = {'x': 0., 'y': 0.}
    for sigma, weight in zip(sigmas, weights):
        component_re = math.sqrt(2.*math.log(2.))*float(sigma)*radius
        component = sersic_pixel_reference(.5, component_re, pa_deg, axis_ratio,
            psf_fwhm_arcsec, pixel_scale_arcsec, geometry['x_edges_pixel'], geometry['y_edges_pixel'])
        px += weight*component['x_probability']
        py += weight*component['y_probability']
        for axis in outside:
            outside[axis] += float(weight)*component['metadata']['windows'][axis]['outside_probability_integrated']
    windows = {}
    accounting_tolerance = 256.*np.finfo(float).eps*(len(px)+len(py)+len(weights))
    for axis, values in (('x', px), ('y', py)):
        if not np.all(np.isfinite(values)) or np.any(values < 0.) or np.any(values > 1.):
            raise MixtureConvergenceError('Invalid mixture bin probability; no clipping')
        mass = float(np.sum(values, dtype=np.float64))
        accounted = mass+outside[axis]
        if not math.isfinite(accounted) or abs(accounted-1.) > accounting_tolerance:
            raise MixtureConvergenceError('Mixture probability accounting failed', {'axis': axis, 'accounted': accounted})
        windows[axis] = {'returned_probability': mass,
                         'outside_probability_integrated': outside[axis], 'probability_accounted': accounted}
    inputs = deepcopy(geometry['metadata']['input'])
    inputs['n'] = n
    metadata = {'schema': MGE_SCHEMA, 'version': 1, 'algorithm_revision': MGE_REVISION,
        'status': 'experimental_source_approximation_not_production_renderer', 'input': inputs,
        'axis_scales': deepcopy(geometry['metadata']['axis_scales']),
        'normalization': {'source_total_probability': 1., 'orthogonal_domain': 'all real',
                          'bin_values': 'absolute unit-source probability per bin; not density',
                          'window_renormalized': False},
        'models': {'source': 'positive 2D Gaussian approximation to the untruncated circularized Sersic source',
                   'psf': 'adopted circular Gaussian',
                   'psf_sigma_pixel': geometry['metadata']['models']['psf_sigma_pixel'],
                   'pixel': 'analytical Gaussian CDF bin integrals, reused from C9'},
        'source_fit': deepcopy(fit), 'windows': windows,
        'mixture': {'sigma_over_re': sigmas.tolist(), 'weights': weights.tolist()},
        'accounting_tolerance': accounting_tolerance,
        'software': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
                     'machine': platform.machine()},
        'limitations': ['source approximation; L1 estimates/refinement are not certified continuum bounds',
            'small absolute discrepancies do not guarantee relative accuracy in very faint tails',
            'two marginal window probabilities do not determine a joint 2D aperture fraction',
            'adopted Gaussian PSF is not an instrument calibration; no velocity field is modeled',
            'not connected to profile generator, production lookup or source extraction fractions',
            'no finite-window renormalization, radius cap, empirical replacement or negative clipping']}
    return {'x_edges_pixel': geometry['x_edges_pixel'], 'y_edges_pixel': geometry['y_edges_pixel'],
            'x_probability': px, 'y_probability': py,
            'mixture_sigma_over_re': sigmas.copy(), 'mixture_weights': weights.copy(),
            'metadata': metadata}
