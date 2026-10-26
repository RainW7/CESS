"""Positive, FFT-free marginal references for Gaussian/exponential Sersic models.

The reference describes an infinite, unit-flux source convolved with a circular
Gaussian PSF and integrated over 1D pixel bins. The orthogonal coordinate is
integrated over all real values, unlike the legacy finite-square stamp. It is
not a replacement renderer, an aperture calibration, or a joint 2D image model.
Only n=0.5 and n=1 are implemented; other indices fail without substitution.
Import performs no configuration, catalogue or calibration-file I/O.
"""
from numbers import Real
import math
import platform
import sys

import numpy as np
import scipy
from scipy.integrate import quad_vec
from scipy.special import erf, gammaincc, gammainccinv, gammaincinv, ndtr


REFERENCE_SCHEMA = 'cess.sersic_marginal_reference'
REFERENCE_VERSION = 1
REFERENCE_REVISION = '20261021-C9'
MAX_BINS_PER_AXIS = 2049


class ReferenceConvergenceError(RuntimeError):
    """The requested numerical reference did not satisfy its convergence checks."""
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = {} if diagnostics is None else diagnostics


def _real(value, name, minimum=0., positive=True):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(name + ' must be a finite real number, excluding booleans')
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError):
        raise ValueError(name + ' must be finite and representable')
    if not math.isfinite(value) or (value <= minimum if positive else value < minimum):
        raise ValueError(name + ' is outside its finite permitted range')
    return value


def _integer(value, name, maximum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 1 <= value <= maximum:
        raise ValueError('{} must be an integer in [1, {}]'.format(name, maximum))
    return int(value)


def _edges(value, name):
    if np.ma.isMaskedArray(value) or hasattr(value, 'unit'):
        raise ValueError(name + ' must be unmasked plain numeric coordinates in pixels')
    raw = np.asarray(value)
    if (raw.ndim != 1 or not 2 <= raw.size <= MAX_BINS_PER_AXIS + 1 or
            raw.dtype.kind not in 'fiu'):
        raise ValueError(name + ' must be a real numeric 1D array with 1..2049 bins')
    if any(isinstance(v, (bool, np.bool_)) for v in np.asarray(value, dtype=object)):
        raise ValueError(name + ' must exclude boolean coordinates')
    values = np.array(raw, dtype=float, copy=True)
    if not np.all(np.isfinite(values)) or np.any(values[1:] <= values[:-1]):
        raise ValueError(name + ' must be finite and strictly increasing')
    return values


def _normal_interval(lower, upper):
    """Stable Gaussian bin mass, including positive tails beyond 9 sigma.

    Evaluate a survival-function difference on the positive side, a CDF
    difference on the negative side, and erf across zero. No negative clipping
    or per-window normalization is applied. Subnormal tails may underflow.
    """
    lower, upper = np.broadcast_arrays(lower, upper)
    result = np.empty(lower.shape, dtype=float)
    right = lower >= 0.
    left = upper <= 0.
    middle = ~(right | left)
    result[right] = ndtr(-lower[right]) - ndtr(-upper[right])
    result[left] = ndtr(upper[left]) - ndtr(lower[left])
    result[middle] = .5 * (erf(upper[middle] / math.sqrt(2.)) - erf(lower[middle] / math.sqrt(2.)))
    return result


def _gaussian_bins(edges, sigma):
    if not math.isfinite(sigma) or sigma <= 0.:
        raise ValueError('Derived Gaussian sigma must be positive and finite in pixels')
    with np.errstate(over='ignore', under='ignore'):
        scaled = edges / sigma
    probabilities = _normal_interval(scaled[:-1], scaled[1:])
    outside = float(ndtr(scaled[0]) + ndtr(-scaled[-1]))
    return probabilities, outside


def sersic_pixel_reference(n, re_pixel, pa_deg, axis_ratio,
                          psf_fwhm_arcsec, pixel_scale_arcsec,
                          x_edges_pixel, y_edges_pixel=None, *,
                          absolute_tolerance=1.e-10,
                          mixing_tail_probability=1.e-12,
                          max_subdivisions=256):
    """Return two absolute, unit-source pixel probability marginals and metadata.

    Re is the circularized half-light radius in pixels. Area-preserving shear
    uses q=b/a and PA from the image x axis, in degrees. PSF FWHM and pixel scale
    are arcsec and arcsec/pixel. Finite increasing bin edges are relative to the
    source center at 0; bins may have unequal widths. If y edges are omitted,
    an independent copy of x edges is used. Each axis permits at most 2049 bins.

    n=0.5 is an analytic Gaussian pixel CDF. n=1 is an exact positive Gaussian
    variance-mixture identity, evaluated with adaptive Gauss-Kronrod quadrature.
    Its latent T~Gamma(3/2,1) is cut at an explicit upper-tail probability; this
    is NOT a physical radius cut. The omitted mixing probability bounds the
    exact per-bin truncation contribution. Quadrature error is an estimate,
    not a certified bound. absolute_tolerance applies to that estimate alone.

    Returned arrays are never normalized by their finite window sums. Outside
    mass is integrated separately. The two marginals do not define a joint 2D
    aperture fraction. No Re caps, Gaussian fallbacks, GEMS draws or FFTs occur.
    Unsupported n raises ValueError. Convergence failure raises
    ReferenceConvergenceError with diagnostics, never a substituted profile.
    """
    n = _real(n, 'n')
    if n not in (.5, 1.):
        raise ValueError('Reference supports only n=0.5 and n=1; no model substitution is used')
    radius = _real(re_pixel, 're_pixel')
    angle = _real(pa_deg, 'pa_deg', minimum=-math.inf, positive=False)
    q = _real(axis_ratio, 'axis_ratio')
    if q > 1.:
        raise ValueError('axis_ratio must lie in (0, 1]')
    fwhm = _real(psf_fwhm_arcsec, 'psf_fwhm_arcsec', positive=False)
    pixel_scale = _real(pixel_scale_arcsec, 'pixel_scale_arcsec')
    tolerance = _real(absolute_tolerance, 'absolute_tolerance')
    tail_request = _real(mixing_tail_probability, 'mixing_tail_probability')
    if tail_request >= .1:
        raise ValueError('mixing_tail_probability must lie in (0, 0.1)')
    limit = _integer(max_subdivisions, 'max_subdivisions', 10000)
    x_edges = _edges(x_edges_pixel, 'x_edges_pixel')
    y_edges = x_edges.copy() if y_edges_pixel is None else _edges(y_edges_pixel, 'y_edges_pixel')
    folded = angle % 180.
    radians = math.radians(folded)
    # Preserve exact quarter-turn geometry even for extremely flattened test
    # sources: cos(pi/2)'s residual must not be amplified by 1/sqrt(q).
    cosine, sine = ((0., 1.) if folded == 90. else
                    (math.cos(radians), math.sin(radians)))
    root_q = math.sqrt(q)
    sx = math.hypot(cosine / root_q, sine * root_q)
    sy = math.hypot(sine / root_q, cosine * root_q)
    sigma_psf = fwhm / pixel_scale / math.sqrt(8. * math.log(2.))
    if (not all(math.isfinite(v) for v in (sx, sy, sigma_psf, radius * sx, radius * sy)) or
            (fwhm > 0. and sigma_psf == 0.)):
        raise ValueError('Derived physical scales must be finite and representable')
    nx, ny = x_edges.size - 1, y_edges.size - 1

    if n == .5:
        sigma_source = radius / math.sqrt(2. * math.log(2.))
        px, outside_x = _gaussian_bins(x_edges, math.hypot(sigma_source * sx, sigma_psf))
        py, outside_y = _gaussian_bins(y_edges, math.hypot(sigma_source * sy, sigma_psf))
        integration = {
            'method': 'analytic_gaussian_pixel_cdf', 'converged': True,
            'variance': '(Re*axis_scale)^2/(2*ln(2)) + sigma_PSF^2',
            'estimated_absolute_error': 0.,
            'error_interpretation': 'no quadrature; floating-point CDF evaluation is not certified exact',
            'omitted_mixing_tail_probability': 0., 'mixing_t_cutoff': None,
            'truncation_variable': None, 'integrated_mixing_probability': 1.,
            'evaluations': 0, 'subdivisions': 0}
    else:
        b1 = float(gammaincinv(2., .5))
        cutoff_t = float(gammainccinv(1.5, tail_request))
        if not math.isfinite(cutoff_t):
            raise ValueError('Mixing tail cutoff is not representable')
        cutoff_u = math.sqrt(cutoff_t)
        omitted = float(gammaincc(1.5, cutoff_t))
        if not math.isfinite(omitted) or omitted <= 0.:
            raise ValueError('Omitted mixing-tail probability is not representable')
        source_scale_x = math.sqrt(2.) * radius * sx / b1
        source_scale_y = math.sqrt(2.) * radius * sy / b1
        if not all(math.isfinite(v * cutoff_u) for v in (source_scale_x, source_scale_y)):
            raise ValueError('Derived mixture widths must be finite and representable')

        def integrand(u):
            # u=sqrt(T) removes the endpoint square-root density; weights are
            # positive Maxwell density, 4/sqrt(pi) u^2 exp(-u^2).
            weight = 4. / math.sqrt(math.pi) * u * u * math.exp(-u * u)
            if weight == 0.:
                return np.zeros(nx + ny + 3, dtype=float)
            gx, ox = _gaussian_bins(x_edges, math.hypot(source_scale_x * u, sigma_psf))
            gy, oy = _gaussian_bins(y_edges, math.hypot(source_scale_y * u, sigma_psf))
            return np.concatenate((gx, gy, [ox, oy, 1.])) * weight

        values, error, info = quad_vec(integrand, 0., cutoff_u, epsabs=tolerance * .5,
                                      epsrel=0., norm='max', limit=limit,
                                      workers=1, full_output=True)
        diagnostics = {'estimated_absolute_error': float(error),
                       'requested_absolute_tolerance': tolerance,
                       'evaluations': int(info.neval),
                       'subdivisions': int(len(info.intervals)),
                       'success': bool(info.success), 'message': str(info.message)}
        if not info.success or not math.isfinite(error) or error > tolerance:
            raise ReferenceConvergenceError('Positive n=1 reference did not converge; no fallback used', diagnostics)
        px, py = values[:nx].copy(), values[nx:nx + ny].copy()
        outside_x, outside_y = float(values[-3]), float(values[-2])
        integration = {
            'method': 'positive_exponential_gaussian_variance_mixture',
            'identity': 'T~Gamma(3/2,1); variance=2*(Re*axis_scale/b1)^2*T+sigma_PSF^2',
            'b1': b1, 'transformation': 'u=sqrt(T)',
            'converged': True, 'estimated_absolute_error': float(error),
            'error_interpretation': 'adaptive quadrature estimate, not a certified full-error bound',
            'omitted_mixing_tail_probability': omitted,
            'mixing_t_cutoff': cutoff_t,
            'truncation_variable': 'latent Gaussian variance mixture T~Gamma(3/2,1); not physical radial T',
            'integrated_mixing_probability': float(values[-1]),
            'evaluations': diagnostics['evaluations'], 'subdivisions': diagnostics['subdivisions']}

    for name, values in (('x', px), ('y', py)):
        if not np.all(np.isfinite(values)) or np.any(values < 0.) or np.any(values > 1.):
            raise ReferenceConvergenceError('Invalid probability in ' + name + '; no clipping used', integration)
    windows = {}
    integrated_mass = integration['integrated_mixing_probability']
    expected_mass = 1. - integration['omitted_mixing_tail_probability']
    roundoff_budget = 64. * np.finfo(float).eps * (nx + ny + 3)
    accounting_budget = max(8. * integration['estimated_absolute_error'], roundoff_budget)
    if abs(integrated_mass - expected_mass) > accounting_budget:
        raise ReferenceConvergenceError('Integrated mixing mass failed conservation check', integration)
    for name, probabilities, outside in (('x', px, outside_x), ('y', py, outside_y)):
        returned = float(np.sum(probabilities, dtype=np.float64))
        if not math.isfinite(outside) or not 0. <= outside <= 1.:
            raise ReferenceConvergenceError('Invalid outside probability; no clipping used', integration)
        if abs(returned + outside - integrated_mass) > accounting_budget:
            raise ReferenceConvergenceError('Probability accounting failed for ' + name, integration)
        windows[name] = {'returned_probability': returned,
                         'outside_probability_integrated': outside,
                         'probability_accounted': returned + outside}
    metadata = {
        'schema': REFERENCE_SCHEMA, 'version': REFERENCE_VERSION,
        'algorithm_revision': REFERENCE_REVISION,
        'status': 'independent_model_reference_not_production_renderer',
        'input': {'n': n, 're_pixel': radius, 'pa_deg': angle, 'axis_ratio': q,
                  'psf_fwhm_arcsec': fwhm, 'pixel_scale_arcsec': pixel_scale},
        'axis_scales': {'x': sx, 'y': sy},
        'models': {'source': 'untruncated unit-flux Sersic with area-preserving shear',
                   'psf': 'circular Gaussian', 'psf_sigma_pixel': sigma_psf,
                   'detector': 'top-hat bin integral; floating-point result is not certified exact'},
        'normalization': {'total_source_probability': 1., 'orthogonal_domain': 'all real',
                          'bin_values': 'absolute unit-source probability per bin; not density',
                          'window_renormalized': False, 'source_center_pixel': [0., 0.],
                          're_definition': 'circularized half-light radius under area-preserving shear'},
        'integration': integration, 'windows': windows,
        'accounting': {'numerical_check_tolerance': accounting_budget,
                       'interpretation': 'conservation check tolerance, not a scientific error bound'},
        'controls': {'absolute_tolerance': tolerance, 'mixing_tail_probability': tail_request,
                     'max_subdivisions': limit, 'max_bins_per_axis': MAX_BINS_PER_AXIS,
                     'absolute_tolerance_scope': 'adaptive quadrature estimate only; not analytic CDF floating-point error'},
        'software': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
                     'machine': platform.machine()},
        'limitations': ['only n=0.5 and n=1 are implemented; no substitution for other n',
                        'infinite-orthogonal marginal differs from legacy finite-square conditional projection',
                        'two marginal window masses must not be multiplied into a joint 2D aperture fraction',
                        'PSF is an adopted circular Gaussian, not an instrument calibration',
                        'no Re cap, empirical Gaussian replacement or finite-window renormalization',
                        'floating-point evaluation and quadrature estimates are not certified bounds',
                        'not enabled in the C7 generator or production runners']}
    return {'x_edges_pixel': x_edges, 'y_edges_pixel': y_edges,
            'x_probability': px, 'y_probability': py, 'metadata': metadata}
