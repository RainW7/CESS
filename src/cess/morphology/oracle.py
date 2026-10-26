"""Slow positive single-bin Sersic marginal oracle, independent of FFT renderers.

This experimental reference integrates the orthogonal coordinate over all real
values. It is not an efficient profile renderer, a 2D aperture model, or an
instrument calibration. Import performs no resource or configuration I/O.
"""
import math
from numbers import Integral, Real
import platform
import warnings

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import gammaincc, gammainccinv, gammaincinv, ndtr

ORACLE_SCHEMA = 'cess.sersic_bin_oracle'
ORACLE_VERSION = 1
ORACLE_REVISION = '20261021-C9'


class OracleInputError(ValueError):
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = {} if diagnostics is None else diagnostics


class OracleConvergenceError(RuntimeError):
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = {} if diagnostics is None else diagnostics


def _real(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise OracleInputError(name + ' must be a real scalar, excluding booleans', {'parameter': name})
    try:
        result = float(value)
    except (ValueError, OverflowError):
        raise OracleInputError(name + ' must be finite and representable', {'parameter': name})
    if not math.isfinite(result) or (positive and result <= 0.):
        raise OracleInputError(name + ' is outside its finite permitted range', {'parameter': name})
    return result


def _normal_interval(lower, upper):
    # Complementary tails avoid subtracting two values rounded to one.
    # erf also preserves very narrow central intervals whose CDFs round to 0.5.
    if lower < 1. and upper > -1.:
        return .5*(math.erf(upper/math.sqrt(2.))-math.erf(lower/math.sqrt(2.)))
    return float(ndtr(-lower) - ndtr(-upper) if lower >= 0.
                 else ndtr(upper) - ndtr(lower))


def sersic_bin_oracle(n, re_pixel, pa_deg, axis_ratio, psf_fwhm_arcsec,
                      pixel_scale_arcsec, lower_edge_pixel, upper_edge_pixel,
                      axis='x', *, absolute_tolerance=1.e-8,
                      radial_tail_probability=1.e-12, gaussian_tail_sigma=10.,
                      max_evaluations=200000):
    """Return an absolute all-source probability for one 1D detector bin.

    Supports 0.5<=n<=5 and a positive circular Gaussian PSF. Re is the circular
    parent half-light radius in pixels before area-preserving shear q=b/a; PA
    is measured in degrees from image x. Edges are finite pixel coordinates
    relative to source center zero. The orthogonal coordinate is integrated
    over all real values, not a finite stamp or aperture. No cap, fallback,
    negative clipping or finite-window normalization occurs.

    Positive radial Gamma and conditional-angle integrals use adaptive quad.
    absolute_tolerance controls the combined *estimated* outer plus largest
    sampled inner error; this is not a certified bound. Explicit radial and
    Gaussian angular omission probabilities are recorded separately. Every
    warning, nonconvergence or evaluation-budget failure raises an exception
    with diagnostics, never a substitute probability.
    """
    n = _real(n, 'n')
    radius = _real(re_pixel, 're_pixel', True)
    pa = _real(pa_deg, 'pa_deg')
    q = _real(axis_ratio, 'axis_ratio', True)
    fwhm = _real(psf_fwhm_arcsec, 'psf_fwhm_arcsec', True)
    scale = _real(pixel_scale_arcsec, 'pixel_scale_arcsec', True)
    lo = _real(lower_edge_pixel, 'lower_edge_pixel')
    hi = _real(upper_edge_pixel, 'upper_edge_pixel')
    tolerance = _real(absolute_tolerance, 'absolute_tolerance', True)
    radial_tail = _real(radial_tail_probability, 'radial_tail_probability', True)
    gaussian_tail = _real(gaussian_tail_sigma, 'gaussian_tail_sigma', True)
    if not .5 <= n <= 5. or q > 1. or lo >= hi or not isinstance(axis, str) or axis not in ('x', 'y'):
        raise OracleInputError('Require 0.5<=n<=5, 0<q<=1, lower<upper and axis x or y',
                               {'n': n, 'axis_ratio': q, 'edges': [lo, hi], 'axis': str(axis)})
    if radial_tail >= 1.:
        raise OracleInputError('radial_tail_probability must lie in (0,1)', {'parameter': 'radial_tail_probability'})
    if (isinstance(max_evaluations, bool) or not isinstance(max_evaluations, Integral)
            or max_evaluations < 1):
        raise OracleInputError('max_evaluations must be a positive integer', {'parameter': 'max_evaluations'})
    maximum = int(max_evaluations)
    angle = abs(pa)
    if angle > 90.:
        angle %= 180.
        if angle > 90.:
            angle = 180. - angle
    # Exact cardinal angles avoid tiny cos(90deg) amplified by extreme q.
    if angle == 0.:
        cosine, sine = 1., 0.
    elif angle == 90.:
        cosine, sine = 0., 1.
    else:
        cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    if axis == 'y':
        cosine, sine = sine, cosine
    try:
        projected_radius = radius * math.sqrt(cosine*cosine/q + q*sine*sine)
        sigma = fwhm / scale / math.sqrt(8.*math.log(2.))
        margin = gaussian_tail * sigma
        shape = 2.*n
        bn = float(gammaincinv(shape, .5))
        tmax = float(gammainccinv(shape, radial_tail))
    except (ValueError, OverflowError, ZeroDivisionError) as error:
        raise OracleInputError('Derived geometry is not representable: ' + str(error))
    if not all(math.isfinite(value) and value > 0. for value in (projected_radius, sigma, margin, bn, tmax)):
        raise OracleInputError('Derived radius, sigma and integration limits must be finite and positive')

    diagnostics = {
        'method': 'positive-Gamma-radial-conditional-angle-nested-quad-v1',
        'definition': 'absolute unit all-source 1D bin probability; orthogonal coordinate over all real values',
        'input': {'n': n, 're_pixel': radius, 'pa_deg': pa, 'axis_ratio': q,
                  'psf_fwhm_arcsec': fwhm, 'pixel_scale_arcsec': scale,
                  'lower_edge_pixel': lo, 'upper_edge_pixel': hi, 'axis': axis},
        'projected_re_pixel': projected_radius, 'psf_sigma_pixel': sigma, 'b_n': bn,
        'radial_gamma_shape': shape,
        'radial_variable_definition': 't=b_n*(r/Re)^(1/n), Gamma(shape=2n,scale=1); not the n=1 Normal-Gamma mixing variable',
        'radial_t_max': tmax, 'absolute_tolerance': tolerance,
        'radial_tail_probability_requested': radial_tail,
        'radial_omission_probability_bound': float(gammaincc(shape, tmax)),
        'gaussian_tail_sigma': gaussian_tail,
        'angular_omission_probability_bound': float(2.*ndtr(-gaussian_tail)),
        'outer_error_estimate': None, 'largest_inner_error_estimate': 0.,
        'combined_quadrature_error_estimate': None,
        'evaluations': {'radial': 0, 'angular': 0, 'total': 0, 'maximum': maximum},
        'subinterval_limits': {'inner': 150, 'outer': 200},
        'maximum_inner_subintervals': 0, 'outer_subintervals': None,
        'convergence_basis': 'adaptive error estimates and limits; not a certified bound; omission probabilities separate',
        'warnings': []}

    def fail(message):
        raise OracleConvergenceError(message, diagnostics)

    def consume(kind):
        counts = diagnostics['evaluations']
        if counts['total'] >= maximum:
            fail('Oracle evaluation budget exceeded')
        counts[kind] += 1
        counts['total'] += 1

    def checked_quad(function, left, right, phase, **kwargs):
        answer = quad(function, left, right, full_output=1, **kwargs)
        if len(answer) != 3:
            diagnostics['failed_phase'] = phase
            diagnostics['quad_message'] = str(answer[3:])
            fail('Oracle quadrature did not converge: ' + phase)
        value, error, info = answer
        if not math.isfinite(value) or value < 0. or not math.isfinite(error) or error < 0.:
            diagnostics['failed_phase'] = phase
            fail('Oracle quadrature returned invalid value/error: ' + phase)
        return float(value), float(error), info

    def radial_integrand(t):
        consume('radial')
        density = (1. if shape == 1. else 0.) if t == 0. else math.exp((shape-1.)*math.log(t)-t-math.lgamma(shape))
        amplitude = projected_radius * (t/bn)**n
        if not math.isfinite(amplitude):
            fail('Oracle source radius overflowed')
        if amplitude == 0.:
            return density * _normal_interval(lo/sigma, hi/sigma)
        mean_min, mean_max = max(-amplitude, lo-margin), min(amplitude, hi+margin)
        if mean_max <= mean_min:
            return 0.
        # Clamping is solely a geometric arccos-domain guard, never flux clipping.
        left = math.acos(max(-1., min(1., mean_max/amplitude)))
        right = math.acos(max(-1., min(1., mean_min/amplitude)))
        points = []
        for location in (lo-sigma, lo, lo+sigma, .5*lo+.5*hi, hi-sigma, hi, hi+sigma):
            if -amplitude < location < amplitude:
                point = math.acos(location/amplitude)
                if left < point < right:
                    points.append(point)

        def angle_integrand(theta):
            consume('angular')
            mean = amplitude * math.cos(theta)
            probability = _normal_interval((lo-mean)/sigma, (hi-mean)/sigma)
            if not math.isfinite(probability) or probability < 0.:
                fail('Gaussian CDF difference is nonfinite or negative; no clipping applied')
            return probability/math.pi

        probability, error, info = checked_quad(angle_integrand, left, right, 'inner-angle',
            points=sorted(set(points)), epsabs=tolerance*.1, epsrel=tolerance*.1, limit=150)
        diagnostics['largest_inner_error_estimate'] = max(diagnostics['largest_inner_error_estimate'], error)
        diagnostics['maximum_inner_subintervals'] = max(diagnostics['maximum_inner_subintervals'], int(info['last']))
        return density*probability

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            probability, error, info = checked_quad(radial_integrand, 0., tmax, 'outer-radial',
                                                   epsabs=tolerance, epsrel=tolerance, limit=200)
        except OracleConvergenceError:
            diagnostics['warnings'] = [str(item.message) for item in caught]
            raise
        except Exception as error:
            diagnostics['warnings'] = [str(item.message) for item in caught]
            diagnostics['exception_type'] = type(error).__name__
            fail('Oracle integration failed: ' + str(error))
    diagnostics['warnings'] = [str(item.message) for item in caught]
    diagnostics['outer_error_estimate'] = error
    diagnostics['outer_subintervals'] = int(info['last'])
    diagnostics['combined_quadrature_error_estimate'] = error + diagnostics['largest_inner_error_estimate']
    if caught:
        fail('Oracle emitted a warning; result rejected')
    if diagnostics['combined_quadrature_error_estimate'] > tolerance:
        fail('Oracle combined quadrature error estimate exceeds absolute_tolerance')
    if not 0. <= probability <= 1.:
        fail('Oracle returned a probability outside [0,1]; no clipping applied')
    return {
        'schema': ORACLE_SCHEMA, 'version': ORACLE_VERSION,
        'revision': ORACLE_REVISION, 'status': 'experimental_reference',
        'probability': probability, 'converged': True, 'diagnostics': diagnostics,
        'normalization': {'source_total_probability': 1., 'orthogonal_domain': 'all real',
                          'bin_quantity': 'absolute unit-source probability; not a density',
                          'window_renormalization': False,
                          'omission_probabilities': 'reported separately; not restored to the bin'},
        'software': {'python': platform.python_version(), 'machine': platform.machine(),
                     'numpy': np.__version__, 'scipy': scipy.__version__},
        'limitations': [
            'untruncated Sersic source; Re is circularized parent half-light radius under area-preserving shear',
            'positive circular Gaussian PSF is an adopted model, not an independent instrument calibration',
            'infinite orthogonal marginal is not a finite 2D aperture fraction or velocity-field model',
            'quadrature estimates are not certified bounds; explicit omission probabilities are separate',
            'slow experimental single-bin oracle; not a production profile renderer',
            'no legacy radius cap, fallback, clipping or finite-window renormalization']}
