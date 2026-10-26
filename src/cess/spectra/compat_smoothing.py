"""Gaussian point-sample smoothing for the historical public ``varsmooth`` API.

This independent real-space implementation integrates the piecewise-linear
input. It is separate from the photon/bin integration used by CESS simulations.
"""

import numpy as np
from scipy.special import ndtr


def gaussian_sample_smooth(x, y, sig_x, xout=None, oversample=1):
    """Smooth finite point samples with Gaussian kernels evaluated at ``x``.

    ``x`` is a finite, strictly increasing vector of at least two coordinates
    in any one consistent unit; irregular spacing is supported. ``y`` is a
    finite vector ``(n_x,)`` or matrix ``(n_x, n_spectrum)`` with spectra in
    columns. Negative values are allowed. ``sig_x`` is a nonnegative scalar
    or vector ``(n_x,)`` in the same unit as ``x``. Its entries specify the
    kernel width at each *output* coordinate, not the width of input samples.

    Between samples the input is linear, and outside ``[x[0], x[-1]]`` it is
    zero. Each output is the integral of that input against a unit-area
    Gaussian. The finite boundaries are not renormalized: flux can leave the
    interval. A zero-width entry returns its input sample exactly. Variable
    output widths define a smoothing operator, not a flux-conserving transport.

    If ``xout`` is supplied, the already-smoothed samples are interpolated
    linearly onto that finite vector, with zero outside the input interval.
    This preserves the historical scalar-width interpolation contract. The
    return shape is ``(n_xout,)`` or ``(n_xout, n_spectrum)`` respectively.

    ``oversample`` is retained only for signature compatibility and must be 1:
    analytic interval integration needs no oversampling. Other values raise
    ValueError rather than silently changing or ignoring a precision setting.
    This definition intentionally supersedes the former FFT approximation.
    """
    coordinates = np.asarray(x, dtype=float)
    samples = np.asarray(y, dtype=float)
    widths = np.asarray(sig_x, dtype=float)
    sampling = np.asarray(oversample)
    if (sampling.ndim != 0 or not np.issubdtype(sampling.dtype, np.number)
            or not np.isfinite(sampling) or sampling != 1):
        raise ValueError('oversample must be 1; analytic smoothing needs no oversampling.')
    if (coordinates.ndim != 1 or coordinates.size < 2
            or not np.all(np.isfinite(coordinates))):
        raise ValueError('x must be a finite vector of at least two coordinates.')
    spacing = np.diff(coordinates)
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('x must have finite, strictly positive spacing.')
    if (samples.ndim not in (1, 2) or samples.shape[0] != coordinates.size
            or not np.all(np.isfinite(samples))):
        raise ValueError('y must be finite with shape (n_x,) or (n_x, n_spectrum).')
    if widths.ndim == 0:
        widths = np.full(coordinates.size, float(widths))
    if (widths.shape != coordinates.shape or not np.all(np.isfinite(widths))
            or np.any(widths < 0)):
        raise ValueError('sig_x must be finite, nonnegative, and scalar or shape (n_x,).')
    target = coordinates if xout is None else np.asarray(xout, dtype=float)
    if target.ndim != 1 or not np.all(np.isfinite(target)):
        raise ValueError('xout must be a finite vector.')

    columns = samples[:, None] if samples.ndim == 1 else samples
    smoothed = np.empty_like(columns)
    inverse_root_two_pi = 1. / np.sqrt(2. * np.pi)
    # Gaussian moments give exact weights for each linear input segment.
    # A small-interval quadrature avoids subtracting nearly equal CDF values
    # when sigma is much larger than an input interval.
    nodes, quadrature = np.polynomial.legendre.leggauss(8)
    fraction = (nodes + 1.) / 2.
    quadrature = quadrature / 2.
    for index, (center, sigma) in enumerate(zip(coordinates, widths)):
        if sigma == 0:
            smoothed[index] = columns[index]
            continue
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            lower = (coordinates[:-1] - center) / sigma
            upper = (coordinates[1:] - center) / sigma
            mass = np.where(lower > 0, ndtr(-lower) - ndtr(-upper),
                            ndtr(upper) - ndtr(lower))
            density_difference = inverse_root_two_pi * (
                np.exp(-.5 * lower ** 2) - np.exp(-.5 * upper ** 2))
            right_weight = ((center - coordinates[:-1]) * mass
                            + sigma * density_difference) / spacing
        right_weight = np.where(mass == 0, 0., right_weight)
        # Linear hat functions have nonnegative weights summing to mass.
        right_weight = np.clip(right_weight, 0., mass)
        left_weight = mass - right_weight
        with np.errstate(over='ignore', under='ignore'):
            narrow = spacing / sigma < 1.e-3
        if np.any(narrow):
            standardized = lower[narrow, None] + (
                spacing[narrow, None] / sigma) * fraction
            with np.errstate(over='ignore', under='ignore'):
                density = inverse_root_two_pi * np.exp(-.5 * standardized ** 2)
            scale = spacing[narrow] / sigma
            left_weight[narrow] = scale * (density @ (quadrature * (1. - fraction)))
            right_weight[narrow] = scale * (density @ (quadrature * fraction))
        smoothed[index] = (left_weight @ columns[:-1]
                           + right_weight @ columns[1:])

    if xout is not None:
        smoothed = np.column_stack([
            np.interp(target, coordinates, values, left=0., right=0.)
            for values in smoothed.T
        ]) if columns.shape[1] else np.empty((target.size, 0))
    return smoothed[:, 0] if samples.ndim == 1 else smoothed
