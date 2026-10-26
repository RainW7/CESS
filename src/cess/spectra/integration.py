"""Conservative Gaussian redistribution of point-sampled spectral f_nu.

Input f_nu is piecewise linear in observed wavelength. Output f_nu is a
photon-equivalent *pixel-bin value*, not the value of that linear function at a
pixel center: sum(fnu_out*bin_width/center) is the returned photon integral with
the common factor 1/h omitted. Energy-flux conservation is not asserted.

This module performs no configuration/calibration I/O. Gaussian sigma describes
the additional pre-pixel LSF in observed Angstrom; template-resolution policy
belongs to the caller. Pixel integration is an additional response, not another
Gaussian width to infer from the sampling grid.
"""

import numpy as np
from scipy.special import ndtr, erfc


def _grid(values, name):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)) or
            np.any(values <= 0) or np.any(np.diff(values) <= 0)):
        raise ValueError(name + ' must be positive, finite and strictly increasing.')
    return values


def bin_edges(centers):
    """Midpoint edges, with a half-step extrapolation at either end."""
    centers = _grid(centers, 'output_centers')
    edges = np.r_[centers[0] - (centers[1] - centers[0]) / 2.,
                  centers[:-1] + np.diff(centers) / 2.,
                  centers[-1] + (centers[-1] - centers[-2]) / 2.]
    if edges[0] <= 0:
        raise ValueError('Inferred output-bin edges must be positive.')
    return edges


def response_curve(response_wave, response):
    """Validate an incident-wavelength photon-to-electron response curve.

    Wavelength is observed Angstrom and strictly increasing; response is finite
    and dimensionless in [0,1]. An all-zero response is valid. This function
    performs no extrapolation, clipping, sorting or file I/O.
    """
    wavelengths = _grid(response_wave, 'response_wave')
    transmission = np.asarray(response, dtype=float)
    if (transmission.shape != wavelengths.shape or np.any(~np.isfinite(transmission)) or
            np.any(transmission < 0) or np.any(transmission > 1)):
        raise ValueError('response must be finite, in [0,1], and match response_wave.')
    return wavelengths, transmission


def _basis_integrals(left, right):
    """Exact integrals of the two linear endpoint bases divided by lambda."""
    ratio = (right - left) / left
    logarithm = np.log1p(ratio)
    # w1=1-log1p(r)/r loses precision for narrow native wavelength intervals.
    small = ratio < .01
    safe_ratio = np.where(small, 1., ratio)
    w1 = 1. - logarithm / safe_ratio
    series = np.zeros_like(ratio)
    power = ratio.copy()
    for order in range(1, 11):
        series += ((-1.) ** (order + 1) / (order + 1)) * power
        power *= ratio
    w1 = np.where(small, series, w1)
    return logarithm - w1, w1


def _integral(left, right, flux_left, flux_right):
    w0, w1 = _basis_integrals(left, right)
    return flux_left * w0 + flux_right * w1


def _product_basis_integrals(left, right):
    """Positive quadratic Bernstein-basis integrals divided by wavelength."""
    ratio = (right - left) / left
    logarithm = np.log1p(ratio)
    _, moment1 = _basis_integrals(left, right)
    small = ratio < .01
    safe_ratio = np.where(small, 1., ratio)
    moment2 = (.5 * ratio ** 2 - ratio + logarithm) / safe_ratio ** 2
    series = np.zeros_like(ratio)
    power = ratio.copy()
    for order in range(1, 11):
        series += ((-1.) ** (order + 1) / (order + 2)) * power
        power *= ratio
    moment2 = np.where(small, series, moment2)
    return logarithm - 2. * moment1 + moment2, moment1 - moment2, moment2


def _response_integral(left, right, flux_left, flux_right, response_left, response_right):
    """Integrate the product of separately linear fnu and response over lambda."""
    b00, b01, b11 = _product_basis_integrals(left, right)
    return (flux_left * response_left * b00 +
            (flux_left * response_right + flux_right * response_left) * b01 +
            flux_right * response_right * b11)


def integrate_response(wave, fnu, response_wave, response):
    """Return integral(fnu*T/lambda) over the finite response calibration range.

    fnu and T are separately piecewise linear on their own observed-Angstrom
    grids. The full response-curve interval must be covered by the source SED;
    there is no extrapolation and no assumption that the response is zero beyond
    its calibrated endpoints, even when an endpoint sample happens to be zero.
    The result omits 1/h, collecting area, time and solid angle. For a diffuse
    slitless sky these factors may subsequently convert the full-band integral
    into electrons per detector pixel; this routine does not apply a source LSF.
    """
    wave = _grid(wave, 'wave')
    flux = np.asarray(fnu, dtype=float)
    if flux.shape != wave.shape or np.any(~np.isfinite(flux)):
        raise ValueError('fnu must be finite and have the same shape as wave.')
    response_wave, response = response_curve(response_wave, response)
    lower, upper = response_wave[0], response_wave[-1]
    if wave[0] > lower or wave[-1] < upper:
        raise ValueError('Insufficient source coverage for the full response calibration range; '
                         'no response/source extrapolation.')
    native = np.unique(np.r_[response_wave, wave[(wave > lower) & (wave < upper)]])
    source_values = np.interp(native, wave, flux)
    response_values = np.interp(native, response_wave, response)
    result = float(np.sum(_response_integral(native[:-1], native[1:],
        source_values[:-1], source_values[1:], response_values[:-1], response_values[1:])))
    if not np.isfinite(result):
        raise ValueError('Response integral is not finite.')
    return result


def _cumulative_at(wave, flux, positions):
    """Integral of the native piecewise-linear fnu/lambda, clipped to coverage."""
    pieces = _integral(wave[:-1], wave[1:], flux[:-1], flux[1:])
    cumulative = np.r_[0., np.cumsum(pieces)]
    position = np.clip(positions, wave[0], wave[-1])
    interval = np.clip(np.searchsorted(wave, position, side='right') - 1, 0, len(wave) - 2)
    fraction = (position - wave[interval]) / (wave[interval + 1] - wave[interval])
    at_position = flux[interval] + fraction * (flux[interval + 1] - flux[interval])
    return cumulative[interval] + _integral(wave[interval], position, flux[interval], at_position)


def _probability(low, high):
    """Stable normal-CDF difference, including the positive far tail."""
    positive = low >= 0
    # Choose the stable signed limits before evaluating the CDF. np.where is
    # eager, so selecting two completed differences would evaluate four CDFs.
    return (ndtr(np.where(positive, -low, high)) -
            ndtr(np.where(positive, -high, low)))


def _add_to_bins(out, index, weights):
    """Accumulate a block into float bins; callers supply legal 1-D indices.

    Grouping a block before adding it to out changes floating-point summation
    order relative to np.add.at. No clipping or filtering is performed here;
    all three callers explicitly exclude indices outside the detector bins.
    """
    out += np.bincount(index, weights=weights, minlength=len(out))


def convolve_to_bins(wave, fnu, output_centers, sigma, *, response_wave=None,
                     response=None, max_sigma=None,
                     quadrature_order=8, tail_sigma=8., max_work_items=131072,
                     require_full_coverage=True, max_subintervals=2000000):
    """Redistribute a linear-interpolated fnu SED into finite Gaussian bins.

    Parameters use observed Angstrom and any consistent fnu units. sigma may be
    a nonnegative scalar, an array matching wave (linearly interpolated), or a
    callable evaluated at source wavelengths. Providing max_sigma bounds the
    required input support to output edges +/- tail_sigma*max_sigma *before*
    evaluating a callable, avoiding unrelated out-of-band resolution policies.
    Without it the bound is inferred over the supplied native grid.

    Optional response_wave/response must be supplied together. Both fnu and T
    are separately piecewise linear, with every response knot included before
    integration; their product is quadratic, not a linear interpolation of
    endpoint products. Response coverage must contain the full required support;
    outside values are never guessed as constant or zero. With a response, the
    returned bin fnu is already T-weighted: source counts are
    area*time/h * sum(fnu_out*bin_width/center), without another T multiplication.
    It is a count proxy, not a throughput-corrected physical fnu spectrum. With
    neither response argument, the C1 unit-response numerical path is retained.

    Returns (fnu_out, diagnostics). Full input support is required by default;
    missing coverage raises ValueError instead of inventing zero/flat padding.
    With require_full_coverage=False the result is explicitly the contribution
    of the finite available input, and coverage_complete remains False.

    Positive widths use 8/16-point Gaussian quadrature, with source intervals
    subdivided to at most half the minimum sampled positive sigma/output-bin
    width. Endpoint-basis weights are normalized to their analytic fnu/lambda
    integrals, conserving each input segment's photon weight. This does *not*
    renormalize truncated Gaussian tails or light outside the output window.
    Zero-width intervals use analytic bin integrals. Memory is bounded by input
    arrays plus max_work_items local node/bin pairs; no full dense operator is
    allocated. max_subintervals bounds pathological near-zero-width requests.
    """
    wave = _grid(wave, 'wave')
    centers = _grid(output_centers, 'output_centers')
    flux = np.asarray(fnu, dtype=float)
    if flux.shape != wave.shape or np.any(~np.isfinite(flux)):
        raise ValueError('fnu must be finite and have the same shape as wave.')
    if (response_wave is None) != (response is None):
        raise ValueError('response_wave and response must be supplied together.')
    weighted = response_wave is not None
    if weighted:
        response_wave, response = response_curve(response_wave, response)
    if quadrature_order not in (8, 16):
        raise ValueError('quadrature_order must be 8 or 16.')
    if not np.isfinite(tail_sigma) or tail_sigma <= 0:
        raise ValueError('tail_sigma must be finite and positive.')
    for value, name, minimum in ((max_work_items, 'max_work_items', 64),
                                  (max_subintervals, 'max_subintervals', 1)):
        if isinstance(value, bool) or int(value) != value or value < minimum:
            raise ValueError(name + ' must be an integer >= ' + str(minimum))
    max_work_items, max_subintervals = int(max_work_items), int(max_subintervals)
    if not isinstance(require_full_coverage, (bool, np.bool_)):
        raise ValueError('require_full_coverage must be boolean.')
    edges = bin_edges(centers)
    widths = np.diff(edges)
    minimum_pixel = float(widths.min())
    scalar_sigma = None
    if callable(sigma):
        evaluate_raw = sigma
    else:
        supplied = np.asarray(sigma, dtype=float)
        if supplied.ndim == 0:
            scalar_sigma = float(supplied)
            evaluate_raw = lambda position: np.full(np.shape(position), scalar_sigma)
        elif supplied.shape == wave.shape:
            evaluate_raw = lambda position: np.interp(position, wave, supplied)
        else:
            raise ValueError('sigma must be scalar, callable, or have the same shape as wave.')
    def evaluate(position):
        values = np.asarray(evaluate_raw(position), dtype=float)
        if values.ndim == 0:
            values = np.full(np.shape(position), float(values))
        if values.shape != np.shape(position) or np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError('sigma must return finite nonnegative values matching source wavelengths.')
        if max_sigma is not None and np.any(values > bound * (1. + 1e-12) + 1e-14):
            raise ValueError('sigma exceeds the supplied max_sigma support bound.')
        return values
    if max_sigma is None:
        initial = evaluate(wave)
        bound = float(initial.max())
    else:
        bound = float(max_sigma)
        if not np.isfinite(bound) or bound < 0:
            raise ValueError('max_sigma must be finite and nonnegative.')
    required = np.array([edges[0] - tail_sigma * bound, edges[-1] + tail_sigma * bound])
    if weighted and (response_wave[0] > required[0] or response_wave[-1] < required[1]):
        raise ValueError('Insufficient response coverage: require [{:.12g}, {:.12g}] Angstrom '
                         'including Gaussian support; no response extrapolation.'.format(*required))
    covered = bool(wave[0] <= required[0] and wave[-1] >= required[1])
    if require_full_coverage and not covered:
        raise ValueError('Insufficient input coverage: require [{:.12g}, {:.12g}] Angstrom '
                         'including Gaussian support.'.format(*required))
    left_limit, right_limit = max(wave[0], required[0]), min(wave[-1], required[1])
    if right_limit <= left_limit:
        raise ValueError('Input spectrum does not overlap the required convolution support.')
    # Include exact crop boundaries and output-bin boundaries in the original
    # linear function; never cut to the nearest native point or extrapolate it.
    native = np.unique(np.r_[left_limit, wave[(wave > left_limit) & (wave < right_limit)],
                             edges[(edges > left_limit) & (edges < right_limit)], right_limit])
    if weighted:
        native = np.unique(np.r_[native, response_wave[
            (response_wave > left_limit) & (response_wave < right_limit)]])
    values = np.interp(native, wave, flux)
    midpoints = native[:-1] + np.diff(native) / 2.
    at_edges, at_middle = evaluate(native), evaluate(midpoints)
    sigmas = np.vstack((at_edges[:-1], at_middle, at_edges[1:]))
    # A sigma approaching zero at one endpoint must not force millions of
    # equal subdivisions across the entire original interval. Refine just
    # the stiff neighborhood first, in vectorized batches of midpoint knots.
    # This is geometry refinement, not a lower clipping of the physical sigma.
    for refinement in range(64):
        positive_min = np.min(np.where(sigmas > 0, sigmas, np.inf), axis=0)
        local_max = np.max(sigmas, axis=0)
        mixed_zero = np.any(sigmas == 0, axis=0) & (local_max > 0)
        trial_step = .5 * np.minimum(minimum_pixel, positive_min)
        stiff = ((np.diff(native) / trial_step > 64) &
                 ((local_max > 4. * positive_min) | mixed_zero))
        if not np.any(stiff):
            break
        added = midpoints[stiff]
        if np.any((added <= native[:-1][stiff]) | (added >= native[1:][stiff])):
            raise ValueError('Near-zero sigma transition cannot be resolved at wavelength precision.')
        if native.size + added.size > max_subintervals + 1:
            raise ValueError('Sigma transition requires too many refinement intervals.')
        native = np.sort(np.r_[native, added])
        values = np.interp(native, wave, flux)
        midpoints = native[:-1] + np.diff(native) / 2.
        at_edges, at_middle = evaluate(native), evaluate(midpoints)
        sigmas = np.vstack((at_edges[:-1], at_middle, at_edges[1:]))
    else:
        raise ValueError('Near-zero sigma geometry did not converge within 64 refinements.')
    positive_min = np.min(np.where(sigmas > 0, sigmas, np.inf), axis=0)
    zero_intervals = np.all(sigmas == 0, axis=0)
    steps = .5 * np.minimum(minimum_pixel, positive_min)
    steps[zero_intervals] = .5 * minimum_pixel
    parts_float = np.maximum(1., np.ceil(np.diff(native) / steps))
    if np.any(~np.isfinite(parts_float)) or np.sum(parts_float) > max_subintervals:
        raise ValueError('Convolution requires too many subintervals; check near-zero sigma or '
                         'raise max_subintervals explicitly.')
    parts = parts_float.astype(np.int64)
    cumulative_parts = np.cumsum(parts)
    total = int(cumulative_parts[-1])
    unweighted_integral = float(np.sum(_integral(native[:-1], native[1:], values[:-1], values[1:])))
    if weighted:
        native_response = np.interp(native, response_wave, response)
        input_integral = float(np.sum(_response_integral(native[:-1], native[1:],
            values[:-1], values[1:], native_response[:-1], native_response[1:])))
        absolute_bound = float(np.sum(_response_integral(native[:-1], native[1:],
            np.abs(values[:-1]), np.abs(values[1:]), native_response[:-1], native_response[1:])))
    else:
        input_integral = unweighted_integral
        absolute_bound = float(np.sum(_integral(native[:-1], native[1:],
                                                np.abs(values[:-1]), np.abs(values[1:]))))
    photons = np.zeros(centers.size)
    sigma_min = float(min(at_edges.min(), at_middle.min()))
    sigma_max = float(max(at_edges.max(), at_middle.max()))
    if np.all(zero_intervals):
        # The grid includes all bin edges. Accumulate each local positive
        # integral directly in its bin; differences of a global cumulative
        # integral can erase faint continuum after a very strong emission line.
        target = np.searchsorted(edges, midpoints, side='right') - 1
        inside = (target >= 0) & (target < centers.size)
        pieces = (_response_integral(native[:-1], native[1:], values[:-1], values[1:],
                                    native_response[:-1], native_response[1:]) if weighted else
                  _integral(native[:-1], native[1:], values[:-1], values[1:]))
        photons = np.bincount(target[inside], weights=pieces[inside], minlength=centers.size)
    else:
        nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
        fractions = (nodes + 1.) / 2.
        maximum_bins = min(centers.size, int(np.ceil(2. * tail_sigma * bound / minimum_pixel)) + 2)
        segment_block = max(1, max_work_items // (quadrature_order * max(1, maximum_bins)))
        for start in range(0, total, segment_block):
            flat = np.arange(start, min(start + segment_block, total), dtype=np.int64)
            original = np.searchsorted(cumulative_parts, flat, side='right')
            previous = np.where(original > 0, cumulative_parts[np.maximum(0, original - 1)], 0)
            local = flat - previous
            fraction_left, fraction_right = local / parts[original], (local + 1.) / parts[original]
            span = native[original + 1] - native[original]
            left = native[original] + fraction_left * span
            right = native[original] + fraction_right * span
            flux_span = values[original + 1] - values[original]
            fl = values[original] + fraction_left * flux_span
            fr = values[original] + fraction_right * flux_span
            if weighted:
                response_span = native_response[original + 1] - native_response[original]
                tl = native_response[original] + fraction_left * response_span
                tr = native_response[original] + fraction_right * response_span
            positions = left[:, None] + (right - left)[:, None] * fractions
            node_sigma = evaluate(positions)
            sigma_min = min(sigma_min, float(node_sigma.min()))
            sigma_max = max(sigma_max, float(node_sigma.max()))
            w0, w1 = _basis_integrals(left, right)
            common = (right - left)[:, None] * weights / (2. * positions)
            basis0, basis1 = common * (1. - fractions), common * fractions
            basis0 *= (w0 / basis0.sum(axis=1))[:, None]
            basis1 *= (w1 / basis1.sum(axis=1))[:, None]
            if weighted:
                # Each factor is linear on this merged interval. Normalize
                # three positive quadratic bases, rather than interpolating
                # the product of endpoint samples as a new linear SED.
                b00, b01, b11 = _product_basis_integrals(left, right)
                basis00 = common * (1. - fractions) ** 2
                basis01 = common * fractions * (1. - fractions)
                basis11 = common * fractions ** 2
                basis00 *= (b00 / basis00.sum(axis=1))[:, None]
                basis01 *= (b01 / basis01.sum(axis=1))[:, None]
                basis11 *= (b11 / basis11.sum(axis=1))[:, None]
                photon_weights = ((fl * tl)[:, None] * basis00 +
                                  (fl * tr + fr * tl)[:, None] * basis01 +
                                  (fr * tr)[:, None] * basis11)
            else:
                photon_weights = fl[:, None] * basis0 + fr[:, None] * basis1
            # Intervals with identically zero sampled sigma are integrated
            # analytically. Added bin-edge knots ensure each lies in one bin.
            exact_zero = np.all(node_sigma == 0, axis=1) & zero_intervals[original]
            if np.any(exact_zero):
                target = np.searchsorted(edges, (left[exact_zero] + right[exact_zero]) / 2., side='right') - 1
                keep = (target >= 0) & (target < centers.size)
                exact = (_response_integral(left[exact_zero], right[exact_zero],
                    fl[exact_zero], fr[exact_zero], tl[exact_zero], tr[exact_zero]) if weighted else
                    _integral(left[exact_zero], right[exact_zero], fl[exact_zero], fr[exact_zero]))
                _add_to_bins(photons, target[keep], exact[keep])
            active = ~exact_zero
            position = positions[active].ravel()
            node_width = node_sigma[active].ravel()
            node_flux = photon_weights[active].ravel()
            delta = node_width == 0
            if np.any(delta):
                target = np.searchsorted(edges, position[delta], side='right') - 1
                keep = (target >= 0) & (target < centers.size)
                _add_to_bins(photons, target[keep], node_flux[delta][keep])
            position, node_width, node_flux = position[~delta], node_width[~delta], node_flux[~delta]
            if not position.size:
                continue
            first = np.maximum(0, np.searchsorted(edges, position - tail_sigma * node_width, side='right') - 1)
            stop = np.minimum(centers.size, np.searchsorted(edges, position + tail_sigma * node_width, side='left'))
            count = np.maximum(0, stop - first)
            local_max = int(count.max()) if count.size else 0
            # Tile bin offsets as well, so even a very broad kernel cannot
            # exceed the configured node/bin working-array budget.
            tile = max(1, max_work_items // max(1, position.size))
            for offset in range(0, local_max, tile):
                columns = np.arange(offset, min(offset + tile, local_max))
                index = first[:, None] + columns
                good = columns < count[:, None]
                index = np.clip(index, 0, centers.size - 1)
                lower = (edges[index] - position[:, None]) / node_width[:, None]
                upper = (edges[index + 1] - position[:, None]) / node_width[:, None]
                contribution = node_flux[:, None] * _probability(lower, upper)
                _add_to_bins(photons, index[good], contribution[good])
    output = centers / widths * photons
    tail_bound = float(erfc(tail_sigma / np.sqrt(2.)))
    diagnostics = dict(edges=edges, widths=widths, bin_edges=edges, bin_widths=widths,
        coverage_complete=covered, required_input_bounds=required,
        input_support_bounds=np.array([left_limit, right_limit]),
        sigma_min=sigma_min, sigma_max=sigma_max, max_sigma_bound=bound,
        input_photon_integral=input_integral, output_photon_integral=float(photons.sum()),
        input_unweighted_photon_integral=unweighted_integral,
        unreturned_photon_integral=float(input_integral - photons.sum()),
        truncated_probability_bound=tail_bound, tail_loss_absolute_bound=absolute_bound * tail_bound,
        tail_sigma=float(tail_sigma), subinterval_count=total, quadrature_order=quadrature_order,
        response_weighted=weighted, response_coverage_complete=True,
        response_model=('incident-wavelength piecewise-linear photon response' if weighted else 'unit_response'),
        response_wavelength_bounds=(np.array([response_wave[0], response_wave[-1]]) if weighted else None),
        flux_sampling=('pixel-bin response-weighted photon-equivalent fnu count proxy; fnu and T separately linear'
                       if weighted else 'pixel-bin photon-equivalent fnu; input fnu piecewise linear'))
    return output, diagnostics
