"""Fixed-window, continuum-subtracted emission-line diagnostics.

The quantity is a nominal-center-calibration energy-flux proxy over a finite
window. CESS bin fnu is photon equivalent, and its center-response calibration
does not invert the response/LSF/morphology operator. This module therefore does
not infer an intrinsic total line flux, a deblended atomic-line identity, or a
survey detection significance. Windows must be specified independently when
validating noise; selecting a peak from these same data introduces selection.
"""
import numbers

import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded

from cess.spectra.integration import bin_edges

C_ANGSTROM_S = 2.99792458e18
FLAG_WINDOW = 1
FLAG_MASK = 2
FLAG_NONFINITE = 4
FLAG_ERROR = 8
FLAG_COVARIANCE = 16
FLAG_NOT_POSITIVE_DEFINITE = 32
FLAG_CONTINUUM = 64
FLAG_VARIANCE = 128


def _integer(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(name + ' must be an integer.')
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(name + ' must be at least ' + str(minimum) + '.')
    return value


def _cholesky_correlation(off_diagonal):
    banded = np.zeros((2, len(off_diagonal) + 1))
    banded[0] = 1.
    banded[1, :-1] = off_diagonal
    return cholesky_banded(banded, lower=True, check_finite=False)


def measure_line_window(wave, flux_ujy, error_ujy, center_index, line_width_pixels,
                        sideband_width_pixels, sideband_gap_pixels=0,
                        covariance_lag1=None, mask=None, model_ujy=None):
    """Measure one complete pixel-aligned line window and two sidebands.

    wave is a positive increasing observed-Angstrom vector; flux/error are uJy
    and 1-sigma uJy. covariance_lag1[i] is Cov(flux[i], flux[i+1]) in uJy**2;
    more distant covariance is assumed zero. None explicitly chooses diagonal
    independent noise; callers must not use that assumption for unrecorded WCE
    covariance without labelling the approximation. mask=True means available.

    Line pixels are [center-width//2, center-width//2+width). Each sideband has
    sideband_width_pixels, separated from the line by sideband_gap_pixels. No
    pixel is selected by flux/SNR. All requested pixels in line and sidebands
    must be present, unmasked, finite, with finite positive variance. Gap pixels
    are unused. No missing/edge pixels are silently dropped or renormalized.

    At each center f_lambda=flux_ujy*1e-29*c_Angstrom_s/wave**2. The continuum
    is affine in f_lambda, fitted by GLS to both sidebands. The result sums
    (f_lambda-continuum)*bin_width over the line bins. This is a defined nominal
    energy proxy, not an exact inverse of photon-equivalent binning. Error is
    the quadratic form of the complete line+sideband linear weights, including
    continuum-fit uncertainty and line/sideband cross covariance. Banded local
    Cholesky factors avoid allocating a full N_wave covariance matrix.

    Negative observed flux/line flux is retained. model_flux uses exactly the
    same linear weights on optional model_ujy; model_snr uses the same measured
    uncertainty. The dictionary contains only JSON-compatible Python scalars;
    invalid or unavailable metrics are None. Malformed shapes/parameters/wave
    raise ValueError; unavailable measurements return valid=False and flags.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux_ujy, dtype=float)
    error = np.asarray(error_ujy, dtype=float)
    if wave.ndim != 1 or flux.shape != wave.shape or error.shape != wave.shape:
        raise ValueError('wave, flux_ujy and error_ujy must be aligned 1-D arrays.')
    if np.any(~np.isfinite(wave)) or np.any(wave <= 0) or np.any(np.diff(wave) <= 0):
        raise ValueError('wave must be positive, finite and strictly increasing.')
    center = _integer(center_index, 'center_index')
    nline = _integer(line_width_pixels, 'line_width_pixels', 1)
    nside_each = _integer(sideband_width_pixels, 'sideband_width_pixels', 1)
    gap = _integer(sideband_gap_pixels, 'sideband_gap_pixels', 0)
    if mask is None:
        mask = np.ones(len(wave), dtype=bool)
    else:
        mask = np.asarray(mask)
        if mask.shape != wave.shape or mask.dtype.kind != 'b':
            raise ValueError('mask must be a boolean array matching wave.')
    if model_ujy is not None:
        model = np.asarray(model_ujy, dtype=float)
        if model.shape != wave.shape:
            raise ValueError('model_ujy must match wave.')
    else:
        model = None
    if covariance_lag1 is not None:
        covariance = np.asarray(covariance_lag1, dtype=float)
        if covariance.shape != (max(0, len(wave) - 1),):
            raise ValueError('covariance_lag1 must have length len(wave)-1.')
    else:
        covariance = None
    start = center - nline // 2
    stop = start + nline
    left_stop, right_start = start - gap, stop + gap
    left_start, right_stop = left_stop - nside_each, right_start + nside_each
    result = dict(flux=None, error=None, snr=None, model_flux=None, model_snr=None,
        valid=False, flag=0, reason='ok', flux_unit='erg s^-1 cm^-2',
        flux_definition='nominal_center_calibration_window_energy_proxy',
        continuum_model='affine_f_lambda_gls',
        covariance_model='diagonal' if covariance is None else 'diagonal_plus_lag1',
        center_index=center, line_start=start, line_stop=stop,
        left_start=left_start, left_stop=left_stop,
        right_start=right_start, right_stop=right_stop,
        line_wave_min=None, line_wave_max=None, left_sideband_wave_min=None,
        left_sideband_wave_max=None, right_sideband_wave_min=None,
        right_sideband_wave_max=None, nline=nline, nside=2 * nside_each,
        sideband_gap_pixels=gap)

    def invalid(flag, reason):
        result.update(flag=int(flag), reason=reason)
        return result

    if len(wave) < 2 or left_start < 0 or right_stop > len(wave):
        return invalid(FLAG_WINDOW, 'window_out_of_bounds')
    edges = bin_edges(wave)
    for prefix, begin, end in (('line', start, stop), ('left_sideband', left_start, left_stop),
                                ('right_sideband', right_start, right_stop)):
        result[prefix + '_wave_min'] = float(edges[begin])
        result[prefix + '_wave_max'] = float(edges[end])
    left = np.arange(left_start, left_stop)
    line = np.arange(start, stop)
    right = np.arange(right_start, right_stop)
    side = np.r_[left, right]
    used = np.r_[left, line, right]
    flag, reasons = 0, []
    if not np.all(mask[used]):
        flag |= FLAG_MASK
        reasons.append('masked_pixel')
    if np.any(~np.isfinite(flux[used])) or (model is not None and np.any(~np.isfinite(model[used]))):
        flag |= FLAG_NONFINITE
        reasons.append('nonfinite_data')
    if np.any(~np.isfinite(error[used])) or np.any(error[used] <= 0):
        flag |= FLAG_ERROR
        reasons.append('nonpositive_or_nonfinite_error')
    adjacent = np.diff(used) == 1
    if covariance is not None and np.any(~np.isfinite(covariance[used[:-1][adjacent]])):
        flag |= FLAG_COVARIANCE
        reasons.append('nonfinite_covariance')
    if flag:
        return invalid(flag, ';'.join(reasons))
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        conversion = 1.e-29 * C_ANGSTROM_S / wave[used] ** 2
        values = flux[used] * conversion
        sigma = error[used] * conversion
        variance = sigma ** 2
        rho = np.zeros(len(used) - 1)
        if covariance is not None:
            i, j = used[:-1][adjacent], used[1:][adjacent]
            rho[adjacent] = covariance[i] / error[i] / error[j]
    if np.any(~np.isfinite(values)) or (model is not None and np.any(~np.isfinite(model[used] * conversion))):
        return invalid(FLAG_NONFINITE, 'nonfinite_converted_data')
    if np.any(~np.isfinite(variance)) or np.any(variance <= 0):
        return invalid(FLAG_ERROR, 'nonpositive_or_nonfinite_variance')
    if np.any(~np.isfinite(rho)):
        return invalid(FLAG_COVARIANCE, 'nonfinite_correlation')
    side_positions = np.r_[np.arange(nside_each), np.arange(nside_each + nline, len(used))]
    line_positions = np.arange(nside_each, nside_each + nline)
    side_rho = np.zeros(len(side) - 1)
    side_adjacent = np.diff(side) == 1
    if covariance is not None:
        i, j = side[:-1][side_adjacent], side[1:][side_adjacent]
        side_rho[side_adjacent] = covariance[i] / error[i] / error[j]
    try:
        full_factor = _cholesky_correlation(rho)
        side_factor = _cholesky_correlation(side_rho)
    except np.linalg.LinAlgError:
        return invalid(FLAG_NOT_POSITIVE_DEFINITE, 'covariance_not_positive_definite')
    # Center/scale the affine basis to avoid a badly conditioned raw-lambda
    # intercept. This changes neither the continuum family nor its prediction.
    scale = wave[right[-1]] - wave[left[0]]
    design = np.column_stack((np.ones(len(used)), (wave[used] - wave[center]) / scale))
    side_design = design[side_positions]
    sigma_side = sigma[side_positions]
    whitened_design = side_design / sigma_side[:, None]
    weighted_design = cho_solve_banded((side_factor, True), whitened_design, check_finite=False)
    normal = whitened_design.T @ weighted_design
    widths = np.diff(edges)[line]
    integrated_basis = widths @ design[line_positions]
    try:
        if not np.all(np.isfinite(normal)) or np.linalg.matrix_rank(normal) < 2:
            return invalid(FLAG_CONTINUUM, 'continuum_fit_failed')
        coefficients = np.linalg.solve(normal, integrated_basis)
    except np.linalg.LinAlgError:
        return invalid(FLAG_CONTINUUM, 'continuum_fit_failed')
    weights = np.zeros(len(used))
    weights[line_positions] = widths
    weights[side_positions] = -(weighted_design @ coefficients) / sigma_side
    with np.errstate(over='ignore', invalid='ignore'):
        measured_flux = float(weights @ values)
        noise_weights = weights * sigma
        # ||L.T @ (weights*sigma)||**2 evaluates w.T C w without subtracting
        # large positive/negative cross terms near a covariance boundary.
        transformed = full_factor[0] * noise_weights
        transformed[:-1] += full_factor[1, :-1] * noise_weights[1:]
        measured_variance = float(transformed @ transformed)
    if (not np.isfinite(measured_flux) or not np.isfinite(measured_variance) or
            measured_variance <= 0):
        return invalid(FLAG_VARIANCE, 'invalid_integral_variance')
    uncertainty = float(np.sqrt(measured_variance))
    model_flux = None if model is None else float(weights @ (model[used] * conversion))
    if model_flux is not None and not np.isfinite(model_flux):
        return invalid(FLAG_NONFINITE, 'nonfinite_model_integral')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        snr = float(np.divide(measured_flux, uncertainty))
        model_snr = None if model_flux is None else float(np.divide(model_flux, uncertainty))
    if not np.isfinite(snr) or (model_snr is not None and not np.isfinite(model_snr)):
        return invalid(FLAG_VARIANCE, 'nonfinite_integral_snr')
    result.update(flux=measured_flux, error=uncertainty, snr=snr,
                  model_flux=model_flux, model_snr=model_snr,
                  valid=True)
    return result
