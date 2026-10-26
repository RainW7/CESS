"""Photon-response synthetic photometry and an aperture electron-noise model.

Arrays without units use the units in the argument names. Astropy Quantities
are converted and checked. The input spectrum is the *total-source*, noiseless
observed-frame f_nu; response includes optical losses and electron-producing QE.
No configuration, calibration files, or catalogues are loaded at import time.
"""

import numpy as np
from astropy import units as u


PLANCK_ERG_S = 6.62607015e-27
FNU_CGS = u.erg / (u.s * u.cm ** 2 * u.Hz)
UJY_CGS = 1.e-29
MAG_DERIVATIVE = 2.5 / np.log(10.)

# Flags describe measurement availability separately from detection.
FLAG_INCOMPLETE_COVERAGE = 1
FLAG_NONFINITE_MODEL = 2
FLAG_NEGATIVE_MODEL = 4
FLAG_BELOW_DETECTION = 8
FLAG_NONPOSITIVE_OBSERVED = 16

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(8)
_FRACTIONS = (_NODES + 1.) / 2.


def _value(value, unit, name):
    if hasattr(value, 'unit'):
        try:
            value = value.to_value(unit)
        except (u.UnitConversionError, TypeError, AttributeError) as error:
            raise ValueError('{} must have units compatible with {}'.format(name, unit)) from error
    if np.ma.isMaskedArray(value):
        value = np.ma.asarray(value, dtype=float).filled(np.nan)
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('{} must contain numeric values'.format(name)) from error


def _scalar(value, unit, name, positive=False, nonnegative=False):
    result = _value(value, unit, name)
    if result.ndim != 0 or not np.isfinite(result):
        raise ValueError('{} must be a finite scalar'.format(name))
    result = float(result)
    if positive and result <= 0:
        raise ValueError('{} must be positive'.format(name))
    if nonnegative and result < 0:
        raise ValueError('{} must be non-negative'.format(name))
    return result


def _wavelengths(value, name):
    wave = _value(value, u.AA, name)
    if (wave.ndim != 1 or len(wave) < 2 or not np.all(np.isfinite(wave))
            or np.any(wave <= 0) or np.any(np.diff(wave) <= 0)):
        raise ValueError('{} must be a finite, positive, strictly increasing 1-D grid'.format(name))
    return wave


def _refine_wide_intervals(grid):
    """Bound interval wavelength ratios for accurate smooth quadrature."""
    pieces = [grid]
    subdivisions = np.ceil(np.diff(np.log(grid)) / np.log(2.)).astype(int)
    for index in np.flatnonzero(subdivisions > 1):
        pieces.append(np.geomspace(grid[index], grid[index + 1],
                                   int(subdivisions[index]) + 1)[1:-1])
    return np.unique(np.concatenate(pieces))


def _integrals(grid, flux, response):
    """Integrate T/lambda and fnu*T/lambda for piecewise-linear inputs.

    Merged input knots preserve narrow sampled lines and response edges. Within
    each interval eight-point Gaussian quadrature evaluates the linear-input
    product divided by wavelength, rather than weighting each sample equally.
    """
    half_width = np.diff(grid) / 2.
    wavelength = (grid[:-1] + half_width)[:, None] + half_width[:, None] * _NODES
    throughput = response[:-1, None] + np.diff(response)[:, None] * _FRACTIONS
    weighted_response = half_width[:, None] * _WEIGHTS * throughput / wavelength
    denominator = float(np.sum(weighted_response))
    if flux is None:
        return denominator, None
    fnu = flux[:-1, None] + np.diff(flux)[:, None] * _FRACTIONS
    return denominator, float(np.sum(weighted_response * fnu))


def _abmag(fnu_cgs):
    return float(-2.5 * np.log10(fnu_cgs) - 48.6) if fnu_cgs > 0 else np.nan


def measure_photometry(wave_aa, fnu_cgs, response_wave_aa, response, *,
                       area_cm2, exposure_s, n_exposures, aperture_fraction,
                       n_pixels, sky_rate_e_s_pixel, dark_current_e_s_pixel,
                       read_noise_e_pixel, rng=None, systematic_mag=0.02,
                       detection_snr=5.0):
    """Measure one band and draw a sky-subtracted aperture observation.

    Parameters
    ----------
    wave_aa, response_wave_aa : 1-D arrays or wavelength Quantities
        Strictly increasing observed wavelengths, in Angstrom without units.
    fnu_cgs : 1-D array or spectral-flux-density Quantity
        Noiseless total-source f_nu in erg/s/cm^2/Hz; same shape as wave_aa.
        Negative/non-finite model values invalidate the measurement, without
        clipping, absolute values, or a Poisson draw.
    response : 1-D array or dimensionless Quantity
        Full photon-to-electron fractional throughput (0 <= T <= 1), sampled at
        response_wave_aa. No extra QE or arbitrary flux calibration is applied.
    area_cm2, exposure_s : positive scalars or Quantities
        Collecting area in cm^2 and time in seconds of *each* exposure.
    n_exposures : positive integer
        Independent exposures/readouts. Total time is exposure_s*n_exposures.
    aperture_fraction : scalar in (0, 1]
        Achromatic fraction of total-source light enclosed by the aperture.
        Both model and measured fluxes are calibrated to the total source.
    n_pixels : positive float
        Aperture area in detector pixels (a geometric effective area is allowed).
        This is an unweighted aperture-noise approximation, not arbitrary pixel
        weights or a correlated/drizzled image noise model.
    sky_rate_e_s_pixel, dark_current_e_s_pixel : non-negative scalars
        Already detected electron rates per second per pixel. Quantities may use
        electron/(s*pixel). Expected sky and dark levels are subtracted exactly;
        their Poisson fluctuations remain, with no background-estimation term.
    read_noise_e_pixel : non-negative scalar
        RMS electrons in each detector pixel in one exposure; Quantity unit is
        electron (a per-pixel RMS, not an integrated aperture noise).
    rng : NumPy RandomState, Generator, or compatible object, optional
        Must provide poisson and normal methods. None uses np.random globally.
        Seed provenance belongs to the caller; no internal reseeding occurs.
    systematic_mag : non-negative scalar
        Calibration uncertainty in mag, recorded as a separate first-order flux
        scale uncertainty. It is not sampled as independent source noise.
    detection_snr : positive scalar
        Threshold on *observed* statistical S/N, default 5.

    Returns
    -------
    dict
        Flux densities/errors are total-source microJy, magnitudes are AB.
        flux_error_ujy is statistical 1-sigma; total_flux_error_ujy additionally
        summarizes the separate calibration term in quadrature. snr and
        expected_snr use statistical noise only. valid means a calculable
        measurement, not detection. Non-positive observed flux is preserved,
        with observed_mag/error NaN. Partial band coverage returns valid=False,
        reason/flag and NaN measurements, without extrapolating or drawing RNG.
        Geometric/noise inputs are validated before this availability check.
    """
    wave = _wavelengths(wave_aa, 'wave_aa')
    response_wave = _wavelengths(response_wave_aa, 'response_wave_aa')
    flux = _value(fnu_cgs, FNU_CGS, 'fnu_cgs')
    throughput = _value(response, u.dimensionless_unscaled, 'response')
    if flux.shape != wave.shape:
        raise ValueError('fnu_cgs must have the same 1-D shape as wave_aa')
    if throughput.shape != response_wave.shape:
        raise ValueError('response must have the same 1-D shape as response_wave_aa')
    if (not np.all(np.isfinite(throughput)) or np.any(throughput < 0)
            or np.any(throughput > 1) or not np.any(throughput > 0)):
        raise ValueError('response must be finite, in [0, 1], and non-zero somewhere')

    area = _scalar(area_cm2, u.cm ** 2, 'area_cm2', positive=True)
    exposure = _scalar(exposure_s, u.s, 'exposure_s', positive=True)
    exposures = _scalar(n_exposures, u.dimensionless_unscaled, 'n_exposures', positive=True)
    if isinstance(n_exposures, (bool, np.bool_)) or int(exposures) != exposures:
        raise ValueError('n_exposures must be a positive integer')
    exposures = int(exposures)
    fraction = _scalar(aperture_fraction, u.dimensionless_unscaled, 'aperture_fraction', positive=True)
    if fraction > 1:
        raise ValueError('aperture_fraction must not exceed 1')
    pixels = _scalar(n_pixels, u.pixel, 'n_pixels', positive=True)
    sky_rate = _scalar(sky_rate_e_s_pixel, u.electron / (u.s * u.pixel),
                       'sky_rate_e_s_pixel', nonnegative=True)
    dark_rate = _scalar(dark_current_e_s_pixel, u.electron / (u.s * u.pixel),
                        'dark_current_e_s_pixel', nonnegative=True)
    read_noise = _scalar(read_noise_e_pixel, u.electron, 'read_noise_e_pixel', nonnegative=True)
    systematic = _scalar(systematic_mag, u.mag, 'systematic_mag', nonnegative=True)
    threshold = _scalar(detection_snr, u.dimensionless_unscaled, 'detection_snr', positive=True)
    generator = np.random if rng is None else rng
    if not callable(getattr(generator, 'poisson', None)) or not callable(getattr(generator, 'normal', None)):
        raise TypeError('rng must provide poisson and normal methods')

    total_time = exposure * exposures
    numeric_keys = ('model_flux_ujy', 'model_mag', 'observed_flux_ujy',
                    'flux_error_ujy', 'statistical_flux_error_ujy',
                    'systematic_flux_error_ujy', 'total_flux_error_ujy',
                    'model_mag_error', 'observed_mag', 'observed_mag_error',
                    'snr', 'expected_snr', 'source_counts_e', 'observed_counts_e',
                    'sky_counts_e', 'dark_counts_e', 'read_variance_e2', 'variance_e2')
    result = {key: np.nan for key in numeric_keys}
    result.update(valid=False, detected=False, flag=0, reason='unavailable',
                  coverage_complete=False, coverage_fraction=0., detection_snr=threshold,
                  total_exposure_s=total_time, aperture_fraction=fraction,
                  n_pixels=pixels, systematic_mag=systematic)

    positive = np.flatnonzero(throughput > 0)
    lower = response_wave[max(0, int(positive[0]) - 1)]
    upper = response_wave[min(len(response_wave) - 1, int(positive[-1]) + 1)]
    band_grid = _refine_wide_intervals(np.unique(np.r_[
        lower, response_wave[(response_wave > lower) & (response_wave < upper)], upper]))
    efficiency, _ = _integrals(band_grid, None, np.interp(band_grid, response_wave, throughput))
    if not np.isfinite(efficiency) or efficiency <= 0:
        raise ValueError('response has no finite, positive integrated photon efficiency')
    result['response_efficiency'] = efficiency
    overlap_low, overlap_high = max(lower, wave[0]), min(upper, wave[-1])
    if overlap_high > overlap_low:
        covered_grid = _refine_wide_intervals(np.unique(np.r_[
            overlap_low, response_wave[(response_wave > overlap_low) & (response_wave < overlap_high)],
            overlap_high]))
        covered_efficiency, _ = _integrals(covered_grid, None,
                                          np.interp(covered_grid, response_wave, throughput))
        result['coverage_fraction'] = min(1., covered_efficiency / efficiency)
    if wave[0] > lower or wave[-1] < upper:
        result.update(flag=FLAG_INCOMPLETE_COVERAGE, reason='incomplete_spectral_coverage')
        return result
    result.update(coverage_complete=True, coverage_fraction=1.)
    if not np.all(np.isfinite(flux)):
        result.update(flag=FLAG_NONFINITE_MODEL, reason='nonfinite_model_flux')
        return result
    if np.any(flux < 0):
        result.update(flag=FLAG_NEGATIVE_MODEL, reason='negative_model_flux')
        return result

    merged_grid = _refine_wide_intervals(np.unique(np.r_[
        band_grid, wave[(wave > lower) & (wave < upper)]]))
    efficiency, integral = _integrals(merged_grid, np.interp(merged_grid, wave, flux),
                                    np.interp(merged_grid, response_wave, throughput))
    mean_fnu = integral / efficiency
    source_counts = area * total_time * fraction * integral / PLANCK_ERG_S
    sky_counts = pixels * sky_rate * total_time
    dark_counts = pixels * dark_rate * total_time
    read_variance = pixels * exposures * read_noise ** 2
    poisson_mean = source_counts + sky_counts + dark_counts
    variance = poisson_mean + read_variance
    conversion = area * total_time * fraction * efficiency * UJY_CGS / PLANCK_ERG_S
    if (not np.all(np.isfinite([poisson_mean, variance, conversion]))
            or conversion <= 0):
        raise ValueError('electron-count expectation or calibration is not finite/positive')
    observed_counts = float(generator.poisson(poisson_mean)) - sky_counts - dark_counts
    if read_variance > 0:
        observed_counts += float(generator.normal(0., np.sqrt(read_variance)))
    observed_flux = observed_counts / conversion
    sigma_flux = np.sqrt(variance) / conversion
    sigma_systematic = mean_fnu / UJY_CGS * systematic / MAG_DERIVATIVE
    expected_snr = source_counts / np.sqrt(variance) if variance > 0 else 0.
    snr = observed_counts / np.sqrt(variance) if variance > 0 else 0.
    observed_mag = _abmag(observed_flux * UJY_CGS)
    flag = 0
    if snr < threshold:
        flag |= FLAG_BELOW_DETECTION
    if observed_flux <= 0:
        flag |= FLAG_NONPOSITIVE_OBSERVED
    result.update(
        model_flux_ujy=mean_fnu / UJY_CGS, model_mag=_abmag(mean_fnu),
        observed_flux_ujy=observed_flux, flux_error_ujy=sigma_flux,
        statistical_flux_error_ujy=sigma_flux, systematic_flux_error_ujy=sigma_systematic,
        total_flux_error_ujy=float(np.hypot(sigma_flux, sigma_systematic)),
        model_mag_error=MAG_DERIVATIVE / expected_snr if expected_snr > 0 else np.nan,
        observed_mag=observed_mag,
        observed_mag_error=MAG_DERIVATIVE / snr if snr > 0 else np.nan,
        snr=snr, expected_snr=expected_snr, valid=True,
        detected=bool(snr >= threshold and observed_flux > 0), flag=flag,
        reason=('nonpositive_observed_flux' if observed_flux <= 0 else
                'below_detection_threshold' if snr < threshold else 'ok'),
        source_counts_e=source_counts, observed_counts_e=observed_counts,
        sky_counts_e=sky_counts, dark_counts_e=dark_counts,
        read_variance_e2=read_variance, variance_e2=variance,
        response_efficiency=efficiency)
    return result
