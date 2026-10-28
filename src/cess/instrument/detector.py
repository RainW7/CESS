"""Detector noise, background and registration methods retained from C13.

The static covariance helper historically resolves the public emulator class.
The composition root binds that class explicitly; this module never imports the
engine and therefore introduces no reverse import or configuration I/O.
"""
import warnings
from numbers import Real
import numpy as np
from cess.spectra.numerics import trapezoid

from cess.constants import h
import cess.spectra.units as utils
from cess.spectra.integration import integrate_response, response_curve
from cess.instrument.spectral import window_indices


hst_vband_bkg = { 
    ### (longitude, latitude): V-mag per arcsec^2, SA stands for Solar Avoidance zone (HST pointing within 55 deg of the sun)
    ### Define the lists of ecliptic latitude, longitude, and magnitude
    ### latitudes = [0, 15, 30, 45, 60, 75, 90]
    ### longitudes = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 160, 165, 180]
    ### magnitudes = [21.3, 21.7, 21.9, 22.0, 22.1, 22.2, 22.3, 22.4, 22.5, 22.6, 22.7, 22.8, 22.9, 23.0, 23.1, 23.2, 23.3, 23.4]

    ### original data sets for low longitude and latitude
    ### (0, 0) : 'SA',   (15, 0):  'SA',   (30, 0):  'SA',   (45, 0):  'SA', 
    ### (0, 15): 'SA',   (15, 15): 'SA',   (30, 15): 'SA',   (45, 15): 'SA',
    ### (0, 30): 'SA',   (15, 30): 'SA',   (30, 30): 'SA', 
    ### (0, 45): 'SA',   (15, 45): 'SA', 
    (0, 0) : 20.0, (15, 0):  20.3, (30, 0):  20.7, (45, 0):  21.0, (60, 0):  21.3, (75, 0):  21.7, (90, 0):  22.0, (105, 0):  22.2, (120, 0):  22.4, (135, 0):  22.4, (150, 0):  22.4, (165, 0):  22.3, (180, 0):  22.1, 
    (0, 15): 20.7, (15, 15): 21.0, (30, 15): 21.3, (45, 15): 21.6, (60, 15): 21.9, (75, 15): 22.2, (90, 15): 22.3, (105, 15): 22.5, (120, 15): 22.6, (135, 15): 22.6, (150, 15): 22.6, (165, 15): 22.5, (180, 15): 22.4, 
    (0, 30): 21.5, (15, 30): 21.6, (30, 30): 21.8, (45, 30): 22.1, (60, 30): 22.4, (75, 30): 22.6, (90, 30): 22.7, (105, 30): 22.9, (120, 30): 22.9, (135, 30): 22.9, (150, 30): 22.9, (165, 30): 22.8, (180, 30): 22.7, 
    (0, 45): 22.3, (15, 45): 22.0, (30, 45): 22.3, (45, 45): 22.5, (60, 45): 22.7, (75, 45): 22.9, (90, 45): 23.0, (105, 45): 23.1, (120, 45): 23.2, (135, 45): 23.2, (150, 45): 23.1, (165, 45): 23.0, (180, 45): 23.0, 
    (0, 60): 22.6, (15, 60): 22.6, (30, 60): 22.7, (45, 60): 22.9, (60, 60): 23.0, (75, 60): 23.1, (90, 60): 23.2, (105, 60): 23.3, (120, 60): 23.3, (135, 60): 23.3, (150, 60): 23.3, (165, 60): 23.2, (180, 60): 23.2, 
    (0, 75): 23.0, (15, 75): 23.1, (30, 75): 23.1, (45, 75): 23.1, (60, 75): 23.2, (75, 75): 23.2, (90, 75): 23.3, (105, 75): 23.3, (120, 75): 23.3, (135, 75): 23.4, (150, 75): 23.4, (165, 75): 23.4, (180, 75): 23.4, 
    (0, 90): 23.3, (15, 90): 23.3, (30, 90): 23.3, (45, 90): 23.3, (60, 90): 23.3, (75, 90): 23.3, (90, 90): 23.3, (105, 90): 23.3, (120, 90): 23.3, (135, 90): 23.3, (150, 90): 23.3, (165, 90): 23.3, (180, 90): 23.3, 
}

longitudes = np.unique(np.array(list(hst_vband_bkg.keys()))[:,0])
latitudes = np.unique(np.array(list(hst_vband_bkg.keys()))[:,1])


class DetectorMixin:
    @staticmethod
    def _resample_pixel_shift(values, offset, variance=None, mask=None):
        """Resample a 1-D array on the same pixel grid, using linear weights.

        Positive offsets move a feature to larger pixel index: output[j] samples
        input[j-offset].  Independent input variances use squared interpolation
        weights.  Out-of-bounds or invalid inputs yield NaN and a False mask.
        The induced covariance between neighboring output pixels is not returned.
        Zero offset is an exact copy; flux conservation only applies away from
        finite-grid boundary losses.  This helper accepts signed extracted flux.
        """
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or not np.isfinite(offset):
            raise ValueError('Expected a 1-D array and a finite offset in pixels.')
        valid_input = np.isfinite(values)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != values.shape:
                raise ValueError('mask and values must have the same shape.')
            valid_input &= mask
        if variance is not None:
            variance = np.asarray(variance, dtype=float)
            if variance.shape != values.shape:
                raise ValueError('variance and values must have the same shape.')
            valid_input &= np.isfinite(variance) & (variance >= 0)
        if offset == 0:
            shifted = np.where(valid_input, values, np.nan)
            shifted_var = None if variance is None else np.where(valid_input, variance, np.nan)
            return shifted, shifted_var, valid_input.copy()
        n = values.size
        if n == 0:
            return values.copy(), None if variance is None else variance.copy(), valid_input
        position = np.arange(n, dtype=float) - float(offset)
        inside = (position >= 0) & (position <= n - 1)
        left = np.floor(position).astype(int)
        right_weight = position - left
        left = np.clip(left, 0, n - 1)
        right = np.clip(left + 1, 0, n - 1)
        left_weight = 1 - right_weight
        valid = (inside & ((left_weight == 0) | valid_input[left]) &
                 ((right_weight == 0) | valid_input[right]))
        clean = np.where(valid_input, values, 0.)
        shifted = left_weight * clean[left] + right_weight * clean[right]
        shifted[~valid] = np.nan
        shifted_var = None
        if variance is not None:
            clean_var = np.where(valid_input, variance, 0.)
            shifted_var = left_weight ** 2 * clean_var[left] + right_weight ** 2 * clean_var[right]
            shifted_var[~valid] = np.nan
        return shifted, shifted_var, valid

    @staticmethod
    def _sample_wave_offset(rng=None):
        """Draw the legacy zero-point-error mixture, in pixel units.

        Its integer component is a sigma=0.4-pixel discrete Gaussian on [-3,3],
        followed by a uniform subpixel component and 0.1-pixel rounding.  This
        empirical mixture is retained; it is not a validated CSST calibration PDF.
        """
        rng = np.random if rng is None else rng
        integer = int(rng.choice(np.arange(-3, 4), p=utils.gaussian(3, 0.4)))
        return round(float(rng.uniform(integer - 0.5, integer + 0.5)), 1)

    def wave_err(self, flux, rng=None, offset=None):
        """Return the applied pixel offset and the spectrum resampled by it.

        This models registration on a common pixel grid, rather than changing the
        wavelength-axis metadata.  Boundary samples without input support are NaN.
        Pass offset=0 for exact identity, or a fixed offset for reproducible tests.
        For errors/masks use _resample_pixel_shift instead of shifting S/N as flux.
        """
        wave_off = self._sample_wave_offset(rng) if offset is None else float(offset)
        shifted, _, _ = self._resample_pixel_shift(flux, wave_off)
        return wave_off, shifted

    def match_bkg_mag(self, ra, dec):
        """Return a background reference label, or the explicit legacy lookup.

        fixed_template (the default) ignores RA/Dec and returns background_v_mag
        as a provenance label only. The supplied template is used at its own
        normalization times background_scale; that label does not recalibrate it.
        legacy_radec reproduces the historical, physically incorrect RA/Dec-as-
        ecliptic lookup, including its ad hoc solar-avoidance replacements. It is
        retained solely for explicit comparisons and emits a warning once per
        instance. Neither mode implements time/solar-relative zodiacal geometry.
        """
        if self.config['background_mode'] == 'fixed_template':
            return self.config['background_v_mag']
        if not self._legacy_background_warning_issued:
            warnings.warn(
                'legacy_radec treats equatorial RA/Dec as ecliptic coordinates and '
                'uses historical ad hoc solar-avoidance entries; this is not a '
                'validated zodiacal-background model.', RuntimeWarning, stacklevel=2)
            self._legacy_background_warning_issued = True
        if not np.isfinite(ra) or not np.isfinite(dec):
            raise ValueError('legacy_radec requires finite RA and Dec.')
        lon = 360 - ra if ra > 180 else ra
        lat = np.abs(dec)
        lon_idx = utils.find_nearest(longitudes, lon)
        lat_idx = utils.find_nearest(latitudes, lat)
        key = (longitudes[lon_idx], latitudes[lat_idx])
        return hst_vband_bkg[key]

    def bkg_per_pix_time(self, ra, dec, bkgwave, bkgflux, tp):
        """Integrate a sky template into e-/s/detector pixel.

        bkgwave is Angstrom; bkgflux is erg/s/cm**2/Hz/arcsec**2; tp is a
        dimensionless photon-to-electron response on that same wavelength grid.
        The default fixed_template assumption uses template*background_scale and
        ignores RA/Dec. background_v_mag is only a label in this mode. The input
        template is not asserted to be a validated physical CSST sky model.
        Explicit legacy_radec applies the deprecated coordinate-dependent factor;
        see match_bkg_mag. No observing-time or solar-relative geometry is modeled.
        """
        bkgwave = np.asarray(bkgwave, dtype=float)
        bkgflux = np.asarray(bkgflux, dtype=float)
        tp = np.asarray(tp, dtype=float)
        if (bkgwave.ndim != 1 or bkgwave.size < 2 or bkgflux.shape != bkgwave.shape or
                tp.shape != bkgwave.shape or np.any(~np.isfinite(bkgwave)) or
                np.any(bkgwave <= 0) or np.any(np.diff(bkgwave) <= 0) or
                np.any(~np.isfinite(bkgflux)) or np.any(bkgflux < 0) or
                np.any(~np.isfinite(tp)) or np.any(tp < 0)):
            raise ValueError('Sky wavelength/flux/response must be finite aligned 1-D arrays, '
                             'with increasing positive wavelength and nonnegative flux/response.')
        scale = self.config['background_scale']
        if self.config['background_mode'] == 'legacy_radec':
            scale *= 10 ** (0.4 * (self.config['background_v_mag'] - self.match_bkg_mag(ra, dec)))
        photons = utils.fnu2fphot(bkgwave, bkgflux * scale)
        return float(trapezoid(tp * photons, bkgwave) * self.colarea *
                     self.config['arcsecperpix'] ** 2)

    def native_sky_rate(self, ra, dec, bkgwave, bkgflux, response_wave, response,
                        apply_position_scale=True):
        """Uniform diffuse-sky rate, e-/s/pixel, over the native response range.

        Sky fnu is erg/s/cm**2/Hz/arcsec**2 and wavelength is observed Angstrom.
        Integrates separate piecewise-linear sky/response on merged knots across
        the *entire supplied* response interval, not just spectral output bins.
        Uniform slitless illumination uses the full band integral for each sky
        pixel. The model ignores field boundaries, trace variation and sky
        structure; no calibration outside the supplied response range is implied.

        apply_position_scale=False returns the fixed template reference including
        background_scale, allowing runners to cache it before legacy scaling.
        """
        wave = np.asarray(bkgwave, dtype=float)
        flux = np.asarray(bkgflux, dtype=float)
        if (wave.ndim != 1 or wave.size < 2 or flux.shape != wave.shape or
                np.any(~np.isfinite(wave)) or np.any(wave <= 0) or
                np.any(np.diff(wave) < 0) or np.any(~np.isfinite(flux)) or np.any(flux < 0)):
            raise ValueError('Sky wavelength/fnu must be finite aligned 1-D arrays, '
                             'with positive nondecreasing wavelength and nonnegative flux.')
        if not isinstance(apply_position_scale, (bool, np.bool_)):
            raise ValueError('apply_position_scale must be boolean.')
        scale = self.config['background_scale']
        if apply_position_scale and self.config['background_mode'] == 'legacy_radec':
            scale *= 10 ** (.4 * (self.config['background_v_mag'] - self.match_bkg_mag(ra, dec)))
        response_wave, response = response_curve(response_wave, response)
        if wave[0] > response_wave[0] or wave[-1] < response_wave[-1]:
            raise ValueError('Sky must cover the full response calibration range.')
        # The supplied background has concatenation seams with repeated lambda
        # and sometimes different fnu. Preserve the ordered left/right limits:
        # a vertical jump has zero integral, unlike averaging the adjacent ramps.
        cuts = np.r_[0, np.flatnonzero(np.diff(wave) == 0) + 1, wave.size]
        proxy = 0.
        for start, stop in zip(cuts[:-1], cuts[1:]):
            if stop - start < 2:
                continue
            lower = max(wave[start], response_wave[0])
            upper = min(wave[stop - 1], response_wave[-1])
            if lower >= upper:
                continue
            native_response = np.r_[lower, response_wave[(response_wave > lower) &
                                                       (response_wave < upper)], upper]
            transmission = np.interp(native_response, response_wave, response)
            proxy += integrate_response(wave[start:stop], flux[start:stop],
                                        native_response, transmission)
        return float(proxy * scale / h * self.colarea * self.config['arcsecperpix'] ** 2)

    def _noise_realization(self, ra, dec, wave, fnu, bkgwave, bkgflux, tp,
                           height, width, apply_morphology_effect, rng=None,
                           source_rate_e_s_cm2=None, sky_rate_e_s_pixel=None,
                           source_fraction=1.0, extraction_n_pixels=None,
                           normalize_morphology_kernel=True):
        """Generate independent, equally weighted extracted-column electron counts.

        The model uses known sky/dark means and independent Poisson source, sky
        and dark fluctuations. Read-noise variance is RN**2*N_exp*N_extract,
        with RN in e-/pixel/exposure; the realization uses the same Gaussian
        variance. Background-estimation uncertainty and single-pixel saturation
        are not modeled. In particular, the old 90000-e- cap on a cumulative
        extracted column is removed, and background-subtracted counts may be
        negative. Morphology redistributes source electron bins with a normalized profile,
        without restoring finite-array boundaries. source_fraction explicitly
        scales the response-folded, spatially redistributed source electron bins
        before their Poisson realization/variance. It is finite in [0,1]; the
        default 1 retains the legacy full-source model. No fraction is inferred
        from height.sum() or read implicitly from configuration. Its physical
        value needs an independently validated morphology/extraction model.
        extraction_n_pixels is a positive integral count of equally weighted
        extraction pixels, defaulting to height.size. It scales sky, dark and
        read variance, independently of source_fraction and dispersion broadening.
        normalize_morphology_kernel=False preserves the finite mass of an
        aperture-conditioned kernel; source_fraction still applies once. The
        default retains the historical unit-sum dispersion-kernel convention.

        source_rate_e_s_cm2 optionally supplies the response-folded electron
        rate per collecting area, one value per detector bin. It is multiplied
        only by collecting area and total exposure. Nominal center throughput
        is then used only for flux calibration, not for source counts. A zero
        center throughput can still receive electrons through LSF spillover;
        its detector model remains available while calibrated fnu is undefined.
        sky_rate_e_s_pixel optionally supplies a native-response sky integral.
        """
        rng = np.random if rng is None else rng
        wave = np.asarray(wave, dtype=float)
        fnu = np.asarray(fnu, dtype=float)
        tp = np.asarray(tp, dtype=float)
        height = np.asarray(height, dtype=float)
        if (wave.ndim != 1 or wave.size < 2 or np.any(~np.isfinite(wave)) or
                np.any(wave <= 0) or np.any(np.diff(wave) <= 0)):
            raise ValueError('wave must be positive, finite and strictly increasing.')
        if fnu.shape != wave.shape or tp.shape != wave.shape:
            raise ValueError('fnu and throughput must match the 1-D wavelength grid.')
        if (height.ndim != 1 or height.size == 0 or
                np.any(~np.isfinite(height)) or np.any(height < 0)):
            raise ValueError('height must be a nonempty, finite, nonnegative profile.')
        if (isinstance(source_fraction, (bool, np.bool_)) or not isinstance(source_fraction, Real) or
                not np.isfinite(source_fraction) or not 0 <= source_fraction <= 1):
            raise ValueError('source_fraction must be a finite numeric value in [0,1], not boolean.')
        source_fraction = float(source_fraction)
        if extraction_n_pixels is None:
            extraction_n_pixels = height.size
        elif (isinstance(extraction_n_pixels, (bool, np.bool_)) or not isinstance(extraction_n_pixels, Real) or
                not np.isfinite(extraction_n_pixels) or extraction_n_pixels <= 0 or
                int(extraction_n_pixels) != extraction_n_pixels):
            raise ValueError('extraction_n_pixels must be a positive integer, not boolean.')
        extraction_n_pixels = int(extraction_n_pixels)
        if not isinstance(apply_morphology_effect, (bool, np.bool_)):
            raise ValueError('apply_morphology_effect must be boolean.')
        if not isinstance(normalize_morphology_kernel, (bool, np.bool_)):
            raise ValueError('normalize_morphology_kernel must be boolean.')
        if apply_morphology_effect:
            width = np.asarray(width, dtype=float)
            if (width.ndim != 1 or width.size == 0 or np.any(~np.isfinite(width)) or
                    np.any(width < 0) or width.sum() <= 0):
                raise ValueError('width must have finite nonnegative weights and positive sum.')
        # Bin-equivalent fnu converts directly to photon/electron bin integrals.
        # Apply spatial redistribution to detected electrons, after the spectral
        # response integral (or legacy center response fallback).
        dlambda = np.gradient(wave)
        safe_tp = np.where(np.isfinite(tp) & (tp >= 0), tp, 0.)
        source = np.zeros_like(wave)
        folded_source = source_rate_e_s_cm2 is not None
        if folded_source:
            rate = np.asarray(source_rate_e_s_cm2, dtype=float)
            if rate.shape != wave.shape:
                raise ValueError('source_rate_e_s_cm2 must match wave.')
            if np.any(np.isinf(rate)) or np.any(rate[np.isfinite(rate)] < 0):
                raise ValueError('source_rate_e_s_cm2 must be nonnegative; missing bins may be NaN.')
            known = np.isfinite(rate)
            valid = known.copy()
            source[valid] = rate[valid] * self.colarea * self.config['expt'] * self.config['expnum']
        else:
            known = np.isfinite(fnu) & (fnu >= 0) & np.isfinite(tp) & (tp >= 0)
            valid = known & (tp > 0)
            source[valid] = (utils.fnu2fphot(wave[valid], fnu[valid]) * self.colarea *
                             self.config['expt'] * self.config['expnum'] * dlambda[valid] * safe_tp[valid])
        if apply_morphology_effect:
            source = self.lamshift_width(source, width, normalize=normalize_morphology_kernel)
            # Missing input support cannot be silently converted to a zero-flux
            # measurement. All nonzero profile contributors must be known.
            center = round((len(width) - 1) / 2.)
            supported = np.ones(wave.size, dtype=bool)
            for index in np.flatnonzero(width > 0):
                shift = int(index - center)
                positions = np.arange(wave.size) + shift
                inside = (positions >= 0) & (positions < wave.size)
                contributor = np.zeros(wave.size, dtype=bool)
                contributor[inside] = known[positions[inside]]
                supported &= contributor
            # Calibration uses the response at the destination. A missing
            # original center is irrelevant when its profile weight is zero.
            valid = supported if folded_source else np.isfinite(tp) & (tp > 0) & supported
        # The explicit aperture fraction affects source signal and shot noise
        # once. Width redistribution is already complete; sky/dark/read are
        # properties of the selected detector pixels, not of this light fraction.
        if source_fraction != 1.:
            source *= source_fraction
        sky_rate = (self.bkg_per_pix_time(ra, dec, bkgwave, bkgflux, safe_tp)
                    if sky_rate_e_s_pixel is None else sky_rate_e_s_pixel)
        if np.ndim(sky_rate) != 0 or not np.isfinite(sky_rate) or sky_rate < 0:
            raise ValueError('Sky electron rate must be finite and nonnegative.')
        if (not np.isfinite(self.config['dc_value']) or self.config['dc_value'] < 0 or self.config['expt'] <= 0 or
                self.config['expnum'] < 1 or int(self.config['expnum']) != self.config['expnum']):
            raise ValueError('Require nonnegative dark current and positive exposure settings.')
        read_noise = float(self.config.get('read_noise', 5.))
        if not np.isfinite(read_noise) or read_noise < 0:
            raise ValueError('read_noise must be finite and nonnegative, in e-/pixel/exposure.')
        sky = sky_rate * self.config['expt'] * self.config['expnum'] * extraction_n_pixels
        dark = self.config['dc_value'] * self.config['expt'] * self.config['expnum'] * extraction_n_pixels
        read_variance = read_noise ** 2 * self.config['expnum'] * extraction_n_pixels
        expectation = source + sky + dark
        # numpy's Poisson sampler cannot represent arbitrary huge model counts.
        valid &= np.isfinite(expectation) & (expectation < 9e18)
        observed = np.full_like(wave, np.nan)
        observed[valid] = (rng.poisson(expectation[valid]) - sky - dark +
                           rng.normal(0., np.sqrt(read_variance), np.count_nonzero(valid)))
        variance = source + sky + dark + read_variance
        source[~valid] = np.nan
        variance[~valid] = np.nan
        return dict(wave=wave, tp=tp, dlambda=dlambda, model=source,
                    observed=observed, variance=variance, valid=valid,
                    source_fraction=source_fraction, extraction_n_pixels=extraction_n_pixels)

    @staticmethod
    def _resample_covariance_lag1(variance, offset, mask=None):
        """Adjacent covariance from linear registration of independent bins.

        ``variance`` has the input bin-count squared unit. The output has shape
        ``(n_pixel-1,)`` and element j is Cov(output[j], output[j+1]). Constant
        pixel shifts use at most two input bins, so all longer lags are zero.
        An unsupported endpoint makes the pair NaN, including for zero shift.
        This describes registration of independent extracted-bin noise only;
        it is not a model for additional detector or extraction correlations.
        """
        variance = np.asarray(variance, dtype=float)
        _, _, valid = emulator._resample_pixel_shift(
            np.zeros_like(variance), offset, variance, mask)
        n = variance.size
        if n < 2:
            return np.empty(0, dtype=float)
        position = np.arange(n, dtype=float) - float(offset)
        floor = np.floor(position)
        right_weight = position - floor
        left = np.clip(floor.astype(int), 0, n - 1)
        right = np.clip(left + 1, 0, n - 1)
        weights = (1. - right_weight, right_weight)
        indices = (left, right)
        clean = np.where(np.isfinite(variance) & (variance >= 0), variance, 0.)
        covariance = np.zeros(n - 1, dtype=float)
        for a in (0, 1):
            for b in (0, 1):
                common = indices[a][:-1] == indices[b][1:]
                covariance += (common * weights[a][:-1] * weights[b][1:] *
                               clean[indices[a][:-1]])
        covariance[~(valid[:-1] & valid[1:])] = np.nan
        return covariance

    def _registered_covariance(self, realization, offset):
        """Return lag-one covariance on the nominal calibrated fnu grid."""
        covariance = self._resample_covariance_lag1(
            realization['variance'], offset, realization['valid'])
        denominator = (self.colarea * self.config['expt'] * self.config['expnum'] *
                       realization['dlambda'] * realization['tp'])
        conversion = np.full_like(realization['wave'], np.nan)
        np.divide(h * realization['wave'], denominator, out=conversion,
                  where=np.isfinite(denominator) & (denominator > 0))
        return covariance * conversion[:-1] * conversion[1:]

    def _noise_product(self, realization, offset, grism):
        """Register model/counts/variance on one pixel grid and calibrate to fnu.

        WCE is explicitly a resampling of the extracted electron spectrum with
        the nominal output wavelength/response calibration held fixed. Positive
        offset moves features to larger pixel index. Variances use squared
        weights; add_noise(return_covariance=True) also returns the induced
        neighbor covariance. Longer lags vanish for this registration model.
        Both model and noisy products use the same offset and support mask.
        """
        model, _, model_valid = self._resample_pixel_shift(
            realization['model'], offset, mask=realization['valid'])
        observed, variance, valid = self._resample_pixel_shift(
            realization['observed'], offset, realization['variance'], realization['valid'])
        valid &= model_valid
        wave = realization['wave']
        conversion = np.full_like(wave, np.nan)
        denominator = self.colarea * self.config['expt'] * self.config['expnum'] * realization['dlambda'] * realization['tp']
        response_valid = np.isfinite(denominator) & (denominator > 0)
        valid &= response_valid
        conversion[valid] = h * wave[valid] / denominator[valid]
        # Detector counts can be known where nominal center-response flux
        # calibration is undefined (e.g. LSF spillover into a zero-T center).
        model[~model_valid] = np.nan
        observed[~valid] = np.nan
        variance[~valid] = np.nan
        fnu_with_noise = observed * conversion
        fnu_error = np.sqrt(variance) * conversion
        snr = np.full_like(wave, np.nan)
        np.divide(observed, np.sqrt(variance), out=snr, where=valid & (variance > 0))
        spec_mask = np.zeros(wave.size, dtype=bool)
        start, stop = window_indices(self.config, grism, wave)
        spec_mask[start:stop] = True
        spec_mask &= valid
        finite_snr = spec_mask & np.isfinite(snr)
        snr_mean = float(np.mean(snr[finite_snr])) if np.any(finite_snr) else np.nan
        # RMS is the theoretical per-column noise std averaged in variance space,
        # not sqrt of a noisy count (which can be negative after subtraction).
        rms = float(np.sqrt(np.mean(variance[valid]))) if np.any(valid) else np.nan
        snr_mask = finite_snr & (snr >= 1)
        return (model, fnu_with_noise, fnu_error, rms, snr, snr_mean,
                float(offset), spec_mask, snr_mask)

    def add_noise(self, ra, dec, wave, fnu, bkgwave, bkgflux, tp, height, width,
                  grism, apply_morphology_effect, apply_wavelength_calibration_error,
                  rng=None, source_rate_e_s_cm2=None, sky_rate_e_s_pixel=None,
                  return_covariance=False, source_fraction=1.0, extraction_n_pixels=None,
                  normalize_morphology_kernel=True):
        """Return model electrons, noisy fnu, fnu error and S/N on one pixel grid.

        wave is in Angstrom; fnu and its error are erg/s/cm**2/Hz. The noise
        model and limitations are documented in _noise_realization. WCE applies
        linear registration after the shared noise realization, with propagated
        diagonal variance and invalid boundary pixels (_noise_product). False
        leaves registration exactly unchanged. rng may be numpy.random, a
        RandomState or a Generator; None retains the module-level random stream.
        rms_in_e is sqrt(mean(theoretical column variance)); snr is realized
        background-subtracted signal/sigma, and snr_mean uses valid spec_mask.
        Optional native-response source/sky rates follow _noise_realization;
        calibrated flux still divides by nominal center response, not an inverse
        of the full response matrix. Model electrons retain detector support
        even where flux calibration is undefined.
        return_covariance=True appends Cov(fnu[j],fnu[j+1]), shape (n_pixel-1,),
        in (erg/s/cm**2/Hz)**2. Together with fnu_error**2 it describes the full
        within-spectrum covariance of this independent-bin/linear-shift model.
        The default positional return contract and RNG sequence are unchanged.
        source_fraction/extraction_n_pixels follow _noise_realization. Nominal
        response calibration does not divide by the light fraction: returned
        flux/error are aperture-referenced proxies, with no total-source aperture
        correction. The pre-morphology supplied source rate remains full-source.
        """
        if not isinstance(apply_wavelength_calibration_error, (bool, np.bool_)):
            raise ValueError('apply_wavelength_calibration_error must be boolean.')
        realization = self._noise_realization(
            ra, dec, wave, fnu, bkgwave, bkgflux, tp, height, width,
            apply_morphology_effect, rng, source_rate_e_s_cm2, sky_rate_e_s_pixel,
            source_fraction=source_fraction, extraction_n_pixels=extraction_n_pixels,
            normalize_morphology_kernel=normalize_morphology_kernel)
        offset = self._sample_wave_offset(rng) if apply_wavelength_calibration_error else 0.
        result = self._noise_product(realization, offset, grism)
        return result + (self._registered_covariance(realization, offset),) if return_covariance else result

    def wave_err_zouhu(self, flux, flux1, flux2, snr, rng=None, offset=None):
        """Legacy paired-spectrum wrapper with variance-aware registration.

        flux1 is noisy fnu and flux2 is its 1-sigma error. S/N is recomputed from
        shifted flux1 and squared-weight propagated variance; the input snr is
        retained for positional API compatibility and is not shifted as a flux.
        Boundary values are NaN. Adjacent covariance is not represented.
        """
        if np.shape(snr) != np.shape(flux1):
            raise ValueError('snr must have the same shape as flux1.')
        wave_off = self._sample_wave_offset(rng) if offset is None else float(offset)
        model, _, valid_model = self._resample_pixel_shift(flux, wave_off)
        noisy, variance, valid = self._resample_pixel_shift(
            flux1, wave_off, np.asarray(flux2, dtype=float) ** 2)
        valid &= valid_model
        error = np.sqrt(variance)
        shifted_snr = np.full_like(noisy, np.nan)
        np.divide(noisy, error, out=shifted_snr, where=valid & (error > 0))
        model[~valid] = np.nan
        noisy[~valid] = np.nan
        error[~valid] = np.nan
        return wave_off, model, noisy, error, shifted_snr

    def add_noise_zouhu(self, ra, dec, wave, fnu, bkgwave, bkgflux, tp, height,
                        width, grism, apply_morphology_effect,
                        apply_wavelength_calibration_error, rng=None,
                        source_rate_e_s_cm2=None, sky_rate_e_s_pixel=None,
                        return_covariance=False, source_fraction=1.0, extraction_n_pixels=None,
                        normalize_morphology_kernel=True):
        """Return paired unregistered/WCE products sharing one noise realization.

        The positional return contract is unchanged. Morphology is applied once
        only when enabled. WCE=False makes every paired array exactly identical
        (including masks); WCE=True resamples electrons and propagates variance
        before fnu/SNR calculation. The single returned rms/mean-SNR describe the
        unregistered product. The mask is the intersection of both supports.
        Background-subtracted negative signals are preserved; saturation is not
        modeled. return_covariance=True appends unregistered and WCE lag-one
        calibrated-fnu covariances. The two spectra share a realization and are
        statistically dependent; cross-variant covariance is not returned.
        Explicit source_fraction attenuates the shared source expectation/shot
        noise once; extraction_n_pixels sets sky/dark/read terms. Calibrated
        products remain aperture-referenced, without division by the fraction.
        """
        if not isinstance(apply_wavelength_calibration_error, (bool, np.bool_)):
            raise ValueError('apply_wavelength_calibration_error must be boolean.')
        realization = self._noise_realization(
            ra, dec, wave, fnu, bkgwave, bkgflux, tp, height, width,
            apply_morphology_effect, rng, source_rate_e_s_cm2, sky_rate_e_s_pixel,
            source_fraction=source_fraction, extraction_n_pixels=extraction_n_pixels,
            normalize_morphology_kernel=normalize_morphology_kernel)
        unshifted = self._noise_product(realization, 0., grism)
        offset = self._sample_wave_offset(rng) if apply_wavelength_calibration_error else 0.
        shifted = self._noise_product(realization, offset, grism)
        conversion = np.full_like(realization['wave'], np.nan)
        denominator = self.colarea * self.config['expt'] * self.config['expnum'] * realization['dlambda'] * realization['tp']
        np.divide(h * realization['wave'], denominator, out=conversion,
                  where=np.isfinite(denominator) & (denominator > 0))
        flux_wce = shifted[0] * conversion
        spec_mask = unshifted[7] & shifted[7]
        snr_mask = spec_mask & np.isfinite(shifted[4]) & (shifted[4] >= 1)
        result = (flux_wce, unshifted[0], shifted[0], unshifted[1], shifted[1],
                unshifted[2], shifted[2], unshifted[3], unshifted[4], shifted[4],
                unshifted[5], float(offset), spec_mask, snr_mask)
        if return_covariance:
            result += (self._registered_covariance(realization, 0.),
                       self._registered_covariance(realization, offset))
        return result


# Direct mixin use is valid before the public facade is imported. Once composed,
# preserve the legacy static helper's class-level monkeypatch lookup exactly.
emulator = DetectorMixin


def _bind_emulator_class(cls):
    global emulator
    emulator = cls
