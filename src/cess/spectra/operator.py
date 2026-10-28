"""Spectral-grid, LSF and morphology operators retained from C13.

Method bodies preserve the prior calculation order. This mixin imports no
engine, runner, configuration file or instrument resources.
"""
from collections.abc import Mapping
import warnings
import numpy as np

from cess.constants import h
import cess.spectra.units as utils
from cess.spectra.integration import bin_edges, convolve_to_bins, response_curve
from cess.instrument.spectral import target_fwhm, output_wavelengths, MAX_SPECTRAL_PIXELS


class SpectraMixin:
    def sls_wave_spec(self, highres_wave, highres_flux, redshift, grism):
        """Validate SED arrays and return the legacy resolution-element grid.

        Inputs are rest wavelength (n_wave,), observed-normalized fnu
        (n_source,n_wave), redshift (n_source,) and GU/GV/GI. SEDs are never sampled
        on this coarse grid. Only legacy output sampling uses its pixel count;
        explicit resolution-element sampling is defined separately. The second
        return is the requested instrumental sigma in observed
        Angstrom, not a template-corrected width. Per-source template corrections
        are computed by kernels_convolve. Final returns are valid/invalid row
        masks; wavelength coverage is checked later by kernels_convolve.
        """

        if grism == 'GU':
            start_wl = self.config['gu_start_wl']
            end_wl = self.config['gu_end_wl']
            delta_lambda = target_fwhm(self.config, grism)
        elif grism == 'GV':
            start_wl = self.config['gv_start_wl']
            end_wl = self.config['gv_end_wl']
            delta_lambda = target_fwhm(self.config, grism)
        elif grism == 'GI':
            start_wl = self.config['gi_start_wl']
            end_wl = self.config['gi_end_wl']
            delta_lambda = target_fwhm(self.config, grism)
        else:
            raise ValueError('Invalid grism, please input GU, GV, or GI.')

        # obtain the numbers of spectral resolution units based on mean delta lambda
        coarse_count = (end_wl - start_wl) / delta_lambda
        if not np.isfinite(coarse_count) or coarse_count + 1 > MAX_SPECTRAL_PIXELS:
            raise ValueError('Coarse spectral grid exceeds the maximum supported pixel count.')
        N = int(coarse_count) + 1
        # generate an array for spectral resolution unit based on the grism resolution in the corresponding wavelength range
        spectral_unit_wavelength = np.array([start_wl + i * delta_lambda for i in range(N)])

        # calculating observed wavelength
        highres_wave = np.asarray(highres_wave, dtype=float)
        highres_flux = np.asarray(highres_flux, dtype=float)
        redshift = np.asarray(redshift, dtype=float)
        if (highres_wave.ndim != 1 or highres_wave.size < 2 or
                not np.all(np.isfinite(highres_wave)) or
                np.any(highres_wave <= 0) or np.any(np.diff(highres_wave) <= 0)):
            raise ValueError('highres_wave must be finite, positive and strictly increasing.')
        if (highres_flux.ndim != 2 or highres_flux.shape[1] != highres_wave.size or
                redshift.shape != (highres_flux.shape[0],)):
            raise ValueError('Expected flux (n_source, n_wave) and redshift (n_source,).')
        # Physical intrinsic models may be zero but cannot be negative. Reject
        # the entire bad source, like NaN/Inf, before any response or RNG work;
        # a nonphysical source must not abort a catalogue in the Poisson stage.
        mask = ((redshift >= 1e-8) & (redshift <= 10) & np.isfinite(redshift) &
                np.all(np.isfinite(highres_flux), axis=1) &
                np.all(highres_flux >= 0, axis=1) &
                np.all(np.abs(highres_flux) <= 1e100, axis=1))
        mask_nan = ~mask
        # This is the requested instrumental Gaussian width, not an estimate
        # of the template LSF. Sampling intervals carry no LSF information.
        sigma = np.full_like(spectral_unit_wavelength, delta_lambda / 2.3548200450309493)
        return spectral_unit_wavelength, sigma, mask, mask_nan

    def _additional_lsf_sigma(self, wavelengths, redshift, grism, warn=True):
        """Additional Gaussian sigma in observed Angstrom, at source wavelength.

        Only the upstream *instrumental* template LSF is subtracted in variance;
        intrinsic astrophysical line widths must never be subtracted. Constant
        rest FWHM scales by (1+z); constant resolving power uses observed lambda/R.
        unknown applies the full requested kernel and cannot certify a final LSF.
        preserve explicitly keeps an overbroad input instead of deconvolving it.
        """
        wavelengths = np.asarray(wavelengths, dtype=float)
        target = target_fwhm(self.config, grism)
        model = self.config['template_lsf']
        mode = model['mode']
        if mode == 'unknown':
            if warn and not getattr(self, '_unknown_lsf_warned', False):
                warnings.warn('Template LSF shape is unknown to the operator: applying the full instrumental '
                              'kernel as additional broadening; final spectral resolution '
                              'is not certified. Set template_lsf from upstream evidence.',
                              RuntimeWarning, stacklevel=3)
                self._unknown_lsf_warned = True
            template = np.zeros_like(wavelengths)
        elif mode == 'delta':
            template = np.zeros_like(wavelengths)
        elif mode == 'fwhm_rest':
            template = np.full_like(wavelengths, model['value'] * (1. + redshift))
        else:  # resolving_power, validated by cess_config
            template = wavelengths / model['value']
        broader = template > target * (1. + 1.e-12)
        if np.any(broader):
            if self.config['template_lsf_broader_policy'] == 'error':
                raise ValueError('Template LSF is broader than requested {} LSF; '
                                 'cannot deconvolve. Use a higher-resolution template or '
                                 'explicit template_lsf_broader_policy=preserve.'.format(grism))
            key = '_broad_lsf_warned_' + grism
            if warn and not getattr(self, key, False):
                warnings.warn('Template LSF is broader than requested {} LSF: preserve '
                              'policy adds zero broadening there; requested final '
                              'resolution is not attained.'.format(grism),
                              RuntimeWarning, stacklevel=3)
                setattr(self, key, True)
        return np.sqrt(np.maximum(target ** 2 - template ** 2, 0.)) / 2.3548200450309493

    def kernels_convolve(self, highres_wave, highres_flux, redshift, grism, fwhm,
                         responses=None):
        """Integrate a Gaussian LSF directly from the original SED to pixel bins.

        Input wavelength is a strictly increasing positive rest-Angstrom vector;
        fnu is (source,wavelength), in any common flux-density unit (runners use
        uJy), interpreted as already normalized observed fnu (the existing
        BayeSED input contract). Redshift moves the wavelength axis only; it
        does not apply another luminosity/K correction. Upstream normalization
        is not independently inferred from the HDF5 unit label.
        The SED is piecewise linear in fnu on its native observed wavelength axis.

        Output keeps the legacy centers, shape and four-value return contract.
        Flux is now photon-equivalent *bin* fnu: center/bin_width times the
        redistributed integral of fnu/lambda. Thus sum(fnu*width/center)/h
        conserves unit-response photons, including for narrow resolved lines.
        It is not a point sample or an energy-conserving f_lambda convolution.
        Optional responses maps names to (observed wavelength, throughput)
        arrays. It adds a fifth return, a dict of response-folded bin fnu arrays:
        T is multiplied at incident wavelength inside the integral. These are
        converted to source rates using response_electron_rate, then passed to
        add_noise without another throughput multiplication. With responses=None
        the four-return C1 API and center-response noise fallback remain usable.

        Missing wavelength coverage within eight instrumental sigmas of output
        edges invalidates the entire source row; no constant extrapolation or
        boundary renormalization is used. Intrinsic unresolved features cannot
        be reconstructed. Template LSF treatment is configured explicitly.
        """
        coarse_wave, _, mask, _ = self.sls_wave_spec(
            highres_wave, highres_flux, redshift, grism)
        highres_wave = np.asarray(highres_wave, dtype=float)
        highres_flux = np.asarray(highres_flux, dtype=float)
        redshift = np.asarray(redshift, dtype=float)
        if not np.isfinite(fwhm) or fwhm <= 0:
            raise ValueError('fwhm must be finite and positive, in arcsec.')
        new_wave = output_wavelengths(self.config, grism, fwhm_arcsec=fwhm)
        n_pixel = len(new_wave)
        prefix = grism.lower()
        edges = bin_edges(new_wave)
        target_sigma = target_fwhm(self.config, grism) / 2.3548200450309493
        # Use a conservative common support also for zero additional broadening.
        # Restrict LSF validity checks to the region that can feed these pixels.
        support = (edges[0] - 8. * target_sigma, edges[-1] + 8. * target_sigma)
        curves = {}
        if responses is not None:
            if not isinstance(responses, Mapping):
                raise ValueError('responses must map names to (wavelength, throughput) pairs.')
            for name, curve in responses.items():
                if not isinstance(name, str) or not name or not isinstance(curve, (tuple, list)) or len(curve) != 2:
                    raise ValueError('Each response needs a nonempty string name and a two-array pair.')
                response_wave, transmission = response_curve(*curve)
                if response_wave[0] > support[0] or response_wave[-1] < support[1]:
                    raise ValueError('Response {} does not cover required {} LSF support.'.format(name, grism))
                curves[name] = (response_wave, transmission)
        new_flux = np.full((highres_flux.shape[0], n_pixel), np.nan)
        folded = {name: np.full_like(new_flux, np.nan) for name in curves}
        for i in np.flatnonzero(mask):
            observed_wave = highres_wave * (1. + redshift[i])
            if observed_wave[0] > support[0] or observed_wave[-1] < support[1]:
                mask[i] = False
                continue
            self._additional_lsf_sigma(np.asarray(support), redshift[i], grism)
            def sigma_at(wavelengths):
                return self._additional_lsf_sigma(wavelengths, redshift[i], grism, warn=False)
            sigma = sigma_at if self.config['template_lsf']['mode'] == 'resolving_power' else float(
                self._additional_lsf_sigma(new_wave[0], redshift[i], grism, warn=False))
            # Fixed input/output geometry is independent of out-of-band padding.
            new_flux[i], _ = convolve_to_bins(
                observed_wave, highres_flux[i], new_wave, sigma,
                max_sigma=target_sigma)
            for name, (response_wave, transmission) in curves.items():
                folded[name][i], _ = convolve_to_bins(
                    observed_wave, highres_flux[i], new_wave, sigma,
                    max_sigma=target_sigma, response_wave=response_wave, response=transmission)
        result = (new_wave, new_flux, mask, ~mask)
        return result if responses is None else result + (folded,)

    @staticmethod
    def response_electron_rate(wave, response_fnu_cgs):
        """Convert folded bin fnu to e-/s/cm**2, before morphology/exposure.

        response_fnu_cgs has final axis matching wave and already includes
        photon-to-electron throughput. NaN source rows remain NaN. This method
        applies no second response, area, exposure or aperture correction.
        """
        wave = np.asarray(wave, dtype=float)
        edges = bin_edges(wave)
        flux = np.asarray(response_fnu_cgs, dtype=float)
        if flux.ndim < 1 or flux.shape[-1] != wave.size:
            raise ValueError('Response-folded fnu must have a final axis matching wave.')
        return flux * np.diff(edges) / (h * wave)

    def varsmooth(self, x, y, sig_x, xout=None, oversample=1):
        """Compatibility wrapper for independently defined Gaussian smoothing.

        See ``cess.spectra.smoothing.gaussian_sample_smooth`` for the
        point-sample, column-array, finite-boundary and interpolation contract.
        This helper is not used by the production spectral-bin integrator.
        """
        from cess.spectra.smoothing import gaussian_sample_smooth
        return gaussian_sample_smooth(x, y, sig_x, xout, oversample)

    def lamshift_width(self, flux, width, *, normalize=True):
        """Redistribute integrated pixel counts, preserving finite-array losses.

        Width positions retain the legacy integer centering/orientation. Flux
        outside the finite detector array is lost, never restored or normalized
        back into the array. Apply this to electron/photon bin integrals, not to
        fnu point values. normalize=True preserves the legacy unit-sum kernel.
        False accepts an absolute/conditional finite kernel with sum<=1 within
        floating-point tolerance, retaining its omitted dispersion probability.
        It does not infer or apply a spatial aperture fraction.
        """
        flux = np.asarray(flux, dtype=float)
        width = np.asarray(width, dtype=float)
        if (flux.ndim != 1 or flux.size == 0 or width.ndim != 1 or width.size == 0 or
                np.any(~np.isfinite(width)) or np.any(width < 0) or width.sum() <= 0):
            raise ValueError('Require 1-D flux and finite nonnegative profile with positive sum.')
        if not isinstance(normalize, (bool, np.bool_)):
            raise ValueError('normalize must be boolean.')
        if not normalize and width.sum() > 1.+1.e-9:
            raise ValueError('An absolute/conditional kernel must not have probability sum above one.')
        center = round((len(width) - 1) / 2.)
        start = len(width) - 1 - center
        kernel = width / width.sum() if normalize else width
        return np.convolve(flux, kernel[::-1], mode='full')[start:start + len(flux)]

