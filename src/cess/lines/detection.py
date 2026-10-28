"""Emission detection and matching methods retained from C13.

Dependency monkeypatches belong to this service module (Spectrum1D,
fit_generic_continuum, find_lines_derivative, linear_sum_assignment). Class and
instance method monkeypatches on engine.emulator retain their usual semantics.
"""
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from astropy import units as u
from astropy.modeling.models import Chebyshev1D
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.table import QTable
from astropy.utils.exceptions import AstropyUserWarning
from specutils.fitting import (find_lines_derivative,
                              fit_generic_continuum as _fit_generic_continuum)
import specutils

# Spectrum1D remains the dependency hook used by legacy monkeypatches. In
# Specutils 2 it names Spectrum, avoiding the deprecated Spectrum1D wrapper;
# older Specutils supplies the original class through the same hook.
if hasattr(specutils, 'Spectrum'):
    Spectrum1D = specutils.Spectrum
else:
    from specutils.spectra import Spectrum1D

# Preserve the existing Specutils convention without reading configuration.
specutils.conf.do_continuum_function_check = False


def fit_generic_continuum(spectrum, median_window=3, model=None, fitter=None,
                          exclude_regions=None, weights=None):
    """Retain the Specutils 1.8 continuum convention across dependency versions.

    Specutils 2 changed its default fitter to TRFLSQFitter. Keep the adopted
    Chebyshev degree 3, median window 3 and LevMar fit explicit. This public
    service hook still accepts a single spectrum for legacy monkeypatches.

    Astropy advises using a linear solver for this polynomial. Suppress only
    that known advisory for our built-in model/fitter pair, whose numerical
    convention is retained deliberately. Custom fits and all other diagnostics
    remain visible.
    """
    with warnings.catch_warnings():
        if model is None and fitter is None:
            warnings.filterwarnings(
                'ignore',
                message=(r'^Model is linear in parameters; consider using '
                         r'linear fitting methods\.$'),
                category=AstropyUserWarning,
                module=r'^astropy\.modeling\.fitting$')
        return _fit_generic_continuum(
            spectrum, median_window=median_window,
            model=Chebyshev1D(3) if model is None else model,
            fitter=LevMarLSQFitter() if fitter is None else fitter,
            exclude_regions=exclude_regions, weights=weights)


class LineDetectionMixin:
    def el_detect(self,wave,flux,init_wave,end_wave,threshold):
        """ 
        Detect emission lines in the spectra using 'find_lines_derivative' function in specutils package.

        features:
        --------
            Using 'find_lines_derivative' function to derive emission lines with a normalized spectra ( spectrum/fitted_continuum )
            Create a NaN QTable to store the invalid mock spectra information.
            Choosing the initial and end index of wavelength grid to avoid the extremely noisy region at the end of each spectra.
            Wavelength unit should be in 'angstrom' while flux unit should be in 'uJy'.
            Different threshold can result in different line detections results, please have some test before choosing the reasonable threshold.

        parameters:
        ----------
            wave - wavelength grid of the galaxy - [array]
            flux - flux grid of the galaxy - [array]
            init_wave - the initial index in wavelength grid - [int]
            end_wave - the end index in wavelength grid - [int]
            # x_unit - the unit used in modeling specutils.spectrum for wavelength, recommanding 'angstrom'
            # y_unit - the unit used in modeling specutils.spectrum for flux, recommanding 'Jy'
            threshold - threshold of the 'find_lines_derivative' - [float]

        return:
        ------
            lines - QTable created by specutils, contains three columns of 'line_center', 'line_tpe', 'line_center_index' - [QTable]
            Numerically unusable windows return an empty table. Dependency,
            fitting and API errors propagate, rather than implying no lines.
        """

        # create a new empty QTable template
        lines_template = QTable(names=('line_center', 'line_type', 'line_center_index'),
                            dtype=(float, 'U10', int))

        # Only the requested detection window must be finite. WCE registration
        # deliberately marks unsupported boundary pixels NaN outside this window.
        wave = np.asarray(wave, dtype=float)
        flux = np.asarray(flux, dtype=float)
        if (wave.ndim != 1 or flux.shape != wave.shape or
            int(init_wave) != init_wave or int(end_wave) != end_wave or
            init_wave >= end_wave or init_wave < 0 or end_wave > len(wave)):
            # return a QTable with NaN values for all columns
            nan_line = lines_template.copy()
            nan_line['line_center'] = np.nan
            nan_line['line_type'] = 'NaN'
            nan_line['line_center_index'] = -1
            return nan_line

        init_wave, end_wave = int(init_wave), int(end_wave)
        x = wave[init_wave:end_wave]
        y = flux[init_wave:end_wave]
        # Whole-band windows include unsupported WCE/calibration edge bins.
        # Trim only those edges, never interpolate across an internal gap.
        # Returned peak indexes remain relative to the original requested
        # window so all writers and diagnostics retain their source alignment.
        edge_offset = 0
        if getattr(self, 'config', {}).get('spectral_window', {}).get('mode') == 'bandpass':
            available = np.flatnonzero(np.isfinite(y))
            if available.size < 4:
                return lines_template.copy()
            edge_offset, last = int(available[0]), int(available[-1]) + 1
            x, y = x[edge_offset:last], y[edge_offset:last]
        if (x.size < 4 or not np.all(np.isfinite(x)) or
                not np.all(np.isfinite(y)) or np.any(x <= 0.) or
                np.any(np.diff(x) <= 0.)):
            return lines_template.copy()

        if not np.isscalar(threshold) or not np.isfinite(threshold):
            raise ValueError('Line-detection threshold must be a finite scalar')

        # Keep the continuum fit, floor and normalized detection order intact.
        spectrum = Spectrum1D(flux=y * u.uJy, spectral_axis=x * u.angstrom)
        g1_fit = fit_generic_continuum(spectrum)
        mean = np.mean(spectrum.flux)
        continuum = spectrum.flux[np.argmin(np.abs(spectrum.flux - mean))]
        y_continuum_fitted = g1_fit(x * u.angstrom)
        if y_continuum_fitted.shape != y.shape:
            raise ValueError('Continuum model returned an unexpected shape')
        y_continuum_fitted[y_continuum_fitted < continuum] = continuum
        if (not np.all(np.isfinite(y_continuum_fitted)) or
                np.any(y_continuum_fitted == 0.)):
            return lines_template.copy()
        spec_normalized = spectrum / y_continuum_fitted
        lines = find_lines_derivative(spec_normalized, flux_threshold=threshold)
        required = ('line_center', 'line_type', 'line_center_index')
        if any(name not in lines.colnames for name in required):
            raise ValueError('Line detector returned an incompatible table')
        if edge_offset:
            lines['line_center_index'] = lines['line_center_index'] + edge_offset
        return lines

    @staticmethod
    def _emission_peaks(lines):
        """Extract finite emission peaks and their original QTable row indices."""
        required = ('line_type', 'line_center', 'line_center_index')
        if any(name not in lines.colnames for name in required):
            raise ValueError('Emission-line tables require ' + ', '.join(required))
        # specutils can return a numeric dtype for an empty line-type column;
        # older FITS/HDF5 readers may supply bytes for nonempty labels.
        rows = np.flatnonzero(np.asarray(lines['line_type']).astype(str) == 'emission')
        centers = lines['line_center'][rows]
        if hasattr(centers, 'to_value'):
            centers = centers.to_value(u.angstrom)
        else:
            centers = np.asarray(centers, dtype=float)
        indices = np.asarray(lines['line_center_index'][rows], dtype=float)
        good = (np.isfinite(centers) & np.isfinite(indices) &
                (indices >= 0) & (indices == np.floor(indices)))
        return np.asarray(centers)[good], indices[good].astype(int), rows[good]

    def el_success(self, noisy, intri, threshold, return_details=False):
        """Match noisy emission candidates one-to-one with independent truth.

        Source IDs remain zero-based input row indices; wavelengths are Angstrom
        and peak indices retain the caller's cropped-spectrum coordinates. Truth
        includes every intrinsic emission source, even when noisy has no peaks.
        Matching maximizes the number of pairs within threshold, then minimizes
        their total absolute wavelength separation (Hungarian assignment).

        The two legacy return tables have stable 1-D rank and structured dtype,
        including zero/one source. Only matched noisy peaks appear in the second
        table. The original noisy/intri QTables are never modified. Optional
        return_details=True adds a dict of candidates, missed, false_positive
        source tables and a per-pair association table. Neighboring blended peaks
        are still algorithmic candidates; this is not a physical line identity or
        integrated line-S/N classifier.
        """
        if len(noisy) != len(intri):
            raise ValueError('noisy and intri must contain the same source rows.')
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError('threshold must be finite and nonnegative, in Angstrom.')
        def table_dtype(number_name):
            return np.dtype([('id', np.int64), ('wave', object), ('idx', object),
                             (number_name, np.int64)])
        def make_table(records, number_name):
            result = np.empty(len(records), dtype=table_dtype(number_name))
            for k, record in enumerate(records):
                result[k] = record
            return result
        truth_records, matched_records = [], []
        candidate_records, missed_records, false_records, pair_records = [], [], [], []
        for source_id, (noisy_lines, intrinsic_lines) in enumerate(zip(noisy, intri)):
            nw, ni, nr = self._emission_peaks(noisy_lines)
            tw, ti, tr = self._emission_peaks(intrinsic_lines)
            if len(tw):
                truth_records.append((source_id, tw.copy(), ti.copy(), len(tw)))
            if len(nw):
                candidate_records.append((source_id, nw.copy(), ni.copy(), len(nw)))
            matched_n = np.empty(0, dtype=int)
            matched_t = np.empty(0, dtype=int)
            if len(nw) and len(tw):
                distance = np.abs(nw[:, None] - tw[None, :])
                # Dummy columns permit any noisy peak to remain unmatched. A
                # penalty larger than every possible total distance makes the
                # primary objective maximum cardinality, then minimum distance.
                penalty = (len(nw) + len(tw) + 1) * (float(threshold) + 1)
                cost = np.full((len(nw), len(tw) + len(nw)), penalty)
                cost[:, :len(tw)] = np.where(distance <= threshold, distance, penalty * 2)
                row, column = linear_sum_assignment(cost)
                keep = column < len(tw)
                matched_n, matched_t = row[keep], column[keep]
                if len(matched_n):
                    matched_records.append((source_id, nw[matched_n].copy(),
                                            ni[matched_n].copy(), len(matched_n)))
                    for j, k in zip(matched_n, matched_t):
                        pair_records.append((source_id, nr[j], tr[k], nw[j], tw[k],
                                             nw[j] - tw[k], ni[j], ti[k]))
            unmatched_n = np.setdiff1d(np.arange(len(nw)), matched_n)
            unmatched_t = np.setdiff1d(np.arange(len(tw)), matched_t)
            if len(unmatched_n):
                false_records.append((source_id, nw[unmatched_n].copy(),
                                      ni[unmatched_n].copy(), len(unmatched_n)))
            if len(unmatched_t):
                missed_records.append((source_id, tw[unmatched_t].copy(),
                                       ti[unmatched_t].copy(), len(unmatched_t)))
        truth = make_table(truth_records, 'intri_el_number')
        matched = make_table(matched_records, 'detect_el_number')
        pair_dtype = np.dtype([
            ('id', np.int64), ('noisy_peak_id', np.int64), ('intrinsic_peak_id', np.int64),
            ('noisy_wave', float), ('intrinsic_wave', float), ('delta_wave', float),
            ('noisy_idx', np.int64), ('intrinsic_idx', np.int64)])
        pairs = np.array(pair_records, dtype=pair_dtype)
        details = dict(candidates=make_table(candidate_records, 'candidate_el_number'),
                       missed=make_table(missed_records, 'missed_el_number'),
                       false_positive=make_table(false_records, 'false_el_number'), pairs=pairs)
        self.last_el_match_details = details
        if return_details:
            return truth, matched, details
        return truth, matched

    def get_el_snr(self, snr, flag, detect_el, init_wave_idx, elwidth):
        """Mean per-pixel S/N around each matched peak, indexed by source row ID.

        This is a local average of per-pixel S/N, not continuum-subtracted,
        integrated emission-line S/N. Windows are clipped at spectrum boundaries;
        finite samples are averaged, or NaN is returned when none are available.
        A 1-D object array is returned even for zero/one source or equal line
        counts. flag is retained for compatibility and shape validation only;
        ordering is defined by detect_el['id'], never by flag==2 row order.
        """
        snr = np.asarray(snr, dtype=float)
        flag = np.asarray(flag)
        if snr.ndim != 2 or flag.shape != (snr.shape[0],):
            raise ValueError('Expected snr (n_source,n_wave) and flag (n_source,).')
        if int(elwidth) != elwidth or elwidth < 1 or int(init_wave_idx) != init_wave_idx:
            raise ValueError('elwidth must be a positive integer and init_wave_idx an integer.')
        result = np.empty(len(detect_el), dtype=object)
        for k, detection in enumerate(detect_el):
            source_id = int(detection['id'])
            if source_id < 0 or source_id >= snr.shape[0]:
                raise ValueError('Matched source id is outside the input S/N rows.')
            values = []
            for index in detection['idx']:
                center = int(init_wave_idx) + int(index)
                start = center - int(elwidth) // 2
                stop = start + int(elwidth)
                start, stop = max(0, start), min(snr.shape[1], stop)
                window = snr[source_id, start:stop] if stop > start else np.empty(0)
                window = window[np.isfinite(window)]
                values.append(float(window.mean()) if window.size else np.nan)
            result[k] = np.asarray(values, dtype=float)
        return result

    # def dlambda(self,grism,Res):
    #     """
    #     degrade the grism wavelength grid to the given resolution

    #     parameters
    #     ----------
    #         grism: CSST grism using pysynphot to read
    #         Res: resolution of CSST grism

    #     retrun
    #     ------
    #         lrlam: wavelength grid with CSST grism resolution
    #         dlam: the wavelength interval (delta lambda) for each wavelength point
    #     """

    #     lam = []; dlam = []
    #     lam.append(grism.wave.min())
    #     dlam.append(lam[0]/(Res-0.5))
    #     Nlam = 1

    #     for i in range(1,1000):

    #         lam.append(lam[i-1] + dlam[i-1])
    #         dlam.append(lam[i]/(Res-0.5))
    #         Nlam = Nlam + 1
    #         if lam[i]>grism.wave.max():
    #             break
    #     return np.array(lam) #, np.array(dlam)

    # def convert_dpix_to_dlam(self,grism,Res,wave,flux,fluxnoise,eflux):
        
    #     # print('\nNow converting dx to dlam of {0}:'.format(grism.name[-11:-9]))

    #     lam = self.dlambda(grism,Res)
    #     dlamall = np.tile(lam,(len(wave),1))
    #     newflux = np.ones([len(wave),len(lam)])
    #     newfluxnoise = np.ones([len(wave),len(lam)])
    #     neweflux = np.ones([len(wave),len(lam)])


    #     for i in range(len(wave)):
    #         newflux[i] = np.interp(dlamall[i],wave[i],flux[i])
    #         newfluxnoise[i] = np.interp(dlamall[i],wave[i],fluxnoise[i])
    #         neweflux[i] = np.interp(dlamall[i],wave[i],eflux[i])

    #     # print('Converting finished!')

    #     return lam, newflux, newfluxnoise, neweflux
