"""Source-aligned HDF5 writers for the retained one-dimensional schema.

Scientific services are imported only when invoked. This module does not import
pipeline.support or the engine facade; shared schema constants are leaf data.
"""
from pathlib import Path
import json
import os
import h5py
import numpy as np
from cess.instrument.spectral import target_fwhm, output_wavelengths, window_indices
from cess.contracts import (SCHEMA_VERSION, REPAIR_REVISION, COVARIANCE_MODEL,
    MGE_MORPHOLOGY_FIELDS, MGE_MORPHOLOGY_ATTRIBUTES, LINE_MEASUREMENT_DTYPE)


def write_spectral_extraction(handle, rows, parameters, profile_provenance):
    """Record the achromatic aperture and equal-weight spatial noise area."""
    if len(rows) != len(handle['ID']):
        raise ValueError('Spectral extraction metadata is not aligned with root ID')
    group = handle.create_group('spectral_extraction')
    group.attrs['model_json'] = json.dumps(parameters['spectral_extraction'], sort_keys=True)
    group.attrs['source_flux_reference'] = (
        'aperture-referenced nominal-center-response flux; source mean and source '
        'Poisson variance include source_fraction; no division by that fraction')
    group.attrs['noise_area_definition'] = (
        'integer count of equally weighted spatial pixels per spectral column per exposure; '
        'not a weighted-extraction effective area')
    group.attrs['profile_height_sum_definition'] = (
        'observed sum of selected legacy profile weights only; normalization unknown; '
        'not an all-source aperture fraction')
    if parameters['spectral_extraction']['mode'] == 'mge_strip':
        group.attrs['profile_height_sum_definition'] = (
            'sum of absolute all-x probabilities in the declared y-strip bins; '
            'equals modeled source_fraction within numerical tolerance; not an instrument aperture calibration')
    group.attrs['legacy_full_limitation'] = (
        'legacy_full assumes all source photons with height-length noise area; '
        'kept for reproducibility, not a calibrated aperture model')
    group.attrs['explicit_model_limitation'] = (
        'explicit source_fraction and spatial_pixels are supplied achromatic assumptions; '
        'they do not certify aperture/PSF geometry or line-versus-continuum morphology')
    for name in ('source_fraction', 'spatial_pixels', 'profile_height_sum'):
        values = np.array([np.nan if row is None else row[name] for row in rows], dtype=float)
        dataset = group.create_dataset(name, data=values)
        dataset.attrs['axis_order'] = 'source'
        dataset.attrs['unit'] = 'pixel' if name == 'spatial_pixels' else '1'
    valid = group.create_dataset('valid', data=np.array([row is not None for row in rows], dtype=bool))
    valid.attrs['axis_order'] = 'source'
    valid.attrs['unit'] = '1'
    handle.attrs['profile_library_provenance_json'] = json.dumps(profile_provenance, sort_keys=True, allow_nan=False)
    return group

def write_mge_morphology(handle, rows, parameters):
    """Persist source-aligned joint-strip diagnostics; invalid rows stay NaN."""
    if len(rows) != len(handle['ID']):
        raise ValueError('MGE diagnostics are not aligned with root ID')
    group = handle.create_group('mge_morphology')
    group.attrs['model_json'] = json.dumps(parameters['spectral_extraction'], sort_keys=True)
    for name, value in MGE_MORPHOLOGY_ATTRIBUTES.items():
        group.attrs[name] = value
    for name in MGE_MORPHOLOGY_FIELDS:
        boolean = name in ('valid', 'kernel_applied')
        if boolean:
            values = np.asarray([row is not None if name == 'valid' else
                                 False if row is None else row[name] for row in rows], dtype=bool)
        elif name in ('selected_parameters', 'selected_indices'):
            values = np.asarray([np.full(4, np.nan) if row is None else row[name]
                                 for row in rows], dtype=float).reshape(len(rows), 4)
        else:
            values = np.asarray([np.nan if row is None else row[name] for row in rows], dtype=float)
        dataset = group.create_dataset(name, data=values)
        dataset.attrs['axis_order'] = 'source,parameter_axis' if values.ndim == 2 else 'source'
        dataset.attrs['unit'] = 'mixed; see selected_parameter_units' if name == 'selected_parameters' else '1'
    return group

def write_response_products(group, response_flux_ujy, source_rate_e_s_cm2,
                            center_response, sky_reference, sky_rates,
                            parameters, suffix='', paired=False):
    """Persist C2 incident-response products and nominally calibrated models.

    The rate is already response weighted, per collecting area, before spatial
    redistribution or WCE. The derived model uses exactly the noisy spectrum's
    nominal center-response calibration, which is not an inverse of the LSF.
    """
    from cess.constants import h
    wave = np.asarray(group['wave'][:], dtype=float)
    from cess.spectra.integration import bin_edges
    width = np.diff(bin_edges(wave))
    response = np.asarray(center_response, dtype=float)
    folded = group.create_dataset('response_flux_ujy' + suffix, data=response_flux_ujy)
    folded.attrs['unit'] = 'uJy'
    folded.attrs['axis_order'] = 'source,pixel'
    folded.attrs['definition'] = (
        'response-weighted photon-equivalent bin fnu before morphology/WCE; '
        'lambda_center/pixel_width * integral fnu(lambda)*T(lambda)/lambda '
        'P(bin|lambda) dlambda; not response corrected')
    rate = group.create_dataset('source_electron_rate_pre_morph' + suffix,
                                data=source_rate_e_s_cm2)
    rate.attrs['unit'] = 'electron s^-1 cm^-2'
    rate.attrs['axis_order'] = 'source,pixel'
    rate.attrs['definition'] = (
        'response-weighted expected source electron rate per collecting area, '
        'before morphology/WCE; multiply collecting area and total exposure once')
    center = group.create_dataset('response_at_pixel_center' + suffix, data=response)
    center.attrs['unit'] = '1'
    center.attrs['axis_order'] = 'pixel'
    center.attrs['definition'] = 'nominal response used only for fnu calibration, not source-count integration'
    sky = group.create_dataset('sky_rate_e_s_pixel' + suffix, data=sky_rates)
    sky.attrs['unit'] = 'electron s^-1 pixel^-1'
    sky.attrs['axis_order'] = 'source'
    sky.attrs['reference_rate_e_s_pixel'] = float(sky_reference)
    sky.attrs['reference_definition'] = (
        'native sky/response-knot integration including background_scale, before legacy position scaling')
    sky.attrs['position_scaling_definition'] = (
        '1 for fixed_template; 10**(0.4*(background_v_mag-match_bkg_mag(RA,Dec))) for explicit legacy_radec')
    sky.attrs['sky_duplicate_knots'] = 'ordered left/right limits; no averaging or source mutation'
    area_time = np.pi * parameters['radi'] ** 2 * parameters['expt'] * parameters['expnum']
    denominator = area_time * width * response
    for extra in (('', '_wce') if paired else ('',)):
        counts_name = 'flux_elec' + extra + suffix
        counts = np.asarray(group[counts_name][:], dtype=float)
        model = np.full_like(counts, np.nan)
        np.divide(counts * h * wave / 1.e-29, denominator, out=model,
                  where=np.isfinite(counts) & np.isfinite(denominator) & (denominator > 0))
        dataset = group.create_dataset('flux_ujy_model' + extra + suffix, data=model)
        dataset.attrs['unit'] = 'uJy'
        dataset.attrs['axis_order'] = 'source,pixel'
        dataset.attrs['definition'] = (
            'noiseless expected electron bins with morphology/WCE as indicated, '
            'calibrated by nominal output-center response exactly as flux_ujy_with_noise; '
            'not an unbiased inverse-response reconstruction of incident fnu')
        for name, values, definition in (
                ('detector_model_mask', np.isfinite(counts),
                 'finite expected detector electron bins, including response spill into zero center-response pixels'),
                ('calibration_support_mask', np.isfinite(group['ferr' + extra + suffix][:]),
                 'finite calibrated error: detector support and positive nominal output-center response')):
            mask = group.create_dataset(name + extra + suffix, data=values)
            mask.attrs['axis_order'] = 'source,pixel'
            mask.attrs['definition'] = definition
    if suffix == '_0th':
        for name in group:
            if name.endswith('_0th'):
                group[name].attrs['product_role'] = (
                    'legacy diagnostic proxy: zero-order efficiency on first-order grid/LSF; '
                    'not a physical zero-order image or its aperture SNR')
                group[name].attrs['n_exposures'] = int(parameters['expnum'])
                group[name].attrs['total_exposure_s'] = float(parameters['expt'] * parameters['expnum'])

_PHOTOMETRY_NUMERIC = (
    'model_flux_ujy', 'model_mag', 'observed_flux_ujy', 'flux_error_ujy',
    'statistical_flux_error_ujy', 'systematic_flux_error_ujy', 'total_flux_error_ujy',
    'model_mag_error', 'observed_mag', 'observed_mag_error', 'snr', 'expected_snr',
    'source_counts_e', 'observed_counts_e', 'sky_counts_e', 'dark_counts_e',
    'read_variance_e2', 'variance_e2', 'coverage_fraction', 'detection_snr',
    'total_exposure_s', 'aperture_fraction', 'n_pixels', 'systematic_mag',
    'response_efficiency')

def empty_photometry_result(reason='invalid_source', flag=32):
    result = {name: np.nan for name in _PHOTOMETRY_NUMERIC}
    result.update(valid=False, detected=False, coverage_complete=False,
                  flag=int(flag), reason=reason)
    return result

def write_photometry_results(parent, name, bands, parameters):
    """Save model/observed flux, errors and availability for each input ID."""
    group = parent.create_group(name)
    group.attrs['aperture_radius_re'] = parameters.get('photometry_aperture_radius_re', 15.)
    group.attrs['aperture_model'] = 'legacy geometric pi*(radius_re*Re_pixel)^2; not independently validated'
    group.attrs['flux_definition'] = 'total-source flux calibrated by the aperture light fraction'
    group.attrs['systematic_error_model'] = 'reported calibration uncertainty; not independently drawn per source'
    group.attrs['flag_meanings'] = (
        '1 incomplete_source_coverage;2 nonfinite_model;4 negative_model;'
        '8 below_detection;16 nonpositive_observed;32 invalid_source;64 incomplete_background_coverage')
    template = empty_photometry_result()
    for band, rows in bands.items():
        target = group.create_group(band)
        for key in template:
            values = [row[key] for row in rows]
            if key == 'reason':
                dataset = target.create_dataset(key, data=np.asarray(values, dtype=object),
                    dtype=h5py.string_dtype(encoding='utf-8'))
            else:
                dtype = bool if key in ('valid', 'detected', 'coverage_complete') else int if key == 'flag' else float
                dataset = target.create_dataset(key, data=np.asarray(values, dtype=dtype))
            if key.endswith('_ujy'):
                dataset.attrs['unit'] = 'uJy'
            elif 'mag' in key:
                dataset.attrs['unit'] = 'AB mag' if key in ('model_mag', 'observed_mag') else 'mag'
            elif key.endswith('_e2'):
                dataset.attrs['unit'] = 'electron^2'
            elif key.endswith('_e'):
                dataset.attrs['unit'] = 'electron'
            elif key == 'total_exposure_s':
                dataset.attrs['unit'] = 's'
            elif key == 'n_pixels':
                dataset.attrs['unit'] = 'pixel'
            else:
                dataset.attrs['unit'] = '1'
            dataset.attrs['axis_order'] = 'source'
    return group

def pad_spectral_rows(rows, n_sources, n_wave, name='spectrum'):
    """Make an (N, W) float array, including empty or entirely invalid input.

    A scalar NaN marks an invalid source. Short one-dimensional rows are
    padded with NaN; surplus rows/pixels and finite scalar rows are errors.
    """
    if len(rows) != n_sources:
        raise ValueError('{} has {} rows for {} IDs'.format(name, len(rows), n_sources))
    padded = np.full((n_sources, n_wave), np.nan, dtype=float)
    for index, row in enumerate(rows):
        value = np.asarray(row, dtype=float)
        if value.ndim == 0 and np.isnan(value):
            continue
        if value.ndim != 1 or value.size > n_wave:
            raise ValueError('{} row {} does not fit the wavelength grid'.format(name, index))
        padded[index, :value.size] = value
    return padded

def write_vlen_rows(dataset, rows):
    """Write 0/1/many ragged rows without h5py's singleton broadcasting traps."""
    if len(rows) != len(dataset):
        raise ValueError('Variable-length dataset row count mismatch')
    for index, row in enumerate(rows):
        value = np.asarray(row)
        if value.ndim != 1:
            raise ValueError('Each variable-length value must be one-dimensional')
        dataset[index] = value

def write_el_details(parent, details, suffix=''):
    """Save independent EL candidates, missed truth, false positives and pairs."""
    diagnostics = parent.create_group('el_diagnostics' + suffix)
    diagnostics.attrs['id_definition'] = 'source row index into root ID'
    diagnostics.attrs['wavelength_unit'] = 'Angstrom'
    diagnostics.attrs['idx_definition'] = 'relative to the configured EL wavelength slice'
    for name in ('candidates', 'missed', 'false_positive'):
        table = details[name]
        group = diagnostics.create_group(name)
        for column in table.dtype.names:
            values = table[column]
            if column in ('wave', 'idx'):
                dtype = np.dtype('float64' if column == 'wave' else 'int64')
                dataset = group.create_dataset(column, (len(table),),
                                               dtype=h5py.vlen_dtype(dtype))
                write_vlen_rows(dataset, values)
            else:
                group.create_dataset(column, data=values)
    diagnostics.create_dataset('pairs', data=details['pairs'])

def write_spectral_covariance(group, rows, suffix=''):
    """Persist Cov(fnu[j],fnu[j+1]) in uJy squared, aligned with root IDs.

    The diagonal remains ferr**2. Longer lags vanish only under the stated
    independent-bin, single constant linear-shift model. Different paired
    variants share noise; their cross covariance is not represented here.
    """
    values = np.asarray(rows, dtype=float)
    n_wave = len(group['wave'])
    expected = (group['ferr' + suffix].shape[0], n_wave - 1)
    if values.shape != expected:
        raise ValueError('covariance_lag1 must have shape {} for {}'.format(expected, group.name))
    dataset = group.create_dataset('covariance_lag1' + suffix, data=values)
    dataset.attrs['unit'] = 'uJy^2'
    dataset.attrs['axis_order'] = 'source,pixel_pair'
    dataset.attrs['covariance_model'] = COVARIANCE_MODEL
    dataset.attrs['definition'] = 'element j = Cov(calibrated fnu[j], calibrated fnu[j+1]); diagonal = ferr**2'
    dataset.attrs['longer_lags'] = 'zero for the declared independent-bin/constant-linear-shift model only'
    dataset.attrs['cross_variant_covariance'] = 'not stored; paired variants share one realization'
    return dataset

def write_line_measurements(parent, emulator, noisy_lines, intrinsic_lines,
                            parameters, source_group=None, suffix=''):
    """Save window-limited continuum-subtracted flux proxies for every peak.

    Two flat, fixed-dtype tables retain original detector QTable peak IDs:
    ``candidates`` includes all noisy peaks and ``truth`` includes all model
    peaks, including missed ones. Rows link to el_diagnostics pairs through
    (source_row, peak_id), never through rounded wavelength or row ordinal.
    Legacy parents without a certified covariance product retain peak rows
    flagged unavailable, rather than inventing integrated S/N from pixel S/N.
    """
    from cess.lines.measurement import measure_line_window
    source = parent if source_group is None else source_group
    if len(noisy_lines) != len(intrinsic_lines):
        raise ValueError('Line lists must have the same source count')
    count = len(noisy_lines)
    band = parent.name.strip('/').split('/')[-1]
    design = parameters['line_measurement']
    width = design['line_width_pixels'][band]
    side = design['sideband_width_pixels']
    gap = design['sideband_gap_pixels']
    wave = np.asarray(source['wave'][:], dtype=float)
    initial, _ = window_indices(parameters, band, wave)
    required = ('flux_ujy_with_noise' + suffix, 'ferr' + suffix)
    missing_spectra = any(name not in source for name in required)
    covariance_name = 'covariance_lag1' + suffix
    covariance_known = (covariance_name in source and
                        source[covariance_name].attrs.get('covariance_model') == COVARIANCE_MODEL)
    data = error = covariance = model = masks = None
    if not missing_spectra:
        data, error = [np.asarray(source[name][:], dtype=float) for name in required]
        if data.shape != (count, wave.size) or error.shape != data.shape:
            raise ValueError('Line flux/error arrays are not aligned with the peak source rows')
        if covariance_known:
            covariance = np.asarray(source[covariance_name][:], dtype=float)
            if covariance.shape != (count, wave.size - 1):
                raise ValueError('Line covariance is not aligned with flux/error')
        model_name = 'flux_ujy_model' + suffix
        if model_name in source:
            model = np.asarray(source[model_name][:], dtype=float)
            if model.shape != data.shape:
                raise ValueError('Line noiseless model is not aligned with flux/error')
        masks = np.ones(data.shape, dtype=bool)
        if 'spec_mask' in source:
            mask_values = np.asarray(source['spec_mask'][:])
            if mask_values.shape != data.shape:
                raise ValueError('Line analysis mask is not aligned with flux/error')
            masks &= mask_values == 1
        support_name = 'calibration_support_mask' + suffix
        if support_name in source:
            support = np.asarray(source[support_name][:])
            if support.shape != data.shape:
                raise ValueError('Line calibration mask is not aligned with flux/error')
            masks &= support == 1
    target = parent.create_group('line_measurements' + suffix)
    target.attrs['algorithm'] = 'cess-affine-sideband-window-v1'
    target.attrs['variant'] = ('registered_wce' if suffix == '_wce' else
                               'unregistered' if 'flux_ujy_with_noise_wce' in source else 'standard')
    target.attrs['source_id_definition'] = 'source_row is local row into root ID; peak_id is original detector QTable row'
    target.attrs['window_config_json'] = json.dumps(design, sort_keys=True)
    target.attrs['flux_definition'] = (
        'sum residual_fnu_uJy*1e-29*c_Angstrom_s*pixel_width/nominal_lambda_center**2; '
        'affine f_lambda continuum from two sidebands; nominal center-response calibrated '
        'window energy-flux proxy, not total incident atomic-line flux')
    target.attrs['covariance_model'] = COVARIANCE_MODEL if covariance_known else 'unknown; metrics unavailable'
    target.attrs['uncertainty_definition'] = (
        'fixed-window linear estimator w^T C w; includes sideband-fit uncertainty and '
        'lag-one cross terms; conditional on supplied noise model; no selection/centering uncertainty')
    target.attrs['aperture_reference'] = (
        'same aperture/nominal-center flux reference as the parent calibrated spectrum; '
        'see parent spectral_extraction metadata when available; no new aperture correction')
    target.attrs['truth_definition'] = (
        'all algorithmic emission peaks in incident unweighted reference spectrum; '
        'not a labelled atomic-line or complete intrinsic-emission catalogue')
    target.attrs['selection_definition'] = (
        'candidate centers selected from noisy data; forced truth centers selected from reference; '
        'no integrated-SNR cut applied; candidate selection bias is not corrected')
    target.attrs['valid_definition'] = (
        'numerically valid full-window estimator; separately require not contaminated for '
        'isolated-peak use; does not certify continuum shape, line wings, aperture or LSF')
    target.attrs['contaminated_definition'] = (
        'another peak center from the same candidate/truth list lies in the line or sidebands; '
        'advisory only, unrecognized blends and line wings can remain')
    target.attrs['flag_meanings'] = (
        '1 window_out_of_bounds;2 masked_pixel;4 nonfinite_data;'
        '8 nonpositive_or_nonfinite_error;16 nonfinite_covariance;'
        '32 covariance_not_positive_definite;64 continuum_fit_failed;'
        '128 invalid_integral_variance;256 missing_spectra;512 unknown_covariance;'
        '1024 other_peak_in_window_advisory')
    for kind, lists in (('candidates', noisy_lines), ('truth', intrinsic_lines)):
        rows = []
        for source_row, peaks in enumerate(lists):
            centers, indices, peak_ids = emulator._emission_peaks(peaks)
            full_indices = indices + initial
            for peak_wave, center, peak_id in zip(centers, full_indices, peak_ids):
                start = int(center) - width // 2
                stop = start + width
                bounds = dict(line_start=start, line_stop=stop,
                    left_start=start-gap-side, left_stop=start-gap,
                    right_start=stop+gap, right_stop=stop+gap+side, nline=width, nside=2*side)
                if missing_spectra or not covariance_known:
                    result = dict(bounds, valid=False,
                        flag=256 if missing_spectra else 512,
                        reason='missing_spectra' if missing_spectra else 'unknown_covariance')
                else:
                    result = measure_line_window(wave, data[source_row], error[source_row],
                        int(center), width, side, sideband_gap_pixels=gap,
                        covariance_lag1=covariance[source_row], mask=masks[source_row],
                        model_ujy=None if model is None else model[source_row])
                other = full_indices[peak_ids != peak_id]
                contaminated = bool(np.any(
                    ((other >= bounds['line_start']) & (other < bounds['line_stop'])) |
                    ((other >= bounds['left_start']) & (other < bounds['left_stop'])) |
                    ((other >= bounds['right_start']) & (other < bounds['right_stop']))))
                flag = int(result['flag']) | (1024 if contaminated else 0)
                reason = result['reason']
                if contaminated:
                    reason = ('other_peak_in_window' if reason == 'ok' else
                              reason + ';other_peak_in_window')
                def metric(name):
                    value = result.get(name)
                    return np.nan if value is None else value
                rows.append((source_row, int(peak_id), int(center), float(peak_wave)) +
                    tuple(int(result.get(name, bounds[name])) for name in (
                        'line_start', 'line_stop', 'left_start', 'left_stop',
                        'right_start', 'right_stop', 'nline', 'nside')) +
                    tuple(metric(name) for name in ('flux', 'error', 'snr', 'model_flux', 'model_snr')) +
                    (bool(result['valid']), flag, contaminated, reason.encode('ascii')))
        dataset = target.create_dataset(kind, data=np.array(rows, dtype=LINE_MEASUREMENT_DTYPE))
        dataset.attrs['axis_order'] = 'peak'
        dataset.attrs['flux_unit'] = 'erg/s/cm^2'
        dataset.attrs['wavelength_unit'] = 'Angstrom'
        dataset.attrs['snr_unit'] = '1'
        dataset.attrs['window_index_definition'] = 'zero-based full-spectrum pixel ranges [start,stop)'
    return target

def _spectral_provenance(parameters):
    """Describe the photon-bin contract without asserting a measured input LSF."""
    template = parameters.get('template_lsf', {'mode': 'unknown'})
    central_model = parameters.get('instrument_lsf', {}).get('mode') == 'central_resolving_power'
    target_widths = {}
    for band in ('GU', 'GV', 'GI'):
        keys = ((band.lower() + '_central_wl', band.lower() + '_Res') if central_model
                else (band.lower() + '_dlambda',))
        # Metadata-only consumers can legitimately provide a partial config.
        # Omission is unknown; explicitly supplied invalid widths still raise.
        target_widths[band] = target_fwhm(parameters, band) if all(key in parameters for key in keys) else None
    comparison = {'scope': 'declared pre-pixel Gaussian widths; excludes morphology and pixel response',
                  'template_mode': template['mode']}
    if template['mode'] == 'resolving_power':
        comparison['bands'] = {}
        for band in ('GU', 'GV', 'GI'):
            prefix = band.lower()
            if (target_widths[band] is None or prefix + '_min_wl' not in parameters or
                    prefix + '_max_wl' not in parameters):
                comparison['bands'][band] = {'status': 'unknown; complete band metadata not recorded'}
                continue
            low, high = parameters[prefix + '_min_wl'], parameters[prefix + '_max_wl']
            crossing = target_widths[band] * template['value']
            comparison['bands'][band] = {
                'template_broader_above_observed_Angstrom': crossing,
                'overbroad_band_interval_Angstrom': [max(low, crossing), high] if crossing < high else None,
                'preserve_definition': 'no additional instrumental broadening where input is broader; narrower target not attained'}
    return {
        'spectral_convolution_algorithm': 'cess-photon-bin-gauss-v1',
        'spectral_quantity': (
            'photon-equivalent bin fnu; Q=integral fnu(lambda)/lambda '
            'P(bin|lambda) dlambda; fnu_eff=lambda_center*Q/pixel_width'),
        'spectral_conserved_quantity': (
            'incident photon-rate proxy Q; response products integrate '
            'Q_T=integral fnu(lambda)*T(lambda)/lambda P(bin|lambda) dlambda; '
            'finite output windows lose out-of-band photons; not energy-flux conservation'),
        'spectral_response_algorithm': 'cess-native-response-photon-bin-v1',
        'spectral_response_quantity': (
            'response-weighted photon-equivalent bin fnu; '
            'fnu_T=lambda_center*Q_T/pixel_width; native incident-wavelength response '
            'inside the LSF integral, before morphology/WCE'),
        'spectral_flux_calibration': (
            'flux_ujy is incident/unweighted bin fnu; flux_ujy_model, '
            'flux_ujy_with_noise and ferr use nominal output-center response '
            'calibration for the selected extraction aperture, without division by '
            'source_fraction; this is not an inverse-response reconstruction of total incident fnu'),
        'spectral_extraction_json': json.dumps(
            parameters.get('spectral_extraction', {'mode': 'legacy_full'}), sort_keys=True),
        'spectral_aperture_definition': (
            'achromatic source_fraction multiplies expected source electrons before noise; '
            'integer spatial_pixels sets independent equal-weight sky/dark/read noise area; '
            'legacy_full assumes fraction=1 with height-length area, not a calibrated aperture; '
            'photometry and actual zero_order use their separate aperture model'),
        'spectral_source_model_validity': (
            'intrinsic fnu must be finite, nonnegative and <=1e100 in every native sample; '
            'redshift must be finite in [1e-8,10]; invalid source rows remain NaN '
            'and do not abort other catalogue rows; no negative-model clipping'),
        'spectral_detector_support': (
            'flux_elec and detector_model_mask preserve finite expected detector counts '
            'even where nominal center response is zero; calibrated flux/error/SNR '
            'require positive center response and calibration_support_mask'),
        'spectral_zero_order_legacy_definition': (
            'datasets ending _0th are zero-order-efficiency diagnostic proxies '
            'on first-order grid/LSF with expnum exposures; not physical zero-order images; '
            'zero_order photometry separately integrates the full response with zero_order_exposures'),
        'spectral_sky_integration': (
            'native sky and throughput knots integrated across the positive response '
            'support; reference includes background_scale; fixed_template ignores RA/Dec; '
            'explicit legacy_radec applies its deprecated position scale'),
        'sky_duplicate_knots': 'ordered left/right limits; no averaging or source mutation',
        'spectral_convolution_validation': (
            'synthetic photon-conservation/phase/edge validation only; '
            'real template LSF and survey sensitivity are not independently validated'),
        'spectral_bin_geometry': (
            'wave contains pixel centers; interior edges are adjacent-center '
            'midpoints; outer edges extrapolate half of the nearest center interval'),
        'template_lsf_json': json.dumps(template, sort_keys=True),
        'template_lsf_nominal_evidence_json': json.dumps(
            parameters.get('template_lsf_nominal_evidence', {'status': 'not recorded'}), sort_keys=True),
        'template_lsf_assumption_json': json.dumps(
            parameters.get('template_lsf_assumption', {'status': 'not separately recorded'}), sort_keys=True),
        'template_lsf_broader_policy': parameters.get('template_lsf_broader_policy', 'error'),
        'spectral_target_lsf_json': json.dumps({
            'family': 'Gaussian', 'stage': 'pre-pixel', 'frame': 'observed',
            'model': parameters.get('instrument_lsf', {'mode': 'fixed_fwhm'}),
            'central_wavelength_Angstrom': {band: parameters.get(band.lower() + '_central_wl')
                                            for band in ('GU', 'GV', 'GI')},
            'central_resolving_power': {band: parameters.get(band.lower() + '_Res')
                                       for band in ('GU', 'GV', 'GI')},
            'fwhm_Angstrom': target_widths}, sort_keys=True),
        'spectral_sampling_json': json.dumps(
            parameters.get('spectral_sampling', {'mode': 'legacy'}), sort_keys=True),
        'spectral_instrument_reference_json': json.dumps(
            parameters.get('instrument_reference', {'status': 'not recorded'}), sort_keys=True),
        'spectral_template_target_comparison_json': json.dumps(comparison, sort_keys=True),
        'spectral_final_lsf_definition': (
            'unknown physical final LSF when template mode is unknown; otherwise '
            'conditional on declared template LSF and broader policy; pixel '
            'integration broadens the pre-pixel response; never inferred from sampling'),
        'spectral_throughput_approximation': (
            'native piecewise-linear throughput evaluated at incident wavelength '
            'inside spectral redistribution before morphology; nominal output-center '
            'throughput retained only for flux calibration; morphology redistributes '
            'response-weighted expected electron bins'),
    }

def _record_spectral_geometry(handle):
    """Persist the exact center-derived integration geometry for spectral groups."""
    for band in ('GU', 'GV', 'GI'):
        if band not in handle or 'wave' not in handle[band]:
            continue
        from cess.spectra.integration import bin_edges
        group = handle[band]
        centers = np.asarray(group['wave'][:], dtype=float)
        edges = bin_edges(centers)
        widths = np.diff(edges)
        for name, values in (('wave_edges', edges), ('pixel_width', widths)):
            if name in group:
                if not np.array_equal(group[name][:], values):
                    raise ValueError('{} geometry disagrees with wave centers'.format(name))
                dataset = group[name]
            else:
                dataset = group.create_dataset(name, data=values)
            dataset.attrs['unit'] = 'Angstrom'
            dataset.attrs['axis_order'] = 'pixel_edge' if name == 'wave_edges' else 'pixel'
        group['wave'].attrs['definition'] = 'nominal output pixel centers'
        group.attrs['spectral_quantity'] = handle.attrs['spectral_quantity']

def record_runner_provenance(handle, parameters, switches, source=None, seed=None,
                             input_sha256=None, input_path=None,
                             source_start=0, source_stop=None):
    """Record the source-major schema and reproducible source-stream identity."""
    from cess.pipeline.random import STREAM_VERSION, file_identity
    import platform
    import scipy
    import astropy
    handle.attrs['cess_repair_revision'] = REPAIR_REVISION
    handle.attrs['schema_version'] = SCHEMA_VERSION
    spectral = _spectral_provenance(parameters)
    if source is None:
        for name, value in spectral.items():
            handle.attrs[name] = value
    else:
        for name in spectral:
            handle.attrs[name] = source.attrs.get(
                name, 'unknown; input product did not record this')
        # Preserve any additional spectral attributes from future/older parents;
        # postprocessing does not run this generation's convolution operator.
        for name in source.attrs:
            if name.startswith(('spectral_', 'template_lsf')):
                handle.attrs[name] = source.attrs[name]
    handle.attrs['schema_layout'] = 'all parameter tables use source,field'
    handle.attrs['config_json'] = json.dumps(parameters, sort_keys=True)
    windows = {}
    wavelength_window = parameters.get('spectral_window', {}).get('mode') == 'bandpass'
    for band in ('GU', 'GV', 'GI'):
        if band not in handle:
            continue
        if 'wave' in handle[band]:
            wave_source = handle[band]
        elif source is not None and band in source and 'wave' in source[band]:
            # EL-only outputs contain tables, not duplicate parent spectra.
            # Their analysis interval must still refer to the actual parent
            # grid used for detection, not the new simulation's nominal grid.
            wave_source = source[band]
        else:
            continue
        suffixes = ('min_wl', 'max_wl') if wavelength_window else ('init_idx', 'end_idx')
        if all(band.lower() + '_' + suffix in parameters for suffix in suffixes):
            windows[band] = dict(zip(('start_index', 'stop_index'), window_indices(
                parameters, band, np.asarray(wave_source['wave'][:], dtype=float))))
        else:
            windows[band] = {'status': 'unknown; analysis interval not recorded'}
    handle.attrs['analysis_window_json'] = json.dumps(windows, sort_keys=True)
    handle.attrs['analysis_nonfinite_edge_policy'] = (
        'bandpass line detection trims only leading/trailing nonfinite flux bins; '
        'peak indexes retain the original window offset; internal nonfinite gaps reject the window'
        if parameters.get('spectral_window', {}).get('mode') == 'bandpass' else
        'explicit index window must be finite throughout')
    handle.attrs['switches_json'] = json.dumps(switches, sort_keys=True)
    for name, value in switches.items():
        handle.attrs[name] = bool(value)
    handle.attrs['master_seed'] = 'None' if seed is None else str(seed)
    handle.attrs['random_stream_version'] = STREAM_VERSION
    handle.attrs['input_sha256'] = '' if input_sha256 is None else str(input_sha256)
    handle.attrs['input_product'] = '' if input_path is None else str(Path(input_path).resolve())
    handle.attrs['source_start'] = int(source_start)
    handle.attrs['source_stop'] = len(handle['ID']) if source_stop is None and 'ID' in handle else (
        -1 if source_stop is None else int(source_stop))
    handle.attrs['software_versions_json'] = json.dumps({
        'python': platform.python_version(), 'numpy': np.__version__,
        'scipy': scipy.__version__, 'astropy': astropy.__version__,
        'h5py': h5py.__version__}, sort_keys=True)
    from cess.provenance import code_sha256
    from cess import __version__
    handle.attrs['software_version'] = __version__
    handle.attrs['code_sha256_json'] = json.dumps(code_sha256(), sort_keys=True)
    resources = {}
    for name, value in parameters.items():
        if (name.endswith(('_path', '_pkl')) and isinstance(value, (str, os.PathLike))
                and Path(value).is_file()):
            resources[name] = {'path': str(Path(value).resolve()), 'sha256': file_identity(value)}
    handle.attrs['resource_sha256_json'] = json.dumps(resources, sort_keys=True)
    if source is None:
        handle.attrs['pipeline_stage'] = 'simulation'
        handle.attrs['noise_model'] = 'poisson+independent_gaussian_read_noise; no saturation'
        handle.attrs['wce_model'] = ('linear pixel-grid resampling; ferr**2 diagonal plus '
                                     'covariance_lag1 for within-spectrum noise; cross-variant covariance not stored')
        handle.attrs['wce_applied'] = bool(switches['apply_wavelength_calibration_error'])
        handle.attrs['morphology_pa_lookup'] = (
            'original signed PA retained in parameters_desi; unsigned 0..90 deg '
            'center-symmetric projected-profile libraries use '
            'abs(((PA+90) mod 180)-90) before nearest lookup; '
            'custom signed/full-angle library lookup retains original nearest convention')
        if 'profile_library_provenance_json' in handle.attrs:
            recorded = json.loads(handle.attrs['profile_library_provenance_json'])
            if parameters['spectral_extraction']['mode'] == 'mge_strip':
                from cess.morphology.runtime import verify_mge_source_binding
                verify_mge_source_binding(parameters, recorded)
                identity = recorded['resources']['source_basis']
                actual = resources['positive_profile_library_path']
                if identity['path'] != actual['path'] or identity['sha256'] != actual['sha256']:
                    raise ValueError('MGE source-basis bytes changed before publication')
            else:
                if 'profile_library_manifest_path' in parameters:
                    from cess.morphology.libraries import verify_profile_bundle_binding
                    verify_profile_bundle_binding(parameters, recorded)
                for name, setting in (('width', 'widthlib_pkl'), ('height', 'heightlib_pkl')):
                    identity = recorded['resources'][name]
                    if (setting not in resources or
                            identity['sha256'] != resources[setting]['sha256'] or
                            str(Path(identity['path']).resolve()) != resources[setting]['path']):
                        raise ValueError('Profile resource bytes changed before publication: ' + name)
    else:
        handle.attrs['pipeline_stage'] = 'emission_line_postprocess'
        handle.attrs['source_product'] = str(source.filename)
        for name in ('noise_model', 'wce_model', 'wce_applied', 'morphology_pa_lookup'):
            handle.attrs[name] = source.attrs.get(name, 'unknown; input product did not record this')
        handle.attrs['profile_library_provenance_json'] = source.attrs.get(
            'profile_library_provenance_json', json.dumps({'provenance_status': 'unknown; parent did not record profile libraries'}))
        if 'spectral_extraction' in source:
            source_group = source['spectral_extraction']
            target_group = handle.create_group('spectral_extraction')
            for name, value in source_group.attrs.items():
                target_group.attrs[name] = value
            for name in ('source_fraction', 'spatial_pixels', 'profile_height_sum', 'valid'):
                data = source_group[name]
                values = data[:]
                if values.shape != (len(handle['ID']),):
                    raise ValueError('Parent extraction metadata is not aligned with selected ID')
                copied = target_group.create_dataset(name, data=values)
                for attribute, value in data.attrs.items():
                    copied.attrs[attribute] = value
        if 'mge_morphology' in source:
            parent_group = source['mge_morphology']
            target_group = handle.create_group('mge_morphology')
            for name, value in parent_group.attrs.items():
                target_group.attrs[name] = value
            for name in MGE_MORPHOLOGY_FIELDS:
                data = parent_group[name]
                values = data[:]
                expected = (len(handle['ID']), 4) if name in ('selected_parameters', 'selected_indices') else (len(handle['ID']),)
                if values.shape != expected:
                    raise ValueError('Parent MGE metadata is not aligned with selected ID: ' + name)
                copied = target_group.create_dataset(name, data=values)
                for attribute, value in data.attrs.items():
                    copied.attrs[attribute] = value
    for name in ('parameters_desi', 'parameters_grism', 'parameters_elg',
                 'parameters_phot', 'zero_order_phot'):
        if name in handle:
            if handle[name].ndim != 2:
                raise ValueError('Schema 2.0 parameter tables must be (source,field)')
            handle[name].attrs['axis_order'] = 'source,field'
            if name in ('parameters_phot', 'zero_order_phot'):
                handle[name].attrs['magnitude_definition'] = 'model AB magnitude'
                handle[name].attrs['magerr_definition'] = 'statistical linearised model magnitude error'
                handle[name].attrs['snr_definition'] = 'expected source electron counts / statistical sigma'
                names = handle[name].attrs.get('name', [])
                handle[name].attrs['units'] = [
                    'mag' if field.startswith('magerr') else 'AB mag' if field.startswith('mag') else '1'
                    for field in names]
            if 'name' in handle[name].attrs:
                handle[name].attrs['columns'] = handle[name].attrs['name']
    if 'parameters_desi' in handle:
        handle['parameters_desi'].attrs['units'] = [
            'deg', 'deg', '1', 'AB mag', 'AB mag', 'AB mag',
            '1', 'pixel', 'deg', '1', 'log10(Msun)']
    if 'parameters_grism' in handle:
        names = handle['parameters_grism'].attrs.get('name', [])
        handle['parameters_grism'].attrs['units'] = [
            'electron' if 'rms_in_e' in name else 'pixel' if 'wave_off' in name else '1'
            for name in names]
    def annotate(name, value):
        if isinstance(value, h5py.Dataset):
            leaf = name.rsplit('/', 1)[-1]
            if leaf in ('wave', 'intri_el_wave', 'detect_el_wave'):
                value.attrs['unit'] = 'Angstrom'
            elif leaf.startswith(('flux_ujy', 'ferr', 'response_flux_ujy')):
                value.attrs['unit'] = 'uJy'
                if leaf == 'flux_ujy':
                    value.attrs['spectral_quantity'] = handle.attrs['spectral_quantity']
                    value.attrs['definition'] = 'incident/unweighted photon-equivalent bin fnu before morphology/WCE'
                elif leaf.startswith('response_flux_ujy'):
                    value.attrs['spectral_quantity'] = handle.attrs['spectral_response_quantity']
                else:
                    value.attrs['spectral_quantity'] = handle.attrs['spectral_flux_calibration']
            elif leaf.startswith('flux_elec'):
                value.attrs['unit'] = 'electron'
                value.attrs['definition'] = (
                    'expected source detector electrons per output bin, summed over exposures; '
                    'response integrated at incident wavelength, followed by morphology/WCE; '
                    'finite detector support can exceed calibrated fnu support')
            elif leaf.startswith('spec_mask'):
                value.attrs['definition'] = 'configured analysis window and valid calibrated pixel support'
            elif leaf.startswith('snr_mask'):
                value.attrs['definition'] = 'analysis support with observed statistical SNR >= 1'
    handle.visititems(annotate)
    _record_spectral_geometry(handle)

