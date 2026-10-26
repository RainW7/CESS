"""Stateless common preparation and writers for the two legacy runners.

Noise calls, tuple unpacking and source-loop decisions stay in each runner.
Operations are supplied by that runner at task time so its public bindings and
the existing monkeypatch hooks retain their meaning. No resources load here at
import time and no random stream or configuration state is kept in this module.
"""
import numpy as np
from astropy.table import Table
from scipy.interpolate import interp1d


BANDS = ('GU', 'GV', 'GI')
PHOTOMETRIC_BANDS = ('nuv', 'u', 'g', 'r', 'i', 'z', 'y')


def validate_output_layout(layout):
    """Reject an unsupported storage layout before any scientific calculation."""
    if layout not in ('legacy', 'structured'):
        raise ValueError('output_layout must be legacy or structured')
    return layout


def finalize_product(handle, layout):
    """Finalize a newly written product inside its atomic publication scope.

    The structured layout moves HDF5 objects, retaining values, dtypes and
    dataset attributes without recalculation or a second spectral data copy.
    """
    validate_output_layout(layout)
    if layout == 'structured':
        from cess.io.products import structure_product
        structure_product(handle)


def prepare_resources(settings, legacy_axes, photometry, include_zero, operations):
    """Load existing cached resources and the explicitly selected profile model."""
    operations['configure_morphology'](settings)
    context = None
    widthlib = heightlib = None
    axes = legacy_axes
    noise_options = {}
    if settings['spectral_extraction']['mode'] == 'mge_strip':
        from cess.morphology.runtime import load_mge_source_library
        context = load_mge_source_library(settings)
        provenance = context['provenance']
        noise_options = {'normalize_morphology_kernel': False}
    else:
        widthlib, heightlib, axes, provenance = operations['load_profile_libraries'](settings, legacy_axes)
    cached = operations['cached_table']
    bkg = cached(settings['bkg_spec_path'])
    first = {band: cached(settings[band.lower() + 'tp_path']) for band in BANDS}
    zero = {band: cached(settings[band.lower() + 'tp0_path']) for band in BANDS} if include_zero else {}
    curves = {band: {'first': operations['throughput_arrays'](first[band])} for band in BANDS}
    if include_zero:
        for band in BANDS:
            curves[band]['zero_legacy'] = operations['throughput_arrays'](zero[band])
    interpolators = {band: {key: operations['throughput_interpolator'](table)
        for key, table in [('first', first[band])] +
            ([('zero_legacy', zero[band])] if include_zero else [])} for band in BANDS}
    photometric = {band: cached(settings[band + 'tp_path']) for band in PHOTOMETRIC_BANDS} if photometry else {}
    return dict(mge_context=context, widthlib=widthlib, heightlib=heightlib,
                axes=axes, profile_provenance=provenance,
                morphology_noise_options=noise_options, bkg=bkg,
                background_interpolator=interp1d(bkg['wavelength'], bkg['fnu']),
                response_curves=curves, interpolators=interpolators,
                zero_tables=zero, photometric_tables=photometric)


def prepare_input(engine, selected, settings, resources, operations):
    """Prepare the selected source interval without changing the forward model."""
    from cess.io.input_quality import input_quality_flags
    columns = [value.decode('utf-8') if isinstance(value, bytes) else str(value)
               for value in selected['parameters_name'][:]]
    parameters = Table(selected['parameters'][:], names=columns, copy=False)
    spec_name = next(name for name in selected['best_fit'] if name.startswith('spec_'))
    flux = selected['best_fit'][spec_name][:]
    redshift = selected['best_fit']['z'][:]
    quality = input_quality_flags(parameters, flux, redshift)
    rejected = quality != 0
    if rejected.any():
        # Only a private in-memory copy is masked. The HDF5 input and cached
        # source arrays retain their original values and row ordering.
        flux = flux.copy()
        flux[rejected] = np.nan
    bands = {}
    masks = []
    for band in BANDS:
        wave, incident, valid, invalid, folded = engine.kernels_convolve(
            selected['best_fit']['wavelength_rest'][:] * 1.e4,
            flux, redshift,
            band, settings[band.lower() + '_ree80'],
            responses=resources['response_curves'][band])
        bands[band] = dict(wave=wave, incident=incident, folded=folded)
        masks.append((valid, invalid))
    valid, invalid = operations['combine_band_masks'](masks, len(selected['ID']))
    quality[(quality == 0) & ~valid] |= np.uint32(256)
    valid &= quality == 0
    invalid = ~valid
    for spectrum in ([bands[band]['incident'] for band in BANDS] +
                     [value for band in BANDS for value in bands[band]['folded'].values()]):
        spectrum[~valid] = np.nan
    bkg = resources['bkg']
    reference = {band: {key: engine.native_sky_rate(
        0., 0., bkg['wavelength'], bkg['fnu'], native_wave, response,
        apply_position_scale=False)
        for key, (native_wave, response) in resources['response_curves'][band].items()}
        for band in BANDS}
    sky_rates = {band: {key: np.full(len(selected['ID']), np.nan)
                       for key in resources['response_curves'][band]} for band in BANDS}
    for band in BANDS:
        data = bands[band]
        data['response'] = {key: function(data['wave'])
                            for key, function in resources['interpolators'][band].items()}
        data['background_fnu'] = resources['background_interpolator'](data['wave'])
        data['source_rates'] = {key: engine.response_electron_rate(data['wave'], value * 1.e-29)
                                for key, value in data['folded'].items()}
        data['fnu'] = data['incident'] * 1.e-29
    return dict(spec_name=spec_name, bands=bands, mask=valid, mask_nan=invalid,
                sky_reference=reference, sky_rates=sky_rates, parameters=parameters,
                input_quality_flags=quality)


def prepare_source_profile(engine, morphology, settings, resources, prepared,
                           index, row_rng, apply_morphology, operations):
    """Use the original draw and lookup order; return original sampled values."""
    row = prepared['parameters'][index]
    ra, dec = row['RA'], row['DEC']
    scale = 1.
    if settings['background_mode'] == 'legacy_radec':
        scale = 10. ** (0.4 * (settings['background_v_mag'] - engine.match_bkg_mag(ra, dec)))
    for band, curves in prepared['sky_reference'].items():
        for key, reference in curves.items():
            prepared['sky_rates'][band][key][index] = reference * scale
    with operations['legacy_rng'](row_rng('morphology')):
        n, re, pa, ar = morphology.get_2d_param(
            row['MAG_Z'], 14, 21, 0.5, 5, row['log(M*)[0,1]_{MAL}'], row['z_{MAL}'])
    diagnostics = None
    if resources['mge_context'] is not None:
        from cess.morphology.runtime import select_mge_extraction
        selected = select_mge_extraction(resources['mge_context'], n, re, pa, ar,
                                         settings, apply_morphology=apply_morphology)
        width, height = selected['width'], selected['height']
        extraction = {name: selected[name] for name in
                      ('source_fraction', 'spatial_pixels', 'profile_height_sum')}
        diagnostics = selected['diagnostics']
    else:
        axes = [resources['axes'][name] for name in ('n', 're_pixel', 'pa_deg', 'axis_ratio')]
        width = morphology.match_input_paramters(resources['widthlib'], n, re, pa, ar, *axes)
        height = morphology.match_input_paramters(resources['heightlib'], n, re, pa, ar, *axes)
        extraction = operations['resolve_spectral_extraction'](settings, height)
    return n, re, pa, ar, width, height, extraction, diagnostics


def measure_source_bands(engine, ra, dec, wave, flux, tables, bkg, settings,
                         exposures, re, row_rng, component_prefix, operations):
    """Measure in the caller's declared order, including unseeded RNG behavior."""
    return {band: operations['measure_runner_band'](
        engine, ra, dec, wave, flux, table, bkg, settings, exposures[band], re,
        rng=row_rng(component_prefix + band)) for band, table in tables.items()}


def photometry_columns(results, bands):
    """Legacy parameter order: all model mags, all errors, all expected S/N."""
    return tuple(results[band][field] for field in
                 ('model_mag', 'model_mag_error', 'expected_snr') for band in bands)


def write_legacy_el_products(bands, operations):
    """Write existing ragged compatibility tables and details without reinterpretation."""
    # Preserve original order: all intrinsic tables, then all matched tables,
    # then diagnostics. `bands` contains (group, intrinsic, matched, S/N, details).
    for prefix, table_index, count_column in (('intri', 1, 'intri_el_number'),
                                               ('detect', 2, 'detect_el_number')):
        for item in bands:
            group, table = item[0], item[table_index]
            group.create_dataset(prefix + '_el_id', data=table['id'])
            for name, dtype in (('wave', np.dtype('float64')), ('idx', np.dtype('int64'))):
                import h5py
                data = group.create_dataset(prefix + '_el_' + name, (len(table),),
                                             dtype=h5py.vlen_dtype(dtype))
                operations['write_vlen_rows'](data, table[name])
                if name == 'wave':
                    data.attrs['unit'] = 'Angstrom'
            if prefix == 'detect':
                data = group.create_dataset('detect_el_snr', (len(table),),
                                             dtype=h5py.vlen_dtype(np.dtype('float64')))
                operations['write_vlen_rows'](data, item[3])
                data.attrs['unit'] = '1'
                data.attrs['definition'] = 'local mean per-pixel S/N around matched candidate; not integrated line S/N'
            group.create_dataset(prefix + '_el_elnumber', data=table[count_column])
    for group, _, _, _, details in bands:
        operations['write_el_details'](group, details)


def write_common_products(handle, groups, selected, settings, resources, prepared,
                          extraction_rows, mge_rows, covariance_rows, paired, operations):
    """Write geometry and forward products before any integrated line measurement."""
    handle.create_dataset('input_row_indices', data=selected.source_rows)
    if 'input_quality_flags' in prepared:
        from cess.io.input_quality import write_input_quality
        write_input_quality(handle, prepared['input_quality_flags'])
    operations['write_spectral_extraction'](handle, extraction_rows, settings,
                                            resources['profile_provenance'])
    if resources['mge_context'] is not None:
        operations['write_mge_morphology'](handle, mge_rows, settings)
    for band in BANDS:
        group, data = groups[band], prepared['bands'][band]
        variants = (('first', ''),) if paired else (('first', ''), ('zero_legacy', '_0th'))
        for key, suffix in variants:
            wave, response = resources['response_curves'][band][key]
            center_response = np.interp(group['wave'][:], wave, response)
            operations['write_response_products'](group, data['folded'][key],
                data['source_rates'][key], center_response, prepared['sky_reference'][band][key],
                prepared['sky_rates'][band][key], settings, suffix=suffix, paired=paired)
    for band in BANDS:
        for suffix, rows in covariance_rows[band].items():
            values = np.asarray(rows, dtype=float).reshape((len(selected['ID']),
                                                           len(prepared['bands'][band]['wave']) - 1))
            operations['write_spectral_covariance'](groups[band], values, suffix=suffix)


def write_base_spectra(handle, waves, incident, padded, *, paired):
    """Write existing base/partner spectra in field-band-variant order.

    Inputs are already converted/padded arrays. Standard partner keys use gu0
    and output _0th; paired partner keys/output use _wce. Intrinsic partner flux
    exists only for paired products, whereas partner masks exist only for
    standard products. No arrays, units, dtypes or attributes are changed here.
    """
    groups = {band: handle.create_group(band) for band in BANDS}
    for band in BANDS:
        groups[band].create_dataset('wave', data=waves[band])
    fields = (('flux_elec', 'elec_per_column'), ('flux_ujy', None),
              ('flux_ujy_with_noise', 'ujy_with_noise'), ('ferr', 'ferr'),
              ('snr', 'snr'), ('spec_mask', 'spec_mask'), ('snr_mask', 'snr_mask'))
    for dataset_name, field in fields:
        for band in BANDS:
            group, prefix = groups[band], band.lower()
            data = incident[band] if field is None else padded['padded_' + prefix + '_' + field]
            group.create_dataset(dataset_name, data=data)
            if field is None:
                if paired:
                    group.create_dataset('flux_ujy_wce', data=padded['padded_' + prefix + '_ujy_wce'])
            elif not (paired and field.endswith('_mask')):
                suffix = '_wce' if paired else '_0th'
                key = prefix + '_' + field + '_wce' if paired else prefix + '0_' + field
                group.create_dataset(dataset_name + suffix, data=padded['padded_' + key])
    return groups
