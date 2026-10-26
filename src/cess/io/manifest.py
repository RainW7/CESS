"""Strict, read-only product reconciliation and atomic local run manifests.

Only manifest format 1 is supported. Verification establishes provenance and
minimum source-aligned structure; it is not a validation of scientific models.
No existing simulation product is changed by this module. Imports perform no
configuration, calibration or catalogue I/O.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

from cess.config.loader import load_config
from cess.io.products import ProductView, open_product
from cess.pipeline.random import STREAM_VERSION
from cess.pipeline.support import (SourceSelection, normalise_options, simulation_filename,
                            SCHEMA_VERSION, REPAIR_REVISION, COVARIANCE_MODEL,
                            LINE_MEASUREMENT_DTYPE, MGE_MORPHOLOGY_FIELDS,
                            MGE_MORPHOLOGY_ATTRIBUTES)

MANIFEST_VERSION = 1
_MODES = ('standard', 'paired-wce', 'el-detect')
_OUTPUT_LAYOUTS = ('legacy', 'structured')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _input(job):
    return (Path(job['input_path']) / job['filename']).resolve()


def _output_layout(job):
    layout = job.get('output_layout', 'legacy')
    _require(layout in _OUTPUT_LAYOUTS, 'Unknown output layout: ' + str(layout))
    return layout


@contextmanager
def _source_product(path, mode):
    """Expose parent products logically; BayeSED simulation inputs stay raw."""
    if mode == 'el-detect':
        with open_product(path) as product:
            yield product.legacy_view()
    else:
        with h5py.File(str(path), 'r') as source:
            yield source


def _input_schema(path):
    with open_product(path) as product:
        return dict(schema_version=product.schema_version,
                    layout_version=_text(product.attrs.get('layout_version', '')))


def snapshot(config, options, mode, seed, jobs, output_layout='legacy'):
    """Capture effective settings and byte hashes for the planned local run.

    Code/resource selection matches record_runner_provenance. Hashes are read
    freshly, without a size/mtime cache, so resume does not trust file metadata.
    Worker count and scheduling memory estimates are deliberately absent.
    """
    if mode not in _MODES:
        raise ValueError('Unknown manifest mode: ' + str(mode))
    _require(output_layout in _OUTPUT_LAYOUTS, 'Unknown output layout: ' + str(output_layout))
    jobs = list(jobs)
    _require(all(_output_layout(job) == output_layout for job in jobs),
             'Job output layout differs from snapshot')
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError('Manifest seed must be a nonnegative integer or None.')
    config = load_config(config)
    from cess.provenance import code_sha256
    resources = {}
    for name, value in config.items():
        if (name.endswith(('_path', '_pkl')) and isinstance(value, (str, os.PathLike))
                and Path(value).is_file()):
            resources[name] = dict(path=str(Path(value).resolve()), sha256=_sha256(value))
    paths = sorted({_input(job) for job in jobs})
    return dict(config=_json_value(config), options=normalise_options(options), mode=mode,
                output_layout=output_layout,
                seed=None if seed is None else int(seed), random_stream_version=STREAM_VERSION,
                code_sha256=code_sha256(),
                resource_sha256=resources,
                inputs={str(path): _sha256(path) for path in paths},
                input_schemas={str(path): _input_schema(path) for path in paths}
                              if mode == 'el-detect' else {})


def expected_product(job):
    """Resolve exactly the existing runner's whole-file/source-slice name."""
    mode = job['mode']
    if mode not in _MODES:
        raise ValueError('Unknown manifest mode: ' + str(mode))
    _output_layout(job)
    with _source_product(_input(job), mode) as source:
        selected = SourceSelection(source, job.get('source_start', 0), job.get('source_stop'))
        if mode == 'el-detect' and len(selected['ID']) == selected.total_sources:
            filename = 'elg_info_' + Path(job['filename']).name
        else:
            filename = simulation_filename(job['filename'], selected,
                prefix='elg_info' if mode == 'el-detect' else 'CSST_grism')
    # Resolve the directory, not an existing final symlink masquerading as the
    # independently published product. Reconciliation rejects such leaves.
    return Path(job['output_dir']).resolve() / filename


def _text(value):
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _dataset(handle, name, shape=None, ndim=None):
    _require(name in handle and isinstance(handle[name], h5py.Dataset), 'Missing dataset: ' + name)
    data = handle[name]
    _require(shape is None or data.shape == shape, 'Wrong shape: ' + name)
    _require(ndim is None or data.ndim == ndim, 'Wrong rank: ' + name)
    return data


def _ragged_peaks(handle, stem, count_name, count, wave, initial, end, with_snr=False):
    """Validate parallel source-grouped peak arrays; return flattened centers."""
    ids = _dataset(handle, stem + 'id', ndim=1)[:]
    _require(ids.dtype.kind in 'iu' and np.all((ids >= 0) & (ids < count)),
             'Emission-line source row is outside root ID: ' + stem)
    _require(len(np.unique(ids)) == len(ids), 'Duplicate emission-line source row: ' + stem)
    counts = _dataset(handle, stem + count_name, shape=(len(ids),))[:]
    _require(counts.dtype.kind in 'iu' and np.all(counts >= 0), 'Invalid peak counts: ' + stem)
    arrays = {}
    for field, dtype in (('wave', np.dtype('float64')), ('idx', np.dtype('int64'))):
        data = _dataset(handle, stem + field, shape=(len(ids),))
        _require(h5py.check_dtype(vlen=data.dtype) == dtype, 'Wrong ragged dtype: ' + stem + field)
        arrays[field] = data
    if with_snr:
        data = _dataset(handle, stem + 'snr', shape=(len(ids),))
        _require(h5py.check_dtype(vlen=data.dtype) == np.dtype('float64'), 'Wrong ragged SNR dtype: ' + stem)
        arrays['snr'] = data
    result = []
    for row, source in enumerate(ids):
        values = {name: data[row] for name, data in arrays.items()}
        _require(all(value.ndim == 1 and len(value) == counts[row] for value in values.values()),
                 'Ragged wave/idx/count/SNR length mismatch: ' + stem)
        indices = values['idx']
        full = indices + initial
        _require(np.all((indices >= 0) & (indices < end-initial) & (full < len(wave))),
                 'Emission-line peak index outside spectrum: ' + stem)
        _require(np.all(np.isfinite(values['wave'])) and
                 np.all(np.isclose(values['wave'], wave[full], rtol=1.e-12, atol=0.)),
                 'Emission-line wavelength/index mismatch: ' + stem)
        result.extend((int(source), int(index), float(center))
                      for index, center in zip(full, values['wave']))
    return sorted(result)


def _flat_measurements(handle, band, suffix, count, wave, config, covariance_known=True,
                       missing_spectra=False):
    prefix = band + '/line_measurements' + suffix
    _require(prefix in handle, 'Missing line measurement group: ' + prefix)
    group = handle[prefix]
    _require(_text(group.attrs.get('algorithm', '')) == 'cess-affine-sideband-window-v1',
             'Unknown line measurement algorithm: ' + prefix)
    _require(json.loads(_text(group.attrs.get('window_config_json', '{}'))) == config['line_measurement'],
             'Line window config mismatch: ' + prefix)
    covariance_label = COVARIANCE_MODEL if covariance_known else 'unknown; metrics unavailable'
    _require(_text(group.attrs.get('covariance_model', '')) == covariance_label,
             'Line covariance metadata mismatch: ' + prefix)
    tables = {}
    for kind in ('candidates', 'truth'):
        data = _dataset(handle, prefix + '/' + kind, ndim=1)
        _require(data.dtype == LINE_MEASUREMENT_DTYPE, 'Wrong line measurement dtype: ' + data.name)
        for attr, expected in (('axis_order', 'peak'), ('flux_unit', 'erg/s/cm^2'),
                                ('wavelength_unit', 'Angstrom'), ('snr_unit', '1')):
            _require(_text(data.attrs.get(attr, '')) == expected, 'Line measurement metadata mismatch: ' + attr)
        rows = data[:]
        sources, peaks, centers = rows['source_row'], rows['peak_id'], rows['center_index']
        _require(np.all((sources >= 0) & (sources < count)) and np.all(peaks >= 0),
                 'Line measurement source/peak ID outside range: ' + data.name)
        keys = list(zip(sources.tolist(), peaks.tolist()))
        _require(len(set(keys)) == len(keys), 'Duplicate line measurement source/peak ID: ' + data.name)
        _require(np.all((centers >= 0) & (centers < len(wave))), 'Line center outside spectrum: ' + data.name)
        _require(np.all(np.isfinite(rows['center_wave'])) and
                 np.all(np.isclose(rows['center_wave'], wave[centers], rtol=1.e-12, atol=0.)),
                 'Line center wavelength/index mismatch: ' + data.name)
        design = config['line_measurement']
        width, side, gap = (design['line_width_pixels'][band], design['sideband_width_pixels'],
                             design['sideband_gap_pixels'])
        start = centers-width//2
        bounds = dict(line_start=start, line_stop=start+width,
            left_start=start-gap-side, left_stop=start-gap,
            right_start=start+width+gap, right_stop=start+width+gap+side,
            nline=width, nside=2*side)
        _require(all(np.all(rows[field] == value) for field, value in bounds.items()),
                 'Line window index/config mismatch: ' + data.name)
        if not covariance_known or missing_spectra:
            unavailable_flag = 256 if missing_spectra else 512
            _require(np.all(~rows['valid']) and np.all((rows['flag'] & unavailable_flag) != 0),
                     'Unavailable covariance/spectra has valid line metrics: ' + data.name)
            _require(all(np.all(np.isnan(rows[field])) for field in
                          ('flux', 'flux_error', 'snr', 'model_flux', 'model_snr')),
                     'Unavailable covariance/spectra has finite line metrics: ' + data.name)
        tables[kind] = {key: row for key, row in zip(keys, rows)}
    return tables


def _peak_values(table):
    return sorted((int(row['source_row']), int(row['center_index']), float(row['center_wave']))
                  for row in table.values())


def _el_tables(handle, count, band, wave, config, flat=False, suffix='', covariance_known=True,
               missing_spectra=False, baseline_truth=None):
    initial, end = config[band.lower() + '_init_idx'], config[band.lower() + '_end_idx']
    tables = _flat_measurements(handle, band, suffix, count, wave, config,
                                covariance_known, missing_spectra)
    if baseline_truth is not None:
        _require(set(tables['truth']) == set(baseline_truth) and
                 all((row['center_index'], row['center_wave']) ==
                     (baseline_truth[key]['center_index'], baseline_truth[key]['center_wave'])
                     for key, row in tables['truth'].items()), 'WCE truth differs from baseline peaks: ' + band)
    legacy = {}
    if not suffix:
        prefix = band.lower() + '_' if flat else band + '/'
        for kind in ('intri', 'detect'):
            legacy[kind] = _ragged_peaks(handle, prefix + kind + '_el_', 'elnumber',
                count, wave, initial, end, with_snr=kind == 'detect')
        _require(legacy['intri'] == _peak_values(tables['truth']), 'Truth table differs from intrinsic peaks: ' + band)
    diagnostic = band + '/el_diagnostics' + suffix
    _require(diagnostic in handle, 'Missing emission-line diagnostics: ' + band)
    ragged = {}
    for kind, number in (('candidates', 'candidate_el_number'), ('missed', 'missed_el_number'),
                          ('false_positive', 'false_el_number')):
        ragged[kind] = _ragged_peaks(handle, diagnostic + '/' + kind + '/', number,
                                     count, wave, initial, end)
    _require(ragged['candidates'] == _peak_values(tables['candidates']),
             'Candidate measurements differ from detector peaks: ' + band)
    pairs = _dataset(handle, diagnostic + '/pairs', ndim=1)
    pair_fields = {'id': 'iu', 'noisy_peak_id': 'iu', 'intrinsic_peak_id': 'iu',
                   'noisy_idx': 'iu', 'intrinsic_idx': 'iu', 'noisy_wave': 'f',
                   'intrinsic_wave': 'f', 'delta_wave': 'f'}
    _require(pairs.dtype.names is not None and all(name in pairs.dtype.names and
             pairs.dtype.fields[name][0].kind in kinds for name, kinds in pair_fields.items()),
             'Wrong emission-line pair structure: ' + band)
    matched_candidates, matched_truth = {}, {}
    for pair in pairs:
        source = int(pair['id'])
        noisy_key, truth_key = (source, int(pair['noisy_peak_id'])), (source, int(pair['intrinsic_peak_id']))
        _require(noisy_key in tables['candidates'] and truth_key in tables['truth'],
                 'Emission-line pair references missing source/peak ID: ' + band)
        _require(noisy_key not in matched_candidates and truth_key not in matched_truth,
                 'Emission-line peak reused in pairs: ' + band)
        noisy, truth = tables['candidates'][noisy_key], tables['truth'][truth_key]
        for kind, row in (('noisy', noisy), ('intrinsic', truth)):
            _require(int(pair[kind + '_idx']) + initial == row['center_index'] and
                     np.isclose(pair[kind + '_wave'], row['center_wave'], rtol=1.e-12, atol=0.),
                     'Emission-line pair wavelength/index mismatch: ' + band)
        _require(np.isclose(pair['delta_wave'], pair['noisy_wave']-pair['intrinsic_wave'],
                            rtol=1.e-12, atol=1.e-12), 'Emission-line pair delta mismatch: ' + band)
        matched_candidates[noisy_key], matched_truth[truth_key] = noisy, truth
    if not suffix:
        _require(legacy['detect'] == _peak_values(matched_candidates), 'Matched legacy peaks differ from pairs: ' + band)
    _require(ragged['missed'] == _peak_values({key: row for key, row in tables['truth'].items()
                                             if key not in matched_truth}), 'Missed peaks differ from pairs: ' + band)
    _require(ragged['false_positive'] == _peak_values({key: row for key, row in tables['candidates'].items()
                                                     if key not in matched_candidates}),
             'False-positive peaks differ from pairs: ' + band)
    return tables['truth']


def _covariance_dataset(handle, band, suffix, count, width):
    data = _dataset(handle, band + '/covariance_lag1' + suffix, shape=(count, width-1))
    _require(data.dtype == np.dtype('float64'), 'Wrong covariance dtype: ' + data.name)
    for attr, expected in (('unit', 'uJy^2'), ('axis_order', 'source,pixel_pair'),
                            ('covariance_model', COVARIANCE_MODEL)):
        _require(_text(data.attrs.get(attr, '')) == expected, 'Covariance metadata mismatch: ' + data.name)


def _profile_provenance(description, state):
    """Check legacy unknown provenance or an explicitly selected C7 binding."""
    _require(description.get('schema') == 'cess.profile_library_description',
             'Unknown profile-library provenance schema')
    if state['config']['spectral_extraction']['mode'] == 'mge_strip':
        from cess.morphology.runtime import load_mge_source_library, verify_mge_source_binding
        expected = load_mge_source_library(state['config'])['provenance']
        _require(description.get('version') == 3 and
                 description.get('provenance_status') == 'positive_source_basis_verified' and
                 description == expected, 'MGE source-basis provenance/config mismatch')
        verify_mge_source_binding(state['config'], description)
        for key, identity in (
                ('positive_profile_library_manifest_path', description['bundle_binding']),
                ('positive_profile_library_path', description['resources']['source_basis'])):
            resource = state['resource_sha256'][key]
            _require(identity['path'] == resource['path'] and identity['sha256'] == resource['sha256'],
                     'MGE source-basis identity differs from run snapshot: ' + key)
        return
    manifest = None
    generation = description['generation']
    if 'profile_library_manifest_path' in state['config']:
        from cess.morphology.profile_bundle import _manifest_contract
        from cess.morphology.libraries import _BUNDLE_LOOKUP
        _require(description.get('version') == 2 and
                 description.get('provenance_status') == 'manifest_verified_generation',
                 'Explicit profile bundle requires version-2 provenance')
        binding = description['bundle_binding']
        expected = state['resource_sha256']['profile_library_manifest_path']
        _require(binding['path'] == expected['path'] and binding['sha256'] == expected['sha256'] and
                 hashlib.sha256(binding['manifest_json'].encode('utf-8')).hexdigest() == expected['sha256'],
                 'Profile manifest snapshot/path/SHA mismatch')
        manifest = json.loads(binding['manifest_json'])
        _manifest_contract(manifest)
        recipe = manifest['rendering_recipe']
        _require(generation == {
            'status': 'manifest_recorded_generation',
            'physical_validation_status': recipe['physical_validation_status'],
            'psf': recipe['psf'], 'pixel_scale_arcsec_pixel': recipe['pixel_scale_arcsec_pixel'],
            'normalization': recipe['normalization'], 'rendering_recipe': recipe,
            'source_sha256': manifest['source_sha256'], 'software': manifest['software']},
            'Profile declared generation differs from its bound manifest')
        _require(description['runtime_interpretation'].get('lookup') == _BUNDLE_LOOKUP,
                 'Bundle nearest/endpoint lookup contract mismatch')
    else:
        _require(description.get('version') == 1 and
                 description.get('provenance_status') == 'legacy_unverified_generation',
                 'Legacy profile paths require unknown generation provenance')
        _require(generation.get('status') == 'unknown' and all(generation.get(field) is None for
                 field in ('psf', 'pixel_scale_arcsec_pixel', 'normalization')),
                 'Profile generation metadata must remain unknown')
    runtime = description['runtime_interpretation']
    axes = ('n', 're_pixel', 'pa_deg', 'axis_ratio')
    _require(runtime.get('axis_order') == list(axes) and set(runtime['axes']) == set(axes),
             'Profile runtime axis order/names mismatch')
    shape = []
    for name in axes:
        values = np.asarray(runtime['axes'][name], dtype=float)
        _require(values.ndim == 1 and values.size > 0 and np.all(np.isfinite(values)) and
                 np.all(np.diff(values) > 0), 'Invalid profile runtime axis: ' + name)
        shape.append(len(values))
    _require(runtime.get('arcsecperpix') == state['config']['arcsecperpix'] and
             runtime.get('arcsecperpix_unit') == 'arcsec/pixel', 'Profile runtime pixel scale mismatch')
    if manifest is not None:
        _require(runtime['axes'] == manifest['axes'] and shape == manifest['shape'] and
                 generation['pixel_scale_arcsec_pixel'] == runtime['arcsecperpix'] and
                 np.all((np.asarray(runtime['axes']['pa_deg']) >= 0.) &
                        (np.asarray(runtime['axes']['pa_deg']) <= 90.)),
                 'Profile manifest axes/shape/scale/PA contract mismatch')
    for role in ('width', 'height'):
        resource = description['resources'][role]
        expected = state['resource_sha256'][role + 'lib_pkl']
        _require(resource['path'] == expected['path'] and resource['sha256'] == expected['sha256'],
                 'Profile resource path/SHA mismatch: ' + role)
        _require(resource.get('shape') == shape, 'Profile resource shape/axes mismatch: ' + role)
        if manifest is not None:
            declared = manifest['resources'][role]
            _require(resource['path'] == str(Path(binding['path']).parent / declared['filename']) and
                     resource['sha256'] == declared['sha256'] and
                     resource['size_bytes'] == declared['size_bytes'],
                     'Profile member binding differs from manifest: ' + role)
            for name in ('profile_count', 'length_range', 'sum_range', 'profile_dtype_counts'):
                _require(description['observed_profile_statistics'][role][name] ==
                         manifest['observed_profile_statistics'][role][name],
                         'Decoded profile statistics differ from manifest: ' + role)
        _require(description['observed_profile_statistics'][role].get('sum_interpretation') ==
                 'observed stored profile sum; not a certified all-source aperture fraction',
                 'Profile sums must not be certified as aperture fractions: ' + role)
    setting = description['configured_generator_setting']
    _require(setting.get('status') == 'generator-setting-not-bound-to-existing-library',
             'Configured PSF is not generation provenance')


def _attribute_equal(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape:
        return False
    return np.array_equal(a, b, equal_nan=True) if a.dtype.kind in 'fc' and b.dtype.kind in 'fc' else np.array_equal(a, b)


def _same_attributes(actual, expected, label):
    _require(set(actual) == set(expected) and all(_attribute_equal(actual[name], expected[name])
             for name in expected), 'Inherited extraction attributes differ: ' + label)


def _verify_extraction(handle, count, mode, state, source):
    attribute = 'profile_library_provenance_json'
    _require(attribute in handle.attrs, 'Missing profile-library provenance')
    recorded = _text(handle.attrs[attribute])
    fields = ('source_fraction', 'spatial_pixels', 'profile_height_sum', 'valid')
    if mode == 'el-detect':
        expected = _text(source.attrs.get(attribute,
            json.dumps({'provenance_status': 'unknown; parent did not record profile libraries'})))
        _require(recorded == expected, 'EL profile provenance differs from parent')
        if 'spectral_extraction' not in source:
            _require('spectral_extraction' not in handle, 'EL invented extraction metadata absent from parent')
            _require(json.loads(recorded).get('provenance_status', '').startswith('unknown'),
                     'EL parent without extraction metadata must remain unknown')
            return
        _require('spectral_extraction' in handle, 'EL lost parent extraction metadata')
        parent, target = source['spectral_extraction'], handle['spectral_extraction']
        _same_attributes(target.attrs, parent.attrs, 'spectral_extraction')
        _require(set(target) == set(fields), 'Wrong inherited extraction fields')
        for name in fields:
            actual, original = _dataset(target, name, shape=(count,)), parent[name]
            # SourceSelection returns _SelectedDataset proxies, not h5py.Dataset.
            _require(original.shape == (count,) and actual.dtype == original.dtype and
                     _attribute_equal(actual[:], original[:]), 'Inherited extraction values differ: ' + name)
            _same_attributes(actual.attrs, original.attrs, name)
        return
    _profile_provenance(json.loads(recorded), state)
    _require('spectral_extraction' in handle, 'Missing spectral extraction metadata')
    group = handle['spectral_extraction']
    design = state['config']['spectral_extraction']
    _require(json.loads(_text(group.attrs.get('model_json', '{}'))) == design,
             'Spectral extraction model/config mismatch')
    _require(set(group) == set(fields), 'Wrong spectral extraction fields')
    values = {}
    for name in fields:
        data = _dataset(group, name, shape=(count,))
        _require(data.dtype == np.dtype(bool if name == 'valid' else float),
                 'Wrong spectral extraction dtype: ' + name)
        _require(_text(data.attrs.get('axis_order', '')) == 'source' and
                 _text(data.attrs.get('unit', '')) == ('pixel' if name == 'spatial_pixels' else '1'),
                 'Wrong spectral extraction metadata: ' + name)
        values[name] = data[:]
    valid = values['valid']
    _require(np.array_equal(valid, handle['data_mask'][:] == 1), 'Extraction validity differs from data_mask')
    for name in fields[:-1]:
        _require(np.all(np.isnan(values[name][~valid])), 'Invalid source has extraction values: ' + name)
    fraction, pixels, height_sum = [values[name][valid] for name in fields[:-1]]
    _require(np.all(np.isfinite(fraction)) and np.all((fraction >= 0) & (fraction <= 1)) and
             np.all(np.isfinite(pixels)) and np.all((pixels > 0) & (pixels == np.floor(pixels))) and
             np.all(np.isfinite(height_sum)) and np.all(height_sum > 0), 'Invalid extraction numerical values')
    if design['mode'] != 'mge_strip':
        _require(np.all(fraction == (1. if design['mode'] == 'legacy_full' else design['source_fraction'])),
                 'Extraction fraction differs from selected model')
    if design['mode'] == 'explicit':
        _require(np.all(pixels == design['spatial_pixels']), 'Extraction area differs from explicit model')


def _verify_mge_morphology(handle, count, mode, state, source):
    """Verify source links and joint-strip diagnostics without recomputing kernels."""
    if mode == 'el-detect':
        if 'mge_morphology' not in source:
            _require('mge_morphology' not in handle, 'EL invented MGE metadata')
            return
        parent, target = source['mge_morphology'], handle['mge_morphology']
        _same_attributes(target.attrs, parent.attrs, 'mge_morphology')
        _require(set(target) == set(MGE_MORPHOLOGY_FIELDS), 'Wrong inherited MGE fields')
        for name in MGE_MORPHOLOGY_FIELDS:
            shape = (count, 4) if name in ('selected_parameters', 'selected_indices') else (count,)
            actual, original = _dataset(target, name, shape=shape), parent[name]
            _require(actual.dtype == original.dtype and _attribute_equal(actual[:], original[:]),
                     'EL inherited MGE values differ: ' + name)
            _same_attributes(actual.attrs, original.attrs, name)
        return
    design = state['config']['spectral_extraction']
    if design['mode'] != 'mge_strip':
        _require('mge_morphology' not in handle, 'Legacy mode has unexpected MGE metadata')
        return
    group = handle['mge_morphology']
    _require(set(group) == set(MGE_MORPHOLOGY_FIELDS), 'Wrong MGE morphology fields')
    _require(json.loads(_text(group.attrs['model_json'])) == design, 'MGE extraction model/config mismatch')
    for name, expected in MGE_MORPHOLOGY_ATTRIBUTES.items():
        _require(name in group.attrs and _attribute_equal(group.attrs[name], expected),
                 'Wrong MGE model metadata: ' + name)
    values = {}
    for name in MGE_MORPHOLOGY_FIELDS:
        shape = (count, 4) if name in ('selected_parameters', 'selected_indices') else (count,)
        data = _dataset(group, name, shape=shape)
        boolean = name in ('valid', 'kernel_applied')
        _require(data.dtype == np.dtype(bool if boolean else float), 'Wrong MGE dtype: ' + name)
        _require(_text(data.attrs['axis_order']) == ('source,parameter_axis' if len(shape) == 2 else 'source'),
                 'Wrong MGE source axis: ' + name)
        _require(_text(data.attrs.get('unit', '')) ==
                 ('mixed; see selected_parameter_units' if name == 'selected_parameters' else '1'),
                 'Wrong MGE unit: ' + name)
        values[name] = data[:]
    valid = values['valid']
    _require(np.array_equal(valid, handle['spectral_extraction/valid'][:]), 'MGE validity/source-row mismatch')
    applied = state['options']['apply_morphology_effect']
    _require(np.array_equal(values['kernel_applied'], valid & applied), 'MGE kernel-applied switch mismatch')
    for name in MGE_MORPHOLOGY_FIELDS:
        if name not in ('valid', 'kernel_applied'):
            _require(np.all(np.isnan(values[name][~valid])) and np.all(np.isfinite(values[name][valid])),
                     'MGE invalid/missing numerical rows: ' + name)
    fraction = handle['spectral_extraction/source_fraction'][:][valid]
    pixels = handle['spectral_extraction/spatial_pixels'][:][valid]
    height_sum = handle['spectral_extraction/profile_height_sum'][:][valid]
    mass, absolute = values['kernel_window_fraction'][valid], values['kernel_absolute_fraction'][valid]
    _require(np.all(fraction > 0.) and np.all(pixels == design['spatial_pixels']) and
             np.allclose(height_sum, fraction, rtol=2.e-12, atol=2.e-14), 'MGE strip fraction/height/noise-area mismatch')
    _require(np.all((mass >= 0.) & (mass <= 1. + 1.e-12)) and
             np.allclose(absolute, fraction * mass, rtol=2.e-12, atol=2.e-14), 'MGE joint fraction mismatch')
    expected_bins = 2 * design['kernel_radius_pixels'] + 1 if applied else 1
    _require(np.all(values['kernel_n_bins'][valid] == expected_bins), 'MGE kernel support mismatch')
    if not applied:
        _require(np.all(mass == 1.) and np.all(values['quadrature_error_estimate'][valid] == 0.) and
                 np.all(values['omitted_probability_bound'][valid] == 0.), 'Disabled MGE dispersion changed kernel')
    for name in ('source_fit_l1_estimate', 'conditional_fit_error_indicator',
                 'quadrature_error_estimate', 'omitted_probability_bound'):
        _require(np.all(values[name][valid] >= 0.), 'Negative MGE diagnostic: ' + name)
    estimate = values['source_fit_l1_estimate'][valid]
    _require(np.allclose(values['conditional_fit_error_indicator'][valid], 2. * estimate / fraction,
                         rtol=2.e-12, atol=2.e-14), 'MGE conditional fit indicator mismatch')
    from cess.io.parameters import read_parameter_table
    from cess.morphology.runtime import load_mge_source_library, select_mge_grid, verify_mge_source_binding
    description = json.loads(_text(handle.attrs['profile_library_provenance_json']))
    context = load_mge_source_library(state['config'])
    table = read_parameter_table(handle['parameters_desi'], ids=handle['ID'], required=('n', 'Re', 'PA', 'baratio'))
    original = np.column_stack([table[name][valid] for name in ('n', 'Re', 'PA', 'baratio')])
    indices = values['selected_indices'][valid]
    selected = values['selected_parameters'][valid]
    for row, sampled in enumerate(original):
        expected = select_mge_grid(context, *sampled)
        _require(np.array_equal(indices[row], expected['selected_indices']) and
                 np.array_equal(selected[row], expected['selected_parameters']),
                 'MGE selected profile is not linked to the original source row')
    fit_l1 = {i: context['fits'][i]['metadata']['estimated_radial_L1_including_outside']
              for i in set(indices[:, 0].astype(int))}
    _require(np.array_equal(estimate, [fit_l1[int(i)] for i in indices[:, 0]]),
             'MGE row source-fit estimate differs from selected n fit')
    verify_mge_source_binding(state['config'], description)


def _input_quality(handle, count, mode, source=None):
    """Validate optional new quality flags; old parents retain unknown status."""
    present = 'input_quality_flags' in handle
    if mode == 'el-detect':
        inherited = source is not None and 'input_quality_flags' in source
        _require(present == inherited, 'EL input quality availability differs from parent')
    if not present:
        return
    from cess.io.input_quality import POLICY_VERSION, POLICY, FLAG_MEANINGS
    data = _dataset(handle, 'input_quality_flags', shape=(count,))
    _require(data.dtype.kind == 'u' and data.dtype.itemsize == 4,
             'Input quality flags must be uint32')
    flags = data[:]
    _require(_text(data.attrs.get('unit', '')) == '1' and
             _text(data.attrs.get('axis_order', '')) == 'source',
             'Input quality flags require source axis and dimensionless unit')
    _require(_text(data.attrs.get('policy_version', '')) == POLICY_VERSION and
             json.loads(_text(data.attrs.get('policy_json', 'null'))) == POLICY,
             'Input quality policy metadata differs from current policy')
    _require(np.array_equal(data.attrs.get('flag_masks'), np.array(list(FLAG_MEANINGS), dtype=np.uint32)) and
             _text(data.attrs.get('flag_meanings', '')) == ' '.join(FLAG_MEANINGS.values()),
             'Input quality flag definitions are missing or inconsistent')
    known = np.uint32(0)
    for flag in FLAG_MEANINGS:
        known |= np.uint32(flag)
    _require(np.all((flags & ~known) == 0), 'Input quality flags contain undefined bits')
    if mode == 'el-detect':
        parent = source['input_quality_flags']
        _require(data.dtype == parent.dtype and np.array_equal(flags, parent[:]),
                 'EL input quality flags differ from selected parent rows')
        _require(set(data.attrs) == set(parent.attrs), 'EL input quality attributes differ from parent')
        for name in data.attrs:
            _require(data.attrs.get_id(name).dtype == parent.attrs.get_id(name).dtype and
                     np.array_equal(data.attrs[name], parent.attrs[name]),
                     'EL input quality attribute differs from parent: ' + name)
    else:
        mask = _dataset(handle, 'data_mask', shape=(count,))[:]
        _require(np.array_equal(flags == 0, mask == 1) and
                 np.array_equal(flags != 0, mask == 2),
                 'Input quality flags differ from data_mask validity')


def _structure(handle, count, mode, options, source=None, config=None):
    config = json.loads(_text(handle.attrs['config_json'])) if config is None else config
    _input_quality(handle, count, mode, source)
    if mode == 'el-detect':
        _dataset(handle, 'parameters_elg', shape=(count, 2))
        for band in ('GV', 'GI'):
            _require(source is not None, 'EL verification requires its immediate source product')
            wave = _dataset(source, band + '/wave', ndim=1)[:]
            def availability(suffix):
                covariance_name = band + '/covariance_lag1' + suffix
                known = (covariance_name in source and
                         _text(source[covariance_name].attrs.get('covariance_model', '')) == COVARIANCE_MODEL)
                missing = any(band + '/' + name + suffix not in source for name in ('flux_ujy_with_noise', 'ferr'))
                return known, missing
            known, missing = availability('')
            truth = _el_tables(handle, count, band, wave, config, flat=True,
                               covariance_known=known, missing_spectra=missing)
            if band + '/flux_ujy_with_noise_wce' in source:
                known, missing = availability('_wce')
                _el_tables(handle, count, band, wave, config, suffix='_wce',
                           covariance_known=known, missing_spectra=missing, baseline_truth=truth)
    else:
        for table in ('parameters_desi', 'parameters_grism'):
            data = _dataset(handle, table, ndim=2)
            _require(data.shape[0] == count, 'Unaligned source table: ' + table)
        _dataset(handle, 'data_mask', shape=(count,))
        if options['apply_photometric']:
            table = _dataset(handle, 'parameters_phot', ndim=2)
            _require(table.shape[0] == count, 'Unaligned parameters_phot')
            _require('photometry' in handle, 'Missing photometry group')
            for band in ('nuv', 'u', 'g', 'r', 'i', 'z', 'y'):
                _dataset(handle, 'photometry/' + band + '/valid', shape=(count,))
        for band in ('GU', 'GV', 'GI'):
            wave = _dataset(handle, band + '/wave', ndim=1)[:]
            _require(len(wave) >= 2 and np.all(np.isfinite(wave)) and
                     np.all(wave > 0) and np.all(np.diff(wave) > 0), 'Invalid wavelength grid: ' + band)
            width = len(wave)
            _dataset(handle, band + '/wave_edges', shape=(width + 1,))
            _dataset(handle, band + '/pixel_width', shape=(width,))
            names = ('flux_ujy', 'flux_ujy_with_noise', 'ferr', 'snr', 'flux_elec',
                     'spec_mask', 'snr_mask', 'response_flux_ujy', 'source_electron_rate_pre_morph',
                     'flux_ujy_model', 'detector_model_mask', 'calibration_support_mask')
            for name in names:
                _dataset(handle, band + '/' + name, shape=(count, width))
            _dataset(handle, band + '/response_at_pixel_center', shape=(width,))
            _dataset(handle, band + '/sky_rate_e_s_pixel', shape=(count,))
            _covariance_dataset(handle, band, '', count, width)
            if mode == 'standard':
                for name in names:
                    if name != 'flux_ujy':
                        _dataset(handle, band + '/' + name + '_0th', shape=(count, width))
                _dataset(handle, band + '/response_at_pixel_center_0th', shape=(width,))
                _dataset(handle, band + '/sky_rate_e_s_pixel_0th', shape=(count,))
                _covariance_dataset(handle, band, '_0th', count, width)
            if mode == 'paired-wce':
                for name in ('flux_elec_wce', 'flux_ujy_wce', 'flux_ujy_with_noise_wce',
                             'ferr_wce', 'snr_wce', 'flux_ujy_model_wce',
                             'detector_model_mask_wce', 'calibration_support_mask_wce'):
                    _dataset(handle, band + '/' + name, shape=(count, width))
                _covariance_dataset(handle, band, '_wce', count, width)
            if options['apply_el_detect'] and band in ('GV', 'GI'):
                truth = _el_tables(handle, count, band, wave, config)
                if mode == 'paired-wce':
                    _el_tables(handle, count, band, wave, config, suffix='_wce', baseline_truth=truth)
        if mode == 'standard':
            _require('zero_order' in handle, 'Missing zero_order photometry group')
            for band in ('GU', 'GV', 'GI'):
                _dataset(handle, 'zero_order/' + band + '/valid', shape=(count,))

    def aligned(name, value):
        if isinstance(value, h5py.Dataset):
            order = _text(value.attrs.get('axis_order', ''))
            if order.split(',')[0] == 'source':
                _require(value.ndim > 0 and value.shape[0] == count,
                         'Unaligned declared source axis: ' + name)
    handle.visititems(aligned)


def verify_product(job, snapshot, expected_sha256=None):
    """Return {output, sha256} for a valid product, None only when absent.

    Existing files must match this run's bytes/provenance, IDs, row mapping and
    minimum schema-2.5 scientific structure in the requested physical layout.
    Structured containers are validated before their lossless legacy view is
    examined. Any mismatch raises ValueError; no repair,
    overwrite, science-value clipping or historical-schema inference is done.
    EL outputs inherit original-catalogue rows and additionally record the
    selected local rows of their immediate parent product.
    """
    layout = _output_layout(job)
    _require(snapshot.get('output_layout') == layout,
             'Job output layout differs from snapshot, or snapshot predates layout identity')
    path = expected_product(job)
    _require(not path.is_symlink(), 'Expected product must not be a symlink: ' + str(path))
    if not path.exists():
        return None
    try:
        source_path = _input(job)
        expected_input = snapshot['inputs'][str(source_path)]
        _require(_sha256(source_path) == expected_input, 'Input bytes changed after snapshot')
        _require(job['mode'] == snapshot['mode'], 'Job mode differs from snapshot')
        _require(load_config(job['config']) == snapshot['config'], 'Job config differs from snapshot')
        _require(normalise_options(job['options']) == snapshot['options'], 'Job switches differ from snapshot')
        _require(job.get('seed') == snapshot['seed'], 'Job seed differs from snapshot')
        if job['mode'] == 'el-detect':
            _require(snapshot.get('input_schemas', {}).get(str(source_path)) == _input_schema(source_path),
                     'EL input schema/layout identity differs from snapshot')
        digest = _sha256(path)
        _require(expected_sha256 is None or digest == expected_sha256, 'Output SHA256 mismatch')
        with _source_product(source_path, job['mode']) as source:
            selected = SourceSelection(source, job.get('source_start', 0), job.get('source_stop'))
            ids = selected['ID'][:]
            local_rows = selected.source_rows
            catalogue_rows = (selected['input_row_indices'][:] if job['mode'] == 'el-detect'
                              and 'input_row_indices' in selected else local_rows)
            with h5py.File(str(path), 'r') as raw_product:
                actual_schema = _text(raw_product.attrs.get('schema_version', ''))
                if layout == 'structured':
                    _require(actual_schema == '3.0', 'Requested structured output has wrong physical schema')
                    view = ProductView(raw_product)
                    _require(_text(view.attrs.get('mode', '')) == job['mode'],
                             'Structured product mode differs from job')
                    _require(_text(view.attrs.get('product_kind', '')) ==
                             ('emission_line_postprocess' if job['mode'] == 'el-detect' else 'simulation'),
                             'Structured product kind differs from job')
                    product = view.legacy_view()
                else:
                    _require(actual_schema == SCHEMA_VERSION,
                             'Requested legacy output has wrong physical schema')
                    product = raw_product
                _require('merge_history' not in product,
                         'Merged products are not native simulation job outputs')
                for attr, expected in (
                    ('schema_version', SCHEMA_VERSION), ('cess_repair_revision', REPAIR_REVISION),
                    ('input_sha256', expected_input),
                    ('input_product', str(source_path)),
                    ('master_seed', 'None' if snapshot['seed'] is None else str(snapshot['seed'])),
                    ('random_stream_version', snapshot['random_stream_version']),
                    ('pipeline_stage', 'emission_line_postprocess' if job['mode'] == 'el-detect' else 'simulation')):
                    _require(attr in product.attrs and _text(product.attrs[attr]) == expected,
                             'Product attribute mismatch: ' + attr)
                for attr, expected in (('config_json', snapshot['config']),
                    ('switches_json', snapshot['options']), ('code_sha256_json', snapshot['code_sha256']),
                    ('resource_sha256_json', snapshot['resource_sha256'])):
                    _require(attr in product.attrs and json.loads(_text(product.attrs[attr])) == expected,
                             'Product provenance mismatch: ' + attr)
                for attr, expected in snapshot['options'].items():
                    _require(attr in product.attrs and bool(product.attrs[attr]) == expected,
                             'Product switch mismatch: ' + attr)
                for attr, expected in (('source_start', selected.selection.start),
                                       ('source_stop', selected.selection.stop)):
                    _require(attr in product.attrs and product.attrs[attr] == expected,
                             'Product source interval mismatch: ' + attr)
                output_ids = _dataset(product, 'ID', shape=(len(ids),))
                actual = output_ids[:]
                input_dtype = selected['ID'].dtype
                _require(output_ids.dtype == input_dtype and output_ids.dtype.metadata == input_dtype.metadata
                         and np.array_equal(actual, ids), 'Product ID/order/dtype mismatch')
                rows = _dataset(product, 'input_row_indices', shape=(len(ids),))[:]
                _require(rows.dtype.kind in 'iu' and np.array_equal(rows, catalogue_rows),
                         'Product input_row_indices mismatch')
                if job['mode'] == 'el-detect':
                    rows = _dataset(product, 'parent_row_indices', shape=(len(ids),))[:]
                    _require(rows.dtype.kind in 'iu' and np.array_equal(rows, local_rows),
                             'Product parent_row_indices mismatch')
                _structure(product, len(ids), job['mode'], snapshot['options'], source=selected,
                           config=snapshot['config'])
                _verify_extraction(product, len(ids), job['mode'], snapshot, selected)
                _verify_mge_morphology(product, len(ids), job['mode'], snapshot, selected)
        return dict(output=str(path), sha256=digest)
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('Cannot verify product {}: {}'.format(path, exc)) from exc


def _sync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_manifest(path, summary, replace=False):
    """Atomically persist JSON, fsyncing before publication in the same directory.

    Initial writes use an atomic hard-link no-clobber operation. Updates use
    os.replace and require the caller's manifest_lock for concurrent parents.
    Serialization or temp-write failure leaves a previously published file
    intact. The lock inode is independent of this replaced manifest inode.
    """
    path = Path(path).expanduser().resolve()
    encoded = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                prefix='.' + path.name + '.', suffix='.tmp', dir=str(path.parent), delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(str(temporary), str(path))
        else:
            os.link(str(temporary), str(path))
        _sync_directory(path.parent)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return path


@contextmanager
def manifest_lock(path):
    """Hold an advisory nonblocking parent-process lock on Mac/Linux.

    The persistent .lock file is never unlinked: deleting a busy inode would
    let another parent acquire a different lock. OS descriptor cleanup releases
    the flock after process exit/crash; leftover lock files are harmless.
    """
    lock_path = Path(str(Path(path).expanduser().resolve()) + '.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Manifest is locked by another parent: ' + str(lock_path)) from exc
        try:
            yield lock_path
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
