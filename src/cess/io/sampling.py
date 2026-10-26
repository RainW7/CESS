"""Seed-compatible stratified subset export through the unified product reader.

Exports retain the old extracted-mock-spectra-1.1 schema and API. They preserve
selected spectral arrays, recorded parent metadata and row links, without
claiming to clone all unselected science/unknown extension data. Pandas is a
lazy tools dependency used only when redshift qcut is actually required.
"""
import argparse
import h5py
import numpy as np
from pathlib import Path
import json
import sys
from .parameters import read_parameter_table
from .products import open_product
from .atomic import atomic_hdf5

_ROW_LINKS = ('input_row_indices', 'parent_row_indices')
_ROW_PREFIXES = ('flux_', 'response_flux_ujy', 'ferr', 'snr', 'spec_mask',
    'covariance_lag1', 'source_electron_rate_pre_morph', 'sky_rate_e_s_pixel',
    'detector_model_mask', 'calibration_support_mask')
_SHARED_FIELDS = ('response_at_pixel_center', 'wave_edges', 'pixel_width')

_BANDS = ('GU', 'GV', 'GI')
_OPTIONAL_SPECTRAL = ('ferr', 'spec_mask', 'snr_mask')

def read_hdf5_files(file_paths):
    selected_sources = []
    gu_wave, gv_wave, gi_wave = None, None, None

    for file_path in file_paths:
        with open_product(file_path) as product:
            f = product.legacy_view()
            # Check wavelength consistency
            current_gu_wave = f['GU/wave'][:]
            current_gv_wave = f['GV/wave'][:]
            current_gi_wave = f['GI/wave'][:]

            if gu_wave is None:
                gu_wave = current_gu_wave
                gv_wave = current_gv_wave
                gi_wave = current_gi_wave
            else:
                if (not np.array_equal(gu_wave, current_gu_wave) or
                    not np.array_equal(gv_wave, current_gv_wave) or
                    not np.array_equal(gi_wave, current_gi_wave)):
                    raise ValueError(f"Wave data mismatch in {file_path}")

            ids = f['ID'][:]
            params_desi = read_parameter_table(
                f['parameters_desi'], ids=ids, required=('z_best', 'str_mass'))
            params_grism = read_parameter_table(
                f['parameters_grism'], ids=ids,
                required=('gu_snr_mean', 'gv_snr_mean', 'gi_snr_mean'))
            spectral_keys = {band: _spectral_fields(f[band], len(ids), len(f[band]['wave']), file_path)
                             for band in _BANDS}
            metadata = _parent_metadata(product, f)
            for key in _ROW_LINKS:
                if key in f and (f[key].shape != (len(ids),) or f[key].dtype.kind not in 'iu'):
                    raise ValueError('{} must contain one integer per ID'.format(key))
            if 'data_mask' in f and f['data_mask'].shape not in ((len(ids),), (len(ids), 1)):
                raise ValueError('data_mask must contain one value per ID')

            # Iterate through sources
            for i in range(len(ids)):
                gv_snr = params_grism['gv_snr_mean'][i]
                gi_snr = params_grism['gi_snr_mean'][i]

                if gv_snr >= 3 and gi_snr >= 3:
                    source = {
                        'id': ids[i],
                        'z_best': params_desi['z_best'][i],
                        'str_mass': params_desi['str_mass'][i],
                        'gu_snr': params_grism['gu_snr_mean'][i],
                        'gv_snr': gv_snr,
                        'gi_snr': gi_snr,
                        'gu_flux': f['GU/flux_ujy'][i],
                        'gu_flux_noise': f['GU/flux_ujy_with_noise'][i],
                        'gv_flux': f['GV/flux_ujy'][i],
                        'gv_flux_noise': f['GV/flux_ujy_with_noise'][i],
                        'gi_flux': f['GI/flux_ujy'][i],
                        'gi_flux_noise': f['GI/flux_ujy_with_noise'][i],
                        'source_file': str(Path(file_path).resolve()),
                        'source_row': i,
                        'source_metadata': metadata,
                    }
                    for band in _BANDS:
                        source[band.lower() + '_extras'] = {
                            key: f[band][key][i] for key in spectral_keys[band]
                            if key + '_available' not in f[band] or bool(f[band][key + '_available'][i])
                        }
                    if 'data_mask' in f and ('data_mask_available' not in f or bool(f['data_mask_available'][i])):
                        source['data_mask'] = np.asarray(f['data_mask'][i]).item()
                    for key in _ROW_LINKS:
                        if key in f and (key + '_available' not in f or bool(f[key + '_available'][i])):
                            source[key] = f[key][i]
                    if 'provenance' in f:
                        prior = f['provenance']
                        source['upstream_lineage'] = {key: prior[key][i] for key in ('source_file', 'source_row')
                            if key in prior and isinstance(prior[key], h5py.Dataset)
                            and prior[key].shape == (len(ids),)}
                    selected_sources.append(source)

    return selected_sources, gu_wave, gv_wave, gi_wave

def stratified_sampling(sources, target=1000, n_bins=10, seed=None):
    """Sample without replacement, with proportional redshift quantile quotas.

    ``seed=None`` retains the caller's global NumPy RNG convention. A supplied
    seed uses an isolated RandomState. Constant redshifts and non-finite values
    form a fallback stratum rather than leaving an empty-bin adjustment loop.
    """
    if isinstance(target, (bool, np.bool_)) or int(target) != target or target < 0:
        raise ValueError('target must be a non-negative integer')
    if isinstance(n_bins, (bool, np.bool_)) or int(n_bins) != n_bins or n_bins < 1:
        raise ValueError('n_bins must be a positive integer')
    target, n_bins = int(target), int(n_bins)
    if not sources:
        return []
    if target == 0:
        return []
    if len(sources) <= target:
        return list(sources)
    rng = np.random if seed is None else np.random.RandomState(seed)
    redshifts = np.asarray([source['z_best'] for source in sources], dtype=float)
    finite = np.isfinite(redshifts)
    labels = np.full(len(sources), -1, dtype=int)
    if np.any(finite):
        import pandas as pd
        try:
            quantiles = np.asarray(pd.qcut(
                redshifts[finite], min(n_bins, int(finite.sum())),
                labels=False, duplicates='drop'), dtype=float)
            valid = np.isfinite(quantiles)
            indices = np.flatnonzero(finite)[valid]
            labels[indices] = quantiles[valid].astype(int)
        except ValueError:
            # An entirely degenerate distribution is one valid random stratum.
            pass
    bin_ids, counts = np.unique(labels, return_counts=True)
    ideal = counts.astype(float) * target / len(sources)
    quotas = np.floor(ideal).astype(int)
    remainder = target - int(quotas.sum())
    # Largest-remainder apportionment is bounded and never exceeds capacity.
    order = np.argsort(-(ideal - quotas), kind='mergesort')
    for index in order:
        if remainder and quotas[index] < counts[index]:
            quotas[index] += 1
            remainder -= 1
    if remainder:
        raise RuntimeError('unable to allocate the requested sample count')
    selected_indices = []
    for bin_id, count in zip(bin_ids, quotas):
        candidates = np.flatnonzero(labels == bin_id)
        selected_indices.extend(rng.choice(candidates, int(count), replace=False))
    rng.shuffle(selected_indices)
    return [sources[int(index)] for index in selected_indices]

def write_output(output_path, sources, gu_wave, gv_wave, gi_wave,
                 seed=None, input_files=None, target=None, n_bins=None):
    with atomic_hdf5(output_path) as f:
        f.attrs['schema_version'] = 'extracted-mock-spectra-1.1'
        f.attrs['selection'] = 'gv_snr_mean >= 3 and gi_snr_mean >= 3'
        f.attrs['sampling_method'] = 'redshift qcut; proportional largest-remainder quotas; without replacement'
        f.attrs['sampling_seed'] = 'global NumPy RNG (seed not recorded)' if seed is None else int(seed)
        if target is not None:
            f.attrs['sampling_target'] = int(target)
        if n_bins is not None:
            f.attrs['sampling_n_bins'] = int(n_bins)
        if input_files is not None:
            f.attrs['input_files'] = json.dumps([str(Path(path).resolve()) for path in input_files])
        # Write IDs
        ids = [s['id'] for s in sources]
        ids = np.asarray(ids) if ids else np.asarray([], dtype='S1')
        if ids.dtype.kind in ('U', 'O'):
            ids = np.asarray([value.decode('utf-8') if isinstance(value, bytes) else str(value)
                              for value in ids], dtype=h5py.string_dtype('utf-8'))
        source_dtypes = [s.get('source_metadata', {}).get('id_dtype') for s in sources]
        common_dtype = source_dtypes[0] if source_dtypes and all(
            value is not None and value == source_dtypes[0] and value.metadata == source_dtypes[0].metadata
            for value in source_dtypes) else None
        f.create_dataset('ID', data=ids, dtype=common_dtype)
        
        # Write spectral data
        for group, wave in [('GU', gu_wave), ('GV', gv_wave), ('GI', gi_wave)]:
            grp = f.create_group(group)
            grp.create_dataset('wave', data=wave)
            
            flux_data = np.array([s[f'{group.lower()}_flux'] for s in sources]).reshape(len(sources), len(wave))
            flux_noise_data = np.array([s[f'{group.lower()}_flux_noise'] for s in sources]).reshape(len(sources), len(wave))
            
            grp.create_dataset('flux_ujy', data=flux_data)
            grp.create_dataset('flux_ujy_with_noise', data=flux_noise_data)
            extra_key = group.lower() + '_extras'
            keys = sorted({key for source in sources for key in source.get(extra_key, {})})
            for key in keys:
                available = np.asarray([key in source.get(extra_key, {}) for source in sources], dtype=bool)
                reference = np.asarray(next(source[extra_key][key] for source in sources if key in source.get(extra_key, {})))
                fill_value = False if 'mask' in key or reference.dtype.kind == 'b' else np.nan
                rows = [np.asarray(source.get(extra_key, {}).get(key, np.full(reference.shape, fill_value)))
                        for source in sources]
                if any(row.shape != reference.shape for row in rows):
                    raise ValueError('Selected-source shape mismatch for {}/{}'.format(group, key))
                grp.create_dataset(key, data=np.asarray(rows))
                grp.create_dataset(key + '_available', data=available)
                grp[key].attrs['missing_policy'] = 'check the per-source {}_available flag'.format(key)
            _write_common_band_attributes(grp, group, sources)

        # Write parameters
        desi_dtype = np.dtype([('z_best', 'f4'), ('str_mass', 'f4')])
        desi_data = np.array([(s['z_best'], s['str_mass']) for s in sources], dtype=desi_dtype)
        desi = f.create_dataset('parameters_desi', data=desi_data)
        desi.attrs['name'] = list(desi_dtype.names)
        desi.attrs['axis_order'] = 'source'
        
        grism_dtype = np.dtype([
            ('gu_snr_mean', 'f4'),
            ('gv_snr_mean', 'f4'),
            ('gi_snr_mean', 'f4')
        ])
        grism_data = np.array([(s['gu_snr'], s['gv_snr'], s['gi_snr']) for s in sources], dtype=grism_dtype)
        grism = f.create_dataset('parameters_grism', data=grism_data)
        grism.attrs['name'] = list(grism_dtype.names)
        grism.attrs['axis_order'] = 'source'
        if any('data_mask' in source for source in sources):
            f.create_dataset('data_mask', data=np.asarray([source.get('data_mask', -1) for source in sources]))
            f.create_dataset('data_mask_available', data=np.asarray(['data_mask' in source for source in sources]))
        for key in _ROW_LINKS:
            if any(key in source for source in sources):
                f[key] = np.asarray([source.get(key, -1) for source in sources])
                f[key + '_available'] = np.asarray([key in source for source in sources])
        provenance = f.create_group('provenance')
        provenance.create_dataset('source_file', data=np.asarray(
            [source.get('source_file', '') for source in sources], dtype=h5py.string_dtype('utf-8')))
        provenance.create_dataset('source_row', data=np.asarray(
            [source.get('source_row', -1) for source in sources], dtype=np.int64))
        provenance.attrs['source_row_convention'] = 'zero-based row in source_file; -1 means unknown'
        _write_parent_metadata(provenance, sources)

def main(input_files, output_file, target=1000, n_bins=10, seed=None):
    # Read and filter data
    input_files = list(input_files)
    sources, gu_w, gv_w, gi_w = read_hdf5_files(input_files)
    
    if not sources:
        print("No sources meet selection criteria.")
        return
    
    # Stratified sampling
    selected = stratified_sampling(sources, target=target, n_bins=n_bins, seed=seed)
    
    # Write output
    write_output(output_file, selected, gu_w, gv_w, gi_w, seed=seed,
                 input_files=input_files, target=target, n_bins=n_bins)
    print(f"Successfully wrote {len(selected)} sources to {output_file}")


def _attrs(obj):
    return {name: value.copy() if isinstance(value, np.ndarray) else value
            for name, value in obj.attrs.items()}


def _spectral_fields(group, count, pixels, path):
    keys = []
    for key in group:
        if not key.startswith(_ROW_PREFIXES) or key.endswith('_available'):
            continue
        value = group[key]
        if not isinstance(value, h5py.Dataset):
            continue
        shape = ((count, pixels - 1) if key.startswith('covariance_lag1') else
                 (count,) if key.startswith('sky_rate_e_s_pixel') else (count, pixels))
        if value.shape != shape:
            raise ValueError('{}/{} must have shape {}'.format(path, value.name, shape))
        if key not in ('flux_ujy', 'flux_ujy_with_noise'):
            keys.append(key)
    for key in ('flux_ujy', 'flux_ujy_with_noise'):
        if key not in group:
            raise ValueError('Required spectrum missing: ' + key)
    return keys


def _parent_metadata(product, legacy):
    result = dict(root_attributes=_attrs(product), generation_attributes=_attrs(legacy),
                  id_dtype=legacy['ID'].dtype,
                  bands={}, provenance_attributes={})
    for band in _BANDS:
        group = legacy[band]
        result['bands'][band] = dict(attributes=_attrs(group), datasets={
            key: _attrs(group[key]) for key in group if isinstance(group[key], h5py.Dataset)},
            shared={key: group[key][:] for key in _SHARED_FIELDS if key in group})
    seen = set()
    def visit(obj, prefix=''):
        result['provenance_attributes'][prefix] = _attrs(obj)
        if not isinstance(obj, h5py.Group):
            return
        identity = h5py.h5o.get_info(obj.id).addr
        if identity in seen:
            return
        seen.add(identity)
        for key in obj:
            visit(obj[key], prefix + '/' + key if prefix else key)
    visit(product.provenance, 'product')
    if 'provenance' in legacy:
        visit(legacy['provenance'], 'legacy')
    return result


def _copy_attrs(target, values):
    for key, value in values.items():
        target.attrs[key] = value


def _write_common_band_attributes(target, band, sources):
    """Keep directly usable units/covariance model when parents agree.

    Conflicting or missing definitions stay in the per-parent snapshots; no
    arbitrary parent's calibration is silently promoted to a common definition.
    """
    metadata = [s.get('source_metadata', {}).get('bands', {}).get(band, {}) for s in sources]
    def common(obj, mappings):
        if not mappings:
            return
        keys = set(mappings[0]).intersection(*(set(value) for value in mappings[1:]))
        for key in keys:
            first = mappings[0][key]
            if all(np.array_equal(first, values[key]) for values in mappings[1:]):
                obj.attrs[key] = first
    common(target, [value.get('attributes', {}) for value in metadata])
    for key in target:
        if key.endswith('_available'):
            continue
        definitions = [value.get('datasets', {}).get(key, {}) for value in metadata
            if key in value.get('datasets', {})]
        common(target[key], definitions)


def _write_parent_metadata(provenance, sources):
    parents = provenance.create_group('parents')
    lookup, indices = {}, []
    for source in sources:
        path = source.get('source_file', '')
        if path not in lookup:
            lookup[path] = len(lookup)
            target = parents.create_group('parent_{:06d}'.format(lookup[path]))
            target.attrs['source_file'] = path
            metadata = source.get('source_metadata', {})
            if 'id_dtype' in metadata:
                target.attrs['source_id_numpy_dtype'] = str(metadata['id_dtype'])
                string = h5py.check_string_dtype(metadata['id_dtype'])
                if string is not None:
                    target.attrs['source_id_string_encoding'] = string.encoding
            for key in ('root_attributes', 'generation_attributes'):
                _copy_attrs(target.create_group(key), metadata.get(key, {}))
            for band, values in metadata.get('bands', {}).items():
                group = target.require_group('bands/' + band)
                _copy_attrs(group, values['attributes'])
                for key, attrs in values['datasets'].items():
                    _copy_attrs(group.require_group('dataset_attributes/' + key), attrs)
                for key, value in values['shared'].items():
                    group.require_group('shared').create_dataset(key, data=value)
            for key, attrs in metadata.get('provenance_attributes', {}).items():
                _copy_attrs(target.require_group('provenance_attributes' + ('/' + key if key else '')), attrs)
        indices.append(lookup[path])
    provenance['parent_index'] = np.asarray(indices, dtype=np.int64)
    provenance.attrs['parent_metadata_scope'] = (
        'recorded root/generation/group/dataset attributes and shared response/bin geometry; '
        'not a clone of unselected parent science or unknown extension datasets')
    prior = provenance.create_group('upstream_lineage')
    for key in ('source_file', 'source_row'):
        if any(key in source.get('upstream_lineage', {}) for source in sources):
            values = [source.get('upstream_lineage', {}).get(key, '' if key == 'source_file' else -1) for source in sources]
            prior.create_dataset(key, data=np.asarray(values, dtype=h5py.string_dtype('utf-8') if key == 'source_file' else np.int64))
            prior[key + '_available'] = np.asarray([key in source.get('upstream_lineage', {}) for source in sources])


def cli_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_files', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--target', type=int, default=1000)
    parser.add_argument('--n-bins', type=int, default=10)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args(argv)
    try:
        main(args.input_files, args.output, args.target, args.n_bins, args.seed)
        return 0
    except (OSError, ValueError, KeyError, ImportError) as error:
        print('CESS sampling: {}'.format(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(cli_main())
