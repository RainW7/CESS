"""Concatenate one simulation catalogue's products without recalculating spectra.

Only the explicit 2.5/3.0 simulation contracts are supported. Shared detector
geometry is checked once per input; source arrays are streamed in bounded row
blocks; sparse emission-line source references become rows of the merged ID.
Original generation attributes remain generation attributes, while merge_history
records this storage operation and every input product's immutable identity.
"""
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
import json
import os

import h5py
import numpy as np

from cess._version import __version__
from cess.contracts import MGE_MORPHOLOGY_FIELDS
from .atomic import atomic_hdf5
from .products import (BANDS, ProductView, _check_members, _copy_attrs, _kind_mode,
                       _object_id, _sha256, _text, structure_product)

_RANGE_ATTRIBUTES = {'source_start', 'source_stop', 'output_product'}
_GENERATION_ATTRIBUTES = ('schema_version', 'pipeline_stage', 'input_sha256',
    'input_product', 'config_json', 'switches_json', 'master_seed',
    'random_stream_version', 'code_sha256_json', 'resource_sha256_json',
    'software_version', 'cess_repair_revision')
_ROOT_SOURCE = {'ID', 'input_row_indices', 'data_mask', 'input_quality_flags',
    'parameters_desi', 'parameters_grism', 'parameters_phot', 'zero_order_phot'}
_PHOT_FIELDS = {'model_flux_ujy', 'model_mag', 'observed_flux_ujy', 'flux_error_ujy',
    'statistical_flux_error_ujy', 'systematic_flux_error_ujy', 'total_flux_error_ujy',
    'model_mag_error', 'observed_mag', 'observed_mag_error', 'snr', 'expected_snr',
    'source_counts_e', 'observed_counts_e', 'sky_counts_e', 'dark_counts_e',
    'read_variance_e2', 'variance_e2', 'coverage_fraction', 'detection_snr',
    'total_exposure_s', 'aperture_fraction', 'n_pixels', 'systematic_mag',
    'response_efficiency', 'valid', 'detected', 'coverage_complete', 'flag', 'reason'}
_BASE_SOURCE = {'flux_elec', 'flux_ujy', 'flux_ujy_with_noise', 'ferr', 'snr',
    'spec_mask', 'snr_mask', 'response_flux_ujy', 'source_electron_rate_pre_morph',
    'sky_rate_e_s_pixel', 'flux_ujy_model', 'detector_model_mask',
    'calibration_support_mask', 'covariance_lag1'}
_PAIRED_SOURCE = {'flux_elec', 'flux_ujy', 'flux_ujy_with_noise', 'ferr', 'snr',
    'flux_ujy_model', 'detector_model_mask', 'calibration_support_mask', 'covariance_lag1'}
_SHARED = {'wave', 'wave_edges', 'pixel_width', 'response_at_pixel_center'}


def _same_dtype(first, second):
    return first == second and first.metadata == second.metadata


def _attributes_equal(first, second, ignored=()):
    names = set(first.attrs) - set(ignored)
    if names != set(second.attrs) - set(ignored):
        return False
    for name in names:
        a, b = first.attrs.get_id(name), second.attrs.get_id(name)
        if not _same_dtype(a.dtype, b.dtype) or a.shape != b.shape:
            return False
        try:
            np.testing.assert_array_equal(first.attrs[name], second.attrs[name])
        except AssertionError:
            return False
    return True


def _signature(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _stable_hash(path):
    before = _signature(path)
    digest = _sha256(path)
    if _signature(path) != before:
        raise ValueError('Source changed while computing its SHA256: ' + str(path))
    return before, digest


def _unchanged(entry):
    path = Path(entry['path'])
    if _signature(path) != entry['signature'] or _sha256(path) != entry['sha256']:
        raise ValueError('Source changed during merge: ' + str(path))
    if _signature(path) != entry['signature']:
        raise ValueError('Source changed during merge: ' + str(path))


def _group_allowed(path, mode):
    parts = path.split('/')
    if len(parts) == 1:
        return path in {'spectral_extraction', 'mge_morphology', 'photometry',
                        'zero_order', *BANDS}
    if parts[0] in ('photometry', 'zero_order'):
        bands = ('nuv', 'u', 'g', 'r', 'i', 'z', 'y') if parts[0] == 'photometry' else BANDS
        return len(parts) == 2 and parts[1] in bands
    if parts[0] not in ('GV', 'GI'):
        return False
    variants = ('', '_wce') if mode == 'paired-wce' else ('',)
    if len(parts) == 2:
        return parts[1] in {stem + suffix for stem in ('el_diagnostics', 'line_measurements')
                            for suffix in variants}
    return (len(parts) == 3 and parts[1] in {'el_diagnostics' + suffix for suffix in variants}
            and parts[2] in ('candidates', 'missed', 'false_positive'))


def _category(path, mode):
    parts = path.split('/')
    if len(parts) == 1 and path in _ROOT_SOURCE:
        return 'source'
    if len(parts) == 2 and parts[0] == 'spectral_extraction' and parts[1] in (
            'source_fraction', 'spatial_pixels', 'profile_height_sum', 'valid'):
        return 'source'
    if len(parts) == 2 and parts[0] == 'mge_morphology' and parts[1] in MGE_MORPHOLOGY_FIELDS:
        return 'source'
    if len(parts) == 3 and parts[0] in ('photometry', 'zero_order') and parts[2] in _PHOT_FIELDS:
        return 'source'
    if len(parts) < 2 or parts[0] not in BANDS:
        raise ValueError('Unknown simulation dataset; source axis cannot be guessed: ' + path)
    name = parts[1]
    if len(parts) == 2:
        shared = _SHARED | ({'response_at_pixel_center_0th'} if mode == 'standard' else set())
        if name in shared:
            return 'shared'
        source = _BASE_SOURCE | ({name + '_0th' for name in _BASE_SOURCE - {'flux_ujy'}}
            if mode == 'standard' else {name + '_wce' for name in _PAIRED_SOURCE})
        if name in source:
            return 'source'
        if parts[0] in ('GV', 'GI'):
            for stem in ('intri_el_', 'detect_el_'):
                fields = {'id', 'wave', 'idx', 'elnumber'} | ({'snr'} if stem.startswith('detect') else set())
                if name in {stem + field for field in fields}:
                    return 'row_id' if name == stem + 'id' else 'sparse'
    variants = ('', '_wce') if mode == 'paired-wce' else ('',)
    if parts[0] in ('GV', 'GI'):
        if len(parts) == 3:
            if name in {'el_diagnostics' + suffix for suffix in variants} and parts[2] == 'pairs':
                return 'compound_id'
            if name in {'line_measurements' + suffix for suffix in variants} and parts[2] in ('candidates', 'truth'):
                return 'compound_source_row'
        if len(parts) == 4 and name in {'el_diagnostics' + suffix for suffix in variants}:
            counts = {'candidates': 'candidate_el_number', 'missed': 'missed_el_number',
                      'false_positive': 'false_el_number'}
            if parts[2] in counts and parts[3] in ('id', 'wave', 'idx', counts[parts[2]]):
                return 'row_id' if parts[3] == 'id' else 'sparse'
    raise ValueError('Unknown simulation dataset; source axis cannot be guessed: ' + path)


def _tree(handle, mode, count):
    groups, leaves, seen = {}, {}, set()
    def visit(group, prefix=''):
        for name in group:
            path, value = prefix + name, group[name]
            if isinstance(value, h5py.Dataset):
                identity = _object_id(value)
                if identity in seen:
                    raise ValueError('Aliased datasets are unsupported in merge: ' + path)
                seen.add(identity)
                category = _category(path, mode)
                if value.shape is None or value.ndim < 1:
                    raise ValueError('Expected nonscalar simulation dataset: ' + path)
                if category == 'source' and value.shape[0] != count:
                    raise ValueError('Unaligned source dataset: ' + path)
                if category not in ('source', 'shared') and value.ndim != 1:
                    raise ValueError('Expected one-dimensional sparse line dataset: ' + path)
                if value.scaleoffset is not None:
                    raise ValueError('Lossy scale-offset storage is unsupported in merge: ' + path)
                leaves[path] = category
            elif isinstance(value, (h5py.Group, Mapping)):
                if not _group_allowed(path, mode):
                    raise ValueError('Unknown simulation group: ' + path)
                groups[path] = value
                visit(value, path + '/')
            else:
                raise ValueError('Unknown simulation object: ' + path)
    visit(handle)
    return groups, leaves


def _arrays_equal(first, second, chunk_rows):
    if first.shape != second.shape or not _same_dtype(first.dtype, second.dtype):
        return False
    try:
        for start in range(0, len(first), chunk_rows):
            np.testing.assert_array_equal(first[start:start + chunk_rows], second[start:start + chunk_rows])
    except AssertionError:
        return False
    return True


def _integer_attribute(handle, name):
    value = handle.attrs.get(name)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError('Missing/noninteger source interval attribute: ' + name)
    return int(value)


def _generation(handle):
    for name in _GENERATION_ATTRIBUTES:
        if name not in handle.attrs:
            raise ValueError('Missing simulation generation attribute: ' + name)
    if _text(handle.attrs['pipeline_stage']) != 'simulation':
        raise ValueError('Only simulation products can be merged; EL-only products are unsupported')
    digest = _text(handle.attrs['input_sha256'])
    if len(digest) != 64 or any(letter not in '0123456789abcdef' for letter in digest):
        raise ValueError('A recorded original input SHA256 is required')
    try:
        config = json.loads(_text(handle.attrs['config_json']))
        options = json.loads(_text(handle.attrs['switches_json']))
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError('Invalid simulation generation settings') from error
    required = {'apply_morphology_effect', 'apply_wavelength_calibration_error',
                'apply_photometric', 'apply_el_detect'}
    if not isinstance(config, dict) or not isinstance(options, dict) or set(options) != required:
        raise ValueError('Invalid simulation configuration/switches')
    if not all(isinstance(options[key], bool) for key in options):
        raise ValueError('Simulation switches must be boolean')
    return config, options


def _prepare(sources, destination, expected_input, chunk_rows):
    if isinstance(chunk_rows, (bool, np.bool_)) or not isinstance(chunk_rows, (int, np.integer)) or chunk_rows < 1:
        raise ValueError('chunk_rows must be a positive integer')
    if isinstance(sources, (str, os.PathLike)):
        sources = [sources]
    sources = list(sources)
    if not sources:
        raise ValueError('At least one input simulation product is required')
    paths = [Path(source).expanduser().resolve() for source in sources]
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate input product path')
    destination = Path(destination).expanduser().absolute()
    if os.path.lexists(destination):
        raise FileExistsError('Output already exists: ' + str(destination))
    if destination.resolve() in paths:
        raise ValueError('Destination must differ from every source')
    entries = []
    expected = None
    with ExitStack() as stack:
        raw = None
        if expected_input is not None:
            path = Path(expected_input).expanduser().resolve()
            signature, digest = _stable_hash(path)
            expected = dict(path=str(path), signature=signature, sha256=digest)
            raw = stack.enter_context(h5py.File(path, 'r'))
            if 'ID' not in raw or not isinstance(raw['ID'], h5py.Dataset) or raw['ID'].ndim != 1:
                raise ValueError('Expected original input requires one-dimensional ID')
        reference_raw = stack.enter_context(h5py.File(paths[0], 'r'))
        reference_view = ProductView(reference_raw)
        schema = reference_view.schema_version
        if schema not in ('2.5', '3.0'):
            raise ValueError('Merge requires explicit schema 2.5 or 3.0 simulation products')
        reference = reference_view.legacy_view()
        kind, mode = _kind_mode(reference)
        if kind != 'simulation':
            raise ValueError('EL-only products cannot be merged')
        config, options = _generation(reference)
        reference_count = len(reference['ID'])
        ref_groups, leaves = _tree(reference, mode, reference_count)
        for path in paths:
            signature, digest = _stable_hash(path)
            with h5py.File(path, 'r') as handle:
                _check_members(handle)
                view = ProductView(handle)
                if view.schema_version != schema:
                    raise ValueError('Mixed product schemas cannot be merged')
                source = view.legacy_view()
                if _kind_mode(source) != (kind, mode):
                    raise ValueError('Mixed product modes/kinds cannot be merged')
                _generation(source)
                if not _attributes_equal(reference, source, _RANGE_ATTRIBUTES):
                    raise ValueError('Simulation generation attributes differ: ' + str(path))
                count = len(source['ID'])
                if count < 1:
                    raise ValueError('Empty simulation chunks cannot define a merge interval')
                start, stop = (_integer_attribute(source, name) for name in ('source_start', 'source_stop'))
                rows = source['input_row_indices'][:]
                if (start < 0 or stop - start != count or rows.dtype.kind not in 'iu' or
                        rows.shape != (count,) or not np.array_equal(rows, np.arange(start, stop))):
                    raise ValueError('Source rows must match one contiguous original input interval: ' + str(path))
                groups, actual_leaves = _tree(source, mode, count)
                if set(groups) != set(ref_groups) or actual_leaves != leaves:
                    raise ValueError('Simulation group/dataset sets differ: ' + str(path))
                for name in groups:
                    if not _attributes_equal(ref_groups[name], groups[name]):
                        raise ValueError('Group attributes differ: ' + name)
                for name, category in leaves.items():
                    before, after = reference[name], source[name]
                    if (not _same_dtype(before.dtype, after.dtype) or
                            before.shape[1:] != after.shape[1:] or
                            not _attributes_equal(before, after)):
                        raise ValueError('Dataset dtype/shape/attributes differ: ' + name)
                    if category == 'shared' and not _arrays_equal(before, after, chunk_rows):
                        raise ValueError('Shared wavelength/response data differ: ' + name)
                from .manifest import _structure
                _structure(source, count, mode, options, config=config)
                if raw is not None:
                    if stop > len(raw['ID']) or not _same_dtype(source['ID'].dtype, raw['ID'].dtype):
                        raise ValueError('Original input ID shape/dtype differs from product')
                    if _text(source.attrs['input_sha256']) != expected['sha256']:
                        raise ValueError('Original input SHA256 differs from product generation')
                    for first in range(0, count, chunk_rows):
                        last = min(first + chunk_rows, count)
                        if not np.array_equal(source['ID'][first:last], raw['ID'][start + first:start + last]):
                            raise ValueError('Original input ID/order differs from product')
                entries.append(dict(path=str(path), signature=signature, sha256=digest,
                    source_start=start, source_stop=stop, rows=count,
                    shapes={name: source[name].shape for name in leaves}))
            if _signature(path) != signature:
                raise ValueError('Source changed during merge planning: ' + str(path))
        entries.sort(key=lambda entry: entry['source_start'])
        offset = 0
        for index, entry in enumerate(entries):
            if index and entry['source_start'] != entries[index - 1]['source_stop']:
                raise ValueError('Merge intervals overlap, repeat, or contain a gap')
            entry['offset'] = offset
            offset += entry['rows']
        if expected is not None and _signature(Path(expected['path'])) != expected['signature']:
            raise ValueError('Original input changed during merge planning')
        plan = dict(operation='merge_simulation_chunks', output=str(destination),
            schema_version=schema, mode=mode, rows=offset,
            source_start=entries[0]['source_start'], source_stop=entries[-1]['source_stop'],
            generation_version=_text(reference.attrs['software_version']), tool_version=__version__,
            source_products=[{key: value for key, value in entry.items()
                if key not in ('signature', 'shapes')} for entry in entries],
            expected_input=None if expected is None else expected['path'],
            original_input_rows=None if raw is None else len(raw['ID']),
            covers_entire_input=(None if raw is None else
                entries[0]['source_start'] == 0 and entries[-1]['source_stop'] == len(raw['ID'])))
    return plan, entries, leaves, expected, config, options


def plan_merge(sources, destination, *, expected_input=None, chunk_rows=128):
    """Return a JSON-serializable validated merge plan without creating output."""
    return _prepare(sources, destination, expected_input, chunk_rows)[0]


def _copy_values(source, target, offset, category, source_offset, chunk_rows):
    numeric_vlen = isinstance(h5py.check_dtype(vlen=source.dtype), np.dtype)
    field = {'compound_id': 'id', 'compound_source_row': 'source_row'}.get(category)
    for first in range(0, len(source), chunk_rows):
        last = min(first + chunk_rows, len(source))
        if numeric_vlen:
            for row in range(first, last):
                target[offset + row] = source[row]
            continue
        values = source[first:last]
        if category == 'row_id':
            values = values + source_offset
        elif field is not None:
            values[field] += source_offset
        target[offset + first:offset + last] = values


def merge_products(sources, destination, *, expected_input=None, chunk_rows=128, dry_run=False):
    """Merge contiguous chunks of one catalogue, returning Path or dry-run plan.

    Inputs are opened read-only; no existing destination is replaced. Supplying
    expected_input also verifies its SHA256 and ID/order, and the returned plan
    says explicitly whether the selection covers that entire original input.
    Sparse line id/source_row fields refer to the merged local ID row, while
    input_row_indices retains the original catalogue row. No SED, noise draw,
    wavelength, measurement, or generation-version metadata is recomputed.
    """
    plan, entries, leaves, expected, config, options = _prepare(
        sources, destination, expected_input, chunk_rows)
    if dry_run:
        return plan
    destination = Path(plan['output'])
    with atomic_hdf5(destination) as output:
        with h5py.File(entries[0]['path'], 'r') as first_raw:
            first = ProductView(first_raw).legacy_view()
            for name in first.attrs:
                if name not in _RANGE_ATTRIBUTES:
                    attr = first.attrs.get_id(name)
                    output.attrs.create(name, first.attrs[name], dtype=attr.dtype, shape=attr.shape)
            output.attrs['source_start'] = plan['source_start']
            output.attrs['source_stop'] = plan['source_stop']
            groups, _ = _tree(first, plan['mode'], entries[0]['rows'])
            for name in sorted(groups, key=lambda name: (name.count('/'), name)):
                _copy_attrs(groups[name], output.create_group(name))
            for name, category in leaves.items():
                source = first[name]
                count = (source.shape[0] if category == 'shared' else
                    sum(entry['shapes'][name][0] for entry in entries))
                target = output.create_dataset(name, shape=(count,) + source.shape[1:], dtype=source.dtype)
                _copy_attrs(source, target)
                if category == 'shared':
                    _copy_values(source, target, 0, category, 0, chunk_rows)
        history = output.create_group('merge_history')
        history.attrs['operation'] = plan['operation']
        history.attrs['tool_version'] = __version__
        history.attrs['generation_version'] = plan['generation_version']
        history.attrs['source_reference_remapping'] = (
            'EL id/source_row += chunk offset into merged ID; original catalogue rows and peak/pixel indices unchanged')
        history.attrs['source_product_count'] = len(entries)
        if expected is not None:
            history.attrs['expected_input'] = expected['path']
            history.attrs['expected_input_sha256'] = expected['sha256']
            history.attrs['covers_entire_input'] = plan['covers_entire_input']
        offsets = {name: 0 for name, category in leaves.items() if category != 'shared'}
        for index, entry in enumerate(entries):
            _unchanged(entry)
            with h5py.File(entry['path'], 'r') as raw:
                source = ProductView(raw).legacy_view()
                record = history.create_group('{:06d}'.format(index))
                for name in ('path', 'sha256', 'source_start', 'source_stop', 'rows', 'offset'):
                    record.attrs[name] = entry[name]
                record.attrs['source_stat_json'] = json.dumps(entry['signature'])
                _copy_attrs(raw, record.create_group('original_root_attributes'))
                _copy_attrs(source, record.create_group('original_generation_attributes'))
                if plan['schema_version'] == '3.0':
                    _copy_attrs(raw['provenance/conversion'], record.create_group('original_layout_conversion_attributes'))
                for name, category in leaves.items():
                    if category == 'shared':
                        continue
                    _copy_values(source[name], output[name], offsets[name], category, entry['offset'], chunk_rows)
                    offsets[name] += entry['shapes'][name][0]
            _unchanged(entry)
        from .manifest import _structure
        _structure(output, plan['rows'], plan['mode'], options, config=config)
        if plan['schema_version'] == '3.0':
            structure_product(output)
            _structure(ProductView(output).legacy_view(), plan['rows'], plan['mode'], options, config=config)
        for entry in entries:
            if _signature(Path(entry['path'])) != entry['signature']:
                raise ValueError('Source changed before merge publication: ' + entry['path'])
        if expected is not None:
            _unchanged(expected)
    return destination.resolve()
