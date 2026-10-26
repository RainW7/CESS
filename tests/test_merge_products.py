"""Lossless chunk assembly, especially sparse source references and provenance."""
import hashlib
import importlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest

from cess.config.loader import load_config
from cess.io.merge import merge_products, plan_merge
from cess.io.products import ProductView, open_product, validate_structure
from test_pipeline_smoke import make_input


def _copy_attrs(source, target):
    for name in source.attrs:
        attr = source.attrs.get_id(name)
        target.attrs.create(name, source.attrs[name], dtype=attr.dtype, shape=attr.shape)


def _flatten(source, destination):
    """Independent physical-copy helper for the legacy fixtures."""
    with open_product(source) as product, h5py.File(destination, 'w') as target:
        view = product.legacy_view()
        _copy_attrs(view, target)
        def copy_group(original, copied):
            for name in original:
                value = original[name]
                if isinstance(value, h5py.Dataset):
                    value.file.copy(value, copied, name=name)
                else:
                    group = copied.create_group(name)
                    _copy_attrs(value, group)
                    copy_group(value, group)
        copy_group(view, target)


@pytest.fixture(scope='module')
def catalogues(tmp_path_factory):
    root = tmp_path_factory.mktemp('merge-products')
    source = root / 'input.hdf5'
    make_input(source, count=4, invalid=(1,))
    with h5py.File(source, 'r+') as handle:
        # Duplicated catalogue ID values demonstrate row-based, not ID-based,
        # ordering. Bright peaks exercise real noisy/truth/pair references.
        handle['ID'][:] = np.array([b'duplicate', b'invalid', b'last', b'duplicate'], dtype='S20')
        flux = handle['best_fit/spec_synthetic'][:]
        handle['best_fit/spec_synthetic'][:] = 1. + (flux - 1.) * 20.
    config = load_config()
    config['line_measurement'] = dict(line_width_pixels=dict(GV=10, GI=10),
        sideband_width_pixels=5, sideband_gap_pixels=3)
    options = dict(morph=True, wave_cal=True, photo=True, el_detect=True)
    result = dict(source=source, original_sha=hashlib.sha256(source.read_bytes()).hexdigest())
    for mode, module in [('standard', 'standard'), ('paired-wce', 'paired')]:
        driver = importlib.import_module('cess.pipeline.' + module)
        chunks = []
        for start, stop in [(0, 3), (3, 4)]:
            chunks.append(Path(driver.simulate_file(str(root), source.name, config=config,
                options=options, seed=1214, output_dir=root / mode,
                source_start=start, source_stop=stop)))
        legacy = []
        for number, path in enumerate(chunks):
            destination = root / ('legacy-' + mode + '-' + str(number) + '.h5')
            _flatten(path, destination)
            legacy.append(destination)
        result[mode] = {'structured': chunks, 'legacy': legacy}
    return result


def _datasets(handle):
    result = {}
    def visit(group, prefix=''):
        for key in group:
            value, name = group[key], prefix + key
            if isinstance(value, h5py.Dataset):
                result[name] = value
            else:
                visit(value, name + '/')
    visit(handle)
    return result


def _reference_field(name):
    if name.endswith(('/intri_el_id', '/detect_el_id')) or '/el_diagnostics' in name and name.endswith('/id'):
        return ''
    if '/el_diagnostics' in name and name.endswith('/pairs'):
        return 'id'
    if '/line_measurements' in name and name.endswith(('/candidates', '/truth')):
        return 'source_row'
    return None


def _source_aligned(name):
    if '/' not in name:
        return True
    if name.startswith(('spectral_extraction/', 'mge_morphology/', 'photometry/', 'zero_order/')):
        return True
    return ('/el_diagnostics' not in name and '/line_measurements' not in name
            and name.split('/')[1] not in ('wave', 'wave_edges', 'pixel_width',
                'response_at_pixel_center', 'response_at_pixel_center_0th')
            and not name.split('/')[1].startswith(('intri_el_', 'detect_el_')))


@pytest.mark.parametrize('mode', ['standard', 'paired-wce'])
@pytest.mark.parametrize('layout', ['legacy', 'structured'])
def test_merge_preserves_all_science_values_and_remaps_only_source_references(tmp_path, catalogues, mode, layout):
    chunks = catalogues[mode][layout]
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in chunks]
    output = tmp_path / 'complete.hdf5'
    result = merge_products(chunks[::-1], output, expected_input=catalogues['source'], chunk_rows=1)
    assert result == output
    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in chunks] == before
    assert hashlib.sha256(catalogues['source'].read_bytes()).hexdigest() == catalogues['original_sha']
    with open_product(output) as merged:
        assert merged.schema_version == ('3.0' if layout == 'structured' else '2.5')
        if layout == 'structured':
            validate_structure(merged.handle)
            assert 'merge_history' in merged.handle['provenance/extensions/root']
        target = merged.legacy_view()
        np.testing.assert_array_equal(target['input_row_indices'][:], [0, 1, 2, 3])
        np.testing.assert_array_equal(target['data_mask'][:], [1, 2, 1, 1])
        history = target['merge_history']
        assert history.attrs['operation'] == 'merge_simulation_chunks'
        assert history.attrs['covers_entire_input']
        assert history.attrs['source_product_count'] == 2
        assert target.attrs['source_start'] == 0 and target.attrs['source_stop'] == 4
        cursors = {}
        checked_references = 0
        for number, path in enumerate(chunks):
            with open_product(path) as original:
                source = original.legacy_view()
                assert target.attrs['software_version'] == source.attrs['software_version']
                assert target.attrs['code_sha256_json'] == source.attrs['code_sha256_json']
                recorded = history['{:06d}/original_generation_attributes'.format(number)]
                assert set(recorded.attrs) == set(source.attrs)
                assert history['{:06d}'.format(number)].attrs['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
                for key in source.attrs:
                    np.testing.assert_array_equal(recorded.attrs[key], source.attrs[key])
                    assert recorded.attrs.get_id(key).dtype == source.attrs.get_id(key).dtype
                for name, data in _datasets(source).items():
                    actual = target[name]
                    assert actual.dtype == data.dtype and actual.dtype.metadata == data.dtype.metadata
                    assert set(actual.attrs) == set(data.attrs)
                    for attr in data.attrs:
                        np.testing.assert_array_equal(actual.attrs[attr], data.attrs[attr])
                        assert actual.attrs.get_id(attr).dtype == data.attrs.get_id(attr).dtype
                    shared = not _source_aligned(name) and _reference_field(name) is None and not (
                        '/el_diagnostics' in name or '/line_measurements' in name
                        or name.split('/')[-1].startswith(('intri_el_', 'detect_el_')))
                    if shared:
                        np.testing.assert_array_equal(actual[:], data[:])
                        continue
                    cursor = cursors.get(name, 0)
                    values = data[:]
                    field = _reference_field(name)
                    offset = 0 if number == 0 else 3
                    if field == '':
                        values = values + offset
                        checked_references += len(values)
                    elif field is not None:
                        values[field] += offset
                        checked_references += len(values)
                    if isinstance(h5py.check_dtype(vlen=data.dtype), np.dtype):
                        for row in range(len(data)):
                            np.testing.assert_array_equal(actual[cursor + row], data[row])
                    elif values.dtype.names:
                        for field_name in values.dtype.names:
                            np.testing.assert_array_equal(
                                actual[cursor:cursor + len(data)][field_name], values[field_name])
                    else:
                        np.testing.assert_array_equal(actual[cursor:cursor + len(data)], values)
                    cursors[name] = cursor + len(data)
        assert checked_references > 0
        assert np.isnan(target['GU/flux_ujy_with_noise'][1]).all()


def test_dry_run_is_serializable_read_only_and_reports_partial_coverage(tmp_path, catalogues):
    output = tmp_path / 'not-created' / 'complete.h5'
    chunks = catalogues['standard']['structured']
    plan = merge_products(chunks[::-1], output, expected_input=catalogues['source'], dry_run=True)
    assert json.loads(json.dumps(plan)) == plan
    assert plan['rows'] == 4 and plan['covers_entire_input'] is True
    assert plan['source_products'][0]['source_start'] == 0
    assert not output.parent.exists()
    partial = plan_merge(chunks[1:], output, expected_input=catalogues['source'])
    assert partial['rows'] == 1 and partial['source_start'] == 3
    assert partial['covers_entire_input'] is False


def test_nonzero_original_start_still_maps_emission_references_to_local_merged_rows(tmp_path, catalogues):
    path = catalogues['standard']['structured'][1]
    output = merge_products([path], tmp_path / 'partial.h5', expected_input=catalogues['source'])
    with open_product(output) as product:
        source = product.legacy_view()
        np.testing.assert_array_equal(source['input_row_indices'][:], [3])
        checked = 0
        for name, data in _datasets(source).items():
            if name.startswith('merge_history/'):
                continue
            field = _reference_field(name)
            if field is not None:
                rows = data[:] if field == '' else data[:][field]
                np.testing.assert_array_equal(rows, np.zeros_like(rows))
                checked += len(rows)
        assert checked


def test_original_scientific_generation_version_is_not_the_merge_tool_version(tmp_path, catalogues):
    from cess import __version__
    chunks = []
    for path in catalogues['standard']['legacy']:
        target = _edited_copy(path, tmp_path)
        with h5py.File(target, 'r+') as handle:
            handle.attrs['software_version'] = '0.9.0.dev9'
            handle.attrs['code_sha256_json'] = json.dumps({'historical/source.py': 'a' * 64})
        chunks.append(target)
    output = merge_products(chunks, tmp_path / 'old-generation.h5')
    with open_product(output) as product:
        assert product.provenance.attrs['software_version'] == '0.9.0.dev9'
        assert product.legacy_view()['merge_history'].attrs['tool_version'] == __version__
        assert json.loads(product.provenance.attrs['code_sha256_json']) == {'historical/source.py': 'a' * 64}


def test_empty_line_tables_preserve_vlen_and_compound_dtypes(tmp_path, catalogues):
    chunks = []
    for path in catalogues['paired-wce']['legacy']:
        target = _edited_copy(path, tmp_path)
        with h5py.File(target, 'r+') as handle:
            paths = [name for name in _datasets(handle) if '/el_diagnostics' in name
                     or '/line_measurements' in name
                     or name.split('/')[-1].startswith(('intri_el_', 'detect_el_'))]
            for name in paths:
                original = handle[name]
                attributes = [(key, original.attrs[key], original.attrs.get_id(key).dtype,
                               original.attrs.get_id(key).shape) for key in original.attrs]
                dtype = original.dtype
                del handle[name]
                empty = handle.create_dataset(name, shape=(0,), dtype=dtype)
                for key, value, attribute_dtype, shape in attributes:
                    empty.attrs.create(key, value, dtype=attribute_dtype, shape=shape)
        chunks.append(target)
    output = merge_products(chunks, tmp_path / 'empty-lines.h5', chunk_rows=1)
    with open_product(output) as product:
        view = product.legacy_view()
        assert len(view['GI/detect_el_wave']) == 0
        assert h5py.check_dtype(vlen=view['GI/detect_el_wave'].dtype) == np.dtype('float64')
        assert view['GI/el_diagnostics_wce/pairs'].dtype.names[0] == 'id'
        assert view['GI/line_measurements_wce/candidates'].shape == (0,)


def _edited_copy(path, tmp_path):
    target = tmp_path / ('edited-' + path.name)
    shutil.copyfile(path, target)
    return target


@pytest.mark.parametrize('attribute', ['master_seed', 'input_sha256', 'config_json', 'code_sha256_json', 'software_version'])
def test_rejects_mixed_generation_without_output(tmp_path, catalogues, attribute):
    chunks = catalogues['standard']['legacy']
    edited = _edited_copy(chunks[1], tmp_path)
    with h5py.File(edited, 'r+') as handle:
        handle.attrs[attribute] = ('0' * 64 if attribute == 'input_sha256' else
            '{}' if attribute.endswith('_json') else 'different')
    output = tmp_path / 'bad.h5'
    with pytest.raises(ValueError):
        merge_products([chunks[0], edited], output)
    assert not output.exists() and not list(tmp_path.glob('.*.partial'))


@pytest.mark.parametrize('problem', ['duplicate_path', 'duplicate_interval', 'overlap', 'gap', 'mixed_schema', 'mixed_mode',
    'unknown_root', 'unknown_dataset', 'wrong_grid', 'wrong_dataset_attr', 'wrong_group_attr', 'softlink', 'vds', 'reference'])
def test_invalid_storage_or_intervals_are_rejected(tmp_path, catalogues, problem):
    chunks = catalogues['standard']['legacy']
    edited = _edited_copy(chunks[1], tmp_path)
    sources = [chunks[0], edited]
    if problem == 'duplicate_path':
        sources = [chunks[0], chunks[0]]
    elif problem == 'mixed_schema':
        sources[1] = catalogues['standard']['structured'][1]
    elif problem == 'mixed_mode':
        sources[1] = catalogues['paired-wce']['legacy'][1]
    else:
        with h5py.File(edited, 'r+') as handle:
            if problem in ('duplicate_interval', 'overlap', 'gap'):
                first = {'duplicate_interval': 0, 'overlap': 2, 'gap': 4}[problem]
                handle['input_row_indices'][:] = [first]
                handle.attrs['source_start'], handle.attrs['source_stop'] = first, first + 1
            elif problem == 'unknown_root':
                handle.create_group('future')
            elif problem == 'unknown_dataset':
                handle['GU/future_flux'] = np.ones((1, len(handle['GU/wave'])))
            elif problem == 'wrong_grid':
                handle['GU/wave'][0] += .1
            elif problem == 'wrong_dataset_attr':
                handle['GU/ferr'].attrs['unit'] = 'wrong'
            elif problem == 'wrong_group_attr':
                handle['GU'].attrs['spectral_quantity'] = 'wrong'
            elif problem == 'softlink':
                handle['alias'] = h5py.SoftLink('/ID')
            elif problem == 'vds':
                del handle['GU/flux_ujy']
                layout = h5py.VirtualLayout(shape=(1, 2), dtype='f8')
                layout[:] = h5py.VirtualSource(str(chunks[0]), 'GU/flux_ujy', shape=(3, 2))[0:1]
                handle['GU'].create_virtual_dataset('flux_ujy', layout)
            elif problem == 'reference':
                handle.attrs['bad_reference'] = handle['ID'].ref
    output = tmp_path / 'bad.h5'
    with pytest.raises(ValueError):
        merge_products(sources, output)
    assert not output.exists()


def test_expected_input_sha_and_id_validation_are_read_only(tmp_path, catalogues):
    raw = _edited_copy(catalogues['source'], tmp_path)
    chunks = catalogues['standard']['legacy']
    with h5py.File(raw, 'r+') as handle:
        handle['ID'][0] = b'changed'
    with pytest.raises(ValueError, match='SHA256'):
        merge_products(chunks, tmp_path / 'bad.h5', expected_input=raw)
    with h5py.File(raw, 'r') as handle:
        assert handle['ID'][0] == b'changed'


def test_no_clobber_and_validation_failure_leave_no_partial(tmp_path, catalogues, monkeypatch):
    chunks = catalogues['standard']['legacy']
    output = tmp_path / 'complete.h5'
    output.write_text('do not replace')
    with pytest.raises(FileExistsError):
        merge_products(chunks, output)
    assert output.read_text() == 'do not replace'
    from cess.io import merge
    original = merge._copy_values
    def fail_after_first_copy(*args, **kwargs):
        original(*args, **kwargs)
        raise ValueError('injected write failure')
    monkeypatch.setattr(merge, '_copy_values', fail_after_first_copy)
    missing = tmp_path / 'failed.h5'
    with pytest.raises(ValueError, match='injected write failure'):
        merge_products(chunks, missing)
    assert not missing.exists() and not list(tmp_path.glob('.*.partial'))


def test_source_drift_is_detected_before_publication(tmp_path, catalogues, monkeypatch):
    from cess.io import merge
    chunks = [_edited_copy(path, tmp_path) for path in catalogues['standard']['legacy']]
    original = merge._copy_values
    changed = False
    def alter_source(*args, **kwargs):
        nonlocal changed
        original(*args, **kwargs)
        if not changed:
            # Change a different source before it is copied. The signature and
            # SHA checks must abort without touching either original fixture.
            with h5py.File(chunks[1], 'r+') as handle:
                handle.attrs['external_edit'] = 'changed during copy'
            changed = True
    monkeypatch.setattr(merge, '_copy_values', alter_source)
    output = tmp_path / 'failed.h5'
    with pytest.raises(ValueError, match='Source changed'):
        merge_products(chunks, output)
    assert not output.exists() and not list(tmp_path.glob('.*.partial'))


@pytest.mark.parametrize('chunk_rows', [0, -1, True, 1.5])
def test_invalid_block_sizes_fail_before_io(tmp_path, chunk_rows):
    with pytest.raises(ValueError, match='chunk_rows'):
        merge_products([], tmp_path / 'missing.h5', chunk_rows=chunk_rows)
