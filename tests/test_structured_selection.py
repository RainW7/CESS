"""Structured-product/legacy-view selection and strict metadata boundaries."""
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cess.io.products import ProductView, structure_product, validate_structure, _object_id
from cess.io.selection import SourceSelection
from test_product_layout import fixture


@pytest.mark.parametrize('kind', ['standard', 'paired', 'el'])
@pytest.mark.parametrize('count', [0, 1, 3])
def test_legacy_namespace_and_source_selection(tmp_path, kind, count):
    path = tmp_path / 'native.h5'
    fixture(path, kind, count, extension=True)
    with h5py.File(str(path), 'r+') as handle:
        structure_product(handle)
        product = ProductView(handle)
        legacy = product.legacy_view()
        assert legacy.attrs is not handle.attrs
        assert legacy.attrs['schema_version'] == '2.5'
        assert legacy.name == '/' and legacy.file.id == handle.id
        assert legacy.filename == str(path)
        assert isinstance(legacy['ID'], h5py.Dataset)
        assert isinstance(legacy['GV/intri_el_wave'], h5py.Dataset)
        assert legacy.get('absent') is None and 'absent' not in legacy
        assert 'future' in legacy.keys()
        np.testing.assert_array_equal(legacy['future/not_lost'][:], [42.])
        start = int(count > 1)
        for source in (handle, product, legacy):
            selected = SourceSelection(source, start, count)
            np.testing.assert_array_equal(selected['ID'][:], handle['catalog/ID'][start:count])
            np.testing.assert_array_equal(selected['input_row_indices'][:], np.arange(count)[start:]+100)
            assert selected.attrs['schema_version'] == '2.5'
            assert selected.file.id == handle.id and selected.filename == str(path)
            assert selected.name == '/' and selected['spectral_extraction']['valid'].shape == (count-start,)
            assert selected['mge_morphology']['selected_indices'].shape == (count-start, 4)
            # Sparse peak tables are not source-major, including singleton N=1.
            assert selected['GV']['intri_el_wave'] is not None
            assert isinstance(selected['GV']['intri_el_wave'], h5py.Dataset)
            assert len(selected['GV']['intri_el_wave']) == int(count > 0)
            if kind != 'el':
                spectrum = selected['GV']['flux_ujy']
                assert spectrum.shape == (count-start, 3)
                np.testing.assert_array_equal(spectrum[:], handle['spectra/GV/flux_ujy'][start:count])
                assert spectrum.name == '/spectra/GV/flux_ujy' and spectrum.file.id == handle.id
                assert selected['GV']['wave'].shape == (3,)
            else:
                assert selected['parameters_elg'].shape == (count-start, 2)
                assert _object_id(legacy['gv_intri_el_wave']) == _object_id(legacy['GV/intri_el_wave'])


def test_coincident_wave_and_line_table_lengths_are_never_source_sliced(tmp_path):
    path = tmp_path / 'equal-lengths.h5'
    fixture(path, count=3)
    with h5py.File(str(path), 'r+') as handle:
        del handle['GV/line_measurements/candidates']
        handle['GV/line_measurements/candidates'] = np.zeros(3, dtype=[('source_row','i8'),('peak_id','i8')])
        shared = handle['GV'].create_dataset('future_shared', data=[10.,20.,30.])
        shared.attrs['axis_order'] = 'source'
        structure_product(handle)
        selected = SourceSelection(handle, 1, 2)
        assert selected['GV']['flux_ujy'].shape == (1, 3)
        assert isinstance(selected['GV']['wave'], h5py.Dataset)
        assert isinstance(selected['GV']['line_measurements']['candidates'], h5py.Dataset)
        assert len(selected['GV']['line_measurements']['candidates']) == 3
        # Unknown extensions retain their original objects; their axes are not guessed.
        assert isinstance(selected['GV']['future_shared'], h5py.Dataset)
        np.testing.assert_array_equal(selected['GV']['future_shared'][:], [10.,20.,30.])


def test_el_unknown_band_extensions_and_empty_root_groups_survive(tmp_path):
    path = tmp_path / 'extensions.h5'
    fixture(path, kind='el', count=3)
    with h5py.File(str(path), 'r+') as handle:
        handle['GV/future_payload'] = [7., 8., 9.]
        handle.create_group('photometry').attrs['original_empty_note'] = 'keep'
        structure_product(handle)
        legacy = ProductView(handle).legacy_view()
        assert 'photometry' in legacy and len(legacy['photometry']) == 0
        assert legacy['photometry'].attrs['original_empty_note'] == 'keep'
        np.testing.assert_array_equal(legacy['GV/future_payload'][:], [7.,8.,9.])
        assert _object_id(legacy['GV/future_payload']) == _object_id(handle['provenance/extensions/bands/GV/future_payload'])


def test_visititems_is_metadata_only_and_deduplicates_el_aliases(tmp_path, monkeypatch):
    path = tmp_path / 'el.h5'
    fixture(path, kind='el', count=3, extension=True)
    with h5py.File(str(path), 'r+') as handle:
        structure_product(handle)
        def forbidden(*args, **kwargs):
            pytest.fail('Structural validation/visitation must not read dataset arrays')
        monkeypatch.setattr(h5py.Dataset, '__getitem__', forbidden)
        assert validate_structure(handle)['mode'] == 'el-detect'
        legacy = ProductView(handle).legacy_view()
        visited = []
        legacy.visititems(lambda name, value: visited.append((name, value)))
        datasets = [(name, value) for name, value in visited if isinstance(value, h5py.Dataset)]
        identities = [_object_id(value) for _, value in datasets]
        assert len(identities) == len(set(identities))
        assert _object_id(legacy['gv_intri_el_wave']) in identities
        assert all(isinstance(value, h5py.Dataset) for _, value in datasets)
        assert legacy.visititems(lambda name, value: 'stop') == 'stop'


@pytest.mark.parametrize('damage', [
    'root_alias', 'missing_group', 'status', 'mode', 'original_schema', 'tool_version',
    'invented_source', 'missing_target', 'wrong_route', 'duplicate_target',
    'duplicate_json_key', 'undeclared', 'undeclared_hardlink', 'split_attrs'])
def test_damaged_or_ambiguous_structured_products_fail_closed(tmp_path, damage):
    path = tmp_path / 'damaged.h5'
    fixture(path, count=3)
    with h5py.File(str(path), 'r+') as handle:
        structure_product(handle)
        conversion = handle['provenance/conversion']
        mapping = json.loads(conversion.attrs['mapping_json'])
        if damage == 'root_alias':
            handle['ID'] = handle['catalog/ID']
        elif damage == 'missing_group':
            del handle['photometry']
        elif damage == 'status':
            handle['spectra'].attrs['cess_layout_status'] = 'disabled'
        elif damage == 'mode':
            handle.attrs['mode'] = 'paired-wce'
        elif damage == 'original_schema':
            handle['provenance'].attrs['schema_version'] = '3.0'
        elif damage == 'tool_version':
            conversion.attrs['tool_version'] = 'other'
        elif damage == 'invented_source':
            conversion.attrs['source_sha256'] = '0'*64
        elif damage == 'missing_target':
            del handle['spectra/GV/flux_ujy']
        elif damage == 'wrong_route':
            mapping['ID'] = 'spectra/GV/wave'
        elif damage == 'duplicate_target':
            mapping['gv_intri_el_id'] = mapping['GV/intri_el_id']
        elif damage == 'duplicate_json_key':
            conversion.attrs['mapping_json'] = '{"ID":"catalog/ID","ID":"catalog/ID"}'
        elif damage == 'undeclared':
            handle['spectra/GV/undeclared'] = [1., 2., 3.]
        elif damage == 'undeclared_hardlink':
            handle['spectra/GV/undeclared'] = handle['catalog/ID']
        elif damage == 'split_attrs':
            handle['lines/GV'].attrs['custom_band_attribute'] = 'different'
        if damage in ('wrong_route', 'duplicate_target'):
            conversion.attrs['mapping_json'] = json.dumps(mapping)
        with pytest.raises(ValueError):
            validate_structure(handle)
        with pytest.raises(ValueError):
            ProductView(handle)
        with pytest.raises(ValueError):
            structure_product(handle)  # Idempotence must not bypass validation.


def test_unknown_legacy_sentinel_and_misaligned_declared_rows(tmp_path):
    path = tmp_path / 'legacy.h5'
    with h5py.File(str(path), 'w') as handle:
        handle.attrs['schema_version'] = 'legacy'
        handle['ID'] = [10, 20, 30]
        handle['parameters_grism'] = np.zeros((2, 4))
        assert ProductView(handle).legacy_view() is handle
        with pytest.raises(ValueError, match='not aligned'):
            SourceSelection(handle, 1, 2)['parameters_grism']
