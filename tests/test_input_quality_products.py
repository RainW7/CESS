"""Input quality survives layout adapters and EL subsets without repairing inputs."""
import json

import h5py
import numpy as np
import pytest

from cess.io.input_quality import FLAG_MEANINGS, write_input_quality
from cess.io.manifest import _input_quality
from cess.io.products import convert_product, open_product
from cess.io.selection import SourceSelection


def test_quality_catalogue_mapping_row_reader_and_slice(tmp_path):
    source, target = tmp_path / 'old.h5', tmp_path / 'new.h5'
    flags = np.array([0, 2, 0, 32], dtype=np.uint32)
    with h5py.File(source, 'w') as handle:
        handle.attrs['schema_version'] = '2.5'
        handle['ID'] = [10, 20, 30, 40]
        handle['data_mask'] = np.where(flags == 0, 1, 2)
        write_input_quality(handle, flags)
    original = source.read_bytes()
    convert_product(source, target)
    assert source.read_bytes() == original
    with open_product(target) as product:
        np.testing.assert_array_equal(product.catalog['input_quality_flags'][:], flags)
        assert product.read_row(1, bands=())['input_quality_flags'] == 2
        mapping = json.loads(product.handle['provenance/conversion'].attrs['mapping_json'])
        assert mapping['input_quality_flags'] == 'catalog/input_quality_flags'
        selected = SourceSelection(product, 1, 3)
        np.testing.assert_array_equal(selected['input_quality_flags'][:], [2, 0])
        _input_quality(product.legacy_view(), 4, 'standard')


@pytest.mark.parametrize('error', ['dtype', 'alignment', 'axis', 'policy', 'legend', 'bits', 'mask'])
def test_manifest_rejects_quality_contract_corruption(tmp_path, error):
    with h5py.File(tmp_path / 'product.h5', 'w') as handle:
        handle['data_mask'] = [1, 2]
        data = write_input_quality(handle, [0, next(iter(FLAG_MEANINGS))])
        if error in ('dtype', 'alignment'):
            attrs = dict(data.attrs)
            del handle['input_quality_flags']
            data = handle.create_dataset('input_quality_flags',
                data=[0.] if error == 'alignment' else [0., 1.])
            data.attrs.update(attrs)
        elif error == 'axis':
            data.attrs['axis_order'] = 'field'
        elif error == 'policy':
            data.attrs['policy_version'] = 'unknown'
        elif error == 'legend':
            del data.attrs['flag_masks']
        elif error == 'bits':
            data[1] = 2**31
        else:
            handle['data_mask'][1] = 1
        with pytest.raises(ValueError):
            _input_quality(handle, 2, 'standard')


def test_old_products_keep_unknown_quality_and_el_checks_inheritance(tmp_path):
    with h5py.File(tmp_path / 'parent.h5', 'w') as parent, \
            h5py.File(tmp_path / 'child.h5', 'w') as child:
        _input_quality(parent, 2, 'standard')
        _input_quality(child, 2, 'el-detect', parent)
        write_input_quality(parent, [0, 4])
        with pytest.raises(ValueError, match='availability'):
            _input_quality(child, 2, 'el-detect', parent)
        write_input_quality(child, [0, 4])
        _input_quality(child, 2, 'el-detect', parent)
        child['input_quality_flags'][1] = 8
        with pytest.raises(ValueError, match='selected parent rows'):
            _input_quality(child, 2, 'el-detect', parent)


@pytest.mark.parametrize('structured', [False, True])
def test_el_postprocess_inherits_flags_and_policy_for_selected_sources(tmp_path, structured):
    from cess.config.loader import load_config
    from cess.pipeline.emission import simulate_file
    source = tmp_path / 'parent.h5'
    with h5py.File(source, 'w') as handle:
        handle.attrs['schema_version'] = '2.5'
        handle['ID'] = np.array([b'a', b'b', b'c'])
        handle['input_row_indices'] = [10, 20, 30]
        write_input_quality(handle, [0, 4, 0])
        for band in ('GV', 'GI'):
            group = handle.create_group(band)
            group['wave'] = [4000., 5000., 6000.]
            for name in ('flux_ujy', 'flux_ujy_with_noise', 'snr'):
                group[name] = np.zeros((3, 3))
    if structured:
        target = tmp_path / 'structured.h5'
        convert_product(source, target)
        source = target
    original = source.read_bytes()
    output = simulate_file(str(tmp_path), source.name, output_dir=tmp_path / 'el',
        config=load_config(), source_start=1, source_stop=3)
    assert source.read_bytes() == original
    with open_product(source) as parent, open_product(output) as child:
        selected = SourceSelection(parent, 1, 3)
        np.testing.assert_array_equal(child.catalog['ID'][:], [b'b', b'c'])
        np.testing.assert_array_equal(child.catalog['input_quality_flags'][:], [4, 0])
        _input_quality(child.legacy_view(), 2, 'el-detect', selected)
