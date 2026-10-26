"""R3 strict layout identity and unchanged scientific reconciliation on tiny files."""
from copy import deepcopy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from cess.io import manifest, products
from cess.config.loader import load_config
from cess.pipeline.support import SourceSelection, record_runner_provenance
from test_manifest_repair import (fixture_job, make_product, populated_product,
                                 make_el, make_empty_lines)
from test_mge_pipeline import bundle, run_tiny


def structured(job, path):
    job['output_layout'] = 'structured'
    with h5py.File(str(path), 'r+') as handle:
        products.structure_product(handle)
    return manifest.snapshot(job['config'], job['options'], job['mode'],
                             job['seed'], [job], output_layout='structured')


@pytest.mark.parametrize('mode', ['standard', 'paired-wce', 'el-detect'])
@pytest.mark.parametrize('interval', [(0, 0), (1, 2), (1, 3)])
def test_all_modes_empty_single_invalid_and_saved_sha(tmp_path, mode, interval):
    job, _ = fixture_job(tmp_path, mode, *interval)
    path = make_product(job)
    state = structured(job, path)
    before = path.read_bytes()
    result = manifest.verify_product(job, state)
    assert result == dict(output=str(path), sha256=manifest._sha256(path))
    assert manifest.verify_product(job, state, result['sha256']) == result
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='Output SHA256'):
        manifest.verify_product(job, state, '0'*64)


def test_snapshot_layout_identity_and_legacy_defaults(tmp_path):
    job, state = fixture_job(tmp_path)
    assert state['output_layout'] == 'legacy' and state['input_schemas'] == {}
    path = make_product(job)
    assert manifest.verify_product(job, state)
    old = deepcopy(state)
    del old['output_layout']
    with pytest.raises(ValueError, match='predates layout identity'):
        manifest.verify_product(job, old)
    job['output_layout'] = 'structured'
    with pytest.raises(ValueError, match='Job output layout'):
        manifest.snapshot(job['config'], job['options'], job['mode'], job['seed'], [job])
    job['output_layout'] = 'future'
    with pytest.raises(ValueError, match='Unknown output layout'):
        manifest.expected_product(job)
    assert path.is_file()


@pytest.mark.parametrize('layout', ['legacy', 'structured'])
def test_storage_merge_cannot_be_adopted_as_native_simulation_job(tmp_path, layout):
    job, state = fixture_job(tmp_path)
    path = make_product(job)
    with h5py.File(path, 'r+') as handle:
        handle.create_group('merge_history').attrs['operation'] = 'merge_simulation_chunks'
    if layout == 'structured':
        state = structured(job, path)
    with pytest.raises(ValueError, match='Merged products are not native'):
        manifest.verify_product(job, state)


@pytest.mark.parametrize('actual', ['legacy', 'structured'])
def test_physical_layout_cannot_be_adopted_by_other_layout(tmp_path, actual):
    job, state = fixture_job(tmp_path)
    path = make_product(job)
    if actual == 'structured':
        structured(job, path)
    requested = 'structured' if actual == 'legacy' else 'legacy'
    job['output_layout'] = requested
    state = manifest.snapshot(job['config'], job['options'], job['mode'], job['seed'],
                              [job], output_layout=requested)
    with pytest.raises(ValueError, match='physical schema'):
        manifest.verify_product(job, state)


@pytest.mark.parametrize('dangling', [False, True])
def test_expected_product_symlink_is_not_a_missing_or_recoverable_product(tmp_path, dangling):
    job, state = fixture_job(tmp_path)
    path = make_product(job)
    target = tmp_path / 'external.hdf5'
    path.replace(target)
    if dangling:
        target.unlink()
    path.symlink_to(target)
    assert manifest.expected_product(job) == path
    with pytest.raises(ValueError, match='must not be a symlink'):
        manifest.verify_product(job, state)


@pytest.mark.parametrize('damage', ['layout', 'mode', 'kind', 'group', 'status',
                                   'mapping', 'original_schema', 'original_stage'])
def test_structured_header_and_original_provenance_must_agree(tmp_path, damage):
    job, _ = fixture_job(tmp_path)
    path = make_product(job)
    state = structured(job, path)
    with h5py.File(str(path), 'r+') as handle:
        if damage == 'layout':
            handle.attrs['layout_version'] = 'future'
        elif damage == 'mode':
            handle.attrs['mode'] = 'paired-wce'
        elif damage == 'kind':
            handle.attrs['product_kind'] = 'emission_line_postprocess'
        elif damage == 'group':
            del handle['lines']
        elif damage == 'status':
            handle['zero_order'].attrs['cess_layout_status'] = 'disabled'
        elif damage == 'mapping':
            handle['provenance/conversion'].attrs['mapping_json'] = '{}'
        elif damage == 'original_schema':
            handle['provenance'].attrs['schema_version'] = '3.0'
        else:
            handle['provenance'].attrs['pipeline_stage'] = 'emission_line_postprocess'
    with pytest.raises(ValueError):
        manifest.verify_product(job, state)


@pytest.mark.parametrize('damage', ['id', 'rows', 'covariance_shape', 'covariance_metadata',
                                   'source_axis', 'config', 'profile_binding'])
def test_structured_view_retains_scientific_and_source_checks(tmp_path, damage):
    job, _ = fixture_job(tmp_path)
    path = make_product(job)
    if damage == 'source_axis':
        # Preserve unknown fields, while retaining their declared row contract.
        with h5py.File(str(path), 'r+') as handle:
            handle.create_dataset('extra', data=np.zeros(3)).attrs['axis_order'] = 'source'
    state = structured(job, path)
    with h5py.File(str(path), 'r+') as handle:
        if damage == 'id':
            handle['catalog/ID'][0] = 999
        elif damage == 'rows':
            handle['catalog/input_row_indices'][0] = 42
        elif damage == 'covariance_shape':
            del handle['spectra/GV/covariance_lag1']
            handle['spectra/GV/covariance_lag1'] = np.zeros((2, 3))
        elif damage == 'covariance_metadata':
            handle['spectra/GV/covariance_lag1'].attrs['covariance_model'] = 'unknown'
        elif damage == 'config':
            handle['provenance'].attrs['config_json'] = '{}'
        elif damage == 'profile_binding':
            description = json.loads(handle['provenance'].attrs['profile_library_provenance_json'])
            description['resources']['width']['sha256'] = '0'*64
            handle['provenance'].attrs['profile_library_provenance_json'] = json.dumps(description)
    with pytest.raises(ValueError):
        manifest.verify_product(job, state)


@pytest.mark.parametrize('mode', ['paired-wce', 'el-detect'])
def test_sparse_lines_and_wce_pair_joins_still_checked(tmp_path, mode):
    job, _, path = populated_product(tmp_path, mode)
    state = structured(job, path)
    assert manifest.verify_product(job, state)
    with h5py.File(str(path), 'r+') as handle:
        pairs = handle['lines/GV/el_diagnostics_wce/pairs']
        rows = pairs[:]
        rows['noisy_peak_id'][0] = 999
        pairs[:] = rows
    with pytest.raises(ValueError, match='missing source/peak ID'):
        manifest.verify_product(job, state)


def make_el_from_parent(job):
    """Use the public reader; no duplicated spectral arrays or science run."""
    path = manifest.expected_product(job)
    path.parent.mkdir(exist_ok=True)
    source_path = Path(job['input_path']) / job['filename']
    with products.open_product(source_path) as parent:
        selected = SourceSelection(parent.legacy_view(), job['source_start'], job['source_stop'])
        with h5py.File(str(path), 'w') as output:
            output['ID'] = selected['ID'][:]
            output['input_row_indices'] = selected['input_row_indices'][:]
            output['parent_row_indices'] = selected.source_rows
            output['parameters_elg'] = np.zeros((len(selected['ID']), 2))
            for band in ('GV', 'GI'):
                make_el(output, band, flat=True)
                make_empty_lines(output, band, job, known=True)
            record_runner_provenance(output, load_config(job['config']),
                manifest.normalise_options(job['options']), source=selected, seed=job['seed'],
                input_sha256=manifest._sha256(source_path), input_path=source_path,
                source_start=job['source_start'], source_stop=job['source_stop'])
    return path


@pytest.mark.parametrize('layout', ['legacy', 'structured'])
def test_el_structured_parent_inherits_rows_extraction_and_real_schema(tmp_path, layout):
    parent_job, _ = fixture_job(tmp_path, start=0, stop=4)
    parent_path = make_product(parent_job)
    with h5py.File(str(parent_path), 'r+') as handle:
        handle['input_row_indices'][:] = [50, 70, 90, 110]
        products.structure_product(handle)
    job = dict(parent_job, input_path=str(parent_path.parent), filename=parent_path.name,
               output_dir=str(tmp_path / 'el-output'), mode='el-detect', output_layout=layout,
               source_start=1, source_stop=3,
               options=dict(morph=False, wave_cal=False, photo=False, el_detect=True))
    state = manifest.snapshot(job['config'], job['options'], job['mode'], job['seed'], [job],
                              output_layout=layout)
    assert state['input_schemas'][str(parent_path)] == dict(
        schema_version='3.0', layout_version=products.LAYOUT_VERSION)
    assert state['inputs'][str(parent_path)] == manifest._sha256(parent_path)
    path = make_el_from_parent(job)
    if layout == 'structured':
        structured(job, path)
    assert manifest.verify_product(job, state)
    with products.open_product(path) as product:
        original = product.legacy_view()
        np.testing.assert_array_equal(original['input_row_indices'][:], [70, 90])
        np.testing.assert_array_equal(original['parent_row_indices'][:], [1, 2])
        assert 'spectral_extraction' in original
    wrong = deepcopy(state)
    wrong['input_schemas'][str(parent_path)]['schema_version'] = '2.5'
    with pytest.raises(ValueError, match='input schema/layout'):
        manifest.verify_product(job, wrong)


def test_structured_mge_source_links_and_inherited_descriptor_are_not_weakened(
        tmp_path, bundle, monkeypatch):
    path, _, _, job, _, _ = run_tiny(tmp_path, bundle, monkeypatch)
    state = structured(job, path)
    assert manifest.verify_product(job, state)
    with h5py.File(str(path), 'r+') as handle:
        handle['catalog/mge_morphology/selected_parameters'][0, 0] = 999.
    with pytest.raises(ValueError, match='linked to the original source row'):
        manifest.verify_product(job, state)
