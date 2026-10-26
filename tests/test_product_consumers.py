"""Subset sampling and plotting consume both product layouts without recalculation."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import h5py
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cess.cli import demo
from cess.io import sampling
from cess.io.products import convert_product, open_product


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def product_pair(folder, extras=True, numeric_ids=False):
    legacy, structured = folder / 'legacy.hdf5', folder / 'structured.hdf5'
    with h5py.File(str(legacy), 'w') as file:
        file.attrs['schema_version'] = '2.5'
        file.attrs['noise_model'] = 'parent-noise-definition'
        file.attrs['code_sha256_json'] = '{"parent.py":"recorded-source-hash"}'
        file['ID'] = np.arange(4, dtype='i8') if numeric_ids else np.array([b'a0', b'a1', b'a2', b'a3'])
        file['input_row_indices'] = [10, 11, 12, 13]
        file['parent_row_indices'] = [20, 21, 22, 23]
        file['data_mask'] = [1, 1, 1, 1]
        parameters = file.create_dataset('parameters_desi', data=np.array([
            [10., .1, 20., 21., 22.], [11., .2, 20., 21., 22.],
            [12., .3, 20., 21., 22.], [13., .4, 20., 21., 22.]]))
        parameters.attrs['name'] = ['str_mass', 'z_best', 'MAG_G', 'MAG_R', 'MAG_Z']
        parameters.attrs['axis_order'] = 'source,field'
        parameters = file.create_dataset('parameters_grism', data=np.array([
            [6., 999., 4., 5.], [6., 999., 5., 1.],
            [8., 999., 6., 7.], [1., 999., 3., 4.]]))
        parameters.attrs['name'] = ['gi_snr_mean', 'gu_rms_in_e', 'gu_snr_mean', 'gv_snr_mean']
        parameters.attrs['axis_order'] = 'source,field'
        for band in ('GU', 'GV', 'GI'):
            group = file.create_group(band)
            group.attrs['calibration'] = 'fixture-center-response'
            group['wave'] = [5000., 5010., 5020.]
            group['wave'].attrs['unit'] = 'Angstrom'
            group['flux_ujy'] = np.arange(12.).reshape(4, 3)
            group['flux_ujy_with_noise'] = np.arange(12.).reshape(4, 3) + .5
            group['response_at_pixel_center'] = [.1, .2, .3]
            if extras:
                group['ferr'] = np.full((4, 3), .2)
                group['spec_mask'] = np.tile([False, True, True], (4, 1))
                group['snr_mask'] = np.tile([False, False, True], (4, 1))
                group['flux_ujy_with_noise_wce'] = np.arange(12.).reshape(4, 3) + .7
                group['ferr_wce'] = np.full((4, 3), .3)
                for name, value in (('covariance_lag1', .001), ('covariance_lag1_wce', .002)):
                    group[name] = np.full((4, 2), value)
                    group[name].attrs['unit'] = 'uJy^2'
                    group[name].attrs['covariance_model'] = 'parent-lag1-model'
    before = digest(legacy)
    convert_product(legacy, structured)
    assert digest(legacy) == before
    return legacy, structured


def test_reader_and_demo_are_layout_independent(tmp_path):
    legacy, structured = product_pair(tmp_path)
    first, second = [sampling.read_hdf5_files([path]) for path in (legacy, structured)]
    assert [s['id'] for s in first[0]] == [b'a0', b'a2'] == [s['id'] for s in second[0]]
    for a, b in zip(first[0], second[0]):
        assert a['source_row'] == b['source_row']
        assert a['input_row_indices'] == b['input_row_indices']
        np.testing.assert_array_equal(a['gv_extras']['covariance_lag1_wce'], b['gv_extras']['covariance_lag1_wce'])
    a, b = demo.read_demo_data(legacy, 2), demo.read_demo_data(structured, 2)
    assert a['id'] == b['id'] == b'a2'
    assert a['input_row'] == b['input_row'] == 12
    assert a['grism'] == b['grism'] and a['desi'] == b['desi']
    for band in ('GU', 'GV', 'GI'):
        for key in ('wave_um', 'model', 'observed'):
            np.testing.assert_array_equal(a['bands'][band][key], b['bands'][band][key])


@pytest.mark.parametrize('numeric_ids', [False, True])
def test_subset_preserves_covariance_metadata_and_row_links(tmp_path, numeric_ids):
    legacy, structured = product_pair(tmp_path, numeric_ids=numeric_ids)
    before = [digest(path) for path in (legacy, structured)]
    sources, gu, gv, gi = sampling.read_hdf5_files([legacy, structured])
    output = tmp_path / 'sample.hdf5'
    sampling.write_output(output, sources, gu, gv, gi, seed=17, input_files=[legacy, structured])
    with h5py.File(str(output), 'r') as file:
        assert file['ID'].dtype == (np.dtype('i8') if numeric_ids else np.dtype('S2'))
        np.testing.assert_array_equal(file['input_row_indices'][:], [10, 12, 10, 12])
        np.testing.assert_array_equal(file['parent_row_indices'][:], [20, 22, 20, 22])
        np.testing.assert_array_equal(file['provenance/source_row'][:], [0, 2, 0, 2])
        np.testing.assert_array_equal(file['provenance/parent_index'][:], [0, 0, 1, 1])
        for name, value in (('covariance_lag1', .001), ('covariance_lag1_wce', .002)):
            assert file['GV/' + name].shape == (4, 2)
            np.testing.assert_array_equal(file['GV/' + name][:], np.full((4, 2), value))
            assert file['GV/' + name].attrs['covariance_model'] == 'parent-lag1-model'
        parent = file['provenance/parents/parent_000001']
        assert parent['root_attributes'].attrs['schema_version'] == '3.0'
        assert parent['generation_attributes'].attrs['noise_model'] == 'parent-noise-definition'
        assert parent['generation_attributes'].attrs['code_sha256_json'] == '{"parent.py":"recorded-source-hash"}'
        np.testing.assert_array_equal(parent['bands/GV/shared/response_at_pixel_center'][:], [.1, .2, .3])
    with open_product(output) as view:
        assert len(view.catalog['ID']) == 4
    assert [digest(path) for path in (legacy, structured)] == before
    first = digest(output)
    with pytest.raises(FileExistsError):
        sampling.write_output(output, sources, gu, gv, gi)
    assert digest(output) == first


def test_missing_lag1_fills_pair_axis_and_keeps_availability(tmp_path):
    one, two = tmp_path / 'one', tmp_path / 'two'
    one.mkdir(); two.mkdir()
    legacy, _ = product_pair(one)
    _, structured = product_pair(two, extras=False)
    sources, *waves = sampling.read_hdf5_files([legacy, structured])
    output = tmp_path / 'mixed.hdf5'
    sampling.write_output(output, sources, *waves)
    with h5py.File(str(output), 'r') as file:
        assert file['GU/covariance_lag1'].shape == (4, 2)
        assert np.isnan(file['GU/covariance_lag1'][2:]).all()
        np.testing.assert_array_equal(file['GU/covariance_lag1_available'][:], [True, True, False, False])
    again, *waves = sampling.read_hdf5_files([output])
    assert 'covariance_lag1' not in again[-1]['gu_extras']
    second = tmp_path / 'second-generation.hdf5'
    sampling.write_output(second, again, *waves)
    with h5py.File(str(second), 'r') as file:
        np.testing.assert_array_equal(file['provenance/upstream_lineage/source_row'][:], [0, 2, 0, 2])
        np.testing.assert_array_equal(file['GU/covariance_lag1_available'][:], [True, True, False, False])


def test_sampling_seed_matches_archived_algorithm_and_isolates_global_rng():
    sources = [{'id': i, 'z_best': z} for i, z in enumerate(np.linspace(0., 2., 50))]
    state = np.random.get_state()
    try:
        np.random.seed(21)
        expected_next = np.random.RandomState(21).random_sample()
        selected = sampling.stratified_sampling(sources, target=11, n_bins=4, seed=17)
        assert [s['id'] for s in selected] == [17, 42, 23, 13, 33, 49, 2, 3, 12, 43, 28]
        assert np.random.random() == expected_next
    finally:
        np.random.set_state(state)


def test_empty_export_has_two_dimensional_spectra(tmp_path):
    waves = [np.array([1., 2., 3.])] * 3
    sampling.write_output(tmp_path / 'empty.hdf5', [], *waves, target=0)
    with open_product(tmp_path / 'empty.hdf5') as view:
        assert len(view.catalog['ID']) == 0
        assert view.band('GU')['flux_ujy'].shape == (0, 3)


def test_import_and_passthrough_sampling_do_not_require_pandas(tmp_path):
    code = r'''
import builtins
import sys
real = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'pandas' or name.startswith('pandas.'): raise AssertionError('eager pandas import')
    return real(name, *args, **kwargs)
builtins.__import__ = blocked
from cess.io import sampling
import cess.io.sampling as legacy
assert legacy is sampling
assert sampling.stratified_sampling([{'id':1,'z_best':.2}],target=2)[0]['id']==1
assert 'pandas' not in sys.modules
'''
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / 'src'), str(ROOT)]),
               PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=str(tmp_path), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    assert result.returncode == 0, result.stderr.decode('utf-8', errors='replace')


def test_console_adapter_keeps_callable_main_api(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sampling, 'main', lambda *args: calls.append(args))
    assert sampling.cli_main(['input.hdf5', '--output', str(tmp_path / 'out.hdf5'),
                              '--target', '7', '--n-bins', '3', '--seed', '11']) == 0
    assert calls == [([Path('input.hdf5')], tmp_path / 'out.hdf5', 7, 3, 11)]
