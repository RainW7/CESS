"""Small canonical I/O checks for the support-module responsibility split."""
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import h5py
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cess import contracts
from cess.io import atomic, cache, science_writer, selection
from cess.pipeline import support


def test_aliases_share_contracts_and_one_cache_owner():
    assert support._RESOURCE_CACHE is cache._RESOURCE_CACHE
    assert support.cached_pickle is cache.cached_pickle
    assert support.cached_table is cache.cached_table
    assert support.SourceSelection is selection.SourceSelection
    assert support.atomic_hdf5 is atomic.atomic_hdf5
    assert support.write_vlen_rows is science_writer.write_vlen_rows
    assert support.LINE_MEASUREMENT_DTYPE is contracts.LINE_MEASUREMENT_DTYPE
    assert support.MGE_MORPHOLOGY_FIELDS is contracts.MGE_MORPHOLOGY_FIELDS
    assert support.SCHEMA_VERSION == '2.5'
    assert support.REPAIR_REVISION == '20261024-D4'


def test_pickle_cache_reuses_identity_and_invalidates_changed_file(tmp_path):
    path = tmp_path / 'trusted.pkl'
    path.write_bytes(pickle.dumps({'value': [1]}))
    first = support.cached_pickle(path)
    assert cache.cached_pickle(path) is first
    assert first == {'value': [1]}
    path.write_bytes(pickle.dumps({'value': [1, 2, 3]}))
    second = support.cached_pickle(path)
    assert second is cache.cached_pickle(path) and second is not first
    assert second == {'value': [1, 2, 3]}


def test_profile_loader_preserves_support_cached_pickle_patch(tmp_path, monkeypatch):
    from cess.morphology import libraries
    paths = [tmp_path / 'width.pkl', tmp_path / 'height.pkl']
    for path in paths:
        path.write_bytes(b'identity-only fixture; never unpickle')
    objects = {str(paths[0]): np.array([1.]), str(paths[1]): np.array([2.])}
    calls = []
    def loader(path):
        calls.append(str(path))
        return objects[str(path)]
    descriptions = []
    def describe(width, height, parameters, axes):
        descriptions.append((width, height))
        return {'generation': {'status': 'unknown'}}
    monkeypatch.setattr(support, 'cached_pickle', loader)
    monkeypatch.setattr(libraries, 'describe_profile_libraries', describe)
    settings = dict(widthlib_pkl=str(paths[0]), heightlib_pkl=str(paths[1]), arcsecperpix=.074)
    axes = dict(n=[1.], re_pixel=[4.], pa_deg=[0.], axis_ratio=[1.])
    width, height, returned_axes, metadata = support.load_profile_libraries(settings, axes)
    assert width is objects[str(paths[0])] and height is objects[str(paths[1])]
    assert returned_axes is axes
    assert calls == [str(path) for path in paths] * 2
    assert len(descriptions) == 1
    metadata['generation']['status'] = 'changed return value'
    again = support.profile_library_provenance(settings, axes, width, height)
    assert again['generation']['status'] == 'unknown'
    assert len(descriptions) == 1


def test_selected_rows_keep_shared_axes_and_empty_interval(tmp_path):
    path = tmp_path / 'catalogue.hdf5'
    with h5py.File(str(path), 'w') as file:
        file['ID'] = np.arange(4)
        file['parameters'] = np.arange(12).reshape(4, 3)
        group = file.create_group('best_fit')
        group['wavelength_rest'] = np.arange(4.)
        group['z'] = np.arange(4.)
        group['spec_0'] = np.arange(16.).reshape(4, 4)
    with h5py.File(str(path), 'r') as file:
        view = support.SourceSelection(file, 1, 3)
        np.testing.assert_array_equal(view['ID'][:], [1, 2])
        np.testing.assert_array_equal(view['best_fit']['spec_0'][:], np.arange(16.).reshape(4, 4)[1:3])
        np.testing.assert_array_equal(view['best_fit']['wavelength_rest'][:], np.arange(4.))
        assert support.simulation_filename('catalogue.hdf5', view) == 'CSST_grism_catalogue_rows1-3_2.hdf5'
        empty = selection.SourceSelection(file, 0, 0)
        assert empty['ID'].shape == (0,)
        assert empty['best_fit']['wavelength_rest'].shape == (4,)


def test_atomic_failure_cleanup_and_no_clobber(tmp_path):
    destination = tmp_path / 'product.hdf5'
    with pytest.raises(RuntimeError, match='deliberate'):
        with support.atomic_hdf5(destination) as file:
            file['ID'] = [3]
            raise RuntimeError('deliberate')
    assert not destination.exists() and not list(tmp_path.glob('*.partial'))
    with atomic.atomic_hdf5(destination) as file:
        file['ID'] = [7]
    original = destination.read_bytes()
    with pytest.raises(FileExistsError):
        with support.atomic_hdf5(destination):
            raise AssertionError('must not open an existing destination')
    assert destination.read_bytes() == original


def test_writer_aliases_preserve_rows_masks_and_attributes(tmp_path):
    path = tmp_path / 'product.hdf5'
    with h5py.File(str(path), 'w') as file:
        file['ID'] = [1, 2]
        rows = [dict(source_fraction=.7, spatial_pixels=3, profile_height_sum=.6), None]
        group = support.write_spectral_extraction(file, rows,
            {'spectral_extraction': {'mode': 'explicit'}}, {'status': 'fixture'})
        np.testing.assert_array_equal(group['valid'][:], [True, False])
        np.testing.assert_array_equal(group['source_fraction'][:], [.7, np.nan])
        assert group['spatial_pixels'].attrs['unit'] == 'pixel'
        ragged = file.create_dataset('ragged', (2,), dtype=h5py.vlen_dtype(np.dtype('float64')))
        science_writer.write_vlen_rows(ragged, [[4.], []])
        np.testing.assert_array_equal(ragged[0], [4.])
        assert ragged[1].size == 0
    padded = support.pad_spectral_rows([np.array([2.]), np.nan], 2, 3)
    np.testing.assert_array_equal(padded, [[2., np.nan, np.nan], [np.nan] * 3])


def test_provenance_retains_r2_and_shared_code_hash_provider(tmp_path, monkeypatch):
    import cess.provenance as provenance
    from cess import __version__
    expected = {'/fixture/package/source.py': 'a' * 64}
    calls = []
    def hashes():
        calls.append(True)
        return expected
    monkeypatch.setattr(provenance, 'code_sha256', hashes)
    with h5py.File(str(tmp_path / 'metadata.hdf5'), 'w') as file:
        file['ID'] = np.empty(0, dtype='i8')
        support.record_runner_provenance(file, {}, dict(contracts.OPTION_DEFAULTS))
        assert file.attrs['cess_repair_revision'] == '20261024-D4'
        assert file.attrs['schema_version'] == '2.5'
        assert file.attrs['software_version'] == __version__
        assert json.loads(file.attrs['code_sha256_json']) == expected
    assert calls == [True]


def test_support_import_does_not_load_config_resources_or_engine(tmp_path):
    code = r'''
import importlib
import pickle
import sys
import numpy as np
import h5py
import scipy.interpolate
from astropy.table import Table
def forbidden(*args, **kwargs): raise AssertionError('unexpected resource read')
h5py.File = forbidden
pickle.load = forbidden
Table.read = forbidden
np.random.seed(319)
before = np.random.get_state()
module = importlib.import_module('cess.pipeline.support')
after = np.random.get_state()
assert np.array_equal(before[1],after[1]) and before[2:]==after[2:]
assert 'cess.engine' not in sys.modules and 'galsim' not in sys.modules
assert 'cess.config.loader' not in sys.modules
assert not module._RESOURCE_CACHE
'''
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=str(tmp_path), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
    assert result.returncode == 0, result.stderr.decode('utf-8', errors='replace')
