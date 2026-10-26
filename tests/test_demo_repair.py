"""Small schema/provenance and noninteractive-rendering checks for demo.py."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import h5py
import numpy as np
import pytest
CORE = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(CORE))
spec = importlib.util.spec_from_file_location('cess_demo_tests', str(CORE / 'cess/cli/demo.py'))
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def _table(file, name, values, names, layout):
    if layout == 'structured':
        data = np.empty(len(values), dtype=[(field, 'f8') for field in names])
        for index, field in enumerate(names):
            data[field] = values[:, index]
    else:
        data = values if layout == 'dense' else values.T[:, :, None]
    table = file.create_dataset(name, data=data)
    table.attrs['name'] = names
    if layout == 'dense':
        table.attrs['axis_order'] = 'source,field'


def _fixture(folder, layout='dense'):
    source, product = folder / 'actual-input.hdf5', folder / 'misleading_rows999-1001_2.hdf5'
    with h5py.File(str(source), 'w') as file:
        file['ID'] = np.array([b'm0', b'm1', b'm2', b'm3'])
        file['best_fit/wavelength_rest'] = [.2, .5, 1.]
        file['best_fit/z'] = [.1, .2, .3, .4]
        file['best_fit/spec_model'] = np.arange(1., 5.)[:, None] * np.ones((4, 3))
    with h5py.File(str(product), 'w') as file:
        file['ID'] = np.array([b'm2', b'm3'])
        file['input_row_indices'] = [2, 3]
        file.attrs['input_product'] = str(source)
        file.attrs['schema_version'] = '2.0' if layout == 'dense' else 'legacy'
        names = ['RA', 'Dec', 'z_best', 'MAG_G', 'MAG_R', 'MAG_Z',
                 'n', 'Re', 'PA', 'baratio', 'str_mass']
        values = np.array([[120., 20., .3, 20., 21., 22., 1., 4., 10., .7, 10.],
                           [121., 21., .4, 21., 22., 23., 2., 5., 20., .8, 11.]])
        _table(file, 'parameters_desi', values, names, layout)
        # Deliberately put S/N in different positions from every legacy layout.
        _table(file, 'parameters_grism', np.tile([333., 9999., 111., 222.], (2, 1)),
               ['gi_snr_mean', 'gu_rms_in_e', 'gu_snr_mean', 'gv_snr_mean'], layout)
        _table(file, 'parameters_phot', np.tile([7., 24., .1], (2, 1)),
               ['snr_r', 'mag_r', 'magerr_r'], layout)
        for band, limits in [('GU', (2500., 4200.)), ('GV', (3800., 6600.)), ('GI', (5800., 10500.))]:
            file[band + '/wave'] = np.linspace(*limits, 20)
            file[band + '/flux_ujy'] = np.ones((2, 20))
            file[band + '/flux_ujy_with_noise'] = np.ones((2, 20)) * 1.1
    return source, product


@pytest.mark.parametrize('layout', ['dense', 'legacy', 'structured'])
def test_named_tables_and_chunk_native_model_mapping(tmp_path, layout):
    source, product = _fixture(tmp_path, layout)
    data = demo.read_demo_data(product, 0)
    assert data['grism']['gu_snr_mean'] == 111.
    assert data['grism']['gv_snr_mean'] == 222.
    assert data['grism']['gi_snr_mean'] == 333.
    assert data['phot']['mag_r'] == 24. and data['phot']['snr_r'] == 7.
    assert data['model']['path'] == source and data['model']['row'] == 2
    np.testing.assert_allclose(data['model']['flux_ujy'], 3.)
    np.testing.assert_allclose(data['model']['wave_um'], np.array([.2, .5, 1.]) * 1.3)


def test_legacy_explicit_model_by_id_and_incorrect_provenance_is_not_plotted(tmp_path):
    source, product = _fixture(tmp_path, 'legacy')
    with h5py.File(str(product), 'a') as file:
        del file.attrs['input_product']
        del file['input_row_indices']
    assert demo.read_demo_data(product, 0, model_input=source)['model']['row'] == 2
    with h5py.File(str(product), 'a') as file:
        file['input_row_indices'] = [0, 1]
    data = demo.read_demo_data(product, 0, model_input=source)
    assert data['model'] is None and 'row/ID' in data['model_note']


def test_save_png_without_model_or_gui(tmp_path):
    source, product = _fixture(tmp_path)
    source.unlink()
    destination = tmp_path / 'demo.png'
    env = dict(os.environ, PYTHONPATH=str(CORE), MPLBACKEND='Agg', MPLCONFIGDIR=str(tmp_path / 'mpl'),
               PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-B', '-m', 'cess.cli.demo', str(product),
                             '0', 'emissionline', '--save', str(destination)],
                            cwd=str(tmp_path), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True, timeout=40)
    assert result.returncode == 0, result.stdout
    assert 'Input model unavailable' in result.stdout
    assert destination.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')


def test_import_is_safe_with_unrelated_argv(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('demo import attempted HDF5 I/O')
    monkeypatch.setattr(h5py, 'File', forbidden)
    monkeypatch.setattr(sys, 'argv', ['another-program', '--unrelated'])
    monkeypatch.chdir(str(tmp_path))
    spec = importlib.util.spec_from_file_location('cess_demo_import_probe', str(CORE / 'cess/cli/demo.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
