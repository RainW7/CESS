"""Runner regression checks using temporary products and tiny synthetic inputs."""
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import pytest
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / 'src'
sys.path.insert(0, str(CORE))
from cess.pipeline.support import (atomic_hdf5, combine_band_masks,
                            output_destination, pad_spectral_rows,
                            record_runner_provenance, throughput_interpolator,
                            write_el_details, write_vlen_rows, SourceSelection,
                            simulation_filename, cached_table, cached_pickle,
                            measure_runner_band, throughput_arrays,
                            write_response_products)
from test_pipeline_smoke import run_fixture, make_input


@pytest.mark.parametrize('columns', [('WAVELENGTH', 'SENSITIVITY'),
                                     ('Wavelength', 'Efficiency')])
def test_throughput_column_contract(columns):
    table = Table([[1000., 2000., 3000.], [.1, .2, .3]], names=columns)
    np.testing.assert_allclose(throughput_interpolator(table)([1500., 2500.]),
                               [.15, .25])
    wave, response = throughput_arrays(table)
    np.testing.assert_array_equal(wave, [1000., 2000., 3000.])
    np.testing.assert_array_equal(response, [.1, .2, .3])


@pytest.mark.parametrize('wave,response', [
    ([0., 2.], [.1, .2]), ([2., 1.], [.1, .2]),
    ([1., 2.], [-.1, .2]), ([1., 2.], [.1, 1.1]),
    ([1., 2.], [.1, np.nan])])
def test_throughput_arrays_reject_invalid_fractional_response(wave, response):
    with pytest.raises(ValueError, match='Throughput columns'):
        throughput_arrays(Table([wave, response], names=['Wavelength', 'Efficiency']))


def test_band_masks_require_all_three_bands():
    valid, invalid = combine_band_masks([
        ([True, False, True], [False, True, False]),
        ([True, True, False], [False, False, True]),
        ([True, True, True], [False, False, False])], 3)
    np.testing.assert_array_equal(valid, [True, False, False])
    np.testing.assert_array_equal(invalid, [False, True, True])
    with pytest.raises(ValueError, match='per input ID'):
        combine_band_masks([([True], [False])], 3)


def test_spectral_padding_empty_all_invalid_and_row_alignment():
    assert pad_spectral_rows([], 0, 7).shape == (0, 7)
    invalid = pad_spectral_rows([np.nan, np.full(7, np.nan)], 2, 7)
    assert invalid.shape == (2, 7) and np.isnan(invalid).all()
    padded = pad_spectral_rows([np.array([1., 2.]), np.nan], 2, 3)
    np.testing.assert_array_equal(padded[0, :2], [1., 2.])
    assert np.isnan(padded[0, 2]) and np.isnan(padded[1]).all()
    with pytest.raises(ValueError, match='rows for'):
        pad_spectral_rows([np.nan], 2, 7)


def test_atomic_output_no_overwrite_and_exception_cleanup(tmp_path):
    destination = tmp_path / 'product.hdf5'
    with pytest.raises(RuntimeError, match='simulated'):
        with atomic_hdf5(destination) as handle:
            handle.create_dataset('unfinished', data=[1])
            raise RuntimeError('simulated failure')
    assert not destination.exists() and not list(tmp_path.iterdir())
    with atomic_hdf5(destination) as handle:
        handle.create_dataset('ID', data=[7])
    with pytest.raises(FileExistsError):
        output_destination(tmp_path, destination.name)
    with pytest.raises(FileExistsError):
        with atomic_hdf5(destination):
            pass
    with h5py.File(str(destination), 'r') as handle:
        np.testing.assert_array_equal(handle['ID'][:], [7])


def test_atomic_publish_race_does_not_replace_winner(tmp_path):
    destination = tmp_path / 'race.hdf5'
    with pytest.raises(FileExistsError):
        with atomic_hdf5(destination) as first:
            first.create_dataset('winner', data=1)
            with atomic_hdf5(destination) as second:
                second.create_dataset('winner', data=2)
    with h5py.File(str(destination), 'r') as handle:
        assert handle['winner'][()] == 2
    assert len(list(tmp_path.iterdir())) == 1




def empty_peak_table(count_name):
    return np.empty(0, dtype=[('id', 'i8'), ('wave', object),
                              ('idx', object), (count_name, 'i8')])


def test_singleton_ragged_rows_and_empty_el_diagnostics(tmp_path):
    path = tmp_path / 'diagnostics.hdf5'
    with atomic_hdf5(path) as handle:
        wave = handle.create_dataset('wave', (1,), dtype=h5py.vlen_dtype(np.dtype('f8')))
        write_vlen_rows(wave, [np.array([5007.])])
        details = {'candidates': empty_peak_table('candidate_el_number'),
                   'missed': empty_peak_table('missed_el_number'),
                   'false_positive': empty_peak_table('false_el_number'),
                   'pairs': np.empty(0, dtype=[('id', 'i8'), ('delta_wave', 'f8')])}
        write_el_details(handle.create_group('GV'), details)
        handle.create_dataset('parameters_desi', data=np.zeros((11, 11)))
        handle.create_dataset('parameters_phot', data=np.zeros((1, 21)))
        record_runner_provenance(handle, {'expt': 150}, {
            'apply_morphology_effect': False,
            'apply_wavelength_calibration_error': False,
            'apply_photometric': False, 'apply_el_detect': True})
    with h5py.File(str(path), 'r') as handle:
        np.testing.assert_array_equal(handle['wave'][0], [5007.])
        assert handle['GV/el_diagnostics/pairs'].shape == (0,)
        assert handle['parameters_desi'].attrs['axis_order'] == 'source,field'
        assert handle['parameters_phot'].attrs['axis_order'] == 'source,field'
        assert not handle.attrs['wce_applied']
        assert json.loads(handle.attrs['config_json']) == {'expt': 150}


def test_el_postprocess_does_not_invent_input_noise_provenance(tmp_path):
    source_path, target = tmp_path / 'legacy.hdf5', tmp_path / 'new.hdf5'
    with h5py.File(str(source_path), 'w'):
        pass
    with h5py.File(str(source_path), 'r') as source:
        with atomic_hdf5(target) as handle:
            record_runner_provenance(handle, {}, {}, source=source)
    with h5py.File(str(target), 'r') as handle:
        assert handle.attrs['cess_repair_revision'] == '20261024-D4'
        assert handle.attrs['noise_model'].startswith('unknown')
        assert handle.attrs['wce_model'].startswith('unknown')
        for name in ('spectral_response_algorithm', 'spectral_response_quantity',
                     'spectral_flux_calibration', 'spectral_sky_integration',
                     'template_lsf_nominal_evidence_json',
                     'spectral_convolution_algorithm', 'spectral_quantity',
                     'spectral_convolution_validation', 'template_lsf_json',
                     'template_lsf_broader_policy', 'spectral_target_lsf_json',
                     'spectral_final_lsf_definition', 'spectral_throughput_approximation'):
            assert handle.attrs[name].startswith('unknown')


@pytest.mark.parametrize('module', ['cess.pipeline.standard', 'cess.pipeline.paired'])
def test_zero_source_output_keeps_spectral_axes(tmp_path, monkeypatch, module):
    monkeypatch.setenv('PYTHONPATH', str(ROOT))
    path = run_fixture(tmp_path, module=module, count=0)
    with h5py.File(str(path), 'r') as handle:
        assert handle['parameters_desi'].shape == (0, 11)
        assert handle['data_mask'].shape == (0,)
        for band in ('GU', 'GV', 'GI'):
            width = len(handle[band + '/wave'])
            for name in ('flux_ujy', 'flux_ujy_with_noise', 'ferr', 'snr'):
                assert handle[band + '/' + name].shape == (0, width)
        assert handle.attrs['cess_repair_revision'] == '20261024-D4'


def test_paired_invalid_wce_rows_align(tmp_path, monkeypatch):
    monkeypatch.setenv('PYTHONPATH', str(ROOT))
    path = run_fixture(tmp_path, module='cess.pipeline.paired',
                       switches=('wave_cal=true', 'el_detect=true'),
                       count=2, invalid=(0,))
    with h5py.File(str(path), 'r') as handle:
        assert np.isnan(handle['parameters_desi'][0, 6:10]).all()
        for name in ('ferr_wce', 'snr_wce'):
            assert handle['GI/' + name].shape == (2, len(handle['GI/wave']))
            assert np.isnan(handle['GI/' + name][0]).all()
        assert 'el_diagnostics' in handle['GV']


def test_source_selection_slices_declared_axes_before_loading(tmp_path):
    path = tmp_path / 'input.hdf5'
    with h5py.File(str(path), 'w') as handle:
        handle['ID'] = np.array([b'a', b'b', b'c'])
        handle['parameters'] = np.arange(12).reshape(3, 4)
        handle['parameters_name'] = np.array([b'a', b'b', b'c', b'd'])
        handle['input_row_indices'] = [10, 20, 30]
        band = handle.create_group('GV')
        band['wave'] = [3800., 5200., 6600.]
        band['intri_el_id'] = [2]
        band['response_flux_ujy'] = np.arange(9).reshape(3, 3)
        band['source_electron_rate_pre_morph'] = np.arange(9).reshape(3, 3)
        band['sky_rate_e_s_pixel'] = [1., 2., 3.]
        band['response_at_pixel_center'] = [.1, .2, .3]
        model = handle.create_group('best_fit')
        model['wavelength_rest'] = [1., 2., 3.]
        model['z'] = [.1, .2, .3]
        model['spec_model'] = np.arange(9).reshape(3, 3)
    with h5py.File(str(path), 'r') as handle:
        view = SourceSelection(handle, 1, 3)
        spec = view['best_fit']['spec_model']
        assert spec._data is None
        assert spec.shape == (2, 3) and spec._data is None
        np.testing.assert_array_equal(spec[:], [[3, 4, 5], [6, 7, 8]])
        np.testing.assert_array_equal(view['ID'][:], [b'b', b'c'])
        np.testing.assert_array_equal(view['input_row_indices'][:], [20, 30])
        # Sparse EL tables have a peak/source subset axis, not the root N axis.
        np.testing.assert_array_equal(view['GV']['intri_el_id'][:], [2])
        np.testing.assert_array_equal(view['GV']['response_flux_ujy'][:], [[3, 4, 5], [6, 7, 8]])
        np.testing.assert_array_equal(view['GV']['source_electron_rate_pre_morph'][:], [[3, 4, 5], [6, 7, 8]])
        np.testing.assert_array_equal(view['GV']['sky_rate_e_s_pixel'][:], [2., 3.])
        np.testing.assert_array_equal(view['GV']['response_at_pixel_center'][:], [.1, .2, .3])
        np.testing.assert_array_equal(view['best_fit']['wavelength_rest'][:], [1., 2., 3.])
        assert view['best_fit']['spec_model'] is spec
        np.testing.assert_array_equal(view.source_rows, [1, 2])
        assert '_rows1-3_' in simulation_filename('input.hdf5', view)
        with pytest.raises(ValueError, match='Source bounds'):
            SourceSelection(handle, -1, 2)


def test_resource_cache_reuses_only_unchanged_resources(tmp_path):
    import pickle
    table_path = tmp_path / 'response.fits'
    Table([[1000., 2000.], [.1, .2]], names=['Wavelength', 'Efficiency']).write(
        str(table_path), format='fits')
    first = cached_table(table_path)
    assert cached_table(table_path) is first
    library_path = tmp_path / 'profiles.pkl'
    with library_path.open('wb') as handle:
        pickle.dump([1, 2, 3], handle)
    library = cached_pickle(library_path)
    assert cached_pickle(library_path) is library


def test_runner_imports_ignore_argv_and_do_not_open_resources(tmp_path):
    import os
    import subprocess
    code = """
import sys
import builtins
from pathlib import Path
sys.path[:0] = [CORE, ROOT]
sys.argv = ['unrelated-program', '--not-a-CESS-argument']
old_open = builtins.open
def checked_open(file, *args, **kwargs):
    path = str(file)
    if path.endswith(('.fits', '.pkl', 'emulator_parameters.json')):
        raise AssertionError('import tried to read a scientific resource: ' + path)
    return old_open(file, *args, **kwargs)
builtins.open = checked_open
import cess.pipeline.standard as run
import cess.pipeline.paired as run_gravity_zouhu
import cess.pipeline.emission as run_el_detect
assert run._CONFIGURATION is None
assert run_gravity_zouhu._CONFIGURATION is None
assert run_el_detect._CONFIGURATION is None
""".replace('CORE, ROOT', repr(str(CORE)) + ', ' + repr(str(ROOT)))
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=str(tmp_path),
        env=dict(os.environ, MPLCONFIGDIR=str(tmp_path / 'mpl'), PYTHONDONTWRITEBYTECODE='1'),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
        timeout=30)
    assert result.returncode == 0, result.stdout


def test_configure_is_explicit_and_keeps_fits_and_pickle_lazy(tmp_path, monkeypatch):
    import cess.pipeline.standard as run
    from cess.config.loader import load_config
    parameters = load_config(CORE / 'cess/data/emulator_parameters.json')
    parameters['output_path'] = str(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('configure loaded a scientific resource')
    monkeypatch.setattr(Table, 'read', forbidden)
    result = run.configure(parameters, {'morph': True, 'photo': False})
    assert result['output_path'] == str(tmp_path)
    assert run.apply_morphology_effect and not run.apply_photometric


@pytest.mark.parametrize('module_name', ['cess.pipeline.standard', 'cess.pipeline.paired'])
def test_per_source_seed_is_independent_of_chunk_boundaries(tmp_path, module_name):
    import importlib
    from cess.config.loader import load_config
    module = importlib.import_module(module_name)
    input_dir = tmp_path / 'input'
    input_dir.mkdir()
    make_input(input_dir / 'synthetic.hdf5', count=4, invalid=(0,))
    parameters = load_config(CORE / 'cess/data/emulator_parameters.json')
    parameters['output_path'] = str(tmp_path / 'whole')
    parameters['photometry_aperture_fraction'] = .5
    options = {'morph': True, 'wave_cal': True, 'photo': True, 'el_detect': False}
    whole = module.task(str(input_dir), 'synthetic.hdf5', config=parameters,
                         options=options, seed=8675309)
    chunks = [module.task(str(input_dir), 'synthetic.hdf5',
               output_dir=tmp_path / 'chunks', config=parameters, options=options,
               seed=8675309, source_start=start, source_stop=stop)
              for start, stop in [(0, 2), (2, 4)]]
    overlapping = module.task(str(input_dir), 'synthetic.hdf5',
        output_dir=tmp_path / 'chunks', config=parameters, options=options,
        seed=8675309, source_start=1, source_stop=3)
    assert len(set(chunks + [overlapping])) == 3
    with h5py.File(str(whole), 'r') as full:
        assert full.attrs['schema_version'] == '2.5'
        assert full['parameters_phot'].shape == (4, 21)
        assert full['photometry/nuv/valid'].shape == (4,)
        assert full['photometry/nuv/model_flux_ujy'].attrs['unit'] == 'uJy'
        assert full['photometry/r/aperture_fraction'][1] == .5
        assert full['GU/spec_mask'].dtype == np.dtype(bool)
        for band in ('GU', 'GV', 'GI'):
            centers = full[band + '/wave'][:]
            edges = full[band + '/wave_edges'][:]
            assert edges.shape == (centers.size + 1,)
            np.testing.assert_allclose(edges[1:-1], (centers[:-1] + centers[1:]) / 2.)
            np.testing.assert_allclose(full[band + '/pixel_width'][:], np.diff(edges))
            assert full[band + '/wave_edges'].attrs['unit'] == 'Angstrom'
        for path in ('parameters_desi', 'GU/flux_ujy_with_noise',
                     'GI/ferr', 'GI/snr', 'photometry/r/observed_flux_ujy'):
            rows = []
            for chunk in chunks:
                with h5py.File(str(chunk), 'r') as part:
                    rows.append(part[path][:])
                    assert part.attrs['input_sha256'] == full.attrs['input_sha256']
            np.testing.assert_array_equal(np.concatenate(rows), full[path][:])
        with h5py.File(str(overlapping), 'r') as part:
            np.testing.assert_array_equal(part['input_row_indices'][:], [1, 2])
            np.testing.assert_array_equal(part['parameters_desi'][:], full['parameters_desi'][1:3])
        if module_name == 'cess.pipeline.standard':
            assert full['zero_order_phot'].shape == (4, 9)
            assert full['zero_order/GU/total_exposure_s'][1] == 300.
            assert full['zero_order/GU/aperture_fraction'][1] == .5


def test_nondefault_aperture_fraction_counts_and_flux_calibration():
    from cess.config.loader import load_config
    from cess.engine import emulator
    parameters = load_config(CORE / 'cess/data/emulator_parameters.json')
    parameters['photometry_aperture_fraction'] = .5
    engine = emulator(config=parameters)
    wave = np.linspace(1000., 10000., 51)
    flux = np.full(wave.size, 3.e-29)
    response = Table([[2500., 4000., 5000.], [.2, .2, .2]],
                     names=['Wavelength', 'Efficiency'])
    background = Table([[1000., 10000.], [0., 0.]], names=['wavelength', 'fnu'])
    half = measure_runner_band(engine, 0., 0., wave, flux, response, background,
                              parameters, 2, 4., rng=np.random.RandomState(12))
    full_parameters = dict(parameters, photometry_aperture_fraction=1.)
    full = measure_runner_band(engine, 0., 0., wave, flux, response, background,
                              full_parameters, 2, 4., rng=np.random.RandomState(12))
    assert half['aperture_fraction'] == .5
    np.testing.assert_allclose(half['source_counts_e'], .5 * full['source_counts_e'])
    np.testing.assert_allclose(half['model_flux_ujy'], 3.)
    np.testing.assert_allclose(full['model_flux_ujy'], 3.)
    conversion = half['source_counts_e'] / half['model_flux_ujy']
    np.testing.assert_allclose(half['observed_flux_ujy'], half['observed_counts_e'] / conversion)


def test_el_postprocessing_preserves_catalogue_and_parent_row_indices(tmp_path):
    import cess.pipeline.emission as run_el_detect
    from cess.config.loader import load_config
    path = tmp_path / 'parent.hdf5'
    with h5py.File(str(path), 'w') as handle:
        handle['ID'] = np.array([b'a', b'b', b'c'])
        handle['input_row_indices'] = [10, 20, 30]
        for band in ('GV', 'GI'):
            group = handle.create_group(band)
            group['wave'] = [4000., 5000., 6000.]
            for name in ('flux_ujy', 'flux_ujy_with_noise', 'snr'):
                group[name] = np.zeros((3, 3))
    parameters = load_config(CORE / 'cess/data/emulator_parameters.json')
    output = run_el_detect.task(str(tmp_path), path.name, output_dir=tmp_path / 'el',
        config=parameters, source_start=1, source_stop=3)
    with h5py.File(str(output), 'r') as handle:
        np.testing.assert_array_equal(handle['ID'][:], [b'b', b'c'])
        np.testing.assert_array_equal(handle['parent_row_indices'][:], [1, 2])
        np.testing.assert_array_equal(handle['input_row_indices'][:], [20, 30])


def test_photon_bin_metadata_and_exact_center_derived_edges(tmp_path):
    from cess.config.loader import load_config
    parameters = load_config(CORE / 'cess/data/emulator_parameters.json')
    parameters['template_lsf'] = {'mode': 'fwhm_rest', 'value': 2.}
    parameters['template_lsf_broader_policy'] = 'preserve'
    parameters['template_lsf_nominal_evidence'] = {
        'resolving_power': 300., 'lsf_shape': 'unverified'}
    target = tmp_path / 'bins.hdf5'
    with atomic_hdf5(target) as handle:
        handle['ID'] = [1]
        band = handle.create_group('GU')
        band['wave'] = [4000., 4010., 4030.]
        band['flux_ujy'] = np.ones((1, 3))
        band['flux_elec'] = np.ones((1, 3))
        band['response_flux_ujy'] = np.ones((1, 3))
        band['flux_ujy_model'] = np.ones((1, 3))
        record_runner_provenance(handle, parameters, {
            'apply_morphology_effect': False,
            'apply_wavelength_calibration_error': False,
            'apply_photometric': False, 'apply_el_detect': False})
    with h5py.File(str(target), 'r') as handle:
        assert handle.attrs['cess_repair_revision'] == '20261024-D4'
        assert handle.attrs['schema_version'] == '2.5'
        assert handle.attrs['spectral_convolution_algorithm'] == 'cess-photon-bin-gauss-v1'
        assert 'fnu_eff=lambda_center*Q/pixel_width' in handle.attrs['spectral_quantity']
        assert 'synthetic' in handle.attrs['spectral_convolution_validation']
        assert json.loads(handle.attrs['template_lsf_json']) == {'mode': 'fwhm_rest', 'value': 2.}
        assert handle.attrs['template_lsf_broader_policy'] == 'preserve'
        assert json.loads(handle.attrs['template_lsf_nominal_evidence_json']) == parameters['template_lsf_nominal_evidence']
        target_lsf = json.loads(handle.attrs['spectral_target_lsf_json'])
        assert target_lsf['family'] == 'Gaussian' and target_lsf['stage'] == 'pre-pixel'
        assert 'before morphology' in handle.attrs['spectral_throughput_approximation']
        np.testing.assert_array_equal(handle['GU/wave_edges'][:], [3995., 4005., 4020., 4040.])
        np.testing.assert_array_equal(handle['GU/pixel_width'][:], [10., 15., 20.])
        assert handle['GU/flux_ujy'].attrs['unit'] == 'uJy'
        assert handle['GU/flux_ujy'].attrs['spectral_quantity'] == handle.attrs['spectral_quantity']
        assert handle['GU/response_flux_ujy'].attrs['spectral_quantity'] == handle.attrs['spectral_response_quantity']
        assert handle['GU/flux_ujy_model'].attrs['spectral_quantity'] == handle.attrs['spectral_flux_calibration']
        assert handle.attrs['spectral_response_algorithm'] == 'cess-native-response-photon-bin-v1'


def test_el_postprocessing_inherits_parent_spectral_semantics_and_future_notes(tmp_path):
    source_path, target = tmp_path / 'parent.hdf5', tmp_path / 'el.hdf5'
    inherited = {
        'spectral_convolution_algorithm': 'parent-generation-operator',
        'spectral_quantity': 'parent-defined density',
        'spectral_convolution_validation': 'parent synthetic checks only',
        'template_lsf_json': json.dumps({'mode': 'resolving_power', 'value': 300.}),
        'template_lsf_broader_policy': 'preserve',
        'spectral_target_lsf_json': '{"parent": true}',
        'spectral_final_lsf_definition': 'parent conditional LSF',
        'spectral_throughput_approximation': 'parent approximation',
        'spectral_response_algorithm': 'parent-response-operator',
        'spectral_flux_calibration': 'parent flux calibration',
        'spectral_sky_integration': 'parent sky prescription',
        'template_lsf_nominal_evidence_json': '{"resolving_power": 300}',
        'spectral_future_note': 'preserve even an unrecognised future spectral attribute',
    }
    with h5py.File(str(source_path), 'w') as handle:
        for name, value in inherited.items():
            handle.attrs[name] = value
    # The current config deliberately conflicts with the parent; no convolution
    # runs during postprocessing, so it cannot define the input's spectral model.
    with h5py.File(str(source_path), 'r') as source:
        with atomic_hdf5(target) as handle:
            record_runner_provenance(handle, {'template_lsf': {'mode': 'delta'}}, {}, source=source)
    with h5py.File(str(target), 'r') as handle:
        for name, value in inherited.items():
            assert handle.attrs[name] == value
        assert handle.attrs['cess_repair_revision'] == '20261024-D4'


def test_response_products_keep_detector_counts_separate_from_calibration(tmp_path):
    from cess.engine import h
    parameters = dict(radi=1., expt=10., expnum=2)
    path = tmp_path / 'response.hdf5'
    with atomic_hdf5(path) as handle:
        group = handle.create_group('GU')
        group['wave'] = [4000., 4010., 4020.]
        for suffix in ('', '_0th'):
            group['flux_elec' + suffix] = [[2., 3., np.nan]]
            group['ferr' + suffix] = [[np.nan, 1., np.nan]]
            write_response_products(group, np.array([[.1, .2, np.nan]]),
                np.array([[.01, .02, np.nan]]), [0., .5, .5],
                7., [14.], parameters, suffix=suffix)
        expected = 3. * h * 4010. / (np.pi * 20. * 10. * .5) / 1.e-29
        assert group['flux_ujy_model'][0, 1] == pytest.approx(expected)
        assert np.isnan(group['flux_ujy_model'][0, 0])
        np.testing.assert_array_equal(group['detector_model_mask'][:], [[True, True, False]])
        np.testing.assert_array_equal(group['calibration_support_mask'][:], [[False, True, False]])
        assert group['source_electron_rate_pre_morph'].attrs['unit'] == 'electron s^-1 cm^-2'
        assert group['sky_rate_e_s_pixel'].attrs['reference_rate_e_s_pixel'] == 7.
        assert 'diagnostic proxy' in group['flux_elec_0th'].attrs['product_role']
        assert group['flux_elec_0th'].attrs['n_exposures'] == 2


@pytest.mark.parametrize('module_name', ['cess.pipeline.standard', 'cess.pipeline.paired'])
def test_runner_response_rates_counts_native_sky_and_calibrated_model(tmp_path, module_name):
    import importlib
    from cess.config.loader import load_config
    from cess.engine import emulator, h
    parameters = load_config(CORE / 'cess/data/emulator_parameters.json')
    input_dir = tmp_path / 'input'
    input_dir.mkdir()
    make_input(input_dir / 'synthetic.hdf5', count=2, invalid=(0,))
    parameters['output_path'] = str(tmp_path / 'out')
    options = dict(morph=False, wave_cal=False, photo=False, el_detect=False)
    path = importlib.import_module(module_name).task(str(input_dir), 'synthetic.hdf5',
        config=parameters, options=options, seed=54321)
    engine = emulator(config=parameters)
    background = cached_table(parameters['bkg_spec_path'])
    area_time = np.pi * parameters['radi'] ** 2 * parameters['expt'] * parameters['expnum']
    with h5py.File(str(path), 'r') as handle:
        assert handle.attrs['schema_version'] == '2.5'
        for band in ('GU', 'GV', 'GI'):
            group = handle[band]
            suffixes = ('', '_0th') if module_name == 'cess.pipeline.standard' else ('',)
            for suffix in suffixes:
                counts = group['flux_elec' + suffix][:]
                rate = group['source_electron_rate_pre_morph' + suffix][:]
                assert counts.shape == rate.shape == group['flux_ujy'].shape
                assert np.isnan(rate[0]).all()
                np.testing.assert_allclose(counts[1], rate[1] * area_time, rtol=2.e-13)
                wave = group['wave'][:]
                folded = group['response_flux_ujy' + suffix][:]
                expected_rate = folded * 1.e-29 * group['pixel_width'][:] / (h * wave)
                np.testing.assert_allclose(rate, expected_rate, rtol=2.e-13)
                response = group['response_at_pixel_center' + suffix][:]
                denominator = area_time * group['pixel_width'][:] * response
                expected_model = np.full_like(counts, np.nan)
                np.divide(counts * h * wave / 1.e-29, denominator,
                    out=expected_model, where=np.isfinite(counts) & (denominator > 0))
                np.testing.assert_allclose(group['flux_ujy_model' + suffix][:], expected_model)
                key = band.lower() + ('tp0_path' if suffix else 'tp_path')
                native_wave, native_response = throughput_arrays(cached_table(parameters[key]))
                sky = engine.native_sky_rate(120., 20., background['wavelength'],
                    background['fnu'], native_wave, native_response)
                assert group['sky_rate_e_s_pixel' + suffix][1] == pytest.approx(sky)
                np.testing.assert_array_equal(group['detector_model_mask' + suffix][:], np.isfinite(counts))
                np.testing.assert_array_equal(group['calibration_support_mask' + suffix][:],
                    np.isfinite(group['ferr' + suffix][:]))
            if module_name == 'cess.pipeline.paired':
                np.testing.assert_array_equal(group['flux_ujy_model'][:], group['flux_ujy_model_wce'][:])
                np.testing.assert_array_equal(group['detector_model_mask'][:], group['detector_model_mask_wce'][:])
