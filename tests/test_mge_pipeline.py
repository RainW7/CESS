"""C12 opt-in strip extraction: source dose, row provenance and inheritance."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / 'src'
sys.path[:0] = [str(CORE), str(ROOT)]

from cess.config.loader import load_config, validate_config
from cess.cli.main import build_jobs
from cess.io.manifest import snapshot, verify_product
from cess.morphology.generator import generate_mge_library
from cess.morphology.runtime import load_mge_source_library, select_mge_extraction
from cess.engine import emulator, h
from cess.pipeline.support import MGE_MORPHOLOGY_FIELDS, SCHEMA_VERSION, REPAIR_REVISION
from test_pipeline_smoke import make_input


def strip(convention='point_source', pixels=3, radius=0):
    return dict(mode='mge_strip', spatial_pixels=pixels, kernel_radius_pixels=radius,
                lsf_convention=convention)


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    path = tmp_path_factory.mktemp('c12-source-basis') / 'basis'
    generate_mge_library(load_config(CORE / 'cess/data/emulator_parameters.json'),
        dict(n=[.5], re_pixel=[4.], pa_deg=[30.], axis_ratio=[.6]),
        [-10., -1., 0., 1., 10.], path)
    return path


def settings(bundle, design=None):
    config = load_config(CORE / 'cess/data/emulator_parameters.json')
    config.pop('widthlib_pkl')
    config.pop('heightlib_pkl')
    config.update(positive_profile_library_manifest_path=str(bundle / 'manifest.json'),
                  positive_profile_library_path=str(bundle / 'marginals.hdf5'),
                  spectral_extraction=strip() if design is None else design)
    return config


@pytest.mark.parametrize('change', [
    {'lsf_convention': None}, {'lsf_convention': 'implicit'}, {'source_fraction': .8},
    {'spatial_pixels': 0}, {'spatial_pixels': 1002}, {'spatial_pixels': True},
    {'kernel_radius_pixels': -1}, {'kernel_radius_pixels': 1025}, {'kernel_radius_pixels': 1.0},
])
def test_mge_configuration_requires_explicit_bounded_model(bundle, change):
    design = strip()
    design.update(change)
    if change == {'lsf_convention': None}:
        design.pop('lsf_convention')
    with pytest.raises(ValueError):
        validate_config(settings(bundle, design))


@pytest.mark.parametrize('problem', ['missing_manifest', 'missing_basis', 'legacy_mode', 'c7_manifest'])
def test_positive_paths_are_explicit_paired_and_exclusive(bundle, problem):
    config = settings(bundle)
    if problem == 'missing_manifest':
        config.pop('positive_profile_library_manifest_path')
    elif problem == 'missing_basis':
        config.pop('positive_profile_library_path')
    elif problem == 'legacy_mode':
        config['spectral_extraction'] = {'mode': 'legacy_full'}
    else:
        config['profile_library_manifest_path'] = str(bundle / 'manifest.json')
    with pytest.raises(ValueError):
        validate_config(config)


def run_tiny(tmp_path, bundle, monkeypatch, module='cess.pipeline.standard', design=None,
             morph=True, count=1, invalid=(), el=False, wave=False):
    import cess.morphology.empirical as morphology
    import cess.pipeline.support as runner_support
    monkeypatch.setattr(morphology, 'get_2d_param', lambda *args: (.5, 4., -30., .6))
    monkeypatch.setattr(runner_support, 'cached_pickle',
                        lambda *args: pytest.fail('mge_strip must not load legacy pickles'))
    source = tmp_path / 'synthetic.hdf5'
    make_input(source, count=count, invalid=invalid)
    config = settings(bundle, design)
    options = dict(morph=morph, wave_cal=wave, photo=False, el_detect=el)
    destination = tmp_path / 'products'
    product = Path(importlib.import_module(module).task(str(source.parent), source.name,
        output_dir=destination, config=config, options=options, seed=1214))
    mode = 'standard' if module == 'cess.pipeline.standard' else 'paired-wce'
    job = build_jobs([source], destination, config, options, mode, seed=1214)[0]
    state = snapshot(config, options, mode, 1214, [job])
    result = verify_product(job, state)
    return product, config, options, job, state, result


@pytest.mark.parametrize('module', ['cess.pipeline.standard', 'cess.pipeline.paired'])
@pytest.mark.parametrize('convention', ['point_source', 'pre_psf'])
def test_joint_fraction_and_finite_kernel_are_applied_once_to_counts_and_noise(tmp_path, bundle, monkeypatch, module, convention):
    name = 'add_noise' if module == 'cess.pipeline.standard' else 'add_noise_zouhu'
    original = getattr(emulator, name)
    calls = []

    def capture(self, *args, **kwargs):
        assert kwargs['normalize_morphology_kernel'] is False
        calls.append(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(emulator, name, capture)
    product, config, options, job, state, result = run_tiny(tmp_path, bundle, monkeypatch,
        module=module, design=strip(convention))
    assert len(calls) == (6 if module == 'cess.pipeline.standard' else 3)
    selected = select_mge_extraction(load_mge_source_library(validate_config(config)),
        .5, 4., -30., .6, validate_config(config), apply_morphology=True)
    fraction, mass = selected['source_fraction'], float(selected['width'].sum())
    assert 0. < fraction < 1. and 0. < mass < 1.
    assert all(call['source_fraction'] == fraction and call['extraction_n_pixels'] == 3 for call in calls)
    with h5py.File(str(product), 'r') as handle:
        assert handle.attrs['schema_version'] == SCHEMA_VERSION == '2.5'
        assert handle.attrs['cess_repair_revision'] == REPAIR_REVISION == '20261024-D4'
        description = json.loads(handle.attrs['profile_library_provenance_json'])
        assert description['version'] == 3 and description['provenance_status'] == 'positive_source_basis_verified'
        group = handle['mge_morphology']
        assert set(group) == set(MGE_MORPHOLOGY_FIELDS)
        np.testing.assert_array_equal(group['selected_parameters'][:], [[.5, 4., 30., .6]])
        np.testing.assert_array_equal(group['selected_indices'][:], [[0., 0., 0., 0.]])
        assert handle['parameters_desi'][0, 8] == -30.  # Original sample is preserved.
        assert group['kernel_window_fraction'][0] == pytest.approx(mass, abs=2.e-15)
        assert group['kernel_absolute_fraction'][0] == pytest.approx(fraction * mass, abs=2.e-15)
        assert group['kernel_n_bins'][0] == 1. and group['kernel_applied'][0]
        assert handle['spectral_extraction/source_fraction'][0] == fraction
        assert handle['spectral_extraction/spatial_pixels'][0] == 3.
        total_time = config['expt'] * config['expnum']
        area = np.pi * config['radi'] ** 2
        for band in ('GU', 'GV', 'GI'):
            data = handle[band]
            expected = data['source_electron_rate_pre_morph'][0] * area * total_time * fraction * mass
            actual = data['flux_elec'][0]
            valid = np.isfinite(actual) & np.isfinite(expected)
            assert valid.any()
            np.testing.assert_allclose(actual[valid], expected[valid], rtol=3.e-13, atol=0.)
            wave, width, response = [data[key][:] for key in ('wave', 'pixel_width', 'response_at_pixel_center')]
            conversion = np.full_like(wave, np.nan)
            np.divide(h * wave / 1.e-29, area * total_time * width * response,
                      out=conversion, where=response > 0.)
            background = (data['sky_rate_e_s_pixel'][0] + config['dc_value']) * total_time
            background += config['read_noise'] ** 2 * config['expnum']
            expected_variance = (expected + 3 * background) * conversion ** 2
            support = np.isfinite(expected_variance) & np.isfinite(data['ferr'][0])
            np.testing.assert_allclose(data['ferr'][0][support] ** 2,
                                       expected_variance[support], rtol=3.e-12, atol=0.)
    before = (product.read_bytes(), product.stat().st_mtime_ns)
    assert verify_product(job, state, expected_sha256=result['sha256']) == result
    assert before == (product.read_bytes(), product.stat().st_mtime_ns)


def test_morphology_disabled_preserves_strip_fraction_and_noise_area_and_rng(tmp_path, bundle, monkeypatch):
    before = np.random.get_state()
    product, config, _, _, _, _ = run_tiny(tmp_path, bundle, monkeypatch, morph=False,
                                         design=strip(radius=16))
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    with h5py.File(str(product), 'r') as handle:
        fraction = handle['spectral_extraction/source_fraction'][0]
        assert 0. < fraction < 1.
        group = handle['mge_morphology']
        assert group['valid'][0] and not group['kernel_applied'][0]
        assert group['kernel_window_fraction'][0] == 1.
        assert group['kernel_n_bins'][0] == 1.
        assert group['quadrature_error_estimate'][0] == 0.
        assert group['omitted_probability_bound'][0] == 0.
        for band in ('GU', 'GV', 'GI'):
            expected = handle[band + '/source_electron_rate_pre_morph'][0] * np.pi * config['radi'] ** 2
            expected *= config['expt'] * config['expnum'] * fraction
            actual = handle[band + '/flux_elec'][0]
            use = np.isfinite(actual) & np.isfinite(expected)
            np.testing.assert_allclose(actual[use], expected[use], rtol=3.e-13, atol=0.)


@pytest.mark.parametrize('count,invalid', [(0, ()), (2, (0, 1))])
def test_empty_and_invalid_sources_have_aligned_nan_metadata(tmp_path, bundle, monkeypatch, count, invalid):
    product, _, _, _, _, _ = run_tiny(tmp_path, bundle, monkeypatch, count=count, invalid=invalid)
    with h5py.File(str(product), 'r') as handle:
        for name in MGE_MORPHOLOGY_FIELDS:
            data = handle['mge_morphology/' + name]
            if name in ('valid', 'kernel_applied'):
                assert data.dtype == np.dtype(bool) and not data[:].any()
            else:
                assert data.dtype == np.dtype(float) and np.isnan(data[:]).all()
            assert data.shape == ((count, 4) if name in ('selected_parameters', 'selected_indices') else (count,))


def test_paired_wce_and_el_subset_inherit_parent_model_with_legacy_el_config(tmp_path, bundle, monkeypatch):
    monkeypatch.setattr(emulator, '_sample_wave_offset', staticmethod(lambda rng=None: .5))
    product, config, _, _, _, _ = run_tiny(tmp_path, bundle, monkeypatch,
        module='cess.pipeline.paired', design=strip('pre_psf', radius=2),
        count=2, invalid=(1,), wave=True, el=True)
    import cess.pipeline.emission as run_el_detect
    el_config = load_config(CORE / 'cess/data/emulator_parameters.json')
    destination = tmp_path / 'el'
    child = run_el_detect.task(str(product.parent), product.name, output_dir=destination,
        config=el_config, seed=1214, source_start=0, source_stop=1)
    with h5py.File(str(product), 'r') as parent, h5py.File(str(child), 'r') as handle:
        assert handle.attrs['profile_library_provenance_json'] == parent.attrs['profile_library_provenance_json']
        for group in ('mge_morphology', 'spectral_extraction'):
            assert set(handle[group].attrs) == set(parent[group].attrs)
            for attr in parent[group].attrs:
                np.testing.assert_array_equal(handle[group].attrs[attr], parent[group].attrs[attr])
            for name in parent[group]:
                np.testing.assert_array_equal(handle[group + '/' + name][:], parent[group + '/' + name][:1])
                assert dict(handle[group + '/' + name].attrs) == dict(parent[group + '/' + name].attrs)
        for band in ('GU', 'GV', 'GI'):
            assert np.isfinite(parent[band + '/ferr_wce'][0]).any()
            assert np.isnan(parent[band + '/ferr_wce'][1]).all()
    options = dict(el_detect=True)
    job = build_jobs([product], destination, el_config, options, 'el-detect', seed=1214,
                     source_start=0, source_stop=1)[0]
    state = snapshot(el_config, options, 'el-detect', 1214, [job])
    assert verify_product(job, state)['output'] == str(child)


@pytest.mark.parametrize('field', ['selected_indices', 'kernel_absolute_fraction', 'source_fit_l1_estimate'])
def test_resume_rejects_semantic_row_tampering(tmp_path, bundle, monkeypatch, field):
    product, _, _, job, state, _ = run_tiny(tmp_path, bundle, monkeypatch)
    with h5py.File(str(product), 'r+') as handle:
        data = handle['mge_morphology/' + field]
        if data.ndim == 2:
            data[0, 0] += 1.
        else:
            data[0] += .01
    with pytest.raises(ValueError):
        verify_product(job, state)


def test_resume_rejects_mge_model_metadata_tampering(tmp_path, bundle, monkeypatch):
    product, _, _, job, state, _ = run_tiny(tmp_path, bundle, monkeypatch)
    with h5py.File(str(product), 'r+') as handle:
        handle['mge_morphology'].attrs['selected_parameter_units'] = ['1', 'arcsec', 'degree', '1']
    with pytest.raises(ValueError, match='MGE model metadata'):
        verify_product(job, state)
