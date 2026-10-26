"""Explicit C7 runtime resources retain verified bytes and declared axes."""
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / 'CESS_v0.8.6'
sys.path[:0] = [str(CORE), str(ROOT)]

from cess_config import load_config, validate_config
from cess_manifest import snapshot, verify_product
from cess_cli import build_jobs
from cess_profile_generator import generate_profile_libraries
from cess_profiles import load_verified_profile_bundle, verify_profile_bundle_binding
from runner_support import load_profile_libraries
from test_pipeline_smoke import make_input

AXES = {'n': [1.], 're_pixel': [4.], 'pa_deg': [0., 45.], 'axis_ratio': [1.]}
LEGACY_SHAPE_AXES = {'n': np.arange(20.), 're_pixel': np.arange(20.),
                     'pa_deg': np.arange(10.), 'axis_ratio': np.arange(10.)}


@pytest.fixture(scope='module')
def generated(tmp_path_factory):
    path = tmp_path_factory.mktemp('runtime-bundle') / 'bundle'
    generate_profile_libraries(load_config(CORE / 'emulator_parameters.json'), AXES, path)
    return path


@pytest.fixture
def selected(tmp_path, generated):
    path = tmp_path / 'bundle'
    shutil.copytree(str(generated), str(path))
    config = load_config(CORE / 'emulator_parameters.json')
    config.update(profile_library_manifest_path=str(path / 'manifest.json'),
                  widthlib_pkl=str(path / 'widthlib.pkl'), heightlib_pkl=str(path / 'heightlib.pkl'))
    return path, config


def test_optional_configuration_resolves_three_files_without_changing_default(selected, tmp_path):
    path, config = selected
    assert 'profile_library_manifest_path' not in load_config(CORE / 'emulator_parameters.json')
    relative = dict(config, profile_library_manifest_path='bundle/manifest.json',
                    widthlib_pkl='bundle/widthlib.pkl', heightlib_pkl='bundle/heightlib.pkl')
    resolved = validate_config(relative, base_dir=tmp_path)
    for key in ('profile_library_manifest_path', 'widthlib_pkl', 'heightlib_pkl'):
        assert resolved[key] == config[key]
    with pytest.raises(ValueError, match='manifest.json'):
        validate_config(dict(config, profile_library_manifest_path=str(path / 'other.json')))
    with pytest.raises(ValueError):
        validate_config(dict(config, profile_library_manifest_path=None))


def test_verified_bytes_and_readonly_worker_cache_are_isolated(selected, monkeypatch):
    directory, config = selected
    import cess_profiles
    import runner_support
    original_loads = pickle.loads
    seen = []

    def loads(payload):
        seen.append(hashlib.sha256(payload).hexdigest())
        return original_loads(payload)

    monkeypatch.setattr(cess_profiles.pickle, 'loads', loads)
    monkeypatch.setattr(runner_support, 'cached_pickle',
                        lambda *args: pytest.fail('explicit bundle must not use legacy pickle cache'))
    width, height, axes, description = load_profile_libraries(config, LEGACY_SHAPE_AXES)
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert seen == [manifest['resources'][r]['sha256'] for r in ('width', 'height')]
    assert description['version'] == 2
    assert description['provenance_status'] == 'manifest_verified_generation'
    assert description['runtime_interpretation']['axes'] == AXES
    assert description['generation']['physical_validation_status'] == 'unvalidated_legacy_rendering_recipe'
    assert 'nearest endpoint' in description['runtime_interpretation']['lookup']['outside_grid']
    assert width.shape == height.shape == (1, 1, 2, 1)
    with pytest.raises(ValueError):
        width[0, 0, 0, 0][0] = 999.
    with pytest.raises(ValueError):
        height[0, 0, 0, 0] = np.asarray([999.])
    axes['n'][0] = 999.
    description['runtime_interpretation']['axes']['n'][0] = 999.
    width2, height2, axes2, description2 = load_profile_libraries(config, LEGACY_SHAPE_AXES)
    assert width2 is width and height2 is height
    assert len(seen) == 2
    assert axes2['n'][0] == 1.
    assert description2['runtime_interpretation']['axes'] == AXES


@pytest.mark.parametrize('problem', ['scale', 'width_path', 'height_path', 'pa_axis', 'hash', 'shape'])
def test_invalid_bindings_fail_without_silent_legacy_fallback(selected, problem):
    directory, config = selected
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if problem == 'scale':
        config['arcsecperpix'] *= 2.
    elif problem in ('width_path', 'height_path'):
        role = problem.split('_')[0]
        config[role + 'lib_pkl'] = load_config(CORE / 'emulator_parameters.json')[role + 'lib_pkl']
    elif problem == 'pa_axis':
        manifest['axes']['pa_deg'][0] = -1.
        manifest_path.write_text(json.dumps(manifest))
    elif problem == 'hash':
        with (directory / 'widthlib.pkl').open('ab') as stream:
            stream.write(b'corrupt')
    elif problem == 'shape':
        wrong = np.empty((2, 1, 2, 1), dtype=object)
        for index in np.ndindex(wrong.shape):
            wrong[index] = np.asarray([1.])
        payload = pickle.dumps(wrong, protocol=4)
        (directory / 'widthlib.pkl').write_bytes(payload)
        manifest['resources']['width'].update(sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload))
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_profile_libraries(config, LEGACY_SHAPE_AXES)


def test_legacy_tiny_grid_is_still_rejected_without_optin(selected):
    _, config = selected
    config.pop('profile_library_manifest_path')
    with pytest.raises(ValueError, match='shape must match axes'):
        load_profile_libraries(config, LEGACY_SHAPE_AXES)


def test_cache_hit_checks_real_bytes_even_when_size_and_mtime_are_preserved(selected):
    directory, config = selected
    _, _, _, description = load_profile_libraries(config, LEGACY_SHAPE_AXES)
    path = directory / 'widthlib.pkl'
    before = path.stat()
    library = pickle.loads(path.read_bytes())
    library[0, 0, 0, 0] = library[0, 0, 0, 0].copy()
    library[0, 0, 0, 0][0] *= 1.01
    payload = pickle.dumps(library, protocol=4)
    assert len(payload) == before.st_size
    path.write_bytes(payload)
    os.utime(str(path), ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match='SHA256'):
        load_profile_libraries(config, LEGACY_SHAPE_AXES)
    with pytest.raises(ValueError, match='SHA256'):
        verify_profile_bundle_binding(config, description)


def test_manifest_changed_after_load_is_not_published_under_old_binding(selected):
    directory, config = selected
    _, _, _, description = load_verified_profile_bundle(config)
    path = directory / 'manifest.json'
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='manifest changed'):
        verify_profile_bundle_binding(config, description)


@pytest.mark.parametrize('module,mode', [('run', 'standard'), ('run_gravity_zouhu', 'paired-wce')])
def test_runtime_axes_product_resume_validation_and_el_parent_inheritance(selected, tmp_path, monkeypatch, module, mode):
    _, config = selected
    source = tmp_path / 'synthetic.hdf5'
    make_input(source, count=2, invalid=(1,))
    output = tmp_path / 'products'
    options = dict(morph=True, wave_cal=True, photo=False, el_detect=True)
    import morphology
    match = morphology.match_input_paramters
    used_axes = []

    def capture(library, n, radius, pa, q, ns, rs, ps, qs):
        actual = dict(zip(AXES, (ns, rs, ps, qs)))
        assert all(np.array_equal(actual[name], AXES[name]) for name in AXES)
        profile = match(library, n, radius, pa, q, ns, rs, ps, qs)
        used_axes.append(int(profile.size))
        return profile

    monkeypatch.setattr(morphology, 'match_input_paramters', capture)
    product = Path(importlib.import_module(module).task(str(source.parent), source.name,
        output_dir=output, config=config, options=options, seed=1214))
    assert used_axes
    job = build_jobs([source], output, config, options, mode=mode, seed=1214)[0]
    state = snapshot(config, options, mode, 1214, [job])
    verified = verify_product(job, state)
    assert Path(verified['output']) == product
    unchanged = (product.read_bytes(), product.stat().st_mtime_ns)
    assert verify_product(job, state, expected_sha256=verified['sha256']) == verified
    assert unchanged == (product.read_bytes(), product.stat().st_mtime_ns)
    with h5py.File(str(product), 'r') as handle:
        description = json.loads(handle.attrs['profile_library_provenance_json'])
        assert description['version'] == 2
        assert description['runtime_interpretation']['axes'] == AXES
        assert handle['spectral_extraction/spatial_pixels'][0] == used_axes[1]
        assert np.isnan(handle['spectral_extraction/spatial_pixels'][1])
        assert json.loads(handle.attrs['resource_sha256_json'])['profile_library_manifest_path']['sha256'] == description['bundle_binding']['sha256']
        for band in ('GU', 'GV', 'GI'):
            assert np.isfinite(handle[band + '/ferr'][0]).any()
            assert np.isnan(handle[band + '/ferr'][1]).all()
        old_binding = handle.attrs['profile_library_provenance_json']

    import run_el_detect
    el_config = load_config(CORE / 'emulator_parameters.json')
    child_output = tmp_path / 'postprocess'
    child = run_el_detect.task(str(product.parent), product.name, output_dir=child_output,
        config=el_config, seed=1214, source_start=0, source_stop=1)
    with h5py.File(str(child), 'r') as handle:
        assert handle.attrs['profile_library_provenance_json'] == old_binding
        np.testing.assert_array_equal(handle['input_row_indices'][:], [0])
    el_options = dict(el_detect=True)
    el_job = build_jobs([product], child_output, el_config, el_options, mode='el-detect', seed=1214,
                        source_start=0, source_stop=1)[0]
    el_state = snapshot(el_config, el_options, 'el-detect', 1214, [el_job])
    assert verify_product(el_job, el_state)['output'] == str(child)

    tampered = deepcopy(state)
    tampered['resource_sha256']['profile_library_manifest_path']['sha256'] = '0' * 64
    with pytest.raises(ValueError):
        verify_product(job, tampered)
