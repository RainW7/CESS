"""Traceable generation: units, actual profile bytes and failure isolation."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'CESS_v0.8.6'
sys.path.insert(0, str(CODE))
import cess_profile_generator as generator
from cess_config import load_config
import morphology


@pytest.fixture
def config():
    return load_config(CODE / 'emulator_parameters.json')


@pytest.fixture
def axes():
    return {'n': [1.], 're_pixel': [4.], 'pa_deg': [0., 45.], 'axis_ratio': [1.]}


@pytest.mark.parametrize('name,values', [
    ('n', []), ('n', [[1.]]), ('n', [True]), ('n', ['1']), ('n', [.2]),
    ('n', [6.3]), ('n', [1., 1.]), ('n', [2., 1.]), ('n', [np.nan]),
    ('pa_deg', [10**400]),
    ('re_pixel', [0.]), ('re_pixel', [-1.]), ('pa_deg', [np.inf]),
    ('axis_ratio', [0.]), ('axis_ratio', [1.1]), ('axis_ratio', [1j]),
    ('n', np.ma.array([1.], mask=[False])), ('n', np.asarray([[1.]]))])
def test_bad_axes_fail_before_rendering(axes, name, values):
    axes[name] = values
    with pytest.raises(ValueError):
        generator.validate_axes(axes)


def test_plan_owns_values_preserves_explicit_pixel_grid(config, axes):
    axes['pa_deg'] = [-90., 0., 90.]
    axes['re_pixel'] = [4.25]
    plan = generator.plan_generation(config, axes)
    assert plan['shape'] == [1, 1, 3, 1]
    assert plan['profiles_per_library'] == 3
    assert plan['axes']['re_pixel'] == [4.25]
    assert plan['effective_config']['morphology_psf_fwhm'] == .2625
    assert plan['config_source'] == {'kind': 'mapping', 'path': None, 'sha256': None,
                                    'adopted_defaults': {'morphology_psf_fwhm': .2625}}
    axes['n'][0] = 2.
    config['arcsecperpix'] = .1
    assert plan['axes']['n'] == [1.]
    assert plan['effective_config']['arcsecperpix'] == .074
    assert hashlib.sha256(generator._json_bytes(plan['effective_config'])).hexdigest() == plan['effective_config_sha256']


def test_legacy_grid_scale_conversion_is_explicit():
    grid = generator.legacy_axes(.074)
    assert [len(grid[name]) for name in generator.AXIS_ORDER] == [20, 20, 10, 10]
    expected = np.round(np.asarray([.3, .5, .7, .9, 1., 1.2, 1.4, 1.6, 1.8, 2.,
                                   2.5, 3., 3.5, 4.5, 5., 5.5, 6., 6.5, 7., 7.4]) / .074)
    np.testing.assert_array_equal(grid['re_pixel'], expected)
    assert generator.legacy_axes(.05)['re_pixel'][0] == 6.
    with pytest.raises(ValueError):
        generator.legacy_axes(1.)  # duplicate/zero radii are not dropped


def test_import_and_help_do_not_render_or_read_config(tmp_path):
    entry = CODE / 'basic_file/create_lib.py'
    spec = importlib.util.spec_from_file_location('safe_profile_entry', str(entry))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
    (tmp_path / 'morphology.py').write_text("raise RuntimeError('foreign cwd module')\n")
    env = dict(os.environ, CESS_CONFIG='/nonexistent/config.json', PYTHONDONTWRITEBYTECODE='1')
    before = set(tmp_path.iterdir())
    result = subprocess.run([sys.executable, '-B', str(entry), '--help'], cwd=str(tmp_path),
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    assert result.returncode == 0, result.stderr
    assert '--grid' in result.stdout and '--legacy-grid' in result.stdout
    assert set(tmp_path.iterdir()) == before


def test_foreign_cwd_dry_run_and_config_snapshot(tmp_path, axes):
    grid = tmp_path / 'grid.json'
    grid.write_text(json.dumps(axes))
    output = tmp_path / 'never_created'
    result = subprocess.run([sys.executable, '-B', str(CODE / 'basic_file/create_lib.py'),
                             '--config', str(CODE / 'emulator_parameters.json'),
                             '--grid', str(grid), '--output', str(output), '--dry-run'],
                            cwd=str(tmp_path), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan['shape'] == [1, 1, 2, 1]
    assert plan['config_source']['sha256'] == hashlib.sha256((CODE / 'emulator_parameters.json').read_bytes()).hexdigest()
    assert not output.exists()


def test_real_small_grid_round_trip_matches_renderer(tmp_path, config, axes):
    output = tmp_path / 'bundle'
    rng_before = np.random.get_state()
    manifest = generator.generate_profile_libraries(config, axes, output)
    rng_after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(rng_before, rng_after))
    assert sorted(path.name for path in output.iterdir()) == ['heightlib.pkl', 'manifest.json', 'widthlib.pkl']
    assert json.loads((output / 'manifest.json').read_text()) == manifest
    assert manifest['status'] == 'complete'
    assert manifest['shape'] == [1, 1, 2, 1]
    for role in ('width', 'height'):
        resource = manifest['resources'][role]
        path = output / resource['filename']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == resource['sha256']
        with path.open('rb') as stream:
            array = pickle.load(stream)
        assert array.shape == (1, 1, 2, 1) and array.dtype == object
        morphology.configure(config)
        for i, pa in enumerate(axes['pa_deg']):
            expected = morphology.get_2d_profile(1., 4., pa, 1.)[0 if role == 'width' else 1]
            np.testing.assert_array_equal(array[0, 0, i, 0], expected)
        assert manifest['observed_profile_statistics'][role]['profile_count'] == 2
    assert not list(tmp_path.glob('.*.staging-*'))


@pytest.mark.parametrize('kind', ['directory', 'file', 'link', 'broken_link'])
def test_existing_output_rejected_before_rendering(tmp_path, config, axes, monkeypatch, kind):
    output = tmp_path / 'old'
    if kind == 'directory':
        output.mkdir()
    elif kind == 'file':
        output.write_bytes(b'old data')
    else:
        target = tmp_path / 'target'
        if kind == 'link':
            target.mkdir()
        output.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(morphology, 'create_lib', lambda *args: pytest.fail('must not render'))
    with pytest.raises(FileExistsError):
        generator.generate_profile_libraries(config, axes, output)
    assert os.path.lexists(str(output))


def _fake_arrays(shape=(1, 1, 2, 1)):
    width, height = np.empty(shape, dtype=object), np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        width[index], height[index] = np.asarray([.2, .6, .2]), np.asarray([.3, .4, .3])
    return width, height


def test_invalid_profiles_do_not_publish_or_clip(tmp_path, config, axes, monkeypatch):
    arrays = _fake_arrays()
    arrays[0][0, 0, 0, 0][0] = -1.e-12
    monkeypatch.setattr(morphology, 'create_lib', lambda *args: arrays)
    with pytest.raises(ValueError, match='nonnegative'):
        generator.generate_profile_libraries(config, axes, tmp_path / 'invalid')
    assert not list(tmp_path.iterdir())
    assert arrays[0][0, 0, 0, 0][0] == -1.e-12


def test_write_failure_retains_unpublished_staging(tmp_path, config, axes, monkeypatch):
    monkeypatch.setattr(morphology, 'create_lib', lambda *args: _fake_arrays())
    old = tmp_path / 'unrelated.txt'
    old.write_bytes(b'preserve')
    real_write = generator._write_pickle
    def fail_second(path, array):
        if path.name == 'heightlib.pkl':
            raise OSError('injected disk failure')
        return real_write(path, array)
    monkeypatch.setattr(generator, '_write_pickle', fail_second)
    with pytest.raises(OSError, match='injected') as failure:
        generator.generate_profile_libraries(config, axes, tmp_path / 'bundle')
    staging = Path(failure.value.staging_directory)
    assert staging.parent == tmp_path
    assert sorted(p.name for p in staging.iterdir()) == ['widthlib.pkl']
    assert not (tmp_path / 'bundle').exists()
    assert old.read_bytes() == b'preserve'


def test_publication_race_preserves_competing_directory(tmp_path, config, axes, monkeypatch):
    import cess_profile_publish
    monkeypatch.setattr(morphology, 'create_lib', lambda *args: _fake_arrays())
    publish = cess_profile_publish.publish_directory
    def race(staging, output):
        output.mkdir()
        (output / 'other.txt').write_bytes(b'competing data')
        return publish(staging, output)
    monkeypatch.setattr(cess_profile_publish, 'publish_directory', race)
    output = tmp_path / 'bundle'
    with pytest.raises(FileExistsError) as failure:
        generator.generate_profile_libraries(config, axes, output)
    assert [p.name for p in output.iterdir()] == ['other.txt']
    assert Path(failure.value.staging_directory).is_dir()


def test_replaced_staging_is_never_recursively_deleted(tmp_path, config, axes, monkeypatch):
    import cess_profile_publish
    monkeypatch.setattr(morphology, 'create_lib', lambda *args: _fake_arrays())
    original = tmp_path / 'original-stage'
    def replace_then_fail(staging, output):
        staging.rename(original)
        staging.mkdir()
        (staging / 'replacement.txt').write_bytes(b'other owner')
        raise OSError('injected stage replacement')
    monkeypatch.setattr(cess_profile_publish, 'publish_directory', replace_then_fail)
    with pytest.raises(OSError, match='replacement') as failure:
        generator.generate_profile_libraries(config, axes, tmp_path / 'bundle')
    retained = Path(failure.value.staging_directory)
    assert (retained / 'replacement.txt').read_bytes() == b'other owner'
    assert sorted(p.name for p in original.iterdir()) == ['heightlib.pkl', 'manifest.json', 'widthlib.pkl']
    assert not (tmp_path / 'bundle').exists()


def test_changed_config_aborts_publication(tmp_path, config, axes, monkeypatch):
    source = tmp_path / 'config.json'
    source.write_text(json.dumps(config))
    def changing_render(*args):
        source.write_text(source.read_text() + '\n')
        return _fake_arrays()
    monkeypatch.setattr(morphology, 'create_lib', changing_render)
    with pytest.raises(RuntimeError, match='Configuration changed'):
        generator.generate_profile_libraries(source, axes, tmp_path / 'bundle')
    assert [p.name for p in tmp_path.iterdir()] == ['config.json']
