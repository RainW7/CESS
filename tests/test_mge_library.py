"""Independent engineering checks of the standalone C11 streaming writer.

Only tiny temporary bundles are generated. Numeric agreement below checks the
writer's binding to C10; it does not independently certify the source MGE fit.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / 'CESS_v0.8.6'
sys.path.insert(0, str(CORE))

import cess_mge
import cess_mge_library as writer
from cess_config import load_config

AXES = {'n': [.5, 1.], 're_pixel': [4.], 'pa_deg': [0., 30.], 'axis_ratio': [.6]}
X_EDGES = np.asarray([-1.5, -.5, .5, 1.5])
Y_EDGES = np.asarray([-3., -1., 0., 2., 4.])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config():
    return load_config(CORE / 'emulator_parameters.json')


def tree(directory):
    return {str(p.relative_to(directory)): (p.lstat().st_ino, p.lstat().st_mtime_ns,
        sha(p) if p.is_file() else None) for p in [directory] + sorted(directory.rglob('*'))}


@pytest.fixture(scope='module')
def generated(tmp_path_factory):
    directory = tmp_path_factory.mktemp('mge-writer') / 'new-bundle'
    settings = config()
    writes = []
    original = h5py.Dataset.__setitem__

    def observe(dataset, selection, values):
        if dataset.name in ('/profiles/x_probability', '/profiles/y_probability'):
            writes.append((dataset.name, np.asarray(values).shape, np.asarray(values).size))
        return original(dataset, selection, values)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(h5py.Dataset, '__setitem__', observe)
    try:
        writer.generate_mge_library(settings, AXES, X_EDGES, directory, y_edges_pixel=Y_EDGES)
    finally:
        monkeypatch.undo()
    return directory, settings, writes


def test_plan_is_json_safe_explicit_and_has_no_science_or_write_side_effects(tmp_path, monkeypatch):
    settings = config()
    for name in list(settings):
        if not name.startswith('_') and name.endswith(('_path', '_pkl')):
            settings[name] = str(tmp_path / ('must-not-open-' + name))

    def forbidden(*args, **kwargs):
        raise AssertionError('plan must not fit, create HDF5 or load a resource')

    monkeypatch.setattr(cess_mge, 'sersic_mge_marginals', forbidden)
    monkeypatch.setattr(cess_mge, '_fit', forbidden)
    monkeypatch.setattr(h5py, 'File', forbidden)
    before = list(tmp_path.iterdir())
    plan = writer.plan_mge_library(settings, AXES, X_EDGES, y_edges_pixel=Y_EDGES)
    assert json.loads(json.dumps(plan, allow_nan=False)) == plan
    assert plan['schema'] == 'cess.positive_marginal_library' and plan['version'] == 1
    assert plan['shape'] == [2, 1, 2, 1] and plan['profile_count'] == 4
    assert plan['axis_order'] == list(AXES)
    assert plan['quantity'] == 'absolute unit-source probability per bin'
    assert plan['orthogonal_domain'] == 'all real' and plan['window_renormalized'] is False
    assert plan['x_edges_pixel'] == X_EDGES.tolist() and plan['y_edges_pixel'] == Y_EDGES.tolist()
    assert list(tmp_path.iterdir()) == before


def test_plan_owns_inputs_and_omitted_y_defaults_to_x():
    axes = {key: list(value) for key, value in AXES.items()}
    edges = X_EDGES.copy()
    plan = writer.plan_mge_library(config(), axes, edges)
    axes['n'][0] = 5.
    edges[0] = -100.
    assert plan['axes'] == AXES
    assert plan['x_edges_pixel'] == plan['y_edges_pixel'] == X_EDGES.tolist()


def test_zero_psf_is_rejected_by_the_explicit_c11_configuration_contract():
    settings = config()
    settings['morphology_psf_fwhm'] = 0.
    with pytest.raises(ValueError, match='morphology_psf_fwhm.*(greater than|positive)'):
        writer.plan_mge_library(settings, AXES, X_EDGES)


@pytest.mark.parametrize('field,value', [('n', [True]), ('n', [.4]),
                                        ('pa_deg', [0., 0.]), ('axis_ratio', [1.1])])
def test_invalid_axes_fail_before_fitting(field, value, monkeypatch):
    monkeypatch.setattr(cess_mge, '_fit', lambda *args: pytest.fail('invalid axes fitted'))
    axes = dict(AXES, **{field: value})
    with pytest.raises(ValueError):
        writer.plan_mge_library(config(), axes, X_EDGES)


@pytest.mark.parametrize('edges', [[1., 0.], [0., np.nan], [False, True]])
def test_invalid_edges_reject_without_generation(edges):
    with pytest.raises(ValueError):
        writer.plan_mge_library(config(), AXES, edges)


@pytest.mark.parametrize('controls', [{'unexpected': 1}, {'max_components': True}])
def test_invalid_controls_are_not_silently_dropped_or_coerced(controls):
    with pytest.raises(ValueError):
        writer.plan_mge_library(config(), AXES, X_EDGES, controls=controls)


def test_plan_resource_limits_reject_too_many_bins_and_profiles():
    with pytest.raises(ValueError):
        writer.plan_mge_library(config(), AXES, np.arange(2051.))
    huge = dict(AXES, re_pixel=np.arange(1., 25002.))
    with pytest.raises(ValueError):
        writer.plan_mge_library(config(), huge, X_EDGES)


def test_streamed_tiny_bundle_matches_direct_mge_and_preserves_window_loss(generated):
    directory, settings, writes = generated
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert set(p.name for p in directory.iterdir()) == {'manifest.json', 'marginals.hdf5'}
    assert manifest['status'] == 'complete' and manifest['generator_revision'] == writer.GENERATOR_REVISION
    assert manifest['physical_validation_status'] == 'experimental_source_approximation_not_instrument_calibration'
    resource = manifest['resource']
    path = directory / 'marginals.hdf5'
    assert resource == {'filename': 'marginals.hdf5', 'sha256': sha(path), 'size_bytes': path.stat().st_size}
    encoded = (json.dumps(manifest['effective_config'], indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    assert hashlib.sha256(encoded).hexdigest() == manifest['effective_config_sha256']
    assert manifest['source_sha256'] and manifest['software'] and manifest['limitations']
    assert len(writes) >= 8
    assert all(size == (3 if name.endswith('x_probability') else 4) for name, shape, size in writes)
    with h5py.File(str(path), 'r') as handle:
        assert set(handle) == {'axes', 'bins', 'profiles', 'fits'}
        assert handle.attrs['schema'] == manifest['schema']
        assert handle.attrs['version'] == 1 and handle.attrs['status'] == 'complete'
        assert not bool(handle.attrs['window_renormalized'])
        assert json.loads(handle.attrs['axis_order_json']) == list(AXES)
        for key in ('model', 'controls'):
            assert json.loads(handle.attrs[key + '_json']) == manifest[key]
        for name in AXES:
            np.testing.assert_array_equal(handle['axes/' + name][:], AXES[name])
            assert handle['axes/' + name].dtype == np.dtype('float64')
        np.testing.assert_array_equal(handle['bins/x_edges_pixel'][:], X_EDGES)
        np.testing.assert_array_equal(handle['bins/y_edges_pixel'][:], Y_EDGES)
        rows = np.asarray(list(np.ndindex(2, 1, 2, 1)), dtype=np.int64)
        np.testing.assert_array_equal(handle['profiles/row_indices'][:], rows)
        assert handle['profiles/row_indices'].dtype == np.dtype('int64')
        assert set(handle['fits']) == {'n_000000', 'n_000001'}
        for row, index in enumerate(rows):
            n, re, pa, q = [AXES[name][index[j]] for j, name in enumerate(AXES)]
            direct = cess_mge.sersic_mge_marginals(n, re, pa, q,
                manifest['model']['psf_fwhm_arcsec'], manifest['model']['pixel_scale_arcsec_pixel'],
                X_EDGES, Y_EDGES, **manifest['controls'])
            for column, axis in enumerate(('x', 'y')):
                dataset = handle['profiles/' + axis + '_probability']
                actual = dataset[row]
                assert dataset.dtype == np.dtype('float64')
                np.testing.assert_allclose(actual, direct[axis + '_probability'], rtol=0., atol=0.)
                mass = float(actual.sum())
                outside = handle['profiles/outside_probability'][row, column]
                assert np.all(np.isfinite(actual)) and np.all(actual >= 0.)
                assert mass < .9 and outside > .1  # Narrow windows, deliberately resolvable physical loss.
                assert handle['profiles/window_probability'][row, column] == pytest.approx(mass, abs=2.e-15)
                assert outside == pytest.approx(direct['metadata']['windows'][axis]['outside_probability_integrated'], abs=2.e-15)
                assert mass + outside == pytest.approx(1., abs=1.e-9)
            fit = handle['fits/n_{:06d}'.format(index[0])]
            fit_metadata = json.loads(fit.attrs['metadata_json'])
            assert fit.attrs['n'] == n
            assert fit_metadata == direct['metadata']['source_fit']
            np.testing.assert_array_equal(fit['weights'][:], direct['mixture_weights'])
            np.testing.assert_array_equal(fit['sigma_over_re'][:], direct['mixture_sigma_over_re'])
            assert np.all(fit['weights'][:] > 0.) and np.all(fit['sigma_over_re'][:] > 0.)
            assert len(fit['weights']) == fit_metadata['attempts'][-1]['active_components']
            assert handle['profiles/source_fit_l1_estimate'][row] == fit_metadata['estimated_radial_L1_including_outside']


@pytest.mark.parametrize('kind', ['empty-directory', 'file', 'bundle'])
def test_existing_output_is_refused_before_fit_and_left_unchanged(tmp_path, generated, monkeypatch, kind):
    if kind == 'bundle':
        output = generated[0]
    else:
        output = tmp_path / 'existing'
        output.mkdir() if kind == 'empty-directory' else output.write_bytes(b'original')
    before = tree(output) if output.is_dir() else (output.read_bytes(), output.stat().st_mtime_ns)
    monkeypatch.setattr(cess_mge, '_fit', lambda *args: pytest.fail('existing output must be refused before fit'))
    with pytest.raises(FileExistsError):
        writer.generate_mge_library(config(), AXES, X_EDGES, output)
    after = tree(output) if output.is_dir() else (output.read_bytes(), output.stat().st_mtime_ns)
    assert before == after


def test_failed_fit_retains_unpublished_staging_without_substitute(tmp_path):
    output = tmp_path / 'failed-fit'
    grid = dict(n=[5.], re_pixel=[4.], pa_deg=[0.], axis_ratio=[.6])
    with pytest.raises(cess_mge.MixtureConvergenceError) as error:
        writer.generate_mge_library(config(), grid, X_EDGES, output,
            controls=dict(l1_tolerance=1.e-12, tail_probability=1.e-12, max_components=16))
    assert not os.path.lexists(str(output))
    staging = Path(error.value.staging_directory)
    assert staging.is_dir() and staging.parent == tmp_path
    assert error.value.diagnostics


def test_config_change_during_generation_prevents_publication(tmp_path, monkeypatch):
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config()))
    original_fit = cess_mge._fit
    changed = []

    def mutate(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        if not changed:
            config_path.write_bytes(config_path.read_bytes() + b'\n')
            changed.append(True)
        return result

    monkeypatch.setattr(cess_mge, '_fit', mutate)
    output = tmp_path / 'config-drift'
    with pytest.raises(RuntimeError, match='[Cc]onfig') as error:
        writer.generate_mge_library(config_path, AXES, X_EDGES, output)
    assert changed and not output.exists()
    assert Path(error.value.staging_directory).is_dir()


@pytest.mark.parametrize('source_name', ['morphology/mge.py', 'spectra/numerics.py', 'spectra/__init__.py'])
def test_source_hash_drift_is_rejected_without_editing_any_source(tmp_path, monkeypatch, source_name):
    original_fit = cess_mge._fit
    original_sha = writer._sha256
    after_fit = []

    def fit(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        after_fit.append(True)
        return result

    def changed_hash(path):
        actual = original_sha(path)
        return '0' * 64 if after_fit and Path(path) == writer.PROJECT/source_name else actual

    monkeypatch.setattr(cess_mge, '_fit', fit)
    monkeypatch.setattr(writer, '_sha256', changed_hash)
    output = tmp_path / 'source-drift'
    with pytest.raises(RuntimeError, match='[Ss]ource') as error:
        writer.generate_mge_library(config(), AXES, X_EDGES, output)
    assert after_fit and not output.exists()
    assert Path(error.value.staging_directory).is_dir()


def test_profile_write_failure_keeps_staging_and_unrelated_files(tmp_path, monkeypatch):
    original = h5py.Dataset.__setitem__
    unrelated = tmp_path / 'unrelated'
    unrelated.write_bytes(b'keep')

    def broken(dataset, selection, values):
        if dataset.name == '/profiles/x_probability':
            raise OSError('injected profile write failure')
        return original(dataset, selection, values)

    monkeypatch.setattr(h5py.Dataset, '__setitem__', broken)
    output = tmp_path / 'io-failure'
    with pytest.raises(OSError, match='injected') as error:
        writer.generate_mge_library(config(), AXES, X_EDGES, output)
    assert not output.exists()
    assert Path(error.value.staging_directory).is_dir()
    assert unrelated.read_bytes() == b'keep'


def test_foreign_cwd_cli_dryrun_then_tiny_generation(tmp_path):
    foreign = tmp_path / 'foreign'
    foreign.mkdir()
    grid, edges = tmp_path / 'grid.json', tmp_path / 'edges.json'
    grid.write_text(json.dumps(dict(n=[.5], re_pixel=[4.], pa_deg=[0., 30.], axis_ratio=[1.])))
    edges.write_text(json.dumps({'x_edges_pixel': X_EDGES.tolist(), 'y_edges_pixel': Y_EDGES.tolist()}))
    output = tmp_path / 'foreign-bundle'
    command = [sys.executable, '-B', str(CORE / 'cess_mge_library.py'),
        '--config', str(CORE / 'emulator_parameters.json'), '--grid', str(grid),
        '--edges', str(edges), '--output', str(output)]
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    environment.pop('PYTHONPATH', None)
    planned = subprocess.run(command + ['--dry-run'], cwd=str(foreign), env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
    assert planned.returncode == 0, planned.stderr
    assert json.loads(planned.stdout)['profile_count'] == 2
    assert not output.exists() and not list(tmp_path.glob('.*.staging-*'))
    completed = subprocess.run(command, cwd=str(foreign), env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['profile_count'] == 2
    assert set(p.name for p in output.iterdir()) == {'manifest.json', 'marginals.hdf5'}


def test_c7_verifier_does_not_accept_the_new_experimental_artifact(generated):
    from cess_profile_bundle import verify_profile_bundle
    with pytest.raises(ValueError):
        verify_profile_bundle(generated[0])
