"""Standalone byte/manifest verification must never unpickle trusted or hostile bytes."""
from copy import deepcopy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import pytest
CORE = Path(__file__).resolve().parents[1] / 'src'
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from cess.morphology.profile_bundle import VERIFICATION_SCOPE, verify_profile_bundle
import cess.morphology.profile_bundle as bundle_module
from cess.morphology.profile_generator import plan_generation


@pytest.fixture
def bundle(tmp_path):
    plan = plan_generation(CORE / 'cess/data/emulator_parameters.json',
        {'n': [1.], 're_pixel': [4.], 'pa_deg': [0., 45.], 'axis_ratio': [1.]})
    path = tmp_path / 'bundle'
    path.mkdir()
    plan.update(status='complete', source_sha256={'historical/morphology.py': 'a' * 64},
                serialization={'format': 'pickle', 'protocol': 4},
                resources={}, observed_profile_statistics={})
    for role in ('width', 'height'):
        data = (role + ' deliberately opaque bytes, not a pickle').encode('ascii')
        filename = role + 'lib.pkl'
        (path / filename).write_bytes(data)
        plan['resources'][role] = dict(filename=filename, sha256=hashlib.sha256(data).hexdigest(),
                                      size_bytes=len(data), shape=[1, 1, 2, 1], array_dtype='object')
        plan['observed_profile_statistics'][role] = dict(profile_count=2, all_profiles_valid=True,
            length_range=[3, 5], sum_range=[.5, 1.], gaussian_like_count=0, profile_dtype_counts={'float64': 2})
    write_manifest(path, plan)
    return path, plan


def write_manifest(path, manifest):
    (path / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')


def test_summary_is_explicitly_declared_not_decoded_and_no_input_mutation(bundle, monkeypatch):
    path, manifest = bundle
    before = {entry.name: entry.read_bytes() for entry in path.iterdir()}
    def forbidden(*args, **kwargs):
        pytest.fail('verifier must never decode pickle')
    monkeypatch.setattr(pickle, 'load', forbidden)
    monkeypatch.setattr(pickle, 'loads', forbidden)
    result = verify_profile_bundle(path)
    assert result['verified'] is True
    assert result['verification_scope'] == VERIFICATION_SCOPE
    assert result['declared_shape'] == [1, 1, 2, 1]
    assert result['declared_profiles_per_library'] == 2
    assert result['physical_validation_status'] == 'unvalidated_legacy_rendering_recipe'
    assert result['source_sha256'] == manifest['source_sha256']
    assert result['manifest_sha256'] == hashlib.sha256(before['manifest.json']).hexdigest()
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert {entry.name: entry.read_bytes() for entry in path.iterdir()} == before


@pytest.mark.parametrize('key,value', [
    ('schema', 'other'), ('version', 2), ('version', True), ('status', 'planned'),
    ('axis_order', ['re_pixel', 'n', 'pa_deg', 'axis_ratio']),
    ('shape', [1, 2, 1, 1]), ('shape', [True, 1, 2, 1]),
    ('profiles_per_library', 3), ('profiles_per_library', True),
    ('effective_config_sha256', 'b' * 64), ('effective_config_hash_encoding', 'other'),
    ('source_sha256', {}), ('source_sha256', {'morphology.py': 'invalid'}),
    ('serialization', {'format': 'pickle', 'protocol': 5})])
def test_top_level_contract_rejected(bundle, key, value):
    path, manifest = bundle
    manifest[key] = value
    write_manifest(path, manifest)
    with pytest.raises(ValueError):
        verify_profile_bundle(path)


@pytest.mark.parametrize('name,values', [
    ('n', []), ('n', [True]), ('n', ['1']), ('n', [.2]), ('n', [6.3]),
    ('n', [[1.]]), ('n', [1., 1.]), ('pa_deg', [45., 0.]),
    ('re_pixel', [0.]), ('axis_ratio', [0.]), ('axis_ratio', [1.1])])
def test_axis_domain_and_order_are_checked(bundle, name, values):
    path, manifest = bundle
    manifest['axes'][name] = values
    write_manifest(path, manifest)
    with pytest.raises(ValueError):
        verify_profile_bundle(path)


@pytest.mark.parametrize('field,value', [
    ('filename', '../widthlib.pkl'), ('filename', '/tmp/widthlib.pkl'),
    ('filename', 'heightlib.pkl'), ('sha256', 'x' * 64),
    ('size_bytes', True), ('size_bytes', 0), ('size_bytes', 999),
    ('shape', [1, 1, 1, 1]), ('array_dtype', 'float64')])
def test_resources_are_fixed_members_with_consistent_declarations(bundle, field, value):
    path, manifest = bundle
    manifest['resources']['width'][field] = value
    write_manifest(path, manifest)
    with pytest.raises(ValueError):
        verify_profile_bundle(path)


@pytest.mark.parametrize('field,value', [
    ('profile_count', 3), ('profile_count', True), ('all_profiles_valid', False),
    ('length_range', [5, 3]), ('length_range', [True, 3]),
    ('sum_range', [0., 1.]), ('sum_range', [2., 1.]),
    ('gaussian_like_count', 3), ('profile_dtype_counts', {'float64': 1})])
def test_statistics_are_declared_consistency_only(bundle, field, value):
    path, manifest = bundle
    manifest['observed_profile_statistics']['height'][field] = value
    write_manifest(path, manifest)
    with pytest.raises(ValueError):
        verify_profile_bundle(path)


@pytest.mark.parametrize('field', ['pixel_scale_arcsec_pixel', 'physical_validation_status', 'psf'])
def test_recipe_binds_effective_scale_psf_and_preserves_unvalidated_status(bundle, field):
    path, manifest = bundle
    values = {'pixel_scale_arcsec_pixel': .1, 'physical_validation_status': 'calibrated',
              'psf': {'model': 'circular_Gaussian', 'fwhm_arcsec': .3}}
    manifest['rendering_recipe'][field] = values[field]
    write_manifest(path, manifest)
    with pytest.raises(ValueError):
        verify_profile_bundle(path)


def test_config_snapshot_cannot_change_without_hash_and_recipe_binding(bundle):
    path, manifest = bundle
    manifest['effective_config']['arcsecperpix'] = .2
    write_manifest(path, manifest)
    with pytest.raises(ValueError, match='effective_config SHA256'):
        verify_profile_bundle(path)
    payload = (json.dumps(manifest['effective_config'], indent=2, sort_keys=True, allow_nan=False)+'\n').encode('utf-8')
    manifest['effective_config_sha256'] = hashlib.sha256(payload).hexdigest()
    write_manifest(path, manifest)
    with pytest.raises(ValueError, match='Recipe pixel scale'):
        verify_profile_bundle(path)


@pytest.mark.parametrize('field,value', [
    ('total_profiles', 3), ('total_profiles', True),
    ('axis_units', {'n': 'dimensionless', 're_pixel': 'arcsec',
                    'pa_deg': 'degree', 'axis_ratio': 'dimensionless'})])
def test_optional_units_and_total_are_checked_if_present(bundle, field, value):
    path, manifest = bundle
    manifest[field] = value
    write_manifest(path, manifest)
    with pytest.raises(ValueError):
        verify_profile_bundle(path)


@pytest.mark.parametrize('changed', ['manifest.json', 'widthlib.pkl'])
def test_change_after_member_was_read_is_caught_at_final_snapshot(bundle, monkeypatch, changed):
    path, _ = bundle
    original = bundle_module._regular_member
    @contextmanager
    def changing(directory_fd, name):
        with original(directory_fd, name) as stream:
            if name == 'heightlib.pkl':
                target = path / changed
                target.write_bytes(target.read_bytes() + b' ')
            yield stream
    monkeypatch.setattr(bundle_module, '_regular_member', changing)
    with pytest.raises(ValueError, match='Bundle members changed during verification'):
        verify_profile_bundle(path)


def test_resource_byte_corruption_same_size_is_detected(bundle):
    path, _ = bundle
    resource = path / 'heightlib.pkl'
    before = resource.read_bytes()
    resource.write_bytes(b'X' + before[1:])
    with pytest.raises(ValueError, match='SHA256 disagrees with actual bytes'):
        verify_profile_bundle(path)


@pytest.mark.parametrize('member', ['manifest.json', 'widthlib.pkl', 'heightlib.pkl'])
def test_member_symlink_is_rejected(bundle, tmp_path, member):
    path, _ = bundle
    target = tmp_path / ('original_' + member)
    (path / member).rename(target)
    (path / member).symlink_to(target)
    with pytest.raises(ValueError, match='non-symlink'):
        verify_profile_bundle(path)


def test_bundle_symlink_parent_traversal_and_extra_member_are_rejected(bundle, tmp_path):
    path, _ = bundle
    alias = tmp_path / 'alias'
    alias.symlink_to(path, target_is_directory=True)
    with pytest.raises(ValueError, match='directory must not be a symlink'):
        verify_profile_bundle(alias)
    with pytest.raises(ValueError, match='parent traversal'):
        verify_profile_bundle(path / '..' / path.name)
    (path / 'unexpected').mkdir()
    with pytest.raises(ValueError, match='exactly'):
        verify_profile_bundle(path)


def test_duplicate_keys_and_nonfinite_json_are_rejected(bundle):
    path, _ = bundle
    (path / 'manifest.json').write_text('{"schema":1,"schema":2}')
    with pytest.raises(ValueError, match='Duplicate'):
        verify_profile_bundle(path)
    (path / 'manifest.json').write_text('{"bad":NaN}')
    with pytest.raises(ValueError, match='Nonfinite'):
        verify_profile_bundle(path)


def test_cli_success_failure_and_help_do_not_generate(bundle, tmp_path):
    path, _ = bundle
    command = [sys.executable, '-B', str(CORE / 'cess/morphology/profile_bundle.py')]
    success = subprocess.run(command + [str(path)], cwd=str(tmp_path),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    assert success.returncode == 0, success.stderr
    assert json.loads(success.stdout)['verification_scope'] == VERIFICATION_SCOPE
    failed = subprocess.run(command + [str(tmp_path / 'missing')], cwd=str(tmp_path),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    assert failed.returncode != 0 and 'verification failed' in failed.stderr
    help_result = subprocess.run(command + ['--help'], cwd=str(tmp_path),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    assert help_result.returncode == 0
