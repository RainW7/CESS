"""Legacy resources have measurable identity, not an inferred historical PSF."""
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

CORE = Path(__file__).resolve().parents[1] / 'CESS_v0.8.6'
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from cess_profiles import describe_profile_libraries


@pytest.fixture
def fixture(tmp_path):
    width_path, height_path = tmp_path / 'width.pkl', tmp_path / 'height.pkl'
    # The helper hashes files but does not unpickle them; trusted loading belongs
    # to its caller. These small byte fixtures test that separation explicitly.
    width_path.write_bytes(b'width resource bytes\x00')
    height_path.write_bytes(b'height resource bytes\xff')
    axes = {'n': np.asarray([1., 4.]), 're_pixel': [4.],
            'pa_deg': [0.], 'axis_ratio': [.5]}
    width, height = np.empty((2, 1, 1, 1), dtype=object), np.empty((2, 1, 1, 1), dtype=object)
    x = np.arange(-3., 4.)
    gaussian = np.exp(-.5 * x ** 2)
    gaussian /= gaussian.sum()
    width[0, 0, 0, 0] = gaussian.copy()
    width[1, 0, 0, 0] = np.asarray([0., .2, .4, .2, 0.])
    height[0, 0, 0, 0] = .7 * gaussian
    height[1, 0, 0, 0] = np.asarray([.1, .3, .2])
    parameters = {'widthlib_pkl': str(width_path), 'heightlib_pkl': str(height_path),
                  'arcsecperpix': .074}
    return width, height, parameters, axes


def test_identity_unknown_generation_statistics_and_no_mutation(fixture):
    width, height, parameters, axes = fixture
    originals = [[np.copy(v) for v in library.flat] for library in (width, height)]
    file_bytes = {name: Path(parameters[name]).read_bytes()
                  for name in ('widthlib_pkl', 'heightlib_pkl')}
    metadata = describe_profile_libraries(*fixture)
    assert json.loads(json.dumps(metadata, allow_nan=False)) == metadata
    assert metadata['provenance_status'] == 'legacy_unverified_generation'
    assert metadata['generation'] == {'status': 'unknown', 'psf': None,
        'pixel_scale_arcsec_pixel': None, 'normalization': None}
    assert metadata['configured_generator_setting']['provided_in_parameters'] is False
    assert metadata['configured_generator_setting']['morphology_psf_fwhm_arcsec'] is None
    interpretation = metadata['runtime_interpretation']
    assert interpretation['axis_order'] == ['n', 're_pixel', 'pa_deg', 'axis_ratio']
    assert interpretation['axes']['n'] == [1., 4.]
    assert interpretation['arcsecperpix'] == .074
    for role, originals_for_role, library in zip(('width', 'height'), originals, (width, height)):
        resource = metadata['resources'][role]
        path = Path(parameters[role + 'lib_pkl'])
        assert resource['path'] == str(path.resolve())
        assert resource['sha256'] == hashlib.sha256(file_bytes[role + 'lib_pkl']).hexdigest()
        assert resource['size_bytes'] == path.stat().st_size
        assert resource['shape'] == [2, 1, 1, 1]
        assert path.read_bytes() == file_bytes[role + 'lib_pkl']
        for actual, original in zip(library.flat, originals_for_role):
            np.testing.assert_array_equal(actual, original)
    stats = metadata['observed_profile_statistics']
    assert stats['width']['profile_count'] == 2
    assert stats['width']['length_range'] == [5, 7]
    np.testing.assert_allclose(stats['height']['sum_range'], [.6, .7])
    assert stats['height']['length_range'] == [3, 7]
    assert stats['width']['gaussian_like_count'] == stats['height']['gaussian_like_count'] == 1
    assert 'not a certified all-source aperture fraction' in stats['height']['sum_interpretation']
    metadata['runtime_interpretation']['axes']['n'][0] = 99.
    assert axes['n'][0] == 1.


def test_configured_psf_is_only_an_unbound_generator_setting(fixture):
    fixture[2]['morphology_psf_fwhm'] = np.float64(.3)
    metadata = describe_profile_libraries(*fixture)
    assert metadata['generation']['psf'] is None
    setting = metadata['configured_generator_setting']
    assert setting == {'status': 'generator-setting-not-bound-to-existing-library',
                       'morphology_psf_fwhm_arcsec': .3, 'provided_in_parameters': True}
    assert any('does not rebuild' in note for note in metadata['limitations'])


@pytest.mark.parametrize('bad_axis', [[], [1., 1.], [4., 1.], [1., np.nan],
                                    [[1., 4.]], [1., np.inf], [1.+1.j, 4.]])
def test_invalid_axes_are_rejected(fixture, bad_axis):
    fixture[3]['n'] = bad_axis
    with pytest.raises(ValueError, match='axis n'):
        describe_profile_libraries(*fixture)


@pytest.mark.parametrize('name', ['n', 're_pixel', 'pa_deg', 'axis_ratio'])
def test_axis_length_must_match_all_library_dimensions(fixture, name):
    fixture[3][name] = [1., 2., 3.]
    with pytest.raises(ValueError, match='width library shape'):
        describe_profile_libraries(*fixture)


def test_height_shape_and_exact_axis_names_are_checked(fixture):
    with pytest.raises(ValueError, match='height library shape'):
        describe_profile_libraries(fixture[0], fixture[1][:1], fixture[2], fixture[3])
    fixture[3]['unexpected'] = [1.]
    with pytest.raises(ValueError, match='exactly'):
        describe_profile_libraries(*fixture)


@pytest.mark.parametrize('bad_profile, reason', [
    ([], 'nonempty'), ([[1., 2.]], 'one-dimensional'),
    ([1., np.nan], 'finite'), ([np.inf], 'finite'),
    ([-1.e-30, 1.], 'nonnegative'), ([0., 0.], 'positive finite sum'),
    ([1.e308, 1.e308], 'positive finite sum'),
    ([1.+0.j], 'real numeric'), (['bad'], 'real numeric')])
def test_bad_profile_reports_role_and_grid_index_without_clipping(fixture, bad_profile, reason):
    fixture[1][1, 0, 0, 0] = np.asarray(bad_profile)
    with pytest.raises(ValueError, match='height profile \\(1, 0, 0, 0\\).*' + reason):
        describe_profile_libraries(*fixture)


@pytest.mark.parametrize('value', [0., -1., np.nan, np.inf, True])
def test_scale_must_be_positive_finite(fixture, value):
    fixture[2]['arcsecperpix'] = value
    with pytest.raises(ValueError, match='arcsecperpix'):
        describe_profile_libraries(*fixture)


def test_missing_resource_and_changed_file_identity(fixture):
    first = describe_profile_libraries(*fixture)
    path = Path(fixture[2]['widthlib_pkl'])
    path.write_bytes(b'changed width resource with a different size')
    second = describe_profile_libraries(*fixture)
    assert first['resources']['width']['sha256'] != second['resources']['width']['sha256']
    del fixture[2]['widthlib_pkl']
    with pytest.raises(ValueError, match='widthlib_pkl'):
        describe_profile_libraries(*fixture)


def test_module_import_performs_no_file_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('import attempted file I/O')
    # Read source before disabling open; execute its module body independently.
    path = CORE.parent / 'src/cess/morphology/libraries.py'
    code = compile(path.read_text(), str(path), 'exec')
    monkeypatch.setattr(builtins, 'open', forbidden)
    monkeypatch.setattr(Path, 'open', forbidden)
    namespace = {'__name__': 'isolated_profile_import'}
    exec(code, namespace)
    assert callable(namespace['describe_profile_libraries'])
