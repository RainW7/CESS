"""Explicit extraction choices and immutable, byte-bound profile descriptions."""
import json
from pathlib import Path
import pickle
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'CESS_v0.8.6'))
from cess_config import load_config
from runner_support import (cached_pickle, profile_library_provenance,
    resolve_spectral_extraction, write_spectral_extraction,
    record_runner_provenance, atomic_hdf5)


@pytest.mark.parametrize('design', [None, [], {}, {'mode': 'unknown'}, {'mode': False},
    {'mode': 'legacy_full', 'source_fraction': 1.},
    {'mode': 'explicit'}, {'mode': 'explicit', 'source_fraction': .4, 'spatial_pixels': 3, 'typo': 1},
    {'mode': 'explicit', 'source_fraction': -1, 'spatial_pixels': 3},
    {'mode': 'explicit', 'source_fraction': 1.01, 'spatial_pixels': 3},
    {'mode': 'explicit', 'source_fraction': float('nan'), 'spatial_pixels': 3},
    {'mode': 'explicit', 'source_fraction': True, 'spatial_pixels': 3},
    {'mode': 'explicit', 'source_fraction': .4, 'spatial_pixels': 0},
    {'mode': 'explicit', 'source_fraction': .4, 'spatial_pixels': 3.5},
    {'mode': 'explicit', 'source_fraction': .4, 'spatial_pixels': True}])
def test_bad_extraction_design_fails_before_resource_io(design):
    config = load_config()
    config['spectral_extraction'] = design
    with pytest.raises(ValueError, match='spectral_extraction|source_fraction|spatial_pixels'):
        load_config(config)


def test_default_compatibility_and_explicit_zero_are_independent_configurations():
    config = load_config()
    config.pop('spectral_extraction')
    first, second = load_config(config), load_config(config)
    assert first['spectral_extraction'] == {'mode': 'legacy_full'}
    first['spectral_extraction']['mode'] = 'changed'
    assert second['spectral_extraction'] == {'mode': 'legacy_full'}
    height = np.ones(9) * .6 / 9.
    legacy = resolve_spectral_extraction(second, height)
    assert legacy['source_fraction'] == 1. and legacy['spatial_pixels'] == 9
    assert legacy['profile_height_sum'] == pytest.approx(.6)
    second['spectral_extraction'] = dict(mode='explicit', source_fraction=0., spatial_pixels=4.)
    explicit = resolve_spectral_extraction(load_config(second), height)
    assert explicit['source_fraction'] == 0. and explicit['spatial_pixels'] == 4


@pytest.mark.parametrize('height', [[], [0., 0.], [-.1, .8], [np.nan, 1.], [[1.]]])
def test_invalid_height_is_not_a_valid_resource_diagnostic(height):
    with pytest.raises(ValueError, match='Extraction height'):
        resolve_spectral_extraction(load_config(), height)


def profile_fixture(tmp_path):
    axes = dict(n=np.array([1.]), re_pixel=np.array([4.]),
                pa_deg=np.array([0.]), axis_ratio=np.array([.5]))
    profile = np.empty((1, 1, 1, 1), dtype=object)
    profile[0, 0, 0, 0] = np.array([.1, .4, .1])
    config = load_config()
    for role in ('width', 'height'):
        path = tmp_path / (role + '.pkl')
        with path.open('wb') as handle:
            pickle.dump(profile, handle)
        config[role + 'lib_pkl'] = str(path)
    return config, axes


def test_profile_cache_metadata_cannot_be_mutated_by_other_tasks(tmp_path):
    config, axes = profile_fixture(tmp_path)
    width, height = [cached_pickle(config[role + 'lib_pkl']) for role in ('width', 'height')]
    first = profile_library_provenance(config, axes, widthlib=width, heightlib=height)
    original = json.loads(json.dumps(first, allow_nan=False))
    first['generation']['psf'] = 'unsupported claim'
    first['runtime_interpretation']['axes']['n'][0] = 500.
    assert profile_library_provenance(config, axes, widthlib=width, heightlib=height) == original


def test_stale_loaded_profile_array_cannot_be_bound_to_new_file_bytes(tmp_path):
    config, axes = profile_fixture(tmp_path)
    old_width, old_height = [cached_pickle(config[role + 'lib_pkl']) for role in ('width', 'height')]
    profile_library_provenance(config, axes, widthlib=old_width, heightlib=old_height)
    # Changing the bytes/size invalidates the cache key and the old loaded object.
    path = Path(config['heightlib_pkl'])
    with path.open('wb') as handle:
        replacement = np.empty((1, 1, 1, 1), dtype=object)
        replacement[0, 0, 0, 0] = np.array([.1, .2, .3, .4])
        pickle.dump(replacement, handle)
    with pytest.raises(ValueError, match='changed after'):
        profile_library_provenance(config, axes, widthlib=old_width, heightlib=old_height)


def test_resource_change_before_publication_aborts_atomic_product(tmp_path):
    config, axes = profile_fixture(tmp_path)
    description = profile_library_provenance(config, axes)
    destination = tmp_path / 'product.h5'
    with pytest.raises(ValueError, match='changed before publication'):
        with atomic_hdf5(destination) as output:
            output['ID'] = np.array([10])
            row = resolve_spectral_extraction(config, np.array([.1, .4, .1]))
            write_spectral_extraction(output, [row], config, description)
            Path(config['heightlib_pkl']).write_bytes(b'changed calibration resource')
            record_runner_provenance(output, config,
                dict(apply_morphology_effect=False, apply_wavelength_calibration_error=False,
                     apply_photometric=False, apply_el_detect=False))
    assert not destination.exists()
    assert not list(tmp_path.glob('*.partial'))


def test_zero_fraction_and_invalid_rows_are_distinct_in_metadata(tmp_path):
    config, axes = profile_fixture(tmp_path)
    config['spectral_extraction'] = dict(mode='explicit', source_fraction=0., spatial_pixels=4)
    row = resolve_spectral_extraction(config, np.array([.1, .4, .1]))
    with h5py.File(str(tmp_path / 'metadata.h5'), 'w') as output:
        output['ID'] = np.array([10, 11])
        group = write_spectral_extraction(output, [row, None], config,
                                          profile_library_provenance(config, axes))
        np.testing.assert_array_equal(group['valid'][:], [True, False])
        assert group['source_fraction'][0] == 0.
        assert group['spatial_pixels'][0] == 4.
        for name in ('source_fraction', 'spatial_pixels', 'profile_height_sum'):
            assert np.isnan(group[name][1])
