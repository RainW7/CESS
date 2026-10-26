"""PA folding changes only symmetric projection-library lookup, not source PA."""
from pathlib import Path
import pickle
import random
import sys

import numpy as np
import pytest

CORE = Path(__file__).resolve().parents[1] / 'CESS_v0.8.6'
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

import morphology


def test_fold_scalar_array_boundaries_and_no_input_mutation():
    angles = np.asarray([[-360., -270., -180., -100., -90.],
                         [-60., -30., 0., 30., 60.],
                         [90., 100., 180., 270., 360.]])
    original = angles.copy()
    expected = [[0., 90., 0., 80., 90.], [60., 30., 0., 30., 60.],
                [90., 80., 0., 90., 0.]]
    np.testing.assert_array_equal(morphology.fold_position_angle(angles), expected)
    np.testing.assert_array_equal(angles, original)
    assert morphology.fold_position_angle(-60.) == 60.
    assert isinstance(morphology.fold_position_angle(np.asarray(90.)), float)
    assert morphology.fold_position_angle([]).shape == (0,)


def test_fold_mirror_and_180_degree_periodicity():
    angles = np.asarray([0., .125, 5., 15., 45., 75., 89.875, 90.])
    np.testing.assert_array_equal(morphology.fold_position_angle(-angles), angles)
    for turns in (-4, -1, 1, 4):
        np.testing.assert_array_equal(morphology.fold_position_angle(angles + 180. * turns), angles)


@pytest.mark.parametrize('angles', [np.nan, np.inf, -np.inf, [0., np.nan]])
def test_nonfinite_pa_is_rejected_for_helper_and_every_library_domain(angles):
    with pytest.raises(ValueError, match='finite angles'):
        morphology.fold_position_angle(angles)
    for grid in ([0., 90.], [-90., 0., 90.], [0., 90., 180., 270.]):
        with pytest.raises(ValueError, match='finite angles'):
            morphology.match_input_paramters(
                np.zeros((1, 1, len(grid), 1)), 1., 4., angles, .5,
                [1.], [4.], grid, [.5])


def test_fixed_library_indices_other_axes_and_nearest_ties_are_preserved():
    library = np.arange(2 * 2 * 10 * 2).reshape(2, 2, 10, 2)
    nseries, reseries = np.asarray([1., 4.]), np.asarray([4., 20.])
    paseries, baseries = np.arange(0., 91., 10.), np.asarray([.2, .8])
    for angle, index in [(-60., 6), (-90., 9), (90., 9), (180., 0),
                         (-15., 1), (15., 1), (-45., 4), (45., 4)]:
        chosen = morphology.match_input_paramters(library, 3.9, 19., angle, .75,
            nseries, reseries, paseries, baseries)
        assert chosen == library[1, 1, index, 1]
    # Existing np.argmin tie behavior selects the first grid entry, including
    # unsorted custom samples within the symmetric projection domain.
    assert morphology.match_input_paramters(np.asarray([[[[7], [8]]]]), 1., 4., -15., .5,
        np.asarray([1.]), np.asarray([4.]), np.asarray([20., 10.]), np.asarray([.5])) == 7


def test_principal_range_nextafter_ties_keep_original_nearest_precision():
    paseries = np.arange(0., 91., 10.)
    library = np.arange(10).reshape(1, 1, 10, 1)
    angles = np.asarray([np.nextafter(5., -np.inf), 5., np.nextafter(5., np.inf)])
    np.testing.assert_array_equal(morphology.fold_position_angle(angles), angles)
    np.testing.assert_array_equal(morphology.fold_position_angle(-angles), angles)
    for angle in angles:
        expected = int(np.argmin(np.abs(paseries - angle)))
        for signed_angle in (angle, -angle):
            actual = morphology.match_input_paramters(library, 1., 4., signed_angle, .5,
                np.asarray([1.]), np.asarray([4.]), paseries, np.asarray([.5]))
            assert actual == expected


@pytest.mark.parametrize('grid', [[-90., -60., 0., 60., 90.],
                                 [0., 90., 180., 270.]])
def test_signed_and_full_angle_custom_libraries_keep_original_nearest(grid):
    grid = np.asarray(grid)
    library = np.arange(len(grid)).reshape(1, 1, len(grid), 1)
    for angle in (-270., -90., -60., -15., 0., 60., 90., 180., 270., 360.):
        expected = int(np.argmin(np.abs(np.asarray(grid) - angle)))
        assert morphology.match_input_paramters(library, 1., 4., angle, .5,
            np.asarray([1.]), np.asarray([4.]), grid, np.asarray([.5])) == expected


def test_lookup_does_not_change_source_pa_or_consume_numpy_or_python_rng():
    numpy_state, python_state = np.random.get_state(), random.getstate()
    source_pa = np.asarray(-60.)
    library = np.arange(10).reshape(1, 1, 10, 1)
    chosen = morphology.match_input_paramters(library, 1., 4., source_pa, .5,
        np.asarray([1.]), np.asarray([4.]), np.arange(0., 91., 10.), np.asarray([.5]))
    assert chosen == 6 and source_pa == -60.
    after = np.random.get_state()
    assert numpy_state[0] == after[0] and numpy_state[2:] == after[2:]
    np.testing.assert_array_equal(numpy_state[1], after[1])
    assert random.getstate() == python_state


def test_existing_profile_libraries_use_positive_mirror_and_preserve_positive_lookup():
    nseries = np.asarray([.5, .6, .7, .8, .9, 1., 1.1, 1.2, 1.3, 1.4,
                          1.5, 1.6, 1.8, 2., 2.5, 3., 3.5, 4., 4.5, 5.])
    reseries = np.round(np.asarray([.3, .5, .7, .9, 1., 1.2, 1.4, 1.6, 1.8, 2.,
                                   2.5, 3., 3.5, 4.5, 5., 5.5, 6., 6.5, 7., 7.4]) / .074)
    paseries, baseries = np.arange(0., 91., 10.), np.arange(1., 11.) / 10.
    indices = (5, 5, 6, 1)
    changed_profiles = 0
    for kind in ('width', 'height'):
        with (CORE / 'basic_file' / (kind + 'lib_20x20x10x10.pkl')).open('rb') as stream:
            library = pickle.load(stream)
        nidx, reidx, paidx, baidx = indices
        args = (library, nseries[nidx], reseries[reidx], -60., baseries[baidx],
                nseries, reseries, paseries, baseries)
        selected = morphology.match_input_paramters(*args)
        assert selected is library[indices]
        old_clamped = library[nidx, reidx, 0, baidx]
        changed_profiles += (selected.shape != old_clamped.shape or
                             not np.array_equal(selected, old_clamped))
        for angle in (0., 15., 30., 45., 60., 90.):
            selected = morphology.match_input_paramters(library, nseries[nidx], reseries[reidx],
                angle, baseries[baidx], nseries, reseries, paseries, baseries)
            expected_index = int(np.argmin(np.abs(paseries - angle)))
            assert selected is library[nidx, reidx, expected_index, baidx]
    # The representative negative angle previously selected the zero-degree
    # entry; at least one real marginal now differs from that erroneous entry.
    assert changed_profiles > 0
