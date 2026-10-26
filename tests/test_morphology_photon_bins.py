"""Spatial redistribution must preserve detected counts rather than fnu area."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'CESS_v0.8.6'))
from main import emulator


@pytest.mark.parametrize('profile', [[1.], [.1, .2, .3, .4], [.2, .6, .2]])
def test_spatial_profile_count_sum_and_no_boundary_restoration(profile):
    model = emulator()
    counts = np.zeros(31)
    counts[15] = 123.
    broadened = model.lamshift_width(counts, profile)
    assert broadened.sum() == pytest.approx(counts.sum())
    edge = np.zeros(31)
    edge[0] = 123.
    cropped = model.lamshift_width(edge, profile)
    assert 0. <= cropped.sum() <= edge.sum()
    if len(profile) > 1:
        assert cropped[0] != edge[0]


def test_noise_morphology_redistributes_electrons_on_nonconstant_response():
    model = emulator()
    model.bkg_per_pix_time = lambda *args: 0.
    wave = np.linspace(4000., 6000., 201)
    flux = np.zeros_like(wave)
    flux[100] = 1.e-25
    response = np.linspace(.05, .9, len(wave))
    profile = np.array([.1, .2, .3, .4])
    args = (0., 0., wave, flux, wave, np.zeros_like(wave), response,
            np.ones(3), profile)
    plain = model._noise_realization(*args, False, rng=np.random.RandomState(1))
    spread = model._noise_realization(*args, True, rng=np.random.RandomState(1))
    expected = model.lamshift_width(plain['model'], profile)
    np.testing.assert_allclose(spread['model'][spread['valid']], expected[spread['valid']])
    assert np.nansum(spread['model']) == pytest.approx(np.sum(plain['model']))
    # The two finite-array edge regions lack full spatial-profile support.
    assert not spread['valid'][0] and not spread['valid'][-1]
    assert np.isnan(spread['model'][0]) and np.isnan(spread['variance'][-1])


def test_morphology_missing_input_support_is_not_zero_filled():
    model = emulator()
    model.bkg_per_pix_time = lambda *args: 0.
    wave = np.linspace(4000., 6000., 51)
    flux = np.zeros_like(wave)
    flux[25] = np.nan
    result = model._noise_realization(
        0., 0., wave, flux, wave, np.zeros_like(wave), np.ones_like(wave),
        np.ones(3), np.ones(3), True, rng=np.random.RandomState(1))
    np.testing.assert_array_equal(result['valid'][24:27], False)
    assert result['valid'][23] and result['valid'][27]


def test_zero_center_profile_weight_uses_only_required_neighbor_support():
    model = emulator()
    model.bkg_per_pix_time = lambda *args: 0.
    wave = np.linspace(4000., 6000., 51)
    flux = np.full_like(wave, 1.e-27)
    flux[25] = np.nan
    result = model._noise_realization(
        0., 0., wave, flux, wave, np.zeros_like(wave), np.ones_like(wave),
        np.ones(3), np.array([1., 0., 1.]), True, rng=np.random.RandomState(1))
    assert result['valid'][25] and np.isfinite(result['model'][25])
    assert not result['valid'][24] and not result['valid'][26]
