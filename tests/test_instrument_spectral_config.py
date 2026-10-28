"""Instrument definitions, sampling geometry and old-mapping compatibility.

Reference numbers here test a declared 1D approximation to paper central R.
They do not certify a measured CSST wavelength-dependent LSF or pixel dispersion.
"""
from copy import deepcopy

import numpy as np
import pytest

from cess.config.loader import validate_config
from cess.instrument.spectral import (MAX_SPECTRAL_PIXELS, output_pixel_count,
    output_wavelengths, target_fwhm, window_indices)
from cess.spectra.integration import bin_edges


def old_mapping():
    result = dict(radi=100., expt=150., expnum=4, arcsecperpix=.074, dc_value=.02)
    for band, start, stop, low, high, delta, resolving, spatial, index_stop in (
            ('gu', 2000., 4400., 2500., 4200., 15., 225., .17, 318),
            ('gv', 3600., 6800., 3800., 6600., 21., 250., .19, 373),
            ('gi', 5600., 12000., 5800., 10500., 33., 250., .19, 496)):
        result.update({band + '_start_wl': start, band + '_end_wl': stop,
            band + '_min_wl': low, band + '_max_wl': high, band + '_dlambda': delta,
            band + '_Res': resolving, band + '_ree80': spatial,
            band + '_init_idx': 20, band + '_end_idx': index_stop})
    return result


def paper_mapping():
    result = old_mapping()
    result.update(instrument_lsf={'mode': 'central_resolving_power', 'wavelength_law': 'constant_fwhm'},
        spectral_sampling={'mode': 'resolution_elements', 'pixels_per_resolution_element': 4.},
        spectral_window={'mode': 'bandpass'})
    for band, center, low, high, resolving in (
            ('gu', 3370., 2550., 4100., 287.),
            ('gv', 5250., 4000., 6400., 232.),
            ('gi', 8100., 6200., 10000., 207.)):
        result.update({band + '_central_wl': center, band + '_Res': resolving,
            band + '_start_wl': low, band + '_end_wl': high,
            band + '_min_wl': low, band + '_max_wl': high})
    return result


def test_new_model_derives_target_without_mutating_input_or_trusting_old_cache():
    original = paper_mapping()
    frozen = deepcopy(original)
    config = validate_config(original)
    assert original == frozen
    for band, reference in [('gu', 3370. / 287.), ('gv', 5250. / 232.), ('gi', 8100. / 207.)]:
        assert config[band + '_dlambda'] == reference == target_fwhm(config, band.upper())
        assert config[band + '_init_idx'] == 0
        assert config[band + '_end_idx'] == output_pixel_count(config, band)
    config['gu_Res'] *= 2
    assert target_fwhm(config, 'GU') == frozen['gu_central_wl'] / (frozen['gu_Res'] * 2)
    assert validate_config(config)['gu_dlambda'] == target_fwhm(config, 'GU')


@pytest.mark.parametrize('band,count', [('GU', 529), ('GV', 425), ('GI', 389)])
def test_resolution_sampling_has_exact_band_edges_and_bound_pixel_width(band, count):
    config = validate_config(paper_mapping())
    prefix = band.lower()
    centers = output_wavelengths(config, band)
    assert output_pixel_count(config, band) == len(centers) == count
    edges = bin_edges(centers)
    np.testing.assert_allclose(edges[[0, -1]],
        [config[prefix + '_min_wl'], config[prefix + '_max_wl']], rtol=0, atol=3.e-11)
    assert np.max(np.diff(edges)) <= target_fwhm(config, band) / 4 * (1 + 2.e-12)
    assert window_indices(config, band, centers) == (0, count)
    # New sampling cannot change when a caller supplies a different spatial size.
    np.testing.assert_array_equal(output_wavelengths(config, band, .8), centers)


@pytest.mark.parametrize('band,count', [('GU', 370), ('GV', 393), ('GI', 498)])
def test_old_mapping_keeps_exact_counts_centers_and_dlambda_authority(band, count):
    raw = old_mapping()
    config = validate_config(raw)
    prefix = band.lower()
    assert config['instrument_lsf'] == {'mode': 'fixed_fwhm'}
    assert config['spectral_sampling'] == {'mode': 'legacy'}
    assert config['spectral_window'] == {'mode': 'indices'}
    assert output_pixel_count(config, band) == count
    expected = np.linspace(raw[prefix + '_min_wl'], raw[prefix + '_max_wl'], count)
    np.testing.assert_array_equal(output_wavelengths(config, band), expected)
    old_width = target_fwhm(config, band)
    config[prefix + '_Res'] *= 10
    assert target_fwhm(config, band) == old_width
    override = .21
    coarse = int((raw[prefix + '_end_wl'] - raw[prefix + '_start_wl']) / raw[prefix + '_dlambda']) + 1
    assert output_pixel_count(config, band, override) == round(coarse * override / raw['arcsecperpix'])


def test_bandpass_window_is_resolved_on_an_older_parent_grid():
    config = validate_config(paper_mapping())
    parent = np.array([3500., 3800., 4000., 4500., 5000., 6000., 6400., 6600., 7000.])
    assert window_indices(config, 'GV', parent) == (2, 7)
    assert window_indices(config, 'GV', np.array([1000., 2000.])) == (2, 2)
    # Cached indexes refer to the generated grid and must not be used for parent data.
    assert config['gv_end_idx'] != window_indices(config, 'GV', parent)[1]


def test_old_explicit_window_is_neither_clamped_nor_recomputed():
    config = validate_config(old_mapping())
    assert window_indices(config, 'GV', np.array([4000., 5000., 6000.])) == (20, 373)


@pytest.mark.parametrize('name,value', [
    ('instrument_lsf', {'mode': 'central_resolving_power', 'wavelength_law': 'constant_R'}),
    ('instrument_lsf', {'mode': 'fixed_fwhm', 'value': 1.}),
    ('instrument_lsf', []),
    ('spectral_sampling', {'mode': 'resolution_elements', 'pixels_per_resolution_element': 0.}),
    ('spectral_sampling', {'mode': 'resolution_elements', 'pixels_per_resolution_element': True}),
    ('spectral_sampling', {'mode': 'resolution_elements', 'pixels_per_resolution_element': np.bool_(True)}),
    ('spectral_sampling', {'mode': 'resolution_elements', 'pixels_per_resolution_element': np.nan}),
    ('spectral_sampling', {'mode': 'resolution_elements', 'pixels_per_resolution_element': 1.e10}),
    ('spectral_sampling', {'mode': 'invented'}),
    ('spectral_window', {'mode': 'bandpass', 'start': 1}),
    ('gu_Res', True), ('gu_Res', np.bool_(True)), ('gu_Res', 0.), ('gu_Res', np.inf),
    ('gu_central_wl', False), ('gu_central_wl', 1.), ('gu_central_wl', np.nan),
])
def test_invalid_new_models_reject_before_grid_allocation(name, value):
    config = paper_mapping()
    config[name] = value
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize('missing', ['gu_central_wl', 'gv_Res', 'gi_min_wl'])
def test_missing_new_source_of_truth_has_an_explicit_error(missing):
    config = paper_mapping()
    del config[missing]
    with pytest.raises(ValueError, match=missing):
        validate_config(config)


def test_new_derived_fields_need_not_be_supplied_as_duplicate_controls():
    config = paper_mapping()
    for band in ('gu', 'gv', 'gi'):
        for suffix in ('dlambda', 'init_idx', 'end_idx'):
            del config[band + '_' + suffix]
    result = validate_config(config)
    assert result['gu_dlambda'] == 3370. / 287.
    assert result['gi_end_idx'] == 389


@pytest.mark.parametrize('wave', [[], [5000., 4000.], [4000., np.nan], [[4000., 5000.]]])
def test_bandpass_window_rejects_an_invalid_actual_wave_grid(wave):
    config = validate_config(paper_mapping())
    with pytest.raises(ValueError, match='wavelength centers'):
        window_indices(config, 'GV', wave)


def test_sampling_limit_is_checked_without_allocating_an_unbounded_array():
    config = paper_mapping()
    config['spectral_sampling']['pixels_per_resolution_element'] = MAX_SPECTRAL_PIXELS
    with pytest.raises(ValueError, match='maximum'):
        output_wavelengths(config, 'GU')


def test_extreme_legacy_sampling_ratio_is_rejected_before_integer_conversion():
    config = old_mapping()
    config['gu_dlambda'] = 1.e-320
    with pytest.raises(ValueError, match='finite grid limit'):
        output_pixel_count(config, 'GU')
