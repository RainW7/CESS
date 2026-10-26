"""C13 shared preparation keeps resource, random-stream and product contracts."""
from pathlib import Path
from types import SimpleNamespace
import sys

import h5py
import numpy as np
import pytest
from astropy.table import Table

CORE = Path(__file__).resolve().parents[1] / 'CESS_v0.8.6'
sys.path.insert(0, str(CORE))
from cess_pipeline import (BANDS, PHOTOMETRIC_BANDS, prepare_resources,
    prepare_source_profile, measure_source_bands, photometry_columns,
    write_legacy_el_products, write_common_products)
from cess_random import source_rng, legacy_rng
from runner_support import throughput_arrays, throughput_interpolator, write_vlen_rows


@pytest.mark.parametrize('paired', [False, True])
@pytest.mark.parametrize('mge', [False, True])
def test_resource_setup_uses_only_mode_resources_and_late_bound_operations(monkeypatch, paired, mge):
    calls = []
    background = Table([[1000., 2000.], [1., 1.]], names=['wavelength', 'fnu'])
    response = Table([[1000., 2000.], [.1, .2]], names=['Wavelength', 'Efficiency'])
    def cached(path):
        calls.append(path)
        return background if path == 'background' else response
    axes = {name: np.array([1.]) for name in ('n', 're_pixel', 'pa_deg', 'axis_ratio')}
    config = dict(bkg_spec_path='background', spectral_extraction={'mode': 'mge_strip' if mge else 'legacy_full'})
    for band in BANDS:
        config[band.lower() + 'tp_path'] = band + '-first'
        config[band.lower() + 'tp0_path'] = band + '-zero'
    for band in PHOTOMETRIC_BANDS:
        config[band + 'tp_path'] = band
    context = {'provenance': {'version': 3}}
    import cess_mge_runtime
    monkeypatch.setattr(cess_mge_runtime, 'load_mge_source_library', lambda settings: context)
    def libraries(settings, supplied_axes):
        assert not mge and supplied_axes is axes
        return 'width', 'height', axes, {'version': 1}
    operations = dict(configure_morphology=lambda settings: calls.append('configure'),
        load_profile_libraries=libraries, cached_table=cached,
        throughput_arrays=throughput_arrays, throughput_interpolator=throughput_interpolator)
    result = prepare_resources(config, axes, True, not paired, operations)
    expected = ['configure', 'background'] + [band + '-first' for band in BANDS]
    if not paired:
        expected += [band + '-zero' for band in BANDS]
    assert calls == expected + list(PHOTOMETRIC_BANDS)
    assert tuple(result['photometric_tables']) == PHOTOMETRIC_BANDS
    assert tuple(result['zero_tables']) == (() if paired else BANDS)
    assert all(set(curves) == ({'first'} if paired else {'first', 'zero_legacy'})
               for curves in result['response_curves'].values())
    assert result['mge_context'] is (context if mge else None)
    assert result['morphology_noise_options'] == ({'normalize_morphology_kernel': False} if mge else {})


@pytest.mark.parametrize('seed', [None, 1214])
@pytest.mark.parametrize('paired', [False, True])
def test_source_preparation_preserves_component_order_and_unseeded_global_stream(seed, paired):
    saved = np.random.get_state()
    try:
        np.random.seed(317)
        calls, draws, lookups = [], [], []
        def row_rng(component):
            calls.append(component)
            return source_rng(seed, 'a' * 64, 9, 5, component)
        def draw_profile(*args):
            draws.append(np.random.random())
            return .7, 4., -30., .6
        def lookup(library, *args):
            lookups.append((library, args))
            return np.array([.2, .6, .2])
        def measure(*args, rng=None):
            value = np.random.random() if rng is None else rng.random_sample()
            draws.append(value)
            return dict(model_mag=value, model_mag_error=value+1., expected_snr=value+2.)
        morphology = SimpleNamespace(get_2d_param=draw_profile, match_input_paramters=lookup)
        engine = SimpleNamespace(match_bkg_mag=lambda ra, dec: 22.)
        settings = dict(background_mode='fixed_template')
        axes = {name: [1.] for name in ('n', 're_pixel', 'pa_deg', 'axis_ratio')}
        resources = dict(mge_context=None, widthlib='width', heightlib='height', axes=axes)
        parameters = Table([[12.], [13.], [20.], [9.], [.2]],
                           names=['RA', 'DEC', 'MAG_Z', 'log(M*)[0,1]_{MAL}', 'z_{MAL}'])
        prepared = dict(parameters=parameters, sky_reference={'GU': {'first': 3.}},
                        sky_rates={'GU': {'first': np.full(1, np.nan)}})
        operations = dict(legacy_rng=legacy_rng, measure_runner_band=measure,
            resolve_spectral_extraction=lambda settings, height: dict(
                source_fraction=1., spatial_pixels=len(height), profile_height_sum=height.sum()))
        selected = prepare_source_profile(engine, morphology, settings, resources, prepared,
                                           0, row_rng, True, operations)
        assert selected[:4] == (.7, 4., -30., .6)
        assert [entry[0] for entry in lookups] == ['width', 'height']
        assert prepared['sky_rates']['GU']['first'][0] == 3.
        if not paired:
            measure_source_bands(engine, 12., 13., [], [], {band: band for band in BANDS}, None,
                settings, dict.fromkeys(BANDS, 2), 4., row_rng, 'zero_', operations)
        measure_source_bands(engine, 12., 13., [], [], {band: band for band in PHOTOMETRIC_BANDS}, None,
            settings, dict.fromkeys(PHOTOMETRIC_BANDS, 2), 4., row_rng, 'photometry_', operations)
        expected_components = ['morphology'] + ([] if paired else ['zero_' + b for b in BANDS])
        expected_components += ['photometry_' + b for b in PHOTOMETRIC_BANDS]
        assert calls == expected_components
        if seed is None:
            expected_rng = np.random.RandomState(317)
            np.testing.assert_array_equal(draws, expected_rng.random_sample(len(draws)))
            assert np.random.random() == expected_rng.random_sample()
        else:
            expected = [source_rng(seed, 'a' * 64, 9, 5, component).random_sample()
                        for component in expected_components]
            np.testing.assert_array_equal(draws, expected)
            assert np.random.random() == np.random.RandomState(317).random_sample()
    finally:
        np.random.set_state(saved)


def test_legacy_photometric_columns_keep_field_then_band_order():
    results = {band: dict(model_mag=i, model_mag_error=i+10., expected_snr=i+20.)
               for i, band in enumerate(PHOTOMETRIC_BANDS)}
    assert photometry_columns(results, PHOTOMETRIC_BANDS) == tuple(
        list(range(7)) + list(np.arange(7)+10.) + list(np.arange(7)+20.))


def peak_table(count, count_name):
    table = np.empty(count, dtype=[('id', 'i8'), ('wave', object), ('idx', object), (count_name, 'i8')])
    for i in range(count):
        table[i] = (i, np.array([5007.]), np.array([3]), 1)
    return table


@pytest.mark.parametrize('count', [0, 1])
def test_shared_legacy_el_writer_keeps_empty_and_singleton_types_and_units(tmp_path, count):
    seen = []
    operations = dict(write_vlen_rows=write_vlen_rows,
                      write_el_details=lambda group, details: seen.append((group.name, details)))
    with h5py.File(str(tmp_path / 'products.hdf5'), 'w') as handle:
        rows = [(handle.create_group(band), peak_table(count, 'intri_el_number'),
                 peak_table(count, 'detect_el_number'), [np.array([4.])] * count, band)
                for band in ('GV', 'GI')]
        write_legacy_el_products(rows, operations)
        for band in ('GV', 'GI'):
            group = handle[band]
            assert group['intri_el_wave'].shape == (count,)
            assert h5py.check_dtype(vlen=group['detect_el_wave'].dtype) == np.dtype('float64')
            assert h5py.check_dtype(vlen=group['detect_el_idx'].dtype) == np.dtype('int64')
            assert group['detect_el_snr'].attrs['unit'] == '1'
            assert 'not integrated' in group['detect_el_snr'].attrs['definition']
            assert group['intri_el_wave'].attrs['unit'] == 'Angstrom'
            if count:
                np.testing.assert_array_equal(group['detect_el_snr'][0], [4.])
                np.testing.assert_array_equal(group['intri_el_wave'][0], [5007.])
    assert seen == [('/GV', 'GV'), ('/GI', 'GI')]


@pytest.mark.parametrize('paired', [False, True])
@pytest.mark.parametrize('count', [0, 1])
def test_common_forward_writer_keeps_variant_boundary_and_covariance_axes(tmp_path, paired, count):
    class Selection(dict):
        source_rows = np.arange(count, dtype=int) + 5
    selected = Selection(ID=np.arange(count))
    resources = dict(mge_context=None, profile_provenance={'version': 1}, response_curves={})
    prepared = dict(bands={}, sky_reference={}, sky_rates={})
    covariance, calls = {}, []
    variants = ('first',) if paired else ('first', 'zero_legacy')
    suffixes = ('', '_wce') if paired else ('', '_0th')
    wave = np.array([1000., 1500., 2000.])
    for band in BANDS:
        resources['response_curves'][band] = {key: (wave, np.ones(3)) for key in variants}
        prepared['bands'][band] = dict(wave=wave,
            folded={key: np.ones((count, 3)) for key in variants},
            source_rates={key: np.ones((count, 3)) for key in variants})
        prepared['sky_reference'][band] = dict.fromkeys(variants, 2.)
        prepared['sky_rates'][band] = {key: np.ones(count) * 2 for key in variants}
        covariance[band] = {suffix: [np.zeros(2)] * count for suffix in suffixes}
    def response(group, *args, **kwargs):
        calls.append(('response', group.name, kwargs['suffix'], kwargs['paired']))
    def cov(group, values, suffix):
        assert values.shape == (count, 2)
        calls.append(('covariance', group.name, suffix))
    operations = dict(write_spectral_extraction=lambda *args: calls.append(('extraction',)),
        write_mge_morphology=lambda *args: pytest.fail('legacy must not invent MGE diagnostics'),
        write_response_products=response, write_spectral_covariance=cov)
    with h5py.File(str(tmp_path / 'forward.hdf5'), 'w') as handle:
        groups = {band: handle.create_group(band) for band in BANDS}
        for group in groups.values():
            group['wave'] = wave
        write_common_products(handle, groups, selected, {}, resources, prepared,
                              [None]*count, [], covariance, paired, operations)
        np.testing.assert_array_equal(handle['input_row_indices'][:], selected.source_rows)
    expected = [('extraction',)]
    for band in BANDS:
        expected += [('response', '/' + band, suffix, paired)
                     for suffix in (('',) if paired else ('', '_0th'))]
    for band in BANDS:
        expected += [('covariance', '/' + band, suffix) for suffix in suffixes]
    assert calls == expected
