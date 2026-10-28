"""Independent numerical and pipeline checks for the declared 2511.06917 profile.

The tests check the implemented Gaussian approximation and pixel geometry, not
an independently calibrated CSST LSF or the scientific fidelity of R=300 input.
Historical quadrature tests retain a separate, explicit D4 configuration.
"""
import importlib
import json
import math
from pathlib import Path
import warnings

import h5py
import numpy as np
import pytest
from scipy.special import ndtr

from cess.config.loader import load_config
from cess.engine import emulator
from cess.spectra.integration import bin_edges
from test_pipeline_smoke import make_input
from test_spectral_repair import (FWHM_SIGMA, gaussian_input,
                                 photon_bins, reference_bin_fractions)


PAPER = {
    'GU': (2550., 4100., 3370., 287., 529),
    'GV': (4000., 6400., 5250., 232., 425),
    'GI': (6200., 10000., 8100., 207., 389),
}


@pytest.mark.parametrize('band,center', [('GU', 3200.), ('GV', 5250.), ('GI', 8100.)])
def test_paper_profile_gaussian_response_against_independent_continuous_quad(band, center):
    config = load_config()
    engine = emulator(config=config)
    lo, hi, reference, resolving_power, count = PAPER[band]
    target = reference / resolving_power
    assert count == math.ceil((hi - lo) / (target / 4.))
    redshift = .5
    input_sigma = center / 300. / FWHM_SIGMA
    rest, flux, truth = gaussian_input(center, input_sigma, redshift, samples_per_sigma=100)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Template LSF is broader')
        wave, output, valid, invalid = engine.kernels_convolve(
            rest, flux[None, :], [redshift], band, config[band.lower() + '_ree80'])
    assert valid.tolist() == [True] and invalid.tolist() == [False]
    edges = bin_edges(wave)
    expected_edges = np.linspace(lo, hi, count + 1)
    np.testing.assert_allclose(edges, expected_edges, rtol=0, atol=4.e-12)
    np.testing.assert_allclose(wave, .5 * (expected_edges[:-1] + expected_edges[1:]),
                               rtol=0, atol=2.e-12)
    def additional_sigma(wavelength):
        return np.sqrt(max(0., target ** 2 - (wavelength / 300.) ** 2)) / FWHM_SIGMA
    expected = reference_bin_fractions(edges, center, input_sigma, additional_sigma)
    actual = photon_bins(wave, output[0]) / truth
    np.testing.assert_allclose(actual, expected, rtol=8.e-5, atol=3.e-8)
    assert actual.sum() == pytest.approx(1., rel=2.e-6)
    assert (wave @ actual) / actual.sum() == pytest.approx(
        (wave @ expected) / expected.sum(), abs=3.e-4)


def test_paper_profile_gu_red_template_is_preserved_without_deconvolution():
    config = load_config()
    center, redshift = 3800., .4
    sigma = center / 300. / FWHM_SIGMA
    rest, flux, truth = gaussian_input(center, sigma, redshift, samples_per_sigma=100)
    config['template_lsf_broader_policy'] = 'error'
    with pytest.raises(ValueError, match='broader'):
        emulator(config=config).kernels_convolve(rest, flux[None, :], [redshift], 'GU', config['gu_ree80'])
    config['template_lsf_broader_policy'] = 'preserve'
    with pytest.warns(RuntimeWarning, match='broader'):
        wave, output, valid, _ = emulator(config=config).kernels_convolve(
            rest, flux[None, :], [redshift], 'GU', config['gu_ree80'])
    assert valid[0]
    edges = bin_edges(wave)
    a, b = (edges[:-1] - center) / sigma, (edges[1:] - center) / sigma
    normal_a = np.exp(-.5 * a ** 2) / np.sqrt(2. * np.pi)
    normal_b = np.exp(-.5 * b ** 2) / np.sqrt(2. * np.pi)
    expected = ndtr(b) - ndtr(a) + sigma / center * (normal_a - normal_b)
    np.testing.assert_allclose(photon_bins(wave, output[0]) / truth, expected,
                               rtol=8.e-5, atol=2.e-8)


@pytest.mark.parametrize('module_name', ['cess.pipeline.standard', 'cess.pipeline.paired'])
def test_new_default_all_effects_keeps_whole_file_geometry_invalid_rows_and_lines(tmp_path, module_name):
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    source = inputs / 'paper_profile.hdf5'
    make_input(source, count=2, invalid=(0,))
    with h5py.File(source, 'r+') as handle:
        values = handle['best_fit/spec_synthetic'][:]
        handle['best_fit/spec_synthetic'][:] = 1. + (values - 1.) * 20.
    config = load_config()
    options = dict(morph=True, wave_cal=True, photo=True, el_detect=True)
    path = importlib.import_module(module_name).simulate_file(
        str(inputs), source.name, output_dir=tmp_path / 'products',
        config=config, options=options, seed=20261027, output_layout='structured')
    with h5py.File(path) as output:
        assert output.attrs['schema_version'] == '3.0'
        assert output['provenance'].attrs['cess_repair_revision'] == '20261027-D5'
        np.testing.assert_array_equal(output['catalog/data_mask'][:], [2, 1])
        np.testing.assert_array_equal(output['catalog/input_row_indices'][:], [0, 1])
        for band, (lo, hi, _, _, count) in PAPER.items():
            group = output['spectra/' + band]
            assert group['flux_ujy'].shape == (2, count)
            assert np.isnan(group['flux_ujy_with_noise'][0]).all()
            edges = group['wave_edges'][:]
            assert edges[0] == pytest.approx(lo, abs=4.e-12)
            assert edges[-1] == pytest.approx(hi, abs=4.e-12)
            assert np.any(group['spec_mask'][1])
        # Testing an explicitly strong line prevents an empty-table regression
        # when new grid sizes invalidate historical hard-coded end indexes.
        for band in ('GV', 'GI'):
            assert len(output['lines/' + band + '/line_measurements/truth']) > 0


def test_new_bandpass_postprocessing_uses_the_actual_old_parent_grid(tmp_path):
    source = tmp_path / 'old_grid.hdf5'
    grids = {}
    with h5py.File(source, 'w') as parent:
        parent['ID'] = np.array([b'old-grid-source'])
        parent['input_row_indices'] = [123]
        for band, lo, hi, peaks in (
                ('GV', 3800., 6600., (3950., 5200., 6500.)),
                ('GI', 5800., 10500., (6000., 8100., 10200.))):
            wave = np.linspace(lo, hi, 500)
            flux = np.ones(500)
            for center in peaks:
                flux += 500. * np.exp(-.5 * ((wave - center) / 15.) ** 2)
            group = parent.create_group(band)
            group['wave'] = wave
            group['flux_ujy'] = flux[None, :]
            group['flux_ujy_with_noise'] = flux[None, :]
            group['snr'] = np.ones((1, 500))
            grids[band] = wave
    config = load_config()
    path = importlib.import_module('cess.pipeline.emission').task(
        str(tmp_path), source.name, output_dir=tmp_path / 'postprocessed', config=config, seed=20261027)
    with h5py.File(path) as output:
        recorded_windows = json.loads(output.attrs['analysis_window_json'])
        assert set(recorded_windows) == {'GV', 'GI'}
        for band, expected_center in (('GV', 5200.), ('GI', 8100.)):
            wave = grids[band]
            initial = int(np.searchsorted(wave, PAPER[band][0], side='left'))
            stop = int(np.searchsorted(wave, PAPER[band][1], side='right'))
            assert recorded_windows[band] == {'start_index': initial, 'stop_index': stop}
            centers = output[band + '/line_measurements/truth'][:]
            assert len(centers) == 1
            assert centers['center_wave'][0] == pytest.approx(expected_center, abs=6.)
            index = int(centers['center_index'][0])
            assert wave[index] == centers['center_wave'][0]
            local_index = int(output[band.lower() + '_intri_el_idx'][0][0])
            assert local_index + initial == index


@pytest.mark.parametrize('band', ['GV', 'GI'])
def test_bandpass_detection_trims_only_nonfinite_edges_and_keeps_index_origin(band):
    from legacy_config import load_legacy_config
    lo, hi, _, _, _ = PAPER[band]
    wave = np.linspace(lo, hi, 500)
    peak = 250
    flux = 1. + 500. * np.exp(-.5 * ((np.arange(500) - peak) / 3.) ** 2)
    flux[:20], flux[-17:] = np.nan, np.nan
    initial, stop = 5, 495
    model = emulator(config=load_config())
    lines = model.el_detect(wave, flux, initial, stop, 1.5)
    assert len(lines) == 1
    local_index = int(lines['line_center_index'][0])
    assert initial + local_index == peak
    assert lines['line_center'].value[0] == wave[peak]
    # The edge repair is not permission to interpolate across missing internal
    # wavelength bins. Such a window must remain explicitly unavailable.
    with_gap = flux.copy()
    with_gap[200] = np.nan
    assert len(model.el_detect(wave, with_gap, initial, stop, 1.5)) == 0
    assert len(emulator(config=load_legacy_config()).el_detect(
        wave, flux, initial, stop, 1.5)) == 0
