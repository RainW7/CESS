"""C4 line products on tiny temporary standard/paired/legacy-parent fixtures."""
import importlib
from pathlib import Path
import sys
import h5py
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / 'src'
sys.path[:0] = [str(CORE), str(ROOT)]

from cess.config.loader import load_config
from cess.lines.measurement import measure_line_window
from cess.engine import emulator
from test_pipeline_smoke import make_input


def simulation(tmp_path, module='cess.pipeline.standard', count=2, invalid=(0,), wave_error=False):
    inputs = tmp_path / 'input'
    inputs.mkdir()
    source = inputs / 'synthetic.hdf5'
    make_input(source, count=count, invalid=invalid)
    # Strong, finely resolved synthetic features make the I/O checks independent
    # of a marginal noisy detection. Invalid rows remain entirely NaN.
    with h5py.File(str(source), 'r+') as handle:
        values = handle['best_fit/spec_synthetic'][:]
        handle['best_fit/spec_synthetic'][:] = 1. + (values - 1.) * 20.
    config = load_config(CORE / 'cess/data/emulator_parameters.json')
    config['output_path'] = str(tmp_path / 'output')
    config['line_measurement'] = dict(line_width_pixels=dict(GV=10, GI=10),
        sideband_width_pixels=5, sideband_gap_pixels=3)
    options = dict(morph=False, wave_cal=wave_error, photo=False, el_detect=True)
    output = importlib.import_module(module).task(str(inputs), source.name,
        config=config, options=options, seed=1214)
    return Path(output), config


def check_tables(group, suffix='', allowed_source_rows=()):
    measurements = group['line_measurements' + suffix]
    assert 'sideband' in measurements.attrs['algorithm']
    for name in ('candidates', 'truth'):
        dataset = measurements[name]
        assert dataset.ndim == 1 and dataset.dtype.names is not None
        assert dataset.attrs['axis_order'] == 'peak'
        assert dataset.attrs['flux_unit'] == 'erg/s/cm^2'
        rows = dataset[:]
        assert set(rows['source_row']).issubset(set(allowed_source_rows))
        assert rows['source_row'].dtype.kind in 'iu'
        assert rows['peak_id'].dtype.kind in 'iu'
        for field in ('flux', 'flux_error', 'snr', 'model_flux', 'model_snr'):
            assert rows[field].dtype.kind == 'f'
        assert rows['valid'].dtype.kind == 'b'
        if len(rows):
            assert np.all(rows['center_index'] >= 0)
    return measurements


@pytest.mark.parametrize('count,invalid', [(0, ()), (1, (0,))])
def test_empty_and_invalid_standard_line_products_keep_fixed_rank(tmp_path, count, invalid):
    path, _ = simulation(tmp_path, count=count, invalid=invalid)
    with h5py.File(str(path), 'r') as output:
        for band in ('GU', 'GV', 'GI'):
            group = output[band]
            for suffix in ('', '_0th'):
                covariance = group['covariance_lag1' + suffix]
                assert covariance.shape == (count, len(group['wave']) - 1)
                assert covariance.attrs['unit'] == 'uJy^2'
                assert covariance.attrs['axis_order'] == 'source,pixel_pair'
                assert np.isnan(covariance[:]).all()
            if band != 'GU':
                tables = check_tables(group)
                assert tables['candidates'].shape == tables['truth'].shape == (0,)
                assert group['detect_el_snr'].attrs['unit'] == '1'
                assert 'not integrated' in group['detect_el_snr'].attrs['definition']


def test_single_standard_line_flux_matches_saved_spectral_inputs(tmp_path):
    path, config = simulation(tmp_path, count=1, invalid=())
    checked = 0
    with h5py.File(str(path), 'r') as output:
        for band in ('GV', 'GI'):
            group = output[band]
            tables = check_tables(group, allowed_source_rows=(0,))
            covariance = group['covariance_lag1'][0]
            assert np.isfinite(covariance).any()
            np.testing.assert_array_equal(covariance[np.isfinite(covariance)], 0.)
            rows = tables['candidates'][:]
            good = rows[rows['valid']]
            if len(good):
                row = good[0]
                design = config['line_measurement']
                result = measure_line_window(group['wave'][:], group['flux_ujy_with_noise'][0],
                    group['ferr'][0], int(row['center_index']),
                    design['line_width_pixels'][band], design['sideband_width_pixels'],
                    sideband_gap_pixels=design['sideband_gap_pixels'],
                    covariance_lag1=covariance,
                    mask=(group['spec_mask'][0] & group['calibration_support_mask'][0]),
                    model_ujy=group['flux_ujy_model'][0])
                assert row['flux'] == pytest.approx(result['flux'], rel=1.e-12)
                assert row['flux_error'] == pytest.approx(result['error'], rel=1.e-12)
                assert row['snr'] == pytest.approx(result['snr'], rel=1.e-12)
                checked += 1
        assert checked > 0


def test_paired_wce_covariance_variants_and_el_only_subset(tmp_path, monkeypatch):
    # A controlled half-pixel registration supplies a nonzero covariance and
    # exercises independent WCE peak detection without relying on a random draw.
    monkeypatch.setattr(emulator, '_sample_wave_offset', staticmethod(lambda rng=None: .5))
    path, config = simulation(tmp_path, module='cess.pipeline.paired', wave_error=True)
    with h5py.File(str(path), 'r') as parent:
        for band in ('GU', 'GV', 'GI'):
            group = parent[band]
            base = group['covariance_lag1'][:]
            shifted = group['covariance_lag1_wce'][:]
            assert base.shape == shifted.shape == (2, len(group['wave']) - 1)
            assert np.isnan(base[0]).all() and np.isnan(shifted[0]).all()
            np.testing.assert_array_equal(base[1, np.isfinite(base[1])], 0.)
            assert np.any(shifted[1, np.isfinite(shifted[1])] > 0)
            wave, width, response = [group[key][:] for key in
                                      ('wave', 'pixel_width', 'response_at_pixel_center')]
            calibration = np.full_like(wave, np.nan)
            np.divide(wave, width * response, out=calibration, where=response > 0)
            expected = .25 * group['ferr'][1, :-1] ** 2 * calibration[1:] / calibration[:-1]
            valid = np.isfinite(expected) & np.isfinite(shifted[1])
            np.testing.assert_allclose(shifted[1, valid], expected[valid], rtol=2.e-12)
            if band != 'GU':
                for suffix in ('', '_wce'):
                    tables = check_tables(group, suffix, allowed_source_rows=(1,))
                    candidates = {(int(row['source_row']), int(row['peak_id']))
                                  for row in tables['candidates'][:]}
                    truths = {(int(row['source_row']), int(row['peak_id']))
                              for row in tables['truth'][:]}
                    for pair in group['el_diagnostics' + suffix + '/pairs'][:]:
                        assert (int(pair['id']), int(pair['noisy_peak_id'])) in candidates
                        assert (int(pair['id']), int(pair['intrinsic_peak_id'])) in truths
                assert group['line_measurements'].attrs['variant'] == 'unregistered'
                assert group['line_measurements_wce'].attrs['variant'] == 'registered_wce'

    import cess.pipeline.emission as run_el_detect
    derived = run_el_detect.task(str(path.parent), path.name, output_dir=tmp_path / 'el',
        config=config, seed=1214, source_start=1, source_stop=2)
    with h5py.File(str(path), 'r') as parent, h5py.File(str(derived), 'r') as output:
        np.testing.assert_array_equal(output['parent_row_indices'][:], [1])
        np.testing.assert_array_equal(output['input_row_indices'][:], [1])
        for band in ('GV', 'GI'):
            for suffix in ('', '_wce'):
                tables = check_tables(output[band], suffix, allowed_source_rows=(0,))
                for kind in ('candidates', 'truth'):
                    old, new = parent[band + '/line_measurements' + suffix + '/' + kind][:], tables[kind][:]
                    assert len(old) == len(new)
                    for name in new.dtype.names:
                        if name == 'source_row':
                            np.testing.assert_array_equal(new[name], 0)
                        elif new[name].dtype.kind == 'f':
                            np.testing.assert_allclose(new[name], old[name], rtol=0, atol=0, equal_nan=True)
                        else:
                            np.testing.assert_array_equal(new[name], old[name])


def test_old_parent_missing_uncertainties_keeps_peak_rows_flagged_unknown(tmp_path):
    import cess.pipeline.emission as run_el_detect
    source = tmp_path / 'legacy.hdf5'
    with h5py.File(str(source), 'w') as parent:
        parent['ID'] = np.array([b'invalid', b'legacy'])
        parent['input_row_indices'] = [100, 205]
        for band, limits in (('GV', (3800., 6600.)), ('GI', (5800., 10500.))):
            group = parent.create_group(band)
            wave = np.linspace(limits[0], limits[1], 500)
            values = 1. + 200. * np.exp(-.5 * ((np.arange(500) - 250.) / 3.) ** 2)
            flux = np.vstack([np.full(500, np.nan), values])
            group['wave'] = wave
            group['flux_ujy'] = flux
            group['flux_ujy_with_noise'] = flux
            group['snr'] = np.ones_like(flux)
            if band == 'GV':
                group['ferr'] = np.ones_like(flux)
                group['flux_ujy_with_noise_wce'] = flux
                group['ferr_wce'] = np.ones_like(flux)
    config = load_config(CORE / 'cess/data/emulator_parameters.json')
    output_path = run_el_detect.task(str(tmp_path), source.name, output_dir=tmp_path / 'el',
        config=config, source_start=1, source_stop=2)
    with h5py.File(str(output_path), 'r') as output:
        np.testing.assert_array_equal(output['parent_row_indices'][:], [1])
        np.testing.assert_array_equal(output['input_row_indices'][:], [205])
        assert output.attrs['spectral_convolution_algorithm'].startswith('unknown')
        for band, flag in (('GV', 512), ('GI', 256)):
            measurements = check_tables(output[band], allowed_source_rows=(0,))
            for kind in ('candidates', 'truth'):
                rows = measurements[kind][:]
                assert len(rows) == 1 and not rows['valid'][0]
                assert rows['flag'][0] & flag
                assert np.isnan(rows['flux_error'][0]) and np.isnan(rows['snr'][0])
            for prefix in ('intri', 'detect'):
                assert output[band.lower() + '_' + prefix + '_el_wave'].attrs['unit'] == 'Angstrom'
            assert 'not integrated' in output[band.lower() + '_detect_el_snr'].attrs['definition']
        assert 'line_measurements_wce' in output['GV']
        assert 'el_diagnostics_wce' in output['GV']
        assert 'line_measurements_wce' not in output['GI']
