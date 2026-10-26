"""C6 extraction assumptions and metadata on tiny temporary source fixtures."""
import importlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / 'src'
sys.path[:0] = [str(CORE), str(ROOT)]

from cess.config.loader import load_config
from cess.engine import emulator, h
from test_pipeline_smoke import make_input


def source_fixture(directory, count=1, invalid=()):
    inputs = directory / 'input'
    inputs.mkdir()
    source = inputs / 'synthetic.hdf5'
    make_input(source, count=count, invalid=invalid)
    config = load_config(CORE / 'cess/data/emulator_parameters.json')
    config['input_path'] = str(inputs)
    return source, config


def simulate(source, config, destination, extraction=None, module='cess.pipeline.standard', wave_error=False):
    settings = dict(config, output_path=str(destination))
    if extraction is not None:
        settings['spectral_extraction'] = extraction
    options = dict(morph=False, wave_cal=wave_error, photo=False, el_detect=False)
    path = importlib.import_module(module).task(str(source.parent), source.name,
        config=settings, options=options, seed=1214)
    return Path(path)


def explicit(fraction, pixels):
    return dict(mode='explicit', source_fraction=fraction, spatial_pixels=pixels)


def check_metadata(product, valid):
    group = product['spectral_extraction']
    count = len(product['ID'])
    np.testing.assert_array_equal(group['valid'][:], valid)
    assert group['valid'].shape == (count,) and group['valid'].dtype.kind == 'b'
    for name in ('source_fraction', 'spatial_pixels', 'profile_height_sum'):
        dataset = group[name]
        assert dataset.shape == (count,) and dataset.dtype.kind == 'f'
        assert dataset.attrs['axis_order'] == 'source'
        assert dataset.attrs['unit'] == ('pixel' if name == 'spatial_pixels' else '1')
        assert np.isnan(dataset[:][~np.asarray(valid, dtype=bool)]).all()
    description = json.loads(product.attrs['profile_library_provenance_json'])
    for role in ('width', 'height'):
        resource = description['resources'][role]
        assert len(resource['sha256']) == 64 and Path(resource['path']).is_absolute()
        assert tuple(resource['shape']) == (20, 20, 10, 10)
    return group


def test_default_explicit_unity_fraction_and_noise_area_contract(tmp_path):
    source, config = source_fixture(tmp_path)
    legacy_path = simulate(source, config, tmp_path / 'legacy')
    with h5py.File(str(legacy_path), 'r') as product:
        metadata = check_metadata(product, [True])
        assert json.loads(metadata.attrs['model_json']) == {'mode': 'legacy_full'}
        pixels = int(metadata['spatial_pixels'][0])
        assert metadata['source_fraction'][0] == 1.
    one_path = simulate(source, config, tmp_path / 'one', explicit(1., pixels))
    low_path = simulate(source, config, tmp_path / 'low', explicit(.4, pixels))
    wide_path = simulate(source, config, tmp_path / 'wide', explicit(.4, pixels + 7))
    zero_path = simulate(source, config, tmp_path / 'zero', explicit(0., pixels))
    with h5py.File(str(legacy_path), 'r') as legacy, h5py.File(str(one_path), 'r') as one:
        for name in ('ID', 'parameters_desi', 'parameters_grism', 'data_mask', 'input_row_indices'):
            np.testing.assert_array_equal(legacy[name][:], one[name][:])
        check_metadata(one, [True])
        for band in ('GU', 'GV', 'GI'):
            for name in ('flux_ujy', 'response_flux_ujy', 'source_electron_rate_pre_morph',
                         'flux_elec', 'flux_ujy_model', 'flux_ujy_with_noise', 'ferr', 'snr',
                         'covariance_lag1', 'flux_elec_0th', 'flux_ujy_with_noise_0th', 'ferr_0th'):
                np.testing.assert_array_equal(legacy[band + '/' + name][:], one[band + '/' + name][:])

    for path, fraction, count in ((low_path, .4, pixels), (wide_path, .4, pixels + 7),
                                  (zero_path, 0., pixels)):
        with h5py.File(str(legacy_path), 'r') as legacy, h5py.File(str(path), 'r') as product:
            metadata = check_metadata(product, [True])
            assert metadata['source_fraction'][0] == fraction
            assert metadata['spatial_pixels'][0] == count
            assert metadata['profile_height_sum'][0] == legacy['spectral_extraction/profile_height_sum'][0]
            for band in ('GU', 'GV', 'GI'):
                baseline, changed = legacy[band], product[band]
                # Incident/response products precede extraction; counts and the
                # calibrated noiseless model are in-aperture, without /fraction.
                np.testing.assert_array_equal(changed['response_flux_ujy'][:], baseline['response_flux_ujy'][:])
                for name in ('flux_elec', 'flux_ujy_model', 'flux_elec_0th', 'flux_ujy_model_0th'):
                    np.testing.assert_allclose(changed[name][:], fraction * baseline[name][:], rtol=2.e-13)
                wave, width, response = [baseline[key][:] for key in
                                          ('wave', 'pixel_width', 'response_at_pixel_center')]
                total_time = config['expt'] * config['expnum']
                area = np.pi * config['radi'] ** 2
                conversion = np.full_like(wave, np.nan)
                np.divide(h * wave / 1.e-29, area * total_time * width * response,
                          out=conversion, where=response > 0)
                sky = baseline['sky_rate_e_s_pixel'][0]
                noise_per_pixel = (sky + config['dc_value']) * total_time + config['read_noise'] ** 2 * config['expnum']
                expected_variance_e = fraction * baseline['flux_elec'][0] + count * noise_per_pixel
                valid = np.isfinite(conversion) & np.isfinite(expected_variance_e)
                np.testing.assert_allclose(changed['ferr'][0][valid] ** 2,
                    expected_variance_e[valid] * conversion[valid] ** 2, rtol=3.e-12)
                assert np.all(changed['ferr'][0][valid] > 0)
                if fraction == 0.:
                    np.testing.assert_array_equal(changed['flux_elec'][0][valid], 0.)
                    np.testing.assert_array_equal(changed['flux_ujy_model'][0][valid], 0.)
                if fraction == .4 and count == pixels:
                    # Background/dark/read variance is not reduced by source fraction.
                    assert not np.allclose(changed['ferr'][0][valid] ** 2,
                                           .4 * baseline['ferr'][0][valid] ** 2)
                # Full-response zero-order aperture photometry remains independent.
                for field in ('source_counts_e', 'observed_flux_ujy', 'variance_e2'):
                    np.testing.assert_array_equal(product['zero_order/' + band + '/' + field][:],
                                                  legacy['zero_order/' + band + '/' + field][:])


@pytest.mark.parametrize('count,invalid', [(0, ()), (2, (0, 1))])
def test_extraction_empty_and_invalid_rows_are_not_physical_zero_fraction(tmp_path, count, invalid):
    source, config = source_fixture(tmp_path, count=count, invalid=invalid)
    path = simulate(source, config, tmp_path / 'output', explicit(0., 3))
    with h5py.File(str(path), 'r') as product:
        metadata = check_metadata(product, [False] * count)
        assert metadata['source_fraction'].shape == (count,)
        assert np.isnan(metadata['source_fraction'][:]).all()


def test_paired_extraction_and_el_subset_inherit_parent_model(tmp_path, monkeypatch):
    monkeypatch.setattr(emulator, '_sample_wave_offset', staticmethod(lambda rng=None: .5))
    source, config = source_fixture(tmp_path, count=2, invalid=(0,))
    parent_path = simulate(source, config, tmp_path / 'paired', explicit(.4, 7),
                           module='cess.pipeline.paired', wave_error=True)
    with h5py.File(str(parent_path), 'r') as parent:
        metadata = check_metadata(parent, [False, True])
        assert metadata['source_fraction'][1] == .4 and metadata['spatial_pixels'][1] == 7
        for band in ('GU', 'GV', 'GI'):
            for suffix in ('', '_wce'):
                assert np.isnan(parent[band + '/covariance_lag1' + suffix][0]).all()
                assert np.isfinite(parent[band + '/ferr' + suffix][1]).any()
            covariance = parent[band + '/covariance_lag1_wce'][1]
            assert np.any(covariance[np.isfinite(covariance)] > 0)

    import cess.pipeline.emission as run_el_detect
    # A conflicting EL-only configuration must not redefine its parent's aperture.
    post_config = dict(config, spectral_extraction=explicit(.9, 23))
    child_path = run_el_detect.task(str(parent_path.parent), parent_path.name,
        output_dir=tmp_path / 'el', config=post_config, seed=1214,
        source_start=1, source_stop=2)
    with h5py.File(str(parent_path), 'r') as parent, h5py.File(str(child_path), 'r') as child:
        np.testing.assert_array_equal(child['parent_row_indices'][:], [1])
        np.testing.assert_array_equal(child['input_row_indices'][:], [1])
        metadata = check_metadata(child, [True])
        assert dict(metadata.attrs) == dict(parent['spectral_extraction'].attrs)
        assert json.loads(metadata.attrs['model_json']) == explicit(.4, 7)
        assert json.loads(child.attrs['config_json'])['spectral_extraction'] == explicit(.9, 23)
        assert child.attrs['profile_library_provenance_json'] == parent.attrs['profile_library_provenance_json']
        for name in ('source_fraction', 'spatial_pixels', 'profile_height_sum', 'valid'):
            np.testing.assert_array_equal(metadata[name][:], parent['spectral_extraction/' + name][1:2])
            assert dict(metadata[name].attrs) == dict(parent['spectral_extraction/' + name].attrs)


def test_legacy_el_parent_does_not_gain_current_config_aperture(tmp_path):
    source = tmp_path / 'old.hdf5'
    with h5py.File(str(source), 'w') as parent:
        parent['ID'] = np.empty(0, dtype='S20')
        for band, wave in (('GV', [4000., 5000., 6000.]), ('GI', [6000., 8000., 10000.])):
            group = parent.create_group(band)
            group['wave'] = wave
            for name in ('flux_ujy', 'flux_ujy_with_noise', 'snr'):
                group[name] = np.empty((0, 3))
    import cess.pipeline.emission as run_el_detect
    config = load_config(CORE / 'cess/data/emulator_parameters.json')
    config['spectral_extraction'] = explicit(.4, 7)
    child_path = run_el_detect.task(str(tmp_path), source.name, output_dir=tmp_path / 'el', config=config)
    with h5py.File(str(child_path), 'r') as child:
        assert 'spectral_extraction' not in child
        assert 'unknown' in child.attrs['spectral_extraction_json']
        assert 'unknown' in child.attrs['profile_library_provenance_json']
