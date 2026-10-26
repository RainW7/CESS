"""Small regression fixtures for parameter layouts, extraction and morphology.

Run with the project's grizli Python. Existing catalogues are only read for
their small parameter tables; all generated products live under tmp_path.
"""

import importlib
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / 'CESS_v0.8.6'
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(ROOT))
from cess_io import read_parameter_table
import extract_mock_spectra_ds as extract


@pytest.fixture(scope='module')
def morphology():
    previous = os.getcwd()
    try:
        os.chdir(str(CORE))
        module = importlib.import_module('morphology')
    finally:
        os.chdir(previous)
    return module


def test_extraction_returns_the_tested_central_aperture(morphology):
    profile = np.asarray([.01, .04, .2, .5, .2, .04, .01])
    aperture = morphology.extraction_height(profile, .65)
    np.testing.assert_array_equal(aperture, profile[2:5])
    assert aperture.sum() / profile.sum() >= .65
    # The smaller aperture does not contain the requested fraction.
    assert profile[3] / profile.sum() < .65
    np.testing.assert_array_equal(morphology.extraction_height(profile, 1.), profile)
    np.testing.assert_array_equal(morphology.extraction_height([1.], .8), [1.])
    np.testing.assert_array_equal(morphology.extraction_height([.4, .6], .8), [.4, .6])


@pytest.mark.parametrize('profile,ratio', [([], .8), ([0., 0.], .8),
                                         ([1., np.nan], .8), ([-1., 2.], .8),
                                         ([1.], 0.), ([1.], 1.1)])
def test_extraction_rejects_invalid_profiles(morphology, profile, ratio):
    with pytest.raises(ValueError):
        morphology.extraction_height(profile, ratio)


def _legacy_draw(morphology, mag, magmin=14., magmax=21., delta=.5, nmax=5.):
    if mag < round(magmin):
        mag = round(magmin)
    bins = np.arange(round(magmin), round(magmax) + delta, delta)
    indices = np.digitize(morphology.gemsmag, bins)
    samples = [morphology.gemsn[indices == index]
               for index in range(1, len(bins) + 1)]
    values = samples[np.abs(bins - mag).argmin()]
    if len(values):
        unique, counts = np.unique(values, return_counts=True)
        n = np.random.choice(unique, p=counts / len(values))
    else:
        n = np.random.uniform(nmax)
    if n > nmax:
        n = np.random.uniform(nmax)
    unique, counts = np.unique(morphology.gemsar, return_counts=True)
    ar = np.random.choice(unique, p=counts / len(morphology.gemsar))
    return n, ar


def test_cached_morphology_retains_seeded_draws_and_rng_state(morphology):
    magnitudes = [10., 14., 14.25, 17., 20.9, 21., 99., np.nan] * 4
    np.random.seed(20261019)
    expected = np.asarray([_legacy_draw(morphology, mag) for mag in magnitudes])
    expected_next = np.random.random()
    morphology._n_distributions.cache_clear()
    np.random.seed(20261019)
    actual = np.asarray([(morphology.get_n_from_mag(mag, 14., 21., .5, 5.),
                          morphology.get_ar()) for mag in magnitudes])
    np.testing.assert_array_equal(actual, expected)
    assert np.random.random() == expected_next
    assert morphology._n_distributions.cache_info().hits >= len(magnitudes) - 1
    assert morphology.gems_cat._file.closed


@pytest.mark.parametrize('layout', ['dense', 'legacy', 'structured'])
def test_named_parameter_layouts(tmp_path, layout):
    matrix = np.arange(12.).reshape(4, 3)
    names = ['snr', 'mass', 'z']
    with h5py.File(str(tmp_path / 'table.h5'), 'w') as file:
        file['ID'] = np.arange(4)
        if layout == 'structured':
            data = np.empty(4, dtype=[(name, 'f8') for name in names])
            for index, name in enumerate(names):
                data[name] = matrix[:, index]
        else:
            data = matrix if layout == 'dense' else matrix.T[:, :, None]
        table = file.create_dataset('parameters', data=data)
        table.attrs['name'] = names
        result = read_parameter_table(table, required=('snr', 'z'))
        np.testing.assert_array_equal(result.values, matrix)
        np.testing.assert_array_equal(result['z'], matrix[:, 2])


def test_reader_rejects_ambiguous_or_mislabeled_tables(tmp_path):
    with h5py.File(str(tmp_path / 'invalid.h5'), 'w') as file:
        file['ID'] = np.arange(2)
        dataset = file.create_dataset('parameters', data=[[1., 2.], [3., 4.]])
        dataset.attrs['name'] = ['a', 'b']
        with pytest.raises(ValueError, match='ambiguous'):
            read_parameter_table(dataset)
        dataset.attrs['axis_order'] = 'source,field'
        np.testing.assert_array_equal(read_parameter_table(dataset)['b'], [2., 4.])
        with pytest.raises(ValueError, match='ID length'):
            read_parameter_table(dataset, ids=[1], n_sources=2)
        with pytest.raises(ValueError, match='required'):
            read_parameter_table(dataset, required=('missing',))
        dataset.attrs['name'] = ['a', 'a']
        with pytest.raises(ValueError, match='unique'):
            read_parameter_table(dataset)
    with pytest.raises(ValueError, match='named columns'):
        read_parameter_table(np.ones((4, 2)), n_sources=4)
    with pytest.raises(ValueError, match='shape'):
        read_parameter_table(np.ones((3, 4)), n_sources=4, names=['a', 'b', 'c'])
    structured = np.ones(4, dtype=[('a', float), ('b', float)])
    with pytest.raises(ValueError, match='dtype fields'):
        read_parameter_table(structured, n_sources=4, names=['b', 'a'])


def _catalogue(path, structured=False, extras=True):
    names = ['dummy_a', 'dummy_b', 'dummy_c', 'dummy_d', 'dummy_e', 'dummy_f',
             'gu_snr_mean', 'gv_snr_mean', 'gi_snr_mean']
    params = np.zeros((4, len(names)))
    params[:, 6:] = [[4., 5., 6.], [5., 1., 6.], [6., 7., 8.], [3., 4., 1.]]
    with h5py.File(str(path), 'w') as file:
        file['ID'] = np.asarray([b'a', b'b', b'c', b'd'])
        if structured:
            grism = np.empty(4, dtype=[(name, 'f8') for name in names])
            for index, name in enumerate(names):
                grism[name] = params[:, index]
        else:
            grism = params
        dataset = file.create_dataset('parameters_grism', data=grism)
        dataset.attrs['name'] = names
        desi = file.create_dataset('parameters_desi', data=np.column_stack([
            np.full(4, 10.), [.1, .2, .3, .4]]))
        desi.attrs['name'] = ['str_mass', 'z_best']
        file['data_mask'] = [1, 1, 1, 1]
        for band in ('GU', 'GV', 'GI'):
            group = file.create_group(band)
            group['wave'] = [5000., 5010., 5020.]
            group['flux_ujy'] = np.arange(12.).reshape(4, 3)
            group['flux_ujy_with_noise'] = np.arange(12.).reshape(4, 3) + .5
            if extras:
                group['ferr'] = np.full((4, 3), .2)
                group['spec_mask'] = np.tile([False, True, True], (4, 1))
                group['snr_mask'] = np.tile([False, False, True], (4, 1))


def test_extraction_uses_named_snr_and_preserves_errors_masks_provenance(tmp_path):
    first, second = tmp_path / 'dense.h5', tmp_path / 'structured.h5'
    _catalogue(first)
    _catalogue(second, structured=True, extras=False)
    sources, gu, gv, gi = extract.read_hdf5_files([first, second])
    assert [source['id'] for source in sources] == [b'a', b'c', b'a', b'c']
    assert [source['source_row'] for source in sources] == [0, 2, 0, 2]
    np.testing.assert_array_equal([source['gv_snr'] for source in sources], [5., 7., 5., 7.])
    product = tmp_path / 'selected.h5'
    extract.write_output(product, sources, gu, gv, gi, seed=17,
                         input_files=[first, second], target=4, n_bins=2)
    with h5py.File(str(product), 'r') as file:
        assert file.attrs['sampling_seed'] == 17
        assert file['GU/flux_ujy'].shape == (4, 3)
        np.testing.assert_array_equal(file['GU/ferr_available'], [True, True, False, False])
        np.testing.assert_allclose(file['GU/ferr'][:2], .2)
        assert np.isnan(file['GU/ferr'][2:]).all()
        np.testing.assert_array_equal(file['GU/spec_mask'][0], [False, True, True])
        np.testing.assert_array_equal(file['provenance/source_row'], [0, 2, 0, 2])
        assert file['provenance/source_file'][0].decode() == str(first.resolve())
        np.testing.assert_allclose(read_parameter_table(file['parameters_desi'])['z_best'],
                                   [.1, .3, .1, .3])
        np.testing.assert_allclose(read_parameter_table(file['parameters_grism'])['gi_snr_mean'],
                                   [6., 8., 6., 8.])


@pytest.mark.parametrize('redshifts', [np.ones(200), np.repeat([.1, .2], [199, 1]),
                                      np.linspace(0., 2., 200),
                                      np.r_[np.linspace(0., 2., 180), [np.nan] * 20]])
def test_stratified_sampling_exact_count_seed_and_no_replacement(redshifts):
    sources = [{'id': index, 'z_best': redshift} for index, redshift in enumerate(redshifts)]
    np.random.seed(71)
    expected_next = np.random.RandomState(71).random_sample()
    selected = extract.stratified_sampling(sources, target=57, n_bins=10, seed=9)
    repeated = extract.stratified_sampling(sources, target=57, n_bins=10, seed=9)
    assert np.random.random() == expected_next
    assert len(selected) == 57
    assert len(set(source['id'] for source in selected)) == 57
    assert [source['id'] for source in selected] == [source['id'] for source in repeated]
    assert extract.stratified_sampling(sources, target=0) == []


def test_real_catalogue_parameter_metadata():
    candidates = sorted((CORE / 'CESS_output').glob('*.hdf5'))
    if not candidates:
        pytest.skip('local real catalogue is unavailable')
    with h5py.File(str(candidates[0]), 'r') as file:
        for key in ('parameters_desi', 'parameters_grism', 'zero_order_phot'):
            if key in file:
                table = read_parameter_table(file[key])
                assert table.values.shape == (len(file['ID']), len(table.names))
        grism = read_parameter_table(file['parameters_grism'],
                                    required=('gv_snr_mean', 'gi_snr_mean'))
        assert len(grism['gv_snr_mean']) == len(file['ID'])
