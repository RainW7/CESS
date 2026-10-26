"""Input corruption is isolated per source without editing the template file."""
import hashlib
import importlib

import h5py
import numpy as np
import pytest
from astropy.table import Table

from cess.io.input_quality import (REQUIRED_PARAMETERS, input_quality_flags,
                                  parameter_quality_flags)
from cess.io.products import open_product
from test_pipeline_smoke import make_input


def parameters(count):
    return Table(np.tile([120., 20., 20.5, 20.2, 20., 10., .2], (count, 1)),
                 names=REQUIRED_PARAMETERS)


@pytest.mark.parametrize('column,value,bit', [
    ('RA', 1.e267, 2), ('RA', -1.e-10, 2), ('DEC', 90.1, 4),
    ('MAG_Z', -1.e200, 8), ('MAG_Z', 101., 8), ('MAG_Z', np.nan, 1),
    ('log(M*)[0,1]_{MAL}', 1.e200, 16), ('log(M*)[0,1]_{MAL}', np.inf, 1),
    ('z_{MAL}', -2.9e-259, 32), ('z_{MAL}', 1.e-132, 32),
])
def test_parameter_corruption_is_local_and_does_not_mutate(column, value, bit):
    table = parameters(3)
    table[column][0] = value
    original = table.copy()
    np.testing.assert_array_equal(parameter_quality_flags(table), [bit, 0, 0])
    for name in table.colnames:
        np.testing.assert_array_equal(table[name], original[name])


def test_broad_limits_are_inclusive_and_not_a_population_selection():
    table = parameters(2)
    table['RA'] = [0., 360.]
    table['DEC'] = [-90., 90.]
    table['MAG_Z'] = [-100., 100.]
    table['MAG_G'] = [-999., np.nan]
    table['MAG_R'] = [np.inf, -999.]
    table['log(M*)[0,1]_{MAL}'] = [-5., 20.]
    table['z_{MAL}'] = [1.e-8, 10.]
    np.testing.assert_array_equal(parameter_quality_flags(table), [0, 0])
    assert parameter_quality_flags(parameters(0)).shape == (0,)
    del table['MAG_G']
    with pytest.raises(ValueError, match='Missing required.*MAG_G'):
        parameter_quality_flags(table)


def test_sed_and_spectral_redshift_bits_combine_without_z_equality_assumption():
    table, flux, z = parameters(5), np.ones((5, 3)), np.full(5, .5)
    flux[0, 0], flux[1, 1], flux[2, 2] = np.nan, -1., 1.e101
    z[3] = np.inf
    np.testing.assert_array_equal(input_quality_flags(table, flux, z), [64, 64, 64, 128, 0])
    with pytest.raises(ValueError, match='source-aligned'):
        input_quality_flags(table, flux[:2], z)


@pytest.mark.parametrize('mode', ['standard', 'paired'])
@pytest.mark.parametrize('all_bad', [False, True])
def test_bad_metadata_never_reaches_morphology_and_keeps_source_rows(tmp_path, mode, all_bad):
    source = tmp_path / 'template.hdf5'
    make_input(source, count=3)
    with h5py.File(source, 'r+') as handle:
        handle['parameters'][0, 0] = 1.e267
        handle['parameters'][2, 5] = 1.e200
        if all_bad:
            handle['parameters'][1, 6] = -2.9e-259
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    driver = importlib.import_module('cess.pipeline.' + mode)
    result = driver.simulate_file(str(tmp_path), source.name, output_dir=str(tmp_path / 'out'),
        options=dict(morph=True, wave_cal=True, photo=True, el_detect=True), seed=1214)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original
    with open_product(result) as product:
        catalog, view = product.catalog, product.legacy_view()
        expected = [2, 32 if all_bad else 0, 16]
        np.testing.assert_array_equal(catalog['input_quality_flags'][:], expected)
        np.testing.assert_array_equal(catalog['data_mask'][:], [2, 2 if all_bad else 1, 2])
        np.testing.assert_array_equal(catalog['input_row_indices'][:], [0, 1, 2])
        assert catalog['parameters_desi'][0, 0] == 1.e267
        assert catalog['parameters_desi'][2, 10] == 1.e200
        assert catalog['input_quality_flags'].attrs['axis_order'] == 'source'
        for band in ('GU', 'GV', 'GI'):
            bad = np.flatnonzero(np.asarray(expected) != 0)
            assert np.isnan(view[band + '/flux_ujy'][:][bad]).all()
            assert np.isnan(view[band + '/flux_ujy_with_noise'][:][bad]).all()
            assert not view[band + '/spec_mask'][:][bad].any()
            if not all_bad:
                assert np.isfinite(view[band + '/flux_ujy'][1]).all()
