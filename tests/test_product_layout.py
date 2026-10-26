"""R1 lossless layout conversion on tiny HDF5 fixtures; no science simulation."""
from pathlib import Path
import hashlib
import sys

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cess.io import products


def fixture(path, kind='standard', count=2, extension=False):
    with h5py.File(str(path), 'w') as out:
        out.attrs['schema_version'] = '2.5'
        out.attrs['cess_repair_revision'] = '20261022-C13'
        out.attrs['pipeline_stage'] = 'emission_line_postprocess' if kind == 'el' else 'simulation'
        for name in ('config_json', 'code_sha256_json', 'resource_sha256_json'):
            out.attrs[name] = '{"preserve":"original"}'
        out.attrs.create('custom_bytes', np.array([b'x', b'y'], dtype='S7'))
        out['ID'] = np.array(['id-α-{}'.format(i) for i in range(count)], dtype=h5py.string_dtype('utf-8'))
        out['input_row_indices'] = np.arange(count, dtype='int64') + 100
        if kind == 'el':
            out['parent_row_indices'] = np.arange(count, dtype='int64') + 5
            out['parameters_elg'] = np.zeros((count, 2))
        else:
            out['data_mask'] = np.ones(count, dtype='int64')
            table = out.create_dataset('parameters_desi', data=np.arange(count*2.).reshape(count, 2))
            table.attrs['columns'] = table.attrs['name'] = ['n', 'Re']
            table.attrs['axis_order'] = 'source,field'
            table.attrs['units'] = ['1', 'pixel']
        extraction = out.create_group('spectral_extraction')
        extraction.attrs['model_json'] = '{"mode":"explicit","source_fraction":0.4,"spatial_pixels":3}'
        extraction['source_fraction'] = np.full(count, .4)
        extraction['spatial_pixels'] = np.full(count, 3.)
        extraction['profile_height_sum'] = np.full(count, .5)
        extraction['valid'] = np.ones(count, dtype=bool)
        mge = out.create_group('mge_morphology')
        mge['selected_indices'] = np.zeros((count, 4), dtype='float64')
        mge['valid'] = np.ones(count, dtype=bool)
        for band in ('GU', 'GV', 'GI') if kind != 'el' else ('GV', 'GI'):
            group = out.create_group(band)
            group.attrs['custom_band_attribute'] = band
            if kind != 'el':
                group['wave'] = [4000., 4010., 4020.]
                values = np.arange(count*3., dtype='float64').reshape(count, 3)
                if count > 1:
                    values[1] = np.nan
                for key in ('flux_ujy', 'flux_ujy_with_noise', 'ferr', 'snr'):
                    data = group.create_dataset(key, data=values, chunks=True, compression='gzip')
                    data.attrs['unit'] = 'uJy' if key != 'snr' else '1'
                group['spec_mask'] = np.isfinite(values)
                group['covariance_lag1'] = np.zeros((count, 2))
                group['covariance_lag1'].attrs['unit'] = 'uJy^2'
                if kind == 'paired':
                    group['flux_ujy_with_noise_wce'] = values + .25
                    group['covariance_lag1_wce'] = np.full((count, 2), .5)
                else:
                    group['flux_elec_0th'] = values + 1.
            if band != 'GU':
                holder, prefix = (out, band.lower() + '_') if kind == 'el' else (group, '')
                rows = int(count > 0)
                holder[prefix + 'intri_el_id'] = np.arange(rows, dtype='int64')
                for key, dtype, values in (('wave', 'float64', [5007., 6563.]), ('idx', 'int64', [1, 2])):
                    data = holder.create_dataset(prefix + 'intri_el_' + key, (rows,),
                        dtype=h5py.vlen_dtype(np.dtype(dtype)))
                    if rows:
                        data[0] = np.asarray(values, dtype=dtype)
                    data.attrs['unit'] = 'Angstrom' if key == 'wave' else '1'
                group.create_group('el_diagnostics')['pairs'] = np.empty(0,
                    dtype=[('id', 'i8'), ('noisy_peak_id', 'i8')])
                lines = group.create_group('line_measurements')
                lines.attrs['unknown_parent_note'] = 'preserve'
                lines['candidates'] = np.empty(0, dtype=[('source_row', 'i8'), ('peak_id', 'i8'), ('snr', 'f8')])
        if kind != 'el':
            phot = out.create_group('photometry')
            phot.attrs['aperture_model'] = 'legacy assumption'
            phot.create_group('g')['reason'] = np.asarray(['ok'] * count, dtype=h5py.string_dtype('utf-8'))
            if kind == 'standard':
                out.create_group('zero_order').create_group('GU')['snr'] = np.ones(count)
        if extension:
            out.create_group('future')['not_lost'] = np.array([42.])


def equal_dataset(a, b):
    assert a.dtype == b.dtype and a.dtype.metadata == b.dtype.metadata and a.shape == b.shape
    if isinstance(h5py.check_dtype(vlen=a.dtype), np.dtype):
        for index in np.ndindex(a.shape):
            np.testing.assert_array_equal(a[index], b[index])
    else:
        np.testing.assert_array_equal(a[:], b[:])
    assert set(a.attrs) == set(b.attrs)
    for name in a.attrs:
        np.testing.assert_array_equal(a.attrs[name], b.attrs[name])
        assert a.attrs.get_id(name).dtype == b.attrs.get_id(name).dtype


@pytest.mark.parametrize('kind', ['standard', 'paired', 'el'])
@pytest.mark.parametrize('count', [0, 1, 3])
def test_conversion_preserves_every_legacy_dataset_attribute_and_view(tmp_path, kind, count):
    source, target = tmp_path / 'old.h5', tmp_path / 'new.h5'
    fixture(source, kind, count, extension=True)
    before = source.read_bytes()
    assert products.convert_product(source, target, chunk_rows=1) == target
    assert source.read_bytes() == before
    with h5py.File(str(source), 'r') as old, products.open_product(target) as new:
        assert new.schema_version == '3.0' and new.attrs['layout_version'] == products.LAYOUT_VERSION
        legacy = new.legacy_view()
        for attr in old.attrs:
            np.testing.assert_array_equal(legacy.attrs[attr], old.attrs[attr])
        def check(name, obj):
            if isinstance(obj, h5py.Dataset):
                equal_dataset(obj, legacy[name])
        old.visititems(check)
        assert set(new.handle) == set(products.ROOT_GROUPS)
        assert 'GU' not in new.handle and 'ID' not in new.handle
        assert new.handle['provenance/conversion'].attrs['source_sha256'] == hashlib.sha256(before).hexdigest()
        assert new.handle['provenance/conversion'].attrs['operation'] == 'legacy_file_conversion'
        assert h5py.check_string_dtype(new.catalog['ID'].dtype).encoding == 'utf-8'
        assert isinstance(legacy['GV/intri_el_wave'], h5py.Dataset)
        if kind != 'el':
            assert new.parameter_table(required=('n',)).names == ('n', 'Re')
            if count:
                np.testing.assert_array_equal(new.read_row(0)['spectra']['GV']['flux_ujy'], [0., 1., 2.])
        else:
            assert not len(new.handle['spectra']) and 'gv_intri_el_id' in legacy
            np.testing.assert_array_equal(new.catalog['parent_row_indices'][:], np.arange(count)+5)
    summary = products.read_product_summary(target)
    assert summary['rows'] == count and summary['schema_version'] == '3.0'


def test_native_move_preserves_array_object_and_never_invents_source_sha(tmp_path):
    path = tmp_path / 'native.h5'
    fixture(path)
    with h5py.File(str(path), 'r+') as handle:
        identity = products._object_id(handle['GV/flux_ujy'])
        products.structure_product(handle)
        assert products._object_id(handle['spectra/GV/flux_ujy']) == identity
        assert 'source_sha256' not in handle['provenance/conversion'].attrs
        assert handle['provenance/conversion'].attrs['operation'] == 'native_layout_adapter'
        assert products.structure_product(handle) is handle


def test_unknown_root_policy_and_conflict_do_not_publish(tmp_path):
    source, target = tmp_path / 'old.h5', tmp_path / 'new.h5'
    fixture(source, extension=True)
    with pytest.raises(ValueError, match='Unknown root'):
        products.convert_product(source, target, unknown='error')
    assert not target.exists()
    with h5py.File(str(source), 'r+') as handle:
        handle['gv_intri_el_id'] = [99]
    with pytest.raises(ValueError, match='Conflicting'):
        products.convert_product(source, target)
    assert not target.exists()


@pytest.mark.parametrize('chunk', [0, -1, True, 1.5])
def test_invalid_chunks_fail_before_output(tmp_path, chunk):
    with pytest.raises(ValueError, match='chunk_rows'):
        products.convert_product(tmp_path/'missing.h5', tmp_path/'new.h5', chunk_rows=chunk)


@pytest.mark.parametrize('link', [False, True])
def test_existing_targets_and_symlinks_never_overwritten(tmp_path, link):
    source, target = tmp_path/'old.h5', tmp_path/'new.h5'
    fixture(source)
    if link:
        target.symlink_to(tmp_path/'absent')
    else:
        target.write_bytes(b'winner')
    with pytest.raises(FileExistsError):
        products.convert_product(source, target)
    assert target.is_symlink() if link else target.read_bytes() == b'winner'


@pytest.mark.parametrize('failure', ['copy', 'drift', 'race'])
def test_failed_copy_source_drift_and_publication_race_cleanup(tmp_path, monkeypatch, failure):
    source, target = tmp_path/'old.h5', tmp_path/'new.h5'
    fixture(source)
    if failure == 'copy':
        def fail(*args):
            raise RuntimeError('copy failed')
        monkeypatch.setattr(products, '_copy_tree', fail)
    elif failure == 'drift':
        real = products._sha256
        calls = []
        def changed(path):
            calls.append(path)
            return real(path) if len(calls) == 1 else '0'*64
        monkeypatch.setattr(products, '_sha256', changed)
    else:
        def win(first, second):
            Path(second).write_bytes(b'race winner')
            raise FileExistsError('winner exists')
        monkeypatch.setattr(products.os, 'link', win)
    with pytest.raises((RuntimeError, ValueError, FileExistsError)):
        products.convert_product(source, target)
    assert not list(tmp_path.glob('*.partial')) and not list(tmp_path.glob('.*.partial'))
    assert target.read_bytes() == b'race winner' if failure == 'race' else not target.exists()


def test_summary_reads_metadata_only_and_legacy_open_does_not_modify(tmp_path, monkeypatch):
    source = tmp_path/'old.h5'
    fixture(source, 'paired')
    before = source.read_bytes()
    def forbidden(*args):
        pytest.fail('summary must not read arrays')
    monkeypatch.setattr(h5py.Dataset, '__getitem__', forbidden)
    summary = products.read_product_summary(source)
    assert summary['rows'] == 2 and summary['bands']['GV']['paired']
    assert source.read_bytes() == before


def test_soft_links_and_unknown_schema_are_explicit_errors(tmp_path):
    source, target = tmp_path/'old.h5', tmp_path/'new.h5'
    fixture(source)
    with h5py.File(str(source), 'r+') as handle:
        handle['link'] = h5py.SoftLink('/ID')
    with pytest.raises(ValueError, match='links are unsupported'):
        products.convert_product(source, target)
    with h5py.File(str(source), 'r+') as handle:
        handle.attrs['schema_version'] = 'future'
    with pytest.raises(ValueError, match='Unsupported'):
        with products.open_product(source):
            pass


def test_reference_attributes_are_rejected_instead_of_copied_as_dangling_refs(tmp_path):
    source, target = tmp_path/'old.h5', tmp_path/'new.h5'
    fixture(source)
    with h5py.File(str(source), 'r+') as handle:
        handle.attrs['reference'] = handle['ID'].ref
    with pytest.raises(ValueError, match='Reference-valued attributes'):
        products.convert_product(source, target)
    assert not target.exists()
