"""Temporary-file checks of manifest provenance, reconciliation and locking."""
import json
import os
from pathlib import Path
import subprocess
import sys
import h5py
import numpy as np
import pytest
CORE = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(CORE))
import cess.io.manifest as manifest
from cess.config.loader import load_config
from cess.pipeline.support import (SourceSelection, record_runner_provenance, write_spectral_covariance,
                            LINE_MEASUREMENT_DTYPE, COVARIANCE_MODEL, write_el_details,
                            write_line_measurements, write_spectral_extraction, SCHEMA_VERSION)


def fixture_job(tmp_path, mode='standard', start=1, stop=3):
    source = tmp_path / 'source.h5'
    with h5py.File(str(source), 'w') as handle:
        handle['ID'] = np.array([11, 13, 17, 19], dtype=np.int64)
        handle['input_row_indices'] = np.array([50, 70, 90, 110], dtype=np.int64)
        if mode == 'el-detect':
            for band in ('GV', 'GI'):
                group = handle.create_group(band)
                group['wave'] = np.array([4000., 4100., 4200.])
                group['flux_ujy_with_noise'] = np.full((4, 3), np.nan)
                group['ferr'] = np.full((4, 3), np.nan)
    config = load_config()
    resource = tmp_path / 'resource.bin'
    resource.write_bytes(b'calibration fixture')
    for name, value in list(config.items()):
        if not name.startswith('_') and name.endswith(('_path', '_pkl')):
            config[name] = str(resource)
    config.update(_config_path=None, _config_dir=str(tmp_path))
    options = {'morph': False, 'wave_cal': False, 'photo': False,
               'el_detect': mode == 'el-detect'}
    job = dict(input_path=str(tmp_path), filename=source.name, output_dir=str(tmp_path / 'output'),
               config=config, options=options, mode=mode, seed=13,
               source_start=start, source_stop=stop)
    state = manifest.snapshot(config, options, mode, 13, [job])
    return job, state


def make_product(job):
    destination = manifest.expected_product(job)
    destination.parent.mkdir(exist_ok=True)
    with h5py.File(str(Path(job['input_path']) / job['filename']), 'r') as source:
        selected = SourceSelection(source, job['source_start'], job['source_stop'])
        n = len(selected['ID'])
        with h5py.File(str(destination), 'w') as output:
            output['ID'] = selected['ID'][:]
            output['input_row_indices'] = (selected['input_row_indices'][:] if job['mode'] == 'el-detect'
                                           else selected.source_rows)
            if job['mode'] == 'el-detect':
                output['parent_row_indices'] = selected.source_rows
                output['parameters_elg'] = np.zeros((n, 2))
                for band in ('GV', 'GI'):
                    make_el(output, band, flat=True)
                    known = ('covariance_lag1' in source[band] and
                             source[band + '/covariance_lag1'].attrs.get('covariance_model') == COVARIANCE_MODEL)
                    make_empty_lines(output, band, job, known=known)
                    if 'flux_ujy_with_noise_wce' in source[band]:
                        make_el(output, band, suffix='_wce', legacy=False)
                        known = ('covariance_lag1_wce' in source[band] and
                                 source[band + '/covariance_lag1_wce'].attrs.get('covariance_model') == COVARIANCE_MODEL)
                        make_empty_lines(output, band, job, suffix='_wce', known=known)
            else:
                output['parameters_desi'] = np.full((n, 11), np.nan)
                output['parameters_grism'] = np.full((n, 3), np.nan)
                output['data_mask'] = np.full(n, 2)
                for band in ('GU', 'GV', 'GI'):
                    group = output.create_group(band)
                    group['wave'] = np.array([4000., 4100., 4200.])
                    names = ('flux_ujy', 'flux_ujy_with_noise', 'ferr', 'snr', 'flux_elec',
                             'spec_mask', 'snr_mask', 'response_flux_ujy', 'source_electron_rate_pre_morph',
                             'flux_ujy_model', 'detector_model_mask', 'calibration_support_mask')
                    for name in names:
                        group[name] = np.full((n, 3), np.nan)
                    group['response_at_pixel_center'] = np.full(3, .3)
                    group['sky_rate_e_s_pixel'] = np.full(n, .1)
                    if job['mode'] == 'standard':
                        for name in names:
                            if name != 'flux_ujy':
                                group[name + '_0th'] = np.full((n, 3), np.nan)
                        group['response_at_pixel_center_0th'] = np.full(3, .1)
                        group['sky_rate_e_s_pixel_0th'] = np.full(n, .05)
                        output.require_group('zero_order/' + band)['valid'] = np.zeros(n, bool)
                        write_spectral_covariance(group, np.full((n, 2), np.nan), suffix='_0th')
                    else:
                        for name in ('flux_elec_wce', 'flux_ujy_wce', 'flux_ujy_with_noise_wce',
                                     'ferr_wce', 'snr_wce', 'flux_ujy_model_wce',
                                     'detector_model_mask_wce', 'calibration_support_mask_wce'):
                            group[name] = np.full((n, 3), np.nan)
                        write_spectral_covariance(group, np.full((n, 2), np.nan), suffix='_wce')
                    write_spectral_covariance(group, np.full((n, 2), np.nan))
                    if job['options']['el_detect'] and band in ('GV', 'GI'):
                        make_el(output, band)
                        make_empty_lines(output, band, job)
                        if job['mode'] == 'paired-wce':
                            make_el(output, band, suffix='_wce', legacy=False)
                            make_empty_lines(output, band, job, suffix='_wce')
                write_spectral_extraction(output, [None]*n, load_config(job['config']),
                                          fixture_profile_description(job['config']))
            record_runner_provenance(output, load_config(job['config']),
                manifest.normalise_options(job['options']), source=selected if job['mode'] == 'el-detect' else None,
                seed=job['seed'], input_sha256=manifest._sha256(source.filename), input_path=source.filename,
                source_start=job['source_start'], source_stop=job['source_stop'])
    return destination


def fixture_profile_description(config):
    # Minimal descriptor binding tiny fake resource bytes, without unpickling or
    # asserting that the fixture's observational statistics are a real library.
    return dict(schema='cess.profile_library_description',version=1,
        provenance_status='legacy_unverified_generation',
        resources={role:dict(path=str(Path(config[role+'lib_pkl']).resolve()),
            sha256=manifest._sha256(config[role+'lib_pkl']),shape=[1,1,1,1])
            for role in ('width','height')},
        generation=dict(status='unknown',psf=None,pixel_scale_arcsec_pixel=None,normalization=None),
        runtime_interpretation=dict(axis_order=['n','re_pixel','pa_deg','axis_ratio'],
            axes=dict(n=[1.],re_pixel=[4.],pa_deg=[0.],axis_ratio=[1.]),
            arcsecperpix=config['arcsecperpix'],arcsecperpix_unit='arcsec/pixel'),
        configured_generator_setting=dict(status='generator-setting-not-bound-to-existing-library'),
        observed_profile_statistics={role:dict(sum_range=[.6,1.2],
            sum_interpretation='observed stored profile sum; not a certified all-source aperture fraction')
            for role in ('width','height')})


PAIR_DTYPE = np.dtype([('id', '<i8'), ('noisy_peak_id', '<i8'), ('intrinsic_peak_id', '<i8'),
    ('noisy_wave', '<f8'), ('intrinsic_wave', '<f8'), ('delta_wave', '<f8'),
    ('noisy_idx', '<i8'), ('intrinsic_idx', '<i8')])


def make_el(output, band, flat=False, suffix='', legacy=True):
    prefix = band.lower() + '_' if flat else band + '/'
    if legacy:
        for kind in ('intri', 'detect'):
            stem = prefix + kind + '_el_'
            output[stem + 'id'] = np.empty(0, dtype=np.int64)
            output[stem + 'elnumber'] = np.empty(0, dtype=np.int64)
            for field in ('wave', 'idx') + (('snr',) if kind == 'detect' else ()):
                output.create_dataset(stem + field, (0,), dtype=h5py.vlen_dtype(
                    np.dtype('int64' if field == 'idx' else 'float64')))
    root = output.require_group(band + '/el_diagnostics' + suffix)
    root['pairs'] = np.empty(0, dtype=PAIR_DTYPE)
    for kind, number in (('candidates', 'candidate_el_number'), ('missed', 'missed_el_number'),
                          ('false_positive', 'false_el_number')):
        group = root.create_group(kind)
        group['id'] = np.empty(0, dtype=np.int64)
        group[number] = np.empty(0, dtype=np.int64)
        group.create_dataset('wave', (0,), dtype=h5py.vlen_dtype(np.dtype('float64')))
        group.create_dataset('idx', (0,), dtype=h5py.vlen_dtype(np.dtype('int64')))


def make_empty_lines(output, band, job, suffix='', known=True):
    group = output.require_group(band + '/line_measurements' + suffix)
    group.attrs['algorithm'] = 'cess-affine-sideband-window-v1'
    group.attrs['window_config_json'] = json.dumps(job['config']['line_measurement'], sort_keys=True)
    group.attrs['covariance_model'] = COVARIANCE_MODEL if known else 'unknown; metrics unavailable'
    for kind in ('candidates', 'truth'):
        data = group.create_dataset(kind, data=np.empty(0, dtype=LINE_MEASUREMENT_DTYPE))
        for name, value in dict(axis_order='peak', flux_unit='erg/s/cm^2',
                                 wavelength_unit='Angstrom', snr_unit='1').items():
            data.attrs[name] = value


def replace_dataset(parent, name, values, dtype=None):
    attributes = dict(parent[name].attrs) if name in parent else {}
    if name in parent:
        del parent[name]
    data = parent.create_dataset(name, data=values, dtype=dtype)
    for key, value in attributes.items():
        data.attrs[key] = value
    return data


def populated_product(tmp_path, mode='paired-wce', legacy_parent=False):
    """Use actual EL match/writers, retaining nonordinal original peak IDs."""
    from astropy import units as u
    from astropy.table import QTable
    from cess.engine import emulator
    from cess.spectra.integration import bin_edges
    job, _ = fixture_job(tmp_path, mode)
    job['options']['el_detect'] = True
    # This writer/peak-ID fixture deliberately supplies the same tiny arbitrary
    # 4000--4100-A grid for GV and GI. Its synthetic peak indexes are explicit;
    # do not reinterpret them as the modern physical bandpass window.
    job['config']['spectral_window'] = {'mode': 'indices'}
    for band in ('gv', 'gi'):
        job['config'][band + '_init_idx'], job['config'][band + '_end_idx'] = 0, 11
    job['config']['line_measurement'] = dict(line_width_pixels=dict(GV=3, GI=3),
        sideband_width_pixels=2, sideband_gap_pixels=0)
    wave = np.linspace(4000., 4100., 11)
    source_path = Path(job['input_path']) / job['filename']
    if mode == 'el-detect':
        with h5py.File(str(source_path), 'r+') as source:
            for band in ('GV', 'GI'):
                group = source[band]
                replace_dataset(group, 'wave', wave)
                for name in ('flux_ujy_with_noise', 'ferr'):
                    replace_dataset(group, name, np.ones((4, 11)))
                group['flux_ujy_with_noise_wce'] = np.ones((4, 11))
                group['ferr_wce'] = np.ones((4, 11))
                if not legacy_parent:
                    write_spectral_covariance(group, np.zeros((4, 10)))
                    write_spectral_covariance(group, np.zeros((4, 10)), suffix='_wce')
    state = manifest.snapshot(job['config'], job['options'], mode, job['seed'], [job])
    path = make_product(job)

    def peaks(indices, types):
        return QTable(dict(line_center=wave[indices]*u.AA, line_type=types,
                           line_center_index=np.array(indices, dtype=int)))
    empty = peaks([], np.array([], dtype='U10'))
    noisy = [peaks([0,3,5,9], ['absorption','emission','absorption','emission']), empty]
    truth = [peaks([0,3,7], ['absorption','emission','emission']), empty]
    wce = [peaks([0,4,5,10], ['absorption','emission','absorption','emission']), empty]
    engine = emulator(config=job['config'])
    with h5py.File(str(source_path), 'r') as source, h5py.File(str(path), 'r+') as output:
        selected = SourceSelection(source, job['source_start'], job['source_stop'])
        if mode != 'el-detect':
            for band in ('GU', 'GV', 'GI'):
                group = output[band]
                for name in list(group):
                    data = group[name]
                    if not isinstance(data, h5py.Dataset):
                        continue
                    if name == 'wave':
                        values = wave
                    elif name == 'wave_edges':
                        values = bin_edges(wave)
                    elif name == 'pixel_width':
                        values = np.diff(bin_edges(wave))
                    elif name.startswith('covariance_lag1'):
                        values = np.zeros((2, 10))
                    elif data.shape == (2,3):
                        values = np.ones((2, 11))
                    elif data.shape == (3,):
                        values = np.ones(11)
                    else:
                        continue
                    replace_dataset(group, name, values)
        for band in ('GV', 'GI'):
            group = output[band]
            baseline_truth, matched, details = engine.el_success(noisy, truth, 15., return_details=True)
            prefix = band.lower() + '_' if mode == 'el-detect' else band + '/'
            for kind, table in [('intri', baseline_truth), ('detect', matched)]:
                stem = prefix + kind + '_el_'
                replace_dataset(output, stem + 'id', table['id'])
                replace_dataset(output, stem + 'elnumber', table[kind + '_el_number' if kind == 'intri' else 'detect_el_number'])
                for field in ('wave', 'idx') + (('snr',) if kind == 'detect' else ()):
                    del output[stem + field]
                    data = output.create_dataset(stem + field, (len(table),),
                        dtype=h5py.vlen_dtype(np.dtype('int64' if field == 'idx' else 'float64')))
                    for row in range(len(table)):
                        data[row] = np.ones(len(table[row]['wave'])) if field == 'snr' else table[row][field]
            del group['el_diagnostics']
            del group['line_measurements']
            write_el_details(group, details)
            write_line_measurements(group, engine, noisy, truth, load_config(job['config']),
                                    source_group=selected[band] if mode == 'el-detect' else None)
            if mode in ('paired-wce', 'el-detect'):
                for name in ('el_diagnostics_wce', 'line_measurements_wce'):
                    del group[name]
                _, _, details = engine.el_success(wce, truth, 15., return_details=True)
                write_el_details(group, details, suffix='_wce')
                write_line_measurements(group, engine, wce, truth, load_config(job['config']),
                    source_group=selected[band] if mode == 'el-detect' else None, suffix='_wce')
    return job, state, path


@pytest.mark.parametrize('mode', ['standard', 'paired-wce', 'el-detect'])
@pytest.mark.parametrize('interval', [(1, 3), (0, 0), (0, 4)])
def test_verified_products_include_empty_invalid_and_el_parent_mapping(tmp_path, mode, interval):
    job, state = fixture_job(tmp_path, mode, *interval)
    assert manifest.verify_product(job, state) is None
    path = make_product(job)
    result = manifest.verify_product(job, state)
    assert result == dict(output=str(path), sha256=manifest._sha256(path))
    assert manifest.verify_product(job, state, result['sha256']) == result
    if mode == 'el-detect':
        with h5py.File(str(path), 'r') as output:
            np.testing.assert_array_equal(output['parent_row_indices'][:], np.arange(*interval))
            np.testing.assert_array_equal(output['input_row_indices'][:], np.array([50, 70, 90, 110])[slice(*interval)])


def test_expected_filename_uses_whole_and_partial_runner_rules(tmp_path):
    job, _ = fixture_job(tmp_path, 'el-detect', 0, 4)
    assert manifest.expected_product(job).name == 'elg_info_source.h5'
    job['source_start'] = 1
    assert manifest.expected_product(job).name == 'elg_info_source_rows1-4_3.hdf5'
    job['mode'] = 'standard'
    assert manifest.expected_product(job).name == 'CSST_grism_source_rows1-4_3.hdf5'


@pytest.mark.parametrize('attribute', ['input_sha256', 'input_product', 'master_seed', 'random_stream_version',
    'config_json', 'switches_json', 'code_sha256_json', 'resource_sha256_json', 'source_start',
    'schema_version', 'pipeline_stage'])
def test_provenance_mismatch_refuses_existing_product(tmp_path, attribute):
    job, state = fixture_job(tmp_path)
    path = make_product(job)
    with h5py.File(str(path), 'r+') as output:
        output.attrs[attribute] = -123 if attribute == 'source_start' else '{}'
    with pytest.raises(ValueError):
        manifest.verify_product(job, state)


@pytest.mark.parametrize('damage', ['id', 'id_dtype', 'rows', 'shape', 'missing', 'source_axis'])
def test_id_mapping_and_structure_mismatch_refused(tmp_path, damage):
    job, state = fixture_job(tmp_path)
    path = make_product(job)
    with h5py.File(str(path), 'r+') as output:
        if damage == 'id':
            output['ID'][0] = 999
        elif damage == 'id_dtype':
            values = output['ID'][:].astype(float)
            del output['ID']
            output['ID'] = values
        elif damage == 'rows':
            output['input_row_indices'][0] = 8
        elif damage == 'shape':
            del output['GU/ferr']
            output['GU/ferr'] = np.zeros((2, 2))
        elif damage == 'missing':
            del output['GV/flux_elec_0th']
        else:
            dataset = output.create_dataset('extra', data=np.zeros(3))
            dataset.attrs['axis_order'] = 'source'
    with pytest.raises(ValueError):
        manifest.verify_product(job, state)


def test_input_and_output_sha_changes_refused(tmp_path):
    job, state = fixture_job(tmp_path)
    path = make_product(job)
    with pytest.raises(ValueError, match='Output SHA256'):
        manifest.verify_product(job, state, '0' * 64)
    with h5py.File(str(Path(job['input_path']) / job['filename']), 'r+') as source:
        source['ID'][0] = 101
    with pytest.raises(ValueError, match='Input bytes'):
        manifest.verify_product(job, state)


@pytest.mark.parametrize('empty', [False, True])
def test_string_id_encoding_metadata_mismatch_rejected_even_when_empty(tmp_path, empty):
    job, _ = fixture_job(tmp_path, stop=1 if empty else 3)
    source_path = Path(job['input_path']) / job['filename']
    with h5py.File(str(source_path), 'r+') as source:
        del source['ID']
        source.create_dataset('ID', data=np.array(['a', 'b', 'c', 'd'], dtype=object),
                              dtype=h5py.string_dtype(encoding='utf-8'))
    state = manifest.snapshot(job['config'], job['options'], job['mode'], job['seed'], [job])
    path = make_product(job)
    assert manifest.verify_product(job, state) is not None
    with h5py.File(str(path), 'r+') as output:
        values = output['ID'][:]
        del output['ID']
        output.create_dataset('ID', data=values, dtype=h5py.string_dtype(encoding='ascii'))
    with pytest.raises(ValueError, match='dtype'):
        manifest.verify_product(job, state)


@pytest.mark.parametrize('mode,legacy', [('standard',False), ('paired-wce',False),
                                       ('el-detect',False), ('el-detect',True)])
def test_nonordinal_peak_ids_and_wce_truth_join_real_writer_contract(tmp_path, mode, legacy):
    job, state, path = populated_product(tmp_path, mode, legacy)
    assert manifest.verify_product(job, state) is not None
    with h5py.File(str(path), 'r') as output:
        assert output.attrs['schema_version'] == SCHEMA_VERSION
        for band in ('GV', 'GI'):
            rows = output[band + '/line_measurements/candidates'][:]
            np.testing.assert_array_equal(rows['peak_id'], [1,3])
            pairs = output[band + '/el_diagnostics/pairs'][:]
            assert pairs['noisy_peak_id'].tolist() == [1]  # row ordinal is 0
            if legacy:
                assert np.all(~rows['valid']) and np.all((rows['flag'] & 512) != 0)
            if mode in ('paired-wce','el-detect'):
                baseline = output[band + '/line_measurements/truth'][:]
                wce = output[band + '/line_measurements_wce/truth'][:]
                for field in ('source_row','peak_id','center_index','center_wave'):
                    np.testing.assert_array_equal(baseline[field], wce[field])


@pytest.mark.parametrize('damage', ['missing', 'shape', 'dtype', 'unit', 'model', 'axis',
                                   'old_schema', 'old_revision'])
def test_new_covariance_structure_and_metadata_required(tmp_path, damage):
    job, state = fixture_job(tmp_path)
    path = make_product(job)
    with h5py.File(str(path), 'r+') as output:
        if damage == 'old_schema':
            output.attrs['schema_version'] = '2.2'
        elif damage == 'old_revision':
            output.attrs['cess_repair_revision'] = '20261020-C3'
        elif damage == 'missing':
            del output['GV/covariance_lag1_0th']
        elif damage == 'shape':
            replace_dataset(output['GV'], 'covariance_lag1', np.zeros((2,3)))
        elif damage == 'dtype':
            replace_dataset(output['GV'], 'covariance_lag1', np.zeros((2,2),dtype=np.float32))
        else:
            attr = dict(unit='unit', model='covariance_model', axis='axis_order')[damage]
            output['GV/covariance_lag1'].attrs[attr] = 'wrong'
    with pytest.raises(ValueError):
        manifest.verify_product(job, state)


@pytest.mark.parametrize('damage', ['duplicate', 'negative_peak', 'bad_source', 'center',
    'wavelength', 'dtype', 'window', 'missing_wce_measurement', 'missing_wce_diagnostic',
    'pair_reuse', 'pair_peak', 'pair_idx', 'pair_wave', 'pair_structure', 'wce_truth_id'])
def test_line_tables_and_pair_join_damage_rejected(tmp_path, damage):
    job, state, path = populated_product(tmp_path)
    with h5py.File(str(path), 'r+') as output:
        group = output['GV/line_measurements']
        rows = group['candidates'][:]
        if damage == 'duplicate':
            replace_dataset(group, 'candidates', np.r_[rows,rows[:1]])
        elif damage == 'negative_peak':
            rows['peak_id'][0] = -1
            group['candidates'][:] = rows
        elif damage == 'bad_source':
            rows['source_row'][0] = 2
            group['candidates'][:] = rows
        elif damage == 'center':
            rows['center_index'][0] = 11
            group['candidates'][:] = rows
        elif damage == 'wavelength':
            rows['center_wave'][0] += .1
            group['candidates'][:] = rows
        elif damage == 'dtype':
            replace_dataset(group, 'candidates', np.empty(0,dtype=[('source_row',int)]))
        elif damage == 'window':
            rows['line_stop'][0] += 1
            group['candidates'][:] = rows
        elif damage.startswith('missing_wce'):
            del output['GV/' + ('line_measurements_wce' if damage.endswith('measurement') else 'el_diagnostics_wce')]
        elif damage == 'wce_truth_id':
            data = output['GV/line_measurements_wce/truth']
            rows = data[:]; rows['peak_id'][0] = 9; data[:] = rows
        else:
            diagnostics = output['GV/el_diagnostics']
            pairs = diagnostics['pairs'][:]
            if damage == 'pair_reuse':
                pairs = np.r_[pairs,pairs]
            elif damage == 'pair_peak':
                pairs['noisy_peak_id'][0] = 9
            elif damage == 'pair_idx':
                pairs['noisy_idx'][0] += 1
            elif damage == 'pair_wave':
                pairs['noisy_wave'][0] += .1
            elif damage == 'pair_structure':
                pairs = np.empty(0,dtype=[('id',int)])
            replace_dataset(diagnostics,'pairs',pairs)
    with pytest.raises(ValueError):
        manifest.verify_product(job, state)


@pytest.mark.parametrize('field', ['wave','idx','elnumber','snr','diagnostic_count'])
def test_ragged_parallel_lengths_are_checked_per_source(tmp_path, field):
    job, state, path = populated_product(tmp_path)
    with h5py.File(str(path), 'r+') as output:
        if field == 'diagnostic_count':
            output['GV/el_diagnostics/candidates/candidate_el_number'][0] += 1
        elif field == 'elnumber':
            output['GV/detect_el_elnumber'][0] += 1
        else:
            output['GV/detect_el_' + field][0] = np.array([],dtype=int if field == 'idx' else float)
    with pytest.raises(ValueError, match='length mismatch'):
        manifest.verify_product(job, state)


def test_legacy_parent_unknown_covariance_cannot_publish_valid_metrics(tmp_path):
    job, state, path = populated_product(tmp_path, 'el-detect', True)
    with h5py.File(str(path), 'r+') as output:
        data = output['GV/line_measurements/candidates']
        rows = data[:]; rows['valid'][0] = True; data[:] = rows
    with pytest.raises(ValueError, match='Unavailable'):
        manifest.verify_product(job, state)


def extraction_product(tmp_path, fraction=.4, legacy=False):
    job, _ = fixture_job(tmp_path)
    job['config']['spectral_extraction'] = ({'mode':'legacy_full'} if legacy else
        dict(mode='explicit',source_fraction=fraction,spatial_pixels=5))
    state = manifest.snapshot(job['config'],job['options'],job['mode'],job['seed'],[job])
    path = make_product(job)
    with h5py.File(str(path),'r+') as output:
        output['data_mask'][0] = 1
        extraction = output['spectral_extraction']
        extraction['valid'][0] = True
        extraction['source_fraction'][0] = 1. if legacy else fraction
        extraction['spatial_pixels'][0] = 9. if legacy else 5.
        extraction['profile_height_sum'][0] = 1.2  # deliberately not an aperture fraction
    return job,state,path


@pytest.mark.parametrize('fraction,legacy', [(0.,False),(.4,False),(1.,False),(1.,True)])
def test_c6_extraction_modes_allow_observed_height_sums_above_one(tmp_path,fraction,legacy):
    job,state,_ = extraction_product(tmp_path,fraction,legacy)
    assert manifest.verify_product(job,state) is not None


@pytest.mark.parametrize('damage', ['missing_group','missing_provenance','model','fraction','pixels',
    'height_sum','invalid_row','valid','unit','resource_sha','generation_psf','generation_normalization',
    'runtime_scale','axes_shape','axes_nonfinite'])
def test_c6_extraction_and_profile_binding_damage_rejected(tmp_path,damage):
    job,state,path = extraction_product(tmp_path)
    with h5py.File(str(path),'r+') as output:
        group = output['spectral_extraction']
        if damage == 'missing_group':
            del output['spectral_extraction']
        elif damage == 'missing_provenance':
            del output.attrs['profile_library_provenance_json']
        elif damage == 'model':
            group.attrs['model_json'] = json.dumps({'mode':'legacy_full'})
        elif damage == 'fraction':
            group['source_fraction'][0] = .5
        elif damage == 'pixels':
            group['spatial_pixels'][0] = 5.5
        elif damage == 'height_sum':
            group['profile_height_sum'][0] = 0.
        elif damage == 'invalid_row':
            group['profile_height_sum'][1] = .9
        elif damage == 'valid':
            group['valid'][0] = False
        elif damage == 'unit':
            group['source_fraction'].attrs['unit'] = 'pixel'
        else:
            description = json.loads(output.attrs['profile_library_provenance_json'])
            if damage == 'resource_sha':
                description['resources']['width']['sha256'] = '0'*64
            elif damage == 'generation_psf':
                description['generation']['psf'] = {'fwhm_arcsec':.2625}
            elif damage == 'generation_normalization':
                description['generation']['normalization'] = 'full_source'
            elif damage == 'runtime_scale':
                description['runtime_interpretation']['arcsecperpix'] *= 2.
            elif damage == 'axes_shape':
                description['resources']['height']['shape'] = [2,1,1,1]
            elif damage == 'axes_nonfinite':
                description['runtime_interpretation']['axes']['n'] = [float('nan')]
            output.attrs['profile_library_provenance_json'] = json.dumps(description)
    with pytest.raises(ValueError):
        manifest.verify_product(job,state)


def extraction_parent_product(tmp_path):
    job,_ = fixture_job(tmp_path,'el-detect')
    source_path = Path(job['input_path'])/job['filename']
    parent_config = load_config(job['config'])
    parent_config['spectral_extraction'] = dict(mode='explicit',source_fraction=.4,spatial_pixels=5)
    parent_config['arcsecperpix'] = .09  # inherited generation, not EL runtime configuration
    row = dict(source_fraction=.4,spatial_pixels=5,profile_height_sum=1.2)
    with h5py.File(str(source_path),'r+') as source:
        write_spectral_extraction(source,[row,None,row,row],parent_config,
                                  fixture_profile_description(parent_config))
        source['spectral_extraction'].attrs['extra_parent_annotation'] = 'inherited, not recomputed'
    state = manifest.snapshot(job['config'],job['options'],job['mode'],job['seed'],[job])
    return job,state,make_product(job)


def test_c6_el_inherits_selected_parent_values_nan_and_foreign_configuration(tmp_path):
    job,state,path = extraction_parent_product(tmp_path)
    assert manifest.verify_product(job,state) is not None
    with h5py.File(str(path),'r') as output:
        group = output['spectral_extraction']
        np.testing.assert_array_equal(group['valid'][:],[False,True])
        assert np.isnan(group['source_fraction'][0]) and group['source_fraction'][1] == .4
        assert json.loads(group.attrs['model_json'])['mode'] == 'explicit'
        assert job['config']['spectral_extraction']['mode'] == 'legacy_full'
        description = json.loads(output.attrs['profile_library_provenance_json'])
        assert description['runtime_interpretation']['arcsecperpix'] == .09


@pytest.mark.parametrize('damage',['value','group_attribute','dataset_attribute','profile','missing_field'])
def test_c6_el_extraction_inheritance_damage_refused(tmp_path,damage):
    job,state,path = extraction_parent_product(tmp_path)
    with h5py.File(str(path),'r+') as output:
        group = output['spectral_extraction']
        if damage == 'value':
            group['source_fraction'][0] = .4
        elif damage == 'group_attribute':
            group.attrs['model_json'] = json.dumps(job['config']['spectral_extraction'])
        elif damage == 'dataset_attribute':
            group['spatial_pixels'].attrs['unit'] = 'arcsec'
        elif damage == 'profile':
            output.attrs['profile_library_provenance_json'] = json.dumps({'provenance_status':'unknown'})
        else:
            del group['profile_height_sum']
    with pytest.raises(ValueError):
        manifest.verify_product(job,state)


def test_c6_el_cannot_invent_extraction_absent_from_old_parent(tmp_path):
    job,state = fixture_job(tmp_path,'el-detect')
    path = make_product(job)
    with h5py.File(str(path),'r+') as output:
        output.create_group('spectral_extraction')
    with pytest.raises(ValueError,match='invented'):
        manifest.verify_product(job,state)


def test_snapshot_hashes_match_runner_and_normalize_aliases(tmp_path):
    job, state = fixture_job(tmp_path)
    path = make_product(job)
    with h5py.File(str(path), 'r') as product:
        assert state['code_sha256'] == json.loads(product.attrs['code_sha256_json'])
        assert state['resource_sha256'] == json.loads(product.attrs['resource_sha256_json'])
    assert 'apply_morphology_effect' in state['options'] and 'morph' not in state['options']
    assert str(Path(manifest.__file__).resolve()) in state['code_sha256']


def test_atomic_initial_write_no_clobber_replace_and_failed_serialization(tmp_path, monkeypatch):
    path = tmp_path / 'manifest.json'
    manifest.write_manifest(path, dict(manifest_version=1, value='old'))
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        manifest.write_manifest(path, dict(value='duplicate'))
    assert path.read_bytes() == original
    with pytest.raises(TypeError):
        manifest.write_manifest(path, dict(value=object()), replace=True)
    assert path.read_bytes() == original
    with monkeypatch.context() as patch:
        patch.setattr(manifest.os, 'fsync', lambda descriptor: (_ for _ in ()).throw(OSError('disk failure')))
        with pytest.raises(OSError, match='disk failure'):
            manifest.write_manifest(path, dict(value='unpublished'), replace=True)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob('*.tmp'))
    manifest.write_manifest(path, dict(value='new'), replace=True)
    assert json.loads(path.read_text()) == dict(value='new')


def test_lock_nonblocking_competition_and_release_after_abrupt_exit(tmp_path):
    path = tmp_path / 'manifest.json'
    environment = dict(os.environ, PYTHONPATH=str(CORE), PYTHONDONTWRITEBYTECODE='1')
    code = ('from cess.io.manifest import manifest_lock; import os; '
            'lock=manifest_lock({!r}); lock.__enter__(); os._exit(0)').format(str(path))
    with manifest.manifest_lock(path) as lock_path:
        with pytest.raises(ValueError, match='locked'):
            with manifest.manifest_lock(path):
                pass
        contender = subprocess.run([sys.executable, '-B', '-c', code], env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
        assert contender.returncode != 0 and 'locked' in contender.stderr
        assert lock_path.exists()
    abandoned = subprocess.run([sys.executable, '-B', '-c', code], env=environment, timeout=30)
    assert abandoned.returncode == 0
    with manifest.manifest_lock(path):
        pass
    assert Path(str(path) + '.lock').exists()
