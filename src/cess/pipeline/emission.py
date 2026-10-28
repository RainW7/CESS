"""Emission-line measurements from an existing CESS spectral product.

Preserve the parent source mapping, uncertainty and covariance definitions.
Use simulate_file() or cess el-detect. Original author: Run Wen.
"""

import sys
from pathlib import Path
import numpy as np
from cess.instrument.spectral import window_indices
import h5py
from astropy.table import Table
from collections.abc import Mapping
from scipy import interpolate
import time
import json
from cess.config.loader import load_config, validate_config
from cess.pipeline.random import source_rng, legacy_rng, file_identity
from cess.pipeline.support import (throughput_interpolator, combine_band_masks,
    pad_spectral_rows, output_destination, atomic_hdf5, write_vlen_rows,
    write_el_details, write_line_measurements, record_runner_provenance, normalise_options, cached_table,
    cached_pickle, configure_morphology, SourceSelection, simulation_filename,
    )

from cess.pipeline.common import validate_output_layout, finalize_product

_CONFIGURATION = None
_OPTIONS = None
_PARAMETER_NAMES = ('input_path', 'output_path', 'el_input_path', 'el_output_path', 'bkg_spec_path', 'gems_path', 'widthlib_pkl', 'heightlib_pkl', 'gutp_path', 'gvtp_path', 'gitp_path', 'gutp0_path', 'gvtp0_path', 'gitp0_path', 'nuvtp_path', 'utp_path', 'gtp_path', 'rtp_path', 'itp_path', 'ztp_path', 'ytp_path', 'radi', 'expt', 'expnum', 'arcsecperpix', 'dc_value', 'gu_Res', 'gu_dlambda', 'gu_ree80', 'gv_Res', 'gv_dlambda', 'gv_ree80', 'gi_Res', 'gi_dlambda', 'gi_ree80', 'gu_start_wl', 'gu_end_wl', 'gu_min_wl', 'gu_max_wl', 'gv_start_wl', 'gv_end_wl', 'gv_min_wl', 'gv_max_wl', 'gi_start_wl', 'gi_end_wl', 'gi_min_wl', 'gi_max_wl', 'gu_init_idx', 'gu_end_idx', 'gv_init_idx', 'gv_end_idx', 'gi_init_idx', 'gi_end_idx', 'noisy_el_detect_th', 'intri_el_detect_th', 'el_detect_success_th', 'gv_elwidth', 'gi_elwidth')


def configure(config=None, options=None):
    """Configure scalar parameters and switches; instrument resources stay lazy.

    config is a JSON path or mapping. With neither a supplied nor previous
    configuration, load_config selects CESS_CONFIG or the bundled default.
    options accepts morph/wave_cal/photo/el_detect and canonical apply_* names.
    Explicit mappings reset unspecified switches to False; None retains the
    previous configure() choices for the legacy configure-then-task workflow.
    """
    global _CONFIGURATION, _OPTIONS, emulator_parameters
    if config is None:
        parameters = load_config() if _CONFIGURATION is None else dict(_CONFIGURATION)
    elif isinstance(config, Mapping):
        parameters = validate_config(config, base_dir=config.get('_config_dir'))
    else:
        parameters = load_config(config)
    _OPTIONS = normalise_options(options, _OPTIONS if options is None else None)
    _CONFIGURATION = dict(parameters)
    emulator_parameters = dict(parameters)
    for name in _PARAMETER_NAMES:
        if name in parameters:
            globals()[name] = parameters[name]
    globals().update(_OPTIONS)
    globals()['colarea'] = np.pi * parameters['radi'] ** 2
    globals()['delta_p_sys'] = parameters.get('photometry_systematic_mag', 0.02)
    globals()['nseries'] = np.array([0.5,0.6,0.7,0.8,0.9,1.,1.1,1.2,1.3,1.4,
        1.5,1.6,1.8,2.,2.5,3.,3.5,4.,4.5,5.])
    globals()['reseries'] = np.round(np.array([.3,.5,.7,.9,1.,1.2,1.4,1.6,1.8,
        2.,2.5,3.,3.5,4.5,5.,5.5,6.,6.5,7.,7.4]) / parameters['arcsecperpix'])
    globals()['paseries'] = np.arange(0., 91., 10.)
    globals()['baseries'] = np.arange(1., 11.) / 10.
    return dict(parameters)

def task(input_path, hdf5filename, output_dir=None, config=None, options=None,
         seed=None, source_start=0, source_stop=None):
    """Compatibility API: preserve configure-then-task semantics explicitly."""
    from cess.pipeline.context import SimulationContext, _OPERATION_NAMES
    from cess.morphology import empirical as morphology
    settings = configure(config, options)
    operations = {name: globals()[name] for name in _OPERATION_NAMES if name in globals()}
    operations.update(legacy_rng=legacy_rng, configure_morphology=configure_morphology)
    context = SimulationContext(settings, _OPTIONS, operations, morphology)
    return simulate_file(input_path, hdf5filename, output_dir=output_dir, seed=seed,
                         source_start=source_start, source_stop=source_stop, context=context,
                         output_layout='legacy')


def simulate_file(input_path, hdf5filename, output_dir=None, config=None, options=None,
                  seed=None, source_start=0, source_stop=None, *, context=None,
                  output_layout='structured'):
    """Measure lines from a legacy or structured parent; publish a new product."""
    validate_output_layout(output_layout)
    from cess.pipeline.context import SimulationContext
    from cess.io.products import open_product
    context = SimulationContext(config, options) if context is None else context
    from cess.engine import emulator
    # Initialize emulator
    parameters = context.config
    emulator = emulator(SED_file_path=str(input_path), file_name=hdf5filename, config=parameters)
    input_file = Path(input_path) / hdf5filename
    input_identity = file_identity(input_file)
    
    tic = time.perf_counter()
    
    # Load CESS output data
    with open_product(input_file) as product:
        f = SourceSelection(product.legacy_view(), source_start, source_stop)
        source_rows = f.source_rows
        catalog_rows = (np.asarray(f['input_row_indices'][:], dtype=np.int64)
                        if 'input_row_indices' in f else source_rows)
        destination = output_destination(
            context.config['el_output_path'] if output_dir is None else output_dir,
            ('elg_info_' + Path(hdf5filename).name if
             len(f['ID']) == f.total_sources else
             simulation_filename(hdf5filename, f, prefix='elg_info')))
        print('Start detecting emission lines of {0}:\n'.format(hdf5filename))

        # Initialize lists
        gv_noisy_lines = []
        gv_intrinsic_lines = []
        gi_noisy_lines = []
        gi_intrinsic_lines = []

        # Extract data
        gvwave = f['GV']['wave'][:]
        giwave = f['GI']['wave'][:]
        gv_initial, gv_end = window_indices(context.config, 'GV', gvwave)
        gi_initial, gi_end = window_indices(context.config, 'GI', giwave)
        gv_ujy = f['GV']['flux_ujy'][:]
        gi_ujy = f['GI']['flux_ujy'][:]
        gv_ujy_with_noise = f['GV']['flux_ujy_with_noise'][:]
        gi_ujy_with_noise = f['GI']['flux_ujy_with_noise'][:]
        gv_snr = f['GV']['snr'][:]
        gi_snr = f['GI']['snr'][:]

        # Paired parent products carry an additional independently measured WCE variant.
        wce_noisy_lines = {}
        for band, wave, start, stop in (('GV', gvwave, gv_initial, gv_end),
                                        ('GI', giwave, gi_initial, gi_end)):
            if 'flux_ujy_with_noise_wce' in f[band]:
                wce_noisy_lines[band] = [emulator.el_detect(
                    wave, spectrum, start, stop, context.config['noisy_el_detect_th'])
                    for spectrum in f[band]['flux_ujy_with_noise_wce'][:]]

        # initialize the Qtable list
        gv_noisy_lines = []
        gv_intrinsic_lines = []
        gi_noisy_lines = []
        gi_intrinsic_lines = []
        # detect emission lines in both intrinsic and noisy spectra in GV and GI band
        for i in (range(len(f['ID'][:]))):
            gv_noisy_lines.append(emulator.el_detect(gvwave,gv_ujy_with_noise[i],gv_initial,gv_end,context.config['noisy_el_detect_th']))
            gv_intrinsic_lines.append(emulator.el_detect(gvwave,gv_ujy[i],gv_initial,gv_end,context.config['intri_el_detect_th']))

            gi_noisy_lines.append(emulator.el_detect(giwave,gi_ujy_with_noise[i],gi_initial,gi_end,context.config['noisy_el_detect_th']))
            gi_intrinsic_lines.append(emulator.el_detect(giwave,gi_ujy[i],gi_initial,gi_end,context.config['intri_el_detect_th']))
        # get the total emission line information array of intrinsic and noisy spectra
        gv_intri_el, gv_detect_el, gv_el_details = emulator.el_success(gv_noisy_lines,gv_intrinsic_lines,context.config['el_detect_success_th'], return_details=True)
        gi_intri_el, gi_detect_el, gi_el_details = emulator.el_success(gi_noisy_lines,gi_intrinsic_lines,context.config['el_detect_success_th'], return_details=True)

        # Initialize flags
        gv_el_flag = np.zeros(len(gv_noisy_lines))
        gi_el_flag = np.zeros(len(gi_noisy_lines))

        # Set flags for gv
        for i in range(len(f['ID'][:])):
            if i in gv_intri_el['id']:
                gv_el_flag[i] = 1
                if i in gv_detect_el['id']:
                    gv_el_flag[i] = 2
            elif i not in gv_intri_el['id']:
                gv_el_flag[i] = 0

        # Set flags for gi
        for i in range(len(f['ID'][:])):
            if i in gi_intri_el['id']:
                gi_el_flag[i] = 1
                if i in gi_detect_el['id']:
                    gi_el_flag[i] = 2
            elif i not in gi_intri_el['id']:
                gi_el_flag[i] = 0

        gv_el_flag = np.array(gv_el_flag)
        gi_el_flag = np.array(gi_el_flag)

        # Get SNR for emission lines
        gv_el_snr = emulator.get_el_snr(gv_snr, gv_el_flag, gv_detect_el, gv_initial, context.config['gv_elwidth'])
        gi_el_snr = emulator.get_el_snr(gi_snr, gi_el_flag, gi_detect_el, gi_initial, context.config['gi_elwidth'])

        # Save results to HDF5 file
        parameters_elg = np.hstack([gv_el_flag.reshape((len(f['ID'][:]), 1)), gi_el_flag.reshape((len(f['ID'][:]), 1))])

        with atomic_hdf5(destination) as file:
            dataset1 = file.create_dataset('ID', data = f['ID'][:], dtype=f['ID'].dtype)
            dataset2 = file.create_dataset('parameters_elg', data = parameters_elg)
            dataset2.attrs['name'] = ['gv_el_flag','gi_el_flag']
            
            dt1 = h5py.special_dtype(vlen=np.dtype('float64'))
            dt2 = h5py.special_dtype(vlen=np.dtype('int64'))
            # store intrinsic emission line results for gv
            dataset27 = file.create_dataset('gv_intri_el_id', data = gv_intri_el['id'])
            gv_intri_wave_dset = file.create_dataset('gv_intri_el_wave', (len(gv_intri_el),), dtype=dt1)
            write_vlen_rows(gv_intri_wave_dset, gv_intri_el['wave'])
            gv_intri_wave_dset.attrs['unit'] = 'Angstrom'
            gv_intri_idx_dset = file.create_dataset('gv_intri_el_idx', (len(gv_intri_el),), dtype=dt2)
            write_vlen_rows(gv_intri_idx_dset, gv_intri_el['idx'])
            dataset28 = file.create_dataset('gv_intri_el_elnumber', data = gv_intri_el['intri_el_number'])
            
            # store intrinsic emission line results for gi
            dataset29 = file.create_dataset('gi_intri_el_id', data = gi_intri_el['id'])
            gi_intri_wave_dset = file.create_dataset('gi_intri_el_wave', (len(gi_intri_el),), dtype=dt1)
            write_vlen_rows(gi_intri_wave_dset, gi_intri_el['wave'])
            gi_intri_wave_dset.attrs['unit'] = 'Angstrom'
            gi_intri_idx_dset = file.create_dataset('gi_intri_el_idx', (len(gi_intri_el),), dtype=dt2)
            write_vlen_rows(gi_intri_idx_dset, gi_intri_el['idx'])
            dataset30 = file.create_dataset('gi_intri_el_elnumber', data = gi_intri_el['intri_el_number'])
            
            # store detected emission line results for gv
            dataset31 = file.create_dataset('gv_detect_el_id', data = gv_detect_el['id'])
            gv_detect_wave_dset = file.create_dataset('gv_detect_el_wave', (len(gv_detect_el),), dtype=dt1)
            write_vlen_rows(gv_detect_wave_dset, gv_detect_el['wave'])
            gv_detect_wave_dset.attrs['unit'] = 'Angstrom'
            gv_detect_idx_dset = file.create_dataset('gv_detect_el_idx', (len(gv_detect_el),), dtype=dt2)
            write_vlen_rows(gv_detect_idx_dset, gv_detect_el['idx'])
            gv_el_snr_dset = file.create_dataset('gv_detect_el_snr',  (len(gv_detect_el),), dtype=dt1)
            write_vlen_rows(gv_el_snr_dset, gv_el_snr)
            gv_el_snr_dset.attrs['unit'] = '1'
            gv_el_snr_dset.attrs['definition'] = 'local mean per-pixel S/N around matched candidate; not integrated line S/N'
            dataset32 = file.create_dataset('gv_detect_el_elnumber', data = gv_detect_el['detect_el_number'])
            
            # store detected emission line results for gi
            dataset33 = file.create_dataset('gi_detect_el_id', data = gi_detect_el['id'])
            gi_detect_wave_dset = file.create_dataset('gi_detect_el_wave', (len(gi_detect_el),), dtype=dt1)
            write_vlen_rows(gi_detect_wave_dset, gi_detect_el['wave'])
            gi_detect_wave_dset.attrs['unit'] = 'Angstrom'
            gi_detect_idx_dset = file.create_dataset('gi_detect_el_idx', (len(gi_detect_el),), dtype=dt2)
            write_vlen_rows(gi_detect_idx_dset, gi_detect_el['idx'])
            gi_el_snr_dset = file.create_dataset('gi_detect_el_snr',  (len(gi_detect_el),), dtype=dt1)
            write_vlen_rows(gi_el_snr_dset, gi_el_snr)
            gi_el_snr_dset.attrs['unit'] = '1'
            gi_el_snr_dset.attrs['definition'] = 'local mean per-pixel S/N around matched candidate; not integrated line S/N'
            dataset34 = file.create_dataset('gi_detect_el_elnumber', data = gi_detect_el['detect_el_number'])
            
            write_el_details(file.require_group('GV'), gv_el_details)
            write_el_details(file.require_group('GI'), gi_el_details)
            for band, noisy_lines, intrinsic_lines in (
                ('GV', gv_noisy_lines, gv_intrinsic_lines),
                ('GI', gi_noisy_lines, gi_intrinsic_lines)):
                group = file[band]
                write_line_measurements(group, emulator, noisy_lines, intrinsic_lines,
                    parameters, source_group=f[band])
                if band in wce_noisy_lines:
                    _, _, details = emulator.el_success(wce_noisy_lines[band], intrinsic_lines,
                        context.config['el_detect_success_th'], return_details=True)
                    write_el_details(group, details, suffix='_wce')
                    write_line_measurements(group, emulator, wce_noisy_lines[band], intrinsic_lines,
                        parameters, source_group=f[band], suffix='_wce')

            file.create_dataset('parent_row_indices', data=source_rows)
            file.create_dataset('input_row_indices', data=catalog_rows)
            file['parent_row_indices'].attrs['definition'] = 'row index into the immediate input product'
            file['input_row_indices'].attrs['definition'] = 'original catalogue row, inherited from parent when available'
            if 'input_quality_flags' in f:
                inherited = f['input_quality_flags']
                quality = file.create_dataset('input_quality_flags', data=inherited[:], dtype=inherited.dtype)
                for name in inherited.attrs:
                    attr = inherited.attrs.get_id(name)
                    quality.attrs.create(name, inherited.attrs[name], dtype=attr.dtype, shape=attr.shape)
            record_runner_provenance(file, context.config, {
                'apply_morphology_effect': False,
                'apply_wavelength_calibration_error': False,
                'apply_photometric': False, 'apply_el_detect': True}, source=f,
                seed=seed, input_sha256=input_identity, input_path=str(input_file),
                source_start=f.selection.start, source_stop=f.selection.stop)
            finalize_product(file, output_layout)

        toc = time.perf_counter()
        print(f"Finished, total time costs {toc - tic:0.4f} seconds for "+hdf5filename)
        return str(destination)


if __name__ == '__main__':
    from cess.cli.main import main
    raise SystemExit(main(['el-detect'] + sys.argv[1:]))
