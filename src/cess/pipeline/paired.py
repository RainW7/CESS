#!/usr/bin/env python3
# -*-coding:utf-8 -*-
'''
@Env     		:   CESS (Python 3.7.11) on Gravity
@File    		:   ~/emulator/CESS_v0.8.5/run.py
@Time    		:   2025/02/14 19:03:41
@Author  		:   Run Wen
@Version 		:   0.8.5
@Contact 		:   wenrun@pmo.ac.cn
@Description	:   Create simulation data for CSST slitless spectroscopy

Change log:
1.  Fixed the type of the mask array in the hdf5 file, avoding 'TypeError: Object dtype dtype('O') has no native HDF5 equivalent'
'''

import sys
from pathlib import Path
import numpy as np
import h5py
from astropy.table import Table
from collections.abc import Mapping
from scipy import interpolate
import time
import json
from cess.config.loader import load_config, validate_config
from cess.pipeline.random import source_rng, legacy_rng, file_identity
from cess.pipeline.support import (throughput_interpolator, throughput_arrays,
    write_response_products, write_spectral_covariance, write_line_measurements,
    resolve_spectral_extraction, write_spectral_extraction, write_mge_morphology, load_profile_libraries,
    combine_band_masks,
    pad_spectral_rows, output_destination, atomic_hdf5, write_vlen_rows,
    write_el_details, record_runner_provenance, normalise_options, cached_table,
    configure_morphology, SourceSelection, simulation_filename,
    legacy_main, measure_runner_band, empty_photometry_result,
    write_photometry_results)

from cess.pipeline.common import (prepare_resources, prepare_input, prepare_source_profile,
    measure_source_bands, photometry_columns, write_legacy_el_products,
    write_common_products, write_base_spectra, PHOTOMETRIC_BANDS)

from cess.pipeline.common import validate_output_layout, finalize_product
from cess.pipeline.collection import SpectralResults
from cess.pipeline.results import pad_results, emission_flags, desi_parameters, grism_parameters

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
    parameters.setdefault('photometry_exposures', dict(nuv=4,u=2,g=2,r=2,i=2,z=2,y=4))
    parameters.setdefault('zero_order_exposures', dict(GU=2,GV=2,GI=2))
    parameters.setdefault('photometry_aperture_fraction', .8)
    parameters.setdefault('photometry_aperture_radius_re', 15.)
    parameters.setdefault('photometry_systematic_mag', .02)
    parameters.setdefault('photometry_detection_snr', 5.)
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
    from cess.morphology import legacy as morphology
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
    """Simulate the half-open original source interval [start, stop).

    config accepts a JSON path or mapping; options accepts runner switches.
    seed derives stable per-source/component streams from input SHA256, ID and
    original row index. None preserves unseeded global-RNG compatibility.
    output_layout defaults to structured schema 3.0; legacy writes schema 2.5.
    Existing output files are never replaced. Return the new absolute path.
    """
    validate_output_layout(output_layout)
    from cess.pipeline.context import SimulationContext
    context = SimulationContext(config, options) if context is None else context
    import cess.spectra.units as utils
    morphology = context.morphology
    from cess.engine import emulator

    settings = context.config
    legacy_axes = context.axes
    resources = prepare_resources(settings, legacy_axes, context.options['apply_photometric'],
                                  include_zero=False, operations=context.operations)
    mge_context = resources['mge_context']
    mge_rows = []
    morphology_noise_options = resources['morphology_noise_options']
    bkg = resources['bkg']
    input_file = Path(input_path) / hdf5filename
    input_identity = file_identity(input_file)
    tic = time.perf_counter()
    print('Start generating CSST intrinsic spectra of {0}:\n'.format(hdf5filename))

    emulator = emulator(SED_file_path=str(input_path), file_name=hdf5filename, config=settings)

    with emulator.read_hdf5() as input_handle:
        f = SourceSelection(input_handle, source_start, source_stop)
        source_rows = f.source_rows
        destination = output_destination(
            context.config['output_path'] if output_dir is None else output_dir,
            simulation_filename(hdf5filename, f))
        # get the name of the best fit spectrum
        prepared = prepare_input(emulator, f, settings, resources, context.operations)
        spec_name = prepared['spec_name']
        mask, mask_nan = prepared['mask'], prepared['mask_nan']
        band_data = prepared['bands']
        guwave, gvwave, giwave = [band_data[band]['wave'] for band in ('GU', 'GV', 'GI')]
        gu_ujy, gv_ujy, gi_ujy = [band_data[band]['incident'] for band in ('GU', 'GV', 'GI')]
        gu_fnu, gv_fnu, gi_fnu = [band_data[band]['fnu'] for band in ('GU', 'GV', 'GI')]
        gu_source_rates, gv_source_rates, gi_source_rates = [band_data[band]['source_rates'] for band in ('GU', 'GV', 'GI')]
        gutp, gvtp, gitp = [band_data[band]['response']['first'] for band in ('GU', 'GV', 'GI')]
        gu_bkg_fnu, gv_bkg_fnu, gi_bkg_fnu = [band_data[band]['background_fnu'] for band in ('GU', 'GV', 'GI')]
        sky_reference, sky_rates = prepared['sky_reference'], prepared['sky_rates']
        parameters = prepared['parameters']
        raall, decall = parameters['RA'], parameters['DEC']
        m_starall, zall = parameters['log(M*)[0,1]_{MAL}'], parameters['z_{MAL}']
        maggall, magrall, magzall = parameters['MAG_G'], parameters['MAG_R'], parameters['MAG_Z']

        # create empty morphology parameters lists
        ntotal = []
        retotal = []
        patotal = []
        artotal = []

        spectral_results = SpectralResults({'GU': guwave, 'GV': gvwave, 'GI': giwave},
            paired=True,
            wave_error=context.options['apply_wavelength_calibration_error'])

        if context.options['apply_photometric'] == True:
            mag_nuv_total = []
            mag_u_total = []
            mag_g_total = []
            mag_r_total = []
            mag_i_total = []
            mag_z_total = []
            mag_y_total = []
            magerr_nuv_total = []
            magerr_u_total = []
            magerr_g_total = []
            magerr_r_total = []
            magerr_i_total = []
            magerr_z_total = []
            magerr_y_total = []
            snr_nuv_total = []
            snr_u_total = []
            snr_g_total = []
            snr_r_total = []
            snr_i_total = []
            snr_z_total = []
            snr_y_total = []
        
        photometry_results = {band: [] for band in ('nuv','u','g','r','i','z','y')} if context.options['apply_photometric'] else {}

        print('Start simulating CSST observed spectra of {0}:\n'.format(hdf5filename))
        spectral_waves = {'GU': guwave, 'GV': gvwave, 'GI': giwave}
        covariance_rows = {band: {suffix: [] for suffix in ('', '_wce')}
                           for band in spectral_waves}

        extraction_rows = []

        # loop over all the sources in one hdf5 file
        for i in (range(len(f['ID'][:]))):
            # get parameters from hdf5 file, total parameter names and the way to find the idx see the end of this file
            ra = raall[i]
            dec = decall[i]
            m_star = m_starall[i]
            z = zall[i]
            magz = magzall[i]

            def row_rng(component):
                return source_rng(seed, input_identity, f['ID'][i],
                                  int(source_rows[i]), component)

            row_covariances = {band: {suffix: np.full(len(wave) - 1, np.nan)
                for suffix in ('', '_wce')} for band, wave in spectral_waves.items()}
            row_extraction = None
            row_mge = None
            row_photometry = {band: empty_photometry_result() for band in photometry_results}

            if mask[i]:
                n, re, pa, ar, width, height, row_extraction, row_mge = prepare_source_profile(
                    emulator, morphology, settings, resources, prepared, i, row_rng,
                    context.options['apply_morphology_effect'], context.operations)

                phot_wave = f['best_fit']['wavelength_rest'][:] * 1.e4 * (1. + f['best_fit']['z'][i])
                phot_flux = f['best_fit'][spec_name][i] * 1.e-29
                if context.options['apply_photometric']:
                    row_photometry = measure_source_bands(emulator, ra, dec, phot_wave, phot_flux,
                        resources['photometric_tables'], bkg, settings, settings['photometry_exposures'],
                        re, row_rng, 'photometry_', context.operations)
                    (mag_nuv, mag_u, mag_g, mag_r, mag_i, mag_z, mag_y,
                     magerr_nuv, magerr_u, magerr_g, magerr_r, magerr_i, magerr_z, magerr_y,
                     snr_nuv, snr_u, snr_g, snr_r, snr_i, snr_z, snr_y) = photometry_columns(
                        row_photometry, PHOTOMETRIC_BANDS)

                # get simulated observed spectrum with noise added, including morphology effect and wavelength calibration error or not
                gu_flux_wce_single, gu_elec_per_column_single, gu_elec_per_column_wce_single, gu_fnu_with_noise_single, gu_fnu_with_noise_wce_single, gu_fnu_error_single, gu_efnu_wce_single, gu_rms_in_e_single, gu_snr_single, gu_snr_wce_single, gu_snr_mean_single, gu_wave_off_single, gu_spec_mask_single, gu_snr_mask_single, gu_covariance_single, gu_covariance_wce_single = emulator.add_noise_zouhu(ra,dec,guwave,gu_fnu[i],guwave,gu_bkg_fnu,gutp,height,width,'GU',context.options['apply_morphology_effect'],context.options['apply_wavelength_calibration_error'], rng=row_rng('noise_GU'), source_rate_e_s_cm2=gu_source_rates['first'][i], sky_rate_e_s_pixel=sky_rates['GU']['first'][i], return_covariance=True, source_fraction=row_extraction['source_fraction'], extraction_n_pixels=row_extraction['spatial_pixels'], **morphology_noise_options)
                gv_flux_wce_single, gv_elec_per_column_single, gv_elec_per_column_wce_single, gv_fnu_with_noise_single, gv_fnu_with_noise_wce_single, gv_fnu_error_single, gv_efnu_wce_single, gv_rms_in_e_single, gv_snr_single, gv_snr_wce_single, gv_snr_mean_single, gv_wave_off_single, gv_spec_mask_single, gv_snr_mask_single, gv_covariance_single, gv_covariance_wce_single = emulator.add_noise_zouhu(ra,dec,gvwave,gv_fnu[i],gvwave,gv_bkg_fnu,gvtp,height,width,'GV',context.options['apply_morphology_effect'],context.options['apply_wavelength_calibration_error'], rng=row_rng('noise_GV'), source_rate_e_s_cm2=gv_source_rates['first'][i], sky_rate_e_s_pixel=sky_rates['GV']['first'][i], return_covariance=True, source_fraction=row_extraction['source_fraction'], extraction_n_pixels=row_extraction['spatial_pixels'], **morphology_noise_options)
                gi_flux_wce_single, gi_elec_per_column_single, gi_elec_per_column_wce_single, gi_fnu_with_noise_single, gi_fnu_with_noise_wce_single, gi_fnu_error_single, gi_efnu_wce_single, gi_rms_in_e_single, gi_snr_single, gi_snr_wce_single, gi_snr_mean_single, gi_wave_off_single, gi_spec_mask_single, gi_snr_mask_single, gi_covariance_single, gi_covariance_wce_single = emulator.add_noise_zouhu(ra,dec,giwave,gi_fnu[i],giwave,gi_bkg_fnu,gitp,height,width,'GI',context.options['apply_morphology_effect'],context.options['apply_wavelength_calibration_error'], rng=row_rng('noise_GI'), source_rate_e_s_cm2=gi_source_rates['first'][i], sky_rate_e_s_pixel=sky_rates['GI']['first'][i], return_covariance=True, source_fraction=row_extraction['source_fraction'], extraction_n_pixels=row_extraction['spatial_pixels'], **morphology_noise_options)

                row_covariances['GU'][''] = gu_covariance_single * 1.e58
                row_covariances['GU']['_wce'] = gu_covariance_wce_single * 1.e58
                row_covariances['GV'][''] = gv_covariance_single * 1.e58
                row_covariances['GV']['_wce'] = gv_covariance_wce_single * 1.e58
                row_covariances['GI'][''] = gi_covariance_single * 1.e58
                row_covariances['GI']['_wce'] = gi_covariance_wce_single * 1.e58
             
                # append the results
                ntotal.append(n)
                retotal.append(re)
                patotal.append(pa)
                artotal.append(ar)

                spectral_results.append('GU', '',
                    (gu_elec_per_column_single, gu_fnu_with_noise_single, gu_fnu_error_single,
                     gu_rms_in_e_single, gu_snr_single, gu_snr_mean_single,
                     gu_spec_mask_single, gu_snr_mask_single),
                    wave_offset=gu_wave_off_single
                        if context.options['apply_wavelength_calibration_error'] else None)
                spectral_results.append('GU', '_wce',
                    (gu_flux_wce_single, gu_elec_per_column_wce_single, gu_fnu_with_noise_wce_single,
                     gu_efnu_wce_single, gu_snr_wce_single))
                spectral_results.append('GV', '',
                    (gv_elec_per_column_single, gv_fnu_with_noise_single, gv_fnu_error_single,
                     gv_rms_in_e_single, gv_snr_single, gv_snr_mean_single,
                     gv_spec_mask_single, gv_snr_mask_single),
                    wave_offset=gv_wave_off_single
                        if context.options['apply_wavelength_calibration_error'] else None)
                spectral_results.append('GV', '_wce',
                    (gv_flux_wce_single, gv_elec_per_column_wce_single, gv_fnu_with_noise_wce_single,
                     gv_efnu_wce_single, gv_snr_wce_single))
                spectral_results.append('GI', '',
                    (gi_elec_per_column_single, gi_fnu_with_noise_single, gi_fnu_error_single,
                     gi_rms_in_e_single, gi_snr_single, gi_snr_mean_single,
                     gi_spec_mask_single, gi_snr_mask_single),
                    wave_offset=gi_wave_off_single
                        if context.options['apply_wavelength_calibration_error'] else None)
                spectral_results.append('GI', '_wce',
                    (gi_flux_wce_single, gi_elec_per_column_wce_single, gi_fnu_with_noise_wce_single,
                     gi_efnu_wce_single, gi_snr_wce_single))

                if context.options['apply_photometric'] == True:
                    mag_nuv_total.append(mag_nuv)
                    mag_u_total.append(mag_u)
                    mag_g_total.append(mag_g)
                    mag_r_total.append(mag_r)
                    mag_i_total.append(mag_i)
                    mag_z_total.append(mag_z)
                    mag_y_total.append(mag_y)
                    magerr_nuv_total.append(magerr_nuv)
                    magerr_u_total.append(magerr_u)
                    magerr_g_total.append(magerr_g)
                    magerr_r_total.append(magerr_r)
                    magerr_i_total.append(magerr_i)
                    magerr_z_total.append(magerr_z)
                    magerr_y_total.append(magerr_y)
                    snr_nuv_total.append(snr_nuv)
                    snr_u_total.append(snr_u)
                    snr_g_total.append(snr_g)
                    snr_r_total.append(snr_r)
                    snr_i_total.append(snr_i)
                    snr_z_total.append(snr_z)
                    snr_y_total.append(snr_y)
                
            elif mask_nan[i]:

                ntotal.append(np.nan)
                retotal.append(np.nan)
                patotal.append(np.nan)
                artotal.append(np.nan)

                spectral_results.append_invalid()

                if context.options['apply_photometric'] == True:
                    mag_nuv_total.append(np.nan)
                    mag_u_total.append(np.nan)
                    mag_g_total.append(np.nan)
                    mag_r_total.append(np.nan)
                    mag_i_total.append(np.nan)
                    mag_z_total.append(np.nan)
                    mag_y_total.append(np.nan)
                    magerr_nuv_total.append(np.nan)
                    magerr_u_total.append(np.nan)
                    magerr_g_total.append(np.nan)
                    magerr_r_total.append(np.nan)
                    magerr_i_total.append(np.nan)
                    magerr_z_total.append(np.nan)
                    magerr_y_total.append(np.nan)
                    snr_nuv_total.append(np.nan)
                    snr_u_total.append(np.nan)
                    snr_g_total.append(np.nan)
                    snr_r_total.append(np.nan)
                    snr_i_total.append(np.nan)
                    snr_z_total.append(np.nan)
                    snr_y_total.append(np.nan)

            for band in spectral_waves:
                for suffix in covariance_rows[band]:
                    covariance_rows[band][suffix].append(row_covariances[band][suffix])

            extraction_rows.append(row_extraction)
            mge_rows.append(row_mge)

            for band in photometry_results:
                photometry_results[band].append(row_photometry[band])

        # array the results and reshape them to be stored in the hdf5 file and parameters table
        data_mask = np.zeros(f['ID'][:].shape, dtype=int)
        data_mask[mask] = 1
        data_mask[mask_nan] = 2

        ntotal = np.array(ntotal).reshape((len(f['ID'][:]),1))
        retotal = np.array(retotal).reshape((len(f['ID'][:]),1))
        patotal = np.array(patotal).reshape((len(f['ID'][:]),1))
        artotal = np.array(artotal).reshape((len(f['ID'][:]),1))
        
        data_dict, grism_stats = spectral_results.spectra(len(f['ID']))

        padded_data_dict = pad_results(data_dict, len(f['ID']),
            {'gu': guwave, 'gv': gvwave, 'gi': giwave}, pad_spectral_rows)

        grism_offsets = spectral_results.offsets(len(f['ID']))

        if context.options['apply_photometric'] == True:
            mag_nuv_total = np.array(mag_nuv_total).reshape((len(f['ID'][:]),1))
            mag_u_total = np.array(mag_u_total).reshape((len(f['ID'][:]),1))
            mag_g_total = np.array(mag_g_total).reshape((len(f['ID'][:]),1))
            mag_r_total = np.array(mag_r_total).reshape((len(f['ID'][:]),1))
            mag_i_total = np.array(mag_i_total).reshape((len(f['ID'][:]),1))
            mag_z_total = np.array(mag_z_total).reshape((len(f['ID'][:]),1))
            mag_y_total = np.array(mag_y_total).reshape((len(f['ID'][:]),1))
            magerr_nuv_total = np.array(magerr_nuv_total).reshape((len(f['ID'][:]),1))
            magerr_u_total = np.array(magerr_u_total).reshape((len(f['ID'][:]),1))
            magerr_g_total = np.array(magerr_g_total).reshape((len(f['ID'][:]),1))
            magerr_r_total = np.array(magerr_r_total).reshape((len(f['ID'][:]),1))
            magerr_i_total = np.array(magerr_i_total).reshape((len(f['ID'][:]),1))
            magerr_z_total = np.array(magerr_z_total).reshape((len(f['ID'][:]),1))
            magerr_y_total = np.array(magerr_y_total).reshape((len(f['ID'][:]),1))
            snr_nuv_total = np.array(snr_nuv_total).reshape((len(f['ID'][:]),1))
            snr_u_total = np.array(snr_u_total).reshape((len(f['ID'][:]),1))
            snr_g_total = np.array(snr_g_total).reshape((len(f['ID'][:]),1))
            snr_r_total = np.array(snr_r_total).reshape((len(f['ID'][:]),1))
            snr_i_total = np.array(snr_i_total).reshape((len(f['ID'][:]),1))
            snr_z_total = np.array(snr_z_total).reshape((len(f['ID'][:]),1))
            snr_y_total = np.array(snr_y_total).reshape((len(f['ID'][:]),1))
        
        if context.options['apply_el_detect'] == True:
            print('Start detecting emission lines of {0}:\n'.format(hdf5filename))
            # initialize the Qtable list
            gv_noisy_lines = []
            gv_intrinsic_lines = []
            gi_noisy_lines = []
            gi_intrinsic_lines = []
            # detect emission lines in both intrinsic and noisy spectra in GV and GI band
            for i in (range(len(f['ID'][:]))):
                gv_noisy_lines.append(emulator.el_detect(gvwave,padded_data_dict['padded_gv_ujy_with_noise'][i],context.config['gv_init_idx'],context.config['gv_end_idx'],context.config['noisy_el_detect_th']))
                gv_intrinsic_lines.append(emulator.el_detect(gvwave,gv_ujy[i],context.config['gv_init_idx'],context.config['gv_end_idx'],context.config['intri_el_detect_th']))
                gi_noisy_lines.append(emulator.el_detect(giwave,padded_data_dict['padded_gi_ujy_with_noise'][i],context.config['gi_init_idx'],context.config['gi_end_idx'],context.config['noisy_el_detect_th']))
                gi_intrinsic_lines.append(emulator.el_detect(giwave,gi_ujy[i],context.config['gi_init_idx'],context.config['gi_end_idx'],context.config['intri_el_detect_th']))
            # get the total emission line information array of intrinsic and noisy spectra
            gv_noisy_wce_lines = [emulator.el_detect(gvwave, spectrum, context.config['gv_init_idx'], context.config['gv_end_idx'], context.config['noisy_el_detect_th'])
                                  for spectrum in padded_data_dict['padded_gv_ujy_with_noise_wce']]
            gi_noisy_wce_lines = [emulator.el_detect(giwave, spectrum, context.config['gi_init_idx'], context.config['gi_end_idx'], context.config['noisy_el_detect_th'])
                                  for spectrum in padded_data_dict['padded_gi_ujy_with_noise_wce']]
            _, _, gv_wce_details = emulator.el_success(gv_noisy_wce_lines, gv_intrinsic_lines, context.config['el_detect_success_th'], return_details=True)
            _, _, gi_wce_details = emulator.el_success(gi_noisy_wce_lines, gi_intrinsic_lines, context.config['el_detect_success_th'], return_details=True)
            gv_intri_el, gv_detect_el, gv_el_details = emulator.el_success(gv_noisy_lines,gv_intrinsic_lines,context.config['el_detect_success_th'], return_details=True)
            gi_intri_el, gi_detect_el, gi_el_details = emulator.el_success(gi_noisy_lines,gi_intrinsic_lines,context.config['el_detect_success_th'], return_details=True)
            # flag the emission line detections situation, 0 = none el detect, 1 = only intrinsic el detection, 2 = both intrinsic and noisy detection
            gv_el_flag = emission_flags(len(gv_noisy_lines), gv_intri_el['id'], gv_detect_el['id'])
            gi_el_flag = emission_flags(len(gi_noisy_lines), gi_intri_el['id'], gi_detect_el['id'])
            # Get the mean signal-to-noise ratio of each emission line within the given width (in index)
            gv_el_snr = emulator.get_el_snr(padded_data_dict['padded_gv_snr'],gv_el_flag,gv_detect_el,context.config['gv_init_idx'],context.config['gv_elwidth'])
            gi_el_snr = emulator.get_el_snr(padded_data_dict['padded_gi_snr'],gi_el_flag,gi_detect_el,context.config['gi_init_idx'],context.config['gi_elwidth'])

        # saveing data 
        parameters_desi, parameters_desi_names = desi_parameters(len(f['ID']),
            raall, decall, zall, maggall, magrall, magzall,
            ntotal, retotal, patotal, artotal, m_starall)

        parameters_grism, parameters_grism_names = grism_parameters(len(f['ID']),
            (grism_stats['gu_rms_in_e'], grism_stats['gv_rms_in_e'], grism_stats['gi_rms_in_e']),
            (grism_stats['gu_snr_mean'], grism_stats['gv_snr_mean'], grism_stats['gi_snr_mean']),
            wave_offsets=(grism_offsets['gu_wave_off'], grism_offsets['gv_wave_off'], grism_offsets['gi_wave_off'])
                if context.options['apply_wavelength_calibration_error'] else None,
            el_flags=(gv_el_flag, gi_el_flag) if context.options['apply_el_detect'] else None)

        if context.options['apply_photometric'] == True:
            parameters_phot = np.hstack([mag_nuv_total,mag_u_total,mag_g_total,mag_r_total,mag_i_total,mag_z_total,mag_y_total,
                                        magerr_nuv_total,magerr_u_total,magerr_g_total,magerr_r_total,magerr_i_total,magerr_z_total,magerr_y_total,
                                        snr_nuv_total,snr_u_total,snr_g_total,snr_r_total,snr_i_total,snr_z_total,snr_y_total])
        
        with atomic_hdf5(destination) as file:

            dataset1 = file.create_dataset('ID', data = f['ID'][:], dtype=f['ID'].dtype)
            dataset2 = file.create_dataset('parameters_desi', data = parameters_desi)
            dataset2.attrs['name'] = parameters_desi_names
            
            dataset99 = file.create_dataset('parameters_grism', data=parameters_grism)
            dataset99.attrs['name'] = parameters_grism_names

            if context.options['apply_photometric'] == True:
                dataset98 = file.create_dataset('parameters_phot', data = parameters_phot)
                dataset98.attrs['name'] = ['mag_nuv','mag_u','mag_g','mag_r','mag_i','mag_z','mag_y',
                                           'magerr_nuv','magerr_u','magerr_g','magerr_r','magerr_i','magerr_z','magerr_y',
                                           'snr_nuv','snr_u','snr_g','snr_r','snr_i','snr_z','snr_y']
            
            # data mask 
            dataset97 = file.create_dataset('data_mask', data = data_mask)

            groups = write_base_spectra(file, spectral_waves,
                {'GU': gu_ujy, 'GV': gv_ujy, 'GI': gi_ujy}, padded_data_dict,
                paired=True)
            group_gu, group_gv, group_gi = (groups[band] for band in ('GU', 'GV', 'GI'))

            if context.options['apply_el_detect']:
                write_legacy_el_products((
                    (group_gv, gv_intri_el, gv_detect_el, gv_el_snr, gv_el_details),
                    (group_gi, gi_intri_el, gi_detect_el, gi_el_snr, gi_el_details)), context.operations)

            if context.options['apply_photometric']:
                write_photometry_results(file, 'photometry', photometry_results, settings)
            write_common_products(file, {'GU': group_gu, 'GV': group_gv, 'GI': group_gi},
                f, settings, resources, prepared, extraction_rows, mge_rows, covariance_rows,
                paired=True, operations=context.operations)
            if context.options['apply_el_detect']:
                write_line_measurements(group_gv, emulator, gv_noisy_lines, gv_intrinsic_lines, settings)
                write_line_measurements(group_gi, emulator, gi_noisy_lines, gi_intrinsic_lines, settings)
                write_el_details(group_gv, gv_wce_details, suffix='_wce')
                write_el_details(group_gi, gi_wce_details, suffix='_wce')
                write_line_measurements(group_gv, emulator, gv_noisy_wce_lines, gv_intrinsic_lines, settings, suffix='_wce')
                write_line_measurements(group_gi, emulator, gi_noisy_wce_lines, gi_intrinsic_lines, settings, suffix='_wce')

            record_runner_provenance(file, context.config, {
                'apply_morphology_effect': context.options['apply_morphology_effect'],
                'apply_wavelength_calibration_error': context.options['apply_wavelength_calibration_error'],
                'apply_photometric': context.options['apply_photometric'],
                'apply_el_detect': context.options['apply_el_detect']}, seed=seed,
                input_sha256=input_identity, input_path=str(input_file),
                source_start=f.selection.start, source_stop=f.selection.stop)
            finalize_product(file, output_layout)

        toc = time.perf_counter()
        print(f"Finished, total time costs {toc - tic:0.4f} seconds for "+str(input_file))
        return str(destination)

hdf5filenames = [
'seedcat2_0420_1199_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1204_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_842_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_971_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
]

if __name__ == '__main__':
    sys.exit(legacy_main(task, configure, hdf5filenames))
