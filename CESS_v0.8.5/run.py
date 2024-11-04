#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@Env     		:   grizli (Python 3.7.11) on Macbook Pro
@File    		:   ~/emulator/emulator_v0.8.5/run.py
@Time    		:   2024/09/25 10:19:26
@Author  		:   Run Wen
@Version 		:   0.8.5
@Contact 		:   wenrun@pmo.ac.cn
@Description	:   Slitless simulation data for CSST
'''

import sys
import numpy as np
import pickle
# from tqdm import tqdm
import h5py
from multiprocessing import Process
from scipy import interpolate
from astropy.table import Table
import time
from astropy.cosmology import FlatLambdaCDM
import json
import random
hubble=70
cosmo = FlatLambdaCDM(H0=hubble,Om0=0.3)

desipath = '/home/runwen/hanyk/CSST_mock_spec_phot_sedlib3/'
bkg = Table.read('/home/runwen/emulator_v0.8.5/bkg_spec.fits')

# set a single run length in one run and switchs of morphology effect and wavelength calibration error
# seed_value = 42
# random.seed(seed_value)
begin = int(sys.argv[1])
end = int(sys.argv[2])
apply_morphology_effect = False
apply_wavelength_calibration_error = False
apply_photometric = False
apply_el_detect = False

for arg in sys.argv[1:]:
    if arg.lower() == 'morph=true' or arg.lower() == 'morph=yes':
        apply_morphology_effect = True
    elif arg.lower() == 'wave_cal=true' or arg.lower() == 'wave_cal=yes':
        apply_wavelength_calibration_error = True
    elif arg.lower() == 'photo=true' or arg.lower() == 'photo=yes':
        apply_photometric = True
    elif arg.lower() == 'el_detect=true' or arg.lower() == 'el_detect=yes':
        apply_el_detect = True

with open('emulator_parameters.json', 'r') as f:
    emulator_parameters = json.load(f)

radi = emulator_parameters['radi'] # telescope radius, cm
colarea = np.pi*radi**2 # cm^2
expt = emulator_parameters['expt'] # exposure time, s
dc_value = emulator_parameters['dc_value']
arcsecperpix = emulator_parameters['arcsecperpix']

if apply_el_detect == True:
    gv_init_idx = emulator_parameters['gv_init_idx']
    gv_end_idx = emulator_parameters['gv_end_idx']
    gi_init_idx = emulator_parameters['gi_init_idx']
    gi_end_idx = emulator_parameters['gi_end_idx']
    noisy_el_detect_th = emulator_parameters['noisy_el_detect_th']
    intri_el_detect_th = emulator_parameters['intri_el_detect_th']
    el_detect_success_th = emulator_parameters['el_detect_success_th']
    gv_elwidth = emulator_parameters['gv_elwidth']
    gi_elwidth = emulator_parameters['gi_elwidth']

gutp_path = emulator_parameters['gutp_path']
gvtp_path = emulator_parameters['gvtp_path']
gitp_path = emulator_parameters['gitp_path']

gu_ree80 = emulator_parameters['gu_ree80']
gv_ree80 = emulator_parameters['gv_ree80']
gi_ree80 = emulator_parameters['gi_ree80']

nuvtp_path = emulator_parameters['nuvtp_path']
utp_path = emulator_parameters['utp_path']
gtp_path = emulator_parameters['gtp_path']
rtp_path = emulator_parameters['rtp_path']
itp_path = emulator_parameters['itp_path']
ztp_path = emulator_parameters['ztp_path']
ytp_path = emulator_parameters['ytp_path']

delta_p_sys = 0.02

with open('widthlib_20x20x10x10.pkl', 'rb') as f:
    widthlib = pickle.load(f)
    
with open('heightlib_20x20x10x10.pkl', 'rb') as f:
    heightlib = pickle.load(f)

nseries = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1. ,1.1, 1.2, 1.3, 1.4, 
                    1.5, 1.6, 1.8, 2. ,2.5, 3., 3.5, 4., 4.5, 5.]) # 20
reseries = np.round(np.array([0.3, 0.5, 0.7, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 
                              2.5, 3, 3.5,  4.5, 5, 5.5, 6, 6.5, 7, 7.4])/arcsecperpix) # 20
paseries = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90]) # 10
baseries = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) # 10

def task(desipath,hdf5filename):
    import utils
    import morphology 
    from main import emulator

    tic = time.perf_counter()

    print('Start generating CSST intrinsic spectra of {0}:\n'.format(hdf5filename))

    # read the 3 slitless throughput curve from fits files, length ~ 1600+
    gu_1st_tp = Table.read(gutp_path)
    gv_1st_tp = Table.read(gvtp_path)
    gi_1st_tp = Table.read(gitp_path)

    # create the transmission curve function to prepare for interpolation
    gu_tpf = interpolate.interp1d(gu_1st_tp['WAVELENGTH'],gu_1st_tp['SENSITIVITY'])
    gv_tpf = interpolate.interp1d(gv_1st_tp['WAVELENGTH'],gv_1st_tp['SENSITIVITY'])
    gi_tpf = interpolate.interp1d(gi_1st_tp['WAVELENGTH'],gi_1st_tp['SENSITIVITY'])

    if apply_photometric == True:
        # read the 7 photometric throughput curve from fits files, length ~ 100-
        phot_nuv = Table.read(nuvtp_path)
        phot_u = Table.read(utp_path)
        phot_g = Table.read(gtp_path)
        phot_r = Table.read(rtp_path)
        phot_i = Table.read(itp_path)
        phot_z = Table.read(ztp_path)
        phot_y = Table.read(ytp_path)

        phot_nuv_tpf = interpolate.interp1d(phot_nuv['WAVELENGTH'],phot_nuv['SENSITIVITY'])
        phot_u_tpf = interpolate.interp1d(phot_u['WAVELENGTH'],phot_u['SENSITIVITY'])
        phot_g_tpf = interpolate.interp1d(phot_g['WAVELENGTH'],phot_g['SENSITIVITY'])
        phot_r_tpf = interpolate.interp1d(phot_r['WAVELENGTH'],phot_r['SENSITIVITY'])
        phot_i_tpf = interpolate.interp1d(phot_i['WAVELENGTH'],phot_i['SENSITIVITY'])
        phot_z_tpf = interpolate.interp1d(phot_z['WAVELENGTH'],phot_z['SENSITIVITY'])
        phot_y_tpf = interpolate.interp1d(phot_y['WAVELENGTH'],phot_y['SENSITIVITY'])

    # create the sky background curve function to prepare for interpolation
    bkgf = interpolate.interp1d(bkg['wavelength'], bkg['fnu'])

    emulator = emulator(SED_file_path = desipath, file_name = hdf5filename)

    f = emulator.read_hdf5()
    # get the name of the best fit spectrum
    spec_name = next(name for name in f['best_fit'] if name.startswith('spec_'))
    # generate intrinsic CSST simulated slitless spectra data, in pixels of length ~ 500+
    # input DESI SED library wavelength unit is um, times 1e4 to convert into angstrom, 
    # and the spectral unit is ujy.
    guwave, gu_ujy, mask, mask_nan = emulator.kernels_convolve(f['best_fit']['wavelength_rest'][:]*1e4,
                                               f['best_fit'][spec_name][:], 
                                               f['best_fit']['z'], 
                                               'GU', gu_ree80)
    gvwave, gv_ujy, mask, mask_nan = emulator.kernels_convolve(f['best_fit']['wavelength_rest'][:]*1e4,
                                               f['best_fit'][spec_name][:], 
                                               f['best_fit']['z'], 
                                               'GV', gv_ree80)
    giwave, gi_ujy, mask, mask_nan = emulator.kernels_convolve(f['best_fit']['wavelength_rest'][:]*1e4,
                                               f['best_fit'][spec_name][:], 
                                               f['best_fit']['z'], 
                                               'GI', gi_ree80)

    # interpolate the throughput for each grism bands
    gutp = gu_tpf(guwave)
    gvtp = gv_tpf(gvwave)
    gitp = gi_tpf(giwave)
    gu_bkg_fnu = bkgf(guwave)
    gv_bkg_fnu = bkgf(gvwave)
    gi_bkg_fnu = bkgf(giwave)

    # convert ujy into fnu
    gu_fnu = gu_ujy*1e-29
    gv_fnu = gv_ujy*1e-29
    gi_fnu = gi_ujy*1e-29
    # the median wavelength of GI band, convert to GI magnitude
    # gimag = utils.fnu2mAB(gi_fnu[:,(np.abs(giwave-gi_wave_mid)).argmin()])
    COLNAMES=[x.decode('utf-8') for x in f['parameters_name'][:]]
    parameters=Table(f['parameters'][:],names=COLNAMES,copy=False)
    raall = parameters['RA']
    decall = parameters['DEC']
    m_starall = parameters['log(M*)[0,1]_{MAL}']
    zall = parameters['z_{MAL}']
    maggall = parameters['MAG_G']
    magrall = parameters['MAG_R']
    magzall = parameters['MAG_Z']

    # create empty morphology parameters lists
    ntotal = []
    retotal = []
    patotal = []
    artotal = []

    # create 7 empty lists of the add_noise function for 3 CSST grism bands
    gu_elec_per_column = []
    gu_fnu_with_noise = []
    gu_efnu = []
    gu_rms_in_e = []
    gu_snr = []
    gu_snr_mean = []        
    gu_spec_mask = []
    gu_snr_mask = []

    gv_elec_per_column = []
    gv_fnu_with_noise = []
    gv_efnu = []
    gv_rms_in_e = []
    gv_snr = []
    gv_snr_mean = []        
    gv_spec_mask = []
    gv_snr_mask = []

    gi_elec_per_column = []
    gi_fnu_with_noise = []
    gi_efnu = []
    gi_rms_in_e = []
    gi_snr = []
    gi_snr_mean = []
    gi_spec_mask = []
    gi_snr_mask = []
        
    if apply_wavelength_calibration_error == True:
        gu_wave_off = []
        gv_wave_off = []
        gi_wave_off = []
    
    if apply_photometric == True:
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
    
    print('Start simulating CSST observed spectra of {0}:\n'.format(hdf5filename))
    # loop over all the sources in one hdf5 file
    for i in (range(len(f['ID'][:]))):
        # get parameters from hdf5 file, total parameter names and the way to find the idx see the end of this file
        ra = raall[i]
        dec = decall[i]
        m_star = m_starall[i]
        z = zall[i]
        magz = magzall[i]

        if mask[i]:
            # generate 2d parameters
            n,re,pa,ar = morphology.get_2d_param(magz, 14, 21, 0.5, 5, m_star, z)

            # get 2d profile distribution function
            width = morphology.match_input_paramters(widthlib,n,re,pa,ar,nseries,reseries,paseries,baseries)
            height = morphology.match_input_paramters(heightlib,n,re,pa,ar,nseries,reseries,paseries,baseries)

            if apply_photometric == True:
                # get spectra in AA and erg/s/cm^2/Hz for the photometric estimation
                phot_wave = f['best_fit']['wavelength_rest'][:]*1e4*(1+f['best_fit']['z'][i])
                phot_flux = f['best_fit'][spec_name][i]*1e-29

                # get truncated spectra for 7 photometric bands
                phot_nuv_wave, phot_nuv_flux = utils.array_cut(phot_wave, phot_flux, phot_nuv['WAVELENGTH'])
                phot_u_wave, phot_u_flux = utils.array_cut(phot_wave, phot_flux, phot_u['WAVELENGTH'])
                phot_g_wave, phot_g_flux = utils.array_cut(phot_wave, phot_flux, phot_g['WAVELENGTH'])
                phot_r_wave, phot_r_flux = utils.array_cut(phot_wave, phot_flux, phot_r['WAVELENGTH'])
                phot_i_wave, phot_i_flux = utils.array_cut(phot_wave, phot_flux, phot_i['WAVELENGTH'])
                phot_y_wave, phot_y_flux = utils.array_cut(phot_wave, phot_flux, phot_y['WAVELENGTH'])
                phot_z_wave, phot_z_flux = utils.array_cut(phot_wave, phot_flux, phot_z['WAVELENGTH'])
            
                # get AB magnitudes for 7 photometric bands
                mag_nuv = utils.fnu2mAB(np.mean(phot_nuv_flux))
                mag_u = utils.fnu2mAB(np.mean(phot_u_flux))
                mag_g = utils.fnu2mAB(np.mean(phot_g_flux))
                mag_r = utils.fnu2mAB(np.mean(phot_r_flux))
                mag_i = utils.fnu2mAB(np.mean(phot_i_flux))
                mag_z = utils.fnu2mAB(np.mean(phot_z_flux))
                mag_y = utils.fnu2mAB(np.mean(phot_y_flux))

                # get total electrons value for 7 photometric bands
                total_elec_nuv = 0.8 * np.trapz(phot_nuv_tpf(phot_nuv_wave)*utils.fnu2fphot(phot_nuv_wave, phot_nuv_flux), phot_nuv_wave)*colarea*expt*4
                total_elec_u = 0.8 * np.trapz(phot_u_tpf(phot_u_wave)*utils.fnu2fphot(phot_u_wave, phot_u_flux), phot_u_wave)*colarea*expt*2
                total_elec_g = 0.8 * np.trapz(phot_g_tpf(phot_g_wave)*utils.fnu2fphot(phot_g_wave, phot_g_flux), phot_g_wave)*colarea*expt*2
                total_elec_r = 0.8 * np.trapz(phot_r_tpf(phot_r_wave)*utils.fnu2fphot(phot_r_wave, phot_r_flux), phot_r_wave)*colarea*expt*2
                total_elec_i = 0.8 * np.trapz(phot_i_tpf(phot_i_wave)*utils.fnu2fphot(phot_i_wave, phot_i_flux), phot_i_wave)*colarea*expt*2
                total_elec_z = 0.8 * np.trapz(phot_z_tpf(phot_z_wave)*utils.fnu2fphot(phot_z_wave, phot_z_flux), phot_z_wave)*colarea*expt*2
                total_elec_y = 0.8 * np.trapz(phot_y_tpf(phot_y_wave)*utils.fnu2fphot(phot_y_wave, phot_y_flux), phot_y_wave)*colarea*expt*4
                
                # get bkg value for 7 photometric bands 
                total_bkg_nuv = emulator.bkg_per_pix_time(ra,dec,phot_nuv_wave,bkgf(phot_nuv_wave),phot_nuv_tpf(phot_nuv_wave))*150*4*np.pi*(15*re)**2
                total_bkg_u = emulator.bkg_per_pix_time(ra,dec,phot_u_wave,bkgf(phot_u_wave),phot_u_tpf(phot_u_wave))*expt*2*np.pi*(15*re)**2
                total_bkg_g = emulator.bkg_per_pix_time(ra,dec,phot_g_wave,bkgf(phot_g_wave),phot_g_tpf(phot_g_wave))*expt*2*np.pi*(15*re)**2
                total_bkg_r = emulator.bkg_per_pix_time(ra,dec,phot_r_wave,bkgf(phot_r_wave),phot_r_tpf(phot_r_wave))*expt*2*np.pi*(15*re)**2
                total_bkg_i = emulator.bkg_per_pix_time(ra,dec,phot_i_wave,bkgf(phot_i_wave),phot_i_tpf(phot_i_wave))*expt*2*np.pi*(15*re)**2
                total_bkg_z = emulator.bkg_per_pix_time(ra,dec,phot_z_wave,bkgf(phot_z_wave),phot_z_tpf(phot_z_wave))*expt*2*np.pi*(15*re)**2
                total_bkg_y = emulator.bkg_per_pix_time(ra,dec,phot_y_wave,bkgf(phot_y_wave),phot_y_tpf(phot_y_wave))*expt*4*np.pi*(15*re)**2
                
                # get snr value for 7 photometric bands 
                snr_nuv = total_elec_nuv / np.sqrt(total_elec_nuv + total_bkg_nuv + 0.02*expt*4*np.pi*(15*re)**2 + np.pi*(15*re)**2*4*25)
                snr_u = total_elec_u / np.sqrt(total_elec_u + total_bkg_u + 0.02*expt*2*np.pi*(15*re)**2 + np.pi*(15*re)**2*2*25)
                snr_g = total_elec_g / np.sqrt(total_elec_g + total_bkg_g + 0.02*expt*2*np.pi*(15*re)**2 + np.pi*(15*re)**2*2*25)
                snr_r = total_elec_r / np.sqrt(total_elec_r + total_bkg_r + 0.02*expt*2*np.pi*(15*re)**2 + np.pi*(15*re)**2*2*25)
                snr_i = total_elec_i / np.sqrt(total_elec_i + total_bkg_i + 0.02*expt*2*np.pi*(15*re)**2 + np.pi*(15*re)**2*2*25)
                snr_z = total_elec_z / np.sqrt(total_elec_z + total_bkg_z + 0.02*expt*2*np.pi*(15*re)**2 + np.pi*(15*re)**2*2*25)
                snr_y = total_elec_y / np.sqrt(total_elec_y + total_bkg_y + 0.02*expt*4*np.pi*(15*re)**2 + np.pi*(15*re)**2*4*25)
        
                # calculate the magnitude error
                delta_p_nuv = 2.5*np.log10(1+1/snr_nuv)
                delta_p_u = 2.5*np.log10(1+1/snr_u)
                delta_p_g = 2.5*np.log10(1+1/snr_g)
                delta_p_r = 2.5*np.log10(1+1/snr_r)
                delta_p_i = 2.5*np.log10(1+1/snr_i)
                delta_p_z = 2.5*np.log10(1+1/snr_z)
                delta_p_y = 2.5*np.log10(1+1/snr_y)
                magerr_nuv = np.sqrt(delta_p_nuv**2 + delta_p_sys**2)
                magerr_u = np.sqrt(delta_p_u**2 + delta_p_sys**2)
                magerr_g = np.sqrt(delta_p_g**2 + delta_p_sys**2)
                magerr_r = np.sqrt(delta_p_r**2 + delta_p_sys**2)
                magerr_i = np.sqrt(delta_p_i**2 + delta_p_sys**2)
                magerr_z = np.sqrt(delta_p_z**2 + delta_p_sys**2)
                magerr_y = np.sqrt(delta_p_y**2 + delta_p_sys**2)

            # get simulated observed spectrum with noise added, including morphology effect and wavelength calibration error or not
            gu_elec_per_column_single, gu_fnu_with_noise_single, gu_fnu_error_single, gu_rms_in_e_single, gu_snr_single, gu_snr_mean_single, gu_wave_off_single, gu_spec_mask_single, gu_snr_mask_single = emulator.add_noise(ra,dec,guwave,gu_fnu[i],guwave,gu_bkg_fnu,gutp,height,width,'GU',apply_morphology_effect,apply_wavelength_calibration_error)
            gv_elec_per_column_single, gv_fnu_with_noise_single, gv_fnu_error_single, gv_rms_in_e_single, gv_snr_single, gv_snr_mean_single, gv_wave_off_single, gv_spec_mask_single, gv_snr_mask_single = emulator.add_noise(ra,dec,gvwave,gv_fnu[i],gvwave,gv_bkg_fnu,gvtp,height,width,'GV',apply_morphology_effect,apply_wavelength_calibration_error)
            gi_elec_per_column_single, gi_fnu_with_noise_single, gi_fnu_error_single, gi_rms_in_e_single, gi_snr_single, gi_snr_mean_single, gi_wave_off_single, gi_spec_mask_single, gi_snr_mask_single = emulator.add_noise(ra,dec,giwave,gi_fnu[i],giwave,gi_bkg_fnu,gitp,height,width,'GI',apply_morphology_effect,apply_wavelength_calibration_error)
            
            # append the results
            ntotal.append(n)
            retotal.append(re)
            patotal.append(pa)
            artotal.append(ar)

            gu_elec_per_column.append(gu_elec_per_column_single)
            gu_fnu_with_noise.append(gu_fnu_with_noise_single)
            gu_efnu.append(gu_fnu_error_single)
            gu_rms_in_e.append(gu_rms_in_e_single)
            gu_snr.append(gu_snr_single)
            gu_snr_mean.append(gu_snr_mean_single)
            gu_spec_mask.append(gu_spec_mask_single)
            gu_snr_mask.append(gu_snr_mask_single)

            gv_elec_per_column.append(gv_elec_per_column_single)
            gv_fnu_with_noise.append(gv_fnu_with_noise_single)
            gv_efnu.append(gv_fnu_error_single)
            gv_rms_in_e.append(gv_rms_in_e_single)
            gv_snr.append(gv_snr_single)
            gv_snr_mean.append(gv_snr_mean_single)
            gv_spec_mask.append(gv_spec_mask_single)
            gv_snr_mask.append(gv_snr_mask_single)

            gi_elec_per_column.append(gi_elec_per_column_single)
            gi_fnu_with_noise.append(gi_fnu_with_noise_single)
            gi_efnu.append(gi_fnu_error_single)
            gi_rms_in_e.append(gi_rms_in_e_single)
            gi_snr.append(gi_snr_single)
            gi_snr_mean.append(gi_snr_mean_single)
            gi_spec_mask.append(gi_spec_mask_single)
            gi_snr_mask.append(gi_snr_mask_single)

            if apply_wavelength_calibration_error == True:
                gu_wave_off.append(gu_wave_off_single)
                gv_wave_off.append(gv_wave_off_single)
                gi_wave_off.append(gi_wave_off_single)

            if apply_photometric == True:
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

            ntotal.append(n)
            retotal.append(re)
            patotal.append(pa)
            artotal.append(ar)

            gu_elec_per_column.append(np.nan)
            gu_fnu_with_noise.append(np.nan)
            gu_efnu.append(np.nan)
            gu_rms_in_e.append(np.nan)
            gu_snr.append(np.nan)
            gu_snr_mean.append(np.nan)
            gu_spec_mask.append(np.nan)
            gu_snr_mask.append(np.nan)

            gv_elec_per_column.append(np.nan)
            gv_fnu_with_noise.append(np.nan)
            gv_efnu.append(np.nan)
            gv_rms_in_e.append(np.nan)
            gv_snr.append(np.nan)
            gv_snr_mean.append(np.nan)
            gv_spec_mask.append(np.nan)
            gv_snr_mask.append(np.nan)

            gi_elec_per_column.append(np.nan)
            gi_fnu_with_noise.append(np.nan)
            gi_efnu.append(np.nan)
            gi_rms_in_e.append(np.nan)
            gi_snr.append(np.nan)
            gi_snr_mean.append(np.nan)
            gi_spec_mask.append(np.nan)
            gi_snr_mask.append(np.nan)

            if apply_wavelength_calibration_error == True:
                gu_wave_off.append(np.nan)
                gv_wave_off.append(np.nan)
                gi_wave_off.append(np.nan)

            if apply_photometric == True:
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

    # array the results and reshape them to be stored in the hdf5 file and parameters table
    data_mask = np.zeros(f['ID'][:].shape, dtype=int)
    data_mask[mask] = 1
    data_mask[mask_nan] = 2
    
    ntotal = np.array(ntotal).reshape((len(f['ID'][:]),1))
    retotal = np.array(retotal).reshape((len(f['ID'][:]),1))
    patotal = np.array(patotal).reshape((len(f['ID'][:]),1))
    artotal = np.array(artotal).reshape((len(f['ID'][:]),1))
    
    gu_elec_per_column = np.array(gu_elec_per_column)
    gu_ujy_with_noise = np.array(gu_fnu_with_noise)*1e29
    gu_ferr = np.array(gu_efnu)*1e29
    gu_rms_in_e = np.array(gu_rms_in_e).reshape((len(f['ID'][:]),1))
    gu_snr = np.array(gu_snr)
    gu_snr_mean = np.array(gu_snr_mean).reshape((len(f['ID'][:]),1))
    gu_spec_mask = np.array(gu_spec_mask)
    gu_snr_mask = np.array(gu_snr_mask)

    gv_elec_per_column = np.array(gv_elec_per_column)
    gv_ujy_with_noise = np.array(gv_fnu_with_noise)*1e29
    gv_ferr = np.array(gv_efnu)*1e29
    gv_rms_in_e = np.array(gv_rms_in_e).reshape((len(f['ID'][:]),1))
    gv_snr = np.array(gv_snr)
    gv_snr_mean = np.array(gv_snr_mean).reshape((len(f['ID'][:]),1))
    gv_spec_mask = np.array(gv_spec_mask)
    gv_snr_mask = np.array(gv_snr_mask)

    gi_elec_per_column = np.array(gi_elec_per_column)
    gi_ujy_with_noise = np.array(gi_fnu_with_noise)*1e29
    gi_ferr = np.array(gi_efnu)*1e29
    gi_rms_in_e = np.array(gi_rms_in_e).reshape((len(f['ID'][:]),1))
    gi_snr = np.array(gi_snr)
    gi_snr_mean = np.array(gi_snr_mean).reshape((len(f['ID'][:]),1))
    gi_spec_mask = np.array(gi_spec_mask)
    gi_snr_mask = np.array(gi_snr_mask)

    # create a dict and connect the name with variable
    data_dict = {
        'gu_elec_per_column': gu_elec_per_column,
        'gu_ujy_with_noise': gu_ujy_with_noise,
        'gu_ferr': gu_ferr,
        'gu_snr': gu_snr,
        'gu_spec_mask': gu_spec_mask,
        'gu_snr_mask': gu_snr_mask,

        'gv_elec_per_column': gv_elec_per_column,
        'gv_ujy_with_noise': gv_ujy_with_noise,
        'gv_ferr': gv_ferr,
        'gv_snr': gv_snr,
        'gv_spec_mask': gv_spec_mask,
        'gv_snr_mask': gv_snr_mask,

        'gi_elec_per_column': gi_elec_per_column,
        'gi_ujy_with_noise': gi_ujy_with_noise,
        'gi_ferr': gi_ferr,
        'gi_snr': gi_snr,
        'gi_spec_mask': gi_spec_mask,
        'gi_snr_mask': gi_snr_mask
    }

    # create a dict to store the padded data
    padded_data_dict = {}

    # loop the variable and pad the data
    for key, data in data_dict.items():
        valid_arrays = [arr for arr in data if isinstance(arr, np.ndarray)]
        max_len = max(len(arr) for arr in valid_arrays)
        padded_data = np.array([
            np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan) if isinstance(arr, np.ndarray) else np.full(max_len, np.nan)
            for arr in data
        ], dtype=np.float64)
        
        padded_data_dict[f'padded_{key}'] = padded_data

    if apply_wavelength_calibration_error == True:
        gu_wave_off = np.array(gu_wave_off).reshape((len(f['ID'][:]),1))
        gv_wave_off = np.array(gv_wave_off).reshape((len(f['ID'][:]),1))
        gi_wave_off = np.array(gi_wave_off).reshape((len(f['ID'][:]),1))

    if apply_photometric == True:
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
    
    if apply_el_detect == True:
        print('Start detecting emission lines of {0}:\n'.format(hdf5filename))
        # initialize the Qtable list
        gv_noisy_lines = []
        gv_intrinsic_lines = []
        gi_noisy_lines = []
        gi_intrinsic_lines = []
        # detect emission lines in both intrinsic and noisy spectra in GV and GI band
        for i in (range(len(f['ID'][:]))):
            gv_noisy_lines.append(emulator.el_detect(gvwave,gv_ujy_with_noise[i],gv_init_idx,gv_end_idx,noisy_el_detect_th))
            gv_intrinsic_lines.append(emulator.el_detect(gvwave,gv_ujy[i],gv_init_idx,gv_end_idx,intri_el_detect_th))
            gi_noisy_lines.append(emulator.el_detect(giwave,gi_ujy_with_noise[i],gi_init_idx,gi_end_idx,noisy_el_detect_th))
            gi_intrinsic_lines.append(emulator.el_detect(giwave,gi_ujy[i],gi_init_idx,gi_end_idx,intri_el_detect_th))
        # get the total emission line information array of intrinsic and noisy spectra
        gv_intri_el, gv_detect_el = emulator.el_success(gv_noisy_lines,gv_intrinsic_lines,el_detect_success_th)
        gi_intri_el, gi_detect_el = emulator.el_success(gi_noisy_lines,gi_intrinsic_lines,el_detect_success_th)
        # flag the emission line detections situation, 0 = none el detect, 1 = only intrinsic el detection, 2 = both intrinsic and noisy detection
        gv_el_flag = np.zeros(len(gv_noisy_lines))
        gi_el_flag = np.zeros(len(gi_noisy_lines))
        for i in range(len(f['ID'][:])):
            if i in gv_intri_el['id']:
                gv_el_flag[i] = 1
                if i in gv_detect_el['id']:
                    gv_el_flag[i] = 2
            elif i not in gv_intri_el['id']:
                gv_el_flag[i] = 0
        for i in range(len(f['ID'][:])):
            if i in gi_intri_el['id']:
                gi_el_flag[i] = 1
                if i in gi_detect_el['id']:
                    gi_el_flag[i] = 2
            elif i not in gi_intri_el['id']:
                gi_el_flag[i] = 0
        gv_el_flag = np.array(gv_el_flag)
        gi_el_flag = np.array(gi_el_flag)
        # Get the mean signal-to-noise ratio of each emission line within the given width (in index)
        gv_el_snr = emulator.get_el_snr(gv_snr,gv_el_flag,gv_detect_el,gv_init_idx,gv_elwidth)
        gi_el_snr = emulator.get_el_snr(gi_snr,gi_el_flag,gi_detect_el,gi_init_idx,gi_elwidth)

    # saveing data 
    parameters_desi_original = np.hstack([raall.reshape((len(f['ID'][:]),1)),decall.reshape((len(f['ID'][:]),1)),zall.reshape((len(f['ID'][:]),1)),
                                maggall.reshape((len(f['ID'][:]),1)),magrall.reshape((len(f['ID'][:]),1)),magzall.reshape((len(f['ID'][:]),1)),
                                ntotal,retotal,patotal,artotal,m_starall.reshape((len(f['ID'][:]),1))])
    
    # change the parameters_desi dtype into float64
    try:
        parameters_desi = np.array(parameters_desi_original, dtype=np.float64)
    except:
        print('parameters_desi type error')

    if apply_el_detect == True and apply_wavelength_calibration_error == True:
        parameters_grism = np.hstack([gu_rms_in_e,gv_rms_in_e,gi_rms_in_e,
                                    gu_snr_mean,gv_snr_mean,gi_snr_mean,
                                    gu_wave_off,gv_wave_off,gi_wave_off,
                                    gv_el_flag.reshape((len(f['ID'][:]),1)),gi_el_flag.reshape((len(f['ID'][:]),1))])
    elif apply_el_detect == False and apply_wavelength_calibration_error == True:
        parameters_grism = np.hstack([gu_rms_in_e,gv_rms_in_e,gi_rms_in_e,
                                    gu_snr_mean,gv_snr_mean,gi_snr_mean,
                                    gu_wave_off,gv_wave_off,gi_wave_off])
    elif apply_el_detect == True and apply_wavelength_calibration_error == False:
        parameters_grism = np.hstack([gu_rms_in_e,gv_rms_in_e,gi_rms_in_e,
                                   gu_snr_mean,gv_snr_mean,gi_snr_mean,
                                   gv_el_flag.reshape((len(f['ID'][:]),1)),gi_el_flag.reshape((len(f['ID'][:]),1))])
    elif apply_el_detect == False and apply_wavelength_calibration_error == False:
        parameters_grism = np.hstack([gu_rms_in_e,gv_rms_in_e,gi_rms_in_e,
                                   gu_snr_mean,gv_snr_mean,gi_snr_mean])
        
    if apply_photometric == True:
        parameters_phot = np.stack([mag_nuv_total,mag_u_total,mag_g_total,mag_r_total,mag_i_total,mag_z_total,mag_y_total,
                                    magerr_nuv_total,magerr_u_total,magerr_g_total,magerr_r_total,magerr_i_total,magerr_z_total,magerr_y_total,
                                    snr_nuv_total,snr_u_total,snr_g_total,snr_r_total,snr_i_total,snr_z_total,snr_y_total])
    
    with h5py.File("CSST_grism"+'_'+str(len(f['ID'][:]))+'_'+hdf5filename, "w") as file:

        dataset1 = file.create_dataset('ID', data = f['ID'][:])
        dataset2 = file.create_dataset('parameters_desi', data = parameters_desi)
        dataset2.attrs['name'] = ['RA','Dec','z_best',
                                'MAG_G','MAG_R','MAG_Z',
                                'n','Re','PA','baratio','str_mass']
        
        if apply_el_detect == True and apply_wavelength_calibration_error == True:
            dataset99 = file.create_dataset('parameters_grism', data = parameters_grism)
            dataset99.attrs['name'] = ['gu_rms_in_e','gv_rms_in_e','gi_rms_in_e',
                                    'gu_snr_mean','gv_snr_mean','gi_snr_mean',
                                    'gu_wave_off','gv_wave_off','gi_wave_off',
                                    'gv_el_flag','gi_el_flag']
        elif apply_el_detect == False and apply_wavelength_calibration_error == True:
            dataset99 = file.create_dataset('parameters_grism', data = parameters_grism)
            dataset99.attrs['name'] = ['gu_rms_in_e','gv_rms_in_e','gi_rms_in_e',
                                    'gu_snr_mean','gv_snr_mean','gi_snr_mean',
                                    'gu_wave_off','gv_wave_off','gi_wave_off']
        elif apply_el_detect == True and apply_wavelength_calibration_error == False:
            dataset99 = file.create_dataset('parameters_grism', data = parameters_grism)
            dataset99.attrs['name'] = ['gu_rms_in_e','gv_rms_in_e','gi_rms_in_e',
                                    'gu_snr_mean','gv_snr_mean','gi_snr_mean',
                                    'gv_el_flag','gi_el_flag']
        elif apply_el_detect == False and apply_wavelength_calibration_error == False:
            dataset99 = file.create_dataset('parameters_grism', data = parameters_grism)
            dataset99.attrs['name'] = ['gu_rms_in_e','gv_rms_in_e','gi_rms_in_e',
                                    'gu_snr_mean','gv_snr_mean','gi_snr_mean']
            
        if apply_photometric == True:
            dataset98 = file.create_dataset('parameters_phot', data = parameters_phot)
            dataset98.attrs['name'] = ['mag_nuv','mag_u','mag_g','mag_r','mag_i','mag_z','mag_y',
                                       'magerr_nuv','magerr_u','magerr_g','magerr_r','magerr_i','magerr_z','magerr_y',
                                       'snr_nuv','snr_u','snr_g','snr_r','snr_i','snr_z','snr_y']
        
        # data mask 
        dataset97 = file.create_dataset('data_mask', data = data_mask)

        # create GU, GV, and GI three h5py.group
        group_gu = file.create_group('GU')
        group_gv = file.create_group('GV')
        group_gi = file.create_group('GI')
        # wavelength
        dataset3 = group_gu.create_dataset('wave', data = guwave)
        dataset4 = group_gv.create_dataset('wave', data = gvwave)
        dataset5 = group_gi.create_dataset('wave', data = giwave)
        # flux in electrons
        dataset6 = group_gu.create_dataset('flux_elec', data = padded_data_dict['padded_gu_elec_per_column'])
        dataset7 = group_gv.create_dataset('flux_elec', data = padded_data_dict['padded_gv_elec_per_column'])
        dataset8 = group_gi.create_dataset('flux_elec', data = padded_data_dict['padded_gi_elec_per_column'])
        # intrinsic flux in ujy
        dataset9 = group_gu.create_dataset('flux_ujy', data = gu_ujy)
        dataset10 = group_gv.create_dataset('flux_ujy', data = gv_ujy)
        dataset11 = group_gi.create_dataset('flux_ujy', data = gi_ujy) 
        # # flux with noise in ujy
        dataset12 = group_gu.create_dataset('flux_ujy_with_noise', data = padded_data_dict['padded_gu_ujy_with_noise'])
        dataset13 = group_gv.create_dataset('flux_ujy_with_noise', data = padded_data_dict['padded_gv_ujy_with_noise'])
        dataset14 = group_gi.create_dataset('flux_ujy_with_noise', data = padded_data_dict['padded_gi_ujy_with_noise'])
        # ferr
        dataset15 = group_gu.create_dataset('ferr', data = padded_data_dict['padded_gu_ferr'])
        dataset16 = group_gv.create_dataset('ferr', data = padded_data_dict['padded_gv_ferr'])
        dataset17 = group_gi.create_dataset('ferr', data = padded_data_dict['padded_gi_ferr'])
        #  snr
        dataset18 = group_gu.create_dataset('snr', data = padded_data_dict['padded_gu_snr'])
        dataset19 = group_gv.create_dataset('snr', data = padded_data_dict['padded_gv_snr'])
        dataset20 = group_gi.create_dataset('snr', data = padded_data_dict['padded_gi_snr'])
        # spec mask
        dataset21 = group_gu.create_dataset('spec_mask', data = padded_data_dict['padded_gu_spec_mask'])
        dataset22 = group_gv.create_dataset('spec_mask', data = padded_data_dict['padded_gv_spec_mask'])
        dataset23 = group_gi.create_dataset('spec_mask', data = padded_data_dict['padded_gi_spec_mask'])
        # snr mask 
        dataset24 = group_gu.create_dataset('snr_mask', data = padded_data_dict['padded_gu_snr_mask'])
        dataset25 = group_gv.create_dataset('snr_mask', data = padded_data_dict['padded_gv_snr_mask'])
        dataset26 = group_gi.create_dataset('snr_mask', data = padded_data_dict['padded_gi_snr_mask'])

        if apply_el_detect == True:
            dt1 = h5py.special_dtype(vlen=np.dtype('float64'))
            dt2 = h5py.special_dtype(vlen=np.dtype('int64'))
            # store intrinsic emission line results for gv
            dataset27 = group_gv.create_dataset('intri_el_id', data = gv_intri_el['id'])
            gv_intri_wave_dset = group_gv.create_dataset('intri_el_wave', (len(gv_intri_el),), dtype=dt1)
            gv_intri_wave_dset[:] = gv_intri_el['wave']
            gv_intri_idx_dset = group_gv.create_dataset('intri_el_idx', (len(gv_intri_el),), dtype=dt2)
            gv_intri_idx_dset[:] = gv_intri_el['idx']
            dataset28 = group_gv.create_dataset('intri_el_elnumber', data = gv_intri_el['intri_el_number'])
            # store intrinsic emission line results for gi
            dataset29 = group_gi.create_dataset('intri_el_id', data = gi_intri_el['id'])
            gi_intri_wave_dset = group_gi.create_dataset('intri_el_wave', (len(gi_intri_el),), dtype=dt1)
            gi_intri_wave_dset[:] = gi_intri_el['wave']
            gi_intri_idx_dset = group_gi.create_dataset('intri_el_idx', (len(gi_intri_el),), dtype=dt2)
            gi_intri_idx_dset[:] = gi_intri_el['idx']
            dataset30 = group_gi.create_dataset('intri_el_elnumber', data = gi_intri_el['intri_el_number'])
            # store detected emission line results for gv
            dataset31 = group_gv.create_dataset('detect_el_id', data = gv_detect_el['id'])
            gv_detect_wave_dset = group_gv.create_dataset('detect_el_wave', (len(gv_detect_el),), dtype=dt1)
            gv_detect_wave_dset[:] = gv_detect_el['wave']
            gv_detect_idx_dset = group_gv.create_dataset('detect_el_idx', (len(gv_detect_el),), dtype=dt2)
            gv_detect_idx_dset[:] = gv_detect_el['idx']
            gv_el_snr_dset = group_gv.create_dataset('detect_el_snr',  (len(gv_detect_el),), dtype=dt1)
            gv_el_snr_dset[:] = gv_el_snr
            dataset32 = group_gv.create_dataset('detect_el_elnumber', data = gv_detect_el['detect_el_number'])
            # store detected emission line results for gi
            dataset33 = group_gi.create_dataset('detect_el_id', data = gi_detect_el['id'])
            gi_detect_wave_dset = group_gi.create_dataset('detect_el_wave', (len(gi_detect_el),), dtype=dt1)
            gi_detect_wave_dset[:] = gi_detect_el['wave']
            gi_detect_idx_dset = group_gi.create_dataset('detect_el_idx', (len(gi_detect_el),), dtype=dt2)
            gi_detect_idx_dset[:] = gi_detect_el['idx']
            gi_el_snr_dset = group_gi.create_dataset('detect_el_snr',  (len(gi_detect_el),), dtype=dt1)
            gi_el_snr_dset[:] = gi_el_snr
            dataset34 = group_gi.create_dataset('detect_el_elnumber', data = gi_detect_el['detect_el_number'])

    file.close()
    toc = time.perf_counter()
    print(f"Finished, total time costs {toc - tic:0.4f} seconds for "+desipath+hdf5filename)

hdf5filenames = [
'seedcat2_0420_1194_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1195_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1196_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1197_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1198_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_119_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_11_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1200_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1201_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1202_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1203_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1205_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1206_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1207_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1208_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1209_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_120_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1210_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1211_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1212_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1213_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1214_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1215_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1216_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1217_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1218_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1219_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_121_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1220_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1221_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1222_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1223_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1224_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1225_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1226_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1227_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1228_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1229_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_122_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1230_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1231_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1232_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1233_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1234_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1235_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1236_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1237_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1238_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1239_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_123_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1240_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1241_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1242_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1243_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1244_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1245_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1246_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1247_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1248_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1249_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_124_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1250_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1251_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1252_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1253_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1254_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1255_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1256_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1257_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1258_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1259_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_125_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1260_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1261_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1262_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1263_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1264_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1265_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1266_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1267_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1268_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1269_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_126_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1270_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1271_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1272_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1273_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1274_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1275_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1276_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1277_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1278_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1279_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_127_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1280_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1281_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1282_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1283_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1284_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1285_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1286_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1287_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1288_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1289_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_128_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1290_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1291_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1292_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1293_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1294_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1295_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1296_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1297_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1298_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1299_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_129_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_12_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1300_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1301_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1302_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1303_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1304_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1305_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1306_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1307_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1308_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1309_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_130_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1310_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1311_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1312_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1313_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1314_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1315_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1316_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1317_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1318_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1319_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_131_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1320_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1321_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1322_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1323_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1324_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1325_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1326_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1327_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1328_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1329_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_132_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1330_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1331_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1332_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1333_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1334_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1335_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1336_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1337_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1338_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1339_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_133_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1340_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1342_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1343_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1344_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1345_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1346_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1347_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1348_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1349_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_134_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1350_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1351_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1352_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1353_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1354_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1355_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1356_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1357_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1358_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1359_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_135_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1360_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1361_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1362_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1363_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1364_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1365_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1366_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1367_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1368_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1369_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_136_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1370_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1371_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1372_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1373_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1374_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1375_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1376_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1377_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1378_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1379_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_137_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1380_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1381_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1382_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_138_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_139_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_13_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_140_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_141_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_142_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_143_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_144_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_145_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_146_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_147_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_148_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_149_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_14_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_150_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_151_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_152_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_153_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_154_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_155_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_156_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_157_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_158_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_159_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_15_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_160_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_161_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_162_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_163_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_164_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_165_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_166_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_167_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_168_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_169_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_16_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1194_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1195_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1196_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1197_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1198_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1199_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_119_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_11_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1200_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1201_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1202_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1203_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1204_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1205_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1206_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1207_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1208_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1209_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1210_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1211_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1212_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1213_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1214_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1215_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1216_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1217_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1218_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1219_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_121_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1220_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1221_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1222_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1223_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1224_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1225_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1226_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1227_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1228_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1229_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_122_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1230_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1231_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1232_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1233_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1234_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1235_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1236_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1237_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1238_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1239_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1240_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1241_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1242_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1243_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1244_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1245_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1246_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1247_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1248_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1249_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_124_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1250_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1251_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1252_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1253_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1254_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1255_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1256_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1257_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1258_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1259_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_125_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1260_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1261_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1262_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1263_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1264_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1265_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1266_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1267_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1268_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1269_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_126_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1270_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1271_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1272_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1273_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1274_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1275_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1276_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1277_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1278_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1279_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_127_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1280_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1281_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1282_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1283_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1284_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1285_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1286_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1287_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1288_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1289_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_128_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1290_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1291_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1292_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1293_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1294_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1295_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1296_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1297_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1298_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1299_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_129_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_12_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1300_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1301_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1302_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1303_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1304_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1305_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1306_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1307_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1308_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1309_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_130_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1310_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1311_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1312_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1313_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1314_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1315_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1316_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1317_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1318_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1319_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_131_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1320_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1321_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1322_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1323_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1324_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1325_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1326_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1327_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1328_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1329_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_132_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1330_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1331_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1332_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1333_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1334_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1335_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1336_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1337_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1338_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1339_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_133_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1340_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1341_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1342_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1343_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1344_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1345_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1346_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1347_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1348_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1349_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_134_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1351_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1352_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1353_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1354_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1355_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1356_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1357_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1358_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1359_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_135_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1360_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1361_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1362_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1363_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1364_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1365_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1366_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1367_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1368_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1369_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_136_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1370_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1371_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1372_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1373_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1374_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1375_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1376_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1377_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1378_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1379_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_137_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1380_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1381_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_1382_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_138_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_139_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_13_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_140_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_141_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_142_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_143_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_144_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_145_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_146_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_147_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_148_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_149_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_14_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_150_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_151_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_152_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_153_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_154_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_155_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_156_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_157_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_158_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_159_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_15_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_160_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_161_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_162_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_163_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_164_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_165_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_166_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_167_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_168_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_169_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_16_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
'seedcat2_0420_170_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
]

if __name__ == '__main__':
    process_list = []
    for hdf5filename in hdf5filenames[begin:end]:
        p = Process(target=task,args=(desipath,hdf5filename)) #实例化进程对象
        p.start()
        process_list.append(p)

    for i in process_list:
        p.join()

    print('This part of processes is END!')
