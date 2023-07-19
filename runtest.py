#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@Env     		:   grizli
@File    		:   ~/emulator/emulator_v0.7/run.py
@Time    		:   2023/06/21 09:26:17
@Author  		:   Run Wen
@Version 		:   0.7
@Contact 		:   wenrun@pmo.ac.cn
@Description	:   The multiprocess file for determining the 2-D profile for sources, running main functions and 

Change log:
1.  Separate the run.py file into utils.py and morphology.py with each functions in them.
2.  Add an emisslion line identify modoule to detect emission lines with specutils, the emission lines are detected in both intrinsic 
    and noisy spectra to test the identify success. By comparing the emission lines detected in two spectra, we are able to show the 
    emission line identify success in noisy spectra. The emission lines information from intrinsic spectra and noisy spectra are 
    stored in datasets while the emission line detect flag in parameters for GV and GI band (since no el in GU at z ~ 0-1).
'''

import numpy as np
import pickle
from tqdm import tqdm
import h5py
from multiprocessing import Process
from scipy import interpolate
from astropy.table import Table
import time
from astropy.cosmology import FlatLambdaCDM
import json
hubble=70
cosmo = FlatLambdaCDM(H0=hubble,Om0=0.3)

desipath = '/Users/rain/emulator/emulator_v0.7/'
bkg = Table.read('/Users/rain/emulator/bkg_spec.fits')

with open('emulator_parameters.json', 'r') as f:
    emulator_parameters = json.load(f)

gu_wave_mid = emulator_parameters['gu_wave_mid']
gv_wave_mid = emulator_parameters['gv_wave_mid']
gi_wave_mid = emulator_parameters['gi_wave_mid']
gv_init_idx = emulator_parameters['gv_init_idx']
gv_end_idx = emulator_parameters['gv_end_idx']
gi_init_idx = emulator_parameters['gi_init_idx']
gi_end_idx = emulator_parameters['gi_end_idx']
noisy_el_detect_th = emulator_parameters['noisy_el_detect_th']
intri_el_detect_th = emulator_parameters['intri_el_detect_th']
el_detect_success_th = emulator_parameters['el_detect_success_th']
gv_elwidth = emulator_parameters['gv_elwidth']
gi_elwidth = emulator_parameters['gi_elwidth']

with open('widthlib_20x20x10x10.pkl', 'rb') as f:
    widthlib = pickle.load(f)
    
with open('heightlib_20x20x10x10.pkl', 'rb') as f:
    heightlib = pickle.load(f)

nseries = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1. ,1.1, 1.2, 1.3, 1.4, 
                    1.5, 1.6, 1.8, 2. ,2.5, 3., 3.5, 4., 4.5, 5.]) # 20
reseries = np.round(np.array([0.3, 0.5, 0.7, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 
                              2.5, 3, 3.5,  4.5, 5, 5.5, 6, 6.5, 7, 7.4])/0.074) # 20
paseries = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90]) # 10
baseries = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) # 10

def task(desipath,hdf5filename):
    import utils
    import morphology 
    from main import emulator

    tic = time.perf_counter()

    print('Start generating CSST intrinsic spectra of {0}:\n'.format(hdf5filename))

    # read the throughput curve from fits files, length ~ 1600+
    gu_1st_tp = Table.read('/Users/rain/emulator/GU.Throughput.1st.fits')
    gv_1st_tp = Table.read('/Users/rain/emulator/GV.Throughput.1st.fits')
    gi_1st_tp = Table.read('/Users/rain/emulator/GI.Throughput.1st.fits')
    # create the transmission curve function to prepare for interpolation
    gu_tpf = interpolate.interp1d(gu_1st_tp['WAVELENGTH'],gu_1st_tp['SENSITIVITY'])
    gv_tpf = interpolate.interp1d(gv_1st_tp['WAVELENGTH'],gv_1st_tp['SENSITIVITY'])
    gi_tpf = interpolate.interp1d(gi_1st_tp['WAVELENGTH'],gi_1st_tp['SENSITIVITY'])
    # create the sky background curve function to prepare for interpolation
    bkgf = interpolate.interp1d(bkg['wavelength'], bkg['fnu'])

    emulator = emulator(SED_file_path = desipath, file_name = hdf5filename)

    f = emulator.read_hdf5()
    # generate intrinsic CSST simulated slitless spectra data, in pixels of length ~ 500+
    guwave, gu_ujy = emulator.kernels_convolve(f['best_fit']['wavelength_rest'][:]*1e4,
                                               f['best_fit']['spec_csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8[0,1]'], 
                                               f['best_fit']['z'], 
                                               'GU')
    gvwave, gv_ujy = emulator.kernels_convolve(f['best_fit']['wavelength_rest'][:]*1e4,
                                               f['best_fit']['spec_csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8[0,1]'], 
                                               f['best_fit']['z'], 
                                               'GV')
    giwave, gi_ujy = emulator.kernels_convolve(f['best_fit']['wavelength_rest'][:]*1e4,
                                               f['best_fit']['spec_csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8[0,1]'], 
                                               f['best_fit']['z'], 
                                               'GI')
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
    gimag = utils.fnu2mAB(gi_fnu[:,(np.abs(giwave-gi_wave_mid)).argmin()])
    print(gimag)
    print(gimag.min())
    print(gimag.max())
    # create empty morphology parameters lists
    ntotal = []
    retotal = []
    patotal = []
    artotal = []
    # create empty 2d profile distribution functions lists
    width_total = []
    height_total = []
    # create 7 empty lists of the add_noise function for 3 CSST grism bands
    gu_elec_per_column = []
    gu_fnu_with_noise = []
    gu_efnu = []
    gu_rms_in_e = []
    gu_snr = []
    gu_snr_mean = []
    gu_wave_off = []

    gv_elec_per_column = []
    gv_fnu_with_noise = []
    gv_efnu = []
    gv_rms_in_e = []
    gv_snr = []
    gv_snr_mean = []
    gv_wave_off = []

    gi_elec_per_column = []
    gi_fnu_with_noise = []
    gi_efnu = []
    gi_rms_in_e = []
    gi_snr = []
    gi_snr_mean = []
    gi_wave_off = []

    print('Start simulating CSST observed spectra of {0}:\n'.format(hdf5filename))

    for i in tqdm(range(len(f['ID'][:]))):
        # get parameters from hdf5 file, total parameter names and the way to find the idx see the end of this file
        ra = f['parameters'][i,14]
        dec = f['parameters'][i,15]
        m_star = f['parameters'][i,115]
        z = f['parameters'][i,31]
        # generate 2d parameters
        n,re,pa,ar = morphology.get_2d_param(f['parameters'][i,8], 16, 21, 0.5, 5, m_star, z)
        # get 2d profile distribution function
        width = morphology.match_input_paramters(widthlib, n,re,pa,ar,nseries,reseries,paseries,baseries)
        height = morphology.match_input_paramters(heightlib, n,re,pa,ar,nseries,reseries,paseries,baseries)
        # get simulated observed spectrum with noise added
        gu_elec_per_column_single, gu_fnu_with_noise_single, gu_fnu_error_single, gu_rms_in_e_single, gu_snr_single, gu_snr_mean_single, gu_wave_off_single = emulator.add_noise(ra,dec,guwave,gu_fnu[i],guwave,gu_bkg_fnu,gutp,height,width)
        gv_elec_per_column_single, gv_fnu_with_noise_single, gv_fnu_error_single, gv_rms_in_e_single, gv_snr_single, gv_snr_mean_single, gv_wave_off_single = emulator.add_noise(ra,dec,gvwave,gv_fnu[i],gvwave,gv_bkg_fnu,gvtp,height,width)
        gi_elec_per_column_single, gi_fnu_with_noise_single, gi_fnu_error_single, gi_rms_in_e_single, gi_snr_single, gi_snr_mean_single, gi_wave_off_single = emulator.add_noise(ra,dec,giwave,gi_fnu[i],giwave,gi_bkg_fnu,gitp,height,width)
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
        gu_wave_off.append(gu_wave_off_single)

        gv_elec_per_column.append(gv_elec_per_column_single)
        gv_fnu_with_noise.append(gv_fnu_with_noise_single)
        gv_efnu.append(gv_fnu_error_single)
        gv_rms_in_e.append(gv_rms_in_e_single)
        gv_snr.append(gv_snr_single)
        gv_snr_mean.append(gv_snr_mean_single)
        gv_wave_off.append(gv_wave_off_single)

        gi_elec_per_column.append(gi_elec_per_column_single)
        gi_fnu_with_noise.append(gi_fnu_with_noise_single)
        gi_efnu.append(gi_fnu_error_single)
        gi_rms_in_e.append(gi_rms_in_e_single)
        gi_snr.append(gi_snr_single)
        gi_snr_mean.append(gi_snr_mean_single)
        gi_wave_off.append(gi_wave_off_single)
    # array the results and reshape them to be stored in the hdf5 file and parameters table
    ntotal = np.array(ntotal).reshape((len(f['ID']),1))
    retotal = np.array(retotal).reshape((len(f['ID']),1))
    patotal = np.array(patotal).reshape((len(f['ID']),1))
    artotal = np.array(artotal).reshape((len(f['ID']),1))
    
    gu_elec_per_column = np.array(gu_elec_per_column)
    gu_ujy_with_noise = np.array(gu_fnu_with_noise)*1e29
    gu_ferr = np.array(gu_efnu)*1e29
    gu_rms_in_e = np.array(gu_rms_in_e).reshape((len(f['ID']),1))
    gu_snr = np.array(gu_snr)
    gu_snr_mean = np.array(gu_snr_mean).reshape((len(f['ID']),1))
    gu_wave_off = np.array(gu_wave_off).reshape((len(f['ID']),1))

    gv_elec_per_column = np.array(gv_elec_per_column)
    gv_ujy_with_noise = np.array(gv_fnu_with_noise)*1e29
    gv_ferr = np.array(gv_efnu)*1e29
    gv_rms_in_e = np.array(gv_rms_in_e).reshape((len(f['ID']),1))
    gv_snr = np.array(gv_snr)
    gv_snr_mean = np.array(gv_snr_mean).reshape((len(f['ID']),1))
    gv_wave_off = np.array(gv_wave_off).reshape((len(f['ID']),1))

    gi_elec_per_column = np.array(gi_elec_per_column)
    gi_ujy_with_noise = np.array(gi_fnu_with_noise)*1e29
    gi_ferr = np.array(gi_efnu)*1e29
    gi_rms_in_e = np.array(gi_rms_in_e).reshape((len(f['ID']),1))
    gi_snr = np.array(gi_snr)
    gi_snr_mean = np.array(gi_snr_mean).reshape((len(f['ID']),1))
    gi_wave_off = np.array(gi_wave_off).reshape((len(f['ID']),1))

    print('Start detecting emission lines of {0}:\n'.format(hdf5filename))
    # initialize the Qtable list
    gv_noisy_lines = []
    gv_intrinsic_lines = []
    gi_noisy_lines = []
    gi_intrinsic_lines = []
    # detect emission lines in both intrinsic and noisy spectra in GV and GI band
    for i in tqdm(range(len(f['ID'][:]))):
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
    new_parameters = np.hstack([f['parameters'][:,14].reshape((len(f['ID']),1)),f['parameters'][:,15].reshape((len(f['ID']),1)),f['parameters'][:,31].reshape((len(f['ID']),1)),
                                f['parameters'][:,4].reshape((len(f['ID']),1)),f['parameters'][:,6].reshape((len(f['ID']),1)),f['parameters'][:,8].reshape((len(f['ID']),1)),
                                ntotal,retotal,patotal,artotal,f['parameters'][:,108].reshape((len(f['ID']),1)),
                                gu_rms_in_e,gv_rms_in_e,gi_rms_in_e,
                                gu_snr_mean,gv_snr_mean,gi_snr_mean,
                                gu_wave_off,gv_wave_off,gi_wave_off,
                                gv_el_flag.reshape((len(f['ID']),1)),gi_el_flag.reshape((len(f['ID']),1))])

    with h5py.File("CSST_grism"+'_'+str(len(f['ID'][:]))+'_'+hdf5filename, "w") as file:

        dataset1 = file.create_dataset('ID', data = f['ID'])
        dataset2 = file.create_dataset('parameters', data = new_parameters)
        dataset2.attrs['name'] = ['RA','Dec','z_best',
                                'MAG_G','MAG_R','MAG_Z',
                                'n','Re','PA','baratio','str_mass',
                                'gu_rms_in_e','gv_rms_in_e','gi_rms_in_e',
                                'gu_snr_mean','gv_snr_mean','gi_snr_mean',
                                'gu_wave_off','gv_wave_off','gi_wave_off',
                                'gv_el_flag','gi_el_flag']
        # 创建三个组
        group_gu = file.create_group('GU')
        group_gv = file.create_group('GV')
        group_gi = file.create_group('GI')
        # 各波段波长
        dataset3 = group_gu.create_dataset('wave', data = guwave)
        dataset4 = group_gv.create_dataset('wave', data = gvwave)
        dataset5 = group_gi.create_dataset('wave', data = giwave)
        # 
        dataset6 = group_gu.create_dataset('flux_elec', data = gu_elec_per_column)
        dataset7 = group_gv.create_dataset('flux_elec', data = gv_elec_per_column)
        dataset8 = group_gi.create_dataset('flux_elec', data = gi_elec_per_column)
        # 各波段按grism分辨率卷积后的原始谱线
        dataset9 = group_gu.create_dataset('flux_ujy', data = gu_ujy)
        dataset10 = group_gv.create_dataset('flux_ujy', data = gv_ujy)
        dataset11 = group_gi.create_dataset('flux_ujy', data = gi_ujy)
        # 各波段添加噪声后的光谱数据
        dataset12 = group_gu.create_dataset('flux_ujy_with_noise', data = gu_ujy_with_noise)
        dataset13 = group_gv.create_dataset('flux_ujy_with_noise', data = gv_ujy_with_noise)
        dataset14 = group_gi.create_dataset('flux_ujy_with_noise', data = gi_ujy_with_noise)
        # 各波段流量的误差
        dataset15 = group_gu.create_dataset('ferr', data = gu_ferr)
        dataset16 = group_gv.create_dataset('ferr', data = gv_ferr)
        dataset17 = group_gi.create_dataset('ferr', data = gi_ferr)
        #  信噪比曲线
        dataset18 = group_gu.create_dataset('snr', data = gu_snr)
        dataset19 = group_gv.create_dataset('snr', data = gv_snr)
        dataset20 = group_gi.create_dataset('snr', data = gi_snr)
        
        dt1 = h5py.special_dtype(vlen=np.dtype('float64'))
        dt2 = h5py.special_dtype(vlen=np.dtype('int64'))
        # store intrinsic emission line results for gv
        dataset21 = group_gv.create_dataset('intri_el_id', data = gv_intri_el['id'])
        gv_intri_wave_dset = group_gv.create_dataset('intri_el_wave', (len(gv_intri_el),), dtype=dt1)
        gv_intri_wave_dset[:] = gv_intri_el['wave']
        gv_intri_idx_dset = group_gv.create_dataset('intri_el_idx', (len(gv_intri_el),), dtype=dt2)
        gv_intri_idx_dset[:] = gv_intri_el['idx']
        dataset22 = group_gv.create_dataset('intri_el_elnumber', data = gv_intri_el['intri_el_number'])
        # store intrinsic emission line results for gi
        dataset23 = group_gi.create_dataset('intri_el_id', data = gi_intri_el['id'])
        gi_intri_wave_dset = group_gi.create_dataset('intri_el_wave', (len(gi_intri_el),), dtype=dt1)
        gi_intri_wave_dset[:] = gi_intri_el['wave']
        gi_intri_idx_dset = group_gi.create_dataset('intri_el_idx', (len(gi_intri_el),), dtype=dt2)
        gi_intri_idx_dset[:] = gi_intri_el['idx']
        dataset24 = group_gi.create_dataset('intri_el_elnumber', data = gi_intri_el['intri_el_number'])
        # store detected emission line results for gv
        dataset25 = group_gv.create_dataset('detect_el_id', data = gv_detect_el['id'])
        gv_detect_wave_dset = group_gv.create_dataset('detect_el_wave', (len(gv_detect_el),), dtype=dt1)
        gv_detect_wave_dset[:] = gv_detect_el['wave']
        gv_detect_idx_dset = group_gv.create_dataset('detect_el_idx', (len(gv_detect_el),), dtype=dt2)
        gv_detect_idx_dset[:] = gv_detect_el['idx']
        gv_el_snr_dset = group_gv.create_dataset('detect_el_snr',  (len(gv_detect_el),), dtype=dt1)
        gv_el_snr_dset[:] = gv_el_snr
        dataset26 = group_gv.create_dataset('detect_el_elnumber', data = gv_detect_el['detect_el_number'])
        # store detected emission line results for gi
        dataset27 = group_gi.create_dataset('detect_el_id', data = gi_detect_el['id'])
        gi_detect_wave_dset = group_gi.create_dataset('detect_el_wave', (len(gi_detect_el),), dtype=dt1)
        gi_detect_wave_dset[:] = gi_detect_el['wave']
        gi_detect_idx_dset = group_gi.create_dataset('detect_el_idx', (len(gi_detect_el),), dtype=dt2)
        gi_detect_idx_dset[:] = gi_detect_el['idx']
        gi_el_snr_dset = group_gi.create_dataset('detect_el_snr',  (len(gi_detect_el),), dtype=dt1)
        gi_el_snr_dset[:] = gi_el_snr
        dataset28 = group_gi.create_dataset('detect_el_elnumber', data = gi_detect_el['detect_el_number'])

    file.close()
    toc = time.perf_counter()
    print(f"Finished, total time costs {toc - tic:0.4f} seconds for "+desipath+hdf5filename)

hdf5filenames = [
    'seedcat2_0702_0_MzLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5',
    'seedcat2_0702_0_DECaLS_0csp_sfh200_bc2003_hr_stelib_chab_neb_300r_i0100_2dal8_10.hdf5']
if __name__ == '__main__':
    process_list = []
    for hdf5filename in hdf5filenames:
        p = Process(target=task,args=('/Users/rain/emulator/seedcat_0702/',hdf5filename)) #实例化进程对象
        p.start()
        process_list.append(p)

    for i in process_list:
        p.join()

    print('This part of processes is END!')

"""
>>> list(file['parameters_name'][:]).index(b'z_{MAL}')
    31

>>> f['parameters_name'][:]
    array([b'z_min', b'z_max', b'd/Mpc', b'E(B-V)', b'MAG_G', b'MAG_G_ERR',
       b'MAG_R', b'MAG_R_ERR', b'MAG_Z', b'MAG_Z_ERR', b'MAG_W1',
       b'MAG_W1_ERR', b'MAG_W2', b'MAG_W2_ERR', b'RA', b'DEC', b'TYPE',
       b'SERSIC', b'SERSIC_ERR', b'SHAPE_R', b'SHAPE_R_ERR', b'SHAPE_E1',
       b'SHAPE_E1_ERR', b'SHAPE_E2', b'SHAPE_E2_ERR',
       b'vd_{eml}[km/s][0,1]', b'f[0,1]', b'sys_err0', b'sys_err1',
       b'z_{mean}', b'z_{sigma}', b'z_{MAL}', b'z_{MAP}', b'z_{median}',
       b'z_{0.16}', b'z_{0.84}', b'log(age/yr)[0,1]_{mean}',
       b'log(age/yr)[0,1]_{sigma}', b'log(age/yr)[0,1]_{MAL}',
       b'log(age/yr)[0,1]_{MAP}', b'log(age/yr)[0,1]_{median}',
       b'log(age/yr)[0,1]_{0.16}', b'log(age/yr)[0,1]_{0.84}',
       b'log(tau/yr)[0,1]_{mean}', b'log(tau/yr)[0,1]_{sigma}',
       b'log(tau/yr)[0,1]_{MAL}', b'log(tau/yr)[0,1]_{MAP}',
       b'log(tau/yr)[0,1]_{median}', b'log(tau/yr)[0,1]_{0.16}',
       b'log(tau/yr)[0,1]_{0.84}', b'log(Z/Zsun)[0,1]_{mean}',
       b'log(Z/Zsun)[0,1]_{sigma}', b'log(Z/Zsun)[0,1]_{MAL}',
       b'log(Z/Zsun)[0,1]_{MAP}', b'log(Z/Zsun)[0,1]_{median}',
       b'log(Z/Zsun)[0,1]_{0.16}', b'log(Z/Zsun)[0,1]_{0.84}',
       b'Av_{2dal8}[0,1]_{mean}', b'Av_{2dal8}[0,1]_{sigma}',
       b'Av_{2dal8}[0,1]_{MAL}', b'Av_{2dal8}[0,1]_{MAP}',
       b'Av_{2dal8}[0,1]_{median}', b'Av_{2dal8}[0,1]_{0.16}',
       b'Av_{2dal8}[0,1]_{0.84}', b'log(scale)[0,1]_{mean}',
       b'log(scale)[0,1]_{sigma}', b'log(scale)[0,1]_{MAL}',
       b'log(scale)[0,1]_{MAP}', b'log(scale)[0,1]_{median}',
       b'log(scale)[0,1]_{0.16}', b'log(scale)[0,1]_{0.84}',
       b'ageU(zform)/Gyr[0,1]_{mean}', b'ageU(zform)/Gyr[0,1]_{sigma}',
       b'ageU(zform)/Gyr[0,1]_{MAL}', b'ageU(zform)/Gyr[0,1]_{MAP}',
       b'ageU(zform)/Gyr[0,1]_{median}', b'ageU(zform)/Gyr[0,1]_{0.16}',
       b'ageU(zform)/Gyr[0,1]_{0.84}', b'log(ageMW/yr)[0,1]_{mean}',
       b'log(ageMW/yr)[0,1]_{sigma}', b'log(ageMW/yr)[0,1]_{MAL}',
       b'log(ageMW/yr)[0,1]_{MAP}', b'log(ageMW/yr)[0,1]_{median}',
       b'log(ageMW/yr)[0,1]_{0.16}', b'log(ageMW/yr)[0,1]_{0.84}',
       b'log(Zt/Zsun)[0,1]_{mean}', b'log(Zt/Zsun)[0,1]_{sigma}',
       b'log(Zt/Zsun)[0,1]_{MAL}', b'log(Zt/Zsun)[0,1]_{MAP}',
       b'log(Zt/Zsun)[0,1]_{median}', b'log(Zt/Zsun)[0,1]_{0.16}',
       b'log(Zt/Zsun)[0,1]_{0.84}',
       b'log(SFH(0)/[M_{sun}/yr])[0,1]_{mean}',
       b'log(SFH(0)/[M_{sun}/yr])[0,1]_{sigma}',
       b'log(SFH(0)/[M_{sun}/yr])[0,1]_{MAL}',
       b'log(SFH(0)/[M_{sun}/yr])[0,1]_{MAP}',
       b'log(SFH(0)/[M_{sun}/yr])[0,1]_{median}',
       b'log(SFH(0)/[M_{sun}/yr])[0,1]_{0.16}',
       b'log(SFH(0)/[M_{sun}/yr])[0,1]_{0.84}',
       b'log(SFR_100Myr/[M_{sun}/yr])[0,1]_{mean}',
       b'log(SFR_100Myr/[M_{sun}/yr])[0,1]_{sigma}',
       b'log(SFR_100Myr/[M_{sun}/yr])[0,1]_{MAL}',
       b'log(SFR_100Myr/[M_{sun}/yr])[0,1]_{MAP}',
       b'log(SFR_100Myr/[M_{sun}/yr])[0,1]_{median}',
       b'log(SFR_100Myr/[M_{sun}/yr])[0,1]_{0.16}',
       b'log(SFR_100Myr/[M_{sun}/yr])[0,1]_{0.84}',
       b'log(M*_formed)[0,1]_{mean}', b'log(M*_formed)[0,1]_{sigma}',
       b'log(M*_formed)[0,1]_{MAL}', b'log(M*_formed)[0,1]_{MAP}',
       b'log(M*_formed)[0,1]_{median}', b'log(M*_formed)[0,1]_{0.16}',
       b'log(M*_formed)[0,1]_{0.84}', b'log(M*)[0,1]_{mean}',
       b'log(M*)[0,1]_{sigma}', b'log(M*)[0,1]_{MAL}',
       b'log(M*)[0,1]_{MAP}', b'log(M*)[0,1]_{median}',
       b'log(M*)[0,1]_{0.16}', b'log(M*)[0,1]_{0.84}',
       b'log(M*_liv)[0,1]_{mean}', b'log(M*_liv)[0,1]_{sigma}',
       b'log(M*_liv)[0,1]_{MAL}', b'log(M*_liv)[0,1]_{MAP}',
       b'log(M*_liv)[0,1]_{median}', b'log(M*_liv)[0,1]_{0.16}',
       b'log(M*_liv)[0,1]_{0.84}', b'log(H1_1025.72)[0,1]_{mean}',
       b'log(H1_1025.72)[0,1]_{sigma}', b'log(H1_1025.72)[0,1]_{MAL}',
       b'log(H1_1025.72)[0,1]_{MAP}', b'log(H1_1025.72)[0,1]_{median}',
       b'log(H1_1025.72)[0,1]_{0.16}', b'log(H1_1025.72)[0,1]_{0.84}',
       b'log(BLND_1035.00)[0,1]_{mean}',
       b'log(BLND_1035.00)[0,1]_{sigma}', b'log(BLND_1035.00)[0,1]_{MAL}',
       b'log(BLND_1035.00)[0,1]_{MAP}',
       b'log(BLND_1035.00)[0,1]_{median}',
       b'log(BLND_1035.00)[0,1]_{0.16}', b'log(BLND_1035.00)[0,1]_{0.84}',
       b'log(H1_1215.67)[0,1]_{mean}', b'log(H1_1215.67)[0,1]_{sigma}',
       b'log(H1_1215.67)[0,1]_{MAL}', b'log(H1_1215.67)[0,1]_{MAP}',
       b'log(H1_1215.67)[0,1]_{median}', b'log(H1_1215.67)[0,1]_{0.16}',
       b'log(H1_1215.67)[0,1]_{0.84}', b'log(BLND_1240.00)[0,1]_{mean}',
       b'log(BLND_1240.00)[0,1]_{sigma}', b'log(BLND_1240.00)[0,1]_{MAL}',
       b'log(BLND_1240.00)[0,1]_{MAP}',
       b'log(BLND_1240.00)[0,1]_{median}',
       b'log(BLND_1240.00)[0,1]_{0.16}', b'log(BLND_1240.00)[0,1]_{0.84}',
       b'log(BLND_1397.00)[0,1]_{mean}',
       b'log(BLND_1397.00)[0,1]_{sigma}', b'log(BLND_1397.00)[0,1]_{MAL}',
       b'log(BLND_1397.00)[0,1]_{MAP}',
       b'log(BLND_1397.00)[0,1]_{median}',
       b'log(BLND_1397.00)[0,1]_{0.16}', b'log(BLND_1397.00)[0,1]_{0.84}',
       b'log(BLND_1406.00)[0,1]_{mean}',
       b'log(BLND_1406.00)[0,1]_{sigma}', b'log(BLND_1406.00)[0,1]_{MAL}',
       b'log(BLND_1406.00)[0,1]_{MAP}',
       b'log(BLND_1406.00)[0,1]_{median}',
       b'log(BLND_1406.00)[0,1]_{0.16}', b'log(BLND_1406.00)[0,1]_{0.84}',
       b'log(BLND_1486.00)[0,1]_{mean}',
       b'log(BLND_1486.00)[0,1]_{sigma}', b'log(BLND_1486.00)[0,1]_{MAL}',
       b'log(BLND_1486.00)[0,1]_{MAP}',
       b'log(BLND_1486.00)[0,1]_{median}',
       b'log(BLND_1486.00)[0,1]_{0.16}', b'log(BLND_1486.00)[0,1]_{0.84}',
       b'log(BLND_1549.00)[0,1]_{mean}',
       b'log(BLND_1549.00)[0,1]_{sigma}', b'log(BLND_1549.00)[0,1]_{MAL}',
       b'log(BLND_1549.00)[0,1]_{MAP}',
       b'log(BLND_1549.00)[0,1]_{median}',
       b'log(BLND_1549.00)[0,1]_{0.16}', b'log(BLND_1549.00)[0,1]_{0.84}',
       b'log(HE2_1640.43)[0,1]_{mean}', b'log(HE2_1640.43)[0,1]_{sigma}',
       b'log(HE2_1640.43)[0,1]_{MAL}', b'log(HE2_1640.43)[0,1]_{MAP}',
       b'log(HE2_1640.43)[0,1]_{median}', b'log(HE2_1640.43)[0,1]_{0.16}',
       b'log(HE2_1640.43)[0,1]_{0.84}', b'log(BLND_1750.00)[0,1]_{mean}',
       b'log(BLND_1750.00)[0,1]_{sigma}', b'log(BLND_1750.00)[0,1]_{MAL}',
       b'log(BLND_1750.00)[0,1]_{MAP}',
       b'log(BLND_1750.00)[0,1]_{median}',
       b'log(BLND_1750.00)[0,1]_{0.16}', b'log(BLND_1750.00)[0,1]_{0.84}',
       b'log(BLND_1860.00)[0,1]_{mean}',
       b'log(BLND_1860.00)[0,1]_{sigma}', b'log(BLND_1860.00)[0,1]_{MAL}',
       b'log(BLND_1860.00)[0,1]_{MAP}',
       b'log(BLND_1860.00)[0,1]_{median}',
       b'log(BLND_1860.00)[0,1]_{0.16}', b'log(BLND_1860.00)[0,1]_{0.84}',
       b'log(BLND_1909.00)[0,1]_{mean}',
       b'log(BLND_1909.00)[0,1]_{sigma}', b'log(BLND_1909.00)[0,1]_{MAL}',
       b'log(BLND_1909.00)[0,1]_{MAP}',
       b'log(BLND_1909.00)[0,1]_{median}',
       b'log(BLND_1909.00)[0,1]_{0.16}', b'log(BLND_1909.00)[0,1]_{0.84}',
       b'log(O3_2320.95)[0,1]_{mean}', b'log(O3_2320.95)[0,1]_{sigma}',
       b'log(O3_2320.95)[0,1]_{MAL}', b'log(O3_2320.95)[0,1]_{MAP}',
       b'log(O3_2320.95)[0,1]_{median}', b'log(O3_2320.95)[0,1]_{0.16}',
       b'log(O3_2320.95)[0,1]_{0.84}', b'log(BLND_2335.00)[0,1]_{mean}',
       b'log(BLND_2335.00)[0,1]_{sigma}', b'log(BLND_2335.00)[0,1]_{MAL}',
       b'log(BLND_2335.00)[0,1]_{MAP}',
       b'log(BLND_2335.00)[0,1]_{median}',
       b'log(BLND_2335.00)[0,1]_{0.16}', b'log(BLND_2335.00)[0,1]_{0.84}',
       b'log(BLND_2665.00)[0,1]_{mean}',
       b'log(BLND_2665.00)[0,1]_{sigma}', b'log(BLND_2665.00)[0,1]_{MAL}',
       b'log(BLND_2665.00)[0,1]_{MAP}',
       b'log(BLND_2665.00)[0,1]_{median}',
       b'log(BLND_2665.00)[0,1]_{0.16}', b'log(BLND_2665.00)[0,1]_{0.84}',
       b'log(BLND_2798.00)[0,1]_{mean}',
       b'log(BLND_2798.00)[0,1]_{sigma}', b'log(BLND_2798.00)[0,1]_{MAL}',
       b'log(BLND_2798.00)[0,1]_{MAP}',
       b'log(BLND_2798.00)[0,1]_{median}',
       b'log(BLND_2798.00)[0,1]_{0.16}', b'log(BLND_2798.00)[0,1]_{0.84}',
       b'log(NE5_3426.03)[0,1]_{mean}', b'log(NE5_3426.03)[0,1]_{sigma}',
       b'log(NE5_3426.03)[0,1]_{MAL}', b'log(NE5_3426.03)[0,1]_{MAP}',
       b'log(NE5_3426.03)[0,1]_{median}', b'log(NE5_3426.03)[0,1]_{0.16}',
       b'log(NE5_3426.03)[0,1]_{0.84}', b'log(O2_3726.03)[0,1]_{mean}',
       b'log(O2_3726.03)[0,1]_{sigma}', b'log(O2_3726.03)[0,1]_{MAL}',
       b'log(O2_3726.03)[0,1]_{MAP}', b'log(O2_3726.03)[0,1]_{median}',
       b'log(O2_3726.03)[0,1]_{0.16}', b'log(O2_3726.03)[0,1]_{0.84}',
       b'log(O2_3728.81)[0,1]_{mean}', b'log(O2_3728.81)[0,1]_{sigma}',
       b'log(O2_3728.81)[0,1]_{MAL}', b'log(O2_3728.81)[0,1]_{MAP}',
       b'log(O2_3728.81)[0,1]_{median}', b'log(O2_3728.81)[0,1]_{0.16}',
       b'log(O2_3728.81)[0,1]_{0.84}', b'log(NE3_3868.76)[0,1]_{mean}',
       b'log(NE3_3868.76)[0,1]_{sigma}', b'log(NE3_3868.76)[0,1]_{MAL}',
       b'log(NE3_3868.76)[0,1]_{MAP}', b'log(NE3_3868.76)[0,1]_{median}',
       b'log(NE3_3868.76)[0,1]_{0.16}', b'log(NE3_3868.76)[0,1]_{0.84}',
       b'log(HE1_3888.63)[0,1]_{mean}', b'log(HE1_3888.63)[0,1]_{sigma}',
       b'log(HE1_3888.63)[0,1]_{MAL}', b'log(HE1_3888.63)[0,1]_{MAP}',
       b'log(HE1_3888.63)[0,1]_{median}', b'log(HE1_3888.63)[0,1]_{0.16}',
       b'log(HE1_3888.63)[0,1]_{0.84}', b'log(CA2_3933.66)[0,1]_{mean}',
       b'log(CA2_3933.66)[0,1]_{sigma}', b'log(CA2_3933.66)[0,1]_{MAL}',
       b'log(CA2_3933.66)[0,1]_{MAP}', b'log(CA2_3933.66)[0,1]_{median}',
       b'log(CA2_3933.66)[0,1]_{0.16}', b'log(CA2_3933.66)[0,1]_{0.84}',
       b'log(S2_4068.60)[0,1]_{mean}', b'log(S2_4068.60)[0,1]_{sigma}',
       b'log(S2_4068.60)[0,1]_{MAL}', b'log(S2_4068.60)[0,1]_{MAP}',
       b'log(S2_4068.60)[0,1]_{median}', b'log(S2_4068.60)[0,1]_{0.16}',
       b'log(S2_4068.60)[0,1]_{0.84}', b'log(BLND_4074.00)[0,1]_{mean}',
       b'log(BLND_4074.00)[0,1]_{sigma}', b'log(BLND_4074.00)[0,1]_{MAL}',
       b'log(BLND_4074.00)[0,1]_{MAP}',
       b'log(BLND_4074.00)[0,1]_{median}',
       b'log(BLND_4074.00)[0,1]_{0.16}', b'log(BLND_4074.00)[0,1]_{0.84}',
       b'log(S2_4076.35)[0,1]_{mean}', b'log(S2_4076.35)[0,1]_{sigma}',
       b'log(S2_4076.35)[0,1]_{MAL}', b'log(S2_4076.35)[0,1]_{MAP}',
       b'log(S2_4076.35)[0,1]_{median}', b'log(S2_4076.35)[0,1]_{0.16}',
       b'log(S2_4076.35)[0,1]_{0.84}', b'log(H1_4101.73)[0,1]_{mean}',
       b'log(H1_4101.73)[0,1]_{sigma}', b'log(H1_4101.73)[0,1]_{MAL}',
       b'log(H1_4101.73)[0,1]_{MAP}', b'log(H1_4101.73)[0,1]_{median}',
       b'log(H1_4101.73)[0,1]_{0.16}', b'log(H1_4101.73)[0,1]_{0.84}',
       b'log(H1_4340.46)[0,1]_{mean}', b'log(H1_4340.46)[0,1]_{sigma}',
       b'log(H1_4340.46)[0,1]_{MAL}', b'log(H1_4340.46)[0,1]_{MAP}',
       b'log(H1_4340.46)[0,1]_{median}', b'log(H1_4340.46)[0,1]_{0.16}',
       b'log(H1_4340.46)[0,1]_{0.84}', b'log(HE2_4685.64)[0,1]_{mean}',
       b'log(HE2_4685.64)[0,1]_{sigma}', b'log(HE2_4685.64)[0,1]_{MAL}',
       b'log(HE2_4685.64)[0,1]_{MAP}', b'log(HE2_4685.64)[0,1]_{median}',
       b'log(HE2_4685.64)[0,1]_{0.16}', b'log(HE2_4685.64)[0,1]_{0.84}',
       b'log(NE4_4724.17)[0,1]_{mean}', b'log(NE4_4724.17)[0,1]_{sigma}',
       b'log(NE4_4724.17)[0,1]_{MAL}', b'log(NE4_4724.17)[0,1]_{MAP}',
       b'log(NE4_4724.17)[0,1]_{median}', b'log(NE4_4724.17)[0,1]_{0.16}',
       b'log(NE4_4724.17)[0,1]_{0.84}', b'log(AR4_4740.12)[0,1]_{mean}',
       b'log(AR4_4740.12)[0,1]_{sigma}', b'log(AR4_4740.12)[0,1]_{MAL}',
       b'log(AR4_4740.12)[0,1]_{MAP}', b'log(AR4_4740.12)[0,1]_{median}',
       b'log(AR4_4740.12)[0,1]_{0.16}', b'log(AR4_4740.12)[0,1]_{0.84}',
       b'log(H1_4861.33)[0,1]_{mean}', b'log(H1_4861.33)[0,1]_{sigma}',
       b'log(H1_4861.33)[0,1]_{MAL}', b'log(H1_4861.33)[0,1]_{MAP}',
       b'log(H1_4861.33)[0,1]_{median}', b'log(H1_4861.33)[0,1]_{0.16}',
       b'log(H1_4861.33)[0,1]_{0.84}', b'log(O3_4958.91)[0,1]_{mean}',
       b'log(O3_4958.91)[0,1]_{sigma}', b'log(O3_4958.91)[0,1]_{MAL}',
       b'log(O3_4958.91)[0,1]_{MAP}', b'log(O3_4958.91)[0,1]_{median}',
       b'log(O3_4958.91)[0,1]_{0.16}', b'log(O3_4958.91)[0,1]_{0.84}',
       b'log(O3_5006.84)[0,1]_{mean}', b'log(O3_5006.84)[0,1]_{sigma}',
       b'log(O3_5006.84)[0,1]_{MAL}', b'log(O3_5006.84)[0,1]_{MAP}',
       b'log(O3_5006.84)[0,1]_{median}', b'log(O3_5006.84)[0,1]_{0.16}',
       b'log(O3_5006.84)[0,1]_{0.84}', b'log(N1_5200.26)[0,1]_{mean}',
       b'log(N1_5200.26)[0,1]_{sigma}', b'log(N1_5200.26)[0,1]_{MAL}',
       b'log(N1_5200.26)[0,1]_{MAP}', b'log(N1_5200.26)[0,1]_{median}',
       b'log(N1_5200.26)[0,1]_{0.16}', b'log(N1_5200.26)[0,1]_{0.84}',
       b'log(O1_5577.34)[0,1]_{mean}', b'log(O1_5577.34)[0,1]_{sigma}',
       b'log(O1_5577.34)[0,1]_{MAL}', b'log(O1_5577.34)[0,1]_{MAP}',
       b'log(O1_5577.34)[0,1]_{median}', b'log(O1_5577.34)[0,1]_{0.16}',
       b'log(O1_5577.34)[0,1]_{0.84}', b'log(N2_5754.61)[0,1]_{mean}',
       b'log(N2_5754.61)[0,1]_{sigma}', b'log(N2_5754.61)[0,1]_{MAL}',
       b'log(N2_5754.61)[0,1]_{MAP}', b'log(N2_5754.61)[0,1]_{median}',
       b'log(N2_5754.61)[0,1]_{0.16}', b'log(N2_5754.61)[0,1]_{0.84}',
       b'log(HE1_5875.64)[0,1]_{mean}', b'log(HE1_5875.64)[0,1]_{sigma}',
       b'log(HE1_5875.64)[0,1]_{MAL}', b'log(HE1_5875.64)[0,1]_{MAP}',
       b'log(HE1_5875.64)[0,1]_{median}', b'log(HE1_5875.64)[0,1]_{0.16}',
       b'log(HE1_5875.64)[0,1]_{0.84}', b'log(BLND_6300.00)[0,1]_{mean}',
       b'log(BLND_6300.00)[0,1]_{sigma}', b'log(BLND_6300.00)[0,1]_{MAL}',
       b'log(BLND_6300.00)[0,1]_{MAP}',
       b'log(BLND_6300.00)[0,1]_{median}',
       b'log(BLND_6300.00)[0,1]_{0.16}', b'log(BLND_6300.00)[0,1]_{0.84}',
       b'log(O1_6363.78)[0,1]_{mean}', b'log(O1_6363.78)[0,1]_{sigma}',
       b'log(O1_6363.78)[0,1]_{MAL}', b'log(O1_6363.78)[0,1]_{MAP}',
       b'log(O1_6363.78)[0,1]_{median}', b'log(O1_6363.78)[0,1]_{0.16}',
       b'log(O1_6363.78)[0,1]_{0.84}', b'log(H1_6562.81)[0,1]_{mean}',
       b'log(H1_6562.81)[0,1]_{sigma}', b'log(H1_6562.81)[0,1]_{MAL}',
       b'log(H1_6562.81)[0,1]_{MAP}', b'log(H1_6562.81)[0,1]_{median}',
       b'log(H1_6562.81)[0,1]_{0.16}', b'log(H1_6562.81)[0,1]_{0.84}',
       b'log(N2_6583.45)[0,1]_{mean}', b'log(N2_6583.45)[0,1]_{sigma}',
       b'log(N2_6583.45)[0,1]_{MAL}', b'log(N2_6583.45)[0,1]_{MAP}',
       b'log(N2_6583.45)[0,1]_{median}', b'log(N2_6583.45)[0,1]_{0.16}',
       b'log(N2_6583.45)[0,1]_{0.84}', b'log(S2_6716.44)[0,1]_{mean}',
       b'log(S2_6716.44)[0,1]_{sigma}', b'log(S2_6716.44)[0,1]_{MAL}',
       b'log(S2_6716.44)[0,1]_{MAP}', b'log(S2_6716.44)[0,1]_{median}',
       b'log(S2_6716.44)[0,1]_{0.16}', b'log(S2_6716.44)[0,1]_{0.84}',
       b'log(BLND_6720.00)[0,1]_{mean}',
       b'log(BLND_6720.00)[0,1]_{sigma}', b'log(BLND_6720.00)[0,1]_{MAL}',
       b'log(BLND_6720.00)[0,1]_{MAP}',
       b'log(BLND_6720.00)[0,1]_{median}',
       b'log(BLND_6720.00)[0,1]_{0.16}', b'log(BLND_6720.00)[0,1]_{0.84}',
       b'log(S2_6730.82)[0,1]_{mean}', b'log(S2_6730.82)[0,1]_{sigma}',
       b'log(S2_6730.82)[0,1]_{MAL}', b'log(S2_6730.82)[0,1]_{MAP}',
       b'log(S2_6730.82)[0,1]_{median}', b'log(S2_6730.82)[0,1]_{0.16}',
       b'log(S2_6730.82)[0,1]_{0.84}', b'log(AR5_7005.83)[0,1]_{mean}',
       b'log(AR5_7005.83)[0,1]_{sigma}', b'log(AR5_7005.83)[0,1]_{MAL}',
       b'log(AR5_7005.83)[0,1]_{MAP}', b'log(AR5_7005.83)[0,1]_{median}',
       b'log(AR5_7005.83)[0,1]_{0.16}', b'log(AR5_7005.83)[0,1]_{0.84}',
       b'log(AR3_7135.79)[0,1]_{mean}', b'log(AR3_7135.79)[0,1]_{sigma}',
       b'log(AR3_7135.79)[0,1]_{MAL}', b'log(AR3_7135.79)[0,1]_{MAP}',
       b'log(AR3_7135.79)[0,1]_{median}', b'log(AR3_7135.79)[0,1]_{0.16}',
       b'log(AR3_7135.79)[0,1]_{0.84}', b'log(AR4_7332.15)[0,1]_{mean}',
       b'log(AR4_7332.15)[0,1]_{sigma}', b'log(AR4_7332.15)[0,1]_{MAL}',
       b'log(AR4_7332.15)[0,1]_{MAP}', b'log(AR4_7332.15)[0,1]_{median}',
       b'log(AR4_7332.15)[0,1]_{0.16}', b'log(AR4_7332.15)[0,1]_{0.84}',
       b'log(AR3_7751.11)[0,1]_{mean}', b'log(AR3_7751.11)[0,1]_{sigma}',
       b'log(AR3_7751.11)[0,1]_{MAL}', b'log(AR3_7751.11)[0,1]_{MAP}',
       b'log(AR3_7751.11)[0,1]_{median}', b'log(AR3_7751.11)[0,1]_{0.16}',
       b'log(AR3_7751.11)[0,1]_{0.84}', b'log(BLND_8446.00)[0,1]_{mean}',
       b'log(BLND_8446.00)[0,1]_{sigma}', b'log(BLND_8446.00)[0,1]_{MAL}',
       b'log(BLND_8446.00)[0,1]_{MAP}',
       b'log(BLND_8446.00)[0,1]_{median}',
       b'log(BLND_8446.00)[0,1]_{0.16}', b'log(BLND_8446.00)[0,1]_{0.84}',
       b'log(CA2_8498.02)[0,1]_{mean}', b'log(CA2_8498.02)[0,1]_{sigma}',
       b'log(CA2_8498.02)[0,1]_{MAL}', b'log(CA2_8498.02)[0,1]_{MAP}',
       b'log(CA2_8498.02)[0,1]_{median}', b'log(CA2_8498.02)[0,1]_{0.16}',
       b'log(CA2_8498.02)[0,1]_{0.84}', b'log(CA2_8542.09)[0,1]_{mean}',
       b'log(CA2_8542.09)[0,1]_{sigma}', b'log(CA2_8542.09)[0,1]_{MAL}',
       b'log(CA2_8542.09)[0,1]_{MAP}', b'log(CA2_8542.09)[0,1]_{median}',
       b'log(CA2_8542.09)[0,1]_{0.16}', b'log(CA2_8542.09)[0,1]_{0.84}',
       b'log(CA2_8662.14)[0,1]_{mean}', b'log(CA2_8662.14)[0,1]_{sigma}',
       b'log(CA2_8662.14)[0,1]_{MAL}', b'log(CA2_8662.14)[0,1]_{MAP}',
       b'log(CA2_8662.14)[0,1]_{median}', b'log(CA2_8662.14)[0,1]_{0.16}',
       b'log(CA2_8662.14)[0,1]_{0.84}', b'log(S3_9068.62)[0,1]_{mean}',
       b'log(S3_9068.62)[0,1]_{sigma}', b'log(S3_9068.62)[0,1]_{MAL}',
       b'log(S3_9068.62)[0,1]_{MAP}', b'log(S3_9068.62)[0,1]_{median}',
       b'log(S3_9068.62)[0,1]_{0.16}', b'log(S3_9068.62)[0,1]_{0.84}',
       b'log(S3_9530.62)[0,1]_{mean}', b'log(S3_9530.62)[0,1]_{sigma}',
       b'log(S3_9530.62)[0,1]_{MAL}', b'log(S3_9530.62)[0,1]_{MAP}',
       b'log(S3_9530.62)[0,1]_{median}', b'log(S3_9530.62)[0,1]_{0.16}',
       b'log(S3_9530.62)[0,1]_{0.84}', b'log(H1_9545.93)[0,1]_{mean}',
       b'log(H1_9545.93)[0,1]_{sigma}', b'log(H1_9545.93)[0,1]_{MAL}',
       b'log(H1_9545.93)[0,1]_{MAP}', b'log(H1_9545.93)[0,1]_{median}',
       b'log(H1_9545.93)[0,1]_{0.16}', b'log(H1_9545.93)[0,1]_{0.84}',
       b'log(HE1_10830.30)[0,1]_{mean}',
       b'log(HE1_10830.30)[0,1]_{sigma}', b'log(HE1_10830.30)[0,1]_{MAL}',
       b'log(HE1_10830.30)[0,1]_{MAP}',
       b'log(HE1_10830.30)[0,1]_{median}',
       b'log(HE1_10830.30)[0,1]_{0.16}', b'log(HE1_10830.30)[0,1]_{0.84}',
       b'log(H1_18751.00)[0,1]_{mean}', b'log(H1_18751.00)[0,1]_{sigma}',
       b'log(H1_18751.00)[0,1]_{MAL}', b'log(H1_18751.00)[0,1]_{MAP}',
       b'log(H1_18751.00)[0,1]_{median}', b'log(H1_18751.00)[0,1]_{0.16}',
       b'log(H1_18751.00)[0,1]_{0.84}', b'log(H1_12818.00)[0,1]_{mean}',
       b'log(H1_12818.00)[0,1]_{sigma}', b'log(H1_12818.00)[0,1]_{MAL}',
       b'log(H1_12818.00)[0,1]_{MAP}', b'log(H1_12818.00)[0,1]_{median}',
       b'log(H1_12818.00)[0,1]_{0.16}', b'log(H1_12818.00)[0,1]_{0.84}',
       b'log(H1_10938.00)[0,1]_{mean}', b'log(H1_10938.00)[0,1]_{sigma}',
       b'log(H1_10938.00)[0,1]_{MAL}', b'log(H1_10938.00)[0,1]_{MAP}',
       b'log(H1_10938.00)[0,1]_{median}', b'log(H1_10938.00)[0,1]_{0.16}',
       b'log(H1_10938.00)[0,1]_{0.84}', b'log(H1_10049.30)[0,1]_{mean}',
       b'log(H1_10049.30)[0,1]_{sigma}', b'log(H1_10049.30)[0,1]_{MAL}',
       b'log(H1_10049.30)[0,1]_{MAP}', b'log(H1_10049.30)[0,1]_{median}',
       b'log(H1_10049.30)[0,1]_{0.16}', b'log(H1_10049.30)[0,1]_{0.84}',
       b'log(H1_40511.30)[0,1]_{mean}', b'log(H1_40511.30)[0,1]_{sigma}',
       b'log(H1_40511.30)[0,1]_{MAL}', b'log(H1_40511.30)[0,1]_{MAP}',
       b'log(H1_40511.30)[0,1]_{median}', b'log(H1_40511.30)[0,1]_{0.16}',
       b'log(H1_40511.30)[0,1]_{0.84}', b'log(H1_26251.30)[0,1]_{mean}',
       b'log(H1_26251.30)[0,1]_{sigma}', b'log(H1_26251.30)[0,1]_{MAL}',
       b'log(H1_26251.30)[0,1]_{MAP}', b'log(H1_26251.30)[0,1]_{median}',
       b'log(H1_26251.30)[0,1]_{0.16}', b'log(H1_26251.30)[0,1]_{0.84}',
       b'log(H1_21655.10)[0,1]_{mean}', b'log(H1_21655.10)[0,1]_{sigma}',
       b'log(H1_21655.10)[0,1]_{MAL}', b'log(H1_21655.10)[0,1]_{MAP}',
       b'log(H1_21655.10)[0,1]_{median}', b'log(H1_21655.10)[0,1]_{0.16}',
       b'log(H1_21655.10)[0,1]_{0.84}', b'log(H1_19445.40)[0,1]_{mean}',
       b'log(H1_19445.40)[0,1]_{sigma}', b'log(H1_19445.40)[0,1]_{MAL}',
       b'log(H1_19445.40)[0,1]_{MAP}', b'log(H1_19445.40)[0,1]_{median}',
       b'log(H1_19445.40)[0,1]_{0.16}', b'log(H1_19445.40)[0,1]_{0.84}',
       b'log(NE6_76431.80)[0,1]_{mean}',
       b'log(NE6_76431.80)[0,1]_{sigma}', b'log(NE6_76431.80)[0,1]_{MAL}',
       b'log(NE6_76431.80)[0,1]_{MAP}',
       b'log(NE6_76431.80)[0,1]_{median}',
       b'log(NE6_76431.80)[0,1]_{0.16}', b'log(NE6_76431.80)[0,1]_{0.84}',
       b'log(NA3_73170.60)[0,1]_{mean}',
       b'log(NA3_73170.60)[0,1]_{sigma}', b'log(NA3_73170.60)[0,1]_{MAL}',
       b'log(NA3_73170.60)[0,1]_{MAP}',
       b'log(NA3_73170.60)[0,1]_{median}',
       b'log(NA3_73170.60)[0,1]_{0.16}', b'log(NA3_73170.60)[0,1]_{0.84}',
       b'log(NA4_90309.80)[0,1]_{mean}',
       b'log(NA4_90309.80)[0,1]_{sigma}', b'log(NA4_90309.80)[0,1]_{MAL}',
       b'log(NA4_90309.80)[0,1]_{MAP}',
       b'log(NA4_90309.80)[0,1]_{median}',
       b'log(NA4_90309.80)[0,1]_{0.16}', b'log(NA4_90309.80)[0,1]_{0.84}',
       b'log(NA6_86183.60)[0,1]_{mean}',
       b'log(NA6_86183.60)[0,1]_{sigma}', b'log(NA6_86183.60)[0,1]_{MAL}',
       b'log(NA6_86183.60)[0,1]_{MAP}',
       b'log(NA6_86183.60)[0,1]_{median}',
       b'log(NA6_86183.60)[0,1]_{0.16}', b'log(NA6_86183.60)[0,1]_{0.84}',
       b'log(MG4_44871.20)[0,1]_{mean}',
       b'log(MG4_44871.20)[0,1]_{sigma}', b'log(MG4_44871.20)[0,1]_{MAL}',
       b'log(MG4_44871.20)[0,1]_{MAP}',
       b'log(MG4_44871.20)[0,1]_{median}',
       b'log(MG4_44871.20)[0,1]_{0.16}', b'log(MG4_44871.20)[0,1]_{0.84}',
       b'log(MG5_56070.00)[0,1]_{mean}',
       b'log(MG5_56070.00)[0,1]_{sigma}', b'log(MG5_56070.00)[0,1]_{MAL}',
       b'log(MG5_56070.00)[0,1]_{MAP}',
       b'log(MG5_56070.00)[0,1]_{median}',
       b'log(MG5_56070.00)[0,1]_{0.16}', b'log(MG5_56070.00)[0,1]_{0.84}',
       b'log(AL5_29045.00)[0,1]_{mean}',
       b'log(AL5_29045.00)[0,1]_{sigma}', b'log(AL5_29045.00)[0,1]_{MAL}',
       b'log(AL5_29045.00)[0,1]_{MAP}',
       b'log(AL5_29045.00)[0,1]_{median}',
       b'log(AL5_29045.00)[0,1]_{0.16}', b'log(AL5_29045.00)[0,1]_{0.84}',
       b'log(AL6_36593.20)[0,1]_{mean}',
       b'log(AL6_36593.20)[0,1]_{sigma}', b'log(AL6_36593.20)[0,1]_{MAL}',
       b'log(AL6_36593.20)[0,1]_{MAP}',
       b'log(AL6_36593.20)[0,1]_{median}',
       b'log(AL6_36593.20)[0,1]_{0.16}', b'log(AL6_36593.20)[0,1]_{0.84}',
       b'log(AL6_91132.90)[0,1]_{mean}',
       b'log(AL6_91132.90)[0,1]_{sigma}', b'log(AL6_91132.90)[0,1]_{MAL}',
       b'log(AL6_91132.90)[0,1]_{MAP}',
       b'log(AL6_91132.90)[0,1]_{median}',
       b'log(AL6_91132.90)[0,1]_{0.16}', b'log(AL6_91132.90)[0,1]_{0.84}',
       b'log(SI6_19624.70)[0,1]_{mean}',
       b'log(SI6_19624.70)[0,1]_{sigma}', b'log(SI6_19624.70)[0,1]_{MAL}',
       b'log(SI6_19624.70)[0,1]_{MAP}',
       b'log(SI6_19624.70)[0,1]_{median}',
       b'log(SI6_19624.70)[0,1]_{0.16}', b'log(SI6_19624.70)[0,1]_{0.84}',
       b'log(S8_9914.00)[0,1]_{mean}', b'log(S8_9914.00)[0,1]_{sigma}',
       b'log(S8_9914.00)[0,1]_{MAL}', b'log(S8_9914.00)[0,1]_{MAP}',
       b'log(S8_9914.00)[0,1]_{median}', b'log(S8_9914.00)[0,1]_{0.16}',
       b'log(S8_9914.00)[0,1]_{0.84}', b'log(AR2_69833.70)[0,1]_{mean}',
       b'log(AR2_69833.70)[0,1]_{sigma}', b'log(AR2_69833.70)[0,1]_{MAL}',
       b'log(AR2_69833.70)[0,1]_{MAP}',
       b'log(AR2_69833.70)[0,1]_{median}',
       b'log(AR2_69833.70)[0,1]_{0.16}', b'log(AR2_69833.70)[0,1]_{0.84}',
       b'log(AR3_89889.80)[0,1]_{mean}',
       b'log(AR3_89889.80)[0,1]_{sigma}', b'log(AR3_89889.80)[0,1]_{MAL}',
       b'log(AR3_89889.80)[0,1]_{MAP}',
       b'log(AR3_89889.80)[0,1]_{median}',
       b'log(AR3_89889.80)[0,1]_{0.16}', b'log(AR3_89889.80)[0,1]_{0.84}',
       b'log(AR5_78997.10)[0,1]_{mean}',
       b'log(AR5_78997.10)[0,1]_{sigma}', b'log(AR5_78997.10)[0,1]_{MAL}',
       b'log(AR5_78997.10)[0,1]_{MAP}',
       b'log(AR5_78997.10)[0,1]_{median}',
       b'log(AR5_78997.10)[0,1]_{0.16}', b'log(AR5_78997.10)[0,1]_{0.84}',
       b'log(AR6_45280.00)[0,1]_{mean}',
       b'log(AR6_45280.00)[0,1]_{sigma}', b'log(AR6_45280.00)[0,1]_{MAL}',
       b'log(AR6_45280.00)[0,1]_{MAP}',
       b'log(AR6_45280.00)[0,1]_{median}',
       b'log(AR6_45280.00)[0,1]_{0.16}', b'log(AR6_45280.00)[0,1]_{0.84}',
       b'log(CA4_32061.00)[0,1]_{mean}',
       b'log(CA4_32061.00)[0,1]_{sigma}', b'log(CA4_32061.00)[0,1]_{MAL}',
       b'log(CA4_32061.00)[0,1]_{MAP}',
       b'log(CA4_32061.00)[0,1]_{median}',
       b'log(CA4_32061.00)[0,1]_{0.16}', b'log(CA4_32061.00)[0,1]_{0.84}',
       b'log(CA5_41573.90)[0,1]_{mean}',
       b'log(CA5_41573.90)[0,1]_{sigma}', b'log(CA5_41573.90)[0,1]_{MAL}',
       b'log(CA5_41573.90)[0,1]_{MAP}',
       b'log(CA5_41573.90)[0,1]_{median}',
       b'log(CA5_41573.90)[0,1]_{0.16}', b'log(CA5_41573.90)[0,1]_{0.84}',
       b'log(CA8_23211.70)[0,1]_{mean}',
       b'log(CA8_23211.70)[0,1]_{sigma}', b'log(CA8_23211.70)[0,1]_{MAL}',
       b'log(CA8_23211.70)[0,1]_{MAP}',
       b'log(CA8_23211.70)[0,1]_{median}',
       b'log(CA8_23211.70)[0,1]_{0.16}', b'log(CA8_23211.70)[0,1]_{0.84}',
       b'log(SC5_23111.90)[0,1]_{mean}',
       b'log(SC5_23111.90)[0,1]_{sigma}', b'log(SC5_23111.90)[0,1]_{MAL}',
       b'log(SC5_23111.90)[0,1]_{MAP}',
       b'log(SC5_23111.90)[0,1]_{median}',
       b'log(SC5_23111.90)[0,1]_{0.16}', b'log(SC5_23111.90)[0,1]_{0.84}',
       b'log(TI6_17150.90)[0,1]_{mean}',
       b'log(TI6_17150.90)[0,1]_{sigma}', b'log(TI6_17150.90)[0,1]_{MAL}',
       b'log(TI6_17150.90)[0,1]_{MAP}',
       b'log(TI6_17150.90)[0,1]_{median}',
       b'log(TI6_17150.90)[0,1]_{0.16}', b'log(TI6_17150.90)[0,1]_{0.84}',
       b'log(V7_13037.60)[0,1]_{mean}', b'log(V7_13037.60)[0,1]_{sigma}',
       b'log(V7_13037.60)[0,1]_{MAL}', b'log(V7_13037.60)[0,1]_{MAP}',
       b'log(V7_13037.60)[0,1]_{median}', b'log(V7_13037.60)[0,1]_{0.16}',
       b'log(V7_13037.60)[0,1]_{0.84}', b'log(CR8_10106.50)[0,1]_{mean}',
       b'log(CR8_10106.50)[0,1]_{sigma}', b'log(CR8_10106.50)[0,1]_{MAL}',
       b'log(CR8_10106.50)[0,1]_{MAP}',
       b'log(CR8_10106.50)[0,1]_{median}',
       b'log(CR8_10106.50)[0,1]_{0.16}', b'log(CR8_10106.50)[0,1]_{0.84}',
       b'log(MN9_7968.49)[0,1]_{mean}', b'log(MN9_7968.49)[0,1]_{sigma}',
       b'log(MN9_7968.49)[0,1]_{MAL}', b'log(MN9_7968.49)[0,1]_{MAP}',
       b'log(MN9_7968.49)[0,1]_{median}', b'log(MN9_7968.49)[0,1]_{0.16}',
       b'log(MN9_7968.49)[0,1]_{0.84}', b'log(O3_1660.81)[0,1]_{mean}',
       b'log(O3_1660.81)[0,1]_{sigma}', b'log(O3_1660.81)[0,1]_{MAL}',
       b'log(O3_1660.81)[0,1]_{MAP}', b'log(O3_1660.81)[0,1]_{median}',
       b'log(O3_1660.81)[0,1]_{0.16}', b'log(O3_1660.81)[0,1]_{0.84}',
       b'log(O3_1666.15)[0,1]_{mean}', b'log(O3_1666.15)[0,1]_{sigma}',
       b'log(O3_1666.15)[0,1]_{MAL}', b'log(O3_1666.15)[0,1]_{MAP}',
       b'log(O3_1666.15)[0,1]_{median}', b'log(O3_1666.15)[0,1]_{0.16}',
       b'log(O3_1666.15)[0,1]_{0.84}', b'log(O5_1218.34)[0,1]_{mean}',
       b'log(O5_1218.34)[0,1]_{sigma}', b'log(O5_1218.34)[0,1]_{MAL}',
       b'log(O5_1218.34)[0,1]_{MAP}', b'log(O5_1218.34)[0,1]_{median}',
       b'log(O5_1218.34)[0,1]_{0.16}', b'log(O5_1218.34)[0,1]_{0.84}',
       b'log(SI3_1892.03)[0,1]_{mean}', b'log(SI3_1892.03)[0,1]_{sigma}',
       b'log(SI3_1892.03)[0,1]_{MAL}', b'log(SI3_1892.03)[0,1]_{MAP}',
       b'log(SI3_1892.03)[0,1]_{median}', b'log(SI3_1892.03)[0,1]_{0.16}',
       b'log(SI3_1892.03)[0,1]_{0.84}',
       b'log(L_{unabsorbed}/[erg/s])[0,1]_{mean}',
       b'log(L_{unabsorbed}/[erg/s])[0,1]_{sigma}',
       b'log(L_{unabsorbed}/[erg/s])[0,1]_{MAL}',
       b'log(L_{unabsorbed}/[erg/s])[0,1]_{MAP}',
       b'log(L_{unabsorbed}/[erg/s])[0,1]_{median}',
       b'log(L_{unabsorbed}/[erg/s])[0,1]_{0.16}',
       b'log(L_{unabsorbed}/[erg/s])[0,1]_{0.84}',
       b'log(L_{absorbed}/[erg/s])[0,1]_{mean}',
       b'log(L_{absorbed}/[erg/s])[0,1]_{sigma}',
       b'log(L_{absorbed}/[erg/s])[0,1]_{MAL}',
       b'log(L_{absorbed}/[erg/s])[0,1]_{MAP}',
       b'log(L_{absorbed}/[erg/s])[0,1]_{median}',
       b'log(L_{absorbed}/[erg/s])[0,1]_{0.16}',
       b'log(L_{absorbed}/[erg/s])[0,1]_{0.84}', b'logZ', b'INSlogZ',
       b'logZerr', b'Nd_phot', b'Nn_phot', b'Nd_spec', b'Nn_spec', b'SNR',
       b'Np', b'Ne', b'AIC', b'BIC', b'DIC', b'WAIC', b'Xmin^2/Nd',
       b'maxLogLike'], dtype=object)
"""