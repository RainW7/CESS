#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@Env     		:   grizli (Python 3.7.11) on Macbook Pro
@File    		:   ~/emulator/emulator_v0.8/demo.py
@Time    		:   2024/04/30 12:24:22
@Author  		:   Run Wen
@Version 		:   0.8
@Contact 		:   wenrun@pmo.ac.cn
@Description	:   create demo fig for simulated CSST slitless spectrum for the new DESI-phot + SDSS-spec data
'''

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
from astropy.table import Table

gbandlam = 0.4954320432589493
rbandlam = 0.6493723922982192
zbandlam = 0.9248643145995256

def uJy2mAB(flux): return -2.5*np.log10(flux)+23.9
def fnu2mAB(flux): return -2.5*np.log10(flux)-48.6
def mAB2uJy(mag): return np.power(10,(mag-23.9)/-2.5)
def mAB2fnu(mag): return np.power(10,(mag+48.6)/-2.5)
def dflux(mag_err,flux): return mag_err*flux/1.08574
def row2arr(x): return np.array(list(x))

desi_hdf5 = sys.argv[1]
hdf5 = h5py.File(desi_hdf5, 'r')

if sys.argv[2] == 'random':
    idx = np.random.randint(len(hdf5['ID']))
else: 
    idx = int(sys.argv[2])

wavelength_obs_center = np.array([gbandlam,rbandlam,zbandlam])
flux_obs = mAB2uJy(hdf5['parameters'][idx,[i for i in [3,4,5]]])

# 'ID', 'gi_efnu', 'gi_fnu', 'gi_fnu_with_noise', 'gi_wave', 'gu_efnu', 'gu_fnu', 
# 'gu_fnu_with_noise', 'gu_wave', 'gv_efnu', 'gv_fnu', 'gv_fnu_with_noise', 'gv_wave', 'parameters'

guwave = hdf5['GU']['wave'][:]
gu_fnu = hdf5['GU']['flux_ujy']
gu_fnu_with_noise = hdf5['GU']['flux_ujy_with_noise']
gu_efnu = hdf5['GU']['ferr']

gvwave = hdf5['GV']['wave'][:]
gv_fnu = hdf5['GV']['flux_ujy']
gv_fnu_with_noise = hdf5['GV']['flux_ujy_with_noise']
gv_efnu = hdf5['GV']['ferr']

giwave = hdf5['GI']['wave'][:]
gi_fnu = hdf5['GI']['flux_ujy']
gi_fnu_with_noise = hdf5['GI']['flux_ujy_with_noise']
gi_efnu = hdf5['GI']['ferr']

# parameters：
# ['RA', 'Dec', 'z_best', 'MAG_G', 'MAG_R', 'MAG_Z', 'n', 'Re', 'PA',
# 'baratio', 'str_mass', 'gu_rms_in_e', 'gv_rms_in_e', 'gi_rms_in_e',
# 'gu_snr_mean', 'gv_snr_mean', 'gi_snr_mean']

redshift = (1+hdf5['parameters'][idx,2])

fig = plt.figure(figsize=(14,8))

ax = fig.add_subplot(111)
ax.plot(guwave/1e4,gu_fnu[idx],c='k',label='GU SED at obs_wave',linewidth=1.5,zorder=6,alpha=1)
ax.plot(gvwave/1e4,gv_fnu[idx],c='k',label='GV SED at obs_wave',linewidth=1.5,zorder=7,alpha=1)
ax.plot(giwave/1e4,gi_fnu[idx],c='k',label='GI SED at obs_wave',linewidth=1.5,zorder=8,alpha=1)
# ax.errorbar(guwave/1e4, gu_fnu_with_noise[idx], yerr=gu_efnu[idx], marker='.', ecolor = 'dodgerblue',color='dodgerblue',ms=3,elinewidth=1,linestyle='None', alpha=0.8,label='GU observed flux',zorder=3)
# ax.errorbar(gvwave/1e4, gv_fnu_with_noise[idx], yerr=gv_efnu[idx], marker='.', ecolor = 'limegreen',color='limegreen',ms=3,elinewidth=1,linestyle='None', alpha=0.8,label='GV observed flux',zorder=4)
# ax.errorbar(giwave/1e4, gi_fnu_with_noise[idx], yerr=gi_efnu[idx], marker='.', ecolor = 'orangered',color='red',ms=3,elinewidth=1,linestyle='None', alpha=0.8,label='GI observed flux',zorder=5)
ax.plot(guwave/1e4, gu_fnu_with_noise[idx], color='dodgerblue',linestyle='-', alpha=0.8,label='GU observed flux',zorder=3)
ax.plot(gvwave/1e4, gv_fnu_with_noise[idx], color='limegreen',linestyle='-', alpha=0.8,label='GV observed flux',zorder=4)
ax.plot(giwave/1e4, gi_fnu_with_noise[idx], color='red',linestyle='-', alpha=0.8,label='GI observed flux',zorder=5)

ax.scatter(wavelength_obs_center,flux_obs,marker='^',color='black',s=80,label='DESI observed flux', alpha=0.7,zorder=11)

maxloc = gi_fnu[idx].max()
minloc = gu_fnu[idx].min()

if sys.argv[3] == 'emissionline':    
    
    plt.axvline(redshift*3729/1e4,linestyle='--',zorder=10)
    plt.text(redshift*3829/1e4,maxloc*5,s='[OII]',zorder=10)
    plt.axvline(redshift*4861/1e4,linestyle='--',zorder=10)
    plt.text(redshift*4561/1e4,maxloc*5,s='Hb',zorder=10)
    plt.axvline(redshift*5007/1e4,linestyle='--',zorder=10)
    plt.text(redshift*5107/1e4,maxloc*5,s='[OIII]',zorder=10)
    plt.axvline(redshift*6563/1e4,linestyle='--',zorder=10)
    plt.text(redshift*6263/1e4,maxloc*5,s='Ha',zorder=10)
    plt.axvline(redshift*6717/1e4,linestyle='--',zorder=10)
    plt.text(redshift*6817/1e4,maxloc*5,s='[SII]',zorder=10)
    plt.semilogy()
    plt.legend(loc='upper left')
    plt.xlim(0.2,1.1);plt.ylim(minloc/2,maxloc*10)
    plt.xlabel(r'observed wavelength [$\mu m$]')
    plt.ylabel(r'fnu [uJy]')
    #plt.title('fnu in grism resolution')
elif sys.argv[3] == 'none':
    plt.semilogy()
    plt.legend(loc='upper left')
    plt.xlim(0.2,1.1);plt.ylim(minloc/2,maxloc*10)
    plt.xlabel(r'wavelength [$\mu m$]')
    plt.ylabel(r'fnu [uJy]')
    # plt.title('fnu in grism resolution')

plt.suptitle('File: {0}, \n index = {1}, DESI_ID = {2}, (RA, DEC) = ({3}, {4}), z = {5}\n'
             'MAG_G = {6}, MAG_R = {7}, MAG_Z = {8}\n'
             'Sersic = {9}, Re = {10} pix, PA = {11} deg, b/a = {12}, mass = 10^{13} M$_\odot$\n'
             'GU_SNR = {14}, GV_SNR = {15}, GI_SNR = {16}'
            .format(hdf5.filename.split('/')[-1], idx+1, hdf5['ID'][idx], hdf5['parameters'][idx,0], hdf5['parameters'][idx,1], format(redshift-1,'.2f'),
                    hdf5['parameters'][idx,3], hdf5['parameters'][idx,4], hdf5['parameters'][idx,5],
                    format(hdf5['parameters'][idx,6],'.2f'), format(hdf5['parameters'][idx,7],'.2f'), format(hdf5['parameters'][idx,8],'.2f'), format(hdf5['parameters'][idx,9],'.2f'),format(hdf5['parameters'][idx,10],'.1f'),
                    format(hdf5['parameters'][idx,14],'.2f'),format(hdf5['parameters'][idx,15],'.2f'),format(hdf5['parameters'][idx,16],'.2f')),fontsize=9)
plt.tight_layout()
plt.savefig('{0}_{1}.png'.format(str(hdf5.filename.split('/')[-1]),idx+1),dpi=200)

print('The spectrum (ID = {0}) is shown!'.format(idx+1))
plt.show()