#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@Env     		:   grizli (Python 3.7.11) on Macbook Pro
@File    		:   ~/emulator/emulator_v0.8/main.py
@Time    		:   2023/08/14 12:25:18
@Author  		:   Run Wen
@Version 		:   0.8
@Contact 		:   wenrun@pmo.ac.cn
@Description	:   The main file of emulator containing functions for convolving intrinsic spectra, adding noise and detecting emission lines.

Change log:
1.  Setting an upper limit for electron spectra with 90000 e-.
2.  Add a mask array in the noise-adding function to mask the too noisy to be used data points, the mask for each source is universal.
3.  Change the wavelength calibration error from 1% to 0.1%-0.5%.
4.  For the valid source (e.g., with extremely high redshift which is not able to be simulated), emulator produces a empty
'''

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import os
import h5py
from astropy.convolution import Gaussian1DKernel,convolve
from astropy import units as u
from specutils.spectra import Spectrum1D
from specutils.fitting import find_lines_derivative
from specutils.fitting import fit_generic_continuum
import specutils
import warnings
import json
warnings.filterwarnings("ignore")
specutils.conf.do_continuum_function_check = False

import utils

with open('emulator_parameters.json', 'r') as f:
    emulator_parameters = json.load(f)

radi = emulator_parameters['radi'] # telescope radius, cm
expt = emulator_parameters['expt'] # exposure time, s
expnum = emulator_parameters['expnum'] # number
# gu_Res = emulator_parameters['guRes'] #resolution of grism
# gv_Res = emulator_parameters['gvRes']
# gi_Res = emulator_parameters['giRes']
arcsecperpix = emulator_parameters['arcsecperpix']# arcsec/pixel
dc_value = emulator_parameters['dc_value']

gu_start_wl = emulator_parameters['gu_start_wl']
gu_end_wl = emulator_parameters['gu_end_wl']
gv_start_wl = emulator_parameters['gv_start_wl']
gv_end_wl = emulator_parameters['gv_end_wl']
gi_start_wl = emulator_parameters['gi_start_wl']
gi_end_wl = emulator_parameters['gi_end_wl']
gu_min_wl = emulator_parameters['gu_min_wl']
gu_max_wl = emulator_parameters['gu_max_wl']
gv_min_wl = emulator_parameters['gv_min_wl']
gv_max_wl = emulator_parameters['gv_max_wl']
gi_min_wl = emulator_parameters['gi_min_wl']
gi_max_wl = emulator_parameters['gi_max_wl']

gu_dlambda = emulator_parameters['gu_dlambda']
gv_dlambda = emulator_parameters['gv_dlambda']
gi_dlambda = emulator_parameters['gi_dlambda']
gu_wave_mid = emulator_parameters['gu_wave_mid']
gv_wave_mid = emulator_parameters['gv_wave_mid']
gi_wave_mid = emulator_parameters['gi_wave_mid']

gu_init_idx = emulator_parameters['gu_init_idx']
gu_end_idx = emulator_parameters['gu_end_idx']
gv_init_idx = emulator_parameters['gv_init_idx']
gv_end_idx = emulator_parameters['gv_end_idx']
gi_init_idx = emulator_parameters['gi_init_idx']
gi_end_idx = emulator_parameters['gi_end_idx']

c = 2.9979e8 # m/s
c_aa = 2.9979e18 #AA/s
h = 6.626e-27 # erg*s
colarea = np.pi*radi**2 # cm^2
# gupixlen = 317
# gvpixlen = 342
# gipixlen = 332

hst_vband_bkg = { 
    ### (longitude, latitude): V-mag per arcsec^2, SA stands for Solar Avoidance zone (HST pointing within 55 deg of the sun)
    ### Define the lists of ecliptic latitude, longitude, and magnitude
    ### latitudes = [0, 15, 30, 45, 60, 75, 90]
    ### longitudes = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 160, 165, 180]
    ### magnitudes = [21.3, 21.7, 21.9, 22.0, 22.1, 22.2, 22.3, 22.4, 22.5, 22.6, 22.7, 22.8, 22.9, 23.0, 23.1, 23.2, 23.3, 23.4]

    ### original data sets for low longitude and latitude
    ### (0, 0) : 'SA',   (15, 0):  'SA',   (30, 0):  'SA',   (45, 0):  'SA', 
    ### (0, 15): 'SA',   (15, 15): 'SA',   (30, 15): 'SA',   (45, 15): 'SA',
    ### (0, 30): 'SA',   (15, 30): 'SA',   (30, 30): 'SA', 
    ### (0, 45): 'SA',   (15, 45): 'SA', 
    (0, 0) : 20.0, (15, 0):  20.3, (30, 0):  20.7, (45, 0):  21.0, (60, 0):  21.3, (75, 0):  21.7, (90, 0):  22.0, (105, 0):  22.2, (120, 0):  22.4, (135, 0):  22.4, (150, 0):  22.4, (165, 0):  22.3, (180, 0):  22.1, 
    (0, 15): 20.7, (15, 15): 21.0, (30, 15): 21.3, (45, 15): 21.6, (60, 15): 21.9, (75, 15): 22.2, (90, 15): 22.3, (105, 15): 22.5, (120, 15): 22.6, (135, 15): 22.6, (150, 15): 22.6, (165, 15): 22.5, (180, 15): 22.4, 
    (0, 30): 21.5, (15, 30): 21.6, (30, 30): 21.8, (45, 30): 22.1, (60, 30): 22.4, (75, 30): 22.6, (90, 30): 22.7, (105, 30): 22.9, (120, 30): 22.9, (135, 30): 22.9, (150, 30): 22.9, (165, 30): 22.8, (180, 30): 22.7, 
    (0, 45): 22.3, (15, 45): 22.0, (30, 45): 22.3, (45, 45): 22.5, (60, 45): 22.7, (75, 45): 22.9, (90, 45): 23.0, (105, 45): 23.1, (120, 45): 23.2, (135, 45): 23.2, (150, 45): 23.1, (165, 45): 23.0, (180, 45): 23.0, 
    (0, 60): 22.6, (15, 60): 22.6, (30, 60): 22.7, (45, 60): 22.9, (60, 60): 23.0, (75, 60): 23.1, (90, 60): 23.2, (105, 60): 23.3, (120, 60): 23.3, (135, 60): 23.3, (150, 60): 23.3, (165, 60): 23.2, (180, 60): 23.2, 
    (0, 75): 23.0, (15, 75): 23.1, (30, 75): 23.1, (45, 75): 23.1, (60, 75): 23.2, (75, 75): 23.2, (90, 75): 23.3, (105, 75): 23.3, (120, 75): 23.3, (135, 75): 23.4, (150, 75): 23.4, (165, 75): 23.4, (180, 75): 23.4, 
    (0, 90): 23.3, (15, 90): 23.3, (30, 90): 23.3, (45, 90): 23.3, (60, 90): 23.3, (75, 90): 23.3, (90, 90): 23.3, (105, 90): 23.3, (120, 90): 23.3, (135, 90): 23.3, (150, 90): 23.3, (165, 90): 23.3, (180, 90): 23.3, 
}

longitudes = np.unique(np.array(list(hst_vband_bkg.keys()))[:,0])
latitudes = np.unique(np.array(list(hst_vband_bkg.keys()))[:,1])

class emulator(object):
    def __init__(self, SED_file_path = '', file_name = ''):
        self.rootpath = os.getcwd()
        self.sedpath = os.path.join(SED_file_path)
        # self.csstgu = S.FileBandpass(rootpath+'/throughput/GUdesi.fits')
        # self.csstgv = S.FileBandpass(rootpath+'/throughput/GVdesi.fits')
        # self.csstgi = S.FileBandpass(rootpath+'/throughput/GIdesi.fits')
        self.filename = file_name

    def read_hdf5(self):
        """
        Read input BayeSED hdf5 file.
        
        return: 
        ------
            cat - hdf5 file catalog - [<HDF5 file (mode r)>]
        """
        cat = h5py.File(self.sedpath+self.filename, 'r')
        return cat
        
    # def parameters(self, save = None):
    #     """
    #     Extract some parameters from BayeSED hdf5 file.

    #     parameters:
    #     ----------
    #         save - choose to save the parameters into '.fits' file or not, True or None - [string]
    #     return: 
    #     ------
    #         parameters - fits catalog - [Table]
    #     """
    #     cat = self.read_hdf5()

    #     catcols = cat['parameters'].attrs['names']
    #     used = ['RA','DEC']+list(filter(lambda s: re.search("^MAG|^mAB|^z_|Nd|SNR",s), catcols))
    #     cols=list(list(catcols).index(used[i]) for i in range(0,len(used)))
    #     names_cols = {k: v for k, v in sorted(dict(zip(used, cols)).items(), key=lambda item: item[1])}
    #     parameters=Table(cat['parameters'][:,list(names_cols.values())], names=list(names_cols.keys()),copy=False)
    #     parameters['ID']=cat['ID'][:]

    #     if save:
    #         parameters.write(self.filename+'.fits',format='fits',overwrite=True)
    #     return parameters
    
    def sls_wave_spec(self, highres_wave, highres_flux, redshift, grism):
        """ 
        Generate CSST intrinsic spectra with CSST grsim resolution and corresponding length.

        parameters:
        ----------
            highres_wave - wavelength array for the input catalog, one single array, [2079,] - [array]
            highres_flux - flux array for the input catalog, arrays, [10000+, 2079]  - [array]
            redshift - redshift array for the input catalog, one single array, [10000+,] - [array]
            grism - CSST grism bands, in 'GU', 'GV', and 'GI' - [string]

        return:
        ------
            spectral_unit_wavelength - CSST grism wavelength grid in spectral units (length = 100+), arrays, [100+, ] - [array]
            spectral_unit_flux - CSST grism flux grid in spectral units (length = 100+), arrays, [10000+, 100+] - [array]
            mask - exclude some data with extreme error, see errorlog.rtf problem 1 and 2 for reference - [array]
        """

        if grism == 'GU':
            # R = gu_Res
            start_wl = gu_start_wl
            end_wl = gu_end_wl
            center_wl = gu_wave_mid
            delta_lambda = gu_dlambda
        elif grism == 'GV':
            # R = gv_Res
            start_wl = gv_start_wl
            end_wl = gv_end_wl
            center_wl = gv_wave_mid
            delta_lambda = gv_dlambda
        elif grism == 'GI':
            # R = gi_Res
            start_wl = gi_start_wl
            end_wl = gi_end_wl
            center_wl = gi_wave_mid
            delta_lambda = gi_dlambda
        else:
            raise ValueError('Invalid grism, please input GU, GV, or GI.')

        # center_wl = (start_wl + end_wl) / 2
        # calculating the mean delta lambda
        # delta_lambda = center_wl / R
        # obtain the numbers of spectral resolution units based on mean delta lambda
        N = int((end_wl - start_wl) / delta_lambda) + 1
        # generate an array for spectral resolution unit based on the grism resolution in the corresponding wavelength range
        spectral_unit_wavelength = np.array([start_wl + i * delta_lambda for i in range(N)])

        # calculating observed wavelength
        mask = (redshift[:] >= 0.00000001) & (redshift[:] <= 10) # there are some problems in some DESI phot catalog, see problem 1 in errors.log
        mask_nan = (redshift[:] < 0.00000001) & (redshift[:] > 10)

        redshift = redshift[mask]
        highres_flux = highres_flux[:][mask]
        wave_obs = np.ones_like(highres_flux)
        for i in (range(len(wave_obs))):
            wave_obs[i] = highres_wave * (1+redshift[i])
        grism_wave_list = []
        grism_flux_list = []
        for i in (range(len(wave_obs))):
            grism_wave_list.append(wave_obs[i][np.where( (wave_obs[i] >= start_wl) & (wave_obs[i] <= end_wl) )])
            grism_flux_list.append(highres_flux[i][np.where( (wave_obs[i] >= start_wl) & (wave_obs[i] <= end_wl) )])
        grism_wave_array = np.array(grism_wave_list, dtype=object) # data array in [xxx]
        grism_flux_array = np.array(grism_flux_list, dtype=object) # data array in [xxxxx,xxx]

        spectral_unit_flux = np.ones((len(grism_flux_array),len(spectral_unit_wavelength)))
        for i in range(len(spectral_unit_flux)):
            spectral_unit_flux[i] = np.interp(spectral_unit_wavelength, grism_wave_array[i], grism_flux_array[i])
        
        sigma = np.ones_like(spectral_unit_wavelength) * np.sqrt(delta_lambda**2 - (center_wl/(highres_wave[0]/(highres_wave[1]-highres_wave[0]))**2))

        return spectral_unit_wavelength, spectral_unit_flux, sigma, mask, mask_nan
    

    def kernels_convolve(self, highres_wave, highres_flux, redshift, grism, REE80):
        """ 
        Convolve the input high resolution spectra with CSST grsim resolution and expanding grid points.

        features:
        --------
            Using astropy.convolve package to convolve the spectra into CSST resolution along the dispersion direction 
        through a varing kernel based on the wavelength.
            This process involves dividing the convolution kernel into several corresponding kernels according to the 
        change in wavelength, and performing convolution on the corresponding wavelength range to complete it.

        parameters:
        ----------
            highres_wave - wavelength array for the input catalog, one single array, [2079,] - [array]
            highres_flux - flux array for the input catalog, arrays, [10000+, 2079]  - [array]
            redshift - redshift array for the input catalog, one single array, [10000+,] - [array]
            grism - CSST grism bands, in 'GU', 'GV', and 'GI' - [string]
            REE80 - CSST PSF 80 %\energy concentration radius for slitless spectroscopy, average <= 0.3'', max <= 0.4'' - [float]

        return:
        ------
            new_wave - CSST grism wavelength grid in pixels (length = 500+), arrays, [500+, ] - [array]
            new_flux - CSST grism flux grid in pixels (length = 500+), arrays, [10000+, 500+] - [array]
            mask - exclude some data with extreme error, see errorlog.rtf problem 1 and 2 for reference - [array]
        """

        if grism == 'GU':
            R = guRes
        elif grism == 'GV':
            R = gvRes
        elif grism == 'GI':
            R = giRes
        else:
            raise ValueError('Invalid grism, please input GU, GV, or GI.')
        
        wave, flux, mask = self.sls_wave_spec(highres_wave,highres_flux,redshift,grism)
        lrf = np.zeros([len(flux),len(wave)]) # create a grism-lengthed spectrum in pixel
        bk = [0]
        for i in range(1, len(wave)):
            kernel_range = utils.odd(wave[i]/R/2.35482)
            if kernel_range != utils.odd(wave[i-1]/R/2.35482):
                bk.append(i)
        bk.append(-1)
        for l in range(len(flux)):
            kernel = []
            fnew = []
            con = []
            # final = []
            for j in bk:
                kernel.append(Gaussian1DKernel(utils.odd(wave[j]/R/2.35482)/8))

            if len(bk) <= 3:
                fnew.append(flux[l][bk[0]:bk[1]+int((len(kernel[0].array)-1)/2)])
                con.append(convolve(fnew[0],kernel[0],boundary='extend')[bk[0]:bk[0+1]])
                fnew.append(flux[l][bk[1]-int((len(kernel[1].array)-1)/2):bk[1+1]])
                con.append(convolve(fnew[1],kernel[1],boundary='extend')[int((len(kernel[1].array)-1)/2):])
                fnew.append(flux[l][bk[2]-int((len(kernel[2].array)-1)/2):])
                con.append(convolve(fnew[2],kernel[2],boundary='extend')[int((len(kernel[2].array)-1)/2)-1:-1])
                lrf[l] = np.hstack((np.array([y for x in con for y in x])))
            elif len(bk) > 3:
                fnew.append(flux[l][bk[0]:bk[1]+int((len(kernel[0].array)-1)/2)])
                con.append(convolve(fnew[0],kernel[0],boundary='extend')[bk[0]:bk[1]])
                for k in range(1,len(bk)-2):
                    # print(k+1)
                    if bk[k+1]+int((len(kernel[k+1].array)-1)/2) > len(wave):
                        fnew.append(flux[l][bk[k]-int((len(kernel[k].array)-1)/2):bk[k+1]])
                        con.append(convolve(fnew[k],kernel[k],boundary='extend')[int((len(kernel[k].array)-1)/2):])
                    else:
                        fnew.append(flux[l][bk[k]-int((len(kernel[k].array)-1)/2):bk[k+1]+int((len(kernel[k].array)-1)/2)])
                        con.append(convolve(fnew[k],kernel[k],boundary='extend')[int((len(kernel[k].array)-1)/2):-int((len(kernel[k].array)-1)/2)])
                fnew.append(flux[l][bk[-2]-int((len(kernel[k].array)-1)/2):])
                con.append(convolve(fnew[-1],kernel[-1],boundary='extend')[int((len(kernel[-1].array)-1)/2)-1:-1])
                lrf[l] = np.hstack((np.array([y for x in con for y in x]),flux[l][-1]))
        gif = interp1d(wave, lrf)
        # create a new flux grid by expanding the length to be round(REE80/arcsecperpix) times longer,
        # the value of round(REE80/arcsecperpix) should be 4 pixel if REE80 = 0.3'' and arcsecperpix = 0.074''/pix
        new_wave = np.linspace(wave[0], wave[-1], len(wave) * round(REE80/arcsecperpix))
        new_flux = gif(new_wave)
        return new_wave, new_flux, mask


    def kernels_convolve_new(self, highres_wave, highres_flux, redshift, grism, fwhm):
        """ 
        Convolve the input high resolution spectra with CSST grsim resolution and expanding grid points.

        features:
        --------
            Using astropy.convolve package to convolve the spectra into CSST resolution along the dispersion direction 
        through a varing kernel based on the wavelength.
            This process involves dividing the convolution kernel into several corresponding kernels according to the 
        change in wavelength, and performing convolution on the corresponding wavelength range to complete it.

        parameters:
        ----------
            highres_wave - wavelength array for the input catalog, one single array, [2079,] - [array]
            highres_flux - flux array for the input catalog, arrays, [10000+, 2079]  - [array]
            redshift - redshift array for the input catalog, one single array, [10000+,] - [array]
            grism - CSST grism bands, in 'GU', 'GV', and 'GI' - [string]
            fwhm - CSST PSF fwhm concentration radius for slitless spectroscopy, average <= 0.2'', max <= 0.22'' - [float]

        return:
        ------
            new_wave - CSST grism wavelength grid in pixels (length = 500+), arrays, [500+, ] - [array]
            new_flux - CSST grism flux grid in pixels (length = 500+), arrays, [10000+, 500+] - [array]
            mask - exclude some data with extreme error, see errorlog.rtf problem 1 and 2 for reference - [array]
        """

        if grism == 'GU':
            min_wl = gu_min_wl
            max_wl = gu_max_wl
        elif grism == 'GV':
            min_wl = gv_min_wl
            max_wl = gv_max_wl
        elif grism == 'GI':
            min_wl = gi_min_wl
            max_wl = gi_max_wl
        else:
            raise ValueError('Invalid grism, please input GU, GV, or GI.')
        
        wave, flux, sigma, mask, mask_nan = self.sls_wave_spec(highres_wave,highres_flux,redshift,grism)
        
        fnew_total = np.zeros([len(flux),len(wave)])
        for i in range(len(flux)):
            fnew_total[i]  = self.varsmooth(wave, flux[i], sigma)
        fnew_total_arr = np.array(fnew_total)
        flux_func = interp1d(wave, fnew_total_arr)
        new_wave = np.linspace(min_wl, max_wl, round(len(wave) * fwhm/arcsecperpix))
        new_flux = flux_func(new_wave)
        return new_wave, new_flux, mask, mask_nan

    def varsmooth(self, x, y, sig_x, xout=None, oversample=1):
        """    
        Fast and accurate convolution with a Gaussian of variable width.

        This function performs an accurate Fourier convolution of a vector, or the
        columns of an array, with a Gaussian kernel that has a varying or constant
        standard deviation (sigma) per pixel. The convolution is done using fast
        Fourier transform (FFT) and the analytic expression of the Fourier
        transform of the Gaussian function, like in the pPXF method. This allows
        for an accurate convolution even when the Gaussian is severely
        undersampled.

        This function is recommended over standard convolution even when dealing
        with a constant Gaussian width, due to its more accurate handling of
        undersampling issues.

        This function implements Algorithm 1 in `Cappellari (2023)
        <https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3273C>`_

        Input Parameters
        ----------------

        x : array_like
            Coordinate of every pixel in `y`.
        y : array_like
            Input vector or array of column-spectra.
        sig_x : float or array_like
            Gaussian sigma of every pixel in units of `x`.
            If sigma is constant, `sig_x` can be a scalar. 
            In this case, `x` must be uniformly sampled.
        oversample : float, optional
            Oversampling factor before convolution (default: 1).
        xout : array_like, optional
            Output `x` coordinate used to compute the convolved `y`.

        Output Parameters
        -----------------

        yout : array_like
            Convolved vector or columns of the array `y`.

        """
        assert len(x) == len(y), "`x` and `y` must have the same length"

        if np.isscalar(sig_x):
            dx = np.diff(x)
            assert np.all(np.isclose(dx[0], dx)), "`x` must be uniformly spaced, when `sig_x` is a scalar"
            n = len(x)
            sig_max = sig_x*(n - 1)/(x[-1] - x[0])
            y_new = y.T
        else:
            assert len(x) == len(sig_x), "`x` and `sig_x` must have the same length"
            # Stretches spectrum to have equal sigma in the new coordinate
            sig = sig_x/np.gradient(x)
            sig = sig.clip(0.1)   # Clip to >=0.1 pixels
            sig_max = np.max(sig)*oversample
            xs = np.cumsum(sig_max/sig)
            n = int(np.ceil(xs[-1] - xs[0]))
            x_new = np.linspace(xs[0], xs[-1], n)
            y_new = utils.interp(x_new, xs, y.T)

        # Convolve spectrum with a Gaussian using analytic FT like pPXF
        npad = 2**int(np.ceil(np.log2(n)))
        ft = np.fft.rfft(y_new, npad)
        w = np.linspace(0, np.pi*sig_max, ft.shape[-1])
        ft_gau = np.exp(-0.5*w**2)
        yout = np.fft.irfft(ft*ft_gau, npad).T[:n]

        if not np.isscalar(sig_x):
            if xout is not None:
                xs = np.interp(xout, x, xs)  # xs is 1-dim
            yout = utils.interp(xs, x_new, yout.T).T

        return yout

    def phot_convolve(self,highres_wave, highres_flux):
        """
        Convolve the input high resolution spectra with CSST photometric resolution
        """

        sigma = 5
        flux_lowres = gaussian_filter1d(highres_flux, sigma)

        return highres_wave, highres_flux
    

    def lamshift_width(self,flux,width):
        """ 
        Broadening and smoothing the spectrum shape with width profile distribution function (along the dispersion direction).

        parameters:
        ----------
            flux - flux of the slitless spectrum - [array]
            width - width profile distribution function of the spectrum - [array]

        return:
        ------
            dxshift - smoothed slitless spectrum with morpological broadening effect - [array]
        """

        dxshift = np.zeros(len(flux))
        widthfunc = width/width.sum()
        dx = np.arange(len(width))-round((len(width)-1)/2)
        for i in range(len(width)):
            if dx[i] < 0:
                dxshift+=(np.hstack( [np.zeros(abs(dx[i])), widthfunc[i]*flux[:dx[i]]] ))
            elif dx[i] == 0:
                dxshift+=(widthfunc[i]*flux)
            elif dx[i] > 0:
                dxshift+=(np.hstack( [widthfunc[i]*flux[dx[i]:], np.zeros(abs(dx[i]))] ))
        dxshift[0: np.int64((len(dx)-1)/2 -1)] = flux[0:np.int64((len(dx)-1)/2 -1)]
        dxshift[-np.int64((len(dx)-1)/2 -1):] = flux[-np.int64((len(dx)-1)/2 -1):]

        return dxshift

    def wave_err(self,flux):
        """ 
        Add a wavelength offset in slitless spectrum according to the width profile. 
        0.1 to 0.5 percent error is given. 
        
        parameters:
        ----------
            flux - flux of the slitless spectrum - [array]

        return:
        ------
            wave_off - the wavelength calibration error in pixel - [float]
            flux_new - flux of the slitless spectrum with wavelength extraction errors within 3 pixels - [array]
        """
        wave_err_distribution = utils.gaussian(3,0.4)

        x_center_pos = int(wave_err_distribution.shape[0]/2)
        wave_off_int = np.random.choice(np.arange(-x_center_pos,x_center_pos+1,1), p = wave_err_distribution)

        if (wave_off_int <= 0.5) & (wave_off_int >= -0.5):
            wave_off = round(np.random.uniform(-0.5,0.5),1)
            flux_new = flux

        # if (wave_off_int <= 3.5) & (wave_off_int >= 0.5):
        #     wave_off = round(np.random.uniform(0.5,3.5),1)
        #     flux_new = np.hstack((flux[:wave_off_int]*wave_err_distribution[:wave_off_int],flux))[:-wave_off_int]

        if (wave_off_int <= 1.5) & (wave_off_int >= 0.5):
            wave_off = round(np.random.uniform(0.5,1.5),1)
            flux_new = np.hstack((flux[:wave_off_int]*wave_err_distribution[:wave_off_int],flux))[:-wave_off_int]
        if (wave_off_int <= 2.5) & (wave_off_int >= 1.5):
            wave_off = round(np.random.uniform(1.5,2.5),1)
            flux_new = np.hstack((flux[:wave_off_int]*wave_err_distribution[:wave_off_int],flux))[:-wave_off_int]
        if (wave_off_int <= 3.5) & (wave_off_int >= 2.5):
            wave_off = round(np.random.uniform(2.5,3.5),1)
            flux_new = np.hstack((flux[:wave_off_int]*wave_err_distribution[:wave_off_int],flux))[:-wave_off_int]

        # if (wave_off_int <= -0.5) & (wave_off_int >= -3.5):
        #     wave_off = round(np.random.uniform(-3.5,-0.5),1)
        #     flux_new = np.hstack((flux,flux[:-wave_off_int]*wave_err_distribution[:-wave_off_int]))[-wave_off_int:]

        if (wave_off_int <= -0.5) & (wave_off_int >= -1.5):
            wave_off = round(np.random.uniform(-1.5,-0.5),1)
            flux_new = np.hstack((flux,flux[:-wave_off_int]*wave_err_distribution[:-wave_off_int]))[-wave_off_int:]
        if (wave_off_int <= -1.5) & (wave_off_int >= -2.5):
            wave_off = round(np.random.uniform(-2.5,-1.5),1)
            flux_new = np.hstack((flux,flux[:-wave_off_int]*wave_err_distribution[:-wave_off_int]))[-wave_off_int:]
        if (wave_off_int <= -2.5) & (wave_off_int >= -3.5):
            wave_off = round(np.random.uniform(-3.5,-2.5),1)
            flux_new = np.hstack((flux,flux[:-wave_off_int]*wave_err_distribution[:-wave_off_int]))[-wave_off_int:]

        return wave_off, flux_new

    def match_bkg_mag(self, ra, dec):
        """
        HST approximate zodiacal sky background as a function of heliocentric ecliptic longitude and latitude in V-mag per arcsec^2.

        parameters:
        ----------
            ra - ra of the object - [float]
            dec - dec of the object - [float]

        return:
        ------
            mag - V-band magnitude per arcsec^2 of given ra, dec - [float]
        """

        if ra > 180:
            lon = 360 - ra
        else: 
            lon = ra
        lat = np.abs(dec)

        lon_idx = utils.find_nearest(longitudes,lon)
        lat_idx = utils.find_nearest(latitudes,lat)
        key = (longitudes[lon_idx], latitudes[lat_idx])
        return hst_vband_bkg.get(key)

    def bkg_per_pix_time(self,ra,dec,bkgwave,bkgflux,tp):
        """
        Calculate sky background count per pixel per time of three CSST grism bands.
        
        features:
        --------
            Sky background noise is calculated following the sky emission data from HST/WFC3 Handbook,
        see [Dressel \& Marinelli(2023)]{Dressel+2023} Dressel, L. \& Marinelli, M., 2023, WFC3 Instrument Handbook for Cycle 31 v. 15.0, 15

        parameters:
        ----------
            ra - ra of the object - [float]
            dec - dec of the object - [float]
            bkgwave - sky background spectrum wavelength in Å - [array]
            bkgflux - sky background spectrum flux in erg/s/cm^2/Hz/arcsec^2 - [array]
            tp - throughput of CSST grism band - [array]

        retrun:
        ------
            bkg_per_pix_time - sky background count per pixel per time of the input CSST grism in e-/s/pixel - [float]
        """

        bkg_v_mag = self.match_bkg_mag(ra,dec)
        bkgflux_ = 10**(0.4*(22.1-bkg_v_mag))*bkgflux

        bkg_in_phot = utils.fnu2fphot(bkgwave,bkgflux_) # get photons/s/cm^2/Å/arcsec^2
        bkg_per_pix_time = np.trapz(tp*bkg_in_phot,bkgwave) * colarea * arcsecperpix **2 # integration with throughput to get electrons/s/cm^2/arcsec^2
                                                                                        # and multiply collection area cm^2 and arcsecperpix 0.074 (arcsec/pixel)^2
                                                                                        # to get average sky bkg counts of each pixel in this grism band.
        return bkg_per_pix_time

    def add_noise(self, ra, dec, wave, fnu, bkgwave, bkgflux, tp, height, width, grism):
        """
        Add sky background and instrumental effect noise to the observed spectrum

        features:
        --------
            The noise is added in electron spectrum for each spectrum-covered pixel.
            Sky background noise is calculated following the sky emission data from HST/WFC3 Handbook,
        see [Dressel \& Marinelli(2023)]{Dressel+2023} Dressel, L. \& Marinelli, M., 2023, WFC3 Instrument Handbook for Cycle 31 v. 15.0, 15.
            The 2-D profile distribution function along and perpendicular to the dispersion direction is considered, 
        which is dispertion projection named "height" and spatial projection named "width". 
            The "height" profile stands for the total columns used in calculating the sky background and instrumental effect counts, 
        which affect the noise level of the simulated observed spectra.
            The "width" profile is used to estimate the self-blending effect along the dispersion direction, which affect the spectra 
        profile and somehow smoothed by the "width" profile.

        parameters:
        ----------
            ra - ra of the object - [float]
            dec - dec of the object - [float]
            wave - input wavelength array, in angstrom (Å) unit, single array - [array]
            fnu - input flux array, in fnu (erg/s/cm^2/Hz) unit, single array - [array]
            bkgwave - sky background spectrum wavelength in angstrom (Å) unit, single array - [array]
            bkgflux - sky background spectrum flux in erg/s/cm^2/Hz/arcsec^2 unit, single array - [array]
            tp - throughput of CSST grism band - [array]
            height - height (dispersion projection) profile distribution function - [array]
            width - width (spatial projection) profile distribution function - [array]

        retrun:
        ------
            elec_per_column - simulated electron spectrum in e- - [array]
            fnu_with_noise - simulated observed CSST spectrum with noise - [array]
            fnu_error - simulated observed CSST spectrum errors - [array]
            rms_in_e - the root mean square of noise in electrons (e-) - [float]
            snr - the signal-to-noise ratio curve calculated in electrons - [array]
            snr_mean - the mean signal-to-noise ratio of spectrum - [float]
            wave_off - the wavelength calibration error in pixel - [float]
            spec_mask - a universal mask for the spectrum to avoid the tranmission curve edge with huge noise - [array]
            snr_mask - a special mask for the spectrum with snr >= 1 - [array]
        """

        # broadening and smoothing the intrinsic slitless spectrum with width profile
        flux_ = self.lamshift_width(fnu,width)
        # adding a 0.1% to 0.5% precision error (up to 3 pixel offset) in wavelength grid, affect on the flux grid
        wave_off, flux = self.wave_err(flux_)
        throughput = tp

        # transform fnu into fphot（erg/s/cm2/Hz -> photons/s/cm2/Å)
        fphot = utils.fnu2fphot(wave,flux) # photons/s/cm2/A
        # a mean lambda interval for each pixel (4 pixels consist a spectral resolution unit)
        dlambda = (wave.max()-wave.min())/len(wave)
        # photons of each pixel, since the perpendicular directio (height) source values are all considered
        photon_per_column = fphot * colarea * expt * expnum * dlambda
        # electrons of each pixel
        elec_per_column = photon_per_column * throughput
        # photon noise of each pixel and transform into electron noise
        # flux_noise_per_res = np.random.poisson(photon_per_res) * throughput
        # sky background counts of each column in theory, e-/s/pix * s * pix
        bkg_counts_per_column_theory = self.bkg_per_pix_time(ra,dec,bkgwave,bkgflux,throughput) * expt * expnum * len(height)
        # sky background counts of each pixel in theory, e-/s/pix * s 
        # bkg_counts_per_pix = np.ones_like(wave) * self.bkg_per_pix_time(ra,dec,bkgwave,bkgflux,throughput) * expt * expnum
        # performing a Poisson distribution processing on the sky background column in the perpendicular dispersion direction 
        # of each spectral pixel that needs to be calculated.
        # bkg_counts_per_column = np.zeros_like(wave)
        # for k in range(len(height)):
        #     # sky background counts of each spatial column in e-
        #     bkg_counts_per_column += np.random.poisson(bkg_counts_per_pix) 
        # dark current value of each column
        dc_noise_column = dc_value * expt * expnum * len(height) # dark current of each pixel for all columns
        # read-out noise of each pixel
        read_noise_along_dis = np.zeros_like(wave)
        for l in range(expnum):
            for o in range(len(read_noise_along_dis)):
                read_noise_along_dis[o] += round(np.random.uniform(-5,5))
        # total noise =  sky background noise + poisson(photon noise from source + dark current) + read-out noise
        totalnoise =  np.random.poisson(elec_per_column + bkg_counts_per_column_theory + dc_noise_column) + read_noise_along_dis
        # root mean square of noise in electrons
        rms_in_e = np.mean(np.sqrt(totalnoise))
        # all the electrons poissoned are considered as simulated observed electrons on CCD, 
        # after subtracting sky background counts and dark currents, the simulated observed 
        # spectrum in electrons is obtained
        elec_with_noise = totalnoise - bkg_counts_per_column_theory - dc_noise_column
        # if elec_with_noise < 0, set elec_with_noise = 0, and a upper limit of saturation pixel is set as 90000 e-
        elec_with_noise = [lowlim if lowlim >= 0 else 0 for lowlim in elec_with_noise] 
        elec_with_noise = [uplim if uplim <= 90000 else 90000 for uplim in elec_with_noise] 
        # converting electrons into photons and then fnu
        photon_with_noise = elec_with_noise / throughput
        fnu_with_noise = utils.fphot2fnu(wave, photon_with_noise / (colarea*expt*expnum*dlambda))
        # calculating flux error
        elec_err = elec_per_column + bkg_counts_per_column_theory + dc_noise_column + 5**2 * len(height) * expnum
        phot_err = np.sqrt(elec_err) / throughput
        fnu_error = utils.fphot2fnu(wave, phot_err/(colarea*expt*expnum*dlambda))
        # calculating signal-to-noise ratio
        snr = elec_with_noise/np.sqrt(elec_per_column + bkg_counts_per_column_theory + dc_noise_column + expnum*(len(height))*5**2)
        snr_mean = np.mean(snr)

        if grism == 'GU':
            spec_mask = np.zeros_like(fnu_with_noise, dtype=bool)
            spec_mask[gu_init_idx:gu_end_idx] = True
        if grism == 'GV':
            spec_mask = np.zeros_like(fnu_with_noise, dtype=bool)
            spec_mask[gv_init_idx:gv_end_idx] = True
        if grism == 'GI':
            spec_mask = np.zeros_like(fnu_with_noise, dtype=bool)
            spec_mask[gi_init_idx:gi_end_idx] = True

        snr_mask = snr >= 1

        return elec_per_column, fnu_with_noise, fnu_error, rms_in_e, snr, snr_mean, wave_off, spec_mask, snr_mask


    def wave_err_zouhu(self,flux,flux1,flux2,snr):

        wave_err_distribution = utils.gaussian(3,0.4)

        x_center_pos = int(wave_err_distribution.shape[0]/2)
        wave_off_int = np.random.choice(np.arange(-x_center_pos,x_center_pos+1,1), p = wave_err_distribution)

        if (wave_off_int <= 0.5) & (wave_off_int >= -0.5):
            wave_off = round(np.random.uniform(-0.5,0.5),1)
            flux_new = flux
            flux1_new = flux1
            flux2_new = flux2
            snr_new = snr

        if (wave_off_int <= 1.5) & (wave_off_int >= 0.5):
            wave_off = round(np.random.uniform(0.5,1.5),1)
            flux_new = np.hstack((flux[:wave_off_int]*wave_err_distribution[:wave_off_int],flux))[:-wave_off_int]
            flux1_new = np.hstack((flux1[:wave_off_int]*wave_err_distribution[:wave_off_int],flux1))[:-wave_off_int]
            flux2_new = np.hstack((flux2[:wave_off_int]*wave_err_distribution[:wave_off_int],flux2))[:-wave_off_int]
            snr_new = np.hstack((snr[:wave_off_int]*wave_err_distribution[:wave_off_int],snr))[:-wave_off_int]

        if (wave_off_int <= 2.5) & (wave_off_int >= 1.5):
            wave_off = round(np.random.uniform(1.5,2.5),1)
            flux_new = np.hstack((flux[:wave_off_int]*wave_err_distribution[:wave_off_int],flux))[:-wave_off_int]
            flux1_new = np.hstack((flux1[:wave_off_int]*wave_err_distribution[:wave_off_int],flux1))[:-wave_off_int]
            flux2_new = np.hstack((flux2[:wave_off_int]*wave_err_distribution[:wave_off_int],flux2))[:-wave_off_int]
            snr_new = np.hstack((snr[:wave_off_int]*wave_err_distribution[:wave_off_int],snr))[:-wave_off_int]

        if (wave_off_int <= 3.5) & (wave_off_int >= 2.5):
            wave_off = round(np.random.uniform(2.5,3.5),1)
            flux_new = np.hstack((flux[:wave_off_int]*wave_err_distribution[:wave_off_int],flux))[:-wave_off_int]
            flux1_new = np.hstack((flux1[:wave_off_int]*wave_err_distribution[:wave_off_int],flux1))[:-wave_off_int]
            flux2_new = np.hstack((flux2[:wave_off_int]*wave_err_distribution[:wave_off_int],flux2))[:-wave_off_int]
            snr_new = np.hstack((snr[:wave_off_int]*wave_err_distribution[:wave_off_int],snr))[:-wave_off_int]

        # if (wave_off_int <= 3.5) & (wave_off_int >= 0.5):
        #     wave_off = round(np.random.uniform(0.5,3.5),1)
        #     flux_new = np.hstack((flux[:wave_off_int]*wave_err_distribution[:wave_off_int],flux))[:-wave_off_int]
        #     flux1_new = np.hstack((flux1[:wave_off_int]*wave_err_distribution[:wave_off_int],flux))[:-wave_off_int]
        #     flux2_new = np.hstack((flux2[:wave_off_int]*wave_err_distribution[:wave_off_int],flux))[:-wave_off_int]
        #     snr_new = np.hstack((snr[:wave_off_int]*wave_err_distribution[:wave_off_int],flux))[:-wave_off_int]

        # if (wave_off_int <= -0.5) & (wave_off_int >= -3.5):
        #     wave_off = round(np.random.uniform(-3.5,-0.5),1)
        #     flux_new = np.hstack((flux,flux[:-wave_off_int]*wave_err_distribution[:-wave_off_int]))[-wave_off_int:]
        #     flux1_new = np.hstack((flux,flux[:-wave_off_int]*wave_err_distribution[:-wave_off_int]))[-wave_off_int:]
        #     flux2_new = np.hstack((flux,flux[:-wave_off_int]*wave_err_distribution[:-wave_off_int]))[-wave_off_int:]
        #     snr_new = np.hstack((flux,flux[:-wave_off_int]*wave_err_distribution[:-wave_off_int]))[-wave_off_int:]

        if (wave_off_int <= -0.5) & (wave_off_int >= -1.5):
            wave_off = round(np.random.uniform(-1.5,-0.5),1)
            flux_new = np.hstack((flux,flux[:-wave_off_int]*wave_err_distribution[:-wave_off_int]))[-wave_off_int:]
            flux1_new = np.hstack((flux1,flux1[:-wave_off_int]*wave_err_distribution[:-wave_off_int]))[-wave_off_int:]
            flux2_new = np.hstack((flux2,flux2[:-wave_off_int]*wave_err_distribution[:-wave_off_int]))[-wave_off_int:]
            snr_new = np.hstack((snr,snr[:-wave_off_int]*wave_err_distribution[:-wave_off_int]))[-wave_off_int:]

        if (wave_off_int <= -1.5) & (wave_off_int >= -2.5):
            wave_off = round(np.random.uniform(-2.5,-1.5),1)
            flux_new = np.hstack((flux,flux[:-wave_off_int]*wave_err_distribution[:-wave_off_int]))[-wave_off_int:]
            flux1_new = np.hstack((flux1,flux1[:-wave_off_int]*wave_err_distribution[:-wave_off_int]))[-wave_off_int:]
            flux2_new = np.hstack((flux2,flux2[:-wave_off_int]*wave_err_distribution[:-wave_off_int]))[-wave_off_int:]
            snr_new = np.hstack((snr,snr[:-wave_off_int]*wave_err_distribution[:-wave_off_int]))[-wave_off_int:]

        if (wave_off_int <= -2.5) & (wave_off_int >= -3.5):
            wave_off = round(np.random.uniform(-3.5,-2.5),1)
            flux_new = np.hstack((flux,flux[:-wave_off_int]*wave_err_distribution[:-wave_off_int]))[-wave_off_int:]
            flux1_new = np.hstack((flux1,flux1[:-wave_off_int]*wave_err_distribution[:-wave_off_int]))[-wave_off_int:]
            flux2_new = np.hstack((flux2,flux2[:-wave_off_int]*wave_err_distribution[:-wave_off_int]))[-wave_off_int:]
            snr_new = np.hstack((snr,snr[:-wave_off_int]*wave_err_distribution[:-wave_off_int]))[-wave_off_int:]
            
        return wave_off, flux_new, flux1_new, flux2_new, snr_new
    
    def add_noise_zouhu(self, ra, dec, wave, fnu, bkgwave, bkgflux, tp, height, width, grism):

        flux = self.lamshift_width(fnu,width)
        throughput = tp
        fphot = utils.fnu2fphot(wave,flux)
        dlambda = (wave.max()-wave.min())/len(wave)
        photon_per_column = fphot * colarea * expt * expnum * dlambda
        elec_per_column = photon_per_column * throughput
        bkg_counts_per_column_theory = self.bkg_per_pix_time(ra,dec,bkgwave,bkgflux,throughput) * expt * expnum * len(height)
        dc_noise_column = dc_value * expt * expnum * len(height)
        read_noise_along_dis = np.zeros_like(wave)
        for l in range(expnum):
            for o in range(len(read_noise_along_dis)):
                read_noise_along_dis[o] += round(np.random.uniform(-5,5))
        totalnoise =  np.random.poisson(elec_per_column + bkg_counts_per_column_theory + dc_noise_column) + read_noise_along_dis
        rms_in_e = np.mean(np.sqrt(totalnoise))
        elec_with_noise = totalnoise - bkg_counts_per_column_theory - dc_noise_column
        elec_with_noise = [lowlim if lowlim >= 0 else 0 for lowlim in elec_with_noise] 
        elec_with_noise = [uplim if uplim <= 90000 else 90000 for uplim in elec_with_noise] 
        photon_with_noise = elec_with_noise / throughput
        fnu_with_noise = utils.fphot2fnu(wave, photon_with_noise / (colarea*expt*expnum*dlambda))
        elec_err = elec_per_column + bkg_counts_per_column_theory + dc_noise_column + 5**2 * len(height) * expnum
        phot_err = np.sqrt(elec_err) / throughput
        fnu_error = utils.fphot2fnu(wave, phot_err/(colarea*expt*expnum*dlambda))
        snr = elec_with_noise/np.sqrt(elec_per_column + bkg_counts_per_column_theory + dc_noise_column + expnum*(len(height))*5**2)
        snr_mean = np.mean(snr)

        wave_off, flux_wce, fnu_with_noise_wce, fnu_error_wce, snr_wce = self.wave_err_zouhu(flux,fnu_with_noise,fnu_error,snr)
        fphot_wce = utils.fnu2fphot(wave,flux_wce)
        photon_per_column_wce = fphot_wce * colarea * expt * expnum * dlambda
        elec_per_column_wce = photon_per_column_wce * throughput

        if grism == 'GU':
            spec_mask = np.zeros_like(fnu_with_noise, dtype=bool)
            spec_mask[gu_init_idx:gu_end_idx] = True
        if grism == 'GV':
            spec_mask = np.zeros_like(fnu_with_noise, dtype=bool)
            spec_mask[gv_init_idx:gv_end_idx] = True
        if grism == 'GI':
            spec_mask = np.zeros_like(fnu_with_noise, dtype=bool)
            spec_mask[gi_init_idx:gi_end_idx] = True

        snr_mask = snr >= 1

        return flux_wce, elec_per_column,elec_per_column_wce, fnu_with_noise,fnu_with_noise_wce, fnu_error,fnu_error_wce, rms_in_e, snr,snr_wce, snr_mean, wave_off, spec_mask, snr_mask

    def el_detect(self,wave,flux,init_wave,end_wave,threshold):
        """ 
        Detect emission lines in the spectra using 'find_lines_derivative' function in specutils package.

        features:
        --------
            Using 'find_lines_derivative' function to derive emission lines with a normalized spectra ( spectrum/fitted_continuum )
            Choosing the initial and end index of wavelength grid to avoid the extremely noisy region at the end of each spectra.
            Wavelength unit should be in 'angstrom' while flux unit should be in 'uJy'.
            Different threshold can result in different line detections results, please have some test before choosing the reasonable threshold.

        parameters:
        ----------
            wave - wavelength grid of the galaxy - [array]
            flux - flux grid of the galaxy - [array]
            init_wave - the initial index in wavelength grid - [int]
            end_wave - the end index in wavelength grid - [int]
            # x_unit - the unit used in modeling specutils.spectrum for wavelength, recommanding 'angstrom'
            # y_unit - the unit used in modeling specutils.spectrum for flux, recommanding 'Jy'
            threshold - threshold of the 'find_lines_derivative' - [float]

        return:
        ------
            lines - QTable created by specutils, contains three columns of 'line_center', 'line_tpe', 'line_center_index' - [QTable]
        """

        x = wave[init_wave:end_wave]
        y = flux[init_wave:end_wave]

        spectrum = Spectrum1D(flux=y*u.uJy, spectral_axis=x*u.angstrom)
        g1_fit = fit_generic_continuum(spectrum)
        mean = np.mean(spectrum.flux)
        continumm = spectrum.flux[np.argmin(np.abs(spectrum.flux - mean))]

        y_continuum_fitted = g1_fit(x*u.angstrom)
        y_continuum_fitted[y_continuum_fitted < continumm] = continumm
        spec_normalized = spectrum / y_continuum_fitted
        lines = find_lines_derivative(spec_normalized, flux_threshold=threshold)  

        return lines

    def el_success(self, noisy, intri, threshold):
        """ 
        Detect emission lines in two QTable created by specutils

        features:
        --------
            Using 'find_lines_derivative' function to derive emission lines with a normalized spectra ( spectrum/fitted_continuum )
            Choosing the initial and end index of wavelength grid to avoid the extremely noisy region at the end of each spectra.
            Wavelength unit should be in 'angstrom' while flux unit should be in 'uJy'.
            Different threshold can result in different line detections results, please have some test before choosing the reasonable threshold.

        parameters:
        ----------
            noisy - QTable with emission lines detection from simulated noisy spectra - [QTable]
            intri - QTable with emission lines detection from simulated intrinsic spectra - [QTable]
            threshold - threshold of the emission line center wavelength between noisy and intrinsic spectra - [float]

        return:
        ------
            intrinsic_el_arr - the emission line detection imformation array from intrinsic spectra, this result shows the total emission-line galaxies 
                        satisfying the 'find_lines_derivative' method in the corresponding hdf5 - [array]
            final_detec_el - the emission line detection imformation array from noisy spectra and satisfies the threshold of 'line_center' compared with 
                        intrinsic spectra (typicallly 1 PSF, 0.3 arcsec, 50 angstrom?) - [array]
            # success_rate - the successful emission line detection rate from noisy spectra compared with intrinsic spectra - [float]
        
        result example:
        --------------
            intrinsic_el_arr = array([
                (   2, list([7186.501393170599, 9379.743969557454, 9661.915529092606]), list([79, 250, 272]), 3),
                (   4, list([7994.538131839441, 8199.753811501369, 8699.967030677319]), list([142, 158, 197]), 3),
                (   6, list([7699.540592325419]), list([119]), 1), ...,
                (9995, list([7609.758732473326, 9969.739048585498]), list([112, 296]), 2),
                (9996, list([8097.145971670405, 9059.094470085693, 9341.266029620843]), list([150, 225, 247]), 3),
                (9997, list([7301.935212980434, 9854.305228775664]), list([88, 287]), 2)],
                dtype=[('id', '<i8'), ('wave', 'O'), ('idx', 'O'), ('el_number', '<i8')])

            final_detec_el = array([
                (   2, list([7186.501393170599, 7750.844512240901, 8007.364111818311, 8276.709691374592, 8584.533210867485, 8738.44497061393, 8853.878790423765, 9174.528289895527, 9405.395929515196, 9508.00376934616, 9815.827288839053, 9867.131208754534, 9944.087088627757]), list([79, 123, 143, 164, 188, 200, 209, 234, 252, 260, 284, 288, 294]), 2),
                (   4, list([7994.538131839441, 8199.753811501369]), list([142, 158]), 2),
                (   6, list([7699.540592325419]), list([119]), 1), 
                ...,
                (9995, list([7609.758732473326, 9200.180249853267, 9777.34934890244, 9956.913068606627]), list([112, 236, 281, 295]), 2),
                (9996, list([6955.63375355093, 7314.761192959304, 7366.065112874786, 7571.280792536714, 7840.626372092995, 8058.668031733793, 8199.753811501369, 8789.748890529412, 8969.3126102336, 9071.920450064563, 9136.050349958916, 9379.743969557454, 9495.17778936729, 9610.611609177124, 9764.52336892357, 9854.305228775664]), list([61, 89, 93, 109, 130, 147, 158, 204, 218, 226, 231, 250, 259, 268, 280, 287]), 3),
                (9997, list([7301.935212980434, 9854.305228775664]), list([88, 287]), 2)],
                dtype=[('id', '<i8'), ('wave', 'O'), ('idx', 'O'), ('el_number', '<i8')])
        """
        
        intrinsic_el_list = []
        success_detect_list = []

        dtype1 = [('id', int), ('wave', np.ndarray), ('idx', np.ndarray), ('intri_el_number', int)]
        dtype2 = [('id', int), ('wave', np.ndarray), ('idx', np.ndarray), ('detect_el_number', int)]
        for i in range(len(noisy)):
            if len(noisy[i]) == 0:
                pass
            elif len(noisy[i]) >= 1:
                if len(noisy[i][noisy[i]['line_type'] == 'emission']) !=0:

                    noisy_wave = noisy[i][noisy[i]['line_type'] == 'emission']['line_center'][:] # wave
                    noisy_wave_idx = noisy[i][noisy[i]['line_type'] == 'emission']['line_center_index'][:] # idx

                    if len(intri[i]) == 0:
                        pass
                    elif len(intri[i]) >= 1:

                        intri_wave = intri[i][intri[i]['line_type'] == 'emission']['line_center'][:] # wave
                        intri_wave_idx = intri[i][intri[i]['line_type'] == 'emission']['line_center_index'][:] # idx
                        elnumber1 = len(intri_wave)   
                        data1 = np.array([(i, list(intri_wave.value), list(intri_wave_idx), elnumber1)],dtype=dtype1)
                        intrinsic_el_list.append(data1)

                        elnumber2 = 0
                        for j in range(len(noisy_wave)):
                            for k in range(len(intri_wave)):
                                if abs(noisy_wave[j].value-intri_wave[k].value) <= threshold:
                                    elnumber2+=1
                                    data2 = np.array([(i, list(noisy_wave.value), list(noisy_wave_idx), elnumber2)],dtype=dtype2)
                                    success_detect_list.append(data2)
        intri_el_arr = np.array(intrinsic_el_list)
        intrinsic_el_arr = np.squeeze(intri_el_arr)

        success_detect_arr = np.array(success_detect_list)
        new_detect_el = []
        unique_ids = np.unique(success_detect_arr['id'])
        for i, uid in enumerate(unique_ids):
            index = np.where(success_detect_arr['id'] == uid)
            eln_value = np.max(success_detect_arr['detect_el_number'][index])
            new_detect_el.append(success_detect_arr[index][eln_value-1])
        final_detec_el = np.array(new_detect_el)

        # success_rate = len(final_detec_el)/len(intrinsic_el_arr)
        return intrinsic_el_arr, final_detec_el #, success_rate
    
    def get_el_snr(self,snr,flag,detect_el,init_wave_idx,elwidth):
        """ 
        Get the mean signal-to-noise ratio of each emission line within the given width (in index)

        parameters:
        ----------
            snr - signal-to-noise ratio array of all spectra - [array]
            flag - flags which represent whether the emission lines are detected among the all spectra
                   0 = none emission line detection, 
                   1 = only intrinsic emission line detection, 
                   2 = emission line detection in noisy spectra - [array]
            detect_el - the emission line detection imformation array from noisy spectra - [array]
            init_wave_idx - the initial wavelength index used in the 'el_detect' function - [int]
            elwidth - the width of each emission line used to extract the mean snr, in form of index, 
                      for GV should be 40, GI should be 30 - [int]
        
        return:
        ------
            snr_detect_arr - mean snr of each emission lines of each spectrum - [array]

        result example:
        --------------
            snr_detect_arr = array([list([3.122047028492964, 2.359194308387778, 1.5952671141778956, 0.6816463131378714, 0.5561439522857118, 0.3874376023249922]),
                list([48.37682725488458, 42.6838483858841]),
                list([27.431890618985765, 32.02865687740623]), 
                ...,
                list([18.601049714826956, 10.09261165162766]),
                list([11.793673108906741, 2.076331069055863]),
                list([12.288952682452512, 9.213392602860704])], dtype=object)
        """
        snr_detect_list = []
        snr_list = snr[flag==2]
        for idx1 in range(len(detect_el)):
            el_mean_snr = []
            for idx2 in range(len(detect_el['idx'][idx1])):
                elidx = init_wave_idx+detect_el['idx'][idx1][idx2]
                # in index, corresponding to ~ 200 Å width in GI band (since pixel-based dlambda = 7.6)
                el_mean_snr.append((snr_list[idx1][elidx-round(elwidth/2) : elidx+round(elwidth/2)]).mean())
            snr_detect_list.append(el_mean_snr)
        snr_detect_arr = np.array(snr_detect_list)
        return snr_detect_arr

    # def dlambda(self,grism,Res):
    #     """
    #     degrade the grism wavelength grid to the given resolution

    #     parameters
    #     ----------
    #         grism: CSST grism using pysynphot to read
    #         Res: resolution of CSST grism

    #     retrun
    #     ------
    #         lrlam: wavelength grid with CSST grism resolution
    #         dlam: the wavelength interval (delta lambda) for each wavelength point
    #     """

    #     lam = []; dlam = []
    #     lam.append(grism.wave.min())
    #     dlam.append(lam[0]/(Res-0.5))
    #     Nlam = 1

    #     for i in range(1,1000):

    #         lam.append(lam[i-1] + dlam[i-1])
    #         dlam.append(lam[i]/(Res-0.5))
    #         Nlam = Nlam + 1
    #         if lam[i]>grism.wave.max():
    #             break
    #     return np.array(lam) #, np.array(dlam)

    # def convert_dpix_to_dlam(self,grism,Res,wave,flux,fluxnoise,eflux):
        
    #     # print('\nNow converting dx to dlam of {0}:'.format(grism.name[-11:-9]))

    #     lam = self.dlambda(grism,Res)
    #     dlamall = np.tile(lam,(len(wave),1))
    #     newflux = np.ones([len(wave),len(lam)])
    #     newfluxnoise = np.ones([len(wave),len(lam)])
    #     neweflux = np.ones([len(wave),len(lam)])


    #     for i in range(len(wave)):
    #         newflux[i] = np.interp(dlamall[i],wave[i],flux[i])
    #         newfluxnoise[i] = np.interp(dlamall[i],wave[i],fluxnoise[i])
    #         neweflux[i] = np.interp(dlamall[i],wave[i],eflux[i])

    #     # print('Converting finished!')

    #     return lam, newflux, newfluxnoise, neweflux