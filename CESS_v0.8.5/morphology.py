#!/Users/rain/miniconda3/envs/grizli/bin/python
# -*-coding:utf-8 -*-
'''
@Env     		:   grizli (Python 3.7.11) on Macbook Pro
@File    		:   ~/emulator/CESS_v0.8.5/morphology.py
@Time    		:   2024/11/04 16:20:10
@Author  		:   Run Wen
@Version 		:   0.8.5
@Contact 		:   wenrun@pmo.ac.cn
@Description	:   Add 2d morphological parameters to the galaxy and get 2d profile distribution functions.

Change log:
1.  set a lower limit of ar for the large Re to avoid the extremely edge-on galaxies.
'''

import numpy as np
import math
from tqdm import tqdm
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import galsim
import json
hubble=70
cosmo = FlatLambdaCDM(H0=hubble,Om0=0.3)

import utils
with open('emulator_parameters.json', 'r') as f:
    emulator_parameters = json.load(f)

radi = emulator_parameters['radi'] # telescope radius, cm
expt = emulator_parameters['expt'] # exposure time, s
expnum = emulator_parameters['expnum'] # number
arcsecperpix = emulator_parameters['arcsecperpix']# arcsec/pixel
gems_path = emulator_parameters['gems_path']
c = 2.9979e8 # m/s
c_aa = 2.9979e18 #AA/s
h = 6.626e-27 # erg*s
colarea = np.pi*radi**2 # cm^2
# gupixlen = 317
# gvpixlen = 342
# gipixlen = 332

gems_cat = fits.open(gems_path) # remember to change the path in emulator_parameters.json

mask = gems_cat[1].data['SCIENCE_FLAG'] == 1 
gemsmag = gems_cat[1].data['FIT_MAG'][mask]
gemsn = gems_cat[1].data['FIT_N'][mask]
gemsar = gems_cat[1].data['SEX_AR'][mask]

# Data from van der Wel+2014 Table 2 and Figure 8
Mstar = np.array([9.25, 9.75, 10.25, 10.75, 11.25])

et_z_025_lower = np.array([10**0.03, 10**0.04, 10**0.13, 10**0.42, 10**0.65])
et_z_025_median = np.array([10**0.27, 10**0.28, 10**0.38, 10**0.67, 10**0.76])
et_z_025_upper = np.array([10**0.46, 10**0.46, 10**0.58, 10**0.92, 10**1.08])

lt_z_025_lower = np.array([10**0.24, 10**0.36, 10**0.42, 10**0.61, 10**0.74])
lt_z_025_median = np.array([10**0.49, 10**0.61, 10**0.66, 10**0.83, 10**0.99])
lt_z_025_upper = np.array([10**0.70, 10**0.80, 10**0.85, 10**1.01, 10**1.06])

et_z_075_lower = np.array([10**-0.02, 10**-0.14, 10**0.02, 10**0.26, 10**0.62])
et_z_075_median = np.array([10**0.23, 10**0.21, 10**0.23, 10**0.45, 10**0.81])    
et_z_075_upper = np.array([10**0.43, 10**0.44, 10**0.42, 10**0.64, 10**0.97])

lt_z_075_lower = np.array([10**0.18, 10**0.32, 10**0.39, 10**0.51, 10**0.77])
lt_z_075_median = np.array([10**0.43, 10**0.56, 10**0.64, 10**0.75, 10**0.90])    
lt_z_075_upper = np.array([10**0.65, 10**0.76, 10**0.83, 10**0.90, 10**1.12])


def extraction_height(height,fluxratio):
    '''
    extract the height column pixels of slitless spectrum for each spectral resolution unit.

    parameters:
    ----------
        height - the dispersion profile distribution of the slitless spectrum - [array]
        fluxratio - the extracted spectrum flux over total spectrum flux in spatial direction - [float]

    return:
    ------
        extracted_height - the extracted dispersion profile distribution satisfying fluxratio limit
    '''
    tFlux = np.sum(height)
    y_cent_pos = int((height.shape[0]-1)/2)
    for istop in range(y_cent_pos):
        pFlux = np.sum(height[y_cent_pos-istop:y_cent_pos+istop+1])
        fluxRatio = pFlux/tFlux
        if fluxRatio>fluxratio:
            break
    return height[y_cent_pos-istop+1:y_cent_pos+istop+2]

def match_input_paramters(lib,sersic,re,pa,baratio,nseries,reseries,paseries,baseries):
    '''
    match the nearest 2d parameters in the 2d profile library.

    parameters:
    ----------
        lib - 2d profile library created by 'create_lib'
        sersic - sersic index of the galaxy
        re - effective radius of the galaxy
        pa - position angle of the galaxy
        baratio - b/a ratio of the galaxy

    return:
    ------
        lib - the matched parameters in lib
    '''
    nidx = utils.find_nearest(nseries,sersic)
    reidx = utils.find_nearest(reseries,re)
    paidx = utils.find_nearest(paseries,pa)
    baidx = utils.find_nearest(baseries,baratio)
    return lib[nidx][reidx][paidx][baidx]
    
def get_pa():
    """ 
    Get position angle in int form from GEMS catalog of with uniform distribution from (-90,90).

    return:
    ------
        pa - position angle in degree - [float]
    """
    return np.random.uniform(-90,90)

def get_ar():
    """ 
    Get axis ratio (b/a) from GEMS catalog of following the probability distribution.

    return:
    ------
        ar - axis ratio (b/a) - [float]
    """

    unique, counts = np.unique(gemsar, return_counts=True)
    probabilities = counts / len(gemsar)
    ar = np.random.choice(unique, p=probabilities)
    return ar

def get_n_from_mag(mag, magmin, magmax, deltamag, nmax):
    """ 
    Get Sersic index with a given mag from GEMS catalog, the magnitudes are divided into slices and in each slice
    the probabilities of the n is calculated to give the final sersic index.

    parameters:
    ----------
        mag - input mag - [float]
        magmin, magmax, deltamag - determine the mag slices in GEMS - [float]
        nmax - the max Sersic index for those with no data slice - [float]

    return:
    ------
        n - Sersic index of the input mag based on the probability distribution - [float]
    """

    # define the slices of mag
    if mag < round(magmin):
        mag = round(magmin)
    bins = np.arange(round(magmin), round(magmax)+deltamag, deltamag)

    # divide the mag data into slices using digitize function
    indices = np.digitize(gemsmag, bins)
    n_values=[]
    # obtain the corresponding n value of each mag slices
    for i in range(1, len(bins)+1):
        n_values.append(gemsn[indices == i])
    # jugde whether the slices contains 
    if len(n_values[(np.abs(bins-mag)).argmin()]) == 0:
        n = (np.random.uniform(nmax))

    elif len(n_values[(np.abs(bins-mag)).argmin()]) != 0:
        unique, counts = np.unique(n_values[(np.abs(bins-mag)).argmin()], return_counts=True)
        probabilities = counts / len(n_values[(np.abs(bins-mag)).argmin()])
        n = (np.random.choice(unique, p=probabilities))
    if n > nmax:
        n = (np.random.uniform(nmax))
    return n

def get_n_re_from_zm(mag, magmin, magmax, deltamag, nmax, mstar, z):
    """ 
    Get effective radius with given stellar mass following van der Wel et al. (2014) size-mass relation.

    parameters:
    ----------
        mag - input mag - [float]
        magmin, magmax, deltamag - determine the mag slices in GEMS - [float]
        nmax - the max Sersic index for those with no data slice - [float]
        mstar - stellar mass of the galaxy - [float]
        z - redshift of the galaxy - [float]

    return:
    ------
        n - Sersic index of the input mag based on the probability distribution - [float]
        re - effective radius of the galaxy in pix - [float]
    """

    # get sersic index
    n = get_n_from_mag(mag, magmin, magmax, deltamag, nmax)

    # late-type
    idx = (np.abs(Mstar-mstar)).argmin()
    if n < 2.5:
        if z < 0.5:
            # get Re in kpc following the mass-size relation of van der Wel+14
            if z < 0.0001: # setting a lower limit for exclude z = 0 situation
                z = 0.0001
            re_kpc = np.random.normal(lt_z_025_median[idx],(lt_z_025_upper[idx]-lt_z_025_lower[idx])/2)
            # get redshift evolution scale in pixel/kpc
            zscale = 60/cosmo.kpc_proper_per_arcmin(z).value/arcsecperpix
            re = re_kpc * zscale
        if z >= 0.5:
            re_kpc = np.random.normal(lt_z_075_median[idx],(lt_z_075_upper[idx]-lt_z_075_lower[idx])/2)
            zscale = 60/cosmo.kpc_proper_per_arcmin(z).value/arcsecperpix
            re = re_kpc * zscale

    # early-type
    if n >= 2.5:
        if z < 0.5:
            if z < 0.0001: # setting a lower limit for exclude z = 0 situation
                z = 0.0001
            re_kpc = np.random.normal(et_z_025_median[idx],(et_z_025_upper[idx]-et_z_025_lower[idx])/2)
            zscale = 60/cosmo.kpc_proper_per_arcmin(z).value/arcsecperpix
            re = re_kpc * zscale
        if z >= 0.5:
            re_kpc = np.random.normal(et_z_075_median[idx],(et_z_075_upper[idx]-et_z_075_lower[idx])/2)
            zscale = 60/cosmo.kpc_proper_per_arcmin(z).value/arcsecperpix
            re = re_kpc * zscale

    if re <= 0.3/0.074: # set a lower limit for the effective radius to avoid extremely small galaxy
        re = 4
    if re > 100: # set a upper limit for the effective radius to avoid extremely large galaxy
        re = 100

    return n, round(re) # in pix

def get_2d_param(mag, magmin, magmax, deltamag, nmax, mstar, z):
    """
    Get all 2d parameters, set a lower limit of ar for the large Re to avoid the extremely edge-on galaxies.

    parameters:
    ----------
        mag - input mag - [float]
        magmin, magmax, deltamag - determine the mag slices in GEMS - [float]
        nmax - the max Sersic index for those with no data slice - [float]
        mstar - stellar mass of the galaxy - [float]
        z - redshift of the galaxy - [float]

    return:
    ------
        n - Sersic index - [float]
        re - effective radius in pix - [float]
        pa - position angle in degree - [float]
        ar - axis ratio (b/a) - [float]
    """

    n, re = get_n_re_from_zm(mag, magmin, magmax, deltamag, nmax, mstar, z)
    pa = get_pa()
    ar = get_ar()

    if re >= 15 and ar <= 0.4:
        ar = np.random.uniform(0.4, 1)
    if re >= 30 and ar <= 0.7:
        ar = np.random.uniform(0.7, 1)

    return n, re, pa, ar

def get_2d_profile(n,re,pa,ar):
    """
    Get the 2D profile distribution in dispersion and spatial direction from 2D parameters.
    
    features:
    --------
        Due to the limit of galsim package, the larger n have larger profile distribution, 
    so for large n, we choose a smaller fluxlimit.
        We also set upper limits for the effective radius to avoid extremely large galaxy which 
    could causing some problems.
        The upper limit for width (spatial) distribution is set to be a bit larger than Re to avoid
    extremely smoothed slitless spectrum.
        The upper limit for height (dispersion) distribution is set to be not larger than lenght of 200.

    parameters:
    ----------
        n - Sersic index - [float]
        re - effective radius in pix - [float]
        pa - position angle in degree - [float]
        ar - axis ratio (b/a) - [float]

    return:
    ------
        widthfunc - the spatial profile distribution (y-axis, axis=0)
        heightfunc - the dispersion profile distribution (x-axis, axis=1)
    
    examples for 2D profile: 
    -----------------------
        import numpy as np
        from astropy.modeling.models import Sersic2D
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['figure.figsize'] = [8, 8]

        re = 20
        pa = np.pi
        sersic = 1
        baratio = 0.1
        fig, ax = plt.subplots(nrows=2, ncols=2)

        x,y = np.meshgrid(np.arange(np.round(5*re)), np.arange(np.round(5*re)))
        mod = Sersic2D(x_0=5*re/2,y_0=5*re/2, amplitude = 1, r_eff = re, n = sersic, ellip = (1-baratio), theta=pa)
        img = mod(x, y)
        ax[0,1].imshow(img,origin='lower',cmap='gray_r',vmin=0,vmax=2)

        width = img.sum(axis=0)
        height = img.sum(axis=1)
        x1 = np.arange(0,len(width),1)
        x2 = np.arange(0,len(height),1)
        frac = img[0][0]/img.sum()
        widthfunc = width/width.sum()
        heightfunc = height/height.sum()
        ax[0,0].plot(heightfunc,x2,c='b') #axis=1, dispersion projection, in blue, stands for height
        ax[0,0].set_title('dispersion projection, axis=1, "height"',c='blue')
        ax[1,1].plot(x1, widthfunc,c='g') #axis=0, spatial projection, in green, stands for length(width)
        ax[1,1].set_title('spatial projection, axis=0, "width"',c='green')
        ax[1,0].axis('off')
        ax[0,1].annotate('dispersion axis=1', xy=(10, 60), xytext=(50, 60), c='blue',
                    arrowprops=dict(facecolor='blue', shrink=0.00))
        ax[0,1].annotate('spatial axis=0', xy=(60, 10), xytext=(60, 40), c='green',
                    arrowprops=dict(facecolor='green', shrink=0.00))
        plt.show()
    """
    fluxlimit = 0.8
    if round(n) == 5:
        fluxlimit = 0.5
        if round(re) > 30:
            re = 30
    if round(n) == 4:
        fluxlimit = 0.6
        if round(re) > 40:
            re = 40
    elif round(n) == 3:
        fluxlimit = 0.7
        if round(re) > 60:
            re = 60
    elif round(n) == 2:
        fluxlimit = 0.8
        if round(re) > 80:
            re = 80
    elif round(n) == 1:
        fluxlimit = 0.8
        if round(re) > 100:
            re = 100

    xcenter = 300
    ycenter = 300
    x_nominal = int(np.floor(xcenter + 0.5))
    y_nominal = int(np.floor(ycenter + 0.5))
    dx = xcenter - x_nominal+0.5
    dy = ycenter - y_nominal+0.5
    offset = galsim.PositionD(dx, dy)

    gal = galsim.Sersic(n, half_light_radius=0.074*re)
    gal_pa = pa * galsim.degrees # PA should be in degree
    gal_ell = gal.shear(q=ar, beta=gal_pa)
    psf = galsim.Gaussian(fwhm=0.2625) # Assume Ree80 = 0.2"
    conv_gal = galsim.Convolve([gal_ell,psf])
    stamp = conv_gal.drawImage(wcs=galsim.PixelScale(0.074), offset=offset) * expt * expnum * math.pi * (radi/100)**2
        
    width = stamp.array.sum(axis=0)
    height = stamp.array.sum(axis=1)
    widthfunc = extraction_height(width/width.sum(),fluxlimit)
    heightfunc = extraction_height(height/height.sum(),fluxlimit+0.1)

    if len(widthfunc)<4:
        widthfunc = utils.gaussian(2,1)
    elif len(widthfunc)>re:
        widthfunc = utils.gaussian(round(re+3),round(re/3))
    if len(heightfunc)<9:
        heightfunc = utils.gaussian(4,2)
    elif len(heightfunc)>200:
        heightfunc = utils.gaussian(100,30)

    return widthfunc, heightfunc

def create_lib(nseries,reseries,paseries,baseries):
    """
    Create width and height profile distribution library, store in a 4-D array.

    parameters:
    ----------
        nseries - Sersic index series array - [array]
        reseries - effective radius series array in pix - [array]
        paseries - position angle series array in degree - [array]
        arseries - axis ratio series array (b/a) - [array]

    return:
    ------
        wflib - the 4-D spatial profile distribution array
        hflib - the 4-D dispersion profile distribution array
    """
    wfn=[]
    hfn=[]
    for i in tqdm(nseries):
        for j in (reseries):
            for k in (paseries):
                for l in (baseries):
                    widthwf, heightwf = get_2d_profile(i,j,k,l)
                    wfn.append(widthwf)
                    hfn.append(heightwf)
    wfn = np.array(wfn,dtype=object)
    hfn = np.array(hfn,dtype=object)
    wflib = np.reshape(wfn, (len(nseries),len(reseries),len(paseries),len(baseries)))
    hflib = np.reshape(hfn, (len(nseries),len(reseries),len(paseries),len(baseries)))
    return wflib, hflib