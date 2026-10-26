"""Empirical galaxy morphology and sampled one-dimensional source profiles.

The GEMS catalogue supplies the distribution of Sersic index and axis ratio.
Re uses detector pixels, PA uses degrees, and generated profiles follow the
configured sampling and rendering conventions. Optional image rendering loads
GalSim only when requested. Original implementation: Run Wen.
"""

import numpy as np
import math
from tqdm import tqdm
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import importlib
from copy import deepcopy
from cess.config.loader import load_config
from functools import lru_cache
from numbers import Real
hubble=70
cosmo = FlatLambdaCDM(H0=hubble,Om0=0.3)

import cess.spectra.units as utils
# Configuration and calibration state are initialized only by configure() or a
# science call. Each worker should configure its task explicitly before drawing
# morphologies; this compatibility module keeps one active configuration/worker.
_config = None
_gems_data = None
_ar_unique = None
_ar_probabilities = None
c = 2.9979e8 # m/s
c_aa = 2.9979e18 # Angstrom/s
h = 6.626e-27 # erg*s


def _renderer():
    """Load the optional renderer only for rendering or explicit .galsim access.

    A module-level override remains supported for callers/tests that patch
    morphology.galsim. Sampling and precomputed profile lookup need no GalSim.
    Failed imports are not cached, and the original ImportError is chained.
    """
    if 'galsim' not in globals():
        try:
            module = importlib.import_module('galsim')
        except ImportError as error:
            raise ImportError('GalSim is an optional rendering dependency required by '
                              'get_2d_profile/create_lib. Install it in the active Python '
                              'environment to render profiles; morphology sampling and '
                              'precomputed one-dimensional/MGE libraries do not require it.') from error
        globals()['galsim'] = module
    return globals()['galsim']


def configure(config=None):
    """Set a mapping/JSON configuration and invalidate all GEMS caches.

    This performs configuration validation but no FITS I/O. The next call that
    actually needs GEMS loads independent arrays and closes the FITS handle.
    Calling configure again clears empirical distributions even for the same
    filename, so a changed resource cannot silently reuse old draws. Cache setup
    consumes no random numbers. This module's active worker configuration is
    separate from every main.emulator instance.
    """
    global _config, _gems_data, _ar_unique, _ar_probabilities
    validated = load_config(config)
    _config = validated
    _gems_data = None
    _ar_unique = None
    _ar_probabilities = None
    _n_distributions.cache_clear()
    return deepcopy(_config)


def _ensure_config():
    if _config is None:
        configure()
    return _config


def _ensure_gems():
    global _gems_data, _ar_unique, _ar_probabilities
    if _gems_data is None:
        config = _ensure_config()
        if 'gems_path' not in config:
            raise ValueError('gems_path is required for empirical morphology draws.')
        with fits.open(config['gems_path']) as catalogue:
            data = catalogue[1].data
            selected = data['SCIENCE_FLAG'] == 1
            arrays = dict(mag=np.array(data['FIT_MAG'][selected], copy=True),
                          n=np.array(data['FIT_N'][selected], copy=True),
                          ar=np.array(data['SEX_AR'][selected], copy=True))
        if arrays['mag'].size == 0:
            raise ValueError('GEMS calibration contains no SCIENCE_FLAG=1 rows.')
        unique, counts = np.unique(arrays['ar'], return_counts=True)
        _ar_unique = unique
        _ar_probabilities = counts / len(arrays['ar'])
        # Keep the already-closed object only for legacy diagnostics. No FITS
        # descriptor or live memmap is retained; science uses owned arrays.
        arrays['catalogue'] = catalogue
        _gems_data = arrays
    return _gems_data


def __getattr__(name):
    if name == 'galsim':
        return _renderer()
    aliases = {'gemsmag': 'mag', 'gemsn': 'n', 'gemsar': 'ar', 'gems_cat': 'catalogue'}
    if name in aliases:
        return _ensure_gems()[aliases[name]]
    if name == 'emulator_parameters':
        return deepcopy(_ensure_config())
    if name == 'colarea':
        return np.pi * _ensure_config()['radi'] ** 2
    if name in ('radi', 'expt', 'expnum', 'arcsecperpix', 'gems_path'):
        return _ensure_config()[name]
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


@lru_cache(maxsize=32)
def _n_distributions(magmin, magmax, deltamag):
    """Cache the legacy magnitude bins and empirical Sersic distributions.

    The nearest-bin lookup and empty-bin fallback are deliberately unchanged;
    this is a performance repair rather than a new morphology prescription.
    """
    data = _ensure_gems()
    bins = np.arange(round(magmin), round(magmax) + deltamag, deltamag)
    indices = np.digitize(data['mag'], bins)
    distributions = []
    for index in range(1, len(bins) + 1):
        values = data['n'][indices == index]
        if len(values):
            unique, counts = np.unique(values, return_counts=True)
            distributions.append((unique, counts / len(values)))
        else:
            distributions.append(None)
    return bins, tuple(distributions)

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
    height = np.asarray(height)
    if height.ndim != 1 or height.size == 0:
        raise ValueError("height must be a non-empty one-dimensional profile")
    if not np.all(np.isfinite(height)) or np.any(height < 0):
        raise ValueError("height must contain finite, non-negative fluxes")
    if not np.isfinite(fluxratio) or not 0 < fluxratio <= 1:
        raise ValueError("fluxratio must lie in (0, 1]")
    total_flux = np.sum(height)
    if total_flux <= 0:
        raise ValueError("height must have positive total flux")
    # Retain the legacy central pixel (left central pixel for even profiles),
    # but return exactly the aperture whose enclosed flux was tested.
    center = (height.size - 1) // 2
    for radius in range(max(center, height.size - center - 1) + 1):
        start = max(0, center - radius)
        stop = min(height.size, center + radius + 1)
        if np.sum(height[start:stop]) >= fluxratio * total_flux:
            return height[start:stop]
    return height


def fold_position_angle(pa):
    """Fold finite degree angles to [0, 90] for symmetric 1-D projections.

    A centered Sersic profile convolved with a circular PSF has identical
    marginal profiles at mirrored/180-degree-equivalent orientations. This is
    a lookup convention for that projection library, not a 2-D image direction
    or velocity-field orientation. Scalars return float; arrays retain shape.
    """
    angles = np.asarray(pa, dtype=float)
    if not np.all(np.isfinite(angles)):
        raise ValueError('PA must contain finite angles in degrees.')
    absolute = np.abs(angles)
    # Preserve the original precision near nearest-neighbor ties: adding and
    # subtracting 90 can round nextafter(5,+inf) back to exactly 5 degrees.
    folded = np.where(absolute <= 90., absolute, np.abs((angles + 90.) % 180. - 90.))
    return float(folded) if angles.ndim == 0 else folded


def match_input_paramters(lib,sersic,re,pa,baratio,nseries,reseries,paseries,baseries):
    '''
    match the nearest 2d parameters in the 2d profile library.

    parameters:
    ----------
        lib - 2d profile library created by 'create_lib'
        sersic - sersic index of the galaxy
        re - effective radius of the galaxy
        pa - original position angle of the galaxy, in degrees
        baratio - b/a ratio of the galaxy

    return:
    ------
        lib - the matched parameters in lib
    '''
    # Only the symmetric 0..90-degree projection library uses folded angles.
    # Signed/full-angle custom grids keep their original nearest semantics;
    # the caller's PA and random stream are never changed by this lookup.
    folded_pa = fold_position_angle(pa)
    pa_grid = np.asarray(paseries, dtype=float)
    lookup_pa = (folded_pa if np.all((pa_grid >= 0.) & (pa_grid <= 90.)) else pa)
    nidx = utils.find_nearest(nseries,sersic)
    reidx = utils.find_nearest(reseries,re)
    paidx = utils.find_nearest(paseries,lookup_pa)
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

    _ensure_gems()
    ar = np.random.choice(_ar_unique, p=_ar_probabilities)
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
    bins, distributions = _n_distributions(magmin, magmax, deltamag)
    distribution = distributions[(np.abs(bins-mag)).argmin()]
    if distribution is None:
        n = (np.random.uniform(nmax))
    else:
        unique, probabilities = distribution
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

    pixel_scale = _ensure_config()['arcsecperpix']
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
            zscale = 60/cosmo.kpc_proper_per_arcmin(z).value/pixel_scale
            re = re_kpc * zscale
        if z >= 0.5:
            re_kpc = np.random.normal(lt_z_075_median[idx],(lt_z_075_upper[idx]-lt_z_075_lower[idx])/2)
            zscale = 60/cosmo.kpc_proper_per_arcmin(z).value/pixel_scale
            re = re_kpc * zscale

    # early-type
    if n >= 2.5:
        if z < 0.5:
            if z < 0.0001: # setting a lower limit for exclude z = 0 situation
                z = 0.0001
            re_kpc = np.random.normal(et_z_025_median[idx],(et_z_025_upper[idx]-et_z_025_lower[idx])/2)
            zscale = 60/cosmo.kpc_proper_per_arcmin(z).value/pixel_scale
            re = re_kpc * zscale
        if z >= 0.5:
            re_kpc = np.random.normal(et_z_075_median[idx],(et_z_075_upper[idx]-et_z_075_lower[idx])/2)
            zscale = 60/cosmo.kpc_proper_per_arcmin(z).value/pixel_scale
            re = re_kpc * zscale

    if re <= 0.3/pixel_scale: # set a lower limit for the effective radius to avoid extremely small galaxy
        re = 0.3/pixel_scale
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

def get_2d_profile(n,re,pa,ar,centered=False):
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
        centered - optional experimental registration, bool. False preserves
            the legacy automatic stamp, half-pixel offset and float32 draw.
            True requests an odd square stamp from GalSim's good image size,
            zero offset and float64 draw so the central extraction pixel is
            the image's geometric center. This does not change PSF, source,
            truncation, extraction targets or Gaussian fallback prescriptions.
            Negative/nonfinite profiles fail explicitly; no FFT-tail repair
            or all-source aperture interpretation is implied.

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
    if not isinstance(centered,(bool,np.bool_)):
        raise ValueError('centered must be boolean.')
    galsim = _renderer()
    config = _ensure_config()
    pixel_scale = config['arcsecperpix']
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

    gal = galsim.Sersic(n, half_light_radius=pixel_scale*re)
    gal_pa = pa * galsim.degrees # PA should be in degree
    gal_ell = gal.shear(q=ar, beta=gal_pa)
    psf = galsim.Gaussian(fwhm=config.get('morphology_psf_fwhm', 0.2625)) # Assume Ree80 = 0.2"
    conv_gal = galsim.Convolve([gal_ell,psf])
    if centered:
        size = int(conv_gal.getGoodImageSize(pixel_scale))
        size += 1-size%2
        stamp = conv_gal.drawImage(nx=size,ny=size,wcs=galsim.PixelScale(pixel_scale),
                                  offset=galsim.PositionD(0.,0.),dtype=np.float64)
        stamp = stamp * config['expt'] * config['expnum'] * math.pi * (config['radi']/100)**2
    else:
        stamp = conv_gal.drawImage(wcs=galsim.PixelScale(pixel_scale), offset=offset) * config['expt'] * config['expnum'] * math.pi * (config['radi']/100)**2
        
    width = stamp.array.sum(axis=0)
    height = stamp.array.sum(axis=1)
    try:
        widthfunc = extraction_height(width/width.sum(),fluxlimit)
        heightfunc = extraction_height(height/height.sum(),fluxlimit+0.1)
    except ValueError as error:
        if centered:
            raise ValueError('Centered profile (n={}, effective_Re_pixel={}, PA_deg={}, b/a={}) '
                             'failed without clipping: {}'.format(n,re,pa,ar,error)) from error
        raise

    if len(widthfunc)<4:
        widthfunc = utils.gaussian(2,1)
    elif len(widthfunc)>re:
        widthfunc = utils.gaussian(round(re+3),round(re/3))
    if len(heightfunc)<9:
        heightfunc = utils.gaussian(4,2)
    elif len(heightfunc)>200:
        heightfunc = utils.gaussian(100,30)

    if centered:
        for role,profile in (('width',widthfunc),('height',heightfunc)):
            if (not np.all(np.isfinite(profile)) or np.any(profile<0) or
                    not np.isfinite(profile.sum()) or profile.sum()<=0):
                raise ValueError('Centered {} profile (n={}, effective_Re_pixel={}, PA_deg={}, b/a={}) '
                                 'is invalid; no clipping or renormalization applied.'.format(role,n,re,pa,ar))

    return widthfunc, heightfunc

def _profile_grid_axis(values, name):
    """Validate a lookup axis without sorting, masking or changing its values."""
    if np.ma.isMaskedArray(values):
        raise ValueError(name + ' must be unmasked.')
    try:
        raw = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as error:
        raise ValueError(name + ' must be a nonempty 1-D real numeric axis.') from error
    if (raw.ndim != 1 or raw.size == 0 or
            any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in raw)):
        raise ValueError(name + ' must be a nonempty 1-D real numeric axis, without booleans.')
    try:
        axis = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(name + ' must contain finite real numbers.') from error
    if not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
        raise ValueError(name + ' must be finite and strictly increasing.')
    return axis


def create_lib(nseries,reseries,paseries,baseries):
    """Render projection profiles into two 4-D object containers.

    Axes are nonempty, finite, real, strictly increasing 1-D numeric sequences:
    n is a positive Sersic index supported by the installed GalSim, Re is
    positive in pixels, PA is any finite degree angle, and 0 < b/a <= 1. All
    axes and the renderer's n range are checked before rendering any grid cell.
    Invalid grids raise ValueError; inputs are never sorted or silently clipped.

    Call configure(config) first to select the rendering configuration; otherwise
    get_2d_profile retains its lazy default configuration behavior. The traversal
    remains n -> Re -> PA -> b/a. Return (widthlib, heightlib), each an ndarray
    with shape (len(n),len(Re),len(PA),len(b/a)) and dtype object. Each cell holds
    one complete nonempty 1-D profile, including when all profile lengths match.

    This function does not change the PSF, stamp, cropping, fallbacks or profile
    normalization of get_2d_profile. Profile values (including numerical negative
    tails) are preserved. A publication caller must validate their finite,
    nonnegative, positive-sum contract separately before writing a library.
    """
    nseries = _profile_grid_axis(nseries, 'nseries')
    reseries = _profile_grid_axis(reseries, 'reseries')
    paseries = _profile_grid_axis(paseries, 'paseries')
    baseries = _profile_grid_axis(baseries, 'baseries')
    if np.any(nseries <= 0):
        raise ValueError('nseries must be positive.')
    if np.any(reseries <= 0):
        raise ValueError('reseries must be positive, in pixels.')
    if np.any(baseries <= 0) or np.any(baseries > 1):
        raise ValueError('baseries must satisfy 0 < b/a <= 1.')
    galsim = _renderer()
    # Use the renderer's public constructor rather than hard-coding one
    # release's allowed n range. These unit-size objects are not rendered.
    for index in (nseries[0], nseries[-1]):
        try:
            galsim.Sersic(float(index), half_light_radius=1.)
        except galsim.GalSimRangeError as error:
            raise ValueError('nseries is outside the installed GalSim Sersic range: ' + str(error)) from error
    shape = (len(nseries),len(reseries),len(paseries),len(baseries))
    widthlib, heightlib = np.empty(shape,dtype=object), np.empty(shape,dtype=object)
    for ni, n in enumerate(tqdm(nseries)):
        for ri, re in enumerate(reseries):
            for pi, pa in enumerate(paseries):
                for bi, ba in enumerate(baseries):
                    index = (ni,ri,pi,bi)
                    try:
                        width, height = get_2d_profile(n,re,pa,ba)
                    except (ValueError, galsim.GalSimError) as error:
                        raise ValueError('Profile grid {} (n={}, Re_pixel={}, PA_deg={}, b/a={}) '
                                         'failed: {}'.format(index,n,re,pa,ba,error)) from error
                    for role, profile in (('width',width),('height',height)):
                        values = np.asarray(profile)
                        if values.ndim != 1 or values.size == 0:
                            raise ValueError('{} profile {} must be a nonempty 1-D vector.'.format(role,index))
                    # Assign individual scalar object cells. np.array(list_of_
                    # equal_length_arrays,dtype=object) expands the profile axis
                    # and cannot be reshaped to the requested 4-D grid.
                    widthlib[index], heightlib[index] = width, height
    return widthlib, heightlib
