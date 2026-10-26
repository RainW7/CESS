"""Task-owned empirical morphology state; legacy draw order and recipe retained."""
from copy import deepcopy
import numpy as np
from astropy.io import fits
from cess.config.loader import load_config
from cess.morphology import legacy as _legacy

cosmo = _legacy.cosmo
Mstar = _legacy.Mstar
et_z_025_lower = _legacy.et_z_025_lower
et_z_025_median = _legacy.et_z_025_median
et_z_025_upper = _legacy.et_z_025_upper
lt_z_025_lower = _legacy.lt_z_025_lower
lt_z_025_median = _legacy.lt_z_025_median
lt_z_025_upper = _legacy.lt_z_025_upper
et_z_075_lower = _legacy.et_z_075_lower
et_z_075_median = _legacy.et_z_075_median
et_z_075_upper = _legacy.et_z_075_upper
lt_z_075_lower = _legacy.lt_z_075_lower
lt_z_075_median = _legacy.lt_z_075_median
lt_z_075_upper = _legacy.lt_z_075_upper


class MorphologySampler:
    """Own configuration/distributions for one task; FITS data load on first draw."""

    match_input_paramters = staticmethod(_legacy.match_input_paramters)

    def __init__(self, config):
        self.configure(config)

    def configure(self, config):
        self._config = load_config(config)
        self._gems_data = None
        self._ar_unique = self._ar_probabilities = None
        self._distribution_cache = {}
        return deepcopy(self._config)

    def _ensure_config(self):
        return self._config

    def _ensure_gems(self):
        if self._gems_data is None:
            config = self._ensure_config()
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
            self._ar_unique = unique
            self._ar_probabilities = counts / len(arrays['ar'])
            # Keep the already-closed object only for legacy diagnostics. No FITS
            # descriptor or live memmap is retained; science uses owned arrays.
            arrays['catalogue'] = catalogue
            self._gems_data = arrays
        return self._gems_data

    def _n_distributions(self, magmin, magmax, deltamag):
        """Cache the legacy magnitude bins and empirical Sersic distributions.

        The nearest-bin lookup and empty-bin fallback are deliberately unchanged;
        this is a performance repair rather than a new morphology prescription.
        """
        key = (magmin, magmax, deltamag)
        if key in self._distribution_cache:
            return self._distribution_cache[key]
        data = self._ensure_gems()
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
        result = (bins, tuple(distributions))
        self._distribution_cache[key] = result
        return result

    def get_pa(self):
        """ 
        Get position angle in int form from GEMS catalog of with uniform distribution from (-90,90).

        return:
        ------
            pa - position angle in degree - [float]
        """
        return np.random.uniform(-90,90)

    def get_ar(self):
        """ 
        Get axis ratio (b/a) from GEMS catalog of following the probability distribution.

        return:
        ------
            ar - axis ratio (b/a) - [float]
        """

        self._ensure_gems()
        ar = np.random.choice(self._ar_unique, p=self._ar_probabilities)
        return ar

    def get_n_from_mag(self, mag, magmin, magmax, deltamag, nmax):
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
        bins, distributions = self._n_distributions(magmin, magmax, deltamag)
        distribution = distributions[(np.abs(bins-mag)).argmin()]
        if distribution is None:
            n = (np.random.uniform(nmax))
        else:
            unique, probabilities = distribution
            n = (np.random.choice(unique, p=probabilities))
        if n > nmax:
            n = (np.random.uniform(nmax))
        return n

    def get_n_re_from_zm(self, mag, magmin, magmax, deltamag, nmax, mstar, z):
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

        pixel_scale = self._ensure_config()['arcsecperpix']
        # get sersic index
        n = self.get_n_from_mag(mag, magmin, magmax, deltamag, nmax)

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

    def get_2d_param(self, mag, magmin, magmax, deltamag, nmax, mstar, z):
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

        n, re = self.get_n_re_from_zm(mag, magmin, magmax, deltamag, nmax, mstar, z)
        pa = self.get_pa()
        ar = self.get_ar()

        if re >= 15 and ar <= 0.4:
            ar = np.random.uniform(0.4, 1)
        if re >= 30 and ar <= 0.7:
            ar = np.random.uniform(0.7, 1)

        return n, re, pa, ar

