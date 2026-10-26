"""Vectorized source safety checks; never repair or write the input catalogue.

These deliberately broad limits guard the fields used by the 1D runner, rather
than selecting a galaxy population. They do not certify upstream model quality.
Input fnu is observed-normalized microJy; both redshifts are dimensionless.
"""
import json

import numpy as np


POLICY_VERSION = 'cess.input-quality.v1'
REQUIRED_PARAMETERS = ('RA', 'DEC', 'MAG_G', 'MAG_R', 'MAG_Z',
                       'log(M*)[0,1]_{MAL}', 'z_{MAL}')
ACTIVE_PARAMETERS = ('RA', 'DEC', 'MAG_Z', 'log(M*)[0,1]_{MAL}', 'z_{MAL}')
FLAG_MEANINGS = {
    1: 'nonfinite_parameter', 2: 'ra_out_of_range', 4: 'dec_out_of_range',
    8: 'magnitude_out_of_safety_range', 16: 'logmass_out_of_safety_range',
    32: 'morphology_redshift_out_of_range', 64: 'invalid_input_sed',
    128: 'spectral_redshift_out_of_range', 256: 'insufficient_spectral_support',
}
POLICY = {
    'version': POLICY_VERSION,
    'required_parameters': list(REQUIRED_PARAMETERS),
    'active_parameters': list(ACTIVE_PARAMETERS),
    'unchecked_passthrough_parameters': ['MAG_G', 'MAG_R'],
    'inclusive_ranges': {'RA_deg': [0., 360.], 'DEC_deg': [-90., 90.],
                         'MAG_Z': [-100., 100.], 'log10_mass_solar': [-5., 20.],
                         'z_MAL': [1.e-8, 10.], 'best_fit_z': [1.e-8, 10.]},
    'sed': 'finite, nonnegative observed fnu, absolute value <= 1e100 (existing operator contract)',
    'support': 'all GU/GV/GI bands must have the required wavelength/LSF support',
    'action': 'preserve ID/order/raw catalogue values; skip simulation; data_mask=2; simulated values NaN',
    'scope': 'broad numerical safety guards, not population cuts or scientific calibration; no equality assumed between z_MAL and best_fit_z',
}


def parameter_quality_flags(parameters):
    """Return uint32 flags (n_source,) from an Astropy table or column mapping.

    RA/DEC are degrees, magnitudes are AB, mass is log10(M*/Msun). Missing
    required columns are a file-contract error; bad row values are flagged.
    """
    missing = [name for name in REQUIRED_PARAMETERS if name not in parameters.keys()]
    if missing:
        raise ValueError('Missing required input parameters: ' + ', '.join(missing))
    # MAG_G/R are copied to the output table, not used by the forward model.
    # Missing/sentinel values there must not discard an otherwise usable SED.
    values = np.column_stack([np.asarray(parameters[name], dtype=float)
                              for name in ACTIVE_PARAMETERS])
    flags = np.zeros(values.shape[0], dtype=np.uint32)
    flags[~np.isfinite(values).all(axis=1)] |= np.uint32(1)
    def outside(array, lower, upper):
        return np.isfinite(array) & ((array < lower) | (array > upper))
    for column, lower, upper, bit in ((0, 0., 360., 2), (1, -90., 90., 4),
            (2, -100., 100., 8), (3, -5., 20., 16), (4, 1.e-8, 10., 32)):
        flags[outside(values[:, column], lower, upper)] |= np.uint32(bit)
    return flags


def input_quality_flags(parameters, flux, redshift):
    """Check source-aligned metadata and SED values before response convolution.

    flux must be (n_source,n_wave), redshift (n_source,). Shared wavelength
    validity is a file-contract error checked by the existing spectral operator.
    No inputs are modified. Bit 256 is added after the operator's support check.
    """
    flags = parameter_quality_flags(parameters)
    flux, redshift = np.asarray(flux), np.asarray(redshift)
    if flux.ndim != 2 or flux.shape[0] != flags.size or redshift.shape != flags.shape:
        raise ValueError('Input quality expects source-aligned parameters, flux (N,W) and redshift (N,).')
    invalid_sed = (~np.isfinite(flux).all(axis=1) | (flux < 0).any(axis=1) |
                   (np.abs(flux) > 1.e100).any(axis=1))
    flags[invalid_sed] |= np.uint32(64)
    flags[~np.isfinite(redshift) | (redshift < 1.e-8) | (redshift > 10.)] |= np.uint32(128)
    return flags


def write_input_quality(handle, flags):
    """Write one source-aligned bitmask, with self-contained policy and legend."""
    dataset = handle.create_dataset('input_quality_flags', data=np.asarray(flags, dtype=np.uint32))
    dataset.attrs['unit'] = '1'
    dataset.attrs['axis_order'] = 'source'
    dataset.attrs['policy_version'] = POLICY_VERSION
    dataset.attrs['flag_masks'] = np.array(list(FLAG_MEANINGS), dtype=np.uint32)
    dataset.attrs['flag_meanings'] = ' '.join(FLAG_MEANINGS.values())
    dataset.attrs['policy_json'] = json.dumps(POLICY, sort_keys=True)
    return dataset
