"""Deterministic result formatting shared by the two simulation runners.

No resource, RNG, detector, source-loop or HDF5 behavior belongs here. The
padding callable remains explicit so legacy runner monkeypatches keep working.
"""
import numpy as np


DESI_PARAMETER_NAMES = ('RA', 'Dec', 'z_best', 'MAG_G', 'MAG_R', 'MAG_Z',
                        'n', 'Re', 'PA', 'baratio', 'str_mass')


def pad_results(data, source_count, waves, pad_spectral_rows):
    """Pad in existing key order; only mask fields become finite-positive bools."""
    padded = {}
    for key, rows in data.items():
        wave = waves[key[:2]]
        padded['padded_' + key] = pad_spectral_rows(rows, source_count, len(wave), key)
        if key.endswith(('spec_mask', 'snr_mask')):
            values = padded['padded_' + key]
            padded['padded_' + key] = np.isfinite(values) & (values > 0)
    return padded


def emission_flags(source_count, intrinsic_ids, matched_ids):
    """Existing float64 flags: 0 absent, 1 intrinsic, 2 intrinsic and matched."""
    rows = np.arange(source_count)
    intrinsic = np.isin(rows, intrinsic_ids)
    flags = np.zeros(source_count, dtype=np.float64)
    flags[intrinsic] = 1.
    flags[intrinsic & np.isin(rows, matched_ids)] = 2.
    return flags


def desi_parameters(source_count, ra, dec, redshift, mag_g, mag_r, mag_z,
                    n, re, pa, axis_ratio, stellar_mass):
    """Keep the established eleven-column order and hstack-then-float64 cast.

    Catalogue inputs are source vectors; morphology inputs have already been
    reshaped to (N, 1) by the runner, including invalid-source NaN placeholders.
    Names describe the unchanged legacy fields rather than a new catalogue.
    """
    original = np.hstack([ra.reshape((source_count, 1)), dec.reshape((source_count, 1)),
        redshift.reshape((source_count, 1)), mag_g.reshape((source_count, 1)),
        mag_r.reshape((source_count, 1)), mag_z.reshape((source_count, 1)),
        n, re, pa, axis_ratio, stellar_mass.reshape((source_count, 1))])
    return np.array(original, dtype=np.float64), list(DESI_PARAMETER_NAMES)


def grism_parameters(source_count, rms, mean_snr, *, wave_offsets=None,
                     el_flags=None, zero_rms=None, zero_mean_snr=None,
                     zero_wave_offsets=None):
    """Stack existing GU/GV/GI statistic columns and their legacy names.

    Statistic tuples contain three (N, 1) arrays; optional EL flags contain two
    length-N vectors for GV/GI. Zero-order columns belong to the standard runner
    and enter its table only when EL flags are present. The paired runner omits
    all zero-order tuples. Keep hstack dtype inference and empty-source axes.
    """
    names, values = [], []
    for field, base, zero in (('rms_in_e', rms, zero_rms),
                              ('snr_mean', mean_snr, zero_mean_snr),
                              ('wave_off', wave_offsets, zero_wave_offsets)):
        if base is not None:
            names.extend(band + '_' + field for band in ('gu', 'gv', 'gi'))
            values.extend(base)
            if el_flags is not None and zero is not None:
                names.extend(band + '0_' + field for band in ('gu', 'gv', 'gi'))
                values.extend(zero)
    if el_flags is not None:
        names.extend(('gv_el_flag', 'gi_el_flag'))
        values.extend(flag.reshape((source_count, 1)) for flag in el_flags)
    return np.hstack(values), names
