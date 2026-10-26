"""Compatibility and orchestration helpers for simulation runners.

I/O implementations live under cess.io. Profile loading stays here so existing
cached_pickle monkeypatches are resolved at call time. Resource cache identity
is shared with cess.io.cache; importing performs no resource/configuration I/O.
"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import os
import sys
import tempfile
import json
import pickle
import traceback
import h5py
import numpy as np
from scipy.interpolate import interp1d

from cess.contracts import (SCHEMA_VERSION, REPAIR_REVISION, COVARIANCE_MODEL,
    MGE_MORPHOLOGY_FIELDS, MGE_MORPHOLOGY_ATTRIBUTES, LINE_MEASUREMENT_DTYPE,
    OPTION_DEFAULTS)
from cess.io.cache import _RESOURCE_CACHE, _resource_key, cached_table, cached_pickle
from cess.io.selection import _SelectedDataset, SourceSelection, simulation_filename
from cess.io.atomic import output_destination, atomic_hdf5
from cess.io.legacy_writer import (write_spectral_extraction, write_mge_morphology,
    write_response_products, _PHOTOMETRY_NUMERIC, empty_photometry_result,
    write_photometry_results, pad_spectral_rows, write_vlen_rows, write_el_details,
    write_spectral_covariance, write_line_measurements, _spectral_provenance,
    _record_spectral_geometry, record_runner_provenance)


_MORPHOLOGY_CONFIGURATION = None

def normalise_options(options=None, defaults=None):
    """Accept the canonical switch names and the legacy CLI aliases."""
    result = dict(OPTION_DEFAULTS if defaults is None else defaults)
    aliases = {'morph': 'apply_morphology_effect',
               'wave_cal': 'apply_wavelength_calibration_error',
               'photo': 'apply_photometric', 'el_detect': 'apply_el_detect'}
    for name, value in ({} if options is None else dict(options)).items():
        name = aliases.get(name, name)
        if name not in OPTION_DEFAULTS:
            raise ValueError('Unknown runner option: {}'.format(name))
        if isinstance(value, str):
            if value.lower() not in ('true', 'yes', '1', 'false', 'no', '0'):
                raise ValueError('Invalid boolean option: {}'.format(value))
            value = value.lower() in ('true', 'yes', '1')
        result[name] = bool(value)
    return result

def profile_library_provenance(parameters, axes, widthlib=None, heightlib=None):
    """Describe loaded profile resources without inventing generation metadata.

    Runtime lookup axes are assumptions of this reader, separate from the
    unknown historical PSF, pixel scale and normalization. Cache identity and
    unchanged size/mtime checks tie the caller's loaded arrays to the resources
    being hashed; final publication checks those hashes again.
    """
    from cess.morphology.libraries import describe_profile_libraries
    keys = tuple(_resource_key('pickle', parameters[name])
                 for name in ('widthlib_pkl', 'heightlib_pkl'))
    current_width = cached_pickle(parameters['widthlib_pkl'])
    current_height = cached_pickle(parameters['heightlib_pkl'])
    if ((widthlib is not None and widthlib is not current_width) or
            (heightlib is not None and heightlib is not current_height)):
        raise ValueError('Profile resources changed after the runner loaded its arrays')
    signature = json.dumps({name: np.asarray(value).tolist() for name, value in axes.items()}, sort_keys=True)
    key = ('profile_provenance', keys, signature, parameters['arcsecperpix'],
           parameters.get('morphology_psf_fwhm'))
    if key not in _RESOURCE_CACHE:
        description = describe_profile_libraries(current_width, current_height, parameters, axes)
        after = tuple(_resource_key('pickle', parameters[name])
                      for name in ('widthlib_pkl', 'heightlib_pkl'))
        if after != keys:
            raise ValueError('Profile resources changed while their provenance was being recorded')
        _RESOURCE_CACHE[key] = description
    return deepcopy(_RESOURCE_CACHE[key])

def load_profile_libraries(parameters, legacy_axes):
    """Return two cached trusted libraries, their actual lookup axes and provenance.

    Omitted manifest preserves the existing legacy path. Explicit C7 bundles
    are rehashed before each cache use; their cached arrays originate from the
    same verified bytes passed to pickle.loads, never the legacy pickle cache.
    """
    if 'profile_library_manifest_path' not in parameters:
        width = cached_pickle(parameters['widthlib_pkl'])
        height = cached_pickle(parameters['heightlib_pkl'])
        description = profile_library_provenance(parameters, legacy_axes,
                                                 widthlib=width, heightlib=height)
        return width, height, legacy_axes, description
    from cess.morphology.libraries import verified_bundle_identity, load_verified_profile_bundle
    verified = verified_bundle_identity(parameters)
    key = ('verified_profile_bundle', str(Path(parameters['profile_library_manifest_path']).resolve()),
           verified['manifest_sha256'],
           tuple(verified['resources'][role]['sha256'] for role in ('width', 'height')),
           parameters['arcsecperpix'], parameters.get('morphology_psf_fwhm'))
    if key not in _RESOURCE_CACHE:
        _RESOURCE_CACHE[key] = load_verified_profile_bundle(parameters, verification=verified)
    width, height, axes, description = _RESOURCE_CACHE[key]
    return width, height, {name: values.copy() for name, values in axes.items()}, deepcopy(description)

def resolve_spectral_extraction(parameters, height):
    """Return explicit source/area assumptions; height sum is diagnostic only."""
    profile = np.asarray(height, dtype=float)
    if (profile.ndim != 1 or profile.size == 0 or np.any(~np.isfinite(profile)) or
            np.any(profile < 0) or not np.isfinite(profile.sum()) or profile.sum() <= 0):
        raise ValueError('Extraction height must be a nonempty finite nonnegative profile with positive sum')
    design = parameters['spectral_extraction']
    if design['mode'] == 'legacy_full':
        fraction, pixels = 1., int(profile.size)
    elif design['mode'] == 'explicit':
        fraction, pixels = design['source_fraction'], design['spatial_pixels']
    else:
        raise ValueError('Unknown spectral extraction mode')
    return dict(source_fraction=float(fraction), spatial_pixels=int(pixels),
                profile_height_sum=float(profile.sum()))

def configure_morphology(parameters):
    """Reuse lazy GEMS state between tasks with the same worker configuration."""
    import cess.morphology.legacy as morphology
    global _MORPHOLOGY_CONFIGURATION
    signature = json.dumps(parameters, sort_keys=True)
    if signature != _MORPHOLOGY_CONFIGURATION:
        morphology.configure(parameters)
        _MORPHOLOGY_CONFIGURATION = signature

def legacy_main(task, configure, filenames, emission_lines=False):
    """Compatibility adapter; parse arguments only when explicitly executed.

    The new cess_cli module provides the production bounded worker pool.
    This legacy file-range adapter is deliberately serial and memory-bounded.
    """
    if len(sys.argv) < 3:
        raise SystemExit('Usage: {} begin end [morph=true wave_cal=true photo=true '
                         'el_detect=true config=/path output_dir=/path seed=123]'.format(sys.argv[0]))
    begin, end = int(sys.argv[1]), int(sys.argv[2])
    arguments = dict(argument.split('=', 1) for argument in sys.argv[3:])
    options = {key: arguments.pop(key) for key in tuple(arguments)
               if key in ('morph', 'wave_cal', 'photo', 'el_detect')}
    parameters = configure(arguments.pop('config', None), options)
    output = arguments.pop('output_dir', None)
    seed = arguments.pop('seed', None)
    seed = None if seed is None else int(seed)
    if arguments:
        raise ValueError('Unknown legacy arguments: {}'.format(sorted(arguments)))
    path = parameters['el_input_path' if emission_lines else 'input_path']
    failed = False
    for filename in filenames[begin:end]:
        try:
            task(path, filename, output_dir=output, config=parameters,
                 options=options, seed=seed)
        except Exception:
            traceback.print_exc()
            failed = True
    if not failed:
        print('This part of processes is END!')
    return 1 if failed else 0

def throughput_arrays(table):
    """Return native Angstrom/fractional-response arrays for either FITS schema.

    A missing FITS unit retains the established Angstrom input contract. Values
    outside the native wavelength coverage are never invented by this helper.
    """
    names = {name.lower(): name for name in table.colnames}
    wave_name = names.get('wavelength')
    value_name = names.get('efficiency', names.get('sensitivity'))
    if wave_name is None or value_name is None:
        raise ValueError('Throughput requires wavelength and efficiency/sensitivity columns')
    wave = np.asarray(table[wave_name], dtype=float)
    response = np.asarray(table[value_name], dtype=float)
    if (wave.ndim != 1 or response.shape != wave.shape or wave.size < 2 or
            not np.all(np.isfinite(wave)) or not np.all(np.isfinite(response)) or
            np.any(wave <= 0) or not np.all(np.diff(wave) > 0) or
            np.any(response < 0) or np.any(response > 1)):
        raise ValueError('Throughput columns must be finite, aligned, positive/increasing '
                         'in wavelength, and have fractional response between 0 and 1')
    return wave, response

def throughput_interpolator(table):
    """Interpolate a validated native throughput without out-of-range padding."""
    wave, response = throughput_arrays(table)
    return interp1d(wave, response)

def measure_runner_band(emulator, ra, dec, wave, flux, response_table,
                        background_table, parameters, n_exposures, re, rng=None):
    """Measure a full response support using one consistent aperture/noise model."""
    from cess.photometry.measurement import measure_photometry
    response_wave, response = throughput_arrays(response_table)
    positive = np.flatnonzero(response > 0)
    if positive.size == 0:
        raise ValueError('Photometric response has no positive support')
    lower = max(0, int(positive[0]) - 1)
    upper = min(len(response) - 1, int(positive[-1]) + 1)
    response_wave = response_wave[lower:upper + 1]
    response = response[lower:upper + 1]
    bkg_wave = np.asarray(background_table['wavelength'], dtype=float)
    if response_wave[0] < bkg_wave[0] or response_wave[-1] > bkg_wave[-1]:
        return empty_photometry_result('incomplete_background_coverage', 64)
    # Use the same native, ordered-left/right-limit sky integral as spectra.
    sky_rate = emulator.native_sky_rate(ra, dec, bkg_wave, background_table['fnu'],
                                       response_wave, response)
    return measure_photometry(
        wave, flux, response_wave, response,
        area_cm2=np.pi * parameters['radi'] ** 2,
        exposure_s=parameters['expt'], n_exposures=n_exposures,
        aperture_fraction=parameters.get('photometry_aperture_fraction', .8),
        n_pixels=np.pi * (parameters.get('photometry_aperture_radius_re', 15.) * re) ** 2,
        sky_rate_e_s_pixel=sky_rate, dark_current_e_s_pixel=parameters['dc_value'],
        read_noise_e_pixel=parameters.get('read_noise', 5.), rng=rng,
        systematic_mag=parameters.get('photometry_systematic_mag', .02),
        detection_snr=parameters.get('photometry_detection_snr', 5.))

def combine_band_masks(band_masks, n_sources):
    """Require validity in every band; all other rows receive invalid status."""
    valid = np.ones(n_sources, dtype=bool)
    for good, bad in band_masks:
        good = np.asarray(good, dtype=bool)
        bad = np.asarray(bad, dtype=bool)
        if good.shape != (n_sources,) or bad.shape != (n_sources,):
            raise ValueError('Each band mask must have one entry per input ID')
        valid &= good & ~bad
    return valid, ~valid

def join_processes(processes):
    """Wait for every worker and return a nonzero CLI status on any failure."""
    failed = []
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failed.append((getattr(process, 'name', 'worker'), process.exitcode))
    for name, exitcode in failed:
        print('{} failed with exit code {}'.format(name, exitcode), file=sys.stderr)
    return 1 if failed else 0

