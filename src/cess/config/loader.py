"""Explicit CESS configuration loading, validation and resource-path resolution.

Importing this module performs no configuration or calibration-file I/O. Resource
paths are resolved relative to the JSON file, not the process working directory.
Resource existence is checked by the code that actually uses each resource.
"""

from collections.abc import Mapping
from copy import deepcopy
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
from cess.resources import default_config_path
from cess.instrument.spectral import normalise_spectral_configuration


DEFAULT_CONFIG_PATH = default_config_path()
_RUNTIME_PATH_FIELDS = ('input_path', 'output_path', 'el_input_path', 'el_output_path')
_INSTRUMENT_FIELDS = ('radi', 'expt', 'expnum', 'arcsecperpix', 'dc_value')
_BAND_FIELDS = ('start_wl', 'end_wl', 'min_wl', 'max_wl')


def _absolute_path(value, base_dir):
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise ValueError('Resource paths must be nonempty strings or path-like values.')
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _number(config, name, minimum=0., strictly_positive=False, integer=False):
    value = config[name]
    if (isinstance(value, bool) or not isinstance(value, Real) or
            not math.isfinite(float(value))):
        raise ValueError('{} must be a finite numeric value.'.format(name))
    if value < minimum or (strictly_positive and value <= minimum):
        relation = 'greater than' if strictly_positive else 'at least'
        raise ValueError('{} must be {} {}.'.format(name, relation, minimum))
    if integer and int(value) != value:
        raise ValueError('{} must be an integer.'.format(name))
    config[name] = int(value) if integer else float(value)


def validate_config(mapping, base_dir=None):
    """Return an independent validated dict without reading any data files.

    A full instrument/grism mapping is required; extra entries are retained for
    runner and science extensions. Numeric values use the units of the existing
    JSON: radius cm, exposure s, pixel scale arcsec/pixel, dark current e-/s/pixel,
    read noise e-/pixel/exposure and wavelengths Angstrom. Documented defaults
    cover read noise, fixed-template background, the legacy photometric
    exposure/aperture design and an explicitly unknown template LSF. Effective
    values are returned and recorded in provenance. They are assumptions, not
    validated CSST specifications. Template LSF must be specified independently
    of wavelength sampling: unknown/delta have no value, resolving_power uses a
    positive dimensionless value, and fwhm_rest uses a positive FWHM in rest-frame
    Angstrom. A broader-than-target template is governed by error/preserve;
    neither policy authorizes silent sharpening or variance clamping. Explicit
    central_resolving_power instruments derive a constant observed FWHM from
    each band's central_wl/Res; omission preserves historical dlambda controls.
    Explicit resolution-element sampling separates pixel spacing from that LSF
    and treats min/max as integration edges. Bandpass analysis windows are
    resolved on actual spectral centers; old mappings retain explicit indexes.

    Paths named *_path or *_pkl resolve against base_dir, or existing
    _config_dir/_config_path metadata, or the bundled data directory for a
    standalone mapping. _config_path is None when no source JSON is known.
    An optional profile_library_manifest_path explicitly selects a trusted C7
    bundle at runtime; omission preserves legacy lookup. Its two *_pkl paths
    must still be supplied and are checked against that manifest by the loader.
    """
    if not isinstance(mapping, Mapping):
        raise TypeError('CESS configuration must be a mapping.')
    config = deepcopy(dict(mapping))
    required = list(_INSTRUMENT_FIELDS)
    required.extend(band + '_' + name for band in ('gu', 'gv', 'gi') for name in _BAND_FIELDS)
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError('Missing required CESS configuration fields: ' + ', '.join(missing))
    if base_dir is None:
        base_dir = config.get('_config_dir')
        if base_dir is None and config.get('_config_path'):
            base_dir = Path(config['_config_path']).expanduser().resolve().parent
        if base_dir is None:
            base_dir = DEFAULT_CONFIG_PATH.parent
    base_dir = Path(base_dir).expanduser().resolve()
    for name in ('radi', 'expt', 'arcsecperpix'):
        _number(config, name, strictly_positive=True)
    _number(config, 'expnum', minimum=1, integer=True)
    _number(config, 'dc_value')
    config.setdefault('read_noise', 5.)
    _number(config, 'read_noise')
    config.setdefault('background_mode', 'fixed_template')
    if config['background_mode'] not in ('fixed_template', 'legacy_radec'):
        raise ValueError('background_mode must be fixed_template or legacy_radec.')
    config.setdefault('background_scale', 1.)
    _number(config, 'background_scale')
    config.setdefault('background_v_mag', 22.1)
    _number(config, 'background_v_mag', minimum=-math.inf)
    template_lsf = config.setdefault('template_lsf', {'mode': 'unknown'})
    if not isinstance(template_lsf, Mapping):
        raise ValueError('template_lsf must be a mapping containing an explicit mode.')
    template_lsf = dict(template_lsf)
    mode = template_lsf.get('mode')
    if not isinstance(mode, str) or mode not in (
            'unknown', 'delta', 'resolving_power', 'fwhm_rest'):
        raise ValueError('template_lsf.mode must be unknown, delta, resolving_power or fwhm_rest.')
    required_lsf_keys = {'mode', 'value'} if mode in (
        'resolving_power', 'fwhm_rest') else {'mode'}
    if set(template_lsf) != required_lsf_keys:
        raise ValueError('template_lsf mode {} requires exactly these keys: {}'.format(
            mode, ', '.join(sorted(required_lsf_keys))))
    if 'value' in template_lsf:
        try:
            _number(template_lsf, 'value', strictly_positive=True)
        except ValueError as error:
            raise ValueError('template_lsf.' + str(error)) from error
    config['template_lsf'] = template_lsf
    policy = config.setdefault('template_lsf_broader_policy', 'error')
    if not isinstance(policy, str) or policy not in ('error', 'preserve'):
        raise ValueError('template_lsf_broader_policy must be error or preserve.')
    designs = {
        'photometry_exposures': {'nuv': 4, 'u': 2, 'g': 2, 'r': 2, 'i': 2, 'z': 2, 'y': 4},
        'zero_order_exposures': {'GU': 2, 'GV': 2, 'GI': 2},
    }
    for name, defaults in designs.items():
        design = config.setdefault(name, defaults)
        if not isinstance(design, Mapping) or set(design) != set(defaults):
            raise ValueError('{} must specify exactly these bands: {}'.format(name, ', '.join(defaults)))
        design = dict(design)
        for band in defaults:
            _number(design, band, minimum=1, integer=True)
        config[name] = design
    for name, default in (('photometry_aperture_fraction', .8),
                          ('photometry_aperture_radius_re', 15.),
                          ('photometry_systematic_mag', .02),
                          ('photometry_detection_snr', 5.)):
        config.setdefault(name, default)
        _number(config, name, strictly_positive=name != 'photometry_systematic_mag')
    if config['photometry_aperture_fraction'] > 1:
        raise ValueError('photometry_aperture_fraction must not exceed 1.')
    line_design = config.setdefault('line_measurement', {
        'line_width_pixels': {'GV': 40, 'GI': 40},
        'sideband_width_pixels': 20, 'sideband_gap_pixels': 0})
    line_keys = {'line_width_pixels', 'sideband_width_pixels', 'sideband_gap_pixels'}
    if not isinstance(line_design, Mapping) or set(line_design) != line_keys:
        raise ValueError('line_measurement requires line_width_pixels, '
                         'sideband_width_pixels and sideband_gap_pixels.')
    line_design = deepcopy(dict(line_design))
    widths = line_design['line_width_pixels']
    if not isinstance(widths, Mapping) or set(widths) != {'GV', 'GI'}:
        raise ValueError('line_measurement.line_width_pixels requires GV and GI.')
    widths = dict(widths)
    for band in ('GV', 'GI'):
        _number(widths, band, minimum=1, integer=True)
    line_design['line_width_pixels'] = widths
    _number(line_design, 'sideband_width_pixels', minimum=2, integer=True)
    _number(line_design, 'sideband_gap_pixels', integer=True)
    config['line_measurement'] = line_design
    extraction = config.setdefault('spectral_extraction', {'mode': 'legacy_full'})
    if not isinstance(extraction, Mapping):
        raise ValueError('spectral_extraction must be a mapping with an explicit mode.')
    extraction = dict(extraction)
    extraction_mode = extraction.get('mode')
    if not isinstance(extraction_mode, str) or extraction_mode not in ('legacy_full', 'explicit', 'mge_strip'):
        raise ValueError('spectral_extraction.mode must be legacy_full, explicit or mge_strip.')
    extraction_keys = ({'mode'} if extraction_mode == 'legacy_full' else
                       {'mode', 'spatial_pixels', 'kernel_radius_pixels', 'lsf_convention'}
                       if extraction_mode == 'mge_strip' else
                       {'mode', 'source_fraction', 'spatial_pixels'})
    if set(extraction) != extraction_keys:
        raise ValueError('spectral_extraction mode {} requires exactly: {}'.format(
            extraction_mode, ', '.join(sorted(extraction_keys))))
    if extraction_mode == 'explicit':
        _number(extraction, 'source_fraction')
        if extraction['source_fraction'] > 1.:
            raise ValueError('spectral_extraction.source_fraction must not exceed 1.')
        _number(extraction, 'spatial_pixels', minimum=1, integer=True)
    if extraction_mode == 'mge_strip':
        for name, minimum, maximum in (('spatial_pixels', 1, 1001), ('kernel_radius_pixels', 0, 1024)):
            if isinstance(extraction[name], bool) or not isinstance(extraction[name], Integral):
                raise ValueError('spectral_extraction.{} must be an integer.'.format(name))
            _number(extraction, name, minimum=minimum, integer=True)
            if extraction[name] > maximum:
                raise ValueError('spectral_extraction.{} must be <= {}.'.format(name, maximum))
        if extraction['lsf_convention'] not in ('point_source', 'pre_psf'):
            raise ValueError('spectral_extraction.lsf_convention must explicitly be point_source or pre_psf.')
    positive_keys = ('positive_profile_library_manifest_path', 'positive_profile_library_path')
    positive_count = sum(name in config for name in positive_keys)
    if (extraction_mode == 'mge_strip' and positive_count != 2) or (
            extraction_mode != 'mge_strip' and positive_count):
        raise ValueError('Both positive profile-library file paths are required if and only if mode=mge_strip.')
    if positive_count and 'profile_library_manifest_path' in config:
        raise ValueError('C7 profile_library_manifest_path cannot coexist with mge_strip source-basis paths.')
    config['spectral_extraction'] = extraction
    normalise_spectral_configuration(config)
    for band in ('gu', 'gv', 'gi'):
        for suffix in ('start_wl', 'end_wl', 'min_wl', 'max_wl', 'dlambda'):
            _number(config, band + '_' + suffix, strictly_positive=True)
        for suffix in ('init_idx', 'end_idx'):
            _number(config, band + '_' + suffix, integer=True)
        if not (config[band + '_start_wl'] <= config[band + '_min_wl'] <
                config[band + '_max_wl'] <= config[band + '_end_wl']):
            raise ValueError('{} requires start <= min < max <= end wavelength.'.format(band))
        if config[band + '_init_idx'] >= config[band + '_end_idx']:
            raise ValueError('{} requires init_idx < end_idx.'.format(band))
        if config[band + '_dlambda'] > config[band + '_end_wl'] - config[band + '_start_wl']:
            raise ValueError('{} dlambda must provide at least two spectral samples.'.format(band))
        for suffix in ('Res', 'ree80'):
            if band + '_' + suffix in config:
                _number(config, band + '_' + suffix, strictly_positive=True)
    for name in ('morphology_psf_fwhm', 'noisy_el_detect_th', 'intri_el_detect_th'):
        if name in config:
            _number(config, name, strictly_positive=True)
    if 'el_detect_success_th' in config:
        _number(config, 'el_detect_success_th')
    for name in ('gv_elwidth', 'gi_elwidth'):
        if name in config:
            _number(config, name, minimum=1, integer=True)
    for name, value in list(config.items()):
        if not name.startswith('_') and (name.endswith('_path') or name.endswith('_pkl')):
            config[name] = _absolute_path(value, base_dir)
    if ('profile_library_manifest_path' in config and
            Path(config['profile_library_manifest_path']).name != 'manifest.json'):
        raise ValueError('profile_library_manifest_path must point to manifest.json.')
    if positive_count and (Path(config[positive_keys[0]]).name != 'manifest.json' or
                           Path(config[positive_keys[1]]).name != 'marginals.hdf5'):
        raise ValueError('Positive source-basis paths must name manifest.json and marginals.hdf5.')
    config['_config_dir'] = str(base_dir)
    source = config.get('_config_path')
    config['_config_path'] = _absolute_path(source, base_dir) if source else None
    return config


def load_config(path=None):
    """Load and validate a JSON path, mapping, CESS_CONFIG, or the bundled JSON.

    Precedence is explicit path/mapping, then a nonempty CESS_CONFIG environment
    variable, then DEFAULT_CONFIG_PATH. Explicit relative JSON paths are relative
    to the caller's cwd; resources inside them are relative to that JSON's folder.
    For the bundled JSON only (also when explicitly selected), the four runtime
    input/output directories use the caller's cwd. Instrument resources always
    remain relative to package data. This avoids writing/searching site-packages;
    no directories are created and effective absolute paths are recorded.
    Every call returns its own dict, including independent nested extra entries.
    """
    if isinstance(path, Mapping):
        return validate_config(path)
    if path is None:
        path = os.environ.get('CESS_CONFIG') or DEFAULT_CONFIG_PATH
    source = Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()
    with source.open('r') as stream:
        mapping = json.load(stream)
    if not isinstance(mapping, Mapping):
        raise ValueError('CESS JSON configuration must contain an object/mapping.')
    if source == DEFAULT_CONFIG_PATH:
        for name in _RUNTIME_PATH_FIELDS:
            if name in mapping:
                mapping[name] = _absolute_path(mapping[name], Path.cwd())
    mapping['_config_path'] = str(source)
    mapping['_config_dir'] = str(source.parent)
    return validate_config(mapping, base_dir=source.parent)
