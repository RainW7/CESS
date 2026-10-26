"""Describe trusted legacy projection libraries without inventing provenance.

Importing this module does not read resources. The caller supplies arrays already
loaded from ``parameters['widthlib_pkl']`` / ``['heightlib_pkl']`` by cached_pickle;
that array-to-file association is a caller precondition, not inferred here.
"""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import pickle

import numpy as np


_AXIS_NAMES = ('n', 're_pixel', 'pa_deg', 'axis_ratio')
_GAUSSIAN_RTOL = 1.e-5
_GAUSSIAN_LOG_ATOL = 1.e-8
_GAUSSIAN_BATCH_SIZE = 256
_GAUSSIAN_BATCH_ELEMENTS = 262144
_BUNDLE_LOOKUP = {
    'method': 'nearest on each declared axis; no interpolation',
    'outside_grid': 'nearest endpoint retained; not a calibrated extrapolation',
    'position_angle': 'fold mirrored/180-degree-equivalent PA to [0,90] before nearest lookup',
    'scientific_status': 'explicit user-selected grid; coverage and physical calibration not certified'
}


def _gaussian_like(values):
    """Observe symmetry and a concave quadratic log profile, not its origin."""
    if values.size < 5 or not np.all(values > 0.):
        return False
    peak = float(np.max(values))
    if not np.allclose(values, values[::-1], rtol=_GAUSSIAN_RTOL,
                       atol=peak * 1.e-12):
        return False
    second = np.diff(np.log(values), n=2)
    curvature = float(np.median(second))
    return bool(curvature < 0. and np.allclose(
        second, curvature, rtol=_GAUSSIAN_RTOL, atol=_GAUSSIAN_LOG_ATOL))


def _gaussian_like_batch(profiles):
    """The scalar criterion along rows of one bounded, equal-length batch.

    Inputs have already passed _profile_statistics validation and float64
    conversion. Keep the scalar short-circuit stages; neither zero-valued nor
    asymmetric rows reach the logarithm. This is a shape observation only.
    """
    values = np.asarray(profiles, dtype=np.float64)
    result = np.zeros(len(values), dtype=bool)
    if values.shape[1] < 5:
        return result
    rows = np.flatnonzero(np.all(values > 0., axis=1))
    if not rows.size:
        return result
    positive = values[rows]
    # The scalar's float(peak)*1e-12 is a Python operation, with no NumPy
    # underflow warning; retain that behavior for subnormal finite profiles.
    with np.errstate(under='ignore'):
        atol = np.max(positive, axis=1)[:, None] * 1.e-12
    symmetric = np.all(np.isclose(positive, positive[:, ::-1],
        rtol=_GAUSSIAN_RTOL, atol=atol), axis=1)
    rows = rows[symmetric]
    if not rows.size:
        return result
    second = np.diff(np.log(values[rows]), n=2, axis=1)
    curvature = np.median(second, axis=1)
    negative = curvature < 0.
    result[rows[negative]] = np.all(np.isclose(second[negative], curvature[negative, None],
        rtol=_GAUSSIAN_RTOL, atol=_GAUSSIAN_LOG_ATOL), axis=1)
    return result


def _profile_statistics(library, role):
    lengths, sums = [], []
    dtype_counts = {}
    gaussian_count = 0
    pending, pending_count, pending_elements = {}, 0, 0
    # Retain scalar exception/callback order under explicitly strict global
    # floating-point policies; normal workers use bounded batch observations.
    error_policy = np.geterr()
    scalar = (error_policy['under'] != 'ignore' or
              any(value in ('raise', 'call', 'log') for value in error_policy.values()))
    def flush():
        count = sum(int(np.count_nonzero(_gaussian_like_batch(rows))) for rows in pending.values())
        pending.clear()
        return count
    for index in np.ndindex(library.shape):
        original = library[index]
        location = '{} profile {}'.format(role, index)
        if np.ma.isMaskedArray(original):
            raise ValueError('{} must not be masked'.format(location))
        profile = np.asarray(original)
        if profile.ndim != 1 or profile.size == 0:
            raise ValueError('{} must be nonempty and one-dimensional'.format(location))
        if not (np.issubdtype(profile.dtype, np.integer) or
                np.issubdtype(profile.dtype, np.floating)):
            raise ValueError('{} must contain real numeric values'.format(location))
        if not np.all(np.isfinite(profile)):
            raise ValueError('{} must contain only finite values'.format(location))
        if np.any(profile < 0.):
            raise ValueError('{} must be nonnegative; values are not clipped'.format(location))
        # Accumulate float32 stored profiles in float64; do not alter the array.
        values = np.asarray(profile, dtype=np.float64)
        with np.errstate(over='ignore', invalid='ignore'):
            total = float(np.sum(values, dtype=np.float64))
        if not np.isfinite(total) or total <= 0.:
            raise ValueError('{} must have a positive finite sum'.format(location))
        lengths.append(int(profile.size))
        sums.append(total)
        dtype = str(profile.dtype)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        if scalar or values.size > _GAUSSIAN_BATCH_ELEMENTS:
            gaussian_count += flush()
            pending_count = pending_elements = 0
            gaussian_count += int(_gaussian_like(values))
        else:
            if (pending_count >= _GAUSSIAN_BATCH_SIZE or
                    pending_elements + values.size > _GAUSSIAN_BATCH_ELEMENTS):
                gaussian_count += flush()
                pending_count = pending_elements = 0
            pending.setdefault(values.size, []).append(values)
            pending_count += 1
            pending_elements += values.size
    gaussian_count += flush()
    return {
        'profile_count': int(library.size),
        'all_profiles_valid': True,
        'length_range': [min(lengths), max(lengths)],
        'sum_range': [min(sums), max(sums)],
        'sum_accumulation_dtype': 'float64',
        'profile_dtype_counts': dtype_counts,
        'gaussian_like_count': gaussian_count,
        'gaussian_like_criterion': {
            'description': 'positive symmetric profile with constant negative second difference of log(values)',
            'minimum_length': 5,
            'relative_tolerance': _GAUSSIAN_RTOL,
            'symmetry_absolute_tolerance_over_peak': 1.e-12,
            'log_second_difference_absolute_tolerance': _GAUSSIAN_LOG_ATOL,
            'interpretation': 'observed-shape-only; does not identify generator fallback or PSF'
        },
        'sum_interpretation': 'observed stored profile sum; not a certified all-source aperture fraction'
    }


def describe_profile_libraries(widthlib, heightlib, parameters, axes):
    """Validate and describe two existing projection libraries, with JSON-safe output.

    ``widthlib`` and ``heightlib`` are trusted local 4D ndarrays whose elements
    are nonempty, finite, nonnegative 1D real profiles with positive finite sums.
    ``axes`` maps n/re_pixel/pa_deg/axis_ratio to finite strictly increasing 1D
    arrays, in that logical order; their lengths must match both library shapes.
    ``parameters`` supplies resource paths and positive arcsecperpix (arcsec/pix).

    The function reads actual file bytes for SHA256 identity, never imports a
    generator or unpickles/rebuilds a library. It does not certify that supplied
    arrays were loaded from those paths: the caller must preserve that binding.
    Generation PSF, scale and normalization remain unknown. A configured PSF is
    recorded solely as an unbound setting of the current generator. Invalid
    input raises ValueError, identifying the role/index for a bad profile.
    """
    if not isinstance(parameters, Mapping) or not isinstance(axes, Mapping):
        raise ValueError('parameters and axes must be mappings')
    if set(axes) != set(_AXIS_NAMES):
        raise ValueError('axes must contain exactly {}'.format(', '.join(_AXIS_NAMES)))
    axis_values = {}
    for name in _AXIS_NAMES:
        raw = np.asarray(axes[name])
        if np.ma.isMaskedArray(axes[name]) or np.iscomplexobj(raw):
            raise ValueError('axis {} must be real and unmasked'.format(name))
        try:
            values = np.asarray(raw, dtype=float)
        except (TypeError, ValueError):
            raise ValueError('axis {} must contain finite numbers'.format(name))
        if (values.ndim != 1 or values.size == 0 or
                not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0.)):
            raise ValueError('axis {} must be nonempty, finite, one-dimensional and strictly increasing'.format(name))
        axis_values[name] = values.tolist()
    expected_shape = tuple(len(axis_values[name]) for name in _AXIS_NAMES)
    for role, library in (('width', widthlib), ('height', heightlib)):
        if (not isinstance(library, np.ndarray) or np.ma.isMaskedArray(library) or
                library.ndim != 4 or library.shape != expected_shape):
            raise ValueError('{} library shape must match axes {}'.format(role, expected_shape))
    try:
        scale = float(parameters['arcsecperpix'])
    except (KeyError, TypeError, ValueError):
        raise ValueError('arcsecperpix must be a positive finite number in arcsec/pixel')
    if (isinstance(parameters['arcsecperpix'], (bool, np.bool_)) or
            not np.isfinite(scale) or scale <= 0.):
        raise ValueError('arcsecperpix must be a positive finite number in arcsec/pixel')

    generator_setting = {
        'status': 'generator-setting-not-bound-to-existing-library',
        'morphology_psf_fwhm_arcsec': None,
        'provided_in_parameters': 'morphology_psf_fwhm' in parameters
    }
    if generator_setting['provided_in_parameters']:
        try:
            fwhm = float(parameters['morphology_psf_fwhm'])
        except (TypeError, ValueError):
            raise ValueError('configured morphology_psf_fwhm must be positive and finite')
        if (isinstance(parameters['morphology_psf_fwhm'], (bool, np.bool_)) or
                not np.isfinite(fwhm) or fwhm <= 0.):
            raise ValueError('configured morphology_psf_fwhm must be positive and finite')
        generator_setting['morphology_psf_fwhm_arcsec'] = fwhm

    statistics = {
        'width': _profile_statistics(widthlib, 'width'),
        'height': _profile_statistics(heightlib, 'height')
    }
    from cess.pipeline.random import file_identity
    resources = {}
    for role, library in (('width', widthlib), ('height', heightlib)):
        key = role + 'lib_pkl'
        if key not in parameters:
            raise ValueError('parameters must provide {}'.format(key))
        path = Path(parameters[key]).resolve()
        resources[role] = {
            'path': str(path),
            'sha256': file_identity(path),
            'size_bytes': int(path.stat().st_size),
            'shape': list(library.shape),
            'array_dtype': str(library.dtype),
            'array_file_binding': 'caller-supplied-array-loaded-from-this-path'
        }
    return {
        'schema': 'cess.profile_library_description',
        'version': 1,
        'provenance_status': 'legacy_unverified_generation',
        'resources': resources,
        'runtime_interpretation': {
            'status': 'explicit-lookup-contract-not-certified-generation-parameters',
            'axis_order': list(_AXIS_NAMES),
            'axes': axis_values,
            'arcsecperpix': scale,
            'arcsecperpix_unit': 'arcsec/pixel'
        },
        'generation': {
            'status': 'unknown',
            'psf': None,
            'pixel_scale_arcsec_pixel': None,
            'normalization': None
        },
        'configured_generator_setting': generator_setting,
        'observed_profile_statistics': statistics,
        'limitations': [
            'height.sum is not a certified all-source aperture fraction',
            'profile sums and Gaussian-like counts are observations, not generation provenance',
            'a configured PSF does not rebuild or change existing loaded libraries',
            'file SHA256 identifies resource bytes; the caller preserves the loaded-array/path association'
        ]
    }


def _bundle_paths(parameters):
    manifest_path = Path(parameters['profile_library_manifest_path'])
    if manifest_path.name != 'manifest.json':
        raise ValueError('profile_library_manifest_path must name manifest.json')
    directory = manifest_path.parent.resolve(strict=True)
    for role in ('width', 'height'):
        if Path(parameters[role + 'lib_pkl']).resolve() != directory / (role + 'lib.pkl'):
            raise ValueError('Configured {}lib_pkl must be the exact bundle member'.format(role))
    return directory, directory / 'manifest.json'


def _check_runtime_bundle(manifest, parameters):
    if not isinstance(manifest.get('software'), Mapping):
        raise ValueError('Runtime C7 bundle must record generation software metadata')
    if manifest['rendering_recipe']['pixel_scale_arcsec_pixel'] != parameters['arcsecperpix']:
        raise ValueError('Profile bundle pixel scale must equal config arcsecperpix')
    angles = np.asarray(manifest['axes']['pa_deg'], dtype=float)
    if np.any((angles < 0.) | (angles > 90.)):
        raise ValueError('Runtime profile bundle PA axis must lie inside [0,90] degrees')


def verified_bundle_identity(parameters):
    """Verify current member bytes before using a runtime bundle cache entry."""
    from cess.morphology.profile_bundle import verify_profile_bundle
    directory, _ = _bundle_paths(parameters)
    return verify_profile_bundle(directory)


def load_verified_profile_bundle(parameters, verification=None):
    """Load trusted C7 pickles from the exact regular-file bytes whose SHA matches.

    The optional verification is a just-obtained verify_profile_bundle result;
    every manifest/pickle read below is independently matched to its digest.
    Pickle remains trusted-local-only. Hash verification does not make an
    untrusted pickle safe. No configured path, grid coordinate or PSF is changed.
    Return (width, height, declared_axes, version-2 runtime description).
    """
    from cess.morphology.profile_bundle import _regular_member
    directory, manifest_path = _bundle_paths(parameters)
    verification = verified_bundle_identity(parameters) if verification is None else verification
    if Path(verification['directory']) != directory:
        raise ValueError('Verified profile bundle directory does not match configured paths')
    descriptor = os.open(str(directory), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        with _regular_member(descriptor, 'manifest.json') as stream:
            payload = stream.read()
        if hashlib.sha256(payload).hexdigest() != verification['manifest_sha256']:
            raise ValueError('Profile manifest changed after bundle verification')
        manifest_text = payload.decode('utf-8')
        manifest = json.loads(manifest_text)
        _check_runtime_bundle(manifest, parameters)
        buffers = {}
        for role in ('width', 'height'):
            resource = verification['resources'][role]
            with _regular_member(descriptor, role + 'lib.pkl') as stream:
                buffers[role] = stream.read()
            if (len(buffers[role]) != resource['size_bytes'] or
                    hashlib.sha256(buffers[role]).hexdigest() != resource['sha256']):
                raise ValueError('Profile {} bytes changed after bundle verification'.format(role))
    finally:
        os.close(descriptor)

    # Never reopen a pathname between hashing these bytes and decoding them.
    libraries = {role: pickle.loads(buffers[role]) for role in ('width', 'height')}
    statistics, resources = {}, {}
    shape = tuple(manifest['shape'])
    for role in ('width', 'height'):
        library = libraries[role]
        if (not isinstance(library, np.ndarray) or np.ma.isMaskedArray(library) or
                library.shape != shape or library.ndim != 4 or library.dtype != np.dtype(object)):
            raise ValueError('Decoded {} profile library shape/dtype differs from bundle grid'.format(role))
        statistics[role] = _profile_statistics(library, role)
        for name in ('profile_count', 'length_range', 'sum_range', 'profile_dtype_counts'):
            if statistics[role][name] != manifest['observed_profile_statistics'][role][name]:
                raise ValueError('Decoded {} profile statistics differ from manifest: {}'.format(role, name))
        actual = verification['resources'][role]
        resources[role] = {'path': str(directory / actual['filename']),
            'sha256': actual['sha256'], 'size_bytes': actual['size_bytes'],
            'shape': list(shape), 'array_dtype': 'object',
            'array_file_binding': 'pickle.loads of the same SHA256-verified regular-file bytes'}
        # Prevent accidental mutation of arrays reused by another task in this
        # worker. Axes/provenance are copied separately by the runner cache.
        for index in np.ndindex(library.shape):
            profile = np.asarray(library[index])
            profile.setflags(write=False)
            library[index] = profile
        library.setflags(write=False)
    recipe = manifest['rendering_recipe']
    setting = {'status': 'generator-setting-not-bound-to-existing-library',
        'morphology_psf_fwhm_arcsec': parameters.get('morphology_psf_fwhm'),
        'provided_in_parameters': 'morphology_psf_fwhm' in parameters}
    description = {
        'schema': 'cess.profile_library_description', 'version': 2,
        'provenance_status': 'manifest_verified_generation',
        'bundle_binding': {'path': str(manifest_path), 'sha256': verification['manifest_sha256'],
                           'manifest_json': manifest_text,
                           'verification_scope': verification['verification_scope']},
        'resources': resources,
        'runtime_interpretation': {'status': 'explicit-manifest-grid-not-physical-calibration',
            'axis_order': list(_AXIS_NAMES), 'axes': deepcopy(manifest['axes']),
            'arcsecperpix': parameters['arcsecperpix'], 'arcsecperpix_unit': 'arcsec/pixel',
            'lookup': deepcopy(_BUNDLE_LOOKUP)},
        'generation': {'status': 'manifest_recorded_generation',
            'physical_validation_status': recipe['physical_validation_status'],
            'psf': deepcopy(recipe['psf']),
            'pixel_scale_arcsec_pixel': recipe['pixel_scale_arcsec_pixel'],
            'normalization': recipe['normalization'], 'rendering_recipe': deepcopy(recipe),
            'source_sha256': deepcopy(manifest['source_sha256']),
            'software': deepcopy(manifest['software'])},
        'configured_generator_setting': setting,
        'observed_profile_statistics': statistics,
        'limitations': ['trusted local pickle only; byte integrity does not establish trust',
            'manifest records generation assumptions, not independent PSF or instrument calibration',
            'height.sum is not a certified all-source aperture fraction',
            'nearest endpoint selection outside a limited grid is not validated physical extrapolation',
            'the C7 legacy crop, Re caps and Gaussian substitutions remain in the declared recipe']}
    # A changed bundle cannot populate a cache with a stale current-path binding.
    verify_profile_bundle_binding(parameters, description)
    axes = {name: np.array(manifest['axes'][name], dtype=float) for name in _AXIS_NAMES}
    return libraries['width'], libraries['height'], axes, description


def verify_profile_bundle_binding(parameters, description):
    """Recheck bundle contents before publishing a science product using them."""
    if description.get('version') != 2 or description.get('provenance_status') != 'manifest_verified_generation':
        raise ValueError('Explicit runtime bundle requires version-2 profile provenance')
    verification = verified_bundle_identity(parameters)
    _, manifest_path = _bundle_paths(parameters)
    binding = description['bundle_binding']
    if (binding['path'] != str(manifest_path) or
            binding['sha256'] != verification['manifest_sha256'] or
            hashlib.sha256(binding['manifest_json'].encode('utf-8')).hexdigest() != binding['sha256']):
        raise ValueError('Profile manifest changed before science publication')
    manifest = json.loads(binding['manifest_json'])
    _check_runtime_bundle(manifest, parameters)
    for role in ('width', 'height'):
        resource = description['resources'][role]
        actual = verification['resources'][role]
        if (resource['path'] != str(manifest_path.parent / actual['filename']) or
                resource['sha256'] != actual['sha256'] or resource['size_bytes'] != actual['size_bytes']):
            raise ValueError('Profile {} resource changed before science publication'.format(role))
    return verification
