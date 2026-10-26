"""Read-only integrity verification of standalone generated profile bundles.

No pickle is decoded and no generation/runtime configuration is loaded. Shapes,
counts and profile statistics are declarations checked for internal consistency,
not independently measured array properties. Import performs no file I/O.
"""
from contextlib import contextmanager
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat


_SCHEMA = 'cess.generated_profile_library'
_AXES = ('n', 're_pixel', 'pa_deg', 'axis_ratio')
_AXIS_UNITS = {'n': 'dimensionless', 're_pixel': 'pixel',
               'pa_deg': 'degree', 'axis_ratio': 'dimensionless'}
_MEMBERS = {'manifest.json', 'widthlib.pkl', 'heightlib.pkl'}
_HASH_ENCODING = 'UTF-8 JSON sorted keys indent=2 plus LF'
VERIFICATION_SCOPE = 'manifest consistency and artifact byte integrity; pickle contents not decoded'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, minimum=1):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _number(value, name, positive=False):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             '{} must be a real JSON number'.format(name))
    try:
        result = float(value)
    except (ValueError, OverflowError):
        raise ValueError('{} must be finite'.format(name))
    _require(math.isfinite(result) and (not positive or result > 0.),
             '{} must be finite{}'.format(name, ' and positive' if positive else ''))
    return result


def _digest(value, name):
    _require(isinstance(value, str) and re.fullmatch(r'[0-9a-fA-F]{64}', value) is not None,
             '{} must contain 64 hexadecimal SHA256 characters'.format(name))
    return value.lower()


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        _require(name not in result, 'Duplicate manifest JSON key: ' + name)
        result[name] = value
    return result


def _invalid_constant(value):
    raise ValueError('Nonfinite manifest JSON constant: ' + value)


def _state(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


@contextmanager
def _regular_member(directory_fd, name):
    """Open only a fixed regular member, without following a member symlink."""
    before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    _require(stat.S_ISREG(before.st_mode), 'Bundle member must be a regular non-symlink file: ' + name)
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, 'O_NONBLOCK', 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    with os.fdopen(descriptor, 'rb') as stream:
        _require(_state(os.fstat(stream.fileno())) == _state(before),
                 'Bundle member changed before reading: ' + name)
        yield stream
        after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        _require(_state(os.fstat(stream.fileno())) == _state(before) == _state(after),
                 'Bundle member changed during verification: ' + name)


def _manifest_contract(manifest):
    _require(isinstance(manifest, dict), 'Manifest must be a JSON object')
    _require(manifest.get('schema') == _SCHEMA, 'Unsupported profile bundle schema')
    _require(type(manifest.get('version')) is int and manifest['version'] == 1,
             'Unsupported profile bundle version')
    _require(manifest.get('status') == 'complete', 'Profile bundle status must be complete')
    _require(manifest.get('axis_order') == list(_AXES), 'Profile bundle axis_order must match the v1 contract')
    if 'axis_units' in manifest:
        _require(manifest['axis_units'] == _AXIS_UNITS, 'Declared axis_units disagree with the v1 physical units')
    axes = manifest.get('axes')
    _require(isinstance(axes, dict) and set(axes) == set(_AXES), 'Profile bundle axes must contain the four declared axes')
    for name in _AXES:
        values = axes[name]
        _require(isinstance(values, list) and len(values) > 0, 'Axis {} must be a nonempty one-dimensional list'.format(name))
        numeric = [_number(value, 'axis ' + name) for value in values]
        _require(all(right > left for left, right in zip(numeric, numeric[1:])),
                 'Axis {} must be strictly increasing'.format(name))
        if name == 'n':
            _require(all(.3 <= value <= 6.2 for value in numeric), 'Axis n must lie in [0.3,6.2]')
        elif name == 're_pixel':
            _require(all(value > 0. for value in numeric), 'Axis re_pixel must be positive')
        elif name == 'axis_ratio':
            _require(all(0. < value <= 1. for value in numeric), 'Axis axis_ratio must lie in (0,1]')
    shape = manifest.get('shape')
    expected_shape = [len(axes[name]) for name in _AXES]
    _require(isinstance(shape, list) and all(_integer(value) for value in shape) and shape == expected_shape,
             'Declared shape must match the four axis lengths')
    count = 1
    for value in shape:
        count *= value
    _require(_integer(manifest.get('profiles_per_library')) and manifest['profiles_per_library'] == count,
             'Declared profiles_per_library must equal the shape product')
    if 'total_profiles' in manifest:
        _require(_integer(manifest['total_profiles']) and manifest['total_profiles'] == 2 * count,
                 'Declared total_profiles must equal twice profiles_per_library')

    config = manifest.get('effective_config')
    _require(isinstance(config, dict), 'effective_config must be an object')
    _require(manifest.get('effective_config_hash_encoding') == _HASH_ENCODING,
             'Unsupported effective_config hash encoding')
    encoded = (json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    _require(hashlib.sha256(encoded).hexdigest() == _digest(manifest.get('effective_config_sha256'), 'effective_config_sha256'),
             'effective_config SHA256 disagrees with its snapshot')
    scale = _number(config.get('arcsecperpix'), 'effective_config.arcsecperpix', positive=True)
    fwhm = _number(config.get('morphology_psf_fwhm'), 'effective_config.morphology_psf_fwhm', positive=True)
    recipe = manifest.get('rendering_recipe')
    _require(isinstance(recipe, dict) and recipe.get('model') == 'legacy_get_2d_profile',
             'Unsupported rendering_recipe model')
    _require(recipe.get('physical_validation_status') == 'unvalidated_legacy_rendering_recipe',
             'rendering_recipe must retain its unvalidated physical status')
    _require(_number(recipe.get('pixel_scale_arcsec_pixel'), 'recipe pixel scale', positive=True) == scale,
             'Recipe pixel scale disagrees with effective_config')
    psf = recipe.get('psf')
    _require(isinstance(psf, dict) and psf.get('model') == 'circular_Gaussian', 'Unsupported recipe PSF model')
    _require(_number(psf.get('fwhm_arcsec'), 'recipe PSF FWHM', positive=True) == fwhm,
             'Recipe PSF FWHM disagrees with effective_config')

    snapshots = manifest.get('source_sha256')
    _require(isinstance(snapshots, dict) and bool(snapshots), 'source_sha256 must record generation-time source snapshots')
    for name, digest in snapshots.items():
        _require(isinstance(name, str) and bool(name), 'Source snapshot names must be nonempty strings')
        _digest(digest, 'source snapshot ' + name)
    resources = manifest.get('resources')
    _require(isinstance(resources, dict) and set(resources) == {'width', 'height'}, 'Manifest must declare width and height resources')
    statistics = manifest.get('observed_profile_statistics')
    _require(isinstance(statistics, dict) and set(statistics) == {'width', 'height'}, 'Manifest must declare both profile statistics')
    for role in ('width', 'height'):
        resource = resources[role]
        _require(isinstance(resource, dict), role + ' resource must be an object')
        _require(resource.get('filename') == role + 'lib.pkl', 'Unexpected resource filename for ' + role)
        _digest(resource.get('sha256'), role + ' resource SHA256')
        _require(_integer(resource.get('size_bytes')), role + ' resource size_bytes must be a positive integer')
        _require(resource.get('shape') == shape and all(_integer(v) for v in resource['shape']),
                 role + ' resource declared shape disagrees with bundle axes')
        _require(resource.get('array_dtype') == 'object', role + ' resource must declare object array_dtype')
        observed = statistics[role]
        _require(isinstance(observed, dict) and _integer(observed.get('profile_count')) and observed['profile_count'] == count,
                 role + ' statistics declared profile_count disagrees with bundle shape')
        _require(observed.get('all_profiles_valid') is True, role + ' statistics must declare all_profiles_valid=true')
        lengths = observed.get('length_range')
        _require(isinstance(lengths, list) and len(lengths) == 2 and
                 all(_integer(value) for value in lengths) and lengths[0] <= lengths[1],
                 role + ' statistics length_range is invalid')
        sums = observed.get('sum_range')
        _require(isinstance(sums, list) and len(sums) == 2, role + ' statistics sum_range is invalid')
        low, high = [_number(value, role + ' statistics sum_range', positive=True) for value in sums]
        _require(low <= high, role + ' statistics sum_range must be ordered')
        if 'gaussian_like_count' in observed:
            _require(_integer(observed['gaussian_like_count'], minimum=0) and observed['gaussian_like_count'] <= count,
                     role + ' statistics gaussian_like_count is invalid')
        if 'profile_dtype_counts' in observed:
            dtypes = observed['profile_dtype_counts']
            _require(isinstance(dtypes, dict) and bool(dtypes) and
                     all(isinstance(name, str) and name and _integer(value) for name, value in dtypes.items()) and
                     sum(dtypes.values()) == count, role + ' statistics dtype counts disagree with profile_count')
    serialization = manifest.get('serialization')
    _require(isinstance(serialization, dict) and serialization.get('format') == 'pickle' and
             type(serialization.get('protocol')) is int and serialization['protocol'] == 4,
             'Unsupported declared bundle serialization')
    return shape, count


def verify_profile_bundle(directory):
    """Check v1 manifest consistency and exact member bytes, without unpickling.

    The directory itself and its three fixed members must not be symlinks.
    Parent aliases are resolved; explicit parent traversal is rejected. Source
    hashes are generation-time snapshot metadata, never compared with current
    source files. No absolute input/config/code path in the manifest is opened.
    Return a JSON-safe summary. Invalid bundles raise ValueError or OSError.
    """
    requested = Path(directory).expanduser()
    _require('..' not in requested.parts, 'Bundle path must not contain parent traversal')
    requested = requested.absolute()
    _require(not requested.is_symlink(), 'Bundle directory must not be a symlink')
    directory = requested.parent.resolve(strict=True) / requested.name
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(str(directory), flags)
    try:
        directory_state = os.fstat(directory_fd)
        _require(set(os.listdir(directory_fd)) == _MEMBERS,
                 'Bundle must contain exactly manifest.json, widthlib.pkl and heightlib.pkl')
        member_states = {name: _state(os.stat(name, dir_fd=directory_fd, follow_symlinks=False))
                         for name in _MEMBERS}
        with _regular_member(directory_fd, 'manifest.json') as stream:
            payload = stream.read()
        manifest = json.loads(payload.decode('utf-8'), object_pairs_hook=_unique_object,
                              parse_constant=_invalid_constant)
        shape, count = _manifest_contract(manifest)
        resources = {}
        for role in ('width', 'height'):
            declared = manifest['resources'][role]
            digest, size = hashlib.sha256(), 0
            with _regular_member(directory_fd, declared['filename']) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
                    size += len(block)
            _require(size == declared['size_bytes'], role + ' resource size disagrees with actual bytes')
            _require(digest.hexdigest() == _digest(declared['sha256'], role + ' resource SHA256'),
                     role + ' resource SHA256 disagrees with actual bytes')
            resources[role] = {'filename': declared['filename'], 'sha256': digest.hexdigest(), 'size_bytes': size}
        final_directory = directory.lstat()
        _require(stat.S_ISDIR(final_directory.st_mode) and
                 (final_directory.st_dev, final_directory.st_ino) == (directory_state.st_dev, directory_state.st_ino) and
                 set(os.listdir(directory_fd)) == _MEMBERS, 'Bundle directory changed during verification')
        _require(all(_state(os.stat(name, dir_fd=directory_fd, follow_symlinks=False)) == original
                     for name, original in member_states.items()), 'Bundle members changed during verification')
        return {'verified': True, 'directory': str(directory), 'schema': _SCHEMA,
                'version': 1, 'status': 'complete', 'verification_scope': VERIFICATION_SCOPE,
                'manifest_sha256': hashlib.sha256(payload).hexdigest(),
                'declared_shape': list(shape), 'declared_profiles_per_library': count,
                'resources': resources, 'source_sha256': dict(manifest['source_sha256']),
                'source_snapshot_scope': 'generation-time metadata format checked; current source files not compared',
                'physical_validation_status': 'unvalidated_legacy_rendering_recipe'}
    finally:
        os.close(directory_fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Verify a standalone CESS profile bundle without decoding pickle contents.')
    parser.add_argument('directory', help='Generated bundle directory')
    args = parser.parse_args(argv)
    try:
        result = verify_profile_bundle(args.directory)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, 'Bundle verification failed: {}\n'.format(error))
    print(json.dumps({name: result[name] for name in
                      ('verified', 'directory', 'declared_shape', 'declared_profiles_per_library', 'verification_scope')},
                     sort_keys=True))
    return 0


if __name__ == '__main__':
    main()
