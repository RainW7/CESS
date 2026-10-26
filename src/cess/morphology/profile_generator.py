"""Generate a new profile-library bundle with explicit configuration and grid.

Import and --help perform no rendering or calibration-file I/O. Existing
resources are never replaced. The saved recipe describes the legacy renderer;
it does not certify its PSF, centering or all-source aperture interpretation.
"""
from collections.abc import Mapping
from copy import deepcopy
import argparse
import datetime
import hashlib
import importlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import platform
import pickle
import sys
import tempfile
from cess._version import __version__
from cess.provenance import PACKAGE_ROOT

PROJECT = PACKAGE_ROOT
AXIS_ORDER = ('n', 're_pixel', 'pa_deg', 'axis_ratio')
GENERATOR_REVISION = '20261023-v0.9.0.dev1-R1'
BUNDLE_SCHEMA = 'cess.generated_profile_library'
BUNDLE_VERSION = 1
_SOURCES = ('morphology/profile_generator.py', 'io/publish.py',
            'morphology/profile_bundle.py', 'morphology/empirical.py', 'spectra/units.py',
            'config/loader.py', 'morphology/libraries.py', 'provenance.py',
            'resources.py', '_version.py', '__init__.py')


def _project_module(name):
    """Import only the canonical package module and validate its source location.

    Only fully qualified package names are accepted. The current directory and
    top-level sys.modules entries are never used. No sys.path is changed.
    """
    canonical = name
    if not isinstance(canonical, str) or not canonical.startswith('cess.'):
        raise ValueError('Expected a canonical CESS package module: ' + str(name))
    relative = canonical[len('cess.'):].split('.')
    if not all(part.isidentifier() for part in relative):
        raise ValueError('Invalid CESS module name: ' + canonical)
    expected = PROJECT.joinpath(*relative).with_suffix('.py')
    module = importlib.import_module(canonical)
    if (getattr(module, '__name__', None) != canonical or
            Path(getattr(module, '__file__', '')).resolve() != expected):
        raise RuntimeError('Expected package module {} at {}'.format(canonical, expected))
    return module


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def validate_axes(axes):
    """Own finite increasing numeric 1D axes: n, Re[pixel], PA[deg], b/a.

    n is in GalSim's [0.3,6.2] range, Re>0 and 0<b/a<=1. No PA folding or
    Re rounding is applied to explicit grids. Booleans and strings fail.
    """
    if not isinstance(axes, Mapping) or set(axes) != set(AXIS_ORDER):
        raise ValueError('Grid requires exactly: ' + ', '.join(AXIS_ORDER))
    result = {}
    for name in AXIS_ORDER:
        original = axes[name]
        if (isinstance(original, (str, bytes)) or getattr(original, 'ndim', 1) != 1
                or hasattr(original, 'mask')):
            raise ValueError('{} must be an unmasked one-dimensional real axis'.format(name))
        try:
            raw = list(original)
        except TypeError:
            raise ValueError('{} must be a nonempty one-dimensional real axis'.format(name))
        if not raw or any(isinstance(v, bool) or not isinstance(v, Real) for v in raw):
            raise ValueError('{} requires real numeric values, excluding booleans'.format(name))
        try:
            values = [float(v) for v in raw]
        except (TypeError, ValueError, OverflowError):
            raise ValueError('{} must contain finite representable real numbers'.format(name))
        if (not all(math.isfinite(v) for v in values) or
                any(b <= a for a, b in zip(values, values[1:]))):
            raise ValueError('{} must be finite and strictly increasing'.format(name))
        if name == 'n' and not all(.3 <= v <= 6.2 for v in values):
            raise ValueError('n must lie in GalSim Sersic range [0.3, 6.2]')
        if name == 're_pixel' and any(v <= 0. for v in values):
            raise ValueError('re_pixel must be positive')
        if name == 'axis_ratio' and any(not 0. < v <= 1. for v in values):
            raise ValueError('axis_ratio must lie in (0, 1]')
        result[name] = values
    return result


def legacy_axes(pixel_scale):
    """Original angular Re grid, rounded at the supplied arcsec/pixel scale.

    At 0.074 this reproduces the original 20x20x10x10 grid. Other scales create
    a new grid; duplicate rounded radii are rejected, never silently dropped.
    """
    try:
        valid = (not isinstance(pixel_scale, bool) and isinstance(pixel_scale, Real)
                 and math.isfinite(float(pixel_scale)) and pixel_scale > 0.)
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError('pixel_scale must be positive and finite in arcsec/pixel')
    radii = [.3, .5, .7, .9, 1., 1.2, 1.4, 1.6, 1.8, 2., 2.5, 3., 3.5,
             4.5, 5., 5.5, 6., 6.5, 7., 7.4]
    return validate_axes({
        'n': [.5, .6, .7, .8, .9, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8,
              2., 2.5, 3., 3.5, 4., 4.5, 5.],
        're_pixel': [float(round(value / pixel_scale)) for value in radii],
        'pa_deg': list(range(0, 91, 10)),
        'axis_ratio': [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]})


def _configuration(config):
    module = _project_module('cess.config.loader')
    if isinstance(config, Mapping):
        effective = module.validate_config(config)
        source = {'kind': 'mapping', 'path': None, 'sha256': None}
    else:
        path = Path(os.path.expandvars(os.path.expanduser(str(config)))).resolve()
        payload = path.read_bytes()
        mapping = json.loads(payload.decode('utf-8'))
        if not isinstance(mapping, Mapping):
            raise ValueError('Configuration JSON must contain an object')
        mapping['_config_path'], mapping['_config_dir'] = str(path), str(path.parent)
        effective = module.validate_config(mapping, base_dir=path.parent)
        source = {'kind': 'json', 'path': str(path),
                  'sha256': hashlib.sha256(payload).hexdigest()}
    source['adopted_defaults'] = ({} if 'morphology_psf_fwhm' in effective else
                                  {'morphology_psf_fwhm': .2625})
    effective.setdefault('morphology_psf_fwhm', .2625)
    return effective, source


def rendering_recipe(config):
    """Record actual code conventions, including unresolved approximations."""
    return {
        'model': 'legacy_get_2d_profile',
        'physical_validation_status': 'unvalidated_legacy_rendering_recipe',
        'pixel_scale_arcsec_pixel': config['arcsecperpix'],
        'psf': {'model': 'circular_Gaussian', 'fwhm_arcsec': config['morphology_psf_fwhm'],
                'wavelength_dependence': 'none', 'parameter_role': 'adopted generator setting; not independent calibration'},
        'galaxy': 'GalSim Sersic(n, half_light_radius=pixel_scale*effective_re), shear(q=b/a,beta=PA_deg)',
        'galaxy_defaults': {'flux': 1., 'trunc': 0., 'flux_untruncated': False},
        're_definition': 'circular parent half-light radius; shear preserves area, so not elliptical semi-major half-light radius',
        'pa_definition': 'GalSim image-x angle, degrees; not a certified east-of-north sky PA',
        'radius_caps_pixel_by_python_round_n': {'5': 30, '4': 40, '3': 60, '2': 80, '1': 100},
        'radius_cap_condition': 'cap when Python round(re_pixel) exceeds bound',
        'crop_fraction_by_python_round_n': {'5': .5, '4': .6, '3': .7, '2': .8, '1': .8, 'other': .8},
        'height_crop_fraction_offset': .1,
        'rounding': 'Python round: nearest, ties to even',
        'stamp': {'size': 'GalSim automatic', 'draw_method': 'drawImage default auto',
                  'pixel_response': 'GalSim default pixel integration', 'offset_pixel': [.5, .5],
                  'dtype': 'GalSim default float32', 'center': None, 'use_true_center': True,
                  'explicit_draw_kwargs': ['wcs', 'offset'],
                  'default_contract': 'other settings are installed GalSim defaults, not explicitly passed'},
        'stamp_scale': 'expt_s * expnum * pi * (radi_cm/100)^2; cancels in normalization',
        'normalization': 'each finite-stamp marginal normalized before crop; no post-crop renormalization',
        'crop': 'extraction_height expands symmetrically about floor((length-1)/2) until fraction of finite marginal is enclosed',
        'width': {'stamp_sum_axis': 0, 'runner_use': 'dispersion broadening kernel',
                  'substitution': 'length<4: gaussian(2,1); else length>effective_re: gaussian(round(effective_re+3),round(effective_re/3))'},
        'height': {'stamp_sum_axis': 1, 'runner_use': 'spatial extraction profile',
                   'substitution': 'length<9: gaussian(4,2); else length>200: gaussian(100,30)'},
        'gaussian_function': 'utils.gaussian(length,std): exp(-x^2/(2*std^2)), x=-length..length, divided by discrete sum',
        'validation_policy': 'reject negative/nonfinite/empty/zero-sum final profiles; no clipping or renormalization',
        'per_profile_render_diagnostics': 'not recorded by legacy two-profile return API; reproduce from grid/config/code/software',
        'limitations': ['profile sums are not certified all-source aperture fractions',
                        'finite-stamp normalization, crop centering and Gaussian substitutions retain legacy approximations',
                        'FFT negative tails cause failure rather than silent repair',
                        'morphological PSF is not reconciled with the spectral instrumental LSF',
                        'grid Re can differ from effective renderer Re after caps',
                        'new grid is not automatically enabled in production runners']}


def plan_generation(config, axes):
    """Read only config; validate and describe a grid without rendering."""
    effective, source = _configuration(config)
    grid = validate_axes(axes)
    shape = [len(grid[name]) for name in AXIS_ORDER]
    count = shape[0] * shape[1] * shape[2] * shape[3]
    return {'schema': BUNDLE_SCHEMA, 'version': BUNDLE_VERSION,
            'generator_revision': GENERATOR_REVISION,
            'generator_source_layout': 'source_sha256 keys are paths relative to the cess package',
            'software_version': __version__,
            'axis_order': list(AXIS_ORDER), 'axes': grid, 'shape': shape,
            'axis_units': {'n': 'dimensionless', 're_pixel': 'pixel',
                           'pa_deg': 'degree', 'axis_ratio': 'dimensionless'},
            'profiles_per_library': count, 'total_profiles': 2 * count,
            'config_source': source, 'effective_config': effective,
            'effective_config_sha256': hashlib.sha256(_json_bytes(effective)).hexdigest(),
            'effective_config_hash_encoding': 'UTF-8 JSON sorted keys indent=2 plus LF',
            'rendering_recipe': rendering_recipe(effective)}


def _write_pickle(path, array):
    with path.open('xb') as stream:
        pickle.dump(array, stream, protocol=4)
        stream.flush()
        os.fsync(stream.fileno())


def generate_profile_libraries(config, axes, output):
    """Render a new complete bundle, atomically publishing without overwrite.

    Output's parent must exist. All 4D object-array cells are validated profiles.
    A changed source/config aborts publication. An unsupported filesystem atomic
    no-replace operation also fails; it never falls back to overwriting rename.
    No old resource is unpickled, rebuilt in place or automatically configured.
    Failures after staging creation retain that unpublished directory for
    inspection; the exception exposes its path as staging_directory. No
    pathname-based recursive cleanup risks deleting a replacement directory.
    """
    requested = Path(output).expanduser().absolute()
    output = requested.parent.resolve() / requested.name
    if os.path.lexists(str(output)):
        raise FileExistsError('Refusing to replace existing output: {}'.format(output))
    if not output.parent.is_dir():
        raise FileNotFoundError('Output parent must exist: {}'.format(output.parent))
    plan = plan_generation(config, axes)
    code = {name: _sha256(PROJECT / name) for name in _SOURCES}
    morphology = _project_module('cess.morphology.empirical')
    _project_module('cess.spectra.units')
    profiles = _project_module('cess.morphology.libraries')
    publisher = _project_module('cess.io.publish')
    verifier = _project_module('cess.morphology.profile_bundle')
    morphology.configure(plan['effective_config'])
    width, height = morphology.create_lib(*(plan['axes'][name] for name in AXIS_ORDER))
    if list(width.shape) != plan['shape'] or list(height.shape) != plan['shape']:
        raise ValueError('Rendered library shape does not match grid')
    statistics = {'width': profiles._profile_statistics(width, 'width'),
                  'height': profiles._profile_statistics(height, 'height')}
    for name, digest in code.items():
        if _sha256(PROJECT / name) != digest:
            raise RuntimeError('Source changed during generation: ' + name)
    source = plan['config_source']
    if source['path'] and _sha256(source['path']) != source['sha256']:
        raise RuntimeError('Configuration changed during generation')
    staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '.staging-', dir=str(output.parent)))
    try:
        manifest = deepcopy(plan)
        manifest.update({
            'status': 'complete',
            'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'source_sha256': code,
            'software': {'python': sys.version, 'python_executable': sys.executable,
                         'platform': sys.platform, 'machine': platform.machine(),
                         'numpy': importlib.import_module('numpy').__version__,
                         'galsim': importlib.import_module('galsim').__version__,
                         'galsim_module': importlib.import_module('galsim').__file__,
                         'galsim_default_gsparams': repr(importlib.import_module('galsim').GSParams()),
                         'astropy': importlib.import_module('astropy').__version__},
            'serialization': {'format': 'pickle', 'protocol': 4, 'trust': 'trusted local resources only'},
            'publication': {'protocol': 'atomic directory rename with no-replace',
                            'durability': 'process-failure atomicity; no power-loss guarantee'},
            'resources': {}, 'observed_profile_statistics': statistics})
        for role, name, array in (('width', 'widthlib.pkl', width), ('height', 'heightlib.pkl', height)):
            _write_pickle(staging / name, array)
            manifest['resources'][role] = {'filename': name, 'sha256': _sha256(staging / name),
                                           'size_bytes': (staging / name).stat().st_size,
                                           'shape': list(array.shape), 'array_dtype': str(array.dtype)}
        with (staging / 'manifest.json').open('xb') as stream:
            stream.write(_json_bytes(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        verifier.verify_profile_bundle(staging)
        for name, digest in code.items():
            if _sha256(PROJECT / name) != digest:
                raise RuntimeError('Source changed before publication: ' + name)
        if source['path'] and _sha256(source['path']) != source['sha256']:
            raise RuntimeError('Configuration changed before publication')
        publisher.publish_directory(staging, output)
        return manifest
    except BaseException as error:
        # A pathname identity check followed by rmtree still has a check/use
        # race. Preserve the unpublished files, with an explicit diagnostic,
        # rather than recursively removing a potentially replaced directory.
        error.staging_directory = str(staging)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description='Generate a new CESS profile library; never overwrite an existing path.')
    parser.add_argument('--config', required=True, help='Explicit CESS JSON config')
    parser.add_argument('--output', required=True, help='New bundle directory; parent must exist')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--grid', help='JSON with n, re_pixel, pa_deg and axis_ratio axes')
    group.add_argument('--legacy-grid', action='store_true', help='Explicitly request full 20x20x10x10 grid at config scale')
    parser.add_argument('--dry-run', action='store_true', help='Print validated plan without rendering or writing')
    args = parser.parse_args(argv)
    try:
        if args.legacy_grid:
            effective, _ = _configuration(args.config)
            axes = legacy_axes(effective['arcsecperpix'])
        else:
            with Path(args.grid).expanduser().open('r') as stream:
                axes = json.load(stream)
        if args.dry_run:
            print(_json_bytes(plan_generation(args.config, axes)).decode('utf-8'), end='')
        else:
            manifest = generate_profile_libraries(args.config, axes, args.output)
            print('Generated {} profiles per library: {}'.format(manifest['profiles_per_library'], Path(args.output).absolute()))
            print('Manifest: {}'.format(Path(args.output).absolute() / 'manifest.json'))
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        detail = ('\nUnpublished staging retained: ' + error.staging_directory
                  if hasattr(error, 'staging_directory') else '')
        parser.exit(2, 'Generation failed: {}{}\n'.format(error, detail))
    return 0


if __name__ == '__main__':
    main()
