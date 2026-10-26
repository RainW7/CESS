"""Stream a standalone experimental positive-marginal library, never overwrite.

The absolute full-source probabilities are incompatible with legacy conditional
profiles. No production configuration is changed. Import performs no data I/O.
"""
import argparse
from collections.abc import Mapping
from copy import deepcopy
import datetime
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import platform
import sys
import tempfile

from cess.morphology.profile_generator import _configuration, _project_module, validate_axes
from cess._version import __version__
from cess.provenance import PACKAGE_ROOT

PROJECT = PACKAGE_ROOT
SCHEMA = 'cess.positive_marginal_library'
VERSION = 1
GENERATOR_REVISION = '20261023-v0.9.0.dev1-R1'
AXIS_ORDER = ('n', 're_pixel', 'pa_deg', 'axis_ratio')
AXIS_UNITS = dict(n='dimensionless', re_pixel='pixel', pa_deg='degree', axis_ratio='dimensionless')
_SOURCES = ('morphology/generator.py', 'morphology/bundle.py', 'morphology/mge.py',
            'morphology/reference.py', 'morphology/profile_generator.py',
            'morphology/profile_bundle.py', 'spectra/numerics.py', 'spectra/__init__.py',
            'io/publish.py', 'config/loader.py',
            'provenance.py', 'resources.py', '_version.py', '__init__.py')


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf-8')


def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def _source_hashes():
    return {name: _sha256(PROJECT/name) for name in _SOURCES}


def _controls(value):
    defaults = dict(l1_tolerance=1.e-7, tail_probability=1.e-12, max_components=160)
    if value is None:
        return defaults
    if not isinstance(value, Mapping) or set(value) != set(defaults):
        raise ValueError('controls require exactly l1_tolerance, tail_probability, max_components')
    result = dict(value)
    for name in ('l1_tolerance', 'tail_probability'):
        raw = result[name]
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise ValueError(name+' must be a positive finite real number')
        try:
            number = float(raw)
        except (ValueError, OverflowError):
            raise ValueError(name+' must be representable and finite')
        if not math.isfinite(number) or number <= 0.:
            raise ValueError(name+' must be a positive finite real number')
        result[name] = number
    if not 1.e-14 <= result['tail_probability'] <= 1.e-6:
        raise ValueError('tail_probability must lie in [1e-14,1e-6]')
    maximum = result['max_components']
    if isinstance(maximum, bool) or not isinstance(maximum, Integral) or not 16 <= maximum <= 160:
        raise ValueError('max_components must be an integer in [16,160]')
    result['max_components'] = int(maximum)
    return result


def plan_mge_library(config, axes, x_edges_pixel, y_edges_pixel=None, controls=None):
    """Validate and describe generation; no fitting, calibration reads or writes.

    CESS config supplies only pixel scale and adopted Gaussian PSF. Re is in
    pixels; PA in image-x degrees; q=b/a. Edges are relative to source center 0.
    Controls constrain numerical estimates, not a certified continuum bound.
    """
    effective, source = _configuration(config)
    grid = validate_axes(axes)
    if any(not .5 <= n <= 5. for n in grid['n']):
        raise ValueError('Positive marginal library requires 0.5<=n<=5')
    reference = _project_module('cess.morphology.reference')
    x = reference._edges(x_edges_pixel, 'x_edges_pixel')
    y = x.copy() if y_edges_pixel is None else reference._edges(y_edges_pixel, 'y_edges_pixel')
    shape = [len(grid[name]) for name in AXIS_ORDER]
    count = math.prod(shape) if hasattr(math, 'prod') else shape[0]*shape[1]*shape[2]*shape[3]
    if count > 100000:
        raise ValueError('Positive library is limited to 100000 profiles per artifact')
    design = _controls(controls)
    return {'schema': SCHEMA, 'version': VERSION, 'generator_revision': GENERATOR_REVISION,
        'generator_source_layout': 'source_sha256 keys are paths relative to the cess package',
        'software_version': __version__,
        'axis_order': list(AXIS_ORDER), 'axis_units': dict(AXIS_UNITS), 'axes': grid,
        'shape': shape, 'profile_count': count, 'x_edges_pixel': x.tolist(), 'y_edges_pixel': y.tolist(),
        'quantity': 'absolute unit-source probability per bin', 'orthogonal_domain': 'all real',
        'window_renormalized': False,
        'physical_validation_status': 'experimental_source_approximation_not_instrument_calibration',
        'model': {'pixel_scale_arcsec_pixel': effective['arcsecperpix'],
                  'psf_fwhm_arcsec': effective['morphology_psf_fwhm'], 'psf': 'circular_Gaussian',
                  'source': 'positive_Gaussian_approximation_to_untruncated_Sersic',
                  're_definition': 'circularized half-light radius; area-preserving shear',
                  'pa_definition': 'image-x angle in degrees'},
        'controls': design, 'probability_accounting_tolerance': 1.e-9,
        'effective_config': effective, 'config_source': source,
        'effective_config_sha256': hashlib.sha256(_json_bytes(effective)).hexdigest(),
        'effective_config_hash_encoding': 'UTF-8 JSON sorted keys indent=2 plus LF',
        'configuration_scope': 'only arcsecperpix and morphology_psf_fwhm consumed; remaining configuration recorded but unused',
        'estimated_numeric_payload_bytes': 8*(count*(x.size+y.size-2+9)+sum(shape)+x.size+y.size),
        'limitations': ['experimental source approximation; L1 estimates are not certified continuum bounds',
            'each axis integrates the orthogonal coordinate over all real values',
            'two marginal window masses do not determine a joint 2D aperture fraction',
            'positive source-mixture normalization precedes fitting checks; finite windows are not renormalized',
            'no legacy crop, radius cap or empirical profile substitution',
            'not compatible with legacy profile pickle schema or current runtime lookup',
            'adopted Gaussian PSF is not an instrument calibration; no velocity field',
            'working arrays streamed per profile; payload estimate excludes HDF5/fit-metadata overhead']}


def _unchanged(code, source):
    if _source_hashes() != code:
        raise RuntimeError('Source changed during positive library generation')
    if source['path'] and _sha256(source['path']) != source['sha256']:
        raise RuntimeError('Configuration changed during positive library generation')


def _dataset(group, name, *, shape=None, data=None, dtype='float64', unit='1', order):
    result = group.create_dataset(name, shape=shape, data=data, dtype=dtype)
    result.attrs['unit'], result.attrs['axis_order'] = unit, order
    return result


def _write_library(path, plan):
    import h5py
    import numpy as np
    mge = _project_module('cess.morphology.mge')
    _project_module('cess.morphology.reference')
    count = plan['profile_count']
    nx, ny = len(plan['x_edges_pixel'])-1, len(plan['y_edges_pixel'])-1
    observed = {'profile_count': 0, 'fits_count': 0, 'minimum_probability': 1.,
                'maximum_probability': 0., 'maximum_source_fit_l1_estimate': 0.,
                'x_window_probability_range': [1., 0.], 'y_window_probability_range': [1., 0.]}
    with h5py.File(str(path), 'x') as handle:
        for name in ('schema','version','generator_revision','quantity','orthogonal_domain',
                     'window_renormalized','probability_accounting_tolerance'):
            handle.attrs[name] = plan[name]
        handle.attrs['status'] = 'in_progress'
        for name, value in (('axis_order_json',plan['axis_order']),('model_json',plan['model']),
                            ('controls_json',plan['controls'])):
            handle.attrs[name] = json.dumps(value,sort_keys=True,allow_nan=False)
        axes, bins, profiles, fits = [handle.create_group(name) for name in ('axes','bins','profiles','fits')]
        for name in AXIS_ORDER:
            _dataset(axes,name,data=plan['axes'][name],unit=AXIS_UNITS[name],order='grid')
        for name in ('x_edges_pixel','y_edges_pixel'):
            _dataset(bins,name,data=plan[name],unit='pixel',order='edge')
        rows = _dataset(profiles,'row_indices',shape=(count,4),dtype='int64',order='profile,parameter_axis')
        px = _dataset(profiles,'x_probability',shape=(count,nx),order='profile,x_bin')
        py = _dataset(profiles,'y_probability',shape=(count,ny),order='profile,y_bin')
        windows = _dataset(profiles,'window_probability',shape=(count,2),order='profile,axis')
        outside = _dataset(profiles,'outside_probability',shape=(count,2),order='profile,axis')
        error = _dataset(profiles,'source_fit_l1_estimate',shape=(count,),order='profile')
        for row, index in enumerate(np.ndindex(tuple(plan['shape']))):
            n, radius, pa, q = [plan['axes'][name][i] for name,i in zip(AXIS_ORDER,index)]
            result = mge.sersic_mge_marginals(n,radius,pa,q,plan['model']['psf_fwhm_arcsec'],
                plan['model']['pixel_scale_arcsec_pixel'],plan['x_edges_pixel'],plan['y_edges_pixel'],
                **plan['controls'])
            metadata = result['metadata']
            fit = metadata['source_fit']
            fit_name = 'n_{:06d}'.format(index[0])
            if index[1:] == (0,0,0):
                group = fits.create_group(fit_name)
                group.attrs['n'] = n
                group.attrs['metadata_json'] = json.dumps(fit,sort_keys=True,allow_nan=False)
                _dataset(group,'sigma_over_re',data=result['mixture_sigma_over_re'],order='component')
                _dataset(group,'weights',data=result['mixture_weights'],order='component')
                observed['fits_count'] += 1
            rows[row] = index
            px[row], py[row] = result['x_probability'], result['y_probability']
            windows[row] = [metadata['windows'][axis]['returned_probability'] for axis in ('x','y')]
            outside[row] = [metadata['windows'][axis]['outside_probability_integrated'] for axis in ('x','y')]
            error[row] = fit['estimated_radial_L1_including_outside']
            observed['profile_count'] += 1
            observed['minimum_probability'] = min(observed['minimum_probability'],float(result['x_probability'].min()),float(result['y_probability'].min()))
            observed['maximum_probability'] = max(observed['maximum_probability'],float(result['x_probability'].max()),float(result['y_probability'].max()))
            observed['maximum_source_fit_l1_estimate'] = max(observed['maximum_source_fit_l1_estimate'],float(error[row]))
            for axis in ('x','y'):
                mass = metadata['windows'][axis]['returned_probability']
                bounds = observed[axis+'_window_probability_range']
                bounds[0], bounds[1] = min(bounds[0],mass), max(bounds[1],mass)
        handle.attrs['status'] = 'complete'
        handle.flush()
    return observed


def generate_mge_library(config, axes, x_edges_pixel, output, *, y_edges_pixel=None, controls=None):
    """Stream and publish a new complete artifact; retain any failed staging.

    Parent must exist. No current resource is replaced or auto-configured.
    File integrity and probability accounting do not certify the source/PSF.
    """
    request = Path(output).expanduser().absolute()
    destination = request.parent.resolve()/request.name
    if os.path.lexists(str(destination)):
        raise FileExistsError('Refusing to replace existing output: '+str(destination))
    if not destination.parent.is_dir():
        raise FileNotFoundError('Output parent must exist: '+str(destination.parent))
    plan = plan_mge_library(config,axes,x_edges_pixel,y_edges_pixel,controls)
    code = _source_hashes()
    publisher = _project_module('cess.io.publish')
    verifier = _project_module('cess.morphology.bundle')
    staging = Path(tempfile.mkdtemp(prefix='.'+destination.name+'.staging-',dir=str(destination.parent)))
    try:
        observed = _write_library(staging/'marginals.hdf5',plan)
        _unchanged(code,plan['config_source'])
        import h5py
        import numpy as np
        import scipy
        manifest = deepcopy(plan)
        resource = staging/'marginals.hdf5'
        manifest.update(status='complete',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            source_sha256=code,observed=observed,
            software={'python':sys.version,'python_executable':sys.executable,'numpy':np.__version__,
                      'scipy':scipy.__version__,'h5py':h5py.__version__,'machine':platform.machine()},
            resource={'filename':'marginals.hdf5','size_bytes':resource.stat().st_size,'sha256':_sha256(resource)},
            publication={'protocol':'atomic directory rename with no-replace','durability':'process-failure atomicity; no power-loss guarantee'})
        with (staging/'manifest.json').open('xb') as stream:
            stream.write(_json_bytes(manifest));stream.flush();os.fsync(stream.fileno())
        verifier.verify_mge_bundle(staging)
        _unchanged(code,plan['config_source'])
        publisher.publish_directory(staging,destination)
        return manifest
    except BaseException as error:
        error.staging_directory = str(staging)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description='Generate a standalone positive marginal library; never overwrite.')
    for name in ('config','grid','edges','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--dry-run',action='store_true',help='Validate plan only; no MGE fit or output writes')
    parser.add_argument('--l1-tolerance',type=float,default=1.e-7)
    parser.add_argument('--tail-probability',type=float,default=1.e-12)
    parser.add_argument('--max-components',type=int,default=160)
    args = parser.parse_args(argv)
    try:
        grid = json.loads(Path(args.grid).read_text())
        edges = json.loads(Path(args.edges).read_text())
        if not isinstance(edges,dict) or 'x_edges_pixel' not in edges or set(edges)-{'x_edges_pixel','y_edges_pixel'}:
            raise ValueError('edges JSON requires x_edges_pixel and optional y_edges_pixel only')
        controls = dict(l1_tolerance=args.l1_tolerance,tail_probability=args.tail_probability,max_components=args.max_components)
        if args.dry_run:
            result = plan_mge_library(args.config,grid,edges['x_edges_pixel'],edges.get('y_edges_pixel'),controls)
        else:
            result = generate_mge_library(args.config,grid,edges['x_edges_pixel'],args.output,
                                         y_edges_pixel=edges.get('y_edges_pixel'),controls=controls)
    except (OSError,ValueError,TypeError,RuntimeError) as error:
        retained = getattr(error,'staging_directory',None)
        suffix = '\nRetained unpublished staging: '+retained if retained else ''
        parser.exit(2,'Positive library generation failed: {}{}\n'.format(error,suffix))
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    return 0


if __name__=='__main__':
    main()
