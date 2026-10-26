"""Read-only structure/accounting/byte verification of experimental MGE bundles.

No source fit, renderer, pickle, config loader or external calibration is used.
Generation code/config paths remain snapshot metadata, never opened here.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat

import h5py
import numpy as np

from cess.morphology.legacy_bundle import (_require, _number, _integer, _digest,
    _unique_object, _invalid_constant, _regular_member, _state)

SCHEMA = 'cess.positive_marginal_library'
VERSION = 1
GENERATOR_REVISION = '20261022-C11'
AXIS_ORDER = ('n', 're_pixel', 'pa_deg', 'axis_ratio')
AXIS_UNITS = dict(zip(AXIS_ORDER, ('dimensionless', 'pixel', 'degree', 'dimensionless')))
QUANTITY = 'absolute unit-source probability per bin'
ORTHOGONAL_DOMAIN = 'all real'
HASH_ENCODING = 'UTF-8 JSON sorted keys indent=2 plus LF'
ACCOUNTING_TOLERANCE = 1.e-9
CHUNK_ROWS = 256
_MEMBERS = {'manifest.json', 'marginals.hdf5'}
VERIFICATION_SCOPE = ('regular-file byte integrity, shared declarations, numeric structure, '
                      'row/fit mapping and probability accounting; no source approximation '
                      'or instrument calibration certification')


def _json(text):
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)


def _text(value):
    if isinstance(value, bytes):
        return value.decode('utf-8')
    _require(isinstance(value, str), 'Expected a text attribute')
    return value


def _numeric_list(value, name, minimum_length=1, maximum_length=None):
    _require(isinstance(value, list) and len(value) >= minimum_length and
             (maximum_length is None or len(value) <= maximum_length), 'Invalid list: ' + name)
    numbers = [_number(item, name) for item in value]
    _require(all(b > a for a, b in zip(numbers, numbers[1:])), 'List must strictly increase: ' + name)
    return numbers


def _manifest_contract(manifest):
    _require(isinstance(manifest, dict), 'Manifest must be an object')
    _require(manifest.get('schema') == SCHEMA and type(manifest.get('version')) is int and
             manifest['version'] == VERSION, 'Unsupported MGE bundle schema/version')
    _require(manifest.get('status') == 'complete' and manifest.get('generator_revision') in
             (GENERATOR_REVISION, '20261023-v0.9.0.dev1-R1'),
             'Unsupported MGE bundle status/revision')
    _require(isinstance(manifest.get('created_at_utc'), str) and bool(manifest['created_at_utc']),
             'Missing creation timestamp')
    _require(manifest.get('axis_order') == list(AXIS_ORDER) and manifest.get('axis_units') == AXIS_UNITS,
             'Wrong axis order/units')
    axes = manifest.get('axes')
    _require(isinstance(axes, dict) and set(axes) == set(AXIS_ORDER), 'Wrong declared axes')
    for name in AXIS_ORDER:
        values = _numeric_list(axes[name], 'axis ' + name)
        if name == 'n':_require(all(.5 <= v <= 5. for v in values), 'n outside [0.5,5]')
        elif name == 're_pixel':_require(all(v > 0. for v in values), 'Re must be positive')
        elif name == 'axis_ratio':_require(all(0. < v <= 1. for v in values), 'q outside (0,1]')
    shape = [len(axes[name]) for name in AXIS_ORDER]
    _require(manifest.get('shape') == shape and all(_integer(v) for v in manifest['shape']), 'Shape/axes mismatch')
    count = math.prod(shape) if hasattr(math, 'prod') else int(np.prod(shape, dtype=object))
    _require(1 <= count <= 100000 and _integer(manifest.get('profile_count')) and
             manifest['profile_count'] == count, 'Invalid declared profile count')
    for axis in ('x', 'y'):_numeric_list(manifest.get(axis + '_edges_pixel'), axis + ' edges', 2, 2050)
    _require(manifest.get('quantity') == QUANTITY and manifest.get('orthogonal_domain') == ORTHOGONAL_DOMAIN and
             manifest.get('window_renormalized') is False, 'Wrong probability definition/normalization')
    _require(manifest.get('physical_validation_status') ==
             'experimental_source_approximation_not_instrument_calibration', 'Wrong physical validation status')
    model = manifest.get('model')
    _require(isinstance(model, dict), 'Missing model')
    scale = _number(model.get('pixel_scale_arcsec_pixel'), 'model pixel scale', positive=True)
    psf = _number(model.get('psf_fwhm_arcsec'), 'model PSF', positive=True)
    _require(model.get('psf') == 'circular_Gaussian' and
             model.get('source') == 'positive_Gaussian_approximation_to_untruncated_Sersic' and
             model.get('re_definition') == 'circularized half-light radius; area-preserving shear' and
             model.get('pa_definition') == 'image-x angle in degrees', 'Invalid model definition')
    controls = manifest.get('controls')
    _require(isinstance(controls, dict) and set(controls) == {'l1_tolerance','tail_probability','max_components'},
             'Wrong controls')
    _number(controls['l1_tolerance'], 'l1_tolerance', positive=True)
    tail = _number(controls['tail_probability'], 'tail_probability')
    _require(1.e-14 <= tail <= 1.e-6 and _integer(controls['max_components']) and
             16 <= controls['max_components'] <= 160, 'Controls outside permitted range')
    _require(type(manifest.get('probability_accounting_tolerance')) in (int,float) and
             manifest['probability_accounting_tolerance'] == ACCOUNTING_TOLERANCE, 'Wrong accounting tolerance')
    config = manifest.get('effective_config')
    _require(isinstance(config, dict) and manifest.get('effective_config_hash_encoding') == HASH_ENCODING,
             'Invalid effective config snapshot')
    encoded = (json.dumps(config, indent=2, sort_keys=True, allow_nan=False)+'\n').encode('utf-8')
    _require(hashlib.sha256(encoded).hexdigest() == _digest(manifest.get('effective_config_sha256'), 'config SHA256'),
             'Effective config SHA256 mismatch')
    _require(_number(config.get('arcsecperpix'), 'config pixel scale', positive=True) == scale and
             _number(config.get('morphology_psf_fwhm'), 'config PSF', positive=True) == psf, 'Config/model scale or PSF mismatch')
    source = manifest.get('config_source')
    _require(isinstance(source, dict) and source.get('kind') in ('mapping','json') and
             isinstance(source.get('adopted_defaults'), dict), 'Invalid config_source')
    if source['kind'] == 'mapping':
        _require(source.get('path') is None and source.get('sha256') is None, 'Mapping config must not invent file identity')
    else:
        _require(isinstance(source.get('path'), str) and bool(source['path']), 'Missing config snapshot path')
        _digest(source.get('sha256'), 'config source SHA256')
    snapshots = manifest.get('source_sha256')
    _require(isinstance(snapshots, dict) and bool(snapshots), 'Missing generation code snapshots')
    for name,digest in snapshots.items():
        _require(isinstance(name,str) and bool(name), 'Invalid source snapshot name')
        _digest(digest, 'source snapshot '+name)
    _require(isinstance(manifest.get('software'),dict) and bool(manifest['software']), 'Missing software metadata')
    limitations = manifest.get('limitations')
    _require(isinstance(limitations,list) and bool(limitations) and
             all(isinstance(v,str) and bool(v) for v in limitations), 'Missing explicit limitations')
    _require(isinstance(manifest.get('observed'),dict), 'Missing observed summary')
    resource = manifest.get('resource')
    _require(isinstance(resource,dict) and resource.get('filename') == 'marginals.hdf5' and
             _integer(resource.get('size_bytes')), 'Invalid HDF5 resource declaration')
    _digest(resource.get('sha256'), 'HDF5 SHA256')
    return shape, count


def _member(group, name, kind):
    _require(isinstance(group.get(name,getlink=True), h5py.HardLink), 'External/soft HDF5 links are not permitted: '+name)
    value = group[name]
    _require(isinstance(value,kind), 'Wrong HDF5 object type: '+name)
    return value


def _dataset(group,name,shape,dtype,unit=None,order=None):
    data = _member(group,name,h5py.Dataset)
    _require(data.shape == tuple(shape) and data.dtype == np.dtype(dtype), 'Wrong shape/dtype: '+data.name)
    _require(not getattr(data,'is_virtual',False) and data.id.get_create_plist().get_external_count() == 0,
             'External/virtual HDF5 dataset storage is not permitted: '+data.name)
    for key,expected in (('unit',unit),('axis_order',order)):
        if key in data.attrs:
            _require(expected is not None and _text(data.attrs[key]) == expected, 'Wrong dataset '+key+': '+data.name)
    return data


def _fit_metadata(fit, controls, weights):
    _require(isinstance(fit,dict), 'Fit metadata must be an object')
    tolerance = controls['l1_tolerance']
    components = len(weights)
    total = _number(fit.get('normalized_weight_sum'), 'normalized_weight_sum', positive=True)
    _require(abs(total-float(weights.sum())) <= ACCOUNTING_TOLERANCE, 'Fit normalized weight sum mismatch')
    for key,expected in (('requested_estimated_L1_tolerance',tolerance),
                         ('requested_source_tail_each_end',controls['tail_probability']),
                         ('maximum_width_grid_count',controls['max_components'])):
        _require(type(fit.get(key)) in (int,float) and fit[key] == expected, 'Fit/control mismatch: '+key)
    estimate = _number(fit.get('estimated_radial_L1_including_outside'),'fit L1')
    refinement = _number(fit.get('radial_L1_refinement_difference'),'fit refinement')
    _require(0. <= estimate <= tolerance and 0. <= refinement <= tolerance*.1, 'Fit did not meet declared target')
    for key in ('source_outside_fit_domain_probability','mixture_outside_fit_domain_probability'):
        value = _number(fit.get(key),key)
        _require(0. <= value <= 1., 'Invalid fit outside probability')
    attempts = fit.get('attempts')
    _require(isinstance(attempts,list) and bool(attempts) and all(isinstance(a,dict) for a in attempts), 'Missing fit attempts')
    last = attempts[-1]
    _require(last.get('status') == 'evaluated' and _integer(last.get('active_components')) and
             last['active_components'] == components, 'Fit active component count mismatch')
    _require(_integer(last.get('width_grid_count'),minimum=16) and last['width_grid_count'] <= controls['max_components'] and
             _integer(last.get('candidate_components')) and components <= last['candidate_components'] and
             last['candidate_components'] in (last['width_grid_count'],last['width_grid_count']+1),
             'Fit component budget mismatch')
    _require(last.get('estimated_radial_L1_including_outside') == estimate and
             last.get('radial_L1_refinement_difference') == refinement, 'Final fit diagnostics mismatch')
    for key in ('source_outside_fit_domain_probability','mixture_outside_fit_domain_probability'):
        _require(last.get(key) == fit[key], 'Final fit outside metadata mismatch')
    return estimate


def _hdf_contract(handle, manifest, shape, count):
    _require(set(handle) == {'axes','bins','profiles','fits'}, 'Wrong HDF5 root keys')
    attrs=handle.attrs
    for key,expected in (('schema',SCHEMA),('status','complete'),('generator_revision',manifest['generator_revision']),
                         ('quantity',QUANTITY),('orthogonal_domain',ORTHOGONAL_DOMAIN)):
        _require(key in attrs and _text(attrs[key]) == expected, 'Shared HDF5 attribute mismatch: '+key)
    _require('version' in attrs and isinstance(attrs['version'],(int,np.integer)) and not isinstance(attrs['version'],(bool,np.bool_)) and attrs['version']==VERSION,
             'Wrong HDF5 version')
    _require(isinstance(attrs.get('window_renormalized'),(bool,np.bool_)) and not attrs['window_renormalized'], 'HDF5 normalization must be false')
    for name,expected in (('axis_order_json',list(AXIS_ORDER)),('model_json',manifest['model']),('controls_json',manifest['controls'])):
        _require(name in attrs and json.dumps(_json(_text(attrs[name])),sort_keys=True,allow_nan=False) ==
                 json.dumps(expected,sort_keys=True,allow_nan=False), 'Shared HDF5 JSON mismatch: '+name)
    _require(type(attrs.get('probability_accounting_tolerance')) is not bool and
             attrs.get('probability_accounting_tolerance') == ACCOUNTING_TOLERANCE, 'HDF5 accounting tolerance mismatch')
    axes=_member(handle,'axes',h5py.Group);bins=_member(handle,'bins',h5py.Group)
    profiles=_member(handle,'profiles',h5py.Group);fits=_member(handle,'fits',h5py.Group)
    _require(set(axes)==set(AXIS_ORDER) and set(bins)=={'x_edges_pixel','y_edges_pixel'}, 'Wrong axes/bins keys')
    for name in AXIS_ORDER:
        data=_dataset(axes,name,(len(manifest['axes'][name]),),'float64',AXIS_UNITS[name],'grid')
        _require(np.array_equal(data[:],np.asarray(manifest['axes'][name],dtype=float)), 'HDF5/manifest axis mismatch: '+name)
    for axis in ('x','y'):
        values=manifest[axis+'_edges_pixel'];data=_dataset(bins,axis+'_edges_pixel',(len(values),),'float64','pixel','edge')
        _require(np.array_equal(data[:],np.asarray(values,dtype=float)), 'HDF5/manifest edges mismatch: '+axis)
    _require(set(fits)=={'n_{:06d}'.format(i) for i in range(shape[0])}, 'Wrong fit index groups')
    estimates=[]
    for i,n in enumerate(manifest['axes']['n']):
        fit=_member(fits,'n_{:06d}'.format(i),h5py.Group)
        _require(set(fit)=={'sigma_over_re','weights'} and isinstance(fit.attrs.get('n'),(float,np.floating)) and
                 math.isfinite(float(fit.attrs['n'])) and float(fit.attrs['n']) == n, 'Fit n/index mismatch')
        sigmas=_member(fit,'sigma_over_re',h5py.Dataset)
        _require(sigmas.ndim==1 and 1<=sigmas.size<=manifest['controls']['max_components']+1, 'Invalid fit component shape')
        s=_dataset(fit,'sigma_over_re',sigmas.shape,'float64','1','component')[:]
        w=_dataset(fit,'weights',sigmas.shape,'float64','1','component')[:]
        _require(np.all(np.isfinite(s)) and np.all(s>0) and np.all(np.isfinite(w)) and np.all(w>0), 'Fit coefficients must be finite positive')
        _require(abs(float(w.sum())-1.) <= ACCOUNTING_TOLERANCE, 'Fit weights are not unit-normalized')
        _require('metadata_json' in fit.attrs, 'Missing fit metadata')
        estimates.append(_fit_metadata(_json(_text(fit.attrs['metadata_json'])),manifest['controls'],w))
    nx,ny=len(manifest['x_edges_pixel'])-1,len(manifest['y_edges_pixel'])-1
    fields={'row_indices','x_probability','y_probability','window_probability','outside_probability','source_fit_l1_estimate'}
    _require(set(profiles)==fields, 'Wrong profile dataset keys')
    rows=_dataset(profiles,'row_indices',(count,4),'int64','1','profile,parameter_axis')
    xp=_dataset(profiles,'x_probability',(count,nx),'float64','1','profile,x_bin')
    yp=_dataset(profiles,'y_probability',(count,ny),'float64','1','profile,y_bin')
    windows=_dataset(profiles,'window_probability',(count,2),'float64','1','profile,axis')
    outside=_dataset(profiles,'outside_probability',(count,2),'float64','1','profile,axis')
    l1=_dataset(profiles,'source_fit_l1_estimate',(count,),'float64','1','profile')
    for start in range(0,count,CHUNK_ROWS):
        stop=min(start+CHUNK_ROWS,count)
        expected=np.asarray(np.unravel_index(np.arange(start,stop),shape),dtype=np.int64).T
        actual=rows[start:stop]
        _require(np.array_equal(actual,expected), 'Profile rows are not C-order cartesian indices')
        x,y=xp[start:stop],yp[start:stop];window=windows[start:stop];out=outside[start:stop]
        for name,value in (('x',x),('y',y),('window',window),('outside',out)):
            upper = 1. if name in ('x','y') else 1.+ACCOUNTING_TOLERANCE
            _require(np.all(np.isfinite(value)) and np.all(value>=0.) and np.all(value<=upper), 'Invalid probability in '+name)
        summed=np.column_stack((x.sum(axis=1),y.sum(axis=1)))
        _require(np.all(np.abs(window-summed)<=ACCOUNTING_TOLERANCE), 'Stored window probability differs from bin sums')
        _require(np.all(np.abs(window+out-1.)<=ACCOUNTING_TOLERANCE), 'Window/outside probability accounting failed')
        values=l1[start:stop]
        _require(np.all(np.isfinite(values)) and np.all(values>=0.) and
                 np.array_equal(values,np.asarray(estimates)[actual[:,0]]), 'Profile/fit L1 mapping mismatch')


def verify_mge_bundle(directory):
    """Verify the fixed two-file artifact without fitting or external resource I/O.

    Reads numeric HDF5 probabilities in bounded row chunks. Generation-time
    source/config identities are metadata only, not current-path assertions.
    Invalid artifacts raise ValueError or OSError and are never modified.
    """
    requested=Path(directory).expanduser()
    _require('..' not in requested.parts,'Bundle path must not contain parent traversal')
    requested=requested.absolute()
    _require(not requested.is_symlink(),'Bundle directory must not be a symlink')
    directory=requested.parent.resolve(strict=True)/requested.name
    descriptor=os.open(str(directory),os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        directory_state=os.fstat(descriptor)
        _require(set(os.listdir(descriptor))==_MEMBERS,'Bundle requires exactly manifest.json and marginals.hdf5')
        states={name:_state(os.stat(name,dir_fd=descriptor,follow_symlinks=False)) for name in _MEMBERS}
        with _regular_member(descriptor,'manifest.json') as stream:payload=stream.read()
        manifest=_json(payload.decode('utf-8'))
        shape,count=_manifest_contract(manifest)
        declared=manifest['resource'];digest=hashlib.sha256();size=0
        with _regular_member(descriptor,'marginals.hdf5') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):
                digest.update(block);size+=len(block)
            _require(size==declared['size_bytes'] and digest.hexdigest()==_digest(declared['sha256'],'HDF5 SHA256'),
                     'HDF5 resource size/SHA256 mismatch')
            stream.seek(0)
            with h5py.File(stream,'r') as handle:_hdf_contract(handle,manifest,shape,count)
        current=directory.lstat()
        _require(stat.S_ISDIR(current.st_mode) and (current.st_dev,current.st_ino)==(directory_state.st_dev,directory_state.st_ino) and
                 set(os.listdir(descriptor))==_MEMBERS,'Bundle directory changed during verification')
        _require(all(_state(os.stat(name,dir_fd=descriptor,follow_symlinks=False))==old for name,old in states.items()),
                 'Bundle members changed during verification')
        return {'verified':True,'directory':str(directory),'schema':SCHEMA,'version':VERSION,
                'status':'complete','declared_shape':shape,'profile_count':count,
                'manifest_sha256':hashlib.sha256(payload).hexdigest(),
                'resource':{'filename':'marginals.hdf5','sha256':digest.hexdigest(),'size_bytes':size},
                'source_sha256':dict(manifest['source_sha256']),
                'verification_scope':VERIFICATION_SCOPE,
                'source_snapshot_scope':'generation-time metadata checked; current source/config files not opened',
                'physical_validation_status':manifest['physical_validation_status']}
    finally:os.close(descriptor)


def main(argv=None):
    parser=argparse.ArgumentParser(description='Verify an experimental CESS positive marginal bundle; no re-rendering.')
    parser.add_argument('directory')
    args=parser.parse_args(argv)
    try:result=verify_mge_bundle(args.directory)
    except (OSError,ValueError,TypeError,KeyError) as error:parser.exit(2,'MGE bundle verification failed: {}\n'.format(error))
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 0


if __name__=='__main__':main()
