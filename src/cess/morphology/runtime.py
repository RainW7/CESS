"""Explicit C11 source-basis loading and aperture-conditioned runtime lookup.

No stored x/y marginals are multiplied. Kernels are derived from the verified
Gaussian source coefficients and an explicitly declared aperture/LSF model.
Import has no data/config I/O. Production defaults do not select this backend.
"""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np

from cess.morphology.bundle import verify_mge_bundle
from cess.morphology.profile_bundle import _regular_member

AXES = ('n','re_pixel','pa_deg','axis_ratio')
_LIBRARIES = OrderedDict()
_KERNELS = OrderedDict()
_LOOKUP = {'method':'nearest on each declared axis; no interpolation',
    'outside_grid':'nearest endpoint; quantization/extrapolation uncertainty is not in the fit L1 estimate',
    'pa':'fold to [0,90] only for axes contained in that interval; otherwise use original PA'}


def _paths(settings):
    manifest = Path(settings['positive_profile_library_manifest_path']).resolve()
    data = Path(settings['positive_profile_library_path']).resolve()
    if manifest.name != 'manifest.json' or data != manifest.parent/'marginals.hdf5':
        raise ValueError('Positive profile paths must name the exact C11 manifest/HDF5 members')
    if 'profile_library_manifest_path' in settings:
        raise ValueError('Positive source basis and C7 profile manifest are mutually exclusive')
    if settings['spectral_extraction']['mode'] != 'mge_strip':
        raise ValueError('Positive source basis requires spectral_extraction.mode=mge_strip')
    return manifest, data


def _identity(settings):
    manifest, data = _paths(settings)
    result = verify_mge_bundle(manifest.parent)
    descriptor = os.open(str(manifest.parent),os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        with _regular_member(descriptor,'manifest.json') as stream:
            payload = stream.read()
        if hashlib.sha256(payload).hexdigest() != result['manifest_sha256']:
            raise ValueError('Positive manifest changed after verification')
    finally:
        os.close(descriptor)
    declaration = json.loads(payload.decode('utf-8'))
    model = declaration['model']
    if (model['pixel_scale_arcsec_pixel'] != settings['arcsecperpix'] or
            model['psf_fwhm_arcsec'] != settings.get('morphology_psf_fwhm',.2625)):
        raise ValueError('Positive library pixel scale/PSF must equal the configured adopted model')
    return manifest, data, result, payload.decode('utf-8'), declaration


def _copy_context(context):
    return {'axes':{name:values.copy() for name,values in context['axes'].items()},
        'fits':tuple({'sigma_over_re':f['sigma_over_re'],'weights':f['weights'],
                      'metadata':deepcopy(f['metadata'])} for f in context['fits']),
        'provenance':deepcopy(context['provenance']), 'identity_key':context['identity_key']}


def load_mge_source_library(settings):
    """Load only source coefficients from a same-stream SHA-verified C11 HDF5.

    settings is an already validated effective CESS mapping. C11 verification
    checks every stored probability but does not re-render or certify the PSF.
    Fits/axes are cached by actual byte identities and extraction convention.
    No legacy pickle is loaded. Returned metadata/axes are independent copies;
    shared coefficients are read-only. At most four basis contexts are cached.
    """
    manifest,data,verified,text,declaration = _identity(settings)
    design = settings['spectral_extraction']
    key = (str(manifest),verified['manifest_sha256'],verified['resource']['sha256'],
           json.dumps(design,sort_keys=True,allow_nan=False))
    if key in _LIBRARIES:
        _LIBRARIES.move_to_end(key)
        return _copy_context(_LIBRARIES[key])
    descriptor = os.open(str(manifest.parent),os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        with _regular_member(descriptor,'marginals.hdf5') as stream:
            digest = hashlib.sha256()
            size = 0
            for block in iter(lambda:stream.read(1024*1024),b''):
                digest.update(block);size+=len(block)
            if digest.hexdigest()!=verified['resource']['sha256'] or size!=verified['resource']['size_bytes']:
                raise ValueError('Positive source coefficients changed after verification')
            stream.seek(0)
            fits=[]
            with h5py.File(stream,'r') as handle:
                for i in range(len(declaration['axes']['n'])):
                    group=handle['fits/n_{:06d}'.format(i)]
                    # Immutable byte backing also prevents a caller from
                    # re-enabling writes to a shared owning ndarray.
                    sigma=np.frombuffer(group['sigma_over_re'][:].tobytes(),dtype=np.float64)
                    weights=np.frombuffer(group['weights'][:].tobytes(),dtype=np.float64)
                    fits.append({'sigma_over_re':sigma,'weights':weights,
                                 'metadata':json.loads(group.attrs['metadata_json'])})
    finally:
        os.close(descriptor)
    resource=verified['resource']
    provenance={'schema':'cess.profile_library_description','version':3,
        'provenance_status':'positive_source_basis_verified',
        'bundle_binding':{'path':str(manifest),'sha256':verified['manifest_sha256'],
                          'manifest_json':text,'verification_scope':verified['verification_scope']},
        'resources':{'source_basis':{'path':str(data),'sha256':resource['sha256'],
                                    'size_bytes':resource['size_bytes'],
                                    'array_file_binding':'coefficients read from the same SHA256-verified regular HDF5 stream'}},
        'runtime_interpretation':{'axis_order':list(AXES),'axes':deepcopy(declaration['axes']),
            'arcsecperpix':settings['arcsecperpix'],'arcsecperpix_unit':'arcsec/pixel',
            'lookup':deepcopy(_LOOKUP),'extraction_model':deepcopy(design),
            'source_basis_role':'C11 Gaussian source coefficients only; full-PSF stored marginals not used as kernels',
            'kernel_normalization':'conditional on spatial strip; finite dispersion window not renormalized',
            'lsf_convention_status':'explicit model assumption; not an independent instrument calibration'},
        'generation':{'status':'manifest_recorded_generation',
            'physical_validation_status':declaration['physical_validation_status'],
            'model':deepcopy(declaration['model']),'controls':deepcopy(declaration['controls']),
            'source_sha256':deepcopy(declaration['source_sha256']),
            'software':deepcopy(declaration['software'])},
        'limitations':['spatial/spectral source distribution assumed separable and achromatic',
            'conditional Gaussian strip model does not include velocity fields or arbitrary 2D apertures',
            'source fit L1 and conditional error indicators are numerical estimates, not certified bounds',
            'nearest-grid differences from original source parameters are excluded from fit estimates',
            'dispersion uses discrete integer-pixel redistribution of an already pixel-integrated spectrum',
            'LSF role and adopted Gaussian PSF require independent instrument validation']}
    context={'axes':{name:np.array(declaration['axes'][name],dtype=float) for name in AXES},
             'fits':tuple(fits),'provenance':provenance,'identity_key':key}
    _LIBRARIES[key]=context
    while len(_LIBRARIES)>4:_LIBRARIES.popitem(last=False)
    return _copy_context(context)


def verify_mge_source_binding(settings,provenance):
    """Check current manifest/HDF bytes against the exact source-basis binding."""
    manifest,data,verified,text,declaration=_identity(settings)
    if (provenance.get('version')!=3 or
            provenance.get('provenance_status')!='positive_source_basis_verified'):
        raise ValueError('Expected positive source-basis provenance version3')
    binding=provenance['bundle_binding'];resource=provenance['resources']['source_basis']
    if (binding['path']!=str(manifest) or binding['sha256']!=verified['manifest_sha256'] or
            binding['manifest_json']!=text or resource['path']!=str(data) or
            resource['sha256']!=verified['resource']['sha256'] or
            resource['size_bytes']!=verified['resource']['size_bytes'] or
            provenance['runtime_interpretation']['extraction_model']!=settings['spectral_extraction']):
        raise ValueError('Positive source-basis bytes/path/convention changed before publication')
    return verified


def select_mge_grid(context,n,re,pa,q):
    """Return nearest grid parameters and indices using the established PA fold."""
    from cess.morphology.empirical import fold_position_angle
    values=(n,re,pa,q)
    if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float,np.integer,np.floating))
           or not np.isfinite(v) for v in values):
        raise ValueError('Source morphology parameters must be finite numeric scalars')
    if n<=0. or re<=0. or not 0.<q<=1.:
        raise ValueError('Source n/Re must be positive and axis ratio in (0,1]')
    angle=pa
    pa_grid=context['axes']['pa_deg']
    if np.all((pa_grid>=0.) & (pa_grid<=90.)):
        angle=fold_position_angle(pa)
    values=(n,re,angle,q)
    indices=tuple(int(np.argmin(np.abs(context['axes'][name]-value))) for name,value in zip(AXES,values))
    selected=tuple(float(context['axes'][name][index]) for name,index in zip(AXES,indices))
    return {'selected_parameters':list(selected),'selected_indices':list(indices)}


def select_mge_extraction(context,n,re,pa,q,settings,apply_morphology=True):
    """Select a declared model, derive its joint strip kernel, return owned data.

    The original sampled parameters are not overwritten. The selected model
    and grid indices are explicit diagnostics. Shape-lookup quantization is not
    part of the source-fit L1 estimate. Morphology=False disables dispersion
    redistribution but retains the modeled spatial aperture fraction.
    """
    from cess.morphology.aperture import mge_strip_kernel
    if not isinstance(apply_morphology,(bool,np.bool_)):
        raise ValueError('apply_morphology must be boolean')
    selection=select_mge_grid(context,n,re,pa,q)
    indices=tuple(selection['selected_indices'])
    selected=tuple(selection['selected_parameters'])
    design=settings['spectral_extraction']
    if design!=context['provenance']['runtime_interpretation']['extraction_model']:
        raise ValueError('Source-basis context cannot be reused with a different extraction model')
    key=(context['identity_key'],indices,bool(apply_morphology))
    if key not in _KERNELS:
        fit=context['fits'][indices[0]]
        model=context['provenance']['generation']['model']
        result=mge_strip_kernel(fit['sigma_over_re'],fit['weights'],selected[1],selected[2],selected[3],
            model['psf_fwhm_arcsec'],model['pixel_scale_arcsec_pixel'],design['spatial_pixels'],
            design['kernel_radius_pixels'],design['lsf_convention'],apply_morphology=apply_morphology)
        estimated=fit['metadata']['estimated_radial_L1_including_outside']
        indicator=2.*estimated/result['source_fraction']
        if not math.isfinite(indicator):
            raise ValueError('Conditional source-fit indicator is not representable')
        result['diagnostics'].update(selected_parameters=list(selected),selected_indices=list(indices),
            source_fit_l1_estimate=estimated,conditional_fit_error_indicator=indicator)
        result['profile_height_sum']=float(np.sum(result['height'],dtype=np.float64))
        result['width'].setflags(write=False);result['height'].setflags(write=False)
        _KERNELS[key]=result
        while len(_KERNELS)>512:_KERNELS.popitem(last=False)
    _KERNELS.move_to_end(key)
    cached=_KERNELS[key]
    return dict(width=cached['width'].copy(),height=cached['height'].copy(),
        source_fraction=cached['source_fraction'],spatial_pixels=cached['spatial_pixels'],
        profile_height_sum=cached['profile_height_sum'],diagnostics=deepcopy(cached['diagnostics']))
