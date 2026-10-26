"""Tiny synthetic numeric artifacts test verifier structure, not source physics."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

CORE=Path(__file__).resolve().parents[1]/'CESS_v0.8.6'
sys.path.insert(0,str(CORE))
import cess_mge_bundle as verifier
from cess_profile_bundle import verify_profile_bundle


def refresh(directory, manifest=None):
    if manifest is None:manifest=json.loads((directory/'manifest.json').read_text())
    payload=(directory/'marginals.hdf5').read_bytes()
    manifest['resource']={'filename':'marginals.hdf5','size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
    (directory/'manifest.json').write_text(json.dumps(manifest,sort_keys=True)+'\n')
    return manifest


def fit_metadata(estimate,controls):
    attempt=dict(status='evaluated',width_grid_count=32,candidate_components=33,active_components=2,
        estimated_radial_L1_including_outside=estimate,radial_L1_refinement_difference=0.,
        source_outside_fit_domain_probability=2e-12,mixture_outside_fit_domain_probability=2e-12)
    return dict(method='synthetic non-rendered fixture',normalized_weight_sum=1.,attempts=[attempt],
        requested_estimated_L1_tolerance=controls['l1_tolerance'],
        requested_source_tail_each_end=controls['tail_probability'],maximum_width_grid_count=controls['max_components'],
        estimated_radial_L1_including_outside=estimate,radial_L1_refinement_difference=0.,
        source_outside_fit_domain_probability=2e-12,mixture_outside_fit_domain_probability=2e-12)


@pytest.fixture
def bundle(tmp_path):
    directory=tmp_path/'bundle';directory.mkdir()
    controls=dict(l1_tolerance=1e-7,tail_probability=1e-12,max_components=160)
    model=dict(pixel_scale_arcsec_pixel=.074,psf_fwhm_arcsec=.2625,psf='circular_Gaussian',
        source='positive_Gaussian_approximation_to_untruncated_Sersic',
        re_definition='circularized half-light radius; area-preserving shear',pa_definition='image-x angle in degrees')
    axes=dict(n=[.5,1.],re_pixel=[4.],pa_deg=[-30.],axis_ratio=[.5,1.])
    config=dict(arcsecperpix=.074,morphology_psf_fwhm=.2625,unused_path='/missing/calibration/file')
    manifest=dict(schema=verifier.SCHEMA,version=1,status='complete',generator_revision=verifier.GENERATOR_REVISION,
        created_at_utc='2026-10-22T00:00:00+00:00',axis_order=list(verifier.AXIS_ORDER),axes=axes,
        axis_units=verifier.AXIS_UNITS,shape=[2,1,1,2],profile_count=4,
        x_edges_pixel=[-2.,-.5,.5,2.],y_edges_pixel=[-1.,0.,1.],quantity=verifier.QUANTITY,
        orthogonal_domain='all real',window_renormalized=False,
        physical_validation_status='experimental_source_approximation_not_instrument_calibration',
        model=model,controls=controls,probability_accounting_tolerance=1e-9,effective_config=config,
        effective_config_hash_encoding=verifier.HASH_ENCODING,
        effective_config_sha256=hashlib.sha256((json.dumps(config,indent=2,sort_keys=True)+'\n').encode()).hexdigest(),
        config_source=dict(kind='mapping',path=None,sha256=None,adopted_defaults={}),
        source_sha256={'/missing/generation/code.py':'a'*64},software={'python':'fixture'},
        limitations=['synthetic fixture; no source approximation certification'],observed={})
    with h5py.File(str(directory/'marginals.hdf5'),'w') as f:
        for key in ('schema','version','status','generator_revision','quantity','orthogonal_domain','window_renormalized','probability_accounting_tolerance'):
            f.attrs[key]=manifest[key]
        for key,value in [('axis_order_json',manifest['axis_order']),('model_json',model),('controls_json',controls)]:
            f.attrs[key]=json.dumps(value)
        a=f.create_group('axes');b=f.create_group('bins');p=f.create_group('profiles');fits=f.create_group('fits')
        for name,values in axes.items():
            d=a.create_dataset(name,data=np.asarray(values,dtype='float64'));d.attrs['unit']=verifier.AXIS_UNITS[name];d.attrs['axis_order']='grid'
        for name in ('x_edges_pixel','y_edges_pixel'):
            d=b.create_dataset(name,data=np.asarray(manifest[name],dtype='float64'));d.attrs['unit']='pixel';d.attrs['axis_order']='edge'
        data={'row_indices':np.array(list(np.ndindex(2,1,1,2)),dtype='int64'),
              'x_probability':np.tile([.1,.2,.3],(4,1)),'y_probability':np.tile([.25,.25],(4,1)),
              'window_probability':np.tile([.6,.5],(4,1)),'outside_probability':np.tile([.4,.5],(4,1)),
              'source_fit_l1_estimate':np.array([1e-10,1e-10,2e-10,2e-10])}
        orders={'row_indices':'profile,parameter_axis','x_probability':'profile,x_bin','y_probability':'profile,y_bin',
                'window_probability':'profile,axis','outside_probability':'profile,axis','source_fit_l1_estimate':'profile'}
        for name,value in data.items():
            d=p.create_dataset(name,data=value);d.attrs['unit']='1';d.attrs['axis_order']=orders[name]
        for i,n in enumerate(axes['n']):
            g=fits.create_group('n_{:06d}'.format(i));g.attrs['n']=float(n)
            g.attrs['metadata_json']=json.dumps(fit_metadata((i+1)*1e-10,controls))
            for name,values in [('sigma_over_re',[.5,1.]),('weights',[.4,.6])]:
                d=g.create_dataset(name,data=np.asarray(values,dtype='float64'));d.attrs['unit']='1';d.attrs['axis_order']='component'
    refresh(directory,manifest)
    return directory


def test_valid_artifact_is_readonly_jsonsafe_and_uses_bounded_chunks(bundle,monkeypatch):
    original=h5py.Dataset.__getitem__;reads=[]
    def capture(self,key):
        if self.name.startswith('/profiles/'):
            reads.append(key)
        return original(self,key)
    before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in bundle.iterdir()}
    monkeypatch.setattr(verifier,'CHUNK_ROWS',2);monkeypatch.setattr(h5py.Dataset,'__getitem__',capture)
    result=verifier.verify_mge_bundle(bundle)
    assert result['verified'] and result['profile_count']==4
    assert 'no source approximation' in result['verification_scope']
    assert reads and all(isinstance(k,slice) and k.stop-k.start<=2 for k in reads)
    json.dumps(result,allow_nan=False)
    assert before=={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in bundle.iterdir()}


@pytest.mark.parametrize('problem',[
    'negative','nan','above_one','row_order','window_sum','accounting','fit_mapping',
    'shared_model','unit','dtype','truncated_shape','fit_weights','fit_sigma','fit_n','fit_active','fit_l1'])
def test_rehashed_numeric_or_shared_metadata_corruption_is_rejected(bundle,problem):
    with h5py.File(str(bundle/'marginals.hdf5'),'r+') as f:
        p=f['profiles'];g=f['fits/n_000000']
        if problem in ('negative','nan','above_one'):p['x_probability'][0,0]={'negative':-1e-20,'nan':np.nan,'above_one':1.1}[problem]
        elif problem=='row_order':p['row_indices'][0]=p['row_indices'][1]
        elif problem=='window_sum':p['window_probability'][0,0]=.59
        elif problem=='accounting':p['outside_probability'][0,0]=.39
        elif problem=='fit_mapping':p['source_fit_l1_estimate'][0]=2e-10
        elif problem=='shared_model':
            value=json.loads(f.attrs['model_json']);value['psf_fwhm_arcsec']*=2;f.attrs['model_json']=json.dumps(value)
        elif problem=='unit':p['x_probability'].attrs['unit']='uJy'
        elif problem in ('dtype','truncated_shape'):
            value=p['x_probability'][:];del p['x_probability']
            p.create_dataset('x_probability',data=value.astype('float32') if problem=='dtype' else value[:3])
        elif problem=='fit_weights':g['weights'][0]=.3
        elif problem=='fit_sigma':g['sigma_over_re'][0]=0.
        elif problem=='fit_n':g.attrs['n']=1.
        elif problem in ('fit_active','fit_l1'):
            meta=json.loads(g.attrs['metadata_json'])
            if problem=='fit_active':meta['attempts'][-1]['active_components']=1
            else:meta['estimated_radial_L1_including_outside']=1.
            g.attrs['metadata_json']=json.dumps(meta)
    refresh(bundle)
    with pytest.raises(ValueError):verifier.verify_mge_bundle(bundle)


@pytest.mark.parametrize('problem',['hash','config_hash','axis_units','unknown_version'])
def test_manifest_corruption_is_rejected(bundle,problem):
    manifest=json.loads((bundle/'manifest.json').read_text())
    if problem=='hash':manifest['resource']['sha256']='0'*64
    elif problem=='config_hash':manifest['effective_config']['arcsecperpix']=.1
    elif problem=='axis_units':manifest['axis_units']['re_pixel']='arcsec'
    else:manifest['version']=2
    (bundle/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):verifier.verify_mge_bundle(bundle)


def test_extra_files_and_symlink_members_are_rejected(bundle,tmp_path):
    extra=bundle/'extra';extra.write_text('unexpected')
    with pytest.raises(ValueError):verifier.verify_mge_bundle(bundle)
    extra.unlink();target=tmp_path/'outside.h5';(bundle/'marginals.hdf5').rename(target)
    (bundle/'marginals.hdf5').symlink_to(target)
    with pytest.raises(ValueError,match='non-symlink'):verifier.verify_mge_bundle(bundle)


@pytest.mark.parametrize('kind',['soft','external'])
def test_hdf_links_cannot_load_resources_outside_declared_member(bundle,kind):
    with h5py.File(str(bundle/'marginals.hdf5'),'r+') as f:
        del f['profiles/x_probability']
        f['profiles/x_probability']=(h5py.SoftLink('/not-present') if kind=='soft' else
            h5py.ExternalLink('/missing/external-resource.h5','/p'))
    refresh(bundle)
    with pytest.raises(ValueError,match='links are not permitted'):verifier.verify_mge_bundle(bundle)


def test_c7_rejects_new_schema_and_new_verifier_import_cli_are_safe(bundle,tmp_path):
    with pytest.raises(ValueError):verify_profile_bundle(bundle)
    code="""import sys,h5py
sys.path.insert(0,sys.argv[1])
h5py.File=lambda *a,**k:(_ for _ in ()).throw(AssertionError('import performed HDF I/O'))
import cess_mge_bundle
assert not any(n in sys.modules for n in ['cess_mge','cess_config','morphology'])
"""
    result=subprocess.run([sys.executable,'-B','-c',code,str(CORE)],cwd=str(tmp_path),stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    assert result.returncode==0,result.stderr
    cli=subprocess.run([sys.executable,'-B',str(CORE/'cess_mge_bundle.py'),str(bundle)],cwd=str(tmp_path),stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    assert cli.returncode==0,cli.stderr
    assert json.loads(cli.stdout)['verified'] is True


@pytest.mark.parametrize('excess,accepted',[(np.finfo(float).eps,True),(1e-5,False)])
def test_aggregate_roundoff_is_retained_but_material_mass_excess_is_rejected(bundle,excess,accepted):
    with h5py.File(str(bundle/'marginals.hdf5'),'r+') as f:
        f['profiles/x_probability'][0]=[.5,.5,excess]
        f['profiles/window_probability'][0,0]=1.+excess
        f['profiles/outside_probability'][0,0]=0.
    refresh(bundle)
    if accepted:
        assert verifier.verify_mge_bundle(bundle)['verified']
        with h5py.File(str(bundle/'marginals.hdf5'),'r') as f:
            assert f['profiles/window_probability'][0,0]>1.  # no clipping/rewriting
    else:
        with pytest.raises(ValueError):verifier.verify_mge_bundle(bundle)
