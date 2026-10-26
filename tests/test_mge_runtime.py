"""Opt-in source-coefficient binding, selection and ownership boundaries."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

CORE=Path(__file__).resolve().parents[1]/'CESS_v0.8.6'
sys.path.insert(0,str(CORE))
from cess_config import load_config
from cess_mge_library import generate_mge_library
import cess_mge_runtime as runtime


@pytest.fixture(scope='module')
def settings(tmp_path_factory):
    folder=tmp_path_factory.mktemp('mge-runtime')/'basis'
    config=load_config()
    axes=dict(n=[.5,1.],re_pixel=[4.],pa_deg=[0.,10.,90.],axis_ratio=[.4,1.])
    generate_mge_library(config,axes,np.arange(-5.5,6.,1.),folder)
    config.pop('widthlib_pkl',None);config.pop('heightlib_pkl',None)
    config.update(positive_profile_library_manifest_path=str(folder/'manifest.json'),
                  positive_profile_library_path=str(folder/'marginals.hdf5'),
                  spectral_extraction=dict(mode='mge_strip',spatial_pixels=5,
                    kernel_radius_pixels=8,lsf_convention='point_source'))
    return config


def test_same_verified_source_coefficients_json_provenance_and_owned_metadata(settings):
    context=runtime.load_mge_source_library(settings)
    provenance=context['provenance']
    assert json.loads(json.dumps(provenance,allow_nan=False))==provenance
    assert provenance['version']==3 and provenance['provenance_status']=='positive_source_basis_verified'
    assert runtime.verify_mge_source_binding(settings,provenance)['verified']
    assert not context['fits'][0]['weights'].flags.writeable
    with pytest.raises(ValueError):context['fits'][0]['weights'].setflags(write=True)
    context['axes']['n'][0]=99.
    context['provenance']['limitations'].clear()
    context['fits'][0]['metadata'].clear()
    again=runtime.load_mge_source_library(settings)
    assert again['axes']['n'][0]==.5 and again['provenance']['limitations']
    assert again['fits'][0]['metadata']['estimated_radial_L1_including_outside']>=0.


def test_joint_kernel_cache_is_owned_and_records_selected_model(settings):
    context=runtime.load_mge_source_library(settings)
    first=runtime.select_mge_extraction(context,.6,4.,np.nextafter(5.,np.inf),.4,settings)
    assert first['diagnostics']['selected_indices']==[0,0,1,0]
    assert first['diagnostics']['selected_parameters']==[.5,4.,10.,.4]
    assert 0.<first['source_fraction']<1.
    assert 0.<first['width'].sum()<1.
    np.testing.assert_allclose(first['height'].sum(),first['source_fraction'],atol=1.e-14)
    expected=first['width'].copy()
    first['width'][:]=0.;first['diagnostics'].clear()
    again=runtime.select_mge_extraction(context,.6,4.,np.nextafter(5.,np.inf),.4,settings)
    np.testing.assert_array_equal(again['width'],expected)
    assert again['diagnostics']['conditional_fit_error_indicator']>=0.
    disabled=runtime.select_mge_extraction(context,.6,4.,10.,.4,settings,False)
    np.testing.assert_array_equal(disabled['width'],[1.])
    assert disabled['source_fraction']==again['source_fraction']


@pytest.mark.parametrize('angle,chosen',[(5.,0),(-5.,0),(np.nextafter(5.,np.inf),1),
    (-np.nextafter(5.,np.inf),1),(170.,1),(190.,1)])
def test_nearest_pa_matches_existing_fold_and_preserves_ties(settings,angle,chosen):
    context=runtime.load_mge_source_library(settings)
    assert runtime.select_mge_grid(context,.5,4.,angle,.4)['selected_indices'][2]==chosen


@pytest.mark.parametrize('field,value',[('arcsecperpix',.08),('morphology_psf_fwhm',.3),
    ('positive_profile_library_path','/tmp/wrong.hdf5'),('profile_library_manifest_path','/tmp/legacy.json')])
def test_wrong_physical_model_or_bundle_identity_rejected(settings,field,value):
    config=copy.deepcopy(settings);config[field]=value
    with pytest.raises(ValueError):runtime.load_mge_source_library(config)


def test_convention_cannot_be_reinterpreted_after_loading(settings):
    context=runtime.load_mge_source_library(settings)
    changed=copy.deepcopy(settings);changed['spectral_extraction']['lsf_convention']='pre_psf'
    with pytest.raises(ValueError,match='different extraction model'):
        runtime.select_mge_extraction(context,.5,4.,10.,.4,changed)
    with pytest.raises(ValueError,match='convention changed'):
        runtime.verify_mge_source_binding(changed,context['provenance'])


@pytest.mark.parametrize('args',[(True,4.,0.,1.),(.5,'4',0.,1.),(.5,4.,np.nan,1.),
    (.5,0.,0.,1.),(.5,4.,0.,1.1)])
def test_bad_source_parameters_fail_before_nearest_lookup(settings,args):
    with pytest.raises(ValueError):runtime.select_mge_grid(runtime.load_mge_source_library(settings),*args)


def test_nonboolean_switch_rejected_even_after_cache_hit(settings):
    context=runtime.load_mge_source_library(settings)
    runtime.select_mge_extraction(context,.5,4.,0.,1.,settings,True)
    with pytest.raises(ValueError,match='boolean'):
        runtime.select_mge_extraction(context,.5,4.,0.,1.,settings,'yes')
