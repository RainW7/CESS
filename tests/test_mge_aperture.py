"""Independent joint Gaussian rectangle checks, not products of marginals."""
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings
import numpy as np
import pytest
from scipy.integrate import dblquad
from scipy.special import ndtr

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import cess.morphology.aperture as aperture
def call(sigmas=(1.,),weights=(1.,),re=3.,pa=35.,q=.3,psf=.2625,pixel=.074,
         n=3,k=1,role='pre_psf',**kw):
    return aperture.mge_strip_kernel(sigmas,weights,re,pa,q,psf,pixel,n,k,role,**kw)


def covariance(sigma,re,q,pa,psf,pixel,role):
    a=np.deg2rad(pa);rotation=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    source=rotation.dot(np.diag([(sigma*re)**2/q,(sigma*re)**2*q])).dot(rotation.T)
    p=(psf/pixel)**2/(8*np.log(2))
    return source+np.diag([p if role=='pre_psf' else 0.,p])


def rectangle(cov,xlo,xhi,ylo,yhi):
    # Independent direct bivariate density integration, using matrix inverse.
    inverse=np.linalg.inv(cov);factor=1/(2*np.pi*np.sqrt(np.linalg.det(cov)))
    def density(y,x):
        v=np.array([x,y]);return factor*np.exp(-.5*v.dot(inverse).dot(v))
    return dblquad(density,xlo,xhi,lambda x:ylo,lambda x:yhi,epsabs=2e-12,epsrel=2e-11)[0]


@pytest.mark.parametrize('q,pa',[(1.,35.),(.3,0.),(.3,90.)])
@pytest.mark.parametrize('role',['point_source','pre_psf'])
def test_separable_exact_gaussian_pixel_and_strip_integrals(q,pa,role):
    result=call(q=q,pa=pa,role=role)
    cov=covariance(1.,3.,q,pa,.2625,.074,role)
    sx,sy=np.sqrt(np.diag(cov));xe=np.arange(-1,3)-.5;ye=np.arange(4)-1.5
    f=float(ndtr(1.5/sy)-ndtr(-1.5/sy))
    np.testing.assert_allclose(result['width'],ndtr(xe[1:]/sx)-ndtr(xe[:-1]/sx),rtol=1e-12,atol=1e-15)
    np.testing.assert_allclose(result['height'],ndtr(ye[1:]/sy)-ndtr(ye[:-1]/sy),rtol=1e-12,atol=1e-15)
    assert result['source_fraction']==pytest.approx(f,abs=2e-15)
    assert result['diagnostics']['quadrature_error_estimate']==0.


@pytest.mark.parametrize('role',['point_source','pre_psf'])
def test_correlated_strip_matches_independent_two_dimensional_density(role):
    result=call(role=role)
    cov=covariance(1.,3.,.3,35.,.2625,.074,role)
    sy=np.sqrt(cov[1,1]);f=float(ndtr(1.5/sy)-ndtr(-1.5/sy))
    expected=[rectangle(cov,x-.5,x+.5,-1.5,1.5)/f for x in [-1.,0.,1.]]
    np.testing.assert_allclose(result['width'],expected,rtol=0,atol=2e-11)
    assert result['diagnostics']['quadrature_components']==1
    assert result['diagnostics']['quadrature_error_estimate']<=1e-9
    assert result['diagnostics']['kernel_absolute_fraction']==pytest.approx(f*sum(expected),abs=2e-11)


def test_mixture_is_conditioned_after_full_source_combination():
    sigmas=np.array([.2,3.]);weights=np.array([.2,.8])
    result=call(sigmas,weights,re=2.,pa=45.,q=.2,psf=0.,n=1)
    fractions=[];joints=[]
    for s in sigmas:
        cov=covariance(s,2.,.2,45.,0.,.074,'pre_psf');sy=np.sqrt(cov[1,1])
        fractions.append(float(ndtr(.5/sy)-ndtr(-.5/sy)))
        joints.append([rectangle(cov,x-.5,x+.5,-.5,.5) for x in [-1.,0.,1.]])
    f=float(weights.dot(fractions));expected=weights.dot(joints)/f
    wrong=weights.dot(np.array(joints)/np.array(fractions)[:,None])
    np.testing.assert_allclose(result['width'],expected,rtol=0,atol=2e-10)
    assert np.max(abs(expected-wrong))>.05
    assert result['source_fraction']==pytest.approx(f,abs=2e-15)
    scaled=call(sigmas,weights*10,re=2.,pa=45.,q=.2,psf=0.,n=1)
    np.testing.assert_allclose(scaled['width'],result['width'],rtol=0,atol=2e-14)


@pytest.mark.parametrize('role',['point_source','pre_psf'])
def test_morphology_off_retains_spatial_loss_and_height(role):
    on=call(role=role);off=call(role=role,apply_morphology=False)
    np.testing.assert_array_equal(off['width'],[1.])
    np.testing.assert_array_equal(off['height'],on['height'])
    assert off['source_fraction']==on['source_fraction']<1.
    assert off['spatial_pixels']==3
    d=off['diagnostics'];assert not d['kernel_applied']
    assert d['kernel_window_fraction']==1. and d['kernel_absolute_fraction']==off['source_fraction']
    assert d['quadrature_error_estimate']==d['omitted_probability_bound']==0.


def test_extreme_correlation_keeps_positive_determinant_conditional_variance():
    # Direct A-B^2/C loses this strictly positive conditional variance.
    result=call(re=1.,q=1e-16,pa=45.,psf=0.,n=9,k=4)
    assert np.all(result['width']>0) and np.all(np.isfinite(result['width']))
    np.testing.assert_allclose(result['width'],np.full(9,1/9),rtol=0,atol=2e-8)
    assert result['source_fraction']>0.


def test_wide_spatial_strip_approaches_full_marginal_and_reduces_coupling():
    narrow=call(n=1,k=20);wide=call(n=1001,k=20)
    cov=covariance(1.,3.,.3,35.,.2625,.074,'pre_psf');sx=np.sqrt(cov[0,0])
    edges=np.arange(-20,22)-.5
    expected=ndtr(edges[1:]/sx)-ndtr(edges[:-1]/sx)
    np.testing.assert_allclose(wide['width'],expected,rtol=0,atol=2e-11)
    assert narrow['source_fraction']<wide['source_fraction']<=1.
    x=np.arange(-20,21,dtype=float)
    assert np.dot(x*x,narrow['width'])<np.dot(x*x,wide['width'])


def test_finite_kernel_mass_is_not_normalized_and_both_psf_roles_are_explicit():
    small=call(k=0);large=call(k=10)
    assert len(small['width'])==1 and small['width'].sum()<.5
    assert small['source_fraction']==large['source_fraction']
    assert small['width'][0]==pytest.approx(large['width'][10],abs=2e-11)
    assert small['diagnostics']['kernel_window_fraction']<large['diagnostics']['kernel_window_fraction']
    point=call(k=0,role='point_source')
    assert point['source_fraction']==small['source_fraction']
    assert point['width'][0]>small['width'][0]  # no double imaging-PSF broadening in x


def test_pa_mirror_units_rng_inputs_and_json_diagnostics():
    sigmas=np.array([.5,1.5]);weights=np.array([2.,3.]);copies=sigmas.copy(),weights.copy()
    state=np.random.get_state()
    a=call(sigmas,weights);b=call(sigmas,weights,pa=-35.,psf=2.625,pixel=.74)
    np.testing.assert_allclose(a['width'],b['width'],rtol=0,atol=2e-13)
    np.testing.assert_allclose(a['width'],a['width'][::-1],rtol=0,atol=2e-13)
    np.testing.assert_allclose(a['height'],b['height'],rtol=0,atol=2e-14)
    np.testing.assert_array_equal(sigmas,copies[0]);np.testing.assert_array_equal(weights,copies[1])
    after=np.random.get_state();assert state[0]==after[0] and state[2:]==after[2:]
    np.testing.assert_array_equal(state[1],after[1]);json.dumps(a['diagnostics'],allow_nan=False)


@pytest.mark.parametrize('change',[
    {'sigmas':[]},{'sigmas':[0.]},{'weights':[-1.]},{'weights':[True]},
    {'re':0.},{'q':1.1},{'pa':np.nan},{'psf':-1.},{'n':True},{'k':1025},
    {'role':'implicit'},{'absolute_tolerance':0.},{'max_evaluations':False}])
def test_invalid_inputs_fail_without_substitution(change):
    with pytest.raises(ValueError):call(**change)


def test_global_quadrature_budget_is_enforced():
    with pytest.raises(aperture.StripConvergenceError,match='budget') as caught:
        call(max_evaluations=1)
    assert caught.value.diagnostics['evaluations']==1


@pytest.mark.parametrize('mode',['warning','nonconvergence','negative'])
def test_bad_quadrature_is_rejected_without_clipping(monkeypatch,mode):
    def fake(function,left,right,**kwargs):
        if mode=='warning':warnings.warn('synthetic quadrature warning')
        value=np.zeros(3) if mode!='negative' else -np.ones(3)*1e-20
        return value,0.,SimpleNamespace(success=mode!='nonconvergence')
    monkeypatch.setattr(aperture,'quad_vec',fake)
    with pytest.raises(aperture.StripConvergenceError):call()
