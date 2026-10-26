"""Independent physics and failure checks of the experimental positive MGE."""
import json
import math
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaincinv,k1,ndtr

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'CESS_v0.8.6'))
import cess_mge as mge
from cess_oracle import sersic_bin_oracle


def call(n=1.,re=4.,pa=30.,q=.6,psf=.2625,pixel=.074,x=None,y=None,**kwargs):
    if x is None:x=np.arange(-15.5,16.5)
    return mge.sersic_mge_marginals(n,re,pa,q,psf,pixel,x,y,**kwargs)


def scales(q,pa):
    angle=np.deg2rad(pa)
    return np.sqrt(np.cos(angle)**2/q+q*np.sin(angle)**2),np.sqrt(np.sin(angle)**2/q+q*np.cos(angle)**2)


def normal_mass(a,b):
    a,b=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float))
    return ndtr(np.where(a>=0,-a,b))-ndtr(np.where(a>=0,-b,a))


@pytest.mark.parametrize('q,pa',[(1.,0.),(.3,47.),(.3,90.)])
def test_gaussian_against_absolute_analytic_bins(q,pa):
    edges=np.array([-30.,-2.,-.5,.5,3.,30.])
    result=call(n=.5,re=3.,q=q,pa=pa,x=edges)
    for axis,s in zip(('x','y'),scales(q,pa)):
        sigma=math.hypot(3.*s/math.sqrt(2*math.log(2)),.2625/.074/math.sqrt(8*math.log(2)))
        expected=normal_mass(edges[:-1]/sigma,edges[1:]/sigma)
        np.testing.assert_allclose(result[axis+'_probability'],expected,rtol=1e-11,atol=2e-13)


def bessel_bin(lo,hi,re,s,psf,pixel):
    a=re*s/gammaincinv(2.,.5)
    def density(u):
        z=abs(u)/a
        return 1/(math.pi*a) if z==0 else z*k1(z)/(math.pi*a)
    if psf==0:
        return quad(density,lo,hi,epsabs=1e-12,epsrel=1e-11,points=[0.] if lo<0<hi else None)[0]
    sigma=psf/pixel/math.sqrt(8*math.log(2))
    f=lambda u:density(u)*float(normal_mass((lo-u)/sigma,(hi-u)/sigma))
    return sum(quad(f,a,b,epsabs=1e-12,epsrel=1e-11,limit=200)[0] for a,b in [(-np.inf,0.),(0.,np.inf)])


@pytest.mark.parametrize('psf',[0.,.2625])
def test_exponential_against_independent_bessel_density(psf):
    edges=np.array([-1.5,-.5,.5,1.5,7.])
    result=call(n=1.,re=4.,q=.6,pa=30.,psf=psf,x=edges)
    for axis,s in zip(('x','y'),scales(.6,30.)):
        expected=[bessel_bin(a,b,4.,s,psf,.074) for a,b in zip(edges[:-1],edges[1:])]
        np.testing.assert_allclose(result[axis+'_probability'],expected,rtol=2e-8,atol=2e-10)


@pytest.mark.parametrize('n,center',[(4.,0.),(4.,20.),(5.,0.),(5.,20.)])
def test_high_index_bins_against_independent_scalar_oracle(n,center):
    edges=np.array([center-.5,center+.5])
    result=call(n=n,re=20.,pa=90.,q=.3,x=edges)
    independent=sersic_bin_oracle(n,20.,90.,.3,.2625,.074,*edges,
                                 absolute_tolerance=1e-9,max_evaluations=200000)
    assert result['x_probability'][0]==pytest.approx(independent['probability'],abs=2e-9)
    assert result['metadata']['source_fit']['estimated_radial_L1_including_outside']<=1e-7


@pytest.mark.parametrize('n',[1.,4.])
def test_geometry_symmetries_coordinate_rescaling_and_window_loss(n):
    x=np.array([-1.5,-.5,.5,1.5])
    a=call(n=n,pa=30.,x=x)
    mirror=call(n=n,pa=-30.,x=x)
    quarter=call(n=n,pa=120.,x=x)
    rescaled=call(n=n,re=12.,pixel=.074/3,x=x*3)
    # A finite wider window with resolvable remaining loss, rather than a
    # near-all-source window whose sum can round to one.
    large=call(n=n,x=np.array([-20.,-1.5,-.5,.5,1.5,20.]))
    for axis,other in [('x','y'),('y','x')]:
        np.testing.assert_allclose(a[axis+'_probability'],mirror[axis+'_probability'],rtol=1e-12,atol=2e-14)
        np.testing.assert_allclose(a[axis+'_probability'],quarter[other+'_probability'],rtol=1e-12,atol=2e-14)
        np.testing.assert_allclose(a[axis+'_probability'],rescaled[axis+'_probability'],rtol=1e-12,atol=2e-14)
        np.testing.assert_allclose(a[axis+'_probability'],large[axis+'_probability'][1:-1],rtol=1e-12,atol=2e-14)
        assert a[axis+'_probability'].sum()<large[axis+'_probability'].sum()<1.


def test_positive_source_mass_error_metadata_and_probability_accounting():
    result=call(n=4.)
    assert np.all(result['mixture_weights']>0) and np.all(result['mixture_sigma_over_re']>0)
    assert result['mixture_weights'].sum()==pytest.approx(1.,abs=2e-14)
    metadata=result['metadata'];fit=metadata['source_fit']
    assert metadata['normalization']['window_renormalized'] is False
    assert metadata['normalization']['orthogonal_domain']=='all real'
    assert fit['estimated_radial_L1_including_outside']<=1e-7
    assert fit['radial_L1_refinement_difference']<=1e-8
    assert fit['source_outside_fit_domain_probability']==pytest.approx(2e-12,rel=1e-10)
    assert 'not certified' in fit['error_interpretation']
    assert 'sampled estimates' in fit['L1_transport']
    for axis in ('x','y'):
        p=result[axis+'_probability'];window=metadata['windows'][axis]
        assert np.all(np.isfinite(p)) and np.all(p>=0)
        assert p.sum()+window['outside_probability_integrated']==pytest.approx(1.,abs=2e-12)
    json.dumps(metadata,allow_nan=False)


def test_public_arrays_metadata_rng_and_cached_source_are_isolated():
    before=np.random.get_state()
    a=call(n=1.)
    expected=a['x_probability'].copy();weights=a['mixture_weights'].copy()
    a['x_probability'][:]=-1
    a['mixture_weights'][:]=0
    a['mixture_sigma_over_re'][:]=0
    a['metadata']['source_fit']['attempts'][0]['status']='poisoned'
    b=call(n=1.)
    np.testing.assert_array_equal(b['x_probability'],expected)
    np.testing.assert_array_equal(b['mixture_weights'],weights)
    assert b['metadata']['source_fit']['attempts'][0]['status']!='poisoned'
    after=np.random.get_state()
    assert before[0]==after[0] and before[2:]==after[2:]
    np.testing.assert_array_equal(before[1],after[1])


@pytest.mark.parametrize('kwargs',[
    {'n':.4},{'n':5.1},{'n':True},{'re':0.},{'q':1.1},
    {'x':np.array([1.,0.])},{'tail_probability':1e-15},{'max_components':161},
])
def test_invalid_input_fails_before_fitting(kwargs,monkeypatch):
    def forbidden(*args):raise AssertionError('must validate before fit')
    monkeypatch.setattr(mge,'_fit',forbidden)
    with pytest.raises(ValueError):call(**kwargs)


def test_insufficient_fit_budget_fails_without_substitution():
    with pytest.raises(mge.MixtureConvergenceError) as error:
        call(n=5.,max_components=16,l1_tolerance=1e-12)
    assert error.value.diagnostics['attempts']
    assert 'probability' not in error.value.diagnostics


def test_near_gaussian_index_needs_larger_budget_without_relaxing_target():
    result=call(n=.6)
    fit=result['metadata']['source_fit']
    assert fit['estimated_radial_L1_including_outside']<=1e-7
    assert fit['attempts'][-1]['width_grid_count']==160
    with pytest.raises(mge.MixtureConvergenceError) as error:
        call(n=.6,max_components=128)
    assert error.value.diagnostics['attempts'][-1]['estimated_radial_L1_including_outside']>1e-7


@pytest.mark.parametrize('mode',['failure','warning'])
def test_nnls_failure_and_warning_are_not_silently_replaced(monkeypatch,mode):
    mge._fit.cache_clear()
    original=mge.nnls
    def broken(*args,**kwargs):
        if mode=='failure':raise RuntimeError('synthetic NNLS iteration failure')
        result=original(*args,**kwargs)
        warnings.warn('synthetic NNLS warning',UserWarning)
        return result
    monkeypatch.setattr(mge,'nnls',broken)
    with pytest.raises(mge.MixtureConvergenceError):call(n=.5)
