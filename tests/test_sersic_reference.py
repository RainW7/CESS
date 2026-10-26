"""Independent physical checks of positive infinite-axis Sérsic marginal references.

References below use analytic Normal bin masses or the exponential's Abel/Bessel
projection, rather than the production variance-mixture representation.
"""
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaincinv, k1, ndtr

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cess.morphology.reference import sersic_pixel_reference


def _normal_mass(lo,hi):
    lo,hi=np.broadcast_arrays(np.asarray(lo,float),np.asarray(hi,float))
    return ndtr(np.where(lo>=0,-lo,hi))-ndtr(np.where(lo>=0,-hi,lo))


def _scale(q,pa):
    angle=np.deg2rad(pa)
    return (np.sqrt(np.cos(angle)**2/q+q*np.sin(angle)**2),
            np.sqrt(np.sin(angle)**2/q+q*np.cos(angle)**2))


def _reference(n=.5,re=2.5,pa=37.,q=.6,psf=.2625,pixel=.074,
               x=None,y=None,**kwargs):
    if x is None:x=np.arange(-12.5,13.5)
    return sersic_pixel_reference(n,re,pa,q,psf,pixel,x,y,**kwargs)


def _bessel_density(u,a):
    z=abs(u)/a
    return 1/(np.pi*a) if z==0 else z*k1(z)/(np.pi*a)


def _exponential_bin(lo,hi,re,scale,psf,pixel):
    """Independent Abel projection; optionally convolve its density with the PSF."""
    a=re*scale/gammaincinv(2.,.5)
    if psf==0:
        return quad(lambda u:_bessel_density(u,a),lo,hi,
                    points=[0.] if lo<0<hi else None,
                    epsabs=2e-13,epsrel=2e-12,limit=200)[0]
    sigma=psf/pixel/np.sqrt(8*np.log(2))
    def integrand(u):
        return _bessel_density(u,a)*float(_normal_mass((lo-u)/sigma,(hi-u)/sigma))
    # Split at the finite central Bessel limit. These infinite integrals do not
    # use the implementation's Gamma mixing law or its finite cutoff.
    return sum(quad(integrand,a,b,epsabs=2e-13,epsrel=2e-12,limit=300)[0]
               for a,b in [(-np.inf,0.),(0.,np.inf)])


@pytest.mark.parametrize('q,pa',[(1.,0.),(1.,67.),(.3,0.),(.3,47.),(.3,90.)])
@pytest.mark.parametrize('psf',[0.,.2625])
def test_gaussian_absolute_pixel_mass_geometry_and_psf_quadrature(q,pa,psf):
    re,pixel=2.5,.074
    x=np.array([-30.,-8.,-1.5,-.5,.5,1.5,8.,30.])
    y=np.array([-31.,-6.,-1.,0.,2.,7.,31.])
    result=_reference(re=re,pa=pa,q=q,psf=psf,pixel=pixel,x=x,y=y)
    for axis,edges,s in zip(('x','y'),(x,y),_scale(q,pa)):
        sigma=np.sqrt((re*s)**2/(2*np.log(2))+(psf/pixel)**2/(8*np.log(2)))
        exact=_normal_mass(edges[:-1]/sigma,edges[1:]/sigma)
        np.testing.assert_allclose(result[axis+'_probability'],exact,rtol=3e-13,atol=3e-16)
        np.testing.assert_array_equal(result[axis+'_edges_pixel'],edges)
        assert np.all(result[axis+'_probability']>=0)


def test_gaussian_far_tail_survives_naive_cdf_cancellation():
    # Re chosen so the n=.5 Gaussian has unit source sigma; no PSF.
    result=_reference(re=np.sqrt(2*np.log(2)),psf=0.,q=1.,pa=0.,x=np.array([9.,10.]))
    expected=float(ndtr(-9.)-ndtr(-10.))
    assert expected>0 and ndtr(10.)-ndtr(9.)==0
    np.testing.assert_allclose(result['x_probability'],[expected],rtol=3e-13,atol=0)


@pytest.mark.parametrize('n',[.5,1.])
def test_circular_pa_invariance_mirror_symmetry_and_axis_exchange(n):
    kw=dict(n=n,x=np.arange(-15.5,16.5),absolute_tolerance=1e-11,mixing_tail_probability=1e-13)
    circular_a=_reference(q=1.,pa=-81.,**kw)
    circular_b=_reference(q=1.,pa=29.,**kw)
    plus=_reference(q=.4,pa=31.,**kw)
    minus=_reference(q=.4,pa=-31.,**kw)
    quarter=_reference(q=.4,pa=121.,**kw)
    for axis in ('x','y'):
        np.testing.assert_allclose(circular_a[axis+'_probability'],circular_b[axis+'_probability'],rtol=2e-12,atol=1e-13)
        np.testing.assert_allclose(plus[axis+'_probability'],minus[axis+'_probability'],rtol=2e-12,atol=1e-13)
        np.testing.assert_allclose(plus[axis+'_probability'],plus[axis+'_probability'][::-1],rtol=2e-12,atol=1e-13)
    np.testing.assert_allclose(plus['x_probability'],quarter['y_probability'],rtol=2e-12,atol=1e-13)
    np.testing.assert_allclose(plus['y_probability'],quarter['x_probability'],rtol=2e-12,atol=1e-13)


@pytest.mark.parametrize('psf',[0.,.2625])
def test_exponential_against_independent_abel_bessel_reference(psf):
    edges=np.array([-8.,-1.5,-.5,.5,1.5,8.])
    y_edges=np.array([-2.,-.25,0.,1.,7.])
    re,q,pa,pixel=2.3,.6,37.,.074
    result=_reference(n=1.,re=re,q=q,pa=pa,psf=psf,pixel=pixel,x=edges,y=y_edges,
                      absolute_tolerance=1e-11,mixing_tail_probability=1e-13)
    for axis,axis_edges,scale in zip(('x','y'),(edges,y_edges),_scale(q,pa)):
        reference=[_exponential_bin(lo,hi,re,scale,psf,pixel) for lo,hi in zip(axis_edges[:-1],axis_edges[1:])]
        np.testing.assert_allclose(result[axis+'_probability'],reference,rtol=3e-10,atol=2e-11)


def test_mixing_tail_is_not_renormalized_and_metadata_accounts_for_mass():
    tail=1e-5
    result=_reference(n=1.,x=np.array([-1000.,0.,1000.]),
                      absolute_tolerance=1e-11,mixing_tail_probability=tail)
    metadata=result['metadata']
    assert metadata['schema']=='cess.sersic_marginal_reference' and metadata['version']==1
    assert metadata['integration']['converged'] is True
    assert metadata['normalization']['total_source_probability']==1.
    assert metadata['normalization']['orthogonal_domain']=='all real'
    assert metadata['normalization']['window_renormalized'] is False
    assert metadata['integration']['omitted_mixing_tail_probability']==pytest.approx(tail,rel=2e-10)
    assert 'mixture' in metadata['integration']['truncation_variable'].lower()
    assert 'omitted_radial_tail_probability' not in metadata['integration']
    json.dumps(metadata,allow_nan=False)
    for axis in ('x','y'):
        returned=float(result[axis+'_probability'].sum())
        assert returned==pytest.approx(1-tail,abs=2e-10)
        assert returned<1-tail/2
        window=metadata['windows'][axis]
        assert window['returned_probability']==pytest.approx(returned,abs=2e-13)
        assert window['outside_probability_integrated']>=0
        assert window['probability_accounted']==pytest.approx(returned+window['outside_probability_integrated'],abs=2e-13)
        # The same exact omitted mixing probability accounts for every axis;
        # it is not a known physical radius cutoff.
        assert returned+window['outside_probability_integrated']+tail==pytest.approx(1.,abs=2e-10)


@pytest.mark.parametrize('n',[.5,1.])
def test_window_growth_retains_central_absolute_bin_mass(n):
    kw=dict(n=n,absolute_tolerance=1e-11,mixing_tail_probability=1e-13)
    small=_reference(x=np.array([-1.5,-.5,.5,1.5]),**kw)
    large=_reference(x=np.array([-100.,-1.5,-.5,.5,1.5,100.]),**kw)
    for axis in ('x','y'):
        np.testing.assert_allclose(small[axis+'_probability'],large[axis+'_probability'][1:-1],rtol=2e-11,atol=2e-12)
        assert small[axis+'_probability'].sum()<large[axis+'_probability'].sum()
        assert small[axis+'_probability'].sum()<.8
        assert large[axis+'_probability'].sum()<=1+2e-12


@pytest.mark.parametrize('n',[.5,1.])
def test_coordinate_rescaling_and_unresolved_source_limit(n):
    edges=np.array([-6.,-1.5,-.5,.5,1.5,6.])
    kw=dict(n=n,x=edges,absolute_tolerance=1e-11,mixing_tail_probability=1e-13)
    original=_reference(re=2.5,pixel=.074,**kw)
    scaled=_reference(n=n,re=7.5,pixel=.074/3,x=edges*3,
                      absolute_tolerance=1e-11,mixing_tail_probability=1e-13)
    point=_reference(n=n,re=1e-7,x=edges,absolute_tolerance=1e-11,mixing_tail_probability=1e-13)
    sigma=.2625/.074/np.sqrt(8*np.log(2))
    normal=_normal_mass(edges[:-1]/sigma,edges[1:]/sigma)
    for axis in ('x','y'):
        np.testing.assert_allclose(original[axis+'_probability'],scaled[axis+'_probability'],rtol=3e-12,atol=2e-12)
        np.testing.assert_allclose(point[axis+'_probability'],normal,rtol=0,atol=3e-11)


def test_deterministic_reference_does_not_change_rng_or_inputs():
    state=np.random.get_state()
    edges=np.array([-5.,-.5,.5,5.]);before=edges.copy()
    try:
        np.random.seed(9281)
        expected=np.random.get_state()
        result=_reference(n=1.,x=edges)
        after=np.random.get_state()
        assert expected[0]==after[0] and expected[2:]==after[2:]
        np.testing.assert_array_equal(expected[1],after[1])
        np.testing.assert_array_equal(edges,before)
        assert result['x_probability'].shape==(len(edges)-1,)
    finally:np.random.set_state(state)


@pytest.mark.parametrize('change',[
    {'n':.3},{'n':2.},{'n':True},{'re':0.},{'re':-1.},{'re':np.nan},
    {'q':0.},{'q':1.1},{'q':np.inf},{'pa':np.nan},{'psf':-1.},{'psf':np.inf},
    {'pixel':0.},{'pixel':np.nan},{'x':np.array([0.])},{'x':np.array([0.,0.])},
    {'x':np.array([1.,0.])},{'x':np.array([0.,np.nan])},
    {'absolute_tolerance':0.},{'absolute_tolerance':np.inf},
    {'mixing_tail_probability':0.},{'mixing_tail_probability':1.},
    {'max_subdivisions':0},{'max_subdivisions':True},
])
def test_invalid_or_unimplemented_inputs_fail_without_fallback(change):
    with pytest.raises((ValueError,TypeError)):_reference(**change)


def test_exponential_resource_limit_fails_explicit_convergence():
    with pytest.raises(RuntimeError) as captured:
        _reference(n=1.,re=1.,psf=0.,x=np.array([-.1,0.,.1]),
                   absolute_tolerance=1e-13,mixing_tail_probability=1e-15,max_subdivisions=1)
    assert hasattr(captured.value,'diagnostics')


def test_bin_resource_boundary_and_invalid_other_axis_fail_early():
    edges=np.linspace(-20.,20.,2050)  # 2049 bins: documented permitted boundary
    result=_reference(x=edges)
    assert result['x_probability'].shape==(2049,)
    with pytest.raises(ValueError):_reference(x=np.linspace(-20.,20.,2051))
    with pytest.raises(ValueError):_reference(y=np.array([0.,-1.]))


def test_exact_quarter_turn_survives_extreme_axis_ratio():
    # A residual cos(pi/2) amplified by 1/sqrt(q) would broaden the minor axis
    # by hundreds of orders of magnitude and erase these central bin masses.
    edges=np.array([-1e10,-1e-290,0.,1e-290,1e10])
    zero=_reference(re=1e-140,q=1e-300,pa=0.,psf=0.,x=edges)
    quarter=_reference(re=1e-140,q=1e-300,pa=90.,psf=0.,x=edges)
    np.testing.assert_array_equal(zero['x_probability'],quarter['y_probability'])
    np.testing.assert_array_equal(zero['y_probability'],quarter['x_probability'])
    assert zero['y_probability'][1:3].sum()>.7
    assert quarter['x_probability'][1:3].sum()>.7


def test_edge_units_and_mixed_booleans_are_not_silently_coerced():
    from astropy import units as u
    with pytest.raises(ValueError):_reference(x=np.array([-1.,1.])*u.arcsec)
    with pytest.raises(ValueError):_reference(x=[-1.,True,2.])
    with pytest.raises(ValueError):_reference(y=[-1.,np.bool_(False),1.])
