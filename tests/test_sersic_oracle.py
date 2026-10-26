"""Targeted mathematical checks for the experimental scalar general-n oracle."""
import math
import json
from pathlib import Path
import pickle
import sys
import warnings
import numpy as np
import pytest
from scipy.special import ndtr

CORE = Path(__file__).resolve().parents[1] / 'src'
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))
import cess.morphology.oracle as oracle
def call(n=1., radius=4., angle=0., q=1., lower=-.5, upper=.5, axis='x', **kwargs):
    return oracle.sersic_bin_oracle(n, radius, angle, q, .2625, .074,
                                     lower, upper, axis, **kwargs)


@pytest.mark.parametrize('radius,angle,q,axis', [(20.,45.,.5,'x'),(40.,85.,.1,'y')])
def test_gaussian_old_failures_match_analytic_full_source(radius, angle, q, axis):
    result=call(.5,radius,angle,q,axis=axis)
    a=math.radians(angle)
    factor=(math.cos(a)**2/q+q*math.sin(a)**2 if axis=='x' else
            math.sin(a)**2/q+q*math.cos(a)**2)
    sigma=math.sqrt(radius**2*factor/(2.*math.log(2.))+(.2625/.074)**2/(8.*math.log(2.)))
    expected=float(ndtr(.5/sigma)-ndtr(-.5/sigma))
    assert result['probability']==pytest.approx(expected,abs=2.e-10)
    assert 0.<result['probability']<1.
    diag=result['diagnostics']
    assert json.loads(json.dumps(result,allow_nan=False))==result
    assert result['schema']=='cess.sersic_bin_oracle' and result['version']==1
    assert result['normalization']['window_renormalization'] is False
    assert result['normalization']['source_total_probability']==1.
    assert result['normalization']['orthogonal_domain']=='all real'
    assert diag['combined_quadrature_error_estimate']<=1.e-8
    assert diag['radial_omission_probability_bound']==pytest.approx(1.e-12,rel=1.e-10)


def test_exponential_old_failure_matches_independent_special_api():
    from cess.morphology.reference import sersic_pixel_reference
    result=call(1.,20.,90.,.1)
    independent=sersic_pixel_reference(1.,20.,90.,.1,.2625,.074,[-.5,.5])
    assert result['probability']==pytest.approx(float(independent['x_probability'][0]),abs=2.e-10)


@pytest.mark.parametrize('center,expected', [(0.,.06484851339292817),(20.,.005034651418287707)])
def test_n4_refinement_agrees_without_calling_estimate_a_certified_bound(center,expected):
    coarse=call(4.,20.,90.,.3,center-.5,center+.5,absolute_tolerance=1.e-7)
    fine=call(4.,20.,90.,.3,center-.5,center+.5,absolute_tolerance=1.e-9)
    assert fine['probability']==pytest.approx(expected,abs=2.e-10)
    assert abs(coarse['probability']-fine['probability'])<2.e-9
    assert 'not a certified bound' in fine['diagnostics']['convergence_basis']


def test_pa_mirror_axis_exchange_and_bin_reflection():
    # Three real calls cover both orientation invariants and reflected bins.
    a=call(1.,8.,30.,.3,4.5,5.5,'x')
    b=call(1.,8.,-30.,.3,4.5,5.5,'x')
    c=call(1.,8.,60.,.3,-5.5,-4.5,'y')
    assert a['probability']==pytest.approx(b['probability'],abs=2.e-10)
    assert a['probability']==pytest.approx(c['probability'],abs=2.e-10)


def test_exact_cardinal_angle_handles_extreme_q_and_keeps_absolute_probability():
    result=call(1.,1.,90.,1.e-100)
    sigma=.2625/.074/math.sqrt(8.*math.log(2.))
    expected=float(ndtr(.5/sigma)-ndtr(-.5/sigma))
    assert result['probability']==pytest.approx(expected,abs=2.e-9)
    assert result['diagnostics']['projected_re_pixel']==pytest.approx(1.e-50,rel=1.e-14)
    assert result['probability']<.5  # single finite bin is never renormalized.


def test_budget_failure_has_picklable_diagnostics_and_no_probability():
    with pytest.raises(oracle.OracleConvergenceError,match='budget') as captured:
        call(max_evaluations=1)
    error=pickle.loads(pickle.dumps(captured.value))
    assert error.diagnostics['evaluations']['total']==1
    assert 'probability' not in error.diagnostics


@pytest.mark.parametrize('field,value', [
    ('n',.4),('n',5.1),('n',True),('n',np.nan),('re_pixel',0.),
    ('axis_ratio',0.),('axis_ratio',1.1),('psf_fwhm_arcsec',0.),
    ('pixel_scale_arcsec',0.),('pa_deg',np.inf),('lower_edge_pixel',np.nan),
    ('upper_edge_pixel',-.5),('axis','width'),('absolute_tolerance',0.),
    ('radial_tail_probability',1.),('gaussian_tail_sigma',0.),('max_evaluations',True)])
def test_invalid_arguments_fail_with_diagnostics(field,value):
    args=dict(n=1.,re_pixel=4.,pa_deg=0.,axis_ratio=1.,psf_fwhm_arcsec=.2625,
              pixel_scale_arcsec=.074,lower_edge_pixel=-.5,upper_edge_pixel=.5)
    args[field]=value
    with pytest.raises(oracle.OracleInputError) as caught:
        oracle.sersic_bin_oracle(**args)
    assert isinstance(caught.value.diagnostics,dict)


def test_any_warning_and_quad_nonconvergence_are_rejected(monkeypatch):
    def warning_quad(*args,**kwargs):
        warnings.warn('synthetic warning',UserWarning)
        return .1,0.,{'last':1}
    monkeypatch.setattr(oracle,'quad',warning_quad)
    with pytest.raises(oracle.OracleConvergenceError,match='warning') as caught:
        call()
    assert caught.value.diagnostics['warnings']==['synthetic warning']
    monkeypatch.setattr(oracle,'quad',lambda *args,**kwargs:(.1,0.,{'last':1},'not converged'))
    with pytest.raises(oracle.OracleConvergenceError,match='did not converge'):
        call()


def test_negative_quad_result_is_rejected_without_clipping(monkeypatch):
    monkeypatch.setattr(oracle,'quad',lambda *args,**kwargs:(-1.e-30,0.,{'last':1}))
    with pytest.raises(oracle.OracleConvergenceError,match='invalid value'):
        call()


def test_excess_error_estimate_is_a_failure(monkeypatch):
    monkeypatch.setattr(oracle,'quad',lambda *args,**kwargs:(.1,1.e-3,{'last':1}))
    with pytest.raises(oracle.OracleConvergenceError,match='exceeds'):
        call()


def test_central_cdf_interval_does_not_subtract_two_rounded_halves():
    actual=oracle._normal_interval(-5.e-30,5.e-30)
    assert actual>0.
    assert actual==pytest.approx(1.e-29/math.sqrt(2.*math.pi),rel=2.e-15,abs=0.)
