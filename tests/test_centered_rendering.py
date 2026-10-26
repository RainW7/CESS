"""Explicit odd-stamp centering; the default renderer remains the legacy recipe."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import ndtr

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'CESS_v0.8.6'))
import morphology


@pytest.fixture(autouse=True)
def configured():
    morphology.configure()


def test_legacy_default_keeps_draw_call_and_exact_arrays(monkeypatch):
    calls=[]
    original=morphology.galsim.GSObject.drawImage
    def spy(self,*args,**kwargs):
        calls.append(kwargs.copy())
        return original(self,*args,**kwargs)
    monkeypatch.setattr(morphology.galsim.GSObject,'drawImage',spy)
    default=morphology.get_2d_profile(1.,4.,0.,1.)
    explicit=morphology.get_2d_profile(1.,4.,0.,1.,centered=False)
    for a,b in zip(default,explicit):np.testing.assert_array_equal(a,b)
    for kwargs in calls:
        assert set(kwargs)=={'wcs','offset'}
        assert kwargs['offset'].x==kwargs['offset'].y==.5


def test_odd_float64_stamp_centers_retained_nonfallback_profile(monkeypatch):
    calls=[]
    original=morphology.galsim.GSObject.drawImage
    def spy(self,*args,**kwargs):
        image=original(self,*args,**kwargs)
        calls.append((kwargs.copy(),image.array.dtype))
        return image
    monkeypatch.setattr(morphology.galsim.GSObject,'drawImage',spy)
    width,height=morphology.get_2d_profile(4.,20.,90.,.3,centered=True)
    kwargs,dtype=calls[0]
    assert kwargs['nx']==kwargs['ny'] and kwargs['nx']%2==1
    assert kwargs['offset'].x==kwargs['offset'].y==0.
    assert dtype==np.dtype('float64')
    assert 4<=len(width)<=20 and width.sum()<.8  # retained crop, not Gaussian fallback
    for profile in (width,height):
        assert np.all(np.isfinite(profile)) and np.all(profile>=0)
        x=np.arange(len(profile))-(len(profile)-1)/2
        assert abs(np.dot(x,profile)/profile.sum())<1e-4


def _capture_marginals(monkeypatch,parameters):
    captured=[]
    original=morphology.extraction_height
    def spy(profile,target):
        captured.append(np.array(profile,copy=True))
        return original(profile,target)
    with monkeypatch.context() as patch:
        patch.setattr(morphology,'extraction_height',spy)
        result=morphology.get_2d_profile(*parameters,centered=True)
    return result,captured


def test_circular_gaussian_pixel_integrals_match_independent_cdf(monkeypatch):
    # GalSim auto/FFT accuracy is retained; this is not a new FFT precision model.
    (width,height),marginals=_capture_marginals(monkeypatch,(.5,2.5,0.,1.))
    config=morphology._ensure_config()
    sigma=np.sqrt((2.5/np.sqrt(2*np.log(2)))**2+
        (config.get('morphology_psf_fwhm',.2625)/config['arcsecperpix']/np.sqrt(8*np.log(2)))**2)
    x=np.arange(len(marginals[1]))-(len(marginals[1])-1)/2
    reference=ndtr((x+.5)/sigma)-ndtr((x-.5)/sigma)
    reference/=reference.sum()  # same finite-stamp normalization, not all-source fraction
    np.testing.assert_allclose(marginals[1],reference,rtol=0,atol=1.5e-7)
    expected=morphology.extraction_height(reference,.9)
    assert len(height)==len(expected)==9
    np.testing.assert_allclose(height,expected,rtol=0,atol=1.5e-7)
    np.testing.assert_allclose(height,height[::-1],rtol=0,atol=1e-13)


def test_centered_mirrored_pa_and_quarter_turn_exchange_marginals(monkeypatch):
    _,zero=_capture_marginals(monkeypatch,(1.,4.,0.,.6))
    _,quarter=_capture_marginals(monkeypatch,(1.,4.,90.,.6))
    _,positive=_capture_marginals(monkeypatch,(1.,4.,30.,.6))
    _,negative=_capture_marginals(monkeypatch,(1.,4.,-30.,.6))
    for a,b in zip(zero,quarter[::-1]):np.testing.assert_allclose(a,b,rtol=0,atol=1e-13)
    for a,b in zip(positive,negative):np.testing.assert_allclose(a,b,rtol=0,atol=1e-13)


@pytest.mark.parametrize('value',[0,1,None,'true'])
def test_centered_requires_explicit_boolean(value):
    with pytest.raises(ValueError,match='centered must be boolean'):
        morphology.get_2d_profile(1.,4.,0.,1.,centered=value)


def test_centered_negative_fft_projection_is_rejected_with_context():
    # A real small case with negative projected FFT tails: no clipping is allowed.
    with pytest.raises(ValueError,match=r'Centered profile .*n=0.5.*failed without clipping'):
        morphology.get_2d_profile(.5,3.,0.,1.,centered=True)
