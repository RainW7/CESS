"""Explicit spectral source-light fractions and detector extraction areas."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'CESS_v0.8.6'))
from cess_config import load_config
from main import emulator, h


def simulator(dc=.05, read=2.):
    config = load_config()
    config.update(radi=1./np.sqrt(np.pi), expt=2., expnum=3,
                  dc_value=dc, read_noise=read)
    return emulator(config=config)


def arguments(n=51):
    wave = np.linspace(4000., 6000., n)
    return (0.,0.,wave,np.full(n,1.e-25),wave,np.zeros(n),np.full(n,.5),
            np.ones(9)/9.,np.array([.2,.6,.2]))


@pytest.mark.parametrize('fraction', [0., .4, 1.])
@pytest.mark.parametrize('morph', [False, True])
@pytest.mark.parametrize('folded', [False, True])
def test_fraction_applies_once_after_response_and_morphology(fraction, morph, folded):
    engine, args = simulator(), arguments()
    wave = args[2]
    total_time = engine.config['expt']*engine.config['expnum']
    rate = np.full(len(wave), 10.); rate[25] = 100.
    original_rate = rate.copy()
    if folded:
        full_source = rate * engine.colarea * total_time
    else:
        full_source = (args[3]/(h*wave) * engine.colarea * total_time *
                       np.gradient(wave) * args[6])
    if morph:
        full_source = np.convolve(full_source, [.2,.6,.2], mode='same')
    result = engine._noise_realization(*args,morph,rng=np.random.RandomState(51),
        source_rate_e_s_cm2=rate if folded else None,sky_rate_e_s_pixel=.7,
        source_fraction=fraction,extraction_n_pixels=5)
    valid = result['valid']
    expected = fraction*full_source
    background = (.7*total_time + engine.config['dc_value']*total_time +
                  engine.config['read_noise']**2*engine.config['expnum'])*5
    np.testing.assert_allclose(result['model'][valid], expected[valid], rtol=3.e-15, atol=0.)
    np.testing.assert_allclose(result['variance'][valid], expected[valid]+background, rtol=3.e-15)
    assert result['source_fraction'] == fraction and result['extraction_n_pixels'] == 5
    np.testing.assert_array_equal(rate, original_rate)
    assert valid.all() if not morph else (not valid[0] and not valid[-1] and valid[1:-1].all())


def test_extraction_pixel_count_changes_only_background_and_read_terms():
    engine, args = simulator(), arguments()
    results = [engine._noise_realization(*args,False,rng=np.random.RandomState(1),
        source_rate_e_s_cm2=np.full(51,10.),sky_rate_e_s_pixel=.7,
        source_fraction=.4,extraction_n_pixels=count) for count in (2,7)]
    a,b = results
    np.testing.assert_array_equal(a['model'],b['model'])
    np.testing.assert_allclose((b['variance']-b['model'])/(a['variance']-a['model']), 7./2., rtol=2.e-15)


@pytest.mark.parametrize('explicit_count', [None, 4])
def test_height_sum_never_sets_source_fraction_or_weighted_extraction(explicit_count):
    engine, args = simulator(), list(arguments())
    outcomes = []
    for total in (.1,1.,3.):
        args[7] = np.ones(9)*total/9.
        outcomes.append(engine._noise_realization(*args,False,rng=np.random.RandomState(41),
            source_rate_e_s_cm2=np.full(51,10.),sky_rate_e_s_pixel=.7,
            source_fraction=.4,extraction_n_pixels=explicit_count))
    for result in outcomes:
        assert result['extraction_n_pixels'] == (9 if explicit_count is None else 4)
        for field in ('model','observed','variance','valid'):
            np.testing.assert_array_equal(result[field],outcomes[0][field])


@pytest.mark.parametrize('fraction', [True,np.bool_(False),-.1,1.1,np.nan,np.inf,'0.4',np.array(.4)])
def test_invalid_source_fraction_rejected(fraction):
    with pytest.raises(ValueError, match='source_fraction'):
        simulator()._noise_realization(*arguments(),False,source_fraction=fraction,
            sky_rate_e_s_pixel=0.)


@pytest.mark.parametrize('count', [True,np.bool_(False),0,-1,1.1,np.nan,np.inf,'3',np.array(3)])
def test_invalid_extraction_count_rejected(count):
    with pytest.raises(ValueError, match='extraction_n_pixels'):
        simulator()._noise_realization(*arguments(),False,extraction_n_pixels=count,
            sky_rate_e_s_pixel=0.)


@pytest.mark.parametrize('count', [3,3.,np.int64(3),np.float64(3)])
def test_positive_integral_numeric_pixel_counts_accepted(count):
    result = simulator()._noise_realization(*arguments(),False,
        extraction_n_pixels=count,sky_rate_e_s_pixel=0.)
    assert result['extraction_n_pixels'] == 3 and isinstance(result['extraction_n_pixels'],int)


@pytest.mark.parametrize('paired', [False, True])
@pytest.mark.parametrize('morph', [False, True])
@pytest.mark.parametrize('wce', [False, True])
@pytest.mark.parametrize('covariance', [False, True])
def test_explicit_unity_defaults_preserve_tuple_and_random_sequence(paired,morph,wce,covariance):
    engine, args = simulator(), arguments()
    method = engine.add_noise_zouhu if paired else engine.add_noise
    first_rng, second_rng = np.random.RandomState(315), np.random.RandomState(315)
    kwargs = dict(source_rate_e_s_cm2=np.full(51,10.),sky_rate_e_s_pixel=.7,
                  return_covariance=covariance)
    implicit = method(*args,'GV',morph,wce,rng=first_rng,**kwargs)
    explicit = method(*args,'GV',morph,wce,rng=second_rng,
                      source_fraction=1.,extraction_n_pixels=len(args[7]),**kwargs)
    assert len(implicit) == (14 if paired else 9) + ((2 if paired else 1) if covariance else 0)
    for actual,expected in zip(implicit,explicit):
        np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(first_rng.random_sample(16),second_rng.random_sample(16))


@pytest.mark.parametrize('fraction', [0.,.4,1.])
@pytest.mark.parametrize('regime', ['source_only','background_only','read_only','mixed'])
def test_fixed_seed_monte_carlo_mean_and_variance(fraction,regime):
    source = 10. if regime in ('source_only','mixed') else 0.
    sky = .7 if regime in ('background_only','mixed') else 0.
    dc = .05 if regime in ('background_only','mixed') else 0.
    rn = 2. if regime in ('read_only','mixed') else 0.
    engine, args = simulator(dc,rn), arguments(20000)
    total_time = engine.config['expt']*engine.config['expnum']
    signal = fraction*source*engine.colarea*total_time
    variance = signal + (sky*total_time+dc*total_time+rn**2*engine.config['expnum'])*5
    result = engine._noise_realization(*args,False,rng=np.random.RandomState(20261021),
        source_rate_e_s_cm2=np.full(len(args[2]),source),sky_rate_e_s_pixel=sky,
        source_fraction=fraction,extraction_n_pixels=5)
    np.testing.assert_allclose(result['model'],signal,rtol=3.e-15,atol=0.)
    np.testing.assert_allclose(result['variance'],variance,rtol=3.e-15,atol=0.)
    if variance == 0:
        np.testing.assert_array_equal(result['observed'],np.zeros(len(args[2])))
    else:
        assert abs(result['observed'].mean()-signal) < 5.*np.sqrt(variance/len(args[2]))
        assert abs(result['observed'].var(ddof=1)/variance-1.) < .05


def test_aperture_signal_and_covariance_scaling_and_shared_paired_noise(monkeypatch):
    engine, args = simulator(0.,0.), arguments()
    monkeypatch.setattr(engine,'_sample_wave_offset',lambda rng=None:.5)
    rate = np.full(51,10.); rate[25] = 100.
    kwargs = dict(source_rate_e_s_cm2=rate,sky_rate_e_s_pixel=0.,
                  extraction_n_pixels=5,return_covariance=True)
    full = engine.add_noise(*args,'GV',True,True,rng=np.random.RandomState(7),**kwargs)
    aperture = engine.add_noise(*args,'GV',True,True,rng=np.random.RandomState(7),source_fraction=.4,**kwargs)
    np.testing.assert_allclose(aperture[0],.4*full[0],rtol=3.e-15,equal_nan=True)
    np.testing.assert_allclose(aperture[2],np.sqrt(.4)*full[2],rtol=3.e-15,equal_nan=True)
    np.testing.assert_allclose(aperture[9],.4*full[9],rtol=3.e-15,equal_nan=True)
    paired = engine.add_noise_zouhu(*args,'GV',True,True,rng=np.random.RandomState(7),source_fraction=.4,**kwargs)
    for left,right in [(aperture[0],paired[2]),(aperture[1],paired[4]),
                       (aperture[2],paired[6]),(aperture[4],paired[9]),(aperture[9],paired[15])]:
        np.testing.assert_array_equal(left,right)


def test_nominal_calibration_keeps_aperture_flux_without_total_source_correction():
    class MeanNoise:
        def poisson(self,expectation): return expectation.copy()
        def normal(self,mean,sigma,size): return np.zeros(size)
    engine, args = simulator(0.,0.), arguments()
    kwargs = dict(source_rate_e_s_cm2=np.full(51,10.),sky_rate_e_s_pixel=0.)
    full = engine.add_noise(*args,'GV',False,False,rng=MeanNoise(),**kwargs)
    aperture = engine.add_noise(*args,'GV',False,False,rng=MeanNoise(),source_fraction=.4,**kwargs)
    np.testing.assert_allclose(aperture[1],.4*full[1],rtol=2.e-15)
    np.testing.assert_allclose(aperture[2],np.sqrt(.4)*full[2],rtol=2.e-15)
    np.testing.assert_allclose(aperture[4],np.sqrt(.4)*full[4],rtol=2.e-15)


def test_fraction_and_extraction_are_not_implicitly_read_from_configuration():
    engine, args = simulator(), arguments()
    engine.config.update(source_fraction=.2,extraction_n_pixels=2)
    result = engine._noise_realization(*args,False,sky_rate_e_s_pixel=0.)
    assert result['source_fraction'] == 1. and result['extraction_n_pixels'] == 9
