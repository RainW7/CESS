"""Conditional finite dispersion windows must not restore lost source light."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'CESS_v0.8.6'))
from test_extraction_noise import simulator,arguments


@pytest.mark.parametrize('morph',[False,True])
def test_conditional_window_mass_and_spatial_fraction_each_applied_once(morph):
    engine,args=simulator(),list(arguments())
    args[-1]=np.array([.1,.3,.1])
    rate=np.full(51,10.);rate[25]=100.
    full=rate*engine.colarea*engine.config['expnum']*engine.config['expt']
    expectation=.4*(np.convolve(full,args[-1],mode='same') if morph else full)
    result=engine._noise_realization(*args,morph,rng=np.random.RandomState(9),
        source_rate_e_s_cm2=rate,sky_rate_e_s_pixel=.7,source_fraction=.4,
        extraction_n_pixels=5,normalize_morphology_kernel=False)
    valid=result['valid']
    np.testing.assert_allclose(result['model'][valid],expectation[valid],rtol=2.e-15)
    time=engine.config['expnum']*engine.config['expt']
    background=(.7*time+engine.config['dc_value']*time+engine.config['read_noise']**2*engine.config['expnum'])*5
    np.testing.assert_allclose(result['variance'][valid],expectation[valid]+background,rtol=2.e-15)


@pytest.mark.parametrize('paired',[False,True])
def test_default_normalization_keyword_preserves_values_and_rng(paired):
    engine,args=simulator(),arguments()
    method=engine.add_noise_zouhu if paired else engine.add_noise
    random_a,random_b=np.random.RandomState(5),np.random.RandomState(5)
    a=method(*args,'GV',True,True,rng=random_a)
    b=method(*args,'GV',True,True,rng=random_b,normalize_morphology_kernel=True)
    for left,right in zip(a,b):np.testing.assert_array_equal(left,right)
    np.testing.assert_array_equal(random_a.random_sample(10),random_b.random_sample(10))


def test_probability_kernel_rejects_mass_exceeding_unity():
    with pytest.raises(ValueError,match='sum'):
        simulator().lamshift_width(np.ones(20),np.ones(3),normalize=False)


@pytest.mark.parametrize('value',[1,'yes',None])
def test_normalization_switch_requires_boolean_even_when_morphology_off(value):
    with pytest.raises(ValueError,match='normalize_morphology_kernel'):
        simulator()._noise_realization(*arguments(),False,normalize_morphology_kernel=value)
