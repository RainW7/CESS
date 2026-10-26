"""GalSim is optional for sampling/lookup; explicitly requested rendering needs it."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from astropy.io import fits

CORE=Path(__file__).resolve().parents[1]/'CESS_v0.8.6'
sys.path.insert(0,str(CORE))

PREFIX='''
import sys,importlib.abc,json,numpy as np
sys.path.insert(0,sys.argv[1])
class NoGalSim(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='galsim' or fullname.startswith('galsim.'):
            raise ModuleNotFoundError('blocked optional GalSim for test')
if sys.argv[2]=='blocked':sys.meta_path.insert(0,NoGalSim())
import morphology
assert 'galsim' not in sys.modules and 'galsim' not in morphology.__dict__
'''


def execute(code,tmp_path,mode='blocked',arguments=()):
    result=subprocess.run([sys.executable,'-B','-c',PREFIX+code,str(CORE),mode]+list(arguments),
        cwd=str(tmp_path),stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    return result.stdout


def test_import_sampling_lookup_and_rng_do_not_need_renderer(tmp_path):
    data=np.array([(1,18.,1.,.4),(1,19.,2.,.7),(1,20.,3.,.9),(0,19.,4.,.2)],
        dtype=[('SCIENCE_FLAG','i4'),('FIT_MAG','f8'),('FIT_N','f8'),('SEX_AR','f8')])
    path=tmp_path/'gems.fits'
    fits.HDUList([fits.PrimaryHDU(),fits.BinTableHDU(data)]).writeto(str(path))
    code='''
from cess_config import load_config
configuration=load_config(sys.argv[1]+'/emulator_parameters.json')
configuration['gems_path']=sys.argv[3]
morphology.configure(configuration)
np.random.seed(1729)
parameters=morphology.get_2d_param(19.,14.,21.,.5,5.,10.,.7)
library=np.empty((1,1,1,1),dtype=object);library[0,0,0,0]=np.array([.2,.6,.2])
selected=morphology.match_input_paramters(library,*parameters,np.array([1.]),np.array([4.]),np.array([0.]),np.array([1.]))
assert selected is library[0,0,0,0]
assert np.all(np.isfinite(parameters))
assert 'galsim' not in sys.modules and 'galsim' not in morphology.__dict__
print(json.dumps({'parameters':list(parameters),'next_random':np.random.random(4).tolist()}))
'''
    blocked=execute(code,tmp_path,arguments=(str(path),))
    unblocked=execute(code,tmp_path,mode='normal',arguments=(str(path),))
    assert json.loads(blocked)==json.loads(unblocked)


@pytest.mark.parametrize('renderer_call',[
    'morphology.get_2d_profile(1.,4.,0.,1.)',
    'morphology.get_2d_profile(1.,4.,0.,1.,centered=True)',
    'morphology.create_lib([1.],[4.],[0.],[1.])',
    'morphology.galsim',
])
def test_missing_renderer_has_explicit_optional_dependency_error(tmp_path,renderer_call):
    execute('''
try:
    '''+renderer_call+'''
except ImportError as error:
    assert 'GalSim' in str(error) and 'optional rendering dependency' in str(error)
    assert error.__cause__ is not None
else:
    raise AssertionError('requested rendering silently succeeded without GalSim')
assert 'galsim' not in morphology.__dict__
''',tmp_path)


def test_explicit_renderer_attribute_is_cached_patchable_and_rng_neutral(tmp_path):
    pytest.importorskip('galsim')
    execute('''
np.random.seed(73);before=np.random.get_state()
module=morphology.galsim
after=np.random.get_state()
assert module is morphology.galsim
assert before[0]==after[0] and before[2:]==after[2:]
assert np.array_equal(before[1],after[1])
marker=object();morphology.galsim=marker
assert morphology._renderer() is marker
morphology.galsim=module
''',tmp_path,mode='normal')


def test_legacy_normal_renderer_matches_direct_original_draw_recipe():
    galsim=pytest.importorskip('galsim')
    import morphology
    morphology.configure()
    configuration=morphology._ensure_config();pixel=configuration['arcsecperpix']
    galaxy=galsim.Sersic(1.,half_light_radius=pixel*4.).shear(q=1.,beta=0.*galsim.degrees)
    psf=galsim.Gaussian(fwhm=configuration.get('morphology_psf_fwhm',.2625))
    stamp=galsim.Convolve([galaxy,psf]).drawImage(wcs=galsim.PixelScale(pixel),offset=galsim.PositionD(.5,.5))
    stamp=stamp*configuration['expt']*configuration['expnum']*np.pi*(configuration['radi']/100)**2
    raw=stamp.array.sum(axis=1)
    expected_height=morphology.extraction_height(raw/raw.sum(),.9)
    assert 9<=len(expected_height)<=200  # no spatial Gaussian substitution
    expected_width=morphology.utils.gaussian(7,1)  # original n=1/Re=4 width rule
    state=np.random.get_state()
    width,height=morphology.get_2d_profile(1.,4.,0.,1.)
    after=np.random.get_state()
    np.testing.assert_array_equal(width,expected_width)
    np.testing.assert_array_equal(height,expected_height)
    assert state[0]==after[0] and state[2:]==after[2:]
    np.testing.assert_array_equal(state[1],after[1])
