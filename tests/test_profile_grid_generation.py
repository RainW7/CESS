"""Profile grid shape/order checks, independent of the rendering recipe."""
import itertools
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'CESS_v0.8.6'))
import morphology


@pytest.fixture(autouse=True)
def quiet_progress(monkeypatch):
    monkeypatch.setattr(morphology,'tqdm',lambda values:values)


@pytest.mark.parametrize('shape', [(1,1,1,1),(2,2,3,2)])
@pytest.mark.parametrize('same_lengths', [False,True])
def test_complete_profile_objects_keep_four_axes_and_legacy_iteration_order(monkeypatch,shape,same_lengths):
    axes = [np.linspace(.5,1.5,shape[0]),np.linspace(2.,4.,shape[1]),
            np.linspace(-30.,120.,shape[2]),np.linspace(.3,1.,shape[3])]
    calls, profiles = [], []
    def render(n,re,pa,ba):
        k=len(calls)
        width=np.arange(3 if same_lengths else 2+k%3,dtype=float)+k+.1
        height=np.arange(5 if same_lengths else 3+k%4,dtype=float)+k+.2
        calls.append((n,re,pa,ba));profiles.append((width,height))
        return width,height
    monkeypatch.setattr(morphology,'get_2d_profile',render)
    copies=[axis.copy() for axis in axes]
    width,height=morphology.create_lib(*axes)
    assert width.shape==height.shape==shape
    assert width.dtype==height.dtype==object
    assert calls==list(itertools.product(*axes))
    for index,original in zip(np.ndindex(shape),profiles):
        assert width[index] is original[0] and height[index] is original[1]
        np.testing.assert_array_equal(width[index],original[0])
        np.testing.assert_array_equal(height[index],original[1])
    for axis,copy in zip(axes,copies):np.testing.assert_array_equal(axis,copy)


def test_original_equal_length_object_array_failure_is_not_reintroduced(monkeypatch):
    vectors=[np.ones(3),np.ones(3)]
    assert np.array(vectors,dtype=object).shape==(2,3)
    with pytest.raises(ValueError):
        np.array(vectors,dtype=object).reshape((2,1,1,1))
    monkeypatch.setattr(morphology,'get_2d_profile',lambda *args:(vectors[0],vectors[1]))
    width,height=morphology.create_lib([.5,1.],[2.],[0.],[1.])
    assert width.shape==height.shape==(2,1,1,1)
    assert width[0,0,0,0].shape==(3,)


@pytest.mark.parametrize('axis',range(4))
@pytest.mark.parametrize('bad',[[],1.,[[1.,2.]],[1.,np.nan],[1.,np.inf],[1.,1.],[2.,1.],
    [True],[.5,True],['1.'],[1.+0.j],np.ma.array([1.],mask=[False])])
def test_invalid_axes_fail_before_any_render(monkeypatch,axis,bad):
    axes=[[.5,1.],[2.,3.],[-10.,10.],[.5,1.]]
    axes[axis]=bad
    calls=[]
    monkeypatch.setattr(morphology,'get_2d_profile',lambda *args:calls.append(args))
    with pytest.raises(ValueError):morphology.create_lib(*axes)
    assert calls==[]


@pytest.mark.parametrize('axis,bad',[(0,[0.]),(0,[-1.]),(0,[.2]),(0,[.5,7.]),
    (1,[0.]),(1,[-1.]),(3,[0.]),(3,[-.5]),(3,[1.1])])
def test_physical_or_renderer_domain_fails_before_any_render(monkeypatch,axis,bad):
    axes=[[.5,1.],[2.,3.],[-10.,10.],[.5,1.]]
    axes[axis]=bad
    calls=[]
    monkeypatch.setattr(morphology,'get_2d_profile',lambda *args:calls.append(args))
    with pytest.raises(ValueError):morphology.create_lib(*axes)
    assert calls==[]


def test_signed_full_angle_and_renderer_endpoints_remain_allowed(monkeypatch):
    calls=[]
    def render(*args):
        calls.append(args)
        return np.array([.2,.8]),np.array([.3,.7])
    monkeypatch.setattr(morphology,'get_2d_profile',render)
    result=morphology.create_lib([.3,6.2],[.1,100.],[-720.,-90.,0.,90.,720.],[.01,1.])
    assert result[0].shape==(2,2,5,2)
    assert calls==list(itertools.product([.3,6.2],[.1,100.],[-720.,-90.,0.,90.,720.],[.01,1.]))


@pytest.mark.parametrize('width,height',[([], [1.]),([[1.]], [1.]),([1.],[]),([1.],[[1.]])])
def test_structurally_invalid_renderer_output_has_grid_location(monkeypatch,width,height):
    monkeypatch.setattr(morphology,'get_2d_profile',lambda *args:(width,height))
    with pytest.raises(ValueError,match=r'profile \(0, 0, 0, 0\)'):
        morphology.create_lib([.5],[2.],[0.],[1.])


def test_container_does_not_clip_normalize_or_reinterpret_renderer_values(monkeypatch):
    width=np.array([-.00000001,.2,.8],dtype=np.float32)
    height=np.array([.01,.4,.2],dtype=np.float32)
    monkeypatch.setattr(morphology,'get_2d_profile',lambda *args:(width,height))
    a,b=morphology.create_lib([.5],[2.],[0.],[1.])
    assert a[0,0,0,0] is width and b[0,0,0,0] is height
    assert a[0,0,0,0][0]<0 and b[0,0,0,0].sum()!=1.
    assert a[0,0,0,0].dtype==b[0,0,0,0].dtype==np.float32


def test_rendering_failure_names_grid_location_and_preserves_cause(monkeypatch):
    original=ValueError('profile contains negative fluxes')
    calls=[]
    def render(*args):
        calls.append(args)
        if len(calls)==2:raise original
        return np.ones(3),np.ones(5)
    monkeypatch.setattr(morphology,'get_2d_profile',render)
    with pytest.raises(ValueError,match=r'Profile grid \(0, 0, 0, 1\).*negative fluxes') as caught:
        morphology.create_lib([1.],[4.],[0.],[.5,1.])
    assert caught.value.__cause__ is original and len(calls)==2


def test_two_real_equal_length_profiles_in_fresh_process(tmp_path):
    core=Path(__file__).resolve().parents[1]/'CESS_v0.8.6'
    code=('import sys,numpy as np;sys.path.insert(0,{!r});'
          'import morphology;from cess_config import load_config;'
          'morphology.configure(load_config());'
          'a,b=morphology.create_lib([1.],[4.],[0.,90.],[1.]);'
          'assert a.shape==b.shape==(1,1,2,1);'
          'assert a.dtype==b.dtype==object;'
          'assert len(a[0,0,0,0])==len(a[0,0,1,0]);'
          'assert len(b[0,0,0,0])==len(b[0,0,1,0]);'
          'assert all(np.isfinite(v).all() and v.sum()>0 for lib in (a,b) for v in lib.flat)')
    result=subprocess.run([sys.executable,'-B','-c',code.format(str(core))],cwd=str(tmp_path),
        env=dict(os.environ,MPLCONFIGDIR=str(tmp_path/'mpl'),PYTHONDONTWRITEBYTECODE='1'),
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
