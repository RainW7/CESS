"""Task-local state and numerical parity with retained compatibility entrypoints."""
import importlib
from pathlib import Path
import random
import sys

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cess.config.loader import load_config
from cess.morphology import legacy
from cess.morphology.sampling import MorphologySampler
from cess.pipeline.context import SimulationContext
from test_pipeline_smoke import make_input


def test_contexts_do_not_change_legacy_config_or_share_nested_state():
    config = load_config(ROOT/'CESS_v0.8.6/emulator_parameters.json')
    legacy.configure(config)
    active = legacy._config
    first = SimulationContext(config, {'photo': True})
    other = dict(config, arcsecperpix=.1)
    second = SimulationContext(other)
    first.config['photometry_exposures']['u'] = 99
    first.options['apply_el_detect'] = True
    assert second.config['photometry_exposures']['u'] == config['photometry_exposures']['u']
    assert not second.options['apply_el_detect']
    assert not second.options['apply_photometric']
    assert first.morphology is not second.morphology
    assert first.morphology._gems_data is second.morphology._gems_data is None
    assert legacy._config is active
    assert first.config['arcsecperpix'] == .074
    assert second.config['arcsecperpix'] == .1


def test_task_sampler_preserves_draws_and_random_state():
    config = load_config(ROOT/'CESS_v0.8.6/emulator_parameters.json')
    sampler = MorphologySampler(config)
    legacy.configure(config)
    arguments = [(18.,14.,21.,.5,5.,10.,z) for z in (.01,.2,.7)]*3
    np.random.seed(1214)
    expected = [legacy.get_2d_param(*args) for args in arguments]
    after = np.random.random(4)
    np.random.seed(1214)
    actual = [sampler.get_2d_param(*args) for args in arguments]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.random.random(4), after)
    changed = dict(config, arcsecperpix=.1)
    second = MorphologySampler(changed)
    second.get_ar()
    assert sampler._gems_data is not second._gems_data
    assert sampler._distribution_cache is not second._distribution_cache


def assert_values_equal(first, second):
    assert first.dtype == second.dtype and first.shape == second.shape
    if first.dtype.names:
        for name in first.dtype.names:
            assert_values_equal(first[name], second[name])
    elif first.dtype.kind == 'O':
        for before, after in zip(first.flat, second.flat):
            np.testing.assert_array_equal(before, after)
    else:
        np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize('mode', ['standard', 'paired'])
@pytest.mark.parametrize('seed', [None, 1214])
def test_modern_context_matches_legacy_entry_values(tmp_path, mode, seed):
    driver = importlib.import_module('cess.pipeline.'+mode)
    source = tmp_path/'fixture.hdf5'
    make_input(source, count=3, invalid=(0,))
    config = load_config(ROOT/'CESS_v0.8.6/emulator_parameters.json')
    options = dict(morph=True, wave_cal=True, photo=True, el_detect=True)
    products = []
    for label, function in (('compatibility', driver.task), ('context', driver.simulate_file)):
        np.random.seed(733)
        random.seed(811)
        products.append(function(str(tmp_path), source.name, output_dir=str(tmp_path/label),
            config=config, options=options, seed=seed,
            **({'output_layout': 'legacy'} if label == 'context' else {})))
    with h5py.File(products[0], 'r') as before, h5py.File(products[1], 'r') as after:
        names = []
        before.visititems(lambda name, value: names.append(name) if isinstance(value,h5py.Dataset) else None)
        after_names = []
        after.visititems(lambda name, value: after_names.append(name) if isinstance(value,h5py.Dataset) else None)
        assert names == after_names
        for name in names:
            assert_values_equal(before[name][:], after[name][:])
