"""Explicit option mappings are complete task choices, independent of history."""
import importlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'CESS_v0.8.6'), str(ROOT)]
from cess_config import load_config


@pytest.mark.parametrize('name', ['run', 'run_gravity_zouhu', 'run_el_detect'])
def test_explicit_empty_or_partial_options_reset_previous_task_flags(name):
    driver = importlib.import_module(name)
    config = load_config()
    driver.configure(config, options=dict(morph=True, wave_cal=True, photo=True, el_detect=True))
    driver.configure(config, options={})
    assert not any(driver._OPTIONS.values())
    assert not driver.apply_wavelength_calibration_error
    driver.configure(config, options=dict(morph=True, wave_cal=True, el_detect=True))
    driver.configure(config, options={'photo': True})
    assert driver._OPTIONS == dict(apply_morphology_effect=False,
        apply_wavelength_calibration_error=False, apply_photometric=True, apply_el_detect=False)


@pytest.mark.parametrize('name', ['run', 'run_gravity_zouhu', 'run_el_detect'])
def test_none_options_retains_explicit_configure_workflow(name):
    driver = importlib.import_module(name)
    config = load_config()
    driver.configure(config, options={'wave_cal': True})
    driver.configure(config, options=None)
    assert driver.apply_wavelength_calibration_error
    # Leave no persistent flags for subsequent direct API integration checks.
    driver.configure(config, options={})


@pytest.mark.parametrize('options,photo', [(None, False), ({'photo': True}, True)])
def test_job_plans_expand_options_without_using_previous_runner_state(tmp_path, options, photo):
    from cess_cli import build_jobs
    from test_pipeline_smoke import make_input
    driver = importlib.import_module('run_gravity_zouhu')
    config = load_config()
    driver.configure(config, options=dict(morph=True, wave_cal=True, el_detect=True))
    source = tmp_path / 'source.hdf5'
    make_input(source, count=1)
    job = build_jobs([source], tmp_path/'out', config, options)[0]
    assert job['options'] == dict(apply_morphology_effect=False,
        apply_wavelength_calibration_error=False, apply_photometric=photo, apply_el_detect=False)
    driver.configure(config, options={})
