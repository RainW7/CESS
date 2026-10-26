"""Native 3.0 pipeline inheritance; no change to the forward scientific model."""
import importlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cess.cli.main import parser
from cess.config.loader import load_config
from cess.io.products import open_product, ROOT_GROUPS, validate_structure
from test_pipeline_smoke import make_input
from test_task_context import assert_values_equal
from test_cli_repair import call_cli, fixture_files


def compare_science(legacy_path, structured_path):
    with h5py.File(str(legacy_path), 'r') as legacy, open_product(structured_path) as product:
        view = product.legacy_view()
        assert product.schema_version == '3.0'
        assert set(product.handle) == set(ROOT_GROUPS)
        validate_structure(product.handle)
        assert view.attrs['schema_version'] == '2.5'
        names = []
        legacy.visititems(lambda name, obj: names.append(name)
                          if isinstance(obj, h5py.Dataset) else None)
        actual = []
        view.visititems(lambda name, obj: actual.append(name)
                        if isinstance(obj, h5py.Dataset) else None)
        # EL root peak tables also have convenient logical band aliases. Visit
        # once per object, while every original path must still resolve below.
        identity = lambda name: h5py.h5o.get_info(view[name].id).addr
        assert {identity(name) for name in actual} == {identity(name) for name in names}
        for name in names:
            before, after = legacy[name], view[name]
            assert before.dtype == after.dtype
            assert_values_equal(before[()], after[()])
            assert set(before.attrs) == set(after.attrs)
            for key in before.attrs:
                np.testing.assert_array_equal(before.attrs[key], after.attrs[key])


@pytest.mark.parametrize('mode', ['standard', 'paired'])
@pytest.mark.parametrize('count,invalid', [(0, ()), (1, (0,)), (3, (0,))])
def test_native_default_preserves_legacy_datasets(tmp_path, mode, count, invalid):
    source = tmp_path / 'template.hdf5'
    make_input(source, count=count, invalid=invalid)
    driver = importlib.import_module('cess.pipeline.' + mode)
    config = load_config(ROOT / 'CESS_v0.8.6/emulator_parameters.json')
    options = dict(morph=True, wave_cal=True, photo=True, el_detect=True)
    old = driver.task(str(tmp_path), source.name, config=config, options=options,
                      seed=1214, output_dir=str(tmp_path / 'old'))
    new = driver.simulate_file(str(tmp_path), source.name, config=config, options=options,
                              seed=1214, output_dir=str(tmp_path / 'new'))
    compare_science(old, new)
    with open_product(new) as product:
        assert len(product.catalog['ID']) == count
        if count:
            assert product.catalog['data_mask'][0] == 2


@pytest.mark.parametrize('mode', ['standard', 'paired'])
def test_emission_subselection_accepts_structured_parent(tmp_path, mode):
    source = tmp_path / 'template.hdf5'
    make_input(source, count=4, invalid=(0,))
    config = load_config(ROOT / 'CESS_v0.8.6/emulator_parameters.json')
    simulation = importlib.import_module('cess.pipeline.' + mode)
    from cess.pipeline.emission import simulate_file
    parent = Path(simulation.simulate_file(str(tmp_path), source.name, config=config,
        options=dict(morph=True, wave_cal=True, photo=True, el_detect=True), seed=1214,
        output_dir=str(tmp_path / 'parent'), source_start=1, source_stop=4))
    old = simulate_file(str(parent.parent), parent.name, config=config,
        output_dir=str(tmp_path / 'old'), source_start=1, source_stop=3, output_layout='legacy')
    new = simulate_file(str(parent.parent), parent.name, config=config,
        output_dir=str(tmp_path / 'new'), source_start=1, source_stop=3)
    compare_science(old, new)
    with open_product(new) as product:
        np.testing.assert_array_equal(product.catalog['input_row_indices'][:], [2, 3])
        np.testing.assert_array_equal(product.catalog['parent_row_indices'][:], [1, 2])
        assert product.handle.attrs['product_kind'] == 'emission_line_postprocess'


def test_cli_default_and_structured_manifest_resume(tmp_path):
    assert parser().parse_args(['simulate']).output_layout == 'structured'
    assert parser('legacy').parse_args(['simulate']).output_layout == 'legacy'
    source, config_path, _ = fixture_files(tmp_path, count=2)
    manifest = tmp_path / 'run.json'
    args = ['simulate', '--config', str(config_path), '--output-layout', 'structured',
            '--manifest', str(manifest), '--seed', '1214']
    first = call_cli(tmp_path, args)
    assert first.returncode == 0, first.stdout
    state = json.loads(manifest.read_text())
    assert state['snapshot']['output_layout'] == 'structured'
    product = Path(state['results'][0]['output'])
    before = product.read_bytes(), product.stat().st_mtime_ns
    resumed = call_cli(tmp_path, args + ['--resume'])
    assert resumed.returncode == 0, resumed.stdout
    assert (product.read_bytes(), product.stat().st_mtime_ns) == before
    wrong = list(args)
    wrong[wrong.index('structured')] = 'legacy'
    rejected = call_cli(tmp_path, wrong + ['--resume'])
    assert rejected.returncode == 2 and 'snapshot mismatch' in rejected.stdout
    assert (product.read_bytes(), product.stat().st_mtime_ns) == before


def test_bad_output_layout_rejected_before_loading_resources(tmp_path):
    from cess.pipeline.standard import simulate_file
    with pytest.raises(ValueError, match='output_layout'):
        simulate_file(str(tmp_path), 'absent.h5', config='/absent.json', output_layout='bad')
