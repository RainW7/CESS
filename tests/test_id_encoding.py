"""Canonical writers preserve HDF5 ID encoding for strict resume verification."""
import importlib
from pathlib import Path
import sys
import h5py
import numpy as np
import pytest
from test_pipeline_smoke import make_input

CORE = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(CORE))
from cess.config.loader import load_config
from cess.cli.main import build_jobs
from cess.io.manifest import snapshot, verify_product


@pytest.mark.parametrize('driver_name,mode', [('cess.pipeline.standard', 'standard'), ('cess.pipeline.paired', 'paired-wce')])
def test_utf8_id_round_trip_and_resume(tmp_path, driver_name, mode):
    source = tmp_path / 'unicode.hdf5'
    make_input(source, count=1)
    with h5py.File(str(source), 'r+') as handle:
        del handle['ID']
        handle.create_dataset('ID', data=['source-α'], dtype=h5py.string_dtype('utf-8'))
    config = load_config()
    config['template_lsf'] = {'mode': 'delta'}
    options = {}
    jobs = build_jobs([source], tmp_path / 'output', config, options, mode=mode, seed=1214)
    state = snapshot(config, options, mode, 1214, jobs)
    driver = importlib.import_module(driver_name)
    output = driver.task(str(tmp_path), source.name, config=config, output_dir=tmp_path / 'output',
                         options=options, seed=1214)
    with h5py.File(str(output), 'r') as product:
        assert h5py.check_string_dtype(product['ID'].dtype).encoding == 'utf-8'
        np.testing.assert_array_equal(product['ID'][:], ['source-α'.encode('utf-8')])
    assert verify_product(jobs[0], state)['output'] == str(Path(output).resolve())
