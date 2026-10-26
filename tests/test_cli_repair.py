import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

from test_pipeline_smoke import make_input

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "CESS_v0.8.6"
sys.path.insert(0, str(CORE))
from cess_cli import build_jobs, discover_inputs, estimate_peak_gib
from cess_config import load_config


def fixture_files(directory, count=3):
    inputs = directory / "input"
    inputs.mkdir()
    make_input(inputs / "synthetic.hdf5", count=count)
    config = load_config(CORE / "emulator_parameters.json")
    config["input_path"] = str(inputs)
    config["output_path"] = str(directory / "output")
    config_path = directory / "parameters.json"
    config_path.write_text(json.dumps(config))
    return inputs / "synthetic.hdf5", config_path, config


def call_cli(directory, arguments):
    environment = dict(os.environ, MPLCONFIGDIR=str(directory / "mpl"),
                       PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
                       OMP_NUM_THREADS="1")
    return subprocess.run([sys.executable, "-B", str(CORE / "cess_cli.py")] + arguments,
                          cwd=str(directory), env=environment, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, universal_newlines=True, timeout=90)


def test_input_resolution_and_job_chunk_intervals(tmp_path):
    input_path, _, config = fixture_files(tmp_path, count=5)
    files = discover_inputs([str(input_path.parent), str(input_path)])
    assert files == [input_path.resolve()]
    jobs = build_jobs(files, tmp_path / "out", config, {}, seed=8,
                      source_start=1, source_stop=5, chunk_size=2)
    assert [(job["source_start"], job["source_stop"]) for job in jobs] == [(1, 3), (3, 5)]
    assert all(job["estimated_peak_gib"] > 0 for job in jobs)
    with pytest.raises(ValueError):
        discover_inputs([str(tmp_path / "missing*.hdf5")])
    with pytest.raises(ValueError):
        build_jobs(files, tmp_path, config, {}, source_stop=6)


def test_dry_run_does_not_create_output(tmp_path):
    _, config_path, _ = fixture_files(tmp_path)
    result = call_cli(tmp_path, ["simulate", "--config", str(config_path),
                               "--chunk-size", "2", "--workers", "8", "--dry-run"])
    assert result.returncode == 0, result.stdout
    plan = json.loads(result.stdout)
    assert len(plan["jobs"]) == 2 and plan["workers"] == 2
    assert not (tmp_path / "output").exists()


def test_input_stem_collision_is_rejected_before_simulation(tmp_path):
    (tmp_path / 'same.h5').write_bytes(b'first')
    (tmp_path / 'same.hdf5').write_bytes(b'second')
    with pytest.raises(ValueError, match='same output stem'):
        discover_inputs([str(tmp_path)])


def test_cli_worker_chunks_preserve_source_randomness_and_no_overwrite(tmp_path):
    _, config_path, _ = fixture_files(tmp_path, count=3)
    full, chunked = tmp_path / "full", tmp_path / "chunked"
    base = ["simulate", "--config", str(config_path), "--seed", "1214", "--morphology"]
    result = call_cli(tmp_path, base + ["--output", str(full), "--workers", "1"])
    assert result.returncode == 0, result.stdout
    result = call_cli(tmp_path, base + ["--output", str(chunked), "--workers", "2", "--chunk-size", "2"])
    assert result.returncode == 0, result.stdout
    with h5py.File(str(next(full.glob("*.hdf5"))), "r") as handle:
        expected = {name: handle[name][:] for name in
                    ["ID", "parameters_desi", "parameters_grism", "GI/flux_ujy_with_noise"]}
    pieces = []
    for path in chunked.glob("*.hdf5"):
        with h5py.File(str(path), "r") as handle:
            pieces.append((handle["input_row_indices"][:],
                           {name: handle[name][:] for name in expected}))
    rows = np.concatenate([part[0] for part in pieces])
    order = np.argsort(rows)
    for name in expected:
        joined = np.concatenate([part[1][name] for part in pieces], axis=0)[order]
        if name == "ID":
            np.testing.assert_array_equal(joined, expected[name])
        else:
            np.testing.assert_allclose(joined, expected[name], rtol=0, atol=0, equal_nan=True)
    again = call_cli(tmp_path, base + ["--output", str(full)])
    assert again.returncode == 1
    assert "FileExistsError" in again.stdout


def test_memory_budget_rejects_too_large_task_without_running(tmp_path):
    _, config_path, _ = fixture_files(tmp_path)
    result = call_cli(tmp_path, ["simulate", "--config", str(config_path),
                               "--memory-budget-gib", "0.001", "--dry-run"])
    assert result.returncode == 2
    assert "reduce --chunk-size" in result.stdout
