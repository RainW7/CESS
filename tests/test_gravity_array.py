"""Local contracts for catalogue-sized Gravity array submissions.

These checks run the payload against synthetic catalogues and fake PBS
allocation variables. They never contact Gravity, execute qsub, or establish
the memory/runtime requirements of a production catalogue.
"""
import hashlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

import h5py
import pytest

from test_pipeline_smoke import make_input


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts/gravity/catalogue_array.py"
PBS_VARIABLES = (
    "PBS_JOBID", "PBS_ENVIRONMENT", "PBS_ARRAYID", "PBS_ARRAY_INDEX",
    "PBS_NODEFILE", "PBS_NP", "num_nodes", "PBS_NUM_NODES",
)


def environment(tmp_path, **changes):
    env = {key: value for key, value in os.environ.items()
           if key not in PBS_VARIABLES and not key.startswith("CESS_")}
    env.update(PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
               PYTHONNOUSERSITE="1", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MPLCONFIGDIR=str(tmp_path / "mpl"))
    env.update(changes)
    return env


def run_driver(tmp_path, *arguments, env=None, driver=DRIVER):
    return subprocess.run(
        [sys.executable, "-s", "-B", str(driver), *map(str, arguments)],
        cwd=tmp_path, env=environment(tmp_path) if env is None else env,
        text=True, capture_output=True, timeout=180,
    )


def catalogue_pair(tmp_path):
    directory = tmp_path / "input catalogues"
    directory.mkdir()
    make_input(directory / "seedcat2_0001_MzLS.hdf5", count=3, invalid=(0,))
    make_input(directory / "seedcat2_0001_DECaLS.hdf5", count=2)
    return directory


def prepare(tmp_path, inputs, run=None, *extra):
    run = run or tmp_path / "array run"
    result = run_driver(
        tmp_path, "prepare", "-i", inputs, "--run-dir", run,
        "--expected-mzls", "1", "--expected-decals", "1",
        "--mzls-slots", "2", "--decals-slots", "2", *extra,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return run, json.loads((run / "plan.json").read_text())


def allocated_environment(tmp_path, **changes):
    nodefile = tmp_path / "one_cpu_nodefile"
    nodefile.write_text(socket.gethostname() + "\n")
    defaults = dict(PBS_JOBID="12345[1].gravity", PBS_ENVIRONMENT="PBS_BATCH",
                    PBS_ARRAYID="1", PBS_NODEFILE=str(nodefile), PBS_NP="1",
                    num_nodes="1")
    defaults.update(changes)
    return environment(tmp_path, **defaults)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepare_freezes_complete_whole_file_plan_without_running(tmp_path):
    inputs = catalogue_pair(tmp_path)
    before = {path.name: (digest(path), path.stat().st_mtime_ns)
              for path in inputs.iterdir()}
    run, plan = prepare(tmp_path, inputs)
    assert set(plan["groups"]) == {"MzLS", "DECaLS"}
    for group, rows in (("MzLS", 3), ("DECaLS", 2)):
        record = plan["groups"][group]
        assert record["slots"] == 2
        assert len(record["jobs"]) == 1
        job = record["jobs"][0]
        assert job["index"] == 1 and job["rows"] == rows
        source = Path(job["input"])
        assert job["bytes"] == source.stat().st_size
        assert job["mtime_ns"] == source.stat().st_mtime_ns
        assert 0 < job["estimated_peak_gib"] <= record["budget_gib"]
        script = run / (group + ".pbs")
        assert script.is_file()
        check = subprocess.run(["/bin/bash", "-n", str(script)], capture_output=True, text=True)
        assert check.returncode == 0, check.stderr
    assert (run / "catalogue_array.py").is_file()
    assert (run / "config.json").is_file()
    assert (run / "plan.sha256").read_text().split()[0] == digest(run / "plan.json")
    assert not list(run.rglob("*.hdf5"))
    assert {path.name: (digest(path), path.stat().st_mtime_ns)
            for path in inputs.iterdir()} == before


@pytest.mark.parametrize("problem", ["count", "survey", "memory"])
def test_rejected_inventory_does_not_create_run_directory(tmp_path, problem):
    inputs = catalogue_pair(tmp_path)
    run = tmp_path / "rejected"
    extra = []
    if problem == "survey":
        make_input(inputs / "unclassified.hdf5", count=1)
    elif problem == "count":
        extra = ["--expected-mzls", "2"]
    else:
        # Large logical arrays require no payload allocation on disk. Inventory
        # should reject the working-set estimate rather than an argparse typo.
        with h5py.File(inputs / "seedcat2_0001_MzLS.hdf5", "r+") as handle:
            for name, shape, dtype in (
                    ("ID", (100000,), "S20"), ("parameters", (100000, 7), "f8"),
                    ("best_fit/z", (100000,), "f8"),
                    ("best_fit/spec_synthetic", (100000, 2079), "f8")):
                del handle[name]
                handle.create_dataset(name, shape=shape, dtype=dtype, chunks=True)
        extra = ["--mzls-memory-gb", "1"]
    result = run_driver(
        tmp_path, "prepare", "-i", inputs, "--run-dir", run,
        "--expected-mzls", "1", "--expected-decals", "1", *extra,
    )
    assert result.returncode != 0
    if problem == "memory":
        assert "budget" in result.stderr.lower(), result.stdout + result.stderr
    assert not run.exists()


@pytest.mark.parametrize("problem", ["two_dimensional_id", "spectrum_rows"])
def test_unaligned_catalogue_shapes_are_rejected_before_campaign_creation(tmp_path, problem):
    inputs = catalogue_pair(tmp_path)
    with h5py.File(inputs / "seedcat2_0001_MzLS.hdf5", "r+") as handle:
        if problem == "two_dimensional_id":
            ids = handle["ID"][:]
            del handle["ID"]
            handle.create_dataset("ID", data=ids.reshape(3, 1))
        else:
            spectrum = handle["best_fit/spec_synthetic"][:2]
            del handle["best_fit/spec_synthetic"]
            handle.create_dataset("best_fit/spec_synthetic", data=spectrum)
    run = tmp_path / "bad schema run"
    result = run_driver(tmp_path, "prepare", "-i", inputs, "--run-dir", run,
                        "--expected-mzls", "1", "--expected-decals", "1")
    assert result.returncode != 0
    assert "one-dimensional" in result.stderr or "unaligned" in result.stderr
    assert not run.exists()


def test_prepare_refuses_existing_run_without_changing_plan(tmp_path):
    inputs = catalogue_pair(tmp_path)
    run, _ = prepare(tmp_path, inputs)
    before = (run / "plan.json").read_bytes()
    result = run_driver(tmp_path, "prepare", "-i", inputs, "--run-dir", run,
                        "--expected-mzls", "1", "--expected-decals", "1")
    assert result.returncode != 0
    assert (run / "plan.json").read_bytes() == before


def test_preview_without_pbs_is_read_only(tmp_path):
    run, _ = prepare(tmp_path, catalogue_pair(tmp_path))
    before = {path.relative_to(run): (path.read_bytes(), path.stat().st_mtime_ns)
              for path in run.rglob("*") if path.is_file()}
    result = run_driver(tmp_path, "run", "--plan", run / "plan.json",
                        "--group", "MzLS", "--index", "1", "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    after = {path.relative_to(run): (path.read_bytes(), path.stat().st_mtime_ns)
             for path in run.rglob("*") if path.is_file()}
    assert after == before


@pytest.mark.parametrize("index", ["0", "2", "-1"])
def test_out_of_range_array_index_is_rejected(tmp_path, index):
    run, _ = prepare(tmp_path, catalogue_pair(tmp_path))
    result = run_driver(tmp_path, "run", "--plan", run / "plan.json",
                        "--group", "MzLS", "--index", index, "--dry-run")
    assert result.returncode != 0
    assert not list(run.rglob("*.hdf5"))


@pytest.mark.parametrize("problem", ["missing", "interactive", "cpus", "nodes", "nodefile"])
def test_execution_rejects_missing_or_excessive_pbs_allocation(tmp_path, problem):
    run, _ = prepare(tmp_path, catalogue_pair(tmp_path))
    env = allocated_environment(tmp_path)
    if problem == "missing":
        env.pop("PBS_JOBID")
    elif problem == "interactive":
        env["PBS_ENVIRONMENT"] = "PBS_INTERACTIVE"
    elif problem == "cpus":
        env["PBS_NP"] = "2"
    elif problem == "nodes":
        env["num_nodes"] = "2"
    else:
        Path(env["PBS_NODEFILE"]).write_text("node-a\nnode-b\n")
    result = run_driver(tmp_path, "run", "--plan", run / "plan.json",
                        "--group", "MzLS", env=env)
    assert result.returncode != 0
    assert not list(run.rglob("*.hdf5"))


@pytest.mark.parametrize("drift", ["input", "config", "plan"])
def test_prepared_run_rejects_changed_identity(tmp_path, drift):
    inputs = catalogue_pair(tmp_path)
    run, _ = prepare(tmp_path, inputs)
    if drift == "input":
        with h5py.File(inputs / "seedcat2_0001_MzLS.hdf5", "r+") as handle:
            handle["best_fit/spec_synthetic"][1, 0] = 123.0
    elif drift == "config":
        config = run / "config.json"
        value = json.loads(config.read_text())
        value["expt"] += 1
        config.write_text(json.dumps(value))
    else:
        plan = run / "plan.json"
        value = json.loads(plan.read_text())
        value["groups"]["MzLS"]["jobs"][0]["rows"] += 1
        plan.write_text(json.dumps(value))
    result = run_driver(tmp_path, "run", "--plan", run / "plan.json",
                        "--group", "MzLS", "--index", "1", "--dry-run")
    assert result.returncode != 0
    assert not list(run.rglob("*.hdf5"))


def test_source_drift_is_rejected_without_modifying_the_working_repository(tmp_path):
    repo = tmp_path / "isolated repo"
    shutil.copytree(ROOT / "src", repo / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    inputs = catalogue_pair(tmp_path)
    run, _ = prepare(tmp_path, inputs, None, "--repo", repo)
    source = repo / "src/cess/engine.py"
    source.write_text(source.read_text() + "\n# changed after preparation\n")
    result = run_driver(tmp_path, "run", "--plan", run / "plan.json",
                        "--group", "MzLS", "--index", "1", "--dry-run")
    assert result.returncode != 0
    assert not list(run.rglob("*.hdf5"))


def test_duplicate_task_cannot_bypass_an_existing_os_lock(tmp_path):
    run, _ = prepare(tmp_path, catalogue_pair(tmp_path))
    lock = run / "status/MzLS/0001.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+") as stream:
        inode = lock.stat().st_ino
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run_driver(tmp_path, "run", "--plan", run / "plan.json",
                            "--group", "MzLS", env=allocated_environment(tmp_path))
        assert result.returncode != 0
        assert "already running" in result.stderr.lower()
        assert lock.stat().st_ino == inode
    assert not list(run.rglob("*.hdf5"))
    assert not (run / "status/MzLS/0001.json").exists()


def test_local_allocated_payload_publishes_once_and_resume_preserves_bytes(tmp_path):
    run, _ = prepare(tmp_path, catalogue_pair(tmp_path))
    command = ("run", "--plan", run / "plan.json", "--group", "MzLS")
    first = run_driver(tmp_path, *command, env=allocated_environment(tmp_path),
                       driver=run / "catalogue_array.py")
    assert first.returncode == 0, first.stdout + first.stderr
    products = list(run.rglob("*.hdf5"))
    assert len(products) == 1
    product = products[0]
    with h5py.File(product) as handle:
        assert handle.attrs["schema_version"] == "3.0"
        assert len(handle["catalog/ID"]) == 3
        assert handle["catalog/data_mask"][:].tolist() == [2, 1, 1]
    before = product.read_bytes(), product.stat().st_mtime_ns
    second = run_driver(tmp_path, *command, env=allocated_environment(tmp_path),
                        driver=run / "catalogue_array.py")
    assert second.returncode == 0, second.stdout + second.stderr
    assert (product.read_bytes(), product.stat().st_mtime_ns) == before
    assert list(run.rglob("*.hdf5")) == products
    files_before = {path.relative_to(run): (digest(path), path.stat().st_mtime_ns)
                    for path in run.rglob("*") if path.is_file()}
    preview = run_driver(tmp_path, *command, "--index", "1", "--dry-run")
    assert preview.returncode == 0, preview.stdout + preview.stderr
    assert "\"pending_jobs\": 0" in preview.stdout
    assert {path.relative_to(run): (digest(path), path.stat().st_mtime_ns)
            for path in run.rglob("*") if path.is_file()} == files_before
    status = run_driver(tmp_path, "status", "--plan", run / "plan.json")
    assert status.returncode == 0, status.stdout + status.stderr
    summary = json.loads(status.stdout)
    assert "DECaLS" in json.dumps(summary)
    assert "qsub" in status.stdout and "-t" in status.stdout


def test_termination_records_interruption_and_stops_the_payload(tmp_path):
    inputs = catalogue_pair(tmp_path)
    make_input(inputs / "seedcat2_0001_MzLS.hdf5", count=128)
    run, _ = prepare(tmp_path, inputs)
    process = subprocess.Popen(
        [sys.executable, "-s", "-B", str(run / "catalogue_array.py"), "run",
         "--plan", str(run / "plan.json"), "--group", "MzLS"],
        cwd=tmp_path, env=allocated_environment(tmp_path),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 45
        started = False
        while time.monotonic() < deadline and process.poll() is None:
            logs = list((run / "logs/MzLS").glob("*.log"))
            if logs and "cess_job_started" in logs[0].read_text():
                started = True
                break
            time.sleep(0.02)
        assert started, "Scientific worker did not start before the cancellation check"
        # PBS cancellation relies on the worker retaining the controller's
        # session. A new process group is useful; a new session would escape it.
        worker = None
        for line in logs[0].read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("event") == "cess_job_started":
                worker = event["worker_pid"]
                break
        assert worker is not None
        try:
            worker_session = os.getsid(worker)
        except ProcessLookupError:
            pass
        else:
            assert worker_session == os.getsid(process.pid)
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 143, stdout + stderr
        state = json.loads((run / "status/MzLS/0001.json").read_text())
        assert state["state"] == "interrupted" and state["returncode"] == 143
        assert not list(run.rglob("*.hdf5"))
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=5)
