"""Local Bash contract tests for whole-file multiworker PBS payloads.

Executing cess.pbs directly exercises the payload, not a server scheduler or
qsub. All inputs, outputs, logs and manifests live in pytest's temporary tree.
"""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import h5py
import pytest

from cess.config.loader import load_config
from test_pipeline_smoke import make_input


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT/'scripts/gravity/preflight.sh'
PBS = ROOT/'scripts/gravity/cess.pbs'


def setup_run(tmp_path, **changes):
    inputs = tmp_path/'input catalogues'
    inputs.mkdir()
    make_input(inputs/'a catalogue.hdf5', count=3)
    make_input(inputs/'b.h5', count=2)
    config = tmp_path/'runtime.json'
    config.write_text(json.dumps(load_config()))
    environment = {key: value for key, value in os.environ.items() if not key.startswith('CESS_')}
    environment.update(
        CESS_PYTHON=sys.executable, CESS_REPO=str(ROOT), CESS_CONFIG=str(config),
        CESS_INPUT=str(inputs), CESS_OUTPUT=str(tmp_path/'results'),
        CESS_MANIFEST=str(tmp_path/'checkpoints/run.json'), CESS_CPUS='2', CESS_WORKERS='2',
        CESS_EXPECTED_VERSION=runpy.run_path(str(ROOT/'src/cess/_version.py'))['__version__'],
        MPLCONFIGDIR=str(tmp_path/'mpl'), PYTHONDONTWRITEBYTECODE='1',
        OPENBLAS_NUM_THREADS='19', OMP_NUM_THREADS='17')
    environment.update(changes)
    return environment


def run_script(script, environment, cwd, *arguments):
    return subprocess.run(['/bin/bash', str(script), *arguments], cwd=cwd, env=environment,
        text=True, capture_output=True, timeout=120)


def preview_json(result):
    assert result.returncode == 0, result.stdout+result.stderr
    check, preview = result.stdout.split('Preview command:', 1)
    return json.loads(check), json.loads(preview.split('\n', 1)[1])


def test_directory_preview_keeps_whole_file_tasks_and_is_read_only(tmp_path):
    environment = setup_run(tmp_path)
    check, plan = preview_json(run_script(PREFLIGHT, environment, tmp_path, '--dry-run'))
    assert check['input_kind'] == 'directory' and check['chunk_size'] is None
    assert check['workers'] == 2 and check['threads_per_worker'] == 1
    assert plan['workers'] == 2
    assert [(job['filename'], job['source_start'], job['source_stop']) for job in plan['jobs']] == [
        ('a catalogue.hdf5', 0, 3), ('b.h5', 0, 2)]
    assert not Path(environment['CESS_OUTPUT']).exists()
    assert not Path(environment['CESS_MANIFEST']).exists()
    assert not Path(environment['CESS_MANIFEST']).parent.exists()


def test_explicit_chunks_and_memory_budget_are_forwarded(tmp_path):
    environment = setup_run(tmp_path, CESS_CHUNK_SIZE='2', CESS_MEMORY_BUDGET_GIB='1')
    result = run_script(PREFLIGHT, environment, tmp_path, '--dry-run')
    check, plan = preview_json(result)
    assert check['chunk_size'] == 2 and check['memory_budget_gib'] == 1
    assert len(plan['jobs']) == 3
    assert [job['selected_rows'] for job in plan['jobs']] == [2, 1, 2]
    assert '--chunk-size 2' in result.stdout and '--memory-budget-gib 1' in result.stdout


@pytest.mark.parametrize('chunk_size', ['0', '-1', '1.5', 'two'])
def test_bad_explicit_chunk_size_is_rejected_without_outputs(tmp_path, chunk_size):
    environment = setup_run(tmp_path, CESS_CHUNK_SIZE=chunk_size)
    result = run_script(PREFLIGHT, environment, tmp_path, '--dry-run')
    assert result.returncode != 0
    assert 'CESS_CHUNK_SIZE must be empty' in result.stderr
    assert not Path(environment['CESS_OUTPUT']).exists()
    assert not Path(environment['CESS_MANIFEST']).exists()


@pytest.mark.parametrize('budget', ['0', '-2', 'nan', 'inf', 'two'])
def test_bad_optional_memory_budget_is_rejected_without_outputs(tmp_path, budget):
    environment = setup_run(tmp_path, CESS_MEMORY_BUDGET_GIB=budget)
    result = run_script(PREFLIGHT, environment, tmp_path, '--dry-run')
    assert result.returncode != 0
    assert 'CESS_MEMORY_BUDGET_GIB must be positive and finite' in result.stderr
    assert not Path(environment['CESS_OUTPUT']).exists()
    assert not Path(environment['CESS_MANIFEST']).exists()


def test_workers_cannot_exceed_declared_scheduler_cpus(tmp_path):
    environment = setup_run(tmp_path, CESS_CPUS='1')
    result = run_script(PREFLIGHT, environment, tmp_path, '--dry-run')
    assert result.returncode != 0 and 'Workers exceed CESS_CPUS' in result.stderr


def test_local_pbs_payload_runs_two_catalogues_without_splitting_outputs(tmp_path):
    environment = setup_run(tmp_path)
    result = run_script(PBS, environment, tmp_path)
    assert result.returncode == 0, result.stdout+result.stderr
    manifest = json.loads(Path(environment['CESS_MANIFEST']).read_text())
    assert manifest['workers'] == 2 and len(manifest['jobs']) == 2
    assert all(item['status'] == 'complete' for item in manifest['results'])
    products = sorted(Path(environment['CESS_OUTPUT']).glob('*.hdf5'))
    assert len(products) == 2
    for path, count in zip(products, (3, 2)):
        assert '_rows' not in path.name
        with h5py.File(path, 'r') as handle:
            assert handle.attrs['schema_version'] == '3.0'
            assert len(handle['catalog/ID']) == count
    logs = list((Path(environment['CESS_MANIFEST']).parent/'logs').glob('cess.*.log.*'))
    assert len(logs) == 1
    assert 'CESS exit=0' in logs[0].read_text()
    manifest_path = Path(environment['CESS_MANIFEST'])
    before_manifest = manifest_path.read_bytes(), manifest_path.stat().st_mtime_ns
    before_products = [(path.read_bytes(), path.stat().st_mtime_ns) for path in products]
    environment['CESS_RESUME'] = '1'
    _, resumed = preview_json(run_script(PREFLIGHT, environment, tmp_path, '--dry-run'))
    assert resumed['resumed'] is True and resumed['pending_jobs'] == 0
    assert (manifest_path.read_bytes(), manifest_path.stat().st_mtime_ns) == before_manifest
    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in products] == before_products
