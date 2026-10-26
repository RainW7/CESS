"""Verify native thread setup before spawn, restoration and explicit planning."""
from concurrent.futures import Future, ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import time

import pytest

from cess.cli.main import main, parser
from cess.cli.parallel import THREAD_VARIABLES, worker_thread_environment
from test_pipeline_smoke import make_input


def _environment_probe(folder):
    # Waiting for the other distinct worker makes this an actual spawn probe,
    # rather than two short tasks that one process could execute in sequence.
    folder = Path(folder)
    (folder / str(os.getpid())).touch()
    end = time.monotonic() + 30
    while len(list(folder.iterdir())) < 2:
        if time.monotonic() > end:
            raise RuntimeError('Second worker did not start')
        time.sleep(.01)
    return os.getpid(), {name: os.environ.get(name) for name in THREAD_VARIABLES}


def test_spawned_workers_inherit_limit_and_parent_environment_is_restored(tmp_path, monkeypatch):
    for name in THREAD_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('OMP_NUM_THREADS', '7')
    before = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    with worker_thread_environment(2):
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(_environment_probe, str(tmp_path)) for _ in range(2)]
            results = [future.result(timeout=60) for future in futures]
    assert len({pid for pid, settings in results}) == 2
    assert all(settings == dict.fromkeys(THREAD_VARIABLES, '2') for pid, settings in results)
    assert {name: os.environ.get(name) for name in THREAD_VARIABLES} == before


@pytest.mark.parametrize('value', [0, -1, True, 1.5, '2'])
def test_thread_context_rejects_invalid_limits_without_changing_environment(value):
    before = dict(os.environ)
    with pytest.raises(ValueError, match='positive integer'):
        with worker_thread_environment(value):
            pass
    assert dict(os.environ) == before


def test_pool_creation_and_submission_observe_requested_environment_even_on_failure(monkeypatch, capsys):
    import cess.cli.main as cli
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '9')
    before = {name: os.environ.get(name) for name in THREAD_VARIABLES}

    class FailedPool:
        def __init__(self, **kwargs):
            assert kwargs['max_workers'] == 1
            assert all(os.environ[name] == '3' for name in THREAD_VARIABLES)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            assert all(os.environ[name] == '3' for name in THREAD_VARIABLES)

        def submit(self, function, job):
            assert all(os.environ[name] == '3' for name in THREAD_VARIABLES)
            result = Future()
            result.set_exception(RuntimeError('fixture failure'))
            return result

    monkeypatch.setattr(cli, 'ProcessPoolExecutor', FailedPool)
    job = dict(filename='one.hdf5', source_start=0, source_stop=2)
    assert cli._run_jobs([job], {'results': []}, workers=1, threads_per_worker=3) == 1
    assert 'fixture failure' in capsys.readouterr().out
    assert {name: os.environ.get(name) for name in THREAD_VARIABLES} == before


@pytest.mark.parametrize('command', ['simulate', 'el-detect'])
def test_thread_limit_is_explicit_positive_and_defaults_to_one(command):
    assert parser().parse_args([command]).threads_per_worker == 1
    assert parser().parse_args([command, '--threads-per-worker', '3']).threads_per_worker == 3
    with pytest.raises(SystemExit):
        parser().parse_args([command, '--threads-per-worker', '0'])


def test_directory_preview_reports_file_tasks_and_limits_without_mutation(tmp_path, capsys):
    inputs = tmp_path / 'input'
    inputs.mkdir()
    for name in ('first.hdf5', 'second.hdf5'):
        make_input(inputs/name, count=2)
    environment = dict(os.environ)
    assert main(['simulate', '-i', str(inputs), '-o', str(tmp_path/'out'),
                 '-j', '8', '--threads-per-worker', '2', '--dry-run']) == 0
    captured = capsys.readouterr()
    plan = json.loads(captured.out)
    assert captured.err == ''
    assert plan['requested_workers'] == 8 and plan['workers'] == 2
    assert plan['threads_per_worker'] == 2
    assert plan['input_files'] == plan['task_count'] == 2
    assert all(job['selected_rows'] == 2 for job in plan['jobs'])
    assert not (tmp_path/'out').exists() and dict(os.environ) == environment
