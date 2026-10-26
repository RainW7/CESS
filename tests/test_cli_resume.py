"""Small manifest/resume integration plus deterministic scheduler checkpoints.

Only two CLI subprocesses execute science jobs (two one-row chunks, then one
missing chunk). Other cases invoke read-only planning or an in-process fake
executor; no catalogue inputs or pre-existing products are touched.
"""
from concurrent.futures import Future
import hashlib
import json
from pathlib import Path
import sys
import h5py
import pytest
from test_cli_repair import call_cli, fixture_files

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / 'src'
sys.path.insert(0, str(CORE))
import cess.cli.main as cess_cli
def signature(path):
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns


def arguments(config, manifest):
    return ['simulate', '--config', str(config), '--seed', '1214',
            '--chunk-size', '1', '--manifest', str(manifest), '--output-layout', 'legacy']


def test_real_resume_missing_piece_orphan_and_unchanged_complete_outputs(tmp_path, capsys):
    _, config, _ = fixture_files(tmp_path, count=2)
    manifest = tmp_path / 'checkpoint.json'
    base = arguments(config, manifest)
    first = call_cli(tmp_path, base)
    assert first.returncode == 0, first.stdout
    saved = json.loads(manifest.read_text())
    assert saved['manifest_version'] == 1 and 'snapshot' in saved
    assert len(saved['results']) == 2
    assert all(result['status'] == 'complete' for result in saved['results'])
    by_row = {result['source_start']: result for result in saved['results']}
    intact, missing = [Path(by_row[index]['output']) for index in (0, 1)]
    intact_before = signature(intact)
    missing.unlink()

    # Preview identifies the missing product without changing the checkpoint or
    # filesystem. It need not create/acquire a lock sidecar for a read-only run.
    manifest_before = signature(manifest)
    directory_before = sorted(str(path) for path in tmp_path.rglob('*'))
    assert cess_cli.main(base + ['--resume', '--dry-run']) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview['pending_jobs'] == 1
    assert signature(manifest) == manifest_before
    assert sorted(str(path) for path in tmp_path.rglob('*')) == directory_before
    assert not missing.exists()

    # Scheduling settings may change without changing scientific identity.
    resumed = call_cli(tmp_path, base + ['--resume', '--workers', '2',
                                       '--memory-budget-gib', '1'])
    assert resumed.returncode == 0, resumed.stdout
    assert missing.exists() and signature(intact) == intact_before
    saved = json.loads(manifest.read_text())
    assert all(result['status'] == 'complete' for result in saved['results'])
    for result in saved['results']:
        path = Path(result['output'])
        assert result['sha256'] == signature(path)[0]
        with h5py.File(str(path), 'r') as handle:
            assert handle['input_row_indices'][:].tolist() == [result['source_start']]

    # Simulate publication immediately before the parent lost its result update.
    # The existing atomically published product is verified and recovered.
    saved['results'][0] = {key: saved['results'][0][key] for key in
                          ('input', 'source_start', 'source_stop')}
    saved['results'][0]['status'] = 'pending'
    manifest.write_text(json.dumps(saved))
    products_before = {path: signature(path) for path in (intact, missing)}
    assert cess_cli.main(base + ['--resume']) == 0
    capsys.readouterr()
    recovered = json.loads(manifest.read_text())
    assert recovered['pending_jobs'] == 0
    assert recovered['results'][0]['recovered_orphan']
    assert {path: signature(path) for path in products_before} == products_before

    # A second new manifest must never adopt the already existing outputs.
    fresh = tmp_path / 'another.json'
    assert cess_cli.main(arguments(config, fresh)) == 2
    assert 'never adopts existing' in capsys.readouterr().err
    assert not fresh.exists()

    # Existing damage is a hard error and is not overwritten on resume.
    missing.write_bytes(b'damaged output')
    damaged = signature(missing)
    assert cess_cli.main(base + ['--resume']) == 2
    assert 'CESS:' in capsys.readouterr().err
    assert signature(missing) == damaged


def test_resume_requires_manifest_and_rejects_legacy_snapshot(tmp_path, capsys):
    _, config, _ = fixture_files(tmp_path, count=1)
    assert cess_cli.main(['simulate', '--config', str(config), '--resume', '--dry-run']) == 2
    assert '--resume requires --manifest' in capsys.readouterr().err
    manifest = tmp_path / 'old.json'
    manifest.write_text(json.dumps({'jobs': [], 'results': []}))
    original = signature(manifest)
    assert cess_cli.main(arguments(config, manifest) + ['--resume', '--dry-run']) == 2
    assert 'legacy/unversioned' in capsys.readouterr().err
    assert signature(manifest) == original
    assert not (tmp_path / 'output').exists()


def test_resume_rejects_input_config_and_interval_or_destination_drift(tmp_path, capsys):
    source, config_path, config = fixture_files(tmp_path, count=2)
    manifest = tmp_path / 'checkpoint.json'
    base = arguments(config_path, manifest)
    # A new dry-run manifest is a versioned pending plan, without products.
    assert cess_cli.main(base + ['--dry-run']) == 0
    capsys.readouterr()
    checkpoint = signature(manifest)
    changed = dict(config, expt=config['expt'] + 1)
    config_path.write_text(json.dumps(changed))
    assert cess_cli.main(base + ['--resume', '--dry-run']) == 2
    assert 'snapshot mismatch' in capsys.readouterr().err
    config_path.write_text(json.dumps(config))
    assert cess_cli.main(base + ['--resume', '--dry-run', '--source-stop', '1']) == 2
    assert 'job plan mismatch' in capsys.readouterr().err
    another = tmp_path / 'not-created' / 'output'
    assert cess_cli.main(base + ['--resume', '--dry-run', '--output', str(another)]) == 2
    assert 'job plan mismatch' in capsys.readouterr().err
    assert not another.parent.exists()
    with h5py.File(str(source), 'r+') as handle:
        handle['best_fit/spec_synthetic'][0, 0] += 1.
    assert cess_cli.main(base + ['--resume', '--dry-run']) == 2
    assert 'snapshot mismatch' in capsys.readouterr().err
    assert signature(manifest) == checkpoint


def test_manifest_persists_each_success_and_failure_before_next_job(tmp_path, monkeypatch, capsys):
    import cess.io.manifest as cess_manifest
    _, config, _ = fixture_files(tmp_path, count=2)
    manifest = tmp_path / 'checkpoint.json'
    scheduled = {}
    observed = []

    class DeferredPool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, job):
            future = Future()
            scheduled[future] = job
            return future

    def complete_one(futures, return_when):
        future = next(iter(futures))
        job = scheduled[future]
        checkpoint = json.loads(manifest.read_text())
        states = [entry['status'] for entry in checkpoint['results']]
        observed.append(states)
        if job['source_start'] == 0:
            assert states == ['pending', 'pending']
            destination = cess_manifest.expected_product(job)
            destination.write_bytes(b'fake product; scheduler persistence unit test')
            future.set_result(str(destination))
        else:
            assert states == ['complete', 'pending']
            future.set_exception(RuntimeError('intentional worker failure'))
        return {future}, set(futures) - {future}

    def verified(job, snapshot, expected_sha256=None):
        path = cess_manifest.expected_product(job)
        if not path.exists():
            return None
        return dict(output=str(path), sha256=signature(path)[0])

    monkeypatch.setattr(cess_cli, 'ProcessPoolExecutor', DeferredPool)
    monkeypatch.setattr(cess_cli, 'wait', complete_one)
    monkeypatch.setattr(cess_manifest, 'verify_product', verified)
    assert cess_cli.main(arguments(config, manifest)) == 1
    capsys.readouterr()
    saved = json.loads(manifest.read_text())
    assert observed == [['pending', 'pending'], ['complete', 'pending']]
    assert [row['status'] for row in saved['results']] == ['complete', 'failed']
    assert 'intentional worker failure' in saved['results'][1]['error']
