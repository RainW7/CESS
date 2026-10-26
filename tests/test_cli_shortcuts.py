"""The public shortcuts retain the existing simulation plan and defaults."""
import json

import pytest

from cess.cli.main import main, parser
from test_pipeline_smoke import make_input


@pytest.mark.parametrize('command', ['simulate', 'el-detect'])
def test_short_arguments_have_identical_destinations(command):
    short = parser().parse_args([command, '-i', 'a.hdf5', '-o', 'out', '-j', '2'])
    long = parser().parse_args([command, '--input', 'a.hdf5', '--output', 'out', '--workers', '2'])
    assert vars(short) == vars(long)


def test_all_effects_matches_explicit_flags_and_preserves_old_defaults(tmp_path, capsys):
    source = tmp_path / 'toy.hdf5'
    make_input(source, count=3)
    common = ['simulate', '-i', str(source), '-o', str(tmp_path/'out'), '--dry-run',
              '--chunk-size', '2', '-j', '2']
    def plan(flags):
        assert main(common+flags) == 0
        return json.loads(capsys.readouterr().out)
    shortcuts = plan(['--all-effects'])
    assert shortcuts == plan(['--morphology','--wavelength-error','--photometry','--emission-lines'])
    assert all(shortcuts['options'].values())
    assert not any(plan([])['options'].values())
    assert not (tmp_path/'out').exists()


def test_default_is_one_product_per_input_without_implicit_chunking(tmp_path, capsys):
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    for filename, count in [('a.hdf5', 7), ('b.hdf5', 5)]:
        make_input(inputs / filename, count=count)
    assert main(['simulate', '-i', str(inputs), '-o', str(tmp_path/'out'),
                 '--all-effects', '-j', '8', '--dry-run']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['workers'] == 2
    assert [(j['source_start'], j['source_stop']) for j in plan['jobs']] == [(0, 7), (0, 5)]
    assert not (tmp_path/'out').exists()


def test_merge_cli_dispatches_without_loading_simulation_config(tmp_path, monkeypatch, capsys):
    import cess.config.loader as loader
    import cess.io.merge as merger
    source = tmp_path / 'chunk.hdf5'
    source.touch()
    destination = tmp_path / 'complete.hdf5'
    original = tmp_path / 'input.hdf5'
    calls = []

    def fail_config(*args, **kwargs):
        raise AssertionError('Merging must not load instrument configuration')

    def merge(sources, target, **kwargs):
        calls.append((sources, target, kwargs))
        return {'sources': 1, 'rows': 3, 'covers_entire_input': True} if kwargs['dry_run'] else target

    monkeypatch.setattr(loader, 'load_config', fail_config)
    monkeypatch.setattr(merger, 'merge_products', merge)
    base = ['merge', '-i', str(source), '-o', str(destination),
            '--input-source', str(original), '--copy-rows', '7']
    assert main(base + ['--dry-run']) == 0
    assert json.loads(capsys.readouterr().out) == {'sources': 1, 'rows': 3, 'covers_entire_input': True}
    assert calls[-1] == ([source.resolve()], destination,
                         {'expected_input': original, 'chunk_rows': 7, 'dry_run': True})
    assert main(base) == 0
    assert json.loads(capsys.readouterr().out) == {'status': 'complete', 'output': str(destination)}


def test_merge_cli_rejects_partial_when_complete_input_check_requested(tmp_path, monkeypatch, capsys):
    import cess.io.merge as merger
    source = tmp_path / 'chunk.hdf5'
    source.touch()
    calls = []

    def merge(*args, **kwargs):
        calls.append(kwargs)
        return {'covers_entire_input': False}

    monkeypatch.setattr(merger, 'merge_products', merge)
    assert main(['merge', '-i', str(source), '-o', str(tmp_path/'complete.hdf5'),
                 '--input-source', str(tmp_path/'input.hdf5')]) == 2
    assert 'every row' in capsys.readouterr().err
    assert calls == [{'expected_input': tmp_path/'input.hdf5', 'chunk_rows': 128, 'dry_run': True}]
