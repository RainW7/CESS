"""Bounded Gaussian statistics against the byte-bound archived R3 functions."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cess.morphology import libraries

BASELINE = ROOT / 'tests/baselines'
SNAPSHOT_SHA = 'ad354eb59b56dbb2bb0360ceb7b131178e02a4d12c75ae8b2870e09e688f20d6'
SOURCE_ARCHIVE_SHA = 'b93953b7720265359f8348cd84ef3d112662596a3b2ce52984a235e89127640d'
SOURCE_SHA = 'f2e9102b0cda052bfaa78716159cabe73dbf0755dd0cf2f811a7b66905373ef5'


@pytest.fixture(scope='module')
def oracle():
    archive = BASELINE / 'profile_statistics_r3.tar.gz'
    baseline = json.loads((BASELINE / 'PROVENANCE.json').read_text())['profile_statistics_r3']
    name = 'src/cess/morphology/libraries.py'
    assert baseline['source_archive'] == 'docs/refactor_v090_r4/baseline_sources.tar.gz'
    assert baseline['source_archive_sha256'] == SOURCE_ARCHIVE_SHA
    assert baseline['source_archive_members'] == 209
    assert baseline['snapshot'] == archive.name
    assert baseline['snapshot_sha256'] == SNAPSHOT_SHA
    assert baseline['snapshot_members'] == {name: SOURCE_SHA}
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == SNAPSHOT_SHA
    with tarfile.open(str(archive), 'r:gz') as source:
        members = source.getmembers()
        assert len(members) == 1 and members[0].name == name and members[0].isfile()
        payload = source.extractfile(members[0]).read()
    assert hashlib.sha256(payload).hexdigest() == SOURCE_SHA
    module = types.ModuleType('archived_r3_libraries_statistics')
    module.__file__ = str(archive) + '::' + name
    exec(compile(payload, module.__file__, 'exec'), module.__dict__)
    return module


def library(profiles):
    result = np.empty((len(profiles), 1, 1, 1), dtype=object)
    for index, profile in enumerate(profiles):
        result[index, 0, 0, 0] = profile
    return result


def profiles():
    result = []
    random = np.random.RandomState(7621)
    for length in (1, 2, 4, 5, 6, 9, 32, 101):
        x = np.arange(length, dtype=np.float64) - (length - 1.) / 2.
        for dtype in (np.float32, np.float64, np.int64, np.uint16):
            gaussian = np.exp(-.5 * (x / 2.)**2)
            value = gaussian if np.issubdtype(dtype, np.floating) else np.ceil(gaussian * 100)
            result.append(value.astype(dtype))
        result += [np.ones(length), np.exp(.1 * x), random.uniform(.01, 2., length)]
        if length >= 5:
            result.append(np.concatenate(([0.], np.ones(length-2), [0.])))
            for delta in (0., 1.e-5*(1.-2.e-12), 1.e-5*(1.+2.e-12), .01):
                value = np.exp(-.5*(x/3.)**2)
                value[0] *= 1. + delta
                result.append(value)
            for scale in (1.e-300, 1.e-100, 1.e100, 1.e300):
                result.append(np.exp(-.5*(x/5.)**2) * scale)
    result += [np.full(7, np.nextafter(0., 1.)), np.array([1., 1., 1., 1., 1.]),
               np.array([1., 1., 1.+1.e-9, 1., 1.])]
    return result


def test_batch_decisions_and_complete_json_match_archive(oracle):
    values = profiles()
    original = [np.array(value, copy=True) for value in values]
    for length in sorted({len(value) for value in values}):
        equal_length = [np.asarray(value, dtype=np.float64) for value in values if len(value) == length]
        expected = [oracle._gaussian_like(value) for value in equal_length]
        np.testing.assert_array_equal(libraries._gaussian_like_batch(equal_length), expected)
    old = oracle._profile_statistics(library(values), 'width')
    new = libraries._profile_statistics(library(values), 'width')
    assert json.dumps(new, sort_keys=True, allow_nan=False) == json.dumps(old, sort_keys=True, allow_nan=False)
    for value, before in zip(values, original):
        np.testing.assert_array_equal(value, before)
        assert value.dtype == before.dtype


@pytest.mark.parametrize('bad', [
    np.ma.array([1., 2.], mask=[False, True]), np.array([], dtype=float),
    np.ones((2, 2)), np.array([1.+0j]), np.array([True]), np.array(['1']),
    np.array([1., np.nan]), np.array([np.inf]), np.array([-.1, 1.]),
    np.zeros(5), np.array([1.e308, 1.e308]), np.array([1.], dtype=object),
])
def test_first_invalid_error_is_exactly_the_archived_error(oracle, bad):
    valid = np.exp(-.5*np.arange(-3., 4.)**2)
    values = library([valid, valid.astype('float32'), bad, np.array([np.nan])])
    errors = []
    for function in (oracle._profile_statistics, libraries._profile_statistics):
        with pytest.raises(ValueError) as problem:
            function(values, 'height')
        errors.append(str(problem.value))
    assert errors[0] == errors[1]
    assert 'height profile (2, 0, 0, 0)' in errors[1]


def test_batches_are_bounded_for_uniform_and_ragged_lengths(oracle, monkeypatch):
    calls = []
    actual = libraries._gaussian_like_batch
    def checked(rows):
        sizes = [len(row) for row in rows]
        assert len(rows) <= libraries._GAUSSIAN_BATCH_SIZE
        assert sum(sizes) <= libraries._GAUSSIAN_BATCH_ELEMENTS
        assert len(set(sizes)) == 1
        calls.append((len(rows), sizes[0]))
        return actual(rows)
    monkeypatch.setattr(libraries, '_gaussian_like_batch', checked)
    values = []
    for index in range(1031):
        length = 9 if index < 700 else 5 + index % 23
        values.append(np.exp(-.5 * ((np.arange(length) - (length-1)/2.) / 3.)**2))
    assert libraries._profile_statistics(library(values), 'width') == oracle._profile_statistics(library(values), 'width')
    assert calls and sum(count for count, _ in calls) == len(values)
    assert max(count for count, _ in calls) == libraries._GAUSSIAN_BATCH_SIZE


def test_very_long_profile_and_strict_error_policy_keep_scalar_oracle(oracle, monkeypatch):
    values = library([np.ones(libraries._GAUSSIAN_BATCH_ELEMENTS + 1)])
    def forbidden(rows):
        pytest.fail('Oversize/strict-policy profile must not allocate a batch matrix')
    monkeypatch.setattr(libraries, '_gaussian_like_batch', forbidden)
    assert libraries._profile_statistics(values, 'width') == oracle._profile_statistics(values, 'width')
    with np.errstate(all='raise'):
        values = library([np.exp(-.5*np.arange(-3., 4.)**2)])
        assert libraries._profile_statistics(values, 'width') == oracle._profile_statistics(values, 'width')
