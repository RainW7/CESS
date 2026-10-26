import importlib.util
from pathlib import Path
import random
import sys
import numpy as np
import pytest
CORE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(CORE))
from cess.pipeline.random import file_identity, legacy_rng, source_rng, source_seed


def test_source_stream_is_order_and_component_independent():
    def sample(row, component="noise_GI"):
        return source_rng(1214, "abc", "ID-{}".format(row), row, component).normal(size=30)
    baseline = {row: sample(row) for row in range(8)}
    for row in [7, 3, 0, 2, 6]:
        np.testing.assert_array_equal(sample(row), baseline[row])
    assert not np.array_equal(sample(3, "noise_GU"), baseline[3])
    assert source_seed(2, "abc", b"X", 0, "morph") == source_seed(2, "abc", "X", 0, "morph")
    assert source_rng(None, "abc", "X", 0, "morph") is None


def test_legacy_context_restores_globals_and_advances_stream():
    np.random.seed(17)
    random.seed(17)
    numpy_state, python_state = np.random.get_state(), random.getstate()
    rng = np.random.RandomState(93)
    expected = np.random.RandomState(93).normal(size=8)
    with legacy_rng(rng):
        np.testing.assert_array_equal(np.random.normal(size=8), expected)
        random.random()
    for actual, before in zip(np.random.get_state(), numpy_state):
        np.testing.assert_array_equal(actual, before)
    assert random.getstate() == python_state
    assert rng.get_state()[2] != np.random.RandomState(93).get_state()[2]
    with pytest.raises(RuntimeError):
        with legacy_rng(rng):
            np.random.random()
            raise RuntimeError("test restoration")
    for actual, before in zip(np.random.get_state(), numpy_state):
        np.testing.assert_array_equal(actual, before)


def test_content_identity_survives_relocation_and_detects_changes(tmp_path):
    first, moved = tmp_path / "one.hdf5", tmp_path / "two.hdf5"
    first.write_bytes(b"input bytes")
    moved.write_bytes(first.read_bytes())
    assert file_identity(first) == file_identity(moved)
    moved.write_bytes(b"different input bytes")
    assert file_identity(first) != file_identity(moved)


@pytest.mark.parametrize("seed,row", [(-1, 0), (True, 0), (1.5, 0), (1, -1), (1, True)])
def test_invalid_stream_identity(seed, row):
    with pytest.raises(ValueError):
        source_seed(seed, "abc", "ID", row, "noise")
