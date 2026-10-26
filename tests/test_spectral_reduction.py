"""Local spectral optimizations retain CDF bits and integral accuracy."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import cess.spectra.integration as spectral
def old_probability(low, high):
    return np.where(low >= 0, ndtr(-low) - ndtr(-high), ndtr(high) - ndtr(low))


def old_add(out, index, weights):
    np.add.at(out, index, weights)


def test_probability_is_bitwise_equal_including_tails_and_broadcast():
    rng = np.random.RandomState(2214)
    low = np.r_[rng.uniform(-50., 50., 20000), -np.inf, -40., -8., -0., 0., 8., 40.]
    with np.errstate(invalid='ignore'):
        high = low + np.r_[10. ** rng.uniform(-14., 2., 20000), np.inf, 1., 1., 0., 1., 1., np.inf]
    # The indeterminate endpoint above is deliberately retained: NaN payloads
    # must match as well, without replacing either implementation's values.
    with np.errstate(invalid='ignore'):
        expected = old_probability(low, high)
        actual = spectral._probability(low, high)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    low, high = np.array([-9., -1., 0., 8.])[:, None], np.array([9., 10.])[None, :]
    np.testing.assert_array_equal(spectral._probability(low, high).view(np.uint64),
                                  old_probability(low, high).view(np.uint64))


@pytest.mark.parametrize('kind', ['empty', 'positive', 'signed', 'high_dynamic'])
def test_binned_reduction_matches_add_at(kind):
    rng = np.random.RandomState(3154)
    index = rng.randint(0, 51, 3000)
    weights = rng.uniform(.1, 1., len(index))
    if kind == 'signed':
        weights *= rng.choice([-1., 1.], len(index))
    elif kind == 'high_dynamic':
        weights *= 10. ** rng.uniform(-20., 12., len(index))
    elif kind == 'empty':
        index, weights = index[:0], weights[:0]
    original = np.linspace(.1, .2, 51)
    expected, actual = original.copy(), original.copy()
    # Multiple blocks exercise adding to an already nonzero output.
    for start in range(0, max(1, len(index)), 137):
        selected = slice(start, start + 137)
        old_add(expected, index[selected], weights[selected])
        spectral._add_to_bins(actual, index[selected], weights[selected])
    np.testing.assert_allclose(actual, expected, rtol=2.e-14, atol=0.)


def test_reduction_does_not_erase_weak_separate_bins_after_strong_line():
    index = np.array([0, 0, 1, 1, 2, 2])
    weights = np.array([1.e12, 1.e12, 1.e-20, 2.e-20, -1.e-20, -2.e-20])
    out = np.zeros(3)
    spectral._add_to_bins(out, index, weights)
    expected = np.zeros(3)
    old_add(expected, index, weights)
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_allclose(out[1:], [3.e-20, -3.e-20], rtol=3.e-16, atol=0.)


@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('signed', [False, True])
def test_convolution_reduction_retains_original_integral_accuracy(monkeypatch, weighted, signed):
    wave = np.linspace(3600., 4400., 1601)
    fnu = (1.e-9 + 1.e8 * np.exp(-.5 * ((wave - 3990.4) / 1.3) ** 2)
           + .03 * np.exp(-.5 * ((wave - 4070.2) / 4.1) ** 2))
    if signed:
        fnu *= np.sin((wave - 3990.) / 7.)
    centers = np.arange(3800., 4201., 4.)
    kwargs = (dict(response_wave=[3600., 3988., 3992., 4400.], response=[.1, .8, .2, .7])
              if weighted else {})
    actual, diagnostics = spectral.convolve_to_bins(wave, fnu, centers, 5., **kwargs)
    with monkeypatch.context() as reference:
        reference.setattr(spectral, '_probability', old_probability)
        reference.setattr(spectral, '_add_to_bins', old_add)
        expected, old_diagnostics = spectral.convolve_to_bins(wave, fnu, centers, 5., **kwargs)
    # Signed functions may cancel near zero; use an explicit peak-scaled
    # roundoff tolerance there, rather than claiming meaningful relative error.
    absolute = np.max(np.abs(expected)) * 2.e-15 if signed else 0.
    np.testing.assert_allclose(actual, expected, rtol=1.e-12, atol=absolute)
    assert diagnostics['output_photon_integral'] == pytest.approx(
        old_diagnostics['output_photon_integral'], rel=1.e-12)
    if not signed:
        assert diagnostics['output_photon_integral'] == pytest.approx(
            diagnostics['input_photon_integral'], rel=5.e-12)
