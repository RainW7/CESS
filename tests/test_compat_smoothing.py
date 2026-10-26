"""Physical and shape contracts for the independent compatibility smoother."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr

from cess.spectra.smoothing import gaussian_sample_smooth


def test_gaussian_has_expected_broadening_and_amplitude():
    x = np.linspace(-12., 12., 1201)
    intrinsic_sigma, smoothing_sigma = 1.2, .7
    y = np.exp(-.5 * (x / intrinsic_sigma) ** 2)
    combined_sigma = np.hypot(intrinsic_sigma, smoothing_sigma)
    expected = intrinsic_sigma / combined_sigma * np.exp(-.5 * (x / combined_sigma) ** 2)
    actual = gaussian_sample_smooth(x, y, smoothing_sigma)
    # The sampled Gaussian is reconstructed linearly, so its O(dx**2)
    # interpolation error remains relative to the continuous analytic curve.
    np.testing.assert_allclose(actual, expected, atol=1.6e-5, rtol=0.)
    coarse = gaussian_sample_smooth(x[::2], y[::2], smoothing_sigma)
    assert np.max(np.abs(actual - expected)) < np.max(np.abs(coarse - expected[::2])) / 3.9


def test_constant_finite_input_has_analytic_edges_without_renormalization():
    x = np.linspace(-10., 10., 101)
    actual = gaussian_sample_smooth(x, np.ones_like(x), 1.)
    expected = ndtr(10. - x) - ndtr(-10. - x)
    np.testing.assert_allclose(actual, expected, atol=3.e-15, rtol=2.e-14)
    assert actual[0] == pytest.approx(.5)
    assert actual[-1] == pytest.approx(.5)
    assert actual[len(x) // 2] == pytest.approx(1.)


def test_irregular_grid_and_variable_width_match_independent_quadrature():
    x = np.array([0., .12, .7, 1.8, 3.])
    y = np.array([1., -2., 3., .5, 1.])
    widths = np.array([.2, .6, 0., 1.1, 1.6])
    expected = []
    for center, sigma, value in zip(x, widths, y):
        if sigma == 0:
            expected.append(value)
        else:
            expected.append(quad(
                lambda position: np.interp(position, x, y) * np.exp(
                    -.5 * ((position - center) / sigma) ** 2) / (np.sqrt(2. * np.pi) * sigma),
                x[0], x[-1], points=x[1:-1], epsabs=1.e-12, epsrel=1.e-12,
            )[0])
    np.testing.assert_allclose(gaussian_sample_smooth(x, y, widths), expected,
                               atol=3.e-14, rtol=3.e-14)


def test_column_linearity_and_xout_interpolation_contract():
    x = np.linspace(1., 7., 51)
    y = np.sin(x)
    samples = np.column_stack((y, -3. * y, y + 2.))
    target = np.array([0., 1., 1.34, 4.56, 7., 8.])
    smoothed = gaussian_sample_smooth(x, samples, .2)
    actual = gaussian_sample_smooth(x, samples, .2, target)
    np.testing.assert_allclose(smoothed[:, 1], -3. * smoothed[:, 0], atol=2.e-15)
    for column in range(samples.shape[1]):
        np.testing.assert_array_equal(
            actual[:, column], np.interp(target, x, smoothed[:, column], left=0., right=0.))
    np.testing.assert_array_equal(actual[[0, -1]], 0.)


def test_zero_width_is_identity_and_zero_extended_interpolation():
    x = np.array([0., .5, 1.2, 3.])
    y = np.array([-1., 2., 5., -2.])
    np.testing.assert_array_equal(gaussian_sample_smooth(x, y, 0.), y)
    target = np.array([-.1, 0., .25, 3., 3.1])
    np.testing.assert_array_equal(gaussian_sample_smooth(x, y, np.zeros_like(x), target),
                                  np.interp(target, x, y, left=0., right=0.))


def test_broad_kernel_retains_small_nonzero_signal_and_zero_columns():
    x = np.array([0., .2, 1.])
    result = gaussian_sample_smooth(x, np.ones_like(x), 1.e9)
    np.testing.assert_allclose(result, 1. / (np.sqrt(2. * np.pi) * 1.e9), rtol=1.e-14)
    assert gaussian_sample_smooth(x, np.empty((3, 0)), .5).shape == (3, 0)
    assert gaussian_sample_smooth(x, np.empty((3, 0)), .5, np.array([.5])).shape == (1, 0)


@pytest.mark.parametrize('x,y,width,target', [
    ([1.], [1.], .1, None),
    ([1., 1.], [1., 2.], .1, None),
    ([2., 1.], [1., 2.], .1, None),
    ([1., np.nan], [1., 2.], .1, None),
    ([[1., 2.]], [1., 2.], .1, None),
    ([1., 2.], [1.], .1, None),
    ([1., 2.], [[1.], [np.inf]], .1, None),
    ([1., 2.], np.zeros((2, 1, 1)), .1, None),
    ([1., 2.], [1., 2.], -.1, None),
    ([1., 2.], [1., 2.], np.nan, None),
    ([1., 2.], [1., 2.], [.1], None),
    ([1., 2.], [1., 2.], [[.1], [.1]], None),
    ([1., 2.], [1., 2.], [.1, np.inf], None),
    ([1., 2.], [1., 2.], .1, [[1.5]]),
    ([1., 2.], [1., 2.], .1, [np.nan]),
])
def test_invalid_coordinates_shapes_widths_and_targets_raise(x, y, width, target):
    with pytest.raises(ValueError):
        gaussian_sample_smooth(x, y, width, target)


@pytest.mark.parametrize('oversample', [0., -1., 2., 1.5, np.nan, np.inf, [1.], '1', True])
def test_unused_oversampling_settings_are_rejected_explicitly(oversample):
    with pytest.raises(ValueError, match='oversample must be 1'):
        gaussian_sample_smooth([0., 1.], [1., 1.], .2, oversample=oversample)


def test_unit_oversampling_matches_default():
    x = np.array([0., .5, 2.])
    y = np.array([1., 4., 2.])
    np.testing.assert_array_equal(gaussian_sample_smooth(x, y, .3, oversample=1.),
                                  gaussian_sample_smooth(x, y, .3))
