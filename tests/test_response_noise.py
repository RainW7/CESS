"""Forward response counts are independent of the nominal flux calibration."""
import sys
from pathlib import Path

import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cess.engine import emulator, h
from cess.config.loader import load_config


def simulator():
    config = load_config()
    config['template_lsf'] = {'mode': 'delta'}
    return emulator(config=config)


def arguments():
    wave = np.linspace(4000., 6000., 101)
    flux = np.full_like(wave, 1.e-27)
    return (0., 0., wave, flux, wave, np.zeros_like(wave),
            np.linspace(.1, .8, len(wave)), np.ones(3), np.array([.2, .6, .2]))


def test_response_electron_rate_units_and_nan_rows():
    wave = np.array([4000., 4100., 4300.])
    folded = np.array([[1.e-29, 2.e-29, 3.e-29], [np.nan] * 3])
    rate = emulator.response_electron_rate(wave, folded)
    np.testing.assert_allclose(rate[0], folded[0] * np.array([100., 150., 200.]) / (h * wave))
    assert np.isnan(rate[1]).all()
    with pytest.raises(ValueError, match='axis'):
        emulator.response_electron_rate(wave, np.ones((3, 2)))


def test_integrated_source_counts_do_not_multiply_nominal_throughput_again():
    model = simulator()
    args = arguments()
    rate = np.linspace(.001, .002, len(args[2]))
    result = model._noise_realization(*args, False, rng=np.random.RandomState(1),
                                    source_rate_e_s_cm2=rate, sky_rate_e_s_pixel=.4)
    expected = rate * model.colarea * model.config['expt'] * model.config['expnum']
    np.testing.assert_allclose(result['model'], expected)
    sky = .4 * model.config['expt'] * model.config['expnum'] * 3
    dark = model.config['dc_value'] * model.config['expt'] * model.config['expnum'] * 3
    read = model.config['read_noise'] ** 2 * model.config['expnum'] * 3
    np.testing.assert_allclose(result['variance'], expected + sky + dark + read)
    changed = list(args)
    changed[6] = args[6] / 2.
    same = model._noise_realization(*changed, False, rng=np.random.RandomState(1),
                                  source_rate_e_s_cm2=rate, sky_rate_e_s_pixel=.4)
    np.testing.assert_array_equal(result['observed'], same['observed'])
    a = model._noise_product(result, 0., 'GV')
    b = model._noise_product(same, 0., 'GV')
    np.testing.assert_allclose(b[1], 2. * a[1])
    np.testing.assert_allclose(b[2], 2. * a[2])


def test_constant_center_response_fallback_matches_explicit_rate():
    model = simulator()
    args = list(arguments())
    args[6] = np.full_like(args[2], .3)
    rate = model.response_electron_rate(args[2], args[3] * args[6])
    old = model._noise_realization(*args, False, rng=np.random.RandomState(2), sky_rate_e_s_pixel=.1)
    new = model._noise_realization(*args, False, rng=np.random.RandomState(2),
                                 source_rate_e_s_cm2=rate, sky_rate_e_s_pixel=.1)
    np.testing.assert_allclose(old['model'], new['model'], rtol=5.e-15)
    np.testing.assert_allclose(old['variance'], new['variance'], rtol=5.e-15)


def test_zero_center_response_keeps_detector_model_but_not_calibrated_flux():
    model = simulator()
    args = list(arguments())
    args[6][50] = 0.
    rate = np.full_like(args[2], .001)
    realization = model._noise_realization(*args, False, rng=np.random.RandomState(1),
                                         source_rate_e_s_cm2=rate, sky_rate_e_s_pixel=0.)
    assert realization['valid'][50]
    product = model._noise_product(realization, 0., 'GV')
    assert product[0][50] > 0.
    assert np.isnan(product[1][50]) and np.isnan(product[2][50]) and np.isnan(product[4][50])
    assert not product[7][50]


@pytest.mark.parametrize('morph', [False, True])
@pytest.mark.parametrize('wce', [False, True])
def test_native_response_standard_and_paired_share_one_noise(monkeypatch, morph, wce):
    model = simulator()
    monkeypatch.setattr(model, '_sample_wave_offset', lambda rng=None: .5)
    args = arguments()
    rate = np.full_like(args[2], .001)
    rate[50] = .2
    standard = model.add_noise(*args, 'GV', morph, wce, rng=np.random.RandomState(31),
                               source_rate_e_s_cm2=rate, sky_rate_e_s_pixel=.1)
    paired = model.add_noise_zouhu(*args, 'GV', morph, wce, rng=np.random.RandomState(31),
                                  source_rate_e_s_cm2=rate, sky_rate_e_s_pixel=.1)
    for left, right in [(standard[0], paired[2]), (standard[1], paired[4]),
                        (standard[2], paired[6]), (standard[4], paired[9])]:
        np.testing.assert_allclose(left, right, equal_nan=True)
    if not wce:
        np.testing.assert_array_equal(paired[1], paired[2])
        np.testing.assert_array_equal(paired[3], paired[4])
    else:
        assert paired[11] == .5


@pytest.mark.parametrize('rate', [np.full(100, .1), np.full(101, -.1), np.full(101, np.inf)])
def test_invalid_native_source_rates_rejected(rate):
    with pytest.raises(ValueError, match='source_rate_e_s_cm2'):
        simulator()._noise_realization(*arguments(), False, source_rate_e_s_cm2=rate,
                                      sky_rate_e_s_pixel=0.)


def test_native_sky_uses_full_response_and_analytic_product():
    model = simulator()
    wave = np.array([2000., 11097.])
    response_wave = np.array([2010., 5000., 11000.])
    a, b, c, d = 1.e-29, 1.e-33, .1, 1.e-5
    sky = a + b * wave
    response = c + d * response_wave
    lo, hi = response_wave[[0, -1]]
    integral = a * c * np.log(hi / lo) + (a * d + b * c) * (hi - lo) + b * d * (hi ** 2 - lo ** 2) / 2.
    expected = integral / h * model.colarea * model.config['arcsecperpix'] ** 2
    measured = model.native_sky_rate(0., 0., wave, sky, response_wave, response)
    assert measured == pytest.approx(expected, rel=3.e-13)
    # Cutting to the displayed spectrum cannot silently stand in for full sky.
    with pytest.raises(ValueError, match='cover'):
        model.native_sky_rate(0., 0., np.array([2500., 10500.]), np.full(2, 1.e-29),
                              response_wave, response)


def test_duplicate_sky_knots_preserve_ordered_left_and_right_limits():
    model = simulator()
    wave = np.array([2000., 5000., 5000., 7000., 7000., 11097.])
    sky = np.array([1., 1., 2., 2., 2., 2.]) * 1.e-29
    response_wave, response = np.array([2010., 11000.]), np.full(2, .3)
    proxy = .3e-29 * (np.log(5000. / 2010.) + 2. * np.log(11000. / 5000.))
    expected = proxy / h * model.colarea * model.config['arcsecperpix'] ** 2
    assert model.native_sky_rate(0., 0., wave, sky, response_wave, response) == pytest.approx(
        expected, rel=2.e-13)
    with pytest.raises(ValueError, match='nondecreasing'):
        model.native_sky_rate(0., 0., wave[::-1], sky[::-1], response_wave, response)


def test_kernel_response_mapping_empty_invalid_and_zero_batches():
    model = simulator()
    rest = np.geomspace(100., 100000., 301)
    response_wave = np.array([2000., 11000.])
    response = np.array([.3, .3])
    for n in (0, 1):
        flux = np.ones((n, len(rest)))
        wave, raw, valid, invalid, folded = model.kernels_convolve(
            rest, flux, np.full(n, .1), 'GV', .19,
            responses={'first': (response_wave, response)})
        assert raw.shape == folded['first'].shape == (n, len(wave))
        np.testing.assert_allclose(folded['first'], .3 * raw, rtol=1.e-12)
        assert valid.all() and not invalid.any()
    with pytest.raises(ValueError, match='cover'):
        model.kernels_convolve(rest, np.ones((0, len(rest))), np.zeros(0), 'GV', .19,
                               responses={'bad': ([4000., 6000.], [0., 0.])})
    with pytest.raises(ValueError, match='pairs|pair'):
        model.kernels_convolve(rest, np.ones((0, len(rest))), np.zeros(0), 'GV', .19,
                               responses={'bad': 1})


def test_negative_intrinsic_source_is_invalid_without_aborting_good_rows():
    model = simulator()
    rest = np.geomspace(100., 100000., 301)
    flux = np.ones((4, len(rest)))
    flux[1, 100] = -1.e-9
    flux[2, 100] = np.nan
    flux[3] = 0.
    wave, raw, valid, invalid, folded = model.kernels_convolve(
        rest, flux, np.full(4, .1), 'GV', .19,
        responses={'first': (np.array([2000., 11000.]), np.array([.3, .3]))})
    np.testing.assert_array_equal(valid, [True, False, False, True])
    np.testing.assert_array_equal(invalid, ~valid)
    assert np.isnan(raw[~valid]).all() and np.isnan(folded['first'][~valid]).all()
    assert np.isfinite(raw[valid]).all()
    np.testing.assert_array_equal(folded['first'][3], 0.)
    rate = model.response_electron_rate(wave, folded['first'][0] * 1.e-29)
    product = model.add_noise(0., 0., wave, raw[0] * 1.e-29, wave, np.zeros_like(wave),
        np.full_like(wave, .3), np.ones(3), np.ones(3), 'GV', False, False,
        rng=np.random.RandomState(7), source_rate_e_s_cm2=rate, sky_rate_e_s_pixel=0.)
    assert np.isfinite(product[0]).all()
