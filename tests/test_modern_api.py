"""Compatibility regressions for NumPy 2 and modern Specutils, without resources."""
import warnings

import h5py
import numpy as np
import pytest
import specutils
from astropy import units as u
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Chebyshev1D
from astropy.table import QTable
from astropy.utils.exceptions import AstropyUserWarning

from cess.constants import h
from cess.instrument.detector import DetectorMixin
from cess.io.selection import _SelectedDataset
from cess.lines import detection
from cess.morphology.mge import sersic_mge_marginals
from cess.morphology.reference import sersic_pixel_reference
from cess.spectra.numerics import trapezoid


@pytest.fixture
def selected(tmp_path):
    with h5py.File(tmp_path / 'selection.h5', 'w') as handle:
        handle['flux'] = np.arange(15., dtype=np.float64).reshape(5, 3)
        yield _SelectedDataset(handle['flux'], slice(1, 4))


def test_array_copy_false_does_not_materialize_hdf5(selected):
    with pytest.raises(ValueError, match='materialize'):
        selected.__array__(copy=False)
    assert selected._data is None
    if np.lib.NumpyVersion(np.__version__) >= '2.0.0':
        with pytest.raises(ValueError, match='materialize'):
            np.asarray(selected, copy=False)
        assert selected._data is None


def test_array_copy_none_materializes_and_reuses_same_dtype(selected):
    result = selected.__array__(copy=None)
    assert result is selected._data
    np.testing.assert_array_equal(result, np.arange(15.).reshape(5, 3)[1:4])
    assert selected.__array__(copy=False) is result
    assert selected.__array__(dtype=np.float64, copy=False) is result
    if np.lib.NumpyVersion(np.__version__) >= '2.0.0':
        assert np.asarray(selected, copy=False) is result


@pytest.mark.parametrize('dtype', [None, np.float32])
def test_array_copy_true_always_allocates(selected, dtype):
    original = np.asarray(selected)
    result = selected.__array__(dtype=dtype, copy=True)
    assert not np.shares_memory(original, result)
    np.testing.assert_array_equal(result, original)
    result[0, 0] = -10.
    assert original[0, 0] == 3.


def test_array_dtype_conversion_obeys_copy_request(selected):
    original = np.asarray(selected)
    with pytest.raises(ValueError, match='dtype'):
        selected.__array__(dtype=np.float32, copy=False)
    converted = selected.__array__(dtype=np.float32, copy=None)
    assert converted.dtype == np.dtype(np.float32)
    assert not np.shares_memory(converted, original)
    np.testing.assert_array_equal(converted, original)


def test_array_structured_dtype_and_invalid_copy(tmp_path):
    values = np.array([(1, .5), (2, 1.5)], dtype=[('id', 'i8'), ('flux', 'f8')])
    with h5py.File(tmp_path / 'structured.h5', 'w') as handle:
        handle['table'] = values
        selected = _SelectedDataset(handle['table'], slice(0, 2))
        original = np.asarray(selected)
        assert selected.__array__(dtype=values.dtype, copy=False) is original
        copied = selected.__array__(copy=True)
        assert copied.dtype == values.dtype
        assert not np.shares_memory(copied, original)
        np.testing.assert_array_equal(copied, values)
        with pytest.raises(TypeError, match='copy'):
            selected.__array__(copy='false')


def test_trapezoid_retains_irregular_grid_and_axis_convention():
    wave = np.array([0., 1., 3., 6.])
    values = np.array([1., 2., 4., 8.])
    assert trapezoid(values, wave) == 25.5
    assert trapezoid(values[::-1], wave[::-1]) == -25.5
    np.testing.assert_array_equal(
        trapezoid(np.stack((values, 2.*values)), wave, axis=1), [25.5, 51.])


def test_sky_integration_retains_photon_units_and_pixel_area():
    model = DetectorMixin()
    model.config = {'background_mode': 'fixed_template', 'background_scale': .7,
                    'arcsecperpix': .074}
    model.colarea = 10000.
    wave = np.array([4000., 4700., 6100., 8000.])
    fnu = np.array([1., 2., 1.5, .5])*1.e-28
    response = np.array([0., .4, .7, .2])
    rate_per_wavelength = response*fnu*.7/(h*wave)
    expected = np.sum(np.diff(wave)*(rate_per_wavelength[:-1]+rate_per_wavelength[1:])/2.)
    expected *= model.colarea*.074**2
    assert model.bkg_per_pix_time(np.nan, np.nan, wave, fnu, response) == expected


def test_mge_gaussian_fit_exercises_integral_and_retains_absolute_bin_mass():
    edges = np.arange(-6.5, 7.5)
    actual = sersic_mge_marginals(.5, 4., 35., .4, .2625, .074, edges,
                                  max_components=32)
    expected = sersic_pixel_reference(.5, 4., 35., .4, .2625, .074, edges)
    for axis in ('x', 'y'):
        np.testing.assert_allclose(actual[axis+'_probability'], expected[axis+'_probability'],
                                   rtol=0., atol=5.e-11)
        assert 0. < actual[axis+'_probability'].sum() < 1.
    assert actual['metadata']['source_fit']['estimated_radial_L1_including_outside'] < 1.e-7
    assert actual['metadata']['normalization']['window_renormalized'] is False


def test_continuum_hook_explicitly_retains_legacy_mathematical_defaults(monkeypatch):
    calls = []
    sentinel = object()
    def capture(spectrum, **kwargs):
        calls.append((spectrum, kwargs))
        return sentinel
    monkeypatch.setattr(detection, '_fit_generic_continuum', capture)
    assert detection.fit_generic_continuum('spectrum') is sentinel
    spectrum, options = calls[0]
    assert spectrum == 'spectrum'
    assert options['median_window'] == 3
    assert isinstance(options['model'], Chebyshev1D) and options['model'].degree == 3
    np.testing.assert_array_equal(options['model'].parameters, np.zeros(4))
    assert isinstance(options['fitter'], LevMarLSQFitter)
    assert options['exclude_regions'] is None and options['weights'] is None
    assert specutils.conf.do_continuum_function_check is False


def test_default_continuum_removes_only_advisory_without_changing_fit():
    wave = np.linspace(4000., 6000., 501)
    flux = 1.+.0002*(wave-5000.)+5.*np.exp(-.5*((wave-5000.)/12.)**2)
    spectrum = detection.Spectrum1D(flux=flux*u.uJy, spectral_axis=wave*u.angstrom)
    message = 'Model is linear in parameters; consider using linear fitting methods.'
    with pytest.warns(AstropyUserWarning, match='Model is linear in parameters'):
        reference = detection._fit_generic_continuum(
            spectrum, median_window=3, model=Chebyshev1D(3),
            fitter=LevMarLSQFitter(), exclude_regions=None, weights=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        actual = detection.fit_generic_continuum(spectrum)
    assert not any(str(item.message) == message for item in caught)
    np.testing.assert_array_equal(actual.parameters, reference.parameters)
    np.testing.assert_array_equal(actual(wave*u.angstrom), reference(wave*u.angstrom))


def test_continuum_advisory_filter_keeps_other_diagnostics_and_is_local(monkeypatch):
    message = 'Model is linear in parameters; consider using linear fitting methods.'
    diagnostics = [
        (message, AstropyUserWarning, 'astropy.modeling.fitting'),
        ('The fit may be unsuccessful; check fit_info.', AstropyUserWarning,
         'astropy.modeling.fitting'),
        (message, RuntimeWarning, 'astropy.modeling.fitting'),
        (message, AstropyUserWarning, 'custom.fitting'),
        (message+' Additional information.', AstropyUserWarning,
         'astropy.modeling.fitting')]
    sentinel = object()
    def emit(*args, **kwargs):
        for text, category, module in diagnostics:
            warnings.warn_explicit(text, category, __file__, 1, module=module)
        return sentinel
    monkeypatch.setattr(detection, '_fit_generic_continuum', emit)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert detection.fit_generic_continuum('spectrum') is sentinel
        assert len(caught) == len(diagnostics)-1
        assert [str(item.message) for item in caught] == [
            item[0] for item in diagnostics[1:]]
        assert [item.category for item in caught] == [item[1] for item in diagnostics[1:]]
        # The filter must be restored on return, even inside a caller's context.
        warnings.warn_explicit(message, AstropyUserWarning, __file__, 1,
                               module='astropy.modeling.fitting')
        assert len(caught) == len(diagnostics)


@pytest.mark.parametrize('custom', ['model', 'fitter'])
def test_custom_continuum_fit_keeps_linear_solver_advisory(custom):
    wave = np.linspace(4000., 6000., 21)
    spectrum = detection.Spectrum1D(flux=np.ones(21)*u.uJy,
                                    spectral_axis=wave*u.angstrom)
    options = {'model': Chebyshev1D(3)} if custom == 'model' else {
        'fitter': LevMarLSQFitter()}
    with pytest.warns(AstropyUserWarning, match='Model is linear in parameters'):
        detection.fit_generic_continuum(spectrum, **options)


def test_continuum_fitting_failure_propagates(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError('continuum fitting failed')
    monkeypatch.setattr(detection, '_fit_generic_continuum', broken)
    with pytest.raises(RuntimeError, match='continuum fitting failed'):
        detection.fit_generic_continuum('spectrum')


def test_modern_spectrum_preserves_legacy_factory_hook(monkeypatch):
    factory = detection.Spectrum1D
    if hasattr(specutils, 'Spectrum'):
        assert factory is specutils.Spectrum
    calls = []
    def capture(**kwargs):
        calls.append(kwargs)
        return factory(**kwargs)
    expected = QTable({'line_center': [5000.]*u.angstrom, 'line_type': ['emission'],
                       'line_center_index': [10]})
    monkeypatch.setattr(detection, 'Spectrum1D', capture)
    monkeypatch.setattr(detection, 'fit_generic_continuum',
                        lambda spectrum: lambda wave: np.ones(len(wave))*u.uJy)
    monkeypatch.setattr(detection, 'find_lines_derivative',
                        lambda spectrum, flux_threshold: expected)
    result = detection.LineDetectionMixin().el_detect(
        np.linspace(4000., 6000., 21), np.ones(21), 0, 21, 1.5)
    assert result is expected and len(calls) == 1
    assert calls[0]['flux'].unit == u.uJy
    assert calls[0]['spectral_axis'].unit == u.angstrom


@pytest.mark.parametrize('slope', [0., .0002])
def test_true_emission_line_survives_continuum_fit_and_detection(slope):
    wave = np.linspace(4000., 6000., 501)
    flux = 1.+slope*(wave-5000.)+5.*np.exp(-.5*((wave-5000.)/12.)**2)
    flux[[0, -1]] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        lines = detection.LineDetectionMixin().el_detect(wave, flux, 20, 480, 1.5)
    emission = lines[np.asarray(lines['line_type']).astype(str) == 'emission']
    assert len(emission) == 1
    # The derivative zero crossing is localized to one input-grid interval.
    assert abs(emission['line_center'][0].to_value(u.angstrom)-5000.) <= 4.
    assert not any('Spectrum1D' in str(item.message) and 'deprecated' in str(item.message)
                   for item in caught)


@pytest.mark.parametrize('invalid', ['nan', 'zero', 'reversed', 'short'])
def test_numerically_unusable_detection_window_returns_empty(invalid):
    wave = np.linspace(4000., 6000., 21)
    flux = np.ones(21)
    if invalid == 'nan':
        flux[10] = np.nan
    elif invalid == 'zero':
        flux[:] = 0.
    elif invalid == 'reversed':
        wave = wave[::-1]
    elif invalid == 'short':
        wave, flux = wave[:3], flux[:3]
    result = detection.LineDetectionMixin().el_detect(wave, flux, 0, len(wave), 1.5)
    assert len(result) == 0
    assert result.colnames == ['line_center', 'line_type', 'line_center_index']


@pytest.mark.parametrize('hook', ['fit_generic_continuum', 'find_lines_derivative'])
@pytest.mark.parametrize('error', [TypeError, AttributeError, RuntimeError])
def test_dependency_and_programming_errors_propagate(monkeypatch, hook, error):
    monkeypatch.setattr(detection, 'fit_generic_continuum',
                        lambda spectrum: lambda wave: np.ones(len(wave))*u.uJy)
    def broken(*args, **kwargs):
        raise error('dependency failure')
    monkeypatch.setattr(detection, hook, broken)
    with pytest.raises(error, match='dependency failure'):
        detection.LineDetectionMixin().el_detect(
            np.linspace(4000., 6000., 21), np.ones(21), 0, 21, 1.5)


def test_incompatible_line_table_is_not_reported_as_no_detections(monkeypatch):
    monkeypatch.setattr(detection, 'fit_generic_continuum',
                        lambda spectrum: lambda wave: np.ones(len(wave))*u.uJy)
    monkeypatch.setattr(detection, 'find_lines_derivative',
                        lambda *args, **kwargs: QTable({'unexpected_column': []}))
    with pytest.raises(ValueError, match='incompatible table'):
        detection.LineDetectionMixin().el_detect(
            np.linspace(4000., 6000., 21), np.ones(21), 0, 21, 1.5)
