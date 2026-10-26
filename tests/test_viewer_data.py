"""Headless contracts for the optional read-only HDF5 viewer.

These tests exercise HDF5 axes, exported values and input preservation; they do
not certify GUI rendering, instrument physics or an arbitrary HDF5 schema.
"""
import csv
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

from cess.viewer.data import ProductModel, Variable, validate_export_path, write_csv
from cess.viewer.expressions import filter_mask


def _product(path, layout="structured", count=3, pixels=3, names_attr="columns"):
    wave = np.linspace(4000., 6000., pixels)
    flux = np.arange(count * pixels, dtype=float).reshape(count, pixels) + 1.
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "3.0" if layout == "structured" else "2.5"
        catalog = handle.require_group("catalog") if layout == "structured" else handle
        spectra = handle.require_group("spectra/GV") if layout == "structured" else handle.require_group("GV")
        catalog["ID"] = np.asarray(["source-{}".format(i).encode() for i in range(count)], dtype="S20")
        catalog["input_row_indices"] = np.arange(count, dtype=np.int64) + 100
        catalog["redshift"] = np.linspace(.1, .5, count)
        parameters = catalog.create_dataset("parameters_desi", data=np.arange(count * 2.).reshape(count, 2))
        parameters.attrs[names_attr] = ["RA", "Dec"]
        parameters.attrs["units"] = ["deg", "deg"]
        parameters.attrs["axis_order"] = "source,field"
        spectra["wave"] = wave
        spectra["wave"].attrs["unit"] = "Angstrom"
        spectra["wave"].attrs["axis_order"] = "pixel"
        for name in ("flux_ujy_with_noise", "flux_ujy_model"):
            spectra[name] = flux
            spectra[name].attrs["unit"] = "uJy"
            spectra[name].attrs["axis_order"] = "source,pixel"
        spectra["ferr"] = np.ones_like(flux)
        spectra["snr"] = flux
        spectra["spec_mask"] = np.ones_like(flux, dtype=bool)
    return wave, flux


def _band_path(layout):
    return "spectra/GV" if layout == "structured" else "GV"


def _catalog_path(layout, name):
    return "catalog/" + name if layout == "structured" else name


def _specs(layout):
    root = _band_path(layout)
    return (Variable("wavelength", root + "/wave", "vec", units="Angstrom"),
            Variable("flux", root + "/flux_ujy_with_noise", "col2d", units="uJy"))


@pytest.mark.parametrize("layout", ["structured", "legacy"])
@pytest.mark.parametrize("count", [0, 1, 3])
def test_product_axes_and_empty_rows_are_read_only(tmp_path, layout, count):
    path = tmp_path / "product.hdf5"
    wave, flux = _product(path, layout, count)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    model = ProductModel(str(path))
    try:
        assert model.nsrc == count
        assert len(model.ids_str()) == count
        np.testing.assert_array_equal(model.band_wave("GV"), wave)
        if count:
            np.testing.assert_array_equal(model.spectrum_row("GV", "flux_ujy_with_noise", count - 1), flux[-1])
        assert model.spectrum_row("GU", "flux_ujy_with_noise", 0) is None
        assert model.spectrum_row("GV", "absent", 0) is None
        assert model.band_root("GI") is None
    finally:
        model.close()
    assert not model.h5.id.valid
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("layout", ["structured", "legacy"])
@pytest.mark.parametrize("names_attr", ["name", "columns"])
def test_named_parameter_columns_and_ids(tmp_path, layout, names_attr):
    path = tmp_path / "names.hdf5"
    _product(path, layout, names_attr=names_attr)
    model = ProductModel(str(path))
    try:
        assert model.table_columns(_catalog_path(layout, "parameters_desi")) == [("RA", "deg"), ("Dec", "deg")]
        assert model.ids_str().tolist() == ["source-0", "source-1", "source-2"]
        assert model.row_of_id("source-1") == 1
        assert model.row_of_id("SOURCE-2") == 2
        assert model.row_of_id("absent") is None
        assert model.unit_of(_band_path(layout) + "/wave") == "Angstrom"
        assert any("RA" in name for name in model.variable_map())
    finally:
        model.close()


@pytest.mark.parametrize("layout", ["structured", "legacy"])
@pytest.mark.parametrize("pixels", [3, 5])
def test_wavelength_matrix_pair_always_uses_source_row(tmp_path, layout, pixels):
    path = tmp_path / "spectral_axes.hdf5"
    wave, flux = _product(path, layout, count=3, pixels=pixels)
    model = ProductModel(str(path))
    try:
        x_spec, y_spec = _specs(layout)
        x, y, used = model.resolve_xy(x_spec, y_spec, pixel=0, row=1)
        np.testing.assert_array_equal(x, wave)
        np.testing.assert_array_equal(y, flux[1])
        assert used == 1
        y, x, used = model.resolve_xy(y_spec, x_spec, pixel=0, row=2)
        np.testing.assert_array_equal(x, wave)
        np.testing.assert_array_equal(y, flux[2])
        assert used == 2
    finally:
        model.close()


def test_equal_source_vector_length_does_not_imply_wavelength_axis(tmp_path):
    path = tmp_path / "source_axes.hdf5"
    _, flux = _product(path)
    model = ProductModel(str(path))
    try:
        source = Variable("redshift", "catalog/redshift", "vec")
        _, spectrum = _specs("structured")
        x, y, used = model.resolve_xy(source, spectrum, pixel=2, row=1)
        np.testing.assert_array_equal(x, model.h5["catalog/redshift"][:])
        np.testing.assert_array_equal(y, flux[:, 2])
        assert used is None
    finally:
        model.close()


def test_named_parameter_table_is_not_inferred_as_a_spectral_matrix(tmp_path):
    path = tmp_path / "table_axes.hdf5"
    _product(path, count=3, pixels=2)
    model = ProductModel(str(path))
    try:
        wavelength, _ = _specs("structured")
        table = Variable("parameters", "catalog/parameters_desi", "col2d")
        x, y, used = model.resolve_xy(wavelength, table, pixel=1, row=1)
        assert used is None
        assert len(x) == 2 and len(y) == 3
        np.testing.assert_array_equal(y, [1., 3., 5.])
    finally:
        model.close()


@pytest.mark.parametrize("row", [-1, 3])
def test_invalid_source_rows_do_not_silently_select_a_different_source(tmp_path, row):
    path = tmp_path / "row_bounds.hdf5"
    _product(path)
    model = ProductModel(str(path))
    try:
        with pytest.raises(IndexError):
            model.row_vector("spectra/GV/flux_ujy_with_noise", row)
    finally:
        model.close()


def test_spectrum_plot_and_error_read_only_selected_source_row(tmp_path, monkeypatch):
    path = tmp_path / "lazy.hdf5"
    wave, flux = _product(path, count=7, pixels=9)
    model = ProductModel(str(path))
    reads = []
    original = h5py.Dataset.__getitem__

    def observed(dataset, key, *args, **kwargs):
        if dataset.name.startswith("/spectra/GV/") and dataset.ndim == 2:
            reads.append((dataset.name, key))
        return original(dataset, key, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", observed)
    try:
        x_spec, y_spec = _specs("structured")
        x, y, used = model.resolve_xy(x_spec, y_spec, pixel=0, row=4)
        error = Variable("error", "spectra/GV/ferr", "col2d")
        np.testing.assert_array_equal(model.resolve_for_axis(error, 0, used, len(wave)), np.ones(9))
        np.testing.assert_array_equal(x, wave)
        np.testing.assert_array_equal(y, flux[4])
        assert reads
        for _, key in reads:
            first = key[0] if isinstance(key, tuple) else key
            assert isinstance(first, (int, np.integer)) and first == 4, reads
    finally:
        model.close()


def test_oversized_values_are_not_retained_in_the_small_cache(tmp_path, monkeypatch):
    path = tmp_path / "cache_budget.hdf5"
    _product(path, count=4, pixels=12)
    calls = []
    original = h5py.Dataset.__getitem__

    def observed(dataset, key, *args, **kwargs):
        if dataset.name == "/spectra/GV/wave":
            calls.append(key)
        return original(dataset, key, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", observed)
    with ProductModel(str(path), cache_bytes=16) as model:
        first = model.read("spectra/GV/wave")
        second = model.read("spectra/GV/wave")
        np.testing.assert_array_equal(first, second)
        assert len(calls) == 2


@pytest.mark.parametrize("layout", ["structured", "legacy"])
def test_legacy_and_structured_reference_windows_and_vlen_peaks(tmp_path, layout):
    path = tmp_path / "lines.hdf5"
    _product(path, layout)
    line_root = "lines/GV" if layout == "structured" else "GV"
    dtype = np.dtype([("source_row", "i8"), ("center_wave", "f8"), ("snr", "f8"), ("valid", "?")])
    rows = np.array([(1, 4100., 4., True), (0, 4200., -2., True),
                     (1, 4300., np.nan, False), (1, 4400., -3., True)], dtype=dtype)
    with h5py.File(path, "a") as handle:
        root = handle.require_group(line_root)
        root["line_measurements/truth"] = rows
        root["intri_el_id"] = np.array([1], dtype=np.int64)
        root["detect_el_id"] = np.array([1], dtype=np.int64)
        for prefix, peaks in (("intri", [4100., 4300., 4400.]), ("detect", [4101., 4402.])):
            data = root.create_dataset(prefix + "_el_wave", (1,), dtype=h5py.vlen_dtype(np.dtype("f8")))
            data[0] = np.asarray(peaks)
    model = ProductModel(str(path))
    try:
        waves, snr = model.measured_lines("GV", 1)
        np.testing.assert_array_equal(waves, [4100., 4400.])
        np.testing.assert_array_equal(snr, [4., -3.])
        np.testing.assert_array_equal(model.line_peaks("GV", 1, detected=False), [4100., 4300., 4400.])
        np.testing.assert_array_equal(model.line_peaks("GV", 1, detected=True), [4101., 4402.])
        assert model.line_peaks("GV", 2, detected=True).size == 0
    finally:
        model.close()


def test_filter_supports_array_boolean_arithmetic_without_python_execution():
    x = np.array([0., 1., 2., 3.])
    y = np.array([1., np.nan, -4., 9.])
    actual = filter_mask(x, y, "(x > 0) & np.isfinite(y) & (np.abs(y) < 8)")
    np.testing.assert_array_equal(actual, [False, False, True, False])


def test_empty_filter_retains_finite_negative_values():
    actual = filter_mask(np.array([0., 1., np.nan, 3.]), np.array([-2., np.inf, 3., 4.]))
    np.testing.assert_array_equal(actual, [True, False, False, True])


@pytest.mark.parametrize("expression", [
    "__import__('os').getcwd()", "open('private.txt')", "np.load('private.npy')",
    "x.__class__", "np.__dict__", "[value for value in x]", "(lambda: True)()",
    "True", "1", "x", "np.array([True])",
])
def test_filter_rejects_execution_and_non_array_masks(expression):
    with pytest.raises(ValueError):
        filter_mask(np.arange(3.), np.ones(3), expression)


def test_filter_rejects_incompatible_axes():
    with pytest.raises(ValueError):
        filter_mask(np.arange(3.), np.ones(2), "x > 0")


def test_csv_export_roundtrips_full_precision_and_quotes_text(tmp_path):
    source = tmp_path / "source.hdf5"
    with h5py.File(source, "w") as handle:
        handle["ID"] = [1]
    target = tmp_path / "table.csv"
    exact = np.nextafter(1., 2.)
    text = 'a comma, a "quote" and\na new line'
    write_csv(target, ["value, precise", "label"], [[exact, text], [1.2345678901234567, "source-2"]], source)
    with target.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ["value, precise", "label"]
    assert float(rows[1][0]) == exact
    assert rows[1][1] == text
    assert float(rows[2][0]) == 1.2345678901234567


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_csv_export_refuses_input_or_any_alias(tmp_path, alias):
    source = tmp_path / "input.hdf5"
    with h5py.File(source, "w") as handle:
        handle["ID"] = [1]
    destination = source
    if alias != "direct":
        destination = tmp_path / "pretend.csv"
        if alias == "symlink":
            destination.symlink_to(source)
        else:
            os.link(source, destination)
    original = source.read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        validate_export_path(destination, source)
    with pytest.raises((ValueError, FileExistsError)):
        write_csv(destination, ["value"], [[2]], source)
    assert source.read_bytes() == original


def test_csv_export_failure_does_not_leave_partial_destination(tmp_path):
    source = tmp_path / "input.hdf5"
    with h5py.File(source, "w") as handle:
        handle["ID"] = [1]
    destination = tmp_path / "new.csv"

    def incomplete():
        yield [1]
        raise ValueError("failed to read the remaining source rows")

    with pytest.raises(ValueError, match="remaining source rows"):
        write_csv(destination, ["value"], incomplete(), source)
    assert not destination.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_csv_export_cannot_replace_a_different_hdf5_file(tmp_path):
    source, other = tmp_path / "source.hdf5", tmp_path / "other.h5"
    for path in (source, other):
        with h5py.File(path, "w") as handle:
            handle["ID"] = [1]
    before = other.read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        write_csv(other, ["value"], [[3]], source)
    assert other.read_bytes() == before


def test_help_does_not_import_gui_or_numerical_dependencies(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src"
    code = '''
import importlib.abc, sys
class RejectHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'tkinter', 'matplotlib', 'numpy', 'h5py'}:
            raise AssertionError('viewer help imported optional dependency: ' + fullname)
sys.meta_path.insert(0, RejectHeavyImports())
from cess.viewer import main
main(['--help'])
'''
    environment = dict(os.environ, PYTHONPATH=str(source), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-s", "-B", "-c", code], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()
