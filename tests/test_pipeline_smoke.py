"""Tiny end-to-end runner checks; scientific inputs/outputs stay in tmp_path.

Use grizli Python. These checks exercise I/O and repaired failure paths, not the
scientific completeness or observing sensitivity of the full catalogue.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import h5py
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src"
sys.path.insert(0, str(CORE))
from cess.config.loader import load_config


def make_input(path, count=4, invalid=()):
    """BayeSED-shaped synthetic SEDs: rest microns, fnu in microJy."""
    rest_aa = np.geomspace(100., 100000., 2079)
    flux = np.ones((count, len(rest_aa)))
    for center in (3727., 4861., 5007., 6563.):
        flux += 20. * np.exp(-0.5 * ((rest_aa - center) / 12.) ** 2)
    z = np.linspace(.2, .35, count)
    for index in invalid:
        flux[index] = np.nan
    names = ["RA", "DEC", "MAG_G", "MAG_R", "MAG_Z",
             "log(M*)[0,1]_{MAL}", "z_{MAL}"]
    parameters = np.column_stack([np.full(count, 120.), np.full(count, 20.),
                                  np.full(count, 20.5), np.full(count, 20.2),
                                  np.full(count, 20.), np.full(count, 10.), z])
    with h5py.File(str(path), "w") as handle:
        handle.create_dataset("ID", data=np.array(
            ["smoke-{}".format(i).encode() for i in range(count)], dtype="S20"))
        handle.create_dataset("parameters", data=parameters)
        handle.create_dataset("parameters_name", data=np.array(names, dtype="S40"))
        model = handle.create_group("best_fit")
        wave = model.create_dataset("wavelength_rest", data=rest_aa / 1.e4)
        wave.attrs["unit"] = "um"
        model.create_dataset("z", data=z)
        sed = model.create_dataset("spec_synthetic", data=flux)
        sed.attrs["unit"] = "uJy"


def run_fixture(directory, module="cess.pipeline.standard", switches=(), count=4, invalid=()):
    input_dir, output_dir = directory / "input", directory / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    make_input(input_dir / "synthetic.hdf5", count=count, invalid=invalid)
    config = load_config(CORE / "cess/data/emulator_parameters.json")
    config["input_path"] = str(input_dir) + os.sep
    config["output_path"] = str(output_dir) + os.sep
    config["el_input_path"] = str(input_dir) + os.sep
    config["el_output_path"] = str(output_dir) + os.sep
    (directory / "emulator_parameters.json").write_text(json.dumps(config))
    options = {argument.split("=", 1)[0]: argument.split("=", 1)[1] in ("true", "yes")
               for argument in switches}
    code = ("import sys, random, numpy as np; "
            "sys.path.insert(0, {core!r}); "
            "random.seed(1214); np.random.seed(1214); "
            "import {module} as driver; "
            "driver.task({input_dir!r}, 'synthetic.hdf5', config={config!r}, "
            "options={options!r}, seed=1214)").format(
                core=str(CORE), module=module, input_dir=str(input_dir) + os.sep,
                config=str(directory / "emulator_parameters.json"), options=options)
    env = dict(os.environ, MPLCONFIGDIR=str(directory / "mpl"),
               PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-B", "-c", code, "0", "0"] + list(switches),
                            cwd=str(directory), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True,
                            timeout=60)
    assert result.returncode == 0, result.stdout
    products = list(output_dir.glob("*.hdf5"))
    assert len(products) == 1, result.stdout
    return products[0]


@pytest.mark.parametrize("module,switches", [
    ("cess.pipeline.standard", ()),
    ("cess.pipeline.standard", ("morph=true", "wave_cal=true", "photo=true", "el_detect=true")),
    ("cess.pipeline.paired", ()),
    ("cess.pipeline.paired", ("morph=true", "wave_cal=true", "el_detect=true")),
])
def test_mixed_valid_invalid_rows(tmp_path, module, switches):
    path = run_fixture(tmp_path, module, switches, invalid=(0, 2))
    with h5py.File(str(path), "r") as handle:
        assert len(handle["ID"]) == 4
        np.testing.assert_array_equal(handle["data_mask"][:], [2, 1, 2, 1])
        assert np.isnan(handle["parameters_desi"][[0, 2], 6:10]).all()
        assert np.isfinite(handle["parameters_desi"][[1, 3], 6:10]).all()
        for band in ("GU", "GV", "GI"):
            for key in ("flux_ujy", "flux_ujy_with_noise", "ferr", "snr"):
                assert handle[band + "/" + key].shape == (4, len(handle[band + "/wave"]))
            assert np.isnan(handle[band + "/flux_ujy_with_noise"][[0, 2]]).all()
            valid = np.asarray(handle[band + "/spec_mask"][1], dtype=bool)
            assert valid.any()
            assert np.isfinite(handle[band + "/ferr"][1][valid]).all()
        if "parameters_phot" in handle:
            assert handle["parameters_phot"].shape == (4, 21)
        if "el_detect=true" in switches:
            # Known strong synthetic lines must survive invalid rows/WCE edges.
            assert len(handle["GV/intri_el_id"]) > 0
            assert len(handle["GI/intri_el_id"]) > 0
            for band in ("GV", "GI"):
                assert "el_diagnostics" in handle[band]


@pytest.mark.parametrize("count,invalid", [(1, ()), (2, (0, 1))])
def test_single_or_all_invalid(tmp_path, count, invalid):
    path = run_fixture(tmp_path, count=count, invalid=invalid)
    with h5py.File(str(path), "r") as handle:
        assert len(handle["ID"]) == count
        assert handle["GU/flux_ujy_with_noise"].shape[0] == count
        expected = np.full(count, 2 if invalid else 1)
        np.testing.assert_array_equal(handle["data_mask"][:], expected)
