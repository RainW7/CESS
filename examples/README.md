# CESS v0.9.0 One-Dimensional Interface Example

[English](README.md) | [简体中文](README_zh-CN.md)

`make_toy_input.py` creates a small, publicly distributable BayeSED-shaped HDF5.
Its continuum, line profiles, and galaxy parameters are arbitrary, intended for installation and interface demonstrations rather than scientific calibration.
For research, use galaxy templates that satisfy the [input contract](../docs/USER_GUIDE.md).

## 1. Prepare the environment and working directory

Install CESS following the [installation guide](../docs/INSTALL.md) and activate the selected Python 3.12 environment.
Create and enter a new writable working directory. The `toy.hdf5`, `results`, `run.json`, `lines`, and image paths below
are relative to this directory. Replace `/path/to/CESS` with your source location.

```bash
python -c 'import sys, cess; print(sys.executable, cess.__version__, cess.__file__)'
python /path/to/CESS/examples/make_toy_input.py toy.hdf5 --sources 4 --invalid-first
```

## 2. Simulate the complete file

```bash
cess simulate -i toy.hdf5 -o results --all-effects --seed 1214 --manifest run.json
```

The default is one worker. All four source rows are written to `results/CSST_grism_toy_4.hdf5`.
Source 0 deliberately has an invalid SED. Its ID and row index should remain, with `data_mask=2` and unavailable simulation values;
the other three sources are processed normally. `--all-effects` enables all four scientific switches, which are otherwise off by default.

Without an explicit JSON, the bundled default instrument resources are used. The modern entry point defaults to schema 3.0.
For the old layout, choose new output and manifest locations and add `--output-layout legacy`.
Existing inputs, products, and manifests are not silently overwritten. To repeat the exercise, choose a new working directory or use the resume procedure below.

## 3. Read and plot the result

In the same working directory, run this in a Python environment or notebook with CESS installed:

```python
from cess.io.products import open_product, read_product_summary

print(read_product_summary('results/CSST_grism_toy_4.hdf5'))
with open_product('results/CSST_grism_toy_4.hdf5') as product:
    print(product.catalog['ID'][:])
    print(product.catalog['data_mask'][:])
    print(product.catalog['input_row_indices'][:])
    wave = product.band('GV')['wave'][:]                 # Angstrom
    flux = product.band('GV')['flux_ujy_with_noise'][1]   # microJy
```

Plot the valid source at row 1; this requires the optional `plot` dependencies:

```bash
python -m cess.cli.demo results/CSST_grism_toy_4.hdf5 1 emissionline --save spectrum.png
```

## 4. Emission-line postprocessing

```bash
cess el-detect -i results/CSST_grism_toy_4.hdf5 -o lines --manifest lines.json --seed 1214
```

This step reads the existing simulation product and writes a separate line-measurement product without regenerating the noisy spectrum.
EL-only files do not copy the large parent spectral arrays. Retain the parent simulation file for later reading and provenance interpretation.

## 5. Manifests and resume

Confirm that the old workers have stopped and retain the original arguments, then perform read-only preflight before resuming:

```bash
cess simulate -i toy.hdf5 -o results --all-effects --seed 1214 --manifest run.json --resume --dry-run
cess simulate -i toy.hdf5 -o results --all-effects --seed 1214 --manifest run.json --resume
```

Completed tasks whose products pass verification are retained; missing products are recalculated. Do not reuse the original manifest after changing the input, configuration, source interval, chunk plan,
output layout, scientific switches, seed, or code/resource identity.
For a new-run preview, omit the manifest. A new `--dry-run --manifest` invocation creates one.

To run source code without installing the package, replace `cess` in these commands with
`env PYTHONPATH=/path/to/CESS/src python -s -B -m cess`; the interpreter must still have the required dependencies.
For real-catalogue batches, concurrency, chunk merging, logs, and common errors, see the [Bash command guide](../docs/COMMANDS.md).
