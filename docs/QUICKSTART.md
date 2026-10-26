# CESS v0.9.0 quick start

English | [简体中文](QUICKSTART_zh-CN.md)

This page covers the shortest common workflows. The [user guide](USER_GUIDE.md)
defines parameters and products; the [Bash manual](COMMANDS.md) explains paths,
line continuation, logging and further examples. Replace `/path/to/...` with
actual paths. All examples assume an activated Python 3.12 environment.

## 1. Install and check the environment

With Conda available, install from the downloaded or extracted source root:

```bash
conda create -n cess python=3.12 pip
conda activate cess
cd /path/to/CESS
python -m pip install '.[tools,plot]'
python -m pip check
python -c 'import sys, cess; print(sys.executable, cess.__version__, cess.__file__)'
cess --version
cess --help
```

The version should be `0.9.0`. Activate an existing suitable environment rather
than creating it again. For venv, wheel/sdist installation or upgrading, see
[installation](INSTALL.md). Choose Conda or venv; a venv does not appear in
`conda env list`. These steps do not assume that CESS is published on PyPI;
consult the actual GitHub release for online availability.

## 2. Explore a tiny example

Enter a new writable working directory. The following filenames and directories
are relative to that directory:

```bash
python /path/to/CESS/examples/make_toy_input.py toy.hdf5 --sources 4 --invalid-first
cess simulate -i toy.hdf5 -o toy_results --all-effects --seed 1214
```

The result is `toy_results/CSST_grism_toy_4.hdf5` with four source rows. Row 0 is
deliberately invalid; its ID and row mapping remain present. The arbitrary toy
SEDs and parameters test the interface, not scientific calibration.

## 3. Process your complete catalogue

These are independent command templates. Absolute paths allow any working
directory. Preview first, without a new `--manifest`, then remove `--dry-run`:

```bash
cess simulate -i /path/to/input.hdf5 -o /path/to/new_results --all-effects --dry-run
cess simulate -i /path/to/input.hdf5 -o /path/to/new_results --all-effects
```

Without source intervals or chunks, the complete input is processed into one
spectral HDF5. Invalid sources retain original rows, IDs and quality flags;
the input catalogue is not changed.

A directory selects its immediate `.hdf5` and `.h5` files. The default is
sequential execution. If memory can hold two tasks, request two workers:

```bash
cess simulate -i /path/to/catalogues -o /path/to/new_batch --all-effects -j 2
```

Each worker handles one file at a time and then takes the next, retaining one
output per input. `-j` defaults to 1; it neither uses all CPUs automatically nor
splits a single file. `--threads-per-worker 1` requests single-threaded native
libraries by default. The run reports scheduling settings, PIDs and timing.
Memory adds across simultaneous tasks: inspect `--dry-run` estimates and a small
run before increasing workers.

## 4. Common options

| Option | Meaning |
| --- | --- |
| `--all-effects` | Enables morphology, wavelength error, seven-band photometry and line analysis; the four switches are otherwise off |
| `--config /path/to/parameters.json` | External JSON; relative resource paths use its directory |
| `--source-start 100 --source-stop 120` | Original rows 100–119; zero-based, exclusive stop |
| `--seed 1214` | Explicit random seed; default 0 |
| `--mode paired-wce` | Paired branches without/with wavelength error, sharing noise |
| `--output-layout legacy` | Schema 2.5; modern default is structured schema 3.0 |
| `--memory-budget-gib 16` | Conservatively reduces concurrency using estimates; not live RSS or a hard memory limit |

Use separate output locations for trial and full runs. Existing products are
never overwritten. If even one task exceeds the budget, reduce chunk size or
use a host with more memory.

## 5. Save a manifest and resume

These commands describe one task: first run, read-only resume preview after its
workers stop, and resume. Preserve input/output, configuration, science switches,
seed, source interval and chunks:

```bash
cess simulate -i /path/to/input.hdf5 -o /path/to/run_results \
  --all-effects --seed 1214 --manifest /path/to/run_results/run.json

cess simulate -i /path/to/input.hdf5 -o /path/to/run_results \
  --all-effects --seed 1214 --manifest /path/to/run_results/run.json --resume --dry-run

cess simulate -i /path/to/input.hdf5 -o /path/to/run_results \
  --all-effects --seed 1214 --manifest /path/to/run_results/run.json --resume
```

`--resume --dry-run` verifies read-only. **A new task with both `--dry-run` and
`--manifest` creates a manifest.** Omit the manifest when no checkpoint is needed.
Changes to code, resources or configuration cannot silently reuse an old
manifest. Split very large directories into batches or per-input manifests.

## 6. Parallel chunks of one file, merged into one product

In the working directory containing `input.hdf5`:

```bash
cess simulate -i input.hdf5 -o chunks --all-effects --chunk-size 1000 -j 2
cess merge -i chunks -o merged/complete.hdf5 --input-source input.hdf5 --dry-run
cess merge -i chunks -o merged/complete.hdf5 --input-source input.hdf5
```

Merging copies results in original row order, checks coverage/compatibility and
remaps line-table source references without recalculation. The output is
self-contained. Choose a new output path outside the chunk directory. Select
chunks of different original inputs separately; do not mix an entire collection
of unrelated catalogue products into one merge.

## 7. Read, plot and postprocess

Use the installed environment:

```python
from cess.io.products import open_product, read_product_summary

print(read_product_summary('/path/to/results/simulation.hdf5'))
with open_product('/path/to/results/simulation.hdf5') as product:
    ids = product.catalog['ID'][:]
    wave = product.band('GV')['wave'][:]                # Angstrom
    flux = product.band('GV')['flux_ujy_with_noise'][0]  # microJy
    row = product.read_row(0)
```

In the same toy working directory as section 2, plot valid row 1 and analyze
lines in the existing product:

```bash
python -m cess.cli.demo toy_results/CSST_grism_toy_4.hdf5 1 emissionline --save spectrum.png
cess el-detect -i toy_results/CSST_grism_toy_4.hdf5 -o toy_lines
```

Plotting requires `plot`; sampling/export requires `tools`. Interpret fields,
units, masks and availability using the [output layout](OUTPUT_LAYOUT.md) and
user guide rather than fixed column numbers or coincidental array lengths.

Use a separate Linux environment and the [Gravity scripts](../scripts/gravity/README.md)
on the server, requesting CPUs/memory and running preflight first. PSF/LSF,
background, response and selection-function validity still require scientific
assessment; see [models and limits](SCIENCE_MODELS.md).
