# User guide to one-dimensional CESS simulations

[English](USER_GUIDE.md) | [简体中文](USER_GUIDE_zh-CN.md)

Updated 2026-10-26. This guide applies to CESS `0.9.0`, revision `20261024-D4`.

The [README](../README.md) describes the software, capabilities, and version changes. This guide follows the order of a typical workflow.
If you are new to the command line, start with the [Bash command guide](COMMANDS.md), which explains paths, quoting,
line continuation, parallel execution, logs, and resume. The [quick start](QUICKSTART.md) provides the shortest route to a run.

| Goal | Where to look |
| --- | --- |
| Create an environment, install CESS, and check the interpreter | Section 2; see the [installation guide](INSTALL.md) for complete options |
| Run a file, directory, parallel batch, or source-row subset | Section 6; see the [command guide](COMMANDS.md) for complete commands |
| Preview or resume a run | Section 7 |
| Postprocess emission lines | Section 8 |
| Read HDF5, plot spectra, and sample products | Sections 9–10 |
| Advanced morphology and Gravity | Section 12 and its linked documents |

## 1. From input to output

CESS converts an existing BayeSED galaxy-template library and instrument configuration into GU/GV/GI one-dimensional
slitless spectra. It computes spectral response, expected electrons, and observational noise, and writes nominally calibrated
flux, errors, S/N, masks, and provenance. Morphology effects, wavelength-calibration error, seven-band photometry, and
emission-line detection can be enabled separately. Source morphology also uses the configured GEMS/profile resources;
an SED file alone does not replace a complete runtime configuration.

The modern `cess` / `python -m cess` entry point writes structured HDF5 schema 3.0 by default.
`--output-layout legacy` writes compatible schema 2.5. Package version, repair revision, and file schema are separate identifiers;
one cannot be inferred from another.

## 2. Create a cess environment, install, and run

The package requires Python >=3.12,<3.13. `/path/to/...` and `/absolute/...` are placeholders throughout this guide:
replace them with real paths. `/path/to/CESS` is the source root containing `setup.cfg` and `pyproject.toml`.
Relative data paths are resolved from the working directory in which the command runs.

### 2.1 First installation with Conda

If Conda is available, create a separate environment named `cess`, then install CESS.
If that environment already exists and its Python version is suitable, activate it instead:

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

`conda create` creates the environment, which then appears in `conda env list`.
`python -m pip install '.[tools,plot]'` installs CESS, core dependencies, and the sampling/plotting extras automatically;
you do not need to install NumPy, Astropy, and the other packages one by one. Keep the quotes, including in zsh.
`pip check` should report no broken requirements. The reported interpreter and CESS paths should belong to the selected environment.
These checks confirm installation and entry points, not scientific calibration.

For the 1D core alone, use `python -m pip install .`. `tools` includes pandas, while `plot` includes Matplotlib.
The optional `render` extra provides GalSim for selected legacy rendering/library tools; normal 1D simulation with precomputed
libraries does not require it. See the [installation guide](INSTALL.md) for full requirements, development/test installation, and platform limits.

### 2.2 Subsequent sessions and the venv alternative

In each new terminal, activate the existing Conda environment. You can then run from your data working directory:

```bash
conda activate cess
cd /path/to/data
cess simulate -i input.hdf5 -o results --all-effects
```

Here `input.hdf5` and `results` are relative to `/path/to/data`.
The complete input is processed by default, producing one spectral HDF5 with all source rows;
invalid sources retain their original row positions and are flagged. To process every input in a directory, replace
`-i input.hdf5` with `-i input_directory`. Each input file still produces one spectral file.
`--all-effects` enables all four optional science switches. The default is one computational process, so directory inputs
are processed sequentially. If memory permits, add `-j N` to process at most N files simultaneously, retaining one complete output per file.

An existing working environment does not need to be recreated. Without Conda, follow the
[venv installation steps](INSTALL.md#12-using-python-venv), then activate it in each new terminal with
`source /path/to/cess-env/bin/activate`. A venv does not appear in `conda env list`.
Choose one environment method; leave it using `conda deactivate` or `deactivate`, respectively.
Do not copy a Mac environment directory or ARM64 lock to Linux and assume it is usable there.

### 2.3 Prepare the later examples

The examples below use the installed `cess` command. To run repository toy scripts and plotting examples,
set these variables in the active environment:

```bash
REPO=/path/to/CESS
PY="$(python -c 'import sys; print(sys.executable)')"
"$PY" --version
cess simulate --help
```

`REPO` is the real source root, and `PY` points to the active environment's interpreter.
These ordinary shell variables exist only in the current terminal.
During source development, replace `cess` with `env PYTHONPATH="$REPO/src" "$PY" -s -B -m cess`;
the interpreter still needs the dependencies. The current macOS ARM64
[platform lock](migration_py312_20261023/requirements-macos-arm64-py312.lock) records that platform's dependencies only.
Other platforms require their own installation and checks.

## 3. First run: the public toy example

The toy continuum, emission lines, and galaxy parameters are arbitrary interface inputs.
They do not represent a realistic population or calibrate the science model.
The following example creates a fresh temporary working directory and four sources, with the first deliberately invalid
to demonstrate row preservation. Run the steps in the same terminal. `unset CESS_CONFIG` selects the bundled defaults for this example;
record a previous custom value yourself before unsetting it.

```bash
CESS_GUIDE_RUN=$(mktemp -d "${TMPDIR:-/tmp}/cess-guide.XXXXXX")
cd "$CESS_GUIDE_RUN"
unset CESS_CONFIG

"$PY" -s -B "$REPO/examples/make_toy_input.py" \
  "$CESS_GUIDE_RUN/toy.hdf5" --sources 4 --invalid-first

# Preview without a manifest: no science product or new checkpoint is created.
cess simulate --input "$CESS_GUIDE_RUN/toy.hdf5" \
  --output "$CESS_GUIDE_RUN/spectra" --seed 1214 --workers 1 --dry-run \
  --morphology --wavelength-error --photometry --emission-lines

# Execute the new run with fresh output and manifest paths.
cess simulate --input "$CESS_GUIDE_RUN/toy.hdf5" \
  --output "$CESS_GUIDE_RUN/spectra" --manifest "$CESS_GUIDE_RUN/run.json" \
  --seed 1214 --workers 1 \
  --morphology --wavelength-error --photometry --emission-lines
```

The standard product is `"$CESS_GUIDE_RUN/spectra/CSST_grism_toy_4.hdf5"`; task status is stored in `run.json`.
The invalid source still occupies its original row, with unavailable values and masks.
Do not delete that row and then compare using the original IDs or row numbers.
For real catalogues, start with a few sources, inspect field definitions and masks, and then increase the workload.

### Handling invalid input sources

Before simulation, CESS checks the parameters used in computation—`RA`, `DEC`, `MAG_Z`, `log(M*)[0,1]_{MAL}`, and
`z_{MAL}`—as well as the SED and `best_fit/z`, for every source. It does not modify the HDF5 input, delete rows, or reorder them.
Invalid sources have `data_mask=2`, nonzero `input_quality_flags`, NaN simulated spectra, and false spectral masks.
Original IDs, row indices, and input parameters passed through to the product retain their original values.

The current `cess.input-quality.v1` policy uses broad numerical safety ranges, all inclusive:
RA 0–360 deg, DEC −90–90 deg, MAG_Z −100–100 AB mag, log10(M*/Msun) −5–20, and each redshift 1e-8–10.
Computational parameters must also be finite. These are engineering safeguards, not galaxy-selection cuts or proof of upstream scientific validity.
`z_{MAL}` need not equal `best_fit/z`. MAG_G/R are passed through; missing values or sentinels such as `-999`
there do not invalidate an otherwise usable SED.

`input_quality_flags` is a uint32 bitmask. Use bitwise operations to read multiple simultaneous reasons:
1=nonfinite computational parameter, 2=RA out of range, 4=DEC out of range, 8=MAG_Z out of range,
16=logarithmic mass out of range, 32=morphology redshift out of range,64=nonfinite/negative SED or a value exceeding the existing 1e100 limit,
128=spectral redshift out of range, 256=insufficient wavelength or LSF support for the three bands.
The field is under `/catalog` in schema 3.0 and at the root in legacy 2.5. Attributes record its version, definitions, and full policy.
Missing columns, inconsistent array shapes, or an invalid shared wavelength grid are file-contract errors and still raise explicit errors.

## 4. Real BayeSED input contract

Inputs must follow the current BayeSED-shaped HDF5 contract. Arbitrary HDF5 files or 2D spectral FITS images cannot be supplied directly.
N is the source count, K the number of parameter columns, and L the length of the shared rest-frame wavelength grid.

| HDF5 path | Shape | Current convention |
| --- | --- | --- |
| `ID` | `(N,)` | Integer, bytes, or string identifiers; dtype/encoding are retained, and IDs do not replace original-row mapping |
| `parameters` | `(N,K)` | Catalogue parameters, in one-to-one correspondence with the names below |
| `parameters_name` | `(K,)` | Bytes/string names; case and special characters must match |
| `best_fit/wavelength_rest` | `(L,)` | Rest-frame wavelength in μm |
| `best_fit/z` | `(N,)` | Dimensionless redshift used to shift spectral wavelengths |
| `best_fit/spec_*` | `(N,L)` | Input fν in μJy; the first dataset whose name starts with `spec_` in the `best_fit` traversal is selected |

Required parameter columns are `RA`, `DEC`, `MAG_G`, `MAG_R`, `MAG_Z`, `log(M*)[0,1]_{MAL}`, and `z_{MAL}`.
RA/DEC are in degrees; magnitudes follow the upstream input definition. The mass column is the upstream logarithmic stellar-mass
parameter: check its solar-mass convention. These names are not arbitrary aliases. `best_fit/z` and the table's `z_{MAL}`
are consumed by different stages; do not discard one because the names appear similar.

Rest-frame wavelengths in μm are converted to Angstrom and multiplied by `(1+z)`.
The SED is already normalized to the observed-flux convention upstream; CESS does not apply an additional distance or redshift flux factor here.
Rest-frame luminosity densities from other software need an independent units/normalization adapter first.
If several `spec_*` datasets exist, verify which model is actually selected rather than inferring it from the filename.

The source axis must align with ID, and the wavelength axis with spectral length.
Sources are sliced by original row before processing, with validity combined across the three grism bands; invalid sources retain their positions.
A successful plan alone does not establish that every array, physical quantity, or coverage requirement is valid.
See the [public generator](../examples/make_toy_input.py) for a structural example. Scientific normalization of a real library remains the responsibility of its source.

## 5. Configuration, resources, and paths

Configuration precedence is explicit `--config`, then a nonempty `CESS_CONFIG` environment variable, then the bundled default JSON.
A relative `--config` path is first located from the invocation's current working directory.
**Relative resource and runtime-directory paths inside an external JSON are resolved from that JSON's directory.**
The bundled JSON is a special case: instrument resources resolve within the package's data directory, while its four input/output
directories resolve from the invocation's working directory, avoiding writes into site-packages.
Relative explicit CLI `--input/--output/--manifest` paths still resolve from the invocation's working directory.

For production, use absolute paths for inputs, outputs, manifests, and external JSON files.
Do not copy the default JSON elsewhere without adjusting its relative resources: `basic_file/...` would then be sought under the new JSON directory.
Keep the complete resource layout, or replace resource fields with real absolute paths.
Missing resources normally fail when actually consumed; an ordinary CLI dry-run does not replace a resource-read check.

Edit a complete configuration template. A short JSON fragment from a demonstration is not a standalone runtime configuration.
Key settings include:

| Parameter/group | Definition and notes |
| --- | --- |
| `radi`, `expt`, `expnum` | Collecting-aperture radius in cm, exposure time per exposure in s, and number of exposures; radius is not diameter |
| `arcsecperpix` | arcsec/pixel; must agree with the scale assumed by the morphology library |
| `dc_value`, `read_noise` | Dark current in e-/s/pixel; single-pixel RMS read noise in e- per read |
| `gu/gv/gi` wavelength fields | `start_wl/end_wl/min_wl/max_wl/dlambda` are in Angstrom; `init_idx/end_idx` are detection indices |
| Background and template LSF | Defaults are a `fixed_template` background and `template_lsf.mode=unknown`; neither certifies a time-dependent sky or detailed LSF |
| Throughput, background, GEMS, and profile libraries | Use the configured paths; photometric and zero-order resources are read according to products/switches |
| `spectral_extraction` | Defaults to `legacy_full`; see Section 12 for advanced modes |

The upstream Gaussian spectral FWHM is currently controlled by `*_dlambda`; the `*_Res` numbers do not enter that operator.
Changing only `*_Res` therefore does not implement the intended resolution change.
The templates' nominal R=300 declaration also does not determine their detailed LSF.
Validation records effective values and provenance; it does not convert historical assumptions into instrument calibration.

Some values in the bundled default JSON are listed below for inspection. They are **current program settings**,
not independently certified CSST specifications. An external configuration may use different values:

| Default setting | Value |
| --- | --- |
| `radi` | 100 cm |
| Spectroscopic `expt × expnum` | 150 s × 4 = 600 s |
| `arcsecperpix` | 0.074 arcsec/pixel |
| `dc_value` / `read_noise` | 0.02 e-/s/pixel / 5 e-/pixel/exposure RMS |
| GU/GV/GI `*_dlambda` | 15 / 21 / 33 Angstrom; upstream Gaussian FWHM, not directly the final output pixel spacing |
| `background_scale` | 1.0, multiplying the fixed background template |

Seven-band photometry and the true zero-order integrals use their own `photometry_exposures` and `zero_order_exposures`.
Do not interpret every product as a 600 s measurement.

## 6. Modes, switches, and batch runs

| CLI option | Default and behavior |
| --- | --- |
| `--mode` | `standard`; alternatively `paired-wce` |
| `--seed` | **0**; accepts nonnegative integers. The example value 1214 is an explicit choice, not the default |
| `--workers` | 1; bounded process count, no greater than the number of runnable tasks |
| `--threads-per-worker` | 1; numeric-library thread request before spawn starts, not CPU affinity |
| `--source-start/--source-stop` | Start 0, stop at the source count by default; half-open original-row interval |
| `--chunk-size` | No chunking by default: the selected interval of each input is one task and one spectral product; otherwise limits source rows per product |
| `--memory-budget-gib` | Unset by default; restricts concurrency using estimated working memory, not an operating-system hard limit |
| Four optional science switches | `--morphology`, `--wavelength-error`, `--photometry`, and `--emission-lines` are all off by default |

`paired-wce` uses the same noise realization for unshifted and wavelength-error branches, which are not independent exposures.
Explicitly enable `--wavelength-error` to study WCE. `--photometry` controls seven-band photometry;
emission-line postprocessing also has its own subcommand. These are runtime switches, not promises that disabling them removes
every resource requirement: existing morphology preparation may still consult a library, for example.

These commands work from any current directory. Replace `/absolute/...` with real paths:

```bash
# Original rows 100 through 119 of a real library: start with a small subset in a new location.
cess simulate --config /absolute/runtime.json --input /absolute/catalogue.hdf5 \
  --output /absolute/new_subset --manifest /absolute/subset.json \
  --source-start 100 --source-stop 120 --chunk-size 10 --workers 1 --seed 1214 \
  --mode standard --morphology --wavelength-error --photometry --emission-lines

# Directory input: two workers process files, assuming memory for two complete tasks.
cess simulate -i /absolute/catalogues -o /absolute/new_results --all-effects -j 2

# Quote the glob for CLI discovery; preview this explicitly chunked batch plan only.
cess simulate --config /absolute/runtime.json --input '/absolute/catalogues/*.hdf5' \
  --output /absolute/new_batch --chunk-size 100 --workers 2 --seed 1214 --dry-run

# Shared-noise WCE comparison, with separate output and manifest paths.
cess simulate --config /absolute/runtime.json --input /absolute/catalogue.hdf5 \
  --output /absolute/new_paired --manifest /absolute/paired.json \
  --source-stop 4 --workers 1 --seed 1214 --mode paired-wce \
  --morphology --wavelength-error --photometry --emission-lines
```

Inputs can be directories or multiple files. Directory discovery includes only direct-child `.hdf5/.h5` files.
Two inputs with the same stem in one batch could generate the same output name, so the CLI requires separate runs.
The selected source interval applies to all inputs, so its stop cannot exceed the length of a shorter file.
For production, one manifest per input avoids rewriting a very large checkpoint after every task.

Reproducible random streams bind the seed, input contents, original row, ID, and random component.
With a fixed seed, changing worker count/chunking should preserve the corresponding source's random realization.
That does not authorize using an old manifest: changing chunks changes the task plan, which resume rejects.
More workers also do not guarantee linear speedup; consider numeric-library threads, memory, I/O, and task size.

Without `--source-start/--source-stop`, all source rows are processed. `-j` controls concurrency among existing tasks;
it does not split a single input automatically. An input without explicit chunking is one computational task.
Use the directory command above to retain one output per input, and add `--chunk-size` only when you need to limit task memory
or process one input in parallel.

### Merge chunks into a complete file

For example, simulate one input in 100-row chunks, then merge them into one HDF5 containing all original rows:

```bash
cess simulate -i /path/to/data/input.hdf5 -o /path/to/new_chunks \
  --all-effects --chunk-size 100 -j 2
cess merge -i /path/to/new_chunks -o /path/to/merged/complete.hdf5 \
  --input-source /path/to/data/input.hdf5 --dry-run
cess merge -i /path/to/new_chunks -o /path/to/merged/complete.hdf5 \
  --input-source /path/to/data/input.hdf5
```

`merge` sorts by original input row, checks chunk compatibility, and writes actual combined data without rerunning simulation.
Copying is streamed; the final HDF5 is self-contained and does not rely on the chunk files.
`--input-source input.hdf5` optionally verifies full input coverage and is recommended when you need that guarantee.
`--dry-run` performs preflight checks only; remove it to write the product.
Keep chunks and merged products in separate directories, so a merged file is not rediscovered as a chunk later.
If a directory contains results for several original inputs, select and merge the chunks for each input separately.

`merge` supports standard and paired-wce simulation products in schema 2.5/3.0, not EL-only postprocessing products.
Shared wavelengths and responses must match, and row ranges must be contiguous and non-overlapping.
`--copy-rows` limits a copying block, not a new simulation chunk. Final line-table integrity checks still read peak tables,
so memory use grows with the number of peaks. Merge preserves the original simulation identity and chunk provenance;
it does not masquerade as a fresh native simulation task, and its output cannot replace a native task's manifest product.

### 6.1 CPU processes and threads

Parallel execution launches multiple Python processes on one host. Each handles a complete task and takes another when it finishes;
the parent also schedules work, verifies outputs, and updates manifests. There is no MPI distribution across nodes or fixed CPU affinity.

| Setting | Controls | Does not mean |
| --- | --- | --- |
| `--workers N` | At most N computational processes; default 1, also limited by task count and memory budget | Automatic allocation of N CPUs, or a process tree restricted to exactly N cores |
| `--threads-per-worker N` | Numeric-library thread request before spawn, default 1 | Every backend will obey it, or a worker is pinned to a physical core |
| `--chunk-size M` | At most M original rows per task, creating more independent tasks | A thread count or one CPU per source |
| `--memory-budget-gib` | Reduce concurrency based on estimated per-task memory | A live free-memory measurement or operating-system hard cap |

**File-level parallelism.** With CESS activated and sufficient memory for two complete inputs, run from any working directory:

```bash
cess simulate -i /absolute/catalogues -o /absolute/new_parallel \
  --all-effects -j 2 --threads-per-worker 1
```

Each worker processes a complete file, then takes the next file. The modern CLI uses `ProcessPoolExecutor` with spawn and sets
the common OpenBLAS, OpenMP, MKL, Accelerate, NumExpr, and BLIS thread environment variables before the processes start;
their default request is 1. It restores the parent's environment after the pool closes.
Manual exports are normally unnecessary, although backend support determines whether a library obeys the request.
A worker is a Python computational process, not necessarily one physical CPU core.

Before execution, the summary reports input/task counts, requested workers, the planned effective concurrency limit,
and per-worker thread settings. Task start/finish records contain PID, source interval, and elapsed time.
Omitting `-j` defaults to 1 and reports sequential execution when several tasks exist.
`--dry-run` retains JSON planning output; inspect `workers`, `threads_per_worker`, and each task's `estimated_peak_gib`.

Choose concurrency using representative small samples and memory planning. With the current default model, an approximately 23000-source
task is estimated at 8.5 GiB, while 76654 sources are estimated at 27.8 GiB. These are estimates, not measured RSS or operating-system limits.
On a 32 GB machine, do not set workers from CPU count alone when large files will overlap.
`--memory-budget-gib` conservatively reduces concurrency using the largest estimated task.
If even one task does not fit, reduce the chunk size or use a host with sufficient memory.
A 1000-row input with chunk size 100 produces 10 tasks; an unchunked file remains one task and uses one computational process even with 8 requested workers.

**Running on Gravity.** Request CPU and memory on one host through PBS first, then set CESS's worker count.
These are separate steps. Assuming 8 CPUs have been allocated and memory is sufficient, set:

```bash
# Set in the submission terminal; configure all other paths/switches as described in the Gravity guide.
export CESS_INPUT=/path/to/catalogues
export CESS_CPUS=8
export CESS_WORKERS=8
unset CESS_CHUNK_SIZE  # A complete task per input; set a positive value only for explicit chunks.
```

`CESS_CPUS` is a user declaration used to check `CESS_WORKERS <= CESS_CPUS`.
**It neither requests resources nor automatically verifies the CPUs actually allocated by PBS.**
Requesting 8 cores but retaining `CESS_WORKERS=1` still launches one computational worker.
Setting `CESS_CPUS=8` without requesting 8 CPUs does not allocate them.
With a thread request of 1, 8 busy workers usually represent about 8 concurrent computations, with extra parent/helper overhead.

The supplied `cess.pbs` deliberately has no preset CPU, memory, or queue request.
If Gravity uses PBS Pro/OpenPBS syntax, a single-host 8-CPU request has this form:

```text
#PBS -l select=1:ncpus=8:mem=<requested_memory>
#PBS -l walltime=12:00:00
```

Replace `<requested_memory>` with an actual, permitted request based on workload estimates.
The 12-hour walltime is a syntax example, not a completion guarantee.
For Torque syntax the CPU portion usually uses `nodes=1:ppn=8`; do not combine both forms.
Choose according to the actual Gravity configuration. The
[official PBS Pro user guide](https://help.altair.com/2024.1.0/PBS%20Professional/PBSUserGuide2024.1.pdf)
documents that syntax; the site's queue, units, and allocation rules remain authoritative.

The scripts accept one HDF5 or an input directory and fix the per-worker numeric-library thread request at 1.
Optionally set `CESS_MEMORY_BUDGET_GIB` to pass a conservative planning budget to the CLI; see the
[Gravity guide](../scripts/gravity/README.md). After configuring all CESS paths, switches, and resources,
run `preflight.sh --dry-run`, then submit `qsub` according to site rules.
Workers are not automatically spread across multiple requested nodes. Request one host per CESS job, or submit independent input/batch
jobs with distinct outputs and manifests.

## 7. Dry-run and strict resume

The three preview forms have different filesystem behavior:
a new `--dry-run` without a manifest only prints a plan;
a new run with `--dry-run --manifest` creates a checkpoint and may create its parent directories/lock, but computes no science product;
an existing run with `--resume --dry-run --manifest ...` performs strictly read-only verification without creating output or rewriting its checkpoint.
After the second form, use `--resume` for actual execution rather than treating the existing checkpoint as new.

Before resuming, confirm that the old workers have stopped. For the Section 3 toy, repeat the original arguments and add resume:

```bash
cess simulate --input "$CESS_GUIDE_RUN/toy.hdf5" \
  --output "$CESS_GUIDE_RUN/spectra" --manifest "$CESS_GUIDE_RUN/run.json" \
  --seed 1214 --workers 1 \
  --morphology --wavelength-error --photometry --emission-lines --resume --dry-run

# After verification, remove --dry-run and keep --resume to execute pending work.
```

Strict identity includes input paths/content, complete effective configuration, switches, seed, mode, source code, resources,
layout, output directory, and source interval/chunk plan. Complete products are checked for structure, semantics, and checksums before being skipped.
Damage or drift raises an error; deleting a manifest does not make this the same run.
Resume requires a manifest, and an old-revision manifest cannot resume new code.

**Workers, `--threads-per-worker`, and `--memory-budget-gib` may change** because these scheduling settings are not in the snapshot.
The scientific task identity must still match completely, and the new budget must fit one task.
Worker count does not alter the source random stream, although different numeric libraries/thread settings can introduce floating-point rounding differences.
Do not simultaneously change chunks, row selection, paths, switches, or scientific configuration; use fresh outputs and a fresh manifest for those changes.
Preview performs planning/verification, not scientific validation.

## 8. Emission-line postprocessing

`el-detect` reads an existing CESS spectrum without rerunning the spectral simulation.
It accepts legacy and structured parents, defaults to 3.0 output, and retains mappings to the original catalogue rows and immediate parent rows.
If the parent has a WCE branch, the corresponding diagnostics are processed too.
The product stores line measurements and provenance without copying all parent spectra; it is not a complete spectral product suitable for every plotting tool.

```bash
cess el-detect --input "$CESS_GUIDE_RUN/spectra/CSST_grism_toy_4.hdf5" \
  --output "$CESS_GUIDE_RUN/lines" --manifest "$CESS_GUIDE_RUN/lines.json" \
  --workers 1 --seed 1214
```

Real runs can add `--config /absolute/runtime.json`, source intervals, and chunk settings.
Selection indices here refer to rows in the parent product, with original catalogue rows retained separately by mapping;
they can differ when the parent is itself a subset. A valid line S/N requires the parent uncertainties and the required validated covariance;
read the relevant quality flags.

Peak detection, matching, missed peaks, false positives, and integrated-window measurements are distinct results.
`detect_el_snr` is the historical local mean pixel S/N; integrated-window flux/error/SNR are in the line-measurement structures.
They do not automatically represent deblended total atomic-line fluxes or confirmed line identities.
A threshold on a historical column is not a substitute for a scientific selection function.

## 9. Reading products and physical quantities

The principal schema 3.0 root groups are `catalog`, `spectra`, `photometry`, `zero_order`, `lines`, and `provenance`.
Groups may be disabled or not applicable; their presence does not imply valid measurements.
The unified reader reduces dependence on physical 2.5/3.0 locations. Do not use an unevaluated HDF5 dataset after leaving its `with` context.

```python
from cess.io.products import open_product, read_product_summary

path = "/absolute/path/simulation.hdf5"
print(read_product_summary(path))
with open_product(path) as product:
    ids = product.catalog["ID"][:]
    table = product.parameter_table(required=("RA", "DEC"))
    ra = table["RA"]
    gv = product.band("GV")
    wave = gv["wave"][:]                         # Angstrom
    flux = gv["flux_ujy_with_noise"][1]           # microJy, the second source
    error = gv["ferr"][1]                        # microJy
    available = gv["spec_mask"][1]
    print(dict(gv["wave"].attrs))
    print(dict(product.provenance.attrs))
    row = product.read_row(1)
```

Use the installed environment, or set `PYTHONPATH=/path/to/CESS/src`, for this Python code as well.
Read large products by row where practical; reduce the full-column ID/RA reads above to the selection you need.
Interpret parameter tables by field name, units, and `axis_order`, not fixed column indices, and do not infer source count from ragged peak-table length.
Use row mappings to locate duplicate IDs, source slices, and EL subsets.

`flux_ujy` is the photon-equivalent bin fν before response/morphology; a response-folded quantity must not be multiplied by throughput again.
`flux_elec` is the expected noise-free electron count; observed flux and `ferr` use the nominal response scale.
Negative noisy values are valid and should not be clipped uniformly to zero. Interpret errors, masks, and within-spectrum covariance together.
The current `snr` is not a universal S/N per Angstrom or per resolution element.

Historical `*_0th` spectral arrays are proxies that apply zero-order efficiency on the first-order dispersion grid.
True zero-order bandpass integration is under `zero_order`, but is still not a 2D zero-order image.
Products differ between standard and paired modes: an empty group in a paired product must not be mistaken for available true zero-order photometry.
Seven-band photometry also separates model, observed, errors, and status. Unavailable, undetected, and invalid are different states.

To explicitly convert an identified 2.5 product to 3.0, use `convert_product(source, destination)`.
It writes a new destination without changing the source or recalculating scientific arrays.
Do not assume that arbitrary older files or files without an explicit schema can be converted directly.
See the [output specification](OUTPUT_LAYOUT.md) for the interface and layout.

## 10. Plotting and stratified sampling

Plotting requires the `plot` extra. Row 0 of the toy is invalid, so the example selects row 1.
`--save` writes a PNG instead of opening a GUI. Showing the input model requires the source BayeSED file to remain locatable;
after moving products, provide `--model-input` explicitly if needed.

```bash
env PYTHONPATH="$REPO/src" "$PY" -s -B -m cess.cli.demo \
  "$CESS_GUIDE_RUN/spectra/CSST_grism_toy_4.hdf5" 1 emissionline \
  --model-input "$CESS_GUIDE_RUN/toy.hdf5" --save "$CESS_GUIDE_RUN/spectrum.png"

cess-extract \
  /absolute/simulation.hdf5 --output /absolute/new_sample.hdf5 \
  --target 100 --n-bins 10 --seed 1214
```

Sampling first requires mean S/N ≥ 3 in both GV and GI, then draws proportionally without replacement across redshift-quantile strata.
`--n-bins` controls redshift strata, not S/N bins. An insufficient eligible population cannot supply the target count.
This retains a historical selection rule, not an integrated-line S/N selection or a completeness guarantee.
The sampling CLI's seed is unset by default: set it explicitly for reproducibility, and do not confuse it with the simulation CLI's default 0.

The export schema is `extracted-mock-spectra-1.1`; it preserves available errors, masks, and provenance but is not an equivalent copy of the full simulation product.
Some toy inputs produce no sources above the sampling thresholds. Inspect the selection instead of changing data merely to force inclusion.

### Interactive HDF5 browsing

After installing the optional `viewer` extra and Tcl/Tk, run `cess-viewer /path/to/product.hdf5 --lang en`.
Its four tabs provide structure/attribute and paginated data inspection, per-source spectra,
X-Y/histogram plots, and help. Select sources by row or ID, and save figures or CSV tables.
A three-band overlay does not calibrate or merge bands. CSV lacks full covariance, masks and provenance.
Exports use the filtered plot snapshot; robust display limits do not remove exported points.
For paired-wce, Spectrum displays the base branch; inspect `*_wce` in Structure or X-Y.
See the [viewer guide](VIEWER.md) for installation, operation, large-file behavior and troubleshooting.

## 11. Troubleshooting

| Symptom | Check first |
| --- | --- |
| `cess` is missing, or an old version is imported | Print `sys.executable` and `cess.__file__`; check Conda/venv activation, the installed entry point, and `PYTHONPATH` |
| `No input HDF5 files match` | Check absolute paths, glob quoting, direct-child discovery, and suffixes; replace placeholder paths |
| Missing parameter/SED fields | Check Section 4 paths, exact column names/case, axes, and whether the first `spec_*` is the intended model |
| Throughput/pickle cannot be found | Check the JSON path base; after copying JSON, preserve resources or use absolute resource paths |
| Manifest/product already exists | Distinguish a new run from resume; new configurations use new locations, while the same task can resume after verification |
| Resume snapshot/plan mismatch | Compare code, input, resources, config, interval, and chunks; do not edit the checkpoint to bypass verification |
| A task exceeds the memory budget | Reduce chunk size first, then consider workers; estimates are not RSS guarantees, so measure a small sample |
| NaN data or empty line tables | Inspect input spectra/coverage, source masks, switches, thresholds, and flags; these do not automatically indicate a code error |
| Plot cannot find the input model | Retain or supply the BayeSED source using the real `--model-input` path |

Keep the full command, working directory, version, config, log, and manifest.
Separate path/input-contract errors, implementation defects, and model limitations.
Do not delete invalid sources or replace undocumented physical defaults just to obtain an expected plot.
If a result cannot be reproduced, narrow the run to an explicit source interval and compare the corresponding input/output fields and provenance.

## 12. Advanced morphology and Gravity

`mge_strip` is an explicitly selected joint spatial-strip model requiring a verified positive source-basis bundle and declared PSF/LSF roles.
Its source fraction F enters source counts once; the N spatial pixels determine the sky/dark/read-noise area.
Finite dispersion-window mass M is not renormalized to 1. Disabling morphology redistribution does not disable this aperture model's F/N.

Follow the [1D science-model description](SCIENCE_MODELS.md), [MGE model guide](MGE_MODELS.md), and
[MGE bundle specification](MGE_BUNDLE_FORMAT.md), and merge the settings into a complete configuration.
Example spatial-pixel counts, kernel radii, and `pre_psf` are assumptions to verify, not calibrated CSST values.
Library scale, PSF declaration, and runtime config must agree; legacy pickle bundles and positive libraries are not interchangeable.

The Gravity tools are described in [scripts/gravity/README.md](../scripts/gravity/README.md), using `preflight.sh` and `cess.pbs`.
They require absolute paths, version identity, bounded workers/threads, separate logs, and explicit resume.
New-run preflight omits the manifest to avoid creating a checkpoint. PBS flavor, queue, and CPU/memory syntax must be confirmed on the actual server,
and the user submits `qsub`. Local toy-payload verification does not establish server environment, resource, or full-catalogue acceptance.

Source and documentation use the [MIT license](../LICENSE). Bundled resources and provenance are described in the
[resource notice](../RESOURCE_NOTICE.md) and [resource inventory](RESOURCE_MANIFEST.json).

For short commands see the [quick start](QUICKSTART.md); for Bash syntax and step-by-step commands see the [command guide](COMMANDS.md).
`-i/-o/-j` are aliases of `--input/--output/--workers`, and `simulate --all-effects` enables all four optional science switches without changing their individual defaults.
