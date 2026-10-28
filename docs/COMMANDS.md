# CESS 0.9.0: Bash Command Guide

[English](COMMANDS.md) | [简体中文](COMMANDS_zh-CN.md)

This guide is for users who are new to the CESS command line. It covers Bash on macOS and Linux; the common single-line commands also work in zsh.
Scientific quantities, input HDF5 fields, and model assumptions are described in the [user guide](USER_GUIDE.md). This page explains how to compose and run commands.

## 1. The three most common ways to run CESS

First activate an environment in which CESS is already installed:

```bash
conda activate cess
```

Every `/path/to/...` below is a placeholder. Replace it with a real path on your machine before running the command. With absolute paths, these commands can be run from any working directory.
Each example is an independent run. Choose a new output location that does not already contain products with the same names.

**One input file, one complete result:**

```bash
cess simulate -i /path/to/data/input.hdf5 -o /path/to/results/run001 --all-effects
```

**One directory, processing all its inputs in sequence:**

```bash
cess simulate -i /path/to/data/catalogues -o /path/to/results/batch001 --all-effects
```

**One directory, processing different files with two worker processes:**

```bash
cess simulate -i /path/to/data/catalogues -o /path/to/results/parallel001 --all-effects -j 2
```

Without `--source-start/--source-stop`, CESS processes all sources. Without `--chunk-size`, **each input file produces one complete spectral HDF5 file**.
`-j 2` requires enough memory for two tasks at the same time. The default, `-j 1`, processes directory inputs sequentially.
An input that has not been split into chunks is one task. Changing its `-j` value to 8 does not automatically distribute that file across eight processes.

| Argument | Meaning |
| --- | --- |
| `simulate` | Generate simulated one-dimensional spectra from a BayeSED-shaped HDF5 input |
| `-i` / `--input` | Input files, directories, or quoted globs; multiple inputs may follow one `-i` |
| `-o` / `--output` | Output directory for simulation and postprocessing; for `merge`, it is a specific output file |
| `--all-effects` | Enable all four optional switches: morphology, wavelength-calibration error, photometry, and emission-line analysis |
| `-j` / `--workers` | Maximum number of concurrent worker processes; default 1 |
| `--threads-per-worker` | Native numerical-library thread request for each new worker; default 1 |

When `cess simulate ...` is run without `--all-effects`, the four scientific switches are off by default.
`--all-effects` does not enable every possible instrument model. For example, it does not automatically select MGE extraction or establish a known template LSF.

## 2. Install, activate, and check the version

Full dependency and platform information is in the [installation guide](INSTALL.md). If Conda is available, install from the source root:

```bash
conda create -n cess python=3.12 pip
conda activate cess
cd /path/to/CESS
python -m pip install '.[tools,plot]'
python -m pip check
```

`/path/to/CESS` must be the source root containing `pyproject.toml` and `setup.cfg`.
The dot in `.[tools,plot]` means the current directory. Quoting makes Bash/zsh pass the entire argument to pip unchanged.
This installs CESS, its core dependencies, and the sampling and plotting dependencies. It installs the local project rather than looking up a similarly named package on PyPI.
To install a downloaded local wheel in the same environment:

```bash
python -m pip install /path/to/downloads/cess-0.9.0-py3-none-any.whl
```

This wheel command requests only the core dependencies. For plotting or sampling, install the corresponding extras as explained in the installation guide.
If an environment named `cess` already exists with a suitable Python version, activate it directly; do not run `conda create` again.
If you use venv, create it as described in the installation guide, then activate it with:

```bash
source /path/to/cess-env/bin/activate
```

Choose either Conda or venv. Activate the chosen environment in each new terminal. A venv does not appear in `conda env list`.
After activation, check:

```bash
python --version
python -c 'import sys, cess; print(sys.executable); print(cess.__version__); print(cess.__file__)'
cess --version
cess --help
cess simulate --help
cess merge --help
```

Python should be 3.12.x. `cess --version` reports the version used by the command. The Python check also reports the interpreter and import path,
which lets you confirm that both belong to the selected environment. Resolve unexpected versions or paths before continuing.
To run source code directly instead of the installed entry point, use the following from any directory. The interpreter must still have the CESS dependencies installed:

```bash
env PYTHONPATH=/path/to/CESS/src python -s -B -m cess --help
```

## 3. Bash paths, quotes, and line continuation

### 3.1 Know your working directory

```bash
pwd
cd /path/to/data
ls
```

`pwd` shows the working directory, `cd` changes it, and `ls` lists its contents. If the working directory is `/path/to/data`, then:

```bash
cess simulate -i input.hdf5 -o results --all-effects
```

reads `/path/to/data/input.hdf5` and writes to the directory `/path/to/data/results`.
You do not have to run this command from the CESS source directory. With the installed package, the source and data directories can be separate.

### 3.2 A multiline command is still one command

The following is equivalent to the single-line example in Section 1:

```bash
cess simulate \
  -i /path/to/data/input.hdf5 \
  -o /path/to/results/run002 \
  --all-effects
```

- A final `\` continues the command on the next line. No spaces or comments may follow that backslash.
- Do not add a continuation character to the final line. Do not paste the multiline example as one line containing `\   -i`. To use one line, remove the continuation characters and line breaks.
- Do not copy terminal prompts such as `$`, `%`, or `(cess)`. The code blocks already omit them.
- A `>` continuation prompt usually means a backslash, quote, or bracket is still open. Press Ctrl+C to cancel, then paste the complete command.

### 3.3 Paths with spaces and ordinary variables

Quote an entire path if it contains spaces:

```bash
cess simulate -i "/path/to/my data/input.hdf5" -o "/path/to/my results/run001" --all-effects
```

You can also define your own variables in the same terminal. There must be no spaces around the assignment `=`. Keep double quotes when using the variables:

```bash
CESS_DOC_INPUT="/path/to/my data/input.hdf5"
CESS_DOC_OUTPUT="/path/to/my results/run002"
cess simulate -i "$CESS_DOC_INPUT" -o "$CESS_DOC_OUTPUT" --all-effects
```

Ordinary variables do not need `export` here because the shell expands them into command arguments first.
Do not use system variables such as `HOME` or `PATH` to store task paths. For a filename that starts with `-`, using its absolute path avoids interpreting it as an option.

## 4. Directories, multiple files, and globs

Directory input discovers only `.hdf5` and `.h5` files directly inside that directory. It does not search subdirectories recursively.
List multiple inputs after the same `-i`, separated by spaces. Do not use commas or a Python list:

```bash
cess simulate -i /path/to/data/a.hdf5 /path/to/data/b.hdf5 \
  -o /path/to/results/two_files --all-effects -j 2
```

`-i` accepts input values until the next option, such as `-o` or `-j`.
To select filenames by a pattern, quote the entire glob so CESS resolves it. This avoids depending on the different Bash/zsh handling of unmatched patterns:

```bash
cess simulate -i '/path/to/data/catalogues/*_MzLS_*.hdf5' \
  -o /path/to/results/mzls_batch --all-effects -j 2
```

If the pattern contains a variable, use double quotes to expand the variable while preserving the asterisk:

```bash
CESS_DOC_INPUT_DIR=/path/to/data/catalogues
cess simulate -i "${CESS_DOC_INPUT_DIR}/*_MzLS_*.hdf5" \
  -o /path/to/results/mzls_batch_002 --all-effects -j 2
```

Distinct inputs must have different stems, meaning their filenames without extensions. For example, `field.h5` and `field.hdf5` could produce the same output name,
so process them in separate batches. Each input file retains its own spectral result; different input catalogues are not combined into one large catalogue.

## 5. Preview the plan, then choose workers and memory

For a read-only preview of a new run, omit the manifest:

```bash
cess simulate -i /path/to/data/catalogues -o /path/to/results/planned_run \
  --all-effects -j 4 --dry-run
```

The plan fields `input_files`, `task_count`, `workers`, `threads_per_worker`, and each task's `estimated_peak_gib`
report the input count, task count, planned concurrency limit, thread request, and estimated memory. Successful planning means that the required metadata was read and tasks were constructed;
it does not mean that all SED values, resources, or scientific quantities were validated. An ordinary CLI preview does not load every runtime resource.

With enough memory and four allocated CPU execution slots, remove `--dry-run` to run the calculation. The pool uses at most the smaller of the requested worker count and the task count.
A memory budget may reduce concurrency further:

```bash
cess simulate -i /path/to/data/catalogues -o /path/to/results/budgeted_run \
  --all-effects -j 4 --threads-per-worker 1 --memory-budget-gib 32
```

The 32 GiB value only illustrates the budget syntax. Adjust it for your machine or allocated job resources.
Leave memory for the operating system, parent process, and other applications rather than entering the machine's entire RAM capacity.
The current budget conservatively uses the largest task's memory estimate to determine how many workers fit. It is neither real-time memory monitoring nor an operating-system hard limit.
If even one task does not fit, the command refuses to run; use smaller chunks or a host with enough memory.

Each worker requests one native numerical-library thread by default. You usually do not need to export BLAS/OpenMP settings manually.
These settings are applied before computational processes are spawned; they do not pin processes to physical CPUs or allocate server resources.
Concurrent large files add their memory requirements, so choose concurrency using more than the CPU count alone.
File-level parallelism reduces total batch time. One large, unchunked file is still processed by one worker.
Section 9 covers parallelizing one file; Section 13 covers server jobs.

## 6. Source-row intervals, random seeds, and output layout

Select original rows 100–119 from one file, for 20 rows in total:

```bash
cess simulate -i /path/to/data/input.hdf5 -o /path/to/results/subset \
  --all-effects --source-start 100 --source-stop 120 --seed 1214
```

Rows are zero-based. The start is inclusive and the stop is exclusive; `--source-stop 120` does not mean “process 120 sources.”
The default interval starts at 0 and ends at the file's source count. With directory input, the same interval applies to every file and must not exceed the source count of any shorter input.
The default is `--seed 0`. A fixed seed supports reproducibility. For the same input, per-source random streams do not change with worker count or chunking,
although dependencies, platforms, or numerical-library thread backends may introduce floating-point rounding differences.

The modern CLI defaults to schema 3.0. If downstream software requires the old on-disk layout, explicitly select legacy and use a separate output location:

```bash
cess simulate -i /path/to/data/input.hdf5 -o /path/to/results/legacy_run \
  --all-effects --output-layout legacy
```

`legacy` means schema 2.5 output. It does not select an old Python environment or restore old scientific algorithms.
Output filenames include the input stem and selected source count; sliced products also include the original row interval. Each spectral file contains the corresponding sources in all three bands.
The source count, software version, and HDF5 schema are different concepts. Use the [unified product interface](OUTPUT_LAYOUT.md) to read products.

## 7. Scientific switches, WCE comparisons, and external configuration

To enable only selected optional features, list their switches explicitly. For example, request morphology and photometry only:

```bash
cess simulate -i /path/to/data/input.hdf5 -o /path/to/results/selected_effects \
  --morphology --photometry
```

The following creates a shared-noise comparison with and without WCE and enables all four optional switches:

```bash
cess simulate -i /path/to/data/input.hdf5 -o /path/to/results/paired_run \
  --all-effects --mode paired-wce --seed 1214
```

The default mode is `standard`. The two branches share the same noise realization and cannot be added as independent exposures.
`--mode paired-wce` does not replace `--wavelength-error`; `--all-effects` already includes that switch.

Pass an existing complete configuration explicitly:

```bash
cess simulate --config /path/to/config/emulator_parameters.json \
  -i /path/to/data/input.hdf5 -o /path/to/results/custom_config --all-effects
```

Configuration precedence is `--config`, then a nonempty `CESS_CONFIG` environment variable, then the bundled default JSON.
Relative resource paths and input/output directories inside an external JSON resolve against that JSON's directory.
Relative CLI arguments still resolve against the working directory of the invocation. Bundled default resources are located within the package, while its default runtime directories resolve against the current working directory.
After copying the default JSON elsewhere, adjust its resource paths. A few incomplete JSON fields are not a complete instrument configuration.
Configuration fields and assumptions are explained in the [user guide](USER_GUIDE.md) and [scientific model description](SCIENCE_MODELS.md).

To create a complete editable configuration, export the bundled defaults from the installed environment. The following is a Bash here-document that invokes Python.
The text between the `PY` markers is passed to Python. The closing `PY` must be alone on its line, with no indentation before it:

```bash
python - <<'PY'
import json
from pathlib import Path
from cess.config.loader import DEFAULT_CONFIG_PATH, load_config

destination = Path('/path/to/config/emulator_parameters.json')
destination.parent.mkdir(parents=True, exist_ok=True)
config = load_config(DEFAULT_CONFIG_PATH)
config.pop('_config_path', None)
config.pop('_config_dir', None)
with destination.open('x', encoding='utf-8') as stream:
    json.dump(config, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(destination)
PY
```

This refuses to overwrite an existing destination. Resource paths in the exported configuration point to this environment's installation, and runtime directories have also been resolved to absolute paths.
Regenerate or update those paths before using the configuration on another machine or in another environment. When editing exposure, noise, aperture, or other parameters, check their units and scientific definitions,
and use new output and manifest locations for the modified configuration.

If `CESS_CONFIG` was set previously, inspect it. To stop using it, record its old value and then unset it:

```bash
printf '%s\n' "${CESS_CONFIG:-not set}"
unset CESS_CONFIG
```

## 8. New runs, manifests, and strict resume

To recover after interruption, specify a new manifest from the first run:

```bash
cess simulate -i /path/to/data/input.hdf5 -o /path/to/results/resumable \
  --all-effects --seed 1214 --manifest /path/to/manifests/run001.json
```

The manifest records the complete task plan and per-task completion state. Without chunking, one complete input file is one task;
the program cannot resume automatically at an arbitrary row midway through that file. For finer recovery granularity, explicitly configure chunks before the initial run.
Before resuming, confirm that the previous workers have stopped and retain the original inputs, outputs, configuration, and scientific settings.

**First perform a read-only check of the same run:**

```bash
cess simulate -i /path/to/data/input.hdf5 -o /path/to/results/resumable \
  --all-effects --seed 1214 --manifest /path/to/manifests/run001.json --resume --dry-run
```

**After verification succeeds, execute the remaining tasks:**

```bash
cess simulate -i /path/to/data/input.hdf5 -o /path/to/results/resumable \
  --all-effects --seed 1214 --manifest /path/to/manifests/run001.json --resume
```

| Command combination | Filesystem behavior |
| --- | --- |
| New run with `--dry-run`, without `--manifest` | Print a plan without creating scientific products or a new manifest |
| New run with `--dry-run --manifest new.json` | Create a pending manifest and required parent directories/lock, without calculating spectra |
| Existing run with `--resume --dry-run --manifest existing.json` | Verify read-only without changing the manifest or products |

If the second combination has already created a manifest, add `--resume` when executing the run.
Strict identity checks cover input contents and paths, complete configuration, resources, source code, seed, mode, switches, layout, output location, source interval, and chunk plan.
After a source-code upgrade, an old development manifest cannot be resumed as the same 0.9.0 run. Existing complete products remain readable according to their schemas;
they do not need to be recalculated solely to change a software version label.
Worker count, thread requests, and memory budget are scheduling settings and may be adjusted. They do not bypass scientific task identity checks.
Investigate a damaged manifest or identity mismatch. Use new run locations for new settings rather than editing a manifest to bypass checks.
Do not launch two concurrent invocations using the same manifest. For very large directories, use batches or a separate manifest per input file to avoid repeatedly rewriting one huge JSON file.

## 9. Parallelize one file, then produce one complete result

A large input can be split into tasks of 1,000 rows, with at most two concurrent tasks:

```bash
cess simulate -i /path/to/data/input.hdf5 -o /path/to/results/input_chunks \
  --all-effects --chunk-size 1000 -j 2 --seed 1214 \
  --manifest /path/to/manifests/input_chunks.json
```

The chunk size of 1,000 is an example; adjust it for memory and runtime. This still processes all rows, but creates multiple chunk HDF5 files.
After calculation finishes, select only chunks from **the same input and simulation run**. Preview the merge first, then write a new complete file:

```bash
cess merge -i /path/to/results/input_chunks -o /path/to/results/merged/complete.hdf5 \
  --input-source /path/to/data/input.hdf5 --dry-run

cess merge -i /path/to/results/input_chunks -o /path/to/results/merged/complete.hdf5 \
  --input-source /path/to/data/input.hdf5
```

For `merge`, `-o` is a file path; for `simulate`, it is a directory.
Merging does not simulate spectra again. It orders chunks by original source row, verifies configuration, seed, version, shared wavelengths/responses, and continuous coverage, and remaps emission-line table source-row references.
`--input-source` additionally verifies the original input SHA, row-by-row IDs, and full row coverage; keeping this option is recommended.
Chunks must be continuous, unique, and nonoverlapping. Do not mix inputs, runs, or schemas.
Standard and paired-WCE simulation products can be merged. EL-only postprocessing products are outside this feature's scope.

The result is a self-contained HDF5 file and does not read the original chunks through external links or VDS. CESS does not delete the chunks; inspect the complete product before deciding how to manage their storage.
Put the output outside the chunk directory to avoid including it as an input to a later merge.
`--copy-rows 128` is the default copy block size, not the simulation chunk size. Final line-table verification still requires memory that grows with the peak count.
`merge` has no manifest/resume options, and a merged file cannot substitute for a native simulation task product in a manifest.

## 10. Run emission-line postprocessing on existing results

```bash
cess el-detect -i /path/to/results/simulation.hdf5 -o /path/to/results/line_analysis \
  --seed 1214 --manifest /path/to/manifests/line_analysis.json
```

The input is an existing CESS simulation product, not a raw BayeSED library. Schema 2.5 and 3.0 parent products are supported.
This step saves line measurements, quality information, and parent provenance. It neither copies the full parent spectral arrays nor generates noise again.
Subset indices refer to parent-product rows; a separate mapping retains original catalogue rows. A valid line S/N requires the parent uncertainties and the required validated covariance.
The worker, interval, chunk, dry-run, and strict-resume options work similarly to simulation. `--all-effects` belongs to `simulate`, so do not pass it to `el-detect`.

## 11. Logs and exit status for long runs

The simplest logging method sends standard output and standard error to a new log. Run this from any directory:

```bash
mkdir -p /path/to/logs
cess simulate -i /path/to/data/catalogues -o /path/to/results/logged_batch \
  --all-effects -j 2 --manifest /path/to/manifests/logged_batch.json \
  > /path/to/logs/logged_batch.log 2>&1
echo "$?"
```

`>` creates or truncates the log. Choose a new log name, and use a separate new log when resuming.
`2>&1` adds standard error to that file. A log that does not immediately update does not necessarily mean a task has stalled; also inspect start/completion messages, PIDs, and process status.
In another terminal, view the last lines or follow the log:

```bash
tail -n 40 /path/to/logs/logged_batch.log
tail -f /path/to/logs/logged_batch.log
```

Ctrl+C during `tail -f` stops the viewer, not CESS running in the other terminal.
`echo "$?"` must immediately follow the CESS command. If another command intervenes, it reports that command's status instead.
Normally 0 indicates success, 1 indicates one or more task failures, and 2 indicates an argument or planning error. Also inspect the log and manifest.
A warning is not automatically a failure, and a successful command does not establish scientific correctness. For example, the unknown-template-LSF warning retains an explicit physical limitation.

### Display and save a log with an optional Bash script

Successful `tee` execution at the end of a pipeline does not establish that CESS succeeded. If you need this logging style, save the following as `run_cess.sh`,
replace the paths, and run `bash run_cess.sh` from the script's directory in a terminal with the CESS environment activated:

```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p /path/to/logs
cess simulate -i /path/to/data/catalogues -o /path/to/results/tee_batch \
  --all-effects -j 2 --manifest /path/to/manifests/tee_batch.json \
  2>&1 | tee /path/to/logs/tee_batch.log
```

`pipefail` prevents a successful `tee` from hiding an earlier pipeline failure. `set -e` stops this Bash script on failure.
Use a Bash script explicitly, rather than mixing Bash's `PIPESTATUS` with zsh's different `pipestatus` syntax.
Redirection or `tee` does not provide resume support; specify a manifest from the initial run if you need it.
Do not assume that an ordinary foreground command will continue after its terminal is closed. Use the PBS workflow in the next section for server jobs.

## 12. Existing results, invalid sources, and common errors

CESS refuses to overwrite scientific products with the same name, including existing target paths or symbolic links. The output directory may already exist if the products planned for this invocation do not.
A first-run manifest must also use a new path. For new settings, choose new output and manifest locations. To recover the same run, use its original manifest with `--resume`.
If a parallel directory run fails, other files may already have completed. Reconcile the manifest rather than assuming that refusal to overwrite means no tasks ran.

| Symptom | Action |
| --- | --- |
| `command not found: cess` | Activate the environment and confirm CESS is installed there; inspect `python` and `cess.__file__` |
| The terminal keeps showing a continuation prompt | Check quotes, brackets, and final backslashes; cancel with Ctrl+C and enter the complete command |
| Input not found | Check the real absolute path, glob quoting, and direct directory children; replace `/path/to` placeholders |
| Two inputs have the same stem | Use separate batches or output locations to avoid filename collisions |
| Resource not found | Check the external JSON path base; distinguish bundled resources from a copied configuration |
| Multiple files still run sequentially | Pass `-j N` and inspect the effective worker limit; the task count or budget may reduce concurrency |
| One file with `-j 8` still creates one task | Use multiple file inputs, or explicitly chunk the file and merge afterward |
| The budget rejects a single task | Reduce the chunk size or use a host with enough memory; reducing worker count cannot shrink an oversized complete task |
| Output or manifest already exists | Distinguish a new run from resume; use new locations for a new run or resume the original run with its original settings |
| Snapshot or plan mismatch | Check input, config, code, resources, seed, mode, layout, paths, and chunk identity |
| Spectra contain NaNs or line tables are empty | Check switches, source flags, wavelength coverage, and thresholds before interpreting this as a crash |

Invalid sources retain their original IDs, row indices, and order. They are marked by `data_mask=2`, nonzero `input_quality_flags`, and unavailable simulation arrays.
CESS does not repair the input HDF5 or silently remove those rows to make a run finish. Negative fluxes in low-S/N observations may be valid and should not all be clipped to zero.
Flag definitions, output units, and reader examples are in the [user guide](USER_GUIDE.md) and [output description](OUTPUT_LAYOUT.md).

## 13. Gravity / PBS jobs

For large catalogue campaigns, use the [Gravity job arrays](../scripts/gravity/ARRAY.md).
MzLS and DECaLS receive separate memory requests; each input has a one-CPU job and its own resume manifest.
Activate the Python 3.12 `cess` environment on the server and run from the source root:

```bash
python -s -B scripts/gravity/catalogue_array.py prepare \
  -i /path/to/CESS_input --run-dir /path/to/runs/cess_v090_all
```

The defaults require 1,385 files of each survey and generate `MzLS.pbs` and `DECaLS.pbs` without submitting.
See the linked guide for full-file pilots, `qsub`, concurrency limits, and recovery.
The `cess.pbs` entry point below is an alternative for a custom single-node batch; do not run both schemes on the same inputs at once.

CESS `-j` controls process use; it does not request server CPU or memory resources.
Request single-node resources according to the site's PBS rules, then configure matching CESS settings.
For example, after receiving four CPUs on one node and checking that memory is sufficient, set all required absolute paths and scientific switches as described in the [Gravity guide](../scripts/gravity/README.md), then use:

```bash
export CESS_INPUT=/path/to/data/catalogues
export CESS_OUTPUT=/path/to/results/gravity_batch
export CESS_MANIFEST=/path/to/manifests/gravity_batch.json
export CESS_CPUS=4 CESS_WORKERS=4
unset CESS_SOURCE_STOP CESS_CHUNK_SIZE
bash "$CESS_REPO/scripts/gravity/preflight.sh" --dry-run
```

Other required variables, including `CESS_REPO`, must already be configured following the Gravity guide. This fragment alone does not deploy the software.
An unset or empty `CESS_CHUNK_SIZE` means complete-file tasks; a positive integer enables chunks.
The scripts request one numerical-library thread per worker, check that the worker count does not exceed the declared CPU limit, and support an optional estimated memory budget.
`CESS_CPUS=4` does not itself allocate or automatically verify four CPUs.

PBS flavour, queues, memory, and walltime syntax must follow site rules. After configuring the resource header and completing preflight, submit according to those rules.
Local script validation is not Gravity scheduler acceptance. A CESS process pool runs on one host and has no MPI multi-node distribution.
For multi-node production, divide inputs into independent batch jobs, each with its own output directory and manifest.

### Optional desktop viewer

Run the installation command from the CESS source root in the activated environment.
The viewer is optional. Tcl/Tk comes from Python/the operating-system environment, not pip;
with Conda, use `conda install tk` if needed and check `python -m tkinter`.
Opening an HDF5 file does not run a simulation.

```bash
python -m pip install '.[viewer]'
cess-viewer /path/to/product.hdf5 --lang en
# Equivalent installed-package entry point:
python -m cess.viewer /path/to/product.hdf5 --lang zh
# Start an empty window and then choose a file:
cess-viewer
```

The repository script is `python /path/to/CESS/cess_h5_viewer.py /path/to/product.hdf5`.
See the [viewer guide](VIEWER.md) for row/ID selection, filters, exports and headless-server limits.

## 14. What to check after a run

Retain the command, working directory, software version, inputs/configuration, log, and manifest. Use the [unified reader](OUTPUT_LAYOUT.md) to check source counts, IDs/original row mappings, validity flags, and the scientific fields you need.
If photometry or line analysis was enabled, distinguish model from observed quantities, detected from undetected states, and both from invalid data.
Read parameter tables by field name and unit. Model assumptions are explained in the [scientific model guide](SCIENCE_MODELS.md),
and advanced MGE extraction in the [morphology guide](MGE_MODELS.md) and [bundle format](MGE_BUNDLE_FORMAT.md).
Successful command execution means the program completed the requested task; it does not replace scientific validation of the instrument PSF/LSF, background, sensitivity, or sample selection function.
