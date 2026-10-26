# Gravity / PBS Guide

[English](README.md) | [简体中文](README_zh-CN.md)

This directory provides one-dimensional computation scripts. No Gravity login, Linux environment installation, job submission, or site PBS acceptance has been performed as part of this delivery. Preflight expects `0.9.0 / 20261024-D4`; deploy the same source version and repeat preflight. A Mac `.envs/cess` directory or ARM64 dependency lock cannot be used directly as a Linux environment.

`preflight.sh` performs read-only checks of Python 3.12, imported source location, version/revision, configuration, and resource readability, then invokes the actual CLI dry-run. It does not validate instrument science models or establish that every input row is valid. `cess.pbs` executes the simulation, creates a new manifest for an initial run, retains logs and the resume manifest on failure, and propagates failure status. The CLI refuses to overwrite existing products.

All `/path/to/...` values below are placeholders. Replace interpreter, repository, configuration, catalogue, queue, and resource settings with actual available paths and allocated resources. These scripts do not include SSH, passwords, dependency installation, or automatic qsub submission.

```bash
# Set actual absolute server paths; use real local paths for a local toy preflight.
export CESS_PYTHON=/path/to/cess-env/bin/python
export CESS_REPO=/path/to/CESS
export CESS_CONFIG=/path/to/config/emulator_parameters.json
export CESS_INPUT=/path/to/data/catalogue_001.hdf5
export CESS_OUTPUT=/path/to/results/run001/catalogue_001
export CESS_MANIFEST=/path/to/manifests/run001/catalogue_001.json
export CESS_EXPECTED_VERSION=0.9.0 CESS_EXPECTED_REVISION=20261024-D4
export CESS_CPUS=1 CESS_WORKERS=1 CESS_SEED=1214
unset CESS_CHUNK_SIZE CESS_MEMORY_BUDGET_GIB
export CESS_MODE=standard CESS_OUTPUT_LAYOUT=structured CESS_RESUME=0
export CESS_MORPHOLOGY=1 CESS_WAVELENGTH_ERROR=1 CESS_PHOTOMETRY=1 CESS_EMISSION_LINES=1
export CESS_SOURCE_START=0 CESS_SOURCE_STOP=4  # Initially check/compute four rows; stop is exclusive.

bash -n "$CESS_REPO/scripts/gravity/preflight.sh"
bash -n "$CESS_REPO/scripts/gravity/cess.pbs"
bash "$CESS_REPO/scripts/gravity/preflight.sh" --help
bash "$CESS_REPO/scripts/gravity/preflight.sh" --dry-run
```

All commands use Bash arrays to preserve path boundaries, without `eval`, and support Bash 3.2. Preflight and computation set the BLAS/OpenMP/NumExpr and related thread requests to 1 and explicitly pass `--threads-per-worker 1` to the CLI. The default is one worker. `CESS_CPUS` is the user's declared allocated CPU limit; it does not request scheduler resources. `CESS_WORKERS` must not exceed it.

When `CESS_CHUNK_SIZE` is unset or empty, each input file is one task and produces one complete HDF5. Explicitly setting a positive integer, such as `CESS_CHUNK_SIZE=100`, enables original-source-row chunks. Optional `CESS_MEMORY_BUDGET_GIB` must be positive and finite; it forwards an estimated memory budget to the CLI to reduce concurrency. It is not a measurement of available memory, an operating-system hard limit, or an automatic scheduler resource request.

Both the preflight CLI and PBS computation run from `CESS_REPO`, keeping bundled-default runtime path resolution and resume snapshots consistent. The directory from which you invoke preflight does not change this convention.

Choose `standard` or `paired-wce`. Each scientific switch accepts `0/1`; all four default to 0. The default structured layout is schema 3.0; legacy is 2.5. `CESS_INPUT` accepts one HDF5 or a directory. Directory discovery covers only direct `.hdf5/.h5` children, whose stems must be distinct within a batch. Unset `CESS_SOURCE_STOP` to process all remaining rows in the selected files. A directory-wide explicit source interval applies to every file and must not exceed the shortest input's row count. After a small test succeeds, use new output/manifest locations for a full run. Relative instrument/library paths in the configuration resolve against the JSON directory rather than the qsub working directory.

For example, after requesting four CPUs on one node and confirming sufficient memory for concurrent whole-file tasks, process a directory with four workers. Each worker processes one input at a time and takes the next file after finishing:

```bash
export CESS_INPUT=/path/to/data/input_catalogues
export CESS_OUTPUT=/path/to/results/parallel_run
export CESS_MANIFEST=/path/to/manifests/parallel_run.json
export CESS_CPUS=4 CESS_WORKERS=4
export CESS_SOURCE_START=0
unset CESS_SOURCE_STOP CESS_CHUNK_SIZE
# Optional: leave memory headroom within the allocated resources, then set a budget, for example:
# export CESS_MEMORY_BUDGET_GIB=96
bash "$CESS_REPO/scripts/gravity/preflight.sh" --dry-run
```

The final worker count is limited by task count and the optional memory budget. A directory with two files provides only two tasks even if four workers are requested; one unchunked file provides one task. Workers are not pinned to physical CPUs, and there is no MPI multi-node distribution. More workers generally reduce total batch time without guaranteeing a shorter runtime for an individual file. I/O, memory, and the remaining computation of larger files limit speedup.

A directory batch shares one dedicated manifest. The current implementation rewrites the complete JSON after each task, so very large input sets should be divided into directories/batches or use unique per-file outputs/manifests. Do not let multiple jobs compete for the same manifest or output names.

A new-run preview intentionally omits `--manifest` because the CLI creates a manifest for a new `--dry-run --manifest` invocation. This preview creates no directories, manifests, or scientific products. It is not equivalent to every output-conflict/input-hash check performed during execution; the actual CLI still checks and refuses overwrites. The preview reads catalogue metadata and checks resource readability without loading all scientific inputs/resources.

After confirming the server's PBS flavour, available queues, memory, and walltime syntax, add **one** resource-request style to the `cess.pbs` header. Replace the angle-bracket placeholders. Do not combine the two examples or submit them unchanged:

```text
# Torque-style example; confirm with the site:
#PBS -q <queue>
#PBS -l nodes=1:ppn=<allocated_cpus>
#PBS -l mem=<memory>,walltime=<HH:MM:SS>

# PBS Pro/OpenPBS-style example; confirm with the site:
#PBS -q <queue>
#PBS -l select=1:ncpus=<allocated_cpus>:mem=<memory>
#PBS -l walltime=<HH:MM:SS>
```

After confirming site syntax, the user submits with `qsub`. For example, if the site permits exporting the submission environment:

```bash
qsub -V "$CESS_REPO/scripts/gravity/cess.pbs"
```

`-V` explicitly exports the current environment. If the site prohibits that or requires selected variables only, export the CESS variables using its approved method. The script does not call qsub automatically. Logs default to `logs/` under the manifest's parent directory; an absolute `CESS_LOG_DIR` may override it. Each invocation creates a separate log and prints its path first in the PBS output. CPU, memory, and walltime remain subject to scheduler enforcement; these templates do not certify site operation or a particular allocation.

Before resume, confirm that the old processes have stopped and retain the original input/config/resources/source/seed/mode/switches/interval/chunk/output/manifest. Set `CESS_RESUME=1`, repeat the preflight dry-run, then submit with qsub. This preview strictly verifies an existing manifest read-only. Do not reuse an old manifest across code/resource/schema identities; choose new run locations. Never submit two concurrent jobs using the same manifest.

Local validation used temporary toy inputs and local Bash execution, which does not establish site PBS scheduler acceptance. Detailed engineering records are retained separately. Repeat preflight after version or script changes, and do not reuse manifests across versions.
