# Catalogue arrays on Gravity

[English](ARRAY.md) | [简体中文](ARRAY_zh-CN.md)

`catalogue_array.py` prepares separate MzLS and DECaLS Torque arrays. Each array task requests one CPU, processes one complete input catalogue, and writes one complete spectral HDF5. There are no source-row chunks to merge. The default campaign contains **1,385 MzLS and 1,385 DECaLS files: 2,770 files in total**.

## Resources and concurrency

The [Gravity job guide](https://jupyter.gravity.sjtu.edu.cn/doc/#/Basic/Job) uses PBS/Torque and documents `nodes=1:ppn=1`, array requests such as `-t 1-N%slots`, and a `normal` queue limit of 48 hours and 376 GB per node. Run scientific computation on compute nodes. The generated scripts use that Torque syntax.

| Survey | CPUs per task | Requested memory per task | Walltime | Concurrent task limit |
| --- | ---: | ---: | ---: | ---: |
| MzLS | 1 | 16 GB | 12 hours | 32 |
| DECaLS | 1 | 48 GB | 24 hours | 32 |

Submitting both arrays allows at most 64 files to run concurrently by default. The scheduler may run fewer; this neither reserves a full node nor guarantees 64 CPUs. Each task uses `--workers 1 --threads-per-worker 1`, with numerical-library thread requests set to one.

CESS estimates for representative local inputs were about 8.9–9.1 GiB for MzLS and 29.2 GiB for DECaLS. These are scheduling estimates, not measured peaks or upper bounds. The initial 16/48 GB requests need confirmation with complete-file pilot runs. The CLI also receives an estimated memory budget equal to 85% of the requested memory; PBS enforces actual resource limits.

Compressed input size is not working memory: input arrays, three-band spectra, noise products, and output assembly can coexist. A fresh Python process for each file releases memory between tasks.

## Prepare on the server

Install CESS in a Linux Python 3.12 `cess` environment using the [installation guide](../../docs/INSTALL.md). Do not copy a macOS Conda environment. Replace all `/path/to/...` placeholders below with real absolute server paths.

```bash
conda activate cess
cd /path/to/CESS

python -s -B scripts/gravity/catalogue_array.py prepare \
  -i /path/to/CESS_input \
  --run-dir /path/to/runs/cess_v090_all
```

The run directory must not already exist. The bundled instrument configuration is the default. Add `--config /path/to/emulator_parameters.json` to select your own configuration. To place products on another filesystem, add `-o /path/to/CESS_output_all`; this output directory must also be new, and products will use survey subdirectories beneath it. Preparation records the absolute path of the current Python interpreter; batch jobs use it directly without relying on shell activation.

Preparation checks survey names and the expected count of 1,385 files per survey, HDF5 metadata, estimated task memory, source/resource identity, and input size/modification time. Unknown surveys, count mismatches, duplicate output stems, and tasks exceeding their memory budget are rejected. Directory discovery covers direct `.hdf5/.h5` children; multiple directories or quoted globs may follow `-i`.

| Generated file or directory | Purpose |
| --- | --- |
| `MzLS.pbs`, `DECaLS.pbs` | Survey-specific submission scripts |
| `plan.json`, `plan.sha256` | Fixed input order, resources, and campaign identity |
| `config.json`, `catalogue_array.py` | Frozen effective configuration and runner |
| `products/MzLS/`, `products/DECaLS/` | Complete spectral products, created during execution |
| `manifests/<survey>/` | A separate CESS resume manifest for every input |
| `logs/<survey>/`, `status/<survey>/` | Attempt logs, resource measurements, and status |

Tasks are ordered by decreasing estimated memory within each survey, starting at index one. The first task therefore tests a larger input. Do not move the run directory, replace inputs, edit the frozen configuration, or update CESS source/resources or Python dependencies while the campaign is running. Preparation reads catalogue metadata rather than scanning approximately 3 TB of input payload on the login node. CESS calculates full input hashes for each file on the compute node.

A read-only task preview is available before submission:

```bash
python /path/to/runs/cess_v090_all/catalogue_array.py run \
  --plan /path/to/runs/cess_v090_all/plan.json \
  --group MzLS --index 1 --dry-run
```

A new-run preview creates no products or new CESS manifest. Computation uses `standard` mode, `structured` output, `--all-effects`, and seed 1214, without changing the existing instrument/physical model. Set `--seed` during preparation if a different seed is needed.

## Run two complete-file pilots

```bash
qsub -t 1-1%1 /path/to/runs/cess_v090_all/MzLS.pbs
qsub -t 1-1%1 /path/to/runs/cess_v090_all/DECaLS.pbs
```

The command-line array range overrides the full range while retaining the survey's resource request. Pilots use `normal`: a representative DECaLS run took about four hours locally, exceeding the three-hour `debug` limit.

Wait for both tasks to finish. Check exit status, product row counts and sizes, elapsed time, and memory headroom. Logs retain each attempt. The child max-RSS metric is the maximum resident-set statistic reported for the child process tree, not the sum of simultaneously live processes. While the job remains queryable, also inspect PBS `resources_used.mem` and related fields using `qstat -f JOB_ID`; retain logs and use the site's historical accounting method if the job has already disappeared from `qstat`.

The supplied input sizes imply about **3.185 TB (2.897 TiB)** of inputs and roughly **138.5 million sources**; use the actual row total from preparation. Outputs and temporary files require additional space. Estimate output storage from the two pilot products multiplied by 1,385 per survey, plus temporary-write/retry headroom.

If resource settings need adjustment before production, prepare a fresh run directory, for example:

```bash
python -s -B /path/to/CESS/scripts/gravity/catalogue_array.py prepare \
  -i /path/to/CESS_input \
  --run-dir /path/to/runs/cess_v090_all_larger \
  --mzls-slots 32 --decals-slots 16 \
  --mzls-memory-gb 20 --decals-memory-gb 64 \
  --mzls-walltime 16:00:00 --decals-walltime 36:00:00
```

Choose values from the pilots and available account resources. Do not override the generated memory request with a smaller `qsub -l mem=...` while retaining the original budget.

## Submit the campaign

Check current queue and account limits first. The documentation does not establish that every account may submit arrays of 1,385 tasks with 32 slots:

```bash
qstat -q
qstat -f -Q
qstat -Bf
```

After both pilots have stopped, submit the full arrays:

```bash
qsub /path/to/runs/cess_v090_all/MzLS.pbs
qsub /path/to/runs/cess_v090_all/DECaLS.pbs

qstat -tan -u "$USER"
```

The completed pilot task is verified through its original CESS manifest and skipped. Do not submit an overlapping task while its earlier worker is still running.

If the site's current array-length limit is smaller, submit ranges sequentially using the same script:

```bash
qsub -t 1-500%32 /path/to/runs/cess_v090_all/MzLS.pbs
# Wait until the entire first range has stopped before submitting:
qsub -t 501-1000%32 /path/to/runs/cess_v090_all/MzLS.pbs
# Wait until the second range has stopped before submitting:
qsub -t 1001-1385%32 /path/to/runs/cess_v090_all/MzLS.pbs
```

Use the same procedure for DECaLS. A slot limit applies to each submitted array: three simultaneous `%32` arrays permit 96 concurrent tasks for that survey. Administrators can set array-size, slot, user-queue, and user-run limits; see [Torque server parameters](https://docs.adaptivecomputing.com/torque/6-1-1/adminGuide/Content/topics/torque/13-appendices/serverParameters.htm).

CESS's process pool is local to one host. Torque distributes these independent files across nodes. Do not request multiple nodes per array task or increase a single-file task's CESS worker count. The runner checks for one host and one slot in `PBS_NODEFILE`, along with available PBS CPU/node variables.

## Progress and recovery

```bash
python /path/to/runs/cess_v090_all/catalogue_array.py status \
  --plan /path/to/runs/cess_v090_all/plan.json
```

This prints status counts and `retry_commands` for unfinished indices; it does not submit them. Status files are operational records. CESS `--resume` performs the actual product provenance, checksum, and structure verification.

Each input has its own manifest, so one file's failure does not stop other array tasks. Resubmitting an index automatically uses `--resume` when its manifest exists. Valid complete output is verified and skipped; corrupt or mismatched output causes an error rather than being overwritten. An interrupted write may leave a `.partial` file, which is not a completed product.

After cancellation, timeout, or OOM, confirm through PBS that the previous task and workers have stopped before using a printed retry command. `SIGKILL` cannot be caught and may leave a status file marked `running`. Persistent `.lock` files are normal; do not delete them to force a retry.

The runner forwards catchable termination signals to the child process group and retains existing logs/manifests. An interrupted whole-file task restarts that file; other completed files do not need recomputation.

See the [Torque qsub reference](https://docs.adaptivecomputing.com/torque/6-1-1/adminGuide/Content/topics/torque/commands/qsub.htm) for submission options. Local script checks do not establish Gravity scheduling acceptance, complete-file resource requirements, or instrument scientific calibration.
