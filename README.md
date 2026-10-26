# CESS v0.9.0

**CSST Emulator for Slitless Spectroscopy**

English | [简体中文](README_zh-CN.md)

Version: v0.9.0 · Python 3.12

CESS reads galaxy spectral templates and catalogues produced by BayeSED and
generates one-dimensional CSST GU, GV and GI slitless spectra under specified
instrument-response, observing and morphology assumptions. It is intended for
large galaxy samples: convert existing SEDs into noisy spectra, uncertainties,
signal-to-noise ratios and measurements without first rendering a full detector image.

Start with [installation](docs/INSTALL.md) and the [quick start](docs/QUICKSTART.md).
The [Bash command manual](docs/COMMANDS.md) explains commands step by step;
the [user guide](docs/USER_GUIDE.md) covers parameters, inputs and products.

After installing and activating the environment, run this in the directory containing `input.hdf5`:

```bash
cess simulate -i input.hdf5 -o results --all-effects
```

The default is **one complete output HDF5 per input file**, retaining all source
rows, including flagged invalid sources. Replace the file with a directory to
process its HDF5 inputs. The default `-j 1` processes files sequentially;
explicit `-j N` allows up to N concurrent file tasks when memory permits.
`--all-effects` enables the four optional science switches together.

## 1. Purpose and version

| Item | Current behavior |
| --- | --- |
| Software | `0.9.0`, preserving and restructuring the 1D simulation workflow |
| Release status | Local source and artifacts are prepared as v0.9.0; consult the repository for the actual GitHub release |
| Main input | BayeSED HDF5 galaxy templates, JSON configuration and supporting resources |
| Main output | GU/GV/GI 1D spectra, optional photometry and line products, and optional run manifests |
| Python | `>=3.12,<3.13`, in an environment created for the target platform |
| Modern layout | Structured HDF5 schema 3.0; compatible schema 2.5 is available explicitly |

Typical uses include sample observability studies, spectral S/N distributions,
emission-line detection and measurement experiments, and paired comparisons of
wavelength-calibration errors. Check the models and resources for the research
question. CESS does not perform BayeSED fitting, automatic redshift measurement
or a scientific completeness calibration.

## 2. Features

| Feature | Capability | Important distinction |
| --- | --- | --- |
| 1D spectroscopy | GU/GV/GI model and noisy spectra with uncertainties | Preserves the existing BayeSED input contract |
| Response and detector noise | Throughput folding, expected electrons, sky, dark current and read noise | Rates, counts, calibrated fluxes and errors have separate definitions |
| Morphology broadening | Uses sampled galaxy morphology and stored profile resources | Does not import arbitrary observed 2D galaxy images |
| Wavelength-calibration error | Resampling on a common nominal pixel grid | Propagates variance, masks and neighboring-pixel covariance |
| `paired-wce` | Outputs branches without and with wavelength error | The branches share a noise realization |
| Seven-band photometry | NUV/u/g/r/i/z/y response-weighted measurements | Model, observed value, error, S/N and status are separate |
| Zeroth-order photometry | Standard-mode GU/GV/GI passband-integrated observations | Distinct from historical `_0th` 1D diagnostic proxies |
| Emission lines | Detection, truth matching, missed/false-positive diagnostics and window measurements | Primarily GV/GI; candidates are not automatically identified atomic transitions |
| Product postprocessing | `el-detect` on schema 2.5 or 3.0 spectra | Inherits source mapping and parent-model definitions |
| Batch scheduling | Files, directories, globs, source intervals, chunks and bounded multiprocessing | No manual filename-list editing |
| Reproducibility | Seeds, source-level random streams, provenance and strict manifest resume | Existing products are never silently overwritten |
| Product tools | Unified reader, plotting, stratified sampling, subset export and chunk merging | Named fields and units; merging does not recalculate spectra |
| Desktop HDF5 viewer | Structure/data browser, per-source spectra, X-Y plots and CSV/figure export | Optional Tk interface; English/Chinese; opens inputs read-only |
| Advanced morphology | Positive MGE libraries and optional joint-strip extraction | Explicit aperture and PSF/LSF assumptions required |

The four optional `simulate` switches are **off by default**:
`--morphology`, `--wavelength-error`, `--photometry` and `--emission-lines`.
Enable the required effects explicitly or use `--all-effects`.
The default mode is `standard`; the default random seed is 0.

## 3. Workflow and models

```text
BayeSED HDF5 + JSON + throughput/background/morphology resources
    -> select original source rows; validate fields and units
rest wavelength -> observed wavelength; smoothing and pixel integration
    -> integrate the native response and predict source electron rates
morphology selection, dispersion broadening and spatial extraction
    -> source + sky + dark current + read noise
model/noisy spectra, uncertainties, S/N and masks
    -> optional wavelength resampling and covariance propagation
optional photometry, zeroth-order integration and line measurements
    -> atomic publication, provenance and optional manifest updates
versioned HDF5 products + JSON task records
```

Input wavelengths are converted from microns to Angstrom and shifted by the
input redshift. The SED follows the existing BayeSED convention of
observed-normalized `fnu` in microJy; arbitrary intrinsic luminosity spectra are
not interchangeable with it. Response calculations integrate over wavelength;
one throughput value at each output-pixel center does not replace upstream integration.

Noise includes source/background Poisson fluctuations, dark current and Gaussian
read noise. Spatial extraction pixels and exposures enter the noise-area
calculation. Background-subtracted fluctuations can be negative and are retained.

Wavelength-calibration error (WCE) uses linear resampling on a common nominal grid,
rather than a complete instrument wavelength-solution fit. Variance uses squared
weights; adjacent-pixel covariance is saved. Integrated measurements should use
the available covariance. The `paired-wce` branches share noise and are not
independent exposures. Pixel, band-mean, historical peak-neighborhood and
integrated window-line S/N have different definitions.

The default `spectral_extraction.mode=legacy_full` retains the historical
full-source signal approximation. `explicit` accepts a source fraction and
spatial pixel count. `mge_strip` calculates a joint strip and dispersion kernel
from verified positive Gaussian source components. These conventions are not
interchangeable by renaming parameters; MGE requires a compatible library and
explicit PSF/LSF roles. See [scientific models](docs/SCIENCE_MODELS.md) and
[morphology/MGE](docs/MGE_MODELS.md).

## 4. Inputs and outputs

CESS accepts the existing BayeSED-shaped contract, not an arbitrary HDF5 file.

| Input | Content and units |
| --- | --- |
| `ID` | N identifiers, tracked together with original row indices |
| `parameters` / `parameters_name` | N-by-K values and field names, including required position, magnitude, mass and redshift columns |
| `best_fit/wavelength_rest` | Shared W-element rest-wavelength vector; conventionally microns |
| `best_fit/z` | N redshifts |
| `best_fit/spec_*` | N-by-W SED `fnu`, conventionally microJy; the simulation dataset must be unambiguous |
| JSON and resources | Runtime paths, instrument/exposure/background settings, responses and morphology resources |

See the user guide for exact names, units and path rules. Check wavelength support,
source-axis alignment and parameter meanings when changing template libraries.

Modern HDF5 schema 3.0 organizes the result as follows:

```text
product.hdf5
├── catalog/       # IDs, original rows, tables, extraction and morphology diagnostics
├── spectra/       # GU/GV/GI arrays and shared grids
├── photometry/    # optional seven-band photometry
├── zero_order/    # standard-mode zeroth-order passband photometry
├── lines/         # candidates, matches and window measurements
└── provenance/    # input/config/version/code/resource identities and layout mapping
```

All root groups have defined status metadata: presence alone does not mean a
measurement was enabled or applicable. Spectral arrays are usually
`(source,pixel)`; wavelengths are Angstrom and flux/error fields are usually
microJy. GU/GV/GI can have different pixel counts. Read tables by field name,
`axis_order` and unit.

| Common field | Meaning |
| --- | --- |
| `wave` | Shared observed-wavelength grid for a band |
| `flux_ujy` | Photon-equivalent bin `fnu` before throughput folding and morphology |
| `response_flux_ujy` | Response-folded quantity; do not multiply throughput again |
| `source_electron_rate_pre_morph` | Source electron rate before morphology; use the recorded unit |
| `flux_elec` | Noiseless expected electrons |
| `flux_ujy_model` / `flux_ujy_with_noise` | Model/noisy spectra on the nominal response-calibrated scale |
| `ferr` / `snr` | Spectral uncertainty and pixel S/N |
| `spec_mask` / `snr_mask` | Validity under their respective definitions |
| `covariance_lag1` | Adjacent-pixel covariance under the current WCE operator |

Extracted spectra are not automatically aperture-corrected to total-source flux.
Historical `_0th` arrays are diagnostic proxies on the first-order grid; actual
zeroth-order passband measurements are in `zero_order`.

`--output-layout legacy` writes schema 2.5. Legacy script execution and `task(...)`
retain 2.5; modern CLI and `simulate_file(...)` default to 3.0. The unified reader
supports both. Conversion creates a new file. Subset exports have their own schema.

## 5. Changes from earlier versions

The v0.9.0 refactor preserves the main 1D capabilities while improving maintenance,
operation and traceability:

- A modular `src/cess` package replaces repeated script-level logic.
- Explicit configuration/resource loading reduces working-directory coupling.
- A bounded worker pool, complete waits and failure records replace unbounded spawning.
- Fixes cover invalid-source parameter reuse, WCE row alignment, empty line results
  and photometric parameter-table dimensions.
- Improved response/pixel integration, read noise, negative fluctuations and
  resampling covariance have explicit definitions.
- Photometry separates model and observed values; line diagnostics retain false
  positives, missed peaks and window-measurement semantics.
- Versioned HDF5, named tables, resource caches, bounded morphology-statistics
  batches and shared preparation/writing/collection reduce duplication.
- Component-specific source random streams, hashes and strict resume support reproducibility.
- Python 3.12 dependency profiles replace the historical Python 3.7 environment.

This is more than a directory rearrangement: bug fixes and numerical-model changes
can alter v0.8.5 results. Later structural deduplication was checked against frozen references.

## 6. Installation and a first run

With Conda available, install from the downloaded or extracted source root,
replacing `/path/to/CESS`:

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

Activate the environment in each new terminal; do not recreate an existing usable
environment. The [installation guide](docs/INSTALL.md) also covers venv, wheel/sdist
installation and upgrades. This installs source you obtained and does not assume
that CESS has been uploaded to PyPI.

Core-only installation uses `python -m pip install .`. Core packages are NumPy,
SciPy, Astropy, h5py, Specutils and tqdm; see [exact requirements](requirements/core-py312.txt).
pandas supports sampling, Matplotlib supports plotting, and optional GalSim supports
some historical rendering/library tools. Stored-profile 1D runs do not require GalSim.

Continue in the activated environment and source root, using fresh destinations:

```bash
python examples/make_toy_input.py toy.hdf5 --sources 4 --invalid-first
cess simulate -i toy.hdf5 -o toy_results --manifest toy_results/run.json \
  --all-effects --seed 1214 -j 1
```

The arbitrary toy SEDs and parameters exercise the interface, not science calibration.
Without an external JSON, packaged defaults are used. Existing products are not
overwritten: resume the identical task or choose fresh output/manifest paths.

Installed `cess` and `python -m cess` expose the same interface. Relative data paths
use the current directory. A continuation `\` must be the last character of its
line; quote paths containing spaces. See the [Bash manual](docs/COMMANDS.md).

### Optional HDF5 desktop viewer

`cess-viewer` provides read-only structure/attribute inspection, per-source GU/GV/GI spectra,
general X-Y plots, and figure/CSV export, with English and Chinese interfaces.
It is separate from simulation and requires a graphical desktop, Matplotlib and working Tcl/Tk.
From the source root in the activated environment:

```bash
python -m pip install '.[viewer]'
cess-viewer /path/to/product.hdf5 --lang en
```

Omit the file to start an empty window. If Conda lacks Tk, run `conda install tk`
and check `python -m tkinter`. Headless Gravity batch jobs do not need the viewer.
See the [HDF5 viewer guide](docs/VIEWER.md).

## 7. Source layout

```text
CESS/
├── src/cess/
│   ├── cli/            # commands and plotting
│   ├── config/         # loading and validation
│   ├── spectra/        # units, integration and spectral operators
│   ├── instrument/     # background, detector noise and wavelength error
│   ├── morphology/     # sampling, profiles, MGE and reference calculations
│   ├── photometry/     # response-weighted measurements
│   ├── lines/          # detection, matching and window measurements
│   ├── pipeline/       # standard/paired/line pipelines and collection
│   ├── io/             # products, selections, atomic writes, manifests and sampling
│   └── data/           # default configuration and resources
├── tests/              # numerical, contract, compatibility and workflow checks
├── requirements/       # Python 3.12 dependency profiles
├── examples/           # public interface examples
├── scripts/gravity/    # preflight and PBS payload
├── docs/               # installation, operation, formats and models
└── CESS_v0.8.6/        # selected compatibility entry points and required resources
```

The [release contents](docs/RELEASE_CONTENTS_v0.9.0.md) define the distribution.
Real catalogues, outputs, historical research and development archives are not
release files merely because they are present in a workspace.

## 8. Batch runs and parallelism

Without `--chunk-size`, each input file is one task and one output HDF5.
`-j` / `--workers` sets the process limit, default 1. An unchunked single input
remains one task even when more workers are requested.

On a host with memory for two complete input tasks, run from any directory:

```bash
cess simulate -i /path/to/catalogues -o /path/to/new_results --all-effects -j 2
```

Workers take the next file after finishing. The run reports input/task counts,
requested/planned worker limits and threads. Start/finish records include PIDs
and elapsed time; sequential execution is made explicit.

For a large single file, use chunks and then merge. In the directory containing `input.hdf5`:

```bash
cess simulate -i input.hdf5 -o chunks --all-effects --chunk-size 100 -j 2
cess merge -i chunks -o merged/complete.hdf5 --input-source input.hdf5 --dry-run
cess merge -i chunks -o merged/complete.hdf5 --input-source input.hdf5
```

All rows are processed; `100` limits rows per task, not the total sample. Merging
writes real, self-contained datasets in original row order without recalculating
spectra. `--input-source` verifies full coverage, IDs and SHA. Keep the destination
outside the chunk directory and merge different catalogues separately.

The default `--threads-per-worker 1` requests native-library thread limits before
spawn imports numerical libraries, restoring the parent's environment after the
pool closes. It is not CPU affinity or a strict process-tree CPU cap. Workers are
bounded by the request, task count and optional `--memory-budget-gib`, which uses
conservative estimates rather than real-time RSS or an OS hard limit. Simultaneous
large tasks add their memory use: inspect a small run and `--dry-run` first.

On Gravity, scheduler allocation and CESS worker settings are separate. Scripts
fix native thread requests to 1 and check the declared worker/CPU relationship.
Directory input is supported; an unset `CESS_CHUNK_SIZE` keeps whole-file tasks.
See [Gravity instructions](scripts/gravity/README.md). The runner operates on one
host and does not distribute work across nodes with MPI.

## 9. Scientific limits and future work

- The nominal R=300 of identified existing input libraries is known, but their
  detailed template LSF is not certified; the default remains `unknown`.
- Upstream Gaussian FWHM uses `*_dlambda`; `*_Res` does not enter that operator.
  Changing `*_Res` alone or treating FWHM as pixel spacing does not reconfigure resolution.
- The default sky is a fixed template. `background_v_mag` is a provenance label
  in that mode, not a standalone rescaling control or a time/solar-geometry zodiacal model.
- Source morphology, aperture, PSF and template/instrument LSF need consistent
  definitions. MGE/strip assumes separable SED/spatial structure, achromatic
  morphology and the specified PSF form.
- The fast 1D path excludes full-field spectral contamination, CCD-edge projection
  and extraction from a realistic 2D scene.

Engineering checks do not replace instrument or scientific selection-function calibration.
Future **CESS+** work includes single-source zeroth/first-order images, catalogue/WCS
CCD scenes, contaminated-spectrum extraction and kinematic recovery studies.
These are not completed v0.9.0 capabilities; historical prototypes require separate
adaptation and validation. See the [roadmap](docs/ROADMAP.md).

## 10. Documentation and licensing

- [Quick start](docs/QUICKSTART.md)
- [Bash command manual](docs/COMMANDS.md)
- [User guide](docs/USER_GUIDE.md)
- [Installation and environments](docs/INSTALL.md)
- [Output layout and reader](docs/OUTPUT_LAYOUT.md)
- [HDF5 desktop viewer](docs/VIEWER.md)
- [Scientific models and limits](docs/SCIENCE_MODELS.md)
- [Morphology/MGE](docs/MGE_MODELS.md) and [bundle format](docs/MGE_BUNDLE_FORMAT.md)
- [Gravity/PBS](scripts/gravity/README.md)
- [Changes](CHANGELOG.md), [release notes](docs/RELEASE_NOTES_v0.9.0.md) and [release contents](docs/RELEASE_CONTENTS_v0.9.0.md)
- [MIT license](LICENSE) and [resource notice](RESOURCE_NOTICE.md)
- [简体中文首页](README_zh-CN.md)

Repository: [RainW7/CESS](https://github.com/RainW7/CESS). Check the version and exact
artifact: local source may precede the remote repository. CESS-owned code and
documentation use MIT. The maintainer approved distribution of the supplied
throughput, background and morphology resources; their scope and known provenance
are recorded separately. External software/data retain their rights. Actual online
dates and downloads are defined by the GitHub release, not local artifact preparation.
