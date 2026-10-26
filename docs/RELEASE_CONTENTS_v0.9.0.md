# CESS v0.9.0 Release Contents

[English](RELEASE_CONTENTS_v0.9.0.md) | [简体中文](RELEASE_CONTENTS_v0.9.0_zh-CN.md)

This document defines the public scope of the one-dimensional software. The version is 0.9.0. Preparing local artifacts, creating or pushing Git commits, adding a version tag, and publishing a GitHub Release are separate steps. Check the repository for its actual online publication status.

## 1. Content included in the release

| Content | Location and purpose |
| --- | --- |
| Main package | `src/cess/`, organized into configuration, spectra, instrument, morphology, photometry, emission lines, I/O, pipelines, and CLI modules |
| Default resources | `src/cess/data/`: default JSON; seven photometric responses; six zeroth-/first-order responses across GU/GV/GI; background; GEMS; and two precomputed morphology libraries |
| Installation and dependencies | `pyproject.toml`, `setup.cfg`, `requirements/`, and `MANIFEST.in` |
| User documentation | README, installation, quick start, Bash commands, user guide, output description, and scientific model documentation |
| Runnable examples | `examples/make_toy_input.py` and its instructions; no real upstream catalogue |
| Server templates | `scripts/gravity/`, including preflight, a PBS payload, and site-resource configuration instructions |
| Compatibility entry points | Explicitly listed `CESS_v0.8.6` Python aliases, generator, and 17 resource copies required by tests, plus selected legacy entry points at the root |
| Tests | `tests/` and four small source-code numerical oracles; these are not user simulation outputs or complete development archives |
| Licensing and provenance | `LICENSE` (MIT), `RESOURCE_NOTICE.md`, `docs/RESOURCE_MANIFEST.json`, changelog, and release notes |

The source distribution retains compatibility files for existing scripts and a self-contained test suite. It does not publish the entire historical v0.8.6 research directory; consult the artifact file list for the exact contents. Historical author/version comments in Python sources preserve provenance. The current version is defined by `cess.__version__` and the package metadata.

The 18 bundled resource files include the default configuration; 17 are calibration or morphology data. File sizes and SHA values are in the [resource inventory](RESOURCE_MANIFEST.json). The maintainer has approved their distribution; provenance and rights boundaries are explained in the [resource notice](../RESOURCE_NOTICE.md). A currently adopted copy is not presented as an independently verified, unified official upstream release, and external resources are not relicensed by the project.

This release also includes the optional desktop viewer: `src/cess/viewer/`, the `cess-viewer`
entry point, the `cess_h5_viewer.py` launcher, `requirements/viewer-py312.txt`, and the
[bilingual viewer guide](VIEWER.md). Matplotlib is an optional extra; Tk comes from the runtime environment.

## 2. Content kept locally

- Real BayeSED inputs, complete or sampled scientific outputs, run manifests, logs, and private path information.
- Conda/venv environments, caches, temporary build directories, `.git`, local application/agent settings, and credentials.
- Complete historical versions, notebooks, PDF manuals, manuscript drafts, development records, and checkpoints.
- `CESS_kinematics/`, old 2D backends, and CESS+ research materials that have not passed acceptance checks.
- Other personal tools, temporary scripts, and independent data tables not included in the release file list.

The presence of a local directory does not make it part of the release. Do not run `git add .` on the research workspace; use the prepared clean source directory or explicit file list. Release preparation does not delete or move original data or historical records.

The numerical oracles contain only the small historical source fragments needed by tests. Their distribution copies have had machine-specific absolute interpreter paths removed while retaining the same executable AST. Original archive identities and sanitized-copy SHA values are recorded separately in `tests/baselines/PROVENANCE.json`. Publishing or accessing the original large development archives is not required.

## 3. Three deliverables

| Deliverable | Contents | Intended use |
| --- | --- | --- |
| Wheel: `cess-0.9.0-py3-none-any.whl` | Current Python package, 18 default resources, entry points, and license metadata | Installation and execution; dependencies are installed for the target platform |
| sdist: `cess-0.9.0.tar.gz` | Buildable source, public documentation, examples, server templates, compatibility files, and test suite | Source installation or reproduction of engineering tests |
| Clean source directory | Public files and original build configuration, without the local research workspace or generated build metadata | GitHub file review and preparation for a later commit |

`py3-none-any` means CESS itself has no platform-specific binary extension. It does not mean that NumPy/SciPy dependencies can be copied across platforms. A Linux server needs its own Python 3.12 environment; a Mac environment directory cannot simply be moved there.

Local release preparation provides a file inventory, SHA256 checksums, and a validation summary alongside the artifacts. `MANIFEST.in` defines the source distribution's build scope. The wheel and sdist contain different files, so checking one alone does not establish that all deliverables are complete. Scientific products retain their generating-version provenance and are not rewritten when 0.9.0 is installed.

## 4. Documentation entry points

New users should create an environment following the [installation guide](INSTALL.md), then read the [quick start](QUICKSTART.md) and [Bash command guide](COMMANDS.md). The [user guide](USER_GUIDE.md) covers input fields, physical quantities, and advanced settings; the [output description](OUTPUT_LAYOUT.md) defines the layout; and the [scientific model guide](SCIENCE_MODELS.md) explains assumptions.

By default, each input file produces one complete output. Directory concurrency requires explicit `-j` settings, with worker counts chosen according to task memory requirements. Chunked simulations can subsequently be merged into a self-contained product. Gravity resource requests and CESS worker settings are separate; see the [server guide](../scripts/gravity/README.md).

## 5. Release boundaries

v0.9.0 is a one-dimensional software release. It does not claim completed 2D CCD imaging, spectral-contamination extraction, or kinematic inversion. User whole-file runs, engineering tests, and small-sample concurrency comparisons do not replace instrument science calibration or establish that Gravity deployment/scheduler acceptance has been completed.

Before uploading, verify the exact source and artifacts, version, and resource SHA values, and retain the validation summary. Then create the Git commit, `v0.9.0` tag, and GitHub Release under the maintainer's authorization. Publishing the software does not require publishing real inputs or simulation results.
