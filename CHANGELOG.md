# CESS changelog

[English](CHANGELOG.md) | [简体中文](CHANGELOG_zh-CN.md)

## v0.9.0

- Add the optional bilingual HDF5 desktop viewer through `cess-viewer`, `python -m cess.viewer`
  and the compatibility script. Separate read-only data access from GUI dependencies; remove
  personal defaults; correct source-axis selection, filtered CSV export, precision/input protection,
  and quantity/line labels.

- Refresh the optional viewer with a modern workbench, rounded primary buttons,
  file summary, contextual exports, scrollable controls and collapsible plot options.

Local release preparation: 2026-10-26. The source and distribution artifacts use
version `0.9.0`; the actual GitHub release date and download locations are
determined by the published release record.

This version retains the main functionality for generating GU/GV/GI
one-dimensional slitless spectra from BayeSED HDF5 galaxy templates, while
organizing the runtime interfaces, scientific-array definitions, reproducible
provenance, and maintenance structure.

### Architecture and use

- Adopt MIT for the project's own code and documentation, and record the maintainer-approved packaged resources with a SHA256 inventory.
- Simplify the public CLI with `-i/-o/-j` and `--all-effects`, retaining the original defaults and long options.
- Provide English and Chinese installation, quick-start, user, and Bash command guides covering complete-file output, scientific switches, concurrency, logs, and strict resume. Public documentation uses generic paths; development validation records remain separate.
- Continue to control file-level multiprocessing with `-j/--workers`, defaulting to 1. Add `--threads-per-worker`, requesting one native-library thread before spawn by default, with a concurrency summary and per-task PID/timing records.
- Allow directory input through the Gravity wrapper. With `CESS_CHUNK_SIZE` unset, each input file has one complete task/output; chunking and an estimated memory budget remain explicit options.
- Produce one complete spectral HDF5 per input file by default. After explicit chunking, `cess merge` streams the products into one file without rerunning simulations, checking contiguous rows, run identity, and shared geometry while remapping emission-line source references and recording chunk provenance.
- Organize the runtime as the `src/cess` Python package, separating spectra, instrument, morphology, photometry, emission lines, I/O, and pipelines.
- Provide unified `simulate`, `el-detect`, and `merge` commands, with files/directories/globs, source intervals, chunks, and bounded multiprocessing where applicable.
- Retain legacy script compatibility entries while sharing preparation, writing, and result-collection logic to reduce duplication.
- Move to Python 3.12 and separate core, sampling-tool, plotting, and legacy-renderer dependencies.

### Numerical handling and results

- Replace the old compatibility `varsmooth` implementation, which is outside the production pipeline, with an independent real-space Gaussian integral to address upstream redistribution restrictions. Boundary/numerical conventions and the `oversample=1` restriction are documented; production `convolve_to_bins` is unchanged.
- Add vectorized per-source safety checks for computationally required inputs. Invalid rows retain IDs/original row numbers, receive `data_mask=2` and decodable `input_quality_flags`, and skip simulation without modifying the input.
- Fix child-process waiting, stale morphology reuse for invalid sources, row alignment, empty emission-line results, and parameter-table dimensions.
- Locally filter only Astropy's repeated linear-model solver advisory during default continuum fitting, preserving the adopted fitter, numerical behavior, and other diagnostics.
- Improve photon/response integration, read noise, and negative-fluctuation handling; propagate wavelength-resampling variance, masks, and adjacent-pixel covariance.
- Separate model/observed photometry, zeroth-order passband integration, and historical one-dimensional diagnostic proxies; add emission-line false-positive/missed-peak diagnostics and window measurements.
- Support explicit extraction conventions and an optional joint MGE strip model, retaining the historical default and its limitations.

### Reproducibility and products

- Default modern entry points to grouped HDF5 schema 3.0, while retaining compatible 2.5 output and a unified read-only reader.
- Record original source rows, configuration, and input/code/resource identities, with per-source random streams and strict manifest resume.
- Add public toy examples, English and Chinese documentation and user guides, Gravity preflight, and PBS payload scripts.

### Scope

The unified release focuses on one-dimensional simulation. CESS+ single-source
two-dimensional images, CCD scenes, contamination-aware extraction, and
kinematic recovery are separate development stages. Local engineering
validation does not replace PSF/LSF, background, sensitivity, or scientific
selection-function calibration. User acceptance feedback has been received for
complete local catalogue runs; these engineering and use checks do not certify
the instrument model.

The source is MIT-licensed, and the maintainer has confirmed the packaged
resource scope. Public contents are checked against this release's file list.
Preparing local artifacts does not mean GitHub publication or a complete server
rerun has already occurred.
