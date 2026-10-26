# CESS changelog

[English](CHANGELOG.md) | [简体中文](CHANGELOG_zh-CN.md)

## v0.9.0

### Installation and operation

- Package the one-dimensional simulator as `src/cess` for Python 3.12.
- Provide `cess simulate`, `cess el-detect`, and `cess merge`, with file, directory,
  glob, source-range and chunk inputs.
- Produce one complete spectral HDF5 per input by default. Use `-j` for bounded
  file/chunk concurrency and `--threads-per-worker` for native-library threads.
- Add a bilingual desktop HDF5 viewer with per-source spectra, variable plots,
  filtered CSV export, figure saving, and a responsive workbench.
- Include English and Chinese installation, command and user guides, runnable
  examples, and Gravity/PBS scripts.

### Numerical handling

- Improve photon/response integration, read noise and wavelength resampling.
  Retain negative background-subtracted fluctuations and adjacent-pixel covariance.
- Keep invalid-input IDs and source rows, with `data_mask=2` and decodable
  `input_quality_flags`; do not alter the input catalogue.
- Fix process waits, invalid-source parameter reuse, row alignment, empty line
  results, and parameter-table dimensions.
- Separate model and observed photometry, passband-integrated zeroth-order
  measurements, reference peaks, detected candidates, and line-window measurements.
- Support explicit extraction conventions and optional joint MGE strip extraction.

### Products and reproducibility

- Default to structured HDF5 schema 3.0; retain optional schema 2.5 output and
  a unified read-only product interface.
- Record original rows, configuration, input/code/resource identities, and
  component-specific random streams; validate manifests before resuming.
- Merge chunks without recalculating spectra, preserving row order and
  remapping emission-line source references.
