# CESS v0.9.0

[English](RELEASE_NOTES_v0.9.0.md) | [简体中文](RELEASE_NOTES_v0.9.0_zh-CN.md)

CESS v0.9.0 retains the BayeSED HDF5 input interface and its main one-dimensional capabilities, reorganized as a Python 3.12 package. It produces simulated CSST GU/GV/GI spectra, uncertainties, masks, photometry, and emission-line diagnostics.

Main changes:

- Organize source by function, consolidate duplicated pipelines, and provide a unified `cess` command.
- Improve response/photon integration, noise, and uncertainty propagation through wavelength resampling; preserve negative background-subtracted noise fluctuations and adjacent-pixel covariance.
- Scientific revision `20261027-D5` updates the packaged GU/GV/GI wavelength ranges and central resolving powers using Table 1 of arXiv:2511.06917v2, with explicit one-dimensional response, template, sampling, and wavelength-window conventions. See the [instrument model](INSTRUMENT_MODEL_251106917.md).
- Retain invalid-input IDs and original rows, mark quality bits, and skip their simulation without modifying upstream files.
- Default to one complete HDF5 per input. Support directories, globs, and explicit source intervals. `-j` controls file/chunk worker processes; the default native-library thread request is 1.
- Add `cess merge` to stream scientific fields in source-row order, remap emission-line source references, and write a self-contained file.
- Default modern entry points to grouped schema 3.0, while retaining legacy 2.5 and a unified reader. Record configuration, input/code/resource identities, and random streams; enforce strict manifest resume and refuse overwrites.
- Provide optional MGE strip extraction, public toy examples, installation/command/user guides, and Gravity directory-job templates.
- License project source and documentation under MIT, with bundled-resource scope, SHA values, and provenance boundaries documented separately.

- Include the optional bilingual [HDF5 desktop viewer](VIEWER.md) for read-only structure/spectrum exploration, filtered plots and CSV/figure export (`cess-viewer`, `[viewer]` extra).

## Installation and migration

Use a new Python >=3.12,<3.13 environment and follow the [installation guide](INSTALL.md) to install this source or its exact artifacts. Do not reuse the Python 3.7 dependency list or install an unidentified `cess` package from PyPI merely because it has the same name.

Existing BayeSED inputs need no preliminary conversion but must satisfy the named-field and unit contract. Corrections to the scientific operators mean that numerical identity with historical v0.8.5 is not promised; interpret changes using field semantics and model definitions. Schema 2.5 output differs from the default schema 3.0 layout. Use the reader and field names rather than fixed column indices.

Development or older manifests bind their original code/resource identities and cannot be resumed across versions or scientific revisions. Package version `0.9.0` now identifies scientific revision `20261027-D5`; earlier D4 products have different science settings. Existing scientific products retain their original generation provenance; do not rewrite them merely to change a version label. Use a new output directory and a new manifest for D5 recalculations. The default is one worker; explicitly request batch concurrency with `-j N` and provide enough CPU and memory.

## Usage and known boundaries

[Quick start](QUICKSTART.md) · [Bash commands](COMMANDS.md) · [User guide](USER_GUIDE.md)

The packaged defaults adopt a constant-FWHM Gaussian instrument response and four samples per reference FWHM as explicit one-dimensional approximations. The central R values in the reference paper do not establish a full wavelength-dependent LSF. The input library is separately approximated as Gaussian R=300; where a GU template is broader than the target, `preserve` retains its width without sharpening and warns that the target was not attained. Analysis windows use the actual supplied wavelength centers. See the [instrument model](INSTRUMENT_MODEL_251106917.md) for these assumptions and retained observing conventions.

Historical profile properties and actual instrument PSF/background/sensitivity still require checks appropriate to the scientific application. Window line S/N is not completeness or independent detection significance, and zeroth-order integrated photometry is not a CCD zeroth-order image. See the [scientific description](SCIENCE_MODELS.md) for model definitions and limitations.
