# CESS roadmap

[English](ROADMAP.md) | [简体中文](ROADMAP_zh-CN.md)

CESS v0.9.0 focuses on fast one-dimensional slitless spectra. The following
items are future plans, not capabilities delivered in this release, and no
fixed release schedule is promised.

## Maintaining the one-dimensional release

- Preserve readable BayeSED inputs and versioned products; record scientific definitions and migration effects when fixing issues.
- Check dependencies and resource provenance for actual platforms, and measure performance across samples, worker counts, and memory conditions.
- Add research-specific PSF/LSF, background, aperture, and sensitivity calibration, together with emission-line completeness and contamination assessments.

## CESS+: from individual two-dimensional sources to CCD scenes

1. **Interfaces and electron accounting:** define SED, image coordinates/WCS, PSF/LSF, spectral-order response, pixels, exposure, and noise units and their application points. Use static point-source and narrow-line controls to check flux conservation.
2. **Single-source zeroth-/first-order images:** select input sources and observing parameters, then produce noiseless expectations, noisy two-dimensional images, variance/masks, trace/wavelength mappings, and corresponding one-dimensional data for each order.
3. **A single-CCD scene from a real sky field:** project catalogue positions with WCS, sum expectations from all sources before adding detector noise, and record boundary losses, spectral orders, and source provenance.
4. **Scene extraction and contamination handling:** extract source spectra using direct-image/catalogue priors, trace, and response; output errors, contamination models, and quality flags, with separate validation of isolated, overlapping, and edge sources.
5. **Extended disk-galaxy kinematics:** construct a source-plane Doppler forward model, validate static controls first, then assess injection/recovery over PA, S/N, size, and velocity parameters, reporting bias, degeneracies, and selection effects.

Two-dimensional dependencies remain optional: a one-dimensional run should
not require generating a two-dimensional image first. Preserve original
inputs, truth, and noisy observations separately; an already noisy
one-dimensional spectrum must not be treated as a noiseless input and given
another noise realization. Historical two-dimensional prototypes do not
establish a validated CCD/WCS or kinematic-recovery capability.

Interfaces, units, licenses, and Python 3.12 compatibility of existing
prototypes and external backends require separate review. They will not be
copied wholesale into the one-dimensional core without verification. Specific
implementations will be governed by their own versioned design and validation
documents.
