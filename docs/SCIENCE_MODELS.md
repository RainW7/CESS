# Physical conventions and limitations in CESS v0.9.0

[English](SCIENCE_MODELS.md) | [简体中文](SCIENCE_MODELS_zh-CN.md)

This document describes scientific revision `20261027-D5` of CESS `0.9.0`. Parameters make the calculation reproducible; releasing the software does not make them an independently calibrated official CSST instrument model. See the [command guide](COMMANDS.md) for routine execution and the [output layout](OUTPUT_LAYOUT.md) for stored fields.

## 1. Inputs and units

BayeSED rest-frame wavelengths are supplied in μm and converted to Angstrom by the operator; `best_fit/z` shifts the wavelength axis by `1+z`. The input SED convention is `fnu` whose observed normalization has already been established upstream. The runner interprets it in μJy and converts to cgs internally where needed. Do not supply an intrinsic luminosity density directly, or apply an additional distance or redshift flux factor automatically.

Provide source and wavelength axes, parameter names, and units according to the [user guide](USER_GUIDE.md). Source IDs may repeat: the program preserves correspondence using original row indices rather than sorting by ID. Required metadata, SED values, and wavelength support undergo numerical safety checks. Invalid sources retain their IDs and original rows, are marked by `data_mask=2` and `input_quality_flags`, and have invalid simulation arrays; the input file is not repaired. These checks protect numerical operations and do not define a scientific sample selection function.

## 2. Response integration, pixels, and resolution

For a response T that already includes electron detection efficiency, neglecting broadening, the source electron rate over a wavelength interval follows

$$
\dot N_e = \frac{A}{h}\int \frac{f_\nu(\lambda)T(\lambda)}{\lambda}\,d\lambda .
$$

A is the collecting area used in the calculation; wavelength and its integration variable must use consistent units. Do not multiply an already response-folded quantity by throughput again. The actual operator integrates on the native response/SED nodes and handles Gaussian broadening and output pixels; a single response value at a pixel center is not a substitute for this integration.

Output `flux_ujy` is a photon-equivalent bin flux density under the adopted pixel definition. It cannot unconditionally be treated as a continuous-function point value at an arbitrary wavelength. Photon accounting uses the corresponding `pixel_width/wave` weights; shared `wave_edges` and `pixel_width` specify the geometry. Losses at finite passbands, kernel windows, and detector boundaries must remain in the calculation.

The packaged instrument reference uses the central wavelengths and central resolving powers in Table 1 of [Zhang et al., arXiv:2511.06917v2](https://arxiv.org/html/2511.06917#S2). GU/GV/GI output edges are 2550–4100 / 4000–6400 / 6200–10000 Angstrom. With `instrument_lsf.mode=central_resolving_power`, `*_central_wl` and `*_Res` now determine the target Gaussian FWHM as `central_wl/Res`: approximately 11.74216 / 22.62931 / 39.13043 Angstrom.

The explicit `wavelength_law=constant_fwhm` extends that reference width within each band; the paper specifies R at the central wavelength and does not establish a constant R throughout the band. A Gaussian instrumental profile and this extension are CESS one-dimensional approximations. The paper does not resolve the LSF's pixel/PSF decomposition. Consequently, target Gaussian widths are not a certification of the final pixel-integrated, morphology-broadened resolution. The [instrument-model description](INSTRUMENT_MODEL_251106917.md) separates reference data, derivations, and retained assumptions.

Default `spectral_sampling.mode=resolution_elements` requests four pixels per reference resolution element. The operator uses `ceil(band_span/(FWHM/4))` bins and keeps the exact band edges, producing 529 / 425 / 389 bins for GU/GV/GI. Actual bin widths are approximately 2.93006 / 5.64706 / 9.76864 Angstrom; read `wave_edges` and `pixel_width` rather than equating sampling with FWHM. Default `spectral_window.mode=bandpass` selects detector/S/N and emission-line domains from the actual wavelength grid. Older external configurations without these model keys retain their fixed-`*_dlambda`, historical grid, and index-window behavior.

`template_lsf` is independent of wavelength sampling:

| mode | Meaning |
| --- | --- |
| `unknown` | The template LSF shape is unknown. The full instrumental kernel is still applied, with an explicit warning that the final resolution is not certified. |
| `delta` | Explicitly assumes that the input has no appreciable broadening. |
| `resolving_power` | An explicit Gaussian approximation and positive R value, used to match the additional broadening required in the observed frame. |
| `fwhm_rest` | An explicit rest-frame Gaussian FWHM in Angstrom. |

The packaged template model is `resolving_power` with `value=300.0`, adopting the user's explicit Gaussian R=300 approximation for this BayeSED library. This assumption was selected for CESS on 2026-10-27; it is not established by the instrument-reference paper. A different input library requires its own template-LSF setting.

The default `template_lsf_broader_policy=preserve` adds no extra instrumental broadening where a template is already wider than the target, and reports that the requested target was not attained. In particular, the adopted R=300 template is broader than the GU reference above approximately 3522.65 Angstrom. It is retained there, not deconvolved or marked as a successful match. The alternative `error` policy rejects such a calculation. Template LSF describes upstream numerical/instrumental smoothing; intrinsic astrophysical line widths must not be subtracted as a template response.

## 3. Source morphology, PSF, and aperture

The default one-dimensional morphology uses GEMS statistics and precomputed profile libraries. Distinguish original morphology parameters from the library parameters actually selected: nearest-grid selection is not continuous interpolation. PA is folded according to projection symmetry for the one-dimensional projection, while the original PA is retained. This folding cannot replace the directed PA or rotation sign required by a two-dimensional velocity field.

`spectral_extraction` provides three explicit conventions:

| mode | Source-light fraction and noise area |
| --- | --- |
| `legacy_full` | Retains the historical full-source signal approximation. Spatial-profile length enters the noise area; the old height-weight sum is not automatically interpreted as an all-source aperture fraction. |
| `explicit` | The user supplies `source_fraction` and `spatial_pixels` as explicit model inputs. |
| `mge_strip` | Computes the source fraction within a centered spatial strip and the conditional dispersion kernel from the same positive Gaussian source mixture. |

The traceability of an old library is limited by its actual provenance. Settings in a historical generator cannot automatically certify every physical property of an existing pickle. New libraries have independent manifest bindings, but mathematical consistency does not replace instrument PSF calibration. See the [morphology guide](MGE_MODELS.md) for the MGE model, configuration, and derivation.

## 4. Electron counts, noise, and wavelength errors

Source and background photon counts have Poisson fluctuations, with dark current and Gaussian read noise also included. For an equal-weight extraction of N spatial pixels and multiple exposures, background, dark-current, and read-noise variances accumulate according to the actual pixel and exposure counts. The source aperture fraction applies only to the source signal, not to uniform sky, dark current, or read noise.

After subtracting the mean background and dark current, noisy source signals may be negative; they are not clipped to zero. The current model does not additionally simulate uncertainty from an independent background estimate, cosmic rays, flat-field residuals, or general detector-correlated noise.

The default `background_mode=fixed_template` uses the packaged background spectrum and its configured scaling. RA/DEC and a background-magnitude label do not establish a zodiacal-light model constrained by observation time and solar geometry. Actual observing applications require an independent assessment of background normalization and variability.

WCE is represented by linear resampling on the nominal pixel grid. Variance propagates with squared weights, and adjacent-pixel `covariance_lag1` is stored. Under the current single-resampling convention, this supports covariance accounting for measurements across pixels; it is not a full instrument wavelength-solution inversion or a model for all correlated noise. The two `paired-wce` branches share noise: they are not independent exposures and must not be added as independent observations.

Regions with zero response may still have finite expected detector electron counts, while their response-calibrated flux cannot be finite. Read the individual definitions of `detector_model_mask`, `calibration_support_mask`, `spec_mask`, and `snr_mask`; do not collapse them into an ambiguous single “good-pixel” criterion.

## 5. Photometry, zeroth order, and emission lines

Photometry uses photon-counting response weights:

$$
\langle f_\nu\rangle_T=
\frac{\int f_\nu(\lambda)T(\lambda)\lambda^{-1}d\lambda}
{\int T(\lambda)\lambda^{-1}d\lambda} .
$$

Model magnitudes, noisy observations, statistical/systematic errors, S/N, and detection flags are stored separately. Nonpositive noisy fluxes are allowed and must not be forced into finite AB magnitudes. Photometric source-aperture fractions and pixel counts are current configuration conventions and should be assessed alongside the spectral-extraction definition.

In standard mode, `/zero_order` contains passband-integrated photometry. The historical `_0th` suffix on spectral arrays denotes an existing one-dimensional diagnostic proxy, not a real two-dimensional zeroth-order image on a CCD. Version 0.9.0 does not simulate two-dimensional spectral overlap or contamination between sources.

The emission-line module distinguishes intrinsic peaks, noisy candidates, matches, false positives, and missed peaks. Window line fluxes and S/N use an affine sideband continuum and the current covariance model. A candidate peak is not automatically an identified atomic transition. The configured 40-pixel line windows and 20-pixel sidebands remain analysis choices; their physical widths follow the actual output grid and are not calibrated by the instrument paper. Fixed windows may miss line wings or include neighboring lines; choosing centers from the same noisy data also introduces selection/centering bias. Window S/N is neither an independent detection significance nor an empirical certification of completeness or contamination.

## 6. Research use

Record the software version, scientific revision, input SHA, effective configuration, random seed, and relevant field definitions in research applications. Validate the PSF/LSF, response, background, aperture, sensitivity, and selection function for the intended study. A manifest created under an earlier scientific revision cannot resume a D5 calculation; retain old products and prepare a new output/manifest or Gravity run directory.
