# Physical conventions and limitations in CESS v0.9.0

[English](SCIENCE_MODELS.md) | [简体中文](SCIENCE_MODELS_zh-CN.md)

This document describes the models used by the current implementation. Parameters make the calculation reproducible; releasing the software does not make them an independently calibrated official CSST instrument model. See the [command guide](COMMANDS.md) for routine execution and the [output layout](OUTPUT_LAYOUT.md) for stored fields.

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

The upstream instrumental Gaussian FWHM is controlled by `gu_dlambda/gv_dlambda/gi_dlambda`, with packaged values of 15/21/33 Angstrom. The values of `gu_Res/gv_Res/gi_Res` do not participate in this operator; changing only those numbers does not change the simulated resolution. Output pixel spacing is also not interchangeable with these FWHM values: read the actual output grid.

`template_lsf` is independent of wavelength sampling:

| mode | Meaning |
| --- | --- |
| `unknown` | The template LSF shape is unknown. The full instrumental kernel is still applied, with an explicit warning that the final resolution is not certified. |
| `delta` | Explicitly assumes that the input has no appreciable broadening. |
| `resolving_power` | An explicit Gaussian approximation and positive R value, used to match the additional broadening required in the observed frame. |
| `fwhm_rest` | An explicit rest-frame Gaussian FWHM in Angstrom. |

The confirmed nominal R=300 of the existing templates is provenance information; it does not by itself establish that their detailed LSF is Gaussian. The packaged default remains `unknown`. Templates broader than the target are handled explicitly through `template_lsf_broader_policy=error/preserve`; silently sharpening them or clipping a negative variance to zero does not establish a successful match.

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

The emission-line module distinguishes intrinsic peaks, noisy candidates, matches, false positives, and missed peaks. Window line fluxes and S/N use an affine sideband continuum and the current covariance model. A candidate peak is not automatically an identified atomic transition. Fixed windows may miss line wings or include neighboring lines; choosing centers from the same noisy data also introduces selection/centering bias. Window S/N is neither an independent detection significance nor an empirical certification of completeness or contamination.

## 6. Research use

Record the software version, input SHA, effective configuration, random seed, and relevant field definitions in research applications. Validate the PSF/LSF, response, background, aperture, sensitivity, and selection function for the intended study.
