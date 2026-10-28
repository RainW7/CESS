# Instrument reference and one-dimensional spectral model

[English](INSTRUMENT_MODEL_251106917.md) | [简体中文](INSTRUMENT_MODEL_251106917_zh-CN.md)

CESS `0.9.0`, scientific revision `20261027-D5`, uses the reference below for its packaged GU/GV/GI configuration. The wavelength limits and central resolving powers come from Table 1 of [Zhang et al., *Mock Observations for the CSST Mission: Main Surveys – the Slitless Spectroscopy Simulation*, arXiv:2511.06917v2](https://arxiv.org/html/2511.06917#S2). They are design/simulation reference values, not measurements of an operating instrument.

## Reference values and derived widths

| Band | Lower edge (Å) | Upper edge (Å) | Central wavelength (Å) | R at the central wavelength | Derived reference FWHM (Å) |
| --- | ---: | ---: | ---: | ---: | ---: |
| GU | 2550 | 4100 | 3370 | 287 | 11.74216028 |
| GV | 4000 | 6400 | 5250 | 232 | 22.62931034 |
| GI | 6200 | 10000 | 8100 | 207 | 39.13043478 |

The FWHM column is calculated by CESS as `central_wl/Res`; it is not a separate tabulated measurement. The paper defines R at the central wavelength. It does not supply a complete wavelength-dependent LSF, a Gaussian profile measurement, or a separation between optical PSF and pixel response.

The relevant configuration fragment is shown below. It must be part of a complete instrument JSON; it is not a standalone configuration file. See the [user guide](USER_GUIDE.md#5-configuration-resources-and-paths) for resource-path handling.

```json
{
  "instrument_lsf": {
    "mode": "central_resolving_power",
    "wavelength_law": "constant_fwhm"
  },
  "spectral_sampling": {
    "mode": "resolution_elements",
    "pixels_per_resolution_element": 4.0
  },
  "spectral_window": {"mode": "bandpass"},
  "gu_central_wl": 3370.0,
  "gu_Res": 287.0,
  "gu_min_wl": 2550.0,
  "gu_max_wl": 4100.0,
  "gv_central_wl": 5250.0,
  "gv_Res": 232.0,
  "gv_min_wl": 4000.0,
  "gv_max_wl": 6400.0,
  "gi_central_wl": 8100.0,
  "gi_Res": 207.0,
  "gi_min_wl": 6200.0,
  "gi_max_wl": 10000.0,
  "template_lsf": {"mode": "resolving_power", "value": 300.0},
  "template_lsf_broader_policy": "preserve"
}
```

## Instrument LSF and sampling are separate choices

`instrument_lsf.mode=central_resolving_power` makes `*_central_wl` and `*_Res` operative. The explicit `wavelength_law=constant_fwhm` extends their derived width across each band. The target is a Gaussian response before output-pixel integration. Thus the model's target R varies with wavelength even though the reference FWHM is constant. This is an adopted one-dimensional approximation; neither a constant R across the band nor a full detector-position-dependent optical model is inferred from three central R values.

The loader writes the derived widths to `*_dlambda` in the effective configuration for existing consumers. In this mode those fields are derived values, not independent controls. Change the reference fields and declared model rather than editing only a cached width.

The reference paper describes a locally linear wavelength–trace relation, with coefficients varying across the focal plane (Section 3). Section 4 uses an illustrative four-pixel resolution element and a 0.074 arcsec/pixel spatial scale. CESS adopts four samples per reference FWHM as a separate sampling convention. It does not load the paper's optical trace coefficients or establish a measured dispersion from the spatial pixel scale.

For each band, CESS sets

$$
N=\left\lceil\frac{\lambda_{\max}-\lambda_{\min}}{\mathrm{FWHM}_{\mathrm{ref}}/4}\right\rceil
$$

and distributes N bins uniformly between the exact outer band edges. Pixel centers are the bin midpoints.

| Band | Bins | Actual bin width (Å) |
| --- | ---: | ---: |
| GU | 529 | 2.93005671 |
| GV | 425 | 5.64705882 |
| GI | 389 | 9.76863753 |

Rounding N upward makes the actual spacing slightly finer than `FWHM_ref/4`. Read the product's `wave`, `wave_edges`, and `pixel_width`: the first and last centers lie inside the quoted outer limits. Pixel integration and source morphology can broaden the response beyond the declared pre-pixel target.

`spectral_window.mode=bandpass` determines detector/S/N and emission-line analysis domains from the actual supplied wavelength centers. Old `*_init_idx/end_idx` values are not independent window controls in that mode. The configured 40-pixel line-measurement windows, 20-pixel sidebands, and detection thresholds remain separate analysis parameters. Their wavelength widths change with the output grid and are not sensitivity or extraction calibrations from the reference paper.

## Input-template response

The packaged `template_lsf` adopts a Gaussian R=300 approximation for the intended BayeSED input library. This is an input-library assumption, recorded separately from the instrument reference. It is not a result of the CSST paper and must be reassessed for a different library.

Within this Gaussian model, the extra observed-frame instrumental width is

$$
\mathrm{FWHM}_{\mathrm{extra}}(\lambda)
=\sqrt{\mathrm{FWHM}_{\mathrm{ref}}^2-(\lambda/300)^2}
$$

where the expression is real. It subtracts the declared upstream smoothing variance, not an astrophysical velocity dispersion or intrinsic emission-line width.

In GU, the template exceeds the target width above about **3522.65 Å**. Default `template_lsf_broader_policy=preserve` adds zero extra broadening there, retains the input width, and warns that the target was not attained. It does not sharpen the input. Choose `error` to reject such a case; obtaining a narrower response requires a sufficiently high-resolution input. A known template model removes the `unknown`-LSF warning but does not suppress the distinct overbroad-template warning.

The other supported input modes remain `unknown`, `delta`, and `fwhm_rest`; their definitions are in [scientific models](SCIENCE_MODELS.md#2-response-integration-pixels-and-resolution). Wavelength sampling alone does not determine any of these template responses.

## Retained resource and observing conventions

Throughput FITS, GEMS statistics, stored morphology profiles, and the sky spectrum retain their existing contents. The grating-only efficiency requirements in the paper are not replacements for a total throughput that includes optics, filters, grating efficiency, and detector QE. The existing zeroth-order FITS are not rescaled to a fixed fraction of first order.

The collecting radius remains 100 cm, spectral exposures remain 150 s × 4, and the spatial scale remains 0.074 arcsec/pixel. Dark current, read noise, extraction aperture, sky normalization, and photometric/zeroth-order exposure designs are unchanged. The paper's single-150-s demonstration does not prescribe the number of exposures for a CESS campaign. Its R80 quantity is not assigned to a Gaussian PSF FWHM or used to reinterpret existing morphology libraries.

An overall Gaussian LSF before pixel integration does not settle whether a published resolution includes a particular PSF contribution. Keep explicit PSF/LSF conventions for advanced morphology models; do not add another dispersion PSF merely from a tabulated R. The effective resolution and sensitivity remain conditional on the template, instrument, pixel, morphology, extraction, and noise assumptions.

## Existing configurations, products, and resume

External configurations without the new model keys retain the earlier `fixed_fwhm` instrument, `legacy` sampling, and `indices` window contracts. Under that instrument mode, `*_dlambda` controls the width and old `*_Res` entries remain descriptive. The new packaged defaults are selected explicitly; loading an old configuration does not silently convert it to the reference model.

The output schemas remain structured 3.0 or compatible 2.5. The changed model is identified by scientific revision `20261027-D5` and recorded configuration/provenance, with actual wavelength geometry stored in each product. Preserve that information when comparing products: matching software version `0.9.0` alone does not establish matching science settings.

Retain existing products. For a D5 recalculation use a new output directory and a new manifest; an older scientific revision cannot be resumed under the changed code. Gravity campaigns must be prepared again in a new run directory, then resumed only with their frozen inputs, configuration, resources, and software identity. See the [Gravity array guide](../scripts/gravity/ARRAY.md).
