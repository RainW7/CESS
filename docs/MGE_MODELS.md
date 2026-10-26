# Positive morphology libraries and joint strip extraction

[English](MGE_MODELS.md) | [简体中文](MGE_MODELS_zh-CN.md)

This guide applies to CESS v0.9.0. This is an optional one-dimensional model: default runs use the packaged legacy library and do not require generating an MGE library first. See [scientific models](SCIENCE_MODELS.md) for execution and unit conventions.

## 1. Generate a small source basis

`cess-mge-library` creates a `cess.positive_marginal_library` v1 bundle, with `manifest.json` and `marginals.hdf5` in each new directory. It uses a positive Gaussian approximation to an untruncated Sérsic source, a circularized Re, and an area-preserving elliptical transformation; the declared PSF is circular Gaussian. This format differs from both the old pickle libraries and the legacy bundle created by `cess-profile-library`.

The following only demonstrates the interface. With the CESS environment activated, create a separate working directory outside the source root and run:

```bash
mkdir -p /path/to/mge-work
cd /path/to/mge-work
python - <<'PY'
import json
from pathlib import Path
from cess.config.loader import load_config
config = load_config()
config['morphology_psf_fwhm'] = 0.2625  # arcsec; adopted example, not instrument calibration
Path('config.json').write_text(json.dumps(config, indent=2)+'\n')
grid = {'n': [1.0], 're_pixel': [4.0], 'pa_deg': [0.0, 45.0], 'axis_ratio': [0.5, 1.0]}
Path('grid.json').write_text(json.dumps(grid, indent=2)+'\n')
edges = {'x_edges_pixel': [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]}
Path('edges.json').write_text(json.dumps(edges, indent=2)+'\n')
PY
cess-mge-library --config config.json --grid grid.json --edges edges.json \
  --output basis --dry-run
```

After inspecting the plan, remove `--dry-run` to fit and write the bundle. Its parent directory must exist and `basis` must not already exist. The narrow window and sparse grid in this example are for a small interface check, not a recommendation for production morphology coverage.

The four axes `n/re_pixel/pa_deg/axis_ratio` must be finite, nonempty, and strictly increasing; 0.5≤n≤5, Re>0, and 0<q≤1. PA is measured in degrees from the image x axis. Up to 100000 Cartesian profiles and 1–2049 bins per direction are supported. Edges are relative to source center zero and may be nonuniform; omitted y edges are copied from x. The pixel scale comes from `arcsecperpix` and the PSF from `morphology_psf_fwhm`. If the latter is omitted, the generator adopts and records 0.2625 arcsec; this is not a measured CSST PSF.

Default numerical controls are `--l1-tolerance 1e-7 --tail-probability 1e-12 --max-components 160`. A fit that fails the requirements is rejected; the code does not automatically relax precision, clip negative values, or renormalize a finite window. Failed staging is retained for inspection, and chunked resume is not provided.

Use the current Python package interface for read-only verification:

```bash
python - <<'PY'
from cess.morphology.bundle import verify_mge_bundle
print(verify_mge_bundle('/path/to/mge-work/basis'))
PY
```

Verification checks byte SHA, structure, units, row order, source coefficients, and nonnegative probability accounting. It does not refit, open external source paths recorded in the manifest, or certify the real PSF. See the [bundle format](MGE_BUNDLE_FORMAT.md) for the full specification.

## 2. Explicitly enable mge_strip

In a copy of the complete runtime configuration, remove the old `widthlib_pkl` and `heightlib_pkl` entries and set:

```json
{
  "positive_profile_library_manifest_path": "/path/to/mge-work/basis/manifest.json",
  "positive_profile_library_path": "/path/to/mge-work/basis/marginals.hdf5",
  "spectral_extraction": {
    "mode": "mge_strip",
    "spatial_pixels": 9,
    "kernel_radius_pixels": 16,
    "lsf_convention": "pre_psf"
  }
}
```

This fragment must be merged into a complete JSON configuration; it is not a standalone runnable configuration. Runtime PSF FWHM and pixel scale must exactly match the library declaration. Do not also supply the old `profile_library_manifest_path`. Supported integer ranges are N=1–1001 spatial pixels and K=0–1024 dispersion radius, giving a kernel length of 2K+1; do not provide a manual `source_fraction`.

The LSF role must be explicit: `pre_psf` assumes the upstream LSF excludes the point-source PSF and adds the specified PSF in both x and y; `point_source` assumes the dispersion PSF is already included upstream and adds it only in y. The example's `pre_psf` is a working assumption, not a real instrument property inferred by the software.

Start with a few sources:

```bash
cess simulate --config /path/to/runtime.json -i /path/to/catalogue.hdf5 \
  -o /path/to/new_mge_results --source-stop 4 --all-effects --seed 1214
```

Modern output uses schema 3.0, with per-source diagnostics in `/catalog/spectral_extraction` and `/catalog/mge_morphology`; legacy 2.5 uses corresponding root groups. Original sampled morphology parameters are retained, and selected grid parameters are recorded separately. Per-axis nearest selection, endpoint selection, and one-dimensional PA folding are recorded explicitly. Nearest-grid quantization differences are not included in the source-fit error.

## 3. Joint probabilities and electron accounting

For a circular Gaussian source component of scale s, the covariance after an area-preserving elliptical transformation is S, with det(S)=s⁴. Let the PSF standard deviation be p=FWHM/(pixel scale×sqrt(8ln2)). The two roles use

$$
C=S+\mathrm{diag}(0,p^2)\quad(\mathrm{point\_source}),\qquad
C=S+\mathrm{diag}(p^2,p^2)\quad(\mathrm{pre\_psf}).
$$

For a centered spatial strip and whole-source-normalized weights w, combine the full mixture probabilities before conditioning:

$$
F=\sum_k w_kP_k(y\in\mathrm{strip}),\quad
J_j=\sum_k w_kP_k(x\in\mathrm{bin}_j,y\in\mathrm{strip}),\quad
\mathrm{width}_j=J_j/F.
$$

Do not condition each component separately and then retain its original weight. `height` contains absolute spatial-bin probabilities integrated over all x, summing to approximately F. Two independent marginals cannot be multiplied to obtain this joint rectangle probability.

Retain the finite dispersion-window mass M=sum(width), without dividing by M to renormalize. Apply F to the expected source electrons once, then redistribute with the unnormalized width; the total retained fraction is F×M. Sky, dark current, and read noise use N spatial pixels and are not multiplied by F. Disabling the morphology effect only replaces the dispersion width with a delta, while retaining F and the noise area.

For each component, the Gaussian probability of X|Y is integrated with a positive integrand. Conditional variance is computed from positive terms in det(C)/Cyy to avoid subtracting nearly equal values. Separable limits are evaluated analytically; other cases use deterministic integration. `quadrature_error_estimate` is the sum of component max-bin error estimates in conditional units, not a full-kernel L1 or total-flux error bound. Gaussian tail omission is recorded separately; integration warnings or nonconvergence cause explicit failure.

## 4. Scientific limitations

The model assumes separable SED and spatial distributions, achromatic morphology, a centered strip, the declared circular Gaussian PSF, and a separable upstream LSF. It does not include a velocity field, arbitrary two-dimensional apertures, or a complete curved trace.

Discrete redistribution of an already pixel-integrated spectrum using integer source-offset bins is an explicit quantization approximation. It is not strictly equivalent to convolving a continuous source and LSF before a single integration over detector pixels. Source-fit L1 and `2*estimated_L1/F` are numerical error indicators, exclude grid-selection differences, and can be amplified by conditioning on a small F. Production applications require a traceable PSF/LSF choice, appropriate grid, strip, and kernel window, plus the corresponding scientific validation.
