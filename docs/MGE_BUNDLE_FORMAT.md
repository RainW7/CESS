# Positive marginal bundle format, version 1

This document describes the format implemented by CESS v0.9.0 in
`cess.morphology.generator` and `cess.morphology.bundle`. Creation and use are
explained in the [MGE guide](MGE_MODELS.md). This is an experimental source
approximation with explicit assumptions, not an instrument calibration.

## Files and declaration

A complete bundle contains exactly `manifest.json` and `marginals.hdf5`.
Publication creates a new directory without replacing an existing one. Failed
staging is retained. Code/config identities are checked before publication.

The manifest declares `schema='cess.positive_marginal_library'`, integer
`version=1`, `status='complete'`, generator revision, UTC creation time, software
versions, source hashes, effective config/hash, adopted model, controls and
limitations. The generator revision is an algorithm identity; its historical
label is independent of the current package version.

Axes are `n`, `re_pixel`, `pa_deg`, `axis_ratio`, with units respectively
dimensionless, pixel, degree and dimensionless. Values are finite, nonempty,
strictly increasing; 0.5≤n≤5, Re>0, 0<q≤1. The shape is their Cartesian product
in C order, q varying fastest, with 1–100000 profiles. Bin edges are finite,
increasing lists relative to source center zero, 1–2049 bins per direction.

The model records positive Gaussian source approximation to untruncated Sérsic,
circularized half-light radius under area-preserving shear, image-x PA,
positive pixel scale and adopted circular-Gaussian PSF FWHM. Controls record
the requested numerical L1 estimate tolerance, tail probability and component
limit. These estimates do not certify a continuum or instrument error bound.

`quantity='absolute unit-source probability per bin'`,
`orthogonal_domain='all real'`, and `window_renormalized=false` distinguish this
format from legacy conditional profiles. The member SHA256 and size bind the
HDF5 bytes. Config and generation source identities are recorded without
requiring those original paths to exist when the bundle is verified later.

## HDF5 layout

Root keys are exactly `axes`, `bins`, `profiles`, `fits`. Shared root attrs
mirror schema, version, completion status, generator revision, axis order,
quantity, orthogonal domain, window normalization, model and controls.

| Dataset | Type and shape |
| --- | --- |
| `axes/{n,re_pixel,pa_deg,axis_ratio}` | float64 vectors matching manifest axes |
| `bins/{x_edges_pixel,y_edges_pixel}` | float64 bin-edge vectors |
| `profiles/row_indices` | int64 `(N,4)`, exact Cartesian C-order indices |
| `profiles/x_probability` | float64 `(N,nx)` |
| `profiles/y_probability` | float64 `(N,ny)` |
| `profiles/window_probability` | float64 `(N,2)`, x/y finite-window mass |
| `profiles/outside_probability` | float64 `(N,2)`, independently integrated outside mass |
| `profiles/source_fit_l1_estimate` | float64 `(N,)`, associated n-fit estimate |
| `fits/n_000000/...` | one group per n-grid index, with n/metadata attrs and 1-D `sigma_over_re`, `weights` |

Source coefficients are positive and normalized over the whole source plane.
Each n fit is stored once. Probability entries are finite and nonnegative;
window and outside mass sum to approximately one within the declared 1e-9
floating-point accounting tolerance. Finite windows are not renormalized.

## Verification and runtime

`verify_mge_bundle(directory)` reads only the two declared regular files. It
checks hashes, exact schema/structure, dtypes, units, shapes, row order, fit
links, positive coefficients and probability accounting in row blocks. Extra
members, unsupported links/VDS/external storage, malformed declarations and
invalid numeric values are rejected. It does not re-render, refit, unpickle
legacy resources or certify the declared PSF.

The `mge_strip` runtime reads the verified source coefficients to construct a
joint strip kernel. It does not multiply the stored x/y marginals to create a
2-D aperture and does not treat them as legacy kernels. Runtime source-basis
paths, PSF/scale agreement, selected grid and LSF role are independently bound
in simulation provenance.
