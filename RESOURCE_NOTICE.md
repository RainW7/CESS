# Code license and packaged resources

[English](RESOURCE_NOTICE.md) | [简体中文](RESOURCE_NOTICE_zh-CN.md)

CESS's own source code and documentation use the MIT License in the root
`LICENSE` file. Dependencies and external resources retain their original
rights and applicable conditions; this project's MIT declaration does not
relicense them.

## Resources included in this release

On 2026-10-24, the maintainer confirmed distribution of the CSST throughput,
background template, and morphology resources currently used by CESS v0.9.0.
The maintainer describes the throughput FITS copies as the latest adopted
responses for seven CSST photometric bands and three slitless spectral bands.
The local records do not establish a single independently verifiable upstream
release number or date. “Latest” therefore does not mean an independently
verified official global version. The distributed copies are identified by
path, size, and SHA256 in `docs/RESOURCE_MANIFEST.json` in the source release.
The inventory contains 18 files: 17 calibration/morphology resources and the
default configuration.

| Resource group, relative to `src/cess/data` | Files | Purpose and recorded provenance |
| --- | ---: | --- |
| `csst_phot_tp/csst_phot.{nuv,u,g,r,i,z,y}.fits` | 7 | Adopted CSST seven-band photometric response; the maintainer confirms distribution with the package |
| `csst_sls_tp/*throughput*.fits` | 6 | Zeroth- and first-order responses for each of GU/GV/GI: three bands with six order-specific files; the maintainer confirms distribution |
| `basic_file/updated_bkg.fits` | 1 | Current background template, approved for distribution by the maintainer; this does not certify a background model varying with time or solar geometry |
| `basic_file/gems_cat.fits` | 1 | Existing GEMS morphology-statistics catalogue copy, approved by the maintainer for distribution with the morphology resources; no specific upstream release or new attribution is invented |
| `basic_file/{widthlib,heightlib}_20x20x10x10.pkl` | 2 | Precomputed libraries produced by the maintainer's original morphology generator and approved for distribution; unknown historical PSF/normalization properties retain that status |
| `emulator_parameters.json` | 1 | Default configuration and existing provenance records |

The historical filenames `GV_throughput_0th_no56.fits` and
`GV_throughput_1st_no1567.fits` are retained. Processing steps are not inferred
from those names, and no provider, official date, download URL, or data license
is invented. These statements record the maintainer's distribution decision
and currently known provenance, not independent scientific calibration.
Real BayeSED input catalogues, user simulation products, environment directories,
and development checkpoints are not packaged resources.

## Third-party software and the legacy compatibility implementation

Dependencies such as NumPy, SciPy, Astropy, h5py, and Specutils are installed
separately by the package manager and retain their own licenses. They are not
copied wholesale into CESS distributions.

The pPXF implementation formerly used in the compatibility `varsmooth` method
has been removed from the publishable source and replaced with an independently
implemented finite-interval Gaussian integral. pPXF code has not been
relicensed as MIT. The same interface name remains, but reproducing the original
implementation's boundary values is not promised; see the interface
documentation for the supported `oversample` range. This compatibility method
is not part of the current production `kernels_convolve` → `convolve_to_bins`
path. Historical copies in research directories and frozen archives remain
local for traceability and are excluded from the source distribution and wheel.

License references: [Grizli MIT](https://github.com/gbrammer/grizli/blob/master/LICENSE.txt),
[Specutils BSD-3-Clause](https://github.com/astropy/specutils),
and [pPXF upstream license](https://pypi.org/project/ppxf/9.0.2/#license).
The choice of these reference projects does not replace the CESS maintainer's
own licensing decision.
