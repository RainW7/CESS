# Native HDF5 3.0 output and compatibility boundaries

[English](OUTPUT_LAYOUT.md) | [简体中文](OUTPUT_LAYOUT_zh-CN.md)

This document applies to CESS v0.9.0. Schema 3.0 organizes storage and access;
compatible schema 2.5 output remains available. Read parameter tables and
spectral fields by name, units, `axis_order`, and `definition` attributes.
See the [user guide](USER_GUIDE.md) for the interpretation of common quantities.

| 3.0 location | 2.5 origin and meaning |
| --- | --- |
| `/catalog` | IDs, source/parent rows, parameter tables, `data_mask`, `input_quality_flags`, and per-source extraction/MGE diagnostics |
| `/spectra/GU`, `GV`, `GI` | Wavelength, flux, electron counts, errors, S/N, response, masks, and lag-1 covariance from the original band groups; leaf names are retained |
| `/photometry` | Complete seven-band model/observed/error/SNR/status data when enabled |
| `/zero_order` | Standard-mode passband-integrated zeroth-order photometry; the historical `_0th` spectral proxy is not relabeled as a real zeroth-order spectrum |
| `/lines/GV`, `GI` | Matched peaks, false-positive/missed-peak diagnostics, and window line measurements; old EL-only root `gv_*`/`gi_*` fields move here without their band prefix |
| `/provenance` | Original root attributes, complete run/config/code/resource/random-stream identities, and the layout mapping |

All six root groups exist. `cess_layout_status` explicitly records `present`,
`disabled`, or `not_applicable`. For N=0, shared wavelengths and correctly
shaped `(0,W)` arrays are retained. EL-only products do not copy the parent's
large spectral arrays. GU/GV/GI may have different W. Spectral wavelengths are
in Angstrom and flux/errors are generally in microJy. Read each field's units
and definition to distinguish electron rates from electron counts; its name
alone is insufficient.

Root attributes declare the physical schema-3.0 envelope, software version,
mode, and product kind. `/provenance.attrs['schema_version']='2.5'` preserves
the semantic identity of the original scientific payload; it does not
redeclare the file as 2.5. Strict resume checks both the envelope and payload.
Original root-attribute dtypes/shapes are preserved, as are spectral and
catalogue dataset attributes.

`mapping_json` in `/provenance/conversion` explicitly maps original to stored
paths. Native writing first produces scientific data inside an unpublished
temporary file, then moves HDF5 objects. It does not copy a complete second
set of N×W arrays or publish a 2.5 file before modifying it.
`operation='native_layout_adapter'` does not invent an intermediate-file SHA.
Independent conversion of an old file uses
`operation='legacy_file_conversion'` and records the actual source path, SHA,
and conversion block size.

Unknown legacy fields are preserved under `/provenance/extensions` according
to the mapping. A length equal to N does not by itself establish a source axis.
Storage that cannot be reliably remapped, including external links, soft
links, and object references, is explicitly rejected rather than silently
dropped. `validate_structure(...)` checks the six root groups, statuses,
headers, complete mapping coverage, duplicates/conflicts, and metadata
consistency. It neither reads all spectral arrays nor certifies the physical
models.

`open_product(...)` provides one read-only interface for 2.5/3.0:
`catalog`, `band(...)`, `line_group(...)`, `parameter_table(...)`, and
`read_row(...)`. `legacy_view()` exposes old logical paths for existing
algorithms and returns actual h5py Datasets. EL peak tables can be reached
through their original root names and convenient band names, with each
object visited once. This is logical access without duplicating old paths
on disk. Use these objects within the `open_product` context.

`SourceSelection` slices only declared source fields. Source/parent rows,
parameter tables, spectral arrays, extraction diagnostics, and MGE
diagnostics are sliced along their source axis. Shared wave/response/bin
geometry and sparse/ragged line tables are not assigned axes merely from
coincidentally equal lengths. ID dtype and encoding are retained; row
correspondence does not require unique IDs.

A manifest binds output layout, actual input schema, configuration, seed,
code/resource/input SHA, row interval, and expected output path. After its
envelope has been validated, a 3.0 product must also pass the full scientific
structure checks. Layout substitution, missing covariance, incorrect MGE
rows, or a damaged mapping cannot be concealed by the compatibility view.
Existing output paths, including symbolic links, are rejected. The program
does not automatically repair, overwrite, or adopt unverified historical
results.

Sampled products retain `extracted-mock-spectra-1.1`; they are not relabeled
as complete schema-3.0 simulations. Selection retains the requirement that
both GV/GI mean S/N are at least 3, followed by redshift `qcut` stratification.
Available errors, masks, covariance, and row provenance are preserved;
missing fields have availability indicators. Parent metadata and shared
geometry are stored once per parent file. Missing parent measurements are
not fabricated.

See the [user guide](USER_GUIDE.md) for commands. Batch tasks should use an
explicit output layout, source interval, and run manifest.

`cess merge -i chunks -o merged/complete.hdf5` combines contiguous chunks
from the same simulation into one self-contained HDF5. It preserves the
input layout and original simulation identity without recomputing scientific
quantities. Copying uses finite row blocks; only local source-row references
in sparse line tables receive the merged row offset. Shared arrays such as
wavelength and response are stored once. `--copy-rows` limits each copying
block; the final integrity check still reads line tables, so memory use grows
with the number of peaks. Chunk intervals must be contiguous and
nonoverlapping, with matching configuration, seed, generation version, and
shared fields. `--input-source input.hdf5` additionally verifies the original
input SHA, complete coverage, and row-by-row IDs.

In schema 3.0, merge history is preserved through the original-layout mapping
at `/provenance/extensions/root/merge_history`; schema 2.5 uses root
`merge_history`. It retains chunk provenance and original attributes. The
original run manifest continues to describe the original chunks; the merged
product does not impersonate an original task's output and does not depend
on external chunk links or VDS.

Per-source `catalog/input_quality_flags` is a uint32 bitmask. Zero means the
source passed the current input safety checks; a nonzero value marks an invalid
source and corresponds to `data_mask=2`. Its `flag_masks`/`flag_meanings`
attributes describe each bit, while `policy_json` records ranges, units,
and handling policy. An older product without this field has no recorded
check: absence does not mean that its sources passed. Derived EL products
inherit the flags and their attributes through source-row selection.
