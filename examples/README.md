# Example spectra

English | [简体中文](README_zh-CN.md)

| File | Contents |
| --- | --- |
| [toy_input.hdf5](toy_input.hdf5) | Four synthetic, BayeSED-shaped galaxy spectra |
| [toy_output.hdf5](toy_output.hdf5) | Their simulated GU/GV/GI spectra, photometry and line measurements; HDF5 schema 3.0 |
| [make_toy_input.py](make_toy_input.py) | Script for generating another small input file |

The SEDs and galaxy parameters are arbitrary interface examples, not a calibrated physical population. Row 0 deliberately contains an invalid SED. Its ID and row are retained with `data_mask=2`; rows 1–3 contain simulated spectra.

The bundled result uses scientific revision `20261027-D5` and the [declared instrument profile](../docs/INSTRUMENT_MODEL_251106917.md), with 529/425/389 GU/GV/GI bins. The synthetic input is an interface example, not a calibrated template library.

## Open the supplied files

After [installing CESS](../docs/INSTALL.md), run from the repository root:

```bash
cess-viewer examples/toy_output.hdf5 --lang en
```

Choose source row **1** to view a valid spectrum. To inspect the input instead:

```bash
cess-viewer examples/toy_input.hdf5 --lang en
```

The viewer requires the `viewer` extra and working Tcl/Tk; see the [viewer guide](../docs/VIEWER.md).

## Recompute the example

Use the supplied input and a new output directory:

```bash
cess simulate -i examples/toy_input.hdf5 -o toy_results_new --all-effects --seed 1214
```

The result is `toy_results_new/CSST_grism_toy_input_4.hdf5`. All four rows are kept in one file. The example uses standard mode, the packaged default configuration, and all four optional effects. Existing outputs are not overwritten; use another directory for a fresh run.

The bundled output has repository-relative provenance paths for portability. Its scientific arrays, masks, IDs and seed are unchanged from the generated product. Recomputing creates provenance for your own environment; the bundled file is not a resume checkpoint.

To create a different small input:

```bash
python examples/make_toy_input.py toy_input_new.hdf5 --sources 4 --invalid-first
```

Changing the input file changes its content identity and can change random streams even with the same seed.
