# CESS HDF5 viewer

English | [简体中文](VIEWER_zh-CN.md)

The optional desktop viewer opens an HDF5 file, shows its groups, datasets and
attributes, and plots spectra or catalogue variables. It is useful for checking
simulation outputs without writing a separate analysis script. It reads the
input file in read-only mode; figures and CSV tables are separate exports.

The viewer complements the [user guide](USER_GUIDE.md) and
[output-layout reference](OUTPUT_LAYOUT.md). Inspect the dataset definitions,
units and masks before interpreting a plot. A successful display does not
validate a simulation's scientific calibration.

## 1. Install the optional interface

Use the CESS Python 3.12 environment. From the downloaded or extracted CESS
source root:

```bash
conda activate cess
cd /path/to/CESS
python -m pip install '.[viewer]'
python -m pip check
```

The `viewer` extra installs Matplotlib; NumPy and h5py are already core CESS
dependencies. The viewer uses Tkinter and the Matplotlib TkAgg backend.
Tkinter is supplied by the Python/Tcl/Tk installation, **not a pip package**.
For a Conda environment, install or check Tk in that same environment:

```bash
conda activate cess
conda install tk
python -m tkinter
```

The last command should open a small Tk test window on a graphical desktop.
Close that window before continuing. If `_tkinter` is absent from a venv's
base Python, install a Python distribution with Tk support and recreate the
venv; installing a similarly named pip package is not a replacement.

The interface is optional. A batch-only CESS installation and Gravity jobs do
not need Tk or a viewer window. Use a computer with a graphical desktop for
interactive viewing. See [installation](INSTALL.md) for the core environment.

## 2. Launch and choose a language

Run from any working directory, using an absolute path to the product:

```bash
cess-viewer /path/to/results/product.hdf5
```

The default interface language is English. Select Chinese at startup:

```bash
cess-viewer /path/to/results/product.hdf5 --lang zh
```

Start without a filename to open an empty interface, then choose **File →
Open HDF5 file**:

```bash
cess-viewer
```

Equivalent module and source-checkout entry points are:

```bash
python -m cess.viewer /path/to/results/product.hdf5 --lang en
python /path/to/CESS/cess_h5_viewer.py /path/to/results/product.hdf5 --lang en
cess-viewer --help
```

The checkout wrapper prefers its local source tree; its runtime dependencies must still be installed. Paths
containing spaces must be quoted. The **Language** menu switches the interface
between English and Chinese; the environment variable
`CESS_H5_VIEWER_LANG=en` or `CESS_H5_VIEWER_LANG=zh` also sets the startup
language. An explicit `--lang` takes precedence.

## 3. Explore a small example

In the activated environment, use a new writable working directory and fresh
output paths. Replace `/path/to/CESS` with your source directory:

```bash
python /path/to/CESS/examples/make_toy_input.py toy.hdf5 --sources 4 --invalid-first
cess simulate -i toy.hdf5 -o toy_results --all-effects --seed 1214
cess-viewer toy_results/CSST_grism_toy_4.hdf5 --lang en
```

The example contains arbitrary spectra for interface checks. Row 0 is
deliberately invalid: its ID and source mapping are retained, but its simulated
spectra should not be treated as valid measurements. Move to rows 1–3 to browse
the other example spectra. This example is not a physical calibration dataset.

## 4. Workspace pages

The workbench uses a graphite navigation rail, a light workspace and rounded primary buttons.
Switch pages from the left rail. The header shows the current file and summary;
Save figure / Export data become available on the corresponding plot pages.
An empty workspace shows a start page. The initial window fits the screen, spectrum
controls scroll locally, and Filters & display folds away without clearing its settings.
Hover over the filename for its full path or the quantity selector for its dataset field.
The window size, divider positions, page size and the last opened file are stored
in `~/.config/cess-viewer/settings.json` (`$XDG_CONFIG_HOME` is honoured) and
restored on the next start; delete that file to return to the defaults.

| Page | Main use |
| --- | --- |
| Data browser | Browse the HDF5 hierarchy or filter paths; inspect shape, dtype, attributes and table-column definitions; preview, sort and export rows and select plotting variables |
| Spectrum | Select a product row or find an ID; display available GU/GV/GI spectra, model, uncertainty, masks and S/N; overlay several sources, inspect line markers and export a source table |
| Explore plots | Select numeric variables for scatter, line, step, histogram or hexbin plots; optionally choose a color variable, Y errors and a numeric expression filter |
| Guide | Review layout, quantities and interface controls |

### Data browser

Select a group or dataset in the left-hand tree. The information panel reports
stored metadata; the preview controls page through data instead of displaying
an entire large table at once. Named columns are preferred where metadata is
available. Structured arrays expose their numeric fields.

The information panel, the column list and the preview are separate panes: drag
the dividers between them to give a long attribute report more room, or shrink
it when you want more rows of data. **A-** / **A+** change the information text
size and **Copy** puts the whole report on the clipboard. **Go to row** jumps the
preview to the page containing a row of a long dataset, so you do not have to
page through it, and the page selector switches between 200, 500, 1000 and 2000
rows per page.

Clicking a column header sorts the dataset by that column across all rows (the
row numbers stay the original ones); clicking again reverses the order and
clicking the first column clears the sort. **Export page** writes the rows you
are looking at to CSV with the same protection as the other exports.

Right-click a tree node for **Copy path**, **Set X / Set Y / Set color**
(whole dataset, or the selected column) and **Expand / Collapse**. Right-click a
preview row to open that source in the Spectrum tab or copy the row.

Select a column and use **Set as X** or **Set as Y** to send it to the plotting
tab. The two-column action uses the first selected column as X and the second
as Y. These operations select existing values; they do not recalculate a
physical quantity.

### Spectrum

Use the source-row control, slider or arrow buttons to change the displayed
row. Rows are zero-based **within the opened product**. For a subset or chunk,
read `input_row_indices` and any parent mapping to recover the original
catalogue row. An ID is a separate identifier, not necessarily that row number.

Choose available bands and the desired spectral quantity. For the simulated
observation, use `flux_ujy_with_noise`; `flux_ujy_model` is its noiseless model.
`flux_ujy` and electron-count/rate fields have other definitions. In particular,
`source_electron_rate_pre_morph` has unit electron s^-1 cm^-2. Compare an
uncertainty with the quantity and unit it belongs to. Negative background-
subtracted fluctuations are valid; the symmetric-log display can retain them.

**Add to compare** keeps up to four source rows and draws them as thin extra
lines in the same figure, so spectra at different brightness or redshift can be
compared in place; double-click an entry to remove it. The compared rows use the
same quantity and bands as the main curve.

Two options control how they are drawn. **Dashed** renders the compared rows with
a dashed line, which keeps them distinguishable when colours are close. **Normalise**
divides every curve — the main one included — by its own median flux or by its
own median 1σ, so spectra that differ in brightness can be compared by shape; the
factor is stated in the title and on the y axis. Neither option smooths, shifts or
resamples the data, and neither touches the stored values or the exports (exports
from this tab always write the unnormalised spectrum). Normalised curves often
contain spikes a hundred times the continuum, so their y range is deflated to the
1–99 % quantiles and the status bar says so; zoom, the symmetric-log axis or a
different range restores the full view.

The combined-band view places the band curves on a common figure. It does not
perform a new flux calibration, combine overlapping pixels with a statistical
estimator, or fill wavelength gaps. Keep each band's identity and masks when
assessing overlap regions.

Reference markers and truth-centered measurement-window positions come from
stored reference information. The detected-peak overlay shows **matched noisy
peaks**, which can contain several peaks for one source; it is not a display of
every candidate or false-positive peak. A window at a known reference position
is not an independently detected or identified atomic transition, and a peak
marker is not a validated redshift. See [scientific models](SCIENCE_MODELS.md).

For `paired-wce` products, the Spectrum tab displays the base branch without
WCE. Use Explore plots and explicitly select the `*_wce` fields to inspect the WCE
branch. The two branches share a noise realization.

Historical `_0th` spectral arrays are diagnostic proxies on a first-order
grid. They are not a rendered zeroth-order image or the physical zeroth-order
passband integral. Modern passband measurements are stored under `zero_order`.

### Explore plots

Choose X and Y from the registered dataset variables. A two-dimensional
source-by-pixel spectrum can be viewed as a pixel column across sources. When
paired with the corresponding wavelength vector, the source-row control is
used to view an individual spectrum. Always check the resulting axis labels
and lengths: unrelated arrays of equal length do not automatically describe
the same sources.

Optional numeric filters use `x` and `y`. For example:

```text
(x > 22) & (y > 20)
```

Use parentheses around each condition and `&` / `|` for combined array
conditions. Filters accept a restricted mathematical expression with `x`, `y`
and whitelisted NumPy functions, such as `np.isfinite(x)`, `np.abs(y)` or
`np.log10(x)`; they do not execute arbitrary Python. The same finite-value and
expression filtering applies to every plot type. Choose the unused option to
clear a color or Y-error variable.
A robust plotting range can hide outlying points from the displayed limits;
this is a display setting, not a new quality flag or catalogue selection.
Inspect the status message when comparing a plot with the full sample.

The plot toolbar provides zoom, pan and save controls. Subplot configuration
also accepts numerical margins and spacing.

## 5. Save figures and export tables

Use **File → Save current figure** on the Spectrum or X-Y tab to create a
PNG, PDF or SVG. Figure saving affects the displayed plot, not the HDF5 file.

The Spectrum tab exports the current source's selected bands as a CSV table.
The table includes band labels, wavelengths and the displayed spectral fields
and associated columns. Read its headers and retain the band's identity.
It is a convenient inspection export, **not a self-contained replacement for
the CESS product**: it does not preserve all provenance, wavelength edges,
covariance, flags or optional measurements. Use the original HDF5 and the
CESS reader for scientific measurements or reproducible product exchange.

The X-Y tab exports the **last plotted data snapshot**, including its finite-
value and expression filters. For scatter, line and step plots, the CSV includes selected color and
Y-error columns as well as X/Y; a one-dimensional histogram exports X only.
A hexbin export contains the input points, not computed bin counts. Robust
display limits and logarithmic axes change the view without deleting values
from this export. Numbers retain full precision and CSV quoting is applied.
Check the headers before reusing a table elsewhere.

Save to a separate destination and retain the original HDF5. The viewer rejects
CSV export destinations with HDF5 suffixes and protects the currently opened
input file, including an alias through a symbolic or hard link.

On macOS, shortcuts include Cmd-O to open, Cmd-S to save the current figure,
Cmd-E to export, and Left/Right to change the source in the Spectrum tab.
The menus provide the same operations without shortcuts.

## 6. Supported files and large-file behavior

CESS schema 3.0 and compatible legacy band groups can be browsed. The generic
tree also helps inspect BayeSED-shaped inputs and other HDF5 datasets.
For a raw input, select its actual wavelength vector and spectral matrix in
Explore plots, and check their units: viewing does not redshift, convolve, apply
instrument response or add noise to the template.

Not every HDF5 file has CESS spectra. Emission-line-only products may have
catalogue and line tables without parent spectral arrays. Missing or disabled
measurements cannot be reconstructed by the viewer; open the parent simulation
product to inspect its spectra. Layout and available fields can differ across
historical products. Legacy three-dimensional photometric parameter tables are
not covered by the standard table-plotting interface.

Single-source spectral browsing reads selected rows, and structure previews
are paged. Some operations still read entire columns, ID arrays or line tables;
catalogue plots, statistics and repeated selections therefore use memory and
can pause the interface on large files. The viewer is a desktop inspection
tool, not a distributed or out-of-core catalogue analysis engine. Begin with
a few source rows and avoid unnecessary full-column plots on a memory-limited
machine. The viewer does not rerun the simulation when changing rows.

## 7. Troubleshooting

| Symptom | Check or remedy |
| --- | --- |
| `cess-viewer` is not found | Activate the environment where CESS was installed; try `python -m cess.viewer --help` and check `python -m pip show cess` |
| Matplotlib is missing | Install the `viewer` extra into the active CESS environment |
| `_tkinter` is missing or a Tk backend cannot start | Verify the environment's Tk installation with `python -m tkinter`; a desktop session is required |
| No display / cannot connect to display on a server | View the file on a graphical workstation; the GUI is not required by the Gravity simulation scripts |
| Chinese glyphs appear as squares | Install or select a font with Chinese glyphs, or switch to the English interface |
| Spectrum tab is empty | Check the selected row's quality flags and whether the file actually contains spectra; EL-only products and raw inputs may need other tabs or the parent product |
| Plot values look inconsistent | Check dataset units, selected quantity, row mapping, masks, enabled models and axis choices; do not infer meaning from a filename alone |
| Interface pauses on a large catalogue | Avoid full-column plots/statistics and use the single-source controls or a separate analysis script |

Use the [Bash manual](COMMANDS.md) for simulation commands and
[output reference](OUTPUT_LAYOUT.md) for product interpretation.
