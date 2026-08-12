[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8169341.svg)](https://doi.org/10.5281/zenodo.21906688)

# SchottkySA -- Schottky Shape Analyzer

Empirical-template peak fitting and uncertainty analysis for Schottky spectra, with a Qt GUI.

> **Status:** pre-release. Not yet published to PyPI. See
> `docs/SCIENTIFIC_METHOD.md` for the exact scientific behaviour this
> package implements.

## What this is

SchottkySA fits one or more peaks in a Schottky (or similar) frequency-domain
spectrum using a **non-parametric, empirical peak template** built directly
from an isolated, high-signal-to-noise reference peak in your own data --
rather than an assumed analytic line shape (Gaussian, Lorentzian, ...). It
reports fitted centroid, area, and width per component, with uncertainties
from residual block-bootstrap resampling and template-shape propagation.

It is packaged primarily as a **desktop GUI application**; the scientific
core is also importable as a plain Python API (no GUI dependency required).

## Installation

### From source (current -- no PyPI release yet)

```bash
git clone <this-repository>
cd SchottkySA
python -m pip install -e ".[gui]"      # scientific core + Qt GUI
# or, for XLSX/ODS export support too:
python -m pip install -e ".[all]"
```

### From PyPI (once released)

```bash
pip install "SchottkySA[gui]"
```

Requires Python >= 3.10.

## Quickstart

Launch the GUI:

```bash
ssa
```

Or use the Python API directly (no Qt required):

```python
import numpy as np
from ssa import build_peak_template, fit_template_region

# f_ref, y_ref: an isolated, high-SNR reference peak (1D numpy arrays, Hz / linear power)
template = build_peak_template(f_ref, y_ref, name="reference_peak")

# f_target, y_target: the region you want to fit (may contain 1+ overlapping peaks)
fit = fit_template_region(f_target, y_target, template=template, n_peaks=1)
print(fit["info"]["mus"], fit["info"]["areas"], fit["red_chi2"])
```

See `examples/quickstart.py` for a runnable end-to-end example, including
uncertainty propagation.

## CLI usage

```
ssa            # launch the GUI (primary interface -- see docs/SCIENTIFIC_METHOD.md)
ssa --version
```

`ssa run` is a headless mode for scripted/batch analysis: it reads
named-column CSV/TSV/whitespace tables (`frequency_hz`, `power`, optionally
`sigma`) for a reference and a target region, runs the same
template-construction / fit / bootstrap / template-propagation pipeline as
the GUI, and writes a machine-readable JSON summary plus a per-bin curve CSV
(data, model, background, components). No display required.

```bash
ssa run \
  --reference reference_peak.csv --reference-range 1937300:1937360 \
  --target target_region.csv     --target-range     1937370:1937420 \
  --n-peaks 2 --mu-bounds "1937385:1937393;1937396:1937402" \
  --label my_fit --out-dir results/
ssa run --help   # full option list
```

The JSON summary's `known_limitations` field (see `ssa.pipeline.KNOWN_LIMITATIONS`)
lists what the pipeline does *not* yet do. `ssa run` does not replace the GUI for interactive region
selection/inspection; it is for reproducible, scriptable batch runs once you
know the regions and settings you want.

## Python API usage

The scientific core lives in `ssa.preprocessing`, `ssa.templates`,
`ssa.fitting`, `ssa.uncertainty`, and `ssa.io`, and has no Qt dependency --
`pip install SchottkySA` (no extras) is enough to use it. Key entry points
are re-exported from the top-level `ssa` package; see each module's
docstring, and `docs/SCIENTIFIC_METHOD.md` for the full algorithmic
description.

## Input / output

* **Input:** a NumPy `.npz` file containing two 1-D arrays -- a frequency
  axis and an amplitude/power axis. Recognised key names (in priority order,
  see `ssa.constants.PREFERRED_FREQUENCY_KEYS`/`PREFERRED_AMPLITUDE_KEYS`)
  and an "exactly two 1-D arrays present" fallback are used to select them.
* **Output:**
  * A results table and human-readable fit summary in the GUI.
  * A fit-history export (TSV/CSV/JSON, or XLSX/ODS with the `xlsx` extra)
    with one row per fitted component (see `ssa.io.make_export_rows_for_result`
    for the full column schema).
  * A current-fit NPZ overlay export (model, residual, components, template)
    for independent plotting/review.

## Scientific method overview

Peaks are modelled as scaled, shifted copies of a single empirical template
built from your own data (not an assumed analytic shape), fit by weighted
non-linear least squares. Uncertainties come from two independent resampling
procedures (residual block-bootstrap of the target fit, and refitting with a
bank of bootstrap-perturbed templates), combined in quadrature. See
`docs/SCIENTIFIC_METHOD.md` for the full description, conventions, and units.

## Reproducibility

Every stochastic step (template bootstrap, residual bootstrap, template-shape
propagation) uses an explicit, user-visible `numpy.random.Generator` seed
(default `12345`); default parameters live in `ssa.constants`. See
`docs/REPRODUCIBILITY.md` for the exact environment and steps used to
validate this package's numerics against a frozen regression baseline.

## Citation

If you use SchottkySA in published work, please cite it -- see `CITATION.cff`
(author/DOI fields are placeholders pending a public release).

## Known limitations / non-goals

* GUI input is limited to two-array NPZ files; `ssa run` reads named-column
  CSV/TSV/whitespace tables instead (see "CLI usage" above) -- no other
  spectrum formats are supported by either interface.
* The initial multi-peak centroid guess (`scipy.signal.find_peaks`-based) can
  fail on closely spaced ("contaminated") peaks unless you supply explicit
  centroid search ranges (`mu_bounds`). This is a known, documented
  characteristic, not a regression.
* Bootstrap templates do not yet inherit the nominal template's `edge_mode`
  (only relevant if you use a non-default edge mode).

## Documentation and issues

* `docs/SCIENTIFIC_METHOD.md` -- equations, conventions, units, algorithm.
* `docs/REPRODUCIBILITY.md` -- environment and validation steps.
* Issues: use this repository's issue tracker once published.
