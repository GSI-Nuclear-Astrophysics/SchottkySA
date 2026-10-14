[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21906687.svg)](https://doi.org/10.5281/zenodo.21906687)

# SchottkySA — Schottky Shape Analyzer

SchottkySA fits spectral components with empirical line shapes measured in
the same experiment. It returns centroids, integrated areas, widths and joint
resampling distributions through Python, a command line and an interactive
Qt application. Version 1.1 adds component-specific templates, profile comparison
and averaging, independent edge controls and expanded saved-session replay.

## Installation

```sh
python -m pip install SchottkySA         # scientific core
python -m pip install 'SchottkySA[gui]'   # desktop application
python -m pip install 'SchottkySA[all]'   # GUI and spreadsheet exports
```

Python 3.10 or later is required. From a checkout, use
`python -m pip install -e '.[dev]'` for development.

```sh
ssa
ssa --version
```

## A fit in Python

```python
from ssa import build_peak_template, fit_template_region

template = build_peak_template(f_reference, power_reference, name="reference")
fit = fit_template_region(f_target, power_target, template=template, n_peaks=2, init_mus=centres, mu_bounds=bounds)
print(fit["info"]["mus"], fit["info"]["areas"], fit["red_chi2"])
```

Frequencies and centroid bounds are in Hz; power is linear. The template
is normalized to unit area and centered at zero center of gravity. A component
is a translated, optionally dilated copy of that shape. Its area has units
of the input ordinate times Hz.

`component_templates=[template_1, template_2]` assigns a different profile to
each component. Ordered centroid bounds retain identity even when the supports
overlap. A common dilation means a common fractional stretch; different
templates can still give different component widths.

`python examples/quickstart.py` demonstrates the numerical interfaces using
constructed test spectra. These spectra are software fixtures, not experiments.

## Interactive analysis

Load an NPZ containing frequency and linear-power arrays. Select the reference
and target regions, build a template and inspect its edges. The profile workbench
provides two linked spectrum views so distant regions can be compared without
losing the current analysis window.

Double-click to add a centroid, move its marker and edit or drag its allowed
range. The multiple-template selector assigns a profile to each component.
Both template edges have separate measured-point counts and zero-anchor distances.
The preview shows the selected samples and closure before fitting.

Harmonic fields accept integers, decimals and positive fractions such as
`214/214` and `215/214`. Profile transfer scales the frequency coordinate by
the harmonic ratio and preserves unit area. Bidirectional cross-fits compare
each selected reference with the other's measured region.

Several centered shapes can form a weighted empirical mean without extra
smoothing. A source-sensitivity bank retains differences among the source
profiles. Independent-reference mean draws are available when their independence
is justified. Different harmonics of one acquisition are not automatically
independent reference measurements.

## Uncertainties and fit diagnostics

Target residual resampling refits a fixed template after circular-block
resampling. Reference propagation refits the unchanged target with a bank of
perturbed reference shapes. Seeds and replicate counts are explicit. Each
parameter retains the 16th, 50th and 84th percentiles.

Reported combined errors add the two symmetric percentile halfwidths in
quadrature under an independence assumption. Separations and area ratios are
formed from joint component draws, preserving within-fit covariance. Repeated
uses of one reference bank share the selected draw. Distinct banks use an
independent-draw model unless the caller supplies a different treatment.

Reduced chi-square, residuals, optimization status, search-bound proximity
and returned resampling counts accompany each fit. Formal Jacobian covariance
is a separate diagnostic; it is not added again to the resampling result.
Fixed edge choices and source-to-target shape mismatch require separate
sensitivity checks. See [Scientific method](docs/SCIENTIFIC_METHOD.md).

## Headless run and replay

`ssa run` reads named-column CSV, TSV or whitespace tables and writes a JSON
summary plus per-bin curves. `ssa replay` reconstructs a saved TOML session.
Use `ssa run --help` and `ssa replay --help` for their options.

The GUI's **Save config + tables** and **Replay config…** controls create or
restore `DATE_SchottkySA_CONFIG_suffix` bundles containing `config.toml`, fit
history and a current-fit overlay. The input spectrum is referenced by path
and SHA-256. Completed fits preserve template recipes, component assignments,
centroid bounds, harmonics, resampling settings and numerical checkpoints.

Replay reports changed-input warnings and numerical mismatches. Absolute
frequency positions are compared at `1e-5` Hz tolerance; other scalar fields
retain their documented relative/absolute tolerances. See
[Reproducibility](docs/REPRODUCIBILITY.md) and the
[Profile workbench guide](docs/PROFILE_WORKBENCH.md).

## Development and citation

```sh
pytest
ruff check .
ruff format --check .
pyright
python -m build
twine check dist/*
python scripts/check_release.py dist/*
```

Use [CITATION.cff](CITATION.cff) to cite the software. The badge identifies
the software family; a published version has its own archive DOI. See
[release instructions](docs/RELEASE.md), [CONTRIBUTING.md](CONTRIBUTING.md)
and [SECURITY.md](SECURITY.md). SchottkySA is licensed under GPL-3.0-only.
