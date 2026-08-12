---
title: SchottkySA
---

# SchottkySA — Schottky Shape Analyzer

Empirical-template peak fitting and uncertainty analysis for Schottky
spectra, with a Qt GUI.

SchottkySA fits one or more peaks in a Schottky (or similar) frequency-domain
spectrum using a **non-parametric, empirical peak template** built directly
from an isolated, high-signal-to-noise reference peak in your own data —
rather than an assumed analytic line shape (Gaussian, Lorentzian, ...). It
reports fitted centroid, area, and width per component, with uncertainties
from residual block-bootstrap resampling and template-shape propagation.

* [Source code and installation instructions]({{ site.github.repository_url | default: "https://github.com/GSI-Nuclear-Astrophysics/SchottkySA" }})
* [Scientific method](SCIENTIFIC_METHOD.html) — equations, conventions, units, algorithm.
* [Reproducibility](REPRODUCIBILITY.html) — environment and validation steps.

## Citation

If you use SchottkySA in published work, please cite it — see `CITATION.cff`
in the repository root.
