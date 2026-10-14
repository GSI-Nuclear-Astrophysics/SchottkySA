---
title: SchottkySA
---

# SchottkySA

SchottkySA extracts component frequencies, integrated areas and widths from
measured spectra using empirical reference profiles. The Python, command-line
and Qt interfaces share the same numerical pipeline.

The GUI supports component-specific templates, interactive centroid intervals,
independent left and right edge controls, harmonic conversion, reciprocal profile
fits and weighted profile combinations. Target and finite-reference resampling
are exported separately. Saved TOML recipes reconstruct the analysis and check
its numerical results.

Install the graphical package with `pip install 'SchottkySA[gui]'` and launch
`ssa`. The numerical core can be installed without Qt.

* [Source and installation instructions](https://github.com/GSI-Nuclear-Astrophysics/SchottkySA)
* [Scientific method](SCIENTIFIC_METHOD.html): conventions and uncertainty outputs.
* [Profile workbench](PROFILE_WORKBENCH.html): comparisons, combinations and controls.
* [Reproducibility](REPRODUCIBILITY.html): saved recipes and headless replay.

The source distribution includes a portable constructed example. Cite the
software version used in your analysis, as described in `CITATION.cff`.
