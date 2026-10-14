---
title: Reproducibility
---

# Reproducing a spectral fit

Template banks, target residual resampling and template propagation use explicit
NumPy random-generator seeds. A fixed seed, input array, library version and
call sequence define the sampling experiment. Repeated components using one
reference bank retain shared draw indices. Profile-averaging plans are generated
before parallel dispatch, preserving the sequence when worker counts change.

Saved TOML sessions record input array keys and digest, template construction,
component assignments, fit bounds, harmonic expressions, resampling settings
and completed-fit checkpoints. Replay reconstructs profiles and fits using the
same numerical core as the GUI and CLI.

## Comparison tolerances

Centroids, quantile locations and raw reference centers use absolute tolerance
`1e-5` Hz. They are not compared with a relative tolerance on the RF carrier.
Other scalar checkpoints use relative `1e-3` and absolute `1e-6`. The public
regression baseline has its own assertion tolerances for each numerical quantity.
These tolerances describe numerical reproduction, not experimental accuracy.

An input digest mismatch is reported independently of numerical differences.
If the file moved, supply its new path. A changed input is not silently labelled
as the original measurement. Source-to-target profile mismatch and alternative
edge closures remain separate analysis sensitivities.

## Public fixtures and checks

`tests/data` contains constructed reference, isolated and overlapping spectra.
The generation recipe and fixed numerical baseline are under `tests/baseline`.
These files test array handling, normalization, component recovery, covariance
summaries and replay. They are not measured Schottky data.

```sh
QT_QPA_PLATFORM=offscreen PYQTGRAPH_QT_LIB=PySide6 pytest
python examples/quickstart.py
python -m build
twine check dist/*
python scripts/check_release.py dist/*
```

Verify an installed wheel outside the checkout. GUI tests require the optional
GUI dependencies. The offscreen Qt setting changes display handling only.
Record numerical-library versions with an archived analysis; a fixed seed alone
does not identify the full environment.
