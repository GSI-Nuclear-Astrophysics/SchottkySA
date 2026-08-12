# Contributing to SchottkySA

Thanks for considering a contribution. Follow the following norms:

## Physics

**Do not change scientific behaviour** (equations, defaults, data-selection
logic, fitting procedure, uncertainty treatment, units, rounding) in a pull
request that isn't explicitly about a scientific change.
If you believe a formula, default, or convention is wrong: open an issue
describing the evidence and proposed validation first. Do not submit a
silent fix.

## Development setup

```bash
git clone https://github.com/GSI-Nuclear-Astrophysics/SchottkySA.git
cd SchottkySA
python -m pip install -e ".[dev]"
pre-commit install
```

## Before opening a PR

```bash
pytest
ruff check .
ruff format --check .
pyright
python -m build && twine check dist/*
```

All four must pass. 

## Code style

* `ruff` (lint + format) and `pyright` (type checking) are enforced in CI;
  install `pre-commit` locally to catch issues before pushing.
* Prefer plain functions and small dataclasses over classes.
* Keep the scientific core (`ssa.preprocessing`/`templates`/`fitting`/
  `uncertainty`/`io`) free of Qt imports and hidden I/O; GUI-only code lives
  under `ssa.gui`.

## Reporting issues

Please include: SchottkySA version (`ssa --version`), Python version, OS,
and, for numerical issues, a minimal reproducing script or input file (no
confidential/raw experimental data -- use or extend the synthetic fixtures
in `tests/data/` if possible).
