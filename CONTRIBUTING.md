# Contributing

Install with `python -m pip install -e '.[dev]'`. Run `pytest`, `ruff check .`,
`ruff format --check .`, `pyright`, `python -m build` and `twine check dist/*`.

Describe the rationale and validation for changes to equations, defaults,
selection, interpolation or uncertainty conventions. Keep the numerical core
independent of Qt. Use constructed or published inputs for public tests.
Document units and shared-input assumptions on public interfaces.

Comments should explain a numerical or physical choice that is not evident
from the implementation. Record API and scientific-convention changes in
`CHANGELOG.md`. Keep local measurements, derived exports and credentials out
of commits and distributions.

Numerical bug reports should include the package and library versions, settings
and a minimal distributable reproducer.
