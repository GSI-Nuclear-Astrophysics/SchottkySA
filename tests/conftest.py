"""Shared pytest fixtures/helpers for the ssa test suite."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).parent / "data"
BASELINE_PATH = Path(__file__).parent / "baseline" / "baseline_results.json"

# Every regression scenario is fully deterministic (fixed
# numpy.random.Generator seeds, no ordering dependence), so results should
# match bit-for-bit on identical BLAS/LAPACK. These tolerances exist only to
# absorb harmless cross-platform floating-point noise, not to mask a real
# behavioural change -- a genuine regression is expected to produce
# deviations many orders of magnitude larger than 1e-8 relative.
RTOL = 1e-8
ATOL = 1e-10


@pytest.fixture(scope="session")
def baseline() -> dict:
    """The frozen numeric baseline snapshot -- see tests/baseline/."""
    return json.loads(BASELINE_PATH.read_text())


def load_xy(name: str, fkey: str, ykey: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(DATA_DIR / name)
    return np.asarray(d[fkey], dtype=float), np.asarray(d[ykey], dtype=float)


def write_table(
    path: Path,
    f: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray | None = None,
    delimiter: str = ",",
    freq_col: str = "frequency_hz",
    power_col: str = "power",
    sigma_col: str = "sigma",
) -> Path:
    """Write a named-column CSV/TSV/whitespace table for ``ssa.io.load_table``
    tests and ``ssa run`` CLI tests, from an existing NPZ fixture's arrays."""
    # repr(np.float64(...)) includes the "np.float64(...)" wrapper as of
    # NumPy 2.0, which would not parse back as a number -- convert to plain
    # Python float first.
    cols = [freq_col, power_col] + ([sigma_col] if sigma is not None else [])
    lines = [delimiter.join(cols)]
    for i in range(len(f)):
        row = [repr(float(f[i])), repr(float(y[i]))] + ([repr(float(sigma[i]))] if sigma is not None else [])
        lines.append(delimiter.join(row))
    path.write_text("\n".join(lines) + "\n")
    return path


def assert_close(actual, expected, path: str = "") -> None:
    """Recursively compare nested dict/list/number structures within RTOL/ATOL."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected dict, got {type(actual)}"
        for k, v in expected.items():
            assert k in actual, f"{path}.{k}: missing key"
            assert_close(actual[k], v, f"{path}.{k}")
    elif isinstance(expected, list):
        assert isinstance(actual, (list, tuple, np.ndarray)), f"{path}: expected list-like"
        assert len(actual) == len(expected), f"{path}: length mismatch"
        for i, (a, e) in enumerate(zip(actual, expected, strict=False)):
            assert_close(a, e, f"{path}[{i}]")
    elif isinstance(expected, bool):
        assert bool(actual) == expected, f"{path}: {actual!r} != {expected!r}"
    elif isinstance(expected, (int, float)):
        if isinstance(expected, float) and np.isnan(expected):
            assert np.isnan(float(actual)), f"{path}: expected NaN, got {actual!r}"
        else:
            np.testing.assert_allclose(float(actual), float(expected), rtol=RTOL, atol=ATOL, err_msg=path)
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"
