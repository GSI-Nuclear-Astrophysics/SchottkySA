"""Generic numeric preprocessing and residual-statistics helpers.

See ``docs/SCIENTIFIC_METHOD.md`` sections 2 and 3 for the scientific
rationale behind these algorithms and default values.
"""

from __future__ import annotations

import numpy as np

from ssa.constants import (
    DEFAULT_MAX_ACF_LAG,
    DEFAULT_MAX_BLOCK_FRACTION,
    DEFAULT_MIN_BLOCK,
    DEFAULT_ROBUST_SIGMA_MIN_FRACTION,
)
from ssa.exceptions import InputFormatError

__all__ = [
    "trapz",
    "prepare_xy",
    "make_odd",
    "robust_sigma_from_second_difference",
    "circular_block_resample",
    "autocorrelation_1d",
    "estimate_block_size_from_residuals",
    "summarize_distribution",
]


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral of ``y`` over ``x`` (unit-agnostic).

    Thin wrapper around ``numpy.trapezoid`` (requires numpy>=2.0; the
    pre-2.0 ``numpy.trapz`` name was removed from numpy's own stubs/API, so
    this module no longer falls back to it). Numerically identical either
    way -- both are the composite trapezoidal rule.
    """
    return float(np.trapezoid(y, x))


def prepare_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort ``x``/``y``, remove invalid points, and average duplicate x values.

    Parameters
    ----------
    x, y:
        1-D arrays of equal shape, in any order, on any (finite) grid.

    Returns
    -------
    Sorted, de-duplicated (by x-averaging) ``(x, y)`` arrays.

    Raises
    ------
    InputFormatError
        If ``x``/``y`` shapes disagree, or fewer than 3 finite points remain.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if x.shape != y.shape:
        raise InputFormatError(f"x and y must have the same shape. Got {x.shape} and {y.shape}.")

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    if len(x) < 3:
        raise InputFormatError("Need at least three finite data points.")

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    xu, inv = np.unique(x, return_inverse=True)
    if len(xu) != len(x):
        yu = np.zeros_like(xu, dtype=float)
        counts = np.zeros_like(xu, dtype=float)
        np.add.at(yu, inv, y)
        np.add.at(counts, inv, 1.0)
        y = yu / np.maximum(counts, 1.0)
        x = xu

    return x, y


def make_odd(n: int, minimum: int = 5) -> int:
    """Round ``n`` up to the nearest odd integer, floored at ``minimum``."""
    n = int(n)
    if n % 2 == 0:
        n += 1
    return max(n, minimum)


def robust_sigma_from_second_difference(
    y: np.ndarray, min_fraction: float = DEFAULT_ROBUST_SIGMA_MIN_FRACTION
) -> float:
    """
    Estimate an effective white-noise scale from second differences.

    For independent white noise with std sigma:
        std(diff(y, n=2)) ~= sqrt(6) * sigma.

    Schottky spectra often have correlated residuals, so this is only an
    effective scale for fitting/diagnostics, not a final uncertainty model.
    """
    y = np.asarray(y, dtype=float)
    if len(y) < 5:
        sigma = float(np.nanstd(y))
    else:
        d2 = np.diff(y, n=2)
        med = np.nanmedian(d2)
        mad = np.nanmedian(np.abs(d2 - med))
        sigma = float(1.4826 * mad / np.sqrt(6.0))

    ymax = float(np.nanmax(np.abs(y))) if len(y) else 0.0
    floor = min_fraction * ymax if ymax > 0 else 1e-30
    return max(sigma, floor, 1e-30)


def circular_block_resample(residuals: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block residual resampling."""
    residuals = np.asarray(residuals, dtype=float)
    n = len(residuals)
    block_size = int(max(1, block_size))
    starts = rng.integers(0, n, size=int(np.ceil(n / block_size)))
    out = []
    for s in starts:
        idx = (s + np.arange(block_size)) % n
        out.append(residuals[idx])
    return np.concatenate(out)[:n]


def autocorrelation_1d(r: np.ndarray, max_lag: int = DEFAULT_MAX_ACF_LAG) -> tuple[np.ndarray, np.ndarray]:
    """Simple normalized autocorrelation estimate."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 5:
        return np.arange(1), np.array([np.nan])

    r = r - np.nanmean(r)
    denom = np.nansum(r * r)
    if denom <= 0:
        return np.arange(1), np.array([np.nan])

    max_lag = int(min(max_lag, len(r) - 2))
    lags = np.arange(max_lag + 1)
    acf = np.empty_like(lags, dtype=float)
    for lag in lags:
        if lag == 0:
            acf[lag] = 1.0
        else:
            acf[lag] = np.nansum(r[:-lag] * r[lag:]) / denom
    return lags, acf


def estimate_block_size_from_residuals(
    residuals: np.ndarray,
    max_lag: int = DEFAULT_MAX_ACF_LAG,
    min_block: int = DEFAULT_MIN_BLOCK,
    max_fraction: float = DEFAULT_MAX_BLOCK_FRACTION,
) -> tuple[int, float, np.ndarray, np.ndarray]:
    """Estimate a residual bootstrap block size using the positive ACF sequence."""
    lags, acf = autocorrelation_1d(residuals, max_lag=max_lag)
    positive = []
    for a in acf[1:]:
        if not np.isfinite(a) or a <= 0:
            break
        positive.append(float(a))

    tau_int = 1.0 + 2.0 * np.sum(positive) if positive else 1.0
    n = len(residuals)
    max_block = max(min_block, int(max_fraction * n))
    block_size = int(np.ceil(2.0 * tau_int))
    block_size = int(np.clip(block_size, min_block, max_block))
    return block_size, float(tau_int), lags, acf


def summarize_distribution(vals: np.ndarray) -> dict[str, float]:
    """Median and 16/84-percentile-based asymmetric interval of a sample."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"median": np.nan, "minus": np.nan, "plus": np.nan, "q16": np.nan, "q84": np.nan, "n": 0}
    q16, q50, q84 = np.percentile(vals, [16, 50, 84])
    return {
        "median": float(q50),
        "minus": float(q50 - q16),
        "plus": float(q84 - q50),
        "q16": float(q16),
        "q84": float(q84),
        "n": int(len(vals)),
    }
