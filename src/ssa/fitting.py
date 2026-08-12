"""Multi-component empirical-template least-squares fitting.

See ``docs/SCIENTIFIC_METHOD.md`` section 4 for the parameter-vector layout,
initial-guess heuristic, bounds/``mu_bounds``/``min_separation`` semantics,
and diagnostic formulas, and ``docs/OPEN_SCIENTIFIC_QUESTIONS.md`` item 2
for a known failure mode of the unconstrained initial-guess heuristic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from ssa.constants import (
    DEFAULT_ALLOW_SCALE,
    DEFAULT_BACKGROUND_ORDER,
    DEFAULT_COMMON_SCALE,
    DEFAULT_LOSS,
    DEFAULT_MAX_NFEV,
    DEFAULT_MIN_SEPARATION_HZ,
    DEFAULT_SEPARATION_PENALTY_STRENGTH,
)
from ssa.exceptions import FitConfigurationError, FitConstraintViolationError
from ssa.preprocessing import prepare_xy, robust_sigma_from_second_difference, trapz
from ssa.templates import PeakTemplate

__all__ = [
    "template_model",
    "guess_initial_mus",
    "normalize_mu_bounds",
    "ensure_mu_bounds_respect_min_separation",
    "mu_bounds_to_string",
    "clip_p0_to_bounds",
    "build_initial_params_and_bounds",
    "sort_parameter_vector_by_mu",
    "enforce_initial_min_separation",
    "min_separation_penalty_residuals",
    "fit_template_region",
]


def template_model(
    x: np.ndarray,
    p: np.ndarray,
    template: PeakTemplate,
    n_peaks: int,
    background_order: int = 0,
    allow_scale: bool = False,
    common_scale: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Evaluate multi-component empirical-template model."""
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    idx = 0

    mus = p[idx : idx + n_peaks]
    idx += n_peaks

    log_areas = p[idx : idx + n_peaks]
    idx += n_peaks
    areas = np.exp(log_areas)

    if allow_scale:
        if common_scale:
            scale = np.exp(p[idx])
            idx += 1
            scales = np.full(n_peaks, scale)
        else:
            scales = np.exp(p[idx : idx + n_peaks])
            idx += n_peaks
    else:
        scales = np.ones(n_peaks)

    components = []
    for mu, area, scale in zip(mus, areas, scales, strict=False):
        components.append(template.evaluate_component(x, mu=mu, area=area, scale=scale))
    components = np.asarray(components)

    if background_order is None or background_order < 0:
        bg = np.zeros_like(x)
        bg_coeff = np.array([])
    else:
        n_bg = int(background_order) + 1
        bg_coeff = p[idx : idx + n_bg]
        idx += n_bg
        span = np.ptp(x)
        xr = (x - np.mean(x)) / span if span > 0 else x * 0.0
        bg = np.zeros_like(x)
        for j, c in enumerate(bg_coeff):
            bg += c * xr**j

    y_model = bg + np.sum(components, axis=0)
    info = {"mus": mus, "areas": areas, "scales": scales, "bg_coeff": bg_coeff}
    return y_model, components, bg, info


def guess_initial_mus(x: np.ndarray, y: np.ndarray, n_peaks: int, init_mus: list[float] | None = None) -> np.ndarray:
    x, y = prepare_xy(x, y)
    if init_mus is not None and len(init_mus) > 0:
        mus = np.asarray(init_mus, dtype=float)
        if len(mus) != n_peaks:
            raise FitConfigurationError("Number of initial mu guesses must match n_peaks.")
        return np.sort(mus)

    y_pos = np.clip(y, 0.0, None)
    ymax = float(np.nanmax(y_pos)) if len(y_pos) else 0.0
    if ymax <= 0:
        return np.linspace(x.min(), x.max(), n_peaks + 2)[1:-1]

    distance = max(1, len(x) // max(2, 2 * n_peaks))
    peaks, _ = find_peaks(y_pos, distance=distance, prominence=0.03 * ymax)
    if len(peaks) >= n_peaks:
        selected = peaks[np.argsort(y_pos[peaks])[-n_peaks:]]
        return np.sort(x[selected])

    center = x[np.argmax(y_pos)]
    span = np.ptp(x)
    offsets = np.linspace(-0.12, 0.12, n_peaks) * span
    return np.sort(center + offsets)


def normalize_mu_bounds(
    mu_bounds: list[tuple[float, float]] | None,
    n_peaks: int,
    xmin: float,
    xmax: float,
) -> np.ndarray | None:
    """Validate and normalize per-component centroid search ranges.

    Parameters
    ----------
    mu_bounds:
        None or a list [(lo0, hi0), (lo1, hi1), ...] in Hz.
        The order defines the component identity. Ranges should therefore be
        non-overlapping and ordered for multi-component fits.
    n_peaks:
        Number of template components.
    xmin, xmax:
        Data window limits. Bounds are allowed to extend slightly outside the
        selected fit window, but must be finite and lo < hi.

    Returns
    -------
    None or ndarray with shape (n_peaks, 2).
    """
    if mu_bounds is None:
        return None
    if len(mu_bounds) == 0:
        return None
    if len(mu_bounds) != n_peaks:
        raise FitConfigurationError(f"mu_bounds must contain exactly {n_peaks} ranges; got {len(mu_bounds)}.")

    arr = np.asarray(mu_bounds, dtype=float)
    if arr.shape != (n_peaks, 2):
        raise FitConfigurationError(f"mu_bounds must have shape ({n_peaks}, 2); got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise FitConfigurationError("All mu search-range limits must be finite.")

    # Sort endpoints inside each pair, but keep component order unchanged.
    lo = np.minimum(arr[:, 0], arr[:, 1])
    hi = np.maximum(arr[:, 0], arr[:, 1])
    if np.any(hi <= lo):
        raise FitConfigurationError("Each mu search range must satisfy hi > lo.")

    out = np.column_stack([lo, hi])

    # Component ranges must be ordered. This makes component labels stable and
    # lets native least_squares bounds enforce identity without soft penalties.
    if n_peaks > 1 and np.any(out[1:, 0] <= out[:-1, 1]):
        raise FitConfigurationError(
            "Centroid search ranges must be ordered and non-overlapping. Use e.g. lo0:hi0; lo1:hi1 with hi0 < lo1."
        )

    return out


def ensure_mu_bounds_respect_min_separation(mu_bounds_arr: np.ndarray | None, min_separation: float) -> None:
    """Require supplied ranges to guarantee the requested minimum separation.

    If the ranges satisfy lo[k+1] - hi[k] >= min_separation, then every possible
    optimizer solution inside the native bounds also satisfies the separation.
    This is stronger and safer than a penalty term.
    """
    if mu_bounds_arr is None or len(mu_bounds_arr) <= 1:
        return
    dmin = float(max(0.0, min_separation))
    if dmin <= 0:
        return
    gaps = mu_bounds_arr[1:, 0] - mu_bounds_arr[:-1, 1]
    if np.any(gaps < dmin):
        bad = np.where(gaps < dmin)[0]
        details = ", ".join([f"gap[{i}-{i + 1}]={gaps[i]:.9g} Hz" for i in bad])
        raise FitConfigurationError(
            "The supplied centroid ranges do not guarantee the requested minimum separation. "
            f"Need every lo[k+1] - hi[k] >= {dmin:.9g} Hz; {details}."
        )


def mu_bounds_to_string(mu_bounds: Any | None) -> str:
    if mu_bounds is None:
        return ""
    arr = np.asarray(mu_bounds, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return ""
    return "; ".join(f"{lo:.12g}:{hi:.12g}" for lo, hi in arr)


def clip_p0_to_bounds(p0: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Put the initial parameter vector inside scipy least_squares bounds."""
    p0 = np.asarray(p0, dtype=float).copy()
    lb, ub = bounds
    for i in range(len(p0)):
        if np.isfinite(lb[i]) and p0[i] < lb[i]:
            p0[i] = lb[i]
        if np.isfinite(ub[i]) and p0[i] > ub[i]:
            p0[i] = ub[i]
    return p0


def build_initial_params_and_bounds(
    x: np.ndarray,
    y: np.ndarray,
    template: PeakTemplate,
    n_peaks: int,
    init_mus: list[float] | None = None,
    background_order: int = 0,
    allow_scale: bool = False,
    common_scale: bool = True,
    mu_bounds: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], list[str]]:
    x, y = prepare_xy(x, y)
    y_pos = np.clip(y, 0.0, None)
    mus0 = guess_initial_mus(x, y, n_peaks, init_mus=init_mus)

    mu_bounds_arr = normalize_mu_bounds(mu_bounds, n_peaks=n_peaks, xmin=float(x.min()), xmax=float(x.max()))
    if mu_bounds_arr is not None:
        # The search windows define component identity. If no explicit initial
        # centroids were supplied, start each component in the center of its window.
        if init_mus is None:
            mus0 = 0.5 * (mu_bounds_arr[:, 0] + mu_bounds_arr[:, 1])
        else:
            mus0 = np.clip(mus0, mu_bounds_arr[:, 0], mu_bounds_arr[:, 1])

    total_area = max(trapz(y_pos, x), 1e-30)
    areas0 = np.full(n_peaks, total_area / n_peaks)

    if n_peaks > 1:
        mids = 0.5 * (mus0[:-1] + mus0[1:])
        edges = np.concatenate(([x.min() - 1], mids, [x.max() + 1]))
        for k in range(n_peaks):
            m = (x >= edges[k]) & (x < edges[k + 1])
            if np.count_nonzero(m) > 2:
                a = trapz(y_pos[m], x[m])
                if np.isfinite(a) and a > 0:
                    areas0[k] = a

    p0: list[float] = []
    names: list[str] = []

    for k in range(n_peaks):
        p0.append(float(mus0[k]))
        names.append(f"mu_{k}")

    for k in range(n_peaks):
        p0.append(float(np.log(max(areas0[k], 1e-300))))
        names.append(f"log_area_{k}")

    if allow_scale:
        if common_scale:
            p0.append(0.0)
            names.append("log_scale_common")
        else:
            for k in range(n_peaks):
                p0.append(0.0)
                names.append(f"log_scale_{k}")

    if background_order is not None and background_order >= 0:
        for j in range(background_order + 1):
            p0.append(0.0)
            names.append(f"bg_{j}")

    p0_arr = np.asarray(p0, dtype=float)

    lb: list[float] = []
    ub: list[float] = []
    xspan = np.ptp(x)
    xmin = x.min() - 0.05 * xspan
    xmax = x.max() + 0.05 * xspan
    if mu_bounds_arr is None:
        for _ in range(n_peaks):
            lb.append(float(xmin))
            ub.append(float(xmax))
    else:
        for k in range(n_peaks):
            lb.append(float(mu_bounds_arr[k, 0]))
            ub.append(float(mu_bounds_arr[k, 1]))

    log_area_min = float(np.log(max(total_area * 1e-10, 1e-300)))
    log_area_max = float(np.log(max(total_area * 1e5, 1e-300)))
    for _ in range(n_peaks):
        lb.append(log_area_min)
        ub.append(log_area_max)

    if allow_scale:
        n_scale = 1 if common_scale else n_peaks
        for _ in range(n_scale):
            lb.append(float(np.log(0.25)))
            ub.append(float(np.log(4.0)))

    if background_order is not None and background_order >= 0:
        for _ in range(background_order + 1):
            lb.append(-np.inf)
            ub.append(np.inf)

    return p0_arr, (np.asarray(lb), np.asarray(ub)), names


def sort_parameter_vector_by_mu(p: np.ndarray, names: list[str], n_peaks: int) -> np.ndarray:
    """Sort component labels by increasing centroid to reduce label-switching in summaries."""
    p = np.asarray(p, dtype=float).copy()
    if n_peaks <= 1:
        return p

    mus = p[:n_peaks]
    order = np.argsort(mus)
    if np.all(order == np.arange(n_peaks)):
        return p

    new = p.copy()
    new[:n_peaks] = p[order]
    area_start = n_peaks
    new[area_start : area_start + n_peaks] = p[area_start + order]

    for new_k, old_k in enumerate(order):
        old_name = f"log_scale_{old_k}"
        new_name = f"log_scale_{new_k}"
        if old_name in names and new_name in names:
            new[names.index(new_name)] = p[names.index(old_name)]
    return new


def enforce_initial_min_separation(p0: np.ndarray, n_peaks: int, min_separation: float) -> np.ndarray:
    """Return a copy of p0 whose centroid guesses are sorted and separated.

    The optimizer still receives the usual physical parameters. This helper only
    prevents the initial point from violating the requested minimum separation.
    The least-squares residual below adds a strong penalty if the fit tries to
    move components closer than min_separation.
    """
    p0 = np.asarray(p0, dtype=float).copy()
    min_separation = float(max(0.0, min_separation))
    if n_peaks <= 1 or min_separation <= 0.0:
        return p0

    mus = np.sort(p0[:n_peaks])
    for k in range(1, n_peaks):
        if mus[k] - mus[k - 1] < min_separation:
            mus[k] = mus[k - 1] + min_separation
    p0[:n_peaks] = mus
    return p0


def min_separation_penalty_residuals(
    p: np.ndarray,
    n_peaks: int,
    min_separation: float,
    x_span: float,
    penalty_strength: float = DEFAULT_SEPARATION_PENALTY_STRENGTH,
) -> np.ndarray:
    """Dimensionless penalty residuals for enforcing ordered separated peaks.

    For least_squares there is no native pairwise inequality constraint of the
    form mu[k+1]-mu[k] >= d. This penalty makes violations extremely expensive.

    The residual returned is approximately
        sqrt(penalty_strength) * max(0, d - delta_mu) / scale
    with scale=max(d, 1e-6*x_span).
    """
    min_separation = float(max(0.0, min_separation))
    if n_peaks <= 1 or min_separation <= 0.0:
        return np.empty(0, dtype=float)
    mus = np.asarray(p[:n_peaks], dtype=float)
    diffs = np.diff(mus)
    violations = np.maximum(0.0, min_separation - diffs)
    scale = max(min_separation, 1.0e-6 * float(max(abs(x_span), 1.0)), 1.0e-12)
    return np.sqrt(float(penalty_strength)) * violations / scale


def fit_template_region(
    x: np.ndarray,
    y: np.ndarray,
    template: PeakTemplate,
    n_peaks: int,
    init_mus: list[float] | None = None,
    sigma: np.ndarray | float | None = None,
    background_order: int = DEFAULT_BACKGROUND_ORDER,
    allow_scale: bool = DEFAULT_ALLOW_SCALE,
    common_scale: bool = DEFAULT_COMMON_SCALE,
    p0_override: np.ndarray | None = None,
    loss: str = DEFAULT_LOSS,
    max_nfev: int = DEFAULT_MAX_NFEV,
    min_separation: float = DEFAULT_MIN_SEPARATION_HZ,
    separation_penalty_strength: float = DEFAULT_SEPARATION_PENALTY_STRENGTH,
    mu_bounds: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Least-squares empirical-template fit."""
    x, y = prepare_xy(x, y)
    if sigma is None:
        sigma_val = robust_sigma_from_second_difference(y)
        sigma_arr = np.full_like(y, sigma_val, dtype=float)
    else:
        sigma_arr = np.asarray(sigma, dtype=float)
        if sigma_arr.ndim == 0:
            sigma_arr = np.full_like(y, float(sigma_arr), dtype=float)
        if sigma_arr.shape != y.shape:
            raise FitConfigurationError("sigma must be scalar or same shape as y.")
        sigma_arr = np.clip(sigma_arr, 1e-30, np.inf)

    min_separation = float(max(0.0, min_separation))
    mu_bounds_arr = normalize_mu_bounds(mu_bounds, n_peaks=n_peaks, xmin=float(x.min()), xmax=float(x.max()))
    ensure_mu_bounds_respect_min_separation(mu_bounds_arr, min_separation=min_separation)

    p0, bounds, names = build_initial_params_and_bounds(
        x,
        y,
        template,
        n_peaks=n_peaks,
        init_mus=init_mus,
        background_order=background_order,
        allow_scale=allow_scale,
        common_scale=common_scale,
        mu_bounds=mu_bounds_arr.tolist() if mu_bounds_arr is not None else None,
    )
    if p0_override is not None:
        p0 = np.asarray(p0_override, dtype=float)

    p0 = clip_p0_to_bounds(p0, bounds)
    if mu_bounds_arr is None:
        p0 = enforce_initial_min_separation(p0, n_peaks=n_peaks, min_separation=min_separation)

    def residuals(p: np.ndarray) -> np.ndarray:
        y_model, _, _, _ = template_model(
            x,
            p,
            template,
            n_peaks=n_peaks,
            background_order=background_order,
            allow_scale=allow_scale,
            common_scale=common_scale,
        )
        data_resid = (y_model - y) / sigma_arr
        # If component-specific search ranges were supplied, native optimizer
        # bounds enforce the component identity and, if requested, the minimum
        # separation. If not, fall back to a soft penalty residual instead.
        if mu_bounds_arr is None:
            sep_penalty = min_separation_penalty_residuals(
                p,
                n_peaks=n_peaks,
                min_separation=min_separation,
                x_span=float(np.ptp(x)),
                penalty_strength=separation_penalty_strength,
            )
            if len(sep_penalty):
                return np.concatenate([data_resid, sep_penalty])
        return data_resid

    res = least_squares(residuals, p0, bounds=bounds, loss=loss, max_nfev=max_nfev)
    p_fit = sort_parameter_vector_by_mu(res.x, names, n_peaks)

    y_model, components, bg, info = template_model(
        x,
        p_fit,
        template,
        n_peaks=n_peaks,
        background_order=background_order,
        allow_scale=allow_scale,
        common_scale=common_scale,
    )

    r = (y_model - y) / sigma_arr
    chi2 = float(np.sum(r * r))
    dof = int(len(y) - len(p_fit))
    red_chi2 = chi2 / dof if dof > 0 else np.nan

    cov = None
    try:
        J = res.jac
        _U, s, VT = np.linalg.svd(J, full_matrices=False)
        threshold = np.finfo(float).eps * max(J.shape) * s[0]
        keep = s > threshold
        cov = (VT[keep].T / (s[keep] ** 2)) @ VT[keep]
        if np.isfinite(red_chi2):
            cov *= red_chi2
    except Exception:
        cov = None

    mu_diffs = np.diff(info["mus"]) if n_peaks > 1 else np.array([], dtype=float)
    min_observed_separation = float(np.nanmin(mu_diffs)) if len(mu_diffs) else float("nan")
    min_separation_satisfied = bool(
        n_peaks <= 1 or min_separation <= 0.0 or min_observed_separation >= min_separation * (1.0 - 1e-6)
    )
    if n_peaks > 1 and min_separation > 0.0 and not min_separation_satisfied:
        raise FitConstraintViolationError(
            f"Fit violated the requested minimum separation: observed {min_observed_separation:.9g} Hz "
            f"< requested {min_separation:.9g} Hz. Supply non-overlapping mu search ranges "
            "or reduce the requested separation."
        )

    return {
        "x": x,
        "y": y,
        "template": template,
        "n_peaks": n_peaks,
        "params": p_fit,
        "param_names": names,
        "bounds": bounds,
        "success": bool(res.success),
        "message": str(res.message),
        "result": res,
        "sigma_arr": sigma_arr,
        "sigma_eff": float(np.nanmedian(sigma_arr)),
        "y_model": y_model,
        "components": components,
        "background": bg,
        "info": info,
        "cov": cov,
        "chi2": chi2,
        "dof": dof,
        "red_chi2": float(red_chi2),
        "background_order": background_order,
        "allow_scale": allow_scale,
        "common_scale": common_scale,
        "loss": loss,
        "min_separation": float(min_separation),
        "min_observed_separation": min_observed_separation,
        "min_separation_satisfied": min_separation_satisfied,
        "separation_penalty_strength": float(separation_penalty_strength),
        "mu_bounds": mu_bounds_arr.tolist() if mu_bounds_arr is not None else None,
        "mu_bounds_text": mu_bounds_to_string(mu_bounds_arr),
    }
