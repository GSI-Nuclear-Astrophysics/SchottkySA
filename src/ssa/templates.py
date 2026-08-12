"""Empirical peak-template construction and template-shape bootstrap.

See ``docs/SCIENTIFIC_METHOD.md`` section 3 (template construction) and
section 6 (uncertainty quantification) for the scientific rationale, and
``docs/OPEN_SCIENTIFIC_QUESTIONS.md`` for known caveats. Numerical behaviour
here is part of the published fitting contract; changes should be made
deliberately and documented in those files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

from ssa.constants import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_CLIP_NEGATIVE,
    DEFAULT_EDGE_FRACTION,
    DEFAULT_EDGE_MODE,
    DEFAULT_EDGE_WIDTH_HZ,
    DEFAULT_MAX_ACF_LAG,
    DEFAULT_N_TEMPLATE_BOOT,
    DEFAULT_RANDOM_SEED,
    DEFAULT_RESAMPLE_FACTOR,
    DEFAULT_SG_POLY,
    DEFAULT_SMOOTH_NOMINAL_TEMPLATE,
)
from ssa.exceptions import TemplateConstructionError
from ssa.preprocessing import (
    circular_block_resample,
    estimate_block_size_from_residuals,
    make_odd,
    prepare_xy,
    summarize_distribution,
    trapz,
)

__all__ = [
    "PeakTemplate",
    "get_template_std",
    "apply_template_edge_handling",
    "build_peak_template",
    "estimate_reference_residual_model",
    "bootstrap_template_bank",
]


@dataclass
class PeakTemplate:
    """A normalized, COG-centred empirical peak shape (PDF + CDF on grid ``u``).

    ``u`` is expressed in the same units as the original frequency axis
    (typically Hz), relative to the template's centre of gravity (COG=0).
    """

    u: np.ndarray
    pdf: np.ndarray
    cdf: np.ndarray
    raw_area: float
    raw_cog: float
    raw_std: float
    dx: float
    name: str = "template"
    edge_info: dict[str, Any] = field(default_factory=dict)

    def evaluate_pdf(self, z: np.ndarray) -> np.ndarray:
        return np.interp(z, self.u, self.pdf, left=0.0, right=0.0)

    def evaluate_component(self, x: np.ndarray, mu: float, area: float, scale: float = 1.0) -> np.ndarray:
        """Area-preserving, scale-invariant component: ``(area/scale) * pdf((x-mu)/scale)``."""
        scale = max(float(scale), 1e-12)
        return (area / scale) * self.evaluate_pdf((x - mu) / scale)

    def moments(self) -> tuple[float, float]:
        m = trapz(self.u * self.pdf, self.u)
        v = trapz((self.u - m) ** 2 * self.pdf, self.u)
        return m, float(np.sqrt(max(v, 0.0)))

    def quantile(self, q: np.ndarray | list[float] | float) -> Any:
        """Inverse-CDF lookup; returns an ndarray for array-like `q`, a scalar for a single `q`."""
        return np.interp(q, self.cdf, self.u)


def get_template_std(template: PeakTemplate) -> float:
    _, std = template.moments()
    return float(std)


def pchip_tail_to_zero_one_side(
    x: np.ndarray,
    y: np.ndarray,
    side: str,
    n_edge: int,
    dx: float,
    tail_width_hz: float,
    monotone_envelope: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Non-parametric tail continuation to zero using only the local edge shape.

    This is deliberately not an exponential or any other analytic tail model.
    It constructs a shape-preserving PCHIP interpolant through:

        [last/first observed edge segment] + [one zero anchor outside the ROI]

    and evaluates only the extra outside-ROI part. The result is therefore a
    smooth, data-driven closure of the selected ROI to zero.

    If monotone_envelope=True, the edge segment is first converted into a
    monotone non-increasing envelope toward the boundary. This is safer when the
    edge segment has small noise oscillations, but it slightly modifies the local
    shape used to determine the closure.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    side = side.lower()
    info: dict[str, Any] = {
        f"{side}_tail_model": "pchip_to_zero",
        f"{side}_tail_extra_points": 0,
        f"{side}_tail_extra_width_hz": 0.0,
        f"{side}_tail_edge_value": float(y[0] if side == "left" else y[-1]),
        f"{side}_tail_area_added": 0.0,
        f"{side}_tail_monotone_envelope": bool(monotone_envelope),
    }

    if len(x) < max(5, n_edge + 2):
        info[f"{side}_tail_model"] = "none_too_few_points"
        return np.array([]), np.array([]), info

    tail_width_hz = float(max(tail_width_hz, 3.0 * dx))
    n_extra = int(np.clip(np.ceil(tail_width_hz / dx), 3, 5000))
    actual_width = float(n_extra * dx)

    if side == "right":
        x_seg = x[-n_edge:].copy()
        y_seg = y[-n_edge:].copy()
        # Optional monotone envelope toward the right boundary.
        if monotone_envelope:
            y_seg = np.minimum.accumulate(y_seg)
        x_anchor = x[-1] + actual_width
        xp = np.concatenate([x_seg, [x_anchor]])
        yp = np.concatenate([y_seg, [0.0]])
        x_extra = x[-1] + dx * np.arange(1, n_extra + 1)
    else:
        x_seg = x[:n_edge].copy()
        y_seg = y[:n_edge].copy()
        # Optional monotone envelope toward the left boundary.
        # In left-to-right order, values should rise from the zero anchor into
        # the peak; enforce non-decreasing away from left boundary if requested.
        if monotone_envelope:
            y_seg = np.maximum.accumulate(y_seg)
        x_anchor = x[0] - actual_width
        xp = np.concatenate([[x_anchor], x_seg])
        yp = np.concatenate([[0.0], y_seg])
        x_extra = x[0] - dx * np.arange(n_extra, 0, -1)

    # Ensure strictly increasing support.
    order = np.argsort(xp)
    xp = xp[order]
    yp = yp[order]

    try:
        interp = PchipInterpolator(xp, yp, extrapolate=False)
        y_extra = interp(x_extra)
        y_extra = np.nan_to_num(y_extra, nan=0.0, posinf=0.0, neginf=0.0)
        # The template builder will clip negative values later if clip_negative
        # is enabled. Here we also clip the added closure to avoid unphysical
        # negative tail area from small interpolation overshoots.
        y_extra = np.clip(y_extra, 0.0, None)
    except Exception:
        # Non-parametric fallback: a direct linear closure to zero.
        if side == "right":
            edge_y = max(float(y[-1]), 0.0)
            y_extra = edge_y * np.maximum(1.0 - np.arange(1, n_extra + 1) / n_extra, 0.0)
        else:
            edge_y = max(float(y[0]), 0.0)
            y_extra = edge_y * np.maximum(np.arange(1, n_extra + 1) / n_extra, 0.0)
        info[f"{side}_tail_model"] = "linear_to_zero_fallback"

    info[f"{side}_tail_extra_points"] = int(n_extra)
    info[f"{side}_tail_extra_width_hz"] = actual_width
    # Routed through trapz() (a thin wrapper) rather than calling np.trapz
    # directly, to avoid NumPy's trapz deprecation warning; numerically
    # identical to the composite trapezoidal rule.
    info[f"{side}_tail_area_added"] = trapz(y_extra, x_extra) if len(x_extra) > 1 else 0.0
    return x_extra, y_extra, info


def apply_template_edge_handling(
    x: np.ndarray,
    y: np.ndarray,
    mode: str = DEFAULT_EDGE_MODE,
    edge_width_hz: float = DEFAULT_EDGE_WIDTH_HZ,
    edge_fraction: float = DEFAULT_EDGE_FRACTION,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Optionally modify a reference peak before empirical-template construction.

    Modes
    -----
    none
        Current/default behavior. The selected ROI defines the template support.
        The template evaluates to zero outside this support.

    pad_to_zero
        Add one zero-valued anchor point outside each ROI edge. This lets the
        interpolated template tail decay to zero just outside the selected ROI
        without changing the measured values inside the ROI.

    pchip_extrapolate_to_zero
        Non-parametric edge closure. A PCHIP interpolant is constructed through
        the observed edge segment plus a zero anchor outside the ROI. The added
        outside-ROI tail follows the local edge shape and reaches zero at the
        anchor. No exponential or analytic tail model is used.

    pchip_monotone_extrapolate_to_zero
        Same as pchip_extrapolate_to_zero, but first converts the edge segment
        into a monotone envelope to suppress noise oscillations near the edge.

    cosine_taper_to_zero
        Multiply the first/last edge_width by a half-cosine taper so that the
        selected ROI itself goes to zero at both boundaries. This is stronger
        and can move the COG; use as a systematic check.

    subtract_edge_baseline_pad_to_zero
        Estimate a constant baseline from the first/last edge_fraction of the
        ROI, subtract it, then add zero anchors outside the ROI.

    subtract_edge_baseline_pchip_extrapolate_to_zero
        Estimate a constant edge baseline, subtract it, then apply the
        non-parametric PCHIP closure to zero.

    subtract_edge_baseline_pchip_monotone_extrapolate_to_zero
        Baseline-subtracted non-parametric PCHIP closure using a monotone
        envelope near the edges.

    subtract_edge_baseline_cosine_taper_to_zero
        Estimate a constant baseline from the edges, subtract it, then apply a
        cosine taper to zero at both boundaries.

    Notes
    -----
    If the non-zero edge is a baseline/contaminant, baseline subtraction plus
    padding is safer. If it is a genuine truncated physical tail and the tail is
    locally clean, pchip_extrapolate_to_zero gives a non-parametric closure.
    Compare edge modes and treat the COG spread as a template-edge systematic.
    """
    x, y = prepare_xy(x, y)
    mode = (mode or "none").strip().lower()
    info: dict[str, Any] = {
        "edge_mode": mode,
        "edge_width_hz": float(edge_width_hz),
        "edge_fraction": float(edge_fraction),
        "edge_baseline_subtracted": 0.0,
        "edge_points_used": 0,
        "zero_anchor_width_hz": 0.0,
        "left_tail_model": "none",
        "right_tail_model": "none",
        "left_tail_area_added": 0.0,
        "right_tail_area_added": 0.0,
    }

    if mode in ("", "none", "off"):
        return x, y, info

    y_work = np.asarray(y, dtype=float).copy()
    dx = float(np.nanmedian(np.diff(x)))
    if not np.isfinite(dx) or dx <= 0:
        raise TemplateConstructionError("Invalid frequency grid spacing for edge handling.")

    n_edge = (
        int(round(float(edge_width_hz) / dx))
        if edge_width_hz and edge_width_hz > 0
        else int(round(float(edge_fraction) * len(x)))
    )
    n_edge = int(np.clip(n_edge, 2, max(2, len(x) // 2 - 1)))
    width_hz = float(edge_width_hz) if edge_width_hz and edge_width_hz > 0 else float(n_edge * dx)
    width_hz = max(width_hz, 3.0 * dx)
    info["edge_width_hz"] = width_hz
    info["edge_points_used"] = n_edge

    if "subtract_edge_baseline" in mode:
        edge_vals = np.concatenate([y_work[:n_edge], y_work[-n_edge:]])
        baseline = float(np.nanmedian(edge_vals))
        if np.isfinite(baseline):
            y_work = y_work - baseline
            info["edge_baseline_subtracted"] = baseline

    if "cosine_taper" in mode:
        weights = np.ones_like(y_work, dtype=float)
        t = np.linspace(0.0, np.pi, n_edge)
        left = 0.5 * (1.0 - np.cos(t))
        right = left[::-1]
        weights[:n_edge] *= left
        weights[-n_edge:] *= right
        y_work = y_work * weights

    if "pchip" in mode and "extrapolate_to_zero" in mode:
        monotone = "monotone" in mode
        left_x, left_y, left_info = pchip_tail_to_zero_one_side(
            x, y_work, "left", n_edge, dx, tail_width_hz=width_hz, monotone_envelope=monotone
        )
        right_x, right_y, right_info = pchip_tail_to_zero_one_side(
            x, y_work, "right", n_edge, dx, tail_width_hz=width_hz, monotone_envelope=monotone
        )
        info.update(left_info)
        info.update(right_info)
        parts_x = []
        parts_y = []
        if len(left_x):
            parts_x.append(left_x)
            parts_y.append(left_y)
        parts_x.append(x)
        parts_y.append(y_work)
        if len(right_x):
            parts_x.append(right_x)
            parts_y.append(right_y)
        return np.concatenate(parts_x), np.concatenate(parts_y), info

    # Backward-compatible alias: if an old setting says extrapolate_to_zero,
    # use the non-parametric PCHIP closure rather than an analytic exponential.
    if "extrapolate_to_zero" in mode:
        left_x, left_y, left_info = pchip_tail_to_zero_one_side(
            x, y_work, "left", n_edge, dx, tail_width_hz=width_hz, monotone_envelope=False
        )
        right_x, right_y, right_info = pchip_tail_to_zero_one_side(
            x, y_work, "right", n_edge, dx, tail_width_hz=width_hz, monotone_envelope=False
        )
        info.update(left_info)
        info.update(right_info)
        parts_x = []
        parts_y = []
        if len(left_x):
            parts_x.append(left_x)
            parts_y.append(left_y)
        parts_x.append(x)
        parts_y.append(y_work)
        if len(right_x):
            parts_x.append(right_x)
            parts_y.append(right_y)
        return np.concatenate(parts_x), np.concatenate(parts_y), info

    if "pad_to_zero" in mode:
        anchor_width = width_hz if width_hz > 0 else 3.0 * dx
        anchor_width = max(anchor_width, dx)
        x_ext = np.concatenate([[x[0] - anchor_width], x, [x[-1] + anchor_width]])
        y_ext = np.concatenate([[0.0], y_work, [0.0]])
        info["zero_anchor_width_hz"] = float(anchor_width)
        return x_ext, y_ext, info

    return x, y_work, info


def build_peak_template(
    x: np.ndarray,
    y: np.ndarray,
    name: str = "reference",
    clip_negative: bool = DEFAULT_CLIP_NEGATIVE,
    smooth: bool = DEFAULT_SMOOTH_NOMINAL_TEMPLATE,
    sg_window: int | None = None,
    sg_poly: int = DEFAULT_SG_POLY,
    resample_factor: int = DEFAULT_RESAMPLE_FACTOR,
    edge_mode: str = DEFAULT_EDGE_MODE,
    edge_width_hz: float = DEFAULT_EDGE_WIDTH_HZ,
    edge_fraction: float = DEFAULT_EDGE_FRACTION,
) -> PeakTemplate:
    """
    Build a normalized, COG-centered empirical peak template.

    Publication default:
      smooth=False, resample_factor=16.
    This avoids washing out sharp Schottky peak structure.
    """
    x, y = prepare_xy(x, y)
    if len(x) < 7:
        raise TemplateConstructionError("Reference region is too small to build a stable template.")

    y_work = y.copy()
    if smooth:
        if sg_window is None:
            sg_window = make_odd(len(y_work) // 15, minimum=7)
        sg_window = min(make_odd(sg_window), make_odd(len(y_work) - 1, minimum=5))
        if sg_window >= len(y_work):
            sg_window = make_odd(len(y_work) - 2, minimum=5)
        y_work = savgol_filter(y_work, window_length=sg_window, polyorder=min(sg_poly, sg_window - 2))

    x, y_work, edge_info = apply_template_edge_handling(
        x,
        y_work,
        mode=edge_mode,
        edge_width_hz=edge_width_hz,
        edge_fraction=edge_fraction,
    )

    y_pos = np.clip(y_work, 0.0, None) if clip_negative else y_work - min(0.0, float(np.nanmin(y_work)))

    area = trapz(y_pos, x)
    if not np.isfinite(area) or area <= 0:
        raise TemplateConstructionError(
            "Template area is non-positive. Check the selected region/background subtraction."
        )

    cog = trapz(x * y_pos, x) / area
    var = trapz((x - cog) ** 2 * y_pos, x) / area
    std = float(np.sqrt(max(var, 0.0)))

    u_raw = x - cog
    pdf_raw = y_pos / area

    dx = float(np.nanmedian(np.diff(x)))
    if not np.isfinite(dx) or dx <= 0:
        raise TemplateConstructionError("Invalid frequency grid spacing.")

    du = dx / float(max(1, int(resample_factor)))
    u = np.arange(u_raw.min(), u_raw.max() + 0.5 * du, du)

    interp = PchipInterpolator(u_raw, pdf_raw, extrapolate=False)
    pdf = interp(u)
    pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
    pdf = np.clip(pdf, 0.0, None)

    pdf_area = trapz(pdf, u)
    if pdf_area <= 0:
        raise TemplateConstructionError("Interpolated template area is non-positive.")
    pdf /= pdf_area

    # Recenter the interpolated template to exact COG = 0.
    m = trapz(u * pdf, u)
    u = u - m
    pdf_area = trapz(pdf, u)
    pdf /= pdf_area

    cdf = cumulative_trapezoid(pdf, u, initial=0.0)
    if cdf[-1] <= 0:
        raise TemplateConstructionError("Invalid template CDF.")
    cdf /= cdf[-1]

    return PeakTemplate(
        u=u, pdf=pdf, cdf=cdf, raw_area=area, raw_cog=cog, raw_std=std, dx=dx, name=name, edge_info=edge_info
    )


def estimate_reference_residual_model(
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    smooth_window: int | None = None,
    sg_poly: int = DEFAULT_SG_POLY,
    block_size: str | int = DEFAULT_BLOCK_SIZE,
    max_lag: int = DEFAULT_MAX_ACF_LAG,
) -> dict[str, Any]:
    """
    Smooth a reference peak and estimate residual block size for template bootstrap.

    This smoothing is for residual/noise modelling only. It is separate from
    nominal template construction.
    """
    x_ref, y_ref = prepare_xy(x_ref, y_ref)

    if smooth_window is None:
        smooth_window = make_odd(len(y_ref) // 15, minimum=7)
    smooth_window = min(make_odd(smooth_window), make_odd(len(y_ref) - 1, minimum=5))
    if smooth_window >= len(y_ref):
        smooth_window = make_odd(len(y_ref) - 2, minimum=5)

    y_smooth = savgol_filter(y_ref, window_length=smooth_window, polyorder=min(sg_poly, smooth_window - 2))
    residuals = y_ref - y_smooth
    residuals = residuals - np.nanmedian(residuals)

    auto_block, tau_int, lags, acf = estimate_block_size_from_residuals(residuals, max_lag=max_lag)
    used_block = auto_block if block_size == "auto" else int(block_size)

    return {
        "x": x_ref,
        "y": y_ref,
        "y_smooth": y_smooth,
        "residuals": residuals,
        "lags": lags,
        "acf": acf,
        "tau_int": tau_int,
        "block_size": int(max(1, used_block)),
        "smooth_window": smooth_window,
    }


def bootstrap_template_bank(
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    n_boot: int = DEFAULT_N_TEMPLATE_BOOT,
    block_size: str | int = DEFAULT_BLOCK_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
    name: str = "reference_peak",
    clip_negative: bool = DEFAULT_CLIP_NEGATIVE,
    smooth: bool = DEFAULT_SMOOTH_NOMINAL_TEMPLATE,
    sg_window: int | None = None,
    sg_poly: int = DEFAULT_SG_POLY,
    resample_factor: int = DEFAULT_RESAMPLE_FACTOR,
    edge_mode: str = DEFAULT_EDGE_MODE,
    edge_width_hz: float = DEFAULT_EDGE_WIDTH_HZ,
    edge_fraction: float = DEFAULT_EDGE_FRACTION,
) -> dict[str, Any]:
    """Build a nominal template plus bootstrap template bank."""
    rng = np.random.default_rng(random_seed)

    nominal = build_peak_template(
        x_ref,
        y_ref,
        name=name,
        clip_negative=clip_negative,
        smooth=smooth,
        sg_window=sg_window,
        sg_poly=sg_poly,
        resample_factor=resample_factor,
        edge_mode=edge_mode,
        edge_width_hz=edge_width_hz,
        edge_fraction=edge_fraction,
    )

    # Residual model uses the SG window, even if the nominal template is unsmoothed.
    residual_model = estimate_reference_residual_model(
        x_ref, y_ref, smooth_window=sg_window, sg_poly=sg_poly, block_size=block_size
    )

    x_ref = residual_model["x"]
    y_smooth = residual_model["y_smooth"]
    residuals = residual_model["residuals"]
    used_block = residual_model["block_size"]

    templates: list[PeakTemplate] = []
    raw_cog, raw_std, raw_area = [], [], []
    q16, q50, q84 = [], [], []

    for i in range(int(max(0, n_boot))):
        rb = circular_block_resample(residuals, block_size=used_block, rng=rng)
        yb = y_smooth + rb
        try:
            # Bootstrap templates are not smoothed again; the pseudo-reference
            # shape already contains smooth reference + resampled residuals.
            tpl = build_peak_template(
                x_ref,
                yb,
                name=f"{name}_boot_{i}",
                clip_negative=clip_negative,
                smooth=False,
                sg_window=None,
                sg_poly=sg_poly,
                resample_factor=resample_factor,
            )
            templates.append(tpl)
            raw_cog.append(tpl.raw_cog)
            raw_std.append(get_template_std(tpl))
            raw_area.append(tpl.raw_area)
            qs = tpl.quantile([0.16, 0.50, 0.84])
            q16.append(qs[0])
            q50.append(qs[1])
            q84.append(qs[2])
        except Exception:
            continue

    summary = {
        "raw_cog": summarize_distribution(np.asarray(raw_cog)),
        "raw_std": summarize_distribution(np.asarray(raw_std)),
        "raw_area": summarize_distribution(np.asarray(raw_area)),
        "q16": summarize_distribution(np.asarray(q16)),
        "q50": summarize_distribution(np.asarray(q50)),
        "q84": summarize_distribution(np.asarray(q84)),
        "n_templates": len(templates),
    }

    return {"template_nominal": nominal, "templates": templates, "summary": summary, "residual_model": residual_model}
