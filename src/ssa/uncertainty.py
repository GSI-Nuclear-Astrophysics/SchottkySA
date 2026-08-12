"""Uncertainty propagation: residual bootstrap, template-shape propagation,
component-level derived quantities, and fit quality flags.

The residual bootstrap and template-shape propagation (see
``docs/SCIENTIFIC_METHOD.md`` section 6) are the primary, on-by-default
uncertainty methods. ``quality_flag_for_fit`` also checks the *reliability*
of the bootstrap/template-propagation samples themselves (replicate-failure
rate, and whether a component's combined centroid sigma rests on only one of
the two independent sources) -- see its docstring.

See ``docs/OPEN_SCIENTIFIC_QUESTIONS.md`` items 3-5 for the statistical
conventions (covariance rescaling, quality-flag thresholds, two width
definitions) this module documents but does not change.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ssa.constants import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_FIT_BOOTSTRAP_MAX_NFEV,
    DEFAULT_FIT_BOOTSTRAP_N_BOOT_API,
    DEFAULT_N_TEMPLATE_PROPAGATION_DRAWS,
    DEFAULT_RANDOM_SEED,
    QUALITY_MIN_UNCERTAINTY_SAMPLE_FRACTION,
    QUALITY_MIN_VALID_UNCERTAINTY_SAMPLES,
    QUALITY_MU_NEAR_BOUND_FRACTION,
    QUALITY_RED_CHI2_BAD,
    QUALITY_RED_CHI2_WARN,
)
from ssa.fitting import fit_template_region
from ssa.preprocessing import circular_block_resample, estimate_block_size_from_residuals, summarize_distribution
from ssa.templates import PeakTemplate, get_template_std

__all__ = [
    "get_scale_values_from_params",
    "component_widths_from_params",
    "component_quantiles_from_fit",
    "transformed_parameter_samples",
    "summarize_sample_dict",
    "bootstrap_fit",
    "propagate_template_uncertainty_to_target_fit",
    "symmetric_sigma",
    "combined_sigma",
    "correlation_from_samples",
    "summarize_mu_delta_samples",
    "summarize_area_ratio_samples",
    "summarize_mu_corr_samples",
    "summarize_area_corr_samples",
    "sample_count",
    "quality_flag_for_fit",
]


def get_scale_values_from_params(params: np.ndarray, param_names: list[str], n_peaks: int) -> np.ndarray:
    scales = np.ones(n_peaks, dtype=float)
    for i, name in enumerate(param_names):
        if name == "log_scale_common":
            scales[:] = np.exp(params[i])
        elif name.startswith("log_scale_"):
            try:
                k = int(name.split("_")[-1])
                if 0 <= k < n_peaks:
                    scales[k] = np.exp(params[i])
            except Exception:
                pass
    return scales


def component_widths_from_params(
    params: np.ndarray, param_names: list[str], template: PeakTemplate, n_peaks: int
) -> np.ndarray:
    """Component RMS widths (second-moment, scale x template std). Distinct
    from the quantile-based sigma68 in :func:`component_quantiles_from_fit` --
    see docs/OPEN_SCIENTIFIC_QUESTIONS.md item 5."""
    scales = get_scale_values_from_params(params, param_names, n_peaks)
    return scales * get_template_std(template)


def component_quantiles_from_fit(fit: dict[str, Any]) -> list[dict[str, float]]:
    """Return q16/q50/q84 and sigma68 for each component in the same frequency units."""
    tpl = fit["template"]
    q16_t, q50_t, q84_t = tpl.quantile([0.16, 0.50, 0.84])
    rows = []
    widths = component_widths_from_params(fit["params"], fit["param_names"], tpl, fit["n_peaks"])
    mus_areas_scales = zip(fit["info"]["mus"], fit["info"]["areas"], fit["info"]["scales"], strict=False)
    for k, (mu, _area, scale) in enumerate(mus_areas_scales):
        q16 = float(mu + scale * q16_t)
        q50 = float(mu + scale * q50_t)
        q84 = float(mu + scale * q84_t)
        rows.append(
            {
                "component": k,
                "q16_Hz": q16,
                "q50_Hz": q50,
                "q84_Hz": q84,
                "sigma68_Hz": 0.5 * (q84 - q16),
                "sigma68_minus_Hz": q50 - q16,
                "sigma68_plus_Hz": q84 - q50,
                "rms_width_Hz": float(widths[k]),
            }
        )
    return rows


def transformed_parameter_samples(samples: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    samples = np.asarray(samples, dtype=float)
    out: dict[str, np.ndarray] = {}
    if samples.size == 0:
        return out
    for i, name in enumerate(names):
        vals = samples[:, i]
        if name.startswith("log_area"):
            out[name.replace("log_area", "area")] = np.exp(vals)
        elif name.startswith("log_scale"):
            out[name.replace("log_scale", "scale")] = np.exp(vals)
        else:
            out[name] = vals
    return out


def summarize_sample_dict(sample_dict: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {name: summarize_distribution(vals) for name, vals in sample_dict.items()}


def bootstrap_fit(
    fit: dict[str, Any],
    n_boot: int = DEFAULT_FIT_BOOTSTRAP_N_BOOT_API,
    block_size: str | int = DEFAULT_BLOCK_SIZE,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Residual block-bootstrap for fitted template parameters.

    Note: this function's own default is ``n_boot=300``, but the GUI always
    passes its own, separately named default of ``1000`` (its "Bootstrap
    samples" spin box) -- see ``ssa.constants.DEFAULT_FIT_BOOTSTRAP_N_BOOT_GUI``.
    """
    if rng is None:
        rng = np.random.default_rng(DEFAULT_RANDOM_SEED)

    resid = fit["y"] - fit["y_model"]
    resid = resid - np.nanmedian(resid)

    if block_size == "auto":
        used_block, tau_int, lags, acf = estimate_block_size_from_residuals(resid)
    else:
        used_block = int(block_size)
        tau_int, lags, acf = np.nan, np.array([]), np.array([])

    n_requested = int(max(0, n_boot))
    params = []
    for _ in range(n_requested):
        rb = circular_block_resample(resid, block_size=used_block, rng=rng)
        yb = fit["y_model"] + rb
        try:
            fb = fit_template_region(
                fit["x"],
                yb,
                template=fit["template"],
                n_peaks=fit["n_peaks"],
                sigma=fit["sigma_arr"],
                background_order=fit["background_order"],
                allow_scale=fit["allow_scale"],
                common_scale=fit["common_scale"],
                p0_override=fit["params"],
                loss=fit["loss"],
                max_nfev=DEFAULT_FIT_BOOTSTRAP_MAX_NFEV,
                min_separation=float(fit.get("min_separation", 0.0)),
                separation_penalty_strength=float(fit.get("separation_penalty_strength", 1.0e6)),
                mu_bounds=fit.get("mu_bounds"),
            )
            params.append(fb["params"])
        except Exception:
            continue

    params_arr = np.asarray(params, dtype=float)
    summary = summarize_sample_dict(transformed_parameter_samples(params_arr, fit["param_names"]))

    width_samples = (
        np.asarray(
            [component_widths_from_params(p, fit["param_names"], fit["template"], fit["n_peaks"]) for p in params_arr]
        )
        if len(params_arr)
        else np.empty((0, fit["n_peaks"]))
    )

    width_summary = (
        {f"width_{k}": summarize_distribution(width_samples[:, k]) for k in range(fit["n_peaks"])}
        if len(width_samples)
        else {}
    )

    return {
        "params": params_arr,
        "param_names": fit["param_names"],
        "summary": summary,
        "width_samples": width_samples,
        "width_summary": width_summary,
        "block_size": used_block,
        "tau_int": tau_int,
        "lags": lags,
        "acf": acf,
        # Attempted replicate count, distinct from len(params) (= valid
        # count): lets quality_flag_for_fit detect a high replicate-failure
        # rate even when the absolute valid count still looks large.
        "n_requested": n_requested,
    }


def propagate_template_uncertainty_to_target_fit(
    nominal_fit: dict[str, Any],
    template_bank: list[PeakTemplate],
    n_draws: int | None = DEFAULT_N_TEMPLATE_PROPAGATION_DRAWS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    """Refit a target region with bootstrap templates to estimate template-shape uncertainty."""
    rng = np.random.default_rng(random_seed)
    if not template_bank:
        return {
            "params": np.empty((0, len(nominal_fit["params"]))),
            "summary": {},
            "width_summary": {},
            "width_samples": np.empty((0, nominal_fit["n_peaks"])),
            "n_requested": 0,
        }

    if n_draws is None or n_draws <= 0 or n_draws >= len(template_bank):
        selected = np.arange(len(template_bank))
    else:
        selected = rng.choice(len(template_bank), size=int(n_draws), replace=False)

    params = []
    width_samples = []
    used_template_indices = []

    for idx in selected:
        tpl = template_bank[int(idx)]
        try:
            fb = fit_template_region(
                nominal_fit["x"],
                nominal_fit["y"],
                template=tpl,
                n_peaks=nominal_fit["n_peaks"],
                sigma=nominal_fit["sigma_arr"],
                background_order=nominal_fit["background_order"],
                allow_scale=nominal_fit["allow_scale"],
                common_scale=nominal_fit["common_scale"],
                p0_override=nominal_fit["params"],
                loss=nominal_fit["loss"],
                max_nfev=DEFAULT_FIT_BOOTSTRAP_MAX_NFEV,
                min_separation=float(nominal_fit.get("min_separation", 0.0)),
                separation_penalty_strength=float(nominal_fit.get("separation_penalty_strength", 1.0e6)),
                mu_bounds=nominal_fit.get("mu_bounds"),
            )
            params.append(fb["params"])
            width_samples.append(component_widths_from_params(fb["params"], fb["param_names"], tpl, fb["n_peaks"]))
            used_template_indices.append(int(idx))
        except Exception:
            continue

    params_arr = np.asarray(params, dtype=float)
    width_arr = np.asarray(width_samples, dtype=float)
    summary = summarize_sample_dict(transformed_parameter_samples(params_arr, nominal_fit["param_names"]))
    width_summary = (
        {f"width_{k}": summarize_distribution(width_arr[:, k]) for k in range(nominal_fit["n_peaks"])}
        if len(width_arr)
        else {}
    )

    return {
        "params": params_arr,
        "param_names": nominal_fit["param_names"],
        "summary": summary,
        "width_samples": width_arr,
        "width_summary": width_summary,
        "selected_template_indices": used_template_indices,
        # Attempted count (len(selected)), distinct from len(params) (valid
        # count) -- see the matching comment in bootstrap_fit.
        "n_requested": int(len(selected)),
    }


def symmetric_sigma(summary: dict[str, float] | None) -> float:
    if not summary or summary.get("n", 0) == 0:
        return float("nan")
    return float(0.5 * (summary.get("minus", np.nan) + summary.get("plus", np.nan)))


def combined_sigma(*summaries: dict[str, float] | None) -> float:
    """Combine independent symmetrised sigmas in quadrature, ignoring non-finite
    contributors. An approximation -- see docs/SCIENTIFIC_METHOD.md section 6."""
    vals = [symmetric_sigma(s) for s in summaries]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(np.sqrt(np.sum(np.asarray(vals) ** 2)))


def correlation_from_samples(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation of two sample vectors, robust to invalid/constant samples."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(m) < 3:
        return float("nan")
    x = x[m]
    y = y[m]
    if np.nanstd(x) <= 0 or np.nanstd(y) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def summarize_mu_delta_samples(source: dict[str, Any] | None, k: int, ref: int) -> dict[str, float] | None:
    """Summarize mu_k - mu_ref from a bootstrap/template-propagation result dict."""
    if not source or "params" not in source:
        return None
    params = np.asarray(source.get("params"), dtype=float)
    if params.ndim != 2 or params.shape[0] == 0 or params.shape[1] <= max(k, ref):
        return None
    return summarize_distribution(params[:, k] - params[:, ref])


def summarize_area_ratio_samples(
    source: dict[str, Any] | None, k: int, ref: int, n_peaks: int
) -> dict[str, float] | None:
    """Summarize area_k / area_ref using log-area parameter samples."""
    if not source or "params" not in source:
        return None
    params = np.asarray(source.get("params"), dtype=float)
    if params.ndim != 2 or params.shape[0] == 0:
        return None
    i_k = n_peaks + k
    i_r = n_peaks + ref
    if params.shape[1] <= max(i_k, i_r):
        return None
    return summarize_distribution(np.exp(params[:, i_k] - params[:, i_r]))


def summarize_mu_corr_samples(source: dict[str, Any] | None, k: int, ref: int) -> float:
    """Correlation between mu_k and mu_ref from sample cloud."""
    if not source or "params" not in source:
        return float("nan")
    params = np.asarray(source.get("params"), dtype=float)
    if params.ndim != 2 or params.shape[0] == 0 or params.shape[1] <= max(k, ref):
        return float("nan")
    return correlation_from_samples(params[:, k], params[:, ref])


def summarize_area_corr_samples(source: dict[str, Any] | None, k: int, ref: int, n_peaks: int) -> float:
    """Correlation between area_k and area_ref from log-area sample cloud."""
    if not source or "params" not in source:
        return float("nan")
    params = np.asarray(source.get("params"), dtype=float)
    i_k = n_peaks + k
    i_r = n_peaks + ref
    if params.ndim != 2 or params.shape[0] == 0 or params.shape[1] <= max(i_k, i_r):
        return float("nan")
    return correlation_from_samples(np.exp(params[:, i_k]), np.exp(params[:, i_r]))


def sample_count(source: dict[str, Any] | None) -> int:
    if not source:
        return 0
    params = source.get("params")
    if params is None:
        return 0
    arr = np.asarray(params)
    if arr.ndim == 0:
        return 0
    return int(arr.shape[0])


def _uncertainty_source_flags(source: dict[str, Any] | None, label: str) -> list[str]:
    """Flag a bootstrap/template-propagation result as failed or unreliable.

    Returns ``[]`` if the source was not requested (``source is None`` --
    that is a deliberate analysis choice, not a failure) or is healthy.
    Otherwise compares the number of *valid* replicates (``sample_count``)
    against both an absolute floor and a minimum success fraction of the
    number *attempted* (``source["n_requested"]``) -- see
    ``ssa.constants.QUALITY_MIN_VALID_UNCERTAINTY_SAMPLES``/
    ``QUALITY_MIN_UNCERTAINTY_SAMPLE_FRACTION``.
    """
    if source is None:
        return []
    n_valid = sample_count(source)
    n_requested = int(source.get("n_requested", n_valid))
    if n_requested <= 0:
        return []
    if n_valid == 0:
        return [f"{label}_failed"]
    fraction = n_valid / n_requested
    if n_valid < QUALITY_MIN_VALID_UNCERTAINTY_SAMPLES or fraction < QUALITY_MIN_UNCERTAINTY_SAMPLE_FRACTION:
        return [f"{label}_insufficient_samples"]
    return []


def quality_flag_for_fit(
    fit: dict[str, Any],
    boot: dict[str, Any] | None = None,
    tpl_prop: dict[str, Any] | None = None,
) -> str:
    """Compact quality flags for publication review.

    ``boot``/``tpl_prop`` are the optional result dicts from
    :func:`bootstrap_fit`/:func:`propagate_template_uncertainty_to_target_fit`
    for this same fit. Passing them enables three additional checks that a
    fit-only quality flag cannot make: a source failing outright
    (``*_failed``), a source completing but with too few/too-unreliable
    replicates (``*_insufficient_samples``), and -- the check most directly
    relevant to trusting a reported centroid uncertainty --
    ``mu_combined_sigma_single_source``, raised when a component's combined
    centroid sigma (:func:`combined_sigma`) is, in practice, resting on only
    one of the two independent uncertainty sources (the other being absent,
    failed, or otherwise non-finite for that component). A tight combined
    sigma is not evidence the two methods agree if only one of them actually
    contributed a number.

    Thresholds are this project's own convention, not a literature value --
    see docs/OPEN_SCIENTIFIC_QUESTIONS.md item 4.
    """
    flags = []
    red = float(fit.get("red_chi2", np.nan))
    success = bool(fit.get("success", False))
    if not success:
        flags.append("fit_not_converged")
    if np.isfinite(red) and red > QUALITY_RED_CHI2_BAD:
        flags.append("red_chi2_gt_5")
    elif np.isfinite(red) and red > QUALITY_RED_CHI2_WARN:
        flags.append("red_chi2_gt_2")
    if fit.get("min_separation", 0.0) > 0 and not fit.get("min_separation_satisfied", True):
        flags.append("min_separation_violated")
    if fit.get("mu_bounds"):
        try:
            mb = np.asarray(fit.get("mu_bounds"), dtype=float)
            mus = np.asarray(fit.get("info", {}).get("mus", []), dtype=float)
            if len(mus) == len(mb):
                span = np.maximum(mb[:, 1] - mb[:, 0], 1e-30)
                near = ((mus - mb[:, 0]) / span < QUALITY_MU_NEAR_BOUND_FRACTION) | (
                    (mb[:, 1] - mus) / span < QUALITY_MU_NEAR_BOUND_FRACTION
                )
                if np.any(near):
                    flags.append("mu_near_search_bound")
        except Exception:
            pass
    if int(fit.get("dof", 0)) <= 0:
        flags.append("nonpositive_dof")

    flags.extend(_uncertainty_source_flags(boot, "bootstrap"))
    flags.extend(_uncertainty_source_flags(tpl_prop, "template_propagation"))

    if boot is not None or tpl_prop is not None:
        for k in range(int(fit.get("n_peaks", 0))):
            mu_name = f"mu_{k}"
            boot_mu = boot.get("summary", {}).get(mu_name) if boot else None
            tpl_mu = tpl_prop.get("summary", {}).get(mu_name) if tpl_prop else None
            n_sources = int(np.isfinite(symmetric_sigma(boot_mu))) + int(np.isfinite(symmetric_sigma(tpl_mu)))
            if n_sources <= 1:
                flags.append("mu_combined_sigma_single_source")
                break

    return ";".join(dict.fromkeys(flags)) if flags else "ok"
