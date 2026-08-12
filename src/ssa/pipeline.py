"""Headless orchestration: template construction -> fit -> uncertainty.

This is the same sequence of calls as ``ssa.gui.workers.AnalysisWorker``
(build template bank, fit, bootstrap, template-propagation), extracted into
plain, synchronous, Qt-free functions so both the GUI worker and the ``ssa
run`` CLI (``ssa.cli``) share one implementation instead of two.

Nothing here changes any default value or numerical behaviour relative to
``AnalysisWorker._run_build_template``/``_run_fit_region``: every call below
passes the same arguments, in the same order.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ssa import __version__
from ssa.config import FitConfig, TemplateBuildConfig, UncertaintyConfig
from ssa.fitting import fit_template_region
from ssa.io import make_export_rows_for_result
from ssa.templates import PeakTemplate, bootstrap_template_bank
from ssa.uncertainty import (
    bootstrap_fit,
    propagate_template_uncertainty_to_target_fit,
    quality_flag_for_fit,
    sample_count,
)

__all__ = ["build_template_bank", "run_fit_with_uncertainty", "build_run_summary"]

# Known, documented gaps in the currently-implemented pipeline. Surfaced in
# every `ssa run` JSON summary so the tool's own output stays honest about
# its current scope rather than silently implying more than it does.
KNOWN_LIMITATIONS: tuple[str, ...] = (
    "No conservative (Birge-ratio) covariance floor is applied; only raw "
    "target-bootstrap and template-propagation intervals are reported.",
    "Bootstrap templates are built with edge_mode='none' regardless of the "
    "nominal template's edge_mode; this only affects results built with a "
    "non-default edge_mode.",
    "The target-fit residual bootstrap resamples raw (not sigma-standardized) "
    "residuals; a no-op for a scalar/homoscedastic sigma, but not exact for a "
    "genuinely heteroscedastic supplied sigma column.",
)


def build_template_bank(x: np.ndarray, y: np.ndarray, config: TemplateBuildConfig) -> dict[str, Any]:
    """Build the nominal template plus its bootstrap bank from a reference region.

    Thin wrapper around :func:`ssa.templates.bootstrap_template_bank` that
    takes a :class:`~ssa.config.TemplateBuildConfig` instead of individual
    keyword arguments -- the same values, same defaults.
    """
    return bootstrap_template_bank(
        x,
        y,
        n_boot=config.n_template_boot,
        block_size=config.block_size,
        random_seed=config.random_seed,
        name=config.name,
        clip_negative=config.clip_negative,
        smooth=config.smooth,
        sg_window=config.sg_window,
        sg_poly=config.sg_poly,
        resample_factor=config.resample_factor,
        edge_mode=config.edge_mode,
        edge_width_hz=config.edge_width_hz,
        edge_fraction=config.edge_fraction,
    )


def run_fit_with_uncertainty(
    x: np.ndarray,
    y: np.ndarray,
    template: PeakTemplate,
    template_bank: list[PeakTemplate],
    fit_config: FitConfig,
    unc_config: UncertaintyConfig,
    sigma: np.ndarray | float | None = None,
) -> dict[str, Any]:
    """Fit a target region and propagate both uncertainty sources.

    Same three-step sequence as ``AnalysisWorker._run_fit_region``: the
    least-squares fit, then (if requested) the target-residual block
    bootstrap, then (if requested and a template bank is available) the
    template-shape propagation. Returns
    ``{"fit": ..., "bootstrap": ... | None, "template_propagation": ... | None}``.
    """
    fit = fit_template_region(
        x,
        y,
        template=template,
        n_peaks=fit_config.n_peaks,
        init_mus=fit_config.init_mus,
        sigma=sigma,
        background_order=fit_config.background_order,
        allow_scale=fit_config.allow_scale,
        common_scale=fit_config.common_scale,
        loss=fit_config.loss,
        min_separation=fit_config.min_separation,
        separation_penalty_strength=fit_config.separation_penalty_strength,
        mu_bounds=fit_config.mu_bounds,
    )

    rng = np.random.default_rng(unc_config.random_seed)

    boot = None
    if unc_config.run_bootstrap and unc_config.n_boot > 0:
        boot = bootstrap_fit(fit, n_boot=unc_config.n_boot, block_size=unc_config.block_size, rng=rng)

    tpl_prop = None
    if unc_config.run_template_propagation and template_bank:
        tpl_prop = propagate_template_uncertainty_to_target_fit(
            fit, template_bank, n_draws=unc_config.n_template_prop, random_seed=unc_config.random_seed
        )

    return {"fit": fit, "bootstrap": boot, "template_propagation": tpl_prop}


def build_run_summary(
    *,
    label: str,
    reference_source: dict[str, Any],
    target_source: dict[str, Any],
    template_config: TemplateBuildConfig,
    fit_config: FitConfig,
    unc_config: UncertaintyConfig,
    template_bank_result: dict[str, Any],
    analysis_result: dict[str, Any],
    sigma_source: str,
) -> dict[str, Any]:
    """Assemble the ``ssa run`` JSON summary.

    See ``KNOWN_LIMITATIONS`` above for fields this function cannot yet
    populate (calibration/systematic terms, the conservative covariance
    floor).

    ``reference_source``/``target_source`` are
    ``{"path": str, "sha256": str, "range_hz": [lo, hi] | None}``.
    """
    fit = analysis_result["fit"]
    boot = analysis_result["bootstrap"]
    tpl_prop = analysis_result["template_propagation"]
    tpl = fit["template"]

    template_registry = [
        {
            "template": tpl,
            "region_lo": reference_source.get("range_hz", [None, None])[0],
            "region_hi": reference_source.get("range_hz", [None, None])[1],
            "settings": {
                "smooth_nominal_template": template_config.smooth,
                "resample_factor": template_config.resample_factor,
            },
            "template_bank": template_bank_result.get("templates", []),
        }
    ]
    components = make_export_rows_for_result(analysis_result, label=label, template_registry=template_registry)

    residual_model = template_bank_result.get("residual_model", {})
    q16_t, q50_t, q84_t = tpl.quantile([0.16, 0.50, 0.84])

    return {
        "ssa_version": __version__,
        "label": label,
        "input": {
            "reference": reference_source,
            "target": target_source,
            "sigma_source": sigma_source,
        },
        "template_construction": {
            "name": template_config.name,
            "clip_negative": template_config.clip_negative,
            "smooth_nominal_template": template_config.smooth,
            "resample_factor": template_config.resample_factor,
            "edge_mode": template_config.edge_mode,
            "edge_width_hz": template_config.edge_width_hz,
            "edge_fraction": template_config.edge_fraction,
            "raw_area": tpl.raw_area,
            "raw_cog_hz": tpl.raw_cog,
            "raw_std_hz": tpl.raw_std,
            "quantiles_hz": {"q16": float(q16_t), "q50": float(q50_t), "q84": float(q84_t)},
            "bootstrap_bank": {
                "n_requested": template_config.n_template_boot,
                "n_valid": template_bank_result.get("summary", {}).get("n_templates", 0),
                "block_size": residual_model.get("block_size"),
                "tau_int": residual_model.get("tau_int"),
                "seed": template_config.random_seed,
            },
        },
        "fit": {
            "n_peaks": fit_config.n_peaks,
            "background_order": fit_config.background_order,
            "allow_scale": fit_config.allow_scale,
            "common_scale": fit_config.common_scale,
            "loss": fit_config.loss,
            "min_separation_hz": fit_config.min_separation,
            "mu_bounds_hz": fit.get("mu_bounds"),
            "success": fit["success"],
            "message": fit["message"],
            "chi2": fit["chi2"],
            "dof": fit["dof"],
            "red_chi2": fit["red_chi2"],
        },
        "uncertainty": {
            "bootstrap": None
            if boot is None
            else {
                "requested": boot.get("n_requested"),
                "valid": sample_count(boot),
                "block_size": boot.get("block_size"),
                "tau_int": boot.get("tau_int"),
                "seed": unc_config.random_seed,
            },
            "template_propagation": None
            if tpl_prop is None
            else {
                "requested": tpl_prop.get("n_requested"),
                "valid": sample_count(tpl_prop),
                "seed": unc_config.random_seed,
            },
        },
        "quality_flag": quality_flag_for_fit(fit, boot=boot, tpl_prop=tpl_prop),
        "components": components,
        "known_limitations": list(KNOWN_LIMITATIONS),
    }
