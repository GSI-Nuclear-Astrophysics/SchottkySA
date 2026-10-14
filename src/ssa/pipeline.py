"""Qt-independent template construction, fitting and uncertainty orchestration.

The command line and GUI workers use this common pipeline.
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

# Record uncertainty-model assumptions with each exported fit.
KNOWN_LIMITATIONS: tuple[str, ...] = (
    "No conservative (Birge-ratio) covariance floor is applied; only raw "
    "target-bootstrap and template-propagation intervals are reported.",
    "The target-fit residual bootstrap resamples raw (not sigma-standardized) "
    "residuals; a no-op for a scalar/homoscedastic sigma, but not exact for a "
    "genuinely heteroscedastic supplied sigma column.",
)


def build_template_bank(x: np.ndarray, y: np.ndarray, config: TemplateBuildConfig) -> dict[str, Any]:
    """Build a nominal reference and its bootstrap bank from template settings."""
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
        left_edge_points=config.left_edge_points,
        right_edge_points=config.right_edge_points,
        left_tail_width_hz=config.left_tail_width_hz,
        right_tail_width_hz=config.right_tail_width_hz,
    )


def run_fit_with_uncertainty(
    x: np.ndarray,
    y: np.ndarray,
    template: PeakTemplate,
    template_bank: list[PeakTemplate],
    fit_config: FitConfig,
    unc_config: UncertaintyConfig,
    sigma: np.ndarray | float | None = None,
    component_templates: list[PeakTemplate] | None = None,
    component_template_banks: list[list[PeakTemplate]] | None = None,
) -> dict[str, Any]:
    """Fit a target and, when selected, evaluate target and reference resampling.

    Returns the nominal fit and each requested uncertainty stream separately.
    """
    fit = fit_template_region(
        x,
        y,
        template=template,
        component_templates=component_templates,
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
    if unc_config.run_template_propagation and (template_bank or component_template_banks is not None):
        tpl_prop = propagate_template_uncertainty_to_target_fit(
            fit,
            template_bank,
            n_draws=unc_config.n_template_prop,
            random_seed=unc_config.random_seed,
            component_template_banks=component_template_banks,
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
            "left_edge_points": template_config.left_edge_points,
            "right_edge_points": template_config.right_edge_points,
            "left_tail_width_hz": template_config.left_tail_width_hz,
            "right_tail_width_hz": template_config.right_tail_width_hz,
            "edge_info": tpl.edge_info,
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
