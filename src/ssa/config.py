"""Typed configuration objects for the template/fit/uncertainty pipeline.

Named, typed fields (rather than plain ``dict`` payloads) make the Python
API usable without constructing those dicts by hand. Default values are
defined once in ``ssa.constants`` and imported here.
"""

from __future__ import annotations

from dataclasses import dataclass

from ssa.constants import (
    DEFAULT_ALLOW_SCALE,
    DEFAULT_BACKGROUND_ORDER,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_CLIP_NEGATIVE,
    DEFAULT_COMMON_SCALE,
    DEFAULT_EDGE_FRACTION,
    DEFAULT_EDGE_MODE,
    DEFAULT_EDGE_WIDTH_HZ,
    DEFAULT_FIT_BOOTSTRAP_N_BOOT_GUI,
    DEFAULT_LOSS,
    DEFAULT_MIN_SEPARATION_HZ,
    DEFAULT_N_PEAKS,
    DEFAULT_N_TEMPLATE_BOOT,
    DEFAULT_N_TEMPLATE_PROPAGATION_DRAWS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_RESAMPLE_FACTOR,
    DEFAULT_SEPARATION_PENALTY_STRENGTH,
    DEFAULT_SG_POLY,
    DEFAULT_SMOOTH_NOMINAL_TEMPLATE,
)

__all__ = ["TemplateBuildConfig", "FitConfig", "UncertaintyConfig"]


@dataclass
class TemplateBuildConfig:
    """Parameters for :func:`ssa.templates.bootstrap_template_bank`."""

    name: str = "reference_peak"
    n_template_boot: int = DEFAULT_N_TEMPLATE_BOOT
    block_size: str | int = DEFAULT_BLOCK_SIZE
    random_seed: int = DEFAULT_RANDOM_SEED
    clip_negative: bool = DEFAULT_CLIP_NEGATIVE
    smooth: bool = DEFAULT_SMOOTH_NOMINAL_TEMPLATE
    sg_window: int | None = None
    sg_poly: int = DEFAULT_SG_POLY
    resample_factor: int = DEFAULT_RESAMPLE_FACTOR
    edge_mode: str = DEFAULT_EDGE_MODE
    edge_width_hz: float = DEFAULT_EDGE_WIDTH_HZ
    edge_fraction: float = DEFAULT_EDGE_FRACTION


@dataclass
class FitConfig:
    """Parameters for :func:`ssa.fitting.fit_template_region`."""

    n_peaks: int = DEFAULT_N_PEAKS
    init_mus: list[float] | None = None
    mu_bounds: list[tuple[float, float]] | None = None
    background_order: int = DEFAULT_BACKGROUND_ORDER
    allow_scale: bool = DEFAULT_ALLOW_SCALE
    common_scale: bool = DEFAULT_COMMON_SCALE
    loss: str = DEFAULT_LOSS
    min_separation: float = DEFAULT_MIN_SEPARATION_HZ
    separation_penalty_strength: float = DEFAULT_SEPARATION_PENALTY_STRENGTH


@dataclass
class UncertaintyConfig:
    """Parameters for the bootstrap / template-propagation uncertainty step.

    Note ``n_boot`` here is the GUI-facing default (1000), distinct from
    ``bootstrap_fit``'s own Python-level default (300); both are defined
    separately in ``ssa.constants``.
    """

    block_size: str | int = DEFAULT_BLOCK_SIZE
    random_seed: int = DEFAULT_RANDOM_SEED
    run_bootstrap: bool = True
    n_boot: int = DEFAULT_FIT_BOOTSTRAP_N_BOOT_GUI
    run_template_propagation: bool = True
    n_template_prop: int = DEFAULT_N_TEMPLATE_PROPAGATION_DRAWS
