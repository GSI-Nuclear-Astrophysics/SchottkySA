"""SchottkySA (``ssa``): empirical-template fitting of Schottky spectra.

The scientific core (this top-level package) has no GUI dependency and can
be installed/imported/tested without Qt. The GUI (``ssa.gui``) requires the
optional ``gui`` extra (``pip install SchottkySA[gui]``).
"""

from __future__ import annotations

from ssa.fitting import fit_template_region, template_model
from ssa.io import load_spectrum_npz, make_export_rows_for_result
from ssa.preprocessing import prepare_xy
from ssa.templates import PeakTemplate, bootstrap_template_bank, build_peak_template
from ssa.uncertainty import (
    bootstrap_fit,
    combined_sigma,
    component_quantiles_from_fit,
    propagate_template_uncertainty_to_target_fit,
    quality_flag_for_fit,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "PeakTemplate",
    "prepare_xy",
    "build_peak_template",
    "bootstrap_template_bank",
    "template_model",
    "fit_template_region",
    "bootstrap_fit",
    "propagate_template_uncertainty_to_target_fit",
    "combined_sigma",
    "component_quantiles_from_fit",
    "quality_flag_for_fit",
    "load_spectrum_npz",
    "make_export_rows_for_result",
]
