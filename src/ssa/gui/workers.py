"""Background worker thread that runs the scientific core off the GUI thread."""

from __future__ import annotations

import traceback
from typing import Any

import numpy as np
from pyqtgraph.Qt import QtCore

from ssa.fitting import fit_template_region, mu_bounds_to_string
from ssa.templates import bootstrap_template_bank
from ssa.uncertainty import bootstrap_fit, propagate_template_uncertainty_to_target_fit

__all__ = ["AnalysisWorker"]


class AnalysisWorker(QtCore.QThread):
    progress = QtCore.Signal(str)
    finished_ok = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, mode: str, payload: dict[str, Any]):
        super().__init__()
        self.mode = mode
        self.payload = payload

    def run(self) -> None:
        try:
            if self.mode == "build_template":
                self._run_build_template()
            elif self.mode == "fit_region":
                self._run_fit_region()
            else:
                raise ValueError(f"Unknown worker mode: {self.mode}")
        except Exception:
            self.failed.emit(traceback.format_exc())

    def _run_build_template(self) -> None:
        p = self.payload
        self.progress.emit("Building nominal template and template bootstrap bank...")
        result = bootstrap_template_bank(
            p["x"],
            p["y"],
            n_boot=p["n_template_boot"],
            block_size=p["block_size"],
            random_seed=p["random_seed"],
            name=p["name"],
            clip_negative=p["clip_negative"],
            smooth=p["smooth"],
            sg_window=p["sg_window"],
            sg_poly=p["sg_poly"],
            resample_factor=p["resample_factor"],
            edge_mode=p.get("edge_mode", "none"),
            edge_width_hz=p.get("edge_width_hz", 0.0),
            edge_fraction=p.get("edge_fraction", 0.08),
        )
        result["settings"] = {
            "n_template_boot_requested": p["n_template_boot"],
            "block_size_requested": p["block_size"],
            "random_seed": p["random_seed"],
            "clip_negative": p["clip_negative"],
            "smooth_nominal_template": p["smooth"],
            "sg_window": p["sg_window"],
            "sg_poly": p["sg_poly"],
            "resample_factor": p["resample_factor"],
            "edge_mode": p.get("edge_mode", "none"),
            "edge_width_hz": p.get("edge_width_hz", 0.0),
            "edge_fraction": p.get("edge_fraction", 0.08),
        }
        self.finished_ok.emit(result)

    def _run_fit_region(self) -> None:
        p = self.payload
        rng = np.random.default_rng(p["random_seed"])

        self.progress.emit("Running least-squares template fit...")
        fit = fit_template_region(
            p["x"],
            p["y"],
            template=p["template"],
            n_peaks=p["n_peaks"],
            init_mus=p.get("init_mus"),
            background_order=p["background_order"],
            allow_scale=p["allow_scale"],
            common_scale=p["common_scale"],
            loss=p["loss"],
            min_separation=p["min_separation"],
            mu_bounds=p.get("mu_bounds"),
        )

        boot = None
        if p["run_bootstrap"] and p["n_boot"] > 0:
            self.progress.emit("Running residual block bootstrap...")
            boot = bootstrap_fit(fit, n_boot=p["n_boot"], block_size=p["block_size"], rng=rng)

        tpl_prop = None
        if p["run_template_propagation"] and p.get("template_bank"):
            self.progress.emit("Propagating template-shape uncertainty...")
            tpl_prop = propagate_template_uncertainty_to_target_fit(
                fit,
                p["template_bank"],
                n_draws=p["n_template_prop"],
                random_seed=p["random_seed"],
            )

        result = {
            "fit": fit,
            "bootstrap": boot,
            "template_propagation": tpl_prop,
            "settings": {
                "n_boot_requested": p["n_boot"],
                "n_template_prop_requested": p["n_template_prop"],
                "block_size_requested": p["block_size"],
                "min_separation_Hz": p["min_separation"],
                "mu_bounds_Hz": mu_bounds_to_string(p.get("mu_bounds")),
                "random_seed": p["random_seed"],
                "run_bootstrap": p["run_bootstrap"],
                "run_template_propagation": p["run_template_propagation"],
                "loss": p["loss"],
            },
        }
        self.finished_ok.emit(result)
