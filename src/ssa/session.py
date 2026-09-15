"""Save/load a reproducible SchottkySA session (templates + fits + params)
as TOML, and re-verify it by recomputing everything from scratch.

No Qt dependency -- usable from the GUI, the CLI, or a plain script, the
same way ssa.pipeline/ssa.io are.
"""

from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib
import tomli_w

from ssa.config import FitConfig, TemplateBuildConfig, UncertaintyConfig

__all__ = [
    "SCHEMA_VERSION",
    "InputRef",
    "FormState",
    "TemplateRecord",
    "FitRecord",
    "Session",
    "save_session",
    "load_session",
]

SCHEMA_VERSION = 1


@dataclass
class InputRef:
    path: str
    sha256: str
    frequency_key: str
    amplitude_key: str


@dataclass
class FormState:
    template: TemplateBuildConfig
    fit: FitConfig
    fit_label: str
    auto_store: bool
    uncertainty: UncertaintyConfig
    region_hz: tuple[float, float]


@dataclass
class TemplateRecord:
    id: int
    region_hz: tuple[float, float]
    config: TemplateBuildConfig
    verify: dict[str, float]


@dataclass
class FitRecord:
    label: str
    notes: str
    template_id: int
    region_hz: tuple[float, float]
    fit_config: FitConfig
    uncertainty_config: UncertaintyConfig
    verify_rows: list[dict[str, Any]] = field(default_factory=list)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FitRecord):
            return NotImplemented
        if (
            self.label != other.label
            or self.notes != other.notes
            or self.template_id != other.template_id
            or self.region_hz != other.region_hz
            or self.fit_config != other.fit_config
            or self.uncertainty_config != other.uncertainty_config
        ):
            return False
        if len(self.verify_rows) != len(other.verify_rows):
            return False
        for row_a, row_b in zip(self.verify_rows, other.verify_rows, strict=True):
            if set(row_a.keys()) != set(row_b.keys()):
                return False
            for key in row_a:
                val_a = row_a[key]
                val_b = row_b[key]
                if isinstance(val_a, float) and isinstance(val_b, float) and math.isnan(val_a) and math.isnan(val_b):
                    continue
                if val_a != val_b:
                    return False
        return True


@dataclass
class Session:
    schema_version: int
    ssa_version: str
    created_utc: str
    input: InputRef
    form_state: FormState
    templates: list[TemplateRecord] = field(default_factory=list)
    fits: list[FitRecord] = field(default_factory=list)


def _sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_toml_value(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_toml_value(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple)):
        return [_to_toml_value(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_session(path: str | Path, session: Session) -> None:
    payload = _to_toml_value(asdict(session))
    with open(path, "wb") as f:
        tomli_w.dump(payload, f)


def _template_config_from_dict(d: dict[str, Any]) -> TemplateBuildConfig:
    return TemplateBuildConfig(
        name=d["name"],
        n_template_boot=d["n_template_boot"],
        block_size=d["block_size"],
        random_seed=d["random_seed"],
        clip_negative=d["clip_negative"],
        smooth=d["smooth"],
        sg_window=d.get("sg_window"),
        sg_poly=d["sg_poly"],
        resample_factor=d["resample_factor"],
        edge_mode=d["edge_mode"],
        edge_width_hz=d["edge_width_hz"],
        edge_fraction=d["edge_fraction"],
    )


def _fit_config_from_dict(d: dict[str, Any]) -> FitConfig:
    mu_bounds = d.get("mu_bounds")
    return FitConfig(
        n_peaks=d["n_peaks"],
        init_mus=d.get("init_mus"),
        mu_bounds=[tuple(b) for b in mu_bounds] if mu_bounds else None,
        background_order=d["background_order"],
        allow_scale=d["allow_scale"],
        common_scale=d["common_scale"],
        loss=d["loss"],
        min_separation=d["min_separation"],
        separation_penalty_strength=d["separation_penalty_strength"],
    )


def _uncertainty_config_from_dict(d: dict[str, Any]) -> UncertaintyConfig:
    return UncertaintyConfig(
        block_size=d["block_size"],
        random_seed=d["random_seed"],
        run_bootstrap=d["run_bootstrap"],
        n_boot=d["n_boot"],
        run_template_propagation=d["run_template_propagation"],
        n_template_prop=d["n_template_prop"],
    )


def _form_state_from_dict(d: dict[str, Any]) -> FormState:
    return FormState(
        template=_template_config_from_dict(d["template"]),
        fit=_fit_config_from_dict(d["fit"]),
        fit_label=d["fit_label"],
        auto_store=d["auto_store"],
        uncertainty=_uncertainty_config_from_dict(d["uncertainty"]),
        region_hz=tuple(d["region_hz"]),
    )


def _template_record_from_dict(d: dict[str, Any]) -> TemplateRecord:
    return TemplateRecord(
        id=d["id"],
        region_hz=tuple(d["region_hz"]),
        config=_template_config_from_dict(d["config"]),
        verify=d["verify"],
    )


def _fit_record_from_dict(d: dict[str, Any]) -> FitRecord:
    return FitRecord(
        label=d["label"],
        notes=d["notes"],
        template_id=d["template_id"],
        region_hz=tuple(d["region_hz"]),
        fit_config=_fit_config_from_dict(d["fit_config"]),
        uncertainty_config=_uncertainty_config_from_dict(d["uncertainty_config"]),
        verify_rows=d.get("verify_rows", []),
    )


def load_session(path: str | Path) -> Session:
    with open(path, "rb") as f:
        d = tomllib.load(f)
    return Session(
        schema_version=d["schema_version"],
        ssa_version=d["ssa_version"],
        created_utc=d["created_utc"],
        input=InputRef(**d["input"]),
        form_state=_form_state_from_dict(d["form_state"]),
        templates=[_template_record_from_dict(t) for t in d.get("templates", [])],
        fits=[_fit_record_from_dict(f) for f in d.get("fits", [])],
    )
