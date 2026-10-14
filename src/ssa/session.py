"""Store template recipes and fit checkpoints in TOML and replay them without Qt."""

from __future__ import annotations

import hashlib
import math
import sys
from collections.abc import Callable
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
from ssa.io import make_export_rows_for_result, select_frequency_range
from ssa.pipeline import build_template_bank, run_fit_with_uncertainty
from ssa.preprocessing import prepare_xy
from ssa.profile_tools import combine_template_entries, compare_template_entries, transfer_template_entry

__all__ = [
    "SCHEMA_VERSION",
    "InputRef",
    "FormState",
    "TemplateRecord",
    "ComparisonRecord",
    "FitRecord",
    "Session",
    "save_session",
    "load_session",
    "ReplayReport",
    "replay_session",
]

SCHEMA_VERSION = 3


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
    active_template_id: int | None = None
    view_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class TemplateRecord:
    id: int
    region_hz: tuple[float, float]
    config: TemplateBuildConfig
    verify: dict[str, float]
    harmonic: float = 1.0
    combination: dict[str, Any] | None = None
    harmonic_expression: str = ""


@dataclass
class ComparisonRecord:
    source_id: int
    target_id: int
    allow_scale: bool = False
    background_order: int = 0
    workers: int = 2
    verify_rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class FitRecord:
    label: str
    notes: str
    template_id: int
    region_hz: tuple[float, float]
    fit_config: FitConfig
    uncertainty_config: UncertaintyConfig
    verify_rows: list[dict[str, Any]] = field(default_factory=list)
    target_harmonic: float = 1.0
    view_index: int = 0
    component_template_ids: list[int] | None = None
    target_harmonic_expression: str = ""

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
            or self.target_harmonic != other.target_harmonic
            or self.view_index != other.view_index
            or self.target_harmonic_expression != other.target_harmonic_expression
            or self.component_template_ids != other.component_template_ids
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
    comparisons: list[ComparisonRecord] = field(default_factory=list)


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
        left_edge_points=d.get("left_edge_points", 0),
        right_edge_points=d.get("right_edge_points", 0),
        left_tail_width_hz=d.get("left_tail_width_hz", 0),
        right_tail_width_hz=d.get("right_tail_width_hz", 0),
    )


def _fit_config_from_dict(d: dict[str, Any]) -> FitConfig:
    mu_bounds = d.get("mu_bounds")
    return FitConfig(
        n_peaks=d["n_peaks"],
        init_mus=d.get("init_mus"),
        mu_bounds=[tuple(b) for b in mu_bounds] if mu_bounds is not None else None,
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
        active_template_id=d.get("active_template_id"),
        view_state=d.get("view_state", {}),
    )


def _template_record_from_dict(d: dict[str, Any]) -> TemplateRecord:
    return TemplateRecord(
        id=d["id"],
        region_hz=tuple(d["region_hz"]),
        config=_template_config_from_dict(d["config"]),
        verify=d["verify"],
        harmonic=float(d.get("harmonic", 1.0)),
        harmonic_expression=d.get("harmonic_expression", ""),
        combination=d.get("combination"),
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
        target_harmonic=float(d.get("target_harmonic", 1.0)),
        view_index=int(d.get("view_index", 0)),
        component_template_ids=d.get("component_template_ids"),
        target_harmonic_expression=d.get("target_harmonic_expression", ""),
    )


def load_session(path: str | Path) -> Session:
    with open(path, "rb") as f:
        d = tomllib.load(f)
    if d["schema_version"] not in (1, 2, SCHEMA_VERSION):
        raise ValueError(
            f"Unsupported session schema_version {d['schema_version']}; "
            f"this version of SchottkySA supports schema_versions 1 and {SCHEMA_VERSION}."
        )
    return Session(
        schema_version=d["schema_version"],
        ssa_version=d["ssa_version"],
        created_utc=d["created_utc"],
        input=InputRef(**d["input"]),
        form_state=_form_state_from_dict(d["form_state"]),
        templates=[_template_record_from_dict(t) for t in d.get("templates", [])],
        fits=[_fit_record_from_dict(f) for f in d.get("fits", [])],
        comparisons=[ComparisonRecord(**c) for c in d.get("comparisons", [])],
    )


RTOL = 1e-3
ATOL = 1e-6


@dataclass
class ReplayReport:
    input_warnings: list[str]
    templates: list[dict[str, Any]]
    fits: list[dict[str, Any]]
    mismatches: list[str]
    frequency: np.ndarray
    amplitude: np.ndarray
    data_path: Path
    comparisons: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches and not self.input_warnings


def _values_close(a: float, b: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)


def _diff_scalars(label: str, expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    out = []
    for key, exp_val in expected.items():
        if key not in actual:
            out.append(f"{label}: missing field '{key}' on replay")
            continue
        act_val = actual[key]
        if isinstance(exp_val, bool) or isinstance(act_val, bool):
            if exp_val != act_val:
                out.append(f"{label}: '{key}' expected {exp_val!r}, got {act_val!r}")
        elif isinstance(exp_val, (int, float)) and isinstance(act_val, (int, float)):
            # Absolute frequency offsets must not be tolerated relative to the
            # RF carrier (e.g. a fraction of hundreds of MHz).
            is_position = key in {"mu_fit_Hz", "q16_Hz", "q50_Hz", "q84_Hz", "raw_cog"}
            close = (
                math.isclose(float(exp_val), float(act_val), rel_tol=0, abs_tol=1e-5)
                if is_position
                else _values_close(float(exp_val), float(act_val))
            )
            if not close:
                out.append(f"{label}: '{key}' expected {exp_val!r}, got {act_val!r}")
        elif exp_val != act_val:
            out.append(f"{label}: '{key}' expected {exp_val!r}, got {act_val!r}")
    return out


def _diff_rows(label: str, expected_rows: list[dict[str, Any]], actual_rows: list[dict[str, Any]]) -> list[str]:
    if len(expected_rows) != len(actual_rows):
        return [f"{label}: expected {len(expected_rows)} component rows, got {len(actual_rows)}"]
    out = []
    for k, (exp_row, act_row) in enumerate(zip(expected_rows, actual_rows, strict=True)):
        out.extend(_diff_scalars(f"{label} component {k}", exp_row, act_row))
    return out


def _template_verify_values(bank: dict[str, Any]) -> dict[str, float]:
    tpl = bank["template_nominal"]
    q16, q50, q84 = tpl.quantile([0.16, 0.50, 0.84])
    return {
        "raw_area": float(tpl.raw_area),
        "raw_cog": float(tpl.raw_cog),
        "raw_std": float(tpl.raw_std),
        "q16": float(q16),
        "q50": float(q50),
        "q84": float(q84),
        "n_template_bank": len(bank["templates"]),
    }


def _load_input_xy(path: str | Path, frequency_key: str, amplitude_key: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    f = np.asarray(data[frequency_key], dtype=float).ravel()
    y = np.asarray(data[amplitude_key], dtype=float).ravel()
    return prepare_xy(f, y)


def replay_session(
    session: Session,
    *,
    data_path_override: str | Path | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> ReplayReport:
    def progress(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    warnings: list[str] = []
    mismatches: list[str] = []

    data_path = Path(data_path_override) if data_path_override is not None else Path(session.input.path)
    progress(f"Loading {data_path}...")
    actual_sha = _sha256_of(data_path)
    if actual_sha != session.input.sha256:
        warnings.append(
            f"Input file hash mismatch: recorded {session.input.sha256}, found {actual_sha} at {data_path}."
        )
    f, y = _load_input_xy(data_path, session.input.frequency_key, session.input.amplitude_key)

    templates: list[dict[str, Any]] = []
    bank_by_id: dict[int, dict[str, Any]] = {}
    for record in session.templates:
        progress(f"Rebuilding template {record.id}...")
        if record.id in bank_by_id:
            raise ValueError(f"Duplicate template id {record.id}")
        if record.combination is None:
            x_r, y_r, _ = select_frequency_range(f, y, None, record.region_hz[0], record.region_hz[1])
            bank = build_template_bank(x_r, y_r, record.config)
        else:
            recipe = record.combination
            parent_ids = recipe["source_ids"]
            if len(set(parent_ids)) != len(parent_ids) or any(i not in bank_by_id for i in parent_ids):
                raise ValueError("Combined template requires distinct, earlier source template ids")
            bank = combine_template_entries(
                [bank_by_id[i] for i in parent_ids],
                name=record.config.name,
                output_harmonic=record.harmonic,
                weights=recipe["weights"],
                n_boot=record.config.n_template_boot,
                random_seed=record.config.random_seed,
                bank_mode=recipe["bank_mode"],
                workers=recipe.get("workers", 2),
            )
        tpl = bank["template_nominal"]
        mismatches.extend(_diff_scalars(f"template {record.id}", record.verify, _template_verify_values(bank)))
        entry = {
            "name": tpl.name,
            "harmonic": record.harmonic,
            "harmonic_expression": record.harmonic_expression,
            "combination": record.combination,
            "source_leaf_ids": [record.id]
            if record.combination is None
            else sorted(
                {leaf for source in record.combination["source_ids"] for leaf in bank_by_id[source]["source_leaf_ids"]}
            ),
            "region_lo": record.region_hz[0],
            "region_hi": record.region_hz[1],
            "template": tpl,
            "template_bank": bank["templates"],
            "config": record.config,
            "summary": bank.get("summary", {}),
            "settings": bank["settings"]
            if record.combination is not None
            else {
                "smooth_nominal_template": record.config.smooth,
                "resample_factor": record.config.resample_factor,
                "left_edge_points": record.config.left_edge_points,
                "right_edge_points": record.config.right_edge_points,
                "left_tail_width_hz": record.config.left_tail_width_hz,
                "right_tail_width_hz": record.config.right_tail_width_hz,
            },
            "edge_info": getattr(tpl, "edge_info", {}),
            "residual_model": bank.get("residual_model", {}),
            "full": bank,
        }
        templates.append(entry)
        bank_by_id[record.id] = entry

    fits: list[dict[str, Any]] = []
    for record in session.fits:
        progress(f"Refitting '{record.label}'...")
        tpl_entry = bank_by_id.get(record.template_id)
        if tpl_entry is None:
            raise ValueError(
                f"Fit '{record.label}' references template_id {record.template_id}, "
                "which is not among this session's templates."
            )
        x_r, y_r, _ = select_frequency_range(f, y, None, record.region_hz[0], record.region_hz[1])
        component_ids = record.component_template_ids
        entries = None
        if component_ids is not None:
            if len(component_ids) != record.fit_config.n_peaks or any(idx not in bank_by_id for idx in component_ids):
                raise ValueError(f"Fit '{record.label}' has invalid per-component template ids.")
            cache = {
                idx: transfer_template_entry(bank_by_id[idx], record.target_harmonic) for idx in set(component_ids)
            }
            entries = [cache[idx] for idx in component_ids]
            transferred = entries[0]
        else:
            transferred = transfer_template_entry(tpl_entry, record.target_harmonic)
        result = run_fit_with_uncertainty(
            x_r,
            y_r,
            transferred["template"],
            transferred["template_bank"],
            record.fit_config,
            record.uncertainty_config,
            component_templates=[e["template"] for e in entries] if entries is not None else None,
            component_template_banks=[e["template_bank"] for e in entries] if entries is not None else None,
        )
        result["target_harmonic_expression"] = record.target_harmonic_expression
        result["component_template_ids"] = component_ids
        result["component_template_entries"] = entries
        rows = make_export_rows_for_result(
            result, label=record.label, template_registry=[*(entries or [transferred]), *templates], notes=record.notes
        )
        mismatches.extend(_diff_rows(record.label, record.verify_rows, rows))
        fits.append(
            {
                "label": record.label,
                "notes": record.notes,
                "result": result,
                "rows": rows,
                "fit_config": record.fit_config,
                "uncertainty_config": record.uncertainty_config,
                "template_index": record.template_id,
                "region_hz": record.region_hz,
                "target_harmonic": record.target_harmonic,
                "view_index": record.view_index,
            }
        )

    comparisons = []
    for record in session.comparisons:
        if not (0 <= record.source_id < len(templates) and 0 <= record.target_id < len(templates)):
            raise ValueError("Comparison references an unknown source template")
        directions = compare_template_entries(
            f,
            y,
            templates,
            record.source_id,
            record.target_id,
            allow_scale=record.allow_scale,
            background_order=record.background_order,
            workers=record.workers,
        )
        mismatches.extend(_diff_rows("shape comparison", record.verify_rows, [d["row"] for d in directions]))
        comparisons.append({"record": record, "directions": directions})

    return ReplayReport(
        input_warnings=warnings,
        templates=templates,
        fits=fits,
        mismatches=mismatches,
        frequency=f,
        amplitude=y,
        data_path=data_path,
        comparisons=comparisons,
    )
