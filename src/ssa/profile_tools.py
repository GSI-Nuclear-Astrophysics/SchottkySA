"""Harmonic-aware comparison and averaging of centered, unit-area profiles.

Reference-error propagation and between-source sensitivity use distinct banks.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

import numpy as np
from scipy.integrate import cumulative_trapezoid

from ssa.fitting import fit_template_region
from ssa.preprocessing import summarize_distribution, trapz
from ssa.templates import PeakTemplate, get_template_std


def rescale_template(template: PeakTemplate, factor: float) -> PeakTemplate:
    """Change frequency units, preserving PDF area and the known harmonic ratio."""
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Frequency scale factor must be positive and finite")
    if factor == 1.0:
        return template
    return replace(
        template,
        u=template.u * factor,
        pdf=template.pdf / factor,
        cdf=template.cdf.copy(),
        raw_area=template.raw_area * factor,
        raw_cog=template.raw_cog * factor,
        raw_std=template.raw_std * factor,
        dx=template.dx * factor,
        edge_info={**template.edge_info, "frequency_scale_factor": float(factor)},
    )


def transfer_template_entry(entry: dict[str, Any], target_harmonic: float) -> dict[str, Any]:
    source_harmonic = float(entry.get("harmonic", 1.0))
    if source_harmonic <= 0 or not np.isfinite(source_harmonic):
        raise ValueError("Source harmonic must be positive and finite")
    factor = target_harmonic / source_harmonic
    return {
        **entry,
        "template": rescale_template(entry["template"], factor),
        "template_bank": [rescale_template(t, factor) for t in entry["template_bank"]],
        "source_harmonic": source_harmonic,
        "target_harmonic": float(target_harmonic),
        "frequency_scale_factor": factor,
    }


def _density_template(
    u: np.ndarray, density: np.ndarray, *, name: str, anchor: float, metadata: dict[str, Any]
) -> PeakTemplate:
    pdf = np.clip(np.asarray(density, dtype=float), 0, None)
    area = trapz(pdf, u)
    if not np.isfinite(area) or area <= 0:
        raise ValueError("Combined profile has no positive finite area")
    pdf /= area
    centre = trapz(u * pdf, u)
    centred_u = u - centre
    std = float(np.sqrt(max(trapz(centred_u**2 * pdf, u), 0)))
    cdf = cumulative_trapezoid(pdf, centred_u, initial=0)
    cdf /= cdf[-1]
    return PeakTemplate(
        centred_u, pdf, cdf, 1.0, anchor + centre, std, float(np.median(np.diff(centred_u))), name, metadata
    )


def combine_template_entries(
    entries: list[dict[str, Any]],
    *,
    name: str = "combined_profile",
    output_harmonic: float = 1.0,
    weights: list[float] | None = None,
    n_boot: int = 300,
    random_seed: int = 12345,
    bank_mode: str = "source_sensitivity",
    workers: int = 2,
) -> dict[str, Any]:
    """Combine centred PDFs on a union grid with no additional smoothing.

    ``source_sensitivity`` refits with individual source-bank profiles to retain
    between-source differences. It is a sensitivity ensemble, not a confidence
    interval for the mean. ``independent`` instead builds weighted mean draws,
    assuming independent reference errors, which is not established merely by
    observing different harmonics from the same acquisition.
    """
    if len(entries) < 2:
        raise ValueError("Select at least two source templates")
    if bank_mode not in ("source_sensitivity", "independent"):
        raise ValueError("Unknown combined-template uncertainty mode")
    if n_boot < 0 or workers < 1:
        raise ValueError("Draw count must be nonnegative and workers must be positive")
    leaves: set[int] = set()
    for entry in entries:
        source_ids = set(entry.get("source_leaf_ids", []))
        if leaves.intersection(source_ids):
            raise ValueError("Selected profiles reuse the same underlying source")
        leaves.update(source_ids)
    converted = [transfer_template_entry(e, output_harmonic) for e in entries]
    w = np.ones(len(entries)) if weights is None else np.array(weights, dtype=float, copy=True)
    if w.shape != (len(entries),) or not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("Provide one positive finite weight per source template")
    w /= w.sum()
    sources = [e["template"] for e in converted]
    step = min(t.dx for t in sources)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Source template grid step must be positive and finite")
    banks = [e["template_bank"] for e in converted]
    supports = sources + ([t for bank in banks for t in bank] if n_boot > 0 and bank_mode == "independent" else [])
    lo, hi = min(t.u[0] for t in supports), max(t.u[-1] for t in supports)
    if not np.isfinite(lo + hi) or hi <= lo:
        raise ValueError("Source template support must be finite and nonempty")
    n_grid = int(np.ceil((hi - lo) / step)) + 1
    if n_grid > 500_000:
        raise ValueError("Combined grid is too large or invalid; inspect source widths and harmonics")
    grid = np.linspace(lo, hi, n_grid)
    anchor = sources[0].raw_cog
    metadata = {
        "edge_mode": "combined_sources",
        "combination_bank_mode": bank_mode,
        "source_names": [t.name for t in sources],
        "weights": w.tolist(),
        "output_harmonic": output_harmonic,
        "additional_smoothing": False,
    }

    def mixture(templates: list[PeakTemplate], suffix: str) -> PeakTemplate:
        density = np.zeros_like(grid)
        for weight, template in zip(w, templates, strict=True):
            density = density + weight * template.evaluate_pdf(grid)
        return _density_template(grid, density, name=name + suffix, anchor=anchor, metadata=metadata)

    nominal = mixture(sources, "")
    if n_boot > 0 and any(not bank for bank in banks):
        raise ValueError("Every source needs bootstrap templates, or set combined draws to zero")
    rng = np.random.default_rng(random_seed)
    plans = []
    for _ in range(n_boot):
        if bank_mode == "independent":
            plans.append([bank[int(rng.integers(len(bank)))] for bank in banks])
        else:
            j = int(rng.choice(len(banks), p=w))
            plans.append([banks[j][int(rng.integers(len(banks[j])))]])

    def draw(item: tuple[int, list[PeakTemplate]]) -> PeakTemplate:
        index, templates = item
        if bank_mode == "independent":
            return mixture(templates, f"_boot_{index}")
        source = templates[0]
        return replace(source, name=f"{name}_sensitivity_{index}", edge_info=metadata, raw_cog=anchor)

    if workers > 1 and len(plans) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            templates = list(pool.map(draw, enumerate(plans)))
    else:
        templates = [draw(item) for item in enumerate(plans)]
    summary = {
        "n_templates": len(templates),
        "raw_cog": summarize_distribution(np.asarray([t.raw_cog for t in templates])),
        "raw_std": summarize_distribution(np.asarray([get_template_std(t) for t in templates])),
        "raw_area": summarize_distribution(np.asarray([t.raw_area for t in templates])),
    }
    return {
        "template_nominal": nominal,
        "templates": templates,
        "summary": summary,
        "settings": metadata,
        "residual_model": {},
    }


def compare_template_entries(
    frequency: np.ndarray,
    amplitude: np.ndarray,
    entries: list[dict[str, Any]],
    source_id: int,
    target_id: int,
    *,
    allow_scale: bool = False,
    background_order: int = 0,
    workers: int = 2,
) -> list[dict[str, Any]]:
    """Fit each selected profile to the other's measured ROI, using one component."""
    if source_id == target_id:
        raise ValueError("Choose two different source templates")
    chosen = [entries[source_id], entries[target_id]]
    if any(e.get("combination") is not None for e in chosen):
        raise ValueError("Bidirectional comparison needs two measured source regions, not a composite ROI")

    def direction(pair: tuple[int, int]) -> dict[str, Any]:
        source, target = pair
        target_entry = entries[target]
        h = float(target_entry.get("harmonic", 1.0))
        converted = transfer_template_entry(entries[source], h)
        lo, hi = target_entry["region_lo"], target_entry["region_hi"]
        mask = (frequency >= lo) & (frequency <= hi)
        x, y = frequency[mask], amplitude[mask]
        if len(x) < 7:
            raise ValueError("Comparison target region has fewer than seven points")
        fit = fit_template_region(
            x,
            y,
            converted["template"],
            n_peaks=1,
            init_mus=[float(target_entry["template"].raw_cog)],
            allow_scale=allow_scale,
            background_order=background_order,
            loss="linear",
        )
        residual = fit["y"] - fit["y_model"]
        denom = float(np.sum(residual**2))
        dw = float(np.sum(np.diff(residual) ** 2) / denom) if denom > 0 else float("nan")
        row = {
            "source_id": source,
            "target_id": target,
            "source_name": entries[source]["name"],
            "target_name": target_entry["name"],
            "source_harmonic": float(entries[source].get("harmonic", 1)),
            "target_harmonic": h,
            "harmonic_scale_factor": float(converted["frequency_scale_factor"]),
            "extra_width_scale_allowed": allow_scale,
            "extra_width_scale_fit": float(fit["info"]["scales"][0]),
            "mu_fit_Hz": float(fit["info"]["mus"][0]),
            "mu_minus_target_cog_Hz": float(fit["info"]["mus"][0] - target_entry["template"].raw_cog),
            "chi2": float(fit["chi2"]),
            "dof": int(fit["dof"]),
            "red_chi2": float(fit["red_chi2"]),
            "DW": dw,
            "success": bool(fit["success"]),
        }
        return {"fit": fit, "row": row}

    pairs = [(source_id, target_id), (target_id, source_id)]
    if workers > 1:
        with ThreadPoolExecutor(max_workers=min(workers, 2)) as pool:
            return list(pool.map(direction, pairs))
    return [direction(pair) for pair in pairs]
