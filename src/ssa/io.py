"""Input loading and output export -- no Qt/GUI dependency.

Covers NPZ key selection, the fit-history export row schema, delimited/
JSON/XLSX/ODS writers, TSV clipboard formatting, and current-fit NPZ overlay
export. See ``docs/OPEN_SCIENTIFIC_QUESTIONS.md`` item 6 for the rationale
behind the current export column set.

The interactive "ask the user which array is frequency/amplitude" dialog
stays in the GUI layer (``ssa.gui.main_window``); this module raises
:class:`~ssa.exceptions.InputFormatError` instead when key selection is
ambiguous, so it is usable head-less (Python API, tests, CLI).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ssa.constants import PREFERRED_AMPLITUDE_KEYS, PREFERRED_FREQUENCY_KEYS
from ssa.exceptions import InputFormatError
from ssa.preprocessing import prepare_xy
from ssa.uncertainty import (
    combined_sigma,
    component_quantiles_from_fit,
    component_widths_from_params,
    quality_flag_for_fit,
    sample_count,
    summarize_area_corr_samples,
    summarize_area_ratio_samples,
    summarize_mu_corr_samples,
    summarize_mu_delta_samples,
)

__all__ = [
    "select_npz_keys",
    "load_spectrum_npz",
    "load_table",
    "select_frequency_range",
    "make_export_rows_for_result",
    "rows_to_tsv",
    "write_delimited",
    "export_history_rows",
    "export_current_overlay_npz",
    "write_curve_csv",
]


def select_npz_keys(available_keys: list[str], one_d_keys: list[str] | None = None) -> tuple[str | None, str | None]:
    """Pick (frequency_key, amplitude_key) from an NPZ file's array names.

    Preference order: named keys first (see
    ``PREFERRED_FREQUENCY_KEYS``/``PREFERRED_AMPLITUDE_KEYS``), then "if
    there are exactly two 1-D arrays, use those two", otherwise ``(None,
    None)`` -- ambiguous, caller must resolve (interactively in the GUI, or
    by raising in :func:`load_spectrum_npz`).
    """
    f_key = next((k for k in PREFERRED_FREQUENCY_KEYS if k in available_keys), None)
    y_key = next((k for k in PREFERRED_AMPLITUDE_KEYS if k in available_keys), None)
    if f_key is not None and y_key is not None:
        return f_key, y_key

    candidates = one_d_keys if one_d_keys is not None else available_keys
    if len(candidates) == 2:
        return candidates[0], candidates[1]
    return None, None


def load_spectrum_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Load a two-array NPZ spectrum file, preprocessed via :func:`prepare_xy`.

    Returns ``(frequency, amplitude, frequency_key, amplitude_key)``.

    Raises
    ------
    InputFormatError
        If the file has no arrays, or the frequency/amplitude keys cannot be
        determined unambiguously (see :func:`select_npz_keys`).
    """
    path = Path(path)
    data = np.load(path)
    keys = list(data.keys())
    if not keys:
        raise InputFormatError(f"NPZ file contains no arrays: {path}")

    one_d = [k for k in keys if np.asarray(data[k]).ndim == 1]
    f_key, y_key = select_npz_keys(keys, one_d)
    if f_key is None or y_key is None:
        raise InputFormatError(
            f"Could not determine frequency/amplitude arrays in {path} "
            f"(available keys: {keys}). Use the GUI's interactive key picker, "
            "or pass explicit keys."
        )

    f = np.asarray(data[f_key], dtype=float).ravel()
    y = np.asarray(data[y_key], dtype=float).ravel()
    f, y = prepare_xy(f, y)
    return f, y, f_key, y_key


def load_table(
    path: str | Path,
    freq_col: str = "frequency_hz",
    power_col: str = "power",
    sigma_col: str | None = "sigma",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load a named-column spectrum table (CSV, TSV, or whitespace-delimited).

    This is the CLI's (``ssa run``) input format -- see
    ``docs/SCIENTIFIC_METHOD.md`` and the project README. No pandas
    dependency: uses the standard-library ``csv`` module (comma/tab) or a
    plain ``str.split()`` fallback for whitespace-delimited files, so the
    core CLI stays installable without the ``xlsx`` extra.

    Delimiter is chosen from the file suffix: ``.csv`` -> comma, ``.tsv`` ->
    tab, anything else -> whitespace. The first non-blank, non-``#``-comment
    line is the header and must contain ``freq_col`` and ``power_col``;
    ``sigma_col`` is read if present, otherwise the third return value is
    ``None`` (per-point sigma unknown -- see
    ``ssa.preprocessing.robust_sigma_from_second_difference``).

    Returns
    -------
    ``(frequency, power, sigma)`` -- ``sigma`` is ``None`` if the column was
    not requested or not found. Rows are **not** run through
    :func:`~ssa.preprocessing.prepare_xy` here (sigma must stay aligned with
    frequency/power); callers combine and preprocess all three together.

    Raises
    ------
    InputFormatError
        If the file is empty, or ``freq_col``/``power_col`` are missing.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    delimiter = "," if suffix == ".csv" else "\t" if suffix == ".tsv" else None

    with open(path, newline="") as f:
        if delimiter is None:
            raw_rows = [line.split() for line in f if line.strip() and not line.lstrip().startswith("#")]
        else:
            raw_rows = [
                row for row in csv.reader(f, delimiter=delimiter) if row and not row[0].lstrip().startswith("#")
            ]

    if not raw_rows:
        raise InputFormatError(f"Table is empty: {path}")

    header = [h.strip() for h in raw_rows[0]]
    data_rows = raw_rows[1:]
    if freq_col not in header or power_col not in header:
        raise InputFormatError(
            f"Table {path} must contain columns '{freq_col}' and '{power_col}'; found columns {header}."
        )
    if not data_rows:
        raise InputFormatError(f"Table {path} has a header but no data rows.")

    fi = header.index(freq_col)
    pi = header.index(power_col)
    si = header.index(sigma_col) if sigma_col and sigma_col in header else None

    try:
        f_vals = np.array([float(r[fi]) for r in data_rows], dtype=float)
        y_vals = np.array([float(r[pi]) for r in data_rows], dtype=float)
        sigma_vals = np.array([float(r[si]) for r in data_rows], dtype=float) if si is not None else None
    except (ValueError, IndexError) as exc:
        raise InputFormatError(f"Could not parse numeric data in table {path}: {exc}") from exc

    return f_vals, y_vals, sigma_vals


def select_frequency_range(
    f: np.ndarray, y: np.ndarray, sigma: np.ndarray | None, lo: float | None, hi: float | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Restrict ``(f, y, sigma)`` to ``lo <= f <= hi`` (either bound optional).

    Mirrors the GUI's region-selection semantics
    (``MainWindow.selected_region_xy``): inclusive bounds, minimum 7 points.
    """
    if lo is None and hi is None:
        return f, y, sigma
    lo_val = -np.inf if lo is None else float(lo)
    hi_val = np.inf if hi is None else float(hi)
    if hi_val < lo_val:
        lo_val, hi_val = hi_val, lo_val
    mask = (f >= lo_val) & (f <= hi_val)
    if np.count_nonzero(mask) < 7:
        raise InputFormatError(f"Selected range [{lo_val:.9g}, {hi_val:.9g}] Hz has fewer than 7 points.")
    return f[mask], y[mask], (sigma[mask] if sigma is not None else None)


def make_export_rows_for_result(
    result: dict[str, Any],
    label: str,
    template_registry: list[dict[str, Any]] | None = None,
    notes: str = "",
) -> list[dict[str, Any]]:
    """Build one export row per fitted component (fit-history export schema).

    ``template_registry`` is a list of ``{"template": PeakTemplate,
    "region_lo": ..., "region_hi": ..., "settings": ..., "template_bank":
    [...]}`` entries, used to look up template-region provenance for the
    fit's active template. Pass ``None``/``[]`` if that provenance is
    unavailable (all `template_*`/`tpl_*` fields are `NaN`).
    """
    fit = result["fit"]
    boot = result.get("bootstrap")
    tpl_prop = result.get("template_propagation")
    tpl = fit["template"]
    template_name = getattr(tpl, "name", "template")
    fit_lo, fit_hi = float(np.nanmin(fit["x"])), float(np.nanmax(fit["x"]))

    tpl_entry: dict[str, Any] = {}
    for entry in template_registry or []:
        if entry.get("template") is tpl:
            tpl_entry = entry
            break
    tpl_lo = tpl_entry.get("region_lo", np.nan)
    tpl_hi = tpl_entry.get("region_hi", np.nan)
    tpl_settings = tpl_entry.get("settings", {})

    rows = []
    widths = component_widths_from_params(fit["params"], fit["param_names"], tpl, fit["n_peaks"])
    qrows = component_quantiles_from_fit(fit)

    for k in range(fit["n_peaks"]):
        mu_name = f"mu_{k}"
        width_name = f"width_{k}"

        boot_mu = boot.get("summary", {}).get(mu_name) if boot else None
        tpl_mu = tpl_prop.get("summary", {}).get(mu_name) if tpl_prop else None
        boot_w = boot.get("width_summary", {}).get(width_name) if boot else None
        tpl_w = tpl_prop.get("width_summary", {}).get(width_name) if tpl_prop else None

        if k > 0:
            boot_delta_prev = summarize_mu_delta_samples(boot, k, k - 1)
            tpl_delta_prev = summarize_mu_delta_samples(tpl_prop, k, k - 1)
            delta_prev_fit = float(fit["info"]["mus"][k] - fit["info"]["mus"][k - 1])
            mu_corr_prev_boot = summarize_mu_corr_samples(boot, k, k - 1)
            mu_corr_prev_tpl = summarize_mu_corr_samples(tpl_prop, k, k - 1)
        else:
            boot_delta_prev = tpl_delta_prev = None
            delta_prev_fit = np.nan
            mu_corr_prev_boot = mu_corr_prev_tpl = np.nan

        if k > 0 and fit["info"]["areas"][0] > 0:
            boot_ratio0 = summarize_area_ratio_samples(boot, k, 0, fit["n_peaks"])
            tpl_ratio0 = summarize_area_ratio_samples(tpl_prop, k, 0, fit["n_peaks"])
            area_ratio0_fit = float(fit["info"]["areas"][k] / fit["info"]["areas"][0])
            area_corr0_boot = summarize_area_corr_samples(boot, k, 0, fit["n_peaks"])
            area_corr0_tpl = summarize_area_corr_samples(tpl_prop, k, 0, fit["n_peaks"])
        else:
            boot_ratio0 = tpl_ratio0 = None
            area_ratio0_fit = np.nan
            area_corr0_boot = area_corr0_tpl = np.nan

        row = {
            "label": label,
            "component": k,
            "template": template_name,
            "quality_flag": quality_flag_for_fit(fit, boot=boot, tpl_prop=tpl_prop),
            "fit_region_lo": fit_lo,
            "fit_region_hi": fit_hi,
            "template_region_lo": tpl_lo,
            "template_region_hi": tpl_hi,
            "region_lo": fit_lo,
            "region_hi": fit_hi,
            "template_smooth_nominal": tpl_settings.get("smooth_nominal_template", np.nan),
            "template_resample_factor": tpl_settings.get("resample_factor", np.nan),
            "n_peaks": fit["n_peaks"],
            "background_order": fit["background_order"],
            "loss": fit["loss"],
            "mu_fit_Hz": float(fit["info"]["mus"][k]),
            "mu_bootstrap_median_Hz": boot_mu.get("median", np.nan) if boot_mu else np.nan,
            "mu_bootstrap_minus_Hz": boot_mu.get("minus", np.nan) if boot_mu else np.nan,
            "mu_bootstrap_plus_Hz": boot_mu.get("plus", np.nan) if boot_mu else np.nan,
            "mu_template_median_Hz": tpl_mu.get("median", np.nan) if tpl_mu else np.nan,
            "mu_template_minus_Hz": tpl_mu.get("minus", np.nan) if tpl_mu else np.nan,
            "mu_template_plus_Hz": tpl_mu.get("plus", np.nan) if tpl_mu else np.nan,
            "mu_combined_sigma_Hz": combined_sigma(boot_mu, tpl_mu),
            "mu_bound_lo_Hz": float(fit.get("mu_bounds", [[np.nan, np.nan]] * fit["n_peaks"])[k][0])
            if fit.get("mu_bounds")
            else np.nan,
            "mu_bound_hi_Hz": float(fit.get("mu_bounds", [[np.nan, np.nan]] * fit["n_peaks"])[k][1])
            if fit.get("mu_bounds")
            else np.nan,
            "min_separation_Hz": float(fit.get("min_separation", 0.0)),
            "mu_bounds_text": fit.get("mu_bounds_text", ""),
            "min_observed_separation_Hz": float(fit.get("min_observed_separation", np.nan)),
            "area_fit": float(fit["info"]["areas"][k]),
            "scale_fit": float(fit["info"]["scales"][k]),
            "rms_width_Hz": float(widths[k]),
            "width_bootstrap_median_Hz": boot_w.get("median", np.nan) if boot_w else np.nan,
            "width_bootstrap_minus_Hz": boot_w.get("minus", np.nan) if boot_w else np.nan,
            "width_bootstrap_plus_Hz": boot_w.get("plus", np.nan) if boot_w else np.nan,
            "width_template_median_Hz": tpl_w.get("median", np.nan) if tpl_w else np.nan,
            "width_template_minus_Hz": tpl_w.get("minus", np.nan) if tpl_w else np.nan,
            "width_template_plus_Hz": tpl_w.get("plus", np.nan) if tpl_w else np.nan,
            "width_combined_sigma_Hz": combined_sigma(boot_w, tpl_w),
            "q16_Hz": qrows[k]["q16_Hz"],
            "q50_Hz": qrows[k]["q50_Hz"],
            "q84_Hz": qrows[k]["q84_Hz"],
            "sigma68_Hz": qrows[k]["sigma68_Hz"],
            "sigma68_minus_Hz": qrows[k]["sigma68_minus_Hz"],
            "sigma68_plus_Hz": qrows[k]["sigma68_plus_Hz"],
            "delta_mu_prev_fit_Hz": delta_prev_fit,
            "delta_mu_prev_bootstrap_median_Hz": boot_delta_prev.get("median", np.nan) if boot_delta_prev else np.nan,
            "delta_mu_prev_bootstrap_minus_Hz": boot_delta_prev.get("minus", np.nan) if boot_delta_prev else np.nan,
            "delta_mu_prev_bootstrap_plus_Hz": boot_delta_prev.get("plus", np.nan) if boot_delta_prev else np.nan,
            "delta_mu_prev_template_median_Hz": tpl_delta_prev.get("median", np.nan) if tpl_delta_prev else np.nan,
            "delta_mu_prev_template_minus_Hz": tpl_delta_prev.get("minus", np.nan) if tpl_delta_prev else np.nan,
            "delta_mu_prev_template_plus_Hz": tpl_delta_prev.get("plus", np.nan) if tpl_delta_prev else np.nan,
            "delta_mu_prev_combined_sigma_Hz": combined_sigma(boot_delta_prev, tpl_delta_prev),
            "area_ratio_to_comp0_fit": area_ratio0_fit,
            "area_ratio_to_comp0_bootstrap_median": boot_ratio0.get("median", np.nan) if boot_ratio0 else np.nan,
            "area_ratio_to_comp0_bootstrap_minus": boot_ratio0.get("minus", np.nan) if boot_ratio0 else np.nan,
            "area_ratio_to_comp0_bootstrap_plus": boot_ratio0.get("plus", np.nan) if boot_ratio0 else np.nan,
            "area_ratio_to_comp0_template_median": tpl_ratio0.get("median", np.nan) if tpl_ratio0 else np.nan,
            "area_ratio_to_comp0_template_minus": tpl_ratio0.get("minus", np.nan) if tpl_ratio0 else np.nan,
            "area_ratio_to_comp0_template_plus": tpl_ratio0.get("plus", np.nan) if tpl_ratio0 else np.nan,
            "area_ratio_to_comp0_combined_sigma": combined_sigma(boot_ratio0, tpl_ratio0),
            "mu_corr_prev_bootstrap": mu_corr_prev_boot,
            "mu_corr_prev_template": mu_corr_prev_tpl,
            "area_corr_comp0_bootstrap": area_corr0_boot,
            "area_corr_comp0_template": area_corr0_tpl,
            "target_boot_n_valid": sample_count(boot),
            "template_prop_n_valid": sample_count(tpl_prop),
            "template_bank_n": len(tpl_entry.get("template_bank", [])) if tpl_entry else np.nan,
            "red_chi2": float(fit["red_chi2"]),
            "chi2": float(fit["chi2"]),
            "dof": int(fit["dof"]),
            "notes": notes,
        }
        rows.append(row)
    return rows


def rows_to_tsv(rows: list[dict[str, Any]]) -> str:
    """Format export rows as TSV text (for clipboard copy), preferred-column-first."""
    if not rows:
        return ""
    keys: list[str] = []
    preferred = [
        "label",
        "component",
        "template",
        "quality_flag",
        "fit_region_lo",
        "fit_region_hi",
        "template_region_lo",
        "template_region_hi",
        "mu_fit_Hz",
        "mu_combined_sigma_Hz",
        "rms_width_Hz",
        "width_combined_sigma_Hz",
        "area_fit",
        "scale_fit",
        "delta_mu_prev_fit_Hz",
        "delta_mu_prev_combined_sigma_Hz",
        "area_ratio_to_comp0_fit",
        "area_ratio_to_comp0_combined_sigma",
        "q16_Hz",
        "q50_Hz",
        "q84_Hz",
        "sigma68_Hz",
        "red_chi2",
        "target_boot_n_valid",
        "template_prop_n_valid",
        "notes",
    ]
    for k in preferred:
        if any(k in r for r in rows):
            keys.append(k)
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)

    lines = ["\t".join(keys)]
    for r in rows:
        vals = []
        for k in keys:
            v = r.get(k, "")
            if isinstance(v, float) and (not np.isfinite(v)):
                vals.append("")
            else:
                vals.append(str(v))
        lines.append("\t".join(vals))
    return "\n".join(lines)


def write_delimited(path: str | Path, rows: list[dict[str, Any]], delimiter: str = "\t") -> None:
    keys = sorted({k for row in rows for k in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def export_history_rows(rows: list[dict[str, Any]], path: str | Path, source_file: str | Path | None = None) -> None:
    """Write fit-history rows to TSV/CSV/JSON/XLSX/ODS, dispatched by file suffix.

    XLSX/ODS require the optional ``pandas``+``openpyxl``/``odfpy``
    dependencies (``pip install SchottkySA[xlsx]``); TSV/CSV/JSON never did.
    """
    if not rows:
        raise InputFormatError("No rows to export.")
    path = str(path)
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        payload = {"file": str(source_file) if source_file else None, "rows": rows}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
    elif suffix == ".csv":
        write_delimited(path, rows, delimiter=",")
    elif suffix in (".xlsx", ".ods"):
        try:
            import pandas as pd
        except Exception as exc:
            raise InputFormatError("pandas is required for XLSX/ODS export. Use TSV/CSV otherwise.") from exc
        df = pd.DataFrame(rows)
        if suffix == ".ods":
            # pandas-stubs' `engine` Literal doesn't list "odf", but it is a
            # valid runtime engine (via the optional odfpy dependency).
            df.to_excel(path, index=False, engine="odf")  # pyright: ignore[reportArgumentType]
        else:
            df.to_excel(path, index=False)
    else:
        if not suffix:
            path += ".tsv"
        write_delimited(path, rows, delimiter="\t")


def export_current_overlay_npz(fit: dict[str, Any], path: str | Path) -> None:
    """Export the current fit overlay/residual/components to NPZ for independent plotting."""
    path = str(path)
    if not path.lower().endswith(".npz"):
        path += ".npz"
    tpl = fit["template"]
    payload = {
        "x": fit["x"],
        "y": fit["y"],
        "y_model": fit["y_model"],
        "background": fit["background"],
        "residual": fit["y"] - fit["y_model"],
        "params": fit["params"],
        "param_names": np.asarray(fit["param_names"], dtype=object),
        "mus": fit["info"]["mus"],
        "areas": fit["info"]["areas"],
        "scales": fit["info"]["scales"],
        "red_chi2": np.asarray(fit["red_chi2"]),
        "chi2": np.asarray(fit["chi2"]),
        "dof": np.asarray(fit["dof"]),
        "components": np.asarray(fit["components"]),
        "template_u": tpl.u,
        "template_pdf": tpl.pdf,
        "template_cdf": tpl.cdf,
    }
    np.savez(path, **payload)


def write_curve_csv(
    path: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    y_model: np.ndarray,
    background: np.ndarray,
    components: np.ndarray,
) -> None:
    """Write per-bin data/model/background/component curves to CSV (``ssa run``'s
    curve export). Columns: ``frequency_hz, y, y_model, background,
    component_0, ..., component_{K-1}, residual``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_model = np.asarray(y_model, dtype=float)
    background = np.asarray(background, dtype=float)
    components = np.asarray(components, dtype=float)
    n_components = components.shape[0] if components.ndim == 2 else 0

    fieldnames = (
        ["frequency_hz", "y", "y_model", "background"] + [f"component_{k}" for k in range(n_components)] + ["residual"]
    )
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for i in range(len(x)):
            row = [x[i], y[i], y_model[i], background[i]]
            row.extend(components[k, i] for k in range(n_components))
            row.append(y[i] - y_model[i])
            writer.writerow(row)
