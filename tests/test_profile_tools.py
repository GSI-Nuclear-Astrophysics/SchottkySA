"""Public, artificial-profile checks of units, mixing, diagnostics and replay."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from ssa.config import FitConfig, TemplateBuildConfig, UncertaintyConfig
from ssa.fitting import fit_template_region
from ssa.io import make_export_rows_for_result
from ssa.pipeline import build_template_bank, run_fit_with_uncertainty
from ssa.preprocessing import trapz
from ssa.profile_tools import (
    combine_template_entries,
    compare_template_entries,
    rescale_template,
    transfer_template_entry,
)
from ssa.session import (
    ComparisonRecord,
    FitRecord,
    FormState,
    InputRef,
    Session,
    TemplateRecord,
    _sha256_of,
    load_session,
    replay_session,
    save_session,
)
from ssa.templates import PeakTemplate
from ssa.uncertainty import quality_flag_for_fit


def artificial_profile(width=0.4, name="profile"):
    u = np.linspace(-4 * width, 4 * width, 241)
    pdf = np.exp(-0.5 * (u / width) ** 2)
    pdf /= trapz(pdf, u)
    cdf = cumulative_trapezoid(pdf, u, initial=0)
    cdf /= cdf[-1]
    return PeakTemplate(u, pdf, cdf, 5.0, 1260.0, width, float(u[1] - u[0]), name)


def artificial_entry(width=0.4, harmonic=126, index=0):
    tpl = artificial_profile(width, f"source_{index}")
    return {
        "name": tpl.name,
        "harmonic": harmonic,
        "template": tpl,
        "template_bank": [tpl, tpl],
        "source_leaf_ids": [index],
        "region_lo": tpl.raw_cog + tpl.u[0],
        "region_hi": tpl.raw_cog + tpl.u[-1],
    }


def measured_regions(contamination=False):
    """Two noisy measured windows, related by a known harmonic ratio."""
    entries, xs, ys = [], [], []
    for i, h in enumerate((126, 127)):
        factor = h / 126
        x = np.linspace(1260 * factor - 3 * factor, 1260 * factor + 3 * factor, 241)
        tpl = rescale_template(artificial_profile(), factor)
        y = tpl.evaluate_component(x, 1260 * factor, 20) + 0.004 * np.sin(np.arange(len(x)) * 1.7) + 0.01
        if contamination and i == 1:
            y += 3 * np.exp(-0.5 * ((x - (1260 * factor + 1.0)) / 0.16) ** 2)
        config = TemplateBuildConfig(
            name=f"source_{h}", n_template_boot=8, resample_factor=2, sg_window=9, smooth=False, edge_mode="none"
        )
        bank = build_template_bank(x, y, config)
        entries.append(
            {
                "name": config.name,
                "harmonic": h,
                "template": bank["template_nominal"],
                "template_bank": bank["templates"],
                "region_lo": float(x[0]),
                "region_hi": float(x[-1]),
                "config": config,
                "settings": {"smooth_nominal_template": config.smooth, "resample_factor": config.resample_factor},
                "source_leaf_ids": [i],
                "combination": None,
                "full": bank,
                "summary": bank["summary"],
            }
        )
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys), entries


def test_known_harmonic_scaling_preserves_area_and_centres():
    source = artificial_profile()
    assert rescale_template(source, 1) is source
    result = rescale_template(source, 214 / 126)
    assert trapz(result.pdf, result.u) == pytest.approx(1)
    assert result.moments()[0] == pytest.approx(0, abs=1e-14)
    assert result.moments()[1] == pytest.approx(source.moments()[1] * 214 / 126)
    assert result.raw_cog == pytest.approx(source.raw_cog * 214 / 126)
    for bad in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError):
            rescale_template(source, bad)


def test_combination_preserves_widths_union_support_and_user_weights():
    entries = [artificial_entry(0.4, index=0), artificial_entry(1, index=1)]
    weights = np.array([1.0, 3.0])
    bank = combine_template_entries(entries, weights=weights, output_harmonic=126, n_boot=0)
    tpl = bank["template_nominal"]
    expected_variance = (
        0.25 * entries[0]["template"].moments()[1] ** 2 + 0.75 * entries[1]["template"].moments()[1] ** 2
    )
    assert tpl.moments()[1] ** 2 == pytest.approx(expected_variance, rel=5e-4)
    assert trapz(tpl.pdf, tpl.u) == pytest.approx(1)
    assert tpl.moments()[0] == pytest.approx(0, abs=1e-14)
    assert np.ptp(tpl.u) >= 8
    np.testing.assert_array_equal(weights, [1, 3])
    assert not tpl.edge_info["additional_smoothing"]
    assert bank["templates"] == []


def test_harmonic_profiles_average_on_same_frequency_coordinate():
    a = artificial_entry(index=0)
    b = artificial_entry(harmonic=127, index=1)
    b["template"] = rescale_template(a["template"], 127 / 126)
    b["template_bank"] = [b["template"]]
    out = combine_template_entries([a, b], output_harmonic=126, n_boot=4)
    assert out["template_nominal"].moments()[1] == pytest.approx(a["template"].moments()[1], rel=1e-5)
    assert out["settings"]["output_harmonic"] == 126


@pytest.mark.parametrize("mode", ["source_sensitivity", "independent"])
def test_draws_deterministic_with_multiple_workers(mode):
    entries = [artificial_entry(0.4, index=0), artificial_entry(0.8, index=1)]
    serial = combine_template_entries(entries, n_boot=20, output_harmonic=126, bank_mode=mode, workers=1)
    parallel = combine_template_entries(entries, n_boot=20, output_harmonic=126, bank_mode=mode, workers=4)
    for a, b in zip(serial["templates"], parallel["templates"], strict=True):
        np.testing.assert_array_equal(a.u, b.u)
        np.testing.assert_array_equal(a.pdf, b.pdf)
    widths = np.asarray([t.moments()[1] for t in serial["templates"]])
    if mode == "source_sensitivity":
        assert len(np.unique(widths)) == 2  # does not collapse different sources into one narrow error band
    else:
        assert np.ptp(widths) < 1e-12  # both toy banks contain identical draws, so the averaged shape is fixed
    fit = fit_template_region(
        np.linspace(-4, 4, 201),
        artificial_profile().evaluate_component(np.linspace(-4, 4, 201), 0, 10),
        serial["template_nominal"],
        n_peaks=1,
        allow_scale=False,
    )
    expected = (
        "combined_template_source_sensitivity"
        if mode == "source_sensitivity"
        else "combined_template_independence_assumed"
    )
    assert expected in quality_flag_for_fit(fit)


@pytest.mark.parametrize("weights", [[1], [1, -1], [1, np.nan], [0, 1]])
def test_bad_combination_weights_rejected(weights):
    with pytest.raises(ValueError, match="weight"):
        combine_template_entries([artificial_entry(index=0), artificial_entry(index=1)], weights=weights)


def test_combination_rejects_reused_sources_and_absent_banks():
    with pytest.raises(ValueError, match="same underlying source"):
        combine_template_entries([artificial_entry(), artificial_entry()])
    a, b = artificial_entry(index=0), artificial_entry(index=1)
    b["template_bank"] = []
    with pytest.raises(ValueError, match="bootstrap templates"):
        combine_template_entries([a, b])
    a["template"] = replace(a["template"], dx=0)
    with pytest.raises(ValueError, match="grid step"):
        combine_template_entries([a, b], n_boot=0)


def test_cross_fits_use_known_ratio_and_show_contamination_residuals():
    f, y, entries = measured_regions()
    clean = compare_template_entries(f, y, entries, 0, 1)
    for direction in clean:
        row = direction["row"]
        assert row["success"]
        assert row["extra_width_scale_fit"] == 1
        assert abs(row["mu_minus_target_cog_Hz"]) < 0.002
        assert row["harmonic_scale_factor"] == pytest.approx(
            entries[row["target_id"]]["harmonic"] / entries[row["source_id"]]["harmonic"]
        )
    cf, cy, contaminated = measured_regions(contamination=True)
    mismatch = compare_template_entries(cf, cy, contaminated, 0, 1)
    assert np.linalg.norm(mismatch[0]["fit"]["y"] - mismatch[0]["fit"]["y_model"]) > 10 * np.linalg.norm(
        clean[0]["fit"]["y"] - clean[0]["fit"]["y_model"]
    )
    serial = compare_template_entries(f, y, entries, 0, 1, workers=1)
    assert [d["row"] for d in clean] == [d["row"] for d in serial]


def test_session_replays_composite_harmonic_fit_and_comparisons(tmp_path):
    f, y, entries = measured_regions()
    path = tmp_path / "artificial.npz"
    np.savez(path, frequency=f, amplitude=y)
    combination = {"source_ids": [0, 1], "weights": [1, 2], "bank_mode": "source_sensitivity", "workers": 2}
    cfg = TemplateBuildConfig(name="mean", n_template_boot=8, smooth=False)
    bank = combine_template_entries(entries, name=cfg.name, output_harmonic=126, weights=[1, 2], n_boot=8)
    composite = {
        "name": cfg.name,
        "template": bank["template_nominal"],
        "template_bank": bank["templates"],
        "harmonic": 126,
        "settings": bank["settings"],
        "region_lo": 1257,
        "region_hi": 1263,
    }
    transferred = transfer_template_entry(composite, 127)
    mask = f > 1265
    fc = FitConfig(allow_scale=False, init_mus=[1270])
    uc = UncertaintyConfig(n_boot=8, n_template_prop=8)
    result = run_fit_with_uncertainty(f[mask], y[mask], transferred["template"], transferred["template_bank"], fc, uc)
    rows = make_export_rows_for_result(result, label="target", template_registry=[transferred])
    directions = compare_template_entries(f, y, entries, 0, 1)
    session = Session(
        2,
        "test",
        "2026-10-06",
        InputRef(str(path), _sha256_of(path), "frequency", "amplitude"),
        FormState(cfg, fc, "target", True, uc, (1267, 1273), 2, {"harmonics": [126, 127], "active_view": 1}),
        templates=[
            TemplateRecord(i, (e["region_lo"], e["region_hi"]), e["config"], {}, e["harmonic"])
            for i, e in enumerate(entries)
        ]
        + [TemplateRecord(2, (1257, 1263), cfg, {}, 126, combination)],
        fits=[FitRecord("target", "", 2, (float(f[mask][0]), float(f[mask][-1])), fc, uc, rows, 127, 1)],
        comparisons=[ComparisonRecord(0, 1, verify_rows=[d["row"] for d in directions])],
    )
    saved = tmp_path / "session.toml"
    save_session(saved, session)
    loaded = load_session(saved)
    assert loaded == session
    report = replay_session(loaded)
    assert report.ok, report.mismatches
    assert report.fits[0]["view_index"] == 1
    assert report.fits[0]["rows"][0]["harmonic_scale_factor"] == pytest.approx(127 / 126)
    assert len(report.comparisons) == 1
    assert report.templates[-1]["source_leaf_ids"] == [0, 1]
    assert report.templates[-1]["settings"] == bank["settings"]
    # A compound source cannot be mixed with one of its own constituents.
    loaded.templates.append(TemplateRecord(3, (1257, 1263), cfg, {}, 126, {**combination, "source_ids": [0, 2]}))
    with pytest.raises(ValueError, match="same underlying source"):
        replay_session(loaded)
