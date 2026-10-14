"""Joint fitting of different invented profiles; no experimental data."""

from dataclasses import replace

import numpy as np
import pytest

from ssa.config import FitConfig, UncertaintyConfig
from ssa.exceptions import FitConfigurationError
from ssa.fitting import fit_template_region, template_model
from ssa.io import make_export_rows_for_result
from ssa.pipeline import run_fit_with_uncertainty
from ssa.templates import build_peak_template
from ssa.uncertainty import (
    bootstrap_fit,
    component_quantiles_from_fit,
    component_widths_from_params,
    propagate_template_uncertainty_to_target_fit,
    quality_flag_for_fit,
)


def shapes():
    u = np.linspace(-3, 3, 201)
    first = build_peak_template(u, np.exp(-0.5 * (u / 0.32) ** 2), name="narrow")
    second = build_peak_template(
        u, np.exp(-0.5 * (u / 0.5) ** 2) + 0.5 * np.exp(-0.5 * ((u - 0.65) / 0.25) ** 2), name="shoulder"
    )
    return first, second


def problem(templates=None, common_scale=False):
    first, second = shapes()
    profiles = templates or [first, second]
    n = len(profiles)
    x = np.linspace(-3, 5, 401)
    mus = np.arange(n) * 0.85
    areas = np.array([4, 3, 2][:n], dtype=float)
    scales = np.ones(n)
    p = np.r_[mus, np.log(areas)]
    y = sum(
        t.evaluate_component(x, mu=m, area=a, scale=s) for t, m, a, s in zip(profiles, mus, areas, scales, strict=True)
    )
    bounds = [(m - 0.22, m + 0.22) for m in mus]
    fit = fit_template_region(
        x,
        y,
        profiles[0],
        n,
        component_templates=profiles,
        init_mus=(mus + 0.03).tolist(),
        mu_bounds=bounds,
        sigma=0.02,
        allow_scale=False,
        common_scale=common_scale,
        background_order=-1,
    )
    return profiles, fit, p


def test_joint_two_shape_fit_recovers_centres_areas_and_shape_specific_widths():
    profiles, fit, expected = problem()
    np.testing.assert_allclose(fit["params"], expected, atol=1e-7)
    assert fit["chi2"] < 1e-12
    widths = component_widths_from_params(fit["params"], fit["param_names"], profiles[0], 2, profiles)
    assert widths[1] > widths[0]
    rows = component_quantiles_from_fit(fit)
    for k, tpl in enumerate(profiles):
        np.testing.assert_allclose(
            [rows[k][f"q{q}_Hz"] for q in (16, 50, 84)], tpl.quantile([0.16, 0.5, 0.84]) + fit["info"]["mus"][k]
        )
        assert rows[k]["rms_width_Hz"] == pytest.approx(tpl.moments()[1])
    incorrect = fit_template_region(
        fit["x"],
        fit["y"],
        profiles[0],
        2,
        init_mus=[0, 0.85],
        mu_bounds=fit["mu_bounds"],
        sigma=0.02,
        allow_scale=False,
        background_order=-1,
    )
    assert incorrect["chi2"] > fit["chi2"] + 100
    exported = make_export_rows_for_result(
        {"fit": fit},
        label="invented",
        template_registry=[
            {"template": profiles[0], "region_lo": 10, "region_hi": 11},
            {"template": profiles[1], "region_lo": 20, "region_hi": 21},
        ],
    )
    assert [row["template"] for row in exported] == ["narrow", "shoulder"]
    assert [row["template_region_lo"] for row in exported] == [10, 20]
    np.testing.assert_allclose([row["rms_width_Hz"] for row in exported], widths)


@pytest.mark.parametrize("common", [True, False])
def test_scale_parameters_apply_to_assigned_profiles(common):
    first, second = shapes()
    x = np.linspace(-3, 3, 101)
    scales = [1.2, 1.2] if common else [0.8, 1.3]
    p = np.r_[[-0.3, 0.5], np.log([3, 2]), np.log(scales[:1] if common else scales)]
    y, components, _, _ = template_model(
        x, p, first, 2, background_order=-1, allow_scale=True, common_scale=common, component_templates=[first, second]
    )
    np.testing.assert_allclose(components[1], second.evaluate_component(x, mu=0.5, area=2, scale=scales[1]))
    np.testing.assert_allclose(y, components.sum(axis=0))


def test_shared_template_path_matches_existing_numerics():
    first, _ = shapes()
    _, fit, _ = problem([first, first])
    legacy = fit_template_region(
        fit["x"],
        fit["y"],
        first,
        2,
        init_mus=[0.03, 0.88],
        mu_bounds=fit["mu_bounds"],
        sigma=0.02,
        allow_scale=False,
        background_order=-1,
    )
    np.testing.assert_array_equal(fit["params"], legacy["params"])
    np.testing.assert_array_equal(fit["y_model"], legacy["y_model"])


def test_different_profiles_require_stable_component_bounds():
    first, second = shapes()
    with pytest.raises(FitConfigurationError, match="ordered, non-overlapping"):
        fit_template_region(np.linspace(-2, 2, 101), np.ones(101), first, 2, component_templates=[first, second])
    with pytest.raises(FitConfigurationError, match="exactly one"):
        fit_template_region(np.linspace(-2, 2, 101), np.ones(101), first, 2, component_templates=[first])


def test_residual_bootstrap_keeps_the_correct_shapes():
    profiles, fit, expected = problem()
    boot = bootstrap_fit(fit, n_boot=5, block_size=1, rng=np.random.default_rng(8))
    assert len(boot["params"]) == 5
    np.testing.assert_allclose(boot["params"], np.tile(expected, (5, 1)), atol=1e-6)
    np.testing.assert_allclose(boot["width_samples"][:, 1], profiles[1].moments()[1])


def test_template_draws_preserve_shared_banks_and_independently_draw_distinct_banks():
    first, second = shapes()
    profiles, fit, _ = problem([first, second, first])
    bank_a = [replace(first, u=first.u + d, name=f"a_{i}") for i, d in enumerate(np.linspace(-0.02, 0.02, 6))]
    bank_b = [replace(second, u=second.u + d, name=f"b_{i}") for i, d in enumerate(np.linspace(-0.03, 0.03, 6))]
    result = propagate_template_uncertainty_to_target_fit(
        fit, bank_a, n_draws=6, random_seed=12, component_template_banks=[bank_a, bank_b, bank_a]
    )
    indices = np.asarray(result["selected_template_indices"])
    assert indices.shape == (6, 3)
    np.testing.assert_array_equal(indices[:, 0], indices[:, 2])
    assert not np.array_equal(indices[:, 0], indices[:, 1])
    assert result["independent_template_banks_assumed"]
    assert np.cov(result["params"][:, [0, 2]].T)[0, 1] > 0
    repeat = propagate_template_uncertainty_to_target_fit(
        fit, bank_a, n_draws=6, random_seed=12, component_template_banks=[bank_a, bank_b, bank_a]
    )
    np.testing.assert_array_equal(result["params"], repeat["params"])
    assert "independent_template_banks_assumed" in quality_flag_for_fit(fit, tpl_prop=result)


def test_mixed_fit_cannot_silently_use_one_bank_for_every_component():
    profiles, fit, _ = problem()
    with pytest.raises(ValueError, match="component_template_banks"):
        propagate_template_uncertainty_to_target_fit(fit, [profiles[0]])
    result = propagate_template_uncertainty_to_target_fit(
        fit, [profiles[0]], n_draws=1, component_template_banks=[[profiles[0]], []]
    )
    assert result["missing_template_bank_components"] == [1]
    assert "component_template_bank_missing" in quality_flag_for_fit(fit, tpl_prop=result)


def test_pipeline_joint_fit_and_uncertainty_streams():
    profiles, fit, _ = problem()
    result = run_fit_with_uncertainty(
        fit["x"],
        fit["y"],
        profiles[0],
        [profiles[0]],
        FitConfig(n_peaks=2, init_mus=[0, 0.85], mu_bounds=fit["mu_bounds"], allow_scale=False, background_order=-1),
        UncertaintyConfig(n_boot=3, n_template_prop=2),
        sigma=0.02,
        component_templates=profiles,
        component_template_banks=[[profiles[0]], [profiles[1]]],
    )
    assert len(result["bootstrap"]["params"]) == 3
    assert len(result["template_propagation"]["params"]) == 2


def test_overlay_export_retains_each_shape(tmp_path):
    from ssa.io import export_current_overlay_npz

    profiles, fit, _ = problem()
    path = tmp_path / "overlay.npz"
    export_current_overlay_npz(fit, path)
    with np.load(path) as data:
        assert data["component_template_names"].tolist() == [t.name for t in profiles]
        for k, tpl in enumerate(profiles):
            np.testing.assert_array_equal(data[f"component_{k}_template_pdf"], tpl.pdf)
        np.testing.assert_array_equal(data["template_pdf"], profiles[0].pdf)
