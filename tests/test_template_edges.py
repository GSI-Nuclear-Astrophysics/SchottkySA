"""Independent closures tested on invented profiles, without experimental data."""

from dataclasses import asdict

import numpy as np
import pytest

from ssa.config import TemplateBuildConfig
from ssa.pipeline import build_template_bank
from ssa.session import _template_config_from_dict
from ssa.templates import apply_template_edge_handling


def profile():
    x = np.linspace(-1, 1, 81)
    return x, np.exp(-(x**2) / 0.7)


@pytest.mark.parametrize("mode", ["pchip_extrapolate_to_zero", "pchip_monotone_extrapolate_to_zero", "pad_to_zero"])
def test_asymmetric_closure_hits_exact_anchors_and_keeps_samples(mode):
    x, y = profile()
    xx, yy, info = apply_template_edge_handling(
        x,
        y,
        mode=mode,
        left_edge_points=3,
        right_edge_points=9,
        left_tail_width_hz=0.013,
        right_tail_width_hz=0.737,
    )
    assert xx[0] == pytest.approx(x[0] - 0.013)
    assert xx[-1] == pytest.approx(x[-1] + 0.737)
    assert yy[0] == yy[-1] == 0
    assert np.all(np.diff(xx) > 0)
    np.testing.assert_array_equal(yy[(xx >= x[0]) & (xx <= x[-1])], y)
    assert info["left_edge_points_used"] == 3
    assert info["right_edge_points_used"] == 9
    assert info["left_zero_frequency_hz"] == pytest.approx(xx[0])
    assert info["right_zero_frequency_hz"] == pytest.approx(xx[-1])


def test_edge_counts_are_capped_and_cosine_uses_each_count():
    x, y = profile()
    _, yy, info = apply_template_edge_handling(
        x,
        y,
        mode="cosine_taper_to_zero",
        left_edge_points=3,
        right_edge_points=1000,
    )
    assert info["left_edge_points_used"] == 3
    assert info["right_edge_points_used"] == len(x) // 2 - 1
    assert yy[0] == yy[-1] == 0
    np.testing.assert_array_equal(yy[3:-39], y[3:-39])


@pytest.mark.parametrize("mode", ["pchip_extrapolate_to_zero", "local_slope_to_zero"])
def test_template_bank_uses_same_closure_and_is_repeatable(mode):
    x, y = profile()
    config = TemplateBuildConfig(
        n_template_boot=6,
        sg_window=9,
        edge_mode=mode,
        left_edge_points=3,
        right_edge_points=7,
        left_tail_width_hz=0.11,
        right_tail_width_hz=0.73,
    )
    result = build_template_bank(x, y, config)
    repeat = build_template_bank(x, y, config)
    assert len(result["templates"]) == 6
    for template in [result["template_nominal"], *result["templates"]]:
        assert template.edge_info["left_zero_frequency_hz"] == pytest.approx(-1.11)
        assert template.edge_info["right_zero_frequency_hz"] == pytest.approx(1.73)
        assert template.pdf[0] == template.pdf[-1] == 0
        assert np.trapezoid(template.pdf, template.u) == pytest.approx(1, rel=1e-3)
        assert np.trapezoid(template.u * template.pdf, template.u) == pytest.approx(0, abs=1e-12)
    for a, b in zip(result["templates"], repeat["templates"], strict=True):
        np.testing.assert_array_equal(a.pdf, b.pdf)


def test_old_config_defaults_and_new_overrides_round_trip():
    original = TemplateBuildConfig(
        left_edge_points=3, right_edge_points=8, left_tail_width_hz=0.2, right_tail_width_hz=0.8
    )
    assert _template_config_from_dict(asdict(original)) == original
    old = asdict(original)
    for key in ("left_edge_points", "right_edge_points", "left_tail_width_hz", "right_tail_width_hz"):
        old.pop(key)
    assert _template_config_from_dict(old) == TemplateBuildConfig()


@pytest.mark.parametrize(
    "settings",
    [
        {"left_edge_points": -1},
        {"right_edge_points": 2.5},
        {"left_tail_width_hz": float("nan")},
        {"right_tail_width_hz": -1},
    ],
)
def test_invalid_overrides_raise(settings):
    x, y = profile()
    with pytest.raises(ValueError):
        apply_template_edge_handling(x, y, mode="pchip_extrapolate_to_zero", **settings)


def test_local_slope_uses_all_selected_points_and_has_monotone_closure():
    x = np.linspace(-1, 1, 81)
    y = 1 - 0.7 * x**2
    outputs = []
    for n in (3, 11):
        xx, yy, info = apply_template_edge_handling(
            x,
            y,
            mode="local_slope_to_zero",
            left_edge_points=n,
            right_edge_points=6,
            left_tail_width_hz=0.4,
            right_tail_width_hz=0.5,
        )
        outputs.append(info["left_edge_outward_slope_fit"])
        assert xx[0] == pytest.approx(-1.4)
        assert xx[-1] == pytest.approx(1.5)
        assert yy[0] == yy[-1] == 0
        assert np.all(np.diff(yy[xx <= -1]) >= -1e-12)
        assert np.all(np.diff(yy[xx >= 1]) <= 1e-12)
        np.testing.assert_array_equal(yy[(xx >= -1) & (xx <= 1)], y)
    assert outputs[0] != outputs[1]


def test_local_slope_limits_outward_rise_and_closes_nonnegative():
    x = np.linspace(-1, 1, 81)
    y = 1 + 0.7 * x**2
    _, yy, info = apply_template_edge_handling(x, y, mode="local_slope_to_zero")
    assert info["left_edge_slope_limited"]
    assert info["right_edge_slope_limited"]
    assert np.min(yy) >= 0


def test_linear_fallback_left_tail_reaches_zero_at_outer_anchor(monkeypatch):
    import ssa.templates as module

    def fail_interpolation(*args, **kwargs):
        raise ValueError("Force fallback")

    monkeypatch.setattr(module, "PchipInterpolator", fail_interpolation)
    x, y = profile()
    xx, yy, info = module.apply_template_edge_handling(
        x, y, mode="pchip_extrapolate_to_zero", left_tail_width_hz=0.11, right_tail_width_hz=0.21
    )
    assert yy[0] == yy[-1] == 0
    assert info["left_tail_model"] == info["right_tail_model"] == "linear_to_zero_fallback"
    assert np.all(np.diff(yy[xx < x[0]]) >= 0)
    assert np.all(np.diff(yy[xx > x[-1]]) <= 0)
