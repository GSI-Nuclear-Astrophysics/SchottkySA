"""ssa package vs. frozen numeric baseline (regression test).

Each scenario here fits/bootstraps a fixed synthetic dataset and compares
the result against a value captured once in ``tests/baseline/`` (see
``tests/baseline/make_fixtures.py``), so a passing suite is direct evidence
that fit/bootstrap numerics have not silently drifted, not just "the code
runs".

The baseline file also carries a ``mcmc_single_peak_DOCUMENTATION_ONLY``
entry and ``durbin_watson``/``aic``/``bic`` fields on some fit results;
these are intentionally not exercised or compared here, since Bayesian MCMC
and those three diagnostics are not part of ``ssa``.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import assert_close, load_xy
from ssa.fitting import fit_template_region
from ssa.preprocessing import prepare_xy, robust_sigma_from_second_difference
from ssa.templates import bootstrap_template_bank, build_peak_template
from ssa.uncertainty import (
    bootstrap_fit,
    propagate_template_uncertainty_to_target_fit,
    quality_flag_for_fit,
    summarize_mu_delta_samples,
)


def test_prepare_xy_duplicate_averaging(baseline):
    x_dup = np.array([5.0, 1.0, 3.0, 1.0, 3.0, 2.0, 4.0])
    y_dup = np.array([50.0, 10.0, 30.0, 12.0, 32.0, 20.0, 40.0])
    xs, ys = prepare_xy(x_dup, y_dup)
    assert_close(xs.tolist(), baseline["prepare_xy_duplicate_case"]["x"])
    assert_close(ys.tolist(), baseline["prepare_xy_duplicate_case"]["y"])


def test_robust_sigma(baseline):
    rng = np.random.default_rng(7)
    white = rng.normal(0, 1.0, 2000)
    assert robust_sigma_from_second_difference(white) == pytest.approx(baseline["robust_sigma_white_noise"], rel=1e-8)


@pytest.fixture(scope="module")
def reference_template():
    f_ref, y_ref = load_xy("synthetic_reference_peak.npz", "frequency", "amplitude")
    return build_peak_template(
        f_ref,
        y_ref,
        name="reference_peak",
        clip_negative=True,
        smooth=False,
        sg_window=None,
        sg_poly=3,
        resample_factor=16,
        edge_mode="none",
        edge_width_hz=0.0,
        edge_fraction=0.08,
    )


def test_template_default_construction(reference_template, baseline):
    tpl = reference_template
    expected = baseline["template_default"]
    assert tpl.raw_area == pytest.approx(expected["raw_area"], rel=1e-8)
    assert tpl.raw_cog == pytest.approx(expected["raw_cog"], rel=1e-8)
    assert tpl.raw_std == pytest.approx(expected["raw_std"], rel=1e-8)
    assert len(tpl.u) == expected["n_u"]
    assert float(tpl.u.min()) == pytest.approx(expected["u_min"], rel=1e-8)
    assert float(tpl.u.max()) == pytest.approx(expected["u_max"], rel=1e-8)


def test_template_bootstrap_bank(baseline):
    f_ref, y_ref = load_xy("synthetic_reference_peak.npz", "frequency", "amplitude")
    bank = bootstrap_template_bank(
        f_ref,
        y_ref,
        n_boot=300,
        block_size="auto",
        random_seed=12345,
        name="reference_peak",
        clip_negative=True,
        smooth=False,
        sg_window=None,
        sg_poly=3,
        resample_factor=16,
        edge_mode="none",
        edge_width_hz=0.0,
        edge_fraction=0.08,
    )
    expected = baseline["template_bootstrap_bank"]
    assert bank["summary"]["n_templates"] == expected["n_templates"]
    assert bank["residual_model"]["block_size"] == expected["block_size"]
    assert_close(bank["summary"]["raw_cog"], expected["summary_raw_cog"])


def test_single_peak_fit(reference_template, baseline):
    f_single, y_single = load_xy("synthetic_single_peak.npz", "frequency", "amplitude")
    fit1 = fit_template_region(
        f_single,
        y_single,
        template=reference_template,
        n_peaks=1,
        background_order=0,
        allow_scale=True,
        common_scale=True,
        loss="linear",
        min_separation=0.0,
        mu_bounds=None,
    )
    expected = baseline["fit_single_peak"]
    assert fit1["success"] == expected["success"]
    assert_close(fit1["params"].tolist(), expected["params"])
    assert fit1["red_chi2"] == pytest.approx(expected["red_chi2"], rel=1e-8)
    # Not compared to baseline["quality_flag"]: that field could include a
    # Durbin-Watson-based "autocorrelated_residuals" token that ssa's
    # quality_flag_for_fit does not compute. For this scenario both happen
    # to be "ok" (red_chi2 and DW are both within bounds), asserted directly
    # rather than against the frozen baseline value.
    assert quality_flag_for_fit(fit1) == "ok"

    boot1 = bootstrap_fit(fit1, n_boot=1000, block_size="auto", rng=np.random.default_rng(12345))
    expected_boot = baseline["fit_single_peak_bootstrap"]
    assert len(boot1["params"]) == expected_boot["n_valid"]
    assert_close(boot1["summary"]["mu_0"], expected_boot["mu_0"])

    # Passing boot/tpl_prop lets quality_flag_for_fit check replicate
    # reliability. A healthy 1000-replicate bootstrap with no template
    # propagation should be flagged for the *missing* second source, not
    # for the bootstrap itself.
    flag_with_boot_only = quality_flag_for_fit(fit1, boot=boot1, tpl_prop=None)
    assert "bootstrap_failed" not in flag_with_boot_only
    assert "bootstrap_insufficient_samples" not in flag_with_boot_only
    assert "mu_combined_sigma_single_source" in flag_with_boot_only


def test_two_peak_fit_with_mu_bounds(reference_template, baseline):
    """The scientifically meaningful two-peak scenario: explicit, ordered,
    separated centroid search ranges (as recommended for contaminated fits)."""
    f_two, y_two = load_xy("synthetic_two_peak_contaminated.npz", "frequency", "amplitude")
    fit2b = fit_template_region(
        f_two,
        y_two,
        template=reference_template,
        n_peaks=2,
        background_order=0,
        allow_scale=True,
        common_scale=True,
        loss="linear",
        min_separation=3.0,
        mu_bounds=[(1_937_385.0, 1_937_393.0), (1_937_396.0, 1_937_402.0)],
    )
    expected = baseline["fit_two_peak_with_mu_bounds"]
    assert_close(fit2b["params"].tolist(), expected["params"])
    assert fit2b["min_separation_satisfied"] == expected["min_separation_satisfied"]
    assert fit2b["mu_bounds_text"] == expected["mu_bounds_text"]


def test_two_peak_fit_unconstrained_is_degenerate(reference_template):
    """Documents (not endorses) the optimizer's behaviour with default
    (unbounded) initial centroid guesses on a contaminated two-peak region:
    without explicit mu_bounds, one component collapses to near-zero area
    (effectively vanishing) rather than resolving both peaks.

    This fit sits at a genuine bifurcation point in the optimizer's
    landscape: unlike every other scenario in this file, its exact fitted
    parameters are NOT reproducible bit-for-bit across independently
    resolved numpy/scipy environments (observed swings up to ~14% relative
    on individual parameters between CI runs on different dependency
    versions) -- so only the qualitative degenerate signature is checked
    here, not exact values against a frozen baseline. See
    test_two_peak_fit_with_mu_bounds for the recommended, numerically
    stable workflow that avoids this failure mode entirely."""
    f_two, y_two = load_xy("synthetic_two_peak_contaminated.npz", "frequency", "amplitude")
    fit2 = fit_template_region(
        f_two,
        y_two,
        template=reference_template,
        n_peaks=2,
        background_order=0,
        allow_scale=True,
        common_scale=True,
        loss="linear",
        min_separation=0.0,
        mu_bounds=None,
    )
    assert fit2["success"]
    mus = np.asarray(fit2["info"]["mus"])
    areas = np.asarray(fit2["info"]["areas"])
    assert mus.shape == (2,)
    assert mus[0] < mus[1]  # component order preserved
    # The documented degenerate signature: one component's area collapses to
    # a small fraction of the other's, rather than both resolving comparable
    # signal. A healthy two-peak fit on this data has areas within the same
    # order of magnitude; a two-orders-of-magnitude gap is unambiguous.
    assert min(areas) / max(areas) < 1e-2


def test_two_peak_fit_no_bg_no_scale_softl1(reference_template, baseline):
    f_two, y_two = load_xy("synthetic_two_peak_contaminated.npz", "frequency", "amplitude")
    fit2c = fit_template_region(
        f_two,
        y_two,
        template=reference_template,
        n_peaks=2,
        background_order=-1,
        allow_scale=False,
        common_scale=True,
        loss="soft_l1",
        min_separation=0.0,
        mu_bounds=None,
    )
    expected = baseline["fit_two_peak_no_bg_no_scale_softl1"]
    assert_close(fit2c["params"].tolist(), expected["params"])
    assert fit2c["red_chi2"] == pytest.approx(expected["red_chi2"], rel=1e-8)


def test_uncertainty_propagation_and_pairwise_diagnostics(reference_template):
    """Exercises bootstrap_fit/propagate_template_uncertainty_to_target_fit
    and the pairwise-diagnostics helper (summarize_mu_delta_samples) on a
    real bootstrap/propagation run, using the same UNCONSTRAINED
    (fit_two_peak) fit as test_two_peak_fit_unconstrained_is_degenerate.

    That base fit is a documented near-bifurcation case (see its docstring),
    so its bootstrap/propagation *statistics* are not reproducible
    bit-for-bit against a frozen baseline across environments either --
    only structural validity (shape, finiteness, a plausible valid-sample
    count) is checked here, not exact percentile values. Well-conditioned
    scenarios (e.g. test_two_peak_fit_with_mu_bounds) still get exact
    baseline comparison; this test is only about exercising the
    bootstrap/pairwise-diagnostics machinery correctly."""
    f_two, y_two = load_xy("synthetic_two_peak_contaminated.npz", "frequency", "amplitude")
    fit2 = fit_template_region(
        f_two,
        y_two,
        template=reference_template,
        n_peaks=2,
        background_order=0,
        allow_scale=True,
        common_scale=True,
        loss="linear",
        min_separation=0.0,
        mu_bounds=None,
    )
    bank = bootstrap_template_bank(
        *load_xy("synthetic_reference_peak.npz", "frequency", "amplitude"),
        n_boot=300,
        block_size="auto",
        random_seed=12345,
        name="reference_peak",
    )
    boot2 = bootstrap_fit(fit2, n_boot=1000, block_size="auto", rng=np.random.default_rng(12345))
    tplprop2 = propagate_template_uncertainty_to_target_fit(fit2, bank["templates"], n_draws=None, random_seed=12345)

    # Most replicates should converge (a functional break -- e.g. the
    # resampling/refit plumbing silently returning nothing -- would show up
    # as a valid count near zero, not as a percentage-level swing).
    assert len(boot2["params"]) > 500
    assert len(tplprop2["params"]) > 150

    boot_delta = summarize_mu_delta_samples(boot2, 1, 0)
    assert boot_delta is not None and boot_delta["n"] == len(boot2["params"])
    assert all(np.isfinite(boot_delta[k]) for k in ("median", "minus", "plus", "q16", "q84"))

    tpl_delta = summarize_mu_delta_samples(tplprop2, 1, 0)
    assert tpl_delta is not None and tpl_delta["n"] == len(tplprop2["params"])
    assert all(np.isfinite(tpl_delta[k]) for k in ("median", "minus", "plus", "q16", "q84"))
