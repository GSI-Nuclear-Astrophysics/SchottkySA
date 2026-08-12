"""Tests for ssa.uncertainty: quality flags, sample-summary helpers, and the
combination/correlation diagnostics used in fit-history export rows.

The ``boot``/``tpl_prop``-aware checks below (uncertainty-source failure
rate, single-source combined sigma) exist because the
bootstrap/template-propagation spread is only trustworthy if enough
replicates actually converged, and a "combined" sigma resting on a single
source should be visible, not silent.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssa.constants import (
    QUALITY_MIN_UNCERTAINTY_SAMPLE_FRACTION,
    QUALITY_MIN_VALID_UNCERTAINTY_SAMPLES,
    QUALITY_RED_CHI2_BAD,
    QUALITY_RED_CHI2_WARN,
)
from ssa.uncertainty import (
    correlation_from_samples,
    quality_flag_for_fit,
    sample_count,
    summarize_area_ratio_samples,
    summarize_mu_delta_samples,
)


def _base_fit(n_peaks: int = 1, **overrides):
    fit = {
        "n_peaks": n_peaks,
        "red_chi2": 1.0,
        "success": True,
        "min_separation": 0.0,
        "min_separation_satisfied": True,
        "mu_bounds": None,
        "dof": 100,
    }
    fit.update(overrides)
    return fit


def _source(n_requested: int, n_valid: int, mu_sigma: float | None = None) -> dict:
    """A minimal bootstrap/template-propagation-shaped result dict."""
    summary = {}
    if mu_sigma is not None:
        summary["mu_0"] = {
            "median": 0.0,
            "minus": mu_sigma,
            "plus": mu_sigma,
            "q16": -mu_sigma,
            "q84": mu_sigma,
            "n": n_valid,
        }
    return {"params": np.zeros((n_valid, 1)), "n_requested": n_requested, "summary": summary}


def test_quality_flag_ok_for_a_clean_fit():
    assert quality_flag_for_fit(_base_fit()) == "ok"


def test_quality_flag_not_converged():
    assert "fit_not_converged" in quality_flag_for_fit(_base_fit(success=False))


@pytest.mark.parametrize(
    "red_chi2,expected_flag",
    [
        (QUALITY_RED_CHI2_WARN + 0.1, "red_chi2_gt_2"),
        (QUALITY_RED_CHI2_BAD + 0.1, "red_chi2_gt_5"),
    ],
)
def test_quality_flag_red_chi2_thresholds(red_chi2, expected_flag):
    assert expected_flag in quality_flag_for_fit(_base_fit(red_chi2=red_chi2))


def test_quality_flag_min_separation_violated():
    fit = _base_fit(min_separation=5.0, min_separation_satisfied=False)
    assert "min_separation_violated" in quality_flag_for_fit(fit)


def test_quality_flag_nonpositive_dof():
    assert "nonpositive_dof" in quality_flag_for_fit(_base_fit(dof=0))


def test_quality_flag_mu_near_search_bound():
    fit = _base_fit(mu_bounds=[[0.0, 10.0]], info={"mus": [0.05]})  # 0.5% from lower bound
    assert "mu_near_search_bound" in quality_flag_for_fit(fit)


def test_quality_flag_no_uncertainty_checks_when_sources_not_passed():
    """Backward compatible: omitting boot/tpl_prop (the pre-existing
    call signature) must not spuriously raise the new checks."""
    flag = quality_flag_for_fit(_base_fit())
    assert flag == "ok"


def test_quality_flag_bootstrap_not_requested_is_not_a_failure():
    """boot=None means "deliberately not run", not "failed" -- no flag."""
    flag = quality_flag_for_fit(_base_fit(), boot=None, tpl_prop=_source(300, 300, mu_sigma=0.1))
    assert "bootstrap_failed" not in flag
    assert "bootstrap_insufficient_samples" not in flag


def test_quality_flag_bootstrap_failed_outright():
    boot = _source(n_requested=1000, n_valid=0)
    assert "bootstrap_failed" in quality_flag_for_fit(_base_fit(), boot=boot)


@pytest.mark.parametrize(
    "n_requested,n_valid",
    [
        (1000, QUALITY_MIN_VALID_UNCERTAINTY_SAMPLES - 1),  # below absolute floor
        (100, int(100 * QUALITY_MIN_UNCERTAINTY_SAMPLE_FRACTION) - 1),  # below success fraction
    ],
)
def test_quality_flag_bootstrap_insufficient_samples(n_requested, n_valid):
    boot = _source(n_requested=n_requested, n_valid=max(n_valid, 1))
    assert "bootstrap_insufficient_samples" in quality_flag_for_fit(_base_fit(), boot=boot)


def test_quality_flag_bootstrap_healthy_no_flag():
    boot = _source(n_requested=1000, n_valid=950, mu_sigma=0.1)
    flag = quality_flag_for_fit(_base_fit(), boot=boot)
    assert "bootstrap_failed" not in flag
    assert "bootstrap_insufficient_samples" not in flag


def test_quality_flag_template_propagation_failure_uses_its_own_label():
    tpl_prop = _source(n_requested=300, n_valid=0)
    flag = quality_flag_for_fit(_base_fit(), tpl_prop=tpl_prop)
    assert "template_propagation_failed" in flag
    assert "bootstrap_failed" not in flag  # labels must not cross-contaminate


def test_quality_flag_single_source_when_only_bootstrap_available():
    """The check most directly relevant to trusting a reported centroid
    uncertainty: if the combined sigma for mu_0 will, in practice, rest on
    only one source, that must be visible in the flag."""
    boot = _source(n_requested=1000, n_valid=950, mu_sigma=0.1)
    flag = quality_flag_for_fit(_base_fit(), boot=boot, tpl_prop=None)
    assert "mu_combined_sigma_single_source" in flag


def test_quality_flag_single_source_when_the_other_source_failed():
    """Even if template propagation was requested, if it produced no usable
    mu_0 sigma, the combined sigma still rests on one source."""
    boot = _source(n_requested=1000, n_valid=950, mu_sigma=0.1)
    tpl_prop = _source(n_requested=300, n_valid=0)  # requested, but failed
    flag = quality_flag_for_fit(_base_fit(), boot=boot, tpl_prop=tpl_prop)
    assert "mu_combined_sigma_single_source" in flag


def test_quality_flag_no_single_source_flag_when_both_available():
    boot = _source(n_requested=1000, n_valid=950, mu_sigma=0.1)
    tpl_prop = _source(n_requested=300, n_valid=280, mu_sigma=0.2)
    flag = quality_flag_for_fit(_base_fit(), boot=boot, tpl_prop=tpl_prop)
    assert "mu_combined_sigma_single_source" not in flag


def test_quality_flag_single_source_checks_every_component():
    """A two-component fit where only component 1's combined sigma is
    single-sourced must still be flagged."""
    boot = {
        "params": np.zeros((950, 2)),
        "n_requested": 1000,
        "summary": {
            "mu_0": {"median": 0.0, "minus": 0.1, "plus": 0.1, "q16": -0.1, "q84": 0.1, "n": 950},
            "mu_1": {"median": 0.0, "minus": 0.1, "plus": 0.1, "q16": -0.1, "q84": 0.1, "n": 950},
        },
    }
    tpl_prop = {
        "params": np.zeros((280, 2)),
        "n_requested": 300,
        "summary": {
            "mu_0": {"median": 0.0, "minus": 0.2, "plus": 0.2, "q16": -0.2, "q84": 0.2, "n": 280},
            # mu_1 missing from template-propagation summary entirely.
        },
    }
    flag = quality_flag_for_fit(_base_fit(n_peaks=2), boot=boot, tpl_prop=tpl_prop)
    assert "mu_combined_sigma_single_source" in flag


def test_correlation_from_samples_perfect_and_constant():
    x = np.linspace(0, 1, 50)
    assert correlation_from_samples(x, x) == pytest.approx(1.0)
    assert correlation_from_samples(x, -x) == pytest.approx(-1.0)
    assert np.isnan(correlation_from_samples(x, np.ones_like(x)))  # constant -> undefined


def test_sample_count_and_summary_helpers_handle_missing_source():
    assert sample_count(None) == 0
    assert sample_count({"params": np.empty((0, 3))}) == 0
    assert sample_count({"params": np.zeros((7, 3))}) == 7
    assert summarize_mu_delta_samples(None, 1, 0) is None
    assert summarize_area_ratio_samples(None, 1, 0, 2) is None
