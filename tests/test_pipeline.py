"""Tests for ssa.pipeline: the headless build-template/fit/uncertainty
orchestration shared by the GUI worker (conceptually) and the `ssa run` CLI.

These mirror tests/test_regression.py's scenarios but go through the
pipeline/config-object API rather than calling ssa.templates/fitting/
uncertainty functions directly, and additionally check the JSON-summary
builder (`build_run_summary`) used by `ssa run`.
"""

from __future__ import annotations

from conftest import assert_close, load_xy
from ssa.config import FitConfig, TemplateBuildConfig, UncertaintyConfig
from ssa.pipeline import KNOWN_LIMITATIONS, build_run_summary, build_template_bank, run_fit_with_uncertainty


def test_build_template_bank_matches_direct_call(baseline):
    f_ref, y_ref = load_xy("synthetic_reference_peak.npz", "frequency", "amplitude")
    bank = build_template_bank(f_ref, y_ref, TemplateBuildConfig())
    expected = baseline["template_bootstrap_bank"]
    assert bank["summary"]["n_templates"] == expected["n_templates"]
    assert bank["residual_model"]["block_size"] == expected["block_size"]
    assert_close(bank["summary"]["raw_cog"], expected["summary_raw_cog"])


def test_run_fit_with_uncertainty_matches_direct_call(baseline):
    f_ref, y_ref = load_xy("synthetic_reference_peak.npz", "frequency", "amplitude")
    bank = build_template_bank(f_ref, y_ref, TemplateBuildConfig())

    f_single, y_single = load_xy("synthetic_single_peak.npz", "frequency", "amplitude")
    unc_config = UncertaintyConfig(n_boot=1000, random_seed=12345)
    result = run_fit_with_uncertainty(
        f_single, y_single, bank["template_nominal"], bank["templates"], FitConfig(), unc_config
    )

    expected = baseline["fit_single_peak"]
    assert result["fit"]["success"] == expected["success"]
    assert_close(result["fit"]["params"].tolist(), expected["params"])

    expected_boot = baseline["fit_single_peak_bootstrap"]
    assert len(result["bootstrap"]["params"]) == expected_boot["n_valid"]

    expected_tpl = baseline["fit_single_peak_template_propagation"]
    assert len(result["template_propagation"]["params"]) == expected_tpl["n_valid"]


def test_run_fit_with_uncertainty_respects_disabled_sources():
    f_ref, y_ref = load_xy("synthetic_reference_peak.npz", "frequency", "amplitude")
    bank = build_template_bank(f_ref, y_ref, TemplateBuildConfig(n_template_boot=20))
    f_single, y_single = load_xy("synthetic_single_peak.npz", "frequency", "amplitude")
    unc_config = UncertaintyConfig(run_bootstrap=False, run_template_propagation=False)
    result = run_fit_with_uncertainty(
        f_single, y_single, bank["template_nominal"], bank["templates"], FitConfig(), unc_config
    )
    assert result["bootstrap"] is None
    assert result["template_propagation"] is None


def test_build_run_summary_schema_and_honesty():
    """The JSON summary must not overclaim: no conservative-floor field, and
    known_limitations must be present and non-empty (see
    ssa.pipeline.KNOWN_LIMITATIONS)."""
    f_ref, y_ref = load_xy("synthetic_reference_peak.npz", "frequency", "amplitude")
    template_config = TemplateBuildConfig(n_template_boot=20)
    bank = build_template_bank(f_ref, y_ref, template_config)

    f_single, y_single = load_xy("synthetic_single_peak.npz", "frequency", "amplitude")
    fit_config = FitConfig()
    unc_config = UncertaintyConfig(n_boot=50, n_template_prop=20)
    result = run_fit_with_uncertainty(
        f_single, y_single, bank["template_nominal"], bank["templates"], fit_config, unc_config
    )

    summary = build_run_summary(
        label="unit_test",
        reference_source={"path": "ref.csv", "sha256": "deadbeef", "range_hz": [None, None]},
        target_source={"path": "target.csv", "sha256": "deadbeef", "range_hz": [None, None]},
        template_config=template_config,
        fit_config=fit_config,
        unc_config=unc_config,
        template_bank_result=bank,
        analysis_result=result,
        sigma_source="robust_second_difference",
    )

    assert summary["label"] == "unit_test"
    assert summary["ssa_version"]
    assert summary["quality_flag"]
    assert len(summary["components"]) == fit_config.n_peaks
    assert summary["known_limitations"] == list(KNOWN_LIMITATIONS)
    assert "conservative" not in str(summary["fit"]).lower()  # no un-implemented claim
    assert summary["uncertainty"]["bootstrap"]["valid"] > 0
    assert summary["uncertainty"]["template_propagation"]["valid"] > 0
    # No mcmc/aic/bic/dw fields anywhere (removed elsewhere in the package).
    flat_keys = str(summary).lower()
    for forbidden in ("mcmc", "'aic'", "'bic'", "durbin"):
        assert forbidden not in flat_keys


def test_build_run_summary_handles_missing_uncertainty_sources():
    """Both sources deliberately disabled: build_run_summary must not crash,
    and quality_flag_for_fit must NOT raise mu_combined_sigma_single_source
    here -- "neither source was requested" and "not checked" are
    indistinguishable by design (see docs/SCIENTIFIC_METHOD.md section 6),
    so no spurious flag is raised when both are None."""
    f_ref, y_ref = load_xy("synthetic_reference_peak.npz", "frequency", "amplitude")
    template_config = TemplateBuildConfig(n_template_boot=20)
    bank = build_template_bank(f_ref, y_ref, template_config)
    f_single, y_single = load_xy("synthetic_single_peak.npz", "frequency", "amplitude")
    fit_config = FitConfig()
    unc_config = UncertaintyConfig(run_bootstrap=False, run_template_propagation=False)
    result = run_fit_with_uncertainty(
        f_single, y_single, bank["template_nominal"], bank["templates"], fit_config, unc_config
    )
    summary = build_run_summary(
        label="no_unc",
        reference_source={"path": "ref.csv", "sha256": "x", "range_hz": [None, None]},
        target_source={"path": "target.csv", "sha256": "x", "range_hz": [None, None]},
        template_config=template_config,
        fit_config=fit_config,
        unc_config=unc_config,
        template_bank_result=bank,
        analysis_result=result,
        sigma_source="robust_second_difference",
    )
    assert summary["uncertainty"]["bootstrap"] is None
    assert summary["uncertainty"]["template_propagation"] is None
    assert summary["quality_flag"] == "ok"
