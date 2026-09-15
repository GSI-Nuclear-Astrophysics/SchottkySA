"""CLI entry point and headless GUI construction smoke test.

Requires QT_QPA_PLATFORM=offscreen (set in tests/conftest.py) so this runs
without a real display -- MainWindow() is constructed but never shown/exec'd.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import DATA_DIR, load_xy, write_table
from ssa import __version__


def test_version_flag_prints_version_and_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "ssa.cli", "--version"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert __version__ in result.stdout


def test_help_flag_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "ssa.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "ssa" in result.stdout.lower()


def test_main_window_constructs_headless():
    pyside6 = pytest.importorskip("PySide6")
    del pyside6
    from pyqtgraph.Qt import QtWidgets

    from ssa.gui.main_window import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None
    win = MainWindow()
    try:
        assert win.windowTitle().startswith("SchottkySA")
        assert win.n_peaks_spin.value() == 1
        assert win.background_order_spin.value() == 0
        assert win.allow_scale_check.isChecked() is True
        assert win.smooth_check.isChecked() is False
        assert win.results_table.columnCount() == 13
        # MCMC controls must not exist -- Bayesian MCMC is not part of ssa.
        assert not hasattr(win, "run_mcmc_check")
    finally:
        win.close()


# ------------------------------- ssa run (headless) ------------------------------


@pytest.fixture()
def reference_and_target_tables(tmp_path):
    f_ref, y_ref = load_xy("synthetic_reference_peak.npz", "frequency", "amplitude")
    f_tgt, y_tgt = load_xy("synthetic_single_peak.npz", "frequency", "amplitude")
    ref_path = write_table(tmp_path / "reference.csv", f_ref, y_ref, delimiter=",")
    tgt_path = write_table(tmp_path / "target.tsv", f_tgt, y_tgt, delimiter="\t")
    return ref_path, tgt_path


def test_run_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "ssa.cli", "run", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--reference" in result.stdout


def test_run_end_to_end_matches_baseline(tmp_path, reference_and_target_tables, baseline):
    """The CLI must reproduce the exact same single-peak-fit result as the
    Python API against the frozen baseline (tests/test_regression.py's
    test_single_peak_fit) -- proves the CLI plumbing doesn't silently change
    any default."""
    ref_path, tgt_path = reference_and_target_tables
    out_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ssa.cli",
            "run",
            "--reference",
            str(ref_path),
            "--target",
            str(tgt_path),
            "--label",
            "cli_test",
            "--out-dir",
            str(out_dir),
            "--n-boot",
            "1000",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr

    summary_path = out_dir / "cli_test_ssa_summary.json"
    curve_path = out_dir / "cli_test_ssa_curve.csv"
    assert summary_path.exists()
    assert curve_path.exists()

    summary = json.loads(summary_path.read_text())
    expected = baseline["fit_single_peak"]
    assert summary["fit"]["success"] == expected["success"]
    assert summary["fit"]["red_chi2"] == pytest.approx(expected["red_chi2"], rel=1e-8)
    assert summary["quality_flag"] == "ok"
    assert len(summary["components"]) == 1
    assert summary["components"][0]["mu_fit_Hz"] == pytest.approx(expected["params"][0], rel=1e-8)
    assert "known_limitations" in summary and len(summary["known_limitations"]) >= 1

    curve_lines = curve_path.read_text().splitlines()
    assert curve_lines[0].split(",")[:4] == ["frequency_hz", "y", "y_model", "background"]
    assert len(curve_lines) == len(load_xy("synthetic_single_peak.npz", "frequency", "amplitude")[0]) + 1


def test_run_missing_required_argument_fails_with_usage_error():
    result = subprocess.run(
        [sys.executable, "-m", "ssa.cli", "run", "--target", "does_not_matter.csv"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "--reference" in result.stderr


def test_run_reports_error_for_missing_file(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ssa.cli",
            "run",
            "--reference",
            str(tmp_path / "does_not_exist.csv"),
            "--target",
            str(tmp_path / "also_missing.csv"),
            "--out-dir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0


# ------------------------------- ssa replay (session verification) ----------------------------


def _write_replay_session_fixture(tmp_path: Path) -> Path:
    npz_path = tmp_path / "synthetic_reference_peak.npz"
    shutil.copy(DATA_DIR / "synthetic_reference_peak.npz", npz_path)

    from ssa.config import FitConfig, TemplateBuildConfig, UncertaintyConfig
    from ssa.io import load_spectrum_npz, make_export_rows_for_result, select_frequency_range
    from ssa.pipeline import build_template_bank, run_fit_with_uncertainty
    from ssa.session import FitRecord, FormState, InputRef, Session, TemplateRecord, _sha256_of, save_session

    f, y, f_key, y_key = load_spectrum_npz(npz_path)
    region = (float(f.min()), float(f.max()))
    template_config = TemplateBuildConfig(name="reference_peak", n_template_boot=50)
    fx, fy, _ = select_frequency_range(f, y, None, *region)
    bank = build_template_bank(fx, fy, template_config)
    fit_config = FitConfig(n_peaks=1)
    unc_config = UncertaintyConfig(n_boot=50, n_template_prop=50)
    result = run_fit_with_uncertainty(fx, fy, bank["template_nominal"], bank["templates"], fit_config, unc_config)
    tpl = bank["template_nominal"]
    q16, q50, q84 = tpl.quantile([0.16, 0.50, 0.84])
    template_entry = {
        "template": tpl,
        "region_lo": region[0],
        "region_hi": region[1],
        "template_bank": bank["templates"],
        "settings": {
            "smooth_nominal_template": template_config.smooth,
            "resample_factor": template_config.resample_factor,
        },
    }
    rows = make_export_rows_for_result(
        result,
        label="fit_1",
        template_registry=[template_entry],
        notes="",
    )
    session = Session(
        schema_version=1,
        ssa_version="0.1.0",
        created_utc="2026-09-15T12:00:00+00:00",
        input=InputRef(path=str(npz_path), sha256=_sha256_of(npz_path), frequency_key=f_key, amplitude_key=y_key),
        form_state=FormState(
            template=template_config,
            fit=fit_config,
            fit_label="fit_1",
            auto_store=True,
            uncertainty=unc_config,
            region_hz=region,
        ),
        templates=[
            TemplateRecord(
                id=0,
                region_hz=region,
                config=template_config,
                verify={
                    "raw_area": float(tpl.raw_area),
                    "raw_cog": float(tpl.raw_cog),
                    "raw_std": float(tpl.raw_std),
                    "q16": float(q16),
                    "q50": float(q50),
                    "q84": float(q84),
                    "n_template_bank": len(bank["templates"]),
                },
            )
        ],
        fits=[
            FitRecord(
                label="fit_1",
                notes="",
                template_id=0,
                region_hz=region,
                fit_config=fit_config,
                uncertainty_config=unc_config,
                verify_rows=rows,
            )
        ],
    )
    session_path = tmp_path / "session.toml"
    save_session(session_path, session)
    return session_path


def test_replay_command_succeeds_on_matching_session(tmp_path):
    session_path = _write_replay_session_fixture(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "ssa.cli", "replay", str(session_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_replay_command_fails_on_tampered_session(tmp_path):
    from ssa.session import load_session, save_session

    session_path = _write_replay_session_fixture(tmp_path)
    session = load_session(session_path)
    session.templates[0].verify["raw_area"] = session.templates[0].verify["raw_area"] + 1000.0
    save_session(session_path, session)
    result = subprocess.run(
        [sys.executable, "-m", "ssa.cli", "replay", str(session_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1
