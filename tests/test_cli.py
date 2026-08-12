"""CLI entry point and headless GUI construction smoke test.

Requires QT_QPA_PLATFORM=offscreen (set in tests/conftest.py) so this runs
without a real display -- MainWindow() is constructed but never shown/exec'd.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from conftest import load_xy, write_table
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
