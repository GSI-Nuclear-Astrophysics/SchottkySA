"""Headless GUI tests for session-related state capture on MainWindow.

Requires QT_QPA_PLATFORM=offscreen (set in tests/conftest.py).
"""

from __future__ import annotations

import shutil

import pytest

from conftest import DATA_DIR
from ssa.config import FitConfig, TemplateBuildConfig, UncertaintyConfig
from ssa.io import load_spectrum_npz, make_export_rows_for_result, select_frequency_range
from ssa.pipeline import build_template_bank, run_fit_with_uncertainty
from ssa.session import FitRecord, FormState, InputRef, Session, TemplateRecord, _sha256_of


@pytest.fixture
def main_window():
    pyside6 = pytest.importorskip("PySide6")
    del pyside6
    from pyqtgraph.Qt import QtWidgets

    from ssa.gui.main_window import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None
    win = MainWindow()
    yield win
    win.close()


def _load_reference_npz(win) -> None:
    import numpy as np

    from ssa.preprocessing import prepare_xy

    data = np.load(DATA_DIR / "synthetic_reference_peak.npz")
    f, y = prepare_xy(np.asarray(data["frequency"], dtype=float), np.asarray(data["amplitude"], dtype=float))
    win.frequency = f
    win.amplitude = y
    win.data_curve.setData(f, y)
    win.region.setRegion([float(f.min()), float(f.max())])
    # Worker tests use explicit small runs; fresh-window tests check the defaults.
    win.template_boot_spin.setValue(30)
    win.n_boot_spin.setValue(50)
    win.tpl_prop_spin.setValue(30)


def test_template_config_from_widgets_matches_defaults(main_window):
    config = main_window._template_config_from_widgets()
    assert config == TemplateBuildConfig(name="reference_peak")


def test_fit_and_uncertainty_config_from_widgets_match_defaults(main_window):
    fit_config = main_window._fit_config_from_widgets(n_peaks=1, init_mus=None, mu_bounds=None)
    assert fit_config == FitConfig(n_peaks=1)
    unc_config = main_window._uncertainty_config_from_widgets()
    assert unc_config == UncertaintyConfig()


def test_build_template_stores_config_on_entry(main_window):
    from pyqtgraph.Qt import QtWidgets

    _load_reference_npz(main_window)
    main_window.start_build_template()
    main_window.worker.wait()
    QtWidgets.QApplication.processEvents()
    assert len(main_window.templates) == 1
    assert main_window.templates[0]["config"] == TemplateBuildConfig(name="reference_peak", n_template_boot=30)


def test_store_current_fit_records_fit_and_uncertainty_config(main_window):
    from pyqtgraph.Qt import QtWidgets

    _load_reference_npz(main_window)
    main_window.start_build_template()
    main_window.worker.wait()
    QtWidgets.QApplication.processEvents()

    # avoid double-storing: finish_fit_region auto-stores when this is checked (the default)
    main_window.auto_store_check.setChecked(False)
    main_window.start_fit_region()
    main_window.worker.wait()
    QtWidgets.QApplication.processEvents()

    assert main_window.last_result is not None
    main_window.store_current_fit()
    assert len(main_window.fit_records) == 1
    record = main_window.fit_records[0]
    assert record["fit_config"] == FitConfig(n_peaks=1)
    assert record["uncertainty_config"] == UncertaintyConfig(n_boot=50, n_template_prop=30)
    assert record["template_index"] == 0
    lo, hi = record["region_hz"]
    assert lo < hi


def test_session_from_state_round_trips_through_save_and_load(main_window, tmp_path):
    import shutil

    from pyqtgraph.Qt import QtWidgets

    npz_path = tmp_path / "synthetic_reference_peak.npz"
    shutil.copy(DATA_DIR / "synthetic_reference_peak.npz", npz_path)
    main_window.current_file = npz_path
    _load_reference_npz(main_window)

    main_window.start_build_template()
    main_window.worker.wait()
    QtWidgets.QApplication.processEvents()

    # avoid double-storing: finish_fit_region auto-stores when this is checked (the default)
    main_window.auto_store_check.setChecked(False)
    main_window.start_fit_region()
    main_window.worker.wait()
    QtWidgets.QApplication.processEvents()
    main_window.store_current_fit()

    session = main_window._session_from_state()
    assert session.input.path == str(npz_path)
    assert len(session.templates) == 1
    assert len(session.fits) == 1
    assert session.fits[0].template_id == 0

    from ssa.session import load_session, save_session

    path = tmp_path / "session.toml"
    save_session(path, session)
    loaded = load_session(path)
    assert loaded.templates[0].config == session.templates[0].config
    assert loaded.fits[0].fit_config == session.fits[0].fit_config


def _build_replay_fixture(tmp_path):
    npz_path = tmp_path / "synthetic_reference_peak.npz"
    shutil.copy(DATA_DIR / "synthetic_reference_peak.npz", npz_path)

    f, y, f_key, y_key = load_spectrum_npz(npz_path)
    template_config = TemplateBuildConfig(name="reference_peak", n_template_boot=50)
    region = (float(f.min()), float(f.max()))
    fx, fy, _ = select_frequency_range(f, y, None, *region)
    bank = build_template_bank(fx, fy, template_config)

    fit_config = FitConfig(n_peaks=1)
    unc_config = UncertaintyConfig(n_boot=50, n_template_prop=50)
    result = run_fit_with_uncertainty(fx, fy, bank["template_nominal"], bank["templates"], fit_config, unc_config)
    tpl = bank["template_nominal"]
    q16, q50, q84 = tpl.quantile([0.16, 0.50, 0.84])
    template_entry = {
        "name": tpl.name,
        "region_lo": region[0],
        "region_hi": region[1],
        "template": tpl,
        "template_bank": bank["templates"],
        "settings": {
            "smooth_nominal_template": template_config.smooth,
            "resample_factor": template_config.resample_factor,
        },
    }
    rows = make_export_rows_for_result(result, label="fit_1", template_registry=[template_entry], notes="")

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
    return session, npz_path


def test_apply_replay_report_populates_gui_state(main_window, tmp_path):
    session, _ = _build_replay_fixture(tmp_path)
    from ssa.session import replay_session

    report = replay_session(session)
    main_window._apply_replay_report(report, session)

    assert len(main_window.templates) == 1
    assert len(main_window.fit_records) == 1
    assert main_window.template_combo.count() == 1
    assert main_window.history_table.rowCount() == len(main_window.fit_records[0]["rows"])
    assert main_window.fit_label_edit.text() == session.form_state.fit_label
    assert main_window.n_peaks_spin.value() == session.form_state.fit.n_peaks
    assert "Loaded session" in main_window.summary_text.toPlainText() or main_window.status_label.text()


def test_visible_bundle_controls_save_configuration_and_csv(main_window, tmp_path, monkeypatch):
    from pathlib import Path

    from ssa.session import load_session, replay_session

    session, _ = _build_replay_fixture(tmp_path)
    main_window._apply_replay_report(replay_session(session), session)
    main_window.output_parent_edit.setText(str(tmp_path / "outputs"))
    main_window.bundle_label_edit.setText("region 1")
    assert main_window.save_config_btn.text() == "Save config + tables"
    assert "Replay" in main_window.replay_config_btn.text()
    errors = []
    monkeypatch.setattr(main_window, "show_error", errors.append)
    main_window.save_config_btn.click()
    assert not errors
    out = main_window._bundle_dir
    assert isinstance(out, Path)
    assert "_SchottkySA_CONFIG_region_1" in out.name
    assert (out / "config.toml").exists()
    assert (out / "fit_history.csv").exists()
    assert (out / "current_fit_overlay.npz").exists()
    assert not (out / "synthetic_reference_peak.npz").exists()
    saved = load_session(out / "config.toml")
    assert saved.form_state.active_template_id == 0
    assert replay_session(saved).ok
    # Save/export actions for the same state share one folder.
    main_window.table_format_combo.setCurrentText("TSV")
    main_window.export_history()
    assert main_window._bundle_dir == out
    assert (out / "fit_history.tsv").exists()
    assert len(list((tmp_path / "outputs").iterdir())) == 1
    # Different settings get a distinct snapshot instead of overwriting.
    main_window.seed_spin.setValue(main_window.seed_spin.value() + 1)
    main_window.save_session()
    assert main_window._bundle_dir != out
    assert main_window._bundle_dir.name == out.name + "_02"


def test_unstored_fit_is_included_without_mutating_history(main_window, tmp_path):
    from ssa.session import replay_session

    session, _ = _build_replay_fixture(tmp_path)
    main_window._apply_replay_report(replay_session(session), session)
    main_window.fit_records = []
    main_window.auto_store_check.setChecked(False)
    saved = main_window._session_from_state()
    assert len(saved.fits) == 1
    assert len(main_window.fit_records) == 0
    assert replay_session(saved).ok


def test_input_changed_after_loading_cannot_be_saved(main_window, tmp_path):
    from ssa.session import replay_session

    session, path = _build_replay_fixture(tmp_path)
    main_window._apply_replay_report(replay_session(session), session)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="NPZ changed"):
        main_window._session_from_state()


def test_empty_replay_clears_previous_fit(main_window, tmp_path):
    from ssa.session import replay_session

    session, _ = _build_replay_fixture(tmp_path)
    main_window._apply_replay_report(replay_session(session), session)
    session.fits = []
    session.templates = []
    session.form_state.active_template_id = None
    main_window._apply_replay_report(replay_session(session), session)
    assert main_window.last_result is None
    assert main_window.last_fit_config is None


def test_run_directory_names_are_safe_and_unique(tmp_path):
    from ssa.bundles import create_run_directory

    first = create_run_directory(tmp_path, "SchottkySA", "../region/label")
    second = create_run_directory(tmp_path, "SchottkySA", "../region/label")
    assert first.parent == second.parent == tmp_path
    assert second.name == first.name + "_02"


def test_replay_button_runs_saved_configuration(main_window, tmp_path, monkeypatch):
    from pyqtgraph.Qt import QtWidgets

    from ssa.session import save_session

    session, _ = _build_replay_fixture(tmp_path)
    path = tmp_path / "config.toml"
    save_session(path, session)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", lambda *a: (str(path), "TOML"))
    errors = []
    monkeypatch.setattr(main_window, "show_error", errors.append)
    main_window.replay_config_btn.click()
    assert main_window.worker is not None
    assert main_window.worker.wait(30000)
    QtWidgets.QApplication.processEvents()
    assert not errors
    assert len(main_window.fit_records) == 1
    assert len(main_window.templates) == 1
    assert "verified" in main_window.status_label.text()
    assert main_window.save_config_btn.isEnabled()


def test_different_npz_clears_old_fit_session(main_window, tmp_path, monkeypatch):
    import numpy as np
    from pyqtgraph.Qt import QtWidgets

    from ssa.session import replay_session

    session, _ = _build_replay_fixture(tmp_path)
    main_window._apply_replay_report(replay_session(session), session)
    path = tmp_path / "different.npz"
    np.savez(path, frequency=main_window.frequency, amplitude=main_window.amplitude * 2)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", lambda *a: (str(path), "NPZ"))
    main_window.load_npz()
    assert not main_window.fit_records
    assert not main_window.templates
    assert main_window.last_result is None


def test_save_and_replay_controls_visible_without_horizontal_scroll(main_window):
    from pyqtgraph.Qt import QtWidgets

    main_window.resize(1400, 900)
    main_window.show()
    QtWidgets.QApplication.processEvents()
    scroll = main_window.findChild(QtWidgets.QScrollArea)
    assert scroll is not None
    viewport = scroll.viewport()
    for button in (main_window.save_config_btn, main_window.replay_config_btn, main_window.output_browse_btn):
        top_left = button.mapTo(viewport, button.rect().topLeft())
        bottom_right = button.mapTo(viewport, button.rect().bottomRight())
        assert viewport.rect().contains(top_left)
        assert viewport.rect().contains(bottom_right)
