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
    assert main_window.templates[0]["config"] == TemplateBuildConfig(name="reference_peak")


def test_store_current_fit_records_fit_and_uncertainty_config(main_window):
    from pyqtgraph.Qt import QtWidgets

    _load_reference_npz(main_window)
    main_window.start_build_template()
    main_window.worker.wait()
    QtWidgets.QApplication.processEvents()

    main_window.auto_store_check.setChecked(False)
    main_window.start_fit_region()
    main_window.worker.wait()
    QtWidgets.QApplication.processEvents()

    assert main_window.last_result is not None
    main_window.store_current_fit()
    assert len(main_window.fit_records) == 1
    record = main_window.fit_records[0]
    assert record["fit_config"] == FitConfig(n_peaks=1)
    assert record["uncertainty_config"] == UncertaintyConfig()
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
