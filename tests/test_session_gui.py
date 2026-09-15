"""Headless GUI tests for session-related state capture on MainWindow.

Requires QT_QPA_PLATFORM=offscreen (set in tests/conftest.py).
"""

from __future__ import annotations

import pytest

from conftest import DATA_DIR
from ssa.config import FitConfig, TemplateBuildConfig, UncertaintyConfig


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
