"""Offscreen workbench interactions using invented profiles only."""

import numpy as np
import pytest

from ssa.session import _sha256_of, load_session, replay_session
from test_profile_tools import measured_regions


@pytest.fixture
def workbench(tmp_path):
    pytest.importorskip("PySide6")
    from pyqtgraph.Qt import QtWidgets

    from ssa.gui.main_window import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    f, y, entries = measured_regions()
    win.frequency, win.amplitude = f, y
    path = tmp_path / "artificial.npz"
    np.savez(path, frequency=f, amplitude=y)
    win.current_file = path
    win._data_sha256 = _sha256_of(path)
    win._frequency_key, win._amplitude_key = "frequency", "amplitude"
    win.data_curve.setData(f, y)
    win.secondary_data_curve.setData(f, y)
    win.templates = entries
    for e in entries:
        win.template_combo.addItem(e["name"])
    win.refresh_profile_sources()
    win.view_harmonic_spins[0].setValue(126)
    win.view_harmonic_spins[1].setValue(127)
    win.region.setRegion((entries[0]["region_lo"], entries[0]["region_hi"]))
    win.secondary_region.setRegion((entries[1]["region_lo"], entries[1]["region_hi"]))
    win.combine_harmonic_spin.setValue(126)
    win.combine_draws_spin.setValue(8)
    win.template_boot_spin.setValue(8)
    win.n_boot_spin.setValue(8)
    win.tpl_prop_spin.setValue(8)
    errors = []
    win.show_error = errors.append
    yield win, app, errors
    if win.worker is not None:
        win.worker.wait(10000)
    win.close()


def await_worker(win, app, errors):
    assert win.worker is not None
    assert win.worker.wait(10000)
    app.processEvents()
    assert not errors
    assert win.fit_btn.isEnabled()


def select_both(win):
    from pyqtgraph.Qt import QtCore

    for i in range(2):
        win.combine_sources_table.item(i, 0).setCheckState(QtCore.Qt.CheckState.Checked)


def test_navigation_and_independent_selection(workbench):
    win, app, errors = workbench
    assert [win.control_tabs.tabText(i) for i in range(5)] == [
        "Views",
        "Templates",
        "Compare / combine",
        "Fit",
        "Results",
    ]
    original = win.region.getRegion()
    win.active_view_combo.setCurrentIndex(1)
    win.goto_frequency_edit.setText("1270")
    win.view_span_spin.setValue(6)
    win.goto_frequency_btn.click()
    app.processEvents()
    assert win.active_view_index == 1
    assert win.region.getRegion() == original
    assert win.secondary_plot.viewRange()[0] == pytest.approx((1267, 1273))
    assert win.selected_region_bounds() == pytest.approx((1268.5, 1271.5))
    win.roi_lo_edit.setText("1267")
    win.roi_hi_edit.setText("1273")
    win.apply_roi_btn.click()
    assert win.selected_region_bounds() == (1267, 1273)
    win.other_harmonic_btn.click()
    assert win.active_view_index == 0
    assert win.region.getRegion() == pytest.approx((1267 * 126 / 127, 1273 * 126 / 127))
    assert not errors


def test_template_build_captures_harmonic_and_region_before_worker(workbench):
    win, app, errors = workbench
    win.active_view_combo.setCurrentIndex(1)
    bounds = win.selected_region_bounds()
    win.template_name_edit.setText("from_B")
    win.start_build_template()
    # Moving the ROI while a job runs cannot change the saved input window.
    win.secondary_region.setRegion((1269, 1271))
    win.view_harmonic_spins[1].setValue(128)
    await_worker(win, app, errors)
    entry = win.templates[-1]
    assert entry["harmonic"] == 127
    assert (entry["region_lo"], entry["region_hi"]) == bounds
    assert win.combine_sources_table.rowCount() == 3


def test_cross_harmonic_regular_fit_uses_b_overlay_and_records_units(workbench):
    win, app, errors = workbench
    win.active_view_combo.setCurrentIndex(1)
    win.template_combo.setCurrentIndex(0)
    win.allow_scale_check.setChecked(False)
    win.init_mus_edit.setText("1270")
    win.start_fit_region()
    await_worker(win, app, errors)
    assert win.last_target_harmonic == 127
    assert win.last_view_index == 1
    assert abs(win.last_result["fit"]["info"]["mus"][0] - 1270) < 0.002
    assert len(win.secondary_fit_curve.getData()[0]) > 0
    assert win.fit_curve.getData()[0] is None
    row = win.fit_records[-1]["rows"][0]
    assert row["template_harmonic"] == 126
    assert row["target_harmonic"] == 127
    assert row["harmonic_scale_factor"] == pytest.approx(127 / 126)
    assert win.control_tabs.currentIndex() == 4


def test_combine_compare_save_and_replay(workbench, tmp_path):
    win, app, errors = workbench
    select_both(win)
    win.combine_sources_table.item(1, 3).setText("2")
    win.combined_name_edit.setText("two_harmonics")
    win.combine_profiles_btn.click()
    await_worker(win, app, errors)
    assert len(win.templates) == 3
    assert win.template_combo.currentIndex() == 2
    composite = win.templates[-1]
    assert composite["combination"]["weights"] == [1, 2]
    assert composite["combination"]["bank_mode"] == "source_sensitivity"
    assert composite["source_leaf_ids"] == [0, 1]
    assert len(win.profiles_plot.listDataItems()) == 3
    win.compare_a_combo.setCurrentIndex(0)
    win.compare_b_combo.setCurrentIndex(1)
    win.compare_profiles_btn.click()
    await_worker(win, app, errors)
    assert len(win.comparison_runs) == 1
    assert "Both cross-fits completed" in win.statusBar().currentMessage()
    assert win.plot_tabs.currentIndex() == 3
    assert len(win.comparison_plot_sets[0][3].getData()[0]) > 0
    win.active_view_combo.setCurrentIndex(1)
    win.output_parent_edit.setText(str(tmp_path / "outputs"))
    win.bundle_label_edit.setText("profiles")
    win.save_config_btn.click()
    assert not errors
    saved = load_session(win._bundle_dir / "config.toml")
    assert saved.schema_version == 3
    assert saved.form_state.view_state["active_view"] == 1
    assert saved.form_state.view_state["source_weights"][:2] == ["1", "2"]
    assert (win._bundle_dir / "profile_comparisons.csv").exists()
    report = replay_session(saved)
    assert report.ok, report.mismatches
    win._apply_replay_report(report, saved)
    assert win.active_harmonic() == 127
    assert win.template_combo.currentIndex() == 2
    assert len(win.comparison_runs) == 1
    assert win.combine_sources_table.item(1, 3).text() == "2"


def test_preview_shows_edge_closure_and_cog_shift_without_bootstrapping(workbench):
    win, app, errors = workbench
    win.region.setRegion((1259.6, 1260.7))
    win.edge_mode_combo.setCurrentText("pchip_extrapolate_to_zero")
    win.edge_width_spin.setValue(0.3)
    win.preview_template_edges()
    assert win.plot_tabs.currentIndex() == 2
    assert "COG shift" in win.edge_preview_label.text()
    assert "outside the measured ROI" in win.edge_preview_label.text()
    x, y = win.edge_after_curve.getData()
    assert x[0] < win.selected_region_bounds()[0]
    assert x[-1] > win.selected_region_bounds()[1]
    assert all(len(c.getData()[0]) > 0 for c in win.edge_tail_curves)
    assert len(win.templates) == 2
    assert win.worker is None
    assert not errors


def test_independent_mode_and_source_settings_restore(workbench):
    win, app, errors = workbench
    select_both(win)
    win.combine_mode_combo.setCurrentIndex(1)
    state = win._view_state_for_session()
    win.combine_mode_combo.setCurrentIndex(0)
    win._restore_profile_view_state(state)
    assert win.combine_mode_combo.currentData() == "independent"
    assert win._selected_profile_sources() == ([0, 1], [1, 1])


def test_hidden_secondary_view_returns_selection_to_a(workbench):
    win, app, errors = workbench
    win.active_view_combo.setCurrentIndex(1)
    assert win.show_secondary_check.isChecked()
    win.show_secondary_check.setChecked(False)
    assert win.active_view_index == 0
    assert win.active_view_combo.currentIndex() == 0


def test_clear_displays_removes_old_comparison_and_edge_data(workbench):
    win, app, errors = workbench
    win.preview_template_edges()
    win.compare_profiles_btn.click()
    await_worker(win, app, errors)
    assert win.comparison_plot_sets[0][1].getData()[0] is not None
    win._clear_profile_displays()
    assert win.comparison_plot_sets[0][1].getData()[0] is None
    assert win.edge_samples_curve.getData()[0] is None
    assert not win.profiles_plot.listDataItems()


def test_weight_controls_visible_in_source_table(workbench):
    win, app, errors = workbench
    win.resize(1400, 900)
    win.control_tabs.setCurrentIndex(2)
    win.show()
    app.processEvents()
    table = win.combine_sources_table
    assert table.horizontalHeaderItem(3).text() == "Weight"
    # The weight column precedes long ROI metadata, so it needs no horizontal scroll.
    assert table.columnViewportPosition(3) + table.columnWidth(3) <= table.viewport().width()


def test_independent_edges_preview_build_save_and_replay(workbench, tmp_path):
    from ssa.session import save_session

    win, app, errors = workbench
    win.region.setRegion((1259.6, 1260.7))
    win.edge_mode_combo.setCurrentText("pchip_extrapolate_to_zero")
    settings = dict(left_edge_points=3, right_edge_points=7, left_tail_width_hz=0.11, right_tail_width_hz=0.37)
    for field, value in settings.items():
        win.edge_side_spins[field].setValue(value)
    win.preview_template_edges()
    x, _ = win.selected_region_xy()
    assert win.edge_zero_lines[0].value() == pytest.approx(x[0] - 0.11)
    assert win.edge_zero_lines[1].value() == pytest.approx(x[-1] + 0.37)
    assert len(win.edge_used_curves[0].getData()[0]) == 3
    assert len(win.edge_used_curves[1].getData()[0]) == 7
    assert "zero at" in win.edge_preview_label.text()
    win.template_name_edit.setText("asymmetric")
    win.start_build_template()
    await_worker(win, app, errors)
    built = win.templates[-1]
    assert built["template"].edge_info["left_zero_frequency_hz"] == pytest.approx(x[0] - 0.11)
    assert all(t.edge_info["right_zero_frequency_hz"] == pytest.approx(x[-1] + 0.37) for t in built["template_bank"])
    for field, value in settings.items():
        assert built["settings"][field] == value
    path = tmp_path / "edge-session.toml"
    save_session(path, win._session_from_state())
    loaded = load_session(path)
    replay = replay_session(loaded)
    assert replay.ok, replay.mismatches
    restored = replay.templates[-1]["template"]
    np.testing.assert_allclose(restored.pdf, built["template"].pdf)
    for widget in win.edge_side_spins.values():
        widget.setValue(0)
    win._apply_replay_report(replay, loaded)
    for field, value in settings.items():
        assert win.edge_side_spins[field].value() == value
