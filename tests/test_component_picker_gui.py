"""Graphical fitting controls; all spectra in these tests are artificial."""

from types import SimpleNamespace

import pytest

from ssa.session import load_session, replay_session
from test_profile_workbench_gui import await_worker
from test_profile_workbench_gui import workbench as workbench


def test_centres_update_actual_fit_parameters_and_draggable_bounds(workbench):
    win, app, errors = workbench
    win.picker_halfwidth_spin.setValue(0.2)
    win.add_graphical_component(1260.4)
    win.add_graphical_component(1259.4)
    assert win.n_peaks_spin.value() == 2
    assert win.parse_init_mus() == [1259.4, 1260.4]
    bounds = win.parse_mu_bounds_for_fit(2, win.parse_init_mus())
    assert bounds[0] == pytest.approx((1259.2, 1259.6))
    assert bounds[1] == pytest.approx((1260.2, 1260.6))
    first = win.component_markers[0][0]
    first["line"].setValue(1259.5)
    assert win.parse_init_mus()[0] == 1259.5
    assert first["bounds"] == pytest.approx((1259.3, 1259.7))
    first["region"].setRegion((1259.35, 1259.55))
    assert win.parse_mu_bounds_for_fit(2, win.parse_init_mus())[0] == (1259.35, 1259.55)
    assert "ordered" in win.picker_info_label.text()
    assert not errors


def test_range_shrinking_keeps_starting_cog_inside_bounds(workbench):
    win, app, errors = workbench
    win.picker_halfwidth_spin.setValue(0.2)
    win.add_graphical_component(1260)
    marker = win.component_markers[0][0]
    marker["region"].setRegion((1260.05, 1260.2))
    assert marker["line"].value() == 1260.05
    assert win.parse_init_mus() == [1260.05]
    marker["line"].setValue(1269)
    assert marker["bounds"][1] <= win.selected_region_bounds()[1]
    assert marker["bounds"][1] - marker["bounds"][0] == pytest.approx(0.15)


def test_overlapping_ranges_and_min_separation_are_visible(workbench):
    win, app, errors = workbench
    win.picker_halfwidth_spin.setValue(0.3)
    win.add_graphical_component(1259.8)
    win.add_graphical_component(1260.2)
    assert "non-overlapping" in win.picker_info_label.text()
    win.component_markers[0][0]["region"].setRegion((1259.6, 1259.9))
    win.component_markers[0][1]["region"].setRegion((1260.1, 1260.4))
    assert "ordered" in win.picker_info_label.text()
    win.min_sep_spin.setValue(0.3)
    assert "minimum separation" in win.picker_info_label.text()
    win.min_sep_spin.setValue(0.1)
    assert "ordered" in win.picker_info_label.text()


def test_each_spectrum_view_retains_its_own_fit_centres(workbench):
    win, app, errors = workbench
    win.picker_halfwidth_spin.setValue(0.2)
    win.add_graphical_component(1260)
    win.active_view_combo.setCurrentIndex(1)
    assert win.parse_init_mus() is None
    win.add_graphical_component(1270)
    assert win.parse_init_mus() == [1270]
    win.active_view_combo.setCurrentIndex(0)
    assert win.parse_init_mus() == [1260]
    win.component_markers[1][0]["line"].setValue(1270.1)
    assert win.active_view_index == 1
    assert win.parse_init_mus() == [1270.1]
    assert len(win.component_markers[0]) == len(win.component_markers[1]) == 1


def test_manual_numerical_fields_redraw_without_changing_user_inputs(workbench):
    win, app, errors = workbench
    win.n_peaks_spin.setValue(2)
    win.init_mus_edit.setText("1259.4,1260.4")
    win.mu_ranges_edit.setText("1259.2:1259.6;1260.2:1260.6")
    win.draw_components_btn.click()
    assert len(win.component_markers[0]) == 2
    assert win.init_mus_edit.text() == "1259.4,1260.4"
    win.component_markers[0][0]["line"].setValue(1259.45)
    assert win.parse_init_mus()[0] == 1259.45
    win.picker_component_combo.setCurrentIndex(1)
    win.remove_component_btn.click()
    assert win.n_peaks_spin.value() == 1
    assert len(win.component_markers[0]) == 1
    win.clear_components_btn.click()
    assert win.parse_init_mus() is None
    assert win.mu_ranges_edit.text() == ""
    assert win.n_peaks_spin.value() == 1


def test_double_left_click_adds_centre_single_and_right_do_not(workbench):
    from pyqtgraph.Qt import QtCore

    win, app, errors = workbench
    win.resize(1400, 900)
    win.show()
    win.plot.setXRange(1257, 1263, padding=0)
    win.plot.setYRange(0, 25, padding=0)
    app.processEvents()
    win.pick_components_check.setChecked(True)
    pos = win.plot.getViewBox().mapViewToScene(QtCore.QPointF(1260, 10))
    accepted = []

    def event(double, button):
        return SimpleNamespace(
            button=lambda: button, double=lambda: double, scenePos=lambda: pos, accept=lambda: accepted.append(True)
        )

    win._component_scene_click(0, event(False, QtCore.Qt.MouseButton.LeftButton))
    win._component_scene_click(0, event(True, QtCore.Qt.MouseButton.RightButton))
    assert len(win.component_markers[0]) == 0
    win._component_scene_click(0, event(True, QtCore.Qt.MouseButton.LeftButton))
    assert win.parse_init_mus() == pytest.approx([1260])
    assert accepted == [True]
    # Another double click on the same centre selects it rather than duplicating it.
    win._component_scene_click(0, event(True, QtCore.Qt.MouseButton.LeftButton))
    assert len(win.component_markers[0]) == 1


def test_real_mouse_double_click_on_spectrum(workbench):
    from pyqtgraph.Qt import QtCore
    from PySide6.QtTest import QTest

    win, app, errors = workbench
    win.resize(1400, 900)
    win.show()
    win.plot.setXRange(1257, 1263, padding=0)
    win.plot.setYRange(0, 25, padding=0)
    win.pick_components_check.setChecked(True)
    app.processEvents()
    point = win.plot.mapFromScene(win.plot.getViewBox().mapViewToScene(QtCore.QPointF(1260, 10)))
    QTest.mouseDClick(win.plot.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=point)
    QTest.mouseRelease(win.plot.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=point)
    app.processEvents()
    assert len(win.component_markers[0]) == 1
    assert win.parse_init_mus()[0] == pytest.approx(1260, abs=0.02)  # integer pixel location


def test_fit_and_replay_use_graphical_bounds_and_precise_harmonics(workbench, tmp_path):
    win, app, errors = workbench
    win.view_harmonic_spins[0].setValue(126.123456789)
    win.combine_harmonic_spin.setValue(214.987654321)
    win.picker_halfwidth_spin.setValue(0.2)
    win.add_graphical_component(1260)
    win.active_view_combo.setCurrentIndex(1)
    win.picker_halfwidth_spin.setValue(0.3)
    win.add_graphical_component(1270)
    win.template_combo.setCurrentIndex(1)
    win.allow_scale_check.setChecked(False)
    win.pick_components_check.setChecked(True)
    win.start_fit_region()
    await_worker(win, app, errors)
    cfg = win.fit_records[-1]["fit_config"]
    assert cfg.init_mus == [1270]
    assert cfg.mu_bounds[0] == pytest.approx((1269.7, 1270.3))
    win.output_parent_edit.setText(str(tmp_path / "outputs"))
    win.save_config_btn.click()
    assert not errors
    saved = load_session(win._bundle_dir / "config.toml")
    report = replay_session(saved)
    assert report.ok, report.mismatches
    win._apply_replay_report(report, saved)
    assert win.pick_components_check.isChecked()
    assert win.view_harmonic_spins[0].value() == 126.123456789
    assert win.combine_harmonic_spin.value() == 214.987654321
    assert win.active_view_index == 1
    assert win.parse_init_mus() == [1270]
    assert len(win.component_markers[0]) == len(win.component_markers[1]) == 1
    win.active_view_combo.setCurrentIndex(0)
    assert win.parse_init_mus() == [1260]
    assert win.parse_mu_bounds_for_fit(1, win.parse_init_mus())[0] == pytest.approx((1259.8, 1260.2))


def test_empty_or_invalid_manual_fields_leave_no_stale_markers(workbench):
    win, app, errors = workbench
    win.add_graphical_component(1260)
    win.init_mus_edit.setText("nan")
    win.init_mus_edit.editingFinished.emit()
    assert not win.component_markers[0]
    assert "finite" in win.picker_info_label.text()
    win.init_mus_edit.clear()
    win.init_mus_edit.editingFinished.emit()
    assert not win.component_markers[0]


def test_two_component_fit_obeys_graphically_selected_windows(workbench):
    import numpy as np

    from ssa.profile_tools import transfer_template_entry

    win, app, errors = workbench
    win.active_view_combo.setCurrentIndex(1)
    tpl = transfer_template_entry(win.templates[0], 127)["template"]
    mask = win.frequency > 1265
    x = win.frequency[mask]
    y = tpl.evaluate_component(x, 1269.4, 10) + tpl.evaluate_component(x, 1270.6, 8)
    y += 0.01 + 0.004 * np.sin(np.arange(len(x)) * 1.7)
    win.amplitude[mask] = y
    win.secondary_data_curve.setData(win.frequency, win.amplitude)
    win.template_combo.setCurrentIndex(0)
    win.picker_halfwidth_spin.setValue(0.15)
    win.add_graphical_component(1269.45)
    win.add_graphical_component(1270.55)
    win.min_sep_spin.setValue(0.5)
    win.allow_scale_check.setChecked(False)
    win.start_fit_region()
    assert not win.component_markers[1][0]["line"].movable
    await_worker(win, app, errors)
    cfg = win.last_fit_config
    assert cfg.n_peaks == 2
    assert cfg.init_mus == [1269.45, 1270.55]
    assert win.last_result["fit"]["info"]["mus"] == pytest.approx([1269.4, 1270.6], abs=0.002)
    for mu, (lo, hi) in zip(win.last_result["fit"]["info"]["mus"], cfg.mu_bounds, strict=True):
        assert lo <= mu <= hi
    assert win.component_markers[1][0]["line"].movable


def test_adding_to_unbounded_centres_creates_real_ranges_for_every_component(workbench):
    win, app, errors = workbench
    win.init_mus_edit.setText("1259.4")
    win.draw_components_btn.click()
    assert win.component_markers[0][0]["bounds"] is None
    win.picker_halfwidth_spin.setValue(0.2)
    win.add_graphical_component(1260.4)
    assert win.n_peaks_spin.value() == 2
    assert all(marker["bounds"] is not None for marker in win.component_markers[0])
    assert len(win.parse_mu_bounds_for_fit(2, win.parse_init_mus())) == 2
