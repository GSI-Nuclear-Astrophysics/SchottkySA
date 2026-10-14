"""Graphical initial COGs and hard optimizer bounds for spectrum views."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from ssa.fitting import ensure_mu_bounds_respect_min_separation, normalize_mu_bounds


class ComponentPickerMixin:
    def _install_component_picker(self) -> None:
        self.component_markers: list[list[dict[str, Any]]] = [[], []]
        self._component_syncing = False
        self._component_forms = [self._capture_component_form(), self._empty_component_form()]
        group = QtWidgets.QGroupBox("Graphical component centres and search ranges")
        form = QtWidgets.QFormLayout(group)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self.pick_components_check = QtWidgets.QCheckBox("Double-click spectrum to add a centre")
        self.pick_components_check.setToolTip(
            "Enable picking, then double-click the left mouse button inside the active fit ROI."
        )
        form.addRow(self.pick_components_check)
        self.picker_halfwidth_spin = QtWidgets.QDoubleSpinBox()
        self.picker_halfwidth_spin.setDecimals(9)
        self.picker_halfwidth_spin.setRange(0, 1e9)
        self.picker_halfwidth_spin.setSpecialValueText("auto: 2% of view span")
        self.picker_halfwidth_spin.setToolTip(
            "Half-width for newly added ranges only. Drag the range edges to change existing bounds."
        )
        form.addRow("New range half-width [Hz]", self.picker_halfwidth_spin)
        self.picker_component_combo = QtWidgets.QComboBox()
        form.addRow("Selected component", self.picker_component_combo)
        self.use_component_templates_check = QtWidgets.QCheckBox("Use a different template for each component")
        self.use_component_templates_check.setToolTip(
            "Select C1, C2, ... above, then assign its profile below. Reusing the same template shares its reference-shape uncertainty."
        )
        form.addRow(self.use_component_templates_check)
        self.component_template_combo = QtWidgets.QComboBox()
        form.addRow("Template for selected component", self.component_template_combo)
        self.same_template_btn = QtWidgets.QPushButton("Use active template for all components")
        form.addRow(self.same_template_btn)
        self.component_template_info = QtWidgets.QLabel()
        self.component_template_info.setWordWrap(True)
        form.addRow(self.component_template_info)
        self._component_template_ids = [-1] * self.n_peaks_spin.value()
        self._component_template_syncing = False
        self.use_component_templates_check.toggled.connect(self._refresh_component_controls)
        self.picker_component_combo.currentIndexChanged.connect(self._sync_component_template_choices)
        self.component_template_combo.currentIndexChanged.connect(self._component_template_changed)
        self.same_template_btn.clicked.connect(lambda: self.use_component_templates_check.setChecked(False))
        self.template_combo.model().rowsInserted.connect(self._sync_component_template_choices)
        self.template_combo.currentIndexChanged.connect(self._sync_component_template_choices)

        buttons = QtWidgets.QHBoxLayout()
        self.draw_components_btn = QtWidgets.QPushButton("Draw from numerical fields")
        self.remove_component_btn = QtWidgets.QPushButton("Remove selected")
        buttons.addWidget(self.draw_components_btn)
        buttons.addWidget(self.remove_component_btn)
        form.addRow(buttons)
        self.clear_components_btn = QtWidgets.QPushButton("Clear centres and ranges")
        form.addRow(self.clear_components_btn)
        self.picker_info_label = QtWidgets.QLabel(
            "Drag a centre to move its range; drag the shaded range's boundaries to resize it. These are initial COGs and hard bounds, not measured fit results."
        )
        self.picker_info_label.setWordWrap(True)
        form.addRow(self.picker_info_label)
        self.fit_components_btn = QtWidgets.QPushButton("Fit selected components")
        self.fit_components_btn.clicked.connect(self.start_fit_region)
        form.addRow(self.fit_components_btn)
        self._tab_layouts[3].insertWidget(0, group)
        self.draw_components_btn.clicked.connect(lambda: self._draw_components_from_fields(show_error=True))
        self.remove_component_btn.clicked.connect(self.remove_graphical_component)
        self.clear_components_btn.clicked.connect(self.clear_graphical_components)
        self.init_mus_edit.editingFinished.connect(self._draw_components_from_fields)
        self.mu_ranges_edit.editingFinished.connect(self._draw_components_from_fields)
        self.n_peaks_spin.valueChanged.connect(self._on_component_fields_changed)
        self.mu_range_halfwidth_spin.valueChanged.connect(self._on_component_fields_changed)
        self.min_sep_spin.valueChanged.connect(self._refresh_component_controls)
        self.region.sigRegionChanged.connect(self._refresh_component_controls)
        self.secondary_region.sigRegionChanged.connect(self._refresh_component_controls)
        self.plot.scene().sigMouseClicked.connect(lambda event: self._component_scene_click(0, event))
        self.secondary_plot.scene().sigMouseClicked.connect(lambda event: self._component_scene_click(1, event))
        self._sync_component_template_choices()

    @staticmethod
    def _empty_component_form() -> dict[str, Any]:
        return {
            "n_peaks": 1,
            "init_mus": "",
            "mu_ranges": "",
            "auto_halfwidth": 0.0,
            "use_component_templates": False,
            "component_template_ids": [-1],
        }

    def _capture_component_form(self) -> dict[str, Any]:
        return {
            "n_peaks": self.n_peaks_spin.value(),
            "init_mus": self.init_mus_edit.text(),
            "mu_ranges": self.mu_ranges_edit.text(),
            "auto_halfwidth": self.mu_range_halfwidth_spin.value(),
            "use_component_templates": hasattr(self, "use_component_templates_check")
            and self.use_component_templates_check.isChecked(),
            "component_template_ids": list(getattr(self, "_component_template_ids", [-1] * self.n_peaks_spin.value())),
        }

    def _switch_component_view(self, old_index: int, new_index: int) -> None:
        if self._restoring_views or old_index == new_index:
            return
        self._component_forms[old_index] = self._capture_component_form()
        state = self._component_forms[new_index]
        self._component_syncing = True
        try:
            self.n_peaks_spin.setValue(state["n_peaks"])
            self.init_mus_edit.setText(state["init_mus"])
            self.mu_ranges_edit.setText(state["mu_ranges"])
            self.mu_range_halfwidth_spin.setValue(state["auto_halfwidth"])
            self._restore_component_template_form(state)
        finally:
            self._component_syncing = False

    def _component_scene_click(self, index: int, event: Any) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton or not event.double():
            return
        if not self.pick_components_check.isChecked() or (self.worker is not None and self.worker.isRunning()):
            return
        view = self._view_plot(index).getViewBox()
        if not view.sceneBoundingRect().contains(event.scenePos()):
            return
        self._activate_view(index)
        centre = float(view.mapSceneToView(event.scenePos()).x())
        # Double-clicking an existing centre selects it instead of duplicating it.
        tolerance = abs(view.viewPixelSize()[0]) * 5
        for marker in self.component_markers[index]:
            if abs(marker["mu"] - centre) <= tolerance:
                self._refresh_component_controls(selected=marker)
                return
        try:
            self.add_graphical_component(centre)
            event.accept()
        except ValueError as exc:
            self.status(str(exc))

    def add_graphical_component(self, centre: float) -> None:
        if self.frequency is None or not np.isfinite(centre):
            raise ValueError("Load a spectrum and choose a finite centre first.")
        if len(self.component_markers[self.active_view_index]) >= self.n_peaks_spin.maximum():
            raise ValueError("The maximum number of components is already selected.")
        lo, hi = self.selected_region_bounds()
        if not lo <= centre <= hi or hi <= lo:
            raise ValueError("Choose the centre inside the active fit ROI.")
        halfwidth = self.picker_halfwidth_spin.value()
        if halfwidth == 0:
            halfwidth = abs(np.diff(self._view_plot().viewRange()[0])[0]) * 0.02
        halfwidth = max(halfwidth, abs(np.spacing(centre)) * 4, (hi - lo) * 1e-9)
        bounds = (max(lo, centre - halfwidth), min(hi, centre + halfwidth))
        existing = self.component_markers[self.active_view_index]
        if any(marker["bounds"] is None for marker in existing):
            # Hard bounds must cover every component or none. Do not draw a
            # new shaded range while silently leaving all optimizer bounds off.
            if any(not lo <= marker["mu"] <= hi for marker in existing):
                raise ValueError("Move existing initial COGs inside the fit ROI before adding bounded components.")
            states = [
                (
                    marker["mu"],
                    marker["bounds"] or (max(lo, marker["mu"] - halfwidth), min(hi, marker["mu"] + halfwidth)),
                )
                for marker in existing
            ]
            self._remove_component_markers(self.active_view_index)
            for mu, limits in states:
                self._new_component_marker(self.active_view_index, mu, limits)
        self._new_component_marker(self.active_view_index, centre, bounds)
        self._write_component_fields()
        self.plot_tabs.setCurrentIndex(0)
        self.status("Component added. Drag the centre or its range edges; bounds are used by the optimizer.")

    def _new_component_marker(self, index: int, centre: float, bounds: tuple[float, float] | None) -> dict[str, Any]:
        colour = ("c", "m", "g", "orange", "y", "b")[len(self.component_markers[index]) % 6]
        marker: dict[str, Any] = {"mu": float(centre), "bounds": bounds, "view_index": index, "region": None}
        position = len(self.component_markers[index])
        ids = getattr(self, "_component_template_ids", [])
        marker["template_id"] = ids[position] if position < len(ids) else -1
        plot = self._view_plot(index)
        if bounds is not None:
            brush = pg.mkColor(colour)
            brush.setAlpha(25)
            region = pg.LinearRegionItem(bounds, brush=brush, pen=pg.mkPen(colour, style=QtCore.Qt.PenStyle.DashLine))
            region.setZValue(20)
            plot.addItem(region)
            marker["region"] = region
            region.sigRegionChanged.connect(lambda: self._component_range_moved(marker))
        line = pg.InfiniteLine(
            centre,
            angle=90,
            movable=True,
            pen=pg.mkPen(colour, width=2),
            hoverPen=pg.mkPen("w", width=3),
            label="COG {value:.6f}",
            labelOpts={"position": 0.9, "color": colour},
        )
        line.setZValue(25)
        plot.addItem(line)
        marker["line"] = line
        self.component_markers[index].append(marker)
        line.sigPositionChanged.connect(lambda: self._component_centre_moved(marker))
        line.sigClicked.connect(lambda *_args: self._select_component_marker(marker))
        return marker

    def _select_component_marker(self, marker: dict[str, Any]) -> None:
        self._activate_view(marker["view_index"])
        self._refresh_component_controls(selected=marker)

    def _component_centre_moved(self, marker: dict[str, Any]) -> None:
        if self._component_syncing:
            return
        index = marker["view_index"]
        self._activate_view(index)
        new_mu = float(marker["line"].value())
        lo, hi = sorted(self._view_region(index).getRegion())
        new_mu = float(np.clip(new_mu, lo, hi))
        self._component_syncing = True
        try:
            if marker["bounds"] is not None:
                left, right = marker["bounds"]
                delta = new_mu - marker["mu"]
                # Translate the entire range, preserving its width where the ROI permits.
                left, right = left + delta, right + delta
                if right - left <= hi - lo:
                    correction = max(lo - left, 0) - max(right - hi, 0)
                    left, right, new_mu = left + correction, right + correction, new_mu + correction
                else:
                    left, right = lo, hi
                marker["bounds"] = (left, right)
                marker["region"].setRegion((left, right))
            marker["mu"] = new_mu
            marker["line"].setValue(new_mu)
        finally:
            self._component_syncing = False
        self._write_component_fields(selected=marker)

    def _component_range_moved(self, marker: dict[str, Any]) -> None:
        if self._component_syncing:
            return
        self._activate_view(marker["view_index"])
        left, right = sorted(marker["region"].getRegion())
        self._component_syncing = True
        try:
            marker["bounds"] = (float(left), float(right))
            marker["mu"] = float(np.clip(marker["mu"], left, right))
            marker["line"].setValue(marker["mu"])
        finally:
            self._component_syncing = False
        self._write_component_fields(selected=marker)

    def _write_component_fields(self, selected: dict[str, Any] | None = None) -> None:
        markers = self.component_markers[self.active_view_index]
        markers.sort(key=lambda marker: marker["mu"])
        self._component_template_ids = [m.get("template_id", -1) for m in markers] or [-1]
        self._component_syncing = True
        try:
            self.n_peaks_spin.setValue(max(1, len(markers)))
            self.init_mus_edit.setText(", ".join(format(marker["mu"], ".17g") for marker in markers))
            if markers and all(marker["bounds"] is not None for marker in markers):
                self.mu_ranges_edit.setText(
                    "; ".join(f"{left:.17g}:{right:.17g}" for left, right in (m["bounds"] for m in markers))
                )
            else:
                self.mu_ranges_edit.clear()
            self.mu_range_halfwidth_spin.setValue(0)
            self._component_forms[self.active_view_index] = self._capture_component_form()
        finally:
            self._component_syncing = False
        self._refresh_component_controls(selected=selected)

    def _on_component_fields_changed(self) -> None:
        if not self._component_syncing and not self._restoring_views:
            self._draw_components_from_fields()

    def _draw_components_from_fields(self, show_error: bool = False) -> None:
        if self._component_syncing or self._restoring_views:
            return
        try:
            centres = self.parse_init_mus()
            count = self.n_peaks_spin.value()
            if centres is None:
                self._remove_component_markers(self.active_view_index)
                self._refresh_component_controls()
                return
            if len(centres) != count or not np.all(np.isfinite(centres)):
                raise ValueError("Provide one finite initial COG per requested component.")
            bounds = self.parse_mu_bounds_for_fit(count, centres)
            if bounds is not None and any(
                not np.isfinite(lo + hi) or hi <= lo or not lo <= mu <= hi
                for mu, (lo, hi) in zip(centres, bounds, strict=True)
            ):
                raise ValueError("Each initial COG must lie inside a finite, nonempty range.")
            self._remove_component_markers(self.active_view_index)
            for i, centre in enumerate(centres):
                self._new_component_marker(self.active_view_index, centre, bounds[i] if bounds is not None else None)
            self._component_forms[self.active_view_index] = self._capture_component_form()
            self._refresh_component_controls()
        except ValueError as exc:
            self._remove_component_markers(self.active_view_index)
            self.picker_info_label.setText(str(exc))
            if show_error:
                self.show_error(str(exc))

    def _refresh_component_controls(self, *_args: Any, selected: dict[str, Any] | None = None) -> None:
        if not hasattr(self, "picker_component_combo") or self._restoring_views or self._component_syncing:
            return
        markers = self.component_markers[self.active_view_index]
        old = self.picker_component_combo.currentIndex()
        blocker = QtCore.QSignalBlocker(self.picker_component_combo)
        self.picker_component_combo.clear()
        for i, marker in enumerate(markers):
            marker["line"].label.setFormat(f"C{i + 1}")
            marker["line"].label.setPosition(0.93 - 0.06 * (i % 4))
            bounds = marker["bounds"]
            label = f"C{i + 1}: {marker['mu']:.9f} Hz"
            if bounds is not None:
                label += f" [{bounds[0]:.9f}, {bounds[1]:.9f}]"
            marker["line"].setToolTip("Initial COG and optimizer bounds: " + label)
            if marker["region"] is not None:
                marker["region"].setToolTip("Hard optimizer bounds: " + label)
            self.picker_component_combo.addItem(label)
            self.picker_component_combo.setItemData(i, label, QtCore.Qt.ItemDataRole.ToolTipRole)
        if not markers and self.use_component_templates_check.isChecked():
            for i in range(self.n_peaks_spin.value()):
                self.picker_component_combo.addItem(f"C{i + 1}")
        index = next((i for i, marker in enumerate(markers) if marker is selected), old)
        self.picker_component_combo.setCurrentIndex(max(0, min(index, self.picker_component_combo.count() - 1)))
        del blocker
        self._sync_component_template_choices()
        if not markers:
            self.picker_info_label.setText(
                "Enable picking and double-click inside the fit ROI. The first pick creates C1; each further pick adds a component and an optimizer range."
            )
            return
        try:
            bounds = self.parse_mu_bounds_for_fit(self.n_peaks_spin.value(), self.parse_init_mus())
            lo, hi = self.selected_region_bounds()
            normalized = normalize_mu_bounds(bounds, len(markers), lo, hi)
            ensure_mu_bounds_respect_min_separation(normalized, self.min_sep_spin.value())
            outside = any(not lo <= marker["mu"] <= hi for marker in markers)
            if outside:
                raise ValueError("One or more initial COGs lie outside the active fit ROI.")
            self.picker_info_label.setText(
                f"{len(markers)} component(s). Drag centre lines to translate ranges, or drag the range edges. "
                + (
                    "Hard bounds are active and ordered."
                    if bounds is not None
                    else "No hard bounds are active; use numerical ranges or clear and pick again to add them."
                )
            )
        except ValueError as exc:
            self.picker_info_label.setText(f"Adjust ranges before fitting: {exc}")

    def _remove_component_markers(self, index: int) -> None:
        plot = self._view_plot(index)
        for marker in self.component_markers[index]:
            plot.removeItem(marker["line"])
            if marker["region"] is not None:
                plot.removeItem(marker["region"])
        self.component_markers[index] = []

    def remove_graphical_component(self) -> None:
        markers = self.component_markers[self.active_view_index]
        index = self.picker_component_combo.currentIndex()
        if not 0 <= index < len(markers):
            return
        marker = markers.pop(index)
        plot = self._view_plot()
        plot.removeItem(marker["line"])
        if marker["region"] is not None:
            plot.removeItem(marker["region"])
        self._write_component_fields()

    def clear_graphical_components(self) -> None:
        self._remove_component_markers(self.active_view_index)
        self._write_component_fields()

    def _clear_all_graphical_components(self) -> None:
        for index in (0, 1):
            self._remove_component_markers(index)
        self._component_forms = [self._empty_component_form(), self._empty_component_form()]
        self._write_component_fields()

    def _component_state_for_session(self) -> dict[str, Any]:
        self._component_forms[self.active_view_index] = self._capture_component_form()
        return {
            "enabled": self.pick_components_check.isChecked(),
            "new_halfwidth": self.picker_halfwidth_spin.value(),
            "forms": self._component_forms,
        }

    def _restore_component_state(self, state: dict[str, Any]) -> None:
        for index in (0, 1):
            self._remove_component_markers(index)
        forms = state.get("forms", [self._capture_component_form(), self._empty_component_form()])
        self._component_forms = [dict(form) for form in forms]
        self.pick_components_check.setChecked(state.get("enabled", False))
        self.picker_halfwidth_spin.setValue(state.get("new_halfwidth", 0))
        active = self.active_view_index
        # Numerical form state is canonical. Draw both views without overwriting
        # the active form loaded by replay (including manually entered values).
        self._restore_component_template_form(self._component_forms[active])
        self._component_forms[active] = self._capture_component_form()
        for index in (0, 1):
            self.active_view_index = index
            self._component_syncing = True
            try:
                form = self._component_forms[index]
                self.n_peaks_spin.setValue(form["n_peaks"])
                self.init_mus_edit.setText(form["init_mus"])
                self.mu_ranges_edit.setText(form["mu_ranges"])
                self.mu_range_halfwidth_spin.setValue(form["auto_halfwidth"])
                self._restore_component_template_form(form)
            finally:
                self._component_syncing = False
            self._draw_components_from_fields()
        self.active_view_index = active
        self._component_syncing = True
        try:
            form = self._component_forms[active]
            self.n_peaks_spin.setValue(form["n_peaks"])
            self.init_mus_edit.setText(form["init_mus"])
            self.mu_ranges_edit.setText(form["mu_ranges"])
            self.mu_range_halfwidth_spin.setValue(form["auto_halfwidth"])
            self._restore_component_template_form(form)
        finally:
            self._component_syncing = False
        self._refresh_component_controls()

    def _restore_component_template_form(self, state: dict[str, Any]) -> None:
        self._component_template_ids = list(state.get("component_template_ids", [-1] * self.n_peaks_spin.value()))
        with QtCore.QSignalBlocker(self.use_component_templates_check):
            self.use_component_templates_check.setChecked(bool(state.get("use_component_templates", False)))

    def _sync_component_template_choices(self, *_args: Any) -> None:
        if not hasattr(self, "component_template_combo") or self._component_template_syncing:
            return
        self._component_template_syncing = True
        try:
            n = self.n_peaks_spin.value()
            self._component_template_ids = (self._component_template_ids + [-1] * n)[:n]
            k = self.picker_component_combo.currentIndex()
            assigned = self._component_template_ids[k] if 0 <= k < n else -1
            with QtCore.QSignalBlocker(self.component_template_combo):
                self.component_template_combo.clear()
                self.component_template_combo.addItem("Active template (follow selection)", -1)
                for i, entry in enumerate(self.templates):
                    self.component_template_combo.addItem(f"{i + 1}: {entry['name']}", i)
                if assigned >= len(self.templates):
                    self.component_template_combo.addItem(f"Unavailable template {assigned + 1}", assigned)
                self.component_template_combo.setCurrentIndex(max(0, self.component_template_combo.findData(assigned)))
            enabled = self.use_component_templates_check.isChecked()
            self.component_template_combo.setEnabled(
                enabled and k >= 0 and not (self.worker and self.worker.isRunning())
            )
            if enabled:
                active = self.template_combo.currentIndex()
                names = []
                for i, value in enumerate(self._component_template_ids):
                    idx = active if value < 0 else value
                    name = self.templates[idx]["name"] if 0 <= idx < len(self.templates) else "choose a template"
                    names.append(f"C{i + 1}: {name}")
                self.component_template_info.setText(
                    " | ".join(names)
                    + "\nDifferent profiles require ordered, non-overlapping COG ranges. Common scale shares a fractional stretch; uncheck it for independent stretches."
                )
            else:
                self.component_template_info.setText("All components use the active template.")
        finally:
            self._component_template_syncing = False

    def _component_template_changed(self, *_args: Any) -> None:
        if self._component_template_syncing:
            return
        k = self.picker_component_combo.currentIndex()
        if not 0 <= k < len(self._component_template_ids):
            return
        value = int(self.component_template_combo.currentData())
        self._component_template_ids[k] = value
        markers = self.component_markers[self.active_view_index]
        if k < len(markers):
            markers[k]["template_id"] = value
        self._component_forms[self.active_view_index] = self._capture_component_form()
        self._sync_component_template_choices()

    def _fit_component_template_ids(self, n: int) -> list[int] | None:
        if not self.use_component_templates_check.isChecked():
            return None
        self._sync_component_template_choices()
        active = self.template_combo.currentIndex()
        ids = [active if value < 0 else value for value in self._component_template_ids]
        if len(ids) != n or any(not 0 <= idx < len(self.templates) for idx in ids):
            raise ValueError("Assign a stored template to every component before fitting.")
        return ids
