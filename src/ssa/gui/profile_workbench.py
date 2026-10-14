"""Profile workbench UI: linked ROIs, harmonic transfer, mixing and edge previews."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from ssa.config import TemplateBuildConfig
from ssa.gui.harmonic_input import HarmonicInput
from ssa.gui.workers import AnalysisWorker
from ssa.profile_tools import combine_template_entries, transfer_template_entry
from ssa.session import ComparisonRecord
from ssa.templates import build_peak_template


class ProfileWorkbenchMixin:
    """Shared workbench actions used by the main SchottkySA window."""

    def _install_profile_workbench(self) -> None:
        self.active_view_index = 0
        self.last_view_index = 0
        self.last_target_harmonic = 1.0
        self.comparison_runs: list[dict[str, Any]] = []
        self._restoring_views = False
        splitter = self.centralWidget().layout().itemAt(0).widget()
        spectra_container = splitter.widget(0)
        self.plot_tabs = QtWidgets.QTabWidget()
        self.plot_tabs.addTab(spectra_container, "Spectra")
        splitter.insertWidget(0, self.plot_tabs)
        self.secondary_plot = pg.PlotWidget()
        self.secondary_plot.setLabel("bottom", "Frequency", units="Hz")
        self.secondary_plot.setLabel("left", "Linear power")
        self.secondary_plot.showGrid(x=True, y=True, alpha=0.25)
        self.secondary_data_curve = self.secondary_plot.plot([], [], pen=pg.mkPen("w"))
        self.secondary_fit_curve = self.secondary_plot.plot([], [], pen=pg.mkPen("r", width=2))
        self.secondary_background_curve = self.secondary_plot.plot([], [], pen=pg.mkPen("m"))
        self.secondary_region = pg.LinearRegionItem([0, 1], brush=(0, 200, 255, 35))
        self.secondary_plot.addItem(self.secondary_region)
        spectra_container.layout().insertWidget(1, self.secondary_plot, stretch=3)
        self.secondary_plot.hide()

        self.profiles_plot = pg.PlotWidget()
        self.profiles_plot.addLegend()
        self.profiles_plot.setLabel("bottom", "Frequency relative to COG", units="Hz")
        self.profiles_plot.setLabel("left", "Unit-area density")
        self.profiles_plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot_tabs.addTab(self.profiles_plot, "Aligned profiles")
        self._setup_edge_plots()
        self._setup_comparison_plots()

        scroll = splitter.widget(1)
        container = scroll.widget()
        layout = container.layout()
        # Let the tabs occupy the available control-panel height.
        for i in reversed(range(layout.count())):
            if layout.itemAt(i).spacerItem() is not None:
                layout.takeAt(i)
        groups = {g.title(): g for g in container.findChildren(QtWidgets.QGroupBox)}
        template_group = groups["Template construction"]
        taken = template_group.layout().takeRow(self.template_combo)
        if taken.labelItem is not None:
            taken.labelItem.widget().hide()
            taken.labelItem.widget().deleteLater()
        active_group = QtWidgets.QGroupBox("Active template for fitting")
        active_layout = QtWidgets.QVBoxLayout(active_group)
        active_layout.addWidget(self.template_combo)
        layout.insertWidget(2, active_group)
        self.control_tabs = QtWidgets.QTabWidget()
        layout.insertWidget(3, self.control_tabs, stretch=1)
        self._tab_layouts = []
        for title in ("Views", "Templates", "Compare / combine", "Fit", "Results"):
            tab_scroll = QtWidgets.QScrollArea()
            tab_scroll.setWidgetResizable(True)
            page = QtWidgets.QWidget()
            tab_layout = QtWidgets.QVBoxLayout(page)
            tab_scroll.setWidget(page)
            self.control_tabs.addTab(tab_scroll, title)
            self._tab_layouts.append(tab_layout)
        for title, tab_index in (
            ("Template construction", 1),
            ("Fit selected region", 3),
            ("Uncertainty settings", 3),
            ("Results and fit history", 4),
        ):
            group = groups[title]
            layout.removeWidget(group)
            self._tab_layouts[tab_index].addWidget(group)
        self._setup_view_controls(self._tab_layouts[0])
        self._setup_combination_controls(self._tab_layouts[2])
        self.edge_preview_btn = QtWidgets.QPushButton("Preview edges and COG change")
        self.edge_preview_btn.clicked.connect(self.preview_template_edges)
        self._tab_layouts[1].insertWidget(0, self.edge_preview_btn)
        for tab_layout in self._tab_layouts[:4]:
            tab_layout.addStretch(1)
        self.edge_preview_timer = QtCore.QTimer(self)
        self.edge_preview_timer.setSingleShot(True)
        self.edge_preview_timer.setInterval(180)
        self.edge_preview_timer.timeout.connect(self._refresh_visible_edge_preview)
        for widget in (
            self.edge_width_spin,
            self.edge_fraction_spin,
            self.sg_window_spin,
            self.resample_spin,
            *self.edge_side_spins.values(),
        ):
            widget.valueChanged.connect(self._schedule_edge_preview)
        self.edge_mode_combo.currentTextChanged.connect(self._schedule_edge_preview)
        self.smooth_check.toggled.connect(self._schedule_edge_preview)
        self.clip_check.toggled.connect(self._schedule_edge_preview)
        self.region.sigRegionChanged.connect(self._sync_view_controls)
        self.secondary_region.sigRegionChanged.connect(self.update_region_label)
        self.secondary_region.sigRegionChanged.connect(self._sync_view_controls)
        self.plot.scene().sigMouseClicked.connect(lambda _event: self._activate_view(0))
        self.secondary_plot.scene().sigMouseClicked.connect(lambda _event: self._activate_view(1))
        self._sync_view_controls()

    def _setup_view_controls(self, layout: Any) -> None:
        group = QtWidgets.QGroupBox("Independent spectrum regions")
        form = QtWidgets.QFormLayout(group)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self.active_view_combo = QtWidgets.QComboBox()
        self.active_view_combo.addItems(["A", "B"])
        self.active_view_combo.currentIndexChanged.connect(self._activate_view)
        form.addRow("Select / build / fit in", self.active_view_combo)
        self.show_secondary_check = QtWidgets.QCheckBox("Show spectrum region B")
        self.show_secondary_check.toggled.connect(self._set_secondary_visible)
        form.addRow(self.show_secondary_check)
        self.view_harmonic_spins = []
        for name in ("A", "B"):
            spin = HarmonicInput()
            spin.setRange(1e-12, 1e12)
            spin.setDecimals(9)
            spin.setValue(1)
            spin.setToolTip(
                "Enter an integer, decimal or fraction: 214, 214/214, 215/214. Use a consistent harmonic scale for every source and target."
            )
            spin.valueChanged.connect(self._sync_view_controls)
            self.view_harmonic_spins.append(spin)
            form.addRow(f"Harmonic {name}", spin)
        self.goto_frequency_edit = QtWidgets.QLineEdit()
        self.goto_frequency_edit.setPlaceholderText("Absolute frequency in Hz")
        self.view_span_spin = QtWidgets.QDoubleSpinBox()
        self.view_span_spin.setRange(1e-6, 1e10)
        self.view_span_spin.setDecimals(9)
        self.view_span_spin.setValue(4000)
        form.addRow("Go to frequency [Hz]", self.goto_frequency_edit)
        form.addRow("View span [Hz]", self.view_span_spin)
        self.goto_frequency_btn = QtWidgets.QPushButton("Go to frequency")
        self.goto_frequency_btn.clicked.connect(self.goto_frequency)
        form.addRow(self.goto_frequency_btn)
        self.goto_frequency_edit.returnPressed.connect(self.goto_frequency)
        self.other_harmonic_btn = QtWidgets.QPushButton("Send current region to the other harmonic")
        self.other_harmonic_btn.clicked.connect(self.go_to_other_harmonic)
        form.addRow(self.other_harmonic_btn)
        self.roi_lo_edit, self.roi_hi_edit = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        form.addRow("ROI lower [Hz]", self.roi_lo_edit)
        form.addRow("ROI upper [Hz]", self.roi_hi_edit)
        self.apply_roi_btn = QtWidgets.QPushButton("Apply ROI bounds")
        self.apply_roi_btn.clicked.connect(self.apply_roi_bounds)
        form.addRow(self.apply_roi_btn)
        self.view_info_label = QtWidgets.QLabel()
        self.view_info_label.setWordWrap(True)
        form.addRow(self.view_info_label)
        layout.addWidget(group)

    def _setup_combination_controls(self, layout: Any) -> None:
        pair_group = QtWidgets.QGroupBox("Fit each source profile to the other region")
        form = QtWidgets.QFormLayout(pair_group)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self.compare_a_combo, self.compare_b_combo = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        form.addRow("Template A", self.compare_a_combo)
        form.addRow("Template B", self.compare_b_combo)
        self.compare_allow_scale_check = QtWidgets.QCheckBox("Allow extra width stretch after harmonic conversion")
        self.compare_allow_scale_check.setChecked(False)
        form.addRow(self.compare_allow_scale_check)
        self.compare_background_spin = QtWidgets.QSpinBox()
        self.compare_background_spin.setRange(-1, 2)
        form.addRow("Background order", self.compare_background_spin)
        self.profile_workers_spin = QtWidgets.QSpinBox()
        self.profile_workers_spin.setRange(1, 32)
        self.profile_workers_spin.setValue(2)
        form.addRow("Parallel workers", self.profile_workers_spin)
        self.compare_profiles_btn = QtWidgets.QPushButton("Compare A → B and B → A")
        self.compare_profiles_btn.clicked.connect(self.start_profile_comparison)
        form.addRow(self.compare_profiles_btn)
        layout.addWidget(pair_group)
        group = QtWidgets.QGroupBox("Average selected normalized profiles")
        form = QtWidgets.QFormLayout(group)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self.combine_sources_table = QtWidgets.QTableWidget(0, 6)
        self.combine_sources_table.setHorizontalHeaderLabels(["Use", "Template", "h", "Weight", "RMS [Hz]", "ROI [Hz]"])
        for i, width in enumerate((35, 100, 40, 55, 70)):
            self.combine_sources_table.setColumnWidth(i, width)
        self.combine_sources_table.horizontalHeader().setStretchLastSection(True)
        self.combine_sources_table.verticalHeader().hide()
        self.combine_sources_table.setMinimumHeight(150)
        form.addRow(self.combine_sources_table)
        self.combined_name_edit = QtWidgets.QLineEdit("combined_profile")
        form.addRow("New template name", self.combined_name_edit)
        self.combine_harmonic_spin = HarmonicInput()
        self.combine_harmonic_spin.setRange(1e-12, 1e12)
        self.combine_harmonic_spin.setDecimals(9)
        self.combine_harmonic_spin.setValue(1)
        form.addRow("Output harmonic", self.combine_harmonic_spin)
        self.combine_draws_spin = QtWidgets.QSpinBox()
        self.combine_draws_spin.setRange(0, 100000)
        self.combine_draws_spin.setValue(300)
        form.addRow("Profile draws", self.combine_draws_spin)
        self.combine_mode_combo = QtWidgets.QComboBox()
        self.combine_mode_combo.addItem("Source sensitivity (shared data)", "source_sensitivity")
        self.combine_mode_combo.addItem("Mean bootstrap (independent references)", "independent")
        self.combine_mode_combo.setToolTip(
            "Independent mean draws require independent reference errors; different harmonics alone do not establish this."
        )
        form.addRow("Uncertainty construction", self.combine_mode_combo)
        self.preview_profiles_btn = QtWidgets.QPushButton("Show selected profiles on a common axis")
        self.preview_profiles_btn.clicked.connect(self.preview_aligned_profiles)
        self.combine_profiles_btn = QtWidgets.QPushButton("Build averaged template (no extra smoothing)")
        self.combine_profiles_btn.clicked.connect(self.start_combine_profiles)
        form.addRow(self.preview_profiles_btn)
        form.addRow(self.combine_profiles_btn)
        self.profile_info_label = QtWidgets.QLabel(
            "Profiles are centred by COG, converted by harmonic ratio and normalized to unit area. Weights are explicit."
        )
        self.profile_info_label.setWordWrap(True)
        form.addRow(self.profile_info_label)
        layout.addWidget(group)

    def _setup_edge_plots(self) -> None:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        self.edge_plot = pg.PlotWidget()
        self.edge_plot.addLegend()
        self.edge_plot.setLabel("bottom", "Frequency", units="Hz")
        self.edge_plot.setLabel("left", "Normalized density")
        self.edge_before_curve = self.edge_plot.plot([], [], pen=pg.mkPen("w"), name="Observed ROI / no edge treatment")
        self.edge_samples_curve = self.edge_plot.plot(
            [],
            [],
            pen=None,
            symbol="o",
            symbolSize=4,
            symbolBrush=(170, 170, 170),
            name="Measured samples (area normalized)",
        )
        self.edge_after_curve = self.edge_plot.plot([], [], pen=pg.mkPen("c", width=2), name="Selected treatment")
        self.edge_tail_curves = [self.edge_plot.plot([], [], pen=pg.mkPen("orange", width=3)) for _ in range(2)]
        self.edge_used_curves = [
            self.edge_plot.plot(
                [], [], pen=None, symbol="o", symbolSize=7, symbolBrush="g", name=f"{side.title()} edge samples"
            )
            for side in ("left", "right")
        ]
        self.edge_zero_lines = [
            pg.InfiniteLine(angle=90, pen=pg.mkPen("orange", style=QtCore.Qt.PenStyle.DotLine)) for _ in range(2)
        ]
        for line in self.edge_zero_lines:
            self.edge_plot.addItem(line)
            line.hide()
        self.edge_boundary_lines = [
            pg.InfiniteLine(angle=90, pen=pg.mkPen("y", style=QtCore.Qt.PenStyle.DashLine)) for _ in range(2)
        ]
        for line in self.edge_boundary_lines:
            self.edge_plot.addItem(line)
        layout.addWidget(self.edge_plot, 3)
        edges_row = QtWidgets.QHBoxLayout()
        self.edge_zoom_plots = []
        self.edge_zoom_curves = []
        self.edge_zoom_samples = []
        self.edge_zoom_zero_lines = []
        for title in ("Left edge", "Right edge"):
            plot = pg.PlotWidget(title=title)
            plot.setLabel("bottom", "Frequency", units="Hz")
            self.edge_zoom_plots.append(plot)
            self.edge_zoom_curves.append((plot.plot([], [], pen="w"), plot.plot([], [], pen=pg.mkPen("c", width=2))))
            self.edge_zoom_samples.append(plot.plot([], [], pen=None, symbol="o", symbolSize=7, symbolBrush="g"))
            zero_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("orange", style=QtCore.Qt.PenStyle.DotLine))
            plot.addItem(zero_line)
            zero_line.hide()
            self.edge_zoom_zero_lines.append(zero_line)
            edges_row.addWidget(plot)
        layout.addLayout(edges_row, 2)
        self.edge_preview_label = QtWidgets.QLabel("Select a reference ROI and preview its edge treatment.")
        self.edge_preview_label.setWordWrap(True)
        layout.addWidget(self.edge_preview_label)
        self.plot_tabs.addTab(page, "Edges")

    def _setup_comparison_plots(self) -> None:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(page)
        self.comparison_plot_sets = []
        for _ in range(2):
            column = QtWidgets.QVBoxLayout()
            plot, residual_plot = pg.PlotWidget(), pg.PlotWidget()
            plot.addLegend()
            data_curve = plot.plot([], [], pen="w", name="Measured target")
            model_curve = plot.plot([], [], pen=pg.mkPen("orange", width=2), name="Transferred profile fit")
            residual_curve = residual_plot.plot([], [], pen="c")
            residual_plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("w")))
            plot.setLabel("left", "Linear power")
            residual_plot.setLabel("left", "Residual")
            for panel in (plot, residual_plot):
                panel.setLabel("bottom", "Frequency relative to fitted COG / h", units="Hz")
                panel.showGrid(x=True, y=True, alpha=0.2)
            residual_plot.setXLink(plot)
            column.addWidget(plot, 3)
            column.addWidget(residual_plot, 2)
            label = QtWidgets.QLabel()
            label.setWordWrap(True)
            column.addWidget(label)
            layout.addLayout(column)
            self.comparison_plot_sets.append((plot, data_curve, model_curve, residual_curve, label))
        self.plot_tabs.addTab(page, "Cross-fits")

    def _view_region(self, index: int | None = None) -> Any:
        return self.region if (self.active_view_index if index is None else index) == 0 else self.secondary_region

    def _view_plot(self, index: int | None = None) -> Any:
        return self.plot if (self.active_view_index if index is None else index) == 0 else self.secondary_plot

    def active_harmonic_expression(self) -> str:
        return self.view_harmonic_spins[self.active_view_index].expression()

    def active_harmonic(self) -> float:
        return float(self.view_harmonic_spins[self.active_view_index].value())

    def _set_secondary_visible(self, visible: bool) -> None:
        self.secondary_plot.setVisible(visible)
        if not visible and self.active_view_index == 1:
            self._activate_view(0)

    def _clear_profile_displays(self) -> None:
        self.edge_preview_timer.stop()
        self.profiles_plot.clear()
        for curve in (
            self.edge_before_curve,
            self.edge_after_curve,
            self.edge_samples_curve,
            *self.edge_tail_curves,
            *self.edge_used_curves,
            *self.edge_zoom_samples,
            *(c for pair in self.edge_zoom_curves for c in pair),
        ):
            curve.setData([], [])
        for line in (*self.edge_zero_lines, *self.edge_zoom_zero_lines):
            line.hide()
        self.edge_preview_label.setText("Select a reference ROI and preview its edge treatment.")
        for plot, data, model, residual, label in self.comparison_plot_sets:
            for curve in (data, model, residual):
                curve.setData([], [])
            plot.setTitle("")
            label.setText("")

    def _activate_view(self, index: int) -> None:
        if hasattr(self, "component_markers"):
            self._switch_component_view(self.active_view_index, int(index))
        self.active_view_index = int(index)
        if index == 1:
            self.show_secondary_check.setChecked(True)
        if self.active_view_combo.currentIndex() != index:
            self.active_view_combo.setCurrentIndex(index)
        self.update_region_label()
        self._sync_view_controls()
        if hasattr(self, "component_markers"):
            self._refresh_component_controls()

    def _sync_view_controls(self) -> None:
        if not all(s.hasAcceptableInput() for s in self.view_harmonic_spins):
            return
        if not hasattr(self, "roi_lo_edit") or self._restoring_views:
            return
        lo, hi = sorted(self._view_region().getRegion())
        self.roi_lo_edit.setText(f"{lo:.12g}")
        self.roi_hi_edit.setText(f"{hi:.12g}")
        name = "AB"[self.active_view_index]
        self.view_info_label.setText(
            f"Active region {name}; h = {self.active_harmonic():.12g}. Both views use the same input spectrum."
        )
        for i, plot in enumerate((self.plot, self.secondary_plot)):
            plot.setTitle(
                f"Region {'AB'[i]} · h={self.view_harmonic_spins[i].value():.12g}"
                + (" · active" if i == self.active_view_index else "")
            )
        self._schedule_edge_preview()

    def _set_view_window(self, index: int, centre: float, span: float, roi: tuple[float, float] | None = None) -> None:
        if self.frequency is None or self.amplitude is None:
            raise ValueError("Load a spectrum first")
        if not np.isfinite(centre) or not self.frequency.min() <= centre <= self.frequency.max():
            raise ValueError("Requested frequency is outside the loaded spectrum")
        plot = self._view_plot(index)
        plot.setXRange(centre - span / 2, centre + span / 2, padding=0)
        mask = (self.frequency >= centre - span / 2) & (self.frequency <= centre + span / 2)
        if mask.any():
            y = self.amplitude[mask]
            plot.setYRange(float(y.min()), float(y.max()), padding=0.08)
        self._view_region(index).setRegion(roi or (centre - span / 4, centre + span / 4))
        self._activate_view(index)
        self.plot_tabs.setCurrentIndex(0)

    def goto_frequency(self) -> None:
        try:
            self._set_view_window(
                self.active_view_index, float(self.goto_frequency_edit.text()), self.view_span_spin.value()
            )
        except Exception as exc:
            self.show_error(str(exc))

    def go_to_other_harmonic(self) -> None:
        try:
            source, target = self.active_view_index, 1 - self.active_view_index
            factor = self.view_harmonic_spins[target].value() / self.active_harmonic()
            lo, hi = sorted(self._view_region(source).getRegion())
            view_lo, view_hi = self._view_plot(source).viewRange()[0]
            self._set_view_window(
                target, (lo + hi) * factor / 2, (view_hi - view_lo) * factor, (lo * factor, hi * factor)
            )
        except Exception as exc:
            self.show_error(str(exc))

    def apply_roi_bounds(self) -> None:
        try:
            lo, hi = sorted((float(self.roi_lo_edit.text()), float(self.roi_hi_edit.text())))
            if not np.isfinite(lo + hi) or hi <= lo:
                raise ValueError("ROI bounds must be finite and distinct")
            self._view_region().setRegion((lo, hi))
        except Exception as exc:
            self.show_error(str(exc))

    def refresh_profile_sources(self) -> None:
        selected = {}
        for row in range(self.combine_sources_table.rowCount()):
            selected[row] = (
                self.combine_sources_table.item(row, 0).checkState(),
                self.combine_sources_table.item(row, 3).text(),
            )
        pair = (self.compare_a_combo.currentIndex(), self.compare_b_combo.currentIndex())
        self.combine_sources_table.setRowCount(len(self.templates))
        self.compare_a_combo.clear()
        self.compare_b_combo.clear()
        for i, entry in enumerate(self.templates):
            tpl = entry["template"]
            text = f"{i + 1}: {entry['name']} (h={entry.get('harmonic', 1):.12g})"
            self.compare_a_combo.addItem(text)
            self.compare_b_combo.addItem(text)
            values = [
                "",
                entry["name"],
                f"{entry.get('harmonic', 1):.12g}",
                selected.get(i, (None, "1"))[1],
                f"{tpl.raw_std:.6g}",
                f"{entry['region_lo']:.9g}–{entry['region_hi']:.9g}",
            ]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if col == 0:
                    item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(selected.get(i, (QtCore.Qt.CheckState.Unchecked, "1"))[0])
                elif col != 3:
                    item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
                item.setToolTip(value)
                self.combine_sources_table.setItem(i, col, item)
        self.compare_a_combo.setCurrentIndex(max(0, pair[0]))
        self.compare_b_combo.setCurrentIndex(pair[1] if pair[1] >= 1 else min(1, len(self.templates) - 1))

    def _selected_profile_sources(self) -> tuple[list[int], list[float]]:
        ids, weights = [], []
        for row in range(self.combine_sources_table.rowCount()):
            if self.combine_sources_table.item(row, 0).checkState() == QtCore.Qt.CheckState.Checked:
                ids.append(row)
                weights.append(float(self.combine_sources_table.item(row, 3).text()))
        if len(ids) < 2:
            raise ValueError("Check at least two source templates in the Use column")
        leaves = set()
        for index in ids:
            provenance = self.templates[index].get("source_leaf_ids", [index])
            if leaves.intersection(provenance):
                raise ValueError("Selected templates reuse the same source; choose non-duplicated source profiles")
            leaves.update(provenance)
        return ids, weights

    def preview_aligned_profiles(self) -> None:
        try:
            ids, weights = self._selected_profile_sources()
            bank = combine_template_entries(
                [self.templates[i] for i in ids],
                output_harmonic=self.combine_harmonic_spin.value(),
                weights=weights,
                n_boot=0,
            )
            self._plot_aligned_profiles(ids, bank["template_nominal"])
            self.plot_tabs.setCurrentIndex(1)
        except Exception as exc:
            self.show_error(str(exc))

    def _plot_aligned_profiles(self, ids: list[int], combined: Any = None) -> None:
        self.profiles_plot.clear()
        colours = ["c", "y", "m", "g", "b", "w"]
        harmonic = self.combine_harmonic_spin.value()
        for j, index in enumerate(ids):
            entry = transfer_template_entry(self.templates[index], harmonic)
            tpl = entry["template"]
            self.profiles_plot.plot(tpl.u, tpl.pdf, pen=pg.mkPen(colours[j % len(colours)]), name=entry["name"])
        if combined is not None:
            self.profiles_plot.plot(combined.u, combined.pdf, pen=pg.mkPen("orange", width=3), name="Weighted mean")
        self.profiles_plot.setTitle(f"COG-aligned unit-area profiles at output harmonic {harmonic:.12g}")
        self.profiles_plot.autoRange()

    def start_combine_profiles(self) -> None:
        try:
            ids, weights = self._selected_profile_sources()
            config = TemplateBuildConfig(
                name=self.combined_name_edit.text().strip() or "combined_profile",
                n_template_boot=self.combine_draws_spin.value(),
                random_seed=self.seed_spin.value(),
                smooth=False,
            )
            recipe = {
                "source_ids": ids,
                "weights": weights,
                "bank_mode": self.combine_mode_combo.currentData(),
                "workers": self.profile_workers_spin.value(),
            }
            h = self.combine_harmonic_spin.value()
            self._pending_combination = (config, recipe, h)
            payload = {
                "entries": [self.templates[i] for i in ids],
                "name": config.name,
                "output_harmonic": h,
                "weights": weights,
                "n_boot": config.n_template_boot,
                "random_seed": config.random_seed,
                "bank_mode": recipe["bank_mode"],
                "workers": recipe["workers"],
            }
            self.set_busy(True, "Combining aligned reference profiles…")
            self._pending_combination_harmonic_expression = self.combine_harmonic_spin.expression()
            self.worker = AnalysisWorker("combine_profiles", payload)
            self.worker.finished_ok.connect(self.finish_combine_profiles)
            self.worker.failed.connect(self.worker_failed)
            self.worker.start()
        except Exception as exc:
            self.show_error(str(exc))

    def finish_combine_profiles(self, bank: dict[str, Any]) -> None:
        config, recipe, harmonic = self._pending_combination
        tpl = bank["template_nominal"]
        leaves = sorted(
            {i for source in recipe["source_ids"] for i in self.templates[source].get("source_leaf_ids", [source])}
        )
        entry = {
            "name": tpl.name,
            "config": config,
            "harmonic": harmonic,
            "harmonic_expression": self._pending_combination_harmonic_expression,
            "combination": recipe,
            "source_leaf_ids": leaves,
            "region_lo": tpl.raw_cog + tpl.u[0],
            "region_hi": tpl.raw_cog + tpl.u[-1],
            "template": tpl,
            "template_bank": bank["templates"],
            "summary": bank["summary"],
            "settings": bank["settings"],
            "edge_info": tpl.edge_info,
            "residual_model": {},
            "full": bank,
        }
        self.templates.append(entry)
        self.template_combo.addItem(f"{len(self.templates)}: {tpl.name} ({len(bank['templates'])} profile draws)")
        self.template_combo.setCurrentIndex(len(self.templates) - 1)
        self.refresh_profile_sources()
        self._plot_aligned_profiles(recipe["source_ids"], tpl)
        self.plot_tabs.setCurrentIndex(1)
        self.set_busy(False, "Combined template built and selected for fitting.")

    def start_profile_comparison(self) -> None:
        try:
            source, target = self.compare_a_combo.currentIndex(), self.compare_b_combo.currentIndex()
            if source < 0 or target < 0 or source == target:
                raise ValueError("Choose two different measured source templates")
            if any(self.templates[i].get("combination") is not None for i in (source, target)):
                raise ValueError("Choose measured source profiles for bidirectional cross-fits")
            record = ComparisonRecord(
                source,
                target,
                self.compare_allow_scale_check.isChecked(),
                self.compare_background_spin.value(),
                self.profile_workers_spin.value(),
            )
            self._pending_comparison = record
            self.set_busy(True, "Fitting both profile-transfer directions…")
            self.worker = AnalysisWorker(
                "compare_profiles",
                {
                    "frequency": self.frequency,
                    "amplitude": self.amplitude,
                    "entries": self.templates,
                    "source_id": source,
                    "target_id": target,
                    "allow_scale": record.allow_scale,
                    "background_order": record.background_order,
                    "workers": record.workers,
                },
            )
            self.worker.finished_ok.connect(self.finish_profile_comparison)
            self.worker.failed.connect(self.worker_failed)
            self.worker.start()
        except Exception as exc:
            self.show_error(str(exc))

    def finish_profile_comparison(self, directions: list[dict[str, Any]]) -> None:
        record = self._pending_comparison
        record.verify_rows = [d["row"] for d in directions]
        self.comparison_runs.append({"record": record, "directions": directions})
        self.show_profile_comparison(directions)
        self.set_busy(False, "Both cross-fits completed. Inspect residuals and estimated-noise diagnostics.")

    def show_profile_comparison(self, directions: list[dict[str, Any]]) -> None:
        for direction, panels in zip(directions, self.comparison_plot_sets, strict=True):
            fit, row = direction["fit"], direction["row"]
            plot, data_curve, model_curve, residual_curve, label = panels
            x = (fit["x"] - row["mu_fit_Hz"]) / row["target_harmonic"]
            data_curve.setData(x, fit["y"])
            model_curve.setData(x, fit["y_model"])
            residual_curve.setData(x, fit["y"] - fit["y_model"])
            plot.setTitle(f"{row['source_name']} → {row['target_name']}")
            label.setText(
                f"χ²/ν = {row['red_chi2']:.4g} ({row['dof']} dof); DW = {row['DW']:.3g}; extra width scale = {row['extra_width_scale_fit']:.6g}.\nResidual mismatch can reflect contamination, response differences or noise; these diagnostics use an estimated noise level."
            )
            plot.autoRange()
        self.plot_tabs.setCurrentIndex(3)

    def _schedule_edge_preview(self) -> None:
        if hasattr(self, "edge_preview_timer"):
            self.edge_preview_timer.start()

    def _refresh_visible_edge_preview(self) -> None:
        if self.plot_tabs.currentIndex() == 2:
            self.preview_template_edges()

    def preview_template_edges(self) -> None:
        try:
            x, y = self.selected_region_xy()
            config = asdict(self._template_config_from_widgets())
            for key in ("n_template_boot", "block_size", "random_seed"):
                config.pop(key)
            processed = build_peak_template(x, y, **config)
            config["edge_mode"] = "none"
            untreated = build_peak_template(x, y, **config)
            raw_x, processed_x = untreated.u + untreated.raw_cog, processed.u + processed.raw_cog
            sample_area = float(np.trapezoid(np.maximum(y, 0), x))
            self.edge_samples_curve.setData(x, y / sample_area)
            self.edge_before_curve.setData(raw_x, untreated.pdf)
            self.edge_after_curve.setData(processed_x, processed.pdf)
            for i, mask in enumerate((processed_x <= x[0], processed_x >= x[-1])):
                self.edge_tail_curves[i].setData(processed_x[mask], processed.pdf[mask])
                self.edge_boundary_lines[i].setValue(float(x[0] if i == 0 else x[-1]))
            info = processed.edge_info
            side_details = []
            for i, (plot, curves) in enumerate(zip(self.edge_zoom_plots, self.edge_zoom_curves, strict=True)):
                curves[0].setData(raw_x, untreated.pdf)
                curves[1].setData(processed_x, processed.pdf)
                side = "left" if i == 0 else "right"
                boundary = float(x[0] if i == 0 else x[-1])
                n = int(info.get(f"{side}_edge_points_used", 0))
                sl = slice(0, n) if i == 0 else slice(len(x) - n, len(x))
                for curve in (self.edge_used_curves[i], self.edge_zoom_samples[i]):
                    curve.setData(x[sl], y[sl] / sample_area)
                zero = info.get(f"{side}_zero_frequency_hz")
                for line in (self.edge_zero_lines[i], self.edge_zoom_zero_lines[i]):
                    line.setVisible(zero is not None)
                    if zero is not None:
                        line.setValue(float(zero))
                width = max(
                    float(info.get(f"{side}_tail_width_hz", 0)),
                    n * float(np.median(np.diff(x))),
                    float(np.ptp(x)) * 0.15,
                )
                plot.setXRange(boundary - 1.1 * width, boundary + 1.1 * width, padding=0)
                side_details.append(
                    f"{side.title()}: {n} samples; "
                    + (f"zero at {zero:.9f} Hz" if zero is not None else "no outside zero anchor")
                )
            self.edge_preview_label.setText(
                f"Observed ROI: {x[0]:.9f}–{x[-1]:.9f} Hz. COG shift from selected edge handling: {processed.raw_cog - untreated.raw_cog:+.6g} Hz.\nMode: {info.get('edge_mode')}; edge width: {info.get('edge_width_hz', 0):.6g} Hz; subtracted baseline: {info.get('edge_baseline_subtracted', 0):.6g}. Orange segments are outside the measured ROI and represent the chosen closure, not additional observations. Green points are the samples used for each edge; dotted orange lines mark the zero anchors.\n"
                + " | ".join(side_details)
            )
            self.edge_plot.autoRange()
            self.plot_tabs.setCurrentIndex(2)
        except Exception as exc:
            self.edge_preview_label.setText(str(exc))

    def _view_state_for_session(self) -> dict[str, Any]:
        return {
            "component_picker": self._component_state_for_session(),
            "active_view": self.active_view_index,
            "secondary_visible": self.show_secondary_check.isChecked(),
            "regions": [list(self.region.getRegion()), list(self.secondary_region.getRegion())],
            "harmonics": [s.value() for s in self.view_harmonic_spins],
            "harmonic_expressions": [s.expression() for s in self.view_harmonic_spins],
            "x_ranges": [self.plot.viewRange()[0], self.secondary_plot.viewRange()[0]],
            "goto_frequency": self.goto_frequency_edit.text(),
            "view_span": self.view_span_spin.value(),
            "compare_pair": [self.compare_a_combo.currentIndex(), self.compare_b_combo.currentIndex()],
            "compare_allow_scale": self.compare_allow_scale_check.isChecked(),
            "compare_background_order": self.compare_background_spin.value(),
            "workers": self.profile_workers_spin.value(),
            "combined_name": self.combined_name_edit.text(),
            "combined_harmonic": self.combine_harmonic_spin.value(),
            "combined_harmonic_expression": self.combine_harmonic_spin.expression(),
            "combined_draws": self.combine_draws_spin.value(),
            "combined_mode": self.combine_mode_combo.currentData(),
            "source_checks": [
                self.combine_sources_table.item(i, 0).checkState() == QtCore.Qt.CheckState.Checked
                for i in range(len(self.templates))
            ],
            "source_weights": [self.combine_sources_table.item(i, 3).text() for i in range(len(self.templates))],
        }

    def _restore_profile_view_state(self, state: dict[str, Any]) -> None:
        self.refresh_profile_sources()
        self.secondary_data_curve.setData(self.frequency, self.amplitude)
        self._restoring_views = True
        try:
            for i, h in enumerate(state.get("harmonics", [1, 1])):
                expressions = state.get("harmonic_expressions", [])
                if i < len(expressions):
                    self.view_harmonic_spins[i].setExpression(expressions[i])
                else:
                    self.view_harmonic_spins[i].setValue(h)
            for i, bounds in enumerate(state.get("regions", [])):
                self._view_region(i).setRegion(bounds)
            for i, bounds in enumerate(state.get("x_ranges", [])):
                self._view_plot(i).setXRange(*bounds, padding=0)
            self.show_secondary_check.setChecked(state.get("secondary_visible", False))
            self.active_view_combo.setCurrentIndex(state.get("active_view", 0))
            self.active_view_index = state.get("active_view", 0)
            self.goto_frequency_edit.setText(state.get("goto_frequency", ""))
            self.view_span_spin.setValue(state.get("view_span", 4000))
            for combo, index in zip(
                (self.compare_a_combo, self.compare_b_combo), state.get("compare_pair", [0, 1]), strict=True
            ):
                combo.setCurrentIndex(index)
            self.compare_allow_scale_check.setChecked(state.get("compare_allow_scale", False))
            self.compare_background_spin.setValue(state.get("compare_background_order", 0))
            self.profile_workers_spin.setValue(state.get("workers", 2))
            self.combined_name_edit.setText(state.get("combined_name", "combined_profile"))
            if state.get("combined_harmonic_expression"):
                self.combine_harmonic_spin.setExpression(state["combined_harmonic_expression"])
            else:
                self.combine_harmonic_spin.setValue(state.get("combined_harmonic", 1))
            self.combine_draws_spin.setValue(state.get("combined_draws", 300))
            self.combine_mode_combo.setCurrentIndex(1 if state.get("combined_mode") == "independent" else 0)
            for i, checked in enumerate(state.get("source_checks", [])):
                self.combine_sources_table.item(i, 0).setCheckState(
                    QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked
                )
            for i, weight in enumerate(state.get("source_weights", [])):
                self.combine_sources_table.item(i, 3).setText(weight)
        finally:
            self._restoring_views = False
        self._restore_component_state(state.get("component_picker", {}))
        self._sync_view_controls()
