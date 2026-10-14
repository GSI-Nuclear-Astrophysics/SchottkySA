"""The SchottkySA main window."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from ssa import __version__ as SSA_VERSION
from ssa.bundles import create_run_directory
from ssa.config import FitConfig, TemplateBuildConfig, UncertaintyConfig
from ssa.constants import EDGE_MODES, LOSS_CHOICES
from ssa.gui.component_picker import ComponentPickerMixin
from ssa.gui.profile_workbench import ProfileWorkbenchMixin
from ssa.gui.workers import AnalysisWorker, SessionReplayWorker
from ssa.io import export_current_overlay_npz as _export_current_overlay_npz
from ssa.io import export_history_rows, make_export_rows_for_result, rows_to_tsv, select_npz_keys
from ssa.preprocessing import prepare_xy
from ssa.profile_tools import transfer_template_entry
from ssa.session import (
    SCHEMA_VERSION,
    FitRecord,
    FormState,
    InputRef,
    ReplayReport,
    Session,
    TemplateRecord,
    _sha256_of,
    _to_toml_value,
)
from ssa.session import (
    load_session as _load_session,
)
from ssa.session import (
    save_session as _save_session,
)
from ssa.templates import get_template_std
from ssa.uncertainty import (
    combined_sigma,
    component_quantiles_from_fit,
    component_widths_from_params,
    quality_flag_for_fit,
    sample_count,
    summarize_area_ratio_samples,
    summarize_mu_corr_samples,
    summarize_mu_delta_samples,
)

__all__ = ["MainWindow"]


def _template_verify_from_entry(entry: dict[str, Any]) -> dict[str, float]:
    tpl = entry["template"]
    q16, q50, q84 = tpl.quantile([0.16, 0.50, 0.84])
    return {
        "raw_area": float(tpl.raw_area),
        "raw_cog": float(tpl.raw_cog),
        "raw_std": float(tpl.raw_std),
        "q16": float(q16),
        "q50": float(q50),
        "q84": float(q84),
        "n_template_bank": len(entry["template_bank"]),
    }


class MainWindow(ComponentPickerMixin, ProfileWorkbenchMixin, QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SchottkySA -- Schottky Shape Analyzer")
        self.resize(1650, 950)

        self.frequency: np.ndarray | None = None
        self.amplitude: np.ndarray | None = None
        self.current_file: Path | None = None
        self._frequency_key: str = ""
        self._amplitude_key: str = ""
        self.templates: list[dict[str, Any]] = []
        self.fit_records: list[dict[str, Any]] = []
        self.last_result: dict[str, Any] | None = None
        self.last_fit_config: FitConfig | None = None
        self.last_uncertainty_config: UncertaintyConfig | None = None
        self.last_template_index: int = -1
        self.last_region_hz: tuple[float, float] = (0.0, 0.0)
        self.worker: AnalysisWorker | None = None
        self._updating_history_table = False
        self._loading_session: Session | None = None
        self._bundle_dir: Path | None = None
        self._bundle_signature: str | None = None
        self._data_sha256: str | None = None
        self._restored_template_config: TemplateBuildConfig | None = None
        self._restored_fit_config: FitConfig | None = None

        pg.setConfigOptions(antialias=False)

        self._build_ui()
        self._install_profile_workbench()
        self._install_component_picker()
        self._overlay_plot = self.plot
        self._connect_signals()

    # ----------------------------- UI construction ----------------------------

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        file_menu = self.menuBar().addMenu("&File")
        self.save_session_action = file_menu.addAction("Save config + tables…")
        self.load_session_action = file_menu.addAction("Replay config…")

        layout = QtWidgets.QHBoxLayout(central)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left side: plots
        plot_container = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_container)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Frequency", units="Hz")
        self.plot.setLabel("left", "Amplitude / linear power")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.plotItem.setDownsampling(auto=True, mode="peak")
        self.plot.plotItem.setClipToView(True)

        self.data_curve = self.plot.plot([], [], pen=pg.mkPen(width=1), name="data")
        self.fit_curve = self.plot.plot([], [], pen=pg.mkPen("r", width=2), name="fit")
        self.component_curves: list[pg.PlotDataItem] = []
        self.background_curve = self.plot.plot(
            [], [], pen=pg.mkPen("m", width=1, style=QtCore.Qt.PenStyle.DashLine), name="background"
        )

        self.region = pg.LinearRegionItem(values=[0, 1], orientation="vertical", brush=(255, 255, 0, 35), movable=True)
        self.region.setZValue(10)
        self.plot.addItem(self.region)

        self.lower_plot = pg.PlotWidget()
        self.lower_plot.setMaximumHeight(260)
        self.lower_plot.showGrid(x=True, y=True, alpha=0.25)
        self.lower_plot.setLabel("bottom", "x")
        self.lower_plot.setLabel("left", "template / residual")
        self.lower_curve = self.lower_plot.plot([], [], pen=pg.mkPen("c", width=2))
        self.lower_curve_2 = self.lower_plot.plot([], [], pen=pg.mkPen("y", width=1))

        plot_layout.addWidget(self.plot, stretch=4)
        plot_layout.addWidget(self.lower_plot, stretch=1)
        splitter.addWidget(plot_container)

        # Right side: controls/results
        control_container = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_container)
        control_scroll = QtWidgets.QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setWidget(control_container)
        splitter.addWidget(control_scroll)
        splitter.setSizes([1100, 550])

        # File group
        file_group = QtWidgets.QGroupBox("Data")
        file_form = QtWidgets.QVBoxLayout(file_group)
        self.load_btn = QtWidgets.QPushButton("Load NPZ")
        self.file_label = QtWidgets.QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        self.region_label = QtWidgets.QLabel("Selected region: —")
        file_form.addWidget(self.load_btn)
        file_form.addWidget(self.file_label)
        file_form.addWidget(self.region_label)
        control_layout.addWidget(file_group)

        # Visible saved-run controls, independent of platform menu placement.
        session_group = QtWidgets.QGroupBox("Saved runs: configuration and tables")
        session_form = QtWidgets.QFormLayout(session_group)
        self.output_parent_edit = QtWidgets.QLineEdit()
        self.output_parent_edit.setPlaceholderText("Choose an output parent folder")
        self.output_browse_btn = QtWidgets.QPushButton("Browse…")
        parent_row = QtWidgets.QHBoxLayout()
        parent_row.addWidget(self.output_parent_edit)
        parent_row.addWidget(self.output_browse_btn)
        session_form.addRow("Output parent", parent_row)
        self.bundle_label_edit = QtWidgets.QLineEdit()
        self.bundle_label_edit.setPlaceholderText("Optional label, e.g. region or setting")
        session_form.addRow("Folder label", self.bundle_label_edit)
        self.table_format_combo = QtWidgets.QComboBox()
        self.table_format_combo.addItems(["CSV", "TSV", "JSON", "XLSX", "ODS"])
        session_form.addRow("Additional table format", self.table_format_combo)
        self.save_config_btn = QtWidgets.QPushButton("Save config + tables")
        self.replay_config_btn = QtWidgets.QPushButton("Replay config…")
        session_buttons = QtWidgets.QHBoxLayout()
        session_buttons.addWidget(self.save_config_btn)
        session_buttons.addWidget(self.replay_config_btn)
        session_form.addRow(session_buttons)
        self.bundle_path_label = QtWidgets.QLabel("DATE_SchottkySA_CONFIG[_label]")
        self.bundle_path_label.setWordWrap(True)
        session_form.addRow(self.bundle_path_label)
        control_layout.addWidget(session_group)

        # Template group
        tpl_group = QtWidgets.QGroupBox("Template construction")
        tpl_layout = QtWidgets.QFormLayout(tpl_group)
        self.template_name_edit = QtWidgets.QLineEdit("reference_peak")
        self.template_boot_spin = QtWidgets.QSpinBox()
        self.template_boot_spin.setRange(0, 100000)
        self.template_boot_spin.setValue(TemplateBuildConfig().n_template_boot)
        self.template_boot_spin.setSingleStep(100)
        self.resample_spin = QtWidgets.QSpinBox()
        self.resample_spin.setRange(1, 64)
        self.resample_spin.setValue(16)
        self.smooth_check = QtWidgets.QCheckBox()
        self.smooth_check.setChecked(False)
        self.clip_check = QtWidgets.QCheckBox()
        self.clip_check.setChecked(True)
        self.sg_window_spin = QtWidgets.QSpinBox()
        self.sg_window_spin.setRange(0, 10001)
        self.sg_window_spin.setValue(0)
        self.sg_window_spin.setSpecialValueText("auto")
        self.edge_mode_combo = QtWidgets.QComboBox()
        self.edge_mode_combo.addItems(list(EDGE_MODES))
        self.edge_width_spin = QtWidgets.QDoubleSpinBox()
        self.edge_width_spin.setRange(0.0, 1e9)
        self.edge_width_spin.setDecimals(6)
        self.edge_width_spin.setValue(0.0)
        self.edge_width_spin.setSpecialValueText("auto")
        self.edge_fraction_spin = QtWidgets.QDoubleSpinBox()
        self.edge_fraction_spin.setRange(0.01, 0.49)
        self.edge_fraction_spin.setDecimals(3)
        self.edge_fraction_spin.setValue(0.08)
        self.edge_side_spins = {}
        for side in ("left", "right"):
            points = QtWidgets.QSpinBox()
            points.setRange(0, 100000)
            points.setSpecialValueText("auto")
            points.setToolTip(
                "Number of measured edge samples used for this side. 0 inherits the shared width/fraction; actual counts appear in the preview."
            )
            distance = QtWidgets.QDoubleSpinBox()
            distance.setRange(0.0, 1e9)
            distance.setDecimals(9)
            distance.setSpecialValueText("auto")
            distance.setToolTip(
                "Distance in spectrum Hz from this ROI boundary to the zero anchor outside the ROI. Applies to local-slope/PCHIP/padding; 0 inherits the shared width. Smaller values give a sharper closure."
            )
            self.edge_side_spins[f"{side}_edge_points"] = points
            self.edge_side_spins[f"{side}_tail_width_hz"] = distance
        self.build_template_btn = QtWidgets.QPushButton("Build/store template from selected region")
        self.template_combo = QtWidgets.QComboBox()
        tpl_layout.addRow("Name", self.template_name_edit)
        tpl_layout.addRow("Bootstrap templates", self.template_boot_spin)
        tpl_layout.addRow("Resample factor", self.resample_spin)
        tpl_layout.addRow("Smooth nominal template", self.smooth_check)
        tpl_layout.addRow("Clip negative", self.clip_check)
        tpl_layout.addRow("Residual SG window", self.sg_window_spin)
        tpl_layout.addRow("Template edge mode", self.edge_mode_combo)
        tpl_layout.addRow("Edge width [Hz] (0=auto)", self.edge_width_spin)
        tpl_layout.addRow("Edge fraction", self.edge_fraction_spin)
        for side in ("left", "right"):
            tpl_layout.addRow(f"{side.title()} edge samples (0=auto)", self.edge_side_spins[f"{side}_edge_points"])
            tpl_layout.addRow(f"{side.title()} zero distance [Hz]", self.edge_side_spins[f"{side}_tail_width_hz"])
        tpl_layout.addRow(self.build_template_btn)
        tpl_layout.addRow("Active template", self.template_combo)
        control_layout.addWidget(tpl_group)

        # Fit group
        fit_group = QtWidgets.QGroupBox("Fit selected region")
        fit_layout = QtWidgets.QFormLayout(fit_group)
        self.fit_label_edit = QtWidgets.QLineEdit("fit_1")
        self.n_peaks_spin = QtWidgets.QSpinBox()
        self.n_peaks_spin.setRange(1, 12)
        self.n_peaks_spin.setValue(1)
        self.background_order_spin = QtWidgets.QSpinBox()
        self.background_order_spin.setRange(-1, 5)
        self.background_order_spin.setValue(0)
        self.background_order_spin.setSpecialValueText("none")
        self.allow_scale_check = QtWidgets.QCheckBox()
        self.allow_scale_check.setChecked(True)
        self.common_scale_check = QtWidgets.QCheckBox()
        self.common_scale_check.setChecked(True)
        self.loss_combo = QtWidgets.QComboBox()
        self.loss_combo.addItems(list(LOSS_CHOICES))
        self.init_mus_edit = QtWidgets.QLineEdit("")
        self.init_mus_edit.setPlaceholderText("optional: mu0, mu1, ...")
        self.mu_ranges_edit = QtWidgets.QLineEdit("")
        self.mu_ranges_edit.setPlaceholderText("e.g. 1937348:1937355; 1937380:1937390")
        self.mu_ranges_edit.setToolTip(
            "Hard search windows for each centroid, one per component. Use semicolon-separated lo:hi ranges in Hz. If set, the optimizer cannot leave these ranges."
        )
        self.mu_range_halfwidth_spin = QtWidgets.QDoubleSpinBox()
        self.mu_range_halfwidth_spin.setRange(0.0, 1e9)
        self.mu_range_halfwidth_spin.setDecimals(6)
        self.mu_range_halfwidth_spin.setValue(0.0)
        self.mu_range_halfwidth_spin.setSpecialValueText("off")
        self.mu_range_halfwidth_spin.setToolTip(
            "Optional convenience: if Initial mus are provided and explicit ranges are empty, build hard ranges mu_i ± this half-width."
        )
        self.min_sep_spin = QtWidgets.QDoubleSpinBox()
        self.min_sep_spin.setRange(0.0, 1e9)
        self.min_sep_spin.setDecimals(6)
        self.min_sep_spin.setValue(0.0)
        self.min_sep_spin.setToolTip(
            "Minimum allowed separation. With mu search ranges, the code verifies that the ranges guarantee this separation; otherwise the fit fails instead of accepting a violation."
        )
        self.auto_store_check = QtWidgets.QCheckBox()
        self.auto_store_check.setChecked(True)
        self.fit_btn = QtWidgets.QPushButton("Fit selected region with template")
        fit_layout.addRow("Fit label", self.fit_label_edit)
        fit_layout.addRow("Components", self.n_peaks_spin)
        fit_layout.addRow("Background order", self.background_order_spin)
        fit_layout.addRow("Allow scale", self.allow_scale_check)
        fit_layout.addRow("Common scale", self.common_scale_check)
        fit_layout.addRow("Loss", self.loss_combo)
        fit_layout.addRow("Initial mus CSV", self.init_mus_edit)
        fit_layout.addRow("Mu search ranges [Hz]", self.mu_ranges_edit)
        fit_layout.addRow("Auto range half-width [Hz]", self.mu_range_halfwidth_spin)
        fit_layout.addRow("Min separation [Hz]", self.min_sep_spin)
        fit_layout.addRow("Auto-store fit", self.auto_store_check)
        fit_layout.addRow(self.fit_btn)
        control_layout.addWidget(fit_group)

        # Uncertainty group
        unc_group = QtWidgets.QGroupBox("Uncertainty settings")
        unc_layout = QtWidgets.QFormLayout(unc_group)
        self.block_size_edit = QtWidgets.QLineEdit("auto")
        self.run_boot_check = QtWidgets.QCheckBox()
        self.run_boot_check.setChecked(True)
        self.n_boot_spin = QtWidgets.QSpinBox()
        self.n_boot_spin.setRange(0, 200000)
        self.n_boot_spin.setValue(UncertaintyConfig().n_boot)
        self.n_boot_spin.setSingleStep(100)
        self.run_tpl_prop_check = QtWidgets.QCheckBox()
        self.run_tpl_prop_check.setChecked(True)
        self.tpl_prop_spin = QtWidgets.QSpinBox()
        self.tpl_prop_spin.setRange(0, 100000)
        self.tpl_prop_spin.setValue(UncertaintyConfig().n_template_prop)
        self.tpl_prop_spin.setSingleStep(100)
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 2_000_000_000)
        self.seed_spin.setValue(12345)
        unc_layout.addRow("Block size", self.block_size_edit)
        unc_layout.addRow("Run bootstrap", self.run_boot_check)
        unc_layout.addRow("Bootstrap samples", self.n_boot_spin)
        unc_layout.addRow("Propagate template", self.run_tpl_prop_check)
        unc_layout.addRow("Template-prop draws", self.tpl_prop_spin)
        unc_layout.addRow("Random seed", self.seed_spin)
        control_layout.addWidget(unc_group)

        # Results tabs
        results_group = QtWidgets.QGroupBox("Results and fit history")
        res_layout = QtWidgets.QVBoxLayout(results_group)
        self.status_label = QtWidgets.QLabel("Ready")

        self.tabs = QtWidgets.QTabWidget()

        # Current result table
        current_widget = QtWidgets.QWidget()
        current_layout = QtWidgets.QVBoxLayout(current_widget)
        self.results_table = QtWidgets.QTableWidget(0, 13)
        self.results_table.setHorizontalHeaderLabels(
            [
                "component",
                "COG fit",
                "COG boot",
                "COG tpl",
                "COG combined σ",
                "area fit",
                "scale fit",
                "RMS width fit",
                "width boot",
                "width tpl",
                "width combined σ",
                "sigma68",
                "red chi2",
            ]
        )
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setMinimumHeight(170)
        current_layout.addWidget(self.results_table)
        self.tabs.addTab(current_widget, "Current fit")

        # Stored fit history table
        history_widget = QtWidgets.QWidget()
        history_layout = QtWidgets.QVBoxLayout(history_widget)
        self.history_columns = [
            "label",
            "component",
            "template",
            "quality_flag",
            "fit_region_lo",
            "fit_region_hi",
            "template_region_lo",
            "template_region_hi",
            "mu_fit_Hz",
            "mu_combined_sigma_Hz",
            "mu_bound_lo_Hz",
            "mu_bound_hi_Hz",
            "min_separation_Hz",
            "area_fit",
            "scale_fit",
            "rms_width_Hz",
            "width_combined_sigma_Hz",
            "q16_Hz",
            "q50_Hz",
            "q84_Hz",
            "sigma68_Hz",
            "delta_mu_prev_fit_Hz",
            "delta_mu_prev_combined_sigma_Hz",
            "area_ratio_to_comp0_fit",
            "area_ratio_to_comp0_combined_sigma",
            "mu_corr_prev_bootstrap",
            "area_corr_comp0_bootstrap",
            "red_chi2",
            "target_boot_n_valid",
            "template_prop_n_valid",
            "notes",
        ]
        self.history_table = QtWidgets.QTableWidget(0, len(self.history_columns))
        self.history_table.setHorizontalHeaderLabels(self.history_columns)
        self.history_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.history_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.history_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
        )
        history_layout.addWidget(self.history_table)

        hist_buttons = QtWidgets.QGridLayout()
        self.store_current_btn = QtWidgets.QPushButton("Store current fit")
        self.copy_selected_btn = QtWidgets.QPushButton("Copy selected rows")
        self.copy_all_btn = QtWidgets.QPushButton("Copy all rows")
        self.export_history_btn = QtWidgets.QPushButton("Export fit history")
        self.export_overlay_btn = QtWidgets.QPushButton("Export current overlay NPZ")
        self.delete_selected_btn = QtWidgets.QPushButton("Delete selected")
        hist_buttons.addWidget(self.store_current_btn, 0, 0)
        hist_buttons.addWidget(self.copy_selected_btn, 0, 1)
        hist_buttons.addWidget(self.copy_all_btn, 1, 0)
        hist_buttons.addWidget(self.export_history_btn, 1, 1)
        hist_buttons.addWidget(self.export_overlay_btn, 2, 0)
        hist_buttons.addWidget(self.delete_selected_btn, 2, 1)
        history_layout.addLayout(hist_buttons)

        self.tabs.addTab(history_widget, "Stored fits")

        # Summary text
        summary_widget = QtWidgets.QWidget()
        summary_layout = QtWidgets.QVBoxLayout(summary_widget)
        self.summary_text = QtWidgets.QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMinimumHeight(260)
        summary_layout.addWidget(self.summary_text)
        self.tabs.addTab(summary_widget, "Summary")

        res_layout.addWidget(self.status_label)
        res_layout.addWidget(self.tabs)
        control_layout.addWidget(results_group)

        # Keep the saved-run buttons reachable when long option labels need wrapping.
        for form in control_container.findChildren(QtWidgets.QFormLayout):
            form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        control_layout.addStretch(1)
        self.statusBar().showMessage("Ready")

    def _connect_signals(self) -> None:
        self.save_config_btn.clicked.connect(self.save_session)
        self.replay_config_btn.clicked.connect(self.load_session)
        self.output_browse_btn.clicked.connect(self._choose_output_parent)
        self.save_session_action.triggered.connect(self.save_session)
        self.load_session_action.triggered.connect(self.load_session)
        self.load_btn.clicked.connect(self.load_npz)
        self.build_template_btn.clicked.connect(self.start_build_template)
        self.fit_btn.clicked.connect(self.start_fit_region)
        self.store_current_btn.clicked.connect(self.store_current_fit)
        self.copy_selected_btn.clicked.connect(self.copy_selected_history_rows)
        self.copy_all_btn.clicked.connect(self.copy_all_history_rows)
        self.export_history_btn.clicked.connect(self.export_history)
        self.export_overlay_btn.clicked.connect(self.export_current_overlay_npz)
        self.delete_selected_btn.clicked.connect(self.delete_selected_history_rows)
        self.region.sigRegionChanged.connect(self.update_region_label)
        self.template_combo.currentIndexChanged.connect(self.show_active_template)
        self.history_table.itemChanged.connect(self.on_history_item_changed)
        self.history_table.itemSelectionChanged.connect(self.show_selected_history_fit)

    # ------------------------------- Data loading -----------------------------

    def load_npz(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open NPZ spectrum", "", "NPZ files (*.npz);;All files (*)"
        )
        if not path:
            return
        try:
            data = np.load(path)
            keys = list(data.keys())
            if not keys:
                raise ValueError("NPZ file contains no arrays.")

            f_key, y_key = self._choose_npz_keys(data, keys)
            if f_key is None or y_key is None:
                return

            f = np.asarray(data[f_key], dtype=float).ravel()
            y = np.asarray(data[y_key], dtype=float).ravel()
            f, y = prepare_xy(f, y)

            new_sha256 = _sha256_of(path)
            if self._data_sha256 != new_sha256:
                # A session must never associate old fits with a different spectrum.
                self.templates = []
                self.fit_records = []
                self.comparison_runs = []
                self._clear_profile_displays()
                self._clear_all_graphical_components()
                self.refresh_profile_sources()
                self.template_combo.clear()
                self.refresh_history_table()
                self.last_result = None
                self.last_fit_config = None
                self.last_uncertainty_config = None
                self._restored_template_config = None
                self._restored_fit_config = None
            self.frequency = f
            self.amplitude = y
            self.current_file = Path(path).resolve()
            self._data_sha256 = new_sha256
            self._frequency_key = f_key
            self._amplitude_key = y_key
            self.file_label.setText(f"{Path(path).name}\nfrequency: {f_key}, amplitude: {y_key}\nN={len(f):,}")
            self.data_curve.setData(f, y)
            self.secondary_data_curve.setData(f, y)
            self.plot.setXRange(float(f.min()), float(f.max()), padding=0.02)
            self.plot.setYRange(float(np.nanmin(y)), float(np.nanmax(y)), padding=0.05)

            span = float(f.max() - f.min())
            center = float(0.5 * (f.min() + f.max()))
            self.active_view_index = 0
            self.active_view_combo.setCurrentIndex(0)
            self.region.setRegion([center - 0.05 * span, center + 0.05 * span])
            self.secondary_region.setRegion(self.region.getRegion())
            self.secondary_plot.setXRange(*self.plot.viewRange()[0], padding=0)
            self.clear_fit_overlay()
            self.status("Loaded NPZ file.")
        except Exception as exc:
            self.show_error(f"Failed to load NPZ:\n{exc}")

    def _choose_npz_keys(self, data: Any, keys: list[str]) -> tuple[str | None, str | None]:
        one_d = [k for k in keys if np.asarray(data[k]).ndim == 1]
        f_key, y_key = select_npz_keys(keys, one_d)
        if f_key is not None and y_key is not None:
            return f_key, y_key

        f_key, ok = QtWidgets.QInputDialog.getItem(
            self, "Choose frequency array", "Frequency key:", one_d or keys, 0, False
        )
        if not ok:
            return None, None
        remaining = [k for k in (one_d or keys) if k != f_key]
        y_key, ok = QtWidgets.QInputDialog.getItem(
            self, "Choose amplitude array", "Amplitude key:", remaining, 0, False
        )
        if not ok:
            return None, None
        return f_key, y_key

    # ------------------------------- Regions ---------------------------------

    def selected_region_xy(self) -> tuple[np.ndarray, np.ndarray]:
        if self.frequency is None or self.amplitude is None:
            raise ValueError("No data loaded.")
        lo, hi = self._view_region().getRegion()
        if hi < lo:
            lo, hi = hi, lo
        mask = (self.frequency >= lo) & (self.frequency <= hi)
        if np.count_nonzero(mask) < 7:
            raise ValueError("Selected region has too few points.")
        return self.frequency[mask], self.amplitude[mask]

    def selected_region_bounds(self) -> tuple[float, float]:
        lo, hi = self._view_region().getRegion()
        if hi < lo:
            lo, hi = hi, lo
        return float(lo), float(hi)

    def update_region_label(self) -> None:
        lo, hi = self.selected_region_bounds()
        self.region_label.setText(f"Selected region: {lo:.9g} to {hi:.9g} Hz")

    def parse_block_size(self) -> str | int:
        txt = self.block_size_edit.text().strip().lower()
        if txt in ("", "auto"):
            return "auto"
        return max(1, int(float(txt)))

    def _template_config_from_widgets(self) -> TemplateBuildConfig:
        sg_win = self.sg_window_spin.value()
        return TemplateBuildConfig(
            name=self.template_name_edit.text().strip() or f"template_{len(self.templates) + 1}",
            n_template_boot=int(self.template_boot_spin.value()),
            block_size=self.parse_block_size(),
            random_seed=int(self.seed_spin.value()),
            clip_negative=bool(self.clip_check.isChecked()),
            smooth=bool(self.smooth_check.isChecked()),
            sg_window=None if sg_win == 0 else int(sg_win),
            sg_poly=self._restored_template_config.sg_poly
            if self._restored_template_config
            else TemplateBuildConfig().sg_poly,
            resample_factor=int(self.resample_spin.value()),
            edge_mode=self.edge_mode_combo.currentText(),
            edge_width_hz=float(self.edge_width_spin.value()),
            edge_fraction=float(self.edge_fraction_spin.value()),
            left_edge_points=int(self.edge_side_spins["left_edge_points"].value()),
            right_edge_points=int(self.edge_side_spins["right_edge_points"].value()),
            left_tail_width_hz=float(self.edge_side_spins["left_tail_width_hz"].value()),
            right_tail_width_hz=float(self.edge_side_spins["right_tail_width_hz"].value()),
        )

    def _fit_config_from_widgets(
        self, n_peaks: int, init_mus: list[float] | None, mu_bounds: list[tuple[float, float]] | None
    ) -> FitConfig:
        return FitConfig(
            n_peaks=n_peaks,
            init_mus=init_mus,
            mu_bounds=mu_bounds,
            background_order=int(self.background_order_spin.value()),
            allow_scale=bool(self.allow_scale_check.isChecked()),
            common_scale=bool(self.common_scale_check.isChecked()),
            loss=self.loss_combo.currentText(),
            min_separation=float(self.min_sep_spin.value()),
            separation_penalty_strength=(
                self._restored_fit_config.separation_penalty_strength
                if self._restored_fit_config
                else FitConfig().separation_penalty_strength
            ),
        )

    def _uncertainty_config_from_widgets(self) -> UncertaintyConfig:
        return UncertaintyConfig(
            block_size=self.parse_block_size(),
            random_seed=int(self.seed_spin.value()),
            run_bootstrap=bool(self.run_boot_check.isChecked()),
            n_boot=int(self.n_boot_spin.value()),
            run_template_propagation=bool(self.run_tpl_prop_check.isChecked()),
            n_template_prop=int(self.tpl_prop_spin.value()),
        )

    # ----------------------------- Template actions ---------------------------

    def start_build_template(self) -> None:
        try:
            x, y = self.selected_region_xy()
            config = self._template_config_from_widgets()
            payload = {"x": x, "y": y, **asdict(config)}
            self._pending_template_config = config
            self._pending_template_region = self.selected_region_bounds()
            self._pending_template_harmonic = self.active_harmonic()
            self._pending_template_harmonic_expression = self.active_harmonic_expression()
            self.set_busy(True, "Building template...")
            self.worker = AnalysisWorker("build_template", payload)
            self.worker.progress.connect(self.status)
            self.worker.finished_ok.connect(self.finish_build_template)
            self.worker.failed.connect(self.worker_failed)
            self.worker.start()
        except Exception as exc:
            self.show_error(str(exc))

    def finish_build_template(self, result: dict[str, Any]) -> None:
        self.set_busy(False, "Template built.")
        tpl = result["template_nominal"]
        lo, hi = self._pending_template_region
        entry = {
            "name": tpl.name,
            "harmonic": self._pending_template_harmonic,
            "harmonic_expression": self._pending_template_harmonic_expression,
            "combination": None,
            "source_leaf_ids": [len(self.templates)],
            "config": self._pending_template_config,
            "region_lo": lo,
            "region_hi": hi,
            "template": tpl,
            "template_bank": result["templates"],
            "summary": result["summary"],
            "settings": result.get("settings", {}),
            "edge_info": getattr(tpl, "edge_info", {}),
            "residual_model": result.get("residual_model", {}),
            "full": result,
        }
        self.templates.append(entry)
        self.template_combo.addItem(f"{len(self.templates)}: {tpl.name} ({len(result['templates'])} boot)")
        self.template_combo.setCurrentIndex(len(self.templates) - 1)
        self.refresh_profile_sources()
        self.show_template_summary(entry)
        self.show_active_template()

    def active_template_entry(self) -> dict[str, Any]:
        idx = self.template_combo.currentIndex()
        if idx < 0 or idx >= len(self.templates):
            raise ValueError("No active template. Build a template first.")
        return self.templates[idx]

    def show_active_template(self) -> None:
        if not self.templates:
            return
        try:
            entry = self.active_template_entry()
            tpl = entry["template"]
            self.lower_plot.setLabel("bottom", "Frequency relative to COG", units="Hz")
            self.lower_plot.setLabel("left", "Normalized density")
            self.lower_curve.setData(tpl.u, tpl.pdf)
            self.lower_curve_2.setData([], [])
            self.show_template_summary(entry)
        except Exception:
            pass

    def show_template_summary(self, entry: dict[str, Any]) -> None:
        tpl = entry["template"]
        qs = tpl.quantile([0.16, 0.50, 0.84])
        lines = [
            f"Template: {tpl.name}",
            f"Region: {entry.get('region_lo', np.nan):.9g} to {entry.get('region_hi', np.nan):.9g} Hz",
            f"Raw area: {tpl.raw_area:.9g}",
            f"Raw COG:  {tpl.raw_cog:.9f} Hz",
            f"Raw std:  {tpl.raw_std:.9g} Hz",
            f"Centered std from moments: {get_template_std(tpl):.9g} Hz",
            f"Centered q16/q50/q84: {qs[0]:.9g}, {qs[1]:.9g}, {qs[2]:.9g} Hz",
            f"sigma68 = {(qs[2] - qs[0]) / 2:.9g} Hz",
            f"Bootstrap templates: {len(entry['template_bank'])}",
        ]
        edge_info = entry.get("edge_info", getattr(tpl, "edge_info", {}))
        if edge_info:
            lines += [
                f"Edge mode: {edge_info.get('edge_mode', 'none')}",
                f"Edge width: {edge_info.get('edge_width_hz', 0.0):.9g} Hz",
                f"Zero-anchor width: {edge_info.get('zero_anchor_width_hz', 0.0):.9g} Hz",
                f"Left/right edge samples used: {edge_info.get('left_edge_points_used', 0)}, {edge_info.get('right_edge_points_used', 0)}",
                f"Left/right zero distances: {edge_info.get('left_tail_width_hz', 0.0):.9g}, {edge_info.get('right_tail_width_hz', 0.0):.9g} Hz",
                f"Edge baseline subtracted: {edge_info.get('edge_baseline_subtracted', 0.0):.9g}",
                f"Left tail model: {edge_info.get('left_tail_model', 'none')}",
                f"Right tail model: {edge_info.get('right_tail_model', 'none')}",
                f"Left/right tail area added: {edge_info.get('left_tail_area_added', 0.0):.9g}, {edge_info.get('right_tail_area_added', 0.0):.9g}",
            ]
        summary = entry.get("summary", {})
        for key in ["raw_cog", "raw_std", "raw_area", "q16", "q50", "q84"]:
            if key in summary:
                s = summary[key]
                lines.append(f"{key:>8}: {s['median']:.9g} -{s['minus']:.3g} +{s['plus']:.3g}  N={s['n']}")
        self.summary_text.setPlainText("\n".join(lines))

    # ------------------------------- Fitting ---------------------------------

    def parse_init_mus(self) -> list[float] | None:
        txt = self.init_mus_edit.text().strip()
        if not txt:
            return None
        vals = [float(v.strip()) for v in txt.split(",") if v.strip()]
        if not vals:
            return None
        return vals

    def parse_mu_bounds_for_fit(self, n_peaks: int, init_mus: list[float] | None) -> list[tuple[float, float]] | None:
        """Parse hard centroid search ranges from the GUI.

        Preferred syntax:
            lo0:hi0; lo1:hi1; lo2:hi2
        in Hz. Ranges define component identity and must be ordered.

        If the range field is empty but Initial mus and Auto range half-width
        are provided, ranges are generated as mu_i +/- half_width.
        """
        txt = self.mu_ranges_edit.text().strip()
        if txt:
            parts = [p.strip() for p in txt.split(";") if p.strip()]
            if len(parts) != n_peaks:
                raise ValueError(f"Expected {n_peaks} centroid ranges separated by semicolons; got {len(parts)}.")
            ranges: list[tuple[float, float]] = []
            for part in parts:
                if ":" in part:
                    lo_s, hi_s = part.split(":", 1)
                elif ".." in part:
                    lo_s, hi_s = part.split("..", 1)
                else:
                    raise ValueError("Each centroid range must use 'lo:hi' or 'lo..hi'.")
                lo = float(lo_s.strip())
                hi = float(hi_s.strip())
                if hi < lo:
                    lo, hi = hi, lo
                ranges.append((lo, hi))
            # Validation with min separation is repeated in the numerical core.
            return ranges

        half = float(self.mu_range_halfwidth_spin.value())
        if half > 0.0:
            if init_mus is None or len(init_mus) != n_peaks:
                raise ValueError("Auto range half-width requires Initial mus CSV with one value per component.")
            return [(float(m) - half, float(m) + half) for m in init_mus]

        return None

    def next_default_fit_label(self) -> str:
        return f"fit_{len(self.fit_records) + 1}"

    def start_fit_region(self) -> None:
        try:
            x, y = self.selected_region_xy()
            n_peaks = int(self.n_peaks_spin.value())
            component_ids = self._fit_component_template_ids(n_peaks)
            if component_ids is None:
                tpl_entry = transfer_template_entry(self.active_template_entry(), self.active_harmonic())
                entries = None
            else:
                cache = {
                    idx: transfer_template_entry(self.templates[idx], self.active_harmonic())
                    for idx in set(component_ids)
                }
                entries = [cache[idx] for idx in component_ids]
                tpl_entry = entries[0]
            init_mus = self.parse_init_mus()
            mu_bounds = self.parse_mu_bounds_for_fit(n_peaks=n_peaks, init_mus=init_mus)
            fit_config = self._fit_config_from_widgets(n_peaks, init_mus, mu_bounds)
            unc_config = self._uncertainty_config_from_widgets()
            payload = {
                "x": x,
                "y": y,
                "template": tpl_entry["template"],
                "template_bank": tpl_entry["template_bank"],
                "component_templates": [e["template"] for e in entries] if entries is not None else None,
                "component_template_banks": [e["template_bank"] for e in entries] if entries is not None else None,
                **asdict(fit_config),
                **asdict(unc_config),
            }
            self._pending_fit_config = fit_config
            self._pending_uncertainty_config = unc_config
            self._pending_template_index = component_ids[0] if component_ids else self.template_combo.currentIndex()
            self._pending_component_template_ids = component_ids
            self._pending_component_entries = entries
            self._pending_region_hz = self.selected_region_bounds()
            self._pending_target_harmonic = self.active_harmonic()
            self._pending_target_harmonic_expression = self.active_harmonic_expression()
            self._pending_fit_view = self.active_view_index
            self.set_busy(True, "Fitting selected region...")
            self.worker = AnalysisWorker("fit_region", payload)
            self.worker.progress.connect(self.status)
            self.worker.finished_ok.connect(self.finish_fit_region)
            self.worker.failed.connect(self.worker_failed)
            self.worker.start()
        except Exception as exc:
            self.show_error(str(exc))

    def finish_fit_region(self, result: dict[str, Any]) -> None:
        self.set_busy(False, "Fit finished.")
        result["target_harmonic_expression"] = self._pending_target_harmonic_expression
        result["component_template_ids"] = self._pending_component_template_ids
        result["component_template_entries"] = self._pending_component_entries
        self.last_result = result
        self.last_fit_config = self._pending_fit_config
        self.last_uncertainty_config = self._pending_uncertainty_config
        self.last_template_index = self._pending_template_index
        self.last_region_hz = self._pending_region_hz
        self.last_target_harmonic = self._pending_target_harmonic
        self.last_view_index = self._pending_fit_view
        self.plot_fit_overlay(result["fit"])
        self.fill_results_table(result)
        self.fill_fit_summary(result)
        self.control_tabs.setCurrentIndex(4)
        if self.auto_store_check.isChecked():
            self.store_current_fit()
            self.fit_label_edit.setText(self.next_default_fit_label())

    def clear_fit_overlay(self) -> None:
        self.fit_curve.setData([], [])
        self.secondary_fit_curve.setData([], [])
        self.background_curve.setData([], [])
        self.secondary_background_curve.setData([], [])
        for c in self.component_curves:
            self._overlay_plot.removeItem(c)
        self.component_curves = []
        self.lower_curve.setData([], [])
        self.lower_curve_2.setData([], [])

    def plot_fit_overlay(self, fit: dict[str, Any]) -> None:
        self.clear_fit_overlay()
        x = fit["x"]
        self._overlay_plot = self._view_plot(self.last_view_index)
        fit_curve = self.fit_curve if self.last_view_index == 0 else self.secondary_fit_curve
        background_curve = self.background_curve if self.last_view_index == 0 else self.secondary_background_curve
        if self.last_view_index == 1:
            self.show_secondary_check.setChecked(True)
        fit_curve.setData(x, fit["y_model"])
        background_curve.setData(x, fit["background"])
        colors = ["g", "y", "c", "m", "w", "b"]
        for k, comp in enumerate(fit["components"]):
            pen = pg.mkPen(colors[k % len(colors)], width=1.5, style=QtCore.Qt.PenStyle.DashLine)
            curve = self._overlay_plot.plot(x, comp + fit["background"], pen=pen, name=f"component {k}")
            self.component_curves.append(curve)

        self.lower_plot.setLabel("bottom", "Frequency", units="Hz")
        self.lower_plot.setLabel("left", "Residual")
        residual = fit["y"] - fit["y_model"]
        self.lower_curve.setData(x, residual)
        self.lower_curve_2.setData(x, np.zeros_like(x))

    # ----------------------------- Results display ----------------------------

    @staticmethod
    def fmt(v: Any, nd: int = 7) -> str:
        try:
            v = float(v)
            if not np.isfinite(v):
                return "—"
            return f"{v:.{nd}g}"
        except Exception:
            return str(v)

    @staticmethod
    def fmt_interval(summary: dict[str, float] | None, nd: int = 7) -> str:
        if not summary or summary.get("n", 0) == 0:
            return "—"
        return f"{summary['median']:.{nd}g} -{summary['minus']:.2g} +{summary['plus']:.2g}"

    def fill_results_table(self, result: dict[str, Any]) -> None:
        fit = result["fit"]
        boot = result.get("bootstrap")
        tpl_prop = result.get("template_propagation")
        n = fit["n_peaks"]
        self.results_table.setRowCount(n)
        qrows = component_quantiles_from_fit(fit)

        for k in range(n):
            mu_name = f"mu_{k}"
            width_name = f"width_{k}"
            mus = fit["info"]["mus"]
            areas = fit["info"]["areas"]
            scales = fit["info"]["scales"]
            widths = component_widths_from_params(
                fit["params"], fit["param_names"], fit["template"], fit["n_peaks"], fit.get("component_templates")
            )

            boot_mu = boot["summary"].get(mu_name) if boot else None
            tpl_mu = tpl_prop["summary"].get(mu_name) if tpl_prop else None
            boot_w = boot["width_summary"].get(width_name) if boot else None
            tpl_w = tpl_prop["width_summary"].get(width_name) if tpl_prop else None

            values = [
                f"C{k + 1}",
                self.fmt(mus[k], 10),
                self.fmt_interval(boot_mu, 10),
                self.fmt_interval(tpl_mu, 10),
                self.fmt(combined_sigma(boot_mu, tpl_mu), 6),
                self.fmt(areas[k], 8),
                self.fmt(scales[k], 8),
                self.fmt(widths[k], 8),
                self.fmt_interval(boot_w, 8),
                self.fmt_interval(tpl_w, 8),
                self.fmt(combined_sigma(boot_w, tpl_w), 6),
                self.fmt(qrows[k]["sigma68_Hz"], 8),
                self.fmt(fit["red_chi2"], 6),
            ]
            for col, text in enumerate(values):
                item = QtWidgets.QTableWidgetItem(text)
                if fit.get("component_templates"):
                    item.setToolTip(f"C{k + 1}: {fit['component_templates'][k].name}")
                self.results_table.setItem(k, col, item)
        self.results_table.resizeColumnsToContents()

    def fill_fit_summary(self, result: dict[str, Any]) -> None:
        fit = result["fit"]
        boot = result.get("bootstrap")
        tpl_prop = result.get("template_propagation")
        qrows = component_quantiles_from_fit(fit)

        lines = []
        lines.append("FIT SUMMARY")
        lines.append("=" * 72)
        lines.append(f"Success: {fit['success']}   Message: {fit['message']}")
        lines.append(f"chi2/dof = {fit['chi2']:.6g} / {fit['dof']} = {fit['red_chi2']:.6g}")
        if fit.get("mu_bounds"):
            lines.append(f"Centroid search ranges [Hz]: {fit.get('mu_bounds_text', '')}")
        if fit.get("min_separation", 0.0) > 0:
            lines.append(
                f"Minimum separation requested = {fit.get('min_separation'):.9g} Hz; observed = {fit.get('min_observed_separation', np.nan):.9g} Hz"
            )
        if fit["red_chi2"] > 5:
            lines.append("WARNING: residuals are strongly non-white and/or model mismatch is present.")
            lines.append("Use bootstrap/template propagation as primary uncertainty.")
        lines.append("")

        widths = component_widths_from_params(
            fit["params"], fit["param_names"], fit["template"], fit["n_peaks"], fit.get("component_templates")
        )
        for k in range(fit["n_peaks"]):
            tpl = (fit.get("component_templates") or [fit["template"]] * fit["n_peaks"])[k]
            lines.append(f"Peak C{k + 1} — template: {tpl.name}")
            lines.append(f"  COG fit       = {fit['info']['mus'][k]:.12g} Hz")
            lines.append(f"  area fit      = {fit['info']['areas'][k]:.12g}")
            lines.append(f"  scale fit     = {fit['info']['scales'][k]:.12g}")
            lines.append(f"  RMS width fit = {widths[k]:.12g} Hz")
            lines.append(
                f"  q16/q50/q84   = {qrows[k]['q16_Hz']:.12g}, {qrows[k]['q50_Hz']:.12g}, {qrows[k]['q84_Hz']:.12g} Hz"
            )
            lines.append(f"  sigma68       = {qrows[k]['sigma68_Hz']:.12g} Hz")

            for label, obj in [("bootstrap", boot), ("template-shape", tpl_prop)]:
                if not obj:
                    continue
                mu_s = obj.get("summary", {}).get(f"mu_{k}")
                w_s = obj.get("width_summary", {}).get(f"width_{k}")
                if mu_s:
                    lines.append(
                        f"  COG {label:14s}= {mu_s['median']:.12g} -{mu_s['minus']:.3g} +{mu_s['plus']:.3g} Hz"
                    )
                if w_s:
                    lines.append(f"  width {label:12s}= {w_s['median']:.12g} -{w_s['minus']:.3g} +{w_s['plus']:.3g} Hz")
            if boot or tpl_prop:
                lines.append(
                    f"  COG combined sigma = {combined_sigma(boot.get('summary', {}).get(f'mu_{k}') if boot else None, tpl_prop.get('summary', {}).get(f'mu_{k}') if tpl_prop else None):.6g} Hz"
                )
                lines.append(
                    f"  width combined sigma = {combined_sigma(boot.get('width_summary', {}).get(f'width_{k}') if boot else None, tpl_prop.get('width_summary', {}).get(f'width_{k}') if tpl_prop else None):.6g} Hz"
                )
            lines.append("")

        if boot:
            lines.append(
                f"Bootstrap samples succeeded: {len(boot['params'])}, block_size={boot['block_size']}, tau_int={boot['tau_int']:.3g}"
            )
        if tpl_prop:
            lines.append(f"Template propagation samples succeeded: {len(tpl_prop['params'])}")

        if fit["n_peaks"] > 1:
            lines.append("")
            lines.append("PAIRWISE / OVERLAP DIAGNOSTICS")
            for k in range(1, fit["n_peaks"]):
                d_fit = fit["info"]["mus"][k] - fit["info"]["mus"][k - 1]
                boot_d = summarize_mu_delta_samples(boot, k, k - 1)
                tpl_d = summarize_mu_delta_samples(tpl_prop, k, k - 1)
                ratio_fit = fit["info"]["areas"][k] / fit["info"]["areas"][0] if fit["info"]["areas"][0] > 0 else np.nan
                boot_r = summarize_area_ratio_samples(boot, k, 0, fit["n_peaks"])
                tpl_r = summarize_area_ratio_samples(tpl_prop, k, 0, fit["n_peaks"])
                lines.append(
                    f"  Δmu[{k}-{k - 1}] fit = {d_fit:.12g} Hz, combined σ = {combined_sigma(boot_d, tpl_d):.6g} Hz"
                )
                lines.append(
                    f"  area[{k}]/area[0] fit = {ratio_fit:.12g}, combined σ = {combined_sigma(boot_r, tpl_r):.6g}"
                )
                lines.append(
                    f"  mu correlation bootstrap with previous = {summarize_mu_corr_samples(boot, k, k - 1):.4g}"
                )
        lines.append("")
        lines.append("VALID SAMPLE COUNTS")
        lines.append(f"  bootstrap valid fits = {sample_count(boot)}")
        lines.append(f"  template-propagation valid fits = {sample_count(tpl_prop)}")
        lines.append(f"  quality flag = {quality_flag_for_fit(fit, boot=boot, tpl_prop=tpl_prop)}")

        self.summary_text.setPlainText("\n".join(lines))

    # -------------------------- Fit history / export --------------------------

    def store_current_fit(self) -> None:
        if not self.last_result:
            self.show_error("No current fit to store.")
            return
        label = self.fit_label_edit.text().strip() or self.next_default_fit_label()
        notes = ""
        rows = make_export_rows_for_result(
            self.last_result, label=label, template_registry=self._registry_for_result(self.last_result), notes=notes
        )
        record = {
            "label": label,
            "notes": notes,
            "result": self.last_result,
            "rows": rows,
            "fit_config": self.last_fit_config,
            "uncertainty_config": self.last_uncertainty_config,
            "template_index": self.last_template_index,
            "region_hz": self.last_region_hz,
            "target_harmonic": self.last_target_harmonic,
            "view_index": self.last_view_index,
        }
        self.fit_records.append(record)
        self.refresh_history_table(select_last=True)
        self.tabs.setCurrentIndex(1)
        self.status(f"Stored fit as '{label}'.")

    def _registry_for_result(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        if result.get("component_template_entries") is not None:
            return [*result["component_template_entries"], *self.templates]
        if 0 <= self.last_template_index < len(self.templates):
            entry = transfer_template_entry(self.templates[self.last_template_index], self.last_target_harmonic)
            entry["template"] = result["fit"]["template"]
            return [entry, *self.templates]
        return self.templates

    def _current_form_state(self) -> FormState:
        n_peaks = int(self.n_peaks_spin.value())
        init_mus = self.parse_init_mus()
        mu_bounds = self.parse_mu_bounds_for_fit(n_peaks=n_peaks, init_mus=init_mus)
        return FormState(
            template=self._template_config_from_widgets(),
            fit=self._fit_config_from_widgets(n_peaks, init_mus, mu_bounds),
            fit_label=self.fit_label_edit.text().strip() or self.next_default_fit_label(),
            auto_store=bool(self.auto_store_check.isChecked()),
            uncertainty=self._uncertainty_config_from_widgets(),
            region_hz=self.selected_region_bounds(),
            active_template_id=self.template_combo.currentIndex() if self.templates else None,
            view_state=self._view_state_for_session(),
        )

    def _session_from_state(self) -> Session:
        if self.current_file is None:
            raise ValueError("No data file loaded.")
        sha256 = _sha256_of(self.current_file)
        if self._data_sha256 is not None and sha256 != self._data_sha256:
            raise ValueError("The loaded NPZ changed on disk. Reload it before saving a reproducible configuration.")
        templates = [
            TemplateRecord(
                id=idx,
                region_hz=(entry["region_lo"], entry["region_hi"]),
                config=entry["config"],
                verify=_template_verify_from_entry(entry),
                harmonic=float(entry.get("harmonic", 1.0)),
                harmonic_expression=entry.get("harmonic_expression", ""),
                combination=entry.get("combination"),
            )
            for idx, entry in enumerate(self.templates)
        ]
        records = list(self.fit_records)
        if self.last_result is not None and not any(rec["result"] is self.last_result for rec in records):
            label = self.fit_label_edit.text().strip() or self.next_default_fit_label()
            records.append(
                {
                    "label": label,
                    "notes": "",
                    "template_index": self.last_template_index,
                    "result": self.last_result,
                    "region_hz": self.last_region_hz,
                    "target_harmonic": self.last_target_harmonic,
                    "view_index": self.last_view_index,
                    "fit_config": self.last_fit_config,
                    "uncertainty_config": self.last_uncertainty_config,
                    "rows": make_export_rows_for_result(
                        self.last_result, label=label, template_registry=self._registry_for_result(self.last_result)
                    ),
                }
            )
        for rec in records:
            if rec["fit_config"] is None or rec["uncertainty_config"] is None:
                raise ValueError(f"Fit '{rec['label']}' has no recorded fit configuration; cannot save session.")
        fits = [
            FitRecord(
                label=rec["label"],
                notes=rec["notes"],
                template_id=rec["template_index"],
                region_hz=rec["region_hz"],
                fit_config=rec["fit_config"],
                uncertainty_config=rec["uncertainty_config"],
                verify_rows=rec["rows"],
                target_harmonic=float(rec.get("target_harmonic", 1.0)),
                view_index=int(rec.get("view_index", 0)),
                component_template_ids=rec.get("result", {}).get("component_template_ids"),
                target_harmonic_expression=rec.get("result", {}).get("target_harmonic_expression", ""),
            )
            for rec in records
        ]
        return Session(
            schema_version=SCHEMA_VERSION,
            ssa_version=SSA_VERSION,
            created_utc=datetime.now(timezone.utc).isoformat(),
            input=InputRef(
                path=str(self.current_file.resolve()),
                sha256=sha256,
                frequency_key=self._frequency_key,
                amplitude_key=self._amplitude_key,
            ),
            form_state=self._current_form_state(),
            templates=templates,
            fits=fits,
            comparisons=[run["record"] for run in self.comparison_runs],
        )

    def _choose_output_parent(self) -> None:
        parent = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output parent folder")
        if parent:
            self.output_parent_edit.setText(parent)

    def _save_run_bundle(self) -> Path | None:
        session = self._session_from_state()
        parent = self.output_parent_edit.text().strip()
        if not parent:
            self._choose_output_parent()
            parent = self.output_parent_edit.text().strip()
        if not parent:
            return None
        label = self.bundle_label_edit.text().strip()
        state = _to_toml_value(asdict(session))
        state.pop("created_utc", None)
        signature = hashlib.sha256(
            json.dumps(
                {
                    "parent": str(Path(parent).expanduser().resolve()),
                    "label": label,
                    "session": state,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if self._bundle_dir is None or not self._bundle_dir.exists() or signature != self._bundle_signature:
            out = create_run_directory(parent, "SchottkySA", label)
            _save_session(out / "config.toml", session)
            rows = [row for fit in session.fits for row in fit.verify_rows]
            if rows:
                export_history_rows(rows, out / "fit_history.csv", source_file=self.current_file)
            if self.comparison_runs:
                comparison_rows = [row for run in self.comparison_runs for row in run["record"].verify_rows]
                export_history_rows(comparison_rows, out / "profile_comparisons.csv", source_file=self.current_file)
            if self.last_result is not None:
                _export_current_overlay_npz(self.last_result["fit"], out / "current_fit_overlay.npz")
            self._bundle_dir, self._bundle_signature = out, signature
        self.bundle_path_label.setText(str(self._bundle_dir))
        return self._bundle_dir

    def save_session(self) -> None:
        try:
            out = self._save_run_bundle()
            if out is not None:
                self.status(f"Saved configuration and available tables to {out}")
        except Exception as exc:
            self.show_error(f"Save failed:\n{exc}")

    def load_session(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Replay saved configuration", "", "TOML (*.toml);;All files (*)"
        )
        if not path:
            return
        if self.templates or self.fit_records:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Load session",
                "Loading a session replaces the current templates and stored fits. Continue?",
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        try:
            session = _load_session(path)
        except Exception as exc:
            self.show_error(f"Could not read session file:\n{exc}")
            return
        override = None
        if not Path(session.input.path).is_file():
            override, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Locate original NPZ input", "", "NPZ (*.npz)")
            if not override:
                return
        self.set_busy(True, "Replaying session...")
        self._loading_session = session
        self.worker = SessionReplayWorker(session, data_path_override=override)
        self.worker.progress.connect(self.status)
        self.worker.finished_ok.connect(self._finish_load_session)
        self.worker.failed.connect(self.worker_failed)
        self.worker.start()

    def _finish_load_session(self, report: ReplayReport) -> None:
        self.set_busy(False, "Session loaded.")
        self._apply_replay_report(report, self._loading_session)

    def _apply_replay_report(self, report: ReplayReport, session: Session) -> None:
        self.frequency = report.frequency
        self.amplitude = report.amplitude
        self.current_file = report.data_path
        self._data_sha256 = _sha256_of(report.data_path)
        self._frequency_key = session.input.frequency_key
        self._amplitude_key = session.input.amplitude_key
        self.file_label.setText(
            f"{report.data_path.name}\n"
            f"frequency: {self._frequency_key}, amplitude: {self._amplitude_key}\n"
            f"N={len(self.frequency):,}"
        )
        self.data_curve.setData(self.frequency, self.amplitude)
        self.plot.setXRange(float(self.frequency.min()), float(self.frequency.max()), padding=0.02)
        self.plot.setYRange(float(np.nanmin(self.amplitude)), float(np.nanmax(self.amplitude)), padding=0.05)

        self.templates = report.templates
        self.comparison_runs = report.comparisons
        self._clear_profile_displays()
        self.template_combo.clear()
        for idx, entry in enumerate(self.templates):
            self.template_combo.addItem(f"{idx + 1}: {entry['name']} ({len(entry['template_bank'])} boot)")
        if self.templates:
            self.template_combo.setCurrentIndex(len(self.templates) - 1)

        self.fit_records = report.fits
        self.refresh_history_table()
        self.clear_fit_overlay()
        self.last_result = None
        self.last_fit_config = None
        self.last_uncertainty_config = None
        if self.fit_records:
            last_record = self.fit_records[-1]
            last_result = last_record["result"]
            self.last_result = last_result
            self.last_fit_config = last_record["fit_config"]
            self.last_uncertainty_config = last_record["uncertainty_config"]
            self.last_template_index = last_record["template_index"]
            self.last_region_hz = last_record["region_hz"]
            self.last_target_harmonic = last_record.get("target_harmonic", 1.0)
            self.last_view_index = last_record.get("view_index", 0)
            self.plot_fit_overlay(last_result["fit"])

        fs = session.form_state
        self._restored_template_config, self._restored_fit_config = fs.template, fs.fit
        if fs.active_template_id is not None:
            if not 0 <= fs.active_template_id < len(self.templates):
                raise ValueError("Saved active template id is not in this session")
            self.template_combo.setCurrentIndex(fs.active_template_id)
        self.template_name_edit.setText(fs.template.name)
        self.template_boot_spin.setValue(fs.template.n_template_boot)
        self.resample_spin.setValue(fs.template.resample_factor)
        self.smooth_check.setChecked(fs.template.smooth)
        self.clip_check.setChecked(fs.template.clip_negative)
        self.sg_window_spin.setValue(fs.template.sg_window or 0)
        self.edge_mode_combo.setCurrentText(fs.template.edge_mode)
        self.edge_width_spin.setValue(fs.template.edge_width_hz)
        self.edge_fraction_spin.setValue(fs.template.edge_fraction)
        for field, widget in self.edge_side_spins.items():
            widget.setValue(getattr(fs.template, field))

        self.fit_label_edit.setText(fs.fit_label)
        self.n_peaks_spin.setValue(fs.fit.n_peaks)
        self.background_order_spin.setValue(fs.fit.background_order)
        self.allow_scale_check.setChecked(fs.fit.allow_scale)
        self.common_scale_check.setChecked(fs.fit.common_scale)
        self.loss_combo.setCurrentText(fs.fit.loss)
        self.init_mus_edit.setText(", ".join(str(v) for v in fs.fit.init_mus) if fs.fit.init_mus else "")
        self.mu_ranges_edit.setText("; ".join(f"{lo}:{hi}" for lo, hi in fs.fit.mu_bounds) if fs.fit.mu_bounds else "")
        self.min_sep_spin.setValue(fs.fit.min_separation)
        self.auto_store_check.setChecked(fs.auto_store)

        self.block_size_edit.setText(str(fs.uncertainty.block_size))
        self.run_boot_check.setChecked(fs.uncertainty.run_bootstrap)
        self.n_boot_spin.setValue(fs.uncertainty.n_boot)
        self.run_tpl_prop_check.setChecked(fs.uncertainty.run_template_propagation)
        self.tpl_prop_spin.setValue(fs.uncertainty.n_template_prop)
        self.seed_spin.setValue(fs.uncertainty.random_seed)
        self.region.setRegion(list(fs.region_hz))
        self._restore_profile_view_state(fs.view_state)
        if self.comparison_runs:
            self.show_profile_comparison(self.comparison_runs[-1]["directions"])
        self.control_tabs.setCurrentIndex(4)

        lines = [f"Loaded session: {len(self.templates)} template(s), {len(self.fit_records)} fit(s)."]
        lines.extend(f"WARNING: {w}" for w in report.input_warnings)
        if report.ok:
            lines.append("All saved values reproduced within tolerance.")
        else:
            lines.append(f"{len(report.mismatches)} mismatch(es):")
            lines.extend(f"  {m}" for m in report.mismatches)
        self.summary_text.setPlainText("\n".join(lines))
        self.tabs.setCurrentIndex(2)
        if report.ok:
            self.status("Session loaded and verified.")
        else:
            self.status("Session replay requires review — see Summary tab.")
            QtWidgets.QMessageBox.warning(
                self,
                "Session verification",
                f"{len(report.input_warnings)} input warning(s); {len(report.mismatches)} numerical difference(s). "
                "See the Summary tab for details.",
            )

    def refresh_history_table(self, select_last: bool = False) -> None:
        self._updating_history_table = True
        try:
            all_rows = []
            for rec_idx, rec in enumerate(self.fit_records):
                for row_idx, row in enumerate(rec["rows"]):
                    all_rows.append((rec_idx, row_idx, row))

            self.history_table.setRowCount(len(all_rows))
            for table_row, (rec_idx, row_idx, row) in enumerate(all_rows):
                for col, key in enumerate(self.history_columns):
                    value = row.get(key, "")
                    text = self.format_cell_for_table(value)
                    item = QtWidgets.QTableWidgetItem(text)
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, (rec_idx, row_idx))
                    if key in ("label", "notes"):
                        item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
                    else:
                        item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                    self.history_table.setItem(table_row, col, item)
            self.history_table.resizeColumnsToContents()

            if select_last and self.history_table.rowCount() > 0:
                self.history_table.selectRow(self.history_table.rowCount() - 1)
        finally:
            self._updating_history_table = False

    @staticmethod
    def format_cell_for_table(value: Any) -> str:
        if value is None:
            return ""
        try:
            v = float(value)
            if not np.isfinite(v):
                return ""
            if abs(v) >= 1e5 or (abs(v) < 1e-3 and v != 0):
                return f"{v:.10e}"
            return f"{v:.10g}"
        except Exception:
            return str(value)

    def on_history_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._updating_history_table:
            return
        data = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not data:
            return
        rec_idx, row_idx = data
        if not (0 <= rec_idx < len(self.fit_records)):
            return
        col = item.column()
        key = self.history_columns[col]
        if key not in ("label", "notes"):
            return

        new_value = item.text()
        rec = self.fit_records[rec_idx]
        if key == "label":
            rec["label"] = new_value
            for row in rec["rows"]:
                row["label"] = new_value
        elif key == "notes":
            rec["notes"] = new_value
            for row in rec["rows"]:
                row["notes"] = new_value
        self.refresh_history_table()

    def show_selected_history_fit(self) -> None:
        if self._updating_history_table:
            return
        rows = self.history_table.selectionModel().selectedRows()
        if not rows:
            return
        item = self.history_table.item(rows[0].row(), 0)
        if not item:
            return
        data = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not data:
            return
        rec_idx, _ = data
        if 0 <= rec_idx < len(self.fit_records):
            record = self.fit_records[rec_idx]
            result = record["result"]
            previous_view = self.last_view_index
            self.last_view_index = record.get("view_index", 0)
            self.plot_fit_overlay(result["fit"])
            self.last_view_index = previous_view

    def selected_history_rows(self) -> list[dict[str, Any]]:
        selected = self.history_table.selectionModel().selectedRows()
        if not selected:
            return []
        out = []
        seen = set()
        for idx in selected:
            item = self.history_table.item(idx.row(), 0)
            if not item:
                continue
            data = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if not data:
                continue
            rec_idx, row_idx = data
            key = (rec_idx, row_idx)
            if key in seen:
                continue
            seen.add(key)
            out.append(self.fit_records[rec_idx]["rows"][row_idx])
        return out

    def all_history_rows(self) -> list[dict[str, Any]]:
        rows = []
        for rec in self.fit_records:
            rows.extend(rec["rows"])
        return rows

    def copy_selected_history_rows(self) -> None:
        rows = self.selected_history_rows()
        if not rows:
            self.show_error("No stored rows selected.")
            return
        QtWidgets.QApplication.clipboard().setText(rows_to_tsv(rows))
        self.status(f"Copied {len(rows)} row(s) to clipboard as TSV.")

    def copy_all_history_rows(self) -> None:
        rows = self.all_history_rows()
        if not rows:
            self.show_error("No stored fit rows to copy.")
            return
        QtWidgets.QApplication.clipboard().setText(rows_to_tsv(rows))
        self.status(f"Copied {len(rows)} stored row(s) to clipboard as TSV.")

    def delete_selected_history_rows(self) -> None:
        selected = self.history_table.selectionModel().selectedRows()
        if not selected:
            return
        rec_indices = set()
        for idx in selected:
            item = self.history_table.item(idx.row(), 0)
            if item:
                data = item.data(QtCore.Qt.ItemDataRole.UserRole)
                if data:
                    rec_indices.add(data[0])
        if not rec_indices:
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "Delete stored fits",
            f"Delete {len(rec_indices)} stored fit record(s)?",
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.fit_records = [rec for i, rec in enumerate(self.fit_records) if i not in rec_indices]
        self.refresh_history_table()
        self.status("Deleted selected fit record(s).")

    def export_current_overlay_npz(self) -> None:
        """Save overlay together with its reproducible configuration and tables."""
        if not self.last_result:
            self.show_error("No current fit to export.")
            return
        self.save_session()

    def export_history(self) -> None:
        try:
            out = self._save_run_bundle()
            if out is None:
                return
            rows = [row for fit in self._session_from_state().fits for row in fit.verify_rows]
            if not rows:
                self.show_error("No fitted results to export. Configuration was saved.")
                return
            extension = self.table_format_combo.currentText().lower()
            path = out / f"fit_history.{extension}"
            if not path.exists():
                export_history_rows(rows, path, source_file=self.current_file)
            self.status(f"Configuration and {extension.upper()} tables saved to {out}")
        except Exception as exc:
            self.show_error(f"Export failed:\n{exc}")

    # -------------------------------- Utilities -------------------------------

    def set_busy(self, busy: bool, message: str = "") -> None:
        for w in [
            self.load_btn,
            self.build_template_btn,
            self.fit_btn,
            self.store_current_btn,
            self.copy_selected_btn,
            self.copy_all_btn,
            self.export_history_btn,
            self.export_overlay_btn,
            self.delete_selected_btn,
            self.save_session_action,
            self.load_session_action,
            self.save_config_btn,
            self.replay_config_btn,
            self.output_browse_btn,
            self.combine_profiles_btn,
            self.compare_profiles_btn,
            self.preview_profiles_btn,
            self.pick_components_check,
            self.draw_components_btn,
            self.remove_component_btn,
            self.clear_components_btn,
            self.fit_components_btn,
            self.picker_halfwidth_spin,
            self.picker_component_combo,
            self.n_peaks_spin,
            self.init_mus_edit,
            self.mu_ranges_edit,
            self.mu_range_halfwidth_spin,
            self.min_sep_spin,
            *self.edge_side_spins.values(),
            self.use_component_templates_check,
            self.component_template_combo,
            self.same_template_btn,
        ]:
            w.setEnabled(not busy)
        for markers in self.component_markers:
            for marker in markers:
                marker["line"].setMovable(not busy)
                if marker["region"] is not None:
                    marker["region"].setMovable(not busy)
        if not busy:
            self._sync_component_template_choices()
        self.status(message or ("Busy..." if busy else "Ready"))

    def worker_failed(self, tb: str) -> None:
        self.set_busy(False, "Failed.")
        self.show_error(tb)

    def status(self, msg: str) -> None:
        self.status_label.setText(msg)
        self.statusBar().showMessage(msg)
        self.statusBar().showMessage(msg)

    def show_error(self, msg: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Error", msg)
        self.status("Error")

    def closeEvent(self, event: Any) -> None:
        if self.worker is not None and self.worker.isRunning():
            event.ignore()
            return
        super().closeEvent(event)
