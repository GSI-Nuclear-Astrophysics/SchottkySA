"""Component assignment and fractional harmonics on invented spectra only."""

import numpy as np
import pytest

from ssa.session import _sha256_of, load_session, replay_session, save_session
from test_profile_workbench_gui import await_worker
from test_profile_workbench_gui import workbench as workbench


def assign(win, component, template):
    win.picker_component_combo.setCurrentIndex(component)
    win.component_template_combo.setCurrentIndex(win.component_template_combo.findData(template))


def test_assignments_follow_markers_reorder_remove_and_view_switch(workbench):
    win, app, errors = workbench
    win.picker_halfwidth_spin.setValue(0.1)
    win.add_graphical_component(1259.4)
    win.add_graphical_component(1260.4)
    win.use_component_templates_check.setChecked(True)
    assign(win, 0, 0)
    assign(win, 1, 1)
    assert win._fit_component_template_ids(2) == [0, 1]
    left = win.component_markers[0][0]
    left["line"].setValue(1261)
    assert win._fit_component_template_ids(2) == [1, 0]
    win.active_view_combo.setCurrentIndex(1)
    assert not win.use_component_templates_check.isChecked()
    win.add_graphical_component(1270)
    win.use_component_templates_check.setChecked(True)
    assign(win, 0, 1)
    win.active_view_combo.setCurrentIndex(0)
    assert win._fit_component_template_ids(2) == [1, 0]
    win.picker_component_combo.setCurrentIndex(0)
    win.remove_component_btn.click()
    assert win._fit_component_template_ids(1) == [0]
    win.same_template_btn.click()
    assert win._fit_component_template_ids(1) is None
    assert not errors


@pytest.mark.parametrize("auto_store", [False, True])
def test_multi_template_fit_export_and_replay(workbench, tmp_path, auto_store):
    from ssa.config import TemplateBuildConfig
    from ssa.pipeline import build_template_bank

    win, app, errors = workbench
    x = np.linspace(-3, 3, 181)
    profiles = [
        np.exp(-0.5 * (x / 0.32) ** 2),
        np.exp(-0.5 * (x / 0.5) ** 2) + 0.5 * np.exp(-0.5 * ((x - 0.7) / 0.22) ** 2),
    ]
    pieces, entries = [], []
    for i, y in enumerate(profiles):
        f = x + 10 * i
        config = TemplateBuildConfig(
            name=f"shape_{i}", smooth=False, resample_factor=2, edge_mode="none", n_template_boot=8, random_seed=31 + i
        )
        bank = build_template_bank(f, y, config)
        entries.append(
            dict(
                name=config.name,
                template=bank["template_nominal"],
                template_bank=bank["templates"],
                config=config,
                harmonic=1.0,
                harmonic_expression="214/214",
                region_lo=float(f[0]),
                region_hi=float(f[-1]),
                source_leaf_ids=[i],
                settings=dict(smooth_nominal_template=False, resample_factor=2),
            )
        )
        pieces.append((f, y))
    target_x = np.linspace(26, 35, 271)
    target_y = sum(
        e["template"].evaluate_component(target_x, mu=m, area=a)
        for e, m, a in zip(entries, [30, 30.9], [4, 3], strict=True)
    )
    target_y += 0.0002 * np.sin(np.arange(len(target_x)) * 1.4)
    pieces.append((target_x, target_y))
    f, y = [np.concatenate([p[k] for p in pieces]) for k in (0, 1)]
    win.frequency, win.amplitude = f, y
    win.data_curve.setData(f, y)
    win.secondary_data_curve.setData(f, y)
    np.savez(win.current_file, frequency=f, amplitude=y)
    win._data_sha256 = _sha256_of(win.current_file)
    win.templates = entries
    win.template_combo.clear()
    for e in entries:
        win.template_combo.addItem(e["name"])
    win.refresh_profile_sources()
    win.view_harmonic_spins[0].setExpression("214/214")
    win.view_harmonic_spins[1].setExpression("215/214")
    win.combine_harmonic_spin.setExpression("214/214")
    win.region.setRegion((26, 35))
    win.secondary_region.setRegion((26, 35))
    win.allow_scale_check.setChecked(False)
    win.background_order_spin.setValue(-1)
    win.picker_halfwidth_spin.setValue(0.2)
    win.add_graphical_component(30.03)
    win.add_graphical_component(30.93)
    win.use_component_templates_check.setChecked(True)
    assign(win, 0, 0)
    assign(win, 1, 1)
    win.auto_store_check.setChecked(auto_store)
    win.start_fit_region()
    await_worker(win, app, errors)
    assert win.last_result["component_template_ids"] == [0, 1]
    assert win.last_result["fit"]["info"]["mus"] == pytest.approx([30, 30.9], abs=0.001)
    rows = win._registry_for_result(win.last_result)
    assert rows[0]["name"] == "shape_0"
    assert rows[1]["name"] == "shape_1"
    session = win._session_from_state()
    assert session.fits[0].component_template_ids == [0, 1]
    assert session.fits[0].target_harmonic_expression == "214/214"
    assert [r["template"] for r in session.fits[0].verify_rows] == ["shape_0", "shape_1"]
    path = tmp_path / "multi.toml"
    save_session(path, session)
    saved = load_session(path)
    report = replay_session(saved)
    assert report.ok, report.mismatches
    win._apply_replay_report(report, saved)
    assert win._fit_component_template_ids(2) == [0, 1]
    assert win.view_harmonic_spins[1].expression() == "215/214"
    assert win.templates[0]["harmonic_expression"] == "214/214"
    assert win._session_from_state().fits[0].component_template_ids == [0, 1]
    assert not errors


def test_fraction_input_transfer_and_saved_expression(workbench, tmp_path):
    win, app, errors = workbench
    # The spectrum coordinates are kept unchanged, but harmonic coordinates
    # are divided by the same common factor, as for pre-scaled data.
    for i, entry in enumerate(win.templates):
        entry["harmonic"] /= 126
        entry["harmonic_expression"] = f"{126 + i}/126"
    win.view_harmonic_spins[0].setExpression("126/126")
    win.view_harmonic_spins[1].setExpression("127/126")
    win.combine_harmonic_spin.setExpression("126/126")
    win.active_view_combo.setCurrentIndex(1)
    win.template_combo.setCurrentIndex(0)
    win.allow_scale_check.setChecked(False)
    win.init_mus_edit.setText("1270")
    win.start_fit_region()
    await_worker(win, app, errors)
    row = win.fit_records[-1]["rows"][0]
    assert row["harmonic_scale_factor"] == 127 / 126
    assert row["target_harmonic_expression"] == "127/126"
    assert row["template_harmonic_expression"] == "126/126"
    path = tmp_path / "fractions.toml"
    save_session(path, win._session_from_state())
    saved = load_session(path)
    report = replay_session(saved)
    assert report.ok, report.mismatches
    win._apply_replay_report(report, saved)
    assert win.active_harmonic() == 127 / 126
    assert win.view_harmonic_spins[1].expression() == "127/126"
    # Invalid partially typed fractions neither start a worker nor save silently.
    win.view_harmonic_spins[1].setText("127/0")
    win.start_fit_region()
    assert errors and "harmonic" in errors[-1].lower()
    with pytest.raises(ValueError):
        win._session_from_state()
