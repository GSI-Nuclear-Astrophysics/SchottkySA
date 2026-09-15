"""ssa.session TOML schema round-trip tests."""

from __future__ import annotations

import math

from conftest import DATA_DIR
from ssa.config import FitConfig, TemplateBuildConfig, UncertaintyConfig
from ssa.io import load_spectrum_npz, make_export_rows_for_result, select_frequency_range
from ssa.pipeline import build_template_bank, run_fit_with_uncertainty
from ssa.session import (
    FitRecord,
    FormState,
    InputRef,
    Session,
    TemplateRecord,
    load_session,
    replay_session,
    save_session,
)


def _sample_session() -> Session:
    return Session(
        schema_version=1,
        ssa_version="0.1.0",
        created_utc="2026-09-15T12:00:00+00:00",
        input=InputRef(
            path="/data/spectrum.npz",
            sha256="abc123",
            frequency_key="frequency",
            amplitude_key="amplitude",
        ),
        form_state=FormState(
            template=TemplateBuildConfig(name="reference_peak", sg_window=None),
            fit=FitConfig(n_peaks=2, init_mus=None, mu_bounds=None),
            fit_label="fit_1",
            auto_store=True,
            uncertainty=UncertaintyConfig(),
            region_hz=(1937370.0, 1937420.0),
        ),
        templates=[
            TemplateRecord(
                id=0,
                region_hz=(1937300.0, 1937360.0),
                config=TemplateBuildConfig(name="reference_peak", sg_window=None),
                verify={
                    "raw_area": 12.5,
                    "raw_cog": 1937330.1,
                    "raw_std": 3.2,
                    "q16": -2.1,
                    "q50": 0.0,
                    "q84": 2.1,
                    "n_template_bank": 300,
                },
            )
        ],
        fits=[
            FitRecord(
                label="fit_1",
                notes="",
                template_id=0,
                region_hz=(1937370.0, 1937420.0),
                fit_config=FitConfig(
                    n_peaks=2,
                    init_mus=[1937388.0, 1937399.0],
                    mu_bounds=[(1937385.0, 1937393.0), (1937396.0, 1937402.0)],
                    min_separation=3.0,
                ),
                uncertainty_config=UncertaintyConfig(n_boot=500),
                verify_rows=[
                    {
                        "label": "fit_1",
                        "component": 0,
                        "mu_fit_Hz": 1937389.0,
                        "delta_mu_prev_fit_Hz": float("nan"),
                        "notes": "",
                    },
                    {
                        "label": "fit_1",
                        "component": 1,
                        "mu_fit_Hz": 1937400.0,
                        "delta_mu_prev_fit_Hz": 11.0,
                        "notes": "",
                    },
                ],
            )
        ],
    )


def test_session_round_trip(tmp_path):
    original = _sample_session()
    path = tmp_path / "session.toml"
    save_session(path, original)
    loaded = load_session(path)
    assert loaded == original


def test_session_round_trip_preserves_nan(tmp_path):
    original = _sample_session()
    path = tmp_path / "session.toml"
    save_session(path, original)
    loaded = load_session(path)
    assert math.isnan(loaded.fits[0].verify_rows[0]["delta_mu_prev_fit_Hz"])


def _build_real_session(tmp_path):
    import shutil

    npz_src = DATA_DIR / "synthetic_reference_peak.npz"
    npz_path = tmp_path / "synthetic_reference_peak.npz"
    shutil.copy(npz_src, npz_path)

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
        "config": template_config,
        "summary": bank.get("summary", {}),
        "settings": {
            "smooth_nominal_template": template_config.smooth,
            "resample_factor": template_config.resample_factor,
        },
        "edge_info": getattr(tpl, "edge_info", {}),
        "residual_model": bank.get("residual_model", {}),
        "full": bank,
    }
    rows = make_export_rows_for_result(result, label="fit_1", template_registry=[template_entry], notes="")

    from ssa.session import (
        _sha256_of,
    )

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


def test_replay_session_matches_live_run(tmp_path):
    session, _ = _build_real_session(tmp_path)
    report = replay_session(session)
    assert report.mismatches == []
    assert report.ok is True
    assert len(report.templates) == 1
    assert len(report.fits) == 1


def test_replay_session_detects_tampered_verify_value(tmp_path):
    session, _ = _build_real_session(tmp_path)
    session.templates[0].verify["raw_area"] = session.templates[0].verify["raw_area"] + 1000.0
    report = replay_session(session)
    assert report.ok is False
    assert any("raw_area" in m for m in report.mismatches)
