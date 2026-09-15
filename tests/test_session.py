"""ssa.session TOML schema round-trip tests."""

from __future__ import annotations

import math

from ssa.config import FitConfig, TemplateBuildConfig, UncertaintyConfig
from ssa.session import (
    FitRecord,
    FormState,
    InputRef,
    Session,
    TemplateRecord,
    load_session,
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
