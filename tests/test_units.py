"""Scientific-invariant tests: things that must hold regardless of exact
numeric values (dimensions, normalisation, monotonicity, symmetry).
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import load_xy
from ssa.preprocessing import trapz
from ssa.templates import build_peak_template
from ssa.uncertainty import combined_sigma, symmetric_sigma


@pytest.fixture(scope="module")
def reference_template():
    f_ref, y_ref = load_xy("synthetic_reference_peak.npz", "frequency", "amplitude")
    return build_peak_template(f_ref, y_ref, name="reference_peak")


def test_template_pdf_integrates_to_one(reference_template):
    tpl = reference_template
    assert trapz(tpl.pdf, tpl.u) == pytest.approx(1.0, abs=1e-9)


def test_template_is_cog_centered(reference_template):
    """build_peak_template recenters the *interpolated* grid to exact COG=0
    -- see docs/SCIENTIFIC_METHOD.md section 3."""
    tpl = reference_template
    assert trapz(tpl.u * tpl.pdf, tpl.u) == pytest.approx(0.0, abs=1e-9)


def test_template_cdf_is_monotone_and_bounded(reference_template):
    tpl = reference_template
    assert tpl.cdf[0] >= 0.0
    assert tpl.cdf[-1] == pytest.approx(1.0, abs=1e-9)
    assert np.all(np.diff(tpl.cdf) >= -1e-12)  # non-decreasing (allow fp noise)


def test_evaluate_component_preserves_area_under_scale_change(reference_template):
    """(area/scale) * pdf((x-mu)/scale) must integrate back to `area` for any
    scale -- this is the defining property of the model's scale invariance.
    The template has finite support in `u`, so the integration grid must
    widen with `scale` to actually capture the stretched kernel -- a
    fixed-width grid would truncate it and is a test-harness artefact, not a
    template defect."""
    tpl = reference_template
    mu = 1_937_330.0
    for scale in (0.5, 1.0, 2.0):
        margin = 1.05 * scale * max(abs(tpl.u.min()), abs(tpl.u.max()))
        x = mu + np.linspace(-margin, margin, 20_000)
        y = tpl.evaluate_component(x, mu=mu, area=17.0, scale=scale)
        area_back = trapz(y, x)
        assert area_back == pytest.approx(17.0, rel=1e-3)


def test_combined_sigma_is_nonnegative_or_nan():
    ok = {"median": 1.0, "minus": 0.2, "plus": 0.3, "q16": 0.8, "q84": 1.3, "n": 100}
    empty = {"median": np.nan, "minus": np.nan, "plus": np.nan, "q16": np.nan, "q84": np.nan, "n": 0}
    assert symmetric_sigma(ok) >= 0.0
    assert np.isnan(symmetric_sigma(empty))
    assert np.isnan(symmetric_sigma(None))
    c = combined_sigma(ok, ok)
    assert c >= 0.0
    assert c >= symmetric_sigma(ok)  # quadrature combination of two positive sigmas grows
    assert np.isnan(combined_sigma(None, empty))
