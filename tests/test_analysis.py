"""Branch coverage for ssa.templates: edge modes, smoothing, clip_negative,
and the error paths raised on malformed/too-small input regions.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import load_xy
from ssa.constants import EDGE_MODES
from ssa.exceptions import TemplateConstructionError
from ssa.templates import build_peak_template


@pytest.fixture(scope="module")
def reference_xy():
    return load_xy("synthetic_reference_peak.npz", "frequency", "amplitude")


@pytest.mark.parametrize("mode", EDGE_MODES)
def test_all_edge_modes_build_a_valid_template(reference_xy, mode):
    f_ref, y_ref = reference_xy
    tpl = build_peak_template(f_ref, y_ref, name=f"tpl_{mode}", edge_mode=mode)
    assert np.isfinite(tpl.raw_area) and tpl.raw_area > 0
    assert np.isfinite(tpl.raw_cog)
    assert np.all(np.isfinite(tpl.pdf))


def test_smooth_true_changes_but_does_not_break_the_template(reference_xy):
    f_ref, y_ref = reference_xy
    tpl_off = build_peak_template(f_ref, y_ref, smooth=False)
    tpl_on = build_peak_template(f_ref, y_ref, smooth=True)
    assert np.isfinite(tpl_on.raw_area) and tpl_on.raw_area > 0
    # Smoothing is expected to change the raw std (it removes noise); off is
    # the "safer publication default" and must remain the function default.
    assert tpl_off.raw_std != pytest.approx(tpl_on.raw_std)


def test_clip_negative_false_shifts_instead_of_clipping(reference_xy):
    f_ref, y_ref = reference_xy
    y_with_dip = y_ref.copy()
    y_with_dip[:5] -= 1000.0  # force a large negative excursion
    tpl_clip = build_peak_template(f_ref, y_with_dip, clip_negative=True)
    tpl_shift = build_peak_template(f_ref, y_with_dip, clip_negative=False)
    assert np.isfinite(tpl_clip.raw_area)
    assert np.isfinite(tpl_shift.raw_area)
    assert tpl_clip.raw_area != pytest.approx(tpl_shift.raw_area)


def test_too_few_points_raises_template_construction_error():
    x = np.linspace(0, 1, 5)
    y = np.ones(5)
    with pytest.raises(TemplateConstructionError):
        build_peak_template(x, y)


def test_nonpositive_area_raises_template_construction_error():
    x = np.linspace(0, 10, 20)
    y = np.full_like(x, -1.0)  # all-negative -> clipped area is 0
    with pytest.raises(TemplateConstructionError):
        build_peak_template(x, y)


def test_template_construction_error_is_a_value_error():
    """Specific exceptions subclass the matching builtin, so plain
    `except ValueError` call sites are unaffected -- see ssa.exceptions."""
    assert issubclass(TemplateConstructionError, ValueError)
