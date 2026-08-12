"""Branch coverage for ssa.fitting: mu_bounds validation, min_separation
enforcement, background_order/allow_scale/common_scale/loss combinations.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import load_xy
from ssa.constants import LOSS_CHOICES
from ssa.exceptions import FitConfigurationError, FitConstraintViolationError
from ssa.fitting import fit_template_region, normalize_mu_bounds
from ssa.templates import build_peak_template


@pytest.fixture(scope="module")
def reference_template():
    f_ref, y_ref = load_xy("synthetic_reference_peak.npz", "frequency", "amplitude")
    return build_peak_template(f_ref, y_ref, name="reference_peak")


@pytest.fixture(scope="module")
def two_peak_xy():
    return load_xy("synthetic_two_peak_contaminated.npz", "frequency", "amplitude")


def test_normalize_mu_bounds_rejects_overlapping_ranges():
    with pytest.raises(FitConfigurationError):
        normalize_mu_bounds([(0.0, 10.0), (5.0, 15.0)], n_peaks=2, xmin=0.0, xmax=20.0)


def test_normalize_mu_bounds_rejects_wrong_count():
    with pytest.raises(FitConfigurationError):
        normalize_mu_bounds([(0.0, 10.0)], n_peaks=2, xmin=0.0, xmax=20.0)


def test_normalize_mu_bounds_sorts_lo_hi_within_a_pair():
    arr = normalize_mu_bounds([(10.0, 0.0)], n_peaks=1, xmin=-1.0, xmax=20.0)
    assert arr is not None
    assert arr[0, 0] == 0.0 and arr[0, 1] == 10.0


def test_mu_bounds_that_violate_min_separation_raise_before_fitting(reference_template, two_peak_xy):
    f_two, y_two = two_peak_xy
    with pytest.raises(FitConfigurationError):
        fit_template_region(
            f_two,
            y_two,
            template=reference_template,
            n_peaks=2,
            min_separation=10.0,
            mu_bounds=[(1_937_385.0, 1_937_393.0), (1_937_393.5, 1_937_402.0)],  # gap 0.5 Hz < 10 Hz
        )


@pytest.mark.parametrize("background_order", [-1, 0, 1])
def test_background_order_variants_fit_successfully(reference_template, two_peak_xy, background_order):
    f_two, y_two = two_peak_xy
    fit = fit_template_region(
        f_two,
        y_two,
        template=reference_template,
        n_peaks=2,
        background_order=background_order,
        mu_bounds=[(1_937_385.0, 1_937_393.0), (1_937_396.0, 1_937_402.0)],
        min_separation=3.0,
    )
    n_bg_expected = 0 if background_order < 0 else background_order + 1
    n_bg_actual = sum(1 for name in fit["param_names"] if name.startswith("bg_"))
    assert n_bg_actual == n_bg_expected


@pytest.mark.parametrize("allow_scale,common_scale", [(True, True), (True, False), (False, True)])
def test_allow_scale_common_scale_variants(reference_template, two_peak_xy, allow_scale, common_scale):
    f_two, y_two = two_peak_xy
    fit = fit_template_region(
        f_two,
        y_two,
        template=reference_template,
        n_peaks=2,
        allow_scale=allow_scale,
        common_scale=common_scale,
        mu_bounds=[(1_937_385.0, 1_937_393.0), (1_937_396.0, 1_937_402.0)],
        min_separation=3.0,
    )
    scale_params = [name for name in fit["param_names"] if name.startswith("log_scale")]
    if not allow_scale:
        assert scale_params == []
    elif common_scale:
        assert scale_params == ["log_scale_common"]
    else:
        assert scale_params == ["log_scale_0", "log_scale_1"]


@pytest.mark.parametrize("loss", LOSS_CHOICES)
def test_all_loss_functions_produce_a_finite_fit(reference_template, two_peak_xy, loss):
    f_two, y_two = two_peak_xy
    fit = fit_template_region(
        f_two,
        y_two,
        template=reference_template,
        n_peaks=2,
        loss=loss,
        mu_bounds=[(1_937_385.0, 1_937_393.0), (1_937_396.0, 1_937_402.0)],
        min_separation=3.0,
    )
    assert np.all(np.isfinite(fit["params"]))
    assert np.isfinite(fit["chi2"])


def test_sigma_shape_mismatch_raises_fit_configuration_error(reference_template):
    f_single, y_single = load_xy("synthetic_single_peak.npz", "frequency", "amplitude")
    with pytest.raises(FitConfigurationError):
        fit_template_region(f_single, y_single, template=reference_template, n_peaks=1, sigma=np.ones(3))


def test_fit_constraint_violation_error_is_a_runtime_error():
    assert issubclass(FitConstraintViolationError, RuntimeError)
