"""I/O tests: NPZ key selection, export row schema, delimited/JSON writers."""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import DATA_DIR, load_xy, write_table
from ssa.exceptions import InputFormatError
from ssa.fitting import fit_template_region
from ssa.io import (
    export_history_rows,
    load_spectrum_npz,
    load_table,
    make_export_rows_for_result,
    select_frequency_range,
    select_npz_keys,
    write_curve_csv,
    write_delimited,
)
from ssa.templates import build_peak_template


def test_select_npz_keys_prefers_named_keys():
    assert select_npz_keys(["frequency", "amplitude", "extra"]) == ("frequency", "amplitude")
    assert select_npz_keys(["f", "y"]) == ("f", "y")
    assert select_npz_keys(["f_sum_tdms_corr2", "av_corr"]) == ("f_sum_tdms_corr2", "av_corr")


def test_select_npz_keys_falls_back_to_exactly_two_1d_arrays():
    assert select_npz_keys(["foo", "bar"], one_d_keys=["foo", "bar"]) == ("foo", "bar")


def test_select_npz_keys_ambiguous_returns_none_none():
    assert select_npz_keys(["a", "b", "c"], one_d_keys=["a", "b", "c"]) == (None, None)


def test_load_spectrum_npz_named_keys():
    f, y, fkey, ykey = load_spectrum_npz(DATA_DIR / "synthetic_reference_peak.npz")
    assert fkey == "frequency"
    assert ykey == "amplitude"
    assert len(f) == len(y)
    assert np.all(np.diff(f) > 0)  # prepare_xy sorts


def test_load_spectrum_npz_alias_keys():
    f, y, fkey, ykey = load_spectrum_npz(DATA_DIR / "synthetic_full_spectrum.npz")
    assert fkey == "f"
    assert ykey == "y"
    assert f.shape == y.shape


def test_load_spectrum_npz_missing_file_raises(tmp_path):
    with pytest.raises((InputFormatError, FileNotFoundError, OSError)):
        load_spectrum_npz(tmp_path / "does_not_exist.npz")


@pytest.fixture(scope="module")
def sample_fit():
    f_ref, y_ref = load_xy("synthetic_reference_peak.npz", "frequency", "amplitude")
    tpl = build_peak_template(f_ref, y_ref, name="reference_peak")
    f_single, y_single = load_xy("synthetic_single_peak.npz", "frequency", "amplitude")
    fit = fit_template_region(f_single, y_single, template=tpl, n_peaks=1)
    return {"fit": fit, "bootstrap": None, "template_propagation": None}


def test_export_rows_have_no_mcmc_columns(sample_fit):
    """Bayesian MCMC is not part of ssa -- its *_mcmc_* export columns
    must not appear."""
    rows = make_export_rows_for_result(sample_fit, label="fit_1")
    assert len(rows) == 1
    keys = set(rows[0].keys())
    mcmc_keys = {k for k in keys if "mcmc" in k.lower()}
    assert not mcmc_keys, f"Unexpected MCMC columns in export schema: {mcmc_keys}"


def test_export_rows_have_expected_core_columns(sample_fit):
    rows = make_export_rows_for_result(sample_fit, label="fit_1")
    expected_subset = {
        "label",
        "component",
        "template",
        "quality_flag",
        "mu_fit_Hz",
        "mu_bootstrap_median_Hz",
        "mu_template_median_Hz",
        "mu_combined_sigma_Hz",
        "area_fit",
        "scale_fit",
        "rms_width_Hz",
        "q16_Hz",
        "q50_Hz",
        "q84_Hz",
        "sigma68_Hz",
        "red_chi2",
        "chi2",
        "dof",
        "notes",
    }
    assert expected_subset.issubset(rows[0].keys())


def test_export_rows_have_no_aic_bic_dw_columns(sample_fit):
    """AIC/BIC/Durbin-Watson are not part of ssa's export schema."""
    rows = make_export_rows_for_result(sample_fit, label="fit_1")
    assert "AIC" not in rows[0]
    assert "BIC" not in rows[0]
    assert "DW" not in rows[0]


def test_write_delimited_roundtrip(tmp_path, sample_fit):
    rows = make_export_rows_for_result(sample_fit, label="fit_1")
    path = tmp_path / "out.tsv"
    write_delimited(path, rows, delimiter="\t")
    lines = path.read_text().splitlines()
    assert len(lines) == 2  # header + one row
    assert "mu_fit_Hz" in lines[0].split("\t")


def test_export_history_rows_json(tmp_path, sample_fit):
    rows = make_export_rows_for_result(sample_fit, label="fit_1")
    path = tmp_path / "out.json"
    export_history_rows(rows, path, source_file="dummy.npz")
    payload = json.loads(path.read_text())
    assert payload["file"] == "dummy.npz"
    assert len(payload["rows"]) == 1


def test_export_history_rows_empty_raises(tmp_path):
    with pytest.raises(InputFormatError):
        export_history_rows([], tmp_path / "out.tsv")


def test_export_history_rows_xlsx_requires_pandas(tmp_path, sample_fit):
    pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    rows = make_export_rows_for_result(sample_fit, label="fit_1")
    path = tmp_path / "out.xlsx"
    export_history_rows(rows, path)
    assert path.exists()


# --------------------------- load_table / ssa run I/O ---------------------------


def test_load_table_csv(tmp_path):
    f, y = load_xy("synthetic_reference_peak.npz", "frequency", "amplitude")
    path = write_table(tmp_path / "ref.csv", f[:20], y[:20], delimiter=",")
    f2, y2, sigma2 = load_table(path)
    assert sigma2 is None
    np.testing.assert_allclose(f2, f[:20])
    np.testing.assert_allclose(y2, y[:20])


def test_load_table_tsv(tmp_path):
    f, y = load_xy("synthetic_reference_peak.npz", "frequency", "amplitude")
    path = write_table(tmp_path / "ref.tsv", f[:20], y[:20], delimiter="\t")
    f2, y2, sigma2 = load_table(path)
    np.testing.assert_allclose(f2, f[:20])
    np.testing.assert_allclose(y2, y[:20])


def test_load_table_whitespace(tmp_path):
    f, y = load_xy("synthetic_reference_peak.npz", "frequency", "amplitude")
    path = write_table(tmp_path / "ref.dat", f[:20], y[:20], delimiter=" ")
    f2, y2, sigma2 = load_table(path)
    np.testing.assert_allclose(f2, f[:20])
    np.testing.assert_allclose(y2, y[:20])


def test_load_table_with_sigma_column(tmp_path):
    f, y = load_xy("synthetic_reference_peak.npz", "frequency", "amplitude")
    sigma = np.full_like(f[:20], 0.5)
    path = write_table(tmp_path / "ref.csv", f[:20], y[:20], sigma=sigma, delimiter=",")
    f2, y2, sigma2 = load_table(path)
    assert sigma2 is not None
    np.testing.assert_allclose(sigma2, sigma)


def test_load_table_custom_column_names(tmp_path):
    f, y = load_xy("synthetic_reference_peak.npz", "frequency", "amplitude")
    path = write_table(tmp_path / "ref.csv", f[:20], y[:20], delimiter=",", freq_col="f", power_col="amp")
    f2, y2, _ = load_table(path, freq_col="f", power_col="amp")
    np.testing.assert_allclose(f2, f[:20])


def test_load_table_missing_column_raises(tmp_path):
    f, y = load_xy("synthetic_reference_peak.npz", "frequency", "amplitude")
    path = write_table(tmp_path / "ref.csv", f[:20], y[:20], delimiter=",")
    with pytest.raises(InputFormatError):
        load_table(path, freq_col="not_a_column")


def test_load_table_empty_file_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(InputFormatError):
        load_table(path)


def test_load_table_ignores_comment_lines(tmp_path):
    path = tmp_path / "ref.csv"
    path.write_text("# a comment\nfrequency_hz,power\n1.0,2.0\n# another comment\n3.0,4.0\n")
    f, y, sigma = load_table(path)
    np.testing.assert_allclose(f, [1.0, 3.0])
    np.testing.assert_allclose(y, [2.0, 4.0])


def test_select_frequency_range_inclusive_bounds():
    f = np.arange(20.0)
    y = np.arange(20.0) * 2
    f2, y2, sigma2 = select_frequency_range(f, y, None, 5.0, 15.0)
    assert f2.min() == 5.0
    assert f2.max() == 15.0
    assert sigma2 is None


def test_select_frequency_range_none_bounds_is_noop():
    f = np.arange(20.0)
    y = np.arange(20.0)
    f2, y2, _ = select_frequency_range(f, y, None, None, None)
    assert f2 is f
    assert y2 is y


def test_select_frequency_range_too_few_points_raises():
    f = np.arange(20.0)
    y = np.arange(20.0)
    with pytest.raises(InputFormatError):
        select_frequency_range(f, y, None, 5.0, 5.5)


def test_write_curve_csv(tmp_path):
    x = np.linspace(0, 1, 10)
    y = x * 2
    y_model = x * 2 + 0.01
    background = np.zeros_like(x)
    components = np.asarray([x * 2 + 0.01])
    path = tmp_path / "curve.csv"
    write_curve_csv(path, x, y, y_model, background, components)
    lines = path.read_text().splitlines()
    assert lines[0].split(",") == ["frequency_hz", "y", "y_model", "background", "component_0", "residual"]
    assert len(lines) == 11  # header + 10 rows
