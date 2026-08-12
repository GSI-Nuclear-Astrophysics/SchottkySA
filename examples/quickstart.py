"""Minimal end-to-end SchottkySA example: no GUI required.

Builds an empirical template from an isolated reference peak, fits a target
region, and prints the fitted centroid/area with residual-bootstrap and
template-shape-propagation uncertainties.

Run with:
    python examples/quickstart.py

Uses the same synthetic fixtures as the test suite (tests/data/*.npz) -- not
real experimental data. See tests/baseline/make_fixtures.py for how they
were generated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ssa import (
    bootstrap_fit,
    build_peak_template,
    combined_sigma,
    component_quantiles_from_fit,
    fit_template_region,
    load_spectrum_npz,
    quality_flag_for_fit,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "tests" / "data"


def main() -> None:
    # 1. Build the empirical template from an isolated, high-SNR reference peak.
    f_ref, y_ref, fkey, ykey = load_spectrum_npz(DATA_DIR / "synthetic_reference_peak.npz")
    print(f"Loaded reference peak: {len(f_ref)} points ({fkey}/{ykey})")
    template = build_peak_template(f_ref, y_ref, name="reference_peak")
    print(f"Template: raw_area={template.raw_area:.6g}, raw_cog={template.raw_cog:.6f} Hz")

    # 2. Fit a single-peak target region with that template.
    f_target, y_target, _, _ = load_spectrum_npz(DATA_DIR / "synthetic_single_peak.npz")
    fit = fit_template_region(f_target, y_target, template=template, n_peaks=1)
    print(
        f"\nFit: mu={fit['info']['mus'][0]:.6f} Hz, area={fit['info']['areas'][0]:.6g}, "
        f"red_chi2={fit['red_chi2']:.4g}, quality={quality_flag_for_fit(fit)}"
    )

    # 3. Residual block-bootstrap uncertainty on the fitted centroid.
    boot = bootstrap_fit(fit, n_boot=300, rng=np.random.default_rng(12345))
    mu_boot = boot["summary"]["mu_0"]
    print(f"Bootstrap mu: {mu_boot['median']:.6f} -{mu_boot['minus']:.3g} +{mu_boot['plus']:.3g} Hz")

    # 4. Component width/quantile summary.
    qrows = component_quantiles_from_fit(fit)
    print(f"sigma68 = {qrows[0]['sigma68_Hz']:.4g} Hz")
    print(f"Combined mu sigma (bootstrap only here) = {combined_sigma(mu_boot):.4g} Hz")


if __name__ == "__main__":
    main()
