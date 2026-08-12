"""
Generate small synthetic Schottky-like spectra for baseline/regression testing.

These are NOT real experimental data. The shape (asymmetric two-sided Gaussian
"tent") is chosen only to be non-trivial (non-symmetric, non-analytic template)
and to exercise isolated-peak template construction plus a contaminated
(overlapping) two-peak fit region, on a frequency scale representative of
real Schottky spectra (order 1.9 MHz, sub-100 Hz peak spacing).

Deterministic: fixed numpy Generator seeds throughout.
"""

from __future__ import annotations

import numpy as np


def asym_two_sided_gaussian(f: np.ndarray, mu: float, sigma_left: float, sigma_right: float, amp: float) -> np.ndarray:
    out = np.empty_like(f, dtype=float)
    left = f < mu
    out[left] = amp * np.exp(-0.5 * ((f[left] - mu) / sigma_left) ** 2)
    out[~left] = amp * np.exp(-0.5 * ((f[~left] - mu) / sigma_right) ** 2)
    return out


def make_reference_peak(seed: int = 1001):
    """Isolated, high-SNR peak used to build the empirical template."""
    f = np.arange(1_937_300.0, 1_937_360.0 + 1e-9, 0.02)
    mu, sl, sr, amp = 1_937_330.0, 0.90, 1.70, 100.0
    y_clean = asym_two_sided_gaussian(f, mu, sl, sr, amp)
    rng = np.random.default_rng(seed)
    y = y_clean + rng.normal(0.0, 0.5, size=f.shape)
    return f, y, {"mu": mu, "sigma_left": sl, "sigma_right": sr, "amp": amp, "noise_sigma": 0.5}


def make_single_peak_target(seed: int = 1002):
    """A second isolated peak (different region) for a simple n_peaks=1 fit."""
    f = np.arange(1_937_440.0, 1_937_500.0 + 1e-9, 0.02)
    mu, sl, sr, amp = 1_937_470.0, 0.90, 1.70, 60.0
    bg = 1.5
    y_clean = bg + asym_two_sided_gaussian(f, mu, sl, sr, amp)
    rng = np.random.default_rng(seed)
    y = y_clean + rng.normal(0.0, 0.5, size=f.shape)
    return f, y, {"mu": mu, "sigma_left": sl, "sigma_right": sr, "amp": amp, "bg": bg, "noise_sigma": 0.5}


def make_contaminated_two_peak_target(seed: int = 1003):
    """Two overlapping (close) peaks sharing the same line shape as the reference."""
    f = np.arange(1_937_370.0, 1_937_420.0 + 1e-9, 0.02)
    mu1, mu2 = 1_937_390.0, 1_937_397.5
    sl, sr = 0.90, 1.70
    a1, a2 = 80.0, 40.0
    bg = 2.0
    y_clean = bg + asym_two_sided_gaussian(f, mu1, sl, sr, a1) + asym_two_sided_gaussian(f, mu2, sl, sr, a2)
    rng = np.random.default_rng(seed)
    y = y_clean + rng.normal(0.0, 0.5, size=f.shape)
    return (
        f,
        y,
        {
            "mu1": mu1,
            "mu2": mu2,
            "sigma_left": sl,
            "sigma_right": sr,
            "amp1": a1,
            "amp2": a2,
            "bg": bg,
            "noise_sigma": 0.5,
        },
    )


def make_duplicate_x_case():
    """A few duplicate x-values (unsorted) to exercise prepare_xy's averaging branch."""
    x = np.array([5.0, 1.0, 3.0, 1.0, 3.0, 2.0, 4.0])
    y = np.array([50.0, 10.0, 30.0, 12.0, 32.0, 20.0, 40.0])
    return x, y


if __name__ == "__main__":
    import json
    from pathlib import Path

    out_dir = Path(__file__).parent
    f_ref, y_ref, meta_ref = make_reference_peak()
    f_single, y_single, meta_single = make_single_peak_target()
    f_two, y_two, meta_two = make_contaminated_two_peak_target()

    np.savez(out_dir / "synthetic_reference_peak.npz", frequency=f_ref, amplitude=y_ref)
    np.savez(out_dir / "synthetic_single_peak.npz", frequency=f_single, amplitude=y_single)
    np.savez(out_dir / "synthetic_two_peak_contaminated.npz", frequency=f_two, amplitude=y_two)

    # Also one combined spectrum NPZ mimicking a real loaded file (covers both
    # reference and target regions in one array with a gap between them),
    # using the alternate accepted key names "f"/"y" for io coverage.
    gap_f = np.arange(1_937_361.0, 1_937_369.0, 0.5)
    gap_y = np.full_like(gap_f, 0.2)
    f_full = np.concatenate([f_ref, gap_f, f_two])
    y_full = np.concatenate([y_ref, gap_y, y_two])
    order = np.argsort(f_full)
    np.savez(out_dir / "synthetic_full_spectrum.npz", f=f_full[order], y=y_full[order])

    meta = {"reference": meta_ref, "single_peak": meta_single, "two_peak": meta_two}
    (out_dir / "fixture_metadata.json").write_text(json.dumps(meta, indent=2))
    print("Wrote fixtures to", out_dir)
