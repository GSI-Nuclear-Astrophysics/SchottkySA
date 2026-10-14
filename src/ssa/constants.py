"""Shared numerical defaults and reporting thresholds.

The API and GUI retain separate target-bootstrap count defaults. Fixed seeds
make the chosen settings reproducible; changing a default is a numerical change.
"""

from __future__ import annotations

# --- Template construction (build_peak_template) ----------------------------
DEFAULT_CLIP_NEGATIVE: bool = True
DEFAULT_SMOOTH_NOMINAL_TEMPLATE: bool = False  # Preserve measured structure unless smoothing is selected.
DEFAULT_SG_POLY: int = 3
DEFAULT_RESAMPLE_FACTOR: int = 16
DEFAULT_EDGE_MODE: str = "none"
DEFAULT_EDGE_WIDTH_HZ: float = 0.0  # 0 => derive from DEFAULT_EDGE_FRACTION
DEFAULT_EDGE_FRACTION: float = 0.08

EDGE_MODES: tuple[str, ...] = (
    "none",
    "pad_to_zero",
    "pchip_extrapolate_to_zero",
    "pchip_monotone_extrapolate_to_zero",
    "local_slope_to_zero",
    "cosine_taper_to_zero",
    "subtract_edge_baseline_pad_to_zero",
    "subtract_edge_baseline_pchip_extrapolate_to_zero",
    "subtract_edge_baseline_pchip_monotone_extrapolate_to_zero",
    "subtract_edge_baseline_local_slope_to_zero",
    "subtract_edge_baseline_cosine_taper_to_zero",
)

# --- Template bootstrap bank (bootstrap_template_bank) ----------------------
DEFAULT_N_TEMPLATE_BOOT: int = 6000
DEFAULT_BLOCK_SIZE: str = "auto"
DEFAULT_RANDOM_SEED: int = 12345
DEFAULT_MAX_ACF_LAG: int = 80
DEFAULT_MIN_BLOCK: int = 2
DEFAULT_MAX_BLOCK_FRACTION: float = 0.25

# --- Multi-component fit (fit_template_region) -------------------------------
DEFAULT_N_PEAKS: int = 1
DEFAULT_BACKGROUND_ORDER: int = 0  # -1 (GUI "none") disables the background term
DEFAULT_ALLOW_SCALE: bool = True
DEFAULT_COMMON_SCALE: bool = True
DEFAULT_LOSS: str = "linear"
LOSS_CHOICES: tuple[str, ...] = ("linear", "soft_l1", "huber", "cauchy", "arctan")
DEFAULT_MAX_NFEV: int = 50000
DEFAULT_MIN_SEPARATION_HZ: float = 0.0
DEFAULT_SEPARATION_PENALTY_STRENGTH: float = 1.0e6

# --- Uncertainty propagation --------------------------------------------------
# Function-level default for ssa.uncertainty.bootstrap_fit(n_boot=...).
DEFAULT_FIT_BOOTSTRAP_N_BOOT_API: int = 6000
# GUI "Bootstrap samples" spin box default / always-explicit value.
DEFAULT_FIT_BOOTSTRAP_N_BOOT_GUI: int = 1000
DEFAULT_FIT_BOOTSTRAP_MAX_NFEV: int = 20000

DEFAULT_N_TEMPLATE_PROPAGATION_DRAWS: int = 3000

# --- Quality-flag thresholds (quality_flag_for_fit) --------------------------
QUALITY_RED_CHI2_WARN: float = 2.0
QUALITY_RED_CHI2_BAD: float = 5.0
QUALITY_MU_NEAR_BOUND_FRACTION: float = 0.02

# Check both sample count and success fraction: either can limit percentile stability.
QUALITY_MIN_VALID_UNCERTAINTY_SAMPLES: int = 50
QUALITY_MIN_UNCERTAINTY_SAMPLE_FRACTION: float = 0.5

# --- Robust noise-scale estimate ---------------------------------------------
DEFAULT_ROBUST_SIGMA_MIN_FRACTION: float = 1e-4

# --- NPZ key-selection precedence (ssa.io.load_spectrum_npz) -----------------
PREFERRED_FREQUENCY_KEYS: tuple[str, ...] = ("frequency", "freq", "f", "x", "f_sum_tdms_corr2")
PREFERRED_AMPLITUDE_KEYS: tuple[str, ...] = ("amplitude", "amp", "y", "power", "psd", "av_corr")
