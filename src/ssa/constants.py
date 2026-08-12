"""Named default values shared across the fitting pipeline.

This is the single source of truth used by :mod:`ssa.config` (typed config
objects), :mod:`ssa.gui.main_window` (widget initial values), and the Python
API function defaults in :mod:`ssa.templates`, :mod:`ssa.fitting`, and
:mod:`ssa.uncertainty`. Changing a value here changes reported scientific
output; do so deliberately and document the change.

Two constants intentionally have both a *function* default and a *GUI*
default:

* ``bootstrap_fit``'s own Python-level default is ``n_boot=300``, but the
  GUI's "Bootstrap samples" spin box defaults to, and always explicitly
  passes, ``1000``. Both values are named distinctly below, so neither the
  Python API's nor the GUI's default behaviour is coupled to the other.
"""

from __future__ import annotations

# --- Template construction (build_peak_template) ----------------------------
DEFAULT_CLIP_NEGATIVE: bool = True
DEFAULT_SMOOTH_NOMINAL_TEMPLATE: bool = False  # "safer publication default"
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
    "cosine_taper_to_zero",
    "subtract_edge_baseline_pad_to_zero",
    "subtract_edge_baseline_pchip_extrapolate_to_zero",
    "subtract_edge_baseline_pchip_monotone_extrapolate_to_zero",
    "subtract_edge_baseline_cosine_taper_to_zero",
)

# --- Template bootstrap bank (bootstrap_template_bank) ----------------------
DEFAULT_N_TEMPLATE_BOOT: int = 300
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
DEFAULT_FIT_BOOTSTRAP_N_BOOT_API: int = 300
# GUI "Bootstrap samples" spin box default / always-explicit value.
DEFAULT_FIT_BOOTSTRAP_N_BOOT_GUI: int = 1000
DEFAULT_FIT_BOOTSTRAP_MAX_NFEV: int = 20000

DEFAULT_N_TEMPLATE_PROPAGATION_DRAWS: int = 300

# --- Quality-flag thresholds (quality_flag_for_fit) --------------------------
QUALITY_RED_CHI2_WARN: float = 2.0
QUALITY_RED_CHI2_BAD: float = 5.0
QUALITY_MU_NEAR_BOUND_FRACTION: float = 0.02

# An uncertainty source (residual bootstrap or template-shape propagation) is
# flagged as unreliable if EITHER fewer than this many replicates converged,
# OR fewer than this fraction of the requested/attempted replicates
# converged. Both conditions matter independently: a fit requesting only
# n_boot=20 that gets 15 valid replicates has a fine fraction (75%) but too
# few samples for a stable 16/84-percentile estimate; a fit requesting
# n_boot=1000 that gets 40 valid replicates has "enough" in absolute terms
# but a failure rate high enough to distrust the result. These are project
# conventions (not derived from a formal minimum-sample-size analysis for
# percentile estimation), chosen to be conservative for precision work; see
# docs/SCIENTIFIC_METHOD.md section 6 and docs/OPEN_SCIENTIFIC_QUESTIONS.md.
QUALITY_MIN_VALID_UNCERTAINTY_SAMPLES: int = 50
QUALITY_MIN_UNCERTAINTY_SAMPLE_FRACTION: float = 0.5

# --- Robust noise-scale estimate ---------------------------------------------
DEFAULT_ROBUST_SIGMA_MIN_FRACTION: float = 1e-4

# --- NPZ key-selection precedence (ssa.io.load_spectrum_npz) -----------------
PREFERRED_FREQUENCY_KEYS: tuple[str, ...] = ("frequency", "freq", "f", "x", "f_sum_tdms_corr2")
PREFERRED_AMPLITUDE_KEYS: tuple[str, ...] = ("amplitude", "amp", "y", "power", "psd", "av_corr")
