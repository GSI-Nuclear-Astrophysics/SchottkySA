---
title: Scientific method
---

# Scientific method

This document describes the algorithm SchottkySA implements: what it
assumes, the equations it evaluates, its units and conventions, and the
uncertainty model.

## 1. Problem statement

Given a frequency-domain spectrum (frequency in Hz on the x-axis, an
amplitude/linear-power quantity on the y-axis), SchottkySA fits one or more
peaks in a user-selected region as scaled, shifted copies of a single
**empirical peak template** -- a non-parametric line shape built directly
from an isolated, high-signal-to-noise reference peak in the same dataset, nearby the desired target peak -- rather than assuming an analytic shape (Gaussian, Lorentzian, Voigt, ...).
This is extremely necessary in Schottky + Isochronous Mass Spectrometry, where the true physical line shape is not well described by a simple closed form, and additionally, depending on the mass to charge ratio each peak has a slightly different shape. The further in m/q from a reference peak the peak is, the more it will deviate from the reference peak.
## 2. Units and conventions

* Frequency axis: Hz (arbitrary absolute offset; only differences and the
  local grid spacing matter to the algorithm).
* Amplitude axis: unitless / linear power 
* All areas, widths (`u`, `sigma68`, RMS width), and centroids (`mu`) are
  reported in the same frequency units as the input x-axis (Hz for the
  intended use case).
* `x` must be finite, and is internally sorted with duplicate x-values
  averaged together (`ssa.preprocessing.prepare_xy`) before any further
  processing.

## 3. Template construction

1. **Preprocessing** (`ssa.preprocessing.prepare_xy`): sort by x, drop
   non-finite points, average y for duplicate x. Usually the data is provided in such a way this is already fulfilled, just double checking. Requires >= 3 points.
2. **Optional edge handling** (`ssa.templates.apply_template_edge_handling`):
   nine modes (default: `none`), from doing nothing to non-parametric
   PCHIP-based tail closures to zero, half-cosine tapering, and/or constant
   edge-baseline subtraction. Closures follow only the locally observed edge shape or a
   simple taper. See the mode docstrings in `ssa.templates` for the full
   list and when each is appropriate. Only use when the edges of a reference are contaminated.
3. **Optional Savitzky-Golay smoothing** of the reference peak
   (default: **off** ; sharp Schottky peak
   structures should not be washed out by default, however can be useful when interpolating or when no uncontaminated reference peak is available).
4. **Normalisation to a template PDF:**
   * clip (default) or shift the (possibly edge-handled, possibly smoothed)
     y-values to be non-negative;
   * `area = integral(y_pos dx)` (trapezoidal rule);
   * `cog = integral(x * y_pos dx) / area` -- the centre of gravity (first
     moment);
   * recentre `u = x - cog`; `pdf_raw = y_pos / area`;
   * resample `u` onto a uniform grid with spacing `dx_original /
     resample_factor` (default factor 16) via **shape-preserving PCHIP
     interpolation** (not a spline, not linear -- PCHIP avoids introducing
     spurious overshoot/ringing near sharp features);
   * renormalise to unit area, then **recentre a second time** on the
     resampled grid so the final template has exact `COG = 0`;
   * `cdf = cumulative_trapezoid(pdf, u)`, normalised to `[0, 1]`.

The result, `PeakTemplate(u, pdf, cdf, ...)`, is a normalized probability
density in a frequency offset `u` relative to the reference peak's own COG,
plus the raw (pre-resample) area/COG/RMS-width of the reference peak itself
for provenance.

## 4. Multi-component fit model

A fit region with `n` components is modelled as:

```
y_model(x) = background(x) + sum_k (area_k / scale_k) * pdf((x - mu_k) / scale_k)
```

* `mu_k`: centroid of component `k` (Hz).
* `area_k`: total signal area of component `k` (same units as `integral(y
  dx)`); fit in log-space (`log_area_k`) to enforce positivity without an
  explicit inequality constraint.
* `scale_k`: a dimensionless width-scaling factor relative to the template's
  own width (`scale=1` reproduces the template's width exactly). Optional
  (`allow_scale`, default on); when enabled, either one common scale for all
  components (`common_scale`, default on) or a per-component scale. Also fit
  in log-space, bounded to `[0.25, 4.0]` in real space.
* `background(x)`: an optional polynomial of order `background_order`
  (default 0, i.e. a constant offset; `-1` disables it entirely), evaluated
  in the *span-normalised* coordinate `(x - mean(x)) / ptp(x)`, not raw Hz --
  so background coefficients are not directly comparable to a polynomial fit
  in absolute frequency.

### 4.1 Initial guess and identifiability

Centroids are seeded either from user-supplied values, or by
`scipy.signal.find_peaks` on the region (falling back to evenly spaced
offsets around the global maximum if fewer than `n` prominent peaks are
found). **For closely spaced ("contaminated") multi-peak regions, this
automatic seeding can fail to resolve the components** (one can collapse
toward a search-bound with near-zero area). The robust workflow for such
cases is to supply explicit, ordered, non-overlapping per-component centroid
search ranges (`mu_bounds`), which both seed the initial guess (range
midpoint) and become hard optimizer bounds -- guaranteeing component
identity and, if requested, a minimum separation, without relying on a soft
penalty.

### 4.2 Optimizer

`scipy.optimize.least_squares` with explicit parameter bounds and a
selectable robust loss (`linear` default, or `soft_l1`/`huber`/`cauchy`/
`arctan`). The per-point weight is, unless supplied explicitly, a single
scalar effective noise scale estimated from the second-difference statistics
of the region (`ssa.preprocessing.robust_sigma_from_second_difference`) -- since Schottky-spectrum residuals are typically correlated, not white.

## 5. Fit diagnostics

* `chi2 = sum(((y_model - y) / sigma)^2)`, `dof = n_points - n_params`,
  `red_chi2 = chi2 / dof`.
* Parameter covariance: SVD-pseudo-inverse of the Jacobian, rescaled by
  `red_chi2` if finite -- the standard `curve_fit(absolute_sigma=False)`
  convention.
* A compact `quality_flag` string (`ok`, or `;`-joined flags). Always
  available from the fit alone: `fit_not_converged`, `red_chi2_gt_2`/`_gt_5`,
  `min_separation_violated`, `mu_near_search_bound`, `nonpositive_dof`.

## 6. Uncertainty quantification

Two independent resampling procedures, both **residual block-bootstrap**
based (moving-block circular resampling, block size auto-estimated from the
residual's own autocorrelation time by default), each with an explicit
`numpy.random.Generator` seed:

1. **Target-fit residual bootstrap** (`ssa.uncertainty.bootstrap_fit`):
   resample the *fit residual*, add back to the fitted model, refit (from
   the original optimum) -- estimates uncertainty from residual noise in the
   target region alone, holding the template shape fixed.
2. **Template-shape propagation**
   (`ssa.uncertainty.propagate_template_uncertainty_to_target_fit`): refit
   the *same, unperturbed* target data with each of a bank of
   bootstrap-perturbed templates (built the same way, from resampled
   reference-peak residuals) -- estimates how much the result depends on
   the specific (finite-SNR) reference peak used to build the template.

Each yields, per parameter, a 16/50/84-percentile summary
(`ssa.preprocessing.summarize_distribution`); the two are combined as
symmetrised sigmas added **in quadrature**
(`ssa.uncertainty.combined_sigma`), which assumes the two sources are
independent and each interval is approximately symmetric. Pairwise diagnostics (centroid
separations, area ratios, and their correlations across the *same* sample
cloud, preserving joint structure) are available for multi-component fits.

Component widths are reported in two ways, which are: an RMS (second-moment) width (`scale * template_std`), and a quantile-based `sigma68 = 0.5*(q84-q16)` derived from the template.

`combined_sigma` quantifies the *statistical spread* of the centroid (or width, or any other
fitted quantity) given the assumed model -- i.e. how much the estimate would
plausibly move under the noise actually observed in the target and
reference regions. 


## 7. What is out of scope

* Mass-to-charge calibration, atomic-mass-excess conversion, leave-one-out
  calibrant validation, flow-of-information/leverage diagnostics, and
  half-life/decay-chain fitting are **not implemented anywhere in this
  package**. SchottkySA produces per-component centroids, areas, widths, and
  uncertainties (this document); everything downstream of that is out of
  scope for this software, and will be published separately.
