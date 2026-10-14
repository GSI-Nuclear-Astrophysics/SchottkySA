# Multiple reference profiles and harmonic comparison

## Frequency convention and construction

A source template is a unit-area density `T_j(u)` on a frequency coordinate
relative to its centre of gravity. Its recorded source harmonic is `h_j`.
For output/target harmonic `h_*`, set `r_j = h_*/h_j` and transform

```text
u_* = r_j u_j
T_j,*(u_*) = T_j(u_*/r_j) / r_j.
```

The area remains one and the RMS width scales by `r_j`. This is a change of
frequency units, not a fitted width change. Source COGs are discarded as shape
origins, so neighboring nuclides can also be combined without changing their
absolute frequency assignments. At equal harmonics there is no width stretch.
The absolute location recorded on a composite uses the converted first-source
anchor for display/provenance only; it is not an additional measured frequency.
Its `raw_area=1` records the normalization, not ion counts or measured power.

For positive user weights `w_j`, normalized to sum to one, the nominal template is

```text
T_mean(u) = sum_j w_j T_j,*(u).
```

Linear interpolation places source densities on a common union grid at the
finest converted source grid spacing. Interpolation evaluates to zero outside
each source's support. The mixture is normalized and its interpolated COG is
recentred to zero. No new smoothing is applied. For independent mean draws,
the grid also covers all contributing bootstrap supports. Excessively large
grids are rejected rather than silently coarsened.

For centred input profiles the mixture variance is the weighted mean of the
source variances (up to interpolation). Consequently averaging different widths
does not yield the width of a single clean species automatically. The workbench
intentionally preserves tails, shoulders and differences between selected
sources. Averaging can reduce fluctuations if errors are sufficiently independent;
it cannot remove shared bias or make unlike physical profiles interchangeable.
Weights are explicit and do not infer ion counts from template normalization.

## Cross-fits

`compare_template_entries` fits each measured ROI with the other source's
converted density. Each direction has one component, a free COG and area,
and the selected background polynomial (`-1` means no background). Additional
width scaling is fixed at one by default. If enabled, its fitted value is shown
separately from the known harmonic ratio. A composite has no measured target
ROI and is therefore excluded from this bidirectional operation.

Plots show the measured target, fitted total model and residuals on a frequency
coordinate divided by the target harmonic. The CSV includes the two source IDs,
names, harmonics, known scale ratio, optional fitted width scale, fitted COG,
COG difference relative to the target reference, chi-square, degrees of freedom,
reduced chi-square, Durbin–Watson statistic and optimizer success.

These are conditional shape-compatibility diagnostics. The fitter estimates a
noise scale when measurement uncertainties are unavailable. A template taken
from the same data as its target, correlated frequency bins, shared detector
response and overlapping source ROIs can invalidate a literal independent-bin
chi-square distribution. The GUI does not apply a contamination threshold or
assign a contamination probability. A clean cross-fit also cannot exclude common
contamination affecting both references.

## Two distinct uncertainty constructions

### Source sensitivity (default)

The nominal profile is the weighted mean. Each propagation draw instead chooses
one source with probability `w_j`, then one template from that source's existing
bootstrap bank. Refitting the target with these individual profiles retains
finite-reference shape variability and differences between source shapes.
This ensemble supports a source-sensitivity assessment; it does not construct
confidence intervals for a mean template or guarantee coverage. Sparse draw
counts may poorly sample rare, low-weight sources.

The existing target residual-bootstrap scatter remains the other stream of the
usual combined output. In this mode the reported template contribution includes
between-source sensitivity, so the combined sigma is not purely a statistical
standard error. Exports carry
`template_combination_uncertainty_mode=source_sensitivity` and the quality flag
`combined_template_source_sensitivity`. The nominal mean's discrepancy from
individual-source refits should be examined as well as their scatter; that
possible bias is not automatically converted into a correction by this routine.

### Mean bootstrap (independent references)

For each draw, independently select one template from each source bank, apply
its harmonic conversion and compute their weighted mean. This samples uncertainty
of the mean under independent reference errors. It deliberately makes that
assumption explicit through `template_combination_uncertainty_mode=independent`
and `combined_template_independence_assumed`.

For independent sources the density covariance is `sum_j w_j^2 C_j(u,v)`.
For correlated sources it is instead
`sum_jk w_j w_k C_jk(u,v)`. Independent draws omit cross-covariances. Harmonics of
the same acquisition can share ion populations, drift, alignment or detector
noise, so harmonic separation alone is insufficient to select this mode.
A future joint injection/frame bootstrap would need synchronized source-bank
indices and the corresponding provenance. No joint-acquisition inference is
claimed by the present independent mode.

Reusing a source both directly and through a previously combined profile is
rejected by its source ancestry. This prevents duplicated weighting but does
not establish independence of different sources or distinct extractions from
the same ROI. All new fits retain the existing fitting and uncertainty routines;
there is no new formal Jacobian term or automatic reduced-chi-square inflation.
The source bootstrap retains the existing residual model and now applies the
same selected edge handling to nominal and bootstrap templates. Edge variation
remains a separate sensitivity check; the residual bootstrap does not sample
uncertainty in the chosen closure settings.

## Edge preview

The preview uses the active ROI and the same deterministic nominal construction
as template building. It displays raw measured samples with area normalization,
a profile with `edge_mode=none`, and the selected treatment. Dashed vertical
lines mark the first and last included measured bins, and orange curves mark
outside-ROI extensions. Left and right zooms show closure locally. The displayed
COG change is the difference between the two nominal COGs with otherwise
identical settings; it is neither an assigned uncertainty nor a proof of the
correct tail model. Automatic updates use a short GUI debounce and do not
create bootstrap draws. The selected edge treatment is saved with each source.

## Python API and replay

```python
from ssa.profile_tools import combine_template_entries, compare_template_entries

# entries are the GUI/session template registry records, including harmonic,
# measured region bounds, template and template_bank.
comparison = compare_template_entries(
    frequency,
    amplitude,
    entries,
    0,
    1,
    allow_scale=False,
    background_order=0,
    workers=2,
)
bank = combine_template_entries(
    [entries[0], entries[1]],
    name="joint_reference",
    output_harmonic=126,
    weights=[1, 1],
    n_boot=1000,
    random_seed=12345,
    bank_mode="source_sensitivity",
    workers=4,
)
```

Inputs to combination require at least two profiles, positive finite harmonic
ratios, positive finite weights, valid grids and nonempty source banks when
draws are requested. Zero combined draws is allowed for nominal comparisons;
it does not provide template uncertainty. Random choices are planned in stable
order before worker execution. Serial and parallel results therefore have the
same ordering and values in a fixed numerical environment.

Session schema 2 adds source harmonic and combination recipes to template
records, target harmonic and view index to fits, comparison records, and GUI
view state. The input NPZ remains referenced by path and SHA256. Replay rebuilds
source banks before their composites, fits saved measured windows, and verifies
both target-fit and comparison checkpoints. Schema 1 continues to load with
unit harmonic and an empty second-view/comparison state. Existing single-template
scientific results are unchanged when source and target harmonics both equal one.


## Graphical multi-component initialization

The **Fit** tab adds one coloured initial-COG line and optional hard search range
per component to the selected spectrum view. With picking enabled, a left-button
double-click inside the fit ROI creates a component with a range centred on the
clicked frequency. The chosen initial values need not be FFT-bin centres; they
are continuous optimizer coordinates. The automatic new-range half-width is 2%
of the displayed span, clipped to the selected ROI, and an explicit Hz half-width
can be supplied. This convenience default is not a physical uncertainty estimate.

Dragging a centre translates its range, preserving its width when the ROI permits.
Dragging a boundary resizes the range and keeps the initial COG inside the new
bounds. Numerical fields remain canonical: graphics write the component count,
initial values and explicit ranges into the existing `FitConfig` path. Manual
numerical values can also be drawn, including unbounded centres. Graphical
components are sorted by frequency when moved/added. The existing scientific core
continues to require ordered, non-overlapping hard ranges and gaps guaranteeing
any positive minimum separation; the GUI displays those validation messages.
No prior, new uncertainty estimator or altered minimization objective is added.

Each view retains its own numerical initialization. Session `view_state` stores
those forms plus picking enablement and new-range half-width. Replay reconstructs
the lines/ranges from the forms, while each `FitRecord` records the actual fit
parameters captured at job start. Previous configurations without picking state
continue to load. Source, target-view and output harmonic spin boxes retain nine
decimal places through save/load.

### Separate left and right closures

The Templates tab provides a sample count and an outward zero-anchor distance
for each edge. Sample count and distance are independent: a long tail need not
use many observed points. Set a positive distance to put the zero anchor exactly
that many spectrum Hz beyond the ROI boundary. `0` inherits the shared controls
and their historical grid-rounded PCHIP distance. Edge counts are capped at
half the ROI minus one point, and the actual count is reported. Green markers
show the observed samples used by each closure and orange dotted markers show
its zero anchor in the full preview and edge zooms. Cosine taper works inside
the ROI and uses counts, so outside zero distances do not affect that mode.

Each stored source's configuration contains its own edge overrides. Session
replay, GUI workers, Python pipeline, CLI and exported history preserve them.
The bootstrap bank uses the same closure as the nominal profile; this corrects
the previous omission of non-default edge handling from bootstrap draws.

Use `local_slope_to_zero` when you want **all selected edge points** to estimate
an average outward slope. It fits a local linear slope and uses a monotone
cubic closure ending at the specified zero distance; it does not replace
observed data inside the ROI. The slope is limited if needed to avoid an
overshoot. PCHIP modes preserve their existing interpolation rule, whose
boundary slope depends mainly on the closest samples. Both choices and actual
edge counts are recorded for replay and inspection.


## Component-specific shapes and harmonic expressions (schema 3)

A simultaneous model may assign a separate centred, unit-area profile to each
component:

```text
y(f) = B(f) + sum_k A_k / s_k * T_k((f - mu_k) / s_k).
```

The stored template is converted from its own source harmonic to the target
harmonic before any fitted scale is applied. `common_scale=True` means one
fractional stretch `s` for every profile; it does not impose equal absolute
widths. Ordered nonoverlapping COG bounds are required for distinct profiles,
so template identity cannot be exchanged by sorting fitted centres. A and B
spectrum views keep separate assignments. Dragging a centre past another centre
reorders its C-number and retains its template assignment.

In the GUI, enable **Use a different template for each component**, choose the
component in **Selected component**, and select its reference. Numerical-only
initialization is also supported. The shared-template button disables overrides.
The Python API accepts `component_templates=[T1, T2, ...]` in
`fit_template_region` and `run_fit_with_uncertainty`. The latter also accepts
`component_template_banks=[bank1, bank2, ...]`.

Each target-residual bootstrap refits the full simultaneous model. For shape
propagation, repeated uses of the same bank object share a single draw within a
replicate, preserving the induced component covariance. Different banks are
sampled independently, explicitly flagged `independent_template_banks_assumed`.
This does not establish independence of separate templates from the same data,
overlapping ROIs, or a composite and one of its source profiles. Such analyses
need joint reference resampling for those cross-covariances. An empty component
bank leaves that profile fixed and emits `component_template_bank_missing`.
Joint parameter samples retain overlap-induced frequency and area covariance;
frequency separations and area ratios are evaluated from those joint samples.

Exports use each component's assigned shape for its width, quantiles, source
bounds, bank count and harmonic conversion. Overlay NPZs retain the legacy
first-template keys and add `component_template_names` and
`component_0_template_u/pdf/cdf`, `component_1_template_u/pdf/cdf`, etc., without
requiring equal support sizes.

Session schema 3 stores per-component source IDs, original harmonic text and
numeric harmonic values. Old schema-1/2 sessions still load with their existing
shared-template convention. Harmonic fields accept positive integers, decimals
and integer fractions (`214/214`, `215/214`). Fractions are safely parsed without
expression evaluation and are not rounded to a fixed number of decimal places.
Use the same reference scale for every harmonic coordinate. Loading a spectrum
already divided by 214 and entering `214/214`, `215/214` is valid; the software
does not itself divide the loaded frequency axis or change the units of fitted
observables. Compare/combine and replay use the same evaluated ratios.
