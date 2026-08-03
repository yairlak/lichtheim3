# Sprint 3 — word frequency: frozen analysis specification

**Frozen 2026-08-03, before any frequency summary, slope or figure was computed
or inspected.** Machine-readable twin: `_control/frequency_analysis_spec.json`.

Scope is the word-frequency effect only. Feature importance, error taxonomy,
SSP, morphology extensions and causal length-effect diagnostics are out of
scope for this sprint.

Frequency is defined **only for real words**. No pseudoword is ever assigned a
frequency value, and no pseudoword enters any model containing continuous Zipf.

## Dataset regimes

| Regime | n | Role |
|---|---|---|
| `TRAINED_REAL_FREQUENCY_PRIMARY` | 671 `TRAINED_REAL_EXACT` | **primary** |
| `TRAINED_REAL_FREQUENCY_SENSITIVITY` | 678 (671 exact + 7 pronunciation variants) | sensitivity; variant status kept explicit |
| `UNTRAINED_REAL` | 122 | analysed **separately**, never merged with trained words absent an exposure factor |
| all 800 source-labelled real words | 800 | descriptive only; never the primary Lichtheim3 frequency claim |

## Estimands

### 5.1 Continuous frequency slope (primary)

Per `seed × route × dataset_regime`, fit
`raw_edit_distance ~ standardized_zipf`, recording intercept, Zipf slope,
`n_items`, `n_erroneous_items` and model status.

**A negative Zipf slope means higher-frequency words have fewer errors.**
The word-error counterpart is computed descriptively only when estimable.

### 5.2 Length-adjusted model

Per `seed × route × dataset_regime`, fit
`raw_edit_distance ~ standardized_zipf + standardized_phoneme_length +
standardized_zipf × standardized_phoneme_length`, recording the Zipf main
effect, the length coefficient, the interaction and model status. Because both
predictors are centred, the Zipf main effect is the slope **at mean length**,
and is not interpreted without that note.

**Standardization procedure (fixed now).** Means and standard deviations are
computed once per dataset regime from the item-level values of a single seed
and route (items are identical across seeds and routes), and the resulting
standardized item covariates are reused unchanged in every seed and route model
within that regime. Nothing is re-standardized per seed, per route or per fit.

### 5.3 High versus low (secondary)

Per `seed × route × dataset_regime`,
`low_frequency_mean_error − high_frequency_mean_error` for raw edit distance
and word error. **A positive contrast means low-frequency words are harder.**
Secondary to continuous Zipf.

### 5.4 Frequency × route

Per seed and regime, compare continuous Zipf slopes across routes. Both
orientations are reported, with no sign ambiguity permitted anywhere:

```
raw_route_slope_difference        = slope_A − slope_B
frequency_benefit_route_difference = (−slope_A) − (−slope_B) = slope_B − slope_A
```

The second is oriented so that **positive means route A shows the larger
frequency benefit**. Pairs: LTM−WM, FULL−WM, LTM−FULL.

### 5.5 Frequency × lexical confidence

Per seed and regime, `lexical_confidence ~ standardized_zipf`. The outcome is
the **top-1 cosine similarity to the semantic bank** — never called word
probability, never called phonological similarity.

### 5.6 Frequency × gate

Per seed and regime, `gate ~ standardized_zipf`. The gate is word-level and
belongs to the FULL-route computation. Because the gate is a deterministic
monotonic transform of lexical confidence
(`g = sigmoid(2.0 · (confidence − 0.7))`), gate and confidence are treated as
**linked outcomes, not independent evidence**.

### 5.7 Statistical policy

Primary reporting unit is the **seed-level estimate**. Uncertainty uses the
frozen hierarchical bootstrap, unchanged since Sprint 1: resample seeds with
replacement, then items with replacement within the declared analysis set;
**B = 10,000; random seed 20260730; 95 % percentile interval**.

Every quantity reports all four seed estimates, the four-seed mean, the range,
the bootstrap interval, and the exact-zero sensitivity on seeds 19, 20 and 22.
Seed 21 is never silently excluded. P-values are optional and are never the
sole basis of a claim; unstable models are not forced.

Allowed model statuses: `OK`, `ALL_ZERO_OUTCOME`, `COMPLETE_SEPARATION`,
`INSUFFICIENT_ERRORS`, `CONSTANT_OUTCOME`, `NON_ESTIMABLE`,
`NUMERICAL_FAILURE`.

### 5.8 Ceiling policy (frozen before results)

If a `seed × route` cell has zero total edit distance and zero word errors:

- the raw edit-distance slope is structurally 0 for descriptive purposes;
- inferential status is `ALL_ZERO_OUTCOME`;
- no word-error logistic fit is attempted;
- **no claim of absence of frequency encoding is permitted.** A zero
  behavioural slope under ceiling means the measure cannot identify an effect,
  not that the route lacks frequency information.

### 5.9 Multiple comparisons

Primary planned family: continuous Zipf slopes for FULL, WM and LTM on the
primary trained set, raw edit distance. Secondary families: high/low;
Zipf × length; confidence; gate; sensitivity datasets. If p-values are
produced, Holm correction is applied within each declared family.

## Presentation

Frequency uses a **neutral colour scale**. Red and blue remain reserved for
real/pseudo lexicality and must not appear in any frequency figure. All four
seed points are shown individually. Continuous Zipf is identified as primary
and high/low as secondary. Confidence and gate are plotted on separate panels,
never on a shared y-axis.

## Result categories

`ROBUST`, `CONSISTENT_BUT_SMALL`, `INCONSISTENT_ACROSS_SEEDS`,
`CEILING_LIMITED`, `SPARSE_ERROR_LIMITED`, `NON_ESTIMABLE`,
`DESCRIPTIVE_ONLY`.

No causal claim is made, and no zero slope is presented as proof of absence.
