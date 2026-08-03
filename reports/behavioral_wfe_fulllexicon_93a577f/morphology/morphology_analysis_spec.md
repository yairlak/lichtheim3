# Sprint 2 — morphology × phoneme length: frozen analysis specification

**Frozen 2026-08-03, before any morphology summary, contrast or figure was
computed or inspected.** Machine-readable twin:
`_control/morphology_analysis_spec.json`.

Scope is morphology × phoneme length only. Frequency, feature importance,
error taxonomy, SSP and causal length-effect diagnostics are out of scope for
this sprint.

## Two separate analyses — never merged into one statistical claim

**A. Script-faithful morphology replication.** `FAITHFUL_WFE_ALL`, all 1,200
original WFE items, FULL route only, the article's own real/pseudo and
simple/complex labels, phoneme lengths 3–5 and 7–9 (no length 6). Primary
metric raw Levenshtein edit distance; word error as a secondary descriptive
output. This is a stimulus-level replication, **not** a trained-versus-novel
analysis.

**B. Lichtheim3 adapted morphology analysis.** `LICHTHEIM_CLEAN`, FULL / WM /
LTM, Real words = `TRAINED_REAL_EXACT` (671), Pseudowords =
`NOVEL_PSEUDOWORD` (391), same morphology and length factors, seed-level
estimates visible, inter-seed uncertainty, exact-zero sensitivity.

## Estimands

### Primary descriptive cells

Per `seed × route × dataset_regime × source_lexicality × morphology ×
phoneme_length`: `n_items`, mean raw edit distance, word-error rate, total raw
edit distance, number of erroneous items.

### Primary morphology contrast

Within each `seed × route × dataset_regime × source_lexicality`:

```
morphology_contrast = mean(simple) − mean(complex)
```

computed for raw edit distance and for word-error rate. **A positive contrast
means more errors for morphologically simple items.**

### Morphology × length contrast

Within each `seed × route × dataset_regime × source_lexicality`, fit the
continuous length slope separately for simple and for complex items, then:

```
morphology_length_interaction = simple_length_slope − complex_length_slope
```

**A positive value means the length effect is stronger for simple items.**

### Route contrasts — adapted analysis only

For `LICHTHEIM_CLEAN`, compare both the morphology contrast and the
morphology × length interaction across routes: `LTM − WM`, `FULL − WM`,
`LTM − FULL`.

## Statistical policy

Primary inference unit is the **seed-level estimate**. Uncertainty uses the
already-frozen hierarchical bootstrap, unchanged from Sprint 1: resample seeds
with replacement, then items with replacement within the relevant cells;
**B = 10,000; random seed 20260730; 95 % percentile interval**.

For every quantity report: each seed's estimate, the four-seed mean, the range,
the bootstrap interval, and the exact-zero sensitivity restricted to seeds
19, 20 and 22. Seed 21 is never silently excluded and appears in every primary
table and figure.

Visual divergence alone is not evidence. Where a route is at ceiling, absence
of a morphology effect is **not** claimed. A complex high-order regression is
never the sole result; optional supporting OLS models may be fitted, but the
primary claims remain the transparent seed-level contrasts above. If formal
p-values are produced for multiple planned comparisons, Holm correction is
applied within each declared analysis family; p-values are not required.

## Word-error limitation

If complete separation or ceiling prevents a stable logistic model, the fit is
not forced. Descriptive word-error contrasts and bootstrap intervals are
reported instead, and the model is marked `NON_ESTIMABLE_DUE_TO_CEILING` or
`COMPLETE_SEPARATION`.

## Small-cell policy

**No cell is excluded.** Every cell records its `n`. Flags, fixed now and not
revisable after seeing results:

| Flag | Condition |
|---|---|
| `VERY_SMALL_CELL` | n < 10 |
| `SMALL_CELL` | 10 ≤ n < 20 |
| `OK` | n ≥ 20 |

Results resting on very small cells are descriptive only.

## Visual encoding (inherited from Sprint 1, unchanged)

Real words = **red**, Pseudowords = **blue**, and those colours encode nothing
else. Morphologically complex = **solid**, simple = **dashed**. Morphology is
carried by line style alone in Figure-2A-like plots. Individual seed traces are
light and thin; the across-seed mean is prominent.

## Finding categories used in the results report

`ROBUST`, `CONSISTENT_BUT_SMALL`, `INCONSISTENT_ACROSS_SEEDS`,
`CEILING_LIMITED`, `SMALL_CELL_DESCRIPTIVE`, `NON_ESTIMABLE`.

No causal claim about why morphology does or does not matter, and no discussion
of the mechanism of the LTM length effect, will be made in this sprint.
