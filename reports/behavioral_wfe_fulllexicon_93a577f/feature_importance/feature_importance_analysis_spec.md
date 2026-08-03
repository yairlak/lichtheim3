# Sprint 5 — adapted feature importance (A15): frozen specification

**Frozen 2026-08-03, before any model was fitted or any importance value was
computed or inspected.** Machine-readable twin:
`_control/feature_importance_analysis_spec.json`.

Scope: **adapted** feature importance only. The faithful Dager analysis (A11)
already exists, is read-only here, and is never recomputed, replaced or pooled
with these results. No SSP, no causal claim, no architectural recommendation.

## 1. Identifiability constraints, fixed in advance

On `LICHTHEIM_CLEAN` (1,062 items) **lexicality and training exposure are
perfectly confounded**: every Real item is `TRAINED_REAL_EXACT` (671) and every
Pseudo item is `NOVEL_PSEUDOWORD` (391). The 2 × 2 cross-tabulation has exactly
two non-empty cells. Therefore:

- lexicality and exposure status **never enter the same clean-set model**;
- no claim is made that the clean model separates lexicality from exposure;
- the factor is named the **lexicality/exposure contrast** wherever the
  distinction matters.

Continuous Zipf frequency is **undefined for pseudowords** and is never imputed.
It is therefore **excluded from every all-item clean model**. The trained-real
frequency analysis is Sprint 3 (A14) and is not repeated here.

Morphology (simple/complex) and phoneme length (3, 4, 5, 7, 8, 9; length 6 is
absent from the WFE by construction) are valid clean-set factors.

## 2. The three analyses

| | dataset | rows per seed | raw factors | role |
|---|---|---|---|---|
| **A** clean joint main effects | `LICHTHEIM_CLEAN` | 1,062 × 3 = 3,186 | route, lexicality, length, morphology | **primary** |
| **B** clean interactions | same | same | A + five predeclared two-way blocks | secondary |
| **C** route-specific | same, split by route | 1,062 | lexicality, length, morphology | secondary |

Predeclared interaction blocks, fixed now and never selected post hoc:
`route × length`, `route × lexicality`, `route × morphology`,
`lexicality × length`, `morphology × length`. **No three-way or four-way
interaction.**

An `ALL_WITH_EXPOSURE_STRATA` model may be added as a **clearly separate
descriptive sensitivity** — exposure status in, lexicality out, frequency out,
route/length/morphology kept, n < 20 and n < 10 flagged — and is never merged
with the clean-set claim.

## 3. Outcomes

Primary: **raw Levenshtein edit distance**. Secondary descriptive: **word
error**, reported only where estimable.

## 4. Estimator

`Ridge(alpha=1.0)`. **Alpha is not tuned on the WFE**, and no hyperparameter of
any kind is searched. This matches the faithful A11 estimator, which is a
deliberate methodological continuity, not a claim that the two estimands are
comparable.

Primary score: **held-out R²**. Secondary score: **held-out mean absolute
error**. The secondary score is required because sparse, zero-heavy outcomes
can produce unstable or negative held-out R².

## 5. Data split — the leakage rule

An **80/20 train/test split grouped by `item_id`**, `random_state = 42`, drawn
once over the sorted unique clean item ids.

- **All three route rows of an item stay in the same split.**
- **The identical item split is reused across all four seeds** and across
  models A, B and C, so seed differences are model differences and not split
  differences.
- **No row-level random split is permitted.**

The exact ids are recorded in `_control/fi_train_items.tsv` and
`_control/fi_test_items.tsv`.

## 6. Standardization and encoding

Per declared dataset regime:

- numeric standardization (phoneme length) is **fitted on training items only**
  and applied unchanged to test items;
- the same item-level transformed covariates are reused across seeds;
- categorical encoding is **fitted on training data only**.

Frozen reference levels:

| factor | reference level |
|---|---|
| route | **WM** |
| lexicality | **Pseudoword** |
| morphology | **Complex** |

## 7. Grouped permutation importance

Permutation operates on **raw factors before feature engineering**, on
**held-out test data**, with **`n_repeats = 100`** and
**`random_state = 42`**.

- **Item-level factors** (lexicality, length, morphology): the factor's values
  are permuted **across held-out `item_id`s**, and the permuted value is applied
  to **all three route rows of that item**.
- **Route**: labels are permuted **within each held-out item**, preserving
  exactly one FULL, one WM and one LTM row per item.

After permuting a raw factor, **encoding, standardization and every declared
interaction term are rebuilt from it**, so permuting one factor perturbs every
model column derived from it. **Dummy columns and interaction columns are never
permuted independently.**

Importance is reported as **drop in held-out R²** and **increase in held-out
MAE**. For every seed × factor the mean, standard deviation, minimum, maximum
and the full repeat-level table are saved.

## 8. Sign policy

**Grouped permutation importance is unsigned.** No artificial single sign is
assigned to a multilevel factor such as route. Coefficients are reported
**separately** from importance:

- continuous length — standardized coefficient with its sign;
- binary lexicality and morphology — coefficient relative to the frozen
  reference level;
- route — **FULL versus WM** and **LTM versus WM**, reported as two numbers;
- any factor spanning several main or interaction columns — **never collapsed
  into one signed number**.

The faithful A11 analysis retains its historical signed convention; the adapted
analysis uses the policy above, and the two are not placed on one axis.

## 9. Interaction utility

Compared with the main-effects model **on the same held-out items**:

```
interaction_model_delta_r2  = interaction_test_r2  - main_effects_test_r2
interaction_model_delta_mae = main_effects_test_mae - interaction_test_mae
```

Positive means improvement. Per-block utility is measured by **refitting without
each predeclared block** (drop-block), never by post-hoc selection. **No
interaction is called important because one coefficient is large.**

A figure is produced **only if the interaction model shows meaningful held-out
utility in at least two seeds**; otherwise the tables record
`FIGURE_NOT_CREATED_DUE_TO_NO_STABLE_INCREMENTAL_UTILITY`.

## 10. Outcome-variance policy

Statuses: `OK`, `ALL_ZERO_OUTCOME`, `NEAR_ZERO_VARIANCE`, `NEGATIVE_TEST_R2`,
`INSUFFICIENT_ERRORS`, `NON_ESTIMABLE`, `NUMERICAL_FAILURE`.

- **All-zero outcome**: no fit, no interpretation, record `ALL_ZERO_OUTCOME`
  and state that behaviour is ceiling-limited.
- **Negative held-out R²**: **retain the result**, mark `NEGATIVE_TEST_R2`, and
  use MAE-based importance as the more interpretable sensitivity. **Never
  suppressed.**
- An unstable importance ranking is **never** read as evidence that a factor is
  not represented.

## 11. Uncertainty and stability

Reported for every quantity: all four seed estimates, the mean, the range, each
factor's rank within each seed, and rank stability across seeds.

**Within-seed permutation variability and between-seed variability are shown
separately.** The 100 permutation repeats are perturbations of one fitted model
on one test set and are **never treated as independent model seeds**.

A coarse interval may be computed by resampling the four seeds with replacement
(10,000 resamples, random seed 20260730). It is labelled
**"seed-resampling interval over four checkpoints"**. The term *hierarchical
bootstrap* is **not** reused here, because items are not resampled and the model
is not refitted — that term stays reserved for Sprints 1–4.

Exact-zero sensitivity (seeds 19, 20, 22) is reported **separately** and never
replaces the four-seed primary result.

## 12. Presentation

Factors use a **neutral palette**. **Red and blue are not used**: these figures
plot model diagnostics, not real/pseudo observations, so the reserved lexicality
colours would be misleading. All four seed points are visible, the mean is
prominent, and within-seed permutation spread is visually distinguishable from
between-seed spread. The primary visual measure is labelled **held-out R²
drop**; the MAE sensitivity appears in a companion panel or table.
Ceiling-limited routes are **labelled, never drawn as artificial zero
importance**.

## 13. Result categories

`ROBUST`, `STABLE_RANKING`, `CONSISTENT_BUT_SMALL`, `UNSTABLE_ACROSS_SEEDS`,
`NEGATIVE_TEST_R2`, `CEILING_LIMITED`, `SPARSE_ERROR_LIMITED`, `NON_ESTIMABLE`,
`DESCRIPTIVE_ONLY`.

No causal claim and no architectural recommendation.
