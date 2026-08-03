# Faithful (A11) versus adapted (A15) feature importance

**This is a methodological comparison, not a pooled result.** The two analyses
estimate different things on different data with different permutation
semantics. Their numbers are **never placed on one quantitative axis**, and no
combined ranking, difference or average is computed anywhere in this sprint.

## 1. What each analysis is

| | **Faithful A11** | **Adapted A15** |
|---|---|---|
| item set | 1,200 source-labelled WFE items | 1,062 clean items (671 `TRAINED_REAL_EXACT`, 391 `NOVEL_PSEUDOWORD`) |
| routes | FULL only | FULL, WM and LTM |
| route as a factor | absent | present in models A and B; models C are fitted per route |
| labels | original Dager Real/Pseudo coding | training-exposure-verified labels |
| exposure correction | none — 122 source-Real items were never trained and 9 source-Pseudo items collide with training forms | items with ambiguous exposure are excluded from the clean set by the frozen Sprint-1 definition |
| factors | lexicality, length, morphology | lexicality/exposure, length, morphology (+ route) |
| frequency | absent | excluded, because Zipf is undefined for pseudowords and is never imputed |
| split | historical 80/20, `random_state = 42` | 80/20 **grouped by `item_id`**, `random_state = 42`, identical split reused across all four seeds and all three models |
| permutation | historical procedure, `n_repeats = 100`, `random_state = 42` | **factor-level grouped** permutation on raw factors, `n_repeats = 100`, `random_state = 42` |
| sign | historical signed convention retained | grouped importance **unsigned**; coefficients reported separately |
| interactions | none | five predeclared two-way blocks, drop-block utility |
| estimator | Ridge, alpha = 1.0 | Ridge, alpha = 1.0 |
| location | `outputs/.../behavioral_analysis/faithful_replication/faithful_figure2B*` | `reports/.../feature_importance/` |

## 2. Why the values are not comparable

Four differences each break comparability on their own.

1. **Different item populations.** A11 includes the 122 untrained "real" words
   and the 9 pseudoword/training collisions; A15 excludes them. Those items
   carry a large share of A11's real-word errors, so its lexicality factor is
   partly an exposure factor with an unknown mixing weight.
2. **Different route scope.** A11 has no route factor at all; A15's primary
   model contains three routes, and route turns out to be one of its two
   strongest factors. An importance value computed with route absent cannot be
   compared with one computed with route present.
3. **Different split units.** A15's split is grouped by item, so no item's route
   rows straddle train and test. A11 predates that constraint. A grouped split
   is strictly more conservative and generally yields lower held-out scores.
4. **Different permutation semantics.** A15 permutes **raw factors** and rebuilds
   every encoded, standardized and interaction column derived from them, and
   permutes route **within** an item. A11 uses the historical column-level
   procedure. The two therefore answer different questions about what a factor
   contributes.

## 3. What may legitimately be said

- Both analyses are reported, in their own sections, with their own estimands
  and their own caveats.
- Qualitative agreement or disagreement in the **ordering** of factors that
  both analyses contain may be described in words, provided the differences in
  §2 are stated in the same breath and no shared numeric axis is drawn.
- On that qualitative level the two **disagree about the ordering of length and
  lexicality**. A11 places **length first in all four seeds**; A15 places
  **lexicality/exposure above length in all four seeds**. A15 places morphology
  last in all four seeds, whereas in A11 morphology exceeds lexicality in seeds
  20 and 22 — cells where both values are near zero and A11's lexicality
  importance is slightly negative.

  This divergence is expected and is **not** evidence that either analysis is
  wrong. The two most likely contributors are listed in §2 and are not
  separable here: A15's lexicality factor is a lexicality/exposure contrast on a
  set where the two coincide exactly, while A11's lexicality factor is diluted
  by the 122 untrained "real" items; and A15's models contain route, which A11
  does not. This is a description of two separate analyses, **not** a validation
  of one by the other, and it carries no causal reading.

## 4. What is forbidden

- Pooling, averaging or differencing A11 and A15 importance values.
- Plotting them on one axis or in one table of comparable magnitudes.
- Presenting A15 as a correction, replacement or improvement of A11 — A11 is a
  faithful replication and remains valid on its own terms.
- Presenting A11 as validation of A15, or vice versa.
- Recomputing, overwriting or re-rendering any A11 output. A11 was read only to
  record its file hashes in the Sprint-5 preflight; its three files are
  unchanged.
