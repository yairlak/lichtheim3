# Sprint 2 — morphology × phoneme length: results

Analysis specification frozen in `morphology_analysis_spec.md` **before any
result below was computed or inspected**. Sign conventions are as frozen:
`morphology_contrast = mean(simple) − mean(complex)` (positive ⇒ more errors
for simple items); `morphology_length_interaction = simple_slope −
complex_slope` (positive ⇒ length effect stronger for simple items).

## 1. Scope

Morphology × phoneme length only, on outputs already validated in Sprints 0–1.
No checkpoint inference, no frequency, feature-importance, error-taxonomy, SSP
or causal analysis. Sprint-1 length, slope, serial-position and gate results are
untouched.

## 2. Faithful replication (FAITHFUL_WFE_ALL, FULL route, 1,200 items)

Original Dager real/pseudo and simple/complex labels preserved. Mean over the
four seeds, with 95 % hierarchical bootstrap intervals (B = 10,000, seed
20260730):

| Stratum | Quantity | Mean | 95 % CI | Seed signs |
|---|---|---|---|---|
| real | morphology contrast | −0.0013 | [−0.0100, +0.0069] | −, 0, −, 0 |
| real | morphology × length | −0.0009 | [−0.0063, +0.0048] | −, +, −, − |
| pseudo | morphology contrast | +0.0062 | [−0.0188, +0.0338] | +, −, 0, + |
| pseudo | morphology × length | +0.0036 | [−0.0120, +0.0207] | +, −, −, + |

Component length slopes (mean over seeds): real simple 0.0020 vs complex
0.0028; pseudo simple 0.0121 vs complex 0.0084.

Every interval contains zero and no quantity holds its sign across all four
seeds.

## 3. Clean adapted analysis (LICHTHEIM_CLEAN, three routes)

Morphology contrast, mean over seeds with 95 % CI:

| Route | Stratum | Mean | 95 % CI | Status |
|---|---|---|---|---|
| FULL | Real words | 0.0000 | [0.0000, 0.0000] | CEILING_LIMITED |
| FULL | Pseudowords | +0.0068 | [−0.0189, +0.0349] | inconsistent |
| WM | Real words | 0.0000 | [0.0000, 0.0000] | CEILING_LIMITED |
| WM | Pseudowords | −0.0021 | [−0.0418, +0.0366] | inconsistent |
| LTM | Real words | −0.0056 | [−0.0306, +0.0175] | inconsistent |
| LTM | Pseudowords | +0.0370 | [−0.1566, +0.2325] | inconsistent |

The exact zeros for real words under FULL and WM are structural: those routes
make **no errors at all** on the 671 trained real words in any seed, so a
difference of cell means is necessarily zero. This is a ceiling, not evidence
that morphology has no effect there.

## 4. Cell counts and balance

`tables/faithful_morphology_cell_counts.tsv`,
`tables/clean_morphology_cell_counts.tsv`,
`tables/morphology_cell_balance_summary.tsv`.

Faithful (FULL): 24 cells, all n ≥ 20, total 1,200. Cells are **not** balanced
across exact phoneme lengths — the WFE guarantees 100 items per each of the 12
broad experimental conditions, not per (lexicality × morphology × exact
length) cell. Real complex ranges 29–130 items and real simple 28–124.

Clean: 22 cells OK plus **2 SMALL_CELL** — real/complex/length-3 (n = 17) and
pseudo/complex/length-3 (n = 19). No VERY_SMALL_CELL. Filtering to
`TRAINED_REAL_EXACT` and `NOVEL_PSEUDOWORD` shrinks every real cell
(e.g. real complex length-5 130 → 101; real simple length-7 124 → 112) and
trims pseudo cells slightly (pseudo complex length-3 21 → 19). The remaining
design is **not** balanced and was not rebalanced.

## 5. Main morphology contrasts

No morphology contrast is distinguishable from zero in either analysis. Every
one of the ten estimable intervals contains zero, and `all_same_sign` is
`False` for every stratum × route combination — the sign of the contrast flips
across the four seeds in every case.

The numerically largest estimate, LTM pseudowords (+0.0370), has by far the
widest interval ([−0.157, +0.233]) and per-seed values of +0.045, −0.028,
+0.085, +0.044 — one seed reverses.

## 6. Morphology × length

| Route | Stratum | Mean | 95 % CI |
|---|---|---|---|
| FULL | Pseudowords | +0.0039 | [−0.0128, +0.0214] |
| WM | Pseudowords | −0.0014 | [−0.0266, +0.0228] |
| LTM | Pseudowords | −0.0207 | [−0.1181, +0.0748] |
| LTM | Real words | −0.0023 | [−0.0211, +0.0134] |
| FULL / WM | Real words | 0.0000 | [0.0000, 0.0000] (ceiling) |

For LTM pseudowords — the only cell with a substantial length effect at all —
the component slopes are simple 0.2112 and complex 0.2322. The two morphology
classes show length effects of very similar magnitude; their difference is not
separable from zero and reverses across seeds (+0.022, −0.083, +0.014, −0.037).

## 7. Morphology × route

Route contrasts of the morphology effect (clean set, mean over seeds, range in
brackets):

| Contrast | Stratum | Morphology contrast diff. | Morphology × length diff. |
|---|---|---|---|
| LTM − WM | Pseudowords | +0.0389 [−0.033, +0.089] | −0.0196 [−0.087, +0.016] |
| LTM − FULL | Pseudowords | +0.0301 [−0.018, +0.085] | −0.0249 [−0.076, +0.014] |
| FULL − WM | Pseudowords | +0.0088 [−0.016, +0.031] | +0.0052 [−0.011, +0.020] |
| LTM − WM | Real words | −0.0055 [−0.015, +0.002] | −0.0022 [−0.010, +0.002] |
| LTM − FULL | Real words | −0.0055 [−0.015, +0.002] | −0.0022 [−0.010, +0.002] |
| FULL − WM | Real words | 0.0000 | 0.0000 (ceiling both sides) |

Every route contrast spans zero across seeds. Nothing here supports a claim
that morphology behaves differently in one route than another.

## 8. Four-seed consistency

`all_same_sign` is `False` for **every** morphology quantity in both analyses.
Seed 21 is present in every table and figure and is not an outlier: its values
sit inside the range set by the other three in all cells.

## 9. Exact-zero sensitivity

`clean_adapted/tables/clean_morphology_exact_zero_sensitivity.tsv`. Restricting
to seeds 19, 20 and 22 does not change any conclusion: e.g. LTM pseudoword
contrast 0.0367 → 0.0206, LTM pseudoword interaction −0.0210 → −0.0325, FULL
pseudoword contrast 0.0066 → 0.0087. All remain small relative to their spread.

## 10. Word-error secondary results

`tables/faithful_morphology_word_error.tsv`,
`tables/clean_morphology_word_error.tsv`,
`tables/morphology_word_error_bootstrap.tsv`.

Word-error rates by morphology (mean over seeds): FULL pseudo 0.0088 complex /
0.0101 simple; WM pseudo 0.0151 / 0.0114; LTM pseudo 0.2240 / 0.2405; LTM real
0.0061 / 0.0061; FULL and WM real 0.0000 / 0.0000. Bootstrap contrasts span
zero everywhere (largest LTM pseudo +0.0252 [−0.0412, +0.0922]).

**FIGURE_NOT_CREATED_DUE_TO_CEILING.** No word-error figure was produced. The
reason, recorded before any such figure was attempted: real words under FULL
and WM sit at exactly 0.0000 error in all seeds and all cells, LTM real is a
flat 0.0061, and the pseudoword differences are an order of magnitude smaller
than their intervals. A figure would show a floor, not a pattern. The
supporting cell tables above are provided in full instead.

A logistic model of word error on morphology is marked
**COMPLETE_SEPARATION** for real words under FULL and WM (zero errors in every
cell, so no finite maximum-likelihood estimate exists) and was not forced.

## 11. Ceiling and small-cell limitations

- **Ceiling.** FULL and WM produce zero errors on all 671 trained real words in
  every seed. Morphology contrasts there are structurally zero and carry no
  information; absence of a morphology effect is not claimed for those cells.
- **Small cells.** Two clean cells fall in 10 ≤ n < 20 (real complex length-3
  n = 17; pseudo complex length-3 n = 19) and are flagged `SMALL_CELL`. They
  were retained; no item was excluded on account of a flag. Any length-3
  complex reading is descriptive only.
- **Unbalanced design.** Exact-length cells vary by a factor of ~7 in the
  faithful set and ~6 in the clean set. No weighting or resampling was applied
  to make them look balanced.

## 12. Robust findings

**None.** No morphology contrast or morphology × length interaction reaches the
`ROBUST` bar (consistent sign across all four seeds *and* an interval excluding
zero) in either analysis, in any route or stratum.

## 13. Null or non-estimable findings

| Finding | Category |
|---|---|
| Morphology contrast, real words, FULL and WM (clean) | `CEILING_LIMITED` |
| Morphology × length, real words, FULL and WM (clean) | `CEILING_LIMITED` |
| Word-error logistic, real words, FULL and WM | `COMPLETE_SEPARATION` / `NON_ESTIMABLE` |
| Morphology contrast, pseudowords, all routes | `INCONSISTENT_ACROSS_SEEDS` |
| Morphology × length, pseudowords, all routes | `INCONSISTENT_ACROSS_SEEDS` |
| Morphology contrast and interaction, faithful set, both strata | `INCONSISTENT_ACROSS_SEEDS` |
| Morphology contrast, real words, LTM (clean) | `INCONSISTENT_ACROSS_SEEDS` |
| Route contrasts of morphology (all three, both strata) | `INCONSISTENT_ACROSS_SEEDS` |
| Length-3 complex cells (clean, both strata) | `SMALL_CELL_DESCRIPTIVE` |

No finding qualifies as `CONSISTENT_BUT_SMALL`: that category requires a stable
sign across seeds, which no morphology quantity achieved.

## 14. Exploratory observations

Recorded as observations only, not claims, and not to be carried into a
conclusion without further work:

- In the faithful set the pseudoword length slope is numerically larger for
  simple than complex items (0.0121 vs 0.0084), but the difference is smaller
  than its seed-to-seed spread.
- Under LTM the pseudoword slopes are nearly equal (simple 0.2112, complex
  0.2322), so the large LTM length effect established in Sprint 1 does not
  appear to be concentrated in one morphological class.
- The clean-set filter removes proportionally more real complex items than real
  simple items at short lengths, which is why the length-3 complex cells became
  the only flagged cells.

No causal account of why morphology does or does not matter is offered, and the
mechanism of the LTM length effect is not discussed.

## 15. Files generated

```
morphology/
  morphology_analysis_spec.md              frozen before results
  morphology_results.md                    this file
  _control/  morphology_preflight.json, morphology_analysis_spec.json,
             morphology_figure_manifest.json
  tables/    faithful_morphology_cell_counts.tsv
             clean_morphology_cell_counts.tsv
             morphology_cell_balance_summary.tsv
             faithful_morphology_word_error.tsv
             clean_morphology_word_error.tsv
             morphology_word_error_bootstrap.tsv
  faithful_replication/
    figures/ faithful_length_lexicality_morphology.{png,pdf,svg} + caption
    tables/  ..._plot.tsv, ..._seed_contrasts.tsv,
             ..._length_interactions.tsv, ..._bootstrap.tsv
  clean_adapted/
    figures/ clean_length_morphology_by_route.{png,pdf,svg} + caption
    tables/  ..._plot.tsv, ..._seed_contrasts.tsv,
             ..._length_interactions.tsv, ..._route_contrasts.tsv,
             ..._bootstrap.tsv, ..._exact_zero_sensitivity.tsv
  validation/ morphology_validation.json, morphology_test_log.txt,
              morphology_output_inventory.tsv, morphology_outputs.sha256,
              morphology_diff_review.md
```
