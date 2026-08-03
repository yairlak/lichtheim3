# Length-effect mechanism — factual handoff

**This is a handoff, not an interpretation.** It collects, with exact paths, the
measurements a mechanistic follow-up would need. It contains no causal claim, no
mechanism, no feature importance, no SSP and no architectural recommendation.
Nothing here answers the questions in the final section.

Cohort `fulllexicon_cohort_93a577f`, training commit `93a577f`, seeds 19, 20,
21, 22. All paths are relative to `reports/behavioral_wfe_fulllexicon_93a577f/`.

---

## 1. The length effect as already validated (Sprint 1)

Slope of raw Levenshtein edit distance on continuous phoneme length, clean set.

| route | lexicality | seed 19 | seed 20 | seed 21 | seed 22 |
|---|---|---:|---:|---:|---:|
| LTM | pseudo | 0.2560 | 0.1973 | 0.2213 | 0.2114 |
| WM | pseudo | 0.0101 | 0.0145 | 0.0375 | 0.0067 |
| FULL | pseudo | 0.0151 | 0.0034 | 0.0202 | 0.0034 |
| LTM | real | 0.0177 | 0.0054 | 0.0076 | 0.0007 |
| WM / FULL | real | 0 | 0 | 0 | 0 |

`LTM − WM` on pseudowords: +0.2459, +0.1828, +0.1838, +0.2047 — positive in all
four seeds.

- `figures/clean_length_slopes_by_seed.tsv`
- `figures/clean_route_length_contrasts.tsv`
- `figures/yair_clean_length_by_route.{png,pdf,svg,tsv}`
- `figures/yair_clean_length_slopes.{png,pdf,svg}`

Serial-position profile, already validated and unchanged by this sprint:

- `figures/yair_clean_serial_position.{png,pdf,svg,tsv}`
- `figures/yair_clean_serial_position_interpolated.tsv`
- method: faithful zip-mismatch positions (Dager `Error_Indices`), relative
  position (index−1)/(length−1), PCHIP interpolation to 100 points, item-count
  weighted across lengths; **no Levenshtein alignment is used**

---

## 2. Operation counts by route, lexicality and Short/Long

`error_taxonomy/clean/tables/clean_error_taxonomy_cells.tsv` (per seed),
`…/clean_error_taxonomy_summary.tsv` (mean, range, per-seed values).

Mean operations per evaluated item, mean over seeds; Short = 3, 4, 5;
Long = 7, 8, 9.

| route | lexicality | length | subs | dels | ins | total |
|---|---|---|---:|---:|---:|---:|
| LTM | pseudo | Long | 0.5738 | 0.2450 | 0.1400 | 0.9588 |
| LTM | pseudo | Short | 0.0982 | 0.0236 | 0.0183 | 0.1401 |
| WM | pseudo | Long | 0.0250 | 0.0150 | 0.0125 | 0.0525 |
| WM | pseudo | Short | 0 | 0 | 0 | 0 |
| FULL | pseudo | Long | 0.0175 | 0.0088 | 0.0050 | 0.0313 |
| FULL | pseudo | Short | 0 | 0 | 0 | 0 |
| LTM | real | Long | 0.0220 | 0.0015 | 0.0015 | 0.0250 |
| LTM | real | Short | 0.0022 | 0 | 0 | 0.0022 |
| WM / FULL | real | both | 0 | 0 | 0 | 0 |

Item counts per cell (identical in every seed): pseudo Short 191, pseudo Long
200, real Short 341, real Long 330.

---

## 3. Exact-length operation profiles

`error_taxonomy/clean/tables/clean_error_taxonomy_by_exact_length.tsv` — per
seed × route × lexicality × phoneme length ∈ {3, 4, 5, 7, 8, 9}, with
`n_items`, `n_erroneous_items`, `cell_flag`, per-operation totals, means per
item, means per erroneous item and proportions. Length 6 is absent from the WFE
by construction.

---

## 4. Substitutions, deletions and insertions separately

Route contrasts, per operation, at seed level:

- `error_taxonomy/clean/tables/clean_error_taxonomy_route_contrasts.tsv`
- `…/clean_error_taxonomy_route_contrasts_summary.tsv`

Clean pseudowords, mean over seeds [min, max]:

| contrast | subs | dels | ins | total |
|---|---|---|---|---|
| LTM − WM | +0.3286 [0.3069, 0.3402] | +0.1292 [0.1074, 0.1611] | +0.0742 [0.0435, 0.0895] | +0.5320 [0.4910, 0.5857] |
| LTM − FULL | +0.3325 [0.3120, 0.3453] | +0.1324 [0.1176, 0.1560] | +0.0780 [0.0563, 0.0895] | +0.5428 [0.5192, 0.5780] |
| FULL − WM | −0.0038 [−0.0051, 0.0000] | −0.0032 [−0.0102, 0.0051] | −0.0038 [−0.0128, 0.0026] | −0.0109 [−0.0281, 0.0077] |

Composition (secondary): `…/clean_error_taxonomy_composition.tsv`, with
`NO_ERRORS_FOR_COMPOSITION` on zero-operation cells.

Bootstrap intervals: `…/clean_error_taxonomy_bootstrap.tsv`
(B = 10,000, random seed 20260730, 95 % percentile).

Caveat carried forward: `Levenshtein.editops` tie-breaking can move counts
between the three operation types **without changing the total edit distance**.

---

## 5. Premature-EOS event counts by route

`error_taxonomy/eos/tables/premature_eos_by_seed_route.tsv`,
`…/premature_eos_class_counts.tsv`.

**87 events in total.** LTM 82, FULL 3, WM 2. All on pseudowords; zero on
trained real words in any route or seed.

| route | seed 19 | seed 20 | seed 21 | seed 22 | total |
|---|---:|---:|---:|---:|---:|
| LTM (pseudo) | 26 | 14 | 24 | 18 | 82 |
| FULL (pseudo) | 2 | 0 | 1 | 0 | 3 |
| WM (pseudo) | 1 | 1 | 0 | 0 | 2 |
| all routes (real) | 0 | 0 | 0 | 0 | 0 |

Observed class totals: `ON_TIME_EOS` 0, `LATE_EOS` 0, `EOS_UNAVAILABLE` 0,
`EOS_NOT_OBSERVED` 4,245 (FULL) / 4,246 (WM) / 4,166 (LTM).

---

## 6. Premature-EOS rate by exact length

`error_taxonomy/eos/tables/premature_eos_by_exact_length.tsv`;
broad grouping in `…/premature_eos_by_broad_length.tsv`.

Pseudowords, mean over seeds:

| length | n items | LTM rate | FULL rate | WM rate |
|---:|---:|---:|---:|---:|
| 3 | 47 | 0.0053 | 0 | 0 |
| 4 | 78 | 0.0064 | 0 | 0 |
| 5 | 66 | 0.0038 | 0 | 0 |
| 7 | 66 | 0.0341 | 0 | 0 |
| 8 | 68 | 0.0699 | 0 | 0 |
| 9 | 66 | 0.1894 | 0.0114 | 0.0076 |

Linear-probability length slopes, `error_taxonomy/eos/tables/premature_eos_length_slopes.tsv`:

| route | lexicality | 19 | 20 | 21 | 22 | status |
|---|---|---:|---:|---:|---:|---|
| LTM | pseudo | +0.0356 | +0.0177 | +0.0316 | +0.0198 | `OK` ×4 |
| FULL | pseudo | +0.0034 | 0 | +0.0017 | 0 | `INSUFFICIENT_EVENTS` / `ALL_ZERO_OUTCOME` |
| WM | pseudo | +0.0017 | +0.0017 | 0 | 0 | `INSUFFICIENT_EVENTS` / `ALL_ZERO_OUTCOME` |
| all | real | 0 | 0 | 0 | 0 | `ALL_ZERO_OUTCOME` |

**How much behaviour carries an observed early stop — stated as counts.**
Across the four seeds, on clean pseudowords under LTM:

| quantity | value |
|---|---:|
| observed premature-EOS events | **82** |
| erroneous items | 365 |
| share of erroneous items with an observed premature EOS | **≈ 22 %** |
| deletion-bearing items | 189 |
| share of deletion-bearing items with an observed premature EOS | **43.4 %** |
| total edit operations | **874** |

These are co-occurrence counts. They say how often an observed early stop
accompanies erroneous behaviour; they do not say that it produces any of it.

> **Technical note — not a decomposition.** The LTM premature-EOS length slope
> (0.0177–0.0356 **events** per phoneme) and the LTM edit-distance length slope
> (0.1973–0.2560 **operations** per phoneme, §1) are outcomes in **different
> units**, fitted separately over the same items. Their numerical ratio is
> therefore **not a fraction of the length effect explained**, not a
> decomposition and not an attribution, and it is deliberately kept out of every
> summary and finding in this sprint. Matching events to operations item by item
> — which this sprint does not do — is open question 7 below.

---

## 7. EOS shortfall

`eos_shortfall = expected_eos_position − observed_eos_position`, positive means
early; range [1, L].

- **All-item mean shortfall** (primary; zero assigned to non-premature items as
  a summary convention that does not relabel their class) —
  `error_taxonomy/eos/tables/premature_eos_by_seed_route.tsv`, column
  `mean_eos_shortfall_per_item`. LTM pseudowords: 0.0742, 0.0384, 0.0639,
  0.0486. FULL ≤ 0.0051, WM ≤ 0.0026, trained real words 0.
- **Conditional mean shortfall** (secondary; denominator is `PREMATURE_EOS`
  items only) — same table, column `conditional_mean_eos_shortfall`. LTM
  pseudowords: 1.115, 1.071, 1.042, 1.056. FULL and WM exactly 1.000 where an
  event exists, undefined otherwise.

Summaries with per-seed values, ranges and exact-zero-seed means:
`…/premature_eos_by_seed_route_summary.tsv`.

---

## 8. EOS / deletion 2 × 2 overlap

`error_taxonomy/eos/tables/premature_eos_deletion_overlap.tsv` and
`…/premature_eos_deletion_overlap_by_length.tsv`.

Pooled over seeds, clean pseudowords (1,564 seed × item rows per route):

| route | EOS ∧ del | EOS ∧ no del | del ∧ no EOS | neither | P(del \| EOS) | P(EOS \| del) |
|---|---:|---:|---:|---:|---:|---:|
| FULL | 3 | 0 | 4 | 1,557 | 1.000 | 0.429 |
| WM | 2 | 0 | 8 | 1,554 | 1.000 | 0.200 |
| LTM | 82 | 0 | 107 | 1,375 | 1.000 | 0.434 |

By broad length, LTM: Long 78 of 171 deletion-bearing items; Short 4 of 18.

**Neither probability is causal.** `P(deletion | premature EOS) = 1.000` is a
structural consequence of trimming the prediction at the first EOS — the output
is then shorter than the target, so any alignment must contain at least one
deletion. `P(premature EOS | deletion) = 0.434` under LTM means the majority of
deletion-bearing items (107 of 189) carry **no** observed premature EOS.

---

## 9. Four-seed estimates and exact-zero sensitivity

Every summary table carries a `seed_values` column with all four per-seed
values, plus `mean_over_seeds`, `min`, `max`, `range`, `seed21_included` and
`exact_zero_seeds_mean` (seeds 19, 20, 22 only, reported separately and never
substituted for the four-seed mean).

Bootstrap policy, unchanged since Sprint 1: hierarchical — resample seeds with
replacement, then items with replacement within each analysis-set × stratum
cell; **B = 10,000, random seed 20260730, 95 % percentile interval**.

---

## 10. Seed-22 item illustrations

`error_taxonomy/examples/` — `seed22_illustrative_pseudoword_errors.tsv` (22
rows: LTM 20, FULL 1, WM 1), `seed22_illustrative_premature_eos.tsv` (18 rows,
all LTM), and `README.md` describing the frozen deterministic ordering.

Columns include `target`, `prediction`, `target_length`, `predicted_length`,
`eos_position`, `expected_eos_position`, `eos_class`, `eos_shortfall` and the
three operation counts, so an item can be inspected end to end.

**Deterministic illustrations, not a representative sample.** No claim rests on
them.

---

## 11. Limitations that constrain any mechanistic follow-up

- **Forced-length readout.** The prediction is bounded by the gold target
  length. **Terminal insertions beyond the horizon are unobservable**, so
  insertion counts are a lower bound.
- **EOS observability.** The readout window holds exactly L tokens at indices
  0 … L−1, so an EOS at the correct boundary (index L) is outside it. Only
  `PREMATURE_EOS` is positively observable; `ON_TIME_EOS` and `LATE_EOS` are
  **structurally unobservable**; `EOS_NOT_OBSERVED` means only that **no EOS was
  observed within the instrumented evaluation horizon** and is **ambiguous with
  respect to eventual stopping**. Zero observed on-time events does not mean the
  decoder never stops correctly; zero observed late events does not mean late
  stopping never occurs.
- **Tie-breaking.** The insertion/deletion/substitution split is
  `Levenshtein.editops` 0.27.3–dependent; the total edit distance is not.
- **Ceiling.** Trained real words produce exactly zero operations under FULL and
  WM in every seed, so route comparisons there are structurally zero.
- **Sparsity.** FULL and WM carry 15–42 operations and 2–3 EOS events over four
  seeds; their internal structure is not estimable.
- **Small cells.** Three exposure categories have n ≤ 7 and are descriptive
  only. No category was excluded.
- A mechanistic follow-up that needs on-time or late EOS timing, or terminal
  insertions, **requires a change to the evaluation horizon**, which is out of
  scope here and would invalidate byte-comparison against the frozen cohort
  unless run as a separate, separately provenanced evaluation.

---

## QUESTIONS LEFT OPEN

Factual questions only. **None of them is answered in this document or in
`error_taxonomy_results.md`.**

1. Which internal stage generates the non-EOS substitutions that dominate the
   LTM pseudoword error profile?
2. Are early-EOS events preceded by degraded hidden representations, and if so
   at which layer and from which position?
3. Why does the operation burden grow with length among items that show **no**
   observed premature EOS — the 107 of 189 deletion-bearing LTM pseudoword items
   in §8, and the erroneous items with substitutions only?
4. Does the LTM encoder, the LTM decoder, or the semantic representation
   contribute most to the length dependence?
5. Is the substitution profile positionally structured in the same way as the
   already-validated serial-position curve, and does it differ between items
   with and without an observed premature EOS?
6. Under an extended readout horizon, what is the on-time and late EOS
   distribution that the current instrumentation cannot see, and how many of the
   4,166 LTM `EOS_NOT_OBSERVED` rows correspond to correct stopping?
7. How much of the 0.197–0.256 LTM edit-distance length slope co-occurs with an
   observed early stop, once the events are matched item by item rather than
   compared as aggregate slopes?
8. Do the 122 untrained real words, which pattern with novel pseudowords under
   LTM, show the same operation composition and the same EOS profile as novel
   pseudowords, or a distinguishable one?
