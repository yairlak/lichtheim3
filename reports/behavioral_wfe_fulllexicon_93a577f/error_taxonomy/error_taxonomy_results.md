# Sprint 4 — error taxonomy and premature EOS: results

Cohort `fulllexicon_cohort_93a577f`, training commit `93a577f`, seeds 19, 20,
21, 22 (epochs 155, 130, 145, 140). Every number below comes from the canonical
seed × item × route table produced in the production evaluation. **No checkpoint
was loaded, no inference was run and no sequence was re-aligned in this sprint.**

Specification frozen before any summary was computed:
`error_taxonomy_analysis_spec.md`; EOS convention audited first and frozen
separately in `eos_convention.md`.

---

## 1. Scope

Two analyses are reported, and they are kept apart at every level — separate
modules, separate directories, separate tables, separate claims:

1. the **Levenshtein taxonomy** — substitutions, deletions, insertions;
2. the **premature-EOS decoder diagnostic**;
3. their **descriptive relationship** — overlap counts and conditional
   probabilities, with no causal reading.

Out of scope by instruction: feature importance, SSP, ablations, new morphology
or frequency inference, causal explanation of the length effect, and any
architectural recommendation.

---

## 2. Levenshtein operation convention

Operations are exactly **substitution, deletion, insertion**, taken from the
counts produced by `Levenshtein.editops` version 0.27.3 (distribution
`Levenshtein`, rapidfuzz-backed) during the production evaluation. **No fourth
operation is introduced anywhere in this sprint**, and premature EOS is never
counted as one.

Two measurement limits apply to every operation number in this report:

- **Tie-breaking.** `editops` tie-breaking can move counts between substitution,
  deletion and insertion **without changing the total edit distance**. The
  split is backend-dependent; the total is not.
- **Forced-length readout.** Each prediction is bounded by the gold target
  length, so **terminal insertions beyond the target horizon are unobservable**
  and insertion counts are a lower bound.

Preflight confirmed that substitutions + deletions + insertions reconstruct the
stored raw edit distance for **all 14,400 rows** of the canonical table.

---

## 3. EOS instrumentation convention and observability

The convention was audited from committed source before any EOS distribution was
read (`eos_convention.md`). `eos_position` is a **0-based index into the item's
readout window**, equal to the number of phonemes emitted before EOS; BOS is not
counted; the expected boundary for a target of length L is index **L**; the
window is `dec_input[i, 1:n_steps+1]` and therefore holds exactly L tokens at
indices 0 … L−1.

An EOS at the correct boundary would occupy index L, one past the end of the
window. Consequently, **with the current instrumentation**:

- only `PREMATURE_EOS` is positively observable;
- `ON_TIME_EOS` is **structurally unobservable**;
- `LATE_EOS` is **structurally unobservable**;
- `EOS_NOT_OBSERVED` means exactly **"no EOS was observed within the
  instrumented evaluation horizon"**.

The frozen class labels are retained unchanged; only their observability is
clarified. In particular:

- zero observed `ON_TIME_EOS` **does not** mean the decoder never stops
  correctly;
- zero observed `LATE_EOS` **does not** mean late stopping never occurs;
- `EOS_NOT_OBSERVED` is **ambiguous with respect to eventual stopping** and is
  never read as correct stopping, on-time stopping, successful completion, or
  the absence of an EOS-related problem.

Observed class counts across 4 seeds × 3 routes × 1,062 clean items
(`eos/tables/premature_eos_class_counts.tsv`):

| route | PREMATURE_EOS | EOS_NOT_OBSERVED | ON_TIME_EOS | LATE_EOS | EOS_UNAVAILABLE |
|---|---:|---:|---:|---:|---:|
| FULL | 3 | 4,245 | 0 | 0 | 0 |
| WM | 2 | 4,246 | 0 | 0 | 0 |
| LTM | 82 | 4,166 | 0 | 0 | 0 |

The two structural zeros are a property of the readout horizon, not a finding
about the model.

---

## 4. Faithful Figure-8A replication

`FAITHFUL_WFE_ALL` (1,200 items), FULL route, the original twelve WFE conditions
in source order, all four seeds. Figure:
`faithful/figures/faithful_figure8a_error_types.*`; table:
`faithful/tables/faithful_condition_error_types.tsv`.

Mean operations per item, averaged over seeds — only four of twelve conditions
are non-zero:

| condition | source label | size | morphology | subs | dels | ins | total |
|---|---|---|---|---:|---:|---:|---:|
| RLCL | Real | long | complex | 0.0000 | 0.0125 | 0.0050 | 0.0175 |
| RLSL | Real | long | simple | 0.0125 | 0.0000 | 0.0000 | 0.0125 |
| PLS | Pseudo | long | simple | 0.0150 | 0.0125 | 0.0100 | 0.0375 |
| PLC | Pseudo | long | complex | 0.0200 | 0.0050 | 0.0000 | 0.0250 |

All eight remaining conditions — every short condition, and the two
high-frequency long real conditions — are exactly zero in all four seeds.

This is a faithful stimulus-and-metric replication adapted to four Lichtheim3
checkpoints, **not** a reproduction of the SWP model. The Real/Pseudo labels
here are **WFE source labels, not training exposure**: 122 source-Real items were
never trained and 9 source-Pseudo items collide with training forms, which is
why the clean-set analysis is reported separately. Dager Figures 8B and 8C
concern ablated SWP models and are out of scope.

---

## 5. Clean error taxonomy: overall picture

`LICHTHEIM_CLEAN` — 671 trained real words with the same phonological form, 391
novel pseudowords. Figure `clean/figures/clean_error_taxonomy_by_route.*` uses
**one common absolute y-scale across the three route panels**.

Erroneous items and total operations, summed over the four seeds
(`clean/tables/clean_error_taxonomy_cells.tsv`):

| route | lexicality | erroneous items | total operations |
|---|---|---:|---:|
| FULL | pseudo | 15 | 25 |
| WM | pseudo | 21 | 42 |
| LTM | pseudo | 365 | 874 |
| FULL | real | 0 | 0 |
| WM | real | 0 | 0 |
| LTM | real | 14 | 36 |

**The LTM route carries a much larger operation burden on pseudowords than FULL
or WM, and the burden is concentrated in Long items.** FULL and WM remain far
lower on the same absolute scale. Trained real words are at or near floor
everywhere: exactly zero errors under FULL and WM in every seed, and 14
erroneous items in total under LTM.

Because the LTM/FULL-WM magnitude ratio is large, the frozen zoom rule was
evaluated (`clean/tables/clean_error_taxonomy_zoom_rule.tsv`):

```
max mean LTM operation per item        0.57375
max mean FULL/WM operation per item    0.02500
ratio                                  22.95
trigger                                ratio > 10
rule_fires                             True
zoom_replaces_primary                  False
```

The rule fired, so the explicitly labelled companion figure
`clean/figures/clean_error_taxonomy_full_wm_zoom.*` (FULL and WM only) was
produced **in addition to** the common-scale primary figure, which remains the
primary presentation.

---

## 6. Substitution results

Mean substitutions per evaluated item (never conditioned on being erroneous),
mean over seeds with the four-seed range
(`clean/tables/clean_error_taxonomy_summary.tsv`):

| route | lexicality | Short (3,4,5) | Long (7,8,9) |
|---|---|---|---|
| LTM | pseudo | 0.0982 [0.0838, 0.1309] | **0.5738 [0.4950, 0.6250]** |
| WM | pseudo | 0.0000 [0, 0] | 0.0250 [0.0150, 0.0450] |
| FULL | pseudo | 0.0000 [0, 0] | 0.0175 [0.0100, 0.0350] |
| LTM | real | 0.0022 [0, 0.0088] | 0.0220 [0.0061, 0.0424] |
| WM / FULL | real | 0.0000 | 0.0000 |

**Substitutions are the dominant LTM pseudoword operation.** They exceed
deletions and insertions in every route × lexicality × length cell that has any
errors at all. The Long-versus-Short separation is complete: the *minimum*
across seeds for LTM pseudoword Long (0.495) is above the *maximum* across seeds
for LTM pseudoword Short (0.131).

---

## 7. Deletion results

| route | lexicality | Short | Long |
|---|---|---|---|
| LTM | pseudo | 0.0236 [0.0157, 0.0314] | **0.2450 [0.2200, 0.3100]** |
| WM | pseudo | 0.0000 | 0.0150 [0.0000, 0.0350] |
| FULL | pseudo | 0.0000 | 0.0088 [0.0000, 0.0200] |
| LTM | real | 0.0000 | 0.0015 [0.0000, 0.0030] |

**Deletions are also elevated in the LTM route on pseudowords**, at roughly 43 %
of the substitution count in the Long cell, and again the Long/Short separation
is complete across seeds (minimum Long 0.220 > maximum Short 0.031). Deletions
are a Levenshtein alignment operation; they are **not** the same event as a
premature EOS (§16).

---

## 8. Insertion results

| route | lexicality | Short | Long |
|---|---|---|---|
| LTM | pseudo | 0.0183 [0.0105, 0.0262] | **0.1400 [0.1050, 0.1700]** |
| WM | pseudo | 0.0000 | 0.0125 [0.0000, 0.0350] |
| FULL | pseudo | 0.0000 | 0.0050 [0.0000, 0.0100] |
| LTM | real | 0.0000 | 0.0015 [0.0000, 0.0030] |

Insertions are the smallest of the three operations everywhere, and again
elevated in LTM pseudowords with a complete Long/Short separation (minimum Long
0.105 > maximum Short 0.026).

**Insertion counts here are a lower bound.** The forced-length readout bounds
each prediction by the gold target length, so an insertion that would extend the
output beyond the target horizon cannot be produced or observed. Insertions in
this table are therefore only those that fit inside the horizon by displacing
other material.

---

## 9. Error composition (secondary, descriptive)

`clean/tables/clean_error_taxonomy_composition.tsv`. Proportions are computed
only where a cell has at least one operation; **no 0/0 proportion is
manufactured** — 27 of 48 seed × cell combinations carry
`NO_ERRORS_FOR_COMPOSITION` and 21 carry `OK`.

Mean of the per-seed proportions, cells with errors only:

| route | lexicality | length | subs | dels | ins |
|---|---|---|---:|---:|---:|
| LTM | pseudo | Long | 0.599 | 0.254 | 0.146 |
| LTM | pseudo | Short | 0.703 | 0.168 | 0.129 |
| LTM | real | Long | 0.869 | 0.066 | 0.066 |
| WM | pseudo | Long | 0.584 | 0.243 | 0.173 |
| FULL | pseudo | Long | 0.729 | 0.174 | 0.097 |

Composition is secondary and inherits the tie-breaking limit of §2 in full: the
*split* between operation types is backend-dependent even where the total is
not. The FULL and WM rows rest on 25 and 42 total operations respectively and
are `SPARSE_ERROR_LIMITED`.

---

## 10. Operation route contrasts

Seed-level differences in mean operations per item
(`clean/tables/clean_error_taxonomy_route_contrasts_summary.tsv`), clean
pseudowords, mean over seeds with range:

| contrast | subs | dels | ins | total |
|---|---|---|---|---|
| LTM − WM | +0.3286 [0.3069, 0.3402] | +0.1292 [0.1074, 0.1611] | +0.0742 [0.0435, 0.0895] | +0.5320 [0.4910, 0.5857] |
| LTM − FULL | +0.3325 [0.3120, 0.3453] | +0.1324 [0.1176, 0.1560] | +0.0780 [0.0563, 0.0895] | +0.5428 [0.5192, 0.5780] |
| FULL − WM | −0.0038 [−0.0051, 0.0000] | −0.0032 [−0.0102, 0.0051] | −0.0038 [−0.0128, 0.0026] | −0.0109 [−0.0281, 0.0077] |

Every LTM − WM and LTM − FULL contrast is positive in all four seeds, for all
three operations and for the total. The FULL − WM contrast changes sign across
seeds and is `SPARSE_ERROR_LIMITED`.

The same contrasts for trained real words are in the same table and are
`CEILING_LIMITED`: FULL and WM produce exactly zero operations on every trained
real item in every seed, so FULL − WM is structurally 0.000 and carries no
information, while LTM − WM and LTM − FULL reduce to the LTM value alone
(+0.0119 substitutions, +0.0007 deletions, +0.0007 insertions).

---

## 11. Hierarchical bootstrap intervals

`clean/tables/clean_error_taxonomy_bootstrap.tsv`. Policy unchanged since
Sprint 1: resample seeds with replacement, then items with replacement within
each analysis-set × stratum cell; **B = 10,000, random seed 20260730, 95 %
percentile interval**.

| quantity | lexicality | route | mean | 95 % interval |
|---|---|---|---:|---|
| mean substitutions / item | pseudo | LTM | 0.3413 | [0.2762, 0.4111] |
| mean deletions / item | pseudo | LTM | 0.1370 | [0.1049, 0.1720] |
| mean insertions / item | pseudo | LTM | 0.0807 | [0.0569, 0.1055] |
| mean edit distance / item | pseudo | LTM | 0.5589 | [0.4616, 0.6605] |
| mean edit distance / item | pseudo | WM | 0.0265 | [0.0045, 0.0556] |
| mean edit distance / item | pseudo | FULL | 0.0159 | [0.0026, 0.0339] |
| mean edit distance / item | real | LTM | 0.0134 | [0.0037, 0.0294] |
| mean edit distance / item | real | WM / FULL | 0.0000 | [0.0000, 0.0000] |

The trained-real FULL and WM intervals are degenerate at zero by ceiling, not by
precision.

---

## 12. Premature-EOS results

`eos/tables/premature_eos_by_seed_route.tsv`. **87 premature-EOS events in
total** across four seeds and three routes: **LTM 82, FULL 3, WM 2**.

| route | lexicality | seed 19 | seed 20 | seed 21 | seed 22 | total |
|---|---|---:|---:|---:|---:|---:|
| LTM | pseudo | 26 | 14 | 24 | 18 | **82** |
| FULL | pseudo | 2 | 0 | 1 | 0 | 3 |
| WM | pseudo | 1 | 1 | 0 | 0 | 2 |
| LTM / FULL / WM | real | 0 | 0 | 0 | 0 | **0** |

**Every observed premature-EOS event is on a pseudoword; trained real words show
zero events in every seed and every route.** Premature EOS is therefore strongly
concentrated in LTM pseudowords, which account for 82 of the 87 events (94 %).

Per-seed LTM pseudoword rates are 0.0665, 0.0358, 0.0614 and 0.0460 (mean
0.0524). FULL and WM rates are 0–0.0051 and are `SPARSE_EOS_LIMITED`.

**Scale.** Over the same four seeds the LTM route produced **874 edit operations
across 365 erroneous pseudoword items**, against **82 premature-EOS events**. An
observed premature EOS is present on about 22 % of erroneous LTM pseudoword
items. **Premature EOS therefore accounts for only a subset of erroneous
behaviour, and the size of this gap is exactly why early EOS cannot be treated
as a complete explanation of the error pattern.** No causal claim is made in
either direction.

---

## 13. EOS length profile

`eos/tables/premature_eos_by_exact_length.tsv`, pseudowords, mean over seeds:

| length | items | LTM rate | LTM events (mean/seed) | FULL rate | WM rate |
|---:|---:|---:|---:|---:|---:|
| 3 | 47 | 0.0053 | 0.25 | 0 | 0 |
| 4 | 78 | 0.0064 | 0.50 | 0 | 0 |
| 5 | 66 | 0.0038 | 0.25 | 0 | 0 |
| 7 | 66 | 0.0341 | 2.25 | 0 | 0 |
| 8 | 68 | 0.0699 | 4.75 | 0 | 0 |
| 9 | 66 | **0.1894** | 12.50 | 0.0114 | 0.0076 |

Length 6 is absent from the WFE by construction. The LTM premature-EOS rate is
roughly flat across the short lengths and rises steeply from length 7 to
length 9. In broad terms, 78 of the 82 LTM events fall in the Long group and 4
in the Short group.

**Linear-probability length slopes** (`eos/tables/premature_eos_length_slopes.tsv`),
`premature_eos_binary ~ phoneme_length`, descriptive and deliberately not
logistic where events are sparse or completely separated:

| route | lexicality | seed 19 | seed 20 | seed 21 | seed 22 | status |
|---|---|---:|---:|---:|---:|---|
| LTM | pseudo | +0.0356 | +0.0177 | +0.0316 | +0.0198 | `OK` (all four) |
| FULL | pseudo | +0.0034 | 0 | +0.0017 | 0 | `INSUFFICIENT_EVENTS` / `ALL_ZERO_OUTCOME` |
| WM | pseudo | +0.0017 | +0.0017 | 0 | 0 | `INSUFFICIENT_EVENTS` / `ALL_ZERO_OUTCOME` |
| all | real | 0 | 0 | 0 | 0 | `ALL_ZERO_OUTCOME` |

**The LTM premature-EOS length slope is positive in all four seeds**, with a
supported model status in every seed. FULL and WM slopes are not interpretable:
they rest on 0–2 events per seed.

---

## 14. EOS shortfall magnitude

`eos_shortfall = expected − observed`, positive meaning EOS came early; range
[1, L] by construction.

- **Primary, all-item mean shortfall** (zero assigned to items with no observed
  premature EOS — a summary convention only, which does **not** relabel their
  EOS class): LTM pseudowords 0.0742, 0.0384, 0.0639, 0.0486 by seed; FULL and
  WM ≤ 0.0051; trained real words exactly 0 everywhere.
- **Secondary, conditional mean shortfall**, denominator restricted to
  `PREMATURE_EOS` items only: LTM pseudowords 1.115, 1.071, 1.042, 1.056 by
  seed; FULL and WM exactly 1.000 where any event exists, and undefined
  (`NaN`) where none does.

Observed premature stops are therefore **almost always exactly one position
early**. The conditional estimates for FULL and WM rest on 1–2 events and are
`SPARSE_EOS_LIMITED`.

---

## 15. Exposure and morphology descriptives

**Exposure status** (`strata/tables/exposure_error_taxonomy.tsv`,
`ALL_WITH_EXPOSURE_STRATA`, all 1,200 items, no group excluded). Mean
substitutions per item under LTM, mean over seeds:

| exposure status | n | flag | LTM subs | LTM mean edit distance (seed 19) |
|---|---:|---|---:|---:|
| TRAINED_REAL_EXACT | 671 | OK | 0.0119 | 0.024 |
| TRAINED_REAL_PRON_VARIANT | 7 | VERY_SMALL_CELL, descriptive only | 0.214 | 0.000 |
| UNTRAINED_REAL | 122 | OK | 0.334 | 0.549 |
| NOVEL_PSEUDOWORD | 391 | OK | 0.341 | 0.601 |
| PSEUDO_TRAINING_WORD | 5 | VERY_SMALL_CELL, descriptive only | 0.000 | 0.000 |
| PSEUDO_TRAINING_HOMOPHONE | 4 | VERY_SMALL_CELL, descriptive only | 0.000 | 0.000 |

Descriptively, the 122 untrained real words pattern with novel pseudowords
rather than with trained real words under LTM. This restates the exposure
stratification already established in Sprint 1 and is **not** a new inference.
The three cells with n ≤ 7 are `DESCRIPTIVE_ONLY` and support no claim; the
`TRAINED_REAL_PRON_VARIANT` value of 0.214 rests on 7 items and a single
erroneous item in one seed.

**Morphology** (`strata/tables/morphology_error_taxonomy.tsv`, clean set), LTM,
mean over seeds with range:

| lexicality | morphology | subs | dels | ins |
|---|---|---|---|---|
| pseudo | complex | 0.3135 [0.2741, 0.3452] | 0.1421 [0.1117, 0.1624] | 0.0850 [0.0508, 0.0964] |
| pseudo | simple | 0.3698 [0.3196, 0.4124] | 0.1314 [0.0979, 0.1701] | 0.0760 [0.0619, 0.0876] |
| real | complex | 0.0164 [0.0066, 0.0230] | 0.0000 | 0.0000 |
| real | simple | 0.0082 [0.0027, 0.0191] | 0.0014 [0, 0.0027] | 0.0014 [0, 0.0027] |

The complex/simple ranges overlap in every operation, so **no morphology
difference is claimed here**; this table is a descriptive extension of the
Sprint-2 morphology analysis, not a new morphology inference.
`strata/tables/morphology_premature_eos.tsv` carries the matching EOS
descriptives.

---

## 16. EOS/deletion overlap

`eos/tables/premature_eos_deletion_overlap.tsv`, pooled over the four seeds,
clean pseudowords (n = 1,564 seed × item rows per route):

| route | EOS ∧ del | EOS ∧ no del | del ∧ no EOS | neither | P(del \| EOS) | P(EOS \| del) |
|---|---:|---:|---:|---:|---:|---:|
| FULL | 3 | 0 | 4 | 1,557 | 1.000 | 0.429 |
| WM | 2 | 0 | 8 | 1,554 | 1.000 | 0.200 |
| LTM | 82 | 0 | 107 | 1,375 | 1.000 | 0.434 |

**Neither probability is causal.** Two readings must be avoided:

- **`P(deletion | premature EOS) = 1.000` is a structural consequence of the
  readout, not an empirical discovery.** The prediction is trimmed at the first
  EOS, so an item with a premature EOS is strictly shorter than its target and
  any alignment to that target must contain at least one deletion. This
  identity does **not** license treating deletions as premature EOS or premature
  EOS as one deletion.
- `P(premature EOS | deletion) = 0.434` for LTM: **most deletion-bearing items —
  107 of 189 — carry no observed premature EOS at all.** Deletions and premature
  EOS are distinct measurements over largely non-identical item sets.

By broad length (`premature_eos_deletion_overlap_by_length.tsv`), the LTM
overlap is 78 events among 171 deletion-bearing Long items and 4 among 18 Short
items. The FULL and WM cells rest on 2–3 events and are `SPARSE_EOS_LIMITED`.

---

## 17. Four-seed consistency

All four seeds are primary and none is excluded; seed 21 is included
everywhere. Seed-level values are visible in every summary table via the
`seed_values` column.

Consistent in **all four seeds**:

- LTM pseudoword operation counts exceed FULL and WM for substitutions,
  deletions, insertions and the total (§10, all contrasts same-signed);
- Long > Short for all three LTM pseudoword operations, with non-overlapping
  seed ranges (§6–§8);
- substitutions > deletions > insertions in the LTM pseudoword cells;
- all observed premature-EOS events on pseudowords, none on trained real words;
- positive LTM premature-EOS length slope.

Not consistent across seeds:

- the FULL − WM operation contrast changes sign (seed 19 positive, seeds 20 and
  21 negative, seed 22 zero) — `SPARSE_ERROR_LIMITED`;
- FULL and WM premature-EOS counts (2, 0, 1, 0 and 1, 1, 0, 0) —
  `SPARSE_EOS_LIMITED`.

---

## 18. Exact-zero sensitivity

Seeds 19, 20 and 22 are the exact-ceiling seeds; every summary table carries an
`exact_zero_seeds_mean` column computed over that subset alone, kept **separate
from** the four-seed primary mean and never substituted for it.

The two agree closely wherever the quantity is estimable at all — for example
LTM − WM substitutions on pseudowords: four-seed mean +0.3286 versus exact-zero
mean +0.3248; LTM − WM total operations +0.5320 versus +0.5456. No conclusion in
this report changes under the exact-zero subset.

---

## 19. Seed-22 illustrations

`examples/`, declared in the frozen spec before any item was inspected. Seed 22
only, clean novel pseudowords, up to 20 rows per route, ordered deterministically
(raw edit distance descending, then `eos_shortfall` descending with missing
last, then `item_id` ascending).

`seed22_illustrative_pseudoword_errors.tsv` contains 22 rows — 20 for LTM and
only 1 each for FULL and WM, because those routes have no more erroneous clean
pseudowords in seed 22. `seed22_illustrative_premature_eos.tsv` contains 18
rows, all LTM, matching the seed-22 event count exactly.

These are **deterministic illustrations, not a representative sample**. They are
`DESCRIPTIVE_ONLY` and no claim in this report rests on them.

---

## 20. Ceiling, sparsity and forced-length limitations

- **Ceiling / floor.** Trained real words produce exactly zero operations under
  FULL and WM in every seed. Route comparisons on trained real words are
  therefore `CEILING_LIMITED`: a zero difference there is structural and is not
  evidence that the routes behave identically.
- **Sparsity.** FULL and WM error and EOS counts are small enough
  (15–42 operations, 2–3 EOS events over four seeds) that their internal
  structure — composition, contrasts, conditional shortfall, overlap
  probabilities — is `SPARSE_ERROR_LIMITED` or `SPARSE_EOS_LIMITED`.
- **Small cells.** Three exposure categories have n ≤ 7 and are
  `DESCRIPTIVE_ONLY`; none was excluded, and all are flagged in the tables.
- **Tie-breaking.** The insertion/deletion/substitution split depends on
  `editops` tie-breaking; the total edit distance does not.
- **Forced-length horizon.** Predictions are bounded by the gold target length.
  **Terminal insertions beyond the horizon are unobservable**, and **EOS timing
  at or after the correct boundary is wholly unobservable** — so `ON_TIME_EOS`
  and `LATE_EOS` cannot occur in these tables and `EOS_NOT_OBSERVED` remains
  ambiguous with respect to eventual stopping. Any analysis that would depend on
  those classes is `DESCRIPTIVE_ONLY` at best and is not attempted here.

---

## 21. Findings classified

**ROBUST**

- Operation-count reconstruction: substitutions + deletions + insertions equal
  the stored raw edit distance for all 14,400 canonical rows.
- The LTM route carries a substantially larger pseudoword operation burden than
  FULL or WM, in all four seeds, for all three operations and the total.
- Within LTM pseudowords, Long (7, 8, 9) exceeds Short (3, 4, 5) for
  substitutions, deletions and insertions, with non-overlapping four-seed
  ranges.
- Substitutions are the dominant LTM pseudoword operation, with deletions and
  insertions also elevated.
- Observed premature EOS is concentrated in LTM pseudowords: 82 of 87 events,
  and zero events on trained real words in any route or seed.
- The LTM premature-EOS length slope is positive in all four seeds with an `OK`
  model status in each.

**CONSISTENT_BUT_UNCERTAIN / SPARSE_EOS_LIMITED / SPARSE_ERROR_LIMITED**

- FULL versus WM comparisons of any kind — operations, EOS counts, composition.
- Conditional EOS shortfall estimates for FULL and WM (1–2 events each).
- Rare overlap configurations, in particular every FULL and WM cell of the
  2 × 2, and the LTM Short cell (4 events).
- The FULL − WM operation contrast, which changes sign across seeds.

**CEILING_LIMITED**

- All trained-real-word comparisons under FULL and WM, where the outcome is
  structurally zero.

**DESCRIPTIVE_ONLY**

- The three exposure categories with n ≤ 7.
- The seed-22 item illustrations.
- Anything that would rest on `ON_TIME_EOS` or `LATE_EOS`, which are
  structurally unobservable, or on reading `EOS_NOT_OBSERVED` as correct
  stopping, which it does not license.

**Explicitly not claimed anywhere in this report:** that premature EOS causes the
route length effect; that deletion errors are premature EOS; that missing EOS
means correct stopping; or any recommendation to change the LTM architecture.

---

## 22. Files generated

```
error_taxonomy/
  eos_convention.md                          audited EOS convention (Phase 1)
  error_taxonomy_analysis_spec.md            frozen specification (Phase 2)
  error_taxonomy_results.md                  this report
  length_effect_mechanism_handoff.md         factual handoff
  error_taxonomy_provenance.json             provenance record
  error_taxonomy_commit_plan.md              proposed commit, not executed
  _control/                                  spec/convention JSON twins, preflight, manifest
  faithful/figures/  faithful/tables/        Figure-8A replication (1 figure, 4 tables)
  clean/figures/     clean/tables/           clean taxonomy (2 figures, 8 tables)
  eos/figures/       eos/tables/             premature EOS (1 figure, 10 tables)
  strata/tables/                             exposure and morphology (5 tables)
  examples/                                  seed-22 illustrations (2 TSVs + README)
  validation/                                validation artefacts
```

Every figure is written as PNG (300 dpi), PDF and SVG with a standalone caption
file and the exact TSV that produced it. Regeneration is deterministic:

```
python -m scripts.behavioral_analysis.plot_error_taxonomy \
    --out_root reports/behavioral_wfe_fulllexicon_93a577f/error_taxonomy
```
