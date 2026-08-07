# Yair corrections pass — working report

**Status: working outputs for human validation. Nothing here is staged,
committed, pushed, or promoted into the final release.**

Not a new WFE analysis programme. Every number is a reaggregation of the frozen
canonical predictions (`canonical_behavioral_item_table.tsv`, SHA256
`8988aff6…`, 14,400 rows = 4 seeds × 1,200 items × 3 routes). No checkpoint was
loaded, no inference run, no model trained. A test parses both new modules and
asserts they cannot import torch or call any model loader.

Frozen specification: `_control/yair_corrections_spec.md` and `.json`, written
before any result in this pass was computed (a test compares mtimes).

---

## T1 — Word error rate by exact length

**Question.** At each exact phoneme length, what fraction of *whole words* does
each route get wrong, for trained real words versus novel pseudowords?

**Population.** `LICHTHEIM_CLEAN`: 671 `TRAINED_REAL_EXACT`, 391
`NOVEL_PSEUDOWORD`.

**Method.** Direct aggregation of `word_error` (1 − exact match) by seed × route
× lexicality × exact length. All four seeds plotted; across-seed mean; frozen
`cell_mean_bootstrap` 95 % band; item counts printed per bin; no smoothing; no
slope presented as the central result. Mean raw edit distance travels in the TSVs
as `mean_raw_edit_distance_severity_only`.

**Descriptive result.** Whole-word error rate, averaged over seeds:

| length | FULL real | FULL pseudo | WM real | WM pseudo | LTM real | LTM pseudo |
|---|---|---|---|---|---|---|
| 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.074 |
| 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.051 |
| 5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.004 | 0.121 |
| 7 | 0.000 | 0.000 | 0.000 | 0.000 | 0.006 | 0.224 |
| 8 | 0.000 | 0.000 | 0.000 | 0.007 | 0.003 | 0.379 |
| 9 | 0.000 | 0.057 | 0.000 | 0.072 | 0.027 | 0.534 |

**FULL and WM are at the floor.** Zero whole-word errors on real words at every
length; zero on pseudowords up to length 8; the first non-zero values appear only
at length 9. The entire length effect is an **LTM-on-pseudowords** phenomenon,
rising monotonically from 0.074 to 0.534.

This is the correction the metric change buys: reading the same data as mean edit
distance produces a shallow non-zero slope in all three routes and hides the fact
that two of them never fail a whole word.

**Limitation.** Inside `LICHTHEIM_CLEAN` lexicality and training exposure coincide
exactly, so this figure cannot separate them.

**Files.** `tables/word_error_by_length_{seed,summary,item_counts}.tsv`,
`tables/word_error_by_length_faithful_companion.tsv`,
`figures/yc1_word_error_by_length.{png,pdf,svg}` + caption.

**Estimability verdict.** ESTIMABLE for every cell; floor cells are reported as
zeros, not dropped.

---

## T2 — Faithful source-real error audit

**Question.** Among the 800 items WFE labels *real*, which fail and what do the
failures look like?

**Population.** All 800 `source_lexicality == "real"` items, 4 seeds × 3 routes =
9,600 rows.

**Method.** Exhaustive listing of every `word_error == 1` event with the literal
existing columns; EOS class derived with the frozen
`eos_diagnostics.classify_eos`. Descriptive summaries only.

**Descriptive result.**

| route | error events | unique items | event rate | unique-item rate |
|---|---|---|---|---|
| FULL | 8 | 5 | 0.0025 | 0.0063 |
| WM | 7 | 5 | 0.0022 | 0.0063 |
| LTM | 126 | 70 | 0.0394 | 0.0875 |

- **Errors are strongly concentrated in LTM** — 126 events against 8 and 7.
- **`UNTRAINED_REAL` share of errors: 100 % for FULL, 100 % for WM, 86.5 % for
  LTM** (109 events / 57 items, from a stratum of 122).
- **`TRAINED_REAL_PRON_VARIANT`**: 3 LTM events on 1 item out of 7; none in FULL
  or WM.
- **Do errors remain in `TRAINED_REAL_EXACT`? Yes, but only in LTM** — 14 events
  on 12 items out of 671, a 0.52 % event rate. FULL and WM make zero errors on
  trained-exact real words.
- **Recurrence.** In LTM, 53 % of erroneous items fail in exactly one seed and
  only 7 items (10 %) fail in all four. For FULL and WM no item fails in all
  four seeds.
- **Length and frequency**: `tables/faithful_real_error_descriptive_bins.tsv`.

**Limitation.** Descriptive. The pronunciation-variant stratum has 7 items and
the FULL/WM error sets 5 items each — enumeration only. The length and frequency
associations are **not** adjusted for the exposure confound: inside the faithful
real label, untrained words are both rarer and differently distributed over
length, so those apparent effects partly re-express exposure. No confirmatory
model was fitted, per the instruction.

**Files.** `tables/faithful_real_error_{events,summary,by_exposure,recurrence,descriptive_bins}.tsv`,
`figures/yc2_faithful_real_error_composition.{png,pdf,svg}` + caption.

**Estimability verdict.** ESTIMABLE as description; NOT attempted as inference.

---

## T3 — LTM successful-pseudoword audit

**Question.** Which novel pseudowords does LTM-only reproduce exactly, and how do
the groups differ on already-validated fields?

**Population.** 391 `NOVEL_PSEUDOWORD` items, LTM route only, 4 seeds.

**Method.** Per-item count of exact-match seeds → `ALWAYS_SUCCESSFUL` (4/4),
`MIXED_SUCCESS` (1–3/4), `ALWAYS_FAILED` (0/4). Exhaustive and mutually exclusive
by construction; a test verifies both properties and the arithmetic.

**Descriptive result.**

| group | n | share | mean length | mean confidence | mean gate (auxiliary) |
|---|---|---|---|---|---|
| ALWAYS_SUCCESSFUL (4/4) | 201 | 51.4 % | 5.10 | 0.562 | 0.432 |
| MIXED_SUCCESS (1–3/4) | 173 | 44.3 % | 7.01 | 0.540 | 0.421 |
| ALWAYS_FAILED (0/4) | 17 | 4.3 % | 8.53 | 0.507 | 0.405 |

LTM reproduces **a majority of novel pseudowords perfectly and consistently** —
it is not a route that simply fails on unfamiliar forms. Target length separates
the groups far more sharply than anything else available. In failed seeds the
always-failed group shows more of every error type (subs 1.88 vs 1.29,
deletions 0.76 vs 0.51, insertions 0.46 vs 0.31) and more premature EOS
(1.18 vs 0.36 per item).

**Confidence and gate are one variable, not two.** `gate = sigmoid(2.0 ×
(lexical_confidence − 0.7))`; a test verifies this identity numerically to
1e-6 on the frozen table. The gate is reported as auxiliary and never as
independent evidence, and it is not given its own panel.

**No lexicalization conclusion is drawn.**

**Missing validated measures.** Phonotacticity, distance to the training lexicon,
and suffix/phonemic complexity were searched for across `scripts/`, `reports/`,
`outputs/` and `docs/`. **None exists as a validated documented feature for WFE
items.** All three are recorded as `UNAVAILABLE_VALIDATED_MEASURE` with the
evidence in `tables/ltm_pseudoword_unavailable_measures.tsv`. **No proxy was
invented and no new feature computed.** A test asserts no substitute feature
appears in the comparison table.

**Limitation.** Length, confidence and success are mutually entangled; nothing
here identifies a cause. The always-failed group has 17 items.

**Files.** `tables/ltm_pseudoword_{item_success,group_summary,feature_summary,unavailable_measures}.tsv`,
`figures/yc3_ltm_pseudoword_success.{png,pdf,svg}` + caption.

**Estimability verdict.** ESTIMABLE as description; the three requested
discriminating features are `UNAVAILABLE_VALIDATED_MEASURE`.

---

## T4 — Faithful Figure 2C by route

**Question.** Does the faithful serial-position error profile differ across FULL,
WM and LTM?

**Method recovery — verified, not assumed.** The original producing driver
(`outputs/.../_control/run_faithful_and_confirmatory.py`) is **not in the current
tree**. Its logic was promoted verbatim into `compute.serial_position_tables` and
`compute.zip_mismatch_positions` (`analysis_code_migration.tsv`: PROMOTED,
"zip-mismatch and PCHIP unchanged"). Applying that frozen function to the
faithful subset at `route == "full"` reproduces the frozen
`faithful_figure2C_table.tsv`:

| quantity | n compared | max abs diff | rows only in frozen | rows only in recomputed |
|---|---|---|---|---|
| `error_rate_per_item` | 72 | 8.80e-17 | 0 | 0 |
| `n_items_x_seeds` | 72 | 0.00e+00 | 0 | 0 |
| `relative_position` | 72 | 5.55e-17 | 0 | 0 |

`reproduces_frozen_figure2C = True`. The by-route extension is written **only if
this gate passes** — the driver raises otherwise. No historical artefact was
regenerated: the frozen table was read, not rewritten.

**Verified method properties.** Relative position `(i−1)/(L−1)`; numerator =
item × seed rows mismatching at position i; denominator = item × seed rows in
that (lexicality, length) cell; predictions trimmed at EOS are **re-padded to
`<PAD>`**, recovering Dager's blanking so post-EOS positions count as mismatches;
seeds **pooled**, not averaged; PCHIP interpolation to 100 points, item-count
weighted across lengths; **no Levenshtein alignment**; lengths < 2 skipped (none
present).

**Descriptive result.** The rising serial-position profile is essentially an LTM
phenomenon. FULL and WM stay near the floor across the whole word for both
stimulus classes; LTM climbs, far more steeply for pseudowords (to ≈0.15 weighted
at the final position, with individual long-length curves reaching ≈0.38).

**Limitation.** Under zip alignment part of the climb is mechanical — once a
position is wrong, later positions are compared against a shifted target, so
error accumulates by construction. That is a property of the faithful estimand,
not a separate finding. Red/blue here are the **faithful stimulus labels**
(800/400), not exposure categories; 122 source-real items were never trained and
9 source pseudowords collide with the lexicon.

**Files.** `tables/faithful_figure2C_by_route.tsv`,
`tables/faithful_figure2C_by_route_interpolated.tsv`,
`tables/faithful_figure2C_reproduction_check.tsv`,
`figures/yc4_faithful_serial_position_by_route.{png,pdf,svg}` + caption.
The existing faithful Figure 2C files are untouched (verified by test).

**Estimability verdict.** ESTIMABLE for all three routes; no source limitation
forced a reduction to WM and LTM only.

---

## T5 — Feature importance by route

**Question.** Can route-specific feature importance be estimated, and for which
routes?

**Method. Audit, not refit.** The adapted (A15) route-specific analysis already
exists and is validated. Its protocol was recorded and its outputs read: Ridge
α = 1.0; item-grouped 80/20 split `random_state=42` (all three route rows of an
item stay together, and the identical split is reused across seeds and models);
`permutation_importance` `n_repeats=100`, `random_state=42`; permutation acts on
**raw factors**, rewriting every derived column, never on dummies or interaction
terms independently; outcome `raw_edit_distance`; one model per route. The
faithful (A11) FI is read-only and was not pooled with it. **Nothing was
refitted.**

**Outcome density on `LICHTHEIM_CLEAN`** (the quantity that governs
estimability):

| route | rows | non-zero | density | variance |
|---|---|---|---|---|
| FULL | 4,248 | 15 | 0.0035 | 0.0106 |
| WM | 4,248 | 21 | 0.0049 | 0.0253 |
| LTM | 4,248 | 379 | 0.0892 | 0.6147 |

**Estimability verdict per route:**

| route | seeds | verdict | original status | importance reported |
|---|---|---|---|---|
| **LTM** | 19, 20, 21, 22 | `ESTIMABLE` | `OK` (4/4) | **yes** — read from the validated A15 run |
| FULL | 19, 21 | `NOT_ESTIMABLE_CEILING_OR_SPARSE_OUTCOME` | `INSUFFICIENT_ERRORS` (2 non-zero test rows of 212) | no |
| FULL | 20, 22 | `NOT_ESTIMABLE_CEILING_OR_SPARSE_OUTCOME` | `NON_ESTIMABLE` (0 non-zero test rows) | no |
| WM | 19, 20, 21 | `NOT_ESTIMABLE_CEILING_OR_SPARSE_OUTCOME` | `INSUFFICIENT_ERRORS` (1 non-zero test row) | no |
| WM | 22 | `NOT_ESTIMABLE_CEILING_OR_SPARSE_OUTCOME` | `NON_ESTIMABLE` (0 non-zero test rows) | no |

**A non-estimable cell is never assigned zero importance** — the column
`importance_is_zero_by_fiat` is `False` on every row and a test enforces it.
**Negative test R² is preserved, not clipped**: WM seed 21 keeps
`test_r2 = −0.0257`.

**No new figure was produced for T5.** LTM is the only estimable route, its
importances already exist in the validated A15 release, and a one-route figure
would duplicate a published one.

**Files.** `tables/fi_route_estimability.tsv`, `tables/fi_outcome_density.tsv`.

---

## Non-regression

331 files across six validated output inventories were hash-checked:
**329 verified byte-identical, 0 missing, 2 mismatched — neither caused by this
pass**:

1. `outputs/.../behavioral_analysis/_control/analysis_execution_log.txt` —
   mtime 2026-07-31 17:47, months before this pass. Pre-existing: the inventory
   records a hash of an execution log that was still being appended to when the
   inventory was written, so the log cannot contain its own final hash.
2. `reports/.../final_release/.DS_Store` — a gitignored macOS Finder artifact
   (`.gitignore:16`), not an analysis output; it should not have been
   inventoried in the first place.

**Zero closed-release scientific files changed.** All writes went to
`reports/behavioral_wfe_fulllexicon_93a577f/yair_corrections/`. Details:
`validation/non_regression.json`; hashes of the 36 new files:
`validation/yair_corrections_output_inventory.tsv`.

## Tests

- `tests/test_behavioral_yair_corrections.py` — **38 passed**
- full suite (`tests/`, excluding `smoke_test.py`) — **646 passed**

---

---

# Revision pass (second round)

Figure polish, two audits and a diagram specification. **No analysis was rerun
for a different result**: all T1-T5 backing TSVs were re-derived and are
byte-identical to the first pass, so every plotted value is unchanged. No
training, no inference, no long-pseudoword generation, no architecture change.

## Figure revisions

| figure | change |
|---|---|
| `yc1_word_error_by_length` | item counts printed **once**, under the leftmost panel only, with `real`/`pseudo` row labels; explicit legend for across-seed mean, 95 % bootstrap band and individual seeds; exact lengths 3,4,5,7,8,9 preserved; **scientific values unchanged** |
| `yc2_faithful_real_error_composition` | exposure counts annotated directly on every segment (FULL: 8 events / 5 items all `UNTRAINED_REAL`; WM: 7 / 5 all `UNTRAINED_REAL`; LTM: 109/57 `UNTRAINED_REAL`, 3/1 `TRAINED_REAL_PRON_VARIANT`, 14/12 `TRAINED_REAL_EXACT`); legend now spells out the events-versus-unique-items distinction |
| `yc3_ltm_pseudoword_success` | caption revised: states explicitly that it **does not identify why** LTM succeeds, that length and confidence are **associated descriptors** of outcome-defined groups, and that the top-1 semantic-bank neighbour is **never injected into the decoder**; no new feature, no lexicalization claim |
| `yc4_faithful_serial_position_wm_ltm` | **new** simplified dorsal-versus-ventral presentation; per-length curves retained but visually subordinated and labelled; caption states seeds pooled, zip mismatch, no Levenshtein alignment, post-EOS re-padding |
| `yc4s_faithful_serial_position_all_routes` | the previous three-route figure, retained as **supplementary** (same verified estimator) |

New table: **`tables/trained_real_exact_ltm_errors.tsv`** and its Markdown twin
`.md` — the 12 `TRAINED_REAL_EXACT` words LTM gets wrong, with word, Zipf,
length, target, per-seed predictions, failing seeds and per-seed
substitution/deletion/insertion counts. **No frequency model is fitted**; Zipf is
a descriptive column and a test asserts no regression estimator is called.

## Stable-zero audit

`stable_zero_audit/` — derived from the frozen cohort table
`all_checkpoints.tsv`. No training, no inference, no checkpoint loaded.

Evaluated grid: **epochs 105-200, step 5, 20 evaluations per seed, complete and
regular for all four seeds.** No missing evaluation is inferred anywhere.

Criterion: the selected checkpoint is the **first** checkpoint of a streak of X
consecutive evaluated zero-error checkpoints; training can stop only once the
**Xth** zero is observed. `stop_epoch_earliest_knowable` records that second
quantity separately.

All zero-error streaks (train AR errors, FULL route, 29,571-word lexicon):

| seed | zero streaks (first → last, length) | longest |
|---|---|---|
| 19 | 140→140 (1); **155→180 (6)** | 6 |
| 20 | **130→135 (2)**; 195→195 (1) | 2 |
| 21 | **none** | 0 |
| 22 | **140→200 (13)** | 13 |

| X | seed 19 | seed 20 | seed 21 | seed 22 |
|---|---|---|---|---|
| **2** | ✅ select 155, stop 160 | ✅ select 130, stop 135 | ❌ | ✅ select 140, stop 145 |
| **3** | ✅ select 155, stop 165 | ❌ | ❌ | ✅ select 140, stop 150 |
| **5** | ✅ select 155, stop 175 | ❌ | ❌ | ✅ select 140, stop 160 |

The X=2 audit **reproduces the cohort's own selection exactly** for the three
seeds it applied to (19→155, 20→130, 22→140); seed 21 was selected by the
fallback rule `earliest_checkpoint_with_minimum_error_count` at 1 error.

## Architecture / premotor audit

`architecture_audit/architecture_audit.md` + `.json`, regenerable with
`python scripts/audit_architecture.py`. Evidence is AST inspection of
`models/*.py` cross-checked against the frozen checkpoint's own `cfg_*` dicts and
`model_state_dict` shapes. The checkpoint was opened only to read configuration
and shapes — no model constructed, no forward pass, no token generated.

## Updated diagram specification

`architecture_audit/architecture_diagram_spec.md`. Replaces the stale
biGRU/masked-mean depiction with the executable `unigru_last_hidden` path, and
lists nodes, edges, mandatory annotations, visual grammar, a provenance block and
a nine-point acceptance checklist. **No diagram is rendered or committed.**

---

## Proposed figure selection for Yair

**Not applied — the final release selection is unchanged.** This is a proposal
only.

| priority | figure | why |
|---|---|---|
| **1** | `yc1_word_error_by_length` | Answers Yair's actual question with the right metric. Makes the ceiling on FULL and WM visible instead of hiding it inside a slope, and shows the length effect is confined to LTM-on-pseudowords. Strongest single replacement for the current length figure. |
| **2** | `yc2_faithful_real_error_composition` | Settles "are the real-word errors real?" — they are almost entirely untrained words, and FULL/WM make none at all on trained-exact words. Directly addresses the faithful-versus-clean confusion. |
| **3** | `yc4_faithful_serial_position_wm_ltm` | The requested by-route split of Figure 2C, simplified to the dorsal/ventral contrast and method-verified against the frozen original. Three-route version available as `yc4s_…` supplementary. |
| 4 | `yc3_ltm_pseudoword_success` | Useful nuance — LTM reproduces 51 % of novel pseudowords perfectly in all four seeds — but the group comparison is limited by the three unavailable validated features. |

Suggested pairing: **yc1 + yc2** carry the correction. yc4 is the direct answer
to the Figure 2C request. yc3 is supporting material.

## Human decision required

1. Is `reports/behavioral_wfe_fulllexicon_93a577f/yair_corrections/` the right
   home, or should these move under `final_release/` once approved?
2. Should any of yc1–yc4 replace or supplement a current release figure? (Not
   done here.)
3. T2's length and frequency associations are descriptive and confounded with
   exposure. Do you want a confirmatory model in a later pass, and if so with
   which population and adjustment set?
4. The three unavailable measures (phonotacticity, lexicon distance,
   suffix/phonemic complexity) would each require a **new validated feature
   definition**. Do you want one specified?
5. `analysis_execution_log.txt` has a stale inventory hash from the original
   sprint. Leave it, or re-issue that inventory row with a documented reason?
