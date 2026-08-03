# Behavioral WFE analysis — full-lexicon cohort 93a577f

How the Word Feature Evaluation (WFE) behavioral results are produced, where
the code and outputs live, and how to regenerate the figures.

## Purpose

Answer, for the full-lexicon Lichtheim3 cohort, whether the length effect sits
in the ventral (LTM) route or the dorsal (WM) route, using the WFE stimulus set
from Dager et al. The analysis protocol was frozen **before any behavioral
result was inspected**; see
`outputs/behavioral_wfe_fulllexicon_93a577f/README_behavioral_analysis_design.md`
and its machine-readable twin `_control/behavioral_analysis_design.json`.

## Dataset regimes

| Set | n | Purpose |
|---|---|---|
| `FAITHFUL_WFE_ALL` | 1200 | Faithful stimulus-level replication with the article's original real/pseudo labels |
| `LICHTHEIM_CLEAN` | 1062 | Primary set: 671 trained real words + 391 novel pseudowords |
| `ALL_WITH_EXPOSURE_STRATA` | 1200 | Six training-exposure categories |
| `TRAINED_REAL_FREQUENCY_PRIMARY` | 671 | Frequency analysis on trained real words |
| `TRAINED_REAL_FREQUENCY_SENSITIVITY` | 678 | Adds the 7 pronunciation-variant words |

"Real words" in the primary figures means WFE words encountered during training
with the same phonological form; "Pseudowords" means WFE pseudowords whose
phonological form is absent from the training lexicon. The distinction matters
because the WFE was built against Dager's 50k lexicon, not our 29,571-word one.

## Checkpoints

Cohort `fulllexicon_cohort_93a577f`, training commit `93a577fd…`: seed 19
epoch 155, seed 20 epoch 130, seed 21 epoch 145, seed 22 epoch 140. All four
seeds are primary; **seed 21 is never excluded**, and seeds 19/20/22 form an
exact-ceiling sensitivity set only.

## Routes and metrics

Routes: `full` (gated), `wm` (dorsal only), `ltm` (ventral only), all decoded
under one convention — deterministic autoregressive, forced-length readout, no
noise, no teacher forcing.

Primary metric: raw Levenshtein edit distance over atomic ARPAbet tokens.
Secondary: word error (1 − exact match). Operation counts (insertions,
deletions, substitutions) come from `Levenshtein.editops`, the backend Dager
used; they feed the error taxonomy only.

## Faithful versus adapted

**Faithful** analyses reproduce Dager's method exactly: Figure 2A line styles
hard-coded (complex solid, simple dashed); Figure 2B via Ridge(α = 1.0), 80/20
split `random_state=42`, `permutation_importance(n_repeats=100,
random_state=42)`, no interactions and no p-values; Figure 2C via zip-mismatch
`Error_Indices` with **no Levenshtein alignment**.

**Adapted** analyses are Lichtheim3 extensions (clean-set restriction,
route contrasts, hierarchical bootstrap) and are always labelled as such. The
two are never mixed in one figure or table.

## Analysis package

`scripts/behavioral_analysis/` is inference-free — it never loads a checkpoint.

```
common.py                     frozen constants, colours, repo-relative paths
io.py                         loading, validation, deterministic TSV writing
bootstrap.py                  OLS slope + hierarchical bootstrap
compute.py                    every plotting table (statistics layer)
plotting.py                   the five figures (presentation layer)
build_canonical_table.py      production outputs -> canonical table
make_figures.py               regenerate all five figures
validate_outputs.py           structural validation
close_production_manifest.py  scientific/operational manifest split
```

### Regenerating the five figures

```bash
python -m scripts.behavioral_analysis.make_figures \
    --out_dir reports/behavioral_wfe_fulllexicon_93a577f/figures

python -m scripts.behavioral_analysis.validate_outputs \
    --figures reports/behavioral_wfe_fulllexicon_93a577f/figures
```

Each figure is emitted as PNG (300 dpi), PDF and SVG, beside the exact TSV that
produced it and a standalone caption. Regeneration is deterministic: the TSVs
are byte-identical across runs.

To rebuild the canonical table from the production predictions:

```bash
python -m scripts.behavioral_analysis.build_canonical_table
```

## Frozen choices that must not be edited

Analysis sets; seed policy; route definitions; metric definitions; the
hierarchical bootstrap (seeds resampled, then items within each analysis-set ×
stratum cell; B = 10,000; random seed 20260730; 95 % percentile interval); the
faithful zip-mismatch serial-position method. Red and blue encode lexicality
only and are never reused for another variable.

## Morphology analysis (Sprint 2)

`scripts/behavioral_analysis/morphology.py` and `plot_morphology.py` implement
the morphology × phoneme-length estimands, frozen in
`reports/behavioral_wfe_fulllexicon_93a577f/morphology/morphology_analysis_spec.md`
before any morphology result was inspected.

```bash
python -m scripts.behavioral_analysis.plot_morphology \
    --out_root reports/behavioral_wfe_fulllexicon_93a577f/morphology
```

Sign conventions: `morphology_contrast = mean(simple) − mean(complex)`
(positive ⇒ simple items worse); `morphology_length_interaction =
simple_slope − complex_slope` (positive ⇒ length effect stronger for simple).
Morphology is carried by **line style only** — complex solid, simple dashed —
because red and blue remain reserved for lexicality.

Small-cell flags are frozen: `VERY_SMALL_CELL` n < 10, `SMALL_CELL`
10 ≤ n < 20. No cell is ever excluded; flags are descriptive.

Results: `morphology/morphology_results.md`.

## Frequency analysis (Sprint 3)

`scripts/behavioral_analysis/frequency.py` and `plot_frequency.py` implement
the word-frequency estimands, frozen in
`reports/behavioral_wfe_fulllexicon_93a577f/frequency/frequency_analysis_spec.md`
before any frequency result was inspected.

```bash
python -m scripts.behavioral_analysis.plot_frequency \
    --out_root reports/behavioral_wfe_fulllexicon_93a577f/frequency
```

Frequency is defined **only for real words**; no pseudoword is ever assigned a
Zipf value or admitted to a Zipf model. Sign conventions: a **negative Zipf
slope** means higher-frequency words have fewer errors; a **positive low−high
contrast** means low-frequency words are harder. Route contrasts are reported
in both orientations (`raw_route_slope_difference` and
`frequency_benefit_route_difference`) so no sign is ambiguous.

Standardization is fixed once per dataset regime from the item-level values and
reused unchanged across every seed and route. Frequency figures use a
**neutral colour scale** — red and blue stay reserved for lexicality.

Under the frozen ceiling policy, a `seed × route` cell with zero errors and
zero edit distance is labelled `ALL_ZERO_OUTCOME`, no logistic fit is
attempted, and **no absence of frequency encoding may be claimed**.

Results: `frequency/frequency_results.md`.

## Error taxonomy and premature EOS (Sprint 4)

`scripts/behavioral_analysis/error_taxonomy.py` (Levenshtein operations),
`eos_diagnostics.py` (decoder EOS diagnostic) and `plot_error_taxonomy.py`
(figures, tables, examples). Estimands frozen in
`reports/behavioral_wfe_fulllexicon_93a577f/error_taxonomy/error_taxonomy_analysis_spec.md`;
the EOS indexing convention was audited from committed source **before** any EOS
distribution was read and frozen in `error_taxonomy/eos_convention.md`.

```bash
python -m scripts.behavioral_analysis.plot_error_taxonomy \
    --out_root reports/behavioral_wfe_fulllexicon_93a577f/error_taxonomy
```

**Two analyses, never conflated.** Operations are exactly substitution,
deletion and insertion, read from the counts produced by `Levenshtein.editops`
0.27.3 during the production evaluation; nothing re-aligns a sequence and **no
fourth operation exists**. Premature EOS is a separate decoder diagnostic. A
deletion is not automatically a premature EOS, a premature EOS is not one
deletion, several deletions may follow one early stop, and early stops may
coexist with substitutions or insertions.

**EOS observability.** `eos_position` is a 0-based index into the item's readout
window (`dec_input[i, 1:n_steps+1]`, exactly L tokens at indices 0…L−1); the
expected boundary is index L, which is outside the window. Therefore:

| Class | Meaning | Observable? |
|---|---|---|
| `PREMATURE_EOS` | EOS observed at index < L | yes — the only positively observable class |
| `ON_TIME_EOS` | EOS at index L | **no** — structurally unobservable |
| `LATE_EOS` | EOS beyond L | **no** — structurally unobservable |
| `EOS_NOT_OBSERVED` | **no EOS was observed within the instrumented evaluation horizon** | yes, but **ambiguous** with respect to eventual stopping |
| `EOS_UNAVAILABLE` | field absent or non-numeric | yes (none occur in this cohort) |

Zero observed `ON_TIME_EOS` does **not** mean the decoder never stops correctly,
and zero observed `LATE_EOS` does **not** mean late stopping never occurs.
`EOS_NOT_OBSERVED` is never read as correct stopping, on-time stopping,
successful completion, or the absence of an EOS-related problem. The frozen
class labels are kept unchanged; only their observability is clarified.

Presentation: operation type is encoded by **hatch**, never by red or blue,
which stay reserved for lexicality. The clean taxonomy figure keeps **one common
absolute y-scale across routes**; a `_full_wm_zoom` companion is emitted only
when the frozen rule *max mean LTM operation count > 10 × max mean FULL/WM*
evaluates true, and it never replaces the primary.

Results: `error_taxonomy/error_taxonomy_results.md`. Factual, non-causal handoff
for a future mechanistic study: `error_taxonomy/length_effect_mechanism_handoff.md`.

## Adapted feature importance (Sprint 5)

`scripts/behavioral_analysis/feature_importance.py` and
`plot_feature_importance.py` implement the adapted feature-importance estimands
(A15), frozen in
`reports/behavioral_wfe_fulllexicon_93a577f/feature_importance/feature_importance_analysis_spec.md`
before any model was fitted.

```bash
python -m scripts.behavioral_analysis.plot_feature_importance \
    --out_root reports/behavioral_wfe_fulllexicon_93a577f/feature_importance
```

**Adapted, not faithful.** The faithful Dager analysis (A11) is a separate
analysis on 1,200 source-labelled items, FULL route only, with no route factor.
It is never recomputed, replaced or pooled with A15, and the two are never
placed on one quantitative axis: `feature_importance/faithful_vs_adapted.md`.

**Two identifiability constraints.** On `LICHTHEIM_CLEAN` lexicality and
training exposure are **perfectly confounded** — every Real item is
`TRAINED_REAL_EXACT` and every Pseudo item is `NOVEL_PSEUDOWORD` — so the two
never enter the same model and the factor is reported as a **lexicality/exposure
contrast**; no claim separates them. **Zipf frequency is undefined for
pseudowords**, is never imputed as zero, and is therefore excluded from every
all-item clean model.

**Leakage rule.** The 80/20 split (`random_state = 42`) is drawn over items, so
all three route rows of an item stay in the same side, and the **identical item
split is reused across all four seeds and all three models** — a seed difference
is then a model difference, never a split difference. No row-level split is
permitted. The exact ids are in `_control/fi_train_items.tsv` and
`fi_test_items.tsv`.

**Permutation semantics.** Importance permutes **raw factors**, not model
columns: item-level factors are permuted across held-out items with the same
value applied to all three route rows, route labels are permuted **within** an
item preserving one FULL/WM/LTM row each, and encoding, standardization and
every declared interaction term are then rebuilt. Dummy and interaction columns
are never permuted independently. 100 repeats, `random_state = 42`,
Ridge `alpha = 1.0` (never tuned on the WFE).

**Sign policy.** Grouped importance is **unsigned** — no artificial single sign
for a multilevel factor. Coefficients live in separate tables, relative to the
frozen reference levels route = WM, lexicality = Pseudoword, morphology =
Complex, with route reported as two contrasts (LTM − WM and FULL − WM).

**Scores.** Held-out R² is primary and held-out MAE is the required sensitivity,
because the outcome is zero-heavy. A negative held-out R² is retained and
labelled `NEGATIVE_TEST_R2`, never suppressed; an all-zero outcome is not
fitted; a ceiling-limited route is labelled rather than given an artificial zero
importance. An unstable ranking is never read as evidence that a factor is
unrepresented.

**Uncertainty.** Within-seed permutation spread and between-seed spread are
reported separately, and the 100 repeats are never treated as independent model
seeds. The four-checkpoint interval is labelled a **"seed-resampling interval
over four checkpoints"**; the term *hierarchical bootstrap* stays reserved for
Sprints 1–4, where items are genuinely resampled.

Results: `feature_importance/feature_importance_results.md`.

## Where future analyses go

The deferred SSP analysis (A19) would be added as a further module inside
`scripts/behavioral_analysis/`, following the same pattern: estimands frozen in
a spec file first, statistics in a compute module, presentation in a plot
module. The sprint order and status of every planned
analysis is `docs/behavioral_wfe_analysis_matrix.md`.

## Provenance

Every published figure directory carries
`behavioral_analysis_provenance.json` with training commit, evaluation-code
commit, checkpoint and dataset hashes, package versions and the bootstrap
configuration. Production outputs are pinned by
`production_scientific_outputs_FINAL.sha256`; append-only operational logs are
pinned separately (see `manifest_closure_note.md`).

## Limitations

- Four seeds only; the bootstrap draws from 35 distinct seed multisets, so
  intervals are coarse and descriptive.
- Forced-length readout means terminal insertions are unobservable — this
  matches the original Dager implementation. The same horizon makes EOS timing
  at or after the correct boundary unobservable, so `ON_TIME_EOS` and
  `LATE_EOS` cannot appear in any table and `EOS_NOT_OBSERVED` is ambiguous.
- `Levenshtein.editops` tie-breaking can move counts between substitution,
  deletion and insertion without changing the total edit distance, so the
  operation split is backend-dependent while the total is not.
- FULL and WM are at ceiling on trained real words, so their zero slope there
  is a floor rather than a demonstrated absence of a length effect.
- The clean-set restriction was frozen in advance but removes the harder
  untrained real words from the primary figures; exposure strata are reported
  alongside for that reason.

## Deferred and out of scope

SSP / sonority (Figure 2D) remains deferred. Neural-representation analyses,
route ablations and any causal account of the length effect are a separate
project: they need new experiments, not a re-reading of these outputs.
