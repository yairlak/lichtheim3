# Sprint 4 — error taxonomy and premature EOS: frozen specification

**Frozen 2026-08-03, before any taxonomy or EOS summary was computed or
inspected.** Machine-readable twin: `_control/error_taxonomy_analysis_spec.json`.
The EOS convention was audited first and is frozen separately in
`eos_convention.md`.

Scope: error taxonomy and premature-EOS description only. No feature
importance, SSP, new morphology or frequency inference, ablation or causal
explanation.

## Two separate analyses, never conflated

Levenshtein operations are exactly **substitution, deletion, insertion**, read
from the stored counts produced by `Levenshtein.editops` 0.27.3. **No fourth
operation is introduced.** Premature EOS is a *decoder diagnostic* derived from
the raw EOS-position instrumentation and lives in separate tables, figures and
claims. A deletion is not automatically a premature EOS; a premature EOS is not
one deletion; several deletions may follow one early stop; early stops may
coexist with substitutions or insertions.

Two measurement limits apply throughout: `editops` tie-breaking can move counts
between insertion/deletion/substitution **without changing total edit
distance**, and the forced-length readout makes **terminal insertions beyond
the target horizon unobservable**.

## Estimands

### 5.1 Primary operation outcome

Mean operations per **evaluated** item (never conditioned on being erroneous),
per `seed × dataset_regime × route × lexicality/exposure × short/long ×
morphology where relevant`. Also recorded: `n_items`, `n_erroneous_items`,
total operation count, mean per item, and mean per **erroneous** item.

### 5.2 Error composition (secondary, descriptive)

`operation_proportion = operation_count / total_edit_operations`, only where
total operations > 0. Cells with zero operations are marked
`NO_ERRORS_FOR_COMPOSITION`; **no 0/0 proportion is manufactured**.

### 5.3 Faithful Figure-8A analysis

`FAITHFUL_WFE_ALL`, FULL route, the original 12 WFE conditions in their source
ordering, all four seeds. Mean substitutions / deletions / insertions per item.
A faithful stimulus-and-metric replication adapted to four Lichtheim3
checkpoints; **source Real/Pseudo labels are not training exposure**. Figures
8B and 8C are out of scope (they concern ablated SWP models).

### 5.4 Clean route analysis

`LICHTHEIM_CLEAN`. Factors: route; lexicality (trained Real words = 671, novel
Pseudowords = 391); broad length **Short = 3,4,5** and **Long = 7,8,9**;
operation type. Morphology and exact length stay in detailed tables.

### 5.5 Route contrasts

Seed-level per-operation differences **LTM − WM**, **FULL − WM**, **LTM − FULL**
for clean pseudowords, plus the same contrasts in total edit operations. The
identical tables are produced for trained real words with ceiling-limited
comparisons marked explicitly.

### 5.6 Premature-EOS outcomes

1. `premature_eos_rate` — fraction of evaluated items classified
   `PREMATURE_EOS`;
2. `mean_eos_shortfall_per_item` — shortfall averaged over **all** evaluated
   items, zero for non-premature items — **primary**;
3. `conditional_mean_eos_shortfall` — mean among premature items only —
   secondary.

By seed, route, clean lexicality, exact phoneme length, broad short/long group,
and exposure status.

### 5.7 EOS length slope

Per `seed × route × clean lexicality`, a transparent **linear probability**
slope `premature_eos_binary ~ phoneme_length`. Statuses: `OK`,
`ALL_ZERO_OUTCOME`, `ALL_ONE_OUTCOME`, `INSUFFICIENT_EVENTS`, `NON_ESTIMABLE`.
Logistic regression is **not** forced under separation or sparse events. This
is descriptive, not causal.

### 5.8 EOS/deletion overlap

A 2 × 2 per declared stratum: premature EOS with ≥ 1 deletion; premature EOS
without deletion; deletion without premature EOS; neither. Report
`P(deletion | premature EOS)` and `P(premature EOS | deletion)` with counts and
denominators. **Neither probability is causal.**

### 5.9 Statistical policy

All four seeds primary with seed-level values visible; hierarchical bootstrap
unchanged since Sprint 1 (seeds then items, **B = 10,000, random seed
20260730, 95 % percentile**); exact-zero sensitivity on seeds 19, 20, 22. Each
quantity reports every seed, the mean, the range and the interval. No p-value
required. Sparse outcomes are labelled `SPARSE_ERROR_LIMITED`,
`SPARSE_EOS_LIMITED`, `CEILING_LIMITED`, `DESCRIPTIVE_ONLY` or `NON_ESTIMABLE`.

### 5.10 Small-cell policy

**No exposure group is excluded.** Flags: `SMALL_CELL` n < 20,
`VERY_SMALL_CELL` n < 10. Statuses with n ≤ 7 are descriptive only.

## Presentation

Red and blue remain reserved for **lexicality** (Real = red, Pseudoword =
blue). **Operation type is never encoded by red or blue** — it uses hatch,
marker shape or neutral shading. All four seed values are visible; the mean is
prominent; intervals are shown where estimable.

The clean taxonomy figure uses **one common absolute y-scale across routes** so
the route magnitude difference is not hidden. Frozen zoom rule, fixed now: *if
the maximum mean LTM operation count exceeds **10 ×** the maximum mean FULL/WM
operation count, an additional explicitly labelled `…_full_wm_zoom` figure is
produced containing FULL and WM only.* The zoom never replaces the
absolute-scale primary figure.

## Seed-22 illustrations

Seed 22 only, declared before analysis. Up to 20 erroneous clean pseudowords
per route, ordered deterministically by raw edit distance descending, then
`eos_shortfall` descending with missing last, then `item_id` ascending. These
are **deterministic illustrations, not a representative sample**, and carry no
inference.

## Result categories

`ROBUST`, `CONSISTENT_BUT_SMALL`, `INCONSISTENT_ACROSS_SEEDS`,
`CEILING_LIMITED`, `SPARSE_ERROR_LIMITED`, `SPARSE_EOS_LIMITED`,
`DESCRIPTIVE_ONLY`, `NON_ESTIMABLE`.

No causal claim, and in particular **no statement that premature EOS causes the
route length effect**. No architectural recommendation.
