# Robust findings and limitations

Machine-readable twin: `tables/key_results_summary.tsv` (18 principal results).
Classifications are the frozen set; **no common effect-size scale is invented
across analyses**, because the outcomes are in different units (edit operations,
event counts, probabilities, R² drops).

## Distribution of classifications

| classification | n |
|---|---:|
| `ROBUST` | 7 |
| `CONSISTENT_BUT_SMALL` | 3 |
| `DESCRIPTIVE_ONLY` | 3 |
| `CEILING_LIMITED` | 2 |
| `NON_ESTIMABLE` | 1 |
| `SPARSE_EOS_LIMITED` | 1 |
| `OPTIONAL_DEFERRED` | 1 |

## ROBUST

1. **LTM pseudoword length slope** 0.197–0.256 edit operations per phoneme,
   positive in all four seeds (F2).
2. **LTM − WM pseudoword slope contrast** +0.246, +0.183, +0.184, +0.205 —
   positive in all four seeds (F2). *This is the answer to the original
   question.*
3. **Exposure ordering**: untrained real words (0.549) pattern with novel
   pseudowords (0.601), far from trained real words (0.024), under LTM (S12).
4. **Operation profile**: substitutions dominate (0.574/item in LTM long
   pseudowords), then deletions (0.245), then insertions (0.140); Long > Short
   with non-overlapping seed ranges (F4).
5. **LTM − WM total operations** +0.532 per item, positive in all four seeds
   (F4).
6. **Premature EOS is LTM- and pseudoword-specific**: 87 events, 82 in LTM,
   **zero on trained real words** in any route or seed (F5).
7. **Premature-EOS length slope in LTM** positive in all four seeds with `OK`
   model status in each (F5).

## CONSISTENT_BUT_SMALL

- LTM length slope on trained real words, 0.0007–0.0177 — positive in all four
  seeds but an order of magnitude below the pseudoword slope (F1).
- LTM Zipf slope −0.0130, CI [−0.0288, −0.0027], negative in all four seeds but
  resting on sparse errors (F6).
- Adapted feature importance: length rank 3 and morphology rank 4 in every seed;
  route and lexicality/exposure lead jointly but their **relative order is not
  resolved** — the gap of 0.0064 R² is smaller than the within-seed permutation
  SD of ≈ 0.015, they swap in seed 21, and MAE ranks lexicality/exposure first
  in every seed (F7).

## CEILING_LIMITED

- FULL and WM length slopes on trained real words are **exactly zero** —
  structural, from zero errors on every item in every seed. **No absence of
  length information may be inferred** (F1).
- Route-specific feature importance is estimable **only for LTM**; FULL and WM
  are `INSUFFICIENT_ERRORS` or `NON_ESTIMABLE` and are labelled in the figure,
  never given an artificial zero (S8).

## SPARSE_EOS_LIMITED

- Premature EOS covers only ~22 % of erroneous LTM pseudoword items (82 of 365),
  and 107 of 189 deletion-bearing items show no observed early stop.
  **Premature EOS is not a complete explanation of the error pattern** (F5).

## NON_ESTIMABLE

- Morphology: no contrast, no morphology × length interaction, no morphology ×
  route interaction distinguishable from zero; every interval spans zero and
  signs vary across seeds (S6, S7). **This concerns predictive contribution to
  these behavioural outcomes and is not a claim that morphology is absent from
  internal representations.**

## DESCRIPTIVE_ONLY

- Serial-position profiles (pooled across seeds by design; zip-mismatch
  positions with no alignment) (F3, S2).
- Lexical confidence and gate frequency slopes — **linked outcomes**, since the
  gate is a deterministic monotonic transform of confidence (S10).
- Faithful feature importance (A11): length leads in all four seeds, but on
  source labels that are not exposure, and never pooled with the adapted
  analysis (S3).

## OPTIONAL_DEFERRED

- **A19 SSP / sonority — not computed.** Deferred since the protocol freeze; a
  different stimulus set (2,859 CCV/VCC triphones) answering a different
  question.

## Cross-cutting limitations

| limitation | consequence |
|---|---|
| **Forced-length readout** | terminal insertions are unobservable and insertion counts are a lower bound; EOS timing at or after the boundary is wholly unobservable, so `ON_TIME_EOS` and `LATE_EOS` cannot appear and `EOS_NOT_OBSERVED` is ambiguous |
| **editops tie-breaking** | the substitution/deletion/insertion split is backend-dependent; the **total** edit distance is not |
| **Clean-set confounding** | lexicality and training exposure coincide exactly, so no analysis on that set separates them |
| **Zipf undefined for pseudowords** | frequency is absent from every all-item clean model; this is a scope limit, not a finding |
| **Ceiling on trained real words** | FULL and WM produce no errors there, so many contrasts are structurally zero |
| **Four seeds** | intervals are coarse and descriptive; the feature-importance interval is a *seed-resampling interval over four checkpoints*, not a hierarchical bootstrap |
| **Mechanism not localized** | the behavioural data constrain but do not identify encoder, representation, decoder, accumulation or gate |
