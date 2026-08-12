# Why can LTM repeat some pseudowords? — frozen analysis specification

**Written before the length-controlled comparison was computed.**

Refined question: *what empirical evidence explains why some genuinely unseen
pseudowords can still be reconstructed exactly by the LTM encoder–decoder?*

---

## Classification of existing work

Every candidate analysis was located and classified before anything new was run.
**No category-A analysis is recomputed.**

| analysis | location | class | use here |
|---|---|---|---|
| AR vs gold-prefix, per seed × item | `m2_gold_prefix/word_level_ar_vs_gold.tsv` | **A** | read directly |
| AR vs gold-prefix length slopes | `m2_gold_prefix/length_slopes_ar_vs_gold.tsv` | **A** | cited, not recomputed |
| first-error position / type | `m1_origin_propagation/first_error_events.tsv` | **A** | read directly |
| first-error hazard | `m1_origin_propagation/first_error_hazard.tsv` | **A** | cited |
| post-divergence burden | `m1_origin_propagation/post_divergence_burden.tsv` | **A** | cited |
| lexical-neighbour attraction (incl. `confidence`, `correct`, `d_pred_top1`, `pred_is_training_form`) | `m3_lexical_attraction/lexical_attraction_items.tsv` | **A** | read directly |
| bank-structure audit | `m3_lexical_attraction/bank_structure_audit.*` | **A** | cited |
| encoder / `s_hat` / h0 / premotor ordered probes | `m4_representation/ordered_probe_*` | **A** | cited |
| unordered phoneme-content probe | `m4_representation/unordered_probe_summary.tsv` | **A** | cited |
| success groups over 391 novel pseudowords | `yair_corrections/tables/ltm_pseudoword_item_success.tsv` | **A** | read directly |
| **success rate within each exact length** | derivable from the above | **B** | minimal reaggregation |
| **confidence → success at fixed length** | derivable from `m3` | **B** | minimal reaggregation |
| **AR vs gold-prefix within the pseudoword population, by length** | derivable from `m2` | **B** | minimal reaggregation |
| new phonotactic / lexical-distance / suffix features | — | **D** | **not created** |
| per-position target rank/margin re-analysis | `instrumented/timestep_metrics.tsv` | **D** | not needed; M4 already answers the representational question |

Nothing is classified **C**. Everything needed is already computed; what was
missing is the **length control**, which is a reaggregation, not a new
experiment. That is the whole of the new work here.

---

## Pre-registered comparison

**Question.** At fixed phoneme length, does lexical confidence (top-1 cosine
between the item's `s_hat` and the frozen GloVe bank) predict whether the LTM
route reproduces a novel pseudoword exactly?

**Population.** The 391 `NOVEL_PSEUDOWORD` items, LTM route, 4 seeds
(1,564 seed × item observations). Seeds are never treated as independent items:
every within-length statistic is computed per seed and then averaged over seeds,
and the item-level grouping is preserved.

**Success definition.** Primary: per seed × item `exact_match` (1 = the emitted
phoneme sequence equals the target exactly). Secondary: the frozen 4/4, 1–3/4,
0/4 item grouping.

**Length control.** Exact-length stratification — lengths 3, 4, 5, 7, 8, 9 are
analysed separately and never pooled without weights. Within each stratum,
confidence is split at the **within-stratum median** (so the split cannot import
the length→confidence relationship). The pooled effect is the n-weighted mean of
the within-stratum differences, i.e. a stratified (Mantel–Haenszel-style)
contrast, not a marginal one.

**Metric.** Within-stratum success-rate difference, high-confidence half minus
low-confidence half. Reported per length, per seed, and pooled.

**Expected discriminating result.** If lexical proximity contributes beyond
length, the within-stratum difference is positive and consistent across lengths
and seeds. If it is near zero once length is held fixed, the marginal
confidence–success association reported earlier is a length artefact.

**Interpretation if positive.** Some contribution of proximity to trained
semantic space. It would still **not** be retrieval: M3 already established no
measurable attraction toward the tested `s_hat` neighbours, and no bank vector
is ever passed to the decoder.

**Interpretation if null.** Success is not explained by lexical proximity;
consistent with a length-limited phonological autoencoder account.

**What cannot be concluded either way.** Causality — the groups are defined by
the outcome. Confidence and gate are deterministically linked
(`gate = σ(2·(confidence − 0.7))`) and are one variable, never two pieces of
evidence.

## Second pre-registered contrast

**Question.** Is LTM's whole-word failure on novel pseudowords decided at
encoding, or produced by autoregressive error propagation?

**Method.** Compare `ar_word_error` with `gp_word_error` (gold-prefix tokenwise
argmax) per seed × item from `m2_gold_prefix/word_level_ar_vs_gold.tsv`, within
each exact length. Also compare `ar_edit_distance` with `gp_edit_distance`.

**Interpretation if the two word-error columns agree item-by-item.** Whole-word
success is determined before any feedback: AR propagation changes *how wrong* an
error is, not *whether* the word is wrong.

**Interpretation if they differ substantially.** A material share of whole-word
failures is manufactured by feedback.

## Outputs

At most one new compact figure, and only if it adds information beyond the
existing yc3 figure. Tables under
`yair_corrections/ltm_pseudoword_success/tables/`.
