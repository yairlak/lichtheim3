# Behavioral WFE report — full-lexicon cohort 93a577f

Publication-ready figures, their exact plotting tables and captions, plus the
provenance needed to reproduce them. Generated in Sprint 1 of the finalization
phase; the analysis protocol was frozen before any result was inspected.

Method and usage: `docs/behavioral_wfe_fulllexicon.md`.
Planned/completed analyses: `docs/behavioral_wfe_analysis_matrix.md`.

## Figures

Each figure ships as PNG (300 dpi), PDF and SVG, with the exact TSV that
produced it and a standalone caption. Regeneration is deterministic: all five
formats are byte-identical across runs.

| Figure | Plotting table | Caption |
|---|---|---|
| `figures/yair_clean_length_by_route` | `yair_clean_length_by_route.tsv` | `yair_clean_length_by_route_caption.md` |
| `figures/yair_clean_length_slopes` | `clean_length_slopes_by_seed.tsv`, `clean_route_length_contrasts.tsv`, `clean_bootstrap_results.tsv` | `yair_clean_length_slopes_caption.md` |
| `figures/yair_clean_serial_position` | `yair_clean_serial_position_interpolated.tsv`, `yair_clean_serial_position.tsv` | `yair_clean_serial_position_caption.md` |
| `figures/gate_by_clean_lexicality` | `gate_by_clean_lexicality.tsv` | `gate_by_clean_lexicality_caption.md` |
| `figures/gate_by_exposure_status` | `gate_by_exposure_status.tsv` | `gate_by_exposure_status_caption.md` |

`figures/figure_manifest.json` records every path plus the canonical-table hash.

## Sprint 2 — morphology × phoneme length

`morphology/` holds the Sprint-2 analysis: a script-faithful replication on all
1,200 original WFE items (FULL route) and an adapted clean-set analysis across
FULL/WM/LTM. The specification was frozen in
`morphology/morphology_analysis_spec.md` before any result was inspected.

| Figure | Plotting table | Caption |
|---|---|---|
| `morphology/faithful_replication/figures/faithful_length_lexicality_morphology` | `faithful_length_lexicality_morphology_plot.tsv` | `..._caption.md` |
| `morphology/clean_adapted/figures/clean_length_morphology_by_route` | `clean_length_morphology_by_route_plot.tsv` | `..._caption.md` |

Contrast, interaction, route-contrast, bootstrap, exact-zero-sensitivity and
cell-count tables sit beside each figure. Word-error counterparts are tables
only — no word-error figure was produced (`FIGURE_NOT_CREATED_DUE_TO_CEILING`).
Results and finding categories: `morphology/morphology_results.md`.

## Sprint 3 — word frequency

`frequency/` holds the Sprint-3 analysis on trained real words (671), with
pronunciation-variant (678), exact-zero-seed and untrained-real (122)
sensitivity analyses kept separate. The specification was frozen in
`frequency/frequency_analysis_spec.md` before any result was inspected.

| Figure | Plotting tables | Caption |
|---|---|---|
| `frequency/primary/figures/trained_real_frequency_by_route` | `trained_real_frequency_slopes.tsv`, `..._bootstrap.tsv`, `..._route_contrasts.tsv`, `trained_real_high_low_descriptives.tsv`, `trained_real_zipf_length_models.tsv` | `..._caption.md` |
| `frequency/gate_confidence/figures/frequency_confidence_gate` | `frequency_confidence_slopes.tsv`, `frequency_gate_slopes.tsv`, `..._bootstrap.tsv` | `..._caption.md` |

Distribution and confound tables are in `frequency/tables/`. Word-error
counterparts are tables only — no word-error figure was produced
(`FIGURE_NOT_CREATED_DUE_TO_CEILING_OR_SPARSE_ERRORS`). Results and finding
categories: `frequency/frequency_results.md`.

## Sprint 4 — error taxonomy and premature EOS

`error_taxonomy/` holds two analyses that are deliberately **never merged**: the
Levenshtein taxonomy (substitutions, deletions, insertions) and the
premature-EOS decoder diagnostic. A deletion is not a premature EOS, and no
fourth edit operation exists. The EOS indexing convention was audited from
committed source first (`error_taxonomy/eos_convention.md`), and the analysis
specification was frozen before any summary was computed
(`error_taxonomy/error_taxonomy_analysis_spec.md`).

| Figure | Plotting tables | Caption |
|---|---|---|
| `error_taxonomy/faithful/figures/faithful_figure8a_error_types` | `faithful_condition_error_types.tsv`, `faithful_condition_summary.tsv`, `faithful_condition_composition.tsv`, `faithful_condition_cell_counts.tsv` | `..._caption.md` |
| `error_taxonomy/clean/figures/clean_error_taxonomy_by_route` | `clean_error_taxonomy_cells.tsv`, `..._summary.tsv`, `..._by_exact_length.tsv`, `..._composition.tsv`, `..._route_contrasts.tsv`, `..._route_contrasts_summary.tsv`, `..._bootstrap.tsv`, `..._zoom_rule.tsv` | `..._caption.md` |
| `error_taxonomy/clean/figures/clean_error_taxonomy_full_wm_zoom` | same tables as above | `..._caption.md` |
| `error_taxonomy/eos/figures/premature_eos_by_route` | ten `premature_eos_*.tsv` tables | `..._caption.md` |

The clean taxonomy figure uses **one common absolute y-scale across routes**.
The `_full_wm_zoom` companion exists because the frozen >10× rule evaluated
true (observed ratio 22.95, `clean_error_taxonomy_zoom_rule.tsv`); it **does not
replace** the common-scale primary.

**EOS observability.** The readout window holds exactly L tokens at indices
0…L−1, so an EOS at the correct boundary (index L) is outside it. Only
`PREMATURE_EOS` is positively observable; `ON_TIME_EOS` and `LATE_EOS` are
structurally unobservable; `EOS_NOT_OBSERVED` means **no EOS was observed within
the instrumented evaluation horizon** and is ambiguous with respect to eventual
stopping — it is never read as correct stopping.

Exposure and morphology descriptives are in `error_taxonomy/strata/tables/`;
deterministic seed-22 illustrations (no inference) in `error_taxonomy/examples/`.
Results and finding categories: `error_taxonomy/error_taxonomy_results.md`.
Factual handoff for a future mechanistic study, with no causal claim:
`error_taxonomy/length_effect_mechanism_handoff.md`.

## Sprint 5 — adapted feature importance

`feature_importance/` holds the Sprint-5 adapted analysis (A15): a clean-set
joint main-effects model across all three routes (primary), a predeclared
two-way interaction model, and route-specific models. The specification was
frozen in `feature_importance/feature_importance_analysis_spec.md` before any
model was fitted.

| Figure | Plotting tables | Caption |
|---|---|---|
| `feature_importance/clean_joint/figures/clean_adapted_factor_importance` | `clean_main_factor_importance.tsv`, `..._repeats.tsv`, `clean_main_model_fit.tsv`, `clean_main_model_coefficients.tsv`, `clean_main_factor_ranks.tsv`, `..._rank_stability.tsv`, `clean_main_seed_summary.tsv`, `clean_main_exact_zero_sensitivity.tsv` | `..._caption.md` |
| `feature_importance/clean_interactions/figures/interaction_block_utility` | `interaction_block_drop_utility.tsv`, `interaction_model_incremental_utility.tsv`, `interaction_model_fit.tsv`, `interaction_model_coefficients.tsv`, `interaction_figure_decision.tsv`, `interaction_exact_zero_sensitivity.tsv` | `..._caption.md` |
| `feature_importance/route_specific/figures/route_specific_factor_importance` | `route_specific_factor_importance.tsv`, `..._repeats.tsv`, `route_specific_model_fit.tsv`, `route_specific_coefficients.tsv`, `route_specific_factor_ranks.tsv`, `route_specific_exact_zero_sensitivity.tsv` | `..._caption.md` |

**Two constraints govern every claim here.** On the clean set **lexicality and
training exposure are perfectly confounded** — every Real item is
`TRAINED_REAL_EXACT` and every Pseudo item is `NOVEL_PSEUDOWORD` — so the two
never enter one model and the factor is reported as a **lexicality/exposure
contrast**. **Zipf frequency is undefined for pseudowords**, is never imputed,
and is excluded from every all-item clean model (see Sprint 3 for the
trained-real frequency analysis).

The split is **grouped by `item_id`** so all three route rows of an item stay
together, and the identical split is reused across all four seeds and all three
models. Permutation acts on **raw factors**, rebuilding every derived column.
Grouped importance is **unsigned**; coefficients are reported separately, with
route as two contrasts (LTM − WM, FULL − WM) and never one collapsed number.

The **faithful** Dager feature importance (A11) is a **separate** analysis and
is not recomputed, replaced or pooled: `feature_importance/faithful_vs_adapted.md`.
Results and finding categories: `feature_importance/feature_importance_results.md`.

## Source tables

Figures derive from the validated canonical table
`outputs/behavioral_wfe_fulllexicon_93a577f/behavioral_analysis/tables/canonical_behavioral_item_table.tsv`
(14,400 rows = 4 seeds × 1,200 items × 3 routes), itself built from the four
per-seed enriched production tables under
`outputs/behavioral_wfe_fulllexicon_93a577f/full_wfe_evaluation/seed*/wfe_ar/`.

## Manifests

| Manifest | Scope | Status |
|---|---|---|
| `full_wfe_evaluation/_control/production_outputs.sha256` | original whole-tree snapshot | HISTORICAL / SUPERSEDED |
| `full_wfe_evaluation/_control/production_scientific_outputs_FINAL.sha256` | 36 immutable scientific outputs | authoritative, verifies 100 % |
| `full_wfe_evaluation/_control/production_operational_logs_FINAL.sha256` | 5 append-only operational logs | authoritative as of closure |
| `validation/sprint1_outputs.sha256` | every file in this report directory | authoritative |
| `morphology/validation/morphology_outputs.sha256` | Sprint-2 outputs | authoritative |
| `frequency/validation/frequency_outputs.sha256` | Sprint-3 outputs | authoritative |
| `error_taxonomy/validation/error_taxonomy_outputs.sha256` | Sprint-4 outputs | authoritative |
| `feature_importance/validation/feature_importance_outputs.sha256` | Sprint-5 outputs | authoritative |

**Living-file policy.** `README.md` and `analysis_matrix.tsv` are living
documents that each sprint extends; every earlier manifest lists them, so their
hashes are expected to change and are not scientific regressions. Every other
artefact recorded in an earlier manifest must stay byte-identical.

Why the original manifest reads 40/41: `full_wfe_evaluation/_control/manifest_closure_note.md`.

## Commits and environment

| Item | Value |
|---|---|
| Training commit | `93a577fd9822955fa272ee733fa7e2acf81f1333` |
| Evaluation-code commit | `e876b755d0475ed11e5fbc0419a0bd8860dfd325` |
| Analysis-code base commit | `e876b755d0475ed11e5fbc0419a0bd8860dfd325` (Sprint-1 package not yet committed) |
| Analysis package | `scripts/behavioral_analysis/` |
| Environment | Python 3.11.15, numpy 2.4.6, pandas 3.0.3, scipy 1.17.1, scikit-learn 1.9.0, matplotlib 3.11.0, Levenshtein 0.27.3 |

Full machine-readable record: `behavioral_analysis_provenance.json`.

## Bootstrap configuration

Hierarchical: seeds resampled with replacement, then items with replacement
within each analysis-set × stratum cell (cell sizes preserved); the statistic
is recomputed per replicate and averaged over the resampled seeds.
**B = 10,000, random seed 20260730, 95 % percentile interval.** Frozen before
results were seen and unchanged since.

## Seed and route policy

Seeds 19, 20, 21 and 22 are all primary; **seed 21 is never excluded** and is
individually visible in every figure that shows seeds. Seeds 19/20/22 form an
exact-ceiling sensitivity set only. Routes `full`, `wm` and `ltm` share one
decoding convention: deterministic autoregressive, forced-length readout, no
noise, no teacher forcing.

## Analysis sets

`LICHTHEIM_CLEAN` (1,062 = 671 trained real + 391 novel pseudo) backs the
primary figures. `FAITHFUL_WFE_ALL` (1,200) backs the faithful replication.
`ALL_WITH_EXPOSURE_STRATA` (1,200) backs the exposure-status figure. Two
frequency sets (671 / 678) are computed but not yet plotted.

## Validation

`validation/sprint1_validation.json` — assertions and verdicts.
`validation/sprint1_test_log.txt` — commands and exit codes.
`validation/sprint1_output_inventory.tsv` — every file with size and hash.
`validation/sprint1_diff_review.md` — review of the tracked diff.
