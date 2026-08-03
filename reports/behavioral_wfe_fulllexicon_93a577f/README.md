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
