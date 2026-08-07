# Yair corrections pass — frozen specification

**Written and saved before any result in this pass was computed.** Machine-
readable twin: `yair_corrections_spec.json`.

This is **not** a new WFE analysis programme. Every number below is a
reaggregation of the frozen predictions already in
`outputs/behavioral_wfe_fulllexicon_93a577f/behavioral_analysis/tables/canonical_behavioral_item_table.tsv`
(SHA256 `8988aff6…`, 14,400 rows = 4 seeds × 1,200 items × 3 routes), plus the
already-validated metadata columns it carries. No checkpoint is loaded, no
inference is run, no retraining, no SSP, no long-pseudoword generation, no
mechanistic experiment.

No closed-release figure, table or manifest is modified. All output goes to
`reports/behavioral_wfe_fulllexicon_93a577f/yair_corrections/`, chosen because
it matches the existing per-topic report convention (`error_taxonomy/`,
`feature_importance/`, `frequency/`, `morphology/`, each with
`_control/ tables/ figures/ validation/`).

**No new statistical model is introduced.** Where an interval is reported it
comes from the already-validated frozen estimator `bootstrap.cell_mean_bootstrap`
(B = 10,000, seed 20260730, 95 % percentile, seeds resampled then items),
reused unchanged on a different outcome column. The serial-position analysis
reuses `compute.serial_position_tables` and `compute.zip_mismatch_positions`
verbatim.

---

## T1 — Word error rate by exact length, route and lexicality

- **Question.** At each exact phoneme length, what fraction of *whole words* does
  each route get wrong, separately for trained real words and novel pseudowords?
- **Population.** `LICHTHEIM_CLEAN`: 671 `TRAINED_REAL_EXACT` +
  391 `NOVEL_PSEUDOWORD` = 1,062 items.
- **Metric.** `word_error` (1 − exact match), the **primary** y-axis. Mean raw
  edit distance is carried in the TSV and named in the caption only as severity
  *after* word-level failure; it is never the headline and never plotted as the
  primary quantity.
- **Grain.** seed × route × lexicality × exact length (4 × 3 × 2 × 6), plus an
  across-seed summary at route × lexicality × length.
- **Joins.** None; single-source aggregation of the canonical table.
- **Exclusions.** Rows outside `in_LICHTHEIM_CLEAN`. Length 6 does not exist in
  WFE by construction and is not imputed.
- **Uncertainty / display.** All four seeds visible as individual points at every
  bin. Across-seed mean drawn as a line. `cell_mean_bootstrap` 95 % interval
  drawn as a light band. Item count printed for every lexicality × length bin.
  No smoothing anywhere; no slope is presented as the central result.
- **Estimability.** A cell is reported whenever it contains ≥ 1 item. Cells are
  never dropped for being at floor; a zero word-error rate is a result.
- **Sources.** canonical table only.
- **Outputs.** `tables/word_error_by_length_seed.tsv`,
  `tables/word_error_by_length_summary.tsv`,
  `tables/word_error_by_length_item_counts.tsv`,
  `figures/yc1_word_error_by_length.{png,pdf,svg}` + caption.
- **Faithful companion.** Produced only if it materially clarifies why the old
  faithful Figure 2A differs from the clean analysis, as a **separate** table and
  figure. Faithful and clean populations are never pooled.

## T2 — Faithful source-real error audit

- **Question.** Among the 800 items the WFE stimulus set labels *real*, which
  ones does the model actually get wrong, and what do those errors look like?
- **Population.** All 800 `source_lexicality == "real"` items in
  `FAITHFUL_WFE_ALL`, all 4 seeds, all 3 routes (9,600 seed × route × item rows).
- **Metric.** Exhaustive item-level listing of every row with `word_error == 1`,
  carrying the **literal existing columns** — no derived scores.
- **Grain.** One row per erroneous seed × route × item event.
- **Joins.** None; the canonical table already carries phonology, prediction,
  exposure status, Zipf frequency, morphology, edit operations and
  `eos_position`. EOS class is derived with the frozen
  `eos_diagnostics.classify_eos`, reused unchanged.
- **Exclusions.** None inside the 800.
- **Uncertainty / display.** Descriptive only. Both **unique erroneous item
  counts** and **seed × item error-event counts** are reported wherever a
  proportion appears, because they answer different questions.
- **Estimability.** Descriptive; no model is fitted. Association with length and
  low frequency is reported as observed rates per bin, explicitly labelled
  descriptive and not adjusted for the exposure confound.
- **Sources.** canonical table; `eos_diagnostics.classify_eos`.
- **Outputs.** `tables/faithful_real_error_events.tsv` (exhaustive),
  `tables/faithful_real_error_summary.tsv`,
  `tables/faithful_real_error_by_exposure.tsv`,
  `tables/faithful_real_error_recurrence.tsv`,
  `figures/yc2_faithful_real_error_composition.{png,pdf,svg}` + caption.

## T3 — LTM successful-pseudoword audit

- **Question.** Which novel pseudowords does the LTM-only route reproduce
  exactly, and how do the always-successful, mixed and always-failed groups
  differ on already-validated fields?
- **Population.** 391 `NOVEL_PSEUDOWORD` items, **LTM route only**, 4 seeds.
- **Metric.** Per item, the count of seeds with `exact_match == 1`, mapped to
  `ALWAYS_SUCCESSFUL` (4/4), `MIXED_SUCCESS` (1–3/4), `ALWAYS_FAILED` (0/4).
  These three classes are exhaustive and mutually exclusive by construction.
- **Grain.** item (with all four seed outcomes preserved), then group.
- **Joins.** None.
- **Exclusions.** None inside the 391.
- **Comparison fields — already-validated only.** `target_length`;
  `lexical_confidence`; `gate` **as an auxiliary linked variable only**;
  edit-operation counts in failed seeds; EOS status in failed seeds.
- **Explicitly forbidden here.** Confidence and gate are **not** treated as
  independent evidence — `gate = sigmoid(2.0 · (confidence − 0.7))` is a
  deterministic monotone function of confidence, so they are one variable
  reported twice. No lexicalization conclusion is drawn.
- **Missing measures.** Phonotacticity, distance to the training lexicon, and
  suffix/phonemic complexity were searched for across `scripts/`, `reports/`,
  `outputs/` and `docs/`. **None exists as a validated documented feature for
  WFE items.** They are recorded as `UNAVAILABLE_VALIDATED_MEASURE`. No proxy is
  invented and no new feature is computed.
- **Estimability.** Descriptive group comparison. Group means are reported with
  n; no significance test is introduced.
- **Sources.** canonical table only.
- **Outputs.** `tables/ltm_pseudoword_item_success.tsv`,
  `tables/ltm_pseudoword_group_summary.tsv`,
  `tables/ltm_pseudoword_feature_summary.tsv`,
  `tables/ltm_pseudoword_unavailable_measures.tsv`,
  `figures/yc3_ltm_pseudoword_success.{png,pdf,svg}` + caption.

## T4 — Faithful Figure 2C by route

- **Question.** Does the faithful serial-position error profile differ across
  FULL, WM and LTM?
- **Recovered method (verified, not assumed).** The producing driver
  (`outputs/.../_control/run_faithful_and_confirmatory.py`) is not in the current
  tree, but its logic was promoted verbatim into
  `compute.serial_position_tables` + `compute.zip_mismatch_positions`
  (`analysis_code_migration.tsv`: PROMOTED, "zip-mismatch and PCHIP unchanged").
  Applying that frozen function to the faithful subset at `route == "full"`
  reproduces the frozen `faithful_figure2C_table.tsv` with
  **max |diff| = 8.8e-17 on the error rate and 0 on the counts** — a
  reproduction gate that must pass before any extension is written.
- **Estimand, stated precisely.** For each (lexicality, length) cell: 1-based
  positional mismatch between target and prediction under **zip alignment with
  the prediction re-padded to the target length** — *not* a Levenshtein
  alignment. Numerator: number of items × seeds mismatching at that position.
  Denominator: items × seeds in that (lexicality, length) cell. Everything after
  a trimmed prediction becomes `<PAD>` and therefore counts as a mismatch, which
  is how Dager's blanking-after-EOS is recovered. Seeds are **pooled**, not
  averaged. Relative position is `(index − 1)/(length − 1)`; lengths < 2 are
  skipped.
- **Population.** `FAITHFUL_WFE_ALL`, faithful `real`/`pseudo` source labels
  retained (800/400) — **not** exposure categories.
- **Display.** Empirical per-length values plotted as points and thin lines. The
  PCHIP-interpolated curve is shown only alongside its empirical points, never
  alone.
- **Non-overwrite.** The existing faithful Figure 2C files are read-only in this
  pass; new files carry the `yc4_` prefix in the corrections directory.
- **Outputs.** `tables/faithful_figure2C_by_route.tsv`,
  `tables/faithful_figure2C_by_route_interpolated.tsv`,
  `tables/faithful_figure2C_reproduction_check.tsv`,
  `figures/yc4_faithful_serial_position_by_route.{png,pdf,svg}` + caption.

## T5 — Feature importance by route: estimability audit

- **Question.** Can route-specific feature importance be estimated, and if so for
  which routes?
- **Method.** **Audit, not refit.** The adapted (A15) route-specific analysis
  already exists and is validated:
  `reports/.../feature_importance/route_specific/tables/`. Its frozen protocol —
  Ridge α = 1.0, item-grouped 80/20 split `random_state=42`,
  `permutation_importance` `n_repeats=100 random_state=42`, permutation on raw
  factors, outcome `raw_edit_distance` — is recorded and reused **by reading its
  outputs**, not by refitting. The faithful (A11) FI is read-only and is never
  pooled with it.
- **Added quantity.** Outcome density per route × population (share of non-zero
  outcomes, train and test), which is what governs estimability.
- **Rule.** A route is reported as estimable only where the existing validated
  run recorded `model_status == "OK"`. Otherwise the diagnostic table carries
  `NOT_ESTIMABLE_CEILING_OR_SPARSE_OUTCOME` together with the original status.
  **A non-estimable cell is never given an importance of zero**, and negative
  test R² is preserved as recorded rather than clipped.
- **Outputs.** `tables/fi_route_estimability.tsv`,
  `tables/fi_outcome_density.tsv`. A figure is produced only for routes that are
  estimable.

---

## Global rules for this pass

1. Frozen predictions only; no torch import, no checkpoint load, no inference.
2. No scientific value is hardcoded — every number is recomputed from the
   canonical table at run time and asserted against the frozen counts
   (671/391/1062, 800/400, 14,400 rows, 4 seeds, 3 routes, lengths 3-9 without
   6).
3. Faithful and clean populations are never pooled in one estimate.
4. Existing loaders (`io.load_canonical`, `io.clean_subset`, `io.write_table`),
   plotting helpers (`plotting.save_figure`) and constants (`common`) are reused.
5. Red = real, blue = pseudo, reserved; exposure categories use the neutral grey
   palette.
6. Prior release outputs are verified byte-unchanged at the end of the pass.
7. Nothing is staged, committed, pushed, or promoted into the final release.
