# Full-Lexicon Ceiling

> **Status note (added later; the rest of this document is unchanged).**
> This document is the **historical record of cohort `93a577f` as it was created
> and archived**, under the selection criterion in force at the time:
> **X = 2** consecutive zero-error checkpoints. Everything below — the summary,
> the per-seed table, the selected-checkpoint table and the archive contents —
> is preserved exactly as written and is still accurate *for that criterion*.
>
> The **current canonical policy is X = 5**, under which only **seed 19 / epoch
> 155** and **seed 22 / epoch 140** qualify. Seed 20 (streak of 2) and seed 21
> (never reached zero) are historical, non-canonical checkpoints.
>
> Raising the criterion did **not** change any selected epoch: seeds 19 and 22
> select 155 and 140 at every value of X. Only cohort membership changes.
>
> For which result family uses which cohort, see
> [`canonical_selection_X5.md`](canonical_selection_X5.md). For the recovered
> training recipe see `configs/canonical_93a577f.yaml`.

## Summary

The full-lexicon ceiling experiment trains the Lichtheim3 dual-route model on the entire 29,571-word GloVe-covered lexicon used in this project (no validation set) and measures whether the model achieves near-perfect
autoregressive accuracy on all training items.

**Result**: 3/4 seeds reach exact-zero training errors; 3/4 achieve stable-zero (two or more
consecutive zero-error checkpoints); all 4 seeds reach at most 1 error. All cohort criteria pass.

## Training configuration

| Parameter | Value |
|-----------|-------|
| `split_mode` | `full_lexicon` |
| `train_all_words` | `True` |
| `max_words` | 30,000 |
| `n_train` | 29,571 |
| `n_val` | 0 |
| `epochs` | 200 |
| `seeds` | 19, 20, 21, 22 |
| Evaluation schedule | epochs 105–200, every 5 (20 checkpoints/seed, 80 total) |
| Evaluation mode | deterministic autoregressive |

Note: `--train_all_words` does not override `--max_words`. Setting `--max_words 30000` is required
to include all 29,571 words from the lexicon TSV; the default (4,000) would silently truncate.

## Dataset provenance (F0)

| Field | Value |
|-------|-------|
| `lexicon_file_sha256` | `ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66` |
| `ordered_training_words_sha256` | `0cb1c6172a7c2aea8a503549ffdf32543da820e1c505e0885f3999d6e50f7fa1` |
| `sorted_training_words_sha256` | `f9721c17f97d0f2a1afeb97ce917075db202f21ad471dcb8755a26226de7d63a` |
| `n_glove_found` | 29,571 |
| `n_glove_fallback` | 0 |
| `split_seed_effective` | `null` (no split; verified match at evaluation time) |

These hashes are verified at evaluation time by `scripts/evaluate_train_lexicon_ceiling.py`
(provenance schema v1, V3.2). In full-lexicon mode `split_seed_effective` is legitimately `null`;
the evaluator confirms `null == null` as a passing mandatory check, not a skipped one.

## Cohort results

80 evaluations validated (80/80).

| Criterion | Result |
|-----------|--------|
| ≥1 seed with zero errors | **PASS** |
| ≥2 seeds with zero errors | **PASS** |
| ≥2 seeds with stable zero | **PASS** |
| All 4 seeds with ≤1 error | **PASS** |

Stable zero = at least two consecutive scheduled checkpoints with zero training errors.
(This is the **historical X = 2** criterion under which this cohort was selected and
archived. The current canonical criterion is X = 5; see the status note above.)

### Per-seed results

| Seed | Best errors | First zero | Stable zero | Longest streak | Errors @ e200 | Selected epoch |
|---:|---:|---:|:---:|---:|---:|---:|
| 19 | 0 | 140 | yes | 6 | 2 | 155 |
| 20 | 0 | 130 | yes | 2 | 2 | 130 |
| 21 | 1 | — | no | 0 | 1 | 145 |
| 22 | 0 | 140 | yes | 13 | 0 | 140 |

### Non-monotonic behavior

Seeds 19 and 20 show non-monotonic training accuracy: both reach zero errors mid-training but end
epoch 200 with 2 errors each. Checkpoint selection therefore uses the first stable-zero streak onset,
not the first zero epoch or the final checkpoint.

- **Seed 19**: isolated zero at epoch 140; stable-zero streak epochs 155–180; drifts back to 2 errors
  by epoch 200. Selected: epoch 155 (first stable-zero onset).
- **Seed 20**: stable-zero epochs 130–135; isolated zero again at epoch 195; 2 errors at epoch 200.
  Selected: epoch 130.
- **Seed 22**: continuous stable-zero from epoch 140 through epoch 200 (13 consecutive checkpoints).
  Selected: epoch 140.
- **Seed 21**: never reaches zero; best is 1 error, achieved at epochs 145–200 with interruptions.
  Selected: epoch 145 (earliest minimum-error checkpoint).

## Selected checkpoints

Selection rule (applied per seed, in priority order):
1. Earliest checkpoint beginning the first stable-zero streak.
2. If no stable-zero streak: earliest exact-zero checkpoint.
3. If no exact-zero: earliest checkpoint with the minimum error count.

| Seed | Epoch | n_errors | exact_match | Stable-zero streak | Selection reason | SHA256 |
|---:|---:|---:|---:|---|---|---|
| 19 | 155 | 0 | 1.000000 | 155–180 | first_stable_zero | `7d05f9c2ad5a53e705f7d55ccde2581754918938d8ca888da35c0a859666478e` |
| 20 | 130 | 0 | 1.000000 | 130–135 | first_stable_zero | `b44548b6916ea89c6f099402b78031063445e572932acee8dd7558a73dfc6cfb` |
| 21 | 145 | 1 | 0.999966 | — | earliest_min_error | `ab58092e7c2bfac42ab977352e6d5d6416ca605b71a3eacb777300060b30f5cf` |
| 22 | 140 | 0 | 1.000000 | 140–200 (13) | first_stable_zero | `a15846cbf3c7df88ed289512bbb20cbefd2121d0deec1b39f363932a743da595` |

## Checkpoint policy

| Use case | Seeds |
|----------|-------|
| Main results (full cohort) | 19, 20, 21, 22 |
| Exact-zero sensitivity analysis | 19, 20, 22 |
| Canonical illustrative checkpoint | 22, epoch 140 |

Checkpoints are **not** committed to git. They reside in the archive and on the original compute
cluster at `/lustre/fsn1/projects/rech/llg/uss35bp/lichtheim3/checkpoints/fulllexicon_cohort_93a577f/`.

## Archive

```
archives/fulllexicon_93a577f/lichtheim3_fulllexicon_93a577f_final_20260729.tar.gz
SHA256: 5ef9872f17b957ea5bc4589f490e7506f20e4c520666e4d35add52f9ed1086e5
```

Archive structure (extracted under `archives/fulllexicon_93a577f/extracted/`):

```
fulllexicon_final_bundle_93a577f/
  selected_checkpoints/       — 4 selected .pt files (one per seed)
  selected_evaluations/       — 4 directories, each with metrics.json, predictions, summary
  aggregate_results/          — cohort_summary.json/.md, selected_checkpoints.tsv, seed_summary.tsv
  control/                    — SLURM submission scripts and manifests
  bundle_manifest.json        — machine-readable bundle provenance
fulllexicon_controls/
  production_93a577f/diagnostics/seed21_residual_diagnostic/
```

**Not committed to git**: `.pt` files, `item_level_predictions.tsv`, `train_errors.tsv`, `.tar.gz`.

## Evaluator

Script: `scripts/evaluate_train_lexicon_ceiling.py` (V3.2, provenance schema v1)

All 15 mandatory provenance fields are checked at evaluation time. Output: `metrics.json` with native
Python types (not stringified), verified by `test_full_lexicon.py` T37–T40. Schema version is
encoded in each checkpoint under `provenance_schema_version`; absent = legacy warning; 1 = strict v1
checks; other value = fail-fast.

## Commits

| Role | SHA |
|------|-----|
| `checkpoint_training_commit` | `93a577fd9822955fa272ee733fa7e2acf81f1333` |
| `documentation_commit` | See Git history for the commit containing this file |

## Report

Detailed per-seed tables: [`reports/fulllexicon_cohort_93a577f/`](../reports/fulllexicon_cohort_93a577f/)
