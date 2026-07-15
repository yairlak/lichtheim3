# Lichtheim3 Gridsearch and Training Plan

> **Status: DRAFT — do not launch training from this document before repo inspection and supervisor validation.**

Status: draft strategy to validate with Yair before launching any large run.
Scope: `lichtheim3`, branch `eval/external-csv-datasets`, current prototype checkpoint `checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt`.
Main rule: do not use teacher forcing as the main behavioral evaluation; do not use WFE/SSP as training data; do not treat held-out real words as the main figure category.

---

## 1. Scientific goal

The immediate goal is not to redesign Lichtheim3. The goal is to choose training and evaluation hyperparameters that make the current dual-route prototype defensible:

1. The model must learn trained real-word repetition robustly.
2. Evaluation must be autoregressive for behavioral claims.
3. WM noise must be explicitly documented as either part of training, part of evaluation/lesion analysis, or both.
4. The gate alpha and semantic alignment loss must not remain arbitrary if they materially affect WFE behavior.
5. The grid must remain small enough to interpret and discuss with Yair.

---

## 2. SWP/Dager inspiration, but not blind copying

SWP/Dager is useful for protocol discipline:

- Train a simple seq2seq model until it reaches perfect training performance.
- Use autoregressive decoding: previous model output is fed into the next decoder step.
- Use error rate and edit distance, not just per-position accuracy.
- Select models by: perfect CV train accuracy, best CV validation accuracy, then smallest model complexity.
- Test robustness across seeds after selecting hyperparameters.

Attested SWP/Dager reference values:

| Field | SWP/Dager value |
|---|---:|
| Selected architecture | LSTM encoder-decoder |
| Selected hidden size | 128 |
| Selected batch size | 2048 |
| Selected dropout | 0.0 |
| Selected learning rate | 0.001 |
| LSTM training epochs | 100 |
| Optimizer | Adam |
| Loss | Cross-entropy ignoring PAD |
| Final-grid hidden sizes | 64, 128 |
| Final-grid dropout | 0.0, 0.1, 0.2 |
| Final-grid batch sizes | 1024, 2048, 4096 |
| Final-grid learning rates | 5e-4, 1e-3, 5e-3 |
| Preliminary teacher-forcing range | 0.0-0.7 |
| Teacher forcing conclusion | detrimental to learning in SWP/Dager |
| CV | 5-fold |
| Robustness | 10 seeds after selected HPs |

For Lichtheim3, do not copy the full SWP grid. The architecture is different: dual-route, GloVe alignment, gate, WM noise, semantic bank. Therefore, SWP gives a reference for scale and discipline, not a list of mandatory values.

---

## 3. Model-selection principle

Do not select only on train ceiling. The selection hierarchy should be:

### Hard filters

A candidate fails if any of these occurs:

1. Training does not converge to near-ceiling on trained real words within the planned budget.
2. Autoregressive train/validation evaluation collapses relative to teacher-forced ceiling.
3. WFE pseudoword autoregressive performance collapses relative to the current checkpoint baseline.
4. The model degenerates into EOS/truncation errors on long items.
5. The gate saturates pathologically for most items, unless this is explicitly intended and documented.
6. Full/gated route is consistently worse than both route-isolated controls on the key behavioral split.
7. Runs are non-reproducible because evaluation noise is accidentally on in deterministic evaluation.

### Primary selection metrics

Use these to rank surviving candidates:

1. Train exact match, deterministic and autoregressive.
2. Validation exact match, deterministic and autoregressive.
3. Validation edit distance and normalized edit distance, autoregressive.
4. WFE Dager-strict real exact match: only train-seen real WFE words.
5. WFE pseudoword exact match, autoregressive.
6. WFE pseudoword edit distance and normalized edit distance, autoregressive.
7. Length-effect slope on WFE pseudowords, using edit distance and normalized edit distance.
8. Long-short contrast on WFE pseudowords.
9. Route dissociation: full/gated, WM-only, LTM-only.
10. Stability across seeds for top candidates.

### Diagnostic metrics

Do not use these alone to select the final model, but always log them:

- Teacher-forced train/val/WFE ceiling.
- Route-isolated full/WM/LTM exact match and edit distance.
- Gate confidence and gate value distributions.
- WM-noise sweep performance, explicitly labelled.
- Serial-position error curves.
- Error-type decomposition: substitutions, insertions, deletions, premature EOS.
- LTM nearest-neighbor lexicalization examples.
- SSP autoregressive results, only second-level after WFE.

### Candidate final-figure metrics

Use for final paper-style figures after the full retrain:

- WFE real words vs pseudowords, with real restricted to trained lexicon words.
- WFE pseudoword accuracy/edit distance by length.
- Length-effect slope and long-short contrast.
- Serial-position curves.
- Route dissociation full/WM/LTM.
- WM-noise figures only if explicitly labelled as noisy/cognitive or lesion-like.

Do not use `unseen forms` as a main figure category. Held-out real + novel real + pseudowords is a familiarity/generalization diagnostic, not a lexicality condition.

---

## 4. Stage 0 - sanity checks before gridsearch

No training. Run these on the current checkpoint and write a baseline summary.

### 4.1 Required checks

1. Confirm train deterministic teacher-forced ceiling, WM noise off.
2. Confirm train deterministic autoregressive evaluation, WM noise off.
3. Confirm validation deterministic teacher-forced evaluation, WM noise off.
4. Confirm validation deterministic autoregressive evaluation, WM noise off.
5. Confirm WFE teacher-forced, WM noise off, as ceiling/debug.
6. Confirm WFE autoregressive, WM noise off, as main baseline.
7. Confirm route outputs full/WM/LTM are all written.
8. Confirm metrics include exact match, edit distance, normalized edit distance, and item-level predictions.
9. Confirm repeated deterministic eval gives identical metrics.
10. Confirm whether autoregressive + WM noise is implemented end-to-end. If not, mark Regime C as not available for standard runs.

### 4.2 Stage 0 outputs

Use one output root:

```text
outputs/gridsearch_baselines/current_ckpt/
```

Expected files:

```text
train_tf_metrics.json
train_ar_metrics.json
val_tf_metrics.json
val_ar_metrics.json
wfe_tf_metrics.json
wfe_ar_metrics.json
wfe_ar_item_level_predictions.tsv
route_metrics_full_wm_ltm.json
length_effects_wfe_ar.json
summary.md
```

Exact file names should be adjusted to the repo's actual evaluator outputs.

### 4.3 Stage 0 commands: to verify in repo

Known WFE AR command from supplied docs:

```bash
python scripts/external_eval.py \
  --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt \
  --out_dir outputs/gridsearch_baselines/current_ckpt \
  --decode autoregressive \
  --wfe_only
```

Known WFE teacher-forced ceiling/debug command:

```bash
python scripts/external_eval.py \
  --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt \
  --out_dir outputs/gridsearch_baselines/current_ckpt \
  --decode teacher_forced \
  --wfe_only
```

Train/validation teacher-forced ceiling command:

```bash
python scripts/evaluate_train_lexicon_ceiling.py \
  --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt \
  --lexicon_path data/lexicon_en_glove_covered.tsv \
  --out_dir outputs/gridsearch_baselines/current_ckpt/train_tf \
  --include_val
```

Train/validation autoregressive command (outputs to `<out_dir>/ar/`):

```bash
python scripts/evaluate_train_lexicon_ceiling.py \
  --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt \
  --lexicon_path data/lexicon_en_glove_covered.tsv \
  --out_dir outputs/gridsearch_baselines/current_ckpt/train_ar \
  --decode autoregressive \
  --include_val
```

---

## 5. Stage A - optimization and autoregressive-compatible training

### 5.1 Objective

Find a training regime that learns repetition well while remaining compatible with autoregressive evaluation. This stage is about optimization and decoding exposure, not about gate or noise theory.

### 5.2 Candidates — UPDATED AFTER CODE PATCH (2026-07)

> ✓ **Patch applied:** `teacher_forcing_ratio`, `--interference_noise`, and `--gate_alpha` are now CLI-exposed. Scheduled sampling is implemented in `train.py`. Train/val AR evaluation is now available via `--decode autoregressive` in `evaluate_train_lexicon_ceiling.py`.

**What can be varied now (all CLI-exposed):**

| Dimension | Candidate values | Rationale | CLI status |
|---|---:|---|---|
| Learning rate | `1e-3`, `5e-4` | Both in SWP/Dager final grid; `1e-3` is L3 default | `--lr` ✓ |
| Teacher forcing ratio | `1.0` (default, TF), `0.0` (fully AR), or intermediate | Scheduled sampling now implemented | `--teacher_forcing_ratio` ✓ PATCHED |
| WM noise during training | `0.10` (default), `0.0` (off) | Stage B; now CLI-exposed | `--interference_noise` ✓ PATCHED |
| Gate alpha | `4.0` (default), `2.0`, `8.0` | Stage C; now CLI-exposed | `--gate_alpha` ✓ PATCHED |
| Batch size | keep 64 (default) | SWP used 1024-4096; test only if needed | `--batch_size` ✓ |
| Dropout | **N/A — not implemented** | No dropout in any module | Not available |
| Semantic alignment loss weight | current value 1.0 (fixed) | Stage C only; still config-only | Still requires code support |

**Stage A grid (now feasible with patched CLI):**

```text
2 learning rates (1e-3, 5e-4) × 2 TF ratios (1.0, 0.0) = 4 runs (1 seed each)
```

Or if Yair approves expanding TF:

```text
2 learning rates × 3 TF ratios (1.0, 0.2, 0.0) × noise=0.10 fixed = 6 runs
```

**Stage A + B combined:**

```text
2 LR × 2 TF ratios × 2 noise (0.0, 0.10) = 8 runs (1 seed each); reduce by screening
```

**Important: scheduled sampling overhead.** For `tf_ratio < 1.0`, the training loop runs the model step-by-step (S encoder calls per batch instead of 1), making training ~S× slower. With max_phonemes=9, expect ~10× slower training for fully AR. Plan accordingly for budget.

### 5.3 Seeds

Run Stage A first with 1 seed only after a short smoke test. Then rerun the top two candidates with 2 additional seeds. Do not start with 10 seeds.

Approximate run count (all CLI flags now available after 2026-07 patch):

- Minimal Stage A (LR only, TF fixed at 1.0): 2 single-seed runs + 1 top candidate × 2 extra seeds = 4 runs.
- Stage A with TF ratio variation (LR × TF ratio): 4 single-seed runs + 2 top candidates × 2 extra seeds = 8 runs.

### 5.4 Stage A success

A candidate is promising if:

- it reaches train ceiling or the best attainable train AR performance within the budget;
- validation AR exact/edit distance is not worse than the current checkpoint baseline;
- WFE pseudoword AR exact/edit distance is not worse than the current checkpoint baseline;
- it does not show pathological EOS/truncation on long pseudowords;
- its teacher-forced ceiling remains high, but teacher-forced is not the selection metric.

---

## 6. Stage B - robustness and length effect

### 6.1 Objective

Decide whether WM noise is part of training, only an evaluation/lesion manipulation, or both. Also test whether small regularization improves AR pseudoword generalization without destroying trained-word ceiling.

### 6.2 Candidates

Run only on the top 1-2 Stage A candidates.

| Dimension | Candidate values | Status |
|---|---:|---|
| WM noise during training | `0.0`, current default `0.10` | Attested mechanism; current default sigma attested |
| Dropout | `0.0`, optional `0.2` | Only if existing in L3 code path; SWP tested 0.2 but selected 0.0 |
| Teacher forcing ratio | fixed to top Stage A value | Do not cross with all TF ratios |
| Learning rate | fixed to top Stage A value | Do not cross with all LRs |

Recommended Stage B run count:

- If dropout is not implemented or not approved: top 2 Stage A configs x 2 noise values = 4 runs.
- If dropout is implemented and Yair approves: top 1 Stage A config x 2 noise values x 2 dropout values = 4 runs.
- Do not run both expansions simultaneously unless Stage A clearly fails.

### 6.3 Evaluation

Primary evaluation remains WFE autoregressive no-noise. Then add explicitly labelled noisy evaluation:

1. WFE AR no-noise: main comparability condition.
2. WFE AR + WM noise at the trained/default sigma: cognitive/noisy condition, if implemented.
3. Optional light sweep only after selecting a candidate: `sigma=0.0`, `0.10`, and possibly `0.20` as diagnostic. Values above `0.20` are stress tests only.

### 6.4 Stage B success

A Stage B candidate is useful if it improves WFE pseudoword AR robustness or length-effect interpretability without reducing trained-real performance and without making the gate/route dissociation uninterpretable.

---

## 7. Stage C - gate and semantic balance

### 7.1 Objective

Check whether the current `gate_alpha = 4.0` is reasonable, and whether the semantic alignment loss is over- or under-weighted.

### 7.2 Gate alpha candidates

Use only one top Stage A/B training setup.

| Dimension | Candidate values | Rationale |
|---|---:|---|
| Gate alpha | `2.0`, `4.0`, `8.0` | Limited around current default 4.0; tests softer vs sharper routing |

Run count: 3 runs, 1 seed first. If alpha changes are large, rerun the best and the current `4.0` baseline with 2 extra seeds.

Do not test a large alpha grid. Do not introduce a learned gate now.

### 7.3 Semantic alignment loss candidates

The semantic alignment loss weight is `L_align = 1.0` (confirmed from `config.py:LossConfig.align`). If varied, use at most:

```text
current_value      (1.0)
0.5 * current      (0.5)
2.0 * current      (2.0)
```

Only run this if Stage C gate-alpha analysis or LTM diagnostics indicate that the LTM route is too weak, too dominant, or poorly calibrated. Otherwise keep it fixed.

### 7.4 Fixed-gate diagnostics

Fixed gates are diagnostics, not final model candidates:

```text
g = 0.0   WM-only equivalent
g = 0.25  mostly WM
g = 0.5   equal blend
g = 0.75  mostly LTM
g = 1.0   LTM-only equivalent
```

Use them to answer: is the learned/default gate blend better than a simple constant mixture for real vs pseudowords and short vs long items?

---

## 8. Approximate run plan

This is the recommended limited plan before final full retrain.

| Phase | Runs | Seeds | Purpose |
|---|---:|---:|---|
| Stage 0 baseline eval | 0 training runs | n/a | Current checkpoint baseline |
| Stage A initial | 4-6 | 1 | LR x TF ratio |
| Stage A confirmation | 4 | 2 extra seeds for top 2 | Stability of top candidates |
| Stage B | 4 | 1 | Training noise/dropout robustness |
| Stage C gate alpha | 3 | 1 | Gate alpha sensitivity |
| Final confirmation before full retrain | 3 | 3 seeds for top config if compute allows | Stability |

Total training runs before full retrain: approximately 15-20, not hundreds.

If compute is expensive, reduce to:

```text
Stage A: 4 runs
Stage B: 2 runs on top Stage A
Stage C: 3 alpha runs
Top config: 2 extra seeds
Total: 11 runs
```

---

## 9. Duration and compute

Duration is not available in the supplied files for Lichtheim3. Measure it by smoke runs before submitting the grid.

Required smoke measurements:

1. Time for 1 epoch on the current 29,571-word lexicon.
2. Time for deterministic train+val eval.
3. Time for WFE AR eval.
4. Time for WFE AR route-isolated eval full/WM/LTM.
5. GPU memory and CPU memory.

Use SWP wrapper only as rough context: the supplied SWP SLURM wrapper requested 1 A40 GPU, 4 CPUs, 10 GB RAM, and 10 hours for a 100-epoch LSTM run. Do not assume Lichtheim3 has identical runtime.

---

## 10. Standard evaluation package for every training run

Every completed checkpoint should run the same evaluation package:

1. Train deterministic teacher-forced, no WM noise.
2. Train deterministic autoregressive, no WM noise.
3. Validation deterministic teacher-forced, no WM noise.
4. Validation deterministic autoregressive, no WM noise.
5. WFE autoregressive, no WM noise: main behavioral result.
6. WFE teacher-forced, no WM noise: ceiling/debug appendix.
7. WFE route-isolated full/WM/LTM, autoregressive, no WM noise.
8. Length-effect analysis from WFE AR item-level predictions.
9. Optional WFE autoregressive with WM noise if Regime C is implemented and explicitly labelled.
10. Optional SSP autoregressive after WFE is satisfactory.

Output root convention:

```text
outputs/gridsearch/{stage}/{run_id}/
```

Checkpoint convention:

```text
checkpoints/gridsearch/{stage}/{run_id}.pt
```

Run ID convention:

```text
stageA_lr1e-3_tf0p0_noise0p10_alpha4_seed0
stageA_lr5e-4_tf0p2_noise0p10_alpha4_seed0
stageB_lr1e-3_tf0p0_noise0p00_alpha4_seed0
stageC_lr1e-3_tf0p0_noise0p10_alpha2_seed0
```

---

## 11. Summary tables to produce

Create these after each stage:

```text
outputs/gridsearch/summary/stageA_runs.tsv
outputs/gridsearch/summary/stageA_selection.md
outputs/gridsearch/summary/stageB_runs.tsv
outputs/gridsearch/summary/stageB_selection.md
outputs/gridsearch/summary/stageC_runs.tsv
outputs/gridsearch/summary/final_candidate_selection.md
```

Minimum columns:

```text
run_id
stage
seed
lr
teacher_forcing_ratio
train_wm_noise_sigma
dropout
gate_alpha
semantic_align_weight
train_exact_tf
train_exact_ar
val_exact_ar
val_edit_ar
val_norm_edit_ar
wfe_real_seen_exact_ar
wfe_pseudo_exact_ar
wfe_pseudo_edit_ar
wfe_pseudo_norm_edit_ar
length_slope_pseudo_edit_ar
long_short_pseudo_edit_ar
wm_route_pseudo_exact_ar
ltm_route_pseudo_exact_ar
full_route_pseudo_exact_ar
mean_gate_real_seen
mean_gate_pseudo
failure_flag
notes
```

---

## 12. Command templates

The exact Lichtheim3 training CLI must be inspected before use. The following is the desired command shape, not a verified command.

```bash
python scripts/train_checkpoint.py \
  --lexicon_path data/lexicon_en_glove_covered.tsv \
  --max_words 29571 \
  --epochs <EPOCHS_TO_VALIDATE> \
  --seed <SEED> \
  --lr <LR> \
  --teacher_forcing_ratio <TF_RATIO> \
  --interference_noise <TRAIN_WM_NOISE_SIGMA> \
  --gate_alpha <ALPHA> \
  --ckpt checkpoints/gridsearch/<STAGE>/<RUN_ID>.pt
```

Now-available flags (✓ patched):

```text
--teacher_forcing_ratio   float, default 1.0  (1.0=TF, 0.0=AR, intermediate=scheduled sampling)
--interference_noise      float, default None (uses WMConfig default 0.10; set 0.0 to disable)
--gate_alpha              float, default None (uses GatingConfig default 4.0)
--lr, --batch_size, --seed, --epochs, --max_words, --lexicon_path  (pre-existing)
```

Still requires code support:

```text
--dropout               Not implemented in any model module.
--semantic_align_weight LossConfig.align=1.0 is config-only; no CLI flag yet.
--eval_every / --early_stopping   Not implemented.
```

Known WFE evaluation command shape:

```bash
python scripts/external_eval.py \
  --ckpt checkpoints/gridsearch/<STAGE>/<RUN_ID>.pt \
  --out_dir outputs/gridsearch/<STAGE>/<RUN_ID> \
  --decode autoregressive \
  --wfe_only
```

Known WFE teacher-forced debug command shape:

```bash
python scripts/external_eval.py \
  --ckpt checkpoints/gridsearch/<STAGE>/<RUN_ID>.pt \
  --out_dir outputs/gridsearch/<STAGE>/<RUN_ID> \
  --decode teacher_forced \
  --wfe_only
```

---

## 13. Questions to send Yair before launching large runs

1. Should training-time teacher forcing be treated as an optimization aid only, or disallowed for cognitive plausibility?
2. Which teacher-forcing ratios should be approved: `0.0`, `0.2`, and optionally `0.5`?
3. Is WM noise supposed to be part of the trained cognitive mechanism, or only an evaluation/lesion perturbation?
4. Should the main no-noise model be trained with `interference_noise=0.0` or with the current default `0.10`?
5. Is `gate_alpha=4.0` an arbitrary default or already tuned historically?
6. Should gate alpha be chosen by validation/WFE AR behavior, or fixed for cognitive interpretability?
7. Should semantic alignment loss be tuned now, or held fixed until the clean Lichtheim3 architecture?
8. Are we allowed to use validation real words for hyperparameter selection, while keeping WFE external?
9. How many seeds are scientifically sufficient before the full retrain: 3 or 10?
10. Should final training use the full 29,571 GloVe-covered lexicon without a held-out real split, as in SWP/Dager, and leave WFE real-vs-pseudo as the external figure?

---

## 14. Risks

| Risk | Mitigation |
|---|---|
| Teacher-forced training looks good but AR behavior fails | Select on AR train/val/WFE metrics, not TF ceiling |
| Evaluation noise accidentally on | Determinism check before every stage |
| Full Cartesian explosion | Stage A/B/C sequential plan; freeze non-target dimensions |
| Gate alpha overfit to WFE | Use WFE as diagnostic; prefer validation and route interpretability |
| WFE real category contaminated by held-out real words | Main WFE real category = train-seen real only |
| LTM padding artifact changes results | Keep fixed batch/padding protocol; document; defer fix to retrained architecture |
| Training noise confounded with evaluation noise | Separate `train_wm_noise_sigma` from eval `--wm_noise` |
| Dropout changes architecture if not already implemented | Only vary dropout if existing code path exists |
| Semantic alignment weight unknown | Inspect before proposing numeric values |
| Final full retrain loses held-out validation | Use WFE and pseudowords for final external figures; keep teacher-forced ceiling appendix |

---

## 15. Full retrain after gridsearch

After hyperparameters are selected:

1. Retrain using the full 29,571 GloVe-covered lexicon.
2. Do not use WFE or SSP as training data.
3. Decide explicitly whether the final full retrain keeps a validation split for monitoring or trains on all 29,571 words after selection.
4. Main WFE figure: real words versus pseudowords. Real words should be trained real words for Dager comparability.
5. Do not use `unseen forms` as a main figure category.
6. Main evaluation: autoregressive, no WM noise unless the figure is explicitly noisy/cognitive.
7. Teacher-forced evaluation: ceiling/debug appendix only.
8. Route-isolated analysis: full/gated, WM-only, LTM-only.
9. Noise condition: explicitly documented in methods and figure captions.
10. Save final model config, optimizer/training metadata, seeds, and exact commit hash.
