# Lichtheim3 Training and Evaluation Regimes

> **Status: DRAFT — do not launch training from this document before repo inspection and supervisor validation.**

Status: draft protocol for gridsearch and final retrain.
Scope: current `lichtheim3` prototype, branch `eval/external-csv-datasets`.

---

## 1. Core distinction

Training regime and evaluation regime must be documented separately.

A model can be trained with one of several decoder-input policies, then evaluated under teacher-forced or autoregressive decoding. For behavioral claims, the primary evaluation must be autoregressive because the model uses its own previous output, so errors can propagate. Teacher-forced evaluation remains useful only as a ceiling/debug probe.

---

## 2. Training regimes — UPDATED AFTER CODE PATCH (2026-07)

> ✓ **Patch applied:** Scheduled sampling is now implemented via `train.py:_forward_scheduled_sampling`. The `--teacher_forcing_ratio` flag is available in `train_checkpoint.py`. All three regimes T0, T1, T2 are now runnable.
>
> **Performance note:** `tf_ratio < 1.0` uses a step-by-step training loop (~S× slower than TF=1.0 per batch, where S=max sequence length ≈ 10). Plan compute budget accordingly.

### Train Regime T2 — full teacher forcing (default, historical behavior)

```text
teacher_forcing_ratio = 1.0   (default)
dec_in = [BOS, p1, p2, ..., pT]    ← gold prefix at every step
```

Source: `data/dataset.py:43-57`; `train.py:run_epoch` (vectorized path when `tf_ratio >= 1.0`).
Status: `CONFIRMED_IN_CODE`. CLI: `--teacher_forcing_ratio 1.0` or omit the flag.

### Train Regime T1 — partial teacher forcing / scheduled sampling

```text
teacher_forcing_ratio ∈ (0.0, 1.0)
At each decoder step t: gold token with probability tf_ratio, model's argmax otherwise.
```

Source: `train.py:_forward_scheduled_sampling`.
Status: `IMPLEMENTED`. CLI: `--teacher_forcing_ratio 0.2` (or any float in (0, 1)).
Note: argmax selection at each step uses `.detach()` — gradients do not flow back through the token selection decision.

### Train Regime T0 — fully autoregressive training

```text
teacher_forcing_ratio = 0.0
Decoder input at every step t = model's own argmax prediction at t-1.
```

Source: `train.py:_forward_scheduled_sampling` with `tf_ratio=0.0`.
Status: `IMPLEMENTED`. CLI: `--teacher_forcing_ratio 0.0`.
Note (noise): with `tf_ratio < 1.0`, the WM encoder is called at each of the S decoder steps, so WM interference noise (if sigma > 0) is drawn independently per step — this differs from TF=1.0 where noise is drawn once per batch item. Document this in gridsearch logs.

### Validation always uses TF=1.0

Regardless of the training regime, `run_epoch` with `optim=None` (validation mode) always uses the fast vectorized TF=1.0 path. This gives a consistent loss estimate across training regimes.

### Recommendation

SWP/Dager found teacher forcing detrimental (found `tf_ratio=0.0` best). Testing `tf_ratio=1.0` (baseline) vs `tf_ratio=0.0` (SWP-aligned) in Stage A is now feasible. Intermediate ratios such as `0.2` are also available but increase compute ~10×.

---

## 2b. CRITICAL: Noise semantics with TF < 1 (Phase 1 audit finding, 2026-07-15)

> **This is a blocking issue for Phase 6 (noise grid) if TF<1 is selected.**

### What the current code does (TF<1 path)

In `train.py:_forward_scheduled_sampling` (lines 46-109), the training loop calls the full model at **each of the S decoder steps**:

```python
# train.py:78
out = model(enc_in, enc_mask, current_dec)   # full model re-run at each step
```

This means:
- WM encoder (`models/wm_route.py:47-54`) is called S times per batch.
- Each call generates a **new independent noise draw**: `h = h + torch.randn_like(h) * sigma`
- Result: **S different noise vectors** are applied to the WM hidden state across the S decode steps for the same stimulus.

**Intended semantics:** ONE noise draw per stimulus per training step (one corruption of the phonological buffer, consistent throughout the decoding of that item).

**Actual semantics with TF<1 + sigma>0:** S independent corruptions — the WM state is re-corrupted at each decoder step with a different noise sample.

This was documented in the code at the time of writing (`train.py:60-63`):
> "WM interference noise (if configured) is applied independently at each encoder call, producing different noise per step. This differs from the single-noise draw in the vectorised path. Document in gridsearch logs."

### Impact

| Configuration | Noise semantics | Correct? |
|---|---|---|
| TF=1.0, sigma>0 | 1 draw per batch item (vectorized path) | **YES** — intended |
| TF<1, sigma=0.0 | No noise | **YES** — no issue |
| TF<1, sigma>0 | S draws per batch item (step-by-step) | **NO** — bug |

### Phase impact

- **Phase 4 (H×TF×LR grid, sigma=0):** NOT AFFECTED. Phase 4 runs with `sigma_wm=0.0` — no noise drawn in any path.
- **Phase 6 (noise grid):** AFFECTED IF TF<1 IS SELECTED. If the Phase 5 winner has `TF<1`, the Phase 6 noise grid MUST NOT launch until the encode-once fix is applied.
- **Phase 5 (multi-seed):** NOT AFFECTED (same as Phase 4, sigma=0).

### Required fix: encode-once in TF<1 path

Refactor `train.py:_forward_scheduled_sampling` to:
1. Call WM encoder ONCE before the decode loop → get `h_WM` with ONE noise draw.
2. Call LTM encoder ONCE before the decode loop → get `s_hat` (no noise).
3. In the decode loop, pass pre-encoded states into the decoder directly, skipping re-encoding.

This requires separating the WM and LTM encode steps from the decode steps in the model forward interface. The fix is non-trivial (architecture-level refactor of how `model.forward()` is called) but must be done before Phase 6.

### Decision rule

```
After Phase 5: inspect best TF value.
If best TF = 1.0: encode-once fix NOT needed before Phase 6.
If best TF < 1.0: encode-once fix REQUIRED before Phase 6.
```

---

## 3. WM noise during training

### Noise-off training

```text
train_wm_noise_sigma = 0.0
```

Purpose: clean optimization and direct comparison to deterministic behavior.

### Current/default noise training

```text
train_wm_noise_sigma = 0.10
```

Purpose: preserve the current model's cognitive WM-interference mechanism if Yair wants it active during learning.

### Weak-noise training

A weak value such as `0.03` has appeared in evaluation sweeps, but it is not established as a training hyperparameter. Use only if the repo already supports it and Yair approves it. Otherwise, do not add it.

### Rule

Always record training-time noise separately from evaluation-time noise. Do not write simply `noise=true`; write:

```text
train_wm_noise_sigma = ...
eval_wm_noise_sigma = ...
collect = true/false
```

---

## 4. Evaluation regimes

### Eval Regime A - deterministic teacher-forced

Purpose: ceiling and debugging.

Definition:

```text
Decoder input at step t = gold phoneme at t-1.
WM noise OFF.
collect = False.
```

Use:

- train ceiling;
- per-item debugging;
- checking whether each route can in principle produce the target;
- appendix/diagnostic tables.

Do not use for main behavioral WFE figures.

### Eval Regime B - deterministic autoregressive

Purpose: primary behavioral comparison.

Definition:

```text
Decoder input at step t = model's own predicted phoneme at t-1.
WM noise OFF.
collect = False.
```

Use:

- WFE main real vs pseudoword analysis;
- validation model selection;
- length-effect slope;
- long-short contrast;
- route dissociation full/WM/LTM;
- serial-position curves.

### Eval Regime C - autoregressive + WM noise — CORRECTED AFTER REPO INSPECTION

Purpose: cognitive/noisy WM analysis and lesion-like robustness.

Definition:

```text
Decoder input at step t = model's own predicted phoneme at t-1.
WM noise ON.
collect = True (for WM route).
```

**Confirmed implementation status (from `scripts/external_eval.py`):**

Regime C IS implemented and works end-to-end for the **WM-isolated route** via:

```bash
python scripts/external_eval.py \
  --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt \
  --wfe_only \
  --decode autoregressive \
  --wm_noise
```

This sets `collect = (route == "wm") and wm_noise`, so:
- `route="wm"`: AR + noise ✓ — noise applied at each encoder call in the AR loop.
- `route="ltm"`: AR, no noise ✓ — LTM unaffected.
- `route="full"`: AR, but **WM noise NOT applied to the WM component inside the gated route** — `collect=False` for the full route.

**Important caveat:** `--wm_noise` in `external_eval.py` does NOT add noise to the WM component within the full/gated route. To add noise to both WM and the WM component inside full, use `run_wm_noise_sweep_wfe.py` which directly overrides `model.wm.cfg.interference_noise`.

Use:

- Length-effect/serial-position analysis under WM interference.
- Robustness plots (labelled as noisy/cognitive or lesion-like, not main behavioral result).
- Route compensation diagnostics.

Do not use for direct comparison to SWP/Dager unless noise conditions are explicitly matched and labelled.

---

## 5. Mandatory evaluation package for each gridsearch checkpoint

For each trained checkpoint:

```text
1. train_tf_no_noise
2. train_ar_no_noise
3. val_tf_no_noise
4. val_ar_no_noise
5. wfe_tf_no_noise
6. wfe_ar_no_noise
7. wfe_ar_no_noise_route_isolated_full_wm_ltm
8. length_effects_from_wfe_ar
9. optional wfe_ar_with_wm_noise, if Regime C implemented
10. optional ssp_ar_no_noise, after WFE passes
```

---

## 6. Metrics required for each checkpoint

### Selection metrics

```text
train_exact_match_ar
val_exact_match_ar
val_edit_distance_ar
val_normalized_edit_distance_ar
wfe_real_seen_exact_match_ar
wfe_pseudoword_exact_match_ar
wfe_pseudoword_edit_distance_ar
wfe_pseudoword_normalized_edit_distance_ar
length_effect_slope_pseudoword_edit_ar
long_short_contrast_pseudoword_edit_ar
```

### Diagnostic metrics

```text
train_exact_match_tf
val_exact_match_tf
wfe_exact_match_tf
route_full_exact_ar
route_wm_exact_ar
route_ltm_exact_ar
route_full_edit_ar
route_wm_edit_ar
route_ltm_edit_ar
mean_gate_real_seen
mean_gate_pseudoword
gate_distribution_by_length
wm_noise_sensitivity_if_available
serial_position_error_curve
error_type_counts
premature_eos_rate
```

### Figure candidates after final retrain

```text
wfe_real_seen_vs_pseudoword_accuracy_by_length_ar
wfe_pseudoword_edit_distance_by_length_ar
length_slope_full_wm_ltm_ar
serial_position_curve_ar
route_dissociation_full_wm_ltm_ar
wm_noise_effects_only_if_labelled
```

---

## 7. File and directory conventions

For every run:

```text
checkpoints/gridsearch/<stage>/<run_id>.pt
outputs/gridsearch/<stage>/<run_id>/metrics.json
outputs/gridsearch/<stage>/<run_id>/item_level_predictions.tsv
outputs/gridsearch/<stage>/<run_id>/length_effects.json
outputs/gridsearch/<stage>/<run_id>/route_metrics.json
outputs/gridsearch/<stage>/<run_id>/run_config.json
```

Run config must include:

```text
git_commit
branch
checkpoint_path
lexicon_path
max_words
train_split_seed
seed
optimizer
lr
batch_size
epochs
teacher_forcing_ratio
train_wm_noise_sigma
dropout
gate_alpha
semantic_alignment_loss_weight
loss_weights
selection_metric
```

---

## 8. Failure criteria

Mark a run as failed if:

1. deterministic evaluation is not deterministic across repeated calls;
2. train ceiling is far below the current baseline after the expected training budget;
3. validation AR collapses while TF remains high, indicating exposure-bias failure;
4. WFE pseudoword AR error rate or edit distance sharply worsens versus current baseline;
5. premature EOS/truncation dominates long-item errors;
6. all items route to LTM or WM with no meaningful gate variation, unless expected from a diagnostic;
7. checkpoint metadata is incomplete and the run cannot be reproduced;
8. WFE or SSP accidentally enters the training data.

---

## 9. Final full-lexicon retrain regime

After gridsearch selection:

1. Train final model on the 29,571 GloVe-covered lexicon.
2. Use selected LR, teacher-forcing ratio, training-time WM noise, gate alpha, and fixed loss weights.
3. Decide whether to keep an internal validation split for monitoring or train on all 29,571 words after selection. If training on all words, do not report held-out real-word validation as a final result.
4. Main external evaluation: WFE autoregressive no-noise.
5. Main WFE grouping: trained real words vs pseudowords.
6. Teacher-forced WFE: appendix/debug only.
7. WM-noise WFE: separate labelled cognitive/noisy analysis only.
8. SSP: secondary autoregressive diagnostic.
9. Save full run metadata and exact code commit.

---

## 9b. Evaluation scripts — status after code patch (2026-07)

| Regime | Status | Command |
|---|---|---|
| Train split TF eval (no noise) | ✓ AVAILABLE | `evaluate_train_lexicon_ceiling.py --decode teacher_forced` |
| Val split TF eval (no noise) | ✓ AVAILABLE | same + `--include_val` |
| Train split AR eval (no noise) | ✓ AVAILABLE ← PATCHED | `evaluate_train_lexicon_ceiling.py --decode autoregressive` |
| Val split AR eval (no noise) | ✓ AVAILABLE ← PATCHED | same + `--include_val` |
| WFE TF eval | ✓ AVAILABLE | `external_eval.py --decode teacher_forced --wfe_only` |
| WFE AR eval | ✓ AVAILABLE | `external_eval.py --decode autoregressive --wfe_only` |
| WFE AR + WM noise (WM-isolated) | ✓ AVAILABLE | `external_eval.py --decode autoregressive --wm_noise --wfe_only` |

AR outputs from `evaluate_train_lexicon_ceiling.py` go to `<out_dir>/ar/` to preserve backward compatibility of existing TF calls.

---

## 10. Minimal commands to verify before gridsearch

WFE AR (main behavioral baseline):

```bash
python scripts/external_eval.py \
  --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt \
  --out_dir outputs/gridsearch_baselines/current_ckpt \
  --decode autoregressive \
  --wfe_only
```

WFE TF (ceiling/debug):

```bash
python scripts/external_eval.py \
  --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt \
  --out_dir outputs/gridsearch_baselines/current_ckpt \
  --decode teacher_forced \
  --wfe_only
```

Train/val TF ceiling:

```bash
python scripts/evaluate_train_lexicon_ceiling.py \
  --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt \
  --lexicon_path data/lexicon_en_glove_covered.tsv \
  --out_dir outputs/gridsearch_baselines/current_ckpt/train_tf \
  --include_val
```

Train/val AR (primary gridsearch selection metric):

```bash
python scripts/evaluate_train_lexicon_ceiling.py \
  --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt \
  --lexicon_path data/lexicon_en_glove_covered.tsv \
  --out_dir outputs/gridsearch_baselines/current_ckpt/train_ar \
  --decode autoregressive \
  --include_val
# outputs to: outputs/gridsearch_baselines/current_ckpt/train_ar/ar/
```
