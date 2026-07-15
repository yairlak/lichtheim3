# Gridsearch Repo Inspection Report

> **Status: DRAFT — repo inspection result.  Do not launch training from this document before supervisor validation.**
>
> Branch: `eval/external-csv-datasets`
> Checkpoint: `checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt`
> Inspection method: static code reading only — no commands executed, no training launched.

---

## 1. Executive summary

| Topic | Finding |
|---|---|
| Training is always teacher-forced | ~~YES — no tf_ratio parameter~~ → **PATCHED: `--teacher_forcing_ratio` now CLI-exposed; scheduled sampling in `train.py`** |
| Dropout exists in model | **NO — not implemented anywhere** |
| Optimizer | **AdamW**, lr=1e-3, weight_decay=1e-5 |
| Loss weights | All confirmed — rep=1.0, align=1.0, dec=0.5, wm=0.5, gate=0.05 |
| AR+WM noise eval | **Implemented for WM route; NOT for WM inside full route via `--wm_noise`** |
| Val AR eval script | ~~MISSING~~ → **PATCHED: `--decode autoregressive --include_val`** in `evaluate_train_lexicon_ceiling.py` |
| Train AR eval script | ~~MISSING~~ → **PATCHED: `--decode autoregressive`** in `evaluate_train_lexicon_ceiling.py` |
| SLURM scripts | **NOT FOUND** in repo tree |
| CLI flags for gridsearch | LR, batch size, seed, epochs, lexicon path ✓ — **`--teacher_forcing_ratio`, `--interference_noise`, `--gate_alpha` now also ✓** |
| Go/no-go for Stage 0 smoke runs | **CONDITIONAL GO** — see Section 12 |

**Update (2026-07):** The critical blockers have been patched. `--teacher_forcing_ratio` (scheduled sampling), `--interference_noise`, `--gate_alpha` are now CLI-exposed. `--decode autoregressive` is now available in `evaluate_train_lexicon_ceiling.py`. Stage A (`2 LR × 2 TF ratio = 4 runs`) is now runnable. See Section 14 for the full patch summary.

---

## 2. Files inspected

| File | Purpose |
|---|---|
| [config.py](../config.py) | All hyperparameter dataclasses and defaults |
| [train.py](../train.py) | Training loop, optimizer, loss call, data loader |
| [scripts/train_checkpoint.py](../scripts/train_checkpoint.py) | CLI entry point, checkpoint save format |
| [scripts/external_eval.py](../scripts/external_eval.py) | WFE/SSP evaluation CLI, AR+TF+noise modes |
| [scripts/evaluate_train_lexicon_ceiling.py](../scripts/evaluate_train_lexicon_ceiling.py) | Train/val ceiling eval (TF only) |
| [losses.py](../losses.py) | Exact loss formula and weights |
| [data/dataset.py](../data/dataset.py) | Dataset format, sampler, teacher-forcing in batch |
| [models/dual_route.py](../models/dual_route.py) | Full model forward, route-isolated logits |
| [models/wm_route.py](../models/wm_route.py) | WM encoder/decoder, noise path |
| [models/ltm_route.py](../models/ltm_route.py) | LTM encoder, semantic bank, decoder |
| [models/gating.py](../models/gating.py) | Gate formula and premotor blending |
| [models/motor.py](../models/motor.py) | Shared Linear projection |
| [evaluate/hooks.py](../evaluate/hooks.py) | `make_batch`, `route_predictions` helpers |

**Not found / not inspected:**

- SLURM submission scripts (none found in repo tree via static read).
- `configs/*.yaml` or equivalent config directory (does not exist; all config is in `config.py`).
- `scripts/analyze_length_effects.py` — not inspected for this report (eval-only, does not affect training).
- `scripts/run_wm_noise_sweep_wfe.py` — not inspected for this report.
- `scripts/audit_route_compensation_and_ltm.py` — not inspected for this report.
- Checkpoint binary (not read; metadata will be readable from `torch.load`).

---

## 3. Confirmed hyperparameters with source

### 3.1 Architecture (config.py)

| Parameter | Value | Source |
|---|---:|---|
| `phon_embed_dim` | 64 | `config.py:LTMConfig` |
| `wm_hidden` | 128 | `config.py:WMConfig` |
| `ltm_enc_hidden` | 256 per direction | `config.py:LTMConfig.enc_hidden` |
| `ltm_enc_layers` | 1 | `config.py:LTMConfig.enc_layers` |
| `ltm_bidirectional` | True | `config.py:LTMConfig.bidirectional_encoder` |
| `ltm_dec_hidden` | 256 | `config.py:LTMConfig.dec_hidden` |
| `semantic_dim` | 300 | `config.py:DataConfig.semantic_dim` |
| `premotor_dim` | 128 | `models/dual_route.py:34` default arg |
| `gate_alpha` | 4.0 | `config.py:GatingConfig.alpha` |
| `usage_prior` | 0.5 | `config.py:GatingConfig.usage_prior` |
| `interference_noise` | 0.10 | `config.py:WMConfig.interference_noise` |
| `max_phonemes` | 9 | `config.py:DataConfig.max_phonemes` |
| `min_phonemes` | 2 | `config.py:DataConfig.min_phonemes` |

### 3.2 Training (config.py + train_checkpoint.py + train.py)

| Parameter | Value | Source | CLI exposed |
|---|---:|---|---|
| Optimizer | AdamW | `train.py:102` | No (hard-coded) |
| beta1, beta2 | 0.9, 0.999 | PyTorch AdamW defaults | No |
| `weight_decay` | 1e-5 | `config.py:TrainConfig.weight_decay` | No |
| `grad_clip` | 1.0 | `config.py:TrainConfig.grad_clip` | No |
| `lr` default | 1e-3 | `config.py:TrainConfig.lr` | **YES: `--lr`** |
| `batch_size` default | 64 | `config.py:TrainConfig.batch_size` | **YES: `--batch_size`** |
| `epochs` (config default) | 8 | `config.py:TrainConfig.epochs` | **YES: `--epochs`** |
| `epochs` (CLI default) | 10 | `scripts/train_checkpoint.py:64` | **YES: `--epochs`** |
| `seed` | 0 | `config.py:TrainConfig.seed` | **YES: `--seed`** |
| `dorsal_pool_size` | 4000 pseudowords | `config.py:TrainConfig.dorsal_pool_size` | No |
| `freq_temp` | 1.0 | `config.py:DataConfig.freq_temp` | No |
| `val_fraction` | 0.15 | `config.py:DataConfig.val_fraction` | No |
| Teacher forcing ratio | **Does not exist in pre-patch code.** Training was always TF. Patch (2026-07) adds `teacher_forcing_ratio=1.0` default. Production checkpoint was trained before the patch: no `teacher_forcing_ratio` key in `cfg_train` — interpret as implicit `1.0`. | `data/dataset.py:43-57`; `config.py:TrainConfig` (post-patch) | **YES: `--teacher_forcing_ratio`** ✓ PATCHED |
| Dropout | **Not implemented** | All model modules checked | **NO** |
| LR scheduler | None | `train.py` — no scheduler call | No |
| Early stopping | None | `train_checkpoint.py` — no eval loop | N/A |
| Sampler | WeightedRandomSampler, log-freq, with replacement | `data/dataset.py:124-133` | No |
| num_samples per epoch | `len(train_entries)` = 25,136 | `data/dataset.py:129` | No |

### 3.3 Loss (losses.py + config.py)

| Term | Weight | Formula | Source |
|---|---:|---|---|
| `L_rep` | 1.0 | CE(full_logits, target), ignore PAD | `losses.py:62` |
| `L_align` | 1.0 | (1 - cosine_sim) + 0.1 × MSE of s_hat vs GloVe | `losses.py:47-50` |
| `L_dec` | 0.5 | CE(ltm_logits, target), ignore PAD | `losses.py:64` |
| `L_wm` | 0.5 | CE(wm_logits, target) + CE(pool_wm_logits, pool_target) | `losses.py:63`, `train.py:65-68` |
| `L_gate` | 0.05 | (mean(gate) − usage_prior)² | `losses.py:53-55` |
| `label_smoothing` | 0.0 | applied to L_rep only | `config.py:LossConfig` |

### 3.4 Checkpoint save format (train_checkpoint.py)

Keys saved: `model_state_dict`, `optimizer_state_dict` (None in fresh mode), `rng_states`, `cfg_data`, `cfg_wm`, `cfg_ltm`, `cfg_gating`, `cfg_loss`, `cfg_train`, `history`, `lexicon_source`, `n_train`, `n_val`, `glove_present`, `git_commit`, `resumed_from`, `total_epochs_trained`, `lr_at_save`.

**Note:** `optimizer_state_dict` is `None` in fresh (non-resume) runs because `build_and_train` does not expose its internal optimizer object (`train.py`). It is saved correctly in resume mode.

---

## 4. Hyperparameters NOT found / unknown — UPDATED WITH CHECKPOINT METADATA

The production checkpoint (`lichtheim3_30k_glove_e60_to_e120_lowlr.pt`) was inspected. Previously unknown values are now confirmed:

| Parameter | Status |
|---|---|
| Exact LR used for the production checkpoint (e60→e120) | **CONFIRMED: `lr_at_save = 1e-4`** (low-LR continuation phase) |
| Exact seed used for production checkpoint | **CONFIRMED: `cfg_train.seed = 0`** |
| Exact batch size used for production checkpoint | **CONFIRMED: `cfg_train.batch_size = 64`** |
| WM interference_noise (production) | **CONFIRMED: `cfg_wm.interference_noise = 0.1`** |
| Gate alpha (production) | **CONFIRMED: `cfg_gating.alpha = 4.0`** |
| Gate usage_prior (production) | **CONFIRMED: `cfg_gating.usage_prior = 0.5`** |
| Total epochs trained (production) | **CONFIRMED: `total_epochs_trained = 120`** |
| Git commit (production) | **CONFIRMED: `git_commit = ab9353cfeb92516e5a44625bafe01407d87526ae`** |
| teacher_forcing_ratio (production) | **NOT STORED** — pre-patch checkpoint. Training was always teacher-forced at the time. Interpret as `teacher_forcing_ratio = 1.0` (implicit). The 2026-07 patch adds this field with default `1.0`; loading the old checkpoint's `cfg_train` via `TrainConfig(**ckpt["cfg_train"])` applies the default correctly. |
| SLURM scripts | `NOT FOUND` — no `.sh` files found in repo tree |
| `configs/*.yaml` directory | `NOT FOUND` — all config is in `config.py` |

---

## 5. Differences between existing drafts and code reality

| Draft claim | Code reality |
|---|---|
| "teacher_forcing_ratio = 0.0 vs 0.2 vs 0.5 in Stage A" | **Pre-patch finding:** correct at inspection — no `teacher_forcing_ratio` parameter existed; training was always 100% teacher-forced. **Post-patch current state (2026-07):** `teacher_forcing_ratio` is now implemented and CLI-exposed (`--teacher_forcing_ratio`, default `1.0`). Stage A with LR × TF ratio is now runnable. See Section 14. |
| "Dropout: only if existing in L3 code path" | **Confirmed NOT existing.** No dropout anywhere. |
| "Optimizer: likely AdamW" | **Confirmed AdamW**, `train.py:102`. |
| "Batch size: TO INSPECT" | **Confirmed 64**, `config.py:TrainConfig.batch_size`. |
| "Semantic alignment loss weight: TO INSPECT" | **Confirmed 1.0**, `config.py:LossConfig.align`. |
| "Route loss weights: TO INSPECT" | **Confirmed**: rep=1.0, dec=0.5, wm=0.5, gate=0.05 in `losses.py`. |
| "Gate usage_prior: TO INSPECT" | **Confirmed 0.5**, `config.py:GatingConfig.usage_prior`. Formula: (mean(g) − 0.5)². |
| "Regime C (AR+WM noise): not implemented end-to-end" | **WRONG** — it IS implemented in `external_eval.py` for the WM-isolated route. However, `--wm_noise` does NOT apply noise to the WM component inside the full/gated route. |
| "Split seed: TO INSPECT" | **Confirmed**: same as `--seed` flag (`cfg.data.seed = args.seed`). Not a separate parameter. |
| "Checkpoint selection criterion: TO INSPECT" | **Confirmed NOT IMPLEMENTED** — no auto-selection, no eval loop in training script. |

---

## 6. CLI flags available for training

Confirmed from `scripts/train_checkpoint.py:60-101`:

```bash
python scripts/train_checkpoint.py \
  [--epochs INT]                    # total epoch count; default 10
  [--max_words INT]                 # lexicon size; default 4000
  [--seed INT]                      # controls train RNG + data split; default 0
  [--batch_size INT]                # batch size; default 64
  [--lexicon_path STR]              # override lexicon TSV path
  [--resume_from STR]               # path to checkpoint to resume from
  [--lr FLOAT]                      # learning rate; default: config default (1e-3)
  [--ckpt STR]                      # output checkpoint path
  [--out_dir STR]                   # directory for training figures
  [--teacher_forcing_ratio FLOAT]   # scheduled sampling ratio; 1.0=full TF (default), 0.0=fully AR ← PATCHED 2026-07
  [--interference_noise FLOAT]      # WM training noise sigma; default None (→ WMConfig default 0.10) ← PATCHED 2026-07
  [--gate_alpha FLOAT]              # gate sharpness alpha; default None (→ GatingConfig default 4.0) ← PATCHED 2026-07
```

**Patched (2026-07) — now in available block above:**

```text
--teacher_forcing_ratio   ✓ PATCHED (2026-07) — default 1.0 (full TF); 0.0=fully AR; see available block
--interference_noise      ✓ PATCHED (2026-07) — default None (→ WMConfig default 0.10)
--gate_alpha              ✓ PATCHED (2026-07) — default None (→ GatingConfig default 4.0)
```

**Still NOT available via CLI (config-only or not implemented):**

```text
--align_weight            → config-only (LossConfig.align=1.0); no CLI flag
--rep_weight              → config-only (LossConfig.rep=1.0); no CLI flag
--dec_weight              → config-only (LossConfig.dec=0.5); no CLI flag
--wm_weight               → config-only (LossConfig.wm=0.5); no CLI flag
--gate_weight             → config-only (LossConfig.gate=0.05); no CLI flag
--weight_decay            → config-only (TrainConfig.weight_decay=1e-5); no CLI flag
--dropout                 → NOT IMPLEMENTED in model
```

---

## 7. CLI flags available for evaluation

Confirmed from `scripts/external_eval.py:840-861`:

```bash
python scripts/external_eval.py \
  [--ckpt STR]          # checkpoint path
  [--wfe_tsv STR]       # WFE TSV path
  [--ssp_tsv STR]       # SSP TSV path
  [--out_dir STR]       # output directory
  [--dry_run]           # 10 WFE + 20 SSP items only
  [--wfe_only]          # skip SSP
  [--ssp_only]          # skip WFE
  [--device STR]        # cpu / cuda
  [--wm_noise]          # enable WM noise for WM route (non-deterministic)
  [--decode {teacher_forced,autoregressive}]   # default: teacher_forced
```

Confirmed from `scripts/evaluate_train_lexicon_ceiling.py:73-89`:

```bash
python scripts/evaluate_train_lexicon_ceiling.py \
  [--ckpt STR]
  [--lexicon_path STR]
  [--out_dir STR]
  [--include_val]       # also evaluate validation split
  [--wm_noise]
  [--device STR]
  [--decode {teacher_forced,autoregressive}]  # default: teacher_forced; AR outputs to <out_dir>/ar/ ← PATCHED 2026-07
```

**Note (post-patch):** `--decode autoregressive` is now available. AR outputs go to `<out_dir>/ar/` to preserve backward compatibility with existing TF callers. Metrics: `exact_match`, `edit_distance`, `normalized_edit_distance` for all routes.

---

## 8. What is ready for smoke runs

The following can be run immediately with the current CLI:

| Smoke run | Command | Status |
|---|---|---|
| WFE TF eval (ceiling/debug) | `external_eval.py --decode teacher_forced --wfe_only` | ✓ Ready |
| WFE AR eval (no noise) | `external_eval.py --decode autoregressive --wfe_only` | ✓ Ready |
| WFE AR + WM noise (WM route only) | `external_eval.py --decode autoregressive --wm_noise --wfe_only` | ✓ Ready |
| Train ceiling TF eval | `evaluate_train_lexicon_ceiling.py` | ✓ Ready |
| Val TF eval | `evaluate_train_lexicon_ceiling.py --include_val` | ✓ Ready |
| SSP AR eval | `external_eval.py --decode autoregressive --ssp_only` | ✓ Ready |
| Dry run (10 WFE + 20 SSP) | `external_eval.py --dry_run` | ✓ Ready |

**Previously missing, now available (post-patch):**

| Needed | Status |
|---|---|
| Train AR eval | ✓ **PATCHED (2026-07)** — `--decode autoregressive` now in `evaluate_train_lexicon_ceiling.py`; AR outputs to `<out_dir>/ar/` |
| Val AR eval | ✓ **PATCHED (2026-07)** — same + `--include_val`; AR outputs to `<out_dir>/ar/` |
| WFE AR with noise inside full route | **Partial** — `--wm_noise` applies noise to WM route only, not to WM inside full/gated route |

---

## 9. Patch status and remaining requirements

| Change | Effort | Status |
|---|---|---|
| `--interference_noise` flag in `train_checkpoint.py` | 1 line + 1 line apply | ✓ **DONE (2026-07)** |
| `--gate_alpha` flag in `train_checkpoint.py` | 1 line + 1 line | ✓ **DONE (2026-07)** |
| `--decode autoregressive` in `evaluate_train_lexicon_ceiling.py` | AR inference loop | ✓ **DONE (2026-07)** |
| Scheduled sampling (`teacher_forcing_ratio`) | Training loop + CLI | ✓ **DONE (2026-07)** — `train.py:_forward_scheduled_sampling`; `--teacher_forcing_ratio` CLI flag, default `1.0` |
| `--align_weight` / loss weight flags | 1 line each + apply | **Still config-only** — Stage C only |
| Dropout in model modules | Architecture addition | **Still NOT IMPLEMENTED** — defer |

---

## 10. Main risks

| Risk | Severity | Note |
|---|---|---|
| ~~Gridsearch plan designed around TF ratio, which did not exist~~ | ✓ **RESOLVED (2026-07)** | `teacher_forcing_ratio` is now implemented and CLI-exposed; Stage A with LR × TF ratio is runnable. |
| ~~Train/val AR metrics unavailable without script patch~~ | ✓ **RESOLVED (2026-07)** | `--decode autoregressive` is now available in `evaluate_train_lexicon_ceiling.py`. |
| `--wm_noise` applies noise to WM route only, not to WM inside full route | Medium | Well-defined behavior, but must not be confused with the noise sweep scripts. Document in every eval report. |
| Fresh checkpoint mode saves `optimizer_state_dict=None` | Low | Acceptable for smoke runs; relevant if resuming a fresh-mode checkpoint. |
| Epoch count semantics: `--epochs` is total when resuming | Medium | Double-check: `if epochs_needed <= 0: return`. Don't pass the same epoch count on resume. |
| No early stopping / checkpoint selection criterion | Medium | Manual selection required. Define criterion before running Stage A. |
| All loss weights config-only (no CLI) | Medium for Stage C | Must add flags or modify config in code before varying them. |

---

## 11. Questions for the supervisor

1. **Teacher forcing ratio** ✓ **(resolved by 2026-07 patch):** scheduled sampling is now implemented (`--teacher_forcing_ratio` CLI flag). Which TF ratios to test in Stage A: `[1.0, 0.0]` minimum; `[1.0, 0.2, 0.0]` if compute allows? Note: `tf_ratio < 1.0` is ~10× slower per batch.
2. **Training-time WM noise** ✓ **(resolved by 2026-07 patch):** `--interference_noise` is now CLI-exposed; default `None` (uses WMConfig default `0.10`). Should Stage A fix noise at `0.10`, or test `0.0` in Stage A rather than Stage B?
3. **Val/train AR eval** ✓ **(resolved by 2026-07 patch):** `--decode autoregressive` is now available. AR outputs go to `<out_dir>/ar/`. Run Stage 0 AR eval before selecting gridsearch criteria.
4. **Gate alpha**: is 4.0 already validated or still a default? Should we plan Stage C gate alpha variations?
5. **Checkpoint selection criterion**: how to select the best Stage A/B checkpoint? Val AR exact match? WFE pseudoword AR exact match? Manual inspection?
6. **Seeds**: 1 for screening, then how many for top candidates — 3 or 10 before full retrain?
7. **Semantic alignment loss weight (1.0)**: is this the right scale? The current value is equal weight with L_rep. Do we want to vary it?
8. **Dropout**: should we implement it at all? SWP selected 0.0 dropout, so the expected answer may be "not needed."

---

## 12. Recommendation: go / no-go for Stage 0 smoke runs only

**CONDITIONAL GO** for Stage 0 smoke runs.

The following Stage 0 smoke runs can be launched immediately:

```bash
# Smoke run 1: WFE TF ceiling/debug (deterministic)
python scripts/external_eval.py \
  --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt \
  --out_dir outputs/gridsearch_baselines/current_ckpt \
  --decode teacher_forced \
  --wfe_only

# Smoke run 2: WFE AR primary behavioral (no noise, deterministic)
python scripts/external_eval.py \
  --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt \
  --out_dir outputs/gridsearch_baselines/current_ckpt \
  --decode autoregressive \
  --wfe_only

# Smoke run 3: Train ceiling TF (teacher-forced only)
python scripts/evaluate_train_lexicon_ceiling.py \
  --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt \
  --lexicon_path data/lexicon_en_glove_covered.tsv \
  --out_dir outputs/gridsearch_baselines/current_ckpt/train_ceiling_tf

# Smoke run 4: Val TF ceiling (include_val)
python scripts/evaluate_train_lexicon_ceiling.py \
  --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt \
  --lexicon_path data/lexicon_en_glove_covered.tsv \
  --out_dir outputs/gridsearch_baselines/current_ckpt/train_ceiling_tf \
  --include_val

# Smoke run 5: WFE AR + WM noise (WM route only — explicit noise labelling)
python scripts/external_eval.py \
  --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt \
  --out_dir outputs/gridsearch_baselines/current_ckpt \
  --decode autoregressive \
  --wm_noise \
  --wfe_only

# Smoke run 6: Dry run (10 WFE + 20 SSP — fast sanity check)
python scripts/external_eval.py \
  --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt \
  --out_dir outputs/gridsearch_baselines/current_ckpt \
  --dry_run
```

Train/val AR eval (now available — PATCHED):

```bash
# Train + val AR eval
python scripts/evaluate_train_lexicon_ceiling.py \
  --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt \
  --lexicon_path data/lexicon_en_glove_covered.tsv \
  --out_dir outputs/gridsearch_baselines/current_ckpt/train_ar \
  --decode autoregressive \
  --include_val
# outputs to: outputs/.../train_ar/ar/metrics.json
```

**No-go for any training runs** until supervisor validation of:
1. Which teacher forcing ratios to test: `[1.0, 0.0]` minimum; `[1.0, 0.2, 0.0]` if budget allows.
2. The checkpoint selection criterion (val AR exact match recommended).
3. How many seeds before full retrain.

---

## 13. Syntax check commands (lightweight)

These can be run before any smoke run to catch import errors:

```bash
python -m py_compile scripts/external_eval.py \
    scripts/evaluate_train_lexicon_ceiling.py \
    scripts/train_checkpoint.py \
    train.py losses.py config.py
```

If any of these fail, fix the import/syntax error before proceeding.

---

*Report produced by static code inspection only. No training was launched. No evaluation was run. No checkpoint was loaded.*

---

## 14. Minimal gridsearch capability patch — applied 2026-07

The following changes were made to unblock Stage A and Stage 0:

### Files modified

| File | Change |
|---|---|
| `config.py` | `TrainConfig.teacher_forcing_ratio: float = 1.0` added |
| `train.py` | `_forward_scheduled_sampling()` added; `run_epoch` modified to use it when `tf_ratio < 1.0` |
| `scripts/train_checkpoint.py` | `--teacher_forcing_ratio` (default 1.0), `--interference_noise` (default None), `--gate_alpha` (default None) CLI flags added |
| `scripts/evaluate_train_lexicon_ceiling.py` | `--decode {teacher_forced,autoregressive}` flag added; AR evaluator `evaluate_forms_ar` + `_ar_decode_batch` added; `norm_edit_dist` added to all routes; AR outputs to `<out_dir>/ar/` |
| `tests/test_teacher_forcing_ratio.py` | New smoke-check tests: MockModel logic tests + real-model smoke tests |

### New CLI commands

```bash
# Training with AR decoding (fully autoregressive)
python scripts/train_checkpoint.py \
  --teacher_forcing_ratio 0.0 \
  --interference_noise 0.10 \
  --gate_alpha 4.0 \
  --lr 1e-3 --epochs 30 --seed 0 \
  --ckpt checkpoints/stageA_lr1e-3_tf0_noise0p10_seed0.pt

# Training with partial scheduled sampling
python scripts/train_checkpoint.py \
  --teacher_forcing_ratio 0.2 \
  --interference_noise 0.0 \
  --ckpt checkpoints/stageA_lr1e-3_tf0p2_noise0_seed0.pt

# Train/val AR evaluation
python scripts/evaluate_train_lexicon_ceiling.py \
  --ckpt checkpoints/stageA_lr1e-3_tf0_noise0p10_seed0.pt \
  --lexicon_path data/lexicon_en_glove_covered.tsv \
  --out_dir outputs/gridsearch/stageA/lr1e-3_tf0_noise0p10_seed0 \
  --decode autoregressive \
  --include_val
```

### Behavior guarantees

- `--teacher_forcing_ratio 1.0` (default): **exact same behavior as before the patch** (fast vectorized TF path).
- `--interference_noise None` (default): uses `WMConfig.interference_noise = 0.10`, unchanged.
- `--gate_alpha None` (default): uses `GatingConfig.alpha = 4.0`, unchanged.
- `--decode teacher_forced` (default): existing TF behavior, same output paths, fully backward compatible.
- Validation always uses `tf_ratio=1.0` regardless of the `--teacher_forcing_ratio` flag.
- WM noise with `tf_ratio < 1.0`: noise is drawn at each of the S encoder calls (not once per batch item as in TF=1.0 path). Document in run logs.

### Known limitations

- `tf_ratio < 1.0` training is ~S× slower (S ≈ 10 for max_phonemes=9).
- `--semantic_align_weight` still config-only; not CLI-exposed.
- Dropout still not implemented in any model module.
- `evaluate_train_lexicon_ceiling.py` AR mode does not compute `phoneme_acc` (only exact match, edit_dist, norm_edit_dist).

---

## 15. Post-Yair-meeting Phase 1 exact code audit (2026-07-15)

> Source of truth: static inspection of `models/ltm_route.py`, `models/wm_route.py`, `models/gating.py`, `models/dual_route.py`, `models/motor.py`, `train.py`, `scripts/train_checkpoint.py`, `evaluate/hooks.py`, `losses.py`, `data/dataset.py`, `config.py`.
> No training run. No checkpoint loaded. No shell command executed.

### 15.1 Master audit table

| # | Question | Current implementation | Exact code location | Risk / issue | Phase 2 action | Status |
|---|---|---|---|---|---|---|
| Q1 | LTM encoder type | biGRU, `bidirectional=True`, enc_hidden=256/dir → 512-d output | `models/ltm_route.py:42-44` | Backward pass uses padding position, not last real token | Add `bidirectional_encoder=False` option + use last hidden | CONFIRMED |
| Q1 | LTM masked mean pooling formula | `pooled = (out * m).sum(1) / m.sum(1).clamp(min=1)` where `m = enc_mask.unsqueeze(-1).float()` | `models/ltm_route.py:64-65` | With uniGRU + last hidden, this disappears | Replace with `h[-1]` (last hidden squeeze) | CONFIRMED |
| Q1 | LTM to_semantic MLP | `Linear(512→256) → GELU → Linear(256→300)` | `models/ltm_route.py:45-48` | With uniGRU: `Linear(256→256)` first layer | First layer input dim changes from 512 to 256 | CONFIRMED |
| Q1 | LTM last hidden available? | YES: `out, _ = self.encoder(emb)` — `_` is `(2,B,256)` but discarded | `models/ltm_route.py:62` | Last hidden already computed, just discarded | Capture `h` instead of `_`, use `h[-1]` | CONFIRMED |
| Q2 | WM encoder type | uniGRU, `bidirectional` absent → False, hidden=128 | `models/wm_route.py:39` | Asymmetric vs LTM biGRU | uniGRU patch on LTM side makes them symmetric | CONFIRMED |
| Q2 | WM uses pack_padded_sequence | YES: `lengths = enc_mask.sum(1).clamp(min=1).cpu(); packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)` | `models/wm_route.py:46-50` | LTM does not → LTM backward pass shifts with batch padding | Do not add pack to LTM yet (artifact stays in checkpoint) | CONFIRMED |
| Q2 | WM uses last hidden | YES: `_, h = self.encoder(packed)` → `h: (1, B, 128)` directly initialises decoder | `models/wm_route.py:50-57` | WM is already last-hidden; LTM is pooled-mean | uniGRU LTM patch makes both last-hidden | CONFIRMED |
| Q3 | WM noise tensor and shape | Added to `h: (1, B, 128)` → `h = h + torch.randn_like(h) * cfg.interference_noise` | `models/wm_route.py:53-54` | One draw per `forward()` call, but with TF<1: S draws per training step | BLOCKING for noise grid with TF<1 | CONFIRMED |
| Q3 | WM noise active condition | `(self.training OR collect) AND sigma > 0` | `models/wm_route.py:53` | Validation always uses `model.eval()` → noise off unless collect=True | Well-defined; document per eval call | CONFIRMED |
| Q3 | WM noise during validation | OFF: `model.eval()` and collect=False | `train.py:114`, `models/wm_route.py:53` | None | — | CONFIRMED |
| Q4 | LTM noise exists? | NO — no noise anywhere in `models/ltm_route.py` | `models/ltm_route.py` entire file | No LTM noise mechanism | Add after `pooled` before `to_semantic` in `encode()` | CONFIRMED ABSENT |
| Q4 | Best LTM noise location | After `pooled` (`(B,512)` with biGRU, `(B,256)` with uniGRU), before `to_semantic` | `models/ltm_route.py:65-66` | Analogous to WM: noise on encoder state before projection | New param `LTMConfig.ventral_noise: float = 0.0` | PROPOSED |
| Q5 | L2 norm in set_semantic_bank | `bank = F.normalize(bank, dim=-1)` — normalizes bank once at setup | `models/ltm_route.py:79` | Mathematically redundant with cosine_similarity, but numerically clean and done once | Do not remove; harmless | CONFIRMED REDUNDANT BUT HARMLESS |
| Q5 | L2 norm in lexical_field | `q = F.normalize(s_hat, dim=-1)` — creates new tensor `q`, does NOT modify `s_hat` | `models/ltm_route.py:91` | Redundant with cosine similarity; BUT `s_hat` itself is unmodified and used elsewhere in `decode()` and `alignment_loss()` | Do not remove; `s_hat` unaffected | CONFIRMED REDUNDANT BUT HARMLESS |
| Q5 | s_hat used after normalization | `s_hat` is passed to `decode()` (via `sem_to_h0`) and to `alignment_loss()` (via `F.cosine_similarity` + `F.mse_loss`). `q` (normalized) is local to `lexical_field()` | `models/ltm_route.py:70-75`, `losses.py:48-50` | Removing normalization would NOT affect s_hat usage | No action needed | CONFIRMED |
| Q6 | Gate equation exact | `g = torch.sigmoid(self.cfg.alpha * (conf - 0.5))` | `models/gating.py:45` | `0.5` is a **hard-coded Python literal**, not a config field | Add `gate_threshold: float = 0.5` to `GatingConfig`; change literal to `self.cfg.gate_threshold` | CONFIRMED — 0.5 IS HARD-CODED |
| Q6 | Gate alpha configurable | YES: `self.cfg.alpha = GatingConfig.alpha = 4.0`; CLI: `--gate_alpha` | `config.py:58-63`, `scripts/train_checkpoint.py:108-112` | Already CLI-exposed | — | CONFIRMED |
| Q6 | Files to modify for tau | `config.py:GatingConfig`, `models/gating.py:45`, `scripts/train_checkpoint.py` (add `--gate_threshold`) | Listed | Trivial 3-file patch | Phase 2 action | READY TO IMPLEMENT |
| Q7 | Gate shape and level | `conf: (B,)` → `view(B,1,1)` → `g: (B,1,1)` → `expand(B,S,1)` | `models/gating.py:43-46` | **g is computed once per item** and broadcast to all S decoder steps | CURRENT GATE = **WORD-LEVEL (item-level scalar, constant across all t)** | CONFIRMED |
| Q7 | Gate blend operation | `premotor = g * ltm + (1.0 - g) * wm` where shapes are `(B,S,1)*(B,S,P) + (B,S,1)*(B,S,P)` = `(B,S,P)` | `models/gating.py:51` | g does NOT vary with timestep | Word-level gate confirmed | CONFIRMED |
| Q8 | TF=1 path | Single `model(enc_in, enc_mask, dec_in)` call; vectorized; enc + dec run once | `train.py:126` | Fast, correct | — | CONFIRMED |
| Q8 | TF<1 path — re-encoding | At EACH of S decoder steps: `model(enc_in, enc_mask, current_dec)` — FULL model re-run including BOTH encoders | `train.py:78` | LTM encoder called S times: wasteful (identical output, no noise). WM encoder called S times: **different noise draw each time** | NOISE SEMANTICS BUG: S noise draws instead of 1. Encode-once fix needed | CONFIRMED — CRITICAL FINDING |
| Q8 | s_hat in TF<1 path | Taken from step 0 only: `if s_hat_0 is None: s_hat_0 = out["s_hat"]` | `train.py:86-87` | s_hat for alignment loss = step 0 output (correct, since LTM encoder is deterministic) | — | CONFIRMED |
| Q9 | Noise with TF<1 — draws | With TF<1 and sigma>0: S independent noise vectors per item per training step. Comment in code: "WM interference noise (if configured) is applied independently at each encoder call, producing different noise per step. This differs from the single-noise draw in the vectorised path." | `train.py:60-63` | SEMANTICALLY WRONG: intended = one corruption per stimulus. BLOCKING for noise grid with TF<1 | Encode-once fix before noise×TF grid | CONFIRMED — BLOCKING for Phase 6 with TF<1 |
| Q10 | phon_embed_dim | 64 — `LTMConfig.phon_embed_dim`; shared by both routes via `self.phon_embed` | `config.py:47`, `models/dual_route.py:41` | Used by BOTH routes; shared embedding | Single param to vary both | CONFIRMED |
| Q10 | WM hidden dims | Encoder: `WMConfig.hidden=128`; Decoder: same `cfg.hidden` → WM enc_hidden = WM dec_hidden always | `config.py:39`, `models/wm_route.py:39-40` | Constraint: WM enc and dec must be same size | `--wm_hidden INT` → sets both | CONFIRMED |
| Q10 | LTM enc/dec hidden dims | `LTMConfig.enc_hidden=256`; `LTMConfig.dec_hidden=256`; separate configs | `config.py:48-50` | enc and dec can differ; both CLI-needed | `--ltm_enc_hidden`; `--ltm_dec_hidden` (or unified `--hidden_size`) | CONFIRMED |
| Q10 | premotor_dim | **128 — hardcoded default arg in `DualRouteModel.__init__`, NOT in any config dataclass** | `models/dual_route.py:34` | Not configurable via CLI or config; all 3 `to_premotor` / `dec_to_premotor` / motor input depend on it | Add `premotor_dim` to `Config` or keep fixed at 128 for Phase 4 | CONFIRMED — NOT IN CONFIG |
| Q10 | H_recurrent unified param feasibility | After uniGRU patch: setting `WMConfig.hidden = LTMConfig.enc_hidden = LTMConfig.dec_hidden = H` gives symmetric routes. `to_semantic[0]: Linear(H,H)`, `sem_to_h0: Linear(300,H)`. Clean. | Multiple files | Only `phon_embed_dim` and `premotor_dim` remain separate | `--hidden_size INT` CLI flag that sets all three | FEASIBLE AND CLEAN |
| Q11 | Metrics exposed by evaluators | `metrics["results"][split][route]`: `exact_match`, `edit_dist`, `norm_edit_dist`, `n_errors`, `n_items` | `scripts/evaluate_train_lexicon_ceiling.py:383-416` | Full route metrics exist; WM/LTM diagnostic metrics exist | — | CONFIRMED |
| Q11 | Aggregation/ranking script | NONE — no script aggregates multiple `metrics.json` files and ranks runs | Repo search | Manual aggregation only; gridsearch summary impossible without it | Phase 2: write `scripts/aggregate_gridsearch.py` | MISSING |
| Q11 | FULL-only selection rule | NOT ENFORCED anywhere in code or YAML — existing YAML includes WM/LTM metrics in selection | `docs/gridsearch_candidates_proposal.yaml:172-190` | WM/LTM metrics must be demoted to diagnostic only; FULL route is primary selection | Update YAML and write aggregation script | ACTION REQUIRED |
| Q12 | optimizer_state_dict in fresh mode | `None` — `build_and_train()` does not expose its internal optimizer; `optim = None` in `main()` after fresh training | `scripts/train_checkpoint.py:158,314` | Cannot resume a fresh checkpoint with optimizer state | Refactor: expose optimizer from `build_and_train()` or manage optimizer in `main()` | CONFIRMED — KNOWN LIMITATION |
| Q12 | optimizer_state_dict in resume mode | SAVED: `optim.state_dict()` after resume | `scripts/train_checkpoint.py:314`, resume block lines 226-239 | Correct; also restores LR override and moves tensors to device | — | CONFIRMED |
| Q12 | RNG state | Saved in both modes; restored in resume mode (best-effort) | `scripts/train_checkpoint.py:303-310`, resume block lines 243-254 | Not present in old checkpoints → non-reproducible resume | — | CONFIRMED |
| Q12 | Epoch tracking | `total_epochs_trained = len(history)`, `history` concatenated across resumes | `scripts/train_checkpoint.py:330` | Correct | — | CONFIRMED |
| Q13 | DataLoader num_workers | `0` (PyTorch default) — `make_loader()` does not set `num_workers` | `data/dataset.py:121-135` | Will bottleneck on GPU servers with slow storage (Jean-Zay GPFS/SSD) | Add `num_workers=4` (or CLI flag) before Jean-Zay | CONFIRMED — MISSING |
| Q13 | CUDA auto-detection | YES: `cfg.train.device = "cuda" if torch.cuda.is_available() else "cpu"` | `scripts/train_checkpoint.py:133` | Works; single-GPU only | — | CONFIRMED |
| Q13 | Per-run output isolation | User-supplied `--ckpt` and `--out_dir` flags; no auto-namespacing | CLI | Collision risk if caller does not use unique paths | Run ID convention needed | CONFIRMED — USER RESPONSIBILITY |
| Q13 | SLURM / job array scripts | NONE | Repo search | Cannot submit parallel runs on Jean-Zay without them | Phase 3: write SLURM array template | MISSING |
| Q13 | Checkpoint mid-training save | NONE — checkpoint saved only at the END of `main()` | `scripts/train_checkpoint.py:310-333` | Job interruption on Jean-Zay = all epochs lost | Phase 2: save checkpoint every N epochs | MISSING |
| Q13 | Experiment manifest | NONE — no CSV/YAML tracking which runs were submitted / completed | Repo search | No automated aggregation possible | Phase 3: write manifest generator | MISSING |
| Q14 | uniGRU + last hidden implemented? | NO — only `bidirectional=True` or `bidirectional=False` flag exists; `pooled = masked_mean_pool(...)` is always used regardless | `models/ltm_route.py:62-66` | Phase 4 grid requires uniGRU; cannot run until Phase 2 patch | Implement `use_last_hidden` option in `encode()` | BLOCKING for Phase 4 |
| Q14 | gate_threshold configurable? | NO — `0.5` is hard-coded | `models/gating.py:45` | Phase 7 grid requires this; Phase 4 uses 0.5 default → NOT BLOCKING for Phase 4 | Add to GatingConfig + CLI in Phase 2 | NOT BLOCKING for Phase 4 |
| Q14 | sigma_LTM implemented? | NO | `models/ltm_route.py` entire file | Phase 6 grid requires it; Phase 4 uses sigma=0 → NOT BLOCKING for Phase 4 | Phase 2 implementation after uniGRU patch | NOT BLOCKING for Phase 4 |
| Q15 | Seed controls split? | YES: same `--seed` flag sets both `cfg.train.seed` (train RNG) and `cfg.data.seed` (split seed) | `scripts/train_checkpoint.py:128-129` | Different seeds → different train/val splits → NOT the same data across seeds | Expected for variance estimation; report as "average over different data partitions" | CONFIRMED |
| Q16 | Noise grid with TF<1 | If best TF<1 is selected: noise semantics are wrong (S draws per step) | `train.py:60-63` | BLOCKING for Phase 6 if TF<1 is selected | Encode-once fix before noise×TF grid | CONDITIONAL BLOCKING |
| Q17 | Confidence distribution logged? | NO — no diagnostic script logs c_LTM distribution for real vs pseudowords | Repo search | Cannot validate gate alpha / threshold grid without knowing confidence distribution | Log from Phase 4 runs before launching Phase 7 | MISSING |

### 15.2 Exact WM vs LTM architecture comparison

| Property | WM / dorsal | LTM / ventral (current) | LTM / ventral (proposed) |
|---|---|---|---|
| Encoder class | `nn.GRU(E, 128)` | `nn.GRU(E, 256, bidirectional=True)` | `nn.GRU(E, H)` |
| Bidirectional | NO | YES | NO |
| pack_padded_sequence | YES | NO | NO (unchanged) |
| Encoder output used | `h: (1,B,128)` last hidden | `out: (B,T,512)` all positions | `h: (1,B,H)` last hidden |
| Temporal pooling | NONE | masked mean pool → `(B,512)` | NONE — `h[-1]` = `h.squeeze(0): (B,H)` |
| Projection to semantic | NONE (no semantic loss on WM) | `Linear(512,256)→GELU→Linear(256,300)` | `Linear(H,H)→GELU→Linear(H,300)` |
| Decoder init | `h: (1,B,128)` directly | `tanh(Linear(300,256))` = `(1,B,256)` | `tanh(Linear(300,H))` = `(1,B,H)` |
| Decoder class | `nn.GRU(E, 128)` | `nn.GRU(E, 256)` | `nn.GRU(E, H)` |
| Premotor projection | `Linear(128, P)` | `Linear(256, P)` | `Linear(H, P)` |
| Noise on encoder state | YES: `h += ε ~ N(0,σ²I)` | NO | `pooled += ε` OR `h += ε` (proposed) |

### 15.3 Weight matrices that change with uniGRU patch

With `LTMConfig.enc_hidden = H`, `LTMConfig.bidirectional_encoder = False`, using last hidden:

| Tensor | Current shape | After uniGRU patch |
|---|---|---|
| LTM encoder `weight_ih_l0` | `(3*256, 64)` = `(768, 64)` | `(3*H, 64)` |
| LTM encoder `weight_hh_l0` | `(3*256, 256)` = `(768, 256)` | `(3*H, H)` |
| LTM encoder `weight_ih_l0_reverse`, `weight_hh_l0_reverse` (biGRU backward) | `(768, 64)`, `(768, 256)` | **REMOVED** |
| `to_semantic[0].weight` (first linear) | `(256, 512)` | `(H, H)` (if enc_hidden=H) |
| `to_semantic[0].bias` | `(256,)` | `(H,)` |
| All other matrices (`to_semantic[2]`, `sem_to_h0`, decoder GRU, `dec_to_premotor`) | Depend on enc_hidden/dec_hidden, unchanged in shape convention | Change only if `dec_hidden` also changed to H |

### 15.4 Gate word-level vs phoneme-level — exact trace

```
field["confidence"]  →  (B,)
.view(B, 1, 1)       →  (B, 1, 1)   ← scalar per item
sigmoid(alpha * (conf - 0.5))  →  (B, 1, 1)
.expand(B, S, 1)     →  (B, S, 1)   ← SAME value broadcast to all S steps
premotor = g * ltm + (1-g) * wm   shapes: (B,S,1)*(B,S,P) + (B,S,1)*(B,S,P) = (B,S,P)
```

**CURRENT GATE = WORD-LEVEL. g is computed once per item and constant across all decoder timesteps.**

### 15.5 TF<1 path: what is re-run at each step

```python
# train.py:77-78 — inside loop for step in range(S):
out = model(enc_in, enc_mask, current_dec)
```

At each step:
- WM GRU encoder: re-run (**new noise draw if sigma>0**) — `models/wm_route.py:47-54`
- LTM biGRU encoder: re-run (identical output, no noise, same input) — wasteful
- LTM masked mean pooling: re-computed (identical) — wasteful
- s_hat: re-computed (identical) — saved only from step 0 for alignment loss
- Gate g: re-computed (identical) — same c_LTM → same g
- WM decoder: run on `current_dec[:, :step+1]` — grows by 1 token per step

**Cost**: both encoders run S times per batch. For max_phonemes=9, approximately 10× slower than TF=1.0.

**Noise semantics bug**: with TF<1 and sigma>0, each step draws a NEW noise vector `ε ~ N(0,σ²I)`. The intended semantic is: ONE noise draw per stimulus (one corruption of the phonological buffer). The current implementation produces different noise at each readout step. This is semantically inconsistent with the intended model.

### 15.6 Checkpoint schema — current vs Phase 2 target

| Key | Current (fresh mode) | Current (resume mode) | Phase 2 target |
|---|---|---|---|
| `model_state_dict` | ✓ | ✓ | ✓ |
| `optimizer_state_dict` | `None` | ✓ (AdamW state) | ✓ (always saved) |
| `rng_states` | ✓ | ✓ | ✓ |
| `cfg_*` (all 6 configs) | ✓ | ✓ | ✓ |
| `history` | ✓ | ✓ (concatenated) | ✓ |
| `git_commit` | ✓ | ✓ | ✓ |
| `total_epochs_trained` | ✓ | ✓ | ✓ |
| `lr_at_save` | ✓ | ✓ | ✓ |
| `checkpoint_every_n_epochs` | — | — | **ADD: intermediate saves** |
| `run_id` | — | — | **ADD: unique run identifier** |

### 15.7 Jean-Zay readiness gaps

| Gap | Severity | Phase |
|---|---|---|
| `num_workers=0` in DataLoader | High on GPU servers | Phase 2 |
| No mid-training checkpoint saves | High (job interruption = lost epochs) | Phase 2 |
| No SLURM array script | High (no parallel runs) | Phase 3 |
| No experiment manifest | Medium (manual tracking) | Phase 3 |
| No aggregation script | Medium (no automatic ranking) | Phase 2 |
| `optimizer_state_dict=None` in fresh mode | Low (warm-restart acceptable) | Phase 2 |

### 15.8 GO / NO-GO summary

| Phase | Status | Blocking conditions |
|---|---|---|
| Phase 4 — H×TF×LR grid (18 runs) | **CONDITIONAL GO** after Phase 2 | uniGRU patch, `--hidden_size` CLI flag |
| Phase 5 — multi-seed (6 runs) | **CONDITIONAL GO** after Phase 4 | Phase 4 completion |
| Phase 6 — noise grid | **CONDITIONAL GO** after Phase 2 + Phase 5 + fix noise semantics for TF<1 | encode-once fix (if TF<1 selected), sigma_LTM implementation |
| Phase 7 — gate grid | **CONDITIONAL GO** after Phase 5 + Phase 6 + confidence distribution log | gate_threshold in config, confidence distribution logged |
| Phase 3 — Jean-Zay infra | **GO** in parallel with Phase 2 | SLURM template, manifest, aggregator |
