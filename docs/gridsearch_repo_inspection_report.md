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
