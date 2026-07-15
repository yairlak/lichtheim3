# Lichtheim3 Hyperparameter Inventory

> **Status: DRAFT — do not launch training from this document before repo inspection and supervisor validation.**

Status: draft for Yair/Lichtheim3 gridsearch discussion.
Scope: current `lichtheim3` prototype, branch `eval/external-csv-datasets`, checkpoint `checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt`.
Rule: values are listed only when attested in the supplied files. Missing values are marked `TO INSPECT IN REPO`.

---

## 1. Source hierarchy used

Essential sources:

1. `grid_search.sh` and `train_repetition.sh` for SWP/Dager-style job organization and CLI arguments.
2. `article_SWP.pdf` for scientific protocol, model selection, optimizer/loss, and gridsearch ranges.
3. `current_pipeline_summary.md` for the current Lichtheim3 checkpoint, data split, GloVe role, route architecture, and evaluation caveats.
4. `evaluation_regimes.md` for teacher-forced vs autoregressive vs noisy evaluation regimes.
5. `current_and_proposed_architecture_equations.md` for implemented architecture equations and known dimensions.
6. `wfe_route_noise_and_ltm_audit_summary.md` only as background for why AR, length effects, WM noise, gate alpha, and LTM sensitivity matter.

Legacy sources `scale_up_training_data_recommendation.md` and `wfe_retraining_recommendation.md` are used only for historical context, not as current-state truth.

---

## 2. SWP / Dager reference hyperparameters

### 2.1 Scientific protocol from article and appendix

| Field | SWP/Dager value or protocol | Status |
|---|---:|---|
| Architecture | Standard encoder-decoder sequence model | Attested in paper |
| Recurrent types explored | Elman RNN and LSTM | Attested in paper |
| Selected recurrent type | LSTM | Attested in paper |
| Layers | Final fine grid: single-layer models; preliminary range 1-2 | Attested in paper |
| Hidden size final optimum | 128 | Attested in paper |
| Hidden sizes in final grid | 64, 128 | Attested in appendix |
| Batch sizes in final grid | 1024, 2048, 4096 | Attested in appendix |
| Selected batch size | 2048 | Attested in Results |
| Dropout final grid | 0.0, 0.1, 0.2 | Attested in appendix |
| Selected dropout | 0.0 | Attested in Results |
| Learning rates final grid | 5e-4, 1e-3, 5e-3 | Attested in appendix |
| Selected learning rate | 1e-3 | Attested in Results |
| Optimizer | Adam | Attested in paper |
| Loss | Cross-entropy variant ignoring PAD tokens | Attested in paper |
| LSTM epochs | 100 | Attested in paper |
| RNN epochs | 150, but RNN failed to reach zero error | Attested in paper |
| Teacher forcing | Implemented in preliminary search; found detrimental | Attested in appendix |
| Teacher-forcing preliminary range | 0.0-0.7 | Attested in appendix |
| Cross-validation | 5-fold CV | Attested in appendix |
| Sampling | Each CV split: 30k training words sampled by frequency to generate 1e6 samples | Attested in appendix |
| Early stopping | Chose epoch 75, middle of stable zero-error period epochs 65-85 | Attested in appendix |
| Model selection | 1. perfect CV train accuracy; 2. highest CV val accuracy; 3. smallest model complexity | Attested in paper |
| Robustness seeds | 10 additional seeds with selected hyperparameters | Attested in appendix |
| Evaluation metrics | Error rate and edit distance | Attested in paper |

### 2.2 SWP-style launcher files supplied here

`grid_search.sh` is not the full article grid; it is a submitted-run wrapper with one hyperparameter setting and 10 seeds.

| Field | Value in supplied `grid_search.sh` | Interpretation |
|---|---:|---|
| Epochs | 100 | Compatible with paper LSTM training length |
| Batch size | 1024 | Not the selected paper optimum of 2048; likely a run variant or cluster wrapper default |
| Recurrent type | `lstm` | Compatible with selected model |
| Hidden size | 128 | Compatible with selected model |
| Num layers | 1 | Compatible with selected model |
| Learning rate | 0.001 | Compatible with selected model |
| Dropout | 0.0 | Compatible with selected model |
| Teacher forcing ratio | 0.0 | Compatible with appendix finding that TF was detrimental |
| Seeds | 70, 37, 96, 45, 5, 68, 83, 1, 95, 4 | 10-seed robustness loop |
| Fold ids | empty string | No explicit fold id in this supplied wrapper |

`train_repetition.sh` passes these arguments to `scripts/train_repetition.py`:

```bash
--num_epochs
--batch_size
--recur_type
--hidden_size
--num_layers
--learn_rate
--dropout
--tf_ratio
--seed
--verbose
# optional: --fold_id, if non-empty
```

Compute assumptions in the supplied wrapper: SLURM GPU partition, 1 A40 GPU, 4 CPUs, 10 GB RAM, 10-hour time limit, conda env `swpm`, Python 3.12.

---

## 3. Current Lichtheim3 data and checkpoint state

| Field | Current value | Status |
|---|---:|---|
| Checkpoint | `checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt` | Attested |
| git_commit | `ab9353cfeb92516e5a44625bafe01407d87526ae` | **Confirmed from checkpoint metadata** |
| total_epochs_trained | 120 (60 initial + 60 low-LR continuation) | **Confirmed from checkpoint metadata** |
| batch_size (production run) | 64 | **Confirmed from `cfg_train.batch_size` in checkpoint** |
| lr_at_save | 1e-4 (low-LR continuation phase) | **Confirmed from `lr_at_save` field in checkpoint** |
| seed (train + split) | 0 | **Confirmed from `cfg_train.seed` in checkpoint** |
| WM interference_noise | 0.1 | **Confirmed from `cfg_wm.interference_noise` in checkpoint** |
| gate_alpha | 4.0 | **Confirmed from `cfg_gating.alpha` in checkpoint** |
| gate_usage_prior | 0.5 | **Confirmed from `cfg_gating.usage_prior` in checkpoint** |
| teacher_forcing_ratio | **Not stored** — pre-patch checkpoint; training was always teacher-forced. Interpret as implicit `teacher_forcing_ratio=1.0`. The new `TrainConfig.teacher_forcing_ratio` field was added by the 2026-07 patch and is absent from this checkpoint's `cfg_train`. | Note below |
| Lexicon source | CMU Pronouncing Dictionary filtered to GloVe-300 coverage | Attested |
| Lexicon file | `data/lexicon_en_glove_covered.tsv` | Attested |
| Initial lexicon | 30,000 words | Attested by user/context |
| GloVe-covered lexicon | 29,571 words | Attested |
| Train split | 25,136 words, 85% | Attested |
| Validation split | 4,435 words, 15% | Attested |
| Split seed | 0 (same as `--seed`; confirmed from checkpoint) | **Confirmed from checkpoint metadata** |
| Validation role | Held out throughout training; not used for previous hyperparameter selection | Attested |
| Current train ceiling | full/gated exact match = 1.0000, train errors = 0 | Attested |
| Route ceiling on train | WM alone = 2 errors; LTM alone = 324 errors | Attested by user/context |
| Current WFE teacher-forced deterministic | full/gated approx 0.987; WM approx 0.991 or 0.987 depending source; LTM approx 0.790 | Attested, reconcile exact JSON in repo |
| Current main behavior regime | Autoregressive should become primary; teacher-forced only ceiling/debug | Design decision from meeting and docs |

---

## 4. Current Lichtheim3 architecture hyperparameters

These are architecture dimensions or fixed model settings. They should not all become gridsearch dimensions now.

| Hyperparameter | Current value | Status / source note | Gridsearch now? |
|---|---:|---|---|
| Shared phoneme embedding dim `phon_embed_dim` | 64 | Attested in architecture equations | No, defer |
| WM GRU hidden dim `H_WM` | 128 | Attested | No, defer |
| WM encoder layers | 1 | Attested in equations | No, defer |
| WM decoder layers | 1 | Attested in equations | No, defer |
| LTM encoder type | bidirectional GRU | Attested | No, but padding fix deferred |
| LTM encoder hidden dim per direction | 256 | Attested | No, defer |
| LTM decoder type | GRU | Attested | No, defer |
| LTM decoder hidden dim | 256 | Attested | No, defer |
| Semantic / GloVe dim | 300 | Attested | No |
| Premotor dim | 128 | Attested as default, not stored in config | No, defer |
| Motor output | shared linear `premotor_dim -> vocab_size` | Attested | No |
| Gate formula | `g = sigmoid(alpha * (confidence - 0.5))` | Attested | Gate alpha only, limited |
| Gate alpha | 4.0 | Attested | Yes, Stage C limited |
| Gate learned? | No, parameter-free | Attested | Do not change now |
| Gate input | LTM confidence only | Attested | Do not change now |
| Gate usage prior | 0.5 | **Confirmed from `cfg_gating.usage_prior` in checkpoint** | Not a gridsearch candidate |
| WM interference noise sigma | 0.10 default | Attested | Yes, Stage B limited |
| WM noise application | added to WM encoder final state `h` after encoding | Attested | Existing mechanism only |
| WM noise during training | active when `self.training` and sigma > 0 | Attested | Yes, decide/include explicitly |
| WM noise during eval | active only in collect/noisy mode; off for deterministic eval | Attested | Evaluation regime, not model HP |
| LTM padding behavior | biGRU does not use packing; batch-padding sensitive | Attested | Architecture fix deferred |

---

## 5. Current Lichtheim3 training hyperparameters

Confirmed by static inspection of `train.py`, `train_checkpoint.py`, `config.py`, `losses.py`, and `data/dataset.py`, **plus direct checkpoint metadata read from `lichtheim3_30k_glove_e60_to_e120_lowlr.pt`**.

> **Pre-patch checkpoint note:** The production checkpoint (`lichtheim3_30k_glove_e60_to_e120_lowlr.pt`) was trained before the 2026-07 gridsearch capability patch. Its `cfg_train` dict does **not** contain a `teacher_forcing_ratio` key, because that field did not exist when the checkpoint was saved. The training code at the time of that checkpoint always used full teacher forcing. The correct interpretation is `teacher_forcing_ratio = 1.0` (implicit). The 2026-07 patch adds `teacher_forcing_ratio` to `TrainConfig` with default `1.0`, so loading this checkpoint's `cfg_train` via `TrainConfig(**ckpt["cfg_train"])` will silently apply the default `1.0` — the correct value. New checkpoints produced after the patch will store `teacher_forcing_ratio` explicitly.

| Hyperparameter | Confirmed value | Source | CLI exposed? | Gridsearch now? |
|---|---:|---|---|---|
| Optimizer | `torch.optim.AdamW` | `train.py:102` | No (hard-coded) | N/A |
| Optimizer betas | PyTorch defaults: (0.9, 0.999) | PyTorch AdamW default | No | N/A |
| Weight decay | `1e-5` | `config.py:TrainConfig.weight_decay` | No | No |
| Gradient clip | `1.0` | `config.py:TrainConfig.grad_clip` | No | No |
| Initial learning rate | `1e-3` (config default) | `config.py:TrainConfig.lr` | **YES: `--lr`** | Yes, Stage A |
| **Production lr_at_save** | **`1e-4`** | **`lr_at_save` field in checkpoint — low-LR continuation phase** | **YES: `--lr`** | Yes, used for resume |
| **Production total_epochs_trained** | **120** | **`total_epochs_trained` in checkpoint (60 initial + 60 low-LR)** | **YES: `--epochs`** | No |
| **Production batch_size** | **64** | **`cfg_train.batch_size` in checkpoint** | **YES: `--batch_size`** | No (test if needed) |
| **Production seed** | **0** | **`cfg_train.seed` in checkpoint** | **YES: `--seed`** | Use 1 seed for screening |
| **Production git_commit** | **`ab9353c`** | **`git_commit` field in checkpoint** | N/A | N/A |
| Epochs | Config default `8`; CLI default `10`; production run 120 total | `config.py:TrainConfig.epochs`, `train_checkpoint.py:64` | **YES: `--epochs`** | No (verify by smoke run) |
| Batch size | `64` | `config.py:TrainConfig.batch_size` | **YES: `--batch_size`** | No (test if needed) |
| Seed (train + split) | `0` default; same flag controls train RNG and data split | `config.py:TrainConfig.seed` and `DataConfig.seed`, `train_checkpoint.py:73` | **YES: `--seed`** | Use 1 seed for screening |
| Training sampler | `WeightedRandomSampler` with log-frequency weights, `num_samples=len(entries)`, `replacement=True` | `data/dataset.py:124-133` | No | N/A |
| Dorsal pseudoword pool | `4000` pronounceable pseudowords, frequency-flat, added as auxiliary WM CE | `config.py:TrainConfig.dorsal_pool_size`, `train.py:60-68` | No | No |
| **Teacher forcing ratio during training** | **`1.0` default (full TF = historical behavior). Scheduled sampling implemented.** `0.0`=fully AR; `0.0–1.0`=partial. Step-by-step at ratio<1; vectorized path preserved at ratio=1. | `config.py:TrainConfig.teacher_forcing_ratio`, `train.py:_forward_scheduled_sampling` | **YES: `--teacher_forcing_ratio`** ✓ PATCHED | Stage A candidate |
| Autoregressive training support | **IMPLEMENTED via scheduled sampling.** `--teacher_forcing_ratio 0.0` = fully AR training. Validation always uses TF=1.0. | `train.py:_forward_scheduled_sampling`, `train.py:run_epoch` | **YES: `--teacher_forcing_ratio`** ✓ PATCHED | Stage A candidate |
| **Dropout** | **NOT IMPLEMENTED anywhere in the model or training pipeline.** | Checked `wm_route.py`, `ltm_route.py`, `gating.py`, `motor.py`, `config.py` | **NO** | Cannot gridsearch |
| Loss: `L_rep` weight | `1.0` — CE on full/gated motor output vs target | `config.py:LossConfig.rep` | No (config only) | Config-only change |
| Loss: `L_align` weight | `1.0` — cosine + 0.1×MSE of s_hat vs GloVe target | `config.py:LossConfig.align` | No (config only) | Stage C only |
| Loss: `L_dec` weight | `0.5` — CE on LTM-only logits vs target | `config.py:LossConfig.dec` | No (config only) | Stage C only |
| Loss: `L_wm` weight | `0.5` — CE on WM-only logits (main + pseudoword pool) | `config.py:LossConfig.wm` | No (config only) | Stage C only |
| Loss: `L_gate` weight | `0.05` — `(mean(g) - usage_prior)²` regularizer | `config.py:LossConfig.gate` | No (config only) | Do not touch |
| Loss: `label_smoothing` | `0.0` | `config.py:LossConfig.label_smoothing` | No | No |
| Loss: `usage_prior` | `0.5` | `config.py:GatingConfig.usage_prior` | No | No |
| Alignment loss formula | `(1 - cosine_sim) + 0.1 × MSE` | `losses.py:48-50` | No | N/A |
| Gate regularizer formula | `(gate.mean() - usage_prior)²` | `losses.py:53-55` | No | N/A |
| WM interference noise (training) | `0.10` — default and **confirmed from `cfg_wm.interference_noise` in production checkpoint**. Added to WM encoder state when `self.training and sigma > 0`. `0.0` disables. | `config.py:WMConfig.interference_noise`, `models/wm_route.py:53-54` | **YES: `--interference_noise`** ✓ PATCHED | Stage B candidate |
| Split seed | Same as `--seed` (not a separate parameter) | `config.py:DataConfig.seed`, set from `cfg.data.seed = args.seed` | **YES via `--seed`** | Use fixed seed for comparability |
| Val fraction | `0.15` (fixed; not CLI-exposed) | `config.py:DataConfig.val_fraction` | No | No |
| Checkpoint selection criterion | Not implemented; no early stopping or auto-selection | `train_checkpoint.py` (no eval loop) | N/A | Define before gridsearch |
| Early stopping | **NOT IMPLEMENTED** | `train_checkpoint.py` | N/A | Add separately if needed |
| SLURM / shell scripts | TO INSPECT / NOT FOUND in repo | Bash background command timed out; no scripts found in main tree | N/A | Check with user |
| Log every N steps | `50` | `config.py:TrainConfig.log_every` | No | N/A |

---

## 6. Hyperparameters already existing vs candidates vs diagnostics vs deferred architecture

### 6.1 Already existing / attested in current model

- `phon_embed_dim = 64`
- `H_WM = 128`
- `H_LTM_enc = 256` per direction
- `H_LTM_dec = 256`
- `semantic_dim = 300`
- `premotor_dim = 128`
- `gate_alpha = 4.0`
- `interference_noise = 0.10`
- fixed train/validation split: 25,136 / 4,435 words
- GloVe alignment target and frozen semantic bank at inference
- teacher-forced and autoregressive evaluation modes
- deterministic/noisy evaluation distinction via `collect`

### 6.2 Hyperparameters to gridsearch now, limited — UPDATED AFTER INSPECTION

**Directly gridsearchable via current CLI without code changes:**

1. **Learning rate** — `--lr` flag; current default 1e-3. Candidate: 5e-4. `CONFIRMED_IN_CODE / CLI_EXPOSED`.
2. **Seed** — `--seed` flag; controls both training RNG and data split. Use 1 seed for screening. `CLI_EXPOSED`.
3. **Batch size** — `--batch_size` flag; current default 64. Not a first priority. `CLI_EXPOSED`.

**Require code change before gridsearch (add CLI flag or modify training loop):**

4. **Teacher forcing ratio** — ✓ **NOW IMPLEMENTED**. `--teacher_forcing_ratio` flag in `train_checkpoint.py`. Default `1.0` (full TF, historical behavior). `0.0`=fully AR. Intermediate values use scheduled sampling. `train.py:_forward_scheduled_sampling`.
5. **WM noise during training** — ✓ **NOW CLI-EXPOSED**. `--interference_noise` flag in `train_checkpoint.py`. Default `None` (uses config value `0.10`). Set `0.0` to disable.
6. **Gate alpha** — ✓ **NOW CLI-EXPOSED**. `--gate_alpha` flag in `train_checkpoint.py`. Default `None` (uses config value `4.0`).
7. **Semantic alignment loss weight** (`L_align`)— `REQUIRES CODE SUPPORT`. Current value `1.0` confirmed in `LossConfig`. No CLI flag.
8. **Dropout** — `NOT IMPLEMENTED IN MODEL`. No dropout module anywhere in `wm_route.py`, `ltm_route.py`, `gating.py`, or `motor.py`. Would require architecture additions before it can be tested.

### 6.3 Diagnostic variants, not model-selection dimensions

- Teacher-forced evaluation of every checkpoint as ceiling/debug only.
- Route-isolated evaluation: full/gated, WM-only, LTM-only.
- Fixed-gate evaluation: `g=0.0`, `0.25`, `0.5`, `0.75`, `1.0`, only if available or added as evaluation-only diagnostic.
- WM-noise evaluation sweeps after training, labelled as cognitive/noisy or stress tests.
- LTM nearest-neighbor inspection for lexicalization errors.
- Gate calibration plots by lexicality/length/confidence.

### 6.4 Architecture changes to defer

- LTM biGRU packing fix.
- Learned/noise-sensitive gate.
- Position-level gate.
- Separate lexical confidence from semantic comprehension.
- Comprehension/naming heads.
- New route dimensions or deeper recurrent stacks.
- Any new task labelled naming/comprehension without actual semantic-input training.

---

## 7. Claude Code prompt for repository inspection

Copy-paste this into Claude Code inside the `lichtheim3` repo on branch `eval/external-csv-datasets`.

```text
We need to prepare a scientifically defensible gridsearch/training plan for the current Lichtheim3 prototype. Do not modify architecture and do not launch long training.

Please inspect the repo and produce a report at `docs/hyperparameter_repo_inspection.md` plus a machine-readable JSON at `outputs/hyperparameter_repo_inspection.json`.

Questions to answer exactly, with file paths and line numbers:

1. Training entry points
- What script is used for the current 30k/GloVe checkpoint training? Inspect `scripts/train_checkpoint.py` and any wrappers.
- List all CLI flags, defaults, and config fields relevant to training.

2. Optimisation
- Exact optimizer class, learning rate default, betas, weight decay, scheduler if any.
- Whether optimizer state and RNG state are saved/restored in checkpoints.
- Whether checkpoint metadata stores epoch, lr, seed, batch size, config, optimizer state.

3. Data
- Exact lexicon path defaults, max_words behavior, train/validation split seed, split fraction.
- Exact sampler: frequency weighting, replacement, number of samples per epoch, batch size, shuffle behavior.

4. Decoding during training
- Does training use teacher forcing? If yes, what ratio/default and where is it set?
- Is autoregressive-compatible training implemented? Is scheduled sampling implemented?
- Can `teacher_forcing_ratio` be set from CLI/config without code changes?

5. Losses
- Exact loss terms and weights: phoneme CE, full/WM/LTM route losses if any, semantic alignment loss, gate regularization, usage prior.
- How PAD/BOS/EOS tokens are handled in loss.

6. Regularization/noise
- Is dropout implemented anywhere in the current Lichtheim3 model/training path? If yes, exact modules and CLI/config flags.
- Exact WM noise parameter name, default, and whether it is active during training.
- Can training-time WM noise be set to 0.0 without architecture change?

7. Gate
- Exact `GatingConfig` fields and defaults: alpha, usage_prior, any regularization terms.
- Can gate alpha be set from CLI/config for training and/or evaluation?

8. Evaluation scripts
- Confirm exact CLI for train deterministic eval, val deterministic eval, WFE teacher-forced, WFE autoregressive, WFE noisy, SSP autoregressive if available.
- Confirm which metrics are written to JSON/TSV.
- Confirm whether Regime C (autoregressive + WM noise) is implemented end-to-end. If not, say exactly what is missing.

9. Current checkpoint metadata
- Load `checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt` only for metadata/config, not for training.
- Report stored hyperparameters if present: lr, optimizer, epochs, batch size, seed, loss weights, noise sigma, gate alpha.

10. Output
- Produce a compact table of current hyperparameters with columns: name, value, source file/line or checkpoint metadata, can vary without architecture change yes/no, recommended gridsearch yes/no.
- Do not propose new architecture.
- Do not run training.
```
