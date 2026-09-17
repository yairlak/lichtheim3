# VENTRAL_INTERFACE_TRAINING_SUPERVISION_AUDIT

Workstream: VENTRAL SEMANTIC DIRECTIONAL DOSE (design + audit pass). POST_STAGE / PAPER_PROGRAMME.
Written 2026-09-17. **No training, no optimizer step, no parameter update, no scientific evaluation.**

**Sources**
* Live code at `0f25b5b87efd8038dfc4c86b5ca2935bf7b178c7`. Model and training code is byte-identical to the V6 code commit `78f550570b82b3230cb621d9d000600ed192eb74` (established in the closed lineage).

| file | blob |
|---|---|
| `losses.py` | `67ddf64f37d5e3210b79e9f4591f5ab80394c4a3` |
| `models/ltm_route.py` | `548cc68735830986ee38801997c83b2781252d22` |
| `models/dual_route.py` | `5166df420e0f70994e76c68a3542f6a3abfe9809` |
| `models/gating.py` | `8b9244abeb1ad9150c481ab25b316d0eda3edf8e` |
| `models/wm_route.py` | `e58b9670e708d419d88367f4cc8f27b92b4f4c3e` |
| `models/motor.py` | `cf8ffef02450451b7e84f5ce554bd82ac41bdd6f` |
| `config.py` | `799b513dc1acc0caa352efb22a550895f907138f` |
| `scripts/naming_comprehension/train_joint_scratch.py` | `95295d63560ae4c235a6beee8dfb47166f4ed30d` |
| `scripts/naming_comprehension/train_tasks.py` | `0b3a1f19addcc2351edf75248ba31333ccd7cf59` |

* Canonical provenance stored **inside the two authoritative V6 checkpoints** (read-only load of the archival copies `archives/ventral_interface_sources_20260916/s19_u3825_step_10625850.pt` = `a5f21de9…aad76c` and `s20_u3040_step_08445120.pt` = `0657f410…dc79f3`): `lr_policy`, `phase_transitions`, `resolved_settings`, `config.loss`, `resume_provenance`.
* An empirical gradient-routing probe on a **randomly initialised** model at the V6 architecture (synthetic data, no checkpoint, no optimizer), stored **outside git**:
  * `archives/ventral_directional_dose_design_20260917/gradient_routing_probe_synthetic.py`
  * `…/gradient_routing_probe_synthetic.out.json`
  * SHA256 values in `provenance/DESIGN_PASS_EXTERNAL_ARTIFACTS.md`.

Status labels:
* **SUPPORTED_BY_CODE**: the code path exists.
* **ACTIVE_IN_V6_RECIPE**: the code path is exercised with a nonzero weight in the authoritative V6 lineage, as recorded in the checkpoints.
* **VERIFIED_BY_PROBE**: gradient routing was confirmed empirically by the synthetic probe.

---

## 0. The authoritative V6 recipe (read from both checkpoints)

| field | value (identical for s19 u3825 and s20 u3040) | source |
|---|---|---|
| regime | `j0` (retrieval ON, naming ON) | ckpt `regime`; `resolved_settings.active_training_streams = [comprehension, naming, pool, repetition]` |
| schedule | `interleaved_123`: macro-cycles of 6 optimizer steps holding exactly 1 R, 2 N, 3 C, deterministically shuffled | ckpt `schedule`, `schedule_ratio [1,2,3]`; `train_joint_scratch.py:246–290` |
| update | every step is its own `zero_grad(set_to_none=True)` → backward → `clip_grad_norm_(all params, 1.0)` → `step` | `train_joint_scratch.py:1307–1311` |
| optimizer | ONE shared AdamW over all parameters (`shared_adamw`), weight decay 1e-5 | ckpt `optimizer_policy`, `resolved_settings.optimizer_convention` |
| LR, final policy | task-specific: **R 3e-5, N 3e-5, C 1e-4** | ckpt `lr_policy` |
| LR history | two-stage 1e-3 → 1e-4 (repetition-cursor boundary 46,300) until step 2,083,500; task-specific 3e-5 for all tasks until 2,361,300; then R/N 3e-5 and C 1e-4 | ckpt `phase_transitions` (6 entries; s19 and s20 identical except the commit and date on the last re-anchor) |
| loss weights | rep 1.0 · align 1.0 · dec 0.5 · wm 0.5 · gate 0.05 · label_smoothing 0.0 | ckpt `config.loss`, `resolved_settings.loss_weights` |
| dorsal pool CE weight | 0.5 (= `cfg.loss.wm`) | `resolved_settings.dorsal_pool_loss_weight` |
| λ_C / λ_N / τ | 0.087 / 1.0 / 0.10 | `resolved_settings.lambda_C/lambda_N/tau` |
| c_align_weight | **0.0** | ckpt `c_align_weight` |
| dec_weight | 0.5 at every recorded transition | ckpt `phase_transitions[*].new_dec_weight` |
| teacher forcing | 1.0 (all training decoders teacher-forced) | `resolved_settings.teacher_forcing_ratio` |
| noises | interference 0.0, ventral 0.0 | ckpt config |
| gate | alpha 2.0, threshold 0.7, usage_prior 0.5 | ckpt config |
| exposures at the witnesses | s19: R 3825, N 7650, C 12,130 epochs · s20: R 3040, N 6080, C 9641 | ckpt `exposures` |

**Optimizer semantics [STRUCTURAL_CODE_FACT, torch 2.12.1].** `torch.optim.Adam`/`AdamW` only update parameters whose `.grad is not None` (`torch/optim/adam.py:151`), and decoupled weight decay is applied only to those parameters. Because each step calls `zero_grad(set_to_none=True)`, **a task step changes only the parameters that its loss graph reaches**. Their AdamW moment buffers are shared across tasks.

**Provenance caveat.** The recipe before step 1,389,000 is known only through the `old_*` fields of the first recorded phase transition (`interleaved_123`, two-stage LR, dec 0.5). No schedule or objective change is recorded anywhere in the lineage.

## A. L_dec: ventral form regeneration from encoder-produced ŝ

**Code.** `losses.py:64`: `L_dec = CE(out["ltm_logits"], dec_tgt, ignore_index=pad)`, without label smoothing. It is added as `cfg.loss.dec · L_dec` (`losses.py:68`).
* **Input:** `out = model(enc_in, enc_mask, dec_in)` (`train_joint_scratch.py:1268`).
* **`ltm_logits`:** `motor(ltm.decode_from_s_hat(ltm.encode(enc_in, enc_mask), dec_in))` (`dual_route.py:79,90`; `ltm_route.py:146,150–156,195–196`).
* **Target:** `form + [EOS]`.
* **Decoder input:** `[BOS] + form`, teacher-forced (make_batches, `train_tasks.py:223–266`). The encoder-produced ŝ enters `h0 = tanh(sem_to_h0(ŝ))` without normalization.
* **Weight:** 0.5 (`dec_weight` 0.5 throughout).
* **Where it runs:** R steps only (1 of every 6 optimizer steps), at LR 3e-5.
* **Gradients reach [VERIFIED_BY_PROBE]:** phon_embed, LTM encoder GRU, `to_semantic.0`, `to_semantic.2`, `sem_to_h0`, LTM decoder GRU, `dec_to_premotor`, motor. **Not reached:** the WM route.

**Confirmation: YES, ACTIVE_IN_V6_RECIPE.** L_dec already trains the ventral decoder, teacher-forced, from encoder-produced ŝ, and backpropagates into the semantic encoder.

## B. L_align: semantic alignment to raw GloVe

**Code.** `losses.py:47–50`: `L_align = (1 − mean_i cos(ŝ_i, g_i)) + 0.1 · MSE(ŝ, g)`, where `g = batch["semantic"] = bank_raw[item]`, i.e. **raw, unnormalized GloVe** (`train_tasks.py:260`).
* The cosine term is scale-invariant and constrains direction.
* The MSE term is the mean over all 300 dimensions and batch items; it constrains magnitude and direction in raw GloVe coordinates.

| use | status | weight | where |
|---|---|---|---|
| R-step alignment inside `total_loss` | **ACTIVE_IN_V6_RECIPE** | `cfg.loss.align` = 1.0 | R steps (1/6), LR 3e-5; target = the repetition item's own raw GloVe |
| C-step alignment (`c_align_weight · alignment_loss(ŝ, g)`) | SUPPORTED_BY_CODE, **NOT ACTIVE** (weight 0.0 throughout the lineage; at 0.0 nothing enters the graph, `train_joint_scratch.py:1297–1302`) | 0.0 | would be C steps |
| Phase-2 comprehension objective "C0" (`train_tasks.comprehension_objective`) | SUPPORTED_BY_CODE, not used by `_interleaved_step` | — | historical task driver only |

**Gradients reach [VERIFIED_BY_PROBE]:** phon_embed, LTM encoder, `to_semantic.0`, `to_semantic.2`. They do **not** reach `sem_to_h0`, the decoder, the premotor projection, motor or WM.

**Confirmation: a semantic alignment loss already exists and is active.** It pulls ŝ toward the item's own raw GloVe vector in both direction (cosine) and scale (MSE).

## C. Naming supervision: raw GloVe → ventral decoder

**Code.**
* `naming_objective` (`train_tasks.py:322–327`) computes `_seq_ce(naming_forward(model, batch["semantic"], dec_in), dec_tgt)`.
* `naming_forward` (`train_tasks.py:275–284`) is `motor(ltm.decode_from_s_hat(sem, dec_in))`.
* The path is raw `bank_raw[item]` → `h0 = tanh(sem_to_h0(g))` → LTM decoder GRU over `phon_embed([BOS] + form)` → `dec_to_premotor` → `motor.proj` → CE against `form + [EOS]`.

**Settings:**
* Teacher forcing: yes.
* Weight: λ_N = 1.0.
* **Where it runs:** N steps, i.e. **2 of every 6 optimizer steps**, at LR 3e-5, 64 items per step.

**Gradients reach [VERIFIED_BY_PROBE]:** `sem_to_h0`, LTM decoder, `dec_to_premotor`, motor, **and phon_embed** (the decoder input embeddings, which the WM route and both encoders share). **The semantic encoder is bypassed:** no gradient reaches the LTM encoder, `to_semantic.0` or `to_semantic.2`.

**Confirmation: ACTIVE_IN_V6_RECIPE.** Naming trains the ventral decoder strongly from raw lexical GloVe. Per macro-cycle it gets twice as many optimizer steps as the ŝ-driven decoder supervision of R steps, at the same LR and with twice L_dec's loss weight. **[STRUCTURAL_CODE_FACT, no causal claim]**

## D. Comprehension supervision: retrieval

**Code.** `retrieval_loss` (`train_tasks.py:286–301`) computes `CE( normalize(ŝ) @ semantic_bank.T / τ, target_bank_idx )` with τ = 0.10.
* `ŝ = comprehension_forward = ltm.encode(enc_in, enc_mask)` (`train_tasks.py:269–272`).
* `semantic_bank` is the non-persistent **buffer** `F.normalize(bank_raw)` (`ltm_route.py:103,163–165`). It is not a parameter and has `requires_grad=False` [VERIFIED_BY_PROBE].
* Retrieval runs over all 29,571 rows against the canonical target row. Population: the 27,981 canonical C targets (homophone policy frozen).
* Settings: λ_C = 0.087, τ = 0.10. **Where it runs:** C steps, **3 of every 6 optimizer steps**, at **LR 1e-4**. `c_align_weight` = 0.0, so no alignment term is added on C steps.
* **Representation:** only the **direction** of ŝ enters (the normalized query), matched against normalized GloVe rows.

**Gradients reach [VERIFIED_BY_PROBE]:** phon_embed, LTM encoder, `to_semantic.0`, `to_semantic.2`. **The decoder receives no C gradient:** `sem_to_h0`, LTM decoder, `dec_to_premotor` and motor all have grad None, and AdamW therefore leaves them untouched on C steps.

## E. Repetition step: FULL, isolated and dorsal terms

An R step (`train_joint_scratch.py:1263–1280`) computes, in **one backward pass**:

```
loss_R = 1.0·L_rep + 1.0·L_align + 0.5·L_dec + 0.5·L_wm + 0.05·L_gate + 0.5·pool_CE
```

The terms are:

| term | formula | source |
|---|---|---|
| `L_rep` | `CE(out["logits"], tgt)`, with `logits = motor(g·ltm_premotor + (1−g)·wm_premotor)` (FULL / gated) | `losses.py:62`; `dual_route.py:85,88`; `gating.py:56` |
| `L_wm` | `CE(out["wm_logits"], tgt)`, the isolated dorsal route | `losses.py:63` |
| `L_dec` | isolated ventral (§A) | |
| `L_align` | §B | |
| `L_gate` | `(mean g − 0.5)²` | `losses.py:53–55` |
| `pool_CE` | `CE(pout["wm_logits"], pool targets)` on a separate batch of 64 pseudowords, weight `cfg.loss.wm` = 0.5 | `train_joint_scratch.py:1273–1277` |

**Gradients from FULL `L_rep` [VERIFIED_BY_PROBE]:** every block (phon_embed, LTM encoder, both `to_semantic` layers, `sem_to_h0`, LTM decoder, `dec_to_premotor`, motor, WM encoder, WM decoder, `wm.to_premotor`). Into ŝ they arrive by **two paths**:
1. through the ventral premotor, scaled by g;
2. through the gate confidence (§F).

**`L_wm` and `pool_CE`** reach phon_embed, motor and WM only.

## F. Gate-related gradients (EXISTING_GATE)

**Code (`gating.py:44–57`, `ltm_route.py:167–190`, `dual_route.py:79–85`):**

```
q         = normalize(ŝ)                             (no detach)
sims      = q @ semantic_bank.T                      (bank: buffer, no grad)
c_LTM     = max_j sims_j   (confidence = top2[:,0])  (no detach)
g         = sigmoid(alpha · (c_LTM − gate_threshold)),   alpha = 2.0, threshold = 0.7
premotor  = g · ltm_premotor + (1 − g) · wm_premotor     (word-level g, broadcast over steps)
```

* **The gate has no parameters** [VERIFIED_BY_PROBE: `gate.*` is empty]. It is a fixed function of ŝ.
* **Nothing is detached:** `confidence.requires_grad` is True, dL_rep/dc_LTM ≠ 0, and dL_gate/dc_LTM ≠ 0 [VERIFIED_BY_PROBE].
* **Losses that send gradient into ŝ through the gate:** `L_rep` (FULL) and `L_gate` (weight 0.05), both on R steps. They reach phon_embed, LTM encoder and `to_semantic`.
* **Losses that don't:** `L_wm`, `L_dec`, `L_align`, `pool_CE`, naming and retrieval do not involve g.
* **[STRUCTURAL_CODE_FACT]** The gate can therefore shape semantic geometry during training: the direction of ŝ, through the max cosine to the bank. It does so via FULL repetition CE and the usage regularizer, even though in the closed GATING/GXLR records native and fixed-0.5 fusion had 0 discordant predictions on intact W3_REP and W4_REP (`wt-gate-x-lesion/paper_programme/gate_x_lesion_recovery/LIVE_CODE_AUDIT.md` §6). **No claim is made about the magnitude or effect of this pressure.**

## G. Task schedule and who can alter what (V6)

| task step | share of steps | LR | parts of the ventral interface it can change |
|---|---|---|---|
| R | 1/6 | 3e-5 | encoder, `to_semantic`, `sem_to_h0`, ventral decoder, premotor, motor, WM, phon_embed (from L_rep, L_dec, L_align, L_gate, L_wm, pool) |
| N | 2/6 | 3e-5 | `sem_to_h0`, ventral decoder, `dec_to_premotor`, motor, phon_embed; **not** encoder or `to_semantic` |
| C | 3/6 | 1e-4 | phon_embed, encoder, `to_semantic`; **not** `sem_to_h0`, decoder or motor |

### Gradient-routing table

Rows are loss terms; columns are parameter blocks. YES means the block receives a non-None, nonzero gradient. **Every cell is VERIFIED_BY_PROBE** (synthetic, randomly initialised V6 architecture; `gradient_routing_probe_synthetic.out.json`), with the source reference in the row.

| loss / task (weight, step type) | phon_embed | LTM encoder | to_semantic.0 | to_semantic.2 | sem_to_h0 | ventral decoder GRU | ventral dec_to_premotor | motor | WM (enc, dec, to_premotor) | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| L_rep FULL (1.0, R) `losses.py:62` | YES | YES | YES | YES | YES | YES | YES | YES | YES | no params; CONDITIONAL→ŝ via c_LTM: YES |
| L_dec (0.5, R) `losses.py:64` | YES | YES | YES | YES | YES | YES | YES | YES | NO | not involved |
| L_align (1.0, R) `losses.py:47–50,65` | YES | YES | YES | YES | NO | NO | NO | NO | NO | not involved |
| L_gate (0.05, R) `losses.py:53–55,66` | YES | YES | YES | YES | NO | NO | NO | NO | NO | no params; →ŝ via c_LTM: YES |
| L_wm (0.5, R) `losses.py:63` | YES | NO | NO | NO | NO | NO | NO | YES | YES | not involved |
| pool_CE (0.5, R) `train_joint_scratch.py:1273–1277` | YES | NO | NO | NO | NO | NO | NO | YES | YES | not involved |
| Naming CE (λ_N 1.0, N) `train_tasks.py:275–284,322–327` | YES | NO | NO | NO | YES | YES | YES | YES | NO | not involved |
| Retrieval CE (λ_C 0.087, C) `train_tasks.py:286–301` | YES | YES | YES | YES | NO | NO | NO | NO | NO | not involved (bank buffer, no grad) |
| C-step alignment (c_align 0.0) | CONDITIONAL: SUPPORTED_BY_CODE, inactive in V6 (weight 0.0 → not in graph) | | | | | | | | | |

## 13. Historical constraint for later CENTRAL decisions

Each of the following is **already present and active** in the V6 recipe. None may be proposed later as "new" without stating what is changed relative to it:
1. **Decoder supervision from encoder-produced ŝ:** `L_dec` (0.5, teacher-forced, R steps). The FULL `L_rep` also trains the decoder from ŝ, through the g-weighted ventral premotor.
2. **Semantic alignment of ŝ to raw GloVe:** `L_align` = cosine + 0.1·MSE (1.0, R steps). Discriminative directional pressure also comes from retrieval CE (τ 0.1, C steps at LR 1e-4).
3. **Decoder supervision from raw lexical GloVe:** Naming CE (λ_N 1.0, twice per macro-cycle), which never passes through the encoder.

The scientifically relevant later question is therefore not "should the decoder be trained on ŝ?". It is **why these existing pressures yield a decoder/interface that is far more compatible with lexical-prototype direction than with native ŝ direction.** This audit does **not** answer that causal question.

## Unresolved in this audit
* The recipe before global step 1,389,000 is known only through the `old_*` fields of the first recorded transition.
* No per-term loss magnitudes, gradient norms or relative update sizes from the actual V6 training were inspected. The probe establishes routing (which blocks are reached), not magnitude.
* Shared AdamW moment interactions across tasks are structural facts only; their effect is not quantified.
