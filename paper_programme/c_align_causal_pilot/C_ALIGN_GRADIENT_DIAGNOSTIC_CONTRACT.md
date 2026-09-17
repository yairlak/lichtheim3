# C_ALIGN_GRADIENT_DIAGNOSTIC_CONTRACT — LICHTHEIM3 VENTRAL INTERFACE C-ALIGN CAUSAL PILOT (read-only gradient diagnostic)

GRADIENT_DIAGNOSTIC_CONTRACT_STATUS=FROZEN_BEFORE_REAL_GRADIENTS

Programme: POST_STAGE / PAPER_PROGRAMME · Decision authority: CENTRAL STEERING · Written 2026-09-17.
Worktree / branch: `/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/wt-c-align-design` · `paper-programme/c-align-causal-pilot-design`, created from `6522926ea84bdff632097e15a003f1f876e3c9ef` (directional-dose results, CLOSED_RESULT_SAFE).

This contract and its implementation (`scripts/c_align_design/c_align_gradient_diagnostic.py`, `tests/test_c_align_gradient_diagnostic.py`) are committed and hashed **before any real-state gradient quantity is computed**.

**This diagnostic never does any of the following:**
* call `optimizer.step()`;
* update any parameter or optimizer state;
* write `.grad` on parameters (it uses `torch.autograd.grad`);
* modify any checkpoint.

It hard-stops unless the per-state `state_dict` SHA256 and the optimizer-state SHA256 are identical before and after.

---

## 1. Question

**Read-only question:** in the mature SOURCE states, do the existing C-step retrieval objective and the candidate C-step raw-GloVe alignment objective exert **different gradient pressures** on the semantic encoder? And is the prospective reference weight **w = 0.1** of a sensible scale relative to the existing C-step retrieval pressure?

**What this diagnostic does not establish:** any causal training effect. Gradient measurements alone support no training claim.

## 2. States (exactly two)

| state | witness | SOURCE checkpoint SHA256 | reconstructed_state_identity |
|---|---|---|---|
| W3_SRC | V6 seed19 u3825 step 10,625,850 | `a5f21de9edb4b2090cb144c666ca4bd8e1f0572333000facb13f6c51cfaad76c` | `6d7282854323b810727cd9c0c90cace6dd8a54d306408cafaa8c40c5455d10a0` |
| W4_SRC | V6 seed20 u3040 step 8,445,120 | `0657f41030e54eac11816116af6c76b66d597c393fc72579a3755401c9dc79f3` | `32928707acc81af215faa829ad94bba566804982bc192bb5116ea32d43d56e29` |

* Reconstruction: the unmodified `run_gate_route_audit.build_state` (GATING manifest blob `e5f0bb4fa68ebe1b200a3fcabe3544ebbff5483b`), which rebuilds the `JointScratchTrainer` at the checkpoint's own recipe and loads model and AdamW state.
* Closed parameter hashes must match: W3_SRC `4a8d4807f1a1928faebebf696d41f4c222b3997b52432d7015fc038881eac785`; W4_SRC `004dda2d9a846ce194d4b0b3f7adc408d5c9d909aadbe413a35e96ceaa411974`.
* W3_REP and W4_REP are **not** used.

## 3. Authoritative V6 training semantics reproduced (STRUCTURAL_CODE_FACT)

Sources: live code at `6522926e…` (model and training code byte-identical to the V6 code commit `78f5505`) and both checkpoints' `resolved_settings`, `config`, `lr_policy`, `phase_transitions` and `optimizer_state_dict`.

| element | V6 value / code | used here |
|---|---|---|
| model mode during training | `self.model.train(True)` (`train_joint_scratch.py:1250`); ventral_noise = interference_noise = 0.0, so train mode draws no noise | `model.train(True)`, verified noise 0 |
| batch size | 64 (`resolved_settings.batch_size`) | 64 |
| C population | `canonical_phonology_indices(entries)`, n = 27,981, hash `10c2f06eda769bf620ca3dbb9889204e4431cac2bfe0d0f5dd37fa4df2bb9f50` | same, asserted |
| C batch collation | `trainer.batch("comprehension")` → `build_batch(entries, bank_raw, vocab, idx)` → `make_batches` | the same frozen `build_batch` |
| C target | `bank_idx` = the item's own bank row; `semantic` = `bank_raw[bank_idx]` (**raw, unnormalized GloVe**) | same |
| ŝ on C steps | `comprehension_forward(model, enc_in, enc_mask)` = `model.ltm.encode(...)` | same |
| retrieval loss | `retrieval_loss(ŝ, model.ltm.semantic_bank, bank_idx, TAU)` = CE(normalize(ŝ)·bank_n / τ) | same |
| λ_C / τ | `LAMBDA_C = 0.087`, `TAU = 0.10` (constants; `resolved_settings.lambda_C/tau`) | same, imported |
| c_align path (FINAL-2A) | `if self.c_align_weight > 0: loss += c_align_weight · alignment_loss(ŝ, c["semantic"])` (`train_joint_scratch.py:1297–1302`); V6 weight 0.0 | evaluated at unit weight (measurement only) |
| alignment formula | `losses.alignment_loss = (1 − mean_i cos(ŝ_i, g_i)) + 0.1 · MSE(ŝ, g)` (`losses.py:47–50`) | same function; components also reported |
| R-step L_align | `total_loss(model(enc_in, enc_mask, dec_in), batch, cfg.loss, pad)["align"]`, weight `cfg.loss.align = 1.0` (`losses.py:65`) | same full-model forward on the **same** batch |
| learning rates | task-specific: R 3e-5, N 3e-5, C 1e-4 (`lr_policy`) | used only in the secondary Adam proxy |
| schedule | interleaved 1R : 2N : 3C per 6-step macro-cycle | exposure argument (§8) |
| optimizer | one shared AdamW over all parameters; betas (0.9, 0.999), eps 1e-8, weight_decay 1e-5; per-parameter `step`, `exp_avg`, `exp_avg_sq` in the checkpoint | read-only, secondary proxy only |
| zero_grad | `optim.zero_grad(set_to_none=True)`, so an unreached parameter has grad None and is skipped by Adam | reproduced via `autograd.grad(..., allow_unused=True)` |
| clipping | `clip_grad_norm_(model.parameters(), 1.0)` over the step's total loss; coefficient = min(1, 1.0 / (‖g‖ + 1e-6)) | global norms and clip coefficients reported |

No simplified surrogate is used; the real loss graphs are evaluated.

## 4. Batch policy (frozen)

* **Population.** The full canonical C population in canonical order: `tr.comp_idx`, the sorted bank-row order produced by `canonical_phonology_indices`.
* **Partition.** Consecutive slices of 64 → **438 batches**: 437 × 64 plus one final batch of 13 items (437 × 64 + 13 = 27,981).
* **Collation.** Each batch uses the frozen `build_batch`.
* **No selection.** No sampling, no difficulty or native-failure selection, and no cherry-picking. Every canonical C item appears exactly once.
* **Summaries.** Primary summaries use all 438 batches. A sensitivity summary over the 437 full-size batches only is also reported.
* **R-step matching.** L_align is evaluated on **exactly the same 438 batches (same items)**. R-step batches are normally frequency-weighted over all 29,571 rows, but the loss is a per-batch function of the items, so the matched evaluation is well defined.
* **Why not the historical stream order.** The historical C stream draws a seeded permutation per epoch. The canonical-order partition is used instead because every item is covered exactly once, independent of stream cursor, and the gradient functions do not depend on order across batches.

## 5. Losses measured per batch

| name | definition |
|---|---|
| `L_C_retrieval_raw` | `retrieval_loss(ŝ, semantic_bank, bank_idx, 0.10)` |
| `L_C_retrieval_effective` | `0.087 · L_C_retrieval_raw` (the actual V6 C-step loss) |
| `L_C_align_cos_term` | `1 − mean cos(ŝ, g)` |
| `L_C_align_mse_term` | `0.1 · MSE(ŝ, g)` |
| `L_C_align_unit` | `alignment_loss(ŝ, g)` = cos_term + mse_term (unit weight; **measurement, not the proposed training weight**) |
| `L_R_align_unit` | R-step `total_loss(...)["align"]` from a full model forward on the same batch |
| `abs_diff_C_align_vs_R_align` | \|L_C_align_unit − L_R_align_unit\| |

## 6. Gradient blocks (frozen)

| block | parameters |
|---|---|
| `phon_embed` | `phon_embed.weight` (shared module; counted once) |
| `ltm_encoder` | `ltm.encoder.*` |
| `to_semantic_0` | `ltm.to_semantic.0.weight`, `.bias` |
| `to_semantic_2` | `ltm.to_semantic.2.weight`, `.bias` |
| **`SEMANTIC_ENCODER_BLOCK` (SEB)** | concatenation of `ltm_encoder` + `to_semantic_0` + `to_semantic_2` |

Decoder parameters are excluded; retrieval and alignment do not reach them (VERIFIED_BY_PROBE, design pass). Parameters are identified by name and flattened in `named_parameters()` order.

**For every batch × loss × block:** L2 norm, all-finite flag, number of elements, and all-zero flag.

Losses with gradients: `g_ret_raw` (of `L_C_retrieval_raw`), `g_ret_eff = 0.087 · g_ret_raw` (computed by scaling), `g_align_unit` (of `L_C_align_unit`), `g_R_align_unit` (of `L_R_align_unit`). Secondary gradients of the two alignment components (`cos_term`, `mse_term`) are computed for the SEB only.

## 7. Direction and conflict (frozen)

Per batch, cosine similarity between **`g_ret_eff`** (λ_C-scaled; a positive scalar does not change the cosine) and **`g_align_unit`**, for `ltm_encoder`, `to_semantic_0`, `to_semantic_2` and `SEB`. It is computed when both norms are > 0 and finite; otherwise it is recorded as NA with a reason. Supplementary: the same cosine for `phon_embed`, and cos(g_ret_eff, g_cos_term) and cos(g_ret_eff, g_mse_term) for the SEB.

**Summaries per state × block**, over batches:
* mean, median, standard deviation (ddof = 1), min, max;
* quantiles **q05, q25, q75, q95** (`numpy.quantile`, linear);
* fraction cos < 0;
* fraction |cos| < 0.1. This is a **descriptive** "near-orthogonal" bin fixed now. No other conflict category is defined, and raw cosines are authoritative.

**R/C equivalence check.** For every batch and block: max abs difference between `g_align_unit` and `g_R_align_unit`, relative L2 difference ‖g_C − g_R‖ / ‖g_C‖, and cosine(g_C, g_R).

## 8. Effective scale diagnostics and the prospective reference weight

For each block, batchwise and aggregated (median, mean, q05, q95):
* `‖g_ret_eff‖` = 0.087 · ‖g_ret_raw‖
* `‖g_align_unit‖`
* `‖g_R_align_unit‖` (matched items)
* `‖0.1 · g_align_unit‖` (**reference weight, scale comparison only; this does not authorize c_align_weight = 0.1**)
* ratios `ρ_ref = ‖0.1·g_align_unit‖ / ‖g_ret_eff‖`, `‖g_align_unit‖ / ‖g_ret_eff‖` and `‖g_align_unit‖ / ‖g_R_align_unit‖`

**Global (all-parameter) C-step norms and clipping** (per batch):
* `‖∇(0.087·L_ret)‖_all`, i.e. the V6 C step, with its clip coefficient `min(1, 1/(‖·‖+1e-6))`;
* `‖∇(0.087·L_ret + 0.1·L_align)‖_all`, i.e. the hypothetical ON C step at the reference weight, with its clip coefficient;
* `‖∇(R-step total_loss excluding pool_CE)‖_all` on the matched batch (context for R-step clipping of L_align; pool_CE uses a different pseudoword batch and is excluded by definition).

**Secondary Adam-preconditioned proxy** (descriptive only; not used in the weight rule):
* Definition: `p(g) = g / (sqrt(v̂) + eps)`, with `v̂ = exp_avg_sq / (1 − β2^step)` taken per parameter from the checkpoint's shared AdamW state (β2 = 0.999, eps = 1e-8; parameters mapped by optimizer order = `model.parameters()` order, shapes asserted).
* Reported for the SEB: `‖lr_C·p(g_ret_eff)‖`, `‖lr_C·p(0.1·g_align_unit)‖`, `‖lr_R·p(g_R_align_unit)‖`, and the per-macro-cycle exposures `3·‖lr_C·p(0.1·g_align)‖` vs `1·‖lr_R·p(g_R_align)‖`.
* Limitation: it ignores first moments, the evolving second moments and clipping; it is a first-step proxy.

### 8a. Why w = 0.1 is the prospective reference (argument recorded before any gradient)

* **Historical R-step L_align:** 1 R step per 6-step macro-cycle, weight 1.0, LR 3e-5.
* **Candidate C-step alignment:** 3 C steps per macro-cycle, LR 1e-4, weight w.
* **First-order scalar exposure match:** `1 · 3e-5 · 1.0 = 3 · 1e-4 · w`, so **w = 0.1**.
* **Assumptions:** roughly comparable matched-item unit alignment gradients (tested by §7's R/C equivalence check), and no optimizer-state dynamics (only partially addressed by the secondary Adam proxy). It is a reference argument only.

## 9. Frozen weight-selection rule (recorded before real gradients)

The pilot may use **one** nonzero `c_align_weight`. There is no sweep, and no value other than 0.1 may be derived from these gradients.

**Numerical criterion for "grossly disproportionate" (PRIMARY, frozen):**
* Let `ρ_ref,b = ‖0.1·g_align_unit‖_SEB / ‖0.087·g_ret_raw‖_SEB` for batch b, and `M_state = median_b ρ_ref,b` over all 438 batches.
* w = 0.1 is **grossly disproportionate** if, in **either** state, `M_state < 0.1` or `M_state > 10`.
* **Justification (pre-result):**
  * One order of magnitude on either side is the conventional "same scale" band.
  * Below 0.1, the added alignment pressure would be under ~10% of the existing per-step C retrieval pressure on the semantic encoder, so the ON arm risks being an uninformative near-null manipulation.
  * Above 10, alignment would dominate the C step by more than an order of magnitude, so the ON arm would effectively replace rather than supplement the C objective.
  * The band is fixed now and never revised after results.

**Additional pathology conditions** (any one → REQUIRES_CENTRAL_ARBITRATION):
* any non-finite loss or gradient;
* an all-zero SEB gradient for retrieval or alignment in any batch;
* R/C equivalence failure: median relative L2 difference ‖g_C − g_R‖/‖g_C‖ over batches > 1e-4 for the SEB in either state;
* a parameter or optimizer-state hash change.

**Decision:**
* **A.** If the criterion is **not** met (0.1 ≤ M_state ≤ 10 in both states) **and** no pathology condition occurs → recommend **c_align_weight = 0.1** as the single pilot value. The R-step compatibility requirement is satisfied by construction when the R/C equivalence holds, since w = 0.1 then equals the first-order exposure match.
* **B.** Otherwise → **C_ALIGN_WEIGHT_STATUS=REQUIRES_CENTRAL_ARBITRATION**, with raw ratios reported and no alternative value proposed.

## 10. Validity and immutability gates (hard stops → no results document, blocker instead)

| gate | requirement |
|---|---|
| G1 identity | SOURCE checkpoint SHA256 and composite identities (§2); lexicon `ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66`; GloVe `91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed`; C population n = 27,981 with its hash |
| G2 parameters | `state_dict` SHA256 equals the closed value before and after |
| G3 optimizer state | SHA256 of all optimizer-state tensors and step values identical before and after; no optimizer method other than `state_dict()` is called |
| G4 no .grad | every parameter's `.grad` is None before and after (the diagnostic uses `autograd.grad` only) |
| G5 noise off | `ltm.cfg.ventral_noise == 0.0` and `wm.cfg.interference_noise == 0.0`, so the train-mode forward is deterministic |
| G6 determinism | batch 0 of each state recomputed twice gives bitwise-identical loss values and SEB gradient norms |
| G7 semantic bank | `model.ltm.semantic_bank` is bitwise equal to `normalize(bank_raw)` |

## 11. Outputs (frozen names)

Under `paper_programme/c_align_causal_pilot/gradient_diagnostic/`:
* `gradient_batch_metrics.tsv`: one row per state × batch, columns fixed by the script.
* `gradient_summary.json`: per-state summaries, rule inputs and outcome, gates.
* `gradient_diagnostic_manifest.json`: input hashes, code hashes, environment, commit.
* `SHA256SUMS`.

Plus `C_ALIGN_GRADIENT_DIAGNOSTIC_RESULTS.md` (sections STRUCTURAL_CODE_FACT / READ_ONLY_EMPIRICAL_RESULT / INTERPRETATION / NOT_ESTABLISHED).

## 12. Execution condition

Execute once, from the commit that freezes this contract and code, **only if** the historical audit does not return `C_ALIGN_HISTORY_STATUS=EXACT_PRIOR_TEST_DECISIVE`. Batches, aggregation, blocks, cosine definition and the weight rule are never changed after results.
