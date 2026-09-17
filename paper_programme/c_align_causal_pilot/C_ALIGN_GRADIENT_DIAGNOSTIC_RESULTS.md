# C_ALIGN GRADIENT DIAGNOSTIC — RESULTS

GRADIENT_DIAGNOSTIC_CONTRACT_STATUS=FROZEN_BEFORE_REAL_GRADIENTS (contract sha256 `58ffc34215890ebf113901585fe490e048de3246561d5313777ebe32bfa88cbe`, frozen at commit `8427d25f32d3ada689923fcc1ca5009444056980`)
GRADIENT_DIAGNOSTIC_EXECUTION=ONCE, from clean HEAD `8427d25f`, exit 0, `ALL_GATES_PASS`
C_ALIGN_WEIGHT_STATUS=RECOMMEND_c_align_weight=0.1 (frozen rule A)
OPTIMIZER_STEP_COUNT=0 · PARAMETER_UPDATES=0 · TRAINING_RUN=NO

Evidence labels: **[STRUCTURAL_CODE_FACT]**, **[READ_ONLY_EMPIRICAL_RESULT]**, **[INTERPRETATION]**, **[NOT_ESTABLISHED]**. No causal training claim is made from gradients.

## 1. Execution record

* [READ_ONLY_EMPIRICAL_RESULT] Command (worktree root): `PYTHONDONTWRITEBYTECODE=1 python3 scripts/c_align_design/c_align_gradient_diagnostic.py --out paper_programme/c_align_causal_pilot/gradient_diagnostic`. Precondition met: `C_ALIGN_HISTORY_STATUS=EXACT_PRIOR_TEST_NOT_DECISIVE` (not `…_DECISIVE`).
* Environment: CPU, torch 2.12.1, Python 3.11.15, deterministic algorithms on. (V6 trained on CUDA V100, torch 2.6.0; see §9.)
* States: W3_SRC (source `a5f21de9…`, params `4a8d4807…`), W4_SRC (source `0657f410…`, params `004dda2d…`). W3_REP/W4_REP not used.
* Batches: canonical C population (N=27,981, sha `10c2f06e…`) in `tr.comp_idx` order, consecutive slices of 64 → 438 batches per state (437 full + 1 of 13); R-align on the same items.
* Gates: G1 data/identity PASS; G2 params sha before = after = closed record PASS (both); G3 optimizer-state sha before = after PASS (W3 `86791e6e…`, W4 `77432134…`); G4 no `.grad` PASS; G5 noise off PASS; G6 batch-0 recompute determinism PASS; G7 `semantic_bank == normalize(bank_raw)` PASS.
* Outputs (`gradient_diagnostic/`, sha256 in `SHA256SUMS`): `gradient_batch_metrics.tsv` (876 rows × 123 columns) `b888a1e6…`, `gradient_summary.json` `87ee4118…`, `gradient_diagnostic_manifest.json` `dc701c90…` (`optimizer_step_calls: 0`, `parameter_updates: 0`), `run_stdout.log` `3fc08892…` (read-only copy also at `archives/c_align_design_20260917/gradient_diagnostic_run.log`).

## 2. Structural facts pinned for interpretation

* [STRUCTURAL_CODE_FACT] V6 C step (`train_joint_scratch.py:1297–1302`): `loss = 0.087·retrieval_loss(ŝ, semantic_bank, bank_idx, τ=0.1) + c_align_weight·alignment_loss(ŝ, c["semantic"])`, `c_align_weight=0.0` in V6. `c["semantic"] = bank_raw[bank_idx]` (raw, unnormalized GloVe).
* [STRUCTURAL_CODE_FACT] `alignment_loss = (1 − mean cos) + 0.1·MSE` (`losses.py:47–50`). The R-step `L_align` is the same function of the same ŝ (from `ltm.to_semantic(h[-1])`) and the same raw GloVe target, weight 1.0.
* [STRUCTURAL_CODE_FACT] Retrieval and alignment gradients reach only `phon_embed`, `ltm.encoder`, `ltm.to_semantic.{0,2}`. No decoder, `sem_to_h0`, motor, WM or gate parameter.
* [STRUCTURAL_CODE_FACT] Cosine convention: g_ret_eff = 0.087·∇L_ret_raw. Positive scaling does not change cosine.

## 3. Loss scale (median [q05, q95] over 438 batches)

| quantity | W3_SRC | W4_SRC |
|---|---|---|
| L_C_retrieval_raw | 3.492 [2.991, 3.730] | 3.591 [3.050, 3.831] |
| L_C_retrieval_effective (×0.087) | 0.304 [0.260, 0.325] | 0.312 [0.265, 0.333] |
| L_C_align_unit (total) | 0.280 [0.197, 0.306] | 0.290 [0.204, 0.317] |
| — 1 − cos term | 0.269 [0.188, 0.294] | 0.278 [0.194, 0.304] |
| — 0.1·MSE term | 0.0101 [0.0080, 0.0143] | 0.0114 [0.0088, 0.0165] |
| L_R_align_unit (matched items) | 0.280 (identical) | 0.290 (identical) |
| |L_C_align − L_R_align| | 0 (max 0) | 0 (max 0) |

* [READ_ONLY_EMPIRICAL_RESULT] C alignment and R alignment are **identical** on matched items: loss absolute difference 0 in every batch; per-block gradient relative L2 difference 0, gradient cosine 1 ± 2e-14. This matches the structural identity: the candidate C-step term is the existing R-step L_align applied on C steps.
* [INTERPRETATION] At unit weight, the alignment loss is about the same size as the effective retrieval loss (0.28 vs 0.30). About 96% of it is the cosine term.

## 4. Gradient norms (median over 438 batches)

| block | numel | ‖g_ret_raw‖ W3/W4 | ‖g_ret_eff‖ W3/W4 | ‖g_align_unit‖ = ‖g_R_align_unit‖ W3/W4 | ‖0.1·g_align‖ W3/W4 |
|---|---|---|---|---|---|
| phon_embed | 2,688 | 1.159 / 0.973 | 0.1008 / 0.0847 | 0.1496 / 0.1393 | 0.0150 / 0.0139 |
| ltm_encoder | 887,808 | 1.591 / 1.417 | 0.1384 / 0.1233 | 0.2104 / 0.1903 | 0.0210 / 0.0190 |
| to_semantic.0 | 262,656 | 0.345 / 0.347 | 0.0300 / 0.0302 | 0.0493 / 0.0501 | 0.0049 / 0.0050 |
| to_semantic.2 | 153,900 | 0.413 / 0.391 | 0.0359 / 0.0340 | 0.0540 / 0.0515 | 0.0054 / 0.0052 |
| **SEB** | 1,304,364 | 1.687 / 1.519 | 0.1468 / 0.1322 | 0.2236 / 0.2047 | 0.0224 / 0.0205 |

* [READ_ONLY_EMPIRICAL_RESULT] Every block, loss and state: nonfinite count 0, all-zero batches 0.
* [READ_ONLY_EMPIRICAL_RESULT] SEB alignment sub-components (median): ‖∇(1−cos)‖ 0.217 (W3) / 0.199 (W4); ‖∇(0.1·MSE)‖ 0.0295 / 0.0336.
* [READ_ONLY_EMPIRICAL_RESULT] Global all-parameter norms (median): V6 C step 0.177 (W3) / 0.158 (W4); hypothetical ON C step at w=0.1 0.199 / 0.177. Clip coefficient at max-norm 1.0 is 1.000 in all 438 batches for both (C steps are never clipped). R step total excluding pool: 4.55 / 5.01, clip coefficient median 0.220 / 0.199 (q05 0.033, q95 1.0).

## 5. Scale ratios

| ratio (median [min, max]) | W3_SRC | W4_SRC |
|---|---|---|
| ‖g_align_unit‖/‖g_ret_eff‖, SEB | 1.507 [1.277, 2.006] | 1.560 [1.310, 2.192] |
| **ρ_ref = ‖0.1·g_align‖/‖g_ret_eff‖, SEB (frozen rule)** | **0.1507** [0.1277, 0.2006] | **0.1560** [0.1310, 0.2192] |
| ρ_ref, ltm_encoder | 0.1503 | 0.1562 |
| ρ_ref, to_semantic.0 | 0.1629 | 0.1656 |
| ρ_ref, to_semantic.2 | 0.1505 | 0.1517 |
| ρ_ref, phon_embed | 0.1516 | 0.1592 |
| ‖g_align_unit‖/‖g_R_align_unit‖ (all blocks) | 1.000 | 1.000 |
| full-size-437 sensitivity: ρ_ref SEB median | 0.1507 | 0.1559 |

* [READ_ONLY_EMPIRICAL_RESULT] Adam-denominator proxy on SEB (`g/(√v̂+ε)`, from the checkpoint's own `exp_avg_sq`), median norms:
  * C_ret_eff 0.0394 (W3) / 0.0354 (W4)
  * C_align at w=0.1: 0.0059 / 0.0054
  * R_align unit: 0.0178 / 0.0161
* [STRUCTURAL_CODE_FACT] The per-macro-cycle proxies "3 × LR_C × 0.1 × C_align" and "1 × LR_R × 1.0 × R_align" are equal by construction: identical gradients, identical denominator, and 3·1e-4·0.1 = 3e-5. That equality restates the prospective 0.1 argument; it is not an independent empirical confirmation.

## 6. Gradient direction: cos(g_ret_eff, g_align_unit)

| block | state | mean | median | SD | min | max | q05 | q25 | q75 | q95 | frac<0 | frac abs<0.1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ltm_encoder | W3 | 0.717 | 0.782 | 0.226 | −0.757 | 0.969 | 0.212 | 0.700 | 0.842 | 0.905 | 0.025 | 0.018 |
| ltm_encoder | W4 | 0.697 | 0.767 | 0.234 | −0.734 | 0.923 | 0.194 | 0.692 | 0.821 | 0.878 | 0.037 | 0.021 |
| to_semantic.0 | W3 | 0.648 | 0.700 | 0.189 | −0.671 | 0.828 | 0.251 | 0.640 | 0.748 | 0.790 | 0.023 | 0.023 |
| to_semantic.0 | W4 | 0.643 | 0.704 | 0.201 | −0.651 | 0.827 | 0.229 | 0.639 | 0.743 | 0.787 | 0.032 | 0.023 |
| to_semantic.2 | W3 | 0.758 | 0.784 | 0.090 | 0.280 | 0.940 | 0.562 | 0.746 | 0.808 | 0.831 | 0.000 | 0.000 |
| to_semantic.2 | W4 | 0.752 | 0.778 | 0.095 | 0.146 | 0.938 | 0.562 | 0.743 | 0.805 | 0.828 | 0.000 | 0.000 |
| **SEB** | W3 | 0.719 | **0.779** | 0.212 | −0.661 | 0.967 | 0.245 | 0.704 | 0.836 | 0.899 | 0.021 | 0.018 |
| **SEB** | W4 | 0.700 | **0.764** | 0.221 | −0.655 | 0.917 | 0.218 | 0.697 | 0.815 | 0.872 | 0.032 | 0.027 |
| phon_embed (suppl.) | W3 | 0.678 | 0.784 | 0.305 | −0.853 | 0.966 | −0.043 | 0.667 | 0.855 | 0.922 | 0.059 | 0.032 |
| phon_embed (suppl.) | W4 | 0.660 | 0.756 | 0.297 | −0.841 | 0.948 | −0.015 | 0.642 | 0.821 | 0.902 | 0.055 | 0.023 |

* [READ_ONLY_EMPIRICAL_RESULT] SEB decomposition (median): cos(g_ret_eff, ∇cos-term) is 0.769 (W3) / 0.753 (W4). cos(g_ret_eff, ∇0.1·MSE-term) is 0.224 / 0.197.
* [READ_ONLY_EMPIRICAL_RESULT] full-size-437 sensitivity: SEB median cosine 0.779 (W3) / 0.764 (W4), unchanged.
* [READ_ONLY_EMPIRICAL_RESULT] Historical consistency: FINAL-1 H128 (`grad_interference_audit.py`) reported C_align vs C_retrieval cosine +0.31…+0.48. The mature H512 states are more aligned. The estimates differ in state, width and batch policy and are not strictly comparable.

## 7. Frozen weight rule outcome

* [READ_ONLY_EMPIRICAL_RESULT] M = median over 438 batches of ρ_ref (SEB): **W3 0.1507, W4 0.1560**. Both lie inside [0.1, 10], so neither is grossly disproportionate.
* [READ_ONLY_EMPIRICAL_RESULT] No pathology: nonfinite 0, all-zero 0, R/C equivalence median relative L2 = 0 (≤ 1e-4), all gates pass.
* **Outcome (rule A): RECOMMEND c_align_weight = 0.1.** No other value was derived, examined or proposed.

## 8. Interpretation

1. [INTERPRETATION] **Retrieval and raw alignment exert largely shared, not opposite, first-order pressure on the semantic encoder at the mature SOURCE states.**
   * Batch-mean gradients agree strongly: SEB median cosine is about 0.77, and only 2–3% of batches have negative cosine.
   * The part of alignment pressure most distinct from retrieval is the raw-configuration MSE term (cosine with retrieval about 0.2). Its gradient is small: about 13–16% of the alignment gradient norm at unit weight.
   * At w=0.1, rough arithmetic from the medians gives:
     * The orthogonal-to-retrieval component of the added gradient ≈ 0.15 × √(1−0.77²) ≈ 0.10 of ‖g_ret_eff‖.
     * The MSE-term part ≈ 0.1 × 0.03 / 0.14 ≈ 2% of ‖g_ret_eff‖.

   These are not frozen metrics, only approximate arithmetic on frozen medians. They lower, but do not remove, the prior that a w=0.1 pilot will show a material effect (T2 is a plausible outcome).
2. [INTERPRETATION] **The C step is never clipped.**
   * V6 C-step global norm ≈ 0.16–0.18 ≪ 1, and ON at 0.1 is about 0.18–0.20, still unclipped. The ON term therefore enters the Adam update unattenuated.
   * R steps are clipped heavily (median coefficient ≈ 0.2), so the existing R-step L_align is effectively scaled down about 5× by clipping.
   * The first-order "exposure match" in the 0.1 argument therefore gives the C-step alignment roughly 5× the effective per-cycle alignment pressure the R stream currently delivers. That is ignoring Adam second-moment coupling, and it is a weakness of the matching argument that CENTRAL should see.
   * It does not change the frozen rule outcome.
3. [INTERPRETATION] At w=0.1 the added term is small relative to retrieval (≈15% in norm, ≈15% in the Adam proxy). It is also far from FINAL-2A's alignment-dominated w=1.0 (≈1.5× retrieval in norm on these states). This is consistent with a minimal, non-dominating single intervention.
4. [INTERPRETATION] W3 and W4 agree closely on every summary (ρ_ref 0.151 vs 0.156; cosine 0.78 vs 0.76), so there is no state-specific pathology.

## 9. Not established

* [NOT_ESTABLISHED] Any causal effect of C-step alignment on native ŝ→decoder compatibility, repetition, Naming, C or routing. Gradients at one parameter point do not predict the training trajectory under Adam momentum, the 1:2:3 interleave, or drift.
* [NOT_ESTABLISHED] Item-level or failure-subgroup pressure. Batch-mean gradient cosines can hide opposing per-item components; per-item gradients were not measured, by contract.
* [NOT_ESTABLISHED] Exact numerical transfer to CUDA/torch 2.6.0. The diagnostic ran on CPU/torch 2.12.1; norms and cosines are expected to agree to floating-point tolerance, but this was not re-measured on the training device.
* [NOT_ESTABLISHED] That 0.1 is optimal or sufficient. It is the single prospectively motivated reference value that passed a coarse proportionality rule, not a tuned value.
* [NOT_ESTABLISHED] Transfer of FINAL-2A's harm (summed, H128, w=1.0) to this regime.
