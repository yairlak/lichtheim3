# CENTRAL STEERING HANDOFF — C-ALIGN CAUSAL PILOT DESIGN

Workstream: LICHTHEIM3 — VENTRAL INTERFACE / C-ALIGN CAUSAL PILOT — DESIGN AND PREREGISTRATION.
Worktree `wt-c-align-design`, branch `paper-programme/c-align-causal-pilot-design`, from `6522926ea84bdff632097e15a003f1f876e3c9ef`.
Gradient-contract freeze `8427d25f32d3ada689923fcc1ca5009444056980`; the design-only commit SHA is reported in the session response.

C_ALIGN_DESIGN=COMPLETE · PILOT_CONTRACT_STATUS=DESIGN_COMPLETE_NOT_TRAINING_AUTHORIZED · RETURN_TO_CENTRAL=YES

Documents:

* `C_ALIGN_HISTORICAL_AUDIT.md`
* `C_ALIGN_GRADIENT_DIAGNOSTIC_CONTRACT.md` (sha `58ffc342…`)
* `C_ALIGN_GRADIENT_DIAGNOSTIC_RESULTS.md`
* `gradient_diagnostic/` (TSV, summary, manifest, log, SHA256SUMS)
* `C_ALIGN_CAUSAL_PILOT_CONTRACT.md`

---

## 1. HAS_EQUIVALENT_C_ALIGN_ALREADY_BEEN_TESTED=

**PARTIALLY: C_ALIGN_HISTORY_STATUS=EXACT_PRIOR_TEST_NOT_DECISIVE.**

An exact C-stream raw-GloVe alignment run was executed once: FINAL-2A, weight 1.0. It was not done in a regime that transfers to the mature interleaved-123 H512 V6 states.

## 2. HISTORICAL_EVIDENCE=

* **FINAL-2A** (`03ae097`, `final2a_run.slurm` `C_ALIGN=1.0`, Jean-Zay job 1678465 COMPLETED).
  * Design: summed J0, from scratch, seed 22, H128, 700 N-exposures, paired with FINAL-1 (c_align 0).
  * Result, from user-pasted chat evidence (primaries only on Jean-Zay, not recomputed):
    * worse isolated LTM: .791 vs .836 at 700
    * worse C top-1: .0483 vs .0559
    * similar FULL and Naming
  * Closed as "dead end / return to c_align_weight=0". The canonical SoT does not register it (PHASE06:105).
  * Not decisive: summed rather than interleaved per-task clipping, H128, early, single seed, alignment-dominated weight, no item-level interface analysis.
* **Related, not equivalent:**
  * Phase-2 C0/C3 single-task warm-start H128: unrehearsed raw alignment took LTM from .989 to .008/.014, and cosine rose while top-1 fell.
  * Phase-3 multitask: C3 with alignment in every arm, never toggled.
* **Not relevant:** FINAL-1, 3P–9E, base-123, V-series including V6. All 223 checkpoints carrying the key have c_align 0.0.

## 3. DO_RETRIEVAL_AND_RAW_ALIGNMENT_EXERT_DIFFERENT_PRESSURES=

**MOSTLY SHARED, PARTLY DISTINCT** (read-only, first-order, batch-mean). The evidence covers all 438 canonical C batches (27,981 items), W3_SRC and W4_SRC; all gates passed; params and optimizer hashes were unchanged.

* **Losses** (median):
  * L_ret_raw 3.49 / 3.59, so effective ×0.087 = 0.304 / 0.312
  * L_align_unit 0.280 / 0.290 (1−cos 0.269 / 0.278; 0.1·MSE 0.010 / 0.011)
  * L_R_align on matched items is identical (abs diff 0), and the gradient is identical (rel L2 0, cos 1). The candidate C term *is* the R-step L_align applied on C steps.
* **Gradient norms** on the semantic-encoder block (SEB = LTM encoder + to_semantic.0 + to_semantic.2, 1,304,364 params), median:
  * ‖0.087·g_ret‖ 0.147 / 0.132
  * ‖g_align_unit‖ 0.224 / 0.205
  * ‖0.1·g_align‖ 0.022 / 0.020
  * Per-block ρ_ref ranges 0.150–0.166.
  * Nonfinite 0; all-zero 0.
* **Gradient cosine** cos(0.087·g_ret, g_align) on SEB:
  * median 0.779 / 0.764; mean 0.719 / 0.700; SD 0.21 / 0.22
  * q05 0.245 / 0.218; q95 0.899 / 0.872
  * min −0.66 / −0.65; frac<0 2.1% / 3.2%; |cos|<0.1 1.8% / 2.7%
  * By block: to_semantic.2 median ≈0.78 and never negative; to_semantic.0 ≈0.70; ltm_encoder ≈0.78.
  * The distinct part is the raw-configuration MSE term: its gradient has cosine with retrieval ≈0.22 / 0.20, but it is only ≈13–16% of the alignment gradient norm.
* **Clipping:** V6 C-step global norms are 0.18 / 0.16, and ON at 0.1 gives 0.20 / 0.18, so C steps are never clipped. R steps are clipped with median coefficient ≈0.2.
* **State consistency:** W3 and W4 agree within a few percent on every summary. The earlier H128 FINAL-1 measurement gave cosine +0.31…+0.48.
* **Limitations:**
  * batch-mean gradients at one parameter point; no per-item gradients
  * CPU/torch 2.12.1 rather than CUDA/torch 2.6.0
  * no causal inference
  * co-directionality lowers but does not remove the prior of an effect

## 4. RECOMMENDED_PILOT=

**WARM_START**: paired mature continuation, W3_SRC OFF/ON and W4_SRC OFF/ON.

| dimension | A. paired mature warm-start | B. short fresh from-scratch |
|---|---|---|
| causal isolation of c_align | High. Identical checkpoint, optimizer moments, RNG, task and batch order; one differing scalar; batch-hash verified. | Lower. ON/OFF from scratch diverge early and chaotically, and single-factor attribution is confounded by convergence variance. |
| time / compute | ≈4 GPU-h training (4 × 138,900 steps) plus ≈1–2 GPU-h readouts | Reaching mature ceiling takes about 3,000–3,800u (~8–10M steps per arm); a "short" run cannot reach the regime where the deficit exists |
| answers the present mechanism (mature ŝ→decoder mismatch) | Directly, on the exact states where S0/S1 and DR3 localized it | Only indirectly; the deficit may never form the same way |
| dependence on path-dependent development | Deliberately conditioned on the actual V6 path | Tests development itself, which is path-dependent and seed-sensitive |
| interpretability of a negative result | Clear but bounded: "not repairable at w=0.1 over 50u" | Weak: a negative short run is uninformative about maturity; long runs are expensive and noisy |
| provenance preservation | Full: hashed SOURCE files with closed S0/S1/DR3 readouts, and historical OFF trajectories at the same steps | New lineage with no closed interface diagnostics |
| sensitivity to stochastic convergence | Low within pairs; drift is bounded by measured V6 variability | High: seed-level convergence variance dominates |
| relevance to eventual final from-scratch model | Indirect (repair, not development) | More direct, but only at full scale, which is not authorized |

Rationale: CENTRAL's smallest discriminative question is the mature-repair question. B is more final-model relevant but much costlier, confounded by convergence, and would require a full-ceiling-scale run (GO_FOR_FULL_CEILING_RUN=NO).

## 5. RECOMMENDED_C_ALIGN_WEIGHT=

**0.1** (single value; frozen rule A; C_ALIGN_WEIGHT_STATUS=RECOMMEND_c_align_weight=0.1).

## 6. WEIGHT_RATIONALE=

* **Prospective argument:** 1·3e-5·1.0 = 3·1e-4·w, so w = 0.1.
* **Frozen rule:** median SEB ρ_ref = ‖0.1·g_align‖/‖0.087·g_ret‖ is 0.1507 (W3) and 0.1560 (W4), inside [0.1, 10]; no pathology. No other value was derived.
* **Caveats:**
  1. Because R steps are clipped (≈0.2) and C steps are not, 0.1 gives about 5× the effective per-cycle alignment pressure the R stream currently delivers.
  2. 0.1 sits near the lower edge of the band in ρ terms, and the added pressure is mostly retrieval-parallel.
  3. 0.1 is 10× below FINAL-2A's harmful 1.0.
* Any different weight is CENTRAL's arbitration, not a design output.

## 7. FIXED_TRAINING_BUDGET=

**50u = 138,900 optimizer steps per arm**, which is:

* 23,150 macro-cycles
* 23,150 R updates (≈50.1 R epochs)
* 46,300 N updates (≈100.2 N epochs)
* 69,450 C updates (≈158.9 C epochs)

End steps are W3 10,764,750 and W4 8,584,020.

Justification:

* The native-LTM count is plastic on a 5u scale in V6 (lag-1 SD 132 / 189).
* Lag-10 (50u) V6 drift is empirically measured (|Δ| q95 291 / 340).
* The added pressure is small, so shorter horizons risk a false null.
* 50u is 1.3% / 1.6% of the training history, so it stays a repair test.
* Compute is small.

No adaptive extension, no `--stop-at-ceiling`.

## 8. EVALUATION_CADENCE=

* Baseline (k=0, the SOURCE file) plus every 13,890 steps (5u) for k = 1…10, on the V6 detector grid.
* All readouts are post hoc on saved checkpoints with one frozen evaluation script, so evaluation cannot affect training.
* Primary contrast at k = 10 only; k = 8–10 feed a frozen robustness condition; other points are trajectory diagnostics.
* Readouts:
  * FULL canonical and free-AR repetition, Naming, strict C top-1, gate stats
  * native isolated-ventral (free-AR primary, canonical) and WM-only repetition
  * S0 and a contemporaneous S1c (the checkpoint's own top-1, raw prototype injected) with gap G
  * semantic geometry (cosine, norm, norm ratio, MSE, margins)

## 9. SUCCESS_FAILURE_RULES=

Everything is frozen from pre-pilot information only. Δ = ON−OFF errors at k=10, per pair.

* **D_LTM = 560.** This is the maximum |difference| between any two V6 points 5u–50u apart, both seeds, 444 pairs. Free-AR and canonical agree within 1 item. It is ≈3.8σ (W3) / 2.9σ (W4) of two-trajectory drift, and 17.7% / 15.6% of the closed S0−S1 deficit (3157 / 3596).
* **Pair classification:**
  * **IMPROVE:** ΔLTM_freeAR ≤ −560, ΔLTM_canonical ≤ −560, and mean Δ over k=8–10 ≤ −280.
  * **HARM:** Δ ≥ +560 and mean Δ over k=8–10 ≥ +280.
  * **NULL:** otherwise.
* **Preservation floors (ON−OFF, k=10):** FULL canonical ≤ 7; FULL free-AR ≤ 7; WM ≤ 1; Naming ≤ 1; C ≤ 13. Each is the full pre-pilot range or maximum drift. Exceeding any floor is DAMAGE.
* **Collapse limits** (any point): FULL > 70, WM > 10, Naming > 10, C > 510, LTM > 2× the seed's pre-pilot maximum.
* **Outcome order:**
  * **INVALID:** a validity gate fails or an OFF arm collapses.
  * **T4:** nonfinite values or collapse in an ON arm, or IMPROVE in one pair and HARM in the other.
  * **T3:** both IMPROVE with DAMAGE.
  * **T1:** both IMPROVE, no DAMAGE.
  * **T2:** otherwise, sub-labelled T2_NULL, T2_NOT_REPLICATED or T2_HARM, with a DAMAGE flag.
* The gap/geometry readouts are secondary. A T1 without gap movement is labelled MECHANISM_UNCONFIRMED.

## 10. SHOULD_CENTRAL_AUTHORIZE_SINGLE_TRAINING_PILOT=

**CONDITIONAL**

## 11. WHY=

* **For:**
  * The question is not settled by history (FINAL-2A is not transferable).
  * The intervention is an existing, tested code path.
  * The weight passed a prospectively frozen proportionality rule.
  * The design is cheap (≈5–6 GPU-h), fully matched, and has pre-pilot-derived thresholds.
  * T1, T2, T3 and T4 are each interpretable, and a T2 closes the mature-repair hypothesis at w=0.1 over 50u.
* **Conditions:**
  1. CENTRAL first authorizes a separate preregistered **driver-amendment + preflight** step. The current `load_state_dict` refuses any `c_align_weight` change unconditionally, so a minimal explicit declaration (0.0→0.1 only, recorded in `phase_transitions`) plus batch-hash logging is required. It must pass toy-state bitwise equivalence at w=0 and V3–V10 gates before any real-state update.
  2. CENTRAL accepts the design-time priors: FINAL-2A was negative, and gradients are mostly co-directional, so T2 is a plausible outcome. It commits in advance not to tune after T2/T3/T4.
  3. Strict determinism is available on the training device; otherwise HARD STOP.
* **Against (for CENTRAL to weigh):**
  * the expected effect size may be small at w=0.1;
  * a negative warm-start result does not speak to development;
  * a positive result does not authorize any recipe change.

## 12. NOT_ESTABLISHED=

* Any causal effect of C-step alignment on the native interface, repetition, Naming, C or routing.
* Transfer of FINAL-2A's harm to this regime; FINAL-2A numbers are not recomputed from primaries.
* Per-item gradient conflict, since only batch-mean gradients were measured.
* Numeric identity of gradient summaries on CUDA/torch 2.6.0.
* Sufficiency of 50u for a slow effect, or optimality of 0.1.
* Validity of S1c as an upper bound at future checkpoints (flagged per checkpoint).
* Generalization beyond seeds 19 and 20, or to from-scratch development.

---

## Confirmations

TRAINING_RUN=NO
OPTIMIZER_STEP_COUNT=0
ARCHITECTURE_CHANGED=NO
GATE_CHANGED=NO
ATTRACTOR_IMPLEMENTED=NO
YAIR_FLAG_IMPLEMENTED=NO
LESIONING_RUN=NO
FULL_CEILING_RUN=NO

Also:

* The read-only gradient diagnostic ran exactly once, after the freeze; params and optimizer-state hashes were unchanged for both states.
* No SOURCE checkpoint, closed result or closed worktree was modified.
* No training launcher was created.
* YAIR_SEMANTIC_ATTRACTOR=LATER_IF_T_FAILS (not implemented, not used).

---

## READY_TO_PASTE_CENTRAL_PROMPT

```
CENTRAL STEERING — ARBITRATION REQUEST
LICHTHEIM3 — VENTRAL INTERFACE / C-ALIGN CAUSAL PILOT — DESIGN PACKAGE RETURNED

Status: C_ALIGN_DESIGN=COMPLETE; PILOT_CONTRACT_STATUS=DESIGN_COMPLETE_NOT_TRAINING_AUTHORIZED;
TRAINING_RUN=NO; OPTIMIZER_STEP_COUNT=0; no architecture/gate/attractor/flag/lesion/full-ceiling work.
Branch paper-programme/c-align-causal-pilot-design (from 6522926e); gradient-contract freeze 8427d25f;
design-only commit: <SHA from session response>.

REAL RESULTS

1. Historical audit: C_ALIGN_HISTORY_STATUS=EXACT_PRIOR_TEST_NOT_DECISIVE.
   FINAL-2A (03ae097, Jean-Zay job 1678465) ran C-step raw-GloVe alignment at w=1.0 in a SUMMED, H128,
   from-scratch, single-seed regime. It was worse than FINAL-1 on isolated LTM (.791 vs .836 @700) and C top-1
   (.0483 vs .0559); the numbers are chat evidence and the primaries are not local. It does not transfer to
   mature interleaved-123 H512 states.
   Related, not equivalent: Phase-2 C0/C3 (unrehearsed alignment destroyed H128 ventral repetition) and
   Phase-3 multitask (alignment never toggled). All V-series checkpoints have c_align 0.0.

2. Read-only gradient diagnostic (frozen before gradients; executed once; all gates pass; params and
   optimizer hashes unchanged). States W3_SRC and W4_SRC; all 438 canonical C batches (27,981 items).
   - The C-step alignment term is identical to the R-step L_align on matched items (loss diff 0,
     gradient cos 1).
   - Losses (median): L_ret_eff 0.304/0.312; L_align_unit 0.280/0.290 (1-cos 0.269/0.278; 0.1*MSE 0.010/0.011).
   - Semantic-encoder-block norms (median): ||0.087 g_ret|| 0.147/0.132; ||g_align|| 0.224/0.205;
     ||0.1 g_align|| 0.022/0.020. Nonfinite 0, zero 0.
   - cos(0.087 g_ret, g_align) on SEB: median 0.779/0.764, q05 0.245/0.218, frac<0 2.1%/3.2%.
     Pressures are mostly shared. The distinct part is the raw-norm MSE term (cos ~0.2), which is only
     ~13-16% of the alignment gradient.
   - C steps are never clipped (global norm ~0.16-0.20 even with ON at 0.1). R steps are clipped
     (median coefficient ~0.2).
   - Frozen rule: median rho_ref = ||0.1 g_align||/||0.087 g_ret|| = 0.1507 (W3), 0.1560 (W4), inside
     [0.1, 10] => RECOMMEND c_align_weight=0.1. No other value was derived.

3. Pilot design (C_ALIGN_CAUSAL_PILOT_CONTRACT.md, 22 sections): paired mature WARM-START,
   W3_SRC OFF/ON and W4_SRC OFF/ON.
   - The only difference within a pair is c_align_weight 0 vs 0.1.
   - Exact continuation of the shared AdamW state, RNG, cursors and schedule; strict determinism;
     batch-hash identity gate.
   - Budget 50u = 138,900 steps = 23,150 macro-cycles = 23,150 R / 46,300 N / 69,450 C updates per arm.
   - Evaluation at 0 and every 5u to 50u, post hoc on checkpoints; primary contrast at 50u.
   - T rules from pre-pilot V6 drift: D_LTM=560 (max V6 5u-50u point difference). IMPROVE requires
     ON-OFF <= -560 in free-AR and canonical, and a mean over the last 3 points <= -280.
     Floors (ON-OFF): FULL<=7, WM<=1, Naming<=1, C<=13.
     T1 = both pairs IMPROVE and no damage; T3 = both IMPROVE with damage;
     T4 = instability/collapse or opposite pair effects; T2 = otherwise (NULL / NOT_REPLICATED / HARM).
   - Prerequisite: the existing resume guard refuses any c_align change, so a separate preregistered
     minimal driver amendment and preflight are needed before any real-state update.
   - Recommendation: SHOULD_CENTRAL_AUTHORIZE_SINGLE_TRAINING_PILOT=CONDITIONAL.

CENTRAL IS ASKED TO:
1. accept or reject the historical audit (EXACT_PRIOR_TEST_NOT_DECISIVE);
2. accept or reject the read-only gradient diagnostic and its interpretation (mostly shared pressure;
   T2 plausible);
3. accept or reject WARM_START over FROM_SCRATCH;
4. accept or reject the single c_align_weight=0.1 (or arbitrate another single value, knowingly);
5. accept, modify or reject the fixed 50u budget;
6. accept, modify or reject the T1-T4 criteria, thresholds and preservation floors;
7. decide whether the ONE training pilot is authorized, including whether to authorize first the
   separate driver-amendment + real-state preflight workstream (no optimizer step until its gates pass);
8. confirm that after the pilot NO tuning or follow-up (weight, schedule, LR, duration, decoder weight,
   gate, start convention, from-scratch rescue, attractor) happens without a new CENTRAL arbitration.
```
