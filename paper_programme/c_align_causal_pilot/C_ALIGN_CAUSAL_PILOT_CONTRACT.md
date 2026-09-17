# C_ALIGN CAUSAL PILOT — PROSPECTIVE CONTRACT

## 1. STATUS

PILOT_CONTRACT_STATUS=AMENDED_CONDITIONALLY_AUTHORIZED
(prior status at design commit `9f36f2e0cf913ffe5f453a82ca43c20fa4ec824d`: DESIGN_COMPLETE_NOT_TRAINING_AUTHORIZED)

**CENTRAL arbitration recorded in this amendment (binding):**

* C_ALIGN_DESIGN_STATUS=ACCEPTED_WITH_MODIFICATION
* HISTORICAL_AUDIT_STATUS=ACCEPTED_WITH_QUALIFICATION
* GRADIENT_DIAGNOSTIC_STATUS=ACCEPTED_WITH_QUALIFICATION
* PILOT_FORM=WARM_START · C_ALIGN_WEIGHT=0.1 · TRAINING_BUDGET=ACCEPT_50U
* SUCCESS_FAILURE_CONTRACT=ACCEPTED_WITH_MODIFICATION (terminology only; every threshold is unchanged)
* GO_FOR_DRIVER_AMENDMENT_AND_PREFLIGHT=YES
* GO_FOR_SINGLE_C_ALIGN_TRAINING_PILOT=CONDITIONAL_YES — effective if and only if every pre-training prerequisite in §19 passes
* GO_FOR_ATTRACTOR=NO · GO_FOR_GATE_CHANGE=NO · GO_FOR_LESIONING=NO · GO_FOR_FULL_CEILING_RUN=NO
* FINAL_2A_STATUS=NEGATIVE_PRIOR_NOT_DECISIVE (carried, not reinterpreted)

**What this amendment changed:** outcome terminology (§17), prospectively frozen descriptive T2 sublabels (§17.4), and this authorization block. **Nothing else.** D_LTM = 560, D_LTM_ROBUST = 280, preservation floors, collapse limits, budget, evaluation cadence, start states, optimizer policy, weight, and readouts are unchanged from `9f36f2e0`.

**Interpretive boundary (CENTRAL, binding).** This pilot tests only whether adding C-step raw-GloVe alignment at w=0.1 improves the already-mature native ventral interface over 50u relative to a matched OFF continuation. It does not test whether C updates caused the deficit, whether 0.1 is optimal, whether C-align should have been present from step 0, from-scratch reachability, the need for an attractor, or the general usefulness of semantic refinement. A negative result is bounded to w=0.1, this mature warm-start design, and this fixed 50u horizon.

* No optimizer step is authorized until the §19 gates pass (see §22).

Lineage:

* Design branch `paper-programme/c-align-causal-pilot-design`, created from closed directional-dose results `6522926ea84bdff632097e15a003f1f876e3c9ef`.
* Gradient-diagnostic freeze `8427d25f32d3ada689923fcc1ca5009444056980`.
* Inputs:
  * `C_ALIGN_HISTORICAL_AUDIT.md`
  * `C_ALIGN_GRADIENT_DIAGNOSTIC_RESULTS.md`
  * closed ventral-interface S0–S3 results (`wt-ventral-interface`, `0f25b5b8`)
  * closed directional-dose results (DR3)
  * pre-pilot V6 continued-training detector trajectories (`probe_v6_collect_20260912/s19_metrics.tsv` and `s20_metrics.tsv`)

## 2. PRIMARY HYPOTHESIS

**H_C (hypothesis, not an established cause):**

* C updates are frequent and optimize normalized retrieval geometry (0.087·CE at τ=0.1).
* They may therefore maintain lexical retrieval identity without preserving the raw-GloVe semantic configuration that the isolated ventral decoder handles most reliably.
* Activating the existing C-step term `c_align_weight · alignment_loss(ŝ, raw GloVe)` should reduce the native ŝ→decoder compatibility deficit, relative to a matched OFF continuation.

**Question tested:**

* The warm-start form: "Can adding C-step raw-GloVe alignment causally reduce the already-present mature ŝ→decoder compatibility mismatch over a fixed 50u horizon, without damaging core tasks?"
* Not tested: developmental effects from scratch.

**Pre-pilot priors CENTRAL should weigh (from the design package, not results):**

* FINAL-2A (w=1.0, summed, H128, from scratch) was negative, with worse LTM and C.
* At the mature SOURCE states, retrieval and alignment gradients are largely co-directional (SEB median cosine ≈0.77). At w=0.1 the added gradient is ≈15% of the retrieval gradient norm, of which the retrieval-orthogonal part is roughly ≈10% and the raw-norm (MSE) part is ≈2%.
* T2 is therefore a plausible outcome. The pilot is designed so that T2 is interpretable and closes the question over this horizon.

## 3. HISTORICAL ARCHAEOLOGY RESULT

**C_ALIGN_HISTORY_STATUS=EXACT_PRIOR_TEST_NOT_DECISIVE**

* The exact prior test is FINAL-2A (`03ae097`, Jean-Zay job 1678465).
* It fails transfer criterion (6): summed schedule, H128, early from-scratch, single seed, weight 1.0, primaries not locally recoverable.
* Related but not equivalent: Phase-2 C0/C3 (unrehearsed alignment destroyed H128 ventral repetition) and Phase-3 multitask (alignment never toggled).
* Consequence for design:
  * the pilot is not redundant;
  * preservation floors are strict (§18);
  * the weight is non-dominating (§6–7).

## 4. AUTHORITATIVE START STATES

| state | file (sha256) | global step | u | params state_dict sha | optimizer_state sha (diagnostic G3) |
|---|---|---|---|---|---|
| W3_SRC | `probe_v6_completion_20260912/sources/s19_step_10625850.pt` (`a5f21de9edb4b2090cb144c666ca4bd8e1f0572333000facb13f6c51cfaad76c`) | 10,625,850 | 3825 | `4a8d4807f1a1928faebebf696d41f4c222b3997b52432d7015fc038881eac785` | `86791e6e24298768fb9141a4984200c4b1907089e07e81a55e4a8999843fc9ad` |
| W4_SRC | `probe_v6_completion_20260912/sources/s20_step_08445120.pt` (`0657f41030e54eac11816116af6c76b66d597c393fc72579a3755401c9dc79f3`) | 8,445,120 | 3040 | `004dda2d9a846ce194d4b0b3f7adc408d5c9d909aadbe413a35e96ceaa411974` | `77432134e128f9109ac30fa7b01ec903cf46e8dadebed55e55357a3dcaf09fd2` |

* Both are unmodified V6 SOURCE checkpoints, pre-Arm-A. Arm-A heads (W3_REP/W4_REP) are **not** used and are not a factor.
* Both steps are multiples of 6 (macro-cycle boundary) and of 13,890 (5u detector grid).
* Recipe carried in the checkpoints:
  * j0, `interleaved_123`, shared AdamW (β 0.9/0.999, ε 1e-8, wd 1e-5)
  * task LRs R 3e-5 / N 3e-5 / C 1e-4
  * loss weights rep 1.0, align 1.0, dec 0.5, wm 0.5, gate 0.05; pool 0.5
  * λ_N 1.0, λ_C 0.087, τ 0.1, `c_align_weight` 0.0
  * batch 64, grad clip 1.0, final_full populations (C 27,981, sha `10c2f06e…`)
  * widths WM 128 / ENC 512 / DEC 512
* Closed frozen readouts at these exact states (all counts are errors):

| state | native LTM S0 (free-AR = canonical) | S1 raw-retrieved | S0−S1 deficit | C top-1 | WM | FULL | Naming |
|---|---|---|---|---|---|---|---|
| W3_SRC | 3199 | 42 | 3157 | 34 | 0 | 0 | 0 |
| W4_SRC | 3647 | 51 | 3596 | 42 | 1 | 0 | 0 |

## 5. OFF / ON FACTOR

One factor, two levels, two matched pairs, four arms:

| arm | start | c_align_weight |
|---|---|---|
| W3_OFF | W3_SRC | 0.0 |
| W3_ON | W3_SRC | 0.1 |
| W4_OFF | W4_SRC | 0.0 |
| W4_ON | W4_SRC | 0.1 |

The only intended difference within a pair is `c_align_weight`. On C steps this adds `0.1·alignment_loss(ŝ, bank_raw[bank_idx])` to `0.087·retrieval_loss`, through the existing code path `train_joint_scratch.py:1297–1302`. Everything else is identical: start state, optimizer state, RNG, task and batch order, LR, schedule, budget, clip, device and software, and evaluation points. No third weight, no Arm-A factor, no gate/architecture/decoder-weight change.

## 6. SINGLE c_align_weight

**c_align_weight = 0.1** (ON arms). No sweep, no alternative value.

## 7. WEIGHT JUSTIFICATION

* **Prospective argument (recorded before gradients).**
  * Existing R-step L_align: 1 R step per macro-cycle × LR 3e-5 × weight 1.0.
  * Candidate: 3 C steps × LR 1e-4 × w.
  * Matching gives w = 0.1.
* **Frozen read-only rule** (contract `58ffc342…`):
  * ρ_ref = ‖0.1·g_align‖/‖0.087·g_ret‖ on the semantic-encoder block, median over all 438 canonical C batches.
  * Observed: W3 0.1507, W4 0.1560, both inside [0.1, 10].
  * No pathology; C- and R-alignment gradients are identical on matched items.
  * Outcome: RECOMMEND 0.1 (rule A).
* **Caveats for CENTRAL** (interpretation, not reasons to change the value):
  1. C steps are never clipped, while R steps are clipped heavily (median coefficient ≈0.2). The exposure match therefore delivers roughly 5× the effective per-cycle alignment pressure of the current R stream.
  2. The added gradient is mostly co-directional with retrieval.
  3. 0.1 is 10× smaller than FINAL-2A's harmful 1.0.

  Any different value requires CENTRAL arbitration, never a post-hoc choice.

## 8. OPTIMIZER-STATE POLICY

* **Exact continuation.** Both SOURCE checkpoints carry exactly one shared AdamW `optimizer_state_dict` (V6 preflight: `optimizer_policy == shared_adamw`, `optimizer_states` absent; 29 parameter tensors with per-parameter `step`, `exp_avg`, `exp_avg_sq`). The diagnostic hashed it (G3) and confirmed shape-compatible mapping to `model.parameters()` order.
* Every arm restores model, optimizer state, cursors, `rng_states`, `schedule_anchor_step` and phase-transition history exactly via `JointScratchTrainer.load_state_dict`.
* **No reinitialization** of moments in any arm; no optimizer-policy change; no `--reanchor-schedule`; no LR phase transition.
* Pre-launch validity gate: after load and before the first update, the optimizer-state sha equals the value in §4 in **both** arms of a pair.
* Limitation: the ON arm introduces a new gradient component into moments accumulated under OFF. This is inherent to warm-start and identical in convention across arms.

## 9. RNG / TASK / BATCH MATCHING

* **Same start file and same restored RNG** (`restore_rng_states`: torch, numpy, python, cuda) within each pair. Stream seeds are derived and checked by `load_state_dict`.
* **Same task order:** `interleaved_123` from the checkpoint anchor, with no re-anchor. Each 6-step macro-cycle holds 1 R, 2 N and 3 C labels in a seeded order, `macro_cycle_n(ratio, schedule_seed, cycle)` (`train_joint_scratch.py:271, 1147–1163`). This is a pure function of the schedule seed and step, so it is identical across arms by construction.
* **Same batches:** streams are cursor-driven from the checkpoint cursors. The alignment term consumes no RNG, so batch identity must be identical across ON/OFF.
* **Validity gate:** the amended driver (§19) logs a per-macro-cycle hash of (task, batch item indices). The ON and OFF hash sequences must be identical over the whole budget; any mismatch makes the pair **INVALID** (not T4).
* **Device and software:**
  * same cluster partition (Jean-Zay V100 `gpu_p13`, as V6), same module `pytorch-gpu/py3/2.6.0`, same code commit for all four arms;
  * `--torch-deterministic` with `CUBLAS_WORKSPACE_CONFIG=:4096:8` in all four arms;
  * if strict determinism cannot be enabled on the training device, HARD STOP before any update and return to CENTRAL (no silent fallback).
* **Noise:** off (V6: noise 0). `model.train(True)` during updates, as V6.
* **Stopping:** no `--stop-at-ceiling`; no early stopping of any kind.

## 10. FIXED TRAINING BUDGET

**50u = 138,900 optimizer steps per arm** (1u = 2,778 steps), fixed, no extension or truncation.

| unit | count per arm |
|---|---|
| optimizer steps | 138,900 |
| macro-cycles (6 steps) | 23,150 |
| R updates / exposures | 23,150 updates ≈ 1,481,600 items ≈ 50.1 R epochs (29,571) |
| N updates | 46,300 ≈ 2,963,200 items ≈ 100.2 N epochs |
| C updates | 69,450 ≈ 4,444,800 items ≈ 158.9 C epochs (27,981) |
| end step W3 / W4 | 10,764,750 / 8,584,020 |

Justification (pre-pilot information only):

1. **Plasticity scale.** Under the existing recipe, native LTM error counts already move between adjacent 5u detector points (SD of lag-1 differences 132 for s19 and 189 for s20). The ventral interface is plastic on a 5u scale, so 50u gives ten such intervals.
2. **Measured null at exactly this horizon.** The V6 trajectories provide 50 lag-10 (50u) difference pairs per seed (s19 |Δ| q95 291, max 330; s20 q95 340, max 406), so the null drift band is empirically characterized at the chosen horizon, not extrapolated.
3. **Small added pressure.** The gradient diagnostic shows the w=0.1 term is ≈15% of retrieval gradient norm and mostly co-directional. Shorter horizons (e.g. 25u ≈ 79 C epochs) would risk a false T2 from insufficient cumulative exposure. 50u gives ≈159 C epochs of the added term, above the ≈100-epoch scale on which historical unrehearsed raw alignment reorganized ventral repetition (Phase-2 C0/C3).
4. **Still a mature-repair test.** 50u is 1.3% (W3) and 1.6% (W4) of each state's training history, so it tests repair of the existing interface, not re-development.
5. **Compute.** V6 ran 833,400 steps plus 60 full-lexicon evaluations in about 3h20 per seed on one V100. Four arms of 138,900 steps are about 1 GPU-h each, including deterministic-mode slowdown (≤2× assumed); post-hoc readouts are about 1–2 GPU-h total.

## 11. FIXED EVALUATION CADENCE

* **Evaluation steps** (offsets from the start step): 0 (baseline, before any update), then 13,890 × k for k = 1…10 (5u … 50u). This is 11 points per arm, on the V6 5u detector grid.
  * W3: 10,625,850 + 13,890k
  * W4: 8,445,120 + 13,890k
  * All absolute steps are multiples of 13,890.
* **Readouts are post hoc on saved checkpoints.** Training saves a checkpoint at each of the 10 non-baseline steps (`--save-every 13890`). The baseline readout uses the SOURCE file itself. All readouts (§12–15) are computed afterwards by one frozen evaluation script, identical for all arms, in eval mode under `torch.no_grad`. Evaluation therefore cannot affect training.
* In-training evaluation settings are identical across arms. None is required; if the driver performs any, it must be identical in all four arms, and batch-hash identity (§9) verifies it did not alter training.
* **Primary causal comparison:** k = 10 (50u) only.
  * Points k = 8, 9, 10 feed the frozen robustness condition in §17.
  * All other points are trajectory diagnostics.
  * No best-checkpoint selection.

## 12. GLOBAL READOUTS (every evaluation point, every arm)

Full populations: repetition and Naming over 29,571 rows; C over 27,981 canonical targets. All implementations are the historical ones.

* Repetition, canonical forced-length, FULL route: exact error count (`full_rep_errors` definition).
* Repetition, genuine free-AR (max 12 steps), FULL route: exact error count.
* Naming: strict exact error count (historical full Naming battery).
* C strict top-1: lexical-row identity error count over 27,981; homophone retrieval counts as an error (closed contract §3.2).
* Gate statistics (descriptive): mean, SD, p05, p95.

## 13. ROUTE READOUTS

* **Native isolated ventral repetition** (`route="ltm"`, no gate/WM/fusion): error counts under genuine free-AR (primary) and canonical forced-length (secondary). Decoding contract identical to the closed ventral-interface contract §4.
* **Dorsal (WM-only) isolated repetition, canonical:** error count, as a preservation control. It is historically valid: 0–1 errors across all 120 V6 detector points.

## 14. SEMANTIC-INTERFACE READOUTS

Per arm and evaluation point, over all 29,571 rows (C-population subsets as in the closed contract):

* S0 native isolated-ventral errors (free-AR primary, canonical secondary). This is the same quantity as §13, listed here as the interface anchor.
* Contemporaneous S1-style upper bound and gap (§15).
* Strict retrieval identity: C top-1 errors (27,981); lexical-identity errors over 29,571; homophone retrievals.
* Native semantic geometry, as means and medians over 29,571:
  * cos(ŝ, bank_raw target)
  * ‖ŝ‖₂
  * ‖ŝ‖₂ / ‖bank_raw target‖₂
  * MSE(ŝ, bank_raw target)
  * retrieval top-1 cosine and top-1 − top-2 margin
  * target cosine and historical target margin
* Loss-scale descriptors: mean L_C_retrieval_raw and L_align on the §10 canonical batch partition, at the checkpoint (no gradients).

No new architecture metrics.

## 15. S1-STYLE CONTEMPORANEOUS CONTROL

For **each** checkpoint separately (never stale identities):

1. `r_i^t = comprehension_metrics(encode_all(model_t, all 29,571 forms), bank_raw, range(29571)).top1_idx[i]`. This is the checkpoint's own top-1 row, computed exactly as closed contract §3.4.
2. S1c: inject `bank_raw[r_i^t]` (raw, unnormalized) through the frozen `ventral_interface` `SemanticInjection("S1", fixed=…)` hook into the same isolated ventral decode (free-AR primary, canonical secondary).
3. Report:
   * S1c error count
   * interface gap `G_t = S0_err − S1c_err`
   * S0-wrong→S1c-correct count
   * S0-correct→S1c-wrong count
4. **Validity flag:** S1C_BOUND_DEGRADED if `S1c_err > 500`, or if the gap is negative, at that checkpoint. The flag is reported and not decisive; S1c then no longer serves as an upper bound.
5. Baseline check (k = 0): S1c counts must equal the closed S1 counts, 42 (W3) and 51 (W4). Otherwise the evaluation script is INVALID.

## 16. PRIMARY ON−OFF CAUSAL CONTRAST

* For each pair P ∈ {W3, W4} and each point k: `Δ_P(k) = metric(P_ON, k) − metric(P_OFF, k)`, where negative means ON has fewer errors.
* **Primary quantity:** `ΔLTM_P = Δ_P(10)` for native isolated-ventral free-AR errors.
* **Co-primary convention check:** the same quantity for canonical.
* Reported for every §12–15 metric at every k:
  * Δ_W3(k) and Δ_W4(k)
  * absolute OFF and ON trajectories
  * OFF trajectory against the historical V6 detector values at the same absolute steps (reproducibility; descriptive, not gating, because V6 ran non-deterministically)
* ON changes relative to ON's own baseline are never used as the causal quantity.

## 17. T1/T2/T3/T4 RULES

### 17.1 Frozen numbers and their pre-pilot justification

**D_LTM = 560 errors: material native-ventral change.**

* It equals the maximum absolute difference between any two V6 detector points 5u, 10u, 25u or 50u apart, over both pre-pilot trajectories (s19, s20; 60 points each; lags 1, 2, 5, 10; 444 pairs). The observed maximum is 560 (s20, lag 2). It is identical for genuine free-AR and canonical native-LTM errors, which differ by at most 1 item at every one of the 120 V6 points.
  * s20 lag-10 q95 is 340 and max 406; s19 lag-10 q95 is 291 and max 330.
* So an ON−OFF difference beyond D_LTM exceeds every drift difference observed in the unperturbed recipe at any horizon from 5u to 50u.
* Two independently drifting continuations would differ with SD ≈ √2 × level SD: √2×103 ≈ 146 for s19 and √2×136 ≈ 192 for s20. 560 is therefore ≈3.8σ for W3 and ≈2.9σ for W4, and requiring it in both pairs makes a chance joint T1 very unlikely.
* It is 17.7% (W3) and 15.6% (W4) of the closed S0−S1 interface deficit: far from "full ceiling only" and well above drift.

**D_LTM_ROBUST = 280.** Half of D_LTM, applied to the mean of Δ over k = 8, 9, 10. It guards against a single-point fluctuation without requiring a full-size effect at every late point.

**Preservation floors**, ON−OFF at k = 10, evaluated per pair. Each is exceeded only beyond the full pre-pilot range:

| metric (errors) | allowed ON−OFF | pre-pilot basis |
|---|---|---|
| FULL canonical repetition | ≤ 7 | max over all 120 V6 detector points = 7 (s19); source 0 |
| FULL genuine free-AR repetition | ≤ 7 | same (free-AR FULL equals canonical FULL at V6 points) |
| WM-only canonical repetition | ≤ 1 | range 0–1 at all 120 points |
| Naming strict exact | ≤ 1 | range 0–1 at all 120 points |
| C strict top-1 | ≤ 13 | max absolute difference between any two V6 points 5u–50u apart (lags 1–10, both seeds) = 13 (s20); per-point SD 2.8–2.9 |

A pair has **DAMAGE** if any floor is exceeded.

**Collapse limits**, at any evaluation point, any arm, 10× the pre-pilot maximum: FULL canonical > 70; WM > 10; Naming > 10; C > 510; native LTM > 2× the seed's pre-pilot maximum (W3 > 7,102; W4 > 8,050).

### 17.2 Per-pair classification (k = 10)

* **IMPROVE_P:** all of
  * ΔLTM_freeAR ≤ −560
  * ΔLTM_canonical ≤ −560
  * mean over k = 8, 9, 10 of ΔLTM_freeAR ≤ −280
* **HARM_P:** ΔLTM_freeAR ≥ +560 and mean over k = 8–10 ≥ +280.
* **NULL_P:** otherwise.

### 17.3 Outcome (CENTRAL terminology; evaluated in this order, first match wins)

| label | CENTRAL name |
|---|---|
| T1 | **MATERIAL_REPLICATED_TARGETED_IMPROVEMENT** |
| T2 | **NO_MATERIAL_REPLICATED_TARGETED_IMPROVEMENT** |
| T3 | **MATERIAL_IMPROVEMENT_WITH_TRADEOFF** |
| T4 | **INSTABILITY_OR_MATERIAL_CROSS_SEED_REVERSAL** |

0. **INVALID:** any §19 validity gate fails, or any OFF arm collapses. Not interpreted; return to CENTRAL.
1. **T4 = INSTABILITY_OR_MATERIAL_CROSS_SEED_REVERSAL:** any of
   * nonfinite loss, gradient norm or parameter in any ON arm
   * collapse limit reached in any ON arm
   * W3 IMPROVE and W4 HARM, or W4 IMPROVE and W3 HARM
2. **T3 = MATERIAL_IMPROVEMENT_WITH_TRADEOFF:** IMPROVE_W3 and IMPROVE_W4, and DAMAGE in at least one pair.
3. **T1 = MATERIAL_REPLICATED_TARGETED_IMPROVEMENT:** IMPROVE_W3 and IMPROVE_W4, and no DAMAGE in either pair.
4. **T2 = NO_MATERIAL_REPLICATED_TARGETED_IMPROVEMENT:** everything else, with a mandatory descriptive sublabel (§17.4) and a separately reported DAMAGE flag.

   Semantic geometry or retrieval movement (§14) does not change a T2 classification. **T2 is never summarized as "no effect".** The permitted statement is: "the frozen mature-repair pilot did not meet the prospectively defined material replicated improvement criterion."

### 17.4 T2 descriptive sublabels (frozen prospectively; no new threshold)

These use only the already-frozen IMPROVE / HARM / NULL classifications and the signs of the raw paired differences. They introduce no new materiality threshold and do not modify T1–T4. Evaluated in order, first match wins:

1. **T2_NOT_REPLICATED:** exactly one pair satisfies IMPROVE and the other does not satisfy HARM (a HARM counterpart is already T4).
2. **T2_HARM:** at least one pair satisfies HARM, and the outcome is not T4.
3. **T2_SUBMATERIAL_SAME_DIRECTION:** neither pair satisfies IMPROVE or HARM, **and** both pairs have ΔLTM_freeAR(50u) < 0 **and** ΔLTM_canonical(50u) < 0. This is a descriptive sign-consistency label only.
4. **T2_NULL_OR_NEGLIGIBLE:** all remaining T2 cases.

Raw W3 and W4 trajectories and DAMAGE=YES/NO are always reported alongside the sublabel.

**Secondary, non-decisive mechanism readouts** are reported for any outcome:

* ΔG (contemporaneous gap)
* fraction of OFF gap closed, `−ΔG_P(10)/G_{P_OFF}(10)`
* Δ cos(ŝ, target), Δ norm ratio

A T1 whose ΔG does not move in the same direction is reported as T1 with MECHANISM_UNCONFIRMED.

## 18. PRESERVATION RULES

* The §17.1 floors are the preservation rules. They are fixed and apply per pair at k = 10.
* Trajectory violations at k < 10 are reported, not decisive, except collapse limits (T4/INVALID).
* No relaxed percentage floors.
* The tasks at ceiling at SOURCE (FULL 0, Naming 0, WM 0–1) are held to near-exact preservation relative to the matched OFF arm.

## 19. VALIDITY GATES (future authorized workstream; all must pass)

* **V1 — Driver amendment, preregistered and frozen before any real-state update.**
  * The existing `load_state_dict` refuses any change of `c_align_weight` unconditionally, including under `--phase-transition`.
  * A minimal amendment must add one explicit declaration (e.g. `--declare-c-align-transition`) permitting only `0.0 → 0.1`. It records `{"changed": ["c_align_weight"], "old": 0.0, "new": 0.1, "transition_step": …}` in `phase_transitions` and relaxes no other guard.
  * Per-macro-cycle (task, batch-index) hash logging is added.
  * All four arms use the same amended commit. OFF arms do not declare anything.
* **V2 — Amendment equivalence on synthetic/toy states only** (tests, no real state): at `c_align_weight=0` the amended driver is bitwise identical to the unamended driver over N toy steps (model and optimizer tensors, RNG). At 0.1 only the C-step loss differs.
* **V3 — Start identity:** source file sha, params sha, optimizer-state sha and composite id equal §4, in both arms of each pair.
* **V4 — Recipe identity after load:** schedule, anchor, cursors, LRs, λ_C, τ, dec weight, widths, populations, `optimizer_policy` equal to the checkpoint; ON differs only in `c_align_weight`; no `--stop-at-ceiling`, no re-anchor.
* **V5 — Determinism enabled** (§9) in all arms; otherwise HARD STOP before the first update.
* **V6 — Batch/task hash sequences identical** ON vs OFF per pair over the full budget.
* **V7 — Baseline readouts (k = 0) equal the closed values in §4** for every arm: S0 3199/3647, S1c 42/51, C 34/42, WM 0/1, FULL 0/0, Naming 0/0.
* **V8 — Exact step counts:** 138,900 steps; 10 checkpoints at the §11 steps; sha256 recorded.
* **V9 — Evaluation script frozen and hashed before any pilot checkpoint exists;** identical for all arms.
* **V10 — No SOURCE, closed-result or closed-worktree file modified** (hash re-verification before and after).

## 20. STOP-LOSS

* ONE causal pilot: four arms, one weight, one budget.
* After execution, whatever the outcome (T1, T2, T3, T4 or INVALID): **no** change to `c_align_weight`, schedule, LR, duration, decoder weight, gate setting, start-checkpoint convention, seeds, or optimizer-state policy without new CENTRAL arbitration. No extension of the budget, no additional arm, no re-run "to confirm".
* **Warm-start interpretation limit.**
  * A negative result (T2/T3/T4) directly falsifies "C-step alignment at w=0.1 can repair the mature interface over this 50u horizon".
  * It does NOT prove that C-step alignment could never matter if present throughout from-scratch development.
  * A negative result does **not** authorize a from-scratch rescue experiment. Any developmental follow-up requires a new CENTRAL decision.
* A T1 result does not authorize a full-ceiling run, an attractor, a Yair flag, or adoption of c_align in a final recipe. Each requires CENTRAL.
* `YAIR_SEMANTIC_ATTRACTOR=LATER_IF_T_FAILS`: not implemented, not a control, not tested by this pilot.

## 21. NOT_ESTABLISHED

* That C-step alignment causally affects the native ŝ→decoder interface in either direction.
* That gradient co-directionality at the start states predicts the training outcome.
* That FINAL-2A's harm transfers to interleaved/H512/mature/w=0.1.
* That determinism on V100 fully removes run-to-run variation, or that the V6 historical trajectory will be reproduced by the OFF arms.
* That 50u is sufficient for a slow-acting effect. A T2 is bounded to this horizon (§20).
* Any generalization beyond seeds 19 and 20, or to from-scratch development.
* That S1c remains a valid upper bound at every future checkpoint (flagged per checkpoint, §15).

## 22. STOP BEFORE TRAINING (authorization state after the CENTRAL amendment)

* The design pass created no training launcher, ran no OFF or ON arm, and performed no optimizer step on any state.
* CENTRAL has since issued GO_FOR_DRIVER_AMENDMENT_AND_PREFLIGHT=YES and GO_FOR_SINGLE_C_ALIGN_TRAINING_PILOT=CONDITIONAL_YES. The first optimizer step is authorized **if and only if** every §19 pre-training gate (V1–V10) passes from the frozen implementation commit; otherwise TRAINING_RUN=NO, OPTIMIZER_STEP_COUNT=0, and the blocker returns to CENTRAL without patch-and-continue.
* ARCHITECTURE_CHANGED=NO · GATE_CHANGED=NO · ATTRACTOR_IMPLEMENTED=NO · YAIR_FLAG_IMPLEMENTED=NO · LESIONING_RUN=NO · FULL_CEILING_RUN=NO remain binding for this workstream regardless of outcome.
* Indicative invocation, subject to the V1 amendment and the frozen launch configs:

  ```
  # NOT AUTHORIZED — illustrative only
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  python scripts/naming_comprehension/train_joint_scratch.py --regime j0 --seed {19|20} \
    --subset-mode final_full --schedule interleaved_123 --wm-hidden 128 --enc-hidden 512 --dec-hidden 512 \
    --device cuda --max-steps {10764750|8584020} --lr-repetition 3e-5 --lr-naming 3e-5 --lr-comprehension 1e-4 \
    --glove-path <GloVe sha 91125602…> --save-every 13890 --torch-deterministic \
    --c-align-weight {0.0|0.1} [--declare-c-align-transition  # ON only, after V1] \
    --out-dir <pilot runs namespace> --run-id c_align_pilot_{W3|W4}_{OFF|ON} --resume <SOURCE file>
  ```
