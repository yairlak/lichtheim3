# DIRECTIONAL_DOSE_EXPERIMENT_CONTRACT — LICHTHEIM3 VENTRAL SEMANTIC DIRECTIONAL DOSE (FROZEN DIAGNOSTIC DESIGN)

Programme: POST_STAGE / PAPER_PROGRAMME · Decision authority: CENTRAL STEERING · Written 2026-09-17.

## 1. STATUS

CONTRACT_STATUS=DESIGN_COMPLETE_NOT_EXECUTION_AUTHORIZED

* This document is a **design**. No driver exists, no α condition has been computed on any real state or item, and no dose-response quantity has been observed.
* Implementation and execution require an explicit CENTRAL authorization. At that point the contract must be re-frozen (hashed, `CONTRACT_STATUS=FROZEN`) together with its implementation, following the protocol of the closed ventral-interface diagnostic.
* Design worktree / branch: `/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/wt-ventral-directional-dose` · `paper-programme/ventral-directional-dose-design`, created from `0f25b5b87efd8038dfc4c86b5ca2935bf7b178c7`.

## 2. SCIENTIFIC QUESTION

**How much directional movement of native ŝ toward its already-retrieved lexical prototype is required to recover ventral decoder compatibility?**

The question is posed to discriminate, before CENTRAL chooses, between **T (training-only compatibility)** and **R (minimal semantic refinement)**. This contract makes no implementation decision.

**Accepted prior localization** (CENTRAL, CLOSED_RESULT_SAFE): the dominant residual isolated-ventral repetition deficit lies at the encoder-produced ŝ → ventral-decoder interface. Lexical identity is already recoverable from ŝ for essentially all relevant failures. Replacing ŝ by the raw lexical prototype restores decoding, while replacing only its norm does not. A recurrent semantic attractor is **not** established as necessary.

## 3. IMMUTABLE PRIOR CONTROLS

The closed diagnostic is reused **read-only**. Its outputs are never modified, regenerated or reinterpreted.

| item | value |
|---|---|
| closed worktree / branch | `/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/wt-ventral-interface` · `paper-programme/ventral-interface-diagnostic` |
| frozen implementation commit | `4ad20048e20da84b9f22a92965098b84c2bf7dd6` |
| results-only commit | `0f25b5b87efd8038dfc4c86b5ca2935bf7b178c7` |
| closed contract SHA256 | `a5ca7b83717b43ab77cedae0cd3e4bef17901ffff051f5ca90b9340b45bda5d3` |
| `item_level_factorization.tsv` | `37f2fb170e53596067ad1bb8a64189dabec6d808fd31a1ffff942649de175663` |
| `summary_metrics.json` | `455ccaf07bd7e2adb6047c24ac40667ba9b3e1256bdfeeb6772589b12b610c80` |
| `ar_diagnostic_native_freear.tsv` | `75106959da8156131fc9f19d4d053b8307825dbcb7f0a45ead48a8eb15fdf027` |
| closed results SHA256SUMS | `paper_programme/ventral_semantic_interface/scientific_execution/SHA256SUMS` (37 entries) |

**Controls used by reference:**
* **S0** (native ŝ; this is α = 0): native correctness per state × convention, taken from `S0_{freear,canonical}_exact_correct`.
* **S1** (raw retrieved prototype: prototype direction **and** prototype norm): the full-prototype control, from `S1_*_exact_correct`.
* **Previous S1 rescue** of item i in state × convention: `S0_{conv}_exact_correct == 0 ∧ S1_{conv}_exact_correct == 1`, read **only** from the immutable item-level file.
* **S3** (native direction, prototype norm) and **S2** (raw true GloVe): reported alongside, unchanged.

## 4. AUTHORITATIVE STATES (exactly four; no other checkpoint, seed or state)

| state | witness | base SHA256 | Arm-A head SHA256 | reconstructed_state_identity |
|---|---|---|---|---|
| W3_SRC | V6 seed19 u3825 step 10625850 | `a5f21de9edb4b2090cb144c666ca4bd8e1f0572333000facb13f6c51cfaad76c` | NA | `6d7282854323b810727cd9c0c90cace6dd8a54d306408cafaa8c40c5455d10a0` |
| W3_REP | same + Arm-A | same | `8865ba9539e4a445ffc8847a71be698779a98478aa5e6a1bce2393f3de7afcfc` | `9185aa5671e2ae3ab1e0860d52424b472f616d2515057d7ccbc6b383cf43dff1` |
| W4_SRC | V6 seed20 u3040 step 8445120 | `0657f41030e54eac11816116af6c76b66d597c393fc72579a3755401c9dc79f3` | NA | `32928707acc81af215faa829ad94bba566804982bc192bb5116ea32d43d56e29` |
| W4_REP | same + Arm-A | same | `724ed4c6a84fba07b6d9f11d535b8a0eb726daff21b9def9a5e60d9c7b972d03` | `e151b306f706e9db9828b3c9e742984a0ae836dbd97e73f86e4e2063bd7c5cd6` |

* **Reconstruction:** the unmodified `scripts/gating_diagnostics/run_gate_route_audit.build_state` (manifest blob `e5f0bb4fa68ebe1b200a3fcabe3544ebbff5483b`). Composite identity `sha256("gxlr-state-v1|"+base+"|"+head)`.
* **Data:** lexicon `ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66`; GloVe file `91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed`; raw bank tensor `4658e11e6a8f60a468472cc3fa62e71064e33da6790f924cc3005e602762ddb4`; semantic bank = row-normalized raw bank (non-persistent buffer).

## 5. POPULATION

The population and identity contract are **identical** to the closed contract §3:
* 29,571 lexical repetition rows (all evaluated);
* 27,981 canonical C targets;
* 1,590 non-canonical homophone rows;
* historical C correctness = exact lexical-row identity on P_C only (NA elsewhere);
* `lexical_identity_correct` and `phonology_correct` exposed separately.

The **retrieved lexical row** for every item and state is the immutable `retrieved_index` of the closed item-level file (the frozen historical top-1 rule). It is not recomputed under a new definition; DOSE-J checks it against the frozen rule. Every item has a retrieved row, so "valid retrieved prototype" excludes only the degenerate cases of §9.

## 6. DIRECTIONAL INTERVENTION

For item i in state x:

```
ŝ        = the native encoder-produced semantic vector (the live to_semantic output, exactly as S0/S3 in the closed diagnostic)
v        = bank_raw[retrieved_index_i]      (RAW / UNNORMALIZED GloVe row of the frozen top-1 lexical identity)
u_s      = ŝ / ‖ŝ‖
u_p      = v / ‖v‖
u_α      = SLERP(u_s, u_p, α)               (§8, §9)
s_α      = ‖ŝ‖ · u_α                        (NATIVE ŝ NORM PRESERVED; only direction changes)
```

**Explicit endpoint distinction:**

| label | direction | norm | status |
|---|---|---|---|
| S0 (closed) = α 0 | native u_s | ‖ŝ‖ | immutable control |
| S3 (closed) | native u_s | ‖v‖ | immutable control |
| **α = 1.00 (this design)** | **prototype u_p** | **‖ŝ‖** | new dose endpoint — **NOT S1** |
| S1 (closed) | prototype u_p | ‖v‖ | immutable control (raw prototype) |

At α = 1.00 the raw prototype is **never** substituted. [STRUCTURAL_CODE_FACT] With S0, S3 and S1, the α = 1 endpoint completes a direction × norm arrangement of existing vectors. It is not a new scientific condition family, and no interpretation beyond §16 is attached to that arrangement.

**Injection (future implementation constraint).**
* The vector must enter through the same single forward hook on `model.ltm.to_semantic` used by the closed diagnostic, via a **new** supplier.
* Frozen files of the closed implementation (`ventral_interface/*`, its driver, config and tests) must **not** be modified. New code must live in a separate package and import the frozen decode, injection and population functions.

## 7. EXACT ALPHAS

**α ∈ {0.25, 0.50, 0.75, 1.00}.** No other value is a scientific condition.

α = 0 is **not** re-executed as a condition. S0 is taken from the immutable control. α = 0 appears only inside validity gate DOSE-B, through the frozen α = 0 short-circuit (§8).

## 8. SLERP MATHEMATICS (ordinary case)

All interpolation arithmetic uses **float64**. The result is cast once, deterministically, to the model semantic dtype (float32) immediately before it is returned by the hook.

```
ns = ‖ŝ‖₂ (float64),  nv = ‖v‖₂ (float64)
u_s = ŝ / ns,  u_p = v / nv
d   = clip(u_s · u_p, −1, +1)
θ   = arccos(d)
r   = u_p − d · u_s                  (‖r‖ = sin θ; no division by sin θ)
ORDINARY iff ‖r‖ > EPS_ORTHO
q   = r / ‖r‖
u_α = cos(αθ) · u_s + sin(αθ) · q
u_α = u_α / ‖u_α‖                    (renormalisation; changes u_α by ≤ ~1e-16)
s_α = cast_float32( ns · u_α )
```

* **α = 0 short-circuit (frozen):** `s_0 := ŝ` exactly (the input tensor, uncast), for every case. This avoids spurious last-bit float32 differences in the DOSE-B reproduction gate.
* **α = 1:** no endpoint assignment in the ordinary and near-collinear cases, where the formula reaches u_p to within ~1e-15. There is an explicit endpoint assignment only in the near-antipodal case (§9C).

**Frozen tolerances:**

| name | value | use |
|---|---|---|
| EPS_NORM | 1e-6 (float64 ‖·‖₂) | zero-norm threshold, identical to the closed `S3_DEGENERATE_NORM` |
| EPS_ORTHO | 1e-6 (float64 ‖r‖ = sin θ) | ordinary vs near-(anti)collinear |
| TOL_NORM_REL | 1e-6 | DOSE-C: \|‖s_α‖ / ‖ŝ‖ − 1\| on the float32 vector (float64 norms) |
| TOL_ENDPOINT_COS | 1e-6 | DOSE-D: 1 − cos(s_1, v) (float64 on float32 vector) |
| TOL_ANGLE_RAD | 1e-6 | DOSE-E: \|∠(ŝ, s_α) − α·∠(ŝ, v)\|, angles by `atan2(‖a − (a·b)b‖, a·b)` on unit float64 vectors from the float32 values |
| TOL_MONOTONE_RAD | 1e-9 | DOSE-E: ∠(ŝ, s_α) nondecreasing in α (ordinary items) |

**Numerical verification of the specification** (synthetic vectors only; no model, no real item). Script and output are outside git in `archives/ventral_directional_dose_design_20260917/`; hashes are in `provenance/DESIGN_PASS_EXTERNAL_ARTIFACTS.md`.
* 20,000 random 300-d pairs spanning cos ∈ (−0.99, 0.999), with norms matching the observed ranges (ŝ 5–12, prototype 2.6–14), × 4 alphas: 80,000 ORDINARY evaluations.
* Worst-case errors:
  * norm relative error 4.4e-16 in float64, 1.2e-8 after the float32 cast;
  * endpoint cosine deficit 5.6e-16 / 8.9e-16;
  * angle error 3.0e-15 / 6.4e-9 rad;
  * 0 monotonicity violations.
* All frozen tolerances clear these errors by ≥ 2 orders of magnitude.

## 9. NUMERICAL FALLBACKS (deterministic, preregistered; chosen without inspecting any dose result)

Evaluated in this order for every item × α ≠ 0:

**A. Zero / effectively-zero norm**
* **A1.** `ns ≤ EPS_NORM`: flag `ZERO_SHAT`. `s_α := ŝ` unchanged (decode stays defined), and the item is **excluded from every α denominator and transition count** in that state. The count is reported. This matches the closed S3 contract (0 degenerate ŝ occurred there).
* **A2.** `nv ≤ EPS_NORM`: **HARD STOP**. This is impossible under the frozen bank (minimum raw row norm 2.61) and would indicate a corrupted bank or retrieval identity.

**B. Near-collinear, same direction** (`‖r‖ ≤ EPS_ORTHO` and `d > 0`): normalized linear interpolation, flag `NEAR_COLLINEAR_NLERP`.

```
w_α = (1 − α)·u_s + α·u_p ;  u_α = w_α / ‖w_α‖ ;  s_α = cast(ns · u_α)
```

Here ‖w_α‖ ≥ cos(θ/2) ≈ 1, so it is always valid. At α = 1 this gives u_p exactly (w = u_p).

**C. Near-antipodal** (`‖r‖ ≤ EPS_ORTHO` and `d ≤ 0`): ONE fixed great circle, flag `NEAR_ANTIPODAL_BASIS`.
1. `k = argmin_k |u_s[k]|`, taking the **first index** on ties.
2. `q0 = e_k − u_s[k]·u_s`; `q = q0 / ‖q0‖`. The orthogonal component r is *not* used here, because ‖r‖ ≤ EPS_ORTHO is numerically unusable by definition.
3. `u_α = cos(αθ)·u_s + sin(αθ)·q`, with θ = arccos(d).
4. **α = 1 endpoint assignment (prospective):** `u_1 := u_p`, flag `NEAR_ANTIPODAL_ENDPOINT_ASSIGNED`. The circle through q ends at −u_s, which may differ from u_p by up to ~EPS_ORTHO.

The synthetic verification covered exact and near (1e-9-perturbed) same-direction and antipodal pairs: all outputs are finite, norms are preserved (≤ 2.2e-16), α = 1 reaches u_p (angle ≤ 1e-16), antipodal angles follow α·π exactly, the construction is deterministic, and zero ŝ and zero prototype behave as specified.

**Reporting.** Every item × α carries `dose_case ∈ {ORDINARY, NEAR_COLLINEAR_NLERP, NEAR_ANTIPODAL_BASIS, NEAR_ANTIPODAL_ENDPOINT_ASSIGNED, ZERO_SHAT}`. Per state, counts per case are reported. Fallback items (B, C) are **included** in denominators and additionally reported as a separate stratum. DOSE-E angular checks apply to ORDINARY items and, for C, to the α-fraction of π.

## 10. DECODING CONTRACT

The contract is identical to closed contract §4, with no change:
* isolated ventral route `route="ltm"` only, via the unmodified `gating_diagnostics.gate_probe.ar_decode_free` (cap `FREE_AR_MAX_STEPS = 12`, imported and asserted) and `ar_decode_forced_length`;
* BOS id 1, greedy argmax (first index on ties), feedback of the argmaxed token, EOS id 2 with a cut at the first EOS;
* free-AR breaks only when every row has emitted EOS; forced-length uses a per-item `len+1` window;
* exact match means prediction == form;
* decode batch 256 in bank order, CPU, float32, deterministic algorithms, `eval()`, inference mode;
* no gate, no FULL fusion, no dorsal readout;
* raw tokens recorded through the passive motor hook and checked against the historical return values.

## 11. PRIMARY / SECONDARY READOUTS

* **PRIMARY:** genuine free-AR isolated ventral repetition.
* **SECONDARY:** canonical forced-length isolated ventral repetition.
* They are never pooled. Every quantity is reported per **state × α × convention**.

## 12. ITEM-LEVEL SCHEMA (`item_level_directional_dose.tsv`; one row per state × item; column order frozen at implementation freeze)

* **Identity, copied from the immutable file and verified equal:**
  * state: `witness_id, state_id, source_or_repaired, seed, source_u, source_checkpoint_sha256, repaired_head_sha256, reconstructed_state_identity`
  * item: `item_index, lexical_identity, canonical_C_index, canonical_C_identity, in_C_population, target_phonology, phoneme_length, homophone_group, homophone_group_size`
  * retrieval: `retrieved_index, retrieved_lexical_identity, retrieved_phonology, C_contract_correct, lexical_identity_correct, phonology_correct`
* **Immutable controls, copied:** `S0_{freear,canonical}_exact_correct`, `S1_{freear,canonical}_exact_correct`, `S2_{freear,canonical}_exact_correct`, `S3_{freear,canonical}_exact_correct`, and `prev_S1_rescue_{freear,canonical}` (defined in §3).
* **Geometry (α-independent):** `shat_norm`, `retrieved_raw_glove_norm`, `cos_us_up` (= d, float64), `theta_rad`, `norm_r` (= sin θ), `dose_case_base` (ORDINARY / NEAR_COLLINEAR / NEAR_ANTIPODAL / ZERO_SHAT).
* **Per α ∈ {a025, a050, a075, a100}:**
  * vector checks: `{a}_dose_case`, `{a}_s_alpha_norm`, `{a}_norm_rel_err`, `{a}_cos_us_ualpha`, `{a}_cos_ualpha_up`, `{a}_angle_s_to_alpha_rad`, `{a}_angular_fraction` (= angle / θ; NA if θ ≤ EPS_ORTHO), `{a}_angle_err_rad`, `{a}_live_shat_vs_s0_max_abs_dev`;
  * per convention c ∈ {freear, canonical}: `{a}_{c}_exact_correct`, `{a}_{c}_predicted_phonology`, `{a}_{c}_pred_length`, `{a}_{c}_eos_emitted`, `{a}_{c}_first_eos_step`, `{a}_{c}_eos_before_target_length`, `{a}_{c}_eos_after_target_length`, `{a}_{c}_terminated_by_cap` (freear; NA canonical), `{a}_{c}_first_divergence_step`, `{a}_{c}_transition_vs_S0` ∈ {WRONG_TO_CORRECT, CORRECT_TO_WRONG, CORRECT_TO_CORRECT, WRONG_TO_WRONG}, `{a}_{c}_recovers_prev_S1_rescue` (1/0 if `prev_S1_rescue`, else NA).

No metric may be added after results are seen.

## 13. SUMMARY SCHEMA (`summary_metrics_directional_dose.json`)

**Structure:** `results → state_id → convention → stratum → alpha (a025 … a100)`. Each leaf holds:

```
{ denominator, exact_count, exact_proportion,
  transitions_vs_S0: {WRONG_TO_CORRECT, CORRECT_TO_WRONG, CORRECT_TO_CORRECT, WRONG_TO_WRONG},
  prev_S1_rescues_in_stratum, prev_S1_rescues_recovered, prev_S1_rescues_recovered_fraction,
  new_regressions (= CORRECT_TO_WRONG) }
```

It also carries, per state × convention × stratum, the **immutable controls** S0/S1/S2/S3 exact counts, copied from the closed summary.

**Strata (frozen):**

| stratum | definition |
|---|---|
| ALL_REPETITION_ITEMS | all 29,571 rows |
| C_POPULATION | canonical C targets |
| NONCANONICAL_HOMOPHONE_MEMBERS | the 1,590 non-canonical rows |
| NATIVE_LTM_WRONG | S0 wrong (same convention) |
| **C_CORRECT_AND_NATIVE_LTM_WRONG** | closed definition |
| PREV_S1_RESCUES | S0 wrong ∧ S1 correct |
| FALLBACK_CASES | dose_case ≠ ORDINARY at any α |

Denominators exclude only ZERO_SHAT items.

**Paired SOURCE / POST_REPAIR block** (`paired → witness → convention → alpha`):
* SRC and REP exact counts;
* item-level SRC→REP transitions under the same α;
* Jaccard of the α-rescue sets;
* per-α difference of `prev_S1_rescues_recovered_fraction`.

**Gates block:** DOSE-A … DOSE-J reports.

**Dose curves** are reported as the ordered raw sequences over α ∈ {0 (=S0), 0.25, 0.50, 0.75, 1.00}, with the S1 control shown separately. **No fitted curve, threshold or categorical label is computed by code.**

## 14. SOURCE / REPAIR PAIRING

Paired reporting is **mandatory** for W3_SRC ↔ W3_REP and W4_SRC ↔ W4_REP, for every α and convention. The dose diagnostic must state whether the α at which previous S1 rescues are recovered is stable across SOURCE and POST_REPAIR. It is reported as raw per-α recovered fractions and item-set overlaps, not as a thresholded verdict. No other seeds or states.

## 15. VALIDITY GATES FOR FUTURE EXECUTION (all hard stops)

On failure, the run writes `GATE_FAILURE.json` (gate, state, evidence) and stops. No summary is written, results are not interpreted, and the case returns to CENTRAL.

| gate | requirement |
|---|---|
| **DOSE-A** prior state / control identity | All four states reconstruct with SOURCE, head, archival and composite hashes exactly as §4. The closed contract, item-level, summary and AR files hash exactly as §3. Lexicon, GloVe and bank hashes as §4. |
| **DOSE-B** endpoint α = 0 | The α = 0 short-circuit decoded through the **new** supplier and hook reproduces immutable S0 item by item for **both** conventions: exact_correct **and** predicted_phonology on all 29,571 items per state (0 mismatches). |
| **DOSE-C** norm preservation | For every non-ZERO_SHAT item and α: \|‖s_α‖/‖ŝ‖ − 1\| ≤ 1e-6 (float32 vector, float64 norms). |
| **DOSE-D** directional endpoint | For every non-ZERO_SHAT item: 1 − cos(s_1, v) ≤ 1e-6. |
| **DOSE-E** angular monotonicity | For ORDINARY items: \|∠(ŝ, s_α) − α·θ\| ≤ 1e-6 rad for all α, and ∠ nondecreasing in α within 1e-9 rad. For NEAR_ANTIPODAL_BASIS: \|∠ − α·θ\| ≤ 1e-6 rad for α < 1. |
| **DOSE-F** shared downstream path | Every α uses the identical hook site, `decode_from_s_hat` (`sem_to_h0`, decoder, `dec_to_premotor`), motor and `gate_probe` decoders as S0. Recorded tokens reproduce the historical return values. Encoder output is bitwise stable across AR steps. |
| **DOSE-G** no forbidden access | The FULL-fusion, gate and dorsal guards are never hit (the frozen `ventral_only` guard is reused). |
| **DOSE-H** immutable previous controls | Before and after execution, the closed `scientific_execution/SHA256SUMS` (37 entries) and freeze `SHA256SUMS` (17 entries) verify, and the closed worktree tracked tree equals `0f25b5b8…`. |
| **DOSE-I** parameter immutability | Per-state `state_dict` SHA256 is identical before and after evaluation, and equal to the closed preflight values (W3_SRC `4a8d4807…`, W3_REP `c72b87d9…`, W4_SRC `004dda2d…`, W4_REP `d1102732…`). |
| **DOSE-J** retrieval identity reuse | `retrieved_index` recomputed by the frozen historical rule equals the immutable file for every item and state. Live ŝ in the decode matches the retrieval ŝ within 1e-5 max abs, with identical top-1 (closed SHAT rule). |

## 16. INTERPRETATION FAMILIES (CENTRAL, embedded verbatim in substance)

* **DR1 — EARLY RESCUE.** Substantial rescue appears by α ≤ 0.50 with low regression. *Implication:* a small continuous semantic correction may be sufficient; minimal semantic refinement becomes high priority.
* **DR2 — LATE / NEAR-PROTOTYPE RESCUE.** Large rescue appears mainly at α ≥ 0.75. *Implication:* the inference solution approaches lexical prototype replacement. Do not immediately build an attractor; training-only compatibility becomes the preferred next family.
* **DR3 — NON-MONOTONIC / REGRESSION-HEAVY.** Intermediate movement causes substantial regressions or unstable behaviour. *Implication:* do not prioritize prototype-directed architectural refinement; training-only becomes preferred.
* **DR4 — SMOOTH MONOTONIC DOSE RESPONSE.** Decoder compatibility improves progressively as semantic direction approaches the prototype. *Implication:* strong evidence for a graded functional compatibility axis; a minimal differentiable refinement pilot becomes justified.

**Binding application rules:**
* No categorical threshold for "substantial", "large" or "low" is defined. None may be introduced after results are seen.
* The recap must expose raw per-α rescue and regression counts, previous-S1-rescue recovery fractions and the paired SOURCE/REPAIR sequences, then state only conclusions literally supported by those numbers.
* Classification is per state and convention, with the primary convention leading.
* If states or conventions materially disagree, no single family is forced; the disagreement is reported.
* The α = 1 result must always be read against **S1** (same direction, different norm) and **S3** (same norm source, different direction), never substituted for them.
* "Off-manifold" is not an admissible conclusion (no operational manifold test is preregistered).

## 17. NOT_ESTABLISHED (before execution)

* Any dose-response shape. No α condition has been evaluated.
* That a small directional correction suffices, or that near-prototype replacement is needed.
* That a semantic attractor or any refinement architecture is necessary.
* Why existing supervision (`L_dec` from ŝ, `L_align`, Naming from raw GloVe, retrieval CE; see `VENTRAL_INTERFACE_TRAINING_SUPERVISION_AUDIT.md`) leaves the decoder more compatible with prototype direction than with native ŝ direction.
* Generality beyond the four frozen states.
* The content of the "flag" proposal (`YAIR_FLAG_PROVENANCE_AUDIT.md`).

## 18. STOP CONDITIONS

* **This design pass:** STOP after the design package. No driver, no α execution, no dose figures, no training, no architecture, gate or lesion work, no attractor, no flag implementation.
* **Future execution** (only if CENTRAL authorizes):
  1. freeze contract, implementation and tests with hashes before any α run;
  2. execute exactly once;
  3. any DOSE gate failure → STOP and return to CENTRAL, with no interpretation;
  4. after packaging → STOP and return to CENTRAL; no training or refinement follows automatically.
