# DIRECTIONAL_DOSE_EXPERIMENT_CONTRACT — LICHTHEIM3 VENTRAL SEMANTIC DIRECTIONAL DOSE (FROZEN DIAGNOSTIC)

Programme: POST_STAGE / PAPER_PROGRAMME · Decision authority: CENTRAL STEERING
Design written 2026-09-17 · Amended 2026-09-17 per CENTRAL arbitration (§0).

## 1. STATUS

CONTRACT_STATUS=AMENDED_PER_CENTRAL_NOT_YET_FROZEN

* This revision incorporates CENTRAL's binding amendments (§0). It is a **pre-implementation amendment**: no directional-dose code exists yet and no α condition has been decoded on any real state or item.
* The status becomes `CONTRACT_STATUS=FROZEN` only at the implementation-freeze commit, together with hashed implementation, tests and config.
* Worktree / branch: `/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/wt-ventral-directional-dose` · `paper-programme/ventral-directional-dose-design`.
* Lineage: design commit `0b758696a99577f9ead43b33c295522dc158dc98` → results commit `0f25b5b87efd8038dfc4c86b5ca2935bf7b178c7` → freeze commit `4ad20048e20da84b9f22a92965098b84c2bf7dd6`.

### §0. CENTRAL arbitration record (binding)

```
DIRECTIONAL_DOSE_CONTRACT_STATUS=ACCEPTED_WITH_MODIFICATION
SLERP_STATUS=ACCEPTED_WITH_MODIFICATION
NEAR_ANTIPODAL_POLICY=HARD_STOP
TRAINING_SUPERVISION_AUDIT_STATUS=ACCEPTED_WITH_QUALIFICATION
YAIR_FLAG_PROVENANCE_STATUS=ACCEPTED_WITH_QUALIFICATION
YAIR_FLAG_IDENTITY=SEMANTIC_ATTRACTOR_FLAG_CONFIRMED
GO_FOR_DIRECTIONAL_DOSE_IMPLEMENTATION=YES
GO_FOR_DIRECTIONAL_DOSE_EXECUTION=YES
GO_FOR_TRAINING=NO
GO_FOR_ARCHITECTURE_CHANGE=NO
```

Amendments applied relative to the design commit (`0b758696…`):

| id | design rule | amended rule |
|---|---|---|
| AM-1 | near-antipodal (sin θ ≤ 1e-6, d ≤ 0) → fixed basis great circle, α = 1 assigned u_p | **HARD STOP.** The basis fallback is removed; no scientific α decoding may begin. |
| AM-2 | ZERO_SHAT (‖ŝ‖ ≤ 1e-6) → excluded from denominators, counted | **HARD STOP.** The preflight fails and execution does not begin. |
| AM-3 | zero prototype → HARD STOP | unchanged (HARD STOP) |
| AM-4 | near-collinear same direction → NLERP | accepted unchanged |
| AM-5 | ordinary SLERP | accepted unchanged |
| AM-6 | FALLBACK_CASES stratum | replaced by **NEAR_COLLINEAR_CASES**; no ZERO_SHAT or near-antipodal strata exist (both are hard stops) |
| AM-7 | gates DOSE-A…J | amended wording (§15) plus hard-stop preconditions **NO_ZERO_SHAT, NO_ZERO_PROTOTYPE, NO_NEAR_ANTIPODAL** |
| AM-8 | — | mandatory **real-state geometry preflight** (§15b) before any scientific α decode |
| AM-9 | — | mandatory **α = 1 vs S1 vs S3 factorization** reporting (§16b) and the **T vs R steering question** (§16c) |
| AM-10 | — | binding training-audit boundary (§17b) and Yair-flag bookkeeping (§17c) |

## 2. SCIENTIFIC QUESTION

**How much directional movement of native ŝ toward its already-retrieved lexical prototype is required to recover ventral decoder compatibility?**

This steers CENTRAL's choice between **T (training-only compatibility)** and **R (minimal semantic refinement)**. The diagnostic implements neither.

**Accepted prior localization** (CLOSED_RESULT_SAFE):
* The dominant residual isolated-ventral repetition deficit lies at the encoder ŝ → ventral-decoder interface.
* Lexical identity is recoverable from ŝ for essentially all relevant failures.
* Raw-prototype substitution restores decoding; norm-only substitution does not.
* A recurrent semantic attractor is **not** established as necessary.

## 3. IMMUTABLE PRIOR CONTROLS

Read-only; never modified, regenerated or redefined.

| item | value |
|---|---|
| closed worktree / branch | `/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/wt-ventral-interface` · `paper-programme/ventral-interface-diagnostic` |
| closed freeze commit | `4ad20048e20da84b9f22a92965098b84c2bf7dd6` |
| closed results commit | `0f25b5b87efd8038dfc4c86b5ca2935bf7b178c7` (the directional-dose branch descends from it, so identical tracked copies exist in this worktree) |
| closed contract SHA256 | `a5ca7b83717b43ab77cedae0cd3e4bef17901ffff051f5ca90b9340b45bda5d3` |
| `item_level_factorization.tsv` | `37f2fb170e53596067ad1bb8a64189dabec6d808fd31a1ffff942649de175663` |
| `summary_metrics.json` | `455ccaf07bd7e2adb6047c24ac40667ba9b3e1256bdfeeb6772589b12b610c80` |
| `ar_diagnostic_native_freear.tsv` | `75106959da8156131fc9f19d4d053b8307825dbcb7f0a45ead48a8eb15fdf027` |
| closed SHA256SUMS | freeze manifest (17 entries) and results manifest (37 entries) |

Controls by reference:

| control | definition | reused field |
|---|---|---|
| **S0** | native ŝ (≡ α = 0) | `S0_{conv}_exact_correct`, `S0_{conv}_predicted_phonology` |
| **S1** | RAW retrieved prototype: **prototype direction + prototype norm** | |
| **S2** | RAW true GloVe | |
| **S3** | **native direction + prototype norm** | |
| **α = 1.00 (new)** | **prototype direction + NATIVE ŝ norm**; never labelled or substituted as S1 | |
| **previous S1 rescue** (item i, state, convention) | `S0_{conv}_exact_correct == 0 ∧ S1_{conv}_exact_correct == 1`, read only from the immutable item-level file | |

## 4. AUTHORITATIVE STATES (exactly four)

| state | witness | base SHA256 | Arm-A head SHA256 | reconstructed_state_identity | closed params `state_dict` SHA256 |
|---|---|---|---|---|---|
| W3_SRC | V6 s19 u3825 step 10625850 | `a5f21de9edb4b2090cb144c666ca4bd8e1f0572333000facb13f6c51cfaad76c` | NA | `6d7282854323b810727cd9c0c90cace6dd8a54d306408cafaa8c40c5455d10a0` | `4a8d4807f1a1928faebebf696d41f4c222b3997b52432d7015fc038881eac785` |
| W3_REP | + Arm-A | same | `8865ba9539e4a445ffc8847a71be698779a98478aa5e6a1bce2393f3de7afcfc` | `9185aa5671e2ae3ab1e0860d52424b472f616d2515057d7ccbc6b383cf43dff1` | `c72b87d925bab8afc7690c18bea8f978736f9e3e91cdc8b17b5dcba37bae7bf2` |
| W4_SRC | V6 s20 u3040 step 8445120 | `0657f41030e54eac11816116af6c76b66d597c393fc72579a3755401c9dc79f3` | NA | `32928707acc81af215faa829ad94bba566804982bc192bb5116ea32d43d56e29` | `004dda2d9a846ce194d4b0b3f7adc408d5c9d909aadbe413a35e96ceaa411974` |
| W4_REP | + Arm-A | same | `724ed4c6a84fba07b6d9f11d535b8a0eb726daff21b9def9a5e60d9c7b972d03` | `e151b306f706e9db9828b3c9e742984a0ae836dbd97e73f86e4e2063bd7c5cd6` | `d11027323a71884775a64f29974ff8e4cfc5931b8cf531d32492330cc40e3008` |

* Reconstruction: the unmodified `run_gate_route_audit.build_state` (manifest blob `e5f0bb4fa68ebe1b200a3fcabe3544ebbff5483b`).
* Data: lexicon `ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66`; GloVe `91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed`; raw bank tensor `4658e11e6a8f60a468472cc3fa62e71064e33da6790f924cc3005e602762ddb4`.
* No other seed, checkpoint, witness or repaired state.

## 5. POPULATION

Identical to the closed contract §3:
* 29,571 lexical repetition rows, all evaluated;
* 27,981 canonical C targets;
* 1,590 non-canonical homophone rows;
* historical C correctness = exact lexical-row identity on P_C (NA elsewhere);
* separate lexical-identity and phonology booleans.

The retrieved row is the **immutable** `retrieved_index`. DOSE-J independently recomputes it with the frozen rule (`frozen_probe.encode_all` + `comprehension_metrics`, batch 512). There is no new retrieval definition.

## 6. DIRECTIONAL INTERVENTION

```
ŝ   = live native encoder output at model.ltm.to_semantic (the tensor S0 used)
v   = bank_raw[retrieved_index_i]   (RAW, unnormalized)
u_s = ŝ/‖ŝ‖,  u_p = v/‖v‖,  d = clip(u_s·u_p, −1, 1),  θ = arccos(d),  r = u_p − d·u_s,  norm_r = ‖r‖
s_α = ‖ŝ‖ · u_α        (native norm preserved; direction only)
```

**Injection.**
* A **new** supplier registered through the frozen `ventral_interface.injection.injected` context, on the **same** `model.ltm.to_semantic` forward hook, decoded by the frozen `ventral_interface.decode.decode_condition`.
* The frozen `ventral_interface/` package, its driver, contract, config, tests and closed outputs are **not modified**.
* The new code lives in `ventral_directional_dose/` and `scripts/ventral_directional_dose/`.

## 7. EXACT ALPHAS

**Scientific α ∈ {0.25, 0.50, 0.75, 1.00}.** No other value.

**α = 0 is not a scientific condition.**
* It exists only for DOSE-B and short-circuits **exactly** to `s_0 := ŝ`: the original live tensor object is returned, with no recast or reconstruction.
* In analyses, α = 0 is always read from the immutable S0 control.

## 8. SLERP MATHEMATICS

All interpolation arithmetic is float64, with exactly one deterministic cast to the model semantic dtype (float32).

**ORDINARY** (`norm_r > 1e-6`):

```
q = r / norm_r
u_α = cos(αθ)·u_s + sin(αθ)·q
u_α = u_α / ‖u_α‖
s_α = cast_float32(‖ŝ‖ · u_α)
```

**NEAR_COLLINEAR_NLERP** (`norm_r ≤ 1e-6 ∧ d > 0`):

```
w_α = (1−α)·u_s + α·u_p
u_α = w_α / ‖w_α‖
s_α = cast_float32(‖ŝ‖ · u_α)
```

**Hard stops:** see §9.

**Frozen tolerances:**

| name | value |
|---|---|
| EPS_NORM | 1e-6 |
| EPS_ORTHO | 1e-6 |
| TOL_NORM_REL | 1e-6 |
| TOL_ENDPOINT_COS | 1e-6 |
| TOL_ANGLE_RAD | 1e-6 |
| TOL_MONOTONE_RAD | 1e-9 |
| TOL_LIVE_SHAT | 1e-5 max abs, identical top-1 (closed SHAT rule) |

Angles are computed as `atan2(‖a − (a·b)b‖, a·b)` on unit float64 vectors built from the float32 values. Synthetic verification of the design-pass formulas is recorded in `provenance/DESIGN_PASS_EXTERNAL_ARTIFACTS.md`.

## 9. NUMERICAL HARD STOPS AND THE REMAINING FALLBACK (amended)

| condition on any REAL item (any state) | action |
|---|---|
| `‖ŝ‖ ≤ 1e-6` (**ZERO_SHAT**) | **HARD STOP**: preflight fails; no scientific α decoding |
| `‖v‖ ≤ 1e-6` (**ZERO_PROTOTYPE**) | **HARD STOP** |
| `norm_r ≤ 1e-6 ∧ d ≤ 0` (**NEAR_ANTIPODAL**) | **HARD STOP**: no basis fallback, no exclusion, no special stratum |
| `norm_r ≤ 1e-6 ∧ d > 0` | NEAR_COLLINEAR_NLERP (included in all denominators; also stratum NEAR_COLLINEAR_CASES) |
| `norm_r > 1e-6` | ORDINARY |

* The hard stops are evaluated over **all 4 × 29,571 real items** by the real-state geometry preflight (§15b), **before** any scientific α decode.
* The supplier also raises on these conditions if ever reached at runtime.
* `dose_case ∈ {ORDINARY, NEAR_COLLINEAR_NLERP}` only.

## 10. DECODING CONTRACT

Identical to the closed contract §4:
* isolated ventral route `ltm` only;
* frozen `gate_probe.ar_decode_free` (cap 12, imported and asserted) and `ar_decode_forced_length`;
* BOS 1, greedy argmax (first index on ties) with feedback, EOS 2 with a cut at the first EOS;
* free-AR breaks only when every row has EOS; forced-length uses a per-item `len+1` window;
* exact match = prediction == form;
* decode batch 256 in bank order, CPU, float32, deterministic algorithms, `eval()`, inference mode, no optimizer and no backward;
* passive motor-token recording checked against the historical returns;
* no gate, no FULL fusion, no dorsal readout (frozen `ventral_only` guard).

## 11. PRIMARY / SECONDARY READOUTS

PRIMARY: genuine free-AR isolated ventral repetition. SECONDARY: canonical forced-length. Never pooled.

## 12. ITEM-LEVEL SCHEMA (`item_level_directional_dose.tsv`; one row per state × item; exact column order frozen in `ventral_directional_dose/schema.py`)

* **Immutable identity and controls (copied and verified):**
  * `witness_id, state_id, source_or_repaired, seed, source_u, source_checkpoint_sha256, repaired_head_sha256, reconstructed_state_identity`
  * `item_index, lexical_identity, canonical_C_index, canonical_C_identity, in_C_population, target_phonology, phoneme_length, homophone_group, homophone_group_size`
  * `retrieved_index, retrieved_lexical_identity, retrieved_phonology, C_contract_correct, lexical_identity_correct, phonology_correct`
  * `S{0,1,2,3}_{freear,canonical}_exact_correct`, `prev_S1_rescue_{freear,canonical}`
* **Base geometry (α-independent, float64 from the live float32 ŝ):** `shat_norm, retrieved_raw_glove_norm, cos_us_up, theta_rad, norm_r, dose_case, live_shat_vs_preflight_shat_max_abs_dev`
* **Per α ∈ {a025, a050, a075, a100}:**
  * vector checks: `{a}_s_alpha_norm, {a}_norm_rel_err, {a}_cos_us_ualpha, {a}_cos_ualpha_up, {a}_angle_from_native_rad, {a}_angular_fraction, {a}_angle_err_rad` (`angle_err` is NA for near-collinear items; `angular_fraction` is NA if θ ≤ 1e-6)
  * per convention c: `{a}_{c}_exact_correct, {a}_{c}_predicted_phonology, {a}_{c}_pred_length, {a}_{c}_eos_emitted, {a}_{c}_first_eos_step, {a}_{c}_eos_before_target_length, {a}_{c}_eos_after_target_length, {a}_{c}_terminated_by_cap, {a}_{c}_first_divergence_step, {a}_{c}_transition_vs_S0, {a}_{c}_prev_S1_rescue, {a}_{c}_recovers_prev_S1_rescue`

No metric may be added after results.

## 13. SUMMARY SCHEMA (`summary_metrics_directional_dose.json`)

**Structure:** `results → state → convention → stratum → {a025, a050, a075, a100}`. Each leaf holds:

```
{ denominator, exact_count, exact_proportion,
  WRONG_TO_CORRECT, CORRECT_TO_WRONG, CORRECT_TO_CORRECT, WRONG_TO_WRONG,
  previous_S1_rescues, previous_S1_rescues_recovered, recovery_fraction,
  new_regressions (= CORRECT_TO_WRONG) }
```

Each stratum also carries `immutable_controls` = S0/S1/S2/S3 exact counts and the denominator, recomputed from the immutable item file.

**Strata:**

| stratum | definition |
|---|---|
| ALL_REPETITION_ITEMS | all 29,571 rows |
| C_POPULATION | canonical C targets |
| NONCANONICAL_HOMOPHONE_MEMBERS | the 1,590 non-canonical rows |
| NATIVE_LTM_WRONG | S0 wrong, same convention |
| C_CORRECT_AND_NATIVE_LTM_WRONG | closed definition |
| PREV_S1_RESCUES | S0 wrong ∧ S1 correct, same convention |
| NEAR_COLLINEAR_CASES | `dose_case == NEAR_COLLINEAR_NLERP` |

**Additional blocks:**
* `paired → witness → convention → alpha`:
  * SRC and REP exact;
  * SRC and REP rescue fraction (W→C / native failures);
  * SRC and REP regressions;
  * Jaccard of W→C rescue sets;
  * difference in previous-S1-rescue recovered fraction;
  * item-level SRC→REP transitions.
* `gates` and `hard_stop_preconditions`.
* `geometry_case_counts` per state.

## 14. SOURCE / REPAIR PAIRING

Paired reporting is mandatory for W3_SRC↔W3_REP and W4_SRC↔W4_REP, for every α and convention. Stability across repair is described from raw values, not thresholds.

## 15. VALIDITY GATES (all hard stops → `GATE_FAILURE.json` / blocker; no interpretation)

**Hard-stop preconditions (real-state geometry preflight, all states, before any scientific α decode):**
* `NO_ZERO_SHAT`
* `NO_ZERO_PROTOTYPE`
* `NO_NEAR_ANTIPODAL`

| gate | requirement |
|---|---|
| **DOSE-A** | State, control and data identity: SOURCE, head, archival and composite hashes; closed contract and outputs; lexicon, GloVe and bank. |
| **DOSE-B** | α = 0 through the **new** supplier reproduces immutable S0 item by item: exact_correct **and** predicted_phonology, **both** conventions, all 29,571 items per state (0 mismatches). |
| **DOSE-C** | For every item and scientific α: \|‖s_α‖/‖ŝ‖ − 1\| ≤ 1e-6. |
| **DOSE-D** | For every item: 1 − cos(s_1, v) ≤ 1e-6. |
| **DOSE-E** | ORDINARY items: \|∠(ŝ, s_α) − α·θ\| ≤ 1e-6 rad and ∠ nondecreasing in α within 1e-9 rad. NEAR_COLLINEAR items: ∠(ŝ, s_α) ≤ θ + 1e-6 and nondecreasing within 1e-9 rad. |
| **DOSE-F** | Shared downstream path: every decode runs through the frozen `decode_condition` (route `ltm`). The tensor entering `sem_to_h0` equals the supplied vector (bitwise, checked every bind). Recorded tokens reproduce the historical returns. Encoder output is bitwise stable across AR steps. |
| **DOSE-G** | The frozen `ventral_only` guard is never hit (no gate, FULL or dorsal). |
| **DOSE-H** | The closed freeze and results SHA256SUMS verify before and after, and the immutable controls hash exactly. |
| **DOSE-I** | Per-state `state_dict` SHA256 is unchanged before and after, and equal to the closed preflight values (§4). |
| **DOSE-J** | Recomputed top-1 retrieval equals the immutable `retrieved_index` for every item. The live ŝ in dose decodes equals the preflight ŝ within 1e-5 max abs, with identical top-1. |

### 15b. Real-state geometry / provenance preflight (mandatory, after implementation freeze)

For each state and all 29,571 items:
* reconstruct the state;
* DOSE-A, DOSE-I and DOSE-J;
* compute ŝ (frozen `encode_all`, batch 256, bank order), v, d, norm_r;
* count ZERO_SHAT, ZERO_PROTOTYPE, NEAR_ANTIPODAL, NEAR_COLLINEAR and ORDINARY;
* run DOSE-B with the α = 0 supplier, both conventions.

**No α ∈ {0.25, 0.50, 0.75, 1.00} is decoded in the preflight.**

If any hard stop or gate fails, write `CENTRAL_STEERING_BLOCKER_DIRECTIONAL_DOSE.md` with evidence and STOP (`SCIENTIFIC_EXECUTION=NO`).

**`--execute` refuses unless all of the following hold:**
* contract status FROZEN and SHA256 equal to the CLI value;
* every `IMPLEMENTATION_MANIFEST.json` hash matches;
* HEAD equals the CLI-provided implementation-freeze commit;
* `git status --porcelain` is empty;
* a fresh in-process real-state preflight (§15b) passes for all four states **before** any scientific decode;
* the scientific output directory does not exist.

Execution runs exactly once.

## 16. INTERPRETATION FAMILIES (CENTRAL)

* **DR1 — EARLY RESCUE.** Substantial rescue by α ≤ 0.50 with low regression. *Implication:* a small continuous correction may suffice; minimal semantic refinement becomes high priority.
* **DR2 — LATE / NEAR-PROTOTYPE RESCUE.** Large rescue mainly at α ≥ 0.75. *Implication:* the solution approaches prototype replacement; do not immediately build an attractor; training-only compatibility becomes preferred.
* **DR3 — NON-MONOTONIC / REGRESSION-HEAVY.** Intermediate movement creates substantial regressions or instability. *Implication:* do not prioritize prototype-directed refinement; training-only preferred.
* **DR4 — SMOOTH MONOTONIC DOSE RESPONSE.** Compatibility improves progressively toward the prototype. *Implication:* graded functional compatibility axis; a minimal differentiable refinement becomes justified.

**Binding rules:**
* No numeric threshold for "substantial", "large" or "low", invented before or after results.
* Raw curves first.
* If several families apply descriptively, say so.
* If states or conventions materially disagree, do not force one family.
* "Off-manifold" is inadmissible.

### 16b. α = 1 factorization (mandatory)

α = 1 (prototype direction + native norm) is compared against **S1** (prototype direction + prototype norm) and **S3** (native direction + prototype norm), per state and convention. Required fields:
* exact counts;
* α1 vs S1 exact difference and item disagreement count;
* α1 rescue overlap with S1 rescues;
* α1 regressions;
* a descriptive statement of whether prototype direction alone is sufficient despite the native norm.

No unpreregistered condition may be created.

### 16c. Programme question

The final handoff answers whether the frozen dose response supports prioritizing **T** (training-only compatibility) or **R** (minimal semantic refinement), or **NO_DECISIVE_PRIORITY**. This is a steering recommendation only; neither is implemented.

## 17. NOT_ESTABLISHED (before execution)

* The dose-response shape.
* Whether a small correction suffices.
* Whether refinement or an attractor is necessary.
* Why existing supervision leaves the decoder more compatible with prototype direction than with native ŝ direction.
* Generality beyond the four states.

### 17b. Training-audit boundary (binding)

Established historical facts (routing only; not gradient magnitude or causal effect):
* `L_dec` already supervises the decoder from encoder-produced ŝ;
* `L_align` already combines cosine and MSE alignment;
* Naming already supervises the decoder from raw GloVe;
* C retrieval trains the encoder and `to_semantic` but not the decoder;
* FULL repetition and `L_gate` send gradient through the confidence into ŝ.

None of "train decoder on ŝ", "add cosine alignment", "add MSE alignment" or "train from raw GloVe" may be proposed as new without identifying a materially new mechanism relative to V6. No such mechanism is designed in this workstream.

### 17c. Yair flag bookkeeping (binding)

`YAIR_FLAG_IDENTITY=SEMANTIC_ATTRACTOR_FLAG_CONFIRMED`: `semantic_attractor = True / False`, controlling a proposed recurrent semantic-refinement mechanism. Evidence is substantive but comes from paraphrased contemporary meeting notes; the exact wiring is unspecified. **The flag is not implemented and no attractor is implemented.**

## 18. STOP CONDITIONS

* Contract amendment → implementation → tests → freeze → real-state preflight → (if it passes) exactly-one execution → validation → analysis → handoff → **STOP**.
* Any hard stop or gate failure → STOP with blocker, no interpretation.
* Never: training, refinement, attractor, flag, gate change, lesion, or full-ceiling run.
