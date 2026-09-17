# CENTRAL_STEERING_HANDOFF_DIRECTIONAL_DOSE_DESIGN

**Workstream:** VENTRAL SEMANTIC DIRECTIONAL DOSE · FROZEN DIAGNOSTIC (design + audit only) · POST_STAGE / PAPER_PROGRAMME · 2026-09-17

```
DIRECTIONAL_DOSE_DESIGN=COMPLETE
CONTRACT_STATUS=DESIGN_COMPLETE_NOT_EXECUTION_AUTHORIZED
IMPLEMENTATION_AUTHORIZED=NO
RETURN_TO_CENTRAL=YES
```

## Status and lineage reused
* **Design worktree / branch:** `/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/wt-ventral-directional-dose` · `paper-programme/ventral-directional-dose-design`, created from the results commit `0f25b5b87efd8038dfc4c86b5ca2935bf7b178c7` (clean, verified ancestry). The design commit SHA is reported in the final response; a file cannot contain its own commit hash.
* **Closed diagnostic reused read-only:**
  * freeze commit `4ad20048e20da84b9f22a92965098b84c2bf7dd6`;
  * results commit `0f25b5b8…`;
  * contract `a5ca7b83717b43ab77cedae0cd3e4bef17901ffff051f5ca90b9340b45bda5d3`;
  * item-level `37f2fb17…d175663`, summary `455ccaf0…10c80`, AR `75106959…fdf027`.
  * Both closed SHA256SUMS files (37 and 17 entries) re-verified: 0 failures. The closed worktree is unmodified.
* **Same four states** W3_SRC / W3_REP / W4_SRC / W4_REP (identities `6d728285…`, `9185aa56…`, `32928707…`, `e151b306…`). Same lexicon, GloVe and raw bank. Same 29,571 / 27,981 / 1,590 populations and homophone rules. Same isolated-ventral free-AR (cap 12) and forced-length evaluators, BOS/EOS and scoring.

## Exact dose contract (`DIRECTIONAL_DOSE_EXPERIMENT_CONTRACT.md`)
* **Intervention:**
  * `u_s = ŝ/‖ŝ‖`;
  * `u_p = v/‖v‖`, where `v` = raw GloVe of the **immutable** frozen top-1 retrieved row;
  * `u_α = SLERP(u_s, u_p, α)`;
  * **`s_α = ‖ŝ‖·u_α`**, so the native norm is preserved and only direction changes.
* **α ∈ {0.25, 0.50, 0.75, 1.00} only.** α = 0 is S0 (immutable), used only inside gate DOSE-B.
* **α = 1.00 is NOT S1:** it has prototype direction with the native norm. S1 (prototype direction, prototype norm) and S3 (native direction, prototype norm) remain immutable controls, reported alongside.
* **Readouts:** primary genuine free-AR, secondary canonical forced-length, never pooled. The downstream path is identical to the closed diagnostic (same `to_semantic` hook site via a new supplier, `gate_probe` decoders, route `ltm`). No gate, no FULL fusion, no dorsal readout.
* **Reported per state × α × convention × stratum:**
  * denominator, exact count and proportion;
  * the four transitions vs S0;
  * previous-S1-rescues recovered (count and fraction), with S1 rescues defined only from the immutable file;
  * new regressions.
* **Strata:** ALL, C_POPULATION, NONCANONICAL_HOMOPHONE_MEMBERS, NATIVE_LTM_WRONG, C_CORRECT_AND_NATIVE_LTM_WRONG, PREV_S1_RESCUES, FALLBACK_CASES.
* **Paired SRC↔REP block** for W3 and W4, as raw sequences with no thresholds.
* **Item-level schema:** full identity discipline, immutable S0–S3 correctness and previous-S1-rescue flags, geometry (‖ŝ‖, ‖s_α‖, cos(u_s,u_p), cos(u_s,u_α), cos(u_α,u_p), angular fraction, fallback flag) and per-α EOS and transition fields. No post-hoc metrics.
* **Interpretation:** DR1–DR4 embedded. No numeric cutoffs; raw curves and counts are exposed. States that disagree are not forced into one family, and "off-manifold" is not admissible.

## Numerical fallback decision (preregistered before any dose result)
* **Ordinary case** (`‖r‖ = sin θ > 1e-6`): float64 SLERP via the orthogonal-component form `u_α = cos(αθ)u_s + sin(αθ)·r/‖r‖`, renormalised, then cast once to float32. **α = 0 short-circuits to ŝ exactly.**
* **Zero norm:**
  * `‖ŝ‖ ≤ 1e-6` → `ZERO_SHAT`: ŝ is kept unchanged, and the item is excluded from α denominators and counted (consistent with closed S3).
  * `‖v‖ ≤ 1e-6` → HARD STOP (impossible under the frozen bank, whose minimum norm is 2.61).
* **Near-collinear** (`‖r‖ ≤ 1e-6`, d > 0): normalized linear interpolation (NLERP).
* **Near-antipodal** (`‖r‖ ≤ 1e-6`, d ≤ 0):
  * fixed great circle through `q ∝ e_k − u_s[k]·u_s`, with `k = argmin|u_s[k]|` (first index on ties);
  * α = 1 is explicitly assigned `u_1 := u_p`.
* **Synthetic verification only** (outside git, hashes in `provenance/DESIGN_PASS_EXTERNAL_ARTIFACTS.md`):
  * 80,000 ordinary evaluations: norm error ≤ 1.2e-8 (after float32), endpoint cosine deficit ≤ 8.9e-16, angle error ≤ 6.4e-9 rad, 0 monotonicity violations;
  * all zero, collinear and antipodal cases deterministic and finite, with correct endpoints.
* **Frozen tolerances:** 1e-6 (norm), 1e-6 (endpoint cosine), 1e-6 rad (angle), 1e-9 rad (monotonicity).

## Future validity gates (all hard stops → `GATE_FAILURE.json`, no interpretation, return to CENTRAL)
* **DOSE-A:** state, control, data and composite hashes.
* **DOSE-B:** the α = 0 short-circuit, through the new supplier, reproduces immutable S0 item by item (exact **and** predicted phonology, both conventions).
* **DOSE-C:** norm preservation ≤ 1e-6 relative.
* **DOSE-D:** α = 1 direction, 1 − cos ≤ 1e-6.
* **DOSE-E:** angle = α·θ within 1e-6 rad, monotone in α.
* **DOSE-F:** shared downstream path.
* **DOSE-G:** no gate, FULL or dorsal access.
* **DOSE-H:** immutable prior outputs unchanged, before and after.
* **DOSE-I:** parameter `state_dict` hashes unchanged and equal to the closed preflight values.
* **DOSE-J:** retrieved identity equals the immutable file; live ŝ is consistent with the retrieval ŝ.

## Training supervision already present in V6 (`VENTRAL_INTERFACE_TRAINING_SUPERVISION_AUDIT.md`)
**Recipe, read from both V6 checkpoints:**
* j0, interleaved 1R:2N:3C per 6-step macro-cycle, one shared AdamW;
* LR R 3e-5 / N 3e-5 / C 1e-4 (since step 2,361,300);
* loss weights rep 1.0, align 1.0, **dec 0.5**, wm 0.5, gate 0.05; pool 0.5;
* λ_N 1.0, λ_C 0.087, τ 0.1, c_align 0.0, teacher forcing 1.0.

**Active supervision:**
* **Decoder from encoder-produced ŝ:** `L_dec` (0.5, teacher-forced, R steps, 1/6 of steps). FULL `L_rep` also trains the decoder from ŝ via the g-weighted ventral premotor. → **already present**.
* **Semantic alignment:** `L_align = (1−cos(ŝ,g)) + 0.1·MSE(ŝ,g)` to the item's raw GloVe (1.0, R steps). → **already present and active**. C-step alignment exists in code but is **inactive** (weight 0.0).
* **Decoder from raw GloVe:** Naming CE (λ_N 1.0, **2/6 of steps**), encoder bypassed. → **already present**, with twice the step share of ŝ-driven decoder supervision.
* **Retrieval CE** (normalized ŝ vs normalized bank, τ 0.1, 3/6 of steps, LR 1e-4): trains encoder, `to_semantic` and phon_embed only. **No C gradient reaches the decoder.** AdamW skips parameters whose grad is None.

## Gradient-routing headline (verified by a synthetic backprop probe on the V6 architecture; no training)

| path | reaches | does not reach |
|---|---|---|
| encoder / to_semantic | L_rep, L_dec, L_align, L_gate, retrieval CE | Naming, L_wm, pool |
| ventral decoder / sem_to_h0 / dec_to_premotor | L_rep, L_dec, **Naming** | retrieval, L_align, L_gate |
| motor | L_rep, L_dec, Naming, L_wm, pool | — |
| WM | L_rep, L_wm, pool | — |
| phon_embed | every term | — |

* **Gate:** no parameters and nothing detached. **FULL L_rep and L_gate send gradient into ŝ through c_LTM**, so the gate can shape semantic direction during training. Its magnitude is not quantified.

**Question for later, not answered here:** why do these existing pressures leave the decoder far more compatible with prototype direction than with native ŝ direction?

## Yair flag archaeology (`YAIR_FLAG_PROVENANCE_AUDIT.md`)
**FLAG_PROPOSAL_STATUS=DOCUMENTED_SUBSTANTIVELY**

* **Source:** Louis's paraphrased post-meeting notes of 14 Sept 2026 (Yair × Emmanuel × Louis), `~/Downloads/MEETING_NOTES_YAIR_EMMANUEL_2026-09-14.md`, SHA256 `c1983db3…b70e`. A read-only archival copy is kept outside git. The same hash is already cited in the GATING recap.
* **§3 "Proposition de Yair : dynamique d'attracteur sémantique":** "Ajouter un flag du type : `semantic_attractor = True / False`" (False = current behaviour; True = recurrent semantic loop).
  * **Variable:** a switch for recurrent refinement of ŝ.
  * **Entry point:** the ventral route; ŝ is fed back through GloVe or semantic retrieval into `z_LTM`, producing a new ŝ. Framed as a change to "gating / ventral dynamics".
  * **Problem:** ŝ "pas nécessairement suffisamment précise" (not necessarily precise enough).
  * **Status in the notes:** "PROPOSITION À TESTER". The loop's definition, what is fed back, iterations, gradient flow and gate interaction were left unspecified.
* **Not identified as the flag:** the fixed 0.5/0.5 gate baseline, gate saturation constraints, threshold-as-hyperparameter and the bilateral gate (all POSSIBLE_BUT_NOT_IDENTIFIED_AS_FLAG). Task identity, lexicality and the tied lesion mask are NOT_RECOVERED as flag.
* **EXISTING_GATE, documented separately:** `c_LTM = max_j cos(ŝ, bank_j)`, `g = sigmoid(2.0·(c_LTM − 0.7))`, `premotor = g·ltm + (1−g)·wm`, with purpose "error-suppression (lexicality routing)". No source equates it with the flag.

## Unresolved facts
1. The flag loop mechanism is undefined in the only source, and that source is a paraphrase. No verbatim Yair wording was found.
2. CENTRAL lists "attractor" and "flag" as separate prohibitions. No local source documents a flag distinct from `semantic_attractor`.
3. The meeting notes are outside the repository and the Source-of-Truth. Gmail and Drive were not searchable in this session.
4. The V6 recipe before step 1,389,000 is known only via the `old_*` fields of the first recorded transition.
5. Loss and gradient **magnitudes** in real V6 training were not examined; routing only.
6. No dose-response result exists. Every DR family remains open.

## Confirmation
```
SCIENTIFIC_EXECUTION=NO
TRAINING_RUN=NO
ARCHITECTURE_CHANGED=NO
GATE_CHANGED=NO
LESIONING_RUN=NO
DOSE_DRIVER_IMPLEMENTED=NO
ALPHA_CONDITIONS_EVALUATED=NO
ATTRACTOR_OR_FLAG_IMPLEMENTED=NO
```

---

## READY_TO_PASTE_CENTRAL_PROMPT

```
CENTRAL STEERING — ARBITRATION REQUEST
LICHTHEIM3 — VENTRAL SEMANTIC DIRECTIONAL DOSE — DESIGN + AUDIT PASS (NO EXECUTION)
Programme: POST_STAGE / PAPER_PROGRAMME

Context: the VENTRAL SEMANTIC INTERFACE diagnostic is CLOSED_RESULT_SAFE (freeze 4ad20048…, results
0f25b5b87efd8038dfc4c86b5ca2935bf7b178c7, contract a5ca7b83…bda5d3). Accepted localization: the residual
isolated-ventral repetition deficit is at the encoder ŝ → ventral-decoder interface; raw prototype
substitution (S1) restores decoding (99.83–100%), norm-only substitution (S3) does not (80.6–83.5%, below
native 87.6–89.2%); an attractor is not established as necessary.

This pass produced, on branch paper-programme/ventral-directional-dose-design (from 0f25b5b8…), a
documentation-only package; nothing was executed, trained, or implemented.

1) DIRECTIONAL-DOSE CONTRACT (CONTRACT_STATUS=DESIGN_COMPLETE_NOT_EXECUTION_AUTHORIZED)
   s_α = ‖ŝ‖ · SLERP(ŝ/‖ŝ‖, v/‖v‖, α), v = RAW GloVe of the immutable frozen top-1 retrieved row;
   α ∈ {0.25, 0.50, 0.75, 1.00} only; native norm preserved, only direction changes.
   α=1.00 ≠ S1 (prototype direction with native norm); S0/S1/S2/S3 remain immutable controls read from
   the closed files; "previous S1 rescue" = S0 wrong ∧ S1 correct from the immutable item-level TSV.
   Same four states (W3_SRC/W3_REP/W4_SRC/W4_REP), same populations (29,571 / 27,981 / 1,590), same
   isolated-ventral free-AR (primary, cap 12) and forced-length (secondary) evaluators, no gate/FULL/dorsal.
   Reported per state × α × convention × stratum: denominator, exact, 4 transitions vs S0, previous S1
   rescues recovered (count/fraction), new regressions; strata incl. C_CORRECT_AND_NATIVE_LTM_WRONG and
   PREV_S1_RESCUES; mandatory paired SRC↔REP block; raw dose sequences, no fitted thresholds.
   Interpretation families DR1–DR4 embedded without numeric cutoffs; disagreement across states → not forced.

2) SLERP NUMERICAL FALLBACK (float64, cast once to float32; tolerances frozen before any result)
   ordinary (sin θ > 1e-6): u_α = cos(αθ)u_s + sin(αθ)·r/‖r‖, r = u_p − d·u_s; α=0 short-circuits to ŝ.
   ‖ŝ‖ ≤ 1e-6 → ZERO_SHAT (unchanged, excluded, counted); ‖v‖ ≤ 1e-6 → HARD STOP.
   near-collinear (sin θ ≤ 1e-6, d>0) → NLERP; near-antipodal (d≤0) → fixed circle via
   q ∝ e_k − u_s[k]u_s, k = argmin|u_s[k]| (first index), α=1 assigned u_p.
   Synthetic-only verification: 80,000 cases, norm err ≤1.2e-8 (float32), endpoint cos deficit ≤8.9e-16,
   angle err ≤6.4e-9 rad, monotone; all fallback cases deterministic.
   Future hard-stop gates DOSE-A…J: identities/hashes, α=0 reproduces S0 item-by-item (exact AND phonology),
   norm ≤1e-6 rel, α=1 direction 1−cos ≤1e-6, angle=αθ ±1e-6 rad and monotone, shared downstream path,
   no forbidden access, immutable prior outputs, parameter hashes unchanged, retrieval identity reuse.

3) TRAINING-SUPERVISION RECONSTRUCTION (V6 recipe from both checkpoints + synthetic gradient-routing probe)
   j0, interleaved 1R:2N:3C, shared AdamW, LR R 3e-5 / N 3e-5 / C 1e-4; weights rep 1.0, align 1.0,
   dec 0.5, wm 0.5, gate 0.05, pool 0.5; λ_N 1.0, λ_C 0.087, τ 0.1; c_align 0.0; teacher forcing 1.0.
   ALREADY ACTIVE: L_dec trains the ventral decoder from encoder-produced ŝ (R steps, 1/6 of steps; FULL L_rep
   also, via g-weighted ventral premotor); L_align = (1−cos) + 0.1·MSE to raw GloVe on ŝ (R steps); Naming
   CE trains sem_to_h0/decoder/premotor/motor/phon_embed from raw GloVe on 2/6 of steps (encoder bypassed);
   retrieval CE (normalized ŝ vs normalized bank) trains encoder/to_semantic/phon_embed only (no decoder
   gradient; AdamW skips None-grad params). Gate has no parameters and is not detached: FULL L_rep and L_gate
   send gradient into ŝ through c_LTM. C-step alignment is supported by code but inactive (weight 0).
   Open (not answered): why these pressures leave the decoder far more compatible with prototype direction
   than with native ŝ direction.

4) YAIR FLAG PROVENANCE: FLAG_PROPOSAL_STATUS=DOCUMENTED_SUBSTANTIVELY
   Only source: Louis's paraphrased post-meeting notes, 14 Sept 2026 (Yair × Emmanuel × Louis),
   MEETING_NOTES_YAIR_EMMANUEL_2026-09-14.md, SHA256 c1983db3…b70e (outside repo; archived read-only).
   §3 "Proposition de Yair : dynamique d'attracteur sémantique": "Ajouter un flag du type :
   semantic_attractor = True / False" (False = current; True = recurrent semantic loop). Variable: switch for
   recurrent refinement of ŝ; entry: ventral route, ŝ fed back via GloVe/semantic retrieval into z_LTM → new
   ŝ; problem: ŝ "pas nécessairement suffisamment précise". Mechanism explicitly unspecified; status
   "PROPOSITION À TESTER". Not identified as the flag: fixed 0.5/0.5 baseline, gate saturation constraints,
   threshold hyperparameter, bilateral gate; task identity / lexicality / tied lesion mask NOT_RECOVERED.
   EXISTING_GATE (c_LTM = max cos(ŝ, bank); g = sigmoid(2.0·(c_LTM − 0.7)); premotor = g·ltm + (1−g)·wm;
   "error-suppression / lexicality routing") is documented separately; no source equates it with the flag.
   Unresolved: CENTRAL's separate "attractor" and "flag" prohibitions — no local source documents a flag
   distinct from semantic_attractor.

CENTRAL is asked to:
1. Accept, modify or reject the directional-dose contract.
2. Accept, modify or reject the SLERP numerical fallback and frozen tolerances.
3. Arbitrate the training-supervision reconstruction and gradient-routing table.
4. Arbitrate the flag provenance result (including whether CENTRAL's "flag" is semantic_attractor or a
   different, still unrecovered concept).
5. Decide whether implementation + single execution of the frozen dose diagnostic is authorized (which would
   require re-freezing contract + implementation with hashes before any α run).
6. Forbid training, architecture, gate, lesion, attractor/flag and full-ceiling work even if dose execution is
   authorized, until CENTRAL separately selects the next scientific workstream.
```
