# CENTRAL_STEERING_HANDOFF_DIRECTIONAL_DOSE_RESULTS

**Workstream:** LICHTHEIM3 — VENTRAL SEMANTIC DIRECTIONAL DOSE — FROZEN DIAGNOSTIC · POST_STAGE / PAPER_PROGRAMME · 2026-09-17

```
SCIENTIFIC_EXECUTION=COMPLETE
ALL_HARD_STOP_PRECONDITIONS=PASS
ALL_VALIDITY_GATES=PASS
DR_CLASSIFICATION=DR3 (non-monotonic / regression-heavy) with DR2-like endpoint concentration
STEERING_RECOMMENDATION=PRIORITIZE_T
```

| item | value |
|---|---|
| amended contract (FROZEN) SHA256 | `f5399058d4bf53c313fc4bd179c88b1cebb35348fecf74192a67bb107ec7608c` |
| contract-amendment commit | `1108ea6b8d35d6fed1c530d47153e5d828f20ffc` |
| implementation-freeze commit | `ea1098771ac52cf3a4c591ba650e58d73b9836e4` |
| results-only commit | reported in the final response (a file cannot contain its own commit SHA); frozen files are byte-identical to `ea109877…` |
| exactly-once evidence | `scientific_execution/logs/02_execute_attempt1.log` is the single attempt (2026-09-17T00:38:27Z → 00:55:31Z, exit 0, `EXECUTE=COMPLETE`). `--execute` refused to start without a clean tree, HEAD == freeze, matching contract/manifest hashes and an absent output directory. No rerun. |
| lineage | design `0b758696…` → closed results `0f25b5b8…` → closed freeze `4ad20048…`; closed S0–S3 outputs unchanged (DOSE-H pre/post) |

## Real-state geometry preflight (4 × 29,571)
* **ORDINARY 118,284 · NEAR_COLLINEAR 0 · ZERO_SHAT 0 · ZERO_PROTOTYPE 0 · NEAR_ANTIPODAL 0.**
* cos(u_s, u_p) ranges over [0.380, 0.913], median 0.738–0.747; min sin θ 0.408.
* DOSE-B: α = 0 through the new supplier reproduces S0 item by item, with 0 exact and 0 phonology mismatches in both conventions.

## Validity gates
| gate | result (all four states) |
|---|---|
| NO_ZERO_SHAT / NO_ZERO_PROTOTYPE / NO_NEAR_ANTIPODAL | pass |
| DOSE-A identity · DOSE-H immutable prior outputs (pre+post) | pass |
| DOSE-B α=0 ≡ S0 | 0 mismatches |
| DOSE-C norm | max relative error ≤ 2.13e-8 |
| DOSE-D α=1 direction | 1−cos ≤ 6.7e-16 |
| DOSE-E angle | error ≤ 7.6e-9 rad; 0 monotonicity violations |
| DOSE-F shared path | sem_to_h0 input ≡ supplied vector; encoder step dev 0.0; identical vectors across conventions |
| DOSE-G | 0 guard hits |
| DOSE-I | parameters unchanged (= closed values) |
| DOSE-J | retrieval 0 mismatches; live ŝ ≡ preflight ŝ (0.0) |

## Primary free-AR raw dose curves (exact / 29,571)
| state | α0 = S0 | .25 | .50 | .75 | 1.00 | S1 | S3 |
|---|---|---|---|---|---|---|---|
| W3_SRC | 26,372 (89.18%) | 20,447 (69.15%) | 13,020 (44.03%) | 14,380 (48.63%) | 24,083 (81.44%) | 29,529 (99.86%) | 24,684 (83.47%) |
| W3_REP | 26,378 (89.20%) | 20,441 (69.13%) | 13,052 (44.14%) | 14,398 (48.69%) | 24,112 (81.54%) | 29,571 (100%) | 24,650 (83.36%) |
| W4_SRC | 25,924 (87.67%) | 18,718 (63.30%) | 11,034 (37.31%) | 12,161 (41.12%) | 22,949 (77.61%) | 29,520 (99.83%) | 23,867 (80.71%) |
| W4_REP | 25,909 (87.62%) | 18,727 (63.33%) | 11,014 (37.25%) | 12,205 (41.27%) | 22,991 (77.75%) | 29,571 (100%) | 23,842 (80.63%) |

**Forced-length:** identical counts in every stratum; 0 item-level disagreements with free-AR at every α.

## Previous-S1-rescue recovery and regressions (free-AR)
| state | recovery .25 / .50 / .75 / 1.00 | regressions (C→W) .25 / .50 / .75 / 1.00 |
|---|---|---|
| W3_SRC | 23.26% / 17.87% / 28.45% / 74.40% (of 3,195) | 6,668 / 13,923 / 12,901 / 4,666 |
| W3_REP | 23.46% / 18.04% / 28.69% / 74.35% (of 3,193) | 6,686 / 13,902 / 12,896 / 4,640 |
| W4_SRC | 22.19% / 14.72% / 21.80% / 71.72% (of 3,642) | 8,016 / 15,427 / 14,557 / 5,587 |
| W4_REP | 21.68% / 14.17% / 21.76% / 71.68% (of 3,662) | 7,976 / 15,414 / 14,501 / 5,543 |

**C_CORRECT_AND_NATIVE_LTM_WRONG** (S0 0%, S1 100%) at α .25 / .50 / .75 / 1.00:
* W3_SRC 23.20 / 17.72 / 28.20 / 74.21%
* W3_REP 23.39 / 17.89 / 28.43 / 74.17%
* W4_SRC 22.22 / 14.71 / 21.78 / 71.65%
* W4_REP 21.75 / 14.17 / 21.78 / 71.64%

## α = 1 vs S1 vs S3 (free-AR; rescues / regressions)
| state | S3: native dir + proto norm | α=1: proto dir + native norm | S1: proto dir + proto norm | α=1 ⊂ S1 rescues? |
|---|---|---|---|---|
| W3_SRC | 397 / 2,085 | 2,377 / 4,666 | 3,195 / 38 | yes (0 α1-only) |
| W3_REP | 382 / 2,110 | 2,374 / 4,640 | 3,193 / 0 | yes |
| W4_SRC | 488 / 2,545 | 2,612 / 5,587 | 3,642 / 46 | yes |
| W4_REP | 489 / 2,556 | 2,625 / 5,543 | 3,662 / 0 | yes |

**Prototype direction alone is not sufficient** while the native norm is kept (α1 − S1 exact = −5,446 to −6,580). Norm alone (S3) is not sufficient either. Only the joint prototype vector (S1) is near-universally compatible.

## SOURCE ↔ POST_REPAIR
Curves nearly coincide at every α:
* W3 exact: 20,447/20,441, 13,020/13,052, 14,380/14,398, 24,083/24,112.
* W4 exact: 18,718/18,727, 11,034/11,014, 12,161/12,205, 22,949/22,991.
* Rescue-set Jaccard 0.87–0.96.
* |Δ previous-S1 recovery| ≤ 0.54 percentage points.

**Stable across Arm-A.**

## Best-supported DR family
* **DR3 (non-monotonic / regression-heavy): supported.** Accuracy is below S0 at every α, with the trough at α = 0.50 (37–44%). Regressions peak at α = 0.50 (52.7–59.5% of natively correct items). Rescue and recovery dip at 0.50 in every state.
* **DR2-like feature:** rescue is concentrated at α = 1 (72–74% of S1 rescues) but remains incomplete and regression-laden without the prototype norm.
* **DR1:** not supported (≤ 23.5% rescue by α ≤ 0.50 with 9–30× more regressions than rescues).
* **DR4:** not supported (non-monotonic).
* All states and both readouts agree.

## Recommendation: PRIORITIZE_T
Rationale:
1. **Continuous movement is harmful.** Continuous, norm-preserving movement of ŝ toward the prototype is harmful at every intermediate dose (DR3: "do not prioritize prototype-directed refinement; training-only preferred").
2. **Benefit only at the prototype.** Benefit appears only at the endpoint, and full compatibility needs the complete prototype vector (DR2: "approaches prototype replacement; training-only preferred; do not build an attractor").
3. **No graded axis.** The decoder is compatible with native ŝ (~88–89%) and with full prototypes (~100%) but not with the great-circle path between them, so there is no graded functional axis that a minimal refinement could follow.
4. **Stable across repair and witnesses.**

Qualifications:
* **Joint and discrete paths untested.** A joint direction+norm path or a discrete snap to the prototype was not tested (NOT_ESTABLISHED).
* **S1 is not minimal refinement.** S1 prototype substitution (~100%) is a retrieval-substitution scheme, not "minimal refinement"; its architectural status is a separate CENTRAL question.
* **Training-audit boundary.** A T intervention must name a materially new mechanism relative to V6. `L_dec` from ŝ, cosine + MSE `L_align`, Naming from raw GloVe, and FULL/gate gradient into ŝ already exist. None is designed here.

## NOT_ESTABLISHED
* Smoothness of joint direction+norm or discrete-snap paths.
* The cause of decoder incompatibility with intermediate vectors ("off-manifold" not concluded).
* That any specific T mechanism would work.
* Necessity or non-necessity of a semantic attractor or the Yair `semantic_attractor` flag (neither implemented or tested).
* Generality beyond these two seeds and four states.
* FULL-fusion or behavioural implications (isolated ventral route only).

## Confirmations
```
TRAINING_RUN=NO
ARCHITECTURE_CHANGED=NO
ATTRACTOR_IMPLEMENTED=NO
YAIR_FLAG_IMPLEMENTED=NO
GATE_CHANGED=NO
LESIONING_RUN=NO
FULL_CEILING_RUN=NO
```

---

## READY_TO_PASTE_CENTRAL_PROMPT

```
CENTRAL STEERING — ARBITRATION REQUEST
LICHTHEIM3 — VENTRAL SEMANTIC DIRECTIONAL DOSE — FROZEN DIAGNOSTIC (EXECUTED)
Programme: POST_STAGE / PAPER_PROGRAMME

Status: SCIENTIFIC_EXECUTION=COMPLETE, executed exactly once from implementation-freeze commit
ea1098771ac52cf3a4c591ba650e58d73b9836e4 (branch paper-programme/ventral-directional-dose-design; contract
amendment 1108ea6b8d35d6fed1c530d47153e5d828f20ffc applying CENTRAL's binding amendments; frozen contract
SHA256 f5399058d4bf53c313fc4bd179c88b1cebb35348fecf74192a67bb107ec7608c). Closed ventral-interface S0–S3
outputs (0f25b5b8…) reused read-only and verified unchanged before and after. No training, no architecture,
gate, lesion, attractor or Yair-flag work.

Intervention: s_α = ‖ŝ‖ · SLERP(ŝ/‖ŝ‖, v/‖v‖, α), v = raw GloVe of the immutable retrieved row;
α ∈ {0.25, 0.50, 0.75, 1.00}; native norm preserved (α=1 = prototype direction + native norm, ≠ S1).
States W3_SRC/W3_REP (V6 s19 u3825 ± Arm-A), W4_SRC/W4_REP (V6 s20 u3040 ± Arm-A); 29,571 items each;
isolated ventral route; genuine free-AR primary, forced-length secondary.

Real-state preflight: 118,284/118,284 ORDINARY; 0 ZERO_SHAT, 0 ZERO_PROTOTYPE, 0 NEAR_ANTIPODAL,
0 NEAR_COLLINEAR; cos(u_s,u_p) ∈ [0.380, 0.913]. All gates pass: DOSE-B α=0 reproduces S0 item-by-item
(exact + phonology, both conventions, 0 mismatches); norm error ≤2.1e-8; α=1 direction 1−cos ≤6.7e-16;
angle error ≤7.6e-9 rad, monotone; sem_to_h0 input ≡ supplied vector; no gate/FULL/dorsal access;
parameters unchanged; retrieval identity exact; live ŝ ≡ preflight ŝ.

RESULTS (free-AR; forced-length identical, 0 item disagreements):
Exact / 29,571 at α = 0(S0) / .25 / .50 / .75 / 1.00 | S1 | S3
 W3_SRC 89.18 / 69.15 / 44.03 / 48.63 / 81.44% | 99.86 | 83.47
 W3_REP 89.20 / 69.13 / 44.14 / 48.69 / 81.54% | 100   | 83.36
 W4_SRC 87.67 / 63.30 / 37.31 / 41.12 / 77.61% | 99.83 | 80.71
 W4_REP 87.62 / 63.33 / 37.25 / 41.27 / 77.75% | 100   | 80.63
Regressions (S0 correct→wrong) at .25/.50/.75/1.00: W3_SRC 6,668/13,923/12,901/4,666;
 W3_REP 6,686/13,902/12,896/4,640; W4_SRC 8,016/15,427/14,557/5,587; W4_REP 7,976/15,414/14,501/5,543.
Previous-S1-rescue recovery at .25/.50/.75/1.00: W3_SRC 23.3/17.9/28.5/74.4%; W3_REP 23.5/18.0/28.7/74.4%;
 W4_SRC 22.2/14.7/21.8/71.7%; W4_REP 21.7/14.2/21.8/71.7%. C_CORRECT_AND_NATIVE_LTM_WRONG shows the same
 non-monotonic curve (e.g. W3_SRC 23.2/17.7/28.2/74.2%; S0 0%, S1 100%).
Direction×norm (rescues/regressions): S3 (native dir + proto norm) 382–489 / 2,085–2,556; α=1 (proto dir +
 native norm) 2,374–2,625 / 4,640–5,587, α=1 rescues ⊂ S1 rescues in every state; S1 (proto dir + proto norm)
 3,193–3,662 / 0–46. Neither direction nor norm alone reproduces S1.
SOURCE vs POST_REPAIR: curves nearly identical at every α (rescue-set Jaccard 0.87–0.96; |Δ recovery| ≤0.54 pp).

Executing agent's classification: DR3 (non-monotonic / regression-heavy) with a DR2-like concentration of
rescue at α=1; DR1 and DR4 not supported; all states and conventions agree.
Steering recommendation: PRIORITIZE_T — continuous norm-preserving directional correction is harmful at all
intermediate doses and incomplete at the endpoint; compatibility requires the full prototype vector; no graded
axis for a minimal refinement was found. Qualifications: joint direction+norm or discrete-snap paths were not
tested; S1 prototype substitution (~100%) is a retrieval-substitution scheme whose architectural status is a
separate decision; per the binding audit boundary, V6 already has L_dec from ŝ, cosine+MSE L_align, Naming
from raw GloVe and FULL/gate gradient into ŝ, so any T intervention must name a materially new mechanism.
Not established: cause of the incompatibility ("off-manifold" not concluded), any specific T mechanism's
efficacy, attractor / semantic_attractor-flag necessity, generality beyond two seeds, FULL-fusion effects.

CENTRAL is asked to:
1. Accept or reject the directional-dose result (execution validity, gates, packaging).
2. Accept or reject the DR classification (DR3 with DR2-like endpoint concentration).
3. Arbitrate PRIORITIZE_T vs PRIORITIZE_R vs NO_DECISIVE_PRIORITY.
4. Decide the next single scientific workstream.
5. Before any T intervention, require a reconstruction of the existing V6 supervision (L_dec, L_align, Naming,
   retrieval, FULL/gate gradients) and identification of a materially new mechanism relative to it.
6. Before any R intervention (including prototype substitution or attractor/semantic_attractor flag), require a
   separately frozen architecture proposal.
7. Preserve the publication-path constraint.
8. Forbid execution of any next intervention (training, refinement, attractor, flag, gate, lesion, full-ceiling)
   until CENTRAL explicitly authorizes it.
```
