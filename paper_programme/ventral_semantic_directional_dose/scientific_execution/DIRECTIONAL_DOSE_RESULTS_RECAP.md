# DIRECTIONAL_DOSE_RESULTS_RECAP — LICHTHEIM3 VENTRAL SEMANTIC DIRECTIONAL DOSE (executed)

SCIENTIFIC_EXECUTION=COMPLETE · every hard-stop precondition and validity gate PASS on all four states · one execution attempt.

Labels: **[STRUCTURAL_CODE_FACT]** · **[EMPIRICAL_RESULT]** · **[INTERPRETATION]** · **[NOT_ESTABLISHED]**.
All numbers come from the frozen outputs through the read-only `analysis/build_results_package.py`, and every plotted value is in `figure_source_data/`. Free-AR and forced-length results are **identical** item by item (§6), so the tables show the primary (free-AR) values.

| provenance | value |
|---|---|
| worktree / branch | `/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/wt-ventral-directional-dose` · `paper-programme/ventral-directional-dose-design` |
| contract (FROZEN) SHA256 | `f5399058d4bf53c313fc4bd179c88b1cebb35348fecf74192a67bb107ec7608c` |
| contract-amendment commit | `1108ea6b8d35d6fed1c530d47153e5d828f20ffc` |
| implementation-freeze commit (HEAD at execution) | `ea1098771ac52cf3a4c591ba650e58d73b9836e4` |
| standalone real-state preflight | 2026-09-17T00:34:12Z → 00:38:08Z, `PREFLIGHT=PASS` (`logs/standalone_preflight/`) |
| command | `python3 scripts/ventral_directional_dose/run_directional_dose.py --execute --contract-sha256 f5399058… --freeze-commit ea109877…` |
| run | attempt 1 of 1; 2026-09-17T00:38:27Z → 00:55:31Z; exit 0; `EXECUTE=COMPLETE` (`logs/02_execute_attempt1.log`) |
| environment | Apple M2 Pro, macOS 14.2.1 arm64, python 3.11.15, torch 2.12.1, CPU, float32, deterministic algorithms |

## 1. EXECUTIVE RESULT

**[EMPIRICAL_RESULT]** Moving native ŝ toward its retrieved lexical-prototype direction while keeping the native norm produces a **non-monotonic, regression-heavy** dose response. The shape is the same in all four states and both readouts:
* **Overall accuracy is below native S0 at every α.** It is lowest at α = 0.50 (37.3–44.1%) and partially recovers at α = 1.00 (77.6–81.5%, still below S0's 87.6–89.2%).
* **Regressions dominate at intermediate α.** S0-correct items broken: 6,668–8,016 at α = 0.25, **13,902–15,427** at α = 0.50, 12,896–14,557 at α = 0.75, and 4,640–5,587 at α = 1.00.
* **Recovery of previous S1 rescues is non-monotonic.** It is 21.7–23.5% at α = 0.25, falls to 14.2–18.0% at 0.50, rises to 21.8–28.7% at 0.75, and reaches 71.7–74.4% at 1.00 (S1 recovers 100% by definition).
* **Direction × norm.** Prototype direction with native norm (α = 1) recovers 71.7–74.4% of S1 rescues but breaks 4,640–5,587 items. Native direction with prototype norm (S3) rescues 12–13% and breaks 2,085–2,556. Prototype direction with prototype norm (S1) rescues ~100% and breaks ≤ 46. **Neither factor alone reproduces S1.**
* **Stability across repair.** SOURCE and POST_REPAIR curves are almost identical (rescue-set Jaccard 0.87–0.96; recovery-fraction differences ≤ 0.54 percentage points).

**[INTERPRETATION]** The best-supported frozen family is **DR3 (non-monotonic / regression-heavy)**, with a **DR2-like** concentration of rescue at the α = 1 endpoint. DR1 and DR4 are not supported. **Steering recommendation: PRIORITIZE_T**, with the qualifications in §13.

## 2. SOURCE INTEGRITY

**[EMPIRICAL_RESULT]** Verified before, inside and after execution (DOSE-A, DOSE-H):
* Lineage: HEAD at the freeze commit, ancestry to amendment `1108ea6b…`, design `0b758696…` and closed results `0f25b5b8…`; clean tree.
* Hash manifests: implementation manifest hashes, design/freeze `SHA256SUMS` and `SHA256SUMS_CODE`.
* Closed controls: contract `a5ca7b83…`, item-level `37f2fb17…`, summary `455ccaf0…`, AR `75106959…`; the closed `SHA256SUMS` (37 and 17 entries) in both this worktree and the closed worktree, which is still at `0f25b5b8…` and clean.
* Model artifacts: SOURCE checkpoints `a5f21de9…aad76c` / `0657f410…dc79f3` and their archival copies; Arm-A heads `8865ba95…7afcfc` / `724ed4c6…b972d03`.
* State identities: composites `6d728285…`, `9185aa56…`, `32928707…`, `e151b306…`.
* Data: GloVe `91125602…`, lexicon `ae809181…`.

Per-state parameter hashes equal the closed preflight values before and after (DOSE-I).

## 3. REAL-STATE PREFLIGHT (§15b)

**[EMPIRICAL_RESULT]** Over all 4 × 29,571 items:

| state | ORDINARY | NEAR_COLLINEAR | ZERO_SHAT | ZERO_PROTOTYPE | NEAR_ANTIPODAL | min cos(u_s, u_p) | max cos(u_s, u_p) | min norm_r | min ‖ŝ‖ | min ‖v‖ |
|---|---|---|---|---|---|---|---|---|---|---|
| W3_SRC | 29,571 | 0 | 0 | 0 | 0 | 0.390 | 0.913 | 0.408 | 2.631 | 2.610 |
| W3_REP | 29,571 | 0 | 0 | 0 | 0 | 0.391 | 0.911 | 0.413 | 2.625 | 2.610 |
| W4_SRC | 29,571 | 0 | 0 | 0 | 0 | 0.380 | 0.907 | 0.421 | 3.182 | 2.610 |
| W4_REP | 29,571 | 0 | 0 | 0 | 0 | 0.380 | 0.907 | 0.420 | 3.176 | 2.610 |

* Median cos(u_s, u_p) is 0.747 / 0.747 / 0.738 / 0.738, so the median θ is ≈ 0.73–0.74 rad.
* **No hard stop fired, and no fallback was needed.** Every item used ordinary SLERP (NEAR_COLLINEAR_CASES is empty).
* DOSE-J retrieval identity: 0 mismatches.
* DOSE-B, α = 0 through the new supplier: **0 mismatches** in exact correctness **and** predicted phonology, both conventions, all four states.
* The standalone and in-process preflight geometry reports are byte-identical in content.

## 4. VALIDITY GATES (all PASS, all states)

| gate | W3_SRC | W3_REP | W4_SRC | W4_REP |
|---|---|---|---|---|
| NO_ZERO_SHAT / NO_ZERO_PROTOTYPE / NO_NEAR_ANTIPODAL | pass | pass | pass | pass |
| DOSE-A identity | pass | pass | pass | pass |
| DOSE-B α=0 ≡ S0 (exact + phonology, both conventions) | 0 mismatch | 0 | 0 | 0 |
| DOSE-C max relative norm error | 2.13e-8 | 1.73e-8 | 2.09e-8 | 1.61e-8 |
| DOSE-D α=1: max 1 − cos(s_1, v) | 6.7e-16 | 5.6e-16 | 6.7e-16 | 5.6e-16 |
| DOSE-E max \|angle − αθ\| (rad); monotonicity violations | 7.6e-9; 0 | 7.0e-9; 0 | 7.1e-9; 0 | 7.0e-9; 0 |
| DOSE-F shared path (sem_to_h0 input == supplied; encoder step dev; free-AR vs forced-length vectors identical) | 0 / 0.0 / 0 | 0 / 0.0 / 0 | 0 / 0.0 / 0 | 0 / 0.0 / 0 |
| DOSE-G guard hits | 0 | 0 | 0 | 0 |
| DOSE-H closed outputs unchanged (pre and post) | pass | pass | pass | pass |
| DOSE-I parameters unchanged (= closed values) | pass | pass | pass | pass |
| DOSE-J live ŝ vs preflight ŝ (max abs; top-1 changes) | 0.0; 0 | 0.0; 0 | 0.0; 0 | 0.0; 0 |

**[STRUCTURAL_CODE_FACT]** Every α decode ran through the frozen `ventral_interface.decode.decode_condition` (isolated route `ltm`), fed by a supplier on the frozen `ltm.to_semantic` hook. There was no gate, no FULL fusion and no dorsal readout.

## 5. PRIMARY FREE-AR DOSE RESPONSE (all 29,571 items)

**[EMPIRICAL_RESULT]** Exact count (proportion). α = 0 is the immutable S0; S1 and S3 are immutable controls.

| state | α=0 (S0) | α=0.25 | α=0.50 | α=0.75 | α=1.00 | S1 (proto dir + proto norm) | S3 (native dir + proto norm) |
|---|---|---|---|---|---|---|---|
| W3_SRC | 26,372 (89.18%) | 20,447 (69.15%) | 13,020 (44.03%) | 14,380 (48.63%) | 24,083 (81.44%) | 29,529 (99.86%) | 24,684 (83.47%) |
| W3_REP | 26,378 (89.20%) | 20,441 (69.13%) | 13,052 (44.14%) | 14,398 (48.69%) | 24,112 (81.54%) | 29,571 (100.00%) | 24,650 (83.36%) |
| W4_SRC | 25,924 (87.67%) | 18,718 (63.30%) | 11,034 (37.31%) | 12,161 (41.12%) | 22,949 (77.61%) | 29,520 (99.83%) | 23,867 (80.71%) |
| W4_REP | 25,909 (87.62%) | 18,727 (63.33%) | 11,014 (37.25%) | 12,205 (41.27%) | 22,991 (77.75%) | 29,571 (100.00%) | 23,842 (80.63%) |

S0 WRONG→CORRECT / S0 CORRECT→WRONG (the remaining transitions follow from the denominators):

| state | α=0.25 | α=0.50 | α=0.75 | α=1.00 |
|---|---|---|---|---|
| W3_SRC | 743 / 6,668 | 571 / 13,923 | 909 / 12,901 | 2,377 / 4,666 |
| W3_REP | 749 / 6,686 | 576 / 13,902 | 916 / 12,896 | 2,374 / 4,640 |
| W4_SRC | 810 / 8,016 | 537 / 15,427 | 794 / 14,557 | 2,612 / 5,587 |
| W4_REP | 794 / 7,976 | 519 / 15,414 | 797 / 14,501 | 2,625 / 5,543 |

**Item-level sequence descriptives** (free-AR, over α = 0 → 1; `table_item_sequence_nonmonotonicity_freear.tsv`):
* Items correct at all five α: 8,491 / 8,498 / 6,580 / 6,597.
* Items with at least one correct→wrong step: 18,555 / 18,551 / 20,111 / 20,075.
* S0-correct items that are wrong at α = 0.50 but correct again at α = 1.00: 10,855 / 10,862 / 11,453 / 11,465.
* Native failures rescued at α = 0.25 but wrong again at α = 0.50: 392 / 390 / 475 / 476.

## 6. FORCED-LENGTH DOSE RESPONSE (secondary)

**[EMPIRICAL_RESULT]** Every count in §5, §7, §8 and §9 is identical under canonical forced-length AR. Exact-correct disagrees between the two conventions on **0 items** at every α in every state (`table_convention_item_agreement.tsv`). The paired and α = 1 factorization blocks are identical across conventions.

## 7. PREVIOUS S1 RESCUE RECOVERY

**[EMPIRICAL_RESULT]** Recovered / previous S1 rescues (fraction), free-AR:

| state | α=0.25 | α=0.50 | α=0.75 | α=1.00 |
|---|---|---|---|---|
| W3_SRC | 743/3,195 (23.26%) | 571/3,195 (17.87%) | 909/3,195 (28.45%) | 2,377/3,195 (74.40%) |
| W3_REP | 749/3,193 (23.46%) | 576/3,193 (18.04%) | 916/3,193 (28.69%) | 2,374/3,193 (74.35%) |
| W4_SRC | 808/3,642 (22.19%) | 536/3,642 (14.72%) | 794/3,642 (21.80%) | 2,612/3,642 (71.72%) |
| W4_REP | 794/3,662 (21.68%) | 519/3,662 (14.17%) | 797/3,662 (21.76%) | 2,625/3,662 (71.68%) |

* Recovery is **not monotonic** in α: it dips at 0.50 in all four states.
* Recovery is **concentrated at α = 1**, and even there 25.6–28.3% of S1 rescues are not recovered with the native norm.
* Nearly all α rescues are previous S1 rescues. Exceptions: 2 items at α = 0.25 and 1 at α = 0.50 in W4_SRC, which are native failures outside the S1-rescue set.

## 8. REGRESSIONS

**[EMPIRICAL_RESULT]**
* S0 CORRECT→WRONG counts are in §5. As a share of S0-correct items at α = 0.50: 52.8% (W3_SRC), 52.7% (W3_REP), 59.5% (W4_SRC), 59.5% (W4_REP).
* Regressions are unchanged in the failure strata (`C_CORRECT_AND_NATIVE_LTM_WRONG` and `PREV_S1_RESCUES` contain no S0-correct items).
* **Failure profile of regressions** (free-AR frozen EOS fields; `table_regression_eos_profile_freear.tsv`). At α = 0.50, W3_SRC:
  * of 13,923: 5,033 EOS before target length, 4,624 EOS after, 2 cap terminations, 4,264 with same-length predictions;
  * first divergence at step 0 in 3,487.
  * The other states show the same pattern.

## 9. C_CORRECT_AND_NATIVE_LTM_WRONG

**[EMPIRICAL_RESULT]** Exact within stratum (free-AR). S0 = 0 and S1 = 100% in every state.

| state | n | α=0.25 | α=0.50 | α=0.75 | α=1.00 | S3 |
|---|---|---|---|---|---|---|
| W3_SRC | 3,160 | 733 (23.20%) | 560 (17.72%) | 891 (28.20%) | 2,345 (74.21%) | 390 (12.34%) |
| W3_REP | 3,159 | 739 (23.39%) | 565 (17.89%) | 898 (28.43%) | 2,343 (74.17%) | 375 (11.87%) |
| W4_SRC | 3,609 | 802 (22.22%) | 531 (14.71%) | 786 (21.78%) | 2,586 (71.65%) | 479 (13.27%) |
| W4_REP | 3,628 | 789 (21.75%) | 514 (14.17%) | 790 (21.78%) | 2,599 (71.64%) | 482 (13.29%) |

Same non-monotonic shape. These items have correct lexical retrieval from ŝ, yet only the full prototype vector (S1) decodes them all.

## 10. ALPHA = 1 VS S1 VS S3 (direction × norm)

**[EMPIRICAL_RESULT]** Free-AR, identical under forced-length.

| state | S3 rescues / regressions | α=1 rescues / regressions | S1 rescues / regressions | α=1 − S1 exact | α=1 vs S1 disagreements (α=1 right & S1 wrong) | S1 rescues also α=1 rescues | α=1 − S3 exact |
|---|---|---|---|---|---|---|---|
| W3_SRC | 397 / 2,085 | 2,377 / 4,666 | 3,195 / 38 | −5,446 | 5,446 (0) | 74.40% | −601 |
| W3_REP | 382 / 2,110 | 2,374 / 4,640 | 3,193 / 0 | −5,459 | 5,459 (0) | 74.35% | −538 |
| W4_SRC | 488 / 2,545 | 2,612 / 5,587 | 3,642 / 46 | −6,571 | 6,571 (0) | 71.72% | −918 |
| W4_REP | 489 / 2,556 | 2,625 / 5,543 | 3,662 / 0 | −6,580 | 6,580 (0) | 71.68% | −851 |

* **α = 1 rescues are a strict subset of S1 rescues** in every state (0 items correct at α = 1 but wrong under S1).
* **Prototype direction alone is not sufficient while the native norm is retained.** It recovers ~72–74% of S1 rescues and introduces 4,640–5,587 regressions, which is net below S0 and below S3.
* **Prototype norm alone (S3) is not sufficient** either.
* **Only the joint prototype vector (S1) is near-universally compatible** (≤ 46 regressions, all SOURCE retrieval errors in the closed diagnostic).

## 11. SOURCE VS POST_REPAIR

**[EMPIRICAL_RESULT]** Paired, free-AR (`fig5_paired_src_rep.tsv`):

| witness | α | exact SRC / REP | rescue fraction SRC / REP | regressions SRC / REP | rescue-set Jaccard | Δ prev-S1 recovery (REP − SRC) |
|---|---|---|---|---|---|---|
| W3 | 0.25 | 20,447 / 20,441 | 23.23% / 23.46% | 6,668 / 6,686 | 0.908 | +0.20 pp |
| W3 | 0.50 | 13,020 / 13,052 | 17.85% / 18.04% | 13,923 / 13,902 | 0.912 | +0.17 pp |
| W3 | 0.75 | 14,380 / 14,398 | 28.42% / 28.69% | 12,901 / 12,896 | 0.927 | +0.24 pp |
| W3 | 1.00 | 24,083 / 24,112 | 74.30% / 74.35% | 4,666 / 4,640 | 0.960 | −0.05 pp |
| W4 | 0.25 | 18,718 / 18,727 | 22.21% / 21.68% | 8,016 / 7,976 | 0.867 | −0.50 pp |
| W4 | 0.50 | 11,034 / 11,014 | 14.72% / 14.17% | 15,427 / 15,414 | 0.896 | −0.54 pp |
| W4 | 0.75 | 12,161 / 12,205 | 21.77% / 21.76% | 14,557 / 14,501 | 0.910 | −0.04 pp |
| W4 | 1.00 | 22,949 / 22,991 | 71.62% / 71.68% | 5,587 / 5,543 | 0.937 | −0.04 pp |

**[INTERPRETATION]**
* The dose response, including the α = 0.50 trough and the α = 1 partial recovery, is **stable across the Arm-A repair**. This is consistent with Arm-A changing only `to_semantic.2`, a structural fact.
* The W3/W4 difference in depth (W4 lower at intermediate α) is a between-seed difference, not a repair effect.

## 12. DR FAMILY

**[INTERPRETATION]** Using only the raw curves above; no thresholds are introduced.
* **DR1 — EARLY RESCUE: not supported.** By α ≤ 0.50 rescue is ≤ 23.5% of native failures, while regressions (6,668–15,427) exceed rescues by roughly 9–10× at α = 0.25 (e.g. 6,668 vs 743) and 24–30× at 0.50 (e.g. 13,923 vs 571).
* **DR4 — SMOOTH MONOTONIC: not supported.** Accuracy, rescue and recovery all dip at α = 0.50, and regressions peak there, in every state.
* **DR3 — NON-MONOTONIC / REGRESSION-HEAVY: supported.** Intermediate directional movement breaks roughly half or more of the natively correct items (52.7–59.5% at α = 0.50) and makes the response non-monotonic.
* **DR2 — LATE / NEAR-PROTOTYPE RESCUE: descriptively present as a feature.** The largest rescue is at α = 1.00 (71.7–74.4% of S1 rescues vs ≤ 28.7% at α ≤ 0.75). Even at α = 1, prototype direction with the native norm leaves ~26–28% of S1 rescues unrecovered and adds 4,640–5,587 regressions. Only the full prototype vector (S1) is near-universally compatible.
* **Consistency:** all four states and both conventions show the same pattern, so no disagreement prevents classification.
* **Classification: DR3, with a DR2-like endpoint concentration.**

## 13. PROGRAMME IMPLICATION: T VS R

**Recommendation: PRIORITIZE_T** (training-only compatibility). This is a steering recommendation; nothing is implemented.

**[INTERPRETATION] Rationale**
1. **Against minimal continuous refinement (R).** Small or intermediate continuous movement of ŝ toward the prototype direction, the natural shape of a minimal differentiable refinement, is harmful in these frozen states. It trades a few hundred rescues for thousands to over fifteen thousand regressions, and the response is non-monotonic. DR3's frozen implication is "do not prioritize prototype-directed refinement; training-only preferred".
2. **Near-prototype endpoint.** Beneficial recovery appears only at the endpoint, and full compatibility requires the complete lexical prototype vector (direction **and** norm). DR2's frozen implication is "solution approaches prototype replacement; training-only compatibility preferred; do not build an attractor".
3. **Narrow decoder compatibility.** The decoder is compatible with native ŝ for ~88–89% of items and with raw prototype vectors for ~100%. It is **not** compatible along the great-circle path between them. This localizes the problem to how the decoder generalizes over semantic input space, which a training-level intervention targets directly.
4. **Stability.** The conclusion holds identically in SOURCE and POST_REPAIR and in both witnesses.

**Qualifications**
* **Joint path untested.** Only norm-preserving directional movement was tested. A path that moves direction **and** norm jointly toward the prototype, or snaps discretely to it, was not preregistered. **[NOT_ESTABLISHED]** whether such a path is smooth.
* **S1 is not a minimal refinement.** S1 (discrete substitution of the retrieved prototype) remains ~100% compatible, but it is a retrieval-substitution scheme, not a minimal refinement. Its architectural status belongs to a separate CENTRAL decision.
* **Training-audit boundary (binding).** A T intervention must identify a materially new mechanism relative to V6. V6 already trains the decoder from ŝ (`L_dec`), aligns ŝ with cosine + MSE (`L_align`), trains the decoder from raw GloVe (Naming), and sends FULL/gate gradient into ŝ. "Train decoder on ŝ", "add cosine/MSE alignment" and "train from raw GloVe" are not new. No mechanism is designed here.

## 14. WHAT IS ESTABLISHED

1. **[EMPIRICAL_RESULT]** In the four frozen states, norm-preserving SLERP of ŝ toward the retrieved prototype direction gives exact isolated-ventral repetition of 63–69% (α = 0.25), 37–44% (0.50), 41–49% (0.75) and 78–82% (1.00), against S0 at 88–89%. The response is identical under free-AR and forced-length.
2. **[EMPIRICAL_RESULT]** Intermediate α produces 6.7k–15.4k regressions per state, peaking at α = 0.50.
3. **[EMPIRICAL_RESULT]** α = 1 recovers 71.7–74.4% of S1 rescues, all within the S1 rescue set, with 4.6k–5.6k regressions. S3 recovers ~12–13% with 2.1k–2.6k regressions. S1 recovers 100% with ≤ 46.
4. **[EMPIRICAL_RESULT]** The pattern is the same in SOURCE and POST_REPAIR (Jaccard 0.87–0.96).
5. **[INTERPRETATION]** Decoder compatibility is not a graded axis along the norm-preserving direction path from ŝ to the prototype. It is high at both ends of the original data distribution (native ŝ, full prototype vectors) and low in between.

## 15. WHAT IS NOT ESTABLISHED

* Whether a joint direction-and-norm path, or a discrete snap, toward the prototype is smooth (untested).
* Why the decoder is incompatible with intermediate vectors: the training cause, or the geometry of `sem_to_h0`/GRU initial states. No causal test was run; "off-manifold" is not concluded.
* That any specific training-only mechanism would succeed, or what it would be. None was designed or tested.
* That a semantic attractor or the Yair `semantic_attractor` flag is necessary or unnecessary as an architecture. Neither was implemented or tested.
* Generality beyond these two seeds and four mature states.
* Any FULL-fusion or behavioural implication. Isolated ventral route only.

## 16. STOP

The frozen diagnostic was executed once, validated and packaged. **STOP.** There was no training, refinement, attractor, Yair flag, gate change, lesioning or full-ceiling run. The next workstream awaits CENTRAL STEERING.
