# CENTRAL_STEERING_HANDOFF — LICHTHEIM3 VENTRAL SEMANTIC INTERFACE · FROZEN FACTORIZATION DIAGNOSTIC

**DIAGNOSTIC STATUS:** SCIENTIFIC_EXECUTION=COMPLETE · all frozen validity gates PASS · executed exactly once.

| item | value |
|---|---|
| worktree / branch | `/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/wt-ventral-interface` · `paper-programme/ventral-interface-diagnostic` |
| freeze commit (executed) | `4ad20048e20da84b9f22a92965098b84c2bf7dd6` (descends from `79f4e5bd94a9c1f82594f4050028b39c220b82b0`) |
| results commit | recorded in the commit message and in the final report; it contains results only, and the frozen implementation is byte-identical to the freeze commit |
| contract SHA256 | `a5ca7b83717b43ab77cedae0cd3e4bef17901ffff051f5ca90b9340b45bda5d3` |
| full package | `scientific_execution/` (recap, item-level TSV, summary JSON, AR diagnostic, DIAGNOSTIC_ONLY prefix-correction TSV, figures, figure_source_data, logs, validation_report, SHA256SUMS) |

## Validity gates (per state W3_SRC / W3_REP / W4_SRC / W4_REP)
| gate | result |
|---|---|
| BANK | pass ×4 |
| R | C errors 34 / 0 / 42 / 0, equal to archived; 0 top-1 mismatches |
| A | 0 violations |
| B | S1 ≡ S2 on 27,947 / 27,981 / 27,939 / 27,981 equal rows; max diff 0.0 |
| D | min cos(S3, ŝ) ≥ 0.9999999999999993; 0 rank changes; 0 degenerate |
| C | matched Naming (cap 256): 0 errors == archived; 0/29,571 item mismatches |
| SHAT | ≤ 6.42e-6; 0 top-1 changes |
| H | native S0 LTM errors 3199 / 3193 / 3647 / 3662, equal to archived, both conventions |
| E | shared downstream path; 0 guard hits; params unchanged; encoder bitwise across steps |

## Four-state identity
| state | witness | base SHA256 | Arm-A head SHA256 | reconstructed_state_identity |
|---|---|---|---|---|
| W3_SRC | V6 s19 u3825 | `a5f21de9edb4b2090cb144c666ca4bd8e1f0572333000facb13f6c51cfaad76c` | NA | `6d7282854323b810727cd9c0c90cace6dd8a54d306408cafaa8c40c5455d10a0` |
| W3_REP | V6 s19 u3825 | same | `8865ba9539e4a445ffc8847a71be698779a98478aa5e6a1bce2393f3de7afcfc` | `9185aa5671e2ae3ab1e0860d52424b472f616d2515057d7ccbc6b383cf43dff1` |
| W4_SRC | V6 s20 u3040 | `0657f41030e54eac11816116af6c76b66d597c393fc72579a3755401c9dc79f3` | NA | `32928707acc81af215faa829ad94bba566804982bc192bb5116ea32d43d56e29` |
| W4_REP | V6 s20 u3040 | same | `724ed4c6a84fba07b6d9f11d535b8a0eb726daff21b9def9a5e60d9c7b972d03` | `e151b306f706e9db9828b3c9e742984a0ae836dbd97e73f86e4e2063bd7c5cd6` |

## Primary: genuine free-AR isolated ventral repetition (exact / 29,571)
| state | S0 native | S1 raw retrieved | S2 raw true | S3 radial |
|---|---|---|---|---|
| W3_SRC | 26,372 (89.18%) | 29,529 (99.86%) | 29,571 (100%) | 24,684 (83.47%) |
| W3_REP | 26,378 (89.20%) | 29,571 (100%) | 29,571 (100%) | 24,650 (83.36%) |
| W4_SRC | 25,924 (87.67%) | 29,520 (99.83%) | 29,571 (100%) | 23,867 (80.71%) |
| W4_REP | 25,909 (87.62%) | 29,571 (100%) | 29,571 (100%) | 23,842 (80.63%) |

## Forced-length (secondary)
All counts are identical to free-AR. Exact-correct disagrees between the conventions on 0 items per state and condition. The conventions differ only in the predicted strings of wrong items.

## Principal transitions vs S0 (free-AR, all items: W→C / C→W)
| state | S1 | S2 | S3 |
|---|---|---|---|
| W3_SRC | 3,195 / 38 | 3,199 / 0 | 397 / 2,085 |
| W3_REP | 3,193 / 0 | 3,193 / 0 | 382 / 2,110 |
| W4_SRC | 3,642 / 46 | 3,647 / 0 | 488 / 2,545 |
| W4_REP | 3,662 / 0 | 3,662 / 0 | 489 / 2,556 |

## C × LTM decomposition (free-AR; n · S0/S1/S2/S3 exact)
| stratum | W3_SRC | W3_REP | W4_SRC | W4_REP |
|---|---|---|---|---|
| C_CORRECT_AND_NATIVE_LTM_WRONG | 3,160 · 0/3,160/3,160/390 | 3,159 · 0/3,159/3,159/375 | 3,609 · 0/3,609/3,609/479 | 3,628 · 0/3,628/3,628/482 |
| C_WRONG_AND_NATIVE_LTM_WRONG | 4 · 0/0/4/0 | 0 | 5 · 0/0/5/2 | 0 |
| C_WRONG_BUT_NATIVE_LTM_PHONOLOGY_CORRECT | 30 · 30/0/30/28 | 0 | 37 · 37/0/37/25 | 0 |
| C_WRONG_BUT_RETRIEVED_PHONOLOGY_CORRECT (separate diagnostic) | 0 | 0 | 0 | 0 |
| non-canonical homophone rows (1,590) | S0 1,555 · S1 1,582 · S3 1,487 | 1,556 · 1,590 · 1,486 | 1,557 · 1,581 · 1,483 | 1,556 · 1,590 · 1,476 |

## S1 vs S3: radial vs directional
S1 rescues 3,195 / 3,193 / 3,642 / 3,662 native failures. S3 reproduces 397 / 382 / 486 / 489 of them (**12.43% / 11.96% / 13.34% / 13.35%**). S1-only rescues number 2,798 / 2,811 / 3,156 / 3,173. S3-only rescues: 0 / 0 / 2 / 0. S3 also breaks 2,085–2,556 natively correct items per state. **The majority of S1 rescues were not reproduced by S3.** Descriptively, native failures do not differ from successes in ‖ŝ‖ (median ≈ 6.9 / 7.5 in both groups), and median cos(target, ŝ) ≈ 0.72–0.75 even for successes.

## S1 vs S2: retrieval-limited
* S2 correct but S1 wrong on native failures: 4 (W3_SRC) and 5 (W4_SRC); 0 in REP states. All have wrong lexical identity and a different retrieved phonology.
* S1 regressions (38 / 46) occur only in SOURCE, all with wrong retrieval identity.
* S1 is never wrong when the lexical identity is correct.
* F3 is minor (≤ 0.14% of native failures), and Arm-A removes it.

## AR diagnostic headline (native S0 free-AR failures)
* **Where the first error falls:** 35–37% of first divergences are at step 0. About 2% are length-only (at the EOS position).
* **Cascades:** 91–93% of failures cascade into two or more positional mismatches.
* **Margins:** median chosen−gold margin is 3.2 logits; about 20% are below 1 logit and about 36% are 5 logits or more. The gold token ranks 2nd in about 69% of failures. None are numerically ambiguous.
* **DIAGNOSTIC_ONLY_PREFIX_CORRECTION:** correcting the first divergent token reaches the exact target in 61.2–61.5% of failures (about 40% after a step-0 error, about 74% after a later one, 100% for EOS-position errors). This is not performance and not a mechanism.
* The profile is identical in SOURCE and POST_REPAIR.

## SOURCE vs POST_REPAIR headline
* **What Arm-A fixes:** C errors 34→0 and 42→0 (none broken); S1 errors 42→0 and 51→0.
* **What it leaves unchanged:** S0 native errors go 3,199→3,193 and 3,647→3,662 (failure-set Jaccard 0.963 / 0.944). S3 structure, geometry medians (Δ ≤ 0.0003 cosine) and AR diagnostics are unchanged.
* **Reading:** Arm-A repairs retrieval identity, not decoder compatibility with ŝ. Structurally it cannot change the decoder.

## Best-supported frozen result family
* **F2, DIRECTIONAL / PROTOTYPE RESCUE: primary.** It holds in both witnesses, in SOURCE and POST_REPAIR, and under both readouts. Allowed interpretation: movement toward the retrieved lexical prototype substantially increases decoder compatibility. Not concluded: that a semantic attractor is necessary.
* **F3, RETRIEVAL-LIMITED: minor, SOURCE only**, removed by Arm-A.
* **Not supported:** F1 (counts contradict it) and F5 as dominant (the semantic intervention rescues nearly all failures under the same decoder).
* **Not encountered:** F4.
* **Not required:** F6.

## At most two candidate next interventions (not implemented)
1. **Semantic-refinement pilot:** move ŝ toward its retrieved lexical prototype before `sem_to_h0`, on the ventral path, using the frozen GloVe bank. Tied to F2 and to C being ≥ 99.85% (100% post-repair). It is an architectural change and needs publication-path arbitration.
2. **Training-only pilot:** make the unchanged ventral decoder compatible with encoder-produced ŝ, targeting the directional ŝ–prototype discrepancy. The readout would be the S0-vs-S1/S2 gap on this frozen protocol. One pilot, no sweep.

## NOT_ESTABLISHED
* That an attractor or refinement is *necessary*, or how much movement toward the prototype suffices (no interpolation was preregistered).
* "Off-manifold" (no operational test).
* The cause of the decoder–ŝ incompatibility.
* Generality beyond these two seeds and states.
* Any FULL-fusion implication.
* Any value of prefix correction or scheduled sampling as a mechanism.
* Any causal role of lexical rarity.

## Confirmation
TRAINING_RUN=NO
ARCHITECTURE_CHANGED=NO
GATE_CHANGED=NO
LESIONING_RUN=NO
FULL_FUSION_SCIENTIFIC_CONDITION=NO · DORSAL_RESCUE=NO · ATTRACTOR_IMPLEMENTED=NO · FROZEN_CONTRACT_OR_DRIVER_CHANGED=NO

---

## READY_TO_PASTE_CENTRAL_PROMPT

```
CENTRAL STEERING — ARBITRATION REQUEST
LICHTHEIM3 — VENTRAL SEMANTIC INTERFACE — FROZEN FACTORIZATION DIAGNOSTIC (EXECUTED)
Programme: POST_STAGE / PAPER_PROGRAMME

Status: SCIENTIFIC_EXECUTION=COMPLETE. Executed exactly once from freeze commit
4ad20048e20da84b9f22a92965098b84c2bf7dd6 (branch paper-programme/ventral-interface-diagnostic,
ancestry 79f4e5bd94a9c1f82594f4050028b39c220b82b0), contract SHA256
a5ca7b83717b43ab77cedae0cd3e4bef17901ffff051f5ca90b9340b45bda5d3. No training, no architecture change,
no gate change, no lesion, no FULL fusion condition, no dorsal rescue, no attractor.

States (mature H512): W3_SRC/W3_REP = V6 seed19 u3825 SOURCE / + Arm-A head; W4_SRC/W4_REP = V6 seed20
u3040 SOURCE / + Arm-A head. Arm-A changes only ltm.to_semantic.2.{weight,bias}.

All frozen validity gates PASS on all four states: BANK, R (C errors 34/0/42/0 = archived), A, B (S1≡S2 on
equal rows, max diff 0.0), D (S3 direction preserved, 0 degenerate), C (matched Naming reproduced item by
item: 0/29,571 mismatches), SHAT, H (native S0 reproduces archived LTM errors 3199/3193/3647/3662), E.

Conditions (only the vector entering the ventral decoder changes; identical decoder/readout):
S0 native ŝ; S1 raw GloVe of the frozen C top-1 row; S2 raw true GloVe; S3 = ||retrieved raw GloVe|| · ŝ/||ŝ||.

PRIMARY genuine free-AR isolated ventral repetition, exact / 29,571 (W3_SRC, W3_REP, W4_SRC, W4_REP):
S0 89.18%, 89.20%, 87.67%, 87.62%  (errors 3199, 3193, 3647, 3662)
S1 99.86%, 100%,   99.83%, 100%
S2 100%,   100%,   100%,   100%
S3 83.47%, 83.36%, 80.71%, 80.63%
Forced-length secondary readout: identical counts; 0 item-level exact disagreements.

Transitions vs S0 (WRONG→CORRECT / CORRECT→WRONG): S1 3195/38, 3193/0, 3642/46, 3662/0;
S2 3199/0, 3193/0, 3647/0, 3662/0; S3 397/2085, 382/2110, 488/2545, 489/2556.
S3 reproduces only 12.43%, 11.96%, 13.34%, 13.35% of S1 rescues; S1-only rescues 2798/2811/3156/3173;
S3 is net harmful in every state. Native failures do not differ from successes in ||ŝ||; median
cos(target, ŝ) ≈ 0.72–0.75 even for successes.

C × LTM: C_CORRECT_AND_NATIVE_LTM_WRONG = 3160/3159/3609/3628 items; S1 and S2 rescue all of them,
S3 rescues 390/375/479/482. Retrieval-limited (S2 right, S1 wrong) native failures: 4 and 5 (SOURCE only),
all wrong lexical identity with a different phonology; S1 regressions 38 and 46 (SOURCE only), all retrieval
identity errors; none in POST_REPAIR. C_WRONG_BUT_NATIVE_LTM_PHONOLOGY_CORRECT = 30 and 37 (SOURCE),
0 (REP). C_WRONG_BUT_RETRIEVED_PHONOLOGY_CORRECT = 0 everywhere. 1,590 non-canonical homophone rows
handled separately.

AR diagnostic (native free-AR failures): first divergence at step 0 in 35–37%; ~2% are EOS-position
(length-only) errors; 91–93% cascade to ≥2 positional mismatches; median chosen−gold margin 3.2 logits
(~20% <1 logit, ~36% ≥5); gold ranked 2nd in ~69%. DIAGNOSTIC_ONLY_PREFIX_CORRECTION (not performance,
not a mechanism): correcting the first divergent token yields the exact target in 61.2–61.5%.

SOURCE vs POST_REPAIR: Arm-A fixes C (34→0, 42→0) and S1 retrieval errors (42→0, 51→0) but leaves native
ventral errors (3199→3193, 3647→3662; failure-set Jaccard 0.963/0.944), S3 structure, geometry and AR
diagnostics essentially unchanged.

Executing agent's frozen-family classification: F2 DIRECTIONAL/PROTOTYPE RESCUE (primary); F3
RETRIEVAL-LIMITED minor and SOURCE-only; F1 not supported; F5 not supported as dominant; F4 not
encountered; F6 not required. Not established: necessity of an attractor/refinement, "off-manifold",
the cause of the decoder–ŝ incompatibility, the amount of movement toward the prototype that suffices,
generality beyond two seeds, FULL-fusion implications.

Candidate next interventions proposed (at most two, not implemented):
(1) semantic-refinement pilot: move ŝ toward its retrieved lexical prototype before sem_to_h0 on the
    ventral path (architectural; publication-path arbitration required);
(2) training-only pilot: make the unchanged ventral decoder compatible with encoder-produced ŝ,
    targeting the directional ŝ–prototype discrepancy; readout = S0 vs S1/S2 gap on this frozen protocol.

CENTRAL is asked to:
1. Arbitrate the frozen diagnostic (validity, execution, packaging).
2. Accept or reject the mechanistic localization: residual native ventral repetition error lies at the
   ŝ→ventral-decoder interface and is directional/prototype-related rather than scale-related, with
   retrieval identity limiting only a small SOURCE-only subset.
3. Select zero, one, or at most two next candidate interventions.
4. Decide whether a training-only pilot, a semantic-refinement pilot, or no architectural change is justified.
5. Preserve the publication-path constraint in that decision.
6. Prohibit any execution (training, refinement, attractor, gate, lesion, full-ceiling runs) until CENTRAL
   explicitly selects and freezes the next scientific workstream.
```
