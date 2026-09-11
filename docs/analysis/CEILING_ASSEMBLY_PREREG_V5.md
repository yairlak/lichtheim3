# Amendment V5 — ceiling-assembly test (preregistration)

**Date: 2026-09-12.** Committed **before any repair is run**. Branch
`probe/frozen-semantic-head`, parent `3912d758e4864a2e8a297f38311c6162772562d8`.

## 1. Question

The V4 Arm-A terminal correction (active-constraint raw-margin gamma hinge +
frozen J_lin_h0 preconditioner) reached strict C = 0 on 4/4 mature seeds while
preserving the ventral route essentially exactly. Seed 20's repaired u3600 head
already shows R canonical = R free-AR = N = C = 0.

Do the EXISTING official SETTLE **control** trajectories contain, per seed, a
source state with `Rcan = Rfree = N = 0` such that the already-fixed Arm-A
repair yields `Rcan = Rfree = N = C = 0` simultaneously?

## 2. Scope

Only existing official SETTLE **control** checkpoints, seeds 19/20/21/22. No
new joint training, no change to the SETTLE trajectory, no CHIGH parents, no
treatment C-LR arms, no AutoResearch contact, no scheduler. Arm A is used
exactly as preregistered in V4 (`054633cd`) and implemented in `3912d758`:
trainable = `ltm.to_semantic.2.{weight,bias}` only; gamma_raw = 1e-3; frozen
J_lin_h0 preconditioner; identical line search (R0 0.1, beta 0.5, c 1e-4, 20
halvings), stopping (all margins >= gamma - 1e-7, MAX_ITERS 150, 7200 s),
official C evaluator, fp64 fitting with fp32 deployed re-evaluation.
**No I / J_z / B / CE arms** — those causal comparisons are complete.

## 3. Eligibility and the deterministic selection rule (frozen before repair)

`CEILING_SOURCE_ELIGIBLE` iff, in the authoritative official SETTLE report
metrics (`settle_u3600_20260910/report/settle_by_milestone.tsv`),
`rep_canonical_errors = 0` AND `rep_freear_errors = 0` AND `naming_errors = 0`.

Per seed, among eligible checkpoints: (1) minimum strict C error count;
(2) tie-break by LATEST u; (3) then lowest global step. If no checkpoint
qualifies, report `NO_EXISTING_CEILING_SOURCE` for that seed. The criterion is
not weakened.

## 4. Inventory outcome (from pre-existing metrics; recorded before any repair)

| seed | eligible / 24 | C among eligible | selected |
|---|---|---|---|
| 19 | **0** | — | `NO_EXISTING_CEILING_SOURCE` |
| 20 | 11 | 32,37,37,38,39,40,40,40,41,42,43 | **u3600, step 10,000,800, C=32** |
| 21 | 1 | 26 | **u3400, step 9,445,200, C=26** |
| 22 | **0** | — | `NO_EXISTING_CEILING_SOURCE` |

Seed 20's selection coincides with the already-verified V4 Arm-A witness, so it
serves as the **regression control**: its scientific metrics must reproduce.

## 5. Why this is not checkpoint HPO

The checkpoints were produced by the official SETTLE run long before this
terminal correction was designed; the eligibility criterion and the selection
rule are fixed above, before any repair outcome is observed; and the claim is
not that the selected milestone is a training recipe. The test asks only
whether the already-observed joint trajectory contains a source state from
which the validated deterministic repair reaches the exact three-task ceiling.

## 6. Source-specific metric

For each selected checkpoint the frozen compatibility metric (D_i, H_lin_h0)
and the cached representations are rebuilt **from that checkpoint**. The u3600
H/D are not reused for an earlier milestone. This is the existing definition of
Arm A applied at another source point, not a new lever.

## 7. Endpoints

PRIMARY, in an isolated deployed model copy at the first official C = 0 head:
repetition canonical (full lexicon), genuine free-AR repetition, naming, and
strict canonical top-1 C against the full 29,571 bank.

`EXACT_100_100_100_WITNESS` iff Rcan = 0 AND Rfree = 0 AND N = 0 AND C = 0.
No top-k substitution, no averaging, no relaxation.

SECONDARY: LTM-only exact and delta vs source, gate, J_z, J_lin_h0, J_true,
||Delta W||, iterations, minimum raw margin, source C residual count.

## 8. Interpretation (frozen)

Report exactly N/4. Absence of an eligible historical checkpoint is **not** a
failure of the repair mechanism; it means the archived SETTLE trajectory never
exposed R = N = ceiling at the retained milestones for that seed.

Language, even if all succeed: "from-scratch joint-trained model + deterministic
terminal compatibility repair" — never "ordinary joint training reached
100/100/100". A from-scratch source and a terminal repair are established;
a pure joint-training ceiling is NOT demonstrated.
