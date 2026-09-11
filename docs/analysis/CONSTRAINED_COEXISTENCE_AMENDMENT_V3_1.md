# Amendment V3.1 — numerical feasibility tolerance for solver termination

**Date: 2026-09-11.** Parent `348d4eda495f7b67df0d405f570918c610ca09d4`.
Preregistration amended: `CONSTRAINED_COEXISTENCE_AMENDMENT_V3.md`
(`fb634bec7f1490be588a17ab3a12f0531b7ea749`).

This is a **numerical amendment only**. It changes no scientific quantity.

## 1. The defect

The V3 termination predicate required `worst_margin >= GAMMA_RAW` with no
numerical tolerance. On the converged seed-19 witnesses the achieved minimum
raw margin was

    worst_margin = 0.00099998011995872105
    GAMMA_RAW    = 0.001
    deficit      = 1.988e-08

so 19 active constraints, sitting a few ULPs below `gamma`, were re-flagged as
violations on every round. At that point the run was demonstrably converged:

* the active set stopped growing after round 3 (110 / 114 constraints);
* the objective was constant from round 2 (0.008722);
* KKT residuals were exactly 0.0 (primal slack, dual `lambda >= 0`,
  complementarity, stationarity);
* **no item had margin <= 0**;
* the **official strict evaluator returned `C_errors = 0 / 27,981`**;
* the source checkpoint was byte-identical before and after.

The loop nevertheless ran to `MAX_ROUNDS = 30` and labelled the outcome
`ALGORITHM_FAILED_TO_FIND_WITNESS`. That label was wrong; the solutions were
valid. This is a termination defect, not a scientific failure.

## 2. The amendment

    FEAS_TOL_RAW = 1e-7

**A SOLVER TERMINATION TOLERANCE ONLY.** The scientific constraint is
unchanged and remains exactly

    d^T s >= GAMMA_RAW,    GAMMA_RAW = 1e-3.

Termination and the violation set both use the single coherent predicate

    numerically_feasible(i)  <=>  worst_margin_i >= GAMMA_RAW - FEAS_TOL_RAW

No other tolerance is introduced anywhere in the solver. The **solved QP is
unchanged**: the right-hand sides stay `r_k = GAMMA_RAW - d_k^T s^0_{i_k}`
(the solver still targets the full `gamma`), the Gram matrix, the objective,
the dual and the active-set construction are all untouched.

## 3. Why 1e-7

| quantity | value |
|---|---|
| observed numerical deficit | ~1.99e-08 |
| chosen tolerance `FEAS_TOL_RAW` | 1e-7 |
| ratio to observed deficit | ~5x larger |
| ratio to `GAMMA_RAW` | 1e-7 / 1e-3 = **1e-4 = 0.01%** |

The tolerance is five times the observed roundoff-scale deficit while
remaining four orders of magnitude below the nominal solver margin. It cannot
admit a genuine strict error: an actual C error has margin `<= 0`, which is
`1e-3` away from the acceptance threshold, i.e. **four orders of magnitude**
outside it.

## 4. What is NOT changed

`GAMMA_RAW`, `TOP_V`, `K_MAX`, `MAX_ROUNDS`, the three objective definitions,
the seed set, the arm set, the CG tolerance and iteration cap, the
active-set policy, the official evaluators, and the route-health battery are
all unchanged.

## 5. Reporting discipline

The **exact achieved minimum raw margin is always reported**, in full
precision, in both raw and cosine units. No reported margin is ever clipped,
rounded up, or replaced by `gamma`.

A witness is **never** accepted scientifically on the strength of this
tolerance. Every witness still requires

    official strict C_errors == 0

from the unchanged official evaluator, run on a head reloaded from disk into
an isolated model copy, with the source checkpoint hashed before and after.

## 6. Rerun policy

The complete preregistered 12-run grid (seeds 19/20/21/22 x arms
I / J_z / J_lin_h0) is rerun **from scratch**, including seed 19 `I` and
`J_z`, which serve as deterministic regression controls for this amendment.
The pre-fix 2/12 results remain archived unaltered as historical evidence and
are **not** reused as substitutes. Expected outcome for the regression
controls: the same scientific solution, reached with earlier termination.
