# Amendment V3 — off-line constrained feasibility search for a route-compatible C=0 head

**Date: 2026-09-11.** Written and committed **BEFORE any constrained-search
outcome on real data.** Branch `probe/frozen-semantic-head`, parent
`cab75066af0f183939a0b93d5c30ff65f603c79c`.

## 1. Why V2 is superseded

V2 (commit `14e61c1`) tested the **straight line** `W(a)=(1-a)W0+aW*` and
completed 32/32 points. Result: `NO_COMPATIBLE_WITNESS_ON_PATH`. At `alpha_C0`
LTM is 0.0953-0.2216 against a 0.877-0.886 baseline. That refutes compatibility
**on that one line only**, not in the 153,900-parameter final-layer space.

## 2. New question

Does an **off-line** final `Linear(512->300)` exist that, on the SAME frozen
`phi` and SAME frozen GloVe bank, reaches strict canonical C = 0 AND preserves
compatibility with the frozen LTM decoder?

H512 capacity is NOT reopened; the head is NOT enlarged; R/N are NOT tuned;
AutoResearch Stage A stays superseded.

## 3. Derivation (independently verified)

`s_i = W phi_i + b`. For any non-zero `s_i`, bank rows being L2-normalised,
`argmax_j cos(s_i,t_j) = argmax_j s_i^T t_j`. Strict correctness at item `i`
with target `T` is therefore

    s_i^T (t_T - t_j) > 0   for all j != T,

which is **linear in (W,b)**: `s_i^T d = <W, d phi_i^T> + <b, d>`.

Let `z_i = A s_i + c` (`A,c` = frozen `sem_to_h0`), `h0_i = tanh(z_i)`, and
`delta_s_i = s_i - s_i^0`. Two candidate compatibility objectives, both convex
quadratics because `delta_s_i` is affine in `(W,b)`:

    J_z       = mean_i || A delta_s_i ||^2                      (pre-tanh, metric A)
    J_lin_h0  = mean_i || D_i A delta_s_i ||^2,  D_i = diag(1 - tanh(z_i^0)^2)

The true functional quantity is `J_true = mean_i || tanh(z_i^0 + A delta_s_i) -
tanh(z_i^0) ||^2`, which is NOT convex and is used for **reporting only**.

Closed form, verified numerically to 1e-12: for one constraint `d^T(s0+delta)
>= gamma` with `m = d^T s0`, `r = gamma - m > 0`, and PD metric `Q`,

    delta* = r Q^-1 d / (d^T Q^-1 d),      J* = r^2 / (d^T Q^-1 d).

Per-item minima are **lower bounds** on each item's cost term, hence on the
mean; `W` is shared, so the achievable shared cost is `>=` their mean, never
`<=`. They are bounds, not predictions.

## 4. Compatibility metric — comparison and preregistered choice

Measured on the completed V2 path (seed 19), ratio of each proxy to `J_true`:

| alpha | J_z | J_lin_h0 | J_true | J_z/J_true | J_lin_h0/J_true |
|---|---|---|---|---|---|
| 0.25 aC0 | 1.507e+01 | 3.420e+00 | 3.277e+00 | 4.60 | **1.043** |
| 0.50 aC0 | 6.029e+01 | 1.368e+01 | 1.238e+01 | 4.87 | **1.105** |
| aC0      | 2.412e+02 | 5.471e+01 | 4.288e+01 | 5.62 | **1.276** |
| 1.0      | 4.325e+02 | 9.811e+01 | 6.890e+01 | 6.28 | **1.424** |

`A` has `sigma_max 34.097, sigma_min 1.322, cond 25.79, rank 300, nullspace
dim 0`; 43.3% of `z0` coordinates satisfy `|tanh| > 0.9`, 19.5% `> 0.99`,
and 34.4% have `D < 0.10` — so saturation is widespread and ignoring it (as
`J_z` does) materially misprices movement.

**PREREGISTERED CHOICE: `J_lin_h0` is PRIMARY; `J_z` is a declared SECONDARY
arm, reported for every seed.** Rationale, decided on the evidence above and
not after seeing constrained results: `J_lin_h0` tracks `J_true` to within
1.04-1.42x across the entire measured displacement range whereas `J_z` is off
by 4.6-6.3x; both are convex quadratics; and in this regime `J_lin_h0` is
**conservative** (ratio > 1), because tanh is compressive. Its known failure
mode — under-pricing de-saturating moves, since `D` is the local slope — is
retained as a stated limitation, and is why `J_true` plus the full official
route battery, never a proxy, decide every witness. `J_z` is kept as an arm
because it over-prices rather than under-prices, so any witness it yields is
genuine, and because it admits an exact Kronecker solve (section 7) that
cross-checks the solver.

## 5. Numerical feasibility margin gamma

Constraints are imposed in **raw inner-product units** (`d^T s_i >= gamma`) to
keep them linear; a cosine-unit margin would divide by `||s_i||` and destroy
linearity. Evidence: the V2 fp32 flicker floor is a cosine margin of ~2e-07
(seeds 20/21/22 unresolvable at that scale) and 4.8e-06 was resolvable (seed
19). `||s_i^0||` ranges 2.758-32.170 (median 6.998), so

| gamma_raw | implied cosine margin range | cost penalty vs gamma->0 |
|---|---|---|
| 1e-4 | 3.11e-06 .. 3.63e-05 | +0.00% |
| **1e-3** | **3.11e-05 .. 3.63e-04** | **+0.25%** |
| 1e-2 | 3.11e-04 .. 3.63e-03 | large |

**PREREGISTERED: `gamma_raw = 1e-3`.** Its worst-case implied cosine margin
(3.11e-05) is ~155x the measured fp32 floor and ~6x the smallest resolved
margin, while the per-item lower-bound sum rises only from 1.7519e+02 to
1.7563e+02 (+0.25%) versus `gamma -> 0`. It therefore buys numerical
robustness without imposing a semantic margin objective. For contrast,
`cab7506` uses a **cosine** hinge scale `m = 0.01`, ~70x larger than this
choice's median implied cosine margin; since cost scales as `r^2`, that
demands a far stronger condition than strict correctness.

The **official strict evaluator remains the final authority**; `gamma` is a
solver-side robustness device only.

## 6. Headroom — why off-line search is warranted

Seed 19, `gamma_raw = 1e-3`: sum of per-item `J*_z` over the 34 residuals is
1.7563e+02, i.e. mean over 27,981 items **6.2768e-03**. The straight line at
`alpha_C0` achieves mean `J_z` **2.4117e+02** — about **38,000x** the lower
bound. Baseline scale `mean ||A s^0||^2 = 2.5544e+03`, so the bound is ~0.0002%
of it. Enormous headroom exists between what the constraints require and what
the straight line spends. This is the quantitative case for V3.

## 7. Scalable constraint strategy (cutting-plane / active set)

All 27,981 x 29,570 inequalities are never materialised.

1. start from `W0,b0`;
2. full-bank evaluation; for every item with margin `< gamma`, add its most
   violated competitor(s) (up to `TOP_V = 3` per item per round);
3. solve the convex QP restricted to the current active set `K`;
4. re-evaluate the full bank; go to 2.

Terminate on: all items `>= gamma` (**SUCCESS**, then official verification);
or `MAX_ROUNDS = 30`; or no new violated constraint and the QP is solved to
tolerance; or active set `> K_MAX = 20000`.

Inner solve: the dual in `K` variables with `lambda >= 0`, requiring `H^{-1}a`
products. For the **J_z arm** `H` is exactly Kronecker, `H_W = (A^T A) ⊗ Phi`
with `Phi = (1/N) sum_i phi_i phi_i^T`, so `H^{-1}` applies via one 300x300 and
one 512x512 inverse — an exact cheap solve that cross-checks the iterative
path. For the **J_lin_h0 arm** `D_i` varies per item, `H = (1/N) sum_i
(phi_i phi_i^T) ⊗ (A^T D_i^2 A)` is a sum of Kronecker terms, so `H^{-1}a` is
obtained by **conjugate gradient** on cheap `H`-vector products
(`rtol = 1e-8`, `max_cg = 500`).

## 8. Endpoints

PRIMARY: **official strict `C_errors == 0`** on all 27,981 canonical targets
against the full 29,571 bank, via the unmodified official evaluator, on a head
reloaded from disk into an isolated model copy, verified twice.

SECONDARY, on every C=0 witness: LTM-only repetition exact; full R canonical;
full R free-AR; naming exact; gate/confidence **if exposed** (V2 established
`repetition_snapshot` does not expose gate — expect UNAVAILABLE); plus `J_z`,
`J_lin_h0`, `J_true`, `||delta_W||`, achieved minimum margin in both raw and
cosine units.

## 9. Comparators (mandatory, all five)

1. deployed source `W0,b0`; 2. P-last C=0 `W*,b*`; 3. straight-line
`alpha_C0` witness (V2, measured); 4. minimum-weight-distance C=0 solution
(same cutting plane with `Q = I`, a declared third arm); 5. the chosen
compatibility-metric C=0 solution. Report whether off-line search materially
moves the Pareto frontier.

## 10. Route-health reference lines

As in V2: the only numerically specified pre-existing lines are the Phase-0
AutoResearch invariants (`-0.02`/`-0.03` relative, `0.8498` absolute, R `+2`,
N `<= 1`), preregistered for AutoResearch candidate promotion over matched
200u **training** continuations, not for a derived non-trained head. Applied as
**INHERITED REFERENCE LINES ONLY**;
`ROUTE_COMPATIBILITY_THRESHOLD_STATUS = UNPREREGISTERED`. No threshold is
invented; continuous Pareto values are reported regardless.

## 11. Interpretation rules

* C=0 found and route metrics within the inherited lines -> **compatible
  witness exists off-line**. Major positive; still not a final model, since
  reproducible *training* to such a point is a separate question.
* C=0 found but route metrics poor -> **this proxy and this optimisation did
  not find a route-compatible witness**. Global coexistence stays OPEN.
* C=0 not found -> `ALGORITHM_FAILED_TO_FIND_WITNESS`. **Never** infeasibility,
  absent a genuine mathematical certificate. No certificate is attempted: with
  `<= 513` linearly independent augmented features an affine map can realise
  arbitrary outputs on small subsets, so small residual sets cannot witness
  infeasibility.
* Per-item lower bounds are bounds on the shared problem, never predictions.

## 12. Scope

Determinism: the cutting plane and QP are deterministic; no data shuffling and
no random initialisation, so **no random seed is required**. If any stochastic
step is later added it must be declared with an explicit generator.

No encoder, `to_semantic.0`, decoder, `sem_to_h0`, gate, WM, bank or R/N
parameter is touched. Source checkpoints are opened read-only and hashed
before and after. Derived heads are labelled
`DERIVED_CONSTRAINED_DIAGNOSTIC / NOT_OFFICIAL_MODEL / NOT_TRAINED` and written
only under a new diagnostic output root. AutoResearch base `3f2a6cc` is not
touched and its registry is not read or written.

No scientific code may change after constrained-search outcomes are seen.
