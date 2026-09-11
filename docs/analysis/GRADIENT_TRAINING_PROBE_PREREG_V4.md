# Amendment V4 — frozen-encoder final-layer GRADIENT-TRAINING probe (preregistration)

**Date: 2026-09-11.** Committed **before any real gradient-training result
exists and before the training code is written.** Branch
`probe/frozen-semantic-head`, parent `a35dbd6551840ce061699184ba9fb62c63086d8f`.
Implementation (separate commit, after this one):
`scripts/naming_comprehension/gradient_training_probe.py` + focused tests.

## 1. Question

The V3.1 convex solver found C=0 + route-compatible final-layer witnesses on
4/4 seeds, and the read-only characterization
(`archives/feasible_region_characterization_20260911`) found that at W0 the
deployed cosine-CE descent direction is ~orthogonal to them (cos 0.001–0.004)
while the active-constraint gamma-hinge preconditioned by the frozen J_lin_h0
metric is aligned 0.49–0.78. That is FIRST-ORDER evidence at one point. This
probe asks whether the mechanism works as a **learning dynamic**: can an
iterative gradient-based rule, starting from the deployed W0, reach official
strict C=0 while staying route-compatible?

Two separable ingredients are tested: CONTENT (which constraints generate
gradient) and GEOMETRY (how that gradient is transformed in parameter space).

## 2. Scope (frozen)

Trainable: ONLY `ltm.to_semantic.2.{weight,bias}` = theta = [W|b] (300 x 513).
Frozen: encoder, `to_semantic.0`, GELU, `sem_to_h0`, decoder, readout, WM, gate,
GloVe bank, populations, tau. Inputs: cached frozen phi (27,981 canonical
items) from `frozen_semantic_head_probe_20260911`. Start: deployed W0 of each
official SETTLE-control u3600 seed. Seeds: **19, 20, 21, 22 — all four, no
screening.** 3 arms x 4 seeds = **12 runs**. CPU, float64, full batch.
No SLURM, no GPU, no AutoResearch contact. a35dbd6 is NOT run.

## 3. Arms (exactly three)

Raw margin of item i at theta: m_i = s_i^T t_T - max_{j != T} s_i^T t_j, with
s_i = theta x_i, x_i = [phi_i; 1], bank rows t L2-normalised, target excluded;
the hardest competitor j*(i) is the argmax (ties -> lowest index, torch.max).

* **ARM A — preconditioned active-constraint training.**
  Loss L_gamma(theta) = SUM_{i: m_i < gamma} (gamma - m_i), gamma_raw = 1e-3.
  Gradient g = - SUM_{i: m_i < gamma} d_i x_i^T, d_i = t_T - t_{j*(i)} (only
  currently violated / insufficient-margin constraints generate gradient).
  Direction p = - H^{-1} g, with H the J_lin_h0 operator
  H(theta) = (1/N) SUM_i A^T D_i^2 A theta x_i x_i^T, **D_i = diag(1 - tanh(z_i^0)^2)
  FROZEN at W0 and never updated** (not a changing natural-gradient metric).
  H^{-1} via the V3 `Metric("Jlinh0").Hinv` CG (rtol 1e-8, max 500 — V3 constants).
  No compatibility gradient term, no lambda, no route constraint.
* **ARM B — raw active-constraint training.** Identical L_gamma and g;
  direction p = -g. Isolates GEOMETRY.
* **ARM C — deployed cosine-CE control.** L_CE = mean_i CE(normalize(s_i) @ t^T / 0.10, T);
  direction p = -grad L_CE. No preconditioning. Isolates CONTENT (vs A/B).

## 4. Step rule — direction test, not LR search

Calibration at W0 (seed 19, timing/norms only, no step applied, no trial point
evaluated): ||g_gamma|| = 18.77, ||H^{-1}g|| = 477.4, ||grad L_CE|| = 0.0447.
Scales differ by four orders of magnitude, so a shared learning rate would be
meaningless. Every arm therefore uses the SAME **length-normalised Armijo
backtracking** on its OWN objective:

    alpha_0 = R0 / ||p||_F          (first trial moves exactly R0 in theta)
    accept first alpha = alpha_0 * BETA^k with
        L(theta + alpha p) <= L(theta) + ARMIJO_C * alpha * g^T p
    R0 = 0.1, BETA = 0.5, ARMIJO_C = 1e-4, MAX_BACKTRACKS = 20

R0 = 0.1 was fixed from pre-existing scale evidence only: it is ~0.55x the
official per-25u final-layer drift (0.18) and 7–15% of the V3 witness
distances (0.83–1.49). Arms differ ONLY in direction p. No post-result change.

## 5. Stopping (checked in this order each iteration)

1. **CONVERGED** — every raw margin >= GAMMA_RAW - FEAS_TOL_RAW (V3.1 threshold,
   1e-7): the common gamma robustness target, identical for all arms
   (equivalently L_gamma = 0).
2. **LINE_SEARCH_FAILED** — no Armijo-acceptable step within MAX_BACKTRACKS.
3. **NUMERICAL_FAILURE** — non-finite theta/loss, g^T p >= 0 for nonzero g, or
   CG relative residual > 1e-6.
4. **MAX_ITERS** = 150 (max total motion 15 in theta, i.e. P-last scale).
5. **WALL_CAP** = 7200 s per run.

All are algorithm outcomes; none is evidence of infeasibility.

## 6. Primary endpoint and C=0 detection

PRIMARY (per seed/arm): **did gradient training reach official strict
C_errors == 0? YES/NO**, with the first iteration. Whenever the internal count
of items with m_i <= 0 is zero, the UNCHANGED official evaluator
(`frozen_probe.comprehension_metrics` via `official_strict_errors`) is run; the
first iterate it scores 0/27,981 is FIRST_C0. gamma is the training robustness
target, not the scientific success condition.

## 7. Evaluation cadence and route battery

Every iteration (cheap, no battery): objective, #(m<gamma), internal strict
errors, min raw margin, ||Delta theta||, cosine and Euclidean distance to the
V3 reference Delta, J_z, J_lin_h0, J_true, overlap of the violating set with the
V3 active set and with the source C residual.

V3 reference per seed: the V3.1 **J_lin_h0** witness (the exact minimum-J_lin_h0
C-feasible point, same metric as Arm A's preconditioner); the V3.1 best witness
(J_z for seed 19) is reported as secondary.

Full official route battery (`coexistence_probe.full_battery` on an isolated
model, unchanged): at FIRST_C0 and at the final iterate if different. Source
values are the V3.1 baselines. Reported: LTM-only exact, R canonical, R free-AR,
N exact, gate, J_z, J_lin_h0, J_true, ||Delta W||, distance to V3.

## 8. Interpretation (frozen)

Historical reference lines (context only, NOT promoted to official criteria):
dLTM >= -0.02, absolute LTM >= 0.8498, R excess <= +2, N <= 1. A learned C=0
point meeting all four is labelled descriptively **TRAINED_COMPATIBLE_WITNESS**.

Arm A **dominates** a control on a seed iff that control either did not reach
official C=0, or reached it with LTM at its FIRST_C0 lower than Arm A's by
>= 0.02, or violates a historical line that Arm A satisfies.

n_A = number of seeds where Arm A yields a TRAINED_COMPATIBLE_WITNESS at FIRST_C0
AND dominates both Arm B and Arm C.

| n_A | status | AutoResearch readiness |
|---|---|---|
| 4 | GRADIENT_TRAINING_STATUS=LEARNABLE_MECHANISM_CONFIRMED | READY |
| 3 | GRADIENT_TRAINING_STATUS=PARTIAL | READY |
| 2 | GRADIENT_TRAINING_STATUS=PARTIAL | NOT_READY |
| <=1 | GRADIENT_TRAINING_STATUS=MECHANISM_NOT_CONFIRMED | NOT_READY |
| defect stops 12/12 | GRADIENT_TRAINING_STATUS=BLOCKED | NOT_READY |

## 9. Determinism

Full batch, no shuffling, no RNG, fixed start W0, fixed thread count per
process, deterministic tie handling. A rerun of a synthetic fixture must be
bit-identical (tested). No scientific code changes after any real result.
