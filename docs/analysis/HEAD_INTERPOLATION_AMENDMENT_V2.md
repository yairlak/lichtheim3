# Ceiling root-cause diagnostic — Amendment V2: weight-interpolation compatibility test

**Date: 2026-09-11.** Written and committed **BEFORE any interpolation outcome
was computed.** Branch `probe/frozen-semantic-head`, parent commit
`aa67ca904617018c28d5669ba563d37a3290d729`.

## 1. Why the previous preregistration is superseded

The head-only probe preregistration (`READY_FOR_IMPLEMENTATION_REVIEW`) is
**withdrawn**. Three reasons, all established from the verified probe at
`aa67ca9`:

1. **Isolated C feasibility is already proven.** Strict canonical C = 0 was
   reached on 4/4 SETTLE-control u3600 seeds by training only the final
   `to_semantic` `Linear(512,300)` on frozen phi, FEASIBLE_VERIFIED by the
   official evaluator through a reloaded isolated model, source checkpoints
   byte-identical. The question that preregistration asked is answered YES.
2. **Route compatibility was omitted, and that omission was disqualifying.**
   `to_semantic` feeds `sem_to_h0 -> decoder` (`h0 = tanh(sem_to_h0(s_hat))`),
   so it sits inside the LTM repetition path. Freezing `sem_to_h0` and the
   decoder does NOT protect LTM repetition, because training `to_semantic`
   moves the decoder's input distribution. Every verified C = 0 head collapsed
   LTM-only repetition from ~0.88 to 0.03-0.11.
3. **The scientific question therefore changed** from capacity/expressivity to
   a JOINT-CONSTRAINT / COMPATIBILITY problem.

## 2. What is NOT claimed

The `aa67ca9` result does **not** establish: that cosine-CE alone can reach
C = 0 (single-LR 1e-3 cosine-CE did not, labelled
`EMPIRICAL_FAILURE_NO_CERTIFICATE` — this is NOT "cosine-CE cannot solve C");
that a C = 0 solution compatible with the frozen LTM decoder exists; that the
joint model can reach 100/100/100; or that encoder capacity could never matter
under some future architecture.

## 3. New question

Does there exist, **within the same final-linear head family**, a semantic
mapping that simultaneously reaches strict canonical C = 0 under the real
frozen GloVe/cosine task AND remains compatible with the existing frozen LTM
decoder?

This amendment tests only the **straight line** between the two heads we
already possess. It is a deterministic analysis, not a search.

## 4. Construction

Per seed, with `phi` frozen and the final layer affine:

    W(alpha) = (1-alpha) W0 + alpha W*
    b(alpha) = (1-alpha) b0 + alpha b*

    s_i(alpha) = W(alpha) phi_i + b(alpha)
               = (1-alpha) s_i(0) + alpha s_i(1)          (exact, affine)

`W0,b0` = deployed u3600 final layer; `W*,b*` = verified P-last C = 0 final
layer. Weight-space interpolation therefore equals output-space interpolation;
this is asserted numerically, not assumed.

## 5. Analytic C threshold (primary method; no dense sweep)

Bank rows `t_j` are L2-normalised, so for any non-zero `s`,
`argmax_j cos(s,t_j) = argmax_j s^T t_j` (the `1/||s||` denominator is common
and positive). Strict correctness at item `i` with target row `T=tau(i)`
requires, for every `j != T`:

    s_i(alpha)^T (t_T - t_j) > 0

Writing `u_i = s_i(0)`, `delta_i = s_i(1) - s_i(0)`, `s_i(alpha) = u_i + alpha*delta_i`:

    A_ij = u_i^T (t_T - t_j)          (value at alpha = 0)
    B_ij = delta_i^T (t_T - t_j)      (slope)
    constraint: A_ij + alpha*B_ij > 0

Each constraint is **affine in alpha**, so:

* `B_ij > 0` -> satisfied for `alpha > -A_ij/B_ij`  (lower bound)
* `B_ij < 0` -> satisfied for `alpha < -A_ij/B_ij`  (upper bound)
* `B_ij = 0` -> satisfied for all alpha iff `A_ij > 0`; otherwise infeasible
  at every alpha.

**Derived, not assumed:** an intersection of half-lines in one variable is
convex, so each item's feasible set is a single open interval `(L_i, U_i)`, and
the global feasible set over all 27,981 items is the single open interval
`(max_i L_i, min_i U_i)`. Non-monotone per-item feasible sets are therefore
algebraically impossible along this line. `alpha_C0 := max_i L_i` is the
infimum of the global feasible interval; C = 0 holds for `alpha` strictly
greater than it. Items already correct at `alpha = 0` with all `A_ij > 0`
contribute `L_i = 0`.

Handling: strictness is respected by reporting `alpha_C0` as an infimum and
evaluating at `alpha_C0 - eps` and `alpha_C0 + eps`; `eps = 1e-6` (fixed here,
before results). Zero-slope constraints are tested exactly as above. The
analytic boundary is **verified against the official strict evaluator** on
both sides.

`alpha = 1` is known feasible (C = 0 verified at `aa67ca9`), so the global
interval is non-empty and contains 1.

## 6. Evaluation points — fixed before results

`alpha in {0, 0.25*aC0, 0.50*aC0, 0.75*aC0, 0.90*aC0, aC0-eps, aC0+eps, 1.0}`.
No point is chosen or added after seeing any metric.

At each point, via the OFFICIAL evaluators only
(`frozen_probe.comprehension_metrics`, `train_tasks.evaluate_comprehension_subset`,
`repetition_snapshot`, `evaluate_naming`, and the trainer's own
`free_ar_repetition`), on an isolated deep-copied model:

C strict top-1 errors; C CE at tau = 0.10; C margin distribution; LTM-only
repetition exact; full repetition canonical; full repetition free-AR; naming
exact; gate/confidence diagnostics where the existing snapshot exposes them
(reported UNAVAILABLE otherwise).

## 7. Route-compatibility criterion — provenance and status

The only numerically specified route-health guards that pre-exist are the
Phase-0 AutoResearch invariants (`autoresearch/phase0/invariants.json`):

    relative_ltm_mean_guard            = -0.02
    relative_ltm_per_seed_guard        = -0.03
    absolute_ltm_review_line           =  0.8498   (= 0.8698 - 0.02)
    canneal_ctrl_u2000_ltm_reference   =  0.8698
    repetition_guard_max_excess_errors =  2.0
    naming_guard_max_candidate_errors  =  1

These were preregistered for **AutoResearch Phase-0 candidate promotion over
matched 200u training continuations**, not for a non-trained derived
interpolated head inside an official diagnostic. They are therefore applied
here as **INHERITED REFERENCE LINES ONLY**, and the binary route-compatibility
verdict for this probe class is labelled:

    ROUTE_COMPATIBILITY_THRESHOLD_STATUS = UNPREREGISTERED

No threshold is invented. Full continuous Pareto values are reported
regardless, and any verdict is stated against the inherited lines with that
label attached.

## 8. Preregistered interpretation table

| case | observation | interpretation |
|---|---|---|
| 1 | C = 0 reached while LTM and full R/N stay within the inherited lines | a compatible solution ALREADY EXISTS on the straight line. Major positive; still NOT the final model — reproducible training remains required |
| 2 | C = 0 only after substantial LTM collapse, smooth monotone trade-off | the straight line exposes a genuine task-compatibility trade-off; a multi-objective/constrained probe is justified. Does NOT prove global incompatibility |
| 3 | LTM healthy through most of the path, abrupt collapse near alpha_C0 | the final few C constraints conflict specifically with decoder-compatible semantic geometry; identify the responsible items |
| 4 | non-monotone / mixed | do not reduce to a scalar trade-off; inspect item- and decoder-state structure before choosing any objective |

## 9. Scope

No `backward()`, no optimizer, no Adam/AdamW, no new objective, no SLURM, no
GPU requirement, no training of any kind. No source checkpoint is written.
Derived artifacts are labelled `DERIVED_INTERPOLATION_DIAGNOSTIC` /
`NOT_OFFICIAL_MODEL` / `NOT_TRAINED` and live only under a new diagnostic
result directory. AutoResearch base `3f2a6cc` and Phase-0 Stage A remain
untouched and superseded.

No scientific code may change after interpolation outcomes are seen.
