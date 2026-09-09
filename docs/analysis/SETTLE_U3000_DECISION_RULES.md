# SETTLE u3000 → u3600 — pre-result decision rules

**Date: 2026-09-09.** Recorded, committed and pushed **before** the SETTLE
experiment was submitted and before any u3000 → u3600 milestone existed.

This memo is a record, not an instrument. It changes no training code, no
launcher, no hyperparameter, no threshold, no checkpoint and no experimental
design. The strict ceiling / streak criterion is unchanged: **5 consecutive
distinct scheduled full-lexicon evaluations at R canonical = 0, R free-AR = 0,
N free-AR = 0 and C strict top-1 = 0.**

At the time of writing:

- launcher `scripts/cluster/jeanzay/final_settle_u3000_to_u3600.slurm`;
- reporter `scripts/naming_comprehension/settle_report.py`;
- training driver blob `95295d63560ae4c235a6beee8dfb47166f4ed30d`, unchanged
  since u1400;
- **no SETTLE job had been submitted** — the run ids below have never existed
  on any machine.

---

## AUTHORIZING EVIDENCE — THE u3000 READ-ONLY AUDIT

Executed with the canonical, unchanged tooling (`base123_error_audit.py`
`9884be44…`, `base123_persistence_report.py` `6d4f1522…`,
`last_window_ratio.py` `320cf7eb…`, `c_residual_features.py` `0890509b…`,
all last modified on or before 2026-09-08, i.e. before the u2000 audit ran, so
u2000 ↔ u3000 comparisons are definition-identical). Every audited count
reproduced the CUDA training metrics exactly; the parameter-fingerprint
read-only proof passed in 4/4 runs. Evidence frozen at
`archives/chigh_u3000_residual_audit_20260909/` (read-only, SHA256SUMS).

Strict-C residual counts at u3000 (control arm): **[38, 43, 33, 35]**,
N = [0,0,0,0], R = [5,0,3,3].

u2000 → u3000 persistence (canonical definitions):

| seed | \|u2000\| | \|u3000\| | ∩ | retention | survivor frac | new frac | union |
|---|---|---|---|---|---|---|---|
| 19 | 52 | 38 | 32 | 0.615385 | 0.842 | 0.157895 | 58 |
| 20 | 62 | 43 | 35 | 0.564516 | 0.814 | 0.186047 | 70 |
| 21 | 51 | 33 | 29 | 0.568627 | 0.879 | 0.121212 | 55 |
| 22 | 56 | 35 | 27 | 0.482143 | 0.771 | 0.228571 | 64 |

Mean retention **0.557668**. Four-seed shared core: u2000 = **2** →
u3000 = **1** (`their → they`, bank 55) — **shrank** (growth 0.5). Survivor
top-5 fraction **0.959732**; survivor margin-within-0.01 fraction
**0.496644**; homophone-competitor fraction **0/149 = 0.0**;
mathematically-unavoidable **0/149**. u2950 → u3000 mean-of-seed residual
ratio **0.965495** (per-seed [0.9500, 0.9348, 1.0313, 0.9459]).

Pre-existing hard-tail trigger, thresholds untouched:

| # | condition | value | status |
|---|---|---|---|
| 1 | last-window ratio ≥ 0.85 | 0.965495 | MET |
| 2 | within-seed retention ≥ 0.80 | 0.557668 | NOT_MET |
| 3 | top5 ≥ 0.90 **and** margin-within-.01 ≥ 0.50 | 0.959732 / 0.496644 | NOT_MET |
| 4 | shared core growing (> 1.0) | 0.5 | NOT_MET |

**U3000_HARD_TAIL_TRIGGER = FALSE.**
**RESIDUAL_REGIME = MIXED** — churning by the preregistered retention metric,
with influx nearly stopped (77–88 % of current errors are old) and a shrinking
1-item shared core. No hard-negative or margin intervention is authorized.

---

## EXPERIMENT

**Parents:** `final_chigh_ctrl_u3000_h512_s{19,20,21,22}/checkpoints/step_08334000.pt`

| seed | source sha256 |
|---|---|
| 19 | `f783c2c3878a739921fcd24425337fe88da52e79ceed912695062b681e906d56` |
| 20 | `171a8e2bc681382b4697c299956e4b76e45866ed341cd7de0774b944898832a4` |
| 21 | `e4fd9e9b1ad00c93a754e8b8211b232dbb5ced46079e44e308dad0a951f9aa40` |
| 22 | `dd0a0ca4e2b95e440504d555232f8d4f14c9ff30b3186a2144d931707b544f73` |

**Arms** (seeds 19–22, schedule 1:2:3, horizon u3000 → u3600, full-lexicon
evaluation every 25u = 24 milestones, common re-anchor at step **8,334,000**
in ALL arms including the control):

| arm | run id | R LR | N LR | C LR |
|---|---|---|---|---|
| A ctrl | `final_settle_ctrl_h512_s{seed}` | 3e-5 | 3e-5 | **1e-4** |
| B settle-5e5 | `final_settle_5e5_h512_s{seed}` | 3e-5 | 3e-5 | **5e-5** |
| C settle-3e5 | `final_settle_3e5_h512_s{seed}` | 3e-5 | 3e-5 | **3e-5** |

All other architecture (wm128/enc512/dec512), the shared AdamW and its
moments, RNG, sampler cursors, losses and weights (λ_C .087, λ_N 1.0, τ .10,
dec .5, c_align 0), data, populations and hashes, task definitions,
evaluation definitions and ceiling semantics are **fixed**. This is the exact
CANNEAL bracket rerun at u3000 instead of u1400.

---

## PRIMARY RULE

For each seed *s* and arm *a*:

```
last4_mean_C(a, s) = mean of the strict-C error counts of arm a, seed s,
                     over the LAST FOUR COMMON full-lexicon milestones
```

"Common" = milestones present in every run that produced any data; arms are
never compared across different u. For each settle arm:

```
d_s          = last4_mean_C(settle, s) − last4_mean_C(ctrl, s)
paired_mean  = mean over seeds of d_s
paired_sd    = sample SD (n−1) over seeds of d_s
```

A settle arm is an **eligible winner** only if **all** of:

1. paired_mean < 0;
2. d_s < 0 in **≥ 3 of 4** paired seeds;
3. all guards pass (below);
4. **all four paired seeds are available** (missing-run policy below).

If both settle arms qualify: the one with the **lower paired_mean** wins.
**Effectively tied** is defined as |paired_mean(B) − paired_mean(C)| ≤ 1.0
error: then the **higher-LR / control-proximal arm (5e-5)** wins. If neither
qualifies: **the control wins.** The single endpoint milestone is never the
primary.

## MISSING-RUN POLICY

The reporter may produce descriptive analyses with fewer seeds, and it labels
them: any treatment with fewer than 4 paired seeds prints
`FORMAL_PREREGISTERED_WINNER = INCOMPLETE_FOR_PREREGISTERED_DECISION` and its
`formal_decision_evaluable` flag is 0. A missing paired seed must never
shrink the denominator into a formal "3/4" win. A failed run may be resumed
from its own latest checkpoint (the launcher's requeue path, which re-declares
nothing); the formal decision waits for it.

## STATE-DEPENDENCE CLAIM

CANNEAL (u1400 → u2000, same bracket): paired ΔC vs control was **adverse**:
**+7.75** (5e-5) and **+10.5** (3e-5). SETTLE supports
**STATE_DEPENDENT_SIGN_FLIP** only if a settle arm is an eligible primary
winner here **and** its paired_mean < 0. The reporter prints the historical
CANNEAL effect beside the new effect. Falling variance or rising LTM does
**not** by itself support the claim.

## VARIANCE / STRUCTURAL TEST (exact, no judgment calls)

Within-run variability, per run (arm a, seed s):

```
sd8(a, s) = sample SD (n−1) of the strict-C error counts over the
            LAST EIGHT COMMON milestones
```

Aggregation (fixed now): per arm, `variability(a) = mean over seeds of
sd8(a, s)`; then

```
variance_ratio(settle) = variability(settle) / variability(ctrl)
```

**STRUCTURAL_TAIL_SUPPORTED** (frozen tail) holds for a settle arm iff **all**
of:

1. it is **not** an eligible winner;
2. |paired_mean| ≤ max(1.0, paired_sd)  — "approximately matches control";
3. variance_ratio ≤ 0.5;
4. guards pass and all four paired seeds are available.

## GUARDS

- **Naming:** arm-level mean over seeds of last-4-milestone mean N free-AR
  errors < 0.25.
- **Repetition (canonical):** treatment's arm-level mean of last-4 mean R
  errors must not exceed the control's by > 2.
- **Repetition (genuine free-AR):** same rule, same threshold.
- **LTM:** reported as a scientific secondary metric only. The expected LTM
  recovery in the settle arms is **not** a pass/fail requirement (no
  independently justified threshold exists).
- **Ceiling:** unchanged (5 consecutive distinct full evals at 0/0/0/0).

## DECISION BRANCHES (frozen; evaluated in this order)

1. **CEILING_CONFIRMED** — the driver's own 5-evaluation streak fires in any
   run: stop further optimization of that seed/model; the multi-seed /
   final-replication decision is taken separately.
2. **STATE_DEPENDENT_SETTLING_SUPPORTED** — an eligible winner exists (its
   paired_mean < 0 where CANNEAL's was > 0). Then decide whether to extend
   the winner toward the strict ceiling from its trajectory and guards.
3. **STRUCTURAL_TAIL_SUPPORTED** — no eligible winner, but a settle arm meets
   the frozen structural criterion. Then run the read-only residual audit on
   the settled state and re-evaluate the **existing** hard-tail trigger before
   any targeted intervention. No new criterion may replace it.
4. **STATE_DEPENDENT_SETTLING_REJECTED** — both settle arms have
   paired_mean > 0 and neither meets the structural criterion (the u1400
   result reproduced at u3000). Do not continue lower-LR settling by inertia.
5. **MIXED_SETTLE_THEN_AUDIT** — annotation on branch 2 or 3: the selected
   arm's mean-of-seed last-window ratio over u3550 → u3600 is ≥ 0.98 while
   its last-4 mean C > 0 (material decrease that stalls above zero): audit
   the settled state before further training.

Any later deviation from these rules must be recorded as a deviation, with
its reason, and must not be presented as the original plan.

## SCOPE

`docs/analysis/autoresearch/` is untouched by this memo and takes nothing
from it. The formal CHIGH winner (1.5e-4) remains preserved and archived; the
parents above are the scientifically preferred operating point, chosen and
recorded before this experiment produced any data.
