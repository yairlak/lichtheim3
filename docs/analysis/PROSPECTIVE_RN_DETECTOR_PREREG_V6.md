# Amendment V6 — prospective R/N detector + first-hit terminal repair (preregistration)

**Date: 2026-09-12.** Committed **before any pilot run is submitted**. Worktree
`wt-prospective-recipe`, branch `prospective/rn-detector-v6`, base
`2e559e503b9672a3b9e89951fb114abfb77e0ad2`.

## 1. Question

C is closed for the ceiling: compatible strict C=0 endpoints exist 4/4, Arm A
reaches them by gradient training 4/4, and both *tested* R/N-perfect official
sources were converted to exact `Rcan=Rfree=N=C=0` (seed 20 u3600; seed 21
u3400, a non-endpoint source with rebuilt φ, `sem_to_h0` geometry and
source-specific `J_lin_h0`). Seeds 19/22 were never repair failures — their
retained trajectories never exposed an eligible milestone.

**Prospectively: can an `Rcan=Rfree=N=0` source be detected and retained online
under a denser detector, and does the already-fixed Arm A then yield exact
`Rcan=Rfree=N=C=0`?**

## 2. Scope and what is frozen

No C-side search. Arm A is a **constant**, not a variable. Training recipe is
the official SETTLE control recipe, unchanged. The only intended difference
from the official runs is the **denser full-evaluation / checkpoint grid**.

## 3. The three sources (frozen, with hashes)

| task | seed | source run | step | u | SHA256 |
|---|---|---|---|---|---|
| 0 | 19 | `final_settle_ctrl_h512_s19` | 10,000,800 | 3600 | `6070317ee5296996ea2b30bd191cbcd53b6a24991d3b49136408513d9510c58a` |
| 1 | 22 | `final_settle_ctrl_h512_s22` | 10,000,800 | 3600 | `f6cd1679cb8fd8b7d4c1352b9b3bdd77ec2ec4249e06af179b21ddc6cdbffae3` |
| 2 | 20 | `final_chigh_ctrl_u3000_h512_s20` | 8,334,000 | 3000 | `171a8e2bc681382b4697c299956e4b76e45866ed341cd7de0774b944898832a4` |

Seed 20's parent is provably the **control** arm: its `lr_policy` is
`{repetition 3e-5, naming 3e-5, comprehension 1e-4}`, versus the CHIGH
treatments at 1.5e-4 (`15e5`) and 2e-4 (`2e4`); `c_align_weight` 0.0,
`dec_weight` 0.5. It is the same parent the official SETTLE control resumed
from. All three sources carry `optimizer_state_dict` with
`optimizer_policy = shared_adamw` (single shared moment bank) and full
R/N/C/pool cursors.

## 4. Horizons and exact detector grids

`1u = 2778 optimizer steps` (verified: u3600·2778 = 10,000,800 = official
END_STEP; u3000·2778 = 8,334,000 = official SOURCE_STEP; u3400·2778 =
9,445,200 = the seed-21 V5 source). Grids are generated programmatically, never
approximated.

- **Seeds 19 / 22**: u3600 → u3900 (+300u). Detector steps 10,014,690 …
  10,834,200, spacing 13,890 (5u), **60 points**.
- **Seed 20**: u3000 → u3300 (+300u). Detector steps 8,347,890 … 9,167,400,
  spacing 13,890 (5u), **60 points**.

Seed 20's +300u horizon (rather than +100u) is frozen **before** submission:
historically it produced eligible milestones frequently (11/24 on the 25u
grid), but the continuation is **not bit-exact**, so a larger fixed horizon
gives the positive control a materially better chance to express eligibility
**without introducing any training hyperparameter search**.

## 5. Resume flags — derived from the driver's own guards, not chosen freely

`load_state_dict` collects `changed` from: schedule ratio, `schedule_anchor`
(only when `--reanchor-schedule` and the checkpoint's anchor ≠ its
`global_step`), `lr_policy`, `dec_weight`, `optimizer_policy`. A non-empty
`changed` demands `--phase-transition`.

- **Seeds 19/22 (u3600 continuation): neither flag.** Their `lr_policy`,
  `dec_weight`, `optimizer_policy`, `schedule` and ratio already match, so
  `changed` is empty — a pure continuation that preserves the macro-cycle from
  anchor 8,334,000. Passing `--reanchor-schedule` would append
  `schedule_anchor` and **restart the task-order cycle**, a training-stream
  change this design forbids.
- **Seed 20 (u3000 branch): `--reanchor-schedule --phase-transition`**, exactly
  as the official SETTLE control arm did from this same parent (the official
  launcher states all arms, control included, re-anchor at the branch step).

## 6. Eligibility (frozen)

At each full detector point, in chronological order:

```
eligible  iff  Rcan_errors == 0  AND  Rfree_errors == 0  AND  Naming_errors == 0
```

Full lexicon (29,571) for all three. **C does NOT enter source eligibility** —
repairing C is the intervention's purpose.

## 7. Selection: FIRST HIT, no lookahead

The **first** eligible detector point in time order is the selected source.
Forbidden: retrospective minimum-C, best-LTM, latest-hit, comparing repair
outcomes across hits, any lookahead. If the first hit's repair fails, **no
other source is tried** — that would violate the prospective rule.

Applying "first in time order" to a complete detector log is equivalent to an
online trigger, because "first" cannot see the future. The only difference from
an in-loop trigger is that training does not stop at the hit, which costs a
little compute and is accepted to keep the official driver untouched.

## 8. Retention

Every detector point writes a checkpoint: the driver's `torch.save` is tied
exactly to `at_full`, so a 5u `--full-eval-at` grid retains all 60 checkpoints
per task. **No two-consecutive requirement for intervention eligibility.**
Seed 21 proved one isolated eligible state suffices. The separate *training
ceiling declaration* (R/N **and C**, two consecutive milestones) is unchanged
and not used here.

## 9. Fixed Arm A (no search)

Trainable only `ltm.to_semantic.2.{weight,bias}`; raw active-constraint γ-hinge;
`γ_raw = 1e-3`; frozen **source-specific** `J_lin_h0` preconditioner with D
frozen at the selected source; `R0 = 0.1`, `β = 0.5`, Armijo `c = 1e-4`, 20
halvings; `FEAS_TOL_RAW = 1e-7`; `MAX_ITERS = 150`; 7200 s cap; unchanged CG
semantics; fp64 fitting, fp32 deployed evaluation; official evaluators. Code:
`gradient_training_probe.train_run(arm="A")` driven by
`ceiling_source_completion.py`, both byte-identical to the V5 result
(`2e559e5`). Every source-dependent quantity is rebuilt from the selected
checkpoint; no earlier cache or metric is reused.

## 10. Endpoint

In an isolated deployed model copy at the first official C=0 head:
R canonical (full lexicon), genuine free-AR repetition, naming, and strict
canonical top-1 C against the full 29,571-row bank.

```
PROSPECTIVE_MATURE_STATE_EXACT_WITNESS
  iff  first-hit source selected without lookahead
  AND  Rcan = 0 AND Rfree = 0 AND N = 0 AND C strict top-1 = 0
```

No relaxation. Secondary: LTM exact and Δ vs source, gate, J_z, J_lin_h0,
J_true, ‖ΔW‖, ‖Δb‖, iterations, minimum raw margin, source C residual.

## 11. Compute budget

Measured from the official control log: training 0.0116 s/step, one
full-lexicon detector evaluation + checkpoint 45.6 s. Per +300u task:
833,400 steps ≈ 2.69 h training + 60 × 45.6 s ≈ 0.76 h evaluation ≈ **3.45 h**,
plus Arm A (~2–6 min CPU) and the official battery (~5–10 min CPU).
**≈10.35 GPU-h total** for three tasks run concurrently; wall ≈4 h.
Requested walltime 10:00:00 per task (margin for requeue, post-hit repair,
battery and archiving). Storage 60 × 29.7 MB ≈ 1.8 GB per task.

## 12. Non-interference gate (must PASS before submission)

From the same checkpoint, in two independent processes: A = K training steps;
B = one full-lexicon detector evaluation, then K training steps. Model
parameters, optimizer moments, explicit `Generator` states, the global RNG
state, cursors, `rep_epoch` and `global_step` must be **bitwise equal**. B also
snapshots immediately before and after the evaluation. Any mismatch ⇒ BLOCKED,
no submission. This is distinct from the known non-bit-exact historical-resume
issue.

## 13. Interpretation (frozen before results)

**`torch_deterministic = False`**, and the official provenance states the
recipe is "deliberately NOT a bit-exact replay". **These continuations are
prospective NEW realizations, not bit-exact replays.** Consequently no
experiment can recover whether a *specific archived* trajectory passed through
an eligible state between its 25u milestones; that question is answerable only
statistically, going forward.

- Success on a task ⇒ `PROSPECTIVE_MATURE_STATE_EXACT_WITNESS`. This is **not**
  a full from-scratch prospective replication: seeds 19/22 start at u3600 and
  seed 20 at u3000. Fresh step-0 replication remains later.
- Seeds 19/22 no hit ⇒ `NO_ELIGIBLE_HIT_IN_60_DETECTOR_POINTS`. Sufficient
  practical evidence to consider a targeted R-acquisition intervention next;
  **not** a proof that eligibility cannot occur.
- Seed 20 no hit ⇒ `NO_ELIGIBLE_HIT_WITHIN_CONTROL_HORIZON`. **Not** evidence
  that the infrastructure is broken — it is a new non-bit-exact realization.
- A null does not establish impossibility, and no eligible state found after
  the first hit is ever examined.

## 14. Out of scope

No C-side search. No architecture, schedule, LR, `dec_weight`, `c_align`,
optimizer, lexicon, population or evaluator change. No AutoResearch contact:
registry, enablement, deployment, reconciliation and Stage A are untouched.
No other job is submitted.
