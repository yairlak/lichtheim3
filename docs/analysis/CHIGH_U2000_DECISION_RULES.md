# CHIGH u2000 → u3000 — pre-result decision rules

**Date: 2026-09-08.** Recorded **before** the CHIGH experiment was submitted and
before any u2000 → u3000 milestone existed.

This memo is a record, not an instrument. It changes no training code, no
launcher, no hyperparameter, no threshold, no checkpoint and no experimental
design. In particular **the strict ceiling / streak criterion is unchanged**:
5 consecutive distinct scheduled full-lexicon evaluations at R canonical = 0,
R free-AR = 0, N = 0 and C = 0.

At the time of writing:

- CHIGH launcher `scripts/cluster/jeanzay/final_chigh_u2000_to_u3000.slurm`
  blob `2fc7f2e75a037cfbd1bd4f52cf33ff3daa4717ff`, byte-identical to commit
  `82b4857897089607412e1505ff8517b1b6128eb5`;
- training driver `scripts/naming_comprehension/train_joint_scratch.py` blob
  `95295d63560ae4c235a6beee8dfb47166f4ed30d`, unchanged;
- **no CHIGH job had been submitted.**

---

## SOURCE

An independent scientific review was performed after the CANNEAL u2000 freeze
and the read-only u2000 persistence audit. Its conclusions, adopted here:

- **GO** on the already-prepared CHIGH u2000 → u3000 experiment, unchanged.
- **Keep the Ueno-style task allocation fixed at 1:2:3.** 2:2:3 was tested
  from the same mature state with N and C exposure exactly matched and lost on
  comprehension in 4/4 seeds at all 8 paired milestones (`PREFER_223 = false`).
  No ratio change on the table helps the binding task.
- **No hard-negative and no margin intervention**, because the preregistered
  switch trigger remained FALSE. The decision is the trigger's, not a
  judgement call made after seeing the residuals.
- No lexical curriculum: the u1400 residual characterisation found the
  surviving C items are *shorter* than the population (orthographic length
  SMD ≈ −0.53, phoneme length SMD ≈ −0.52, same sign in 4/4 seeds), so there
  is no defensible length or frequency schedule to aim at.

The reviewer's reasoning for GO rather than a plain continuation: the sweep's
control arm **is** the continuation of the current best recipe, on the same
seeds, horizon and milestones. The 8 additional array slots buy the only thing
continuation cannot provide — which side of 1e-4 the C-LR optimum lies on —
and if both raised arms fail, nothing is lost.

---

## ARMS

Branching from the exact u2000 CANNEAL **control** endpoints
`final_canneal_ctrl_h512_s{19,20,21,22}/checkpoints/step_05556000.pt`,
schedule fixed at 1:2:3, seeds 19–22, u2000 → u3000, evaluation every 25u:

| arm | run id | C LR | R LR | N LR |
|---|---|---|---|---|
| control | `final_chigh_ctrl_u3000_h512_s{seed}` | **1e-4** | 3e-5 | 3e-5 |
| high | `final_chigh_15e5_u3000_h512_s{seed}` | **1.5e-4** | 3e-5 | 3e-5 |
| higher | `final_chigh_2e4_u3000_h512_s{seed}` | **2e-4** | 3e-5 | 3e-5 |

R LR = N LR = 3e-5 throughout, in every arm. The comprehension learning rate
is the only treatment factor.

---

## PRE-RESULT DECISION RULES

Fixed on 2026-09-08, before any result. Any later deviation must be recorded
as a deviation, with its reason, and must not be presented as the original
plan.

### Primary winner

- Evaluate at the **deepest common milestone** — the deepest milestone present
  in all 12 runs. Arms are never compared across different u.
- The winner is the arm with the **lowest mean strict C top-1 error count**.
- The winner **must beat the control in ≥ 3 of the 4 paired seeds**. An arm
  that wins on the mean but not on the paired seed count does not win.
- **Ties go to the lower LR.**

### Operational early failure

- If an arm has mean C errors **> 1.5 × control at two consecutive milestones
  at or after u2200**, it may be cancelled to free the second allocation.
- This is an operational allowance, not a scientific verdict: a cancelled arm
  is reported as cancelled at the milestone where the rule fired.

### Plateau failure

- If an arm has a **last-window C residual ratio ≥ 0.98 for two consecutive
  50u windows while remaining worse than the control**, that arm is declared
  **failed**, regardless of its endpoint value.
- Rationale: residual error under a constant LR settles at a noise floor that
  scales with the LR. A high-LR arm can transit faster early and then flatten
  above the control. The trajectory, not the endpoint, distinguishes the two.
- The ratio uses the single definition already in use: **mean over seeds of
  C_errors(u_final) / C_errors(u_prev)**, never a ratio of pooled means.

### Guards

- **Naming must stay at 0 errors.**
- **Mean R errors must not exceed the control by more than 2.**

An arm violating a guard does not win, whatever its C errors.

### Escalation

- If **2e-4 wins** AND its marginal improvement over 1.5e-4 **exceeds the
  across-seed SD**, then **at most ONE further LR bracket above 2e-4** is
  allowed.
- Otherwise the **C-LR escalation search is closed.**

This is the line between dose-finding and unbounded hyperparameter search, and
it is drawn here in advance rather than after seeing which arm won.

### Not changed

The strict ceiling / streak criterion is untouched: 5 consecutive distinct
scheduled full-lexicon evaluations at 0/0/0/0 on R canonical, R free-AR, N and
C. No top-k relaxation. No change to the C population (27,981 canonical
phonological inputs retrieved against the complete 29,571-word bank), the N
population (29,571 true GloVe inputs, free AR), or the R population (29,571
words, canonical and free AR).

---

## RECORDED AUDIT STATE AT u2000

Read-only per-item audit of the four u2000 CANNEAL control endpoints, and the
preregistered u1400 → u2000 persistence chain. **No threshold was changed to
produce these numbers.**

Last 50u window, u1950 → u2000, per seed:

| seed | C errors u1950 | C errors u2000 | ratio |
|---|---|---|---|
| 19 | 55 | 52 | 0.945455 |
| 20 | 62 | 62 | 1.000000 |
| 21 | 55 | 51 | 0.927273 |
| 22 | 60 | 56 | 0.933333 |

**Mean-of-seed residual ratio = 0.951515.**

Preregistered C switch trigger:

| # | condition | value | met |
|---|---|---|---|
| 1 | last-window ratio ≥ 0.85 | **0.951515** | **MET** |
| 2 | within-seed residual retention ≥ 0.80 | **0.486988** — per seed [0.489362, 0.540000, 0.451923, 0.466667] | **NOT MET** |
| 3 | survivors predominantly near-tie / top5 (top5 ≥ 0.90 **and** margin-within-0.01 ≥ 0.50) | top5 **0.945701**, margin ≤ 0.01 **0.438914** | **NOT MET** |
| 4 | shared persistent core growing (> 1.0) | **0.2222** | **NOT MET** |

**ALL_CONDITIONS_MET = FALSE.**

Core structure at u2000:

- **four-way C residual core = 2 items**
- **four-way R residual core = 0 items**

Interpretation recorded at the time: the residual C set continues to churn
strongly — roughly half of each seed's residual turns over across the window,
the shared core is shrinking rather than growing, and margins are not
predominantly near-tie. This is **not** a stable hard-negative tail, and a
margin or hard-negative intervention has nothing stable to mine. The trigger
exists precisely to keep that intervention from being deployed on the basis of
a post-hoc reading of the residuals; it says no, and it is respected.

For context, the C trajectory under the retained recipe (C LR = 1e-4) is still
moving substantially: mean C errors 100.75 at u1400 → 55.25 at u2000. The
earlier u1350 → u1400 apparent stall (mean-of-seed ratio 1.027615) was a
short-window fluctuation, not a plateau.

---

## SCOPE

Recorded, not acted on: the four-way R core of 0 at a persistent ~3 mean R
errors. The strict ceiling requires R = 0 simultaneously with C = 0 across
five consecutive evaluations, so R may become the binding constraint once C
resolves. No R intervention is authorised by this memo, and none is prepared.

Any separate AutoResearch-style exploration is out of scope here and must not
share run ids, checkpoints or lineage with this controlled sequence.
