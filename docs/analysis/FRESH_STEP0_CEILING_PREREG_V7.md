# V7 — Fresh step-0 ceiling replication (preregistration)

```
PREREGISTRATION_STATUS=FROZEN_BEFORE_ANY_V7_TRAINING_RESULT
```

**Date: 2026-09-12.** Worktree `wt-fresh-v7`, branch `fresh/step0-ceiling-v7`,
based on the V6 completion commit `78f550570b82b3230cb621d9d000600ed192eb74`.
Executable commit (orchestration + tests, no scientific change):
**`73a968edcda4dee951acc4b41a05647606414318`**. Authoritative execution
contract: `LICHTHEIM3_CEILING_EXECUTION_V7_1.md` (handoff), with the V6
terminal-report correction that the completion archive is
`archives/prospective_rn_detector_v6_completion_20260912/` (22/22 verified
before and after lock, 24/24 read-only, verdict
`EXACT_WITNESS_FOUND_BOTH_FIRST_HIT_SOURCES`).

Committed **before** any V7 training run exists; no V7 outcome has been
observed. Nothing below may change after the first result is seen.

## 1. The only question

> Can the already validated **detector → FIRST-HIT → fixed Arm A → exact
> official battery** policy reproduce from **fresh step 0** under the frozen
> H512 joint recipe?

C is closed. During this cohort nothing is retuned: not Arm A, its objective,
active constraints, compatibility geometry / preconditioner, C margin, repair
LR / optimizer / stopping, strict C evaluator; no W2-only experiment, no
architecture / hidden-size / LR / task-ratio sweep, no resurrection of
Stage-A / DEC1 / RLR searches.

Anchors (V6, unchanged): preregistration `ab25f7c93813d85272ac142cd6209dc30ae689ae`;
prospective-lineage base `2e559e503b9672a3b9e89951fb114abfb77e0ad2`; V6
completion fix `78f550570b82b3230cb621d9d000600ed192eb74`; Arm-A
preregistration `054633cdf173c27296bdbc406775329bd0bc38a4`; Arm-A implementation
`3912d758e4864a2e8a297f38311c6162772562d8`. Seed-19 FIRST-HIT u3825 / step
10,625,850 / sha256 `a5f21de9…aad76c`; seed-20 FIRST-HIT u3040 / step 8,445,120 /
sha256 `0657f410…9dc79f3`.

## 2. Cohort — frozen, with cleanliness evidence

| slot | seed | run id |
|---|---:|---|
| **P1** | **31** | `fresh_ceiling_v7_p1_s31` |
| **P2** | **32** | `fresh_ceiling_v7_p2_s32` |
| **P3** | **33** | `fresh_ceiling_v7_p3_s33` |
| **P4** | **34** | `fresh_ceiling_v7_p4_s34` |

Canonical priority is **P1 > P2 > P3 > P4**, fixed here, never chosen from
outcomes.

**Cleanliness proof (local, 2026-09-12).** `fresh_ceiling_v7.py seed-audit`
over every archive (`archives/*`, 24 trees incl. manifests, SHA256SUMS,
SACCT and provenance texts), every probe directory, `outputs/`,
`checkpoints/`, the AutoResearch registry (`autoresearch/registry/events.jsonl`,
seeds present: 19 only) and lineage table (`AUTORESEARCH_LINEAGE.tsv`: seeds
19–22 only), plus `git grep` over all eleven worktrees (docs, preregistrations,
configs, launchers, tests): **0 hits for seeds 31, 32, 33, 34 (and 35)**. The
only mentions of 31–34 anywhere are the V7 handoff and this document, as FUTURE
candidates — explicitly not contamination. Every Lichtheim3 model-training
realization on record uses seeds 19–22 (official lineage, SETTLE, CHIGH,
CANNEAL, rescue, base123, V5/V6) or seed 22 (capacity probes).

**Cluster-side proof** is mandatory and enforced before any optimizer step:
`fresh_ceiling_v7_submit.sh` (login node) and the launcher's first-launch
preflight scan the official runs root `lichtheim3_runs`, the V6 root
`l3_prospective_v6_runs`, the `$WORK` trees and `sacct` job names since
2026-06-01 for the same seed tokens and **refuse to train** on any hit. A
refusal is not a result; it triggers the replacement rule below, and this
document is amended (a new commit) before any training.

**Replacement rule (deterministic, predeclared).** If a candidate is
contaminated (already used for a relevant Lichtheim3 model-training
realization whose outcome exists or was inspected), scan upward from seed 35
and take the first seed passing the same clean/unseen proof that is not
already in the cohort; it inherits the contaminated candidate's slot. Local
evidence already shows 35 clean.

## 3. Horizon and cadence — frozen

```
maximum horizon  = u3900   (step 10,834,200)
detector cadence = every 5u = 13,890 global steps
detector points  = 780 (+ the step-0 start evaluation)
```

Verified against the executable (`tests/test_fresh_ceiling_v7.py`):
`per_epoch = ceil(29,571/64) = 463`, macro-cycle 6 → **2,778 steps per u**;
u3900 → 10,834,200; 5u → 13,890; the seed-19 witness step 10,625,850 = u3825
and the seed-20 witness step 8,445,120 = u3040 lie on this grid.

Rationale: the seed-19 FIRST-HIT was at u3825, so u3600 would censor a known
valid manifestation; the V6 seed-19 detector prospectively validated the whole
u3600→u3900 window, so u3900 is the natural fixed endpoint; u3825 would put the
witness on the boundary; u4000 would add an unvalidated 100u. **No extension
beyond u3900 for this cohort, whatever the results.**

## 4. Cohort outcome contract (verbatim)

**A. ≥1 fresh final exact witness.** Report k/4; fresh step-0 prospective
existence is established for those seeds; archive every successful witness; do
NOT automatically open Acquisition optimization. If more than one seed
succeeds, the canonical fresh model is the successful seed in the highest
preregistered slot (P1 > P2 > P3 > P4) — never by lower C before repair,
earlier FIRST-HIT, better LTM, prettier curve, fewer Arm-A iterations, or any
post-result metric. All successful seeds remain replication witnesses.

**B. 0/4 final exact witnesses but ≥1 R/N-eligible FIRST-HIT.** Not an
R/N-acquisition failure; audit the FROZEN terminal pipeline only; do NOT open
AutoResearch Acquisition automatically. FIRST-HIT means FIRST-HIT: for a seed
the first eligible checkpoint is the sole source; never reselect a
later / lower-C / better-LTM eligible checkpoint after seeing the repair
outcome; never turn a terminal-pipeline failure into a source-selection search.

**C. 0/4 seeds produce any R/N-eligible FIRST-HIT by u3900.** Evidence for a
fresh R/N-acquisition bottleneck; only then may a bounded R-focused
AutoResearch Acquisition phase be proposed. Even then forbidden: C tuning,
architecture changes, hidden-size sweeps, broad LR sweeps, broad ratio sweeps,
automatic resurrection of old Stage-A branches.

The cohort is classified only after all four seeds reach their preregistered
terminal outcome. No policy change after an individual seed.

## 5. Eligibility, FIRST-HIT, success battery (exact)

```
eligible  iff  R canonical errors = 0/29,571
           AND R genuine free-AR errors = 0/29,571
           AND Naming greedy free-AR errors = 0/29,571      (C excluded)
```
FIRST-HIT = the first eligible detector point in chronological order
(`first_hit_selector.py`, byte-identical to V6). The checkpoint is frozen
**immediately**: an in-allocation watcher stops the driver as soon as that
checkpoint is completely written (sha256 recorded in
`post/FIRST_HIT_FROZEN.json`); no later point is ever evaluated or selectable;
the frozen selector must then return the same step and sha, else the task
aborts. Then, unchanged: extract → source battery → fixed Arm A → official
battery.

```
final exact witness  iff  R canonical       = 0 / 29,571
                      AND R genuine free-AR = 0 / 29,571
                      AND Naming            = 0 / 29,571
                      AND C strict top-1    = 0 / 27,981   (full 29,571 bank)
```
No top-k fallback. Detector C is never a selection signal.

## 6. Executable recipe (audited from code, checkpoints and provenance)

Architecture: `wm_hidden 128`, `ltm_enc_hidden 512`, `ltm_dec_hidden 512`,
`phon_embed 64`, premotor 128, `unigru_last_hidden`; gate alpha 2.0 /
threshold 0.7 / usage prior 0.5; interference and ventral noise 0.0.

Populations and hashes (asserted by the driver in `final_full` mode):
R 29,571; N 29,571 (`78e46871d86efaaf34e9ef41891eb22afa93cc58467e4262225b79e8caffd50f`);
C 27,981 canonical (`10c2f06eda769bf620ca3dbb9889204e4431cac2bfe0d0f5dd37fa4df2bb9f50`);
retrieval bank 29,571 (never shrunk); dorsal pool 4,000; lexicon
`ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66`;
GloVe 6B-300d `91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed`,
`glove_found 29571`, `glove_fallback 0` (driver refuses fallback without
`--allow-glove-fallback`, which is never passed; verified in the V6 log header
and asserted by the checkpoint gate).

Losses: R step = `total_loss` (rep 1.0, align 1.0, dec 0.5, wm 0.5, gate 0.05,
label smoothing 0) + 0.5 × dorsal-pool CE in the same backward; N step =
1.0 × naming CE, teacher forcing 1.0, gold-prefix decoder input; C step =
0.087 × full-bank cosine retrieval CE, tau 0.10, `c_align_weight 0`.

Optimizer: ONE shared `torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-5)`
— betas (0.9, 0.999) and eps 1e-8 are the PyTorch defaults (no override in
code); grad clip 1.0 per task step; batch 64; no accumulation; no autocast /
GradScaler anywhere in the path; `torch_deterministic False` (as every
official run — V7 realizations are new draws, not bit-exact replays).

Schedule: `interleaved_123`; one macro-cycle = six SEPARATE optimizer steps
(1 R, 2 N, 3 C) in a deterministic per-cycle shuffle
`macro_cycle_n(ratio, schedule_seed, cycle)`; pool rides each R step; R
stream log-frequency weighted with replacement; N / C / pool counter-addressed
permutations. Seeds: streams `seed·1,000,003 + {0,1,2,3}`, schedule
`seed·1,000,003 + 4` (for seed 31: 31,000,093..96 and 31,000,097), re-derived on
resume and verified against the checkpoint.

## 7. Complete LR / phase-transition / re-anchor ledger — the retained recipe

Reconstructed from checkpoint-level `phase_transitions` of the seed-19 and
seed-20 witness lineage (archived provenance of `final_rep_rescue123`,
`final_canneal_ctrl`, `final_chigh_ctrl_u3000`, `final_settle_ctrl` and the V6
`prospective_rn_*` runs), the launchers, and the driver's `load_state_dict`.
Both witnesses carry the **identical** six-entry ledger; the V6 continuation
added nothing (seeds 19/22 pure continuation; seed 20 branched at u3000 with
the same re-anchor the SETTLE control had). `tests/test_fresh_ceiling_v7.py`
asserts the V7 ledger equals these archived entries one by one.

| event | u | global step | LR old → new | optimizer | anchor old → new | task order after | evidence |
|---|---:|---:|---|---|---|---|---|
| two-stage drop | 100 | R-cursor 46,300 (≈ step 277,800) | 1e-3 → 1e-4 (in-process, clock = completed R batches) | moments kept | 0 | unchanged | `lr_for_step`; base123 provenance |
| LR transition | 750 | 2,083,500 | two_stage → task_specific 3e-5/3e-5/3e-5 | `moment_initialization: unchanged` | 0 → 0 | unchanged | lrpilot_u850 provenance, commit 8307c8b |
| LR transition | 850 | 2,361,300 | C 3e-5 → 1e-4 (R, N 3e-5) | unchanged | 0 → 0 | unchanged | lrpilot1200_chigh provenance, ffa3659 |
| **re-anchor** | 1200 | 3,333,600 | unchanged | unchanged | 0 → 3,333,600 | **changes** (cycle index restarts at 0) | rep_rescue123 provenance, 9eeccea |
| **re-anchor** | 1400 | 3,889,200 | unchanged | unchanged | 3,333,600 → 3,889,200 | **changes** | canneal_ctrl provenance, 6b10262 |
| **re-anchor** | 2000 | 5,556,000 | unchanged | unchanged | 3,889,200 → 5,556,000 | **changes** | chigh_ctrl provenance, 82b4857 |
| **re-anchor** | 3000 | 8,334,000 | unchanged | unchanged | 5,556,000 → 8,334,000 | **changes** | settle_ctrl provenance, 96468c6 |
| (none) | 3600 | 10,000,800 | — | — | 8,334,000 kept | unchanged | V6 tasks 0/1: no flags; task 2 re-anchored at 8,334,000 |

Every event kept `dec_weight 0.5`, `shared_adamw`, ratio [1,2,3]. A re-anchor
is trajectory-relevant (verified: for seed 31 the cycle-0 order differs from
the cycle-555,600 order), so the fresh run reproduces all four at the same
global steps, and none elsewhere. No ledger event is deleted or invented.

**Implementation (no driver change).** The driver only declares transitions on
resume, so a fresh run is executed as **seven legs** of one run directory:

| leg | steps | flags at the boundary launch |
|---|---|---|
| L1 | 0 → 2,083,500 | from scratch, `--eval-at-start` (u0 full evaluation, as base123 did) |
| L2 | 2,083,500 → 2,361,300 | `--lr-* 3e-5 3e-5 3e-5 --phase-transition` |
| L3 | 2,361,300 → 3,333,600 | `--lr-* 3e-5 3e-5 1e-4 --phase-transition` |
| L4 | 3,333,600 → 3,889,200 | `--lr-* 3e-5 3e-5 1e-4 --reanchor-schedule --phase-transition` |
| L5 | 3,889,200 → 5,556,000 | same |
| L6 | 5,556,000 → 8,334,000 | same |
| L7 | 8,334,000 → 10,834,200 | same |

Every boundary is on the detector grid and a `save_every` multiple; a mid-leg
requeue passes the leg's LR flags only (never `--reanchor-schedule`, never
`--phase-transition`). The gate `verify_checkpoint` refuses any checkpoint whose
`lr_policy`, `schedule_anchor_step`, transition ledger, seed, widths, schedule,
optimizer policy, populations, hashes or cursors deviate from this table at
its step (validated PASS on three archived H512 checkpoints; FAIL on a wrong
seed and on a synthetic mid-leg re-anchor). Historical legs lived in separate
branch directories; V7 keeps one directory — a bookkeeping difference only
(`config_from_step_*.json` / `provenance_from_step_*.json` per launch).

The historical u0→u500 driver blob `e6c48ff8…` and u500→u1200 blob `24742adf…`
differ from the current `95295d63…` only in stopping/evaluation control and
the `--reanchor-schedule` flag (diffs inspected; the 1:2:3 training update is
unchanged and tested bit-identical in `tests/test_base123_continue750.py`).
Ceiling stop: `--stop-at-ceiling --ceiling-consecutive-required 2` (SETTLE
Amendment 2 / V6 semantics; the four-way 0/0/0/0 stop has never fired and can
only fire on a natural exact witness).

## 8. Byte-identity / lineage gates — PASS

* `2e559e5 → 78f5505` touched only V6 tooling: the V6 preregistration, the V6
  launcher, `first_hit_selector.py` (new), `noninterference_smoke.py` (new),
  and `ceiling_source_completion.py` (+30/−3, `cmd_extract` sampled-determinism
  assertion only). `models/`, `losses.py`, `utils/`, `evaluate/`, `config.py`,
  `train_joint_scratch.py`, `train_tasks.py` are byte-identical.
* Scientific dependency closure `96468c6` (official SETTLE executable) →
  `73a968e` (V7): `git diff --stat` over `models/`, `losses.py`, `utils/`,
  `evaluate/`, `config.py`, `data/lexicon.py`, `data/phonemes.py`,
  `data/dataset.py`, `train_multitask.py`, the driver and every evaluator file
  is **empty** (the driver's full import closure).
* Arm A: `gradient_training_probe.py` blob `548b61d2ef50b32129faff6a5bdaa5c6f1f1bfb5`
  at `3912d758` == at `78f5505` == at `73a968e`; full sha256
  `a7064708878365fc2875c3db6f903b3e370b16023bab12ce2be23c0f1ea5dcbf` == the
  V6 completion archive copy. All frozen constants imported, none redefined.
* V6 fix scope: commit `78f5505` = 1 file, `ceiling_source_completion.py`,
  `cmd_extract` only; `gradient_training_probe.py` untouched.

Frozen blobs at the executable commit `73a968e` (sha256 pinned in the launcher):

| file | git blob | sha256 |
|---|---|---|
| scripts/naming_comprehension/train_joint_scratch.py | 95295d63560ae4c235a6beee8dfb47166f4ed30d | 766244c1363d998f34fe3e4f725b57a9b3f95cf716876d71470d804cf962765c |
| scripts/naming_comprehension/train_tasks.py | 0b3a1f19addcc2351edf75248ba31333ccd7cf59 | a867b6b5ec474fb79fcd18e769fa8d16d8bdc3de920f7704bfa9be2e7b907303 |
| losses.py | 67ddf64f37d5e3210b79e9f4591f5ab80394c4a3 | 382eed2335ee2add7a00f6de9b73a788cdef5e94155ae1970e38bd5d7dcefd80 |
| scripts/naming_comprehension/frozen_probe.py | 8e54f361b7cd13647f31434d0477f391c38da547 | d4c7bb9289cef56d09d05b1cbcbd9b47f0fa32dda549b333d29d8a4753689d4f |
| scripts/naming_comprehension/base123_error_audit.py | 9884be44b7f86e2b3c125f5a8f1a61c9cda3a517 | 15008114d5e5ff3bde51b550193c07e62d5566347a29510ec9a75bb8ae910b37 |
| scripts/naming_comprehension/ceiling_source_completion.py | 4e6fa297cba53871158109feb50e05b43371f5e6 | fc1a0865bba0810ba03a3a01a54feb234dd6cca7c341a94527c093f41f18088f |
| scripts/naming_comprehension/gradient_training_probe.py | 548b61d2ef50b32129faff6a5bdaa5c6f1f1bfb5 | a7064708878365fc2875c3db6f903b3e370b16023bab12ce2be23c0f1ea5dcbf |
| scripts/naming_comprehension/constrained_coexistence_v3.py | f0434068b420d0fb7a0309443a8c4e1016ec3548 | 1c69a59c525f11f54946fe7fd42412be601b4b451a55706925ed516b4e495419 |
| scripts/naming_comprehension/frozen_head_probe.py | 72b5f46d8a007adde37a4915912c1e9ed949a21b | f2a30620aad79bba81bbd9e56ab0b1a38a74801c05e35b0f9e74f7bb63fb851b |
| scripts/naming_comprehension/coexistence_probe.py | 513d7ed1c8acadf8b31937210a3d8a5658b3c1de | de715e3c573c44d00c0fd8bb56fef1e916c79b4f1e58af3965cb3f48d828ecef |
| scripts/naming_comprehension/first_hit_selector.py | af10b546f316b58b073442a8286a90ba6198732b | 2482adb2ab89ced1249303f8e898a00e2a6927e40d22e1cc5f57d55db50a6c5c |
| data/lexicon_en_glove_covered.tsv | 5aaab2af7b6dfaabf8dffc241f7ab41324b6cb7b | ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66 |
| scripts/naming_comprehension/fresh_ceiling_v7.py (new, orchestration) | 8f8e49e0446061d3468fd604c2b1c7fcd02a851a | fd67cd7b51e75d23aeb30dea494b645e890df462544c092be68050ae969bfe28 |
| scripts/cluster/jeanzay/fresh_ceiling_v7.slurm (new) | 8942c3d2224c373f6cd552ebc8f55e39679b4b29 | 6933c5239bd26b33516cc5cdbf900782d565cd4855f197c6d93d228efbc336e5 |

## 9. Detector non-interference — PASS (existing evidence)

`evaluate()` runs under `preserved_rng()` and `torch.no_grad()`; the V6
non-interference gate (`noninterference_smoke.py`, two independent processes,
A = K steps vs B = full detector evaluation + K steps) was **bitwise equal** on
model parameters, optimizer moments, Generator states, global RNG, cursors,
`rep_epoch`, `global_step`, and the evaluation alone perturbed nothing
(V6 archive `PROVENANCE.txt`). The V7 driver blob is identical, so the result
transfers. The watcher only reads files and signals the process after the
checkpoint is written; it touches no training state.

## 10. Restart / requeue contract

`load_state_dict` restores model, the one shared AdamW and all moments, LR
policy, `phase_transitions`, `global_step`, R/N/C/pool cursors, schedule
anchor, ceiling streak, and all RNG states; LR is re-derived per step. The
detector grid is the full fixed list on every launch; the loop evaluates only
steps it reaches after `train_step()`, so a resume on a milestone re-evaluates
nothing and skips nothing; every detector point saves its own checkpoint. A
row appended before an interrupted save is superseded by the re-realized row
(`dedupe_metrics`, last row per step — that is the row describing the
checkpoint on disk; source `metrics.tsv` untouched). FIRST-HIT status is a
file (`post/FIRST_HIT_FROZEN.json`); any later launch refuses to train once
it exists. Legs and flags are derived from the newest own checkpoint by
`plan()`; a scheduler interruption cannot change the horizon, reset training,
restart the 5u grid, re-anchor mid-leg, or replace a recorded FIRST-HIT.
Validated by the real-driver smoke (six launches incl. a simulated mid-leg
kill; ledger checked entry by entry). Allocation walltime is 20 h; the same
idempotent script is chained (`afterany`, 4 elements) so no manual action is
required; `--requeue` covers preemption / node failure.

## 11. Compute / storage — planning maximum

Measured on V6 job 2051441 (V100): 0.0114 s/step training, ~46 s per detector
point. Per no-hit seed: 10,834,200 × 0.0114 ≈ 34.3 h + 780 × 46 s ≈ 10.0 h ≈
**44 GPU-h** (planning cap **45**); cohort worst case **≈ 177 GPU-h** (cap
**180**); 3 allocations of 20 h per seed. A seed that hits stops at once.
Storage: 781 checkpoints × ~30 MB ≈ 23 GB per seed, ≈ 95 GB worst case;
the submit script prints SCRATCH free space and the launcher refuses < 30 GB
per seed. No horizon change if throughput differs.

## 12. Run IDs, namespaces, archive policy

Run root `$SCRATCH/l3_fresh_v7_runs` (never `lichtheim3_runs` /
`l3_prospective_v6_runs`, refused by the launcher); code root
`$WORK/l3_fresh_v7/repo` at `73a968e` (detached, clean, sha256-verified);
logs `$WORK/l3_fresh_v7/logs`. Per seed: `checkpoints/step_*.pt`,
`metrics.tsv`, `post/` (`FIRST_HIT_FROZEN.json`, `metrics_dedup.tsv`,
`first_hit_decision.json`, `extract_validation.json`, `source_battery.json`,
`seed<S>_u<U>_A/`, `repaired_battery.json`, `RESULT_SUMMARY.json`,
`CHECKPOINT_SHA256SUMS`). Statuses: `FRESH_STEP0_PROSPECTIVE_EXACT_WITNESS`,
`REPAIRED_BUT_NOT_FOUR_WAY_EXACT`, `REPAIR_NO_C0`, `NO_ELIGIBLE_HIT_BY_U3900`.
Every FIRST-HIT source checkpoint, its post directory and every witness are
archived under `archives/fresh_ceiling_v7_<date>/` with a manifest and
SHA256SUMS, verified before and after locking, and copied to STORE with
checksums. Checkpoint provenance is never weakened to save space.

## 13. Explicit forbidden modifications

No change to: Arm-A objective / constraints / preconditioner / γ / repair LR
/ stopping / MAX_ITERS; strict C evaluator; source-selection rule; horizon;
cadence; architecture; widths; LR ledger; optimizer; loss weights; task ratio;
populations; lexicon; GloVe; seeds or priority; `torch_deterministic`. No
second chance at source selection, no reselection, no continuation beyond
u3900, no AutoResearch Acquisition during the cohort. **C is closed.**

## 14. Before execution

```
QUESTION             Can the frozen detector → FIRST-HIT → Arm-A policy reproduce from fresh step 0?
DECISION CONSEQUENCE A/B/C cohort contract (section 4).
STOP RULE            Per seed: first Rcan=Rfree=N=0 checkpoint is the sole source; fixed terminal
                     pipeline; stop that seed. Otherwise stop at u3900. Cohort: classify only after
                     all four seeds reach their preregistered terminal outcome.
MAX COMPUTE          4 × u3900, ≈180 V100 GPU-h worst case; no post-result extension.
```
