# C_ALIGN HISTORICAL AUDIT

Workstream: LICHTHEIM3 — VENTRAL INTERFACE / C-ALIGN CAUSAL PILOT — DESIGN AND PREREGISTRATION
Date: 2026-09-17. Mode: read-only archaeology (git, filesystem, checkpoint metadata, docs, local transcripts). Nothing modified.

**C_ALIGN_HISTORY_STATUS=EXACT_PRIOR_TEST_NOT_DECISIVE**

## 1. Question

Has comprehension-stream (C-step) semantic alignment — `c_align_weight · alignment_loss(ŝ, GloVe target)` added to the C objective — already been tested as an intervention, in a way that decides whether it repairs the native ŝ→decoder interface in mature interleaved-123 H512 J0 states (V6 seed19 u3825 / seed20 u3040)?

"Exact prior test" criteria: (1) nonzero C-stream alignment term actually executed; (2) same loss (`(1−cos)+0.1·MSE`); (3) same target (GloVe row of the item); (4) gradient reaches the comprehension/ventral encoder and `to_semantic`; (5) a matched control without the term; (6) regime transferable to the mature interleaved H512 question (schedule, width, maturity, seeds, readouts).

## 2. Search log

* **Git** (shared git dir, all refs, 200 commits): `git log --all -S c_align_weight` (31 commits), `-G c_align` (33), `--grep=FINAL-2` (3), `-i --grep=align` (26); value search for nonzero `c_align_weight` / `C_ALIGN=` → only `03ae097` (slurm + tests), test additions in `a93c88a`, `9969e95`, `45eab2e`, and this workstream's contract `8427d25`. Full bodies read for `03ae097`, `a93c88a`, `73982f3`, `9969e95`, `45eab2e`, `855eca6`, `0b75869`; `git blame` on the driver; driver inspected at `581406f`, `8acd1b7`, `bdeaa1a` (no C alignment before FINAL-2A).
* **Filesystem:** grep for `final2a|c_align|c-align|calign` across the project tree (1,066 files). Outside tests, all values 0.0.
* **Checkpoints:** metadata of all 1,082 local `.pt` files loaded read-only. 223 carry `c_align_weight`, all 0.0; no local checkpoint with a nonzero value; no `final2a*` artefact anywhere locally. Phase-2 `objective` c0 (12) and c3 (6) found.
* **Docs:** `stage_source_of_truth/` (registries, phase SoTs, forensic audits), `CHAT_ARCHAEOLOGY/`, `lichtheim3-autoresearch` docs + `registry/events.jsonl`, `paper_programme/**`, `~/Downloads/LICHTHEIM3_*.md`.
* **Transcripts (read-only):** session `3a8ddf1c` (2026-09-02/03), grep of other sessions for `final2a`.
* **Inaccessible:** Jean-Zay `$SCRATCH`/`$STORE` (FINAL-2A primaries); Gmail/Drive connectors (unauthenticated).
* **Direct verification by the main session (not only the search agent):** `git show 03ae097` body and `final2a_run.slurm` (`C_ALIGN=1.0`, `SEED=22`); `CHAT_TIMELINE.md:262-268`; `CHAT_TO_EXPERIMENT_MAP.tsv` row E034a; `PHASE04_SOURCE_OF_TRUTH.md:138-144,174-183`; `PHASE06_SOURCE_OF_TRUTH.md:105`.

## 3. Candidates

| ID | Candidate | Classification |
|---|---|---|
| C1 | FINAL-2A `final2a_calign_seed22_final_full` | **EXACT_PRIOR_TEST** (not decisive) |
| C2 | Phase-2 single-task C0 / C3 warm-start (H128) | RELATED_BUT_NOT_EQUIVALENT |
| C3 | Phase-3A/3B/3C interleaved multitask warm-start (H128, subset) | RELATED_BUT_NOT_EQUIVALENT |
| C4 | Phase-4 factorial, FINAL-1, FINAL-3P…9E, base-123, V-series (incl. V6) | NOT_RELEVANT |
| C5 | `grad_interference_audit.py`, `audit_update_geometry.py` | NOT_RELEVANT (measurement only) |
| C6 | This workstream's frozen gradient diagnostic contract | NOT_RELEVANT (not a training test) |

### C1 — FINAL-2A — EXACT_PRIOR_TEST, NOT DECISIVE

* **Identity.** Commit `03ae097ef144f8dba5e492a2ad01a3e6b53635c1` (2026-09-02 17:50 +0200) "FINAL-2A: comprehension-stream semantic alignment (--c-align-weight, default 0.0)"; `scripts/cluster/jeanzay/final2a_run.slurm` with `C_ALIGN=1.0  # <-- the single intervention`, `SEED=22`; Jean-Zay job 1678465 `l3_final2a`, COMPLETED (autoresearch forensic sacct record). Exact launch commit (`03ae097` vs `a93c88a`) not recovered.
* **Stated question (commit body):** "does giving the comprehension stream its own direct semantic-target alignment, in addition to lexical retrieval CE, fix the full-scale ventral semantic optimization problem observed in FINAL-1?"
* **Objective.** C contribution `0.087·retrieval_CE → 0.087·retrieval_CE + 1.0·alignment_loss(ŝ, raw GloVe)` (not scaled by λ_C). Gradient reaches `phon_embed`, `ltm.encoder`, `to_semantic` only (`tests/test_c_align.py`). Bitwise identity at 0.0 vs pre-change `bdeaa1a` recorded in the commit body.
* **Regime.** **Summed** J0 step (interleaving did not yet exist), from scratch, seed 22, H128 widths (inferred from the paired FINAL-1 config and the slurm "only delta is C_ALIGN" contract), LR 1e-3→1e-4 at 46,300, 324,100 steps (700 N-exposures), global clip 1.0 (3,241/3,241 logged steps clipped). Matched control: FINAL-1 (same seed/batches, `c_align_weight=0`).
* **Results** (user-pasted into session `3a8ddf1c`, 2026-09-02 21:28; no local primaries):

| N-exposures | FULL 2A / F1 | LTM 2A / F1 | C top-1 2A / F1 | Naming 2A / F1 |
|---|---|---|---|---|
| 300 | 1.000 / .99997 | .8143 / .86318 | .04310 / .04818 | .04873 / .04880 |
| 500 | 1.000 / 1.000 | .8021 / .85330 | .045995 / .05271 | .05553 / .05806 |
| 700 | .999932 / 1.000 | .79084 / .83602 | .048319 / .05593 | .06050 / .06182 |

* **Closure then:** "consistently worse than FINAL-1 for C and LTM … Return to c_align_weight=0"; "H4 falsified: direct semantic-target alignment on the C stream does not fix the problem" (`CHAT_TIMELINE.md:264-265`; `CHAT_TO_EXPERIMENT_MAP.tsv` E034a "dead end"; `HYPOTHESIS_EVOLUTION.md:78`; `final_baseline123_audit.md:79`; commit `73982f3` "c_align_weight 0 (FINAL-2A closed)"). The canonical SoT does **not** register it: `PHASE06_SOURCE_OF_TRUTH.md:105` "individual FINAL-2…9 headline outcomes are not independently recoverable".
* **Criteria:** (1)–(5) met. **(6) fails**: summed (not interleaved per-task-clipped) schedule; H128 (not H512); early from-scratch (≤700 N-exposures), not mature V6 states; weight 1.0 (vs the retrieval term at 0.087 — alignment-dominated); single seed; no local primaries; no item-level native ŝ→decoder analysis; outcomes unrecomputable.
* **What it does establish (as chat-level evidence):** in that regime, C-stream alignment at 1.0 did not rescue and was associated with worse isolated LTM repetition (≈ −4.5 pp at 700) and worse C top-1. This is a relevant prior against large weights and is carried into the pilot contract as a harm expectation, not as a decision.

### C2 — Phase-2 single-task C0 / C3 — RELATED_BUT_NOT_EQUIVALENT

Warm-start from mature H128 repetition checkpoint seed22/e140; comprehension-only (no R rehearsal). C0 = alignment (w 1.0) on raw GloVe; C3 = C0 + 0.087·retrieval. `PHASE04_SOURCE_OF_TRUTH.md:138-144,174-183`: isolated LTM repetition .989449 → .007846 (C0) / .014068 (C3); cosine .2967→.3476 while top-1 2.58%→2.44% ("alignment ≠ identification"). Not exact: single-task, no rehearsal, no alignment-off control within an interleaved regime. Relevance: unrehearsed alignment destroys ventral repetition; improved cosine need not imply identification.

### C3 — Phase-3A/3B/3C multitask — RELATED_BUT_NOT_EQUIVALENT

`train_multitask.py`, warm-start H128 seed22/e140, 1:1:1 or 1:2:3 per-task steps, C step always C3 (alignment w 1.0 + 0.087 retrieval), R = pure LTM sequence CE, dorsal frozen, N/C subset 3,288 (3C: R full 29,571 rehearsal → LTM .971459, C .9155). Alignment was present in every arm and never toggled → no causal isolation.

### C4 — NOT_RELEVANT

All from-scratch factorial / FINAL-1 / FINAL-3P…9E / base-123 / V-series runs used C = 0.087·retrieval only; every checkpoint carrying the key has 0.0 (incl. V6 seeds 19/20).

### C5 — NOT_RELEVANT

`grad_interference_audit.py` on FINAL-1: C_align vs C_retrieval gradient cosine +0.31…+0.48 (measurement). `audit_update_geometry.py` asserts `c_align_weight == 0.0` (diagnostic only).

### C6 — NOT_RELEVANT

`8427d25` freezes this workstream's read-only gradient diagnostic; no training.

## 4. Code provenance

* Introduced: `03ae097` (FINAL-2A) — constructor validation, summed-step term, resume guard, CLI `--c-align-weight` (default 0.0), 14 tests.
* Interleaved C-step branch: `73982f3` (FINAL-3P) — `train_joint_scratch.py:1297-1302`.
* Resume guard: `load_state_dict` raises if checkpoint `c_align_weight` ≠ configured; per `45eab2e` `--phase-transition` does **not** relax it ("lambda_C, c_align still refuse unconditionally").
* Later status: AutoResearch "POTENTIALLY_SEARCHABLE_LATER", retained at 0.0; V6 supervision audit: "SUPPORTED_BY_CODE, NOT ACTIVE".

## 5. Conclusion

**C_ALIGN_HISTORY_STATUS=EXACT_PRIOR_TEST_NOT_DECISIVE**

An exact prior test (FINAL-2A) exists and was negative in its regime, but it does not decide the current question (mature interleaved-123 H512 V6 states, warm-start, small weight, per-task clipping, native ŝ→decoder readouts). A pilot is therefore not redundant, provided it (a) cites FINAL-2A as a harm prior, (b) uses a weight not dominated by alignment, and (c) freezes preservation stop criteria.

## 6. Unresolved

1. FINAL-2A primaries (metrics.tsv, losses.tsv, checkpoints, `.out`) only on Jean-Zay, if retained; numbers are user-pasted chat evidence, not recomputed.
2. Exact launch commit unknown.
3. Single seed; no item-level overlap vs FINAL-1.
4. Transfer of FINAL-2A's harm to interleaved/H512/mature states: NOT_ESTABLISHED.
