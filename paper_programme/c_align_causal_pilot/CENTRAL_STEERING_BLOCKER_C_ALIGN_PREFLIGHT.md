# CENTRAL STEERING — C-ALIGN PILOT PREFLIGHT BLOCKER

PREFLIGHT_STATUS=FAILED_HARD_STOP
FAILED_GATE=V5_DETERMINISM (training environment unavailable from this session)
TRAINING_RUN=NO · OPTIMIZER_STEP_COUNT=0 · RETURN_TO_CENTRAL_BLOCKER=YES

Everything CENTRAL authorized up to the first optimizer step is built, frozen, committed and passing. The pilot did not start because the authoritative training environment cannot be reached or substituted from this session without a CENTRAL decision.

## 1. Lineage

| item | value |
|---|---|
| design commit (CENTRAL-accepted) | `9f36f2e0cf913ffe5f453a82ca43c20fa4ec824d` |
| contract amendment commit | `24f70cfaab7ca7b6e3be7d2c76e02ac83e7a8ee5` |
| implementation freeze commit | `d937dda1a3c579048815fde2298d1cb4317f4330` |
| gate-check fix + re-freeze | `eda94861750b7a2344efc702d092d5df85bfd75f` |
| amended contract sha256 | `6edf8f8eaab3171ea2ab1570127ad91cc809f396fbdba437d10e7fb16f11683f` |
| implementation digest | `5c6ee74d921204aa6c4f81a49df94e042914be195c16a8115c55eb97d521db5d` |
| worktree / branch | `wt-c-align-pilot` / `paper-programme/c-align-causal-pilot-execution` |

## 2. The failed gate

**V5 — strict deterministic execution on the ACTUAL training environment.**

The frozen launch configs pin `device=cuda`, GPU class V100, Jean-Zay `gpu_p13`, module `pytorch-gpu/py3/2.6.0`, `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` (contract §9, CENTRAL §18).

Evidence gathered in this session:

| probe | result |
|---|---|
| `ssh jz` (jean-zay.idris.fr, ProxyJump flores) | `Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password)` |
| `ssh-add -l` | `The agent has no identities` |
| `ssh flores` | authenticates; from there `ssh uss35bp@jean-zay.idris.fr` also returns `Permission denied (publickey,…)` |
| local machine | Darwin arm64 (Apple silicon), `torch.cuda.is_available() == False`, MPS only |
| `determinism_probe("cuda")` | FAIL — "the configured training device is CUDA but no CUDA device is visible in this environment" |

No credentials were requested and none were used. This session cannot authenticate to IDRIS; that is **not** proof that Jean-Zay is unavailable to the account holder.

CENTRAL §18 is explicit: *"If strict deterministic execution cannot be enabled: HARD STOP BEFORE OPTIMIZER STEP 1. Return blocker to CENTRAL. Do not silently use nondeterministic mode."* That is what happened.

## 3. Why this was not resolved unilaterally

CENTRAL §18 permits deviation only if "live infrastructure proves that this exact historical environment is unavailable". What is proven here is narrower: it is unreachable **from this session**. Substituting a different machine would change GPU class, CUDA/cuDNN and torch version relative to V6 — a scientific parameter of the frozen contract (§9), and therefore a design change reserved to CENTRAL (return condition C).

Candidate substitute found (not used, nothing installed, no job submitted):

* `oberon` (cognitive-ml.fr), Slurm, partitions `gpu-p1` (10 × A40) and `gpu-p2` (4 × H100), both idle; `cuda/12.x` modules available; no torch environment present yet.

A CPU fallback on this laptop was also considered and rejected as non-viable and non-compliant:

* it cannot satisfy the pinned CUDA determinism requirement at all;
* cost estimate from measured local throughput (the read-only gradient diagnostic ran 876 real-state batches, ≈8 backward-equivalents each, in ≈19 min): a macro-cycle of 1 R + 2 N + 3 C steps costs ≈1.5–2 s, so 23,150 macro-cycles ≈ 10–13 h per arm, ≈2 days for four arms, **plus** 44 post-hoc checkpoint evaluations (the V7 baseline battery measured ≈40 min per state on this machine, so 44 checkpoint evaluations ≈ 30 h) — on the order of a week of continuous laptop compute, with no determinism guarantee.

## 4. What passed

See `preflight/preflight_report.json` for the full machine-readable evidence.

| gate | what | result | evidence |
|---|---|---|---|
| V1 | driver amendment | PASS | declaration permits exactly 0.0 → 0.1; all 11 other resume guards present in code; the exception is not tied to `--phase-transition`; the constructor refuses any other target weight |
| V2 | toy equivalence | PASS | `24 passed` — incl. amended driver at weight 0 bitwise identical to the pre-amendment driver over 12 toy steps (parameters, optimizer tensors, Adam step counters, cursors, task order, batch order), and at 0.1 identical until the first C step |
| V3 | start identity | PASS | all four arms: SOURCE file, parameter and optimizer-state sha256 equal the frozen values; single shared AdamW; per-parameter Adam step present; RNG states present (torch/numpy/python/cuda) |
| V4 | recipe identity | PASS | j0, interleaved_123, final_full, LR 3e-5/3e-5/1e-4, dec 0.5, widths 128/512/512, shared_adamw, C population 27,981 (`10c2f06e…`), checkpoint c_align 0.0; no stop-at-ceiling, no re-anchor; declaration set iff weight 0.1 |
| V5 | determinism | **FAIL** | configured device `cuda`; `torch.cuda.is_available() == False` in this environment → HARD STOP before optimizer step 1 |
| V6 | expected task/batch sequence | PASS | both pairs: 23,150 macro-cycles, 23,150 R / 46,300 N / 69,450 C; cumulative digest W3 `a91e72714c2d7d76…`, W4 `f2e8f168f13f0cc9…`; SOURCE sha unchanged, trainer cursors untouched |
| V7 | baseline readouts | PASS | W3 S0 3199, S1c 42, C 34, WM 0, FULL 0, Naming 0 · W4 S0 3647, S1c 51, C 42, WM 1, FULL 0, Naming 0 — every value equals the frozen expectation, free-AR and canonical, and S0 equals the historical battery's isolated-LTM count |
| V8 | step/checkpoint plan | PASS | configuration: 138,900 steps, save every 13,890, exactly 10 checkpoints ending on the frozen final step (actual counts verified post-run) |
| V9 | evaluator freeze | PASS | evaluator `c_align_pilot_evaluator_v1` hashed before any pilot checkpoint exists; no pilot run namespace exists |
| V10 | closed artifacts | PASS | W3_SRC/W4_SRC sha unchanged; ventral-interface package 37/37 files verified; directional-dose package 33/33 verified |

Launcher refusals were exercised for all four arms and correctly refuse to start (determinism unavailable; and, at the time of the run, an untracked file in the tree).

## 5. What CENTRAL is asked to decide

1. **Restore the authoritative path** — the account holder runs the four frozen arms on Jean-Zay themselves, or enables key-based access for this session. Nothing in the package changes; the launcher and configs are ready as frozen.
2. **Authorize an environment substitution** (e.g. the `oberon` A40/H100 partitions), accepting explicitly that:
   * GPU class, CUDA and torch version differ from V6;
   * the OFF arms will not reproduce the historical V6 detector trajectory numerically (already declared descriptive, non-gating, in contract §16);
   * all four arms would still share one environment, one code commit and strict determinism, so the ON−OFF contrast and every frozen threshold remain valid;
   * a new environment must first pass V5 there, and the contract's §9 environment clause would need a recorded amendment.
3. **Defer the pilot**, leaving the frozen implementation as-is until an authoritative V100 environment is available.

Options 2 and 3 are scientific decisions. Option 1 is purely operational.

## 6. Confirmations

TRAINING_RUN=NO
OPTIMIZER_STEP_COUNT=0 (on any real state; toy-state updates occur only inside `tests/test_c_align_pilot.py`)
ARCHITECTURE_CHANGED=NO
GATE_CHANGED=NO
ATTRACTOR_IMPLEMENTED=NO
YAIR_FLAG_IMPLEMENTED=NO
LESIONING_RUN=NO
FULL_CEILING_RUN=NO
NO_ARM_STARTED=YES (no run namespace was created; `L3_PILOT_RUNS` is empty)
CLOSED_ARTIFACTS_UNCHANGED=YES (SOURCE checkpoints, ventral-interface and directional-dose packages re-verified by hash)
DESIGN_CHANGE_REQUIRED=ONLY_IF_CENTRAL_CHOOSES_OPTION_2

---

## READY_TO_PASTE_CENTRAL_PROMPT

```
CENTRAL STEERING — C-ALIGN CAUSAL PILOT: PREFLIGHT BLOCKER, NO TRAINING RUN

Status: PREFLIGHT_STATUS=FAILED_HARD_STOP; FAILED_GATE=V5 (training environment).
TRAINING_RUN=NO; OPTIMIZER_STEP_COUNT=0; no arm started; no run namespace created;
closed artifacts re-verified unchanged.

Commits: contract amendment 24f70cfa; implementation freeze d937dda1;
gate-check fix + re-freeze eda94861. Amended contract sha 6edf8f8e…;
implementation digest 5c6ee74d…

DONE AND FROZEN
- Contract amended per CENTRAL sections 4-6: T1..T4 renamed, T2 sublabels frozen
  prospectively, authorization state recorded. No threshold, budget, cadence,
  start state, optimizer policy, weight or readout changed.
- Minimal V1 driver amendment: --declare-c-align-transition permits exactly
  c_align_weight 0.0 -> 0.1, records one transition with pilot provenance, and
  relaxes no other resume guard (a 0.0 -> anything-else and an undeclared
  0.0 -> 0.1 both still raise; --phase-transition is not a second route).
- Per-macro-cycle task/batch digest logging implemented (gate V6).
- Frozen post-hoc evaluator + four immutable launch configs + launcher whose
  refusals were exercised.
- 24 pilot tests pass, including: the amended driver at weight 0 is BITWISE
  identical to the pre-amendment driver over toy steps (parameters, optimizer
  tensors, Adam step counters, cursors, task order, batch order).

PREFLIGHT RESULT: 9 of 10 gates PASS.
- V7 baseline, recomputed with the frozen evaluator on the real SOURCE states,
  reproduces the closed values EXACTLY: W3 S0 3199 / S1c 42 / C 34 / WM 0 /
  FULL 0 / Naming 0; W4 S0 3647 / S1c 51 / C 42 / WM 1 / FULL 0 / Naming 0
  (free-AR and canonical), and S0 equals the historical isolated-LTM battery.
- V6 expected sequences derived without training: 23,150 macro-cycles,
  23,150 R / 46,300 N / 69,450 C per arm; cumulative digests recorded for both
  pairs; SOURCE files and cursors untouched.
- V3/V4 identity and recipe pass for all four arms; V10 closed packages verified
  (37/37 and 33/33 files, both SOURCE checkpoints unchanged).

- V5 FAILS: the pinned environment (Jean-Zay V100 gpu_p13, pytorch-gpu/py3/2.6.0,
  deterministic algorithms + CUBLAS_WORKSPACE_CONFIG=:4096:8) is not reachable
  from this session. ssh to jean-zay.idris.fr is refused for every available key
  (publickey), the ssh agent holds no identities, and the local machine has no
  CUDA device. No credentials were requested or used. Per CENTRAL section 18 this
  is a HARD STOP before optimizer step 1; nondeterministic mode was NOT used.

CENTRAL IS ASKED TO DECIDE:
1. Restore the authoritative path (the account holder launches the four frozen
   arms on Jean-Zay, or key access is enabled) — nothing in the package changes.
2. OR authorize an environment substitution — a GPU cluster is reachable
   (oberon: 10x A40, 4x H100, Slurm, idle; nothing installed, no job submitted),
   accepting that GPU class / CUDA / torch differ from V6, that the OFF arms will
   not numerically reproduce the historical V6 trajectory (already non-gating),
   that all four arms would still share one environment, one commit and strict
   determinism, and that contract section 9 needs a recorded environment
   amendment plus a passing V5 there.
3. OR defer the pilot, leaving the frozen implementation untouched.
4. Confirm that the stop-loss and the interpretive boundary remain in force, and
   that no tuning, extra arm, extra weight, from-scratch run, attractor, gate
   change, lesioning or full-ceiling work is authorized in the meantime.
```
