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
| amended contract sha256 | `6edf8f8eaab3171ea2ab1570127ad91cc809f396fbdba437d10e7fb16f11683f` |
| implementation digest | `a619669da78aad25999f4043fa1d7389dd143c9e703704fa5853488961422d6a` |
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
* cost estimate from measured local throughput (the read-only gradient diagnostic ran 876 real-state batches, ≈8 backward-equivalents each, in ≈19 min): a macro-cycle of 1 R + 2 N + 3 C steps costs ≈1.5–2 s, so 23,150 macro-cycles ≈ 10–13 h per arm, ≈2 days for four arms, **plus** 44 post-hoc checkpoint evaluations (see §4 for the measured per-checkpoint cost) — on the order of a week of continuous laptop compute, with no determinism guarantee.

## 4. What passed

See `preflight/preflight_report.json` for the full machine-readable evidence.

<!--GATE_TABLE-->

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
