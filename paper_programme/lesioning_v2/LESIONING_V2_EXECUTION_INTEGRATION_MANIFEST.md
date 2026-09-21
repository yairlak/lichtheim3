# LESIONING V2 — EXECUTION INTEGRATION MANIFEST

    BRANCH    paper-programme/lesioning-v2-execution-integration-repair
    BASE      dfdd2b6d023c788e2693f1dbee4cdd9a78219abc
    SCOPE     EXECUTION INTEGRATION ONLY
    DATE      2026-09-21

No scientific lesion was executed. No P1-P4 model was loaded. No nonzero
forward pass occurred on any real V7 state.

## 1. Archaeology — what was RECOVERED, not invented

Every scientific operation dispatches to an already-validated implementation.
Nothing was reimplemented.

| need | exact source |
|---|---|
| POST_REPAIR reconstruction (SOURCE ckpt + head_first_c0) | `paper_programme/v7_prelesion_validation/scripts/prelesion_eval.py::build_state`, with `load_repair_head` asserting the deployed keys are exactly `{2.weight, 2.bias}` |
| source immutability | `prelesion_eval.assert_source_unchanged`, `sha256_file` |
| model state digest | `prelesion_eval.model_state_digest`; `lesion_operator.context.parameter_digest` |
| GloVe | `scripts.naming_comprehension.ceiling_source_completion.GLOVE` (real file; no fallback embedding) |
| trainer / populations | `ceiling_source_completion.build_trainer` -> `tr.entries`, `tr.vocab`, `tr.bank_raw`, `tr.comp_idx` |
| repetition, canonical target-length | `prelesion_eval.canonical_items` -> `scripts.evaluate_train_lexicon_ceiling.evaluate_forms_ar` (the exact function `train_tasks.repetition_snapshot` uses for its primary readout) |
| repetition, genuine free-AR | `prelesion_eval.free_ar_items`, parity-tested against `train_joint_scratch.JointScratchTrainer.free_ar_repetition` (BOS start, first-EOS trim, global `FREE_AR_MAX_STEPS=12`, target length never terminates decoding) |
| naming | `scripts.naming_comprehension.train_tasks.evaluate_naming(return_per_item=True)` |
| comprehension | `train_tasks.evaluate_comprehension_subset(return_per_item=True)`, probed against the full bank |
| FULL / WM-only / LTM-only routes | the model's own `logits` / `wm_logits` / `ltm_logits`, as used by both evaluators above |
| gating diagnostics | `prelesion_eval.gating_items` (`field_confidence`, `gate`) |
| exact-match metric | the evaluators' own `exact_match` / `{route}_exact_match`; aggregation computes only `correct / n` |
| primary vs diagnostic endpoints | `lesion_operator.battery.PRIMARY` / `DIAGNOSTIC`, frozen at the implementation freeze |
| item identity and ordering | bank index rendered `bank_{i}`, matching the V7 pre-lesion pipeline exactly |

Every frozen endpoint mapped unambiguously to an existing evaluator. Nothing
was BLOCKED for want of semantics.

## 2. Change boundary

Frozen scientific files, **byte-unchanged** from `dfdd2b6`:

    lesion_operator/masks.py       lesion_operator/noise.py
    lesion_operator/context.py     lesion_operator/sites.py
    lesion_operator/seeds.py       lesion_operator/sd_procedure.py
    lesion_operator/battery.py     contract/LESIONING_V2_RUN_MATRIX.json

`lesion_operator/guard.py` is also unchanged; the two new bindings CENTRAL
required (execution commit, cluster matrix hash) are implemented in the NEW
`execution/authorization.py`, which wraps the guard rather than editing it.

New execution-only surfaces:

    execution/cells.py          cell identity, staging, atomic finalize, retry
    execution/injection.py      activation injection at the frozen sites
    execution/evaluators.py     dispatch to the validated evaluators
    execution/preflight.py      the full A-L preflight
    execution/authorization.py  production authorization schema + validation
    execution/aggregate.py      COMPLETE-cells-only aggregation
    scripts/run_lesion_v2.py    execution body (rewritten)
    scripts/submit_lesion_v2.slurm  orchestration, NOT submitted

## 3. Activation injection

The operator contract fixes WHERE the frozen perturbation lands; this is the
wiring that puts it there, using temporary forward hooks removed on exit. No
module is modified and no parameter is touched.

    L1  forward hook on `wm.encoder`   -> h_n + eta
    L2  forward hook on `ltm.encoder`  -> h_n + eta, BEFORE to_semantic
    L3  forward PRE-hook on `ltm.decoder` -> h0 + eta, i.e. after
        tanh(sem_to_h0(s_hat)) and before recurrent generation

L3 is hooked on the decoder's initial hidden state because that value IS
`tanh(sem_to_h0(s_hat))`, supplied once per item — which is also why the
perturbation is automatically frozen across the whole autoregressive
trajectory: there is no per-step redraw to suppress.

A per-item base `u ~ Uniform(-1,+1)` is cached per cell, so the same draw
serves every severity and every compared task/decoder. `eta_fn=None` is the
k=0 control: hooks are installed and removed but nothing is added, so the
intact control traverses a byte-identical evaluator path.

## 4. Cell lifecycle

`cell_identity` = SHA-256 over `(state_id, state_sha256, site, severity_k,
realization)` — the authoritative row's scientific identity. A cell is written
to a staging directory and becomes visible through ONE atomic rename; the
`CELL_COMPLETE.json` marker, carrying per-file hashes, is the last file written
before it. A finalized cell is never overwritten, and a cell whose parameter
restoration was not verified is never finalized.

## 5. Scheduler decomposition

`--shard i --n-shards N` partitions cells by `(state, site)` only. That
boundary cannot alter science: mask and noise identities are pure functions of
`(state_sha256, site, realization)` and `(..., item_id)` and contain no
severity, task, decoder or shard index. Exactly-once is enforced by the cell
lifecycle, not the scheduler — a resubmitted shard skips COMPLETE cells.

## 6. Future submission plan (NOT SUBMITTED)

`scripts/submit_lesion_v2.slurm`, a 12-task array (4 states x 3 sites). It
requires `REPO`, `L3_LESION_V2_CONTROL_DIR` (the external execution-control
directory holding the authorization and the two transfer-required artifacts)
and `L3_LESION_V2_RESULTS` (a namespace OUTSIDE the git worktree, so the tree
stays clean for preflight check B).
