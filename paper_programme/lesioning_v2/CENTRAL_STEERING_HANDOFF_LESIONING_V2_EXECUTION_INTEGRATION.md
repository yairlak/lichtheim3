# CENTRAL STEERING HANDOFF — LESIONING V2 EXECUTION INTEGRATION

    BRANCH    paper-programme/lesioning-v2-execution-integration-repair
    WORKTREE  ../wt-lesioning-v2-execution-integration-repair
    BASE      dfdd2b6d023c788e2693f1dbee4cdd9a78219abc
    DATE      2026-09-21

## Status

    EXECUTION_INTEGRATION_STATUS=READY_FOR_FINAL_AUTHORIZATION
    FINAL_EXECUTION_COMMIT=<HEAD of this branch; `git rev-parse HEAD`>
    (deliberately NOT hard-coded: a commit cannot contain its own hash, and
     section 11A forbids baking in the yet-unknown repaired commit)
    RUN_MATRIX_SHA256=cd48e99cc95fe959315b599d3a1ccbfa4093f0fd5ff3aa7cd3c124a41071732a

    SCIENTIFIC_DESIGN_CHANGED=NO
    OPERATOR_SEMANTICS_CHANGED=NO
    EVALUATOR_SEMANTICS_CHANGED=NO
    MATRIX_CHANGED=NO

    FULL_PREFLIGHT_IMPLEMENTED=YES
    EXECUTION_LOOP_IMPLEMENTED=YES
    ITEM_OUTPUT_IMPLEMENTED=YES
    SUMMARY_OUTPUT_IMPLEMENTED=YES
    EXACTLY_ONCE_SEMANTICS_IMPLEMENTED=YES
    AUTHORIZATION_GUARD_IMPLEMENTED=YES
    SYNTHETIC_TESTS=PASS
    MATRIX_MAPPING_VALIDATION=PASS

    SCIENTIFIC_LESION_EXECUTED=NO
    GO_FOR_SCIENTIFIC_EXECUTION=NO

## Matrix mapping validation

    TOTAL_ROWS=1812
    UNIQUE_EXECUTABLE_CELL_IDENTITIES=1812
    DUPLICATE_CELL_IDENTITIES=0
    MISSING_CELL_IDENTITIES=0
    INTACT_CONTROL_ROWS=12
    NONZERO_LESION_ROWS=1800
    POST_REPAIR_ONLY=YES

The runner consumes `contract/LESIONING_V2_RUN_MATRIX.json` and never
regenerates it; a test asserts `build_run_matrix` is not importable from the
runner.

## What changed, and why none of it is science

Frozen scientific files are byte-unchanged from `dfdd2b6`:
`git diff dfdd2b6 -- lesion_operator/ contract/LESIONING_V2_RUN_MATRIX.json` is
EMPTY. `lesion_operator/guard.py` is untouched too — the two new bindings you
required (execution commit, cluster matrix hash) live in the NEW
`execution/authorization.py`, which wraps the guard rather than editing it.
That was the cleanest way to satisfy section 12 without touching a frozen file.

All other changes are new execution-only modules plus the rewritten
`scripts/run_lesion_v2.py`, which previously had no execution body at all.

`SHA256SUMS` was regenerated at this commit, as section 11E permits, because
the repair adds execution files. It hides nothing: the frozen scientific files
retain their previous hashes, which the empty diff above independently
confirms.

## Archaeology outcome

Every frozen endpoint mapped unambiguously to an already-validated evaluator —
`evaluate_forms_ar` for canonical repetition, the parity-tested free-AR
wrapper, `evaluate_naming`, `evaluate_comprehension_subset` — so nothing was
reimplemented and nothing was BLOCKED for want of semantics. Full table in the
integration manifest, section 1.

## Two implementation points worth your attention

**L3 injection hooks the decoder's initial hidden state.** That value *is*
`tanh(sem_to_h0(s_hat))`, supplied once per item, so the perturbation is frozen
across the autoregressive trajectory by construction — there is no per-step
redraw that could leak in. The same applies to L1/L2, whose sites are the
encoders' final states.

**k=0 traverses an identical evaluator path.** The intact control installs and
removes the same hooks with a null perturbation rather than taking a shortcut,
so a control cell and a lesion cell differ only in the frozen operator values.

## Failure/restart semantics

Frozen in `LESIONING_V2_EXECUTION_FAILURE_SEMANTICS.md` and performance-blind
by construction: `retry_allowed` reads one filesystem marker and has no access
to any metric. A COMPLETE cell is never rerun or overwritten; a cell whose
parameter restoration was not verified is never finalized; failed attempts are
retained as `FAILED_*` beside the cell and never enter aggregation.

## Evidence

    tests                 110 passed, 0 failed
    SHA256SUMS            regenerated at this commit, fully verified
    matrix semantic SHA   cd48e99c…1732a  before AND after (unchanged)
    operator file diff    EMPTY vs dfdd2b6
    evaluator file diff   EMPTY vs dfdd2b6 for all frozen files
    real P1-P4 model loaded          NO
    real P1-P4 nonzero forward       NO
    valid authorization artifact     NONE (template is false-GO; 4 invalid fixtures)
    scientific results namespace     ABSENT

## What CENTRAL must issue next

A NEW authorization naming **this** commit and the authoritative matrix:

    {
      "token": "CENTRAL_GO_FOR_SCIENTIFIC_EXECUTION_LESIONING_V2",
      "go_for_scientific_execution": true,
      "execution_commit": "<git rev-parse HEAD of this branch>",
      "run_matrix_sha256":
        "cd48e99cc95fe959315b599d3a1ccbfa4093f0fd5ff3aa7cd3c124a41071732a"
    }

Keep it and the results namespace OUTSIDE the worktree, or preflight check B
(clean tree) will refuse the run. Exact commands are in
`LESIONING_V2_EXECUTION_PREFLIGHT_FINAL.md`.

    STOP. DO NOT EXECUTE LESIONING V2.
