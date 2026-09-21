# CENTRAL STEERING HANDOFF — LESIONING V2 IMPLEMENTATION FREEZE

    BRANCH    paper-programme/lesioning-v2-implementation
    WORKTREE  ../wt-lesioning-v2-implementation
    BASE      5d1849a0c1d01065ff926d324703e8b767c05223
    DATE      2026-09-21

## Status

    IMPLEMENTATION_FREEZE_STATUS=
    BLOCKED

    L1_TENSOR_VERIFIED=NO
    L2_TENSOR_VERIFIED=NO
    L3_TENSOR_VERIFIED=NO
    TORCH_GRU_LAYOUT_VERIFIED=NO
    INTACT_SD_PROCEDURE_FROZEN=YES
    INTACT_SD_CONSTANTS_FROZEN=NO
    MASK_IMPLEMENTATION_VALID=YES
    ACTIVATION_IMPLEMENTATION_VALID=YES
    COMPOSITE_OPERATOR_VALID=YES
    RUN_MATRIX_FROZEN=YES
    EVALUATOR_FROZEN=YES
    SCIENTIFIC_LESION_EXECUTED=NO
    GO_FOR_SCIENTIFIC_EXECUTION=NO

BLOCKED is narrow and mechanical, not scientific. The operator, evaluator,
matrix, schema and guard are implemented, frozen and fully tested. The five NO
fields are the three gates that require the P1-P4 artifacts and the canonical
torch 2.6.0 environment — both of which live on Jean Zay, not on this machine.
Nothing about the design is unresolved.

Per contract section 3, GATE 1 must run against the ACTUAL state_dicts. I
therefore did not mark L1/L2/L3_TENSOR_VERIFIED=YES on the strength of the
frozen-code proof from the operator audit: that proof stands, but it is not the
direct file verification CENTRAL required, and claiming otherwise would
misreport what was checked.

TORCH_GRU_LAYOUT_VERIFIED is NO for the same reason — it PASSES locally under
torch 2.12.1 and the artifact is committed, but the canonical environment is
2.6.0. This one carries little risk: the operator depends solely on the
(3H, D) shape invariant, never on gate names or their order, and a test proves
the tiling is invariant under permutation of the three blocks.

## What is implemented and frozen

    L1  wm.encoder.weight_ih_l0    (384,64)     8,192 logical edges
    L2  ltm.encoder.weight_ih_l0   (1536,64)    32,768 logical edges
    L3  ltm.sem_to_h0.weight       (512,300)    153,600 logical edges

    MASK_GRANULARITY=LOGICAL_SOURCE_TARGET_EDGE
    MASK_SAMPLING=EXACT_COUNT_UNIFORM_WITHOUT_REPLACEMENT
    MASK_NESTING=NESTED_PREFIX        BIAS_LESIONABLE=NO
    CONNECTIVITY_P_MAX=0.30           N_SEVERITY_LEVELS=15 (nonzero)
    ACTIVATION_DISTRIBUTION=SYMMETRIC_UNIFORM
    ACTIVATION_NOISE_TIMING=PER_ITEM_FROZEN_ACROSS_AR_TRAJECTORY

    run matrix   b4d3c98816b46f59afb3e32f081fa073dd5a48cd27f2279fdaccb1f47ca6887c
                 1,812 cells = 1,800 lesion + 12 intact controls
                 POST_REPAIR only; no SOURCE state
    SD procedure 122ae40c600e639f875bdcb18e588bd32242981af0fa42c0e975a296bc851942

## Three points CENTRAL should note

**1. The maximum is a half-width, not an SD.** At k=15 the support is
[-SD_site, +SD_site]; the actual distribution SD is `SD_site / sqrt(3)`
~= 0.577 * SD_site. The code exposes both (`max_half_width`,
`distribution_sd`) so the manuscript cannot accidentally describe the amplitude
as one SD of noise.

**2. L3 is a new SD site.** The historical calibration covered the encoder
states only; `h0 = tanh(sem_to_h0(s_hat))` was never part of it. The same
statistical recipe is reused unchanged, and every L3 constant is stamped
`site_historically_calibrated: false`.

**3. I renamed the package.** It was `lesioning_v2/operator/`, which shadows the
Python standard library `operator` module for anything executed from that
directory — it broke tooling during implementation. It is now
`lesioning_v2/lesion_operator/`. No semantics changed.

## Execution guard

A nonzero lesion cell is refused unless `L3_LESION_V2_AUTHORIZATION` names an
artifact carrying the exact token
`CENTRAL_GO_FOR_SCIENTIFIC_EXECUTION_LESIONING_V2`,
`go_for_scientific_execution: true`, and the matching `run_matrix_sha256`.
Binding the matrix hash means a stale authorization cannot license a changed
matrix. There is no bypass flag; `--dry-run` always remains available.

## To reach READY_FOR_EXECUTION

Run the three gates on Jean Zay (exact commands in
`LESIONING_V2_PREEXECUTION_VALIDATION.md` section 3), then return the three
artifacts. Any mismatch is a hard stop, and none of the gates attempts repair.

    STOP. RETURN TO CENTRAL FOR FINAL GO / NO-GO.
