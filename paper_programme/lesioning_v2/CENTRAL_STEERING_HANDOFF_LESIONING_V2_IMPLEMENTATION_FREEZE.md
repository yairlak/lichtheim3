# CENTRAL STEERING HANDOFF — LESIONING V2 IMPLEMENTATION FREEZE

    BRANCH    paper-programme/lesioning-v2-implementation
    WORKTREE  ../wt-lesioning-v2-implementation
    BASE      5d1849a0c1d01065ff926d324703e8b767c05223 (operator audit)
    IMPL      e1aa93cf1c0703af64afb44c874adb1277116682 (implementation freeze)
    ENV       torch 2.6.0 (canonical Jean-Zay validation)
    DATE      2026-09-21

## Status

    IMPLEMENTATION_FREEZE_STATUS=READY_FOR_EXECUTION
    L1_TENSOR_VERIFIED=YES
    L2_TENSOR_VERIFIED=YES
    L3_TENSOR_VERIFIED=YES
    TORCH_GRU_LAYOUT_VERIFIED=YES
    INTACT_SD_PROCEDURE_FROZEN=YES
    INTACT_SD_CONSTANTS_FROZEN=YES
    MASK_IMPLEMENTATION_VALID=YES
    ACTIVATION_IMPLEMENTATION_VALID=YES
    COMPOSITE_OPERATOR_VALID=YES
    RUN_MATRIX_FROZEN=YES
    EVALUATOR_FROZEN=YES
    SCIENTIFIC_LESION_EXECUTED=NO
    GO_FOR_SCIENTIFIC_EXECUTION=NO

## Binding authorization hash

Any future execution authorization MUST bind to:

    cd48e99cc95fe959315b599d3a1ccbfa4093f0fd5ff3aa7cd3c124a41071732a

The artifact must carry the token
`CENTRAL_GO_FOR_SCIENTIFIC_EXECUTION_LESIONING_V2`,
`go_for_scientific_execution: true`, and that exact `run_matrix_sha256`, and be
exposed via `L3_LESION_V2_AUTHORIZATION`. An authorization naming the earlier
pre-cluster matrix `b4d3c988...` is refused; a test pins that.

## Cluster validation incorporated

    IMPLEMENTATION_TESTS = 64_PASS on the cluster
    GATE 1 verify_states       PASS — actual P1-P4 POST_REPAIR state_dicts
    GATE 2 verify_gru_layout   PASS — torch 2.6.0
    GATE 3 extract_intact_sd   PASS — intact-only SD constants

GRU runtime: `three_contiguous_H_blocks=true`,
`logical_mask_tiles_identically=true`, `operator_depends_on_gate_names=false`,
documented order `r,z,n`.

Final resolved matrix: 1,812 cells = 1,800 lesion + 12 intact controls,
`state_sha_resolved=true`.

Final dry run: PREFLIGHT_OK, `no_training_path=true`,
`superseded_head_unreferenced=true`, no model loaded, no forward pass,
`GO_FOR_SCIENTIFIC_EXECUTION=NO`.

### Frozen intact site SDs

    state              L1         L2         L3
    P1_POST_REPAIR     0.492585   0.524076   0.820609
    P2_POST_REPAIR     0.477633   0.535038   0.824531
    P3_POST_REPAIR     0.491430   0.572050   0.839828
    P4_POST_REPAIR     0.482054   0.489595   0.802448

    procedure hash 122ae40c600e639f875bdcb18e588bd32242981af0fa42c0e975a296bc851942

The L1/L2 encoder scales sit near 0.48-0.57 and the L3 production state near
0.80-0.84, so at k=15 the L3 perturbation half-width is roughly 1.7x the L1 one
in absolute terms. That is the intended consequence of scaling to each site's
own measured SD rather than to a common absolute range, and it is worth stating
plainly in the manuscript: severity is matched in site-relative units, not in
absolute activation units.

Reminder on terminology, since the numbers above are half-widths: at k=15 the
distribution SD is the tabulated value divided by sqrt(3) (P1/L3: half-width
0.820609, distribution SD 0.473779).

## Artifact provenance — two reproduced, two transfer-required

    LESIONING_V2_RUN_MATRIX.json          751a71d6...de150   IN REPO
    LESIONING_V2_GRU_LAYOUT.json          762d9547...b49c37  IN REPO
    LESIONING_V2_STATE_VERIFICATION.json  d216fe56...f96349  TRANSFER REQUIRED
    LESIONING_V2_INTACT_SD_CONSTANTS.json 3c6dcd35...b0602   TRANSFER REQUIRED

The run matrix and the GRU layout record were **reproduced byte-identically off
the cluster** from the frozen code and verified against the canonical hashes —
independent confirmation that the cluster ran this code and nothing else.

The other two were deliberately NOT reconstructed. They contain cluster-local
absolute paths, deployed-head digests, the population hash and descriptive
statistics that cannot be derived off-cluster, and fabricating a file to match
a published hash would defeat the purpose of the hash. They must be transferred
into `contract/` and verified against the hashes above.

**Consequence, stated plainly:** with `LESIONING_V2_INTACT_SD_CONSTANTS.json`
absent from this repository, the runner refuses any real run at the SD gate
before reaching the authorization guard. READY_FOR_EXECUTION describes the
frozen design and its validated implementation; the execution host must hold
both transferred artifacts. A test asserts they are absent while declared
transfer-required, so the two states cannot be silently confused.

## Closure record

    closure tests   70 passed (64 implementation + 6 closure)
    SHA256SUMS      recomputed and verified
    operator semantics, evaluator semantics, SD procedure, run-matrix
    semantics, lesion sites, p_max, activation amplitude and seed hierarchy
    are all UNCHANGED by this closure commit

    STOP. DO NOT EXECUTE LESIONING V2. RETURN TO CENTRAL.
