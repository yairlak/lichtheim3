# CENTRAL STEERING HANDOFF — V7 PRE-LESION VALIDATION

    DATE           2026-09-18
    BASE COMMIT    46e638628e2c610c640189523c7918ade60209ad
    BRANCH         feat/v7-prelesion-validation
    DESIGN COMMIT  2d240e1fd4b81ca93b4144b772eb21e8700d8ddf
    TRACKED TREE   CLEAN

## 1. Status

    PRELESION_VALIDATION_STATUS =
    CONTRACT_FROZEN_EXECUTION_BLOCKED_ARTIFACTS_UNAVAILABLE

    ROUTE_BEHAVIOR_STATUS = UNKNOWN_NOT_MEASURED

    P1_POST_REPAIR_ROUTE_PROFILE = UNKNOWN_NOT_MEASURED
    P2_POST_REPAIR_ROUTE_PROFILE = UNKNOWN_NOT_MEASURED
    P3_POST_REPAIR_ROUTE_PROFILE = UNKNOWN_NOT_MEASURED
    P4_POST_REPAIR_ROUTE_PROFILE = UNKNOWN_NOT_MEASURED

    PSEUDOWORD_DORSAL_VENTRAL_ORDERING = UNKNOWN_NOT_MEASURED
    ARM_A_PSEUDOWORD_PRESERVATION      = UNKNOWN_NOT_MEASURED

    GO_FOR_LESIONING_V2_IMPLEMENTATION = NO

CENTRAL asked for exactly one of `V7_LESION_READY` /
`V7_LESION_READY_WITH_QUALIFICATION` / `V7_NOT_LESION_READY`. **None can be
returned**, because no scientific inference was run: every one of those three
values is a claim about measured route behaviour, and returning any of them —
including the negative one — would assert a result that does not exist.
`V7_NOT_LESION_READY` in particular would be read as a finding about the models
and would be wrong. The classification remains open and is reachable as soon as
the eight states are evaluated against the now-frozen contract.

## 2. What was completed (CENTRAL's ordered NEXT ACTION, steps 1-8)

    1  worktree/branch on the pinned commit        DONE  feat/v7-prelesion-validation
    2  contract finalized                          DONE  V7_INTACT_ROUTE_VALIDATION_CONTRACT.md
    3  machine-readable manifests frozen           DONE  see below
    4  mechanical tests run                        DONE  27 passed, 4 skipped
    5  design committed                            DONE  2d240e1f
    6  design commit + SHA256SUMS recorded         DONE
    7  tracked tree clean                          DONE
    8  scientific result namespace empty           DONE  results/ does not exist
    9  execute scientific inference                BLOCKED — see §3

Frozen manifests (all under `paper_programme/v7_prelesion_validation/contract/`):

    state_manifest.tsv                            8 states, full identities
    stimulus_manifest_common_unseen_378.tsv       PRIMARY, N=378
    stimulus_manifest_seed_unseen.tsv             SECONDARY, 1,550 rows
                                                  (P1 384 / P2 390 / P3 389 / P4 387)
    stimulus_manifest_dorsal_pool_exposed_13.tsv  DESCRIPTIVE, N=13
    stimulus_manifest_trained_real_exact_671.tsv  DESCRIPTIVE, N=671
    population_manifests.json, validation_contract.json

All four CENTRAL decisions are encoded and test-enforced:

  * **D1** POST_REPAIR = SOURCE + `head_first_c0.pt` for all four slots;
    `head_final` recorded as `LATER_DERIVED_ARM_A_CONVERGENCE_ARTIFACT` /
    `NOT_SCIENTIFICALLY_EVALUATED`, and listed under `forbidden`. A test asserts
    per slot that heads are equal iff `first_c0_iter == steps`. CENTRAL's trace
    values satisfy this exactly — P1 30/30 equal; P2 34/35, P3 67/68, P4 29/30
    not equal — so the mechanism inferred in the arbitration pass is confirmed
    by direct observation, not inference.
  * **D2** PRIMARY = V7_COMMON_UNSEEN N=378, identical items for P1/P2/P3 and
    descriptive for P4; per-seed secondary; 13 exposed descriptive-only. Tests
    verify 378 ∪ 13 = the frozen 391, zero dorsal-pool overlap for every V7
    seed, and zero lexicon phonological overlap.
  * **D3** historical seeds 19-22 handling is carried in the arbitration pack
    (original and cleaned values preserved side by side, retrospective only).
  * **D4** free-AR reuses the frozen decoder unchanged. Tests assert the loop is
    bounded by `FREE_AR_MAX_STEPS = 12` alone and that no target-length symbol
    appears in the decode body — in the frozen implementation *and* in our
    read-only wrapper.

## 3. Why step 9 is blocked

`POST_REPAIR` requires the SOURCE checkpoint and `head_first_c0.pt`. Neither is
on this machine, for any slot. Re-verified immediately before this run:

    9,232 local .pt files enumerated; ZERO match any recorded V7 SOURCE size
    no head_first_c0.pt / head_final.pt for seeds 31-34
    no p1_s31 … p4_s34 run directory outside the metadata-only snapshot
    /lustre not mounted; non-interactive SSH to `jz` refused

The artifacts are on Jean Zay at
`/lustre/fsn1/projects/rech/llg/uss35bp/l3_fresh_v7_runs/…`, per V7's own
`FIRST_HIT_FROZEN.json`. The Jean-Zay tensor and trace audits CENTRAL relied on
were run there, which is consistent: that is where the data is.

This is a logistics gap (D4 of the previous handoff), not a scientific one. The
design is complete and portable.

## 4. What CENTRAL should do next

Run the frozen design where the artifacts already are:

    on Jean Zay, module load pytorch-gpu/py3/2.6.0
    checkout feat/v7-prelesion-validation @ 2d240e1f
    pytest paper_programme/v7_prelesion_validation/tests -q

The four currently-skipped tests must then RUN and pass before inference:
reconstruction changes only `{"2.weight","2.bias"}`; checkpoint and head
immutable across evaluation; deterministic decode repeat on smoke IDs; FULL
canonical and free-AR parity with the frozen V7 battery. If any fails, stop and
return to CENTRAL rather than proceeding.

Then execute the eight states exactly once into a fresh `results/` namespace and
emit the four result documents named in the contract. Nothing in the frozen
design may change at that point.

Alternatively, transfer the four run directories here (SOURCE checkpoints, both
head files per slot, and the battery/trace/summary JSONs) and the same commit
runs unchanged.

## 5. Compliance

    SCIENTIFIC_INFERENCE_EXECUTED = NO
    ROUTE_OUTCOMES_INSPECTED      = NO
    HEAD_FINAL_EVALUATED          = NO
    TRAINING = NO   PARAMETER_UPDATES = NO
    ARCHITECTURE / GATE / REPAIR CHANGES = NO
    V7_RE_EXECUTION = NO
    LESION IMPLEMENTED = NO   LESION EXECUTED = NO
    MONOLITHIC_CHECKPOINT_MATERIALIZED = NO
    FROZEN_V7_RUN_DIRECTORIES_MODIFIED = NO
    ITEMS_REMOVED_AFTER_SEEING_RESULTS = NO
    L1/L2/L3 CONNECTIVITY TENSORS AUDITED = NO
    p_max FROZEN = NO

    Result documents (V7_INTACT_ROUTE_VALIDATION_RESULTS.md,
    V7_PSEUDOWORD_VALIDATION_RESULTS.md,
    V7_SOURCE_REPAIR_PAIRED_ANALYSIS.md) were NOT created, because creating
    them without having run the evaluation would fabricate scientific output.

    STOP. RETURN TO CENTRAL.
