# IMPLEMENTATION HANDOFF — V7 PRE-LESION EXECUTION PREFLIGHT

    DESIGN ANCHOR (unchanged)  2d240e1fd4b81ca93b4144b772eb21e8700d8ddf
    IMPLEMENTATION BRANCH      feat/v7-prelesion-validation-execution
    SCOPE                      machine-portable input resolution +
                               artifact-dependent tests ONLY
    SCIENTIFIC CONTRACT        byte-identical to the design commit (proven)

## 1. What was broken, and what was done

Three defects, all mine, all infrastructure:

  1. `CANON_TABLE` was a hard-coded repo-relative path. On Jean Zay it resolved
     to a file that does not exist there, failing 3 tests.
  2. `_artifacts_present()` read `source_checkpoint_path`, a field that does not
     exist in the frozen `state_manifest.tsv`. `os.path.exists("")` is always
     False, so the guard **always** skipped — the skip was vacuous and could
     never have activated, on any machine.
  3. All four artifact-dependent tests were `NotImplementedError` placeholders,
     so "31 passed" was never reachable.

Repaired:

  * new `execution/inputs.py` resolves inputs per machine, outside the frozen
    contract, and **fails closed** on a missing file or SHA mismatch;
  * the guard now calls real SHA-verified resolution;
  * all four tests are implemented against the real evaluator;
  * static tests prove no placeholder body remains and that the guard cannot
    regress to the vacuous form.

No scientific content changed. No scientific result was inspected.

## 2. The two execution inputs

    L3_CANON_TABLE   canonical_behavioral_item_table.tsv
                     required sha256
                     8988aff6fac55ca36cb43ce758f5684f30ae10a6303bdbd7b0b9f462433d5a67

    L3_V7_RUN_ROOT   directory holding the four V7 run directories
                     on Jean Zay: /lustre/fsn1/projects/rech/llg/uss35bp/l3_fresh_v7_runs

`L3_V7_RUN_ROOT` is an execution root only. It is deliberately NOT written into
the scientific state manifest. Per-slot paths are derived from the frozen slurm
layout:

    <root>/fresh_ceiling_v7_{slot}_s{seed}/checkpoints/step_%08d.pt
    <root>/fresh_ceiling_v7_{slot}_s{seed}/post/seed{seed}_u{u}_A/head_first_c0.pt

    P1 s31 step 04125330 u1485      P3 s33 step 02486310 u895
    P2 s32 step 04083660 u1470      P4 s34 step 06444960 u2320

Each resolved file is checked against the frozen SHA in `state_manifest.tsv`.
A mismatch raises `InputResolutionError` — no related file is ever substituted.

Optional single-file form, for a layout that differs:

    export L3_EXECUTION_INPUTS=/path/execution_inputs.json
    {
      "canon_table":  "/path/canonical_behavioral_item_table.tsv",
      "v7_run_root":  "/lustre/fsn1/projects/rech/llg/uss35bp/l3_fresh_v7_runs",
      "artifacts": {
        "P1": {"source_checkpoint": "...", "repair_head": "..."}
      }
    }

## 3. EXACT Jean-Zay preflight commands

    module load pytorch-gpu/py3/2.6.0

    cd <repo>
    git fetch --all
    git checkout feat/v7-prelesion-validation-execution

    # locate the canonical table on the cluster (it was NOT under fswork
    # lichtheim3/outputs in the failing run) and verify before exporting:
    sha256sum /path/to/canonical_behavioral_item_table.tsv
    # must print 8988aff6fac55ca36cb43ce758f5684f30ae10a6303bdbd7b0b9f462433d5a67

    export L3_CANON_TABLE=/path/to/canonical_behavioral_item_table.tsv
    export L3_V7_RUN_ROOT=/lustre/fsn1/projects/rech/llg/uss35bp/l3_fresh_v7_runs

    python -m pytest paper_programme/v7_prelesion_validation/tests -q -rs

If the canonical table cannot be found on the cluster, copy it from the machine
that holds it (this Mac) rather than regenerating it:

    scp <mac>:.../behavioral_wfe_fulllexicon_93a577f/behavioral_analysis/tables/\
canonical_behavioral_item_table.tsv $WORK/

## 4. Expected result of that command

    40 passed, 10 skipped        -> inputs not configured; NOT ready
    50 passed,  0 skipped        -> READY for scientific execution

The ten currently-skipped tests are:

    test_reconstruction_changes_only_head_tensors            [P1 P2 P3 P4]
    test_checkpoint_and_head_immutable_across_evaluation     [P1 P2 P3 P4]
    test_deterministic_decode_repeat_on_smoke_ids
    test_full_canonical_and_freear_match_frozen_v7_on_smoke

Any FAILURE among them is a stop condition: return to CENTRAL, do not run the
eight scientific states.

## 5. What the four tests actually assert

**reconstruction_changes_only_head_tensors** (per slot) — loads SOURCE and
`head_first_c0` read-only, reconstructs POST_REPAIR through the evaluator's own
code path, and requires the set of changed full-model tensors to be **exactly**
`gradient_training_probe.TRAINABLE_NAMES` = `{ltm.to_semantic.2.weight,
ltm.to_semantic.2.bias}`. Those keys come from frozen code — Arm-A's own
declaration of what it optimises, and the module `_isolated_model` writes into —
not from the observed diff. It also re-checks the deployed-head digest against
the frozen one.

**checkpoint_and_head_immutable_across_evaluation** (per slot) — hashes both
files, runs a real (small) free-AR evaluation, re-hashes, and requires byte
equality with each other and with the frozen SHAs.

**deterministic_decode_repeat_on_smoke_ids** — decodes the predeclared smoke set
twice under the frozen semantics across all three routes and requires identical
output digests, plus an unchanged model `state_dict`. Smoke items are fixed in
`inputs.py` before execution (the 64 lowest bank indices) and were chosen
without reference to any outcome. No stochastic code is involved.

**full_canonical_and_freear_match_frozen_v7_on_smoke** — evaluator parity:
  * free-AR: our item-level wrapper, aggregated, must equal frozen
    `JointScratchTrainer.free_ar_repetition` exactly on the same items and
    model, for full/wm/ltm. No archived item-level output is fabricated.
  * canonical: the contract reuses frozen `repetition_snapshot` unchanged, so
    parity is asserted as reproducibility of that frozen function plus
    well-formedness of its FULL readout.
  * the frozen OFFICIAL AGGREGATE outcomes (P1-P3 `0/0/0/0`, P4 `1/1/0/0`) are
    retained separately in `inputs.FROZEN_OFFICIAL_POST_BATTERY` as the external
    invariant for the full scientific run, and are deliberately NOT asserted on
    this smoke subset, which is not the official population.

## 6. Integrity proof

    execution/CONTRACT_BYTE_IDENTITY.tsv    8/8 files equal=YES
    execution/contract_byte_identity.json   all_equal = true
    git diff 2d240e1f -- .../contract/      EMPTY

    results/ does not exist
    no checkpoint, head or model file was written or modified
    no scientific outcome was evaluated
    no training / lesion / architecture / gate / repair code added
      (enforced by test_no_training_or_lesion_code_added)

## 7. Status

    PRELESION_VALIDATION_STATUS =
        CONTRACT_FROZEN_IMPLEMENTATION_READY_FOR_CLUSTER_PREFLIGHT
    ROUTE_BEHAVIOR_STATUS = UNKNOWN_NOT_MEASURED
    GO_FOR_SCIENTIFIC_PRELESION_INFERENCE = NO_NOT_UNTIL_CLUSTER_TESTS_ALL_PASS
    GO_FOR_LESIONING_V2_IMPLEMENTATION = NO
