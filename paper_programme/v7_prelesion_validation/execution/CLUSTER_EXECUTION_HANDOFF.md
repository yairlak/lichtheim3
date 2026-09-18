# CLUSTER EXECUTION HANDOFF — V7 PRE-LESION SCIENTIFIC RUN

    design anchor        2d240e1fd4b81ca93b4144b772eb21e8700d8ddf   (frozen)
    execution plumbing   4c24efb                                    (frozen)
    runner commit        see `git rev-parse HEAD` after checkout
    contract drift       NONE — 8/8 byte-identical to the design anchor

## 1. What was added

    scripts/run_prelesion_validation.py   eight-state runner, with --dry-run
    scripts/generate_reports.py           reporting, reads frozen results only
    scripts/rules.py                      the frozen decision rules, torch-free

`prelesion_eval.py` remains the evaluator library; it now re-exports the rules
from `rules.py` rather than defining them, so **reporting is structurally
incapable of running a model** — it imports `rules`, which cannot import torch.
That is enforced by a test that blocks the torch import entirely and still
imports the reporting module.

Metric sources are reused, not reimplemented:

    canonical repetition   scripts.evaluate_train_lexicon_ceiling.evaluate_forms_ar
                           (the exact function repetition_snapshot uses)
    genuine free-AR        train_joint_scratch.free_ar_repetition semantics,
                           already parity-tested against the frozen evaluator
    naming                 train_tasks.evaluate_naming(return_per_item=True)
    C top-1                train_tasks.evaluate_comprehension_subset(...)
    c_LTM / g              the model's own field_confidence and gate outputs
                           (ltm_route.lexical_field -> confidence;
                            g = sigmoid(alpha*(c_LTM - gate_threshold)))

## 2. Runner guarantees

  * eight states, ordered from `state_manifest.tsv` — no typed state list
    (enforced by test);
  * POST_REPAIR = SOURCE + `head_first_c0.pt`, reconstructed in memory;
    the superseded head is unnameable in execution code — the checker builds
    its own sentinel from parts so no selectable literal exists anywhere;
  * refuses to start if `results/` exists — checked FIRST, before any input
    resolution, because it is the check that protects an existing result;
  * no `--force`, no `--overwrite`, no `--resume`, no `rmtree`;
  * per-state atomic finalize (`<state>.partial` -> rename), then one atomic
    rename of the whole staging directory into `results/`;
  * on abort, partial output is moved to `results.FAILED_<ts>`, i.e. OUTSIDE
    the scientific namespace, so a state is never silently re-run;
  * source checkpoint, repair head and model `state_dict` are re-verified
    unchanged after every state.

## 3. Preflight hard stops (all before any model forward)

    0  results namespace absent
    1  git HEAD resolvable (recorded as runner_commit)
    2  contract bytes identical to 2d240e1f
    3  canonical table + P1-P4 checkpoints + heads + GloVe resolve and match
       their frozen SHAs  (GloVe 91125602...c21ed; no fallback embeddings)
    6  eight states, each exactly once, order equal to the frozen contract
    7  stimulus manifest hashes recorded; primary is 378
    8  the superseded head is unreferenceable in execution code
    9  no training/backward/step/zero_grad/save path can execute

Any failure exits 2 with `PREFLIGHT_FAIL` and runs nothing.

## 4. EXACT Jean-Zay commands

    # 1. checkout the runner commit
    module load pytorch-gpu/py3/2.6.0
    cd <repo>
    git fetch --all
    git checkout feat/v7-prelesion-validation-execution
    git rev-parse HEAD                     # record this as runner_commit

    # 2. resolve inputs (verify the table hash BEFORE exporting)
    sha256sum /path/to/canonical_behavioral_item_table.tsv
    #   must be 8988aff6fac55ca36cb43ce758f5684f30ae10a6303bdbd7b0b9f462433d5a67
    export L3_CANON_TABLE=/path/to/canonical_behavioral_item_table.tsv
    export L3_V7_RUN_ROOT=/lustre/fsn1/projects/rech/llg/uss35bp/l3_fresh_v7_runs
    #   GloVe is taken from the frozen evaluator (data/glove.6B.300d.txt) and
    #   is SHA-checked by preflight; no fallback embedding is permitted.

    # 3. complete test suite  -> expect 77 passed, 0 skipped
    python -m pytest paper_programme/v7_prelesion_validation/tests -q -rs

    # 4. dry run -> PREFLIGHT_OK, zero model forwards, writes nothing
    python paper_programme/v7_prelesion_validation/scripts/run_prelesion_validation.py \
        --dry-run

    # 5. confirm the namespace is still absent
    ls paper_programme/v7_prelesion_validation/results    # must be "No such file"

    # 6. THE SCIENTIFIC RUN — EXACTLY ONCE  (operator action; not performed here)
    python paper_programme/v7_prelesion_validation/scripts/run_prelesion_validation.py

    # 7. reports, from the frozen results only (no model inference)
    python paper_programme/v7_prelesion_validation/scripts/generate_reports.py

Expected test totals:

    77 passed,  0 skipped   -> READY; proceed to step 4
    67 passed, 10 skipped   -> inputs not configured; NOT ready
    any failure             -> STOP, return to CENTRAL

## 5. Outputs

    results/RUN_MANIFEST.json            design/plumbing/runner/V7 commits,
                                         torch version, every input SHA, state
                                         ids, population manifest SHAs,
                                         timestamps marked METADATA_ONLY
    results/STATE_SUMMARY.{tsv,json}
    results/SOURCE_POST_PAIRED.{tsv,json}   c->c / c->w / w->c / w->w per
                                            endpoint + changed item ids, and
                                            c_LTM/g delta summaries
    results/states/<STATE_ID>/REALWORD_ITEM_LEVEL.tsv
                              PSEUDOWORD_PRIMARY_ITEM_LEVEL.tsv
                              PSEUDOWORD_SEED_SENSITIVITY_ITEM_LEVEL.tsv
                              PSEUDOWORD_DESCRIPTIVE_ITEM_LEVEL.tsv
                              GATING_ITEM_LEVEL.tsv
                              SUMMARY.json
    results/FILE_SHA256SUMS
    results/reports/…                    written by step 7 only

The frozen official POST battery (P1-P3 0/0/0/0, P4 1/1/0/0) is carried in the
run manifest and is checked in the real-word report. **A mismatch is a STOP and
an implementation discrepancy — it is not a model result to reinterpret.**

## 6. Status

    PRELESION_VALIDATION_STATUS =
        CONTRACT_FROZEN_RUNNER_READY_FOR_CLUSTER_FINAL_PREFLIGHT
    ROUTE_BEHAVIOR_STATUS = UNKNOWN_NOT_MEASURED
    GO_FOR_SCIENTIFIC_PRELESION_INFERENCE =
        NO_NOT_UNTIL_RUNNER_CLUSTER_PREFLIGHT_PASSES
    GO_FOR_LESIONING_V2_IMPLEMENTATION = NO

Step 6 was deliberately NOT performed.
