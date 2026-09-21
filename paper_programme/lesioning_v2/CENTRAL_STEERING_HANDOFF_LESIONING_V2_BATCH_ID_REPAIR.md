# CENTRAL STEERING HANDOFF — LESIONING V2 BATCH-ID REPAIR

    BASE      0b22b90455a30b8d2ee1ca86df0fc96955e4e542
    BRANCH    paper-programme/lesioning-v2-batch-id-repair
    WORKTREE  ../wt-lesioning-v2-batch-id-repair
    DATE      2026-09-22

## Status

    BATCH_ID_REPAIR_STATUS=READY_FOR_AUTHORIZATION

    RUN_MATRIX_SHA256=cd48e99cc95fe959315b599d3a1ccbfa4093f0fd5ff3aa7cd3c124a41071732a

    SCIENTIFIC_DESIGN_CHANGED=NO
    INJECTION_SCIENTIFIC_SEMANTICS_CHANGED=NO
    EVALUATOR_SEMANTICS_CHANGED=NO
    SEED_HIERARCHY_CHANGED=NO

    BATCH_SIZE_INVARIANCE=PASS
    ITEM_ORDER_MAPPING=PASS
    FREE_AR_ITEM_ETA_PRESERVATION=PASS

    EXISTING_K0_REUSE_STATUS=SAFE_TO_REUSE
    EXISTING_K0_COMPLETE=12/12  (validated read-only on Jean-Zay)
    NONZERO_LESION_CELL_FINALIZED=0

    SYNTHETIC_TESTS=PASS (148)
    GO_FOR_SCIENTIFIC_EXECUTION=NO

    ORIGINAL_K0_EXECUTION_COMMIT=0b22b90455a30b8d2ee1ca86df0fc96955e4e542
    CONTINUITY_MANIFEST_FILE_SHA256=
      a800a75f79a15ce099a3f2e35b1068bab8fd3ec252547329d07575d9a68e1467
    CONTINUITY_MANIFEST_INTERNAL_SHA256=
      5fc1fd38f3ff6933f7474306255cdf166499da2920bb0648259d844c818943a9

    NEW_EXECUTION_COMMIT=
      FINAL_GIT_HEAD_REPORTED_EXTERNALLY_AFTER_CLOSURE_COMMIT
      (a committed file cannot contain the SHA of the commit that contains it;
       obtain it with `git rev-parse HEAD` after the closure commit)

**Closure.** The cluster continuity manifest has been generated read-only on
Jean-Zay and imported byte-identical. The previous intermediate status
`PENDING_CLUSTER_CONTINUITY_MANIFEST` is therefore closed.

## The defect, and why it happened

`run_cell` built one eta callback over all 29,571 global ids and installed the
hook around the whole evaluation, while each frozen evaluator chunks its own
population: canonical repetition **128** (`evaluate_train_lexicon_ceiling.BATCH_SIZE`),
free-AR **256**, naming **512**, comprehension **64**. The first endpoint
evaluated is canonical repetition, which is why the failure reported 128.

A second recovered fact shaped the fix: inside free-AR the model is called once
per decoding step, so the hook fires up to 12 times on the same batch. A
hook-call cursor would have silently changed eta at every step and destroyed
`PER_ITEM_FROZEN_ACROSS_AUTOREGRESSIVE_TRAJECTORY` — a far worse failure than
the crash, because it would not have crashed. No cursor is used.

## The repair

`execution/lesioned_eval.py` (new, k>0 only) drives each frozen evaluator one
chunk at a time at that evaluator's own batch size, installing the existing hook
per chunk with exactly that chunk's global ids. Chunk == internal batch size, so
each evaluator's internal loop runs exactly once per call and its batching,
padding and ordering are preserved bit-for-bit. Sizes are read from the frozen
code at run time, never hard-coded.

Because the whole chunk evaluation sits inside one hook installation, the
per-item eta is frozen across the trajectory by construction.

## Byte identity vs base

| file | byte-identical |
|---|---|
| `execution/injection.py` | **YES** |
| `execution/evaluators.py` | **YES** |
| `lesion_operator/seeds.py` | **YES** |
| `lesion_operator/noise.py` | **YES** |
| `lesion_operator/masks.py` | **YES** |
| `lesion_operator/context.py` | **YES** |
| `lesion_operator/sites.py` | **YES** |
| `lesion_operator/sd_procedure.py`, `battery.py`, `guard.py` | **YES** |
| `contract/LESIONING_V2_RUN_MATRIX.json` | **YES** |

Changed: `scripts/run_lesion_v2.py`, `execution/preflight.py` (continuation
branch only), `SHA256SUMS`. Added: `lesioned_eval.py`, `continuity.py`,
`generate_continuity_manifest.py`, tests, documents.

A test re-hashes each frozen file against `git show BASE:<path>`, so drift
cannot pass unnoticed.

## k=0 continuity

    EXISTING_K0_REUSE_STATUS=SAFE_TO_REUSE

The k=0 branch is the original path verbatim — null perturbation, unchanged
`evaluators.evaluate_endpoints` over the full population, same hook installed
and removed — and is deliberately not routed through the new batching driver.
Its entire execution path is byte-identical, which is the strongest available
static proof. The 12 real controls were NOT recomputed and their hashes were
NOT fabricated.

## Continuation, fail-closed

Default behaviour is unchanged: an existing namespace is refused.
`--continuity-manifest` is the only way in, and the namespace is RE-READ and
matched hash-for-hash against the manifest — a manifest alone is never trusted.
Refusal is tested for missing/extra/foreign k=0 cells, any COMPLETE k>0,
altered markers, wrong hashes, stale manifests, wrong matrix and duplicate
identities. COMPLETE cells are skipped and cannot be overwritten. No scientific
value enters any of these decisions.

## Authorization

Not weakened. The authorization for `0b22b904` is VOID after this change. A
repaired run requires a NEW authorization naming the new final execution commit
and the same matrix SHA. None was created here.

## Cluster continuity validation — PASSED

    EXISTING_K0_COMPLETE=12/12
    NONZERO_LESION_CELL_FINALIZED=0
    cluster suite: 180 passed, 1 skipped
      (the skip is test_preflight_requires_canonical_torch_for_real_run,
       which skips by design when already on the canonical runtime)
    git diff BASE -- lesion_operator/                EMPTY
    git diff BASE -- contract/LESIONING_V2_RUN_MATRIX.json  EMPTY

Post-scan lifecycle across the twelve shard roots: TOTAL_COMPLETE=12, each
shard COMPLETE=1 STAGING=0 FAILED=0. The read-only scan therefore left the
scientific result namespace unaltered, as its design requires.

The imported manifest was independently re-validated in this worktree: outer
file SHA exact, internal `manifest_sha256` recomputed and exact, 12 COMPLETE
k=0 controls, zero COMPLETE nonzero cells, `run_matrix_sha256` exact, every
control carrying `original_execution_commit = 0b22b904…`, all twelve
authoritative k=0 identities represented exactly once, one control per shard
root `shard_00`..`shard_11`. The repository's own structural validator
(`continuity._check_manifest_structure`) accepts it.

The outer file SHA and the internal `manifest_sha256` are deliberately
different quantities — the first hashes the final JSON bytes, the second is the
generator's logical digest over the manifest body. Neither was "corrected"
toward the other.

## Provenance is mixed, and stays that way

**k=0 controls** were executed under the original authorized execution commit
`0b22b90455a30b8d2ee1ca86df0fc96955e4e542`. They are preserved verbatim, are
NOT recomputed, and are reused only under continuity-manifest validation. No
`CELL_COMPLETE.json` or `provenance.json` was modified. They were **not**
generated by the repaired commit and this package never claims otherwise.

**k>0 cells** have NOT been executed successfully — zero were ever finalized.
They require a NEW CENTRAL authorization and, if authorized, will execute under
the final repaired execution commit, carrying that commit in their own
provenance. Nothing is homogenised.

    STOP. NO REAL LESION EXECUTED.
    GO_FOR_SCIENTIFIC_EXECUTION=NO

---

# ADVERSARIAL STATIC AUDIT — findings and repairs

Two real integration omissions were found and fixed. Both would have surfaced
only on the cluster.

## Finding 1 — `lesioned_eval.py` was outside the scientific identity

The new driver directly determines k>0 output: it chooses the batch boundaries
at which each frozen evaluator is invoked and the ids the perturbation is
derived from. It was in `SHA256SUMS` but **not** in `EVALUATOR_IDENTITY_FILES`,
so it was covered by neither the evaluator identity hash nor the no-training
AST scan. Its bytes could have changed without altering the evaluator identity.

Fixed: added to `EVALUATOR_IDENTITY_FILES` (which the no-training scan iterates),
and `execution/continuity.py` added to the no-training scan. A test mutates
`lesioned_eval.py` and asserts the combined identity actually changes, then
restores it — proving the binding is real rather than declared.

    LESIONED_EVAL_IN_EVALUATOR_IDENTITY=YES
    LESIONED_EVAL_IN_PACKAGE_INTEGRITY=YES
    LESIONED_EVAL_IN_NO_TRAINING_SCAN=YES

## Finding 2 — continuation assumed a flat single-root namespace

`validate_for_continuation` required `abspath(out_root) == manifest.result_root`
and then demanded exactly 12 COMPLETE k=0 cells under that root. The real
namespace is

    $SCRATCH/l3_lesion_v2_results_0b22b904/shard_00 .. shard_11

with ONE control per shard, and the repaired run will again invoke the runner
once per shard with `--out-dir <parent>/shard_XX`. Every shard would therefore
have been refused: it holds one control, not twelve.

Fixed: validation now has two sanctioned shapes, both re-reading the
filesystem.

    GLOBAL  out_root IS the parent        -> all 12 must be present
    SHARD   out_root is a shard beneath it -> only that shard's own control is
            expected; the other 11 are never looked for, moved or copied

The GLOBAL manifest is still structurally checked in both modes, so a shard
cannot continue on a manifest that is missing another shard's control or that
declares any COMPLETE k>0 cell.

## Topology tests added

All 12 shards exercised exactly as the future runner will (global manifest,
`--out-dir parent/shard_XX`): continuation accepted for every shard, the
correct local control recognised and skipped, the other 11 absent locally and
not required, that shard's nonzero rows still eligible. Refused: wrong local
hash, another shard's control substituted, a global manifest missing a
control, a manifest declaring COMPLETE k>0, a COMPLETE k>0 cell inside a
shard, an out_root outside the parent. One test hashes every file in the
namespace before and after generating and validating all 12 shards and asserts
nothing changed.

## Provenance truth

Every manifest entry records
`original_execution_commit = 0b22b90455a30b8d2ee1ca86df0fc96955e4e542`, and
future k>0 cells carry the repaired commit in their own provenance. Nothing is
homogenised and no old artifact is rewritten.

## Corrected result parent

Documentation and commands now use
`$SCRATCH/l3_lesion_v2_results_0b22b904`. The manifest is written under
`$WORK/l3_lesion_v2_control/`, never inside the results parent — the generator
refuses an `--out` inside the result root.
