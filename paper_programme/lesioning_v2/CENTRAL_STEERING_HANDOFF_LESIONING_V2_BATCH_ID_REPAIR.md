# CENTRAL STEERING HANDOFF — LESIONING V2 BATCH-ID REPAIR

    BASE      0b22b90455a30b8d2ee1ca86df0fc96955e4e542
    BRANCH    paper-programme/lesioning-v2-batch-id-repair
    WORKTREE  ../wt-lesioning-v2-batch-id-repair
    DATE      2026-09-22

## Status

    BATCH_ID_REPAIR_STATUS=PENDING_CLUSTER_CONTINUITY_MANIFEST

    RUN_MATRIX_SHA256=cd48e99cc95fe959315b599d3a1ccbfa4093f0fd5ff3aa7cd3c124a41071732a

    SCIENTIFIC_DESIGN_CHANGED=NO
    INJECTION_SCIENTIFIC_SEMANTICS_CHANGED=NO
    EVALUATOR_SEMANTICS_CHANGED=NO
    SEED_HIERARCHY_CHANGED=NO

    BATCH_SIZE_INVARIANCE=PASS
    ITEM_ORDER_MAPPING=PASS
    FREE_AR_ITEM_ETA_PRESERVATION=PASS

    EXISTING_K0_REUSE_STATUS=SAFE_TO_REUSE (static proof; cluster
                             continuity manifest still PENDING)
    EXISTING_K0_COMPLETE=12/12  (as reported by CENTRAL; not re-read here)
    NONZERO_LESION_CELL_FINALIZED=0

    SYNTHETIC_TESTS=PASS (148)
    GO_FOR_SCIENTIFIC_EXECUTION=NO

**Why not READY_FOR_AUTHORIZATION.** The required
`LESIONING_V2_INTACT_CONTROL_CONTINUITY_MANIFEST.json` must be generated
read-only from the real Jean-Zay result namespace, which is not present here.
Fabricating it — or its hashes — would defeat its purpose. The generator and
its tests are complete; only the cluster read remains. This is CENTRAL's own
§11 intermediate status, used deliberately.

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

## Remaining step

Run the read-only generator on Jean-Zay (command in the final response), bring
the exact JSON back, and make the FINAL closure commit. Only that commit goes
to CENTRAL for scientific execution authorization.

    STOP. NO REAL LESION EXECUTED.
