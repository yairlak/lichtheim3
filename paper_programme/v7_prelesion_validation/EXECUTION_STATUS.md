# EXECUTION STATUS — V7 PRE-LESION INTACT VALIDATION

    DATE   2026-09-18
    DESIGN CONTRACT FROZEN AND COMMITTED = YES
    SCIENTIFIC INFERENCE EXECUTED        = NO

## Why execution did not run here

CENTRAL authorized scientific inference (`GO_FOR_SCIENTIFIC_PRELESION_INFERENCE
= YES`). Steps 1-8 of the ordered NEXT ACTION are complete. Step 9 — execute the
eight states — cannot run on this machine because the required artifacts are not
on it.

`POST_REPAIR` is defined as SOURCE checkpoint + `head_first_c0.pt`. Neither
exists locally, for any slot:

    - 9,232 local `.pt` files were enumerated; ZERO match any of the four
      recorded V7 SOURCE byte sizes (31171762 / 31171826 / 31169138 / 31173042).
    - No `head_first_c0.pt` or `head_final.pt` exists for seeds 31-34.
      (The local ones belong to the V4/V5/V6 seed-19..22 lineage.)
    - No V7 run directory (p1_s31 … p4_s34) exists outside the metadata-only
      interview snapshot.

The artifacts live on IDRIS Jean Zay, per V7's own provenance:

    /lustre/fsn1/projects/rech/llg/uss35bp/l3_fresh_v7_runs/
        fresh_ceiling_v7_p1_s31/checkpoints/step_04125330.pt      (and P2-P4)

`/lustre` is not mounted here, and non-interactive SSH to the configured `jz`
host is refused (`Permission denied (publickey,gssapi-keyex,gssapi-with-mic,
password)`). This session cannot complete that authentication.

Re-checked immediately before this run: still absent.

## What IS complete and frozen

    worktree/branch   feat/v7-prelesion-validation @ 46e638628e2c…  (clean)
    contract          V7_INTACT_ROUTE_VALIDATION_CONTRACT.md
                      validation_contract.json
    manifests         state_manifest.tsv                       8 states
                      stimulus_manifest_common_unseen_378.tsv  PRIMARY
                      stimulus_manifest_seed_unseen.tsv        1,550 rows
                      stimulus_manifest_dorsal_pool_exposed_13.tsv
                      stimulus_manifest_trained_real_exact_671.tsv
                      population_manifests.json
    evaluation code   scripts/prelesion_eval.py   (read-only; reuses the frozen
                      free-AR decoder semantics unchanged)
    tests             27 passed, 4 skipped (the 4 require the artifacts)
    SHA256SUMS        present and verified
    results namespace absent/empty, as required

## To execute

On a host that has the V7 run directories (i.e. Jean Zay, under
`module load pytorch-gpu/py3/2.6.0`):

    git fetch && git checkout feat/v7-prelesion-validation   # design commit below
    python -m pytest paper_programme/v7_prelesion_validation/tests -q
        # the 4 skipped tests must now RUN and pass:
        #   reconstruction changes only {'2.weight','2.bias'}
        #   checkpoint/head immutable across evaluation
        #   deterministic decode repeat on smoke IDs
        #   FULL canonical + free-AR parity with the frozen V7 battery

Then run the eight states exactly once, writing into a fresh `results/`
namespace, and produce the four result documents named in the contract.

Nothing in the frozen design may change at that point. If any of the four
artifact-dependent tests fails, stop and return to CENTRAL rather than
proceeding to scientific inference.
