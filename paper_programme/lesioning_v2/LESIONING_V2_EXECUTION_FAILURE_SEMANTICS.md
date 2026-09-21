# LESIONING V2 — EXECUTION FAILURE / RESTART SEMANTICS

    FROZEN BEFORE SCIENTIFIC EXECUTION
    PERFORMANCE-BLIND BY CONSTRUCTION

## Principle

A scheduler or infrastructure retry may rerun a cell **only if that cell was
never atomically finalized COMPLETE**. Nothing else is permitted to influence
the decision, and in particular no accuracy, loss, curve shape or any other
scientific quantity is ever consulted.

`cells.retry_allowed(root, row)` is the whole decision:

    return not is_complete(root, row)

It reads one filesystem marker. It has no access to item rows, summaries or
metrics, and a test asserts its source contains no performance vocabulary.

## When a retry is permitted

All of the following must hold, and all are identity checks:

  * the cell was never finalized COMPLETE;
  * the same authoritative matrix row (same `cell_identity` over
    state_id, state_sha256, site, severity_k, realization);
  * the same model state (same source checkpoint SHA and head_first_c0 SHA);
  * the same seeds — guaranteed, because mask and noise identities exclude
    severity, task, decoder and attempt number;
  * the same operator, evaluator identity, environment and commit contract.

Because the mask and noise identities are pure functions of
`(state_sha256, site, realization)` and `(…, item_id)`, a retry reproduces
byte-identical damage. A test asserts this invariance.

## What may never happen

  * A finalized COMPLETE cell is never rerun and never overwritten.
    `write_cell` raises `CellError` if the destination exists.
  * Partial output is never treated as complete. A cell is written to a staging
    directory and becomes visible only through one atomic `os.rename`; the
    `CELL_COMPLETE.json` marker is the last file written before that rename.
  * A cell whose parameter restoration was not verified is never finalized:
    `write_cell(restoration_verified=False)` raises.
  * Incomplete or failed cells never enter scientific aggregation.
    `aggregate` walks only directories carrying the COMPLETE marker and
    re-verifies each cell's own file hashes before reading it.

## Failed attempts

An interrupted or failed attempt leaves its staging directory renamed to
`FAILED_<cell key>_<staging id>`, beside the cell but never carrying the
COMPLETE marker. Diagnostic metadata and logs are retained there for
inspection. Aggregation ignores it by construction.

## Explicitly forbidden

    - retrying because a result "looks wrong"
    - skipping a cell because it is slow or expensive
    - adaptive scheduling based on observed scientific output
    - resuming a partially written cell in place
    - deleting a FAILED attempt to make a rerun look like a first attempt
