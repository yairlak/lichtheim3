# LESIONING V2 — BATCH ITEM-ID WIRING REPAIR

    BASE      0b22b90455a30b8d2ee1ca86df0fc96955e4e542
    BRANCH    paper-programme/lesioning-v2-batch-id-repair
    CLASS     EXECUTION_INTEGRATION_BATCH_ITEM_ID_WIRING
    DATE      2026-09-22

No real P1-P4 nonzero lesion forward was executed. No scientific semantics
changed.

## 1. Archaeology (recovered from executable code, not memory)

`run_cell` built ONE eta callback over the full 29,571-item global population
and installed the activation hook around the ENTIRE evaluation:

    eta_fn = injection.batch_eta_fn(..., item_ids, k, sd_site)   # 29,571 ids
    with injection.activation_injection(model, row["site"], eta_fn):
        item_rows = evaluators.evaluate_endpoints(tr, model, ..., bank, comp)

But every frozen evaluator iterates its own population in chunks of its OWN
batch size:

| decoding convention | batch size | source |
|---|---|---|
| CANONICAL_FORCED_LENGTH_AR | **128** | `scripts/evaluate_train_lexicon_ceiling.py:62` `BATCH_SIZE = 128`, loop at `:191` |
| GENUINE_FREE_AR | **256** | `prelesion_eval.free_ar_items(batch_size=256)`, loop `for lo in range(0, len(entries), batch_size)` |
| SEMANTIC_GREEDY_AR_GLOBAL_CAP | **512** | `train_tasks.evaluate_naming(batch_size=512)` |
| STRICT_TOP1_RETRIEVAL | **64** | `train_tasks.evaluate_comprehension_subset(batch_size=64)` |

The hook therefore received a 128-item tensor while the callback still carried
29,571 ids — the observed `InjectionError`. The 128 is the canonical
repetition evaluator, which is the first endpoint evaluated.

A second fact, recovered and decisive for the design: inside free-AR the model
is called **once per decoding step**

    for _ in range(FREE_AR_MAX_STEPS):
        o = model(enc_in, enc_mask, dec)

so the L1/L2 encoder hook and the L3 decoder pre-hook fire up to 12 times for
the SAME batch. A hook-call cursor would therefore assign a different eta at
each decoding step and destroy
`PER_ITEM_FROZEN_ACROSS_AUTOREGRESSIVE_TRAJECTORY`. CENTRAL's warning is exactly
right, and no cursor is used anywhere.

## 2. The repair

New file `execution/lesioned_eval.py`, used ONLY for k>0. It drives each frozen
evaluator **one chunk at a time at that evaluator's own recovered batch size**,
and installs the existing hook per chunk with exactly that chunk's GLOBAL item
ids:

    for chunk in chunks(bank_indices, sizes["CANONICAL_FORCED_LENGTH_AR"]):
        ids = [item_id(i) for i in chunk]
        with injection.activation_injection(model, site, make_eta(ids)):
            can_rows, _ = pe.canonical_items(tr, model, chunk, routes=...)

Because the chunk equals the evaluator's internal batch size, the evaluator's
internal loop runs exactly once per call: batching, per-batch padding and
ordering are preserved bit-for-bit. Within a chunk the hook may fire any number
of times and always sees the same ids, so the per-item perturbation is frozen
across the whole trajectory by construction rather than by policing.

Batch sizes are read out of the frozen code at run time (module constant and
signature defaults), never hard-coded here.

Item identity comes from explicit global population indices rendered
`bank_{i}` — never inferred from phoneme content, which homophones would make
unsafe.

## 3. k=0 is untouched

The intact-control branch is the original path verbatim: null perturbation,
`evaluators.evaluate_endpoints` over the full population, same hook installed
and removed. It is deliberately NOT routed through the nonzero batching driver.
A test asserts the k=0 branch contains `evaluators.evaluate_endpoints(` and
does not mention `lesioned_eval`.

That is the static proof that the 12 existing controls remain reproducible by
the repaired code: their entire execution path is byte-identical.

## 4. Byte-identity vs the base commit

Byte-identical (verified by `git diff` and by a test that re-hashes each file
against `git show BASE:<path>`):

    execution/injection.py            execution/evaluators.py
    lesion_operator/seeds.py          lesion_operator/noise.py
    lesion_operator/masks.py          lesion_operator/context.py
    lesion_operator/sites.py          lesion_operator/sd_procedure.py
    lesion_operator/battery.py        lesion_operator/guard.py
    contract/LESIONING_V2_RUN_MATRIX.json

Changed: `scripts/run_lesion_v2.py` (k>0 routing + continuation flag),
`execution/preflight.py` (continuation branch only). Added:
`execution/lesioned_eval.py`, `execution/continuity.py`,
`scripts/generate_continuity_manifest.py`, tests, documents, `SHA256SUMS`.

The run matrix semantic SHA is unchanged:
`cd48e99cc95fe959315b599d3a1ccbfa4093f0fd5ff3aa7cd3c124a41071732a`.

## 5. Continuation

`execution/continuity.py` generates and validates a READ-ONLY manifest of the
12 existing COMPLETE k=0 controls. It performs no write operation at all — a
test walks its AST and asserts no `remove`, `rmtree`, `unlink`, `rename`,
`makedirs` or `mkdir` call exists — and never recomputes a control.

Continuation into an existing namespace requires `--continuity-manifest`
explicitly. Without it, preflight check L refuses an existing namespace exactly
as before. With it, the namespace is RE-READ and must match the manifest
hash-for-hash; a manifest alone is never trusted. Refusals are tested for:
missing k=0, extra k=0, any COMPLETE k>0, foreign cell, altered marker, wrong
hash, stale manifest from another root, wrong matrix, duplicate identity.

Old cells keep their original provenance; nothing is rewritten. The final
package can therefore state truthfully that k=0 carries execution commit
`0b22b904…` while k>0 carries the repaired commit.

## 6. Authorization

Unchanged and not weakened. The previous authorization for `0b22b904` is VOID
after this code change; a repaired run needs a NEW authorization naming the new
final execution commit and the same matrix SHA. No valid production
authorization was created. `GO_FOR_SCIENTIFIC_EXECUTION` remains NO.
