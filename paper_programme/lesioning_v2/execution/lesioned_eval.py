"""Batch-aligned evaluation driver for NONZERO lesion cells.

Why this file exists
--------------------
`run_cell` used to build ONE eta callback over the whole 29,571-item global
population and install the activation hook around the entire evaluation. But
every frozen evaluator iterates its population in chunks of its OWN batch size,
so the hook received a B-item tensor while the callback still carried 29,571
ids -> InjectionError.

The repair drives each frozen evaluator ONE CHUNK AT A TIME, at exactly that
evaluator's own recovered batch size, and installs the existing activation hook
for the whole of that chunk's evaluation. Because the chunk equals the
evaluator's internal batch size, the evaluator's internal loop runs exactly
once per call: batching, padding and ordering are preserved bit-for-bit.

Within a chunk the hook may fire many times -- autoregressive decoding calls the
model once per step on the same batch -- and every firing sees the SAME chunk
ids. The per-item perturbation is therefore frozen across the trajectory by
construction. No cursor is advanced anywhere, and item identity is never
inferred from phoneme content (homophones make that unsafe): ids come from the
explicit global population indices.

This file is used ONLY for k>0. The k=0 intact-control path still calls
`evaluators.evaluate_endpoints` unchanged, which is why `evaluators.py` and
`injection.py` are byte-identical to the base commit.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence

from . import evaluators, injection

ROUTES = evaluators.ROUTES


def canonical_batch_size() -> int:
    """`evaluate_forms_ar`'s own module constant. Recovered, never assumed."""
    from scripts.evaluate_train_lexicon_ceiling import BATCH_SIZE
    return int(BATCH_SIZE)


def free_ar_batch_size() -> int:
    """`prelesion_eval.free_ar_items` default batch size, read from the signature."""
    import inspect

    import prelesion_eval
    return int(inspect.signature(prelesion_eval.free_ar_items)
               .parameters["batch_size"].default)


def naming_batch_size() -> int:
    import inspect

    from scripts.naming_comprehension.train_tasks import evaluate_naming
    return int(inspect.signature(evaluate_naming)
               .parameters["batch_size"].default)


def comprehension_batch_size() -> int:
    import inspect

    from scripts.naming_comprehension.train_tasks import (
        evaluate_comprehension_subset)
    return int(inspect.signature(evaluate_comprehension_subset)
               .parameters["batch_size"].default)


def evaluator_batch_sizes() -> Dict[str, int]:
    return {
        "CANONICAL_FORCED_LENGTH_AR": canonical_batch_size(),
        "GENUINE_FREE_AR": free_ar_batch_size(),
        "SEMANTIC_GREEDY_AR_GLOBAL_CAP": naming_batch_size(),
        "STRICT_TOP1_RETRIEVAL": comprehension_batch_size(),
    }


def chunks(seq: Sequence[int], size: int) -> List[List[int]]:
    """Contiguous, order-preserving partition. The final chunk may be partial."""
    if size <= 0:
        raise ValueError("batch size must be positive")
    return [list(seq[i:i + size]) for i in range(0, len(seq), size)]


def evaluate_endpoints_lesioned(tr, model, endpoints, bank_indices,
                                comp_indices, site: str,
                                make_eta: Callable[[List[str]], Callable]
                                ) -> List[Dict]:
    """Frozen item rows for a NONZERO lesion cell.

    `make_eta(batch_item_ids) -> eta_fn` is built per chunk from exactly that
    chunk's GLOBAL item ids, in the evaluator's own order.
    """
    pe = evaluators._pe()
    sizes = evaluator_batch_sizes()
    wanted = {e.decoding_convention for e in endpoints}
    routes_needed = tuple(sorted({e.route for e in endpoints
                                  if e.decoding_convention in
                                  ("CANONICAL_FORCED_LENGTH_AR",
                                   "GENUINE_FREE_AR")})) or ROUTES
    rows: List[Dict] = []

    if "CANONICAL_FORCED_LENGTH_AR" in wanted:
        for chunk in chunks(bank_indices, sizes["CANONICAL_FORCED_LENGTH_AR"]):
            ids = [evaluators.item_id(i) for i in chunk]
            with injection.activation_injection(model, site, make_eta(ids)):
                can_rows, _ = pe.canonical_items(tr, model, chunk,
                                                 routes=routes_needed)
            for pos, i in enumerate(chunk):
                r = can_rows[pos]
                for route in routes_needed:
                    rows.append({
                        "item_id": evaluators.item_id(i), "task": "repetition",
                        "route": route,
                        "decoding_convention": "CANONICAL_FORCED_LENGTH_AR",
                        "target": r.get("target") or r.get("target_str") or "",
                        "prediction": r.get(f"{route}_pred")
                                      or r.get(f"{route}_prediction") or "",
                        "correct": int(r[f"{route}_exact_match"]),
                    })

    if "GENUINE_FREE_AR" in wanted:
        for chunk in chunks(bank_indices, sizes["GENUINE_FREE_AR"]):
            ids = [evaluators.item_id(i) for i in chunk]
            ents = [{"item_id": evaluators.item_id(i),
                     "phonemes": list(tr.entries[i].phonemes)} for i in chunk]
            with injection.activation_injection(model, site, make_eta(ids)):
                far = pe.free_ar_items(tr, model, ents, routes=routes_needed)
            for route in routes_needed:
                for x in far[route]:
                    rows.append({
                        "item_id": x["item_id"], "task": "repetition",
                        "route": route, "decoding_convention": "GENUINE_FREE_AR",
                        "target": x["target_str"],
                        "prediction": x["predicted_str"],
                        "correct": int(x["exact"]),
                    })

    if "SEMANTIC_GREEDY_AR_GLOBAL_CAP" in wanted:
        from scripts.naming_comprehension.train_joint_scratch import (
            FREE_AR_MAX_STEPS)
        from scripts.naming_comprehension.train_tasks import evaluate_naming
        for chunk in chunks(bank_indices, sizes["SEMANTIC_GREEDY_AR_GLOBAL_CAP"]):
            ids = [evaluators.item_id(i) for i in chunk]
            with injection.activation_injection(model, site, make_eta(ids)):
                nm = evaluate_naming(model, tr.vocab, tr.entries, tr.bank_raw,
                                     list(chunk), "cpu", FREE_AR_MAX_STEPS,
                                     return_per_item=True)
            for i, r in zip(chunk, nm.pop("_per_item", [])):
                rows.append({
                    "item_id": evaluators.item_id(i), "task": "naming",
                    "route": "full",
                    "decoding_convention": "SEMANTIC_GREEDY_AR_GLOBAL_CAP",
                    "target": r.get("target", ""),
                    "prediction": r.get("prediction", ""),
                    "correct": int(r.get("exact_match", r.get("exact", 0))),
                })

    if "STRICT_TOP1_RETRIEVAL" in wanted:
        from scripts.naming_comprehension.train_tasks import (
            evaluate_comprehension_subset)
        for chunk in chunks(comp_indices, sizes["STRICT_TOP1_RETRIEVAL"]):
            ids = [evaluators.item_id(i) for i in chunk]
            with injection.activation_injection(model, site, make_eta(ids)):
                cm = evaluate_comprehension_subset(
                    model, tr.vocab, tr.entries, tr.bank_raw, list(chunk),
                    "cpu", sizes["STRICT_TOP1_RETRIEVAL"], return_per_item=True)
            for i, r in zip(chunk, cm.pop("_per_item", [])):
                rows.append({
                    "item_id": evaluators.item_id(i), "task": "comprehension",
                    "route": "full",
                    "decoding_convention": "STRICT_TOP1_RETRIEVAL",
                    "target": str(r.get("target", i)),
                    "prediction": str(r.get("top1", r.get("prediction", ""))),
                    "correct": int(r.get("correct", r.get("top1_correct", 0))),
                })
    return rows
