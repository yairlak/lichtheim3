"""Binding of the frozen battery to validated evaluator implementations.

NOTHING is reimplemented here. Each frozen endpoint is dispatched to the exact
validated function recovered by archaeology (see the integration manifest):

    CANONICAL_FORCED_LENGTH_AR    prelesion_eval.canonical_items
                                  -> evaluate_train_lexicon_ceiling.evaluate_forms_ar
    GENUINE_FREE_AR               prelesion_eval.free_ar_items
                                  -> train_joint_scratch.free_ar_repetition semantics
    SEMANTIC_GREEDY_AR_GLOBAL_CAP train_tasks.evaluate_naming(return_per_item=True)
    STRICT_TOP1_RETRIEVAL         train_tasks.evaluate_comprehension_subset(...)

Item identity is the bank index, rendered "bank_{i}", matching the V7
pre-lesion pipeline exactly.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

ROUTES = ("full", "wm", "ltm")


def item_id(bank_index: int) -> str:
    return f"bank_{int(bank_index)}"


def _pe():
    import prelesion_eval
    return prelesion_eval


def evaluate_endpoints(tr, model, endpoints, bank_indices: Sequence[int],
                       comp_indices: Sequence[int]) -> List[Dict]:
    """Return frozen item rows for the requested endpoints.

    Rows carry only: item_id, task, decoding_convention, route, target,
    prediction, correct. All provenance is attached by the caller from the
    authoritative matrix row.
    """
    pe = _pe()
    rows: List[Dict] = []
    wanted = {e.decoding_convention for e in endpoints}
    routes_needed = tuple(sorted({e.route for e in endpoints
                                  if e.decoding_convention in
                                  ("CANONICAL_FORCED_LENGTH_AR",
                                   "GENUINE_FREE_AR")}))

    if "CANONICAL_FORCED_LENGTH_AR" in wanted:
        can_rows, _ = pe.canonical_items(tr, model, bank_indices,
                                         routes=routes_needed or ROUTES)
        for pos, i in enumerate(bank_indices):
            r = can_rows[pos]
            for route in (routes_needed or ROUTES):
                rows.append({
                    "item_id": item_id(i), "task": "repetition", "route": route,
                    "decoding_convention": "CANONICAL_FORCED_LENGTH_AR",
                    "target": r.get("target") or r.get("target_str") or "",
                    "prediction": r.get(f"{route}_pred")
                                  or r.get(f"{route}_prediction") or "",
                    "correct": int(r[f"{route}_exact_match"]),
                })

    if "GENUINE_FREE_AR" in wanted:
        ents = [{"item_id": item_id(i),
                 "phonemes": list(tr.entries[i].phonemes)} for i in bank_indices]
        far = pe.free_ar_items(tr, model, ents,
                               routes=routes_needed or ROUTES)
        for route in (routes_needed or ROUTES):
            for x in far[route]:
                rows.append({
                    "item_id": x["item_id"], "task": "repetition", "route": route,
                    "decoding_convention": "GENUINE_FREE_AR",
                    "target": x["target_str"], "prediction": x["predicted_str"],
                    "correct": int(x["exact"]),
                })

    if "SEMANTIC_GREEDY_AR_GLOBAL_CAP" in wanted:
        from scripts.naming_comprehension.train_joint_scratch import FREE_AR_MAX_STEPS
        from scripts.naming_comprehension.train_tasks import evaluate_naming
        nm = evaluate_naming(model, tr.vocab, tr.entries, tr.bank_raw,
                             list(bank_indices), "cpu", FREE_AR_MAX_STEPS,
                             return_per_item=True)
        for i, r in zip(bank_indices, nm.pop("_per_item", [])):
            rows.append({
                "item_id": item_id(i), "task": "naming", "route": "full",
                "decoding_convention": "SEMANTIC_GREEDY_AR_GLOBAL_CAP",
                "target": r.get("target", ""), "prediction": r.get("prediction", ""),
                "correct": int(r.get("exact_match", r.get("exact", 0))),
            })

    if "STRICT_TOP1_RETRIEVAL" in wanted:
        from scripts.naming_comprehension.train_tasks import (
            evaluate_comprehension_subset)
        cm = evaluate_comprehension_subset(model, tr.vocab, tr.entries,
                                           tr.bank_raw, list(comp_indices),
                                           "cpu", 512, return_per_item=True)
        for i, r in zip(comp_indices, cm.pop("_per_item", [])):
            rows.append({
                "item_id": item_id(i), "task": "comprehension", "route": "full",
                "decoding_convention": "STRICT_TOP1_RETRIEVAL",
                "target": str(r.get("target", i)),
                "prediction": str(r.get("top1", r.get("prediction", ""))),
                "correct": int(r.get("correct", r.get("top1_correct", 0))),
            })
    return rows
