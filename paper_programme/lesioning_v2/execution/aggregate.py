"""Minimum frozen aggregation over COMPLETE cells only.

Reads finalized cells, re-verifies their own file hashes, and reduces the frozen
item rows to the frozen summary key. It calculates nothing that the evaluator
contract has not already frozen, and it interprets nothing.

Explicitly NOT done here: choosing a severity, collapsing model roles, pooling
P1-P4, defining a success criterion, adding significance tests.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

from . import cells

SUMMARY_KEY = ("state_id", "site", "severity_k", "realization", "task",
               "decoding_convention")


class AggregationError(RuntimeError):
    pass


def collect(root: str, verify: bool = True) -> List[Dict]:
    out = []
    for m in cells.read_complete_cells(root):
        if verify:
            cells.verify_cell_integrity(m)
        out.append(m)
    return out


def aggregate(root: str, expected_cells: int = None) -> Dict:
    """Reduce finalized cells to per-(cell, task, decoder) accuracy.

    The only quantity computed is the frozen exact-match rate: correct / n,
    derived directly from the frozen item rows.
    """
    complete = collect(root)
    rows: Dict[tuple, Dict] = {}
    for m in complete:
        items_path = os.path.join(m["_dir"], "items.jsonl")
        with open(items_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                key = tuple(r[k] for k in SUMMARY_KEY)
                acc = rows.setdefault(key, {**dict(zip(SUMMARY_KEY, key)),
                                            "n": 0, "correct": 0})
                acc["n"] += 1
                acc["correct"] += int(r["correct"])
    for acc in rows.values():
        acc["exact_match"] = acc["correct"] / acc["n"] if acc["n"] else None
    out = {
        "n_complete_cells": len(complete),
        "n_summary_rows": len(rows),
        "incomplete_cells_excluded": True,
        "summary_key": list(SUMMARY_KEY),
        "metric": "exact_match = correct / n (frozen; nothing else computed)",
        "rows": [rows[k] for k in sorted(rows)],
    }
    if expected_cells is not None:
        out["expected_cells"] = expected_cells
        out["complete"] = len(complete) == expected_cells
    return out
