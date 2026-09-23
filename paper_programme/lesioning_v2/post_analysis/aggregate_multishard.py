#!/usr/bin/env python
"""Multi-shard, route-aware endpoint aggregation. READ-ONLY.

Reads the twelve shard roots, verifies every COMPLETE cell against its own
marker hashes using the frozen cell API, reconciles them against the
authoritative run matrix, and reduces the frozen item rows to the frozen
exact-match metric -- grouped by the scientific endpoint triple so that the
three repetition routes stay separate.

    python aggregate_multishard.py --results-parent <path> --out-dir <path>
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from paper_programme.lesioning_v2.execution import cells, preflight  # noqa: E402
from paper_programme.lesioning_v2.post_analysis import (  # noqa: E402
    comprehension_repair, endpoints, io_utils)
from paper_programme.lesioning_v2.post_analysis.io_utils import AnalysisError  # noqa: E402

CANONICAL_COLUMNS = (
    "state_id", "site", "severity_k", "severity_s", "realization",
    "endpoint", "tier", "route", "task", "decoding_convention",
    "n", "correct", "exact_match", "correctness_source",
)
EXPECTED_CELLS = 1812
EXPECTED_K0 = 12
EXPECTED_NONZERO = 1800
EXPECTED_ROWS = EXPECTED_CELLS * endpoints.EXPECTED_ENDPOINTS_PER_CELL  # 14,496


def discover_shards(results_parent: str) -> List[str]:
    if not os.path.isdir(results_parent):
        raise AnalysisError(f"results parent does not exist: {results_parent}")
    present = sorted(d for d in os.listdir(results_parent)
                     if os.path.isdir(os.path.join(results_parent, d))
                     and d.startswith("shard_"))
    if tuple(present) != io_utils.SHARD_NAMES:
        raise AnalysisError(
            f"expected exactly {list(io_utils.SHARD_NAMES)}, found {present}")
    return [os.path.join(results_parent, d) for d in present]


def lifecycle_counts(results_parent: str) -> Dict[str, int]:
    staging = failed = 0
    for dp, dirs, _ in os.walk(results_parent):
        for d in dirs:
            if d.startswith(cells.STAGING_PREFIX):
                staging += 1
            elif d.startswith(cells.FAILED_PREFIX):
                failed += 1
    return {"staging": staging, "failed": failed}


def collect_cells(results_parent: str, verify: bool = True) -> List[Dict]:
    found = []
    for shard in discover_shards(results_parent):
        for m in cells.read_complete_cells(shard):
            if verify:
                cells.verify_cell_integrity(m)
            m["_shard"] = os.path.basename(shard)
            found.append(m)
    return found


def reconcile(found: List[Dict], matrix: Dict) -> Dict:
    by_id = {cells.cell_identity(r): r for r in matrix["cells"]}
    seen, dupes = {}, []
    for m in found:
        cid = m["cell_identity"]
        if cid in seen:
            dupes.append(m["cell_key"])
        seen[cid] = m
        if cid not in by_id:
            raise AnalysisError(
                f"COMPLETE cell {m['cell_key']} matches no authoritative row")
    if dupes:
        raise AnalysisError(f"duplicate cell identities: {sorted(dupes)[:5]}")
    missing = sorted(set(by_id) - set(seen))
    if missing:
        raise AnalysisError(
            f"{len(missing)} authoritative rows have no COMPLETE cell; "
            f"first: {[by_id[c]['state_id'] + '/' + by_id[c]['site'] for c in missing[:3]]}")
    k0 = sum(1 for cid in seen if int(by_id[cid]["severity_k"]) == 0)
    nz = len(seen) - k0
    if len(seen) != EXPECTED_CELLS or k0 != EXPECTED_K0 or nz != EXPECTED_NONZERO:
        raise AnalysisError(
            f"cell census mismatch: total={len(seen)} k0={k0} nonzero={nz}; "
            f"expected {EXPECTED_CELLS}/{EXPECTED_K0}/{EXPECTED_NONZERO}")
    return {"by_identity": seen, "matrix_by_identity": by_id,
            "n_cells": len(seen), "n_k0": k0, "n_nonzero": nz}


def endpoint_rows_for_cell(marker: Dict, row: Dict,
                           counters: Dict = None) -> List[Dict]:
    """Reduce one cell's item rows to its 8 endpoint rows.

    Grouping is on the scientific triple, so each repetition route keeps its
    OWN denominator; routes are never pooled.

    For the frozen c_top1 endpoint the correctness bit is taken from the
    serialized `prediction` field, which is where the execution adapters
    actually stored it (see `comprehension_repair`). Every other endpoint uses
    its stored `correct` untouched. The metric itself is unchanged.
    """
    acc: Dict[tuple, Dict] = {}
    for r in io_utils.read_items(marker["_dir"]):
        triple = (r["task"], r["route"], r["decoding_convention"])
        e = endpoints.resolve(*triple)
        eff, source = comprehension_repair.effective_correct(r, counters)
        a = acc.setdefault(triple, {"endpoint": e.key, "tier": e.tier,
                                    "task": e.task, "route": e.route,
                                    "decoding_convention": e.decoding_convention,
                                    "n": 0, "correct": 0,
                                    "correctness_source": source})
        if a["correctness_source"] != source:
            raise AnalysisError(
                f"{marker['cell_key']}: {e.key} mixes correctness sources")
        a["n"] += 1
        a["correct"] += eff
    if len(acc) != endpoints.EXPECTED_ENDPOINTS_PER_CELL:
        raise AnalysisError(
            f"{marker['cell_key']}: found {len(acc)} endpoint groups, expected "
            f"{endpoints.EXPECTED_ENDPOINTS_PER_CELL}")
    keys = sorted(a["endpoint"] for a in acc.values())
    if len(set(keys)) != len(keys):
        raise AnalysisError(f"{marker['cell_key']}: duplicate endpoint")
    if set(keys) != set(endpoints.PRIMARY_KEYS) | set(endpoints.DIAGNOSTIC_KEYS):
        raise AnalysisError(
            f"{marker['cell_key']}: endpoint set {keys} != frozen battery")
    out = []
    for a in acc.values():
        out.append({
            "state_id": row["state_id"], "site": row["site"],
            "severity_k": int(row["severity_k"]),
            "severity_s": row["severity_s"],
            "realization": int(row["realization"]),
            "n": a["n"], "correct": a["correct"],
            "exact_match": (a["correct"] / a["n"]) if a["n"] else None,
            **{k: a[k] for k in ("endpoint", "tier", "task", "route",
                                 "decoding_convention", "correctness_source")},
        })
    return out


def build(results_parent: str, verify: bool = True) -> Dict:
    matrix = preflight.load_matrix()
    found = collect_cells(results_parent, verify=verify)
    rec = reconcile(found, matrix)
    life = lifecycle_counts(results_parent)
    if life["staging"] or life["failed"]:
        raise AnalysisError(
            f"lifecycle not clean: staging={life['staging']} "
            f"failed={life['failed']}")
    counters = comprehension_repair.new_counters()
    rows: List[Dict] = []
    for cid, marker in rec["by_identity"].items():
        rows.extend(endpoint_rows_for_cell(
            marker, rec["matrix_by_identity"][cid], counters))
    if len(rows) != EXPECTED_ROWS:
        raise AnalysisError(
            f"produced {len(rows)} endpoint rows, expected {EXPECTED_ROWS}")
    rows.sort(key=lambda r: (r["state_id"], r["site"], r["severity_k"],
                             r["realization"], r["endpoint"]))
    recovered = sum(1 for r in rows
                    if r["correctness_source"] == comprehension_repair.RECOVERED)
    return {"rows": rows, "matrix": matrix, "census": rec, "lifecycle": life,
            "n_shards": len(io_utils.SHARD_NAMES),
            "comprehension_counters": counters,
            "comprehension_repair": comprehension_repair.repair_record(
                counters, recovered)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Route-aware multi-shard aggregation.")
    ap.add_argument("--results-parent", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    out = io_utils.guard_out_dir(a.out_dir, a.results_parent)
    try:
        res = build(a.results_parent)
    except AnalysisError as e:
        print(f"ANALYSIS_FAIL\n{e}", file=sys.stderr)
        return 2
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "CELL_ENDPOINT_EXACT_MATCH.tsv")
    sha = io_utils.write_tsv(p, res["rows"], CANONICAL_COLUMNS)
    print(f"cells={res['census']['n_cells']} k0={res['census']['n_k0']} "
          f"nonzero={res['census']['n_nonzero']}")
    print(f"rows={len(res['rows'])} (expected {EXPECTED_ROWS})")
    print(f"written: {p}\nsha256={sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
