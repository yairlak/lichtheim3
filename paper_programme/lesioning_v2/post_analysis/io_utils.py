"""Shared read-only I/O helpers for post-analysis.

The only writes this package performs are under --out-dir, and `guard_out_dir`
refuses an --out-dir equal to or inside the results parent.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from typing import Dict, Iterable, List, Sequence

SHARD_NAMES = tuple(f"shard_{i:02d}" for i in range(12))


class AnalysisError(RuntimeError):
    pass


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def guard_out_dir(out_dir: str, results_parent: str) -> str:
    """Refuse an output directory that would write into the results namespace."""
    out = os.path.abspath(out_dir)
    root = os.path.abspath(results_parent)
    if out == root:
        raise AnalysisError(
            "REFUSED: --out-dir is the scientific results parent; analysis "
            "output must never be written there")
    if out.startswith(root + os.sep):
        raise AnalysisError(
            f"REFUSED: --out-dir {out} lies inside the scientific results "
            f"namespace {root}")
    return out


def write_tsv(path: str, rows: Sequence[Dict], columns: Sequence[str]) -> str:
    """Deterministic TSV: fixed column order, caller-sorted rows, \\n endings."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), delimiter="\t",
                           extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c))
                        for c in columns})
    return sha256_file(path)


def read_tsv(path: str) -> List[Dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def read_items(cell_dir: str) -> Iterable[Dict]:
    """Stream the finalized item rows of one cell. Read-only."""
    p = os.path.join(cell_dir, "items.jsonl")
    if not os.path.exists(p):
        raise AnalysisError(f"missing items.jsonl in {cell_dir}")
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def fnum(x):
    return None if x in (None, "", "None") else float(x)
