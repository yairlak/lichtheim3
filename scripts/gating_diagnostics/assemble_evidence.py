"""Assemble the combined item-level evidence table from the per-state shards.

Packaging only.  This concatenates the shards the runner already wrote, adding a
`state_id` column.  It computes no new scientific quantity, re-decodes nothing,
and changes no evaluator semantics: the shards remain the primary artifacts and
this file is a convenience view over them, required by the execution contract
(PHASE 3) as `item_level_gate_route_metrics.tsv`.

Hard-stops if any shard is a smoke artifact or if a state is missing.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

STATE_ORDER = ["W1_SRC", "W1_REP", "W2_SRC", "W2_REP",
               "W3_SRC", "W3_REP", "W4_SRC", "W4_REP"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "paper_programme", "gating_route_diagnostics"))
    a = ap.parse_args(argv)
    src = os.path.join(a.out_dir, "figure_source_data")

    shards = {}
    for p in sorted(glob.glob(os.path.join(src, "item_level_*.tsv"))):
        sid = os.path.basename(p)[len("item_level_"):-len(".tsv")]
        if "SMOKE" in sid:
            print(f"HARD STOP: smoke shard present: {p}", file=sys.stderr)
            return 2
        shards[sid] = p

    missing = [s for s in STATE_ORDER if s not in shards]
    if missing:
        print(f"HARD STOP: missing state shards: {missing}", file=sys.stderr)
        return 2
    extra = [s for s in shards if s not in STATE_ORDER]
    if extra:
        print(f"HARD STOP: unexpected shards: {extra}", file=sys.stderr)
        return 2

    out = os.path.join(a.out_dir, "item_level_gate_route_metrics.tsv")
    total = 0
    writer = None
    with open(out, "w", newline="") as fo:
        for sid in STATE_ORDER:
            with open(shards[sid], newline="") as fi:
                rd = csv.DictReader(fi, delimiter="\t")
                for row in rd:
                    if writer is None:
                        writer = csv.DictWriter(
                            fo, fieldnames=["state_id"] + list(row.keys()),
                            delimiter="\t")
                        writer.writeheader()
                    writer.writerow({"state_id": sid, **row})
                    total += 1
            print(f"  {sid}: appended")
    print(f"wrote {out}  ({total} rows across {len(STATE_ORDER)} states)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
