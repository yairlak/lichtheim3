"""V6 FIRST-HIT selector: the frozen prospective source-selection rule.

Reads a run's metrics.tsv, walks the FULL detector points in CHRONOLOGICAL
order, and returns the FIRST point satisfying

    Rcan_errors == 0  AND  Rfree_errors == 0  AND  Naming_errors == 0

C does NOT enter source eligibility.  There is no lookahead, no minimum-C
choice, no best-LTM choice, no latest-hit choice, and no comparison of repair
outcomes: the walk stops at the first eligible point and that checkpoint is
the selected source.  Applying "first in time order" to a complete log is
equivalent to an online trigger -- it cannot see the future -- which is why
this post-pass does not weaken the prospective claim.

Read-only: never writes into the run directory, never touches a checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

N_REP_POP = 29_571


def _f(row: dict, key: str):
    v = row.get(key)
    if v in (None, "", "nan", "NaN"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def detector_points(metrics_tsv: str) -> list:
    """Full detector rows, chronological, with the three eligibility counts."""
    with open(metrics_tsv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    pts = []
    for r in rows:
        rcan = _f(r, "full_rep_errors")
        if rcan is None:                      # not a full detector point
            continue
        rfree = _f(r, "full_rep_freear_errors")
        nam_exact = _f(r, "full_naming_exact")
        n_err = (None if nam_exact is None
                 else int(round((1.0 - nam_exact) * N_REP_POP)))
        pts.append({
            "step": int(float(r["step"])),
            "Rcan_errors": int(round(rcan)),
            "Rfree_errors": (None if rfree is None else int(round(rfree))),
            "Naming_errors": n_err,
            "C_errors": (None if _f(r, "full_comp_errors") is None
                         else int(round(_f(r, "full_comp_errors")))),
            "rep_ltm": _f(r, "full_rep_ltm"),
            "gate_mean": _f(r, "gate_mean"),
        })
    pts.sort(key=lambda p: p["step"])          # chronological, explicitly
    return pts


def eligible(p: dict) -> bool:
    return (p["Rcan_errors"] == 0 and p["Rfree_errors"] == 0
            and p["Naming_errors"] == 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps-per-u", type=int, default=2778)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args(argv)

    pts = detector_points(a.metrics)
    hit = None
    for p in pts:                              # FIRST in time order; stop there
        if eligible(p):
            hit = p
            break

    dec = {
        "seed": a.seed,
        "rule": ("FIRST detector point in chronological order with "
                 "Rcan=0 AND Rfree=0 AND Naming=0; C excluded from eligibility"),
        "n_detector_points": len(pts),
        "first_step": pts[0]["step"] if pts else None,
        "last_step": pts[-1]["step"] if pts else None,
        "n_eligible_total": sum(1 for p in pts if eligible(p)),
        "hit": hit is not None,
        "detector_points": pts,
    }
    if hit is not None:
        ck = os.path.join(a.run_dir, "checkpoints",
                          f"step_{hit['step']:08d}.pt")
        dec.update({
            "selected_step": hit["step"],
            "selected_u": hit["step"] / a.steps_per_u,
            "selected_checkpoint": ck,
            "selected_checkpoint_exists": os.path.exists(ck),
            "selected_metrics": hit,
            "n_points_before_hit": [p["step"] for p in pts].index(hit["step"]),
        })
    json.dump(dec, open(a.out_json, "w"), indent=1)

    if hit is None:
        print(f"== seed {a.seed}: NO_ELIGIBLE_HIT in {len(pts)} detector points",
              flush=True)
        best = min(pts, key=lambda p: (p["Rcan_errors"], p["Naming_errors"])) if pts else None
        if best:
            print(f"   best point: step {best['step']} Rcan={best['Rcan_errors']} "
                  f"Rfree={best['Rfree_errors']} N={best['Naming_errors']}", flush=True)
        return 1
    print(f"== seed {a.seed}: FIRST HIT at step {hit['step']} "
          f"(u{hit['step'] / a.steps_per_u:.0f}), "
          f"point {dec['n_points_before_hit'] + 1}/{len(pts)}: "
          f"Rcan=0 Rfree=0 N=0 C={hit['C_errors']} LTM={hit['rep_ltm']}", flush=True)
    print(f"   source checkpoint: {dec['selected_checkpoint']} "
          f"exists={dec['selected_checkpoint_exists']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
