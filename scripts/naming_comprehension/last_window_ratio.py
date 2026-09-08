"""The last-window C residual ratio, in the ONE definition the audits use.

    per-seed   ratio_s = C_errors(u_final) / C_errors(u_prev)
    reported   mean over seeds of ratio_s

This is a MEAN OF SEED RATIOS, not a ratio of pooled means.  On the
u1350 -> u1400 window the two differ (1.027615 vs 1.025445) and the published
figure was 1.0276, so the distinction is not cosmetic.  Pooled counts across
seeds are not independent observations and are never used here.

The value feeds `base123_persistence_report.py --last-window-ratio`, whose
preregistered condition is `>= 0.85`.  This script only COMPUTES the number;
it changes no threshold and makes no decision.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys
from typing import List, Optional

R_PASS, CYCLE, N_COMP = 463, 6, 27_981
TRIGGER_THRESHOLD = 0.85          # preregistered; never changed here


def steps_for_u(u: float) -> int:
    s = u * R_PASS * CYCLE
    if abs(s - round(s)) > 1e-6:
        raise SystemExit(f"u={u} is not a whole number of optimizer steps")
    return int(round(s))


def comp_errors_at(runs_root: str, run_id: str, step: int) -> Optional[int]:
    """C error count at one scheduled FULL evaluation, from that run's own
    logged column where present, else recomputed from the rate."""
    p = os.path.join(runs_root, run_id, "metrics.tsv")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if int(r["step"]) != step:
                continue
            logged = r.get("full_comp_errors")
            if logged not in (None, "", "nan"):
                return int(float(logged))
            top1 = r.get("full_comp_top1")
            if top1 in (None, "", "nan"):
                return None
            return int(round((1.0 - float(top1)) * N_COMP))
    return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--run-template", required=True,
                    help="e.g. 'final_canneal_ctrl_h512_s{seed}'")
    ap.add_argument("--seeds", default="19,20,21,22")
    ap.add_argument("--u-prev", type=float, required=True)
    ap.add_argument("--u-final", type=float, required=True)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args(argv)

    seeds = [int(s) for s in a.seeds.split(",")]
    sp, sf = steps_for_u(a.u_prev), steps_for_u(a.u_final)
    print(f"[window] u{a.u_prev:g} (step {sp}) -> u{a.u_final:g} (step {sf})")

    per, missing, zeros = [], [], []
    for s in seeds:
        rid = a.run_template.format(seed=s)
        e0 = comp_errors_at(a.runs_root, rid, sp)
        e1 = comp_errors_at(a.runs_root, rid, sf)
        if e0 is None or e1 is None:
            missing.append((s, rid, e0, e1))
            print(f"  seed {s:>2}  {rid}: MISSING "
                  f"(prev={e0}, final={e1}) -- no full eval at that step?")
            continue
        if e0 == 0:
            zeros.append(s)
            print(f"  seed {s:>2}  {rid}: prev = 0 errors, ratio undefined")
            continue
        r = e1 / e0
        per.append({"seed": s, "run_id": rid, "C_err_prev": e0,
                    "C_err_final": e1, "ratio": round(r, 6)})
        print(f"  seed {s:>2}  {e0:>5} -> {e1:>5}   ratio {r:.6f}")

    if not per:
        print("[window] no usable seed -- refusing to report a ratio")
        return 1
    ratios = [p["ratio"] for p in per]
    mean = st.mean(ratios)
    out = {"u_prev": a.u_prev, "u_final": a.u_final,
           "step_prev": sp, "step_final": sf,
           "run_template": a.run_template, "seeds": seeds,
           "definition": "mean over seeds of C_errors(u_final)/C_errors(u_prev)",
           "per_seed": per, "n_seeds_used": len(per),
           "mean_of_seed_ratios": round(mean, 6),
           "sd_of_seed_ratios": round(st.stdev(ratios), 6) if len(ratios) > 1
                                else 0.0,
           # reported only so a reader can see it is NOT what we use
           "ratio_of_pooled_means_NOT_USED": round(
               sum(p["C_err_final"] for p in per)
               / sum(p["C_err_prev"] for p in per), 6),
           "missing_seeds": [m[0] for m in missing],
           "seeds_with_zero_prev": zeros,
           "trigger_threshold_ge": TRIGGER_THRESHOLD,
           "condition_last_window_ratio_ge_0.85": bool(mean >= TRIGGER_THRESHOLD)}
    print(f"[window] MEAN OF SEED RATIOS = {mean:.6f}   (n={len(per)} seeds)")
    print(f"[window]   ratio of pooled means, NOT used: "
          f"{out['ratio_of_pooled_means_NOT_USED']:.6f}")
    print(f"[window]   preregistered condition ratio >= {TRIGGER_THRESHOLD}: "
          + ("MET" if out["condition_last_window_ratio_ge_0.85"]
             else "NOT met"))
    print(f"[window] pass to the persistence report as:\n"
          f"           --last-window-ratio {mean:.6f}")
    if a.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
        json.dump(out, open(a.out_json, "w"), indent=1)
        print(f"[window] wrote {a.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
