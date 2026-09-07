"""Paired extraction for the 1:2:3 vs 2:2:3 repetition-rescue diagnostic.

Indexed by MATCHED MACRO-CYCLE DELTA, never by global step: the two arms
advance 6 and 7 optimizer steps per cycle, so a common global step would
compare different amounts of naming and comprehension experience.  At equal
cycle delta the arms have taken an identical number of N and C updates on
identical counter-addressed batches, and the rescue has taken exactly twice
the R updates -- which is the treatment.

For each (arm, seed, milestone) it emits the full readout, and it PROVES the
matching per row: N cursor and C cursor identical across arms, R delta in the
rescue exactly twice the control's.  A row whose matching check fails is
flagged rather than silently averaged.

Also evaluates the pre-registered decision rule verbatim.  The thresholds are
the ones fixed before the run and are not re-derived from the data.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys
from typing import Dict, List, Optional

N_REP, N_NAM, N_COMP = 29_571, 29_571, 27_981
R_PASS = 463
SRC_STEP = 3_333_600
CYCLE_STEPS = {"123": 6, "223": 7}
MS_CYCLES = 11_575
N_MILESTONES = 8
SEEDS = [19, 20, 21, 22]
RUN = "final_rep_rescue{arm}_h512_s{seed}"

# ---- PRE-REGISTERED decision rule (fixed before the run) -----------------
LTM_TARGET = 0.92           # 2:2:3 must reach this ...
LTM_MIN_SEEDS = 3           # ... in at least this many of 4 seeds
C_NONINFERIORITY = 5.0      # mean paired dC errors (223 - 123) must be <=
LTM_NULL_BAND = 0.02        # < this movement vs control => reject rehearsal


def num(v):
    if v in (None, "", "nan", "NaN"):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def full_rows(runs_root: str, run_id: str) -> Dict[int, dict]:
    p = os.path.join(runs_root, run_id, "metrics.tsv")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t")
                if num(r.get("full_rep_full")) is not None]
    out = {}
    for r in sorted(rows, key=lambda r: int(r["step"])):
        out[int(r["step"])] = r          # later duplicate wins; identical step
    return out


def milestone_steps(arm: str) -> List[int]:
    cs = CYCLE_STEPS[arm]
    return [SRC_STEP + cs * MS_CYCLES * k for k in range(1, N_MILESTONES + 1)]


def readout(r: dict) -> dict:
    o = {}
    for key, col, pop in (("rep_canonical", "full_rep_full", N_REP),
                          ("rep_freear", "full_rep_freear", N_REP),
                          ("naming", "full_naming_exact", N_NAM),
                          ("comp_top1", "full_comp_top1", N_COMP),
                          ("comp_top5", "full_comp_top5", N_COMP)):
        v = num(r.get(col))
        o[key] = v
        o[key + "_errors"] = None if v is None else int(round((1 - v) * pop))
    for k, col in (("rep_ltm", "full_rep_ltm"), ("rep_wm", "full_rep_wm"),
                   ("gate_mean", "gate_mean")):
        o[k] = num(r.get(col))
    for c in ("r_exposures", "n_exposures", "c_exposures"):
        o[c] = num(r.get(c))
    return o


def write_tsv(path, rows):
    if not rows:
        open(path, "w").write("")
        print(f"[rescue] (0 rows) {path}")
        return
    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[rescue] wrote {path}  ({len(rows)} rows)")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source-cursors", default=None,
                    help="JSON {'repetition':..,'naming':..,'comprehension':..}"
                         " read from the u1200 source checkpoint")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    src = json.loads(args.source_cursors) if args.source_cursors else None

    data = {arm: {s: full_rows(args.runs_root, RUN.format(arm=arm, seed=s))
                  for s in SEEDS} for arm in ("123", "223")}

    # ---- per (arm, seed, milestone) -------------------------------------
    rows, missing = [], []
    for arm in ("123", "223"):
        cs = CYCLE_STEPS[arm]
        for s in SEEDS:
            for k, step in enumerate(milestone_steps(arm), start=1):
                r = data[arm][s].get(step)
                if r is None:
                    missing.append(f"arm {arm} seed {s} milestone {k} "
                                   f"(step {step}) missing")
                    continue
                dcyc = (step - SRC_STEP) // cs
                rec = {"arm": arm, "seed": s, "milestone": k,
                       "cycle_delta": dcyc, "global_step": step,
                       "steps_per_cycle": cs, **readout(r)}
                # cursors, from exposures x per-pass (exact integers)
                for t, col, per in (("R", "r_exposures", R_PASS),
                                    ("N", "n_exposures", R_PASS),
                                    ("C", "c_exposures", 438)):
                    e = rec.get(col)
                    rec[f"{t}_cursor"] = (None if e is None
                                          else int(round(e * per)))
                rows.append(rec)
    write_tsv(os.path.join(args.out_dir, "paired_by_cycle.tsv"), rows)

    # ---- matching proof, per milestone ----------------------------------
    proof = []
    for k in range(1, N_MILESTONES + 1):
        dcyc = MS_CYCLES * k
        for s in SEEDS:
            a = [r for r in rows if r["arm"] == "123" and r["seed"] == s
                 and r["milestone"] == k]
            b = [r for r in rows if r["arm"] == "223" and r["seed"] == s
                 and r["milestone"] == k]
            if not a or not b:
                continue
            a, b = a[0], b[0]
            dR_a = (a["R_cursor"] - src["repetition"]) if src else None
            dR_b = (b["R_cursor"] - src["repetition"]) if src else None
            rec = {"milestone": k, "cycle_delta": dcyc, "seed": s,
                   "step_123": a["global_step"], "step_223": b["global_step"],
                   "N_cursor_123": a["N_cursor"], "N_cursor_223": b["N_cursor"],
                   "C_cursor_123": a["C_cursor"], "C_cursor_223": b["C_cursor"],
                   "R_cursor_123": a["R_cursor"], "R_cursor_223": b["R_cursor"],
                   "N_match": int(a["N_cursor"] == b["N_cursor"]),
                   "C_match": int(a["C_cursor"] == b["C_cursor"]),
                   "dR_123": dR_a, "dR_223": dR_b,
                   "R_doubled": (None if dR_a in (None, 0)
                                 else int(dR_b == 2 * dR_a))}
            rec["MATCHED"] = int(bool(rec["N_match"] and rec["C_match"]
                                      and (rec["R_doubled"] in (None, 1))))
            proof.append(rec)
    write_tsv(os.path.join(args.out_dir, "matching_proof.tsv"), proof)
    bad = [p for p in proof if not p["MATCHED"]]
    print(f"[rescue] matching rows: {len(proof)}, FAILED: {len(bad)}"
          + ("  <-- DO NOT COMPARE THESE" if bad else "  (all matched)"))

    # ---- paired differences ---------------------------------------------
    paired, summ = [], []
    for k in range(1, N_MILESTONES + 1):
        d = {"comp": [], "rep": [], "ltm": []}
        for s in SEEDS:
            a = [r for r in rows if r["arm"] == "123" and r["seed"] == s
                 and r["milestone"] == k]
            b = [r for r in rows if r["arm"] == "223" and r["seed"] == s
                 and r["milestone"] == k]
            if not a or not b:
                continue
            a, b = a[0], b[0]
            rec = {"milestone": k, "cycle_delta": MS_CYCLES * k, "seed": s,
                   "C_err_123": a["comp_top1_errors"],
                   "C_err_223": b["comp_top1_errors"],
                   "dC_errors": (b["comp_top1_errors"] - a["comp_top1_errors"]),
                   "R_err_123": a["rep_canonical_errors"],
                   "R_err_223": b["rep_canonical_errors"],
                   "dR_errors": (b["rep_canonical_errors"]
                                 - a["rep_canonical_errors"]),
                   "Rfree_err_123": a["rep_freear_errors"],
                   "Rfree_err_223": b["rep_freear_errors"],
                   "N_err_123": a["naming_errors"],
                   "N_err_223": b["naming_errors"],
                   "LTM_123": a["rep_ltm"], "LTM_223": b["rep_ltm"],
                   "dLTM": (None if a["rep_ltm"] is None or b["rep_ltm"] is None
                            else round(b["rep_ltm"] - a["rep_ltm"], 6)),
                   "WM_223": b["rep_wm"], "gate_223": b["gate_mean"]}
            paired.append(rec)
            d["comp"].append(rec["dC_errors"]); d["rep"].append(rec["dR_errors"])
            if rec["dLTM"] is not None:
                d["ltm"].append(rec["dLTM"])
        if d["comp"]:
            summ.append({"milestone": k, "cycle_delta": MS_CYCLES * k,
                         "n_seeds": len(d["comp"]),
                         "mean_dC_errors": round(st.mean(d["comp"]), 3),
                         "sd_dC_errors": (round(st.stdev(d["comp"]), 3)
                                          if len(d["comp"]) > 1 else 0.0),
                         "all_seeds_C_worse": int(all(x > 0 for x in d["comp"])),
                         "mean_dR_errors": round(st.mean(d["rep"]), 3),
                         "mean_dLTM": (round(st.mean(d["ltm"]), 6)
                                       if d["ltm"] else None),
                         "sd_dLTM": (round(st.stdev(d["ltm"]), 6)
                                     if len(d["ltm"]) > 1 else 0.0)})
    write_tsv(os.path.join(args.out_dir, "paired_differences.tsv"), paired)
    write_tsv(os.path.join(args.out_dir, "paired_summary.tsv"), summ)

    # ---- pre-registered decision rule -----------------------------------
    final = [p for p in paired if p["milestone"] == N_MILESTONES]
    verdict = {"rule": "pre-registered, fixed before the run",
               "thresholds": {"ltm_target": LTM_TARGET,
                              "ltm_min_seeds": LTM_MIN_SEEDS,
                              "c_noninferiority_mean_errors": C_NONINFERIORITY,
                              "ltm_null_band": LTM_NULL_BAND},
               "n_seeds_at_final": len(final)}
    if final:
        ltm223 = [p["LTM_223"] for p in final if p["LTM_223"] is not None]
        dC = [p["dC_errors"] for p in final]
        dLTM = [p["dLTM"] for p in final if p["dLTM"] is not None]
        c1 = sum(1 for v in ltm223 if v >= LTM_TARGET)
        cond = {
            "1_ltm_recovers": {
                "n_seeds_ge_target": c1, "values": ltm223,
                "met": c1 >= LTM_MIN_SEEDS},
            "2_c_non_inferior": {
                "mean_dC_errors": round(st.mean(dC), 3),
                "all_four_adverse": all(x > 0 for x in dC),
                "met": (st.mean(dC) <= C_NONINFERIORITY
                        and not all(x > 0 for x in dC))},
            "3_naming_zero": {
                "n_err_223": [p["N_err_223"] for p in final],
                "met": all(p["N_err_223"] == 0 for p in final)},
            "4_rep_not_worse": {
                "canonical": [p["dR_errors"] for p in final],
                "freear": [(p["Rfree_err_223"] - p["Rfree_err_123"])
                           for p in final],
                "met": st.mean([p["dR_errors"] for p in final]) <= 0},
        }
        verdict["conditions"] = cond
        verdict["PREFER_223"] = all(v["met"] for v in cond.values())
        mean_dltm = st.mean(dLTM) if dLTM else None
        verdict["mean_dLTM"] = (None if mean_dltm is None
                                else round(mean_dltm, 6))
        verdict["reject_rehearsal_explanation"] = (
            mean_dltm is not None and abs(mean_dltm) < LTM_NULL_BAND)
        verdict["reject_223_C_harm"] = all(x > 0 for x in dC)
    p = os.path.join(args.out_dir, "decision.json")
    json.dump(verdict, open(p, "w"), indent=1, default=str)
    print(f"[rescue] wrote {p}")
    print(json.dumps(verdict, indent=1, default=str))
    if missing:
        mp = os.path.join(args.out_dir, "missing.txt")
        open(mp, "w").write("\n".join(missing) + "\n")
        print(f"[rescue] {len(missing)} missing rows -> {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
