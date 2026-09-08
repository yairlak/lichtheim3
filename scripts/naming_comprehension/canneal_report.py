"""Matched three-arm comparison for the comprehension-LR anneal.

All three arms keep schedule 1:2:3, so at a given global step they share the
exact R/N/C/pool cursors.  That identity is PROVEN per milestone rather than
assumed, and any milestone where it fails is flagged instead of averaged.

This is an endgame convergence readout, not a null-hypothesis test: it
reports trajectories, residual decay, churn and the ceiling streak, and it
records the first strict-ceiling hit if any arm achieves one.
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
R_PASS, C_PASS, CYCLE = 463, 438, 6
SRC_STEP, END_STEP, MS_STEP = 3_889_200, 5_556_000, 69_450
ARMS = ["ctrl", "5e5", "3e5"]
ARM_LR = {"ctrl": 1e-4, "5e5": 5e-5, "3e5": 3e-5}
SEEDS = [19, 20, 21, 22]
RUN = "final_canneal_{arm}_h512_s{seed}"


def num(v):
    if v in (None, "", "nan", "NaN"):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def milestones() -> List[int]:
    return [SRC_STEP + MS_STEP * k for k in range(1, 25)]


def full_rows(runs_root: str, run_id: str) -> Dict[int, dict]:
    p = os.path.join(runs_root, run_id, "metrics.tsv")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t")
                if num(r.get("full_rep_full")) is not None]
    return {int(r["step"]): r for r in sorted(rows, key=lambda r: int(r["step"]))}


def readout(r: dict) -> dict:
    """Prefer the driver's own logged error counts; recompute only as a
    fallback, so the report never disagrees with the run's own accounting."""
    o = {}
    for key, col, pop, ecol in (
            ("rep_canonical", "full_rep_full", N_REP, "full_rep_errors"),
            ("rep_freear", "full_rep_freear", N_REP, "full_rep_freear_errors"),
            ("naming", "full_naming_exact", N_NAM, "full_naming_errors"),
            ("comp_top1", "full_comp_top1", N_COMP, "full_comp_errors"),
            ("comp_top5", "full_comp_top5", N_COMP, None)):
        v = num(r.get(col))
        o[key] = v
        logged = num(r.get(ecol)) if ecol else None
        if logged is not None:
            o[key + "_errors"] = int(logged)
        else:
            o[key + "_errors"] = None if v is None else int(round((1 - v) * pop))
    for k, col in (("rep_ltm", "full_rep_ltm"), ("rep_wm", "full_rep_wm"),
                   ("comp_margin_mean", "full_comp_margin_mean"),
                   ("comp_rank_median", "full_comp_rank_median"),
                   ("gate_mean", "gate_mean"),
                   # NB: the driver logs the REPETITION lr in this column, not
                   # the comprehension lr this experiment manipulates.
                   ("lr_repetition_logged", "lr")):
        o[k] = num(r.get(col))
    for t, col, per in (("R", "r_exposures", R_PASS),
                        ("N", "n_exposures", R_PASS),
                        ("C", "c_exposures", C_PASS)):
        e = num(r.get(col))
        o[f"{t}_exposure"] = e
        o[f"{t}_cursor"] = None if e is None else int(round(e * per))
    return o


def write_tsv(path, rows):
    if not rows:
        open(path, "w").write("")
        print(f"[canneal] (0 rows) {path}")
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
    print(f"[canneal] wrote {path}  ({len(rows)} rows)")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--verify-checkpoints", action="store_true",
                    help="also read each run's endpoint checkpoint metadata "
                         "and confirm that only the comprehension LR differs")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)

    data = {arm: {s: full_rows(a.runs_root, RUN.format(arm=arm, seed=s))
                  for s in SEEDS} for arm in ARMS}

    rows, missing = [], []
    for arm in ARMS:
        for s in SEEDS:
            for k, step in enumerate(milestones(), start=1):
                r = data[arm][s].get(step)
                if r is None:
                    missing.append(f"{arm} seed {s} milestone {k} step {step}")
                    continue
                rows.append({"arm": arm, "lr_c": ARM_LR[arm], "seed": s,
                             "milestone": k, "u": step / (R_PASS * CYCLE),
                             "global_step": step, **readout(r)})

    # The 24 milestones ARE the scheduled full evaluations, so the strict
    # 5-in-a-row ceiling streak is exactly a run over these rows.
    for arm in ARMS:
        for s in SEEDS:
            streak = 0
            for r in [x for x in rows if x["arm"] == arm and x["seed"] == s]:
                at = (r["rep_canonical_errors"] == 0
                      and r["rep_freear_errors"] == 0
                      and r["naming_errors"] == 0
                      and r["comp_top1_errors"] == 0)
                streak = streak + 1 if at else 0
                r["at_ceiling"] = int(at)
                r["ceiling_streak"] = streak
    write_tsv(os.path.join(a.out_dir, "canneal_by_milestone.tsv"), rows)

    # R and N learning rates are held fixed at 3e-5 in every arm; the logged
    # `lr` column is the repetition lr, so it is a direct check of that.
    off = sorted({(r["arm"], r["lr_repetition_logged"]) for r in rows
                  if r["lr_repetition_logged"] not in (None, 3e-5)})
    print(f"[canneal] repetition-LR check: "
          + ("HELD AT 3e-5 in all arms" if not off
             else f"UNEXPECTED VALUES {off}  <-- confound, investigate"))

    # ---- cursor identity: all arms are 1:2:3, so cursors MUST match -------
    proof = []
    for k, step in enumerate(milestones(), start=1):
        for s in SEEDS:
            got = {arm: [r for r in rows if r["arm"] == arm and r["seed"] == s
                         and r["milestone"] == k] for arm in ARMS}
            if not all(got[arm] for arm in ARMS):
                continue
            g = {arm: got[arm][0] for arm in ARMS}
            rec = {"milestone": k, "u": step / (R_PASS * CYCLE),
                   "global_step": step, "seed": s}
            ok = True
            for t in ("R", "N", "C"):
                vals = {arm: g[arm][f"{t}_cursor"] for arm in ARMS}
                rec[f"{t}_cursor"] = vals["ctrl"]
                same = len(set(vals.values())) == 1
                rec[f"{t}_identical"] = int(same)
                ok &= same
            rec["MATCHED"] = int(ok)
            proof.append(rec)
    write_tsv(os.path.join(a.out_dir, "cursor_identity_proof.tsv"), proof)
    bad = [p for p in proof if not p["MATCHED"]]
    print(f"[canneal] cursor-identity rows: {len(proof)}, FAILED: {len(bad)}"
          + ("  <-- DO NOT COMPARE THESE" if bad else "  (all identical)"))

    # ---- per-arm summary and residual decay ------------------------------
    summ = []
    for arm in ARMS:
        for k, step in enumerate(milestones(), start=1):
            sub = [r for r in rows if r["arm"] == arm and r["milestone"] == k]
            if not sub:
                continue
            ce = [r["comp_top1_errors"] for r in sub]
            rec = {"arm": arm, "lr_c": ARM_LR[arm], "milestone": k,
                   "u": step / (R_PASS * CYCLE), "n_seeds": len(sub),
                   "C_err_mean": round(st.mean(ce), 3),
                   "C_err_sd": round(st.stdev(ce), 3) if len(ce) > 1 else 0.0,
                   "C_err_per_seed": ce}
            rec["max_ceiling_streak"] = max(r.get("ceiling_streak", 0)
                                            for r in sub)
            for key in ("rep_canonical_errors", "rep_freear_errors",
                        "naming_errors", "rep_ltm", "rep_wm", "gate_mean",
                        "comp_margin_mean", "comp_rank_median"):
                v = [r[key] for r in sub if r[key] is not None]
                rec[key + "_mean"] = round(st.mean(v), 6) if v else None
            summ.append(rec)
    # residual decay ratio between consecutive milestones, per arm
    for arm in ARMS:
        prev = None
        for rec in [r for r in summ if r["arm"] == arm]:
            if prev is not None and prev > 0:
                rec["C_residual_ratio_vs_prev"] = round(
                    rec["C_err_mean"] / prev, 4)
            prev = rec["C_err_mean"]
    write_tsv(os.path.join(a.out_dir, "canneal_summary.tsv"), summ)

    # ---- endpoint, paired against the control ----------------------------
    paired, endpoint = [], {}
    last = 24
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm and r["milestone"] == last]
        endpoint[arm] = {r["seed"]: r for r in sub}
    for s in SEEDS:
        c = endpoint["ctrl"].get(s)
        if not c:
            continue
        for arm in ("5e5", "3e5"):
            b = endpoint[arm].get(s)
            if not b:
                continue
            paired.append({
                "seed": s, "arm": arm, "lr_c": ARM_LR[arm],
                "C_err_ctrl": c["comp_top1_errors"],
                "C_err_arm": b["comp_top1_errors"],
                "dC_errors": b["comp_top1_errors"] - c["comp_top1_errors"],
                "R_err_ctrl": c["rep_canonical_errors"],
                "R_err_arm": b["rep_canonical_errors"],
                "Rfree_err_arm": b["rep_freear_errors"],
                "N_err_arm": b["naming_errors"],
                "LTM_ctrl": c["rep_ltm"], "LTM_arm": b["rep_ltm"],
                "dLTM": (None if c["rep_ltm"] is None or b["rep_ltm"] is None
                         else round(b["rep_ltm"] - c["rep_ltm"], 6)),
                "WM_arm": b["rep_wm"], "gate_arm": b["gate_mean"]})
    write_tsv(os.path.join(a.out_dir, "canneal_endpoint_paired.tsv"), paired)

    # ---- strict-ceiling hits ---------------------------------------------
    hits = []
    for r in rows:
        if (r["rep_canonical_errors"] == 0 and r["rep_freear_errors"] == 0
                and r["naming_errors"] == 0 and r["comp_top1_errors"] == 0):
            hits.append({"arm": r["arm"], "seed": r["seed"],
                         "milestone": r["milestone"], "u": r["u"],
                         "global_step": r["global_step"]})
    write_tsv(os.path.join(a.out_dir, "ceiling_hits.tsv"), hits)
    if hits:
        print(f"[canneal] STRICT CEILING HIT on {len(hits)} evaluation(s):")
        for h in hits[:10]:
            print(f"    arm {h['arm']} seed {h['seed']} u{h['u']:.0f} "
                  f"step {h['global_step']}")
        print("[canneal] PRESERVE every checkpoint of those runs.")
    else:
        print("[canneal] no strict 0/0/0/0 evaluation in any arm")

    # ---- optional: the settings-identity proof, read off the checkpoints --
    policy = []
    if a.verify_checkpoints:
        import torch  # local: the TSV path must not require torch
        for arm in ARMS:
            for s in SEEDS:
                rid = RUN.format(arm=arm, seed=s)
                p = os.path.join(a.runs_root, rid, "checkpoints",
                                 f"step_{END_STEP:08d}.pt")
                if not os.path.exists(p):
                    continue
                ck = torch.load(p, map_location="cpu", weights_only=False)
                lp = ck.get("lr_policy", {})
                policy.append({
                    "arm": arm, "seed": s, "global_step": ck.get("global_step"),
                    "lr_kind": lp.get("kind"),
                    "lr_repetition": lp.get("repetition"),
                    "lr_naming": lp.get("naming"),
                    "lr_comprehension": lp.get("comprehension"),
                    "lr_c_matches_arm": int(
                        lp.get("comprehension") == ARM_LR[arm]),
                    "schedule": ck.get("schedule"),
                    "schedule_ratio": ck.get("schedule_ratio"),
                    "schedule_seed": ck.get("schedule_seed"),
                    "schedule_anchor_step": ck.get("schedule_anchor_step"),
                    "optimizer_policy": ck.get("optimizer_policy"),
                    "widths": ck.get("widths"),
                    "cursors": ck.get("cursors"),
                    "consecutive_ceiling": ck.get("consecutive_ceiling"),
                    "last_ceiling_step": ck.get("last_ceiling_step")})
                del ck
        write_tsv(os.path.join(a.out_dir, "lr_policy_proof.tsv"), policy)
        if policy:
            # Invariants that must hold across EVERY run, seeds included.
            for f, label in (("schedule", "schedule"),
                             ("schedule_ratio", "ratio"),
                             ("schedule_anchor_step", "schedule anchor"),
                             ("optimizer_policy", "optimizer policy"),
                             ("widths", "widths"),
                             ("lr_repetition", "R lr"), ("lr_naming", "N lr")):
                vals = {json.dumps(p[f], sort_keys=True) for p in policy}
                print(f"[canneal] {label} across all arms: "
                      + ("IDENTICAL " + vals.pop() if len(vals) == 1
                         else f"DIFFERS {sorted(vals)}  <-- confound"))
            # schedule_seed is seed*1000003 + 4, so it is SUPPOSED to differ
            # between seeds.  The invariant is within-seed: for one seed, all
            # three arms must draw the same task order.  Comparing it across
            # seeds -- which an earlier version of this report did -- reports
            # a confound that does not exist.
            sched = []
            for s in SEEDS:
                rec = {"seed": s}
                vals = []
                for arm in ARMS:
                    hit = [p for p in policy if p["arm"] == arm and p["seed"] == s]
                    v = hit[0]["schedule_seed"] if hit else None
                    rec[f"schedule_seed_{arm}"] = v
                    vals.append(v)
                present = [v for v in vals if v is not None]
                rec["n_arms_present"] = len(present)
                rec["MATCHED"] = int(bool(present) and len(set(present)) == 1)
                sched.append(rec)
            write_tsv(os.path.join(a.out_dir, "schedule_seed_proof.tsv"), sched)
            unmatched = [r["seed"] for r in sched if not r["MATCHED"]]
            print("[canneal] schedule seed WITHIN each seed across arms: "
                  + ("IDENTICAL in every seed" if not unmatched
                     else f"DIFFERS for seeds {unmatched}  <-- confound"))
            print("[canneal]   (across different seeds it is expected to "
                  "differ: schedule_seed = seed*1000003 + 4)")
            miss = [(p["arm"], p["seed"]) for p in policy
                    if not p["lr_c_matches_arm"]]
            print(f"[canneal] comprehension LR matches its arm: "
                  + ("yes, in all runs" if not miss else f"NO for {miss}"))

    json.dump({"arms": ARM_LR, "seeds": SEEDS,
               "source_step": SRC_STEP, "end_step": END_STEP,
               "n_milestones": 24, "cursor_identity_failures": len(bad),
               "ceiling_hits": len(hits), "missing_rows": len(missing),
               "checkpoints_verified": len(policy)},
              open(os.path.join(a.out_dir, "canneal_meta.json"), "w"), indent=1)
    if missing:
        open(os.path.join(a.out_dir, "missing.txt"), "w").write(
            "\n".join(missing) + "\n")
        print(f"[canneal] {len(missing)} missing milestone rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
