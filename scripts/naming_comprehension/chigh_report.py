"""Matched three-arm comparison for the comprehension-LR raise u2000 -> u3000.

All three arms keep schedule 1:2:3, so at a given global step they share the
exact R/N/C/pool cursors.  That identity is PROVEN per milestone rather than
assumed, and any milestone where it fails is flagged instead of averaged.

Two invariants are checked at different scopes, and confusing them is what
made the previous report cry confound:

  * schedule, ratio, anchor, optimizer policy, widths, R LR and N LR must be
    identical across EVERY run, seeds included;
  * schedule_seed = seed*1000003 + 4, so it is SUPPOSED to differ between
    seeds.  The invariant is WITHIN a seed: for one seed, all three arms must
    draw the same task order.  A per-seed proof table records it.

Primary readout is strict C top-1 error count and whether any run reaches
C = 0; the strict 0/0/0/0 ceiling is reported loudly wherever it appears.
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
SRC_STEP, END_STEP, MS_STEP = 5_556_000, 8_334_000, 69_450
N_MILESTONES = (END_STEP - SRC_STEP) // MS_STEP          # 40
ARMS = ["ctrl", "15e5", "2e4"]
ARM_LR = {"ctrl": 1e-4, "15e5": 1.5e-4, "2e4": 2e-4}
SEEDS = [19, 20, 21, 22]
RUN = "final_chigh_{arm}_u3000_h512_s{seed}"
STREAM_SEED_STRIDE, SCHEDULE_SEED_OFFSET = 1_000_003, 4


def expected_schedule_seed(seed: int) -> int:
    return int(seed) * STREAM_SEED_STRIDE + SCHEDULE_SEED_OFFSET


def num(v):
    if v in (None, "", "nan", "NaN"):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def milestones() -> List[int]:
    return [SRC_STEP + MS_STEP * k for k in range(1, N_MILESTONES + 1)]


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
        print(f"[chigh] (0 rows) {path}")
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
    print(f"[chigh] wrote {path}  ({len(rows)} rows)")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--verify-checkpoints", action="store_true",
                    help="also read each run's latest checkpoint metadata and "
                         "confirm that only the comprehension LR differs")
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

    # The milestones ARE the scheduled full evaluations, so the strict
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
    write_tsv(os.path.join(a.out_dir, "chigh_by_milestone.tsv"), rows)

    # R and N learning rates are held fixed at 3e-5 in every arm; the logged
    # `lr` column is the repetition lr, so it is a direct check of that.
    off = sorted({(r["arm"], r["lr_repetition_logged"]) for r in rows
                  if r["lr_repetition_logged"] not in (None, 3e-5)})
    print("[chigh] repetition-LR check: "
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
    print(f"[chigh] cursor-identity rows: {len(proof)}, FAILED: {len(bad)}"
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
                   "C_err_min": min(ce), "C_err_per_seed": ce,
                   "seeds_at_C_zero": sum(1 for x in ce if x == 0),
                   "max_ceiling_streak": max(r.get("ceiling_streak", 0)
                                             for r in sub)}
            for key in ("rep_canonical_errors", "rep_freear_errors",
                        "naming_errors", "rep_ltm", "rep_wm", "gate_mean",
                        "comp_margin_mean", "comp_rank_median"):
                v = [r[key] for r in sub if r[key] is not None]
                rec[key + "_mean"] = round(st.mean(v), 6) if v else None
            summ.append(rec)
    for arm in ARMS:
        prev = None
        for rec in [r for r in summ if r["arm"] == arm]:
            if prev is not None and prev > 0:
                rec["C_residual_ratio_vs_prev"] = round(
                    rec["C_err_mean"] / prev, 4)
            prev = rec["C_err_mean"]
    write_tsv(os.path.join(a.out_dir, "chigh_summary.tsv"), summ)

    # ---- paired against the control at the deepest COMMON milestone ------
    common = [k for k in range(1, N_MILESTONES + 1)
              if all(any(r["arm"] == arm and r["seed"] == s
                         and r["milestone"] == k for r in rows)
                     for arm in ARMS for s in SEEDS)]
    last = max(common) if common else None
    paired = []
    if last is not None:
        end = {arm: {r["seed"]: r for r in rows
                     if r["arm"] == arm and r["milestone"] == last}
               for arm in ARMS}
        for s in SEEDS:
            c = end["ctrl"].get(s)
            if not c:
                continue
            for arm in ("15e5", "2e4"):
                b = end[arm].get(s)
                if not b:
                    continue
                paired.append({
                    "milestone": last, "u": c["u"], "seed": s, "arm": arm,
                    "lr_c": ARM_LR[arm],
                    "C_err_ctrl": c["comp_top1_errors"],
                    "C_err_arm": b["comp_top1_errors"],
                    "dC_errors": b["comp_top1_errors"] - c["comp_top1_errors"],
                    "R_err_ctrl": c["rep_canonical_errors"],
                    "R_err_arm": b["rep_canonical_errors"],
                    "Rfree_err_ctrl": c["rep_freear_errors"],
                    "Rfree_err_arm": b["rep_freear_errors"],
                    "N_err_ctrl": c["naming_errors"],
                    "N_err_arm": b["naming_errors"],
                    "LTM_ctrl": c["rep_ltm"], "LTM_arm": b["rep_ltm"],
                    "dLTM": (None if c["rep_ltm"] is None or b["rep_ltm"] is None
                             else round(b["rep_ltm"] - c["rep_ltm"], 6)),
                    "WM_ctrl": c["rep_wm"], "WM_arm": b["rep_wm"],
                    "gate_ctrl": c["gate_mean"], "gate_arm": b["gate_mean"]})
        print(f"[chigh] paired at the deepest COMMON milestone: "
              f"{last} (u{SRC_STEP / (R_PASS * CYCLE) + 25 * last:.0f})")
    else:
        print("[chigh] no milestone is present in all 12 runs -- nothing paired")
    write_tsv(os.path.join(a.out_dir, "chigh_paired.tsv"), paired)
    if paired:
        for arm in ("15e5", "2e4"):
            d = [p["dC_errors"] for p in paired if p["arm"] == arm]
            better = sum(1 for x in d if x < 0)
            print(f"[chigh]   C LR {ARM_LR[arm]:g} vs control: mean dC "
                  f"{st.mean(d):+.2f} errors, better in {better}/{len(d)} seeds")

    # ---- the primary question: does C reach zero? ------------------------
    czero = [{"arm": r["arm"], "seed": r["seed"], "milestone": r["milestone"],
              "u": r["u"], "global_step": r["global_step"],
              "rep_canonical_errors": r["rep_canonical_errors"],
              "rep_freear_errors": r["rep_freear_errors"],
              "naming_errors": r["naming_errors"]}
             for r in rows if r["comp_top1_errors"] == 0]
    write_tsv(os.path.join(a.out_dir, "c_zero_hits.tsv"), czero)
    if czero:
        first = {}
        for h in czero:
            first.setdefault((h["arm"], h["seed"]), h)
        print(f"[chigh] STRICT C = 0 reached in {len(first)} run(s):")
        for (arm, s), h in sorted(first.items()):
            print(f"    arm {arm} seed {s} first at u{h['u']:.0f} "
                  f"step {h['global_step']}")
    else:
        print("[chigh] no evaluation reached strict C = 0")

    # ---- strict-ceiling hits ---------------------------------------------
    hits = [{"arm": r["arm"], "seed": r["seed"], "milestone": r["milestone"],
             "u": r["u"], "global_step": r["global_step"],
             "ceiling_streak": r["ceiling_streak"]}
            for r in rows if r.get("at_ceiling")]
    write_tsv(os.path.join(a.out_dir, "ceiling_hits.tsv"), hits)
    if hits:
        runs_hit = len({(h["arm"], h["seed"]) for h in hits})
        banner = [f"STRICT CEILING HIT on {len(hits)} evaluation(s), "
                  f"{runs_hit} run(s)",
                  "R canonical = R freeAR = N = C = 0"]
        w = max(len(b) for b in banner) + 10
        print("")
        print("  " + "*" * w)
        for b in banner:
            print("  ***  " + b.ljust(w - 10) + "  ***")
        print("  " + "*" * w)
        for h in hits[:12]:
            print(f"    arm {h['arm']} seed {h['seed']} u{h['u']:.0f} "
                  f"step {h['global_step']}  streak {h['ceiling_streak']}")
        print("  PRESERVE every checkpoint of those runs.")
        print("")
    else:
        print("[chigh] no strict 0/0/0/0 evaluation in any arm")

    # ---- optional: the settings-identity proof, read off the checkpoints --
    policy, sched = [], []
    if a.verify_checkpoints:
        import torch  # local: the TSV path must not require torch
        for arm in ARMS:
            for s in SEEDS:
                rid = RUN.format(arm=arm, seed=s)
                d = os.path.join(a.runs_root, rid, "checkpoints")
                if not os.path.isdir(d):
                    continue
                cks = sorted(f for f in os.listdir(d)
                             if f.startswith("step_") and f.endswith(".pt"))
                if not cks:
                    continue
                ck = torch.load(os.path.join(d, cks[-1]), map_location="cpu",
                                weights_only=False)
                lp = ck.get("lr_policy", {})
                policy.append({
                    "arm": arm, "seed": s, "checkpoint": cks[-1],
                    "global_step": ck.get("global_step"),
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
        # (a) invariants that must hold across EVERY run, seeds included
        for f, label in (("schedule", "schedule"),
                         ("schedule_ratio", "ratio"),
                         ("schedule_anchor_step", "schedule anchor"),
                         ("optimizer_policy", "optimizer policy"),
                         ("widths", "widths"),
                         ("lr_kind", "lr policy kind"),
                         ("lr_repetition", "R lr"), ("lr_naming", "N lr")):
            vals = {json.dumps(p[f], sort_keys=True) for p in policy}
            print(f"[chigh] {label} across all arms: "
                  + ("IDENTICAL " + vals.pop() if len(vals) == 1
                     else f"DIFFERS {sorted(vals)}  <-- confound"))
        # (b) schedule_seed: the invariant is WITHIN a seed, never across
        for s in SEEDS:
            rec = {"seed": s, "expected": expected_schedule_seed(s)}
            vals = []
            for arm in ARMS:
                hit = [p for p in policy if p["arm"] == arm and p["seed"] == s]
                v = hit[0]["schedule_seed"] if hit else None
                rec[f"schedule_seed_{arm}"] = v
                vals.append(v)
            present = [v for v in vals if v is not None]
            rec["n_arms_present"] = len(present)
            rec["MATCHED"] = int(bool(present) and len(set(present)) == 1)
            rec["matches_formula"] = int(bool(present) and all(
                int(v) == rec["expected"] for v in present))
            sched.append(rec)
        write_tsv(os.path.join(a.out_dir, "schedule_seed_proof.tsv"), sched)
        unmatched = [r["seed"] for r in sched if not r["MATCHED"]]
        offform = [r["seed"] for r in sched if not r["matches_formula"]]
        print("[chigh] schedule seed WITHIN each seed across arms: "
              + ("IDENTICAL in every seed" if not unmatched
                 else f"DIFFERS for seeds {unmatched}  <-- confound"))
        print("[chigh]   and equal to seed*1000003 + 4: "
              + ("yes in every seed" if not offform else f"NO for {offform}"))
        print("[chigh]   (across DIFFERENT seeds it is expected to differ: "
              + ", ".join(f"s{r['seed']}={r['expected']}" for r in sched) + ")")
        miss = [(p["arm"], p["seed"]) for p in policy
                if not p["lr_c_matches_arm"]]
        print("[chigh] comprehension LR matches its arm: "
              + ("yes, in all runs" if not miss else f"NO for {miss}"))

    json.dump({"arms": ARM_LR, "seeds": SEEDS,
               "source_step": SRC_STEP, "end_step": END_STEP,
               "n_milestones": N_MILESTONES,
               "paired_at_milestone": last,
               "cursor_identity_failures": len(bad),
               "c_zero_evaluations": len(czero),
               "ceiling_hits": len(hits), "missing_rows": len(missing),
               "checkpoints_verified": len(policy),
               "schedule_seed_within_seed_failures":
                   sum(1 for r in sched if not r["MATCHED"])},
              open(os.path.join(a.out_dir, "chigh_meta.json"), "w"), indent=1)
    if missing:
        open(os.path.join(a.out_dir, "missing.txt"), "w").write(
            "\n".join(missing) + "\n")
        print(f"[chigh] {len(missing)} missing milestone rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
