#!/usr/bin/env python3
"""Trajectory / clipping analysis for a joint-scratch run (analysis only).

Reads a run's `metrics.tsv` and `logs/losses.tsv` and reports, from exact
recorded values with no smoothing:

  * the developmental trajectory against optimizer step and per-task
    exposures/item (R and N: ceil(29,571/64) = 463 steps per pass; C:
    ceil(27,981/64) = 438);
  * acquisition rates (delta per 100 N-exposures) over named windows, so a
    change of slope at the LR boundary is separated from slowing under a
    constant LR;
  * loss components at chosen steps;
  * gradient-norm / clipping statistics.  NOTE: `grad_norm` in losses.tsv is
    the return value of `clip_grad_norm_`, i.e. the PRE-clip total norm, and
    rows are logged every `--log-every` steps -- so the clipped fraction is a
    property of the LOGGED SAMPLE, never an exhaustive per-step measurement.

Two runs can be compared directly (`--baseline`), which is the intended use
for FINAL-2A vs the exactly-paired FINAL-1 baseline.

Usage:
    python scripts/naming_comprehension/analyze_joint_dynamics.py \
        --run outputs/joint_scratch/final2a_calign_seed22_final_full \
        --baseline outputs/joint_scratch/final1_j0_seed22_final_full
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from typing import Dict, List, Optional, Sequence, Tuple

# Canonical FINAL population pass lengths (batch 64).
STEPS_PER_R_PASS = 463          # ceil(29571 / 64)
STEPS_PER_C_PASS = 438          # ceil(27981 / 64)
LR_BOUNDARY = 46_300

TRAJECTORY_KEYS = ("naming_exact", "comp_top1", "comp_top5",
                   "rep_full", "rep_ltm", "rep_wm", "comp_rank_median")
FULL_KEYS = ("full_rep_full", "full_rep_ltm", "full_rep_wm",
             "full_comp_top1", "full_comp_top5",
             "full_naming_exact", "full_naming_wer")


def read_tsv(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def fnum(row: dict, key: str) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return float("nan")


def exposures(step: int, steps_per_pass: int) -> float:
    """Exposures/item after `step` optimizer updates of a flat stream."""
    return step / steps_per_pass


def series(rows: Sequence[dict], key: str) -> List[Tuple[int, float]]:
    """(step, value) pairs for one metric, de-duplicated and sorted by step."""
    seen: Dict[int, float] = {}
    for r in rows:
        v = fnum(r, key)
        if not math.isnan(v):
            seen.setdefault(int(r["step"]), v)
    return sorted(seen.items())


def slope_per_100_exposures(points: Sequence[Tuple[int, float]],
                            lo: int, hi: int,
                            steps_per_pass: int = STEPS_PER_R_PASS
                            ) -> float:
    """Endpoint slope over [lo, hi], expressed per 100 exposures/item."""
    seg = [(s, v) for s, v in points if lo <= s <= hi]
    if len(seg) < 2 or seg[-1][0] == seg[0][0]:
        return float("nan")
    (s0, v0), (s1, v1) = seg[0], seg[-1]
    return (v1 - v0) / (s1 - s0) * (100 * steps_per_pass)


def clip_stats(loss_rows: Sequence[dict], clip: float = 1.0,
               lo: int = 0, hi: Optional[int] = None) -> dict:
    """Clipping statistics over the LOGGED rows in [lo, hi].

    `fraction_over_clip` is the share of logged points whose pre-clip norm
    exceeded `clip`; it is evidence about the sample, not a census of every
    optimizer step.
    """
    g = [fnum(r, "grad_norm") for r in loss_rows
         if lo <= int(r["step"]) <= (hi if hi is not None else 1 << 62)]
    g = [x for x in g if not math.isnan(x)]
    if not g:
        return {"n_logged": 0}
    return {
        "n_logged": len(g),
        "fraction_over_clip": sum(1 for x in g if x > clip) / len(g),
        "median_pre_clip_norm": statistics.median(g),
        "min_pre_clip_norm": min(g),
        "max_pre_clip_norm": max(g),
        "median_update_scale": statistics.median(
            [min(clip / x, 1.0) for x in g]),
        "note": ("grad_norm is the PRE-clip total norm returned by "
                 "clip_grad_norm_; statistics describe logged rows only"),
    }


def milestone_rows(rows: Sequence[dict]) -> List[dict]:
    """Rows carrying full-population evaluations (the milestones/endpoint)."""
    return [r for r in rows
            if not math.isnan(fnum(r, "full_naming_exact"))
            or not math.isnan(fnum(r, "full_comp_top1"))]


def default_windows(max_step: int) -> Dict[str, Tuple[int, int]]:
    w = {"stage1 (LR 1e-3)": (0, LR_BOUNDARY),
         "first 100 exp after drop": (LR_BOUNDARY, 2 * LR_BOUNDARY)}
    for lo, hi in ((92_600, 138_900), (138_900, 231_500), (231_500, 324_100)):
        if lo < max_step:
            w[f"{lo//1000}k-{hi//1000}k"] = (lo, min(hi, max_step))
    return w


def analyse(run_dir: str, every: int) -> dict:
    rows = read_tsv(os.path.join(run_dir, "metrics.tsv"))
    lpath = os.path.join(run_dir, "logs", "losses.tsv")
    losses = read_tsv(lpath) if os.path.exists(lpath) else []
    cfg_path = os.path.join(run_dir, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    max_step = max(int(r["step"]) for r in rows) if rows else 0

    out = {
        "run_dir": os.path.abspath(run_dir),
        "c_align_weight": cfg.get("c_align_weight"),
        "c_stream_objective": cfg.get("c_stream_objective"),
        "comprehension_population": cfg.get("comprehension_population"),
        "naming_population": cfg.get("naming_population"),
        "max_step": max_step,
        "n_exposures": exposures(max_step, STEPS_PER_R_PASS),
        "trajectory": [], "slopes": {}, "milestones": [], "clipping": {},
    }
    for r in rows:
        s = int(r["step"])
        if s % every and s != max_step:
            continue
        rec = {"step": s,
               "N_exposures": round(exposures(s, STEPS_PER_R_PASS), 2),
               "C_exposures": round(exposures(s, STEPS_PER_C_PASS), 2),
               "lr": fnum(r, "lr")}
        rec.update({k: fnum(r, k) for k in TRAJECTORY_KEYS})
        out["trajectory"].append(rec)

    windows = default_windows(max_step)
    for key in ("naming_exact", "comp_top1", "comp_top5", "rep_ltm"):
        pts = series(rows, key)
        out["slopes"][key] = {n: slope_per_100_exposures(pts, a, b)
                              for n, (a, b) in windows.items()}

    for r in milestone_rows(rows):
        m = {"step": int(r["step"]),
             "N_exposures": round(exposures(int(r["step"]), STEPS_PER_R_PASS), 1)}
        m.update({k: fnum(r, k) for k in FULL_KEYS})
        m["full_rep_errors"] = (1.0 - m["full_rep_full"]) * 29_571
        out["milestones"].append(m)

    if losses:
        out["clipping"] = {n: clip_stats(losses, 1.0, a, b)
                           for n, (a, b) in windows.items()}
        out["clipping"]["ALL"] = clip_stats(losses, 1.0)
        last = losses[-1]
        out["final_losses"] = {k: fnum(last, k) for k in
                               ("joint_total", "rep", "align", "dec", "wm",
                                "pool_ce", "retrieval_ce", "retrieval_weighted",
                                "c_align", "c_align_weighted", "naming_ce",
                                "grad_norm")}
    return out


def print_report(a: dict, baseline: Optional[dict] = None) -> None:
    print(f"\n=== {a['run_dir']} ===")
    print(f"c_align_weight={a['c_align_weight']}  |  {a['c_stream_objective']}")
    print(f"steps={a['max_step']}  N-exposures={a['n_exposures']:.0f}")

    print("\n-- trajectory (exact recorded values) --")
    hdr = ("  {:>8}{:>8}{:>8}{:>8}{:>9}{:>9}{:>9}{:>9}{:>9}{:>10}".format(
        "step", "N_exp", "C_exp", "lr", "naming", "c_top1", "c_top5",
        "rep_full", "rep_ltm", "rank_med"))
    print(hdr)
    for r in a["trajectory"]:
        mark = "  <-- LR drop" if r["step"] == LR_BOUNDARY else ""
        print("  {:>8}{:>8.0f}{:>8.0f}{:>8.0e}{:>9.4f}{:>9.4f}{:>9.4f}"
              "{:>9.4f}{:>9.4f}{:>10.1f}{}".format(
                  r["step"], r["N_exposures"], r["C_exposures"], r["lr"],
                  r["naming_exact"], r["comp_top1"], r["comp_top5"],
                  r["rep_full"], r["rep_ltm"], r["comp_rank_median"], mark))

    print("\n-- acquisition rate (delta per 100 N-exposures) --")
    for key, w in a["slopes"].items():
        print(f"  {key:14s} " + " | ".join(f"{n}: {v:+.4f}"
                                           for n, v in w.items()))

    if a["milestones"]:
        print("\n-- full-population milestones --")
        print("  {:>8}{:>7}{:>11}{:>11}{:>12}{:>11}{:>12}".format(
            "step", "N_exp", "full_rep", "rep_errors", "full_C_top1",
            "full_C_top5", "full_naming"))
        for m in a["milestones"]:
            print("  {:>8}{:>7.0f}{:>11.6f}{:>11.0f}{:>12.6f}{:>11.6f}"
                  "{:>12.6f}".format(
                      m["step"], m["N_exposures"], m["full_rep_full"],
                      m["full_rep_errors"], m["full_comp_top1"],
                      m["full_comp_top5"], m["full_naming_exact"]))
            if baseline:
                b = next((x for x in baseline["milestones"]
                          if x["step"] == m["step"]), None)
                if b:
                    def ratio(x, y):
                        return x / y if y else float("nan")
                    print("      vs baseline: C_top1 {:+.6f} ({:.2f}x)  "
                          "naming {:+.6f} ({:.2f}x)  LTM_rep {:+.6f}".format(
                              m["full_comp_top1"] - b["full_comp_top1"],
                              ratio(m["full_comp_top1"], b["full_comp_top1"]),
                              m["full_naming_exact"] - b["full_naming_exact"],
                              ratio(m["full_naming_exact"], b["full_naming_exact"]),
                              m["full_rep_ltm"] - b["full_rep_ltm"]))

    if a.get("clipping"):
        print("\n-- gradient norm / clipping (LOGGED rows only, not a census) --")
        for n, s in a["clipping"].items():
            if not s.get("n_logged"):
                continue
            print(f"  {n:26s} n={s['n_logged']:5d} "
                  f"over_clip={s['fraction_over_clip']*100:5.1f}% "
                  f"median={s['median_pre_clip_norm']:7.3f} "
                  f"max={s['max_pre_clip_norm']:8.3f} "
                  f"median_scale={s['median_update_scale']:.3f}")
    if a.get("final_losses"):
        print("\n-- final logged losses --")
        print("  " + "  ".join(f"{k}={v:.5g}" for k, v in
                               a["final_losses"].items() if not math.isnan(v)))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", required=True, help="run directory to analyse")
    p.add_argument("--baseline", default=None,
                   help="paired baseline run directory for milestone deltas")
    p.add_argument("--every", type=int, default=11575,
                   help="print a trajectory row every N steps (default 25 exp)")
    p.add_argument("--json-out", default=None)
    args = p.parse_args(argv)

    a = analyse(args.run, args.every)
    b = analyse(args.baseline, args.every) if args.baseline else None
    print_report(a, b)
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)),
                    exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"run": a, "baseline": b}, fh, indent=2, default=str)
        print(f"\n[json] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
