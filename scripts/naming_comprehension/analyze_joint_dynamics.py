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
    """Exposures/item after `step` optimizer updates of a flat stream.

    Only valid for the SUMMED schedule, where every update draws one batch of
    every active stream.  Interleaved runs record their exposures per row (see
    `row_exposures`), because there a step trains only one task.
    """
    return step / steps_per_pass


def row_exposures(row: dict, step: int) -> Dict[str, float]:
    """Per-task exposures for one metrics row.

    Prefers the recorded `{r,n,c}_exposures` columns, which are the cursor
    ledger and are correct under ANY schedule; falls back to the summed-schedule
    arithmetic for runs written before those columns existed (FINAL-1/2A).
    """
    rec = {k: fnum(row, f"{p}_exposures")
           for k, p in (("R", "r"), ("N", "n"), ("C", "c"))}
    if all(not math.isnan(v) for v in rec.values()):
        return rec
    return {"R": exposures(step, STEPS_PER_R_PASS),
            "N": exposures(step, STEPS_PER_R_PASS),
            "C": exposures(step, STEPS_PER_C_PASS)}


def lr_phases(run_dir: str) -> List[Tuple[int, dict]]:
    """(start_step, lr_policy) for every launch of a run, in order.

    A run's first launch writes config.json; later launches write
    config_from_step_<N>.json.  A run whose learning-rate policy changed
    mid-flight (a declared phase transition) therefore has several, and only
    the one covering a given step describes that step truthfully.
    """
    phases: List[Tuple[int, dict]] = []
    for name in sorted(os.listdir(run_dir)) if os.path.isdir(run_dir) else []:
        if not (name.startswith("config") and name.endswith(".json")):
            continue
        start = 0
        if name.startswith("config_from_step_"):
            try:
                start = int(name[len("config_from_step_"):-len(".json")])
            except ValueError:
                continue
        try:
            cfg = json.load(open(os.path.join(run_dir, name), encoding="utf-8"))
        except Exception:
            continue
        pol = cfg.get("lr_policy")
        if not pol:
            # Runs predating the explicit policy (FINAL-1 .. FINAL-3) always
            # used the two-stage schedule; reconstruct it from the fields they
            # did record rather than reporting an empty policy.
            pol = {"kind": "two_stage_rep_cursor",
                   "stage1": cfg.get("lr_stage1"),
                   "stage2": cfg.get("lr_stage2"),
                   "boundary_rep_batches": cfg.get("lr_boundary_steps")}
        phases.append((start, pol))
    return sorted(phases, key=lambda p: p[0])


def policy_at(phases: Sequence[Tuple[int, dict]], step: int) -> dict:
    """The policy that TRAINED `step`.

    A phase recorded as starting at N resumed AT step N and trained steps
    N+1 onward, so the row at exactly N still belongs to the previous phase --
    which matters precisely at a transition boundary, where the two policies
    differ.  The initial phase covers step 0.
    """
    if not phases:
        return {}
    cur = phases[0][1]
    for start, pol in phases:
        if start < step:
            cur = pol
    return cur


def task_lrs_at(phases: Sequence[Tuple[int, dict]], step: int,
                scalar_lr: float) -> Dict[str, float]:
    """Per-task learning rates for one row.

    Under the two-stage policy all three tasks share the row's recorded scalar
    LR.  Under a task-specific policy the three differ and a single number
    would be a fabrication, so the declared rates are used.
    """
    pol = policy_at(phases, step)
    if pol.get("kind") == "task_specific":
        return {"R": float(pol["repetition"]), "N": float(pol["naming"]),
                "C": float(pol["comprehension"])}
    return {"R": scalar_lr, "N": scalar_lr, "C": scalar_lr}


def format_lrs(lrs: Dict[str, float]) -> str:
    if lrs["R"] == lrs["N"] == lrs["C"]:
        return f"{lrs['R']:.0e}"
    return "/".join(f"{lrs[k]:.0e}" for k in ("R", "N", "C"))


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
    phases = lr_phases(run_dir)

    out = {
        "run_dir": os.path.abspath(run_dir),
        "c_align_weight": cfg.get("c_align_weight"),
        "c_stream_objective": cfg.get("c_stream_objective"),
        "comprehension_population": cfg.get("comprehension_population"),
        "naming_population": cfg.get("naming_population"),
        "schedule": cfg.get("schedule", "summed"),
        "schedule_ratio_R_N_C": cfg.get("schedule_ratio_R_N_C"),
        "lr_phases": [{"from_step": st, "lr_policy": pol} for st, pol in phases],
        "max_step": max_step,
        "trajectory": [], "slopes": {}, "milestones": [], "clipping": {},
    }
    for r in rows:
        s = int(r["step"])
        if s % every and s != max_step:
            continue
        ex = row_exposures(r, s)
        lrs = task_lrs_at(phases, s, fnum(r, "lr"))
        rec = {"step": s,
               "R_exposures": round(ex["R"], 2),
               "N_exposures": round(ex["N"], 2),
               "C_exposures": round(ex["C"], 2),
               "lr_R": lrs["R"], "lr_N": lrs["N"], "lr_C": lrs["C"],
               "lr_policy_kind": (policy_at(phases, s).get("kind")
                                  or "two_stage_rep_cursor")}
        rec.update({k: fnum(r, k) for k in TRAJECTORY_KEYS})
        out["trajectory"].append(rec)

    windows = default_windows(max_step)
    for key in ("naming_exact", "comp_top1", "comp_top5", "rep_ltm"):
        pts = series(rows, key)
        out["slopes"][key] = {n: slope_per_100_exposures(pts, a, b)
                              for n, (a, b) in windows.items()}

    for r in milestone_rows(rows):
        ex = row_exposures(r, int(r["step"]))
        mlrs = task_lrs_at(phases, int(r["step"]), fnum(r, "lr"))
        m = {"step": int(r["step"]),
             "lr_R": mlrs["R"], "lr_N": mlrs["N"], "lr_C": mlrs["C"],
             "R_exposures": round(ex["R"], 1),
             "N_exposures": round(ex["N"], 1),
             "C_exposures": round(ex["C"], 1)}
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
    print(f"schedule={a['schedule']} ratio={a['schedule_ratio_R_N_C']}  "
          f"steps={a['max_step']}")
    for ph in a.get("lr_phases", []):
        pol = ph["lr_policy"] or {}
        if pol.get("kind") == "task_specific":
            desc = (f"task_specific R={pol['repetition']:g} "
                    f"N={pol['naming']:g} C={pol['comprehension']:g}")
        else:
            desc = (f"{pol.get('kind','two_stage_rep_cursor')} "
                    f"(stage1={pol.get('stage1')}, stage2={pol.get('stage2')}, "
                    f"boundary={pol.get('boundary_rep_batches')} R batches)")
        print(f"  LR phase from step {ph['from_step']:>8}: {desc}")

    print("\n-- trajectory (exact recorded values) --")
    hdr = ("  {:>8}{:>7}{:>7}{:>7}{:>19}{:>9}{:>9}{:>9}{:>9}{:>9}{:>10}".format(
        "step", "R_exp", "N_exp", "C_exp", "lr R/N/C", "naming", "c_top1",
        "c_top5", "rep_full", "rep_ltm", "rank_med"))
    print(hdr)
    for r in a["trajectory"]:
        mark = "  <-- LR drop" if r["step"] == LR_BOUNDARY else ""
        print("  {:>8}{:>7.0f}{:>7.0f}{:>7.0f}{:>19}{:>9.4f}{:>9.4f}{:>9.4f}"
              "{:>9.4f}{:>9.4f}{:>10.1f}{}".format(
                  r["step"], r["R_exposures"], r["N_exposures"],
                  r["C_exposures"],
                  format_lrs({"R": r["lr_R"], "N": r["lr_N"], "C": r["lr_C"]}),
                  r["naming_exact"], r["comp_top1"], r["comp_top5"],
                  r["rep_full"], r["rep_ltm"], r["comp_rank_median"], mark))

    print("\n-- acquisition rate (delta per 100 N-exposures) --")
    for key, w in a["slopes"].items():
        print(f"  {key:14s} " + " | ".join(f"{n}: {v:+.4f}"
                                           for n, v in w.items()))

    if a["milestones"]:
        print("\n-- full-population milestones --")
        print("  {:>8}{:>7}{:>7}{:>19}{:>11}{:>11}{:>12}{:>11}{:>12}".format(
            "step", "R_exp", "N_exp", "lr R/N/C", "full_rep", "rep_errors",
            "full_C_top1", "full_C_top5", "full_naming"))
        for m in a["milestones"]:
            print("  {:>8}{:>7.0f}{:>7.0f}{:>19}{:>11.6f}{:>11.0f}{:>12.6f}"
                  "{:>11.6f}{:>12.6f}".format(
                      m["step"], m["R_exposures"], m["N_exposures"],
                      format_lrs({"R": m["lr_R"], "N": m["lr_N"],
                                  "C": m["lr_C"]}),
                      m["full_rep_full"],
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
