"""Build the Lichtheim3 final results table and publication figures.

Reads ONLY the archived `metrics.tsv` of each run -- never rounded stdout,
never a hand-copied summary -- and emits:

  results_table.tsv / .csv   one row per (stage, seed) with exact values
  results_summary.tsv / .md  mean +/- sample SD over seeds, per stage
  figA_capacity.{pdf,png}    H256 vs H512 joint trajectories
  figA2_dorsal.{pdf,png}     dorsal WM width probe (separate: not comparable)
  figB_global_lr.{pdf,png}   u750 branch, 1e-4 vs all-3e-5
  figC_task_lr.{pdf,png}     u850 branch, all-3e-5 vs C-high
  figD_route_health.{pdf,png}  FULL / WM-only / LTM-only repetition
  figE_paired_endpoint.{pdf,png}  paired per-seed endpoint effects
  missing_artifacts.txt      anything that could not be found

Seed variability is never hidden: every panel draws faint per-seed
trajectories under the mean, and the endpoint figure is paired per seed.

Stages are declared once in STAGES, so the table and the figures can never
disagree about which directory a number came from.
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
R_PASS, CYCLE = 463, 6
SEEDS = [19, 20, 21, 22]

# (stage key, label, run-id template, u at which to read, colour, arm group)
STAGES = [
    ("h256_u500", "baseline H256 u500", "final_base123_h256_s{s}", 500,
     "tab:blue", "capacity"),
    ("h512_u500", "baseline H512 u500", "final_base123_h512_s{s}", 500,
     "tab:red", "capacity"),
    ("h512_u750", "H512 u750", "final_base123_h512_s{s}", 750,
     "tab:red", "capacity"),
    ("lr1e4_u850", "u850 control 1e-4", "final_lrpilot_control_h512_s{s}", 850,
     "tab:gray", "global_lr"),
    ("lr3e5_u850", "u850 all-3e-5", "final_lrpilot_3e5_h512_s{s}", 850,
     "tab:green", "global_lr"),
    ("all3e5_u1200", "u1200 all-3e-5", "final_lrpilot1200_all3e5_h512_s{s}",
     1200, "tab:green", "task_lr"),
    ("chigh_u1200", "u1200 C-high", "final_lrpilot1200_chigh_h512_s{s}", 1200,
     "tab:purple", "task_lr"),
]

# Trajectory sources: (label, run-id template, colour, u-range)
TRAJ_CAPACITY = [("H256", "final_base123_h256_s{s}", "tab:blue", (0, 500)),
                 ("H512", "final_base123_h512_s{s}", "tab:red", (0, 750))]
TRAJ_GLOBAL_LR = [("1e-4 control", "final_lrpilot_control_h512_s{s}",
                   "tab:gray", (750, 850)),
                  ("all-3e-5", "final_lrpilot_3e5_h512_s{s}", "tab:green",
                   (750, 850))]
TRAJ_TASK_LR = [("all-3e-5", "final_lrpilot1200_all3e5_h512_s{s}", "tab:green",
                 (850, 1200)),
                ("C-high", "final_lrpilot1200_chigh_h512_s{s}", "tab:purple",
                 (850, 1200))]
DORSAL = [("WM128", "cap3_wm128_seed22_full", "tab:blue"),
          ("WM256", "cap3_wm256_seed22_full", "tab:orange"),
          ("WM512", "cap3_wm512_seed22_full", "tab:red")]

METRICS = [
    ("rep_exact", "full_rep_full", N_REP),
    ("rep_freear", "full_rep_freear", N_REP),
    ("rep_wm", "full_rep_wm", N_REP),
    ("rep_ltm", "full_rep_ltm", N_REP),
    ("naming_exact", "full_naming_exact", N_NAM),
    ("comp_top1", "full_comp_top1", N_COMP),
    ("comp_top5", "full_comp_top5", N_COMP),
]
GATE = ["gate_mean", "gate_std", "gate_frac_below_0.05", "gate_frac_above_0.95"]


def num(v):
    if v in (None, "", "nan", "NaN"):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def u_of(step: int) -> float:
    return step / (R_PASS * CYCLE)


def full_rows(runs_root: str, run_id: str) -> List[dict]:
    """Rows carrying a full-population evaluation, de-duplicated by step."""
    p = os.path.join(runs_root, run_id, "metrics.tsv")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t")
                if num(r.get("full_rep_full")) is not None]
    out, seen = [], set()
    for r in sorted(rows, key=lambda r: int(r["step"])):
        s = int(r["step"])
        if s not in seen:            # endpoint-eval can duplicate a step
            seen.add(s)
            out.append(r)
    return out


def at_u(rows: List[dict], u: int) -> Optional[dict]:
    want = u * R_PASS * CYCLE
    for r in rows:
        if int(r["step"]) == want:
            return r
    return None


def extract(r: dict) -> dict:
    out = {}
    for key, col, pop in METRICS:
        v = num(r.get(col))
        out[key] = v
        out[key + "_errors"] = None if v is None else int(round((1 - v) * pop))
    for g in GATE:
        out[g] = num(r.get(g))
    for c in ("r_exposures", "n_exposures", "c_exposures", "lr"):
        out[c] = num(r.get(c))
    return out


def mean_sd(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None, 0
    return (round(st.mean(v), 6),
            (round(st.stdev(v), 6) if len(v) > 1 else 0.0), len(v))


def write_tsv(path, rows, delim="\t"):
    if not rows:
        open(path, "w").write("")
        print(f"[report] (0 rows) {path}")
        return
    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter=delim,
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[report] wrote {path}  ({len(rows)} rows)")


# ---------------------------------------------------------------- figures ---

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 140, "font.size": 9,
                         "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def series(runs_root, template, col, pop, u_lo, u_hi, as_errors):
    """Per-seed (u, value) series, plus the mean series over seeds."""
    per = {}
    for s in SEEDS:
        rows = full_rows(runs_root, template.format(s=s))
        xs, ys = [], []
        for r in rows:
            u = u_of(int(r["step"]))
            if not (u_lo <= u <= u_hi):
                continue
            v = num(r.get(col))
            if v is None:
                continue
            xs.append(u)
            ys.append((1 - v) * pop if as_errors else v)
        if xs:
            per[s] = (xs, ys)
    if not per:
        return per, ([], [], [])
    common = sorted(set.intersection(*[set(x) for x, _ in per.values()]))
    mx, my, sd = [], [], []
    for u in common:
        vals = [ys[xs.index(u)] for xs, ys in per.values()]
        mx.append(u); my.append(st.mean(vals))
        sd.append(st.stdev(vals) if len(vals) > 1 else 0.0)
    return per, (mx, my, sd)


def panel(ax, runs_root, arms, col, pop, title, ylabel, as_errors=False,
          branch_u=None, missing=None):
    drew = False
    for label, template, colour, (lo, hi) in arms:
        per, (mx, my, sd) = series(runs_root, template, col, pop, lo, hi,
                                   as_errors)
        if not per:
            if missing is not None:
                missing.append(f"{title}: no data for {template}")
            continue
        drew = True
        for xs, ys in per.values():                 # faint per-seed lines
            ax.plot(xs, ys, color=colour, alpha=0.25, lw=0.9)
        ax.plot(mx, my, color=colour, lw=2.0, marker="o", ms=3, label=label)
        if len(SEEDS) > 1:
            ax.fill_between(mx, [m - s for m, s in zip(my, sd)],
                            [m + s for m, s in zip(my, sd)],
                            color=colour, alpha=0.12, lw=0)
    if branch_u is not None and drew:
        ax.axvline(branch_u, color="k", ls="--", lw=0.8)
        ax.annotate(f"branch u{branch_u}", xy=(branch_u, 1), xycoords=
                    ("data", "axes fraction"), xytext=(3, -10),
                    textcoords="offset points", fontsize=7, color="k")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("u  (repetition exposures)")
    ax.set_ylabel(ylabel)
    if drew:
        ax.legend(fontsize=7, frameon=False)
    return drew


def save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight")
        print(f"[report] wrote {p}")
    import matplotlib.pyplot as plt
    plt.close(fig)


def figure_block(runs_root, out_dir, name, arms, suptitle, branch_u, missing):
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4))
    specs = [("full_comp_top1", N_COMP, "Comprehension errors (strict top-1)",
              "errors / 27,981", True),
             ("full_naming_exact", N_NAM, "Naming errors (free greedy AR)",
              "errors / 29,571", True),
             ("full_rep_full", N_REP, "Repetition errors (canonical)",
              "errors / 29,571", True),
             ("full_rep_ltm", N_REP, "LTM-only repetition (exact)",
              "proportion", False)]
    drew = False
    for ax, (col, pop, title, ylab, as_err) in zip(axes.ravel(), specs):
        drew |= panel(ax, runs_root, arms, col, pop, title, ylab, as_err,
                      branch_u, missing)
    fig.suptitle(suptitle, fontsize=10.5)
    fig.tight_layout()
    if drew:
        save(fig, out_dir, name)
    else:
        plt.close(fig)
        missing.append(f"{name}: no data at all, figure not written")


def figure_route_health(runs_root, out_dir, arms, missing):
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, (col, title) in zip(axes, [
            ("full_rep_full", "FULL route"),
            ("full_rep_wm", "WM-only route"),
            ("full_rep_ltm", "LTM-only route")]):
        panel(ax, runs_root, arms, col, N_REP, title,
              "repetition exact (proportion)", False, 850, missing)
    fig.suptitle("Route health: WM stays at ceiling while the ventral route "
                 "is sensitive to comprehension pressure", fontsize=10.5)
    fig.tight_layout()
    save(fig, out_dir, "figD_route_health")


def figure_paired_endpoint(runs_root, out_dir, missing):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    pairs = [("comp_top1_errors", "full_comp_top1", N_COMP, True,
              "C errors at u1200"),
             ("rep_ltm", "full_rep_ltm", N_REP, False,
              "LTM-only repetition at u1200")]
    for ax, (_, col, pop, as_err, title) in zip(axes, pairs):
        a_vals, b_vals, used = [], [], []
        for s in SEEDS:
            ra = at_u(full_rows(runs_root,
                                f"final_lrpilot1200_all3e5_h512_s{s}"), 1200)
            rb = at_u(full_rows(runs_root,
                                f"final_lrpilot1200_chigh_h512_s{s}"), 1200)
            if not ra or not rb:
                missing.append(f"figE: missing u1200 row for seed {s}")
                continue
            va, vb = num(ra.get(col)), num(rb.get(col))
            if va is None or vb is None:
                continue
            if as_err:
                va, vb = (1 - va) * pop, (1 - vb) * pop
            a_vals.append(va); b_vals.append(vb); used.append(s)
        if not used:
            continue
        for s, va, vb in zip(used, a_vals, b_vals):
            ax.plot([0, 1], [va, vb], "-o", ms=4, lw=1.2, alpha=0.85,
                    label=f"seed {s}")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["all-3e-5", "C-high"])
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Paired endpoint effect at u1200 (one line per seed)",
                 fontsize=10.5)
    fig.tight_layout()
    save(fig, out_dir, "figE_paired_endpoint")


def figure_dorsal(runs_root, out_dir, missing):
    """Separate figure: the dorsal probe is a single-task isolation
    experiment and its numbers are NOT comparable to the joint runs."""
    plt = _plt()
    found = False
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for label, run_id, colour in DORSAL:
        p = os.path.join(runs_root, run_id, "metrics.tsv")
        if not os.path.exists(p):
            missing.append(f"figA2 dorsal: {p} not found")
            continue
        with open(p, encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        xs = [num(r.get("exposures")) for r in rows]
        for ax, col, ylab in ((axes[0], "lex_exact", "lexical exact"),
                              (axes[1], "pseudo_exact", "pseudoword exact")):
            ys = [num(r.get(col)) for r in rows]
            pts = [(x, y) for x, y in zip(xs, ys)
                   if x is not None and y is not None]
            if pts:
                found = True
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", ms=3,
                        color=colour, label=label)
                ax.set_xlabel("exposures / item")
                ax.set_ylabel(ylab)
    for ax, t in zip(axes, ("WM-route lexical repetition",
                            "WM-route pseudoword repetition")):
        ax.set_title(t, fontsize=9)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Dorsal WM width probe (single-task isolation; NOT "
                 "comparable to the joint runs)", fontsize=10)
    fig.tight_layout()
    if found:
        save(fig, out_dir, "figA2_dorsal")
    else:
        plt.close(fig)


# ------------------------------------------------------------------- main ---

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    missing: List[str] = []

    # ---- per-(stage, seed) rows ----------------------------------------
    rows = []
    for key, label, template, u, _c, group in STAGES:
        for s in SEEDS:
            run_id = template.format(s=s)
            fr = full_rows(args.runs_root, run_id)
            if not fr:
                missing.append(f"{key} seed {s}: no metrics.tsv for {run_id}")
                continue
            r = at_u(fr, u)
            if r is None:
                have = sorted({int(u_of(int(x['step']))) for x in fr})
                missing.append(f"{key} seed {s}: no full eval at u={u} in "
                               f"{run_id} (have u={have})")
                continue
            rows.append({"stage": key, "label": label, "group": group,
                         "run_id": run_id, "seed": s, "u": u,
                         "step": int(r["step"]), **extract(r)})
    write_tsv(os.path.join(args.out_dir, "results_table.tsv"), rows)
    write_tsv(os.path.join(args.out_dir, "results_table.csv"), rows, ",")

    # ---- mean +/- sample SD over seeds ----------------------------------
    summary, md = [], []
    md.append("| stage | n | R exact | R errs | R freeAR | N exact | N errs | "
              "C top1 | C errs | C top5 | LTM rep | WM rep | gate mean |")
    md.append("|" + "---|" * 13)
    for key, label, _t, u, _c, group in STAGES:
        sub = [r for r in rows if r["stage"] == key]
        if not sub:
            continue
        rec = {"stage": key, "label": label, "group": group, "u": u,
               "n_seeds": len(sub)}
        for m in ("rep_exact", "rep_exact_errors", "rep_freear",
                  "naming_exact", "naming_exact_errors", "comp_top1",
                  "comp_top1_errors", "comp_top5", "rep_ltm", "rep_wm",
                  "gate_mean"):
            mu, sd, n = mean_sd([r.get(m) for r in sub])
            rec[m + "_mean"] = mu
            rec[m + "_sd"] = sd
        summary.append(rec)

        def c(m, dec=6):
            mu, sd = rec.get(m + "_mean"), rec.get(m + "_sd")
            if mu is None:
                return "-"
            return f"{mu:.{dec}f} ± {sd:.{dec}f}" if sd else f"{mu:.{dec}f}"
        md.append(f"| {label} | {len(sub)} | {c('rep_exact')} | "
                  f"{c('rep_exact_errors',1)} | {c('rep_freear')} | "
                  f"{c('naming_exact')} | {c('naming_exact_errors',1)} | "
                  f"{c('comp_top1')} | {c('comp_top1_errors',1)} | "
                  f"{c('comp_top5')} | {c('rep_ltm')} | {c('rep_wm')} | "
                  f"{c('gate_mean',4)} |")
    write_tsv(os.path.join(args.out_dir, "results_summary.tsv"), summary)
    open(os.path.join(args.out_dir, "results_summary.md"), "w").write(
        "# Lichtheim3 final joint results (mean ± sample SD over seeds)\n\n"
        + "\n".join(md) + "\n\nErrors are exact counts: "
        "(1 − proportion) × population, R/N = 29,571, C = 27,981.\n")
    print("[report] wrote " + os.path.join(args.out_dir, "results_summary.md"))
    print("\n".join(md))

    # ---- paired endpoint deltas ----------------------------------------
    paired = []
    for s in SEEDS:
        a = [r for r in rows if r["stage"] == "all3e5_u1200" and r["seed"] == s]
        b = [r for r in rows if r["stage"] == "chigh_u1200" and r["seed"] == s]
        if not a or not b:
            continue
        a, b = a[0], b[0]
        paired.append({"seed": s,
                       "comp_errors_all3e5": a["comp_top1_errors"],
                       "comp_errors_chigh": b["comp_top1_errors"],
                       "delta_comp_errors": (b["comp_top1_errors"]
                                             - a["comp_top1_errors"]),
                       "ltm_all3e5": a["rep_ltm"], "ltm_chigh": b["rep_ltm"],
                       "delta_ltm": (None if a["rep_ltm"] is None
                                     or b["rep_ltm"] is None
                                     else round(b["rep_ltm"] - a["rep_ltm"], 6)),
                       "rep_errors_all3e5": a["rep_exact_errors"],
                       "rep_errors_chigh": b["rep_exact_errors"],
                       "naming_errors_all3e5": a["naming_exact_errors"],
                       "naming_errors_chigh": b["naming_exact_errors"]})
    write_tsv(os.path.join(args.out_dir, "paired_u1200.tsv"), paired)

    # ---- figures --------------------------------------------------------
    if not args.no_figures:
        figure_block(args.runs_root, args.out_dir, "figA_capacity",
                     TRAJ_CAPACITY,
                     "A. Ventral capacity: H256 vs H512, joint 1:2:3 from "
                     "scratch (faint = seeds, band = ±SD)", None, missing)
        figure_block(args.runs_root, args.out_dir, "figB_global_lr",
                     TRAJ_GLOBAL_LR,
                     "B. Global late-LR pilot from the shared u750 state",
                     750, missing)
        figure_block(args.runs_root, args.out_dir, "figC_task_lr",
                     TRAJ_TASK_LR,
                     "C. Task-specific LR pilot from the shared u850 state",
                     850, missing)
        figure_route_health(args.runs_root, args.out_dir, TRAJ_TASK_LR,
                            missing)
        figure_paired_endpoint(args.runs_root, args.out_dir, missing)
        figure_dorsal(args.runs_root, args.out_dir, missing)

    p = os.path.join(args.out_dir, "missing_artifacts.txt")
    open(p, "w").write("\n".join(missing) + ("\n" if missing else ""))
    print(f"[report] missing artefacts: {len(missing)}"
          + ("  <-- see " + p if missing else "  (none)"))
    for m in missing[:20]:
        print("   " + m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
