"""Post-run audit of the FINAL base-123 8-run block (metrics/provenance only).

Reads each run's config.json, provenance.json and metrics.tsv -- no
checkpoints, so it is cheap and runs anywhere -- and produces:

  1. a recipe/provenance cross-check over all 8 runs, flagging any field that
     is not identical where it must be, and confirming the only fields that
     may differ are seed and the two ventral widths;
  2. exact ERROR COUNTS at every full-evaluation milestone (fraction x
     population), for canonical R, genuine free-AR R, N and strict C;
  3. paired H256-vs-H512 differences by seed, with mean/SD/min/max and a
     sign-consistency flag (n=4: descriptive only, no significance claim);
  4. a convergence diagnostic for each run: per-milestone slope of top1 and,
     more informatively, the RESIDUAL-ERROR RATIO between consecutive
     milestones.  A constant ratio is geometric convergence; a ratio rising
     toward 1.0 is an asymptote;
  5. route health across milestones (LTM/WM repetition, gate statistics);
  6. an LR-transition window report around the repetition-cursor boundary.

Everything is written as TSV/CSV for downstream use plus a markdown summary.
Nothing is extrapolated: trends are reported as observed ratios, never as
predicted future values.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

N_REP = 29_571
N_NAM = 29_571
N_COMP = 27_981
R_PASS, C_PASS, CYCLE = 463, 438, 6
LR_BOUNDARY_R_BATCHES = 46_300          # = 100 R exposures

# Fields that MUST be identical across all eight runs.
INVARIANT = [
    "regime", "subset_mode", "schedule", "optimizer_policy", "lambda_C",
    "lambda_N", "tau", "c_align_weight", "batch_size", "grad_clip",
    "weight_decay", "lr_stage1", "lr_stage2", "lr_boundary_steps",
    "dorsal_pool_size", "repetition_population", "naming_population",
    "comprehension_population", "retrieval_bank_size",
    "comprehension_population_sha256", "naming_population_sha256",
    "lexicon_file_sha256", "hidden_size", "loss_weights", "max_steps",
    "full_eval_at", "repetition_sampler", "glove_found", "glove_fallback",
]
# Fields that are ALLOWED to differ (and must, in the intended way).
VARIANT = ["seed", "ltm_enc_hidden", "ltm_dec_hidden"]


def read_tsv(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def num(v):
    if v in (None, "", "nan", "NaN"):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def u_of(step: int) -> float:
    """Joint progress variable: u = repetition exposures."""
    return step / (R_PASS * CYCLE)


def flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, f"{key}."))
        else:
            out[key] = v
    return out


class Run:
    def __init__(self, root: str, run_id: str):
        self.run_id = run_id
        self.dir = os.path.join(root, run_id)
        self.config = self._load("config.json")
        self.provenance = self._load("provenance.json")
        self.metrics = []
        m = os.path.join(self.dir, "metrics.tsv")
        if os.path.exists(m):
            self.metrics = read_tsv(m)
        self.flatcfg = flat(self.config)

    def _load(self, name):
        # a resumed run writes config_from_step_*.json; the primary file is
        # the first launch's, which is the one that defines the experiment
        p = os.path.join(self.dir, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def get(self, field):
        for key in (field, f"resolved_settings.{field}", f"settings.{field}"):
            if key in self.flatcfg:
                return self.flatcfg[key]
        for k, v in self.flatcfg.items():
            if k.endswith("." + field) or k == field:
                return v
        return None

    def full_rows(self) -> List[dict]:
        """Rows carrying a full-population evaluation, by step."""
        rows = [r for r in self.metrics if num(r.get("full_rep_full")) is not None]
        seen, out = set(), []
        for r in sorted(rows, key=lambda r: int(r["step"])):
            s = int(r["step"])
            if s not in seen:            # endpoint-eval can duplicate a step
                seen.add(s)
                out.append(r)
        return out


def milestone_table(runs: List[Run]) -> List[dict]:
    out = []
    for run in runs:
        for r in run.full_rows():
            step = int(r["step"])
            rec = {
                "run_id": run.run_id, "step": step, "u": round(u_of(step), 4),
                "r_exposures": r.get("r_exposures"),
                "n_exposures": r.get("n_exposures"),
                "c_exposures": r.get("c_exposures"),
                "lr": r.get("lr"),
            }
            pairs = [("rep_canonical", "full_rep_full", N_REP),
                     ("rep_freear", "full_rep_freear", N_REP),
                     ("rep_wm", "full_rep_wm", N_REP),
                     ("rep_ltm", "full_rep_ltm", N_REP),
                     ("rep_freear_wm", "full_rep_freear_wm", N_REP),
                     ("rep_freear_ltm", "full_rep_freear_ltm", N_REP),
                     ("naming", "full_naming_exact", N_NAM),
                     ("comp_top1", "full_comp_top1", N_COMP),
                     ("comp_top5", "full_comp_top5", N_COMP)]
            for label, col, pop in pairs:
                v = num(r.get(col))
                rec[label] = v
                rec[f"{label}_errors"] = (None if v is None
                                          else int(round((1.0 - v) * pop)))
            # metrics that are counts already, if the driver logged them
            for col in ("full_rep_errors", "full_rep_freear_errors",
                        "full_naming_errors", "full_comp_errors"):
                if r.get(col) not in (None, ""):
                    rec[col + "_logged"] = r[col]
            for col in ("full_comp_rank_median", "full_comp_rank_mean",
                        "full_comp_cos_mean", "full_comp_margin_mean",
                        "full_naming_wer", "full_naming_mean_edit",
                        "full_naming_eos_rate", "gate_mean", "gate_std",
                        "gate_p05", "gate_p95", "gate_frac_below_0.05",
                        "gate_frac_above_0.95"):
                rec[col] = num(r.get(col))
            # do the two repetition conventions genuinely agree?
            a, b = rec["rep_canonical"], rec["rep_freear"]
            rec["rep_conventions_identical"] = (
                None if a is None or b is None else int(a == b))
            rec["rep_convention_gap_errors"] = (
                None if a is None or b is None
                else int(round((a - b) * N_REP)))
            out.append(rec)
    return out


def convergence(runs: List[Run]) -> List[dict]:
    """Residual-error ratio between consecutive milestones.

    ratio = errors(next) / errors(prev).  Constant ratio  -> geometric
    convergence.  Ratio climbing toward 1.0 -> approaching an asymptote.
    Reported, never extrapolated.
    """
    out = []
    for run in runs:
        rows = run.full_rows()
        for metric, col, pop in (("comp_top1", "full_comp_top1", N_COMP),
                                 ("naming", "full_naming_exact", N_NAM),
                                 ("rep_canonical", "full_rep_full", N_REP),
                                 ("rep_ltm", "full_rep_ltm", N_REP)):
            prev = None
            for r in rows:
                v = num(r.get(col))
                if v is None:
                    continue
                step = int(r["step"])
                e = (1.0 - v) * pop
                rec = {"run_id": run.run_id, "metric": metric,
                       "u": round(u_of(step), 4), "value": v,
                       "errors": round(e, 2)}
                if prev is not None:
                    pu, pe, pv = prev
                    du = u_of(step) - pu
                    rec.update({
                        "prev_u": round(pu, 4),
                        "delta_u": round(du, 4),
                        "delta_value": round(v - pv, 6),
                        "errors_removed": round(pe - e, 2),
                        "residual_ratio": (round(e / pe, 4) if pe > 0 else None),
                        "slope_per_100u": (round((v - pv) / du * 100, 6)
                                           if du else None)})
                out.append(rec)
                prev = (u_of(step), e, v)
    return out


def paired(runs: Dict[str, Run], seeds, metrics) -> List[dict]:
    out = []
    for label, col, pop in metrics:
        diffs = []
        for s in seeds:
            a = runs.get(f"final_base123_h256_s{s}")
            b = runs.get(f"final_base123_h512_s{s}")
            if not a or not b or not a.full_rows() or not b.full_rows():
                continue
            ra, rb = a.full_rows()[-1], b.full_rows()[-1]
            va, vb = num(ra.get(col)), num(rb.get(col))
            if va is None or vb is None:
                continue
            diffs.append(vb - va)
            out.append({"metric": label, "seed": s, "h256": va, "h512": vb,
                        "delta_h512_minus_h256": round(vb - va, 6),
                        "h256_errors": int(round((1 - va) * pop)),
                        "h512_errors": int(round((1 - vb) * pop)),
                        "kind": "per_seed"})
        if len(diffs) >= 2:
            out.append({"metric": label, "seed": "ALL", "kind": "summary",
                        "n_seeds": len(diffs),
                        "mean_delta": round(st.mean(diffs), 6),
                        "sd_delta": round(st.stdev(diffs), 6),
                        "min_delta": round(min(diffs), 6),
                        "max_delta": round(max(diffs), 6),
                        "same_sign_all_seeds": int(
                            all(d > 0 for d in diffs) or all(d < 0 for d in diffs))})
    return out


def lr_window(runs: List[Run]) -> List[dict]:
    """Full evaluations bracketing the repetition-cursor LR boundary."""
    out = []
    for run in runs:
        for r in run.full_rows():
            rexp = num(r.get("r_exposures"))
            if rexp is None or not (40 <= rexp <= 250):
                continue
            out.append({
                "run_id": run.run_id, "u": round(u_of(int(r["step"])), 4),
                "r_exposures": rexp, "n_exposures": num(r.get("n_exposures")),
                "c_exposures": num(r.get("c_exposures")), "lr": num(r.get("lr")),
                "stage": ("stage1_1e-3" if (num(r.get("lr")) or 0) > 5e-4
                          else "stage2_1e-4"),
                "rep_canonical": num(r.get("full_rep_full")),
                "rep_freear": num(r.get("full_rep_freear")),
                "rep_ltm": num(r.get("full_rep_ltm")),
                "rep_wm": num(r.get("full_rep_wm")),
                "naming": num(r.get("full_naming_exact")),
                "comp_top1": num(r.get("full_comp_top1")),
                "gate_mean": num(r.get("gate_mean"))})
    return out


def loss_window(run: Run, lo_u=50.0, hi_u=200.0) -> List[dict]:
    """Per-task training loss / grad norm around the LR boundary."""
    p = os.path.join(run.dir, "logs", "losses.tsv")
    if not os.path.exists(p):
        return []
    out = []
    for r in read_tsv(p):
        step = int(r.get("step", 0) or 0)
        u = u_of(step)
        if not (lo_u <= u <= hi_u):
            continue
        out.append({"run_id": run.run_id, "u": round(u, 4), "step": step,
                    "task": r.get("task"), "lr": num(r.get("lr")),
                    "grad_norm": num(r.get("grad_norm")),
                    "joint_total": num(r.get("joint_total")),
                    "naming_ce": num(r.get("naming_ce")),
                    "retrieval_ce": num(r.get("retrieval_ce")),
                    "rep": num(r.get("rep")), "dec": num(r.get("dec")),
                    "pool_ce": num(r.get("pool_ce"))})
    return out


def write(path, rows, delimiter="\t"):
    if not rows:
        print(f"[audit] (no rows) {path}")
        return
    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter=delimiter,
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[audit] wrote {path}  ({len(rows)} rows)")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", default="19,20,21,22")
    ap.add_argument("--widths", default="256,512")
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",")]
    widths = [int(w) for w in args.widths.split(",")]
    ids = [f"final_base123_h{w}_s{s}" for w in widths for s in seeds]
    runs = {i: Run(args.runs_root, i) for i in ids}
    present = [r for r in runs.values() if r.metrics]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[audit] {len(present)}/{len(ids)} runs have metrics.tsv")
    for i in ids:
        if not runs[i].metrics:
            print(f"[audit] MISSING metrics for {i}")

    # ---- 1. provenance / recipe cross-check -----------------------------
    prov = []
    for i in ids:
        run = runs[i]
        rec = {"run_id": i}
        for f in VARIANT + INVARIANT:
            rec[f] = run.get(f)
        g = (run.provenance or {}).get("git") or run.get("git") or {}
        rec["git_commit"] = g.get("commit") if isinstance(g, dict) else None
        rec["git_tracked_dirty"] = (g.get("tracked_dirty")
                                    if isinstance(g, dict) else None)
        anc = (run.provenance or {}).get("ancestry") or {}
        rec["is_branch"] = anc.get("is_branch")
        rec["source_checkpoint"] = anc.get("source_checkpoint")
        rec["n_full_evals"] = len(run.full_rows())
        rec["last_u"] = (round(u_of(int(run.full_rows()[-1]["step"])), 4)
                         if run.full_rows() else None)
        prov.append(rec)
    write(os.path.join(args.out_dir, "provenance_matrix.tsv"), prov)

    disc = []
    for f in INVARIANT:
        vals = {}
        for rec in prov:
            vals.setdefault(json.dumps(rec.get(f), sort_keys=True,
                                       default=str), []).append(rec["run_id"])
        if len(vals) > 1:
            disc.append({"field": f, "n_distinct": len(vals),
                         "groups": json.dumps(
                             {k: v for k, v in vals.items()}, default=str)})
    for rec in prov:
        w = int(rec["run_id"].split("_h")[1].split("_")[0])
        s = int(rec["run_id"].rsplit("_s", 1)[1])
        for field, expect in (("seed", s), ("ltm_enc_hidden", w),
                              ("ltm_dec_hidden", w), ("hidden_size", 128)):
            got = rec.get(field)
            if got is not None and int(got) != expect:
                disc.append({"field": field, "n_distinct": -1,
                             "groups": f"{rec['run_id']}: expected {expect}, got {got}"})
    write(os.path.join(args.out_dir, "recipe_discrepancies.tsv"), disc)
    print(f"[audit] recipe discrepancies: {len(disc)}"
          + ("  <-- INVESTIGATE" if disc else "  (none)"))

    # ---- 2/5/9. milestones with exact error counts ----------------------
    ms = milestone_table(present)
    write(os.path.join(args.out_dir, "milestones.tsv"), ms)
    write(os.path.join(args.out_dir, "milestones.csv"), ms, delimiter=",")

    # ---- 3. paired capacity effect --------------------------------------
    pr = paired(runs, seeds,
                [("rep_canonical", "full_rep_full", N_REP),
                 ("rep_freear", "full_rep_freear", N_REP),
                 ("naming", "full_naming_exact", N_NAM),
                 ("comp_top1", "full_comp_top1", N_COMP),
                 ("comp_top5", "full_comp_top5", N_COMP),
                 ("rep_ltm", "full_rep_ltm", N_REP),
                 ("rep_wm", "full_rep_wm", N_REP),
                 ("gate_mean", "gate_mean", 1)])
    write(os.path.join(args.out_dir, "paired_capacity_effect.tsv"), pr)

    # ---- 4. convergence --------------------------------------------------
    write(os.path.join(args.out_dir, "convergence.tsv"), convergence(present))

    # ---- 8. LR transition -------------------------------------------------
    write(os.path.join(args.out_dir, "lr_transition_window.tsv"),
          lr_window(present))
    lw = []
    for r in present:
        lw.extend(loss_window(r))
    write(os.path.join(args.out_dir, "lr_transition_losses.tsv"), lw)

    # ---- markdown summary -------------------------------------------------
    md = [f"# FINAL base-123 post-run audit", "",
          f"Runs root: `{args.runs_root}`", "",
          f"- runs with metrics: {len(present)}/{len(ids)}",
          f"- recipe discrepancies: **{len(disc)}**", ""]
    md.append("## Endpoint (last full evaluation per run)\n")
    md.append("| run | u | R canon | R errs | R freeAR | freeAR errs | "
              "N | N errs | C top1 | C errs | LTM rep | gate mean |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i in ids:
        rows = [m for m in ms if m["run_id"] == i]
        if not rows:
            continue
        m = rows[-1]
        def f(x, n=6):
            return "-" if x is None else f"{x:.{n}f}"
        md.append(f"| {i} | {m['u']:.0f} | {f(m['rep_canonical'])} | "
                  f"{m['rep_canonical_errors']} | {f(m['rep_freear'])} | "
                  f"{m['rep_freear_errors']} | {f(m['naming'])} | "
                  f"{m['naming_errors']} | {f(m['comp_top1'])} | "
                  f"{m['comp_top1_errors']} | {f(m['rep_ltm'])} | "
                  f"{f(m['gate_mean'], 4)} |")
    md.append("")
    md.append("## Residual-error ratio (errors_next / errors_prev)\n")
    md.append("A ratio roughly constant across milestones is geometric "
              "convergence; a ratio climbing toward 1.0 indicates an "
              "asymptote. No extrapolation is performed.\n")
    open(os.path.join(args.out_dir, "SUMMARY.md"), "w").write("\n".join(md))
    print(f"[audit] wrote {os.path.join(args.out_dir, 'SUMMARY.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
