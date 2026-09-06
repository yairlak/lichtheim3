"""Aggregate persistence / switch-trigger report over base-123 error audits.

Consumes the per-item TSVs written by `base123_error_audit.py` (one
`error_audit_u<U>/` per seed per milestone) and answers the preregistered
questions that a single-checkpoint audit cannot:

  * within-seed persistence chains (u500 -> u600 -> u750): intersection,
    union, Jaccard, retained fraction, resolved, new;
  * across-seed structure at the final milestone: all pairwise overlaps, the
    four-way intersection and union, the intersection EXPECTED under
    independent random residual sets, and the enrichment over that chance;
  * the fate of a SENTINEL core (the item set common to all seeds at an
    earlier milestone), per seed and per milestone, with the surviving items
    listed;
  * the C switch-trigger evaluation, scoring each preregistered condition
    separately and refusing to declare the trigger met unless ALL hold;
  * the same persistence/churn treatment for naming and repetition, plus the
    canonical-vs-free-AR disagreement count.

Read-only.  It never touches a checkpoint or a run's training artefacts; it
writes only into --out-dir.

Chance model for the across-seed comparison: if two seeds each err
independently and uniformly on |A| and |B| of N items, the expected
intersection is |A|*|B|/N.  This is a deliberately weak null -- item
difficulty is obviously not uniform -- so a ratio near 1 is strong evidence
AGAINST shared structure, while a large ratio only shows some shared
difficulty exists, not that it is a fixed core.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys
from itertools import combinations
from typing import Dict, List, Optional, Sequence

N_COMP, N_REP, N_NAM = 27_981, 29_571, 29_571
TASK_FILES = {"comprehension": ("comp_errors.tsv", N_COMP),
              "naming": ("naming_errors.tsv", N_NAM),
              "repetition": ("rep_errors.tsv", N_REP)}
MARGIN_EPS = 0.01


def read(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def audit_dir(runs_root: str, seed: int, u: int, width: int) -> str:
    return os.path.join(runs_root, f"final_base123_h{width}_s{seed}",
                        f"error_audit_u{u}")


def load(runs_root, seed, u, width, task):
    fname, _ = TASK_FILES[task]
    rows = read(os.path.join(audit_dir(runs_root, seed, u, width), fname))
    return {r["target_bank_index"]: r for r in rows}


def overlap(a: Dict[str, dict], b: Dict[str, dict], pop: int) -> dict:
    A, B = set(a), set(b)
    inter, union = A & B, A | B
    return {"n_a": len(A), "n_b": len(B), "intersection": len(inter),
            "union": len(union),
            "jaccard": round(len(inter) / len(union), 6) if union else None,
            "frac_a_retained": round(len(inter) / len(A), 6) if A else None,
            "resolved": len(A - B), "new": len(B - A),
            "frac_b_new": round(len(B - A) / len(B), 6) if B else None,
            "expected_intersection_if_independent":
                round(len(A) * len(B) / pop, 3) if pop else None,
            "enrichment_over_chance":
                round(len(inter) / (len(A) * len(B) / pop), 3)
                if A and B and pop else None}


def describe(rows: Sequence[dict], task: str) -> dict:
    if not rows:
        return {"n": 0}
    out = {"n": len(rows)}
    if task == "comprehension":
        rank = [num(r.get("target_rank")) for r in rows]
        marg = [num(r.get("margin_target_minus_top1")) for r in rows]
        cos = [num(r.get("target_cos")) for r in rows]
        top5 = [int(r.get("in_top5", 0) or 0) for r in rows]
        rank = [x for x in rank if x is not None]
        marg = [x for x in marg if x is not None]
        cos = [x for x in cos if x is not None]
        out.update({
            "in_top5": sum(top5),
            "frac_in_top5": round(sum(top5) / len(rows), 6),
            "outside_top5": len(rows) - sum(top5),
            "rank_median": st.median(rank) if rank else None,
            "rank_mean": round(st.mean(rank), 3) if rank else None,
            "rank_max": max(rank) if rank else None,
            "cos_median": round(st.median(cos), 6) if cos else None,
            "margin_min": round(min(marg), 6) if marg else None,
            "margin_median": round(st.median(marg), 6) if marg else None,
            "margin_max": round(max(marg), 6) if marg else None,
            "n_margin_within_0.01": sum(1 for m in marg if m > -MARGIN_EPS),
            "frac_margin_within_0.01":
                round(sum(1 for m in marg if m > -MARGIN_EPS) / len(marg), 6)
                if marg else None,
            "relations": {},
            "mathematically_unavoidable":
                sum(int(r.get("mathematically_unavoidable", 0) or 0)
                    for r in rows)})
        for r in rows:
            k = r.get("relation", "unknown")
            out["relations"][k] = out["relations"].get(k, 0) + 1
    elif task == "naming":
        ed = [num(r.get("edit_distance")) for r in rows]
        ed = [x for x in ed if x is not None]
        out.update({
            "edit_median": st.median(ed) if ed else None,
            "edit_max": max(ed) if ed else None,
            "no_eos": sum(int(r.get("no_eos", 0) or 0) for r in rows),
            "over_generation": sum(int(r.get("over_generation", 0) or 0)
                                   for r in rows)})
    else:
        out.update({
            "convention_disagreements":
                sum(1 for r in rows
                    if str(r.get("convention_disagrees", "")) == "1"),
            "freear_wrong_full":
                sum(1 for r in rows
                    if str(r.get("freear_exact_full", "")) == "0"),
            "canonical_wrong":
                sum(1 for r in rows if str(r.get("canonical_exact", "")) == "0"),
            "wm_route_wrong":
                sum(1 for r in rows
                    if str(r.get("freear_exact_wm", "")) == "0"),
            "ltm_route_wrong":
                sum(1 for r in rows
                    if str(r.get("freear_exact_ltm", "")) == "0")})
    return out


def switch_trigger(final_desc: dict, retentions: List[float],
                   ratio_last_window: Optional[float],
                   core_growth: Optional[float]) -> dict:
    """Score each preregistered condition SEPARATELY and require all of them.

    ratio_last_window is supplied by the caller from metrics (errors_next /
    errors_prev); it is not derivable from the error TSVs alone.
    """
    c = {}
    c["last_window_ratio_ge_0.85"] = {
        "value": ratio_last_window,
        "met": (ratio_last_window is not None and ratio_last_window >= 0.85)}
    mean_ret = round(st.mean(retentions), 6) if retentions else None
    c["within_seed_retention_ge_0.80"] = {
        "value": mean_ret, "per_seed": retentions,
        "met": (mean_ret is not None and mean_ret >= 0.80)}
    ft5 = final_desc.get("frac_in_top5")
    fm = final_desc.get("frac_margin_within_0.01")
    c["survivors_predominantly_near_tie_or_top5"] = {
        "frac_in_top5": ft5, "frac_margin_within_0.01": fm,
        "met": (ft5 is not None and fm is not None
                and ft5 >= 0.90 and fm >= 0.50)}
    c["shared_persistent_core_growing"] = {
        "value": core_growth,
        "met": (core_growth is not None and core_growth > 1.0)}
    met = all(v["met"] for v in c.values())
    return {"conditions": c, "ALL_CONDITIONS_MET": met,
            "verdict": ("margin / hard-negative intervention JUSTIFIED"
                        if met else
                        "margin / hard-negative trigger NOT met -- at least "
                        "one condition fails")}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--seeds", default="19,20,21,22")
    ap.add_argument("--milestones", default="500,600,750",
                    help="u values with an error_audit_u<U> directory")
    ap.add_argument("--sentinel-u", type=int, default=500,
                    help="milestone whose four-seed intersection defines the "
                         "sentinel core")
    ap.add_argument("--last-window-ratio", type=float, default=None,
                    help="errors(final)/errors(previous) from metrics.tsv, "
                         "for the switch-trigger condition")
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",")]
    us = [int(u) for u in args.milestones.split(",")]
    final_u = us[-1]
    os.makedirs(args.out_dir, exist_ok=True)
    W = args.width

    data = {t: {u: {s: load(args.runs_root, s, u, W, t) for s in seeds}
                for u in us} for t in TASK_FILES}
    missing = [(t, u, s) for t in TASK_FILES for u in us for s in seeds
               if not data[t][u][s]]
    for t, u, s in missing:
        print(f"[persist] NOTE no {t} errors for seed {s} at u{u} "
              f"(zero errors, or the audit was not run)")

    report: dict = {"runs_root": os.path.abspath(args.runs_root),
                    "width": W, "seeds": seeds, "milestones": us,
                    "population": {k: v[1] for k, v in TASK_FILES.items()}}

    # ---- 1. per-seed description at each milestone ----------------------
    rows_desc = []
    for t in TASK_FILES:
        for u in us:
            for s in seeds:
                d = describe(list(data[t][u][s].values()), t)
                rows_desc.append({"task": t, "u": u, "seed": s,
                                  **{k: (json.dumps(v) if isinstance(v, dict)
                                         else v) for k, v in d.items()}})
    write_tsv(os.path.join(args.out_dir, "per_milestone_description.tsv"),
              rows_desc)

    # ---- 2. within-seed persistence chains -------------------------------
    chains = []
    for t, (_, pop) in TASK_FILES.items():
        pairs = [(a, b) for i, a in enumerate(us) for b in us[i + 1:]]
        for a, b in pairs:
            for s in seeds:
                o = overlap(data[t][a][s], data[t][b][s], pop)
                chains.append({"task": t, "seed": s, "from_u": a, "to_u": b,
                               **o})
    write_tsv(os.path.join(args.out_dir, "within_seed_persistence.tsv"), chains)

    # ---- 3. across-seed structure at the final milestone -----------------
    across = []
    for t, (_, pop) in TASK_FILES.items():
        for a, b in combinations(seeds, 2):
            o = overlap(data[t][final_u][a], data[t][final_u][b], pop)
            across.append({"task": t, "u": final_u, "seed_a": a, "seed_b": b,
                           **o})
        sets = [set(data[t][final_u][s]) for s in seeds]
        if all(sets):
            four_i = set.intersection(*sets)
            four_u = set.union(*sets)
            prod = 1.0
            for x in sets:
                prod *= len(x) / pop
            across.append({"task": t, "u": final_u, "seed_a": "ALL",
                           "seed_b": "ALL", "intersection": len(four_i),
                           "union": len(four_u),
                           "jaccard": round(len(four_i) / len(four_u), 6)
                           if four_u else None,
                           "expected_intersection_if_independent":
                               round(prod * pop, 4),
                           "enrichment_over_chance":
                               round(len(four_i) / (prod * pop), 2)
                               if prod > 0 else None})
            report[f"{t}_four_way_intersection_{final_u}"] = sorted(four_i)
    write_tsv(os.path.join(args.out_dir, "across_seed_overlap.tsv"), across)

    # ---- 4. sentinel core -------------------------------------------------
    su = args.sentinel_u
    sentinel = []
    if all(data["comprehension"][su][s] for s in seeds):
        core = set.intersection(*[set(data["comprehension"][su][s])
                                  for s in seeds])
        report["sentinel_core_u"] = su
        report["sentinel_core_size"] = len(core)
        report["sentinel_core_items"] = sorted(core)
        per_u = {}
        for u in us:
            still = {s: sorted(core & set(data["comprehension"][u][s]))
                     for s in seeds}
            all4 = sorted(set.intersection(
                *[set(v) for v in still.values()])) if all(
                    data["comprehension"][u][s] for s in seeds) else []
            per_u[u] = {"per_seed_still_wrong":
                        {s: len(v) for s, v in still.items()},
                        "wrong_in_all_four": len(all4),
                        "items_wrong_in_all_four": all4}
            for s in seeds:
                for item in still[s]:
                    r = data["comprehension"][u][s][item]
                    sentinel.append({
                        "sentinel_u": su, "at_u": u, "seed": s,
                        "target_bank_index": item,
                        "target_word": r.get("target_word"),
                        "pred_word": r.get("pred_word"),
                        "target_rank": r.get("target_rank"),
                        "in_top5": r.get("in_top5"),
                        "margin": r.get("margin_target_minus_top1"),
                        "relation": r.get("relation"),
                        "wrong_in_all_four_at_this_u": int(item in all4)})
        report["sentinel_core_fate"] = per_u
    write_tsv(os.path.join(args.out_dir, "sentinel_core_fate.tsv"), sentinel)

    # ---- 5. switch trigger ------------------------------------------------
    prev_u = us[-2] if len(us) >= 2 else None
    retentions = []
    if prev_u is not None:
        for s in seeds:
            o = overlap(data["comprehension"][prev_u][s],
                        data["comprehension"][final_u][s], N_COMP)
            if o["frac_a_retained"] is not None:
                retentions.append(o["frac_a_retained"])
    final_desc = describe(
        [r for s in seeds for r in data["comprehension"][final_u][s].values()],
        "comprehension")
    core_growth = None
    if "sentinel_core_fate" in report and prev_u in report["sentinel_core_fate"]:
        a = report["sentinel_core_fate"][prev_u]["wrong_in_all_four"]
        b = report["sentinel_core_fate"][final_u]["wrong_in_all_four"]
        core_growth = round(b / a, 4) if a else None
    report["c_switch_trigger"] = switch_trigger(
        final_desc, retentions, args.last_window_ratio, core_growth)
    report["c_final_pooled_description"] = final_desc

    p = os.path.join(args.out_dir, "persistence_report.json")
    json.dump(report, open(p, "w"), indent=1, default=str)
    print(f"[persist] wrote {p}")
    tr = report["c_switch_trigger"]
    print(f"[persist] C SWITCH TRIGGER: {tr['verdict']}")
    for k, v in tr["conditions"].items():
        print(f"    {'MET    ' if v['met'] else 'NOT MET'}  {k}: "
              f"{ {kk: vv for kk, vv in v.items() if kk != 'met'} }")
    return 0


def write_tsv(path: str, rows: List[dict]) -> None:
    if not rows:
        open(path, "w").write("")
        print(f"[persist] (0 rows) {path}")
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
    print(f"[persist] wrote {path}  ({len(rows)} rows)")


if __name__ == "__main__":
    sys.exit(main())
