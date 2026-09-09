"""Matched three-arm comparison for the SETTLE test u3000 -> u3600.

All three arms keep schedule 1:2:3, so at a given global step they share the
exact R/N/C/pool cursors.  That identity is PROVEN per milestone rather than
assumed, and any milestone where it fails is flagged instead of averaged.

The CHIGH lesson is baked in: the PRIMARY comparison is SMOOTHED -- the mean
over the last four common milestones, paired per seed -- because a single
endpoint milestone manufactured a "winner" out of an exact tie at u3000.
A variance readout (within-run milestone-to-milestone SD of the C error
count) accompanies it, because the settle hypothesis predicts a variance
collapse, and a frozen-but-stable tail is a different scientific outcome
from a churning one at the same mean.

STATE DEPENDENCE.  These arms (C LR 1e-4 / 5e-5 / 3e-5, +600u, 24
milestones) are the exact CANNEAL bracket rerun at u3000 instead of u1400.
At u1400 lowering LOST: paired endpoint dC vs control was +7.75 (5e-5) and
+10.5 (3e-5).  The report prints the new deltas next to those frozen values
so the sign comparison is explicit.

Scope invariants are checked as in the chigh report: global identity for
schedule / ratio / anchor / optimizer policy / widths / R LR / N LR, and
schedule_seed WITHIN each seed only (it is seed*1000003 + 4 by design).
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
SRC_STEP, END_STEP, MS_STEP = 8_334_000, 10_000_800, 69_450
N_MILESTONES = (END_STEP - SRC_STEP) // MS_STEP          # 24

# CANNEAL u1400 -> u2000 frozen paired endpoint deltas vs control (mean over
# seeds), from the immutable canneal_u2000 archive.  Historical constants for
# the state-dependence comparison; never recomputed here.
CANNEAL_U1400_DC = {"5e5": +7.75, "3e5": +10.5}
ARMS = ["ctrl", "5e5", "3e5"]
ARM_LR = {"ctrl": 1e-4, "5e5": 5e-5, "3e5": 3e-5}
SEEDS = [19, 20, 21, 22]
RUN = "final_settle_{arm}_h512_s{seed}"
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
        print(f"[settle] (0 rows) {path}")
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
    print(f"[settle] wrote {path}  ({len(rows)} rows)")


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
    write_tsv(os.path.join(a.out_dir, "settle_by_milestone.tsv"), rows)

    # R and N learning rates are held fixed at 3e-5 in every arm; the logged
    # `lr` column is the repetition lr, so it is a direct check of that.
    off = sorted({(r["arm"], r["lr_repetition_logged"]) for r in rows
                  if r["lr_repetition_logged"] not in (None, 3e-5)})
    print("[settle] repetition-LR check: "
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
    print(f"[settle] cursor-identity rows: {len(proof)}, FAILED: {len(bad)}"
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
    write_tsv(os.path.join(a.out_dir, "settle_summary.tsv"), summ)

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
            for arm in ("5e5", "3e5"):
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
        print(f"[settle] paired at the deepest COMMON milestone: "
              f"{last} (u{SRC_STEP / (R_PASS * CYCLE) + 25 * last:.0f})")
    else:
        print("[settle] no milestone is present in all 12 runs -- nothing paired")
    write_tsv(os.path.join(a.out_dir, "settle_paired.tsv"), paired)
    if paired:
        for arm in ("5e5", "3e5"):
            d = [p["dC_errors"] for p in paired if p["arm"] == arm]
            better = sum(1 for x in d if x < 0)
            print(f"[settle]   C LR {ARM_LR[arm]:g} vs control: endpoint mean dC "
                  f"{st.mean(d):+.2f} errors, better in {better}/{len(d)} seeds")

    # ---- PRIMARY: smoothed last-4 common milestones, paired per seed -----
    # "Common" here means common to every run that produced ANY data: a run
    # that never started drops its seed from the pairing (n_seeds shrinks)
    # instead of erasing the whole analysis, while a lagging run still pulls
    # the common depth down so arms are never compared across different u.
    have = {(r["arm"], r["seed"]) for r in rows}
    common2 = [k for k in range(1, N_MILESTONES + 1)
               if all(any(r["arm"] == arm and r["seed"] == s
                          and r["milestone"] == k for r in rows)
                      for (arm, s) in have)] if have else []
    smooth_ms = common2[-4:]
    smoothed = []
    if len(smooth_ms) == 4:
        m4 = {}
        for arm in ARMS:
            for s in SEEDS:
                v = [r["comp_top1_errors"] for r in rows
                     if r["arm"] == arm and r["seed"] == s
                     and r["milestone"] in smooth_ms]
                if len(v) == 4:
                    m4[(arm, s)] = st.mean(v)
        for arm in ("5e5", "3e5"):
            per = [(s, round(m4[(arm, s)] - m4[("ctrl", s)], 2))
                   for s in SEEDS if (arm, s) in m4 and ("ctrl", s) in m4]
            d = [x for _, x in per]
            if not d:
                continue
            rec = {"arm": arm, "lr_c": ARM_LR[arm],
                   "milestones": ",".join(map(str, smooth_ms)),
                   "dC_last4_per_seed": per,
                   "dC_last4_mean": round(st.mean(d), 3),
                   "dC_last4_sd": round(st.stdev(d), 3) if len(d) > 1 else 0.0,
                   "better_seeds": sum(1 for x in d if x < 0),
                   "n_seeds": len(d),
                   # PREREGISTERED missing-run policy: the formal decision
                   # needs ALL FOUR paired seeds.  A smaller denominator is
                   # descriptive only and can never certify a "3/4" winner.
                   "formal_decision_evaluable": int(len(d) == 4),
                   "canneal_u1400_dC": CANNEAL_U1400_DC[arm]}
            rec["sign_flipped_vs_u1400"] = int(rec["dC_last4_mean"] < 0
                                               < CANNEAL_U1400_DC[arm])
            smoothed.append(rec)
        write_tsv(os.path.join(a.out_dir, "settle_primary_smoothed.tsv"),
                  smoothed)
        print(f"[settle] PRIMARY (smoothed, milestones {smooth_ms}):")
        for rec in smoothed:
            print(f"    C LR {rec['lr_c']:g}: last-4 mean dC "
                  f"{rec['dC_last4_mean']:+.3f} (sd {rec['dC_last4_sd']:.3f}, "
                  f"better {rec['better_seeds']}/{rec['n_seeds']}); at u1400 "
                  f"the SAME treatment gave {rec['canneal_u1400_dC']:+.2f} -> "
                  + ("SIGN FLIPPED: settle supported"
                     if rec["sign_flipped_vs_u1400"] else "sign NOT flipped"))
            if not rec["formal_decision_evaluable"]:
                print(f"    C LR {rec['lr_c']:g}: only {rec['n_seeds']}/4 "
                      "paired seeds -> FORMAL_PREREGISTERED_WINNER = "
                      "INCOMPLETE_FOR_PREREGISTERED_DECISION "
                      "(descriptive numbers above are NOT a formal result)")

    # ---- variance: does the low step size collapse the fluctuation? ------
    var_ms = common2[-8:]
    variance = []
    if len(var_ms) == 8:
        for arm in ARMS:
            sds, rsds = [], []
            for s in SEEDS:
                v = [r["comp_top1_errors"] for r in rows
                     if r["arm"] == arm and r["seed"] == s
                     and r["milestone"] in var_ms]
                w = [r["rep_canonical_errors"] for r in rows
                     if r["arm"] == arm and r["seed"] == s
                     and r["milestone"] in var_ms]
                if len(v) == 8:
                    sds.append(st.stdev(v)); rsds.append(st.stdev(w))
            if sds:
                variance.append({"arm": arm, "lr_c": ARM_LR[arm],
                                 "milestones": ",".join(map(str, var_ms)),
                                 "C_sd_mean": round(st.mean(sds), 4),
                                 "C_sd_per_seed": [round(x, 3) for x in sds],
                                 "R_sd_mean": round(st.mean(rsds), 4)})
        ctrl_sd = next((v["C_sd_mean"] for v in variance
                        if v["arm"] == "ctrl"), None)
        for v in variance:
            v["C_sd_ratio_vs_ctrl"] = (round(v["C_sd_mean"] / ctrl_sd, 4)
                                       if ctrl_sd else None)
        write_tsv(os.path.join(a.out_dir, "settle_variance.tsv"), variance)
        print("[settle] within-run C-error SD over the last 8 milestones:")
        for v in variance:
            print(f"    C LR {v['lr_c']:g}: SD {v['C_sd_mean']:.3f} "
                  f"(ratio vs ctrl {v['C_sd_ratio_vs_ctrl']})  "
                  f"R SD {v['R_sd_mean']:.3f}")

    # ---- PREREGISTERED MECHANICAL DECISION (memo Amendment 1) ------------
    # Every complete-run outcome maps to exactly one branch; no judgment
    # call is made here.  Priority: INCOMPLETE precondition, then
    # CEILING > SUPPORTED > STRUCTURAL > REJECTED > MIXED > NO_CLEAR.
    def last4(arm, s, key):
        v = [r[key] for r in rows if r["arm"] == arm and r["seed"] == s
             and r["milestone"] in smooth_ms]
        return st.mean(v) if len(v) == 4 else None

    def arm_last4(arm, key):
        v = [last4(arm, s, key) for s in SEEDS]
        v = [x for x in v if x is not None]
        return st.mean(v) if v else None

    def guards_for(arm):
        n = arm_last4(arm, "naming_errors")
        rc, rc0 = arm_last4(arm, "rep_canonical_errors"), \
            arm_last4("ctrl", "rep_canonical_errors")
        rf, rf0 = arm_last4(arm, "rep_freear_errors"), \
            arm_last4("ctrl", "rep_freear_errors")
        g = {"naming_last4_mean": None if n is None else round(n, 4),
             "naming_ok": n is not None and n < 0.25,
             "rep_canonical_excess": None if None in (rc, rc0)
             else round(rc - rc0, 4),
             "rep_canonical_ok": None not in (rc, rc0) and rc - rc0 <= 2,
             "rep_freear_excess": None if None in (rf, rf0)
             else round(rf - rf0, 4),
             "rep_freear_ok": None not in (rf, rf0) and rf - rf0 <= 2}
        g["all_ok"] = bool(g["naming_ok"] and g["rep_canonical_ok"]
                           and g["rep_freear_ok"])
        return g

    sm_by = {r["arm"]: r for r in smoothed}
    vr_by = {v["arm"]: v.get("C_sd_ratio_vs_ctrl") for v in variance}
    max_streak = max((r.get("ceiling_streak", 0) for r in rows), default=0)
    complete = (len(smooth_ms) == 4 and len(var_ms) == 8
                and all(arm in sm_by
                        and sm_by[arm]["formal_decision_evaluable"]
                        for arm in ("5e5", "3e5"))
                and {v["arm"] for v in variance} == set(ARMS))
    detail = {"complete": bool(complete), "max_ceiling_streak": max_streak,
              "arms": {}}
    branch = None
    if not complete:
        branch = "INCOMPLETE_FOR_PREREGISTERED_DECISION"
    else:
        for arm in ("5e5", "3e5"):
            rec, g = sm_by[arm], guards_for(arm)
            pm = rec["dC_last4_mean"]
            eligible = (pm < 0 and rec["better_seeds"] >= 3 and g["all_ok"])
            structural = (not eligible and g["all_ok"] and abs(pm) <= 1.0
                          and vr_by.get(arm) is not None
                          and vr_by[arm] <= 0.5
                          and (arm_last4(arm, "comp_top1_errors") or 0) > 0)
            detail["arms"][arm] = {
                "paired_mean": pm, "paired_sd": rec["dC_last4_sd"],
                "better_seeds": rec["better_seeds"],
                "variance_ratio": vr_by.get(arm), "guards": g,
                "eligible_primary_winner": bool(eligible),
                "structural_candidate": bool(structural),
                "sign_flipped_vs_u1400": rec["sign_flipped_vs_u1400"]}
        el = [x for x in ("5e5", "3e5")
              if detail["arms"][x]["eligible_primary_winner"]]
        stc = [x for x in ("5e5", "3e5")
               if detail["arms"][x]["structural_candidate"]]
        winner = None
        if len(el) == 2:
            d5, d3 = (detail["arms"]["5e5"]["paired_mean"],
                      detail["arms"]["3e5"]["paired_mean"])
            winner = "5e5" if abs(d5 - d3) <= 1.0 else min(el, key=lambda x:
                detail["arms"][x]["paired_mean"])
        elif el:
            winner = el[0]
        pm5 = detail["arms"]["5e5"]["paired_mean"]
        pm3 = detail["arms"]["3e5"]["paired_mean"]
        if max_streak >= 5:
            branch = "CEILING_CONFIRMED"
        elif winner:
            assert detail["arms"][winner]["sign_flipped_vs_u1400"], \
                "eligible winner without sign flip is impossible given " \
                "CANNEAL's positive deltas"
            branch = "STATE_DEPENDENT_SETTLING_SUPPORTED"
            detail["winner"] = winner
        elif stc:
            if len(stc) == 2:
                v5, v3 = vr_by["5e5"], vr_by["3e5"]
                pick = "5e5" if abs(v5 - v3) <= 0.05 else min(
                    stc, key=lambda x: vr_by[x])
            else:
                pick = stc[0]
            branch = "STRUCTURAL_TAIL_SUPPORTED"
            detail["structural_arm"] = pick
        elif pm5 > 1.0 and pm3 > 1.0:
            branch = "STATE_DEPENDENT_SETTLING_REJECTED"
        elif min(pm5, pm3) <= 1.0:
            branch = "MIXED_SETTLE_THEN_AUDIT"
        else:
            branch = "NO_CLEAR_DECISION"
        # annotation: material decrease that stalls above zero
        sel = detail.get("winner") or detail.get("structural_arm")
        if sel and len(smooth_ms) >= 3:
            m_prev, m_last = smooth_ms[-3], smooth_ms[-1]   # 50u window
            ratios = []
            for s in SEEDS:
                e0 = [r["comp_top1_errors"] for r in rows
                      if r["arm"] == sel and r["seed"] == s
                      and r["milestone"] == m_prev]
                e1 = [r["comp_top1_errors"] for r in rows
                      if r["arm"] == sel and r["seed"] == s
                      and r["milestone"] == m_last]
                if e0 and e1 and e0[0] > 0:
                    ratios.append(e1[0] / e0[0])
            stall = (bool(ratios) and st.mean(ratios) >= 0.98
                     and (arm_last4(sel, "comp_top1_errors") or 0) > 0)
            detail["stalled_above_zero"] = bool(stall)
            if ratios:
                detail["last_window_ratio"] = round(st.mean(ratios), 6)
    detail["branch"] = branch
    json.dump(detail, open(os.path.join(a.out_dir, "settle_decision.json"),
                           "w"), indent=1)
    print(f"[settle] PREREGISTERED_BRANCH = {branch}"
          + (f"  (winner {detail.get('winner')})" if detail.get("winner")
             else "")
          + (f"  (structural arm {detail.get('structural_arm')})"
             if detail.get("structural_arm") else "")
          + ("  [STALLED_ABOVE_ZERO]" if detail.get("stalled_above_zero")
             else ""))

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
        print(f"[settle] STRICT C = 0 reached in {len(first)} run(s):")
        for (arm, s), h in sorted(first.items()):
            print(f"    arm {arm} seed {s} first at u{h['u']:.0f} "
                  f"step {h['global_step']}")
    else:
        print("[settle] no evaluation reached strict C = 0")

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
        print("[settle] no strict 0/0/0/0 evaluation in any arm")

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
            print(f"[settle] {label} across all arms: "
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
        print("[settle] schedule seed WITHIN each seed across arms: "
              + ("IDENTICAL in every seed" if not unmatched
                 else f"DIFFERS for seeds {unmatched}  <-- confound"))
        print("[settle]   and equal to seed*1000003 + 4: "
              + ("yes in every seed" if not offform else f"NO for {offform}"))
        print("[settle]   (across DIFFERENT seeds it is expected to differ: "
              + ", ".join(f"s{r['seed']}={r['expected']}" for r in sched) + ")")
        miss = [(p["arm"], p["seed"]) for p in policy
                if not p["lr_c_matches_arm"]]
        print("[settle] comprehension LR matches its arm: "
              + ("yes, in all runs" if not miss else f"NO for {miss}"))

    json.dump({"arms": ARM_LR, "seeds": SEEDS,
               "source_step": SRC_STEP, "end_step": END_STEP,
               "n_milestones": N_MILESTONES,
               "paired_at_milestone": last,
               "preregistered_branch": branch,
               "primary_smoothed": smoothed,
               "variance": variance,
               "canneal_u1400_reference_dC": CANNEAL_U1400_DC,
               "cursor_identity_failures": len(bad),
               "c_zero_evaluations": len(czero),
               "ceiling_hits": len(hits), "missing_rows": len(missing),
               "checkpoints_verified": len(policy),
               "schedule_seed_within_seed_failures":
                   sum(1 for r in sched if not r["MATCHED"])},
              open(os.path.join(a.out_dir, "settle_meta.json"), "w"), indent=1)
    if missing:
        open(os.path.join(a.out_dir, "missing.txt"), "w").write(
            "\n".join(missing) + "\n")
        print(f"[settle] {len(missing)} missing milestone rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
