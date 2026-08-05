"""M1/M2/M3/M5 analyses over the validated instrumented outputs.

Analysis-only: reads `instrumented/` and the frozen canonical tables, writes
per-phase TSVs.  No model is loaded and no inference is run here.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from typing import List

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "outputs/length_effect_mechanism_93a577f")
INSTR = os.path.join(OUT, "instrumented")
SEEDS = [19, 20, 21, 22]
PRIMARY = ["TRAINED_REAL_EXACT", "NOVEL_PSEUDOWORD"]
EXT = "UNTRAINED_REAL"
BOOT_B, BOOT_SEED = 10000, 20260730


def edit_distance(a: List[str], b: List[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def toks(s):
    return [] if (pd.isna(s) or s == "") else str(s).split()


def hier_boot(per_seed_vals, b=BOOT_B, seed=BOOT_SEED):
    """Seed-level bootstrap (seeds resampled with replacement)."""
    v = np.asarray([x for x in per_seed_vals if np.isfinite(x)], float)
    if v.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    d = v[rng.integers(0, v.size, size=(b, v.size))].mean(axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 2 or np.std(x) == 0:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


# =============================================================== M1
def m1(item, ts, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ar = ts[ts.decode_mode == "autoregressive"]
    ev_rows, burden = [], []
    for (seed, iid, route), g in ar.groupby(["seed", "item_id", "route"]):
        if route == "wm":
            continue
        pref = "ltm_generated" if route == "ltm" else "full_generated"
        g = g[g.prefix_source == pref].sort_values("timestep")
        if g.empty:
            continue
        meta = item[(item.seed == seed) & (item.item_id == iid)].iloc[0]
        tgt = toks(meta.target_tokens)
        L = len(tgt)
        pid = g.predicted_token.tolist()
        tid = g.target_token.tolist()
        eos_pos = meta.ltm_eos_position if route == "ltm" else meta.full_eos_position
        eos_pos = None if pd.isna(eos_pos) else int(eos_pos)
        first_mm = next((k for k in range(len(pid)) if pid[k] != tid[k]), None)
        first_non_eos = next((k for k in range(len(pid))
                              if pid[k] != tid[k] and pid[k] != "<eos>"), None)
        first_eos = eos_pos
        ev_rows.append({"seed": seed, "item_id": iid, "route": route,
                        "exposure_status": meta.exposure_status,
                        "source_label": meta.source_label, "length": L,
                        "FIRST_TOKEN_MISMATCH": first_mm,
                        "FIRST_NON_EOS_MISMATCH": first_non_eos,
                        "FIRST_PREMATURE_EOS": first_eos,
                        "first_divergence_type": (
                            "NONE" if first_mm is None else
                            "EOS" if (first_eos is not None and first_eos == first_mm)
                            else "NON_EOS")})
        if first_mm is not None:
            pred = toks(meta.instrumented_ltm_prediction if route == "ltm"
                        else meta.instrumented_full_prediction)
            rem = L - first_mm
            suf_t, suf_p = tgt[first_mm:], pred[first_mm:]
            direct = sum(1 for a, b in zip(suf_t, suf_p) if a != b) + abs(len(suf_t) - len(suf_p))
            pre = g[g.timestep < first_mm]
            post = g[g.timestep >= first_mm]
            burden.append({
                "seed": seed, "item_id": iid, "route": route,
                "exposure_status": meta.exposure_status, "length": L,
                "first_divergence_position": first_mm,
                "target_positions_remaining": rem,
                "direct_mismatches_after": direct,
                "suffix_levenshtein": edit_distance(suf_t, suf_p),
                "fraction_suffix_wrong": direct / rem if rem else np.nan,
                "eos_shortfall": (L - first_eos) if first_eos is not None else np.nan,
                "margin_before": float(pre.target_margin.mean()) if len(pre) else np.nan,
                "margin_after": float(post.target_margin.mean()) if len(post) else np.nan,
                "entropy_before": float(pre.entropy.mean()) if len(pre) else np.nan,
                "entropy_after": float(post.entropy.mean()) if len(post) else np.nan,
                "divergence_type": ("EOS" if (first_eos is not None and first_eos == first_mm)
                                    else "NON_EOS")})
    ev = pd.DataFrame(ev_rows)
    ev.to_csv(os.path.join(out_dir, "first_error_events.tsv"), sep="\t", index=False)
    bd = pd.DataFrame(burden)
    bd.to_csv(os.path.join(out_dir, "post_divergence_burden.tsv"), sep="\t", index=False)

    # hazard: denominator = event-free before t AND has a target token at t
    hz = []
    for event in ("FIRST_TOKEN_MISMATCH", "FIRST_NON_EOS_MISMATCH",
                  "FIRST_PREMATURE_EOS"):
        for (seed, route, expo), g in ev.groupby(["seed", "route", "exposure_status"]):
            for t in range(0, 9):
                at_risk = g[(g.length > t) &
                            ((g[event].isna()) | (g[event] >= t))]
                n_at = len(at_risk)
                n_ev = int((at_risk[event] == t).sum())
                hz.append({"event": event, "seed": seed, "route": route,
                           "exposure_status": expo, "position": t,
                           "n_at_risk": n_at, "n_events": n_ev,
                           "hazard": (n_ev / n_at) if n_at else np.nan,
                           "length_group": "all"})
    hzd = pd.DataFrame(hz)
    hzd.to_csv(os.path.join(out_dir, "first_error_hazard.tsv"), sep="\t", index=False)
    return ev, bd, hzd


# =============================================================== M2
def m2(item, ts, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ltm_ar = ts[(ts.route == "ltm") & (ts.prefix_source == "ltm_generated")]
    ltm_gp = ts[(ts.route == "ltm") & (ts.prefix_source == "gold")]
    rows = []
    for name, d in (("autoregressive", ltm_ar), ("gold_prefix", ltm_gp)):
        m = d.merge(item[["seed", "item_id", "exposure_status", "phoneme_length"]],
                    on=["seed", "item_id"])
        g = m.groupby(["seed", "exposure_status", "absolute_position"]).agg(
            top1_rate=("is_correct", "mean"),
            target_rank=("target_rank", "mean"),
            target_margin=("target_margin", "mean"),
            entropy=("entropy", "mean"), n=("is_correct", "size")).reset_index()
        g["stream"] = name
        rows.append(g)
    pos = pd.concat(rows, ignore_index=True)
    pos.to_csv(os.path.join(out_dir, "position_profiles.tsv"), sep="\t", index=False)

    # gold-prefix tokenwise argmax word-level outcomes and length slopes
    gp = ltm_gp.merge(item[["seed", "item_id", "exposure_status", "phoneme_length",
                            "target_tokens", "instrumented_ltm_prediction"]],
                      on=["seed", "item_id"])
    wl = []
    for (seed, iid), g in gp.groupby(["seed", "item_id"]):
        g = g.sort_values("timestep")
        L = int(g.phoneme_length.iloc[0])
        # gold-prefix tokenwise argmax over the L form positions (exclude final EOS slot)
        pred = g.predicted_token.tolist()[:L]
        tgt = g.target_token.tolist()[:L]
        wl.append({"seed": seed, "item_id": iid,
                   "exposure_status": g.exposure_status.iloc[0],
                   "phoneme_length": L,
                   "gp_edit_distance": edit_distance(tgt, pred),
                   "gp_word_error": int(pred != tgt)})
    wl = pd.DataFrame(wl)
    ar_w = item[["seed", "item_id", "exposure_status", "phoneme_length",
                 "target_tokens", "instrumented_ltm_prediction"]].copy()
    ar_w["ar_edit_distance"] = [edit_distance(toks(a), toks(b)) for a, b in
                                zip(ar_w.target_tokens, ar_w.instrumented_ltm_prediction)]
    ar_w["ar_word_error"] = (ar_w.ar_edit_distance > 0).astype(int)
    w = wl.merge(ar_w[["seed", "item_id", "ar_edit_distance", "ar_word_error"]],
                 on=["seed", "item_id"])
    w.to_csv(os.path.join(out_dir, "word_level_ar_vs_gold.tsv"), sep="\t", index=False)

    sl = []
    for (seed, expo), g in w.groupby(["seed", "exposure_status"]):
        sl.append({"seed": seed, "exposure_status": expo,
                   "ar_edit_slope": ols(g.phoneme_length, g.ar_edit_distance),
                   "gp_edit_slope": ols(g.phoneme_length, g.gp_edit_distance),
                   "ar_minus_gp_edit_slope": ols(g.phoneme_length, g.ar_edit_distance)
                   - ols(g.phoneme_length, g.gp_edit_distance),
                   "ar_word_error_slope": ols(g.phoneme_length, g.ar_word_error),
                   "gp_word_error_slope": ols(g.phoneme_length, g.gp_word_error),
                   "mean_ar_edit": float(g.ar_edit_distance.mean()),
                   "mean_gp_edit": float(g.gp_edit_distance.mean())})
    sl = pd.DataFrame(sl)
    boot = []
    for expo, g in sl.groupby("exposure_status"):
        lo, hi = hier_boot(g.ar_minus_gp_edit_slope.tolist())
        boot.append({"exposure_status": expo,
                     "quantity": "ar_minus_gp_edit_slope",
                     "mean_over_seeds": float(g.ar_minus_gp_edit_slope.mean()),
                     "seed_values": "; ".join(f"{r.seed}:{r.ar_minus_gp_edit_slope:+.4f}"
                                              for r in g.itertuples()),
                     "boot_lo": lo, "boot_hi": hi,
                     "interval_label": "seed-level bootstrap over four checkpoints, B=10000, seed=20260730"})
    sl.to_csv(os.path.join(out_dir, "length_slopes_ar_vs_gold.tsv"), sep="\t", index=False)
    pd.DataFrame(boot).to_csv(os.path.join(out_dir, "slope_contrast_bootstrap.tsv"),
                              sep="\t", index=False)
    return pos, w, sl, pd.DataFrame(boot)


# =============================================================== M3
def m3(item, neigh, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    nb = neigh.sort_values(["seed", "item_id", "rank"])
    top1 = nb[nb["rank"] == 1].set_index(["seed", "item_id"])
    phon_by = {k: v["phonology"].tolist() for k, v in nb.groupby(["seed", "item_id"])}
    train_forms = set(nb.phonology.unique())

    rows = []
    for r in item.itertuples():
        key = (r.seed, r.item_id)
        tgt, pred = toks(r.target_tokens), toks(r.instrumented_ltm_prediction)
        n1 = toks(top1.loc[key, "phonology"])
        topk = [toks(p) for p in phon_by[key]]
        d_pt = edit_distance(pred, tgt)
        d_pn1 = edit_distance(pred, n1)
        d_pnk = min(edit_distance(pred, p) for p in topk)
        d_tn1 = edit_distance(tgt, n1)
        d_tnk = min(edit_distance(tgt, p) for p in topk)
        pred_s = " ".join(pred)
        rows.append({"seed": r.seed, "item_id": r.item_id,
                     "exposure_status": r.exposure_status,
                     "source_label": r.source_label,
                     "phoneme_length": r.phoneme_length,
                     "confidence": r.confidence, "gate": r.gate,
                     "top1_similarity": r.top1_similarity,
                     "top1_top2_margin": r.top1_top2_margin,
                     "bank_density": r.bank_density,
                     "d_pred_target": d_pt, "d_pred_top1": d_pn1,
                     "d_pred_topk_min": d_pnk, "d_target_top1": d_tn1,
                     "d_target_topk_min": d_tnk,
                     "pred_is_training_form": int(pred_s in train_forms and pred_s != ""),
                     "pred_equals_top1": int(pred == n1),
                     "pred_equals_any_topk": int(any(pred == p for p in topk)),
                     "correct": int(d_pt == 0)})
    d = pd.DataFrame(rows)

    def categorise(x):
        if x.correct:
            return "NO_DETECTED_LEXICAL_ATTRACTION"
        if x.pred_equals_top1:
            return "TOP1_ATTRACTION"
        if x.pred_equals_any_topk:
            return "TOPK_ATTRACTION"
        if x.pred_is_training_form:
            return "COMPLETE_TRAINING_WORD_LEXICALIZATION"
        if x.d_pred_topk_min < x.d_pred_target:
            return "PARTIAL_ATTRACTION"
        return "NO_DETECTED_LEXICAL_ATTRACTION"
    d["attraction_category"] = d.apply(categorise, axis=1)
    d.to_csv(os.path.join(out_dir, "lexical_attraction_items.tsv"), sep="\t", index=False)

    # Matched baseline: permute the NEIGHBOUR ASSIGNMENT within
    # (length x target-neighbour-distance x exposure x seed) strata and
    # RECOMPUTE the edit distances.  Permuting precomputed distances would be
    # degenerate - the stratum mean is invariant under permutation - so the
    # neighbour phonologies themselves are reassigned.
    rng = np.random.default_rng(BOOT_SEED)
    err = d[d.correct == 0].copy()
    err["stratum"] = (err.phoneme_length.astype(str) + "|"
                      + err.d_target_top1.astype(str) + "|"
                      + err.exposure_status + "|" + err.seed.astype(str))
    pred_toks = {(r.seed, r.item_id): toks(
        item[(item.seed == r.seed) & (item.item_id == r.item_id)]
        .iloc[0].instrumented_ltm_prediction) for r in err.itertuples()}
    n1_toks = {(s_, i_): toks(top1.loc[(s_, i_), "phonology"])
               for s_, i_ in pred_toks}
    base = []
    for st, g in err.groupby("stratum"):
        if len(g) < 2:
            continue
        keys = [(r.seed, r.item_id) for r in g.itertuples()]
        own = float(np.mean([edit_distance(pred_toks[k], n1_toks[k])
                             for k in keys]))
        perm = []
        for _ in range(200):
            sh = rng.permutation(len(keys))
            perm.append(float(np.mean([
                edit_distance(pred_toks[keys[a]], n1_toks[keys[sh[a]]])
                for a in range(len(keys))])))
        base.append({"stratum": st, "n": len(g),
                     "observed_pred_to_own_top1": own,
                     "permuted_pred_to_other_top1": float(np.mean(perm)),
                     "attraction_advantage": float(np.mean(perm) - own),
                     "note": "positive = prediction is closer to its OWN s_hat "
                             "top-1 neighbour than to a matched other item's"})
    bl = pd.DataFrame(base)
    bl.to_csv(os.path.join(out_dir, "matched_baseline.tsv"), sep="\t", index=False)
    cat = (d.groupby(["seed", "exposure_status", "attraction_category"])
           .size().reset_index(name="n"))
    cat.to_csv(os.path.join(out_dir, "attraction_categories.tsv"), sep="\t", index=False)
    return d, bl, cat


# =============================================================== M5
def m5(item, ts, canon, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    c = canon.copy()
    c["wm_ok"] = (c.wm_exact_match == 1)
    c["ltm_ok"] = (c.ltm_exact_match == 1)
    c["full_ok"] = (c.full_exact_match == 1)

    def cat(r):
        if r.wm_ok and r.ltm_ok:
            return "BOTH_ROUTES_CORRECT"
        if not r.wm_ok and not r.ltm_ok:
            return "BOTH_ROUTES_WRONG"
        if r.wm_ok and not r.ltm_ok:
            return "WM_CORRECT_LTM_WRONG_FULL_" + ("CORRECT" if r.full_ok else "WRONG")
        return "WM_WRONG_LTM_CORRECT_FULL_" + ("CORRECT" if r.full_ok else "WRONG")
    c["route_outcome_category"] = c.apply(cat, axis=1)
    wl = (c.groupby(["seed", "lichtheim_exposure_status", "route_outcome_category"])
          .size().reset_index(name="n"))
    wl.to_csv(os.path.join(out_dir, "word_level_route_outcomes.tsv"),
              sep="\t", index=False)

    # position-level, common FULL-generated prefix
    fp = ts[ts.prefix_source == "full_generated"]
    p = fp.pivot_table(index=["seed", "item_id", "timestep", "target_token"],
                       columns="route",
                       values=["predicted_token", "is_correct", "target_margin"],
                       aggfunc="first").reset_index()
    p.columns = ["_".join([str(a) for a in c_ if str(a)]).strip("_")
                 for c_ in p.columns]
    p = p.merge(item[["seed", "item_id", "exposure_status", "phoneme_length", "gate"]],
                on=["seed", "item_id"])

    def pcat(r):
        lw = r["is_correct_ltm"] == 0
        ww = r["is_correct_wm"] == 0
        fc = r["is_correct_full"] == 1
        if not lw and not ww:
            return "BOTH_LOCAL_CORRECT"
        if lw and not ww:
            return "LTM_LOCAL_WRONG_WM_LOCAL_CORRECT_FULL_" + ("CORRECT" if fc else "WRONG")
        if not lw and ww:
            return "LTM_LOCAL_CORRECT_WM_LOCAL_WRONG_FULL_" + ("CORRECT" if fc else "WRONG")
        return "BOTH_LOCAL_WRONG_FULL_" + ("CORRECT" if fc else "WRONG")
    p["position_rescue_category"] = p.apply(pcat, axis=1)
    p.to_csv(os.path.join(out_dir, "position_level_common_prefix.tsv"),
             sep="\t", index=False)
    pl = (p.groupby(["seed", "exposure_status", "position_rescue_category"])
          .size().reset_index(name="n"))
    pl.to_csv(os.path.join(out_dir, "position_level_rescue_summary.tsv"),
              sep="\t", index=False)
    return wl, p, pl


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instr", default=INSTR)
    args = ap.parse_args(argv)
    item = pd.read_csv(os.path.join(args.instr, "item_summary.tsv"), sep="\t")
    ts = pd.read_csv(os.path.join(args.instr, "timestep_metrics.tsv"), sep="\t")
    neigh = pd.read_csv(os.path.join(args.instr, "lexical_neighbors.tsv"), sep="\t")
    canon = pd.concat([
        pd.read_csv(os.path.join(
            ROOT, f"outputs/behavioral_wfe_fulllexicon_93a577f/full_wfe_evaluation/"
                  f"seed{s}/wfe_ar/item_level_predictions_enriched.tsv"),
            sep="\t").assign(seed=s) for s in SEEDS], ignore_index=True)
    print("M1 …"); ev, bd, hz = m1(item, ts, os.path.join(OUT, "m1_origin_propagation"))
    print("M2 …"); pos, w, sl, bo = m2(item, ts, os.path.join(OUT, "m2_gold_prefix"))
    print("M3 …"); d3, bl, cat = m3(item, neigh, os.path.join(OUT, "m3_lexical_attraction"))
    print("M5 …"); wl, p5, pl = m5(item, ts, canon, os.path.join(OUT, "m5_dorsal_rescue"))
    print(f"done: events={len(ev)} burden={len(bd)} hazard={len(hz)} "
          f"pos={len(pos)} slopes={len(sl)} attraction={len(d3)} "
          f"positions={len(p5)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
