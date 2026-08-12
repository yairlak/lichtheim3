"""Successful vs failed unseen pseudowords, at controlled exact length.

Pure reaggregation of already-validated item-level tables.  No model is loaded,
no inference is run, no new measure is defined.

CIRCULARITY GUARD — this is the whole difficulty of the question.
Under deterministic greedy decoding, AR-exact-match and gold-prefix-exact-match
are the same event, so any statistic that is a function of *whether the argmax
was the target* is guaranteed to separate the groups and proves nothing.  That
rules out: overall target rank, minimum margin across positions, count of
positions with rank > 1, first-error position.

Two measures survive the guard and are used here:

  1. **Position-0 gold-prefix target margin and rank.**  At position 0 every
     item is conditioned identically - the decoder prefix is BOS for successes
     and failures alike, and no divergence has yet occurred in either group.
     It reads how strongly the initial decoder state, which is
     `tanh(sem_to_h0(s_hat))`, already supports the first phoneme.

  2. **M4 ordered-probe out-of-fold accuracy from `ltm_encoder_hidden` and raw
     `s_hat`.**  These are independent linear readouts fitted out-of-fold, not
     the model's own decision, so comparing them across groups defined by the
     model's decision is not circular.

Both are compared **within exact phoneme length**, per seed, then averaged over
seeds; seeds are never pooled as independent items.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.behavioral_analysis.common import SEEDS                 # noqa: E402

MECH = os.path.join(ROOT, "outputs/length_effect_mechanism_93a577f")
CANON = os.path.join(
    ROOT, "outputs/behavioral_wfe_fulllexicon_93a577f/behavioral_analysis/"
          "tables/canonical_behavioral_item_table.tsv")
TS = os.path.join(MECH, "instrumented/timestep_metrics.tsv")
OOF = os.path.join(MECH, "m4_representation/ordered_probe_oof_predictions.tsv")
OUT = os.path.join(
    ROOT, "reports/behavioral_wfe_fulllexicon_93a577f/yair_corrections/"
          "ltm_pseudoword_success")
NOVEL = "NOVEL_PSEUDOWORD"

CIRCULAR_EXCLUDED = [
    "overall gold-prefix target rank", "minimum target margin over positions",
    "number of positions with rank > 1", "first-error position",
    "gold-prefix exact match",
]


def success_labels() -> pd.DataFrame:
    c = pd.read_csv(CANON, sep="\t")
    c = c[(c["route"] == "ltm")
          & (c["lichtheim_exposure_status"] == NOVEL)]
    return c[["seed", "item_id", "target_length", "exact_match"]].rename(
        columns={"target_length": "phoneme_length", "exact_match": "success"})


def position0_margin(lab: pd.DataFrame) -> pd.DataFrame:
    t = pd.read_csv(TS, sep="\t")
    t = t[(t["route"] == "ltm") & (t["decode_mode"] == "gold_prefix")
          & (t["timestep"] == 0)]
    t = t[["seed", "item_id", "target_margin", "target_rank", "entropy"]]
    m = lab.merge(t, on=["seed", "item_id"], how="inner")
    assert len(m) == len(lab), (len(m), len(lab))
    return m


def probe_accuracy(lab: pd.DataFrame) -> pd.DataFrame:
    o = pd.read_csv(OOF, sep="\t")
    o = o[(o["variant"] == "primary")
          & (o["exposure_status"] == NOVEL)
          & (o["stage"].isin(["ltm_encoder_hidden", "s_hat"]))]
    acc = o.groupby(["seed", "item_id", "stage"], as_index=False)[
        "correct"].mean()
    wide = acc.pivot_table(index=["seed", "item_id"], columns="stage",
                           values="correct").reset_index()
    wide = wide.rename(columns={"ltm_encoder_hidden": "probe_acc_encoder",
                                "s_hat": "probe_acc_s_hat"})
    return lab.merge(wide, on=["seed", "item_id"], how="left")


def contrast(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Success minus failure within each (seed, exact length)."""
    rows = []
    for (seed, L), g in df.groupby(["seed", "phoneme_length"]):
        s, f = g[g["success"] == 1], g[g["success"] == 0]
        rec = {"seed": seed, "phoneme_length": L,
               "n_success": len(s), "n_failure": len(f)}
        for c in cols:
            rec[f"{c}_success"] = float(s[c].mean()) if len(s) else np.nan
            rec[f"{c}_failure"] = float(f[c].mean()) if len(f) else np.nan
            rec[f"{c}_difference"] = (rec[f"{c}_success"] - rec[f"{c}_failure"]
                                      if len(s) and len(f) else np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def pooled(con: pd.DataFrame, cols) -> pd.DataFrame:
    """n-weighted mean of within-length differences, per seed then over seeds."""
    rows = []
    for seed, g in con.groupby("seed"):
        rec = {"seed": seed}
        for c in cols:
            v = g[f"{c}_difference"]
            w = (g["n_success"] + g["n_failure"]).where(v.notna())
            ok = v.notna()
            rec[f"{c}_stratified_difference"] = (
                float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> int:
    lab = success_labels()
    assert lab["item_id"].nunique() == 391
    p0 = position0_margin(lab)
    pr = probe_accuracy(lab)
    df = p0.merge(pr[["seed", "item_id", "probe_acc_encoder",
                      "probe_acc_s_hat"]], on=["seed", "item_id"])

    cols = ["target_margin", "target_rank", "probe_acc_encoder",
            "probe_acc_s_hat"]
    con = contrast(df, cols)
    pool = pooled(con, cols)
    pool["note"] = ("stratified = n-weighted mean of within-exact-length "
                    "success-minus-failure differences")

    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    con.to_csv(os.path.join(OUT, "tables",
                            "success_vs_failure_within_length.tsv"), sep="\t",
               index=False)
    pool.to_csv(os.path.join(OUT, "tables",
                             "success_vs_failure_stratified.tsv"), sep="\t",
                index=False)
    with open(os.path.join(OUT, "_control",
                           "information_analysis_provenance.json"), "w") as f:
        json.dump({
            "stage": "success-vs-failure reaggregation (no model loaded)",
            "classification": {
                "gold_prefix_target_margin_rank": "B - reaggregation only",
                "m4_ordered_probe_item_accuracy": "B - reaggregation only",
                "genuinely_missing": []},
            "circularity_guard_excluded_measures": CIRCULAR_EXCLUDED,
            "why": ("under deterministic greedy decoding AR-exact-match and "
                    "gold-prefix-exact-match are the same event, so any "
                    "statistic that is a function of whether the argmax was "
                    "the target separates the groups by construction"),
            "measures_used": [
                "position-0 gold-prefix target margin and rank (identical "
                "conditioning for both groups: prefix is BOS for everyone)",
                "M4 out-of-fold ordered-probe accuracy from ltm_encoder_hidden "
                "and raw s_hat (independent linear readouts, not the model's "
                "decision)"],
            "length_control": "exact phoneme length stratification",
            "seeds_pooled_as_items": False,
            "model_loaded": False, "inference_run": False,
        }, f, indent=2)
        f.write("\n")

    show = ["phoneme_length", "n_success", "n_failure",
            "target_margin_success", "target_margin_failure",
            "target_margin_difference",
            "probe_acc_encoder_difference", "probe_acc_s_hat_difference"]
    print("=== within-length contrasts, mean over seeds ===")
    print(con.groupby("phoneme_length", as_index=False)[
        [c for c in show if c != "phoneme_length"]].mean().round(4).to_string(
        index=False))
    print("\n=== stratified difference, per seed ===")
    print(pool.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
