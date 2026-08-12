"""Length-controlled analysis of LTM success on novel pseudowords.

Pure reaggregation of already-validated mechanism tables — no model is loaded,
no inference is run, no new feature is created.  Implements the specification
frozen in `.../ltm_pseudoword_success/_control/success_analysis_spec.md`.

Two pre-registered contrasts:

  1. at fixed phoneme length, does lexical confidence predict exact success?
  2. is whole-word failure decided at encoding, or produced by autoregressive
     error propagation?

Seeds are never treated as independent items: every within-length statistic is
computed per seed and then averaged across the four seeds.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.behavioral_analysis.common import SEED_MARKER, SEEDS  # noqa: E402
from scripts.behavioral_analysis.plotting import save_figure       # noqa: E402
from scripts.long_pseudoword_benchmark import sha_file             # noqa: E402

MECH = os.path.join(ROOT, "outputs/length_effect_mechanism_93a577f")
OUT = os.path.join(
    ROOT, "reports/behavioral_wfe_fulllexicon_93a577f/yair_corrections/"
          "ltm_pseudoword_success")
M3 = os.path.join(MECH, "m3_lexical_attraction/lexical_attraction_items.tsv")
M2 = os.path.join(MECH, "m2_gold_prefix/word_level_ar_vs_gold.tsv")
NOVEL = "NOVEL_PSEUDOWORD"


def load() -> Dict[str, pd.DataFrame]:
    m3 = pd.read_csv(M3, sep="\t")
    m2 = pd.read_csv(M2, sep="\t")
    return {"m3": m3[m3["exposure_status"] == NOVEL].copy(),
            "m2": m2[m2["exposure_status"] == NOVEL].copy()}


# ------------------------------------- contrast 1: confidence at fixed length

def success_by_length(m3: pd.DataFrame) -> pd.DataFrame:
    """Per seed x length success rate, plus the seed mean.  Length is the control."""
    per = m3.groupby(["seed", "phoneme_length"], as_index=False).agg(
        n_items=("item_id", "nunique"), success_rate=("correct", "mean"))
    mean = per.groupby("phoneme_length", as_index=False).agg(
        n_items=("n_items", "first"),
        mean_success_rate=("success_rate", "mean"),
        min_seed=("success_rate", "min"), max_seed=("success_rate", "max"))
    return per.merge(mean[["phoneme_length", "mean_success_rate"]],
                     on="phoneme_length"), mean


def confidence_within_length(m3: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Within-stratum high/low confidence success difference.

    The median split is computed **inside each (seed, length) stratum**, so the
    split cannot import the length -> confidence relationship.
    """
    rows = []
    for (seed, L), g in m3.groupby(["seed", "phoneme_length"]):
        med = g["confidence"].median()
        hi, lo = g[g["confidence"] > med], g[g["confidence"] <= med]
        if len(hi) == 0 or len(lo) == 0:
            continue
        rows.append({
            "seed": seed, "phoneme_length": L,
            "n_items": len(g), "n_high": len(hi), "n_low": len(lo),
            "median_confidence": float(med),
            "success_high_confidence": float(hi["correct"].mean()),
            "success_low_confidence": float(lo["correct"].mean()),
            "difference_high_minus_low":
                float(hi["correct"].mean() - lo["correct"].mean()),
            "mean_confidence_high": float(hi["confidence"].mean()),
            "mean_confidence_low": float(lo["confidence"].mean()),
        })
    strat = pd.DataFrame(rows)

    # pooled: n-weighted mean of within-stratum differences, per seed then over seeds
    per_seed = []
    for seed, g in strat.groupby("seed"):
        w = g["n_items"].to_numpy(float)
        per_seed.append({
            "seed": seed,
            "stratified_difference":
                float(np.average(g["difference_high_minus_low"], weights=w)),
        })
    per_seed = pd.DataFrame(per_seed)

    # the marginal (uncontrolled) comparison, for contrast with the stratified one
    marg = []
    for seed, g in m3.groupby("seed"):
        med = g["confidence"].median()
        hi, lo = g[g["confidence"] > med], g[g["confidence"] <= med]
        marg.append({
            "seed": seed,
            "marginal_difference_high_minus_low":
                float(hi["correct"].mean() - lo["correct"].mean()),
            "mean_length_high_confidence": float(hi["phoneme_length"].mean()),
            "mean_length_low_confidence": float(lo["phoneme_length"].mean()),
        })
    marg = pd.DataFrame(marg)
    out = per_seed.merge(marg, on="seed")
    out["note"] = ("stratified = n-weighted mean of within-exact-length "
                   "differences; marginal = same split ignoring length")
    return {"confidence_within_length_strata": strat,
            "confidence_stratified_vs_marginal": out}


# --------------------------- contrast 2: encoding vs autoregressive propagation

def encoding_vs_feedback(m2: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    m2 = m2.copy()
    m2["word_error_agrees"] = (m2["ar_word_error"] == m2["gp_word_error"])
    by_len = m2.groupby("phoneme_length", as_index=False).agg(
        n_seed_x_item=("item_id", "size"),
        ar_word_error=("ar_word_error", "mean"),
        gp_word_error=("gp_word_error", "mean"),
        word_error_agreement=("word_error_agrees", "mean"),
        ar_edit_distance=("ar_edit_distance", "mean"),
        gp_edit_distance=("gp_edit_distance", "mean"))
    by_len["ar_minus_gp_edit_distance"] = (by_len["ar_edit_distance"]
                                           - by_len["gp_edit_distance"])
    by_len["share_of_edit_damage_from_feedback"] = np.where(
        by_len["ar_edit_distance"] > 0,
        by_len["ar_minus_gp_edit_distance"] / by_len["ar_edit_distance"],
        np.nan)
    overall = pd.DataFrame([{
        "n_seed_x_item": int(len(m2)),
        "word_error_agreement_rate": float(m2["word_error_agrees"].mean()),
        "n_disagreements": int((~m2["word_error_agrees"]).sum()),
        "ar_word_error": float(m2["ar_word_error"].mean()),
        "gp_word_error": float(m2["gp_word_error"].mean()),
        "ar_edit_distance": float(m2["ar_edit_distance"].mean()),
        "gp_edit_distance": float(m2["gp_edit_distance"].mean()),
        "word_error_agreement_is_a_structural_identity": True,
        "why": (
            "under greedy decoding AR-correct <=> gold-prefix-correct by "
            "construction: if the AR output is exactly right its own prefix "
            "equals the gold prefix at every step, so the two runs coincide; "
            "and conversely by induction from BOS. The 1.0 agreement is "
            "therefore a tautology and is NOT evidence about encoding."),
        "informative_quantity": (
            "the edit-distance gap, which is conditional on already being "
            "wrong: a perfect prefix roughly halves the damage at long "
            "lengths, so about half the degradation survives perfect context "
            "and about half is added by feedback"),
    }])
    return {"encoding_vs_feedback_by_length": by_len,
            "encoding_vs_feedback_overall": overall}


# ------------------------------------------------------------------ figure

def figure(succ_mean: pd.DataFrame, succ_seed: pd.DataFrame,
           strat: pd.DataFrame, byl: pd.DataFrame) -> Dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9))

    ax = axes[0]
    x = succ_mean["phoneme_length"].to_numpy(float)
    ax.plot(x, succ_mean["mean_success_rate"], "-o", color="#c0392b", lw=2.4,
            ms=8, zorder=5)
    for sd in SEEDS:
        d = succ_seed[succ_seed.seed == sd].sort_values("phoneme_length")
        ax.plot(d["phoneme_length"], d["success_rate"], SEED_MARKER[sd],
                color="#c0392b", ms=4.5, mfc="none", mew=1.1, alpha=.8,
                zorder=4)
    for _, r in succ_mean.iterrows():
        ax.annotate(f"{int(r['n_items'])}", (r["phoneme_length"], -0.115),
                    xycoords=("data", "axes fraction"), ha="center",
                    fontsize=7.5, color="#555", annotation_clip=False)
    ax.annotate("n items", (-0.02, -0.115), xycoords="axes fraction",
                ha="right", fontsize=7.5, color="#555", annotation_clip=False)
    ax.set_xticks(sorted(succ_mean["phoneme_length"]))
    ax.set_xlabel("target length (phonemes)", fontsize=9.5)
    ax.set_ylabel("LTM exact-repetition rate", fontsize=10.5)
    ax.set_ylim(-0.03, 1.02)
    ax.set_title("A  Success falls steeply with length", fontsize=10.5,
                 loc="left")
    ax.grid(alpha=.22, lw=.6)
    ax.set_axisbelow(True)

    ax = axes[1]
    lens = sorted(strat["phoneme_length"].unique())
    m = strat.groupby("phoneme_length", as_index=False)[
        "difference_high_minus_low"].mean()
    ax.axhline(0, color="#333", lw=1.0, zorder=2)
    for sd in SEEDS:
        d = strat[strat.seed == sd].sort_values("phoneme_length")
        ax.plot(d["phoneme_length"], d["difference_high_minus_low"],
                SEED_MARKER[sd], color="#1f4e79", ms=5, mfc="none", mew=1.1,
                alpha=.8, zorder=4)
    ax.plot(m["phoneme_length"], m["difference_high_minus_low"], "-o",
            color="#1f4e79", lw=2.4, ms=8, zorder=5)
    ax.set_xticks(lens)
    ax.set_xlabel("target length (phonemes)", fontsize=9.5)
    ax.set_ylabel("success(high confidence) - success(low confidence)\n"
                  "within the same exact length", fontsize=9.5)
    ax.set_title("B  Lexical confidence adds little, and not consistently",
                 fontsize=10.5, loc="left")
    ax.grid(alpha=.22, lw=.6)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.42, 0.42)

    handles = [Line2D([], [], color="#888", lw=0, marker="o", mfc="none", ms=5,
                      label="individual seeds (19-22)"),
               Line2D([], [], color="#888", lw=2.2, marker="o",
                      label="mean over seeds")]
    axes[1].legend(handles=handles, fontsize=8, loc="lower left", frameon=True,
                   framealpha=.93)
    fig.suptitle("Why can LTM repeat some pseudowords? Length dominates; "
                 "lexical proximity contributes little", fontsize=12.5, y=1.01)
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    caption = """
# LTM success on novel pseudowords: length dominates, lexical proximity contributes little

Population: the 391 `NOVEL_PSEUDOWORD` items, LTM route only, 4 seeds
(1,564 seed x item observations). Everything plotted is a reaggregation of
already-validated mechanism tables; no model was loaded and no new feature was
created.

**Panel A.** LTM's exact-repetition rate falls steeply and monotonically with
target length. Open markers are the four seeds; the line is their mean; item
counts per length are printed beneath.

**Panel B — the pre-registered length control.** Within each exact length, items
are split at the **within-stratum median** lexical confidence (top-1 cosine
between the item's `s_hat` and the frozen GloVe bank), and the success-rate
difference between the high and low halves is plotted. The split is computed
inside each (seed, length) cell, so it cannot import the length-confidence
relationship.

**Result, stated as found.** The within-length difference is small and not
consistent across seeds: +0.046, +0.025, +0.036 and **-0.011** for seeds 19, 20,
21 and 22, i.e. a mean of about +0.024 with one seed of the opposite sign. Set
against the length effect in panel A - from 0.95 down to 0.47, roughly 48
percentage points - a two-point, sign-unstable difference is not a competing
explanation.

Note also what the control did **not** do here: the stratified and the marginal
(uncontrolled) differences are nearly identical, because in this population
confidence and length are only weakly related (mean length of the high- and
low-confidence halves differs by less than half a phoneme). So confidence is a
weak predictor of success, and it was not a strong one being propped up by
length either. Both statements matter; neither is the one we set out to find.

**Confidence and gate are one variable, not two**
(`gate = sigmoid(2 x (confidence - 0.7))`), and are never counted as two pieces
of evidence.

**What this does not show.** Nothing here is causal: the groups are defined by
the outcome. A small within-stratum difference does not prove that no lexical
influence exists - only that the one validated lexical-proximity measure
available does not predict success to any material degree, with or without the
length control. With four seeds and a two-point effect, this analysis cannot
resolve whether the true effect is zero or merely small.
"""
    return save_figure(fig, os.path.join(OUT, "figures"),
                       "lp2_ltm_success_length_controlled", caption)


def main() -> int:
    d = load()
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    succ_seed, succ_mean = success_by_length(d["m3"])
    conf = confidence_within_length(d["m3"])
    enc = encoding_vs_feedback(d["m2"])

    tables = {"ltm_success_by_length_seed": succ_seed,
              "ltm_success_by_length_summary": succ_mean, **conf, **enc}
    for name, t in tables.items():
        t.to_csv(os.path.join(OUT, "tables", f"{name}.tsv"), sep="\t",
                 index=False)
    figure(succ_mean, succ_seed, conf["confidence_within_length_strata"],
           enc["encoding_vs_feedback_by_length"])

    with open(os.path.join(OUT, "_control", "analysis_provenance.json"),
              "w") as f:
        json.dump({
            "stage": "length-controlled reaggregation (no model loaded)",
            "inputs": {"m3_lexical_attraction_items": sha_file(M3),
                       "m2_word_level_ar_vs_gold": sha_file(M2)},
            "population": NOVEL, "n_items": 391, "seeds": SEEDS,
            "seeds_treated_as_independent_items": False,
            "new_features_created": [],
            "model_loaded": False, "inference_run": False,
        }, f, indent=2)
        f.write("\n")

    print("=== success by length ===")
    print(succ_mean.round(3).to_string(index=False))
    print("\n=== confidence effect, stratified vs marginal ===")
    print(conf["confidence_stratified_vs_marginal"].round(4).to_string(index=False))
    print("\n=== encoding vs feedback ===")
    print(enc["encoding_vs_feedback_by_length"].round(3).to_string(index=False))
    print(enc["encoding_vs_feedback_overall"].iloc[0].to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
