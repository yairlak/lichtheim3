"""The two primary mechanism figures, each backed by its own TSV.

Conventions: red = real, blue = pseudoword; all four seed points visible;
routes separated by facet or explicit encoding; no hardcoded scientific value -
every number is read from the analysis TSVs.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "outputs/length_effect_mechanism_93a577f")

COL = {"TRAINED_REAL_EXACT": "red", "NOVEL_PSEUDOWORD": "blue",
       "UNTRAINED_REAL": "#7a4fa3"}
LAB = {"TRAINED_REAL_EXACT": "Trained real", "NOVEL_PSEUDOWORD": "Novel pseudoword",
       "UNTRAINED_REAL": "Untrained real (extension)"}
MK = {19: "o", 20: "s", 21: "^", 22: "D"}
GROUPS = ["TRAINED_REAL_EXACT", "NOVEL_PSEUDOWORD", "UNTRAINED_REAL"]
matplotlib.rcParams["svg.hashsalt"] = "lichtheim3-mechanism"


def save(fig, stem, out_dir, caption):
    os.makedirs(out_dir, exist_ok=True)
    for ext, meta in (("png", None), ("pdf", {"CreationDate": None}),
                      ("svg", {"Date": None})):
        fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"), dpi=300,
                    bbox_inches="tight", metadata=meta)
    plt.close(fig)
    with open(os.path.join(out_dir, f"{stem}_caption.md"), "w") as f:
        f.write(caption.strip() + "\n")


def figure1(out_dir):
    hz = pd.read_csv(os.path.join(OUT, "m1_origin_propagation/first_error_hazard.tsv"), sep="\t")
    pos = pd.read_csv(os.path.join(OUT, "m2_gold_prefix/position_profiles.tsv"), sep="\t")
    bd = pd.read_csv(os.path.join(OUT, "m1_origin_propagation/post_divergence_burden.tsv"), sep="\t")
    ev = pd.read_csv(os.path.join(OUT, "m1_origin_propagation/first_error_events.tsv"), sep="\t")

    fig, ax = plt.subplots(1, 4, figsize=(17.5, 4.3))
    plot = []

    # A: first-error hazard (LTM, FIRST_TOKEN_MISMATCH)
    h = hz[(hz.route == "ltm") & (hz.event == "FIRST_TOKEN_MISMATCH")
           & hz.exposure_status.isin(GROUPS)]
    for gname, g in h.groupby("exposure_status"):
        for s, gs in g.groupby("seed"):
            gs = gs.sort_values("position")
            ax[0].plot(gs.position, gs.hazard, ls=":", lw=0.7, alpha=0.45,
                       marker=MK[s], ms=3, color=COL[gname])
        m = g.groupby("position", as_index=False).hazard.mean()
        ax[0].plot(m.position, m.hazard, lw=2.6, marker="o", ms=6, color=COL[gname])
        for r in g.itertuples():
            plot.append({"panel": "A_hazard", "exposure_status": gname,
                         "seed": r.seed, "position": r.position,
                         "value": r.hazard, "n_at_risk": r.n_at_risk})
    ax[0].set_xlabel("Target position (0-based)")
    ax[0].set_ylabel("First-error hazard")
    ax[0].set_title("A. Origin: first-error hazard (LTM-AR)", fontsize=10)
    ax[0].grid(alpha=0.25)
    ax[0].legend(handles=[Line2D([], [], color=COL[k], lw=2.6, label=LAB[k])
                          for k in GROUPS], loc="upper left", frameon=False, fontsize=7.5)

    # B: LTM AR vs gold-prefix local accuracy by position
    p = pos[pos.exposure_status.isin(GROUPS)]
    for gname, g in p.groupby("exposure_status"):
        for stream, ls in (("autoregressive", "-"), ("gold_prefix", "--")):
            gg = g[g.stream == stream].groupby("absolute_position", as_index=False).top1_rate.mean()
            ax[1].plot(gg.absolute_position, gg.top1_rate, ls=ls, lw=2.4,
                       marker="o" if stream == "autoregressive" else "s",
                       ms=5, color=COL[gname])
            for r in g[g.stream == stream].itertuples():
                plot.append({"panel": "B_local_accuracy", "exposure_status": gname,
                             "seed": r.seed, "position": r.absolute_position,
                             "stream": stream, "value": r.top1_rate})
    ax[1].set_xlabel("Absolute position")
    ax[1].set_ylabel("Local target-is-top1 rate")
    ax[1].set_title("B. AR (solid) vs gold prefix (dashed), LTM", fontsize=10)
    ax[1].grid(alpha=0.25)

    # C: post-divergence suffix burden
    b = bd[(bd.route == "ltm") & bd.exposure_status.isin(GROUPS)]
    for i, gname in enumerate(GROUPS):
        g = b[b.exposure_status == gname]
        if g.empty:
            continue
        for s, gs in g.groupby("seed"):
            ax[2].plot(i + (list(MK).index(s) - 1.5) * 0.15,
                       gs.fraction_suffix_wrong.mean(), marker=MK[s], ms=7,
                       color=COL[gname], alpha=0.85)
            plot.append({"panel": "C_suffix_burden", "exposure_status": gname,
                         "seed": s, "value": float(gs.fraction_suffix_wrong.mean()),
                         "n_items": len(gs)})
        ax[2].hlines(g.fraction_suffix_wrong.mean(), i - 0.3, i + 0.3,
                     color="black", lw=2.2, zorder=4)
    ax[2].set_xticks(range(len(GROUPS)))
    ax[2].set_xticklabels([LAB[k].replace(" (extension)", "") for k in GROUPS],
                          fontsize=7.5, rotation=12)
    ax[2].set_ylabel("Fraction of remaining suffix wrong")
    ax[2].set_title("C. Propagation: post-divergence burden", fontsize=10)
    ax[2].grid(alpha=0.25, axis="y")

    # D: EOS-first vs non-EOS-first contribution
    e = ev[(ev.route == "ltm") & ev.exposure_status.isin(GROUPS)
           & ev.FIRST_TOKEN_MISMATCH.notna()]
    for i, gname in enumerate(GROUPS):
        g = e[e.exposure_status == gname]
        if g.empty:
            continue
        for s, gs in g.groupby("seed"):
            n_eos = int((gs.first_divergence_type == "EOS").sum())
            n_tot = len(gs)
            ax[3].plot(i + (list(MK).index(s) - 1.5) * 0.15, n_eos / n_tot,
                       marker=MK[s], ms=7, color=COL[gname], alpha=0.85)
            plot.append({"panel": "D_eos_first_share", "exposure_status": gname,
                         "seed": s, "value": n_eos / n_tot, "n_errors": n_tot,
                         "n_eos_first": n_eos})
        frac = (g.first_divergence_type == "EOS").mean()
        ax[3].hlines(frac, i - 0.3, i + 0.3, color="black", lw=2.2, zorder=4)
    ax[3].set_xticks(range(len(GROUPS)))
    ax[3].set_xticklabels([LAB[k].replace(" (extension)", "") for k in GROUPS],
                          fontsize=7.5, rotation=12)
    ax[3].set_ylabel("Share of errors whose FIRST divergence is EOS")
    ax[3].set_title("D. EOS-first vs non-EOS-first", fontsize=10)
    ax[3].grid(alpha=0.25, axis="y")
    ax[3].legend(handles=[Line2D([], [], color="0.3", marker=MK[s], lw=0, ms=6,
                                 label=f"seed {s}") for s in MK],
                 loc="upper right", frameon=False, fontsize=7, ncol=2)

    fig.suptitle("Figure 1 — Origin and propagation of the LTM length effect",
                 y=1.03, fontsize=13)
    fig.tight_layout()
    df = pd.DataFrame(plot)
    df.to_csv(os.path.join(out_dir, "figure1_origin_propagation.tsv"),
              sep="\t", index=False)
    save(fig, "figure1_origin_propagation", out_dir, """
**Figure 1 — Origin and propagation of the LTM length effect.**
**A** First-error hazard for the LTM autoregressive stream: the number of items
whose first token mismatch occurs at position *t*, divided by the number of
items still event-free before *t* **and** possessing a target token at *t*. Thin
dotted lines are the four seeds; thick lines the mean. This denominator is built
from item-level first-event records; it is **not** derived from the PCHIP
serial-position curve of the closed behavioural analysis, which is a pooled
zip-mismatch interpolation.
**B** Local target-is-top-1 rate by absolute position for the LTM route under
its own autoregressive prefix (solid) and under the gold prefix (dashed). The
gap between the two curves is the part of the deficit that requires feeding back
a generated token; the level of the dashed curve is the part already present
without any feedback.
**C** Post-divergence burden: fraction of the remaining suffix positions that
are wrong once the first divergence has occurred, one marker per seed, black bar
the mean.
**D** Share of erroneous items whose **first** divergence is an EOS emission
rather than a phoneme substitution.
Colours: red = trained real, blue = novel pseudoword, purple = untrained real
(exposure extension). All four seeds are shown throughout; seed 21 is never
excluded. Every panel is backed by `figure1_origin_propagation.tsv`.
""")
    return df


def figure2(out_dir):
    d3 = pd.read_csv(os.path.join(OUT, "m3_lexical_attraction/lexical_attraction_items.tsv"), sep="\t")
    bl = pd.read_csv(os.path.join(OUT, "m3_lexical_attraction/matched_baseline.tsv"), sep="\t")
    cat = pd.read_csv(os.path.join(OUT, "m3_lexical_attraction/attraction_categories.tsv"), sep="\t")
    wl = pd.read_csv(os.path.join(OUT, "m5_dorsal_rescue/word_level_route_outcomes.tsv"), sep="\t")
    pl = pd.read_csv(os.path.join(OUT, "m5_dorsal_rescue/position_level_rescue_summary.tsv"), sep="\t")

    fig, ax = plt.subplots(1, 4, figsize=(18, 4.5))
    plot = []

    # A: prediction-to-target vs prediction-to-neighbour
    e = d3[(d3.correct == 0) & d3.exposure_status.isin(GROUPS)]
    for gname, g in e.groupby("exposure_status"):
        ax[0].scatter(g.d_pred_target, g.d_pred_top1, s=9, alpha=0.35,
                      color=COL[gname], label=LAB[gname])
        for r in g.itertuples():
            plot.append({"panel": "A_distances", "exposure_status": gname,
                         "seed": r.seed, "item_id": r.item_id,
                         "d_pred_target": r.d_pred_target,
                         "d_pred_top1": r.d_pred_top1,
                         "d_pred_topk_min": r.d_pred_topk_min})
    lim = [0, max(e.d_pred_top1.max(), e.d_pred_target.max()) + 1]
    ax[0].plot(lim, lim, ls="--", color="grey", lw=1)
    ax[0].set_xlabel("Edit distance: prediction → target")
    ax[0].set_ylabel("Edit distance: prediction → s_hat top-1 neighbour")
    ax[0].set_title("A. Errors are near the target, not the neighbour", fontsize=10)
    ax[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    ax[0].grid(alpha=0.25)

    # B: matched-baseline attraction advantage
    w = np.average(bl.attraction_advantage, weights=bl.n)
    ax[1].bar([0, 1], [np.average(bl.observed_pred_to_own_top1, weights=bl.n),
                       np.average(bl.permuted_pred_to_other_top1, weights=bl.n)],
              color=["#4d4d4d", "#b0b0b0"], width=0.55, edgecolor="0.2")
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(["own s_hat\ntop-1", "matched other\nitem's top-1"], fontsize=8)
    ax[1].set_ylabel("Mean edit distance from prediction")
    ax[1].set_title(f"B. Matched baseline\nattraction advantage = {w:+.3f} phonemes",
                    fontsize=10)
    ax[1].grid(alpha=0.25, axis="y")
    plot.append({"panel": "B_matched_baseline", "observed_own_top1":
                 float(np.average(bl.observed_pred_to_own_top1, weights=bl.n)),
                 "permuted_other_top1":
                 float(np.average(bl.permuted_pred_to_other_top1, weights=bl.n)),
                 "attraction_advantage": float(w), "n_strata": len(bl),
                 "n_items": int(bl.n.sum())})

    # C: word-level route outcomes (novel pseudowords)
    key = ["WM_CORRECT_LTM_WRONG_FULL_CORRECT", "WM_CORRECT_LTM_WRONG_FULL_WRONG",
           "WM_WRONG_LTM_CORRECT_FULL_CORRECT", "WM_WRONG_LTM_CORRECT_FULL_WRONG",
           "BOTH_ROUTES_WRONG"]
    ww = wl[wl.lichtheim_exposure_status == "NOVEL_PSEUDOWORD"]
    for i, k in enumerate(key):
        g = ww[ww.route_outcome_category == k]
        for r in g.itertuples():
            ax[2].plot(i + (list(MK).index(r.seed) - 1.5) * 0.15, r.n,
                       marker=MK[r.seed], ms=7, color="blue", alpha=0.85)
            plot.append({"panel": "C_word_rescue", "category": k,
                         "seed": r.seed, "n": r.n})
        if len(g):
            ax[2].hlines(g.n.mean(), i - 0.3, i + 0.3, color="black", lw=2.2, zorder=4)
    ax[2].set_xticks(range(len(key)))
    ax[2].set_xticklabels([k.replace("_", "\n", 2).replace("_", " ") for k in key],
                          fontsize=6)
    ax[2].set_ylabel("Items per seed")
    ax[2].set_title("C. Word-level route outcomes\n(novel pseudowords)", fontsize=10)
    ax[2].grid(alpha=0.25, axis="y")

    # D: position-level rescue under the common FULL prefix
    keyp = ["LTM_LOCAL_WRONG_WM_LOCAL_CORRECT_FULL_CORRECT",
            "LTM_LOCAL_WRONG_WM_LOCAL_CORRECT_FULL_WRONG",
            "LTM_LOCAL_CORRECT_WM_LOCAL_WRONG_FULL_CORRECT",
            "BOTH_LOCAL_WRONG_FULL_CORRECT"]
    pp = pl[pl.exposure_status == "NOVEL_PSEUDOWORD"]
    for i, k in enumerate(keyp):
        g = pp[pp.position_rescue_category == k]
        for r in g.itertuples():
            ax[3].plot(i + (list(MK).index(r.seed) - 1.5) * 0.15, r.n,
                       marker=MK[r.seed], ms=7, color="blue", alpha=0.85)
            plot.append({"panel": "D_position_rescue", "category": k,
                         "seed": r.seed, "n": r.n})
        if len(g):
            ax[3].hlines(g.n.mean(), i - 0.3, i + 0.3, color="black", lw=2.2, zorder=4)
    ax[3].set_xticks(range(len(keyp)))
    ax[3].set_xticklabels(["LTM✗ WM✓\nFULL✓", "LTM✗ WM✓\nFULL✗",
                           "LTM✓ WM✗\nFULL✓", "both✗\nFULL✓"], fontsize=7)
    ax[3].set_ylabel("Positions per seed")
    ax[3].set_title("D. Position-level rescue\n(common FULL prefix)", fontsize=10)
    ax[3].grid(alpha=0.25, axis="y")
    ax[3].legend(handles=[Line2D([], [], color="0.3", marker=MK[s], lw=0, ms=6,
                                 label=f"seed {s}") for s in MK],
                 loc="upper right", frameon=False, fontsize=7, ncol=2)

    fig.suptitle("Figure 2 — Lexical attraction and dorsal rescue", y=1.03, fontsize=13)
    fig.tight_layout()
    df = pd.DataFrame(plot)
    df.to_csv(os.path.join(out_dir, "figure2_attraction_rescue.tsv"),
              sep="\t", index=False)
    save(fig, "figure2_attraction_rescue", out_dir, """
**Figure 2 — Lexical attraction and dorsal rescue.**
**A** For every erroneous LTM item, the edit distance from the prediction to the
target against the distance from the prediction to the top-1 neighbour of `s_hat`
in the frozen bank. Points above the diagonal are predictions closer to the
target than to the neighbour.
**B** Matched baseline: the mean distance from each prediction to **its own**
`s_hat` top-1 neighbour, against the distance to the top-1 neighbour of a
**matched other item** drawn from the same length × target-neighbour-distance ×
exposure × seed stratum, with the neighbour assignment permuted and the edit
distances recomputed. A positive advantage means genuine attraction.
**C** Word-level route-outcome categories for novel pseudowords, one marker per
seed. These are **behavioural co-occurrence categories** computed from
independently generated route trajectories; they do **not** by themselves
demonstrate a causal contribution.
**D** Position-level rescue computed **under the single prefix actually
generated by FULL**, so the WM, LTM and FULL quantities at each position are
directly comparable. This is the panel that supports a mechanistic reading.
Colours: red = real, blue = pseudoword. All four seeds visible; seed 21 never
excluded. No true fixed-g autoregressive curve is shown — changing g changes the
prefix, so such a curve would require new trajectories and is out of scope here.
Every panel is backed by `figure2_attraction_rescue.tsv`.
""")
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(OUT, "figures"))
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    f1 = figure1(args.out_dir)
    f2 = figure2(args.out_dir)
    print(f"figure1 rows={len(f1)}  figure2 rows={len(f2)}")
    print(f"written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
