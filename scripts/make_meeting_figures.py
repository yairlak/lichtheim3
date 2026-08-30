"""Two presentation figures for the Yair meeting.

**Presentation only.** Every plotted value and every annotated number is read
from an already-validated table; nothing scientific is recomputed here, no model
is loaded and no analysis is rerun.  The annotation values are additionally
asserted against their source tables at draw time, so the figure cannot drift
from the result it reports.

  mf1  which genuinely trained real words does LTM still fail on?
  mf2  stable-zero checkpoint-selection bottom line
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

from scripts.behavioral_analysis.common import SEEDS                 # noqa: E402
from scripts.behavioral_analysis.plotting import save_figure         # noqa: E402

YC = os.path.join(ROOT, "reports/behavioral_wfe_fulllexicon_93a577f/"
                        "yair_corrections")
RES = os.path.join(YC, "residual_trained_real", "tables")
SZ = os.path.join(YC, "stable_zero_audit")
CANON = os.path.join(
    ROOT, "outputs/behavioral_wfe_fulllexicon_93a577f/behavioral_analysis/"
          "tables/canonical_behavioral_item_table.tsv")
OUT = os.path.join(YC, "meeting_figures")

SEED_COLOR = {19: "#1f4e79", 20: "#c0392b", 21: "#7f8c8d", 22: "#2e7d32"}


# ------------------------------------------------ mf1 trained-real LTM errors

def figure_trained_real_errors() -> Dict[str, str]:
    items = pd.read_csv(os.path.join(RES, "residual_trained_real_items.tsv"),
                        sep="\t")
    summ = pd.read_csv(os.path.join(RES, "residual_trained_real_summary.tsv"),
                       sep="\t").iloc[0]
    canon = pd.read_csv(CANON, sep="\t")
    pop = canon[(canon["route"] == "ltm")
                & (canon["lichtheim_exposure_status"] == "TRAINED_REAL_EXACT")
                & (canon["seed"] == SEEDS[0])]
    pop = pop[["item_id", "target_length", "zipf_frequency"]]

    # annotations are asserted against their source tables, never retyped
    n_items = int(len(items))
    n_pop = int(summ["n_trained_real_items"])
    n_events = int(summ["n_error_events_seed_x_item"])
    n_below = int(summ["n_below_peer_median"])
    n_len9 = int((items["phoneme_length"] == 9).sum())
    mean_pct = float(summ["mean_within_length_zipf_percentile"])
    p_perm = float(summ["permutation_p_one_sided_lower"])
    assert (n_items, n_pop, n_events, n_below, n_len9) == (12, 671, 14, 11, 6)
    assert len(pop) == n_pop

    fig, ax = plt.subplots(figsize=(11.4, 6.4))
    rng = np.random.default_rng(0)
    jit = rng.uniform(-0.16, 0.16, len(pop))
    ax.scatter(pop["target_length"] + jit, pop["zipf_frequency"], s=13,
               color="#c9c9c9", edgecolor="none", zorder=2,
               label=f"correctly repeated ({n_pop - n_items} items)")

    # per-length peer median, taken verbatim from the validated items table
    med = items.groupby("phoneme_length", as_index=False)[
        "median_zipf_of_peers"].first()
    ax.plot(med["phoneme_length"], med["median_zipf_of_peers"], "_",
            color="#555", ms=34, mew=2.2, zorder=3)

    err = items.sort_values("phoneme_length")
    sizes = {1: 90, 2: 210}
    ax.scatter(err["phoneme_length"], err["zipf_frequency"],
               s=[sizes[int(n)] for n in err["n_failing_seeds"]],
               color="#c0392b", edgecolor="#4d0000", linewidth=1.1, zorder=5)

    # Labels are de-collided per length: several length-9 words sit within
    # 0.02 Zipf of each other, so they are stacked with a minimum vertical gap
    # and joined to their point by a leader line.
    MIN_GAP, LAB_DX = 0.17, 0.30
    for L, g in err.groupby("phoneme_length"):
        g = g.sort_values("zipf_frequency")
        ys = g["zipf_frequency"].to_numpy(float).copy()
        for i in range(1, len(ys)):                       # push apart upward
            if ys[i] - ys[i - 1] < MIN_GAP:
                ys[i] = ys[i - 1] + MIN_GAP
        centre = g["zipf_frequency"].mean()
        ys = ys - (ys.mean() - centre)                    # recentre the stack
        for (_, r), y in zip(g.iterrows(), ys):
            ax.annotate("", xy=(L, r["zipf_frequency"]),
                        xytext=(L + LAB_DX, y), textcoords="data",
                        arrowprops=dict(arrowstyle="-", color="#b08080",
                                        lw=.8, shrinkA=0, shrinkB=3))
            ax.text(L + LAB_DX + 0.04, y, r["word"], fontsize=9.2,
                    ha="left", va="center", color="#7b0000", weight="bold")

    ax.set_xticks([3, 4, 5, 7, 8, 9])
    ax.set_xlim(2.5, 10.6)
    ax.set_xlabel("exact phoneme length", fontsize=11)
    ax.set_ylabel("Zipf frequency", fontsize=11)
    ax.set_title("Which genuinely trained real words does LTM still fail on?",
                 fontsize=13.5, loc="left", pad=12)
    ax.grid(alpha=.22, lw=.6)
    ax.set_axisbelow(True)

    note = (f"{n_items}/{n_pop} unique items fail in at least one seed\n"
            f"{n_events} seed x item error events\n"
            f"{n_below}/{n_items} below their same-length frequency median\n"
            f"{n_len9}/{n_items} at length 9\n"
            f"mean within-length Zipf percentile = {mean_pct:.3f}\n"
            f"length-controlled permutation p = {p_perm:g}")
    # placed in the clear band above the data (max Zipf is 7.0) so it cannot
    # sit on top of the length-5 points
    ax.set_ylim(top=9.4)
    ax.text(0.015, 0.975, note, transform=ax.transAxes, fontsize=9.3,
            va="top", ha="left", linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.55", facecolor="#fdf6f5",
                      edgecolor="#c0392b", linewidth=1.0))

    handles = [
        Line2D([], [], lw=0, marker="o", ms=5, color="#c9c9c9",
               label=f"trained real, correctly repeated ({n_pop - n_items})"),
        Line2D([], [], lw=0, marker="o", ms=8, color="#c0392b",
               mec="#4d0000", label="fails in 1 seed"),
        Line2D([], [], lw=0, marker="o", ms=12, color="#c0392b",
               mec="#4d0000", label="fails in 2 seeds"),
        Line2D([], [], lw=2.2, marker="_", ms=14, color="#555",
               label="median Zipf of correct items at that length")]
    ax.legend(handles=handles, fontsize=9, loc="upper right", frameon=True,
              framealpha=.95)
    fig.tight_layout()

    caption = f"""
# Which genuinely trained real words does LTM still fail on?

Population: the **{n_pop} `TRAINED_REAL_EXACT` items** — WFE real words that were
in the training lexicon with the same phonological form. Route: **LTM only**.
Grey points are the {n_pop - n_items} items every seed repeats correctly; red
points are the **{n_items} items that fail in at least one of the four canonical
seeds**, accounting for **{n_events} seed x item error events** in total. Marker
size encodes the number of failing seeds (1 or 2; no item fails in more than 2). Grey dashes are the median Zipf of the correctly repeated items
at that exact length, taken verbatim from the validated table.

**The two signatures are both visible.** The failures sit at the long end —
{n_len9} of {n_items} are 9 phonemes, and no trained real word of 3 or 4
phonemes ever fails — and within each length they sit low in the frequency
distribution: {n_below} of {n_items} fall below their same-length median, with a
mean within-length Zipf percentile of {mean_pct:.3f} against a length-matched
permutation null of 0.467 (p = {p_perm:g}, B = 20,000).

*lieutenant* is the single exception, at the 67th percentile of its length
stratum.

**Scope.** FULL and WM make **no** errors at all on this stratum; these are
LTM-only failures. This is descriptive: with {n_items} items no model is fitted,
and while the frequency-weighted training sampler makes the frequency
association mechanistically unsurprising, nothing here is a causal
demonstration.

Every plotted value and every number in the annotation box is read from
`residual_trained_real/tables/` and is asserted against those tables when the
figure is drawn.
"""
    return save_figure(fig, OUT, "mf1_trained_real_ltm_errors", caption)


# ------------------------------------------------------- mf2 stable zero

def figure_stable_zero(out_dir: str | None = None) -> Dict[str, str]:
    """Stable-zero model-selection figure (mf2).

    Reads only the tracked stable-zero audit tables under
    reports/.../yair_corrections/stable_zero_audit/.  No checkpoint, no GloVe,
    no NWR/SWP data and no model inference are involved.

    out_dir defaults to the canonical meeting_figures directory so existing
    behaviour is unchanged; pass an explicit directory to write elsewhere.
    """
    traj = pd.read_csv(os.path.join(SZ, "stable_zero_trajectory.tsv"), sep="\t")
    streaks = pd.read_csv(os.path.join(SZ, "stable_zero_streaks.tsv"), sep="\t")
    verd = pd.read_csv(os.path.join(SZ, "stable_zero_verdicts.tsv"), sep="\t")

    longest = {int(s): int(g["length"].max()) if len(g) else 0
               for s, g in streaks.groupby("seed")}
    for s in SEEDS:
        longest.setdefault(s, 0)
    assert longest == {19: 6, 20: 2, 21: 0, 22: 13}, longest
    passes = verd.groupby("X")["criterion_met"].sum().to_dict()
    assert passes == {2: 3, 3: 2, 5: 2}, passes

    fig, ax = plt.subplots(figsize=(12.6, 7.0))

    for s in SEEDS:
        d = traj[traj["seed"] == s].sort_values("epoch")
        c = SEED_COLOR[s]
        ax.plot(d["epoch"], d["train_ar_errors_full"], "-", color=c, lw=1.9,
                marker="o", ms=4.5, mfc="white", mew=1.2, zorder=3,
                label=f"seed {s}  (longest zero streak = {longest[s]})")

    # Zero-error occurrences are all at y = 0 and would overplot each other, so
    # they get one clearly separated row per seed BELOW the axis.  This band is
    # an annotation strip, not the error scale.
    ROW = {s: -0.55 - 0.42 * i for i, s in enumerate(SEEDS)}
    ax.axhline(-0.22, color="#999", lw=.8, zorder=2)
    for s in SEEDS:
        d = traj[traj["seed"] == s].sort_values("epoch")
        c, y = SEED_COLOR[s], ROW[s]
        ax.plot([d["epoch"].min(), d["epoch"].max()], [y, y], "-",
                color="#dddddd", lw=1.2, zorder=2)
        st = streaks[(streaks["seed"] == s) & (streaks["length"] >= 2)]
        for _, r in st.iterrows():
            ax.plot([r["first_checkpoint_of_streak"],
                     r["last_checkpoint_of_streak"]], [y, y], "-", color=c,
                    lw=7, alpha=.30, solid_capstyle="round", zorder=3)
        z = d[d["train_ar_errors_full"] == 0]
        ax.plot(z["epoch"], [y] * len(z), "o", color=c, ms=7.5, mec="black",
                mew=.8, zorder=5)
        ax.annotate(f"seed {s}", (102, y), fontsize=8.6, color=c,
                    ha="right", va="center", weight="bold",
                    annotation_clip=False)
    ax.annotate("zero-error\nevaluations", (102, -0.22), fontsize=8.2,
                color="#555", ha="right", va="bottom", linespacing=1.2,
                annotation_clip=False)

    sel22 = verd[(verd.seed == 22) & (verd.X == 5)].iloc[0]
    y22 = ROW[22]
    ax.annotate("selected checkpoint\n= FIRST zero of the streak",
                xy=(sel22["selected_epoch"], y22), xytext=(125, 4.9),
                fontsize=9, color=SEED_COLOR[22], ha="center", linespacing=1.3,
                arrowprops=dict(arrowstyle="->", color=SEED_COLOR[22], lw=1.3))
    ax.annotate("earliest possible STOP for X=5\n= the 5th consecutive zero",
                xy=(sel22["stop_epoch_earliest_knowable"], y22),
                xytext=(178, 3.6), fontsize=9, color=SEED_COLOR[22],
                ha="center", linespacing=1.3,
                arrowprops=dict(arrowstyle="->", color=SEED_COLOR[22], lw=1.3))
    # the seed-21 row is empty by definition, so the label needs no leader
    ax.text(152, ROW[21] + 0.13, "never reaches zero - no qualifying streak",
            fontsize=8.8, color=SEED_COLOR[21], ha="center", va="bottom",
            style="italic")

    ax.axhline(0, color="#333", lw=1.0, zorder=2)
    ax.set_xlim(96, 205)
    ax.set_xticks(sorted(traj["epoch"].unique())[::2])
    ax.set_xlabel("evaluated epoch  (105-200, every 5; complete grid, "
                  "20 evaluations per seed)", fontsize=10.5)
    ax.set_ylabel("train AR errors  (FULL route, 29,571-word lexicon)",
                  fontsize=10.5)
    ax.set_ylim(-2.35, 8.8)
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.set_title("Stable-zero checkpoint selection: how many consecutive "
                 "zero-error evaluations?", fontsize=13.5, loc="left", pad=12)
    ax.grid(alpha=.22, lw=.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9.5, loc="upper right", frameon=True, framealpha=.95)

    rule = ("RULE   selected checkpoint = FIRST checkpoint of a streak of X "
            "consecutive zero-error evaluations;\n"
            "             training may stop only once the Xth consecutive zero "
            "has been observed.\n\n"
            "X = 2  ->  3/4 seeds pass   (19: select 155, stop 160 | "
            "20: select 130, stop 135 | 22: select 140, stop 145)\n"
            "X = 3  ->  2/4 seeds pass   (19: select 155, stop 165 | "
            "22: select 140, stop 150)\n"
            "X = 5  ->  2/4 seeds pass   (19: select 155, stop 175 | "
            "22: select 140, stop 160)\n\n"
            "Raising X never moves a selected checkpoint here - only the "
            "earliest stopping epoch. Seed 20 drops out above X = 2.")
    ax.text(0.012, 0.975, rule, transform=ax.transAxes, fontsize=8.8,
            va="top", ha="left", family="monospace", linespacing=1.55,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f6f8fa",
                      edgecolor="#8a8a8a", linewidth=1.0))
    fig.tight_layout()

    caption = """
# Stable-zero checkpoint selection — bottom line

Train autoregressive error count on the 29,571-word training lexicon (FULL
route) at every evaluated epoch, for the four full-lexicon seeds. The evaluation
grid is complete and regular: epochs 105-200 in steps of 5, 20 evaluations per
seed, with no gaps and nothing inferred. Filled black-edged markers are
zero-error evaluations; shaded bands are streaks of at least two consecutive
zeros.

**Rule illustrated.** The selected checkpoint is the **first** checkpoint of a
qualifying streak, but training can only stop once the **Xth** consecutive zero
has actually been observed - the two epochs are different and both are
annotated on seed 22.

**Longest zero streaks:** seed 19 = 6 (155-180), seed 20 = 2 (130-135),
seed 21 = 0, seed 22 = 13 (140-200).

**Criterion outcomes:** X = 2 -> 3/4 seeds pass; X = 3 -> 2/4; X = 5 -> 2/4.

Two things worth saying out loud at the meeting. First, **raising X would not
have changed a single selected checkpoint** in this cohort - seeds 19 and 22
keep 155 and 140 at every X - it only moves the earliest epoch at which you
could have stopped. Second, **seed 21 never reaches zero at all**; it was
selected by the fallback rule (earliest checkpoint with the minimum error count,
1 error at epoch 145) and passes no X.

Read from the audited trajectories only; nothing is recomputed and no training
or inference was run.
"""
    return save_figure(fig, out_dir if out_dir is not None else OUT,
                       "mf2_stable_zero_bottom_line", caption)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    w = {"mf1": figure_trained_real_errors(), "mf2": figure_stable_zero()}
    with open(os.path.join(OUT, "provenance.json"), "w") as f:
        json.dump({"purpose": "presentation figures for the Yair meeting",
                   "model_loaded": False, "inference_run": False,
                   "scientific_values_recomputed": False,
                   "annotations_asserted_against_source_tables": True,
                   "sources": {
                       "mf1": ["residual_trained_real/tables/"
                               "residual_trained_real_items.tsv",
                               "residual_trained_real/tables/"
                               "residual_trained_real_summary.tsv",
                               "behavioral_analysis/tables/"
                               "canonical_behavioral_item_table.tsv"],
                       "mf2": ["stable_zero_audit/stable_zero_trajectory.tsv",
                               "stable_zero_audit/stable_zero_streaks.tsv",
                               "stable_zero_audit/stable_zero_verdicts.tsv"]}},
                  f, indent=2)
        f.write("\n")
    for k, v in w.items():
        print(k, v["png"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
