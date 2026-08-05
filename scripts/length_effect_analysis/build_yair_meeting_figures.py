"""Meeting-ready presentation figures for the Yair discussion.

Reads only validated tables (see source_selection_notes.md).  No model, no
inference, no M4 path, no analysis rerun.  Writes to a dedicated directory and
overwrites nothing.
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

MECH = os.path.join(ROOT, "outputs/length_effect_mechanism_93a577f")
WFE = os.path.join(ROOT, "reports/behavioral_wfe_fulllexicon_93a577f/figures")
OUT = os.path.join(MECH, "figures/yair_meeting")

TRAINED, NOVEL, UNTRAINED = "TRAINED_REAL_EXACT", "NOVEL_PSEUDOWORD", "UNTRAINED_REAL"
COL = {TRAINED: "#d62728", NOVEL: "#1f77b4", UNTRAINED: "#7a4fa3"}
NAME = {TRAINED: "Trained real words", NOVEL: "Novel pseudowords",
        UNTRAINED: "Untrained real words"}
SHORT = {TRAINED: "Trained\nreal", NOVEL: "Novel\npseudowords",
         UNTRAINED: "Untrained\nreal"}
MK = {19: "o", 20: "s", 21: "^", 22: "D"}
SEEDS = [19, 20, 21, 22]
matplotlib.rcParams.update({"font.size": 12, "axes.titlesize": 13,
                            "axes.labelsize": 12, "svg.hashsalt": "yair-meeting"})


def save(fig, stem, rows, caption):
    os.makedirs(OUT, exist_ok=True)
    for ext, meta in (("png", None), ("pdf", {"CreationDate": None}),
                      ("svg", {"Date": None})):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), dpi=300,
                    bbox_inches="tight", metadata=meta)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{stem}.tsv"), sep="\t",
                              index=False)
    with open(os.path.join(OUT, f"{stem}_caption.md"), "w") as f:
        f.write(caption.strip() + "\n")


def seeds_and_mean(ax, x, vals_by_seed, colour, jitter=0.13, ms=8):
    for i, (s, v) in enumerate(sorted(vals_by_seed.items())):
        ax.plot(x + (i - 1.5) * jitter, v, marker=MK[s], ms=ms, color=colour,
                alpha=0.55, markeredgecolor="white", markeredgewidth=0.8,
                zorder=3)
    m = float(np.mean(list(vals_by_seed.values())))
    ax.hlines(m, x - 0.30, x + 0.30, color="black", lw=3.2, zorder=5)
    return m


# ------------------------------------------------------------------ Figure A
def figure_a():
    m2 = pd.read_csv(os.path.join(MECH, "m2_gold_prefix/length_slopes_ar_vs_gold.tsv"),
                     sep="\t")
    ct = pd.read_csv(os.path.join(WFE, "clean_route_length_contrasts.tsv"), sep="\t")
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4))

    groups = [TRAINED, UNTRAINED, NOVEL]
    for i, g in enumerate(groups):
        d = m2[m2.exposure_status == g]
        vals = dict(zip(d.seed, d.ar_edit_slope))
        m = seeds_and_mean(ax[0], i, vals, COL[g])
        for s, v in vals.items():
            rows.append({"figure": "A", "panel": "A", "exposure_status": g,
                         "seed": s, "quantity": "ltm_ar_edit_length_slope",
                         "value": v, "group_mean": m,
                         "source": "m2_gold_prefix/length_slopes_ar_vs_gold.tsv"})
    ax[0].axhline(0, color="grey", lw=1.2, ls="--")
    ax[0].set_xticks(range(3))
    ax[0].set_xticklabels([SHORT[g] for g in groups])
    ax[0].set_ylabel("LTM length slope\n(edit operations per phoneme)")
    ax[0].set_title("A. Only unfamiliar forms show a length effect", loc="left")
    ax[0].grid(alpha=0.25, axis="y")
    tm = m2[m2.exposure_status == TRAINED].ar_edit_slope.mean()
    nm = m2[m2.exposure_status == NOVEL].ar_edit_slope.mean()
    ax[0].annotate(f"≈{nm:.2f} per phoneme", xy=(2, nm), xytext=(1.55, nm * 0.55),
                   fontsize=11, color=COL[NOVEL], fontweight="bold",
                   arrowprops=dict(arrowstyle="->", color=COL[NOVEL], lw=1.3))
    ax[0].annotate(f"≈{tm:.3f} — essentially flat", xy=(0, tm),
                   xytext=(-0.35, nm * 0.30), fontsize=11, color=COL[TRAINED],
                   fontweight="bold",
                   arrowprops=dict(arrowstyle="->", color=COL[TRAINED], lw=1.3))

    # Panel B — LTM - WM, clean groups only
    lexmap = {TRAINED: "real", NOVEL: "pseudo"}
    for i, g in enumerate(groups):
        if g not in lexmap:
            ax[1].text(i, 0.02, "not available\n(untrained real words are\n"
                                "outside the clean set,\nso no WM slope exists)",
                       ha="center", va="bottom", fontsize=9.5, color="0.35",
                       style="italic")
            rows.append({"figure": "A", "panel": "B", "exposure_status": g,
                         "seed": "", "quantity": "ltm_minus_wm_length_slope",
                         "value": "", "group_mean": "",
                         "source": "UNAVAILABLE - no validated WM slope for "
                                   "UNTRAINED_REAL"})
            continue
        d = ct[ct.source_lexicality == lexmap[g]]
        vals = dict(zip(d.seed, d.ltm_minus_wm))
        m = seeds_and_mean(ax[1], i, vals, COL[g])
        for s, v in vals.items():
            rows.append({"figure": "A", "panel": "B", "exposure_status": g,
                         "seed": s, "quantity": "ltm_minus_wm_length_slope",
                         "value": v, "group_mean": m,
                         "source": "clean_route_length_contrasts.tsv"})
    ax[1].axhline(0, color="grey", lw=1.2, ls="--")
    ax[1].set_xticks(range(3))
    ax[1].set_xticklabels([SHORT[g] for g in groups])
    ax[1].set_ylabel("LTM slope − WM slope")
    ax[1].set_title("B. The ventral route is the one that suffers", loc="left")
    ax[1].grid(alpha=0.25, axis="y")
    ax[1].legend(handles=[Line2D([], [], color="0.35", marker=MK[s], lw=0, ms=7,
                                 label=f"seed {s}") for s in SEEDS]
                 + [Line2D([], [], color="black", lw=3, label="mean")],
                 loc="upper left", frameon=False, fontsize=9, ncol=2)
    fig.suptitle("LTM length sensitivity depends on phonological exposure",
                 fontsize=15.5, y=1.02)
    fig.tight_layout()
    save(fig, "figureA_exposure_length_effect", rows, f"""
**LTM length sensitivity depends on phonological exposure.**

**A** Slope of LTM edit distance on phoneme length, one marker per checkpoint
(seeds 19, 20, 21, 22), black bar the mean. Trained real words are essentially
flat (≈{tm:.3f} operations per phoneme); untrained real words and novel
pseudowords both show a large slope (novel pseudowords ≈{nm:.2f}). The
determining variable is whether the phonological form was seen in training, not
whether the item is a real word.

**B** The same contrast expressed as LTM slope minus WM slope, showing that the
deficit is specific to the ventral route. **The untrained-real column is empty
because the quantity does not exist in any validated table**: untrained real
words are excluded from the clean analysis set by the frozen Sprint-1
definition, so no WM length slope was ever computed for them. It was not
approximated.

**Caveat on interpretation.** The three groups are *not* matched on frequency,
length distribution or phonotactics, and within the clean set lexicality and
exposure coincide exactly by construction. This figure shows that exposure
tracks the effect; it is **not** a controlled separation of lexical status from
exposure.

Sources: `m2_gold_prefix/length_slopes_ar_vs_gold.tsv` (Panel A),
`clean_route_length_contrasts.tsv` (Panel B). Panel-A values are bit-identical
to the archived WFE release.
""")
    return rows


# ------------------------------------------------------------------ Figure B
def figure_b():
    m2 = pd.read_csv(os.path.join(MECH, "m2_gold_prefix/length_slopes_ar_vs_gold.tsv"),
                     sep="\t")
    hz = pd.read_csv(os.path.join(MECH, "m1_origin_propagation/first_error_hazard.tsv"),
                     sep="\t")
    bd = pd.read_csv(os.path.join(MECH, "m1_origin_propagation/post_divergence_burden.tsv"),
                     sep="\t")
    ev = pd.read_csv(os.path.join(MECH, "m1_origin_propagation/first_error_events.tsv"),
                     sep="\t")
    rows = []
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))

    # A: paired AR vs gold-prefix, novel + trained
    for g, off in ((NOVEL, 0.0), (TRAINED, 1.0)):
        d = m2[m2.exposure_status == g].sort_values("seed")
        for r in d.itertuples():
            ax[0].plot([off, off + 0.42], [r.ar_edit_slope, r.gp_edit_slope],
                       color=COL[g], alpha=0.4, lw=1.4, zorder=2)
            ax[0].plot(off, r.ar_edit_slope, marker=MK[r.seed], ms=8,
                       color=COL[g], alpha=0.65, markeredgecolor="white", zorder=3)
            ax[0].plot(off + 0.42, r.gp_edit_slope, marker=MK[r.seed], ms=8,
                       color=COL[g], alpha=0.65, markeredgecolor="white", zorder=3)
            rows.append({"figure": "B", "panel": "A", "exposure_status": g,
                         "seed": r.seed, "ar_edit_slope": r.ar_edit_slope,
                         "gp_edit_slope": r.gp_edit_slope,
                         "source": "length_slopes_ar_vs_gold.tsv"})
        ax[0].hlines(d.ar_edit_slope.mean(), off - 0.16, off + 0.16, color="black",
                     lw=3.2, zorder=5)
        ax[0].hlines(d.gp_edit_slope.mean(), off + 0.26, off + 0.58, color="black",
                     lw=3.2, zorder=5)
    nv = m2[m2.exposure_status == NOVEL]
    ar_m, gp_m = nv.ar_edit_slope.mean(), nv.gp_edit_slope.mean()
    pct = 100 * gp_m / ar_m
    ax[0].annotate(f"{pct:.0f}% of the slope remains\nwith a perfect prefix",
                   xy=(0.42, gp_m), xytext=(0.05, ar_m * 1.05), fontsize=11,
                   color=COL[NOVEL], fontweight="bold",
                   arrowprops=dict(arrowstyle="->", color=COL[NOVEL], lw=1.4))
    ax[0].set_xticks([0, 0.42, 1.0, 1.42])
    ax[0].set_xticklabels(["own\noutput", "perfect\nprefix", "own\noutput",
                           "perfect\nprefix"], fontsize=10)
    ax[0].set_ylabel("LTM length slope")
    ax[0].set_title("A. Half the effect survives a perfect prefix", loc="left")
    ax[0].grid(alpha=0.25, axis="y")
    ax[0].text(0.21, -0.13, "Novel pseudowords", transform=ax[0].get_xaxis_transform(),
               ha="center", color=COL[NOVEL], fontsize=11, fontweight="bold")
    ax[0].text(1.21, -0.13, "Trained real", transform=ax[0].get_xaxis_transform(),
               ha="center", color=COL[TRAINED], fontsize=11, fontweight="bold")

    # B: first-error hazard
    h = hz[(hz.route == "ltm") & (hz.event == "FIRST_TOKEN_MISMATCH")]
    for g in (TRAINED, NOVEL):
        d = h[h.exposure_status == g]
        m = d.groupby("position", as_index=False).agg(
            hazard=("hazard", "mean"), at_risk=("n_at_risk", "mean"))
        strong = m[m.at_risk >= 50]
        weak = m[m.at_risk < 50]
        ax[1].plot(strong.position, strong.hazard, lw=3, marker="o", ms=8,
                   color=COL[g], label=NAME[g])
        if len(weak):
            ax[1].plot(weak.position, weak.hazard, lw=1.6, ls=":", marker="o",
                       ms=6, color=COL[g], alpha=0.45)
        for r in d.itertuples():
            rows.append({"figure": "B", "panel": "B", "exposure_status": g,
                         "seed": r.seed, "position": r.position,
                         "hazard": r.hazard, "n_at_risk": r.n_at_risk,
                         "source": "first_error_hazard.tsv"})
    ax[1].set_xlabel("Phoneme position in the word")
    ax[1].set_ylabel("Probability the first error starts here")
    ax[1].set_title("B. Errors start later and later", loc="left")
    ax[1].grid(alpha=0.25)
    ax[1].legend(frameon=False, fontsize=10, loc="upper left")
    ax[1].text(0.03, 0.72, "dotted = few items still at risk",
               transform=ax[1].transAxes, ha="left", fontsize=9, color="0.4",
               style="italic")

    # C: post-divergence burden
    b = bd[bd.route == "ltm"]
    groups = [TRAINED, UNTRAINED, NOVEL]
    for i, g in enumerate(groups):
        d = b[b.exposure_status == g]
        if d.empty:
            continue
        vals = {s: float(gg.fraction_suffix_wrong.mean())
                for s, gg in d.groupby("seed")}
        m = seeds_and_mean(ax[2], i, vals, COL[g])
        ax[2].text(i, 0.10, f"n={len(d)} erroneous\nitems (4 seeds)", ha="center",
                   fontsize=9, color="0.35")
        for s, v in vals.items():
            rows.append({"figure": "B", "panel": "C", "exposure_status": g,
                         "seed": s, "fraction_suffix_wrong": v, "group_mean": m,
                         "source": "post_divergence_burden.tsv"})
    ax[2].set_xticks(range(3))
    ax[2].set_xticklabels([SHORT[g] for g in groups])
    ax[2].set_ylim(0, 1.05)
    ax[2].set_ylabel("Fraction of the rest of the word wrong")
    ax[2].set_title("C. Once it derails, it stays derailed", loc="left")
    ax[2].grid(alpha=0.25, axis="y")
    nb = b[b.exposure_status == NOVEL].fraction_suffix_wrong.mean()
    e = ev[(ev.route == "ltm") & (ev.exposure_status == NOVEL)
           & ev.FIRST_TOKEN_MISMATCH.notna()]
    n_eos, n_tot = int((e.first_divergence_type == "EOS").sum()), len(e)
    ax[2].annotate(f"≈{nb*100:.0f}% of the remaining\nphonemes are wrong",
                   xy=(2, nb), xytext=(0.75, 0.55), fontsize=11,
                   color=COL[NOVEL], fontweight="bold",
                   arrowprops=dict(arrowstyle="->", color=COL[NOVEL], lw=1.4))
    ax[2].text(0.5, 0.015, f"only {n_eos} of {n_tot} errors begin by stopping early",
               transform=ax[2].transAxes, ha="center", fontsize=10, color="0.25")
    rows.append({"figure": "B", "panel": "C", "exposure_status": NOVEL,
                 "seed": "", "quantity": "eos_first_share",
                 "n_eos_first": n_eos, "n_errors": n_tot,
                 "source": "first_error_events.tsv"})

    fig.suptitle("The LTM length effect has two components: "
                 "local difficulty and autoregressive amplification",
                 fontsize=15.5, y=1.03)
    fig.tight_layout()
    save(fig, "figureB_origin_amplification", rows, f"""
**The LTM length effect has two components: local difficulty and autoregressive
amplification.**

**A** Length slope of LTM edit distance when the route decodes from its **own
output** (left of each pair) and when it is given a **perfect prefix** at every
step (right). One marker per seed, lines pair the two conditions within a seed,
black bars the means. For novel pseudowords the slope falls from
{ar_m:.4f} to {gp_m:.4f} — **{pct:.0f}% of the length effect is still there even
when every preceding phoneme is correct**. Trained real words are near zero in
both conditions.

**This percentage is a decomposition of the measured slope, not a causal split.**
It does not attribute {pct:.0f}% of the effect to the encoder and the remainder
to feedback; the gold-prefix condition is a diagnostic decoding regime, not a
separate mechanism.

**B** Probability that an item's **first** error begins at each phoneme
position, computed over the items still error-free and still having a phoneme at
that position. For novel pseudowords the risk rises steadily through the word;
for trained real words it stays near zero. Dotted segments mark positions where
few items remain at risk. This is an item-level survival curve — **not** the
older interpolated serial-position curve.

**C** Once the first error occurs, the fraction of the remaining phonemes that
are also wrong: ≈{nb*100:.0f}% for novel pseudowords. Only **{n_eos} of {n_tot}**
erroneous items begin by stopping early, so premature stopping is a late symptom
rather than the origin.

Sources: `length_slopes_ar_vs_gold.tsv`, `first_error_hazard.tsv`,
`post_divergence_burden.tsv`, `first_error_events.tsv`.
""")
    return rows


# ------------------------------------------------------------------ Figure C
def figure_c():
    d3 = pd.read_csv(os.path.join(MECH, "m3_lexical_attraction/lexical_attraction_items.tsv"),
                     sep="\t")
    bl = pd.read_csv(os.path.join(MECH, "m3_lexical_attraction/matched_baseline.tsv"),
                     sep="\t")
    wl = pd.read_csv(os.path.join(MECH, "m5_dorsal_rescue/word_level_route_outcomes.tsv"),
                     sep="\t")
    pl = pd.read_csv(os.path.join(MECH, "m5_dorsal_rescue/position_level_rescue_summary.tsv"),
                     sep="\t")
    it = pd.read_csv(os.path.join(MECH, "instrumented/item_summary.tsv"), sep="\t")
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))

    # A: paired target vs neighbour distance, novel pseudoword errors
    e = d3[(d3.correct == 0) & (d3.exposure_status == NOVEL)]
    for s, g in e.groupby("seed"):
        dt, dn = g.d_pred_target.mean(), g.d_pred_top1.mean()
        ax[0].plot([0, 1], [dt, dn], color=COL[NOVEL], alpha=0.4, lw=1.6, zorder=2)
        ax[0].plot(0, dt, marker=MK[s], ms=9, color=COL[NOVEL], alpha=0.7,
                   markeredgecolor="white", zorder=3)
        ax[0].plot(1, dn, marker=MK[s], ms=9, color=COL[NOVEL], alpha=0.7,
                   markeredgecolor="white", zorder=3)
        rows.append({"figure": "C", "panel": "A", "seed": s,
                     "mean_d_pred_target": dt, "mean_d_pred_top1": dn,
                     "n_error_items": len(g),
                     "source": "lexical_attraction_items.tsv"})
    mt, mn = e.d_pred_target.mean(), e.d_pred_top1.mean()
    ax[0].hlines(mt, -0.18, 0.18, color="black", lw=3.4, zorder=5)
    ax[0].hlines(mn, 0.82, 1.18, color="black", lw=3.4, zorder=5)
    lex = 100 * e.pred_is_training_form.mean()
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(["to the\nintended word", "to the closest\nknown word"],
                          fontsize=11)
    ax[0].set_xlim(-0.45, 1.45)
    ax[0].set_ylabel("Edit distance from what LTM produced")
    ax[0].set_title("A. LTM errors stay close to the target", loc="left")
    ax[0].grid(alpha=0.25, axis="y")
    ax[0].annotate(f"≈{mt:.2f}", xy=(0, mt), xytext=(-0.38, mt), fontsize=12,
                   fontweight="bold", color=COL[NOVEL], va="center")
    ax[0].annotate(f"≈{mn:.2f}", xy=(1, mn), xytext=(1.22, mn), fontsize=12,
                   fontweight="bold", color=COL[NOVEL], va="center")
    ax[0].text(0.5, 0.015, "No measurable attraction toward the tested\n"
                          f"s_hat top-k neighbours ({lex:.2f}% become a real word)",
               transform=ax[0].transAxes, ha="center", fontsize=10.5,
               color="0.2", style="italic")

    # B: rescue rates
    w = wl[wl.lichtheim_exposure_status == NOVEL]
    p = pl[pl.exposure_status == NOVEL]
    def rate(df, col, good, bad):
        out = {}
        for s in SEEDS:
            a = df[(df.seed == s) & (df[col] == good)].n.sum()
            b = df[(df.seed == s) & (df[col] == bad)].n.sum()
            out[s] = a / (a + b) if (a + b) else np.nan
        return out
    wr = rate(w, "route_outcome_category", "WM_CORRECT_LTM_WRONG_FULL_CORRECT",
              "WM_CORRECT_LTM_WRONG_FULL_WRONG")
    pr = rate(p, "position_rescue_category",
              "LTM_LOCAL_WRONG_WM_LOCAL_CORRECT_FULL_CORRECT",
              "LTM_LOCAL_WRONG_WM_LOCAL_CORRECT_FULL_WRONG")
    for i, (lbl, vals, src) in enumerate((
            ("Whole words", wr, "word_level_route_outcomes.tsv"),
            ("Individual phonemes\n(same FULL prefix)", pr,
             "position_level_rescue_summary.tsv"))):
        m = seeds_and_mean(ax[1], i, vals, COL[NOVEL])
        ax[1].annotate(f"{m*100:.1f}%", xy=(i, m), xytext=(i, m - 0.055),
                       ha="center", fontsize=14, fontweight="bold",
                       color=COL[NOVEL])
        for s, v in vals.items():
            rows.append({"figure": "C", "panel": "B", "level": lbl.split("\n")[0],
                         "seed": s, "rescue_rate": v, "group_mean": m,
                         "source": src})
    g_novel = it[it.exposure_status == NOVEL].gate.mean()
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels([lbl for lbl in ("Whole words",
                                           "Individual phonemes\n(same FULL prefix)")],
                          fontsize=11)
    ax[1].set_ylim(0.5, 1.02)
    ax[1].set_ylabel("Corrected by FULL when LTM is wrong\nand WM is right")
    ax[1].set_title("B. WM rescues almost every LTM mistake", loc="left")
    ax[1].grid(alpha=0.25, axis="y")
    ax[1].text(0.5, 0.045, f"FULL leans on WM: mean ventral weight g ≈ {g_novel:.3f}, "
                           f"dorsal weight 1−g ≈ {1-g_novel:.3f}\n"
                           f"(one value per word, constant across the word)",
               transform=ax[1].transAxes, ha="center", fontsize=9.5, color="0.2")
    rows.append({"figure": "C", "panel": "B", "level": "gate",
                 "seed": "", "mean_gate_novel_pseudoword": g_novel,
                 "mean_wm_weight": 1 - g_novel,
                 "source": "instrumented/item_summary.tsv"})
    ax[1].legend(handles=[Line2D([], [], color="0.35", marker=MK[s], lw=0, ms=7,
                                 label=f"seed {s}") for s in SEEDS]
                 + [Line2D([], [], color="black", lw=3, label="mean")],
                 loc="center left", frameon=False, fontsize=9, ncol=1)

    fig.suptitle("LTM errors remain target-like, while WM rescues the FULL route",
                 fontsize=15.5, y=1.02)
    fig.tight_layout()
    adv = np.average(bl.attraction_advantage, weights=bl.n)
    save(fig, "figureC_target_like_rescue", rows, f"""
**LTM errors remain target-like, while WM rescues the FULL route.**

**A** For erroneous LTM outputs on novel pseudowords, the mean edit distance
from what the route produced to the **intended target** (≈{mt:.2f}) and to the
**closest known word** identified by the `s_hat` lexical bank (≈{mn:.2f}). One
marker per seed, lines pair the two within a seed. Errors stay near the target
and far from any known word; only {lex:.2f}% of them are exactly a training
form.

**Scope of this claim.** The matched baseline — reassigning each item's
neighbour within strata of length, target-to-neighbour distance, exposure and
seed, then recomputing the distances — gives an attraction advantage of only
**{adv:+.3f} phonemes** on a ≈{mn:.1f}-phoneme scale. This rules out attraction
toward **the tested `s_hat` top-20 neighbours**. It does **not** show that no
lexical or phonotactic influence of any kind exists; a purely phonological
attractor unrelated to that geometry was not tested. Note also that the bank
vector is never fed to the decoder, so any such relationship would be emergent
geometry, not retrieval.

**B** Among cases where WM is right and LTM is wrong, the proportion that FULL
gets right — at whole-word level and at individual phoneme positions.

**On the exact percentage.** The plotted markers are per-seed rates and the bar
is their mean, giving 98.9% (words) and 99.0% (positions). Pooling all seeds
before dividing gives 98.85% and 98.96%; the mechanism interim summary quoted
98.5% and 98.6%, which is the same quantity computed as a ratio of
seed-averaged counts. All three estimators agree to within half a percentage
point; the figure uses the one that matches the points it draws.

The
position-level figure is computed **under the single prefix FULL actually
generated**, so the three routes are directly comparable there; the word-level
figure compares independently generated trajectories and is descriptive
co-occurrence. The gate is a **single value per word**, constant across the
word — it is not adjusted phoneme by phoneme.

Sources: `lexical_attraction_items.tsv`, `matched_baseline.tsv` (caption),
`word_level_route_outcomes.tsv`, `position_level_rescue_summary.tsv`,
`instrumented/item_summary.tsv` (gate).
""")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    a, b, c = figure_a(), figure_b(), figure_c()
    print(f"Figure A rows={len(a)}  Figure B rows={len(b)}  Figure C rows={len(c)}")
    print(f"written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
