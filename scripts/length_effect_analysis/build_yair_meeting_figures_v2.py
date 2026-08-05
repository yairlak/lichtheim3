"""Yair meeting figures V2 — label, annotation and layout corrections only.

Reads the SAME validated source tables and applies the SAME estimators as V1.
No scientific value is recomputed from raw predictions; the only differences
from V1 are titles, axis labels, annotations, layout, captions, and the removal
of the empty "untrained real" category from Figure A panel B.

Writes to figures/yair_meeting/v2/ and overwrites nothing under V1.
"""
from __future__ import annotations

import argparse
import hashlib
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
V1 = os.path.join(MECH, "figures/yair_meeting")
OUT = os.path.join(V1, "v2")

# Exact source paths, resolved and recorded (identical to V1).
SRC = {
    "slopes": os.path.join(MECH, "m2_gold_prefix/length_slopes_ar_vs_gold.tsv"),
    "contrast": os.path.join(WFE, "clean_route_length_contrasts.tsv"),
    "hazard": os.path.join(MECH, "m1_origin_propagation/first_error_hazard.tsv"),
    "burden": os.path.join(MECH, "m1_origin_propagation/post_divergence_burden.tsv"),
    "events": os.path.join(MECH, "m1_origin_propagation/first_error_events.tsv"),
    "attraction": os.path.join(MECH, "m3_lexical_attraction/lexical_attraction_items.tsv"),
    "baseline": os.path.join(MECH, "m3_lexical_attraction/matched_baseline.tsv"),
    "word_rescue": os.path.join(MECH, "m5_dorsal_rescue/word_level_route_outcomes.tsv"),
    "pos_rescue": os.path.join(MECH, "m5_dorsal_rescue/position_level_rescue_summary.tsv"),
    "items": os.path.join(MECH, "instrumented/item_summary.tsv"),
}

TRAINED, NOVEL, UNTRAINED = "TRAINED_REAL_EXACT", "NOVEL_PSEUDOWORD", "UNTRAINED_REAL"
COL = {TRAINED: "#d62728", NOVEL: "#1f77b4", UNTRAINED: "#7a4fa3"}
SHORT = {TRAINED: "Trained\nreal", NOVEL: "Novel\npseudowords",
         UNTRAINED: "Untrained\nreal"}
NAME = {TRAINED: "Trained real words", NOVEL: "Novel pseudowords",
        UNTRAINED: "Untrained real words"}
MK = {19: "o", 20: "s", 21: "^", 22: "D"}
SEEDS = [19, 20, 21, 22]

# The V1 dotted-line rule, recovered from build_yair_meeting_figures.py:216-217.
RISK_THRESHOLD = 50

matplotlib.rcParams.update({"font.size": 12.5, "axes.titlesize": 13.5,
                            "axes.labelsize": 12.5, "svg.hashsalt": "yair-meeting-v2"})


def rel(p):
    return os.path.relpath(p, ROOT)


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


def seeds_and_mean(ax, x, vals, colour, jitter=0.13, ms=8):
    for i, (s, v) in enumerate(sorted(vals.items())):
        ax.plot(x + (i - 1.5) * jitter, v, marker=MK[s], ms=ms, color=colour,
                alpha=0.55, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    m = float(np.mean(list(vals.values())))
    ax.hlines(m, x - 0.30, x + 0.30, color="black", lw=3.2, zorder=5)
    return m


def seed_legend(ax, loc, ncol=2):
    return ax.legend(handles=[Line2D([], [], color="0.35", marker=MK[s], lw=0,
                                     ms=7, label=f"seed {s}") for s in SEEDS]
                     + [Line2D([], [], color="black", lw=3, label="mean")],
                     loc=loc, frameon=False, fontsize=9, ncol=ncol)


# ------------------------------------------------------------------ Figure A
def figure_a():
    m2 = pd.read_csv(SRC["slopes"], sep="\t")
    ct = pd.read_csv(SRC["contrast"], sep="\t")
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4))

    groups = [TRAINED, UNTRAINED, NOVEL]
    for i, g in enumerate(groups):
        d = m2[m2.exposure_status == g]
        vals = dict(zip(d.seed, d.ar_edit_slope))
        m = seeds_and_mean(ax[0], i, vals, COL[g])
        for s, v in vals.items():
            rows.append({"figure": "A", "panel": "A", "group": g, "seed": s,
                         "metric": "ltm_ar_edit_length_slope", "value": v,
                         "group_mean": m, "source": rel(SRC["slopes"])})
    ax[0].axhline(0, color="grey", lw=1.2, ls="--")
    ax[0].set_xticks(range(3))
    ax[0].set_xticklabels([SHORT[g] for g in groups])
    ax[0].set_ylabel("LTM length slope\n(edit operations per phoneme)")
    ax[0].set_title("A. A strong length effect appears only for unfamiliar forms",
                    loc="left", fontsize=12.5)
    ax[0].grid(alpha=0.25, axis="y")
    tm = m2[m2.exposure_status == TRAINED].ar_edit_slope.mean()
    nm = m2[m2.exposure_status == NOVEL].ar_edit_slope.mean()
    um = m2[m2.exposure_status == UNTRAINED].ar_edit_slope.mean()
    ax[0].annotate(f"≈{nm:.2f}", xy=(2, nm), xytext=(1.62, nm * 0.62),
                   fontsize=12, color=COL[NOVEL], fontweight="bold",
                   arrowprops=dict(arrowstyle="->", color=COL[NOVEL], lw=1.3))
    ax[0].annotate(f"≈{tm:.3f}", xy=(0, tm), xytext=(-0.30, nm * 0.32),
                   fontsize=12, color=COL[TRAINED], fontweight="bold",
                   arrowprops=dict(arrowstyle="->", color=COL[TRAINED], lw=1.3))

    # Panel B — only the two groups for which the contrast exists
    lexmap = {TRAINED: "real", NOVEL: "pseudo"}
    gb = [TRAINED, NOVEL]
    for i, g in enumerate(gb):
        d = ct[ct.source_lexicality == lexmap[g]]
        vals = dict(zip(d.seed, d.ltm_minus_wm))
        m = seeds_and_mean(ax[1], i, vals, COL[g])
        for s, v in vals.items():
            rows.append({"figure": "A", "panel": "B", "group": g, "seed": s,
                         "metric": "ltm_minus_wm_length_slope", "value": v,
                         "group_mean": m, "source": rel(SRC["contrast"])})
    ax[1].axhline(0, color="grey", lw=1.2, ls="--")
    ax[1].set_xticks(range(2))
    ax[1].set_xticklabels([SHORT[g] for g in gb])
    ax[1].set_xlim(-0.6, 1.6)
    ax[1].set_ylabel("LTM slope − WM slope")
    ax[1].set_title("B. The unfamiliar-form deficit is concentrated in LTM",
                    loc="left", fontsize=12.5)
    ax[1].grid(alpha=0.25, axis="y")
    seed_legend(ax[1], "upper left")

    fig.suptitle("LTM length sensitivity tracks phonological exposure",
                 fontsize=16, y=1.02)
    fig.tight_layout()
    save(fig, "figureA_exposure_length_effect_v2", rows, f"""
**LTM length sensitivity tracks phonological exposure.**

**Estimator shown:** per-seed OLS slope of LTM edit distance on continuous
phoneme length; markers are the four checkpoints, the black bar is the
unweighted mean over seeds.

**A — A strong length effect appears only for unfamiliar forms.** Trained real
words are essentially flat (≈{tm:.3f} edit operations per phoneme). Untrained
real words (≈{um:.2f}) and novel pseudowords (≈{nm:.2f}) both show a large
slope. Whether the item is a real word does not determine the outcome; whether
its phonological form was seen during training does.

**B — The unfamiliar-form deficit is concentrated in LTM.** The same comparison
expressed as LTM slope minus WM slope. **The LTM−WM contrast was not estimated
for untrained real words in the frozen analysis because that exposure group lies
outside `LICHTHEIM_CLEAN`.** That category is therefore absent from panel B; it
was not newly calculated for this figure.

**Interpretation limits.** Untrained real words are **not matched** to novel
pseudowords on frequency, phonotactics or length distribution — the untrained
group is defined by absence from the 29,571-word training lexicon, which
correlates with low frequency. Within `LICHTHEIM_CLEAN`, lexicality and exposure
coincide exactly by construction. **Exposure tracks this result; it has not been
causally isolated.**

Sources: `{rel(SRC['slopes'])}` (panel A); `{rel(SRC['contrast'])}` (panel B).
Panel-A values are bit-identical to the archived WFE release.
""")
    return rows


# ------------------------------------------------------------------ Figure B
def figure_b():
    m2 = pd.read_csv(SRC["slopes"], sep="\t")
    hz = pd.read_csv(SRC["hazard"], sep="\t")
    bd = pd.read_csv(SRC["burden"], sep="\t")
    ev = pd.read_csv(SRC["events"], sep="\t")
    rows = []
    fig, ax = plt.subplots(1, 3, figsize=(18.5, 5.6))

    # A: AR vs gold prefix
    for g, off in ((NOVEL, 0.0), (TRAINED, 1.15)):
        d = m2[m2.exposure_status == g].sort_values("seed")
        for r in d.itertuples():
            ax[0].plot([off, off + 0.45], [r.ar_edit_slope, r.gp_edit_slope],
                       color=COL[g], alpha=0.4, lw=1.5, zorder=2)
            for xx, vv in ((off, r.ar_edit_slope), (off + 0.45, r.gp_edit_slope)):
                ax[0].plot(xx, vv, marker=MK[r.seed], ms=8, color=COL[g],
                           alpha=0.65, markeredgecolor="white", zorder=3)
            rows.append({"figure": "B", "panel": "A", "group": g, "seed": r.seed,
                         "metric": "ar_edit_slope", "value": r.ar_edit_slope,
                         "source": rel(SRC["slopes"])})
            rows.append({"figure": "B", "panel": "A", "group": g, "seed": r.seed,
                         "metric": "gp_edit_slope", "value": r.gp_edit_slope,
                         "source": rel(SRC["slopes"])})
        ax[0].hlines(d.ar_edit_slope.mean(), off - 0.17, off + 0.17,
                     color="black", lw=3.2, zorder=5)
        ax[0].hlines(d.gp_edit_slope.mean(), off + 0.28, off + 0.62,
                     color="black", lw=3.2, zorder=5)
    nv = m2[m2.exposure_status == NOVEL]
    ar_m, gp_m = nv.ar_edit_slope.mean(), nv.gp_edit_slope.mean()
    pct = 100 * gp_m / ar_m
    ax[0].annotate(f"{pct:.0f}% of the AR slope remains\nunder the gold prefix",
                   xy=(0.45, gp_m), xytext=(0.02, ar_m * 1.02), fontsize=11.5,
                   color=COL[NOVEL], fontweight="bold",
                   arrowprops=dict(arrowstyle="->", color=COL[NOVEL], lw=1.4))
    ax[0].set_xticks([0, 0.45, 1.15, 1.60])
    ax[0].set_xticklabels(["Autoregressive\nown previous\noutput",
                           "Gold prefix\ntrue previous\nphonemes"] * 2,
                          fontsize=9.5)
    ax[0].set_ylabel("LTM length slope\n(edit operations per phoneme)")
    ax[0].set_title("A. A substantial slope remains under the gold prefix",
                    loc="left", fontsize=12.5)
    ax[0].grid(alpha=0.25, axis="y")
    ax[0].text(0.225, -0.30, "Novel pseudowords", transform=ax[0].get_xaxis_transform(),
               ha="center", color=COL[NOVEL], fontsize=11.5, fontweight="bold")
    ax[0].text(1.375, -0.30, "Trained real", transform=ax[0].get_xaxis_transform(),
               ha="center", color=COL[TRAINED], fontsize=11.5, fontweight="bold")

    # B: hazard
    h = hz[(hz.route == "ltm") & (hz.event == "FIRST_TOKEN_MISMATCH")]
    dotted_note = []
    for g in (TRAINED, NOVEL):
        d = h[h.exposure_status == g]
        m = d.groupby("position", as_index=False).agg(
            hazard=("hazard", "mean"), at_risk=("n_at_risk", "mean"))
        strong, weak = m[m.at_risk >= RISK_THRESHOLD], m[m.at_risk < RISK_THRESHOLD]
        ax[1].plot(strong.position, strong.hazard, lw=3, marker="o", ms=8,
                   color=COL[g], label=NAME[g])
        if len(weak):
            ax[1].plot(weak.position, weak.hazard, lw=1.6, ls=":", marker="o",
                       ms=6, color=COL[g], alpha=0.45)
            for r in weak.itertuples():
                dotted_note.append(f"{NAME[g]} position {int(r.position)}: "
                                   f"{r.at_risk:.2f} items at risk")
        for r in d.itertuples():
            rows.append({"figure": "B", "panel": "B", "group": g, "seed": r.seed,
                         "metric": f"first_error_hazard_pos{int(r.position)}",
                         "value": r.hazard, "n_at_risk": r.n_at_risk,
                         "source": rel(SRC["hazard"])})
    ax[1].set_xlabel("Phoneme position in the word")
    ax[1].set_ylabel("Probability of first error\namong items still error-free")
    ax[1].set_title("B. First-error risk rises across the word", loc="left",
                    fontsize=12.5)
    ax[1].grid(alpha=0.25)
    ax[1].legend(frameon=False, fontsize=10.5, loc="upper left")
    ax[1].text(0.03, 0.70, f"dotted: fewer than {RISK_THRESHOLD} items\n"
                           f"remain at risk", transform=ax[1].transAxes,
               ha="left", fontsize=9.5, color="0.4", style="italic")

    # C: post-divergence burden
    b = bd[bd.route == "ltm"]
    groups = [TRAINED, UNTRAINED, NOVEL]
    for i, g in enumerate(groups):
        d = b[b.exposure_status == g]
        vals = {s: float(gg.fraction_suffix_wrong.mean())
                for s, gg in d.groupby("seed")}
        m = seeds_and_mean(ax[2], i, vals, COL[g])
        ax[2].text(i, 0.09, f"n={len(d)} erroneous\nseed×item observations",
                   ha="center", fontsize=8.5, color="0.35")
        for s, v in vals.items():
            rows.append({"figure": "B", "panel": "C", "group": g, "seed": s,
                         "metric": "fraction_suffix_wrong", "value": v,
                         "group_mean": m, "source": rel(SRC["burden"])})
    ax[2].set_xticks(range(3))
    ax[2].set_xticklabels([SHORT[g] for g in groups])
    ax[2].set_ylim(0, 1.05)
    ax[2].set_ylabel("Fraction of the remaining\ntarget suffix incorrect")
    ax[2].set_title("C. Errors propagate after the first divergence", loc="left",
                    fontsize=12.5)
    ax[2].grid(alpha=0.25, axis="y")
    nb = b[b.exposure_status == NOVEL].fraction_suffix_wrong.mean()
    e = ev[(ev.route == "ltm") & (ev.exposure_status == NOVEL)
           & ev.FIRST_TOKEN_MISMATCH.notna()]
    n_eos, n_tot = int((e.first_divergence_type == "EOS").sum()), len(e)
    ax[2].annotate(f"≈{nb*100:.0f}%", xy=(2, nb), xytext=(1.45, 0.50),
                   fontsize=12, color=COL[NOVEL], fontweight="bold",
                   arrowprops=dict(arrowstyle="->", color=COL[NOVEL], lw=1.4))
    ax[2].text(0.5, 0.015, f"only {n_eos} of {n_tot} erroneous novel pseudowords "
                           f"begin with a premature EOS",
               transform=ax[2].transAxes, ha="center", fontsize=9.5, color="0.25")
    rows.append({"figure": "B", "panel": "C", "group": NOVEL, "seed": "",
                 "metric": "eos_first_count", "value": n_eos,
                 "n_errors": n_tot, "source": rel(SRC["events"])})

    fig.suptitle("The LTM length effect reflects local difficulty and "
                 "autoregressive amplification", fontsize=16, y=1.03)
    fig.tight_layout()
    save(fig, "figureB_origin_amplification_v2", rows, f"""
**The LTM length effect reflects local difficulty and autoregressive
amplification.**

**Estimators shown:** panel A, per-seed OLS length slopes under two decoding
regimes; panel B, seed-averaged first-error hazard; panel C, per-seed mean
fraction of the remaining suffix incorrect, with the black bar the mean over
seeds.

**A — A substantial slope remains under the gold prefix.** Left of each pair the
route decodes **autoregressively**, feeding its own previous output to the next
tick. Right of each pair is the **gold-prefix** diagnostic:

> At each position in the gold-prefix condition, the decoder receives the true
> preceding target phonemes. Its current tokenwise argmax is measured but is not
> fed into the next position.

For novel pseudowords the slope falls from {ar_m:.4f} to {gp_m:.4f}, so
**{pct:.0f}% of the AR slope remains under the gold prefix**.

**The AR and gold-prefix slopes are measurements under two decoding regimes, not
a causal decomposition into representation and feedback.** Nothing here
attributes {pct:.0f}% of the effect to a representational cause and the
remainder to feedback.

**B — First-error risk rises across the word.** For each position, the number of
items whose **first** error occurs there divided by the items still error-free
and still long enough to reach it. Dotted segments mark positions where fewer
than {RISK_THRESHOLD} items remain at risk (the rule used in V1); the affected
points are: {'; '.join(dotted_note) if dotted_note else 'none'}. This is an
item-level survival curve, not the older interpolated serial-position curve.

**C — Errors propagate after the first divergence.** Once the first divergence
occurs, ≈{nb*100:.0f}% of the remaining target suffix is also incorrect for novel
pseudowords. Only **{n_eos} of {n_tot}** erroneous novel pseudowords begin with a
premature EOS, so early stopping is a late symptom rather than the origin.
**The trained-real mean rests on only 14 erroneous seed×item observations and
should not be interpreted strongly**; its per-seed markers span 0.40 to 0.83.

Sources: `{rel(SRC['slopes'])}`, `{rel(SRC['hazard'])}`, `{rel(SRC['burden'])}`,
`{rel(SRC['events'])}`.
""")
    return rows


# ------------------------------------------------------------------ Figure C
def figure_c():
    d3 = pd.read_csv(SRC["attraction"], sep="\t")
    bl = pd.read_csv(SRC["baseline"], sep="\t")
    wl = pd.read_csv(SRC["word_rescue"], sep="\t")
    pl = pd.read_csv(SRC["pos_rescue"], sep="\t")
    it = pd.read_csv(SRC["items"], sep="\t")
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.8))

    e = d3[(d3.correct == 0) & (d3.exposure_status == NOVEL)]
    for s, g in e.groupby("seed"):
        dt, dn = g.d_pred_target.mean(), g.d_pred_top1.mean()
        ax[0].plot([0, 1], [dt, dn], color=COL[NOVEL], alpha=0.4, lw=1.6, zorder=2)
        for xx, vv in ((0, dt), (1, dn)):
            ax[0].plot(xx, vv, marker=MK[s], ms=9, color=COL[NOVEL], alpha=0.7,
                       markeredgecolor="white", zorder=3)
        rows.append({"figure": "C", "panel": "A", "group": NOVEL, "seed": s,
                     "metric": "mean_d_pred_target", "value": dt,
                     "n_error_items": len(g), "source": rel(SRC["attraction"])})
        rows.append({"figure": "C", "panel": "A", "group": NOVEL, "seed": s,
                     "metric": "mean_d_pred_top1", "value": dn,
                     "n_error_items": len(g), "source": rel(SRC["attraction"])})
    mt, mn = e.d_pred_target.mean(), e.d_pred_top1.mean()
    lex = 100 * e.pred_is_training_form.mean()
    ax[0].hlines(mt, -0.18, 0.18, color="black", lw=3.4, zorder=5)
    ax[0].hlines(mn, 0.82, 1.18, color="black", lw=3.4, zorder=5)
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(["to the target form",
                           "to the top-1\ns_hat bank neighbour"], fontsize=11.5)
    ax[0].set_xlim(-0.5, 1.5)
    ax[0].set_ylim(1.6, 7.8)
    ax[0].set_ylabel("Phoneme Levenshtein distance\nfrom the LTM prediction")
    ax[0].set_title("A. LTM errors remain closer to the target than to the "
                    "bank neighbour", loc="left", fontsize=12)
    ax[0].grid(alpha=0.25, axis="y")
    ax[0].annotate(f"≈{mt:.2f}", xy=(0, mt), xytext=(-0.44, mt), fontsize=12.5,
                   fontweight="bold", color=COL[NOVEL], va="center")
    ax[0].annotate(f"≈{mn:.2f}", xy=(1, mn), xytext=(1.22, mn), fontsize=12.5,
                   fontweight="bold", color=COL[NOVEL], va="center")
    ax[0].text(0.5, 0.03, "No measurable attraction toward the tested\n"
                          "s_hat top-k neighbours", transform=ax[0].transAxes,
               ha="center", fontsize=10.5, color="0.2", style="italic")

    # B: rescue
    def rate(df, col, good, bad):
        out = {}
        for s in SEEDS:
            a = df[(df.seed == s) & (df[col] == good)].n.sum()
            b = df[(df.seed == s) & (df[col] == bad)].n.sum()
            out[s] = a / (a + b) if (a + b) else np.nan
        return out
    w = wl[wl.lichtheim_exposure_status == NOVEL]
    p = pl[pl.exposure_status == NOVEL]
    wr = rate(w, "route_outcome_category", "WM_CORRECT_LTM_WRONG_FULL_CORRECT",
              "WM_CORRECT_LTM_WRONG_FULL_WRONG")
    pr = rate(p, "position_rescue_category",
              "LTM_LOCAL_WRONG_WM_LOCAL_CORRECT_FULL_CORRECT",
              "LTM_LOCAL_WRONG_WM_LOCAL_CORRECT_FULL_WRONG")
    labels = ["Whole words", "Individual phonemes\ncommon FULL prefix"]
    for i, (lbl, vals, src) in enumerate((
            (labels[0], wr, rel(SRC["word_rescue"])),
            (labels[1], pr, rel(SRC["pos_rescue"])))):
        m = seeds_and_mean(ax[1], i, vals, COL[NOVEL])
        ax[1].annotate(f"{m*100:.1f}%", xy=(i, m), xytext=(i, m - 0.052),
                       ha="center", fontsize=14.5, fontweight="bold",
                       color=COL[NOVEL])
        for s, v in vals.items():
            rows.append({"figure": "C", "panel": "B", "group": lbl.split("\n")[0],
                         "seed": s, "metric": "rescue_rate", "value": v,
                         "group_mean": m, "source": src})
    g_novel = it[it.exposure_status == NOVEL].gate.mean()
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(labels, fontsize=11.5)
    ax[1].set_ylim(0.5, 1.03)
    ax[1].set_ylabel("Proportion corrected by FULL\namong WM-correct / LTM-wrong cases")
    ax[1].set_title("B. When WM is correct and LTM is wrong, FULL is correct "
                    "about 99% of the time", loc="left", fontsize=11.5)
    ax[1].grid(alpha=0.25, axis="y")
    ax[1].text(0.5, 0.04, f"Mean pseudoword weights: LTM g ≈ {g_novel:.3f}; "
                          f"WM 1−g ≈ {1-g_novel:.3f}.\n"
                          f"The item-level gate is constant across phoneme positions.",
               transform=ax[1].transAxes, ha="center", fontsize=9.5, color="0.2")
    seed_legend(ax[1], "center left", ncol=1)
    rows.append({"figure": "C", "panel": "B", "group": "gate", "seed": "",
                 "metric": "mean_gate_novel_pseudoword", "value": g_novel,
                 "source": rel(SRC["items"])})

    fig.suptitle("LTM errors remain target-like, while WM supports the FULL output",
                 fontsize=16, y=1.02)
    fig.tight_layout()
    adv = np.average(bl.attraction_advantage, weights=bl.n)
    save(fig, "figureC_target_like_rescue_v2", rows, f"""
**LTM errors remain target-like, while WM supports the FULL output.**

**Estimators shown:** panel A, per-seed mean phoneme Levenshtein distance over
erroneous novel-pseudoword items; panel B, the **mean of the four per-seed
rescue rates** (the V1 estimator, unchanged).

**A — LTM errors remain closer to the target than to the bank neighbour.**
Mean distance from the LTM output to the **target form** is ≈{mt:.2f}; to the
**top-1 `s_hat` bank neighbour** it is ≈{mn:.2f}.

- **Phoneme Levenshtein distance** is the minimum number of phoneme
  substitutions, deletions and insertions transforming one sequence into the
  other.
- **The bank neighbour is chosen by `s_hat`–GloVe cosine similarity** — it is
  the training-bank entry with maximum cosine similarity to the item's `s_hat`.
  It is **not** the phonologically closest known word.
- **The bank vector is never passed into the decoder.** Any relationship between
  output and neighbour would be emergent geometry, not retrieval.
- The matched baseline (neighbour reassigned within length ×
  target-neighbour-distance × exposure × seed strata, distances recomputed)
  gives an attraction advantage of only **{adv:+.3f} phonemes**.
- **This does not exclude all lexical or phonotactic influences** — it is scoped
  to the tested `s_hat` top-k neighbours.
- Approximately **{lex:.2f}%** of erroneous outputs exactly match a
  training-lexicon phonological form.

**B — When WM is correct and LTM is wrong, FULL is correct about 99% of the
time.**

- **Whole words**: independently generated AR route outputs; an item-level
  behavioural outcome category.
- **Individual phonemes**: WM, LTM and FULL local outputs evaluated under the
  same actual FULL-generated prefix.

Mean pseudoword weights: LTM g ≈ {g_novel:.3f}; WM 1−g ≈ {1-g_novel:.3f}. The
item-level gate is **constant across phoneme positions** — it is a single scalar
per word and does not detect or react to individual LTM errors.

Sources: `{rel(SRC['attraction'])}`, `{rel(SRC['baseline'])}`,
`{rel(SRC['word_rescue'])}`, `{rel(SRC['pos_rescue'])}`, `{rel(SRC['items'])}`.
""")
    return rows


def _seed_key(v):
    """Canonical seed key.

    The V1 TSVs carry an empty seed on a few summary rows, which makes pandas
    read the whole column as float64 ("19.0").  Normalising to a bare integer
    string is what lets V1 and V2 rows join.
    """
    try:
        if v is None or (isinstance(v, str) and not v.strip()):
            return ""
        f = float(v)
        return "" if np.isnan(f) else str(int(f))
    except (TypeError, ValueError):
        return str(v)


# ------------------------------------------------- V1 -> normalised long form
def v1_long():
    out = []
    a = pd.read_csv(os.path.join(V1, "figureA_exposure_length_effect.tsv"), sep="\t")
    for r in a.itertuples():
        if r.value == "" or pd.isna(r.value):
            continue
        out.append(("A", r.panel, r.exposure_status, r.seed, r.quantity,
                    float(r.value)))
    b = pd.read_csv(os.path.join(V1, "figureB_origin_amplification.tsv"), sep="\t")
    for r in b.itertuples():
        if r.panel == "A":
            out.append(("B", "A", r.exposure_status, r.seed, "ar_edit_slope",
                        float(r.ar_edit_slope)))
            out.append(("B", "A", r.exposure_status, r.seed, "gp_edit_slope",
                        float(r.gp_edit_slope)))
        elif r.panel == "B":
            out.append(("B", "B", r.exposure_status, r.seed,
                        f"first_error_hazard_pos{int(r.position)}",
                        float(r.hazard)))
        elif r.panel == "C" and not pd.isna(r.fraction_suffix_wrong):
            out.append(("B", "C", r.exposure_status, r.seed,
                        "fraction_suffix_wrong", float(r.fraction_suffix_wrong)))
        elif r.panel == "C" and str(getattr(r, "quantity", "")) == "eos_first_share":
            out.append(("B", "C", r.exposure_status, "", "eos_first_count",
                        float(r.n_eos_first)))
    c = pd.read_csv(os.path.join(V1, "figureC_target_like_rescue.tsv"), sep="\t")
    for r in c.itertuples():
        if r.panel == "A":
            out.append(("C", "A", "NOVEL_PSEUDOWORD", r.seed,
                        "mean_d_pred_target", float(r.mean_d_pred_target)))
            out.append(("C", "A", "NOVEL_PSEUDOWORD", r.seed,
                        "mean_d_pred_top1", float(r.mean_d_pred_top1)))
        elif r.panel == "B" and str(r.level) != "gate" and not pd.isna(r.rescue_rate):
            out.append(("C", "B", r.level, r.seed, "rescue_rate",
                        float(r.rescue_rate)))
        elif r.panel == "B" and str(r.level) == "gate":
            out.append(("C", "B", "gate", "", "mean_gate_novel_pseudoword",
                        float(r.mean_gate_novel_pseudoword)))
    return pd.DataFrame(out, columns=["figure", "panel", "group", "seed",
                                      "metric", "v1_value"])


def v2_long_from_disk():
    out = []
    for fig, stem in (("A", "figureA_exposure_length_effect_v2"),
                      ("B", "figureB_origin_amplification_v2"),
                      ("C", "figureC_target_like_rescue_v2")):
        d = pd.read_csv(os.path.join(OUT, f"{stem}.tsv"), sep="\t")
        for r in d.itertuples():
            v = getattr(r, "value", None)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            out.append((r.figure, r.panel, r.group, r.seed, r.metric, float(v)))
    return pd.DataFrame(out, columns=["figure", "panel", "group", "seed",
                                      "metric", "v2_value"])


def v2_long(rows):
    out = []
    for r in rows:
        if r.get("value") in ("", None) or (isinstance(r.get("value"), float)
                                            and np.isnan(r["value"])):
            continue
        out.append((r["figure"], r["panel"], r["group"], r.get("seed", ""),
                    r["metric"], float(r["value"])))
    return pd.DataFrame(out, columns=["figure", "panel", "group", "seed",
                                      "metric", "v2_value"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    ra, rb, rc = figure_a(), figure_b(), figure_c()

    v1 = v1_long()
    # Read V2 back from the written TSVs rather than from memory: V1 values also
    # round-tripped through text, so comparing file-to-file removes the ~1e-16
    # serialisation noise and makes an exact-zero criterion meaningful.
    v2 = v2_long_from_disk()
    for d in (v1, v2):
        d["seed"] = d["seed"].map(_seed_key)
    m = v1.merge(v2, on=["figure", "panel", "group", "seed", "metric"],
                 how="outer", indicator=True)
    m["absolute_difference"] = (m.v1_value - m.v2_value).abs()
    m["status"] = np.where(
        m._merge == "left_only", "V1_ONLY (removed empty category / textual)",
        np.where(m._merge == "right_only", "V2_ONLY",
                 np.where(m.absolute_difference == 0, "IDENTICAL", "DIFFERS")))
    m.drop(columns="_merge").to_csv(
        os.path.join(OUT, "v1_v2_numeric_equivalence.tsv"), sep="\t", index=False)
    both = m[m.status.isin(["IDENTICAL", "DIFFERS"])]
    print(f"equivalence rows: {len(m)}  compared: {len(both)}  "
          f"max abs diff: {both.absolute_difference.max():.3e}  "
          f"differing: {int((both.status=='DIFFERS').sum())}")
    print(f"V1-only (dropped): {int((m.status.str.startswith('V1_ONLY')).sum())}  "
          f"V2-only: {int((m.status=='V2_ONLY').sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
