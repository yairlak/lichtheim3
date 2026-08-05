"""Figure 3 — where does phonological information become length-sensitive?

Reads only the validated M4 backing tables.  No probe is refitted here; every
plotted number is copied from `ordered_probe_summary.tsv`,
`ordered_probe_length_slopes.tsv` and `stage_contrasts.tsv`, and is written out
to `figure3_stagewise_information.tsv` so the figure and its backing data cannot
drift apart.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.length_effect_analysis.m4_probes import LENGTH_NOTE, SEEDS  # noqa: E402

M4 = os.path.join(ROOT, "outputs/length_effect_mechanism_93a577f/m4_representation")
FIG = os.path.join(ROOT, "outputs/length_effect_mechanism_93a577f/figures")
BASE = "figure3_stagewise_information"

LTM = ["ltm_encoder_hidden", "s_hat", "ltm_decoder_h0",
       "ltm_premotor_gold_prefix", "ltm_actual_gold_prefix_output"]
SHORT_LABEL = ["LTM\nencoder h", "raw\ns_hat", "LTM\ndecoder h0",
               "gold-prefix\npremotor", "actual gold-\nprefix output"]
WM = {"wm_encoder_hidden": 0, "wm_premotor_gold_prefix": 3}

C_TRAINED = "#1f4e79"
C_NOVEL = "#c0392b"
C_WM = "#7f8c8d"
GOLD_TINT = "#fdf3e3"
ACTUAL_TINT = "#e9f0e4"


def _long(summary, slopes, contrasts):
    """Assemble the exact backing table the figure draws from."""
    rows = []
    s = summary[(summary["variant"] == "primary")]
    for _, r in s.iterrows():
        if r["stage"] not in LTM and r["stage"] not in WM:
            continue
        if r["exposure_status"] not in ("TRAINED_REAL_EXACT", "NOVEL_PSEUDOWORD"):
            continue
        rows.append({"panel": "A", "stage": r["stage"],
                     "stage_kind": r["stage_kind"],
                     "exposure_status": r["exposure_status"],
                     "length_group": r["length_group"], "seed": r["seed"],
                     "metric": "held_out_token_error", "value": r["token_error"]})
    sl = slopes[slopes["variant"] == "primary"]
    for _, r in sl.iterrows():
        if r["stage"] not in LTM and r["stage"] not in WM:
            continue
        if r["exposure_status"] not in ("TRAINED_REAL_EXACT", "NOVEL_PSEUDOWORD"):
            continue
        rows.append({"panel": "B", "stage": r["stage"],
                     "stage_kind": r["stage_kind"],
                     "exposure_status": r["exposure_status"],
                     "length_group": "all", "seed": r["seed"],
                     "metric": "length_slope_token_error_per_phoneme",
                     "value": r["length_slope_token_error_per_phoneme"]})
    c = contrasts[contrasts["contrast_kind"] == "length_slope"]
    for _, r in c.iterrows():
        for m, v in (("bootstrap_mean_length_slope", r["bootstrap_mean"]),
                     ("ci_low_length_slope", r["ci_low"]),
                     ("ci_high_length_slope", r["ci_high"])):
            rows.append({"panel": "B", "stage": r["stage"], "stage_kind": "",
                         "exposure_status": r["exposure_status"],
                         "length_group": "all", "seed": np.nan,
                         "metric": m, "value": v})
    out = pd.DataFrame(rows)
    out["length_note"] = LENGTH_NOTE
    return out


def _panel_a(ax, tbl):
    x = np.arange(len(LTM))
    ax.axvspan(2.5, 3.5, color=GOLD_TINT, zorder=0)
    ax.axvspan(3.5, 4.5, color=ACTUAL_TINT, zorder=0)
    ax.axvline(2.5, color="#b58900", lw=1.4, ls="--", zorder=1)
    ax.axvline(3.5, color="#4a7023", lw=1.4, ls="-", zorder=1)

    a = tbl[(tbl.panel == "A") & (tbl.metric == "held_out_token_error")]
    # trained and novel land on almost identical values at every probe stage, so
    # the two series are drawn with a small x offset; without it one hides the
    # other and the reader sees a missing line rather than an equality.
    for exp, col, off in (("TRAINED_REAL_EXACT", C_TRAINED, -0.075),
                          ("NOVEL_PSEUDOWORD", C_NOVEL, 0.075)):
        for lg, ls, mfc in (("short (3-5)", ":", "white"),
                            ("long (7-9)", "-", col)):
            sub = a[(a.exposure_status == exp) & (a.length_group == lg)]
            m = [sub[sub.stage == st]["value"].mean() for st in LTM]
            for st_i, st in enumerate(LTM):
                v = sub[sub.stage == st]["value"].to_numpy()
                ax.plot(np.full(len(v), x[st_i] + off * 2.1), v, ".",
                        color=col, ms=3.5, alpha=.5, zorder=3)
            xo = x + off
            # connect only the three global item representations; the two
            # boundaries are measurement-type changes, not steps along one scale
            ax.plot(xo[:3], m[:3], ls, color=col, lw=2, marker="o", ms=7,
                    mfc=mfc, mec=col, mew=1.6, zorder=4)
            ax.plot(xo[2:4], m[2:4], color=col, lw=1.0, ls=(0, (1, 2)),
                    alpha=.55, zorder=3)
            ax.plot([xo[3]], [m[3]], ls, color=col, lw=0, marker="o", ms=7,
                    mfc=mfc, mec=col, mew=1.6, zorder=4)
            ax.plot([xo[4]], [m[4]], color=col, lw=0, marker="s", ms=9,
                    mfc=mfc, mec=col, mew=1.8, zorder=4)
    for st, xi in WM.items():
        sub = a[(a.stage == st) & (a.length_group == "long (7-9)")]
        ax.plot([xi + 0.28], [sub["value"].mean()], "D", color=C_WM, ms=6,
                zorder=5)
        ax.annotate("WM", (xi + 0.28, sub["value"].mean()),
                    textcoords="offset points", xytext=(7, -3), fontsize=7.5,
                    color=C_WM)

    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_LABEL, fontsize=8.5)
    ax.set_ylabel("held-out token error\n(lower = more linearly accessible)",
                  fontsize=9)
    ax.set_ylim(-0.03, 0.66)
    ax.set_xlim(-0.45, 4.55)
    ax.text(3.0, 0.635, "gold-prefix decoder context\ncontributes from here",
            ha="center", va="top", fontsize=7.4, color="#8a6d00",
            linespacing=1.25)
    ax.text(4.0, 0.635, "actual trained-model\noutput — NOT a probe",
            ha="center", va="top", fontsize=7.4, color="#3d5c1c",
            linespacing=1.25)
    ax.text(1.25, 0.635, "diagnostic OOF linear readouts", ha="center",
            va="top", fontsize=7.4, color="#555")
    ax.annotate("trained and novel are nearly\nindistinguishable at every\n"
                "probe stage (red and blue\noverlap; offset for visibility)",
                xy=(1.0, 0.545), xytext=(1.42, 0.335), fontsize=7.6,
                color="#333", linespacing=1.3,
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.0))
    ax.annotate("the exposure gap appears\nonly here: 0.1 % vs 5.1 %",
                xy=(4.0, 0.055), xytext=(2.75, 0.175), fontsize=7.8,
                color="#3d5c1c", linespacing=1.3,
                arrowprops=dict(arrowstyle="->", color="#3d5c1c", lw=1.2))
    ax.set_title("A  Linearly accessible phoneme-at-position information",
                 fontsize=10, loc="left", pad=8)
    ax.grid(axis="y", alpha=.25, lw=.6)
    ax.set_axisbelow(True)

    h = [Line2D([], [], color=C_TRAINED, lw=2, marker="o", mfc=C_TRAINED,
                label="trained real, long (7-9)"),
         Line2D([], [], color=C_TRAINED, lw=2, ls=":", marker="o", mfc="white",
                label="trained real, short (3-5)"),
         Line2D([], [], color=C_NOVEL, lw=2, marker="o", mfc=C_NOVEL,
                label="novel pseudoword, long (7-9)"),
         Line2D([], [], color=C_NOVEL, lw=2, ls=":", marker="o", mfc="white",
                label="novel pseudoword, short (3-5)"),
         Line2D([], [], color=C_WM, lw=0, marker="D", ms=6,
                label="WM control (long)"),
         Line2D([], [], color="#888", lw=0, marker=".", ms=6,
                label="individual seeds (19-22)")]
    ax.legend(handles=h, fontsize=7.2, loc="lower left", frameon=True,
              framealpha=.92, ncol=2, handlelength=2.4, columnspacing=1.1,
              bbox_to_anchor=(0.0, 0.02))


def _panel_b(ax, tbl):
    x = np.arange(len(LTM))
    ax.axvspan(2.5, 3.5, color=GOLD_TINT, zorder=0)
    ax.axvspan(3.5, 4.5, color=ACTUAL_TINT, zorder=0)
    ax.axvline(2.5, color="#b58900", lw=1.4, ls="--", zorder=1)
    ax.axvline(3.5, color="#4a7023", lw=1.4, ls="-", zorder=1)
    ax.axhline(0, color="#333", lw=.9, zorder=2)

    b = tbl[tbl.panel == "B"]
    for exp, col, off in (("TRAINED_REAL_EXACT", C_TRAINED, -0.11),
                          ("NOVEL_PSEUDOWORD", C_NOVEL, 0.11)):
        seeds = b[(b.exposure_status == exp)
                  & (b.metric == "length_slope_token_error_per_phoneme")]
        mean = b[(b.exposure_status == exp)
                 & (b.metric == "bootstrap_mean_length_slope")]
        lo = b[(b.exposure_status == exp) & (b.metric == "ci_low_length_slope")]
        hi = b[(b.exposure_status == exp) & (b.metric == "ci_high_length_slope")]
        mu = np.array([mean[mean.stage == st]["value"].iloc[0] for st in LTM])
        l = np.array([lo[lo.stage == st]["value"].iloc[0] for st in LTM])
        h = np.array([hi[hi.stage == st]["value"].iloc[0] for st in LTM])
        for i, st in enumerate(LTM):
            v = seeds[seeds.stage == st]["value"].to_numpy()
            ax.plot(np.full(len(v), x[i] + off), v, ".", color=col, ms=4.5,
                    alpha=.6, zorder=3)
        ax.errorbar(x[:4] + off, mu[:4], yerr=[mu[:4] - l[:4], h[:4] - mu[:4]],
                    fmt="o", color=col, ms=7, lw=1.6, capsize=3, zorder=4,
                    label=None)
        ax.errorbar(x[4:] + off, mu[4:], yerr=[mu[4:] - l[4:], h[4:] - mu[4:]],
                    fmt="s", color=col, ms=9, lw=1.8, capsize=3, zorder=4)
        ax.plot(x[:4] + off, mu[:4], "-", color=col, lw=1.5, alpha=.75, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_LABEL, fontsize=8.5)
    ax.set_ylabel("length slope\n(token error per additional phoneme)",
                  fontsize=9)
    ax.set_xlim(-0.45, 4.55)
    ax.set_title("B  Length sensitivity by stage", fontsize=10, loc="left",
                 pad=8)
    ax.grid(axis="y", alpha=.25, lw=.6)
    ax.set_axisbelow(True)
    ax.annotate("the ONLY stage where the novel slope\nexceeds the trained "
                "slope — and it is the\nmodel's own output, not a probe",
                xy=(4.0, 0.0145), xytext=(2.30, 0.031), fontsize=7.8,
                color=C_NOVEL, linespacing=1.3,
                arrowprops=dict(arrowstyle="->", color=C_NOVEL, lw=1.2))
    ax.annotate("at every probe stage the TRAINED forms carry\nthe LARGER "
                "length slope — the opposite of the\nbehavioural effect",
                xy=(1.0, 0.0505), fontsize=7.8, color=C_TRAINED,
                linespacing=1.3, va="top")
    h = [Line2D([], [], color=C_TRAINED, lw=1.6, marker="o", label="trained real"),
         Line2D([], [], color=C_NOVEL, lw=1.6, marker="o", label="novel pseudoword"),
         Line2D([], [], color="#888", lw=0, marker=".", ms=6, label="seeds 19-22"),
         Line2D([], [], color="#888", lw=1.4, label="95% hierarchical bootstrap CI")]
    ax.legend(handles=h, fontsize=7.2, loc="lower left", frameon=True,
              framealpha=.92)


def main() -> int:
    summary = pd.read_csv(os.path.join(M4, "ordered_probe_summary.tsv"), sep="\t")
    slopes = pd.read_csv(os.path.join(M4, "ordered_probe_length_slopes.tsv"),
                         sep="\t")
    contrasts = pd.read_csv(os.path.join(M4, "stage_contrasts.tsv"), sep="\t")
    tbl = _long(summary, slopes, contrasts)
    os.makedirs(FIG, exist_ok=True)
    tbl.to_csv(os.path.join(FIG, f"{BASE}.tsv"), sep="\t", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(9.4, 9.0))
    fig.suptitle("Where does phonological information become length-sensitive?",
                 fontsize=13.5, y=0.977, x=0.055, ha="left", weight="bold")
    fig.text(0.055, 0.943,
             "Stages 1-4 are diagnostic out-of-fold linear readouts "
             "(linearly accessible information). Stage 5 is the model's own "
             "output and is not a probe.",
             fontsize=8.6, ha="left", color="#444")
    _panel_a(axes[0], tbl)
    _panel_b(axes[1], tbl)
    fig.tight_layout(rect=(0, 0.005, 1, 0.932))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(FIG, f"{BASE}.{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(FIG, BASE + ".{png,pdf,svg,tsv}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
