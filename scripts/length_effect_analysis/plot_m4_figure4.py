"""Figure 4 — what is lost: serial order, phoneme content, or utilisation?

Adds two conclusions Figure 3 does not carry:

  A  the encoder->s_hat loss is much larger for *ordered* phoneme-at-position
     information than for *unordered* phoneme content, i.e. it is mostly serial
     order that the semantic projection discards;
  B  from the identical gold-prefix premotor state the trained model reads out
     far more accurately than the linear probe does, so the probes are a weak
     instrument and their nulls are not evidence of absence.

Reads only the validated M4 tables.
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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

M4 = os.path.join(ROOT, "outputs/length_effect_mechanism_93a577f/m4_representation")
FIG = os.path.join(ROOT, "outputs/length_effect_mechanism_93a577f/figures")
BASE = "figure4_order_content_utilisation"

STAGES = ["ltm_encoder_hidden", "s_hat", "ltm_decoder_h0", "wm_encoder_hidden"]
LABELS = ["LTM\nencoder h", "raw\ns_hat", "LTM\ndecoder h0", "WM\nencoder h"]
C_ORD = "#8e44ad"
C_UNO = "#16a085"
C_TRAINED = "#1f4e79"
C_NOVEL = "#c0392b"


def _panel_a(ax, summary, unordered, out_rows):
    x = np.arange(len(STAGES))
    o = summary[(summary.variant == "primary") & (summary.length_group == "all")
                & (summary.exposure_status.isin(["TRAINED_REAL_EXACT",
                                                 "NOVEL_PSEUDOWORD"]))]
    ord_acc = [1 - o[o.stage == s]["token_error"].mean() for s in STAGES]
    u = unordered[(unordered.length_group == "all")
                  & (unordered.exposure_status.isin(["TRAINED_REAL_EXACT",
                                                     "NOVEL_PSEUDOWORD"]))]
    uno = [u[u.stage == s]["cosine_pred_target"].mean() for s in STAGES]
    ubase = [u[u.stage == s]["cosine_baseline_target"].mean() for s in STAGES]

    ax.plot(x - 0.06, ord_acc, "-o", color=C_ORD, lw=2, ms=8)
    ax.set_ylabel("ORDERED probe: held-out top-1 accuracy\n"
                  "(phoneme identity at absolute position)", color=C_ORD,
                  fontsize=8.8)
    ax.tick_params(axis="y", colors=C_ORD)
    ax.set_ylim(0.30, 0.75)
    ax2 = ax.twinx()
    ax2.plot(x + 0.06, uno, "-s", color=C_UNO, lw=2, ms=8)
    ax2.plot(x + 0.06, ubase, ":", color=C_UNO, lw=1.4, alpha=.75)
    ax2.set_ylabel("UNORDERED probe: cosine to the\nphoneme-count vector",
                   color=C_UNO, fontsize=8.8)
    ax2.tick_params(axis="y", colors=C_UNO)
    ax2.set_ylim(0.30, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=8.5)
    ax.set_xlim(-0.45, 3.45)
    ax.set_title("A  Encoder → s_hat discards serial order far more than "
                 "phoneme content", fontsize=10, loc="left", pad=8)
    ax.grid(axis="y", alpha=.2, lw=.6)
    ax.set_axisbelow(True)
    ax.annotate(f"ordered: {ord_acc[0]:.3f} → {ord_acc[1]:.3f}\n"
                f"({(ord_acc[0]-ord_acc[1])/ord_acc[0]*100:.0f} % relative drop)",
                xy=(0.52, (ord_acc[0] + ord_acc[1]) / 2), xytext=(0.60, 0.70),
                fontsize=7.8, color=C_ORD, linespacing=1.3,
                arrowprops=dict(arrowstyle="->", color=C_ORD, lw=1.1))
    ax2.annotate(f"unordered: {uno[0]:.3f} → {uno[1]:.3f}\n"
                 f"({(uno[0]-uno[1])/uno[0]*100:.0f} % relative drop)",
                 xy=(0.5, (uno[0] + uno[1]) / 2), xytext=(1.05, 0.935),
                 fontsize=7.8, color=C_UNO, linespacing=1.3,
                 arrowprops=dict(arrowstyle="->", color=C_UNO, lw=1.1))
    ax2.text(3.02, ubase[3], "training-fold\nmean-count baseline", fontsize=7,
             color=C_UNO, va="center", alpha=.9, linespacing=1.2)
    ax.text(-0.40, 0.315, "Two different metrics on two different axes — "
            "their vertical gap is meaningless; only each curve's own shape is.",
            fontsize=7.2, color="#555", style="italic")
    for i, s in enumerate(STAGES):
        out_rows.append({"panel": "A", "stage": s,
                         "metric": "ordered_top1_accuracy", "value": ord_acc[i]})
        out_rows.append({"panel": "A", "stage": s,
                         "metric": "unordered_cosine", "value": uno[i]})
        out_rows.append({"panel": "A", "stage": s,
                         "metric": "unordered_cosine_training_fold_baseline",
                         "value": ubase[i]})


def _panel_b(ax, du, out_rows):
    lens = [3, 4, 5, 7, 8, 9]
    series = [("oof_accuracy_from_ltm_decoder_h0", "h0 probe", ":", "o"),
              ("oof_accuracy_from_ltm_premotor_gold_prefix",
               "gold-prefix premotor probe", "--", "o"),
              ("actual_ltm_gold_prefix_accuracy",
               "ACTUAL model output (same premotor state)", "-", "s")]
    for exp, col in (("TRAINED_REAL_EXACT", C_TRAINED),
                     ("NOVEL_PSEUDOWORD", C_NOVEL)):
        d = du[du.exposure_status == exp]
        for key, _lab, ls, mk in series:
            v = [d[d.phoneme_length == L][key].mean() for L in lens]
            ax.plot(lens, v, ls, color=col, marker=mk, ms=6, lw=1.9,
                    mfc=col if ls == "-" else "white", mew=1.5)
            for L, val in zip(lens, v):
                out_rows.append({"panel": "B", "stage": key,
                                 "metric": f"accuracy|{exp}|len{L}",
                                 "value": val})
    ax.set_xticks(lens)
    ax.set_xlabel("target length (phonemes) — length 6 is absent from WFE "
                  "by design", fontsize=8.8)
    ax.set_ylabel("phoneme-at-position accuracy", fontsize=9)
    ax.set_ylim(0.32, 1.04)
    ax.set_title("B  The trained readout is far better than the probe on the "
                 "identical premotor state", fontsize=10, loc="left", pad=8)
    ax.grid(alpha=.2, lw=.6)
    ax.set_axisbelow(True)
    ax.annotate("the probes sit ~25-45 points below the model's own readout,\n"
                "so a flat probe curve is NOT evidence that information is absent",
                xy=(7, 0.72), xytext=(4.55, 0.55), fontsize=7.8, color="#333",
                linespacing=1.35,
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.0))
    h = [Line2D([], [], color="#555", ls=":", marker="o", mfc="white",
                label="probe on LTM decoder h0"),
         Line2D([], [], color="#555", ls="--", marker="o", mfc="white",
                label="probe on gold-prefix premotor"),
         Line2D([], [], color="#555", ls="-", marker="s", mfc="#555",
                label="ACTUAL model output"),
         Line2D([], [], color=C_TRAINED, lw=3, label="trained real"),
         Line2D([], [], color=C_NOVEL, lw=3, label="novel pseudoword")]
    ax.legend(handles=h, fontsize=7.2, loc="lower left", frameon=True,
              framealpha=.93, ncol=2, columnspacing=1.1)


def main() -> int:
    summary = pd.read_csv(os.path.join(M4, "ordered_probe_summary.tsv"), sep="\t")
    unordered = pd.read_csv(os.path.join(M4, "unordered_probe_summary.tsv"),
                            sep="\t")
    du = pd.read_csv(os.path.join(M4, "decoder_utilisation.tsv"), sep="\t")
    du = du[du.exposure_status.isin(["TRAINED_REAL_EXACT", "NOVEL_PSEUDOWORD"])]
    rows = []
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 8.4))
    fig.suptitle("What is lost: serial order, phoneme content, or utilisation?",
                 fontsize=13, y=0.975, x=0.055, ha="left", weight="bold")
    fig.text(0.055, 0.941,
             "Averages over seeds 19-22 and over the two confirmatory exposure "
             "groups in panel A (they are indistinguishable there).",
             fontsize=8.4, ha="left", color="#444")
    _panel_a(axes[0], summary, unordered, rows)
    _panel_b(axes[1], du, rows)
    fig.tight_layout(rect=(0, 0.005, 1, 0.93))
    os.makedirs(FIG, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(FIG, f"{BASE}.tsv"), sep="\t",
                              index=False)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(FIG, f"{BASE}.{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(FIG, BASE + ".{png,pdf,svg,tsv}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
