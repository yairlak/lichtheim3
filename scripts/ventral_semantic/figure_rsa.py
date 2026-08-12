"""Phase C2 main figure: is the ventral geometry semantic and the dorsal one
phonological?

One panel per canonical seed.  For each route (LTM ventral, WM dorsal) the
Pearson correlation of its representational distance with the semantic reference
(GloVe) and with the phonological reference (Levenshtein) is shown side by side,
so the comparison is immediate.

Raw Levenshtein is the primary phonological reference and is drawn solid.
Normalised Levenshtein is a sensitivity analysis and is drawn hatched next to it,
because it changes the magnitude of the dorsal-phonological correlation
substantially and hiding that would misrepresent the result.

Reads the RSA tables only; nothing is recomputed.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.ventral_semantic.analyse_and_plot import (              # noqa: E402
    COHORT, FIGS, REPORT, TABLES)

SEM_COLOR = "#1f6f8b"
PHON_COLOR = "#b8562f"
STEM = "fig5_rsa_ventral_semantic_dorsal_phonological"


def main() -> int:
    c = pd.read_csv(os.path.join(TABLES, "rsa_full_lexicon_correlations.tsv"),
                    sep="\t")
    reg = pd.read_csv(os.path.join(
        TABLES, "rsa_full_lexicon_partial_regression.tsv"), sep="\t")
    n_pairs = int(c["n_pairs"].iloc[0])
    gp_raw = float(c[(c.reference == "GloVe vs phon raw")]["pearson_r"].iloc[0])
    gp_norm = float(
        c[(c.reference == "GloVe vs phon normalised")]["pearson_r"].iloc[0])

    routes = ["LTM (ventral)", "WM (dorsal)"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.9), sharey=True)
    width = 0.22
    for ax, (seed, epoch) in zip(axes, COHORT):
        d = c[c.seed.astype(str) == str(seed)]
        x = np.arange(len(routes))
        for k, route in enumerate(routes):
            r = d[d.route == route]
            g = float(r[r.reference == "GloVe (semantic)"]["pearson_r"].iloc[0])
            p = float(r[r.reference == "phon raw Levenshtein"]["pearson_r"].iloc[0])
            pn = float(r[r.reference ==
                         "phon normalised Levenshtein"]["pearson_r"].iloc[0])
            bars = [(-1.5 * width, g, SEM_COLOR, None),
                    (0.0, p, PHON_COLOR, None),
                    (1.2 * width, pn, PHON_COLOR, "///")]
            for dx, val, col, hatch in bars:
                ax.bar(x[k] + dx, val, width, color=col, edgecolor="#333",
                       linewidth=.7, hatch=hatch,
                       alpha=1.0 if hatch is None else .55, zorder=3)
                ax.annotate(f"{val:.3f}", (x[k] + dx, val), xytext=(0, 3),
                            textcoords="offset points", ha="center",
                            fontsize=8.6, zorder=4)
        ax.axhline(gp_raw, color="#666", lw=1.1, ls="--", zorder=2)
        ax.text(0.315, 0.985,
                f"dashed line: GloVe vs phon (raw) r = {gp_raw:.3f}\n"
                "the two reference geometries are\nnearly independent, so the "
                "partial\nbetas match the simple correlations",
                transform=ax.transAxes, fontsize=8.0, color="#555",
                ha="left", va="top", linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#bbb", linewidth=.8))
        ax.set_xticks(x)
        ax.set_xticklabels(routes, fontsize=11.5)
        ax.set_title(f"seed {seed} / epoch {epoch}", fontsize=13, pad=8)
        ax.grid(axis="y", alpha=.22, lw=.6)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 0.50)
    axes[0].set_ylabel("Pearson r  of representational distance\n"
                       "with the reference distance", fontsize=11)

    handles = [
        Patch(facecolor=SEM_COLOR, edgecolor="#333",
              label="vs GloVe semantic distance"),
        Patch(facecolor=PHON_COLOR, edgecolor="#333",
              label="vs phonological distance — raw Levenshtein (primary)"),
        Patch(facecolor=PHON_COLOR, edgecolor="#333", hatch="///", alpha=.55,
              label="vs phonological distance — normalised (sensitivity)")]
    fig.legend(handles=handles, fontsize=9.5, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Ventral geometry is semantic, dorsal geometry is phonological",
                 fontsize=14.5, y=0.985)
    fig.text(0.5, 0.925,
             f"Exact RSA over all {n_pairs:,} unique word pairs of the 29,571-word "
             "training lexicon — no sampling, no p-values (pairs are not "
             "independent).",
             ha="center", fontsize=9.5, color="#444")
    fig.tight_layout(rect=(0, 0.055, 1, 0.915))

    os.makedirs(FIGS, exist_ok=True)
    out = {}
    for ext in ("png", "pdf", "svg"):
        p = os.path.join(FIGS, f"{STEM}.{ext}")
        fig.savefig(p, dpi=250, bbox_inches="tight")
        out[ext] = p
    plt.close(fig)

    meta = {"figure": STEM, "n_unique_pairs": n_pairs,
            "primary_phonological_reference": "raw Levenshtein",
            "sensitivity_phonological_reference":
                "Levenshtein / max(len_i, len_j)",
            "reference_collinearity": {"glove_vs_phon_raw": gp_raw,
                                       "glove_vs_phon_normalised": gp_norm},
            "p_values_reported": False,
            "sources": {
                "correlations": "tables/rsa_full_lexicon_correlations.tsv",
                "partial_regression":
                    "tables/rsa_full_lexicon_partial_regression.tsv"},
            "outputs": out,
            "model_loaded": False, "inference_run": False}
    with open(os.path.join(REPORT, "figure5_rsa_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print("wrote", out["png"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
