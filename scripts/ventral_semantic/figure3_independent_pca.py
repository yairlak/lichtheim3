"""Figure 3: three INDEPENDENTLY fitted PCA spaces, drawn as word maps.

Each representation gets its own basis, fitted on its own vectors:

    PCA_GloVe  fitted on the 29,571 L2-normalised GloVe vectors
    PCA_s19    fitted on the 29,571 L2-normalised seed-19 s_hat vectors
    PCA_s22    fitted on the 29,571 L2-normalised seed-22 s_hat vectors

No transform is shared, and the two seeds are never pooled.

Rendering contract:
  * every data item is represented ONLY by its word, drawn with `ax.text()` at
    its exact PCA coordinate;
  * no scatter points, no arrows, no leader lines, no subsampling;
  * **no collision avoidance** — a word is never nudged off its coordinate, so
    overlap in dense regions is expected and is the honest depiction;
  * the natural PCA aspect ratio is preserved (`set_aspect("equal")`).

Primary outputs are three separate full-size figures, one per space, with PDF as
the zoom format.  A combined three-panel overview is produced as an optional
extra, not as the primary figure.

Reads the frozen extraction only.  No model is loaded, no inference is run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.ventral_semantic.analyse_and_plot import (              # noqa: E402
    COHORT, DIM, EXTRACT, FIGS, N_WORDS, REPORT, TABLES, load, sha_file, unit)

# --------------------------------------------------------------- render config
LONG_SIDE_IN = 22.0        # inches on the longer data axis of a single-space map
MARGIN_IN = 1.7            # room for tick labels, axis labels and title
FONT_PT = 4.0              # substantially larger than the previous 1.7 pt
TEXT_ALPHA = 0.95          # near-opaque
PNG_DPI = 250
PAD_FRAC = 0.02

GLOVE_TEXT = "#26323d"
PRED_TEXT = "#7d2718"

SPACES = [("glove", "GloVe semantic bank", GLOVE_TEXT,
           "fig3_wordmap_glove"),
          (f"s_hat_seed{COHORT[0][0]}",
           f"s_hat  —  seed {COHORT[0][0]} / epoch {COHORT[0][1]}", PRED_TEXT,
           f"fig3_wordmap_s_hat_seed{COHORT[0][0]}"),
          (f"s_hat_seed{COHORT[1][0]}",
           f"s_hat  —  seed {COHORT[1][0]} / epoch {COHORT[1][1]}", PRED_TEXT,
           f"fig3_wordmap_s_hat_seed{COHORT[1][0]}")]


def git(*a) -> str:
    return subprocess.run(("git",) + a, cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def fit_pca2(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(2-D scores, explained variance ratio of PC1 and PC2)."""
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    ratio = (S ** 2) / (S ** 2).sum()
    return Xc @ Vt[:2].T, ratio[:2]


def _limits(Y: np.ndarray) -> Tuple[float, float, float, float]:
    dx, dy = float(np.ptp(Y[:, 0])), float(np.ptp(Y[:, 1]))
    px, py = PAD_FRAC * dx, PAD_FRAC * dy
    return (Y[:, 0].min() - px, Y[:, 0].max() + px,
            Y[:, 1].min() - py, Y[:, 1].max() + py)


def draw_words(ax, Y: np.ndarray, words: List[str], color: str,
               fontsize: float) -> int:
    """Every word at its exact coordinate.  No points, no offsets."""
    for (x, y), w in zip(Y, words):
        ax.text(x, y, w, fontsize=fontsize, color=color, alpha=TEXT_ALPHA,
                ha="center", va="center", clip_on=True)
    return len(words)


def style(ax, Y: np.ndarray, evr: np.ndarray, title: str) -> None:
    x0, x1, y0, y1 = _limits(Y)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")     # natural PCA aspect preserved
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel(f"PC1  ({evr[0] * 100:.1f} % of this space's variance)",
                  fontsize=12)
    ax.set_ylabel(f"PC2  ({evr[1] * 100:.1f} % of this space's variance)",
                  fontsize=12)
    ax.grid(alpha=.15, lw=.5)
    ax.set_axisbelow(True)


def render_single(Y: np.ndarray, evr: np.ndarray, words: List[str],
                  color: str, title: str, stem: str) -> Dict[str, object]:
    x0, x1, y0, y1 = _limits(Y)
    dx, dy = x1 - x0, y1 - y0
    scale = LONG_SIDE_IN / max(dx, dy)
    figsize = (dx * scale + MARGIN_IN, dy * scale + MARGIN_IN)

    fig, ax = plt.subplots(figsize=figsize)
    n = draw_words(ax, Y, words, color, FONT_PT)
    assert n == N_WORDS, n
    style(ax, Y, evr, title)
    fig.text(0.5, 0.005,
             f"All {N_WORDS:,} words drawn at their exact PCA coordinates with "
             "ax.text(); no points, no arrows, no collision avoidance. "
             "PCA fitted on this space only — coordinates are not comparable "
             "with the other maps.",
             ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, 0.018, 1, 1))

    out = {}
    for ext, kw in (("pdf", {}), ("png", {"dpi": PNG_DPI})):
        p = os.path.join(FIGS, f"{stem}.{ext}")
        t = time.time()
        fig.savefig(p, bbox_inches="tight", **kw)
        out[ext] = p
        print(f"    {ext:3s} {os.path.getsize(p) / 1e6:6.1f} MB  "
              f"({time.time() - t:.0f}s)")
    plt.close(fig)
    return {"paths": out,
            "figsize_inches": [round(float(v), 2) for v in figsize],
            "aspect_data_dx_dy": [round(float(dx), 4), round(float(dy), 4)],
            "fontsize_pt": FONT_PT, "n_words_drawn": n}


def render_overview(spaces, words: List[str]) -> Dict[str, str]:
    """Optional 3-panel overview.  Not the primary output."""
    fig, axes = plt.subplots(1, 3, figsize=(30.0, 10.5))
    for ax, (key, title, color, _) in zip(axes, SPACES):
        Y, evr = spaces[key]
        draw_words(ax, Y, words, color, 2.2)
        style(ax, Y, evr, title)
    fig.suptitle("Optional overview — each representation in its OWN "
                 "principal-component basis (see the three full-size word maps "
                 "for reading)", fontsize=15, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = {}
    for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):
        p = os.path.join(FIGS, f"fig3_wordmap_overview.{ext}")
        fig.savefig(p, bbox_inches="tight", **kw)
        out[ext] = p
        print(f"    {ext:3s} {os.path.getsize(p) / 1e6:6.1f} MB")
    plt.close(fig)
    return out


def main() -> int:
    bank, s, words = load()
    assert len(words) == N_WORDS

    spaces = {"glove": fit_pca2(bank)}
    for seed, _ in COHORT:
        spaces[f"s_hat_seed{seed}"] = fit_pca2(unit(s[seed]))

    os.makedirs(FIGS, exist_ok=True)
    primary = {}
    for key, title, color, stem in SPACES:
        Y, evr = spaces[key]
        print(f"{stem}:")
        primary[key] = render_single(Y, evr, words, color, title, stem)
    print("overview (optional):")
    overview = render_overview(spaces, words)

    rows = []
    for key, (Y, _) in spaces.items():
        for i, w in enumerate(words):
            rows.append({"space": key, "lexicon_index": i, "word": w,
                         "pc1": float(Y[i, 0]), "pc2": float(Y[i, 1])})
    os.makedirs(TABLES, exist_ok=True)
    tsv = os.path.join(TABLES, "independent_pca_all_word_coordinates.tsv")
    pd.DataFrame(rows).to_csv(tsv, sep="\t", index=False)

    var = {k: {"pc1": float(v[1][0]), "pc2": float(v[1][1]),
               "pc1_plus_pc2": float(v[1].sum())} for k, v in spaces.items()}
    meta = {
        "figure_family": "fig3_wordmap",
        "description": ("three independently fitted 2-component PCAs, each "
                        "rendered as a full-size word map: every one of the "
                        "29,571 words drawn as text at its exact coordinate"),
        "primary_outputs": "three separate single-space figures",
        "overview_is_optional": True,
        "pca_fits": {
            "glove": "fitted on the 29,571 L2-normalised GloVe vectors",
            f"s_hat_seed{COHORT[0][0]}":
                "fitted on the 29,571 L2-normalised seed-19 s_hat vectors",
            f"s_hat_seed{COHORT[1][0]}":
                "fitted on the 29,571 L2-normalised seed-22 s_hat vectors"},
        "glove_transform_applied_to_s_hat": False,
        "seeds_pooled_for_fitting": False,
        "background_scatter_drawn": False,
        "points_drawn": False,
        "arrows_or_leader_lines_drawn": False,
        "collision_avoidance_applied": False,
        "words_at_exact_pca_coordinates": True,
        "subsampled": False,
        "aspect_ratio": "natural PCA aspect preserved via set_aspect('equal')",
        "font_size_pt": FONT_PT,
        "text_alpha": TEXT_ALPHA,
        "long_side_inches": LONG_SIDE_IN,
        "png_dpi": PNG_DPI,
        "coordinates_comparable_across_maps": False,
        "pc_sign_and_rotation": ("each basis is fitted separately and is "
                                 "determined up to a sign per component, so "
                                 "map orientations are arbitrary relative to "
                                 "each other and are not aligned"),
        "explained_variance_ratio": var,
        "n_words": N_WORDS, "semantic_dim": DIM,
        "renders": primary,
        "overview": overview,
        "all_word_coordinates_tsv": tsv,
        "inputs": {
            "glove_bank_normalised.npy":
                sha_file(os.path.join(EXTRACT, "glove_bank_normalised.npy")),
            **{f"s_hat_lexicon_seed{s_}.npy":
               sha_file(os.path.join(EXTRACT, f"s_hat_lexicon_seed{s_}.npy"))
               for s_, _ in COHORT}},
        "model_loaded": False, "inference_run": False,
        "training_performed": False,
        "repository_head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    }
    with open(os.path.join(REPORT, "figure3_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    for k, v in var.items():
        print(f"{k:16s} PC1={v['pc1']*100:6.2f}%  PC2={v['pc2']*100:6.2f}%  "
              f"PC1+PC2={v['pc1_plus_pc2']*100:6.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
