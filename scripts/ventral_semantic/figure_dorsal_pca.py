"""Phase C1: dorsal (h_WM) PCA word maps.

Methodologically identical to the ventral maps: the PCA fit, the rendering
geometry, the font, the aspect handling and the "every word as text at its exact
coordinate" contract are all imported from `figure3_independent_pca`, so the
ventral and dorsal figures differ only in which representation is plotted.

Per seed: L2-normalise h_WM, fit a separate 2-component PCA on that seed's own
vectors, draw all 29,571 words.  No shared transform, no pooling of seeds, no
points, no lines, no offsets.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.ventral_semantic.analyse_and_plot import (              # noqa: E402
    COHORT, EXTRACT, FIGS, N_WORDS, REPORT, TABLES, sha_file, unit)
from scripts.ventral_semantic.figure3_independent_pca import (       # noqa: E402
    FONT_PT, LONG_SIDE_IN, PNG_DPI, TEXT_ALPHA, fit_pca2, git,
    render_single)

WM_TEXT = "#1f4e79"          # dorsal colour, distinct from the ventral maps
WM_DIM = 128


def main() -> int:
    words = [w for w in
             open(os.path.join(EXTRACT, "lexicon_words.txt")).read().split("\n")
             if w]
    assert len(words) == N_WORDS

    spaces, renders, rows = {}, {}, []
    for seed, epoch in COHORT:
        h = np.load(os.path.join(EXTRACT, f"h_wm_lexicon_seed{seed}.npy"))
        assert h.shape == (N_WORDS, WM_DIM), h.shape
        Y, evr = fit_pca2(unit(h))
        spaces[f"h_wm_seed{seed}"] = (Y, evr)
        stem = f"fig4_wordmap_h_wm_seed{seed}"
        print(f"{stem}:")
        renders[f"h_wm_seed{seed}"] = render_single(
            Y, evr, words, WM_TEXT,
            f"h_WM  (dorsal encoder state)  —  seed {seed} / epoch {epoch}",
            stem)
        for i, w in enumerate(words):
            rows.append({"space": f"h_wm_seed{seed}", "lexicon_index": i,
                         "word": w, "pc1": float(Y[i, 0]),
                         "pc2": float(Y[i, 1])})

    os.makedirs(TABLES, exist_ok=True)
    tsv = os.path.join(TABLES, "dorsal_pca_all_word_coordinates.tsv")
    pd.DataFrame(rows).to_csv(tsv, sep="\t", index=False)

    var = {k: {"pc1": float(v[1][0]), "pc2": float(v[1][1]),
               "pc1_plus_pc2": float(v[1].sum())} for k, v in spaces.items()}
    meta = {
        "figure_family": "fig4_wordmap_h_wm",
        "phase": "C1 - dorsal PCA",
        "representation": ("h_WM, the word-level dorsal encoder state from "
                           "WMRecurrent.encode (1,B,128) -> (B,128), "
                           "deterministic pre-noise"),
        "normalisation": "L2 per row before PCA",
        "pca_fits": {f"h_wm_seed{s}": f"fitted on seed-{s} h_WM only"
                     for s, _ in COHORT},
        "seeds_pooled_for_fitting": False,
        "transform_shared_across_seeds": False,
        "methodologically_matched_to_ventral": (
            "fit_pca2, render_single, font, aspect and layout imported "
            "unchanged from figure3_independent_pca"),
        "points_drawn": False, "arrows_or_leader_lines_drawn": False,
        "collision_avoidance_applied": False,
        "words_at_exact_pca_coordinates": True, "subsampled": False,
        "font_size_pt": FONT_PT, "text_alpha": TEXT_ALPHA,
        "long_side_inches": LONG_SIDE_IN, "png_dpi": PNG_DPI,
        "explained_variance_ratio": var,
        "n_words": N_WORDS,
        "renders": renders,
        "all_word_coordinates_tsv": tsv,
        "inputs": {f"h_wm_lexicon_seed{s}.npy":
                   sha_file(os.path.join(EXTRACT, f"h_wm_lexicon_seed{s}.npy"))
                   for s, _ in COHORT},
        "model_loaded": False, "inference_run": False,
        "training_performed": False,
        "repository_head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    }
    with open(os.path.join(REPORT, "figure4_dorsal_pca_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    for k, v in var.items():
        print(f"{k:16s} PC1={v['pc1']*100:6.2f}%  PC2={v['pc2']*100:6.2f}%  "
              f"PC1+PC2={v['pc1_plus_pc2']*100:6.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
