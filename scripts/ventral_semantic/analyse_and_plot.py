"""Phase B figures: GloVe/s_hat PCA alignment, and target semantic identification.

Reads only the frozen extraction produced by `extract_lexicon_s_hat.py`.
No model is loaded, no inference is run, nothing is trained.

Figure 1  one 2-component PCA fitted on the L2-normalised 29,571-word GloVe bank
          ONLY, then applied unchanged to GloVe and to each seed's normalised
          s_hat.  Never fitted jointly, never fitted per seed.

Figure 2  for every word: cosine to its OWN GloVe vector versus cosine to the
          best INCORRECT GloVe vector, with rank / top-1 / top-5 / margin.
          This is target semantic identification, and is deliberately not
          c_LTM (which is max similarity to any bank entry).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

EXTRACT = os.path.join(ROOT, "outputs/ventral_semantic_93a577f")
REPORT = os.path.join(ROOT, "reports/ventral_semantic_93a577f")
FIGS = os.path.join(REPORT, "figures")
TABLES = os.path.join(REPORT, "tables")

COHORT = [(19, 155), (22, 140)]
N_WORDS, DIM = 29571, 300
CHUNK = 1000
N_ANNOT = 20

# Prospective, model-independent annotation rule: 20 words at evenly spaced
# positions in the lexicon's frequency-rank ordering (rank 1 = most frequent).
# Fixed by construction, uses no RNG, and involves no model output whatsoever.
ANNOT_RULE = ("20 words at evenly spaced positions in the frequency-rank "
              "ordering of the 29,571-word training lexicon: "
              "index = round(linspace(0, 29570, 20)), 0-based, rank order as "
              "shipped in data/lexicon_en_glove_covered.tsv. Deterministic, no "
              "RNG, defined before any model output was inspected, and "
              "independent of alignment, error or retrieval performance.")

SEED_COLOR = {19: "#1f4e79", 22: "#2e7d32"}
GLOVE_COLOR = "#9aa5b1"
PRED_COLOR = "#c0392b"


def git(*a) -> str:
    return subprocess.run(("git",) + a, cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def sha_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def load() -> Tuple[np.ndarray, Dict[int, np.ndarray], list]:
    bank = np.load(os.path.join(EXTRACT, "glove_bank_normalised.npy"))
    words = open(os.path.join(EXTRACT, "lexicon_words.txt")).read().split("\n")
    words = [w for w in words if w]
    s = {seed: np.load(os.path.join(EXTRACT, f"s_hat_lexicon_seed{seed}.npy"))
         for seed, _ in COHORT}
    assert bank.shape == (N_WORDS, DIM) and len(words) == N_WORDS
    for v in s.values():
        assert v.shape == (N_WORDS, DIM)
    # the bank ships normalised; confirm rather than assume
    assert np.allclose(np.linalg.norm(bank, axis=1), 1.0, atol=1e-5)
    return bank, s, words


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


# ----------------------------------------------------- identification metrics

def identification(q: np.ndarray, bank: np.ndarray) -> pd.DataFrame:
    """Per-word target cosine, best incorrect cosine, rank, top-1/5, margin."""
    tgt = np.empty(N_WORDS, np.float32)
    best_wrong = np.empty(N_WORDS, np.float32)
    rank = np.empty(N_WORDS, np.int32)
    top5 = np.empty(N_WORDS, bool)
    for a in range(0, N_WORDS, CHUNK):
        b = min(a + CHUNK, N_WORDS)
        sims = q[a:b] @ bank.T                      # (chunk, 29571) cosines
        idx = np.arange(a, b)
        t = sims[np.arange(b - a), idx]
        sims_wo = sims.copy()
        sims_wo[np.arange(b - a), idx] = -np.inf    # mask the target itself
        tgt[a:b] = t
        best_wrong[a:b] = sims_wo.max(axis=1)
        # competition rank of the target among all 29,571 entries
        rank[a:b] = 1 + (sims_wo > t[:, None]).sum(axis=1)
        top5[a:b] = rank[a:b] <= 5
    return pd.DataFrame({
        "target_cosine": tgt, "best_wrong_cosine": best_wrong,
        "target_rank": rank, "top1": (rank == 1), "top5": top5,
        "margin": tgt - best_wrong})


# ------------------------------------------------------------------ figure 1

def figure1(bank: np.ndarray, s: Dict[int, np.ndarray], words: list,
            annot_idx: np.ndarray) -> Tuple[Dict[str, str], dict, pd.DataFrame]:
    # PCA fitted on the normalised GloVe bank ONLY
    mu = bank.mean(axis=0)
    Xc = bank - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:2]                                   # (2, 300)
    ev = (S ** 2) / (S ** 2).sum()
    evr = ev[:2]

    def project(x):                                  # same fitted transform
        return (x - mu) @ comps.T

    G = project(bank)
    P = {seed: project(unit(s[seed])) for seed, _ in COHORT}

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 7.2), sharex=True, sharey=True)
    for ax, (seed, epoch) in zip(axes, COHORT):
        ax.scatter(G[:, 0], G[:, 1], s=3, color=GLOVE_COLOR, alpha=.18,
                   edgecolor="none", zorder=1)
        ax.scatter(P[seed][:, 0], P[seed][:, 1], s=3, color=PRED_COLOR,
                   alpha=.14, edgecolor="none", zorder=2)
        for i in annot_idx:
            gx, gy = G[i]
            px, py = P[seed][i]
            ax.annotate("", xy=(px, py), xytext=(gx, gy),
                        arrowprops=dict(arrowstyle="->", color="#444",
                                        lw=.9, alpha=.85, shrinkA=2,
                                        shrinkB=2), zorder=5)
            ax.plot([gx], [gy], "o", ms=6, color="#111", zorder=6)
            ax.plot([px], [py], "s", ms=5.5, color=PRED_COLOR, mec="#4d0000",
                    mew=.8, zorder=6)
            rows.append({"seed": seed, "epoch": epoch,
                         "lexicon_index": int(i), "word": words[i],
                         "glove_pc1": float(gx), "glove_pc2": float(gy),
                         "s_hat_pc1": float(px), "s_hat_pc2": float(py),
                         "pc_displacement": float(np.hypot(px - gx, py - gy))})
        _label(ax, G, annot_idx, words)
        ax.set_title(f"seed {seed} / epoch {epoch}", fontsize=13, pad=8)
        ax.set_xlabel(f"PC1  ({evr[0] * 100:.1f} % of GloVe variance)",
                      fontsize=10.5)
        ax.grid(alpha=.2, lw=.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(f"PC2  ({evr[1] * 100:.1f} % of GloVe variance)",
                       fontsize=10.5)

    handles = [
        Line2D([], [], lw=0, marker="o", ms=6, color=GLOVE_COLOR,
               label="target GloVe, all 29,571 words"),
        Line2D([], [], lw=0, marker="o", ms=6, color=PRED_COLOR,
               label="predicted s_hat (L2-normalised), all 29,571"),
        Line2D([], [], lw=0, marker="o", ms=7, color="#111",
               label="annotated word: GloVe target"),
        Line2D([], [], lw=0, marker="s", ms=6.5, color=PRED_COLOR,
               mec="#4d0000", label="annotated word: predicted s_hat"),
        Line2D([], [], lw=1.1, color="#444", label="target -> prediction")]
    fig.legend(handles=handles, fontsize=9, loc="lower center", ncol=5,
               frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Do phonology-derived ventral representations occupy the same "
                 "semantic space as their GloVe targets?", fontsize=14,
                 y=0.985)
    fig.text(0.5, 0.938,
             "One 2-component PCA fitted on the L2-normalised GloVe bank ONLY, "
             "then applied unchanged to GloVe and to each seed's s_hat "
             f"(PC1 + PC2 = {evr.sum() * 100:.1f} % of GloVe variance)",
             ha="center", fontsize=9.5, color="#444")
    fig.tight_layout(rect=(0, 0.055, 1, 0.928))
    paths = _save(fig, "fig1_glove_s_hat_pca_alignment")
    meta = {"pca_fit_population": "L2-normalised GloVe bank, 29,571 x 300",
            "pca_fitted_jointly_with_s_hat": False,
            "pca_fitted_per_seed": False,
            "n_components": 2,
            "explained_variance_ratio_pc1": float(evr[0]),
            "explained_variance_ratio_pc2": float(evr[1]),
            "explained_variance_ratio_sum": float(evr.sum()),
            "centering": "GloVe bank mean, reused for s_hat",
            "s_hat_normalisation_for_this_figure": "L2 (cosine space)"}
    return paths, meta, pd.DataFrame(rows)


def _label(ax, G, annot_idx, words, radius=0.60):
    """Radial label placement.

    The annotated GloVe points sit in a dense central blob, so labels are pushed
    out onto an empty annulus at their own angle from the blob centroid, with a
    minimum angular separation enforced by sorting.  Each label keeps a thin
    leader line to its point, and because labels are placed at their own angle
    the leaders do not cross.
    """
    pts = G[annot_idx]
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(ang)
    a = ang[order].copy()
    gap = 2 * np.pi / (len(a) + 4)
    for i in range(1, len(a)):                      # enforce a minimum gap
        if a[i] - a[i - 1] < gap:
            a[i] = a[i - 1] + gap
    shift = ((ang[order] - a).mean())               # recentre the fan
    a = a + shift
    for k, j in enumerate(order):
        i = annot_idx[j]
        lx, ly = c[0] + radius * np.cos(a[k]), c[1] + radius * np.sin(a[k])
        ax.annotate("", xy=(G[i, 0], G[i, 1]), xytext=(lx, ly),
                    arrowprops=dict(arrowstyle="-", color="#8a8a8a", lw=.6,
                                    shrinkA=1, shrinkB=4), zorder=6)
        ax.text(lx, ly, words[i], fontsize=8.4, zorder=8, color="#111",
                ha="left" if np.cos(a[k]) >= 0 else "right", va="center",
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                          edgecolor="none", alpha=.82))


# ------------------------------------------------------------------ figure 2

def figure2(per_word: Dict[int, pd.DataFrame]) -> Tuple[Dict[str, str], dict]:
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.6), sharex=True,
                             sharey=True)
    stats = {}
    for ax, (seed, epoch) in zip(axes, COHORT):
        d = per_word[seed]
        ok = d["top1"].to_numpy()
        ax.scatter(d.loc[~ok, "best_wrong_cosine"], d.loc[~ok, "target_cosine"],
                   s=2.5, color="#b0b7bf", alpha=.35, edgecolor="none",
                   zorder=2)
        ax.scatter(d.loc[ok, "best_wrong_cosine"], d.loc[ok, "target_cosine"],
                   s=2.5, color=SEED_COLOR[seed], alpha=.45, edgecolor="none",
                   zorder=3)
        lo = float(min(d["best_wrong_cosine"].min(), d["target_cosine"].min()))
        hi = float(max(d["best_wrong_cosine"].max(), d["target_cosine"].max()))
        pad = 0.03 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "-", color="#111",
                lw=1.4, zorder=4)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")

        s = {"top1_accuracy": float(d["top1"].mean()),
             "top5_accuracy": float(d["top5"].mean()),
             "median_target_cosine": float(d["target_cosine"].median()),
             "median_margin": float(d["margin"].median()),
             "median_target_rank": float(d["target_rank"].median()),
             "n_words": int(len(d))}
        stats[seed] = s
        box = (f"top-1 accuracy      {s['top1_accuracy'] * 100:6.2f} %\n"
               f"top-5 accuracy      {s['top5_accuracy'] * 100:6.2f} %\n"
               f"median target cos   {s['median_target_cosine']:6.3f}\n"
               f"median margin       {s['median_margin']:+6.3f}")
        ax.text(0.03, 0.97, box, transform=ax.transAxes, fontsize=9.4,
                va="top", ha="left", family="monospace", linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor=SEED_COLOR[seed], linewidth=1.2))
        ax.text(0.97, 0.06, "above the line:\ntarget is the top-1\nnearest "
                            "GloVe neighbour", transform=ax.transAxes,
                fontsize=8.4, ha="right", va="bottom", color="#333",
                linespacing=1.35, style="italic")
        ax.set_title(f"seed {seed} / epoch {epoch}", fontsize=13, pad=8)
        ax.set_xlabel("cosine to the best INCORRECT GloVe vector", fontsize=10.5)
        ax.grid(alpha=.2, lw=.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("cosine to the word's OWN GloVe vector", fontsize=10.5)
    handles = [
        Line2D([], [], lw=0, marker="o", ms=6, color="#1f4e79",
               label="target is top-1 (above the line)"),
        Line2D([], [], lw=0, marker="o", ms=6, color="#b0b7bf",
               label="some other word is nearer (below the line)"),
        Line2D([], [], lw=1.4, color="#111", label="y = x")]
    axes[1].legend(handles=handles, fontsize=8.6, loc="lower right",
                   frameon=True, framealpha=.95, bbox_to_anchor=(1.0, 0.22))
    fig.suptitle("Is s_hat closest to the CORRECT GloVe word, or merely close "
                 "to some lexical vector?", fontsize=14, y=0.985)
    fig.text(0.5, 0.925, "All 29,571 training words. Retrieval is over the full "
                         "GloVe bank; the word's own vector is excluded when "
                         "computing the best incorrect cosine.",
             ha="center", fontsize=9.5, color="#444")
    fig.tight_layout(rect=(0, 0, 1, 0.915))
    return _save(fig, "fig2_target_semantic_identification"), stats


def _save(fig, stem: str) -> Dict[str, str]:
    os.makedirs(FIGS, exist_ok=True)
    out = {}
    for ext in ("png", "pdf", "svg"):
        p = os.path.join(FIGS, f"{stem}.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        out[ext] = p
    plt.close(fig)
    return out


# ---------------------------------------------------------------------- main

def main() -> int:
    bank, s, words = load()
    os.makedirs(TABLES, exist_ok=True)

    annot_idx = np.unique(np.round(
        np.linspace(0, N_WORDS - 1, N_ANNOT)).astype(int))
    assert len(annot_idx) == N_ANNOT

    per_word = {}
    frames = []
    for seed, epoch in COHORT:
        d = identification(unit(s[seed]), bank)
        d.insert(0, "word", words)
        d.insert(1, "lexicon_index", np.arange(N_WORDS))
        d.insert(2, "seed", seed)
        d.insert(3, "epoch", epoch)
        per_word[seed] = d
        frames.append(d)
    allw = pd.concat(frames, ignore_index=True)
    allw.to_csv(os.path.join(TABLES, "per_word_identification.tsv"), sep="\t",
                index=False)

    p1, pca_meta, paired = figure1(bank, s, words, annot_idx)
    paired.to_csv(os.path.join(TABLES, "pca_annotated_pairs.tsv"), sep="\t",
                  index=False)
    p2, stats = figure2(per_word)

    meta = {
        "phase": "Phase B - ventral/semantic frozen-checkpoint figures",
        "cohort": [{"seed": s_, "epoch": e,
                    "rule": "X=5 stable zero, first checkpoint of the run"}
                   for s_, e in COHORT],
        "n_words": N_WORDS, "semantic_dim": DIM,
        "glove_normalisation": "L2 (bank is normalised at set_semantic_bank)",
        "s_hat_used": "RAW s_hat from to_semantic; L2-normalised only for "
                      "cosine-space computation and for figure 1",
        "identification_metric": (
            "cosine(s_hat, own GloVe) vs max cosine to any OTHER GloVe vector; "
            "competition rank over all 29,571 entries. This is NOT c_LTM: "
            "c_LTM is max similarity to ANY bank item including the target."),
        "annotation_selection_rule": ANNOT_RULE,
        "pca": pca_meta,
        "results": {str(k): v for k, v in stats.items()},
        "inputs": {
            "glove_bank_normalised.npy":
                sha_file(os.path.join(EXTRACT, "glove_bank_normalised.npy")),
            **{f"s_hat_lexicon_seed{s_}.npy":
               sha_file(os.path.join(EXTRACT, f"s_hat_lexicon_seed{s_}.npy"))
               for s_, _ in COHORT}},
        "model_loaded": False, "inference_run": False,
        "training_performed": False,
        "figures": {"figure1": p1, "figure2": p2},
        "repository_head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    }
    with open(os.path.join(REPORT, "analysis_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    for seed, _ in COHORT:
        st = stats[seed]
        print(f"seed {seed}: top1={st['top1_accuracy']*100:.2f}%  "
              f"top5={st['top5_accuracy']*100:.2f}%  "
              f"med target cos={st['median_target_cosine']:.3f}  "
              f"med margin={st['median_margin']:+.3f}  "
              f"med rank={st['median_target_rank']:.0f}")
    print(f"PCA explained variance: PC1={pca_meta['explained_variance_ratio_pc1']*100:.2f}% "
          f"PC2={pca_meta['explained_variance_ratio_pc2']*100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
