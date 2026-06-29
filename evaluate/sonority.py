"""Phenomenon 3: Sonority-graded errors.

When the model misfires, does it substitute a *phonetically similar* phoneme?
We collect substitution errors from the noisy dorsal buffer, build a phoneme
confusion matrix, and correlate each confusion's probability with the
articulatory distance between the two phonemes (from `phonemes.py`'s feature
space). A negative correlation — closer phonemes confused more often, errors
respecting the sonority hierarchy — is the human signature.

Outputs:
    * confusion-matrix heatmap (phonemes sorted by sonority)
    * scatter of phonetic distance vs confusion probability, with correlation
"""
from __future__ import annotations

import os
import random
from typing import List

import numpy as np
import torch

from evaluate.hooks import make_batch, route_predictions
from utils.plotting import heatmap


def _random_forms(vocab, n, length, rng):
    phon_ids = [vocab.stoi[s] for s in vocab.itos[3:]]
    return [[rng.choice(phon_ids) for _ in range(length)] for _ in range(n)]


@torch.no_grad()
def run(model, vocab, cfg, out_dir: str, route="wm", n_words=400, length=6,
        n_trials=20, seed=0) -> dict:
    model.eval()
    device = cfg.train.device
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)

    phon_syms = vocab.itos[3:]                      # drop specials
    phon_ids = [vocab.stoi[s] for s in phon_syms]
    id2row = {pid: r for r, pid in enumerate(phon_ids)}
    P = len(phon_ids)
    conf = np.zeros((P, P), dtype=np.float64)       # target x predicted

    forms = _random_forms(vocab, n_words, length, rng)
    batch = make_batch(forms, vocab, device)
    tgt = batch["dec_tgt"]
    for _ in range(n_trials):
        preds, _ = route_predictions(model, batch, route=route, collect=True)
        t = tgt.cpu().numpy().ravel()
        p = preds.cpu().numpy().ravel()
        for ti, pi in zip(t, p):
            if ti in id2row and pi in id2row:
                conf[id2row[ti], id2row[pi]] += 1

    # row-normalize to confusion probabilities
    row_sums = conf.sum(axis=1, keepdims=True)
    conf_prob = np.divide(conf, row_sums, out=np.zeros_like(conf),
                          where=row_sums > 0)

    # order phonemes by sonority for a readable heatmap
    son = np.array([vocab.sonority[pid] for pid in phon_ids])
    order = np.argsort(son)
    heatmap(conf_prob[order][:, order],
            title="Phoneme confusion matrix (sorted by sonority)",
            xlabel="predicted", ylabel="target",
            path=os.path.join(out_dir, "sonority_confusion.png"),
            xticklabels=[phon_syms[i] for i in order],
            yticklabels=[phon_syms[i] for i in order])

    # correlate off-diagonal confusion prob with phonetic distance
    dists, probs = [], []
    for a in range(P):
        for b in range(P):
            if a == b:
                continue
            probs.append(conf_prob[a, b])
            dists.append(vocab.phonetic_distance(phon_ids[a], phon_ids[b]))
    dists = np.array(dists)
    probs = np.array(probs)
    # Pearson correlation (guard against zero variance)
    if probs.std() > 0 and dists.std() > 0:
        r = float(np.corrcoef(dists, probs)[0, 1])
    else:
        r = float("nan")

    _scatter(dists, probs, r, os.path.join(out_dir, "sonority_distance_scatter.png"))
    print(f"[sonority] corr(phonetic distance, confusion prob) = {r:+.3f} "
          f"(expect negative: similar phonemes confused more)")
    return {"distance_confusion_corr": r}


def _scatter(dists, probs, r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(dists, probs, s=8, alpha=0.4)
    if np.isfinite(r) and dists.std() > 0:
        m, c = np.polyfit(dists, probs, 1)
        xs = np.linspace(dists.min(), dists.max(), 50)
        ax.plot(xs, m * xs + c, color="crimson")
    ax.set(title=f"Errors track phonetic distance (r = {r:+.3f})",
           xlabel="articulatory distance", ylabel="confusion probability")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
