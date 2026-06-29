"""Lesion / ablation study for the PyTorch model, after Ueno et al. (2011),
"Lichtheim 2", Neuron 72:385-396.

Damage is simulated the Ueno way: remove a proportion of a pathway's units
(its "incoming links") and add Gaussian noise over its output, titrating
severity and averaging over random "patients" (seeds), mean +/- SE. We lesion
the dorsal (WM) or ventral (LTM) route by attaching a forward hook to the
corresponding submodule that perturbs its pre-motor output -- so the core model
code is untouched. The gate (which reads lexical confidence, not route health)
is left intact, exactly as in the NumPy demo.

Expected, as in Ueno: a DORSAL lesion abolishes NONWORD (unseen) repetition
while sparing real words (conduction aphasia); a VENTRAL lesion does the reverse.

Run via run_all.py, or standalone after training a model.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Dict, List

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate.hooks import make_batch, route_predictions, per_position_correct

SEVERITIES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
N_PATIENTS = 10
DORSAL_NOISE_MAX = 4.0
VENTRAL_NOISE_MAX = 4.0


@contextmanager
def lesion(model, site: str, frac: float, noise: float, generator=None):
    """Attach a forward hook that perturbs a route's pre-motor output."""
    submodule = model.wm if site == "dorsal" else model.ltm

    def hook(_m, _inp, out):
        if not isinstance(out, dict) or "premotor" not in out:
            return out
        pm = out["premotor"]
        if frac > 0:
            keep = (torch.rand(pm.shape[-1], device=pm.device,
                               generator=generator) >= frac).float()
            pm = pm * keep
        if noise > 0:
            pm = pm + torch.randn(pm.shape, device=pm.device,
                                  generator=generator) * noise
        out = dict(out)
        out["premotor"] = pm
        return out

    h = submodule.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


def _word_acc(model, vocab, forms, device) -> float:
    if not forms:
        return float("nan")
    batch = make_batch(forms, vocab, device)
    preds, _ = route_predictions(model, batch, route="full", collect=True)
    corr = per_position_correct(preds, batch["dec_tgt"], vocab.pad_id).cpu().numpy()
    filled = np.where(np.isnan(corr), 1.0, corr)
    return float((filled == 1.0).all(axis=1).mean())


def _se(v):
    v = np.asarray(v, float)
    return v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


@torch.no_grad()
def run(model, vocab, lexicon, cfg, out_dir: str) -> dict:
    model.eval()
    device = cfg.train.device
    os.makedirs(out_dir, exist_ok=True)
    train_e, val_e = lexicon.split(cfg.data.val_fraction, cfg.data.seed)
    words = [e.phonemes for e in train_e][:300]        # trained "words"
    nonwords = [e.phonemes for e in val_e][:300]       # novel "nonwords"

    sweeps = {}
    for site in ("dorsal", "ventral"):
        noise_max = DORSAL_NOISE_MAX if site == "dorsal" else VENTRAL_NOISE_MAX
        res = {}
        for s in SEVERITIES:
            w, nw = [], []
            n = 1 if s == 0 else N_PATIENTS
            for patient in range(n):
                gen = torch.Generator(device=device).manual_seed(2000 + patient)
                with lesion(model, site, s, noise_max * s, generator=gen):
                    w.append(_word_acc(model, vocab, words, device))
                    nw.append(_word_acc(model, vocab, nonwords, device))
            res[s] = {"word": (float(np.mean(w)), _se(w)),
                      "nonword": (float(np.mean(nw)), _se(nw))}
        sweeps[site] = res
        s = 0.8
        print(f"[ablation] {site:7s} @s=0.8 word={res[s]['word'][0]:.3f} "
              f"nonword={res[s]['nonword'][0]:.3f}")

    _plot_severity(sweeps, os.path.join(out_dir, "ablation_severity.png"))
    _plot_dissociation(sweeps, 0.8, os.path.join(out_dir, "ablation_dissociation.png"))
    return sweeps


def _plot_severity(sweeps, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    titles = {"dorsal": "Dorsal (WM) lesion", "ventral": "Ventral (LTM) lesion"}
    xs = [int(s * 100) for s in SEVERITIES]
    for ax, site in zip(axes, ("dorsal", "ventral")):
        r = sweeps[site]
        for cond, st in (("word", dict(marker="o", color="tab:blue")),
                         ("nonword", dict(marker="s", color="tab:red"))):
            ys = [r[s][cond][0] for s in SEVERITIES]
            es = [r[s][cond][1] for s in SEVERITIES]
            ax.errorbar(xs, ys, yerr=es, capsize=3, label=cond, **st)
        ax.set(title=titles[site], xlabel="lesion severity (% units removed + noise)",
               ylim=(0, 1.03))
        ax.grid(alpha=0.3); ax.legend()
    axes[0].set_ylabel("repetition accuracy")
    fig.suptitle("Lesion severity curves (mean ± SE over patients)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _plot_dissociation(sweeps, s, path):
    fig, ax = plt.subplots(figsize=(6.2, 4))
    sites = ["dorsal", "ventral"]
    x = np.arange(len(sites)); w = 0.38
    word = [sweeps[si][s]["word"][0] for si in sites]
    nonword = [sweeps[si][s]["nonword"][0] for si in sites]
    ax.bar(x - w/2, word, w, label="word (trained)", color="tab:blue")
    ax.bar(x + w/2, nonword, w, label="nonword (novel)", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(["Dorsal lesion\n(conduction-aphasia)",
                        "Ventral lesion\n(lexical-semantic)"])
    ax.set_ylabel("repetition accuracy"); ax.set_ylim(0, 1.03)
    ax.set_title(f"Double dissociation under lesion (severity {int(s*100)}%)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
