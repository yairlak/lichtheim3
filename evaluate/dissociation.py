"""Phenomenon 4: The double dissociation (Frequency x Length).

The headline result. We cross two factors on real lexicon items:

    frequency : high vs low (by training-frequency rank)
    length    : short vs long (by phoneme count)

and measure repetition accuracy through each route in isolation:

    Ventral (LTM)  : should be sensitive to FREQUENCY, invariant to LENGTH
    Dorsal (WM)    : should be invariant to FREQUENCY, sensitive to LENGTH

We quantify each route's "frequency effect" and "length effect" (accuracy gaps)
and draw the 2x2 interaction plot. A clean dissociation = the two routes' effect
profiles are mirror images.
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate.hooks import make_batch, route_predictions, per_position_correct


def _route_acc(model, vocab, forms, device, route, n_trials=1) -> float:
    if not forms:
        return float("nan")
    batch = make_batch(forms, vocab, device)
    accs = []
    collect = (route == "wm")          # noise matters only for the buffer
    for _ in range(n_trials if collect else 1):
        preds, _ = route_predictions(model, batch, route=route, collect=collect)
        corr = per_position_correct(preds, batch["dec_tgt"], vocab.pad_id)
        accs.append(float(np.nanmean(corr.cpu().numpy())))
    return float(np.mean(accs))


@torch.no_grad()
def run(model, vocab, lexicon, cfg, out_dir: str, n_trials=12) -> dict:
    model.eval()
    device = cfg.train.device
    os.makedirs(out_dir, exist_ok=True)

    entries = lexicon.entries
    ranks = np.array([e.rank for e in entries])
    lens = np.array([e.length for e in entries])
    rank_thr = np.median(ranks)        # low rank == high frequency
    len_thr = np.median(lens)

    def pick(freq_high: bool, long: bool) -> List[List[int]]:
        out = []
        for e in entries:
            fh = e.rank <= rank_thr
            lg = e.length > len_thr
            if fh == freq_high and lg == long:
                out.append(e.phonemes)
        return out[:300]

    cells = {
        ("high", "short"): pick(True, False),
        ("high", "long"): pick(True, True),
        ("low", "short"): pick(False, False),
        ("low", "long"): pick(False, True),
    }

    results = {"ltm": {}, "wm": {}}
    for route in ("ltm", "wm"):
        for key, forms in cells.items():
            results[route][key] = _route_acc(model, vocab, forms, device, route,
                                             n_trials=n_trials)

    _plot(results, os.path.join(out_dir, "double_dissociation.png"))

    # effect sizes: average accuracy gaps
    def gap(route, factor):
        r = results[route]
        if factor == "freq":   # high - low
            hi = np.nanmean([r[("high", l)] for l in ("short", "long")])
            lo = np.nanmean([r[("low", l)] for l in ("short", "long")])
            return float(hi - lo)
        else:                  # short - long (length cost)
            sh = np.nanmean([r[(f, "short")] for f in ("high", "low")])
            lo = np.nanmean([r[(f, "long")] for f in ("high", "low")])
            return float(sh - lo)

    summary = {
        "ventral_frequency_effect": gap("ltm", "freq"),
        "ventral_length_effect": gap("ltm", "len"),
        "dorsal_frequency_effect": gap("wm", "freq"),
        "dorsal_length_effect": gap("wm", "len"),
    }
    print("[dissociation] VENTRAL freq-effect={ventral_frequency_effect:+.3f} "
          "len-effect={ventral_length_effect:+.3f}".format(**summary))
    print("[dissociation] DORSAL  freq-effect={dorsal_frequency_effect:+.3f} "
          "len-effect={dorsal_length_effect:+.3f}".format(**summary))
    print("[dissociation] expect: ventral big freq / small len ; "
          "dorsal small freq / big len")
    return summary


def _plot(results: Dict, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    titles = {"ltm": "Ventral (LTM) route", "wm": "Dorsal (WM) route"}
    for ax, route in zip(axes, ("ltm", "wm")):
        r = results[route]
        for freq, style in (("high", "-o"), ("low", "--s")):
            ys = [r[(freq, "short")], r[(freq, "long")]]
            ax.plot(["short", "long"], ys, style, label=f"{freq} freq")
        ax.set(title=titles[route], xlabel="word length", ylim=(0, 1.02))
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("repetition accuracy")
    fig.suptitle("Double dissociation: Frequency x Length")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
