"""Phenomenon 1: Primacy & recency (U-shaped serial-position curve).

We feed the *dorsal* buffer long random phoneme lists and measure, for each
serial position, how often it reconstructs the correct phoneme — averaged over
many lists and many noisy retrieval passes. The buffer's primacy write-gain and
recency leak should produce the classic U: high at the start, high at the end,
sagging in the middle.

We isolate `route="wm"` so the lexicon can't rescue the items (random lists are
non-words anyway). `collect=True` keeps interference noise on at eval time, which
is what generates the errors.
"""
from __future__ import annotations

import os
import random
from typing import List

import numpy as np
import torch

from evaluate.hooks import make_batch, route_predictions, per_position_correct
from utils.plotting import line, multiline


def _random_forms(vocab, n: int, length: int, rng: random.Random) -> List[List[int]]:
    phon_ids = [vocab.stoi[s] for s in vocab.itos[3:]]  # skip specials
    return [[rng.choice(phon_ids) for _ in range(length)] for _ in range(n)]


@torch.no_grad()
def serial_position_curve(model, vocab, device, length: int, n_words: int,
                          n_trials: int, seed: int = 0) -> np.ndarray:
    rng = random.Random(seed)
    forms = _random_forms(vocab, n_words, length, rng)
    batch = make_batch(forms, vocab, device)
    acc = np.zeros(length, dtype=np.float64)
    for _ in range(n_trials):  # average over interference noise draws
        preds, _ = route_predictions(model, batch, route="wm", collect=True)
        corr = per_position_correct(preds, batch["dec_tgt"], vocab.pad_id)
        corr = corr[:, :length].cpu().numpy()           # ignore EOS column
        acc += np.nanmean(corr, axis=0)
    return acc / n_trials


def run(model, vocab, cfg, out_dir: str, lengths=(6, 8, 10), n_words=160,
        n_trials=12) -> dict:
    model.eval()
    device = cfg.train.device
    os.makedirs(out_dir, exist_ok=True)

    # main curve at a single representative length
    main_len = lengths[len(lengths) // 2]
    curve = serial_position_curve(model, vocab, device, main_len, n_words, n_trials)
    line(list(range(1, main_len + 1)), curve.tolist(),
         title=f"WM serial-position curve (length {main_len})",
         xlabel="serial position", ylabel="reconstruction accuracy",
         path=os.path.join(out_dir, "primacy_recency_curve.png"))

    # overlay several lengths (normalized position) to show the U is general
    series = {}
    for L in lengths:
        c = serial_position_curve(model, vocab, device, L, n_words, n_trials)
        series[f"len {L}"] = c.tolist() + [np.nan] * (max(lengths) - L)
    multiline(list(range(1, max(lengths) + 1)), series,
              title="WM serial-position curves across list lengths",
              xlabel="serial position", ylabel="reconstruction accuracy",
              path=os.path.join(out_dir, "primacy_recency_by_length.png"))

    primacy = float(curve[0])
    recency = float(curve[-1])
    middle = float(np.mean(curve[len(curve) // 3: 2 * len(curve) // 3]))
    summary = {"primacy": primacy, "recency": recency, "middle": middle,
               "u_shape_gap": min(primacy, recency) - middle}
    print(f"[primacy/recency] primacy={primacy:.3f} recency={recency:.3f} "
          f"middle={middle:.3f} U-gap={summary['u_shape_gap']:+.3f}")
    return summary
