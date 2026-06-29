"""Unseen-word generalization (cf. arXiv:2506.13450).

After training, probe repetition on the held-out (never-trained) words and
compare against trained words, for each route in isolation. The dual-route
prediction is a dissociation:

  * Ventral (LTM)  : strong on trained words, weaker on novel words
                     (lexical knowledge does not transfer to non-words).
  * Dorsal (WM)    : similar on trained and novel words (content-agnostic copy),
                     but capacity-limited, so it falls off with length.
  * Full (gated)   : should match the better route item-by-item, beating both
                     single routes on the novel set when the gate routes
                     unfamiliar items to the buffer.

Outputs a grouped-bar figure (seen vs unseen, per route) and a per-length
breakdown on the unseen set, plus a JSON-able summary dict.
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import torch

from evaluate.hooks import make_batch, route_predictions, per_position_correct
from utils.plotting import grouped_bars, multiline


def _acc(model, vocab, forms: List[List[int]], device, route: str,
         collect: bool = False) -> Dict[str, float]:
    if not forms:
        return {"phoneme": float("nan"), "word": float("nan")}
    batch = make_batch(forms, vocab, device)
    preds, _ = route_predictions(model, batch, route=route, collect=collect)
    corr = per_position_correct(preds, batch["dec_tgt"], vocab.pad_id).cpu().numpy()
    phoneme = float(np.nanmean(corr))
    filled = np.where(np.isnan(corr), 1.0, corr)        # ignore padding
    word = float((filled == 1.0).all(axis=1).mean())
    return {"phoneme": phoneme, "word": word}


@torch.no_grad()
def run(model, vocab, lexicon, cfg, out_dir: str) -> dict:
    model.eval()
    device = cfg.train.device
    os.makedirs(out_dir, exist_ok=True)

    # identical split to training (deterministic): val == unseen words
    train_entries, val_entries = lexicon.split(cfg.data.val_fraction, cfg.data.seed)
    seen = [e.phonemes for e in train_entries][:400]
    unseen = [e.phonemes for e in val_entries][:400]

    routes = ["ltm", "wm", "full"]
    nice = {"ltm": "Ventral", "wm": "Dorsal", "full": "Gated"}
    report = {"seen": {}, "unseen": {}}
    for r in routes:
        report["seen"][r] = _acc(model, vocab, seen, device, r, collect=(r != "ltm"))
        report["unseen"][r] = _acc(model, vocab, unseen, device, r, collect=(r != "ltm"))
        print(f"[generalization] {nice[r]:8s} "
              f"seen word={report['seen'][r]['word']:.3f} "
              f"unseen word={report['unseen'][r]['word']:.3f} "
              f"(gap={report['seen'][r]['word']-report['unseen'][r]['word']:+.3f})")

    grouped_bars(
        [nice[r] for r in routes],
        {"seen (trained)": [report["seen"][r]["word"] for r in routes],
         "unseen (novel)": [report["unseen"][r]["word"] for r in routes]},
        title="Generalization to unseen words, by route",
        ylabel="whole-word repetition accuracy",
        path=os.path.join(out_dir, "generalization.png"))

    # per-length on the unseen set
    by_len: Dict[int, List[List[int]]] = {}
    for p in unseen:
        by_len.setdefault(len(p), []).append(p)
    lens = sorted(L for L, v in by_len.items() if len(v) >= 3)
    series = {nice[r]: [] for r in routes}
    for L in lens:
        for r in routes:
            series[nice[r]].append(
                _acc(model, vocab, by_len[L], device, r, collect=(r != "ltm"))["phoneme"])
    if lens:
        multiline(lens, series,
                  title="Unseen-word repetition vs length",
                  xlabel="word length (phonemes)",
                  ylabel="phoneme accuracy (unseen)",
                  path=os.path.join(out_dir, "unseen_by_length.png"))
    return report
