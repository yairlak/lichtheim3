"""Phenomenon 2: Lexical neighborhood density effects.

We compare three item types through the *full* gated model:

    sparse words  : real lexemes with few one-edit phonological neighbors
    dense  words  : real lexemes living in crowded neighborhoods
    non-words     : random phoneme strings (no lexical entry)

and read off, for each group:
    * mean gate (how much the model leans on the ventral lexicon)
    * lexical confidence (max cosine to a known lexeme)
    * repetition error rate

What we expect, and what makes this a *human* signature: under the
error-suppression gate, non-words get low lexical confidence -> low gate -> the
buffer must carry them -> higher error than real words; dense neighborhoods raise
confusability among lexical competitors.
"""
from __future__ import annotations

import os
import random
from typing import Dict, List

import numpy as np
import torch

from evaluate.hooks import make_batch, route_predictions, per_position_correct
from utils.plotting import grouped_bars


def _word_error(model, vocab, forms, device) -> Dict[str, float]:
    batch = make_batch(forms, vocab, device)
    res = model.forward(batch["enc_in"], batch["enc_mask"], batch["dec_in"],
                        collect=True)
    preds = res["logits"].argmax(-1)
    corr = per_position_correct(preds, batch["dec_tgt"], vocab.pad_id)
    err = 1.0 - float(np.nanmean(corr.cpu().numpy()))
    gate = float(res["gate"].mean())
    conf = float(res["field_confidence"].mean()) if "field_confidence" in res \
        else float("nan")
    dens = float(res["field_density"].mean()) if "field_density" in res \
        else float("nan")
    return {"error": err, "gate": gate, "confidence": conf, "density": dens}


def _random_forms(vocab, n, lengths, rng):
    phon_ids = [vocab.stoi[s] for s in vocab.itos[3:]]
    return [[rng.choice(phon_ids) for _ in range(rng.choice(lengths))]
            for _ in range(n)]


@torch.no_grad()
def run(model, vocab, lexicon, cfg, out_dir: str, max_per_group=200) -> dict:
    model.eval()
    device = cfg.train.device
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(0)

    density = lexicon.neighborhood_density()
    entries = lexicon.entries
    dvals = np.array([density[i] for i in range(len(entries))])
    thresh = np.median(dvals)
    sparse = [entries[i].phonemes for i in range(len(entries)) if dvals[i] <= thresh]
    dense = [entries[i].phonemes for i in range(len(entries)) if dvals[i] > thresh]
    rng.shuffle(sparse)
    rng.shuffle(dense)
    sparse, dense = sparse[:max_per_group], dense[:max_per_group]

    lengths = sorted({len(p) for p in sparse + dense}) or [3, 4, 5]
    nonwords = _random_forms(vocab, max_per_group, lengths, rng)

    groups = {"sparse word": sparse, "dense word": dense, "non-word": nonwords}
    stats = {name: _word_error(model, vocab, forms, device)
             for name, forms in groups.items() if forms}

    labels = list(stats.keys())
    grouped_bars(
        labels,
        {"error rate": [stats[l]["error"] for l in labels],
         "mean gate (LTM use)": [stats[l]["gate"] for l in labels],
         "lexical confidence": [stats[l]["confidence"] for l in labels]},
        title="Neighborhood-density effects (full gated model)",
        ylabel="value", path=os.path.join(out_dir, "neighborhood_effects.png"))

    for l in labels:
        s = stats[l]
        print(f"[neighborhood] {l:12s} err={s['error']:.3f} gate={s['gate']:.3f} "
              f"conf={s['confidence']:.3f} dens={s['density']:.2f}")
    return stats
