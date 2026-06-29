"""Shared data + training helpers for the NumPy demo and the ablation study.

Data now comes from the bundled **realistic English lexicon** (`data/lexicon_en.tsv`,
30k frequency-ranked real words + ARPABET). For tractability the NumPy twin
lexicalizes a frequent *core* of the vocabulary (the most frequent `n_core`
words) and trains on it with **log-frequency** weighting; the full 30k lexicon is
what the PyTorch model uses. Three test pools are produced:

    train   : the n_core most frequent real words (the model's "lexicon")
    unseen  : the next most frequent real words, never trained (held-out words)
    nonword : random ARPABET strings (novel pseudowords), length-matched
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DataConfig
from data.phonemes import build_vocab
from data.lexicon import build_lexicon, logfreq_weights
from numpy_demo.dual_route_numpy import DualRouteNumpy, Adam, encode_dataset


class Entry:
    __slots__ = ("phonemes",)

    def __init__(self, phonemes):
        self.phonemes = phonemes


def _random_nonwords(vocab, n, lengths, rng, real_forms):
    phon_ids = [vocab.stoi[s] for s in vocab.itos[3:]]   # skip specials
    out, seen = [], set(real_forms)
    while len(out) < n:
        L = int(rng.choice(lengths))
        form = tuple(int(rng.choice(phon_ids)) for _ in range(L))
        if form in seen:
            continue
        seen.add(form)
        out.append(Entry(list(form)))
    return out


def make_data(n_core=2500, n_unseen=400, n_nonword=400, L=9, seed=0):
    rng = np.random.default_rng(seed)
    vocab = build_vocab()
    cfg = DataConfig(use_real=True, max_words=n_core + n_unseen, max_phonemes=L,
                     min_phonemes=2)
    lex = build_lexicon(cfg, vocab)
    entries = lex.entries                      # most-frequent first
    train = entries[:n_core]
    unseen = entries[n_core:n_core + n_unseen]

    real_forms = {tuple(e.phonemes) for e in entries}
    train_lengths = [len(e.phonemes) for e in train]
    nonword = _random_nonwords(vocab, n_nonword, train_lengths, rng, real_forms)

    Xtr, Ttr, Mtr = encode_dataset(train, vocab, L)
    Xun, Tun, Mun = encode_dataset(unseen, vocab, L)
    Xnw, Tnw, Mnw = encode_dataset(nonword, vocab, L)

    w_train = logfreq_weights([e.rank for e in train])
    w_train = (w_train / w_train.mean()).astype(np.float32)   # mean 1

    return dict(
        vocab=vocab, L=L, source=lex.source,
        train=train, unseen=unseen, nonword=nonword,
        Xtr=Xtr, Ttr=Ttr, Mtr=Mtr, len_tr=Mtr.sum(1), w_train=w_train,
        Xun=Xun, Tun=Tun, Mun=Mun, len_un=Mun.sum(1),
        Xnw=Xnw, Tnw=Tnw, Mnw=Mnw, len_nw=Mnw.sum(1),
    )


def build_model(data, h_sem=256, seed=0):
    # Lexical bottleneck: the ventral route memorizes (frequency-weighted) real
    # words but is at floor on novel pseudowords. Sharp familiarity gate keeps
    # routing decisive so a lesion in one pathway does not leak into the other's
    # items.
    model = DualRouteNumpy(L=data["L"], V=data["vocab"].size, h_sem=h_sem,
                           beta=6.0, leak=0.9, k_slots=4,
                           gate_alpha=40.0, gate_thr=0.90, seed=seed)
    model.set_familiarity_bank(data["Xtr"])
    return model


def train_model(data, epochs=400, h_sem=256, lr=3e-3, seed=0, verbose=False,
                history=None):
    """Train with log-frequency weighting. If `history` (list) is given, append
    per-epoch train/held-out losses to it."""
    model = build_model(data, h_sem=h_sem, seed=seed)
    opt = Adam(model.get_params(), lr=lr)
    for ep in range(1, epochs + 1):
        loss, grads = model.loss_and_grad(data["Xtr"], data["Ttr"], data["Mtr"],
                                          sample_w=data["w_train"])
        model.set_params(opt.step(model.get_params(), grads))
        if history is not None and (ep % 5 == 0 or ep == 1):
            vloss, _ = model.loss_and_grad(data["Xun"], data["Tun"], data["Mun"])
            history.append({"epoch": ep, "train": float(loss), "unseen": float(vloss)})
        if verbose and (ep % 100 == 0 or ep == 1):
            print(f"  [ep {ep}] loss={loss:.4f}")
    return model
