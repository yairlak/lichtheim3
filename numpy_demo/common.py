"""Shared data + training helpers for the NumPy demo and the ablation study.

Data comes from the bundled realistic English lexicon (`data/lexicon_en.tsv`,
30k frequency-ranked real words + ARPABET). The NumPy twin lexicalizes a frequent
*core* and trains by **presenting words in proportion to (log) frequency** — i.e.
frequency enters through exposure (sampling), as it does for a learner, rather
than through a re-weighted loss. Three test pools are produced:

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
    """Pronounceable pseudowords with (C)V(C) syllable structure -- like the
    nonwords used in human nonword-repetition tasks (not random phoneme salad),
    so the phonotactically-trained dorsal route can actually articulate them."""
    cons = [vocab.stoi[s] for s in vocab.itos[3:] if vocab.sonority[vocab.stoi[s]] < 0.9]
    vow = [vocab.stoi[s] for s in vocab.itos[3:] if vocab.sonority[vocab.stoi[s]] >= 0.95]
    Lmax = int(max(lengths))
    out, seen = [], set(real_forms)
    while len(out) < n:
        form = []
        for _ in range(int(rng.integers(1, 4))):       # 1-3 syllables
            if rng.random() < 0.85:
                form.append(int(rng.choice(cons)))
            form.append(int(rng.choice(vow)))
            if rng.random() < 0.4:
                form.append(int(rng.choice(cons)))
        if not (2 <= len(form) <= Lmax) or tuple(form) in seen:
            continue
        seen.add(tuple(form))
        out.append(Entry(form))
    return out


def make_data(n_core=2000, n_unseen=400, n_nonword=400, L=9, seed=0):
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

    # a frequency-flat pool of pronounceable forms to train the dorsal route's
    # general serial-recall (disjoint rng from the test nonwords)
    pool_rng = np.random.default_rng(seed + 7)
    pool = _random_nonwords(vocab, 3000, train_lengths, pool_rng, real_forms)

    Xtr, Ttr, Mtr = encode_dataset(train, vocab, L)
    Xun, Tun, Mun = encode_dataset(unseen, vocab, L)
    Xnw, Tnw, Mnw = encode_dataset(nonword, vocab, L)
    Xrp, Trp, Mrp = encode_dataset(pool, vocab, L)

    # presentation probability ~ log frequency (Zipfian freq -> log-compressed)
    pres = logfreq_weights([e.rank for e in train])
    pres = (pres / pres.sum()).astype(np.float64)

    return dict(
        vocab=vocab, L=L, source=lex.source,
        train=train, unseen=unseen, nonword=nonword,
        Xtr=Xtr, Ttr=Ttr, Mtr=Mtr, len_tr=Mtr.sum(1), present_prob=pres,
        Xun=Xun, Tun=Tun, Mun=Mun, len_un=Mun.sum(1),
        Xnw=Xnw, Tnw=Tnw, Mnw=Mnw, len_nw=Mnw.sum(1),
        Xrp=Xrp, Trp=Trp, Mrp=Mrp,
    )


def build_model(data, h_sem=32, d_wm=96, emb=24, seed=0):
    # h_sem is small on purpose: the ventral route is a tight lexical bottleneck
    # that memorizes trained words but fails on novel forms (so the dorsal
    # recurrent route is the sole route for nonwords/unseen words).
    vocab = data["vocab"]
    model = DualRouteNumpy(L=data["L"], V=vocab.size,
                           bos_id=vocab.bos_id, pad_id=vocab.pad_id,
                           emb=emb, d_wm=d_wm, h_sem=h_sem,
                           gate_alpha=40.0, gate_thr=0.90, seed=seed)
    model.set_familiarity_bank(data["Xtr"])
    return model


def train_model(data, steps=2500, batch=128, lr=2e-3, h_sem=32, d_wm=96, emb=24,
                seed=0, verbose=False, history=None):
    """Minibatch SGD, presenting words in proportion to log frequency."""
    model = build_model(data, h_sem=h_sem, d_wm=d_wm, emb=emb, seed=seed)
    opt = Adam(model.get_params(), lr=lr)
    rng = np.random.default_rng(seed + 1)
    Xtr, Ttr, Mtr = data["Xtr"], data["Ttr"], data["Mtr"]
    Xrp, Trp, Mrp = data["Xrp"], data["Trp"], data["Mrp"]
    p, N, Nrp = data["present_prob"], len(data["train"]), len(data["Xrp"])
    n_eval = min(400, N)
    for step in range(1, steps + 1):
        idx = rng.choice(N, size=batch, p=p)   # present words ~ log-frequency
        loss, grads = model.loss_and_grad(Xtr[idx], Ttr[idx], Mtr[idx])
        # extra dorsal-only training on a frequency-flat pronounceable stream
        ridx = rng.integers(0, Nrp, size=batch)
        _, gd = model.dorsal_only_loss_and_grad(Xrp[ridx], Trp[ridx], Mrp[ridx])
        for k in gd:
            grads[k] = grads[k] + gd[k]
        model.set_params(opt.step(model.get_params(), grads))
        if history is not None and (step % 50 == 0 or step == 1):
            tl, _ = model.loss_and_grad(Xtr[:n_eval], Ttr[:n_eval], Mtr[:n_eval])
            vl, _ = model.loss_and_grad(data["Xun"], data["Tun"], data["Mun"])
            history.append({"step": step, "train": float(tl), "unseen": float(vl)})
        if verbose and (step % 500 == 0 or step == 1):
            print(f"  [step {step:4d}] batch_loss={loss:.4f}")
    return model
