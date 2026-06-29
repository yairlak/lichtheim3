"""PyTorch Dataset + collation + frequency-weighted sampling.

Repetition is framed as seq2seq: the target sequence equals the input sequence.
For each word we emit:

    enc_in   : [p1 ... pT, EOS]              (encoder reads this)
    dec_in   : [BOS, p1 ... pT]              (teacher forcing input)
    dec_tgt  : [p1 ... pT, EOS]              (what the motor layer must output)

One-hot encoding is produced on the fly in the model from `enc_in` ids (the task
spec asks for one-hot phoneme input; we keep ids in the batch and let the model
do the identity lookup, which is equivalent and memory-light).

The **log-frequency-weighted sampler** is the engine of the double dissociation:
frequent words are drawn more often (on a log scale, matching the word-repetition
modelling literature), so the *parametric* ventral route sees more gradient on
them, while the *non-parametric* dorsal buffer cannot benefit.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import random

from .lexicon import Lexicon, LexEntry, logfreq_weights
from .phonemes import Vocab


class RepetitionDataset(Dataset):
    def __init__(self, entries: List[LexEntry], vocab: Vocab, density: Dict[int, int]):
        self.entries = entries
        self.vocab = vocab
        self.density = density  # keyed by the *original* lexicon index

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        e = self.entries[i]
        v = self.vocab
        enc_in = e.phonemes + [v.eos_id]
        dec_in = [v.bos_id] + e.phonemes
        dec_tgt = e.phonemes + [v.eos_id]
        return {
            "enc_in": torch.tensor(enc_in, dtype=torch.long),
            "dec_in": torch.tensor(dec_in, dtype=torch.long),
            "dec_tgt": torch.tensor(dec_tgt, dtype=torch.long),
            "length": len(e.phonemes),
            "semantic": torch.tensor(e.semantic, dtype=torch.float32),
            "freq": float(e.freq),
            "rank": int(e.rank),
            "word": e.word,
        }


def make_collate(pad_id: int):
    def collate(batch):
        B = len(batch)
        max_enc = max(len(b["enc_in"]) for b in batch)
        max_dec = max(len(b["dec_in"]) for b in batch)

        enc_in = torch.full((B, max_enc), pad_id, dtype=torch.long)
        dec_in = torch.full((B, max_dec), pad_id, dtype=torch.long)
        dec_tgt = torch.full((B, max_dec), pad_id, dtype=torch.long)
        enc_mask = torch.zeros((B, max_enc), dtype=torch.bool)

        for k, b in enumerate(batch):
            le, ld = len(b["enc_in"]), len(b["dec_in"])
            enc_in[k, :le] = b["enc_in"]
            enc_mask[k, :le] = True
            dec_in[k, :ld] = b["dec_in"]
            dec_tgt[k, :ld] = b["dec_tgt"]

        return {
            "enc_in": enc_in,
            "enc_mask": enc_mask,
            "dec_in": dec_in,
            "dec_tgt": dec_tgt,
            "lengths": torch.tensor([b["length"] for b in batch], dtype=torch.long),
            "semantic": torch.stack([b["semantic"] for b in batch]),
            "freq": torch.tensor([b["freq"] for b in batch], dtype=torch.float32),
            "rank": torch.tensor([b["rank"] for b in batch], dtype=torch.long),
            "words": [b["word"] for b in batch],
        }
    return collate


def build_pool_loader(vocab: Vocab, n: int, batch_size: int, semantic_dim: int,
                      min_len: int = 2, max_len: int = 9, seed: int = 0) -> DataLoader:
    """A loader of pronounceable (C)V(C) pseudowords for training the dorsal
    route's general serial-recall (frequency-flat, no semantics)."""
    rng = random.Random(seed)
    cons = [vocab.stoi[s] for s in vocab.itos[3:] if vocab.sonority[vocab.stoi[s]] < 0.9]
    vow = [vocab.stoi[s] for s in vocab.itos[3:] if vocab.sonority[vocab.stoi[s]] >= 0.95]
    entries, seen = [], set()
    while len(entries) < n:
        f = []
        for _ in range(rng.randint(1, 3)):
            if rng.random() < 0.85:
                f.append(rng.choice(cons))
            f.append(rng.choice(vow))
            if rng.random() < 0.4:
                f.append(rng.choice(cons))
        if not (min_len <= len(f) <= max_len) or tuple(f) in seen:
            continue
        seen.add(tuple(f))
        entries.append(LexEntry(word="", phonemes=f,
                                semantic=np.zeros(semantic_dim, np.float32),
                                freq=1.0, rank=1))
    density = {i: 0 for i in range(len(entries))}
    return make_loader(entries, vocab, density, batch_size,
                       frequency_weighted=False, shuffle=True)


def make_loader(entries: List[LexEntry], vocab: Vocab, density: Dict[int, int],
                batch_size: int, frequency_weighted: bool, freq_temp: float = 1.0,
                shuffle: bool = True) -> DataLoader:
    ds = RepetitionDataset(entries, vocab, density)
    collate = make_collate(vocab.pad_id)
    if frequency_weighted:
        # log-frequency sampling from the words' frequency ranks
        w = logfreq_weights([e.rank for e in entries]) ** float(freq_temp)
        w = np.clip(w, 1e-6, None)
        w = w / w.sum()
        sampler = WeightedRandomSampler(weights=torch.as_tensor(w),
                                        num_samples=len(entries),
                                        replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          collate_fn=collate)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate)
