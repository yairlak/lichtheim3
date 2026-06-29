"""Lexicon construction.

A `Lexicon` is just a list of `LexEntry`s, each tying together four things the
model needs:

    word      : the orthographic string (for reporting only)
    phonemes  : list[int] phoneme ids (no specials) -- the input/target form
    semantic  : np.ndarray (semantic_dim,) -- the GloVe vector to align to
    freq      : float in (0, 1] -- training-frequency weight (Zipfian)

Two builders are provided:

  * `build_bundled`   : the realistic English lexicon shipped in
                        `data/lexicon_en.tsv` (30k frequency-ranked real words +
                        ARPABET). Semantics from GloVe if available, else a stable
                        deterministic pseudo-vector.
  * `build_synthetic` : a self-contained pseudo-lexicon with structured semantics
                        and deliberately seeded dense/sparse neighborhoods.

`build_lexicon(cfg)` uses the bundled lexicon when `cfg.use_real` (the default)
and falls back to synthetic only if the bundled file is missing.
"""
from __future__ import annotations

import os
import random
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .phonemes import VOCAB, PHONEMES, Vocab

BUNDLED_PATH = os.path.join(os.path.dirname(__file__), "lexicon_en.tsv")


def logfreq_weights(ranks, n_total: int = None) -> np.ndarray:
    """Log-frequency training weights from frequency ranks (1 = most frequent).

    Word frequencies follow Zipf's law (freq ~ 1/rank), so the *log* frequency
    is ~ log(N/rank). Training on log-frequency (rather than raw frequency)
    compresses the enormous high-frequency tail, matching the practice in the
    word-repetition modelling literature. Returns positive, monotonically
    decreasing weights (rank 1 highest).
    """
    ranks = np.asarray(ranks, dtype=np.float64)
    N = float(n_total or ranks.max())
    return np.log((N + 1.0) / ranks)


@dataclass
class LexEntry:
    word: str
    phonemes: List[int]
    semantic: np.ndarray
    freq: float
    rank: int = 0

    @property
    def length(self) -> int:
        return len(self.phonemes)


class Lexicon:
    def __init__(self, entries: List[LexEntry], vocab: Vocab, semantic_dim: int,
                 source: str):
        self.entries = entries
        self.vocab = vocab
        self.semantic_dim = semantic_dim
        self.source = source  # "real" or "synthetic"
        self._density_cache: Optional[Dict[int, int]] = None

    def __len__(self) -> int:
        return len(self.entries)

    # ---- phonological neighborhood density -----------------------------------
    @staticmethod
    def _is_one_edit(a: List[int], b: List[int]) -> bool:
        """True if a and b are one substitution/insertion/deletion apart.

        This is the standard phonological-neighbor criterion.
        """
        la, lb = len(a), len(b)
        if abs(la - lb) > 1:
            return False
        if la == lb:  # substitution
            diff = sum(1 for x, y in zip(a, b) if x != y)
            return diff == 1
        # insertion/deletion: make `a` the shorter
        if la > lb:
            a, b = b, a
            la, lb = lb, la
        i = j = 0
        skipped = False
        while i < la and j < lb:
            if a[i] == b[j]:
                i += 1
                j += 1
            elif not skipped:
                skipped = True
                j += 1
            else:
                return False
        return True

    def neighborhood_density(self, max_n: int = 6000) -> Dict[int, int]:
        """Map entry-index -> number of one-edit neighbors in the lexicon.

        The computation is O(N^2); for large lexicons we restrict it to the
        `max_n` most frequent entries (the rest report 0). Neighborhood
        evaluations sample from this set, so the cap is harmless.
        """
        if self._density_cache is not None:
            return self._density_cache
        n = min(len(self.entries), max_n)
        forms = [e.phonemes for e in self.entries[:n]]
        dens = {i: 0 for i in range(len(self.entries))}
        for i in range(n):
            for j in range(i + 1, n):
                if self._is_one_edit(forms[i], forms[j]):
                    dens[i] += 1
                    dens[j] += 1
        self._density_cache = dens
        return dens

    def split(self, val_fraction: float, seed: int):
        idx = list(range(len(self.entries)))
        rng = random.Random(seed)
        rng.shuffle(idx)
        n_val = int(len(idx) * val_fraction)
        val_idx = set(idx[:n_val])
        train = [e for i, e in enumerate(self.entries) if i not in val_idx]
        val = [e for i, e in enumerate(self.entries) if i in val_idx]
        return train, val


def _zipf_freq(rank: int) -> float:
    """Zipf-ish frequency weight in (0, 1] from a 0-based frequency rank."""
    return 1.0 / (1.0 + rank)


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------
def build_synthetic(cfg, vocab: Vocab = VOCAB) -> Lexicon:
    rng = np.random.default_rng(cfg.seed)
    py_rng = random.Random(cfg.seed)

    consonants = [p for p in PHONEMES if vocab.sonority[vocab.stoi[p]] < 0.9]
    vowels = [p for p in PHONEMES if vocab.sonority[vocab.stoi[p]] >= 0.95]

    def random_form(n_syll: int) -> List[str]:
        form: List[str] = []
        for _ in range(n_syll):
            if py_rng.random() < 0.85:
                form.append(py_rng.choice(consonants))
            form.append(py_rng.choice(vowels))
            if py_rng.random() < 0.4:
                form.append(py_rng.choice(consonants))
        return form

    # structured semantic space: K categories => phonological neighbors can share
    # meaning, which gives the ventral route something non-trivial to align.
    n_cat = 24
    centroids = rng.normal(size=(n_cat, cfg.semantic_dim)).astype(np.float32)

    forms = set()
    entries: List[LexEntry] = []
    rank = 0
    attempts = 0
    while len(entries) < cfg.max_words and attempts < cfg.max_words * 40:
        attempts += 1
        n_syll = py_rng.randint(1, 3)
        form = random_form(n_syll)
        if not (cfg.min_phonemes <= len(form) <= cfg.max_phonemes):
            continue
        key = tuple(form)
        if key in forms:
            continue
        forms.add(key)
        cat = py_rng.randrange(n_cat)
        vec = centroids[cat] + 0.35 * rng.normal(size=cfg.semantic_dim).astype(np.float32)
        ids = [vocab.stoi[p] for p in form]
        word = "".join(p.lower() for p in form)
        entries.append(LexEntry(word=word, phonemes=ids, semantic=vec,
                                freq=_zipf_freq(rank), rank=rank))
        rank += 1

    # Re-rank by a shuffle so frequency is not confounded with form length
    py_rng.shuffle(entries)
    for r, e in enumerate(entries):
        e.rank = r
        e.freq = _zipf_freq(r)
    return Lexicon(entries, vocab, cfg.semantic_dim, source="synthetic")


# ---------------------------------------------------------------------------
# Bundled realistic English lexicon (real words + ARPABET + frequency rank)
# ---------------------------------------------------------------------------
def _deterministic_semantic(word: str, dim: int) -> np.ndarray:
    """Stable pseudo-semantic vector for a word (used when GloVe is absent).

    Deterministic across runs/machines (seeded by a stable hash of the word), so
    the ventral route has a fixed lexical target to align to even offline.
    """
    seed = zlib.crc32(word.encode("utf-8")) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


def _load_glove_map(cfg) -> Optional[Dict[str, np.ndarray]]:
    path = getattr(cfg, "glove_path", None)
    if not path or not os.path.exists(path):
        return None
    g = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            vec = np.asarray(parts[1:1 + cfg.semantic_dim], dtype=np.float32)
            if vec.shape[0] == cfg.semantic_dim:
                g[parts[0]] = vec
    return g


def build_bundled(cfg, vocab: Vocab = VOCAB, path: str = BUNDLED_PATH
                  ) -> Optional[Lexicon]:
    """Load `lexicon_en.tsv`: real frequency-ranked English words + ARPABET.

    Semantic targets come from GloVe if `cfg.glove_path` is available, else a
    stable deterministic pseudo-vector per word (the lexicon itself is real
    either way). Frequency rank is taken straight from the file.
    """
    if not os.path.exists(path):
        return None
    glove = _load_glove_map(cfg)
    entries: List[LexEntry] = []
    with open(path, "r", encoding="utf-8") as f:
        next(f)  # header: rank\tword\tarpabet
        for line in f:
            r, word, arp = line.rstrip("\n").split("\t")
            phones = arp.split()
            if any(p not in vocab.stoi for p in phones):
                continue
            if not (cfg.min_phonemes <= len(phones) <= cfg.max_phonemes):
                continue
            ids = [vocab.stoi[p] for p in phones]
            sem = (glove.get(word) if glove else None)
            if sem is None:
                sem = _deterministic_semantic(word, cfg.semantic_dim)
            entries.append(LexEntry(word=word, phonemes=ids, semantic=sem,
                                    freq=_zipf_freq(int(r) - 1), rank=int(r)))
            if len(entries) >= cfg.max_words:
                break
    if len(entries) < 50:
        return None
    return Lexicon(entries, vocab, cfg.semantic_dim, source="bundled-en")


def build_lexicon(cfg, vocab: Vocab = VOCAB) -> Lexicon:
    if cfg.use_real:
        lex = build_bundled(cfg, vocab)          # realistic English, no downloads
        if lex is not None:
            return lex
        print("[lexicon] bundled lexicon_en.tsv missing -> synthetic fallback.")
    return build_synthetic(cfg, vocab)
