"""ARPABET phoneme inventory + articulatory feature space.

Two things live here:

1. A vocabulary mapping phoneme symbol <-> integer id, including the special
   tokens PAD / BOS / EOS. One-hot encoding is just an identity lookup on these
   ids, done in `dataset.py`.

2. A small, transparent **articulatory feature matrix**. Each phoneme is given a
   continuous feature vector (sonority, voicing, manner, place, vowel quality).
   Euclidean distance in this space is our proxy for *phonetic similarity*, which
   the sonority/confusion evaluation correlates against the model's substitution
   errors. Nothing here is learned — it is the ground-truth "acoustic-phonetic"
   geometry the model is supposed to (approximately) rediscover.

The features are intentionally coarse and human-readable so students can audit
them. They are not meant to be a definitive phonological theory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
SPECIALS = [PAD, BOS, EOS]

# --- Sonority hierarchy (low = obstruent, high = vowel) ----------------------
# A standard sonority scale, normalized to [0, 1].
SONORITY = {
    "stop": 0.0,
    "affricate": 0.15,
    "fricative": 0.30,
    "nasal": 0.55,
    "liquid": 0.75,
    "glide": 0.88,
    "vowel": 1.0,
}

# manner classes used for the one-hot manner block
MANNERS = ["stop", "affricate", "fricative", "nasal", "liquid", "glide", "vowel"]
# place classes (consonants); vowels get place = "none"
PLACES = ["labial", "labiodental", "dental", "alveolar",
          "postalveolar", "palatal", "velar", "glottal", "none"]

# phoneme -> (manner, place, voiced, [vheight, vback, vround])
# vowel quality is 0 for consonants. height: 0 low .. 1 high; back: 0 front .. 1
# back; round: 0/1.
_P = {
    # --- stops ---
    "P":  ("stop", "labial", 0, (0, 0, 0)),
    "B":  ("stop", "labial", 1, (0, 0, 0)),
    "T":  ("stop", "alveolar", 0, (0, 0, 0)),
    "D":  ("stop", "alveolar", 1, (0, 0, 0)),
    "K":  ("stop", "velar", 0, (0, 0, 0)),
    "G":  ("stop", "velar", 1, (0, 0, 0)),
    # --- affricates ---
    "CH": ("affricate", "postalveolar", 0, (0, 0, 0)),
    "JH": ("affricate", "postalveolar", 1, (0, 0, 0)),
    # --- fricatives ---
    "F":  ("fricative", "labiodental", 0, (0, 0, 0)),
    "V":  ("fricative", "labiodental", 1, (0, 0, 0)),
    "TH": ("fricative", "dental", 0, (0, 0, 0)),
    "DH": ("fricative", "dental", 1, (0, 0, 0)),
    "S":  ("fricative", "alveolar", 0, (0, 0, 0)),
    "Z":  ("fricative", "alveolar", 1, (0, 0, 0)),
    "SH": ("fricative", "postalveolar", 0, (0, 0, 0)),
    "ZH": ("fricative", "postalveolar", 1, (0, 0, 0)),
    "HH": ("fricative", "glottal", 0, (0, 0, 0)),
    # --- nasals ---
    "M":  ("nasal", "labial", 1, (0, 0, 0)),
    "N":  ("nasal", "alveolar", 1, (0, 0, 0)),
    "NG": ("nasal", "velar", 1, (0, 0, 0)),
    # --- liquids ---
    "L":  ("liquid", "alveolar", 1, (0, 0, 0)),
    "R":  ("liquid", "alveolar", 1, (0, 0, 0)),
    # --- glides ---
    "W":  ("glide", "labial", 1, (0, 0, 0)),
    "Y":  ("glide", "palatal", 1, (0, 0, 0)),
    # --- vowels: (height, backness, round) ---
    "IY": ("vowel", "none", 1, (1.00, 0.0, 0)),
    "IH": ("vowel", "none", 1, (0.80, 0.1, 0)),
    "EY": ("vowel", "none", 1, (0.70, 0.1, 0)),
    "EH": ("vowel", "none", 1, (0.55, 0.1, 0)),
    "AE": ("vowel", "none", 1, (0.30, 0.1, 0)),
    "AA": ("vowel", "none", 1, (0.05, 0.9, 0)),
    "AO": ("vowel", "none", 1, (0.25, 0.9, 1)),
    "OW": ("vowel", "none", 1, (0.60, 0.9, 1)),
    "UH": ("vowel", "none", 1, (0.80, 0.9, 1)),
    "UW": ("vowel", "none", 1, (1.00, 1.0, 1)),
    "AH": ("vowel", "none", 1, (0.45, 0.5, 0)),
    "ER": ("vowel", "none", 1, (0.50, 0.5, 0)),
    "AY": ("vowel", "none", 1, (0.40, 0.5, 0)),
    "AW": ("vowel", "none", 1, (0.40, 0.6, 1)),
    "OY": ("vowel", "none", 1, (0.55, 0.7, 1)),
}

PHONEMES: List[str] = list(_P.keys())


def _feature_vector(symbol: str) -> np.ndarray:
    manner, place, voiced, (vh, vb, vr) = _P[symbol]
    son = SONORITY[manner]
    manner_oh = [1.0 if manner == m else 0.0 for m in MANNERS]
    place_oh = [1.0 if place == p else 0.0 for p in PLACES]
    # Weight sonority a bit more — it dominates the perceptual scale we test.
    return np.array(
        [2.0 * son, float(voiced)] + manner_oh + place_oh + [vh, vb, float(vr)],
        dtype=np.float32,
    )


@dataclass
class Vocab:
    """Phoneme <-> id, plus the articulatory feature matrix."""
    itos: List[str]
    stoi: Dict[str, int]
    feature_matrix: np.ndarray  # (vocab, feat_dim); specials get zero features
    sonority: np.ndarray        # (vocab,) scalar sonority, NaN for specials

    @property
    def size(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    def phonetic_distance(self, a: int, b: int) -> float:
        """Euclidean distance between two phoneme ids in feature space."""
        return float(np.linalg.norm(self.feature_matrix[a] - self.feature_matrix[b]))


def build_vocab() -> Vocab:
    itos = list(SPECIALS) + PHONEMES
    stoi = {s: i for i, s in enumerate(itos)}
    feat_dim = _feature_vector(PHONEMES[0]).shape[0]
    fmat = np.zeros((len(itos), feat_dim), dtype=np.float32)
    son = np.full(len(itos), np.nan, dtype=np.float32)
    for s in PHONEMES:
        fmat[stoi[s]] = _feature_vector(s)
        manner = _P[s][0]
        son[stoi[s]] = SONORITY[manner]
    return Vocab(itos=itos, stoi=stoi, feature_matrix=fmat, sonority=son)


# A convenient singleton for scripts that just need the standard vocab.
VOCAB = build_vocab()
