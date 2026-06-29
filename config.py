"""Central configuration.

Everything that a student might want to vary lives here as a plain dataclass, so
the model code stays free of magic numbers. Read top-to-bottom like a spec sheet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    # Lexicon source. If `use_real=True` we load the bundled realistic English
    # lexicon `data/lexicon_en.tsv` (30k frequency-ranked real words + ARPABET);
    # GloVe vectors at `glove_path` are used for semantics if present, otherwise
    # a deterministic per-word pseudo-vector is used. Falls back to synthetic if
    # the bundled file is missing.
    use_real: bool = True
    glove_path: Optional[str] = "data/glove.6B.300d.txt"
    semantic_dim: int = 300
    max_words: int = 30000         # the 30k most frequent words
    min_phonemes: int = 2
    max_phonemes: int = 9
    # Training uses LOG-frequency weighting over the words' frequency ranks
    # (see data.lexicon.logfreq_weights). `freq_temp` sharpens/softens the skew.
    freq_temp: float = 1.0
    val_fraction: float = 0.15
    seed: int = 0


@dataclass
class WMConfig:
    """Dorsal recurrent serial-recall route (Botvinick & Plaut, 2006)."""
    hidden: int = 128              # bounded recurrent state -> capacity/length limits
    interference_noise: float = 0.1  # Gaussian noise on the recalled state


@dataclass
class LTMConfig:
    """Ventral / lexical-semantic route."""
    phon_embed_dim: int = 64
    enc_hidden: int = 256
    enc_layers: int = 1
    dec_hidden: int = 256
    bidirectional_encoder: bool = True


@dataclass
class GatingConfig:
    """Error-suppression gate: a confident lexical match suppresses the WM buffer.

    g = sigmoid(alpha * (lexical_confidence - 0.5)); g->1 trusts the lexicon
    (real words), g->0 hands control to the buffer (novel/non-words).
    """
    alpha: float = 4.0             # gate sharpness
    # prior on route usage: target mean gate (fraction of LTM use) for the
    # regularizer. 0.5 = no bias; raise to encourage lexical reliance.
    usage_prior: float = 0.5


@dataclass
class LossConfig:
    rep: float = 1.0               # main repetition cross-entropy (motor output)
    align: float = 1.0             # semantic alignment to GloVe
    dec: float = 0.5               # ventral form reconstruction
    wm: float = 0.5                # auxiliary WM-only repetition (keeps WM honest)
    gate: float = 0.05             # route-usage regularizer
    label_smoothing: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = 8
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    device: str = "cpu"            # "cuda" if available; run_all auto-detects
    log_every: int = 50
    seed: int = 0


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    wm: WMConfig = field(default_factory=WMConfig)
    ltm: LTMConfig = field(default_factory=LTMConfig)
    gating: GatingConfig = field(default_factory=GatingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    out_dir: str = "outputs"


def default_config() -> Config:
    return Config()
