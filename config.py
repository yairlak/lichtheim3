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
    # Optional override for the lexicon TSV path.  Leave None to use the default
    # data/lexicon_en.tsv.  Useful for GloVe-filtered variants such as
    # data/lexicon_en_glove_covered.tsv without changing any other code.
    lexicon_path: Optional[str] = None
    semantic_dim: int = 300
    max_words: int = 30000         # the 30k most frequent words
    min_phonemes: int = 2
    max_phonemes: int = 9
    # Training uses LOG-frequency weighting over the words' frequency ranks
    # (see data.lexicon.logfreq_weights). `freq_temp` sharpens/softens the skew.
    freq_temp: float = 1.0
    val_fraction: float = 0.15
    # `seed` drives synthetic-lexicon generation (build_synthetic) and, when
    # `split_seed` is None, the train/validation partition as well.
    seed: int = 0
    # Seed for the train/validation split ONLY. Decoupled from `seed` so that
    # multi-seed studies can vary model initialisation and batch order while
    # holding the exact train/val word sets fixed.
    # None  -> fall back to `seed` (pre-2026-07 behaviour, and what every
    #          checkpoint saved before this field existed implies).
    split_seed: Optional[int] = None


def get_effective_split_seed(data_cfg) -> int:
    """Seed actually used for Lexicon.split().

    Backward compatible: checkpoints predating `split_seed` have no such field,
    so DataConfig(**cfg_data) leaves it None and we fall back to `seed`, exactly
    reproducing their original partition.
    """
    split_seed = getattr(data_cfg, "split_seed", None)
    if split_seed is None:
        return int(data_cfg.seed)
    return int(split_seed)


@dataclass
class WMConfig:
    """Dorsal recurrent serial-recall route (Botvinick & Plaut, 2006)."""
    hidden: int = 128              # bounded recurrent state -> capacity/length limits
    # Gaussian noise std on the WM encoder final hidden state.
    # Active when model.training=True OR apply_noise=True in encode/forward calls.
    # collect=True alone does NOT activate noise.
    # Named interference_noise for backward compatibility with old checkpoints.
    interference_noise: float = 0.1


@dataclass
class LTMConfig:
    """Ventral / lexical-semantic route.

    Architecture mode (ltm_encoder_mode):
        "bigru_masked_mean"   – historical default (Phase 0/baseline):
                                bidirectional GRU, sequence outputs, masked mean pooling.
        "unigru_last_hidden"  – Yair 2026-07 proposal (Phase 4+):
                                unidirectional GRU with pack_padded_sequence,
                                last valid hidden state replaces mean pooling.

    Old checkpoints without ltm_encoder_mode in their cfg_ltm dict were produced
    with bigru_masked_mean. Load with ltm_encoder_mode="bigru_masked_mean" (default).
    New checkpoints store ltm_encoder_mode explicitly.

    Weight incompatibility: unigru_last_hidden checkpoints cannot be loaded into
    a bigru_masked_mean model and vice versa (different to_semantic weight shapes).
    torch.load_state_dict will raise a clear shape-mismatch error in that case.
    """
    phon_embed_dim: int = 64
    enc_hidden: int = 256
    enc_layers: int = 1
    dec_hidden: int = 256
    # Kept for backward compatibility with old checkpoint cfg_ltm dicts.
    # ltm_encoder_mode is the authoritative field; this is informational only.
    bidirectional_encoder: bool = True
    # Authoritative architecture mode (see class docstring).
    ltm_encoder_mode: str = "bigru_masked_mean"
    # Gaussian noise std on the aggregated LTM encoder representation (pooled or
    # last hidden) before the semantic projection.  Analogous to WM interference_noise.
    # Active when model.training=True or apply_noise=True in encode calls.
    # Default 0.0 = no ventral noise (Phase 4 setting).
    ventral_noise: float = 0.0

    def __post_init__(self):
        # ltm_encoder_mode is authoritative; bidirectional_encoder is informational.
        # Normalise to prevent silent contradictory configs.
        # Old checkpoints without ltm_encoder_mode land on "bigru_masked_mean"
        # (the default), so bidirectional_encoder=True is correct for them.
        if self.ltm_encoder_mode not in ("bigru_masked_mean", "unigru_last_hidden"):
            raise ValueError(
                f"Unknown ltm_encoder_mode={self.ltm_encoder_mode!r}. "
                f"Valid: 'bigru_masked_mean', 'unigru_last_hidden'."
            )
        self.bidirectional_encoder = (self.ltm_encoder_mode == "bigru_masked_mean")


@dataclass
class GatingConfig:
    """Error-suppression gate: a confident lexical match suppresses the WM buffer.

    g = sigmoid(alpha * (lexical_confidence - gate_threshold))

    g -> 1 trusts the lexicon (real words); g -> 0 hands control to the buffer
    (novel/non-words).  gate_threshold=0.5 means the crossover point is at
    c_LTM=0.5 (equal cosine similarity to nearest known word and to no known word).

    gate_threshold was hard-coded to 0.5 in Phase 0/baseline.  It is now a
    configurable hyperparameter for Phase 7 grid search.
    """
    alpha: float = 4.0             # gate sharpness
    # prior on route usage: target mean gate (fraction of LTM use) for the
    # regularizer. 0.5 = no bias; raise to encourage lexical reliance.
    usage_prior: float = 0.5
    # Routing threshold τ in g = sigmoid(α·(c_LTM − τ)).
    # Phase 4: fixed at 0.5 (historical default).
    # Phase 7: search over {0.3, 0.5, 0.7}.
    gate_threshold: float = 0.5


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
    # Train the dorsal route additionally on a frequency-flat stream of
    # pronounceable pseudowords so it learns a general serial-recall / copy
    # (and stays lexical-frequency-invariant). 0 disables.
    dorsal_pool_size: int = 4000
    # Scheduled sampling ratio (0.0 = fully autoregressive; 1.0 = full teacher
    # forcing, i.e. the historical default).  At each decoder step t, the gold
    # previous token is used with probability teacher_forcing_ratio; the model's
    # own argmax prediction is used otherwise.  Validation always uses TF=1.0
    # regardless of this setting.  See train._forward_scheduled_sampling.
    teacher_forcing_ratio: float = 1.0
    # DataLoader num_workers.  0 = main process (safe default everywhere).
    # For Jean-Zay or other GPU servers with fast storage, test 4 or 8.
    num_workers: int = 0
    # Periodic checkpoint interval in epochs.  0 = disabled (save only at end).
    # With --save_every_epochs 5 the script writes epoch-level checkpoints named
    # <ckpt_stem>.epoch_NNNN<ext> alongside the final checkpoint.
    save_every_epochs: int = 0


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
