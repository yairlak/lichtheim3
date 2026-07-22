"""Regression tests for the train/validation split seed.

Context: before 2026-07, `train_checkpoint.py --seed N` set both
`cfg.train.seed` (initialisation, batch order, dorsal pool) and
`cfg.data.seed`, and `Lexicon.split()` shuffles with `random.Random(seed)`.
Varying the seed therefore also varied *which words* were in train vs val,
making multi-seed studies incomparable (measured val-set overlap between
seed 0 and seed 1: 677/4435).

`DataConfig.split_seed` now controls the partition alone, defaulting to None
so that omitting it reproduces the historical behaviour exactly.

These tests use the real repository helpers; they never retrain anything and
require no GPU.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os

import pytest

from config import Config, DataConfig, get_effective_split_seed
from data.lexicon import build_lexicon
from data.phonemes import build_vocab

# Reference partition, seed 0, data/lexicon_en_glove_covered.tsv, max_words=30000.
# Produced by every Phase 6A / 7A / 7B run.
REF_TRAIN_SHA = "c35228443fa35fb85605d5e43e824c989129688321c5ebee313c822e0982e420"
REF_VAL_SHA = "52fe47702402850f6412311729b7a4b75357bc94eb9105afa0f7188b0a8e66f6"
REF_N_TRAIN = 25136
REF_N_VAL = 4435

LEXICON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "lexicon_en_glove_covered.tsv",
)
needs_lexicon = pytest.mark.skipif(
    not os.path.isfile(LEXICON),
    reason="GloVe-covered lexicon not available in this checkout",
)


def _sha(words) -> str:
    return hashlib.sha256("\n".join(sorted(words)).encode()).hexdigest()


def _split_for(seed: int, split_seed):
    """Build the real lexicon and split it, exactly as the training scripts do."""
    cfg = Config()
    cfg.data.lexicon_path = LEXICON
    cfg.data.max_words = 30000
    cfg.data.seed = seed
    cfg.data.split_seed = split_seed
    cfg.train.seed = seed
    lexicon = build_lexicon(cfg.data, build_vocab())
    train, val = lexicon.split(cfg.data.val_fraction,
                               get_effective_split_seed(cfg.data))
    return (_sha(e.word for e in train), _sha(e.word for e in val),
            len(train), len(val))


# --------------------------------------------------------------------------- #
# 1. backward-compatible fallback
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 3, 4])
def test_fallback_to_training_seed(seed):
    """split_seed=None -> the partition follows `seed`, as it always did."""
    assert get_effective_split_seed(DataConfig(seed=seed)) == seed


def test_fallback_for_checkpoint_without_split_seed():
    """A cfg_data dict saved before the field existed has no `split_seed` key."""
    legacy = {"seed": 3, "val_fraction": 0.15}
    assert get_effective_split_seed(DataConfig(**legacy)) == 3


# --------------------------------------------------------------------------- #
# 2. explicit override
# --------------------------------------------------------------------------- #
def test_explicit_override():
    assert get_effective_split_seed(DataConfig(seed=3, split_seed=0)) == 0


def test_override_is_independent_of_training_seed():
    for seed in (0, 1, 2, 3, 4):
        assert get_effective_split_seed(DataConfig(seed=seed, split_seed=0)) == 0


# --------------------------------------------------------------------------- #
# 3. serialisation round-trip (checkpoints store dataclasses.asdict(cfg.data))
# --------------------------------------------------------------------------- #
def test_serialisation_round_trip():
    serialised = dataclasses.asdict(DataConfig(seed=3, split_seed=0))
    assert serialised["split_seed"] == 0
    restored = DataConfig(**serialised)
    assert restored.seed == 3
    assert restored.split_seed == 0
    assert get_effective_split_seed(restored) == 0


# --------------------------------------------------------------------------- #
# 4. the Phase 7C guarantee: one partition across training seeds
# --------------------------------------------------------------------------- #
@needs_lexicon
def test_split_identical_across_training_seeds():
    results = {s: _split_for(s, 0) for s in (0, 1, 2, 3, 4)}
    assert {r[2] for r in results.values()} == {REF_N_TRAIN}
    assert {r[3] for r in results.values()} == {REF_N_VAL}
    assert len({r[0] for r in results.values()}) == 1, "train partition varies"
    assert len({r[1] for r in results.values()}) == 1, "val partition varies"


@needs_lexicon
def test_split_matches_historical_reference():
    """split_seed=0 must reproduce the partition every earlier phase used."""
    train_sha, val_sha, n_train, n_val = _split_for(4, 0)
    assert train_sha == REF_TRAIN_SHA
    assert val_sha == REF_VAL_SHA
    assert (n_train, n_val) == (REF_N_TRAIN, REF_N_VAL)


# --------------------------------------------------------------------------- #
# 5. the split is still deliberately changeable
# --------------------------------------------------------------------------- #
@needs_lexicon
def test_split_seed_actually_changes_the_partition():
    a_train, a_val, _, _ = _split_for(3, 0)
    b_train, b_val, _, _ = _split_for(3, 1)
    assert a_train != b_train
    assert a_val != b_val
    assert (a_train, a_val) == (REF_TRAIN_SHA, REF_VAL_SHA)
