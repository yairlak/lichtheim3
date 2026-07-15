"""Tests for scheduled sampling (teacher_forcing_ratio) training logic.

Core logic tests use a MockModel (no GloVe, no lexicon, no real weights)
so they run fast and without any data dependencies.

Smoke tests build a tiny real model and run one training step to confirm
the full pipeline doesn't crash at tf_ratio=0.0.  These require the
data/lexicon_en.tsv file but NOT GloVe (pseudo-vectors are used instead).
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train import _forward_scheduled_sampling


# ---------------------------------------------------------------------------
# Shared MockModel: always predicts a fixed token, records every dec_in seen
# ---------------------------------------------------------------------------

class MockModel:
    """Fake dual-route model for unit-testing scheduled sampling logic.

    Implements the encode_all / decode_from_states interface introduced in
    Phase 2B (encode-once / decode-stepwise).

    Always returns logits that strongly predict `always_predict` token.
    Records each `dec_in` fed to decode_from_states so tests can inspect
    the exact token history seen by the decoder at each step.
    `encode_call_count` verifies that the encoder is called exactly once
    per forward pass regardless of sequence length.
    """

    def __init__(self, vocab_size: int = 10, always_predict: int = 3,
                 s_hat_dim: int = 4):
        self.vocab_size        = vocab_size
        self.always_predict    = always_predict
        self.s_hat_dim         = s_hat_dim
        self.inputs_seen: list[torch.Tensor] = []
        self.encode_call_count:     int = 0
        self.wm_encode_call_count:  int = 0
        self.ltm_encode_call_count: int = 0

    def encode_all(self, enc_in, enc_mask,
                   collect: bool = False,
                   apply_noise: bool = False) -> dict:
        self.encode_call_count += 1
        self.wm_encode_call_count  += 1
        self.ltm_encode_call_count += 1
        B = enc_in.shape[0]
        return {
            "h_WM":  torch.zeros(1, B, 8),
            "s_hat": torch.zeros(B, self.s_hat_dim),
            "field": None,
        }

    def decode_from_states(self, h_WM, s_hat, field, dec_in) -> dict:
        self.inputs_seen.append(dec_in.clone())
        B, S = dec_in.shape
        logits = torch.zeros(B, S, self.vocab_size)
        logits[:, :, self.always_predict] = 10.0   # confident prediction
        gate = torch.full((B, S, 1), 0.5)
        return {
            "logits":     logits,
            "wm_logits":  logits,
            "ltm_logits": logits,
            "gate":       gate,
            "s_hat":      s_hat,
        }


def _make_batch(dec_in: list[list[int]], pad_id: int = 0) -> dict:
    """Build a minimal batch dict from a list of dec_in sequences."""
    B   = len(dec_in)
    S   = max(len(d) for d in dec_in)
    enc = torch.zeros(B, S, dtype=torch.long)
    msk = torch.ones(B, S, dtype=torch.bool)
    di  = torch.full((B, S), pad_id, dtype=torch.long)
    dt  = torch.full((B, S), pad_id, dtype=torch.long)
    for i, d in enumerate(dec_in):
        di[i, :len(d)] = torch.tensor(d, dtype=torch.long)
        dt[i, :len(d)] = torch.tensor(d, dtype=torch.long)
    return {"enc_in": enc, "enc_mask": msk, "dec_in": di, "dec_tgt": dt}


# ---------------------------------------------------------------------------
# Test 1: tf_ratio=1.0 feeds gold tokens at every step
# ---------------------------------------------------------------------------

def test_tf1_feeds_gold_tokens():
    """With tf_ratio=1.0 the step-by-step path should replay dec_in exactly."""
    BOS, GOLD, PAD = 9, 7, 0  # gold token (7) differs from mock prediction (3)
    dec_in = [[BOS, GOLD, GOLD, GOLD]]   # (B=1, S=4)
    batch  = _make_batch(dec_in, pad_id=PAD)
    model  = MockModel(always_predict=3)

    _forward_scheduled_sampling(model, batch, tf_ratio=1.0, pad_id=PAD, device="cpu")

    # step 0 → input=[BOS], step 1 → [BOS, GOLD], step 2 → [BOS, GOLD, GOLD], …
    assert model.inputs_seen[0][0].tolist() == [BOS]
    assert model.inputs_seen[1][0].tolist() == [BOS, GOLD]
    assert model.inputs_seen[2][0].tolist() == [BOS, GOLD, GOLD]
    assert model.inputs_seen[3][0].tolist() == [BOS, GOLD, GOLD, GOLD]


# ---------------------------------------------------------------------------
# Test 2: tf_ratio=0.0 feeds model predictions at every step after BOS
# ---------------------------------------------------------------------------

def test_tf0_feeds_predicted_tokens():
    """With tf_ratio=0.0 the decoder should receive model predictions, not gold."""
    BOS, GOLD, PRED, PAD = 9, 7, 3, 0
    dec_in = [[BOS, GOLD, GOLD, GOLD]]
    batch  = _make_batch(dec_in, pad_id=PAD)
    model  = MockModel(always_predict=PRED)  # model always predicts PRED=3

    _forward_scheduled_sampling(model, batch, tf_ratio=0.0, pad_id=PAD, device="cpu")

    # step 0 always starts with [BOS]; subsequent steps must use PRED, not GOLD
    assert model.inputs_seen[0][0].tolist() == [BOS]
    assert model.inputs_seen[1][0].tolist() == [BOS, PRED]
    assert model.inputs_seen[2][0].tolist() == [BOS, PRED, PRED]
    assert model.inputs_seen[3][0].tolist() == [BOS, PRED, PRED, PRED]


# ---------------------------------------------------------------------------
# Test 3: output shapes match (B, S, V) for both tf=0.0 and tf=1.0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tf_ratio", [0.0, 0.5, 1.0])
def test_output_shapes(tf_ratio):
    BOS, PAD = 9, 0
    B, S, V = 3, 6, 10
    dec_in = [[BOS] + [1] * (S - 1)] * B
    batch  = _make_batch(dec_in, pad_id=PAD)
    model  = MockModel(vocab_size=V, always_predict=2)

    out = _forward_scheduled_sampling(model, batch, tf_ratio=tf_ratio,
                                       pad_id=PAD, device="cpu")

    assert out["logits"].shape     == (B, S, V)
    assert out["wm_logits"].shape  == (B, S, V)
    assert out["ltm_logits"].shape == (B, S, V)
    assert out["gate"].shape       == (B, S, 1)
    assert out["s_hat"].shape[0]   == B


# ---------------------------------------------------------------------------
# Test 4: tf=0.0 and tf=1.0 produce different inputs (unless gold==pred)
# ---------------------------------------------------------------------------

def test_tf0_and_tf1_differ():
    """AR and TF runs must feed different tokens to the decoder."""
    BOS, GOLD, PRED, PAD = 9, 7, 3, 0  # GOLD ≠ PRED  ✓
    dec_in = [[BOS, GOLD, GOLD, GOLD]]
    batch  = _make_batch(dec_in, pad_id=PAD)

    model_tf = MockModel(always_predict=PRED)
    model_ar = MockModel(always_predict=PRED)

    _forward_scheduled_sampling(model_tf, batch, tf_ratio=1.0, pad_id=PAD, device="cpu")
    _forward_scheduled_sampling(model_ar, batch, tf_ratio=0.0, pad_id=PAD, device="cpu")

    # At step 1, TF sees gold, AR sees prediction
    assert model_tf.inputs_seen[1][0, 1].item() == GOLD
    assert model_ar.inputs_seen[1][0, 1].item() == PRED


# ---------------------------------------------------------------------------
# Test 5: _edit_distance helper used by ceiling eval
# ---------------------------------------------------------------------------

def test_edit_distance():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "evaluate_train_lexicon_ceiling",
        pathlib.Path(ROOT) / "scripts" / "evaluate_train_lexicon_ceiling.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _edit_distance = mod._edit_distance
    assert _edit_distance([], []) == 0
    assert _edit_distance(["a"], ["a"]) == 0
    assert _edit_distance(["a", "b"], ["a", "c"]) == 1
    assert _edit_distance(["a", "b", "c"], []) == 3
    assert _edit_distance([], ["a", "b"]) == 2
    assert _edit_distance(["k", "ae", "t"], ["k", "ae", "p"]) == 1


# ---------------------------------------------------------------------------
# Smoke test: real tiny model, one training step at tf_ratio=0.0
# ---------------------------------------------------------------------------

def _tiny_cfg():
    from config import Config, DataConfig, WMConfig, LTMConfig, GatingConfig, TrainConfig
    cfg = Config()
    cfg.data.use_real      = True
    cfg.data.glove_path    = None          # use pseudo-vectors, skip large GloVe file
    cfg.data.max_words     = 32
    cfg.data.min_phonemes  = 2
    cfg.data.max_phonemes  = 4
    cfg.data.semantic_dim  = 300
    cfg.wm.hidden          = 16
    cfg.wm.interference_noise = 0.0       # deterministic for test
    cfg.ltm.phon_embed_dim = 16
    cfg.ltm.enc_hidden     = 16
    cfg.ltm.dec_hidden     = 16
    cfg.train.epochs       = 1
    cfg.train.batch_size   = 8
    cfg.train.device       = "cpu"
    cfg.train.dorsal_pool_size = 0        # skip pool for speed
    cfg.train.seed         = 42
    return cfg


@pytest.mark.slow
def test_training_smoke_tf0():
    """One training epoch with tf_ratio=0.0 must complete without error."""
    from train import build_and_train
    cfg = _tiny_cfg()
    cfg.train.teacher_forcing_ratio = 0.0
    model, vocab, lexicon, history, optim = build_and_train(cfg)
    assert len(history) == 1
    assert history[0]["train_total"] > 0
    assert history[0]["train_total"] < 1e6   # not exploded


@pytest.mark.slow
def test_training_smoke_tf1():
    """One training epoch with tf_ratio=1.0 must complete without error."""
    from train import build_and_train
    cfg = _tiny_cfg()
    cfg.train.teacher_forcing_ratio = 1.0
    model, vocab, lexicon, history, optim = build_and_train(cfg)
    assert len(history) == 1
    assert history[0]["train_total"] > 0


# ---------------------------------------------------------------------------
# Test 6: encode_all called exactly once per forward pass (Phase 2B invariant)
# ---------------------------------------------------------------------------

def test_encode_called_once():
    """encode_all must be called exactly once per _forward_scheduled_sampling call,
    regardless of sequence length S.  Re-encoding at every decoder step was the
    Phase 2B blocking bug: noise was drawn S times instead of once per stimulus."""
    BOS, PAD = 9, 0
    S = 6
    dec_in = [[BOS] + [1] * (S - 1)]
    batch  = _make_batch(dec_in, pad_id=PAD)
    model  = MockModel(always_predict=2)

    _forward_scheduled_sampling(model, batch, tf_ratio=0.5, pad_id=PAD, device="cpu")

    assert model.encode_call_count == 1, (
        f"Expected encode_all to be called exactly once, got {model.encode_call_count}. "
        f"This means noise is being drawn {model.encode_call_count} times per stimulus "
        f"instead of once (Phase 2B encode-once invariant violated)."
    )
    # Each route is encoded exactly once inside the single encode_all call
    assert model.wm_encode_call_count  == 1, (
        f"WM encoder called {model.wm_encode_call_count} times (expected 1)"
    )
    assert model.ltm_encode_call_count == 1, (
        f"LTM encoder called {model.ltm_encode_call_count} times (expected 1)"
    )


def test_encode_called_once_large_S():
    """Even for long sequences (S=12), encode_all is still called exactly once.
    The WM and LTM encoders each draw noise exactly once per stimulus."""
    BOS, PAD = 9, 0
    S = 12
    dec_in = [[BOS] + [1] * (S - 1)]
    batch  = _make_batch(dec_in, pad_id=PAD)
    model  = MockModel(always_predict=2)

    _forward_scheduled_sampling(model, batch, tf_ratio=0.3, pad_id=PAD, device="cpu")

    assert model.encode_call_count     == 1, f"encode_all called {model.encode_call_count}× (expected 1)"
    assert model.wm_encode_call_count  == 1, f"WM encoder called {model.wm_encode_call_count}× (expected 1)"
    assert model.ltm_encode_call_count == 1, f"LTM encoder called {model.ltm_encode_call_count}× (expected 1)"
