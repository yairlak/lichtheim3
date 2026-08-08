"""Focused tests for the Phase 1A frozen naming/comprehension probe.

Covers, without touching any checkpoint or data file:
  - tensor shapes of the encoder and the semantic AR wrapper;
  - deterministic eval behaviour;
  - target-bank identity/order hashing;
  - semantic AR starts from BOS and stops on EOS / global cap only
    (it never inspects the target item length);
  - wrapper consistency with the existing LTM decode machinery;
  - no existing repetition API is modified by importing the probe.
"""
from __future__ import annotations

import inspect
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import Config, DataConfig, LTMConfig, TrainConfig            # noqa: E402
from data.phonemes import build_vocab                                    # noqa: E402
from models.dual_route import DualRouteModel                             # noqa: E402
from models.ltm_route import LTMLexicon                                  # noqa: E402
from scripts.naming_comprehension.frozen_probe import (                  # noqa: E402
    comprehension_metrics, encode_all, load_frozen, semantic_greedy_decode,
    teacher_forced_naming)
from utils.provenance import sha256_words_ordered                        # noqa: E402

_DECODE_FROM_S_HAT_BEFORE_IMPORT = LTMLexicon.decode_from_s_hat


def _tiny_model(seed: int = 0) -> tuple:
    torch.manual_seed(seed)
    cfg = Config(
        data=DataConfig(use_real=False, glove_path=None, semantic_dim=32,
                        max_words=50, seed=0),
        ltm=LTMConfig(phon_embed_dim=8, enc_hidden=16, dec_hidden=16,
                      ltm_encoder_mode="unigru_last_hidden"),
        train=TrainConfig(device="cpu", seed=0),
    )
    vocab = build_vocab()
    model = DualRouteModel(cfg, vocab, premotor_dim=12)
    bank = torch.randn(20, cfg.data.semantic_dim)
    model.set_semantic_bank(bank)
    model.eval()
    return model, vocab, cfg, bank


# ------------------------------------------------------------------ shapes

def test_encoder_and_wrapper_shapes():
    model, vocab, cfg, _ = _tiny_model()
    forms = [[5, 6, 7], [8, 9], [10, 11, 12, 13]]
    s_hat = encode_all(model, vocab, forms, "cpu", batch_size=2)
    assert s_hat.shape == (3, cfg.data.semantic_dim)

    sem = torch.randn(3, cfg.data.semantic_dim)
    preds, eos = semantic_greedy_decode(model, sem, vocab, max_steps=10)
    assert len(preds) == 3 and len(eos) == 3
    for p, e in zip(preds, eos):
        assert len(p) <= 10
        assert all(t != vocab.eos_id for t in p)
        if not e:
            assert len(p) == 10  # no EOS within cap -> full-cap emission


# ------------------------------------------------------------- determinism

def test_eval_determinism():
    model, vocab, cfg, _ = _tiny_model()
    forms = [[5, 6, 7], [8, 9, 10, 11]]
    s1 = encode_all(model, vocab, forms, "cpu", batch_size=2)
    s2 = encode_all(model, vocab, forms, "cpu", batch_size=2)
    assert torch.equal(s1, s2)

    sem = torch.randn(4, cfg.data.semantic_dim)
    p1, e1 = semantic_greedy_decode(model, sem, vocab, max_steps=8)
    p2, e2 = semantic_greedy_decode(model, sem, vocab, max_steps=8)
    assert p1 == p2 and e1 == e2

    tf1 = teacher_forced_naming(model, sem, [[5, 6], [7], [8, 9], [10]], vocab)
    tf2 = teacher_forced_naming(model, sem, [[5, 6], [7], [8, 9], [10]], vocab)
    assert tf1 == tf2


# ------------------------------------------------------- bank identity/order

def test_bank_order_hash_is_order_sensitive():
    words = ["the", "of", "and", "sea", "see"]
    h1 = sha256_words_ordered(words)
    h2 = sha256_words_ordered(list(reversed(words)))
    assert h1 != h2
    assert h1 == sha256_words_ordered(list(words))  # stable


def test_comprehension_identity_retrieval():
    # s_hat exactly equal to bank rows -> perfect retrieval on distinct vectors.
    torch.manual_seed(1)
    bank = torch.randn(12, 16)
    res = comprehension_metrics(bank.clone(), bank, list(range(12)), batch_size=5)
    assert res["top1"].all() and res["top5"].all()
    assert (res["target_rank"] == 1).all()
    assert (res["margin"] > 0).all()
    np.testing.assert_allclose(res["target_cos"], 1.0, atol=1e-6)
    assert list(res["top1_idx"]) == list(range(12))


# ------------------------------------- BOS start / EOS-or-cap stop semantics

class _ScriptedModel:
    """Stub emitting a fixed token script; records every decoder prefix fed."""

    def __init__(self, script, vocab_size):
        self.script = script
        self.vocab_size = vocab_size
        self.seen_prefixes = []
        outer = self

        class _LTM:
            def decode_from_s_hat(self, sem, dec_input):
                outer.seen_prefixes.append(dec_input.clone())
                return dec_input.unsqueeze(-1).float()  # (B, S, 1) placeholder

        self.ltm = _LTM()

    def motor(self, premotor):
        B, S, _ = premotor.shape
        logits = torch.zeros(B, S, self.vocab_size)
        step = S - 1  # wrapper reads only the last position
        tok = self.script[min(step, len(self.script) - 1)]
        logits[:, -1, tok] = 10.0
        return logits


def test_wrapper_starts_from_bos_and_stops_on_eos():
    vocab = build_vocab()
    script = [5, 6, 7, vocab.eos_id, 9, 9]     # EOS at step 3
    model = _ScriptedModel(script, vocab.size)
    sem = torch.zeros(2, 4)
    preds, eos = semantic_greedy_decode(model, sem, vocab, max_steps=10)
    assert preds == [[5, 6, 7], [5, 6, 7]]
    assert eos == [True, True]
    # first prefix fed is exactly [BOS]
    first = model.seen_prefixes[0]
    assert first.shape == (2, 1)
    assert (first == vocab.bos_id).all()
    # decoding stopped at EOS: 4 steps, not the cap of 10
    assert len(model.seen_prefixes) == 4


def test_wrapper_stops_at_global_cap_without_eos():
    vocab = build_vocab()
    model = _ScriptedModel([5], vocab.size)     # never emits EOS
    sem = torch.zeros(1, 4)
    preds, eos = semantic_greedy_decode(model, sem, vocab, max_steps=6)
    assert preds == [[5] * 6]
    assert eos == [False]
    assert len(model.seen_prefixes) == 6


def test_wrapper_signature_has_no_length_argument():
    params = inspect.signature(semantic_greedy_decode).parameters
    assert set(params) == {"model", "sem", "vocab", "max_steps"}
    assert not any("len" in p or "length" in p for p in params)


# ------------------------------ consistency with existing LTM machinery

def test_wrapper_consistent_with_full_prefix_decode():
    """Greedy tokens must be reproduced by one full decode_from_s_hat pass
    over the final prefix (GRU prefix property), i.e. the stepwise wrapper
    computes exactly what the existing machinery computes."""
    model, vocab, cfg, _ = _tiny_model()
    sem = torch.randn(3, cfg.data.semantic_dim)
    preds, eos = semantic_greedy_decode(model, sem, vocab, max_steps=7)
    for i in range(3):
        prefix = torch.tensor([[vocab.bos_id] + preds[i]])
        logits = model.motor(model.ltm.decode_from_s_hat(sem[i:i + 1], prefix))
        steps = logits.argmax(-1)[0].tolist()
        expected = preds[i] + ([vocab.eos_id] if eos[i] else [])
        assert steps[:len(expected)] == expected


def test_teacher_forced_naming_matches_direct_ltm_decode():
    model, vocab, cfg, _ = _tiny_model()
    forms = [[5, 6, 7], [8, 9]]
    sem = torch.randn(2, cfg.data.semantic_dim)
    ok = teacher_forced_naming(model, sem, forms, vocab)
    for i, f in enumerate(forms):
        dec_in = torch.tensor([[vocab.bos_id] + f])
        logits = model.motor(model.ltm.decode_from_s_hat(sem[i:i + 1], dec_in))
        pred = logits.argmax(-1)[0].tolist()
        assert ok[i] == (pred == f + [vocab.eos_id])


# ------------------------------------------- no modification of existing API

def test_probe_does_not_modify_model_classes():
    assert LTMLexicon.decode_from_s_hat is _DECODE_FROM_S_HAT_BEFORE_IMPORT
    assert not hasattr(DualRouteModel, "forward_comprehension")
    assert not hasattr(DualRouteModel, "forward_naming")
    model, vocab, _, _ = _tiny_model()
    # repetition forward untouched: standard call still returns expected keys
    enc_in = torch.tensor([[5, 6, vocab.eos_id]])
    enc_mask = torch.ones_like(enc_in, dtype=torch.bool)
    dec_in = torch.tensor([[vocab.bos_id, 5, 6]])
    out = model(enc_in, enc_mask, dec_in)
    for key in ("logits", "wm_logits", "ltm_logits", "s_hat", "gate"):
        assert key in out


# --------------------------------------- GloVe-coverage fail-fast guard

def test_load_frozen_fails_fast_when_glove_missing(tmp_path):
    """A checkpoint trained WITH GloVe must refuse to load when the rebuilt
    lexicon silently fell back to deterministic pseudo-vectors (missing GloVe
    file): the word-order hash alone cannot detect a wrong semantic bank."""
    import pytest
    from dataclasses import asdict
    from data.lexicon import build_lexicon
    from utils.provenance import sha256_words_ordered as h

    cfg = Config(
        data=DataConfig(use_real=True, glove_path=str(tmp_path / "no_glove.txt"),
                        max_words=200),
        train=TrainConfig(device="cpu"),
    )
    vocab = build_vocab()
    lex = build_lexicon(cfg.data, vocab)   # real lexicon, NO glove -> fallbacks
    assert lex.load_stats is not None and lex.load_stats.n_glove_fallback > 0

    ckpt = {
        "split_mode": "full_lexicon",
        "cfg_data": {**asdict(cfg.data), "split_mode": "full_lexicon",
                     "val_fraction": 0.0},
        "cfg_wm": asdict(cfg.wm),
        "cfg_ltm": asdict(cfg.ltm),
        "cfg_gating": asdict(cfg.gating),
        "cfg_loss": asdict(cfg.loss),
        "cfg_train": asdict(cfg.train),
        "ordered_training_words_sha256": h([e.word for e in lex.entries]),
        # provenance says training had FULL GloVe coverage
        "n_glove_fallback": 0,
        "glove_present": True,
        "premotor_dim": 128,
        "model_state_dict": {},   # never reached
    }
    path = tmp_path / "fake_ckpt.pt"
    torch.save(ckpt, path)
    with pytest.raises(RuntimeError, match="GloVe coverage mismatch"):
        load_frozen(str(path), device="cpu")
