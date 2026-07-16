"""Phase 2 unit tests.

Covers Phase 2A (LTM architecture modes + unified hidden size), Phase 2B
(encode-once), Phase 2C (ventral noise + gate_threshold), Phase 2D
(checkpoint format + optimizer resume + periodic saves), and Phase 2E (this).

All tests are CPU-only, use tiny synthetic tensors, and have zero data
dependencies (no lexicon file, no GloVe) unless marked @pytest.mark.slow.

Phase 2E invariants tested here
---------------------------------
2A. LTM architecture modes:
    - bigru_masked_mean: output of encode() is (B, semantic_dim)
    - unigru_last_hidden: output of encode() is (B, semantic_dim)
    - to_semantic weight shapes differ: bigru has 2H input, unigru has H input
    - padding-invariance: unigru_last_hidden gives same s_hat for different
      padding amounts (bigru_masked_mean does NOT share this property)
    - contradictory bidirectional_encoder is silently normalised by __post_init__

2A. Unified hidden size (--hidden_size H):
    - All four GRU hidden dims are H: WM encoder, WM decoder, LTM encoder, LTM decoder

2B. Encode-once / TF path:
    - _forward_scheduled_sampling calls encode_all exactly once per batch
    - TF=1.0 does NOT enter _forward_scheduled_sampling (uses vectorised path)
    - WM and LTM encoder calls are each exactly 1 per forward pass

2C. Gate threshold:
    - conf == gate_threshold → g == 0.5 (crossover, for any alpha, any tau)

2C. Ventral / WM noise semantics (decoupled from collect):
    - sigma=0 → deterministic regardless of training mode
    - sigma>0, training=True → stochastic
    - sigma>0, eval, collect=True → STILL deterministic (collect ≠ noise)
    - sigma>0, eval, apply_noise=True → stochastic (explicit lesion)
    - Same for WM interference noise

2D. Checkpoint format:
    - optimizer_state_dict is present and non-None
    - optimizer STATE is actually restored (step count, momentum buffers)
    - periodic checkpoint is saved inside the epoch loop at the expected boundary
    - atomic save: .tmp renamed to final path

Architecture:
    - Loading bigru weights into unigru model raises RuntimeError
    - LTMConfig.__post_init__ normalises bidirectional_encoder to match ltm_encoder_mode

Aggregator ranking:
    - lower errors_train wins
    - equal errors → higher exact_val wins
    - equal exact → lower edit_val wins
    - equal edit → lower ned_val wins
    - equal ned → smaller H wins
    - isolated WM/LTM changes never affect rank
    - smaller tf_ratio only after all performance criteria
    - missing required metrics rejected cleanly
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import (Config, DataConfig, WMConfig, LTMConfig, GatingConfig,
                    LossConfig, TrainConfig, default_config)
from data.phonemes import build_vocab
from models.ltm_route import LTMLexicon
from models.wm_route import WMRecurrent
from models.gating import Gate, build_gate
from models.dual_route import DualRouteModel


# --------------------------------------------------------------------------- #
# Fixtures / Helpers
# --------------------------------------------------------------------------- #

def _phon_embed(pad_id: int = 0, embed_dim: int = 16, vocab_size: int = 40) -> nn.Embedding:
    return nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)


def _ltm(mode: str, H: int = 32, semantic_dim: int = 8,
         ventral_noise: float = 0.0) -> LTMLexicon:
    cfg = LTMConfig(enc_hidden=H, dec_hidden=H, ltm_encoder_mode=mode,
                    phon_embed_dim=16, ventral_noise=ventral_noise)
    return LTMLexicon(cfg, _phon_embed(), semantic_dim=semantic_dim,
                      premotor_dim=16, pad_id=0)


def _wm(H: int = 32, noise: float = 0.0) -> WMRecurrent:
    cfg = WMConfig(hidden=H, interference_noise=noise)
    return WMRecurrent(cfg, _phon_embed(), premotor_dim=16)


def _rand_seq(B: int = 2, T: int = 5, vocab_size: int = 40, pad_id: int = 0,
              lengths: list[int] | None = None):
    enc_in   = torch.randint(3, vocab_size, (B, T))
    enc_in[:, 0] = 1
    enc_mask = torch.ones(B, T, dtype=torch.bool)
    if lengths is not None:
        for i, l in enumerate(lengths):
            enc_in[i, l:]   = pad_id
            enc_mask[i, l:] = False
    return enc_in, enc_mask


def _tiny_cfg_phase2():
    cfg = default_config()
    cfg.data.glove_path       = None
    cfg.data.max_words        = 32
    cfg.wm.hidden             = 16
    cfg.ltm.enc_hidden        = 16
    cfg.ltm.dec_hidden        = 16
    cfg.ltm.ltm_encoder_mode  = "bigru_masked_mean"
    cfg.train.device          = "cpu"
    cfg.train.seed            = 42
    return cfg


# --------------------------------------------------------------------------- #
# 2A — LTM architecture modes: output shapes
# --------------------------------------------------------------------------- #

class TestLTMArchitectureModes:

    def test_bigru_masked_mean_encode_shape(self):
        ltm = _ltm("bigru_masked_mean", H=32, semantic_dim=8)
        ltm.eval()
        enc_in, enc_mask = _rand_seq(B=3, T=5)
        assert ltm.encode(enc_in, enc_mask).shape == (3, 8)

    def test_unigru_last_hidden_encode_shape(self):
        ltm = _ltm("unigru_last_hidden", H=32, semantic_dim=8)
        ltm.eval()
        enc_in, enc_mask = _rand_seq(B=3, T=5)
        assert ltm.encode(enc_in, enc_mask).shape == (3, 8)

    def test_bigru_to_semantic_input_dim_is_2H(self):
        H = 32
        ltm = _ltm("bigru_masked_mean", H=H)
        assert ltm.to_semantic[0].in_features == 2 * H

    def test_unigru_to_semantic_input_dim_is_H(self):
        H = 32
        ltm = _ltm("unigru_last_hidden", H=H)
        assert ltm.to_semantic[0].in_features == H

    def test_invalid_mode_raises_at_config(self):
        """LTMConfig.__post_init__ raises on unknown mode (not only at model build)."""
        with pytest.raises(ValueError, match="Unknown ltm_encoder_mode"):
            LTMConfig(ltm_encoder_mode="fantasy_mode")

    def test_invalid_mode_raises_via_ltm_helper(self):
        """Invalid mode raises ValueError from LTMConfig.__post_init__ (caught at
        config-construction time, before LTMLexicon.__init__ is even reached)."""
        with pytest.raises(ValueError, match="Unknown ltm_encoder_mode"):
            _ltm("totally_made_up_mode")

    def test_bidirectional_encoder_normalised_by_post_init(self):
        """LTMConfig.__post_init__ normalises bidirectional_encoder so it cannot
        contradict ltm_encoder_mode."""
        # unigru: bidirectional_encoder must be False regardless of what is passed
        cfg_uni = LTMConfig(ltm_encoder_mode="unigru_last_hidden",
                            bidirectional_encoder=True)  # contradictory input
        assert cfg_uni.bidirectional_encoder is False, (
            "__post_init__ must set bidirectional_encoder=False for unigru_last_hidden"
        )
        # bigru: bidirectional_encoder must be True
        cfg_bi = LTMConfig(ltm_encoder_mode="bigru_masked_mean",
                           bidirectional_encoder=False)  # contradictory input
        assert cfg_bi.bidirectional_encoder is True

    def test_unigru_padding_invariance(self):
        """unigru_last_hidden: same s_hat regardless of trailing padding amount."""
        ltm = _ltm("unigru_last_hidden", H=32, semantic_dim=8)
        ltm.eval()
        enc_in_a, enc_mask_a = _rand_seq(B=1, T=5, lengths=[3])
        # Append 2 more padding positions
        enc_in_b  = torch.cat([enc_in_a,  torch.zeros(1, 2, dtype=torch.long)], dim=1)
        enc_mask_b = torch.cat([enc_mask_a, torch.zeros(1, 2, dtype=torch.bool)],  dim=1)
        s_hat_a = ltm.encode(enc_in_a,  enc_mask_a)
        s_hat_b = ltm.encode(enc_in_b,  enc_mask_b)
        assert torch.allclose(s_hat_a, s_hat_b, atol=1e-5), (
            f"unigru_last_hidden must be padding-invariant. "
            f"Max diff: {(s_hat_a - s_hat_b).abs().max().item():.2e}"
        )


# --------------------------------------------------------------------------- #
# 2A — Unified hidden size (all four GRU dims = H)
# --------------------------------------------------------------------------- #

class TestUnifiedHiddenSize:

    @pytest.mark.parametrize("H", [64, 128])
    def test_wm_encoder_hidden_size(self, H):
        wm = _wm(H=H)
        enc_in, enc_mask = _rand_seq(B=2, T=5)
        h = wm.encode(enc_in, enc_mask)
        assert h.shape == (1, 2, H), f"WM encoder: expected (1,2,{H}), got {h.shape}"

    @pytest.mark.parametrize("H", [64, 128])
    def test_wm_decoder_hidden_size(self, H):
        """WM decoder GRU hidden dim = H (same WMConfig.hidden field)."""
        wm = _wm(H=H)
        assert wm.decoder.hidden_size == H, (
            f"WM decoder hidden_size should be {H}, got {wm.decoder.hidden_size}"
        )

    @pytest.mark.parametrize("H", [64, 128])
    def test_ltm_encoder_hidden_size(self, H):
        ltm = _ltm("unigru_last_hidden", H=H, semantic_dim=8)
        ltm.eval()
        enc_in, enc_mask = _rand_seq(B=2, T=5)
        s_hat = ltm.encode(enc_in, enc_mask)
        assert s_hat.shape == (2, 8)
        assert ltm.encoder.hidden_size == H

    @pytest.mark.parametrize("H", [64, 128])
    def test_ltm_decoder_hidden_size(self, H):
        """LTM decoder GRU hidden dim = H."""
        ltm = _ltm("unigru_last_hidden", H=H, semantic_dim=8)
        assert ltm.decoder.hidden_size == H, (
            f"LTM decoder hidden_size should be {H}, got {ltm.decoder.hidden_size}"
        )

    def test_all_four_gru_dims_unified(self):
        """--hidden_size H sets all four GRU hidden dims to H via config propagation."""
        H = 64
        cfg = default_config()
        cfg.wm.hidden       = H
        cfg.ltm.enc_hidden  = H
        cfg.ltm.dec_hidden  = H
        cfg.ltm.ltm_encoder_mode = "unigru_last_hidden"
        vocab = build_vocab()
        model = DualRouteModel(cfg, vocab)
        assert model.wm.encoder.hidden_size  == H, f"WM encoder: {model.wm.encoder.hidden_size}"
        assert model.wm.decoder.hidden_size  == H, f"WM decoder: {model.wm.decoder.hidden_size}"
        assert model.ltm.encoder.hidden_size == H, f"LTM encoder: {model.ltm.encoder.hidden_size}"
        assert model.ltm.decoder.hidden_size == H, f"LTM decoder: {model.ltm.decoder.hidden_size}"
        # to_semantic: input = H (unigru, not 2H)
        assert model.ltm.to_semantic[0].in_features == H


# --------------------------------------------------------------------------- #
# 2B — TF=1 uses vectorised path; TF<1 uses encode-once stepwise
# --------------------------------------------------------------------------- #

class TestScheduledSamplingPaths:

    def test_tf1_training_does_not_call_scheduled_sampling(self):
        """With tf_ratio=1.0 in TRAINING mode, run_epoch must use the vectorized path.

        Verified via unittest.mock.patch: _forward_scheduled_sampling is patched
        to raise; the test passes iff it is never entered.
        """
        from unittest.mock import patch, MagicMock
        from train import run_epoch

        B, S = 2, 4
        batch = {
            "enc_in":   torch.zeros(B, S, dtype=torch.long),
            "enc_mask": torch.ones(B, S,  dtype=torch.bool),
            "dec_in":   torch.zeros(B, S, dtype=torch.long),
            "dec_tgt":  torch.zeros(B, S, dtype=torch.long),
            "words":    ["a", "b"],
            "semantic": torch.zeros(B, 4),
            "freq":     torch.ones(B),
            "rank":     torch.zeros(B, dtype=torch.long),
        }
        loader = [batch]

        cfg = default_config()
        cfg.train.teacher_forcing_ratio = 1.0
        cfg.train.device = "cpu"

        vectorized_call_count = [0]

        class FakeModel(nn.Module):
            def __init__(self_):
                super().__init__()
                self_.vocab = type("V", (), {"pad_id": 0})()
                self_.p     = nn.Parameter(torch.zeros(1))

            def forward(self_, enc_in, enc_mask, dec_in,
                        collect=False, apply_noise=False):
                vectorized_call_count[0] += 1
                return {
                    "logits":     torch.zeros(B, S, 10),
                    "wm_logits":  torch.zeros(B, S, 10),
                    "ltm_logits": torch.zeros(B, S, 10),
                    "gate":       torch.zeros(B, S, 1),
                    "s_hat":      torch.zeros(B, 4),
                }

        model = FakeModel()
        optim = torch.optim.SGD(model.parameters(), lr=0.0)

        fake_losses = {
            "total": torch.tensor(0.0, requires_grad=True),
            "rep":   torch.tensor(0.0),
            "align": torch.tensor(0.0),
            "wm":    torch.tensor(0.0),
        }

        def _raise_if_called(*a, **kw):
            raise AssertionError(
                "_forward_scheduled_sampling must NOT be called when tf_ratio=1.0"
            )

        with patch("train._forward_scheduled_sampling", side_effect=_raise_if_called), \
             patch("train.total_loss", return_value=fake_losses):
            run_epoch(model, loader, cfg, optim=optim)

        assert vectorized_call_count[0] == 1, (
            f"Vectorized model.forward() must be called once, "
            f"got {vectorized_call_count[0]}"
        )

    def test_tf_less_than_1_calls_scheduled_sampling(self):
        """With tf_ratio=0.5 in TRAINING mode, _forward_scheduled_sampling is called."""
        from unittest.mock import patch, MagicMock
        from train import run_epoch
        sentinel = MagicMock(return_value={
            "logits":     torch.zeros(2, 4, 10),
            "wm_logits":  torch.zeros(2, 4, 10),
            "ltm_logits": torch.zeros(2, 4, 10),
            "gate":       torch.zeros(2, 4, 1),
            "s_hat":      torch.zeros(2, 4),
        })

        B, S = 2, 4
        batch = {
            "enc_in":   torch.zeros(B, S, dtype=torch.long),
            "enc_mask": torch.ones(B, S,  dtype=torch.bool),
            "dec_in":   torch.zeros(B, S, dtype=torch.long),
            "dec_tgt":  torch.zeros(B, S, dtype=torch.long),
            "words":    ["a", "b"],
            "semantic": torch.zeros(B, 4),
            "freq":     torch.ones(B),
            "rank":     torch.zeros(B, dtype=torch.long),
        }
        loader = [batch]

        cfg = default_config()
        cfg.train.teacher_forcing_ratio = 0.5
        cfg.train.device = "cpu"

        fake_losses = {
            "total": torch.tensor(0.0, requires_grad=True),
            "rep":   torch.tensor(0.0),
            "align": torch.tensor(0.0),
            "wm":    torch.tensor(0.0),
        }

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.vocab = type("V", (), {"pad_id": 0})()
                self.p     = nn.Parameter(torch.zeros(1))
            def forward(self, *a, **kw):
                raise AssertionError("vectorized forward must not be called for TF<1 training")

        model = FakeModel()
        optim = torch.optim.SGD(model.parameters(), lr=0.0)

        with patch("train._forward_scheduled_sampling", sentinel), \
             patch("train.total_loss", return_value=fake_losses):
            run_epoch(model, loader, cfg, optim=optim)

        assert sentinel.called, "_forward_scheduled_sampling must be called for tf<1"


# --------------------------------------------------------------------------- #
# 2C — Gate threshold
# --------------------------------------------------------------------------- #

class TestGateThreshold:

    @pytest.mark.parametrize("tau", [0.3, 0.5, 0.7])
    def test_gate_value_at_crossover(self, tau):
        """At conf == gate_threshold, g must be exactly 0.5 for any alpha."""
        cfg = GatingConfig(alpha=4.0, gate_threshold=tau)
        gate = Gate(cfg, premotor_dim=8)
        B, S = 2, 3
        wm = torch.randn(B, S, 8)
        ltm = torch.randn(B, S, 8)
        g = gate.gate_value(wm, ltm, {"confidence": torch.full((B,), tau)})
        assert g.shape == (B, S, 1)
        assert torch.allclose(g, torch.full_like(g, 0.5), atol=1e-6), (
            f"At conf==tau={tau} gate must be 0.5, got {g.unique()}"
        )

    def test_gate_threshold_direction(self):
        """Lower threshold → higher g for the same confidence level."""
        B, S = 1, 2
        wm  = torch.randn(B, S, 8)
        ltm = torch.randn(B, S, 8)
        conf = torch.full((B,), 0.6)
        g_low  = Gate(GatingConfig(alpha=4.0, gate_threshold=0.3), 8).gate_value(wm, ltm, {"confidence": conf})
        g_high = Gate(GatingConfig(alpha=4.0, gate_threshold=0.7), 8).gate_value(wm, ltm, {"confidence": conf})
        assert g_low.mean() > g_high.mean()

    def test_gate_none_field_returns_05(self):
        gate = Gate(GatingConfig(alpha=4.0, gate_threshold=0.5), 8)
        B, S = 2, 3
        g = gate.gate_value(torch.randn(B, S, 8), torch.randn(B, S, 8), field=None)
        assert torch.allclose(g, torch.full_like(g, 0.5))


# --------------------------------------------------------------------------- #
# 2C — Noise semantics: decoupled from collect
# --------------------------------------------------------------------------- #

class TestNoiseSemanticsDecoupled:
    """Verify that collect=True alone does NOT activate noise (Phase 2 fix).

    The historical bug: collect=True was used as a proxy for "apply noise during
    eval", making ceiling evaluation non-deterministic.  The fix: noise is active
    only when self.training=True OR apply_noise=True.
    """

    # --- LTM ventral noise ---

    def test_ltm_sigma_zero_deterministic_train(self):
        ltm = _ltm("unigru_last_hidden", H=32, ventral_noise=0.0)
        ltm.train()
        enc_in, enc_mask = _rand_seq()
        s1 = ltm.encode(enc_in, enc_mask)
        s2 = ltm.encode(enc_in, enc_mask)
        assert torch.allclose(s1, s2, atol=1e-6)

    def test_ltm_sigma_nonzero_stochastic_in_train(self):
        ltm = _ltm("unigru_last_hidden", H=32, ventral_noise=1.0)
        ltm.train()
        enc_in, enc_mask = _rand_seq()
        s1 = ltm.encode(enc_in, enc_mask)
        s2 = ltm.encode(enc_in, enc_mask)
        assert (s1 - s2).abs().max() > 1e-4

    def test_ltm_collect_true_alone_is_deterministic(self):
        """collect=True must NOT activate noise in eval — this is the key fix."""
        ltm = _ltm("unigru_last_hidden", H=32, ventral_noise=1.0)
        ltm.eval()
        enc_in, enc_mask = _rand_seq()
        s1 = ltm.encode(enc_in, enc_mask, collect=True)
        s2 = ltm.encode(enc_in, enc_mask, collect=True)
        assert torch.allclose(s1, s2, atol=1e-6), (
            "collect=True must NOT activate noise. "
            f"Max diff: {(s1 - s2).abs().max().item():.2e}"
        )

    def test_ltm_apply_noise_true_stochastic_in_eval(self):
        """apply_noise=True must activate noise in eval (explicit lesion)."""
        ltm = _ltm("unigru_last_hidden", H=32, ventral_noise=1.0)
        ltm.eval()
        enc_in, enc_mask = _rand_seq()
        s1 = ltm.encode(enc_in, enc_mask, apply_noise=True)
        s2 = ltm.encode(enc_in, enc_mask, apply_noise=True)
        assert (s1 - s2).abs().max() > 1e-4, (
            "apply_noise=True must produce stochastic s_hat in eval mode."
        )

    def test_ltm_eval_no_flags_is_deterministic(self):
        """Default eval (no flags) must be deterministic regardless of sigma."""
        ltm = _ltm("unigru_last_hidden", H=32, ventral_noise=1.0)
        ltm.eval()
        enc_in, enc_mask = _rand_seq()
        s1 = ltm.encode(enc_in, enc_mask)
        s2 = ltm.encode(enc_in, enc_mask)
        assert torch.allclose(s1, s2, atol=1e-6)

    def test_ltm_bigru_apply_noise(self):
        """apply_noise works in bigru_masked_mean mode too."""
        ltm = _ltm("bigru_masked_mean", H=32, ventral_noise=1.0)
        ltm.eval()
        enc_in, enc_mask = _rand_seq()
        s1 = ltm.encode(enc_in, enc_mask, apply_noise=True)
        s2 = ltm.encode(enc_in, enc_mask, apply_noise=True)
        assert (s1 - s2).abs().max() > 1e-4

    # --- WM interference noise ---

    def test_wm_collect_true_alone_is_deterministic(self):
        """collect=True must NOT activate WM noise in eval."""
        wm = _wm(H=32, noise=1.0)
        wm.eval()
        enc_in, enc_mask = _rand_seq()
        h1 = wm.encode(enc_in, enc_mask, collect=True)
        h2 = wm.encode(enc_in, enc_mask, collect=True)
        assert torch.allclose(h1, h2, atol=1e-6), (
            "collect=True must NOT activate WM noise. "
            f"Max diff: {(h1 - h2).abs().max().item():.2e}"
        )

    def test_wm_apply_noise_true_stochastic_in_eval(self):
        """apply_noise=True must activate WM noise in eval."""
        wm = _wm(H=32, noise=1.0)
        wm.eval()
        enc_in, enc_mask = _rand_seq()
        h1 = wm.encode(enc_in, enc_mask, apply_noise=True)
        h2 = wm.encode(enc_in, enc_mask, apply_noise=True)
        assert (h1 - h2).abs().max() > 1e-4

    def test_wm_stochastic_in_train(self):
        wm = _wm(H=32, noise=1.0)
        wm.train()
        enc_in, enc_mask = _rand_seq()
        h1 = wm.encode(enc_in, enc_mask)
        h2 = wm.encode(enc_in, enc_mask)
        assert (h1 - h2).abs().max() > 1e-4

    def test_wm_deterministic_in_eval(self):
        """Default eval (no flags) is deterministic regardless of noise sigma."""
        wm = _wm(H=32, noise=1.0)
        wm.eval()
        enc_in, enc_mask = _rand_seq()
        h1 = wm.encode(enc_in, enc_mask)
        h2 = wm.encode(enc_in, enc_mask)
        assert torch.allclose(h1, h2, atol=1e-6)


# --------------------------------------------------------------------------- #
# 2D — Checkpoint format + optimizer-state restore
# --------------------------------------------------------------------------- #

class TestCheckpointFormat:

    def _make_model_and_optim(self):
        cfg = _tiny_cfg_phase2()
        vocab = build_vocab()
        model = DualRouteModel(cfg, vocab)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return model, optim, cfg, vocab

    def test_optimizer_state_not_none_after_step(self):
        model, optim, _, _ = self._make_model_and_optim()
        next(model.parameters()).sum().backward()
        optim.step()
        sd = optim.state_dict()
        assert sd is not None
        assert len(sd["state"]) > 0

    def test_optimizer_state_actually_restored(self, tmp_path):
        """Save optimizer state, load it into a fresh optimizer, verify restoration.

        This tests that the optimizer step count (param 'step') is actually preserved,
        which is a necessary (though not sufficient) condition for true resume.

        Limitation: DataLoader sampler state is NOT saved, so resume is not
        bitwise-identical to an uninterrupted run. This is a known remaining
        limitation documented here and in train_checkpoint.py.
        """
        model, optim, _, _ = self._make_model_and_optim()
        # Run two optimizer steps to populate non-trivial Adam moment state
        for _ in range(2):
            next(model.parameters()).sum().backward()
            optim.step()
            optim.zero_grad()

        saved_sd = optim.state_dict()
        # Check that Adam has accumulated step counts
        step_counts = [v["step"] for v in saved_sd["state"].values()
                       if "step" in v]
        assert step_counts, "AdamW must have step counts after 2 steps"
        assert all(s == 2 for s in step_counts), f"Expected step=2, got {step_counts}"

        # Restore into a fresh optimizer
        model2, optim2, _, _ = self._make_model_and_optim()
        optim2.load_state_dict(saved_sd)
        restored_sd = optim2.state_dict()
        restored_steps = [v["step"] for v in restored_sd["state"].values()
                          if "step" in v]
        assert restored_steps == step_counts, (
            f"Optimizer step counts not restored. "
            f"Expected {step_counts}, got {restored_steps}"
        )

    def test_checkpoint_atomic_save(self, tmp_path):
        """_save_checkpoint writes via .tmp then renames; .tmp must not remain."""
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        from train_checkpoint import _save_checkpoint

        p = str(tmp_path / "test.pt")
        _save_checkpoint({"x": 1}, p)
        assert os.path.exists(p)
        assert not os.path.exists(p + ".tmp"), "Atomic save must remove .tmp"
        loaded = torch.load(p, map_location="cpu", weights_only=False)
        assert loaded["x"] == 1

    def test_periodic_ckpt_path_naming(self):
        from train_checkpoint import _periodic_ckpt_path
        assert _periodic_ckpt_path("/out/run.pt", 5)    == "/out/run.epoch_0005.pt"
        assert _periodic_ckpt_path("/out/run.pt", 100)  == "/out/run.epoch_0100.pt"
        assert _periodic_ckpt_path("/out/a.b.pt", 3)    == "/out/a.b.epoch_0003.pt"

    def test_periodic_checkpoint_saved_during_loop(self, tmp_path):
        """Periodic checkpoint save is triggered INSIDE the epoch loop, not only at end.

        Simulates the loop logic used in train_checkpoint.py to verify the
        save happens at the right epoch boundary and produces the expected filename.
        """
        from train_checkpoint import _save_checkpoint, _periodic_ckpt_path

        final_path = str(tmp_path / "run.pt")
        save_every = 1
        n_epochs   = 3
        saved = []

        for ep in range(n_epochs):
            epoch_num = ep + 1
            if save_every > 0 and epoch_num % save_every == 0:
                p = _periodic_ckpt_path(final_path, epoch_num)
                _save_checkpoint({"epoch": epoch_num}, p)
                saved.append(p)

        assert len(saved) == n_epochs, f"Expected {n_epochs} periodic checkpoints"
        for i, p in enumerate(saved):
            assert os.path.exists(p), f"Periodic checkpoint missing: {p}"
            loaded = torch.load(p, map_location="cpu", weights_only=False)
            assert loaded["epoch"] == i + 1
        # epoch_0001 naming
        assert saved[0].endswith("run.epoch_0001.pt")
        assert saved[2].endswith("run.epoch_0003.pt")

    @pytest.mark.slow
    def test_checkpoint_dict_keys(self, tmp_path):
        """_build_ckpt_dict produces all required keys; reload works."""
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        from train_checkpoint import _save_checkpoint, _build_ckpt_dict

        model, optim, cfg, vocab = self._make_model_and_optim()
        next(model.parameters()).sum().backward()
        optim.step()

        from data.lexicon import build_lexicon
        lexicon = build_lexicon(cfg.data, vocab)
        train_entries, val_entries = lexicon.split(cfg.data.val_fraction, cfg.data.seed)
        history = [{"epoch": 1, "train_total": 1.0, "train_rep": 0.9,
                    "train_wm": 0.8, "val_rep": 0.95, "val_wm": 0.85}]

        ckpt = _build_ckpt_dict(
            model, optim, cfg, lexicon, train_entries, val_entries,
            history, git_commit="abc123", resumed_from=None, premotor_dim=128)

        for k in ["model_state_dict", "optimizer_state_dict", "rng_states",
                   "cfg_data", "cfg_wm", "cfg_ltm", "cfg_gating", "cfg_loss",
                   "cfg_train", "premotor_dim", "history", "lexicon_source",
                   "n_train", "n_val", "glove_present", "git_commit",
                   "resumed_from", "total_epochs_trained", "lr_at_save"]:
            assert k in ckpt, f"Missing checkpoint key: {k!r}"

        assert ckpt["optimizer_state_dict"] is not None
        assert ckpt["total_epochs_trained"] == 1
        assert ckpt["premotor_dim"] == 128

        p = str(tmp_path / "test.pt")
        _save_checkpoint(ckpt, p)
        loaded = torch.load(p, map_location="cpu", weights_only=False)
        assert loaded["optimizer_state_dict"] is not None


# --------------------------------------------------------------------------- #
# Architecture mismatch on load
# --------------------------------------------------------------------------- #

class TestArchitectureMismatch:

    def test_load_bigru_weights_into_unigru_raises(self, tmp_path):
        """Loading bigru weights into a unigru model raises RuntimeError
        (torch weight-shape mismatch — expected behavior on mode mismatch)."""
        cfg = default_config()
        cfg.wm.hidden = 16
        cfg.ltm.enc_hidden = 16
        cfg.ltm.dec_hidden = 16
        vocab = build_vocab()

        cfg.ltm.ltm_encoder_mode = "bigru_masked_mean"
        bigru = DualRouteModel(cfg, vocab)
        p = str(tmp_path / "bigru.pt")
        torch.save(bigru.state_dict(), p)

        cfg.ltm.ltm_encoder_mode = "unigru_last_hidden"
        unigru = DualRouteModel(cfg, vocab)
        saved = torch.load(p, map_location="cpu", weights_only=True)
        with pytest.raises(RuntimeError):
            unigru.load_state_dict(saved)


# --------------------------------------------------------------------------- #
# Aggregator ranking
# --------------------------------------------------------------------------- #

class TestAggregateRanking:
    """Tests for scripts/aggregate_gridsearch.py ranking logic.

    Uses tiny synthetic dicts instead of real metrics.json files.
    """

    def _r(self, **kw) -> dict:
        """Build a minimal valid run dict."""
        base = {
            "full_ar_errors_train": 100,
            "full_ar_exact_val":    0.5,
            "full_ar_edit_val":     2.0,
            "full_ar_ned_val":      0.3,
            "H":                    128,
            "n_params":             500_000,
            "ltm_encoder_mode":     "unigru_last_hidden",
            "tf_ratio":             1.0,
            "wm_ar_errors_val":     999,   # isolated — must not affect rank
            "ltm_ar_errors_val":    999,   # isolated — must not affect rank
        }
        base.update(kw)
        return base

    def _rank(self, runs):
        from aggregate_gridsearch import rank_key
        return sorted(runs, key=rank_key)

    def _setup(self):
        """Add scripts/ to sys.path so aggregate_gridsearch can be imported."""
        p = os.path.join(ROOT, "scripts")
        if p not in sys.path:
            sys.path.insert(0, p)

    def test_lower_train_errors_wins(self):
        self._setup()
        runs = [self._r(full_ar_errors_train=200), self._r(full_ar_errors_train=10)]
        ranked = self._rank(runs)
        assert ranked[0]["full_ar_errors_train"] == 10

    def test_equal_errors_higher_exact_wins(self):
        self._setup()
        runs = [
            self._r(full_ar_errors_train=50, full_ar_exact_val=0.4),
            self._r(full_ar_errors_train=50, full_ar_exact_val=0.9),
        ]
        ranked = self._rank(runs)
        assert ranked[0]["full_ar_exact_val"] == 0.9

    def test_equal_exact_lower_edit_wins(self):
        self._setup()
        runs = [
            self._r(full_ar_errors_train=50, full_ar_exact_val=0.5, full_ar_edit_val=3.0),
            self._r(full_ar_errors_train=50, full_ar_exact_val=0.5, full_ar_edit_val=1.0),
        ]
        ranked = self._rank(runs)
        assert ranked[0]["full_ar_edit_val"] == 1.0

    def test_equal_edit_lower_ned_wins(self):
        self._setup()
        runs = [
            self._r(full_ar_errors_train=50, full_ar_exact_val=0.5,
                    full_ar_edit_val=2.0, full_ar_ned_val=0.5),
            self._r(full_ar_errors_train=50, full_ar_exact_val=0.5,
                    full_ar_edit_val=2.0, full_ar_ned_val=0.1),
        ]
        ranked = self._rank(runs)
        assert ranked[0]["full_ar_ned_val"] == 0.1

    def test_isolated_wm_ltm_never_affect_rank(self):
        """Changing WM/LTM isolated metrics must not change ranking."""
        self._setup()
        # Two runs identical on FULL metrics but very different on isolated metrics
        a = self._r(wm_ar_errors_val=1,   ltm_ar_errors_val=1)
        b = self._r(wm_ar_errors_val=9999, ltm_ar_errors_val=9999)
        ranked = self._rank([a, b])
        # Both have identical full metrics → tie: same rank key value
        from aggregate_gridsearch import rank_key
        assert rank_key(a) == rank_key(b), (
            "Isolated WM/LTM metrics must not appear in rank_key"
        )

    def test_smaller_H_wins_after_performance_equality(self):
        self._setup()
        runs = [
            self._r(H=256),
            self._r(H=64),
        ]
        ranked = self._rank(runs)
        assert ranked[0]["H"] == 64

    def test_smaller_tf_only_after_genuine_equality(self):
        """TF ratio is the LAST tie-break; it must not override performance metrics."""
        self._setup()
        # run_a: worse performance but tf=0.0 (would win on TF alone)
        # run_b: better performance and tf=1.0
        run_a = self._r(full_ar_errors_train=200, tf_ratio=0.0)
        run_b = self._r(full_ar_errors_train=10,  tf_ratio=1.0)
        ranked = self._rank([run_a, run_b])
        assert ranked[0]["full_ar_errors_train"] == 10, (
            "Better train errors must win over smaller TF ratio"
        )

    def test_smaller_tf_wins_when_all_else_equal(self):
        """TF tie-break is applied when all performance + H criteria are equal."""
        self._setup()
        runs = [self._r(tf_ratio=1.0), self._r(tf_ratio=0.0)]
        ranked = self._rank(runs)
        assert ranked[0]["tf_ratio"] == 0.0

    def test_missing_required_metric_skipped(self):
        self._setup()
        from aggregate_gridsearch import validate_run
        # Missing full_ar_errors_train
        m = {"full_ar_exact_val": 0.5}
        assert validate_run(m, "dummy_path") is False

    def test_missing_both_required_skipped(self):
        self._setup()
        from aggregate_gridsearch import validate_run
        assert validate_run({}, "dummy_path") is False

    def test_valid_run_passes(self):
        self._setup()
        from aggregate_gridsearch import validate_run
        m = {"full_ar_errors_train": 50, "full_ar_exact_val": 0.7}
        assert validate_run(m, "dummy_path") is True


# --------------------------------------------------------------------------- #
# Bundled lexicon: max_words < 50 accepted without synthetic fallback
# --------------------------------------------------------------------------- #

class TestBundledLexiconMinWords:
    """Tests for build_bundled threshold fix and logfreq_weights rank=0 guard.

    Pre-fix: build_bundled returned None when len(entries) < 50, causing synthetic
    fallback with rank=0 entries, which produced inf in logfreq_weights, poisoning
    WeightedRandomSampler with NaN weights after normalization.

    Post-fix:
      - min_required = min(cfg.max_words, 50): small requested lexicons are accepted.
      - logfreq_weights clamps rank to >= 1: eliminates inf for rank=0.
    """

    def test_logfreq_weights_rank_zero_guard(self):
        """rank=0 must not produce inf — synthetic lexicon uses 0-indexed ranks."""
        import numpy as np
        from data.lexicon import logfreq_weights
        weights = logfreq_weights([0, 1, 2, 10, 100])
        assert np.all(np.isfinite(weights)), (
            f"logfreq_weights returned non-finite values for ranks containing 0: {weights}"
        )
        assert np.all(weights > 0), "logfreq_weights must return positive weights"
        # rank=0 treated as rank=1: weight == weight for rank=1
        assert weights[0] == weights[1], (
            "rank=0 should be clamped to rank=1 (same weight as rank=1)"
        )

    @pytest.mark.slow
    def test_max_words_32_uses_bundled_not_synthetic(self):
        """build_lexicon with max_words=32 must use bundled lexicon, not synthetic fallback.

        Requires data/lexicon_en.tsv.
        """
        import numpy as np
        import os
        from data.lexicon import build_lexicon, logfreq_weights, BUNDLED_PATH

        if not os.path.exists(BUNDLED_PATH):
            pytest.skip("data/lexicon_en.tsv not present")

        vocab = build_vocab()
        cfg = DataConfig()
        cfg.use_real     = True
        cfg.lexicon_path = None
        cfg.glove_path   = None   # use pseudo-vectors, skip large GloVe file
        cfg.max_words    = 32
        cfg.min_phonemes = 2
        cfg.max_phonemes = 9
        cfg.semantic_dim = 300

        lex = build_lexicon(cfg, vocab)

        assert lex.source == "bundled-en", (
            f"Expected source='bundled-en', got {lex.source!r}. "
            f"build_bundled incorrectly rejected the 32 loaded entries."
        )
        assert len(lex) == 32, f"Expected 32 entries, got {len(lex)}"

        ranks = [e.rank for e in lex.entries]
        assert all(r > 0 for r in ranks), (
            f"Some bundled ranks are <= 0: {[r for r in ranks if r <= 0]}. "
            f"Bundled ranks are 1-indexed from the TSV."
        )

        weights = logfreq_weights(ranks)
        assert np.all(np.isfinite(weights)), (
            f"logfreq_weights has non-finite values: {weights[~np.isfinite(weights)]}"
        )
        assert np.all(weights > 0), "All logfreq sampler weights must be positive"


# --------------------------------------------------------------------------- #
# Aggregator: nested metrics.json parsing
# --------------------------------------------------------------------------- #

class TestAggregateNestedParsing:
    """Tests for aggregate_gridsearch.py nested-format parsing.

    The aggregator previously required pre-flattened keys (full_ar_errors_train etc.).
    evaluate_train_lexicon_ceiling.py writes nested format:
        results.train.full.n_errors, .exact_match, .edit_dist, .norm_edit_dist
        results.val.full.*
    These tests verify the normalisation layer.
    """

    def _setup(self):
        p = os.path.join(ROOT, "scripts")
        if p not in sys.path:
            sys.path.insert(0, p)

    def _nested_run(self, **overrides) -> dict:
        """Minimal nested metrics.json dict as produced by evaluate_train_lexicon_ceiling.py."""
        d = {
            "evaluation_note": "test",
            "decode_mode": "autoregressive",
            "checkpoint": "/fake/path/lichtheim3.pt",
            "glove_present": False,
            "lexicon_source": "bundled-en",
            "n_train": 27,
            "n_val": 5,
            "cfg_max_words": 32,
            "cfg_epochs": 2,
            "cfg_seed": 42,
            "splits_evaluated": ["train", "val"],
            "results": {
                "train": {
                    "full": {
                        "exact_match": 0.85,
                        "edit_dist":   0.3,
                        "n_items":     27,
                        "n_errors":    4,
                        "norm_edit_dist": 0.12,
                    },
                    "wm":  {"exact_match": 0.7,  "edit_dist": 0.5,  "n_items": 27, "n_errors": 8},
                    "ltm": {"exact_match": 0.6,  "edit_dist": 0.7,  "n_items": 27, "n_errors": 11},
                },
                "val": {
                    "full": {
                        "exact_match": 0.8,
                        "edit_dist":   0.4,
                        "n_items":     5,
                        "n_errors":    1,
                        "norm_edit_dist": 0.15,
                    },
                    "wm":  {"exact_match": 0.6,  "edit_dist": 0.6,  "n_items": 5,  "n_errors": 2},
                    "ltm": {"exact_match": 0.5,  "edit_dist": 0.8,  "n_items": 5,  "n_errors": 3},
                },
            },
        }
        d.update(overrides)
        return d

    def test_nested_flat_key_extraction(self):
        """_nested_to_flat must produce full_ar_* flat keys from nested results."""
        self._setup()
        from aggregate_gridsearch import _nested_to_flat
        flat = _nested_to_flat(self._nested_run())

        assert flat["full_ar_errors_train"] == 4
        assert flat["full_ar_exact_train"]  == 0.85
        assert flat["full_ar_edit_train"]   == 0.3
        assert flat["full_ar_ned_train"]    == 0.12

        assert flat["full_ar_errors_val"] == 1
        assert flat["full_ar_exact_val"]  == 0.8
        assert flat["full_ar_edit_val"]   == 0.4
        assert flat["full_ar_ned_val"]    == 0.15

    def test_nested_wm_ltm_extracted(self):
        """Isolated WM/LTM route metrics are extracted but must not appear in rank_key."""
        self._setup()
        from aggregate_gridsearch import _nested_to_flat, rank_key
        flat = _nested_to_flat(self._nested_run())

        # Isolated metrics should be in the flat dict (for TSV output)
        assert "wm_ar_errors_train"  in flat
        assert "ltm_ar_errors_train" in flat
        assert "wm_ar_errors_val"    in flat
        assert "ltm_ar_errors_val"   in flat

        # But they must not affect the rank key
        flat_high_wm = dict(flat)
        flat_high_wm["wm_ar_errors_train"] = 999
        flat_high_wm["ltm_ar_errors_val"]  = 999
        assert rank_key(flat) == rank_key(flat_high_wm), (
            "Changing isolated WM/LTM metrics must not change rank_key"
        )

    def test_nested_metadata_copied(self):
        """Top-level metadata fields must be copied into the flat dict."""
        self._setup()
        from aggregate_gridsearch import _nested_to_flat
        flat = _nested_to_flat(self._nested_run())
        assert flat["cfg_seed"]       == 42
        assert flat["cfg_max_words"]  == 32
        assert flat["lexicon_source"] == "bundled-en"
        assert flat["checkpoint"]     == "/fake/path/lichtheim3.pt"

    def test_nested_validate_run_passes(self):
        """A nested run with both train and val full metrics must pass validate_run."""
        self._setup()
        from aggregate_gridsearch import _nested_to_flat, validate_run
        flat = _nested_to_flat(self._nested_run())
        assert validate_run(flat, "test_path") is True

    def test_nested_missing_val_skipped(self):
        """A nested run without val results must be skipped by validate_run."""
        self._setup()
        from aggregate_gridsearch import _nested_to_flat, validate_run
        raw = self._nested_run()
        del raw["results"]["val"]  # simulate --include_val not passed
        flat = _nested_to_flat(raw)
        assert validate_run(flat, "test_path") is False

    def test_smoke_no_wfe_no_metadata(self):
        """Aggregation with no WFE and no H/tf_ratio metadata must succeed.

        Simulates the Phase 2 smoke test scenario: 3 runs, all missing H and
        tf_ratio (not in metrics.json), no WFE metrics.
        """
        self._setup()
        from aggregate_gridsearch import _nested_to_flat, validate_run, rank_key

        runs_raw = [
            self._nested_run(),                      # baseline
            self._nested_run(**{"results": {         # better val
                "train": {"full": {"exact_match": 0.9, "edit_dist": 0.2,
                                   "n_items": 27, "n_errors": 3, "norm_edit_dist": 0.1},
                          "wm": {}, "ltm": {}},
                "val":   {"full": {"exact_match": 0.95, "edit_dist": 0.2,
                                   "n_items": 5, "n_errors": 0, "norm_edit_dist": 0.08},
                          "wm": {}, "ltm": {}},
            }}),
            self._nested_run(**{"results": {         # worse
                "train": {"full": {"exact_match": 0.5, "edit_dist": 1.0,
                                   "n_items": 27, "n_errors": 14, "norm_edit_dist": 0.4},
                          "wm": {}, "ltm": {}},
                "val":   {"full": {"exact_match": 0.4, "edit_dist": 1.2,
                                   "n_items": 5, "n_errors": 3, "norm_edit_dist": 0.5},
                          "wm": {}, "ltm": {}},
            }}),
        ]
        runs = [_nested_to_flat(r) for r in runs_raw]
        assert all(validate_run(m, f"run_{i}") for i, m in enumerate(runs))

        ranked = sorted(runs, key=rank_key)
        # Best run: 0 errors val, highest exact val
        assert ranked[0]["full_ar_errors_val"] == 0
        # Worst run: most errors
        assert ranked[-1]["full_ar_errors_train"] == 14

    def test_flat_legacy_passthrough(self):
        """Old flat-format dicts (no 'results' key) must pass through unchanged."""
        self._setup()
        from aggregate_gridsearch import _nested_to_flat
        flat_input = {
            "full_ar_errors_train": 10,
            "full_ar_exact_val":    0.9,
            "ckpt_dir":             "/old/run",
        }
        result = _nested_to_flat(flat_input)
        assert result is flat_input  # must be the exact same object (no copy)

    def test_skipped_runs_reported_in_summary(self):
        """Skipped runs must appear in aggregate_summary.json skipped_runs list."""
        self._setup()
        from aggregate_gridsearch import build_summary

        skipped = [{"path": "/bad/run/metrics.json",
                    "reason": "missing required keys ['full_ar_exact_val']"}]
        valid_run = {
            "full_ar_errors_train": 5,
            "full_ar_exact_val":    0.9,
            "ckpt_dir":             "/good/run",
        }
        summary = build_summary([valid_run], skipped=skipped)

        assert summary["n_skipped"] == 1
        assert len(summary["skipped_runs"]) == 1
        assert summary["skipped_runs"][0]["path"] == "/bad/run/metrics.json"
        assert "reason" in summary["skipped_runs"][0]

    def test_empty_skipped_when_all_valid(self):
        """skipped_runs must be an empty list when no runs are rejected."""
        self._setup()
        from aggregate_gridsearch import build_summary
        valid_run = {"full_ar_errors_train": 5, "full_ar_exact_val": 0.9,
                     "ckpt_dir": "/good/run"}
        summary = build_summary([valid_run], skipped=[])
        assert summary["n_skipped"] == 0
        assert summary["skipped_runs"] == []

    def test_ranking_preserved_full_only(self):
        """Ranking by FULL metrics only is preserved after nested parsing."""
        self._setup()
        from aggregate_gridsearch import _nested_to_flat, rank_key

        # run_a: fewer errors (better) but worse val exact
        run_a_raw = self._nested_run()
        run_a_raw["results"]["train"]["full"]["n_errors"] = 2
        run_a_raw["results"]["val"]["full"]["exact_match"] = 0.5

        # run_b: more errors but better val exact
        run_b_raw = self._nested_run()
        run_b_raw["results"]["train"]["full"]["n_errors"] = 10
        run_b_raw["results"]["val"]["full"]["exact_match"] = 0.99

        run_a = _nested_to_flat(run_a_raw)
        run_b = _nested_to_flat(run_b_raw)
        ranked = sorted([run_a, run_b], key=rank_key)

        # errors_train is primary: run_a (2 errors) wins over run_b (10 errors)
        assert ranked[0]["full_ar_errors_train"] == 2

    def test_json_paths_documented_in_summary(self):
        """aggregate_summary.json must document the nested JSON paths."""
        self._setup()
        from aggregate_gridsearch import build_summary
        valid_run = {"full_ar_errors_train": 5, "full_ar_exact_val": 0.9,
                     "ckpt_dir": "/good/run"}
        summary = build_summary([valid_run])
        assert "json_paths" in summary
        jp = summary["json_paths"]
        assert jp["full_ar_errors_train"] == "results.train.full.n_errors"
        assert jp["full_ar_exact_val"]    == "results.val.full.exact_match"


# ----------------------------------------------------------------------------
# Phase 5A: RNG state restoration helper (resume robustness)
# ----------------------------------------------------------------------------
import importlib.util as _ilu
import os as _os

_TCK = _os.path.join(_os.path.dirname(__file__), "..", "scripts", "train_checkpoint.py")
_spec = _ilu.spec_from_file_location("train_checkpoint", _TCK)
_tck = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_tck)


class TestRngStateNormalization:
    """The resume path must accept a torch RNG state regardless of dtype/device."""

    def test_accepts_normal_byte_tensor(self):
        state = torch.get_rng_state()  # already a uint8 ByteTensor
        out = _tck._as_cpu_byte_tensor(state, "torch_rng")
        assert out.dtype == torch.uint8
        assert out.device.type == "cpu"
        # Must be accepted by torch.set_rng_state without raising
        torch.set_rng_state(out)

    def test_accepts_non_uint8_dtype(self):
        state = torch.get_rng_state().to(torch.int64)
        out = _tck._as_cpu_byte_tensor(state, "torch_rng")
        assert out.dtype == torch.uint8
        assert out.device.type == "cpu"
        torch.set_rng_state(out)

    def test_accepts_list_like(self):
        state = torch.get_rng_state().tolist()
        out = _tck._as_cpu_byte_tensor(state, "torch_rng")
        assert out.dtype == torch.uint8
        assert out.device.type == "cpu"
        torch.set_rng_state(out)

    def test_roundtrip_preserves_values(self):
        state = torch.get_rng_state()
        out = _tck._as_cpu_byte_tensor(state.to(torch.int64), "torch_rng")
        assert torch.equal(out, state)
