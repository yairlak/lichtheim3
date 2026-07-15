"""Dorsal route: a parametric recurrent serial-recall network.

After Botvinick & Plaut (2006), the phonological working-memory / dorsal route is
a learned **encoder -> decoder** recurrent network, not a hand-wired copy buffer.
It reads the phoneme sequence into a bounded recurrent state and re-articulates
it, so it:

  * learns the auditory->motor map and the language's phonotactics (it is
    parametric and trained, like the real dorsal pathway);
  * generalizes to novel words and nonwords (it acquires the *general*
    serial-recall computation);
  * is capacity- and length-limited, because the whole sequence must be packed
    into a fixed-size recurrent state, and interference noise corrupts it -- this
    is what produces length effects and the serial-position curve;
  * stays lexical-frequency-invariant, because a generalizing route gains nothing
    from word identity (and it is additionally trained on a frequency-flat stream
    of pronounceable pseudowords; see `config.TrainConfig.dorsal_pool_size`).

Same forward contract as before -- `forward(enc_in, enc_mask, dec_in) ->
{"premotor": (B, S, premotor_dim)}` -- so the gate, motor, and lesion hooks are
unchanged.

Encode-once interface
---------------------
`encode(enc_in, enc_mask, collect, apply_noise)` draws the noise ONCE and returns h.
`decode_from_state(h, dec_in)` decodes from a pre-computed state.
`forward(...)` calls both (convenience wrapper, preserves backward compat).

This separation is used by DualRouteModel.encode_all + decode_from_states for
the scheduled-sampling training path, ensuring that WM interference noise is
drawn exactly once per stimulus per forward pass regardless of how many decoder
steps are unrolled.

Noise semantics (explicit)
--------------------------
Noise is active iff (self.training OR apply_noise) AND cfg.interference_noise > 0.
`collect=True` alone does NOT activate noise — it only means "collect and return
intermediate representations". This decoupling prevents the historical evaluation
bug where ceiling evaluation became non-deterministic when collect=True was used
to enable noise during eval. For explicit noisy evaluation (lesion / robustness
tests), pass apply_noise=True directly.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from config import WMConfig


class WMRecurrent(nn.Module):
    def __init__(self, cfg: WMConfig, phon_embed: nn.Embedding, premotor_dim: int):
        super().__init__()
        self.cfg = cfg
        self.phon_embed = phon_embed                 # shared, general phoneme features
        emb = phon_embed.embedding_dim
        self.encoder = nn.GRU(emb, cfg.hidden, batch_first=True)
        self.decoder = nn.GRU(emb, cfg.hidden, batch_first=True)
        self.to_premotor = nn.Linear(cfg.hidden, premotor_dim)

    def encode(self, enc_in: torch.Tensor, enc_mask: torch.Tensor,
               collect: bool = False,
               apply_noise: bool = False) -> torch.Tensor:
        """Encode phoneme sequence to WM hidden state, optionally applying noise.

        Noise semantics:
            self.training=True  → noise always active (training regime)
            apply_noise=True    → noise active in eval (explicit lesion/stress test)
            collect=True alone  → no noise; only collects intermediate state
            default (eval mode) → deterministic, no noise

        Noise is drawn ONCE per call — not once per decoder step.

        Args:
            enc_in:      (B, T) phoneme IDs
            enc_mask:    (B, T) bool mask (True = real token)
            collect:     if True, return intermediate state in downstream callers
            apply_noise: if True, apply interference noise even in eval mode

        Returns:
            h: (1, B, hidden)  — to be passed to decode_from_state
        """
        lengths = enc_mask.sum(1).clamp(min=1).cpu()
        emb = self.phon_embed(enc_in)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False)
        _, h = self.encoder(packed)                  # h: (1, B, hidden)

        if (self.training or apply_noise) and self.cfg.interference_noise > 0:
            h = h + torch.randn_like(h) * self.cfg.interference_noise
        return h

    def decode_from_state(self, h: torch.Tensor,
                          dec_in: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode from pre-computed encoder state h: (1, B, hidden).

        Args:
            h:      (1, B, hidden) — output of encode()
            dec_in: (B, S) decoder input IDs

        Returns dict with key "premotor": (B, S, premotor_dim).
        """
        dout, _ = self.decoder(self.phon_embed(dec_in), h)
        return {"premotor": self.to_premotor(dout)}

    def forward(self, enc_in: torch.Tensor, enc_mask: torch.Tensor,
                dec_in: torch.Tensor,
                collect: bool = False,
                apply_noise: bool = False) -> Dict[str, torch.Tensor]:
        h = self.encode(enc_in, enc_mask, collect=collect, apply_noise=apply_noise)
        out = self.decode_from_state(h, dec_in)
        if collect:
            out["state"] = h
        return out
