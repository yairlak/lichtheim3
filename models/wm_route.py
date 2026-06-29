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

    def forward(self, enc_in: torch.Tensor, enc_mask: torch.Tensor,
                dec_in: torch.Tensor, collect: bool = False) -> Dict[str, torch.Tensor]:
        # encode the heard sequence into a single bounded state
        lengths = enc_mask.sum(1).clamp(min=1).cpu()
        emb = self.phon_embed(enc_in)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False)
        _, h = self.encoder(packed)                  # h: (1, B, hidden)

        # interference noise on the recalled state -> capacity / length limits
        if (self.training or collect) and self.cfg.interference_noise > 0:
            h = h + torch.randn_like(h) * self.cfg.interference_noise

        # decode (teacher forced) back into a phoneme sequence
        dout, _ = self.decoder(self.phon_embed(dec_in), h)
        premotor = self.to_premotor(dout)            # (B, S, premotor_dim)
        out = {"premotor": premotor}
        if collect:
            out["state"] = h
        return out
