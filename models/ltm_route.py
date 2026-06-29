"""Ventral route: the long-term lexical-semantic memory.

This is a *proper* associative memory, the opposite of the dorsal buffer:

  encoder:  phoneme sequence  ->  300-d semantic vector  (aligned to GloVe)
  decoder:  semantic vector   ->  phoneme sequence        (form regeneration)

Because a word is a single point in semantic space regardless of how many
phonemes it has, regenerating the form from that point is **length-invariant**.
Because the parameters are trained more on frequent words (frequency-weighted
sampler), it is **frequency-sensitive**. Together: the ventral half of the
double dissociation.

It also owns a `semantic_bank` — a frozen matrix of the training lexicon's GloVe
vectors — which lets the gate read off *lexical activation* (how close the
encoded meaning is to a known word) and *neighborhood structure* (how many
known words it is close to). Non-words land far from every bank entry.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LTMConfig


class LTMLexicon(nn.Module):
    def __init__(self, cfg: LTMConfig, phon_embed: nn.Embedding,
                 semantic_dim: int, premotor_dim: int, pad_id: int):
        super().__init__()
        self.cfg = cfg
        self.phon_embed = phon_embed
        self.semantic_dim = semantic_dim
        self.premotor_dim = premotor_dim
        self.pad_id = pad_id
        emb_dim = phon_embed.embedding_dim

        # --- encoder: form -> meaning ---
        self.encoder = nn.GRU(emb_dim, cfg.enc_hidden, num_layers=cfg.enc_layers,
                              batch_first=True, bidirectional=cfg.bidirectional_encoder)
        enc_out_dim = cfg.enc_hidden * (2 if cfg.bidirectional_encoder else 1)
        self.to_semantic = nn.Sequential(
            nn.Linear(enc_out_dim, cfg.enc_hidden), nn.GELU(),
            nn.Linear(cfg.enc_hidden, semantic_dim),
        )

        # --- decoder: meaning -> form ---
        self.sem_to_h0 = nn.Linear(semantic_dim, cfg.dec_hidden)
        self.decoder = nn.GRU(emb_dim, cfg.dec_hidden, batch_first=True)
        self.dec_to_premotor = nn.Linear(cfg.dec_hidden, premotor_dim)

        # frozen lexical memory bank (set after the lexicon is built)
        self.register_buffer("semantic_bank", torch.zeros(1, semantic_dim),
                             persistent=False)

    # ------------------------------------------------------------------ encode
    def encode(self, enc_in: torch.Tensor, enc_mask: torch.Tensor) -> torch.Tensor:
        emb = self.phon_embed(enc_in)                              # (B, T, E)
        out, _ = self.encoder(emb)                                 # (B, T, H*)
        # masked mean-pool over real positions
        m = enc_mask.unsqueeze(-1).float()
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1.0)        # (B, H*)
        s_hat = self.to_semantic(pooled)                           # (B, semantic_dim)
        return s_hat

    # ------------------------------------------------------------------ decode
    def decode(self, s_hat: torch.Tensor, dec_in: torch.Tensor) -> torch.Tensor:
        """Teacher-forced form regeneration -> premotor sequence (B, S, premotor)."""
        h0 = torch.tanh(self.sem_to_h0(s_hat)).unsqueeze(0)       # (1, B, dec_hidden)
        emb = self.phon_embed(dec_in)                             # (B, S, E)
        out, _ = self.decoder(emb, h0)                            # (B, S, dec_hidden)
        return self.dec_to_premotor(out)                          # (B, S, premotor)

    # ------------------------------------------------- lexical activation field
    def set_semantic_bank(self, bank: torch.Tensor) -> None:
        bank = F.normalize(bank, dim=-1)
        self.semantic_bank = bank.to(next(self.parameters()).device)

    def lexical_field(self, s_hat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Cosine similarities of encoded meaning to every known lexeme.

        Returns:
            sims        : (B, n_words) cosine sims to the bank
            confidence  : (B,)  max similarity  -> lexicality signal
            margin      : (B,)  best - 2nd best  -> inverse competition signal
            density     : (B,)  soft count of close competitors -> neighborhood
        """
        q = F.normalize(s_hat, dim=-1)
        sims = q @ self.semantic_bank.t()                         # (B, n_words)
        top2 = torch.topk(sims, k=min(2, sims.shape[1]), dim=1).values
        confidence = top2[:, 0]
        margin = top2[:, 0] - top2[:, -1] if top2.shape[1] > 1 else top2[:, 0]
        # soft neighborhood density: mass of competitors near the top
        density = torch.sigmoid(20.0 * (sims - (confidence.unsqueeze(1) - 0.1))).sum(1)
        return {"sims": sims, "confidence": confidence,
                "margin": margin, "density": density}

    # ------------------------------------------------------------------ forward
    def forward(self, enc_in, enc_mask, dec_in, want_field: bool = False):
        s_hat = self.encode(enc_in, enc_mask)
        premotor = self.decode(s_hat, dec_in)
        out = {"premotor": premotor, "s_hat": s_hat}
        if want_field and self.semantic_bank.shape[0] > 1:
            out.update(self.lexical_field(s_hat))
        return out
