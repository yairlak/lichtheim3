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

Architecture modes (LTMConfig.ltm_encoder_mode)
-------------------------------------------------
"bigru_masked_mean" (default / historical baseline):
    bidirectional GRU -> all-position outputs -> masked mean pool -> to_semantic.
    enc_out_dim = 2 * enc_hidden.
    Backward-pass artifact: the backward GRU starts from the rightmost PADDING
    position in the batch, so pooled/s_hat shifts with the batch's maximum
    sequence length. Known; not patched in the baseline checkpoint.

"unigru_last_hidden" (Yair 2026-07, Phase 4+):
    unidirectional GRU with pack_padded_sequence -> last valid hidden state ->
    to_semantic.  enc_out_dim = enc_hidden. Symmetric with WM route. No
    padding artifact. Not weight-compatible with bigru_masked_mean checkpoints.

Ventral noise
-------------
Gaussian noise is optionally injected on the aggregated LTM encoder
representation (pooled for bigru_masked_mean; last hidden for unigru_last_hidden)
before the semantic projection.  Controlled by LTMConfig.ventral_noise.

Noise semantics (explicit):
    Active when (self.training OR apply_noise) AND cfg.ventral_noise > 0.
    collect=True alone does NOT activate noise — it means only "collect and
    return intermediate representations".  For explicit noisy evaluation
    (lesion / robustness tests), pass apply_noise=True to encode().
Default 0.0 = no ventral noise (Phase 4 setting).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LTMConfig

_VALID_MODES = {"bigru_masked_mean", "unigru_last_hidden"}


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

        mode = cfg.ltm_encoder_mode
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Unknown ltm_encoder_mode={mode!r}. "
                f"Valid values: {sorted(_VALID_MODES)}. "
                f"Old checkpoints without this field use 'bigru_masked_mean'."
            )

        # --- encoder: form -> meaning ---
        if mode == "bigru_masked_mean":
            self.encoder = nn.GRU(
                emb_dim, cfg.enc_hidden, num_layers=cfg.enc_layers,
                batch_first=True, bidirectional=True,
            )
            enc_out_dim = cfg.enc_hidden * 2
        else:  # unigru_last_hidden
            self.encoder = nn.GRU(
                emb_dim, cfg.enc_hidden, num_layers=cfg.enc_layers,
                batch_first=True, bidirectional=False,
            )
            enc_out_dim = cfg.enc_hidden

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
    def encode(self, enc_in: torch.Tensor, enc_mask: torch.Tensor,
               collect: bool = False,
               apply_noise: bool = False) -> torch.Tensor:
        """Encode a phoneme sequence to a semantic vector.

        Noise semantics:
            self.training=True  → noise always active (training regime)
            apply_noise=True    → noise active in eval (explicit lesion/stress test)
            collect=True alone  → no noise; only signals upstream to collect diag
            default (eval mode) → deterministic, no noise

        Args:
            enc_in:      (B, T) phoneme IDs
            enc_mask:    (B, T) bool mask (True = real token)
            collect:     if True, pass collect flag to upstream callers
            apply_noise: if True, apply ventral noise even in eval mode

        Returns:
            s_hat: (B, semantic_dim)
        """
        emb = self.phon_embed(enc_in)   # (B, T, E)
        mode = self.cfg.ltm_encoder_mode

        if mode == "bigru_masked_mean":
            out, _ = self.encoder(emb)                              # (B, T, 2*H)
            m = enc_mask.unsqueeze(-1).float()
            pooled = (out * m).sum(1) / m.sum(1).clamp(min=1.0)    # (B, 2*H)

        else:  # unigru_last_hidden
            lengths = enc_mask.sum(1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, lengths, batch_first=True, enforce_sorted=False)
            _, h = self.encoder(packed)   # h: (num_layers, B, H)
            pooled = h[-1]               # (B, H)  — last layer, last valid position

        # ventral noise: injected on the global encoder representation
        if (self.training or apply_noise) and self.cfg.ventral_noise > 0:
            pooled = pooled + torch.randn_like(pooled) * self.cfg.ventral_noise

        s_hat = self.to_semantic(pooled)   # (B, semantic_dim)
        return s_hat

    # ------------------------------------------------------------------ decode
    def decode_from_s_hat(self, s_hat: torch.Tensor,
                          dec_in: torch.Tensor) -> torch.Tensor:
        """Form regeneration from pre-computed s_hat -> premotor (B, S, premotor)."""
        h0 = torch.tanh(self.sem_to_h0(s_hat)).unsqueeze(0)   # (1, B, dec_hidden)
        emb = self.phon_embed(dec_in)                           # (B, S, E)
        out, _ = self.decoder(emb, h0)                          # (B, S, dec_hidden)
        return self.dec_to_premotor(out)                        # (B, S, premotor)

    def decode(self, s_hat: torch.Tensor, dec_in: torch.Tensor) -> torch.Tensor:
        """Alias for decode_from_s_hat (kept for internal use)."""
        return self.decode_from_s_hat(s_hat, dec_in)

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

        Note: L2-normalising s_hat before the matrix product is mathematically
        redundant with cosine similarity (which normalises internally), but is
        kept for numerical stability and because it creates a separate tensor q
        that does NOT modify s_hat — s_hat is used downstream as-is in
        decode_from_s_hat and alignment_loss.
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
    def forward(self, enc_in, enc_mask, dec_in, want_field: bool = False,
                collect: bool = False, apply_noise: bool = False):
        s_hat = self.encode(enc_in, enc_mask, collect=collect, apply_noise=apply_noise)
        premotor = self.decode_from_s_hat(s_hat, dec_in)
        out = {"premotor": premotor, "s_hat": s_hat}
        if want_field and self.semantic_bank.shape[0] > 1:
            out.update(self.lexical_field(s_hat))
        return out
