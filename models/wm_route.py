"""Dorsal route: the Working-Memory buffer.

Design philosophy
-----------------
This module is built to be *bad in human ways*, not good. It is a fixed-capacity,
activity-based buffer with three hard-wired frailties:

  * **Primacy gradient.** Each incoming phoneme is written with a gain that
    decays across input position (`primacy_gain * exp(-primacy_decay * t)`), so
    early items are encoded most strongly. (Page & Norris-style primacy.)
  * **Recency leak.** The buffer leaks every step (`leak < 1`), so the *last*
    items are the least decayed at read-out time.
    Primacy (write) x recency (leak) gives a U-shaped serial-position curve.
  * **Interference.** Stored traces get Gaussian noise whose scale grows with
    list length relative to capacity (`sqrt(T / n_slots)`). Long lists overload
    the buffer -> transposition/omission errors. This is the *length* sensitivity.

Why it cannot memorize a lexicon
--------------------------------
The only content the buffer ever holds are **phoneme embeddings of the items
actually presented on this trial** (`phon_embed(enc_in)`), which are general
phonetic features, not word-specific codes. Read-out is driven entirely by
*position* queries (learnable position embeddings) — there is no
content-addressable, per-word weight anywhere. Consequently the route is
structurally incapable of producing a form it was not just given, and it is
completely **invariant to training frequency**. These two properties are exactly
what we want for the dorsal half of the double dissociation.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import WMConfig


class WMBuffer(nn.Module):
    def __init__(self, cfg: WMConfig, phon_embed: nn.Embedding, premotor_dim: int):
        super().__init__()
        self.cfg = cfg
        self.phon_embed = phon_embed                 # SHARED, general phoneme features
        self.content_dim = phon_embed.embedding_dim
        self.premotor_dim = premotor_dim

        # Learnable *position* codes (general, content-free). One set for writing
        # (input positions) and one for reading (output/decode positions).
        self.write_pos = nn.Embedding(256, cfg.pos_embed_dim)
        self.read_pos = nn.Embedding(256, cfg.pos_embed_dim)

        # Read head: maps (retrieved content) -> premotor space. Applied
        # identically at every step; it transforms phoneme-level information only.
        self.read_head = nn.Sequential(
            nn.Linear(self.content_dim, cfg.read_hidden),
            nn.GELU(),
            nn.Linear(cfg.read_hidden, premotor_dim),
        )

    def forward(self, enc_in: torch.Tensor, enc_mask: torch.Tensor,
                n_steps: int, collect: bool = False) -> Dict[str, torch.Tensor]:
        """
        enc_in : (B, T) phoneme ids (one-hot equivalent via embedding lookup)
        enc_mask : (B, T) bool, True where real phoneme
        n_steps : number of decode/read steps S to produce
        returns dict with `premotor` (B, S, premotor_dim) and diagnostics.
        """
        cfg = self.cfg
        B, T = enc_in.shape
        device = enc_in.device
        lengths = enc_mask.sum(dim=1).clamp(min=1)               # (B,)

        # --- content traces: phoneme embeddings of the presented items ---------
        content = self.phon_embed(enc_in)                         # (B, T, C)

        # --- trace strength a_t = primacy(write gain) x recency(leak) ----------
        pos = torch.arange(T, device=device).float()             # (T,)
        write_gain = cfg.primacy_gain * torch.exp(-cfg.primacy_decay * pos)  # (T,)
        # steps-from-end depends on each item's own length (padding aware)
        steps_from_end = (lengths.unsqueeze(1) - 1 - pos.unsqueeze(0)).clamp(min=0)  # (B,T)
        recency = cfg.leak ** steps_from_end                      # (B, T)
        strength = write_gain.unsqueeze(0) * recency              # (B, T)

        # --- interference: noise scaled by overload sqrt(T/capacity) -----------
        overload = torch.sqrt(lengths.float() / cfg.n_slots).clamp(min=1.0)  # (B,)
        if self.training or collect:
            noise = torch.randn_like(content) * cfg.interference_std * overload.view(B, 1, 1)
            content = content + noise
            # strength noise too (jitters retrievability -> transpositions)
            strength = strength + torch.randn_like(strength) * cfg.interference_std

        # Padding positions must never be written/retrieved.
        strength = strength.masked_fill(~enc_mask, -1e9)          # (B, T)

        # --- hard capacity bottleneck: keep only the n_slots strongest traces ---
        keep = min(cfg.n_slots, T)
        topv, topi = torch.topk(strength, k=keep, dim=1)          # (B, keep)
        kept_content = torch.gather(
            content, 1, topi.unsqueeze(-1).expand(-1, -1, self.content_dim))  # (B,keep,C)
        kept_strength = topv                                      # (B, keep)
        kept_writepos = self.write_pos(topi.clamp(max=255))       # (B, keep, Pe)

        # --- position-based read-out attention ---------------------------------
        read_idx = torch.arange(n_steps, device=device).clamp(max=255)
        read_q = self.read_pos(read_idx)                          # (S, Pe)
        # scores: query.key + log(strength) so weak traces are hard to retrieve
        scores = torch.einsum("se,bke->bsk", read_q, kept_writepos)  # (B, S, keep)
        scores = scores / (kept_writepos.shape[-1] ** 0.5)
        scores = scores + torch.log(kept_strength.clamp(min=1e-6)).unsqueeze(1)
        attn = F.softmax(scores, dim=-1)                          # (B, S, keep)
        retrieved = torch.einsum("bsk,bkc->bsc", attn, kept_content)  # (B, S, C)

        premotor = self.read_head(retrieved)                     # (B, S, premotor)

        out = {"premotor": premotor}
        if collect:
            out["trace_strength"] = strength                     # (B, T)
            out["kept_index"] = topi                             # (B, keep)
            out["read_attn"] = attn                              # (B, S, keep)
        return out
