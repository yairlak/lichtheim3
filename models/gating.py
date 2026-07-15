"""The gate: error-suppression (lexicality routing).

A confident lexical match exerts top-down suppression on the WM buffer. The gate
returns a value `g in [0, 1]` per decode step and combines the two routes as:

    premotor = g * ltm_premotor + (1 - g) * wm_premotor

so `g -> 1` means "trust the lexicon", `g -> 0` means "trust the buffer".

    g = sigmoid(alpha * (lexical_confidence - gate_threshold))

where `lexical_confidence` is the LTM route's max cosine similarity to a known
lexeme (see `ltm_route.LTMLexicon.lexical_field`). Real words land close to a
known lexeme -> high confidence -> the ventral route wins; non-words land far
from every lexeme -> low confidence -> the dorsal buffer has to carry the trial.

`gate_threshold` (default 0.5) is the crossover point: at c_LTM == gate_threshold
the gate equals exactly 0.5 (equal blend of both routes) for any finite alpha.
Previously hard-coded to 0.5; now a configurable field in GatingConfig for the
Phase 7 alpha×threshold grid.

The gate is WORD-LEVEL: one scalar g per item, constant across all decoder
timesteps (broadcast to shape (B, S, 1) via expand).  It does not vary with
phoneme position or WM noise level.  No learnable parameters.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from config import GatingConfig


class Gate(nn.Module):
    needs_field = True

    def __init__(self, cfg: GatingConfig, premotor_dim: int):
        super().__init__()
        self.cfg = cfg
        self.premotor_dim = premotor_dim

    def gate_value(self, wm: torch.Tensor, ltm: torch.Tensor,
                   field: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        B, S, _ = wm.shape
        if field is None or "confidence" not in field:
            return torch.full((B, S, 1), 0.5, device=wm.device)
        conf = field["confidence"].view(B, 1, 1)            # (B,1,1)
        g = torch.sigmoid(self.cfg.alpha * (conf - self.cfg.gate_threshold))
        return g.expand(B, S, 1)

    def forward(self, wm: torch.Tensor, ltm: torch.Tensor,
                field: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        g = self.gate_value(wm, ltm, field)                 # (B, S, 1)
        premotor = g * ltm + (1.0 - g) * wm
        return {"premotor": premotor, "gate": g}


def build_gate(cfg: GatingConfig, premotor_dim: int) -> Gate:
    return Gate(cfg, premotor_dim)
