"""The gate: error-suppression (lexicality routing).

A confident lexical match exerts top-down suppression on the WM buffer. The gate
returns a value `g in [0, 1]` per decode step and combines the two routes as:

    premotor = g * ltm_premotor + (1 - g) * wm_premotor

so `g -> 1` means "trust the lexicon", `g -> 0` means "trust the buffer".

    g = sigmoid(alpha * (lexical_confidence - 0.5))

where `lexical_confidence` is the LTM route's max cosine similarity to a known
lexeme (see `ltm_route.LTMLexicon.lexical_field`). Real words land close to a
known lexeme -> high confidence -> the ventral route wins; non-words land far
from every lexeme -> low confidence -> the dorsal buffer has to carry the trial.

This is the gate used throughout the evaluations and the lesion studies. (The
project deliberately ships a single gating hypothesis; the git history holds
earlier density-competition and learned-routing variants if you want to compare.)
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
        g = torch.sigmoid(self.cfg.alpha * (conf - 0.5))
        return g.expand(B, S, 1)

    def forward(self, wm: torch.Tensor, ltm: torch.Tensor,
                field: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        g = self.gate_value(wm, ltm, field)                 # (B, S, 1)
        premotor = g * ltm + (1.0 - g) * wm
        return {"premotor": premotor, "gate": g}


def build_gate(cfg: GatingConfig, premotor_dim: int) -> Gate:
    return Gate(cfg, premotor_dim)
