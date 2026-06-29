"""The Motor Cortex bottleneck.

A single shared layer that both routes must speak through. It takes a pre-motor
vector (whatever mix the gate produced) and articulates a phoneme. Keeping this
layer shared and tiny is the architectural commitment that there is *one* output
channel for speech, fed by two streams.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MotorCortex(nn.Module):
    def __init__(self, premotor_dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(premotor_dim, vocab_size)

    def forward(self, premotor: torch.Tensor) -> torch.Tensor:
        """premotor (B, S, premotor_dim) -> phoneme logits (B, S, vocab)."""
        return self.proj(premotor)
