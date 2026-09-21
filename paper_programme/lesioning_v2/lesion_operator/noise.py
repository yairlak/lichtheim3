"""Activation noise: one symmetric-uniform base draw per item, frozen.

Frozen (operator contract section 6).

    u       ~ Uniform(-1, +1), drawn ONCE per (state, site, realization, item)
    eta_k   = s_k * SD_site * u

so the support at severity k is [-s_k*SD_site, +s_k*SD_site] and the maximum
HALF-WIDTH at k=15 is exactly one intact site SD.

NOTE on terminology, which the contract is strict about: the maximum is a
half-width, not a standard deviation. The distribution SD at k=15 is
SD_site / sqrt(3) ~= 0.577 * SD_site.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch

from . import seeds
from .masks import severity_fraction

#: SD of Uniform(-a, +a) is a / sqrt(3).
UNIFORM_SD_FACTOR = 1.0 / math.sqrt(3.0)


def base_draw(state_sha256: str, site_name: str, realization_index: int,
              item_id: str, shape: Sequence[int]) -> torch.Tensor:
    """u ~ Uniform(-1, +1) with the target-state shape.

    Severity, task and decoding convention are absent from the identity, so the
    SAME u is reproduced for every severity and every compared condition.
    """
    g = seeds.generator(seeds.noise_payload(state_sha256, site_name,
                                            realization_index, item_id))
    return torch.rand(tuple(shape), generator=g, dtype=torch.float32) * 2.0 - 1.0


def eta(u: torch.Tensor, k: int, sd_site: float) -> torch.Tensor:
    """eta_k = s_k * SD_site * u."""
    return severity_fraction(k) * float(sd_site) * u


def max_half_width(sd_site: float, k: int = 15) -> float:
    return severity_fraction(k) * float(sd_site)


def distribution_sd(sd_site: float, k: int = 15) -> float:
    """The actual SD of the perturbation -- NOT the half-width."""
    return max_half_width(sd_site, k) * UNIFORM_SD_FACTOR
