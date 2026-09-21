"""Connectivity masks: logical source->target edges, exact count, nested prefix.

Frozen (operator contract sections 2-4).

A GRU `weight_ih_l0` has shape (3H, D). Its three contiguous H-sized row blocks
are the three gate/candidate coefficient sets of the SAME anatomical
connection. The implementation depends ONLY on that shape invariant, never on
gate names or their order: one logical edge (h, d) is removed from rows
h, H+h and 2H+h alike, so any permutation of the three blocks would leave the
result unchanged.
"""
from __future__ import annotations

from typing import Dict

import torch

from . import seeds
from .sites import Site

N_SEVERITY_LEVELS = 15          # frozen: 15 NONZERO severities
CONNECTIVITY_P_MAX = 0.30       # frozen


def severity_fraction(k: int) -> float:
    """s_k = k / 15 for k in 1..15. k=0 is the intact control, not a severity."""
    if not (0 <= k <= N_SEVERITY_LEVELS):
        raise ValueError(f"severity k={k} outside 0..{N_SEVERITY_LEVELS}")
    return k / N_SEVERITY_LEVELS


def connectivity_fraction(k: int) -> float:
    return CONNECTIVITY_P_MAX * severity_fraction(k)


def n_removed(site: Site, k: int) -> int:
    return int(round(connectivity_fraction(k) * site.n_logical_edges))


def edge_permutation(state_sha256: str, site: Site,
                     realization_index: int) -> torch.Tensor:
    """The ONE permutation of all logical edges for this (state, site, realization).

    Severity is absent from the identity, which is what makes severities nested.
    """
    g = seeds.generator(seeds.mask_payload(state_sha256, site.name,
                                           realization_index))
    return torch.randperm(site.n_logical_edges, generator=g)


def logical_mask(site: Site, perm: torch.Tensor, k: int) -> torch.Tensor:
    """(hidden, input_dim) mask; 0 = edge removed. Prefix of `perm` of length n_k."""
    flat = torch.ones(site.n_logical_edges, dtype=torch.float32)
    n_k = n_removed(site, k)
    if n_k:
        flat[perm[:n_k]] = 0.0
    return flat.view(site.hidden, site.input_dim)


def physical_mask(site: Site, logical: torch.Tensor) -> torch.Tensor:
    """Expand a logical edge mask to the parameter's own shape.

    gru_ih : concat(M, M, M, dim=0) -> (3H, D)
    linear : M unchanged            -> (H, D)
    """
    if logical.shape != (site.hidden, site.input_dim):
        raise ValueError(f"logical mask {tuple(logical.shape)} != "
                         f"{(site.hidden, site.input_dim)}")
    if site.kind == "gru_ih":
        out = torch.cat([logical, logical, logical], dim=0)
    elif site.kind == "linear":
        out = logical
    else:                                              # pragma: no cover
        raise ValueError(f"unknown site kind {site.kind!r}")
    if tuple(out.shape) != site.expected_shape:
        raise ValueError(f"physical mask {tuple(out.shape)} != "
                         f"{site.expected_shape}")
    return out


def build_mask(state_sha256: str, site: Site, realization_index: int,
               k: int) -> Dict[str, torch.Tensor]:
    """The complete frozen mask for one lesion cell, keyed by parameter name."""
    perm = edge_permutation(state_sha256, site, realization_index)
    logical = logical_mask(site, perm, k)
    return {site.connectivity_param: physical_mask(site, logical)}
