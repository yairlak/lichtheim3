"""Deterministic base noise and nested severity scaling for GATE x LESION / RECOVERY.

Frozen by `paper_programme/gate_x_lesion_recovery/GATE_X_LESION_EXPERIMENT_CONTRACT.md`
§6 (CENTRAL micro-amendment 2) and audited in `LIVE_CODE_AUDIT.md` §7.1-§7.2.

One base realisation per item:

    epsilon ~ Uniform(-1, +1)

with RNG identity EXACTLY

    (state_sha256, route, lesion_seed, item_id)

`lambda` and the fusion condition are excluded *structurally*, not by convention:
`_identity_payload` builds a fixed four-key dict and takes no other arguments, so there
is no expression in this module by which a severity or a fusion condition could reach the
digest.  Severities are then obtained prospectively by scaling:

    eta(lambda) = lambda * SD * epsilon

Why not a torch.Generator (the historical mechanism, `lesion/masks.py:26-40` @3af0ef9):
a single generator consumed sequentially makes the realisation a function of batch
composition, batch order and call count, and makes it impossible for two severities to
share a draw.  See HISTORICAL_LESION_OPERATOR_AUDIT.md defects H-1..H-3.

Why not Python's `hash()`: `str.__hash__` is salted per process (PYTHONHASHSEED), so an
identity built on it would not survive a restart, let alone another machine.  A canonical
JSON serialisation plus SHA-256 is stable across process, platform and torch version.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

#: Domain separation tag.  Bump only with a contract amendment: changing it changes
#: every epsilon in the experiment.
EPS_DOMAIN = b"GXLR-eps-v1|"

#: The RNG identity schema.  Exactly these four fields, in this order.
IDENTITY_FIELDS = ("state_sha256", "route", "lesion_seed", "item_id")

#: Frozen: excluded from the RNG identity by CENTRAL micro-amendment 2.
IDENTITY_EXCLUDED = ("lambda", "lam", "fusion", "fusion_condition", "convention",
                     "batch_size", "batch_index")

#: The frozen severity grid.  All three are exact binary powers of two, which is what
#: makes the nesting bitwise exact (see `eta`).
FROZEN_LAMBDAS = (0.25, 0.50, 1.00)

_U32 = np.dtype(">u4")


def _identity_payload(state_sha256: str, route: str, lesion_seed: int,
                      item_id: str) -> Dict[str, object]:
    """The RNG identity, as a fixed four-key dict.

    This function is the ONLY place an identity is constructed.  It accepts no severity
    and no fusion condition, so neither can enter the digest.
    """
    return {
        "state_sha256": str(state_sha256),
        "route": str(route),
        "lesion_seed": int(lesion_seed),
        "item_id": str(item_id),
    }


def identity_json(state_sha256: str, route: str, lesion_seed: int, item_id: str) -> str:
    """Canonical serialisation: sorted keys, no whitespace, ASCII-escaped."""
    return json.dumps(_identity_payload(state_sha256, route, lesion_seed, item_id),
                      sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def identity_digest(state_sha256: str, route: str, lesion_seed: int,
                    item_id: str) -> bytes:
    """32-byte cryptographic identity for one item's base realisation."""
    return hashlib.sha256(
        EPS_DOMAIN + identity_json(state_sha256, route, lesion_seed, item_id)
        .encode("utf-8")).digest()


def _uniform_stream(key: bytes, n: int) -> np.ndarray:
    """`n` float64 values uniform on [0, 1), from a SHA-256 counter-mode byte stream.

    Deterministic across process, platform, numpy and torch version: the only
    primitives are SHA-256, big-endian integer decoding and exact float division.
    """
    need = 4 * int(n)
    chunks: List[bytes] = []
    got = 0
    ctr = 0
    while got < need:
        chunks.append(hashlib.sha256(key + ctr.to_bytes(8, "big")).digest())
        got += 32
        ctr += 1
    raw = b"".join(chunks)[:need]
    u = np.frombuffer(raw, dtype=_U32).astype(np.float64)
    return u / 4294967296.0                      # 2**32 -> [0, 1)


def base_epsilon(state_sha256: str, route: str, lesion_seed: int, item_id: str,
                 n_units: int) -> torch.Tensor:
    """One item's base realisation: `epsilon ~ U(-1, +1)`, shape `(n_units,)`, float32.

    Support is the half-open [-1, +1), matching `torch.rand`'s [0, 1) and therefore the
    historical `(rand * 2 - 1)` exactly in distribution (`lesion/masks.py:185` @3af0ef9).

    Pure function of the frozen identity.  Independent of lambda, of the fusion
    condition, of batch size, of batch order and of how many times it is called.
    """
    x = _uniform_stream(identity_digest(state_sha256, route, lesion_seed, item_id),
                        int(n_units))
    return torch.from_numpy((2.0 * x - 1.0).astype(np.float32))


def amplitude(lam: float, sd: float) -> float:
    """`lambda * SD`, the historical `draw_noise` amplitude.

    Computed in float64 and rounded once to float32.  For the frozen dyadic severities
    this makes `amplitude` exactly double from one severity to the next.
    """
    return float(np.float32(float(lam) * float(sd)))


def eta(epsilon: torch.Tensor, lam: float, sd: float) -> torch.Tensor:
    """`eta(lambda) = epsilon * float32(lambda * SD)` — the scaled perturbation.

    Operand order matches the historical path (noise tensor times amplitude,
    `lesion/masks.py:185`), so the floating-point result is the historical one.

    Nesting is BITWISE exact for lambda in {0.25, 0.50, 1.00}: those are exact binary
    powers of two, multiplication by a power of two is exact in IEEE-754, and scaling
    commutes with round-to-nearest.  Hence

        eta(0.50) == 2 * eta(0.25)   and   eta(1.00) == 2 * eta(0.50)

    elementwise and bitwise.  The frozen tolerance is 0.0; the tests assert equality
    rather than closeness, so a change of severity grid fails loudly instead of drifting.
    """
    return epsilon * amplitude(lam, sd)


class EpsilonCache:
    """Memoises base realisations per `(state, route, seed, item)`.

    Purely a performance device.  Because `base_epsilon` is a pure function of the
    identity, a cache miss and a cache hit are bitwise indistinguishable — correctness
    never depends on the cache being warm, which is what makes `per_item_frozen` a
    structural property here rather than a cache heuristic (contract §6.2).
    """

    def __init__(self, state_sha256: str, route: str, lesion_seed: int, n_units: int):
        self.state_sha256 = str(state_sha256)
        self.route = str(route)
        self.lesion_seed = int(lesion_seed)
        self.n_units = int(n_units)
        self._cache: Dict[str, torch.Tensor] = {}
        self.n_computed = 0
        self.n_served = 0

    def epsilon(self, item_id: str) -> torch.Tensor:
        self.n_served += 1
        e = self._cache.get(item_id)
        if e is None:
            e = base_epsilon(self.state_sha256, self.route, self.lesion_seed,
                             item_id, self.n_units)
            self._cache[item_id] = e
            self.n_computed += 1
        return e

    def batch_epsilon(self, item_ids: Sequence[str]) -> torch.Tensor:
        """`(1, B, n_units)` — the site tensor's shape for a GRU `h_n` with one layer."""
        return torch.stack([self.epsilon(i) for i in item_ids], dim=0).unsqueeze(0)

    def batch_eta(self, item_ids: Sequence[str], lam: float, sd: float) -> torch.Tensor:
        return self.batch_epsilon(item_ids) * amplitude(lam, sd)
