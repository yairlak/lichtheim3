"""Deterministic, domain-separated seed derivation for Lesioning V2.

Frozen (operator contract section 5). Python's built-in `hash()` is never used:
it is salted per process and would destroy reproducibility.

    payload -> utf-8 bytes -> sha256 -> first 8 bytes, BIG-endian, unsigned
            -> mask to 63 bits (non-negative int64) -> torch.Generator seed
"""
from __future__ import annotations

import hashlib

NAMESPACE = "L3_LESION_V2_V1"
MASK_ROLE = "MASK"
NOISE_ROLE = "NOISE"
_DIGEST_BYTES = 8          # frozen: first 8 bytes of the digest
_BYTEORDER = "big"         # frozen
_SEED_MASK = (1 << 63) - 1  # frozen: non-negative int64


def mask_payload(state_sha256: str, site: str, realization_index: int) -> str:
    """Connectivity-mask identity. Severity, task, decoder and item are
    deliberately ABSENT so that masks pair across all of them."""
    return f"{NAMESPACE}|{MASK_ROLE}|{state_sha256}|{site}|{int(realization_index)}"


def noise_payload(state_sha256: str, site: str, realization_index: int,
                  item_id: str) -> str:
    """Activation-noise identity. Severity, task and decoder are deliberately
    ABSENT; only the item enters, so one base draw serves all 15 severities."""
    return (f"{NAMESPACE}|{NOISE_ROLE}|{state_sha256}|{site}"
            f"|{int(realization_index)}|{item_id}")


def digest_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derive_seed(payload: str) -> int:
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:_DIGEST_BYTES], byteorder=_BYTEORDER,
                          signed=False) & _SEED_MASK


def generator(payload: str):
    import torch
    g = torch.Generator()
    g.manual_seed(derive_seed(payload))
    return g
