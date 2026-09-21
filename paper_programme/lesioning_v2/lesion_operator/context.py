"""Reversible composite lesion context.

Frozen (operator contract section 8). Order, per lesion cell:

    1. verify pristine parameter state (digest)
    2. apply the realization's fixed connectivity mask
    3. run upstream computation THROUGH the masked connectivity
    4. obtain the site state
    5. add the frozen item-specific activation noise
    6. run downstream computation intact
    7. exit the context
    8. restore pristine parameters
    9. verify exact parameter identity

Masks are deterministic and nested; the LIVE MODEL is never cumulatively
mutated. Every cell starts from pristine parameters.

This module performs NO training: it never calls backward(), never builds an
optimizer, and runs entirely under torch.no_grad() at the caller's discretion.
"""
from __future__ import annotations

import contextlib
import hashlib
from typing import Dict, Optional

import torch

from .sites import NEVER_LESIONED, Site


def parameter_digest(model, names: Optional[tuple] = None) -> str:
    """Deterministic digest over parameter name, dtype, shape and raw bytes."""
    sd = model.state_dict()
    keys = sorted(sd) if names is None else sorted(n for n in names if n in sd)
    h = hashlib.sha256()
    for k in keys:
        t = sd[k]
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _get_param(model, dotted: str) -> torch.nn.Parameter:
    obj = model
    parts = dotted.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return getattr(obj, parts[-1])


class PristineViolation(RuntimeError):
    """Parameters were not restored exactly, or an untouchable tensor moved."""


@contextlib.contextmanager
def connectivity_lesion(model, masks: Dict[str, torch.Tensor],
                        verify: bool = True):
    """Apply masks in place, then restore the exact original tensors.

    The original values are cloned before masking and written back on exit, so
    restoration is exact rather than reconstructed by dividing out a mask.
    """
    before = parameter_digest(model) if verify else None
    untouchable_before = parameter_digest(model, NEVER_LESIONED) if verify else None
    illegal = sorted(set(masks) & set(NEVER_LESIONED))
    if illegal:
        raise PristineViolation(
            f"REFUSED: these tensors may never be lesioned: {illegal}")
    originals: Dict[str, torch.Tensor] = {}
    try:
        with torch.no_grad():
            for name, mask in masks.items():
                p = _get_param(model, name)
                if tuple(p.shape) != tuple(mask.shape):
                    raise PristineViolation(
                        f"{name}: mask {tuple(mask.shape)} != param {tuple(p.shape)}")
                originals[name] = p.detach().clone()
                p.mul_(mask.to(device=p.device, dtype=p.dtype))
        yield model
    finally:
        with torch.no_grad():
            for name, orig in originals.items():
                _get_param(model, name).copy_(orig)
        if verify:
            after = parameter_digest(model)
            if after != before:
                raise PristineViolation(
                    "model parameters were not restored exactly after the "
                    "lesion context")
            if parameter_digest(model, NEVER_LESIONED) != untouchable_before:
                raise PristineViolation(
                    "a tensor that must never be lesioned was modified")


def assert_only_site_changed(model, site: Site, digest_before_full: str,
                             digest_before_untouchable: str) -> None:
    """Inside a lesion context: exactly the site tensor may differ."""
    if parameter_digest(model, NEVER_LESIONED) != digest_before_untouchable:
        raise PristineViolation("an untouchable tensor changed under lesion")
    if parameter_digest(model) == digest_before_full:
        raise PristineViolation("connectivity mask had no effect")
