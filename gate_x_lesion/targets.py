"""The two frozen lesion sites, and the guards that keep them from being silent no-ops.

Ported from `lesion/targets.py` + `lesion/apply.py::_GRU_SLOT` @3af0ef9, restricted to
the two routes CENTRAL froze.  Connectivity damage, composite sites and mask
scope/granularity are deliberately NOT ported (contract §2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass(frozen=True)
class RouteTarget:
    target_id: str
    module_path: str
    family: str
    #: Which element of an `nn.GRU` return tuple is the lesion site.  1 == h_n.
    #: Historical `_GRU_SLOT` (`lesion/apply.py:116-128` @3af0ef9).
    gru_slot: int
    #: If set, the state's `ltm_encoder_mode` must equal this, or the site is refused.
    requires_ltm_encoder_mode: Optional[str]
    note: str


ROUTE_TARGETS: Dict[str, RouteTarget] = {
    "wm_encoder_state": RouteTarget(
        target_id="wm_encoder_state", module_path="wm.encoder", family="dorsal",
        gru_slot=1, requires_ltm_encoder_mode=None,
        note="dorsal recurrent state h_n (1, B, 128); consumed only by "
             "wm.decode_from_state as the decoder's initial hidden state, so it "
             "reaches the fusion solely through wm_premotor and can never touch "
             "s_hat, c_LTM or g (LIVE_CODE_AUDIT.md §2.2)",
    ),
    "ltm_encoder_state": RouteTarget(
        target_id="ltm_encoder_state", module_path="ltm.encoder", family="ventral",
        gru_slot=1, requires_ltm_encoder_mode="unigru_last_hidden",
        note="ventral encoder last hidden state h_n (1, B, 512), pre-to_semantic; "
             "reaches BOTH the ventral premotor and the lexical field, hence c_LTM "
             "and g.  Under ltm_encoder_mode='bigru_masked_mean' the module reads "
             "slot 0 and DISCARDS h_n, so a slot-1 lesion would be a silent no-op — "
             "which is why the mode is asserted (LIVE_CODE_AUDIT.md §2.3)",
    ),
}

FROZEN_ROUTES: Tuple[str, ...] = ("wm_encoder_state", "ltm_encoder_state")


def get_target(route: str) -> RouteTarget:
    try:
        return ROUTE_TARGETS[route]
    except KeyError:
        raise ValueError(
            f"unknown route {route!r}; the contract freezes exactly "
            f"{list(FROZEN_ROUTES)}") from None


def get_module(model: torch.nn.Module, dotted: str) -> torch.nn.Module:
    m = model
    for part in dotted.split("."):
        m = getattr(m, part)
    return m


def assert_site_compatible(model: torch.nn.Module, route: str) -> RouteTarget:
    """Refuse a site that would be a silent no-op or the wrong tensor.

    A lesion that perturbs a discarded tensor produces a null result that is
    indistinguishable, after the fact, from a scientific null.  This is the one failure
    mode that no downstream analysis can detect, so it is checked up front.
    """
    t = get_target(route)
    module = get_module(model, t.module_path)
    if not isinstance(module, torch.nn.GRU):
        raise RuntimeError(
            f"HARD STOP: {route} resolves to {type(module).__name__}, not nn.GRU; "
            f"the frozen slot semantics (slot {t.gru_slot} == h_n) do not apply")

    if t.requires_ltm_encoder_mode is not None:
        mode = getattr(model.ltm.cfg, "ltm_encoder_mode", None)
        if mode != t.requires_ltm_encoder_mode:
            raise RuntimeError(
                f"HARD STOP: {route} requires ltm_encoder_mode="
                f"{t.requires_ltm_encoder_mode!r}, got {mode!r}. Under that mode the "
                f"ventral encoder reads GRU slot 0 and discards h_n, so lesioning "
                f"slot {t.gru_slot} would be a SILENT NO-OP and would report a null "
                f"that is an artefact of wiring, not a result "
                f"(LIVE_CODE_AUDIT.md §2.3).")

    if module.num_layers != 1 or module.bidirectional:
        raise RuntimeError(
            f"HARD STOP: {route} GRU has num_layers={module.num_layers}, "
            f"bidirectional={module.bidirectional}; the frozen site shape (1, B, H) "
            f"and the per-item epsilon assembly assume a single unidirectional layer")
    return t
