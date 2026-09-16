"""Semantic-vector injection, forbidden-route guards and a passive token recorder.

`SemanticInjection` is ONE forward hook on `model.ltm.to_semantic`.  Its output is the
vector the ventral decoder receives (`ltm_route.py:146` -> `decode_from_s_hat`).  The
hook body is identical for every condition: it asks a *supplier* for the rows of the
bound items and returns them.  S0's supplier is the identity on the live output, so S0
is literally the native forward.  There is no condition-specific decoder branch
anywhere: the decode functions called afterwards do not know which condition is bound.

Item identity is BOUND by the caller before each batch (never inferred from tensor
content), following `gate_x_lesion/hooks.py`.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Sequence

import torch

from ventral_interface import CONDITIONS
from ventral_interface.conditions import radial


class ForbiddenRouteAccess(RuntimeError):
    """Raised when anything outside the isolated ventral path is executed."""


class SemanticInjection:
    """Replaces s_hat row-wise with the bound condition vector."""

    def __init__(self, condition: str, *, fixed: Optional[torch.Tensor] = None,
                 target_norm: Optional[torch.Tensor] = None):
        if condition not in CONDITIONS and condition != "FIXED":
            raise ValueError(f"unknown condition {condition!r}")
        if condition in ("S1", "S2", "FIXED") and fixed is None:
            raise ValueError(f"{condition} needs a fixed per-item vector table")
        if condition == "S3" and target_norm is None:
            raise ValueError("S3 needs the per-item retrieved raw GloVe norm")
        self.condition = condition
        self.fixed = fixed                  # (N, 300) indexed by bound POSITION key
        self.target_norm = target_norm      # (N,)
        self._rows: Optional[torch.Tensor] = None
        self.n_calls = 0
        self.first_live: Optional[torch.Tensor] = None   # live s_hat, first call of bind
        self.max_live_step_dev = 0.0        # encoder determinism across AR steps
        self.last_degenerate: Optional[torch.Tensor] = None

    # -- binding --------------------------------------------------------------
    def bind(self, rows: Sequence[int]) -> None:
        """`rows` index the per-item tables (fixed / target_norm) for this batch."""
        self._rows = torch.as_tensor(list(rows), dtype=torch.long)
        self.first_live = None

    def unbind(self) -> None:
        self._rows = None
        self.first_live = None

    # -- the one supplier table ----------------------------------------------
    def _supply(self, live: torch.Tensor) -> torch.Tensor:
        c = self.condition
        if c == "S0":
            return live
        if c in ("S1", "S2", "FIXED"):
            return self.fixed[self._rows].to(device=live.device, dtype=live.dtype)
        vec, deg = radial(live, self.target_norm[self._rows])
        self.last_degenerate = deg
        return vec

    def __call__(self, module, inputs, output):
        if self._rows is None:
            raise RuntimeError("HARD STOP: semantic injection fired with no bound items")
        if not torch.is_tensor(output) or output.dim() != 2:
            raise RuntimeError(f"HARD STOP: to_semantic output is not (B, D): {type(output)}")
        if output.shape[0] != len(self._rows):
            raise RuntimeError(
                f"HARD STOP: batch {output.shape[0]} != bound items {len(self._rows)}")
        live = output.detach()
        if self.first_live is None:
            self.first_live = live.clone()
        else:
            dev = float((live - self.first_live).abs().max())
            self.max_live_step_dev = max(self.max_live_step_dev, dev)
        self.n_calls += 1
        vec = self._supply(live)
        if tuple(vec.shape) != tuple(output.shape):
            raise RuntimeError("HARD STOP: supplied vector shape mismatch")
        return vec


class RouteGuard:
    """Hard-stops on any execution of FULL fusion, the gate, or the dorsal route."""

    def __init__(self):
        self.violations: List[str] = []

    def _raiser(self, name: str) -> Callable:
        def hook(module, inputs):
            self.violations.append(name)
            raise ForbiddenRouteAccess(
                f"HARD STOP: {name} executed inside the ventral interface diagnostic")
        return hook


class TokenRecorder:
    """Passive forward hook on `model.motor`: stores last-position logits per call.

    Under route "ltm" each greedy step calls `motor` exactly once, so the stored
    rows are the step logits and their argmax is the appended token.  Nothing is
    returned, so the forward is unchanged."""

    def __init__(self):
        self.steps: List[torch.Tensor] = []

    def reset(self) -> None:
        self.steps = []

    def __call__(self, module, inputs, output):
        self.steps.append(output[:, -1, :].detach().clone())
        return None

    def tokens(self) -> torch.Tensor:
        """(B, n_steps) greedy tokens, argmax exactly as in gate_probe."""
        return torch.stack([s.argmax(-1) for s in self.steps], dim=1)


@contextmanager
def ventral_only(model):
    """Guards FULL fusion (`DualRouteModel.forward`), the gate and the dorsal route
    for the duration of the block.  `route_logits(route="ltm")` never calls them."""
    guard = RouteGuard()
    handles = [
        model.register_forward_pre_hook(guard._raiser("FULL_FUSION(DualRouteModel.forward)")),
        model.gate.register_forward_pre_hook(guard._raiser("GATE")),
        model.wm.register_forward_pre_hook(guard._raiser("DORSAL(wm)")),
    ]
    try:
        yield guard
    finally:
        for h in handles:
            h.remove()


@contextmanager
def injected(model, injection: SemanticInjection, recorder: Optional[TokenRecorder] = None):
    handles = [model.ltm.to_semantic.register_forward_hook(injection)]
    if recorder is not None:
        handles.append(model.motor.register_forward_hook(recorder))
    try:
        yield injection
    finally:
        for h in handles:
            h.remove()
        injection.unbind()


def state_dict_sha256(model: torch.nn.Module) -> str:
    """Reused verbatim from the audited GXLR implementation."""
    from gate_x_lesion.hooks import state_dict_sha256 as _h
    return _h(model)
