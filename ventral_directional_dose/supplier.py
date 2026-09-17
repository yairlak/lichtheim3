"""The directional-dose supplier on the SAME frozen `model.ltm.to_semantic` hook.

`DoseInjection` subclasses the frozen `ventral_interface.injection.SemanticInjection`, reusing
its hook body unchanged (item binding, live ŝ capture, AR-step encoder stability, shape
checks) and overriding only the vector supplier.  It is registered through the frozen
`injected` context and decoded by the frozen `decode_condition`, so nothing downstream of the
supplied vector differs from S0.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch

from ventral_directional_dose import ALPHAS
from ventral_directional_dose.dose_math import dose_vectors
from ventral_interface.injection import SemanticInjection


class DoseInjection(SemanticInjection):
    def __init__(self, alpha: float, prototype_table: torch.Tensor):
        super().__init__("S0")
        alpha = float(alpha)
        if alpha != 0.0 and alpha not in ALPHAS:
            raise ValueError(f"alpha {alpha} is neither the DOSE-B endpoint 0 nor a frozen alpha")
        self.condition = f"DOSE_alpha_{alpha}"
        self.alpha = alpha
        self.prototype_table = prototype_table            # (n_items, D), RAW, indexed by item row
        self._cache_live: Optional[torch.Tensor] = None
        self.last_supplied: Optional[torch.Tensor] = None
        self.last_live: Optional[torch.Tensor] = None
        self.n_supply = 0

    def bind(self, rows) -> None:
        super().bind(rows)
        self._cache_live = None
        self.last_supplied = None

    def _supply(self, live: torch.Tensor) -> torch.Tensor:
        self.n_supply += 1
        self.last_live = live
        if self.alpha == 0.0:                              # exact short-circuit: s_0 := ŝ
            self.last_supplied = live
            return live
        if self._cache_live is None or not torch.equal(self._cache_live, live):
            v = self.prototype_table[self._rows]
            self.last_supplied = dose_vectors(live, v, self.alpha)
            self._cache_live = live.clone()
        return self.last_supplied


class SemToH0Probe:
    """DOSE-F: the tensor entering `ltm.sem_to_h0` must be exactly the supplied vector."""

    def __init__(self, injection: DoseInjection):
        self.injection = injection
        self.n_calls = 0
        self.n_mismatch = 0

    def __call__(self, module, inputs):
        self.n_calls += 1
        sup = self.injection.last_supplied
        if sup is None or not torch.equal(inputs[0], sup):
            self.n_mismatch += 1
        return None


@contextmanager
def sem_to_h0_probe(model, injection: DoseInjection):
    probe = SemToH0Probe(injection)
    h = model.ltm.sem_to_h0.register_forward_pre_hook(probe)
    try:
        yield probe
    finally:
        h.remove()
