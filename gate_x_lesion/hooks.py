"""The lesion context: additive activation noise on one route's encoder state.

Activation damage only.  No parameter is ever written, so the checkpoint cannot be
mutated; the hook is removed in `finally`, including on exception
(the discipline of `lesion/apply.py:180-185` @3af0ef9).

Two things distinguish this from the historical hook, both required by CENTRAL
micro-amendment 2:

* **Item identity is bound, not inferred.**  The historical `per_item` path guessed
  which stimulus it was looking at from a float fingerprint of the encoder input
  (`lesion/apply.py:43-65`), which can collide.  Here the caller states the batch's item
  ids before decoding, so the mapping is exact.

* **The tensor is a pure function of the item.**  `per_item_frozen` therefore holds
  structurally: every one of the O(max_steps x routes) encoder calls during an item's
  autoregressive decode reconstructs the same value, whether or not a cache is warm.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence

import torch

from gate_x_lesion.noise import EpsilonCache, amplitude
from gate_x_lesion.targets import RouteTarget, assert_site_compatible, get_module


def state_dict_sha256(model: torch.nn.Module) -> str:
    """Content hash of the model's parameters and buffers.

    Stronger than re-hashing the checkpoint file: a forward hook cannot touch the file,
    but could in principle write a parameter in place.  Pinned by T10.
    """
    h = hashlib.sha256()
    sd = model.state_dict()
    for name in sorted(sd):
        t = sd[name]
        h.update(name.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class RouteNoiseHook:
    """Adds `eta = epsilon * (lambda * SD)` to GRU slot 1 (`h_n`) of one encoder."""

    def __init__(self, target: RouteTarget, cache: EpsilonCache, lam: float, sd: float):
        self.target = target
        self.cache = cache
        self.lam = float(lam)
        self.sd = float(sd)
        self.amp = amplitude(lam, sd)
        self._item_ids: Optional[List[str]] = None
        self._eta: Optional[torch.Tensor] = None
        self.n_calls = 0
        self.n_eta_builds = 0

    # -- batch binding -------------------------------------------------------
    def bind(self, item_ids: Sequence[str]) -> None:
        """Declare which items, in order, the next forward passes describe."""
        ids = [str(i) for i in item_ids]
        if ids != self._item_ids:
            self._item_ids = ids
            self._eta = self.cache.batch_epsilon(ids) * self.amp
            self.n_eta_builds += 1

    def unbind(self) -> None:
        self._item_ids = None
        self._eta = None

    # -- torch forward hook --------------------------------------------------
    def __call__(self, module, inputs, output):
        if self._eta is None:
            raise RuntimeError(
                "HARD STOP: the lesion hook fired with no bound item ids. Every "
                "forward under a lesion context must be preceded by bind(item_ids); "
                "inferring item identity from tensor content is exactly the "
                "historical defect H-1 this implementation removes.")
        if not isinstance(output, tuple):
            raise RuntimeError(
                f"HARD STOP: expected nn.GRU to return a tuple, got "
                f"{type(output).__name__}")

        slot = self.target.gru_slot
        perturbed = list(output)
        ref = perturbed[slot]
        if isinstance(ref, torch.nn.utils.rnn.PackedSequence):
            raise RuntimeError(
                f"HARD STOP: slot {slot} of {self.target.target_id} is a "
                f"PackedSequence, not a tensor. The historical operator silently "
                f"returned the output unchanged in this case, which would make the "
                f"lesion a no-op (LIVE_CODE_AUDIT.md §1.1).")
        if not torch.is_tensor(ref):
            raise RuntimeError(
                f"HARD STOP: slot {slot} of {self.target.target_id} is "
                f"{type(ref).__name__}, not a tensor")

        eta = self._eta
        if tuple(ref.shape) != tuple(eta.shape):
            raise RuntimeError(
                f"HARD STOP: site tensor shape {tuple(ref.shape)} does not match the "
                f"bound lesion tensor {tuple(eta.shape)}. The bound item ids "
                f"({len(self._item_ids)}) must match the batch, in order.")

        self.n_calls += 1
        # Out of place: `ref + eta` allocates, so no captured tensor is mutated.
        perturbed[slot] = ref + eta.to(device=ref.device, dtype=ref.dtype)
        return tuple(perturbed)


@contextmanager
def lesioned_route(model: torch.nn.Module, *, route: str, state_sha256: str,
                   lesion_seed: int, lam: float, sd: float,
                   cache: Optional[EpsilonCache] = None):
    """Apply the frozen activation lesion to `route` for the duration of the block.

    Yields the hook, whose `bind(item_ids)` must be called before each batch.

    `lambda * SD == 0` registers NO hook at all, reproducing the historical semantics
    (`lesion/apply.py:169-176` gates on `amp > 0.0`, and `draw_noise` short-circuits at
    zero amplitude).  This is what makes the severity-0 control a genuinely untouched
    intact run rather than an addition of an exactly-zero tensor.
    """
    target = assert_site_compatible(model, route)
    module = get_module(model, target.module_path)
    n_units = int(module.hidden_size)

    if cache is None:
        cache = EpsilonCache(state_sha256, route, lesion_seed, n_units)
    elif cache.n_units != n_units:
        raise RuntimeError(
            f"HARD STOP: epsilon cache is sized for {cache.n_units} units, site has "
            f"{n_units}")

    amp = amplitude(lam, sd)
    hook = RouteNoiseHook(target, cache, lam, sd)

    if amp == 0.0:
        # No hook: an exactly-zero perturbation is the intact model.
        hook.bind = lambda item_ids: None            # type: ignore[assignment]
        hook.unbind = lambda: None                   # type: ignore[assignment]
        yield hook
        return

    handle = module.register_forward_hook(hook)
    try:
        yield hook
    finally:
        handle.remove()
        hook.unbind()
