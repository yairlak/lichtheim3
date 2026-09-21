"""Activation-noise injection at the three frozen sites.

Execution-only wiring. The amplitude, distribution, identity and timing all come
from the frozen `lesion_operator.noise`; this module only decides WHERE the
already-frozen perturbation is added, using the activation targets the operator
contract fixed:

    L1  final wm.encoder h_n
    L2  final ltm.encoder h_n, BEFORE to_semantic
    L3  h0 = tanh(sem_to_h0(s_hat)), AFTER tanh, BEFORE the decoder GRU

Injection uses temporary forward hooks and removes them on exit, so no module is
modified and no parameter is touched. The per-item noise base is drawn once and
held for the entire autoregressive trajectory.
"""
from __future__ import annotations

import contextlib
from typing import Callable, Dict, Optional

import torch


class InjectionError(RuntimeError):
    pass


def _module_for(model, site: str):
    if site == "L1":
        return model.wm.encoder
    if site == "L2":
        return model.ltm.encoder
    if site == "L3":
        return model.ltm.decoder
    raise InjectionError(f"unknown site {site!r}")


@contextlib.contextmanager
def activation_injection(model, site: str,
                         eta_for_batch: Optional[Callable] = None):
    """Add the frozen perturbation at `site` for the duration of the context.

    `eta_for_batch(state_tensor) -> tensor` returns the perturbation already
    scaled by s_k * SD_site, with the item ordering of the current batch. It is
    called once per forward at the site, never per AR step: for L3 the hook sits
    on the decoder's initial hidden state, which is supplied once per item.

    Passing `eta_for_batch=None` is the k=0 intact control: hooks are installed
    and removed, but nothing is added, so the evaluator path is byte-identical
    to a lesioned cell's path.
    """
    handles = []
    try:
        if site in ("L1", "L2"):
            mod = _module_for(model, site)

            def _hook(_m, _inp, output):
                if eta_for_batch is None:
                    return output
                out, h = output
                if not isinstance(h, torch.Tensor):
                    raise InjectionError(f"{site}: unexpected hidden type")
                return out, h + eta_for_batch(h)

            handles.append(mod.register_forward_hook(_hook))

        elif site == "L3":
            mod = _module_for(model, site)

            def _pre(_m, args):
                if eta_for_batch is None:
                    return args
                if len(args) < 2 or not isinstance(args[1], torch.Tensor):
                    raise InjectionError(
                        "L3: decoder was called without an explicit h0; the "
                        "frozen production path supplies tanh(sem_to_h0(s_hat))")
                emb, h0 = args[0], args[1]
                return (emb, h0 + eta_for_batch(h0)) + tuple(args[2:])

            handles.append(mod.register_forward_pre_hook(_pre))
        else:
            raise InjectionError(f"unknown site {site!r}")
        yield model
    finally:
        for h in handles:
            h.remove()


def batch_eta_fn(state_sha256: str, site: str, realization: int,
                 item_ids, severity_k: int, sd_site: float):
    """Build the per-batch perturbation callable for one cell.

    One base draw per item, reused across every severity and every compared
    task/decoder, and held constant for the whole trajectory.
    """
    from paper_programme.lesioning_v2.lesion_operator import noise

    cache: Dict[str, torch.Tensor] = {}

    def _eta(state: torch.Tensor) -> torch.Tensor:
        n = state.shape[-2] if state.dim() == 3 else state.shape[0]
        if n != len(item_ids):
            raise InjectionError(
                f"{site}: batch carries {n} items but {len(item_ids)} item ids "
                "were supplied; item order must match exactly")
        rows = []
        per_item_shape = state.shape[-1:]
        for iid in item_ids:
            if iid not in cache:
                cache[iid] = noise.base_draw(state_sha256, site, realization,
                                             iid, tuple(per_item_shape))
            rows.append(cache[iid])
        u = torch.stack(rows, dim=0).to(state.device, state.dtype)
        if state.dim() == 3:
            u = u.unsqueeze(0)
        return noise.eta(u, severity_k, sd_site).to(state.dtype)

    return _eta
