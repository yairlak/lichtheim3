"""Intact activation SD, measured by the exact audited historical recipe.

Every choice here is inherited, not invented.  The authority is
`HISTORICAL_LESION_OPERATOR_AUDIT.md` §2, which recovers these from executable code at
`3af0ef9`:

    SD_DEFINITION   float(a.std())  -- the `std` column the atlas consumes
                    (`run_premeeting_atlas.py:49`), NOT the `rms` the calibrator prints
    SD_AXES         all axes pooled; `.flatten()` per call then `torch.cat`
    SD_DDOF         1 (PyTorch default, unbiased)
    SD_POPULATION   deterministic_sample(range(N), 2048, seed=7), returned SORTED;
                    batch_size 256; ONE teacher-forced model(...) forward per batch
    SD_TENSOR_SHAPE scalar per (state, route)
    ZERO_SD         no special-casing
    device/dtype    cpu / float32

The historical scales themselves are NOT inherited: they were measured on a different
checkpoint (`chigh_15e5_h512_s22_u3000`).  SD is re-measured per GXLR state, on that
state's own INTACT model, before any lesion, and frozen (contract §5).
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence

import torch

from gate_x_lesion.targets import FROZEN_ROUTES, assert_site_compatible, get_module

#: Frozen historical calibration recipe.  Changing any of these changes the operator.
SD_N_ITEMS = 2048
SD_SAMPLE_SEED = 7
SD_BATCH_SIZE = 256
SD_DDOF = 1
SD_DEVICE = "cpu"
SD_DTYPE = torch.float32


def deterministic_sample(population: Sequence[int], n: int, seed: int) -> List[int]:
    """The lineage's own sampling rule, imported rather than restated.

    `scripts/naming_comprehension/train_joint_scratch.py:234-244`: first `n` of a seeded
    permutation, RETURNED SORTED, a pure function of (population, n, seed) that never
    touches the global RNG.
    """
    from scripts.naming_comprehension.train_joint_scratch import (
        deterministic_sample as _ds)
    return _ds(population, n, seed)


def measure_intact_sd(model, vocab, entries, *, routes: Sequence[str] = FROZEN_ROUTES,
                      n_items: int = SD_N_ITEMS, sample_seed: int = SD_SAMPLE_SEED,
                      batch_size: int = SD_BATCH_SIZE,
                      device: str = SD_DEVICE) -> Dict[str, dict]:
    """Measure each route's intact activation SD.  Returns `{route: {...}}`.

    The model must be INTACT: no lesion context may be open.  The collection hook and
    the lesion hook read the same GRU slot, so the tensor whose SD is measured and the
    tensor that is perturbed are the same tensor by construction.
    """
    from evaluate.hooks import make_batch

    targets = {r: assert_site_compatible(model, r) for r in routes}

    idx = deterministic_sample(range(len(entries)), n_items, sample_seed)
    forms = [entries[i].phonemes for i in idx]
    words = [entries[i].word for i in idx]

    caught: Dict[str, List[torch.Tensor]] = {}

    def make_hook(route: str, slot: int):
        def hook(_m, _i, out):
            t = out[slot] if isinstance(out, tuple) else out
            if torch.is_tensor(t):
                # `.float()` unconditionally, exactly as the historical collector did.
                caught.setdefault(route, []).append(t.detach().float().flatten())
        return hook

    handles = []
    for route, t in targets.items():
        module = get_module(model, t.module_path)
        handles.append(module.register_forward_hook(make_hook(route, t.gru_slot)))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for lo in range(0, len(forms), batch_size):
                b = make_batch([list(f) for f in forms[lo:lo + batch_size]],
                               vocab, device)
                model(b["enc_in"], b["enc_mask"], b["dec_in"])
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    pop_sha = hashlib.sha256(",".join(words).encode("utf-8")).hexdigest()

    out: Dict[str, dict] = {}
    for route in routes:
        chunks = caught.get(route)
        if not chunks:
            raise RuntimeError(
                f"HARD STOP: no activation collected for {route}; the site never fired")
        a = torch.cat(chunks)
        sd = float(a.std())                       # ddof=1, PyTorch default
        out[route] = {
            "route": route,
            "sd": sd,
            "sd_definition": "torch.Tensor.std() of flattened pooled intact activations",
            "sd_ddof": SD_DDOF,
            "sd_axes": "all pooled (batch with hidden)",
            "sd_tensor_shape": "scalar",
            "n_values": int(a.numel()),
            "n_items": len(forms),
            "sample_seed": sample_seed,
            "batch_size": batch_size,
            "population_sha256": pop_sha,
            "device": device,
            "dtype": "float32",
            "gru_slot": targets[route].gru_slot,
            "hidden_size": int(get_module(model, targets[route].module_path).hidden_size),
            # Descriptive only.  Recorded so the std/rms distinction that defect H-4
            # turns on stays visible in the record; NEVER consumed by the operator.
            "rms_DESCRIPTIVE_NOT_USED": float(a.pow(2).mean().sqrt()),
            "mean_DESCRIPTIVE_NOT_USED": float(a.mean()),
            "max_abs_DESCRIPTIVE_NOT_USED": float(a.abs().max()),
        }
    return out
