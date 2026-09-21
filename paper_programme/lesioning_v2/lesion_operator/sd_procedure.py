"""Intact site-SD measurement procedure — FROZEN BEFORE ANY MEASUREMENT.

Frozen (operator contract section 9). Every field below is inherited from the
audited historical recipe in `gate_x_lesion/sd.py`, which itself cites
`HISTORICAL_LESION_OPERATOR_AUDIT.md` section 2 recovering them from executable
code at 3af0ef9. Nothing here is inferred silently.

    SD_DEFINITION    sample std of flattened POOLED intact site activations
    SD_AXES          all sampled items x all hidden dimensions, pooled
    SD_DDOF          1 (PyTorch unbiased)
    SD_TENSOR_SHAPE  one scalar per (state, site); isotropic, no per-unit SD
    SD_POPULATION    deterministic_sample(range(N), 2048, seed=7), SORTED
    SD_BATCH_SIZE    256
    MODEL_MODE       eval
    DTYPE            CPU float32
    FORWARD          teacher-forced intact forward, one per batch

L3 IS A NEW SITE. The historical calibration covered the encoder states; the
production state h0 = tanh(sem_to_h0(s_hat)) was not part of it. The same
statistical RECIPE is reused unchanged, but the L3 constant is a new
measurement, and is labelled as such in the output.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Sequence

SD_N_ITEMS = 2048
SD_SAMPLE_SEED = 7
SD_BATCH_SIZE = 256
SD_DDOF = 1
SD_DEVICE = "cpu"
SD_DTYPE = "float32"
SD_MODEL_MODE = "eval"
SD_FORWARD = "teacher_forced_intact"
SD_AXES = "all sampled items x all hidden dims pooled; flatten then cat"
SD_DEFINITION = "torch.Tensor.std() of flattened pooled intact activations"
SD_TENSOR_SHAPE = "scalar per (state, site)"

#: Which intermediate value is measured at each site. Identical to the
#: activation targets the operator perturbs -- the scale and the perturbation
#: must refer to the same tensor.
SD_SITE_TENSOR = {
    "L1": "final wm.encoder h_n",
    "L2": "final ltm.encoder h_n, BEFORE to_semantic",
    "L3": "h0 = tanh(sem_to_h0(s_hat)), AFTER tanh",
}
SD_SITE_IS_HISTORICALLY_CALIBRATED = {"L1": True, "L2": True, "L3": False}


def deterministic_sample(population: Sequence[int], n: int, seed: int) -> List[int]:
    """The lineage's own sampling rule: a seeded permutation, returned SORTED."""
    import random
    pop = list(population)
    if n >= len(pop):
        return sorted(pop)
    rng = random.Random(seed)
    return sorted(rng.sample(pop, n))


def population_for(n_entries: int) -> List[int]:
    return deterministic_sample(range(n_entries), SD_N_ITEMS, SD_SAMPLE_SEED)


def population_hash(indices: Sequence[int]) -> str:
    return hashlib.sha256(
        json.dumps(list(map(int, indices))).encode("utf-8")).hexdigest()


def procedure_descriptor() -> Dict[str, object]:
    """The exact frozen procedure, hashed into every SD constant it produces."""
    return {
        "sd_definition": SD_DEFINITION,
        "sd_axes": SD_AXES,
        "sd_ddof": SD_DDOF,
        "sd_tensor_shape": SD_TENSOR_SHAPE,
        "sd_n_items": SD_N_ITEMS,
        "sd_sample_seed": SD_SAMPLE_SEED,
        "sd_sample_rule": "deterministic_sample(range(N), 2048, seed=7), sorted",
        "sd_batch_size": SD_BATCH_SIZE,
        "model_mode": SD_MODEL_MODE,
        "device": SD_DEVICE,
        "dtype": SD_DTYPE,
        "forward": SD_FORWARD,
        "site_tensor": dict(SD_SITE_TENSOR),
        "site_historically_calibrated": dict(SD_SITE_IS_HISTORICALLY_CALIBRATED),
        "length_handling": "pack_padded_sequence as in the intact forward; the "
                           "measured h_n is the final state at each item's own "
                           "true length",
        "batch_construction": "sorted population, contiguous slices of 256",
        "flattening": "per-batch .flatten() then torch.cat over batches",
        "no_lesion": True,
        "no_activation_noise": True,
    }


def procedure_hash() -> str:
    return hashlib.sha256(
        json.dumps(procedure_descriptor(), sort_keys=True).encode("utf-8")
    ).hexdigest()
