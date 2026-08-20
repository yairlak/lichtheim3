"""Phase 3A: joint / interleaved multitask training.

Repetition + comprehension + naming trained together in ONE shared LTM
parameter scope, with a single shared AdamW, interleaved at the OPTIMIZER-STEP
level (never one-epoch-per-task).

Scientific question: can the three tasks coexist in the same LTM parameters?
Single-task Phase 2 established that each is learnable alone, that sequential
staging catastrophically forgets, and that single-task training of either side
destroys LTM repetition.

Everything reused verbatim from the validated Phase 2 machinery
--------------------------------------------------------------
* comprehension  : `train_tasks.comprehension_objective` with objective "c3"
                   (C0 = losses.alignment_loss verbatim, tau=0.10,
                   lambda_ret=0.087, full 29,571-word retrieval bank)
* naming         : `train_tasks.naming_objective` (existing sequence CE, TF=1.0)
* repetition     : composed HERE from the two already-validated task forwards
                   (see `repetition_objective`) -- no new model API, no new
                   loss function, and numerically identical to the model's own
                   `route_logits(..., route="ltm")`.
* batching       : `train_tasks.make_batches`
* evaluation     : `train_tasks.evaluate_comprehension_subset`,
                   `train_tasks.evaluate_naming`, `train_tasks.repetition_snapshot`
                   (the last is the canonical forced-length AR evaluator)
* population     : the EXACT Phase 2D3 subset3288, reconstructed by the
                   deterministic selector and checked against the stored hash.

Deliberately NOT used: `losses.total_loss` (it carries gate/WM/alignment terms
that would confound a three-task pilot), the historical `train.py`, and any
gating objective.

Nothing here modifies models/, config.py, train.py, losses.py or the gate.

Usage:
    python scripts/naming_comprehension/train_multitask.py preflight --ckpt <canonical>
    python scripts/naming_comprehension/train_multitask.py run --ckpt <canonical> \
        --schedule m1_111 --out-dir <dir>
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.lexicon import LexEntry                                       # noqa: E402
from data.phonemes import Vocab                                         # noqa: E402
from losses import _seq_ce                                              # noqa: E402
from models.dual_route import DualRouteModel                            # noqa: E402
from scripts.naming_comprehension.frozen_probe import load_frozen       # noqa: E402
from scripts.naming_comprehension.train_tasks import (                  # noqa: E402
    ALWAYS_FROZEN, TRAINABLE_PREFIXES, _write_tsv, assert_dorsal_untouched,
    population_composition,
    changed_parameters, comprehension_forward, comprehension_objective,
    dorsal_fingerprint, evaluate_comprehension_subset, evaluate_naming,
    make_batches, naming_forward, naming_objective, parameter_fingerprint,
    repetition_snapshot, require_real_glove, select_nested_subset,
    subset_definition_hash, subset_records, verify_bank_mapping)
from utils.provenance import git_state, sha256_file                     # noqa: E402

# ==========================================================  constants  ====

TASKS: Tuple[str, str, str] = ("repetition", "naming", "comprehension")

# The multitask scope is exactly the UNION of the two validated single-task
# scopes -- derived from them, never re-typed, so it cannot silently drift.
ENCODER_SIDE: Tuple[str, ...] = TRAINABLE_PREFIXES["comprehension"]
DECODER_SIDE: Tuple[str, ...] = TRAINABLE_PREFIXES["naming"]
UNION_PREFIXES: Tuple[str, ...] = tuple(sorted(set(ENCODER_SIDE + DECODER_SIDE)))

# Task-frequency schedules, expressed as counts per SIX-STEP macro-cycle in
# TASKS order (repetition, naming, comprehension).  Both schedules spend
# exactly six optimizer updates per cycle, so task ratio is never confounded
# with total optimizer-step budget.
SCHEDULES: Dict[str, Tuple[int, int, int]] = {
    "m1_111": (2, 2, 2),        # equal:            1:1:1
    "m2_123": (1, 2, 3),        # Lichtheim2-style: 1:2:3
}
MACRO_CYCLE_STEPS = 6

# Validated objective constants, inherited unchanged from Phase 2.
TAU = 0.10
LAMBDA_RET = 0.087

# Independent seed namespaces: the task-order RNG must never share state with
# the per-task data samplers or with evaluation.
SCHEDULE_SEED_BASE = 1_000_003
TASK_DATA_SEEDS: Dict[str, int] = {"repetition": 11, "naming": 22, "comprehension": 33}

# Predeclared Phase 3 coexistence criterion, fixed BEFORE any run.
COEXISTENCE = {
    "comprehension_top1_min": 0.95,
    "naming_exact_min": 0.95,
    "naming_wer_max": 0.05,
    "ltm_repetition_exact_min": 0.95,
}
CONSECUTIVE_REQUIRED = 2
DEFAULT_TOTAL_STEPS = 400_000
DEFAULT_EVAL_EVERY = 20_000

# Early evaluations are dense because Phase 2J showed catastrophic forgetting
# can complete inside <8k optimizer steps; a 20k-only grid would step straight
# over it.  After 20k the cadence is the regular one.  This changes ONLY when
# the model is measured, never the training schedule between measurements.
EVAL_STEPS_EARLY: Tuple[int, ...] = (2_000, 5_000, 10_000, 20_000)

RESUME_FORMAT_VERSION = 2       # v2 stores PER-TASK population sizes
RESUME_FILENAME = "resume_checkpoint.pt"

# Phase 3C: the repetition task may rehearse the FULL historical lexicon while
# naming and comprehension stay on subset3288.  "subset" reproduces Phase
# 3A/3B exactly and remains the default, so no earlier semantics change.
REPETITION_POPULATIONS = ("subset", "full_lexicon")

# Seed namespace for the out-of-subset repetition probe, distinct from the
# schedule and from every task sampler.
PROBE_SEED_BASE = 2_000_000
DEFAULT_PROBE_N = 3288
BATCH_SIZE = 64
SUBSET_PER_BAND = 822                     # -> the Phase 2D3 subset3288
PHASE2D3_SUBSET_SHA256 = (
    "df48250092cdd8a6d37c33bc008b915f84a1e829ddaa2bafbaa593cce446d5cf")


# ======================================================  parameter scope  ==

def set_multitask_scope(model: DualRouteModel) -> List[str]:
    """Freeze everything, then unfreeze exactly the union of the two scopes."""
    trainable: List[str] = []
    for name, p in model.named_parameters():
        on = name.startswith(UNION_PREFIXES)
        if on and name in ALWAYS_FROZEN:
            raise RuntimeError(
                f"Union scope would train always-frozen parameter {name!r}.")
        p.requires_grad_(on)
        if on:
            trainable.append(name)
    if not trainable:
        raise RuntimeError("Empty multitask trainable scope.")
    params = dict(model.named_parameters())
    for name in ALWAYS_FROZEN:
        if params[name].requires_grad:
            raise RuntimeError(f"{name} must stay frozen but is trainable.")
    for name, p in model.named_parameters():
        if name.startswith("wm.") and p.requires_grad:
            raise RuntimeError(f"WM parameter {name!r} must stay frozen.")
    return sorted(trainable)


def side_of(name: str) -> Optional[str]:
    if name.startswith(ENCODER_SIDE):
        return "encoder_side"
    if name.startswith(DECODER_SIDE):
        return "decoder_side"
    return None


# =========================================================  objectives  ====

def repetition_objective(model: DualRouteModel, batch: Dict[str, object],
                         pad_id: int) -> Dict[str, torch.Tensor]:
    """Pure-LTM repetition: phonology -> s_hat -> phonology, teacher forced.

    Composed from the two ALREADY-VALIDATED Phase 2 task forwards:

        comprehension_forward(model, enc_in, enc_mask)  == ltm.encode
        naming_forward(model, s_hat, dec_in)            == motor(ltm.decode_from_s_hat(.))

    so the single sequence CE provably backpropagates through BOTH sides of the
    union scope: encoder/to_semantic upstream of s_hat, and
    sem_to_h0/decoder/dec_to_premotor downstream of it.  motor.proj is traversed
    but frozen.  WM is never touched, and the gate is never involved.

    This composition is numerically identical to the model's own ventral route,
    `model.route_logits(enc_in, enc_mask, dec_in, route="ltm")["logits"]`
    (asserted in the tests), so no new repetition semantics are invented here.
    The loss is `losses._seq_ce`, the same sequence CE the repository already
    uses, against the repetition phoneme target.
    """
    s_hat = comprehension_forward(model, batch["enc_in"], batch["enc_mask"])
    logits = naming_forward(model, s_hat, batch["dec_in"])
    return {"total": _seq_ce(logits, batch["dec_tgt"], pad_id),
            "logits": logits, "s_hat": s_hat}


def task_objective(model: DualRouteModel, task: str, batch: Dict[str, object],
                   pad_id: int) -> Dict[str, torch.Tensor]:
    """One task batch -> one scalar loss, using the validated objectives."""
    if task == "repetition":
        return repetition_objective(model, batch, pad_id)
    if task == "naming":
        return naming_objective(model, batch, pad_id)
    if task == "comprehension":
        return comprehension_objective(model, batch, "c3", TAU, LAMBDA_RET)
    raise ValueError(f"Unknown task {task!r}; expected one of {TASKS}.")


# ==========================================================  schedule  =====

def macro_cycle(ratio: Sequence[int], schedule_seed: int,
                cycle_index: int) -> List[str]:
    """The six task labels of one macro-cycle, deterministically shuffled.

    The permutation comes from a dedicated `torch.Generator` seeded by
    (schedule_seed, cycle_index) only.  It never draws from the global RNG, so
    task ordering is independent of the data samplers, of model initialisation
    and of evaluation -- and is reproducible from the cycle index alone.
    """
    if len(ratio) != len(TASKS):
        raise ValueError(f"ratio must have {len(TASKS)} entries, got {ratio!r}")
    if any(int(k) < 0 for k in ratio):
        raise ValueError(f"ratio entries must be non-negative, got {ratio!r}")
    if sum(int(k) for k in ratio) != MACRO_CYCLE_STEPS:
        raise ValueError(
            f"ratio {tuple(ratio)} sums to {sum(ratio)}, but every schedule must "
            f"spend exactly {MACRO_CYCLE_STEPS} optimizer steps per macro-cycle "
            "so that task ratio is not confounded with total step budget.")
    multiset: List[str] = []
    for task, k in zip(TASKS, ratio):
        multiset += [task] * int(k)
    g = torch.Generator().manual_seed(schedule_seed * SCHEDULE_SEED_BASE + cycle_index)
    order = torch.randperm(len(multiset), generator=g).tolist()
    return [multiset[i] for i in order]


def task_schedule_stream(ratio: Sequence[int], total_steps: int,
                         schedule_seed: int = 0,
                         start_step: int = 0) -> Iterator[str]:
    """Task labels for optimizer steps [start_step, total_steps).

    The stream is a pure function of (ratio, schedule_seed, step index): step i
    is position i % MACRO_CYCLE_STEPS of cycle i // MACRO_CYCLE_STEPS.  It
    therefore needs no stored RNG state and can be re-entered at any step, which
    is what makes an exact resume possible.
    """
    if start_step < 0:
        raise ValueError(f"start_step must be >= 0, got {start_step}")
    cycle, pos = divmod(start_step, MACRO_CYCLE_STEPS)
    emitted = start_step
    while emitted < total_steps:
        for task in macro_cycle(ratio, schedule_seed, cycle)[pos:]:
            if emitted >= total_steps:
                return
            yield task
            emitted += 1
        cycle += 1
        pos = 0


def schedule_counts(ratio: Sequence[int], total_steps: int,
                    schedule_seed: int = 0) -> Dict[str, int]:
    """Exact per-task optimizer-step counts for a fixed total budget."""
    counts = collections.Counter(
        task_schedule_stream(ratio, total_steps, schedule_seed))
    out = {t: int(counts.get(t, 0)) for t in TASKS}
    if sum(out.values()) != total_steps:
        raise RuntimeError("schedule stream did not emit exactly total_steps.")
    return out


def population_passes(task_steps: int, population: int,
                      batch_size: int = BATCH_SIZE) -> float:
    """Equivalent epochs: completed passes over the task's population.

    One complete pass is `ceil(population / batch_size)` batches, NOT
    `population / batch_size`: with N=3288 and batch 64 a pass is 52 batches
    whose last one holds only 24 items.  Dividing by 52 is what makes this
    directly comparable to the single-task Phase 2 runs -- e.g. 200,200
    comprehension steps is exactly 3,850 passes, the Phase 2D3 figure.
    """
    return task_steps / batches_per_epoch(population, batch_size)


def item_presentations(task_steps: int, population: int,
                       batch_size: int = BATCH_SIZE) -> int:
    """Exact cumulative item presentations, honouring the short final batch.

    `make_batches` emits the short batch last, so within any partial pass the
    consumed batches are all full-size.  For q whole passes plus r extra
    batches the count is therefore q*population + r*batch_size, never
    task_steps*batch_size (which would credit the short batch with 64 items).
    """
    per_epoch = batches_per_epoch(population, batch_size)
    q, r = divmod(task_steps, per_epoch)
    return q * population + r * batch_size


def presentations_per_item(task_steps: int, population: int,
                           batch_size: int = BATCH_SIZE) -> float:
    """Mean number of times each item was presented (flat sampling)."""
    return item_presentations(task_steps, population, batch_size) / population


def exposure_report(task_steps: int, population: int,
                    batch_size: int = BATCH_SIZE) -> dict:
    """The unambiguous exposure bookkeeping for one task.

    Reported instead of a single "exposures per item" number, because the two
    defensible readings differ slightly once the final batch is short.
    """
    return {
        "task_steps": task_steps,
        "population_passes": population_passes(task_steps, population, batch_size),
        "item_presentations": item_presentations(task_steps, population, batch_size),
        "presentations_per_item": presentations_per_item(task_steps, population,
                                                         batch_size),
        "batches_per_pass": batches_per_epoch(population, batch_size),
    }


# =========================================================  population  ====

def load_phase3_population(entries: Sequence[LexEntry], vocab: Vocab
                           ) -> Tuple[List[int], str]:
    """The EXACT Phase 2D3 subset3288, verified against its stored hash."""
    idx = select_nested_subset(entries, SUBSET_PER_BAND, subset_seed=0)
    digest = subset_definition_hash(subset_records(entries, idx, vocab))
    if digest != PHASE2D3_SUBSET_SHA256:
        raise RuntimeError(
            "Phase 3 population does not reproduce the Phase 2D3 subset3288: "
            f"{digest} != {PHASE2D3_SUBSET_SHA256}. Refusing to run on a "
            "different population.")
    return idx, digest


def full_lexicon_population(entries: Sequence[LexEntry]) -> List[int]:
    """Every canonical lexicon item, in the canonical bank order.

    This is exactly the population the historical full-lexicon repetition
    evaluator uses: no homophone filter, no frequency filter, no exclusion of
    items outside subset3288.
    """
    return list(range(len(entries)))


def out_of_subset_probe(entries: Sequence[LexEntry], subset_idx: Sequence[int],
                        n: int = DEFAULT_PROBE_N,
                        probe_seed: int = 0) -> List[int]:
    """A fixed, deterministic repetition probe drawn from OUTSIDE subset3288.

    Design note: the complement cannot support a band-balanced probe -- it
    holds only 177 items of the 1-1k band against 13,880 of 15k-end -- so a
    balanced design would cap at 708 items and would over-represent frequent
    words about five-fold relative to the lexicon.  Since the probe exists to
    track FULL-lexicon repetition, it is instead a uniform sample of the
    complement, which reproduces the lexicon's composition in expectation and
    includes homophones (all 2,889 of which lie outside subset3288).  Taking
    exactly `n = len(subset3288)` also makes the trained and untrained
    repetition numbers comparable at equal sample size.
    """
    complement = sorted(set(range(len(entries))) - set(subset_idx))
    if n > len(complement):
        raise RuntimeError(
            f"probe of {n} exceeds the {len(complement)}-item complement.")
    g = torch.Generator().manual_seed(PROBE_SEED_BASE + probe_seed)
    order = torch.randperm(len(complement), generator=g).tolist()
    probe = [complement[i] for i in order[:n]]
    if set(probe) & set(subset_idx):
        raise RuntimeError("probe overlaps subset3288; it must be disjoint.")
    return probe


def build_task_populations(entries: Sequence[LexEntry],
                           subset_idx: Sequence[int],
                           repetition_population: str = "subset"
                           ) -> Dict[str, List[int]]:
    """Per-task item populations. Only repetition may differ from subset3288."""
    if repetition_population not in REPETITION_POPULATIONS:
        raise ValueError(
            f"Unknown repetition_population {repetition_population!r}; "
            f"expected one of {REPETITION_POPULATIONS}.")
    rep = (full_lexicon_population(entries)
           if repetition_population == "full_lexicon" else list(subset_idx))
    if repetition_population == "full_lexicon":
        if not set(subset_idx) <= set(rep):
            raise RuntimeError("subset3288 is not contained in the repetition "
                               "population; the design is broken.")
        if len(rep) != len(entries):
            raise RuntimeError(f"full-lexicon repetition population is "
                               f"{len(rep)}, expected {len(entries)}.")
    return {"repetition": rep, "naming": list(subset_idx),
            "comprehension": list(subset_idx)}


def batches_per_epoch(population: int, batch_size: int = BATCH_SIZE) -> int:
    return -(-population // batch_size)


def infinite_batches(entries: Sequence[LexEntry], indices: Sequence[int],
                     bank_raw: torch.Tensor, vocab: Vocab, batch_size: int,
                     device: str, seed: int,
                     start_index: int = 0) -> Iterator[Dict[str, object]]:
    """Endless reshuffled batch stream with its OWN sampler state.

    Each task holds a separate generator, so changing one task's frequency
    cannot perturb another task's item ordering.

    `start_index` is the cumulative number of batches already consumed by this
    task.  Because each pass reshuffles with a seed that depends only on
    (seed, epoch), the stream is a pure function of the cumulative count: the
    epoch is start_index // batches_per_epoch and the offset within it is the
    remainder.  Resuming therefore replays no earlier epoch and reconstructs
    the exact next batch.
    """
    if start_index < 0:
        raise ValueError(f"start_index must be >= 0, got {start_index}")
    per_epoch = batches_per_epoch(len(indices), batch_size)
    epoch, skip = divmod(start_index, per_epoch)
    while True:
        for i, b in enumerate(make_batches(entries, indices, bank_raw, vocab,
                                           batch_size, device, shuffle=True,
                                           seed=seed * 100_000 + epoch)):
            if i < skip:
                continue
            yield b
        skip = 0
        epoch += 1


# ========================================================  diagnostics  ====

def grad_norms_by_group(model: DualRouteModel, loss: torch.Tensor
                        ) -> Dict[str, float]:
    """L2 grad norms from `loss` alone, without touching .grad or the weights."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True,
                                allow_unused=True)
    sq: Dict[str, float] = collections.defaultdict(float)
    for (name, _), g in zip(named, grads):
        if g is None:
            continue
        v = float((g.detach() ** 2).sum())
        sq["union"] += v
        s = side_of(name)
        if s:
            sq[s] += v
    return {k: v ** 0.5 for k, v in sq.items()}


def flat_grad(model: DualRouteModel, loss: torch.Tensor,
              prefixes: Sequence[str]) -> torch.Tensor:
    """Flattened gradient restricted to the parameters under `prefixes`."""
    named = [(n, p) for n, p in model.named_parameters()
             if p.requires_grad and n.startswith(tuple(prefixes))]
    grads = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True,
                                allow_unused=True)
    parts = [(g.detach().reshape(-1) if g is not None
              else torch.zeros(p.numel())) for (_, p), g in zip(named, grads)]
    return torch.cat(parts)


def gradient_interaction(model: DualRouteModel, batches: Dict[str, List[dict]],
                         pad_id: int) -> dict:
    """Descriptive gradient norms and pairwise cosines at initialisation.

    Cosines are computed ONLY on parameter groups both tasks actually touch:
    repetition-vs-comprehension on the encoder side, repetition-vs-naming on
    the decoder side.  Comprehension and naming have disjoint direct scopes in
    this pilot, so no whole-vector cosine between them is manufactured.
    """
    import numpy as np

    n_batches = min(len(v) for v in batches.values())
    norms: Dict[str, Dict[str, List[float]]] = {
        t: collections.defaultdict(list) for t in TASKS}
    cos_rc, cos_rn = [], []

    for k in range(n_batches):
        losses = {t: task_objective(model, t, batches[t][k], pad_id)["total"]
                  for t in TASKS}
        for t in TASKS:
            for key, val in grad_norms_by_group(model, losses[t]).items():
                norms[t][key].append(val)
        gr_e = flat_grad(model, losses["repetition"], ENCODER_SIDE)
        gc_e = flat_grad(model, losses["comprehension"], ENCODER_SIDE)
        gr_d = flat_grad(model, losses["repetition"], DECODER_SIDE)
        gn_d = flat_grad(model, losses["naming"], DECODER_SIDE)
        cos_rc.append(float(torch.nn.functional.cosine_similarity(
            gr_e, gc_e, dim=0)))
        cos_rn.append(float(torch.nn.functional.cosine_similarity(
            gr_d, gn_d, dim=0)))
    model.zero_grad(set_to_none=True)

    def stats(v: Sequence[float]) -> dict:
        a = np.asarray(v, dtype=float)
        return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else None,
                "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}

    return {
        "n_batches": n_batches,
        "grad_norms": {t: {k: stats(v) for k, v in norms[t].items()} for t in TASKS},
        "pairwise_cosines": {
            "repetition_vs_comprehension__encoder_side": stats(cos_rc),
            "repetition_vs_naming__decoder_side": stats(cos_rn),
            "note": ("comprehension and naming have disjoint direct parameter "
                     "scopes in this pilot; no cosine is reported for that pair "
                     "because a zero-padded whole-vector cosine would be "
                     "meaningless"),
        },
        "descriptive_only": ("no task weight, LR or schedule is derived from "
                             "these numbers"),
    }


def gradient_scope_audit(model: DualRouteModel, batches: Dict[str, List[dict]],
                         pad_id: int, lr: float, weight_decay: float) -> dict:
    """Per task: which parameters receive gradient, and what actually moves.

    Runs ONE real AdamW step per task on a throwaway optimizer and checks both
    the gradient support and the bit-identity of every frozen parameter.
    """
    expected = {
        "repetition": set(ENCODER_SIDE) | set(DECODER_SIDE),
        "comprehension": set(ENCODER_SIDE),
        "naming": set(DECODER_SIDE),
    }
    audit: Dict[str, dict] = {}
    for task in TASKS:
        before_all = parameter_fingerprint(model)
        dorsal_ref = dorsal_fingerprint(model)
        optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=lr, weight_decay=weight_decay)
        loss = task_objective(model, task, batches[task][0], pad_id)["total"]
        optim.zero_grad(set_to_none=True)
        loss.backward()
        with_grad = sorted(n for n, p in model.named_parameters()
                           if p.requires_grad and p.grad is not None
                           and torch.any(p.grad != 0))
        optim.step()
        moved = changed_parameters(model, before_all)
        assert_dorsal_untouched(model, dorsal_ref)      # raises on any violation

        sides = sorted({side_of(n) for n in with_grad} - {None})
        exp_sides = sorted({"encoder_side" if p in ENCODER_SIDE else "decoder_side"
                            for p in expected[task]})
        frozen_moved = [n for n in moved if not n.startswith(UNION_PREFIXES)]
        audit[task] = {
            "parameters_with_nonzero_grad": with_grad,
            "sides_touched": sides,
            "sides_expected": exp_sides,
            "sides_match_expectation": sides == exp_sides,
            "parameters_moved_by_one_step": moved,
            "all_moves_within_union_scope": not frozen_moved,
            "frozen_parameters_moved": frozen_moved,
            "dorsal_bit_identical": True,
        }
        # restore the pre-step weights so each task is audited from the same point
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(before_all[n])
        model.zero_grad(set_to_none=True)
    return audit


# =========================================================  evaluation  ====

@torch.no_grad()
def evaluate_all_tasks(model: DualRouteModel, vocab: Vocab,
                       entries: Sequence[LexEntry], bank_raw: torch.Tensor,
                       subset_idx: Sequence[int], device: str,
                       max_steps: int,
                       probe_idx: Optional[Sequence[int]] = None) -> dict:
    """All three tasks on subset3288, plus an optional out-of-subset probe.

    Naming and comprehension are always measured on subset3288.  Repetition is
    measured on subset3288 and, when a probe is supplied, on a fixed disjoint
    sample of the complement -- a cheap standing diagnostic of whether the
    historical mapping survives outside the rehearsed items.
    """
    rep = repetition_snapshot(model, vocab, entries, subset_idx, bank_raw,
                              device, include_teacher_forced=False)
    comp = evaluate_comprehension_subset(model, vocab, entries, bank_raw,
                                         subset_idx, device)
    nam = evaluate_naming(model, vocab, entries, bank_raw, subset_idx, device,
                          max_steps, return_per_item=True)
    rows = nam.pop("_per_item")
    nam["mean_pred_length"] = sum(r["pred_len"] for r in rows) / max(len(rows), 1)
    nam["mean_target_length"] = sum(r["length"] for r in rows) / max(len(rows), 1)
    out = {"repetition": rep["primary_readout"]["exact_match"],
           "repetition_convention": rep["primary_readout"]["convention"],
           "comprehension": comp, "naming": nam}
    if probe_idx is not None:
        pr = repetition_snapshot(model, vocab, entries, probe_idx, bank_raw,
                                 device, include_teacher_forced=False)
        out["repetition_probe"] = pr["primary_readout"]["exact_match"]
        out["repetition_probe_n"] = len(probe_idx)
    return out


def should_evaluate(step: int, eval_every: int = DEFAULT_EVAL_EVERY) -> bool:
    """Dense early grid, then the regular cadence. Step 0 is handled separately."""
    if step <= 0:
        return False
    if step in EVAL_STEPS_EARLY:
        return True
    return step % eval_every == 0


def evaluation_steps(total_steps: int,
                     eval_every: int = DEFAULT_EVAL_EVERY) -> List[int]:
    """The full planned evaluation grid for a budget (step 0 included)."""
    steps = {0}
    steps |= {s for s in EVAL_STEPS_EARLY if s <= total_steps}
    steps |= {s for s in range(eval_every, total_steps + 1, eval_every)}
    return sorted(steps)


# ==========================================  exactly resumable state  =====

def sampler_state(counts: Dict[str, int], populations: Dict[str, int],
                  batch_size: int) -> Dict[str, dict]:
    """Per-task data-iterator state, derived from cumulative batch counts.

    Each task carries its OWN population size, so a task rehearsing the full
    lexicon (463 batches per pass) and a task on subset3288 (52) resume
    correctly side by side.
    """
    out = {}
    for t in TASKS:
        per_epoch = batches_per_epoch(populations[t], batch_size)
        epoch, pos = divmod(counts[t], per_epoch)
        out[t] = {"cumulative_batches": counts[t], "epoch": epoch,
                  "position_in_epoch": pos, "seed": TASK_DATA_SEEDS[t],
                  "batches_per_epoch": per_epoch, "population": populations[t]}
    return out


def save_resumable(path: str, *, model: DualRouteModel,
                   optimizer: torch.optim.Optimizer, step: int,
                   counts: Dict[str, int], schedule: str,
                   ratio: Sequence[int], schedule_seed: int,
                   populations: Dict[str, int], batch_size: int, subset_sha256: str,
                   source_checkpoint_sha256: str, snapshots: List[dict],
                   trajectory: List[dict], streak: int,
                   first_met: Optional[int]) -> str:
    """Write a checkpoint sufficient for a BIT-EXACT resume.

    The schedule and the per-task samplers are pure functions of counters
    (see `task_schedule_stream` / `infinite_batches`), so their "RNG state" is
    those counters -- there is no hidden generator state to lose.  The global
    torch RNG is captured anyway: with ventral_noise = interference_noise = 0
    the training path draws from it nowhere, and storing it makes that
    assumption verifiable rather than assumed.
    """
    cycle, pos = divmod(step, MACRO_CYCLE_STEPS)
    state = {
        "format_version": RESUME_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "cycle_index": cycle,
        "position_in_cycle": pos,
        "task_steps": dict(counts),
        "schedule": schedule,
        "ratio": list(ratio),
        "schedule_seed": schedule_seed,
        "schedule_state": {
            "stateless": True,
            "derivation": "cycle = step // 6, position = step % 6",
            "note": "macro_cycle() reseeds a dedicated generator per cycle",
        },
        "task_sampler_state": sampler_state(counts, populations, batch_size),
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": (torch.cuda.get_rng_state_all()
                                 if torch.cuda.is_available() else None),
        "task_populations": dict(populations),
        "batch_size": batch_size,
        "subset_definition_sha256": subset_sha256,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "snapshots": snapshots,
        "trajectory": trajectory,
        "coexistence_streak": streak,
        "first_step_criterion_met": first_met,
        "note": "resumable training state; NOT a validation-selected checkpoint",
    }
    torch.save(state, path)
    return path


def load_resumable(path: str, *, schedule: str, subset_sha256: str,
                   populations: Dict[str, int], batch_size: int) -> dict:
    """Load and validate a resume checkpoint against the current run config.

    Format v1 (Phase 3A/3B) stored a single `population` shared by all tasks;
    it is accepted and widened to the per-task form so those runs stay
    resumable.
    """
    st = torch.load(path, map_location="cpu", weights_only=False)
    ver = st.get("format_version")
    if ver == 1 and "population" in st:
        st["task_populations"] = {t: int(st["population"]) for t in TASKS}
    elif ver != RESUME_FORMAT_VERSION:
        raise RuntimeError(
            f"resume format {ver!r} is neither 1 nor {RESUME_FORMAT_VERSION}; "
            "refusing an ambiguous resume.")
    for key, want, got in (("schedule", schedule, st.get("schedule")),
                           ("subset hash", subset_sha256,
                            st.get("subset_definition_sha256")),
                           ("task populations", dict(populations),
                            st.get("task_populations")),
                           ("batch size", batch_size, st.get("batch_size"))):
        if want != got:
            raise RuntimeError(
                f"resume {key} mismatch: checkpoint has {got!r}, this run "
                f"expects {want!r}. Refusing to resume into a different setup.")
    return st


# ===============================  global preservation (Phase 3C)  =========

# Canonical historical full-lexicon LTM repetition, the reference this phase
# asks whether joint training can preserve.
CANONICAL_FULL_LEXICON_LTM = 0.989449

# PRIMARY, binary, predeclared before the run.
GLOBAL_PRESERVATION = {"full_lexicon_ltm_min": 0.95}

# SECONDARY, descriptive only: it is reported alongside but never controls
# stopping, and must not be substituted for the primary criterion afterwards.
STRICT_PRESERVATION_MAX_DROP = 0.02      # -> LTM >= 0.969449


def global_preservation_met(full_lexicon_ltm: float) -> bool:
    """The PRIMARY global criterion. Nothing else decides Phase 3C success."""
    return full_lexicon_ltm >= GLOBAL_PRESERVATION["full_lexicon_ltm_min"]


def preservation_report(rep_exact: Dict[str, float]) -> dict:
    """Full-lexicon repetition plus both preservation readings."""
    ltm = rep_exact["ltm"]
    drop = CANONICAL_FULL_LEXICON_LTM - ltm
    return {
        "full": rep_exact["full"], "wm": rep_exact["wm"], "ltm": ltm,
        "canonical_ltm": CANONICAL_FULL_LEXICON_LTM,
        "absolute_ltm_drop_from_canonical": drop,
        "primary_criterion_ltm_ge_095": global_preservation_met(ltm),
        "secondary_strict_drop_le_002": drop <= STRICT_PRESERVATION_MAX_DROP,
        "secondary_note": ("descriptive only; does not control stopping and "
                           "must not replace the primary >=0.95 criterion"),
    }


class CoexistenceController:
    """Stopping logic for local coexistence and optional global preservation.

    With `require_global=False` this reproduces Phase 3A/3B exactly: two
    consecutive local successes stop the run.

    With `require_global=True` (Phase 3C) local confirmation only TRIGGERS a
    full-lexicon preservation check.  A failed check resets the streak, so
    training continues and the expensive evaluation is not repeated at every
    ordinary snapshot -- it can only fire again after two fresh consecutive
    local successes.
    """

    def __init__(self, require_global: bool = False) -> None:
        self.require_global = require_global
        self.streak = 0
        self.first_local_step: Optional[int] = None
        self.local_confirmations: List[int] = []
        self.global_checks: List[dict] = []
        self.global_success = False

    def observe_local(self, step: int, local_ok: bool) -> str:
        """Record a local snapshot. Returns 'continue', 'stop' or 'check_global'."""
        if local_ok:
            self.streak += 1
            if self.first_local_step is None:
                self.first_local_step = step
        else:
            self.streak = 0
            self.first_local_step = None
        if self.streak < CONSECUTIVE_REQUIRED:
            return "continue"
        self.local_confirmations.append(step)
        return "check_global" if self.require_global else "stop"

    def record_global(self, step: int, report: dict) -> str:
        """Record a global check. Returns 'stop' on success, else 'continue'."""
        self.global_checks.append({"step": step, **report})
        if report["primary_criterion_ltm_ge_095"]:
            self.global_success = True
            return "stop"
        self.streak = 0          # two fresh local successes needed to re-check
        self.first_local_step = None
        return "continue"


def coexistence_met(snapshot: dict) -> bool:
    """All three predeclared conditions at the SAME evaluation snapshot."""
    c, n, r = (snapshot["comprehension"], snapshot["naming"],
               snapshot["repetition"])
    return (c["top1"] >= COEXISTENCE["comprehension_top1_min"]
            and n["exact_match"] >= COEXISTENCE["naming_exact_min"]
            and n["whole_word_error_rate"] <= COEXISTENCE["naming_wer_max"]
            and r["ltm"] >= COEXISTENCE["ltm_repetition_exact_min"])


# ==============================================================  run  ======

def run_multitask(ckpt_path: str, schedule: str, out_dir: str,
                  total_steps: int = DEFAULT_TOTAL_STEPS,
                  eval_every: int = DEFAULT_EVAL_EVERY,
                  lr: float = 1e-4, weight_decay: float = 1e-5,
                  batch_size: int = BATCH_SIZE, schedule_seed: int = 0,
                  device: str = "cpu",
                  endpoint_full_repetition: bool = True,
                  resume_from: Optional[str] = None,
                  save_resume: bool = True,
                  repetition_population: str = "subset",
                  probe_n: int = DEFAULT_PROBE_N,
                  probe_seed: int = 0,
                  require_global_preservation: bool = False) -> dict:
    """One interleaved multitask run from the canonical checkpoint."""
    if schedule not in SCHEDULES:
        raise ValueError(f"Unknown schedule {schedule!r}; expected one of "
                         f"{sorted(SCHEDULES)}.")
    ratio = SCHEDULES[schedule]
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)

    model, vocab, entries, bank_raw, cfg, ckpt = load_frozen(ckpt_path, device)
    glove = require_real_glove(ckpt, expected_found=len(entries))
    subset_idx, digest = load_phase3_population(entries, vocab)
    verify_bank_mapping(entries, bank_raw, subset_idx)
    max_steps = cfg.data.max_phonemes + 1

    populations = build_task_populations(entries, subset_idx,
                                        repetition_population)
    pop_sizes = {t: len(populations[t]) for t in TASKS}
    probe_idx = (out_of_subset_probe(entries, subset_idx, probe_n, probe_seed)
                 if probe_n else None)
    probe_hash = (subset_definition_hash(subset_records(entries, probe_idx, vocab))
                  if probe_idx else None)
    rep_pop_hash = subset_definition_hash(
        subset_records(entries, populations["repetition"], vocab))

    trainable = set_multitask_scope(model)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay)               # ONE shared optimizer
    initial = parameter_fingerprint(model)
    dorsal_ref = dorsal_fingerprint(model)

    # ---- resume (optional): restore weights, optimizer, counters, RNG ----
    counts = {t: 0 for t in TASKS}
    start_step = 0
    resumed_from: Optional[dict] = None
    snapshots_pre: List[dict] = []
    trajectory_pre: List[dict] = []
    streak_pre, first_met_pre = 0, None
    if resume_from:
        st = load_resumable(resume_from, schedule=schedule,
                            subset_sha256=digest, populations=pop_sizes,
                            batch_size=batch_size)
        model.load_state_dict(st["model_state_dict"])
        optim.load_state_dict(st["optimizer_state_dict"])
        torch.set_rng_state(st["torch_rng_state"])
        if st.get("torch_cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["torch_cuda_rng_state"])
        counts = {t: int(st["task_steps"][t]) for t in TASKS}
        start_step = int(st["step"])
        snapshots_pre = st.get("snapshots", [])
        trajectory_pre = st.get("trajectory", [])
        streak_pre = int(st.get("coexistence_streak", 0))
        first_met_pre = st.get("first_step_criterion_met")
        resumed_from = {"path": os.path.abspath(resume_from),
                        "resumed_at_step": start_step,
                        "task_steps_at_resume": dict(counts)}
        print(f"[{schedule}] resumed at step {start_step} {counts}", flush=True)

    # Each task's stream is fast-forwarded by its own cumulative batch count.
    streams = {t: infinite_batches(entries, populations[t], bank_raw, vocab,
                                   batch_size, device, TASK_DATA_SEEDS[t],
                                   start_index=counts[t])
               for t in TASKS}
    running = {t: 0.0 for t in TASKS}
    running_n = {t: 0 for t in TASKS}

    def snapshot(step: int) -> dict:
        s = evaluate_all_tasks(model, vocab, entries, bank_raw, subset_idx,
                               device, max_steps, probe_idx=probe_idx)
        s["step"] = step
        s["task_steps"] = dict(counts)
        s["exposure"] = {t: exposure_report(counts[t], pop_sizes[t], batch_size)
                         for t in TASKS}
        s["coexistence_met"] = coexistence_met(s)
        return s

    if resumed_from:
        snapshots, trajectory = list(snapshots_pre), list(trajectory_pre)
        ctl = CoexistenceController(require_global_preservation)
        ctl.streak, ctl.first_local_step = streak_pre, first_met_pre
    else:
        snapshots = [snapshot(0)]
        trajectory = []
        print(f"[{schedule}] step 0: " + _fmt(snapshots[0]), flush=True)
        ctl = CoexistenceController(require_global_preservation)
        ctl.observe_local(0, snapshots[0]["coexistence_met"])
    stop_step, stop_reason = total_steps, "fixed budget exhausted"

    def full_lexicon_check(step: int) -> dict:
        """Canonical 29,571-word repetition evaluation. Evaluation only."""
        print(f"[{schedule}] step {step}: local coexistence confirmed -> "
              "running full-lexicon preservation check (slow by design)",
              flush=True)
        was_training = model.training
        rep = repetition_snapshot(model, vocab, entries,
                                  full_lexicon_population(entries), bank_raw,
                                  device, include_teacher_forced=False)
        model.train(was_training)
        rpt = preservation_report(rep["primary_readout"]["exact_match"])
        print(f"[{schedule}]   full-lexicon LTM={rpt['ltm']:.6f} "
              f"drop={rpt['absolute_ltm_drop_from_canonical']:.6f} "
              f"primary>=0.95={rpt['primary_criterion_ltm_ge_095']}", flush=True)
        return rpt

    model.train()
    for step, task in enumerate(task_schedule_stream(ratio, total_steps,
                                                     schedule_seed,
                                                     start_step=start_step),
                                start=start_step + 1):
        batch = next(streams[task])
        loss = task_objective(model, task, batch, vocab.pad_id)["total"]
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        counts[task] += 1
        running[task] += float(loss.detach())
        running_n[task] += 1

        if should_evaluate(step, eval_every):
            assert_dorsal_untouched(model, dorsal_ref)
            rec = {"step": step,
                   **{f"steps_{t}": counts[t] for t in TASKS},
                   **{f"loss_{t}": (running[t] / running_n[t]
                                    if running_n[t] else None) for t in TASKS}}
            trajectory.append(rec)
            running = {t: 0.0 for t in TASKS}
            running_n = {t: 0 for t in TASKS}

            snap = snapshot(step)
            snapshots.append(snap)
            ok = snap["coexistence_met"]
            decision = ctl.observe_local(step, ok)
            print(f"[{schedule}] step {step}: " + _fmt(snap)
                  + f" coexist={ok} streak={ctl.streak}", flush=True)
            if save_resume:
                save_resumable(os.path.join(out_dir, RESUME_FILENAME),
                               model=model, optimizer=optim, step=step,
                               counts=counts, schedule=schedule, ratio=ratio,
                               schedule_seed=schedule_seed,
                               populations=pop_sizes, batch_size=batch_size,
                               subset_sha256=digest,
                               source_checkpoint_sha256=sha256_file(ckpt_path),
                               snapshots=snapshots, trajectory=trajectory,
                               streak=ctl.streak,
                               first_met=ctl.first_local_step)
            if decision == "stop":
                stop_step = step
                stop_reason = (f"predeclared local coexistence criterion met at "
                               f"{CONSECUTIVE_REQUIRED} consecutive evaluations")
                break
            if decision == "check_global":
                snap["global_preservation_check"] = full_lexicon_check(step)
                if ctl.record_global(step, snap["global_preservation_check"]) == "stop":
                    stop_step = step
                    stop_reason = ("local coexistence confirmed AND global "
                                   "preservation criterion (full-lexicon LTM "
                                   ">= 0.95) satisfied")
                    break
            model.train()

    final = snapshot(stop_step)
    if snapshots[-1]["step"] != stop_step:
        snapshots.append(final)

    changed = changed_parameters(model, initial)
    outside = sorted(set(changed) - set(trainable))
    if outside:
        raise RuntimeError(f"Parameters outside the union scope changed: {outside}")

    endpoint_rep = None
    if require_global_preservation and not ctl.global_success:
        endpoint_full_repetition = True     # the cap must always be measured
    if endpoint_full_repetition:
        print(f"[{schedule}] endpoint full-lexicon repetition (slow by design)",
              flush=True)
        endpoint_rep = repetition_snapshot(
            model, vocab, entries, list(range(len(entries))), bank_raw, device,
            include_teacher_forced=False)

    ckpt_out = os.path.join(out_dir, f"final_step_{stop_step}.pt")
    torch.save({"model_state_dict": model.state_dict(), "schedule": schedule,
                "total_steps": stop_step, "task_steps": counts,
                "subset_definition_sha256": digest,
                "note": "fixed-budget or predeclared-criterion endpoint; NOT a "
                        "validation-selected checkpoint",
                "source_checkpoint_sha256": sha256_file(ckpt_path)}, ckpt_out)

    result = {
        "phase": "3A_joint_interleaved_multitask",
        "schedule": {"name": schedule, "ratio_repetition_naming_comprehension": list(ratio),
                     "macro_cycle_steps": MACRO_CYCLE_STEPS,
                     "schedule_seed": schedule_seed,
                     "interleaving": "per optimizer step (never per epoch)",
                     "example_first_cycles": [macro_cycle(ratio, schedule_seed, c)
                                              for c in range(4)]},
        "population": {"n": len(subset_idx),
                       "subset_definition_sha256": digest,
                       "source": "exact Phase 2D3 subset3288 (822 per frozen band, "
                                 "unique phonology), identical for all three tasks",
                       "comprehension_retrieval_bank": int(bank_raw.shape[0])},
        "budget": {"total_optimizer_steps": stop_step,
                   "max_total_optimizer_steps": total_steps,
                   "eval_every": eval_every, "batch_size": batch_size,
                   "evaluation_steps_planned": evaluation_steps(total_steps, eval_every),
                   "early_evaluation_steps": list(EVAL_STEPS_EARLY),
                   "lr": lr, "weight_decay": weight_decay,
                   "lr_schedule": "constant",
                   "optimizer": "ONE shared AdamW over the union scope; fresh; "
                                "checkpoint optimizer state never restored"},
        "task_steps": counts,
        "exposure": {t: exposure_report(counts[t], pop_sizes[t], batch_size)
                     for t in TASKS},
        "task_populations": {
            "repetition_population_kind": repetition_population,
            "sizes": pop_sizes,
            "batches_per_pass": {t: batches_per_epoch(pop_sizes[t], batch_size)
                                 for t in TASKS},
            "repetition_population_sha256": rep_pop_hash,
            "subset_contained_in_repetition_population": True,
        },
        "out_of_subset_probe": ({"n": len(probe_idx), "sha256": probe_hash,
                                 "probe_seed": probe_seed,
                                 "disjoint_from_subset_verified": True,
                                 "diagnostic_only": True}
                                if probe_idx else None),
        "objectives": objective_definitions(),
        "trainable_parameters": trainable,
        "always_frozen": list(ALWAYS_FROZEN),
        "coexistence_criterion": COEXISTENCE,
        "outcome": {
            "local_coexistence_confirmed": bool(ctl.local_confirmations),
            "local_confirmation_steps": ctl.local_confirmations,
            "first_local_crossing_step": ctl.first_local_step,
            "criterion_reached": (ctl.global_success if require_global_preservation
                                  else bool(ctl.local_confirmations)),
            "global_preservation_required": require_global_preservation,
            "global_preservation_success": ctl.global_success,
            "global_preservation_checks": ctl.global_checks,
            "stopped_at_step": stop_step, "stop_reason": stop_reason},
        "global_preservation_criterion": {
            "primary": GLOBAL_PRESERVATION,
            "canonical_full_lexicon_ltm": CANONICAL_FULL_LEXICON_LTM,
            "secondary_strict_max_drop": STRICT_PRESERVATION_MAX_DROP,
            "secondary_note": "descriptive only; never controls stopping"},
        "scope_audit": {"changed_parameters": sorted(changed),
                        "all_changes_within_union_scope": True,
                        "dorsal_bit_identical": True},
        "trajectory": trajectory,
        "snapshots": snapshots,
        "endpoint_full_lexicon_repetition": endpoint_rep,
        "endpoint_preservation_report": (
            preservation_report(endpoint_rep["primary_readout"]["exact_match"])
            if endpoint_rep else None),
        "resumed_from": resumed_from,
        "resume_checkpoint": (os.path.abspath(os.path.join(out_dir, RESUME_FILENAME))
                              if save_resume else None),
        "final_checkpoint": os.path.abspath(ckpt_out),
        "provenance": {
            "source_checkpoint_path": os.path.abspath(ckpt_path),
            "source_checkpoint_sha256": sha256_file(ckpt_path),
            "checkpoint_training_commit": ckpt.get("git_commit"),
            "lexicon_file_sha256": ckpt.get("lexicon_file_sha256"),
            "glove": glove, "device": device,
            "torch_version": torch.__version__,
            "task_data_seeds": TASK_DATA_SEEDS,
            "eval_git": git_state(ROOT),
            "runtime_seconds": round(time.time() - t_start, 1),
        },
    }
    with open(os.path.join(out_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    _write_tsv(os.path.join(out_dir, "trajectory.tsv"), trajectory)
    print(f"[{schedule}] done -> {out_dir}", flush=True)
    return result


def _fmt(s: dict) -> str:
    return (f"LTM_rep={s['repetition']['ltm']:.4f} "
            f"comp_top1={s['comprehension']['top1']:.4f} "
            f"naming={s['naming']['exact_match']:.4f}")


def objective_definitions() -> dict:
    return {
        "comprehension": {
            "formula": "C3 = C0 + 0.087 * retrieval_CE, tau = 0.10",
            "C0": "losses.alignment_loss verbatim ((1-cos) + 0.1*MSE)",
            "input": "phonology", "output": "s_hat", "target": "raw GloVe",
            "retrieval_competitors": "all 29,571 canonical GloVe vectors",
            "gradient_reaches": list(ENCODER_SIDE),
            "helper": "train_tasks.comprehension_objective(objective='c3')",
        },
        "naming": {
            "formula": "sequence CE (losses._seq_ce), teacher forcing 1.0",
            "path": "raw GloVe -> sem_to_h0 -> decoder -> dec_to_premotor -> frozen motor.proj",
            "gradient_reaches": list(DECODER_SIDE),
            "helper": "train_tasks.naming_objective",
        },
        "repetition": {
            "formula": "sequence CE (losses._seq_ce), teacher forcing 1.0",
            "path": ("phonology -> ltm.encoder -> ltm.to_semantic = s_hat -> "
                     "ltm.sem_to_h0 -> ltm.decoder -> ltm.dec_to_premotor -> "
                     "frozen motor.proj"),
            "composition": ("naming_forward(model, comprehension_forward(model, .), .) "
                            "-- the two validated Phase 2 forwards composed"),
            "identical_to": "model.route_logits(..., route='ltm')['logits']",
            "gradient_reaches": list(ENCODER_SIDE) + list(DECODER_SIDE),
            "not_used": ("losses.total_loss (carries gate/WM/alignment terms that "
                         "would confound this pilot); WM route; gate"),
        },
    }


# ==========================================================  preflight  ====

def preflight(ckpt_path: str, device: str, n_diag_batches: int,
              bench_steps: int, total_steps: int, out: Optional[str],
              repetition_population: str = "subset",
              probe_n: int = DEFAULT_PROBE_N, probe_seed: int = 0,
              full_lexicon_step0: bool = False) -> dict:
    """Everything that must hold BEFORE the long runs. Performs no training."""
    model, vocab, entries, bank_raw, cfg, ckpt = load_frozen(ckpt_path, device)
    glove = require_real_glove(ckpt, expected_found=len(entries))
    subset_idx, digest = load_phase3_population(entries, vocab)
    verify_bank_mapping(entries, bank_raw, subset_idx)
    max_steps = cfg.data.max_phonemes + 1

    trainable = set_multitask_scope(model)
    n_train = sum(p.numel() for n, p in model.named_parameters()
                  if n in set(trainable))
    n_total = sum(p.numel() for _, p in model.named_parameters())

    populations = build_task_populations(entries, subset_idx, repetition_population)
    pop_sizes = {t: len(populations[t]) for t in TASKS}
    probe_idx = (out_of_subset_probe(entries, subset_idx, probe_n, probe_seed)
                 if probe_n else None)
    batches = {t: list(_take(infinite_batches(entries, populations[t], bank_raw,
                                              vocab, BATCH_SIZE, device,
                                              TASK_DATA_SEEDS[t]), n_diag_batches))
               for t in TASKS}

    model.eval()
    scope = gradient_scope_audit(model, batches, vocab.pad_id, 1e-4, 1e-5)
    interaction = gradient_interaction(model, batches, vocab.pad_id)
    step0 = evaluate_all_tasks(model, vocab, entries, bank_raw, subset_idx,
                               device, max_steps, probe_idx=probe_idx)
    step0_full = None
    if full_lexicon_step0:
        step0_full = repetition_snapshot(
            model, vocab, entries, full_lexicon_population(entries), bank_raw,
            device, include_teacher_forced=False)["primary_readout"]["exact_match"]

    # ---- runtime benchmark: real fwd+bwd+step, then weights restored ----
    bench = {}
    before = parameter_fingerprint(model)
    for task in TASKS:
        optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=1e-4, weight_decay=1e-5)
        stream = infinite_batches(entries, populations[task], bank_raw, vocab,
                                  BATCH_SIZE, device, TASK_DATA_SEEDS[task])
        for _ in range(2):                                   # warm-up
            loss = task_objective(model, task, next(stream), vocab.pad_id)["total"]
            optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
        t0 = time.time()
        for _ in range(bench_steps):
            loss = task_objective(model, task, next(stream), vocab.pad_id)["total"]
            optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
        bench[task] = {"seconds_per_step": (time.time() - t0) / bench_steps,
                       "benchmarked_steps": bench_steps}
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(before[n])
    model.zero_grad(set_to_none=True)

    schedules = {}
    for name, ratio in SCHEDULES.items():
        counts = schedule_counts(ratio, total_steps)
        est = sum(counts[t] * bench[t]["seconds_per_step"] for t in TASKS)
        schedules[name] = {
            "ratio_repetition_naming_comprehension": list(ratio),
            "task_steps_at_budget": counts,
            "task_fraction": {t: counts[t] / total_steps for t in TASKS},
            "exposure": {t: exposure_report(counts[t], pop_sizes[t])
                         for t in TASKS},
            "first_four_macro_cycles": [macro_cycle(ratio, 0, c) for c in range(4)],
            "estimated_training_seconds": est,
            "estimated_training_hours": est / 3600.0,
        }

    res = {
        "phase": "3A0_multitask_preflight",
        "no_training_performed": True,
        "population": {"n": len(subset_idx), "subset_definition_sha256": digest,
                       "matches_phase2d3": digest == PHASE2D3_SUBSET_SHA256,
                       "comprehension_retrieval_bank": int(bank_raw.shape[0])},
        "scope": {"union_prefixes": list(UNION_PREFIXES),
                  "encoder_side": list(ENCODER_SIDE),
                  "decoder_side": list(DECODER_SIDE),
                  "trainable_parameters": trainable,
                  "n_trainable_tensors": len(trainable),
                  "n_trainable_scalars": n_train,
                  "n_total_scalars": n_total,
                  "gate_has_parameters": any(n.startswith("gate")
                                             for n, _ in model.named_parameters())},
        "objectives": objective_definitions(),
        "gradient_scope_audit": scope,
        "gradient_interaction": interaction,
        "step0_behaviour": step0,
        "step0_full_lexicon_repetition": step0_full,
        "task_populations": {
            "repetition_population_kind": repetition_population,
            "sizes": pop_sizes,
            "batches_per_pass": {t: batches_per_epoch(pop_sizes[t], BATCH_SIZE)
                                 for t in TASKS},
            "repetition_population_sha256": subset_definition_hash(
                subset_records(entries, populations["repetition"], vocab)),
            "subset_contained_in_repetition_population":
                set(subset_idx) <= set(populations["repetition"]),
        },
        "out_of_subset_probe": ({
            "n": len(probe_idx), "probe_seed": probe_seed,
            "sha256": subset_definition_hash(subset_records(entries, probe_idx, vocab)),
            "overlap_with_subset": len(set(probe_idx) & set(subset_idx)),
            "band_counts": population_composition(entries, probe_idx)["band_counts"],
            "diagnostic_only": True} if probe_idx else None),
        "benchmark": bench,
        "schedules_at_budget": {"total_optimizer_steps": total_steps, **schedules},
        "coexistence_criterion": COEXISTENCE,
        "evaluation_schedule": {
            "early_steps": list(EVAL_STEPS_EARLY),
            "then_every": DEFAULT_EVAL_EVERY,
            "planned_steps_at_budget": evaluation_steps(total_steps, DEFAULT_EVAL_EVERY),
            "rationale": ("Phase 2J showed catastrophic forgetting can complete "
                          "within <8k optimizer steps, which a 20k-only grid "
                          "would step over"),
        },
        "provenance": {"checkpoint_path": os.path.abspath(ckpt_path),
                       "checkpoint_sha256": sha256_file(ckpt_path),
                       "glove": glove, "device": device,
                       "torch_version": torch.__version__,
                       "eval_git": git_state(ROOT)},
    }
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"[preflight] -> {out}")
    return res


def _take(it: Iterator, n: int) -> Iterator:
    for _ in range(n):
        yield next(it)


# ===============================================================  main  ====

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("preflight", help="audit only; performs no training")
    pf.add_argument("--ckpt", required=True)
    pf.add_argument("--device", default="cpu")
    pf.add_argument("--n-diag-batches", type=int, default=12)
    pf.add_argument("--bench-steps", type=int, default=40)
    pf.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS)
    pf.add_argument("--out", default=None)
    pf.add_argument("--repetition-population", choices=list(REPETITION_POPULATIONS),
                    default="subset")
    pf.add_argument("--probe-n", type=int, default=DEFAULT_PROBE_N)
    pf.add_argument("--probe-seed", type=int, default=0)
    pf.add_argument("--full-lexicon-step0", action="store_true")

    rn = sub.add_parser("run", help="Phase 3A interleaved multitask run")
    rn.add_argument("--ckpt", required=True)
    rn.add_argument("--schedule", required=True, choices=sorted(SCHEDULES))
    rn.add_argument("--out-dir", required=True)
    rn.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS)
    rn.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    rn.add_argument("--lr", type=float, default=1e-4)
    rn.add_argument("--weight-decay", type=float, default=1e-5)
    rn.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    rn.add_argument("--schedule-seed", type=int, default=0)
    rn.add_argument("--device", default="cpu")
    rn.add_argument("--no-endpoint-full-repetition", action="store_true")
    rn.add_argument("--resume", default=None,
                    help="path to a resume_checkpoint.pt written by a previous "
                         "run of the SAME schedule/population")
    rn.add_argument("--no-save-resume", action="store_true")
    rn.add_argument("--repetition-population", choices=list(REPETITION_POPULATIONS),
                    default="subset",
                    help="'subset' reproduces Phase 3A/3B; 'full_lexicon' "
                         "rehearses all 29,571 words for repetition only")
    rn.add_argument("--probe-n", type=int, default=DEFAULT_PROBE_N)
    rn.add_argument("--probe-seed", type=int, default=0)
    rn.add_argument("--require-global-preservation", action="store_true",
                    help="Phase 3C: local coexistence only TRIGGERS a "
                         "full-lexicon check; success additionally requires "
                         "full-lexicon LTM >= 0.95")
    args = ap.parse_args(argv)

    if args.cmd == "run":
        res = run_multitask(args.ckpt, args.schedule, args.out_dir,
                            total_steps=args.total_steps, eval_every=args.eval_every,
                            lr=args.lr, weight_decay=args.weight_decay,
                            batch_size=args.batch_size,
                            schedule_seed=args.schedule_seed, device=args.device,
                            endpoint_full_repetition=not args.no_endpoint_full_repetition,
                            resume_from=args.resume,
                            save_resume=not args.no_save_resume,
                            repetition_population=args.repetition_population,
                            probe_n=args.probe_n, probe_seed=args.probe_seed,
                            require_global_preservation=args.require_global_preservation)
        print(json.dumps(res["outcome"], indent=2))
        return 0

    res = preflight(args.ckpt, args.device, args.n_diag_batches,
                    args.bench_steps, args.total_steps, args.out,
                    repetition_population=args.repetition_population,
                    probe_n=args.probe_n, probe_seed=args.probe_seed,
                    full_lexicon_step0=args.full_lexicon_step0)
    print(json.dumps({k: res[k] for k in
                      ("population", "gradient_interaction", "schedules_at_budget")},
                     indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
