#!/usr/bin/env python3
"""Phase 4 — joint multitask development from random initialization (H0 / J0).

Scientific question
-------------------
Can repetition, explicit lexical-semantic retrieval and true-GloVe naming
co-develop compatible representations *from initialization*, rather than being
grafted onto an already mature repetition system as in Phases 2 and 3?

The experiment is a 2x2 objective factorial at a fixed experimental seed S,
over the verified historical Lichtheim3 developmental recipe from scratch:

    regime   | retrieval_CE | naming_CE | loss added to the historical base
    ---------+--------------+-----------+----------------------------------
    h0       |     OFF      |    OFF    | (nothing)
    c_only   |     ON       |    OFF    | + LAMBDA_C * retrieval_CE
    n_only   |     OFF      |    ON     | + LAMBDA_N * naming_CE
    j0       |     ON       |    ON     | + both

Everything else is held identical across all four cells, including the
repetition and dorsal-pool batch sequences, so the only causal difference
between any two arms is which extra gradients are present. h0 and j0 keep
exactly the semantics under which their scientific runs were produced.

The design supports the retrieval main effect (c_only - h0), the naming main
effect (n_only - h0), and the interaction (j0 - c_only - n_only + h0).

Relationship to the historical cohort
-------------------------------------
seed19/e155 and seed22/e140 remain the canonical historical *maturity*
references.  They are NOT the paired control for J0: a fresh H0 at the same
seed is.  Historical checkpoints were produced by a two-job SLURM pipeline
whose stage boundary is reproduced here scientifically, not bit-exactly (see
below).

Provenance nuance — deliberate deviation from the historical execution
----------------------------------------------------------------------
Phase 4A0c established that the original epoch-100 -> epoch-101 resume
preserved model weights, the AdamW step counter and both moment buffers, and
changed only the learning rate (classification R3).  It also established that
the historical two-job resume carried an implementation defect: the dorsal
pseudoword pool's `itertools.cycle` cursor was never checkpointed, so on resume
the pool restarted from position 0 and drew a fresh shuffle permutation,
perturbing the global RNG stream from epoch 102 onward.

This driver reproduces the *scientific* recipe and does NOT reproduce that
defect.  The learning-rate boundary is applied in-process, and every sampling
stream is exactly resumable.  H0 is therefore described as a

    paired developmental control using the verified historical scientific recipe

and never as a bit-exact replay of the original historical execution.  The
distinction is recorded in `provenance.json` under `historical_fidelity`.

Determinism design
------------------
Every batch stream is *counter-addressed*: the k-th batch of a stream is a pure
function of (stream seed, k).  The epoch is `k // per_epoch`, the offset within
it is the remainder, and the epoch's item order is drawn from a private
`torch.Generator` seeded by (stream seed, epoch).  Three consequences:

  * no stream ever reads the global RNG, so evaluation, model forwards and the
    other tasks' sampling cannot perturb it;
  * exact resume needs only four integers (the cursors), not a serialized
    sampler state or a materialized index sequence;
  * mid-epoch resume is exact by construction, because the epoch order is
    recomputed rather than replayed.

With the canonical noises at 0.0 the training forward draws nothing from the
global RNG either (both noise sites in `models/` are guarded by `> 0`), so the
whole trajectory is a pure function of the seed and the step counter.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import Config, default_config, validate_split_config          # noqa: E402
from data.lexicon import LexEntry, build_lexicon, logfreq_weights         # noqa: E402
from data.phonemes import Vocab, build_vocab                              # noqa: E402
from losses import alignment_loss, total_loss                             # noqa: E402
from models.dual_route import DualRouteModel                              # noqa: E402
from utils.provenance import git_state, sha256_file                       # noqa: E402
from utils.seed import set_seed                                           # noqa: E402

from scripts.naming_comprehension.train_tasks import (                    # noqa: E402
    canonical_phonology_indices, comprehension_forward,
    evaluate_comprehension_subset, evaluate_naming,
    make_batches, naming_objective, repetition_snapshot, retrieval_loss,
    select_nested_subset, select_representative_subset, subset_definition_hash,
    subset_records, verify_bank_mapping,
)
from scripts.naming_comprehension.train_multitask import (                # noqa: E402
    MACRO_CYCLE_STEPS, TASKS, macro_cycle, out_of_subset_probe,
)


# ===========================================================================
#  FROZEN PHASE 4 SCIENTIFIC CONFIGURATION
#  Nothing critical is inherited silently from config.py: several config.py
#  defaults differ from the canonical historical values (enc/dec hidden 256 vs
#  128, bigru vs unigru, gate alpha 4.0 vs 2.0, gate threshold 0.5 vs 0.7,
#  interference noise 0.1 vs 0.0).  Every scientific parameter is named here.
# ===========================================================================

# The 2x2 objective factorial. Each regime is defined ONLY by which of the two
# additional developmental objectives is present; everything else -- the
# historical base loss, the populations, the streams, the optimizer, the LR
# schedule -- is identical across all four cells.
#
#     regime   | retrieval | naming
#     ---------+-----------+--------
#     h0       |    OFF    |  OFF
#     c_only   |    ON     |  OFF
#     n_only   |    OFF    |  ON
#     j0       |    ON     |  ON
#
# h0 and j0 keep exactly the semantics under which their scientific runs were
# produced (Phases 4A2a-4A2c); c_only and n_only are the two missing cells.
REGIMES = ("h0", "c_only", "n_only", "j0")
RETRIEVAL_REGIMES = ("c_only", "j0")
NAMING_REGIMES = ("n_only", "j0")


def objective_presence(regime: str) -> Dict[str, bool]:
    """Which additional objectives the factorial cell switches on."""
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
    return {"retrieval_enabled": regime in RETRIEVAL_REGIMES,
            "naming_enabled": regime in NAMING_REGIMES}

# ---- repetition population and lexicon ----
CANONICAL_LEXICON_PATH = "data/lexicon_en_glove_covered.tsv"
CANONICAL_MAX_WORDS = 30000
CANONICAL_N_WORDS = 29571          # entries actually surviving the filters
CANONICAL_BATCH_SIZE = 64
CANONICAL_FREQ_TEMP = 1.0
CANONICAL_DORSAL_POOL_SIZE = 4000

# ---- architecture ----
CANONICAL_HIDDEN = 128             # wm.hidden == ltm.enc_hidden == ltm.dec_hidden
CANONICAL_LTM_ENCODER_MODE = "unigru_last_hidden"
CANONICAL_INTERFERENCE_NOISE = 0.0
CANONICAL_VENTRAL_NOISE = 0.0
CANONICAL_GATE_ALPHA = 2.0
CANONICAL_GATE_THRESHOLD = 0.7
CANONICAL_USAGE_PRIOR = 0.5

# ---- optimization ----
CANONICAL_TEACHER_FORCING = 1.0
CANONICAL_WEIGHT_DECAY = 1e-5
CANONICAL_GRAD_CLIP = 1.0
CANONICAL_LOSS_WEIGHTS = {"rep": 1.0, "align": 1.0, "dec": 0.5,
                          "wm": 0.5, "gate": 0.05, "label_smoothing": 0.0}

# ---- historical two-stage learning rate ----
LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-4
# 100 historical epochs x ceil(29571/64) = 100 x 463 optimizer steps.
# Verified against all four archived checkpoints in Phase 4A0c.
LR_BOUNDARY_STEPS = 46_300

# ---- J0 additional objectives ----
TAU = 0.10                          # retrieval temperature (Phase 2 validated)
LAMBDA_C = 0.087                    # weight on retrieval CE
LAMBDA_N = 1.0                      # weight on naming CE
NAMING_MAX_STEPS = 10               # free-AR decode cap, never target length

# ---- C / N population: the frozen Phase 2C subset3288 ----
SUBSET_PER_BAND = 822
SUBSET_SEED = 0
EXPECTED_SUBSET_HASH = (
    "df48250092cdd8a6d37c33bc008b915f84a1e829ddaa2bafbaa593cce446d5cf")

# ---- FINAL population mode (Phase FINAL-1A) ----
# R = all 29,571 entries (unchanged); N = all 29,571 entries; C = one
# canonical target per exact phonological form (highest-frequency member per
# class, tie-break lowest bank index; see canonical_phonology_indices).  The
# retrieval bank stays the FULL 29,571-row GloVe bank in every mode, so the
# excluded homophone IDs remain retrieval competitors.  This makes 100% strict
# word-ID top-1 mathematically attainable on the C population (FINAL-0
# established the full-lexicon word-ID top-1 ceiling is 94.62% because
# homophones are bit-identical encoder inputs).
FINAL_FULL_MODE = "final_full"
EXPECTED_CANONICAL_C_N = 27_981
EXPECTED_CANONICAL_C_HASH = (
    "10c2f06eda769bf620ca3dbb9889204e4431cac2bfe0d0f5dd37fa4df2bb9f50")
HOMOPHONE_POLICY_NOTE = (
    "Comprehension uses ONE canonical lexical target per exact phonological "
    "encoder input (phoneme-ID sequence): the highest-frequency member of each "
    "phonological equivalence class (lowest LexEntry.rank; ties break to the "
    "lowest bank index, provably inert on the canonical lexicon). Excluded "
    "homophone IDs remain full-bank retrieval competitors and full members of "
    "the repetition and naming populations. Related but NOT identical to Ueno "
    "et al., who removed homophones from the training corpus outright."
)

# Developmental (cadence) evaluation in final_full mode runs on fixed
# deterministic samples so evaluation cost stays at the historical subset
# scale; full-population evaluation runs at the endpoint (--endpoint-eval).
# Seed namespace disjoint from the stream seeds (S*1_000_003+{0..3}), the
# subset selectors (0.., 1_000_000+..) and the probe (2_000_000+..).
DEV_EVAL_SIZE = 3288
DEV_EVAL_SEED = 3_000_000


def deterministic_sample(population: Sequence[int], n: int, seed: int) -> List[int]:
    """First `n` of a seeded permutation of `population`, returned sorted.

    A pure function of (population, n, seed); never touches the global RNG.
    """
    pop = [int(i) for i in population]
    if len(pop) <= n:
        return sorted(pop)
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(len(pop), generator=g).tolist()
    return sorted(pop[i] for i in perm[:n])

# ---- FINAL-3 task schedules ----
# "summed"          : the frozen FINAL-1 update -- R + pool + C + N summed into
#                     ONE backward, one global clip, one AdamW step.
# "interleaved_123" : Ueno/Lichtheim2-style.  ONE task per optimizer step, in
#                     macro-cycles of six steps holding exactly 1 R, 2 N and
#                     3 C.  Every step is its own zero_grad -> backward ->
#                     clip(1.0) -> step on the SHARED AdamW state.  All model
#                     parameters stay trainable (unlike the Phase-3 driver,
#                     which warm-started and froze WM/phon_embed/motor).
SUMMED_SCHEDULE = "summed"
INTERLEAVED_123 = "interleaved_123"
SCHEDULES = (SUMMED_SCHEDULE, INTERLEAVED_123)
# Ratio in train_multitask.TASKS order: (repetition, naming, comprehension).
RATIO_123 = (1, 2, 3)
SCHEDULE_RATIOS = {INTERLEAVED_123: RATIO_123}
# The task-order permutation lives in its own seed namespace, disjoint from the
# four data streams (offsets 0..3) so a schedule can never share a generator
# seed with a sampler.
SCHEDULE_SEED_OFFSET = 4
TASK_ORDER_POLICY = (
    "deterministically shuffled macro-cycle: the six labels of cycle c are "
    "macro_cycle(ratio, schedule_seed, c), a pure function of (schedule_seed, "
    "cycle index) drawn from a private torch.Generator. It never reads the "
    "global RNG and needs no stored shuffle state, so cycle position is "
    "recomputed from global_step alone and exact resume is automatic.")


def derive_schedule_seed(seed: int) -> int:
    """Task-order seed for experimental seed S (documented rule, not a magic
    literal), in the same namespace as `derive_stream_seeds` but disjoint."""
    return int(seed) * STREAM_SEED_STRIDE + SCHEDULE_SEED_OFFSET


# ---- default scientific run length (the Phase 4A2 knob) ----
# 160 R epochs spans the whole canonical stable-zero window (e140 seed22,
# e155 seed19) with margin.  Recorded explicitly rather than assumed.
DEFAULT_EPOCHS = 160

# ---- RNG ownership.  Task seeds are derived from the experimental seed S by a
# documented rule, never by magic literals scattered through the code. ----
STREAM_NAMES = ("repetition", "pool", "comprehension", "naming")
STREAM_SEED_OFFSET = {"repetition": 0, "pool": 1, "comprehension": 2, "naming": 3}
STREAM_SEED_STRIDE = 1_000_003      # prime; keeps distinct S far apart
EPOCH_SEED_STRIDE = 7_919           # prime; keeps distinct epochs far apart


def derive_stream_seeds(seed: int) -> Dict[str, int]:
    """Deterministic per-stream seeds from the single experimental seed S.

    `S * STREAM_SEED_STRIDE + offset` with offsets 0..3 cannot collide across
    seeds, and the mapping is reversible by inspection, which is what makes the
    saved provenance auditable.
    """
    return {name: int(seed) * STREAM_SEED_STRIDE + STREAM_SEED_OFFSET[name]
            for name in STREAM_NAMES}


# ===========================================================================
#  Counter-addressed batch streams
# ===========================================================================

class CounterStream:
    """Endless batch stream whose k-th batch is a pure function of (seed, k).

    `population` is a list of indices into the owning entry list (bank indices
    for the lexical streams, pool positions for the dorsal pool).  One "epoch"
    is one pass of `len(population)` draws, i.e. `ceil(n / batch_size)` batches,
    matching the historical `len(train_loader) = 463` for the full lexicon.

    With `weights=None` an epoch is a permutation (sampling without
    replacement); with weights it is `torch.multinomial(..., replacement=True)`,
    which reproduces `WeightedRandomSampler(weights, num_samples=n,
    replacement=True)`.  In both cases the draw uses a private generator seeded
    from (stream seed, epoch), never the global RNG.
    """

    def __init__(self, name: str, population: Sequence[int], batch_size: int,
                 seed: int, weights: Optional[np.ndarray] = None) -> None:
        if len(population) == 0:
            raise ValueError(f"stream {name!r} has an empty population")
        if batch_size <= 0:
            raise ValueError(f"stream {name!r}: batch_size must be > 0")
        self.name = name
        self.population = [int(i) for i in population]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.n = len(self.population)
        self.per_epoch = -(-self.n // self.batch_size)
        self.weights = (None if weights is None
                        else torch.as_tensor(weights, dtype=torch.double))
        if self.weights is not None and len(self.weights) != self.n:
            raise ValueError(f"stream {name!r}: weights/population length mismatch")
        self._cache_epoch: Optional[int] = None
        self._cache_order: Optional[List[int]] = None

    def epoch_seed(self, epoch: int) -> int:
        return self.seed * EPOCH_SEED_STRIDE + int(epoch)

    def epoch_order(self, epoch: int) -> List[int]:
        """Item order for one pass. Recomputed, never replayed — this is what
        makes mid-epoch resume exact without persisting an index sequence."""
        if self._cache_epoch == epoch and self._cache_order is not None:
            return self._cache_order
        g = torch.Generator().manual_seed(self.epoch_seed(epoch))
        if self.weights is None:
            pos = torch.randperm(self.n, generator=g)
        else:
            pos = torch.multinomial(self.weights, self.n,
                                    replacement=True, generator=g)
        order = [self.population[p] for p in pos.tolist()]
        self._cache_epoch, self._cache_order = int(epoch), order
        return order

    def indices(self, cursor: int) -> List[int]:
        """Indices of the `cursor`-th batch of the endless stream."""
        if cursor < 0:
            raise ValueError(f"stream {self.name!r}: negative cursor {cursor}")
        epoch, off = divmod(int(cursor), self.per_epoch)
        order = self.epoch_order(epoch)
        return order[off * self.batch_size:(off + 1) * self.batch_size]

    def location(self, cursor: int) -> dict:
        epoch, off = divmod(int(cursor), self.per_epoch)
        return {"stream": self.name, "cursor": int(cursor), "seed": self.seed,
                "epoch": epoch, "batch_in_epoch": off,
                "batches_per_epoch": self.per_epoch, "population": self.n}


def build_batch(entries: Sequence[LexEntry], bank_raw: torch.Tensor,
                vocab: Vocab, indices: Sequence[int], device: str) -> dict:
    """One batch, built by the validated Phase 2 collation (`make_batches`).

    Called with `batch_size = len(indices)` and `shuffle=False`, so the ordering
    decision belongs entirely to the stream and this function stays a pure
    collator.
    """
    if not indices:
        raise ValueError("cannot build a batch from an empty index list")
    return next(iter(make_batches(entries, list(indices), bank_raw, vocab,
                                  len(indices), device, shuffle=False)))


# ===========================================================================
#  Canonical configuration
# ===========================================================================

def canonical_config(seed: int, device: str, *, max_words: int,
                     lexicon_path: str, dorsal_pool_size: int,
                     batch_size: int,
                     glove_path: Optional[str] = "data/glove.6B.300d.txt") -> Config:
    """A Config with every scientifically relevant field set explicitly."""
    cfg = default_config()

    cfg.data.use_real = True
    cfg.data.lexicon_path = lexicon_path
    cfg.data.glove_path = glove_path
    cfg.data.max_words = int(max_words)
    cfg.data.freq_temp = CANONICAL_FREQ_TEMP
    cfg.data.split_mode = "full_lexicon"
    cfg.data.val_fraction = 0.0
    cfg.data.split_seed = 0
    cfg.data.seed = 0

    cfg.wm.hidden = CANONICAL_HIDDEN
    cfg.wm.interference_noise = CANONICAL_INTERFERENCE_NOISE

    cfg.ltm.enc_hidden = CANONICAL_HIDDEN
    cfg.ltm.dec_hidden = CANONICAL_HIDDEN
    cfg.ltm.ltm_encoder_mode = CANONICAL_LTM_ENCODER_MODE
    cfg.ltm.ventral_noise = CANONICAL_VENTRAL_NOISE
    cfg.ltm.__post_init__()          # re-normalise bidirectional_encoder

    cfg.gating.alpha = CANONICAL_GATE_ALPHA
    cfg.gating.gate_threshold = CANONICAL_GATE_THRESHOLD
    cfg.gating.usage_prior = CANONICAL_USAGE_PRIOR

    for k, v in CANONICAL_LOSS_WEIGHTS.items():
        setattr(cfg.loss, k, v)

    cfg.train.seed = int(seed)
    cfg.train.device = device
    cfg.train.batch_size = int(batch_size)
    cfg.train.lr = LR_STAGE1
    cfg.train.weight_decay = CANONICAL_WEIGHT_DECAY
    cfg.train.grad_clip = CANONICAL_GRAD_CLIP
    cfg.train.teacher_forcing_ratio = CANONICAL_TEACHER_FORCING
    cfg.train.dorsal_pool_size = int(dorsal_pool_size)
    cfg.train.num_workers = 0

    validate_split_config(cfg.data)
    return cfg


# ===========================================================================
#  Learning-rate policy (FINAL-4)
# ===========================================================================
#
# Two policies exist, and a run declares exactly one:
#
#   "two_stage_rep_cursor" : the historical schedule -- LR_STAGE1 while the
#       repetition cursor is below the boundary, LR_STAGE2 after.  This is the
#       default and is what every run through FINAL-3 used; nothing about it
#       changes.
#
#   "task_specific" : one fixed learning rate per task, applied on that task's
#       own optimizer step.  Only meaningful for interleaved schedules, where
#       a step trains exactly one task.  It REPLACES the two-stage schedule
#       rather than modulating it, so the declared eta values are absolute.
#
# The policy is scientific state: it is stored in the checkpoint, and resuming
# with a different one is refused unless an explicit phase transition is
# declared (see JointScratchTrainer.load_state_dict).
LR_POLICY_TWO_STAGE = "two_stage_rep_cursor"
LR_POLICY_TASK = "task_specific"
LR_TASKS = ("repetition", "naming", "comprehension")


def two_stage_lr_policy(boundary: int) -> Dict[str, object]:
    return {"kind": LR_POLICY_TWO_STAGE, "stage1": LR_STAGE1,
            "stage2": LR_STAGE2, "boundary_rep_batches": int(boundary)}


def task_lr_policy(repetition: float, naming: float,
                   comprehension: float) -> Dict[str, object]:
    for name, v in (("repetition", repetition), ("naming", naming),
                    ("comprehension", comprehension)):
        if not (v > 0):
            raise ValueError(f"task learning rate {name}={v} must be > 0")
    return {"kind": LR_POLICY_TASK, "repetition": float(repetition),
            "naming": float(naming), "comprehension": float(comprehension)}


def lr_for_step(rep_cursor: int, boundary: int) -> float:
    """Historical two-stage LR as a pure function of REPETITION progress.

    The argument counts COMPLETED repetition batches, so the R batch numbered
    boundary (the 46,300th = 100 repetition epochs) still runs at stage-1 and
    the next one is the first at stage-2 — exactly the historical e100/e101
    split, whose meaning was always tied to the repetition recipe.

    Under the summed schedule the repetition cursor equals `global_step` at
    every point (R is drawn on every update in every regime), so this is
    numerically identical to the previous global-step rule and FINAL-1 is
    unaffected.  Under an interleaved schedule the two diverge, and anchoring
    to the repetition stream is what preserves the historical meaning of the
    boundary rather than silently moving it.
    """
    return LR_STAGE1 if int(rep_cursor) < int(boundary) else LR_STAGE2


def lr_phase(rep_cursor: int, boundary: int) -> str:
    return "stage1_lr1e-3" if int(rep_cursor) < int(boundary) else "stage2_lr1e-4"


# ===========================================================================
#  RNG state serialization contract
# ===========================================================================
#
# Checkpoints store every global RNG as CPU uint8 ByteTensors, the only form
# `torch.set_rng_state` and `torch.cuda.set_rng_state_all` accept.
#
# `torch.load(..., map_location=device)` relocates EVERY tensor in the file,
# RNG states included, so a checkpoint reloaded on a GPU node hands back CUDA
# tensors.  Those are not `torch.ByteTensor` (a CPU type) and the setters
# reject them with "RNG state must be a torch.ByteTensor".  The tensors'
# VALUES survive the round trip untouched, so moving them back to CPU
# restores the state exactly: this is a representation repair, never a
# re-seed, and it therefore cannot change any trajectory.
#
# Same contract, and the same helper name, as the historical driver
# (`scripts/train_checkpoint.py::_as_cpu_byte_tensor`), which already solved
# this for the Phase-1 resume path.

def _as_cpu_byte_tensor(x: object, name: str = "rng_state") -> torch.Tensor:
    """Normalise one saved RNG state to a CPU torch.uint8 ByteTensor."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(torch.uint8)
    try:
        return torch.as_tensor(x, dtype=torch.uint8, device="cpu")
    except Exception as e:                                   # pragma: no cover
        raise TypeError(
            f"Could not convert {name} to a CPU torch.uint8 ByteTensor "
            f"(type={type(x)!r}, dtype={getattr(x, 'dtype', None)}, "
            f"device={getattr(x, 'device', None)})") from e


def capture_rng_states() -> Dict[str, object]:
    """Every global RNG, in exactly the form `restore_rng_states` expects.

    CUDA states are a LIST of CPU uint8 ByteTensors, one per visible device,
    which is what `torch.cuda.set_rng_state_all` consumes.
    """
    return {
        "torch": _as_cpu_byte_tensor(torch.get_rng_state(), "torch"),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "cuda": ([_as_cpu_byte_tensor(s, f"cuda:{i}")
                  for i, s in enumerate(torch.cuda.get_rng_state_all())]
                 if torch.cuda.is_available() else None),
    }


def restore_rng_states(rs: Dict[str, object]) -> List[str]:
    """Restore the global RNGs saved by `capture_rng_states`.

    Returns the names actually restored.  Accepts checkpoints written by any
    earlier version of this driver, and checkpoints whose tensors were
    relocated by `map_location` (see the module note above).  A CUDA state is
    restored only when CUDA is available, so a GPU checkpoint resumes cleanly
    on a CPU-only machine (and vice versa).
    """
    restored: List[str] = []
    if rs.get("torch") is not None:
        torch.set_rng_state(_as_cpu_byte_tensor(rs["torch"], "torch"))
        restored.append("torch")
    if rs.get("numpy") is not None:
        np.random.set_state(rs["numpy"])
        restored.append("numpy")
    if rs.get("python") is not None:
        random.setstate(rs["python"])
        restored.append("python")

    cuda = rs.get("cuda")
    if cuda is None or not torch.cuda.is_available():
        return restored
    if torch.is_tensor(cuda):            # tolerate a single-device scalar form
        cuda = [cuda]
    states = [_as_cpu_byte_tensor(s, f"cuda:{i}") for i, s in enumerate(cuda)]
    n_dev = torch.cuda.device_count()
    if len(states) != n_dev:
        # Honest partial restore rather than a crash or a silent re-seed.
        print(f"[resume] WARNING: checkpoint holds {len(states)} CUDA RNG "
              f"state(s) but {n_dev} device(s) are visible; restoring "
              f"{min(len(states), n_dev)}.")
        states = states[:n_dev]
    if states:
        torch.cuda.set_rng_state_all(states)
        restored.append(f"cuda[{len(states)}]")
    return restored


@contextlib.contextmanager
def preserved_rng():
    """Restore every global RNG on exit.

    The training streams do not read the global RNG, so this is belt and
    braces; it makes evaluation neutrality hold even if a future evaluator
    starts drawing.  Tested by `test_evaluation_rng_neutrality`.
    """
    t_state = torch.get_rng_state()
    np_state = np.random.get_state()
    py_state = random.getstate()
    cuda_state = (torch.cuda.get_rng_state_all()
                  if torch.cuda.is_available() else None)
    try:
        yield
    finally:
        torch.set_rng_state(t_state)
        np.random.set_state(np_state)
        random.setstate(py_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


# ===========================================================================
#  Dorsal pseudoword pool
# ===========================================================================

def build_pool_entries(vocab: Vocab, n: int, semantic_dim: int, seed: int,
                       min_len: int = 2, max_len: int = 9) -> List[LexEntry]:
    """The historical pseudoword pool, generated by the same rule as
    `data.dataset.build_pool_loader` (a `random.Random(seed)` walk over
    (C)V(C) templates), but returned as a plain entry list so this driver can
    own the sampling order and make it resumable."""
    rng = random.Random(seed)
    cons = [vocab.stoi[s] for s in vocab.itos[3:] if vocab.sonority[vocab.stoi[s]] < 0.9]
    vow = [vocab.stoi[s] for s in vocab.itos[3:] if vocab.sonority[vocab.stoi[s]] >= 0.95]
    entries: List[LexEntry] = []
    seen = set()
    while len(entries) < n:
        f: List[int] = []
        for _ in range(rng.randint(1, 3)):
            if rng.random() < 0.85:
                f.append(rng.choice(cons))
            f.append(rng.choice(vow))
            if rng.random() < 0.4:
                f.append(rng.choice(cons))
        if not (min_len <= len(f) <= max_len) or tuple(f) in seen:
            continue
        seen.add(tuple(f))
        entries.append(LexEntry(word="", phonemes=f,
                                semantic=np.zeros(semantic_dim, np.float32),
                                freq=1.0, rank=1))
    return entries


# ===========================================================================
#  Trainer
# ===========================================================================

class JointScratchTrainer:
    """Owns the model, the optimizer, the four streams and the run state."""

    def __init__(self, *, regime: str, seed: int, device: str,
                 max_words: int, lexicon_path: str, dorsal_pool_size: int,
                 batch_size: int, subset_mode: str, subset_per_band: int,
                 subset_size: int, lr_boundary_steps: int,
                 allow_glove_fallback: bool, require_subset_hash: bool,
                 glove_path: Optional[str] = "data/glove.6B.300d.txt",
                 c_align_weight: float = 0.0,
                 schedule: str = SUMMED_SCHEDULE,
                 task_lrs: Optional[Dict[str, float]] = None,
                 allow_phase_transition: bool = False) -> None:
        presence = objective_presence(regime)          # validates the regime
        self.regime = regime
        self.retrieval_enabled = presence["retrieval_enabled"]
        self.naming_enabled = presence["naming_enabled"]
        # Which counter-addressed streams this cell actually consumes.
        self.stream_active = {
            "repetition": True, "pool": True,
            "comprehension": self.retrieval_enabled,
            "naming": self.naming_enabled,
        }
        self.seed = int(seed)
        self.device = device
        self.lr_boundary_steps = int(lr_boundary_steps)
        # FINAL-3 task schedule.  "summed" = the frozen FINAL-1 update.
        if schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}, got {schedule!r}")
        self.schedule = schedule
        self.schedule_seed = derive_schedule_seed(seed)
        self.ratio = SCHEDULE_RATIOS.get(schedule)
        if schedule != SUMMED_SCHEDULE and not (
                self.retrieval_enabled and self.naming_enabled):
            # An interleaved cycle emits N and C steps by construction, so a
            # regime that switches those objectives off cannot execute it.
            raise RuntimeError(
                f"schedule {schedule!r} emits naming and comprehension steps and "
                f"therefore requires a regime with both objectives enabled; "
                f"regime {regime!r} has retrieval={self.retrieval_enabled}, "
                f"naming={self.naming_enabled}.")

        # FINAL-4 learning-rate policy.  Absent task LRs => the historical
        # two-stage schedule, bit-for-bit as before.
        if task_lrs is None:
            self.lr_policy = two_stage_lr_policy(self.lr_boundary_steps)
        else:
            missing = [t for t in LR_TASKS if t not in task_lrs]
            if missing:
                raise ValueError(
                    f"task-specific LR needs every task; missing {missing}")
            if schedule == SUMMED_SCHEDULE:
                # One summed update trains all tasks at once, so a per-task LR
                # has no meaning there; refuse rather than silently pick one.
                raise RuntimeError(
                    "task-specific learning rates require an interleaved "
                    "schedule (a summed step trains every task at once)")
            self.lr_policy = task_lr_policy(**{t: task_lrs[t] for t in LR_TASKS})
        self.allow_phase_transition = bool(allow_phase_transition)
        self.phase_transitions: List[dict] = []

        # FINAL-2A knob.  0.0 = the frozen FINAL-1 objective.
        self.c_align_weight = float(c_align_weight)
        if self.c_align_weight < 0.0:
            raise ValueError(
                f"c_align_weight must be >= 0, got {self.c_align_weight}")
        if self.c_align_weight > 0.0 and not self.retrieval_enabled:
            # The C stream is only drawn when retrieval is on, so a positive
            # weight here would be a silent no-op rather than an experiment.
            raise RuntimeError(
                f"c_align_weight={self.c_align_weight} requires the "
                f"comprehension stream, which regime {regime!r} does not draw. "
                f"Use a regime with retrieval enabled ({RETRIEVAL_REGIMES}).")

        self.cfg = canonical_config(
            seed, device, max_words=max_words, lexicon_path=lexicon_path,
            dorsal_pool_size=dorsal_pool_size, batch_size=batch_size,
            glove_path=glove_path)

        # ---- lexicon and bank, built BEFORE the model so that model
        # initialization consumes a freshly seeded RNG and is therefore
        # identical between H0 and J0 at the same seed. ----
        self.vocab = build_vocab()
        self.lexicon = build_lexicon(self.cfg.data, self.vocab)
        self.entries: List[LexEntry] = list(self.lexicon.entries)
        stats = self.lexicon.load_stats
        self.glove_found = int(getattr(stats, "n_glove_found", 0))
        self.glove_fallback = int(getattr(stats, "n_glove_fallback", 0))
        if self.glove_fallback and not allow_glove_fallback:
            raise RuntimeError(
                f"{self.glove_fallback} of {len(self.entries)} entries fall back to "
                f"pseudo-vectors instead of real GloVe. A scientific Phase 4 run "
                f"requires real GloVe for every word; pass --allow-glove-fallback "
                f"only for mechanical smoke tests.")
        self.bank_raw = torch.stack(
            [torch.tensor(e.semantic) for e in self.entries]).float()

        # ---- model + optimizer ----
        set_seed(self.seed)
        self.model = DualRouteModel(self.cfg, self.vocab).to(device)
        self.model.set_semantic_bank(self.bank_raw.to(device))
        self.optim = torch.optim.AdamW(
            self.model.parameters(), lr=LR_STAGE1,
            weight_decay=self.cfg.train.weight_decay)

        # ---- C / N populations ----
        # Legacy modes ("nested", "representative"): C and N share one subset
        # population, exactly as in Phases 2-4.  FINAL_FULL_MODE: C = the
        # canonical one-target-per-phonology population, N = the full lexicon.
        self.subset_mode = subset_mode
        if subset_mode == FINAL_FULL_MODE:
            self.comp_idx = canonical_phonology_indices(self.entries)
            self.naming_idx = list(range(len(self.entries)))
            self.comp_population_name = "canonical_phonology"
            self.naming_population_name = "full_lexicon"
            self.comp_hash = subset_definition_hash(
                subset_records(self.entries, self.comp_idx, self.vocab))
            self.naming_hash = subset_definition_hash(
                subset_records(self.entries, self.naming_idx, self.vocab))
            # The frozen expectations apply exactly when the lexicon is the
            # canonical 29,571-entry GloVe-covered lexicon.
            if require_subset_hash:
                if len(self.entries) != CANONICAL_N_WORDS:
                    raise RuntimeError(
                        f"final_full expects the canonical {CANONICAL_N_WORDS}-entry "
                        f"lexicon, got {len(self.entries)} entries.")
                if len(self.comp_idx) != EXPECTED_CANONICAL_C_N:
                    raise RuntimeError(
                        f"canonical comprehension population has {len(self.comp_idx)} "
                        f"targets, expected {EXPECTED_CANONICAL_C_N}.")
                if self.comp_hash != EXPECTED_CANONICAL_C_HASH:
                    raise RuntimeError(
                        f"canonical C population hash mismatch:\n"
                        f"  expected {EXPECTED_CANONICAL_C_HASH}\n"
                        f"  got      {self.comp_hash}")
            # Compatibility aliases: the legacy checkpoint/provenance fields
            # named subset_* refer to the COMPREHENSION population in this mode.
            self.subset_idx = self.comp_idx
            self.subset_hash = self.comp_hash
        else:
            if subset_mode == "nested":
                self.subset_idx = select_nested_subset(
                    self.entries, subset_per_band, subset_seed=SUBSET_SEED)
            elif subset_mode == "representative":
                self.subset_idx = select_representative_subset(
                    self.entries, subset_size, subset_seed=SUBSET_SEED)
            else:
                raise ValueError(f"unknown subset_mode {subset_mode!r}")
            self.subset_hash = subset_definition_hash(
                subset_records(self.entries, self.subset_idx, self.vocab))
            if require_subset_hash and self.subset_hash != EXPECTED_SUBSET_HASH:
                raise RuntimeError(
                    f"subset definition hash mismatch:\n"
                    f"  expected {EXPECTED_SUBSET_HASH}\n"
                    f"  got      {self.subset_hash}\n"
                    f"The C/N population is not the frozen Phase 2C subset3288.")
            self.comp_idx = self.naming_idx = self.subset_idx
            self.comp_population_name = self.naming_population_name = (
                f"subset{len(self.subset_idx)}_{subset_mode}")
            self.comp_hash = self.naming_hash = self.subset_hash
        verify_bank_mapping(self.entries, self.bank_raw, self.comp_idx)
        if self.naming_idx is not self.comp_idx:
            verify_bank_mapping(self.entries, self.bank_raw,
                                self.naming_idx[::97])

        # ---- out-of-subset repetition probe ----
        # Meaningful only when the C/N populations leave a complement.  In
        # FINAL_FULL_MODE the naming/repetition populations cover the whole
        # lexicon, so the probe is explicitly disabled rather than faked.
        if subset_mode == FINAL_FULL_MODE:
            self.probe_idx: List[int] = []
            self.probe_note = (
                "disabled: naming and repetition populations cover the full "
                "lexicon, so no out-of-subset complement exists")
        else:
            self.probe_idx = out_of_subset_probe(self.entries, self.subset_idx,
                                                 n=len(self.subset_idx))
            self.probe_note = "uniform sample of the subset complement"

        # ---- developmental-evaluation populations ----
        # Legacy modes evaluate on the subset itself (bit-identical to the
        # historical behaviour).  final_full evaluates the cadence on fixed
        # deterministic samples and the full populations at the endpoint.
        if subset_mode == FINAL_FULL_MODE:
            self.dev_comp_idx = deterministic_sample(
                self.comp_idx, DEV_EVAL_SIZE, DEV_EVAL_SEED)
            self.dev_naming_idx = deterministic_sample(
                self.naming_idx, DEV_EVAL_SIZE, DEV_EVAL_SEED + 1)
            self.dev_rep_idx = deterministic_sample(
                range(len(self.entries)), DEV_EVAL_SIZE, DEV_EVAL_SEED + 2)
        else:
            self.dev_comp_idx = self.dev_naming_idx = self.dev_rep_idx = (
                self.subset_idx)

        # ---- dorsal pool ----
        self.pool_entries = build_pool_entries(
            self.vocab, self.cfg.train.dorsal_pool_size,
            self.cfg.data.semantic_dim, seed=self.seed)
        self.pool_bank = torch.zeros(len(self.pool_entries),
                                     self.cfg.data.semantic_dim)

        # ---- streams ----
        self.stream_seeds = derive_stream_seeds(self.seed)
        rep_pop = list(range(len(self.entries)))
        w = logfreq_weights([self.entries[i].rank for i in rep_pop]) ** float(
            self.cfg.data.freq_temp)
        w = np.clip(w, 1e-6, None)
        w = w / w.sum()
        self.streams: Dict[str, CounterStream] = {
            "repetition": CounterStream("repetition", rep_pop, batch_size,
                                        self.stream_seeds["repetition"], weights=w),
            "pool": CounterStream("pool", list(range(len(self.pool_entries))),
                                  batch_size, self.stream_seeds["pool"]),
            "comprehension": CounterStream("comprehension", self.comp_idx,
                                           batch_size,
                                           self.stream_seeds["comprehension"]),
            "naming": CounterStream("naming", self.naming_idx, batch_size,
                                    self.stream_seeds["naming"]),
        }
        self.cursors: Dict[str, int] = {k: 0 for k in STREAM_NAMES}
        self.global_step = 0
        self.resume_provenance: List[dict] = []

    # -------------------------------------------------------------- batches
    def batch(self, stream: str) -> dict:
        idx = self.streams[stream].indices(self.cursors[stream])
        if stream == "pool":
            return build_batch(self.pool_entries, self.pool_bank, self.vocab,
                               idx, self.device)
        return build_batch(self.entries, self.bank_raw, self.vocab, idx,
                           self.device)

    def peek_indices(self, stream: str, k: int = 1) -> List[List[int]]:
        """The next `k` index lists of a stream without advancing it."""
        c = self.cursors[stream]
        return [self.streams[stream].indices(c + j) for j in range(k)]

    @property
    def rep_epoch(self) -> int:
        return self.cursors["repetition"] // self.streams["repetition"].per_epoch

    @property
    def rep_cursor(self) -> int:
        """Completed repetition batches — the LR clock (see `lr_for_step`)."""
        return self.cursors["repetition"]

    def current_lr(self, task: Optional[str] = None) -> float:
        """The learning rate for the step about to be taken.

        Under the historical two-stage policy this is a pure function of the
        repetition cursor and `task` is ignored, so every pre-FINAL-4 run is
        unaffected.  Under the task-specific policy the caller must say which
        task the step trains.
        """
        if self.lr_policy["kind"] == LR_POLICY_TWO_STAGE:
            return lr_for_step(self.rep_cursor, self.lr_boundary_steps)
        if task not in LR_TASKS:
            raise RuntimeError(
                f"task-specific LR policy needs the step's task, got {task!r}")
        return float(self.lr_policy[task])

    def lr_phase_label(self, task: Optional[str] = None) -> str:
        if self.lr_policy["kind"] == LR_POLICY_TWO_STAGE:
            return lr_phase(self.rep_cursor, self.lr_boundary_steps)
        return f"task_specific_{task}"

    # ------------------------------------------------------------- schedule
    def task_for_step(self, global_step: int) -> str:
        """Which task the `global_step`-th optimizer update trains.

        Pure function of (schedule seed, step): cycle = step // 6, position =
        step % 6, and the cycle's six labels are recomputed by `macro_cycle`.
        Nothing about the task order is stored, so resume is exact by
        construction — including mid-cycle.
        """
        if self.schedule == SUMMED_SCHEDULE:
            return SUMMED_SCHEDULE
        cycle, pos = divmod(int(global_step), MACRO_CYCLE_STEPS)
        return macro_cycle(self.ratio, self.schedule_seed, cycle)[pos]

    def steps_for_rep_epochs(self, n_epochs: int) -> int:
        """Optimizer steps that deliver `n_epochs` full repetition passes.

        Summed: every update draws an R batch, so it is 463 steps per epoch.
        Interleaved 1:2:3: only one update in six is an R step, so an R epoch
        costs 463 * 6 = 2,778 optimizer steps.
        """
        per_epoch = self.streams["repetition"].per_epoch
        if self.schedule == SUMMED_SCHEDULE:
            return int(n_epochs) * per_epoch
        return int(n_epochs) * per_epoch * MACRO_CYCLE_STEPS

    def exposures(self) -> Dict[str, float]:
        """Exact exposures/item per task, from the cursors and populations."""
        return {name: self.cursors[name] / self.streams[name].per_epoch
                for name in STREAM_NAMES}

    def exposure_accounting(self) -> dict:
        """The schedule's exposure arithmetic, recorded in the provenance."""
        per = {n: self.streams[n].per_epoch for n in STREAM_NAMES}
        acc = {
            "batches_per_pass": per,
            "populations": {n: self.streams[n].n for n in STREAM_NAMES},
            "batch_size": self.cfg.train.batch_size,
        }
        if self.schedule == SUMMED_SCHEDULE:
            acc.update({
                "optimizer_steps_per_cycle": 1,
                "batches_per_cycle": {n: 1 for n in STREAM_NAMES
                                      if self.stream_active[n]},
                "exposures_per_step": {n: 1.0 / per[n] for n in STREAM_NAMES
                                       if self.stream_active[n]},
            })
        else:
            r, n, c = self.ratio
            acc.update({
                "ratio_R_N_C": list(self.ratio),
                "optimizer_steps_per_cycle": MACRO_CYCLE_STEPS,
                "batches_per_cycle": {"repetition": r, "pool": r,
                                      "naming": n, "comprehension": c},
                "exposures_per_cycle": {
                    "repetition": r / per["repetition"],
                    "naming": n / per["naming"],
                    "comprehension": c / per["comprehension"]},
                "cycles_per_repetition_exposure": per["repetition"] / r,
                "steps_per_repetition_exposure":
                    per["repetition"] * MACRO_CYCLE_STEPS / r,
            })
        return acc

    # ----------------------------------------------------------- train step
    def train_step(self) -> dict:
        """One optimizer update, dispatched by the configured schedule."""
        if self.schedule != SUMMED_SCHEDULE:
            return self._interleaved_step()
        return self._summed_step()

    # ------------------------------------------------------ interleaved step
    def _interleaved_step(self) -> dict:
        """One task's optimizer update: zero_grad -> backward -> clip -> step.

        The task is a pure function of the step index (`task_for_step`).  There
        is no cross-task summation: each task owns its own gradient, its own
        global clip at `grad_clip`, and its own AdamW update on the SHARED
        optimizer state.  Only the streams the step actually consumes advance,
        so the cursors are the exposure ledger.

        Objectives are exactly the FINAL-1 ones, unbundled:
            R : total_loss(R) + cfg.loss.wm * pool_CE(pool)   [one backward]
            N : LAMBDA_N * naming_CE
            C : LAMBDA_C * retrieval_CE          (c_align_weight applies here
                                                  too, and is 0.0 in FINAL-3P)
        """
        cfg = self.cfg
        pad_id = self.model.vocab.pad_id
        self.model.train(True)

        # The task is resolved first: under the task-specific policy the LR of
        # this step is a property of the task it trains.
        task = self.task_for_step(self.global_step)
        lr = self.current_lr(task)
        for g in self.optim.param_groups:
            g["lr"] = lr

        rec: Dict[str, object] = {"task": task}
        touched: List[str] = []

        if task == "repetition":
            # The dorsal pool rides every R step inside the same backward,
            # exactly as in the historical recipe, which keeps the pool:R
            # batch ratio at 1:1 and the pool exposure rate unchanged.
            r = self.batch("repetition")
            out = self.model(r["enc_in"], r["enc_mask"], r["dec_in"])
            parts = total_loss(out, r, cfg.loss, pad_id,
                               usage_prior=cfg.gating.usage_prior)
            p = self.batch("pool")
            pout = self.model(p["enc_in"], p["enc_mask"], p["dec_in"])
            V = pout["wm_logits"].shape[-1]
            pool_ce = F.cross_entropy(pout["wm_logits"].reshape(-1, V),
                                      p["dec_tgt"].reshape(-1),
                                      ignore_index=pad_id)
            loss = parts["total"] + cfg.loss.wm * pool_ce
            rec.update({k: float(v.detach()) for k, v in parts.items()})
            rec["pool_ce"] = float(pool_ce.detach())
            touched = ["repetition", "pool"]

        elif task == "naming":
            n = self.batch("naming")
            nam = naming_objective(self.model, n, pad_id)["total"]
            loss = LAMBDA_N * nam
            rec["naming_ce"] = float(nam.detach())
            touched = ["naming"]

        elif task == "comprehension":
            c = self.batch("comprehension")
            s_hat = comprehension_forward(self.model, c["enc_in"], c["enc_mask"])
            ret = retrieval_loss(s_hat, self.model.ltm.semantic_bank,
                                 c["bank_idx"], TAU)
            loss = LAMBDA_C * ret
            rec["retrieval_ce"] = float(ret.detach())
            rec["retrieval_weighted"] = LAMBDA_C * rec["retrieval_ce"]
            if self.c_align_weight > 0.0:
                c_align = alignment_loss(s_hat, c["semantic"])
                loss = loss + self.c_align_weight * c_align
                rec["c_align"] = float(c_align.detach())
                rec["c_align_weighted"] = self.c_align_weight * rec["c_align"]
            touched = ["comprehension"]

        else:                                                # pragma: no cover
            raise RuntimeError(f"unknown scheduled task {task!r}")

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               cfg.train.grad_clip)
        self.optim.step()

        for name in touched:
            self.cursors[name] += 1
        self.global_step += 1

        rec.update({"step": self.global_step, "lr": lr,
                    "lr_phase": self.lr_phase_label(task),
                    "joint_total": float(loss.detach()),
                    "grad_norm": float(gnorm)})
        return rec

    # ----------------------------------------------------------- summed step
    def _summed_step(self) -> dict:
        """One joint optimizer update: one summed loss, one backward, one clip,
        one `optimizer.step()`.  The frozen FINAL-1 update."""
        cfg = self.cfg
        pad_id = self.model.vocab.pad_id
        self.model.train(True)

        lr = self.current_lr()
        for g in self.optim.param_groups:
            g["lr"] = lr

        # --- repetition (historical objective, verbatim total_loss) ---
        r = self.batch("repetition")
        out = self.model(r["enc_in"], r["enc_mask"], r["dec_in"])
        parts = total_loss(out, r, cfg.loss, pad_id,
                           usage_prior=cfg.gating.usage_prior)
        loss = parts["total"]
        rec = {k: float(v.detach()) for k, v in parts.items()}

        # --- dorsal pseudoword pool (historical auxiliary CE, weight cfg.loss.wm) ---
        p = self.batch("pool")
        pout = self.model(p["enc_in"], p["enc_mask"], p["dec_in"])
        V = pout["wm_logits"].shape[-1]
        pool_ce = F.cross_entropy(pout["wm_logits"].reshape(-1, V),
                                  p["dec_tgt"].reshape(-1), ignore_index=pad_id)
        loss = loss + cfg.loss.wm * pool_ce
        rec["pool_ce"] = float(pool_ce.detach())

        # --- factorial objectives: retrieval and/or naming, per regime ---
        # The two terms are added independently and in this fixed order, so
        # j0 (both on) is numerically identical to the pre-factorial code and
        # h0 (both off) adds nothing at all.
        rec["retrieval_ce"] = float("nan")
        rec["retrieval_weighted"] = float("nan")
        rec["c_align"] = float("nan")
        rec["c_align_weighted"] = float("nan")
        rec["naming_ce"] = float("nan")
        if self.retrieval_enabled:
            c = self.batch("comprehension")
            s_hat = comprehension_forward(self.model, c["enc_in"], c["enc_mask"])
            ret = retrieval_loss(s_hat, self.model.ltm.semantic_bank,
                                 c["bank_idx"], TAU)
            loss = loss + LAMBDA_C * ret
            rec["retrieval_ce"] = float(ret.detach())
            rec["retrieval_weighted"] = LAMBDA_C * rec["retrieval_ce"]
            # FINAL-2A intervention (off by default, c_align_weight = 0.0).
            # The C stream gets its own semantic-target alignment, reusing the
            # s_hat already computed for retrieval and the SAME canonical
            # `losses.alignment_loss` the R stream uses.  It is weighted by
            # c_align_weight ALONE -- never by LAMBDA_C -- so the C
            # contribution is  LAMBDA_C * retrieval_CE + c_align_weight *
            # alignment.  At weight 0.0 nothing is computed and nothing enters
            # the graph, so FINAL-1 semantics are preserved exactly.
            if self.c_align_weight > 0.0:
                c_align = alignment_loss(s_hat, c["semantic"])
                loss = loss + self.c_align_weight * c_align
                rec["c_align"] = float(c_align.detach())
                rec["c_align_weighted"] = self.c_align_weight * rec["c_align"]

        if self.naming_enabled:
            n = self.batch("naming")
            nam = naming_objective(self.model, n, pad_id)["total"]
            loss = loss + LAMBDA_N * nam
            rec["naming_ce"] = float(nam.detach())

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               cfg.train.grad_clip)
        self.optim.step()

        # R and pool advance in every cell, which is what makes the four
        # conditions share one repetition trajectory. A semantic cursor moves
        # only where its objective is actually trained, so an inactive stream
        # stays at 0 and cannot silently consume batches.
        for name in STREAM_NAMES:
            if self.stream_active[name]:
                self.cursors[name] += 1
        self.global_step += 1

        rec.update({"step": self.global_step, "lr": lr, "task": SUMMED_SCHEDULE,
                    "lr_phase": self.lr_phase_label(),
                    "joint_total": float(loss.detach()),
                    "grad_norm": float(gnorm)})
        return rec

    # ------------------------------------------------------------ evaluation
    def evaluate(self, *, with_probe: bool = False,
                 with_full_lexicon: bool = False) -> dict:
        """Developmental evaluation. Wrapped in `preserved_rng` so evaluation
        cadence can never become a hidden experimental factor."""
        with preserved_rng(), torch.no_grad():
            rep = repetition_snapshot(self.model, self.vocab, self.entries,
                                      self.dev_rep_idx, self.bank_raw,
                                      self.device, include_teacher_forced=False)
            comp = evaluate_comprehension_subset(
                self.model, self.vocab, self.entries, self.bank_raw,
                self.dev_comp_idx, self.device)
            nam = evaluate_naming(self.model, self.vocab, self.entries,
                                  self.bank_raw, self.dev_naming_idx, self.device,
                                  NAMING_MAX_STEPS, return_per_item=True)
            per_item = nam.pop("_per_item", [])
            exp = self.exposures()
            out = {
                "step": self.global_step,
                "rep_epoch": self.rep_epoch,
                "r_exposures": exp["repetition"],
                "n_exposures": exp["naming"],
                "c_exposures": exp["comprehension"],
                "lr": self.current_lr("repetition"),
                "rep_full": rep["primary_readout"]["exact_match"]["full"],
                "rep_wm": rep["primary_readout"]["exact_match"]["wm"],
                "rep_ltm": rep["primary_readout"]["exact_match"]["ltm"],
                "comp_top1": comp["top1"], "comp_top5": comp["top5"],
                "comp_rank_median": comp["target_rank_median"],
                "comp_rank_mean": comp["target_rank_mean"],
                "comp_cos_mean": comp["target_cos_mean"],
                "comp_margin_mean": comp["margin_mean"],
                "naming_exact": nam["exact_match"],
                "naming_wer": nam["whole_word_error_rate"],
                "naming_mean_edit": nam["mean_edit"],
                "naming_eos_rate": nam["eos_emission_rate"],
                "naming_pred_len_mean": (
                    float(np.mean([r["pred_len"] for r in per_item]))
                    if per_item else float("nan")),
                "naming_target_len_mean": (
                    float(np.mean([r["length"] for r in per_item]))
                    if per_item else float("nan")),
                "probe_rep_ltm": float("nan"),
                "probe_rep_full": float("nan"),
                "full_rep_ltm": float("nan"),
                "full_rep_full": float("nan"),
                "full_rep_wm": float("nan"),
                "full_comp_top1": float("nan"),
                "full_comp_top5": float("nan"),
                "full_naming_exact": float("nan"),
                "full_naming_wer": float("nan"),
            }
            # The probe is skipped (NaN, never fabricated) when no
            # out-of-subset complement exists (final_full mode).
            if with_probe and self.probe_idx:
                pr = repetition_snapshot(self.model, self.vocab, self.entries,
                                         self.probe_idx, self.bank_raw,
                                         self.device,
                                         include_teacher_forced=False)
                out["probe_rep_ltm"] = pr["primary_readout"]["exact_match"]["ltm"]
                out["probe_rep_full"] = pr["primary_readout"]["exact_match"]["full"]
            if with_full_lexicon:
                fr = repetition_snapshot(self.model, self.vocab, self.entries,
                                         list(range(len(self.entries))),
                                         self.bank_raw, self.device,
                                         include_teacher_forced=False)
                out["full_rep_ltm"] = fr["primary_readout"]["exact_match"]["ltm"]
                out["full_rep_full"] = fr["primary_readout"]["exact_match"]["full"]
                out["full_rep_wm"] = fr["primary_readout"]["exact_match"]["wm"]
                # Full C/N population evaluation where the cadence used dev
                # samples (final_full); in legacy modes dev == full subset, so
                # re-evaluating would only duplicate the regular columns.
                if self.dev_comp_idx is not self.comp_idx:
                    fc = evaluate_comprehension_subset(
                        self.model, self.vocab, self.entries, self.bank_raw,
                        self.comp_idx, self.device)
                    out["full_comp_top1"] = fc["top1"]
                    out["full_comp_top5"] = fc["top5"]
                if self.dev_naming_idx is not self.naming_idx:
                    fn = evaluate_naming(self.model, self.vocab, self.entries,
                                         self.bank_raw, self.naming_idx,
                                         self.device, NAMING_MAX_STEPS)
                    out["full_naming_exact"] = fn["exact_match"]
                    out["full_naming_wer"] = fn["whole_word_error_rate"]
        return out

    # ---------------------------------------------------------- checkpoints
    def resolved_settings(self) -> dict:
        """Every scientific value actually in force, for printing and saving."""
        cfg = self.cfg
        return {
            "regime": self.regime,
            # Objective presence is recorded as explicit booleans rather than
            # inferred from a zero weight or a missing field, so every saved
            # config states which factorial cell produced it.
            "retrieval_enabled": self.retrieval_enabled,
            "naming_enabled": self.naming_enabled,
            "active_training_streams": sorted(
                k for k, v in self.stream_active.items() if v),
            "seed": self.seed,
            "device": self.device,
            "repetition_population": len(self.entries),
            "repetition_sampler": "log-frequency weighted, with replacement, "
                                  f"freq_temp={cfg.data.freq_temp}",
            "comprehension_population": len(self.comp_idx),
            "comprehension_population_name": self.comp_population_name,
            "comprehension_population_sha256": self.comp_hash,
            "naming_population": len(self.naming_idx),
            "naming_population_name": self.naming_population_name,
            "naming_population_sha256": self.naming_hash,
            "homophone_policy": (HOMOPHONE_POLICY_NOTE
                                 if self.subset_mode == FINAL_FULL_MODE
                                 else "C/N population is homophone-free by "
                                      "construction (legacy subset mode)"),
            "schedule": self.schedule,
            "schedule_ratio_R_N_C": (list(self.ratio) if self.ratio else None),
            "schedule_seed": self.schedule_seed,
            "task_order_policy": (TASK_ORDER_POLICY
                                  if self.schedule != SUMMED_SCHEDULE
                                  else "single summed update per step"),
            "macro_cycle_steps": (MACRO_CYCLE_STEPS
                                  if self.schedule != SUMMED_SCHEDULE else 1),
            "exposure_accounting": self.exposure_accounting(),
            "lr_policy": dict(self.lr_policy),
            "phase_transitions": list(self.phase_transitions),
            # Under a task-specific policy there is no single scalar LR; the
            # per-step value is exact in losses.tsv, and the metrics.tsv "lr"
            # column reports the REPETITION rate as the anchor.
            "lr_column_convention": (
                "metrics.tsv lr = repetition LR (anchor); losses.tsv lr is the "
                "exact per-step rate"
                if self.lr_policy["kind"] == LR_POLICY_TASK
                else "single scalar LR from the two-stage schedule"),
            "lr_convention": (
                "LR is a pure function of the REPETITION cursor: "
                f"{LR_STAGE1} while completed R batches < {self.lr_boundary_steps} "
                f"({self.lr_boundary_steps // self.streams['repetition'].per_epoch} "
                f"R exposures), then {LR_STAGE2}. Identical to the historical "
                "global-step rule under the summed schedule."),
            "subset_mode": self.subset_mode,
            "subset_definition_sha256": self.subset_hash,
            "out_of_subset_probe_n": len(self.probe_idx),
            "out_of_subset_probe": self.probe_note,
            "dev_eval_populations": {
                "repetition": len(self.dev_rep_idx),
                "comprehension": len(self.dev_comp_idx),
                "naming": len(self.dev_naming_idx),
                "note": ("fixed deterministic samples (final_full); full "
                         "populations evaluated with --endpoint-eval"
                         if self.subset_mode == FINAL_FULL_MODE
                         else "identical to the C/N subset (legacy modes)"),
            },
            "batch_size": cfg.train.batch_size,
            "batches_per_rep_epoch": self.streams["repetition"].per_epoch,
            "dorsal_pool_size": len(self.pool_entries),
            "hidden_size": cfg.wm.hidden,
            "ltm_enc_hidden": cfg.ltm.enc_hidden,
            "ltm_dec_hidden": cfg.ltm.dec_hidden,
            "ltm_encoder_mode": cfg.ltm.ltm_encoder_mode,
            "teacher_forcing_ratio": cfg.train.teacher_forcing_ratio,
            "interference_noise": cfg.wm.interference_noise,
            "ventral_noise": cfg.ltm.ventral_noise,
            "gate_alpha": cfg.gating.alpha,
            "gate_threshold": cfg.gating.gate_threshold,
            "usage_prior": cfg.gating.usage_prior,
            "loss_weights": dict(CANONICAL_LOSS_WEIGHTS),
            "dorsal_pool_loss_weight": cfg.loss.wm,
            # Effective weights (0.0 where the objective is switched off) plus
            # the canonical constants, which are never rewritten by a regime.
            "lambda_C": LAMBDA_C if self.retrieval_enabled else 0.0,
            "lambda_N": LAMBDA_N if self.naming_enabled else 0.0,
            "tau": TAU if self.retrieval_enabled else None,
            "lambda_C_canonical": LAMBDA_C,
            "lambda_N_canonical": LAMBDA_N,
            "tau_canonical": TAU,
            # FINAL-2A: weight on the C stream's own semantic-target alignment
            # (losses.alignment_loss on the retrieval s_hat).  0.0 = FINAL-1.
            "c_align_weight": self.c_align_weight,
            "c_stream_objective": (
                f"{LAMBDA_C} * retrieval_CE + {self.c_align_weight} * "
                f"alignment_loss(s_hat_C, GloVe_C)"
                if self.c_align_weight > 0.0
                else f"{LAMBDA_C} * retrieval_CE (FINAL-1 objective)"),
            "retrieval_bank_size": int(self.bank_raw.shape[0]),
            "optimizer": "AdamW",
            "weight_decay": cfg.train.weight_decay,
            "grad_clip": cfg.train.grad_clip,
            "lr_stage1": LR_STAGE1,
            "lr_stage2": LR_STAGE2,
            "lr_boundary_steps": self.lr_boundary_steps,
            "current_lr": self.current_lr("repetition"),
            "stream_seeds": dict(self.stream_seeds),
            "glove_found": self.glove_found,
            "glove_fallback": self.glove_fallback,
        }

    def state_dict(self) -> dict:
        return {
            "format": "lichtheim3.joint_scratch.v1",
            "regime": self.regime,
            "seed": self.seed,
            "config": {"data": vars(self.cfg.data), "wm": vars(self.cfg.wm),
                       "ltm": vars(self.cfg.ltm), "gating": vars(self.cfg.gating),
                       "loss": vars(self.cfg.loss), "train": vars(self.cfg.train)},
            "resolved_settings": self.resolved_settings(),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "global_step": self.global_step,
            "rep_epoch": self.rep_epoch,
            "batch_in_rep_epoch": (self.cursors["repetition"]
                                   % self.streams["repetition"].per_epoch),
            "cursors": dict(self.cursors),
            "stream_seeds": dict(self.stream_seeds),
            "lr": self.current_lr("repetition"),
            "lr_phase": self.lr_phase_label("repetition"),
            "lr_boundary_steps": self.lr_boundary_steps,
            "subset_definition_sha256": self.subset_hash,
            "subset_mode": self.subset_mode,
            "c_align_weight": self.c_align_weight,
            "lr_policy": dict(self.lr_policy),
            "phase_transitions": list(self.phase_transitions),
            "schedule": self.schedule,
            "schedule_seed": self.schedule_seed,
            "schedule_ratio": (list(self.ratio) if self.ratio else None),
            "exposures": self.exposures(),
            "subset_indices": list(self.subset_idx),
            # Explicit per-task population provenance (FINAL-1A).  In
            # final_full mode the legacy subset_* fields alias the
            # comprehension population; these fields are authoritative.
            "comprehension_population_name": self.comp_population_name,
            "comprehension_population_sha256": self.comp_hash,
            "comprehension_population_n": len(self.comp_idx),
            "naming_population_name": self.naming_population_name,
            "naming_population_sha256": self.naming_hash,
            "naming_population_n": len(self.naming_idx),
            "probe_indices": list(self.probe_idx),
            "probe_note": self.probe_note,
            "lexicon_path": self.cfg.data.lexicon_path,
            "lexicon_file_sha256": sha256_file(
                os.path.join(ROOT, self.cfg.data.lexicon_path or "")),
            "glove_found": self.glove_found,
            "glove_fallback": self.glove_fallback,
            "rng_states": capture_rng_states(),
            "git": git_state(ROOT),
            "resume_provenance": list(self.resume_provenance),
            "historical_fidelity": HISTORICAL_FIDELITY_NOTE,
        }

    def load_state_dict(self, ckpt: dict, *, source: str = "") -> None:
        if ckpt.get("format") != "lichtheim3.joint_scratch.v1":
            raise RuntimeError(f"unexpected checkpoint format {ckpt.get('format')!r}")
        if ckpt["regime"] != self.regime:
            raise RuntimeError(f"checkpoint regime {ckpt['regime']!r} != {self.regime!r}")
        if int(ckpt["seed"]) != self.seed:
            raise RuntimeError(f"checkpoint seed {ckpt['seed']} != {self.seed}")
        if ckpt["subset_definition_sha256"] != self.subset_hash:
            raise RuntimeError("checkpoint subset hash differs from the rebuilt subset")
        if ckpt.get("subset_mode", self.subset_mode) != self.subset_mode:
            raise RuntimeError(
                f"checkpoint subset_mode {ckpt.get('subset_mode')!r} != "
                f"{self.subset_mode!r}")
        # Checkpoints predating FINAL-2A carry no weight; they were produced
        # with the FINAL-1 objective, i.e. 0.0.  Resuming across a change of
        # the C objective would silently splice two different experiments.
        ck_calign = float(ckpt.get("c_align_weight", 0.0))
        if ck_calign != self.c_align_weight:
            raise RuntimeError(
                f"checkpoint c_align_weight {ck_calign} != "
                f"{self.c_align_weight}: this checkpoint was trained under a "
                f"different comprehension objective")
        # Checkpoints predating FINAL-3 carry no schedule; they are summed.
        ck_sched = ckpt.get("schedule", SUMMED_SCHEDULE)
        if ck_sched != self.schedule:
            raise RuntimeError(
                f"checkpoint schedule {ck_sched!r} != {self.schedule!r}: "
                f"summed and interleaved trajectories must never be spliced")
        ck_sseed = ckpt.get("schedule_seed")
        if ck_sseed is not None and int(ck_sseed) != int(self.schedule_seed):
            raise RuntimeError(
                f"checkpoint schedule_seed {ck_sseed} != {self.schedule_seed}: "
                f"the task order would differ after resume")
        ck_ratio = ckpt.get("schedule_ratio")
        if ck_ratio is not None and list(ck_ratio) != list(self.ratio or []):
            raise RuntimeError(
                f"checkpoint schedule_ratio {ck_ratio} != {self.ratio}")
        ck_naming_hash = ckpt.get("naming_population_sha256")
        if ck_naming_hash is not None and ck_naming_hash != self.naming_hash:
            raise RuntimeError(
                "checkpoint naming population hash differs from the rebuilt one")
        if dict(ckpt["stream_seeds"]) != dict(self.stream_seeds):
            raise RuntimeError("checkpoint stream seeds differ from the derived ones")

        # ---- learning-rate policy: a scientific field with ONE sanctioned
        # way to change it, an explicitly declared phase transition.  Every
        # other guard above is unconditional and is NOT relaxed by that flag.
        ck_policy = ckpt.get("lr_policy") or two_stage_lr_policy(
            int(ckpt["lr_boundary_steps"]))
        self.phase_transitions = list(ckpt.get("phase_transitions", []))
        if dict(ck_policy) != dict(self.lr_policy):
            if not self.allow_phase_transition:
                raise RuntimeError(
                    f"checkpoint LR policy {ck_policy} != requested "
                    f"{self.lr_policy}. Changing the learning-rate policy is a "
                    f"PHASE TRANSITION, not a continuation: it makes this a "
                    f"staged recipe rather than the same one. Pass "
                    f"--phase-transition to declare it deliberately.")
            self.phase_transitions.append({
                "transition_step": int(ckpt["global_step"]),
                "old_lr_policy": dict(ck_policy),
                "new_lr_policy": dict(self.lr_policy),
                "source_checkpoint": source,
                "source_commit": (ckpt.get("git") or {}).get("commit"),
                "new_commit": (git_state(ROOT) or {}).get("commit"),
                "exposures_at_transition": {
                    k: int(v) / self.streams[k].per_epoch
                    for k, v in ckpt["cursors"].items()},
                "declared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            print(f"[phase] DECLARED TRANSITION at step {ckpt['global_step']}: "
                  f"{ck_policy} -> {self.lr_policy}")

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optim.load_state_dict(ckpt["optimizer_state_dict"])
        for st in self.optim.state.values():
            for k, v in st.items():
                if isinstance(v, torch.Tensor):
                    st[k] = v.to(self.device)
        self.global_step = int(ckpt["global_step"])
        self.cursors = {k: int(v) for k, v in ckpt["cursors"].items()}
        self.lr_boundary_steps = int(ckpt["lr_boundary_steps"])
        # LR is a pure function of the step counter, so it is re-derived rather
        # than trusted from the file; the optimizer is never reconstructed.
        # Re-derived, never trusted from the file.  Under a task-specific
        # policy the per-step rate is set by _interleaved_step anyway; this
        # just leaves the optimizer in a coherent state after loading.
        lr = self.current_lr("repetition")
        for g in self.optim.param_groups:
            g["lr"] = lr

        # Every RNG goes through the explicit checkpoint contract, which
        # repairs states relocated by `map_location` on GPU nodes.
        restore_rng_states(ckpt.get("rng_states") or {})

        self.resume_provenance = list(ckpt.get("resume_provenance", []))
        self.resume_provenance.append({
            "resumed_from": source,
            "at_global_step": self.global_step,
            "at_rep_epoch": self.rep_epoch,
            "batch_in_rep_epoch": (self.cursors["repetition"]
                                   % self.streams["repetition"].per_epoch),
            "wall_clock": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })


HISTORICAL_FIDELITY_NOTE = (
    "Paired developmental control using the VERIFIED HISTORICAL SCIENTIFIC "
    "RECIPE. This is deliberately NOT a bit-exact replay of the original "
    "two-job SLURM execution: the historical e100->e101 resume did not "
    "checkpoint the dorsal pool's itertools.cycle cursor, so the pool restarted "
    "and perturbed the global RNG stream from e102 onward (Phase 4A0c). Here "
    "the learning-rate boundary is applied in-process with the AdamW moments "
    "and step counter untouched, and every sampling stream is counter-addressed "
    "and exactly resumable."
)


# ===========================================================================
#  Run directory / metrics
# ===========================================================================

METRIC_COLUMNS = [
    "step", "rep_epoch", "r_exposures", "n_exposures", "c_exposures",
    "lr", "rep_full", "rep_wm", "rep_ltm",
    "comp_top1", "comp_top5", "comp_rank_median", "comp_rank_mean",
    "comp_cos_mean", "comp_margin_mean",
    "naming_exact", "naming_wer", "naming_mean_edit", "naming_eos_rate",
    "naming_pred_len_mean", "naming_target_len_mean",
    "probe_rep_ltm", "probe_rep_full",
    "full_rep_ltm", "full_rep_full", "full_rep_wm",
    "full_comp_top1", "full_comp_top5", "full_naming_exact", "full_naming_wer",
]


def append_metrics(path: str, row: dict) -> None:
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        if new:
            fh.write("\t".join(METRIC_COLUMNS) + "\n")
        fh.write("\t".join(str(row.get(c, "")) for c in METRIC_COLUMNS) + "\n")


def append_losses(path: str, row: dict) -> None:
    # Convention: a blank cell means the quantity was NOT computed on this
    # optimizer step (an interleaved step trains exactly one task), never that
    # it was computed and happened to be zero.
    cols = ["step", "task", "lr", "lr_phase", "joint_total", "total", "rep",
            "align", "dec", "wm", "gate", "pool_ce", "retrieval_ce",
            "retrieval_weighted", "c_align", "c_align_weighted", "naming_ce",
            "grad_norm"]
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        if new:
            fh.write("\t".join(cols) + "\n")
        fh.write("\t".join(str(row.get(c, "")) for c in cols) + "\n")


def ckpt_path(run_dir: str, step: int) -> str:
    return os.path.join(run_dir, "checkpoints", f"step_{step:08d}.pt")


def portable_path(path: str) -> str:
    """Repo-relative when the file lives inside the tree, basename otherwise.

    Keeps machine-specific absolute paths out of saved run metadata without
    inventing a fake location for files (smoke checkpoints, scratch dirs) that
    genuinely sit outside the repository.
    """
    ap = os.path.abspath(path)
    root = os.path.abspath(ROOT) + os.sep
    return os.path.relpath(ap, ROOT) if ap.startswith(root) else os.path.basename(ap)


# ===========================================================================
#  CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 4 joint multitask development from scratch (H0/J0).")
    p.add_argument("--regime", required=True, choices=REGIMES,
                   help="factorial cell: h0 (neither), c_only (retrieval), "
                        "n_only (naming), j0 (both)")
    p.add_argument("--seed", type=int, default=22,
                   help="experimental seed S; task stream seeds are derived from it")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                   help="repetition epochs (1 epoch = ceil(N/batch) optimizer steps)")
    p.add_argument("--max-steps", type=int, default=None,
                   help="override the step budget directly (smoke tests)")
    p.add_argument("--device", default="cpu")
    # outputs/ is already gitignored and is where Phases 2 and 3 wrote; runs/ is
    # not ignored, so defaulting there would dirty the tree on every run.
    p.add_argument("--out-dir", default=os.path.join(ROOT, "outputs", "joint_scratch"))
    p.add_argument("--run-id", default=None)
    p.add_argument("--resume", default=None, help="checkpoint to resume from")

    p.add_argument("--eval-every", type=int, default=4630,
                   help="developmental evaluation cadence in optimizer steps")
    p.add_argument("--probe-every", type=int, default=23150,
                   help="out-of-subset repetition probe cadence in steps")
    p.add_argument("--save-every", type=int, default=4630)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--endpoint-eval", action="store_true",
                   help="run the full 29,571-word repetition evaluation at the end")
    p.add_argument("--full-eval-at", default="",
                   help="comma-separated GLOBAL STEP numbers at which to run a "
                        "full-population milestone evaluation (probe + full "
                        "repetition, and full C/N in final_full mode) and save "
                        "a checkpoint, inside one continuous run.  E.g. for "
                        "FINAL-1: 138900,231500,324100 (= 300/500/700 "
                        "N-exposures at 463 steps per pass).")
    p.add_argument("--eval-at-start", action="store_true")

    # deliberately few knobs; these exist for smoke tests only
    p.add_argument("--lexicon-path", default=CANONICAL_LEXICON_PATH)
    p.add_argument("--glove-path", default="data/glove.6B.300d.txt",
                   help="SMOKE ONLY: point elsewhere to skip the 1GB GloVe parse")
    p.add_argument("--max-words", type=int, default=CANONICAL_MAX_WORDS)
    p.add_argument("--batch-size", type=int, default=CANONICAL_BATCH_SIZE)
    p.add_argument("--dorsal-pool-size", type=int, default=CANONICAL_DORSAL_POOL_SIZE)
    p.add_argument("--subset-mode",
                   choices=("nested", "representative", FINAL_FULL_MODE),
                   default="nested",
                   help="C/N populations: 'nested' = frozen subset3288 "
                        "(Phases 2-4); 'representative' = uniform smoke subset; "
                        "'final_full' = FINAL populations (C = canonical "
                        "one-per-phonology 27,981; N = full 29,571)")
    p.add_argument("--subset-per-band", type=int, default=SUBSET_PER_BAND)
    p.add_argument("--subset-size", type=int, default=64,
                   help="only used with --subset-mode representative (smoke)")
    p.add_argument("--lr-boundary-steps", type=int, default=LR_BOUNDARY_STEPS)
    p.add_argument("--allow-glove-fallback", action="store_true",
                   help="SMOKE ONLY: permit pseudo-vectors instead of real GloVe")
    p.add_argument("--no-subset-hash-check", action="store_true",
                   help="SMOKE ONLY: skip the frozen subset3288 hash assertion")
    p.add_argument("--schedule", choices=SCHEDULES, default=SUMMED_SCHEDULE,
                   help="task schedule: 'summed' (default) = the frozen "
                        "FINAL-1 update, R+pool+C+N summed into one backward; "
                        "'interleaved_123' = one task per optimizer step in "
                        "deterministically shuffled six-step macro-cycles "
                        "holding exactly 1 R, 2 N and 3 C (Ueno-style). "
                        "--epochs always counts REPETITION epochs, so an "
                        "interleaved epoch costs 6x the optimizer steps.")
    p.add_argument("--lr-repetition", type=float, default=None,
                   help="FINAL-4: fixed learning rate for repetition steps. "
                        "Giving the three --lr-<task> flags selects the "
                        "task-specific LR policy, which REPLACES the two-stage "
                        "schedule and requires an interleaved schedule. "
                        "Omitting them keeps the historical schedule exactly.")
    p.add_argument("--lr-naming", type=float, default=None,
                   help="FINAL-4: fixed learning rate for naming steps.")
    p.add_argument("--lr-comprehension", type=float, default=None,
                   help="FINAL-4: fixed learning rate for comprehension steps.")
    p.add_argument("--phase-transition", action="store_true",
                   help="declare deliberately that this launch changes the "
                        "learning-rate policy of the checkpoint it resumes, "
                        "making the run a STAGED recipe. Without it such a "
                        "resume is refused. It relaxes no other guard.")
    p.add_argument("--c-align-weight", type=float, default=0.0,
                   help="FINAL-2A: weight on the comprehension stream's own "
                        "semantic-target alignment (the canonical "
                        "losses.alignment_loss on the retrieval s_hat), added "
                        "to LAMBDA_C * retrieval_CE and NOT scaled by "
                        "LAMBDA_C.  0.0 (default) = the frozen FINAL-1 "
                        "objective; FINAL-2A uses 1.0.")
    p.add_argument("--torch-deterministic", action="store_true",
                   help="opt-in strict determinism: "
                        "torch.use_deterministic_algorithms(True) + "
                        "cuDNN deterministic, benchmark off.  On CUDA also "
                        "export CUBLAS_WORKSPACE_CONFIG=:4096:8 before launch. "
                        "May reduce throughput; never enabled silently.")
    return p


def task_lrs_from_args(args) -> Optional[Dict[str, float]]:
    """The three --lr-<task> flags, all-or-nothing.

    Returning None selects the historical two-stage schedule, so a command
    line that does not mention them is unchanged in every respect.
    """
    given = {"repetition": args.lr_repetition, "naming": args.lr_naming,
             "comprehension": args.lr_comprehension}
    present = {k: v for k, v in given.items() if v is not None}
    if not present:
        return None
    if len(present) != len(given):
        raise SystemExit(
            "task-specific learning rates are all-or-nothing: got "
            f"{sorted(present)}, missing "
            f"{sorted(set(given) - set(present))}")
    return present


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.torch_deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("[determinism] torch.use_deterministic_algorithms(True), "
              "cudnn.deterministic=True, cudnn.benchmark=False")
        if "cuda" in args.device and os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            print("[determinism] WARNING: CUBLAS_WORKSPACE_CONFIG is not set; "
                  "CUDA matmuls may raise. Export CUBLAS_WORKSPACE_CONFIG=:4096:8.")

    require_hash = (not args.no_subset_hash_check
                    and ((args.subset_mode == "nested"
                          and args.subset_per_band == SUBSET_PER_BAND
                          and args.max_words == CANONICAL_MAX_WORDS)
                         or (args.subset_mode == FINAL_FULL_MODE
                             and args.max_words == CANONICAL_MAX_WORDS)))

    t0 = time.time()
    trainer = JointScratchTrainer(
        regime=args.regime, seed=args.seed, device=args.device,
        max_words=args.max_words, lexicon_path=args.lexicon_path,
        dorsal_pool_size=args.dorsal_pool_size, batch_size=args.batch_size,
        subset_mode=args.subset_mode, subset_per_band=args.subset_per_band,
        subset_size=args.subset_size,
        lr_boundary_steps=args.lr_boundary_steps,
        allow_glove_fallback=args.allow_glove_fallback,
        require_subset_hash=require_hash, glove_path=args.glove_path,
        c_align_weight=args.c_align_weight, schedule=args.schedule,
        task_lrs=task_lrs_from_args(args),
        allow_phase_transition=args.phase_transition)

    run_id = args.run_id or f"{args.regime}_seed{args.seed}"
    run_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

    if args.resume:
        ck = torch.load(args.resume, map_location=args.device, weights_only=False)
        trainer.load_state_dict(ck, source=portable_path(args.resume))
        print(f"[resume] from {args.resume}")
        print(f"[resume] step={trainer.global_step} rep_epoch={trainer.rep_epoch} "
              f"cursors={trainer.cursors}")

    per_epoch = trainer.streams["repetition"].per_epoch
    # --epochs always counts REPETITION epochs, so the budget is comparable
    # across schedules at matched R exposures rather than raw steps.
    total_steps = (args.max_steps if args.max_steps is not None
                   else trainer.steps_for_rep_epochs(args.epochs))

    settings = trainer.resolved_settings()
    settings["total_steps"] = total_steps
    settings["epochs"] = args.epochs
    settings["eval_every"] = args.eval_every
    settings["probe_every"] = args.probe_every
    settings["save_every"] = args.save_every
    settings["full_eval_at"] = sorted({int(s) for s in args.full_eval_at.split(",")
                                       if s.strip()})
    print(f"\n[joint_scratch] RESOLVED SCIENTIFIC CONFIGURATION ({run_id})")
    for k, v in settings.items():
        print(f"  {k:28s} : {v}")
    print(f"  {'historical_fidelity':28s} : {HISTORICAL_FIDELITY_NOTE}\n")

    # Run-control metadata is per LAUNCH.  A continuation extends the budget
    # and adds future milestones, so writing it back to config.json /
    # provenance.json would erase the original launch's record (its command,
    # start time and budget).  metrics.tsv and losses.tsv are append-only and
    # already accumulate across launches; these files now do the same, by
    # giving each continuation its own step-suffixed pair and leaving the
    # first launch's files untouched.
    suffix = f"_from_step_{trainer.global_step:08d}" if args.resume else ""
    with open(os.path.join(run_dir, f"config{suffix}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(settings, fh, indent=2, default=str)
    with open(os.path.join(run_dir, f"provenance{suffix}.json"), "w",
              encoding="utf-8") as fh:
        json.dump({
            "run_id": run_id, "regime": args.regime, "seed": args.seed,
            "retrieval_enabled": trainer.retrieval_enabled,
            "naming_enabled": trainer.naming_enabled,
            "command": " ".join(sys.argv),
            "git": git_state(ROOT),
            "lexicon_path": args.lexicon_path,
            "lexicon_file_sha256": sha256_file(
                os.path.join(ROOT, args.lexicon_path)),
            "subset_definition_sha256": trainer.subset_hash,
            "subset_mode": trainer.subset_mode,
            "comprehension_population": {
                "name": trainer.comp_population_name,
                "n": len(trainer.comp_idx),
                "sha256": trainer.comp_hash,
            },
            "naming_population": {
                "name": trainer.naming_population_name,
                "n": len(trainer.naming_idx),
                "sha256": trainer.naming_hash,
            },
            "homophone_policy": (HOMOPHONE_POLICY_NOTE
                                 if trainer.subset_mode == FINAL_FULL_MODE
                                 else None),
            "c_align_weight": trainer.c_align_weight,
            "c_stream_objective": trainer.resolved_settings()["c_stream_objective"],
            "lr_policy": dict(trainer.lr_policy),
            "phase_transition_declared": bool(args.phase_transition),
            "phase_transitions": list(trainer.phase_transitions),
            "schedule": trainer.schedule,
            "schedule_ratio_R_N_C": (list(trainer.ratio) if trainer.ratio else None),
            "schedule_seed": trainer.schedule_seed,
            "task_order_policy": trainer.resolved_settings()["task_order_policy"],
            "exposure_accounting": trainer.exposure_accounting(),
            "lr_convention": trainer.resolved_settings()["lr_convention"],
            "torch_deterministic": bool(args.torch_deterministic),
            "stream_seeds": trainer.stream_seeds,
            "historical_fidelity": HISTORICAL_FIDELITY_NOTE,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            # Continuation context: the scientific recipe is validated against
            # the checkpoint by load_state_dict, so a launch that resumes is
            # the SAME trajectory carried further, never a new experiment.
            "resumed_from": portable_path(args.resume) if args.resume else None,
            "resumed_at_step": trainer.global_step if args.resume else None,
            "budget_this_launch": {"total_steps": total_steps,
                                   "rep_epochs": args.epochs,
                                   "full_eval_at": settings["full_eval_at"]},
            "resume_provenance": list(trainer.resume_provenance),
        }, fh, indent=2, default=str)

    metrics = os.path.join(run_dir, "metrics.tsv")
    losses_tsv = os.path.join(run_dir, "logs", "losses.tsv")

    full_eval_steps = settings["full_eval_at"]
    if full_eval_steps:
        print(f"[joint_scratch] milestone full evaluations at steps "
              f"{full_eval_steps}")

    last_eval: tuple = (None, None)          # (step, probe_included)
    if args.eval_at_start:
        append_metrics(metrics, trainer.evaluate())
        last_eval = (trainer.global_step, False)

    while trainer.global_step < total_steps:
        rec = trainer.train_step()
        if args.log_every and trainer.global_step % args.log_every == 0:
            append_losses(losses_tsv, rec)
            # An interleaved step computes exactly one task's losses, so the
            # line reports whatever that step actually produced rather than
            # assuming every component exists.
            parts = " ".join(f"{k}={rec[k]:.4f}" for k in
                             ("rep", "align", "dec", "pool_ce",
                              "retrieval_ce", "naming_ce")
                             if k in rec and rec[k] == rec[k])
            task = rec.get("task", SUMMED_SCHEDULE)
            tag = "" if task == SUMMED_SCHEDULE else f"[{task[:4]}] "
            print(f"[step {trainer.global_step:>7d}/{total_steps}] {tag}"
                  f"lr={rec['lr']:.0e} joint={rec['joint_total']:.4f} "
                  f"{parts} gnorm={rec['grad_norm']:.3f} "
                  f"| {time.time() - t0:6.1f}s", flush=True)
        saved_this_step = False
        if trainer.global_step in full_eval_steps:
            # Milestone: full-population evaluation + checkpoint, inside the
            # continuous run (no human-mediated segmentation, no decision).
            row = trainer.evaluate(with_probe=True, with_full_lexicon=True)
            append_metrics(metrics, row)
            last_eval = (trainer.global_step, True)
            torch.save(trainer.state_dict(), ckpt_path(run_dir, trainer.global_step))
            saved_this_step = True
            print(f"  [MILESTONE @ {row['step']}] "
                  f"full_rep={row['full_rep_full']:.6f} "
                  f"full_naming={row['full_naming_exact']:.6f} "
                  f"full_comp_top1={row['full_comp_top1']:.6f}", flush=True)
        elif args.eval_every and trainer.global_step % args.eval_every == 0:
            probed = bool(args.probe_every
                          and trainer.global_step % args.probe_every == 0)
            row = trainer.evaluate(with_probe=probed)
            append_metrics(metrics, row)
            last_eval = (trainer.global_step, probed)
            print(f"  [eval @ {row['step']}] rep_ltm={row['rep_ltm']:.4f} "
                  f"comp_top1={row['comp_top1']:.4f} "
                  f"naming_exact={row['naming_exact']:.4f}", flush=True)
        if (args.save_every and not saved_this_step
                and trainer.global_step % args.save_every == 0):
            torch.save(trainer.state_dict(), ckpt_path(run_dir, trainer.global_step))

    # The endpoint row is skipped when the cadence already produced an
    # equivalent one at this exact step, so metrics.tsv never carries a
    # duplicated final evaluation.
    if last_eval != (trainer.global_step, True) or args.endpoint_eval:
        final = trainer.evaluate(with_probe=True,
                                 with_full_lexicon=args.endpoint_eval)
        append_metrics(metrics, final)
    torch.save(trainer.state_dict(), ckpt_path(run_dir, trainer.global_step))
    print(f"\n[joint_scratch] done: {trainer.global_step} steps, "
          f"{time.time() - t0:.1f}s -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
