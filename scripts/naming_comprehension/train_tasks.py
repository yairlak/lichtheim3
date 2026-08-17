"""Phase 2A: minimal single-task training infrastructure (comprehension / naming).

This module provides ONLY the machinery: parameter scoping, task forwards,
objectives, deterministic flat batching and a scale-calibration preflight.
It performs no optimizer step unless explicitly asked to run a training loop,
and Phase 2A stops at preflight.

Design constraints honoured here
--------------------------------
* models/ is NOT modified and no model API is added: the task forwards are
  free functions wrapping the existing `ltm.encode`, `ltm.decode_from_s_hat`
  and `motor` machinery.
* losses.py is NOT modified: C0 imports the existing `alignment_loss`
  verbatim, and the additive retrieval term lives here.  The repetition
  training path and `total_loss` are therefore bit-compatible by construction.
* config.py is NOT modified: task-specific knobs are CLI arguments.

Scientific decisions frozen for this stage (trained-item acquisition, NOT
generalization; no new train/validation split):

COMPREHENSION
    training population : the unique-phonology words only (count derived
                          dynamically from the lexicon, never hard-coded)
    retrieval bank      : the FULL canonical bank (all lexicon words), so
                          homophones remain competitors but do not supply
                          primary training targets
    sampler             : FLAT over the training population
    trainable           : ltm.encoder + ltm.to_semantic
    frozen              : everything else, phon_embed explicitly included

NAMING
    training population : all lexicon words
    input               : raw (unnormalised) target GloVe
    teacher forcing     : 1.0
    loss                : the existing sequence CE convention
    trainable (N0)      : ltm.sem_to_h0 + ltm.decoder + ltm.dec_to_premotor
    frozen              : everything else, phon_embed and motor.proj included

Objectives
    C0 = (1 - cosine) + 0.1 * MSE                    [losses.alignment_loss]
    C3 = C0 + lambda_ret * full-bank cosine retrieval CE

Bank indexing
-------------
Dataset row index is never assumed to equal bank index.  Every item carries an
explicit `bank_idx`, the canonical bank/order hash is asserted at load time by
`frozen_probe.load_frozen`, and `verify_bank_mapping` additionally checks that
each item's own GloVe target is the bank row it points at.

Usage (Phase 2A):
    python scripts/naming_comprehension/train_tasks.py preflight \
        --ckpt <seed22 checkpoint> --taus 0.05 0.10 0.20
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import time
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.lexicon import LexEntry                                      # noqa: E402
from data.phonemes import Vocab                                        # noqa: E402
from losses import alignment_loss, _seq_ce                             # noqa: E402
from models.dual_route import DualRouteModel                           # noqa: E402
from scripts.naming_comprehension.frozen_probe import load_frozen      # noqa: E402
from utils.provenance import git_state, sha256_file                    # noqa: E402

# Exact trainable prefixes per task.  Trailing dots matter: "ltm.decoder." must
# not also capture "ltm.dec_to_premotor.".
TRAINABLE_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "comprehension": ("ltm.encoder.", "ltm.to_semantic."),
    "naming": ("ltm.sem_to_h0.", "ltm.decoder.", "ltm.dec_to_premotor."),
}
# Named explicitly so the freeze is an assertion, not an emergent default.
# phon_embed is ONE shared parameter object used by both routes; motor.proj is
# the shared readout.  Both must stay frozen in these first diagnostics.
ALWAYS_FROZEN: Tuple[str, ...] = ("phon_embed.weight",
                                  "motor.proj.weight", "motor.proj.bias")


# ==================================================  parameter scoping  ====

def set_trainable_scope(model: DualRouteModel, task: str) -> List[str]:
    """Freeze every parameter, then unfreeze exactly the task scope.

    Returns the sorted names of the trainable parameters.  Raises if the scope
    would touch a parameter that must always stay frozen.
    """
    if task not in TRAINABLE_PREFIXES:
        raise ValueError(f"Unknown task {task!r}; expected one of "
                         f"{sorted(TRAINABLE_PREFIXES)}.")
    prefixes = TRAINABLE_PREFIXES[task]
    trainable: List[str] = []
    for name, p in model.named_parameters():
        on = name.startswith(prefixes)
        if on and name in ALWAYS_FROZEN:
            raise RuntimeError(
                f"Scope for task {task!r} would train always-frozen parameter "
                f"{name!r}. This is a bug in TRAINABLE_PREFIXES.")
        p.requires_grad_(on)
        if on:
            trainable.append(name)
    if not trainable:
        raise RuntimeError(f"Empty trainable scope for task {task!r}.")
    for name in ALWAYS_FROZEN:
        p = dict(model.named_parameters())[name]
        if p.requires_grad:
            raise RuntimeError(f"{name} must be frozen but is trainable.")
    return sorted(trainable)


def trainable_parameters(model: DualRouteModel) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def fresh_optimizer(model: DualRouteModel, lr: float, weight_decay: float
                    ) -> torch.optim.Optimizer:
    """A FRESH optimizer over the currently-trainable parameters only.

    The canonical checkpoints' `optimizer_state_dict` is never restored here:
    its parameter scope differs from any task scope, so restoring it would be
    incoherent.  Task training always starts the optimizer from scratch.
    """
    params = trainable_parameters(model)
    if not params:
        raise RuntimeError("No trainable parameters; call set_trainable_scope first.")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def parameter_fingerprint(model: DualRouteModel) -> Dict[str, torch.Tensor]:
    """Detached clones of every parameter, for bit-identity checks."""
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def changed_parameters(model: DualRouteModel,
                       before: Dict[str, torch.Tensor]) -> List[str]:
    return sorted(n for n, p in model.named_parameters()
                  if not torch.equal(p.detach(), before[n]))


# ======================================================  populations  ======

def phonology_groups(entries: Sequence[LexEntry]
                     ) -> Dict[tuple, List[int]]:
    groups: Dict[tuple, List[int]] = collections.defaultdict(list)
    for i, e in enumerate(entries):
        groups[tuple(e.phonemes)].append(i)
    return groups


def unique_phonology_indices(entries: Sequence[LexEntry]) -> List[int]:
    """Bank indices of words whose phonology is unique in the lexicon.

    Derived dynamically; the count is never hard-coded.
    """
    groups = phonology_groups(entries)
    return sorted(g[0] for g in groups.values() if len(g) == 1)


def homophone_indices(entries: Sequence[LexEntry]) -> List[int]:
    groups = phonology_groups(entries)
    return sorted(i for g in groups.values() if len(g) > 1 for i in g)


def verify_bank_mapping(entries: Sequence[LexEntry], bank_raw: torch.Tensor,
                        indices: Sequence[int]) -> None:
    """Assert each item's own GloVe target IS the bank row it points at.

    Guards against any silent drift between dataset ordering and bank ordering
    (the ordered-bank hash is separately asserted by frozen_probe.load_frozen).
    """
    if bank_raw.shape[0] != len(entries):
        raise RuntimeError(f"Bank rows {bank_raw.shape[0]} != entries {len(entries)}")
    for i in indices:
        want = torch.as_tensor(entries[i].semantic, dtype=bank_raw.dtype)
        if not torch.equal(bank_raw[i].cpu(), want):
            raise RuntimeError(
                f"Bank mapping mismatch at index {i} (word={entries[i].word!r}): "
                "bank row does not equal the entry's own GloVe vector.")


# ========================================================  batching  =======

def make_batches(entries: Sequence[LexEntry], bank_indices: Sequence[int],
                 bank_raw: torch.Tensor, vocab: Vocab, batch_size: int,
                 device: str, shuffle: bool = False, seed: int = 0,
                 ) -> Iterator[Dict[str, object]]:
    """Deterministic FLAT batches carrying explicit canonical bank indices.

    Flat = every item of the population exactly once per pass, no frequency
    weighting and no replacement.  With shuffle=False the order is the
    population order, which makes preflight batches reproducible.
    """
    order = list(bank_indices)
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        order = [order[i] for i in torch.randperm(len(order), generator=g).tolist()]

    for lo in range(0, len(order), batch_size):
        idx = order[lo:lo + batch_size]
        items = [entries[i] for i in idx]
        forms = [e.phonemes for e in items]
        B = len(items)
        max_enc = max(len(f) for f in forms) + 1
        max_dec = max_enc
        enc_in = torch.full((B, max_enc), vocab.pad_id, dtype=torch.long)
        enc_mask = torch.zeros((B, max_enc), dtype=torch.bool)
        dec_in = torch.full((B, max_dec), vocab.pad_id, dtype=torch.long)
        dec_tgt = torch.full((B, max_dec), vocab.pad_id, dtype=torch.long)
        for k, f in enumerate(forms):
            enc_in[k, :len(f) + 1] = torch.tensor(f + [vocab.eos_id])
            enc_mask[k, :len(f) + 1] = True
            dec_in[k, :len(f) + 1] = torch.tensor([vocab.bos_id] + f)
            dec_tgt[k, :len(f) + 1] = torch.tensor(f + [vocab.eos_id])
        bank_idx = torch.tensor(idx, dtype=torch.long)
        yield {
            "enc_in": enc_in.to(device),
            "enc_mask": enc_mask.to(device),
            "dec_in": dec_in.to(device),
            "dec_tgt": dec_tgt.to(device),
            # raw (unnormalised) GloVe, gathered BY EXPLICIT BANK INDEX
            "semantic": bank_raw[bank_idx].to(device),
            "bank_idx": bank_idx.to(device),
            "words": [e.word for e in items],
        }


# ====================================================  task forwards  ======

def comprehension_forward(model: DualRouteModel, enc_in: torch.Tensor,
                          enc_mask: torch.Tensor) -> torch.Tensor:
    """phonemes -> s_hat.  Thin wrapper over the existing ltm.encode."""
    return model.ltm.encode(enc_in, enc_mask)


def naming_forward(model: DualRouteModel, sem: torch.Tensor,
                   dec_in: torch.Tensor) -> torch.Tensor:
    """semantic vector + gold prefix -> phoneme logits (teacher forced).

    Exactly `motor(ltm.decode_from_s_hat(sem, dec_in))`; no new machinery.
    """
    return model.motor(model.ltm.decode_from_s_hat(sem, dec_in))


# ======================================================  objectives  =======

def retrieval_loss(s_hat: torch.Tensor, bank_normalized: torch.Tensor,
                   target_bank_idx: torch.Tensor, tau: float) -> torch.Tensor:
    """Full-bank cosine retrieval cross-entropy.

        logits = cos(s_hat, bank) / tau        (B, n_bank)
        L      = CE(logits, target_bank_idx)

    The bank is the canonical L2-normalised buffer and receives no gradient.
    `s_hat` is L2-normalised here so the logits are true cosines: magnitude is
    left to the C0 term, which is what keeps sem_to_h0's input scale honest.
    """
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    q = F.normalize(s_hat, dim=-1)
    logits = (q @ bank_normalized.t()) / tau
    return F.cross_entropy(logits, target_bank_idx)


def comprehension_objective(model: DualRouteModel, batch: Dict[str, object],
                            objective: str, tau: float, lambda_ret: float
                            ) -> Dict[str, torch.Tensor]:
    """C0 or C3 on one batch.  C0 is the existing alignment loss verbatim."""
    s_hat = comprehension_forward(model, batch["enc_in"], batch["enc_mask"])
    c0 = alignment_loss(s_hat, batch["semantic"])
    out = {"c0": c0, "s_hat": s_hat}
    if objective == "c0":
        out["total"] = c0
        return out
    if objective != "c3":
        raise ValueError(f"Unknown objective {objective!r}; expected c0 or c3.")
    ret = retrieval_loss(s_hat, model.ltm.semantic_bank, batch["bank_idx"], tau)
    out["retrieval"] = ret
    out["total"] = c0 + lambda_ret * ret
    return out


def naming_objective(model: DualRouteModel, batch: Dict[str, object],
                     pad_id: int) -> Dict[str, torch.Tensor]:
    """Existing sequence CE on the naming path (raw GloVe in, TF=1.0)."""
    logits = naming_forward(model, batch["semantic"], batch["dec_in"])
    return {"total": _seq_ce(logits, batch["dec_tgt"], pad_id), "logits": logits}


# ==========================================  repetition safeguards  =======
# PREPARED FOR PHASE 2B, NOT RUN IN PHASE 2A.
#
# Interpretation note: a mono-task change in LTM or FULL repetition is a
# quantity to MEASURE, not by itself evidence of an architectural limitation.
# WM is different: with phon_embed and motor.proj frozen, no gradient can
# reach any dorsal parameter, so WM weights must be bit-identical and WM
# repetition must be numerically unchanged.  A WM change is a BUG signal.

DORSAL_INVARIANT_PREFIXES: Tuple[str, ...] = ("wm.", "phon_embed.", "motor.")


def dorsal_fingerprint(model: DualRouteModel) -> Dict[str, torch.Tensor]:
    """Clones of every parameter that must not move in these diagnostics."""
    return {n: p.detach().clone() for n, p in model.named_parameters()
            if n.startswith(DORSAL_INVARIANT_PREFIXES)}


def assert_dorsal_untouched(model: DualRouteModel,
                            before: Dict[str, torch.Tensor]) -> None:
    """Hard check that WM / phon_embed / motor are bit-identical."""
    moved = [n for n, ref in before.items()
             if not torch.equal(dict(model.named_parameters())[n].detach(), ref)]
    if moved:
        raise RuntimeError(
            "Dorsal/shared parameters changed during a mono-task diagnostic "
            f"(this is a bug, not a result): {sorted(moved)}")


@torch.no_grad()
def repetition_snapshot(model: DualRouteModel, vocab: Vocab,
                        entries: Sequence[LexEntry], bank_indices: Sequence[int],
                        bank_raw: torch.Tensor, device: str,
                        batch_size: int = 256,
                        routes: Tuple[str, ...] = ("full", "wm", "ltm"),
                        include_teacher_forced: bool = True,
                        ) -> Dict[str, object]:
    """Route-isolated repetition exact-match on a fixed item set.

    PRIMARY readout: the repository's CANONICAL autoregressive repetition
    evaluation, reused verbatim from
    `scripts.evaluate_train_lexicon_ceiling.evaluate_forms_ar`.  This is the
    convention that produced the cohort's stable-zero selection criterion
    (`decode_mode: "autoregressive"` in the bundled selected_evaluations
    metrics), so before/after numbers are directly comparable to the canonical
    baseline.  No new decoding convention is invented here.

    That canonical convention is FORCED-LENGTH: the greedy loop runs to the
    batch maximum, then each item's readout window is truncated to its own
    gold length + 1 (`len(form) + 1`, the +1 allowing a terminal EOS), and the
    prediction is cut at the first EOS.  It therefore consults the target
    length, which is legitimate for repetition (the form is the input) and is
    exactly why the naming probe cannot reuse it: naming has no phonological
    input, so it uses a global cap instead.

    AUXILIARY readout: teacher-forced, route-isolated, via the shared helper
    `evaluate.hooks.route_predictions`.  Diagnostic only.

    Intended use: call once BEFORE task training and once AFTER, on identical
    indices, and diff.
    """
    from evaluate.hooks import route_predictions              # eval-only deps
    from scripts.evaluate_train_lexicon_ceiling import (
        evaluate_forms_ar, DECODE_AR, EVAL_NOTE_AR)

    was_training = model.training
    model.eval()
    items = [entries[i] for i in bank_indices]

    # ---- primary: canonical forced-length AR (routes isolated) ----
    rows = evaluate_forms_ar(model, vocab, items, device, routes=routes,
                             wm_noise=False)
    ar_exact = {r: sum(row[f"{r}_exact_match"] for row in rows) / max(len(rows), 1)
                for r in routes}
    ar_edit = {r: sum(row[f"{r}_edit_dist"] for row in rows) / max(len(rows), 1)
               for r in routes}

    snapshot: Dict[str, object] = {
        "n_items": len(items),
        "primary_readout": {
            "decode_mode": DECODE_AR,
            "convention": "canonical forced-length AR "
                          "(scripts.evaluate_train_lexicon_ceiling.evaluate_forms_ar)",
            "note": EVAL_NOTE_AR,
            "wm_noise": False,
            "exact_match": ar_exact,
            "mean_edit_distance": ar_edit,
        },
    }

    # ---- auxiliary: teacher-forced ----
    if include_teacher_forced:
        correct = {r: 0 for r in routes}
        total = 0
        for b in make_batches(entries, bank_indices, bank_raw, vocab, batch_size,
                              device, shuffle=False):
            tgt = b["dec_tgt"]
            valid = tgt != vocab.pad_id
            total += tgt.shape[0]
            for r in routes:
                preds, _ = route_predictions(model, b, route=r)
                correct[r] += int(((preds == tgt) | ~valid).all(dim=1).sum())
        snapshot["auxiliary_teacher_forced"] = {
            "exact_match": {r: correct[r] / max(total, 1) for r in routes},
            "note": "diagnostic only; not the canonical repetition metric",
        }

    model.train(was_training)
    return snapshot


# =======================================================  preflight  =======

def _grad_norms(model: DualRouteModel, loss: torch.Tensor
                ) -> Dict[str, float]:
    """L2 grad norms from `loss` alone, without touching .grad or the weights."""
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, [p for _, p in params],
                                retain_graph=True, allow_unused=True)
    per_module: Dict[str, float] = {}
    total_sq = 0.0
    for (name, _), g in zip(params, grads):
        if g is None:
            continue
        sq = float((g.detach() ** 2).sum())
        total_sq += sq
        mod = name.rsplit(".", 1)[0] if name.count(".") <= 2 else \
            ".".join(name.split(".")[:2])
        per_module[mod] = per_module.get(mod, 0.0) + sq
    out = {f"grad_{k}": v ** 0.5 for k, v in per_module.items()}
    out["grad_total"] = total_sq ** 0.5
    return out


def preflight(ckpt_path: str, taus: Sequence[float], n_batches: int,
              batch_size: int, device: str, bench_batches: int,
              shuffle: bool = True, sample_seed: int = 0) -> dict:
    """Scale calibration only.  No optimizer, no weight update, ever.

    `shuffle=True` (default) draws the calibration batches the way the FLAT
    training sampler would, spanning the whole frequency range.  With
    shuffle=False the population is walked in rank order, so the first batches
    contain only the most frequent words — a biased sample for calibration,
    kept available for comparison.
    """
    model, vocab, entries, bank_raw, cfg, ckpt = load_frozen(ckpt_path, device)
    pop = unique_phonology_indices(entries)
    homo = homophone_indices(entries)
    verify_bank_mapping(entries, bank_raw, pop[:200] + homo[:200])

    # comprehension scope; model kept in eval() so no noise/dropout perturbs
    # the measured gradients (both route noises are 0.0 in these checkpoints).
    trainable = set_trainable_scope(model, "comprehension")
    model.eval()

    batches = list(make_batches(entries, pop, bank_raw, vocab, batch_size,
                                device, shuffle=shuffle,
                                seed=sample_seed))[:n_batches]

    rows: List[dict] = []
    for tau in taus:
        acc: Dict[str, List[float]] = collections.defaultdict(list)
        for b in batches:
            s_hat = comprehension_forward(model, b["enc_in"], b["enc_mask"])
            c0 = alignment_loss(s_hat, b["semantic"])
            ret = retrieval_loss(s_hat, model.ltm.semantic_bank,
                                 b["bank_idx"], tau)
            gc0 = _grad_norms(model, c0)
            gret = _grad_norms(model, ret)
            acc["c0"].append(float(c0.detach()))
            acc["retrieval_ce"].append(float(ret.detach()))
            for k, v in gc0.items():
                acc[f"c0_{k}"].append(v)
            for k, v in gret.items():
                acc[f"ret_{k}"].append(v)
        row = {"tau": tau, "n_batches": len(batches), "batch_size": batch_size}
        row.update({k: sum(v) / len(v) for k, v in acc.items()})
        # lambda that equalises the INITIAL gradient scales (calibration only)
        row["lambda_ret_grad_matched"] = (
            row["c0_grad_total"] / row["ret_grad_total"]
            if row["ret_grad_total"] > 0 else float("nan"))
        rows.append(row)

    # ---- runtime benchmark (forward+backward, still no optimizer step) ----
    bench = {}
    for task, population in (("comprehension", pop),
                             ("naming", list(range(len(entries))))):
        set_trainable_scope(model, task)
        bb = list(make_batches(entries, population, bank_raw, vocab,
                               batch_size, device, shuffle=False))[:bench_batches]
        # warmup
        for b in bb[:1]:
            loss = (comprehension_objective(model, b, "c3", taus[0], 1.0)["total"]
                    if task == "comprehension"
                    else naming_objective(model, b, vocab.pad_id)["total"])
            loss.backward()
            model.zero_grad(set_to_none=True)
        t0 = time.time()
        for b in bb:
            loss = (comprehension_objective(model, b, "c3", taus[0], 1.0)["total"]
                    if task == "comprehension"
                    else naming_objective(model, b, vocab.pad_id)["total"])
            loss.backward()
            model.zero_grad(set_to_none=True)
        dt = time.time() - t0
        per_batch = dt / max(len(bb), 1)
        n_pop = len(population)
        n_batches_epoch = -(-n_pop // batch_size)
        bench[task] = {
            "population": n_pop,
            "batches_per_epoch": n_batches_epoch,
            "seconds_per_batch_fwd_bwd": per_batch,
            "estimated_seconds_per_epoch": per_batch * n_batches_epoch,
            "benchmarked_batches": len(bb),
        }
    model.zero_grad(set_to_none=True)

    return {
        "phase": "2A_preflight_scale_calibration",
        "no_optimizer_step": True,
        "calibration_sample": {
            "shuffled_like_flat_training_sampler": shuffle,
            "sample_seed": sample_seed,
            "n_items": min(n_batches * batch_size, len(pop)),
        },
        "populations": {
            "lexicon_total": len(entries),
            "unique_phonology_training_population": len(pop),
            "homophone_words_excluded_from_training": len(homo),
            "retrieval_bank_size": int(model.ltm.semantic_bank.shape[0]),
            "note": ("homophones stay in the retrieval bank as competitors but "
                     "supply no primary comprehension training target"),
        },
        "comprehension_trainable_parameters": trainable,
        "always_frozen": list(ALWAYS_FROZEN),
        "tau_table": rows,
        "runtime_benchmark": bench,
        "provenance": {
            "checkpoint_path": os.path.abspath(ckpt_path),
            "checkpoint_sha256": sha256_file(ckpt_path),
            "checkpoint_training_commit": ckpt.get("git_commit"),
            "lexicon_file_sha256": ckpt.get("lexicon_file_sha256"),
            "ordered_training_words_sha256": ckpt.get("ordered_training_words_sha256"),
            "glove_present": ckpt.get("glove_present"),
            "n_glove_fallback": ckpt.get("n_glove_fallback"),
            "ltm_encoder_mode": cfg.ltm.ltm_encoder_mode,
            "eval_git": git_state(ROOT),
            "device": device,
        },
    }


# ==============================================  Phase 2B run driver  =====
# Single-task acquisition experiments.  Every run starts INDEPENDENTLY from the
# canonical checkpoint; a run is never continued from another run.

EVAL_EVERY = 5                       # evaluation snapshot cadence (epochs)
REPETITION_EPOCHS = (0, 5, 10, 20, 50, 100)   # canonical AR safeguard schedule


def require_real_glove(ckpt: dict, expected_found: Optional[int] = None) -> dict:
    """Fail fast unless the checkpoint certifies full real-GloVe coverage."""
    status = {
        "glove_present": bool(ckpt.get("glove_present", False)),
        "n_glove_found": int(ckpt.get("n_glove_found", -1)),
        "n_glove_fallback": int(ckpt.get("n_glove_fallback", -1)),
    }
    ok = (status["glove_present"] and status["n_glove_fallback"] == 0
          and (expected_found is None or status["n_glove_found"] == expected_found))
    if not ok:
        raise SystemExit(
            f"Real-GloVe coverage not certified by the checkpoint: {status} "
            f"(expected glove_present=True, n_glove_fallback=0"
            + (f", n_glove_found={expected_found}" if expected_found else "") + ").")
    return status


@torch.no_grad()
def evaluate_comprehension(model: DualRouteModel, vocab: Vocab,
                           entries: Sequence[LexEntry], bank_raw: torch.Tensor,
                           strict_idx: Sequence[int], homo_idx: Sequence[int],
                           device: str, batch_size: int = 512) -> dict:
    """Canonical comprehension retrieval metrics, reusing the frozen probe.

    Populations and metrics are exactly those of Phase 1A/1B: retrieval runs
    against the FULL canonical bank, the primary population is the
    unique-phonology words, and homophones are reported separately.
    """
    from scripts.naming_comprehension.frozen_probe import (
        comprehension_metrics, encode_all)
    from scripts.naming_comprehension.aggregate_cohort import (
        FREQ_BANDS, BAND_LABELS)          # frozen Phase 1A band definitions

    all_idx = list(strict_idx) + list(homo_idx)
    forms = [entries[i].phonemes for i in all_idx]
    was_training = model.training
    model.eval()
    s_hat = encode_all(model, vocab, forms, device, batch_size)
    m = comprehension_metrics(s_hat.cpu(), bank_raw.cpu(), all_idx, batch_size)
    model.train(was_training)

    n_strict = len(strict_idx)
    same_phon = [
        int(tuple(entries[int(m["top1_idx"][k])].phonemes)
            == tuple(entries[i].phonemes))
        for k, i in enumerate(all_idx)]

    def agg(sl: slice) -> dict:
        import numpy as np
        return {
            "n": len(range(*sl.indices(len(all_idx)))),
            "target_cos_mean": float(np.mean(m["target_cos"][sl])),
            "target_rank_median": float(np.median(m["target_rank"][sl])),
            "top1": float(np.mean(m["top1"][sl])),
            "top5": float(np.mean(m["top5"][sl])),
            "margin_mean": float(np.mean(m["margin"][sl])),
            "c_ltm_mean_aux": float(np.mean(m["c_ltm"][sl])),
        }

    import numpy as np
    out = {
        "strict_unique_phonology": agg(slice(0, n_strict)),
        "homophones_separate": {
            **agg(slice(n_strict, len(all_idx))),
            "class_aware_top1_aux": float(np.mean(same_phon[n_strict:])),
        },
        "frequency_bands_strict": {},
    }
    ranks = np.array([entries[i].rank for i in strict_idx])
    for (lo, hi), lab in zip(FREQ_BANDS, BAND_LABELS):
        sel = (ranks >= lo) & (ranks < hi)
        out["frequency_bands_strict"][lab] = {
            "n": int(sel.sum()),
            "top1": float(m["top1"][:n_strict][sel].mean()) if sel.any() else None,
        }
    return out


@torch.no_grad()
def evaluate_naming(model: DualRouteModel, vocab: Vocab,
                    entries: Sequence[LexEntry], bank_raw: torch.Tensor,
                    indices: Sequence[int], device: str, max_steps: int,
                    batch_size: int = 512,
                    return_per_item: bool = False) -> dict:
    """Naming from RAW target GloVe via the committed semantic greedy AR decoder.

    Decoding is BOS -> greedy -> EOS or the global cap; the target word length
    is never consulted.
    """
    from scripts.naming_comprehension.frozen_probe import semantic_greedy_decode
    from scripts.external_eval import _edit_distance

    was_training = model.training
    model.eval()
    exact = edit = eos = 0
    rows: List[dict] = []
    for lo in range(0, len(indices), batch_size):
        idx = list(indices[lo:lo + batch_size])
        sem = bank_raw[torch.tensor(idx)].to(device)      # RAW GloVe, by bank index
        preds, eos_flags = semantic_greedy_decode(model, sem, vocab, max_steps)
        for k, i in enumerate(idx):
            gold = entries[i].phonemes
            ok = int(preds[k] == gold)
            ed = _edit_distance(preds[k], gold)
            exact += ok
            edit += ed
            eos += int(eos_flags[k])
            if return_per_item:
                rows.append({
                    "bank_index": i, "word": entries[i].word,
                    "length": len(gold), "freq_rank": entries[i].rank,
                    "pred": " ".join(vocab.itos[p] for p in preds[k]),
                    "exact": ok, "edit": ed, "pred_len": len(preds[k]),
                    "eos_emitted": int(eos_flags[k]),
                })
    model.train(was_training)
    n = max(len(indices), 1)
    out = {"n": len(indices), "exact_match": exact / n,
           "whole_word_error_rate": 1.0 - exact / n,
           "mean_edit": edit / n, "eos_emission_rate": eos / n}
    if return_per_item:
        out["_per_item"] = rows
    return out


def _write_tsv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")


def run_task(ckpt_path: str, task: str, objective: str, out_dir: str,
             lr: float = 1e-4, weight_decay: float = 1e-5,
             batch_size: int = 64, epochs: int = 100,
             tau: float = 0.10, lambda_ret: float = 0.087,
             data_seed: int = 0, device: str = "cpu",
             eval_every: int = EVAL_EVERY) -> dict:
    """One single-task acquisition run, always from the canonical checkpoint.

    No validation set exists in this diagnostic, so no checkpoint is ever
    selected on held-out performance and none is called "best": the budget is
    fixed and the final epoch is saved as-is.
    """
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)

    model, vocab, entries, bank_raw, cfg, ckpt = load_frozen(ckpt_path, device)
    glove_status = require_real_glove(ckpt, expected_found=len(entries))

    strict_idx = unique_phonology_indices(entries)
    homo_idx = homophone_indices(entries)
    all_idx = list(range(len(entries)))
    train_pop = strict_idx if task == "comprehension" else all_idx
    verify_bank_mapping(entries, bank_raw, train_pop[::997] + homo_idx[::97])

    trainable = set_trainable_scope(model, task)
    optim = fresh_optimizer(model, lr=lr, weight_decay=weight_decay)
    initial = parameter_fingerprint(model)          # for bit-identity audit
    dorsal_ref = dorsal_fingerprint(model)
    max_steps = cfg.data.max_phonemes + 1           # committed naming cap

    def snapshot(epoch: int) -> dict:
        s: Dict[str, object] = {"epoch": epoch}
        if task == "comprehension":
            s["comprehension"] = evaluate_comprehension(
                model, vocab, entries, bank_raw, strict_idx, homo_idx, device)
        else:
            s["naming"] = evaluate_naming(
                model, vocab, entries, bank_raw, all_idx, device, max_steps)
        if epoch in REPETITION_EPOCHS:
            s["repetition"] = repetition_snapshot(
                model, vocab, entries, all_idx, bank_raw, device,
                include_teacher_forced=True)
        return s

    trajectory: List[dict] = []
    snapshots: List[dict] = [snapshot(0)]
    print(f"[{task}/{objective}] epoch 0 snapshot done", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        acc_total = 0.0
        acc_parts: Dict[str, float] = {}
        nb = 0
        for b in make_batches(entries, train_pop, bank_raw, vocab, batch_size,
                              device, shuffle=True, seed=data_seed * 100000 + epoch):
            if task == "comprehension":
                o = comprehension_objective(model, b, objective, tau, lambda_ret)
            else:
                o = naming_objective(model, b, vocab.pad_id)
            loss = o["total"]
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            acc_total += float(loss.detach())
            for k in ("c0", "retrieval"):
                if k in o:
                    acc_parts[k] = acc_parts.get(k, 0.0) + float(o[k].detach())
            nb += 1
        rec = {"epoch": epoch, "train_loss": acc_total / max(nb, 1), "n_batches": nb}
        rec.update({f"train_{k}": v / max(nb, 1) for k, v in acc_parts.items()})
        trajectory.append(rec)

        # dorsal invariants are checked EVERY epoch: a violation is a bug and
        # must stop the run immediately, not be discovered at the end.
        assert_dorsal_untouched(model, dorsal_ref)

        if epoch % eval_every == 0:
            snapshots.append(snapshot(epoch))
            print(f"[{task}/{objective}] epoch {epoch} loss={rec['train_loss']:.4f}",
                  flush=True)

    # ---- final scope audit: everything outside the scope is bit-identical ----
    changed = changed_parameters(model, initial)
    scope_ok = set(changed) <= set(trainable)
    if not scope_ok:
        raise RuntimeError(
            "Parameters outside the approved trainable scope changed: "
            f"{sorted(set(changed) - set(trainable))}")

    # ---- final per-item outputs at the fixed budget (never 'best') ----
    if task == "naming":
        final_items = evaluate_naming(model, vocab, entries, bank_raw, all_idx,
                                      device, max_steps, return_per_item=True)
        _write_tsv(os.path.join(out_dir, "final_per_item.tsv"),
                   final_items.pop("_per_item"))

    ckpt_out = os.path.join(out_dir, f"final_epoch_{epochs}.pt")
    torch.save({"model_state_dict": model.state_dict(),
                "task": task, "objective": objective, "epochs": epochs,
                "note": "fixed-budget final epoch; NOT a validation-selected checkpoint",
                "source_checkpoint_sha256": sha256_file(ckpt_path)}, ckpt_out)

    result = {
        "phase": "2B_single_task_acquisition",
        "task": task,
        "objective": objective,
        "budget": {"epochs": epochs, "batch_size": batch_size,
                   "eval_every": eval_every, "early_stopping": False,
                   "lr": lr, "weight_decay": weight_decay,
                   "lr_schedule": "constant",
                   "optimizer": "AdamW, fresh, task scope only; checkpoint "
                                "optimizer state never restored"},
        "objective_config": ({"tau": tau, "lambda_ret": lambda_ret}
                             if objective == "c3" else {}),
        "populations": {
            "training_population": len(train_pop),
            "training_population_kind": ("unique-phonology words"
                                         if task == "comprehension"
                                         else "all lexicon words"),
            "sampler": "flat/uniform, without replacement, reshuffled per epoch",
            "retrieval_bank_size": int(model.ltm.semantic_bank.shape[0]),
            "strict_unique_phonology": len(strict_idx),
            "homophones": len(homo_idx),
        },
        "trainable_parameters": trainable,
        "always_frozen": list(ALWAYS_FROZEN),
        "scope_audit": {
            "changed_parameters": sorted(changed),
            "all_changes_within_scope": scope_ok,
            "dorsal_bit_identical": True,
        },
        "trajectory": trajectory,
        "snapshots": snapshots,
        "final_checkpoint": os.path.abspath(ckpt_out),
        "provenance": {
            "source_checkpoint_path": os.path.abspath(ckpt_path),
            "source_checkpoint_sha256": sha256_file(ckpt_path),
            "checkpoint_training_commit": ckpt.get("git_commit"),
            "lexicon_file_sha256": ckpt.get("lexicon_file_sha256"),
            "ordered_training_words_sha256": ckpt.get("ordered_training_words_sha256"),
            "glove": glove_status,
            "ltm_encoder_mode": cfg.ltm.ltm_encoder_mode,
            "naming_decode_cap": max_steps,
            "data_seed": data_seed,
            "device": device,
            "eval_git": git_state(ROOT),
            "runtime_seconds": round(time.time() - t_start, 1),
        },
    }
    with open(os.path.join(out_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    _write_tsv(os.path.join(out_dir, "trajectory.tsv"), trajectory)
    print(f"[{task}/{objective}] done -> {out_dir}", flush=True)
    return result


# ==========================  Phase 2C subset capacity / overfit diagnostic ==
# Question: can the ALREADY VALIDATED C3 and N0 scopes MEMORISE a very small
# set of semantic mappings once scale is removed as the main difficulty?
#
# This is explicitly an OVERFIT diagnostic on TRAINED items.  There is no
# validation set, no held-out population and therefore no generalization
# claim of any kind.  Performance-based stopping is legitimate here precisely
# because nothing is being selected for generalization: the stopping rule is a
# predeclared statement about training-set acquisition.
#
# Nothing about the validated configuration is altered: tau, lambda_ret, lr,
# weight decay, the parameter scopes, the objectives and the decoding
# conventions are all inherited unchanged from Phase 2B.  The ONLY changes are
# the training population (64 items instead of 26,682 / 29,571), the budget,
# and the evaluation cadence.
#
# CRITICAL: only the TRAINING population shrinks.  Retrieval remains a
# full-bank 29,571-way problem, and every target is addressed by its explicit
# canonical bank index.

SUBSET_EVAL_EVERY = 50               # scheduled evaluation cadence (epochs)
SUBSET_MAX_EPOCHS = 1000             # hard budget; 1 epoch == 1 optimizer step
SUBSET_CONSECUTIVE_SUCCESSES = 2     # required before an early stop

# Predeclared success criteria, fixed BEFORE the runs.  Not to be relaxed.
C3_SUBSET_SUCCESS = {
    "top1_min": 0.95,                # strict subset top-1 >= 95%
    "median_target_rank_max": 1.0,   # median target rank == 1
    "margin_min_exclusive": 0.0,     # mean target-vs-best-wrong margin > 0
}
NAMING_SUBSET_SUCCESS = {
    "exact_min": 0.95,               # free-AR exact >= 95%
    "wer_max": 0.05,                 # whole-word error rate <= 5%
}


def band_of_rank(rank: int, bands: Sequence[Tuple[int, int]]) -> Optional[int]:
    """Index of the frequency band containing `rank`, or None."""
    for i, (lo, hi) in enumerate(bands):
        if lo <= rank < hi:
            return i
    return None


def nested_band_ordering(entries: Sequence[LexEntry], subset_seed: int = 0
                         ) -> Dict[str, List[int]]:
    """Deterministic per-band ordering of the unique-phonology bank indices.

    This ordering is the single source of truth for subset selection and is
    deliberately INDEPENDENT of the requested subset size: a subset of k items
    per band is always the first k of the same permutation.  Nesting
    (subset64 subset-of subset512 subset-of ...) is therefore true by
    construction rather than by convention.

    The per-band permutation is seeded by (subset_seed, band index) so the
    bands are permuted independently and reproducibly.  The pre-permutation
    pool is sorted by bank index, which makes the whole construction a pure
    function of the lexicon, never of dict/set iteration order.
    """
    from scripts.naming_comprehension.aggregate_cohort import (
        FREQ_BANDS, BAND_LABELS)          # frozen Phase 1A band definitions

    pools: Dict[str, List[int]] = {lab: [] for lab in BAND_LABELS}
    for i in unique_phonology_indices(entries):
        b = band_of_rank(entries[i].rank, FREQ_BANDS)
        if b is not None:
            pools[BAND_LABELS[b]].append(i)

    ordering: Dict[str, List[int]] = {}
    for bi, lab in enumerate(BAND_LABELS):
        pool = sorted(pools[lab])
        g = torch.Generator().manual_seed(subset_seed * 1000 + bi)
        perm = torch.randperm(len(pool), generator=g).tolist()
        ordering[lab] = [pool[k] for k in perm]
    return ordering


def select_nested_subset(entries: Sequence[LexEntry], per_band: int,
                         subset_seed: int = 0) -> List[int]:
    """Exactly `per_band` unique-phonology bank indices from each fixed band.

    Selection is a seeded permutation, NOT the head of the frequency ranking:
    taking the most frequent items of each band would confound "small" with
    "easy".  Returned in band order, then in per-band permutation order.
    """
    from scripts.naming_comprehension.aggregate_cohort import BAND_LABELS

    ordering = nested_band_ordering(entries, subset_seed)
    out: List[int] = []
    for lab in BAND_LABELS:
        pool = ordering[lab]
        if len(pool) < per_band:
            raise RuntimeError(
                f"Band {lab!r} has only {len(pool)} unique-phonology words; "
                f"cannot select {per_band}.")
        out.extend(pool[:per_band])
    return out


def subset_records(entries: Sequence[LexEntry], indices: Sequence[int],
                   vocab: Vocab) -> List[dict]:
    """Full human-readable definition of the selected subset, in order."""
    from scripts.naming_comprehension.aggregate_cohort import (
        FREQ_BANDS, BAND_LABELS)

    rows: List[dict] = []
    for pos, i in enumerate(indices):
        e = entries[i]
        b = band_of_rank(e.rank, FREQ_BANDS)
        rows.append({
            "position": pos,
            "bank_index": int(i),
            "word": e.word,                 # orthography (LexEntry.word)
            "phonemes": " ".join(vocab.itos[p] for p in e.phonemes),
            "phoneme_ids": " ".join(str(p) for p in e.phonemes),
            "n_phonemes": len(e.phonemes),
            "freq_rank": int(e.rank),
            "band": BAND_LABELS[b] if b is not None else "OUT_OF_BANDS",
        })
    return rows


def subset_definition_hash(records: Sequence[dict]) -> str:
    """SHA256 of the ORDERED subset definition.

    Order-sensitive by construction: the position field is part of the payload,
    so a permutation of the same 64 words yields a different digest.
    """
    payload = "\n".join(
        f"{r['position']}\t{r['bank_index']}\t{r['word']}\t"
        f"{r['phoneme_ids']}\t{r['freq_rank']}" for r in records)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@torch.no_grad()
def evaluate_comprehension_subset(model: DualRouteModel, vocab: Vocab,
                                  entries: Sequence[LexEntry],
                                  bank_raw: torch.Tensor,
                                  subset_idx: Sequence[int], device: str,
                                  batch_size: int = 64,
                                  return_per_item: bool = False) -> dict:
    """Comprehension retrieval for the subset items against the FULL bank.

    Reuses the canonical Phase 1A helpers verbatim (`encode_all` +
    `comprehension_metrics`); only the probed population is restricted.  The
    bank passed in is the full 29,571-row canonical bank, so top-1 remains a
    29,571-way decision for every one of the 64 trained items.
    """
    from scripts.naming_comprehension.frozen_probe import (
        comprehension_metrics, encode_all)
    import numpy as np

    was_training = model.training
    model.eval()
    forms = [entries[i].phonemes for i in subset_idx]
    s_hat = encode_all(model, vocab, forms, device, batch_size)
    m = comprehension_metrics(s_hat.cpu(), bank_raw.cpu(), list(subset_idx),
                              batch_size)
    model.train(was_training)

    out = {
        "n": len(subset_idx),
        "retrieval_bank_size": int(bank_raw.shape[0]),
        "top1": float(np.mean(m["top1"])),
        "top5": float(np.mean(m["top5"])),
        "target_cos_mean": float(np.mean(m["target_cos"])),
        "target_rank_median": float(np.median(m["target_rank"])),
        "target_rank_mean": float(np.mean(m["target_rank"])),
        "target_rank_max": float(np.max(m["target_rank"])),
        "margin_mean": float(np.mean(m["margin"])),
        "c_ltm_mean_aux": float(np.mean(m["c_ltm"])),
    }
    if return_per_item:
        out["_per_item"] = [{
            "position": k,
            "bank_index": int(i),
            "word": entries[i].word,
            "freq_rank": int(entries[i].rank),
            "n_phonemes": len(entries[i].phonemes),
            "target_rank": int(m["target_rank"][k]),
            "target_cos": float(m["target_cos"][k]),
            "top1": int(m["top1"][k]),
            "top5": int(m["top5"][k]),
            "margin": float(m["margin"][k]),
            "top1_word": entries[int(m["top1_idx"][k])].word,
        } for k, i in enumerate(subset_idx)]
    return out


def _agg_comprehension_rows(rows: Sequence[dict]) -> Optional[dict]:
    import numpy as np
    if not rows:
        return None
    return {
        "n": len(rows),
        "top1": float(np.mean([r["top1"] for r in rows])),
        "top5": float(np.mean([r["top5"] for r in rows])),
        "target_rank_median": float(np.median([r["target_rank"] for r in rows])),
        "target_rank_mean": float(np.mean([r["target_rank"] for r in rows])),
        "target_cos_mean": float(np.mean([r["target_cos"] for r in rows])),
        "margin_mean": float(np.mean([r["margin"] for r in rows])),
    }


def _agg_naming_rows(rows: Sequence[dict]) -> Optional[dict]:
    import numpy as np
    if not rows:
        return None
    exact = float(np.mean([r["exact"] for r in rows]))
    return {
        "n": len(rows),
        "exact_match": exact,
        "whole_word_error_rate": 1.0 - exact,
        "mean_edit": float(np.mean([r["edit"] for r in rows])),
        "eos_emission_rate": float(np.mean([r["eos_emitted"] for r in rows])),
        "mean_pred_length": float(np.mean([r["pred_len"] for r in rows])),
        "mean_target_length": float(np.mean([r["length"] for r in rows])),
    }


def nested_scale_metrics(rows: Sequence[dict], cores: Dict[str, set],
                         task: str) -> dict:
    """Track every nested core separately inside the current population.

    The scientific point of a NESTED subset design: because each smaller
    subset is a strict subset of the larger one, the very same items can be
    followed across scales.  A drop on a core at a larger scale is
    interference, not a sampling difference — the items are identical by
    construction.

    `cores` maps a group name to its bank-index set.  "added" is defined
    relative to the LARGEST core, i.e. the items this scale introduced over
    the previous one.  Also reports the four frozen Phase 1A frequency bands.
    """
    from scripts.naming_comprehension.aggregate_cohort import (
        FREQ_BANDS, BAND_LABELS)

    agg = _agg_comprehension_rows if task == "comprehension" else _agg_naming_rows
    out: Dict[str, object] = {"all": agg(rows), "cores": {}}
    for name, idx in sorted(cores.items(), key=lambda kv: len(kv[1])):
        out["cores"][name] = agg([r for r in rows if r["bank_index"] in idx])

    if cores:
        largest_name = max(cores, key=lambda k: len(cores[k]))
        largest = cores[largest_name]
        out["added"] = agg([r for r in rows if r["bank_index"] not in largest])
        out["added_relative_to"] = largest_name
    else:
        out["added"] = None
        out["added_relative_to"] = None

    out["frequency_bands"] = {}
    for bi, lab in enumerate(BAND_LABELS):
        out["frequency_bands"][lab] = agg(
            [r for r in rows if band_of_rank(r["freq_rank"], FREQ_BANDS) == bi])
    return out


def c3_subset_success(comp: dict) -> bool:
    """Predeclared C3-64 acquisition criterion; all three must hold."""
    return (comp["top1"] >= C3_SUBSET_SUCCESS["top1_min"]
            and comp["target_rank_median"] <= C3_SUBSET_SUCCESS["median_target_rank_max"]
            and comp["margin_mean"] > C3_SUBSET_SUCCESS["margin_min_exclusive"])


def naming_subset_success(nam: dict) -> bool:
    """Predeclared N0-64 acquisition criterion; both must hold."""
    return (nam["exact_match"] >= NAMING_SUBSET_SUCCESS["exact_min"]
            and nam["whole_word_error_rate"] <= NAMING_SUBSET_SUCCESS["wer_max"])


@torch.no_grad()
def _subset_loss_components(model: DualRouteModel, entries: Sequence[LexEntry],
                            subset_idx: Sequence[int], bank_raw: torch.Tensor,
                            vocab: Vocab, batch_size: int, device: str,
                            task: str, objective: str, tau: float,
                            lambda_ret: float) -> Dict[str, float]:
    """Objective components over the subset, in eval mode, no gradient.

    Averaged over minibatches of the SAME size used in training rather than
    one giant batch: the full-bank retrieval logits are (B, 29571), so a
    single 4096-item batch would allocate hundreds of MB and would not match
    the training regime.  Batch-mean losses over equal-sized batches average
    correctly; the final batch may be smaller, so it is weighted by its size.
    """
    was_training = model.training
    model.eval()
    totals: Dict[str, float] = {}
    n_seen = 0
    for b in make_batches(entries, subset_idx, bank_raw, vocab, batch_size,
                          device, shuffle=False):
        bs = int(b["bank_idx"].shape[0])
        if task == "comprehension":
            o = comprehension_objective(model, b, objective, tau, lambda_ret)
            parts = {"c0": float(o["c0"]), "total": float(o["total"])}
            if "retrieval" in o:
                parts["retrieval_ce"] = float(o["retrieval"])
        else:
            parts = {"total": float(naming_objective(model, b, vocab.pad_id)["total"])}
        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + v * bs
        n_seen += bs
    model.train(was_training)
    return {k: v / max(n_seen, 1) for k, v in totals.items()}


def run_subset_task(ckpt_path: str, task: str, objective: str, out_dir: str,
                    per_band: int = 16, subset_seed: int = 0,
                    lr: float = 1e-4, weight_decay: float = 1e-5,
                    batch_size: int = 64, epochs: int = SUBSET_MAX_EPOCHS,
                    tau: float = 0.10, lambda_ret: float = 0.087,
                    data_seed: int = 0, device: str = "cpu",
                    eval_every: int = SUBSET_EVAL_EVERY,
                    core_per_band: Optional[object] = None,
                    run_repetition: bool = True) -> dict:
    """One Phase 2C subset capacity run, always from the canonical checkpoint.

    Never continued from a Phase 2B run or from the other Phase 2C run: each
    run independently reloads the frozen canonical checkpoint.

    The saved checkpoint is a fixed-budget-or-predeclared-criterion endpoint.
    It is NOT validation-selected and must never be described as "best".
    """
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)

    model, vocab, entries, bank_raw, cfg, ckpt = load_frozen(ckpt_path, device)
    glove_status = require_real_glove(ckpt, expected_found=len(entries))

    # ---- deterministic nested subset (same 64 words for both tasks) ----
    subset_idx = select_nested_subset(entries, per_band, subset_seed)
    records = subset_records(entries, subset_idx, vocab)
    sub_hash = subset_definition_hash(records)

    strict_set = set(unique_phonology_indices(entries))
    if not set(subset_idx) <= strict_set:
        raise RuntimeError("Subset contains non-unique-phonology words.")
    forms = [tuple(entries[i].phonemes) for i in subset_idx]
    if len(set(forms)) != len(forms):
        raise RuntimeError("Subset contains a repeated phonological form.")
    verify_bank_mapping(entries, bank_raw, subset_idx)

    # ---- nested cores for the cross-scale split (verified, never assumed) ----
    # Each core is regenerated from the SAME deterministic per-band ordering,
    # so membership derives from the existing nested definition rather than
    # from any stored copy.  The whole ascending chain is checked to be
    # strictly nested, and the largest core strictly inside this subset.
    cores: Dict[str, set] = {}
    core_info: Optional[dict] = None
    if core_per_band:
        sizes = sorted({int(c) for c in
                        (core_per_band if isinstance(core_per_band, (list, tuple))
                         else [core_per_band])})
        if any(c > per_band for c in sizes):
            raise ValueError(
                f"core per-band sizes {sizes} exceed per_band={per_band}.")
        chain: List[Tuple[str, List[int]]] = []
        for c in sizes:
            idx = select_nested_subset(entries, c, subset_seed)
            chain.append((f"core{len(idx)}", idx))
        for (na, a), (nb, b) in zip(chain, chain[1:]):
            if not set(a) < set(b):
                raise RuntimeError(
                    f"Nesting violated: {na} is not a strict subset of {nb}. "
                    "The nested design is broken.")
        if not set(chain[-1][1]) < set(subset_idx):
            raise RuntimeError(
                f"Nesting violated: {chain[-1][0]} is not a strict subset of "
                "the current subset. The nested design is broken.")
        cores = {name: set(idx) for name, idx in chain}
        largest = chain[-1]
        core_info = {
            "core_per_band_sizes": sizes,
            "chain": [{"name": name, "n": len(idx),
                       "definition_sha256": subset_definition_hash(
                           subset_records(entries, idx, vocab))}
                      for name, idx in chain],
            "added_relative_to": largest[0],
            "added_n": len(subset_idx) - len(largest[1]),
            "chain_strictly_nested_verified": True,
        }

    all_idx = list(range(len(entries)))
    trainable = set_trainable_scope(model, task)
    optim = fresh_optimizer(model, lr=lr, weight_decay=weight_decay)
    initial = parameter_fingerprint(model)
    dorsal_ref = dorsal_fingerprint(model)
    max_steps = cfg.data.max_phonemes + 1           # committed naming cap

    def evaluate(epoch: int, per_item: bool = False) -> dict:
        s: Dict[str, object] = {"epoch": epoch}
        if task == "comprehension":
            m = evaluate_comprehension_subset(
                model, vocab, entries, bank_raw, subset_idx, device,
                return_per_item=True)
            rows = m.pop("_per_item")
            s["comprehension"] = m
        else:
            m = evaluate_naming(model, vocab, entries, bank_raw, subset_idx,
                                device, max_steps, return_per_item=True)
            rows = m.pop("_per_item")
            m["mean_pred_length"] = (sum(r["pred_len"] for r in rows)
                                     / max(len(rows), 1))
            m["mean_target_length"] = (sum(r["length"] for r in rows)
                                       / max(len(rows), 1))
            s["naming"] = m
        if cores:
            s["nested_scale"] = nested_scale_metrics(rows, cores, task)
        s["loss_components"] = _subset_loss_components(
            model, entries, subset_idx, bank_raw, vocab, batch_size, device,
            task, objective, tau, lambda_ret)
        if per_item:
            for r in rows:
                for name, idx in cores.items():
                    r[f"in_{name}"] = int(r["bank_index"] in idx)
            s["_per_item_rows"] = rows
        return s

    def criterion_met(snap: dict) -> bool:
        return (c3_subset_success(snap["comprehension"])
                if task == "comprehension"
                else naming_subset_success(snap["naming"]))

    # ---- repetition BEFORE training (full lexicon, canonical AR) ----
    # Deferrable: Phase 2B and 2C1 already established the LTM repetition
    # consequence, and this pass dominates wall time.  The canonical evaluator
    # is untouched; only its invocation is skipped.
    repetition_before = None
    if run_repetition:
        repetition_before = repetition_snapshot(
            model, vocab, entries, all_idx, bank_raw, device,
            include_teacher_forced=True)
        print(f"[2C {task}] repetition(before) "
              f"{repetition_before['primary_readout']['exact_match']}", flush=True)

    trajectory: List[dict] = []
    snapshots: List[dict] = [evaluate(0)]
    consecutive = 1 if criterion_met(snapshots[0]) else 0
    first_success_epoch: Optional[int] = 0 if consecutive else None
    stop_epoch = epochs
    stop_reason = "fixed budget exhausted"

    for epoch in range(1, epochs + 1):
        model.train()
        acc_total = 0.0
        acc_parts: Dict[str, float] = {}
        nb = 0
        for b in make_batches(entries, subset_idx, bank_raw, vocab, batch_size,
                              device, shuffle=True,
                              seed=data_seed * 100000 + epoch):
            if task == "comprehension":
                o = comprehension_objective(model, b, objective, tau, lambda_ret)
            else:
                o = naming_objective(model, b, vocab.pad_id)
            loss = o["total"]
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            acc_total += float(loss.detach())
            for k in ("c0", "retrieval"):
                if k in o:
                    acc_parts[k] = acc_parts.get(k, 0.0) + float(o[k].detach())
            nb += 1
        rec = {"epoch": epoch, "train_loss": acc_total / max(nb, 1),
               "n_optimizer_steps": nb}
        rec.update({f"train_{k}": v / max(nb, 1) for k, v in acc_parts.items()})
        trajectory.append(rec)

        # A violation is a BUG and must stop the run immediately.
        assert_dorsal_untouched(model, dorsal_ref)

        if epoch % eval_every == 0:
            snap = evaluate(epoch)
            snapshots.append(snap)
            ok = criterion_met(snap)
            if ok:
                consecutive += 1
                if first_success_epoch is None:
                    first_success_epoch = epoch
            else:
                consecutive = 0
                first_success_epoch = None
            key = (f"top1={snap['comprehension']['top1']:.4f} "
                   f"medrank={snap['comprehension']['target_rank_median']:.0f}"
                   if task == "comprehension"
                   else f"exact={snap['naming']['exact_match']:.4f}")
            print(f"[2C {task}] epoch {epoch} loss={rec['train_loss']:.4f} "
                  f"{key} criterion={ok} streak={consecutive}", flush=True)
            if consecutive >= SUBSET_CONSECUTIVE_SUCCESSES:
                stop_epoch = epoch
                stop_reason = (f"predeclared training-set criterion met at "
                               f"{SUBSET_CONSECUTIVE_SUCCESSES} consecutive "
                               f"scheduled evaluations")
                break

    # ---- final evaluation at the stopping epoch, with per-item detail ----
    final_snapshot = evaluate(stop_epoch, per_item=True)
    per_item = final_snapshot.pop("_per_item_rows")
    if not snapshots or snapshots[-1]["epoch"] != stop_epoch:
        snapshots.append(final_snapshot)
    _write_tsv(os.path.join(out_dir, "final_per_item.tsv"), per_item)

    # ---- repetition AFTER training (identical items and convention) ----
    repetition_after = None
    if run_repetition:
        repetition_after = repetition_snapshot(
            model, vocab, entries, all_idx, bank_raw, device,
            include_teacher_forced=True)

    # ---- scope audit ----
    changed = changed_parameters(model, initial)
    scope_ok = set(changed) <= set(trainable)
    if not scope_ok:
        raise RuntimeError(
            "Parameters outside the approved trainable scope changed: "
            f"{sorted(set(changed) - set(trainable))}")

    ckpt_out = os.path.join(out_dir, f"final_epoch_{stop_epoch}.pt")
    torch.save({"model_state_dict": model.state_dict(),
                "task": task, "objective": objective, "epochs": stop_epoch,
                "subset_definition_sha256": sub_hash,
                "note": "fixed-budget or predeclared training-set criterion "
                        "endpoint; NOT a validation-selected checkpoint",
                "source_checkpoint_sha256": sha256_file(ckpt_path)}, ckpt_out)

    _write_tsv(os.path.join(out_dir, "subset_definition.tsv"), records)
    with open(os.path.join(out_dir, "subset_definition.json"), "w",
              encoding="utf-8") as f:
        json.dump({"subset_definition_sha256": sub_hash,
                   "n": len(subset_idx), "per_band": per_band,
                   "subset_seed": subset_seed,
                   "bank_indices_in_order": [int(i) for i in subset_idx],
                   "items": records}, f, indent=2)

    from scripts.naming_comprehension.aggregate_cohort import BAND_LABELS
    band_counts = collections.Counter(r["band"] for r in records)
    lengths = [r["n_phonemes"] for r in records]

    result = {
        "phase": "2C1_subset64_capacity_overfit_diagnostic",
        "diagnostic_kind": ("training-set memorisation / capacity; NO held-out "
                            "population, NO generalization claim"),
        "task": task,
        "objective": objective,
        "subset": {
            "n": len(subset_idx),
            "per_band": per_band,
            "subset_seed": subset_seed,
            "selection": ("seeded permutation within each frozen Phase 1A "
                          "frequency band, NOT the head of the ranking"),
            "nested": ("per-band ordering is independent of subset size, so a "
                       "larger subset is a strict superset by construction"),
            "band_counts": {lab: band_counts[lab] for lab in BAND_LABELS},
            "subset_definition_sha256": sub_hash,
            "all_unique_phonology": True,
            "phoneme_length": {
                "min": min(lengths), "max": max(lengths),
                "mean": sum(lengths) / len(lengths),
            },
            "nested_core": core_info,
        },
        "budget": {"max_epochs": epochs, "batch_size": batch_size,
                   "eval_every": eval_every,
                   "optimizer_steps_per_epoch": 1,
                   "lr": lr, "weight_decay": weight_decay,
                   "lr_schedule": "constant",
                   "optimizer": "AdamW, fresh, task scope only; checkpoint "
                                "optimizer state never restored",
                   "stopping": ("predeclared training-set acquisition "
                                "criterion at N consecutive scheduled "
                                "evaluations, or budget exhaustion"),
                   "consecutive_successes_required": SUBSET_CONSECUTIVE_SUCCESSES},
        "objective_config": ({"tau": tau, "lambda_ret": lambda_ret}
                             if objective == "c3" else {}),
        "success_criterion": (C3_SUBSET_SUCCESS if task == "comprehension"
                              else NAMING_SUBSET_SUCCESS),
        "outcome": {
            "criterion_reached": first_success_epoch is not None
                                 and consecutive >= SUBSET_CONSECUTIVE_SUCCESSES,
            "first_epoch_criterion_met": first_success_epoch,
            "stopped_at_epoch": stop_epoch,
            "stop_reason": stop_reason,
        },
        "populations": {
            "training_population": len(subset_idx),
            "training_population_kind": "64-word nested unique-phonology subset",
            "sampler": "uniform over the subset, one full pass per step",
            "retrieval_bank_size": int(model.ltm.semantic_bank.shape[0]),
            "retrieval_note": ("bank NOT shrunk to the subset: retrieval stays "
                               "a full-lexicon-way decision"),
        },
        "trainable_parameters": trainable,
        "always_frozen": list(ALWAYS_FROZEN),
        "scope_audit": {
            "changed_parameters": sorted(changed),
            "all_changes_within_scope": scope_ok,
            "dorsal_bit_identical": True,
            "dorsal_checked_every_epoch": True,
        },
        "repetition": {
            "ran": run_repetition,
            "before": repetition_before,
            "after": repetition_after,
            "note": ("full-lexicon canonical AR, before training and once at "
                     "the stopping epoch only" if run_repetition else
                     "DEFERRED: the LTM repetition consequence is already "
                     "established by Phase 2B and 2C1; this pass dominates "
                     "wall time and is not needed to decide subset capacity. "
                     "The canonical evaluator is unmodified, only unused."),
        },
        "trajectory": trajectory,
        "snapshots": snapshots,
        "final_snapshot": final_snapshot,
        "final_checkpoint": os.path.abspath(ckpt_out),
        "provenance": {
            "source_checkpoint_path": os.path.abspath(ckpt_path),
            "source_checkpoint_sha256": sha256_file(ckpt_path),
            "checkpoint_training_commit": ckpt.get("git_commit"),
            "lexicon_file_sha256": ckpt.get("lexicon_file_sha256"),
            "ordered_training_words_sha256": ckpt.get("ordered_training_words_sha256"),
            "glove": glove_status,
            "ltm_encoder_mode": cfg.ltm.ltm_encoder_mode,
            "naming_decode_cap": max_steps,
            "data_seed": data_seed,
            "device": device,
            "torch_version": torch.__version__,
            "eval_git": git_state(ROOT),
            "runtime_seconds": round(time.time() - t_start, 1),
        },
    }
    with open(os.path.join(out_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    _write_tsv(os.path.join(out_dir, "trajectory.tsv"), trajectory)
    print(f"[2C {task}] done -> {out_dir}", flush=True)
    return result


# ============================================================  main  =======

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pf = sub.add_parser("preflight", help="scale calibration; no weight update")
    pf.add_argument("--ckpt", required=True)
    pf.add_argument("--taus", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    pf.add_argument("--n-batches", type=int, default=8)
    pf.add_argument("--batch-size", type=int, default=64)
    pf.add_argument("--bench-batches", type=int, default=10)
    pf.add_argument("--device", default="cpu")
    pf.add_argument("--no-shuffle", action="store_true",
                    help="walk the population in rank order (biased sample; "
                         "default is shuffled like the flat training sampler)")
    pf.add_argument("--sample-seed", type=int, default=0)
    pf.add_argument("--out", default=None, help="optional JSON output path")

    rn = sub.add_parser("run", help="Phase 2B single-task acquisition run")
    rn.add_argument("--ckpt", required=True, help="canonical source checkpoint")
    rn.add_argument("--task", required=True, choices=["comprehension", "naming"])
    rn.add_argument("--objective", default="c0", choices=["c0", "c3"],
                    help="comprehension objective; ignored for naming")
    rn.add_argument("--out-dir", required=True)
    rn.add_argument("--lr", type=float, default=1e-4)
    rn.add_argument("--weight-decay", type=float, default=1e-5)
    rn.add_argument("--batch-size", type=int, default=64)
    rn.add_argument("--epochs", type=int, default=100)
    rn.add_argument("--tau", type=float, default=0.10)
    rn.add_argument("--lambda-ret", type=float, default=0.087)
    rn.add_argument("--data-seed", type=int, default=0)
    rn.add_argument("--device", default="cpu")

    sb = sub.add_parser("subset", help="Phase 2C subset capacity/overfit run")
    sb.add_argument("--ckpt", required=True, help="canonical source checkpoint")
    sb.add_argument("--task", required=True, choices=["comprehension", "naming"])
    sb.add_argument("--objective", default="c3", choices=["c0", "c3"],
                    help="comprehension objective; ignored for naming")
    sb.add_argument("--out-dir", required=True)
    sb.add_argument("--per-band", type=int, default=16)
    sb.add_argument("--subset-seed", type=int, default=0)
    sb.add_argument("--lr", type=float, default=1e-4)
    sb.add_argument("--weight-decay", type=float, default=1e-5)
    sb.add_argument("--batch-size", type=int, default=64)
    sb.add_argument("--epochs", type=int, default=SUBSET_MAX_EPOCHS)
    sb.add_argument("--eval-every", type=int, default=SUBSET_EVAL_EVERY)
    sb.add_argument("--tau", type=float, default=0.10)
    sb.add_argument("--lambda-ret", type=float, default=0.087)
    sb.add_argument("--data-seed", type=int, default=0)
    sb.add_argument("--device", default="cpu")
    sb.add_argument("--core-per-band", type=int, nargs="+", default=None,
                    help="per-band size(s) of the nested CORE subsets; enables "
                         "the core-vs-added cross-scale split. Several may be "
                         "given (e.g. 16 128 -> core64 and core512); 'added' is "
                         "then relative to the largest core.")
    sb.add_argument("--no-repetition", action="store_true",
                    help="defer the expensive full-lexicon canonical repetition "
                         "passes (the evaluator itself is untouched)")
    args = ap.parse_args(argv)

    if args.cmd == "subset":
        res = run_subset_task(args.ckpt, args.task, args.objective, args.out_dir,
                              per_band=args.per_band, subset_seed=args.subset_seed,
                              lr=args.lr, weight_decay=args.weight_decay,
                              batch_size=args.batch_size, epochs=args.epochs,
                              tau=args.tau, lambda_ret=args.lambda_ret,
                              data_seed=args.data_seed, device=args.device,
                              eval_every=args.eval_every,
                              core_per_band=args.core_per_band,
                              run_repetition=not args.no_repetition)
        print(json.dumps({"subset": res["subset"], "outcome": res["outcome"],
                          "final": res["final_snapshot"]}, indent=2))
        return 0

    if args.cmd == "run":
        res = run_task(args.ckpt, args.task, args.objective, args.out_dir,
                       lr=args.lr, weight_decay=args.weight_decay,
                       batch_size=args.batch_size, epochs=args.epochs,
                       tau=args.tau, lambda_ret=args.lambda_ret,
                       data_seed=args.data_seed, device=args.device)
        print(json.dumps(res["snapshots"][-1], indent=2))
        return 0

    res = preflight(args.ckpt, args.taus, args.n_batches, args.batch_size,
                    args.device, args.bench_batches,
                    shuffle=not args.no_shuffle, sample_seed=args.sample_seed)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"[train_tasks] preflight -> {args.out}")
    print(json.dumps({k: res[k] for k in
                      ("populations", "tau_table", "runtime_benchmark")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
