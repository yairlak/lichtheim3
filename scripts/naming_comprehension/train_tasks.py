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
    args = ap.parse_args(argv)

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
