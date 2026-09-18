"""V7 pre-lesion intact-route + pseudoword validation: READ-ONLY evaluation.

Frozen by CENTRAL 2026-09-18.  This module NEVER trains, never updates a
parameter, never writes a checkpoint, and never evaluates `head_final`.

POST_REPAIR is built IN MEMORY as:  SOURCE checkpoint + head_first_c0 deployed
state ({"2.weight","2.bias"}).  The source file is hashed before and after every
state and asserted unchanged.

Decode semantics are REUSED from the frozen V7 tree without modification:
  * genuine free-AR  -> JointScratchTrainer.free_ar_repetition
                        (BOS start, first-EOS trim, global FREE_AR_MAX_STEPS=12,
                         target length never terminates decoding)
  * canonical        -> repetition_snapshot (forced length; reported alongside)
  * naming / C / gate-> the frozen battery helpers
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Sequence, Tuple

import torch

from rules import (  # frozen rules live in a torch-free module
    classify_ordering, first_divergence, levenshtein, preservation, summarize)

CONTRACT_DIR = os.path.join(os.path.dirname(__file__), "..", "contract")
DEPLOYED_HEAD_KEYS = {"2.weight", "2.bias"}


# ----------------------------------------------------------------- hashing --
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def deployed_state_digest(state: Dict[str, torch.Tensor]) -> str:
    """Deterministic digest over key name, dtype, shape and raw tensor bytes."""
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k]
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ------------------------------------------------------------ state loading --
def load_repair_head(head_path: str) -> Tuple[Dict[str, torch.Tensor], str, str]:
    """Read head_first_c0.pt read-only and return its deployed state."""
    blob = torch.load(head_path, map_location="cpu", weights_only=False)
    state = blob["state"]
    if set(state) != DEPLOYED_HEAD_KEYS:
        raise RuntimeError(
            f"REFUSED: derived head must carry exactly {sorted(DEPLOYED_HEAD_KEYS)}; "
            f"got {sorted(state)}")
    return state, sha256_file(head_path), deployed_state_digest(state)


def build_state(ckpt_path: str, head_path: str | None, glove: str):
    """Return (trainer, model, provenance).  head_path=None => SOURCE state.

    The source checkpoint is hashed before and after; callers must invoke
    `assert_source_unchanged` when finished.
    """
    from scripts.naming_comprehension.ceiling_source_completion import (
        build_trainer, _isolated_model)

    before = sha256_file(ckpt_path)
    tr, _ = build_trainer(ckpt_path, "cpu", glove)
    prov = {"source_checkpoint": ckpt_path, "source_sha256": before}

    if head_path is None:
        model = tr.model
        prov.update({"state_kind": "SOURCE", "repair_head_file": None,
                     "repair_head_file_sha256": None,
                     "repair_head_deployed_state_sha256": None})
    else:
        state, hsha, hdig = load_repair_head(head_path)
        model = _isolated_model(tr, "p_last_hinge", state)
        prov.update({"state_kind": "POST_REPAIR",
                     "repair_head_file": os.path.basename(head_path),
                     "repair_head_file_sha256": hsha,
                     "repair_head_deployed_state_sha256": hdig})
    return tr, model, prov


def assert_source_unchanged(ckpt_path: str, before: str) -> None:
    after = sha256_file(ckpt_path)
    if after != before:
        raise RuntimeError(f"REFUSED: source checkpoint mutated: {ckpt_path}")


def model_state_digest(model) -> str:
    return deployed_state_digest({k: v for k, v in model.state_dict().items()})


# ------------------------------------------------------- free-AR item level --
@torch.no_grad()
def free_ar_items(tr, model, entries, routes: Sequence[str] = ("full", "wm", "ltm"),
                  batch_size: int = 256) -> Dict[str, List[dict]]:
    """Item-level genuine free-AR, semantics identical to the frozen
    `free_ar_repetition`: BOS start, global FREE_AR_MAX_STEPS horizon, trim at
    first EOS, exact-sequence scoring.  Target length is NEVER consulted to
    terminate decoding -- it is used only to score and to report diagnostics.
    """
    from scripts.naming_comprehension.train_joint_scratch import FREE_AR_MAX_STEPS

    vocab = tr.vocab
    was_training = model.training
    model.eval()
    out: Dict[str, List[dict]] = {r: [] for r in routes}
    key_of = {"full": "logits", "wm": "wm_logits", "ltm": "ltm_logits"}
    try:
        for route in routes:
            for lo in range(0, len(entries), batch_size):
                chunk = entries[lo:lo + batch_size]
                forms = [e["phonemes"] for e in chunk]
                max_enc = max(len(f) for f in forms) + 1
                enc_in = torch.full((len(chunk), max_enc), vocab.pad_id, dtype=torch.long)
                enc_mask = torch.zeros((len(chunk), max_enc), dtype=torch.bool)
                for k, f in enumerate(forms):
                    enc_in[k, :len(f) + 1] = torch.tensor(f + [vocab.eos_id])
                    enc_mask[k, :len(f) + 1] = True
                enc_in, enc_mask = enc_in.to(tr.device), enc_mask.to(tr.device)
                dec = torch.full((len(chunk), 1), vocab.bos_id,
                                 dtype=torch.long, device=tr.device)
                for _ in range(FREE_AR_MAX_STEPS):
                    o = model(enc_in, enc_mask, dec)
                    nxt = o[key_of[route]][:, -1, :].argmax(-1, keepdim=True)
                    dec = torch.cat([dec, nxt], dim=1)
                    if bool((dec == vocab.eos_id).any(dim=1).all()):
                        break
                for k, e in enumerate(chunk):
                    raw = dec[k, 1:].tolist()
                    if vocab.eos_id in raw:
                        eos_pos, seq = raw.index(vocab.eos_id), raw[:raw.index(vocab.eos_id)]
                    else:
                        eos_pos, seq = None, list(raw)
                    tgt = e["phonemes"]
                    out[route].append({
                        "item_id": e["item_id"], "route": route,
                        "target_ids": " ".join(map(str, tgt)),
                        "predicted_ids": " ".join(map(str, seq)),
                        "target_str": " ".join(vocab.itos[i] for i in tgt),
                        "predicted_str": " ".join(vocab.itos[i] for i in seq),
                        "exact": int(seq == tgt),
                        "raw_edit_distance": levenshtein(seq, tgt),
                        "normalized_edit_distance":
                            levenshtein(seq, tgt) / max(len(tgt), 1),
                        "target_length": len(tgt), "predicted_length": len(seq),
                        "eos_position": eos_pos, "no_eos": int(eos_pos is None),
                        "first_divergence": first_divergence(seq, tgt),
                        "decode_convention": "GENUINE_FREE_AR",
                    })
    finally:
        model.train(was_training)
    return out


# ------------------------------------------------------------- distributions --


# --------------------------------------------------------- ordering / status --


def load_manifest(name: str) -> List[dict]:
    """Read a frozen stimulus manifest (comment lines carry provenance)."""
    import csv
    path = os.path.join(CONTRACT_DIR, name)
    with open(path) as fh:
        lines = [l for l in fh if not l.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


# ------------------------------------------------- canonical (forced length) --
def canonical_items(tr, model, bank_indices, routes=("full", "wm", "ltm")):
    """Item-level CANONICAL repetition, via the frozen evaluator verbatim.

    `evaluate_forms_ar` is exactly the function `train_tasks.repetition_snapshot`
    uses for its primary readout, so this is reuse, not a reimplementation. The
    aggregate below is the identical expression `repetition_snapshot` applies to
    the same rows; a test asserts the two agree on the smoke set.
    """
    from scripts.evaluate_train_lexicon_ceiling import evaluate_forms_ar

    was_training = model.training
    model.eval()
    try:
        items = [tr.entries[i] for i in bank_indices]
        rows = evaluate_forms_ar(model, tr.vocab, items, "cpu",
                                 routes=routes, wm_noise=False)
    finally:
        model.train(was_training)
    agg = {r: sum(row[f"{r}_exact_match"] for row in rows) / max(len(rows), 1)
           for r in routes}
    return rows, agg


# ------------------------------------------------------- gating diagnostics --
@torch.no_grad()
def gating_items(tr, model, bank_indices, batch_size: int = 256):
    """Item-level c_LTM and g, read from the frozen model outputs.

    c_LTM is the LTM route's max cosine similarity to the semantic bank
    (`ltm_route.LTMLexicon.lexical_field` -> "confidence", surfaced by
    `DualRouteModel.forward` as `field_confidence`); g is the gate value the
    model itself computes, g = sigmoid(alpha * (c_LTM - gate_threshold)).
    Neither is recomputed here.
    """
    from scripts.naming_comprehension.train_joint_scratch import build_batch

    was_training = model.training
    model.eval()
    rows = []
    try:
        for lo in range(0, len(bank_indices), batch_size):
            idx = list(bank_indices[lo:lo + batch_size])
            b = build_batch(tr.entries, tr.bank_raw, tr.vocab, idx, tr.device)
            o = model(b["enc_in"], b["enc_mask"], b["dec_in"])
            g = o["gate"].detach().reshape(len(idx), -1).float().mean(dim=1).cpu()
            conf = o.get("field_confidence")
            c = (conf.detach().reshape(len(idx), -1).float().mean(dim=1).cpu()
                 if conf is not None else None)
            for k, i in enumerate(idx):
                rows.append({
                    "bank_index": int(i),
                    "c_ltm": (None if c is None else float(c[k])),
                    "g": float(g[k]),
                })
    finally:
        model.train(was_training)
    return rows
