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


def levenshtein(a: Sequence[int], b: Sequence[int]) -> int:
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def first_divergence(pred: Sequence[int], tgt: Sequence[int]):
    for i in range(min(len(pred), len(tgt))):
        if pred[i] != tgt[i]:
            return i
    return None if len(pred) == len(tgt) else min(len(pred), len(tgt))


# ------------------------------------------------------------- distributions --
def summarize(values: Sequence[float]) -> dict:
    """n / mean / sample SD / min / p01..p99 / max, in float64."""
    import statistics
    v = sorted(float(x) for x in values)
    if not v:
        return {"n": 0}
    def pct(p: float) -> float:
        if len(v) == 1:
            return v[0]
        k = p * (len(v) - 1)
        lo, hi = int(k), min(int(k) + 1, len(v) - 1)
        return v[lo] + (k - lo) * (v[hi] - v[lo])
    return {"n": len(v), "mean": sum(v) / len(v),
            "sd": statistics.stdev(v) if len(v) > 1 else 0.0,
            "min": v[0], "p01": pct(.01), "p05": pct(.05), "p25": pct(.25),
            "p50": pct(.50), "p75": pct(.75), "p95": pct(.95), "p99": pct(.99),
            "max": v[-1]}


# --------------------------------------------------------- ordering / status --
def classify_ordering(acc_wm: float, acc_ltm: float,
                      ned_wm: float, ned_ltm: float) -> dict:
    """Frozen rule.  No post-hoc tolerance; float64 throughout."""
    d_acc = float(acc_wm) - float(acc_ltm)
    d_ned = float(ned_ltm) - float(ned_wm)
    if d_acc == 0.0 and d_ned == 0.0:
        label = "TIE"
    elif d_acc >= 0.0 and d_ned >= 0.0:
        label = "WM_DOMINANT"
    elif d_acc <= 0.0 and d_ned <= 0.0:
        label = "LTM_DOMINANT"
    else:
        label = "MIXED"
    return {"delta_acc": d_acc, "delta_ned": d_ned, "ordering": label}


def preservation(source_label: str, post_label: str) -> str:
    if source_label == "WM_DOMINANT" and post_label == "WM_DOMINANT":
        return "PRESERVED"
    if source_label != "WM_DOMINANT" and post_label == "WM_DOMINANT":
        return "PRESERVED_FROM_NONDOMINANT_SOURCE"
    if source_label == "WM_DOMINANT" and post_label != "WM_DOMINANT":
        return "NOT_PRESERVED"
    return "NO_EXPECTED_PATTERN_AT_SOURCE"


def load_manifest(name: str) -> List[dict]:
    """Read a frozen stimulus manifest (comment lines carry provenance)."""
    import csv
    path = os.path.join(CONTRACT_DIR, name)
    with open(path) as fh:
        lines = [l for l in fh if not l.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))
