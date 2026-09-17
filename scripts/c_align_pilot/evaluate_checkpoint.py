#!/usr/bin/env python3
"""C-ALIGN CAUSAL PILOT — frozen post-hoc evaluator (read-only).

One checkpoint in, one JSON record out.  Every readout is the historical frozen
implementation; nothing here trains, and no parameter, optimizer tensor or RNG
state of a checkpoint is ever written back.

Readouts (contract §12-§15, CENTRAL §11-§12):

GLOBAL      FULL canonical / FULL genuine free-AR / Naming strict / C strict
            top-1 error counts, gate descriptive statistics.
ROUTE       native isolated ventral (LTM) free-AR and canonical errors,
            WM-only canonical errors.
INTERFACE   S0 native errors, contemporaneous S1c errors, the S0-S1c gap, the
            two transition counts, retrieval identity, homophone retrieval,
            semantic geometry, and the C retrieval / alignment losses.

S1c is recomputed from the CHECKPOINT'S OWN top-1 retrieval (never a stale
baseline identity), then injected as the RAW prototype through the unchanged
isolated ventral decoder.

Usage (read-only):
    python3 scripts/c_align_pilot/evaluate_checkpoint.py \
        --checkpoint <path.pt> --label <arm@step> --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.frozen_probe import (  # noqa: E402
    comprehension_metrics, encode_all)
from scripts.naming_comprehension.train_joint_scratch import (  # noqa: E402
    LAMBDA_C, TAU, build_batch)
from scripts.naming_comprehension.train_tasks import (  # noqa: E402
    comprehension_forward, retrieval_loss)
from losses import alignment_loss                                  # noqa: E402
from ventral_interface.decode import decode_condition              # noqa: E402
from ventral_interface.injection import SemanticInjection          # noqa: E402

GLOVE = os.environ.get(
    "L3_GLOVE",
    "/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/lichtheim3/data/glove.6B.300d.txt")
RETRIEVAL_BATCH = 512          # historical C battery batch
DECODE_BATCH = 256             # closed ventral-interface decoding contract
C_LOSS_BATCH = 64              # historical training batch, for the loss readout
CONVENTIONS = ("freear", "canonical")
EVALUATOR_VERSION = "c_align_pilot_evaluator_v1"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(ck_path: str, device: str):
    from scripts.naming_comprehension.frozen_head_probe import build_trainer
    tr, ckd = build_trainer(ck_path, device, GLOVE)
    tr.model.eval()
    return tr, ckd


# ------------------------------------------------------------------ global --
@torch.no_grad()
def global_readouts(tr) -> Dict[str, float]:
    """The historical full-lexicon battery, exactly as the V6 detector ran it."""
    row = tr.evaluate(with_probe=False, with_full_lexicon=True)
    n = len(tr.entries)
    out = {
        "full_rep_canonical_errors": int(round((1.0 - row["full_rep_full"]) * n)),
        "full_rep_freear_errors": int(round((1.0 - row["full_rep_freear"]) * n)),
        "wm_canonical_errors": int(round((1.0 - row["full_rep_wm"]) * n)),
        "wm_freear_errors": int(round((1.0 - row["full_rep_freear_wm"]) * n)),
        "ltm_canonical_errors": int(round((1.0 - row["full_rep_ltm"]) * n)),
        "ltm_freear_errors": int(round((1.0 - row["full_rep_freear_ltm"]) * n)),
        "naming_strict_errors": int(round(
            (1.0 - row["full_naming_exact"]) * n)),
        "c_top1_errors": int(round(
            (1.0 - row["full_comp_top1"]) * len(tr.comp_idx))),
        "n_repetition_items": n,
        "n_comprehension_items": len(tr.comp_idx),
    }
    for k in ("full_rep_full", "full_rep_freear", "full_rep_wm", "full_rep_ltm",
              "full_rep_freear_ltm", "full_rep_freear_wm", "full_comp_top1",
              "full_comp_top5", "full_naming_exact", "full_naming_wer",
              "gate_mean", "gate_std", "gate_p05", "gate_p95",
              "gate_frac_below_0.05", "gate_frac_above_0.95"):
        if k in row and row[k] is not None:
            out[f"rate_{k}"] = float(row[k])
    return out


# --------------------------------------------------------------- interface --
@torch.no_grad()
def decode_all(model, tr, injection: SemanticInjection, convention: str,
               device: str, limit: int = 0) -> np.ndarray:
    """exact_correct per lexical row under one bound condition."""
    forms = [e.phonemes for e in tr.entries][:limit or len(tr.entries)]
    exact = np.zeros(len(forms), dtype=np.int64)
    for lo in range(0, len(forms), DECODE_BATCH):
        rows = list(range(lo, min(lo + DECODE_BATCH, len(forms))))
        res = decode_condition(model, tr.vocab, [forms[i] for i in rows], rows,
                               injection, convention, device)
        for k, i in enumerate(rows):
            exact[i] = int(list(res["preds"][k]) == list(forms[i]))
        injection.unbind()
    return exact


@torch.no_grad()
def interface_readouts(tr, device: str, limit: int = 0) -> Dict[str, object]:
    model = tr.model
    n = limit if limit else len(tr.entries)
    forms = [e.phonemes for e in tr.entries][:n]
    bank_raw = tr.bank_raw.cpu().float()

    # contemporaneous retrieval: this checkpoint's own top-1 row per item
    s_hat = encode_all(model, tr.vocab, forms, device, RETRIEVAL_BATCH).cpu().float()
    m = comprehension_metrics(s_hat, bank_raw, list(range(n)), RETRIEVAL_BATCH)
    top1_idx = np.asarray(m["top1_idx"], dtype=np.int64)

    out: Dict[str, object] = {}
    exact: Dict[str, Dict[str, np.ndarray]] = {}
    for convention in CONVENTIONS:
        s0 = decode_all(model, tr, SemanticInjection("S0"), convention, device,
                        limit=limit)
        s1c = decode_all(model, tr,
                         SemanticInjection("S1", fixed=bank_raw[top1_idx]),
                         convention, device, limit=limit)
        exact[convention] = {"S0": s0, "S1c": s1c}
        gap = int((1 - s0).sum()) - int((1 - s1c).sum())
        out.update({
            f"S0_{convention}_errors": int((1 - s0).sum()),
            f"S1c_{convention}_errors": int((1 - s1c).sum()),
            f"gap_{convention}": gap,
            f"S0wrong_to_S1c_correct_{convention}": int(((s0 == 0) & (s1c == 1)).sum()),
            f"S0correct_to_S1c_wrong_{convention}": int(((s0 == 1) & (s1c == 0)).sum()),
        })
        out[f"S1C_BOUND_DEGRADED_{convention}"] = bool(
            out[f"S1c_{convention}_errors"] > 500 or gap < 0)

    # retrieval identity and homophone structure
    ident = top1_idx == np.arange(n)
    same_phon = np.array([tuple(tr.entries[int(r)].phonemes)
                          == tuple(tr.entries[i].phonemes)
                          for i, r in enumerate(top1_idx)])
    comp_idx = np.asarray([i for i in tr.comp_idx if i < n], dtype=np.int64)
    out.update({
        "lexical_identity_errors_all_rows": int((~ident).sum()),
        "lexical_identity_errors_C_population": int((~ident[comp_idx]).sum()),
        "homophone_retrievals_all_rows": int(((~ident) & same_phon).sum()),
        "phonology_correct_retrievals_all_rows": int(same_phon.sum()),
    })

    # semantic geometry (means over all rows and over the C population)
    tgt = bank_raw[:n]
    cos = F.cosine_similarity(s_hat, tgt, dim=-1).numpy()
    nrm = s_hat.norm(dim=-1).numpy()
    tnrm = tgt.norm(dim=-1).numpy()
    mse = ((s_hat - tgt) ** 2).mean(dim=-1).numpy()
    for label, sel in (("all_rows", np.arange(n)), ("C_population", comp_idx)):
        if sel.size == 0:
            continue
        out.update({
            f"cos_shat_target_mean_{label}": float(cos[sel].mean()),
            f"cos_shat_target_median_{label}": float(np.median(cos[sel])),
            f"norm_shat_mean_{label}": float(nrm[sel].mean()),
            f"norm_target_mean_{label}": float(tnrm[sel].mean()),
            f"norm_ratio_mean_{label}": float((nrm[sel] / tnrm[sel]).mean()),
            f"mse_shat_target_mean_{label}": float(mse[sel].mean()),
            f"retrieval_top1_cos_mean_{label}": float(
                np.asarray(m["c_ltm"])[sel].mean()),
            f"retrieval_margin_mean_{label}": float(np.asarray(m["margin"])[sel].mean()),
            f"target_cos_mean_{label}": float(np.asarray(m["target_cos"])[sel].mean()),
            f"target_rank_median_{label}": float(np.median(np.asarray(m["target_rank"])[sel])),
        })
    return out


@torch.no_grad()
def loss_readouts(tr, device: str) -> Dict[str, float]:
    """C retrieval and unit alignment losses on the canonical C population,
    partitioned by the historical batch size (no gradients, no update)."""
    idx = list(tr.comp_idx)
    rets, aligns, ns = [], [], []
    for lo in range(0, len(idx), C_LOSS_BATCH):
        sl = idx[lo:lo + C_LOSS_BATCH]
        b = build_batch(tr.entries, tr.bank_raw, tr.vocab, sl, device)
        s_hat = comprehension_forward(tr.model, b["enc_in"], b["enc_mask"])
        rets.append(float(retrieval_loss(s_hat, tr.model.ltm.semantic_bank,
                                         b["bank_idx"], TAU).detach()))
        aligns.append(float(alignment_loss(s_hat, b["semantic"]).detach()))
        ns.append(len(sl))
    w = np.asarray(ns, dtype=np.float64) / float(sum(ns))
    return {
        "L_C_retrieval_raw_mean": float(np.average(rets, weights=w)),
        "L_C_retrieval_effective_mean": float(LAMBDA_C * np.average(rets, weights=w)),
        "L_C_align_unit_mean": float(np.average(aligns, weights=w)),
        "n_c_batches": len(ns),
    }


def evaluate(ck_path: str, label: str, device: str,
             smoke_items: int = 0) -> Dict[str, object]:
    """smoke_items > 0 restricts the interface decode to the first N rows and
    skips the global battery.  It exists only for plumbing tests: such a record
    is stamped SMOKE_ONLY and is never a scientific readout."""
    from gate_x_lesion.hooks import state_dict_sha256
    before = sha256_file(ck_path)
    tr, ckd = build(ck_path, device)
    params_before = state_dict_sha256(tr.model)
    rec: Dict[str, object] = {
        "label": label,
        "checkpoint": os.path.abspath(ck_path),
        "checkpoint_sha256": before,
        "global_step": int(ckd.get("global_step", -1)),
        "seed": int(ckd.get("seed", -1)),
        "c_align_weight": float(ckd.get("c_align_weight", 0.0)),
        "schedule": ckd.get("schedule"),
        "schedule_anchor_step": int(ckd.get("schedule_anchor_step", -1)),
        "cursors": {k: int(v) for k, v in (ckd.get("cursors") or {}).items()},
        "phase_transitions": ckd.get("phase_transitions") or [],
        "params_sha256": params_before,
        "evaluator_version": EVALUATOR_VERSION,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "torch": torch.__version__,
        "device": device,
    }
    rec["SMOKE_ONLY"] = bool(smoke_items)
    if smoke_items:
        rec.update(interface_readouts(tr, device, limit=smoke_items))
    else:
        rec.update(global_readouts(tr))
        rec.update(interface_readouts(tr, device))
        rec.update(loss_readouts(tr, device))
        # internal consistency: S0 IS the native isolated ventral decode, so
        # the injection path and the historical battery must agree exactly.
        rec["S0_matches_global_ltm_freear"] = bool(
            rec["S0_freear_errors"] == rec["ltm_freear_errors"])
        rec["S0_matches_global_ltm_canonical"] = bool(
            rec["S0_canonical_errors"] == rec["ltm_canonical_errors"])
    # read-only proof: nothing about the state moved, and the file is untouched
    rec["params_sha256_after"] = state_dict_sha256(tr.model)
    rec["checkpoint_sha256_after"] = sha256_file(ck_path)
    rec["read_only_ok"] = bool(
        rec["params_sha256_after"] == params_before
        and rec["checkpoint_sha256_after"] == before
        and all(p.grad is None for p in tr.model.parameters()))
    if not rec["read_only_ok"]:
        raise RuntimeError("HARD STOP: evaluator mutated the state it evaluated")
    return rec


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke-items", type=int, default=0,
                    help="PLUMBING TEST ONLY: decode the first N rows and skip "
                         "the global battery; the record is stamped SMOKE_ONLY")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rec = evaluate(a.checkpoint, a.label, a.device, smoke_items=a.smoke_items)
    path = os.path.join(a.out, f"{a.label}.json")
    json.dump(rec, open(path, "w"), indent=1, sort_keys=True)
    if rec.get("SMOKE_ONLY"):
        print(f"[evaluator SMOKE_ONLY] {a.label}: "
              f"S0={rec['S0_freear_errors']} S1c={rec['S1c_freear_errors']} "
              f"-> {path}")
    else:
        print(f"[evaluator] {a.label}: "
              f"LTM_freeAR={rec['S0_freear_errors']} "
              f"S1c={rec['S1c_freear_errors']} C={rec['c_top1_errors']} "
              f"FULL={rec['full_rep_canonical_errors']} "
              f"WM={rec['wm_canonical_errors']} "
              f"Naming={rec['naming_strict_errors']} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
