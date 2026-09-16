"""Isolated-ventral repetition decoding under an injected semantic vector.

Decoding logic is NOT reimplemented: both conventions call the frozen GATING
functions with `routes=("ltm",)`:

* GENUINE FREE-AR (primary)  : `gate_probe.ar_decode_free` (cap imported from
  `train_joint_scratch.FREE_AR_MAX_STEPS`, asserted == 12)
* CANONICAL FORCED-LENGTH    : `gate_probe.ar_decode_forced_length`

The passive `TokenRecorder` reconstructs the raw greedy tokens (needed for EOS fields
and first divergence); the reconstruction is checked against the historical function's
own returned predictions for every row, and any mismatch hard-stops.

`native_freear_diagnostic` is new, diagnostic-only code.  Its prefix-corrected
continuation is labelled DIAGNOSTIC_ONLY_PREFIX_CORRECTION and never feeds a primary
metric.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from gating_diagnostics.gate_probe import (
    HISTORICAL_FREE_AR_MAX_STEPS, _route_step_logits, ar_decode_forced_length,
    ar_decode_free)
from ventral_interface import ROUTE
from ventral_interface.injection import SemanticInjection, TokenRecorder, injected

FREE_AR_MAX_STEPS = HISTORICAL_FREE_AR_MAX_STEPS
assert FREE_AR_MAX_STEPS == 12, "historical genuine free-AR cap must be 12"

DIAGNOSTIC_LABEL = "DIAGNOSTIC_ONLY_PREFIX_CORRECTION"
TOP_K = 5
#: Absolute tolerance between free-AR step logits and gold-prefix (teacher-forced)
#: logits on an identical prefix.  Measured float32 kernel noise is ~1e-5 on logits of
#: magnitude ~80 (step 0 only: length-1 vs length-n GRU call), i.e. ~1e-7 relative.
LOGIT_PATH_TOL = 1e-4
NA = "NA"


def _cut_at_eos(tokens: Sequence[int], eos: int) -> List[int]:
    out = []
    for t in tokens:
        if t == eos:
            break
        out.append(int(t))
    return out


def first_divergence(raw: Sequence[int], form: Sequence[int], eos: int) -> Optional[int]:
    """First generated position k (0-based, over the target form + [EOS]) where the
    greedy token differs from the target.  None iff the item is exact-correct."""
    target = list(form) + [eos]
    for k, want in enumerate(target):
        if k >= len(raw):
            raise RuntimeError("HARD STOP: generated sequence shorter than divergence scan")
        if int(raw[k]) != want:
            return k
    return None


def eos_fields(raw: Sequence[int], form: Sequence[int], eos: int, convention: str) -> Dict:
    """EOS protocol fields.  `raw` = all greedy tokens generated for the row
    (free-AR: until the batch loop stopped; canonical: the per-item readout
    window of len(form)+1 tokens)."""
    L = len(form)
    first_eos = next((k for k, t in enumerate(raw) if int(t) == eos), None)
    pred = _cut_at_eos(raw, eos)
    return {
        "pred_length": len(pred),
        "eos_emitted": int(first_eos is not None),
        "first_eos_step": NA if first_eos is None else first_eos,
        "eos_before_target_length": int(first_eos is not None and first_eos < L),
        "eos_after_target_length": int(first_eos is not None and first_eos > L),
        "terminated_by_cap": (int(first_eos is None) if convention == "freear" else NA),
    }


@torch.no_grad()
def decode_condition(model, vocab, forms: Sequence[Sequence[int]], rows: Sequence[int],
                     injection: SemanticInjection, convention: str, device: str,
                     max_steps: int = FREE_AR_MAX_STEPS) -> Dict[str, object]:
    """Decode one batch under one bound condition.  Returns per-item predictions,
    raw greedy tokens, step logits and the live s_hat the hook observed."""
    rec = TokenRecorder()
    with injected(model, injection, rec):
        injection.bind(rows)
        if convention == "freear":
            preds = ar_decode_free(model, vocab, forms, device, routes=(ROUTE,),
                                   max_steps=max_steps)[ROUTE]
        elif convention == "canonical":
            preds = ar_decode_forced_length(model, vocab, forms, device,
                                            routes=(ROUTE,))[ROUTE]
        else:
            raise ValueError(convention)
        live = injection.first_live
        deg = injection.last_degenerate
        step_dev = injection.max_live_step_dev
    toks = rec.tokens()
    raws: List[List[int]] = []
    for k, form in enumerate(forms):
        row = toks[k].tolist()
        if convention == "canonical":
            row = row[:len(form) + 1]            # historical readout window
        if _cut_at_eos(row, vocab.eos_id) != list(preds[k]):
            raise RuntimeError(
                "HARD STOP: recorded greedy tokens do not reproduce the historical "
                "decoder's prediction")
        raws.append(row)
    return {"preds": [list(p) for p in preds], "raw": raws, "step_logits": rec.steps,
            "live_s_hat": live, "degenerate": deg, "encoder_step_dev": step_dev,
            "n_steps": len(rec.steps)}


@torch.no_grad()
def gold_prefix_logits(model, vocab, forms, rows, injection: SemanticInjection,
                       device: str) -> torch.Tensor:
    """Teacher-forced logits under GOLD prefixes, (B, max_len+1, V), one forward of
    the isolated ventral route with the same injection.  Position t predicts the
    target token t given BOS + form[:t]."""
    from evaluate.hooks import make_batch
    b = make_batch([list(f) for f in forms], vocab, device)
    with injected(model, injection):
        injection.bind(rows)
        lg = model.route_logits(b["enc_in"], b["enc_mask"], b["dec_in"], route=ROUTE,
                                collect=False, apply_noise=False)["logits"]
    return lg


@torch.no_grad()
def prefix_corrected_continuation(model, vocab, form: Sequence[int], raw: Sequence[int],
                                  t: int, native_vector: torch.Tensor, device: str,
                                  max_steps: int = FREE_AR_MAX_STEPS) -> Dict:
    """DIAGNOSTIC_ONLY_PREFIX_CORRECTION.  Prefix = BOS + generated[:t] (== gold) +
    gold token t; then the historical greedy step (`_route_step_logits`, route ltm)
    until EOS or `max_steps` generated tokens.  Batch of one; the native vector is
    injected as a FIXED table so s_hat is exactly the one the free-AR decode saw."""
    eos = vocab.eos_id
    target = list(form) + [eos]
    gen = [int(x) for x in raw[:t]] + [target[t]]
    if gen[:t] != target[:t]:
        raise RuntimeError("HARD STOP: prefix before first divergence is not gold")
    inj = SemanticInjection("FIXED", fixed=native_vector.reshape(1, -1))
    enc_in = torch.tensor([list(form) + [eos]], dtype=torch.long, device=device)
    enc_mask = torch.ones_like(enc_in, dtype=torch.bool)
    with injected(model, inj):
        inj.bind([0])
        dec = torch.tensor([[vocab.bos_id] + gen], dtype=torch.long, device=device)
        while eos not in gen and len(gen) < max_steps:
            lg = _route_step_logits(model, enc_in, enc_mask, dec, ROUTE)
            nxt = int(lg[0, -1].argmax(-1))
            gen.append(nxt)
            dec = torch.cat([dec, torch.tensor([[nxt]], device=device)], dim=1)
    pred = _cut_at_eos(gen, eos)
    # gen ends at EOS or has max_steps (12) >= len(target) (<= 10) tokens, so the scan
    # always finds a mismatch or the full target before running out.
    second = first_divergence(gen, form, eos)
    return {
        "label": DIAGNOSTIC_LABEL,
        "corrected_predicted_phonology": " ".join(vocab.itos[p] for p in pred),
        "corrected_exact": int(pred == list(form)),
        "corrected_second_divergence_step": NA if second is None else second,
        "corrected_terminated_by_cap": int(eos not in gen),
    }


@torch.no_grad()
def native_freear_diagnostic(model, vocab, forms, rows, item_ids, out_s0: Dict,
                             native_vectors: torch.Tensor, device: str) -> List[Dict]:
    """First-divergence logit diagnostic for NATIVE (S0) GENUINE FREE-AR failures.

    `out_s0` is the `decode_condition` result for S0/freear on this batch;
    `native_vectors` the live s_hat rows it observed (the fixed native s_hat)."""
    eos = vocab.eos_id
    fixed = SemanticInjection("FIXED", fixed=native_vectors)
    tf = gold_prefix_logits(model, vocab, forms, list(range(len(forms))), fixed, device)
    records = []
    for k, (form, iid) in enumerate(zip(forms, item_ids)):
        raw = out_s0["raw"][k]
        t = first_divergence(raw, form, eos)
        if t is None:
            continue
        target = list(form) + [eos]
        gold, chosen = target[t], int(raw[t])
        lt = tf[k, t].to(torch.float64)
        ls = out_s0["step_logits"][t][k].to(torch.float64)
        topv, topi = torch.topk(lt, k=TOP_K)
        after = raw[t + 1:]
        cut = next((j for j, x in enumerate(after) if int(x) == eos), len(after))
        pred = _cut_at_eos(raw, eos)
        n_pos = max(len(pred), len(form))
        mism = sum(1 for j in range(n_pos)
                   if j >= len(pred) or j >= len(form) or pred[j] != form[j])
        rec = {
            "item_index": iid,
            "divergence_step": t,
            "divergence_is_eos_position": int(t == len(form)),
            "gold_token": vocab.itos[gold],
            "chosen_token": vocab.itos[chosen],
            "gold_rank": int((lt > lt[gold]).sum()) + 1,
            "gold_logit": float(lt[gold]),
            "chosen_logit": float(lt[chosen]),
            "margin_chosen_minus_gold": float(lt[chosen] - lt[gold]),
            "top5_tokens": " ".join(vocab.itos[int(i)] for i in topi),
            "top5_logits": " ".join(f"{float(v):.6f}" for v in topv),
            "step_vs_goldprefix_logit_max_abs_dev": float((ls - lt).abs().max()),
            "margin_numerically_ambiguous": int(abs(float(lt[chosen] - lt[gold]))
                                                <= max(float((ls - lt).abs().max()), LOGIT_PATH_TOL)),
            "n_generated_after_divergence": cut,
            "n_positional_mismatches": mism,
        }
        corr = prefix_corrected_continuation(model, vocab, form, raw, t,
                                             native_vectors[k], device)
        rec.update({f"diag_prefix_correction_{kk}": v for kk, v in corr.items()})
        records.append(rec)
    return records
