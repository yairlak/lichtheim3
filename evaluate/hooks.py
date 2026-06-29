"""Shared inference helpers for the evaluation scripts.

The model's own `collect=True` path already returns rich diagnostics (WM trace
strengths and read attention, LTM lexical confidence/density), so we don't need
generic forward hooks here. These helpers just build eval batches, run a route's
greedy predictions, and score per-position correctness, so the evaluations don't
duplicate boilerplate.
"""
from __future__ import annotations

from typing import Dict, List

import torch


def make_batch(forms: List[List[int]], vocab, device) -> Dict[str, torch.Tensor]:
    """Build a padded eval batch (enc_in/enc_mask/dec_in/dec_tgt) from raw forms.

    `forms` are phoneme-id lists WITHOUT specials. dec_in uses teacher forcing
    with the gold form (we measure per-position reconstruction, not free
    generation, which is the right probe for serial-position fidelity).
    """
    B = len(forms)
    max_enc = max(len(f) for f in forms) + 1
    max_dec = max(len(f) for f in forms) + 1
    enc_in = torch.full((B, max_enc), vocab.pad_id, dtype=torch.long)
    enc_mask = torch.zeros((B, max_enc), dtype=torch.bool)
    dec_in = torch.full((B, max_dec), vocab.pad_id, dtype=torch.long)
    dec_tgt = torch.full((B, max_dec), vocab.pad_id, dtype=torch.long)
    for i, f in enumerate(forms):
        ei = f + [vocab.eos_id]
        di = [vocab.bos_id] + f
        dt = f + [vocab.eos_id]
        enc_in[i, :len(ei)] = torch.tensor(ei)
        enc_mask[i, :len(ei)] = True
        dec_in[i, :len(di)] = torch.tensor(di)
        dec_tgt[i, :len(dt)] = torch.tensor(dt)
    return {k: v.to(device) for k, v in dict(
        enc_in=enc_in, enc_mask=enc_mask, dec_in=dec_in, dec_tgt=dec_tgt).items()}


@torch.no_grad()
def route_predictions(model, batch, route: str, collect: bool = False):
    """Greedy argmax predictions for a route. Returns (preds, extra)."""
    res = model.route_logits(batch["enc_in"], batch["enc_mask"], batch["dec_in"],
                             route=route, collect=collect)
    preds = res["logits"].argmax(-1)            # (B, S)
    return preds, res


def per_position_correct(preds: torch.Tensor, target: torch.Tensor,
                         pad_id: int) -> torch.Tensor:
    """(B, S) float: 1 where correct phoneme, NaN on padding."""
    correct = (preds == target).float()
    correct[target == pad_id] = float("nan")
    return correct
