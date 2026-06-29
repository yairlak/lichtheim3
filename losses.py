"""L_total assembly.

The total loss is a weighted sum of five interpretable terms. Each term has a
job; the weights live in `config.LossConfig`.

    L_total = lambda_rep  * L_rep      repetition CE on the MOTOR output (the task)
            + lambda_align * L_align    encoded meaning -> GloVe (ventral identity)
            + lambda_dec   * L_dec      ventral form regeneration CE (keeps the
                                        meaning->form decoder honest)
            + lambda_wm    * L_wm       WM-ONLY repetition CE (forces the buffer to
                                        actually carry forms, so the gate has a real
                                        competitor and the dissociation is meaningful)
            + lambda_gate  * L_gate     soft prior on average route usage

Notes on balancing
------------------
* `L_align` uses (1 - cosine) + a small MSE. Cosine fixes the *direction* (word
  identity in GloVe space); the MSE term keeps magnitudes sane so the decoder's
  `sem_to_h0` sees vectors on the GloVe scale.
* `lambda_wm` is the subtle one. Without it the optimiser would route everything
  through the powerful ventral decoder and the buffer would never learn to read
  itself out — then there is no dorsal route to dissociate. Keeping a dedicated
  WM-only CE guarantees both routes are independently competent.
* `lambda_gate` is a *weak* prior (default 0.05). It nudges the mean gate toward
  `usage_prior` so a hand-designed gate (hypotheses 1-2) is not overwhelmed and a
  learned gate (hypothesis 3) does not collapse to one route on day one.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from config import LossConfig


def _seq_ce(logits: torch.Tensor, target: torch.Tensor, pad_id: int,
            label_smoothing: float = 0.0) -> torch.Tensor:
    V = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, V), target.reshape(-1),
        ignore_index=pad_id, label_smoothing=label_smoothing,
    )


def alignment_loss(s_hat: torch.Tensor, target_sem: torch.Tensor) -> torch.Tensor:
    cos = 1.0 - F.cosine_similarity(s_hat, target_sem, dim=-1).mean()
    mse = F.mse_loss(s_hat, target_sem)
    return cos + 0.1 * mse


def gate_regularizer(gate: torch.Tensor, usage_prior: float) -> torch.Tensor:
    """Pull the mean gate toward the usage prior (weak)."""
    return (gate.mean() - usage_prior) ** 2


def total_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
               cfg: LossConfig, pad_id: int,
               usage_prior: float = 0.5) -> Dict[str, torch.Tensor]:
    tgt = batch["dec_tgt"]
    L_rep = _seq_ce(out["logits"], tgt, pad_id, cfg.label_smoothing)
    L_wm = _seq_ce(out["wm_logits"], tgt, pad_id)
    L_dec = _seq_ce(out["ltm_logits"], tgt, pad_id)
    L_align = alignment_loss(out["s_hat"], batch["semantic"])
    L_gate = gate_regularizer(out["gate"], usage_prior)

    total = (cfg.rep * L_rep + cfg.align * L_align + cfg.dec * L_dec
             + cfg.wm * L_wm + cfg.gate * L_gate)
    return {
        "total": total, "rep": L_rep, "align": L_align,
        "dec": L_dec, "wm": L_wm, "gate": L_gate,
    }
