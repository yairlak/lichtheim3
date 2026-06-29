"""Top-level dual-route repetition model.

Wires the four pieces together:

    shared phoneme embedding  (general phonetic features, used by both routes)
            |                                   |
        WM buffer (dorsal)               LTM lexicon (ventral)
            |  premotor                        |  premotor + s_hat + field
            \\------------- Gate --------------/
                            |  mixed premotor
                       Motor Cortex (shared)
                            |  phoneme logits

It also exposes route-isolated outputs (`wm_logits`, `ltm_logits`) so the
evaluations can probe each stream on its own — essential for the double
dissociation and the primacy/recency curve.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from config import Config
from data.phonemes import Vocab
from .wm_route import WMBuffer
from .ltm_route import LTMLexicon
from .gating import build_gate
from .motor import MotorCortex


class DualRouteModel(nn.Module):
    def __init__(self, cfg: Config, vocab: Vocab, premotor_dim: int = 128):
        super().__init__()
        self.cfg = cfg
        self.vocab = vocab
        self.premotor_dim = premotor_dim

        # shared, general-purpose phoneme features (NOT a lexicon)
        self.phon_embed = nn.Embedding(vocab.size, cfg.ltm.phon_embed_dim,
                                       padding_idx=vocab.pad_id)

        self.wm = WMBuffer(cfg.wm, self.phon_embed, premotor_dim)
        self.ltm = LTMLexicon(cfg.ltm, self.phon_embed, cfg.data.semantic_dim,
                              premotor_dim, vocab.pad_id)
        self.gate = build_gate(cfg.gating, premotor_dim)
        self.motor = MotorCortex(premotor_dim, vocab.size)

    # ---------------------------------------------------------------- bank
    def set_semantic_bank(self, bank: torch.Tensor) -> None:
        self.ltm.set_semantic_bank(bank)

    # ---------------------------------------------------------------- forward
    def forward(self, enc_in, enc_mask, dec_in, collect: bool = False) -> Dict[str, torch.Tensor]:
        S = dec_in.shape[1]
        wm_out = self.wm(enc_in, enc_mask, n_steps=S, collect=collect)
        ltm_out = self.ltm(enc_in, enc_mask, dec_in,
                           want_field=self.gate.needs_field or collect)

        field = {k: ltm_out[k] for k in ("confidence", "margin", "density")
                 if k in ltm_out} or None
        gated = self.gate(wm_out["premotor"], ltm_out["premotor"], field)

        out = {
            "logits": self.motor(gated["premotor"]),
            "wm_logits": self.motor(wm_out["premotor"]),
            "ltm_logits": self.motor(ltm_out["premotor"]),
            "s_hat": ltm_out["s_hat"],
            "gate": gated["gate"],
        }
        if field is not None:
            out.update({f"field_{k}": v for k, v in field.items()})
        if collect:
            out["wm_diag"] = wm_out
            out["ltm_field"] = ltm_out
        return out

    # ------------------------------------------------- route-isolated logits
    def route_logits(self, enc_in, enc_mask, dec_in, route: str,
                     collect: bool = False) -> Dict[str, torch.Tensor]:
        """route in {"full", "wm", "ltm"}. Returns logits and (if collect) diag."""
        S = dec_in.shape[1]
        if route == "wm":
            wm_out = self.wm(enc_in, enc_mask, n_steps=S, collect=collect)
            res = {"logits": self.motor(wm_out["premotor"])}
            if collect:
                res["wm_diag"] = wm_out
            return res
        if route == "ltm":
            ltm_out = self.ltm(enc_in, enc_mask, dec_in, want_field=collect)
            res = {"logits": self.motor(ltm_out["premotor"]), "s_hat": ltm_out["s_hat"]}
            if collect:
                res["ltm_field"] = ltm_out
            return res
        return self.forward(enc_in, enc_mask, dec_in, collect=collect)
