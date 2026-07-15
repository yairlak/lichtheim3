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

Encode-once / decode-stepwise interface
----------------------------------------
`encode_all(enc_in, enc_mask, collect, apply_noise)` encodes both routes once
and optionally draws route noises once per stimulus. Returns {h_WM, s_hat, field}.

`decode_from_states(h_WM, s_hat, field, dec_in)` decodes one prefix (any length)
from the pre-computed states.  Used in the scheduled-sampling training path
(_forward_scheduled_sampling in train.py) to avoid re-encoding at every step.

The standard `forward(enc_in, enc_mask, dec_in)` method is unchanged and is still
used for TF=1.0 training (vectorised path) and all evaluation code.

Noise semantics
---------------
apply_noise=False (default): noise is controlled entirely by self.training.
apply_noise=True : noise is active even in eval mode (lesion / stress tests).
collect=True alone: collects intermediate representations WITHOUT activating noise.
See wm_route.py and ltm_route.py for the definitive noise condition.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from config import Config
from data.phonemes import Vocab
from .wm_route import WMRecurrent
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

        self.wm = WMRecurrent(cfg.wm, self.phon_embed, premotor_dim)
        self.ltm = LTMLexicon(cfg.ltm, self.phon_embed, cfg.data.semantic_dim,
                              premotor_dim, vocab.pad_id)
        self.gate = build_gate(cfg.gating, premotor_dim)
        self.motor = MotorCortex(premotor_dim, vocab.size)

    # ---------------------------------------------------------------- bank
    def set_semantic_bank(self, bank: torch.Tensor) -> None:
        self.ltm.set_semantic_bank(bank)

    # ---------------------------------------------------------------- forward
    def forward(self, enc_in, enc_mask, dec_in,
                collect: bool = False,
                apply_noise: bool = False) -> Dict[str, torch.Tensor]:
        wm_out = self.wm(enc_in, enc_mask, dec_in,
                         collect=collect, apply_noise=apply_noise)
        ltm_out = self.ltm(enc_in, enc_mask, dec_in,
                           want_field=self.gate.needs_field or collect,
                           collect=collect, apply_noise=apply_noise)

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

    # ----------------------------------------------- encode-once interface
    def encode_all(self, enc_in: torch.Tensor, enc_mask: torch.Tensor,
                   collect: bool = False,
                   apply_noise: bool = False) -> Dict[str, object]:
        """Encode both routes once.  Draws all route noises once per stimulus.

        Args:
            collect:     if True, signal upstream callers to collect diagnostics
            apply_noise: if True, apply route noises even in eval mode (lesion tests)

        Returns:
            dict with keys:
                "h_WM"  : (1, B, wm_hidden)  — WM encoder state
                "s_hat" : (B, semantic_dim)   — LTM semantic vector
                "field" : dict or None         — lexical field for gate
        """
        h_WM = self.wm.encode(enc_in, enc_mask, collect=collect, apply_noise=apply_noise)
        s_hat = self.ltm.encode(enc_in, enc_mask, collect=collect, apply_noise=apply_noise)

        field = None
        if self.gate.needs_field and self.ltm.semantic_bank.shape[0] > 1:
            field_full = self.ltm.lexical_field(s_hat)
            field = {k: field_full[k] for k in ("confidence", "margin", "density")
                     if k in field_full} or None

        return {"h_WM": h_WM, "s_hat": s_hat, "field": field}

    def decode_from_states(self, h_WM: torch.Tensor, s_hat: torch.Tensor,
                           field: Optional[Dict[str, torch.Tensor]],
                           dec_in: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode both routes from pre-computed states.

        Args:
            h_WM:   (1, B, wm_hidden)    WM encoder state from encode_all
            s_hat:  (B, semantic_dim)     LTM semantic vector from encode_all
            field:  dict or None          lexical field from encode_all
            dec_in: (B, S) current decoder prefix IDs

        Returns dict matching forward() structure (logits, wm_logits, ltm_logits,
        gate, s_hat, field_*).
        """
        wm_premotor = self.wm.decode_from_state(h_WM, dec_in)["premotor"]
        ltm_premotor = self.ltm.decode_from_s_hat(s_hat, dec_in)
        gated = self.gate(wm_premotor, ltm_premotor, field)
        out = {
            "logits":     self.motor(gated["premotor"]),
            "wm_logits":  self.motor(wm_premotor),
            "ltm_logits": self.motor(ltm_premotor),
            "s_hat":      s_hat,
            "gate":       gated["gate"],
        }
        if field is not None:
            out.update({f"field_{k}": v for k, v in field.items()})
        return out

    # ------------------------------------------------- route-isolated logits
    def route_logits(self, enc_in, enc_mask, dec_in, route: str,
                     collect: bool = False,
                     apply_noise: bool = False) -> Dict[str, torch.Tensor]:
        """route in {"full", "wm", "ltm"}. Returns logits and (if collect) diag.

        apply_noise=True activates route noise in eval mode (lesion / stress tests).
        collect=True collects intermediate representations WITHOUT activating noise.
        """
        if route == "wm":
            wm_out = self.wm(enc_in, enc_mask, dec_in,
                             collect=collect, apply_noise=apply_noise)
            res = {"logits": self.motor(wm_out["premotor"])}
            if collect:
                res["wm_diag"] = wm_out
            return res
        if route == "ltm":
            ltm_out = self.ltm(enc_in, enc_mask, dec_in,
                               want_field=collect, collect=collect,
                               apply_noise=apply_noise)
            res = {"logits": self.motor(ltm_out["premotor"]), "s_hat": ltm_out["s_hat"]}
            if collect:
                res["ltm_field"] = ltm_out
            return res
        return self.forward(enc_in, enc_mask, dec_in,
                            collect=collect, apply_noise=apply_noise)
