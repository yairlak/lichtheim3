"""Analysis-only instrumentation of the frozen dual-route model.

**This module changes no model output.**  It reproduces the canonical decoding
path exactly and additionally exposes tensors that the frozen evaluator computed
but did not persist.  Equivalence with the canonical predictions is asserted by
`preflight_equivalence.py` and `run_instrumented.py`; any token difference is a
hard failure.

Three decoder streams, matching the M0 plan:

  A  gold-prefix pass    one `forward` over `dec_in = [BOS] + form`, giving WM,
                         LTM and FULL one-step logits at every position under the
                         **gold** prefix.  These are *gold-prefix tokenwise
                         argmax predictions* (fully teacher-forced tokenwise
                         outputs) - NOT an autoregressively generated sequence.
  B  LTM-only AR         the canonical LTM trajectory, instrumented per step.
  C  FULL-AR             the canonical FULL trajectory; at every step the WM and
                         LTM quantities are read **under the same FULL prefix**,
                         which is what makes position-level rescue analysis
                         legitimate.

No WM-only AR stream is run (M0 decision D10).

Reproduction of `DualRouteModel.forward`
----------------------------------------
`forward` (models/dual_route.py:74-99) computes, in order: `self.wm(...)`,
`self.ltm(..., want_field=True)`, `self.gate(wm_premotor, ltm_premotor, field)`,
then `self.motor(...)` on each premotor stream.  The functions below call those
same submodules in that same order, so the arithmetic is identical - they simply
also return the intermediate premotor tensors.

Metric conventions (frozen here, documented in the run manifest)
---------------------------------------------------------------
* entropy       : natural log (nats), softmax over **all 42 vocabulary logits**
                  exactly as the model emits them (PAD, BOS, EOS included).  A
                  second column restricts the distribution to the 39 phoneme
                  entries, renormalised.
* target rank   : competition ranking, `1 + #{j : logit_j > logit_target}`.
                  Ties do not inflate the rank.
* target margin : `logit_target - max_{j != target} logit_j`.  Signed: positive
                  means the target is the argmax.
* BOS           : never a target and never scored.
* EOS           : a legitimate target at the final gold position (`dec_tgt =
                  form + [EOS]`).
* PAD           : never a target; padded positions are dropped by target length.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluate.hooks import make_batch                              # noqa: E402
from models.dual_route import DualRouteModel                       # noqa: E402

BATCH_SIZE = 64          # identical to external_eval.BATCH_SIZE
DEFAULT_TOP_K = 20

ENTROPY_LOG_BASE = "natural (nats)"
ENTROPY_VOCAB = "all 42 logits as emitted; phoneme-only variant renormalises the 39 non-special entries"
TARGET_RANK_TIE_POLICY = "competition ranking: 1 + #{j : logit_j > logit_target}; ties do not inflate rank"
SPECIAL_NOTE = "BOS never a target; EOS is the target at the final gold position; PAD never scored"


# --------------------------------------------------------------- metrics

def _entropy(logits: torch.Tensor, keep: Optional[torch.Tensor] = None
             ) -> torch.Tensor:
    """Shannon entropy in nats of softmax(logits) along the last axis."""
    if keep is not None:
        logits = logits[..., keep]
    logp = torch.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(-1)


def _rank_margin(logits: torch.Tensor, target: torch.Tensor):
    """Competition rank of the target and its signed margin.

    logits: (..., V); target: (...) long.  Returns (rank, target_logit,
    best_non_target_logit, margin).
    """
    tgt_logit = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    rank = 1 + (logits > tgt_logit.unsqueeze(-1)).sum(-1)
    masked = logits.clone()
    masked.scatter_(-1, target.unsqueeze(-1), float("-inf"))
    best_non_target = masked.max(-1).values
    return rank, tgt_logit, best_non_target, tgt_logit - best_non_target


# ------------------------------------------------ exact forward reproduction

@torch.no_grad()
def _route_states(model: DualRouteModel, enc_in, enc_mask, dec_in):
    """Reproduce DualRouteModel.forward while exposing premotor states.

    Identical order of operations to models/dual_route.py:77-93.
    """
    wm_out = model.wm(enc_in, enc_mask, dec_in)
    ltm_out = model.ltm(enc_in, enc_mask, dec_in, want_field=True)
    field = {k: ltm_out[k] for k in ("confidence", "margin", "density")
             if k in ltm_out} or None
    gated = model.gate(wm_out["premotor"], ltm_out["premotor"], field)
    return {
        "wm_premotor": wm_out["premotor"],
        "ltm_premotor": ltm_out["premotor"],
        "full_premotor": gated["premotor"],
        "gate": gated["gate"],
        "s_hat": ltm_out["s_hat"],
        "field": field,
        "wm_logits": model.motor(wm_out["premotor"]),
        "ltm_logits": model.motor(ltm_out["premotor"]),
        "full_logits": model.motor(gated["premotor"]),
    }


# ------------------------------------------------------- encoder extraction

@torch.no_grad()
def encoder_quantities(model: DualRouteModel, batch, top_k: int = DEFAULT_TOP_K):
    """WM/LTM encoder states, s_hat, decoder h0, field and bank top-k."""
    h_wm = model.wm.encode(batch["enc_in"], batch["enc_mask"])       # (1,B,H)
    s_hat = model.ltm.encode(batch["enc_in"], batch["enc_mask"])     # (B,300)
    h0 = torch.tanh(model.ltm.sem_to_h0(s_hat))                      # (B,dec_hidden)
    field = model.ltm.lexical_field(s_hat)
    sims = field["sims"]                                             # (B,n_words)
    k = min(top_k, sims.shape[1])
    # Deterministic ordering: sort by (-similarity, row index) so equal cosines
    # are broken by ascending bank row, per the frozen tie policy.
    top = torch.topk(sims, k=k, dim=1, largest=True, sorted=True)
    top_vals, top_idx = top.values, top.indices
    for b in range(sims.shape[0]):
        order = sorted(range(k), key=lambda j: (-float(top_vals[b, j]),
                                                int(top_idx[b, j])))
        top_vals[b] = top_vals[b][order]
        top_idx[b] = top_idx[b][order]
    conf = field["confidence"]
    g = torch.sigmoid(model.gate.cfg.alpha * (conf - model.gate.cfg.gate_threshold))
    return {"h_wm": h_wm.squeeze(0), "s_hat": s_hat, "ltm_h0": h0,
            "confidence": conf, "margin": field["margin"],
            "density": field["density"], "gate": g,
            "topk_idx": top_idx, "topk_sim": top_vals}


# ------------------------------------------------------- Stream A: gold prefix

@torch.no_grad()
def gold_prefix_pass(model: DualRouteModel, vocab, forms: List[List[int]],
                     device: str):
    """One decoder execution over the gold prefix.

    Returns per-item lists of per-position records.  `dec_in = [BOS] + form`
    and `dec_tgt = form + [EOS]`, so position p predicts `dec_tgt[p]` having
    seen the gold tokens up to p-1.
    """
    batch = make_batch(forms, vocab, device)
    st = _route_states(model, batch["enc_in"], batch["enc_mask"], batch["dec_in"])
    tgt = batch["dec_tgt"]
    keep = torch.tensor([i for i in range(vocab.size)
                         if i not in (vocab.pad_id, vocab.bos_id, vocab.eos_id)],
                        device=tgt.device)
    out = []
    for i, form in enumerate(forms):
        n = len(form) + 1                       # form positions + final EOS
        rec = []
        for p in range(n):
            row = {"timestep": p, "target_token_id": int(tgt[i, p])}
            for route, lg in (("wm", st["wm_logits"]), ("ltm", st["ltm_logits"]),
                              ("full", st["full_logits"])):
                l = lg[i, p]
                r, tl, bn, mg = _rank_margin(l, tgt[i, p])
                row[f"{route}_pred_id"] = int(l.argmax())
                row[f"{route}_target_rank"] = int(r)
                row[f"{route}_target_logit"] = float(tl)
                row[f"{route}_best_non_target_logit"] = float(bn)
                row[f"{route}_target_margin"] = float(mg)
                row[f"{route}_entropy"] = float(_entropy(l))
                row[f"{route}_entropy_phon"] = float(_entropy(l, keep))
            row["gate"] = float(st["gate"][i, p, 0])
            rec.append(row)
        out.append(rec)
    return out, st


# ------------------------------------------------------------ AR streams

@torch.no_grad()
def ar_stream(model: DualRouteModel, vocab, forms: List[List[int]], device: str,
              route: str, capture_all_routes: bool = False):
    """Canonical AR decoding for one route, instrumented per timestep.

    Byte-for-byte the canonical loop of `external_eval.autoregressive_decode_batch`:
    `dec_input` starts at BOS, `max_steps` is the BATCH maximum target length,
    one argmax is appended per step, and the readout window for item i is
    `dec_input[i, 1:len(form_i)+1]`.

    capture_all_routes=True additionally records the WM and LTM quantities under
    the **same** (FULL-generated) prefix - required for position-level rescue.
    """
    batch = make_batch(forms, vocab, device)
    max_steps = max(len(f) for f in forms)
    dec_input = batch["enc_in"].new_full((len(forms), 1), vocab.bos_id)
    keep = torch.tensor([i for i in range(vocab.size)
                         if i not in (vocab.pad_id, vocab.bos_id, vocab.eos_id)],
                        device=dec_input.device)
    steps: List[List[dict]] = [[] for _ in forms]
    prem_wm, prem_ltm = [], []

    for t in range(max_steps):
        if route == "full" or capture_all_routes:
            st = _route_states(model, batch["enc_in"], batch["enc_mask"], dec_input)
            sel_logits = st["full_logits"] if route == "full" else st["ltm_logits"]
        else:
            ltm_out = model.ltm(batch["enc_in"], batch["enc_mask"], dec_input,
                                want_field=False)
            st = {"ltm_premotor": ltm_out["premotor"],
                  "ltm_logits": model.motor(ltm_out["premotor"])}
            sel_logits = st["ltm_logits"]
        last = sel_logits[:, -1, :]
        next_tok = last.argmax(-1, keepdim=True)

        for i, form in enumerate(forms):
            if t >= len(form):
                continue
            tgt_id = form[t] if t < len(form) else vocab.eos_id
            row = {"timestep": t, "target_token_id": int(tgt_id),
                   "predicted_token_id": int(next_tok[i, 0]),
                   "prefix_len": int(dec_input.shape[1])}
            names = (("wm", "wm_logits"), ("ltm", "ltm_logits"),
                     ("full", "full_logits")) if (route == "full" or capture_all_routes) \
                else (("ltm", "ltm_logits"),)
            for nm, key in names:
                l = st[key][i, -1]
                tt = torch.tensor(tgt_id, device=l.device)
                r, tl, bn, mg = _rank_margin(l, tt)
                row[f"{nm}_pred_id"] = int(l.argmax())
                row[f"{nm}_target_rank"] = int(r)
                row[f"{nm}_target_logit"] = float(tl)
                row[f"{nm}_best_non_target_logit"] = float(bn)
                row[f"{nm}_target_margin"] = float(mg)
                row[f"{nm}_entropy"] = float(_entropy(l))
                row[f"{nm}_entropy_phon"] = float(_entropy(l, keep))
            if "gate" in st:
                row["gate"] = float(st["gate"][i, -1, 0])
            row["is_eos_prediction"] = int(int(next_tok[i, 0]) == vocab.eos_id)
            steps[i].append(row)

        if t == 0 and ("wm_premotor" in st or "ltm_premotor" in st):
            prem_wm = st.get("wm_premotor")
            prem_ltm = st.get("ltm_premotor")
        dec_input = torch.cat([dec_input, next_tok], dim=1)

    # canonical readout: window, raw EOS position, then trim
    preds, eos_pos = [], []
    for i, form in enumerate(forms):
        n_steps = len(form)
        pred_ids = dec_input[i, 1:n_steps + 1].tolist()
        e = next((j for j, x in enumerate(pred_ids) if x == vocab.eos_id), None)
        eos_pos.append(e)
        seq = []
        for idx in pred_ids:
            if idx == vocab.eos_id:
                break
            seq.append(idx)
        preds.append(seq)
    return {"preds": preds, "eos_position": eos_pos, "steps": steps,
            "wm_premotor_t0": prem_wm, "ltm_premotor_t0": prem_ltm}
