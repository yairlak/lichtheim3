"""Item-level gate / route-competence probe.

Design constraints, all traceable to `CODE_AUDIT_GATE.md`:

* **No model modification.**  `models/gating.py` is untouched.  The FIXED05
  intervention is a post-hoc recombination of `wm_logits` and `ltm_logits`,
  which is *exactly* the forced-gate result because `motor` is a single affine
  map and the blend weights sum to 1 (audit §3, CLAIM 1).
* **Decoding conventions are inherited, not invented.**  The forced-length AR
  loop mirrors `scripts.evaluate_train_lexicon_ceiling._ar_decode_batch`; the
  free-AR loop mirrors
  `scripts.naming_comprehension.train_joint_scratch.Trainer.free_ar_repetition`.
  Both are reproduced here rather than imported only because they hard-code the
  route set and cannot emit a fourth, intervention route.
* **Each route decodes on its own prefix.**  Under autoregressive decoding the
  trajectories diverge, so FIXED05 must be re-decoded and can never be
  reconstructed from stored per-route predictions (audit §4, caveat 3).
* **The gate is word-level.**  One scalar per item, constant across decoder
  steps, so a single BOS-only forward captures the value used at every step
  (audit §6, CLAIM 4; same argument as `scripts/external_eval.py`).

`forced_gate` is ported from `lichtheim3-brain-damage@a5c787a:lesion/gate_diagnostic.py`
(EXPLORATORY status).  It is used ONLY as an independent cross-check of
`fixed_mix_logits`; no result is reported through it.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

# The native gate, the two isolated routes, and the preregistered intervention.
# "fixed05" is the ONLY intervention in this workstream: no other ratio is
# searched (workstream brief §7).
ROUTES: Tuple[str, ...] = ("full", "wm", "ltm", "fixed05")

FIXED_G = 0.5


# ---------------------------------------------------------------- categories

class CompetenceCategory:
    BOTH = "BOTH_CORRECT"
    WM_ONLY = "WM_ONLY_CORRECT"
    LTM_ONLY = "LTM_ONLY_CORRECT"
    NEITHER = "NEITHER_CORRECT"
    ALL = (BOTH, WM_ONLY, LTM_ONLY, NEITHER)


def competence_category(wm_correct: int, ltm_correct: int) -> str:
    """Route-competence cell for one item.  Defined on the ISOLATED routes."""
    if wm_correct and ltm_correct:
        return CompetenceCategory.BOTH
    if wm_correct:
        return CompetenceCategory.WM_ONLY
    if ltm_correct:
        return CompetenceCategory.LTM_ONLY
    return CompetenceCategory.NEITHER


# ------------------------------------------------------------ interventions

def fixed_mix_logits(out: Dict[str, torch.Tensor], g: float = FIXED_G) -> torch.Tensor:
    """Logits of a forced constant gate, computed post hoc from one forward.

    Exact, not approximate:  motor() is a single nn.Linear and g + (1-g) = 1, so
        motor(g*ltm + (1-g)*wm) == g*motor(ltm) + (1-g)*motor(wm)
    identically (audit §3).  `out` must come from a route="full" forward, which
    returns `wm_logits` and `ltm_logits` alongside `logits`.
    """
    return g * out["ltm_logits"] + (1.0 - g) * out["wm_logits"]


@contextmanager
def forced_gate(model: torch.nn.Module, g: Optional[float]):
    """Force the gate to a constant ventral weight `g` for the duration of the block.

    Ported unchanged in substance from
    `lichtheim3-brain-damage@a5c787a:lesion/gate_diagnostic.py` (EXPLORATORY).
    `g=None` is a no-op.  No parameter is touched; the hook is removed on exit,
    including on exception.

    Used here ONLY to cross-check `fixed_mix_logits` through a second, independent
    code path.  Reported results never go through this hook.
    """
    if g is None:
        yield model
        return

    gv = float(g)

    def hook(_module, inputs, output):
        wm, ltm = inputs[0], inputs[1]
        gt = torch.full_like(output["gate"], gv)
        return {"premotor": gt * ltm + (1.0 - gt) * wm, "gate": gt}

    handle = model.gate.register_forward_hook(hook)
    try:
        yield model
    finally:
        handle.remove()


# --------------------------------------------------------------- gate capture

@torch.no_grad()
def capture_gate_field(model, enc_in: torch.Tensor, enc_mask: torch.Tensor,
                       bos_id: int) -> Dict[str, List[Optional[float]]]:
    """Word-level gate and lexical field for one batch.

    One BOS-only forward.  The gate is constant across decoder steps
    (audit CLAIM 4), so this is exactly the value used at every step of every
    decode below.  `margin` and `density` are recorded as DESCRIPTIVE columns:
    the gate does not read them (audit §9) and no variant using them is built.
    """
    B = enc_in.shape[0]
    dec_bos = enc_in.new_full((B, 1), bos_id)
    res = model.route_logits(enc_in, enc_mask, dec_bos, route="full", collect=False)

    gate_t = res.get("gate")
    gate = ([float(v) for v in gate_t[:, 0, 0].tolist()]
            if gate_t is not None else [None] * B)

    def _field(key: str) -> List[Optional[float]]:
        t = res.get(f"field_{key}")
        return [None] * B if t is None else [float(v) for v in t.reshape(B).tolist()]

    return {
        "gate": gate,
        "effective_ventral_weight": gate,                       # == g
        "effective_dorsal_weight": [None if v is None else 1.0 - v for v in gate],
        "lexical_confidence": _field("confidence"),
        "lexical_margin": _field("margin"),
        "lexical_density": _field("density"),
    }


# ------------------------------------------------------------------ decoding

def _route_step_logits(model, enc_in, enc_mask, dec_in, route: str) -> torch.Tensor:
    """Logits for one decoder prefix under `route`.

    "wm"/"ltm" use the isolated code path, which bypasses the gate entirely and
    is bit-identical to the g=0 / g=1 limits (audit CLAIM 2).  "full" and
    "fixed05" share a single gated forward.
    """
    if route in ("wm", "ltm"):
        return model.route_logits(enc_in, enc_mask, dec_in, route=route,
                                  collect=False, apply_noise=False)["logits"]
    out = model.route_logits(enc_in, enc_mask, dec_in, route="full",
                             collect=False, apply_noise=False)
    return out["logits"] if route == "full" else fixed_mix_logits(out, FIXED_G)


@torch.no_grad()
def ar_decode_forced_length(model, vocab, forms: Sequence[Sequence[int]],
                            device: str, routes: Sequence[str] = ROUTES,
                            ) -> Dict[str, List[List[int]]]:
    """CANONICAL forced-length autoregressive decode.

    Mirrors `scripts.evaluate_train_lexicon_ceiling._ar_decode_batch`: greedy to
    the batch maximum, each item's readout truncated to its own gold length + 1,
    cut at the first EOS.  This is the convention behind every historical
    `rep_canonical_*` number, so results are directly comparable.
    """
    from evaluate.hooks import make_batch

    batch = make_batch([list(f) for f in forms], vocab, device)
    max_steps = max(len(f) for f in forms) + 1
    preds: Dict[str, List[List[int]]] = {r: [] for r in routes}

    for route in routes:
        dec_in = batch["enc_in"].new_full((len(forms), 1), vocab.bos_id)
        for _ in range(max_steps):
            lg = _route_step_logits(model, batch["enc_in"], batch["enc_mask"],
                                    dec_in, route)
            dec_in = torch.cat([dec_in, lg[:, -1, :].argmax(-1, keepdim=True)], dim=1)
        for i, form in enumerate(forms):
            raw = dec_in[i, 1: 1 + len(form) + 1].tolist()
            seq: List[int] = []
            for tok in raw:
                if tok == vocab.eos_id:
                    break
                seq.append(tok)
            preds[route].append(seq)
    return preds


@torch.no_grad()
def ar_decode_free(model, vocab, forms: Sequence[Sequence[int]], device: str,
                   routes: Sequence[str] = ROUTES,
                   max_steps: int = 24) -> Dict[str, List[List[int]]]:
    """GENUINE free-AR decode: one global cap, no use of the target length.

    Mirrors `Trainer.free_ar_repetition`.  Over-generation and non-termination
    count as errors, which the forced-length convention cannot see.  Reported
    ALONGSIDE the canonical metric; neither replaces the other.
    """
    n = len(forms)
    max_enc = max(len(f) for f in forms) + 1
    enc_in = torch.full((n, max_enc), vocab.pad_id, dtype=torch.long)
    enc_mask = torch.zeros((n, max_enc), dtype=torch.bool)
    for k, f in enumerate(forms):
        enc_in[k, :len(f) + 1] = torch.tensor(list(f) + [vocab.eos_id])
        enc_mask[k, :len(f) + 1] = True
    enc_in, enc_mask = enc_in.to(device), enc_mask.to(device)

    preds: Dict[str, List[List[int]]] = {}
    for route in routes:
        dec = torch.full((n, 1), vocab.bos_id, dtype=torch.long, device=device)
        for _ in range(max_steps):
            lg = _route_step_logits(model, enc_in, enc_mask, dec, route)
            dec = torch.cat([dec, lg[:, -1, :].argmax(-1, keepdim=True)], dim=1)
            if bool((dec == vocab.eos_id).any(dim=1).all()):
                break
        out = []
        for k in range(n):
            seq = dec[k, 1:].tolist()
            if vocab.eos_id in seq:
                seq = seq[:seq.index(vocab.eos_id)]
            out.append(seq)
        preds[route] = out
    return preds


# ------------------------------------------------------------- item-level pass

@torch.no_grad()
def collect_item_level(model, vocab, entries, indices: Sequence[int], device: str,
                       batch_size: int = 256, free_ar: bool = True,
                       progress: Optional[Callable[[int, int], None]] = None,
                       ) -> List[dict]:
    """One pass producing the Experiment-1 + Experiment-2 item-level table.

    Returns one row per item with: identity and metadata (word, rank, length,
    zipf frequency — aligned by construction, never joined), the word-level gate
    and lexical field, per-route exact-match under BOTH decoding conventions,
    and the route-competence category.

    Experiment 1 reads the gate against the competence categories.
    Experiment 2 reads `fixed05_*` against `full_*`, paired item by item.
    Both come from this single table, so the two experiments are guaranteed to
    be evaluated on identical items in identical states.
    """
    was_training = model.training
    model.eval()
    rows: List[dict] = []
    total = len(indices)

    try:
        for lo in range(0, total, batch_size):
            idx = list(indices[lo:lo + batch_size])
            items = [entries[i] for i in idx]
            forms = [e.phonemes for e in items]

            fl = ar_decode_forced_length(model, vocab, forms, device)
            fa = ar_decode_free(model, vocab, forms, device) if free_ar else None

            from evaluate.hooks import make_batch
            b = make_batch([list(f) for f in forms], vocab, device)
            field = capture_gate_field(model, b["enc_in"], b["enc_mask"], vocab.bos_id)

            for j, (i, e, form) in enumerate(zip(idx, items, forms)):
                tgt = list(form)
                row = {
                    "item_index": int(i),
                    "word": e.word,
                    "rank": int(getattr(e, "rank", 0)),
                    "target_phonemes": " ".join(vocab.itos[p] for p in tgt),
                    "length": len(tgt),
                    "zipf_approx": round(float(e.freq), 6),
                    "is_word": 1,          # primary population is all real words
                    "gate": field["gate"][j],
                    "effective_ventral_weight": field["effective_ventral_weight"][j],
                    "effective_dorsal_weight": field["effective_dorsal_weight"][j],
                    "lexical_confidence": field["lexical_confidence"][j],
                    "lexical_margin": field["lexical_margin"][j],
                    "lexical_density": field["lexical_density"][j],
                }
                for conv, preds in (("canonical", fl), ("freear", fa)):
                    if preds is None:
                        continue
                    for r in ROUTES:
                        p = preds[r][j]
                        row[f"{conv}_{r}_exact"] = int(p == tgt)
                        row[f"{conv}_{r}_predicted"] = " ".join(vocab.itos[q] for q in p)
                row["competence_category"] = competence_category(
                    row["canonical_wm_exact"], row["canonical_ltm_exact"])
                if fa is not None:
                    row["competence_category_freear"] = competence_category(
                        row["freear_wm_exact"], row["freear_ltm_exact"])
                rows.append(row)

            if progress is not None:
                progress(min(lo + batch_size, total), total)
    finally:
        model.train(was_training)
    return rows
