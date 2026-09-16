"""Item-level pass under a lesion context.

Mirrors `gating_diagnostics.gate_probe.collect_item_level` — same decoders, same routes,
same gate capture, same column names — with exactly two additions:

* the lesion hook is bound to the batch's item ids, in order, before any decode;
* the GXLR columns are emitted (paired NATIVE/FIXED05 discordance, errors
  prevented/introduced, c_LTM and g and their deltas from intact, compact discordant
  traces).

The decoders themselves are IMPORTED, never reimplemented: `ar_decode_forced_length`
keeps the canonical forced-length convention behind every historical `rep_canonical_*`
number, and `ar_decode_free` keeps the genuine free-AR cap, which it imports from
`train_joint_scratch` rather than restating.

`ROUTES` is `("full", "wm", "ltm", "fixed05")`: `full` is NATIVE, `fixed05` is the
preregistered intervention, `wm`/`ltm` are the isolated route competences.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import torch

from gating_diagnostics.gate_probe import (ROUTES, ar_decode_forced_length,
                                           ar_decode_free, capture_gate_field,
                                           competence_category)

#: How many tokens of a discordant prediction to keep in the compact trace.
TRACE_MAX_TOKENS = 12


def _trace(seq: str) -> str:
    toks = seq.split()
    return " ".join(toks[:TRACE_MAX_TOKENS]) + ("…" if len(toks) > TRACE_MAX_TOKENS else "")


@torch.no_grad()
def collect_item_level_lesioned(model, vocab, entries, indices: Sequence[int],
                                device: str, *, hook=None, batch_size: int = 256,
                                free_ar: bool = True,
                                intact_by_item: Optional[Dict[int, dict]] = None,
                                progress: Optional[Callable[[int, int], None]] = None,
                                ) -> List[dict]:
    """One pass over `indices`.  `hook=None` is the intact control.

    `intact_by_item` supplies the unperturbed reference row per item index, used only to
    compute deltas and change counts; it never influences decoding.
    """
    from evaluate.hooks import make_batch

    was_training = model.training
    model.eval()
    rows: List[dict] = []
    total = len(indices)

    try:
        for lo in range(0, total, batch_size):
            idx = list(indices[lo:lo + batch_size])
            items = [entries[i] for i in idx]
            forms = [e.phonemes for e in items]
            words = [e.word for e in items]

            # Bind BEFORE any forward.  Item identity is stated, never inferred.
            if hook is not None:
                hook.bind(words)

            fl = ar_decode_forced_length(model, vocab, forms, device)
            fa = ar_decode_free(model, vocab, forms, device) if free_ar else None

            b = make_batch([list(f) for f in forms], vocab, device)
            field = capture_gate_field(model, b["enc_in"], b["enc_mask"], vocab.bos_id)
            dec_width = int(b["dec_in"].shape[1])

            for j, (i, e, form) in enumerate(zip(idx, items, forms)):
                tgt = list(form)
                row = {
                    "item_index": int(i),
                    "word": e.word,
                    "batch_dec_width": dec_width,
                    "rank": int(getattr(e, "rank", 0)),
                    "target_phonemes": " ".join(vocab.itos[p] for p in tgt),
                    "length": len(tgt),
                    "zipf_approx": round(float(e.freq), 6),
                    "gate": field["gate"][j],
                    "c_LTM": field["lexical_confidence"][j],
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

                    nat_p, fix_p = row[f"{conv}_full_predicted"], row[f"{conv}_fixed05_predicted"]
                    nat_e, fix_e = row[f"{conv}_full_exact"], row[f"{conv}_fixed05_exact"]
                    row[f"{conv}_discordant_prediction"] = int(nat_p != fix_p)
                    row[f"{conv}_discordant_exact"] = int(nat_e != fix_e)
                    # Orientation matches gating_diagnostics.analysis.mcnemar:
                    # a = NATIVE, b = FIXED05.
                    row[f"{conv}_error_introduced_by_fixed05"] = int(nat_e == 1 and fix_e == 0)
                    row[f"{conv}_error_prevented_by_fixed05"] = int(nat_e == 0 and fix_e == 1)
                    row[f"{conv}_trace_native"] = _trace(nat_p) if nat_p != fix_p else ""
                    row[f"{conv}_trace_fixed05"] = _trace(fix_p) if nat_p != fix_p else ""

                row["competence_category"] = competence_category(
                    row["canonical_wm_exact"], row["canonical_ltm_exact"])
                if fa is not None:
                    row["competence_category_freear"] = competence_category(
                        row["freear_wm_exact"], row["freear_ltm_exact"])

                if intact_by_item is not None:
                    ref = intact_by_item.get(int(i))
                    if ref is not None:
                        row["delta_gate"] = _sub(row["gate"], ref.get("gate"))
                        row["delta_c_LTM"] = _sub(row["c_LTM"], ref.get("c_LTM"))
                        changed = 0
                        for conv in ("canonical", "freear"):
                            for r in ROUTES:
                                k = f"{conv}_{r}_predicted"
                                if k in row and k in ref and row[k] != ref[k]:
                                    changed += 1
                                    row[f"{conv}_{r}_changed_vs_intact"] = 1
                                elif k in row:
                                    row[f"{conv}_{r}_changed_vs_intact"] = 0
                        row["n_route_conv_cells_changed_vs_intact"] = changed
                rows.append(row)

            if hook is not None:
                hook.unbind()
            if progress is not None:
                progress(min(lo + batch_size, total), total)
    finally:
        model.train(was_training)
    return rows


def _sub(a, b):
    if a is None or b is None:
        return None
    return float(a) - float(b)
