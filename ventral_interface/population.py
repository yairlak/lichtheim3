"""Population and identity relationships, reconstructed from the frozen code.

Frozen relationships (contract §4):

* LEXICAL ROWS  P_R = range(29,571): one row per lexicon entry.  item_index = bank row
  = `tr.entries` position.  This is the historical REPETITION (and Naming) population.
* PHONOLOGICAL CLASS of i = all rows with identical `tuple(entries[i].phonemes)`
  (`train_tasks.phonology_groups`).
* CANONICAL C TARGET c(i) = min(class, key=(rank, index))
  (`train_tasks.canonical_phonology_indices`).  P_C = {i : c(i) == i}, |P_C| = 27,981.
* HISTORICAL C CORRECTNESS (defined on P_C only):
  argmax_j cos(normalize(s_hat_i), normalize(bank_raw)_j) over ALL 29,571 rows == i
  (`frozen_probe.comprehension_metrics`, via `evaluate_comprehension_subset`).
  This is exact lexical-row identity of the canonical target; homophone rows remain
  competitors and a homophone hit is WRONG.
* For i not in P_C the historical C contract is undefined -> NA; lexical-identity and
  phonology correctness are still computed.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from scripts.naming_comprehension.frozen_probe import comprehension_metrics, encode_all
from scripts.naming_comprehension.train_tasks import (
    canonical_phonology_indices, evaluate_comprehension_subset, phonology_groups)

RETRIEVAL_BATCH = 512      # = coexistence_probe.full_battery / evaluate_comprehension_subset call


def population_structure(entries) -> Dict[str, object]:
    groups = phonology_groups(entries)
    comp = canonical_phonology_indices(entries)
    canon_of: Dict[int, int] = {}
    size_of: Dict[int, int] = {}
    members_of: Dict[int, str] = {}
    for g in groups.values():
        c = min(g, key=lambda i: (entries[i].rank, i))
        words = "|".join(entries[i].word for i in sorted(g))
        for i in g:
            canon_of[i] = c
            size_of[i] = len(g)
            members_of[i] = words
    in_c = set(comp)
    n = len(entries)
    return {
        "n_rows": n,
        "comp_idx": comp,
        "n_comp": len(comp),
        "n_noncanonical_homophone_members": n - len(comp),
        "n_phonology_classes": len(groups),
        "n_homophone_classes": sum(1 for g in groups.values() if len(g) > 1),
        "canonical_of": canon_of,
        "group_size_of": size_of,
        "group_members_of": members_of,
        "in_c": [int(i in in_c) for i in range(n)],
    }


def historical_c_contract(model, tr, comp_idx: Sequence[int]) -> Dict[str, object]:
    """Exactly the historical battery call; returns per-canonical-target C outcome."""
    c = evaluate_comprehension_subset(model, tr.vocab, tr.entries, tr.bank_raw,
                                      list(comp_idx), "cpu", RETRIEVAL_BATCH,
                                      return_per_item=True)
    word_to_idx = {e.word: i for i, e in enumerate(tr.entries)}
    per = {}
    for r in c["_per_item"]:
        per[int(r["bank_index"])] = {"top1": int(r["top1"]),
                                     "top1_index": word_to_idx[r["top1_word"]]}
    errors = int(round((1 - c["top1"]) * len(comp_idx)))
    return {"per_item": per, "c_errors": errors,
            "c_errors_exact_count": sum(1 for v in per.values() if v["top1"] == 0)}


def retrieval_all_rows(model, tr) -> Dict[str, object]:
    """Historical cosine rule applied to every lexical row (targets = own row)."""
    n = len(tr.entries)
    forms = [e.phonemes for e in tr.entries]
    s_hat = encode_all(model, tr.vocab, forms, "cpu", RETRIEVAL_BATCH)
    m = comprehension_metrics(s_hat.cpu().float(), tr.bank_raw.cpu().float(),
                              list(range(n)), RETRIEVAL_BATCH)
    return {"s_hat": s_hat, "metrics": m}


def gate_r(hist: Dict, retr: Dict, comp_idx: Sequence[int], archived_c_errors: int) -> Dict:
    """GATE R (retrieval reproduction): the per-row retrieval used for S1 must equal
    the historical C call on every canonical target, and the historical C error count
    must equal the archived battery value."""
    top1_idx = retr["metrics"]["top1_idx"]
    mism = [i for i in comp_idx if int(top1_idx[i]) != hist["per_item"][i]["top1_index"]]
    return {"archived_c_errors": archived_c_errors, "recomputed_c_errors": hist["c_errors"],
            "n_top1_index_mismatch_vs_historical_call": len(mism),
            "pass": hist["c_errors"] == archived_c_errors == hist["c_errors_exact_count"]
            and not mism}


def gate_a(hist: Dict, comp_idx: Sequence[int], canonical_of: Dict[int, int]) -> Dict:
    """GATE A: every historically C-correct item has retrieved lexical identity ==
    target lexical identity (== canonical target, since i in P_C)."""
    bad = [i for i in comp_idx if hist["per_item"][i]["top1"] == 1
           and (hist["per_item"][i]["top1_index"] != i or canonical_of[i] != i)]
    return {"n_c_correct": sum(1 for i in comp_idx if hist["per_item"][i]["top1"] == 1),
            "n_violations": len(bad), "pass": not bad}
