"""Read-only per-item audit of every strict top-1 comprehension error.

Takes a CAP-3 comprehension checkpoint, re-runs the committed evaluation
(encode_all -> comprehension_metrics: cosine retrieval of each canonical
target against the FULL 29,571 bank) and writes one row per top-1 ERROR with
everything needed to classify it.  The task definition is not touched: this
reads the same numbers the training run reports.

Error taxonomy (a row can fall in several):
  homophone            the predicted word has the EXACT phoneme sequence of
                       the target.  The canonical C population keeps one
                       target per phonology, but the BANK still contains
                       every homophone, so these are competitors that are
                       phonologically indistinguishable from the input --
                       an identifiability limit, not a model error.
  duplicate_semantic   target and prediction have bitwise-identical GloVe
                       vectors.  Retrieval between them is undecidable; this
                       would be a DATA issue, not a capacity one.
  near_tie             cosine margin (target - top1) above -MARGIN_EPS, i.e.
                       the target was beaten by a hair.
  semantic_neighbour   the predicted word's GloVe vector is highly similar to
                       the target's (cos >= NEIGHBOUR_COS): an ordinary
                       semantic confusion.
  far_miss             none of the above; the target is genuinely far down.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.route_capacity_probe import (          # noqa: E402
    ROUTE_COMPREHENSION, RouteCapacityTrainer,
)

MARGIN_EPS = 0.01          # |margin| below this counts as a near tie
NEIGHBOUR_COS = 0.60       # target/prediction GloVe similarity for "neighbour"

COLUMNS = ["target_bank_index", "target_word", "target_phonemes",
           "target_freq_rank", "target_phon_length", "target_cos",
           "target_rank", "pred_bank_index", "pred_word", "pred_phonemes",
           "pred_freq_rank", "pred_cos", "margin_target_minus_top1",
           "pred_target_glove_cos", "is_homophone_of_target",
           "identical_glove_vectors", "homophone_group_size",
           "top5_words", "top5_bank_indices", "top5_cos", "category"]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=512)
    # passthroughs so the tool can be exercised on a small fixture
    ap.add_argument("--max-words", type=int, default=30000)
    ap.add_argument("--glove-path", default="data/glove.6B.300d.txt")
    ap.add_argument("--allow-glove-fallback", action="store_true")
    ap.add_argument("--no-population-hash-check", action="store_true")
    args = ap.parse_args(argv)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if ck.get("route") != ROUTE_COMPREHENSION:
        raise SystemExit(f"not a comprehension checkpoint: {ck.get('route')!r}")
    w = ck["widths"]
    print(f"[audit] {args.ckpt}")
    print(f"[audit] step {ck['global_step']} "
          f"({ck['global_step'] / ck['per_epoch']:.1f} exposures) widths {w}")

    tr = RouteCapacityTrainer(
        route=ROUTE_COMPREHENSION, wm_hidden=w["wm_hidden"],
        enc_hidden=w["ltm_enc_hidden"], dec_hidden=w["ltm_dec_hidden"],
        seed=ck["seed"], device=args.device, max_words=args.max_words,
        glove_path=args.glove_path,
        allow_glove_fallback=args.allow_glove_fallback,
        require_population_hash=not args.no_population_hash_check)
    tr.load_state_dict(ck)
    if tr.population_hash != ck["population_sha256"]:
        raise SystemExit("population hash mismatch")

    from scripts.naming_comprehension.frozen_probe import (
        comprehension_metrics, encode_all)
    tr.model.eval()
    forms = [tr.entries[i].phonemes for i in tr.train_idx]
    with torch.no_grad():
        s_hat = encode_all(tr.model, tr.vocab, forms, args.device,
                           args.batch_size)
    m = comprehension_metrics(s_hat.cpu(), tr.bank_raw.cpu(), tr.train_idx,
                              args.batch_size)

    n = len(tr.train_idx)
    top1 = float(np.mean(m["top1"]))
    top5 = float(np.mean(m["top5"]))
    err_pos = np.where(m["top1"] == 0)[0]
    print(f"[audit] population {n}  bank {len(tr.entries)}")
    print(f"[audit] top1 {top1:.6f}  ERRORS {len(err_pos)}  "
          f"top5 {top5:.6f}  outside-top5 {int((1 - top5) * n + 0.5)}")

    # bank geometry needed for the taxonomy
    bank_n = F.normalize(tr.bank_raw, dim=-1)
    phon = {}
    for i, e in enumerate(tr.entries):
        phon.setdefault(tuple(e.phonemes), []).append(i)
    itos = tr.vocab.itos

    def phon_str(idx):
        return " ".join(itos[p] for p in tr.entries[idx].phonemes)

    q_all = F.normalize(s_hat.cpu(), dim=-1)
    rows, cats = [], Counter()
    for pos in err_pos:
        tgt = tr.train_idx[int(pos)]
        sims = (q_all[int(pos)] @ bank_n.t())
        order = torch.topk(sims, 5)
        top5_idx = order.indices.tolist()
        pred = top5_idx[0]
        te, pe = tr.entries[tgt], tr.entries[pred]
        same_phon = tuple(pe.phonemes) == tuple(te.phonemes)
        identical_vec = bool(torch.equal(tr.bank_raw[tgt], tr.bank_raw[pred]))
        glove_cos = float(F.cosine_similarity(
            tr.bank_raw[tgt].unsqueeze(0), tr.bank_raw[pred].unsqueeze(0)))
        margin = float(m["margin"][int(pos)])
        if identical_vec:
            cat = "duplicate_semantic"
        elif same_phon:
            cat = "homophone"
        elif margin > -MARGIN_EPS:
            cat = "near_tie"
        elif glove_cos >= NEIGHBOUR_COS:
            cat = "semantic_neighbour"
        else:
            cat = "far_miss"
        cats[cat] += 1
        rows.append({
            "target_bank_index": tgt, "target_word": te.word,
            "target_phonemes": phon_str(tgt), "target_freq_rank": te.rank,
            "target_phon_length": len(te.phonemes),
            "target_cos": round(float(m["target_cos"][int(pos)]), 6),
            "target_rank": int(m["target_rank"][int(pos)]),
            "pred_bank_index": pred, "pred_word": pe.word,
            "pred_phonemes": phon_str(pred), "pred_freq_rank": pe.rank,
            "pred_cos": round(float(sims[pred]), 6),
            "margin_target_minus_top1": round(margin, 6),
            "pred_target_glove_cos": round(glove_cos, 6),
            "is_homophone_of_target": int(same_phon),
            "identical_glove_vectors": int(identical_vec),
            "homophone_group_size": len(phon[tuple(te.phonemes)]),
            "top5_words": " | ".join(tr.entries[i].word for i in top5_idx),
            "top5_bank_indices": " ".join(str(i) for i in top5_idx),
            "top5_cos": " ".join(f"{float(sims[i]):.4f}" for i in top5_idx),
            "category": cat})

    os.makedirs(args.out_dir, exist_ok=True)
    tsv = os.path.join(args.out_dir, "c512_top1_errors.tsv")
    with open(tsv, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t")
        wtr.writeheader()
        for r in sorted(rows, key=lambda r: r["margin_target_minus_top1"],
                        reverse=True):
            wtr.writerow(r)

    # ---- population-level contrasts for frequency / length effects -------
    all_ranks = np.array([tr.entries[i].rank for i in tr.train_idx])
    all_lens = np.array([len(tr.entries[i].phonemes) for i in tr.train_idx])
    e_ranks = np.array([r["target_freq_rank"] for r in rows]) if rows else np.array([])
    e_lens = np.array([r["target_phon_length"] for r in rows]) if rows else np.array([])
    summary = {
        "checkpoint": os.path.abspath(args.ckpt),
        "step": int(ck["global_step"]),
        "exposures": round(ck["global_step"] / ck["per_epoch"], 4),
        "widths": w, "population": n, "bank": len(tr.entries),
        "population_sha256": tr.population_hash,
        "top1": top1, "top1_errors": int(len(err_pos)),
        "top5": top5, "outside_top5": int(round((1 - top5) * n)),
        "rank_mean": float(np.mean(m["target_rank"])),
        "rank_median": float(np.median(m["target_rank"])),
        "target_cos_mean": float(np.mean(m["target_cos"])),
        "margin_mean": float(np.mean(m["margin"])),
        "categories": dict(cats),
        "margin_of_errors": {
            "median": float(np.median([r["margin_target_minus_top1"]
                                       for r in rows])) if rows else None,
            "n_within_0.01": int(sum(
                1 for r in rows if r["margin_target_minus_top1"] > -MARGIN_EPS)),
            "n_within_0.05": int(sum(
                1 for r in rows if r["margin_target_minus_top1"] > -0.05)),
        },
        "frequency_effect": {
            "median_rank_all": float(np.median(all_ranks)),
            "median_rank_errors": float(np.median(e_ranks)) if rows else None,
        },
        "length_effect": {
            "mean_length_all": float(np.mean(all_lens)),
            "mean_length_errors": float(np.mean(e_lens)) if rows else None,
        },
        "error_rank_distribution": {
            "target_rank_max": int(max((r["target_rank"] for r in rows),
                                       default=0)),
            "target_rank_median": float(np.median(
                [r["target_rank"] for r in rows])) if rows else None,
        },
        "thresholds": {"margin_eps": MARGIN_EPS,
                       "neighbour_cos": NEIGHBOUR_COS},
    }
    js = os.path.join(args.out_dir, "c512_error_summary.json")
    json.dump(summary, open(js, "w"), indent=1)
    print(f"[audit] categories: {dict(cats)}")
    print(f"[audit] wrote {tsv}")
    print(f"[audit] wrote {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
