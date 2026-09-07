"""Per-item R / N / C error audit of a FINAL base-123 checkpoint.

Read-only.  Rebuilds the trainer at the checkpoint's own widths, re-runs the
committed evaluations, and writes one row per ERROR for each task, plus a
JSON summary.  Two checkpoints (e.g. u400 and u500, or two seeds at u500) can
then be compared with `--compare` to get set overlap / Jaccard, which is what
distinguishes a moving frontier of errors from a stable systematic set.

COMPREHENSION rows carry: target index/word/phonology/frequency rank/length,
target cosine and rank, predicted competitor index/word/cosine, margin, top-5,
whether the competitor is a homophone, whether the two GloVe vectors are
bitwise identical, and a competitor-relation label (morphological /
semantic-neighbour / numeric-calendar / other) derived from surface form and
GloVe geometry.

NAMING rows carry: target word/phonology/length/frequency rank, prediction,
edit distance, EOS emitted, predicted length, and an over/under-generation
flag.

REPETITION rows are produced for BOTH conventions -- canonical forced-length
and genuine free-AR -- with the per-route (full/wm/ltm) outcome, so a
convention-specific or route-specific failure is visible.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.train_joint_scratch import (           # noqa: E402
    FREE_AR_MAX_STEPS, LR_POLICY_TASK, NAMING_MAX_STEPS, OPT_POLICY_SHARED,
    JointScratchTrainer, build_batch,
)

MARGIN_EPS = 0.01
NEIGHBOUR_COS = 0.60
CALENDAR = set("""january february march april may june july august september
october november december monday tuesday wednesday thursday friday saturday
sunday""".split())
NUMBER = set("""zero one two three four five six seven eight nine ten eleven
twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty
thirty forty fifty sixty seventy eighty ninety hundred thousand million
billion first second third fourth fifth""".split())


def relation(target: str, pred: str, glove_cos: float, same_phon: bool) -> str:
    """Descriptive competitor label.  Morphology is judged on surface form
    only (shared stem / affix), so it is a heuristic, not a lemmatiser."""
    t, p = target.lower(), pred.lower()
    if same_phon:
        return "homophone"
    if t in CALENDAR and p in CALENDAR:
        return "calendar"
    if t in NUMBER and p in NUMBER:
        return "numeric"
    stem = min(len(t), len(p))
    shared = 0
    while shared < stem and t[shared] == p[shared]:
        shared += 1
    if shared >= max(4, min(len(t), len(p)) - 3) and shared >= 4:
        return "morphological"
    if t.endswith(p) or p.endswith(t):
        return "morphological"
    if glove_cos >= NEIGHBOUR_COS:
        return "semantic_neighbour"
    return "other"


def build(ckpt_path: str, device: str, *, max_words: int = 30000,
          dorsal_pool_size: int = 4000, batch_size: int = 64,
          glove_path: str = "data/glove.6B.300d.txt",
          allow_glove_fallback: bool = False,
          require_subset_hash: bool = True):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    w = ck["widths"]
    # RECONSTRUCT THE CHECKPOINT'S OWN SCIENTIFIC CONFIGURATION.  The audit
    # previously built the trainer with the driver defaults, so a checkpoint
    # trained under a task-specific LR policy was compared against the
    # historical two-stage one and the resume guard correctly refused it.
    # The fix is to mirror what the checkpoint actually stores, so `changed`
    # is EMPTY and no phase transition is involved: --phase-transition is
    # never passed, and the driver's safety checks are satisfied rather than
    # weakened.
    pol = ck.get("lr_policy") or {}
    task_lrs = ({k: float(pol[k])
                 for k in ("repetition", "naming", "comprehension")}
                if pol.get("kind") == LR_POLICY_TASK else None)
    tr = JointScratchTrainer(
        regime=ck["regime"], seed=ck["seed"], device=device,
        max_words=max_words,
        lexicon_path=ck["lexicon_path"], dorsal_pool_size=dorsal_pool_size,
        batch_size=batch_size, subset_mode=ck["subset_mode"],
        subset_per_band=822,
        subset_size=32, lr_boundary_steps=ck["lr_boundary_steps"],
        allow_glove_fallback=allow_glove_fallback,
        require_subset_hash=require_subset_hash, glove_path=glove_path,
        schedule=ck["schedule"], wm_hidden=w["wm_hidden"],
        enc_hidden=w["ltm_enc_hidden"], dec_hidden=w["ltm_dec_hidden"],
        task_lrs=task_lrs,
        optimizer_policy=ck.get("optimizer_policy", OPT_POLICY_SHARED),
        dec_weight=ck.get("dec_weight"),
        c_align_weight=float(ck.get("c_align_weight") or 0.0))
    # allow_phase_transition stays False: a mismatch must still be an error.
    assert tr.allow_phase_transition is False
    tr.load_state_dict(ck)
    if dict(tr.lr_policy) != dict(ck["lr_policy"]):
        raise RuntimeError(
            f"reconstructed lr_policy {dict(tr.lr_policy)} != checkpoint "
            f"{dict(ck['lr_policy'])}")
    if not tr.phase_transitions == list(ck.get("phase_transitions", [])):
        raise RuntimeError("the audit introduced a phase transition")
    tr.model.eval()
    return tr, ck


@torch.no_grad()
def comp_errors(tr) -> List[dict]:
    from scripts.naming_comprehension.frozen_probe import (
        comprehension_metrics, encode_all)
    idx = list(tr.comp_idx)
    forms = [tr.entries[i].phonemes for i in idx]
    s_hat = encode_all(tr.model, tr.vocab, forms, tr.device, 512)
    m = comprehension_metrics(s_hat.cpu(), tr.bank_raw.cpu(), idx, 512)
    bank_n = F.normalize(tr.bank_raw, dim=-1)
    q = F.normalize(s_hat.cpu(), dim=-1)
    phon: Dict[tuple, List[int]] = {}
    for i, e in enumerate(tr.entries):
        phon.setdefault(tuple(e.phonemes), []).append(i)
    vec = [tr.bank_raw[i].numpy().tobytes() for i in range(len(tr.entries))]
    groups: Dict[bytes, List[int]] = {}
    for i, k in enumerate(vec):
        groups.setdefault(k, []).append(i)
    itos = tr.vocab.itos
    rows = []
    for pos in np.where(m["top1"] == 0)[0]:
        tgt = idx[int(pos)]
        sims = q[int(pos)] @ bank_n.t()
        top5 = torch.topk(sims, 5).indices.tolist()
        pred = top5[0]
        te, pe = tr.entries[tgt], tr.entries[pred]
        same_phon = tuple(pe.phonemes) == tuple(te.phonemes)
        gc = float(F.cosine_similarity(tr.bank_raw[tgt].unsqueeze(0),
                                       tr.bank_raw[pred].unsqueeze(0)))
        dup = groups.get(vec[tgt], [tgt])
        rows.append({
            "task": "comprehension", "target_bank_index": tgt,
            "target_word": te.word,
            "target_phonemes": " ".join(itos[p] for p in te.phonemes),
            "target_freq_rank": te.rank,
            "target_phon_length": len(te.phonemes),
            "target_cos": round(float(m["target_cos"][int(pos)]), 6),
            "target_rank": int(m["target_rank"][int(pos)]),
            "in_top5": int(m["top5"][int(pos)]),
            "pred_bank_index": pred, "pred_word": pe.word,
            "pred_freq_rank": pe.rank,
            "pred_cos": round(float(sims[pred]), 6),
            "margin_target_minus_top1": round(float(m["margin"][int(pos)]), 6),
            "pred_target_glove_cos": round(gc, 6),
            "is_homophone_of_target": int(same_phon),
            "identical_glove_vectors": int(len(dup) > 1
                                           and pred in dup),
            "mathematically_unavoidable": int(any(i < tgt for i in dup)),
            "homophone_group_size": len(phon[tuple(te.phonemes)]),
            "top5_words": " | ".join(tr.entries[i].word for i in top5),
            "relation": relation(te.word, pe.word, gc, same_phon)})
    return rows


@torch.no_grad()
def naming_errors(tr) -> List[dict]:
    from scripts.naming_comprehension.frozen_probe import semantic_greedy_decode
    from scripts.external_eval import _edit_distance
    itos = tr.vocab.itos
    rows = []
    idx = list(tr.naming_idx)
    for lo in range(0, len(idx), 512):
        chunk = idx[lo:lo + 512]
        sem = tr.bank_raw[torch.tensor(chunk)].to(tr.device)
        preds, eos = semantic_greedy_decode(tr.model, sem, tr.vocab,
                                            NAMING_MAX_STEPS)
        for k, i in enumerate(chunk):
            gold = tr.entries[i].phonemes
            if preds[k] == gold:
                continue
            rows.append({
                "task": "naming", "target_bank_index": i,
                "target_word": tr.entries[i].word,
                "target_phonemes": " ".join(itos[p] for p in gold),
                "target_freq_rank": tr.entries[i].rank,
                "target_phon_length": len(gold),
                "pred_phonemes": " ".join(itos[p] for p in preds[k]),
                "pred_length": len(preds[k]),
                "edit_distance": _edit_distance(preds[k], gold),
                "eos_emitted": int(eos[k]),
                "length_delta": len(preds[k]) - len(gold),
                "over_generation": int(len(preds[k]) > len(gold)),
                "no_eos": int(not eos[k])})
    return rows


@torch.no_grad()
def rep_errors(tr) -> List[dict]:
    """Both conventions, all three routes, per item."""
    from scripts.evaluate_train_lexicon_ceiling import evaluate_forms_ar
    itos = tr.vocab.itos
    idx = list(range(len(tr.entries)))
    items = [tr.entries[i] for i in idx]
    # evaluate_forms_ar returns a LIST of per-item rows, aligned with `items`,
    # carrying <route>_exact_match / _edit_dist / _predicted.
    canon = evaluate_forms_ar(tr.model, tr.vocab, items, tr.device,
                              routes=("full", "wm", "ltm"), wm_noise=False)
    ok_canon = {idx[k]: r for k, r in enumerate(canon)}
    rows = []
    # genuine free-AR, per item, per route
    free: Dict[str, List[int]] = {}
    for route in ("full", "wm", "ltm"):
        flags = []
        for lo in range(0, len(idx), 256):
            chunk = idx[lo:lo + 256]
            forms = [tr.entries[i].phonemes for i in chunk]
            me = max(len(f) for f in forms) + 1
            enc = torch.full((len(chunk), me), tr.vocab.pad_id, dtype=torch.long)
            msk = torch.zeros((len(chunk), me), dtype=torch.bool)
            for k, f in enumerate(forms):
                enc[k, :len(f) + 1] = torch.tensor(f + [tr.vocab.eos_id])
                msk[k, :len(f) + 1] = True
            enc, msk = enc.to(tr.device), msk.to(tr.device)
            dec = torch.full((len(chunk), 1), tr.vocab.bos_id,
                             dtype=torch.long, device=tr.device)
            for _ in range(FREE_AR_MAX_STEPS):
                o = tr.model(enc, msk, dec)
                key = {"full": "logits", "wm": "wm_logits",
                       "ltm": "ltm_logits"}[route]
                dec = torch.cat([dec, o[key][:, -1, :].argmax(-1, keepdim=True)],
                                dim=1)
                if bool((dec == tr.vocab.eos_id).any(dim=1).all()):
                    break
            for k, f in enumerate(forms):
                seq = dec[k, 1:].tolist()
                if tr.vocab.eos_id in seq:
                    seq = seq[:seq.index(tr.vocab.eos_id)]
                flags.append((seq == f, seq))
        free[route] = flags
    for pos, i in enumerate(idx):
        gold = tr.entries[i].phonemes
        f_ok, f_seq = free["full"][pos]
        c_row = ok_canon.get(i, {})
        c_ok = c_row.get("full_exact_match")
        if f_ok and (c_ok in (1, True, None)):
            continue
        rows.append({
            "task": "repetition", "target_bank_index": i,
            "target_word": tr.entries[i].word,
            "target_phonemes": " ".join(itos[p] for p in gold),
            "target_freq_rank": tr.entries[i].rank,
            "target_phon_length": len(gold),
            "canonical_exact": ("" if c_ok is None else int(bool(c_ok))),
            "canonical_exact_wm": c_row.get("wm_exact_match", ""),
            "canonical_exact_ltm": c_row.get("ltm_exact_match", ""),
            "canonical_pred": c_row.get("full_predicted", ""),
            "canonical_edit_dist": c_row.get("full_edit_dist", ""),
            "freear_exact_full": int(f_ok),
            "freear_exact_wm": int(free["wm"][pos][0]),
            "freear_exact_ltm": int(free["ltm"][pos][0]),
            "freear_pred": " ".join(itos[p] for p in f_seq),
            "freear_pred_length": len(f_seq),
            "length_delta": len(f_seq) - len(gold),
            "convention_disagrees": ("" if c_ok is None
                                     else int(bool(c_ok) != bool(f_ok)))})
    return rows


def write(path, rows):
    if not rows:
        open(path, "w").write("")
        print(f"[err-audit] (0 rows) {path}")
        return
    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[err-audit] wrote {path}  ({len(rows)} rows)")


def compare(paths: List[str], out: str) -> int:
    """Set overlap / Jaccard between two or more error TSVs of the same task."""
    sets, labels = [], []
    for p in paths:
        lab = os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p)
        with open(p, encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        sets.append({r["target_bank_index"] for r in rows})
        labels.append(lab)
    res = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            inter, union = len(a & b), len(a | b)
            res.append({"a": labels[i], "b": labels[j], "n_a": len(a),
                        "n_b": len(b), "intersection": inter, "union": union,
                        "jaccard": round(inter / union, 6) if union else None,
                        "frac_of_a_retained": round(inter / len(a), 6) if a else None,
                        "frac_of_b_new": round((len(b) - inter) / len(b), 6) if b else None})
    write(out, res)
    for r in res:
        print(f"[err-audit] {r['a']} vs {r['b']}: |A|={r['n_a']} |B|={r['n_b']} "
              f"inter={r['intersection']} jaccard={r['jaccard']}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tasks", default="comprehension,naming,repetition")
    ap.add_argument("--max-words", type=int, default=30000)
    ap.add_argument("--dorsal-pool-size", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--glove-path", default="data/glove.6B.300d.txt")
    ap.add_argument("--allow-glove-fallback", action="store_true")
    ap.add_argument("--no-subset-hash-check", action="store_true")
    ap.add_argument("--compare", nargs="*", default=None,
                    help="error TSVs to compare instead of auditing a ckpt")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.compare:
        return compare(args.compare,
                       os.path.join(args.out_dir, "error_overlap.tsv"))

    if not args.ckpt:
        ap.error("--ckpt is required unless --compare is used")
    def _fingerprint(model):
        return {n: p.detach().clone().cpu()
                for n, p in model.named_parameters()}

    tr, ck = build(args.ckpt, args.device, max_words=args.max_words,
                   dorsal_pool_size=args.dorsal_pool_size,
                   batch_size=args.batch_size, glove_path=args.glove_path,
                   allow_glove_fallback=args.allow_glove_fallback,
                   require_subset_hash=not args.no_subset_hash_check)
    step = int(ck["global_step"])
    u = step / (463 * 6)
    print(f"[err-audit] {args.ckpt}")
    print(f"[err-audit] widths {ck['widths']} seed {ck['seed']} "
          f"step {step} (u={u:.1f})")
    before = _fingerprint(tr.model)
    tasks = args.tasks.split(",")
    summary = {"checkpoint": os.path.abspath(args.ckpt), "step": step,
               "u": round(u, 4), "seed": ck["seed"], "widths": ck["widths"],
               "git_commit": (ck.get("git") or {}).get("commit")}
    if "comprehension" in tasks:
        rows = comp_errors(tr)
        write(os.path.join(args.out_dir, "comp_errors.tsv"), rows)
        from collections import Counter
        summary["comp_errors"] = len(rows)
        summary["comp_population"] = len(tr.comp_idx)
        summary["comp_relations"] = dict(Counter(r["relation"] for r in rows))
        summary["comp_in_top5"] = sum(r["in_top5"] for r in rows)
        summary["comp_unavoidable"] = sum(
            r["mathematically_unavoidable"] for r in rows)
        if rows:
            mg = [r["margin_target_minus_top1"] for r in rows]
            rk = [r["target_rank"] for r in rows]
            summary["comp_margin"] = {
                "min": min(mg), "median": float(np.median(mg)), "max": max(mg),
                "n_within_0.01": sum(1 for m in mg if m > -MARGIN_EPS)}
            summary["comp_rank"] = {"median": float(np.median(rk)),
                                    "max": int(max(rk))}
    if "naming" in tasks:
        rows = naming_errors(tr)
        write(os.path.join(args.out_dir, "naming_errors.tsv"), rows)
        summary["naming_errors"] = len(rows)
        summary["naming_no_eos"] = sum(r["no_eos"] for r in rows)
        summary["naming_over_generation"] = sum(r["over_generation"] for r in rows)
        if rows:
            summary["naming_edit"] = {
                "median": float(np.median([r["edit_distance"] for r in rows])),
                "max": int(max(r["edit_distance"] for r in rows))}
    if "repetition" in tasks:
        rows = rep_errors(tr)
        write(os.path.join(args.out_dir, "rep_errors.tsv"), rows)
        summary["rep_error_rows"] = len(rows)
        summary["rep_convention_disagreements"] = sum(
            1 for r in rows if r["convention_disagrees"] == 1)
    # READ-ONLY PROOF: no optimizer step can have run.
    now = dict(tr.model.named_parameters())
    moved = [n for n, ref in before.items()
             if not torch.equal(now[n].detach().cpu(), ref)]
    if moved:
        raise RuntimeError(
            f"AUDIT MUTATED THE MODEL: {moved[:8]} -- this must never happen")
    print(f"[err-audit] read-only verified: all {len(before)} parameter "
          f"tensors bitwise unchanged")
    summary["read_only_verified"] = True
    summary["lr_policy"] = dict(ck["lr_policy"])
    summary["optimizer_policy"] = ck.get("optimizer_policy")
    summary["phase_transitions_in_checkpoint"] = len(
        ck.get("phase_transitions") or [])
    p = os.path.join(args.out_dir, "error_summary.json")
    json.dump(summary, open(p, "w"), indent=1, default=str)
    print(f"[err-audit] wrote {p}")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("checkpoint",)}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
