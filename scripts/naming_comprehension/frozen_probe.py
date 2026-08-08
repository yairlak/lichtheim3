"""Phase 1A: frozen-checkpoint naming & comprehension diagnostic probe.

Everything here is evaluation-side only: no weights, losses, gate, training
code, data splits or checkpoints are modified, and no methods are added to the
model classes.

Two diagnostic pathways over a frozen checkpoint:

  comprehension:  phonemes -> ltm.encode -> s_hat -> retrieval against the
                  canonical training GloVe bank (cosine).
  naming:         semantic vector -> tanh(sem_to_h0) -> LTM decoder ->
                  dec_to_premotor -> shared motor projection -> phonemes,
                  decoded greedily from BOS until EOS or a global cap.

Populations
-----------
The strict comprehension population is the set of trained-lexicon words whose
phonology is UNIQUE in the lexicon.  Homophones (identical phoneme sequences,
distinct orthography/GloVe) are kept in the per-item output and aggregated
separately: the phonological encoder maps homophones to identical s_hat, so
their target-word retrieval is structurally capped and must not be pooled with
the unambiguous population.  `homophone-class retrieval` (top-1 word is any
member of the item's phonology class) is an auxiliary metric only — it is NOT
semantic comprehension accuracy.

Naming decoding convention (documented global cap)
--------------------------------------------------
No repository-wide free-running AR cap exists (the existing AR evaluation,
scripts/external_eval.py:autoregressive_decode_batch, is forced-length: it is
bounded by each item's gold length, which this probe must not use).  The probe
therefore uses a documented global cap derived from the supported lexical
sequence length:

    max_steps = cfg.data.max_phonemes + 1

i.e. enough steps to emit the longest supported word (max_phonemes phonemes)
plus its EOS.  Failure to emit EOS within the cap is recorded per item
(`*_eos_emitted` = 0) and necessarily scores as a whole-word error.  The
decoder wrapper never inspects the target item length.

Distribution-shift caveat
-------------------------
sem_to_h0 and the LTM decoder were trained exclusively on encoder outputs
s_hat, never on raw GloVe vectors.  Naming from raw GloVe is therefore an
out-of-distribution probe; the A (raw GloVe) vs B (s_hat) control and the
per-item shift diagnostics (cos, L2, norms) exist to separate input-side shift
from decoder-side failure.  Raw (unnormalised) GloVe is used as the naming
input because sem_to_h0 is magnitude-sensitive and the alignment loss targets
raw GloVe vectors.

Usage:
    python scripts/naming_comprehension/frozen_probe.py \
        --ckpt archives/.../seed_22_epoch_0140.pt \
        --out-dir outputs/naming_comprehension_93a577f/seed22_e140 \
        [--limit 200] [--include-words see,sea,key,quay]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import (Config, DataConfig, WMConfig, LTMConfig, GatingConfig,  # noqa: E402
                    LossConfig, TrainConfig)
from data.phonemes import Vocab, build_vocab                                # noqa: E402
from data.lexicon import build_lexicon, LexEntry                            # noqa: E402
from models.dual_route import DualRouteModel                                # noqa: E402
from scripts.external_eval import _edit_distance                            # noqa: E402
from utils.provenance import git_state, sha256_file, sha256_words_ordered   # noqa: E402


# ============================================================  loading  ====

def load_frozen(ckpt_path: str, device: str = "cpu"
                ) -> Tuple[DualRouteModel, Vocab, List[LexEntry], torch.Tensor, Config, dict]:
    """Rebuild model + full-lexicon entries from a frozen checkpoint.

    Same reconstruction pattern as scripts/external_eval.load_model_and_vocab,
    but returns the ordered training entries and the RAW (unnormalised) GloVe
    bank, and hard-verifies the bank order against the checkpoint's
    ordered_training_words_sha256.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt.get("split_mode") != "full_lexicon":
        raise ValueError(
            f"Phase 1A probe expects a full_lexicon checkpoint; "
            f"got split_mode={ckpt.get('split_mode')!r}."
        )

    cfg = Config(
        data   = DataConfig(**ckpt["cfg_data"]),
        wm     = WMConfig(**ckpt["cfg_wm"]),
        ltm    = LTMConfig(**ckpt["cfg_ltm"]),
        gating = GatingConfig(**ckpt["cfg_gating"]),
        loss   = LossConfig(**ckpt["cfg_loss"]),
        train  = TrainConfig(**ckpt["cfg_train"]),
    )
    cfg.train.device = device
    # Checkpoint stores repo-relative data paths; make them robust to CWD.
    for attr in ("lexicon_path", "glove_path"):
        p = getattr(cfg.data, attr, None)
        if p and not os.path.isabs(p):
            setattr(cfg.data, attr, os.path.join(ROOT, p))

    vocab = build_vocab()
    lexicon = build_lexicon(cfg.data, vocab)
    # full_lexicon: all entries train, file order == canonical bank order.
    entries = lexicon.entries

    ordered_hash = sha256_words_ordered([e.word for e in entries])
    if ordered_hash != ckpt["ordered_training_words_sha256"]:
        raise RuntimeError(
            "Rebuilt lexicon order does not match checkpoint provenance: "
            f"{ordered_hash} != {ckpt['ordered_training_words_sha256']}"
        )

    # The word-order hash does NOT cover the semantic vectors: if the GloVe
    # file is missing, build_bundled silently substitutes deterministic
    # pseudo-vectors and every semantic metric becomes meaningless.  Fail fast
    # when the rebuilt lexicon used more fallback vectors than the checkpoint
    # recorded at training time.
    stats = getattr(lexicon, "load_stats", None)
    ckpt_fallback = int(ckpt.get("n_glove_fallback", 0))
    if stats is not None and stats.n_glove_fallback > ckpt_fallback:
        raise RuntimeError(
            "GloVe coverage mismatch vs checkpoint provenance: rebuilt lexicon "
            f"used {stats.n_glove_fallback} deterministic fallback vectors but "
            f"the checkpoint recorded {ckpt_fallback} "
            f"(glove_present={ckpt.get('glove_present')}). The GloVe file is "
            f"probably missing at {cfg.data.glove_path!r}; the semantic bank "
            "would be silently wrong."
        )

    bank_raw = torch.stack([torch.as_tensor(e.semantic) for e in entries]).float()

    model = DualRouteModel(cfg, vocab, premotor_dim=ckpt.get("premotor_dim", 128))
    model.set_semantic_bank(bank_raw.clone())   # canonical L2-normalised bank
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, vocab, entries, bank_raw.to(device), cfg, ckpt


# =================================================  semantic AR wrapper  ====

@torch.no_grad()
def semantic_greedy_decode(model: DualRouteModel, sem: torch.Tensor,
                           vocab: Vocab, max_steps: int
                           ) -> Tuple[List[List[int]], List[bool]]:
    """Greedy autoregressive decoding from a semantic vector.

    Thin evaluation-side wrapper over the existing frozen machinery
    (ltm.decode_from_s_hat -> motor): each step re-decodes the growing prefix
    through ltm.decode_from_s_hat (which applies tanh(sem_to_h0) ->
    decoder GRU -> dec_to_premotor) and reads the shared motor projection.

    Starts from BOS; stops at EOS per item or at the global `max_steps` cap.
    Never inspects target lengths.

    Args:
        sem: (B, semantic_dim) semantic vectors (raw GloVe or s_hat).

    Returns:
        preds:       per item, phoneme ids up to (excluding) the first EOS.
        eos_emitted: per item, whether EOS appeared within the cap.
    """
    B = sem.shape[0]
    device = sem.device
    dec_input = torch.full((B, 1), vocab.bos_id, dtype=torch.long, device=device)
    for _ in range(max_steps):
        premotor = model.ltm.decode_from_s_hat(sem, dec_input)
        logits = model.motor(premotor)
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)         # (B, 1)
        dec_input = torch.cat([dec_input, nxt], dim=1)
        if bool((dec_input == vocab.eos_id).any(dim=1).all()):
            break
    preds: List[List[int]] = []
    eos_emitted: List[bool] = []
    for row in dec_input[:, 1:].tolist():
        seq: List[int] = []
        eos = False
        for tok in row:
            if tok == vocab.eos_id:
                eos = True
                break
            seq.append(tok)
        preds.append(seq)
        eos_emitted.append(eos)
    return preds, eos_emitted


@torch.no_grad()
def teacher_forced_naming(model: DualRouteModel, sem: torch.Tensor,
                          forms: Sequence[List[int]], vocab: Vocab
                          ) -> List[bool]:
    """Diagnostic only: per-item TF exact match (argmax at every gold-prefix
    position equals dec_tgt, EOS included) from a semantic vector."""
    B = len(forms)
    max_dec = max(len(f) for f in forms) + 1
    dec_in = torch.full((B, max_dec), vocab.pad_id, dtype=torch.long, device=sem.device)
    dec_tgt = torch.full((B, max_dec), vocab.pad_id, dtype=torch.long, device=sem.device)
    for i, f in enumerate(forms):
        dec_in[i, :len(f) + 1] = torch.tensor([vocab.bos_id] + f)
        dec_tgt[i, :len(f) + 1] = torch.tensor(f + [vocab.eos_id])
    logits = model.motor(model.ltm.decode_from_s_hat(sem, dec_in))
    pred = logits.argmax(-1)
    ok: List[bool] = []
    for i, f in enumerate(forms):
        n = len(f) + 1
        ok.append(bool((pred[i, :n] == dec_tgt[i, :n]).all()))
    return ok


# ==========================================================  encoding  ====

@torch.no_grad()
def encode_all(model: DualRouteModel, vocab: Vocab, forms: Sequence[List[int]],
               device: str, batch_size: int) -> torch.Tensor:
    """phonemes -> s_hat for every item (deterministic eval; enc_in = form+EOS)."""
    out = []
    for lo in range(0, len(forms), batch_size):
        chunk = forms[lo:lo + batch_size]
        max_enc = max(len(f) for f in chunk) + 1
        enc_in = torch.full((len(chunk), max_enc), vocab.pad_id, dtype=torch.long)
        enc_mask = torch.zeros((len(chunk), max_enc), dtype=torch.bool)
        for i, f in enumerate(chunk):
            enc_in[i, :len(f) + 1] = torch.tensor(f + [vocab.eos_id])
            enc_mask[i, :len(f) + 1] = True
        s_hat = model.ltm.encode(enc_in.to(device), enc_mask.to(device))
        out.append(s_hat)
    return torch.cat(out, dim=0)


# ====================================================  comprehension  =====

@torch.no_grad()
def comprehension_metrics(s_hat: torch.Tensor, bank_raw: torch.Tensor,
                          item_indices: Sequence[int],
                          batch_size: int) -> Dict[str, np.ndarray]:
    """Cosine retrieval of each item's own GloVe target in the full bank.

    `item_indices[i]` is the bank row of probe item i (identity mapping on the
    full lexicon; general to support smoke subsets).
    """
    bank_n = F.normalize(bank_raw, dim=-1)
    q_all = F.normalize(s_hat, dim=-1)
    n = q_all.shape[0]
    res = {k: np.zeros(n, dtype=np.float64) for k in
           ("target_cos", "target_rank", "top1", "top5", "margin", "c_ltm")}
    top1_idx = np.zeros(n, dtype=np.int64)
    for lo in range(0, n, batch_size):
        q = q_all[lo:lo + batch_size]
        sims = q @ bank_n.t()                                    # (b, N)
        tgt = torch.tensor(item_indices[lo:lo + q.shape[0]], dtype=torch.long)
        rows = torch.arange(q.shape[0])
        tgt_sim = sims[rows, tgt]
        rank = (sims > tgt_sim.unsqueeze(1)).sum(dim=1) + 1      # 1 = best
        sims_excl = sims.clone()
        sims_excl[rows, tgt] = -2.0
        best_other = sims_excl.max(dim=1).values
        res["target_cos"][lo:lo + q.shape[0]] = tgt_sim.numpy()
        res["target_rank"][lo:lo + q.shape[0]] = rank.numpy()
        res["top1"][lo:lo + q.shape[0]] = (sims.argmax(dim=1) == tgt).numpy()
        res["top5"][lo:lo + q.shape[0]] = (rank <= 5).numpy()
        res["margin"][lo:lo + q.shape[0]] = (tgt_sim - best_other).numpy()
        res["c_ltm"][lo:lo + q.shape[0]] = sims.max(dim=1).values.numpy()
        top1_idx[lo:lo + q.shape[0]] = sims.argmax(dim=1).numpy()
    res["top1_idx"] = top1_idx
    return res


# ==============================================================  main  =====

def _agg(rows: List[dict], keys: Sequence[str]) -> dict:
    out: dict = {"n": len(rows)}
    for k in keys:
        vals = [r[k] for r in rows]
        out[f"{k}_mean"] = float(np.mean(vals)) if vals else None
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0,
                    help="Probe only the first N lexicon items (smoke mode). 0 = all.")
    ap.add_argument("--include-words", default="",
                    help="Comma-separated words to force-include (smoke inspection).")
    args = ap.parse_args(argv)

    t0 = time.time()
    model, vocab, entries, bank_raw, cfg, ckpt = load_frozen(args.ckpt, args.device)
    n_lex = len(entries)

    # ---- probe item selection (bank/retrieval always uses the FULL lexicon)
    idx_sel = list(range(n_lex if args.limit <= 0 else min(args.limit, n_lex)))
    if args.include_words:
        wanted = {w.strip() for w in args.include_words.split(",") if w.strip()}
        by_word = {e.word: i for i, e in enumerate(entries)}
        missing = sorted(wanted - set(by_word))
        if missing:
            raise SystemExit(f"--include-words not in lexicon: {missing}")
        have = set(idx_sel)
        idx_sel += [by_word[w] for w in sorted(wanted) if by_word[w] not in have]

    items = [entries[i] for i in idx_sel]
    forms = [e.phonemes for e in items]

    # ---- homophone classes over the FULL lexicon (not just the subset)
    phon_groups: Dict[tuple, List[int]] = collections.defaultdict(list)
    for i, e in enumerate(entries):
        phon_groups[tuple(e.phonemes)].append(i)
    group_size = {i: len(phon_groups[tuple(e.phonemes)]) for i, e in enumerate(entries)}
    n_unique_phonology = sum(1 for g in phon_groups.values() if len(g) == 1)

    # ---- comprehension
    s_hat = encode_all(model, vocab, forms, args.device, args.batch_size)
    comp = comprehension_metrics(s_hat.cpu(), bank_raw.cpu(), idx_sel,
                                 args.batch_size)

    # ---- distribution shift diagnostics
    glove = bank_raw[idx_sel].cpu()
    sh = s_hat.cpu()
    ds_cos = F.cosine_similarity(sh, glove, dim=-1).numpy()
    ds_l2 = (sh - glove).norm(dim=-1).numpy()
    ds_mse = ((sh - glove) ** 2).mean(dim=-1).numpy()
    norm_shat = sh.norm(dim=-1).numpy()
    norm_glove = glove.norm(dim=-1).numpy()

    # ---- naming: A = raw GloVe, B = s_hat; same items, same procedure
    max_steps = cfg.data.max_phonemes + 1     # documented global cap (see module docstring)
    naming: Dict[str, dict] = {}
    for cond, sem in (("glove", glove), ("shat", sh)):
        preds_all: List[List[int]] = []
        eos_all: List[bool] = []
        for lo in range(0, len(items), args.batch_size):
            p, e = semantic_greedy_decode(
                model, sem[lo:lo + args.batch_size].to(args.device), vocab, max_steps)
            preds_all += p
            eos_all += e
        tf_all: List[bool] = []
        for lo in range(0, len(items), args.batch_size):
            tf_all += teacher_forced_naming(
                model, sem[lo:lo + args.batch_size].to(args.device),
                forms[lo:lo + args.batch_size], vocab)
        naming[cond] = {"preds": preds_all, "eos": eos_all, "tf": tf_all}

    # ---- per-item rows
    rows: List[dict] = []
    for k, (i, e) in enumerate(zip(idx_sel, items)):
        gsize = group_size[i]
        top1_i = int(comp["top1_idx"][k])
        r = {
            "bank_index": i,
            "word": e.word,
            "phonology": " ".join(vocab.itos[p] for p in e.phonemes),
            "length": len(e.phonemes),
            "freq_rank": e.rank,
            "is_homophone": int(gsize > 1),
            "homophone_group_size": gsize,
            # comprehension (primary)
            "comp_target_cos": float(comp["target_cos"][k]),
            "comp_target_rank": int(comp["target_rank"][k]),
            "comp_top1": int(comp["top1"][k]),
            "comp_top5": int(comp["top5"][k]),
            "comp_margin": float(comp["margin"][k]),
            # comprehension (auxiliary)
            "comp_c_ltm": float(comp["c_ltm"][k]),
            "comp_top1_word": entries[top1_i].word,
            "comp_top1_same_phonology_aux": int(
                tuple(entries[top1_i].phonemes) == tuple(e.phonemes)),
            # distribution shift
            "shift_cos_shat_glove": float(ds_cos[k]),
            "shift_l2": float(ds_l2[k]),
            "shift_mse": float(ds_mse[k]),
            "norm_shat": float(norm_shat[k]),
            "norm_glove": float(norm_glove[k]),
        }
        for cond in ("glove", "shat"):
            pred = naming[cond]["preds"][k]
            r[f"naming_{cond}_pred"] = " ".join(vocab.itos[p] for p in pred)
            r[f"naming_{cond}_exact"] = int(pred == e.phonemes)
            r[f"naming_{cond}_edit"] = _edit_distance(pred, e.phonemes)
            r[f"naming_{cond}_pred_len"] = len(pred)
            r[f"naming_{cond}_eos_emitted"] = int(naming[cond]["eos"][k])
            r[f"naming_{cond}_tf_exact_diag"] = int(naming[cond]["tf"][k])
        rows.append(r)

    # ---- aggregates
    strict = [r for r in rows if not r["is_homophone"]]
    homo = [r for r in rows if r["is_homophone"]]
    comp_keys = ("comp_target_cos", "comp_target_rank", "comp_top1", "comp_top5",
                 "comp_margin", "comp_c_ltm", "comp_top1_same_phonology_aux")
    naming_keys = lambda c: (f"naming_{c}_exact", f"naming_{c}_edit",  # noqa: E731
                             f"naming_{c}_eos_emitted", f"naming_{c}_tf_exact_diag")
    summary = {
        "probe": "naming_comprehension_frozen_phase1a",
        "n_lexicon": n_lex,
        "n_probe_items": len(rows),
        "n_unique_phonology_lexicon": n_unique_phonology,
        "n_homophone_words_lexicon": sum(1 for g in phon_groups.values()
                                         if len(g) > 1 for _ in g),
        "comprehension": {
            "strict_unique_phonology": _agg(strict, comp_keys),
            "homophones_separate": _agg(homo, comp_keys),
            "all_items": _agg(rows, comp_keys),
            "note": ("strict population = unique-phonology words; homophone-class "
                     "top-1 (comp_top1_same_phonology_aux) is auxiliary, NOT "
                     "semantic comprehension accuracy"),
        },
        "naming": {
            cond: {**_agg(rows, naming_keys(cond)),
                   "whole_word_error_rate":
                       1.0 - float(np.mean([r[f"naming_{cond}_exact"] for r in rows]))}
            for cond in ("glove", "shat")
        },
        "naming_convention": {
            "decoding": "greedy AR from BOS until EOS or global cap",
            "global_cap_steps": max_steps,
            "cap_derivation": "cfg.data.max_phonemes + 1 (longest supported word + EOS)",
            "teacher_forced_is_diagnostic_only": True,
            "forced_length_ar": "not run (existing implementation is phoneme-input based)",
        },
        "distribution_shift": {
            "cos_shat_glove_mean": float(np.mean(ds_cos)),
            "cos_shat_glove_median": float(np.median(ds_cos)),
            "l2_mean": float(np.mean(ds_l2)),
            "mse_mean": float(np.mean(ds_mse)),
            "norm_shat_mean": float(np.mean(norm_shat)),
            "norm_glove_mean": float(np.mean(norm_glove)),
        },
        "provenance": {
            "checkpoint_path": os.path.abspath(args.ckpt),
            "checkpoint_sha256": sha256_file(args.ckpt),
            "checkpoint_training_commit": ckpt.get("git_commit"),
            "eval_git": git_state(ROOT),
            "lexicon_file_sha256": ckpt.get("lexicon_file_sha256"),
            "ordered_training_words_sha256": ckpt.get("ordered_training_words_sha256"),
            "ltm_encoder_mode": cfg.ltm.ltm_encoder_mode,
            "device": args.device,
            "limit": args.limit,
            "include_words": args.include_words,
            "runtime_seconds": round(time.time() - t0, 1),
        },
    }

    os.makedirs(args.out_dir, exist_ok=True)
    tsv_path = os.path.join(args.out_dir, "per_item.tsv")
    cols = list(rows[0].keys())
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    json_path = os.path.join(args.out_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[frozen_probe] {len(rows)} items -> {tsv_path}")
    print(f"[frozen_probe] summary -> {json_path}")
    print(json.dumps({k: summary[k] for k in
                      ("comprehension", "naming", "distribution_shift")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
