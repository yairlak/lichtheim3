"""M3 prerequisite: structural audit of the frozen lexical/GloVe bank.

Analysis-only.  Reconstructs the bank exactly as `external_eval.load_model_and_vocab`
does (same lexicon build, same split, same order) and characterises its identity,
duplicate and homophone structure.  **No model is constructed and no forward pass
is run**: this reads the lexicon, not the network.

The bank is a phonology-to-lexical-neighbourhood readout aligned to GloVe
vectors.  Nothing here describes it as conceptual comprehension, and no model
mechanism may be inferred from this audit.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from config import (Config, DataConfig, GatingConfig, LossConfig,  # noqa: E402
                    LTMConfig, TrainConfig, WMConfig,
                    get_effective_split_seed)
from data.lexicon import build_lexicon                             # noqa: E402
from data.phonemes import build_vocab                              # noqa: E402

DEFAULT_CK = ("archives/fulllexicon_93a577f/extracted/"
              "fulllexicon_final_bundle_93a577f/selected_checkpoints/"
              "seed_19_epoch_0155.pt")

# Deterministic tie policies, frozen here and reused by every M3 computation.
COSINE_TIE_POLICY = ("ties in cosine similarity are broken by ascending bank "
                     "row index (the ordered training-word order pinned by "
                     "ordered_training_words_sha256); torch.topk is stable for "
                     "this input and the ordering is re-imposed explicitly")
PHON_TIE_POLICY = ("ties in phonological edit distance are broken by ascending "
                   "bank row index, then by orthographic string; never by "
                   "cosine similarity, so the two orderings stay independent")


def build_bank(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = Config(data=DataConfig(**ck["cfg_data"]), wm=WMConfig(**ck["cfg_wm"]),
                 ltm=LTMConfig(**ck["cfg_ltm"]),
                 gating=GatingConfig(**ck["cfg_gating"]),
                 loss=LossConfig(**ck["cfg_loss"]),
                 train=TrainConfig(**ck["cfg_train"]))
    vocab = build_vocab()
    lex = build_lexicon(cfg.data, vocab)
    train_entries, _ = lex.split(cfg.data.val_fraction,
                                 get_effective_split_seed(cfg.data))
    bank = torch.stack([torch.tensor(e.semantic) for e in train_entries]).float()
    return ck, vocab, train_entries, bank


def phon_string(entry, vocab) -> str:
    return " ".join(vocab.itos[p] for p in entry.phonemes)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, DEFAULT_CK))
    ap.add_argument("--out_dir", default=os.path.join(
        ROOT, "outputs/length_effect_mechanism_93a577f/m3_lexical_attraction"))
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    ck, vocab, entries, bank = build_bank(args.ckpt)
    n = len(entries)
    words = [e.word for e in entries]
    phons = [phon_string(e, vocab) for e in entries]

    ordered_sha = hashlib.sha256("\n".join(words).encode()).hexdigest()
    matches_ckpt = ordered_sha == ck["ordered_training_words_sha256"]

    # ---- orthographic duplicates
    w_counts = collections.Counter(words)
    dup_words = {w: c for w, c in w_counts.items() if c > 1}

    # ---- phonological duplicates (homophone groups)
    p_groups = collections.defaultdict(list)
    for i, p in enumerate(phons):
        p_groups[p].append(i)
    homophone_groups = {p: idx for p, idx in p_groups.items() if len(idx) > 1}
    n_homophone_rows = sum(len(v) for v in homophone_groups.values())

    # ---- duplicate GloVe rows (exact vector equality on the raw bank)
    b = bank.numpy()
    key = [hashlib.sha1(np.ascontiguousarray(r).tobytes()).hexdigest() for r in b]
    v_counts = collections.Counter(key)
    dup_vec_keys = {k: c for k, c in v_counts.items() if c > 1}
    n_dup_vec_rows = sum(dup_vec_keys.values())

    # ---- normalisation check
    norms = np.linalg.norm(b, axis=1)
    zero_rows = int((norms == 0).sum())

    rows = []
    for p, idx in sorted(homophone_groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        rows.append({"group_type": "HOMOPHONE", "key": p, "n_members": len(idx),
                     "member_rows": ";".join(str(i) for i in idx),
                     "member_words": ";".join(words[i] for i in idx)})
    for w, c in sorted(dup_words.items(), key=lambda kv: (-kv[1], kv[0])):
        idx = [i for i, x in enumerate(words) if x == w]
        rows.append({"group_type": "DUPLICATE_ORTHOGRAPHY", "key": w,
                     "n_members": c, "member_rows": ";".join(map(str, idx)),
                     "member_words": ";".join(words[i] for i in idx)})

    tsv = os.path.join(args.out_dir, "bank_structure_audit.tsv")
    import csv
    with open(tsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group_type", "key", "n_members",
                                          "member_rows", "member_words"],
                           delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = {
        "bank_rows": n,
        "unique_orthographic_entries": len(set(words)),
        "unique_phonological_forms": len(set(phons)),
        "duplicate_orthographic_entries": len(dup_words),
        "duplicate_orthographic_rows": sum(dup_words.values()),
        "homophone_groups": len(homophone_groups),
        "rows_in_homophone_groups": n_homophone_rows,
        "duplicate_glove_vector_groups": len(dup_vec_keys),
        "duplicate_glove_vector_rows": n_dup_vec_rows,
        "zero_norm_rows": zero_rows,
        "ordered_training_words_sha256": ordered_sha,
        "matches_checkpoint_hash": bool(matches_ckpt),
        "bank_rows_equal_training_entries": True,
        "multiple_bank_entries_can_share_one_phonological_form":
            bool(homophone_groups),
        "cosine_tie_policy": COSINE_TIE_POLICY,
        "phonological_distance_tie_policy": PHON_TIE_POLICY,
        "model_forward_called": False,
    }
    with open(os.path.join(args.out_dir, "bank_structure_audit.json"), "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    md = [
        "# M3 prerequisite — frozen lexical/GloVe bank: structural audit\n",
        "Analysis-only. The bank was reconstructed exactly as the evaluator "
        "builds it (`build_lexicon` → `Lexicon.split(val_fraction, "
        "effective_split_seed)` → row order preserved). **No model was "
        "constructed and no forward pass was run.**\n",
        "## Identity\n",
        f"- Bank rows: **{n:,}**",
        f"- Rows equal training entries: **yes** (`n_train = "
        f"{ck['n_train']:,}`, `n_val = {ck['n_val']}`)",
        f"- `ordered_training_words_sha256` recomputed: `{ordered_sha}`",
        f"- Matches the value stored in the checkpoint: "
        f"**{'YES' if matches_ckpt else 'NO'}**",
        "- Row → word mapping is the ordered training-word list; row → "
        "phonological form comes from the same `LexEntry` objects.\n",
        "## Duplicate and homophone structure\n",
        f"- Unique orthographic entries: **{len(set(words)):,}**",
        f"- Duplicate orthographic entries: **{len(dup_words)}** "
        f"({sum(dup_words.values())} rows)",
        f"- Unique phonological forms: **{len(set(phons)):,}**",
        f"- Homophone groups (>1 row sharing a phonological form): "
        f"**{len(homophone_groups):,}**, covering **{n_homophone_rows:,}** rows",
        f"- Duplicate GloVe vector groups: **{len(dup_vec_keys)}** "
        f"({n_dup_vec_rows} rows)",
        f"- Zero-norm rows: **{zero_rows}**\n",
        "**Multiple bank entries can share one phonological form: "
        f"{'YES' if homophone_groups else 'NO'}.** This matters for M3: a "
        "prediction matching a phonological form does not identify a unique "
        "bank row, so phonological-match categories are defined over "
        "*phonological forms*, not over rows.\n",
        "## Deterministic tie policies (frozen)\n",
        f"- **Equal cosine similarities**: {COSINE_TIE_POLICY}.",
        f"- **Equal phonological distances**: {PHON_TIE_POLICY}.\n",
        "## Scope\n",
        "This audit describes the bank's structure only. **No model mechanism "
        "may be inferred from it**, and the bank is not a model of conceptual "
        "comprehension — it is a lexical-neighbourhood readout over GloVe "
        "vectors of the training lexicon.\n",
    ]
    with open(os.path.join(args.out_dir, "bank_structure_audit.md"), "w") as f:
        f.write("\n".join(md))

    print(json.dumps(summary, indent=2)[:1200])
    print(f"\nwrote {tsv} ({len(rows)} group rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
