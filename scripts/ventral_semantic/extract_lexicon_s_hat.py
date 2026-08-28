"""Frozen extraction of raw s_hat for the whole training lexicon.

Encoder-only: for each of the two canonical X=5 checkpoints, the LTM encoder and
`to_semantic` are run over all 29,571 training words, in the exact order of
`train_entries` — which is the order of the semantic bank rows, so s_hat row i
and GloVe row i are the same word by construction.

No decoder is executed, no token is generated, no gradient is taken, nothing is
trained.  This is a new diagnostic execution and is provenanced as such.

Reuses `scripts.external_eval.load_model_and_vocab` (canonical checkpoint loader,
rebuilds the non-persistent semantic bank) and
`scripts.length_effect_analysis.bank_audit.build_bank` (bank + train_entries in
canonical order).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Dict, List

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.external_eval import load_model_and_vocab                # noqa: E402
from scripts.length_effect_analysis.bank_audit import build_bank      # noqa: E402
from evaluate.hooks import make_batch                                 # noqa: E402

# Canonical X=5 stable-zero cohort (audit Phase A): first checkpoint of the first
# run of 5 consecutive perfect (n_errors_full == 0) evaluations.
COHORT = {
    19: ("selected_checkpoints/seed_19_epoch_0155.pt", 155,
         "7d05f9c2ad5a53e705f7d55ccde2581754918938d8ca888da35c0a859666478e"),
    22: ("selected_checkpoints/seed_22_epoch_0140.pt", 140,
         "a15846cbf3c7df88ed289512bbb20cbefd2121d0deec1b39f363932a743da595"),
}
BUNDLE = ("archives/fulllexicon_93a577f/extracted/"
          "fulllexicon_final_bundle_93a577f")
N_WORDS = 29571
SEMANTIC_DIM = 300
BATCH = 256
OUT = os.path.join(ROOT, "outputs/ventral_semantic_93a577f")


def sha_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def git(*a) -> str:
    return subprocess.run(("git",) + a, cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def extract_one(seed: int) -> Dict[str, object]:
    rel, epoch, exp_sha = COHORT[seed]
    ck_path = os.path.join(ROOT, BUNDLE, rel)
    assert sha_file(ck_path) == exp_sha, f"checkpoint SHA mismatch seed {seed}"

    # canonical bank + entries, in training order
    ck, vocab, entries, bank_raw = build_bank(ck_path)
    assert len(entries) == N_WORDS, len(entries)
    assert ck["n_glove_found"] == N_WORDS and ck["n_glove_fallback"] == 0
    # GloVe must be genuinely available in THIS worktree, not silently faked
    assert bank_raw.shape == (N_WORDS, SEMANTIC_DIM), bank_raw.shape

    model, vocab2, meta = load_model_and_vocab(ck_path, "cpu")
    model.eval()
    assert vocab2.itos == vocab.itos
    # the model's own bank must equal the normalised entry vectors, same order
    bank_model = model.ltm.semantic_bank.detach().cpu().numpy()
    bank_norm = (bank_raw / bank_raw.norm(dim=-1, keepdim=True)).numpy()
    assert bank_model.shape == (N_WORDS, SEMANTIC_DIM)
    assert np.allclose(bank_model, bank_norm, atol=1e-6), \
        "model bank differs from the rebuilt normalised entry vectors"

    words = [e.word for e in entries]
    forms = [list(e.phonemes) for e in entries]
    assert all(len(f) > 0 for f in forms)

    s_hat = np.empty((N_WORDS, SEMANTIC_DIM), dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for start in range(0, N_WORDS, BATCH):
            chunk = forms[start:start + BATCH]
            batch = make_batch(chunk, vocab, "cpu")
            out = model.ltm.encode(batch["enc_in"], batch["enc_mask"])
            s_hat[start:start + len(chunk)] = out.cpu().numpy()
    assert np.isfinite(s_hat).all()

    # determinism: a re-run of one batch must be bit-identical
    with torch.inference_mode():
        b = make_batch(forms[:BATCH], vocab, "cpu")
        again = model.ltm.encode(b["enc_in"], b["enc_mask"]).cpu().numpy()
    assert np.array_equal(again, s_hat[:BATCH]), "extraction is not deterministic"

    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, f"s_hat_lexicon_seed{seed}.npy"), s_hat)
    if seed == min(COHORT):
        np.save(os.path.join(OUT, "glove_bank_normalised.npy"),
                bank_norm.astype(np.float32))
        with open(os.path.join(OUT, "lexicon_words.txt"), "w") as f:
            f.write("\n".join(words) + "\n")
    return {
        "seed": seed, "epoch": epoch, "checkpoint_sha256": exp_sha,
        "n_words": N_WORDS, "s_hat_shape": list(s_hat.shape),
        "dtype": str(s_hat.dtype),
        "n_glove_found": int(ck["n_glove_found"]),
        "n_glove_fallback": int(ck["n_glove_fallback"]),
        "glove_file_present_in_this_worktree": os.path.exists(
            os.path.join(ROOT, "data/glove.6B.300d.txt")),
        "bank_matches_rebuilt_entries": True,
        "elapsed_seconds": round(time.time() - t0, 1),
        "words_sha256": hashlib.sha256("\n".join(words).encode()).hexdigest(),
    }


def main() -> int:
    recs, word_hashes = [], set()
    for seed in sorted(COHORT):
        r = extract_one(seed)
        recs.append(r)
        word_hashes.add(r["words_sha256"])
        print(f"seed {seed}: s_hat {r['s_hat_shape']} in {r['elapsed_seconds']}s")
    # both checkpoints must enumerate the lexicon in the identical order
    assert len(word_hashes) == 1, "word order differs between checkpoints"

    prov = {
        "phase": "Phase B - frozen lexicon-scale s_hat extraction",
        "execution_type": ("encoder-only representation extraction with no "
                           "decoder execution"),
        "cohort_rule": ("X=5 stable zero: first checkpoint of the first run of "
                        "5 consecutive perfect (n_errors_full == 0) "
                        "evaluations, FULL route, deterministic AR decoding "
                        "over the 29,571-word training lexicon"),
        "checkpoints": recs,
        "decoder_executed": False, "tokens_generated": False,
        "training_performed": False, "weights_modified": False,
        "gradients_taken": False, "model_eval_mode": True,
        "inference_mode": True,
        "noise": {"ltm_ventral_noise": 0.0, "wm_interference_noise": 0.0},
        "s_hat_is_raw": True,
        "row_correspondence": ("s_hat row i, glove_bank_normalised row i and "
                               "lexicon_words.txt line i are the same word; "
                               "order = train_entries = semantic bank order"),
        "bank_normalisation": "F.normalize(bank, dim=-1) in set_semantic_bank",
        "torch_version": torch.__version__,
        "repository_head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    }
    with open(os.path.join(OUT, "extraction_provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
        f.write("\n")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
