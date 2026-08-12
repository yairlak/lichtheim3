"""Frozen extraction of the dorsal encoder state h_WM for the whole lexicon.

Executable definition of the tensor (models/wm_route.py:85-93):

    lengths = enc_mask.sum(1).clamp(min=1).cpu()
    packed  = pack_padded_sequence(phon_embed(enc_in), lengths,
                                   batch_first=True, enforce_sorted=False)
    _, h    = self.encoder(packed)          # h: (1, B, hidden)
    if (self.training or apply_noise) and self.cfg.interference_noise > 0:
        h = h + randn_like(h) * self.cfg.interference_noise
    return h

`h` is the word-level dorsal encoder state at each item's last valid token.  Both
canonical checkpoints have `interference_noise = 0.0` and are run in eval mode
with `apply_noise=False`, so the noise branch is dead twice over and the returned
tensor IS the deterministic pre-noise representation — the interpretation the
protocol asks for is correct for this architecture.

Word order is the `train_entries` order, identical to the s_hat extraction and to
the semantic-bank rows.  Encoder-only: no decoder, no gradient, no training.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Dict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluate.hooks import make_batch                                 # noqa: E402
from scripts.external_eval import load_model_and_vocab                # noqa: E402
from scripts.length_effect_analysis.bank_audit import build_bank      # noqa: E402
from scripts.ventral_semantic.extract_lexicon_s_hat import (          # noqa: E402
    BATCH, BUNDLE, COHORT, N_WORDS, OUT, sha_file)

WM_DIM = 128


def git(*a) -> str:
    return subprocess.run(("git",) + a, cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def extract_one(seed: int) -> Dict[str, object]:
    rel, epoch, exp_sha = COHORT[seed]
    ck_path = os.path.join(ROOT, BUNDLE, rel)
    assert sha_file(ck_path) == exp_sha, f"checkpoint SHA mismatch seed {seed}"

    ck, vocab, entries, _ = build_bank(ck_path)
    assert len(entries) == N_WORDS
    model, vocab2, _ = load_model_and_vocab(ck_path, "cpu")
    model.eval()
    assert vocab2.itos == vocab.itos
    assert model.wm.cfg.interference_noise == 0.0, "WM noise is not zero"
    assert model.wm.encoder.hidden_size == WM_DIM
    assert model.wm.encoder.bidirectional is False
    assert not model.training

    words = [e.word for e in entries]
    forms = [list(e.phonemes) for e in entries]

    h = np.empty((N_WORDS, WM_DIM), dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for start in range(0, N_WORDS, BATCH):
            chunk = forms[start:start + BATCH]
            batch = make_batch(chunk, vocab, "cpu")
            out = model.wm.encode(batch["enc_in"], batch["enc_mask"])
            assert out.shape == (1, len(chunk), WM_DIM), out.shape
            h[start:start + len(chunk)] = out.squeeze(0).cpu().numpy()
    assert np.isfinite(h).all()

    # determinism, and equality with an explicit apply_noise=False call
    with torch.inference_mode():
        b = make_batch(forms[:BATCH], vocab, "cpu")
        again = model.wm.encode(b["enc_in"], b["enc_mask"],
                                apply_noise=False).squeeze(0).cpu().numpy()
    assert np.array_equal(again, h[:BATCH]), "extraction is not deterministic"

    # row order must match the s_hat / GloVe extraction exactly
    ref = open(os.path.join(OUT, "lexicon_words.txt")).read().split("\n")
    ref = [w for w in ref if w]
    assert words == ref, "word order differs from the s_hat extraction"

    np.save(os.path.join(OUT, f"h_wm_lexicon_seed{seed}.npy"), h)
    return {"seed": seed, "epoch": epoch, "checkpoint_sha256": exp_sha,
            "h_wm_shape": list(h.shape), "dtype": str(h.dtype),
            "interference_noise": float(model.wm.cfg.interference_noise),
            "pre_noise_deterministic": True,
            "order_matches_s_hat_extraction": True,
            "elapsed_seconds": round(time.time() - t0, 1)}


def main() -> int:
    recs = [extract_one(s) for s in sorted(COHORT)]
    for r in recs:
        print(f"seed {r['seed']}: h_WM {r['h_wm_shape']} in {r['elapsed_seconds']}s")
    prov = {
        "phase": "C1 - frozen lexicon-scale h_WM extraction",
        "tensor": ("WMRecurrent.encode return value h, shape (1,B,128), "
                   "squeezed to (B,128); the word-level dorsal encoder state at "
                   "each item's last valid token via pack_padded_sequence"),
        "pre_noise": ("interference_noise == 0.0 in both checkpoints and "
                      "apply_noise=False in eval, so the noise branch is never "
                      "taken; the extracted tensor is the deterministic "
                      "pre-noise representation"),
        "execution_type": ("encoder-only representation extraction with no "
                           "decoder execution"),
        "checkpoints": recs,
        "decoder_executed": False, "tokens_generated": False,
        "training_performed": False, "weights_modified": False,
        "gradients_taken": False, "model_eval_mode": True,
        "inference_mode": True,
        "row_correspondence": ("row i is the same word as row i of "
                               "s_hat_lexicon_seed*.npy, "
                               "glove_bank_normalised.npy and "
                               "lexicon_words.txt"),
        "torch_version": torch.__version__,
        "repository_head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    }
    with open(os.path.join(OUT, "h_wm_extraction_provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
        f.write("\n")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
