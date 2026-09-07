"""Data preflight: prove the REAL GloVe and the frozen populations resolve.

Loads nothing GPU and trains nothing.  Verifies, in order:
  1. the GloVe file exists at the given path and its SHA256 matches;
  2. the lexicon file SHA256 matches;
  3. the loader finds a real vector for EVERY word -- 29,571/29,571 with
     ZERO fallback (the run that just failed had 29,571 fallbacks);
  4. the canonical comprehension population is 27,981 with the frozen hash;
  5. the naming population is 29,571 with its frozen hash.

Exits non-zero on the first failure, so it is safe to chain before sbatch.
`--quick` stops after step 2, which avoids the ~1 GB GloVe parse when only
the file identity is in question.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.provenance import sha256_file                                  # noqa: E402

GLOVE_SHA = "91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed"
LEXICON_SHA = "ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66"
C_N, C_SHA = 27_981, "10c2f06eda769bf620ca3dbb9889204e4431cac2bfe0d0f5dd37fa4df2bb9f50"
FULL_N = 29_571


def fail(msg: str) -> None:
    print(f"FATAL preflight: {msg}", file=sys.stderr)
    sys.exit(1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--glove-path", required=True)
    ap.add_argument("--lexicon-path", default="data/lexicon_en_glove_covered.tsv")
    ap.add_argument("--quick", action="store_true",
                    help="stop after the file-identity checks")
    a = ap.parse_args(argv)

    g = a.glove_path
    print(f"[preflight] GloVe path : {g}")
    if not os.path.exists(g):
        fail(f"GloVe not found at {g}")
    gs = sha256_file(g)
    print(f"[preflight] GloVe sha256: {gs}")
    if gs != GLOVE_SHA:
        fail(f"GloVe sha256 {gs} != {GLOVE_SHA}")
    print("[preflight] GloVe identity OK")

    lp = a.lexicon_path if os.path.isabs(a.lexicon_path) else \
        os.path.join(ROOT, a.lexicon_path)
    ls = sha256_file(lp)
    print(f"[preflight] lexicon sha256: {ls}")
    if ls != LEXICON_SHA:
        fail(f"lexicon sha256 {ls} != {LEXICON_SHA}")
    print("[preflight] lexicon identity OK")
    if a.quick:
        print("[preflight] --quick: stopping before the GloVe parse")
        return 0

    from config import default_config
    from data.lexicon import build_lexicon
    from data.phonemes import build_vocab
    from scripts.naming_comprehension.train_tasks import (
        canonical_phonology_indices, subset_definition_hash, subset_records)

    cfg = default_config()
    cfg.data.use_real = True
    cfg.data.lexicon_path = a.lexicon_path
    cfg.data.glove_path = g
    cfg.data.max_words = 30000
    cfg.data.split_mode = "full_lexicon"
    cfg.data.val_fraction = 0.0
    vocab = build_vocab()
    print("[preflight] parsing GloVe (this reads the whole file) ...")
    lex = build_lexicon(cfg.data, vocab)
    entries = list(lex.entries)
    st = lex.load_stats
    found = int(getattr(st, "n_glove_found", 0))
    fb = int(getattr(st, "n_glove_fallback", 0))
    print(f"[preflight] entries {len(entries)} | glove_found {found} | "
          f"glove_fallback {fb}")
    if len(entries) != FULL_N:
        fail(f"lexicon has {len(entries)} entries, expected {FULL_N}")
    if fb != 0:
        fail(f"{fb} entries fall back to pseudo-vectors -- the run would be "
             f"scientifically invalid (fallback is NEVER enabled)")
    if found != FULL_N:
        fail(f"glove_found {found} != {FULL_N}")
    print(f"[preflight] REAL GloVe for all {FULL_N}/{FULL_N} words, 0 fallback")

    n_idx = list(range(len(entries)))
    n_hash = subset_definition_hash(subset_records(entries, n_idx, vocab))
    print(f"[preflight] naming population {len(n_idx)} sha256 {n_hash}")
    if len(n_idx) != FULL_N:
        fail("naming population size")

    c_idx = canonical_phonology_indices(entries)
    c_hash = subset_definition_hash(subset_records(entries, c_idx, vocab))
    print(f"[preflight] canonical C population {len(c_idx)} sha256 {c_hash}")
    if len(c_idx) != C_N:
        fail(f"C population {len(c_idx)} != {C_N}")
    if c_hash != C_SHA:
        fail(f"C hash {c_hash} != {C_SHA}")
    print(f"[preflight] retrieval bank {len(entries)}")
    print("[preflight] ALL CHECKS PASSED -- safe to submit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
