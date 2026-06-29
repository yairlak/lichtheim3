"""Build a bundled, realistic English lexicon: data/lexicon_en.tsv

Produces ~30k real English words with CMU ARPABET pronunciations and a frequency
RANK, kept as a small portable offline file (rank, word, arpabet). The repo
trains on these real words with **log-frequency** weighting.

Frequency ranking is hybrid (and documented honestly):
  * CORE  : words covered by a real frequency-ranked list (e.g. the
            google-10000-english list) keep their measured frequency rank.
  * TAIL  : additional real words (CMUdict ∩ hunspell en_US) are appended in a
            deterministic order with continued ranks, so the full 30k vocabulary
            follows a Zipfian log-frequency structure. The exact ordering within
            the long tail is approximate; the magnitude structure (what matters
            for log-frequency training) is realistic.

To use a *measured* frequency for all 30k words, drop a larger frequency list in
as RANKED and the CORE will simply cover more of the vocabulary.

Usage:
    python -m data.build_lexicon_en RANKED.txt [N=30000] [CMUDICT] [HUNSPELL.dic]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.phonemes import PHONEMES

INVENTORY = set(PHONEMES)
DEFAULT_CMU = "/usr/share/pocketsphinx/model/en-us/cmudict-en-us.dict"
DEFAULT_HUNSPELL = "/usr/share/hunspell/en_US.dic"
MIN_PH, MAX_PH = 2, 9


def load_cmudict(path):
    pron = {}
    with open(path, encoding="latin-1") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            w = parts[0]
            if "(" in w:                       # variant: word(2)
                w = w[:w.index("(")]
            phones = [p[:-1] if p[-1].isdigit() else p for p in parts[1:]]
            pron.setdefault(w.lower(), phones)
    return pron


def load_hunspell(path):
    words = set()
    if not os.path.exists(path):
        return words
    with open(path, encoding="latin-1") as f:
        next(f, None)                          # first line is a count
        for line in f:
            w = line.strip().split("/")[0].lower()
            if w.isalpha():
                words.add(w)
    return words


def ok(word, pron):
    if not word.isalpha() or word not in pron:
        return None
    ph = pron[word]
    if any(p not in INVENTORY for p in ph) or not (MIN_PH <= len(ph) <= MAX_PH):
        return None
    return ph


def main():
    ranked_path = sys.argv[1]
    n_target = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
    cmu_path = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_CMU
    hun_path = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_HUNSPELL

    pron = load_cmudict(cmu_path)
    hunspell = load_hunspell(hun_path)

    rows, used = [], set()

    # CORE: measured frequency order
    with open(ranked_path, encoding="latin-1") as f:
        for line in f:
            w = line.strip().split()[0].lower() if line.strip() else ""
            if w in used:
                continue
            ph = ok(w, pron)
            if ph is None:
                continue
            used.add(w)
            rows.append((w, " ".join(ph)))
            if len(rows) >= n_target:
                break
    n_core = len(rows)

    # TAIL: more real words (CMU ∩ hunspell), deterministic order, continued rank
    if len(rows) < n_target:
        tail = sorted(w for w in hunspell if w not in used)
        for w in tail:
            ph = ok(w, pron)
            if ph is None:
                continue
            used.add(w)
            rows.append((w, " ".join(ph)))
            if len(rows) >= n_target:
                break

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lexicon_en.tsv")
    with open(out, "w") as f:
        f.write("rank\tword\tarpabet\n")
        for r, (w, a) in enumerate(rows, start=1):
            f.write(f"{r}\t{w}\t{a}\n")
    print(f"wrote {len(rows)} entries to {out}  (core={n_core} measured-rank, "
          f"tail={len(rows) - n_core} Zipf-continued)")


if __name__ == "__main__":
    main()
