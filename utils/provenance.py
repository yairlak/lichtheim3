"""Provenance helpers: SHA256 hashes, word-list fingerprints, and git state.

Hash convention for word lists (used identically in training and evaluation):
    - UTF-8 encoded
    - one word per line
    - LF separator
    - NO trailing LF

This convention must not change once checkpoints reference these hashes.

Reference values for data/lexicon_en_glove_covered.tsv (29 571 words):
    lexicon_file_sha256   = ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66
    ordered_words_sha256  = 0cb1c6172a7c2aea8a503549ffdf32543da820e1c505e0885f3999d6e50f7fa1
    sorted_words_sha256   = f9721c17f97d0f2a1afeb97ce917075db202f21ad471dcb8755a26226de7d63a
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import FrozenSet, List, Optional

# ---------------------------------------------------------------------------
# Provenance schema version
# ---------------------------------------------------------------------------

# Bump when the set of required fields or their semantics changes.
#   absent → legacy checkpoint; strict provenance unavailable.
#   1      → all V1_REQUIRED_PROVENANCE_FIELDS present and verified.
#   other  → fail-fast (unknown schema, produced by newer code).
PROVENANCE_SCHEMA_VERSION: int = 1

# Fields whose presence is required in a schema_version=1 checkpoint.
# Absence of any of these is a hard error at resume or evaluation.
V1_REQUIRED_PROVENANCE_FIELDS: FrozenSet[str] = frozenset({
    "split_mode", "train_all_words", "validation_enabled", "val_fraction",
    "n_source_rows", "n_entries_after_loading",
    "n_filtered_unknown_phoneme", "n_filtered_length", "n_unique_loaded_words",
    "n_train", "n_val", "n_unique_train_words",
    "n_glove_found", "n_glove_fallback",
    "lexicon_file_path", "glove_file_path",
    "lexicon_file_sha256", "ordered_training_words_sha256", "sorted_training_words_sha256",
    "split_seed_used", "split_seed_effective", "sampler_config",
})


def sha256_file(path: str) -> Optional[str]:
    """SHA256 of the raw file bytes. Returns None if the file does not exist."""
    p = Path(path)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha256_words_ordered(words: List[str]) -> str:
    """SHA256 of words in their given order.

    Convention: UTF-8, one word per line, LF separator, no trailing LF.
    An empty list produces the SHA256 of an empty byte string.
    """
    return hashlib.sha256("\n".join(words).encode("utf-8")).hexdigest()


def sha256_words_sorted(words: List[str]) -> str:
    """SHA256 of words sorted alphabetically.

    Same encoding convention as sha256_words_ordered.
    """
    return hashlib.sha256("\n".join(sorted(words)).encode("utf-8")).hexdigest()


def git_state(cwd: Optional[str] = None) -> dict:
    """Best-effort git state: commit, branch, dirty flags, untracked paths.

    All values fall back to 'unknown' if git is unavailable or times out.

    Returns a dict with:
        'commit'            (str)
        'branch'            (str)
        'dirty'             (bool | 'unknown')  — LEGACY: true if `git status
                            --porcelain` is non-empty, i.e. tracked changes OR
                            untracked files.  Unchanged semantics: existing
                            checkpoints (incl. the frozen 93a577f cohort)
                            recorded `git_dirty` with exactly this meaning.
        'tracked_dirty'     (bool | 'unknown')  — true iff a TRACKED file is
                            modified, staged or deleted
                            (`git status --porcelain --untracked-files=no`).
                            This is the flag that bears on code identity.
        'untracked_present' (bool | 'unknown')
        'untracked_paths'   (list[str])         — untracked entries as git
                            reports them, directories collapsed (e.g.
                            'archives/').  Never silently dropped: callers
                            decide which untracked paths matter to them.

    Separating the two is what lets an intentionally untracked data directory
    coexist with a precisely identified, committed source tree.
    """
    result: dict = {"commit": "unknown", "branch": "unknown", "dirty": "unknown",
                    "tracked_dirty": "unknown", "untracked_present": "unknown",
                    "untracked_paths": []}
    _timeout = 10

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, stderr=subprocess.DEVNULL, timeout=_timeout,
        )
        result["commit"] = out.decode().strip()
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=cwd, stderr=subprocess.DEVNULL, timeout=_timeout,
        )
        result["branch"] = out.decode().strip() or "unknown"
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=cwd, stderr=subprocess.DEVNULL, timeout=_timeout,
        )
        text = out.decode()
        result["dirty"] = bool(text.strip())
        untracked = [line[3:] for line in text.splitlines()
                     if line.startswith("?? ")]
        result["untracked_paths"] = untracked
        result["untracked_present"] = bool(untracked)
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd, stderr=subprocess.DEVNULL, timeout=_timeout,
        )
        result["tracked_dirty"] = bool(out.decode().strip())
    except Exception:
        pass

    return result
