"""Read-only access to the immutable closed ventral-interface controls (contract §3)."""
from __future__ import annotations

import csv
import hashlib
import os
from typing import Dict, List

from ventral_directional_dose import CONVENTIONS

IDENTITY_FIELDS = [
    "witness_id", "state_id", "source_or_repaired", "seed", "source_u",
    "source_checkpoint_sha256", "repaired_head_sha256", "reconstructed_state_identity",
    "item_index", "lexical_identity", "canonical_C_index", "canonical_C_identity",
    "in_C_population", "target_phonology", "phoneme_length", "homophone_group",
    "homophone_group_size", "retrieved_index", "retrieved_lexical_identity",
    "retrieved_phonology", "C_contract_correct", "lexical_identity_correct", "phonology_correct",
]
CONTROL_FIELDS = [f"S{k}_{c}_exact_correct" for k in range(4) for c in CONVENTIONS]
S0_PHONOLOGY_FIELDS = [f"S0_{c}_predicted_phonology" for c in CONVENTIONS]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_immutable_controls(path: str, expected_sha256: str) -> Dict[str, List[Dict[str, str]]]:
    """state_id -> rows ordered by item_index; hash-checked before reading."""
    got = sha256_file(path)
    if got != expected_sha256:
        raise RuntimeError(f"HARD STOP DOSE-A/H: immutable controls {path} sha256 {got} != {expected_sha256}")
    keep = IDENTITY_FIELDS + CONTROL_FIELDS + S0_PHONOLOGY_FIELDS
    out: Dict[str, List[Dict[str, str]]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f, delimiter="\t")
        hdr = next(rd)
        idx = [hdr.index(k) for k in keep]
        for row in rd:
            rec = {k: row[i] for k, i in zip(keep, idx)}
            for c in CONVENTIONS:
                rec[f"prev_S1_rescue_{c}"] = str(int(rec[f"S0_{c}_exact_correct"] == "0"
                                                     and rec[f"S1_{c}_exact_correct"] == "1"))
            out.setdefault(rec["state_id"], []).append(rec)
    for sid, rows in out.items():
        rows.sort(key=lambda r: int(r["item_index"]))
        if [int(r["item_index"]) for r in rows] != list(range(len(rows))):
            raise RuntimeError(f"HARD STOP DOSE-A: immutable controls for {sid} are not a complete item range")
    return out
