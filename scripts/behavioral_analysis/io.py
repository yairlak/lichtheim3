"""Data loading and deterministic table writing.  No model is ever loaded."""
from __future__ import annotations

import hashlib
import os
from typing import Iterable, Optional

import pandas as pd

from .common import (ANALYSIS_ROOT, CANONICAL_TABLE, CLEAN_EXPOSURES,
                     EXPECTED_CANONICAL_ROWS, EXPECTED_CLEAN_COUNTS, ROUTES,
                     SEEDS)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_canonical(path: Optional[str] = None,
                   validate: bool = True) -> pd.DataFrame:
    """Load the canonical seed x item x route table."""
    df = pd.read_csv(path or CANONICAL_TABLE, sep="\t")
    if validate:
        if len(df) != EXPECTED_CANONICAL_ROWS:
            raise ValueError(f"canonical table has {len(df)} rows, "
                             f"expected {EXPECTED_CANONICAL_ROWS}")
        if sorted(df["seed"].unique()) != SEEDS:
            raise ValueError(f"seeds {sorted(df['seed'].unique())} != {SEEDS}")
        if sorted(df["route"].unique()) != sorted(ROUTES):
            raise ValueError("unexpected route set")
        if df.duplicated(["seed", "item_id", "route"]).any():
            raise ValueError("duplicate seed x item_id x route rows")
    return df


def clean_subset(canon: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
    """LICHTHEIM_CLEAN rows (671 trained real + 391 novel pseudo)."""
    clean = canon[canon["in_LICHTHEIM_CLEAN"]].copy()
    if validate:
        one = clean[(clean["seed"] == SEEDS[0]) & (clean["route"] == "full")]
        counts = one["lichtheim_exposure_status"].value_counts().to_dict()
        for lex, status in CLEAN_EXPOSURES.items():
            want = EXPECTED_CLEAN_COUNTS[lex]
            got = counts.get(status, 0)
            if got != want:
                raise ValueError(f"clean {lex}: {got} items, expected {want}")
        if len(one) != EXPECTED_CLEAN_COUNTS["total"]:
            raise ValueError(f"clean total {len(one)} != "
                             f"{EXPECTED_CLEAN_COUNTS['total']}")
    return clean


def analysis_table(*parts: str) -> str:
    return os.path.join(ANALYSIS_ROOT, *parts)


def write_table(df: pd.DataFrame, path: str,
                sort_by: Optional[Iterable[str]] = None) -> str:
    """Write a TSV with deterministic row order and no index.

    Deterministic ordering is what makes byte-identical regeneration possible,
    so every plotting table goes through here.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = df.sort_values(list(sort_by)).reset_index(drop=True) if sort_by else df
    out.to_csv(path, sep="\t", index=False)
    return path
