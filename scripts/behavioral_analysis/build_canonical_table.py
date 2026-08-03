"""Rebuild the canonical seed x item x route table from production outputs.

    python -m scripts.behavioral_analysis.build_canonical_table [--out PATH]

Promoted unchanged from the validated driver; it reads only the enriched
production prediction tables and the frozen item manifest.  No model is loaded.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from scripts.behavioral_analysis.common import (ANALYSIS_SETS,        # noqa: E402
                                                CANONICAL_TABLE,
                                                EXPECTED_CANONICAL_ROWS,
                                                PRODUCTION_ROOT, ROUTES,
                                                SEED_EPOCHS, SEEDS,
                                                repo_relative)
from scripts.behavioral_analysis.io import write_table                # noqa: E402


def enriched_path(seed: int, production_root: str) -> str:
    return os.path.join(production_root, f"seed{seed}", "wfe_ar",
                        "item_level_predictions_enriched.tsv")


def build(production_root: str = PRODUCTION_ROOT) -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        d = pd.read_csv(enriched_path(seed, production_root), sep="\t")
        if len(d) != 1200:
            raise ValueError(f"seed {seed}: {len(d)} rows, expected 1200")
        for route in ROUTES:
            sub = pd.DataFrame({
                "seed": seed, "epoch": SEED_EPOCHS[seed],
                "item_id": d["item_id"], "route": route,
                "target": d[f"{route}_target"].astype(str),
                "prediction": d[f"{route}_predicted"].fillna("").astype(str),
                "exact_match": d[f"{route}_exact_match"].astype(int),
                "word_error": 1 - d[f"{route}_exact_match"].astype(int),
                "raw_edit_distance": d[f"{route}_edit_dist"].astype(float),
                "normalized_edit_distance": d[f"{route}_norm_edit"].astype(float),
                "insertions": d[f"{route}_insertions"].astype(int),
                "deletions": d[f"{route}_deletions"].astype(int),
                "substitutions": d[f"{route}_substitutions"].astype(int),
                "target_length": d["length_phonemes"].astype(int),
                "predicted_length": d[f"{route}_predicted_length"].astype(int),
                "eos_position": d[f"{route}_eos_position"],
                "source_lexicality": d["lexicality"].astype(str),
                "morphology": d["morphology"].astype(str),
                "size": d["size"].astype(str),
                "condition": d["condition"].astype(str),
                "zipf_frequency": d["zipf_frequency"],
                "frequency_class": d["frequency_class"].astype(str),
                "lichtheim_exposure_status":
                    d["lichtheim_exposure_status"].astype(str),
                "gate": pd.to_numeric(d["gate"], errors="coerce"),
                "lexical_confidence":
                    pd.to_numeric(d["lexical_confidence"], errors="coerce"),
                "margin": pd.to_numeric(d["lexical_margin"], errors="coerce"),
                "density": pd.to_numeric(d["lexical_density"], errors="coerce"),
            })
            for a in ANALYSIS_SETS:
                sub[f"in_{a}"] = d[f"in_{a}"].astype(str).str.lower() == "true"
            frames.append(sub)
    canon = pd.concat(frames, ignore_index=True)
    if len(canon) != EXPECTED_CANONICAL_ROWS:
        raise ValueError(f"{len(canon)} rows, expected {EXPECTED_CANONICAL_ROWS}")
    if canon.duplicated(["seed", "item_id", "route"]).any():
        raise ValueError("duplicate seed x item_id x route")
    bad = (canon["insertions"] + canon["deletions"] + canon["substitutions"]
           != canon["raw_edit_distance"]).sum()
    if bad:
        raise ValueError(f"{bad} rows where operations != edit distance")
    return canon


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--production_root", default=PRODUCTION_ROOT)
    ap.add_argument("--out", default=CANONICAL_TABLE)
    args = ap.parse_args(argv)
    canon = build(args.production_root)
    write_table(canon, args.out)
    print(f"-> {repo_relative(args.out)} ({len(canon)} rows, "
          f"{len(canon.columns)} cols)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
