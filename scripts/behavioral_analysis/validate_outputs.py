"""Structural validation of the behavioral analysis inputs and outputs.

    python -m scripts.behavioral_analysis.validate_outputs [--figures DIR]

Checks the canonical table, the clean-set composition and, when a figures
directory is given, the plotting tables that back each published figure.
Never loads a model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from scripts.behavioral_analysis.common import (BOOTSTRAP_REPLICATES,  # noqa: E402
                                                BOOTSTRAP_SEED,
                                                CANONICAL_TABLE,
                                                EXPECTED_CLEAN_COUNTS,
                                                LENGTHS, ROUTES, SEEDS)
from scripts.behavioral_analysis.io import (clean_subset,              # noqa: E402
                                            load_canonical)


def validate(canonical: str = CANONICAL_TABLE,
             figures_dir: str | None = None) -> dict:
    checks: dict = {}

    def ck(name, cond, detail=""):
        checks[name] = {"pass": bool(cond), "detail": str(detail)}

    canon = load_canonical(canonical, validate=False)
    ck("canonical_14400_rows", len(canon) == 14400, len(canon))
    ck("four_seeds", sorted(canon["seed"].unique()) == SEEDS)
    ck("three_routes", sorted(canon["route"].unique()) == sorted(ROUTES))
    ck("seed21_present", 21 in set(canon["seed"]))
    ck("no_duplicate_seed_item_route",
       not canon.duplicated(["seed", "item_id", "route"]).any())
    ck("ops_sum_to_edit_distance",
       bool((canon["insertions"] + canon["deletions"] + canon["substitutions"]
             == canon["raw_edit_distance"]).all()))

    clean = clean_subset(canon, validate=False)
    one = clean[(clean["seed"] == SEEDS[0]) & (clean["route"] == "full")]
    counts = one["lichtheim_exposure_status"].value_counts().to_dict()
    ck("clean_total_1062", len(one) == EXPECTED_CLEAN_COUNTS["total"], len(one))
    ck("clean_real_671",
       counts.get("TRAINED_REAL_EXACT", 0) == EXPECTED_CLEAN_COUNTS["real"],
       counts.get("TRAINED_REAL_EXACT", 0))
    ck("clean_pseudo_391",
       counts.get("NOVEL_PSEUDOWORD", 0) == EXPECTED_CLEAN_COUNTS["pseudo"],
       counts.get("NOVEL_PSEUDOWORD", 0))
    ck("clean_only_two_strata",
       set(one["lichtheim_exposure_status"]) ==
       {"TRAINED_REAL_EXACT", "NOVEL_PSEUDOWORD"})
    ck("clean_lengths_exclude_6",
       sorted(one["target_length"].unique()) == LENGTHS,
       sorted(one["target_length"].unique()))

    if figures_dir:
        for name, cols in (
            ("yair_clean_length_by_route.tsv",
             {"route", "source_lexicality", "phoneme_length", "seed",
              "mean_raw_edit_distance", "ci_low", "ci_high"}),
            ("clean_length_slopes_by_seed.tsv",
             {"seed", "source_lexicality", "route", "length_slope"}),
            ("clean_route_length_contrasts.tsv",
             {"seed", "source_lexicality", "ltm_minus_wm"}),
            ("yair_clean_serial_position_interpolated.tsv",
             {"route", "source_lexicality", "relative_position",
              "interpolated_error_rate"}),
            ("gate_by_clean_lexicality.tsv",
             {"seed", "source_lexicality", "mean_gate",
              "mean_lexical_confidence"}),
            ("gate_by_exposure_status.tsv",
             {"seed", "exposure_status", "mean_gate",
              "mean_lexical_confidence"}),
        ):
            p = os.path.join(figures_dir, name)
            if not os.path.exists(p):
                ck(f"table_present_{name}", False, "missing")
                continue
            df = pd.read_csv(p, sep="\t")
            ck(f"table_present_{name}", True)
            ck(f"table_columns_{name}", cols <= set(df.columns),
               str(sorted(cols - set(df.columns))))
            if "seed" in df.columns:
                ck(f"table_seeds_{name}",
                   sorted(df["seed"].unique()) == SEEDS,
                   sorted(df["seed"].unique()))
            if "phoneme_length" in df.columns:
                ck(f"table_lengths_{name}",
                   sorted(df["phoneme_length"].unique()) == LENGTHS,
                   sorted(df["phoneme_length"].unique()))
        b = os.path.join(figures_dir, "clean_bootstrap_results.tsv")
        if os.path.exists(b):
            bt = pd.read_csv(b, sep="\t")
            ck("bootstrap_replicates_recorded",
               int(bt["n_replicates"].iloc[0]) == BOOTSTRAP_REPLICATES)
            ck("bootstrap_seed_recorded",
               int(bt["random_seed"].iloc[0]) == BOOTSTRAP_SEED)

    n_fail = sum(1 for v in checks.values() if not v["pass"])
    return {"verdict": "PASS" if n_fail == 0 else "FAIL",
            "n_checks": len(checks), "n_failures": n_fail, "checks": checks,
            "model_inference_performed": False}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", default=CANONICAL_TABLE)
    ap.add_argument("--figures", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    res = validate(args.canonical, args.figures)
    for k, v in res["checks"].items():
        print(f"  [{'PASS' if v['pass'] else 'FAIL'}] {k}"
              f"{(' — ' + v['detail']) if v['detail'] else ''}")
    print(f"\nVERDICT: {res['verdict']} ({res['n_checks']} checks, "
          f"{res['n_failures']} failures)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
