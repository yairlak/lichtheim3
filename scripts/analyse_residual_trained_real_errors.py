"""Are the 12 residual TRAINED_REAL_EXACT LTM errors long and/or low-frequency?

Matched descriptive controls only.  With 12 unique error items a regression
would be uninterpretable, so each error item is compared against the correctly
repeated trained-real words **of its own exact phoneme length**, and its Zipf
frequency is expressed as a percentile within that stratum.

The reference test is a length-matched permutation: draw 12 trained-real items
with the same length composition as the observed error set and recompute the
mean within-stratum frequency percentile.  This asks exactly one question - are
these words unusually rare for their length - without fitting anything.

Reads only frozen tables.  No model is loaded.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CANON = os.path.join(
    ROOT, "outputs/behavioral_wfe_fulllexicon_93a577f/behavioral_analysis/"
          "tables/canonical_behavioral_item_table.tsv")
ERRORS = os.path.join(
    ROOT, "reports/behavioral_wfe_fulllexicon_93a577f/yair_corrections/tables/"
          "trained_real_exact_ltm_errors.tsv")
OUT = os.path.join(
    ROOT, "reports/behavioral_wfe_fulllexicon_93a577f/yair_corrections/"
          "residual_trained_real")
PERM_B = 20000
PERM_SEED = 20260809


def main() -> int:
    canon = pd.read_csv(CANON, sep="\t")
    err_tbl = pd.read_csv(ERRORS, sep="\t")
    tre = canon[(canon["route"] == "ltm")
                & (canon["lichtheim_exposure_status"] == "TRAINED_REAL_EXACT")]
    per = tre.groupby("item_id", as_index=False).agg(
        n_failing_seeds=("word_error", "sum"),
        phoneme_length=("target_length", "first"),
        zipf_frequency=("zipf_frequency", "first"),
        morphology=("morphology", "first"))
    per["is_error_item"] = per["n_failing_seeds"] > 0
    assert len(per) == 671
    assert int(per["is_error_item"].sum()) == len(err_tbl) == 12

    # ---- within-length frequency percentile of every error item
    rows = []
    for _, r in per[per["is_error_item"]].iterrows():
        peers = per[(per["phoneme_length"] == r["phoneme_length"])
                    & (~per["is_error_item"])]
        pct = float((peers["zipf_frequency"] < r["zipf_frequency"]).mean())
        w = err_tbl[err_tbl["item_id"] == r["item_id"]]
        rows.append({
            "item_id": r["item_id"],
            "word": w["word"].iloc[0] if len(w) else "",
            "phoneme_length": int(r["phoneme_length"]),
            "zipf_frequency": float(r["zipf_frequency"]),
            "n_correct_peers_same_length": int(len(peers)),
            "median_zipf_of_peers": float(peers["zipf_frequency"].median()),
            "zipf_percentile_within_length": pct,
            "below_peer_median": bool(pct < 0.5),
            "n_failing_seeds": int(r["n_failing_seeds"]),
            "morphology": r["morphology"],
            "total_substitutions": int(w["total_substitutions"].iloc[0]) if len(w) else -1,
            "total_deletions": int(w["total_deletions"].iloc[0]) if len(w) else -1,
            "total_insertions": int(w["total_insertions"].iloc[0]) if len(w) else -1,
        })
    items = pd.DataFrame(rows).sort_values(
        ["phoneme_length", "zipf_percentile_within_length"])

    # ---- length: error rate per exact length
    by_len = per.groupby("phoneme_length", as_index=False).agg(
        n_items=("item_id", "size"), n_error_items=("is_error_item", "sum"))
    by_len["error_item_rate"] = (by_len["n_error_items"]
                                 / by_len["n_items"])

    # ---- length-matched permutation on the frequency percentile
    rng = np.random.default_rng(PERM_SEED)
    comp = items["phoneme_length"].value_counts().to_dict()
    pools = {L: per[per["phoneme_length"] == L] for L in comp}
    # percentile is defined against non-error peers, as above
    peers = {L: per[(per["phoneme_length"] == L)
                    & (~per["is_error_item"])]["zipf_frequency"].to_numpy()
             for L in comp}
    obs = float(items["zipf_percentile_within_length"].mean())
    draws = np.empty(PERM_B)
    for b in range(PERM_B):
        vals = []
        for L, k in comp.items():
            z = rng.choice(pools[L]["zipf_frequency"].to_numpy(), size=k,
                           replace=False)
            vals.extend((peers[L][None, :] < z[:, None]).mean(axis=1))
        draws[b] = float(np.mean(vals))
    p_low = float((draws <= obs).mean())

    summary = pd.DataFrame([{
        "n_error_items": int(len(items)),
        "n_error_events_seed_x_item": int(items["n_failing_seeds"].sum()),
        "n_trained_real_items": int(len(per)),
        "mean_length_error_items": float(items["phoneme_length"].mean()),
        "mean_length_correct_items":
            float(per[~per["is_error_item"]]["phoneme_length"].mean()),
        "mean_zipf_error_items": float(items["zipf_frequency"].mean()),
        "mean_zipf_correct_items":
            float(per[~per["is_error_item"]]["zipf_frequency"].mean()),
        "mean_within_length_zipf_percentile": obs,
        "n_below_peer_median": int(items["below_peer_median"].sum()),
        "permutation_B": PERM_B, "permutation_seed": PERM_SEED,
        "permutation_p_one_sided_lower": p_low,
        "permutation_null_mean": float(draws.mean()),
        "control": ("length-matched: permutation preserves the observed length "
                    "composition (2 at L=5, 3 at L=7, 1 at L=8, 6 at L=9)"),
    }])

    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    items.to_csv(os.path.join(OUT, "tables",
                              "residual_trained_real_items.tsv"), sep="\t",
                 index=False)
    by_len.to_csv(os.path.join(OUT, "tables",
                               "residual_trained_real_by_length.tsv"),
                  sep="\t", index=False)
    summary.to_csv(os.path.join(OUT, "tables",
                                "residual_trained_real_summary.tsv"), sep="\t",
                   index=False)
    with open(os.path.join(OUT, "provenance.json"), "w") as f:
        json.dump({"model_loaded": False, "inference_run": False,
                   "regression_fitted": False,
                   "inputs": {"canonical": CANON, "error_table": ERRORS}},
                  f, indent=2)
        f.write("\n")

    print(by_len.to_string(index=False))
    print()
    print(items[["word", "phoneme_length", "zipf_frequency",
                 "median_zipf_of_peers", "zipf_percentile_within_length",
                 "n_failing_seeds"]].round(3).to_string(index=False))
    print()
    print(summary.iloc[0][["mean_within_length_zipf_percentile",
                           "n_below_peer_median",
                           "permutation_p_one_sided_lower",
                           "permutation_null_mean"]].to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
