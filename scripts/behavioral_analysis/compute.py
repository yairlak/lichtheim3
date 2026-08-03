"""Statistics layer: builds every plotting table from the canonical table.

Each figure has exactly one writer here, which removes the ordering hazard the
original drivers had (two scripts writing the same statistics file).  All
scientific logic is promoted unchanged from the validated drivers; nothing is
recomputed with different conventions.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate

from .bootstrap import cell_mean_bootstrap, hierarchical_bootstrap, ols_slope
from .common import (CLEAN_EXPOSURES, EXPOSURE_ORDER, LENGTHS,
                     LEXICALITY_LABEL, ROUTES, SEEDS)

LEXICALITIES = ["real", "pseudo"]


# ----------------------------------------------------- figure 1: length curves

def clean_length_table(clean: pd.DataFrame) -> pd.DataFrame:
    """Per-seed cell means plus the across-seed mean and bootstrap band.

    The band is persisted so the figure can be drawn from this table alone.
    """
    rows = []
    for r in ROUTES:
        for lex in LEXICALITIES:
            for L in LENGTHS:
                cell = clean[(clean["route"] == r)
                             & (clean["source_lexicality"] == lex)
                             & (clean["target_length"] == L)]
                if cell.empty:
                    continue
                per_seed = {}
                for s in SEEDS:
                    sub = cell[cell["seed"] == s].sort_values("item_id")
                    per_seed[s] = sub["raw_edit_distance"].to_numpy(float)
                lo, hi = cell_mean_bootstrap(per_seed, SEEDS)
                for s in SEEDS:
                    sub = cell[cell["seed"] == s]
                    rows.append({
                        "route": r, "source_lexicality": lex,
                        "label": LEXICALITY_LABEL[lex], "phoneme_length": L,
                        "seed": s, "n_items": int(sub["item_id"].nunique()),
                        "mean_raw_edit_distance":
                            float(sub["raw_edit_distance"].mean()),
                        "word_error_rate": float(sub["word_error"].mean()),
                        "mean_across_seeds":
                            float(cell["raw_edit_distance"].mean()),
                        "ci_low": lo, "ci_high": hi,
                    })
    return pd.DataFrame(rows)


# ------------------------------------------------------ figure 2: slopes

def clean_slope_tables(clean: pd.DataFrame):
    """(per-seed slopes, per-seed contrasts, bootstrap summary)."""
    slope_rows = []
    for s in SEEDS:
        for r in ROUTES:
            for lex in LEXICALITIES:
                d = clean[(clean["seed"] == s) & (clean["route"] == r)
                          & (clean["source_lexicality"] == lex)]
                b0, b1 = ols_slope(d["target_length"].to_numpy(float),
                                   d["raw_edit_distance"].to_numpy(float))
                slope_rows.append({
                    "seed": s, "source_lexicality": lex,
                    "label": LEXICALITY_LABEL[lex], "route": r,
                    "n_items": int(d["item_id"].nunique()),
                    "intercept": b0, "length_slope": b1,
                    "model_status": "OK" if np.isfinite(b1) else "DEGENERATE"})
    slopes = pd.DataFrame(slope_rows)

    con_rows = []
    for s in SEEDS:
        for lex in LEXICALITIES:
            g = slopes[(slopes["seed"] == s)
                       & (slopes["source_lexicality"] == lex)]
            v = {r: float(g[g["route"] == r]["length_slope"].iloc[0])
                 for r in ROUTES}
            con_rows.append({
                "seed": s, "source_lexicality": lex,
                "label": LEXICALITY_LABEL[lex],
                "wm_length_slope": v["wm"], "ltm_length_slope": v["ltm"],
                "full_length_slope": v["full"],
                "ltm_minus_wm": v["ltm"] - v["wm"]})
    contrasts = pd.DataFrame(con_rows)

    x_by, ids_by, y_by = _stratum_arrays(clean, "source_lexicality",
                                         LEXICALITIES)
    boot_rows = []
    for lex in LEXICALITIES:
        for r in ROUTES:
            res = hierarchical_bootstrap(
                x_by, y_by, SEEDS, lambda per, r=r, lex=lex: per[(r, lex)])
            boot_rows.append({"analysis_set": "LICHTHEIM_CLEAN",
                              "stratum": lex, "label": LEXICALITY_LABEL[lex],
                              "quantity": f"{r}_length_slope",
                              "metric": "raw_edit_distance", **res})
        res = hierarchical_bootstrap(
            x_by, y_by, SEEDS,
            lambda per, lex=lex: per[("ltm", lex)] - per[("wm", lex)])
        obs = contrasts[contrasts["source_lexicality"] == lex]["ltm_minus_wm"]
        boot_rows.append({
            "analysis_set": "LICHTHEIM_CLEAN", "stratum": lex,
            "label": LEXICALITY_LABEL[lex], "quantity": "ltm_minus_wm",
            "metric": "raw_edit_distance",
            "mean_over_seeds": float(obs.mean()),
            "min": float(obs.min()), "max": float(obs.max()),
            "range": float(obs.max() - obs.min()),
            "all_same_sign": bool((obs > 0).all() or (obs < 0).all()), **res})
    return slopes, contrasts, pd.DataFrame(boot_rows)


def _stratum_arrays(df: pd.DataFrame, stratum_col: str, strata: List[str]):
    base = df[(df["seed"] == SEEDS[0]) & (df["route"] == "full")] \
        .sort_values("item_id")
    x_by, ids_by = {}, {}
    for st in strata:
        sel = base[base[stratum_col] == st]
        x_by[st] = sel["target_length"].to_numpy(float)
        ids_by[st] = sel["item_id"].tolist()
    y_by: Dict[tuple, np.ndarray] = {}
    for s in SEEDS:
        for r in ROUTES:
            d = df[(df["seed"] == s) & (df["route"] == r)].set_index("item_id")
            for st in strata:
                y_by[(s, r, st)] = d.loc[ids_by[st],
                                         "raw_edit_distance"].to_numpy(float)
    return x_by, ids_by, y_by


# --------------------------------------------- figure 3: serial position

def zip_mismatch_positions(target: str, prediction: str) -> List[int]:
    """Faithful Dager Error_Indices: 1-based positional mismatches.

    NOT a Levenshtein alignment.  Dager blanks everything after the first
    <eos> to <PAD> and keeps the prediction at exactly len(gold); Lichtheim3
    trims at <eos> instead, so a short prediction is re-padded here to recover
    identical positional semantics.
    """
    t = str(target).split()
    p = str(prediction).split()
    p = p + ["<PAD>"] * (len(t) - len(p))
    return [i + 1 for i, (a, b) in enumerate(zip(t, p)) if a != b]


def serial_position_tables(clean: pd.DataFrame):
    """(per-position counts, 100-point interpolated curves)."""
    grid = np.linspace(0, 1, 100)
    raw_rows, curve_rows = [], []
    for r in ROUTES:
        for lex in LEXICALITIES:
            acc, total = np.zeros(100), 0
            for L in LENGTHS:
                d = clean[(clean["route"] == r)
                          & (clean["source_lexicality"] == lex)
                          & (clean["target_length"] == L)]
                if d.empty or L < 2:
                    continue
                n = len(d)                      # items x seeds in this cell
                counts = np.zeros(L)
                for tgt, pred in zip(d["target"], d["prediction"]):
                    for idx in zip_mismatch_positions(tgt, pred):
                        counts[idx - 1] += 1
                rate = counts / n
                xs = np.array([i / (L - 1) for i in range(L)])
                acc += n * pchip_interpolate(xs, rate, grid)
                total += n
                for i in range(L):
                    raw_rows.append({
                        "route": r, "source_lexicality": lex,
                        "label": LEXICALITY_LABEL[lex], "phoneme_length": L,
                        "position_index_1based": i + 1,
                        "relative_position": float(xs[i]),
                        "n_items_x_seeds": n,
                        "error_count": float(counts[i]),
                        "error_rate_per_item": float(rate[i])})
            curve = acc / max(total, 1)
            for j, x in enumerate(grid):
                curve_rows.append({
                    "route": r, "source_lexicality": lex,
                    "label": LEXICALITY_LABEL[lex],
                    "relative_position": float(x),
                    "interpolated_error_rate": float(curve[j]),
                    "n_items_x_seeds_total": total})
    return pd.DataFrame(raw_rows), pd.DataFrame(curve_rows)


# --------------------------------------------------- figures 4-5: gate

def gate_tables(canon: pd.DataFrame):
    """(clean lexicality gate table, exposure-status gate table).

    Gate and lexical confidence are item-level FULL-route quantities, so the
    full-route rows carry them; WM-only and LTM-only have none by design.
    """
    full = canon[canon["route"] == "full"].copy()
    clean = full[full["in_LICHTHEIM_CLEAN"]]

    lex_rows = []
    for s in SEEDS:
        for lex in LEXICALITIES:
            sub = clean[(clean["seed"] == s)
                        & (clean["source_lexicality"] == lex)]
            lex_rows.append({
                "analysis_set": "LICHTHEIM_CLEAN", "grouping": "lexicality",
                "source_lexicality": lex, "label": LEXICALITY_LABEL[lex],
                "seed": s, "n_items": int(len(sub)),
                "mean_gate": float(sub["gate"].mean()),
                "mean_lexical_confidence":
                    float(sub["lexical_confidence"].mean()),
                "mean_margin": float(sub["margin"].mean()),
                "mean_density": float(sub["density"].mean())})

    exp_rows = []
    for s in SEEDS:
        for st in EXPOSURE_ORDER:
            sub = full[(full["seed"] == s)
                       & (full["lichtheim_exposure_status"] == st)]
            if sub.empty:
                continue
            exp_rows.append({
                "analysis_set": "ALL_WITH_EXPOSURE_STRATA",
                "grouping": "exposure_status", "exposure_status": st,
                "seed": s, "n_items": int(len(sub)),
                "mean_gate": float(sub["gate"].mean()),
                "mean_lexical_confidence":
                    float(sub["lexical_confidence"].mean()),
                "mean_margin": float(sub["margin"].mean()),
                "mean_density": float(sub["density"].mean())})
    return pd.DataFrame(lex_rows), pd.DataFrame(exp_rows)
