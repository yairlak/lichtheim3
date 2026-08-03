"""Sprint 2 — morphology x phoneme length statistics.

Implements exactly the estimands frozen in
reports/behavioral_wfe_fulllexicon_93a577f/morphology/_control/
morphology_analysis_spec.json, which was written before any morphology result
was computed or inspected.

Sign conventions (frozen):
    morphology_contrast            = mean(simple) - mean(complex)
        positive => more errors for morphologically SIMPLE items
    morphology_length_interaction  = simple_length_slope - complex_length_slope
        positive => length effect stronger for SIMPLE items

The faithful (FAITHFUL_WFE_ALL, FULL route, original labels) and adapted
(LICHTHEIM_CLEAN, three routes, exposure-defined labels) analyses are computed
separately and are never combined into one statistical claim.

Reuses the Sprint-1 bootstrap and I/O utilities unchanged; nothing here loads a
model.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .bootstrap import hierarchical_bootstrap, ols_slope
from .common import (CEILING_SEEDS, LENGTHS, LEXICALITY_LABEL, ROUTES, SEEDS)

MORPHOLOGIES = ["complex", "simple"]      # alphabetical == solid, dashed
LEXICALITIES = ["real", "pseudo"]
METRICS = ["raw_edit_distance", "word_error"]

# Frozen small-cell thresholds; not revisable after seeing results.
VERY_SMALL_CELL_MAX = 10
SMALL_CELL_MAX = 20


def cell_flag(n: int) -> str:
    if n < VERY_SMALL_CELL_MAX:
        return "VERY_SMALL_CELL"
    if n < SMALL_CELL_MAX:
        return "SMALL_CELL"
    return "OK"


def _regime_subset(canon: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Rows for one dataset regime, with its route policy applied."""
    if regime == "FAITHFUL_WFE_ALL":
        d = canon[canon["in_FAITHFUL_WFE_ALL"] & (canon["route"] == "full")]
    elif regime == "LICHTHEIM_CLEAN":
        d = canon[canon["in_LICHTHEIM_CLEAN"]]
    else:
        raise ValueError(f"unknown regime {regime!r}")
    return d.copy()


# ------------------------------------------------------------- descriptives

def cell_counts(canon: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Item counts per cell, from one seed (counts are seed-invariant)."""
    d = _regime_subset(canon, regime)
    d = d[d["seed"] == SEEDS[0]]
    rows = []
    for route in sorted(d["route"].unique()):
        for lex in LEXICALITIES:
            for mor in MORPHOLOGIES:
                for L in LENGTHS:
                    cell = d[(d["route"] == route)
                             & (d["source_lexicality"] == lex)
                             & (d["morphology"] == mor)
                             & (d["target_length"] == L)]
                    n = int(cell["item_id"].nunique())
                    rows.append({"dataset_regime": regime, "route": route,
                                 "source_lexicality": lex,
                                 "label": LEXICALITY_LABEL[lex],
                                 "morphology": mor, "phoneme_length": L,
                                 "n_items": n, "cell_flag": cell_flag(n)})
    return pd.DataFrame(rows)


def descriptive_cells(canon: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Per seed x route x lexicality x morphology x length descriptives."""
    d = _regime_subset(canon, regime)
    rows = []
    for seed in SEEDS:
        for route in sorted(d["route"].unique()):
            for lex in LEXICALITIES:
                for mor in MORPHOLOGIES:
                    for L in LENGTHS:
                        cell = d[(d["seed"] == seed) & (d["route"] == route)
                                 & (d["source_lexicality"] == lex)
                                 & (d["morphology"] == mor)
                                 & (d["target_length"] == L)]
                        n = len(cell)
                        rows.append({
                            "dataset_regime": regime, "seed": seed,
                            "route": route, "source_lexicality": lex,
                            "label": LEXICALITY_LABEL[lex], "morphology": mor,
                            "phoneme_length": L, "n_items": n,
                            "cell_flag": cell_flag(n),
                            "mean_raw_edit_distance":
                                float(cell["raw_edit_distance"].mean()) if n else np.nan,
                            "word_error_rate":
                                float(cell["word_error"].mean()) if n else np.nan,
                            "total_raw_edit_distance":
                                float(cell["raw_edit_distance"].sum()) if n else 0.0,
                            "n_erroneous_items":
                                int(cell["word_error"].sum()) if n else 0})
    return pd.DataFrame(rows)


# ------------------------------------------------------ seed-level contrasts

def seed_contrasts(canon: pd.DataFrame, regime: str) -> pd.DataFrame:
    """morphology_contrast = mean(simple) - mean(complex), per seed cell."""
    d = _regime_subset(canon, regime)
    rows = []
    for seed in SEEDS:
        for route in sorted(d["route"].unique()):
            for lex in LEXICALITIES:
                sub = d[(d["seed"] == seed) & (d["route"] == route)
                        & (d["source_lexicality"] == lex)]
                simple = sub[sub["morphology"] == "simple"]
                complex_ = sub[sub["morphology"] == "complex"]
                row = {"dataset_regime": regime, "seed": seed, "route": route,
                       "source_lexicality": lex, "label": LEXICALITY_LABEL[lex],
                       "n_simple": len(simple), "n_complex": len(complex_),
                       "cell_flag_simple": cell_flag(len(simple)),
                       "cell_flag_complex": cell_flag(len(complex_))}
                for metric in METRICS:
                    ms = float(simple[metric].mean()) if len(simple) else np.nan
                    mc = float(complex_[metric].mean()) if len(complex_) else np.nan
                    row[f"mean_simple_{metric}"] = ms
                    row[f"mean_complex_{metric}"] = mc
                    row[f"morphology_contrast_{metric}"] = ms - mc
                rows.append(row)
    return pd.DataFrame(rows)


def seed_length_interactions(canon: pd.DataFrame, regime: str) -> pd.DataFrame:
    """simple_length_slope - complex_length_slope, per seed cell."""
    d = _regime_subset(canon, regime)
    rows = []
    for seed in SEEDS:
        for route in sorted(d["route"].unique()):
            for lex in LEXICALITIES:
                sub = d[(d["seed"] == seed) & (d["route"] == route)
                        & (d["source_lexicality"] == lex)]
                row = {"dataset_regime": regime, "seed": seed, "route": route,
                       "source_lexicality": lex, "label": LEXICALITY_LABEL[lex]}
                slopes = {}
                for mor in MORPHOLOGIES:
                    m = sub[sub["morphology"] == mor]
                    b0, b1 = ols_slope(m["target_length"].to_numpy(float),
                                       m["raw_edit_distance"].to_numpy(float))
                    slopes[mor] = b1
                    row[f"{mor}_n_items"] = len(m)
                    row[f"{mor}_n_lengths"] = int(m["target_length"].nunique())
                    row[f"{mor}_intercept"] = b0
                    row[f"{mor}_length_slope"] = b1
                    row[f"{mor}_model_status"] = ("OK" if np.isfinite(b1)
                                                  else "DEGENERATE")
                row["morphology_length_interaction"] = (slopes["simple"]
                                                        - slopes["complex"])
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- bootstrap

def _arrays_by_morphology(d: pd.DataFrame, route: str, lex: str):
    """Frozen-order item arrays per morphology, aligned across seeds."""
    base = d[(d["seed"] == SEEDS[0]) & (d["route"] == route)
             & (d["source_lexicality"] == lex)].sort_values("item_id")
    x_by, ids_by = {}, {}
    for mor in MORPHOLOGIES:
        sel = base[base["morphology"] == mor]
        x_by[mor] = sel["target_length"].to_numpy(float)
        ids_by[mor] = sel["item_id"].tolist()
    return x_by, ids_by


def bootstrap_morphology(canon: pd.DataFrame, regime: str,
                         metric: str = "raw_edit_distance") -> pd.DataFrame:
    """Bootstrap intervals for the morphology contrast and interaction.

    Uses the Sprint-1 hierarchical bootstrap unchanged (B, seed and CI come
    from the frozen configuration).  Two statistics are drawn per cell:
      * morphology_contrast          - difference of cell means
      * morphology_length_interaction - difference of length slopes
    """
    d = _regime_subset(canon, regime)
    rows = []
    for route in sorted(d["route"].unique()):
        for lex in LEXICALITIES:
            x_by, ids_by = _arrays_by_morphology(d, route, lex)
            if any(len(v) < 3 for v in x_by.values()):
                continue
            y_by: Dict[Tuple[int, str, str], np.ndarray] = {}
            for seed in SEEDS:
                sub = d[(d["seed"] == seed) & (d["route"] == route)
                        & (d["source_lexicality"] == lex)].set_index("item_id")
                for mor in MORPHOLOGIES:
                    y_by[(seed, route, mor)] = sub.loc[ids_by[mor],
                                                       metric].to_numpy(float)

            # slope-difference statistic via the shared machinery
            inter = hierarchical_bootstrap(
                x_by, y_by, SEEDS,
                lambda per, r=route: per[(r, "simple")] - per[(r, "complex")])
            rows.append({"dataset_regime": regime, "route": route,
                         "source_lexicality": lex,
                         "label": LEXICALITY_LABEL[lex], "metric": metric,
                         "quantity": "morphology_length_interaction", **inter})

            # mean-difference statistic: same two-level resampling scheme
            mean_res = _bootstrap_mean_difference(x_by, y_by, route)
            rows.append({"dataset_regime": regime, "route": route,
                         "source_lexicality": lex,
                         "label": LEXICALITY_LABEL[lex], "metric": metric,
                         "quantity": "morphology_contrast", **mean_res})
    return pd.DataFrame(rows)


def _bootstrap_mean_difference(x_by, y_by, route: str,
                               b: int = 10000, seed: int = 20260730,
                               chunk: int = 1000) -> dict:
    """Hierarchical bootstrap of mean(simple) - mean(complex).

    Same two-level scheme as the frozen slope bootstrap: seeds resampled with
    replacement, then items with replacement within each morphology cell.
    """
    rng = np.random.default_rng(seed)
    n_seeds = len(SEEDS)
    draws = np.empty(b)
    done = 0
    while done < b:
        m = min(chunk, b - done)
        sidx = rng.integers(0, n_seeds, size=(m, n_seeds))
        idx = {mor: rng.integers(0, len(x_by[mor]), size=(m, len(x_by[mor])))
               for mor in MORPHOLOGIES}
        per = {}
        for mor in MORPHOLOGIES:
            stacked = np.stack([y_by[(sd, route, mor)][idx[mor]].mean(axis=1)
                                for sd in SEEDS])            # (n_seeds, m)
            per[mor] = stacked[sidx.T, np.arange(m)[None, :]].mean(axis=0)
        draws[done:done + m] = per["simple"] - per["complex"]
        done += m
    finite = draws[np.isfinite(draws)]
    return {"bootstrap_mean": float(np.mean(finite)),
            "ci_low": float(np.percentile(finite, 2.5)),
            "ci_high": float(np.percentile(finite, 97.5)),
            "n_replicates": int(b), "n_finite_replicates": int(finite.size),
            "random_seed": int(seed),
            "ci_definition": "95% percentile interval"}


# ------------------------------------------------------------ route contrasts

def route_contrasts(seed_con: pd.DataFrame,
                    seed_int: pd.DataFrame) -> pd.DataFrame:
    """LTM-WM, FULL-WM and LTM-FULL for both morphology quantities.

    Adapted analysis only; the faithful replication has a single route.
    """
    rows = []
    pairs = [("ltm", "wm", "ltm_minus_wm"), ("full", "wm", "full_minus_wm"),
             ("ltm", "full", "ltm_minus_full")]
    for seed in SEEDS:
        for lex in LEXICALITIES:
            for a, b_, name in pairs:
                ca = seed_con[(seed_con["seed"] == seed)
                              & (seed_con["route"] == a)
                              & (seed_con["source_lexicality"] == lex)]
                cb = seed_con[(seed_con["seed"] == seed)
                              & (seed_con["route"] == b_)
                              & (seed_con["source_lexicality"] == lex)]
                ia = seed_int[(seed_int["seed"] == seed)
                              & (seed_int["route"] == a)
                              & (seed_int["source_lexicality"] == lex)]
                ib = seed_int[(seed_int["seed"] == seed)
                              & (seed_int["route"] == b_)
                              & (seed_int["source_lexicality"] == lex)]
                if ca.empty or cb.empty:
                    continue
                rows.append({
                    "dataset_regime": "LICHTHEIM_CLEAN", "seed": seed,
                    "source_lexicality": lex, "label": LEXICALITY_LABEL[lex],
                    "route_contrast": name,
                    "morphology_contrast_difference":
                        float(ca["morphology_contrast_raw_edit_distance"].iloc[0]
                              - cb["morphology_contrast_raw_edit_distance"].iloc[0]),
                    "morphology_length_interaction_difference":
                        float(ia["morphology_length_interaction"].iloc[0]
                              - ib["morphology_length_interaction"].iloc[0])
                        if not ia.empty and not ib.empty else np.nan,
                })
    return pd.DataFrame(rows)


# -------------------------------------------------------------- summaries

def summarise_across_seeds(df: pd.DataFrame, value_col: str,
                           group_cols: Sequence[str]) -> pd.DataFrame:
    """Four-seed mean, range, per-seed values and exact-zero sensitivity."""
    rows = []
    for key, g in df.groupby(list(group_cols), dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        vals = g.set_index("seed")[value_col]
        present = [s for s in SEEDS if s in vals.index]
        v = vals.loc[present]
        ceiling = [s for s in CEILING_SEEDS if s in vals.index]
        finite = v.dropna()
        rows.append({
            **dict(zip(group_cols, key)),
            "quantity": value_col,
            "seed_values": "; ".join(f"{s}:{vals.loc[s]:+.6f}"
                                     if pd.notna(vals.loc[s]) else f"{s}:nan"
                                     for s in present),
            "n_seeds": len(present),
            "mean_over_seeds": float(finite.mean()) if len(finite) else np.nan,
            "min": float(finite.min()) if len(finite) else np.nan,
            "max": float(finite.max()) if len(finite) else np.nan,
            "range": float(finite.max() - finite.min()) if len(finite) else np.nan,
            "all_same_sign": bool((finite > 0).all() or (finite < 0).all())
            if len(finite) else False,
            "seed21_included": 21 in present,
            "exact_zero_seeds_mean":
                float(vals.loc[ceiling].dropna().mean()) if ceiling else np.nan,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------ plot tables

def plot_table(canon: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Exact plotting table: per-seed cell means plus the across-seed mean."""
    cells = descriptive_cells(canon, regime)
    agg = (cells.groupby(["dataset_regime", "route", "source_lexicality",
                          "morphology", "phoneme_length"], as_index=False)
           .agg(mean_across_seeds=("mean_raw_edit_distance", "mean"),
                word_error_rate_across_seeds=("word_error_rate", "mean"),
                n_items_per_seed=("n_items", "max")))
    out = cells.merge(agg, on=["dataset_regime", "route", "source_lexicality",
                               "morphology", "phoneme_length"], how="left")
    return out
