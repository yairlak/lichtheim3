"""Sprint 4 — Levenshtein error taxonomy.

Implements the operation estimands frozen in
reports/behavioral_wfe_fulllexicon_93a577f/error_taxonomy/_control/
error_taxonomy_analysis_spec.json, written before any result was inspected.

Operations are exactly substitution, deletion and insertion, read from the
counts already produced by `Levenshtein.editops` 0.27.3 and stored in the
validated tables.  Nothing here re-aligns sequences, and **no fourth operation
exists**: premature EOS is a decoder diagnostic handled in `eos_diagnostics.py`.

Two measurement limits carried into every caption: editops tie-breaking can
move counts between operation types without changing total edit distance, and
the forced-length readout makes terminal insertions unobservable.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .common import (BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CEILING_SEEDS,
                     LENGTHS, LEXICALITY_LABEL, ROUTES, SEEDS)

OPERATIONS = ["substitutions", "deletions", "insertions"]
OPERATION_LABEL = {"substitutions": "Substitutions", "deletions": "Deletions",
                   "insertions": "Insertions"}
SHORT_LENGTHS = [3, 4, 5]
LONG_LENGTHS = [7, 8, 9]
LEXICALITIES = ["real", "pseudo"]

EDITOPS_BACKEND = "Levenshtein.editops"
EDITOPS_VERSION = "0.27.3"

SMALL_CELL_MAX = 20
VERY_SMALL_CELL_MAX = 10

NO_COMPOSITION = "NO_ERRORS_FOR_COMPOSITION"


def cell_flag(n: int) -> str:
    if n < VERY_SMALL_CELL_MAX:
        return "VERY_SMALL_CELL"
    if n < SMALL_CELL_MAX:
        return "SMALL_CELL"
    return "OK"


def broad_length(n: int) -> str:
    if n in SHORT_LENGTHS:
        return "Short"
    if n in LONG_LENGTHS:
        return "Long"
    return f"OTHER_{n}"


def regime_subset(canon: pd.DataFrame, regime: str) -> pd.DataFrame:
    if regime == "FAITHFUL_WFE_ALL":
        d = canon[canon["in_FAITHFUL_WFE_ALL"] & (canon["route"] == "full")]
    elif regime == "LICHTHEIM_CLEAN":
        d = canon[canon["in_LICHTHEIM_CLEAN"]]
    elif regime == "ALL_WITH_EXPOSURE_STRATA":
        d = canon[canon["in_ALL_WITH_EXPOSURE_STRATA"]]
    else:
        raise ValueError(f"unknown regime {regime!r}")
    d = d.copy()
    d["broad_length"] = d["target_length"].map(broad_length)
    return d


def _agg(sub: pd.DataFrame) -> Dict[str, float]:
    """Operation summaries for one cell.  Primary means are unconditional."""
    n = len(sub)
    err = sub[sub["word_error"] == 1]
    out: Dict[str, float] = {
        "n_items": int(n), "n_erroneous_items": int(len(err)),
        "cell_flag": cell_flag(n),
        "total_edit_operations": float(sub["raw_edit_distance"].sum()),
        "mean_edit_distance_per_item": float(sub["raw_edit_distance"].mean())
        if n else np.nan,
    }
    total_ops = out["total_edit_operations"]
    for op in OPERATIONS:
        cnt = float(sub[op].sum())
        out[f"total_{op}"] = cnt
        out[f"mean_{op}_per_item"] = cnt / n if n else np.nan
        out[f"mean_{op}_per_erroneous_item"] = (cnt / len(err) if len(err)
                                                else np.nan)
        out[f"proportion_{op}"] = (cnt / total_ops if total_ops > 0
                                   else np.nan)
    out["composition_status"] = "OK" if total_ops > 0 else NO_COMPOSITION
    return out


# --------------------------------------------------- faithful Figure 8A

def faithful_condition_table(canon: pd.DataFrame) -> pd.DataFrame:
    """Per seed x original WFE condition, FULL route, all 1,200 items.

    Condition order is the source order in which the conditions first appear in
    the canonical table, i.e. the WFE file order recovered by the SWP audit.
    """
    d = regime_subset(canon, "FAITHFUL_WFE_ALL")
    order = list(dict.fromkeys(
        d[d["seed"] == SEEDS[0]].sort_values("item_id")["condition"]))
    rows = []
    for seed in SEEDS:
        for i, cond in enumerate(order):
            sub = d[(d["seed"] == seed) & (d["condition"] == cond)]
            rows.append({"dataset_regime": "FAITHFUL_WFE_ALL", "route": "full",
                         "seed": seed, "condition": cond,
                         "condition_order": i,
                         "source_lexicality": sub["source_lexicality"].iloc[0],
                         "size": sub["size"].iloc[0],
                         "morphology": sub["morphology"].iloc[0],
                         **_agg(sub)})
    return pd.DataFrame(rows)


# ------------------------------------------------------ clean taxonomy

def clean_cells(canon: pd.DataFrame) -> pd.DataFrame:
    """seed x route x lexicality x broad length."""
    d = regime_subset(canon, "LICHTHEIM_CLEAN")
    rows = []
    for seed in SEEDS:
        for route in ROUTES:
            for lex in LEXICALITIES:
                for grp in ("Short", "Long"):
                    sub = d[(d["seed"] == seed) & (d["route"] == route)
                            & (d["source_lexicality"] == lex)
                            & (d["broad_length"] == grp)]
                    rows.append({"dataset_regime": "LICHTHEIM_CLEAN",
                                 "seed": seed, "route": route,
                                 "source_lexicality": lex,
                                 "label": LEXICALITY_LABEL[lex],
                                 "broad_length": grp, **_agg(sub)})
    return pd.DataFrame(rows)


def clean_by_exact_length(canon: pd.DataFrame) -> pd.DataFrame:
    d = regime_subset(canon, "LICHTHEIM_CLEAN")
    rows = []
    for seed in SEEDS:
        for route in ROUTES:
            for lex in LEXICALITIES:
                for L in LENGTHS:
                    sub = d[(d["seed"] == seed) & (d["route"] == route)
                            & (d["source_lexicality"] == lex)
                            & (d["target_length"] == L)]
                    rows.append({"dataset_regime": "LICHTHEIM_CLEAN",
                                 "seed": seed, "route": route,
                                 "source_lexicality": lex,
                                 "label": LEXICALITY_LABEL[lex],
                                 "phoneme_length": L, **_agg(sub)})
    return pd.DataFrame(rows)


def clean_by_morphology(canon: pd.DataFrame) -> pd.DataFrame:
    d = regime_subset(canon, "LICHTHEIM_CLEAN")
    rows = []
    for seed in SEEDS:
        for route in ROUTES:
            for lex in LEXICALITIES:
                for mor in ("complex", "simple"):
                    sub = d[(d["seed"] == seed) & (d["route"] == route)
                            & (d["source_lexicality"] == lex)
                            & (d["morphology"] == mor)]
                    rows.append({"dataset_regime": "LICHTHEIM_CLEAN",
                                 "seed": seed, "route": route,
                                 "source_lexicality": lex,
                                 "label": LEXICALITY_LABEL[lex],
                                 "morphology": mor, **_agg(sub)})
    return pd.DataFrame(rows)


def by_exposure_status(canon: pd.DataFrame) -> pd.DataFrame:
    d = regime_subset(canon, "ALL_WITH_EXPOSURE_STRATA")
    rows = []
    for seed in SEEDS:
        for route in ROUTES:
            for st in sorted(d["lichtheim_exposure_status"].unique()):
                sub = d[(d["seed"] == seed) & (d["route"] == route)
                        & (d["lichtheim_exposure_status"] == st)]
                rows.append({"dataset_regime": "ALL_WITH_EXPOSURE_STRATA",
                             "seed": seed, "route": route,
                             "lichtheim_exposure_status": st,
                             "descriptive_only": bool(len(sub) <= 7),
                             **_agg(sub)})
    return pd.DataFrame(rows)


# ------------------------------------------------------ route contrasts

def route_contrasts(cells: pd.DataFrame) -> pd.DataFrame:
    """Per-operation and total route differences at seed level."""
    pairs = [("ltm", "wm", "ltm_minus_wm"), ("full", "wm", "full_minus_wm"),
             ("ltm", "full", "ltm_minus_full")]
    agg = (cells.groupby(["seed", "route", "source_lexicality"], as_index=False)
           .agg({**{f"total_{op}": "sum" for op in OPERATIONS},
                 "total_edit_operations": "sum", "n_items": "sum"}))
    rows = []
    for seed in SEEDS:
        for lex in LEXICALITIES:
            for a, b, name in pairs:
                ra = agg[(agg["seed"] == seed) & (agg["route"] == a)
                         & (agg["source_lexicality"] == lex)]
                rb = agg[(agg["seed"] == seed) & (agg["route"] == b)
                         & (agg["source_lexicality"] == lex)]
                if ra.empty or rb.empty:
                    continue
                n = float(ra["n_items"].iloc[0])
                row = {"dataset_regime": "LICHTHEIM_CLEAN", "seed": seed,
                       "source_lexicality": lex, "label": LEXICALITY_LABEL[lex],
                       "route_contrast": name, "n_items": int(n),
                       "meaning": f"mean per item in {a.upper()} minus {b.upper()}",
                       "ceiling_limited": bool(
                           ra["total_edit_operations"].iloc[0] == 0
                           and rb["total_edit_operations"].iloc[0] == 0)}
                for op in OPERATIONS:
                    row[f"{op}_difference"] = float(
                        ra[f"total_{op}"].iloc[0] / n - rb[f"total_{op}"].iloc[0] / n)
                row["total_edit_operations_difference"] = float(
                    ra["total_edit_operations"].iloc[0] / n
                    - rb["total_edit_operations"].iloc[0] / n)
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------- bootstrap

def bootstrap_operations(canon: pd.DataFrame, regime: str = "LICHTHEIM_CLEAN",
                         ) -> pd.DataFrame:
    """Hierarchical bootstrap of mean operations per item, per route/stratum."""
    d = regime_subset(canon, regime)
    rows = []
    for lex in LEXICALITIES:
        base = d[(d["seed"] == SEEDS[0]) & (d["route"] == "full")
                 & (d["source_lexicality"] == lex)].sort_values("item_id")
        ids = base["item_id"].tolist()
        if len(ids) < 3:
            continue
        for op in OPERATIONS + ["raw_edit_distance"]:
            y_by = {}
            for seed in SEEDS:
                for route in ROUTES:
                    sub = d[(d["seed"] == seed) & (d["route"] == route)
                            & (d["source_lexicality"] == lex)] \
                        .set_index("item_id").loc[ids]
                    y_by[(seed, route, "all")] = sub[op].to_numpy(float)
            for route in ROUTES:
                res = _bootstrap_mean(y_by, route, len(ids))
                rows.append({"dataset_regime": regime, "route": route,
                             "source_lexicality": lex,
                             "label": LEXICALITY_LABEL[lex],
                             "quantity": f"mean_{op}_per_item", **res})
    return pd.DataFrame(rows)


def _bootstrap_mean(y_by, route: str, n: int, b: int = BOOTSTRAP_REPLICATES,
                    seed: int = BOOTSTRAP_SEED, chunk: int = 1000) -> dict:
    """Two-level bootstrap of a cell mean: seeds, then items."""
    rng = np.random.default_rng(seed)
    draws = np.empty(b)
    done = 0
    while done < b:
        m = min(chunk, b - done)
        sidx = rng.integers(0, len(SEEDS), size=(m, len(SEEDS)))
        iidx = rng.integers(0, n, size=(m, n))
        per = np.stack([y_by[(s, route, "all")][iidx].mean(axis=1)
                        for s in SEEDS])
        draws[done:done + m] = per[sidx.T, np.arange(m)[None, :]].mean(axis=0)
        done += m
    f = draws[np.isfinite(draws)]
    return {"bootstrap_mean": float(f.mean()),
            "ci_low": float(np.percentile(f, 2.5)),
            "ci_high": float(np.percentile(f, 97.5)),
            "n_replicates": int(b), "n_finite_replicates": int(f.size),
            "random_seed": int(seed),
            "ci_definition": "95% percentile interval"}


def summarise(df: pd.DataFrame, value_col: str,
              group_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(list(group_cols), dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        vals = g.set_index("seed")[value_col]
        present = [s for s in SEEDS if s in vals.index]
        v = vals.loc[present].dropna()
        ceil = [s for s in CEILING_SEEDS if s in vals.index]
        rows.append({**dict(zip(group_cols, key)), "quantity": value_col,
                     "seed_values": "; ".join(
                         f"{s}:{vals.loc[s]:+.6f}" if pd.notna(vals.loc[s])
                         else f"{s}:nan" for s in present),
                     "mean_over_seeds": float(v.mean()) if len(v) else np.nan,
                     "min": float(v.min()) if len(v) else np.nan,
                     "max": float(v.max()) if len(v) else np.nan,
                     "range": float(v.max() - v.min()) if len(v) else np.nan,
                     "all_same_sign": bool((v > 0).all() or (v < 0).all())
                     if len(v) else False,
                     "seed21_included": 21 in present,
                     "exact_zero_seeds_mean": float(
                         vals.loc[ceil].dropna().mean()) if ceil else np.nan})
    return pd.DataFrame(rows)
