"""Sprint 4 — premature-EOS decoder diagnostics.

Kept in its own module because EOS is **not** part of the Levenshtein taxonomy:
a deletion is not automatically a premature EOS, a premature EOS is not one
deletion, several deletions may follow one early stop, and early stops may
coexist with substitutions or insertions.

The indexing convention was audited from the committed evaluator before any
distribution was read and is frozen in
reports/.../error_taxonomy/_control/eos_convention.json:

    observed eos_position : 0-based index into the item's readout window,
                            equal to the number of phonemes emitted before EOS
                            (and to predicted_length whenever present)
    expected_eos_position : L, the target phoneme length
    readout window        : exactly L tokens, indices 0 .. L-1

Consequence, verified empirically on all four seeds: a boundary EOS would sit
at window index L, one past the end of the slice, so **ON_TIME_EOS and
LATE_EOS are structurally unobservable and every observed EOS is premature**.
`EOS_NOT_OBSERVED` is therefore ambiguous — it conflates correct stopping with
never stopping — and is never read as evidence of correct stopping.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .bootstrap import ols_slope
from .common import CEILING_SEEDS, LENGTHS, LEXICALITY_LABEL, ROUTES, SEEDS

PREMATURE = "PREMATURE_EOS"
ON_TIME = "ON_TIME_EOS"
LATE = "LATE_EOS"
NOT_OBSERVED = "EOS_NOT_OBSERVED"
UNAVAILABLE = "EOS_UNAVAILABLE"

STATUS_OK = "OK"
STATUS_ALL_ZERO = "ALL_ZERO_OUTCOME"
STATUS_ALL_ONE = "ALL_ONE_OUTCOME"
STATUS_INSUFFICIENT = "INSUFFICIENT_EVENTS"
STATUS_NON_ESTIMABLE = "NON_ESTIMABLE"

MIN_EVENTS_FOR_SLOPE = 5
LEXICALITIES = ["real", "pseudo"]


def classify_eos(observed, target_length: int) -> str:
    """Frozen classification.  `observed` is the raw instrumentation value."""
    if observed is None or (isinstance(observed, float) and np.isnan(observed)) \
            or (isinstance(observed, str) and not observed.strip()):
        return NOT_OBSERVED
    try:
        o = int(float(observed))
    except (TypeError, ValueError):
        return UNAVAILABLE
    if o < 0:
        return UNAVAILABLE
    if o < target_length:
        return PREMATURE
    if o == target_length:
        return ON_TIME          # unreachable under this readout horizon
    return LATE                 # unreachable under this readout horizon


def eos_shortfall(observed, target_length: int):
    """expected - observed, for PREMATURE_EOS only; positive means early."""
    if classify_eos(observed, target_length) != PREMATURE:
        return np.nan
    return float(target_length - int(float(observed)))


def item_level(canon: pd.DataFrame, regime: str = "LICHTHEIM_CLEAN"
               ) -> pd.DataFrame:
    """Per seed x item x route EOS classification joined to deletion counts."""
    if regime == "LICHTHEIM_CLEAN":
        d = canon[canon["in_LICHTHEIM_CLEAN"]]
    elif regime == "ALL_WITH_EXPOSURE_STRATA":
        d = canon[canon["in_ALL_WITH_EXPOSURE_STRATA"]]
    else:
        raise ValueError(f"unknown regime {regime!r}")
    d = d.copy()
    d["expected_eos_position"] = d["target_length"].astype(int)
    d["eos_class"] = [classify_eos(o, int(L)) for o, L
                      in zip(d["eos_position"], d["target_length"])]
    d["eos_shortfall"] = [eos_shortfall(o, int(L)) for o, L
                          in zip(d["eos_position"], d["target_length"])]
    d["premature_eos"] = (d["eos_class"] == PREMATURE).astype(int)
    # all-item shortfall: zero when not premature (primary convention)
    d["eos_shortfall_all_items"] = d["eos_shortfall"].fillna(0.0)
    d["has_deletion"] = (d["deletions"] > 0).astype(int)
    d["broad_length"] = d["target_length"].map(
        lambda n: "Short" if n in (3, 4, 5) else ("Long" if n in (7, 8, 9)
                                                  else f"OTHER_{n}"))
    return d


def _rates(sub: pd.DataFrame) -> Dict[str, float]:
    n = len(sub)
    prem = sub[sub["premature_eos"] == 1]
    return {
        "n_items": int(n), "n_premature": int(len(prem)),
        "premature_eos_rate": float(sub["premature_eos"].mean()) if n else np.nan,
        "mean_eos_shortfall_per_item":
            float(sub["eos_shortfall_all_items"].mean()) if n else np.nan,
        "conditional_mean_eos_shortfall":
            float(prem["eos_shortfall"].mean()) if len(prem) else np.nan,
        "n_eos_not_observed": int((sub["eos_class"] == NOT_OBSERVED).sum()),
        "n_eos_unavailable": int((sub["eos_class"] == UNAVAILABLE).sum()),
        "n_on_time_observed": int((sub["eos_class"] == ON_TIME).sum()),
        "n_late_observed": int((sub["eos_class"] == LATE).sum()),
    }


def by_seed(items: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for route in ROUTES:
            for lex in LEXICALITIES:
                sub = items[(items["seed"] == seed) & (items["route"] == route)
                            & (items["source_lexicality"] == lex)]
                rows.append({"seed": seed, "route": route,
                             "source_lexicality": lex,
                             "label": LEXICALITY_LABEL[lex], **_rates(sub)})
    return pd.DataFrame(rows)


def by_length(items: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for route in ROUTES:
            for lex in LEXICALITIES:
                for L in LENGTHS:
                    sub = items[(items["seed"] == seed) & (items["route"] == route)
                                & (items["source_lexicality"] == lex)
                                & (items["target_length"] == L)]
                    rows.append({"seed": seed, "route": route,
                                 "source_lexicality": lex,
                                 "label": LEXICALITY_LABEL[lex],
                                 "phoneme_length": L, **_rates(sub)})
    return pd.DataFrame(rows)


def by_broad_length(items: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for route in ROUTES:
            for lex in LEXICALITIES:
                for grp in ("Short", "Long"):
                    sub = items[(items["seed"] == seed) & (items["route"] == route)
                                & (items["source_lexicality"] == lex)
                                & (items["broad_length"] == grp)]
                    rows.append({"seed": seed, "route": route,
                                 "source_lexicality": lex,
                                 "label": LEXICALITY_LABEL[lex],
                                 "broad_length": grp, **_rates(sub)})
    return pd.DataFrame(rows)


def by_exposure(items: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for route in ROUTES:
            for st in sorted(items["lichtheim_exposure_status"].unique()):
                sub = items[(items["seed"] == seed) & (items["route"] == route)
                            & (items["lichtheim_exposure_status"] == st)]
                rows.append({"seed": seed, "route": route,
                             "lichtheim_exposure_status": st,
                             "descriptive_only": bool(len(sub) <= 7),
                             **_rates(sub)})
    return pd.DataFrame(rows)


def by_morphology(items: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for route in ROUTES:
            for lex in LEXICALITIES:
                for mor in ("complex", "simple"):
                    sub = items[(items["seed"] == seed) & (items["route"] == route)
                                & (items["source_lexicality"] == lex)
                                & (items["morphology"] == mor)]
                    rows.append({"seed": seed, "route": route,
                                 "source_lexicality": lex,
                                 "morphology": mor, **_rates(sub)})
    return pd.DataFrame(rows)


def length_slopes(items: pd.DataFrame) -> pd.DataFrame:
    """Linear probability slope of premature EOS on phoneme length."""
    rows = []
    for seed in SEEDS:
        for route in ROUTES:
            for lex in LEXICALITIES:
                sub = items[(items["seed"] == seed) & (items["route"] == route)
                            & (items["source_lexicality"] == lex)]
                y = sub["premature_eos"].to_numpy(float)
                x = sub["target_length"].to_numpy(float)
                n_ev = int(y.sum())
                if len(y) < 3 or np.std(x) == 0:
                    status, b0, b1 = STATUS_NON_ESTIMABLE, np.nan, np.nan
                elif n_ev == 0:
                    status, b0, b1 = STATUS_ALL_ZERO, 0.0, 0.0
                elif n_ev == len(y):
                    status, b0, b1 = STATUS_ALL_ONE, 1.0, 0.0
                elif n_ev < MIN_EVENTS_FOR_SLOPE:
                    status = STATUS_INSUFFICIENT
                    b0, b1 = ols_slope(x, y)
                else:
                    status = STATUS_OK
                    b0, b1 = ols_slope(x, y)
                rows.append({"seed": seed, "route": route,
                             "source_lexicality": lex,
                             "label": LEXICALITY_LABEL[lex],
                             "n_items": int(len(y)), "n_premature": n_ev,
                             "intercept": b0, "length_slope": b1,
                             "model_status": status,
                             "model_type": "linear probability (descriptive)",
                             "logistic_forced": False})
    return pd.DataFrame(rows)


def deletion_overlap(items: pd.DataFrame,
                     group_cols: Sequence[str] = ("route", "source_lexicality"),
                     ) -> pd.DataFrame:
    """2x2 overlap between premature EOS and the presence of a deletion."""
    rows = []
    for key, g in items.groupby(list(group_cols), dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        both = int(((g["premature_eos"] == 1) & (g["has_deletion"] == 1)).sum())
        eos_only = int(((g["premature_eos"] == 1) & (g["has_deletion"] == 0)).sum())
        del_only = int(((g["premature_eos"] == 0) & (g["has_deletion"] == 1)).sum())
        neither = int(((g["premature_eos"] == 0) & (g["has_deletion"] == 0)).sum())
        n_eos = both + eos_only
        n_del = both + del_only
        rows.append({**dict(zip(group_cols, key)), "n_items": int(len(g)),
                     "premature_eos_and_deletion": both,
                     "premature_eos_without_deletion": eos_only,
                     "deletion_without_premature_eos": del_only,
                     "neither": neither,
                     "n_premature_eos": n_eos, "n_with_deletion": n_del,
                     "p_deletion_given_premature_eos":
                         (both / n_eos) if n_eos else np.nan,
                     "p_premature_eos_given_deletion":
                         (both / n_del) if n_del else np.nan,
                     "not_causal": True})
    return pd.DataFrame(rows)


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
                     "seed21_included": 21 in present,
                     "exact_zero_seeds_mean": float(
                         vals.loc[ceil].dropna().mean()) if ceil else np.nan})
    return pd.DataFrame(rows)
