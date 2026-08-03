"""Sprint 5 — adapted feature importance (A15).

Implements the estimands frozen in
reports/behavioral_wfe_fulllexicon_93a577f/feature_importance/_control/
feature_importance_analysis_spec.json, written before any model was fitted.

This is the **adapted** analysis.  The faithful Dager analysis (A11) lives under
outputs/.../behavioral_analysis/faithful_replication/, is read-only here, and is
never recomputed, replaced or pooled with these values: the two have different
item sets, different route scope and different permutation semantics.

Two design rules do the real work and are enforced by tests:

1. **The split is grouped by item.**  All three route rows of an item stay
   together, and the identical item split is reused across all four seeds and
   all three models, so a seed difference is a model difference and never a
   split difference.
2. **Permutation acts on raw factors, not on model columns.**  Permuting
   `length` rewrites the standardized length column *and* every interaction term
   built from it; permuting `route` reshuffles route labels within an item,
   preserving one FULL, one WM and one LTM row.  Dummies and interaction columns
   are never permuted independently, which would break the correspondence
   between a factor and the terms derived from it.

On LICHTHEIM_CLEAN lexicality and training exposure are perfectly confounded
(Real == TRAINED_REAL_EXACT, Pseudo == NOVEL_PSEUDOWORD), so they never enter
one model and the factor is a **lexicality/exposure contrast**.  Zipf frequency
is undefined for pseudowords and is excluded from every all-item clean model; it
is never imputed.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .common import CEILING_SEEDS, ROUTES, SEEDS

# ------------------------------------------------------------ frozen choices
RIDGE_ALPHA = 1.0
SPLIT_TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 42
PERM_REPEATS = 100
PERM_RANDOM_STATE = 42
SEED_RESAMPLES = 10000
SEED_RESAMPLE_RANDOM_STATE = 20260730
SEED_INTERVAL_LABEL = "seed-resampling interval over four checkpoints"

REFERENCE_LEVELS = {"route": "wm", "lexicality": "pseudo",
                    "morphology": "complex"}
FACTORS = ["route", "lexicality", "length", "morphology"]
ITEM_FACTORS = ["lexicality", "length", "morphology"]
ROUTE_SPECIFIC_FACTORS = ["lexicality", "length", "morphology"]

INTERACTION_BLOCKS = ["route_x_length", "route_x_lexicality",
                      "route_x_morphology", "lexicality_x_length",
                      "morphology_x_length"]

PRIMARY_OUTCOME = "raw_edit_distance"
SECONDARY_OUTCOME = "word_error"

STATUS_OK = "OK"
STATUS_ALL_ZERO = "ALL_ZERO_OUTCOME"
STATUS_NEAR_ZERO = "NEAR_ZERO_VARIANCE"
STATUS_NEG_R2 = "NEGATIVE_TEST_R2"
STATUS_INSUFFICIENT = "INSUFFICIENT_ERRORS"
STATUS_NON_ESTIMABLE = "NON_ESTIMABLE"
STATUS_NUMERICAL = "NUMERICAL_FAILURE"

NEAR_ZERO_VAR = 1e-12
MIN_TEST_ERRORS = 5
NO_FIGURE = "FIGURE_NOT_CREATED_DUE_TO_NO_STABLE_INCREMENTAL_UTILITY"


# --------------------------------------------------------------- item table

def clean_items(canon: pd.DataFrame) -> pd.DataFrame:
    """One row per clean item: the item-level factors, seed-invariant."""
    d = canon[canon["in_LICHTHEIM_CLEAN"]]
    one = d[(d["seed"] == SEEDS[0]) & (d["route"] == "full")]
    out = one[["item_id", "source_lexicality", "target_length", "morphology",
               "lichtheim_exposure_status"]].copy()
    out = out.rename(columns={"source_lexicality": "lexicality",
                              "target_length": "length"})
    return out.sort_values("item_id").reset_index(drop=True)


def split_items(items: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """80/20 split over sorted unique item ids.  Grouped by construction:
    the split is drawn on items, so every route row of an item follows it."""
    ids = sorted(items["item_id"].unique())
    rng = np.random.RandomState(SPLIT_RANDOM_STATE)
    perm = rng.permutation(len(ids))
    n_test = int(round(SPLIT_TEST_SIZE * len(ids)))
    test = sorted(ids[i] for i in perm[:n_test])
    train = sorted(ids[i] for i in perm[n_test:])
    return train, test


def analysis_frame(canon: pd.DataFrame, seed: int,
                   regime: str = "LICHTHEIM_CLEAN") -> pd.DataFrame:
    """Seed x item x route rows carrying raw factors and both outcomes."""
    if regime == "LICHTHEIM_CLEAN":
        d = canon[canon["in_LICHTHEIM_CLEAN"]]
    elif regime == "ALL_WITH_EXPOSURE_STRATA":
        d = canon[canon["in_ALL_WITH_EXPOSURE_STRATA"]]
    else:
        raise ValueError(f"unknown regime {regime!r}")
    d = d[d["seed"] == seed]
    out = d[["item_id", "route", "source_lexicality", "target_length",
             "morphology", "lichtheim_exposure_status", PRIMARY_OUTCOME,
             SECONDARY_OUTCOME]].copy()
    out = out.rename(columns={"source_lexicality": "lexicality",
                              "target_length": "length"})
    return out.sort_values(["item_id", "route"]).reset_index(drop=True)


# ------------------------------------------------------------- design matrix

class Design:
    """Encoder + standardizer fitted on TRAINING rows only.

    Holding the fitted state in one object is what lets permutation rebuild
    every derived column from a permuted raw factor using the *training*
    transform, rather than refitting on permuted data.
    """

    def __init__(self, factors: Sequence[str], interactions: Sequence[str] = ()):
        self.factors = list(factors)
        self.interactions = list(interactions)
        self.length_mean_: Optional[float] = None
        self.length_std_: Optional[float] = None
        self.levels_: Dict[str, List[str]] = {}
        self.columns_: List[str] = []

    def fit(self, df: pd.DataFrame) -> "Design":
        if "length" in self.factors:
            L = df["length"].to_numpy(float)
            self.length_mean_ = float(L.mean())
            sd = float(L.std(ddof=0))
            self.length_std_ = sd if sd > NEAR_ZERO_VAR else 1.0
        for f in self.factors:
            if f == "length":
                continue
            ref = REFERENCE_LEVELS.get(f)
            seen = sorted(df[f].astype(str).unique())
            if ref is not None and ref in seen:
                self.levels_[f] = [ref] + [v for v in seen if v != ref]
            else:
                self.levels_[f] = seen
        self.columns_ = list(self._build(df).columns)
        return self

    def _base(self, df: pd.DataFrame) -> pd.DataFrame:
        cols: Dict[str, np.ndarray] = {}
        for f in self.factors:
            if f == "length":
                cols["length_z"] = ((df["length"].to_numpy(float)
                                     - self.length_mean_) / self.length_std_)
            else:
                for lvl in self.levels_[f][1:]:      # drop the reference level
                    cols[f"{f}_{lvl}"] = (df[f].astype(str) == lvl).to_numpy(float)
        return pd.DataFrame(cols, index=df.index)

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        base = self._base(df)
        if not self.interactions:
            return base
        out = base.copy()
        for block in self.interactions:
            a, b = block.split("_x_")
            for ca in _cols_of(base, a):
                for cb in _cols_of(base, b):
                    out[f"{ca}:{cb}"] = base[ca].to_numpy() * base[cb].to_numpy()
        return out

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = self._build(df)
        for c in self.columns_:                       # level unseen in this frame
            if c not in X.columns:
                X[c] = 0.0
        return X[self.columns_].to_numpy(float)


def _cols_of(base: pd.DataFrame, factor: str) -> List[str]:
    if factor == "length":
        return ["length_z"]
    return [c for c in base.columns if c.startswith(f"{factor}_")]


# -------------------------------------------------------------------- fitting

def _r2(y: np.ndarray, p: np.ndarray) -> float:
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float("nan") if ss_tot <= NEAR_ZERO_VAR else 1.0 - ss_res / ss_tot


def _mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.abs(y - p).mean())


def outcome_status(y_train: np.ndarray, y_test: np.ndarray) -> str:
    if not np.any(y_train) and not np.any(y_test):
        return STATUS_ALL_ZERO
    if float(y_test.std(ddof=0)) <= NEAR_ZERO_VAR:
        return STATUS_NEAR_ZERO
    if int((y_test > 0).sum()) < MIN_TEST_ERRORS:
        return STATUS_INSUFFICIENT
    return STATUS_OK


def fit_model(train: pd.DataFrame, test: pd.DataFrame, factors: Sequence[str],
              interactions: Sequence[str] = (),
              outcome: str = PRIMARY_OUTCOME) -> dict:
    """Fit Ridge on training rows and score on held-out rows."""
    y_tr = train[outcome].to_numpy(float)
    y_te = test[outcome].to_numpy(float)
    status = outcome_status(y_tr, y_te)
    res = {"outcome": outcome, "n_train_rows": int(len(train)),
           "n_test_rows": int(len(test)),
           "n_train_items": int(train["item_id"].nunique()),
           "n_test_items": int(test["item_id"].nunique()),
           "n_test_nonzero": int((y_te > 0).sum()),
           "ridge_alpha": RIDGE_ALPHA,
           # outcome_status diagnoses the DATA before fitting; model_status may
           # be sharpened afterwards by what the fitted score turns out to be.
           # Both are kept so neither limitation is hidden by the other.
           "outcome_status": status, "model_status": status,
           "negative_test_r2": False,
           "design": None, "estimator": None,
           "test_r2": np.nan, "test_mae": np.nan, "train_r2": np.nan,
           "coefficients": {}, "intercept": np.nan}
    if status == STATUS_ALL_ZERO:
        return res
    design = Design(factors, interactions).fit(train)
    X_tr, X_te = design.transform(train), design.transform(test)
    try:
        est = Ridge(alpha=RIDGE_ALPHA).fit(X_tr, y_tr)
    except Exception as exc:                                # pragma: no cover
        res["model_status"] = STATUS_NUMERICAL
        res["numerical_error"] = str(exc)
        return res
    p_te = est.predict(X_te)
    r2 = _r2(y_te, p_te)
    res.update({"design": design, "estimator": est,
                "test_r2": r2, "test_mae": _mae(y_te, p_te),
                "train_r2": _r2(y_tr, est.predict(X_tr)),
                "intercept": float(est.intercept_),
                "coefficients": dict(zip(design.columns_,
                                         [float(c) for c in est.coef_]))})
    # A negative held-out R2 is recorded wherever it occurs, whatever else is
    # also wrong with the cell: it is never suppressed.
    res["negative_test_r2"] = bool(np.isfinite(r2) and r2 < 0)
    if not np.isfinite(r2):
        res["model_status"] = STATUS_NON_ESTIMABLE
    elif status == STATUS_OK and r2 < 0:
        res["model_status"] = STATUS_NEG_R2
    return res


# --------------------------------------------- grouped raw-factor permutation

def _permute_raw(test: pd.DataFrame, factor: str,
                 rng: np.random.RandomState) -> pd.DataFrame:
    """Permute ONE raw factor, respecting the item grouping.

    Item-level factors are permuted across items and the permuted value is
    applied to all route rows of that item.  Route labels are permuted within
    each item, so every item keeps exactly one FULL, one WM and one LTM row.
    """
    out = test.copy()
    if factor == "route":
        idx = out.index.to_numpy()
        for _, pos in out.groupby("item_id", sort=True).indices.items():
            rows = idx[pos]
            out.loc[rows, "route"] = out.loc[rows, "route"].to_numpy()[
                rng.permutation(len(rows))]
        return out
    if factor not in ITEM_FACTORS:
        raise ValueError(f"unknown raw factor {factor!r}")
    ids = np.array(sorted(out["item_id"].unique()))
    vals = (out.drop_duplicates("item_id").set_index("item_id")
            .loc[ids, factor].to_numpy())
    mapping = dict(zip(ids, vals[rng.permutation(len(ids))]))
    out[factor] = out["item_id"].map(mapping)
    return out


def permutation_importance(fit: dict, test: pd.DataFrame,
                           factors: Sequence[str],
                           outcome: str = PRIMARY_OUTCOME) -> pd.DataFrame:
    """Grouped permutation importance, repeat-level.

    The design object is the one fitted on TRAINING rows, so permuting a raw
    factor rebuilds its encoded, standardized and interaction columns through
    the training transform.  No dummy or interaction column is touched directly.
    """
    if fit["estimator"] is None:
        return pd.DataFrame(columns=["factor", "repeat", "r2_drop", "mae_increase"])
    design, est = fit["design"], fit["estimator"]
    y = test[outcome].to_numpy(float)
    base_r2, base_mae = fit["test_r2"], fit["test_mae"]
    rows = []
    for factor in factors:
        rng = np.random.RandomState(PERM_RANDOM_STATE)
        for rep in range(PERM_REPEATS):
            perm = _permute_raw(test, factor, rng)
            p = est.predict(design.transform(perm))
            rows.append({"factor": factor, "repeat": rep,
                         "r2_drop": base_r2 - _r2(y, p),
                         "mae_increase": _mae(y, p) - base_mae})
    return pd.DataFrame(rows)


def summarise_repeats(reps: pd.DataFrame, **keys) -> pd.DataFrame:
    rows = []
    for factor, g in reps.groupby("factor", sort=False):
        row = {**keys, "factor": factor, "n_repeats": int(len(g))}
        for col, name in (("r2_drop", "r2_drop"),
                          ("mae_increase", "mae_increase")):
            v = g[col].to_numpy(float)
            row.update({f"{name}_mean": float(v.mean()),
                        f"{name}_std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                        f"{name}_min": float(v.min()),
                        f"{name}_max": float(v.max())})
        rows.append(row)
    return pd.DataFrame(rows)


def rank_factors(summary: pd.DataFrame, by: str = "r2_drop_mean",
                 group_cols: Sequence[str] = ("seed",)) -> pd.DataFrame:
    out = []
    for key, g in summary.groupby(list(group_cols), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        g = g.copy()
        g["rank"] = g[by].rank(ascending=False, method="min").astype(int)
        for c, v in zip(group_cols, key):
            g[c] = v
        out.append(g[list(group_cols) + ["factor", by, "rank"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def rank_stability(ranks: pd.DataFrame, by: str = "r2_drop_mean") -> pd.DataFrame:
    rows = []
    for factor, g in ranks.groupby("factor", sort=False):
        r = g.set_index("seed")["rank"]
        present = [s for s in SEEDS if s in r.index]
        vals = r.loc[present]
        rows.append({"factor": factor,
                     "ranks_by_seed": "; ".join(f"{s}:{int(vals.loc[s])}"
                                                for s in present),
                     "modal_rank": int(vals.mode().iloc[0]),
                     "min_rank": int(vals.min()), "max_rank": int(vals.max()),
                     "rank_range": int(vals.max() - vals.min()),
                     "identical_in_all_seeds": bool(vals.nunique() == 1),
                     "stability": ("STABLE_RANKING" if vals.nunique() == 1
                                   else "CONSISTENT_BUT_SMALL"
                                   if int(vals.max() - vals.min()) <= 1
                                   else "UNSTABLE_ACROSS_SEEDS")})
    return pd.DataFrame(rows)


def seed_interval(values: Sequence[float]) -> dict:
    """Resample the four checkpoints with replacement.

    Deliberately NOT called a hierarchical bootstrap: items are not resampled
    and the model is not refitted, so this describes checkpoint-to-checkpoint
    spread only.
    """
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"seed_interval_low": np.nan, "seed_interval_high": np.nan,
                "seed_interval_label": SEED_INTERVAL_LABEL,
                "n_resamples": SEED_RESAMPLES,
                "random_seed": SEED_RESAMPLE_RANDOM_STATE,
                "is_hierarchical_bootstrap": False}
    rng = np.random.default_rng(SEED_RESAMPLE_RANDOM_STATE)
    draws = v[rng.integers(0, v.size, size=(SEED_RESAMPLES, v.size))].mean(axis=1)
    return {"seed_interval_low": float(np.percentile(draws, 2.5)),
            "seed_interval_high": float(np.percentile(draws, 97.5)),
            "seed_interval_label": SEED_INTERVAL_LABEL,
            "n_resamples": SEED_RESAMPLES,
            "random_seed": SEED_RESAMPLE_RANDOM_STATE,
            "is_hierarchical_bootstrap": False}


def seed_summary(summary: pd.DataFrame, value_col: str,
                 group_cols: Sequence[str] = ("factor",)) -> pd.DataFrame:
    rows = []
    for key, g in summary.groupby(list(group_cols), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        vals = g.set_index("seed")[value_col]
        present = [s for s in SEEDS if s in vals.index]
        v = vals.loc[present].dropna()
        ceil = [s for s in CEILING_SEEDS if s in vals.index]
        rows.append({**dict(zip(group_cols, key)), "quantity": value_col,
                     "seed_values": "; ".join(f"{s}:{vals.loc[s]:+.6f}"
                                              for s in present),
                     "mean_over_seeds": float(v.mean()) if len(v) else np.nan,
                     "min": float(v.min()) if len(v) else np.nan,
                     "max": float(v.max()) if len(v) else np.nan,
                     "range": float(v.max() - v.min()) if len(v) else np.nan,
                     "seed21_included": 21 in present,
                     "exact_zero_seeds_mean": float(
                         vals.loc[ceil].dropna().mean()) if ceil else np.nan,
                     **seed_interval(v.tolist())})
    return pd.DataFrame(rows)


def coefficient_rows(fit: dict, **keys) -> pd.DataFrame:
    """Coefficients, kept strictly separate from unsigned importance."""
    rows = []
    for name, val in fit["coefficients"].items():
        rows.append({**keys, "term": name, "coefficient": val,
                     "reference_levels": "route=wm; lexicality=pseudo; "
                                         "morphology=complex",
                     "signed": True,
                     "note": "coefficients are signed; grouped permutation "
                             "importance is unsigned and lives in a separate "
                             "table"})
    if rows:
        rows.append({**keys, "term": "(intercept)",
                     "coefficient": fit["intercept"],
                     "reference_levels": "route=wm; lexicality=pseudo; "
                                         "morphology=complex",
                     "signed": True, "note": "model intercept"})
    return pd.DataFrame(rows)
