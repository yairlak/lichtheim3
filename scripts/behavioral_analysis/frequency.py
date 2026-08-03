"""Sprint 3 — word-frequency statistics for trained real words.

Implements exactly the estimands frozen in
reports/behavioral_wfe_fulllexicon_93a577f/frequency/_control/
frequency_analysis_spec.json, written before any frequency result was computed.

Sign conventions (frozen):
    zipf_slope                          negative => higher frequency, fewer errors
    high_low_contrast                   low_mean - high_mean; positive => low harder
    raw_route_slope_difference          slope_A - slope_B
    frequency_benefit_route_difference  (-slope_A) - (-slope_B) = slope_B - slope_A
                                        positive => route A has the larger benefit

Frequency exists only for real words: no pseudoword is ever assigned a Zipf
value or admitted to a model here.  Nothing in this module loads a model.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .bootstrap import hierarchical_bootstrap, ols_slope
from .common import CEILING_SEEDS, ROUTES, SEEDS

ZIPF_LOW_MAX = 3.5
ZIPF_HIGH_MIN = 4.0

REGIMES = {
    "TRAINED_REAL_FREQUENCY_PRIMARY": {"flag": "in_TRAINED_REAL_FREQUENCY_PRIMARY",
                                       "n": 671},
    "TRAINED_REAL_FREQUENCY_SENSITIVITY": {
        "flag": "in_TRAINED_REAL_FREQUENCY_SENSITIVITY", "n": 678},
    "UNTRAINED_REAL": {"exposure": "UNTRAINED_REAL", "n": 122},
    "FAITHFUL_ALL_REAL": {"lexicality": "real", "n": 800},
}

MODEL_OK = "OK"
MODEL_ALL_ZERO = "ALL_ZERO_OUTCOME"
MODEL_CONSTANT = "CONSTANT_OUTCOME"
MODEL_NON_ESTIMABLE = "NON_ESTIMABLE"
MODEL_INSUFFICIENT = "INSUFFICIENT_ERRORS"
MODEL_SEPARATION = "COMPLETE_SEPARATION"


def regime_subset(canon: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Rows for one frequency regime.  Real words only, by construction."""
    spec = REGIMES[regime]
    if "flag" in spec:
        d = canon[canon[spec["flag"]]]
    elif "exposure" in spec:
        d = canon[canon["lichtheim_exposure_status"] == spec["exposure"]]
    else:
        d = canon[(canon["source_lexicality"] == "real")
                  & canon["in_FAITHFUL_WFE_ALL"]]
    d = d.copy()
    if not (d["source_lexicality"] == "real").all():
        raise ValueError(f"{regime}: non-real items present in a frequency set")
    return d


def standardized_covariates(canon: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Item-level standardized Zipf and length, fixed once per regime.

    Items are identical across seeds and routes, so the moments are computed
    from a single seed/route slice and the resulting per-item values are reused
    unchanged in every model within the regime.  Nothing is re-standardized per
    seed, per route or per fit.
    """
    d = regime_subset(canon, regime)
    base = d[(d["seed"] == SEEDS[0]) & (d["route"] == "full")] \
        .sort_values("item_id")
    if base.empty:
        raise ValueError(
            f"{regime}: no rows for seed {SEEDS[0]} / route 'full'; the "
            "standardization anchor slice must exist so that one fixed set of "
            "item covariates can be reused across every seed and route")
    z = pd.to_numeric(base["zipf_frequency"], errors="coerce").to_numpy(float)
    L = base["target_length"].to_numpy(float)
    if not np.isfinite(z).all():
        raise ValueError(f"{regime}: non-finite Zipf values")
    out = pd.DataFrame({
        "item_id": base["item_id"].to_numpy(),
        "zipf": z, "phoneme_length": L,
        "zipf_mean": z.mean(), "zipf_sd": z.std(ddof=0),
        "length_mean": L.mean(), "length_sd": L.std(ddof=0),
        "standardized_zipf": (z - z.mean()) / z.std(ddof=0),
        "standardized_phoneme_length": (L - L.mean()) / L.std(ddof=0),
        "frequency_class": base["frequency_class"].to_numpy(),
        "lichtheim_exposure_status":
            base["lichtheim_exposure_status"].to_numpy(),
    })
    return out


def verify_frequency_classes(cov: pd.DataFrame) -> Dict[str, object]:
    """Recompute high/low from Zipf rather than trusting the stored label."""
    recomputed = np.where(cov["zipf"] <= ZIPF_LOW_MAX, "low",
                          np.where(cov["zipf"] >= ZIPF_HIGH_MIN, "high",
                                   "AMBIGUOUS"))
    return {
        "n_items": int(len(cov)),
        "n_mismatched_labels": int((recomputed != cov["frequency_class"]).sum()),
        "n_in_excluded_gap": int((recomputed == "AMBIGUOUS").sum()),
        "n_low": int((recomputed == "low").sum()),
        "n_high": int((recomputed == "high").sum()),
        "labels_verified_from_zipf": True,
    }


def _cell_status(y: np.ndarray, x: np.ndarray) -> str:
    if len(y) < 3:
        return MODEL_NON_ESTIMABLE
    if np.allclose(y, 0.0):
        return MODEL_ALL_ZERO
    if np.std(y) == 0.0:
        return MODEL_CONSTANT
    if np.std(x) == 0.0:
        return MODEL_NON_ESTIMABLE
    return MODEL_OK


# ------------------------------------------------- 5.1 continuous slope

def continuous_slopes(canon: pd.DataFrame, regime: str,
                      metric: str = "raw_edit_distance") -> pd.DataFrame:
    d = regime_subset(canon, regime)
    cov = standardized_covariates(canon, regime).set_index("item_id")
    rows = []
    for seed in SEEDS:
        for route in sorted(d["route"].unique()):
            sub = d[(d["seed"] == seed) & (d["route"] == route)] \
                .set_index("item_id").loc[cov.index]
            y = sub[metric].to_numpy(float)
            x = cov["standardized_zipf"].to_numpy(float)
            status = _cell_status(y, x)
            if status == MODEL_OK:
                b0, b1 = ols_slope(x, y)
            elif status == MODEL_ALL_ZERO:
                b0, b1 = 0.0, 0.0        # structurally zero, per ceiling policy
            else:
                b0, b1 = np.nan, np.nan
            rows.append({
                "dataset_regime": regime, "seed": seed, "route": route,
                "metric": metric, "n_items": int(len(y)),
                "n_erroneous_items": int((sub["word_error"] == 1).sum()),
                "total_raw_edit_distance": float(sub["raw_edit_distance"].sum()),
                "intercept": b0, "zipf_slope": b1, "model_status": status,
                "sign_convention":
                    "negative slope = higher frequency, fewer errors"})
    return pd.DataFrame(rows)


# ------------------------------------------- 5.2 length-adjusted model

def _ols_multi(X: np.ndarray, y: np.ndarray):
    """Least-squares with an intercept column already present."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def zipf_length_models(canon: pd.DataFrame, regime: str) -> pd.DataFrame:
    d = regime_subset(canon, regime)
    cov = standardized_covariates(canon, regime).set_index("item_id")
    z = cov["standardized_zipf"].to_numpy(float)
    L = cov["standardized_phoneme_length"].to_numpy(float)
    X = np.column_stack([np.ones_like(z), z, L, z * L])
    rows = []
    for seed in SEEDS:
        for route in sorted(d["route"].unique()):
            sub = d[(d["seed"] == seed) & (d["route"] == route)] \
                .set_index("item_id").loc[cov.index]
            y = sub["raw_edit_distance"].to_numpy(float)
            status = _cell_status(y, z)
            if status == MODEL_OK:
                c = _ols_multi(X, y)
                b0, bz, bl, bi = (float(v) for v in c)
            elif status == MODEL_ALL_ZERO:
                b0 = bz = bl = bi = 0.0
            else:
                b0 = bz = bl = bi = np.nan
            rows.append({
                "dataset_regime": regime, "seed": seed, "route": route,
                "n_items": int(len(y)), "intercept": b0,
                "zipf_main_effect": bz, "length_coefficient": bl,
                "zipf_x_length_interaction": bi, "model_status": status,
                "centering_note": ("both predictors centred: the Zipf main "
                                   "effect is the slope at mean length")})
    return pd.DataFrame(rows)


# ------------------------------------------------ 5.3 high vs low

def high_low_contrasts(canon: pd.DataFrame, regime: str) -> pd.DataFrame:
    d = regime_subset(canon, regime)
    cov = standardized_covariates(canon, regime)
    cls = dict(zip(cov["item_id"], np.where(cov["zipf"] <= ZIPF_LOW_MAX, "low",
                                            np.where(cov["zipf"] >= ZIPF_HIGH_MIN,
                                                     "high", "AMBIGUOUS"))))
    d = d.assign(freq_class_verified=d["item_id"].map(cls))
    rows = []
    for seed in SEEDS:
        for route in sorted(d["route"].unique()):
            sub = d[(d["seed"] == seed) & (d["route"] == route)]
            lo = sub[sub["freq_class_verified"] == "low"]
            hi = sub[sub["freq_class_verified"] == "high"]
            row = {"dataset_regime": regime, "seed": seed, "route": route,
                   "n_low": len(lo), "n_high": len(hi),
                   "sign_convention":
                       "positive = low-frequency words harder"}
            for metric in ("raw_edit_distance", "word_error"):
                ml = float(lo[metric].mean()) if len(lo) else np.nan
                mh = float(hi[metric].mean()) if len(hi) else np.nan
                row[f"low_mean_{metric}"] = ml
                row[f"high_mean_{metric}"] = mh
                row[f"high_low_contrast_{metric}"] = ml - mh
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------ 5.4 route contrasts

def route_contrasts(slopes: pd.DataFrame) -> pd.DataFrame:
    pairs = [("ltm", "wm", "ltm_minus_wm"), ("full", "wm", "full_minus_wm"),
             ("ltm", "full", "ltm_minus_full")]
    rows = []
    for seed in SEEDS:
        for a, b, name in pairs:
            ra = slopes[(slopes["seed"] == seed) & (slopes["route"] == a)]
            rb = slopes[(slopes["seed"] == seed) & (slopes["route"] == b)]
            if ra.empty or rb.empty:
                continue
            sa = float(ra["zipf_slope"].iloc[0])
            sb = float(rb["zipf_slope"].iloc[0])
            rows.append({
                "dataset_regime": slopes["dataset_regime"].iloc[0],
                "seed": seed, "route_contrast": name,
                "slope_route_A": sa, "slope_route_B": sb,
                "raw_route_slope_difference": sa - sb,
                "frequency_benefit_route_difference": (-sa) - (-sb),
                "raw_difference_meaning": f"slope({a}) - slope({b})",
                "benefit_meaning": (f"positive = {a.upper()} shows the larger "
                                    f"frequency benefit than {b.upper()}"),
                "status_route_A": ra["model_status"].iloc[0],
                "status_route_B": rb["model_status"].iloc[0]})
    return pd.DataFrame(rows)


# ------------------------------------- 5.5 / 5.6 confidence and gate

def confidence_gate_slopes(canon: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Zipf slopes for lexical confidence and gate (FULL-route quantities)."""
    d = regime_subset(canon, regime)
    d = d[d["route"] == "full"]
    cov = standardized_covariates(canon, regime).set_index("item_id")
    x = cov["standardized_zipf"].to_numpy(float)
    rows = []
    for seed in SEEDS:
        sub = d[d["seed"] == seed].set_index("item_id").loc[cov.index]
        for outcome, label in (("lexical_confidence",
                                "top-1 cosine similarity to the semantic bank"),
                               ("gate", "word-level FULL-route gate")):
            y = pd.to_numeric(sub[outcome], errors="coerce").to_numpy(float)
            status = MODEL_OK if np.isfinite(y).all() and np.std(y) > 0 \
                else MODEL_NON_ESTIMABLE
            b0, b1 = ols_slope(x, y) if status == MODEL_OK else (np.nan, np.nan)
            rows.append({
                "dataset_regime": regime, "seed": seed, "outcome": outcome,
                "outcome_definition": label, "n_items": int(len(y)),
                "intercept": b0, "zipf_slope": b1, "model_status": status,
                "linked_outcomes_note": ("gate is a deterministic monotonic "
                                         "transform of lexical confidence; the "
                                         "two are linked, not independent")})
    return pd.DataFrame(rows)


# ------------------------------------------------------- bootstrap

def bootstrap_slopes(canon: pd.DataFrame, regime: str,
                     metric: str = "raw_edit_distance",
                     outcome_is_item_level: bool = True) -> pd.DataFrame:
    """Hierarchical bootstrap of the continuous Zipf slope per route."""
    d = regime_subset(canon, regime)
    cov = standardized_covariates(canon, regime).set_index("item_id")
    ids = cov.index.tolist()
    x_by = {"all": cov["standardized_zipf"].to_numpy(float)}
    routes = sorted(d["route"].unique())
    y_by: Dict[Tuple[int, str, str], np.ndarray] = {}
    for seed in SEEDS:
        for route in routes:
            sub = d[(d["seed"] == seed) & (d["route"] == route)] \
                .set_index("item_id").loc[ids]
            y_by[(seed, route, "all")] = pd.to_numeric(
                sub[metric], errors="coerce").to_numpy(float)
    rows = []
    for route in routes:
        res = hierarchical_bootstrap(
            x_by, y_by, SEEDS, lambda per, r=route: per[(r, "all")])
        rows.append({"dataset_regime": regime, "route": route, "metric": metric,
                     "quantity": "zipf_slope", **res})
    for a, b, name in (("ltm", "wm", "ltm_minus_wm"),
                       ("full", "wm", "full_minus_wm"),
                       ("ltm", "full", "ltm_minus_full")):
        if a in routes and b in routes:
            res = hierarchical_bootstrap(
                x_by, y_by, SEEDS,
                lambda per, a=a, b=b: per[(a, "all")] - per[(b, "all")])
            rows.append({"dataset_regime": regime, "route": name,
                         "metric": metric,
                         "quantity": "raw_route_slope_difference", **res})
    return pd.DataFrame(rows)


def bootstrap_confidence_gate(canon: pd.DataFrame, regime: str) -> pd.DataFrame:
    d = regime_subset(canon, regime)
    d = d[d["route"] == "full"]
    cov = standardized_covariates(canon, regime).set_index("item_id")
    ids = cov.index.tolist()
    x_by = {"all": cov["standardized_zipf"].to_numpy(float)}
    rows = []
    for outcome in ("lexical_confidence", "gate"):
        y_by = {}
        for seed in SEEDS:
            sub = d[d["seed"] == seed].set_index("item_id").loc[ids]
            y_by[(seed, "full", "all")] = pd.to_numeric(
                sub[outcome], errors="coerce").to_numpy(float)
        res = hierarchical_bootstrap(x_by, y_by, SEEDS,
                                     lambda per: per[("full", "all")])
        rows.append({"dataset_regime": regime, "outcome": outcome,
                     "quantity": "zipf_slope", **res})
    return pd.DataFrame(rows)


# --------------------------------------------------------- summaries

def summarise(df: pd.DataFrame, value_col: str,
              group_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(list(group_cols), dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        vals = g.set_index("seed")[value_col]
        present = [s for s in SEEDS if s in vals.index]
        v = vals.loc[present].dropna()
        ceiling = [s for s in CEILING_SEEDS if s in vals.index]
        rows.append({
            **dict(zip(group_cols, key)), "quantity": value_col,
            "seed_values": "; ".join(
                f"{s}:{vals.loc[s]:+.6f}" if pd.notna(vals.loc[s]) else f"{s}:nan"
                for s in present),
            "n_seeds": len(present),
            "mean_over_seeds": float(v.mean()) if len(v) else np.nan,
            "min": float(v.min()) if len(v) else np.nan,
            "max": float(v.max()) if len(v) else np.nan,
            "range": float(v.max() - v.min()) if len(v) else np.nan,
            "all_same_sign": bool((v > 0).all() or (v < 0).all()) if len(v) else False,
            "seed21_included": 21 in present,
            "exact_zero_seeds_mean":
                float(vals.loc[ceiling].dropna().mean()) if ceiling else np.nan,
        })
    return pd.DataFrame(rows)
