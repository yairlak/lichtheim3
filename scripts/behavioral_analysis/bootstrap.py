"""Frozen estimators: OLS length slope and the hierarchical bootstrap.

Promoted verbatim (logic unchanged) from the validated analysis driver
outputs/.../behavioral_analysis/_control/analysis_lib.py.  The numerical
behaviour is part of the frozen protocol: B = 10,000, random seed 20260730,
95 % percentile interval, seeds resampled first and items second within each
analysis-set x stratum cell.

Items are identical and identically ordered across seeds, so one item-index
resample per stratum is applied to every seed and route within a replicate.
That preserves the pairing of routes on items, which the LTM - WM contrast
requires.
"""
from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import numpy as np

from .common import BOOTSTRAP_CI_LEVEL, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED


def ols_slope(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """(intercept, slope) of y ~ x.  (nan, nan) when x has no variance."""
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    xm, ym = x.mean(), y.mean()
    dx = x - xm
    denom = float((dx * dx).sum())
    if denom == 0.0:
        return float("nan"), float("nan")
    slope = float((dx * (y - ym)).sum() / denom)
    return float(ym - slope * xm), slope


def _slopes_batch(x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Vectorised slopes for a batch of item resamples.

    x   : (n,)    lengths
    y   : (n,)    metric values for one seed/route/stratum
    idx : (B, n)  resampled item indices
    """
    xs, ys = x[idx], y[idx]
    xm = xs.mean(axis=1, keepdims=True)
    ym = ys.mean(axis=1, keepdims=True)
    dx = xs - xm
    denom = (dx * dx).sum(axis=1)
    num = (dx * (ys - ym)).sum(axis=1)
    out = np.full(denom.shape, np.nan)
    nz = denom != 0
    out[nz] = num[nz] / denom[nz]
    return out


def hierarchical_bootstrap(
    x_by_stratum: Dict[str, np.ndarray],
    y_by_seed_route_stratum: Dict[Tuple[int, str, str], np.ndarray],
    seeds: Sequence[int],
    statistic: Callable[[Dict[Tuple[str, str], np.ndarray]], np.ndarray],
    b: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    chunk: int = 500,
) -> dict:
    """Frozen hierarchical bootstrap; returns mean and percentile interval."""
    rng = np.random.default_rng(seed)
    n_seeds = len(seeds)
    routes = sorted({k[1] for k in y_by_seed_route_stratum})
    strata = sorted({k[2] for k in y_by_seed_route_stratum})
    draws, done = [], 0
    while done < b:
        m = min(chunk, b - done)
        seed_idx = rng.integers(0, n_seeds, size=(m, n_seeds))
        idx = {s: rng.integers(0, len(v), size=(m, len(v)))
               for s, v in x_by_stratum.items()}
        per: Dict[Tuple[str, str], np.ndarray] = {}
        for rt in routes:
            for st in strata:
                sl = np.stack([
                    _slopes_batch(x_by_stratum[st],
                                  y_by_seed_route_stratum[(sd, rt, st)],
                                  idx[st])
                    for sd in seeds])                          # (n_seeds, m)
                take = sl[seed_idx.T, np.arange(m)[None, :]]
                with np.errstate(invalid="ignore"):
                    per[(rt, st)] = np.nanmean(take, axis=0)
        draws.append(statistic(per))
        done += m
    vals = np.concatenate(draws)
    vals = vals[np.isfinite(vals)]
    lo = (100 - BOOTSTRAP_CI_LEVEL) / 2
    return {
        "bootstrap_mean": float(np.mean(vals)) if vals.size else float("nan"),
        "ci_low": float(np.percentile(vals, lo)) if vals.size else float("nan"),
        "ci_high": (float(np.percentile(vals, 100 - lo))
                    if vals.size else float("nan")),
        "n_replicates": int(b),
        "n_finite_replicates": int(vals.size),
        "random_seed": int(seed),
        "ci_definition": f"{BOOTSTRAP_CI_LEVEL}% percentile interval",
    }


def cell_mean_bootstrap(y_by_seed: Dict[int, np.ndarray], seeds: Sequence[int],
                        b: int = BOOTSTRAP_REPLICATES,
                        seed: int = BOOTSTRAP_SEED,
                        chunk: int = 1000) -> Tuple[float, float]:
    """Bootstrap interval for one (route, lexicality, length) cell mean.

    Same two-level scheme as `hierarchical_bootstrap`, applied to a plain cell
    mean rather than a slope.  Promoted unchanged from the validated figure-1
    driver so the published band is reproduced exactly.
    """
    n = len(next(iter(y_by_seed.values())))
    if n < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(b)
    for start in range(0, b, chunk):
        m = min(chunk, b - start)
        sidx = rng.integers(0, len(seeds), size=(m, len(seeds)))
        iidx = rng.integers(0, n, size=(m, n))
        per_seed = np.stack([y_by_seed[s][iidx].mean(axis=1) for s in seeds])
        draws[start:start + m] = per_seed[sidx.T, np.arange(m)[None, :]].mean(axis=0)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
