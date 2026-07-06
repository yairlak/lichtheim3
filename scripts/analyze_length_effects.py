"""Quantify length effects from teacher-forced and autoregressive WFE predictions.

For each group × route × decode regime, computes:
  - Per-length-bin summary (mean ± 95 % bootstrap CI)
  - Length-effect slope  (item-level OLS: metric ~ length_phonemes, bootstrapped CI)
  - Long-short contrast  (mean[long] − mean[short], bootstrapped CI)
  - AR − TF delta for slopes and contrasts (tells whether AR worsens the length effect)

Grouping modes
--------------
dager_strict (Dager-comparable):
  train_seen_real  = lexicon_category == real_word_seen_in_training_lexicon
  pseudoword       = lexicality in {pseudo, pseudoword}
  (held-out / novel real excluded)

seen_vs_unseen (generalization analysis — NOT a lexicality analysis):
  train_seen_real  = lexicon_category == real_word_seen_in_training_lexicon
  unseen_forms     = all other items (held-out real + novel real + pseudowords)

Outputs
-------
  per_length_summary.tsv
  length_effect_slopes.tsv
  long_short_contrasts.tsv
  ar_minus_tf_length_effects.tsv
  length_effect_slopes_edit_dist.png
  length_effect_slopes_error_rate.png
  ar_minus_tf_delta_slopes.png
  README.md

Usage
-----
    python scripts/analyze_length_effects.py

    python scripts/analyze_length_effects.py \\
        --tf_pred outputs/external_eval_30k/wfe/item_level_predictions.tsv \\
        --ar_pred outputs/external_eval_30k/wfe_ar/item_level_predictions.tsv \\
        --out_dir outputs/length_effect_analysis \\
        --n_boot 1000 --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TF_PRED_DEFAULT = os.path.join(ROOT, "outputs", "external_eval_30k",
                                "wfe", "item_level_predictions.tsv")
AR_PRED_DEFAULT = os.path.join(ROOT, "outputs", "external_eval_30k",
                                "wfe_ar", "item_level_predictions.tsv")
OUT_DEFAULT     = os.path.join(ROOT, "outputs", "length_effect_analysis")

GROUP_DAGER       = "dager_strict"
GROUP_SEEN_UNSEEN = "seen_vs_unseen"
ALL_GROUP_MODES   = [GROUP_DAGER, GROUP_SEEN_UNSEEN]

_SEEN_CAT    = "real_word_seen_in_training_lexicon"
_ROUTES      = ("full", "wm", "ltm")
_DECODES     = ("teacher_forced", "autoregressive")
_PSEUDO_LEX  = {"pseudo", "pseudoword"}

SHORT_LENGTHS = {3, 4, 5}
LONG_LENGTHS  = {7, 8, 9}

MIN_N_SLOPE    = 5   # minimum items to compute a slope
MIN_N_CONTRAST = 3   # minimum items per short / long bin


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Item-level OLS slope y ~ x.  Returns NaN if underdetermined."""
    if len(x) < 2:
        return np.nan
    xm = x.mean()
    var = float(((x - xm) ** 2).mean())
    if var < 1e-12:
        return 0.0
    return float(((x - xm) * (y - y.mean())).mean() / var)


def _bootstrap_ci(values: np.ndarray, n_boot: int,
                  rng: np.random.RandomState) -> Tuple[float, float]:
    """95 % bootstrap CI for the mean."""
    if len(values) < 2 or n_boot == 0:
        m = float(values.mean()) if len(values) > 0 else np.nan
        return m, m
    boot = np.fromiter(
        (rng.choice(values, size=len(values), replace=True).mean()
         for _ in range(n_boot)),
        dtype=float, count=n_boot,
    )
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _bootstrap_slope_ci(x: np.ndarray, y: np.ndarray,
                         n_boot: int, rng: np.random.RandomState,
                         ) -> Tuple[float, float]:
    """95 % bootstrap CI for the OLS slope y ~ x."""
    if n_boot == 0 or len(x) < MIN_N_SLOPE:
        return np.nan, np.nan
    n = len(x)
    slopes = np.fromiter(
        (_ols_slope(x[idx], y[idx])
         for idx in (rng.randint(0, n, size=n) for _ in range(n_boot))),
        dtype=float, count=n_boot,
    )
    return float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))


def _bootstrap_contrast_ci(vals_long: np.ndarray, vals_short: np.ndarray,
                             n_boot: int, rng: np.random.RandomState,
                             ) -> Tuple[float, float]:
    """95 % bootstrap CI for mean(long) − mean(short)."""
    if n_boot == 0 or len(vals_long) < MIN_N_CONTRAST or len(vals_short) < MIN_N_CONTRAST:
        return np.nan, np.nan
    contrasts = np.fromiter(
        (rng.choice(vals_long,  size=len(vals_long),  replace=True).mean() -
         rng.choice(vals_short, size=len(vals_short), replace=True).mean()
         for _ in range(n_boot)),
        dtype=float, count=n_boot,
    )
    return float(np.percentile(contrasts, 2.5)), float(np.percentile(contrasts, 97.5))


# ---------------------------------------------------------------------------
# Data loading and group building (mirrors plot_tf_vs_ar.py)
# ---------------------------------------------------------------------------

def _load_and_align(tf_path: str, ar_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_tf = pd.read_csv(tf_path, sep="\t")
    df_ar = pd.read_csv(ar_path, sep="\t")
    df_tf = df_tf[df_tf["notes"].fillna("").str.strip() == ""].reset_index(drop=True)
    df_ar = df_ar[df_ar["notes"].fillna("").str.strip() == ""].reset_index(drop=True)
    print(f"  TF: {len(df_tf)} items  |  AR: {len(df_ar)} items")
    if len(df_tf) != len(df_ar):
        raise ValueError(f"TF ({len(df_tf)}) and AR ({len(df_ar)}) row counts differ.")
    return df_tf, df_ar


def _is_pseudo(df: pd.DataFrame) -> pd.Series:
    return df["lexicality"].str.lower().str.strip().isin(_PSEUDO_LEX)


def build_groups(df_tf: pd.DataFrame, df_ar: pd.DataFrame,
                 group_mode: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Return {group_label: {"tf": sub_tf, "ar": sub_ar}} positionally aligned."""
    if group_mode == GROUP_DAGER:
        mask_real = df_tf["lexicon_category"] == _SEEN_CAT
        mask_pseu = _is_pseudo(df_tf)
        return {
            "Train-seen real": {
                "tf": df_tf[mask_real].reset_index(drop=True),
                "ar": df_ar[mask_real].reset_index(drop=True),
            },
            "Pseudowords": {
                "tf": df_tf[mask_pseu].reset_index(drop=True),
                "ar": df_ar[mask_pseu].reset_index(drop=True),
            },
        }
    elif group_mode == GROUP_SEEN_UNSEEN:
        mask_seen = df_tf["lexicon_category"] == _SEEN_CAT
        return {
            "Train-seen real": {
                "tf": df_tf[mask_seen].reset_index(drop=True),
                "ar": df_ar[mask_seen].reset_index(drop=True),
            },
            "Unseen forms": {
                "tf": df_tf[~mask_seen].reset_index(drop=True),
                "ar": df_ar[~mask_seen].reset_index(drop=True),
            },
        }
    raise ValueError(f"Unknown group_mode: {group_mode!r}")


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _metric_arrays(sub: pd.DataFrame, route: str
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (length, error_rate, edit_dist) arrays."""
    length   = sub["length_phonemes"].values.astype(float)
    error    = (1.0 - sub[f"{route}_exact_match"]).values.astype(float)
    edit_dist = sub[f"{route}_edit_dist"].values.astype(float)
    return length, error, edit_dist


def compute_per_length_summary(sub: pd.DataFrame, route: str,
                                n_boot: int, rng: np.random.RandomState,
                                group_mode: str, group_name: str,
                                decode: str) -> List[dict]:
    length, error, edit_dist = _metric_arrays(sub, route)
    rows = []
    for L in sorted(np.unique(length)):
        mask = length == L
        e_vals = error[mask]
        d_vals = edit_dist[mask]
        n = int(mask.sum())
        e_mean = float(e_vals.mean())
        d_mean = float(d_vals.mean())
        e_lo, e_hi = _bootstrap_ci(e_vals, n_boot, rng)
        d_lo, d_hi = _bootstrap_ci(d_vals, n_boot, rng)
        rows.append({
            "group_mode":       group_mode,
            "group_name":       group_name,
            "decode":           decode,
            "route":            route,
            "length_phonemes":  int(L),
            "n_items":          n,
            "error_rate_mean":  round(e_mean, 5),
            "error_rate_ci_lo": round(e_lo,   5),
            "error_rate_ci_hi": round(e_hi,   5),
            "edit_dist_mean":   round(d_mean, 5),
            "edit_dist_ci_lo":  round(d_lo,   5),
            "edit_dist_ci_hi":  round(d_hi,   5),
        })
    return rows


def compute_slopes(sub: pd.DataFrame, route: str,
                   n_boot: int, rng: np.random.RandomState,
                   group_mode: str, group_name: str, decode: str) -> List[dict]:
    length, error, edit_dist = _metric_arrays(sub, route)
    n = len(length)
    rows = []
    for metric_name, vals in [("error_rate", error), ("edit_dist", edit_dist)]:
        if n < MIN_N_SLOPE:
            slope = np.nan
            ci_lo = ci_hi = np.nan
        else:
            slope = _ols_slope(length, vals)
            ci_lo, ci_hi = _bootstrap_slope_ci(length, vals, n_boot, rng)
        rows.append({
            "group_mode":  group_mode,
            "group_name":  group_name,
            "decode":      decode,
            "route":       route,
            "metric":      metric_name,
            "slope":       round(slope, 6) if np.isfinite(slope) else np.nan,
            "ci_lo":       round(ci_lo, 6) if np.isfinite(ci_lo) else np.nan,
            "ci_hi":       round(ci_hi, 6) if np.isfinite(ci_hi) else np.nan,
            "n_items":     n,
            "n_boot":      n_boot,
            "note":        ("positive = longer words worse" if np.isfinite(slope)
                            else f"insufficient data (n={n} < {MIN_N_SLOPE})"),
        })
    return rows


def compute_long_short(sub: pd.DataFrame, route: str,
                        n_boot: int, rng: np.random.RandomState,
                        group_mode: str, group_name: str, decode: str) -> List[dict]:
    length, error, edit_dist = _metric_arrays(sub, route)
    mask_short = np.isin(length, list(SHORT_LENGTHS))
    mask_long  = np.isin(length, list(LONG_LENGTHS))
    rows = []
    for metric_name, vals in [("error_rate", error), ("edit_dist", edit_dist)]:
        v_s = vals[mask_short]
        v_l = vals[mask_long]
        n_s, n_l = int(mask_short.sum()), int(mask_long.sum())
        if n_s >= MIN_N_CONTRAST and n_l >= MIN_N_CONTRAST:
            m_l = float(v_l.mean())
            m_s = float(v_s.mean())
            contrast = m_l - m_s
            ci_lo, ci_hi = _bootstrap_contrast_ci(v_l, v_s, n_boot, rng)
        else:
            m_l = m_s = contrast = np.nan
            ci_lo = ci_hi = np.nan
        rows.append({
            "group_mode":              group_mode,
            "group_name":              group_name,
            "decode":                  decode,
            "route":                   route,
            "metric":                  metric_name,
            "mean_long":               round(m_l, 5) if np.isfinite(m_l) else np.nan,
            "mean_short":              round(m_s, 5) if np.isfinite(m_s) else np.nan,
            "contrast_long_minus_short": round(contrast, 5) if np.isfinite(contrast) else np.nan,
            "ci_lo":                   round(ci_lo, 5) if np.isfinite(ci_lo) else np.nan,
            "ci_hi":                   round(ci_hi, 5) if np.isfinite(ci_hi) else np.nan,
            "n_long":                  n_l,
            "n_short":                 n_s,
            "short_lengths":           str(sorted(SHORT_LENGTHS)),
            "long_lengths":            str(sorted(LONG_LENGTHS)),
            "note":                    "positive = long worse than short",
        })
    return rows


def compute_ar_minus_tf(sub_tf: pd.DataFrame, sub_ar: pd.DataFrame,
                         route: str,
                         n_boot: int, rng: np.random.RandomState,
                         group_mode: str, group_name: str) -> List[dict]:
    """Compute delta (AR − TF) slope and long-short contrast per route / metric."""
    length_tf, error_tf, edit_tf = _metric_arrays(sub_tf, route)
    length_ar, error_ar, edit_ar = _metric_arrays(sub_ar, route)
    assert np.array_equal(length_tf, length_ar), "TF/AR length arrays must match"
    length = length_tf

    delta_error = error_ar - error_tf
    delta_edit  = edit_ar  - edit_tf
    n = len(length)

    rows = []
    for metric_name, delta in [("error_rate", delta_error), ("edit_dist", delta_edit)]:
        # Delta slope (delta ~ length)
        if n >= MIN_N_SLOPE:
            d_slope = _ols_slope(length, delta)
            d_slope_lo, d_slope_hi = _bootstrap_slope_ci(length, delta, n_boot, rng)
        else:
            d_slope = d_slope_lo = d_slope_hi = np.nan

        # Delta long-short contrast
        mask_short = np.isin(length, list(SHORT_LENGTHS))
        mask_long  = np.isin(length, list(LONG_LENGTHS))
        d_s = delta[mask_short]
        d_l = delta[mask_long]
        n_s, n_l = int(mask_short.sum()), int(mask_long.sum())
        if n_s >= MIN_N_CONTRAST and n_l >= MIN_N_CONTRAST:
            d_contrast = float(d_l.mean() - d_s.mean())
            d_c_lo, d_c_hi = _bootstrap_contrast_ci(d_l, d_s, n_boot, rng)
        else:
            d_contrast = d_c_lo = d_c_hi = np.nan

        def _r(v):
            return round(float(v), 6) if (v is not None and np.isfinite(float(v))) else np.nan

        rows.append({
            "group_mode":              group_mode,
            "group_name":              group_name,
            "route":                   route,
            "metric":                  metric_name,
            "delta_slope":             _r(d_slope),
            "delta_slope_ci_lo":       _r(d_slope_lo),
            "delta_slope_ci_hi":       _r(d_slope_hi),
            "delta_long_short":        _r(d_contrast),
            "delta_long_short_ci_lo":  _r(d_c_lo),
            "delta_long_short_ci_hi":  _r(d_c_hi),
            "n_items":                 n,
            "n_long":                  n_l,
            "n_short":                 n_s,
            "n_boot":                  n_boot,
            "note":                    ("positive = AR worsens length effect "
                                        "relative to TF"),
        })
    return rows


# ---------------------------------------------------------------------------
# Run full analysis
# ---------------------------------------------------------------------------

def run_analysis(df_tf: pd.DataFrame, df_ar: pd.DataFrame,
                 routes: Tuple[str, ...], n_boot: int, seed: int,
                 ) -> dict:
    rng = np.random.RandomState(seed)

    rows_per_length: List[dict] = []
    rows_slopes:     List[dict] = []
    rows_contrasts:  List[dict] = []
    rows_delta:      List[dict] = []

    for group_mode in ALL_GROUP_MODES:
        groups = build_groups(df_tf, df_ar, group_mode)
        for group_name, pair in groups.items():
            sub_tf = pair["tf"]
            sub_ar = pair["ar"]
            print(f"  {group_mode} / {group_name}: n={len(sub_tf)}")

            for route in routes:
                # Check columns exist
                missing_tf = [c for c in (f"{route}_exact_match", f"{route}_edit_dist")
                              if c not in sub_tf.columns]
                missing_ar = [c for c in (f"{route}_exact_match", f"{route}_edit_dist")
                              if c not in sub_ar.columns]
                if missing_tf or missing_ar:
                    print(f"    [skip] {route}: missing columns "
                          f"TF={missing_tf} AR={missing_ar}")
                    continue

                for decode, sub in [("teacher_forced", sub_tf),
                                     ("autoregressive", sub_ar)]:
                    rows_per_length += compute_per_length_summary(
                        sub, route, n_boot, rng, group_mode, group_name, decode)
                    rows_slopes += compute_slopes(
                        sub, route, n_boot, rng, group_mode, group_name, decode)
                    rows_contrasts += compute_long_short(
                        sub, route, n_boot, rng, group_mode, group_name, decode)

                # Delta (AR − TF)
                rows_delta += compute_ar_minus_tf(
                    sub_tf, sub_ar, route, n_boot, rng, group_mode, group_name)

    return {
        "per_length":  pd.DataFrame(rows_per_length),
        "slopes":      pd.DataFrame(rows_slopes),
        "contrasts":   pd.DataFrame(rows_contrasts),
        "delta":       pd.DataFrame(rows_delta),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

_DECODE_COLORS = {"teacher_forced": "#2166ac", "autoregressive": "#d6604d"}
_DECODE_LABELS = {"teacher_forced": "Teacher-forced", "autoregressive": "Autoregressive"}
_ROUTE_LABELS  = {"full": "Full (gated)", "wm": "WM (dorsal)", "ltm": "LTM (ventral)"}

# Human-readable labels for figures — internal group_mode strings must not appear on plots.
_ANALYSIS_LABELS = {
    GROUP_DAGER:       "Train-seen real words vs pseudowords",
    GROUP_SEEN_UNSEEN: "Train-seen real words vs unseen forms",
}
_GROUP_DISPLAY = {
    (GROUP_DAGER,       "Train-seen real"): "Train-seen real words",
    (GROUP_DAGER,       "Pseudowords"):     "Pseudowords",
    (GROUP_SEEN_UNSEEN, "Train-seen real"): "Train-seen real words",
    (GROUP_SEEN_UNSEEN, "Unseen forms"):    "Unseen forms",
}
_METRIC_DISPLAY = {"edit_dist": "edit distance", "error_rate": "error rate"}


def _slope_figure(df_slopes: pd.DataFrame, metric: str, ylabel: str,
                  out_path: str, routes: Tuple[str, ...]) -> None:
    """Grouped bar chart: slopes for TF and AR, 4 panels (2 modes × 2 groups)."""
    sub = df_slopes[df_slopes["metric"] == metric].copy()
    if sub.empty:
        print(f"  [skip fig] no slope data for metric={metric}")
        return

    panels = [(gm, gn) for gm in ALL_GROUP_MODES
              for gn in sub[sub["group_mode"] == gm]["group_name"].unique()]
    n_panels = len(panels)
    ncols = min(2, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows),
                              sharey=False, squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    x = np.arange(len(routes))
    bar_w = 0.35

    for panel_idx, (gm, gn) in enumerate(panels):
        ax = axes_flat[panel_idx]
        sub_panel = sub[(sub["group_mode"] == gm) & (sub["group_name"] == gn)]

        for di, decode in enumerate(_DECODES):
            sub_d = sub_panel[sub_panel["decode"] == decode].set_index("route")
            vals = np.array([sub_d.loc[r, "slope"] if r in sub_d.index else np.nan
                             for r in routes])
            lo   = np.array([sub_d.loc[r, "ci_lo"] if r in sub_d.index else np.nan
                             for r in routes])
            hi   = np.array([sub_d.loc[r, "ci_hi"] if r in sub_d.index else np.nan
                             for r in routes])
            err_lo = vals - lo
            err_hi = hi  - vals
            # replace nan error bars with 0
            err_lo = np.where(np.isfinite(err_lo), err_lo, 0.0)
            err_hi = np.where(np.isfinite(err_hi), err_hi, 0.0)

            offset = (di - 0.5) * bar_w
            ax.bar(x + offset, vals, bar_w * 0.92,
                   color=_DECODE_COLORS[decode],
                   label=_DECODE_LABELS[decode], alpha=0.85,
                   yerr=[err_lo, err_hi], capsize=4, error_kw={"lw": 1.2})

        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([_ROUTE_LABELS.get(r, r) for r in routes], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        analysis_label = _ANALYSIS_LABELS.get(gm, gm)
        group_label    = _GROUP_DISPLAY.get((gm, gn), gn)
        ax.set_title(f"{analysis_label} — {group_label}", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.text(0.02, 0.97, "positive = longer words worse",
                transform=ax.transAxes, fontsize=7, va="top", color="#444")

    # Hide unused panels
    for i in range(n_panels, len(axes_flat)):
        axes_flat[i].set_visible(False)

    metric_display = _METRIC_DISPLAY.get(metric, metric.replace("_", " "))
    fig.suptitle(f"WFE length-effect slopes — {metric_display}  "
                 f"(item-level OLS, 95 % bootstrap CI)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [fig] {out_path}")


def _delta_slope_figure(df_delta: pd.DataFrame, out_path: str,
                         routes: Tuple[str, ...]) -> None:
    """AR − TF delta slopes for both metrics, 4 panels."""
    panels = [(gm, gn) for gm in ALL_GROUP_MODES
              for gn in df_delta[df_delta["group_mode"] == gm]["group_name"].unique()]
    n_panels = len(panels)
    ncols = min(2, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows),
                              sharey=False, squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    x = np.arange(len(routes))
    bar_w = 0.35
    metric_colors = {"error_rate": "#8856a7", "edit_dist": "#31a354"}
    metric_labels = {"error_rate": "Error-rate delta", "edit_dist": "Edit-dist delta"}

    for panel_idx, (gm, gn) in enumerate(panels):
        ax = axes_flat[panel_idx]
        sub_panel = df_delta[(df_delta["group_mode"] == gm) &
                              (df_delta["group_name"] == gn)]

        for mi, metric in enumerate(("error_rate", "edit_dist")):
            sub_m = sub_panel[sub_panel["metric"] == metric].set_index("route")
            vals  = np.array([sub_m.loc[r, "delta_slope"] if r in sub_m.index else np.nan
                              for r in routes])
            lo    = np.array([sub_m.loc[r, "delta_slope_ci_lo"] if r in sub_m.index else np.nan
                              for r in routes])
            hi    = np.array([sub_m.loc[r, "delta_slope_ci_hi"] if r in sub_m.index else np.nan
                              for r in routes])
            err_lo = np.where(np.isfinite(vals - lo), vals - lo, 0.0)
            err_hi = np.where(np.isfinite(hi - vals), hi - vals, 0.0)
            offset = (mi - 0.5) * bar_w
            ax.bar(x + offset, vals, bar_w * 0.92,
                   color=metric_colors[metric], label=metric_labels[metric],
                   alpha=0.85, yerr=[err_lo, err_hi], capsize=4,
                   error_kw={"lw": 1.2})

        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([_ROUTE_LABELS.get(r, r) for r in routes], fontsize=9)
        ax.set_ylabel("AR − TF delta slope", fontsize=9)
        analysis_label = _ANALYSIS_LABELS.get(gm, gm)
        group_label    = _GROUP_DISPLAY.get((gm, gn), gn)
        ax.set_title(f"{analysis_label} — {group_label}", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.text(0.02, 0.97, "positive = AR worsens length effect",
                transform=ax.transAxes, fontsize=7, va="top", color="#444")

    for i in range(n_panels, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle("AR − TF delta slope  (positive = autoregressive worsens length effect, "
                 "95 % bootstrap CI)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [fig] {out_path}")


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

_README = """\
# Length-effect analysis: WFE teacher-forced vs autoregressive

## What was analysed

Item-level WFE predictions were used to quantify how strongly each route's output
quality degrades with word length, and whether autoregressive decoding worsens that
degradation relative to teacher-forced decoding.

Two grouping modes are reported side by side:

| Group mode | Group A | Group B |
|---|---|---|
| dager_strict | Train-seen real | Pseudowords |
| seen_vs_unseen | Train-seen real | Unseen forms (held-out real + novel real + pseudowords) |

"Unseen forms" is NOT a lexicality group — it is a familiarity group.
Do not interpret it as equivalent to "pseudowords".

## Metrics

For each (group, route, decode) triple, two item-level metrics are computed:
- error_rate = 1 − exact_match
- edit_dist  = Levenshtein edit distance between prediction and target

## Why slope AND long-short contrast?

The **slope** (item-level OLS: metric ~ length_phonemes) captures the average
per-phoneme increase in error over the full length range.  It can be diluted by
noisy length bins or by ceiling/floor effects at the extremes.

The **long-short contrast** (mean[long] − mean[short]) captures a more robust
comparison at two well-populated length ranges:
  short = length ∈ {3, 4, 5} phonemes
  long  = length ∈ {7, 8, 9} phonemes

Both are reported; interpret them jointly.

## Interpretation

- **Positive slope**: longer words have higher error / edit distance.
- **Positive AR − TF delta**: autoregressive decoding worsens the length effect
  compared to teacher-forced.  This is expected for the WM route (error propagation)
  but may also appear in the LTM and full routes.
- **Near-zero slope for train-seen real**: ceiling effect — most train-seen words
  are reproduced correctly regardless of length in teacher-forced mode.

## Notes

- WM interference noise is DISABLED in all results here (collect=False, no --wm_noise).
- GloVe is NOT used as input at inference time.
- Bootstrap CIs with n_boot=1000 items over item-level values.
- CI may be exactly zero for groups with perfect accuracy (no variance).

## Files

| File | Contents |
|---|---|
| per_length_summary.tsv | Mean ± 95 % CI per (group, route, decode, length) |
| length_effect_slopes.tsv | OLS slope ± CI per (group, route, decode, metric) |
| long_short_contrasts.tsv | Long − short ± CI per (group, route, decode, metric) |
| ar_minus_tf_length_effects.tsv | AR − TF delta slope and contrast per (group, route, metric) |
| length_effect_slopes_edit_dist.png | Edit-distance slopes, TF vs AR |
| length_effect_slopes_error_rate.png | Error-rate slopes, TF vs AR |
| ar_minus_tf_delta_slopes.png | AR − TF delta slopes |
"""


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quantify length effects in TF vs AR WFE predictions.")
    p.add_argument("--tf_pred", default=TF_PRED_DEFAULT)
    p.add_argument("--ar_pred", default=AR_PRED_DEFAULT)
    p.add_argument("--out_dir", default=OUT_DEFAULT)
    p.add_argument("--n_boot", type=int, default=1000,
                   help="Bootstrap resamples (0 = point estimates only)")
    p.add_argument("--seed",   type=int, default=0,
                   help="RNG seed for reproducible bootstrap")
    p.add_argument("--routes", nargs="+", default=list(_ROUTES),
                   help="Routes to analyse (default: full wm ltm)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n[analyze_length_effects]")
    print(f"  TF pred  : {args.tf_pred}")
    print(f"  AR pred  : {args.ar_pred}")
    print(f"  Output   : {args.out_dir}")
    print(f"  n_boot   : {args.n_boot}  seed={args.seed}")
    print(f"  routes   : {args.routes}")

    for path, label in [(args.tf_pred, "TF"), (args.ar_pred, "AR")]:
        if not os.path.exists(path):
            print(f"\nERROR: {label} predictions TSV not found: {path}")
            sys.exit(1)

    df_tf, df_ar = _load_and_align(args.tf_pred, args.ar_pred)
    routes = tuple(args.routes)

    print("\n  Running analysis …")
    results = run_analysis(df_tf, df_ar, routes, args.n_boot, args.seed)

    # Write TSVs
    tsv_paths = {}
    for key, fname in [
        ("per_length", "per_length_summary.tsv"),
        ("slopes",     "length_effect_slopes.tsv"),
        ("contrasts",  "long_short_contrasts.tsv"),
        ("delta",      "ar_minus_tf_length_effects.tsv"),
    ]:
        path = os.path.join(args.out_dir, fname)
        results[key].to_csv(path, sep="\t", index=False)
        tsv_paths[key] = path
        print(f"  [tsv] {path}  ({len(results[key])} rows)")

    # Figures
    _slope_figure(results["slopes"], "edit_dist", "Length-effect slope (edit distance)",
                  os.path.join(args.out_dir, "length_effect_slopes_edit_dist.png"), routes)
    _slope_figure(results["slopes"], "error_rate", "Length-effect slope (error rate)",
                  os.path.join(args.out_dir, "length_effect_slopes_error_rate.png"), routes)
    _delta_slope_figure(results["delta"],
                        os.path.join(args.out_dir, "ar_minus_tf_delta_slopes.png"), routes)

    # README
    readme_path = os.path.join(args.out_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(_README)
    print(f"  [readme] {readme_path}")

    # Compact previews
    print("\n  === length_effect_slopes.tsv (compact) ===")
    df_s = results["slopes"]
    preview_s = df_s[df_s["group_mode"] == GROUP_DAGER][
        ["group_name", "decode", "route", "metric", "slope", "ci_lo", "ci_hi", "n_items"]
    ].sort_values(["group_name", "route", "decode", "metric"])
    print(preview_s.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\n  === long_short_contrasts.tsv (compact) ===")
    df_c = results["contrasts"]
    preview_c = df_c[df_c["group_mode"] == GROUP_DAGER][
        ["group_name", "decode", "route", "metric",
         "contrast_long_minus_short", "ci_lo", "ci_hi", "n_long", "n_short"]
    ].sort_values(["group_name", "route", "decode", "metric"])
    print(preview_c.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print(f"\n[analyze_length_effects] Done.")
    all_files = sorted(
        f for f in os.listdir(args.out_dir)
        if f.endswith((".tsv", ".png", ".md"))
    )
    for f in all_files:
        kb = os.path.getsize(os.path.join(args.out_dir, f)) // 1024
        print(f"    {f}  ({kb} KB)")


if __name__ == "__main__":
    main()
