"""Dager-style WFE figure pack.

Reads the per-item predictions TSV produced by scripts/external_eval.py and
generates three figures and one table:

  Figure 1  wfe_condition_accuracy.png
      Grouped barplot: x = WFE condition code (12 conditions), 3 bars per condition
      (full / wm / ltm exact-match).  Ordered real (RLCH…RSSL) then pseudo (PLC…PSS).

  Figure 2  wfe_factor_main_effects.png
      2×2 panel of factor main effects:
        (a) Lexicality (real vs pseudo)           [all items]
        (b) Length / size (long vs short)         [all items]
        (c) Morphology (complex vs simple)        [all items]
        (d) Frequency (high vs low)               [real items only]

  Figure 3  wfe_logistic_regression.png
      Logistic-regression coefficient plot for full-route exact-match accuracy.
      Model A (all items):        length_phonemes, is_real, is_short, is_simple
      Model B (real items only):  length_phonemes, is_short, is_simple, is_highfreq

  Table     wfe_condition_accuracy.tsv

NOTE: all evaluation is teacher-forced. Exact-match does NOT reflect free-generation
accuracy.

Usage:
    python scripts/plot_wfe_dager_style.py
    python scripts/plot_wfe_dager_style.py \\
        --pred    outputs/external_eval_30k/wfe/item_level_predictions.tsv \\
        --out_dir outputs/external_eval_30k/figures
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PRED_DEFAULT = os.path.join(ROOT, "outputs", "external_eval_30k", "wfe",
                             "item_level_predictions.tsv")
OUT_DEFAULT  = os.path.join(ROOT, "outputs", "external_eval_30k", "figures")

TEACHER_FORCED_NOTE = "Teacher-forced decoding (gold prefix at each step)"

_CONDITION_ORDER = [
    "RLCH", "RLCL", "RLSH", "RLSL",
    "RSCH", "RSCL", "RSSH", "RSSL",
    "PLC",  "PLS",  "PSC",  "PSS",
]

_CONDITION_MAP = {
    "RLCH": ("real",   "long",  "complex", "high"),
    "RLCL": ("real",   "long",  "complex", "low"),
    "RLSH": ("real",   "long",  "simple",  "high"),
    "RLSL": ("real",   "long",  "simple",  "low"),
    "RSCH": ("real",   "short", "complex", "high"),
    "RSCL": ("real",   "short", "complex", "low"),
    "RSSH": ("real",   "short", "simple",  "high"),
    "RSSL": ("real",   "short", "simple",  "low"),
    "PLC":  ("pseudo", "long",  "complex", "N/A"),
    "PLS":  ("pseudo", "long",  "simple",  "N/A"),
    "PSC":  ("pseudo", "short", "complex", "N/A"),
    "PSS":  ("pseudo", "short", "simple",  "N/A"),
}

_ROUTE_COLORS = {"full": "#2b7bba", "wm": "#e05a2b", "ltm": "#2ba34b"}
_ROUTE_LABELS = {"full": "Full (gated)", "wm": "WM (dorsal)", "ltm": "LTM (ventral)"}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _derive_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Add lexicality/size/morphology/freq_group columns derived from condition code."""
    if "condition" not in df.columns:
        return df
    df = df.copy()
    for col, idx in [("lex_derived", 0), ("size_derived", 1),
                     ("morph_derived", 2), ("freq_derived", 3)]:
        df[col] = df["condition"].map(
            lambda c, i=idx: _CONDITION_MAP.get(str(c), ("?","?","?","?"))[i]
        )
    if "lexicality" not in df.columns:
        df["lexicality"] = df["lex_derived"]
    if "size" not in df.columns:
        df["size"] = df["size_derived"]
    if "morphology" not in df.columns:
        df["morphology"] = df["morph_derived"]
    if "freq_group" not in df.columns:
        df["freq_group"] = df["freq_derived"]
    if "length_phonemes" not in df.columns and "target_phonemes" in df.columns:
        df["length_phonemes"] = df["target_phonemes"].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else np.nan)
    return df


def _condition_table(df: pd.DataFrame) -> pd.DataFrame:
    routes = ["full", "wm", "ltm"]
    rows = []
    for cond in _CONDITION_ORDER:
        sub = df[df["condition"] == cond]
        if len(sub) == 0:
            continue
        lex, size, morph, freq = _CONDITION_MAP.get(cond, ("?","?","?","?"))
        row = {"condition": cond, "lexicality": lex, "size": size,
               "morphology": morph, "freq_group": freq, "n": len(sub)}
        for r in routes:
            col = f"{r}_exact_match"
            row[f"{r}_exact_match"] = round(float(sub[col].mean()), 4) if col in sub else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure 1: condition accuracy grouped barplot
# ---------------------------------------------------------------------------

def fig_condition_accuracy(df: pd.DataFrame, out_dir: str) -> None:
    routes = ["full", "wm", "ltm"]
    conds_present = [c for c in _CONDITION_ORDER if (df["condition"] == c).any()]
    n = len(conds_present)
    if n == 0:
        print("[plot_wfe] No conditions found — skipping condition accuracy plot.")
        return

    vals = {r: [] for r in routes}
    for cond in conds_present:
        sub = df[df["condition"] == cond]
        for r in routes:
            col = f"{r}_exact_match"
            vals[r].append(float(sub[col].mean()) if col in sub else 0.0)

    x = np.arange(n)
    width = 0.22
    offsets = {"full": -width, "wm": 0, "ltm": width}

    fig, ax = plt.subplots(figsize=(max(10, n * 0.85), 5))
    for r in routes:
        ax.bar(x + offsets[r], vals[r], width,
               label=_ROUTE_LABELS[r], color=_ROUTE_COLORS[r], alpha=0.85)

    # Separator between real and pseudo
    real_conds = [c for c in conds_present if _CONDITION_MAP.get(c, ("?",))[0] == "real"]
    if real_conds and len(real_conds) < n:
        ax.axvline(len(real_conds) - 0.5, color="gray", lw=1, ls="--", alpha=0.5)
        ax.text(len(real_conds) - 0.5, 1.01, "  pseudo →", fontsize=7,
                color="gray", va="bottom", ha="left", transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(conds_present, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Exact-match accuracy (teacher-forced)")
    ax.set_title(f"WFE accuracy by condition — {TEACHER_FORCED_NOTE}")
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # Annotation: n per condition
    for i, cond in enumerate(conds_present):
        sub = df[df["condition"] == cond]
        ax.text(x[i], -0.07, f"n={len(sub)}", ha="center", va="top",
                fontsize=6, transform=ax.get_xaxis_transform())

    fig.tight_layout()
    path = os.path.join(out_dir, "wfe_condition_accuracy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [fig] {path}")


# ---------------------------------------------------------------------------
# Figure 2: factor main effects
# ---------------------------------------------------------------------------

def _factor_bars(ax, df: pd.DataFrame, col: str, levels: List[str],
                 routes: List[str], level_labels: Optional[Dict[str,str]] = None) -> None:
    """Draw grouped bar chart for one factor on a given axes."""
    x = np.arange(len(levels))
    width = 0.25
    offsets = np.linspace(-width, width, len(routes))
    for i, r in enumerate(routes):
        col_name = f"{r}_exact_match"
        vals = []
        for lvl in levels:
            sub = df[df[col] == lvl]
            vals.append(float(sub[col_name].mean()) if (len(sub) > 0 and col_name in df.columns) else np.nan)
        ax.bar(x + offsets[i], vals, width,
               label=_ROUTE_LABELS[r], color=_ROUTE_COLORS[r], alpha=0.85)

    xlabels = [level_labels[l] if level_labels and l in level_labels else l for l in levels]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(alpha=0.3, axis="y")


def fig_factor_main_effects(df: pd.DataFrame, out_dir: str) -> None:
    routes = ["full", "wm", "ltm"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    # (a) Lexicality
    ax = axes[0]
    levels = [l for l in ["real", "pseudo"] if (df["lexicality"] == l).any()]
    _factor_bars(ax, df, "lexicality", levels, routes)
    ax.set_title("(a) Lexicality")
    ax.set_ylabel("Exact-match accuracy")
    ax.legend(fontsize=7)

    # (b) Size (long vs short)
    ax = axes[1]
    levels = [l for l in ["long", "short"] if (df["size"] == l).any()]
    _factor_bars(ax, df, "size", levels, routes)
    ax.set_title("(b) Size (word length category)")

    # (c) Morphology (complex vs simple)
    ax = axes[2]
    levels = [l for l in ["complex", "simple"] if (df["morphology"] == l).any()]
    _factor_bars(ax, df, "morphology", levels, routes)
    ax.set_title("(c) Morphology")
    ax.set_ylabel("Exact-match accuracy")

    # (d) Frequency — real words only
    ax = axes[3]
    real_df = df[df["lexicality"] == "real"].copy()
    levels = [l for l in ["high", "low"] if (real_df["freq_group"] == l).any()]
    if levels:
        _factor_bars(ax, real_df, "freq_group", levels, routes,
                     level_labels={"high": "high freq", "low": "low freq"})
        ax.set_title("(d) Frequency (real words only)")
    else:
        ax.text(0.5, 0.5, "No frequency column found", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_title("(d) Frequency — N/A")

    fig.suptitle(f"WFE factor main effects — {TEACHER_FORCED_NOTE}", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "wfe_factor_main_effects.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [fig] {path}")


# ---------------------------------------------------------------------------
# Figure 3: logistic regression coefficients
# ---------------------------------------------------------------------------

def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Point-biserial Pearson r between feature x and binary y (numpy-only)."""
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 5:
        return np.nan
    xv, yv = x[valid], y[valid]
    sx, sy = xv.std(), yv.std()
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return float(np.mean((xv - xv.mean()) * (yv - yv.mean())) / (sx * sy))


def _compute_coefs_sklearn(df: pd.DataFrame, features: list, labels: list,
                            y_col: str):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    data = df[features + [y_col]].dropna()
    if len(data) < 20:
        return {}, 0
    Xsc  = StandardScaler().fit_transform(data[features])
    lr   = LogisticRegression(random_state=0, max_iter=500, solver="lbfgs")
    lr.fit(Xsc, data[y_col].astype(int))
    return dict(zip(labels, lr.coef_[0])), len(data)


def _compute_coefs_numpy(df: pd.DataFrame, features: list, labels: list,
                          y_col: str):
    """Pearson r as a sklearn fallback (not logistic coefs, but same sign convention)."""
    data = df[features + [y_col]].dropna()
    if len(data) < 5:
        return {}, 0
    y   = data[y_col].values.astype(float)
    out = {}
    for feat, label in zip(features, labels):
        x = data[feat].values.astype(float)
        out[label] = _pearson_r(x, y)
    return out, len(data)


def fig_logistic_regression(df: pd.DataFrame, out_dir: str) -> None:
    df = df.copy()

    # Derived binary features
    df["is_real"]     = (df["lexicality"] == "real").astype(float)
    df["is_short"]    = (df["size"] == "short").astype(float)
    df["is_simple"]   = (df["morphology"] == "simple").astype(float)
    df["is_highfreq"] = (df["freq_group"] == "high").astype(float)

    if "length_phonemes" not in df.columns:
        print("[plot_wfe] length_phonemes missing — skipping regression.")
        return

    df["length_phonemes"] = pd.to_numeric(df["length_phonemes"], errors="coerce")

    full_col = "full_exact_match"
    if full_col not in df.columns:
        print(f"[plot_wfe] Column {full_col} missing — skipping regression.")
        return

    features_A = ["length_phonemes", "is_real", "is_short", "is_simple"]
    labels_A   = ["Length\n(phonemes)", "Lexicality\n(real)", "Size\n(short)", "Morphology\n(simple)"]
    features_B = ["length_phonemes", "is_short", "is_simple", "is_highfreq"]
    labels_B   = ["Length\n(phonemes)", "Size\n(short)", "Morphology\n(simple)", "Frequency\n(high)"]
    dfB = df[df["lexicality"] == "real"].copy()

    try:
        coefs_A, nA = _compute_coefs_sklearn(df,  features_A, labels_A, full_col)
        coefs_B, nB = _compute_coefs_sklearn(dfB, features_B, labels_B, full_col)
        method_label = "Standardised logistic coefficient"
        method_note  = "sklearn LogisticRegression"
    except ImportError:
        print("[plot_wfe] sklearn not found — using Pearson r as fallback")
        coefs_A, nA = _compute_coefs_numpy(df,  features_A, labels_A, full_col)
        coefs_B, nB = _compute_coefs_numpy(dfB, features_B, labels_B, full_col)
        method_label = "Pearson r (sklearn unavailable — not logistic coef)"
        method_note  = "numpy Pearson r fallback"

    if not coefs_A and not coefs_B:
        print("[plot_wfe] Insufficient data for regression figure.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    def _coef_plot(ax, coefs: dict, title: str, note: str) -> None:
        if not coefs:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(title)
            return
        names = list(coefs.keys())
        vals  = [coefs[n] if not np.isnan(coefs[n]) else 0.0 for n in names]
        colors = ["#2b7bba" if v >= 0 else "#e05a2b" for v in vals]
        y = np.arange(len(names))
        ax.barh(y, vals, color=colors, alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel(f"{method_label}\n(positive = higher accuracy)")
        ax.set_title(title)
        ax.text(0.98, 0.02, note, ha="right", va="bottom", transform=ax.transAxes,
                fontsize=7, color="gray")
        ax.grid(alpha=0.3, axis="x")

    _coef_plot(axes[0], coefs_A, "Model A — all items", f"n={nA}\n{method_note}")
    _coef_plot(axes[1], coefs_B, "Model B — real words only", f"n={nB}\n{method_note}")

    fig.suptitle(
        f"Feature importance: predictors of full-route exact-match accuracy\n"
        f"({TEACHER_FORCED_NOTE})",
        fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "wfe_logistic_regression.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [fig] {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred",    default=PRED_DEFAULT,
                   help="item_level_predictions.tsv from external_eval.py")
    p.add_argument("--out_dir", default=OUT_DEFAULT,
                   help="output directory for figures and table")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.pred):
        print(f"\nERROR: predictions file not found: {args.pred}")
        print("Run first:")
        print("  python scripts/external_eval.py \\")
        print("      --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt \\")
        print("      --out_dir outputs/external_eval_30k")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n[plot_wfe_dager] Loading: {args.pred}")
    df = pd.read_csv(args.pred, sep="\t")
    print(f"  {len(df)} items, columns: {list(df.columns)}")

    df = _derive_factors(df)

    # Save condition table
    cond_table = _condition_table(df)
    tbl_path = os.path.join(args.out_dir, "wfe_condition_accuracy.tsv")
    cond_table.to_csv(tbl_path, sep="\t", index=False)
    print(f"  -> {tbl_path}")

    # Figures
    fig_condition_accuracy(df, args.out_dir)
    fig_factor_main_effects(df, args.out_dir)
    fig_logistic_regression(df, args.out_dir)

    print("\n[plot_wfe_dager] Done.")
    print(f"  Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
