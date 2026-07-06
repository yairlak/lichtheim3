"""Dager-comparable and dataset-centered WFE figures for meeting pack.

Produces three figures directly in outputs/meeting_pack_30k_wfe/, then
copies the four other numbered figures from their source directory
(outputs/external_eval_30k/figures/).

KEY DISTINCTION:
  'strict train-seen real'  = lexicon_category == real_word_seen_in_training_lexicon
                               (Dager-comparable: training set exposure controlled)
  'all WFE real'            = lexicality == real
                               (includes val-held-out + novel real words)
  'pseudoword'              = lexicality == pseudo

Outputs (all to --out_dir, default outputs/meeting_pack_30k_wfe/):
  01_model_centered_categories.png
  02_dager_fig2_like_wfe_strict_train_real.png
  02b_dager_fig2_like_wfe_dataset_lexicality.png
  03_wfe_condition_accuracy.png              (copied from external_eval_30k/figures/)
  04_route_accuracy_by_category.png         (copied)
  05_route_dissociation_wfe.png             (copied)
  06_error_type_breakdown.png               (copied)

Usage:
    python scripts/plot_wfe_dager_strict_real.py
    python scripts/plot_wfe_dager_strict_real.py \\
        --pred outputs/external_eval_30k/wfe/item_level_predictions.tsv \\
        --out_dir outputs/meeting_pack_30k_wfe
    python scripts/plot_wfe_dager_strict_real.py --show_routes
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRED_DEFAULT = os.path.join(ROOT, "outputs", "external_eval_30k",
                             "wfe", "item_level_predictions.tsv")
OUT_DEFAULT  = os.path.join(ROOT, "outputs", "meeting_pack_30k_wfe")
SRC_FIGS     = os.path.join(ROOT, "outputs", "external_eval_30k", "figures")

# Source filename → numbered meeting-pack name (figures 03–06)
COPY_MAP = {
    "wfe_condition_accuracy.png":     "03_wfe_condition_accuracy.png",
    "route_accuracy_by_category.png": "04_route_accuracy_by_category.png",
    "route_dissociation_wfe.png":     "05_route_dissociation_wfe.png",
    "error_type_breakdown.png":       "06_error_type_breakdown.png",
}

# Canonical order for lexicon_category
_CAT_ORDER = [
    "real_word_seen_in_training_lexicon",
    "real_word_in_validation_split",
    "real_word_outside_4000_lexicon",
    "pseudoword",
]
_CAT_LABELS = {
    "real_word_seen_in_training_lexicon": "Train-seen real",
    "real_word_in_validation_split":      "Val-held-out real",
    "real_word_outside_4000_lexicon":     "Novel real",
    "pseudoword":                         "Pseudoword",
}
_CAT_COLORS = {
    "real_word_seen_in_training_lexicon": "#c0392b",
    "real_word_in_validation_split":      "#e67e22",
    "real_word_outside_4000_lexicon":     "#8e44ad",
    "pseudoword":                         "#2c3e7f",
}

_ROUTE_COLORS = {"full": "#2b7bba", "wm": "#e05a2b", "ltm": "#2ba34b"}
_ROUTE_LABELS = {"full": "Full (gated)", "wm": "WM (dorsal)", "ltm": "LTM (ventral)"}
_ROUTE_ORDER  = ["full", "wm", "ltm"]

_STRICT_REAL_COLOR = "#c0392b"
_ALL_REAL_COLOR    = "#c0392b"
_PSEUDO_COLOR      = "#2c3e7f"

NOTE = "Teacher-forced decoding · WM noise disabled (deterministic)"
NOTE_REGIME = "Regime A (teacher-forced, deterministic) — see docs/evaluation_regimes.md"

_X_LABEL = "Word length (phonemes)"


# ---------------------------------------------------------------------------
# Bootstrap CI helper
# ---------------------------------------------------------------------------

def _bootstrap_ci(values: np.ndarray, n_boot: int,
                  rng: np.random.RandomState) -> tuple:
    """95 % bootstrap CI for the mean of values. Returns (lo, hi)."""
    if len(values) < 2 or n_boot == 0:
        m = float(values.mean()) if len(values) > 0 else np.nan
        return m, m
    boot = np.array([rng.choice(values, size=len(values), replace=True).mean()
                     for _ in range(n_boot)])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _length_curve(sub: pd.DataFrame, col: str,
                  n_boot: int = 1000, min_n: int = 3,
                  rng: Optional[np.random.RandomState] = None) -> pd.DataFrame:
    """Mean + 95 % bootstrap CI per exact phoneme length for column `col`."""
    if rng is None:
        rng = np.random.RandomState(42)
    rows = []
    for length, grp in sub.groupby("length_phonemes"):
        vals = grp[col].dropna().values
        if len(vals) < min_n:
            continue
        mean_val = float(vals.mean())
        ci_lo, ci_hi = _bootstrap_ci(vals, n_boot, rng)
        rows.append({"length_phonemes": int(length), "mean": mean_val,
                     "ci_lo": ci_lo, "ci_hi": ci_hi, "n": len(vals)})
    return pd.DataFrame(rows).sort_values("length_phonemes").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figure 01 — model-centered category accuracy
# ---------------------------------------------------------------------------

def fig_model_centered_categories(df: pd.DataFrame, out_dir: str) -> None:
    routes = [r for r in _ROUTE_ORDER if f"{r}_exact_match" in df.columns]
    cats   = [c for c in _CAT_ORDER if c in df["lexicon_category"].values]

    x      = np.arange(len(cats))
    n_r    = len(routes)
    width  = 0.22
    offsets = np.linspace(-(n_r - 1) * width / 2, (n_r - 1) * width / 2, n_r)

    fig, ax = plt.subplots(figsize=(11, 5))
    for route, offset in zip(routes, offsets):
        accs = []
        for cat in cats:
            sub = df[df["lexicon_category"] == cat]
            accs.append(float(sub[f"{route}_exact_match"].mean()) if len(sub) > 0 else np.nan)
        ax.bar(x + offset, accs, width=width * 0.92,
               color=_ROUTE_COLORS[route], label=_ROUTE_LABELS[route], alpha=0.85)

    # Annotate count under each category
    cat_tick_labels = []
    for cat in cats:
        n = int((df["lexicon_category"] == cat).sum())
        cat_tick_labels.append(f"{_CAT_LABELS.get(cat, cat)}\n(n={n})")

    ax.set_xticks(x)
    ax.set_xticklabels(cat_tick_labels, fontsize=9)
    ax.set_ylabel("Exact-match accuracy", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.set_title(
        "WFE accuracy by model-centered lexicon category\n"
        "lichtheim3 30k — " + NOTE,
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    # Annotate bar tops
    for route, offset in zip(routes, offsets):
        for i, cat in enumerate(cats):
            sub = df[df["lexicon_category"] == cat]
            if len(sub) == 0:
                continue
            acc = float(sub[f"{route}_exact_match"].mean())
            ax.text(x[i] + offset, acc + 0.01, f"{acc:.3f}",
                    ha="center", va="bottom", fontsize=6, rotation=90)

    fig.tight_layout()
    path = os.path.join(out_dir, "01_model_centered_categories.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [fig] {path}")


# ---------------------------------------------------------------------------
# Shared drawing helper for figures 02 / 02b
# ---------------------------------------------------------------------------

def _draw_length_panel(ax: plt.Axes,
                       df_real: pd.DataFrame, df_pseudo: pd.DataFrame,
                       col: str,
                       real_label: str, real_color: str,
                       n_boot: int,
                       show_routes: bool = False) -> None:
    """Two main lines (real / pseudo, full route) + optional thin WM/LTM lines."""
    rng = np.random.RandomState(42)

    def _draw(agg, color, label, lw=2.2, ls="-", alpha=1.0, ms=6, zorder=3):
        if agg is None or len(agg) == 0:
            return
        xs = agg["length_phonemes"].values
        ys = agg["mean"].values
        lo = agg["ci_lo"].values
        hi = agg["ci_hi"].values
        ax.plot(xs, ys, color=color, lw=lw, ls=ls, marker="o", ms=ms,
                label=label, alpha=alpha, zorder=zorder)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15 * alpha)

    real_agg   = _length_curve(df_real,   col, n_boot=n_boot, rng=rng)
    pseudo_agg = _length_curve(df_pseudo, col, n_boot=n_boot, rng=rng)
    _draw(real_agg,   real_color,   real_label)
    _draw(pseudo_agg, _PSEUDO_COLOR, "Pseudowords")

    for agg, color in [(real_agg, real_color), (pseudo_agg, _PSEUDO_COLOR)]:
        for _, row in agg.iterrows():
            ax.annotate(f"n={int(row['n'])}",
                        xy=(row["length_phonemes"], row["mean"]),
                        xytext=(2, 5), textcoords="offset points",
                        fontsize=6, color=color, alpha=0.75)

    if show_routes:
        for route, ls_style, alpha in [("wm", "--", 0.50), ("ltm", ":", 0.50)]:
            route_col = col.replace("full_", f"{route}_")
            if route_col not in df_real.columns:
                continue
            _draw(_length_curve(df_real,   route_col, n_boot=0, rng=rng),
                  real_color,   f"Train-seen real — {route.upper()}",
                  lw=1.0, ls=ls_style, alpha=alpha, ms=3, zorder=2)
            _draw(_length_curve(df_pseudo, route_col, n_boot=0, rng=rng),
                  _PSEUDO_COLOR, f"Pseudowords — {route.upper()}",
                  lw=1.0, ls=ls_style, alpha=alpha, ms=3, zorder=2)


def _save_panel(ax: plt.Axes, fig: plt.Figure, subtitle: str, title: str,
                ylabel: str, fname: str) -> None:
    ax.set_xlabel(_X_LABEL, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=10)
    ax.text(0.5, -0.16, subtitle,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8, color="#444444", style="italic")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  [fig] {fname}")


# ---------------------------------------------------------------------------
# Figure 02 — strict Dager-comparable (train-seen real vs pseudo)
# Produces: 02_strict_real_error_rate.png  +  02_strict_real_edit_dist.png
# ---------------------------------------------------------------------------

def fig_strict_dager(df: pd.DataFrame, out_dir: str,
                     show_routes: bool, n_boot: int) -> None:
    df_real   = df[df["lexicon_category"] == "real_word_seen_in_training_lexicon"].copy()
    df_pseudo = df[df["lexicality"] == "pseudo"].copy()

    for sub in (df_real, df_pseudo):
        for route in ("full", "wm", "ltm"):
            sub[f"{route}_error_rate"] = 1.0 - sub[f"{route}_exact_match"]

    n_real, n_pseudo = len(df_real), len(df_pseudo)
    real_label = f"Train-seen real words (n={n_real})"
    subtitle = (f"real = WFE train-seen;  pseudo = WFE pseudowords  ·  "
                f"n_pseudo={n_pseudo}  ·  {NOTE}")

    for df_grp, lbl in [(df_real, "train-real"), (df_pseudo, "pseudo")]:
        counts = df_grp["length_phonemes"].value_counts().sort_index()
        print(f"    {lbl} length counts: {dict(counts)}")

    fig, ax = plt.subplots(figsize=(9, 5))
    _draw_length_panel(ax, df_real, df_pseudo, "full_error_rate",
                       real_label, _STRICT_REAL_COLOR, n_boot, show_routes)
    ax.set_ylim(-0.02, min(1.05, ax.get_ylim()[1] + 0.05))
    _save_panel(ax, fig, subtitle,
                "WFE: error rate by word length — Dager-comparable\n"
                "lichtheim3 30k  ·  full/gated route  ·  95 % bootstrap CI",
                "Error rate  (1 − exact-match)",
                os.path.join(out_dir, "02_strict_real_error_rate.png"))

    fig, ax = plt.subplots(figsize=(9, 5))
    _draw_length_panel(ax, df_real, df_pseudo, "full_edit_dist",
                       real_label, _STRICT_REAL_COLOR, n_boot, show_routes)
    ax.set_ylim(bottom=-0.02)
    _save_panel(ax, fig, subtitle,
                "WFE: edit distance by word length — Dager-comparable\n"
                "lichtheim3 30k  ·  full/gated route  ·  95 % bootstrap CI",
                "Mean edit distance (full route)",
                os.path.join(out_dir, "02_strict_real_edit_dist.png"))


# ---------------------------------------------------------------------------
# Figure 02b — dataset-centered (all WFE real vs pseudo, NOT Dager-comparable)
# Produces: 02b_all_real_error_rate.png  +  02b_all_real_edit_dist.png
# ---------------------------------------------------------------------------

def fig_dataset_centered(df: pd.DataFrame, out_dir: str,
                          show_routes: bool, n_boot: int) -> None:
    df_real   = df[df["lexicality"] == "real"].copy()
    df_pseudo = df[df["lexicality"] == "pseudo"].copy()

    cat_counts = df_real["lexicon_category"].value_counts()
    breakdown  = ", ".join(
        f"{_CAT_LABELS.get(k, k)}={v}" for k, v in cat_counts.items()
    )
    print(f"    all-real breakdown: {breakdown}")

    for sub in (df_real, df_pseudo):
        for route in ("full", "wm", "ltm"):
            sub[f"{route}_error_rate"] = 1.0 - sub[f"{route}_exact_match"]

    n_real, n_pseudo = len(df_real), len(df_pseudo)
    real_label = f"All WFE real words: train + val + novel (n={n_real})"
    subtitle = (f"all WFE real: train + val + novel;  pseudo n={n_pseudo}  ·  {NOTE}")
    flag = "[NOT Dager-comparable — includes unseen real words]"

    fig, ax = plt.subplots(figsize=(9, 5))
    _draw_length_panel(ax, df_real, df_pseudo, "full_error_rate",
                       real_label, _ALL_REAL_COLOR, n_boot, show_routes)
    ax.set_ylim(-0.02, min(1.05, ax.get_ylim()[1] + 0.05))
    _save_panel(ax, fig, subtitle,
                f"WFE: error rate by word length — dataset lexicality\n"
                f"lichtheim3 30k  ·  full/gated route  ·  {flag}",
                "Error rate  (1 − exact-match)",
                os.path.join(out_dir, "02b_all_real_error_rate.png"))

    fig, ax = plt.subplots(figsize=(9, 5))
    _draw_length_panel(ax, df_real, df_pseudo, "full_edit_dist",
                       real_label, _ALL_REAL_COLOR, n_boot, show_routes)
    ax.set_ylim(bottom=-0.02)
    _save_panel(ax, fig, subtitle,
                f"WFE: edit distance by word length — dataset lexicality\n"
                f"lichtheim3 30k  ·  full/gated route  ·  {flag}",
                "Mean edit distance (full route)",
                os.path.join(out_dir, "02b_all_real_edit_dist.png"))


# ---------------------------------------------------------------------------
# Copy figures 03–06 from their source
# ---------------------------------------------------------------------------

def copy_existing_figures(src_dir: str, out_dir: str) -> None:
    print(f"\n  Copying numbered figures from: {src_dir}")
    for src_name, dst_name in COPY_MAP.items():
        src = os.path.join(src_dir, src_name)
        dst = os.path.join(out_dir, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  [copy] {src_name} → {dst_name}")
        else:
            print(f"  [MISSING] {src_name}  (run the eval scripts first)")
            print(f"            Expected at: {src}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred",    default=PRED_DEFAULT,
                   help="Path to item_level_predictions.tsv from external_eval.py")
    p.add_argument("--out_dir", default=OUT_DEFAULT,
                   help="Output directory for meeting pack figures")
    p.add_argument("--src_figs", default=SRC_FIGS,
                   help="Source directory for figures 03–06 (default: external_eval_30k/figures/)")
    p.add_argument("--show_routes", action="store_true",
                   help="Add thin WM and LTM lines to the length curves")
    p.add_argument("--n_boot", type=int, default=1000,
                   help="Bootstrap resamples for 95 %% CI (0 = disable CI, show point estimate only)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n[plot_wfe_dager_strict_real]")
    print(f"  Input   : {args.pred}")
    print(f"  Output  : {args.out_dir}")
    print(f"  WM/LTM  : {'shown (thin lines)' if args.show_routes else 'hidden'}")
    print(f"  n_boot  : {args.n_boot}")

    if not os.path.exists(args.pred):
        print(f"\nERROR: predictions TSV not found: {args.pred}")
        print("  Run: python scripts/external_eval.py")
        sys.exit(1)

    df = pd.read_csv(args.pred, sep="\t")
    df = df[df["notes"].fillna("").str.strip() == ""]  # drop excluded items
    print(f"  {len(df)} items loaded (after dropping excluded)")

    # Sanity: report category breakdown
    print("\n  === Dataset breakdown ===")
    if "lexicon_category" in df.columns:
        for cat, n in df["lexicon_category"].value_counts().items():
            print(f"    {_CAT_LABELS.get(cat, cat):30s} n={n}")
    else:
        print("  WARNING: lexicon_category column not found")

    # --- Figure 01 ---
    print("\n  [01] Model-centered categories …")
    if "lexicon_category" in df.columns:
        fig_model_centered_categories(df, args.out_dir)
    else:
        print("  SKIPPED — no lexicon_category column in TSV")

    # --- Figure 02 ---
    print("\n  [02] Strict Dager (train-seen real vs pseudo) …")
    if "lexicon_category" in df.columns and "lexicality" in df.columns:
        n_strict = int((df["lexicon_category"] == "real_word_seen_in_training_lexicon").sum())
        n_pseudo = int((df["lexicality"] == "pseudo").sum())
        print(f"    strict real n={n_strict},  pseudo n={n_pseudo}")
        fig_strict_dager(df, args.out_dir, args.show_routes, args.n_boot)
    else:
        print("  SKIPPED — missing lexicon_category or lexicality column")

    # --- Figure 02b ---
    print("\n  [02b] Dataset-centered (all WFE real vs pseudo) …")
    if "lexicality" in df.columns:
        fig_dataset_centered(df, args.out_dir, args.show_routes, args.n_boot)
    else:
        print("  SKIPPED — missing lexicality column")

    # --- Copy 03–06 ---
    copy_existing_figures(args.src_figs, args.out_dir)

    # --- Summary ---
    print(f"\n  === Meeting pack contents ===")
    for f in sorted(os.listdir(args.out_dir)):
        if f.endswith(".png") or f.endswith(".md"):
            size_kb = os.path.getsize(os.path.join(args.out_dir, f)) // 1024
            print(f"    {f}  ({size_kb} KB)")

    print(f"\n[plot_wfe_dager_strict_real] Done.  Pack in: {args.out_dir}")


if __name__ == "__main__":
    main()
