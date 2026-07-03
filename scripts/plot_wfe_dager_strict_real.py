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
# Helpers for figures 02 / 02b
# ---------------------------------------------------------------------------

def _length_curve(sub: pd.DataFrame, edit_col: str,
                  min_n: int = 3) -> pd.DataFrame:
    """Mean ± SEM edit distance per exact phoneme length."""
    agg = (sub.groupby("length_phonemes")
              .agg(
                  mean=(edit_col, "mean"),
                  sem=(edit_col, lambda x: float(x.sem()) if len(x) > 1 else 0.0),
                  n=(edit_col, "count"),
              )
              .reset_index())
    return agg[agg["n"] >= min_n].sort_values("length_phonemes")


def _plot_length_curve(ax: plt.Axes,
                       df_real: pd.DataFrame, df_pseudo: pd.DataFrame,
                       real_label: str, real_color: str,
                       edit_col: str,
                       show_routes: bool,
                       df_real_wm: Optional[pd.DataFrame] = None,
                       df_pseudo_wm: Optional[pd.DataFrame] = None,
                       df_real_ltm: Optional[pd.DataFrame] = None,
                       df_pseudo_ltm: Optional[pd.DataFrame] = None) -> None:
    """Plot two lines (real, pseudo) for the given route; optionally thin WM/LTM."""

    def _draw(agg, color, label, lw=2.2, ls="-", alpha=1.0, ms=6, zorder=3):
        if agg is None or len(agg) == 0:
            return
        xs = agg["length_phonemes"].values
        ys = agg["mean"].values
        se = agg["sem"].values
        ax.plot(xs, ys, color=color, lw=lw, ls=ls, marker="o", ms=ms,
                label=label, alpha=alpha, zorder=zorder)
        ax.fill_between(xs, ys - se, ys + se, color=color, alpha=0.12 * alpha)

    # Main lines (full route)
    real_agg   = _length_curve(df_real,   edit_col)
    pseudo_agg = _length_curve(df_pseudo, edit_col)
    _draw(real_agg,   real_color,   real_label)
    _draw(pseudo_agg, _PSEUDO_COLOR, "Pseudoword")

    if show_routes:
        # Thin supplementary lines for WM and LTM
        wm_edit  = edit_col.replace("full_", "wm_")
        ltm_edit = edit_col.replace("full_", "ltm_")
        if df_real_wm is not None:
            _draw(_length_curve(df_real_wm,   wm_edit),
                  real_color,   f"{real_label} — WM",   lw=1, ls="--", alpha=0.55, ms=3, zorder=2)
            _draw(_length_curve(df_pseudo_wm, wm_edit),
                  _PSEUDO_COLOR, "Pseudoword — WM",     lw=1, ls="--", alpha=0.55, ms=3, zorder=2)
        if df_real_ltm is not None:
            _draw(_length_curve(df_real_ltm,   ltm_edit),
                  real_color,   f"{real_label} — LTM",  lw=1, ls=":", alpha=0.55, ms=3, zorder=2)
            _draw(_length_curve(df_pseudo_ltm, ltm_edit),
                  _PSEUDO_COLOR, "Pseudoword — LTM",    lw=1, ls=":", alpha=0.55, ms=3, zorder=2)


# ---------------------------------------------------------------------------
# Figure 02 — strict Dager (train-seen real vs pseudo)
# ---------------------------------------------------------------------------

def fig_strict_dager(df: pd.DataFrame, out_dir: str, show_routes: bool) -> None:
    df_real   = df[df["lexicon_category"] == "real_word_seen_in_training_lexicon"].copy()
    df_pseudo = df[df["lexicality"] == "pseudo"].copy()

    n_real   = len(df_real)
    n_pseudo = len(df_pseudo)

    real_label = f"Train-seen real words (n={n_real})"

    fig, ax = plt.subplots(figsize=(9, 5))

    _plot_length_curve(
        ax, df_real, df_pseudo,
        real_label=real_label, real_color=_STRICT_REAL_COLOR,
        edit_col="full_edit_dist",
        show_routes=show_routes,
        df_real_wm=df_real if show_routes else None,
        df_pseudo_wm=df_pseudo if show_routes else None,
        df_real_ltm=df_real if show_routes else None,
        df_pseudo_ltm=df_pseudo if show_routes else None,
    )

    # Print how many items in each length group
    for df_grp, lbl in [(df_real, "train-real"), (df_pseudo, "pseudo")]:
        counts = df_grp["length_phonemes"].value_counts().sort_index()
        print(f"    {lbl} length counts: {dict(counts)}")

    ax.set_xlabel("Phoneme length", fontsize=11)
    ax.set_ylabel("Mean edit distance (full route)", fontsize=11)
    ax.set_ylim(bottom=-0.02)
    ax.set_title(
        "WFE: edit distance by phoneme length — Dager-comparable\n"
        "lichtheim3 30k",
        fontsize=11,
    )
    # Subtitle
    ax.text(
        0.5, -0.16,
        "real = WFE real words seen during training;  "
        "pseudo = WFE pseudowords;  teacher-forced",
        transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#444444",
        style="italic",
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    # Annotate count per real length group
    real_agg = _length_curve(df_real, "full_edit_dist")
    for _, row in real_agg.iterrows():
        ax.annotate(f"n={int(row['n'])}", xy=(row["length_phonemes"], row["mean"]),
                    xytext=(2, 6), textcoords="offset points", fontsize=6, color=_STRICT_REAL_COLOR)
    pseudo_agg = _length_curve(df_pseudo, "full_edit_dist")
    for _, row in pseudo_agg.iterrows():
        ax.annotate(f"n={int(row['n'])}", xy=(row["length_phonemes"], row["mean"]),
                    xytext=(2, 6), textcoords="offset points", fontsize=6, color=_PSEUDO_COLOR)

    fig.subplots_adjust(bottom=0.2)
    path = os.path.join(out_dir, "02_dager_fig2_like_wfe_strict_train_real.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [fig] {path}")


# ---------------------------------------------------------------------------
# Figure 02b — dataset-centered (all WFE real vs pseudo)
# ---------------------------------------------------------------------------

def fig_dataset_centered(df: pd.DataFrame, out_dir: str, show_routes: bool) -> None:
    df_real   = df[df["lexicality"] == "real"].copy()
    df_pseudo = df[df["lexicality"] == "pseudo"].copy()

    # Count breakdown of real words by category
    cat_counts = df_real["lexicon_category"].value_counts()
    breakdown  = ", ".join(
        f"{_CAT_LABELS.get(k, k)}={v}"
        for k, v in cat_counts.items()
    )
    print(f"    all-real breakdown: {breakdown}")

    n_real   = len(df_real)
    n_pseudo = len(df_pseudo)
    real_label = f"All WFE real words: train + validation + novel (n={n_real})"

    fig, ax = plt.subplots(figsize=(9, 5))

    _plot_length_curve(
        ax, df_real, df_pseudo,
        real_label=real_label, real_color=_ALL_REAL_COLOR,
        edit_col="full_edit_dist",
        show_routes=show_routes,
        df_real_wm=df_real if show_routes else None,
        df_pseudo_wm=df_pseudo if show_routes else None,
        df_real_ltm=df_real if show_routes else None,
        df_pseudo_ltm=df_pseudo if show_routes else None,
    )

    ax.set_xlabel("Phoneme length", fontsize=11)
    ax.set_ylabel("Mean edit distance (full route)", fontsize=11)
    ax.set_ylim(bottom=-0.02)
    ax.set_title(
        "WFE: edit distance by phoneme length — dataset lexicality\n"
        "lichtheim3 30k  [NOT Dager-comparable — includes unseen real words]",
        fontsize=10,
    )
    ax.text(
        0.5, -0.16,
        "all WFE real words: train + validation + novel;  "
        "pseudo = WFE pseudowords;  teacher-forced",
        transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#444444",
        style="italic",
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    fig.subplots_adjust(bottom=0.2)
    path = os.path.join(out_dir, "02b_dager_fig2_like_wfe_dataset_lexicality.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [fig] {path}")


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
                   help="Add thin WM and LTM lines to the edit-distance curves")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n[plot_wfe_dager_strict_real]")
    print(f"  Input   : {args.pred}")
    print(f"  Output  : {args.out_dir}")
    print(f"  WM/LTM  : {'shown (thin lines)' if args.show_routes else 'hidden'}")

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
        fig_strict_dager(df, args.out_dir, args.show_routes)
    else:
        print("  SKIPPED — missing lexicon_category or lexicality column")

    # --- Figure 02b ---
    print("\n  [02b] Dataset-centered (all WFE real vs pseudo) …")
    if "lexicality" in df.columns:
        fig_dataset_centered(df, args.out_dir, args.show_routes)
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
