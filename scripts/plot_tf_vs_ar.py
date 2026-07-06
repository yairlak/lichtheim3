"""Compare teacher-forced vs autoregressive WFE evaluation — full/gated route only.

Two grouping modes
------------------
dager_strict (default)
    Train-seen real  = lexicon_category == real_word_seen_in_training_lexicon
    Pseudowords      = lexicality in {pseudo, pseudoword}
    Held-out / novel real words are excluded from plotted groups.

seen_vs_unseen
    Train-seen real  = lexicon_category == real_word_seen_in_training_lexicon
    Unseen forms     = all other items (held-out real, novel real, pseudowords)
    NOT a lexicality analysis — "unseen forms" must not be labelled "pseudowords".

Visual encoding
---------------
  Color  = item group    red  = train-seen real  |  blue = pseudowords / unseen forms
  Style  = decoding      solid = teacher-forced  |  dashed = autoregressive

Outputs per group_mode (replace <gm> with dager_strict or seen_vs_unseen)
--------------------------------------------------------------------------
  wfe_tf_vs_ar_<gm>_error_rate.png
  wfe_tf_vs_ar_<gm>_edit_dist.png
  wfe_tf_vs_ar_<gm>_delta_error_rate.png
  wfe_tf_vs_ar_<gm>_delta_edit_dist.png
  wfe_tf_vs_ar_<gm>_combined_4panel.png
  wfe_tf_vs_ar_dager_strict_pseudoword_zoom.png   (dager_strict only)
  wfe_group_audit.tsv
  README.md

Usage
-----
    python scripts/plot_tf_vs_ar.py                   # dager_strict, default paths

    python scripts/plot_tf_vs_ar.py \\
        --tf_pred outputs/external_eval_30k/wfe/item_level_predictions.tsv \\
        --ar_pred outputs/external_eval_30k/wfe_ar/item_level_predictions.tsv \\
        --out_dir outputs/figures_tf_vs_ar \\
        --group_mode dager_strict

    python scripts/plot_tf_vs_ar.py \\
        --tf_pred outputs/external_eval_30k/wfe/item_level_predictions.tsv \\
        --ar_pred outputs/external_eval_30k/wfe_ar/item_level_predictions.tsv \\
        --out_dir outputs/figures_tf_vs_ar_seen_vs_unseen \\
        --group_mode seen_vs_unseen
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
OUT_DEFAULT     = os.path.join(ROOT, "outputs", "figures_tf_vs_ar")
LEXICON_PATH    = os.path.join(ROOT, "data", "lexicon_en_glove_covered.tsv")

GROUP_DAGER       = "dager_strict"
GROUP_SEEN_UNSEEN = "seen_vs_unseen"

_SEEN_CAT    = "real_word_seen_in_training_lexicon"
_VAL_CAT     = "real_word_in_validation_split"
_OUTSIDE_CAT = "real_word_outside_4000_lexicon"
_PSEUDO_CAT  = "pseudoword"

_REAL_COLOR   = "#c0392b"   # red  — train-seen real
_UNSEEN_COLOR = "#2c3e7f"   # blue — pseudowords / unseen forms

_X_LABEL = "Word length (phonemes)"
_ENCODING_NOTE = (
    "Color = item group  ·  solid = teacher-forced  ·  dashed = autoregressive  ·  "
    "95 % bootstrap CI (some bands invisible when interval = 0)"
)

# Robust pseudoword detection: accept "pseudo" or "pseudoword" in lexicality column
_PSEUDO_LEX_VALUES = {"pseudo", "pseudoword"}


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def _bootstrap_ci(values: np.ndarray, n_boot: int,
                  rng: np.random.RandomState) -> Tuple[float, float]:
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
                     "ci_lo": ci_lo, "ci_hi": ci_hi, "n": int(len(vals))})
    return pd.DataFrame(rows).sort_values("length_phonemes").reset_index(drop=True)


def _delta_curve(sub_tf: pd.DataFrame, sub_ar: pd.DataFrame,
                 tf_col: str, ar_col: str,
                 n_boot: int = 1000, min_n: int = 3,
                 rng: Optional[np.random.RandomState] = None) -> pd.DataFrame:
    """AR − TF delta per phoneme-length bin with 95 % bootstrap CI.

    sub_tf and sub_ar must be positionally aligned (same items, same order).
    """
    if rng is None:
        rng = np.random.RandomState(42)
    assert len(sub_tf) == len(sub_ar), (
        f"TF ({len(sub_tf)}) and AR ({len(sub_ar)}) must be same size for delta")
    delta  = sub_ar[ar_col].values - sub_tf[tf_col].values
    lengths = sub_tf["length_phonemes"].values
    rows = []
    for length in np.unique(lengths):
        mask = lengths == length
        vals = delta[mask]
        if mask.sum() < min_n:
            continue
        mean_val = float(vals.mean())
        ci_lo, ci_hi = _bootstrap_ci(vals, n_boot, rng)
        rows.append({"length_phonemes": int(length), "mean": mean_val,
                     "ci_lo": ci_lo, "ci_hi": ci_hi, "n": int(mask.sum())})
    return pd.DataFrame(rows).sort_values("length_phonemes").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Data loading and alignment
# ---------------------------------------------------------------------------

def _load_and_align(tf_path: str, ar_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both TSVs, filter excluded items, verify alignment, return (df_tf, df_ar)."""
    df_tf = pd.read_csv(tf_path, sep="\t")
    df_ar = pd.read_csv(ar_path, sep="\t")
    df_tf = df_tf[df_tf["notes"].fillna("").str.strip() == ""].reset_index(drop=True)
    df_ar = df_ar[df_ar["notes"].fillna("").str.strip() == ""].reset_index(drop=True)

    print(f"  TF: {len(df_tf)} items after filtering excluded")
    print(f"  AR: {len(df_ar)} items after filtering excluded")

    if len(df_tf) != len(df_ar):
        raise ValueError(
            f"TF ({len(df_tf)}) and AR ({len(df_ar)}) have different row counts. "
            "They must be evaluated from the same WFE TSV with identical filtering.")

    # Add error_rate columns if missing
    for df in (df_tf, df_ar):
        for route in ("full", "wm", "ltm"):
            em = f"{route}_exact_match"
            if em in df.columns and f"{route}_error_rate" not in df.columns:
                df[f"{route}_error_rate"] = 1.0 - df[em]

    return df_tf, df_ar


# ---------------------------------------------------------------------------
# Group building
# ---------------------------------------------------------------------------

def _is_pseudo(df: pd.DataFrame) -> pd.Series:
    return df["lexicality"].str.lower().str.strip().isin(_PSEUDO_LEX_VALUES)


def build_groups(df_tf: pd.DataFrame, df_ar: pd.DataFrame,
                 group_mode: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Return {group_label: {"tf": sub_tf, "ar": sub_ar}} for the requested mode.

    sub_tf and sub_ar in each group are positionally aligned (same rows selected
    from the identically-ordered df_tf / df_ar).
    """
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
        mask_unseen = ~mask_seen
        return {
            "Train-seen real": {
                "tf": df_tf[mask_seen].reset_index(drop=True),
                "ar": df_ar[mask_seen].reset_index(drop=True),
            },
            "Unseen forms": {
                "tf": df_tf[mask_unseen].reset_index(drop=True),
                "ar": df_ar[mask_unseen].reset_index(drop=True),
            },
        }
    else:
        raise ValueError(f"Unknown group_mode: {group_mode!r}")


def _group_color(group_name: str) -> str:
    return _REAL_COLOR if group_name == "Train-seen real" else _UNSEEN_COLOR


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_line(ax: plt.Axes, agg: pd.DataFrame,
               color: str, ls: str, label: Optional[str],
               lw: float = 2.0, ms: float = 5,
               ci_alpha: float = 0.13, zorder: int = 3) -> None:
    if agg is None or len(agg) == 0:
        return
    xs = agg["length_phonemes"].values
    ys = agg["mean"].values
    lo = agg["ci_lo"].values
    hi = agg["ci_hi"].values
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, marker="o", ms=ms,
            label=label, zorder=zorder)
    ax.fill_between(xs, lo, hi, color=color, alpha=ci_alpha)


def _finalize_ax(ax: plt.Axes, ylabel: str, title: str,
                 bottom: float = -0.02, top: Optional[float] = None,
                 show_legend: bool = True, legend_ncol: int = 2,
                 subtitle: Optional[str] = None) -> None:
    ax.set_xlabel(_X_LABEL, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(bottom=bottom, top=top)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    if show_legend:
        ax.legend(fontsize=8, ncol=legend_ncol)
    if subtitle is not None:
        ax.text(0.5, -0.17, subtitle,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=7.5, color="#444444", style="italic")


def _save(fig: plt.Figure, path: str,
          bottom_adjust: Optional[float] = 0.20) -> None:
    if bottom_adjust is not None:
        fig.subplots_adjust(bottom=bottom_adjust)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [fig] {path}")


# ---------------------------------------------------------------------------
# Raw-curve figure (error rate or edit distance)
# ---------------------------------------------------------------------------

def _draw_raw_panel(ax: plt.Axes,
                    groups: Dict[str, Dict[str, pd.DataFrame]],
                    col: str, n_boot: int,
                    rng: np.random.RandomState) -> None:
    regime_label = {"tf": "Teacher-forced", "ar": "Autoregressive"}
    regime_ls    = {"tf": "-",              "ar": "--"}
    for group_name, pair in groups.items():
        color = _group_color(group_name)
        for regime in ("tf", "ar"):
            sub = pair[regime]
            n   = len(sub)
            agg = _length_curve(sub, col, n_boot=n_boot, rng=rng)
            label = f"{group_name} — {regime_label[regime]}  (n={n})"
            _draw_line(ax, agg, color=color, ls=regime_ls[regime], label=label)


def fig_raw(groups: Dict[str, Dict[str, pd.DataFrame]],
            col: str, ylabel: str,
            group_mode: str, out_dir: str, n_boot: int, prefix: str) -> None:
    rng = np.random.RandomState(42)
    fig, ax = plt.subplots(figsize=(9, 5))
    _draw_raw_panel(ax, groups, col, n_boot, rng)
    _finalize_ax(
        ax, ylabel,
        title=(f"WFE: teacher-forced vs autoregressive\n"
               f"lichtheim3 30k  ·  full/gated route  ·  WM noise disabled"),
        top=(min(1.05, ax.get_ylim()[1] + 0.05) if "error" in col else None),
        subtitle=_ENCODING_NOTE,
    )
    _save(fig, os.path.join(out_dir, f"{prefix}_{col.replace('full_', '')}.png"))


# ---------------------------------------------------------------------------
# Delta figure (AR − TF)
# ---------------------------------------------------------------------------

def _draw_delta_panel(ax: plt.Axes,
                      groups: Dict[str, Dict[str, pd.DataFrame]],
                      col: str, n_boot: int,
                      rng: np.random.RandomState) -> None:
    ax.axhline(0, color="gray", lw=1.0, ls="--", alpha=0.6, zorder=1)
    for group_name, pair in groups.items():
        color = _group_color(group_name)
        n     = len(pair["tf"])
        agg   = _delta_curve(pair["tf"], pair["ar"], col, col,
                              n_boot=n_boot, rng=rng)
        _draw_line(ax, agg, color=color, ls="-",
                   label=f"{group_name}  (n={n})")


def fig_delta(groups: Dict[str, Dict[str, pd.DataFrame]],
              col: str, ylabel: str,
              group_mode: str, out_dir: str, n_boot: int, prefix: str) -> None:
    rng = np.random.RandomState(42)
    fig, ax = plt.subplots(figsize=(9, 5))
    _draw_delta_panel(ax, groups, col, n_boot, rng)
    _finalize_ax(
        ax, f"AR − TF  ({ylabel.lower()})",
        title=(f"WFE: AR − TF delta\n"
               f"lichtheim3 30k  ·  full/gated route  ·  positive = autoregressive worse"),
        subtitle=f"{_ENCODING_NOTE}  ·  dashed grey = no difference",
    )
    _save(fig, os.path.join(out_dir, f"{prefix}_delta_{col.replace('full_', '')}.png"))


# ---------------------------------------------------------------------------
# 4-panel combined figure
# ---------------------------------------------------------------------------

def fig_4panel(groups: Dict[str, Dict[str, pd.DataFrame]],
               group_mode: str, out_dir: str, n_boot: int, prefix: str) -> None:
    rng = np.random.RandomState(42)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    (ax_err, ax_ed), (ax_d_err, ax_d_ed) = axes

    _draw_raw_panel(ax_err, groups, "full_error_rate",  n_boot, rng)
    _draw_raw_panel(ax_ed,  groups, "full_edit_dist",   n_boot, rng)
    _draw_delta_panel(ax_d_err, groups, "full_error_rate", n_boot, rng)
    _draw_delta_panel(ax_d_ed,  groups, "full_edit_dist",  n_boot, rng)

    top_err = min(1.05, ax_err.get_ylim()[1] + 0.05)

    for ax in (ax_err, ax_ed, ax_d_err, ax_d_ed):
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    ax_err.set_ylabel("Error rate  (1 − exact-match)", fontsize=10)
    ax_err.set_ylim(-0.02, top_err)
    ax_err.set_title("Raw — Error rate", fontsize=9)
    ax_err.legend(fontsize=7, ncol=1)

    ax_ed.set_ylabel("Mean edit distance", fontsize=10)
    ax_ed.set_ylim(bottom=-0.02)
    ax_ed.set_title("Raw — Edit distance", fontsize=9)
    ax_ed.legend(fontsize=7, ncol=1)

    ax_d_err.set_ylabel("AR − TF  (error rate)", fontsize=10)
    ax_d_err.set_title("Delta — Error rate", fontsize=9)
    ax_d_err.legend(fontsize=7)

    ax_d_ed.set_ylabel("AR − TF  (edit distance)", fontsize=10)
    ax_d_ed.set_title("Delta — Edit distance", fontsize=9)
    ax_d_ed.legend(fontsize=7)

    for ax in (ax_err, ax_d_err):
        ax.set_xlabel(_X_LABEL, fontsize=10)
    for ax in (ax_ed, ax_d_ed):
        ax.set_xlabel(_X_LABEL, fontsize=10)

    fig.suptitle(
        "WFE: teacher-forced vs autoregressive  ·  lichtheim3 30k  ·  full/gated route  ·  WM noise disabled",
        fontsize=10, y=0.99,
    )
    fig.text(0.5, 0.01, _ENCODING_NOTE,
             ha="center", fontsize=7.5, color="#444444", style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    _save(fig, os.path.join(out_dir, f"{prefix}_combined_4panel.png"),
          bottom_adjust=None)


# ---------------------------------------------------------------------------
# Pseudoword zoom figure (dager_strict only)
# ---------------------------------------------------------------------------

def fig_pseudoword_zoom(groups: Dict[str, Dict[str, pd.DataFrame]],
                         group_name_key: str,
                         out_dir: str, n_boot: int, prefix: str) -> None:
    pair = groups.get(group_name_key)
    if pair is None:
        print(f"  [skip pseudoword zoom] group '{group_name_key}' not in groups")
        return

    rng  = np.random.RandomState(42)
    n_tf = len(pair["tf"])
    n_ar = len(pair["ar"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, ylabel in [
        (ax1, "full_error_rate", "Error rate  (1 − exact-match)"),
        (ax2, "full_edit_dist",  "Mean edit distance"),
    ]:
        agg_tf = _length_curve(pair["tf"], col, n_boot=n_boot, rng=rng)
        agg_ar = _length_curve(pair["ar"], col, n_boot=n_boot, rng=rng)
        color  = _group_color(group_name_key)
        _draw_line(ax, agg_tf, color=color, ls="-",
                   label=f"{group_name_key} — Teacher-forced  (n={n_tf})")
        _draw_line(ax, agg_ar, color=color, ls="--",
                   label=f"{group_name_key} — Autoregressive  (n={n_ar})")
        ax.set_xlabel(_X_LABEL, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        if "error" in col:
            ax.set_ylim(-0.02, min(1.05, ax.get_ylim()[1] + 0.05))
        else:
            ax.set_ylim(bottom=-0.02)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=8)

    fig.suptitle(
        f"WFE zoom: {group_name_key}  ·  teacher-forced vs autoregressive\n"
        "lichtheim3 30k  ·  full/gated route  ·  WM noise disabled  ·  95 % bootstrap CI",
        fontsize=10,
    )
    fig.text(0.5, 0.00, _ENCODING_NOTE,
             ha="center", fontsize=7.5, color="#444444", style="italic")
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    _save(fig, os.path.join(out_dir, f"{prefix}_pseudoword_zoom.png"),
          bottom_adjust=None)


# ---------------------------------------------------------------------------
# Audit table
# ---------------------------------------------------------------------------

def _load_glove_words() -> Optional[set]:
    if not os.path.exists(LEXICON_PATH):
        return None
    try:
        df = pd.read_csv(LEXICON_PATH, sep="\t")
        col = next((c for c in df.columns if c.lower() == "word"), None)
        if col:
            return {str(w).lower().strip() for w in df[col].dropna()}
    except Exception:
        pass
    return None


def write_audit(df_tf: pd.DataFrame, out_dir: str) -> None:
    glove_words = _load_glove_words()

    group_defs = [
        ("Train-seen real",    df_tf["lexicon_category"] == _SEEN_CAT),
        ("Held-out real",      df_tf["lexicon_category"] == _VAL_CAT),
        ("Novel/outside real", df_tf["lexicon_category"] == _OUTSIDE_CAT),
        ("Pseudowords",        _is_pseudo(df_tf)),
        # Unseen forms = everything NOT in train split
        ("Unseen forms",       df_tf["lexicon_category"] != _SEEN_CAT),
    ]

    rows = []
    for group_name, mask in group_defs:
        sub = df_tf[mask]
        n   = int(len(sub))

        lex_vals = (", ".join(sorted(sub["lexicality"].dropna().unique().tolist()))
                    if n > 0 else "")
        cat_vals = (", ".join(sorted(sub["lexicon_category"].dropna().unique().tolist()))
                    if n > 0 else "")

        in_dager     = "yes" if group_name in ("Train-seen real", "Pseudowords") else "no"
        in_seen_vs   = "yes"

        if group_name == "Pseudowords":
            glove_status = "N/A (pseudowords)"
            train_status = "N/A (pseudowords)"
        elif glove_words is None:
            glove_status = "unknown (lexicon file not found)"
            train_status = ("yes (all)" if group_name == "Train-seen real"
                            else "no" if n > 0 else "N/A")
        elif n > 0 and "word" in sub.columns:
            n_in = int(sub["word"].str.lower().str.strip().isin(glove_words).sum())
            glove_status = f"{n_in}/{n} in GloVe-covered lexicon"
            train_status = ("yes (all)" if group_name == "Train-seen real" else "no")
        else:
            glove_status = "N/A"
            train_status = "N/A"

        note_map = {
            "Train-seen real":    "Dager-comparable real-word group.",
            "Held-out real":      "Seen during lexicon construction; never trained on.",
            "Novel/outside real": "Real words not in GloVe-covered lexicon; never trained on.",
            "Pseudowords":        "WFE pseudowords — excluded from seen_vs_unseen split label.",
            "Unseen forms":       ("Aggregate: held-out real + novel real + pseudowords. "
                                   "NOT a lexicality group — do not label as 'pseudowords'."),
        }

        rows.append({
            "group_name":                            group_name,
            "n_items":                               n,
            "lexicality_values_included":            lex_vals,
            "lexicon_categories_included":           cat_vals,
            "included_in_dager_strict":              in_dager,
            "included_in_seen_vs_unseen":            in_seen_vs,
            "in_current_glove_covered_lexicon":      glove_status,
            "in_train_split":                        train_status,
            "notes":                                 note_map.get(group_name, ""),
        })

    path = os.path.join(out_dir, "wfe_group_audit.tsv")
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    print(f"  [audit] {path}")


# ---------------------------------------------------------------------------
# README generation
# ---------------------------------------------------------------------------

_README_DAGER = """\
# WFE TF-vs-AR figures: Dager-strict grouping

## What this is

Teacher-forced vs autoregressive evaluation of the lichtheim3 30k model on the
WFE dataset, using the Dager-comparable grouping.

## Group definitions

| Group | Definition |
|---|---|
| Train-seen real | WFE real words whose lexicon_category == real_word_seen_in_training_lexicon (seen during training) |
| Pseudowords | WFE items with lexicality in {pseudo, pseudoword} |

Held-out real words (real_word_in_validation_split) and novel/outside real words
(real_word_outside_4000_lexicon) are excluded from the main Dager-strict plots.
They are audited in wfe_group_audit.tsv.

When comparing with Dager/SWP: use these figures. Do NOT use the seen_vs_unseen
figures for Dager comparison.

## Visual encoding

| Visual property | Meaning |
|---|---|
| Red | Train-seen real words |
| Blue | Pseudowords |
| Solid line | Teacher-forced decoding (gold prefix at each step) |
| Dashed line | Autoregressive decoding (model's own previous output at each step) |

Some confidence bands are invisible when the 95 % bootstrap CI interval is
exactly zero (e.g., for train-seen real words with perfect accuracy).

## Model notes

- GloVe is NOT provided as input at WFE evaluation time. It was used only as an
  alignment target (L_align loss) during training.
- WM interference noise is disabled for all figures here (collect=False).
- Teacher-forced = upper bound; autoregressive = behaviorally plausible.
- All figures: full/gated route only.

## Files

- wfe_tf_vs_ar_dager_strict_error_rate.png
- wfe_tf_vs_ar_dager_strict_edit_dist.png
- wfe_tf_vs_ar_dager_strict_delta_error_rate.png
- wfe_tf_vs_ar_dager_strict_delta_edit_dist.png
- wfe_tf_vs_ar_dager_strict_combined_4panel.png
- wfe_tf_vs_ar_dager_strict_pseudoword_zoom.png
- wfe_group_audit.tsv (all 5 group categories with GloVe/train audit)
"""

_README_SEEN_UNSEEN = """\
# WFE TF-vs-AR figures: seen_vs_unseen grouping

## What this is

Teacher-forced vs autoregressive evaluation of the lichtheim3 30k model on the
WFE dataset, using a familiarity/generalization grouping.

## IMPORTANT: this is NOT a lexicality analysis

"Unseen forms" is not equivalent to "pseudowords".

| Group | Definition |
|---|---|
| Train-seen real | WFE real words whose lexicon_category == real_word_seen_in_training_lexicon (seen during training) |
| Unseen forms | ALL other WFE items: held-out real + novel/outside real + pseudowords |

The "unseen forms" group contains a mixture of:
- Real words that were held out from training (model never saw them as training items)
- Real words absent from the GloVe-covered lexicon entirely
- WFE pseudowords

Do not interpret "unseen forms" as "pseudowords" in any analysis or write-up.

## Visual encoding

| Visual property | Meaning |
|---|---|
| Red | Train-seen real words |
| Blue | Unseen forms (held-out real + novel real + pseudowords) |
| Solid line | Teacher-forced decoding |
| Dashed line | Autoregressive decoding |

Some confidence bands are invisible when the 95 % bootstrap CI interval is
exactly zero.

## Lexicon notes

- Held-out real words may be GloVe-covered (they have GloVe embeddings) but
  were never used as training items (only in the validation split).
- Novel/outside real words are audited separately for GloVe/lexicon inclusion
  in wfe_group_audit.tsv.
- GloVe is NOT provided as input at WFE evaluation time.

## Files

- wfe_tf_vs_ar_seen_vs_unseen_error_rate.png
- wfe_tf_vs_ar_seen_vs_unseen_edit_dist.png
- wfe_tf_vs_ar_seen_vs_unseen_delta_error_rate.png
- wfe_tf_vs_ar_seen_vs_unseen_delta_edit_dist.png
- wfe_tf_vs_ar_seen_vs_unseen_combined_4panel.png
- wfe_group_audit.tsv (all 5 group categories with GloVe/train audit)
"""


def write_readme(group_mode: str, out_dir: str) -> None:
    content = _README_DAGER if group_mode == GROUP_DAGER else _README_SEEN_UNSEEN
    path = os.path.join(out_dir, "README.md")
    with open(path, "w") as f:
        f.write(content)
    print(f"  [readme] {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare teacher-forced vs autoregressive WFE evaluation.")
    p.add_argument("--tf_pred", default=TF_PRED_DEFAULT,
                   help="item_level_predictions.tsv from teacher-forced eval")
    p.add_argument("--ar_pred", default=AR_PRED_DEFAULT,
                   help="item_level_predictions.tsv from autoregressive eval")
    p.add_argument("--out_dir", default=OUT_DEFAULT,
                   help="Output directory for figures")
    p.add_argument("--group_mode", default=GROUP_DAGER,
                   choices=[GROUP_DAGER, GROUP_SEEN_UNSEEN],
                   help=(f"'{GROUP_DAGER}': train-seen real vs pseudowords "
                         f"(Dager-comparable).  "
                         f"'{GROUP_SEEN_UNSEEN}': train-seen real vs all unseen "
                         f"forms (generalization analysis, NOT lexicality)."))
    p.add_argument("--n_boot", type=int, default=1000,
                   help="Bootstrap resamples for 95 %% CI (0 = point estimate only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n[plot_tf_vs_ar]")
    print(f"  TF pred    : {args.tf_pred}")
    print(f"  AR pred    : {args.ar_pred}")
    print(f"  Output     : {args.out_dir}")
    print(f"  group_mode : {args.group_mode}")
    print(f"  n_boot     : {args.n_boot}")

    for path, label in [(args.tf_pred, "TF"), (args.ar_pred, "AR")]:
        if not os.path.exists(path):
            print(f"\nERROR: {label} predictions TSV not found:\n  {path}")
            if label == "AR":
                print("  Run first:\n"
                      "  python scripts/external_eval.py \\\n"
                      "      --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt \\\n"
                      "      --out_dir outputs/external_eval_30k \\\n"
                      "      --decode autoregressive --wfe_only")
            else:
                print("  Run first:\n"
                      "  python scripts/external_eval.py \\\n"
                      "      --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt \\\n"
                      "      --out_dir outputs/external_eval_30k --wfe_only")
            sys.exit(1)

    df_tf, df_ar = _load_and_align(args.tf_pred, args.ar_pred)

    required_cols = ("lexicon_category", "lexicality", "length_phonemes",
                     "full_exact_match", "full_edit_dist")
    for col in required_cols:
        for df, label in [(df_tf, "TF"), (df_ar, "AR")]:
            if col not in df.columns:
                print(f"ERROR: column '{col}' missing from {label} predictions TSV")
                sys.exit(1)

    # Build groups
    groups = build_groups(df_tf, df_ar, args.group_mode)
    print(f"\n  Group sizes ({args.group_mode}):")
    for gname, pair in groups.items():
        print(f"    {gname:25s}  TF n={len(pair['tf'])}  AR n={len(pair['ar'])}")

    # Output filename prefix
    prefix = f"wfe_tf_vs_ar_{args.group_mode}"

    print(f"\n  Generating figures …  (n_boot={args.n_boot})")

    # A/B: raw error rate and edit distance
    fig_raw(groups, "full_error_rate",
            "Error rate  (1 − exact-match)",
            args.group_mode, args.out_dir, args.n_boot, prefix)
    fig_raw(groups, "full_edit_dist",
            "Mean edit distance",
            args.group_mode, args.out_dir, args.n_boot, prefix)

    # C/D: delta plots
    fig_delta(groups, "full_error_rate",
              "Error rate",
              args.group_mode, args.out_dir, args.n_boot, prefix)
    fig_delta(groups, "full_edit_dist",
              "Edit distance",
              args.group_mode, args.out_dir, args.n_boot, prefix)

    # E: combined 4-panel
    fig_4panel(groups, args.group_mode, args.out_dir, args.n_boot, prefix)

    # Pseudoword zoom (dager_strict only)
    if args.group_mode == GROUP_DAGER:
        fig_pseudoword_zoom(groups, "Pseudowords",
                            args.out_dir, args.n_boot, prefix)

    # Audit table and README
    write_audit(df_tf, args.out_dir)
    write_readme(args.group_mode, args.out_dir)

    # Summary
    print(f"\n[plot_tf_vs_ar] Done.  Output: {args.out_dir}")
    generated = sorted(f for f in os.listdir(args.out_dir)
                       if f.endswith((".png", ".tsv", ".md")))
    for f in generated:
        size_kb = os.path.getsize(os.path.join(args.out_dir, f)) // 1024
        print(f"    {f}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
