#!/usr/bin/env python3
"""Plot Phase 6A gridsearch and low-LR continuation (Phase 6B/C) summary.

Outputs (all in --out-dir):
  phase6a_phase6bc_bottomline.png / .pdf / .svg
  bottomline_e120_clean.csv
  bottomline_e120_table.md
  phase6a_phase6bc_caption.md

Usage:
  python scripts/plot_phase6a_bottomline.py \\
      --tsv outputs/phase6a_e120/aggregate/bottomline_e120.tsv \\
      --out-dir outputs/phase6a_e120/aggregate/figures \\
      --trajectory "30:43,60:44,120:15,160:3,200:4"
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Visual constants (colorblind-friendly palette)
# ---------------------------------------------------------------------------

_TF_COLORS = {0.0: "#0072B2", 0.2: "#E69F00", 1.0: "#009E73"}
_H_MARKERS  = {64: "o", 128: "s", 256: "^"}
_MS         = 10   # marker size
_MS_OPEN    = 10
_LW_OPEN    = 1.8

# ---------------------------------------------------------------------------
# Column alias resolution
# ---------------------------------------------------------------------------

_ALIASES: dict[str, list[str]] = {
    "run_id": [
        "run_id", "run", "run_name", "ckpt_dir", "checkpoint", "config", "experiment",
    ],
    "H": [
        "H", "h", "hidden", "hidden_size", "hidden_dim", "wm_hidden", "ltm_hidden", "model_hidden",
    ],
    "TF": [
        "TF", "tf", "tf_ratio", "teacher_forcing_ratio", "teacher_forcing", "tfr",
    ],
    "LR": [
        "LR", "lr", "learning_rate", "train_lr",
    ],
    "dropout": [
        "dropout", "dropout_rate",
    ],
    "train_errors": [
        "train_full_errors", "full_ar_errors_train", "full_errors_train",
        "errors_train", "train_errors", "train_n_errors", "train_exact_errors",
        "train_word_errors",
    ],
    "train_exact": [
        "train_full_exact", "full_ar_exact_train", "full_exact_train",
        "exact_train", "train_exact", "train_exact_match",
    ],
    "val_exact": [
        "val_full_exact", "full_ar_exact_val", "full_exact_val",
        "exact_val", "val_exact", "validation_exact", "validation_full_exact",
    ],
    "val_edit": [
        "val_edit", "full_ar_edit_val", "full_edit_val", "edit_val", "validation_edit",
    ],
    "val_ned": [
        "val_ned", "full_ar_ned_val", "full_ar_norm_edit_val", "norm_edit_val",
        "ned_val", "validation_ned",
    ],
    "zero_train": [
        "zero_train", "reached_zero_train", "train_zero_errors",
    ],
}

_REQUIRED = {"run_id", "H", "TF", "LR", "train_errors", "val_exact"}
_OPTIONAL  = {"train_exact", "dropout", "val_edit", "val_ned", "zero_train"}


def _norm(name: str) -> str:
    return re.sub(r"[\s\.\-]+", "_", name.strip().lower())


def resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and rename columns to canonical names via alias table."""
    df = df.rename(columns={c: _norm(c) for c in df.columns})
    rev: dict[str, str] = {}
    for canon, aliases in _ALIASES.items():
        for alias in aliases:
            n = _norm(alias)
            if n not in rev:
                rev[n] = canon
    rename = {}
    already_canon: set[str] = set()
    for col in df.columns:
        canon = rev.get(col)
        if canon and canon not in already_canon:
            rename[col] = canon
            already_canon.add(canon)
    return df.rename(columns=rename)


# ---------------------------------------------------------------------------
# Inference from run_id
# ---------------------------------------------------------------------------

_RE_H   = re.compile(r"[Hh]0*(\d+)")
_RE_TF  = re.compile(r"tf(\d+)p(\d+)")
_RE_LR  = re.compile(r"lr(\d+(?:e[+-]?\d+)?(?:e-\d+)?)", re.IGNORECASE)


def _infer_H(run_id: str) -> Optional[int]:
    m = _RE_H.search(run_id)
    return int(m.group(1)) if m else None


def _infer_TF(run_id: str) -> Optional[float]:
    m = _RE_TF.search(run_id)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    return None


def _infer_LR(run_id: str) -> Optional[float]:
    m = _RE_LR.search(run_id)
    if m:
        try:
            return float(m.group(1).replace("e-", "e-"))
        except ValueError:
            return None
    return None


def infer_from_run_id(df: pd.DataFrame) -> pd.DataFrame:
    """Fill H, TF, LR from run_id where missing."""
    for col, fn in [("H", _infer_H), ("TF", _infer_TF), ("LR", _infer_LR)]:
        if col not in df.columns or df[col].isna().any():
            print(f"  [infer] column '{col}' missing or has NaN — inferring from run_id")
            if col not in df.columns:
                df[col] = None
            mask = df[col].isna()
            inferred = df.loc[mask, "run_id"].map(fn)
            df.loc[mask, col] = inferred
    return df


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_trajectory(s: str) -> list[tuple[int, int]]:
    """Parse 'epoch:errors,epoch:errors,...' → sorted list of (epoch, errors)."""
    try:
        pts = []
        for part in s.split(","):
            ep, err = part.strip().split(":")
            pts.append((int(ep.strip()), int(err.strip())))
        if len(pts) < 2:
            raise ValueError("need at least two points")
        return sorted(pts)
    except Exception as e:
        print(f"ERROR: --trajectory format invalid: {s!r} — {e}", file=sys.stderr)
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 6A bottom-line figure")
    p.add_argument("--tsv",        required=True,   help="Input TSV file")
    p.add_argument("--out-dir",    required=True,   help="Output directory")
    p.add_argument("--title",      default="Lichtheim3 — Phase 6A gridsearch and low-LR continuation")
    p.add_argument("--dpi",        type=int,   default=300)
    p.add_argument("--show",       action="store_true")
    p.add_argument("--trajectory", default="30:43,60:44,120:15,160:3,200:4",
                   help="epoch:errors pairs for Panel B, comma-separated")
    p.add_argument("--train-size", type=int,   default=25136)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------

_EXPECTED_H  = {64, 128, 256}
_EXPECTED_TF = {0.0, 0.2, 1.0}
_EXPECTED_LR = {5e-4, 1e-3}
_N_EXPECTED  = 18


def load_and_prepare(tsv: str, train_size: int) -> pd.DataFrame:
    print(f"\n[inspect] using TSV: {tsv}")
    try:
        # Handle optional leading integer-index column written by pandas
        raw = pd.read_csv(tsv, sep="\t")
    except Exception as e:
        print(f"ERROR: cannot read {tsv}: {e}", file=sys.stderr)
        sys.exit(1)

    # Drop unnamed leading index column if present
    unnamed = [c for c in raw.columns if re.match(r"^(unnamed|)\s*:?\s*0$", c.lower().strip())]
    if not unnamed and raw.columns[0] not in [_norm(a) for aliases in _ALIASES.values() for a in aliases]:
        try:
            pd.to_numeric(raw.iloc[:, 0])
            unnamed = [raw.columns[0]]
        except (ValueError, TypeError):
            pass
    if unnamed:
        raw = raw.drop(columns=unnamed)

    print(f"  shape       : {raw.shape}")
    print(f"  columns     : {list(raw.columns)}")
    print(f"  n_rows      : {len(raw)}")
    print(f"\n  first 3 rows:")
    print(raw.head(3).to_string(index=False))
    print()

    df = resolve_columns(raw)
    df = infer_from_run_id(df)

    # --- check required fields ---
    missing_req = _REQUIRED - set(df.columns)
    if missing_req:
        print(f"ERROR: required columns not found or inferrable: {missing_req}", file=sys.stderr)
        print(f"  available (normalised): {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    # --- coerce types ---
    df["H"]  = pd.to_numeric(df["H"],  errors="coerce").astype(int)
    df["TF"] = pd.to_numeric(df["TF"], errors="coerce").astype(float)
    df["LR"] = pd.to_numeric(df["LR"], errors="coerce").astype(float)
    df["train_errors"] = pd.to_numeric(df["train_errors"], errors="coerce").astype(int)
    df["val_exact"]    = pd.to_numeric(df["val_exact"],    errors="coerce").astype(float)

    # --- optional: train_exact ---
    if "train_exact" not in df.columns or df["train_exact"].isna().all():
        print("  [warn] 'train_exact' absent — deriving from train_errors / train_size")
        df["train_exact"] = 1.0 - df["train_errors"] / train_size
        df["_train_exact_derived"] = True
    else:
        df["train_exact"] = pd.to_numeric(df["train_exact"], errors="coerce").astype(float)
        df["_train_exact_derived"] = False

    # --- optional: dropout ---
    if "dropout" not in df.columns or df["dropout"].isna().all():
        print("  [warn] 'dropout' absent — assuming 0.0 (Phase 6A spec guarantees dropout=0.0)")
        df["dropout"] = 0.0

    # --- optional: zero_train ---
    if "zero_train" not in df.columns or df["zero_train"].isna().all():
        df["zero_train"] = df["train_errors"] == 0
    else:
        # Coerce: could be bool string "False"/"True"
        zt = df["zero_train"]
        if zt.dtype == object:
            df["zero_train"] = zt.str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False})
        df["zero_train"] = df["zero_train"].astype(bool)

    # --- optional numeric ---
    for col in ("val_edit", "val_ned"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def validate_grid(df: pd.DataFrame) -> None:
    """Validate the 18-run factorial grid."""
    errors = []

    if len(df) != _N_EXPECTED:
        errors.append(f"expected {_N_EXPECTED} rows, got {len(df)}")

    H_vals  = set(df["H"].unique())
    TF_vals = set(df["TF"].unique())
    LR_vals = set(df["LR"].unique())

    if H_vals != _EXPECTED_H:
        errors.append(f"H values: expected {_EXPECTED_H}, got {H_vals}")
    if TF_vals != _EXPECTED_TF:
        errors.append(f"TF values: expected {_EXPECTED_TF}, got {TF_vals}")
    if not np.allclose(sorted(LR_vals), sorted(_EXPECTED_LR), rtol=1e-3):
        errors.append(f"LR values: expected {_EXPECTED_LR}, got {LR_vals}")

    # Each H×TF×LR appears exactly once
    combos = df.groupby(["H", "TF", "LR"]).size()
    dups = combos[combos > 1]
    if len(dups):
        errors.append(f"duplicate H×TF×LR combinations: {dups.to_dict()}")
    n_combos = len(_EXPECTED_H) * len(_EXPECTED_TF) * len(_EXPECTED_LR)
    if len(combos) != n_combos:
        errors.append(f"expected {n_combos} unique H×TF×LR combinations, got {len(combos)}")

    if df["run_id"].nunique() != len(df):
        errors.append("duplicate run_ids detected")

    if df[["H", "TF", "LR", "train_errors", "val_exact"]].isna().any().any():
        errors.append("NaN values in required numeric columns")

    if (df["train_errors"] < 0).any():
        errors.append("negative train_errors detected")

    if not ((df["train_exact"] >= 0) & (df["train_exact"] <= 1)).all():
        errors.append("train_exact out of [0, 1]")

    if not ((df["val_exact"] >= 0) & (df["val_exact"] <= 1)).all():
        errors.append("val_exact out of [0, 1]")

    if not df.get("_train_exact_derived", pd.Series([True] * len(df))).all():
        # Check internal consistency: train_exact ≈ 1 - errors / train_size
        pass  # will check below after knowing train_size

    if df["dropout"].notna().all():
        if not (df["dropout"] == 0.0).all():
            errors.append(f"dropout not uniformly 0.0: {df['dropout'].unique()}")

    if errors:
        print("GRID VALIDATION FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    print("  [validate] grid OK: 18 rows, H×TF×LR factorial, no duplicates")


def check_consistency(df: pd.DataFrame, train_size: int) -> None:
    """Check train_exact ≈ 1 - train_errors / train_size."""
    derived_flag = df.get("_train_exact_derived", pd.Series([True] * len(df)))
    if derived_flag.all():
        return  # was derived, no inconsistency possible
    expected = 1.0 - df["train_errors"] / train_size
    diff = (df["train_exact"] - expected).abs()
    bad = df[diff > 5e-5]
    if len(bad):
        print("CONSISTENCY ERROR: train_exact vs 1-train_errors/train_size mismatch:",
              file=sys.stderr)
        for _, row in bad.iterrows():
            exp = 1.0 - row["train_errors"] / train_size
            print(f"  run={row['run_id']}: observed={row['train_exact']:.6f} "
                  f"expected≈{exp:.6f} diff={abs(row['train_exact']-exp):.2e}", file=sys.stderr)
        sys.exit(1)
    print("  [check] train_exact consistency OK")


def compute_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Add rank_train, best_train, best_val, zero_train flags. Return (df, best_train_idx, best_val_idx)."""
    sort_cols  = ["train_errors", "val_exact", "H", "LR", "TF", "run_id"]
    sort_asc   = [True, False, True, True, True, True]
    df_sorted  = df.sort_values(sort_cols, ascending=sort_asc).reset_index(drop=True)
    df_sorted["rank_train"] = range(1, len(df_sorted) + 1)

    best_train_idx = 0  # after sort_values, row 0 is best train

    sort_val_cols = ["val_exact", "train_errors", "H", "LR", "TF", "run_id"]
    sort_val_asc  = [False, True, True, True, True, True]
    val_order = df_sorted.sort_values(sort_val_cols, ascending=sort_val_asc).index
    best_val_idx  = int(val_order[0])

    df_sorted["best_train"] = False
    df_sorted["best_val"]   = False
    df_sorted.loc[best_train_idx, "best_train"] = True
    df_sorted.loc[best_val_idx,   "best_val"]   = True
    df_sorted["zero_train"] = df_sorted["train_errors"] == 0

    return df_sorted, best_train_idx, best_val_idx


def check_expected(df: pd.DataFrame, bt: int, bv: int, train_size: int) -> None:
    """Sanity-check computed values against scientifically expected values."""
    EPS = 1e-4
    errs = []
    zero_count = int(df["zero_train"].sum())

    if zero_count != 0:
        errs.append(f"zero_train_count: expected 0, got {zero_count}")

    r_bt = df.iloc[bt]
    if r_bt["H"] != 256:
        errs.append(f"best_train H: expected 256, got {r_bt['H']}")
    if abs(r_bt["TF"] - 1.0) > 0.01:
        errs.append(f"best_train TF: expected 1.0, got {r_bt['TF']}")
    if abs(r_bt["LR"] - 5e-4) > 1e-6:
        errs.append(f"best_train LR: expected 5e-4, got {r_bt['LR']}")
    if int(r_bt["train_errors"]) != 15:
        errs.append(f"best_train errors: expected 15, got {r_bt['train_errors']}")
    if abs(r_bt["train_exact"] - 0.999403) > EPS:
        errs.append(f"best_train exact: expected ≈0.999403, got {r_bt['train_exact']:.6f}")
    if abs(r_bt["val_exact"] - 0.989402) > EPS:
        errs.append(f"best_train val_exact: expected ≈0.989402, got {r_bt['val_exact']:.6f}")

    r_bv = df.iloc[bv]
    if r_bv["H"] != 128:
        errs.append(f"best_val H: expected 128, got {r_bv['H']}")
    if abs(r_bv["TF"] - 1.0) > 0.01:
        errs.append(f"best_val TF: expected 1.0, got {r_bv['TF']}")
    if abs(r_bv["LR"] - 5e-4) > 1e-6:
        errs.append(f"best_val LR: expected 5e-4, got {r_bv['LR']}")
    if abs(r_bv["val_exact"] - 0.991206) > EPS:
        errs.append(f"best_val val_exact: expected ≈0.991206, got {r_bv['val_exact']:.6f}")

    if errs:
        print("\nSANITY CHECK FAILED — computed values differ from expected:", file=sys.stderr)
        for e in errs:
            print(f"  ✗ {e}", file=sys.stderr)
        sys.exit(1)

    print("  [sanity] all expected values match ✓")
    print(f"  zero_train_count = {zero_count}")
    print(f"  best_train: {r_bt['run_id']}  errors={r_bt['train_errors']}  "
          f"train_exact={r_bt['train_exact']:.6f}  val_exact={r_bt['val_exact']:.6f}")
    print(f"  best_val:   {r_bv['run_id']}  val_exact={r_bv['val_exact']:.6f}")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _lr_label(lr: float) -> str:
    if abs(lr - 5e-4) < 1e-6:
        return "5e-4"
    if abs(lr - 1e-3) < 1e-6:
        return "1e-3"
    return f"{lr:.1e}"


def make_figure(
    df: pd.DataFrame,
    bt: int,
    bv: int,
    trajectory: list[tuple[int, int]],
    title: str,
    train_size: int,
    dpi: int,
) -> plt.Figure:
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(16, 9),
        gridspec_kw={"width_ratios": [1.7, 1.0]},
    )
    fig.patch.set_facecolor("white")
    # Ample top margin for title/subtitle; bottom margin for legend + footer
    fig.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.23, wspace=0.30)

    # --- Title + subtitle via fig.text (no dedicated axes) ---
    fig.text(0.5, 0.96, title,
             ha="center", va="top", fontsize=13, fontweight="bold")
    subtitle = ("Panel A: 18 configurations at 120 epochs | "
                "Panel B: best run continued with LR=1e-4 to 160/200 epochs  |  "
                "H: 64, 128, 256 | TF: 0.0, 0.2, 1.0 | LR: 5e-4, 1e-3 | dropout = 0.0 | seed = 0")
    fig.text(0.5, 0.91, subtitle,
             ha="center", va="top", fontsize=8.5, color="#555555")

    # --- Panels ---
    leg_handles, leg_labels = _draw_panel_a(ax_a, df, bt, bv, train_size)
    r_bt = df.iloc[bt]
    panel_b_title = "B. Best train run trajectory + low-LR continuation"
    panel_b_subtitle = f"H{int(r_bt['H'])}, TF={r_bt['TF']:.1f}, initial LR={_lr_label(r_bt['LR'])}, final LR=1e-4"
    _draw_panel_b(ax_b, trajectory, panel_b_title, panel_b_subtitle)

    # --- Unified legend row between panels and footer ---
    fig.legend(handles=leg_handles, labels=leg_labels,
               loc="lower center", bbox_to_anchor=(0.5, 0.10),
               ncol=len(leg_handles), fontsize=7.5, framealpha=0.95,
               columnspacing=1.2, handlelength=1.3, handletextpad=0.5,
               borderaxespad=0.2, edgecolor="#cccccc")

    # --- Footer: one short line, well below panels ---
    zero_count = int(df["zero_train"].sum())
    n = len(df)
    best_err = int(r_bt["train_errors"])
    footer = (
        f"Phase 6A: {zero_count}/{n} reached zero train errors at 120 epochs.  "
        f"Low-LR continuation reduced the best run {best_err}→3 errors at 160, then 4 at 200."
    )
    fig.text(0.5, 0.02, footer,
             ha="center", va="bottom", fontsize=8, color="#666666")

    return fig


def _draw_config_marker(ax: plt.Axes, row: "pd.Series", zorder: int, alpha: float = 1.0) -> None:
    """Redraw the standard TF/H/LR marker encoding for one data row."""
    h  = int(row["H"])
    tf = float(row["TF"])
    lr = float(row["LR"])
    x  = float(row["train_errors"])
    y  = float(row["val_exact"])
    c  = _TF_COLORS[tf]
    mk = _H_MARKERS[h]
    filled = abs(lr - 5e-4) < 1e-6
    if filled:
        ax.plot(x, y, marker=mk, color=c, markersize=_MS,
                linestyle="none", markeredgecolor=c,
                markeredgewidth=0.8, alpha=alpha, zorder=zorder)
    else:
        ax.plot(x, y, marker=mk, markersize=_MS_OPEN,
                linestyle="none", markerfacecolor="none",
                markeredgecolor=c, markeredgewidth=_LW_OPEN,
                alpha=alpha, zorder=zorder)


def _place_best_annotations(
    ax: plt.Axes,
    xt: float, yt: float, label_bt: str,
    xv: float, yv: float, label_bv: str,
) -> None:
    """Place Best train / Best val annotations with basic overlap avoidance.

    Uses offset-points textcoords so the arrow stays short and predictable.
    Iterates over candidate (dx, dy) offset pairs and picks the first combination
    where (a) the two boxes don't intersect and (b) neither box covers either
    highlighted data point.
    """
    _BT_CANDS = [
        (25,   -45, "left",  "top"),
        (-115, -45, "right", "top"),
        (25,    40, "left",  "bottom"),
        (-115,  40, "right", "bottom"),
    ]
    _BV_CANDS = [
        (30,    45, "left",  "bottom"),
        (-115,  45, "right", "bottom"),
        (30,   -55, "left",  "top"),
        (-115, -55, "right", "top"),
    ]

    def _make(label, xy, dx, dy, ha, va, ec):
        return ax.annotate(
            label, xy=xy, xytext=(dx, dy), textcoords="offset points",
            fontsize=7.5, ha=ha, va=va, zorder=6,
            arrowprops=dict(arrowstyle="->", color=ec, lw=0.9,
                            connectionstyle="arc3,rad=0.0"),
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=ec,
                      alpha=0.95, lw=0.8),
        )

    def _overlaps(b1, b2) -> bool:
        return not (b1.x1 < b2.x0 or b2.x1 < b1.x0 or b1.y1 < b2.y0 or b2.y1 < b1.y0)

    def _covers(bb, px, py) -> bool:
        return bb.x0 <= px <= bb.x1 and bb.y0 <= py <= bb.y1

    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ann_bt = ann_bv = None
    found = False

    for bt_c in _BT_CANDS:
        for bv_c in _BV_CANDS:
            if ann_bt is not None:
                ann_bt.remove()
            if ann_bv is not None:
                ann_bv.remove()
            ann_bt = _make(label_bt, (xt, yt), bt_c[0], bt_c[1], bt_c[2], bt_c[3], "#888888")
            ann_bv = _make(label_bv, (xv, yv), bv_c[0], bv_c[1], bv_c[2], bv_c[3], "#CC0000")
            fig.canvas.draw()
            bb_bt = ann_bt.get_window_extent(renderer)
            bb_bv = ann_bv.get_window_extent(renderer)
            pt_bt = ax.transData.transform((xt, yt))
            pt_bv = ax.transData.transform((xv, yv))
            if (_overlaps(bb_bt, bb_bv)
                    or _covers(bb_bt, pt_bv[0], pt_bv[1])
                    or _covers(bb_bv, pt_bt[0], pt_bt[1])
                    or _covers(bb_bt, pt_bt[0], pt_bt[1])
                    or _covers(bb_bv, pt_bv[0], pt_bv[1])):
                continue
            found = True
            break
        if found:
            break
    # If no perfect placement found, keep last-drawn pair as fallback


def _draw_panel_a(
    ax: plt.Axes, df: pd.DataFrame, bt: int, bv: int, train_size: int
) -> tuple[list, list]:
    """Draw Panel A and return (legend_handles, legend_labels) for the figure legend."""
    ax.set_title("A. Phase 6A gridsearch: all 18 configurations at 120 epochs",
                 loc="left", fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Train full-route errors", fontsize=9)
    ax.set_ylabel("Validation full-route exact match", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_facecolor("white")

    # Pre-compute axis limits so annotations stay inside
    x_range = float(df["train_errors"].max() - df["train_errors"].min())
    y_range = float(df["val_exact"].max()    - df["val_exact"].min())
    xmin = max(-2, float(df["train_errors"].min()) - x_range * 0.05)
    xmax = float(df["train_errors"].max()) + x_range * 0.12
    ymin = float(df["val_exact"].min()) - y_range * 0.15
    ymax = float(df["val_exact"].max()) + y_range * 0.40  # headroom for annotation
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Plot each point
    for _, row in df.iterrows():
        h  = int(row["H"])
        tf = float(row["TF"])
        lr = float(row["LR"])
        x  = float(row["train_errors"])
        y  = float(row["val_exact"])
        c  = _TF_COLORS[tf]
        mk = _H_MARKERS[h]
        filled = abs(lr - 5e-4) < 1e-6
        if filled:
            ax.plot(x, y, marker=mk, color=c, markersize=_MS,
                    linestyle="none", markeredgecolor=c,
                    markeredgewidth=0.8, alpha=0.85, zorder=3)
        else:
            ax.plot(x, y, marker=mk, markersize=_MS_OPEN,
                    linestyle="none", markerfacecolor="none",
                    markeredgecolor=c, markeredgewidth=_LW_OPEN, zorder=3)

    # Vertical goal line at x=0
    ax.axvline(0, color="#999999", linestyle="--", linewidth=0.9, zorder=1)
    ax.text(1, ymin + y_range * 0.05, "goal: 0 errors",
            fontsize=6.5, color="#999999", ha="left", va="bottom")

    # --- Highlight best_train ---
    # zorder strategy: base markers at 3 (loop above) | halos at 3 (drawn later = on top
    # within same zorder) | redrawn originals at 4 | badge at 5 | annotations at 6
    r_bt = df.iloc[bt]
    xt = float(r_bt["train_errors"])
    yt = float(r_bt["val_exact"])
    # Golden circle halo (open) — surrounds but does not cover the triangle
    ax.plot(xt, yt, marker="o", markersize=_MS + 9,
            linestyle="none", markerfacecolor="none",
            markeredgecolor="#DDAA00", markeredgewidth=2.2, zorder=3)
    # Redraw original TF/H/LR encoding on top so triangle remains fully visible
    _draw_config_marker(ax, r_bt, zorder=4)
    # Small ★ badge offset so it does not overlap the triangle
    ax.annotate("★", xy=(xt, yt), xytext=(-11, 9), textcoords="offset points",
                fontsize=8, color="#DDAA00", ha="center", va="center", zorder=5)

    # --- Highlight best_val ---
    r_bv = df.iloc[bv]
    xv = float(r_bv["train_errors"])
    yv = float(r_bv["val_exact"])
    # Red diamond halo (open) — surrounds but does not cover the square
    ax.plot(xv, yv, marker="D", markersize=_MS + 9,
            linestyle="none", markerfacecolor="none",
            markeredgecolor="#CC0000", markeredgewidth=2.2, zorder=3)
    # Redraw original TF/H/LR encoding on top so square remains fully visible
    _draw_config_marker(ax, r_bv, zorder=4)

    # --- Annotation labels (overlap-aware, offset points so arrows stay short) ---
    label_bt = (f"★ Best train\n"
                f"H{int(r_bt['H'])}, TF={r_bt['TF']:.1f}, LR={_lr_label(r_bt['LR'])}\n"
                f"{int(r_bt['train_errors'])} errors / {train_size:,}")
    label_bv = (f"◆ Best val\n"
                f"H{int(r_bv['H'])}, TF={r_bv['TF']:.1f}, LR={_lr_label(r_bv['LR'])}\n"
                f"val exact ≈ {r_bv['val_exact']:.4f}")
    _place_best_annotations(ax, xt, yt, label_bt, xv, yv, label_bv)

    # --- Build legend handles (returned for figure-level legend) ---
    handles: list = []
    labels:  list = []
    for tf_val in (0.0, 0.2, 1.0):
        handles.append(mpatches.Patch(color=_TF_COLORS[tf_val]))
        labels.append(f"TF = {tf_val:.1f}")
    for h_val, mk in _H_MARKERS.items():
        handles.append(mlines.Line2D([], [], marker=mk, color="#555555",
                                     markersize=7, linestyle="none"))
        labels.append(f"H = {h_val}")
    handles.append(mlines.Line2D([], [], marker="o", color="#555555",
                                 markersize=7, linestyle="none"))
    labels.append("LR = 5e-4 (filled)")
    handles.append(mlines.Line2D([], [], marker="o", color="#555555",
                                 markersize=7, linestyle="none",
                                 markerfacecolor="none", markeredgewidth=1.5))
    labels.append("LR = 1e-3 (open)")

    return handles, labels


def _draw_panel_b(
    ax: plt.Axes,
    trajectory: list[tuple[int, int]],
    title: str,
    subtitle: str = "",
) -> None:
    title_str = title + (f"\n{subtitle}" if subtitle else "")
    ax.set_title(title_str, loc="left", fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Total training epochs", fontsize=9)
    ax.set_ylabel("Train full-route errors", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_facecolor("white")

    epochs = [pt[0] for pt in trajectory]
    errors = [pt[1] for pt in trajectory]
    ep_span = epochs[-1] - epochs[0]

    # Draw a phase-boundary vertical line if there are 5 points (Phase 6A → 6B/C boundary at ep[2])
    if len(trajectory) >= 5:
        ep_boundary = trajectory[2][0]
        ax.axvline(ep_boundary, color="#dddddd", linestyle=":", linewidth=1.2, zorder=1)
        ax.text(ep_boundary + ep_span * 0.01, max(errors) * 1.42,
                "LR=1e-4", fontsize=6.5, color="#888888", ha="left", va="top")

    ax.plot(epochs, errors, color="#0072B2", linewidth=1.8, zorder=3)
    ax.plot(epochs, errors, marker="o", color="#0072B2", markersize=8,
            linestyle="none", zorder=4)

    # Value labels above each point
    for ep, err in trajectory:
        ax.annotate(str(err), xy=(ep, err), xytext=(0, 10),
                    textcoords="offset points", fontsize=8.5,
                    ha="center", va="bottom", color="#0072B2", fontweight="bold")

    # Horizontal zero-error reference
    ax.axhline(0, color="#999999", linestyle="--", linewidth=0.9, zorder=1)
    ax.text(epochs[0] + ep_span * 0.02, 0.4, "0 errors",
            fontsize=6.5, color="#999999", va="bottom")

    # Axis limits first so annotation coords are meaningful
    y_top = max(errors) * 1.55
    ax.set_ylim(-max(errors) * 0.06, y_top)
    ax.set_xlim(epochs[0] - ep_span * 0.10, epochs[-1] + ep_span * 0.10)
    ax.set_xticks(epochs)

    # Annotations — placed using axis-fraction offsets to stay robust
    if len(trajectory) >= 3:
        ep0, e0 = trajectory[0]
        ep1, e1 = trajectory[1]
        ep2, e2 = trajectory[2]

        # "local plateau" — between epoch 0 and epoch 1, above the midpoint
        ax.annotate("local plateau",
                    xy=((ep0 + ep1) / 2, (e0 + e1) / 2),
                    xytext=((ep0 + ep1) / 2 - ep_span * 0.12,
                            (e0 + e1) / 2 + max(errors) * 0.28),
                    fontsize=7.5, color="#555555", ha="center",
                    arrowprops=dict(arrowstyle="-|>", color="#aaaaaa", lw=0.8))

    if len(trajectory) >= 5:
        ep2, e2 = trajectory[2]
        ep3, e3 = trajectory[3]
        ep4, e4 = trajectory[4]

        # "low-LR helps" — between epochs 2 and 3 (120→160, the big drop)
        ax.annotate("low-LR helps",
                    xy=((ep2 + ep3) / 2, (e2 + e3) / 2),
                    xytext=((ep2 + ep3) / 2 + ep_span * 0.04,
                            (e2 + e3) / 2 + max(errors) * 0.35),
                    fontsize=7.5, color="#555555", ha="center",
                    arrowprops=dict(arrowstyle="-|>", color="#aaaaaa", lw=0.8))

        # "error permutation" — between epochs 3 and 4 (160→200, 3→4, slight uptick)
        ax.annotate("error\npermutation",
                    xy=((ep3 + ep4) / 2, (e3 + e4) / 2),
                    xytext=((ep3 + ep4) / 2 + ep_span * 0.04,
                            (e3 + e4) / 2 + max(errors) * 0.28),
                    fontsize=7.5, color="#555555", ha="center",
                    arrowprops=dict(arrowstyle="-|>", color="#aaaaaa", lw=0.8))


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

_CSV_COLS = [
    "rank_train", "run_id", "H", "TF", "LR", "dropout",
    "train_errors", "train_exact", "val_exact", "val_edit", "val_ned",
    "best_train", "best_val", "zero_train",
]

_MD_COLS = _CSV_COLS  # same order


def write_csv(df: pd.DataFrame, out_dir: str) -> str:
    available = [c for c in _CSV_COLS if c in df.columns]
    missing_opt = set(_CSV_COLS) - set(available)
    if missing_opt:
        print(f"  [csv] optional columns absent, omitting: {missing_opt}")
    path = os.path.join(out_dir, "bottomline_e120_clean.csv")
    df[available].to_csv(path, index=False)
    print(f"  -> {path}  ({len(df)} rows)")
    return path


def _fmt_lr(lr: float) -> str:
    return _lr_label(lr)


def write_markdown(df: pd.DataFrame, out_dir: str, bt: int, bv: int) -> str:
    available = [c for c in _MD_COLS if c in df.columns]
    path = os.path.join(out_dir, "bottomline_e120_table.md")

    r_bt = df.iloc[bt]
    r_bv = df.iloc[bv]
    zero_count = int(df["zero_train"].sum())

    lines = [
        "# Lichtheim3 Phase 6A — bottom-line at 120 epochs",
        "",
        f"**{len(df)} configurations** | "
        f"zero_train_count = {zero_count} | "
        f"best train: {r_bt['run_id']} ({int(r_bt['train_errors'])} errors) | "
        f"best validation: {r_bv['run_id']} (val_exact = {r_bv['val_exact']:.6f})",
        "",
    ]

    # Header
    headers = available[:]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for i, row in df.iterrows():
        cells = []
        prefix = ""
        is_bt  = (i == bt)
        is_bv  = (i == bv)
        if is_bt and is_bv:
            prefix = "**BEST TRAIN + BEST VAL** — "
        elif is_bt:
            prefix = "**BEST TRAIN** — "
        elif is_bv:
            prefix = "**BEST VAL** — "

        for col in headers:
            if col not in row.index:
                cells.append("")
                continue
            v = row[col]
            if col == "run_id":
                cells.append(f"{prefix}{v}")
            elif col == "LR":
                cells.append(_fmt_lr(float(v)))
            elif col == "TF":
                cells.append(f"{float(v):.1f}")
            elif col == "dropout":
                cells.append(f"{float(v):.1f}")
            elif col in ("train_exact", "val_exact"):
                cells.append(f"{float(v):.6f}")
            elif col in ("val_edit", "val_ned") and pd.notna(v):
                cells.append(f"{float(v):.4f}")
            elif col in ("best_train", "best_val", "zero_train"):
                cells.append("✓" if bool(v) else "")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  -> {path}")
    return path


def write_caption(
    df: pd.DataFrame, bt: int, bv: int, out_dir: str, train_size: int
) -> str:
    r_bt = df.iloc[bt]
    r_bv = df.iloc[bv]
    zero_count = int(df["zero_train"].sum())
    n = len(df)
    best_H   = int(r_bt["H"])
    best_TF  = r_bt["TF"]
    best_LR  = _lr_label(r_bt["LR"])
    best_err = int(r_bt["train_errors"])
    bv_H     = int(r_bv["H"])
    bv_TF    = r_bv["TF"]
    bv_LR    = _lr_label(r_bv["LR"])
    bv_exact = r_bv["val_exact"]

    discord = (
        f"Phase 6A+6B/C summary: {n} configurations were trained to 120 epochs (Phase 6A). "
        f"None reached zero train errors; the best model (H{best_H}, TF={best_TF:.1f}, LR={best_LR}) "
        f"gave {best_err} errors. Low-LR continuation (LR=1e-4) reduced that to 3 errors at 160 epochs, "
        f"then 4 at 200 epochs."
    )

    meeting = (
        f"Phase 6A gridsearch ({n} configurations, H ∈ {{64, 128, 256}}, "
        f"TF ∈ {{0.0, 0.2, 1.0}}, LR ∈ {{5e-4, 1e-3}}) ran to 120 epochs. "
        f"{zero_count}/{n} configurations reached the train ceiling. "
        f"The best train model (H{best_H}, TF={best_TF:.1f}, LR={best_LR}) achieves "
        f"{best_err}/{train_size:,} train errors (exact ≈ {r_bt['train_exact']:.4f}) "
        f"and val exact ≈ {r_bt['val_exact']:.4f}. "
        f"The best validation model (H{bv_H}, TF={bv_TF:.1f}, LR={bv_LR}) achieves "
        f"val exact ≈ {bv_exact:.6f}, exceeding the best-train model on val. "
        f"Error trajectory of the best train run: 43 → 44 → 15 errors at epochs 30, 60, 120 "
        f"(plateau through epoch 60, then sustained improvement). "
        f"Phase 6B/C continuation at LR=1e-4: {best_err} → 3 errors at 160 epochs, then 4 at 200 epochs. "
        f"The slight uptick from 3 to 4 at 200 epochs reflects error permutation, not regression. "
        f"Autoregressive and teacher-forced evaluation fail on the same residual items."
    )

    takeaways = f"""\
- Phase 6A (120 epochs): {zero_count}/{n} configurations reached zero train errors.
- Best train model: H{best_H}, TF={best_TF:.1f}, LR={best_LR} — {best_err} errors at 120 epochs.
- Best validation model: H{bv_H}, TF={bv_TF:.1f}, LR={bv_LR} — val exact ≈ {bv_exact:.4f}.
- Phase 6B/C (low-LR continuation at LR=1e-4): {best_err} → 3 errors at 160, then 4 at 200 epochs.
- The 3→4 uptick at 200 epochs is error permutation: different items fail, not more.
- H=64 appears limiting for train-set memorization in this grid.
"""

    content = (
        "## Discord caption\n\n" + discord + "\n\n"
        "## Meeting-notes caption\n\n" + meeting + "\n\n"
        "## Key takeaways\n\n" + takeaways
    )

    path = os.path.join(out_dir, "phase6a_phase6bc_caption.md")
    with open(path, "w") as f:
        f.write(content + "\n")
    print(f"  -> {path}")
    return path


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    print("\n=== NORMALIZED CONFIGURATION SUMMARY (ranked by train errors) ===")
    cols = ["rank_train", "run_id", "H", "TF", "LR", "train_errors",
            "train_exact", "val_exact", "best_train", "best_val"]
    avail = [c for c in cols if c in df.columns]
    for _, row in df.iterrows():
        parts = []
        for c in avail:
            v = row[c]
            if c == "LR":
                parts.append(f"LR={_fmt_lr(float(v))}")
            elif c == "TF":
                parts.append(f"TF={float(v):.1f}")
            elif c in ("train_exact", "val_exact"):
                parts.append(f"{c}={float(v):.6f}")
            elif c in ("best_train", "best_val"):
                parts.append(f"{c}={'✓' if bool(v) else '·'}")
            else:
                parts.append(f"{c}={v}")
        print("  " + "  ".join(parts))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    traj   = parse_trajectory(args.trajectory)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load
    df = load_and_prepare(args.tsv, args.train_size)

    # 2. Validate grid
    validate_grid(df)

    # 3. Consistency check
    check_consistency(df, args.train_size)

    # 4. Compute flags / ranking
    df, bt, bv = compute_flags(df)

    # 5. Sanity check against expected values
    check_expected(df, bt, bv, args.train_size)

    # 6. Figure
    print(f"\n[trajectory] parsed points: {traj}")
    print("\n[figure] generating...")
    fig = make_figure(df, bt, bv, traj, args.title, args.train_size, args.dpi)

    stem = os.path.join(out_dir, "phase6a_phase6bc_bottomline")
    for ext, kw in [
        (".png", dict(dpi=args.dpi, bbox_inches="tight", facecolor="white")),
        (".pdf", dict(bbox_inches="tight", facecolor="white")),
        (".svg", dict(bbox_inches="tight", facecolor="white")),
    ]:
        path = stem + ext
        fig.savefig(path, **kw)
        print(f"[figure] wrote hybrid Phase6A+Phase6B/C figure: {path}")

    if args.show:
        plt.show()
    plt.close(fig)

    # 7. CSV
    print("\n[csv]")
    write_csv(df, out_dir)

    # 8. Markdown table
    print("\n[markdown]")
    write_markdown(df, out_dir, bt, bv)

    # 9. Caption
    print("\n[caption]")
    write_caption(df, bt, bv, out_dir, args.train_size)

    # 10. Console summary
    print_summary(df)

    # 11. Final validation
    print("=== OUTPUT FILE CHECK ===")
    expected_files = [
        "phase6a_phase6bc_bottomline.png",
        "phase6a_phase6bc_bottomline.pdf",
        "phase6a_phase6bc_bottomline.svg",
        "bottomline_e120_clean.csv",
        "bottomline_e120_table.md",
        "phase6a_phase6bc_caption.md",
    ]
    all_ok = True
    for fname in expected_files:
        fpath = os.path.join(out_dir, fname)
        exists = os.path.exists(fpath)
        size   = os.path.getsize(fpath) if exists else 0
        status = "✓" if (exists and size > 0) else "✗ MISSING or EMPTY"
        print(f"  {status}  {fpath}")
        if not (exists and size > 0):
            all_ok = False

    if not all_ok:
        print("ERROR: some output files are missing or empty.", file=sys.stderr)
        sys.exit(1)

    print(f"\n[done] all outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
