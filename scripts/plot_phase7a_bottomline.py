#!/usr/bin/env python3
"""Plot Phase 7A bottom-line: low-LR continuation from e120 to e160 at LR=1e-4.

Outputs (all in --out-dir):
  phase7a_bottomline.png / .pdf / .svg
  phase7a_caption.md

Usage:
  python scripts/plot_phase7a_bottomline.py \\
      --all-tsv outputs/phase7a_all_e120_to_e160_lowlr/_control/phase7a_all_checkpoints.tsv \\
      --best-tsv outputs/phase7a_all_e120_to_e160_lowlr/_control/phase7a_best_checkpoint_per_run.tsv \\
      --confirm-tsv outputs/phase7a_all_e120_to_e160_lowlr/_control/phase7a_zero_confirmation.tsv \\
      --out-dir outputs/phase7a_all_e120_to_e160_lowlr/figures
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Visual constants — same palette as Phase 6A for consistency
# ---------------------------------------------------------------------------
_TF_COLORS = {0.0: "#0072B2", 0.2: "#E69F00", 1.0: "#009E73"}
_H_MARKERS  = {64: "o", 128: "s", 256: "^"}
_MS         = 10
_MS_OPEN    = 10
_LW_OPEN    = 1.8

# Zero-confirmed runs — authoritative source: zero_confirmation.tsv only
_ZERO_RUN_IDS = frozenset({
    "p6a_e120_H064_tf0p0_lr1e-3_s0",
    "p6a_e120_H256_tf1p0_lr1e-3_s0",
})
_BEST_CEILING_RUN   = "p6a_e120_H064_tf0p0_lr1e-3_s0"
_BEST_CEILING_EPOCH = 160

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 7A bottom-line figure")
    p.add_argument("--all-tsv",     required=True, dest="all_tsv",
                   help="phase7a_all_checkpoints.tsv")
    p.add_argument("--best-tsv",    required=True, dest="best_tsv",
                   help="phase7a_best_checkpoint_per_run.tsv")
    p.add_argument("--confirm-tsv", required=True, dest="confirm_tsv",
                   help="phase7a_zero_confirmation.tsv (authoritative zero status)")
    p.add_argument("--out-dir",     required=True)
    p.add_argument("--dpi",         type=int, default=300)
    p.add_argument("--show",        action="store_true")
    p.add_argument("--linear",      action="store_true",
                   help="Use linear y-axis in Panel A instead of log(1+n)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_lr(val) -> float:
    return float(str(val).strip())


def load_all_checkpoints(path: str) -> pd.DataFrame:
    print(f"  [load] {path}")
    df = pd.read_csv(path, sep="\t")
    required = ["source_run_id", "H", "TF", "initial_lr", "epoch",
                "train_errors_ar", "val_exact_ar"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: all_checkpoints missing columns: {missing}", file=sys.stderr)
        sys.exit(1)
    df["H"]               = pd.to_numeric(df["H"],               errors="coerce").astype(int)
    df["TF"]              = pd.to_numeric(df["TF"],              errors="coerce").astype(float)
    df["initial_lr"]      = df["initial_lr"].apply(_parse_lr)
    df["epoch"]           = pd.to_numeric(df["epoch"],           errors="coerce").astype(int)
    df["train_errors_ar"] = pd.to_numeric(df["train_errors_ar"], errors="coerce").astype(int)
    df["val_exact_ar"]    = pd.to_numeric(df["val_exact_ar"],    errors="coerce").astype(float)
    n_runs = df["source_run_id"].nunique()
    epochs = sorted(df["epoch"].unique())
    print(f"         {len(df)} rows  |  {n_runs} runs  |  epochs {epochs}")
    return df


def load_best_per_run(path: str) -> pd.DataFrame:
    print(f"  [load] {path}")
    df = pd.read_csv(path, sep="\t")
    required = ["source_run_id", "source_train_errors_e120",
                "best_train_errors_ar", "best_val_exact_ar"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: best_per_run missing columns: {missing}", file=sys.stderr)
        sys.exit(1)
    df["source_train_errors_e120"] = pd.to_numeric(
        df["source_train_errors_e120"], errors="coerce").astype(int)
    df["best_train_errors_ar"]     = pd.to_numeric(
        df["best_train_errors_ar"], errors="coerce").astype(int)
    df["best_val_exact_ar"]        = pd.to_numeric(
        df["best_val_exact_ar"], errors="coerce").astype(float)
    if "H" in df.columns:
        df["H"] = pd.to_numeric(df["H"], errors="coerce").astype(int)
    if "TF" in df.columns:
        df["TF"] = pd.to_numeric(df["TF"], errors="coerce").astype(float)
    if "initial_lr" in df.columns:
        df["initial_lr"] = df["initial_lr"].apply(_parse_lr)
    print(f"         {len(df)} rows")
    return df


def load_zero_confirmation(path: str) -> pd.DataFrame:
    print(f"  [load] {path}")
    df = pd.read_csv(path, sep="\t")
    required = ["source_run_id", "zero_confirmed_ar"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: zero_confirmation missing columns: {missing}", file=sys.stderr)
        sys.exit(1)
    if df["zero_confirmed_ar"].dtype == object:
        df["zero_confirmed_ar"] = (
            df["zero_confirmed_ar"].str.strip().str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
            .astype(bool)
        )
    n_confirmed = int(df["zero_confirmed_ar"].sum())
    print(f"         {len(df)} rows  |  {n_confirmed} zero_confirmed_ar=true")
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(df_all: pd.DataFrame, df_best: pd.DataFrame,
             df_confirm: pd.DataFrame) -> None:
    errs = []
    n_all        = len(df_all)
    n_runs_all   = df_all["source_run_id"].nunique()
    epochs_found = set(df_all["epoch"].unique())
    n_best       = len(df_best)
    n_confirm    = len(df_confirm)
    n_conf_true  = int(df_confirm["zero_confirmed_ar"].sum())

    if n_all != 72:
        errs.append(f"all_checkpoints: expected 72 rows, got {n_all}")
    if n_runs_all != 18:
        errs.append(f"all_checkpoints: expected 18 source_run_id, got {n_runs_all}")
    if epochs_found != {130, 140, 150, 160}:
        errs.append(f"all_checkpoints: epochs {epochs_found} ≠ {{130,140,150,160}}")
    if n_best != 18:
        errs.append(f"best_per_run: expected 18 rows, got {n_best}")
    if n_confirm != 2:
        errs.append(f"zero_confirmation: expected 2 rows, got {n_confirm}")
    if n_conf_true != 2:
        errs.append(f"zero_confirmation: expected 2 true, got {n_conf_true}")

    if errs:
        print("[validate] FAILED:", file=sys.stderr)
        for e in errs:
            print(f"  ✗ {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[validate] OK — "
          f"all_checkpoints: {n_all} rows / {n_runs_all} runs / epochs {sorted(epochs_found)}  |  "
          f"best_per_run: {n_best} rows  |  "
          f"zero_confirmation: {n_confirm} rows / {n_conf_true} confirmed")


# ---------------------------------------------------------------------------
# Build trajectories for Panel A
# ---------------------------------------------------------------------------

def build_trajectories(df_all: pd.DataFrame,
                       df_best: pd.DataFrame) -> dict[str, tuple[list, list]]:
    """e120 from best_per_run; e130–160 from all_checkpoints."""
    traj: dict[str, tuple[list, list]] = {}
    for _, row in df_best.iterrows():
        rid  = row["source_run_id"]
        e120 = int(row["source_train_errors_e120"])
        sub  = (df_all[df_all["source_run_id"] == rid]
                .sort_values("epoch")[["epoch", "train_errors_ar"]])
        epochs = [120] + list(sub["epoch"].astype(int))
        errors = [e120] + list(sub["train_errors_ar"].astype(int))
        traj[rid] = (epochs, errors)
    return traj


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _lr_label(lr: float) -> str:
    if abs(lr - 5e-4) < 1e-6: return "5e-4"
    if abs(lr - 1e-3) < 1e-6: return "1e-3"
    return f"{lr:.1e}"


# ---------------------------------------------------------------------------
# Panel B marker helper
# ---------------------------------------------------------------------------

def _draw_marker_b(ax: plt.Axes, row: "pd.Series", zorder: int,
                   alpha: float = 1.0) -> None:
    h  = int(row["H"])
    tf = float(row["TF"])
    lr = float(row["initial_lr"])
    x  = float(row["best_train_errors_ar"])
    y  = float(row["best_val_exact_ar"])
    c  = _TF_COLORS.get(tf, "#888888")
    mk = _H_MARKERS.get(h, "o")
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


# ---------------------------------------------------------------------------
# Zero annotation placement for Panel B (overlap-aware)
# ---------------------------------------------------------------------------

def _place_zero_annotations_b(ax: plt.Axes,
                                pts: list[tuple[float, float, str, str]]) -> None:
    """Place two zero-confirmed annotations; pts sorted by y descending."""
    CANDS = [
        [(25,  38, "left",  "bottom"), (25,  -55, "left",  "top")],
        [(25,  50, "left",  "bottom"), (25,  -65, "left",  "top")],
        [(-130, 38, "right", "bottom"), (-130, -55, "right", "top")],
        [(25,  50, "left",  "bottom"), (-130, -55, "right", "top")],
        [(-130, 50, "right", "bottom"), (25,  -65, "left",  "top")],
    ]

    def _ann(label, xy, dx, dy, ha, va, ec):
        return ax.annotate(
            label, xy=xy, xytext=(dx, dy), textcoords="offset points",
            fontsize=7.5, ha=ha, va=va, zorder=6,
            arrowprops=dict(arrowstyle="->", color=ec, lw=0.9,
                            connectionstyle="arc3,rad=0.0"),
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=ec,
                      alpha=0.95, lw=0.8),
        )

    def _overlaps(b1, b2) -> bool:
        return not (b1.x1 < b2.x0 or b2.x1 < b1.x0 or
                    b1.y1 < b2.y0 or b2.y1 < b1.y0)

    x0, y0, lbl0, ec0 = pts[0]
    x1, y1, lbl1, ec1 = pts[1]

    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    a0 = a1 = None
    found = False

    for c0, c1 in CANDS:
        if a0 is not None: a0.remove()
        if a1 is not None: a1.remove()
        a0 = _ann(lbl0, (x0, y0), c0[0], c0[1], c0[2], c0[3], ec0)
        a1 = _ann(lbl1, (x1, y1), c1[0], c1[1], c1[2], c1[3], ec1)
        fig.canvas.draw()
        bb0 = a0.get_window_extent(renderer)
        bb1 = a1.get_window_extent(renderer)
        if not _overlaps(bb0, bb1):
            found = True
            break
    # Keep last pair on fallback if no perfect placement
    if not found:
        print("  [annotate] fallback placement used for Panel B zero annotations")


# ---------------------------------------------------------------------------
# Panel A: trajectories (log1p y-axis)
# ---------------------------------------------------------------------------

def _draw_panel_a(ax: plt.Axes,
                  trajectories: dict[str, tuple[list, list]],
                  zero_confirmed_set: set,
                  df_best: pd.DataFrame,
                  linear: bool = False) -> tuple[list, list]:
    ax.set_title("A. Low-LR continuation trajectories",
                 loc="left", fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Total training epochs", fontsize=9)
    if linear:
        ax.set_ylabel("Train full-route AR errors", fontsize=9)
    else:
        ax.set_ylabel("Train full-route AR errors\n(log1p scale; ticks show raw error counts)",
                      fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_facecolor("white")

    all_errors = [e for (_, errs) in trajectories.values() for e in errs]
    max_err = max(all_errors)

    if linear:
        step = 10
        n_steps = int(np.ceil(max_err / step))
        tick_lin = list(range(0, (n_steps + 1) * step + 1, step))
        ax.set_yticks(tick_lin)
        ax.set_yticklabels([str(v) for v in tick_lin])
        ymax = n_steps * step * 1.22
    else:
        # log1p y-axis: tick positions at log(1+raw), labels show raw counts
        tick_raw = [0, 1, 2, 4, 8, 16, 32, 64]
        tick_raw = [v for v in tick_raw if v <= max_err * 1.15]
        tick_raw = sorted(set(tick_raw))
        ax.set_yticks([np.log1p(v) for v in tick_raw])
        ax.set_yticklabels([str(v) for v in tick_raw])
        ymax = np.log1p(max_err) * 1.22
    ax.set_ylim(-ymax * 0.04, ymax)
    ax.set_xlim(115, 175)
    ax.set_xticks([120, 130, 140, 150, 160])

    # Horizontal zero reference
    ax.axhline(0, color="#999999", linestyle="--", linewidth=0.9, zorder=1)
    ax.text(116.2, ymax * 0.012, "0 errors",
            fontsize=6.5, color="#999999", va="bottom")

    # Non-ceiling runs — gray, thin, semi-transparent
    for rid, (epochs, errors) in trajectories.items():
        if rid not in zero_confirmed_set:
            y_vals = errors if linear else list(np.log1p(errors))
            ax.plot(epochs, y_vals,
                    color="#cccccc", linewidth=1.0, alpha=0.5,
                    zorder=1, solid_capstyle="round")

    # Zero-confirmed runs — highlighted, colored by TF
    best_lookup = df_best.set_index("source_run_id")
    for rid in sorted(zero_confirmed_set):
        if rid not in trajectories:
            continue
        epochs, errors = trajectories[rid]
        row   = best_lookup.loc[rid]
        tf    = float(row["TF"])
        color = _TF_COLORS.get(tf, "#555555")
        y_vals = errors if linear else list(np.log1p(errors))
        ax.plot(epochs, y_vals, color=color, linewidth=2.3,
                alpha=1.0, zorder=3, solid_capstyle="round")
        ax.plot(epochs, y_vals, marker="o", color=color,
                markersize=5, linestyle="none", zorder=4)

    # Boxed annotations for ceiling runs — axes-fraction placement, right side
    _CEIL_META: dict[str, tuple] = {
        "p6a_e120_H064_tf0p0_lr1e-3_s0": (0.97, 0.52, "right", "bottom"),
        "p6a_e120_H256_tf1p0_lr1e-3_s0": (0.97, 0.24, "right", "bottom"),
    }
    for rid, (af_x, af_y, ha, va) in _CEIL_META.items():
        if rid not in trajectories:
            continue
        _, errors_run = trajectories[rid]
        row   = best_lookup.loc[rid]
        h     = int(row["H"])
        tf    = float(row["TF"])
        lr    = _lr_label(float(row["initial_lr"]))
        color = _TF_COLORS.get(tf, "#555555")
        traj_str = " → ".join(str(e) for e in errors_run)
        text = f"H{h}, TF={tf:.1f}, LR₀={lr}\n{traj_str}"
        ax.annotate(
            text,
            xy=(160, 0), xycoords="data",
            xytext=(af_x, af_y), textcoords="axes fraction",
            fontsize=7.5, ha=ha, va=va, zorder=6,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.0,
                            connectionstyle="arc3,rad=0.0"),
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=color,
                      alpha=0.95, lw=1.0),
        )

    # Legend entries for Panel A
    handles = [mlines.Line2D([], [], color="#cccccc", linewidth=2.0)]
    labels  = ["Other runs"]
    return handles, labels


# ---------------------------------------------------------------------------
# Panel B: best checkpoint scatter
# ---------------------------------------------------------------------------

def _draw_panel_b(ax: plt.Axes,
                  df_best: pd.DataFrame,
                  zero_confirmed_set: set) -> tuple[list, list]:
    ax.set_title("B. Best checkpoint observed per run",
                 loc="left", fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Best train AR errors", fontsize=9)
    ax.set_ylabel("Best val AR exact match", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_facecolor("white")

    x_all = df_best["best_train_errors_ar"].astype(float)
    y_all = df_best["best_val_exact_ar"].astype(float)
    x_rng = float(x_all.max() - x_all.min()) or 1.0
    y_rng = float(y_all.max() - y_all.min()) or 1e-3
    ax.set_xlim(-x_rng * 0.10, float(x_all.max()) + x_rng * 0.25)
    ax.set_ylim(float(y_all.min()) - y_rng * 0.15,
                float(y_all.max()) + y_rng * 0.60)

    # Goal line
    ax.axvline(0, color="#999999", linestyle="--", linewidth=0.9, zorder=1)
    ax.text(0.10, float(y_all.min()) - y_rng * 0.12, "0 errors",
            fontsize=6.5, color="#999999", ha="left", va="bottom")

    # All 18 runs at zorder=2
    for _, row in df_best.iterrows():
        _draw_marker_b(ax, row, zorder=2, alpha=0.85)

    # Golden halo + redrawn marker for zero-confirmed runs
    for _, row in df_best.iterrows():
        rid = row["source_run_id"]
        if rid not in zero_confirmed_set:
            continue
        x = float(row["best_train_errors_ar"])
        y = float(row["best_val_exact_ar"])
        ax.plot(x, y, marker="o", markersize=_MS + 9,
                linestyle="none", markerfacecolor="none",
                markeredgecolor="#DDAA00", markeredgewidth=2.2, zorder=3)
        _draw_marker_b(ax, row, zorder=4)

    # Annotations for zero-confirmed runs
    zero_pts: list[tuple[float, float, str, str]] = []
    for _, row in df_best.iterrows():
        rid = row["source_run_id"]
        if rid not in zero_confirmed_set:
            continue
        x  = float(row["best_train_errors_ar"])
        y  = float(row["best_val_exact_ar"])
        h  = int(row["H"])
        tf = float(row["TF"])
        lr = _lr_label(float(row["initial_lr"]))
        ec = _TF_COLORS.get(tf, "#555555")
        if rid == _BEST_CEILING_RUN:
            lbl = (f"★ Best confirmed zero\n"
                   f"H{h}, TF={tf:.1f}, LR₀={lr}\n"
                   f"0 train AR errors\n"
                   f"val={y:.4f}")
        else:
            lbl = (f"H{h}, TF={tf:.1f}, LR₀={lr}\n"
                   f"0 train AR errors\n"
                   f"val={y:.4f}")
        zero_pts.append((x, y, lbl, ec))

    zero_pts.sort(key=lambda t: -t[1])  # top first
    if len(zero_pts) >= 2:
        _place_zero_annotations_b(ax, zero_pts[:2])
    elif len(zero_pts) == 1:
        x, y, lbl, ec = zero_pts[0]
        ax.annotate(lbl, xy=(x, y), xytext=(25, 15),
                    textcoords="offset points", fontsize=7.5,
                    ha="left", va="bottom", zorder=6,
                    arrowprops=dict(arrowstyle="->", color=ec, lw=0.9,
                                    connectionstyle="arc3,rad=0.0"),
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec=ec, alpha=0.95, lw=0.8))

    # Legend handles (TF / H / LR)
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
    labels.append("LR₀ = 5e-4 (filled)")
    handles.append(mlines.Line2D([], [], marker="o", color="#555555",
                                 markersize=7, linestyle="none",
                                 markerfacecolor="none", markeredgewidth=1.5))
    labels.append("LR₀ = 1e-3 (open)")
    return handles, labels


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def make_figure(trajectories: dict,
                df_best: pd.DataFrame,
                zero_confirmed_set: set,
                dpi: int,
                linear: bool = False) -> plt.Figure:
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(16, 9),
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.23, wspace=0.35)

    fig.text(0.5, 0.96,
             "Lichtheim3 — Phase 7A low-LR continuation recovers the train ceiling",
             ha="center", va="top", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.91,
             ("18 Phase 6A checkpoints continued from e120 to e160 at LR=1e-4  |  "
              "deterministic autoregressive evaluation  |  seed 0"),
             ha="center", va="top", fontsize=8.5, color="#555555")

    ha_handles, ha_labels = _draw_panel_a(ax_a, trajectories, zero_confirmed_set, df_best,
                                           linear=linear)
    hb_handles, hb_labels = _draw_panel_b(ax_b, df_best, zero_confirmed_set)

    all_handles = ha_handles + hb_handles
    all_labels  = ha_labels  + hb_labels
    fig.legend(handles=all_handles, labels=all_labels,
               loc="lower center", bbox_to_anchor=(0.5, 0.10),
               ncol=5, fontsize=7.5, framealpha=0.95,
               columnspacing=1.1, handlelength=1.3, handletextpad=0.5,
               borderaxespad=0.2, edgecolor="#cccccc")

    n_zero = len(zero_confirmed_set)
    fig.text(0.5, 0.02,
             (f"{n_zero}/18 runs reached 0 train AR errors at e160; "
              "both zeros were reproduced exactly in a second deterministic AR evaluation."),
             ha="center", va="bottom", fontsize=8, color="#666666")

    return fig


# ---------------------------------------------------------------------------
# Caption
# ---------------------------------------------------------------------------

def write_caption(df_best: pd.DataFrame,
                  zero_confirmed_set: set,
                  out_dir: str) -> str:
    n_runs = len(df_best)
    n_zero = len(zero_confirmed_set)

    zero_rows = df_best[df_best["source_run_id"].isin(zero_confirmed_set)].copy()
    best_row  = df_best[df_best["source_run_id"] == _BEST_CEILING_RUN]

    if not best_row.empty:
        r_best    = best_row.iloc[0]
        best_val  = float(r_best["best_val_exact_ar"])
        best_h    = int(r_best["H"])
        best_tf   = float(r_best["TF"])
        best_lr   = _lr_label(float(r_best["initial_lr"]))
    else:
        best_val, best_h, best_tf, best_lr = 0.993687, 64, 0.0, "1e-3"

    other = zero_rows[zero_rows["source_run_id"] != _BEST_CEILING_RUN]
    if not other.empty:
        r2 = other.iloc[0]
        r2_h, r2_tf = int(r2["H"]), float(r2["TF"])
        r2_lr, r2_val = _lr_label(float(r2["initial_lr"])), float(r2["best_val_exact_ar"])
    else:
        r2_h, r2_tf, r2_lr, r2_val = 256, 1.0, "1e-3", 0.990981

    # Best validation run (may differ from zero runs)
    bvi = df_best["best_val_exact_ar"].idxmax()
    bvr = df_best.loc[bvi]
    bv_h   = int(bvr["H"])
    bv_tf  = float(bvr["TF"])
    bv_lr  = _lr_label(float(bvr["initial_lr"]))
    bv_val = float(bvr["best_val_exact_ar"])
    bv_err = int(bvr["best_train_errors_ar"])

    discord = (
        f"Phase 7A: 18 Phase 6A checkpoints continued from e120 to e160 at LR=1e-4. "
        f"{n_zero}/18 runs reached 0 train AR errors at e160. "
        f"Best ceiling: H{best_h}, TF={best_tf:.1f}, LR₀={best_lr}, "
        f"val AR = {best_val:.4f}. "
        f"Both zero results confirmed in a second deterministic evaluation."
    )

    meeting = f"""\
Phase 7A ran {n_runs} low-LR continuation experiments (LR=1e-4) from Phase 6A \
epoch-120 checkpoints to epoch 160. All evaluations used deterministic autoregressive \
decoding; no teacher-forced results are included. \
Panel A uses a log1p transformation to make differences near zero visible; \
tick labels report the corresponding raw numbers of train AR errors.

**Train ceiling confirmed: {n_zero}/{n_runs} runs.**
- H{best_h}, TF={best_tf:.1f}, LR₀={best_lr}: 0 train AR errors at e160, \
val AR = {best_val:.4f}.
- H{r2_h}, TF={r2_tf:.1f}, LR₀={r2_lr}: 0 train AR errors at e160, \
val AR = {r2_val:.4f}.
Both results were verified in an independent identical-condition evaluation \
(same_error_set = true in both cases).

**Best validation (not at zero train errors):** \
H{bv_h}, TF={bv_tf:.1f}, LR₀={bv_lr} — \
{bv_err} train AR errors, val AR = {bv_val:.4f}.

**Caveats:**
- *Single seed:* All runs use seed 0. No multi-seed screening performed; \
results may not generalise to other seeds.
- *Continuation, not replication:* These are warm-starts from Phase 6A checkpoints, \
not independent retraining from random initialisation. \
There is no evidence the same configurations reach the ceiling if retrained from scratch.
- *No post-e160 data:* Evaluations stop at e160. \
Stability of the ceiling beyond epoch 160 is unknown.
- *Non-monotone trajectories:* Several runs show temporary error increases between \
e120 and e160 (e.g., H256, TF=1.0, LR₀=1e-3 goes 55→21926→322→0). \
Low-LR training does not guarantee monotone convergence.
"""

    content = "## Discord caption\n\n" + discord + "\n\n## Meeting-notes caption\n\n" + meeting

    path = os.path.join(out_dir, "phase7a_caption.md")
    with open(path, "w") as f:
        f.write(content + "\n")
    print(f"[caption] -> {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n[load]")
    df_all     = load_all_checkpoints(args.all_tsv)
    df_best    = load_best_per_run(args.best_tsv)
    df_confirm = load_zero_confirmation(args.confirm_tsv)

    validate(df_all, df_best, df_confirm)

    # Zero-confirmed set is read exclusively from zero_confirmation.tsv
    zero_confirmed_set: set[str] = set(
        df_confirm.loc[df_confirm["zero_confirmed_ar"], "source_run_id"]
    )
    print(f"[confirm] zero-confirmed runs: {sorted(zero_confirmed_set)}")

    print("\n[trajectories] building (e120 from best_per_run + e130–160 from all_checkpoints)...")
    trajectories = build_trajectories(df_all, df_best)
    print(f"  {len(trajectories)} trajectories, {len(list(trajectories.values())[0][0])} points each")

    print("\n[figure] generating...")
    fig = make_figure(trajectories, df_best, zero_confirmed_set, args.dpi,
                      linear=args.linear)

    stem_name = "phase7a_bottomline_linear" if args.linear else "phase7a_bottomline_log1p"
    stem = os.path.join(args.out_dir, stem_name)
    for ext, kw in [
        (".png", dict(dpi=args.dpi, bbox_inches="tight", facecolor="white")),
        (".pdf", dict(bbox_inches="tight", facecolor="white")),
        (".svg", dict(bbox_inches="tight", facecolor="white")),
    ]:
        path = stem + ext
        fig.savefig(path, **kw)
        print(f"[figure] wrote {path}")

    if args.show:
        plt.show()
    plt.close(fig)

    if not args.linear:
        write_caption(df_best, zero_confirmed_set, args.out_dir)

    print("\n=== OUTPUT FILE CHECK ===")
    expected = [f"{stem_name}.png", f"{stem_name}.pdf", f"{stem_name}.svg"]
    if not args.linear:
        expected.append("phase7a_caption.md")
    all_ok = True
    for fname in expected:
        fpath = os.path.join(args.out_dir, fname)
        exists = os.path.exists(fpath)
        size   = os.path.getsize(fpath) if exists else 0
        ok     = exists and size > 0
        print(f"  {'✓' if ok else '✗ MISSING or EMPTY'}  {fpath}")
        if not ok:
            all_ok = False
    if not all_ok:
        print("ERROR: some output files are missing or empty.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
