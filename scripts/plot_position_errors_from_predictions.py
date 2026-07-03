"""Serial-position error curves from a pre-computed predictions TSV.

Reads the item-level predictions TSV produced by external_eval.py (or any
TSV that has `{route}_predicted` and `{route}_target` columns), computes
per-phoneme correctness at each serial position, and plots accuracy vs.
relative serial position.

Unlike plot_position_errors.py (which loads the checkpoint and re-runs the
model on the training lexicon), this script reads directly from the TSV so
it works for WFE / SSP external stimuli without re-loading the model.

Supported routes: full, wm, ltm (whichever columns exist in the TSV).

Outputs (default WFE run):
    outputs/external_eval_30k/figures/position_error_curve_wfe.png
    outputs/external_eval_30k/figures/position_error_curve_wfe_pseudowords.png
    outputs/external_eval_30k/figures/position_error_data.tsv

Usage:
    python scripts/plot_position_errors_from_predictions.py
    python scripts/plot_position_errors_from_predictions.py \\
        --pred outputs/external_eval_30k/wfe/item_level_predictions.tsv \\
        --out_dir outputs/external_eval_30k/figures \\
        --label wfe
    python scripts/plot_position_errors_from_predictions.py \\
        --pred outputs/external_eval_30k/ssp/item_level_predictions.tsv \\
        --label ssp
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRED_DEFAULT    = os.path.join(ROOT, "outputs", "external_eval_30k",
                                "wfe", "item_level_predictions.tsv")
OUT_DEFAULT     = os.path.join(ROOT, "outputs", "external_eval_30k", "figures")
LABEL_DEFAULT   = "wfe"
ROUTES          = ("full", "wm", "ltm")

_ROUTE_COLORS = {"full": "#2b7bba", "wm": "#e05a2b", "ltm": "#2ba34b"}
_ROUTE_LABELS = {"full": "Full (gated)", "wm": "WM (dorsal)", "ltm": "LTM (ventral)"}

NOTE = "Teacher-forced decoding · WM noise disabled (deterministic)"


# ---------------------------------------------------------------------------
# Per-position data extraction
# ---------------------------------------------------------------------------

def _sequences_from_row(row: pd.Series, route: str):
    """Return (predicted_phonemes, target_phonemes) for one row and route."""
    pred_col = f"{route}_predicted"
    tgt_col  = f"{route}_target"
    if pred_col not in row.index or tgt_col not in row.index:
        return None, None
    pred_str = str(row[pred_col]) if pd.notna(row[pred_col]) else ""
    tgt_str  = str(row[tgt_col])  if pd.notna(row[tgt_col])  else ""
    pred = pred_str.split() if pred_str else []
    tgt  = tgt_str.split()  if tgt_str  else []
    return pred, tgt


def build_position_df(df_pred: pd.DataFrame, routes: List[str]) -> pd.DataFrame:
    """Expand each prediction row to one record per (item, position, route)."""
    rows: List[dict] = []

    for item_idx, row in df_pred.iterrows():
        for route in routes:
            pred, tgt = _sequences_from_row(row, route)
            if not tgt:
                continue
            n = len(tgt)
            if n < 2:
                continue
            for pos in range(n):
                rel_pos = pos / (n - 1)
                is_correct = 0
                if pos < len(pred):
                    is_correct = int(pred[pos] == tgt[pos])
                rows.append({
                    "item_idx":    item_idx,
                    "word":        str(row.get("word", "")),
                    "condition":   str(row.get("condition", "")),
                    "lexicality":  str(row.get("lexicality", "")),
                    "route":       route,
                    "length":      n,
                    "position":    pos,
                    "rel_pos":     round(rel_pos, 4),
                    "is_correct":  is_correct,
                })

    return pd.DataFrame(rows)


def bin_position_df(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    bin_edges  = np.linspace(0.0, 1.0, n_bins + 1)
    bin_labels = [f"{bin_edges[i]:.2f}" for i in range(n_bins)]
    df = df.copy()
    df["pos_bin"] = pd.cut(df["rel_pos"], bins=bin_edges, labels=bin_labels,
                           include_lowest=True, right=True)
    agg = (df.groupby(["pos_bin", "route"], observed=True)
             .agg(accuracy=("is_correct", "mean"), n=("is_correct", "count"))
             .reset_index())
    return agg


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _route_list(df: pd.DataFrame) -> List[str]:
    """Routes present in the position DataFrame, in canonical order."""
    present = set(df["route"].unique())
    return [r for r in ROUTES if r in present]


def fig_position_curve(agg: pd.DataFrame, out_dir: str, label: str,
                        title_suffix: str = "", fname_suffix: str = "") -> None:
    routes = _route_list(agg)
    bins   = (agg["pos_bin"].cat.categories.tolist()
              if hasattr(agg["pos_bin"], "cat")
              else sorted(agg["pos_bin"].unique()))
    x      = np.arange(len(bins))

    fig, ax = plt.subplots(figsize=(9, 5))
    for route in routes:
        sub = agg[agg["route"] == route].set_index("pos_bin")
        ys  = [float(sub.loc[b, "accuracy"]) if b in sub.index else np.nan for b in bins]
        ax.plot(x, ys, marker="o", ms=5, lw=2,
                color=_ROUTE_COLORS[route], label=_ROUTE_LABELS[route])

    ax.set_xticks(x)
    ax.set_xticklabels(bins, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Relative serial position (0.0 = first phoneme, 1.0 = last)")
    ax.set_ylabel("Per-position accuracy (teacher-forced)")
    ax.set_ylim(0, 1.05)
    title = (f"Serial-position accuracy — {label.upper()}{title_suffix}\n"
             f"({NOTE})")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    n_total = int(agg["n"].sum() // max(len(routes), 1))
    ax.text(0.01, 0.01, f"n_positions={n_total}", transform=ax.transAxes,
            fontsize=7, color="gray")

    fig.tight_layout()
    fname = f"position_error_curve_{label}{fname_suffix}.png"
    path  = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [fig] {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred",    default=PRED_DEFAULT,
                   help="Path to item_level_predictions.tsv")
    p.add_argument("--out_dir", default=None,
                   help="Output directory (default: --pred's directory/../figures)")
    p.add_argument("--label",   default=LABEL_DEFAULT,
                   help="Label used in filenames and titles (e.g. 'wfe', 'ssp')")
    p.add_argument("--n_bins",  type=int, default=10,
                   help="Number of relative-position bins (default: 10)")
    p.add_argument("--pseudowords_only", action="store_true",
                   help="Also produce a pseudoword-only position curve")
    return p.parse_args()


def _resolve_out_dir(pred_path: str, out_dir: Optional[str]) -> str:
    if out_dir is not None:
        return out_dir
    pred_abs  = os.path.abspath(pred_path)
    parent    = os.path.dirname(pred_abs)
    candidate = os.path.join(os.path.dirname(parent), "figures")
    return candidate


def main():
    args    = parse_args()
    out_dir = _resolve_out_dir(args.pred, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[plot_position_errors_from_predictions]")
    print(f"  Input  : {args.pred}")
    print(f"  Output : {out_dir}")
    print(f"  Label  : {args.label}")

    if not os.path.exists(args.pred):
        print(f"\nERROR: predictions TSV not found: {args.pred}")
        print("  Run: python scripts/external_eval.py")
        sys.exit(1)

    df_pred = pd.read_csv(args.pred, sep="\t")
    print(f"  {len(df_pred)} rows loaded from TSV")

    # Detect which routes have both predicted+target columns
    routes = [r for r in ROUTES
              if f"{r}_predicted" in df_pred.columns
              and f"{r}_target"   in df_pred.columns]
    if not routes:
        print("ERROR: No route columns found in TSV.")
        print("  Expected columns: full_predicted, full_target, wm_predicted, ...")
        sys.exit(1)
    print(f"  Routes detected: {routes}")

    # Detect pseudoword column
    pseudo_col: Optional[str] = None
    for candidate in ("lexicality", "condition"):
        if candidate in df_pred.columns:
            pseudo_col = candidate
            break

    # --- Full dataset ---
    print("\n  Computing per-position records (all items) …")
    df_pos_all = build_position_df(df_pred, routes)
    print(f"  {len(df_pos_all)} (item, position, route) records")

    agg_all = bin_position_df(df_pos_all, args.n_bins)

    # Save raw position data
    tbl_path = os.path.join(out_dir, f"position_error_data_{args.label}.tsv")
    df_pos_all.to_csv(tbl_path, sep="\t", index=False)
    print(f"  -> {tbl_path}")

    # Print per-route summary
    print(f"\n  === SERIAL POSITION SUMMARY ({args.label.upper()}, all items) ===")
    for route in routes:
        sub = agg_all[agg_all["route"] == route].sort_values("pos_bin")
        if len(sub) == 0:
            continue
        a_first = float(sub.iloc[0]["accuracy"])
        a_last  = float(sub.iloc[-1]["accuracy"])
        a_mid   = float(sub.iloc[len(sub) // 2]["accuracy"])
        ushape  = "YES" if min(a_first, a_last) > a_mid else "NO"
        print(f"  [{route:5s}]  first={a_first:.3f}  mid={a_mid:.3f}  "
              f"last={a_last:.3f}  U-shaped={ushape}")

    fig_position_curve(agg_all, out_dir, args.label)

    # --- Pseudowords only ---
    if pseudo_col is not None:
        # Detect pseudoword rows: lexicality == 'pseudo' OR
        # condition starts with 'P' (WFE convention: PLC, PLS, PSC, PSS)
        pseudo_mask = (
            df_pred[pseudo_col].str.lower().str.startswith("p") |
            df_pred[pseudo_col].str.lower().eq("pseudo") |
            df_pred[pseudo_col].str.lower().eq("pseudoword")
        )
        df_pseudo = df_pred[pseudo_mask].copy()
        n_pseudo  = int(pseudo_mask.sum())
        print(f"\n  Pseudoword filter ({pseudo_col}): {n_pseudo} items")

        if n_pseudo > 0:
            df_pos_pseudo = build_position_df(df_pseudo.reset_index(drop=True), routes)
            agg_pseudo    = bin_position_df(df_pos_pseudo, args.n_bins)

            tbl_p = os.path.join(out_dir, f"position_error_data_{args.label}_pseudowords.tsv")
            df_pos_pseudo.to_csv(tbl_p, sep="\t", index=False)
            print(f"  -> {tbl_p}")

            print(f"\n  === SERIAL POSITION SUMMARY ({args.label.upper()}, pseudowords only) ===")
            for route in routes:
                sub = agg_pseudo[agg_pseudo["route"] == route].sort_values("pos_bin")
                if len(sub) == 0:
                    continue
                a_first = float(sub.iloc[0]["accuracy"])
                a_last  = float(sub.iloc[-1]["accuracy"])
                a_mid   = float(sub.iloc[len(sub) // 2]["accuracy"])
                ushape  = "YES" if min(a_first, a_last) > a_mid else "NO"
                print(f"  [{route:5s}]  first={a_first:.3f}  mid={a_mid:.3f}  "
                      f"last={a_last:.3f}  U-shaped={ushape}")

            fig_position_curve(agg_pseudo, out_dir, args.label,
                               title_suffix=" — pseudowords only",
                               fname_suffix="_pseudowords")
        else:
            print("  No pseudoword items found; skipping pseudoword-only plot.")

    elif args.pseudowords_only:
        print("  Warning: --pseudowords_only requested but no lexicality/condition column found.")

    print(f"\n[plot_position_errors_from_predictions] Done.  Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
