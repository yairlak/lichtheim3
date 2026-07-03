"""Error-type breakdown: substitutions, insertions, deletions.

Reads per-item WFE predictions produced by scripts/external_eval.py and classifies
each phoneme error using Levenshtein edit operations.  Does not require the
python-Levenshtein package (a pure-Python DP backtrace is used by default).

Outputs:
    error_type_breakdown.png     — stacked bar: n errors per type, by route
    error_type_by_condition.png  — error types per WFE condition code (full route)
    error_type_by_lexicality.png — error types split real / pseudo (all routes)
    error_types.tsv              — per-item counts

NOTE: error rates are computed over ALL items (correct items contribute 0 errors),
so they represent errors per item, not errors per erroneous item.

Usage:
    python scripts/plot_error_types.py
    python scripts/plot_error_types.py \\
        --pred    outputs/external_eval_30k/wfe/item_level_predictions.tsv \\
        --out_dir outputs/external_eval_30k/figures
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

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

_ROUTE_LABELS = {"full": "Full (gated)", "wm": "WM (dorsal)", "ltm": "LTM (ventral)"}
_TYPE_COLORS  = {"sub": "#e05a2b", "ins": "#2b7bba", "del": "#2ba34b"}
_TYPE_LABELS  = {"sub": "Substitution", "ins": "Insertion", "del": "Deletion"}


# ---------------------------------------------------------------------------
# Levenshtein edit-op backtrace (self-contained, no external library)
# ---------------------------------------------------------------------------

def _editops(src: List[str], tgt: List[str]) -> List[Tuple[str, int, int]]:
    """Return edit operations (op, src_idx, tgt_idx) to transform src into tgt.

    op in {'replace', 'insert', 'delete'}:
      replace  : src[src_idx] -> tgt[tgt_idx]
      delete   : delete src[src_idx]            (src has extra phoneme)
      insert   : insert tgt[tgt_idx] at src_idx (tgt has extra phoneme)
    """
    n, m = len(src), len(tgt)
    INF = n + m + 1
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if src[i - 1] == tgt[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1],  # replace
                                   dp[i - 1][j],       # delete from src
                                   dp[i][j - 1])       # insert into src

    ops: List[Tuple[str, int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and src[i - 1] == tgt[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("replace", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("delete", i - 1, j))
            i -= 1
        else:
            ops.append(("insert", i, j - 1))
            j -= 1
    return ops


def _count_ops(src: List[str], tgt: List[str]) -> Dict[str, int]:
    """Count substitutions, insertions, deletions from the model's perspective.

    src = predicted sequence, tgt = gold sequence.
    editops(src, tgt) gives ops to transform src into tgt:
      'replace' → model said wrong phoneme                    → substitution
      'delete'  → delete from src (src has extra)            → model INSERTED a phoneme
      'insert'  → insert into src (src is missing one)       → model DELETED a phoneme
    """
    ops = _editops(src, tgt)
    return {
        "sub": sum(1 for op in ops if op[0] == "replace"),
        "ins": sum(1 for op in ops if op[0] == "delete"),   # extra in prediction
        "del": sum(1 for op in ops if op[0] == "insert"),   # missing from prediction
    }


# ---------------------------------------------------------------------------
# Per-item computation
# ---------------------------------------------------------------------------

def compute_error_types(df: pd.DataFrame) -> pd.DataFrame:
    """Add n_sub, n_ins, n_del columns for each route."""
    df = df.copy()
    routes = ["full", "wm", "ltm"]
    for r in routes:
        pred_col = f"{r}_predicted"
        tgt_col  = f"{r}_target"
        if pred_col not in df.columns or tgt_col not in df.columns:
            print(f"  [error_types] columns {pred_col}/{tgt_col} missing — skipping route {r}")
            continue
        subs, inss, dels = [], [], []
        for _, row in df.iterrows():
            pred = str(row[pred_col]).split() if pd.notna(row[pred_col]) else []
            tgt  = str(row[tgt_col]).split()  if pd.notna(row[tgt_col])  else []
            c = _count_ops(pred, tgt)
            subs.append(c["sub"])
            inss.append(c["ins"])
            dels.append(c["del"])
        df[f"{r}_n_sub"] = subs
        df[f"{r}_n_ins"] = inss
        df[f"{r}_n_del"] = dels
        df[f"{r}_n_errors"] = df[f"{r}_n_sub"] + df[f"{r}_n_ins"] + df[f"{r}_n_del"]
    return df


# ---------------------------------------------------------------------------
# Figure 1: stacked bar by route
# ---------------------------------------------------------------------------

def fig_error_type_breakdown(df: pd.DataFrame, out_dir: str) -> None:
    routes   = ["full", "wm", "ltm"]
    x        = np.arange(len(routes))
    present  = [r for r in routes if f"{r}_n_sub" in df.columns]
    if not present:
        print("[error_types] No error-type columns found — run compute_error_types first.")
        return

    sub_vals = [df[f"{r}_n_sub"].mean() for r in present]
    ins_vals = [df[f"{r}_n_ins"].mean() for r in present]
    del_vals = [df[f"{r}_n_del"].mean() for r in present]
    xlabels  = [_ROUTE_LABELS.get(r, r) for r in present]

    x = np.arange(len(present))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, sub_vals, label=_TYPE_LABELS["sub"], color=_TYPE_COLORS["sub"])
    ax.bar(x, ins_vals, bottom=sub_vals, label=_TYPE_LABELS["ins"], color=_TYPE_COLORS["ins"])
    ax.bar(x, del_vals,
           bottom=[s + i for s, i in zip(sub_vals, ins_vals)],
           label=_TYPE_LABELS["del"], color=_TYPE_COLORS["del"])

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("Mean errors per item")
    ax.set_title(f"Phoneme error types by route\n({TEACHER_FORCED_NOTE})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    # Annotate total
    totals = [s + i + d for s, i, d in zip(sub_vals, ins_vals, del_vals)]
    for xi, tot in zip(x, totals):
        ax.text(xi, tot + 0.005, f"{tot:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "error_type_breakdown.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [fig] {path}")


# ---------------------------------------------------------------------------
# Figure 2: error types by WFE condition (full route)
# ---------------------------------------------------------------------------

def fig_error_type_by_condition(df: pd.DataFrame, out_dir: str) -> None:
    route = "full"
    if f"{route}_n_sub" not in df.columns or "condition" not in df.columns:
        print(f"[error_types] Missing columns for condition breakdown — skipping.")
        return

    conds   = [c for c in _CONDITION_ORDER if (df["condition"] == c).any()]
    if not conds:
        print("[error_types] No condition codes found — skipping condition plot.")
        return

    sub_vals = [df[df["condition"] == c][f"{route}_n_sub"].mean() for c in conds]
    ins_vals = [df[df["condition"] == c][f"{route}_n_ins"].mean() for c in conds]
    del_vals = [df[df["condition"] == c][f"{route}_n_del"].mean() for c in conds]

    x   = np.arange(len(conds))
    fig, ax = plt.subplots(figsize=(max(9, len(conds) * 0.8), 4))
    ax.bar(x, sub_vals, label=_TYPE_LABELS["sub"], color=_TYPE_COLORS["sub"])
    ax.bar(x, ins_vals, bottom=sub_vals, label=_TYPE_LABELS["ins"], color=_TYPE_COLORS["ins"])
    ax.bar(x, del_vals,
           bottom=[s + i for s, i in zip(sub_vals, ins_vals)],
           label=_TYPE_LABELS["del"], color=_TYPE_COLORS["del"])

    # Separator real/pseudo
    n_real = sum(1 for c in conds if c.startswith("R"))
    if 0 < n_real < len(conds):
        ax.axvline(n_real - 0.5, color="gray", lw=1, ls="--", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean errors per item (full route)")
    ax.set_title(f"Phoneme error types by WFE condition (full / gated route)\n"
                 f"({TEACHER_FORCED_NOTE})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(out_dir, "error_type_by_condition.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [fig] {path}")


# ---------------------------------------------------------------------------
# Figure 3: error types by lexicality × route
# ---------------------------------------------------------------------------

def fig_error_type_by_lexicality(df: pd.DataFrame, out_dir: str) -> None:
    routes = ["full", "wm", "ltm"]
    present = [r for r in routes if f"{r}_n_sub" in df.columns]
    if not present or "lexicality" not in df.columns:
        return

    lexicalities = [l for l in ["real", "pseudo"] if (df["lexicality"] == l).any()]
    n_groups = len(lexicalities) * len(present)
    x = np.arange(len(lexicalities))
    n_routes = len(present)
    width = 0.65 / n_routes
    offsets = np.linspace(-(n_routes - 1) * width / 2, (n_routes - 1) * width / 2, n_routes)

    # Route-specific shades for sub/ins/del within each route bar
    # Use alpha variation within route colour
    route_base = {"full": "#2b7bba", "wm": "#e05a2b", "ltm": "#2ba34b"}

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    type_col  = {"sub": _TYPE_COLORS["sub"], "ins": _TYPE_COLORS["ins"],
                 "del": _TYPE_COLORS["del"]}
    for ax_i, (err_type, label) in enumerate(
            [("sub", "Substitutions"), ("ins", "Insertions"), ("del", "Deletions")]):
        ax = axes[ax_i]
        for ri, r in enumerate(present):
            vals = [df[df["lexicality"] == l][f"{r}_n_{err_type}"].mean()
                    for l in lexicalities]
            ax.bar(x + offsets[ri], vals, width,
                   label=_ROUTE_LABELS.get(r, r), color=route_base[r], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(lexicalities, fontsize=9)
        ax.set_title(label)
        ax.set_ylabel("Mean count per item" if ax_i == 0 else "")
        ax.grid(alpha=0.3, axis="y")
        if ax_i == 0:
            ax.legend(fontsize=7)

    fig.suptitle(f"Error types by lexicality × route\n({TEACHER_FORCED_NOTE})", fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "error_type_by_lexicality.png")
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
    p.add_argument("--out_dir", default=OUT_DEFAULT)
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

    print(f"\n[plot_error_types] Loading: {args.pred}")
    df = pd.read_csv(args.pred, sep="\t")
    print(f"  {len(df)} items")

    # Add lexicality from condition if missing
    if "lexicality" not in df.columns and "condition" in df.columns:
        _CMAP = {
            "RLCH":"real","RLCL":"real","RLSH":"real","RLSL":"real",
            "RSCH":"real","RSCL":"real","RSSH":"real","RSSL":"real",
            "PLC":"pseudo","PLS":"pseudo","PSC":"pseudo","PSS":"pseudo",
        }
        df["lexicality"] = df["condition"].map(_CMAP)

    print("  Computing edit operations …")
    df = compute_error_types(df)

    # Save table
    route_cols = [c for c in df.columns if any(
        c.startswith(f"{r}_n_") for r in ["full","wm","ltm"])]
    keep_cols  = ["word", "condition", "lexicality"] + \
                 [c for c in df.columns if c in
                  ["full_exact_match","wm_exact_match","ltm_exact_match"]] + \
                 route_cols
    keep_cols  = [c for c in keep_cols if c in df.columns]
    tbl_path   = os.path.join(args.out_dir, "error_types.tsv")
    df[keep_cols].to_csv(tbl_path, sep="\t", index=False)
    print(f"  -> {tbl_path}")

    # Summary
    for r in ["full", "wm", "ltm"]:
        if f"{r}_n_sub" not in df.columns:
            continue
        total = df[f"{r}_n_errors"].mean() if f"{r}_n_errors" in df.columns else float("nan")
        sub   = df[f"{r}_n_sub"].mean()
        ins   = df[f"{r}_n_ins"].mean()
        dl    = df[f"{r}_n_del"].mean()
        print(f"  [{r:5s}]  mean errors/item={total:.3f}  "
              f"sub={sub:.3f}  ins={ins:.3f}  del={dl:.3f}")

    fig_error_type_breakdown(df, args.out_dir)
    fig_error_type_by_condition(df, args.out_dir)
    fig_error_type_by_lexicality(df, args.out_dir)

    print("\n[plot_error_types] Done.")


if __name__ == "__main__":
    main()
