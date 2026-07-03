"""Route dissociation / ablation analysis on WFE items.

Reads the per-item predictions TSV produced by scripts/external_eval.py.
The per-route exact-match columns (full / wm / ltm) are already present;
this script builds the dissociation figures without re-running inference.

"Ablation" here means route isolation, not weight zeroing: the per-route
scores reflect the model evaluated through each route alone (via route_logits).
This is a clean route-isolation analysis; it is NOT a biological lesion.

Outputs:
    route_dissociation_wfe.png      — WM vs LTM item-level scatter (jittered), coloured by
                                      lexicality and model-centred lexicon category
    route_accuracy_by_lexicality.png — grouped barplot: full/wm/ltm by real vs pseudo
    route_accuracy_by_category.png  — grouped barplot: per model-centred lexicon category
    route_advantage_scatter.png     — scatter: LTM advantage (ltm-wm) vs WM advantage (wm-ltm),
                                      one point per item; shows double-dissociation cluster

Usage:
    python scripts/run_route_ablation_wfe.py
    python scripts/run_route_ablation_wfe.py \\
        --pred    outputs/external_eval_30k/wfe/item_level_predictions.tsv \\
        --out_dir outputs/external_eval_30k/figures
"""
from __future__ import annotations

import argparse
import os
import sys

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

_ROUTE_COLORS  = {"full": "#2b7bba", "wm": "#e05a2b", "ltm": "#2ba34b"}
_ROUTE_LABELS  = {"full": "Full (gated)", "wm": "WM (dorsal)", "ltm": "LTM (ventral)"}

_LEX_COLORS    = {"real": "#2b7bba", "pseudo": "#e05a2b"}

_CAT_COLORS = {
    "real_word_seen_in_training_lexicon": "#2b7bba",
    "real_word_in_validation_split":      "#5aa5d6",
    "real_word_outside_4000_lexicon":     "#a0c8e8",
    "pseudoword":                         "#e05a2b",
}
_CAT_LABELS = {
    "real_word_seen_in_training_lexicon": "Seen (train)",
    "real_word_in_validation_split":      "Held-out (val)",
    "real_word_outside_4000_lexicon":     "Real, outside lex",
    "pseudoword":                         "Pseudoword",
}
_CAT_ORDER = list(_CAT_COLORS.keys())


# ---------------------------------------------------------------------------
# Figure 1: WM vs LTM jittered scatter
# ---------------------------------------------------------------------------

def fig_dissociation_scatter(df: pd.DataFrame, out_dir: str) -> None:
    wm_col  = "wm_exact_match"
    ltm_col = "ltm_exact_match"
    if wm_col not in df.columns or ltm_col not in df.columns:
        print("[ablation] per-route columns missing — skipping scatter.")
        return

    lex_col  = "lexicality" if "lexicality" in df.columns else None
    cat_col  = "lexicon_category" if "lexicon_category" in df.columns else None
    use_cat  = cat_col is not None

    jitter = 0.06
    rng    = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (group_col, colors_map, labels_map) in zip(
        axes,
        [
            (lex_col,  _LEX_COLORS,  {k: k for k in _LEX_COLORS}),
            (cat_col,  _CAT_COLORS,  _CAT_LABELS),
        ]
    ):
        if group_col is None or group_col not in df.columns:
            ax.text(0.5, 0.5, "Column not available", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            continue

        groups = df[group_col].dropna().unique()
        for g in groups:
            sub  = df[df[group_col] == g]
            wm_j = sub[wm_col].values.astype(float) + rng.uniform(-jitter, jitter, len(sub))
            lt_j = sub[ltm_col].values.astype(float) + rng.uniform(-jitter, jitter, len(sub))
            ax.scatter(wm_j, lt_j, s=8, alpha=0.35,
                       color=colors_map.get(str(g), "gray"),
                       label=labels_map.get(str(g), str(g)))

        # Quadrant lines at 0.5
        ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.4)
        ax.axvline(0.5, color="gray", lw=0.8, ls="--", alpha=0.4)
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_xlabel("WM route accuracy (jittered)")
        ax.set_ylabel("LTM route accuracy (jittered)")
        ax.legend(fontsize=7, markerscale=2)
        ax.grid(alpha=0.2)

        # Quadrant labels
        ax.text(0.05, 0.98, "WM✓ LTM✗", fontsize=7, color="gray",
                ha="left", va="top", transform=ax.transAxes)
        ax.text(0.98, 0.98, "Both ✓", fontsize=7, color="gray",
                ha="right", va="top", transform=ax.transAxes)
        ax.text(0.05, 0.02, "Both ✗", fontsize=7, color="gray",
                ha="left", va="bottom", transform=ax.transAxes)
        ax.text(0.98, 0.02, "WM✗ LTM✓", fontsize=7, color="gray",
                ha="right", va="bottom", transform=ax.transAxes)

    axes[0].set_title("By dataset lexicality")
    axes[1].set_title("By model-centred lexicon category")
    fig.suptitle(f"Route dissociation — WM vs LTM accuracy per item\n"
                 f"({TEACHER_FORCED_NOTE}, jittered binary values)", fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "route_dissociation_wfe.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [fig] {path}")


# ---------------------------------------------------------------------------
# Figure 2: grouped barplot by lexicality
# ---------------------------------------------------------------------------

def fig_accuracy_by_lexicality(df: pd.DataFrame, out_dir: str) -> None:
    routes = [r for r in ["full", "wm", "ltm"] if f"{r}_exact_match" in df.columns]
    lex_col = "lexicality" if "lexicality" in df.columns else None
    if not routes or lex_col is None:
        return

    lexs = [l for l in ["real", "pseudo"] if (df[lex_col] == l).any()]
    n    = len(lexs)
    x    = np.arange(n)
    width = 0.22
    offsets = np.linspace(-width, width, len(routes))

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, r in enumerate(routes):
        vals = [float(df[df[lex_col] == l][f"{r}_exact_match"].mean()) for l in lexs]
        ax.bar(x + offsets[i], vals, width,
               label=_ROUTE_LABELS.get(r, r), color=_ROUTE_COLORS[r], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(lexs, fontsize=10)
    ax.set_ylabel("Exact-match accuracy (teacher-forced)")
    ax.set_title(f"Route accuracy by lexicality\n({TEACHER_FORCED_NOTE})")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # n labels
    for i, l in enumerate(lexs):
        n_l = int((df[lex_col] == l).sum())
        ax.text(x[i], -0.05, f"n={n_l}", ha="center", va="top", fontsize=7,
                transform=ax.get_xaxis_transform())

    fig.tight_layout()
    path = os.path.join(out_dir, "route_accuracy_by_lexicality.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [fig] {path}")


# ---------------------------------------------------------------------------
# Figure 3: grouped barplot by model-centred lexicon category
# ---------------------------------------------------------------------------

def fig_accuracy_by_category(df: pd.DataFrame, out_dir: str) -> None:
    routes = [r for r in ["full", "wm", "ltm"] if f"{r}_exact_match" in df.columns]
    cat_col = "lexicon_category" if "lexicon_category" in df.columns else None
    if not routes or cat_col is None:
        print("[ablation] lexicon_category column missing — skipping category plot.")
        return

    cats   = [c for c in _CAT_ORDER if (df[cat_col] == c).any()]
    clabel = [_CAT_LABELS.get(c, c) for c in cats]
    x      = np.arange(len(cats))
    width  = 0.22
    offsets = np.linspace(-width, width, len(routes))

    fig, ax = plt.subplots(figsize=(max(7, len(cats) * 1.6), 4))
    for i, r in enumerate(routes):
        vals = [float(df[df[cat_col] == c][f"{r}_exact_match"].mean()) for c in cats]
        ax.bar(x + offsets[i], vals, width,
               label=_ROUTE_LABELS.get(r, r), color=_ROUTE_COLORS[r], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(clabel, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Exact-match accuracy (teacher-forced)")
    ax.set_title(f"Route accuracy by model-centred lexicon category\n"
                 f"({TEACHER_FORCED_NOTE})")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    for i, c in enumerate(cats):
        n_c = int((df[cat_col] == c).sum())
        ax.text(x[i], -0.05, f"n={n_c}", ha="center", va="top", fontsize=7,
                transform=ax.get_xaxis_transform())

    fig.tight_layout()
    path = os.path.join(out_dir, "route_accuracy_by_category.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [fig] {path}")


# ---------------------------------------------------------------------------
# Figure 4: LTM advantage vs WM advantage scatter (per condition)
# ---------------------------------------------------------------------------

def fig_route_advantage_scatter(df: pd.DataFrame, out_dir: str) -> None:
    """Scatter: x = LTM advantage (ltm-wm), y = WM advantage (wm-ltm)."""
    if "wm_exact_match" not in df.columns or "ltm_exact_match" not in df.columns:
        return
    if "condition" not in df.columns:
        return

    cond_stats = []
    for cond in sorted(df["condition"].dropna().unique()):
        sub = df[df["condition"] == cond]
        wm_acc  = float(sub["wm_exact_match"].mean())
        ltm_acc = float(sub["ltm_exact_match"].mean())
        full_acc = float(sub["full_exact_match"].mean()) if "full_exact_match" in sub else np.nan
        lex = "real" if cond.startswith("R") else "pseudo"
        cond_stats.append({
            "condition": cond,
            "wm_acc":   wm_acc,
            "ltm_acc":  ltm_acc,
            "full_acc": full_acc,
            "ltm_adv":  ltm_acc - wm_acc,   # positive: LTM better
            "wm_adv":   wm_acc  - ltm_acc,  # positive: WM better
            "lexicality": lex,
        })

    if not cond_stats:
        return

    cs_df = pd.DataFrame(cond_stats)
    fig, ax = plt.subplots(figsize=(6, 5))

    for lex, group in cs_df.groupby("lexicality"):
        ax.scatter(group["ltm_adv"], group["wm_adv"],
                   color=_LEX_COLORS.get(lex, "gray"), s=60, alpha=0.8,
                   label=lex, zorder=3)
        for _, row in group.iterrows():
            ax.annotate(row["condition"], (row["ltm_adv"], row["wm_adv"]),
                        fontsize=6, xytext=(3, 3), textcoords="offset points",
                        color="gray")

    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("LTM advantage over WM  (ltm_acc − wm_acc)")
    ax.set_ylabel("WM advantage over LTM  (wm_acc − ltm_acc)")
    ax.set_title(f"Double-dissociation pattern by WFE condition\n"
                 f"({TEACHER_FORCED_NOTE})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.text(0.98, 0.98, "WM dominant →", fontsize=7, color="gray",
            ha="right", va="top", transform=ax.transAxes)
    ax.text(0.02, 0.02, "← LTM dominant", fontsize=7, color="gray",
            ha="left", va="bottom", transform=ax.transAxes)

    fig.tight_layout()
    path = os.path.join(out_dir, "route_advantage_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [fig] {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred",    default=PRED_DEFAULT)
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

    print(f"\n[run_route_ablation_wfe] Loading: {args.pred}")
    df = pd.read_csv(args.pred, sep="\t")
    print(f"  {len(df)} items, columns: {list(df.columns)}")

    # Derive lexicality from condition if absent
    if "lexicality" not in df.columns and "condition" in df.columns:
        _CMAP = {
            "RLCH":"real","RLCL":"real","RLSH":"real","RLSL":"real",
            "RSCH":"real","RSCL":"real","RSSH":"real","RSSL":"real",
            "PLC":"pseudo","PLS":"pseudo","PSC":"pseudo","PSS":"pseudo",
        }
        df["lexicality"] = df["condition"].map(_CMAP)

    # Summary table
    routes = [r for r in ["full","wm","ltm"] if f"{r}_exact_match" in df.columns]
    print("\n  Route accuracy by lexicality:")
    if "lexicality" in df.columns:
        for lex in ["real","pseudo"]:
            sub = df[df["lexicality"] == lex]
            if len(sub) == 0:
                continue
            vals = {r: round(float(sub[f"{r}_exact_match"].mean()),3) for r in routes}
            print(f"  [{lex:6s}]  n={len(sub)}  " + "  ".join(f"{r}={v}" for r,v in vals.items()))

    fig_dissociation_scatter(df, args.out_dir)
    fig_accuracy_by_lexicality(df, args.out_dir)
    fig_accuracy_by_category(df, args.out_dir)
    fig_route_advantage_scatter(df, args.out_dir)

    print(f"\n[run_route_ablation_wfe] Done.  Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
