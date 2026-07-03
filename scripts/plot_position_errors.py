"""Serial-position accuracy curve for the dual-route model.

Loads the 30k/GloVe checkpoint, evaluates all training entries, and plots
per-position accuracy against relative serial position (0.0 = first phoneme,
1.0 = last phoneme).  A U-shaped curve for the WM (dorsal) route indicates
a primacy and recency advantage consistent with a capacity-limited serial-recall
buffer (Botvinick & Plaut 2006).

Evaluation is deterministic: WM interference noise is disabled (collect=False).
All decoding is teacher-forced (gold prefix at each step).

Outputs:
    outputs/train_ceiling_analysis/serial_position_curve.png
    outputs/train_ceiling_analysis/serial_position_curve_by_length.png
    outputs/train_ceiling_analysis/serial_position_data.tsv

Usage:
    python scripts/plot_position_errors.py
    python scripts/plot_position_errors.py \\
        --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt \\
        --lexicon_path data/lexicon_en_glove_covered.tsv \\
        --out_dir outputs/train_ceiling_analysis \\
        --n_bins 10
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import Config, DataConfig, WMConfig, LTMConfig, GatingConfig, LossConfig, TrainConfig
from data.phonemes import build_vocab
from data.lexicon import build_lexicon
from models.dual_route import DualRouteModel
from evaluate.hooks import make_batch, route_predictions

CKPT_DEFAULT    = os.path.join(ROOT, "checkpoints",
                                "lichtheim3_30k_glove_e60_to_e120_lowlr.pt")
LEXICON_DEFAULT = os.path.join(ROOT, "data", "lexicon_en_glove_covered.tsv")
OUT_DEFAULT     = os.path.join(ROOT, "outputs", "train_ceiling_analysis")
BATCH_SIZE      = 128

TEACHER_FORCED_NOTE = "Teacher-forced decoding · WM noise disabled (deterministic)"

_ROUTE_COLORS = {"full": "#2b7bba", "wm": "#e05a2b", "ltm": "#2ba34b"}
_ROUTE_LABELS = {"full": "Full (gated)", "wm": "WM (dorsal)", "ltm": "LTM (ventral)"}

# Length categories (in phonemes, inclusive)
_LENGTH_BINS = [
    ("short (2–4)",  2, 4),
    ("medium (5–7)", 5, 7),
    ("long (8–9)",   8, 9),
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_data(ckpt_path: str, lexicon_path_override: str | None,
                         device: str):
    if not os.path.exists(ckpt_path):
        print(f"\nERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = Config(
        data   = DataConfig(**ckpt["cfg_data"]),
        wm     = WMConfig(**ckpt["cfg_wm"]),
        ltm    = LTMConfig(**ckpt["cfg_ltm"]),
        gating = GatingConfig(**ckpt["cfg_gating"]),
        loss   = LossConfig(**ckpt["cfg_loss"]),
        train  = TrainConfig(**ckpt["cfg_train"]),
    )
    cfg.train.device = device
    if lexicon_path_override:
        cfg.data.lexicon_path = lexicon_path_override

    vocab   = build_vocab()
    lexicon = build_lexicon(cfg.data, vocab)
    train_entries, _ = lexicon.split(cfg.data.val_fraction, cfg.data.seed)

    bank = torch.stack(
        [torch.tensor(e.semantic) for e in train_entries]
    ).float().to(device)

    model = DualRouteModel(cfg, vocab).to(device)
    model.set_semantic_bank(bank)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, vocab, train_entries


# ---------------------------------------------------------------------------
# Per-position evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_position_data(model, vocab, entries, device: str,
                           n_items: int | None = None) -> pd.DataFrame:
    """Return a DataFrame with one row per (item, position, route)."""
    if n_items:
        entries = entries[:n_items]

    forms   = [e.phonemes for e in entries]
    routes  = ("full", "wm", "ltm")
    rows: List[dict] = []

    for start in range(0, len(forms), BATCH_SIZE):
        bforms   = forms[start: start + BATCH_SIZE]
        bentries = entries[start: start + BATCH_SIZE]
        batch    = make_batch(bforms, vocab, device)

        preds_by_route = {}
        for route in routes:
            # collect=False → no WM noise → deterministic
            preds, _ = route_predictions(model, batch, route=route, collect=False)
            preds_by_route[route] = preds.cpu()

        for i, (entry, fids) in enumerate(zip(bentries, bforms)):
            n   = len(fids)
            if n < 2:
                continue  # skip single-phoneme (shouldn't occur; min_phonemes=2)
            for pos in range(n):
                # Relative position: 0.0 for first, 1.0 for last phoneme
                rel_pos = pos / (n - 1)
                row_base = {
                    "word":     entry.word,
                    "length":   n,
                    "position": pos,
                    "rel_pos":  round(rel_pos, 4),
                }
                for route in routes:
                    pred_id = int(preds_by_route[route][i, pos].item())
                    tgt_id  = fids[pos]
                    row = dict(row_base)
                    row["route"]    = route
                    row["is_correct"] = int(pred_id == tgt_id)
                    rows.append(row)

        if (start // BATCH_SIZE) % 20 == 0:
            print(f"  … {min(start+BATCH_SIZE, len(forms))}/{len(forms)}", end="\r")
    print()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def bin_data(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """Aggregate by relative-position bin per route."""
    df = df.copy()
    bin_edges  = np.linspace(0.0, 1.0, n_bins + 1)
    bin_labels = [f"{bin_edges[i]:.1f}–{bin_edges[i+1]:.1f}" for i in range(n_bins)]
    df["pos_bin"] = pd.cut(df["rel_pos"], bins=bin_edges, labels=bin_labels,
                           include_lowest=True, right=True)
    agg = (df.groupby(["pos_bin", "route"], observed=True)
             .agg(accuracy=("is_correct", "mean"),
                  n=("is_correct", "count"))
             .reset_index())
    return agg


def bin_data_by_length(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """Aggregate by relative-position bin × length category × route."""
    df = df.copy()
    length_map = {}
    for label, lo, hi in _LENGTH_BINS:
        for ln in range(lo, hi + 1):
            length_map[ln] = label
    df["length_cat"] = df["length"].map(length_map).fillna("other")
    bin_edges  = np.linspace(0.0, 1.0, n_bins + 1)
    bin_labels = [f"{bin_edges[i]:.1f}–{bin_edges[i+1]:.1f}" for i in range(n_bins)]
    df["pos_bin"] = pd.cut(df["rel_pos"], bins=bin_edges, labels=bin_labels,
                           include_lowest=True, right=True)
    agg = (df.groupby(["pos_bin", "route", "length_cat"], observed=True)
             .agg(accuracy=("is_correct", "mean"),
                  n=("is_correct", "count"))
             .reset_index())
    return agg


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_serial_position_curve(agg: pd.DataFrame, out_dir: str) -> None:
    routes = ["full", "wm", "ltm"]
    bins   = agg["pos_bin"].cat.categories.tolist() if hasattr(
        agg["pos_bin"], "cat") else sorted(agg["pos_bin"].unique())
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
    ax.set_title(
        f"Serial-position accuracy curve — lichtheim3 30k\n"
        f"({TEACHER_FORCED_NOTE})"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    n_items = int(agg["n"].sum() // len(routes))
    ax.text(0.01, 0.01, f"n_positions={n_items}", transform=ax.transAxes,
            fontsize=7, color="gray")

    fig.tight_layout()
    path = os.path.join(out_dir, "serial_position_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [fig] {path}")


def fig_serial_position_by_length(agg_len: pd.DataFrame, out_dir: str) -> None:
    routes = ["wm", "ltm"]   # focus on route contrast, full would overlap
    length_cats = [l for l, _, _ in _LENGTH_BINS]
    bins = sorted(agg_len["pos_bin"].unique())
    x    = np.arange(len(bins))

    fig, axes = plt.subplots(1, len(length_cats), figsize=(13, 4), sharey=True)
    ls_map = {"wm": "-", "ltm": "--"}

    for ax, lcat in zip(axes, length_cats):
        sub_len = agg_len[agg_len["length_cat"] == lcat]
        for route in routes:
            sub = sub_len[sub_len["route"] == route].set_index("pos_bin")
            ys  = [float(sub.loc[b, "accuracy"]) if b in sub.index else np.nan
                   for b in bins]
            ax.plot(x, ys, marker="o", ms=4, lw=1.8, ls=ls_map[route],
                    color=_ROUTE_COLORS[route], label=_ROUTE_LABELS[route])
        ax.set_title(lcat)
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=45, ha="right", fontsize=6)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("Per-position accuracy")
            ax.legend(fontsize=7)

    fig.suptitle(
        f"Serial-position curve by word length\n({TEACHER_FORCED_NOTE})", fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "serial_position_curve_by_length.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [fig] {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",         default=CKPT_DEFAULT)
    p.add_argument("--lexicon_path", default=None,
                   help="Override lexicon path (default: use checkpoint config)")
    p.add_argument("--out_dir",      default=OUT_DEFAULT)
    p.add_argument("--device",       default=None)
    p.add_argument("--n_bins",       type=int, default=10,
                   help="Number of relative-position bins (default: 10)")
    p.add_argument("--n_items",      type=int, default=None,
                   help="Limit to first N training items for quick testing")
    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Guard: prevent smoke tests from overwriting full-run outputs
    if args.n_items is not None and args.out_dir == OUT_DEFAULT:
        args.out_dir = OUT_DEFAULT + "_smoke"
        print(f"  [smoke test] --n_items={args.n_items} set with default --out_dir.")
        print(f"  Writing to: {args.out_dir}  (to avoid clobbering full outputs)")
        print(f"  Pass --out_dir explicitly to override this behaviour.")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n[plot_position_errors] Checkpoint: {args.ckpt}")
    print(f"  Device: {device}")

    model, vocab, train_entries = load_model_and_data(
        args.ckpt, args.lexicon_path, device)

    n_entries = args.n_items or len(train_entries)
    print(f"  Train entries: {len(train_entries)} → evaluating {n_entries}")

    print("\n  Collecting per-position accuracy …")
    df_pos = collect_position_data(model, vocab, train_entries, device, args.n_items)

    print(f"  Collected {len(df_pos)} (item, position, route) records")

    # Save raw table
    tbl_path = os.path.join(args.out_dir, "serial_position_data.tsv")
    df_pos.to_csv(tbl_path, sep="\t", index=False)
    print(f"  -> {tbl_path}")

    # Aggregate and plot
    agg     = bin_data(df_pos, args.n_bins)
    agg_len = bin_data_by_length(df_pos, args.n_bins)

    # Summary: print accuracy at first, middle, last bin per route
    print("\n  === SERIAL POSITION SUMMARY ===")
    for route in ["wm", "ltm", "full"]:
        sub = agg[agg["route"] == route].sort_values("pos_bin")
        if len(sub) == 0:
            continue
        acc_first = float(sub.iloc[0]["accuracy"])
        acc_last  = float(sub.iloc[-1]["accuracy"])
        acc_mid   = float(sub.iloc[len(sub)//2]["accuracy"])
        print(f"  [{route:5s}]  first={acc_first:.3f}  mid={acc_mid:.3f}  last={acc_last:.3f}  "
              f"primacy+recency={'YES' if min(acc_first,acc_last)>acc_mid else 'NO'}")

    fig_serial_position_curve(agg, args.out_dir)
    fig_serial_position_by_length(agg_len, args.out_dir)

    print(f"\n[plot_position_errors] Done.  Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
