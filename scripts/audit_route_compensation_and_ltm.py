"""Audit: route compensation under WM noise and LTM length sensitivity.

Four-part analysis aimed at two diagnostic questions:

  Q1. Why is the full/gated route more robust than WM under WM/dorsal noise?
  Q2. Why is LTM much more length-sensitive than WM, especially on pseudowords?

Part A  Route compensation / full-route robustness  (uses noise_metrics TSV)
  Quantify the rate at which the full/gated route succeeds while individual
  sub-routes fail, across noise levels, groups, and length bins.

Part B  LTM length sensitivity  (uses no-noise AR predictions TSV)
  Per-length edit distance, error-type breakdown (sub/ins/del), and
  position-wise accuracy for LTM, WM, and full routes.

Part C  LTM semantic / lexical confidence  (uses checkpoint — LTM encoder only)
  Run the LTM encoder on every WFE item: s_hat, confidence (max cosine sim),
  margin (top1−top2), density.  Relate to length, group, and output quality.

Part D  Gate audit  (derived from Part C — no extra model call)
  Gate value g = sigmoid(alpha * (confidence − 0.5)) is computed from Part C
  confidence values.  Show gate by group, length, and rescue pattern.

Usage
-----
    python scripts/audit_route_compensation_and_ltm.py

    python scripts/audit_route_compensation_and_ltm.py \\
        --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt \\
        --wfe_pred_ar outputs/external_eval_30k/wfe_ar/item_level_predictions.tsv \\
        --noise_metrics outputs/wm_noise_stress_wfe/noise_sweep_item_metrics.tsv \\
        --out_dir outputs/route_ltm_audit \\
        --noise_levels 0.0 0.2 0.5 1.0 \\
        --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.external_eval import load_model_and_vocab
from evaluate.hooks import make_batch

# ─────────────────────────────────────────────────────────────── constants ──

_SEEN_CAT    = "real_word_seen_in_training_lexicon"
_PSEUDO_LEX  = {"pseudo", "pseudoword"}
SHORT_LENGTHS = {3, 4, 5}
LONG_LENGTHS  = {7, 8, 9}
ROUTES        = ("full", "wm", "ltm")
BATCH_SIZE    = 128

_GROUP_DISPLAY = {
    "train_seen_real": "Train-seen real words",
    "pseudoword":      "Pseudowords",
    "unseen_forms":    "Unseen forms",
}
_GROUP_COLORS = {
    "train_seen_real": "#2ca02c",
    "pseudoword":      "#1f77b4",
    "unseen_forms":    "#d62728",
}
_ROUTE_COLORS = {"full": "#2ca02c", "wm": "#1f77b4", "ltm": "#d62728"}
_ROUTE_LABELS = {"full": "Full (gated)", "wm": "WM (dorsal)", "ltm": "LTM (ventral)"}

# ─────────────────────────────────────────────────────── group annotation ───

def _is_pseudo(df: pd.DataFrame) -> pd.Series:
    return df["lexicality"].str.lower().str.strip().isin(_PSEUDO_LEX)


def _annotate_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    is_seen   = df["lexicon_category"] == _SEEN_CAT
    is_pseudo = _is_pseudo(df)
    df["group_dager"]       = np.where(is_seen, "train_seen_real",
                               np.where(is_pseudo, "pseudoword", "excluded"))
    df["group_seen_unseen"] = np.where(is_seen, "train_seen_real", "unseen_forms")
    return df


def _ensure_groups(df: pd.DataFrame) -> pd.DataFrame:
    if "group_dager" not in df.columns:
        if "lexicon_category" not in df.columns:
            raise ValueError("Cannot annotate groups: missing lexicon_category column.")
        df = _annotate_groups(df)
    return df


def _length_bin(length: float) -> Optional[str]:
    if length in SHORT_LENGTHS:
        return "short"
    if length in LONG_LENGTHS:
        return "long"
    return None


# ─────────────────────────────────────────── Levenshtein with op-counting ───

def _count_edit_ops(a: list, b: list) -> Tuple[int, int, int]:
    """Levenshtein traceback: return (n_sub, n_ins, n_del).
    a = target (reference), b = prediction.
    Insertion: extra phoneme in prediction; deletion: missing from prediction."""
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 0, 0, 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    n_sub = n_ins = n_del = 0
    i, j = n, m
    while i > 0 or j > 0:
        if (i > 0 and j > 0 and a[i - 1] == b[j - 1]
                and dp[i][j] == dp[i - 1][j - 1]):
            i -= 1; j -= 1                            # match
        elif (i > 0 and j > 0
              and dp[i][j] == dp[i - 1][j - 1] + 1):
            n_sub += 1; i -= 1; j -= 1               # substitution
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            n_del += 1; i -= 1                        # deletion
        else:
            n_ins += 1; j -= 1                        # insertion
    return n_sub, n_ins, n_del


# ══════════════════════════════════════════════════════════════════ PART A ══
# Route compensation / full-route robustness

_RESCUE_GROUPS = ["train_seen_real", "pseudoword", "unseen_forms"]


def _pattern_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add Boolean columns for each route-outcome pattern."""
    df = df.copy()
    fc = df["full_exact_match"].astype(bool)
    wc = df["wm_exact_match"].astype(bool)
    lc = df["ltm_exact_match"].astype(bool)
    df["pat_full_correct"]       = fc
    df["pat_wm_correct"]         = wc
    df["pat_ltm_correct"]        = lc
    df["pat_full_ok_wm_fail"]    = fc & ~wc          # full rescues WM failure
    df["pat_full_ok_ltm_fail"]   = fc & ~lc          # full rescues LTM failure
    df["pat_full_ok_both_fail"]  = fc & ~wc & ~lc    # both fail, full succeeds
    df["pat_wm_ok_full_fail"]    = wc & ~fc
    df["pat_ltm_ok_full_fail"]   = lc & ~fc
    df["pat_both_ok_full_fail"]  = wc & lc & ~fc
    df["pat_all_correct"]        = fc & wc & lc
    df["pat_all_wrong"]          = ~fc & ~wc & ~lc
    df["length_bin"] = df["length_phonemes"].apply(_length_bin)
    return df


def _rescue_rates_by_noise_group(df_noise: pd.DataFrame,
                                  noise_levels: List[float]) -> pd.DataFrame:
    df_f = df_noise[df_noise["noise_level"].isin(noise_levels)].copy()
    df_f = _pattern_flags(df_f)
    pat_cols = [c for c in df_f.columns if c.startswith("pat_")]

    rows = []
    for (nl, rep, grp_col, grp_key), sub in df_f.groupby(
            ["noise_level", "repeat", "_grp_col", "_grp_key"],
            observed=True, sort=False):
        row = {"noise_level": nl, "repeat": int(rep),
               "group_mode": grp_col, "group_name": grp_key,
               "n_items": len(sub)}
        for pat in pat_cols:
            row[pat] = round(float(sub[pat].mean()), 5)
        rows.append(row)
    return pd.DataFrame(rows)


def _prepare_noise_df(df_noise: pd.DataFrame) -> pd.DataFrame:
    df_noise = _ensure_groups(df_noise.copy())
    # Melt into long format with one row per (item, noise_level, repeat, group_mode, group)
    rows = []
    df_noise["_row"] = range(len(df_noise))
    for _, row in df_noise.iterrows():
        for grp_col, grp_key in [
            ("dager",        row.get("group_dager", "unknown")),
            ("seen_unseen",  row.get("group_seen_unseen", "unknown")),
        ]:
            if grp_key in ("excluded", "unknown"):
                continue
            rows.append({**row.to_dict(), "_grp_col": grp_col, "_grp_key": grp_key})
    return pd.DataFrame(rows)


def part_a(df_noise: pd.DataFrame, noise_levels: List[float],
           out_dir: str) -> pd.DataFrame:
    """Route compensation analysis — returns confidence df for Part D join."""
    print("\n[Part A] Route compensation analysis …")
    df_long = _prepare_noise_df(df_noise)

    pat_cols = ["pat_full_correct", "pat_wm_correct", "pat_ltm_correct",
                "pat_full_ok_wm_fail", "pat_full_ok_ltm_fail",
                "pat_full_ok_both_fail", "pat_wm_ok_full_fail",
                "pat_ltm_ok_full_fail", "pat_both_ok_full_fail",
                "pat_all_correct", "pat_all_wrong"]

    df_f = df_long[df_long["noise_level"].isin(noise_levels)].copy()
    df_f = _pattern_flags(df_f)

    # ── rescue summary: per (noise, group, repeat) ───────────────────────────
    summary_rows = []
    for (nl, rep, grp_col, grp_key), sub in df_f.groupby(
            ["noise_level", "repeat", "_grp_col", "_grp_key"],
            observed=True, sort=False):
        r = {"noise_level": nl, "repeat": int(rep),
             "group_mode": grp_col, "group_name": grp_key, "n_items": len(sub)}
        for pat in pat_cols:
            r[pat] = round(float(sub[pat].mean()), 5)
        summary_rows.append(r)
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(out_dir, "route_rescue_summary.tsv"),
                      sep="\t", index=False)
    print(f"  [tsv] route_rescue_summary.tsv  ({len(df_summary)} rows)")

    # ── rescue by length bin ──────────────────────────────────────────────────
    df_fb = df_f[df_f["length_bin"].notna()]
    len_rows = []
    for (nl, rep, grp_col, grp_key, lb), sub in df_fb.groupby(
            ["noise_level", "repeat", "_grp_col", "_grp_key", "length_bin"],
            observed=True, sort=False):
        r = {"noise_level": nl, "repeat": int(rep),
             "group_mode": grp_col, "group_name": grp_key,
             "length_bin": lb, "n_items": len(sub)}
        for pat in pat_cols:
            r[pat] = round(float(sub[pat].mean()), 5)
        len_rows.append(r)
    df_len = pd.DataFrame(len_rows)
    df_len.to_csv(os.path.join(out_dir, "route_rescue_by_length.tsv"),
                  sep="\t", index=False)
    print(f"  [tsv] route_rescue_by_length.tsv  ({len(df_len)} rows)")

    # ── examples: full rescued, WM failed (highest noise level, pseudowords) ─
    max_nl = max(nl for nl in noise_levels if nl > 0) if any(nl > 0 for nl in noise_levels) else noise_levels[-1]
    ex_cols = ["noise_level", "repeat", "word", "lexicality", "lexicon_category",
               "length_phonemes", "group_dager", "group_seen_unseen",
               "full_exact_match", "wm_exact_match", "ltm_exact_match",
               "full_edit_dist", "wm_edit_dist", "ltm_edit_dist"]
    ex_cols_present = [c for c in ex_cols if c in df_f.columns]
    df_ex = (df_f[(df_f["noise_level"] == max_nl) &
                  (df_f["pat_full_ok_wm_fail"] == True)]
             [ex_cols_present]
             .sort_values(["group_dager", "wm_edit_dist"], ascending=[True, False])
             .head(200))
    df_ex.to_csv(os.path.join(out_dir, "route_rescue_examples.tsv"),
                 sep="\t", index=False)
    print(f"  [tsv] route_rescue_examples.tsv  ({len(df_ex)} rows, noise={max_nl})")

    # ── figures ───────────────────────────────────────────────────────────────
    _fig_rescue_vs_noise(df_summary, noise_levels, out_dir)
    _fig_accuracy_gap(df_summary, noise_levels, out_dir)
    _fig_rescue_by_length(df_len, noise_levels, out_dir)

    return df_summary


def _agg_rescue(df_summary: pd.DataFrame, noise_levels, grp_col, grp_key,
                pat_col: str):
    """Return (means, stds) across repeats for each noise level."""
    sub = df_summary[(df_summary["group_mode"] == grp_col) &
                     (df_summary["group_name"] == grp_key)]
    means, stds = [], []
    for nl in noise_levels:
        vals = sub[sub["noise_level"] == nl][pat_col].dropna().values
        means.append(float(np.mean(vals)) if len(vals) > 0 else np.nan)
        stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
    return np.array(means), np.array(stds)


def _fig_rescue_vs_noise(df_summary: pd.DataFrame, noise_levels, out_dir: str):
    groups = [("dager", "train_seen_real"), ("dager", "pseudoword"),
              ("seen_unseen", "unseen_forms")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for ax, (grp_col, grp_key) in zip(axes, groups):
        means, stds = _agg_rescue(df_summary, noise_levels,
                                   grp_col, grp_key, "pat_full_ok_wm_fail")
        ax.plot(noise_levels, means, "-o", lw=1.8, color="#9467bd",
                label="Full correct, WM wrong")
        ax.fill_between(noise_levels, means - stds, means + stds,
                        color="#9467bd", alpha=0.2)
        means2, stds2 = _agg_rescue(df_summary, noise_levels,
                                     grp_col, grp_key, "pat_full_ok_both_fail")
        ax.plot(noise_levels, means2, "--s", lw=1.4, color="#e377c2",
                label="Full correct, both routes wrong")
        ax.fill_between(noise_levels, means2 - stds2, means2 + stds2,
                        color="#e377c2", alpha=0.15)
        ax.set_xlabel("WM/dorsal noise level", fontsize=9)
        ax.set_ylabel("Rescue rate", fontsize=9)
        ax.set_title(_GROUP_DISPLAY.get(grp_key, grp_key), fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
    fig.suptitle("Full-route rescue rate vs WM noise  (±1 SD across repeats)",
                 fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "route_rescue_vs_noise.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [fig] {path}")


def _fig_accuracy_gap(df_summary: pd.DataFrame, noise_levels, out_dir: str):
    groups = [("dager", "train_seen_real"), ("dager", "pseudoword"),
              ("seen_unseen", "unseen_forms")]
    fig, ax = plt.subplots(figsize=(8, 4))
    for grp_col, grp_key in groups:
        fm, fs = _agg_rescue(df_summary, noise_levels, grp_col, grp_key, "pat_full_correct")
        wm, ws = _agg_rescue(df_summary, noise_levels, grp_col, grp_key, "pat_wm_correct")
        gap  = fm - wm
        color = _GROUP_COLORS.get(grp_key, "gray")
        ax.plot(noise_levels, gap, "-o", lw=1.8, color=color,
                label=_GROUP_DISPLAY.get(grp_key, grp_key))
        err_gap = np.sqrt(fs**2 + ws**2)
        ax.fill_between(noise_levels, gap - err_gap, gap + err_gap,
                        color=color, alpha=0.15)
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("WM/dorsal noise level", fontsize=10)
    ax.set_ylabel("Full accuracy − WM accuracy", fontsize=10)
    ax.set_title("Full/gated robustness advantage over WM  (positive = full more robust)",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "full_vs_wm_accuracy_gap.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [fig] {path}")


def _fig_rescue_by_length(df_len: pd.DataFrame, noise_levels, out_dir: str):
    max_nl = max(noise_levels)
    sub_nl = df_len[df_len["noise_level"] == max_nl]
    groups = [("dager", "pseudoword"), ("seen_unseen", "unseen_forms")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, (grp_col, grp_key) in zip(axes, groups):
        for lb, marker, lw in [("short", "o", 2.0), ("long", "s", 2.0)]:
            sub_g = sub_nl[(sub_nl["group_mode"] == grp_col) &
                           (sub_nl["group_name"] == grp_key) &
                           (sub_nl["length_bin"] == lb)]
            means, stds = [], []
            for rep, sg in sub_g.groupby("repeat"):
                means.append(float(sg["pat_full_ok_wm_fail"].mean()))
            if means:
                ax.bar(
                    ["Short\n(3–5 ph)", "Long\n(7–9 ph)"].index(
                        {"short": "Short\n(3–5 ph)", "long": "Long\n(7–9 ph)"}[lb]),
                    np.mean(means),
                    color="#9467bd" if lb == "short" else "#e377c2",
                    alpha=0.8, label=f"{lb.capitalize()} (mean)",
                    yerr=np.std(means) if len(means) > 1 else 0,
                    capsize=5
                )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Short\n(3–5 ph)", "Long\n(7–9 ph)"], fontsize=9)
        ax.set_ylabel("Rescue rate (full correct, WM wrong)", fontsize=8)
        ax.set_title(_GROUP_DISPLAY.get(grp_key, grp_key) +
                     f"\n(noise = {max_nl:.2f})", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)
    fig.suptitle("Route rescue rate: short vs long words at peak noise", fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "rescue_by_length.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [fig] {path}")


# ══════════════════════════════════════════════════════════════════ PART B ══
# LTM length sensitivity

def part_b(df_ar: pd.DataFrame, out_dir: str) -> None:
    print("\n[Part B] LTM length sensitivity …")
    df_ar = _ensure_groups(df_ar.copy())
    df_ar["length_bin"] = df_ar["length_phonemes"].apply(_length_bin)

    # ── per-length mean metrics ───────────────────────────────────────────────
    rows_len = []
    for route in ROUTES:
        for grp_col, grp_key in [("group_dager", "train_seen_real"),
                                  ("group_dager", "pseudoword"),
                                  ("group_seen_unseen", "unseen_forms")]:
            sub = df_ar[df_ar[grp_col] == grp_key]
            for L, sg in sub.groupby("length_phonemes"):
                n = len(sg)
                if n < 3:
                    continue
                rows_len.append({
                    "route": route, "group": grp_key,
                    "length_phonemes": int(L), "n_items": n,
                    "error_rate":    round(1.0 - float(sg[f"{route}_exact_match"].mean()), 5),
                    "edit_dist_mean": round(float(sg[f"{route}_edit_dist"].mean()), 5),
                })
    pd.DataFrame(rows_len).to_csv(
        os.path.join(out_dir, "ltm_length_by_group.tsv"), sep="\t", index=False)
    print(f"  [tsv] ltm_length_by_group.tsv  ({len(rows_len)} rows)")

    # ── error-type breakdown ──────────────────────────────────────────────────
    rows_ops = []
    for route in ROUTES:
        tgt_col  = f"{route}_target"
        pred_col = f"{route}_predicted"
        if tgt_col not in df_ar.columns or pred_col not in df_ar.columns:
            continue
        for grp_col, grp_key in [("group_dager", "train_seen_real"),
                                   ("group_dager", "pseudoword"),
                                   ("group_seen_unseen", "unseen_forms")]:
            sub = df_ar[df_ar[grp_col] == grp_key].copy()
            sub["length_bin"] = sub["length_phonemes"].apply(_length_bin)
            for lb, sg in sub.groupby("length_bin"):
                if lb is None or len(sg) < 3:
                    continue
                total_sub = total_ins = total_del = n_items = 0
                for _, row in sg.iterrows():
                    if not isinstance(row[tgt_col], str):
                        continue
                    tgt  = row[tgt_col].split()
                    pred = row[pred_col].split() if isinstance(row[pred_col], str) else []
                    s, i, d = _count_edit_ops(tgt, pred)
                    total_sub += s; total_ins += i; total_del += d
                    n_items += 1
                if n_items == 0:
                    continue
                rows_ops.append({
                    "route": route, "group": grp_key, "length_bin": lb,
                    "n_items": n_items,
                    "substitutions":  total_sub,
                    "insertions":     total_ins,
                    "deletions":      total_del,
                    "total_ops":      total_sub + total_ins + total_del,
                    "sub_per_item":   round(total_sub / n_items, 4),
                    "ins_per_item":   round(total_ins / n_items, 4),
                    "del_per_item":   round(total_del / n_items, 4),
                })
    pd.DataFrame(rows_ops).to_csv(
        os.path.join(out_dir, "ltm_error_types_by_length.tsv"), sep="\t", index=False)
    print(f"  [tsv] ltm_error_types_by_length.tsv  ({len(rows_ops)} rows)")

    # ── position-wise accuracy ────────────────────────────────────────────────
    rows_pos = []
    for route in ROUTES:
        tgt_col  = f"{route}_target"
        pred_col = f"{route}_predicted"
        if tgt_col not in df_ar.columns or pred_col not in df_ar.columns:
            continue
        for grp_col, grp_key in [("group_dager", "train_seen_real"),
                                   ("group_dager", "pseudoword"),
                                   ("group_seen_unseen", "unseen_forms")]:
            sub = df_ar[df_ar[grp_col] == grp_key]
            max_L = int(df_ar["length_phonemes"].max())
            counts = np.zeros(max_L)
            totals = np.zeros(max_L)
            for _, row in sub.iterrows():
                if not isinstance(row[tgt_col], str):
                    continue
                tgt  = row[tgt_col].split()
                pred = row[pred_col].split() if isinstance(row[pred_col], str) else []
                for pos in range(len(tgt)):
                    totals[pos] += 1
                    if pos < len(pred) and pred[pos] == tgt[pos]:
                        counts[pos] += 1
            for pos in range(max_L):
                if totals[pos] < 3:
                    continue
                rows_pos.append({
                    "route": route, "group": grp_key,
                    "position_1indexed": pos + 1,
                    "accuracy": round(float(counts[pos] / totals[pos]), 5),
                    "n_items": int(totals[pos]),
                })
    pd.DataFrame(rows_pos).to_csv(
        os.path.join(out_dir, "ltm_position_accuracy.tsv"), sep="\t", index=False)
    print(f"  [tsv] ltm_position_accuracy.tsv  ({len(rows_pos)} rows)")

    # ── figures ───────────────────────────────────────────────────────────────
    _fig_ltm_edit_by_length(df_ar, out_dir)
    _fig_ltm_error_types(pd.DataFrame(rows_ops), out_dir)
    _fig_position_accuracy(pd.DataFrame(rows_pos), out_dir)


def _fig_ltm_edit_by_length(df_ar: pd.DataFrame, out_dir: str):
    panels = [("group_dager", "train_seen_real", "Train-seen real words"),
              ("group_dager", "pseudoword",      "Pseudowords")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (grp_col, grp_key, display) in zip(axes, panels):
        sub = df_ar[df_ar[grp_col] == grp_key]
        for route in ROUTES:
            agg = sub.groupby("length_phonemes")[f"{route}_edit_dist"].mean()
            ax.plot(agg.index, agg.values, "-o", lw=1.8, ms=5,
                    color=_ROUTE_COLORS[route], label=_ROUTE_LABELS[route])
        ax.set_xlabel("Word length (phonemes)", fontsize=10)
        ax.set_ylabel("Mean edit distance", fontsize=10)
        ax.set_title(display, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Edit distance by word length — AR decoding, no noise", fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "ltm_vs_wm_edit_by_length.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [fig] {path}")


def _fig_ltm_error_types(df_ops: pd.DataFrame, out_dir: str):
    if df_ops.empty:
        print("  [skip] ltm_error_types_by_length.png — no data"); return
    sub = df_ops[(df_ops["route"] == "ltm") &
                 (df_ops["group"] == "pseudoword") &
                 (df_ops["length_bin"].notna())]
    if sub.empty:
        print("  [skip] ltm_error_types_by_length.png — no pseudoword LTM data"); return
    bins = sorted(sub["length_bin"].unique())
    x = np.arange(len(bins))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (op_col, label, color) in enumerate([
            ("sub_per_item", "Substitutions", "#d62728"),
            ("ins_per_item", "Insertions",    "#1f77b4"),
            ("del_per_item", "Deletions",     "#2ca02c")]):
        vals = [float(sub[sub["length_bin"] == lb][op_col].mean()
                      if not sub[sub["length_bin"] == lb].empty else 0)
                for lb in bins]
        ax.bar(x + i * width, vals, width, color=color, alpha=0.85, label=label)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{b.capitalize()}\n(3–5 ph)" if b == "short"
                        else f"{b.capitalize()}\n(7–9 ph)" for b in bins], fontsize=9)
    ax.set_ylabel("Operations per item (LTM — pseudowords)", fontsize=9)
    ax.set_title("LTM error types by length bin — pseudowords, AR decoding", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "ltm_error_types_by_length.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [fig] {path}")


def _fig_position_accuracy(df_pos: pd.DataFrame, out_dir: str):
    if df_pos.empty:
        print("  [skip] ltm_position_accuracy_pseudowords.png — no data"); return
    sub = df_pos[df_pos["group"] == "pseudoword"]
    if sub.empty:
        print("  [skip] ltm_position_accuracy_pseudowords.png — no pseudoword data"); return
    fig, ax = plt.subplots(figsize=(9, 4))
    for route in ROUTES:
        sg = sub[sub["route"] == route].sort_values("position_1indexed")
        if sg.empty:
            continue
        ax.plot(sg["position_1indexed"], sg["accuracy"], "-o", lw=1.8, ms=4,
                color=_ROUTE_COLORS[route], label=_ROUTE_LABELS[route])
    ax.set_xlabel("Phoneme position (1-indexed, relative to item start)", fontsize=9)
    ax.set_ylabel("Proportion correct", fontsize=9)
    ax.set_title("Position-wise accuracy — Pseudowords, AR decoding, no noise", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path = os.path.join(out_dir, "ltm_position_accuracy_pseudowords.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [fig] {path}")


# ════════════════════════════════════════════════════════════════ PARTS C+D ══
# LTM semantic confidence + gate audit

@torch.no_grad()
def _run_ltm_confidence(model, vocab, df_ar: pd.DataFrame,
                         device: str) -> pd.DataFrame:
    """Run LTM encoder on every WFE item; return df with confidence columns."""
    # Build forms_ids from target_phonemes or the 'target' decoded column
    forms_ids: List[List[int]] = []
    for _, row in df_ar.iterrows():
        src = row.get("target_phonemes", None) or row.get("ltm_target", "")
        syms = str(src).split()
        ids  = [vocab.stoi[s] for s in syms if s in vocab.stoi]
        forms_ids.append(ids if ids else [vocab.eos_id])

    all_conf, all_margin, all_density = [], [], []
    all_top1, all_top2, all_top3 = [], [], []

    n = len(forms_ids)
    for start in range(0, n, BATCH_SIZE):
        batch_forms = forms_ids[start: start + BATCH_SIZE]
        batch = make_batch(batch_forms, vocab, device)
        s_hat = model.ltm.encode(batch["enc_in"], batch["enc_mask"])
        field = model.ltm.lexical_field(s_hat)

        conf    = field["confidence"].cpu().numpy()
        margin  = field["margin"].cpu().numpy()
        density = field["density"].cpu().numpy()
        sims    = field["sims"].cpu().numpy()

        k = min(3, sims.shape[1])
        top_k = np.sort(sims, axis=1)[:, -k:][:, ::-1]  # (B, k) descending

        all_conf.extend(conf.tolist())
        all_margin.extend(margin.tolist())
        all_density.extend(density.tolist())
        all_top1.extend(top_k[:, 0].tolist())
        all_top2.extend((top_k[:, 1].tolist() if k >= 2 else [np.nan] * len(conf)))
        all_top3.extend((top_k[:, 2].tolist() if k >= 3 else [np.nan] * len(conf)))

        if start % (BATCH_SIZE * 10) == 0:
            print(f"    … {min(start + BATCH_SIZE, n)}/{n}", end="\r")
    print()

    df_out = df_ar.copy()
    df_out["ltm_confidence"]   = all_conf
    df_out["ltm_margin"]       = all_margin
    df_out["ltm_density"]      = all_density
    df_out["ltm_top1_cosim"]   = all_top1
    df_out["ltm_top2_cosim"]   = all_top2
    df_out["ltm_top3_cosim"]   = all_top3
    # Gate value: g = sigmoid(alpha * (confidence - 0.5))
    alpha = float(model.gate.cfg.alpha)
    conf_arr = np.array(all_conf)
    df_out["gate_value"] = 1.0 / (1.0 + np.exp(-alpha * (conf_arr - 0.5)))
    df_out.attrs["gate_alpha"] = alpha
    return df_out


def part_cd(df_ar: pd.DataFrame, model, vocab, device: str,
            out_dir: str, noise_levels: List[float],
            df_rescue: Optional[pd.DataFrame] = None) -> None:
    print("\n[Parts C+D] LTM semantic confidence + gate audit …")
    df_ar = _ensure_groups(df_ar.copy())

    df_conf = _run_ltm_confidence(model, vocab, df_ar, device)
    alpha = df_conf.attrs.get("gate_alpha", 4.0)
    print(f"  gate alpha = {alpha}")

    # ── confidence TSV ────────────────────────────────────────────────────────
    conf_cols = ["item_id", "word", "lexicality", "lexicon_category",
                 "length_phonemes", "group_dager", "group_seen_unseen",
                 "ltm_confidence", "ltm_margin", "ltm_density",
                 "ltm_top1_cosim", "ltm_top2_cosim", "ltm_top3_cosim",
                 "gate_value",
                 "ltm_exact_match", "ltm_edit_dist",
                 "wm_exact_match", "wm_edit_dist",
                 "full_exact_match", "full_edit_dist"]
    conf_cols_present = [c for c in conf_cols if c in df_conf.columns]
    df_conf[conf_cols_present].to_csv(
        os.path.join(out_dir, "ltm_semantic_confidence.tsv"), sep="\t", index=False)
    print(f"  [tsv] ltm_semantic_confidence.tsv")

    # ── gate by group, length bin ─────────────────────────────────────────────
    df_conf["length_bin"] = df_conf["length_phonemes"].apply(_length_bin)
    rows_gate = []
    for grp_col, grp_key in [("group_dager", "train_seen_real"),
                               ("group_dager", "pseudoword"),
                               ("group_seen_unseen", "unseen_forms")]:
        sub = df_conf[df_conf[grp_col] == grp_key]
        for lb in (None, "short", "long"):
            sg = sub if lb is None else sub[sub["length_bin"] == lb]
            if len(sg) < 3:
                continue
            rows_gate.append({
                "group": grp_key, "length_bin": lb or "all",
                "n_items":     len(sg),
                "gate_mean":   round(float(sg["gate_value"].mean()), 5),
                "gate_std":    round(float(sg["gate_value"].std()), 5),
                "gate_median": round(float(sg["gate_value"].median()), 5),
                "conf_mean":   round(float(sg["ltm_confidence"].mean()), 5),
                "conf_std":    round(float(sg["ltm_confidence"].std()), 5),
                "note":        (f"gate = sigmoid({alpha}*(conf-0.5)); "
                                "gate>0.5 means LTM dominates"),
            })
    # Gate doesn't change with WM noise — note this explicitly
    pd.DataFrame(rows_gate).to_csv(
        os.path.join(out_dir, "gate_by_noise_group_length.tsv"), sep="\t", index=False)
    print("  [tsv] gate_by_noise_group_length.tsv")
    print(f"  NOTE: gate = sigmoid({alpha}*(confidence-0.5)) "
          "depends only on LTM confidence → invariant to WM noise level.")

    # ── gate on rescued items ─────────────────────────────────────────────────
    if "item_id" in df_conf.columns:
        df_conf["rescued"] = df_conf["full_exact_match"].astype(bool) & ~df_conf["wm_exact_match"].astype(bool)
        gate_rescue_rows = []
        for grp_col, grp_key in [("group_dager", "train_seen_real"),
                                   ("group_dager", "pseudoword"),
                                   ("group_seen_unseen", "unseen_forms")]:
            sub = df_conf[df_conf[grp_col] == grp_key]
            for rescued, sg in sub.groupby("rescued"):
                if len(sg) < 3:
                    continue
                gate_rescue_rows.append({
                    "group": grp_key,
                    "rescued_at_no_noise": bool(rescued),
                    "n_items":   len(sg),
                    "gate_mean": round(float(sg["gate_value"].mean()), 5),
                    "gate_std":  round(float(sg["gate_value"].std()), 5),
                    "conf_mean": round(float(sg["ltm_confidence"].mean()), 5),
                    "note": ("rescued=True: full correct at no-noise baseline "
                             "while WM route was wrong"),
                })
        pd.DataFrame(gate_rescue_rows).to_csv(
            os.path.join(out_dir, "gate_on_rescued_items.tsv"), sep="\t", index=False)
        print("  [tsv] gate_on_rescued_items.tsv")

    # ── figures ───────────────────────────────────────────────────────────────
    _fig_confidence_vs_length(df_conf, out_dir)
    _fig_confidence_vs_edit(df_conf, out_dir)
    _fig_gate_by_group(pd.DataFrame(rows_gate), alpha, out_dir)
    if "rescued" in df_conf.columns:
        _fig_gate_rescue(df_conf, out_dir)


def _fig_confidence_vs_length(df_conf: pd.DataFrame, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (grp_col, grp_key, display) in zip(axes, [
            ("group_dager", "train_seen_real", "Train-seen real words"),
            ("group_dager", "pseudoword",      "Pseudowords")]):
        sub = df_conf[df_conf[grp_col] == grp_key]
        lengths = sorted(sub["length_phonemes"].unique())
        means = [sub[sub["length_phonemes"] == L]["ltm_confidence"].mean()
                 for L in lengths]
        stds  = [sub[sub["length_phonemes"] == L]["ltm_confidence"].std()
                 for L in lengths]
        ax.plot(lengths, means, "-o", lw=1.8, color="#8856a7")
        ax.fill_between(lengths,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        color="#8856a7", alpha=0.2)
        ax.set_xlabel("Word length (phonemes)", fontsize=10)
        ax.set_ylabel("LTM confidence (max cosine sim)", fontsize=10)
        ax.set_title(display, fontsize=9)
        ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5,
                   label="g=0.5 threshold")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
    fig.suptitle("LTM lexical confidence vs word length  (±1 SD)  "
                 "·  confidence > 0.5 → gate favours LTM", fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "ltm_confidence_vs_length.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [fig] {path}")


def _fig_confidence_vs_edit(df_conf: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    for grp_col, grp_key in [("group_dager", "train_seen_real"),
                               ("group_dager", "pseudoword"),
                               ("group_seen_unseen", "unseen_forms")]:
        sub = df_conf[df_conf[grp_col] == grp_key].dropna(
            subset=["ltm_confidence", "ltm_edit_dist"])
        if len(sub) < 5:
            continue
        ax.scatter(sub["ltm_edit_dist"], sub["ltm_confidence"],
                   s=10, alpha=0.35, color=_GROUP_COLORS.get(grp_key, "gray"),
                   label=_GROUP_DISPLAY.get(grp_key, grp_key))
    ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5,
               label="g=0.5 threshold")
    ax.set_xlabel("LTM edit distance (AR decoding, no noise)", fontsize=10)
    ax.set_ylabel("LTM confidence (max cosine sim)", fontsize=10)
    ax.set_title("LTM semantic confidence vs output quality  "
                 "(high confidence but high error = lexicalization artifact)",
                 fontsize=8)
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "ltm_confidence_vs_edit_distance.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [fig] {path}")


def _fig_gate_by_group(df_gate: pd.DataFrame, alpha: float, out_dir: str):
    if df_gate.empty:
        return
    groups   = ["train_seen_real", "pseudoword", "unseen_forms"]
    len_bins = ["all", "short", "long"]
    x = np.arange(len(groups))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))
    lb_colors = {"all": "#8856a7", "short": "#1f77b4", "long": "#d62728"}
    for i, lb in enumerate(len_bins):
        sub = df_gate[df_gate["length_bin"] == lb]
        means = []
        stds  = []
        for grp in groups:
            row = sub[sub["group"] == grp]
            means.append(float(row["gate_mean"].iloc[0]) if len(row) > 0 else np.nan)
            stds.append(float(row["gate_std"].iloc[0]) if len(row) > 0 else 0.0)
        ax.bar(x + i * width, means, width, color=lb_colors[lb],
               alpha=0.85, label=lb.capitalize(),
               yerr=stds, capsize=4)
    ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5,
               label="g=0.5 (equal weighting)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([_GROUP_DISPLAY.get(g, g) for g in groups],
                       fontsize=8, rotation=10, ha="right")
    ax.set_ylabel(f"Gate value  g = σ({alpha}·(conf−0.5))", fontsize=9)
    ax.set_title("Gate value by group and length bin  "
                 "(invariant to WM noise — depends only on LTM confidence)",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "gate_by_noise_group_length.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [fig] {path}")


def _fig_gate_rescue(df_conf: pd.DataFrame, out_dir: str):
    groups = ["train_seen_real", "pseudoword", "unseen_forms"]
    grp_cols = {"train_seen_real": "group_dager",
                "pseudoword":      "group_dager",
                "unseen_forms":    "group_seen_unseen"}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(groups))
    width = 0.35
    for i, rescued in [(0, False), (1, True)]:
        means, stds = [], []
        for grp_key in groups:
            grp_col = grp_cols[grp_key]
            sub = df_conf[(df_conf[grp_col] == grp_key) &
                          (df_conf["rescued"] == rescued)]
            means.append(float(sub["gate_value"].mean()) if len(sub) >= 3 else np.nan)
            stds.append(float(sub["gate_value"].std())   if len(sub) >= 3 else 0.0)
        label = ("Full correct, WM correct (no rescue)" if not rescued
                 else "Full correct, WM wrong (full rescues WM, no-noise)")
        color = "#2ca02c" if not rescued else "#9467bd"
        ax.bar(x + i * width, means, width, color=color, alpha=0.85,
               label=label, yerr=stds, capsize=4)
    ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(x + 0.5 * width)
    ax.set_xticklabels([_GROUP_DISPLAY.get(g, g) for g in groups],
                       fontsize=8, rotation=10, ha="right")
    ax.set_ylabel("Gate value (at no-noise baseline)", fontsize=9)
    ax.set_title("Gate value: rescued items vs non-rescued items  "
                 "(higher gate = LTM more dominant = LTM covers WM gap)",
                 fontsize=8)
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "gate_on_rescued_items.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [fig] {path}")


# ══════════════════════════════════════════════════════════════════ README ══

_README = """\
# Route compensation and LTM length sensitivity — audit

## Why this audit exists

Two findings from the baseline and noise-sweep analyses require explanation:

1. **Full/gated route robustness**: the full (gated) route outperforms the
   isolated WM route under WM/dorsal noise, even for pseudowords.

2. **LTM length sensitivity**: the LTM (ventral) route shows a much stronger
   pseudoword length effect than the WM (dorsal) route in no-noise AR decoding.

This audit collects evidence to assess competing explanations.

---

## Part A — What "full robustness" means mechanistically

The gate value g = σ(α·(confidence − 0.5)) routes the model between WM and LTM.
For **train-seen real words**, LTM confidence is typically high (the phoneme
sequence projects close to a known semantic neighbor) → g ≈ 1 → full ≈ LTM.
Since WM noise does not affect the LTM route at all, the full route inherits
LTM's noise immunity for high-confidence items.

For **pseudowords**, confidence is low (no true semantic neighbor exists) →
g < 0.5 → WM dominates. However, g is never exactly 0: even a small LTM
contribution stabilizes the output and partially offsets WM errors.  This
partial LTM anchor is the likely explanation for why full > WM even for
pseudowords under noise.

**Interpretation of rescue rate**: items where full_correct=1 and wm_correct=0
are items where the LTM contribution was decisive. A rising rescue rate with
noise confirms that the gate's LTM anchor grows in importance as WM degrades.

**Caveat**: word-level rescue statistics can be misleading — a full-correct item
could be correct for different phoneme positions than the WM-correct item would
have been, masking position-level complementarity.

---

## Part B — Why LTM is more length-sensitive than WM

The LTM encoder is a biGRU with masked mean-pooling.  Unlike the WM encoder
(unidirectional GRU with pack_padded_sequence), the LTM backward direction
accumulates hidden state starting from the rightmost PAD position in a
padded batch.  This makes s_hat (the semantic representation) shift with the
batch's maximum sequence length — a known artifact of biGRU without
pack_padded_sequence.

Additionally, the LTM architecture compresses the entire phonological form
into a single 300-d vector s_hat (matching GloVe space).  For real words this
is disambiguated by a nearby semantic neighbor.  For pseudowords there is no
true semantic target; the projected s_hat lands in a diffuse region of the
bank, and the reconstructed form is less constrained.  Longer pseudowords
have more complex phonological structure to compress, increasing reconstruction
difficulty — a genuine capacity limit of the mean-pool encoder.

**Caveat**: GloVe is NOT used as input at WFE inference time.  The semantic
bank exists only to compute the gate value and to provide the LTM decoder
initial state h0.  Pseudowords do not have GloVe vectors; the model maps them
to the nearest training-lexicon semantic neighbor (which is likely wrong).

**Caveat**: error type breakdown (substitutions / insertions / deletions) is
a further decomposition that can reveal whether LTM failures on long items
are systemic (e.g., consistent deletion of final phonemes → truncation) or
diffuse (random substitutions at all positions).

---

## Part C — LTM semantic confidence

LTM confidence = max cosine similarity of s_hat to the frozen semantic bank.
This is the main signal the gate uses.  High confidence means the phoneme
input is projected close to a known word; low confidence means the input is
novel / unseen.

For pseudowords, we expect:
  - lower confidence than real words (no true semantic target)
  - possibly decreasing confidence with length (longer pseudowords project
    further from the known semantic manifold)

For the relationship to output quality:
  - items with HIGH confidence but HIGH LTM edit distance are diagnostic:
    they suggest lexicalization — the LTM is confidently mapping the input
    to the wrong word's phonology.
  - items with LOW confidence AND HIGH edit distance are the expected failure
    mode: the gate correctly assigns low g but LTM still fails.

**Caveat**: pseudowords do not have true semantic targets.  Confidence scores
for pseudowords reflect proximity to the nearest REAL-WORD semantic neighbor,
not accuracy of semantic reconstruction.

---

## Part D — Gate audit

Gate value g = σ(α·(confidence − 0.5)) is derived from Part C confidence.
It does NOT change with WM noise level — the LTM encoder is unaffected by
WM interference noise, so the gate routing is frozen across all noise levels.

This has two implications:
  1. The gate's routing strategy is noise-insensitive by design.
  2. At high noise, pseudoword items (low g) suffer more from WM degradation
     than real-word items (high g → near-full LTM takeover).

Items where the full route rescues WM failure are predicted to have higher-
than-average gate values within their group (more LTM weight at the margin).

---

## Caveats

- This is an audit, not a final explanation.  All findings are associative.
- Checkpoint: lichtheim3_30k_glove_e60_to_e120_lowlr.pt
- Inference: LTM encoder only (for Parts C/D); no retraining.
- AR decoding, no noise, unless otherwise noted.

## TODO (requires model surgery — not done here)

- Expose per-item gate value during the full forward pass at WFE eval time
  so gate values can be aligned to specific noise_level/repeat runs.
- Add pack_padded_sequence to LTMLexicon.encode() in a FUTURE retrained
  architecture to test whether padding-sensitivity explains LTM length effects.
  (Cannot be tested on current checkpoint without retraining.)

## Files

| File | Contents |
|---|---|
| route_rescue_summary.tsv | Rescue rates per (noise, group, repeat) |
| route_rescue_by_length.tsv | Rescue rates per (noise, group, length_bin, repeat) |
| route_rescue_examples.tsv | Example rescued items at peak noise |
| ltm_length_by_group.tsv | Edit dist and error rate by (route, group, length) |
| ltm_error_types_by_length.tsv | Sub/ins/del per item by (route, group, length_bin) |
| ltm_position_accuracy.tsv | Position-wise accuracy by (route, group) |
| ltm_semantic_confidence.tsv | Per-item confidence, margin, density, gate value |
| gate_by_noise_group_length.tsv | Gate statistics by (group, length_bin) |
| gate_on_rescued_items.tsv | Gate statistics split by rescue pattern (no-noise) |
"""


def _write_readme(out_dir: str) -> None:
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(_README)
    print(f"  [readme] {os.path.join(out_dir, 'README.md')}")


# ═══════════════════════════════════════════════════════════════════ CLI ════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit route compensation and LTM length sensitivity.")
    p.add_argument("--ckpt",
                   default=os.path.join(ROOT, "checkpoints",
                                        "lichtheim3_30k_glove_e60_to_e120_lowlr.pt"))
    p.add_argument("--wfe_pred_ar",
                   default=os.path.join(ROOT, "outputs", "external_eval_30k",
                                        "wfe_ar", "item_level_predictions.tsv"))
    p.add_argument("--noise_metrics",
                   default=os.path.join(ROOT, "outputs", "wm_noise_stress_wfe",
                                        "noise_sweep_item_metrics.tsv"))
    p.add_argument("--out_dir",
                   default=os.path.join(ROOT, "outputs", "route_ltm_audit"))
    p.add_argument("--noise_levels", nargs="+", type=float,
                   default=[0.0, 0.2, 0.5, 1.0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    noise_levels = sorted(set(args.noise_levels))

    print("=" * 60)
    print("  Route compensation + LTM length sensitivity audit")
    print("=" * 60)
    print(f"  out_dir       : {args.out_dir}")
    print(f"  noise_levels  : {noise_levels}")
    print(f"  device        : {device}")

    # ── load AR predictions TSV ──────────────────────────────────────────────
    df_ar: Optional[pd.DataFrame] = None
    if os.path.exists(args.wfe_pred_ar):
        df_ar = pd.read_csv(args.wfe_pred_ar, sep="\t")
        df_ar = df_ar[df_ar["notes"].fillna("").str.strip() == ""].reset_index(drop=True)
        print(f"\n  AR predictions: {len(df_ar)} items  ({args.wfe_pred_ar})")
    else:
        print(f"\n  [skip] AR predictions not found: {args.wfe_pred_ar}")

    # ── load noise metrics TSV ────────────────────────────────────────────────
    df_noise: Optional[pd.DataFrame] = None
    if os.path.exists(args.noise_metrics):
        df_noise = pd.read_csv(args.noise_metrics, sep="\t")
        print(f"  Noise metrics : {len(df_noise)} rows  ({args.noise_metrics})")
        # Filter to requested noise levels
        avail = sorted(df_noise["noise_level"].unique())
        missing = [nl for nl in noise_levels if nl not in avail]
        if missing:
            print(f"  [warn] noise_levels {missing} not found in TSV "
                  f"(available: {avail})")
        noise_levels_present = [nl for nl in noise_levels if nl in avail]
    else:
        print(f"  [skip] Noise metrics not found: {args.noise_metrics}")
        noise_levels_present = []

    # ── load model (Parts C+D) ────────────────────────────────────────────────
    model = vocab = None
    if os.path.exists(args.ckpt):
        model, vocab, meta = load_model_and_vocab(args.ckpt, device)
    else:
        print(f"\n  [skip] Checkpoint not found: {args.ckpt}")
        print("  Parts C and D (LTM confidence + gate audit) will be skipped.")

    # ── Part A ────────────────────────────────────────────────────────────────
    df_rescue = None
    if df_noise is not None and noise_levels_present:
        df_rescue = part_a(df_noise, noise_levels_present, args.out_dir)
    else:
        print("\n[Part A] Skipped — noise metrics unavailable.")

    # ── Part B ────────────────────────────────────────────────────────────────
    if df_ar is not None:
        part_b(df_ar, args.out_dir)
    else:
        print("\n[Part B] Skipped — AR predictions unavailable.")

    # ── Parts C + D ───────────────────────────────────────────────────────────
    if model is not None and df_ar is not None:
        part_cd(df_ar, model, vocab, device, args.out_dir,
                noise_levels_present or noise_levels, df_rescue)
    else:
        print("\n[Parts C+D] Skipped — checkpoint or AR predictions unavailable.")
        print("  TODO: run with --ckpt and --wfe_pred_ar to enable gate/confidence audit.")

    # ── README ────────────────────────────────────────────────────────────────
    _write_readme(args.out_dir)

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Done.  Output files:")
    for f in sorted(os.listdir(args.out_dir)):
        fpath = os.path.join(args.out_dir, f)
        if os.path.isfile(fpath):
            kb = os.path.getsize(fpath) // 1024
            print(f"    {f}  ({kb} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
