"""WM/dorsal noise sweep for WFE evaluation.

Scientific context
------------------
Baseline analysis (no noise, teacher-forced and autoregressive) found:
  - Train-seen real words are near ceiling in both decoding regimes.
  - Pseudowords show a modest full/WM length effect (longer = harder).
  - Autoregressive decoding amplifies edit-distance length effects more than
    error-rate effects.
  - The LTM route is more length-sensitive than WM for pseudowords.

Next question: does explicit WM/dorsal noise amplify the pseudoword length
effect, and does it do so more strongly under autoregressive decoding?

Noise mechanism
---------------
WM interference noise is added to the phonological buffer hidden state `h`
after the GRU encoder and before the GRU decoder, in WMRecurrent.forward:

    if (self.training or collect) and self.cfg.interference_noise > 0:
        h = h + torch.randn_like(h) * self.cfg.interference_noise

Noise is enabled at eval time by passing collect=True to the WM route.  We
temporarily override model.wm.cfg.interference_noise to the target noise_level.
Model weights are never modified.  The noise path applies to the "wm" route
and to the WM component inside the "full" gated route (both use collect=True
when noise_level > 0).  The "ltm" route has no WM component and is unaffected
by noise (collect=False for ltm).

For autoregressive decoding: route_logits is called at every decoder step,
re-running the WM encoder with fresh noise each time.  This models noisy
readout from the phonological buffer at each production step.

Usage
-----
    python scripts/run_wm_noise_sweep_wfe.py

    python scripts/run_wm_noise_sweep_wfe.py \\
        --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt \\
        --out_dir outputs/wm_noise_sweep_wfe \\
        --decode autoregressive \\
        --noise_levels 0.0 0.01 0.03 0.05 0.10 \\
        --n_repeats 20 \\
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

from scripts.external_eval import (
    load_model_and_vocab,
    _wfe_lexicon_category,
    _edit_distance,
    DECODE_TF,
    DECODE_AR,
)
from evaluate.hooks import make_batch

# ─────────────────────────────────────────────────────────────── defaults ───

CKPT_DEFAULT = os.path.join(ROOT, "checkpoints",
                             "lichtheim3_30k_glove_e60_to_e120_lowlr.pt")
WFE_DEFAULT  = os.path.join(ROOT, "data", "eval_external", "wfe_eval.tsv")
OUT_DEFAULT  = os.path.join(ROOT, "outputs", "wm_noise_sweep_wfe")

ROUTES     = ("full", "wm", "ltm")
BATCH_SIZE = 64

# ──────────────────────────────────────────────────────── group definitions ─

_SEEN_CAT   = "real_word_seen_in_training_lexicon"
_PSEUDO_LEX = {"pseudo", "pseudoword"}
SHORT_LENGTHS = {3, 4, 5}
LONG_LENGTHS  = {7, 8, 9}
MIN_N_SLOPE    = 5
MIN_N_CONTRAST = 3

# Figure display labels — internal snake_case names must not appear on plots.
_GROUP_DISPLAY = {
    "train_seen_real": "Train-seen real words",
    "pseudoword":      "Pseudowords",
    "unseen_forms":    "Unseen forms",
}
_ROUTE_COLORS = {"full": "#2ca02c", "wm": "#1f77b4", "ltm": "#d62728"}
_ROUTE_LABELS = {"full": "Full (gated)", "wm": "WM (dorsal)", "ltm": "LTM (ventral)"}

_GROUP_SPECS = [
    # (group_mode,     group_col,         [(group_key, group_name), ...])
    ("dager_strict",   "group_dager",      [("train_seen_real", "train_seen_real"),
                                            ("pseudoword",      "pseudoword")]),
    ("seen_vs_unseen", "group_seen_unseen", [("train_seen_real", "train_seen_real"),
                                             ("unseen_forms",   "unseen_forms")]),
]

# ────────────────────────────────────────────────────────────── utilities ───

def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Item-level OLS slope y ~ x.  Returns NaN if underdetermined."""
    if len(x) < 2:
        return np.nan
    xm = x.mean()
    var = float(((x - xm) ** 2).mean())
    if var < 1e-12:
        return 0.0
    return float(((x - xm) * (y - y.mean())).mean() / var)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _set_noise(model, noise_level: float) -> float:
    """Override WM interference noise; returns the original value to restore later."""
    original = float(model.wm.cfg.interference_noise)
    model.wm.cfg.interference_noise = float(noise_level)
    return original


def _restore_noise(model, original: float) -> None:
    model.wm.cfg.interference_noise = original


# ──────────────────────────────────────────────────────── data preparation ─

def _annotate_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Add group_dager and group_seen_unseen columns."""
    df = df.copy()
    is_seen   = df["lexicon_category"] == _SEEN_CAT
    is_pseudo = df["lexicality"].str.lower().str.strip().isin(_PSEUDO_LEX)
    df["group_dager"] = np.where(is_seen, "train_seen_real",
                         np.where(is_pseudo, "pseudoword", "excluded"))
    df["group_seen_unseen"] = np.where(is_seen, "train_seen_real", "unseen_forms")
    return df


def _load_wfe(wfe_path: str, vocab, meta: dict,
              ) -> Tuple[pd.DataFrame, List[List[int]]]:
    """Load + filter WFE TSV; add lexicon_category, group columns.
    Returns (df_valid, forms_ids) positionally aligned."""
    df = pd.read_csv(wfe_path, sep="\t")
    df = df[~df["notes"].fillna("").str.contains("EXCLUDED", na=False)].copy()
    df = df.reset_index(drop=True)

    forms_ids: List[List[int]] = []
    valid_idx: List[int] = []
    for i, row in df.iterrows():
        syms = row["target_phonemes"].split()
        ids  = [vocab.stoi[s] for s in syms if s in vocab.stoi]
        if len(ids) == len(syms):
            forms_ids.append(ids)
            valid_idx.append(i)
        else:
            print(f"  WARNING: item {row.get('item_id', i)} has unknown phonemes, skipping")

    df_valid = df.loc[valid_idx].reset_index(drop=True)
    print(f"  WFE: {len(df_valid)} valid items "
          f"({len(df) - len(df_valid)} phoneme-unknown skipped)")

    df_valid["lexicon_category"] = [
        _wfe_lexicon_category(
            row["lexicality"], row["word"],
            row["target_phonemes"], vocab, meta,
        )
        for _, row in df_valid.iterrows()
    ]
    df_valid = _annotate_groups(df_valid)
    return df_valid, forms_ids


# ─────────────────────────────────────────────── noise-aware decode functions

@torch.no_grad()
def _tf_decode(model, vocab, forms: List[List[int]], device: str,
               noise_level: float) -> Dict[str, List[List[int]]]:
    """Teacher-forced batch decode; applies WM noise to wm and full routes."""
    batch = make_batch(forms, vocab, device)
    preds_by_route: Dict[str, List[List[int]]] = {}
    for route in ROUTES:
        collect = (route in ("wm", "full")) and (noise_level > 0.0)
        res = model.route_logits(batch["enc_in"], batch["enc_mask"],
                                 batch["dec_in"], route=route, collect=collect)
        preds = res["logits"].argmax(-1)  # (B, S)
        route_preds = []
        for i in range(len(forms)):
            route_preds.append(preds[i, :len(forms[i])].tolist())
        preds_by_route[route] = route_preds
    return preds_by_route


@torch.no_grad()
def _ar_decode(model, vocab, forms: List[List[int]], device: str,
               noise_level: float) -> Dict[str, List[List[int]]]:
    """Autoregressive batch decode; applies WM noise to wm and full routes.

    The WM encoder runs once per decoder step, adding fresh noise each time.
    This models continuous noisy readout from the phonological buffer.
    """
    batch = make_batch(forms, vocab, device)
    max_steps = max(len(f) for f in forms)
    preds_by_route: Dict[str, List[List[int]]] = {}
    for route in ROUTES:
        collect = (route in ("wm", "full")) and (noise_level > 0.0)
        dec_input = batch["enc_in"].new_full((len(forms), 1), vocab.bos_id)
        for _ in range(max_steps):
            res = model.route_logits(batch["enc_in"], batch["enc_mask"],
                                     dec_input, route=route, collect=collect)
            next_tok = res["logits"][:, -1, :].argmax(-1, keepdim=True)
            dec_input = torch.cat([dec_input, next_tok], dim=1)
        route_preds = []
        for i, form in enumerate(forms):
            pred_ids = dec_input[i, 1:len(form) + 1].tolist()
            seq: List[int] = []
            for idx in pred_ids:
                if idx == vocab.eos_id:
                    break
                seq.append(idx)
            route_preds.append(seq)
        preds_by_route[route] = route_preds
    return preds_by_route


def _run_one_repeat(model, vocab,
                    df_valid: pd.DataFrame, forms_ids: List[List[int]],
                    noise_level: float, decode: str, device: str,
                    ) -> pd.DataFrame:
    """One evaluation pass → DataFrame with route metrics columns appended."""
    decode_fn = _ar_decode if decode == DECODE_AR else _tf_decode
    n = len(forms_ids)
    results: Dict[str, List[dict]] = {r: [] for r in ROUTES}

    for start in range(0, n, BATCH_SIZE):
        batch_forms = forms_ids[start: start + BATCH_SIZE]
        preds = decode_fn(model, vocab, batch_forms, device, noise_level)
        for route in ROUTES:
            for i, form_ids in enumerate(batch_forms):
                pred_ids  = preds[route][i]
                pred_syms = [vocab.itos[idx] for idx in pred_ids]
                tgt_syms  = [vocab.itos[idx] for idx in form_ids]
                results[route].append({
                    "exact_match": int(pred_syms == tgt_syms),
                    "edit_dist":   _edit_distance(tgt_syms, pred_syms),
                    "predicted":   " ".join(pred_syms),
                    "target":      " ".join(tgt_syms),
                })
        if (start // BATCH_SIZE) % 20 == 0:
            print(f"    … {min(start + BATCH_SIZE, n)}/{n}", end="\r")
    print()

    df = df_valid.copy()
    for route in ROUTES:
        df[f"{route}_exact_match"] = [r["exact_match"] for r in results[route]]
        df[f"{route}_edit_dist"]   = [r["edit_dist"]   for r in results[route]]
    return df


# ────────────────────────────────────────────────────────── analysis helpers

def _compute_per_repeat_stats(df_all: pd.DataFrame) -> pd.DataFrame:
    """For each (noise_level, repeat, group_mode, group_name, route, metric),
    compute mean, OLS slope, and long-short contrast across items."""
    rows = []
    for (noise_level, rep_idx), df_r in df_all.groupby(["noise_level", "repeat"]):
        for group_mode, group_col, group_pairs in _GROUP_SPECS:
            for group_key, group_name in group_pairs:
                sub = df_r[df_r[group_col] == group_key]
                if len(sub) == 0:
                    continue
                length = sub["length_phonemes"].values.astype(float)
                for route in ROUTES:
                    error = (1.0 - sub[f"{route}_exact_match"].values).astype(float)
                    edit  = sub[f"{route}_edit_dist"].values.astype(float)
                    n_items = len(sub)

                    for metric_name, vals in [("error_rate", error),
                                              ("edit_dist",  edit)]:
                        mean_val = float(vals.mean())
                        slope    = (_ols_slope(length, vals)
                                    if n_items >= MIN_N_SLOPE else np.nan)
                        ms = np.isin(length, list(SHORT_LENGTHS))
                        ml = np.isin(length, list(LONG_LENGTHS))
                        lsc = (float(vals[ml].mean() - vals[ms].mean())
                               if (ms.sum() >= MIN_N_CONTRAST and
                                   ml.sum() >= MIN_N_CONTRAST)
                               else np.nan)
                        rows.append({
                            "noise_level": float(noise_level),
                            "repeat":      int(rep_idx),
                            "group_mode":  group_mode,
                            "group_name":  group_name,
                            "route":       route,
                            "metric":      metric_name,
                            "mean_value":  round(mean_val, 5),
                            "slope":       (round(slope, 6)
                                            if np.isfinite(slope) else np.nan),
                            "long_short":  (round(lsc, 5)
                                            if np.isfinite(lsc) else np.nan),
                            "n_items":     n_items,
                        })
    return pd.DataFrame(rows)


def _aggregate_summary(df_per_rep: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-repeat stats into one row per
    (noise_level, group_mode, group_name, route, metric)."""
    keys = ["noise_level", "group_mode", "group_name", "route", "metric"]
    rows = []
    for key_vals, grp in df_per_rep.groupby(keys):

        def _agg(col: str) -> Tuple[float, float, float]:
            vals = grp[col].dropna().values
            if len(vals) == 0:
                return np.nan, np.nan, np.nan
            mean = float(np.mean(vals))
            std  = float(np.std(vals, ddof=0)) if len(vals) > 1 else np.nan
            sem  = float(np.std(vals, ddof=0) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan
            return mean, std, sem

        mean_m, std_m, sem_m = _agg("mean_value")
        slope_m, slope_s, _  = _agg("slope")
        lsc_m, lsc_s, _      = _agg("long_short")

        def _r(v: float, ndig: int = 5) -> float:
            return round(v, ndig) if (v is not None and np.isfinite(v)) else np.nan

        row = dict(zip(keys, key_vals))
        row.update({
            "repeat_mean": _r(mean_m),
            "repeat_std":  _r(std_m),
            "repeat_sem":  _r(sem_m),
            "slope_mean":  _r(slope_m, 6),
            "slope_std":   _r(slope_s, 6),
            "lsc_mean":    _r(lsc_m),
            "lsc_std":     _r(lsc_s),
            "n_items":     int(grp["n_items"].iloc[0]),
            "n_repeats":   len(grp),
            "note":        ("positive slope = longer words worse; "
                            "positive lsc = long > short"),
        })
        rows.append(row)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────── figures ───

def _sweep_panel(ax, df_summary: pd.DataFrame,
                 noise_levels: List[float],
                 group_mode: str, group_name: str,
                 routes: Tuple[str, ...],
                 val_col: str, err_col: str,
                 transform_fn=None) -> None:
    """Draw one sweep panel: lines per route across noise levels."""
    sub = df_summary[(df_summary["group_mode"] == group_mode) &
                     (df_summary["group_name"] == group_name)]
    for route in routes:
        sub_r = sub[sub["route"] == route]
        ys, es = [], []
        for nl in noise_levels:
            row = sub_r[sub_r["noise_level"] == nl]
            if len(row) == 0:
                ys.append(np.nan)
                es.append(0.0)
            else:
                v = float(row[val_col].iloc[0])
                s = row[err_col].iloc[0]
                s = float(s) if (s is not None and np.isfinite(float(s))) else 0.0
                if transform_fn is not None:
                    v = transform_fn(v)
                ys.append(v)
                es.append(s)
        ys = np.array(ys, dtype=float)
        es = np.array(es, dtype=float)
        ax.plot(noise_levels, ys, "-o", lw=1.8, ms=5,
                color=_ROUTE_COLORS[route], label=_ROUTE_LABELS[route])
        ax.fill_between(noise_levels, ys - es, ys + es,
                        color=_ROUTE_COLORS[route], alpha=0.18)


def _make_slope_figure(df_summary: pd.DataFrame, metric: str, ylabel: str,
                       out_path: str, decode: str) -> None:
    """Figures A/B: length-effect slope vs noise level, for both non-seen groups."""
    sub_m = df_summary[df_summary["metric"] == metric]
    noise_levels = sorted(df_summary["noise_level"].unique())

    panels = [
        ("dager_strict",   "pseudoword",    "Pseudowords\n(Dager-comparable analysis)"),
        ("seen_vs_unseen", "unseen_forms",  "Unseen forms\n(Generalization analysis)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, (gm, gn, display) in zip(axes, panels):
        _sweep_panel(ax, sub_m, noise_levels, gm, gn, ROUTES,
                     "slope_mean", "slope_std")
        ax.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.5)
        ax.set_xlabel("WM/dorsal noise level", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(display, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.text(0.02, 0.97, "Positive = longer words worse",
                transform=ax.transAxes, fontsize=7, va="top", color="#555")

    decode_label = "Autoregressive decoding" if decode == DECODE_AR else "Teacher-forced decoding"
    fig.suptitle(
        f"WFE length-effect slope — {metric.replace('_', ' ')}  "
        f"({decode_label})  ·  shaded = ±1 SD across repeats",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [fig] {out_path}")


def _make_lsc_figure(df_summary: pd.DataFrame, out_path: str,
                     decode: str) -> None:
    """Figure C: long-short contrast (edit_dist) vs noise level."""
    sub_m = df_summary[df_summary["metric"] == "edit_dist"]
    noise_levels = sorted(df_summary["noise_level"].unique())

    panels = [
        ("dager_strict",   "pseudoword",    "Pseudowords\n(Dager-comparable analysis)"),
        ("seen_vs_unseen", "unseen_forms",  "Unseen forms\n(Generalization analysis)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, (gm, gn, display) in zip(axes, panels):
        _sweep_panel(ax, sub_m, noise_levels, gm, gn, ROUTES,
                     "lsc_mean", "lsc_std")
        ax.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.5)
        ax.set_xlabel("WM/dorsal noise level", fontsize=10)
        ax.set_ylabel("Long − short contrast (edit distance)", fontsize=10)
        ax.set_title(display, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.text(0.02, 0.97, "Positive = long words worse than short",
                transform=ax.transAxes, fontsize=7, va="top", color="#555")

    decode_label = "Autoregressive decoding" if decode == DECODE_AR else "Teacher-forced decoding"
    fig.suptitle(
        f"WFE long − short contrast — edit distance  ({decode_label})  "
        f"·  short = {{{','.join(str(x) for x in sorted(SHORT_LENGTHS))}}} phones  "
        f"long = {{{','.join(str(x) for x in sorted(LONG_LENGTHS))}}} phones  "
        f"·  shaded = ±1 SD across repeats",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [fig] {out_path}")


def _make_accuracy_figure(df_summary: pd.DataFrame, out_path: str,
                           decode: str) -> None:
    """Figure D: mean exact-match accuracy vs noise level."""
    sub_m = df_summary[df_summary["metric"] == "error_rate"]
    noise_levels = sorted(df_summary["noise_level"].unique())

    # dager_strict groups only; routes = full and wm
    panels = [
        ("dager_strict", "train_seen_real", "Train-seen real words"),
        ("dager_strict", "pseudoword",      "Pseudowords"),
    ]
    acc_routes = ("full", "wm")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, (gm, gn, display) in zip(axes, panels):
        _sweep_panel(ax, sub_m, noise_levels, gm, gn, acc_routes,
                     "repeat_mean", "repeat_std",
                     transform_fn=lambda v: 1.0 - v)   # error → accuracy
        ax.set_xlabel("WM/dorsal noise level", fontsize=10)
        ax.set_ylabel("Exact-match accuracy", fontsize=10)
        ax.set_title(display, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    decode_label = "Autoregressive decoding" if decode == DECODE_AR else "Teacher-forced decoding"
    fig.suptitle(
        f"WFE global accuracy vs WM noise  ({decode_label})  "
        f"·  shaded = ±1 SD across repeats",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [fig] {out_path}")


# ──────────────────────────────────────────────────────────────── README ───

def _write_readme(out_dir: str, decode: str, noise_levels: List[float],
                  n_repeats: int) -> None:
    noise_str = "  ".join(f"{nl:.3f}" for nl in sorted(noise_levels))
    decode_label = ("Autoregressive (AR) — each decoder step receives the model's "
                    "own previous prediction; errors propagate."
                    if decode == DECODE_AR
                    else "Teacher-forced (TF) — each decoder step receives the gold "
                         "previous phoneme; errors do not propagate.")
    readme = f"""\
# WM/dorsal noise sweep — WFE evaluation

## Why this experiment exists

Baseline analysis (no noise) showed a modest WM length effect for pseudowords.
We want to know whether explicitly increasing WM/dorsal interference noise
amplifies the pseudoword length effect, and whether autoregressive decoding
amplifies this further.

Baseline finding (no noise):
- Train-seen real words are near ceiling (both TF and AR decoding).
- Pseudowords show a modest full/WM length effect.
- LTM shows a stronger pseudoword length effect than WM.
- Autoregressive decoding amplifies edit-distance length effects.

## Experimental design

### Decoding regime
{decode_label}

### Noise levels
{noise_str}

For noise_level = 0.000: deterministic baseline (1 repeat, no noise).
For noise_level > 0.000: {n_repeats} stochastic repeats, seeded reproducibly.

### Repeat count
{n_repeats} repeats per noise level > 0.  Each repeat uses a different seed
(seed + repeat_index) so the noise samples are independent.

### Noise mechanism
WM interference noise is Gaussian noise N(0, noise_level) added to the GRU
encoder hidden state h after encoding and before decoding (WMRecurrent.forward).
This is an evaluation-time perturbation — model weights are never changed.
The noise applies to the "wm" route and to the WM component inside the "full"
gated route.  The "ltm" route is unaffected (no WM component).

## Group definitions

### Dager-comparable analysis
- Train-seen real words: lexicon_category == real_word_seen_in_training_lexicon
- Pseudowords: lexicality in {{pseudo, pseudoword}}
- Held-out and novel real words are excluded from this analysis.

### Generalization analysis
- Train-seen real words: as above.
- Unseen forms: all other items (held-out real + novel real + pseudowords).
  IMPORTANT: "Unseen forms" is NOT a lexicality group.  It is a familiarity
  group mixing different item types.  Do not interpret it as "pseudowords."

## How to interpret results

### Length-effect slope (positive = longer words harder)
A positive slope means error rate or edit distance increases with word length.
If this slope increases with noise level, WM noise amplifies the length effect.

### Long − short contrast (positive = long words harder than short)
A positive value means words in {{7,8,9}} phones are harder than words in
{{3,4,5}} phones.

### AR − TF comparison
Under autoregressive decoding, errors propagate.  If noise-amplified length
effects are larger under AR than TF, the combination of noise + error propagation
creates a super-additive length penalty.

## Caveat

This is EVALUATION-TIME noise, not a retrained model.  The model was trained
with interference_noise applied only during training.  Applying larger noise at
evaluation time is an out-of-distribution perturbation.  Results describe the
model's behavior under extrapolated noise, not its trained working regime.

## Files

| File | Contents |
|---|---|
| noise_{{level}}/repeat_{{i}}/item_level_predictions.tsv | Per-item predictions |
| noise_sweep_item_metrics.tsv | All repeats stacked (noise_level, repeat, item) |
| noise_sweep_summary.tsv | Aggregated per-(noise, group, route, metric) statistics |
| wm_noise_length_slope_edit_dist.png | Edit-dist slope vs noise (pseudowords, unseen forms) |
| wm_noise_length_slope_error_rate.png | Error-rate slope vs noise |
| wm_noise_long_short_edit_dist.png | Long-short contrast (edit dist) vs noise |
| wm_noise_global_accuracy.png | Mean accuracy vs noise (seen real, pseudowords) |
"""
    readme_path = os.path.join(out_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme)
    print(f"  [readme] {readme_path}")


# ──────────────────────────────────────────────────────────────────── CLI ───

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="WM/dorsal noise sweep for WFE evaluation.")
    p.add_argument("--ckpt",         default=CKPT_DEFAULT)
    p.add_argument("--wfe_path",     default=WFE_DEFAULT)
    p.add_argument("--out_dir",      default=OUT_DEFAULT)
    p.add_argument("--decode",       default=DECODE_AR,
                   choices=[DECODE_TF, DECODE_AR],
                   help="Decoding regime. Default: autoregressive.")
    p.add_argument("--noise_levels", nargs="+", type=float,
                   default=[0.0, 0.01, 0.03, 0.05, 0.10],
                   help="WM interference noise levels to sweep.")
    p.add_argument("--n_repeats",    type=int, default=20,
                   help="Stochastic repeats per noise_level > 0.")
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--device",       default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    noise_levels = sorted(set(args.noise_levels))
    decode       = args.decode

    print("=" * 60)
    print("  WM/dorsal noise sweep — WFE")
    print("=" * 60)
    print(f"  checkpoint   : {args.ckpt}")
    print(f"  wfe_path     : {args.wfe_path}")
    print(f"  out_dir      : {args.out_dir}")
    print(f"  decode       : {decode}")
    print(f"  noise_levels : {noise_levels}")
    print(f"  n_repeats    : {args.n_repeats}  (for noise > 0)")
    print(f"  seed         : {args.seed}")
    print(f"  device       : {device}")
    print()

    # ── load model ──────────────────────────────────────────────────────────
    model, vocab, meta = load_model_and_vocab(args.ckpt, device)
    original_noise = float(model.wm.cfg.interference_noise)
    print(f"  model loaded  |  original WM interference_noise = {original_noise}")

    # ── load + annotate WFE items ────────────────────────────────────────────
    if not os.path.exists(args.wfe_path):
        print(f"\nERROR: WFE TSV not found: {args.wfe_path}")
        sys.exit(1)
    df_valid, forms_ids = _load_wfe(args.wfe_path, vocab, meta)
    n_items = len(forms_ids)

    # ── sweep ────────────────────────────────────────────────────────────────
    all_item_records: List[pd.DataFrame] = []

    for noise_level in noise_levels:
        n_reps = 1 if noise_level == 0.0 else args.n_repeats
        noise_str = f"{noise_level:.3f}"
        print(f"\n── noise_level = {noise_str}  ({n_reps} repeat(s)) ──")

        saved_noise = _set_noise(model, noise_level)

        for rep_idx in range(n_reps):
            rep_seed = args.seed + rep_idx
            _set_seed(rep_seed)

            rep_dir = os.path.join(
                args.out_dir, f"noise_{noise_str}", f"repeat_{rep_idx:03d}")
            os.makedirs(rep_dir, exist_ok=True)

            print(f"  repeat {rep_idx:03d}  seed={rep_seed}  "
                  f"noise={'ON' if noise_level > 0 else 'OFF (deterministic)'}")
            df_rep = _run_one_repeat(
                model, vocab, df_valid, forms_ids,
                noise_level=noise_level, decode=decode, device=device,
            )

            # per-repeat TSV (required output)
            pred_path = os.path.join(rep_dir, "item_level_predictions.tsv")
            df_rep.to_csv(pred_path, sep="\t", index=False)
            print(f"  -> {pred_path}")

            # tag for aggregation
            df_tag = df_rep.copy()
            df_tag.insert(0, "noise_level", noise_level)
            df_tag.insert(1, "repeat", rep_idx)
            all_item_records.append(df_tag)

        _restore_noise(model, saved_noise)

    # ── aggregated item TSV ──────────────────────────────────────────────────
    agg_cols = [
        "noise_level", "repeat",
        "item_id", "word", "lexicality", "lexicon_category",
        "length_phonemes", "condition",
        "group_dager", "group_seen_unseen",
        "full_exact_match", "full_edit_dist",
        "wm_exact_match",   "wm_edit_dist",
        "ltm_exact_match",  "ltm_edit_dist",
    ]
    df_all = pd.concat(all_item_records, ignore_index=True)
    # keep only columns that exist (graceful if some metadata cols are absent)
    agg_cols_present = [c for c in agg_cols if c in df_all.columns]
    df_all_out = df_all[agg_cols_present]
    item_tsv_path = os.path.join(args.out_dir, "noise_sweep_item_metrics.tsv")
    df_all_out.to_csv(item_tsv_path, sep="\t", index=False)
    print(f"\n  -> {item_tsv_path}  ({len(df_all_out)} rows)")

    # ── per-repeat statistics ────────────────────────────────────────────────
    print("\n  Computing per-repeat statistics …")
    df_per_rep = _compute_per_repeat_stats(df_all)
    df_summary = _aggregate_summary(df_per_rep)

    summary_path = os.path.join(args.out_dir, "noise_sweep_summary.tsv")
    df_summary.to_csv(summary_path, sep="\t", index=False)
    print(f"  -> {summary_path}  ({len(df_summary)} rows)")

    # ── figures ──────────────────────────────────────────────────────────────
    print("\n  Generating figures …")

    _make_slope_figure(
        df_summary, "edit_dist", "Length-effect slope (edit distance)",
        os.path.join(args.out_dir, "wm_noise_length_slope_edit_dist.png"),
        decode,
    )
    _make_slope_figure(
        df_summary, "error_rate", "Length-effect slope (error rate)",
        os.path.join(args.out_dir, "wm_noise_length_slope_error_rate.png"),
        decode,
    )
    _make_lsc_figure(
        df_summary,
        os.path.join(args.out_dir, "wm_noise_long_short_edit_dist.png"),
        decode,
    )
    _make_accuracy_figure(
        df_summary,
        os.path.join(args.out_dir, "wm_noise_global_accuracy.png"),
        decode,
    )

    # ── README ───────────────────────────────────────────────────────────────
    _write_readme(args.out_dir, decode, noise_levels, args.n_repeats)

    # ── summary ──────────────────────────────────────────────────────────────
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
