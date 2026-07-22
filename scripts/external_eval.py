"""Steps 4–6: Run external diagnostics of the Yair-L3 model on WFE and SSP.

IMPORTANT — EVALUATION REGIME
------------------------------
All inference here is TEACHER-FORCED (identical to evaluate/*.py in this repo):
the decoder receives the gold previous phoneme at each step.  This means
predicted[t] is conditioned on correct[0..t-1], not on predicted[0..t-1].
Per-position correctness is therefore not contaminated by error propagation,
but it also does NOT reflect true free-recall performance.  All output files
state this explicitly.

PREREQUISITE
------------
A checkpoint must exist (created by scripts/train_checkpoint.py):
    checkpoints/lichtheim3.pt

If it does not exist, run:
    python scripts/train_checkpoint.py --max_words 4000 --epochs 30 --seed 0

Usage:
    # Dry run (10 WFE + 20 SSP items)
    python scripts/external_eval.py --dry_run

    # Full run
    python scripts/external_eval.py

    # Explicit paths
    python scripts/external_eval.py \\
        --ckpt     checkpoints/lichtheim3.pt \\
        --wfe_tsv  data/eval_external/wfe_yair_l3_format.tsv \\
        --ssp_tsv  data/eval_external/ssp_yair_l3_format.tsv \\
        --out_dir  outputs/external_eval
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import dataclasses
from itertools import zip_longest
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import Config, DataConfig, WMConfig, LTMConfig, GatingConfig, LossConfig, TrainConfig
from config import get_effective_split_seed
from data.phonemes import build_vocab, Vocab
from data.lexicon import build_lexicon
from models.dual_route import DualRouteModel
from evaluate.hooks import make_batch, route_predictions, per_position_correct

# ---------------------------------------------------------------------------
CKPT_PATH    = os.path.join(ROOT, "checkpoints", "lichtheim3.pt")
WFE_TSV      = os.path.join(ROOT, "data", "eval_external", "wfe_eval.tsv")
SSP_TSV      = os.path.join(ROOT, "data", "eval_external", "ssp_eval.tsv")
OUT_DIR      = os.path.join(ROOT, "outputs", "external_eval")
DRY_WFE_N   = 10
DRY_SSP_N   = 20
BATCH_SIZE   = 64
# ---------------------------------------------------------------------------

DECODE_TF = "teacher_forced"
DECODE_AR = "autoregressive"

_REGIME_NOTES = {
    DECODE_TF: (
        "Teacher-forced decoding: the decoder receives the gold previous phoneme "
        "at each step (dec_in = [BOS, p1, ..., pT-1]).  "
        "This does NOT simulate true free-recall error propagation."
    ),
    DECODE_AR: (
        "Autoregressive decoding: the decoder receives its own previous "
        "prediction at each step.  Errors propagate: a wrong phoneme at "
        "position t corrupts all subsequent positions.  This is the "
        "behaviorally plausible regime for comparison with Dager/SWP."
    ),
}

# Keep for backward compatibility
EVAL_REGIME_NOTE = _REGIME_NOTES[DECODE_TF]


# ============================================================  helpers  ====

def _edit_distance(a: list, b: list) -> int:
    """Levenshtein distance between two sequences (no external library needed)."""
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j - 1], prev[j], dp[j - 1])
    return dp[m]


def _norm_edit_distance(a: list, b: list) -> float:
    ed = _edit_distance(a, b)
    denom = max(len(a), len(b), 1)
    return ed / denom


def _argmax_to_phonemes(preds: torch.Tensor, target: torch.Tensor,
                         vocab: Vocab) -> Tuple[List[str], List[str]]:
    """Convert integer tensors (1-D, padded) to phoneme symbol lists (no PAD/BOS/EOS)."""
    def to_syms(t: torch.Tensor) -> List[str]:
        out = []
        for idx in t.tolist():
            if idx == vocab.eos_id or idx == vocab.pad_id:
                break
            if idx == vocab.bos_id:
                continue
            out.append(vocab.itos[idx])
        return out
    return to_syms(preds), to_syms(target)


# ============================================================  loading  ====

def load_model_and_vocab(ckpt_path: str, device: str) -> Tuple[DualRouteModel, Vocab, dict]:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"\n\nCheckpoint not found: {ckpt_path}\n"
            "Run first:  python scripts/train_checkpoint.py\n"
        )
    print(f"[external_eval] Loading checkpoint: {ckpt_path}")
    # weights_only=False required: checkpoint contains Python dicts/lists (config, history)
    # in addition to tensors.  Safe here because we control the checkpoint source.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Rebuild config from saved dicts
    cfg = Config(
        data   = DataConfig(**ckpt["cfg_data"]),
        wm     = WMConfig(**ckpt["cfg_wm"]),
        ltm    = LTMConfig(**ckpt["cfg_ltm"]),
        gating = GatingConfig(**ckpt["cfg_gating"]),
        loss   = LossConfig(**ckpt["cfg_loss"]),
        train  = TrainConfig(**ckpt["cfg_train"]),
    )
    cfg.train.device = device

    vocab = build_vocab()

    # Rebuild lexicon to reconstruct the semantic bank and training word sets
    lexicon = build_lexicon(cfg.data, vocab)
    train_entries, val_entries = lexicon.split(cfg.data.val_fraction, get_effective_split_seed(cfg.data))
    bank = torch.stack(
        [torch.tensor(e.semantic) for e in train_entries]
    ).float().to(device)

    model = DualRouteModel(cfg, vocab).to(device)
    model.set_semantic_bank(bank)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    meta = {
        "checkpoint":    ckpt_path,
        "lexicon_source": ckpt.get("lexicon_source", "unknown"),
        "n_train":        ckpt.get("n_train"),
        "n_val":          ckpt.get("n_val"),
        "glove_present":  ckpt.get("glove_present", False),
        "git_commit":     ckpt.get("git_commit", ""),
        "history":        ckpt.get("history", []),
        "cfg_summary": {
            "epochs":    cfg.train.epochs,
            "max_words": cfg.data.max_words,
            "seed":      cfg.train.seed,
            "device":    cfg.train.device,
        },
        # Word/phoneme sets for WFE lexicon-overlap analysis.
        # Keys are lowercase orthographic strings; phoneme sets use tuples of int IDs.
        "_train_word_set":    {e.word.lower() for e in train_entries},
        "_train_phoneme_set": {tuple(e.phonemes) for e in train_entries},
        "_val_word_set":      {e.word.lower() for e in val_entries},
        "_val_phoneme_set":   {tuple(e.phonemes) for e in val_entries},
    }
    print(f"  lexicon: {meta['lexicon_source']}  "
          f"n_train={meta['n_train']}  n_val={meta['n_val']}")
    print(f"  glove_present: {meta['glove_present']}")
    return model, vocab, meta


# ============================================================  inference ====

@torch.no_grad()
def eval_batch(model: DualRouteModel, vocab: Vocab,
               forms: List[List[int]], device: str,
               routes: Tuple[str, ...] = ("full", "wm", "ltm"),
               wm_noise: bool = False,
               ) -> Dict[str, List[List[int]]]:
    """Return per-route predicted phoneme id sequences for a list of forms.

    wm_noise=False (default): collect=False for WM route → WM interference noise
    disabled → deterministic evaluation.  Pass wm_noise=True only for explicit
    diagnostic runs where noise-in-eval is desired.
    """
    batch = make_batch(forms, vocab, device)
    preds_by_route: Dict[str, List[List[int]]] = {}

    for route in routes:
        collect = (route == "wm") and wm_noise
        preds, _ = route_predictions(model, batch, route=route, collect=collect)
        route_preds = []
        for i in range(len(forms)):
            # Teacher-forced decoding: we ran exactly len(form) meaningful steps.
            # Take only those steps — do NOT rely on the model emitting EOS at the
            # right position, because in a padded batch a short item's predictions
            # beyond its natural length are garbage conditioned on PAD inputs.
            n_steps = len(forms[i])
            seq = preds[i, :n_steps].tolist()
            route_preds.append(seq)
        preds_by_route[route] = route_preds

    return preds_by_route


@torch.no_grad()
def autoregressive_decode_batch(model: DualRouteModel, vocab: Vocab,
                                 forms: List[List[int]], device: str,
                                 routes: Tuple[str, ...] = ("full", "wm", "ltm"),
                                 wm_noise: bool = False,
                                 ) -> Dict[str, List[List[int]]]:
    """Autoregressive decoding: each step feeds the model's own previous prediction.

    Decodes for exactly len(form) steps per item.  If the model predicts EOS
    before the target length is reached, the output is truncated there (the
    remaining positions are treated as missing, contributing to edit distance).
    """
    batch = make_batch(forms, vocab, device)
    max_steps = max(len(f) for f in forms)
    preds_by_route: Dict[str, List[List[int]]] = {}

    for route in routes:
        collect = (route == "wm") and wm_noise
        # Start with BOS only; grow one token per step
        dec_input = batch["enc_in"].new_full((len(forms), 1), vocab.bos_id)

        for _ in range(max_steps):
            res = model.route_logits(batch["enc_in"], batch["enc_mask"],
                                     dec_input, route=route, collect=collect)
            next_tok = res["logits"][:, -1, :].argmax(-1, keepdim=True)  # (B, 1)
            dec_input = torch.cat([dec_input, next_tok], dim=1)

        # dec_input: (B, max_steps+1) = [BOS, pred_0, ..., pred_{max_steps-1}]
        route_preds = []
        for i, form in enumerate(forms):
            n_steps = len(form)
            pred_ids = dec_input[i, 1:n_steps + 1].tolist()
            # Trim at first EOS so output length matches what the model intended
            seq: List[int] = []
            for idx in pred_ids:
                if idx == vocab.eos_id:
                    break
                seq.append(idx)
            route_preds.append(seq)

        preds_by_route[route] = route_preds

    return preds_by_route


def evaluate_items(model: DualRouteModel, vocab: Vocab,
                   forms_ids: List[List[int]], device: str,
                   batch_size: int = BATCH_SIZE,
                   routes: Tuple[str, ...] = ("full", "wm", "ltm"),
                   wm_noise: bool = False,
                   decode: str = DECODE_TF,
                   ) -> Dict[str, List[dict]]:
    """
    Run inference and compute per-item metrics for each route.

    decode="teacher_forced"  — gold prefix at every decoder step (upper bound).
    decode="autoregressive"  — model's own previous output at every step.

    Returns:  {route: [{exact_match, phoneme_acc, edit_dist, norm_edit_dist,
                        predicted_symbols, target_symbols}, ...]}
    """
    results: Dict[str, List[dict]] = {r: [] for r in routes}

    n = len(forms_ids)
    for start in range(0, n, batch_size):
        batch_forms = forms_ids[start: start + batch_size]
        if decode == DECODE_AR:
            preds_by_route = autoregressive_decode_batch(
                model, vocab, batch_forms, device, routes, wm_noise=wm_noise)
        else:
            preds_by_route = eval_batch(
                model, vocab, batch_forms, device, routes, wm_noise=wm_noise)

        for r in routes:
            for i, form_ids in enumerate(batch_forms):
                pred_ids = preds_by_route[r][i]
                tgt_ids  = form_ids   # No specials in form_ids

                pred_syms = [vocab.itos[idx] for idx in pred_ids]
                tgt_syms  = [vocab.itos[idx] for idx in tgt_ids]

                exact_match  = int(pred_syms == tgt_syms)
                # zip_longest pads the shorter sequence with None, so positions
                # where the prediction ran short are counted as wrong — not silently
                # dropped as they would be with plain zip().
                _SENTINEL = object()
                phoneme_acc  = (
                    sum(a == b for a, b in
                        zip_longest(pred_syms, tgt_syms, fillvalue=_SENTINEL))
                    / max(len(tgt_syms), 1)
                )
                ed  = _edit_distance(tgt_syms, pred_syms)
                ned = _norm_edit_distance(tgt_syms, pred_syms)

                results[r].append({
                    "exact_match":  exact_match,
                    "phoneme_acc":  round(phoneme_acc, 4),
                    "edit_dist":    ed,
                    "norm_edit":    round(ned, 4),
                    "predicted":    " ".join(pred_syms),
                    "target":       " ".join(tgt_syms),
                })

        if (start // batch_size) % 10 == 0:
            print(f"  … {min(start+batch_size, n)}/{n}", end="\r")

    print()
    return results


# ===========================================================  WFE eval  ====

# Four-way WFE lexicon overlap categories (mandatory per eval plan)
_OVERLAP_SEEN      = "real_word_seen_in_training_lexicon"
_OVERLAP_VAL       = "real_word_in_validation_split"
_OVERLAP_OUTSIDE   = "real_word_outside_4000_lexicon"
_OVERLAP_PSEUDO    = "pseudoword"


def _wfe_lexicon_category(lexicality: str, word: str,
                           target_phonemes_str: str,
                           vocab: Vocab, meta: dict) -> str:
    """Classify one WFE item by its relationship to the Yair-L3 training lexicon.

    Categories:
      pseudoword                         — lexicality != "real"
      real_word_seen_in_training_lexicon — word or phoneme seq in train split
      real_word_in_validation_split      — word or phoneme seq in val split
      real_word_outside_4000_lexicon     — real word not found in either split
                                           (either beyond max_words cutoff or
                                            absent from lexicon_en.tsv entirely)
    """
    if lexicality != "real":
        return _OVERLAP_PSEUDO

    word_lower = word.lower().strip()
    phon_ids   = tuple(
        vocab.stoi[s] for s in target_phonemes_str.split() if s in vocab.stoi
    )

    if (word_lower in meta["_train_word_set"] or
            phon_ids in meta["_train_phoneme_set"]):
        return _OVERLAP_SEEN

    if (word_lower in meta["_val_word_set"] or
            phon_ids in meta["_val_phoneme_set"]):
        return _OVERLAP_VAL

    return _OVERLAP_OUTSIDE


def figure_bar(groups: list, values: list, title: str, ylabel: str, path: str,
               err: list = None) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(groups)*0.9), 4))
    x = np.arange(len(groups))
    if err:
        ax.bar(x, values, color="steelblue", yerr=err, capsize=4)
    else:
        ax.bar(x, values, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in groups], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    finite_vals = [v for v in values if v == v]  # filter NaN (NaN != NaN)
    ylim_top = max(max(finite_vals) * 1.15, 0.01) if finite_vals else 1.0
    ax.set_ylim(0, ylim_top)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  [fig] {path}")


def figure_scatter(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str,
                   title: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, s=18, alpha=0.4, color="steelblue")
    if len(x) > 1 and x.std() > 0:
        m, c = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, m*xs + c, color="crimson", lw=1.5)
        r = float(np.corrcoef(x, y)[0, 1])
        ax.set_title(f"{title}  (r={r:+.3f})")
    else:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  [fig] {path}")


def run_wfe_eval(model: DualRouteModel, vocab: Vocab, meta: dict,
                 tsv_path: str, out_dir: str, dry_run: bool,
                 device: str, wm_noise: bool = False,
                 decode: str = DECODE_TF) -> dict:
    df = pd.read_csv(tsv_path, sep="\t")
    # Exclude items flagged EXCLUDED (unknown phonemes)
    df = df[~df["notes"].fillna("").str.contains("EXCLUDED", na=False)].copy()
    df = df.reset_index(drop=True)

    subdir = ("dry_run_wfe" if dry_run
              else ("wfe_ar" if decode == DECODE_AR else "wfe"))
    out_dir = os.path.join(out_dir, subdir)

    if dry_run:
        # Balanced sample: 5 real + 5 pseudo, mix of short/long
        real_df  = df[df["lexicality"] == "real"].sample(
            n=min(5, (df["lexicality"]=="real").sum()), random_state=0)
        pseudo_df = df[df["lexicality"] != "real"].sample(
            n=min(5, (df["lexicality"]!="real").sum()), random_state=0)
        df = pd.concat([real_df, pseudo_df]).reset_index(drop=True)
        print(f"\n[WFE DRY RUN] {len(df)} items  -> {out_dir}")
    else:
        print(f"\n[WFE FULL EVAL ({decode})] {len(df)} items  -> {out_dir}")

    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    routes: Tuple[str, ...] = ("full", "wm", "ltm")

    # --- Parse phoneme sequences -> integer id lists ---
    forms_ids: List[List[int]] = []
    valid_idx: List[int] = []
    for i, row in df.iterrows():
        syms = row["target_phonemes"].split()
        ids  = [vocab.stoi[s] for s in syms if s in vocab.stoi]
        if len(ids) == len(syms):
            forms_ids.append(ids)
            valid_idx.append(i)
        else:
            print(f"  WARNING: item {row['item_id']} has unknown phonemes, skipping")

    df_valid = df.loc[valid_idx].reset_index(drop=True)
    n_valid  = len(forms_ids)
    print(f"  {n_valid} items with fully valid phoneme sequences")

    # --- WFE lexicon overlap (mandatory) ---
    df_valid["lexicon_category"] = [
        _wfe_lexicon_category(
            row["lexicality"], row["word"],
            row["target_phonemes"], vocab, meta
        )
        for _, row in df_valid.iterrows()
    ]
    overlap_counts = df_valid["lexicon_category"].value_counts().to_dict()
    print(f"  lexicon overlap: {overlap_counts}")

    # --- Run inference ---
    print(f"  Running {decode} inference …  "
          f"(WM noise: {'ON' if wm_noise else 'OFF — deterministic'})")
    results = evaluate_items(model, vocab, forms_ids, device, routes=routes,
                             wm_noise=wm_noise, decode=decode)

    # --- Attach results to dataframe ---
    for route in routes:
        for metric in ("exact_match", "phoneme_acc", "edit_dist", "norm_edit",
                       "predicted", "target"):
            df_valid[f"{route}_{metric}"] = [r[metric] for r in results[route]]

    # --- Item-level predictions TSV ---
    pred_path = os.path.join(out_dir, "item_level_predictions.tsv")
    df_valid.to_csv(pred_path, sep="\t", index=False)
    print(f"  -> {pred_path}")

    # --- Aggregate metrics ---
    def agg_by(col: str, df_: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for g in sorted(df_[col].dropna().unique()):
            sub = df_[df_[col] == g]
            row = {"group_by": col, "value": g, "n": len(sub)}
            for r in routes:
                for m in ("exact_match", "phoneme_acc", "edit_dist", "norm_edit"):
                    row[f"{r}_{m}"] = round(float(sub[f"{r}_{m}"].mean()), 4)
            rows.append(row)
        return pd.DataFrame(rows)

    group_cols = ["lexicality", "lexicon_category", "size", "morphology",
                  "condition", "length_phonemes", "part_of_speech"]
    summary_frames = []
    for col in group_cols:
        if col in df_valid.columns:
            summary_frames.append(agg_by(col, df_valid))

    # Frequency-bin breakdown (real words only, 5 equal-width bins by Zipf score)
    if "zipf_frequency" in df_valid.columns:
        real_sub = df_valid[df_valid["lexicality"] == "real"].copy()
        real_sub["zipf_frequency"] = pd.to_numeric(
            real_sub["zipf_frequency"], errors="coerce")
        real_sub = real_sub.dropna(subset=["zipf_frequency"])
        if len(real_sub) >= 5:
            real_sub["zipf_bin"] = pd.qcut(
                real_sub["zipf_frequency"], q=5,
                labels=["Q1_lowest", "Q2", "Q3", "Q4", "Q5_highest"],
                duplicates="drop")
            summary_frames.append(agg_by("zipf_bin", real_sub))

    summary_df = pd.concat(summary_frames, ignore_index=True)
    summary_path = os.path.join(out_dir, "summary_table.tsv")
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print(f"  -> {summary_path}")

    # --- Overall metrics dict ---
    overall: dict = {}
    for route in routes:
        overall[route] = {
            m: round(float(df_valid[f"{route}_{m}"].mean()), 4)
            for m in ("exact_match", "phoneme_acc", "edit_dist", "norm_edit")
        }

    # Per-category accuracy breakdown (mandatory WFE overlap analysis)
    overlap_accuracy: dict = {}
    for cat in (_OVERLAP_SEEN, _OVERLAP_VAL, _OVERLAP_OUTSIDE, _OVERLAP_PSEUDO):
        sub = df_valid[df_valid["lexicon_category"] == cat]
        if len(sub) == 0:
            overlap_accuracy[cat] = {"n": 0}
            continue
        overlap_accuracy[cat] = {"n": len(sub)}
        for route in routes:
            overlap_accuracy[cat][f"{route}_exact_match"] = round(
                float(sub[f"{route}_exact_match"].mean()), 4)
            overlap_accuracy[cat][f"{route}_phoneme_acc"] = round(
                float(sub[f"{route}_phoneme_acc"].mean()), 4)
            overlap_accuracy[cat][f"{route}_edit_dist"] = round(
                float(sub[f"{route}_edit_dist"].mean()), 4)

    metrics = {
        "evaluation_regime":    _REGIME_NOTES[decode],
        "decode_mode":          decode,
        "checkpoint":           meta["checkpoint"],
        "git_commit":           meta.get("git_commit", ""),
        "glove_present":        meta["glove_present"],
        "lexicon_source":       meta["lexicon_source"],
        "wm_noise_enabled":     wm_noise,
        "deterministic":        not wm_noise,
        "n_items_evaluated":    n_valid,
        "n_items_excluded":     len(df) - n_valid,
        "dry_run":              dry_run,
        "overall":              overall,
        "wfe_lexicon_overlap":  {
            "counts":   overlap_counts,
            "accuracy": overlap_accuracy,
            "note": (
                "real_word_seen_in_training_lexicon: word/phoneme-seq matched "
                "train split of the 4k-word lexicon.  "
                "real_word_in_validation_split: matched val split.  "
                "real_word_outside_4000_lexicon: real word absent from both "
                "train and val (beyond max_words cutoff or not in CMU dict).  "
                "pseudoword: Lexicality != 'real'."
            ),
        },
    }
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  -> {metrics_path}")

    if not dry_run:
        _make_wfe_figures(df_valid, fig_dir, routes)

    # Console summary
    print(f"\n  === WFE RESULTS ({decode}) ===")
    for route in routes:
        o = overall[route]
        print(f"  [{route:5s}] exact={o['exact_match']:.3f}  "
              f"phon_acc={o['phoneme_acc']:.3f}  "
              f"edit={o['edit_dist']:.2f}  ned={o['norm_edit']:.3f}")

    return metrics


def _make_wfe_figures(df: pd.DataFrame, fig_dir: str,
                      routes: Tuple[str, ...]) -> None:
    def _bar(col: str, metric: str, title: str, fname: str) -> None:
        if col not in df.columns:
            return
        groups = sorted(df[col].dropna().unique())
        for route in routes:
            vals = [df[df[col] == g][f"{route}_{metric}"].mean() for g in groups]
            figure_bar(groups, vals, f"{title} [{route}]", metric,
                       os.path.join(fig_dir, f"{fname}_{route}.png"))

    _bar("lexicality",     "exact_match",  "Accuracy by Lexicality",  "accuracy_by_lexicality")
    _bar("length_phonemes","exact_match",  "Accuracy by Length",       "accuracy_by_length")
    _bar("size",           "exact_match",  "Accuracy by Size",         "accuracy_by_size")
    _bar("morphology",     "exact_match",  "Accuracy by Morphology",   "accuracy_by_morphology")
    _bar("condition",      "exact_match",  "Accuracy by Condition",    "accuracy_by_condition")
    _bar("lexicality",     "edit_dist",    "Edit Distance by Lexicality","edit_distance_by_lexicality")

    # Frequency effect: real words only
    if "zipf_frequency" in df.columns:
        real_df = df[df["lexicality"] == "real"].copy()
        real_df["zipf_frequency"] = pd.to_numeric(real_df["zipf_frequency"], errors="coerce")
        real_df = real_df.dropna(subset=["zipf_frequency"])
        if len(real_df) > 0:
            for route in routes:
                figure_scatter(
                    real_df["zipf_frequency"].values,
                    real_df[f"{route}_exact_match"].values,
                    "Zipf Frequency", "Exact Match",
                    f"Frequency Effect — real words [{route}]",
                    os.path.join(fig_dir, f"frequency_effect_real_words_{route}.png"),
                )

    # Route comparison bar
    metric = "exact_match"
    groups = ["full", "wm", "ltm"]
    for lex in df["lexicality"].unique():
        vals = [df[df["lexicality"]==lex][f"{g}_{metric}"].mean() for g in groups]
        figure_bar(groups, vals,
                   f"Route accuracy — {lex}",
                   metric,
                   os.path.join(fig_dir, f"route_accuracy_barplot_{lex}.png"))


# ===========================================================  SSP eval  ====

def run_ssp_eval(model: DualRouteModel, vocab: Vocab, meta: dict,
                 tsv_path: str, out_dir: str, dry_run: bool,
                 device: str, wm_noise: bool = False,
                 decode: str = DECODE_TF) -> dict:
    df = pd.read_csv(tsv_path, sep="\t")
    df = df[~df["notes"].fillna("").str.contains("EXCLUDED", na=False)].copy()
    df = df.reset_index(drop=True)

    subdir = ("dry_run_ssp" if dry_run
              else ("ssp_ar" if decode == DECODE_AR else "ssp"))
    out_dir = os.path.join(out_dir, subdir)

    if dry_run:
        # Sample 20 items: 10 CCV + 10 VCC, across sonority values
        ccv  = df[df["type"] == "CCV"]
        vcc  = df[df["type"] == "VCC"]
        son_vals = sorted(df["sonority"].unique())
        # Try to cover multiple sonority values
        picked = []
        for sv in son_vals[:5]:
            for tp, sub in [("CCV", ccv), ("VCC", vcc)]:
                s = sub[sub["sonority"] == sv]
                if len(s) >= 2:
                    picked.append(s.sample(2, random_state=0))
        if len(picked) == 0:
            picked = [df.sample(n=min(20, len(df)), random_state=0)]
        df = pd.concat(picked).drop_duplicates("item_id").head(20).reset_index(drop=True)
        print(f"\n[SSP DRY RUN] {len(df)} items  -> {out_dir}")
    else:
        print(f"\n[SSP FULL EVAL ({decode})] {len(df)} items  -> {out_dir}")

    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # SSP: only use WM route (sub-lexical, sequential, the SSP-relevant route)
    # and full gated model for comparison.  LTM is included but is less
    # interpretable on 3-phoneme sequences with no semantic content.
    routes: Tuple[str, ...] = ("full", "wm", "ltm")

    forms_ids: List[List[int]] = []
    valid_idx: List[int] = []
    for i, row in df.iterrows():
        syms = row["target_phonemes"].split()
        ids  = [vocab.stoi[s] for s in syms if s in vocab.stoi]
        if len(ids) == len(syms):
            forms_ids.append(ids)
            valid_idx.append(i)

    df_valid = df.loc[valid_idx].reset_index(drop=True)
    n_valid  = len(forms_ids)
    print(f"  {n_valid} items with valid phoneme sequences")

    print(f"  Running {decode} inference …  "
          f"(WM noise: {'ON' if wm_noise else 'OFF — deterministic'})")
    results = evaluate_items(model, vocab, forms_ids, device, routes=routes,
                             wm_noise=wm_noise, decode=decode)

    for route in routes:
        for metric in ("exact_match", "phoneme_acc", "edit_dist", "norm_edit",
                       "predicted", "target"):
            df_valid[f"{route}_{metric}"] = [r[metric] for r in results[route]]

    pred_path = os.path.join(out_dir, "item_level_predictions.tsv")
    df_valid.to_csv(pred_path, sep="\t", index=False)
    print(f"  -> {pred_path}")

    # Summary by sonority and type
    summary_rows = []
    for col in ("sonority", "type"):
        if col not in df_valid.columns:
            continue
        for g in sorted(df_valid[col].dropna().unique()):
            sub = df_valid[df_valid[col] == g]
            row = {"group_by": col, "value": g, "n": len(sub)}
            for r in routes:
                for m in ("exact_match", "phoneme_acc", "edit_dist", "norm_edit"):
                    row[f"{r}_{m}"] = round(float(sub[f"{r}_{m}"].mean()), 4)
            summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "summary_table.tsv")
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print(f"  -> {summary_path}")

    overall = {}
    for route in routes:
        overall[route] = {
            m: round(float(df_valid[f"{route}_{m}"].mean()), 4)
            for m in ("exact_match", "phoneme_acc", "edit_dist", "norm_edit")
        }

    metrics = {
        "evaluation_regime":    _REGIME_NOTES[decode],
        "decode_mode":          decode,
        "checkpoint":           meta["checkpoint"],
        "glove_present":        meta["glove_present"],
        "wm_noise_enabled":     wm_noise,
        "deterministic":        not wm_noise,
        "n_items_evaluated":    n_valid,
        "dry_run":              dry_run,
        "overall":              overall,
        "note": (
            "SSP items are 3-phoneme CCV/VCC sequences. "
            "WM (dorsal) route is the primary target for SSP analysis. "
            "LTM and full results are provided for reference only."
        ),
    }
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  -> {metrics_path}")

    if not dry_run:
        _make_ssp_figures(df_valid, fig_dir, routes)

    print(f"\n  === SSP RESULTS ({decode}) ===")
    for route in routes:
        o = overall[route]
        print(f"  [{route:5s}] exact={o['exact_match']:.3f}  "
              f"phon_acc={o['phoneme_acc']:.3f}  "
              f"edit={o['edit_dist']:.2f}  ned={o['norm_edit']:.3f}")

    return metrics


def _make_ssp_figures(df: pd.DataFrame, fig_dir: str,
                      routes: Tuple[str, ...]) -> None:
    df = df.copy()
    df["sonority"] = pd.to_numeric(df["sonority"], errors="coerce")

    # Accuracy by Sonority (scatter + line per route)
    for route in routes:
        son_groups = df.groupby("sonority")[f"{route}_exact_match"].mean()
        if len(son_groups) > 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(son_groups.index, son_groups.values, marker="o", label=route)
            ax.set_xlabel("Sonority score")
            ax.set_ylabel("Exact match accuracy")
            ax.set_title(f"Accuracy by Sonority — {route} route")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, f"accuracy_by_sonority_{route}.png"), dpi=130)
            plt.close(fig)
            print(f"  [fig] accuracy_by_sonority_{route}.png")

    # Edit distance by Sonority
    for route in routes:
        son_groups = df.groupby("sonority")[f"{route}_edit_dist"].mean()
        if len(son_groups) > 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(son_groups.index, son_groups.values, marker="s",
                    color="darkorange", label=route)
            ax.set_xlabel("Sonority score")
            ax.set_ylabel("Mean edit distance")
            ax.set_title(f"Edit distance by Sonority — {route} route")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, f"edit_distance_by_sonority_{route}.png"), dpi=130)
            plt.close(fig)
            print(f"  [fig] edit_distance_by_sonority_{route}.png")

    # Accuracy by Type (CCV vs VCC)
    if "type" in df.columns:
        types = sorted(df["type"].dropna().unique())
        for route in routes:
            vals = [df[df["type"] == t][f"{route}_exact_match"].mean() for t in types]
            figure_bar(types, vals,
                       f"Accuracy by CCV/VCC — {route}",
                       "Exact match accuracy",
                       os.path.join(fig_dir, f"accuracy_by_type_CCV_VCC_{route}.png"))

    # Phoneme accuracy by Sonority
    for route in routes:
        son_groups = df.groupby("sonority")[f"{route}_phoneme_acc"].mean()
        if len(son_groups) > 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(son_groups.index, son_groups.values, marker="^",
                    color="forestgreen", label=route)
            ax.set_xlabel("Sonority score")
            ax.set_ylabel("Mean phoneme accuracy")
            ax.set_title(f"Phoneme accuracy by Sonority — {route} route")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(
                os.path.join(fig_dir, f"phoneme_accuracy_by_sonority_{route}.png"), dpi=130)
            plt.close(fig)
            print(f"  [fig] phoneme_accuracy_by_sonority_{route}.png")


# =============================================================  main  ======

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",    default=CKPT_PATH)
    p.add_argument("--wfe_tsv", default=WFE_TSV)
    p.add_argument("--ssp_tsv", default=SSP_TSV)
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--dry_run", action="store_true",
                   help=f"Run on {DRY_WFE_N} WFE + {DRY_SSP_N} SSP items only")
    p.add_argument("--wfe_only", action="store_true")
    p.add_argument("--ssp_only", action="store_true")
    p.add_argument("--device",  default=None,
                   help="cpu / cuda (auto-detected if omitted)")
    p.add_argument("--wm_noise", action="store_true",
                   help="Enable WM interference noise during evaluation (non-deterministic). "
                        "Default: OFF — evaluation is fully deterministic.")
    p.add_argument("--decode", default=DECODE_TF,
                   choices=[DECODE_TF, DECODE_AR],
                   help="Decoding regime: teacher_forced (default, upper-bound) or "
                        "autoregressive (model's own previous output as next input, "
                        "behaviorally plausible, errors propagate).  "
                        "AR outputs go to {out_dir}/wfe_ar/ and {out_dir}/ssp_ar/.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("  Yair-L3 External CSV Evaluation")
    print("=" * 60)
    print(f"  device      : {device}")
    print(f"  decode      : {args.decode}")
    print(f"  dry_run     : {args.dry_run}")
    print(f"  wm_noise    : {args.wm_noise}  "
          f"({'NON-DETERMINISTIC' if args.wm_noise else 'deterministic'})")
    print(f"  REGIME      : {_REGIME_NOTES[args.decode][:80]}…")
    print()

    model, vocab, meta = load_model_and_vocab(args.ckpt, device)

    all_results: dict = {
        "meta": meta,
        "evaluation_regime": _REGIME_NOTES[args.decode],
        "decode_mode": args.decode,
    }

    if not args.ssp_only:
        if not os.path.exists(args.wfe_tsv):
            print(f"[WARNING] WFE TSV not found: {args.wfe_tsv}")
            print("  Run first:  python scripts/convert_csvs.py")
        else:
            wfe_results = run_wfe_eval(
                model, vocab, meta, args.wfe_tsv, args.out_dir,
                args.dry_run, device, wm_noise=args.wm_noise,
                decode=args.decode)
            all_results["wfe"] = wfe_results

    if not args.wfe_only:
        if not os.path.exists(args.ssp_tsv):
            print(f"[WARNING] SSP TSV not found: {args.ssp_tsv}")
            print("  Run first:  python scripts/convert_csvs.py")
        else:
            ssp_results = run_ssp_eval(
                model, vocab, meta, args.ssp_tsv, args.out_dir,
                args.dry_run, device, wm_noise=args.wm_noise,
                decode=args.decode)
            all_results["ssp"] = ssp_results

    # Master summary
    summary_path = os.path.join(args.out_dir, "external_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[done] {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
