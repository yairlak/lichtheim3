"""Step 7: Ceiling evaluation on all trained words.

After training, verify that the model has achieved (near-)perfect exact-match
accuracy on every word it was trained on.  Imperfect ceiling means the model
is still underfitting and needs more epochs.

IMPORTANT: uses teacher-forced decoding (same regime as all evaluate/*.py).
           A ceiling of 1.000 means the model can produce every training word
           correctly given the correct prefix — NOT that it can free-generate them.

Prerequisite: a checkpoint created by scripts/train_checkpoint.py.

Outputs (all in outputs/train_lexicon_ceiling/):
    metrics.json           overall exact-match per route
    summary.md             human-readable summary
    item_level_predictions.tsv   per-word predictions
    train_errors.tsv       training words the full route still gets wrong

Success criterion: full_exact_match = 1.000 AND train_errors.tsv is empty.

Usage:
    python scripts/evaluate_train_lexicon_ceiling.py \\
        --ckpt checkpoints/lichtheim3_30k_glove.pt

    # With explicit lexicon path (must match what was used for training):
    python scripts/evaluate_train_lexicon_ceiling.py \\
        --ckpt checkpoints/lichtheim3_30k_glove.pt \\
        --lexicon_path data/lexicon_en_glove_covered.tsv

    # Include validation split:
    python scripts/evaluate_train_lexicon_ceiling.py \\
        --ckpt checkpoints/lichtheim3_30k_glove.pt \\
        --include_val
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import Config, DataConfig, WMConfig, LTMConfig, GatingConfig, LossConfig, TrainConfig
from data.phonemes import build_vocab, Vocab
from data.lexicon import build_lexicon
from models.dual_route import DualRouteModel
from evaluate.hooks import make_batch, route_predictions

CKPT_DEFAULT = os.path.join(ROOT, "checkpoints", "lichtheim3.pt")
OUT_DIR      = os.path.join(ROOT, "outputs", "train_lexicon_ceiling")
BATCH_SIZE   = 128

EVAL_NOTE = (
    "Teacher-forced decoding: gold prefix fed at each decoder step.  "
    "Accuracy reflects reconstruction quality, NOT free-generation."
)

# For ceiling evaluation we deliberately use collect=False for all routes,
# which disables WM interference noise (wm_route.py applies noise when
# self.training OR collect).  Ceiling evaluation tests memorisation; the
# noise is not wanted and makes results non-deterministic across runs.
# Use --wm_noise to enable noise for consistency with external eval scripts.
CEILING_COLLECT_WM_NOISE = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",         default=CKPT_DEFAULT)
    p.add_argument("--lexicon_path", default=None,
                   help="lexicon TSV used for training (overrides cfg.lexicon_path "
                        "saved in checkpoint if provided)")
    p.add_argument("--out_dir",      default=OUT_DIR)
    p.add_argument("--include_val",  action="store_true",
                   help="also evaluate the validation/held-out split")
    p.add_argument("--wm_noise",     action="store_true",
                   help=(
                       "enable WM interference noise during evaluation "
                       "(makes WM predictions non-deterministic; off by default "
                       "for ceiling evaluation — use for external eval consistency)"
                   ))
    p.add_argument("--device",       default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Levenshtein distance (no external library)
# ---------------------------------------------------------------------------

def _edit_distance(a: list, b: list) -> int:
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            dp[j] = prev[j-1] if a[i-1] == b[j-1] else 1 + min(prev[j-1], prev[j], dp[j-1])
    return dp[m]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_forms(model: DualRouteModel, vocab: Vocab,
                   entries, device: str,
                   routes: Tuple[str, ...] = ("full", "wm", "ltm"),
                   wm_noise: bool = CEILING_COLLECT_WM_NOISE) -> List[dict]:
    """Teacher-forced evaluation of a list of LexEntry objects.

    wm_noise=False (default for ceiling eval) passes collect=False to the WM
    route, disabling interference noise and making predictions fully
    deterministic.  wm_noise=True restores the noisy behaviour used by
    evaluate/*.py for serial-position experiments.
    """
    results = []
    forms_ids = [e.phonemes for e in entries]
    n = len(forms_ids)

    for start in range(0, n, BATCH_SIZE):
        batch_forms = forms_ids[start: start + BATCH_SIZE]
        batch_entries = entries[start: start + BATCH_SIZE]
        batch = make_batch(batch_forms, vocab, device)

        # Single forward pass per route — no dead first loop.
        # collect=True for WM route ONLY if wm_noise is explicitly requested.
        batch_preds: Dict[str, torch.Tensor] = {}
        for route in routes:
            collect = (route == "wm") and wm_noise
            p, _ = route_predictions(model, batch, route=route, collect=collect)
            batch_preds[route] = p

        for i, (entry, form_ids) in enumerate(zip(batch_entries, batch_forms)):
            row: dict = {
                "word":             entry.word,
                "rank":             entry.rank,
                "target_phonemes":  " ".join(vocab.itos[idx] for idx in form_ids),
                "length":           len(form_ids),
                "zipf_approx":      round(float(entry.freq), 6),
            }
            for route in routes:
                pred = batch_preds[route][i, :len(form_ids)].tolist()
                pred_syms = [vocab.itos[idx] for idx in pred]
                tgt_syms  = [vocab.itos[idx] for idx in form_ids]
                exact     = int(pred_syms == tgt_syms)
                ed        = _edit_distance(tgt_syms, pred_syms)
                from itertools import zip_longest
                phon_acc  = (
                    sum(a == b for a, b in zip_longest(pred_syms, tgt_syms, fillvalue=""))
                    / max(len(tgt_syms), 1)
                )
                row[f"{route}_exact_match"]  = exact
                row[f"{route}_phoneme_acc"]  = round(phon_acc, 4)
                row[f"{route}_edit_dist"]    = ed
                row[f"{route}_predicted"]    = " ".join(pred_syms)
            results.append(row)

        if (start // BATCH_SIZE) % 20 == 0:
            print(f"  … {min(start + BATCH_SIZE, n)}/{n}", end="\r")

    print()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.ckpt):
        print(f"\nERROR: checkpoint not found: {args.ckpt}\n"
              "Run: python scripts/train_checkpoint.py --max_words 30000 --epochs 30 --seed 0\n")
        sys.exit(1)

    print(f"\n[ceiling_eval] Loading {args.ckpt} …")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    cfg = Config(
        data   = DataConfig(**ckpt["cfg_data"]),
        wm     = WMConfig(**ckpt["cfg_wm"]),
        ltm    = LTMConfig(**ckpt["cfg_ltm"]),
        gating = GatingConfig(**ckpt["cfg_gating"]),
        loss   = LossConfig(**ckpt["cfg_loss"]),
        train  = TrainConfig(**ckpt["cfg_train"]),
    )
    cfg.train.device = device
    # Allow CLI override of lexicon path (e.g. when checkpoint was saved before
    # lexicon_path was added to DataConfig)
    if args.lexicon_path:
        cfg.data.lexicon_path = args.lexicon_path
    elif not hasattr(cfg.data, "lexicon_path"):
        cfg.data.lexicon_path = None

    vocab   = build_vocab()
    lexicon = build_lexicon(cfg.data, vocab)
    train_entries, val_entries = lexicon.split(cfg.data.val_fraction, cfg.data.seed)

    bank = torch.stack([torch.tensor(e.semantic) for e in train_entries]).float().to(device)
    model = DualRouteModel(cfg, vocab).to(device)
    model.set_semantic_bank(bank)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"  checkpoint : {args.ckpt}")
    print(f"  glove      : {'present' if ckpt.get('glove_present') else 'ABSENT (pseudo-vectors)'}")
    print(f"  lexicon    : {lexicon.source}  max_words={cfg.data.max_words}")
    print(f"  n_train    : {len(train_entries)}  n_val: {len(val_entries)}")
    print(f"  device     : {device}")
    print(f"  regime     : {EVAL_NOTE[:60]}…")

    routes = ("full", "wm", "ltm")
    all_rows = []
    wm_noise = getattr(args, "wm_noise", False)
    if wm_noise:
        print("  WM noise : ENABLED  (non-deterministic)")
    else:
        print("  WM noise : DISABLED (deterministic ceiling eval)")

    # ---- Training split ----
    print(f"\n[ceiling_eval] Evaluating {len(train_entries)} training words …")
    train_results = evaluate_forms(model, vocab, train_entries, device, routes,
                                   wm_noise=wm_noise)
    for r in train_results:
        r["split"] = "train"
    all_rows.extend(train_results)

    # ---- Validation split (optional) ----
    if args.include_val:
        print(f"[ceiling_eval] Evaluating {len(val_entries)} validation words …")
        val_results = evaluate_forms(model, vocab, val_entries, device, routes,
                                     wm_noise=wm_noise)
        for r in val_results:
            r["split"] = "val"
        all_rows.extend(val_results)

    # ---- Results ----
    df = pd.DataFrame(all_rows)
    pred_path = os.path.join(args.out_dir, "item_level_predictions.tsv")
    df.to_csv(pred_path, sep="\t", index=False)
    print(f"\n  -> {pred_path}")

    # Per-split overall metrics
    metrics: dict = {
        "evaluation_note":   EVAL_NOTE,
        "wm_noise_enabled":  wm_noise,
        "deterministic":     not wm_noise,
        "checkpoint":        args.ckpt,
        "glove_present":     bool(ckpt.get("glove_present", False)),
        "lexicon_source":    lexicon.source,
        "n_train":           len(train_entries),
        "n_val":             len(val_entries),
        "cfg_max_words":     cfg.data.max_words,
        "cfg_epochs":        cfg.train.epochs,
        "cfg_seed":          cfg.train.seed,
        "splits_evaluated":  ["train"] + (["val"] if args.include_val else []),
        "results": {},
    }
    for split in metrics["splits_evaluated"]:
        sub = df[df["split"] == split]
        metrics["results"][split] = {}
        for route in routes:
            metrics["results"][split][route] = {
                "exact_match": round(float(sub[f"{route}_exact_match"].mean()), 6),
                "phoneme_acc": round(float(sub[f"{route}_phoneme_acc"].mean()), 6),
                "edit_dist":   round(float(sub[f"{route}_edit_dist"].mean()),   4),
                "n_items":     len(sub),
                "n_errors":    int((sub[f"{route}_exact_match"] == 0).sum()),
            }

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  -> {metrics_path}")

    # Error table: training words the full route gets wrong
    train_df = df[df["split"] == "train"]
    errors_df = train_df[train_df["full_exact_match"] == 0].copy()
    errors_df = errors_df.sort_values("full_edit_dist", ascending=False)
    err_path = os.path.join(args.out_dir, "train_errors.tsv")
    errors_df.to_csv(err_path, sep="\t", index=False)
    print(f"  -> {err_path}  ({len(errors_df)} errors)")

    # Summary markdown
    _write_summary_md(metrics, len(errors_df), args.out_dir)

    # Console summary
    print(f"\n  === CEILING EVALUATION ===")
    print(f"  Note: {EVAL_NOTE[:70]}…")
    for split in metrics["splits_evaluated"]:
        r = metrics["results"][split]
        print(f"\n  [{split}]")
        for route in routes:
            d = r[route]
            ok = "✓ CEILING" if d["exact_match"] == 1.0 else "✗ NOT YET"
            print(f"    {route:5s}  exact={d['exact_match']:.4f}  "
                  f"errors={d['n_errors']}  {ok}")
    if len(errors_df) == 0:
        print("\n  CEILING REACHED: train_errors.tsv is empty.")
    else:
        print(f"\n  {len(errors_df)} training errors remain.  "
              f"Consider more epochs or larger model.")


def _write_summary_md(metrics: dict, n_train_errors: int, out_dir: str) -> None:
    lines = [
        "# Train Lexicon Ceiling Evaluation",
        "",
        f"> {metrics['evaluation_note']}",
        "",
        "## Provenance",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Checkpoint | `{metrics['checkpoint']}` |",
        f"| GloVe present | {'YES' if metrics['glove_present'] else 'NO (pseudo-vectors)'} |",
        f"| Lexicon source | {metrics['lexicon_source']} |",
        f"| max_words | {metrics['cfg_max_words']} |",
        f"| epochs | {metrics['cfg_epochs']} |",
        f"| seed | {metrics['cfg_seed']} |",
        f"| n_train | {metrics['n_train']} |",
        f"| n_val | {metrics['n_val']} |",
        "",
        "## Results",
        "",
    ]
    for split, res in metrics["results"].items():
        lines += [f"### {split.capitalize()} split", "", "| Route | Exact match | Phoneme acc | Edit dist | Errors |", "|---|---|---|---|---|"]
        for route, d in res.items():
            ceiling = " ✓ CEILING" if d["exact_match"] == 1.0 else ""
            lines.append(
                f"| {route} | {d['exact_match']:.4f}{ceiling} | "
                f"{d['phoneme_acc']:.4f} | {d['edit_dist']:.3f} | {d['n_errors']} |"
            )
        lines.append("")

    lines += [
        "## Success criterion",
        "",
        "- `full_exact_match` on train split = **1.0000**",
        "- `train_errors.tsv` is empty",
        "",
        "If not met: increase `--epochs` and retrain.  If full_exact_match is high "
        "but not 1.0, inspect `train_errors.tsv` for patterns (unusual phoneme sequences, "
        "long words, proper nouns).",
    ]
    path = os.path.join(out_dir, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
