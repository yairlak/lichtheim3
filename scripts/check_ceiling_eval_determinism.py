"""Task 1: Verify that ceiling evaluation is deterministic across runs.

Evaluates the same checkpoint TWICE with the same settings, then compares
item-level predictions and error sets.  If evaluation is deterministic, both
runs must produce byte-identical predictions for every item.

Default checkpoint: checkpoints/lichtheim3_30k_glove_e60.pt

Outputs:
    outputs/train_lexicon_ceiling_determinism/report.json
    outputs/train_lexicon_ceiling_determinism/report.md
    outputs/train_lexicon_ceiling_determinism/run1_predictions.tsv  (optional)
    outputs/train_lexicon_ceiling_determinism/run2_predictions.tsv  (optional)
    outputs/train_lexicon_ceiling_determinism/discrepant_items.tsv  (if any)

Usage:
    python scripts/check_ceiling_eval_determinism.py
    python scripts/check_ceiling_eval_determinism.py \\
        --ckpt checkpoints/lichtheim3_30k_glove_e60.pt \\
        --wm_noise      # also test WM noise mode (expected: NOT deterministic)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

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

CKPT_DEFAULT = os.path.join(ROOT, "checkpoints", "lichtheim3_30k_glove_e60.pt")
OUT_DIR      = os.path.join(ROOT, "outputs", "train_lexicon_ceiling_determinism")
BATCH_SIZE   = 128


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",         default=CKPT_DEFAULT)
    p.add_argument("--lexicon_path", default=None)
    p.add_argument("--out_dir",      default=OUT_DIR)
    p.add_argument("--device",       default=None)
    p.add_argument("--wm_noise",     action="store_true",
                   help="run both passes with WM noise ON (expect non-determinism)")
    p.add_argument("--save_preds",   action="store_true",
                   help="save per-run predictions TSVs (large files)")
    p.add_argument("--n_items",      type=int, default=None,
                   help="limit to first N training items for a quick test")
    return p.parse_args()


def _load_model_and_data(ckpt_path: str, lexicon_path_override: str | None,
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
    elif not hasattr(cfg.data, "lexicon_path"):
        cfg.data.lexicon_path = None

    vocab   = build_vocab()
    lexicon = build_lexicon(cfg.data, vocab)
    train_entries, _ = lexicon.split(cfg.data.val_fraction, cfg.data.seed)

    bank = torch.stack([torch.tensor(e.semantic) for e in train_entries]).float().to(device)
    model = DualRouteModel(cfg, vocab).to(device)
    model.set_semantic_bank(bank)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, vocab, train_entries, cfg


@torch.no_grad()
def _single_eval_pass(model, vocab, entries, device: str,
                       wm_noise: bool) -> pd.DataFrame:
    routes = ("full", "wm", "ltm")
    rows = []
    forms_ids = [e.phonemes for e in entries]

    for start in range(0, len(forms_ids), BATCH_SIZE):
        bforms   = forms_ids[start: start + BATCH_SIZE]
        bentries = entries[start: start + BATCH_SIZE]
        batch    = make_batch(bforms, vocab, device)

        preds_by_route = {}
        for route in routes:
            collect = (route == "wm") and wm_noise
            p, _    = route_predictions(model, batch, route=route, collect=collect)
            preds_by_route[route] = p

        for i, (entry, fids) in enumerate(zip(bentries, bforms)):
            row = {"word": entry.word, "rank": entry.rank, "length": len(fids)}
            for route in routes:
                pred = preds_by_route[route][i, :len(fids)].tolist()
                row[f"{route}_pred"] = " ".join(vocab.itos[idx] for idx in pred)
                tgt  = [vocab.itos[idx] for idx in fids]
                row[f"{route}_exact"] = int(
                    [vocab.itos[idx] for idx in pred] == tgt
                )
            rows.append(row)

    return pd.DataFrame(rows)


def compare(df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
    assert list(df1["word"]) == list(df2["word"]), \
        "Word ordering differs between runs — determinism check invalid."

    routes = ["full", "wm", "ltm"]
    discrepant_words: List[str] = []
    n = len(df1)

    col_pairs = {route: f"{route}_pred" for route in routes}
    disagree_mask = pd.Series([False] * n)
    for col in col_pairs.values():
        disagree_mask = disagree_mask | (df1[col] != df2[col])

    discrepant = df1[disagree_mask].copy()
    discrepant["run2_full_pred"] = df2.loc[disagree_mask, "full_pred"].values
    discrepant["run2_wm_pred"]   = df2.loc[disagree_mask, "wm_pred"].values
    discrepant["run2_ltm_pred"]  = df2.loc[disagree_mask, "ltm_pred"].values

    n_discrepant = int(disagree_mask.sum())
    is_deterministic = (n_discrepant == 0)

    per_route = {}
    for route in routes:
        col = f"{route}_pred"
        n_diff = int((df1[col] != df2[col]).sum())
        per_route[route] = {
            "n_different_predictions": n_diff,
            "pct_different":           round(100.0 * n_diff / max(n, 1), 3),
        }

    return {
        "n_items":          n,
        "n_discrepant":     n_discrepant,
        "is_deterministic": is_deterministic,
        "per_route":        per_route,
        "discrepant_df":    discrepant,
    }


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n[determinism_check] Checkpoint : {args.ckpt}")
    print(f"  WM noise mode : {'ON' if args.wm_noise else 'OFF (expected: deterministic)'}")

    model, vocab, train_entries, cfg = _load_model_and_data(
        args.ckpt, args.lexicon_path, device)

    entries = train_entries[:args.n_items] if args.n_items else train_entries
    print(f"  Items to evaluate : {len(entries)}")

    print("\n  Run 1 …")
    torch.manual_seed(0)    # fix seed before run 1
    df1 = _single_eval_pass(model, vocab, entries, device, args.wm_noise)

    print("  Run 2 …")
    torch.manual_seed(0)    # same seed before run 2
    df2 = _single_eval_pass(model, vocab, entries, device, args.wm_noise)

    cmp = compare(df1, df2)
    discrepant_df = cmp.pop("discrepant_df")

    print(f"\n  === DETERMINISM RESULT ===")
    print(f"  Items evaluated  : {cmp['n_items']}")
    print(f"  Discrepant items : {cmp['n_discrepant']}")
    print(f"  Deterministic    : {cmp['is_deterministic']}")
    for route, d in cmp["per_route"].items():
        print(f"    [{route}]  different predictions: "
              f"{d['n_different_predictions']} ({d['pct_different']}%)")

    if cmp["n_discrepant"] > 0:
        disc_path = os.path.join(args.out_dir, "discrepant_items.tsv")
        discrepant_df.to_csv(disc_path, sep="\t", index=False)
        print(f"\n  Discrepant items -> {disc_path}")

    if args.save_preds:
        p1 = os.path.join(args.out_dir, "run1_predictions.tsv")
        p2 = os.path.join(args.out_dir, "run2_predictions.tsv")
        df1.to_csv(p1, sep="\t", index=False)
        df2.to_csv(p2, sep="\t", index=False)
        print(f"  Run 1 preds -> {p1}")
        print(f"  Run 2 preds -> {p2}")

    report = {
        "checkpoint":          args.ckpt,
        "wm_noise_enabled":    args.wm_noise,
        "n_items_checked":     args.n_items or len(train_entries),
        "torch_seed_before_each_run": 0,
        **cmp,
    }

    json_path = os.path.join(args.out_dir, "report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  -> {json_path}")

    _write_md(report, cmp, args.out_dir)


def _write_md(report: dict, cmp: dict, out_dir: str) -> None:
    det = report["is_deterministic"]
    noise = report["wm_noise_enabled"]
    lines = [
        "# Ceiling Evaluation Determinism Check",
        "",
        "> Two identical evaluation passes were run on the same checkpoint with the",
        "> same `torch.manual_seed(0)` before each pass.  Predictions are compared",
        "> item-by-item.",
        "",
        "## Configuration",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| Checkpoint | `{report['checkpoint']}` |",
        f"| WM noise enabled | {'YES (non-deterministic mode)' if noise else 'NO (deterministic mode)'} |",
        f"| Items checked | {report['n_items_checked']} |",
        "",
        "## Result",
        "",
        f"**{'✓ DETERMINISTIC' if det else '✗ NON-DETERMINISTIC'}** — "
        f"{report['n_discrepant']} discrepant items out of {report['n_items_checked']}",
        "",
        "| Route | Different predictions | % |",
        "|---|---|---|",
    ]
    for route, d in report["per_route"].items():
        lines.append(
            f"| {route} | {d['n_different_predictions']} | {d['pct_different']}% |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
    ]
    if det and not noise:
        lines += [
            "Evaluation is fully deterministic.",
            "",
            "The earlier e60 vs e90 discrepancy was caused by the dead first loop",
            "in `evaluate_forms` calling the WM route twice per batch with different",
            "noise draws — now fixed.  The WM noise is disabled by default in",
            "`evaluate_train_lexicon_ceiling.py` (collect=False) so ceiling results",
            "are now stable across runs.",
        ]
    elif not det and noise:
        lines += [
            "Non-determinism is expected because WM interference noise is enabled.",
            "Run without `--wm_noise` for a stable ceiling evaluation.",
        ]
    elif not det and not noise:
        lines += [
            "⚠ Unexpected non-determinism even with WM noise disabled.",
            "Investigate other sources of randomness:",
            "- dropout (model should be in eval() mode)",
            "- DataLoader shuffle (should be off)",
            "- CUDA non-deterministic ops",
            "Check model.training flag and DataLoader shuffle settings.",
        ]
    path = os.path.join(out_dir, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
