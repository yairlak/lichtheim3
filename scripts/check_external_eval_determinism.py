"""Verify that external WFE evaluation is deterministic across runs.

Runs WFE evaluation twice on the same checkpoint with the same settings
(wm_noise=False), then compares item-level predictions for full / WM / LTM routes.
If evaluation is deterministic, both runs must produce identical predictions for
every item.

Expected result after the collect=True fix: 0 discrepant items.

Outputs:
    outputs/external_eval_determinism/report.json
    outputs/external_eval_determinism/report.md
    outputs/external_eval_determinism/discrepant_items.tsv  (only if discrepancies found)

Usage:
    python scripts/check_external_eval_determinism.py
    python scripts/check_external_eval_determinism.py \\
        --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt \\
        --wfe_tsv data/eval_external/wfe_eval.tsv
    python scripts/check_external_eval_determinism.py --wm_noise  # expect NON-deterministic
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

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

CKPT_DEFAULT  = os.path.join(ROOT, "checkpoints",
                              "lichtheim3_30k_glove_e60_to_e120_lowlr.pt")
WFE_DEFAULT   = os.path.join(ROOT, "data", "eval_external", "wfe_eval.tsv")
OUT_DEFAULT   = os.path.join(ROOT, "outputs", "external_eval_determinism")
BATCH_SIZE    = 64
ROUTES        = ("full", "wm", "ltm")


# ---------------------------------------------------------------------------
# Model loading (same pattern as external_eval.py)
# ---------------------------------------------------------------------------

def _load(ckpt_path: str, device: str):
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
    return model, vocab


# ---------------------------------------------------------------------------
# Single evaluation pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_pass(model, vocab, forms_ids: List[List[int]],
               device: str, wm_noise: bool) -> pd.DataFrame:
    rows: List[dict] = []
    for start in range(0, len(forms_ids), BATCH_SIZE):
        batch_forms = forms_ids[start: start + BATCH_SIZE]
        batch = make_batch(batch_forms, vocab, device)
        for route in ROUTES:
            collect = (route == "wm") and wm_noise
            preds, _ = route_predictions(model, batch, route=route, collect=collect)
            for i, fids in enumerate(batch_forms):
                n_steps  = len(fids)
                pred_ids = preds[i, :n_steps].tolist()
                pred_str = " ".join(vocab.itos[idx] for idx in pred_ids)
                tgt_str  = " ".join(vocab.itos[idx] for idx in fids)
                rows.append({
                    "item_idx": start + i,
                    "route":    route,
                    "predicted": pred_str,
                    "target":    tgt_str,
                    "exact_match": int(pred_str == tgt_str),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _compare(df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
    # Pivot to wide: one row per item_idx, one column per route_predicted
    p1 = df1.pivot(index="item_idx", columns="route", values="predicted")
    p2 = df2.pivot(index="item_idx", columns="route", values="predicted")

    assert list(p1.index) == list(p2.index), "Item ordering differs between runs"

    n = len(p1)
    per_route: Dict[str, dict] = {}
    disagree_mask = pd.Series([False] * n, index=p1.index)

    for route in ROUTES:
        if route not in p1.columns or route not in p2.columns:
            continue
        diff = (p1[route] != p2[route])
        n_diff = int(diff.sum())
        per_route[route] = {"n_different": n_diff,
                            "pct": round(100.0 * n_diff / max(n, 1), 3)}
        disagree_mask = disagree_mask | diff

    n_disc = int(disagree_mask.sum())
    return {
        "n_items":          n,
        "n_discrepant":     n_disc,
        "is_deterministic": n_disc == 0,
        "per_route":        per_route,
        "disagree_mask":    disagree_mask,
    }


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _write_json(report: dict, path: str) -> None:
    safe = {k: v for k, v in report.items() if k != "disagree_mask"}
    with open(path, "w") as f:
        json.dump(safe, f, indent=2, default=str)
    print(f"  -> {path}")


def _write_md(report: dict, path: str) -> None:
    det  = report["is_deterministic"]
    noise = report["wm_noise_enabled"]
    lines = [
        "# External Evaluation Determinism Check",
        "",
        "> Two identical WFE evaluation passes were run on the same checkpoint.",
        "> Predictions are compared item-by-item across all three routes.",
        "",
        "## Configuration",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| Checkpoint | `{report['checkpoint']}` |",
        f"| WFE items | {report['n_items']} |",
        f"| WM noise enabled | {'YES (non-deterministic mode)' if noise else 'NO (deterministic mode)'} |",
        "",
        "## Result",
        "",
        f"**{'✓ DETERMINISTIC' if det else '✗ NON-DETERMINISTIC'}** — "
        f"{report['n_discrepant']} discrepant items out of {report['n_items']}",
        "",
        "| Route | Different predictions | % |",
        "|---|---|---|",
    ]
    for route, d in report["per_route"].items():
        lines.append(f"| {route} | {d['n_different']} | {d['pct']}% |")

    lines += ["", "## Interpretation", ""]
    if det and not noise:
        lines += [
            "Evaluation is fully deterministic: the WM interference noise fix is working.",
            "",
            "The `collect=False` default in `external_eval.py` means the WM route's",
            "GRU hidden state is not perturbed by Gaussian noise during inference.",
            "Repeated runs on the same checkpoint will always produce identical scores.",
        ]
    elif not det and noise:
        lines += [
            "Non-determinism is expected: WM noise is enabled (`--wm_noise`).",
            "Run without `--wm_noise` (the default) for stable results.",
        ]
    elif not det and not noise:
        lines += [
            "⚠  Unexpected non-determinism even with WM noise disabled.",
            "",
            "Possible causes:",
            "- Model is not in `eval()` mode",
            "- A different source of stochasticity in the model",
            "- CUDA non-deterministic ops (if running on GPU)",
            "",
            "Check: `model.training` should be False during inference.",
            "If on CUDA, set `torch.backends.cudnn.deterministic = True` to rule out GPU ops.",
        ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",    default=CKPT_DEFAULT)
    p.add_argument("--wfe_tsv", default=WFE_DEFAULT)
    p.add_argument("--out_dir", default=OUT_DEFAULT)
    p.add_argument("--device",  default=None)
    p.add_argument("--wm_noise", action="store_true",
                   help="Enable WM noise (expected result: NON-deterministic)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n[check_external_eval_determinism]")
    print(f"  Checkpoint : {args.ckpt}")
    print(f"  WFE TSV    : {args.wfe_tsv}")
    print(f"  WM noise   : {'ON (expect NON-deterministic)' if args.wm_noise else 'OFF (expect deterministic)'}")

    # Load model
    model, vocab = _load(args.ckpt, device)

    # Load and parse WFE
    if not os.path.exists(args.wfe_tsv):
        print(f"\nERROR: WFE TSV not found: {args.wfe_tsv}")
        print("  Run: python scripts/convert_csvs.py")
        sys.exit(1)

    df_wfe = pd.read_csv(args.wfe_tsv, sep="\t")
    df_wfe = df_wfe[~df_wfe["notes"].fillna("").str.contains("EXCLUDED", na=False)]
    df_wfe = df_wfe.reset_index(drop=True)

    forms_ids: List[List[int]] = []
    item_words: List[str] = []
    for _, row in df_wfe.iterrows():
        syms = row["target_phonemes"].split()
        ids  = [vocab.stoi[s] for s in syms if s in vocab.stoi]
        if len(ids) == len(syms):
            forms_ids.append(ids)
            item_words.append(str(row.get("word", "")))

    print(f"  {len(forms_ids)} valid WFE items")

    # Two passes
    print("\n  Run 1 …")
    torch.manual_seed(0)
    df_run1 = _eval_pass(model, vocab, forms_ids, device, args.wm_noise)

    print("  Run 2 …")
    torch.manual_seed(0)
    df_run2 = _eval_pass(model, vocab, forms_ids, device, args.wm_noise)

    cmp = _compare(df_run1, df_run2)

    print(f"\n  === DETERMINISM RESULT ===")
    print(f"  Items          : {cmp['n_items']}")
    print(f"  Discrepant     : {cmp['n_discrepant']}")
    print(f"  Deterministic  : {cmp['is_deterministic']}")
    for route, d in cmp["per_route"].items():
        print(f"    [{route}]  different: {d['n_different']} ({d['pct']}%)")

    # Save discrepant items if any
    if cmp["n_discrepant"] > 0:
        disagree_mask = cmp["disagree_mask"]
        disc_items = []
        for item_idx in disagree_mask[disagree_mask].index:
            r1_sub = df_run1[df_run1["item_idx"] == item_idx].set_index("route")
            r2_sub = df_run2[df_run2["item_idx"] == item_idx].set_index("route")
            word   = item_words[item_idx] if item_idx < len(item_words) else "?"
            for route in ROUTES:
                if route not in r1_sub.index:
                    continue
                p1 = r1_sub.loc[route, "predicted"]
                p2 = r2_sub.loc[route, "predicted"]
                if p1 != p2:
                    disc_items.append({
                        "item_idx": item_idx,
                        "word":     word,
                        "route":    route,
                        "target":   r1_sub.loc[route, "target"],
                        "run1_pred": p1,
                        "run2_pred": p2,
                    })
        disc_df = pd.DataFrame(disc_items)
        disc_path = os.path.join(args.out_dir, "discrepant_items.tsv")
        disc_df.to_csv(disc_path, sep="\t", index=False)
        print(f"\n  Discrepant items -> {disc_path}")

    report = {
        "checkpoint":       args.ckpt,
        "wfe_tsv":          args.wfe_tsv,
        "wm_noise_enabled": args.wm_noise,
        "n_items":          cmp["n_items"],
        "n_discrepant":     cmp["n_discrepant"],
        "is_deterministic": cmp["is_deterministic"],
        "per_route":        cmp["per_route"],
    }

    _write_json(report, os.path.join(args.out_dir, "report.json"))
    _write_md(report,  os.path.join(args.out_dir, "report.md"))

    print(f"\n[check_external_eval_determinism] Done.  Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
