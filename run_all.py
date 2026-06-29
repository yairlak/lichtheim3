"""End-to-end (PyTorch): train the dual-route model and run every evaluation,
writing all figures into an organized tree:

    figures/
      train/      training_loss.png
      eval/       generalization, primacy_recency, neighborhood, sonority,
                  dissociation (frequency x length)
      ablation/   ablation_severity, ablation_dissociation, ablation_length

Usage:
    pip install -r requirements.txt
    python run_all.py                 # quick run (small lexicon)
    python run_all.py --epochs 15 --max_words 8000   # fuller run

For real semantic targets run `bash data/get_glove.sh` first (otherwise the
ventral route aligns to deterministic per-word pseudo-vectors).
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from config import default_config
from train import build_and_train
from evaluate import (primacy_recency, neighborhood, sonority, dissociation,
                      generalization, ablation)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_words", type=int, default=4000,
                   help="lexicon size (up to 30000; larger is slower)")
    p.add_argument("--no_real", action="store_true",
                   help="force the synthetic lexicon")
    p.add_argument("--fig_dir", default="figures")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = default_config()
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.seed = args.seed
    cfg.data.seed = args.seed
    cfg.data.max_words = args.max_words
    if args.no_real:
        cfg.data.use_real = False
    cfg.train.device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dir = os.path.join(args.fig_dir, "train")
    eval_dir = os.path.join(args.fig_dir, "eval")
    abl_dir = os.path.join(args.fig_dir, "ablation")
    for d in (train_dir, eval_dir, abl_dir):
        os.makedirs(d, exist_ok=True)

    model, vocab, lexicon, history = build_and_train(cfg, out_dir=train_dir)

    summary = {"lexicon_source": lexicon.source, "n_words": len(lexicon),
               "final_loss": history[-1]}
    print("\n==== evaluations ====")
    summary["generalization"] = generalization.run(model, vocab, lexicon, cfg, eval_dir)
    summary["primacy_recency"] = primacy_recency.run(model, vocab, cfg, eval_dir)
    summary["neighborhood"] = neighborhood.run(model, vocab, lexicon, cfg, eval_dir)
    summary["sonority"] = sonority.run(model, vocab, cfg, eval_dir)
    summary["dissociation"] = dissociation.run(model, vocab, lexicon, cfg, eval_dir)
    summary["ablation"] = ablation.run(model, vocab, lexicon, cfg, abl_dir)

    with open(os.path.join(args.fig_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] figures + summary.json under {args.fig_dir}/ (train, eval, ablation)")


if __name__ == "__main__":
    main()
