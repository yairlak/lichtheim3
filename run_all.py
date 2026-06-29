"""End-to-end: train the dual-route model and run all evaluations.

    python run_all.py --epochs 8

Writes figures and a JSON summary into ./outputs/.
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
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_words", type=int, default=1200)
    p.add_argument("--no_real", action="store_true",
                   help="force the synthetic lexicon")
    p.add_argument("--out_dir", default=None)
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
    out_dir = args.out_dir or cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)

    model, vocab, lexicon, history = build_and_train(cfg, out_dir=out_dir)

    summary = {"lexicon_source": lexicon.source,
               "n_words": len(lexicon), "final_loss": history[-1]}
    print("\n==== evaluations ====")
    summary["generalization"] = generalization.run(model, vocab, lexicon, cfg, out_dir)
    summary["primacy_recency"] = primacy_recency.run(model, vocab, cfg, out_dir)
    summary["neighborhood"] = neighborhood.run(model, vocab, lexicon, cfg, out_dir)
    summary["sonority"] = sonority.run(model, vocab, cfg, out_dir)
    summary["dissociation"] = dissociation.run(model, vocab, lexicon, cfg, out_dir)
    summary["ablation"] = ablation.run(model, vocab, lexicon, cfg, out_dir)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] figures + summary.json in {out_dir}/")


if __name__ == "__main__":
    main()
