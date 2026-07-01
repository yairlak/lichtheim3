"""Train the Yair-L3 dual-route model and save a checkpoint.

This script wraps the existing train.build_and_train pipeline and adds
a torch.save() call so that a weight file is available for downstream
external evaluations.  It does NOT touch architecture / losses / gate.

Usage (from repo root):
    python scripts/train_checkpoint.py                    # defaults (4k words, 10 ep)
    python scripts/train_checkpoint.py --max_words 30000 --epochs 30 --seed 0
    python scripts/train_checkpoint.py --ckpt checkpoints/lichtheim3.pt

The checkpoint is a plain dict with keys:
    model_state_dict   : model.state_dict()
    cfg_data           : dataclass-dict of DataConfig
    cfg_wm             : dataclass-dict of WMConfig
    cfg_ltm            : dataclass-dict of LTMConfig
    cfg_gating         : dataclass-dict of GatingConfig
    cfg_loss           : dataclass-dict of LossConfig
    cfg_train          : dataclass-dict of TrainConfig
    history            : list[dict]  training-loss history
    lexicon_source     : str
    n_train            : int         number of training entries
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import subprocess
import sys

import torch

# --------------------------------------------------------------------------- #
# Make sure repo root is on PYTHONPATH regardless of how this script is called #
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import default_config
from train import build_and_train, build_everything
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Yair-L3 and save a checkpoint for external evaluation"
    )
    p.add_argument("--epochs",    type=int, default=10,
                   help=(
                       "training epochs  (default 10).  "
                       "WARNING: the committed figures/summary.json was produced "
                       "with --epochs 30 --max_words 4000 --seed 0.  "
                       "Use those flags to reproduce it exactly."
                   ))
    p.add_argument("--max_words", type=int, default=4000,
                   help="lexicon size (default 4000; up to 30000)")
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--batch_size",type=int, default=64)
    p.add_argument("--ckpt",      type=str,
                   default=os.path.join(ROOT, "checkpoints", "lichtheim3.pt"),
                   help="path to write the checkpoint file")
    p.add_argument("--out_dir",   type=str,
                   default=os.path.join(ROOT, "outputs", "train_run"),
                   help="directory for training figures (loss curve)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = default_config()
    cfg.train.epochs     = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.seed       = args.seed
    cfg.data.seed        = args.seed
    cfg.data.max_words   = args.max_words
    cfg.train.device     = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.epochs != 30 or args.max_words != 4000 or args.seed != 0:
        print(
            "\n  *** NOTE: to reproduce figures/summary.json use "
            "--epochs 30 --max_words 4000 --seed 0 ***\n"
        )
    print(f"\n[train_checkpoint] epochs={args.epochs}  max_words={args.max_words}"
          f"  seed={args.seed}  device={cfg.train.device}")
    print(f"[train_checkpoint] checkpoint -> {args.ckpt}\n")

    model, vocab, lexicon, history = build_and_train(cfg, out_dir=args.out_dir)

    # Reconstruct the split to record provenance — identical seed → identical split
    train_entries, val_entries = lexicon.split(cfg.data.val_fraction, cfg.data.seed)

    # Git commit hash (best-effort; empty string if not in a git repo)
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = ""

    # Save: state dict + all config fields (serialised as dicts, not dataclasses,
    # so the checkpoint is readable even if config.py changes later).
    # Vocab is NOT serialised because build_vocab() is fully deterministic
    # (hardcoded ARPABET list in data/phonemes.py).  n_train/n_val + the
    # config seed are enough to reconstruct the exact split at eval time.
    ckpt = {
        "model_state_dict": model.state_dict(),
        "cfg_data":   dataclasses.asdict(cfg.data),
        "cfg_wm":     dataclasses.asdict(cfg.wm),
        "cfg_ltm":    dataclasses.asdict(cfg.ltm),
        "cfg_gating": dataclasses.asdict(cfg.gating),
        "cfg_loss":   dataclasses.asdict(cfg.loss),
        "cfg_train":  dataclasses.asdict(cfg.train),
        "history":    history,
        "lexicon_source": lexicon.source,
        "n_train":    len(train_entries),
        "n_val":      len(val_entries),
        "glove_present": os.path.exists(
            os.path.join(ROOT, "data", "glove.6B.300d.txt")),
        "git_commit": git_commit,
    }
    torch.save(ckpt, args.ckpt)
    print(f"\n[train_checkpoint] saved  -> {args.ckpt}")
    print(f"  lexicon_source : {lexicon.source}")
    print(f"  n_train / n_val: {ckpt['n_train']} / {ckpt['n_val']}")
    print(f"  glove_present  : {ckpt['glove_present']}")
    print(f"  git_commit     : {ckpt['git_commit'] or '(not available)'}")
    print(f"  final epoch    : {history[-1]}")


if __name__ == "__main__":
    main()
