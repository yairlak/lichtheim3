"""Train the lichtheim3 dual-route model and save a checkpoint.

This script wraps the existing train pipeline and adds a torch.save() call
so that weights are available for downstream evaluations.
It does NOT touch architecture / losses / gate.

Usage (from repo root):
    # Fresh training run
    python scripts/train_checkpoint.py --max_words 30000 --epochs 30 --seed 0 \\
        --lexicon_path data/lexicon_en_glove_covered.tsv \\
        --ckpt checkpoints/lichtheim3_30k_glove.pt

    # Resume from an existing checkpoint (continue to a higher total epoch count)
    python scripts/train_checkpoint.py \\
        --lexicon_path data/lexicon_en_glove_covered.tsv \\
        --max_words 30000 --epochs 60 --seed 0 \\
        --resume_from checkpoints/lichtheim3_30k_glove.pt \\
        --ckpt checkpoints/lichtheim3_30k_glove_e60.pt

    When --resume_from is given, --epochs is the TOTAL epoch count.
    The script calculates how many additional epochs to run and continues
    training from the resumed weights.  The optimizer is re-initialised
    (no saved optimizer state), which is equivalent to a warm-restart.

The checkpoint is a plain dict with keys:
    model_state_dict   : model.state_dict()
    cfg_data           : dataclass-dict of DataConfig
    cfg_wm             : dataclass-dict of WMConfig
    cfg_ltm            : dataclass-dict of LTMConfig
    cfg_gating         : dataclass-dict of GatingConfig
    cfg_loss           : dataclass-dict of LossConfig
    cfg_train          : dataclass-dict of TrainConfig
    history            : list[dict]  training-loss history (all epochs, incl. resumed)
    lexicon_source     : str
    n_train            : int         number of training entries
    resumed_from       : str | None  path of the checkpoint this was resumed from
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

from config import default_config, Config, DataConfig, WMConfig, LTMConfig, GatingConfig, LossConfig, TrainConfig
from train import build_and_train, build_everything, run_epoch
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
    p.add_argument("--lexicon_path", type=str, default=None,
                   help=(
                       "override lexicon TSV path (default: data/lexicon_en.tsv).  "
                       "Use data/lexicon_en_glove_covered.tsv for GloVe-only training."
                   ))
    p.add_argument("--resume_from", type=str, default=None,
                   help=(
                       "path to an existing checkpoint to resume from.  "
                       "--epochs is then the TOTAL epoch count.  "
                       "Optimizer state and RNG states are restored if present in "
                       "the checkpoint (proper resume).  If absent (old checkpoint), "
                       "the optimizer is re-initialised (warm-restart)."
                   ))
    p.add_argument("--lr", "--learn_rate", dest="lr", type=float, default=None,
                   help=(
                       "learning rate override.  For fresh runs, overrides "
                       f"config default ({1e-3}).  For --resume_from, overrides "
                       "the saved optimizer LR — useful for low-LR fine-tuning.  "
                       "Recommended for continuation after plateau: current_lr / 10."
                   ))
    p.add_argument("--teacher_forcing_ratio", type=float, default=1.0,
                   help=(
                       "scheduled sampling ratio: 1.0 = full teacher forcing "
                       "(historical default); 0.0 = fully autoregressive training; "
                       "intermediate values mix gold and predicted tokens step-by-step.  "
                       "Validation always uses TF=1.0 regardless of this flag."
                   ))
    p.add_argument("--interference_noise", type=float, default=None,
                   help=(
                       "WM interference noise sigma during training "
                       f"(default: {0.10} from WMConfig).  "
                       "Set 0.0 to disable training-time WM noise."
                   ))
    p.add_argument("--gate_alpha", type=float, default=None,
                   help=(
                       "gate sharpness: g = sigmoid(alpha*(conf-0.5)) "
                       f"(default: {4.0} from GatingConfig)."
                   ))
    p.add_argument("--ckpt",      type=str,
                   default=os.path.join(ROOT, "checkpoints", "lichtheim3.pt"),
                   help="path to write the checkpoint file")
    p.add_argument("--out_dir",   type=str,
                   default=os.path.join(ROOT, "outputs", "train_run"),
                   help="directory for training figures (loss curve)")
    return p.parse_args()


def main() -> None:
    import itertools
    args = parse_args()

    cfg = default_config()
    cfg.train.epochs     = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.seed       = args.seed
    cfg.data.seed        = args.seed
    cfg.data.max_words   = args.max_words
    cfg.data.lexicon_path = args.lexicon_path
    cfg.train.device     = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.train.teacher_forcing_ratio = args.teacher_forcing_ratio
    # Allow CLI LR override (used for fresh runs and low-LR continuation)
    if args.lr is not None:
        cfg.train.lr = args.lr
    # Optional config overrides: interference_noise and gate_alpha
    if args.interference_noise is not None:
        cfg.wm.interference_noise = args.interference_noise
    if args.gate_alpha is not None:
        cfg.gating.alpha = args.gate_alpha

    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # ----------------------------------------------------------------------- #
    # Git commit hash (best-effort)
    # ----------------------------------------------------------------------- #
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = ""

    # declared here so the save block can reference it whether or not resume mode ran
    optim: "torch.optim.Optimizer | None" = None

    # ----------------------------------------------------------------------- #
    # RESUME MODE: load weights from an existing checkpoint, continue training
    # ----------------------------------------------------------------------- #
    if args.resume_from is not None:
        if not os.path.exists(args.resume_from):
            print(f"\nERROR: --resume_from checkpoint not found: {args.resume_from}\n")
            import sys; sys.exit(1)

        print(f"\n[train_checkpoint] RESUME from {args.resume_from}")
        resume_ckpt = torch.load(args.resume_from, map_location=cfg.train.device,
                                  weights_only=False)
        prior_history = resume_ckpt.get("history", [])
        epochs_done   = len(prior_history)
        epochs_needed = args.epochs - epochs_done

        if epochs_needed <= 0:
            print(f"  Already at {epochs_done} epochs — nothing to do "
                  f"(--epochs {args.epochs} <= epochs already trained).")
            print("  Increase --epochs beyond the current count to continue training.")
            return

        print(f"  Prior epochs : {epochs_done}")
        print(f"  Target total : {args.epochs}")
        print(f"  Will train   : {epochs_needed} more epochs")

        # Override cfg.train.epochs to run only the remaining steps
        cfg.train.epochs = epochs_needed

        # Build model + data (fresh, from CLI flags)
        from models.dual_route import DualRouteModel
        from data.lexicon import build_lexicon
        from data.phonemes import build_vocab
        vocab   = build_vocab()
        lexicon = build_lexicon(cfg.data, vocab)
        train_entries, val_entries = lexicon.split(cfg.data.val_fraction, cfg.data.seed)
        bank = torch.stack([torch.tensor(e.semantic) for e in train_entries]).float()
        bank = bank.to(cfg.train.device)
        model = DualRouteModel(cfg, vocab).to(cfg.train.device)
        model.set_semantic_bank(bank)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        print(f"  Loaded weights from {args.resume_from}")

        # Rebuild loaders
        from data.dataset import make_loader, build_pool_loader
        density = lexicon.neighborhood_density()
        train_loader = make_loader(train_entries, vocab, density,
                                   cfg.train.batch_size,
                                   frequency_weighted=True,
                                   freq_temp=cfg.data.freq_temp,
                                   shuffle=True)
        val_loader = make_loader(val_entries, vocab, density,
                                 cfg.train.batch_size,
                                 frequency_weighted=False, shuffle=False)
        pool_loader = None
        if cfg.train.dorsal_pool_size > 0:
            pool_loader = build_pool_loader(
                vocab, cfg.train.dorsal_pool_size, cfg.train.batch_size,
                cfg.data.semantic_dim, max_len=cfg.data.max_phonemes,
                seed=cfg.train.seed)
        pool_iter = itertools.cycle(pool_loader) if pool_loader is not None else None

        # Optimizer: restore saved state if available (proper resume),
        # otherwise warm-restart.  A CLI --lr overrides the saved/default LR.
        effective_lr = cfg.train.lr   # already updated by args.lr if provided
        optim = torch.optim.AdamW(model.parameters(), lr=effective_lr,
                                   weight_decay=cfg.train.weight_decay)
        if "optimizer_state_dict" in resume_ckpt:
            optim.load_state_dict(resume_ckpt["optimizer_state_dict"])
            # Move optimizer tensors to target device
            for state in optim.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(cfg.train.device)
            # Apply LR override AFTER restoring state so Adam moments are kept
            if args.lr is not None:
                for g in optim.param_groups:
                    g["lr"] = args.lr
            print(f"  Optimizer : restored from checkpoint  (lr={effective_lr})")
        else:
            print(f"  Optimizer : warm-restart (no saved state)  (lr={effective_lr})")

        # Restore RNG states for reproducibility (best-effort)
        if "rng_states" in resume_ckpt:
            rs = resume_ckpt["rng_states"]
            if "torch" in rs:
                torch.set_rng_state(rs["torch"])
            if "cuda" in rs and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rs["cuda"])
            if "numpy" in rs:
                import numpy as np
                np.random.set_state(rs["numpy"])
            print("  RNG states: restored from checkpoint")
        else:
            print("  RNG states: not saved in checkpoint (non-reproducible resume)")

        print(f"\n[train_checkpoint] lexicon={len(lexicon)} ({lexicon.source}) "
              f"vocab={vocab.size} device={cfg.train.device}")
        new_history = []
        for ep in range(epochs_needed):
            tr = run_epoch(model, train_loader, cfg, optim, pool_iter=pool_iter)
            with torch.no_grad():
                va = run_epoch(model, val_loader, cfg, optim=None)
            total_ep = epochs_done + ep + 1
            new_history.append({
                "epoch": total_ep,
                "train_total": tr["total"], "train_rep": tr["rep"],
                "train_wm": tr["wm"], "val_rep": va["rep"], "val_wm": va["wm"],
            })
            print(f"[ep {total_ep:3d}/{args.epochs}] "
                  f"train total={tr['total']:.3f} rep={tr['rep']:.3f} "
                  f"align={tr['align']:.3f} wm={tr['wm']:.3f} | "
                  f"val rep={va['rep']:.3f}")

        history      = prior_history + new_history
        resumed_from = args.resume_from
        # Save loss-curve plot for the full combined history
        from train import plot_loss_history
        plot_loss_history(history, os.path.join(args.out_dir, "training_loss.png"))

    # ----------------------------------------------------------------------- #
    # FRESH MODE: train from scratch using build_and_train
    # ----------------------------------------------------------------------- #
    else:
        if args.epochs != 30 or args.max_words != 4000 or args.seed != 0:
            print(
                "\n  *** NOTE: to reproduce figures/summary.json use "
                "--epochs 30 --max_words 4000 --seed 0 ***\n"
            )
        print(f"\n[train_checkpoint] epochs={args.epochs}  max_words={args.max_words}"
              f"  seed={args.seed}  device={cfg.train.device}")
        print(f"  teacher_forcing_ratio={cfg.train.teacher_forcing_ratio}"
              f"  interference_noise={cfg.wm.interference_noise}"
              f"  gate_alpha={cfg.gating.alpha}")
        print(f"[train_checkpoint] checkpoint -> {args.ckpt}\n")

        model, vocab, lexicon, history = build_and_train(cfg, out_dir=args.out_dir)
        train_entries, val_entries = lexicon.split(cfg.data.val_fraction, cfg.data.seed)
        resumed_from = None

    # ----------------------------------------------------------------------- #
    # Save checkpoint (common to both modes)
    # ----------------------------------------------------------------------- #
    import numpy as _np
    rng_states = {
        "torch": torch.get_rng_state(),
        "numpy": _np.random.get_state(),
    }
    if torch.cuda.is_available():
        rng_states["cuda"] = torch.cuda.get_rng_state_all()

    ckpt = {
        "model_state_dict": model.state_dict(),
        # optimizer_state_dict only available in resume mode; None in fresh mode
        # since build_and_train does not expose its internal optimizer object.
        "optimizer_state_dict": optim.state_dict() if optim is not None else None,
        "rng_states":      rng_states,
        "cfg_data":    dataclasses.asdict(cfg.data),
        "cfg_wm":      dataclasses.asdict(cfg.wm),
        "cfg_ltm":     dataclasses.asdict(cfg.ltm),
        "cfg_gating":  dataclasses.asdict(cfg.gating),
        "cfg_loss":    dataclasses.asdict(cfg.loss),
        "cfg_train":   dataclasses.asdict(cfg.train),
        "history":     history,
        "lexicon_source": lexicon.source,
        "n_train":     len(train_entries),
        "n_val":       len(val_entries),
        "glove_present": os.path.exists(
            os.path.join(ROOT, "data", "glove.6B.300d.txt")),
        "git_commit":  git_commit,
        "resumed_from": resumed_from,
        "total_epochs_trained": len(history),
        "lr_at_save":  cfg.train.lr,
    }
    torch.save(ckpt, args.ckpt)
    print(f"\n[train_checkpoint] saved  -> {args.ckpt}")
    print(f"  lexicon_source       : {lexicon.source}")
    print(f"  n_train / n_val      : {ckpt['n_train']} / {ckpt['n_val']}")
    print(f"  glove_present        : {ckpt['glove_present']}")
    print(f"  total_epochs_trained : {ckpt['total_epochs_trained']}")
    print(f"  resumed_from         : {ckpt['resumed_from'] or 'N/A (fresh run)'}")
    print(f"  git_commit           : {ckpt['git_commit'] or '(not available)'}")
    print(f"  final epoch          : {history[-1]}")


if __name__ == "__main__":
    main()
