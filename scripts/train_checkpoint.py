"""Train the lichtheim3 dual-route model and save a checkpoint.

Phase 2 version: adds configurable LTM architecture mode, unified hidden size,
gate threshold, ventral noise, periodic checkpoint saves, and correct
optimizer_state_dict in fresh mode.

Usage (from repo root):
    # Fresh run — unigru_last_hidden, H=128
    python scripts/train_checkpoint.py \\
        --lexicon_path data/lexicon_en_glove_covered.tsv \\
        --max_words 30000 --epochs 30 --seed 0 \\
        --ltm_encoder_mode unigru_last_hidden --hidden_size 128 \\
        --teacher_forcing_ratio 0.0 --interference_noise 0.0 \\
        --ckpt checkpoints/p4_H128_tf0p0_lr1e-3_s0.pt

    # Resume from an existing checkpoint (continue to a higher total epoch count)
    python scripts/train_checkpoint.py \\
        --lexicon_path data/lexicon_en_glove_covered.tsv \\
        --max_words 30000 --epochs 60 --seed 0 \\
        --resume_from checkpoints/p4_H128_tf0p0_lr1e-3_s0.pt \\
        --ckpt checkpoints/p4_H128_tf0p0_lr1e-3_s0_e60.pt

    When --resume_from is given, --epochs is the TOTAL epoch count.

Checkpoint format (new in Phase 2):
    model_state_dict        : model.state_dict()
    optimizer_state_dict    : optimizer state (now present in fresh AND resume mode)
    rng_states              : torch / numpy / cuda RNG states
    cfg_data/wm/ltm/gating/loss/train : all config dataclass dicts
    premotor_dim            : int  (architecture constant, default 128)
    history                 : list[dict]  training-loss history (all epochs)
    lexicon_source          : str
    n_train / n_val         : int
    glove_present           : bool
    git_commit              : str (best-effort)
    resumed_from            : str | None
    total_epochs_trained    : int
    lr_at_save              : float

Periodic checkpoint naming convention:
    --ckpt run.pt  +  --save_every_epochs 5
    → run.epoch_0005.pt, run.epoch_0010.pt, …, run.pt  (final)

Architecture backward compatibility:
    Old checkpoints (Phase 0 / baseline) use bigru_masked_mean.
    Load them by passing --ltm_encoder_mode bigru_masked_mean (default) and
    NOT setting --hidden_size (so enc_hidden=256 per direction, dec_hidden=256
    matches the saved weights).
    Attempting to load bigru_masked_mean weights into a unigru_last_hidden model
    raises a clear torch weight-shape mismatch.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import (default_config, Config, DataConfig, WMConfig, LTMConfig,
                    GatingConfig, LossConfig, TrainConfig)
from train import build_and_train, build_everything, run_epoch
from utils.seed import set_seed


# --------------------------------------------------------------------------- #
# Atomic checkpoint save helper
# --------------------------------------------------------------------------- #

def _save_checkpoint(ckpt: dict, path: str) -> None:
    """Save checkpoint atomically: write to .tmp then rename."""
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def _periodic_ckpt_path(final_path: str, epoch: int) -> str:
    """Derive periodic checkpoint path from the final path and epoch number."""
    base, ext = os.path.splitext(final_path)
    return f"{base}.epoch_{epoch:04d}{ext}"


def _as_cpu_byte_tensor(x, name="rng_state"):
    """Normalize a saved RNG state to a CPU torch.uint8 ByteTensor."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(torch.uint8)
    try:
        return torch.as_tensor(x, dtype=torch.uint8, device="cpu")
    except Exception as e:
        raise TypeError(
            f"Could not convert {name} to CPU torch.uint8 ByteTensor "
            f"(type={type(x)!r}, dtype={getattr(x, 'dtype', None)}, "
            f"device={getattr(x, 'device', None)})"
        ) from e


def _build_ckpt_dict(model, optim, cfg, lexicon, train_entries, val_entries,
                     history, git_commit, resumed_from,
                     premotor_dim: int = 128) -> dict:
    """Assemble the full checkpoint dict."""
    import numpy as _np
    rng_states = {
        "torch": torch.get_rng_state(),
        "numpy": _np.random.get_state(),
    }
    if torch.cuda.is_available():
        rng_states["cuda"] = torch.cuda.get_rng_state_all()

    return {
        "model_state_dict":      model.state_dict(),
        "optimizer_state_dict":  optim.state_dict() if optim is not None else None,
        "rng_states":            rng_states,
        "cfg_data":   dataclasses.asdict(cfg.data),
        "cfg_wm":     dataclasses.asdict(cfg.wm),
        "cfg_ltm":    dataclasses.asdict(cfg.ltm),
        "cfg_gating": dataclasses.asdict(cfg.gating),
        "cfg_loss":   dataclasses.asdict(cfg.loss),
        "cfg_train":  dataclasses.asdict(cfg.train),
        "premotor_dim":          premotor_dim,
        "history":               history,
        "lexicon_source":        lexicon.source,
        "n_train":               len(train_entries),
        "n_val":                 len(val_entries),
        "glove_present":         os.path.exists(
                                     os.path.join(ROOT, "data", "glove.6B.300d.txt")),
        "git_commit":            git_commit,
        "resumed_from":          resumed_from,
        "total_epochs_trained":  len(history),
        "lr_at_save":            cfg.train.lr,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Lichtheim3 dual-route model and save checkpoint."
    )
    # ---- data / run basics ----
    p.add_argument("--epochs",    type=int, default=10)
    p.add_argument("--max_words", type=int, default=4000)
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lexicon_path", type=str, default=None)
    p.add_argument("--resume_from", type=str, default=None,
                   help=(
                       "Path to checkpoint to resume from. "
                       "--epochs is then the TOTAL count. "
                       "Architecture (ltm_encoder_mode, hidden sizes) must match "
                       "the saved weights — torch will raise a clear error otherwise."
                   ))
    p.add_argument("--ckpt", type=str,
                   default=os.path.join(ROOT, "checkpoints", "lichtheim3.pt"))
    p.add_argument("--out_dir", type=str,
                   default=os.path.join(ROOT, "outputs", "train_run"))
    # ---- optimization ----
    p.add_argument("--lr", "--learn_rate", dest="lr", type=float, default=None)
    p.add_argument("--num_workers", type=int, default=None,
                   help="DataLoader num_workers (default: 0 from TrainConfig).")
    # ---- checkpointing ----
    p.add_argument("--save_every_epochs", type=int, default=None,
                   help=(
                       "Save a checkpoint every N epochs alongside the final one. "
                       "Files: <ckpt_stem>.epoch_NNNN<ext>. 0 = disabled (default)."
                   ))
    # ---- LTM architecture ----
    p.add_argument("--ltm_encoder_mode", type=str, default=None,
                   help=(
                       "LTM architecture: 'bigru_masked_mean' (historical default) "
                       "or 'unigru_last_hidden' (Yair 2026-07, Phase 4+). "
                       "Not compatible across modes for resume."
                   ))
    p.add_argument("--hidden_size", type=int, default=None,
                   help=(
                       "Unified recurrent hidden size H. Sets cfg.wm.hidden, "
                       "cfg.ltm.enc_hidden, cfg.ltm.dec_hidden to H. "
                       "Phase 4 grid: H ∈ {64, 128, 256}."
                   ))
    # ---- training regime ----
    p.add_argument("--teacher_forcing_ratio", type=float, default=1.0)
    p.add_argument("--interference_noise", type=float, default=None,
                   help="WM (dorsal) interference noise sigma. Default: WMConfig value.")
    p.add_argument("--ventral_noise", type=float, default=None,
                   help="LTM (ventral) noise sigma. Default: 0.0 (LTMConfig).")
    # ---- gate ----
    p.add_argument("--gate_alpha", type=float, default=None)
    p.add_argument("--gate_threshold", type=float, default=None,
                   help=(
                       "Gate routing threshold τ in g = sigmoid(α·(c_LTM − τ)). "
                       "Default: 0.5 (GatingConfig). Phase 7: search {0.3,0.5,0.7}."
                   ))
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

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

    # LTM architecture mode
    if args.ltm_encoder_mode is not None:
        cfg.ltm.ltm_encoder_mode = args.ltm_encoder_mode
        # Keep bidirectional_encoder in sync for metadata purposes
        cfg.ltm.bidirectional_encoder = (cfg.ltm.ltm_encoder_mode == "bigru_masked_mean")

    # Unified hidden size: sets WM and LTM recurrent dims together
    if args.hidden_size is not None:
        cfg.wm.hidden       = args.hidden_size
        cfg.ltm.enc_hidden  = args.hidden_size
        cfg.ltm.dec_hidden  = args.hidden_size

    # Optional overrides
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.interference_noise is not None:
        cfg.wm.interference_noise = args.interference_noise
    if args.ventral_noise is not None:
        cfg.ltm.ventral_noise = args.ventral_noise
    if args.gate_alpha is not None:
        cfg.gating.alpha = args.gate_alpha
    if args.gate_threshold is not None:
        cfg.gating.gate_threshold = args.gate_threshold
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    if args.save_every_epochs is not None:
        cfg.train.save_every_epochs = args.save_every_epochs

    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # Git commit hash (best-effort)
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = ""

    save_every = getattr(cfg.train, "save_every_epochs", 0)
    PREMOTOR_DIM = 128   # architecture constant (not in config)

    # ----------------------------------------------------------------------- #
    # RESUME MODE
    # ----------------------------------------------------------------------- #
    if args.resume_from is not None:
        if not os.path.exists(args.resume_from):
            print(f"\nERROR: --resume_from checkpoint not found: {args.resume_from}\n")
            sys.exit(1)

        print(f"\n[train_checkpoint] RESUME from {args.resume_from}")
        resume_ckpt = torch.load(args.resume_from, map_location=cfg.train.device,
                                  weights_only=False)
        prior_history = resume_ckpt.get("history", [])
        epochs_done   = len(prior_history)
        epochs_needed = args.epochs - epochs_done

        if epochs_needed <= 0:
            print(f"  Already at {epochs_done} epochs — nothing to do.")
            return

        print(f"  Prior epochs : {epochs_done}")
        print(f"  Target total : {args.epochs}")
        print(f"  Will train   : {epochs_needed} more epochs")
        print(f"  LTM mode     : {cfg.ltm.ltm_encoder_mode}")

        cfg.train.epochs = epochs_needed

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

        # Architecture check: this will raise a clear error if modes mismatch
        try:
            model.load_state_dict(resume_ckpt["model_state_dict"])
        except RuntimeError as e:
            print(f"\nERROR loading weights from {args.resume_from}:")
            print(f"  {e}")
            print(f"\n  This usually means the LTM architecture mode or hidden size")
            print(f"  does not match the checkpoint. Check --ltm_encoder_mode and")
            print(f"  --hidden_size match the values used when the checkpoint was saved.")
            print(f"  Saved cfg_ltm: {resume_ckpt.get('cfg_ltm', 'unknown')}")
            sys.exit(1)
        print(f"  Loaded weights from {args.resume_from}")

        from data.dataset import make_loader, build_pool_loader
        num_workers = getattr(cfg.train, "num_workers", 0)
        density = lexicon.neighborhood_density()
        train_loader = make_loader(train_entries, vocab, density,
                                   cfg.train.batch_size, frequency_weighted=True,
                                   freq_temp=cfg.data.freq_temp, shuffle=True,
                                   num_workers=num_workers)
        val_loader = make_loader(val_entries, vocab, density,
                                 cfg.train.batch_size, frequency_weighted=False,
                                 shuffle=False, num_workers=num_workers)
        pool_loader = None
        if cfg.train.dorsal_pool_size > 0:
            pool_loader = build_pool_loader(
                vocab, cfg.train.dorsal_pool_size, cfg.train.batch_size,
                cfg.data.semantic_dim, max_len=cfg.data.max_phonemes,
                seed=cfg.train.seed, num_workers=num_workers)
        pool_iter = itertools.cycle(pool_loader) if pool_loader is not None else None

        effective_lr = cfg.train.lr
        optim = torch.optim.AdamW(model.parameters(), lr=effective_lr,
                                   weight_decay=cfg.train.weight_decay)
        if resume_ckpt.get("optimizer_state_dict") is not None:
            optim.load_state_dict(resume_ckpt["optimizer_state_dict"])
            for state in optim.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(cfg.train.device)
            if args.lr is not None:
                for g in optim.param_groups:
                    g["lr"] = args.lr
            print(f"  Optimizer : restored from checkpoint  (lr={effective_lr})")
        else:
            print(f"  Optimizer : warm-restart (no saved state)  (lr={effective_lr})")

        if "rng_states" in resume_ckpt:
            rs = resume_ckpt["rng_states"]
            if "torch" in rs:
                torch.set_rng_state(_as_cpu_byte_tensor(rs["torch"], "torch_rng"))
                print("  [resume] restored torch RNG state")
            if "cuda" in rs and torch.cuda.is_available():
                cuda_states = [_as_cpu_byte_tensor(s, "cuda_rng") for s in rs["cuda"]]
                torch.cuda.set_rng_state_all(cuda_states)
                print("  [resume] restored CUDA RNG state")
            else:
                print("  [resume] no CUDA RNG state to restore")
            if "numpy" in rs:
                import numpy as np
                np.random.set_state(rs["numpy"])
                print("  [resume] restored numpy RNG state")
            print("  RNG states: restored from checkpoint")
        else:
            print("  RNG states: not saved in checkpoint (non-reproducible resume)")

        print(f"\n[train_checkpoint] {cfg.ltm.ltm_encoder_mode}"
              f"  lexicon={len(lexicon)} ({lexicon.source})"
              f"  vocab={vocab.size}  device={cfg.train.device}")

        new_history = []
        for ep in range(epochs_needed):
            tr = run_epoch(model, train_loader, cfg, optim, pool_iter=pool_iter)
            with torch.no_grad():
                va = run_epoch(model, val_loader, cfg, optim=None)
            total_ep = epochs_done + ep + 1
            row = {
                "epoch": total_ep,
                "train_total": tr["total"], "train_rep": tr["rep"],
                "train_wm": tr["wm"], "val_rep": va["rep"], "val_wm": va["wm"],
            }
            new_history.append(row)
            print(f"[ep {total_ep:3d}/{args.epochs}] "
                  f"train total={tr['total']:.3f} rep={tr['rep']:.3f} "
                  f"align={tr['align']:.3f} wm={tr['wm']:.3f} | "
                  f"val rep={va['rep']:.3f}")

            # Periodic checkpoint
            if save_every > 0 and total_ep % save_every == 0:
                partial_history = prior_history + new_history
                periodic_ckpt = _build_ckpt_dict(
                    model, optim, cfg, lexicon, train_entries, val_entries,
                    partial_history, git_commit, args.resume_from, PREMOTOR_DIM)
                p_path = _periodic_ckpt_path(args.ckpt, total_ep)
                _save_checkpoint(periodic_ckpt, p_path)
                print(f"  [periodic ckpt] -> {p_path}")

        history      = prior_history + new_history
        resumed_from = args.resume_from
        from train import plot_loss_history
        plot_loss_history(history, os.path.join(args.out_dir, "training_loss.png"))

    # ----------------------------------------------------------------------- #
    # FRESH MODE
    # ----------------------------------------------------------------------- #
    else:
        print(f"\n[train_checkpoint] FRESH RUN")
        print(f"  epochs={args.epochs}  max_words={args.max_words}  seed={args.seed}")
        print(f"  device={cfg.train.device}")
        print(f"  ltm_encoder_mode={cfg.ltm.ltm_encoder_mode}")
        print(f"  wm_hidden={cfg.wm.hidden}  ltm_enc_hidden={cfg.ltm.enc_hidden}"
              f"  ltm_dec_hidden={cfg.ltm.dec_hidden}")
        print(f"  teacher_forcing_ratio={cfg.train.teacher_forcing_ratio}"
              f"  interference_noise={cfg.wm.interference_noise}"
              f"  ventral_noise={cfg.ltm.ventral_noise}"
              f"  gate_alpha={cfg.gating.alpha}"
              f"  gate_threshold={cfg.gating.gate_threshold}")
        print(f"  num_workers={cfg.train.num_workers}"
              f"  save_every_epochs={getattr(cfg.train, 'save_every_epochs', 0)}")
        print(f"  checkpoint -> {args.ckpt}\n")

        # Use build_and_train (now returns optimizer as 5th value)
        if save_every > 0:
            # Need direct access to the training loop for periodic saves;
            # replicate build_and_train logic here to intercept each epoch.
            import itertools as _it
            from train import plot_loss_history
            if cfg.train.device == "cpu" and torch.cuda.is_available():
                cfg.train.device = "cuda"
            (model, vocab, lexicon,
             train_loader, val_loader, pool_loader) = build_everything(cfg)
            optim = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                       weight_decay=cfg.train.weight_decay)
            pool_iter = _it.cycle(pool_loader) if pool_loader is not None else None
            train_entries, val_entries = lexicon.split(cfg.data.val_fraction,
                                                        cfg.data.seed)
            history = []
            for ep in range(cfg.train.epochs):
                tr = run_epoch(model, train_loader, cfg, optim, pool_iter=pool_iter)
                with torch.no_grad():
                    va = run_epoch(model, val_loader, cfg, optim=None)
                row = {"epoch": ep + 1,
                       "train_total": tr["total"], "train_rep": tr["rep"],
                       "train_wm": tr["wm"], "val_rep": va["rep"], "val_wm": va["wm"]}
                history.append(row)
                print(f"[ep {ep+1:2d}/{cfg.train.epochs}] "
                      f"train total={tr['total']:.3f} rep={tr['rep']:.3f} "
                      f"align={tr['align']:.3f} wm={tr['wm']:.3f} | "
                      f"val rep={va['rep']:.3f} wm={va['wm']:.3f}")
                if save_every > 0 and (ep + 1) % save_every == 0:
                    periodic_ckpt = _build_ckpt_dict(
                        model, optim, cfg, lexicon, train_entries, val_entries,
                        history, git_commit, None, PREMOTOR_DIM)
                    p_path = _periodic_ckpt_path(args.ckpt, ep + 1)
                    _save_checkpoint(periodic_ckpt, p_path)
                    print(f"  [periodic ckpt] -> {p_path}")
            plot_loss_history(history, os.path.join(args.out_dir, "training_loss.png"))
        else:
            model, vocab, lexicon, history, optim = build_and_train(
                cfg, out_dir=args.out_dir)
            train_entries, val_entries = lexicon.split(cfg.data.val_fraction,
                                                        cfg.data.seed)
        resumed_from = None

    # ----------------------------------------------------------------------- #
    # Save final checkpoint (common to both modes)
    # ----------------------------------------------------------------------- #
    ckpt = _build_ckpt_dict(model, optim, cfg, lexicon, train_entries, val_entries,
                             history, git_commit, resumed_from, PREMOTOR_DIM)
    _save_checkpoint(ckpt, args.ckpt)

    print(f"\n[train_checkpoint] saved  -> {args.ckpt}")
    print(f"  ltm_encoder_mode     : {cfg.ltm.ltm_encoder_mode}")
    print(f"  wm_hidden            : {cfg.wm.hidden}")
    print(f"  ltm_enc/dec_hidden   : {cfg.ltm.enc_hidden} / {cfg.ltm.dec_hidden}")
    print(f"  gate_threshold       : {cfg.gating.gate_threshold}")
    print(f"  lexicon_source       : {lexicon.source}")
    print(f"  n_train / n_val      : {ckpt['n_train']} / {ckpt['n_val']}")
    print(f"  glove_present        : {ckpt['glove_present']}")
    print(f"  total_epochs_trained : {ckpt['total_epochs_trained']}")
    print(f"  optimizer_saved      : {ckpt['optimizer_state_dict'] is not None}")
    print(f"  resumed_from         : {ckpt['resumed_from'] or 'N/A (fresh run)'}")
    print(f"  git_commit           : {ckpt['git_commit'] or '(not available)'}")
    print(f"  final epoch          : {history[-1]}")


if __name__ == "__main__":
    main()
