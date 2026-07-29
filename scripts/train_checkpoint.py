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
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import (default_config, get_effective_split_seed, validate_split_config,
                    Config, DataConfig, WMConfig, LTMConfig,
                    GatingConfig, LossConfig, TrainConfig)
from train import build_and_train, build_everything, run_epoch
from utils.seed import set_seed
from utils.provenance import (sha256_words_ordered, sha256_words_sorted,
                               sha256_file, git_state,
                               PROVENANCE_SCHEMA_VERSION, V1_REQUIRED_PROVENANCE_FIELDS)


# PROVENANCE_SCHEMA_VERSION and V1_REQUIRED_PROVENANCE_FIELDS are imported
# from utils.provenance (the canonical location for these shared constants).


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
                     premotor_dim: int = 128,
                     target_total_epochs: int = None,
                     resume_epochs_added: int = None) -> dict:
    """Assemble the full checkpoint dict."""
    import numpy as _np
    rng_states = {
        "torch": torch.get_rng_state(),
        "numpy": _np.random.get_state(),
    }
    if torch.cuda.is_available():
        rng_states["cuda"] = torch.cuda.get_rng_state_all()

    split_mode = getattr(cfg.data, "split_mode", "standard")
    full_lexicon = (split_mode == "full_lexicon")

    # Split seed semantics
    split_seed_effective = (None if full_lexicon
                            else get_effective_split_seed(cfg.data))

    # Word-list hashes using the canonical convention from utils.provenance
    train_words = [e.word for e in train_entries]
    val_words   = [e.word for e in val_entries]
    ordered_training_sha   = sha256_words_ordered(train_words)
    sorted_training_sha    = sha256_words_sorted(train_words)
    # Backward-compat alias: old field used alphabetical sort
    train_split_sha256     = sorted_training_sha
    val_split_sha256       = sha256_words_sorted(val_words) if val_words else sha256_words_sorted([])

    # Load stats from the lexicon object (set by build_bundled; None for synthetic)
    ls = getattr(lexicon, "load_stats", None)

    # Git state (best-effort via utils.provenance)
    gs = git_state(cwd=ROOT)
    # git_commit may already be resolved by the caller; prefer it over gs["commit"]
    resolved_commit = git_commit if git_commit else gs["commit"]

    return {
        # --- model / optimizer / RNG ---
        "model_state_dict":      model.state_dict(),
        "optimizer_state_dict":  optim.state_dict() if optim is not None else None,
        "rng_states":            rng_states,

        # --- full config ---
        "cfg_data":   dataclasses.asdict(cfg.data),
        "cfg_wm":     dataclasses.asdict(cfg.wm),
        "cfg_ltm":    dataclasses.asdict(cfg.ltm),
        "cfg_gating": dataclasses.asdict(cfg.gating),
        "cfg_loss":   dataclasses.asdict(cfg.loss),
        "cfg_train":  dataclasses.asdict(cfg.train),
        "premotor_dim":          premotor_dim,

        # --- training history ---
        "history":               history,

        # --- split regime ---
        "split_mode":            split_mode,
        "train_all_words":       full_lexicon,
        "validation_enabled":    not full_lexicon,
        "val_fraction":          cfg.data.val_fraction,

        # --- lexicon source ---
        "lexicon_source":        lexicon.source,

        # --- dataset counters ---
        "n_source_rows":             ls.n_source_rows             if ls else None,
        "n_entries_after_loading":   ls.n_entries_after_loading   if ls else None,
        "n_filtered_unknown_phoneme":ls.n_filtered_unknown_phoneme if ls else None,
        "n_filtered_length":         ls.n_filtered_length         if ls else None,
        "n_unique_loaded_words":     ls.n_unique_loaded_words     if ls else None,
        "n_glove_found":             ls.n_glove_found             if ls else None,
        "n_glove_fallback":          ls.n_glove_fallback          if ls else None,
        "n_train":               len(train_entries),
        "n_val":                 len(val_entries),
        "n_unique_train_words":  len(set(train_words)),

        # --- hashes ---
        "lexicon_file_sha256":           ls.lexicon_file_sha256   if ls else None,
        "ordered_training_words_sha256": ordered_training_sha,
        "sorted_training_words_sha256":  sorted_training_sha,
        "val_split_sha256":              val_split_sha256,
        # Backward-compat alias (old field name used alphabetical sort)
        "train_split_sha256":            train_split_sha256,

        # --- split seed ---
        "split_seed_used":       not full_lexicon,
        "split_seed_effective":  split_seed_effective,
        "split_seed_configured": cfg.data.split_seed,

        # --- file paths ---
        "lexicon_file_path":  ls.lexicon_file_path  if ls else None,
        "glove_file_path":    ls.glove_file_path     if ls else None,
        # glove_present: True iff GloVe was actually loaded (path recorded in LoadStats).
        # Falls back to filesystem probe for legacy callers without LoadStats.
        "glove_present":      (bool(ls.glove_file_path) if ls is not None
                               else os.path.exists(
                                   os.path.join(ROOT, "data", "glove.6B.300d.txt"))),

        # --- git ---
        "git_commit": resolved_commit,
        "git_branch": gs["branch"],
        "git_dirty":  gs["dirty"],

        # --- sampler ---
        "sampler_config": {
            "type":               "WeightedRandomSampler",
            "replacement":        True,
            "num_samples":        len(train_entries),
            "frequency_weighted": True,
            "freq_temp":          cfg.data.freq_temp,
            "n_train":            len(train_entries),
        },

        # --- provenance schema ---
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,

        # --- misc ---
        "resumed_from":          resumed_from,
        "total_epochs_trained":  len(history),
        # target_total_epochs: the --epochs value passed to train_checkpoint.py for this run.
        # resume_epochs_added: how many new epochs were trained in this continuation (resume only).
        # total_epochs_trained is always authoritative for "how many epochs in the checkpoint".
        "target_total_epochs":   target_total_epochs,
        "resume_epochs_added":   resume_epochs_added,
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
    p.add_argument("--epochs",    type=int, default=10,
                   help="Total epoch count (fresh) or target total (resume).")
    # Dataset / model args: default=None so resume can detect 'not provided'
    p.add_argument("--max_words", type=int, default=None,
                   help="Max words to load from the lexicon TSV (default: DataConfig).")
    p.add_argument("--seed",      type=int, default=None,
                   help="Training seed. At resume, RNG states from checkpoint take "
                        "precedence and this value is informational only.")
    p.add_argument("--split_seed", type=int, default=None,
                   help=(
                       "Seed for the train/val split ONLY. Omitted -> falls back "
                       "to --seed (legacy behaviour). Pass --split_seed 0 to hold "
                       "the partition fixed while varying --seed."
                   ))
    p.add_argument("--batch_size", type=int, default=None,
                   help="Mini-batch size (default: TrainConfig).")
    p.add_argument("--lexicon_path", type=str, default=None)
    p.add_argument("--train_all_words", action="store_true", default=False,
                   help=(
                       "Use all loaded words for training; disable validation split. "
                       "Sets split_mode='full_lexicon' and val_fraction=0.0. "
                       "At resume, split mode is always inherited from the checkpoint; "
                       "this flag is not required and must not contradict the checkpoint."
                   ))
    p.add_argument("--resume_from", type=str, default=None,
                   help=(
                       "Path to checkpoint to resume from. --epochs is the TOTAL count. "
                       "Dataset and model configuration are inherited from the checkpoint. "
                       "Architecture (ltm_encoder_mode, hidden sizes) must match."
                   ))
    p.add_argument("--ckpt", type=str,
                   default=os.path.join(ROOT, "checkpoints", "lichtheim3.pt"))
    p.add_argument("--out_dir", type=str,
                   default=os.path.join(ROOT, "outputs", "train_run"))
    # ---- optimization (operational — can change at resume) ----
    p.add_argument("--lr", "--learn_rate", dest="lr", type=float, default=None)
    p.add_argument("--num_workers", type=int, default=None,
                   help="DataLoader num_workers (default: 0 from TrainConfig).")
    # ---- checkpointing ----
    p.add_argument("--save_every_epochs", type=int, default=None,
                   help=(
                       "Save a checkpoint every N epochs alongside the final one. "
                       "Files: <ckpt_stem>.epoch_NNNN<ext>. 0 = disabled (default)."
                   ))
    # ---- LTM architecture (default=None → inherited from checkpoint at resume) ----
    p.add_argument("--ltm_encoder_mode", type=str, default=None,
                   help=(
                       "LTM architecture: 'bigru_masked_mean' (historical default) "
                       "or 'unigru_last_hidden' (Phase 4+). "
                       "Not compatible across modes for resume."
                   ))
    p.add_argument("--hidden_size", type=int, default=None,
                   help=(
                       "Unified recurrent hidden size H. Sets cfg.wm.hidden, "
                       "cfg.ltm.enc_hidden, cfg.ltm.dec_hidden to H. "
                       "Phase 4 grid: H ∈ {64, 128, 256}."
                   ))
    # ---- training regime (default=None → inherited from checkpoint at resume) ----
    p.add_argument("--teacher_forcing_ratio", type=float, default=None,
                   help="Teacher forcing ratio (default: 1.0 for fresh runs).")
    p.add_argument("--interference_noise", type=float, default=None,
                   help="WM (dorsal) interference noise sigma. Default: WMConfig value.")
    p.add_argument("--ventral_noise", type=float, default=None,
                   help="LTM (ventral) noise sigma. Default: 0.0 (LTMConfig).")
    # ---- gate (default=None → inherited from checkpoint at resume) ----
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

def _apply_fresh_cli(cfg: "Config", args: "argparse.Namespace") -> None:
    """Apply fresh-mode CLI arguments to cfg.

    For args that default to None (to support resume mode's "not provided" detection),
    fresh mode applies backward-compatible CLI defaults when the user omits them.
    These match the old hardcoded defaults so existing scripts are unaffected.
    """
    # --- epochs: always taken from CLI (required arg with default=10) ---
    cfg.train.epochs = args.epochs

    # --- seed: fresh-mode default = 0 (old CLI default) ---
    seed = args.seed if args.seed is not None else 0
    cfg.train.seed = seed
    cfg.data.seed  = seed
    if args.split_seed is not None:
        cfg.data.split_seed = args.split_seed

    # --- max_words: fresh-mode default = 4000 (old CLI default) ---
    cfg.data.max_words = args.max_words if args.max_words is not None else 4000

    # --- batch_size: fresh-mode default = 64 ---
    cfg.train.batch_size = args.batch_size if args.batch_size is not None else 64

    # --- teacher_forcing_ratio: fresh-mode default = 1.0 ---
    cfg.train.teacher_forcing_ratio = (args.teacher_forcing_ratio
                                       if args.teacher_forcing_ratio is not None else 1.0)

    # --- lexicon path: None = use bundled default ---
    cfg.data.lexicon_path = args.lexicon_path

    if args.ltm_encoder_mode is not None:
        cfg.ltm.ltm_encoder_mode = args.ltm_encoder_mode
        cfg.ltm.bidirectional_encoder = (cfg.ltm.ltm_encoder_mode == "bigru_masked_mean")
    if args.hidden_size is not None:
        cfg.wm.hidden      = args.hidden_size
        cfg.ltm.enc_hidden = args.hidden_size
        cfg.ltm.dec_hidden = args.hidden_size
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

    # --- Full-lexicon mode (applied last so validate_split_config sees final state) ---
    if args.train_all_words:
        cfg.data.split_mode   = "full_lexicon"
        cfg.data.val_fraction = 0.0

    validate_split_config(cfg.data)


def _resume_fail_fast(args: "argparse.Namespace", cfg: "Config") -> None:
    """Fail-fast if any CLI arg contradicts the checkpoint config.

    Only checks args that were explicitly provided (not None).
    """
    checks = [
        ("--lexicon_path",         args.lexicon_path,         cfg.data.lexicon_path),
        ("--max_words",            args.max_words,            cfg.data.max_words),
        ("--split_seed",           args.split_seed,           cfg.data.split_seed),
        ("--batch_size",           args.batch_size,           cfg.train.batch_size),
        ("--teacher_forcing_ratio",args.teacher_forcing_ratio,cfg.train.teacher_forcing_ratio),
        ("--interference_noise",   args.interference_noise,   cfg.wm.interference_noise),
        ("--ventral_noise",        args.ventral_noise,        cfg.ltm.ventral_noise),
        ("--gate_alpha",           args.gate_alpha,           cfg.gating.alpha),
        ("--gate_threshold",       args.gate_threshold,       cfg.gating.gate_threshold),
    ]
    errors = []
    for name, cli_val, ckpt_val in checks:
        if cli_val is not None and cli_val != ckpt_val:
            errors.append(
                f"  {name}: CLI={cli_val!r} ≠ checkpoint={ckpt_val!r}"
            )
    # Hidden size: CLI sets all three dims; check first dim only if provided
    if args.hidden_size is not None and args.hidden_size != cfg.wm.hidden:
        errors.append(
            f"  --hidden_size: CLI={args.hidden_size} ≠ checkpoint.wm.hidden={cfg.wm.hidden}"
        )
    if args.ltm_encoder_mode is not None and args.ltm_encoder_mode != cfg.ltm.ltm_encoder_mode:
        errors.append(
            f"  --ltm_encoder_mode: CLI={args.ltm_encoder_mode!r} ≠ "
            f"checkpoint={cfg.ltm.ltm_encoder_mode!r}"
        )
    # --train_all_words: if explicitly set True and checkpoint is standard → error
    ckpt_split_mode = getattr(cfg.data, "split_mode", "standard")
    if args.train_all_words and ckpt_split_mode != "full_lexicon":
        errors.append(
            f"  --train_all_words: checkpoint split_mode={ckpt_split_mode!r}, "
            f"not 'full_lexicon'. Cannot switch mode at resume; start a fresh run."
        )
    if errors:
        print("\nERROR: CLI arguments contradict the checkpoint configuration:")
        for e in errors:
            print(e)
        print("\nTo resume safely, omit contradicting arguments "
              "(they are inherited from the checkpoint).")
        sys.exit(1)


def _verify_dataset_provenance(resume_ckpt: dict,
                               cfg,
                               lexicon,
                               train_entries: list,
                               val_entries: list) -> None:
    """Strict provenance check: compare all dataset invariants between a resumed
    checkpoint and the freshly reconstructed lexicon + split.

    Uses provenance_schema_version to distinguish:
      absent → legacy: emit WARNING, skip strict checks.
      1      → new: all _V1_REQUIRED_PROVENANCE_FIELDS must be present and match.
      other  → fail-fast (unknown schema).

    Raises SystemExit on any mismatch for v1 checkpoints.
    """
    schema_version = resume_ckpt.get("provenance_schema_version")

    if schema_version is None:
        print("  [provenance] WARNING: legacy checkpoint (no provenance_schema_version).")
        print("  [provenance] Strict dataset verification skipped; provenance_verified=False.")
        return

    if schema_version != 1:
        print(f"\nERROR: unsupported provenance_schema_version={schema_version!r}.")
        print("  This checkpoint was produced by a newer version of the training code.")
        sys.exit(1)

    # v1: all required provenance fields must exist in the checkpoint
    missing = sorted(f for f in V1_REQUIRED_PROVENANCE_FIELDS
                     if f not in resume_ckpt)
    if missing:
        print(f"\nERROR: v1 checkpoint missing {len(missing)} required provenance field(s):")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)

    # Sentinel for "rebuilt value cannot be computed" (e.g. ls is None).
    # Distinct from None, which is a legitimate value (e.g. split_seed_effective in
    # full_lexicon mode where None == None is a valid match).
    _NOT_AVAILABLE = object()

    # Compute rebuilt values
    ls             = getattr(lexicon, "load_stats", None)
    train_words    = [e.word for e in train_entries]
    rebuilt_ord    = sha256_words_ordered(train_words)
    rebuilt_srt    = sha256_words_sorted(train_words)
    rebuilt_file_sha = (ls.lexicon_file_sha256 if ls else _NOT_AVAILABLE)
    split_mode     = getattr(cfg.data, "split_mode", "standard")
    is_full_lex    = (split_mode == "full_lexicon")

    def _chk(label, rebuilt, saved):
        """Return (label, rebuilt, saved, ok).

        ok=None   : check skipped (rebuilt is _NOT_AVAILABLE — can't compute).
        ok=True   : rebuilt == saved, including None == None (valid match).
        ok=False  : mismatch, including None vs non-None.
        """
        if rebuilt is _NOT_AVAILABLE:
            return (label, None, saved, None)
        return (label, rebuilt, saved, rebuilt == saved)

    checks = [
        _chk("split_mode",              split_mode,                                    resume_ckpt["split_mode"]),
        _chk("train_all_words",         is_full_lex,                                   resume_ckpt["train_all_words"]),
        _chk("validation_enabled",      not is_full_lex,                               resume_ckpt["validation_enabled"]),
        _chk("val_fraction",            cfg.data.val_fraction,                         resume_ckpt["val_fraction"]),
        _chk("n_source_rows",           ls.n_source_rows if ls else _NOT_AVAILABLE,    resume_ckpt["n_source_rows"]),
        _chk("n_entries_after_loading", ls.n_entries_after_loading if ls else _NOT_AVAILABLE,
             resume_ckpt["n_entries_after_loading"]),
        _chk("n_filtered_unknown_phoneme", ls.n_filtered_unknown_phoneme if ls else _NOT_AVAILABLE,
             resume_ckpt["n_filtered_unknown_phoneme"]),
        _chk("n_filtered_length",       ls.n_filtered_length if ls else _NOT_AVAILABLE,
             resume_ckpt["n_filtered_length"]),
        _chk("n_unique_loaded_words",   ls.n_unique_loaded_words if ls else _NOT_AVAILABLE,
             resume_ckpt["n_unique_loaded_words"]),
        _chk("n_train",                 len(train_entries),                            resume_ckpt["n_train"]),
        _chk("n_val",                   len(val_entries),                              resume_ckpt["n_val"]),
        _chk("n_unique_train_words",    len(set(train_words)),                         resume_ckpt["n_unique_train_words"]),
        _chk("n_glove_found",           ls.n_glove_found if ls else _NOT_AVAILABLE,    resume_ckpt["n_glove_found"]),
        _chk("n_glove_fallback",        ls.n_glove_fallback if ls else _NOT_AVAILABLE, resume_ckpt["n_glove_fallback"]),
        _chk("lexicon_file_sha256",     rebuilt_file_sha,                              resume_ckpt["lexicon_file_sha256"]),
        _chk("ordered_training_words_sha256", rebuilt_ord,                             resume_ckpt["ordered_training_words_sha256"]),
        _chk("sorted_training_words_sha256",  rebuilt_srt,                             resume_ckpt["sorted_training_words_sha256"]),
        # Split seed contract — None is a legitimate value (full_lexicon: both sides are None)
        _chk("split_seed_used",         not is_full_lex,                               resume_ckpt["split_seed_used"]),
        _chk("split_seed_effective",    (None if is_full_lex
                                         else get_effective_split_seed(cfg.data)),
             resume_ckpt["split_seed_effective"]),
        # lexicon_file_path and glove_file_path: presence required (v1), value not compared
        # (absolute paths can legitimately differ between machines).
    ]

    errors = []
    for label, rebuilt, saved, ok in checks:
        if ok is None:
            continue   # check skipped (rebuilt not computable)
        if not ok:
            short_r = str(rebuilt)[:64]
            short_s = str(saved)[:64]
            errors.append(f"  {label}:\n    rebuilt   = {short_r}\n    checkpoint= {short_s}")
        else:
            disp = str(rebuilt)[:20]
            print(f"  [provenance] {label}: OK ({disp})")

    if errors:
        print(f"\nERROR: {len(errors)} provenance check(s) failed at resume (schema_version=1):")
        for e in errors:
            print(e)
        print("\nCheck --lexicon_path, --max_words, split_seed, split_mode, GloVe path.")
        sys.exit(1)

    checked = sum(1 for _, _, _, ok in checks if ok is not None)
    print(f"  [provenance] ALL {checked} CHECKS PASSED (schema_version=1)")


def _run_val_epoch(model, val_loader, cfg, validation_enabled):
    """Run val epoch if validation is enabled; return (val_rep, val_wm) or (None, None)."""
    if validation_enabled:
        import torch
        with torch.no_grad():
            va = run_epoch(model, val_loader, cfg, optim=None)
        return va.get("rep"), va.get("wm")
    return None, None


def main() -> None:
    import itertools
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    git_commit = git_state(cwd=ROOT).get("commit", "")
    PREMOTOR_DIM = 128   # architecture constant (not in config)

    os.makedirs(os.path.dirname(os.path.abspath(args.ckpt)), exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # Epoch metadata (set per-mode, used in final _build_ckpt_dict call)
    _target_total_epochs: int = args.epochs
    _resume_epochs_added: "Optional[int]" = None

    # ----------------------------------------------------------------------- #
    # RESUME MODE
    # ----------------------------------------------------------------------- #
    if args.resume_from is not None:
        if not os.path.exists(args.resume_from):
            print(f"\nERROR: --resume_from checkpoint not found: {args.resume_from}\n")
            sys.exit(1)

        print(f"\n[train_checkpoint] RESUME from {args.resume_from}")
        resume_ckpt = torch.load(args.resume_from, map_location=device,
                                  weights_only=False)

        # 1. Restore full configuration from checkpoint
        train_cfg_dict = resume_ckpt["cfg_train"]
        train_cfg_dict.setdefault("teacher_forcing_ratio", 1.0)
        train_cfg_dict.setdefault("save_every_epochs", 0)
        train_cfg_dict.setdefault("num_workers", 0)

        cfg = Config(
            data   = DataConfig(**resume_ckpt["cfg_data"]),
            wm     = WMConfig(**resume_ckpt["cfg_wm"]),
            ltm    = LTMConfig(**resume_ckpt["cfg_ltm"]),
            gating = GatingConfig(**resume_ckpt["cfg_gating"]),
            loss   = LossConfig(**resume_ckpt["cfg_loss"]),
            train  = TrainConfig(**train_cfg_dict),
        )
        cfg.train.device = device

        # 2. Fail-fast on contradictory CLI args
        _resume_fail_fast(args, cfg)

        # 3. Apply operational overrides (safe to change at resume)
        if args.lr is not None:
            cfg.train.lr = args.lr
        if args.num_workers is not None:
            cfg.train.num_workers = args.num_workers
        if args.save_every_epochs is not None:
            cfg.train.save_every_epochs = args.save_every_epochs

        # 4. Handle --seed: RNG states from checkpoint take precedence
        if args.seed is not None and args.seed != cfg.train.seed:
            print(f"  [resume] WARNING: --seed {args.seed} ignored. "
                  f"RNG states from checkpoint (training_seed={cfg.train.seed}) "
                  f"take precedence.")

        # 5. Epoch scheduling
        prior_history = resume_ckpt.get("history", [])
        epochs_done   = len(prior_history)
        epochs_needed = args.epochs - epochs_done
        if epochs_needed <= 0:
            print(f"  Already at {epochs_done} epochs — nothing to do.")
            return

        split_mode        = getattr(cfg.data, "split_mode", "standard")
        validation_enabled = (split_mode != "full_lexicon")
        SPLIT_SEED        = (None if not validation_enabled
                             else get_effective_split_seed(cfg.data))
        # cfg.train.epochs stores the target total epoch count (for this run).
        # The loop uses epochs_needed; cfg_train in the saved checkpoint will reflect
        # the target total so that a second resume knows how many epochs are done.
        cfg.train.epochs = args.epochs

        validate_split_config(cfg.data)

        print(f"  split_mode      : {split_mode}")
        print(f"  validation      : {'on' if validation_enabled else 'DISABLED'}")
        print(f"  prior epochs    : {epochs_done}")
        print(f"  target total    : {args.epochs}")
        print(f"  will train      : {epochs_needed} more epochs")
        print(f"  LTM mode        : {cfg.ltm.ltm_encoder_mode}")
        print(f"  lr              : {cfg.train.lr}")

        # 6. Rebuild dataset (deterministic from restored config)
        from models.dual_route import DualRouteModel
        from data.lexicon import build_lexicon
        from data.phonemes import build_vocab
        from data.dataset import make_loader, build_pool_loader

        vocab   = build_vocab()
        lexicon = build_lexicon(cfg.data, vocab)
        split_seed_arg = SPLIT_SEED if SPLIT_SEED is not None else 0
        train_entries, val_entries = lexicon.split(cfg.data.val_fraction, split_seed_arg)

        # 7. Strict provenance: verify all dataset invariants against checkpoint
        _verify_dataset_provenance(resume_ckpt, cfg, lexicon, train_entries, val_entries)

        # 8. Build model, loaders
        bank = torch.stack([torch.tensor(e.semantic) for e in train_entries]).float()
        bank = bank.to(device)
        model = DualRouteModel(cfg, vocab).to(device)
        model.set_semantic_bank(bank)

        try:
            model.load_state_dict(resume_ckpt["model_state_dict"])
        except RuntimeError as e:
            print(f"\nERROR loading weights from {args.resume_from}: {e}")
            print(f"  Saved cfg_ltm: {resume_ckpt.get('cfg_ltm', 'unknown')}")
            sys.exit(1)
        print(f"  Loaded weights from {args.resume_from}")

        num_workers = cfg.train.num_workers
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

        # 9. Optimizer
        effective_lr = cfg.train.lr
        optim = torch.optim.AdamW(model.parameters(), lr=effective_lr,
                                   weight_decay=cfg.train.weight_decay)
        if resume_ckpt.get("optimizer_state_dict") is not None:
            optim.load_state_dict(resume_ckpt["optimizer_state_dict"])
            for state in optim.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            if args.lr is not None:
                for g in optim.param_groups:
                    g["lr"] = args.lr
            print(f"  Optimizer : restored  (lr={effective_lr})")
        else:
            print(f"  Optimizer : warm-restart  (lr={effective_lr})")

        # 10. Restore RNG states
        if "rng_states" in resume_ckpt:
            rs = resume_ckpt["rng_states"]
            if "torch" in rs:
                torch.set_rng_state(_as_cpu_byte_tensor(rs["torch"], "torch_rng"))
                print("  [resume] restored torch RNG state")
            if "cuda" in rs and torch.cuda.is_available():
                cuda_states = [_as_cpu_byte_tensor(s, "cuda_rng") for s in rs["cuda"]]
                torch.cuda.set_rng_state_all(cuda_states)
                print("  [resume] restored CUDA RNG state")
            if "numpy" in rs:
                import numpy as np
                np.random.set_state(rs["numpy"])
                print("  [resume] restored numpy RNG state")
            print("  RNG states: restored from checkpoint")
        else:
            print("  RNG states: not saved in checkpoint (non-reproducible resume)")

        save_every = getattr(cfg.train, "save_every_epochs", 0)
        print(f"\n[train_checkpoint] {cfg.ltm.ltm_encoder_mode}"
              f"  lexicon={len(lexicon)} ({lexicon.source})"
              f"  vocab={vocab.size}  device={device}"
              f"  validation={'on' if validation_enabled else 'DISABLED'}")

        new_history = []
        for ep in range(epochs_needed):
            tr = run_epoch(model, train_loader, cfg, optim, pool_iter=pool_iter)
            val_rep, val_wm = _run_val_epoch(model, val_loader, cfg, validation_enabled)
            total_ep = epochs_done + ep + 1
            row = {
                "epoch": total_ep,
                "train_total": tr["total"], "train_rep": tr["rep"],
                "train_wm": tr["wm"], "val_rep": val_rep, "val_wm": val_wm,
            }
            new_history.append(row)
            val_str = (f"val rep={val_rep:.3f}" if val_rep is not None
                       else "val: DISABLED")
            print(f"[ep {total_ep:3d}/{args.epochs}] "
                  f"train total={tr['total']:.3f} rep={tr['rep']:.3f} "
                  f"align={tr['align']:.3f} wm={tr['wm']:.3f} | {val_str}")

            if save_every > 0 and total_ep % save_every == 0:
                partial_history = prior_history + new_history
                periodic_ckpt = _build_ckpt_dict(
                    model, optim, cfg, lexicon, train_entries, val_entries,
                    partial_history, git_commit, args.resume_from, PREMOTOR_DIM,
                    target_total_epochs=args.epochs,
                    resume_epochs_added=len(new_history))
                p_path = _periodic_ckpt_path(args.ckpt, total_ep)
                _save_checkpoint(periodic_ckpt, p_path)
                print(f"  [periodic ckpt] -> {p_path}")

        history               = prior_history + new_history
        _resume_epochs_added  = epochs_needed
        resumed_from          = args.resume_from
        from train import plot_loss_history
        plot_loss_history(history, os.path.join(args.out_dir, "training_loss.png"))

    # ----------------------------------------------------------------------- #
    # FRESH MODE
    # ----------------------------------------------------------------------- #
    else:
        cfg = default_config()
        cfg.train.device = device
        _apply_fresh_cli(cfg, args)
        SPLIT_SEED = (None if cfg.data.split_mode == "full_lexicon"
                      else get_effective_split_seed(cfg.data))
        save_every = getattr(cfg.train, "save_every_epochs", 0)

        split_mode         = cfg.data.split_mode
        validation_enabled = (split_mode != "full_lexicon")

        print(f"\n[train_checkpoint] FRESH RUN")
        print(f"  split_mode={split_mode}  epochs={args.epochs}")
        print(f"  max_words={cfg.data.max_words}  seed={cfg.train.seed}")
        print(f"  split_seed_effective={SPLIT_SEED}  device={device}")
        print(f"  ltm_encoder_mode={cfg.ltm.ltm_encoder_mode}")
        print(f"  wm_hidden={cfg.wm.hidden}  ltm_enc_hidden={cfg.ltm.enc_hidden}"
              f"  ltm_dec_hidden={cfg.ltm.dec_hidden}")
        print(f"  teacher_forcing_ratio={cfg.train.teacher_forcing_ratio}"
              f"  interference_noise={cfg.wm.interference_noise}"
              f"  ventral_noise={cfg.ltm.ventral_noise}"
              f"  gate_alpha={cfg.gating.alpha}"
              f"  gate_threshold={cfg.gating.gate_threshold}")
        print(f"  validation={'on' if validation_enabled else 'DISABLED (full_lexicon)'}")
        print(f"  num_workers={cfg.train.num_workers}"
              f"  save_every_epochs={save_every}")
        print(f"  checkpoint -> {args.ckpt}\n")

        if save_every > 0:
            import itertools as _it
            from train import plot_loss_history
            if cfg.train.device == "cpu" and torch.cuda.is_available():
                cfg.train.device = "cuda"
            (model, vocab, lexicon,
             train_loader, val_loader, pool_loader) = build_everything(cfg)
            optim = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                       weight_decay=cfg.train.weight_decay)
            pool_iter = _it.cycle(pool_loader) if pool_loader is not None else None
            split_seed_arg = SPLIT_SEED if SPLIT_SEED is not None else 0
            train_entries, val_entries = lexicon.split(cfg.data.val_fraction,
                                                        split_seed_arg)
            history = []
            for ep in range(cfg.train.epochs):
                tr = run_epoch(model, train_loader, cfg, optim, pool_iter=pool_iter)
                val_rep, val_wm = _run_val_epoch(model, val_loader, cfg, validation_enabled)
                row = {"epoch": ep + 1,
                       "train_total": tr["total"], "train_rep": tr["rep"],
                       "train_wm": tr["wm"], "val_rep": val_rep, "val_wm": val_wm}
                history.append(row)
                val_str = (f"val rep={val_rep:.3f} wm={val_wm:.3f}"
                           if val_rep is not None else "val: DISABLED")
                print(f"[ep {ep+1:2d}/{cfg.train.epochs}] "
                      f"train total={tr['total']:.3f} rep={tr['rep']:.3f} "
                      f"align={tr['align']:.3f} wm={tr['wm']:.3f} | {val_str}")
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
            split_seed_arg = SPLIT_SEED if SPLIT_SEED is not None else 0
            train_entries, val_entries = lexicon.split(cfg.data.val_fraction,
                                                        split_seed_arg)
        resumed_from = None

    # ----------------------------------------------------------------------- #
    # Save final checkpoint (common to both modes)
    # ----------------------------------------------------------------------- #
    ckpt = _build_ckpt_dict(model, optim, cfg, lexicon, train_entries, val_entries,
                             history, git_commit, resumed_from, PREMOTOR_DIM,
                             target_total_epochs=_target_total_epochs,
                             resume_epochs_added=_resume_epochs_added)
    _save_checkpoint(ckpt, args.ckpt)

    print(f"\n[train_checkpoint] saved  -> {args.ckpt}")
    print(f"  split_mode           : {ckpt['split_mode']}")
    print(f"  validation_enabled   : {ckpt['validation_enabled']}")
    print(f"  ltm_encoder_mode     : {cfg.ltm.ltm_encoder_mode}")
    print(f"  wm_hidden            : {cfg.wm.hidden}")
    print(f"  ltm_enc/dec_hidden   : {cfg.ltm.enc_hidden} / {cfg.ltm.dec_hidden}")
    print(f"  gate_threshold       : {cfg.gating.gate_threshold}")
    print(f"  lexicon_source       : {lexicon.source}")
    print(f"  n_train / n_val      : {ckpt['n_train']} / {ckpt['n_val']}")
    print(f"  n_glove_found        : {ckpt['n_glove_found']}")
    print(f"  n_glove_fallback     : {ckpt['n_glove_fallback']}")
    print(f"  ordered_sha          : {ckpt['ordered_training_words_sha256'][:16]}…")
    print(f"  lexicon_file_sha     : {(ckpt['lexicon_file_sha256'] or 'N/A')[:16]}…")
    print(f"  glove_present        : {ckpt['glove_present']}")
    print(f"  total_epochs_trained : {ckpt['total_epochs_trained']}")
    print(f"  optimizer_saved      : {ckpt['optimizer_state_dict'] is not None}")
    print(f"  resumed_from         : {ckpt['resumed_from'] or 'N/A (fresh run)'}")
    print(f"  git_commit           : {ckpt['git_commit'] or '(not available)'}")
    print(f"  final epoch          : {history[-1]}")


if __name__ == "__main__":
    main()
