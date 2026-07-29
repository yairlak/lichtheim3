"""Step 7: Ceiling evaluation on all trained words.

After training, verify that the model has achieved (near-)perfect exact-match
accuracy on every word it was trained on.  Imperfect ceiling means the model
is still underfitting and needs more epochs.

Supports two decode modes (--decode):
    teacher_forced   : gold prefix fed at each step (historical default, ceiling probe)
    autoregressive   : model's own previous output fed at each step (free-generation)

Prerequisite: a checkpoint created by scripts/train_checkpoint.py.

Outputs — teacher_forced mode: written to <out_dir>/
           autoregressive mode: written to <out_dir>/ar/
    metrics.json                  overall metrics per route
    summary.md                    human-readable summary
    item_level_predictions.tsv    per-word predictions with edit distances
    train_errors.tsv              training words the full route still gets wrong

Success criterion (TF): full_exact_match = 1.000 AND train_errors.tsv is empty.

Usage:
    python scripts/evaluate_train_lexicon_ceiling.py \\
        --ckpt checkpoints/lichtheim3_30k_glove.pt

    # Autoregressive decoding (primary behavioral metric):
    python scripts/evaluate_train_lexicon_ceiling.py \\
        --ckpt checkpoints/lichtheim3_30k_glove.pt \\
        --decode autoregressive --include_val

    # With explicit lexicon path:
    python scripts/evaluate_train_lexicon_ceiling.py \\
        --ckpt checkpoints/lichtheim3_30k_glove.pt \\
        --lexicon_path data/lexicon_en_glove_covered.tsv
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
from config import get_effective_split_seed, validate_split_config
from data.phonemes import build_vocab, Vocab
from data.lexicon import build_lexicon
from models.dual_route import DualRouteModel
from evaluate.hooks import make_batch, route_predictions
from utils.provenance import (sha256_words_ordered, sha256_words_sorted, sha256_file,
                               PROVENANCE_SCHEMA_VERSION, V1_REQUIRED_PROVENANCE_FIELDS)

CKPT_DEFAULT = os.path.join(ROOT, "checkpoints", "lichtheim3.pt")
OUT_DIR      = os.path.join(ROOT, "outputs", "train_lexicon_ceiling")
BATCH_SIZE   = 128

DECODE_TF = "teacher_forced"
DECODE_AR = "autoregressive"

EVAL_NOTE_TF = (
    "Teacher-forced decoding: gold prefix fed at each decoder step.  "
    "Accuracy reflects reconstruction quality, NOT free-generation."
)
EVAL_NOTE_AR = (
    "Autoregressive decoding: model's own previous output fed at each step.  "
    "Reflects free-generation accuracy; errors can propagate."
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
    p.add_argument("--decode", choices=[DECODE_TF, DECODE_AR], default=DECODE_TF,
                   help=(
                       f"decoding mode: '{DECODE_TF}' (default, ceiling probe) or "
                       f"'{DECODE_AR}' (free-generation, primary behavioral metric).  "
                       f"AR outputs go to <out_dir>/ar/ to preserve backward compatibility."
                   ))
    p.add_argument("--wm_noise",     action="store_true",
                   help=(
                       "enable WM interference noise during evaluation "
                       "(makes WM predictions non-deterministic; off by default "
                       "for ceiling evaluation — use for external eval consistency).  "
                       "Note: applies only to the WM-isolated route, not WM inside "
                       "the full/gated route."
                   ))
    p.add_argument("--device",       default=None)
    p.add_argument("--ignore_provenance", action="store_true",
                   help=(
                       "Skip hash verification of the reconstructed training set "
                       "against the checkpoint. For diagnostics only — never use "
                       "to bypass a real mismatch in production."
                   ))
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
# Autoregressive decode helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _ar_decode_batch(model: DualRouteModel, vocab: Vocab,
                     batch_forms: List[List[int]], device: str,
                     routes: Tuple[str, ...],
                     wm_noise: bool) -> Dict[str, List[List[int]]]:
    """Autoregressive greedy decode for a list of phoneme-id forms.

    Returns a dict {route: list_of_pred_id_lists}.  Each prediction is
    stripped at the first EOS token (exclusive).

    Note: wm_noise (apply_noise=True) applies noise to the WM-isolated route only.
    It does NOT apply noise to the WM component inside the full/gated route.
    collect=False here — we want AR decode only, not diagnostic collection.
    """
    batch = make_batch(batch_forms, vocab, device)
    max_steps = max(len(f) for f in batch_forms) + 1  # +1 for possible EOS
    preds: Dict[str, List[List[int]]] = {r: [] for r in routes}

    for route in routes:
        apply_noise = (route == "wm") and wm_noise   # explicit noise, not via collect
        dec_input = batch["enc_in"].new_full((len(batch_forms), 1), vocab.bos_id)
        for _ in range(max_steps):
            res = model.route_logits(batch["enc_in"], batch["enc_mask"],
                                     dec_input, route=route,
                                     collect=False, apply_noise=apply_noise)
            next_tok = res["logits"][:, -1, :].argmax(-1, keepdim=True)
            dec_input = torch.cat([dec_input, next_tok], dim=1)

        for i, form in enumerate(batch_forms):
            n_steps = len(form) + 1   # read at most len(form)+1 tokens
            raw = dec_input[i, 1: 1 + n_steps].tolist()
            seq: List[int] = []
            for tok in raw:
                if tok == vocab.eos_id:
                    break
                seq.append(tok)
            preds[route].append(seq)

    return preds


@torch.no_grad()
def evaluate_forms_ar(model: DualRouteModel, vocab: Vocab,
                      entries, device: str,
                      routes: Tuple[str, ...] = ("full", "wm", "ltm"),
                      wm_noise: bool = CEILING_COLLECT_WM_NOISE) -> List[dict]:
    """Autoregressive evaluation of a list of LexEntry objects.

    Mirrors evaluate_forms() but uses free-generation decoding.
    Outputs the same columns plus norm_edit_dist = edit_dist / max(1, len(target)).
    """
    results = []
    forms_ids = [e.phonemes for e in entries]
    n = len(forms_ids)

    for start in range(0, n, BATCH_SIZE):
        batch_forms   = forms_ids[start: start + BATCH_SIZE]
        batch_entries = entries[start:  start + BATCH_SIZE]

        preds_by_route = _ar_decode_batch(model, vocab, batch_forms, device, routes,
                                          wm_noise)

        for i, (entry, form_ids) in enumerate(zip(batch_entries, batch_forms)):
            tgt_syms = [vocab.itos[idx] for idx in form_ids]
            row: dict = {
                "word":            entry.word,
                "rank":            entry.rank,
                "target_phonemes": " ".join(tgt_syms),
                "length":          len(form_ids),
                "zipf_approx":     round(float(entry.freq), 6),
            }
            for route in routes:
                pred_ids  = preds_by_route[route][i]
                pred_syms = [vocab.itos[idx] for idx in pred_ids]
                exact     = int(pred_syms == tgt_syms)
                ed        = _edit_distance(tgt_syms, pred_syms)
                ned       = round(ed / max(len(tgt_syms), 1), 4)
                row[f"{route}_exact_match"]     = exact
                row[f"{route}_edit_dist"]       = ed
                row[f"{route}_norm_edit_dist"]  = ned
                row[f"{route}_predicted"]       = " ".join(pred_syms)
            results.append(row)

        if (start // BATCH_SIZE) % 20 == 0:
            print(f"  … {min(start + BATCH_SIZE, n)}/{n}", end="\r")

    print()
    return results


# ---------------------------------------------------------------------------
# Inference (teacher-forced)
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
        # apply_noise=True for WM route ONLY if wm_noise is explicitly requested.
        # collect=False: TF ceiling eval is deterministic; no diagnostics needed here.
        batch_preds: Dict[str, torch.Tensor] = {}
        for route in routes:
            apply_noise = (route == "wm") and wm_noise
            p, _ = route_predictions(model, batch, route=route,
                                     collect=False, apply_noise=apply_noise)
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
                ned       = round(ed / max(len(tgt_syms), 1), 4)
                from itertools import zip_longest
                phon_acc  = (
                    sum(a == b for a, b in zip_longest(pred_syms, tgt_syms, fillvalue=""))
                    / max(len(tgt_syms), 1)
                )
                row[f"{route}_exact_match"]     = exact
                row[f"{route}_phoneme_acc"]     = round(phon_acc, 4)
                row[f"{route}_edit_dist"]       = ed
                row[f"{route}_norm_edit_dist"]  = ned
                row[f"{route}_predicted"]       = " ".join(pred_syms)
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

    # AR outputs go into a subdirectory to preserve backward compat for TF callers
    decode_mode = args.decode
    out_dir = args.out_dir if decode_mode == DECODE_TF else os.path.join(args.out_dir, "ar")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(args.ckpt):
        print(f"\nERROR: checkpoint not found: {args.ckpt}\n"
              "Run: python scripts/train_checkpoint.py --max_words 30000 --epochs 30 --seed 0\n")
        sys.exit(1)

    print(f"\n[ceiling_eval] Loading {args.ckpt} …")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    # Backward-compatible config load: handle checkpoints that predate new fields
    train_cfg_dict = ckpt["cfg_train"]
    train_cfg_dict.setdefault("teacher_forcing_ratio", 1.0)
    train_cfg_dict.setdefault("save_every_epochs", 0)
    train_cfg_dict.setdefault("num_workers", 0)

    cfg = Config(
        data   = DataConfig(**ckpt["cfg_data"]),
        wm     = WMConfig(**ckpt["cfg_wm"]),
        ltm    = LTMConfig(**ckpt["cfg_ltm"]),
        gating = GatingConfig(**ckpt["cfg_gating"]),
        loss   = LossConfig(**ckpt["cfg_loss"]),
        train  = TrainConfig(**train_cfg_dict),
    )
    cfg.train.device = device
    if args.lexicon_path:
        cfg.data.lexicon_path = args.lexicon_path
    elif not hasattr(cfg.data, "lexicon_path"):
        cfg.data.lexicon_path = None
    validate_split_config(cfg.data)

    vocab   = build_vocab()
    lexicon = build_lexicon(cfg.data, vocab)
    split_mode      = getattr(cfg.data, "split_mode", "standard")
    split_seed_eval = (0 if split_mode == "full_lexicon"
                       else get_effective_split_seed(cfg.data))
    train_entries, val_entries = lexicon.split(cfg.data.val_fraction, split_seed_eval)

    # --- Provenance: per-check verification of dataset invariants ---
    ignore_provenance = getattr(args, "ignore_provenance", False)
    ls = getattr(lexicon, "load_stats", None)

    # Sentinel for "rebuilt value cannot be computed" (e.g. ls is None).
    # Distinct from None, which is a legitimate rebuilt value (e.g. split_seed_effective
    # in full_lexicon mode, where None == None is a valid match).
    _NOT_AVAILABLE = object()

    # Compute rebuilt values
    train_words         = [e.word for e in train_entries]
    rebuilt_ordered_sha = sha256_words_ordered(train_words)
    rebuilt_sorted_sha  = sha256_words_sorted(train_words)
    rebuilt_file_sha    = (ls.lexicon_file_sha256 if ls else _NOT_AVAILABLE)
    is_full_lex         = (split_mode == "full_lexicon")

    # Determine checkpoint generation from provenance_schema_version
    schema_version = ckpt.get("provenance_schema_version")
    if schema_version is not None and schema_version != PROVENANCE_SCHEMA_VERSION:
        print(f"\n[provenance] ERROR: unsupported provenance_schema_version={schema_version!r}.")
        print(f"  Expected {PROVENANCE_SCHEMA_VERSION}. "
              "This checkpoint was produced by a newer version of the code.")
        if not ignore_provenance:
            sys.exit(1)

    is_legacy = (schema_version is None)

    def _pchk(label, rebuilt, saved_key, mandatory=True):
        """Build a per-check result dict with native Python/JSON types.

        Native values are stored directly so that json.dump produces the correct
        JSON types (null, true/false, numbers, strings) instead of string
        representations ("None", "True", etc.).

        Fields:
          rebuilt_available  : True iff rebuilt can be computed (False when _NOT_AVAILABLE).
          checkpoint_present : True iff saved_key exists in ckpt.
          checked            : True iff both available+present; comparison was performed.
          match              : rebuilt == saved (None==None → True); None if not checked.
          rebuilt            : native Python value or None when not available.
          checkpoint         : native Python value or None when key absent.

        For full_lexicon split_seed_effective:
          rebuilt=None, checkpoint=None, rebuilt_available=True,
          checkpoint_present=True, checked=True, match=True.
        """
        checkpoint_present = (saved_key in ckpt)
        rebuilt_available  = (rebuilt is not _NOT_AVAILABLE)

        if not checkpoint_present:
            return {
                "field": label,
                "rebuilt":             rebuilt if rebuilt_available else None,
                "checkpoint":          None,
                "rebuilt_available":   rebuilt_available,
                "checkpoint_present":  False,
                "checked":             False,
                "match":               None,
                "mandatory":           mandatory,
            }
        saved = ckpt[saved_key]
        if not rebuilt_available:
            return {
                "field": label,
                "rebuilt":             None,
                "checkpoint":          saved,
                "rebuilt_available":   False,
                "checkpoint_present":  True,
                "checked":             False,
                "match":               None,
                "mandatory":           mandatory,
            }
        # Both available and present: compare with native types.
        # None == None → True  (e.g. split_seed_effective in full_lexicon mode).
        match = (rebuilt == saved)
        return {
            "field": label,
            "rebuilt":             rebuilt,
            "checkpoint":          saved,
            "rebuilt_available":   True,
            "checkpoint_present":  True,
            "checked":             True,
            "match":               match,
            "mandatory":           mandatory,
        }

    # For v1: n_glove_found and n_glove_fallback are mandatory (real lexicon)
    glove_mandatory = (schema_version == 1)

    prov_checks = [
        _pchk("split_mode",              split_mode,                                    "split_mode"),
        _pchk("train_all_words",         is_full_lex,                                   "train_all_words"),
        _pchk("validation_enabled",      not is_full_lex,                               "validation_enabled"),
        _pchk("n_source_rows",           ls.n_source_rows if ls else _NOT_AVAILABLE,    "n_source_rows"),
        _pchk("n_entries_after_loading", ls.n_entries_after_loading if ls else _NOT_AVAILABLE,
              "n_entries_after_loading"),
        _pchk("n_train",                 len(train_entries),                            "n_train"),
        _pchk("n_val",                   len(val_entries),                              "n_val"),
        _pchk("n_unique_train_words",    len(set(train_words)),                         "n_unique_train_words"),
        _pchk("n_glove_found",           ls.n_glove_found if ls else _NOT_AVAILABLE,    "n_glove_found",
              mandatory=glove_mandatory),
        _pchk("n_glove_fallback",        ls.n_glove_fallback if ls else _NOT_AVAILABLE, "n_glove_fallback",
              mandatory=glove_mandatory),
        _pchk("lexicon_file_sha256",     rebuilt_file_sha,                              "lexicon_file_sha256"),
        _pchk("ordered_training_words_sha256", rebuilt_ordered_sha,
              "ordered_training_words_sha256"),
        _pchk("sorted_training_words_sha256",  rebuilt_sorted_sha,
              "sorted_training_words_sha256"),
        # Split seed contract — None is a legitimate value for full_lexicon
        _pchk("split_seed_used",         not is_full_lex,                               "split_seed_used"),
        _pchk("split_seed_effective",    (None if is_full_lex
                                          else get_effective_split_seed(cfg.data)),
              "split_seed_effective"),
    ]

    if is_legacy:
        provenance_verified = False
        provenance_status   = "legacy"
        all_checks_passed   = False
        print("  [provenance] WARNING: legacy checkpoint (no provenance_schema_version).")
        print("  [provenance] Strict checks skipped; provenance_verified=False.")
        print(f"  [provenance] rebuilt ordered_sha: {rebuilt_ordered_sha[:16]}…  (unverified)")
    else:
        # v1: check all required fields are present
        missing_fields = sorted(f for f in V1_REQUIRED_PROVENANCE_FIELDS if f not in ckpt)
        if missing_fields:
            print(f"\n  [provenance] ERROR: v1 checkpoint missing required fields:")
            for f in missing_fields:
                print(f"    - {f}")
            if not ignore_provenance:
                sys.exit(1)

        # all_checks_passed requires EVERY mandatory check to be checked=True AND match=True.
        # A mandatory check with checked=False (uncalculable rebuilt) is not a pass.
        mandatory_checks = [c for c in prov_checks if c["mandatory"]]
        optional_checks  = [c for c in prov_checks if not c["mandatory"]]

        mandatory_ok = all(c["checked"] is True and c["match"] is True
                           for c in mandatory_checks)
        optional_ok  = all(c["match"] is not False for c in optional_checks)
        all_checks_passed = mandatory_ok and optional_ok
        provenance_verified = all_checks_passed

        # Collect failures for display
        failed_mandatory = [c for c in mandatory_checks
                            if not (c["checked"] is True and c["match"] is True)]
        failed_optional  = [c for c in optional_checks if c["match"] is False]

        if all_checks_passed:
            checked_n = sum(1 for c in prov_checks if c["checked"])
            provenance_status = "verified"
            print(f"  [provenance] ALL {checked_n} CHECKS PASSED (schema_version=1)")
        else:
            provenance_status = "mismatch"
            print(f"\n  [provenance] CHECK FAILURES (schema_version=1):")
            for c in failed_mandatory + failed_optional:
                tag = "[MANDATORY]" if c["mandatory"] else "[optional]"
                note = " (uncalculable)" if not c["checked"] else ""
                # repr() for log display only; native values stay in metrics.json
                print(f"    {tag}{note} {c['field']}:")
                print(f"      rebuilt   : {repr(c['rebuilt'])}")
                print(f"      checkpoint: {repr(c['checkpoint'])}")
            if failed_mandatory:
                if not ignore_provenance:
                    print("\n  Use --ignore_provenance to bypass (for diagnostics only).")
                    sys.exit(1)
                else:
                    print("\n  --ignore_provenance: continuing with UNVERIFIED dataset.")

    bank = torch.stack([torch.tensor(e.semantic) for e in train_entries]).float().to(device)
    model = DualRouteModel(cfg, vocab).to(device)
    model.set_semantic_bank(bank)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    eval_note = EVAL_NOTE_AR if decode_mode == DECODE_AR else EVAL_NOTE_TF
    wm_noise = getattr(args, "wm_noise", False)

    print(f"  checkpoint : {args.ckpt}")
    print(f"  glove      : {'present' if ckpt.get('glove_present') else 'ABSENT (pseudo-vectors)'}")
    print(f"  lexicon    : {lexicon.source}  max_words={cfg.data.max_words}")
    print(f"  split_mode : {split_mode}")
    print(f"  n_train    : {len(train_entries)}  n_val: {len(val_entries)}")
    _split_seed_display = (None if split_mode == "full_lexicon"
                           else get_effective_split_seed(cfg.data))
    print(f"  checkpoint_training_seed      : {cfg.train.seed}")
    print(f"  checkpoint_split_seed_effective: {_split_seed_display}")
    print(f"  device     : {device}")
    print(f"  decode     : {decode_mode}")
    print(f"  out_dir    : {out_dir}")
    print(f"  regime     : {eval_note[:60]}…")
    if wm_noise:
        print("  WM noise : ENABLED  (non-deterministic, WM-isolated route only)")
    else:
        print("  WM noise : DISABLED (deterministic)")

    routes = ("full", "wm", "ltm")
    all_rows = []

    # Select evaluation function based on decode mode
    eval_fn = evaluate_forms_ar if decode_mode == DECODE_AR else evaluate_forms

    # ---- Training split ----
    print(f"\n[ceiling_eval] Evaluating {len(train_entries)} training words "
          f"({decode_mode}) …")
    train_results = eval_fn(model, vocab, train_entries, device, routes,
                            wm_noise=wm_noise)
    for r in train_results:
        r["split"] = "train"
    all_rows.extend(train_results)

    # ---- Validation split (optional) ----
    if args.include_val:
        print(f"[ceiling_eval] Evaluating {len(val_entries)} validation words …")
        val_results = eval_fn(model, vocab, val_entries, device, routes,
                              wm_noise=wm_noise)
        for r in val_results:
            r["split"] = "val"
        all_rows.extend(val_results)

    # ---- Results ----
    df = pd.DataFrame(all_rows)
    pred_path = os.path.join(out_dir, "item_level_predictions.tsv")
    df.to_csv(pred_path, sep="\t", index=False)
    print(f"\n  -> {pred_path}")

    # Determine which columns are available (AR has norm_edit_dist but not phoneme_acc)
    has_phon_acc    = f"full_phoneme_acc"    in df.columns
    has_norm_edit   = f"full_norm_edit_dist" in df.columns

    # Per-split overall metrics
    splits_evaluated = ["train"] + (["val"] if args.include_val else [])
    metrics: dict = {
        "evaluation_note":   eval_note,
        "decode_mode":       decode_mode,
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
        "checkpoint_training_seed":       cfg.train.seed,
        # split_seed_effective: None when split_mode=full_lexicon (split seed not used)
        "checkpoint_split_seed_effective": (None if split_mode == "full_lexicon"
                                            else get_effective_split_seed(cfg.data)),
        "splits_evaluated":  splits_evaluated,
        "provenance": {
            "verified":               provenance_verified,
            "all_checks_passed":      all_checks_passed,
            "status":                 provenance_status,
            "ignore_provenance_flag": ignore_provenance,
            "legacy_checkpoint":      is_legacy,
            "split_mode":             split_mode,
            "validation_enabled":     ckpt.get("validation_enabled",
                                               split_mode != "full_lexicon"),
            # split_seed_effective: None for full_lexicon (seed not used for split)
            "split_seed_effective":   (None if split_mode == "full_lexicon"
                                       else get_effective_split_seed(cfg.data)),
            "glove_present":          bool(ckpt.get("glove_file_path")
                                          or ckpt.get("glove_present")),
            "n_glove_found":          ckpt.get("n_glove_found"),
            "n_glove_fallback":       ckpt.get("n_glove_fallback"),
            "checks":                 prov_checks,
        },
        "results": {},
    }
    for split in splits_evaluated:
        sub = df[df["split"] == split]
        metrics["results"][split] = {}
        for route in routes:
            entry: dict = {
                "exact_match": round(float(sub[f"{route}_exact_match"].mean()), 6),
                "edit_dist":   round(float(sub[f"{route}_edit_dist"].mean()),   4),
                "n_items":     len(sub),
                "n_errors":    int((sub[f"{route}_exact_match"] == 0).sum()),
            }
            if has_norm_edit:
                entry["norm_edit_dist"] = round(
                    float(sub[f"{route}_norm_edit_dist"].mean()), 4)
            if has_phon_acc:
                entry["phoneme_acc"] = round(
                    float(sub[f"{route}_phoneme_acc"].mean()), 6)
            metrics["results"][split][route] = entry

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  -> {metrics_path}")

    # Error table: training words the full route gets wrong
    train_df  = df[df["split"] == "train"]
    errors_df = train_df[train_df["full_exact_match"] == 0].copy()
    errors_df = errors_df.sort_values("full_edit_dist", ascending=False)
    err_path  = os.path.join(out_dir, "train_errors.tsv")
    errors_df.to_csv(err_path, sep="\t", index=False)
    print(f"  -> {err_path}  ({len(errors_df)} errors)")

    # Summary markdown
    _write_summary_md(metrics, len(errors_df), out_dir)

    # Console summary
    print(f"\n  === {'AR' if decode_mode == DECODE_AR else 'CEILING'} EVALUATION ===")
    print(f"  Note: {eval_note[:70]}…")
    for split in splits_evaluated:
        r = metrics["results"][split]
        print(f"\n  [{split}]")
        for route in routes:
            d = r[route]
            ok = "✓ CEILING" if d["exact_match"] == 1.0 else "✗ NOT YET"
            ned_str = (f"  ned={d['norm_edit_dist']:.3f}" if "norm_edit_dist" in d
                       else "")
            print(f"    {route:5s}  exact={d['exact_match']:.4f}  "
                  f"edit={d['edit_dist']:.3f}{ned_str}  errors={d['n_errors']}  {ok}")
    if decode_mode == DECODE_TF and len(errors_df) == 0:
        print("\n  CEILING REACHED: train_errors.tsv is empty.")
    elif decode_mode == DECODE_TF and len(errors_df) > 0:
        print(f"\n  {len(errors_df)} training errors remain.  "
              f"Consider more epochs or larger model.")


def _write_summary_md(metrics: dict, n_train_errors: int, out_dir: str) -> None:
    decode_mode = metrics.get("decode_mode", DECODE_TF)
    title = ("Train Lexicon AR Evaluation" if decode_mode == DECODE_AR
             else "Train Lexicon Ceiling Evaluation")
    prov = metrics.get("provenance", {})
    prov_status   = prov.get("status", "unknown")
    prov_verified = prov.get("verified", False)
    prov_icon = ("✓ VERIFIED" if prov_verified
                 else ("⚠ legacy" if prov_status in ("legacy", "legacy_no_hash")
                       else "✗ MISMATCH"))
    # Extract rebuilt ordered sha from per-check list (native value, may be None)
    _prov_checks  = prov.get("checks", [])
    _ord_check    = next((c for c in _prov_checks
                          if c["field"] == "ordered_training_words_sha256"), {})
    _ord_raw      = _ord_check.get("rebuilt", prov.get("rebuilt_ordered_sha256"))
    ordered_sha   = (f"{_ord_raw[:32]}…" if isinstance(_ord_raw, str) else "N/A")
    split_mode_str = prov.get("split_mode", "standard")
    lines = [
        f"# {title}",
        "",
        f"> {metrics['evaluation_note']}",
        "",
        "## Provenance",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Checkpoint | `{metrics['checkpoint']}` |",
        f"| Decode mode | {decode_mode} |",
        f"| GloVe present | {'YES' if metrics['glove_present'] else 'NO (pseudo-vectors)'} |",
        f"| Lexicon source | {metrics['lexicon_source']} |",
        f"| split_mode | {split_mode_str} |",
        f"| max_words | {metrics['cfg_max_words']} |",
        f"| epochs | {metrics['cfg_epochs']} |",
        f"| seed | {metrics['cfg_seed']} |",
        f"| n_train | {metrics['n_train']} |",
        f"| n_val | {metrics['n_val']} |",
        f"| Hash verification | {prov_icon} |",
        f"| ordered_training_sha256 | `{ordered_sha}` |",
        "",
        "## Results",
        "",
    ]
    for split, res in metrics["results"].items():
        # Determine which optional columns exist from the first route's dict
        sample = next(iter(res.values()))
        has_ned  = "norm_edit_dist" in sample
        has_pacc = "phoneme_acc"    in sample
        header_cols = ["Route", "Exact match"]
        if has_pacc:
            header_cols.append("Phoneme acc")
        header_cols += ["Edit dist"]
        if has_ned:
            header_cols.append("Norm edit dist")
        header_cols.append("Errors")
        lines += [
            f"### {split.capitalize()} split", "",
            "| " + " | ".join(header_cols) + " |",
            "|" + "|".join(["---"] * len(header_cols)) + "|",
        ]
        for route, d in res.items():
            ceiling = " ✓ CEILING" if d["exact_match"] == 1.0 else ""
            row_vals = [f"{route}", f"{d['exact_match']:.4f}{ceiling}"]
            if has_pacc:
                row_vals.append(f"{d.get('phoneme_acc', 0):.4f}")
            row_vals.append(f"{d['edit_dist']:.3f}")
            if has_ned:
                row_vals.append(f"{d.get('norm_edit_dist', 0):.3f}")
            row_vals.append(str(d["n_errors"]))
            lines.append("| " + " | ".join(row_vals) + " |")
        lines.append("")

    if decode_mode == DECODE_TF:
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
    else:
        lines += [
            "## Notes",
            "",
            "- Autoregressive eval: errors can propagate.  Lower than TF ceiling is expected.",
            "- Use `norm_edit_dist` (edit_dist / word_length) for length-controlled comparison.",
        ]
    path = os.path.join(out_dir, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
