"""Debug per-item prediction discrepancy between ceiling and external eval.

Run both code paths for a single word and print all intermediate tensors,
token IDs, gate values, and s_hat norms so the source of any discrepancy
can be isolated.

Confirmed root cause (verified on "rarely")
--------------------------------------------
LTMLexicon.encode() runs the biGRU over the full padded tensor without using
pack_padded_sequence.  The backward direction of the biGRU starts at the
rightmost position (which may be all PADs for short sequences in a long batch)
and processes left.  Because the GRU has non-zero bias terms, repeated zero-
input steps (PAD embedding = 0) still shift the hidden state.  This means
s_hat (masked mean-pool of the biGRU output) depends not only on the input
phonemes but also on how many trailing PAD positions exist in the batch.

Verified behaviour for "rarely" (R EH R L IY):
  - PATH A  solo batch (no trailing PADs):     WM ✓  LTM ✗  full ✗
  - PATH B  padded to WFE max len:             WM ✓  LTM ✓  full ✓
  - PATH C  actual external-eval batch:        WM ✓  LTM ✗  full ✗

Key observations:
  1. Same word, same phoneme IDs — no transcription mismatch.
  2. WM route: fully stable across all batch contexts.
  3. LTM route: output changes with batch padding context.
     Adding trailing PADs can flip the LTM from wrong to correct or vice versa,
     depending on what the model learned to expect during training (where batches
     always had mixed lengths with implicit trailing PADs for shorter items).
  4. Full/gated route: follows LTM when the gate confidence is high (≥ 0.5)
     for the item — which it is for train-seen real words.
  5. The direction of the effect is not simply "more padding = worse":
     for "rarely", the padded batch is CORRECT and the unpadded batch is WRONG.
     This is consistent with the model having been trained with mixed-length
     batches (always padded) and never seeing a truly unpadded solo forward pass.

Recommended fix for a retrained/clean architecture:
  Wrap the biGRU in LTMLexicon.encode() with pack_padded_sequence /
  pad_packed_sequence, or use mask-safe mean-pooling after setting PAD
  encoder positions to zero output.  Do NOT apply this fix to the current
  trained checkpoint — it would invalidate the learned weights.

Usage:
    python scripts/debug_single_item_prediction.py --word rarely
    python scripts/debug_single_item_prediction.py --word rarely --verbose
    python scripts/debug_single_item_prediction.py \\
        --word rarely \\
        --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt \\
        --wfe_tsv data/eval_external/wfe_eval.tsv
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import Config, DataConfig, WMConfig, LTMConfig, GatingConfig, LossConfig, TrainConfig
from data.phonemes import build_vocab, Vocab
from data.lexicon import build_lexicon
from models.dual_route import DualRouteModel
from evaluate.hooks import make_batch, route_predictions

CKPT_DEFAULT    = os.path.join(ROOT, "checkpoints",
                                "lichtheim3_30k_glove_e60_to_e120_lowlr.pt")
WFE_DEFAULT     = os.path.join(ROOT, "data", "eval_external", "wfe_eval.tsv")
LEXICON_DEFAULT = os.path.join(ROOT, "data", "lexicon_en_glove_covered.tsv")
ROUTES          = ("full", "wm", "ltm")

SEP  = "─" * 70
SEP2 = "═" * 70


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_everything(ckpt_path: str, device: str):
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
    train_entries, val_entries = lexicon.split(cfg.data.val_fraction, cfg.data.seed)
    bank = torch.stack(
        [torch.tensor(e.semantic) for e in train_entries]
    ).float().to(device)
    model = DualRouteModel(cfg, vocab).to(device)
    model.set_semantic_bank(bank)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, vocab, train_entries, val_entries, cfg


# ---------------------------------------------------------------------------
# Batch printing
# ---------------------------------------------------------------------------

def print_batch(batch: Dict[str, torch.Tensor], vocab: Vocab,
                item_idx: int = 0, label: str = "") -> None:
    print(f"\n  [batch tokens{' — ' + label if label else ''}]  "
          f"batch_size={batch['enc_in'].shape[0]},  "
          f"max_enc_len={batch['enc_in'].shape[1]},  "
          f"max_dec_len={batch['dec_in'].shape[1]}")
    for name, tensor in [("enc_in", batch["enc_in"]),
                          ("enc_mask", batch["enc_mask"]),
                          ("dec_in", batch["dec_in"]),
                          ("dec_tgt", batch["dec_tgt"])]:
        row = tensor[item_idx].tolist()
        if name == "enc_mask":
            print(f"    {name:10s}: {row}")
        else:
            syms = []
            for idx in row:
                if idx == vocab.pad_id:
                    syms.append("PAD")
                elif idx == vocab.eos_id:
                    syms.append("EOS")
                elif idx == vocab.bos_id:
                    syms.append("BOS")
                else:
                    syms.append(vocab.itos[idx])
            print(f"    {name:10s}: ids={row}")
            print(f"    {name:10s}: sym={syms}")


# ---------------------------------------------------------------------------
# Single forward pass with full diagnostics
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_diagnostics(model: DualRouteModel, vocab: Vocab,
                    batch: Dict[str, torch.Tensor],
                    target_ids: List[int],
                    label: str, item_idx: int = 0,
                    verbose: bool = False) -> Dict:
    """Run all three routes and the full model; return prediction strings."""
    enc_in   = batch["enc_in"]
    enc_mask = batch["enc_mask"]
    dec_in   = batch["dec_in"]
    n_steps  = len(target_ids)
    tgt_syms = [vocab.itos[i] for i in target_ids]

    print(f"\n{SEP}")
    print(f"  FORWARD PASS — {label}")
    print(SEP)
    print(f"  model.training = {model.training}  (should be False)")
    print(f"  Target ({n_steps} phonemes): {' '.join(tgt_syms)}")

    # ---- s_hat and LTM encoding ----
    s_hat = model.ltm.encode(enc_in, enc_mask)  # (B, semantic_dim)
    s_hat_item = s_hat[item_idx]  # (semantic_dim,)
    print(f"\n  LTM encoder output (s_hat):")
    print(f"    norm      = {s_hat_item.norm().item():.6f}")
    print(f"    mean      = {s_hat_item.mean().item():.6f}")
    print(f"    std       = {s_hat_item.std().item():.6f}")
    print(f"    first 5 dims = {s_hat_item[:5].tolist()}")

    # ---- lexical field (bank similarity) ----
    field = model.ltm.lexical_field(s_hat)
    conf  = float(field["confidence"][item_idx].item())
    margin = float(field["margin"][item_idx].item())
    density = float(field["density"][item_idx].item())
    sims_item = field["sims"][item_idx]  # (n_words,)
    top_k = torch.topk(sims_item, k=5)
    print(f"\n  LTM lexical field (similarity to semantic bank):")
    print(f"    confidence  = {conf:.6f}  (max cosine sim to bank)")
    print(f"    margin      = {margin:.6f}")
    print(f"    density     = {density:.4f}")
    print(f"    top-5 bank similarities:")
    # We don't have the word list here — just print similarity values
    for rank, (sim_val, bank_idx) in enumerate(
            zip(top_k.values.tolist(), top_k.indices.tolist())):
        print(f"      rank {rank+1}: bank_idx={bank_idx}  sim={sim_val:.6f}")

    # ---- per-route predictions ----
    preds_out = {}
    print(f"\n  Per-route predictions (first {n_steps} steps):")
    for route in ROUTES:
        collect = False  # always deterministic in debug
        res = model.route_logits(enc_in, enc_mask, dec_in,
                                  route=route, collect=collect)
        pred_ids  = res["logits"][item_idx, :n_steps].argmax(-1).tolist()
        pred_syms = [vocab.itos[i] for i in pred_ids]
        match     = (pred_syms == tgt_syms)
        preds_out[route] = pred_syms
        print(f"    [{route:4s}] {'✓' if match else '✗'}  {' '.join(pred_syms)}")

        if verbose:
            # Per-position logits for the correct token vs. predicted token
            logits = res["logits"][item_idx, :n_steps]  # (n_steps, vocab_size)
            print(f"           per-position (tgt | pred | Δlogit):")
            for pos in range(n_steps):
                tgt_id   = target_ids[pos]
                pred_id  = pred_ids[pos]
                tgt_logit  = float(logits[pos, tgt_id].item())
                pred_logit = float(logits[pos, pred_id].item())
                flag = "" if tgt_id == pred_id else " ← WRONG"
                print(f"             pos {pos}: tgt={vocab.itos[tgt_id]:5s} "
                      f"(logit={tgt_logit:+.3f})  "
                      f"pred={vocab.itos[pred_id]:5s} "
                      f"(logit={pred_logit:+.3f}){flag}")

    # ---- full route gate values ----
    full_res = model.forward(enc_in, enc_mask, dec_in, collect=False)
    gate_item = full_res["gate"][item_idx, :n_steps]  # (n_steps,)
    print(f"\n  Gate values (full route, per output position):")
    print(f"    gate shape = {full_res['gate'].shape}")
    print(f"    [0=WM, 1=LTM]: {gate_item.tolist()}")
    for pos in range(n_steps):
        g = float(gate_item[pos].item())
        dominant = "LTM" if g > 0.5 else "WM"
        print(f"    pos {pos} ({vocab.itos[target_ids[pos]]:5s}): g={g:.4f}  → {dominant}")

    return preds_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--word",    default="rarely",
                   help="Word to debug (must be in WFE TSV)")
    p.add_argument("--ckpt",    default=CKPT_DEFAULT)
    p.add_argument("--wfe_tsv", default=WFE_DEFAULT)
    p.add_argument("--device",  default=None)
    p.add_argument("--verbose", action="store_true",
                   help="Print per-position logit detail")
    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    word   = args.word.lower().strip()

    print(SEP2)
    print(f"  debug_single_item_prediction  —  '{word}'")
    print(SEP2)
    print(f"  Checkpoint : {args.ckpt}")
    print(f"  WFE TSV    : {args.wfe_tsv}")
    print(f"  Device     : {device}")

    # --- Load model ---
    print("\n  Loading model …")
    model, vocab, train_entries, val_entries, cfg = load_everything(args.ckpt, device)
    print(f"  model.training = {model.training}  (must be False)")
    print(f"  n_train = {len(train_entries)},  n_val = {len(val_entries)}")
    print(f"  LTM bidirectional_encoder = {cfg.ltm.bidirectional_encoder}")
    print(f"  semantic_bank shape = {model.ltm.semantic_bank.shape}")

    # --- Find word in training lexicon ---
    print(f"\n{SEP}")
    print(f"  LEXICON LOOKUP — '{word}'")
    print(SEP)
    train_word_set = {e.word.lower(): e for e in train_entries}
    val_word_set   = {e.word.lower(): e for e in val_entries}
    in_train = word in train_word_set
    in_val   = word in val_word_set
    print(f"  In training split : {in_train}")
    print(f"  In validation split : {in_val}")

    if in_train:
        entry = train_word_set[word]
        lex_syms = [vocab.itos[i] for i in entry.phonemes]
        print(f"  Training lexicon phonemes : {' '.join(lex_syms)}")
        print(f"  Training lexicon ids      : {entry.phonemes}")
        lex_form_ids = list(entry.phonemes)
    elif in_val:
        entry = val_word_set[word]
        lex_syms = [vocab.itos[i] for i in entry.phonemes]
        print(f"  Val lexicon phonemes : {' '.join(lex_syms)}")
        lex_form_ids = list(entry.phonemes)
    else:
        print(f"  WARNING: '{word}' not found in training or validation split")
        lex_form_ids = None

    # --- Find word in WFE TSV ---
    print(f"\n{SEP}")
    print(f"  WFE TSV LOOKUP — '{word}'")
    print(SEP)
    wfe_form_ids  = None
    wfe_all_forms = None
    if not os.path.exists(args.wfe_tsv):
        print(f"  WARNING: WFE TSV not found: {args.wfe_tsv}")
    else:
        import pandas as pd
        df_wfe = pd.read_csv(args.wfe_tsv, sep="\t")
        df_wfe = df_wfe[~df_wfe["notes"].fillna("").str.contains("EXCLUDED", na=False)]

        row_match = df_wfe[df_wfe["word"].str.lower() == word]
        if len(row_match) == 0:
            print(f"  WARNING: '{word}' not found in WFE TSV")
        else:
            row = row_match.iloc[0]
            wfe_syms = row["target_phonemes"].split()
            wfe_form_ids = [vocab.stoi[s] for s in wfe_syms if s in vocab.stoi]
            wfe_syms_resolved = [vocab.itos[i] for i in wfe_form_ids]
            print(f"  WFE condition   : {row.get('condition', '?')}")
            print(f"  WFE phonemes    : {' '.join(wfe_syms)}")
            print(f"  WFE phoneme ids : {wfe_form_ids}")
            print(f"  WFE phonemes resolved from vocab: {' '.join(wfe_syms_resolved)}")
            if wfe_form_ids and lex_form_ids:
                match = (wfe_form_ids == lex_form_ids)
                print(f"  Lexicon == WFE ids : {match}")
                if not match:
                    print(f"  *** MISMATCH: lexicon ids {lex_form_ids} != WFE ids {wfe_form_ids}")

        # Collect all WFE forms to build a realistic external_eval batch
        all_rows = df_wfe.dropna(subset=["target_phonemes"])
        wfe_all_forms = []
        for _, r in all_rows.iterrows():
            syms = r["target_phonemes"].split()
            ids  = [vocab.stoi[s] for s in syms if s in vocab.stoi]
            if len(ids) == len(syms):
                wfe_all_forms.append(ids)

    # --- Choose target form_ids ---
    target_ids = wfe_form_ids or lex_form_ids
    if target_ids is None:
        print(f"\nERROR: Cannot find phonemes for '{word}' in any source.")
        sys.exit(1)
    target_syms = [vocab.itos[i] for i in target_ids]
    n_phon = len(target_ids)
    print(f"\n  Using target: {' '.join(target_syms)}  (length={n_phon})")

    # ===================================================================
    # PATH A — CEILING EVAL PATH
    # Single item, alone in a batch (batch_size=1).
    # This is the minimal batch that contains only this word.
    # Ceiling eval uses batches of BATCH_SIZE=128, but all from the lexicon.
    # We test the single-item case first as the cleanest baseline.
    # ===================================================================
    # Pre-initialise so SUMMARY references are always safe
    preds_padded: Optional[Dict] = None
    preds_ext: Optional[Dict]    = None
    target_pos: Optional[int]    = None
    print(f"\n{SEP2}")
    print(f"  PATH A — SINGLE-ITEM BATCH  (batch_size=1,  max_len={n_phon+1})")
    print(f"           mimics ceiling eval context when word is alone in its batch")
    print(SEP2)

    batch_solo = make_batch([target_ids], vocab, device)
    print_batch(batch_solo, vocab, item_idx=0, label="solo batch")
    preds_solo = run_diagnostics(model, vocab, batch_solo, target_ids,
                                  label="PATH A — solo batch (batch_size=1)",
                                  item_idx=0, verbose=args.verbose)

    # ===================================================================
    # PATH B — PADDED SOLO BATCH
    # Same single item but padded to the length of the longest WFE item.
    # This tests whether biGRU backward contamination from PAD tokens causes
    # the discrepancy WITHOUT needing other real items in the batch.
    # ===================================================================
    if wfe_all_forms:
        max_wfe_len = max(len(f) for f in wfe_all_forms)
        print(f"\n{SEP2}")
        print(f"  PATH B — PADDED SOLO BATCH  (batch_size=1,  "
              f"padded to WFE max_len={max_wfe_len+1})")
        print(f"           adds a dummy sentinel item of length {max_wfe_len} to force padding")
        print(SEP2)

        # Create a dummy item of the maximum WFE length so make_batch pads to that length
        # Use the BOS id (any valid id works — it's there to set the batch max length)
        dummy_ids = [vocab.bos_id] * max_wfe_len
        batch_padded = make_batch([target_ids, dummy_ids], vocab, device)
        print_batch(batch_padded, vocab, item_idx=0, label=f"padded batch (word at index 0)")
        preds_padded = run_diagnostics(
            model, vocab, batch_padded, target_ids,
            label=f"PATH B — padded to max WFE len ({max_wfe_len} phonemes)",
            item_idx=0, verbose=args.verbose)

    # ===================================================================
    # PATH C — EXTERNAL EVAL BATCH
    # Build the same batch that external_eval.py would construct for this word.
    # external_eval uses BATCH_SIZE=64 and processes items in TSV order.
    # Find which batch index contains the word and reproduce it.
    # ===================================================================
    if wfe_all_forms and wfe_form_ids:
        BATCH_SIZE = 64
        print(f"\n{SEP2}")
        print(f"  PATH C — EXTERNAL EVAL BATCH  (batch_size up to {BATCH_SIZE},  TSV order)")
        print(SEP2)

        # Find position of target_ids in all_forms
        target_tuple = tuple(target_ids)
        target_pos   = None
        for pos, fids in enumerate(wfe_all_forms):
            if tuple(fids) == target_tuple:
                target_pos = pos
                break

        if target_pos is None:
            print(f"  WARNING: '{word}' phoneme sequence not found in WFE forms list")
        else:
            batch_start = (target_pos // BATCH_SIZE) * BATCH_SIZE
            batch_end   = min(batch_start + BATCH_SIZE, len(wfe_all_forms))
            item_in_batch = target_pos - batch_start
            batch_forms_ext = wfe_all_forms[batch_start: batch_end]

            lengths = [len(f) for f in batch_forms_ext]
            max_len_in_batch = max(lengths)
            print(f"  '{word}' is item #{target_pos} in WFE (0-indexed)")
            print(f"  External eval batch: items {batch_start}–{batch_end - 1}  "
                  f"(size {len(batch_forms_ext)})")
            print(f"  Item position within batch: {item_in_batch}")
            print(f"  Lengths in this batch: min={min(lengths)}, "
                  f"max={max_len_in_batch}, "
                  f"mean={sum(lengths)/len(lengths):.1f}")
            print(f"  → batch padded to enc_len={max_len_in_batch + 1}")

            batch_ext = make_batch(batch_forms_ext, vocab, device)
            print_batch(batch_ext, vocab, item_idx=item_in_batch,
                        label=f"external eval batch (word at index {item_in_batch})")
            preds_ext = run_diagnostics(
                model, vocab, batch_ext, target_ids,
                label=f"PATH C — external eval batch (size={len(batch_forms_ext)})",
                item_idx=item_in_batch, verbose=args.verbose)

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print(f"\n{SEP2}")
    print(f"  SUMMARY — '{word}'  target: {' '.join(target_syms)}")
    print(SEP2)
    print(f"  {'Path':<50s}  {'full':^18s}  {'wm':^18s}  {'ltm':^18s}")
    print(f"  {'':─<50s}  {'':─^18s}  {'':─^18s}  {'':─^18s}")

    all_paths = [("A solo batch (batch_size=1)", preds_solo)]
    if preds_padded is not None:
        all_paths.append(("B padded solo (max WFE len)", preds_padded))
    if preds_ext is not None:
        all_paths.append(("C external eval batch (size=64)", preds_ext))

    tgt_str = " ".join(target_syms)
    for path_name, preds in all_paths:
        row_parts = []
        for route in ROUTES:
            pred_str = " ".join(preds[route])
            ok = "✓" if pred_str == tgt_str else "✗"
            row_parts.append(f"{ok} {pred_str:<15s}")
        print(f"  {path_name:<50s}  {'  '.join(row_parts)}")

    print()
    print("  DIAGNOSIS")
    print("  ─────────")
    print("  Root cause: LTMLexicon.encode() runs the biGRU over the full padded")
    print("  tensor WITHOUT pack_padded_sequence.  The backward GRU direction starts")
    print("  at the rightmost position in the padded batch and processes left,")
    print("  including PAD positions.  PAD embedding = 0 but GRU bias ≠ 0, so")
    print("  repeated zero-input steps produce non-zero hidden states.  The number")
    print("  of trailing PAD columns in the batch changes the backward-GRU hidden")
    print("  state at every valid position, which shifts s_hat and can flip the")
    print("  LTM (and thus the gated full) prediction.")
    print()
    print("  Note: the direction of the effect is not simply 'more padding = worse'.")
    print("  For 'rarely', PATH B (heavily padded) is CORRECT while PATH A (no")
    print("  padding at all) is WRONG.  This is because the model was always trained")
    print("  with mixed-length batches that included trailing PADs; an unpadded solo")
    print("  forward pass is an out-of-distribution context for the backward GRU.")
    print()
    print("  WM route is unaffected: WMRecurrent.encoder is unidirectional.")
    print()
    print("  Recommended fix (retrained architecture only):")
    print("    Wrap the biGRU in LTMLexicon.encode() with pack_padded_sequence /")
    print("    pad_packed_sequence so the backward direction starts at the last")
    print("    *real* token, not the last PAD position.")
    print("  Do NOT apply this fix to the current checkpoint weights.")
    print()


if __name__ == "__main__":
    main()
