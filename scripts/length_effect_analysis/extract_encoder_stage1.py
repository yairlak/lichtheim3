"""Encoder-only representation extraction (stage 1) + strict equivalence validation.

WHAT THIS IS
------------
This is a **new diagnostic execution**: the four frozen checkpoints are loaded and
their *encoders* are run over the same 1,200 WFE items in the same canonical
order.  It is **encoder-only representation extraction with no decoder
execution** — it is not "no inference".  No token is generated, no decoder GRU
step is taken, no motor projection is applied.

WHY
---
`run_instrumented.py` persisted `wm_encoder_hidden`, `s_hat`,
`ltm_decoder_h0` and the two gold-prefix premotor tensors, but never the LTM
encoder's last hidden state `pooled = h[-1]` that `to_semantic` consumes
(models/ltm_route.py:139-146).  That tensor is stage 1 of the M4 chain and is not
recoverable from anything saved: `to_semantic` is `Linear(128,128) -> GELU ->
Linear(128,300)`, which is neither dimension-preserving nor invertible.

IDENTITY PROOF (not an approximation)
-------------------------------------
The value written to disk is the tensor captured by a `forward_pre_hook` on
`model.ltm.to_semantic[0]` — literally the input to the first layer of the
projection producing s_hat — recorded while the canonical
`LTMLexicon.encode()` method runs.  It is then asserted **bit-identical** to an
independent re-derivation of `h[-1]` from `pack_padded_sequence` +
`self.encoder`.  Both must agree exactly or the run fails closed.

DECODER GUARD
-------------
Before any extraction, every decoder / motor / route-generation entry point is
replaced by a sentinel that appends to a violation log and raises.  The run
asserts the log is empty afterwards; the guard coverage is written to provenance.

Nothing under models/, config.py, training code, scripts/external_eval.py or any
checkpoint file is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluate.hooks import make_batch                                  # noqa: E402
from scripts.external_eval import load_model_and_vocab                 # noqa: E402
from scripts.length_effect_analysis.run_instrumented import (          # noqa: E402
    BUNDLE, CKPTS, WFE, load_items, sha)

BATCH_SIZE = 64                 # identical to run_instrumented / external_eval
RTOL = 1e-6                     # predeclared, Phase B
ATOL = 1e-7                     # predeclared, Phase B

EXECUTION_TYPE = ("encoder-only representation extraction with no decoder "
                  "execution")

# Attribute paths that must never be invoked during extraction.  Every one of
# these is a decoder, a motor readout or a full-route generation entry point.
GUARDED = (
    "ltm.decoder", "ltm.decode", "ltm.decode_from_s_hat", "ltm.dec_to_premotor",
    "wm.decoder", "wm.decode_from_state", "wm.to_premotor",
    "motor", "gate",
)
GUARDED_MODEL_METHODS = ("forward", "route_logits", "decode_from_states")


def git(*a):
    return subprocess.run(("git",) + a, cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


# ------------------------------------------------------------------ guard

class DecoderGuard:
    """Replace every decoder/motor entry point with a raising sentinel."""

    def __init__(self, model):
        self.model = model
        self.violations: List[str] = []
        self.installed: List[str] = []
        self._restore = []

    def _sentinel(self, name):
        def _f(*a, **k):
            self.violations.append(name)
            raise RuntimeError(
                f"decoder guard violated: {name} was called during "
                f"encoder-only extraction")
        return _f

    def __enter__(self):
        for path in GUARDED:
            obj, attr = self.model, None
            parts = path.split(".")
            for p in parts[:-1]:
                obj = getattr(obj, p)
            attr = parts[-1]
            target = getattr(obj, attr)
            if isinstance(target, nn.Module):
                # guard the module call itself, not the attribute binding
                orig = target.forward
                self._restore.append((target, "forward", orig))
                target.forward = self._sentinel(path + ".forward")
            else:
                self._restore.append((obj, attr, target))
                setattr(obj, attr, self._sentinel(path))
            self.installed.append(path)
        for m in GUARDED_MODEL_METHODS:
            if hasattr(self.model, m):
                orig = getattr(self.model, m)
                self._restore.append((self.model, m, orig))
                setattr(self.model, m, self._sentinel("model." + m))
                self.installed.append("model." + m)
        return self

    def __exit__(self, *exc):
        for obj, attr, orig in reversed(self._restore):
            setattr(obj, attr, orig)
        self._restore = []
        return False


# ------------------------------------------------------- encoder extraction

def ltm_encoder_last_hidden(ltm, enc_in, enc_mask):
    """Independent re-derivation of `pooled = h[-1]` (ltm_route.py:136-140).

    Reproduces the `unigru_last_hidden` branch of `LTMLexicon.encode` exactly:
    embed -> pack_padded_sequence(enforce_sorted=False) -> GRU -> last layer of
    the final hidden state.  Noise is inactive (eval mode, ventral_noise = 0).
    """
    assert ltm.cfg.ltm_encoder_mode == "unigru_last_hidden", ltm.cfg.ltm_encoder_mode
    emb = ltm.phon_embed(enc_in)
    lengths = enc_mask.sum(1).clamp(min=1).cpu()
    packed = nn.utils.rnn.pack_padded_sequence(
        emb, lengths, batch_first=True, enforce_sorted=False)
    _, h = ltm.encoder(packed)          # (num_layers, B, H)
    return h[-1]                        # (B, H)


def architecture_assertions(model) -> Dict[str, object]:
    """Structural facts that must hold for the extraction to be meaningful."""
    enc = model.ltm.encoder
    assert isinstance(enc, nn.GRU)
    assert enc.bidirectional is False, "LTM encoder is bidirectional"
    assert enc.num_layers == 1, enc.num_layers
    assert enc.hidden_size == 128, enc.hidden_size
    rev = [k for k in enc.state_dict() if "_reverse" in k]
    assert not rev, f"reverse-direction GRU parameters present: {rev}"
    ts = model.ltm.to_semantic
    assert isinstance(ts, nn.Sequential) and isinstance(ts[0], nn.Linear)
    assert ts[0].in_features == enc.hidden_size, (ts[0].in_features,
                                                  enc.hidden_size)
    assert model.ltm.cfg.ventral_noise == 0.0, model.ltm.cfg.ventral_noise
    assert model.wm.cfg.interference_noise == 0.0, model.wm.cfg.interference_noise
    assert not model.training, "model must be in eval mode"
    return {
        "ltm_encoder_type": type(enc).__name__,
        "ltm_encoder_bidirectional": bool(enc.bidirectional),
        "ltm_encoder_num_layers": int(enc.num_layers),
        "ltm_encoder_hidden_size": int(enc.hidden_size),
        "ltm_encoder_reverse_parameters": rev,
        "to_semantic_first_layer": f"Linear({ts[0].in_features},{ts[0].out_features})",
        "to_semantic_last_layer": f"Linear({ts[-1].in_features},{ts[-1].out_features})",
        "ltm_ventral_noise": float(model.ltm.cfg.ventral_noise),
        "wm_interference_noise": float(model.wm.cfg.interference_noise),
        "ltm_encoder_mode": model.ltm.cfg.ltm_encoder_mode,
    }


def extract_batch(model, batch) -> Dict[str, torch.Tensor]:
    """One encoder-only batch.  Returns the four stage-1 quantities.

    `ltm_encoder_hidden` is the tensor a forward_pre_hook captures at the input
    of `to_semantic[0]` while the canonical `ltm.encode()` runs; it is asserted
    bit-identical to the independent re-derivation.
    """
    captured: List[torch.Tensor] = []

    def _pre(_mod, inp):
        captured.append(inp[0].detach().clone())

    h_seq = model.ltm.to_semantic.register_forward_pre_hook(_pre)
    h_lin = model.ltm.to_semantic[0].register_forward_pre_hook(_pre)
    try:
        s_hat = model.ltm.encode(batch["enc_in"], batch["enc_mask"])
    finally:
        h_seq.remove()
        h_lin.remove()

    assert len(captured) == 2, f"expected 2 hook captures, got {len(captured)}"
    seq_in, lin_in = captured
    assert torch.equal(seq_in, lin_in), (
        "input to to_semantic differs from input to to_semantic[0]")

    independent = ltm_encoder_last_hidden(model.ltm, batch["enc_in"],
                                          batch["enc_mask"])
    assert torch.equal(lin_in, independent), (
        "captured to_semantic input is not bit-identical to the independently "
        "re-derived GRU h[-1]")

    ltm_h0 = torch.tanh(model.ltm.sem_to_h0(s_hat))
    h_wm = model.wm.encode(batch["enc_in"], batch["enc_mask"]).squeeze(0)
    return {"ltm_encoder_hidden": lin_in, "s_hat": s_hat,
            "ltm_decoder_h0": ltm_h0, "wm_encoder_hidden": h_wm}


def run_seed(seed: int, device: str = "cpu"):
    rel, epoch, exp_sha = CKPTS[seed]
    ck_path = os.path.join(ROOT, BUNDLE, rel)
    assert sha(ck_path) == exp_sha, f"checkpoint SHA mismatch for seed {seed}"
    model, vocab, _ = load_model_and_vocab(ck_path, device)
    model.eval()
    arch = architecture_assertions(model)
    df, forms = load_items(vocab)

    acc: Dict[str, List[np.ndarray]] = {k: [] for k in
                                        ("ltm_encoder_hidden", "s_hat",
                                         "ltm_decoder_h0", "wm_encoder_hidden")}
    guard = DecoderGuard(model)
    with guard, torch.inference_mode():
        for start in range(0, len(forms), BATCH_SIZE):
            batch = make_batch(forms[start:start + BATCH_SIZE], vocab, device)
            out = extract_batch(model, batch)
            for k, v in out.items():
                acc[k].append(v.cpu().numpy().copy())
    assert not guard.violations, guard.violations
    arrays = {k: np.concatenate(v) for k, v in acc.items()}
    return df, arrays, arch, guard, epoch, exp_sha


# ------------------------------------------------------------- validation

def compare(name: str, seed: int, new: np.ndarray, old: np.ndarray,
            expected_shape) -> dict:
    shape_ok = (tuple(new.shape) == tuple(old.shape) == tuple(expected_shape))
    n = int(min(new.size, old.size))
    if shape_ok:
        d = np.abs(new.astype(np.float64) - old.astype(np.float64))
        row = {"exact_equality_fraction": float(np.mean(new == old)),
               "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
               "allclose": bool(np.allclose(new, old, rtol=RTOL, atol=ATOL))}
    else:
        row = {"exact_equality_fraction": float("nan"),
               "max_abs_diff": float("nan"), "mean_abs_diff": float("nan"),
               "allclose": False}
    return {"seed": seed, "tensor": name,
            "expected_shape": str(tuple(expected_shape)),
            "observed_shape_recomputed": str(tuple(new.shape)),
            "observed_shape_saved": str(tuple(old.shape)),
            "shape_match": shape_ok, "dtype_recomputed": str(new.dtype),
            "dtype_saved": str(old.dtype), "n_values_compared": n,
            "rtol": RTOL, "atol": ATOL, **row}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="19,20,21,22")
    ap.add_argument("--out_dir", default=os.path.join(
        ROOT, "outputs/length_effect_mechanism_93a577f/instrumented/"
              "stage1_encoder_extraction"))
    ap.add_argument("--instrumented_dir", default=os.path.join(
        ROOT, "outputs/length_effect_mechanism_93a577f/instrumented"))
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    saved = np.load(os.path.join(args.instrumented_dir, "representations.npz"))
    item_summary = pd.read_csv(os.path.join(args.instrumented_dir,
                                            "item_summary.tsv"), sep="\t")

    idx_rows, val_rows, order_rows = [], [], []
    recomputed, arch_all, guard_all = {}, {}, {}
    for seed in seeds:
        df, arrays, arch, guard, epoch, exp_sha = run_seed(seed)
        arch_all[str(seed)] = arch
        guard_all[str(seed)] = {"guarded_entry_points": guard.installed,
                                "violations": guard.violations}
        np.save(os.path.join(args.out_dir, f"ltm_encoder_hidden_seed{seed}.npy"),
                arrays["ltm_encoder_hidden"])
        for k in ("s_hat", "ltm_decoder_h0", "wm_encoder_hidden",
                  "ltm_encoder_hidden"):
            recomputed[f"{k}_seed{seed}"] = arrays[k]

        sub = item_summary[item_summary["seed"] == seed].reset_index(drop=True)
        assert len(sub) == len(df) == 1200, (len(sub), len(df))
        same_order = list(sub["item_id"]) == list(df["item_id"])
        order_rows.append({
            "seed": seed, "n_rows_item_summary": len(sub), "n_rows_extraction": len(df),
            "item_order_identical": same_order,
            "n_duplicate_item_ids": int(len(df) - df["item_id"].nunique()),
            "n_missing_vs_item_summary": int(len(set(sub["item_id"])
                                                 - set(df["item_id"]))),
        })
        for r, (_, row) in enumerate(sub.iterrows()):
            idx_rows.append({"seed": seed, "row_index": r,
                             "item_id": row["item_id"],
                             "exposure_status": row["exposure_status"],
                             "phoneme_length": int(row["phoneme_length"]),
                             "checkpoint_sha256": exp_sha})

        for name, shp in (("s_hat", (1200, 300)),
                          ("ltm_decoder_h0", (1200, 128)),
                          ("wm_encoder_hidden", (1200, 128))):
            key = {"s_hat": "s_hat", "ltm_decoder_h0": "ltm_decoder_h0",
                   "wm_encoder_hidden": "wm_encoder_hidden"}[name]
            val_rows.append(compare(name, seed, arrays[name],
                                    saved[f"{key}_seed{seed}"], shp))
        e = arrays["ltm_encoder_hidden"]
        val_rows.append({
            "seed": seed, "tensor": "ltm_encoder_hidden",
            "expected_shape": "(1200, 128)",
            "observed_shape_recomputed": str(tuple(e.shape)),
            "observed_shape_saved": "ABSENT (never persisted)",
            "shape_match": tuple(e.shape) == (1200, 128),
            "dtype_recomputed": str(e.dtype), "dtype_saved": "",
            "n_values_compared": 0, "rtol": RTOL, "atol": ATOL,
            "exact_equality_fraction": float("nan"),
            "max_abs_diff": float("nan"), "mean_abs_diff": float("nan"),
            "allclose": True})
        print(f"[seed {seed}] extracted  ({time.time()-t0:.1f}s)")

    np.savez_compressed(os.path.join(args.out_dir,
                                     "recomputed_encoder_quantities.npz"),
                        **recomputed)
    idx = pd.DataFrame(idx_rows)
    idx.to_csv(os.path.join(args.out_dir, "item_index.tsv"), sep="\t", index=False)
    val = pd.DataFrame(val_rows)
    val.to_csv(os.path.join(args.out_dir, "extraction_validation.tsv"),
               sep="\t", index=False)
    order = pd.DataFrame(order_rows)

    # ------------------------------------------------ pass condition (B2)
    enc_ok = all(tuple(recomputed[f"ltm_encoder_hidden_seed{s}"].shape)
                 == (1200, 128) for s in seeds)
    order_ok = bool(order["item_order_identical"].all()
                    and (order["n_duplicate_item_ids"] == 0).all()
                    and (order["n_missing_vs_item_summary"] == 0).all()
                    and (order["n_rows_extraction"] == 1200).all())
    cmp_rows = val[val["tensor"] != "ltm_encoder_hidden"]
    tol_ok = bool(cmp_rows["allclose"].all() and cmp_rows["shape_match"].all())
    seeds_ok = sorted(seeds) == [19, 20, 21, 22]
    guard_ok = all(not g["violations"] for g in guard_all.values())
    verdict = "PASS" if (enc_ok and order_ok and tol_ok and seeds_ok
                         and guard_ok) else "FAIL"

    elapsed = round(time.time() - t0, 1)
    prov = {
        "phase": "M4 Phase A - stage-1 encoder extraction",
        "execution_type": EXECUTION_TYPE,
        "decoder_executed": False,
        "tokens_generated": False,
        "autoregressive_decoding": False,
        "gold_prefix_decoding": False,
        "motor_projection_applied": False,
        "training_performed": False,
        "weights_modified": False,
        "architecture_changed": False,
        "checkpoint_training_commit": "93a577fd9822955fa272ee733fa7e2acf81f1333",
        "behavioral_evaluation_commit": "e876b755d0475ed11e5fbc0419a0bd8860dfd325",
        "repository_head": git("rev-parse", "HEAD"),
        "repository_dirty": bool(git("status", "--porcelain").strip()),
        "checkpoints": {str(s): {"path": os.path.join(BUNDLE, CKPTS[s][0]),
                                 "sha256": CKPTS[s][2], "seed": s,
                                 "epoch": CKPTS[s][1]} for s in seeds},
        "dataset_hashes": {"wfe_tsv": sha(os.path.join(ROOT, WFE))},
        "source_representations_npz_sha256": sha(
            os.path.join(args.instrumented_dir, "representations.npz")),
        "source_item_summary_sha256": sha(
            os.path.join(args.instrumented_dir, "item_summary.tsv")),
        "script_sha256": {p: sha(os.path.join(
            ROOT, "scripts/length_effect_analysis", p))
            for p in ("extract_encoder_stage1.py", "run_instrumented.py",
                      "instrument.py")},
        "batch_size": BATCH_SIZE,
        "device": "cpu",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "inference_mode": True,
        "model_eval_mode": True,
        "noise_settings": {"wm_interference_noise": 0.0,
                           "ltm_ventral_noise": 0.0, "apply_noise": False},
        "architecture_assertions": arch_all,
        "decoder_guard": guard_all,
        "identity_proof": (
            "value written = forward_pre_hook capture at the input of "
            "to_semantic[0] during the canonical LTMLexicon.encode(); asserted "
            "bit-identical (torch.equal) to an independent re-derivation of "
            "pack_padded_sequence -> GRU -> h[-1]"),
        "tolerances": {"rtol": RTOL, "atol": ATOL,
                       "predeclared": True,
                       "changed_after_seeing_results": False},
        "encoder_extraction_equivalence": verdict,
        "elapsed_seconds": elapsed,
        "seeds": seeds,
        "n_items_per_seed": 1200,
    }
    with open(os.path.join(args.out_dir, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
        f.write("\n")

    _write_md(os.path.join(args.out_dir, "extraction_validation.md"), val,
              order, verdict, elapsed, seeds, guard_all, arch_all)
    print(f"\nENCODER_EXTRACTION_EQUIVALENCE = {verdict}  ({elapsed}s)")
    return 0 if verdict == "PASS" else 1


def _write_md(path, val, order, verdict, elapsed, seeds, guard_all, arch_all):
    cmp_rows = val[val["tensor"] != "ltm_encoder_hidden"]
    L = []
    L.append("# Stage-1 encoder extraction — validation report\n")
    L.append(f"**Execution type.** {EXECUTION_TYPE}. Four frozen checkpoints "
             "were loaded and their encoders run over the same 1,200 WFE items "
             "in the canonical order. No decoder GRU step, no motor "
             "projection, no token generation, no autoregressive loop, no "
             "gold-prefix decoding. This is a new diagnostic execution, not a "
             "re-read of stored arrays.\n")
    L.append(f"**Runtime.** {elapsed} s (4 checkpoints x 1,200 items, CPU).\n")
    L.append("## Identity of the extracted tensor\n")
    L.append("The value written to `ltm_encoder_hidden_seed*.npy` is captured "
             "by a `forward_pre_hook` at the **input of "
             "`model.ltm.to_semantic[0]`** — the first layer of the projection "
             "producing `s_hat` — while the canonical `LTMLexicon.encode()` "
             "runs. It is asserted `torch.equal` to an independent "
             "re-derivation of `pack_padded_sequence -> self.encoder -> h[-1]` "
             "(models/ltm_route.py:136-146). Every batch of every seed passed "
             "both assertions; a single mismatch would have aborted the run.\n")
    L.append("Structural assertions verified per seed: LTM encoder is a "
             "1-layer **unidirectional** `nn.GRU`, hidden size 128, with **no** "
             "`_reverse` parameters; `to_semantic[0]` is "
             f"`{arch_all[str(seeds[0])]['to_semantic_first_layer']}`; ventral "
             "and interference noise are both 0.0; model in eval mode.\n")
    L.append("## Decoder guard\n")
    inst = guard_all[str(seeds[0])]["guarded_entry_points"]
    L.append("Before extraction, every decoder / motor / route-generation entry "
             "point was replaced by a raising sentinel:\n")
    L.append("".join(f"- `{p}`\n" for p in inst))
    L.append(f"\nViolations recorded across all four seeds: "
             f"**{sum(len(g['violations']) for g in guard_all.values())}**.\n")
    L.append("\n## Equivalence against the canonical saved arrays\n")
    L.append(f"Predeclared tolerances: `rtol = {RTOL}`, `atol = {ATOL}` "
             "(fixed before the run; not changed after seeing results).\n")
    L.append("\n| seed | tensor | shape ok | dtype | n values | exact-equal "
             "fraction | max abs diff | mean abs diff | allclose |\n")
    L.append("|---|---|---|---|---|---|---|---|---|\n")
    for _, r in cmp_rows.iterrows():
        L.append(f"| {r['seed']} | `{r['tensor']}` | {r['shape_match']} | "
                 f"{r['dtype_recomputed']} | {r['n_values_compared']:,} | "
                 f"{r['exact_equality_fraction']:.6f} | "
                 f"{r['max_abs_diff']:.3e} | {r['mean_abs_diff']:.3e} | "
                 f"{r['allclose']} |\n")
    L.append("\n### Newly extracted tensor (no saved counterpart)\n")
    for _, r in val[val["tensor"] == "ltm_encoder_hidden"].iterrows():
        L.append(f"- seed {r['seed']}: `ltm_encoder_hidden` "
                 f"{r['observed_shape_recomputed']} {r['dtype_recomputed']} — "
                 "the original `representations.npz` omitted this tensor.\n")
    L.append("\n## Item coverage and order\n")
    L.append("\n| seed | rows | item order identical to `item_summary.tsv` | "
             "duplicate item_ids | missing rows |\n|---|---|---|---|---|\n")
    for _, r in order.iterrows():
        L.append(f"| {r['seed']} | {r['n_rows_extraction']} | "
                 f"{r['item_order_identical']} | {r['n_duplicate_item_ids']} | "
                 f"{r['n_missing_vs_item_summary']} |\n")
    L.append(f"\n## Verdict\n\n```\nENCODER_EXTRACTION_EQUIVALENCE = {verdict}"
             "\n```\n")
    with open(path, "w") as f:
        f.write("".join(L))


if __name__ == "__main__":
    sys.exit(main())
