#!/usr/bin/env python3
"""Per-objective gradient norms and interference at frozen checkpoints.

ANALYSIS ONLY: no `optimizer.step()` is ever called, no weight is updated and
no scientific artifact is written.  The trainer is rebuilt exactly as the run
configured it, each checkpoint is loaded in turn, and gradients are taken on
FIXED deterministic diagnostic batches -- the first `--batches` batches of each
counter-addressed stream, which are a pure function of (stream seed, cursor)
and therefore identical across checkpoints and across runs at the same seed.

Reported per checkpoint:
  * global gradient norm of each separable objective component;
  * pairwise cosine similarities, globally and per parameter group;
  * the summed pre-clip gradient norm, per batch, comparable with the
    `grad_norm` column of losses.tsv;
  * per-group norms, showing how much of each update a component contributes.

Interpretation caveat recorded with the output: cosines between components
that consume DIFFERENT streams (e.g. R alignment vs C retrieval) mix objective
geometry with population differences.  The same-example comparison --
C alignment vs C retrieval on the SAME C batches -- is the clean one.

Usage:
    python scripts/naming_comprehension/grad_interference_audit.py \
        --run outputs/joint_scratch/final2a_calign_seed22_final_full \
        --json-out reports/final2a/grad_audit.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from losses import alignment_loss, total_loss                            # noqa: E402
from scripts.naming_comprehension.train_joint_scratch import (           # noqa: E402
    FINAL_FULL_MODE, LAMBDA_C, LAMBDA_N, TAU, JointScratchTrainer,
    build_batch,
)
from scripts.naming_comprehension.train_tasks import (                   # noqa: E402
    comprehension_forward, naming_objective, retrieval_loss,
)

# Parameter groups, in ventral-pathway order.
GROUPS: "OrderedDict[str, tuple]" = OrderedDict([
    ("phon_embed", ("phon_embed.",)),
    ("wm", ("wm.",)),
    ("ltm.encoder", ("ltm.encoder.",)),
    ("to_semantic", ("ltm.to_semantic.",)),
    ("sem_to_h0", ("ltm.sem_to_h0.",)),
    ("ltm.decoder", ("ltm.decoder.",)),
    ("dec_to_premotor", ("ltm.dec_to_premotor.",)),
    ("motor", ("motor.",)),
    ("gate", ("gate.",)),
])

KEY_PAIRS = [
    ("R_total", "N_naming"),
    ("R_total", "C_retrieval_raw"),
    ("N_naming", "C_retrieval_raw"),
    ("R_align", "C_retrieval_raw"),
    ("C_align", "C_retrieval_raw"),        # same-example: the clean comparison
    ("R_align", "C_align"),
    ("R_dec", "N_naming"),
]

CROSS_STREAM_NOTE = (
    "Cosines between components consuming DIFFERENT streams (R_* vs C_*/N_*) "
    "mix objective geometry with population differences and are not pure "
    "objective incompatibility. C_align vs C_retrieval is computed on the "
    "SAME C batches and is the clean same-example comparison."
)


def group_of(name: str) -> str:
    for g, prefixes in GROUPS.items():
        if name.startswith(prefixes):
            return g
    return "other"


class Auditor:
    def __init__(self, trainer: JointScratchTrainer, n_batches: int) -> None:
        self.tr = trainer
        self.model = trainer.model
        self.pad_id = trainer.vocab.pad_id
        self.K = n_batches
        self.names = [n for n, _ in self.model.named_parameters()]
        self.group = {n: group_of(n) for n in self.names}
        # FIXED diagnostic batches: cursors 0..K-1 of each stream.
        self.idx = {s: [trainer.streams[s].indices(c) for c in range(n_batches)]
                    for s in ("repetition", "pool", "comprehension", "naming")}

    def _batch(self, stream: str, j: int) -> dict:
        idx = self.idx[stream][j]
        if stream == "pool":
            return build_batch(self.tr.pool_entries, self.tr.pool_bank,
                               self.tr.vocab, idx, self.tr.device)
        return build_batch(self.tr.entries, self.tr.bank_raw, self.tr.vocab,
                           idx, self.tr.device)

    def components(self, j: int) -> Dict[str, torch.Tensor]:
        """Every separable loss component for diagnostic batch set `j`."""
        cfg = self.tr.cfg
        r = self._batch("repetition", j)
        out = self.model(r["enc_in"], r["enc_mask"], r["dec_in"])
        parts = total_loss(out, r, cfg.loss, self.pad_id,
                           usage_prior=cfg.gating.usage_prior)
        p = self._batch("pool", j)
        pout = self.model(p["enc_in"], p["enc_mask"], p["dec_in"])
        V = pout["wm_logits"].shape[-1]
        pool_ce = F.cross_entropy(pout["wm_logits"].reshape(-1, V),
                                  p["dec_tgt"].reshape(-1),
                                  ignore_index=self.pad_id)
        c = self._batch("comprehension", j)
        s_hat = comprehension_forward(self.model, c["enc_in"], c["enc_mask"])
        ret = retrieval_loss(s_hat, self.model.ltm.semantic_bank,
                             c["bank_idx"], TAU)
        c_align = alignment_loss(s_hat, c["semantic"])
        n = self._batch("naming", j)
        nam = naming_objective(self.model, n, self.pad_id)["total"]

        w = self.tr.c_align_weight
        return {
            "R_total": parts["total"],
            "R_rep": parts["rep"],
            "R_align": parts["align"],
            "R_dec": cfg.loss.dec * parts["dec"],
            "pool": cfg.loss.wm * pool_ce,
            "C_retrieval_raw": ret,
            "C_retrieval_weighted": LAMBDA_C * ret,
            "C_align": c_align,
            "C_align_weighted": (w * c_align if w > 0
                                 else torch.zeros((), device=ret.device)),
            "N_naming": LAMBDA_N * nam,
            # exactly the scalar the driver optimizes at this configuration
            "SUMMED": (parts["total"] + cfg.loss.wm * pool_ce
                       + LAMBDA_C * ret + LAMBDA_N * nam
                       + (w * c_align if w > 0 else 0.0)),
        }

    def gradient(self, key: str) -> Dict[str, torch.Tensor]:
        """Gradient of the K-batch mean of one component. No optimizer step."""
        self.model.zero_grad(set_to_none=True)
        total = None
        for j in range(self.K):
            l = self.components(j)[key]
            total = l if total is None else total + l
        (total / self.K).backward()
        g = {n: (p.grad.detach().clone() if p.grad is not None
                 else torch.zeros_like(p))
             for n, p in self.model.named_parameters()}
        self.model.zero_grad(set_to_none=True)
        return g

    def flat(self, g: Dict[str, torch.Tensor],
             group: Optional[str] = None) -> torch.Tensor:
        ts = [g[n].reshape(-1) for n in self.names
              if group is None or self.group[n] == group]
        return torch.cat(ts) if ts else torch.zeros(1)

    def per_batch_summed_norms(self) -> List[float]:
        """Summed pre-clip norm for each single batch (comparable to the
        `grad_norm` column, which is logged per single training step)."""
        out = []
        for j in range(self.K):
            self.model.zero_grad(set_to_none=True)
            self.components(j)["SUMMED"].backward()
            g = torch.cat([(p.grad if p.grad is not None
                            else torch.zeros_like(p)).reshape(-1)
                           for p in self.model.parameters()])
            out.append(float(g.norm()))
            self.model.zero_grad(set_to_none=True)
        return out


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm(), b.norm()
    if float(na) == 0.0 or float(nb) == 0.0:
        return float("nan")
    return float((a @ b) / (na * nb))


def audit_checkpoint(aud: Auditor, ckpt_path: str, clip: float) -> dict:
    ck = torch.load(ckpt_path, map_location=aud.tr.device, weights_only=False)
    aud.tr.load_state_dict(ck, source=os.path.basename(ckpt_path))
    aud.model.train(True)
    step = int(ck["global_step"])

    keys = ["R_total", "R_rep", "R_align", "R_dec", "pool",
            "C_retrieval_raw", "C_retrieval_weighted", "C_align",
            "C_align_weighted", "N_naming", "SUMMED"]
    grads = {k: aud.gradient(k) for k in keys}
    flats = {k: aud.flat(g) for k, g in grads.items()}
    per_batch = aud.per_batch_summed_norms()

    return {
        "checkpoint": os.path.basename(ckpt_path),
        "global_step": step,
        "N_exposures": round(step / 463, 1),
        "c_align_weight": aud.tr.c_align_weight,
        "global_norms": {k: float(v.norm()) for k, v in flats.items()},
        "summed_per_batch_norms": {
            "median": statistics.median(per_batch),
            "min": min(per_batch), "max": max(per_batch),
            "fraction_over_clip": sum(1 for x in per_batch if x > clip) / len(per_batch),
        },
        "cosines_global": {f"{a}~{b}": cosine(flats[a], flats[b])
                           for a, b in KEY_PAIRS},
        "cosines_by_group": {
            g: {f"{a}~{b}": cosine(aud.flat(grads[a], g), aud.flat(grads[b], g))
                for a, b in KEY_PAIRS}
            for g in GROUPS},
        "group_norms": {
            g: {k: float(aud.flat(grads[k], g).norm())
                for k in ("SUMMED", "R_total", "R_align", "C_retrieval_raw",
                          "C_retrieval_weighted", "C_align", "N_naming")}
            for g in GROUPS},
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", required=True,
                   help="run directory (uses its config.json and checkpoints/)")
    p.add_argument("--checkpoints", nargs="*", default=None,
                   help="explicit checkpoint files (default: all in the run)")
    p.add_argument("--batches", type=int, default=8,
                   help="fixed diagnostic batches per stream")
    p.add_argument("--device", default="cpu")
    p.add_argument("--json-out", default=None)
    p.add_argument("--allow-glove-fallback", action="store_true",
                   help="SMOKE ONLY: skip the real-GloVe requirement")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = json.load(open(os.path.join(args.run, "config.json")))
    cks = args.checkpoints or sorted(
        os.path.join(args.run, "checkpoints", f)
        for f in os.listdir(os.path.join(args.run, "checkpoints"))
        if f.startswith("step_") and f.endswith(".pt"))
    if not cks:
        raise SystemExit(f"no checkpoints found under {args.run}")

    tr = JointScratchTrainer(
        regime=cfg.get("regime", "j0"), seed=int(cfg["seed"]),
        device=args.device, max_words=30000,
        lexicon_path=cfg.get("lexicon_path", "data/lexicon_en_glove_covered.tsv"),
        dorsal_pool_size=int(cfg.get("dorsal_pool_size", 4000)),
        batch_size=int(cfg.get("batch_size", 64)),
        subset_mode=cfg.get("subset_mode", FINAL_FULL_MODE),
        subset_per_band=822, subset_size=64,
        lr_boundary_steps=int(cfg.get("lr_boundary_steps", 46300)),
        allow_glove_fallback=args.allow_glove_fallback,
        require_subset_hash=not args.allow_glove_fallback,
        glove_path=("data/glove.6B.300d.txt" if not args.allow_glove_fallback
                    else "_no_such_glove.txt"),
        c_align_weight=float(cfg.get("c_align_weight", 0.0)))
    aud = Auditor(tr, args.batches)
    clip = float(cfg.get("grad_clip", 1.0))

    report = {"analysis_only": True, "optimizer_steps_taken": 0,
              "run_dir": os.path.abspath(args.run),
              "c_align_weight": tr.c_align_weight,
              "diagnostic_batches_per_stream": args.batches,
              "interpretation_note": CROSS_STREAM_NOTE,
              "checkpoints": []}

    for ck in cks:
        res = audit_checkpoint(aud, ck, clip)
        report["checkpoints"].append(res)
        print(f"\n=== {res['checkpoint']}  (step {res['global_step']}, "
              f"{res['N_exposures']:.0f} N-exposures, "
              f"c_align_weight={res['c_align_weight']}) ===")
        print("-- global gradient norms --")
        for k, v in res["global_norms"].items():
            print(f"  {k:22s} {v:9.4f}")
        pb = res["summed_per_batch_norms"]
        print(f"  summed per-batch norm: median={pb['median']:.3f} "
              f"min={pb['min']:.3f} max={pb['max']:.3f} "
              f"over_clip={pb['fraction_over_clip']*100:.0f}%")
        print("-- global cosines --")
        for k, v in res["cosines_global"].items():
            print(f"  {k:38s} {v:+.4f}")
        print("-- per-group norms (SUMMED / C_raw / C_align / N) --")
        for g, n in res["group_norms"].items():
            if max(n.values()) > 0:
                print(f"  {g:16s} {n['SUMMED']:9.4f} {n['C_retrieval_raw']:9.4f} "
                      f"{n['C_align']:9.4f} {n['N_naming']:9.4f}")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\n[json] {args.json_out}")
    print(f"\n[note] {CROSS_STREAM_NOTE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
