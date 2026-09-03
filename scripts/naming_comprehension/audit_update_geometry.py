#!/usr/bin/env python3
"""FINAL-4 update audit: ACTUAL AdamW deltas at a frozen checkpoint.

Analysis only.  The source checkpoint and run are read-only; every measurement
starts from an exact fresh clone of the checkpoint's model AND optimizer
state, performs ONE hypothetical task optimizer step, records the actual
parameter delta, and discards the clone.  No file of the source run is
touched, no training happens.

Usage:
    python audit_update_geometry.py --checkpoint <step_00416700.pt> \
        [--batches 16] [--json-out audit.json] [--self-check]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import sys
from collections import OrderedDict

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from losses import alignment_loss, total_loss                            # noqa: E402
from scripts.naming_comprehension.train_joint_scratch import (           # noqa: E402
    FINAL_FULL_MODE, INTERLEAVED_123, LAMBDA_C, LAMBDA_N, TAU,
    JointScratchTrainer, build_batch,
)
from scripts.naming_comprehension.train_tasks import (                   # noqa: E402
    comprehension_forward, naming_objective, retrieval_loss,
)

# Parameter groups over named_parameters() (deduplicated: the shared phoneme
# embedding appears once, as phon_embed.weight; the wm./ltm. aliases exist
# only in state_dict()).
GROUPS = OrderedDict([
    ("phon_embed", ("phon_embed.",)),
    ("encoder_core", ("ltm.encoder.", "ltm.to_semantic.")),
    ("decoder_core", ("ltm.sem_to_h0.", "ltm.decoder.", "ltm.dec_to_premotor.")),
    ("motor", ("motor.",)),
    ("wm_only", ("wm.",)),
    ("gate", ("gate.",)),
])
ENC_SIDE = ("phon_embed", "encoder_core")            # R~C shared surface
DEC_SIDE = ("decoder_core", "motor")                 # R~N shared surface


def group_of(name):
    for g, p in GROUPS.items():
        if name.startswith(p):
            return g
    return "other"


def med(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return statistics.median(xs) if xs else float("nan")


def mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return statistics.fmean(xs) if xs else float("nan")


def cos(a, b):
    na, nb = float(a.norm()), float(b.norm())
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float((a @ b) / (na * nb))


class Audit:
    def __init__(self, ckpt_path, n_batches):
        self.tr = JointScratchTrainer(
            regime="j0", seed=22, device="cpu", max_words=30000,
            lexicon_path="data/lexicon_en_glove_covered.tsv",
            dorsal_pool_size=4000, batch_size=64,
            subset_mode=FINAL_FULL_MODE, subset_per_band=822, subset_size=64,
            lr_boundary_steps=46300, allow_glove_fallback=False,
            require_subset_hash=True, glove_path="data/glove.6B.300d.txt",
            schedule=INTERLEAVED_123)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.tr.load_state_dict(ck, source=os.path.basename(ckpt_path))
        self.ckpt_step = int(ck["global_step"])
        self.cursors = dict(self.tr.cursors)
        assert float(ck.get("c_align_weight", 0.0)) == 0.0, \
            "checkpoint must carry the retrieval-only C objective"

        self.model, self.optim = self.tr.model, self.tr.optim
        self.pad_id = self.tr.vocab.pad_id
        self.names = [n for n, _ in self.model.named_parameters()]
        self.gidx = {n: group_of(n) for n in self.names}
        self.wd = self.tr.cfg.train.weight_decay
        self.clip = self.tr.cfg.train.grad_clip

        # pristine clones (CPU) -- the single source every measurement restores
        self._model_sd = copy.deepcopy(self.model.state_dict())
        self._optim_sd = copy.deepcopy(self.optim.state_dict())
        # ||theta|| per group at the checkpoint
        self.theta_norm = {g: self._gnorm({n: p.detach()
                                           for n, p in self.model.named_parameters()}, g)
                           for g in list(GROUPS) + ["GLOBAL"]}

        # deterministic frontier batches: the next n per stream from the
        # checkpoint's own cursors (pure counter addressing)
        self.K = n_batches
        self.idx = {s: [self.tr.streams[s].indices(self.cursors[s] + j)
                        for j in range(n_batches)]
                    for s in ("repetition", "pool", "naming", "comprehension")}

    # ------------------------------------------------------------ plumbing
    def restore(self):
        self.model.load_state_dict(self._model_sd)
        self.optim.load_state_dict(copy.deepcopy(self._optim_sd))

    def _batch(self, stream, j):
        pop, bank = ((self.tr.pool_entries, self.tr.pool_bank)
                     if stream == "pool" else (self.tr.entries, self.tr.bank_raw))
        return build_batch(pop, bank, self.tr.vocab, self.idx[stream][j], "cpu")

    def _loss(self, task, j, lam_c):
        cfg = self.tr.cfg
        if task == "repetition":
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
            return parts["total"] + cfg.loss.wm * pool_ce
        if task == "naming":
            n = self._batch("naming", j)
            return LAMBDA_N * naming_objective(self.model, n, self.pad_id)["total"]
        c = self._batch("comprehension", j)
        s_hat = comprehension_forward(self.model, c["enc_in"], c["enc_mask"])
        return lam_c * retrieval_loss(s_hat, self.model.ltm.semantic_bank,
                                      c["bank_idx"], TAU)

    def _gnorm(self, tensors, group):
        ts = [tensors[n].reshape(-1) for n in self.names
              if n in tensors and (group == "GLOBAL" or self.gidx[n] == group)]
        return float(torch.cat(ts).norm()) if ts else 0.0

    def _grad_norm_global(self):
        sq = sum(float((p.grad ** 2).sum()) for p in self.model.parameters()
                 if p.grad is not None)
        return math.sqrt(sq)

    # -------------------------------------------------------- ONE hypothetical step
    def one_step(self, task, j, lam_c, lr, keep_flat=False):
        """Fresh clone -> one task optimizer step -> measure -> discard."""
        self.restore()
        self.model.train(True)
        self.optim.zero_grad(set_to_none=True)
        self._loss(task, j, lam_c).backward()

        preclip = self._grad_norm_global()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        postclip = self._grad_norm_global()
        touched = {n for n, p in self.model.named_parameters()
                   if p.grad is not None}
        grads = {n: p.grad.detach().clone()
                 for n, p in self.model.named_parameters() if p.grad is not None}

        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        for g in self.optim.param_groups:
            g["lr"] = lr
        self.optim.step()
        delta = {n: (p.detach() - before[n])
                 for n, p in self.model.named_parameters()}

        # decoupled weight decay: AdamW multiplies touched params by
        # (1 - lr*wd), i.e. Delta_wd = -lr*wd*theta on touched params only
        wd_norm = lr * self.wd * math.sqrt(
            sum(float((before[n] ** 2).sum()) for n in touched))
        adaptive = {n: (delta[n] + lr * self.wd * before[n]
                        if n in touched else delta[n]) for n in delta}

        # sustained-direction reference: if this exact (clipped) gradient were
        # applied for many steps, momentum saturates at m_hat -> g while v_hat
        # keeps its checkpoint history; the applied update tends to
        # lr * g / (sqrt(v_hat)+eps).  Analytic, no extra stepping.
        sus_sq = 0.0
        opt_state = self.optim.state_dict()["state"]
        pid = {n: i for i, (n, _) in enumerate(self.model.named_parameters())}
        beta2, eps = 0.999, 1e-8
        for n in touched:
            st = self._optim_sd["state"].get(pid[n])
            if st is None:
                continue
            t = float(st["step"]) if not torch.is_tensor(st["step"]) else float(st["step"])
            vhat = st["exp_avg_sq"] / (1 - beta2 ** max(t, 1.0))
            sus = lr * grads[n] / (vhat.sqrt() + eps)
            sus_sq += float((sus ** 2).sum())

        out = {
            "task": task, "batch": j, "lambda_c": lam_c, "lr": lr,
            "preclip": preclip, "postclip": postclip,
            "clipped": preclip > self.clip,
            "clip_factor": min(1.0, self.clip / preclip) if preclip > 0 else 1.0,
            "delta_norm": {g: self._gnorm(delta, g)
                           for g in list(GROUPS) + ["GLOBAL"]},
            "adaptive_norm_global": self._gnorm(adaptive, "GLOBAL"),
            "wd_norm_global": wd_norm,
            "sustained_delta_norm_global": math.sqrt(sus_sq),
        }
        if keep_flat:
            live = {g for g in GROUPS
                    if any(self.gidx[n] == g for n in self.names)}
            params = dict(self.model.named_parameters())
            out["_delta_flat"] = {g: torch.cat(
                [delta[n].reshape(-1) for n in self.names if self.gidx[n] == g])
                for g in live}
            out["_grad_flat"] = {g: torch.cat(
                [(grads[n] if n in grads
                  else torch.zeros_like(params[n])).reshape(-1)
                 for n in self.names if self.gidx[n] == g]) for g in live}
        return out

    # ------------------------------------------------- C diagnostic gradients
    def c_raw_grads(self, j):
        """Raw (unweighted) retrieval vs alignment gradients on the SAME C
        batch.  Diagnostic only: the current C update contains ZERO alignment."""
        self.restore()
        self.model.train(True)
        c = self._batch("comprehension", j)
        outs = {}
        for name, fn in (("retrieval", lambda s: retrieval_loss(
                              s, self.model.ltm.semantic_bank, c["bank_idx"], TAU)),
                         ("alignment", lambda s: alignment_loss(s, c["semantic"]))):
            self.optim.zero_grad(set_to_none=True)
            s_hat = comprehension_forward(self.model, c["enc_in"], c["enc_mask"])
            fn(s_hat).backward()
            outs[name] = {g: torch.cat(
                [(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                 for n, p in self.model.named_parameters() if self.gidx[n] == g])
                for g in ("phon_embed", "encoder_core")}
            outs[name]["enc_side"] = torch.cat(
                [outs[name]["phon_embed"], outs[name]["encoder_core"]])
            # split to_semantic out of encoder_core for the fine view
            outs[name]["to_semantic"] = torch.cat(
                [(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                 for n, p in self.model.named_parameters()
                 if n.startswith("ltm.to_semantic.")])
        return outs

    def self_check(self):
        """Restore fidelity + AdamW semantics asserted on the real objects."""
        a = self.one_step("comprehension", 0, LAMBDA_C, 1e-4)
        b = self.one_step("comprehension", 0, LAMBDA_C, 1e-4)
        assert abs(a["delta_norm"]["GLOBAL"] - b["delta_norm"]["GLOBAL"]) < 1e-12, \
            "restore is not exact: repeated measurement differs"
        # untouched params receive no update and no weight decay
        assert a["delta_norm"]["decoder_core"] == 0.0
        assert a["delta_norm"]["wm_only"] == 0.0
        # LR linearity of the one-step delta (moments unchanged by lr)
        c3 = self.one_step("comprehension", 0, LAMBDA_C, 3e-4)
        r = c3["delta_norm"]["GLOBAL"] / a["delta_norm"]["GLOBAL"]
        assert abs(r - 3.0) < 1e-3, f"one-step delta not linear in lr: {r}"
        # moments themselves are lr-independent
        self.restore()
        print("[self-check] PASS: exact restore, grad=None params untouched, "
              f"one-step delta scales x{r:.6f} under lr x3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    aud = Audit(args.checkpoint, args.batches)
    print(f"[audit] checkpoint step {aud.ckpt_step}, cursors {aud.cursors}")
    print(f"[audit] theta norms: " + " ".join(
        f"{g}={aud.theta_norm[g]:.2f}" for g in aud.theta_norm))
    if args.self_check:
        aud.self_check()

    # ---- measurement grid (deduplicated across the named configs) ----------
    CONDS = [("repetition", LAMBDA_C, 1e-4),
             ("naming", LAMBDA_C, 1e-4), ("naming", LAMBDA_C, 3e-4),
             ("comprehension", LAMBDA_C, 1e-4),
             ("comprehension", LAMBDA_C, 3e-4),
             ("comprehension", LAMBDA_C, 1e-3),
             ("comprehension", 1.0, 1e-4)]
    rows = {c: [] for c in CONDS}
    flats = {}          # baseline-config flats for geometry
    for j in range(aud.K):
        for cond in CONDS:
            task, lam, lr = cond
            keep = (lr == 1e-4 and (lam == LAMBDA_C or task == "comprehension"))
            r = aud.one_step(task, j, lam, lr, keep_flat=keep)
            rows[cond].append({k: v for k, v in r.items()
                               if not k.startswith("_")})
            if keep:
                flats[(task, lam, j)] = {
                    "delta": r["_delta_flat"], "grad": r["_grad_flat"]}
        print(f"[audit] batch {j+1}/{aud.K} done", flush=True)

    rep = {"checkpoint_step": aud.ckpt_step, "cursors": aud.cursors,
           "batches": aud.K, "theta_norm": aud.theta_norm,
           "conditions": {}}

    for cond in CONDS:
        task, lam, lr = cond
        rs = rows[cond]
        key = f"{task}|lambda={lam}|lr={lr:g}"
        rep["conditions"][key] = {
            "preclip": {"median": med([r["preclip"] for r in rs]),
                        "mean": mean([r["preclip"] for r in rs]),
                        "min": min(r["preclip"] for r in rs),
                        "max": max(r["preclip"] for r in rs)},
            "fraction_clipped": mean([1.0 * r["clipped"] for r in rs]),
            "clip_factor_median": med([r["clip_factor"] for r in rs]),
            "postclip_median": med([r["postclip"] for r in rs]),
            "delta_global_median": med([r["delta_norm"]["GLOBAL"] for r in rs]),
            "delta_global_mean": mean([r["delta_norm"]["GLOBAL"] for r in rs]),
            "delta_rel_global": med([r["delta_norm"]["GLOBAL"] for r in rs])
                                / aud.theta_norm["GLOBAL"],
            "delta_by_group_median": {
                g: med([r["delta_norm"][g] for r in rs]) for g in GROUPS},
            "adaptive_median": med([r["adaptive_norm_global"] for r in rs]),
            "wd_median": med([r["wd_norm_global"] for r in rs]),
            "sustained_delta_median": med(
                [r["sustained_delta_norm_global"] for r in rs]),
        }

    # ---- geometry: raw-grad vs actual-delta cosines ------------------------
    def pair_cos(a_key, b_key, kind, groups):
        out = {}
        for g in groups:
            vals = [cos(flats[a_key[:2] + (j,)][kind][g],
                        flats[b_key[:2] + (j,)][kind][g])
                    for j in range(aud.K)]
            out[g] = {"median": med(vals), "mean": mean(vals)}
        return out

    R, N, Cb, C1 = (("repetition", LAMBDA_C), ("naming", LAMBDA_C),
                    ("comprehension", LAMBDA_C), ("comprehension", 1.0))
    rep["geometry"] = {
        "raw_grad_R_vs_C_encoder_side": pair_cos(R, Cb, "grad",
                                                 ["phon_embed", "encoder_core"]),
        "raw_grad_R_vs_N_decoder_side": pair_cos(R, N, "grad",
                                                 ["decoder_core", "motor"]),
        "delta_R_vs_C_encoder_side": pair_cos(R, Cb, "delta",
                                              ["phon_embed", "encoder_core"]),
        "delta_R_vs_N_decoder_side": pair_cos(R, N, "delta",
                                              ["decoder_core", "motor"]),
        "delta_Clam1_vs_Clam087": pair_cos(C1, Cb, "delta",
                                           ["phon_embed", "encoder_core"]),
        "delta_Clam1_over_Clam087_norm_ratio": {
            g: med([float(flats[C1 + (j,)]["delta"][g].norm())
                    / float(flats[Cb + (j,)]["delta"][g].norm())
                    for j in range(aud.K)])
            for g in ("phon_embed", "encoder_core")},
    }

    # ---- C diagnostic: retrieval vs alignment raw gradients ----------------
    diag = {"note": ("DIAGNOSTIC ONLY: the current C update contains ZERO "
                     "alignment contribution (c_align_weight=0).")}
    vals = {k: [] for k in ("enc_side", "phon_embed", "encoder_core",
                            "to_semantic")}
    for j in range(aud.K):
        g2 = aud.c_raw_grads(j)
        for k in vals:
            vals[k].append(cos(g2["retrieval"][k], g2["alignment"][k]))
    diag["cos_retrieval_vs_alignment"] = {
        k: {"median": med(v), "mean": mean(v)} for k, v in vals.items()}
    rep["c_diagnostic"] = diag

    print(json.dumps(rep, indent=2, default=str))
    if args.json_out:
        json.dump(rep, open(args.json_out, "w"), indent=2, default=str)
        print(f"[json] {args.json_out}")


if __name__ == "__main__":
    main()
