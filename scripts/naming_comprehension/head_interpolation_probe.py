"""Weight-interpolation compatibility test between the deployed u3600 final
semantic layer and the verified P-last C=0 final layer (Amendment V2).

DETERMINISTIC ANALYSIS ONLY.  No backward(), no optimizer, no new objective,
no SLURM, no GPU requirement, no source checkpoint write.  Every scientific
quantity is produced by the OFFICIAL evaluators imported from the unmodified
lineage code; nothing is reimplemented here.

Construction (phi frozen, final layer affine):

    W(a) = (1-a) W0 + a W*,   b(a) = (1-a) b0 + a b*
    s_i(a) = W(a) phi_i + b(a) = (1-a) s_i(0) + a s_i(1)        (exact)

Analytic C threshold.  Bank rows are L2-normalised, so for non-zero s,
argmax_j cos(s,t_j) = argmax_j s^T t_j.  Strict correctness at item i with
target row T needs, for all j != T:

    A_ij + a*B_ij > 0,  A_ij = u_i^T(t_T - t_j),  B_ij = delta_i^T(t_T - t_j)

Each constraint is affine in a; an intersection of half-lines in one variable
is convex, so every item's feasible set is ONE open interval (L_i, U_i) and the
global set is (max_i L_i, min_i U_i).  alpha_C0 = max_i L_i is an INFIMUM: C=0
holds strictly above it.  Verified against the official evaluator on both sides.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.frozen_head_probe import (      # noqa: E402
    SEEDS, STEP, N_BANK, SEM_DIM, POOLED_DIM, TAU,
    sha256_file, sha256_tensor, write_tsv, ckpt_path, feat_path,
    official_strict_errors, build_trainer,
)

EPS = 1e-6                     # fixed in the amendment, before any result
ALPHA_FRACTIONS = (0.25, 0.50, 0.75, 0.90)      # fixed before any result
DERIVED_LABEL = "DERIVED_INTERPOLATION_DIAGNOSTIC/NOT_OFFICIAL_MODEL/NOT_TRAINED"

# Inherited Phase-0 reference lines.  Status: UNPREREGISTERED for this probe.
INHERITED = {
    "relative_ltm_mean_guard": -0.02,
    "relative_ltm_per_seed_guard": -0.03,
    "absolute_ltm_review_line": 0.8498,
    "repetition_guard_max_excess_errors": 2.0,
    "naming_guard_max_candidate_errors": 1,
    "status": "UNPREREGISTERED_FOR_THIS_DIAGNOSTIC",
}


# ------------------------------------------------------------------ helpers
# Measured numerical status of the weight/output interpolation identity:
#   fp32-rounded weights  -> max_abs_dev <= 2.4e-07 (one fp32 ULP at |s| ~ 2-5),
#                            exact (0.0) at alpha in {0, 1};
#   fp64-retained weights -> max_abs_dev ~ 1.8e-15 (fp64 ULP), i.e. exact.
# The same 2.4e-07 scale the parent probe reported for its phi-decomposition.
FP32_IDENTITY_TOL = 1e-6        # > 4x the measured fp32 ULP bound


def lerp_state(h0: Dict[str, torch.Tensor], hs: Dict[str, torch.Tensor],
               alpha: float, dtype: Optional[torch.dtype] = None
               ) -> Dict[str, torch.Tensor]:
    """(1-a)*deployed + a*Plast for the final layer keys only.

    `dtype=None` keeps the source dtype (fp32: what a real deployed module is,
    used for OFFICIAL evaluation).  `dtype=torch.float64` retains double and
    makes the weight/output interpolation identity exact, used for the analytic
    threshold.  Both are reported; neither is silently substituted.
    """
    out = {}
    for k in ("2.weight", "2.bias"):
        m = (1.0 - alpha) * h0[k].double() + alpha * hs[k].double()
        out[k] = m if dtype is torch.float64 else m.to(dtype or h0[k].dtype)
    return out


def s_hat_at(phi: torch.Tensor, st: Dict[str, torch.Tensor]) -> torch.Tensor:
    return phi.double() @ st["2.weight"].double().t() + st["2.bias"].double()


def alpha_thresholds(u: torch.Tensor, v: torch.Tensor, bank_n: torch.Tensor,
                     tgt: torch.Tensor, block: int = 512
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-item (L_i, U_i) plus the competitor index attaining L_i.

    Returns (L, U, arg_lower, infeasible_mask).  All in float64.
    """
    n = u.shape[0]
    L = torch.zeros(n, dtype=torch.float64)
    U = torch.full((n,), float("inf"), dtype=torch.float64)
    argL = torch.full((n,), -1, dtype=torch.long)
    bad = torch.zeros(n, dtype=torch.bool)
    ud, vd, bd = u.double(), v.double(), bank_n.double()
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        T = tgt[lo:hi]
        du = vd[lo:hi] - ud[lo:hi]
        Pu = ud[lo:hi] @ bd.t()                       # (b, N)
        Pd = du @ bd.t()                              # (b, N)
        r = torch.arange(hi - lo)
        A = Pu[r, T].unsqueeze(1) - Pu                # u^T(t_T - t_j)
        B = Pd[r, T].unsqueeze(1) - Pd                # delta^T(t_T - t_j)
        A[r, T] = float("inf")                        # exclude the target itself
        B[r, T] = 0.0
        ratio = -A / B
        pos, neg, zer = B > 0, B < 0, B == 0
        # lower bounds from positive slopes that are not yet satisfied
        cand = torch.where(pos & (A <= 0), ratio, torch.full_like(ratio, -float("inf")))
        lmax, li = cand.max(dim=1)
        upd = lmax > L[lo:hi]
        L[lo:hi] = torch.maximum(L[lo:hi], lmax.clamp(min=0.0))
        argL[lo:hi] = torch.where(upd, li, argL[lo:hi])
        # upper bounds from negative slopes
        candU = torch.where(neg, ratio, torch.full_like(ratio, float("inf")))
        U[lo:hi] = torch.minimum(U[lo:hi], candU.min(dim=1).values)
        # zero slope and not already strictly satisfied -> infeasible everywhere
        bad[lo:hi] = (zer & (A <= 0)).any(dim=1)
    return L, U, argL, bad


def isolated_with_head(tr, st: Dict[str, torch.Tensor]):
    """Deep copy of the trainer's model carrying the interpolated final layer."""
    model = copy.deepcopy(tr.model)
    sd = model.ltm.to_semantic.state_dict()
    for k, v in st.items():
        sd[k] = v.clone()
    model.ltm.to_semantic.load_state_dict(sd)
    model.eval()
    return model


def full_eval(tr, model, n_rep: int) -> dict:
    """OFFICIAL evaluators only.  Returns C / LTM / R / N (+gate if exposed)."""
    from scripts.naming_comprehension.train_tasks import (
        evaluate_comprehension_subset, evaluate_naming, repetition_snapshot)
    idx = list(tr.comp_idx)
    rec = {}
    c = evaluate_comprehension_subset(model, tr.vocab, tr.entries,
                                      tr.bank_raw, idx, "cpu", 512)
    rec["c_top1"] = float(c["top1"])
    rec["c_errors"] = int(round((1 - float(c["top1"])) * len(idx)))
    for k in ("retrieval_ce", "margin_mean", "rank_median"):
        if k in c:
            rec[f"c_{k}"] = float(c[k])
    all_idx = list(range(len(tr.entries)))
    fr = repetition_snapshot(model, tr.vocab, tr.entries, all_idx,
                             tr.bank_raw, "cpu", include_teacher_forced=False)
    ex = fr["primary_readout"]["exact_match"]
    for route in ("full", "wm", "ltm"):
        rec[f"rep_canonical_{route}"] = float(ex[route])
    rec["rep_canonical_errors"] = int(round((1 - float(ex["full"])) * n_rep))
    gate = fr.get("gate") or fr.get("gate_statistics")
    rec["gate_mean"] = float(gate["mean"]) if isinstance(gate, dict) and "mean" in gate \
        else "UNAVAILABLE"
    orig = tr.model
    try:
        tr.model = model
        far = tr.free_ar_repetition(all_idx, routes=("full", "wm", "ltm"))
    finally:
        tr.model = orig
    for route in ("full", "wm", "ltm"):
        rec[f"rep_freear_{route}"] = float(far[route])
    rec["rep_freear_errors"] = int(round((1 - float(far["full"])) * n_rep))
    nm = evaluate_naming(model, tr.vocab, tr.entries, tr.bank_raw,
                         all_idx, "cpu", 256)
    rec["naming_exact"] = float(nm.get("exact_match", 0.0))
    rec["naming_errors"] = int(round((1 - rec["naming_exact"]) * n_rep))
    return rec


def refuse_protected(out_dir: str) -> None:
    """Output isolation: never write into a source run, archive or AR root."""
    p = os.path.abspath(out_dir)
    forbidden = ("/lichtheim3_runs/", "/lichtheim3_autoresearch_runs/",
                 "/archives/settle_", "/archives/chigh_", "/archives/canneal_",
                 "/archives/lichtheim3_base123", "/archives/paper_evidence",
                 "/archives/final_report", "/archives/frozen_semantic_head_probe_",
                 "/archives/rootcause_recovery_", "/lichtheim3-joint-scratch",
                 "/lichtheim3-autoresearch")
    for f in forbidden:
        if f in p + "/":
            raise RuntimeError(f"REFUSED: output root {p} is a protected location ({f})")
    if any(os.path.basename(p.rstrip("/")).startswith(x) for x in ("final_", "cap", "l3_")):
        raise RuntimeError(f"REFUSED: output root {p} uses a protected run prefix")
