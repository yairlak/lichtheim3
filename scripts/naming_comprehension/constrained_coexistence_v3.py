"""Amendment-V3 off-line constrained coexistence search (deterministic).

Preregistration: docs/analysis/CONSTRAINED_COEXISTENCE_AMENDMENT_V3.md
                 commit fb634bec7f1490be588a17ab3a12f0531b7ea749

Scope: ONLY ltm.to_semantic.2.{weight,bias} may differ from the source.  The
encoder, to_semantic.0, sem_to_h0, decoder, motor, WM, gate and the GloVe bank
are frozen.  Source checkpoints are opened read-only and hashed before/after.
No training, no Adam, no backward(), no scheduler, no AutoResearch contact.

FORMULATION.  With x_i = [phi_i ; 1] (513) and the final affine map
s_i = Wbar x_i, let theta = Wbar - Wbar0 (300 x 513).  Then

    s_i - s_i^0 = theta x_i                                  (exact, affine)
    <a_k, theta> = d_k^T theta x_{i_k},   a_k = d_k x_{i_k}^T  (rank 1)

Bank rows are L2-normalised so argmax cos = argmax inner product; strict
top-1 is therefore the LINEAR system d_k^T s_{i_k} >= gamma_raw, i.e.
<a_k, theta> >= r_k with r_k = gamma_raw - d_k^T s^0_{i_k}.

Objective J(theta) = <theta, H theta> = mean_i || M_i theta x_i ||^2 with

    arm "I"        M_i = I                 H(th) = th Xc
    arm "Jz"       M_i = A                 H(th) = (A^T A) th Xc
    arm "Jlinh0"   M_i = D_i A             H(th) = (1/N) sum_i A^T D_i^2 A th x_i x_i^T

where Xc = (1/N) sum_i x_i x_i^T and D_i = diag(1 - tanh(z_i^0)^2) is FROZEN at
the deployed baseline (never updated during optimisation).

DUAL.  KKT stationarity gives theta = (1/2) H^{-1}(sum_k lam_k a_k), lam >= 0,
so the optimum lies in the K-dimensional span of the constraint gradients.  The
dual is  min_{lam>=0} (1/4) lam^T G lam - lam^T r  with G_kl = <a_l, H^{-1} a_k>.

For arms "I" and "Jz", H is exactly Kronecker and G has the CLOSED FORM

    G_kl = (d_l^T Q^{-1} d_k) * (x_k^T Xc^{-1} x_l),    Q = I or A^T A

verified to 3.6e-15 against direct evaluation.  For "Jlinh0" the per-item D_i
breaks that structure and H^{-1}a is obtained by preconditioned CG
(rtol 1e-8, max 500 iters); preconditioning changes only the path, never the
solution.

Failure to reach C=0 is ALGORITHM_FAILED_TO_FIND_WITNESS, never infeasibility.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---------------------------------------------------------- frozen constants
GAMMA_RAW = 1e-3
# Amendment V3.1 (commit 274002762a73f3f8a867558a03c9fc20b1c71584): a SOLVER
# TERMINATION TOLERANCE ONLY.  The scientific constraint remains exactly
# d^T s >= GAMMA_RAW.  It is 5x the observed 1.99e-08 roundoff deficit and
# 1e-4 of gamma, so it cannot admit a genuine strict error (margin <= 0 lies
# four orders of magnitude outside the acceptance threshold).  The solved QP
# -- r_k, Gram, objective, dual, active-set construction -- is unchanged.
FEAS_TOL_RAW = 1e-7
TOP_V = 3
K_MAX = 20_000
MAX_ROUNDS = 30
CG_RTOL = 1e-8
CG_MAXIT = 500
ARMS = ("I", "Jz", "Jlinh0")
SEEDS = (19, 20, 21, 22)
DERIVED_LABEL = ("DERIVED_CONSTRAINED_COEXISTENCE_DIAGNOSTIC/"
                 "NOT_OFFICIAL_MODEL/NOT_TRAINED_BY_JOINT_DRIVER")
PREREG_COMMIT = "fb634bec7f1490be588a17ab3a12f0531b7ea749"


def augment(phi: torch.Tensor) -> torch.Tensor:
    """x_i = [phi_i ; 1]."""
    return torch.cat([phi, torch.ones(phi.shape[0], 1, dtype=phi.dtype)], dim=1)


def feasibility_threshold() -> float:
    """The ONE coherent numerical-feasibility threshold (Amendment V3.1).

    An item counts as numerically feasible when its worst raw margin is
    >= GAMMA_RAW - FEAS_TOL_RAW.  This is the only tolerance in the solver and
    is used identically for the violation set and for termination.  It never
    changes the QP, and it can never admit a strict error (margin <= 0).
    """
    return GAMMA_RAW - FEAS_TOL_RAW


def constraint_value(theta: torch.Tensor, d: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """<a, theta> with a = d x^T, i.e. d^T theta x.  Convention is TESTED."""
    return d @ theta @ x


# ================================================================== metrics
class Metric:
    """H and H^{-1} for one preregistered arm."""

    def __init__(self, arm: str, A: torch.Tensor, Xc: torch.Tensor,
                 Dsq_mean: Optional[torch.Tensor] = None,
                 X_all: Optional[torch.Tensor] = None,
                 Dsq: Optional[torch.Tensor] = None):
        self.arm, self.A, self.Xc = arm, A, Xc
        self.Xinv = torch.linalg.inv(Xc)
        if arm == "I":
            self.Q = torch.eye(A.shape[1], dtype=A.dtype)
        elif arm == "Jz":
            self.Q = A.t() @ A
        elif arm == "Jlinh0":
            self.X_all, self.Dsq = X_all, Dsq
            self.Q = A.t() @ torch.diag(Dsq_mean) @ A          # preconditioner only
        else:
            raise ValueError(f"unknown arm {arm!r}")
        self.Qinv = torch.linalg.inv(self.Q)
        self.closed_form = arm in ("I", "Jz")

    # -- H(theta) --------------------------------------------------------
    def H(self, th: torch.Tensor) -> torch.Tensor:
        if self.arm == "I":
            return th @ self.Xc
        if self.arm == "Jz":
            return self.Q @ th @ self.Xc
        s = self.X_all @ th.t()                 # (N, m)
        z = s @ self.A.t()                      # (N, h)
        g = self.Dsq * z                        # (N, h)
        back = g @ self.A                       # (N, m)
        return (back.t() @ self.X_all) / self.X_all.shape[0]

    def _P_inv(self, R: torch.Tensor) -> torch.Tensor:
        return self.Qinv @ R @ self.Xinv

    def Hinv(self, R: torch.Tensor, log: Optional[dict] = None) -> torch.Tensor:
        """Exact for I/Jz (Kronecker); preconditioned CG for Jlinh0."""
        if self.closed_form:
            return self._P_inv(R)
        x = torch.zeros_like(R)
        r = R.clone()
        z = self._P_inv(r)
        p = z.clone()
        rz = float((r * z).sum())
        nrm0 = float(r.norm())
        it = 0
        for it in range(1, CG_MAXIT + 1):
            Hp = self.H(p)
            denom = float((p * Hp).sum())
            if denom <= 0:
                break
            al = rz / denom
            x = x + al * p
            r = r - al * Hp
            if float(r.norm()) <= CG_RTOL * nrm0:
                break
            z = self._P_inv(r)
            rz_new = float((r * z).sum())
            p = z + (rz_new / rz) * p
            rz = rz_new
        if log is not None:
            log.setdefault("cg_iters", []).append(it)
            log.setdefault("cg_relres", []).append(
                float(r.norm()) / nrm0 if nrm0 > 0 else 0.0)
        return x

    # -- Gram matrix of the dual ----------------------------------------
    def gram(self, ds: torch.Tensor, xs: torch.Tensor,
             log: Optional[dict] = None) -> torch.Tensor:
        K = ds.shape[0]
        if self.closed_form:
            Dm = ds @ self.Qinv @ ds.t()                  # (K,K)
            Xm = xs @ self.Xinv @ xs.t()                  # (K,K)
            return Dm * Xm.t()
        G = torch.zeros(K, K, dtype=ds.dtype)
        for k in range(K):
            Hk = self.Hinv(ds[k].outer(xs[k]), log=log)
            G[:, k] = (ds * (Hk @ xs.t()).t()).sum(dim=1)
        return 0.5 * (G + G.t())


# ============================================================== dual solver
def solve_dual(G: torch.Tensor, r: torch.Tensor, iters: int = 20000
               ) -> Tuple[torch.Tensor, dict]:
    """min_{lam>=0} (1/4) lam^T G lam - lam^T r, by projected gradient with
    exact Lipschitz step plus an equality polish on the final active set."""
    K = r.shape[0]
    lam = torch.zeros(K, dtype=r.dtype)
    L = float(torch.linalg.eigvalsh(0.5 * G).max()) if K else 1.0
    L = max(L, 1e-30)
    step = 1.0 / L
    for _ in range(iters):
        grad = 0.5 * (G @ lam) - r
        lam_new = torch.clamp(lam - step * grad, min=0.0)
        if float((lam_new - lam).abs().max()) <= 1e-14 * max(1.0, float(lam.abs().max())):
            lam = lam_new
            break
        lam = lam_new
    # polish: solve the equality system on the support, keep it only if valid
    S = (lam > 0).nonzero().flatten()
    info = {"support": int(S.numel()), "polished": 0}
    if S.numel():
        Gs = G[S][:, S]
        try:
            ls = torch.linalg.solve(Gs, 2.0 * r[S])
            if bool((ls >= -1e-12).all()):
                cand = torch.zeros_like(lam)
                cand[S] = torch.clamp(ls, min=0.0)
                if float((0.25 * cand @ G @ cand - cand @ r)) <= \
                   float((0.25 * lam @ G @ lam - lam @ r)) + 1e-12:
                    lam, info["polished"] = cand, 1
        except Exception:
            pass
    info["dual_obj"] = float(0.25 * lam @ G @ lam - lam @ r)
    return lam, info


def kkt_report(metric: Metric, theta: torch.Tensor, lam: torch.Tensor,
               ds: torch.Tensor, xs: torch.Tensor, r: torch.Tensor) -> dict:
    act = torch.stack([constraint_value(theta, ds[k], xs[k])
                       for k in range(ds.shape[0])]) if ds.shape[0] else torch.zeros(0)
    slack = act - r
    stat = 2.0 * metric.H(theta) - sum(
        (lam[k] * ds[k].outer(xs[k]) for k in range(ds.shape[0])),
        torch.zeros_like(theta))
    sc = max(float((2.0 * metric.H(theta)).abs().max()), 1e-30)
    return {
        "primal_min_slack": float(slack.min()) if slack.numel() else 0.0,
        "dual_min_lambda": float(lam.min()) if lam.numel() else 0.0,
        "max_complementarity": float((lam * slack).abs().max()) if slack.numel() else 0.0,
        "stationarity_rel": float(stat.abs().max()) / sc,
    }


# ========================================================= separation oracle
def margins_and_worst(theta: torch.Tensor, X_all: torch.Tensor, s0: torch.Tensor,
                      bn: torch.Tensor, tgt: torch.Tensor, topv: int = TOP_V,
                      block: int = 2048):
    """Raw target-vs-best-competitor margins, and the topv worst competitors
    for every item whose margin is < GAMMA_RAW.  The target is EXCLUDED."""
    N = X_all.shape[0]
    s = s0 + X_all @ theta.t()
    worst = torch.empty(N, dtype=s.dtype)
    cand = torch.full((N, topv), -1, dtype=torch.long)
    for lo in range(0, N, block):
        hi = min(lo + block, N)
        sc = s[lo:hi] @ bn.t()                            # (b, n_bank)
        rows = torch.arange(hi - lo)
        T = tgt[lo:hi]
        ts = sc[rows, T].clone()
        sc[rows, T] = -float("inf")                       # exclude the target
        top = sc.topk(topv, dim=1)
        worst[lo:hi] = ts - top.values[:, 0]
        cand[lo:hi] = top.indices
    return worst, cand, s


def run_arm(seed: int, arm: str, blob: dict, A: torch.Tensor, c: torch.Tensor,
            out_dir: str, verbose: bool = True) -> dict:
    """One preregistered (seed, arm) constrained search.  Deterministic."""
    phi = blob["phi"].double()
    bank = blob["bank_raw"].double()
    bn = torch.nn.functional.normalize(bank, dim=-1)
    tgt = torch.as_tensor(blob["target_idx"]).long()
    W0 = blob["head_state"]["2.weight"].double()
    b0 = blob["head_state"]["2.bias"].double()
    X_all = augment(phi)
    N = X_all.shape[0]
    s0 = X_all @ torch.cat([W0, b0.unsqueeze(1)], dim=1).t()
    Xc = (X_all.t() @ X_all) / N
    z0 = s0 @ A.t() + c
    Dsq = (1.0 - torch.tanh(z0) ** 2) ** 2
    metric = Metric(arm, A, Xc, Dsq_mean=Dsq.mean(0), X_all=X_all, Dsq=Dsq)

    theta = torch.zeros(W0.shape[0], X_all.shape[1], dtype=torch.float64)
    act: Dict[Tuple[int, int], None] = {}
    trace: List[dict] = []
    status = "ALGORITHM_FAILED_TO_FIND_WITNESS"
    for rnd in range(1, MAX_ROUNDS + 1):
        t0 = time.time()
        worst, cand, _ = margins_and_worst(theta, X_all, s0, bn, tgt)
        viol = (worst < feasibility_threshold()).nonzero().flatten()
        log: dict = {}
        if viol.numel() == 0:
            trace.append({"round": rnd, "violating_items": 0,
                          "active": len(act), "min_raw_margin": float(worst.min()),
                          "objective": float((theta * metric.H(theta)).sum()),
                          "new_constraints": 0, "wall_s": round(time.time() - t0, 2)})
            status = "CONSTRAINTS_SATISFIED"   # margin >= GAMMA_RAW - FEAS_TOL_RAW
            break
        before = len(act)
        for i in viol.tolist():
            for j in cand[i].tolist():
                if j >= 0:
                    act[(i, j)] = None
        if len(act) > K_MAX:
            status = "K_MAX_EXCEEDED"
            trace.append({"round": rnd, "violating_items": int(viol.numel()),
                          "active": len(act), "note": "K_MAX"})
            break
        keys = list(act.keys())
        ds = torch.stack([bn[tgt[i]] - bn[j] for i, j in keys])
        xs = torch.stack([X_all[i] for i, _ in keys])
        r = torch.stack([GAMMA_RAW - ds[k] @ s0[keys[k][0]] for k in range(len(keys))])
        G = metric.gram(ds, xs, log=log)
        lam, dinfo = solve_dual(G, r)
        theta = 0.5 * metric.Hinv(
            sum((lam[k] * ds[k].outer(xs[k]) for k in range(len(keys))),
                torch.zeros_like(theta)), log=log)
        kkt = kkt_report(metric, theta, lam, ds, xs, r)
        w2, _, _ = margins_and_worst(theta, X_all, s0, bn, tgt)
        rec = {"round": rnd, "violating_items": int(viol.numel()),
               "new_constraints": len(act) - before, "active": len(act),
               "min_raw_margin": float(w2.min()),
               "objective": float((theta * metric.H(theta)).sum()),
               "dual_support": dinfo["support"], "dual_polished": dinfo["polished"],
               "wall_s": round(time.time() - t0, 2), **kkt}
        if log.get("cg_iters"):
            rec["cg_iters_max"] = int(max(log["cg_iters"]))
            rec["cg_relres_max"] = float(max(log["cg_relres"]))
        trace.append(rec)
        if verbose:
            print(f"  [{arm} s{seed}] round {rnd}: viol={rec['violating_items']} "
                  f"K={rec['active']} minmarg={rec['min_raw_margin']:.3e} "
                  f"J={rec['objective']:.4e} stat={rec['stationarity_rel']:.1e} "
                  f"({rec['wall_s']}s)", flush=True)
    W = theta[:, :-1] + W0
    b = theta[:, -1] + b0
    ds_final = s0 + X_all @ theta.t()
    dz = (ds_final - s0) @ A.t()
    return {
        "seed": seed, "arm": arm, "status": status, "rounds": len(trace),
        "active_constraints": len(act), "trace": trace,
        "theta_norm": float(theta.norm()),
        "dW_norm": float(theta[:, :-1].norm()), "db_norm": float(theta[:, -1].norm()),
        "J_z": float(((ds_final - s0) @ A.t()).pow(2).sum(1).mean()),
        "J_lin_h0": float((Dsq.sqrt() * dz).pow(2).sum(1).mean()),
        "J_true": float((torch.tanh(z0 + dz) - torch.tanh(z0)).pow(2).sum(1).mean()),
        "min_raw_margin": float(margins_and_worst(theta, X_all, s0, bn, tgt)[0].min()),
        "min_cos_margin": float((margins_and_worst(theta, X_all, s0, bn, tgt)[0]
                                 / ds_final.norm(dim=1)).min()),
        "W": W, "b": b,
    }


def refuse_protected(out_dir: str) -> None:
    """Output isolation: never write into a source run, archive or AR root."""
    p = os.path.abspath(out_dir) + "/"
    forbidden = ("/lichtheim3_runs/", "/lichtheim3_autoresearch_runs/",
                 "/archives/settle_", "/archives/chigh_", "/archives/canneal_",
                 "/archives/lichtheim3_base123", "/archives/paper_evidence",
                 "/archives/final_report", "/archives/frozen_semantic_head_probe_",
                 "/archives/rootcause_recovery_", "/archives/head_interpolation_diag_",
                 "/archives/glove_geometry_",
                 "/lichtheim3-joint-scratch", "/lichtheim3-autoresearch")
    for f in forbidden:
        if f in p:
            raise RuntimeError(f"REFUSED: output root {out_dir} is protected ({f})")
    base = os.path.basename(p.rstrip("/"))
    if any(base.startswith(x) for x in ("final_", "cap", "l3_")):
        raise RuntimeError(f"REFUSED: output root {out_dir} uses a protected run prefix")
