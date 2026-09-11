"""Amendment-V4 frozen-encoder final-layer GRADIENT-TRAINING probe.

Preregistration: docs/analysis/GRADIENT_TRAINING_PROBE_PREREG_V4.md
                 commit 054633cdf173c27296bdbc406775329bd0bc38a4

Only theta = [W|b] of ltm.to_semantic.2 (300 x 513) is trained, on cached frozen
phi, full batch, float64, deterministic.  Three arms, identical step rule:

  A  gamma-hinge over violated constraints; p = -H^{-1} g, H = J_lin_h0 frozen at W0
  B  same gamma-hinge;                        p = -g
  C  deployed cosine CE at tau = 0.10;        p = -grad

Step: length-normalised Armijo backtracking on each arm's OWN objective,
alpha_0 = R0/||p||, BETA, ARMIJO_C, MAX_BACKTRACKS -- arms differ only in p.
No compatibility gradient, no lambda, no route constraint, no D update.
Official strict C=0 (unchanged evaluator) is the primary endpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.constrained_coexistence_v3 import (  # noqa: E402
    Metric, GAMMA_RAW, FEAS_TOL_RAW, feasibility_threshold, augment,
    refuse_protected as _v3_refuse_protected,
)
from scripts.naming_comprehension.frozen_head_probe import (  # noqa: E402
    official_strict_errors, sha256_file, build_trainer, _isolated_model,
)
from scripts.naming_comprehension.coexistence_probe import full_battery  # noqa: E402

# ------------------------------------------------ frozen constants (V4 prereg)
PREREG_COMMIT = "054633cdf173c27296bdbc406775329bd0bc38a4"
ARMS = ("A", "B", "C")
SEEDS = (19, 20, 21, 22)
TAU = 0.10
R0 = 0.1
BETA = 0.5
ARMIJO_C = 1e-4
MAX_BACKTRACKS = 20
MAX_ITERS = 150
WALL_CAP_S = 7200
CG_RELRES_FAIL = 1e-6
TRAINABLE_NAMES = ("ltm.to_semantic.2.weight", "ltm.to_semantic.2.bias")
STATUSES = ("CONVERGED", "LINE_SEARCH_FAILED", "NUMERICAL_FAILURE", "MAX_ITERS", "WALL_CAP")
DERIVED_LABEL = ("DERIVED_GRADIENT_TRAINING_DIAGNOSTIC/NOT_OFFICIAL_MODEL/"
                 "NOT_TRAINED_BY_JOINT_DRIVER")
BEST_V3 = {19: "Jz", 20: "Jlinh0", 21: "Jlinh0", 22: "Jlinh0"}


class NumericalFailure(RuntimeError):
    pass


def refuse_protected(out_dir: str) -> None:
    """V3 protected-path list, plus: never write under any archives/ tree."""
    _v3_refuse_protected(out_dir)
    if "/archives/" in os.path.abspath(out_dir) + "/":
        raise RuntimeError(f"REFUSED: output root {out_dir} is under archives/")


def sha256_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def assert_unchanged(path: str, before: str) -> None:
    after = sha256_file(path)
    if after != before:
        raise RuntimeError(f"SOURCE CHANGED: {path} {before} -> {after}")


# ================================================================ objectives
def margins(theta: torch.Tensor, X: torch.Tensor, bn: torch.Tensor,
            tgt: torch.Tensor, block: int = 1024):
    """Raw target-vs-hardest-competitor margins (target excluded).
    Ties in the hardest competitor resolve to the lowest index (torch.max)."""
    s = X @ theta.t()
    N = X.shape[0]
    m = torch.empty(N, dtype=s.dtype)
    j = torch.empty(N, dtype=torch.long)
    for lo in range(0, N, block):
        hi = min(lo + block, N)
        r = torch.arange(hi - lo)
        T = tgt[lo:hi]
        sc = s[lo:hi] @ bn.t()
        ts = sc[r, T].clone()
        sc[r, T] = float("-inf")
        mx, am = sc.max(dim=1)
        m[lo:hi] = ts - mx
        j[lo:hi] = am
    return m, j, s


def gamma_loss(theta, X, bn, tgt, gamma: float = GAMMA_RAW) -> float:
    m, _, _ = margins(theta, X, bn, tgt)
    V = m < gamma
    return float((gamma - m[V]).sum()) if bool(V.any()) else 0.0


def gamma_loss_and_grad(theta, X, bn, tgt, gamma: float = GAMMA_RAW):
    """L = SUM_{m_i<gamma} (gamma - m_i); g = -SUM_{m_i<gamma} d_i x_i^T.
    Only currently violated / insufficient-margin constraints carry gradient."""
    m, j, s = margins(theta, X, bn, tgt)
    V = (m < gamma).nonzero().flatten()
    if V.numel() == 0:
        return 0.0, torch.zeros_like(theta), m, j, s, V
    d = bn[tgt[V]] - bn[j[V]]
    g = -(d.t() @ X[V])
    return float((gamma - m[V]).sum()), g, m, j, s, V


def ce_loss(theta, X, bn, tgt, tau: float = TAU, block: int = 1024) -> float:
    """Deployed full-bank cosine CE, mean over items (no gradient)."""
    N = X.shape[0]
    tot = 0.0
    with torch.no_grad():
        for lo in range(0, N, block):
            hi = min(lo + block, N)
            q = F.normalize(X[lo:hi] @ theta.t(), dim=-1)
            tot += float(F.cross_entropy(q @ bn.t() / tau, tgt[lo:hi], reduction="sum"))
    return tot / N


def ce_loss_and_grad(theta, X, bn, tgt, tau: float = TAU, block: int = 1024):
    N = X.shape[0]
    th = theta.detach().clone().requires_grad_(True)
    tot = 0.0
    for lo in range(0, N, block):
        hi = min(lo + block, N)
        q = F.normalize(X[lo:hi] @ th.t(), dim=-1)
        L = F.cross_entropy(q @ bn.t() / tau, tgt[lo:hi], reduction="sum") / N
        L.backward()
        tot += float(L)
    return tot, th.grad.detach().clone()


# ========================================================== direction / step
def direction(arm: str, g: torch.Tensor, metric: Optional[Metric] = None,
              log: Optional[dict] = None) -> torch.Tensor:
    if arm == "A":
        lg = {} if log is None else log
        p = -metric.Hinv(g, log=lg)
        rr = lg.get("cg_relres", [0.0])[-1]
        if rr > CG_RELRES_FAIL:
            raise NumericalFailure(f"CG relative residual {rr:.3e} > {CG_RELRES_FAIL}")
        return p
    if arm in ("B", "C"):
        return -g
    raise ValueError(f"unknown arm {arm!r}")


def armijo(loss_fn: Callable[[torch.Tensor], float], theta: torch.Tensor,
           L0: float, g: torch.Tensor, p: torch.Tensor):
    """Length-normalised Armijo backtracking: first trial moves exactly R0."""
    gp = float((g * p).sum())
    pn = float(p.norm())
    if not (math.isfinite(gp) and math.isfinite(pn)) or pn == 0.0 or gp >= 0.0:
        raise NumericalFailure(f"non-descent or degenerate direction: g.p={gp}, |p|={pn}")
    a0 = R0 / pn
    for k in range(MAX_BACKTRACKS + 1):
        a = a0 * (BETA ** k)
        th = theta + a * p
        Lk = loss_fn(th)
        if math.isfinite(Lk) and Lk <= L0 + ARMIJO_C * a * gp:
            return th, a, k, Lk
    return None


# ================================================================ training run
def train_run(arm: str, theta0: torch.Tensor, X: torch.Tensor, bn: torch.Tensor,
              tgt: torch.Tensor, A: torch.Tensor, c: torch.Tensor,
              evaluator: Callable[[torch.Tensor], int],
              theta_v3: torch.Tensor, theta_v3_best: Optional[torch.Tensor] = None,
              v3_active_bank: Optional[set] = None,
              max_iters: int = MAX_ITERS, wall_cap_s: float = WALL_CAP_S,
              verbose: bool = False) -> dict:
    """Deterministic full-batch run from theta0.  `evaluator(s_hat_f32)` must be
    the official strict evaluator (returns #errors); it alone certifies C=0."""
    if arm not in ARMS:
        raise ValueError(arm)
    N = X.shape[0]
    z0 = (X @ theta0.t()) @ A.t() + c
    D = 1.0 - torch.tanh(z0) ** 2
    metric = None
    dsq_sha_start = None
    if arm == "A":
        Xc = (X.t() @ X) / N
        metric = Metric("Jlinh0", A, Xc, Dsq_mean=(D ** 2).mean(0), X_all=X, Dsq=D ** 2)
        dsq_sha_start = sha256_tensor(metric.Dsq)
    loss_fn = ((lambda th: gamma_loss(th, X, bn, tgt)) if arm in ("A", "B")
               else (lambda th: ce_loss(th, X, bn, tgt)))
    dv3 = (theta_v3 - theta0).flatten()
    dv3b = None if theta_v3_best is None else (theta_v3_best - theta0).flatten()
    nv3 = float(dv3.norm())
    nv3b = None if dv3b is None else float(dv3b.norm())
    theta = theta0.clone()
    traj: List[dict] = []
    official_calls: List[Tuple[int, int]] = []
    first_c0, first_c0_theta, src_res, note = None, None, None, ""
    status = None
    t_start = time.time()
    k = 0
    while True:
        t_it = time.time()
        if arm in ("A", "B"):
            L, g, m, j, s, V = gamma_loss_and_grad(theta, X, bn, tgt)
        else:
            m, j, s = margins(theta, X, bn, tgt)
            V = (m < GAMMA_RAW).nonzero().flatten()
        err = (m <= 0)
        errset = {int(x) for x in tgt[err].tolist()}
        if src_res is None:
            src_res = set(errset)
        if not bool(err.any()) and first_c0 is None:
            e = int(evaluator(s.float()))
            official_calls.append((k, e))
            if e == 0:
                first_c0, first_c0_theta = k, theta.clone()
        converged = float(m.min()) >= feasibility_threshold()
        stop = None
        if converged:
            stop = "CONVERGED"
        elif k >= max_iters:
            stop = "MAX_ITERS"
        elif time.time() - t_start > wall_cap_s:
            stop = "WALL_CAP"
        if arm == "C":
            if stop is None:
                L, g = ce_loss_and_grad(theta, X, bn, tgt)
            else:
                L = ce_loss(theta, X, bn, tgt)
        dth = theta - theta0
        dz = (X @ dth.t()) @ A.t()
        dflat = dth.flatten()
        nd = float(dflat.norm())
        viol_bank = {int(x) for x in tgt[V].tolist()}
        rec = {"it": k, "loss": L, "n_viol": int(V.numel()),
               "internal_strict_errors": int(err.sum()), "min_raw_margin": float(m.min()),
               "dtheta_norm": nd, "dW_norm": float(dth[:, :-1].norm()), "db_norm": float(dth[:, -1].norm()),
               "cos_to_v3": (float(dflat @ dv3) / (nd * nv3)) if (nd > 0 and nv3 > 0) else None,
               "dist_to_v3": float((theta - theta_v3).norm()),
               "cos_to_v3_best": (float(dflat @ dv3b) / (nd * nv3b)
                                  if (dv3b is not None and nd > 0 and nv3b > 0) else None),
               "J_z": float((dz ** 2).sum(1).mean()),
               "J_lin_h0": float(((D * dz) ** 2).sum(1).mean()),
               "J_true": float(((torch.tanh(z0 + dz) - torch.tanh(z0)) ** 2).sum(1).mean()),
               "viol_in_v3_active": (len(viol_bank & v3_active_bank) if v3_active_bank is not None else None),
               "errors_in_src_residual": len(errset & src_res),
               "new_errors_vs_src": len(errset - src_res)}
        traj.append(rec)
        if stop is not None:
            status = stop
            break
        lg: dict = {}
        try:
            p = direction(arm, g, metric, lg)
            res = armijo(loss_fn, theta, L, g, p)
        except NumericalFailure as ex:
            status, note = "NUMERICAL_FAILURE", str(ex)
            break
        if res is None:
            status = "LINE_SEARCH_FAILED"
            break
        theta, a, nb, _ = res
        if not bool(torch.isfinite(theta).all()):
            status, note = "NUMERICAL_FAILURE", "non-finite theta"
            break
        rec.update({"step_alpha": a, "backtracks": nb, "step_len": a * float(p.norm()),
                    "cg_iters": (lg.get("cg_iters") or [None])[-1],
                    "cg_relres": (lg.get("cg_relres") or [None])[-1],
                    "wall_s": round(time.time() - t_it, 3)})
        if verbose:
            print(f"  [{arm}] it {k}: L={L:.6g} viol={rec['n_viol']} err={rec['internal_strict_errors']} "
                  f"minm={rec['min_raw_margin']:.3e} |dth|={nd:.4f} cosV3={rec['cos_to_v3']} "
                  f"Jtrue={rec['J_true']:.4e} bt={nb}", flush=True)
        k += 1
    dsq_sha_end = sha256_tensor(metric.Dsq) if metric is not None else None
    return {"arm": arm, "status": status, "note": note, "steps": k,
            "first_c0_iter": first_c0, "official_calls": official_calls,
            "first_c0_theta": first_c0_theta, "final_theta": theta,
            "trajectory": traj, "dsq_sha_start": dsq_sha_start, "dsq_sha_end": dsq_sha_end,
            "wall_total_s": round(time.time() - t_start, 1)}


# ======================================================================= CLI
ARC = os.path.join(os.path.dirname(ROOT), "archives")
REP = os.path.join(ARC, "frozen_semantic_head_probe_20260911", "representations", "seed{s}.pt")
CKPT = os.path.join(ARC, "settle_u3600_20260910", "runs", "final_settle_ctrl_h512_s{s}",
                    "checkpoints", "step_10000800.pt")
V3H = os.path.join(ARC, "constrained_coexistence_v31_20260911", "derived_heads", "seed{s}_{a}.pt")
V3ACT = os.path.join(ARC, "feasible_region_characterization_20260911", "results", "p3_fix.json")
GLOVE = os.path.join(ROOT, "data", "glove.6B.300d.txt")


def _theta_of(W, b) -> torch.Tensor:
    return torch.cat([W.double(), b.double().unsqueeze(1)], 1)


def _save_head(path: str, th: torch.Tensor, meta: dict) -> str:
    torch.save({"label": DERIVED_LABEL, "trainable": list(TRAINABLE_NAMES),
                "state": {"2.weight": th[:, :-1].float().clone(), "2.bias": th[:, -1].float().clone()},
                "theta_float64": th.clone(), **meta}, path)
    return sha256_file(path)


def cmd_train(a) -> int:
    torch.set_num_threads(a.nt)
    run_dir = os.path.join(a.out_dir, f"seed{a.seed}_{a.arm}")
    refuse_protected(run_dir)
    if os.path.exists(run_dir):
        raise RuntimeError(f"REFUSED: run dir exists: {run_dir}")
    os.makedirs(run_dir)
    ck = CKPT.format(s=a.seed)
    ck_before = sha256_file(ck)
    blob = torch.load(REP.format(s=a.seed), map_location="cpu", weights_only=False)
    if blob["checkpoint_sha256"] != ck_before:
        raise RuntimeError("cached representation does not match source checkpoint")
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    msd = sd.get("model") or sd.get("model_state_dict") or sd.get("state_dict")
    A = [v for k, v in msd.items() if k.endswith("sem_to_h0.weight")][0].double()
    c = [v for k, v in msd.items() if k.endswith("sem_to_h0.bias")][0].double()
    X = augment(blob["phi"].double())
    bn = F.normalize(blob["bank_raw"].double(), dim=-1)
    tgt = torch.as_tensor(blob["target_idx"]).long()
    idx = [int(i) for i in blob["target_idx"]]
    theta0 = _theta_of(blob["head_state"]["2.weight"], blob["head_state"]["2.bias"])
    hv = torch.load(V3H.format(s=a.seed, a="Jlinh0"), map_location="cpu", weights_only=False)["state"]
    hb = torch.load(V3H.format(s=a.seed, a=BEST_V3[a.seed]), map_location="cpu", weights_only=False)["state"]
    act = set(json.load(open(V3ACT))["active"][str(a.seed)])
    bank_raw = blob["bank_raw"]
    res = train_run(a.arm, theta0, X, bn, tgt, A, c,
                    evaluator=lambda s: official_strict_errors(s, bank_raw, idx)["errors"],
                    theta_v3=_theta_of(hv["2.weight"], hv["2.bias"]),
                    theta_v3_best=_theta_of(hb["2.weight"], hb["2.bias"]),
                    v3_active_bank=act, verbose=True)
    assert_unchanged(ck, ck_before)
    meta = {"seed": a.seed, "arm": a.arm, "prereg_commit": PREREG_COMMIT,
            "code_commit": a.code_commit, "code_dirty_paths": a.code_dirty}
    heads = {}
    if res["first_c0_theta"] is not None:
        heads["first_c0"] = _save_head(os.path.join(run_dir, "head_first_c0.pt"), res["first_c0_theta"],
                                       {**meta, "iterate": res["first_c0_iter"]})
    heads["final"] = _save_head(os.path.join(run_dir, "head_final.pt"), res["final_theta"],
                                {**meta, "iterate": res["steps"]})
    rec = {k: v for k, v in res.items() if k not in ("first_c0_theta", "final_theta")}
    rec.update({"provenance": {**meta, "source_checkpoint": ck, "source_sha256": ck_before,
                               "source_unchanged": True, "rep_sha256_file": sha256_file(REP.format(s=a.seed)),
                               "v3_ref": V3H.format(s=a.seed, a="Jlinh0"),
                               "v3_ref_sha256": sha256_file(V3H.format(s=a.seed, a="Jlinh0")),
                               "threads": a.nt, "torch": torch.__version__, "argv": sys.argv,
                               "constants": {"GAMMA_RAW": GAMMA_RAW, "FEAS_TOL_RAW": FEAS_TOL_RAW,
                                             "TAU": TAU, "R0": R0, "BETA": BETA, "ARMIJO_C": ARMIJO_C,
                                             "MAX_BACKTRACKS": MAX_BACKTRACKS, "MAX_ITERS": MAX_ITERS,
                                             "WALL_CAP_S": WALL_CAP_S, "CG_RELRES_FAIL": CG_RELRES_FAIL}},
                "heads_sha256": heads})
    json.dump(rec, open(os.path.join(run_dir, "trace.json"), "w"), indent=1)
    print(f"== seed {a.seed} arm {a.arm}: {res['status']} steps={res['steps']} "
          f"first_c0={res['first_c0_iter']} ({res['wall_total_s']}s)", flush=True)
    return 0


def cmd_battery(a) -> int:
    torch.set_num_threads(a.nt)
    ck = CKPT.format(s=a.seed)
    before = sha256_file(ck)
    h = torch.load(a.head, map_location="cpu", weights_only=False)
    if set(h["state"]) != {"2.weight", "2.bias"}:
        raise RuntimeError("derived head must carry exactly the final-layer parameters")
    tr, _ = build_trainer(ck, "cpu", GLOVE)
    model = _isolated_model(tr, "p_last_hinge", h["state"])
    rec = full_battery(tr, model, list(tr.comp_idx))
    assert_unchanged(ck, before)
    rec.update({"seed": a.seed, "head": a.head, "head_sha256": sha256_file(a.head),
                "source_sha256": before, "source_unchanged": True})
    json.dump(rec, open(a.out_json, "w"), indent=1)
    print(f"== battery seed {a.seed} {os.path.basename(os.path.dirname(a.head))}/"
          f"{os.path.basename(a.head)}: C={rec['c_errors']} LTM={rec['rep_canonical_ltm']:.6f} "
          f"Rcan={rec['rep_canonical_full_errors']} N={rec['naming_errors']}", flush=True)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--seed", type=int, choices=SEEDS, required=True)
    t.add_argument("--arm", choices=ARMS, required=True)
    t.add_argument("--out-dir", required=True)
    t.add_argument("--nt", type=int, default=2)
    t.add_argument("--code-commit", required=True)
    t.add_argument("--code-dirty", type=int, required=True)
    b = sub.add_parser("battery")
    b.add_argument("--seed", type=int, choices=SEEDS, required=True)
    b.add_argument("--head", required=True)
    b.add_argument("--out-json", required=True)
    b.add_argument("--nt", type=int, default=2)
    a = ap.parse_args(argv)
    return cmd_train(a) if a.cmd == "train" else cmd_battery(a)


if __name__ == "__main__":
    sys.exit(main())
