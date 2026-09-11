"""FUNCTIONAL COEXISTENCE PROBE -- the FINAL authorised W2-only diagnostic.

Can ONE `ltm.to_semantic.2` on the SAME frozen phi be BOTH strict-C perfect
AND preserve the learned ventral repetition route, when preservation is
optimised through the ACTUAL historical decoder loss L_dec rather than an h0
proxy?

Everything except `ltm.to_semantic.2.{weight,bias}` stays frozen: embeddings,
encoder, to_semantic.0, GELU, sem_to_h0, decoder, dec_to_premotor, readout,
gate, WM route.  Source checkpoints are opened read-only and hashed before and
after.

ALGORITHM (frozen before execution; never re-tuned).

  Name: FEASIBILITY-PRESERVING BACKTRACKED L_dec DESCENT.

  This is NOT projected descent, NOT a constrained-minimisation solver, and
  NOT a projection onto the feasible set.  It follows the instantaneous
  negative full-batch gradient of L_dec and backtracks until a C-feasible,
  Armijo-valid step is found.  LINE_SEARCH_EXHAUSTED therefore means only
  "no acceptable step was found along this descent direction under the
  prespecified procedure" -- never constrained stationarity, local
  optimality, absence of other feasible directions, or C/LTM incompatibility.

  PHASE 1 -- enter a safe C-feasible interior.
    margin_i = cos(z_i, ghat_target) - max_{j != target} cos(z_i, ghat_j)
    L_C      = SUM_i max(0, 1 - margin_i / m),   m = 0.01
    Adam, lr 1e-3, batch 1024, order seed 12345, init = source head.
    ENTRY when count(margin < 0.01 - 1e-6) == 0 at an epoch end (NOT
    `L_C == 0`, which is a floating-point-equality non-termination hazard).
    The entry head is the FIRST qualifying epoch.

  PHASE 2 -- functional recovery.
    L_dec = the exact historical teacher-forced token CE over the repetition
    population, via the canonical modules (motor . ltm.decode_from_s_hat).
    FULL-BATCH plain gradient descent: no Adam, no momentum, no optimizer
    state, so the accepted iterate is exactly the evaluated iterate.
    eta_0 = 1e-3 reset each iteration, backtracking x0.5, MAX 20 halvings.
    A trial is accepted iff BOTH
      (1) cached-phi min cosine margin >= EPSILON = 1e-4, and
      (2) Armijo: L_dec(trial) <= L_dec(theta) - c*eta*||grad||^2, c = 1e-4.
    Budget 300 ACCEPTED iterations.  No stagnation early stop.

EPSILON = 1e-4 is fixed, not swept.  Independent review measured the
cached-vs-deployed cosine-margin discrepancy at source W2 (5.96e-7), the
largest-drift raw-hinge W2 (4.77e-7) and the COEX-2 W2 (5.36e-7): no
amplification with W2 drift, so epsilon retains >= 168x empirical headroom.

m = 0.01 is the project's historical DIAGNOSTIC near-tie constant
(MARGIN_EPS); its use as a training margin is preregistered, not inherited.

L_dec IS NOT THE SUCCESS METRIC.  It is teacher-forced token CE; the ventral
success metric is the canonical forced-length AUTOREGRESSIVE LTM exact.
Optimise and select with L_dec; decide success only with the official AR
metric.  No L_dec threshold ever constitutes ventral preservation.

Naming and WM do not depend on to_semantic.2 (verified from the code path:
naming decodes from raw GloVe through sem_to_h0).  They are therefore EXACT
invariance controls, and any change is an implementation bug, not a result.

Failure is only ever empirical.  Impossibility is never claimed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.frozen_head_probe import (      # noqa: E402
    SEEDS, BASELINE_C_ERRORS, N_COMP_POP, N_BANK, CANON,
    sha256_file, sha256_tensor, write_tsv, append_tsv, ckpt_path,
    build_trainer, official_strict_errors, _isolated_model,
)
from scripts.naming_comprehension.coexistence_probe import (      # noqa: E402
    cos_margins, full_battery, verify_official_twice, MARGIN_M,
)

# ---------------------------------------------------------- frozen constants
LR_P1, BATCH_P1, MAX_EPOCHS_P1 = 1e-3, 1024, 300
DATA_SEED = 12345
ENTRY_TOL = 1e-6                 # count(margin < m - ENTRY_TOL) == 0
EPSILON = 1e-4                   # Phase-2 cached-phi feasibility floor
ETA0 = 1e-3
BACKTRACK = 0.5
MAX_HALVINGS = 20
MAX_ACCEPTED = 300
ARMIJO_C = 1e-4
OFFICIAL_C_EVERY = 50            # accepted iterations
N_REP_POP = 29_571
TRAINABLE_NAMES = ("ltm.to_semantic.2.weight", "ltm.to_semantic.2.bias")

LABELS_ALLOWED = {
    "FULL_FUNCTIONAL_COEXISTENCE",
    "STRICT_C_COEXISTS_WITH_WHOLE_MODEL_BASELINE_VENTRAL_DEGRADED",
    "EMPIRICAL_FUNCTIONAL_COEXISTENCE_FAILURE",
}
ANNOTATIONS_ALLOWED = {
    "LINE_SEARCH_EXHAUSTED", "NUMERICAL_FAILURE",
    "NUMERICAL_FEASIBILITY_DISCREPANCY",
    "FUNCTIONAL_PHASE1_FEASIBILITY_FAILURE", "IMPLEMENTATION_BUG_HARD_ABORT",
    "BUDGET_EXHAUSTED", "ZERO_ACCEPTED_STEPS",
}
LABELS_FORBIDDEN = ("NO_SHARED_HEAD_EXISTS", "INTRINSIC_C_LTM_CONFLICT",
                    "GENUINE_TENSION_PROVEN", "UPSTREAM_FREEDOM_REQUIRED",
                    "H512_INSUFFICIENT")


def feat_path(rep_dir: str, seed: int) -> str:
    return os.path.join(rep_dir, f"seed{seed}.pt")


def head_path(out: str, seed: int, tag: str) -> str:
    return os.path.join(out, "derived_heads", f"seed{seed}_{tag}.pt")


# ============================================================== L_dec ======
def build_dec_batch(tr, idx: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Teacher-forced decoder tensors, constructed EXACTLY as the canonical
    train_tasks repetition batch: dec_in = [BOS] + form, dec_tgt = form + [EOS],
    padded with pad_id (which the CE ignores)."""
    v = tr.vocab
    forms = [list(tr.entries[i].phonemes) for i in idx]
    T = max(len(f) for f in forms) + 1
    din = torch.full((len(forms), T), v.pad_id, dtype=torch.long)
    dtg = torch.full((len(forms), T), v.pad_id, dtype=torch.long)
    for k, f in enumerate(forms):
        din[k, :len(f) + 1] = torch.tensor([v.bos_id] + f)
        dtg[k, :len(f) + 1] = torch.tensor(f + [v.eos_id])
    return din, dtg


def ldec_full(model, phi, W, b, din, dtg, w, pad_id, chunk=2048,
              need_grad=False) -> torch.Tensor:
    """Historical teacher-forced token CE over the repetition population.

    Normalisation is per NON-PAD TOKEN pooled over the whole population, i.e.
    exactly `F.cross_entropy(..., ignore_index=pad_id)` on the full batch.  The
    multiplicity weights w (sum = 29,571 over the 27,981 unique forms) make
    this identical to the direct mean over the historical 29,571-entry
    repetition population, because a repetition item's input IS its output, so
    homophones are the same item.
    """
    num = torch.zeros((), dtype=torch.float64)
    den = torch.zeros((), dtype=torch.float64)
    ctx = torch.enable_grad() if need_grad else torch.no_grad()
    with ctx:
        for lo in range(0, phi.shape[0], chunk):
            hi = lo + chunk
            z = phi[lo:hi] @ W.t() + b
            lg = model.motor(model.ltm.decode_from_s_hat(z, din[lo:hi]))
            ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                 dtg[lo:hi].reshape(-1),
                                 ignore_index=pad_id, reduction="none")
            ce = ce.view(z.shape[0], -1).sum(dim=1)
            ww = w[lo:hi]
            ntok = (dtg[lo:hi] != pad_id).sum(dim=1)
            num = num + (ww.double() * ce.double()).sum()
            den = den + (ww.double() * ntok.double()).sum()
    return (num / den).float()


def ldec_grad(model, phi, W, b, din, dtg, w, pad_id, chunk=2048):
    """Full-batch gradient of L_dec w.r.t. (W, b), accumulated over chunks."""
    if W.grad is not None:
        W.grad = None
    if b.grad is not None:
        b.grad = None
    num = torch.zeros((), dtype=torch.float64)
    den = torch.zeros((), dtype=torch.float64)
    parts = []
    for lo in range(0, phi.shape[0], chunk):
        hi = lo + chunk
        z = phi[lo:hi] @ W.t() + b
        lg = model.motor(model.ltm.decode_from_s_hat(z, din[lo:hi]))
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                             dtg[lo:hi].reshape(-1),
                             ignore_index=pad_id, reduction="none")
        ce = ce.view(z.shape[0], -1).sum(dim=1)
        ww = w[lo:hi]
        parts.append((ww * ce).sum())
        num = num + (ww.double() * ce.double()).sum()
        den = den + (ww.double() * (dtg[lo:hi] != pad_id).sum(dim=1).double()).sum()
    loss = torch.stack(parts).sum() / den.float()
    loss.backward()
    return (float((num / den).detach()), W.grad.detach().clone(),
            b.grad.detach().clone())


# ================================================== margins / feasibility ==
def margins_full(phi, W, b, bank_n, tgt, chunk=2048) -> torch.Tensor:
    outs = []
    with torch.no_grad():
        for lo in range(0, phi.shape[0], chunk):
            z = phi[lo:lo + chunk] @ W.t() + b
            outs.append(cos_margins(z, bank_n, tgt[lo:lo + chunk]))
    return torch.cat(outs)


def entry_satisfied(marg: torch.Tensor) -> Tuple[bool, int, float]:
    """count(margin < m - ENTRY_TOL) == 0.  Count-based, NOT `L_C == 0`."""
    n_bad = int((marg < (MARGIN_M - ENTRY_TOL)).sum())
    return n_bad == 0, n_bad, float(marg.min())


# ====================================================== invariance guards ==
def guards(derived: dict, base: dict) -> dict:
    """Source-relative.  Ventral is the MANDATORY primary scientific guard.
    Naming and WM are EXACT invariance controls, not scientific guards."""
    g = {
        "R_canonical_ok": derived["rep_canonical_full_errors"]
                          <= base["rep_canonical_full_errors"],
        "R_freear_ok": derived["rep_freear_full_errors"]
                       <= base["rep_freear_full_errors"],
        "ventral_ok": derived["rep_canonical_ltm_errors"]
                      <= base["rep_canonical_ltm_errors"],
    }
    # Naming and WM do NOT depend on to_semantic.2 -> exact equality required.
    g["naming_invariant"] = (derived["naming_errors"] == base["naming_errors"])
    g["wm_invariant"] = (derived["rep_canonical_wm_errors"]
                         == base["rep_canonical_wm_errors"])
    g["c_success"] = derived["c_errors"] == 0
    g["whole_model_preservation"] = bool(g["R_canonical_ok"] and g["R_freear_ok"])
    g["ventral_route_preservation"] = bool(g["ventral_ok"])
    g["full_functional_coexistence"] = bool(
        g["c_success"] and g["whole_model_preservation"]
        and g["ventral_route_preservation"]
        and g["naming_invariant"] and g["wm_invariant"])
    return g


def assert_trainable(model) -> None:
    """The ONLY trainable parameters are to_semantic.2.*; anything else aborts."""
    names = tuple(sorted(n for n, p in model.named_parameters() if p.requires_grad))
    if names != tuple(sorted(TRAINABLE_NAMES)):
        raise SystemExit(f"IMPLEMENTATION_BUG_HARD_ABORT trainable set {names}")


def frozen_fingerprint(model) -> str:
    """Hash of every parameter EXCEPT the trainable head."""
    h = []
    for n, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        if n in TRAINABLE_NAMES:
            continue
        h.append(f"{n}:{sha256_tensor(p.detach())}")
    import hashlib
    return hashlib.sha256("|".join(h).encode()).hexdigest()


def ltm_only_errors(tr, model) -> Tuple[int, float]:
    """Canonical forced-length AR LTM-route repetition, LTM route only."""
    from scripts.naming_comprehension.train_tasks import repetition_snapshot
    all_idx = list(range(len(tr.entries)))
    snap = repetition_snapshot(model, tr.vocab, tr.entries, all_idx,
                               tr.bank_raw, "cpu", routes=("ltm",),
                               include_teacher_forced=False)
    ex = float(snap["primary_readout"]["exact_match"]["ltm"])
    return int(round((1 - ex) * len(all_idx))), ex


def save_head(out, seed, tag, W, b, meta) -> str:
    p = head_path(out, seed, tag)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    torch.save({"seed": seed, "tag": tag,
                "state": {"2.weight": W.detach().cpu().clone(),
                          "2.bias": b.detach().cpu().clone()},
                "trainable_names": TRAINABLE_NAMES, **meta}, p)
    return p


# ==================================================================== RUN ==
def cmd_run(a) -> int:
    seed, out = a.seed, a.out_dir
    os.makedirs(out, exist_ok=True)
    t_start = time.time()
    ck = ckpt_path(a.archive_runs, seed)
    sha_before = sha256_file(ck)

    blob = torch.load(feat_path(a.rep_dir, seed), map_location="cpu",
                      weights_only=False)
    tr, _ = build_trainer(ck, "cpu", a.glove)
    model = tr.model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    idx_c = list(tr.comp_idx)
    assert len(idx_c) == N_COMP_POP, f"C population {len(idx_c)}"
    assert int(tr.bank_raw.shape[0]) == N_BANK, "bank size"
    assert len(tr.entries) == N_REP_POP, f"rep population {len(tr.entries)}"

    # ---------- baseline, re-derived through the canonical evaluators ----------
    base = full_battery(tr, model, idx_c)
    if base["c_errors"] != BASELINE_C_ERRORS[seed]:
        print(f"[func] HARD STOP seed {seed}: baseline C {base['c_errors']} "
              f"!= {BASELINE_C_ERRORS[seed]}")
        return 2
    print(f"[func] seed {seed} baseline: C={base['c_errors']} "
          f"R={base['rep_canonical_full_errors']} "
          f"RfAR={base['rep_freear_full_errors']} N={base['naming_errors']} "
          f"LTM_err={base['rep_canonical_ltm_errors']} "
          f"WM_err={base['rep_canonical_wm_errors']}", flush=True)
    append_tsv(os.path.join(out, "func_baselines.tsv"),
               [{"seed": seed, "source": "SOURCE_HEAD", **base}])

    # ---------- cached features, multiplicity weights, decoder batch ----------
    phi = blob["phi"].float()
    bank_n = F.normalize(blob["bank_raw"].float(), dim=-1)
    idx_u = list(blob["target_idx"])
    tgt = torch.tensor(idx_u, dtype=torch.long)
    hb = torch.load(os.path.join(a.h0_dir, f"seed{seed}.pt"),
                    map_location="cpu", weights_only=False)
    w = hb["w"].float()
    assert abs(float(w.sum()) - N_REP_POP) < 1e-3, "multiplicity sum"
    din, dtg = build_dec_batch(tr, idx_u)
    pad_id = tr.vocab.pad_id

    W = nn.Parameter(blob["head_state"]["2.weight"].float().clone())
    b = nn.Parameter(blob["head_state"]["2.bias"].float().clone())
    src_W = W.detach().clone()
    src_norm = float(torch.linalg.matrix_norm(src_W, 2))
    fp_before = frozen_fingerprint(model)

    ldec_source = float(ldec_full(model, phi, src_W, b.detach().clone(),
                                  din, dtg, w, pad_id))
    print(f"[func] seed {seed} source L_dec = {ldec_source:.6f}", flush=True)

    run = {"seed": seed, "sha_before": sha_before,
           "ldec_source": ldec_source,
           "baseline_ltm_errors": base["rep_canonical_ltm_errors"]}
    traj_path = os.path.join(out, "func_trajectory.tsv")

    # =========================== PHASE 1 : ENTRY ============================
    opt = torch.optim.Adam([W, b], lr=LR_P1)     # fresh; never the joint state
    g = torch.Generator().manual_seed(DATA_SEED)
    n = phi.shape[0]
    entry_epoch = None
    for ep in range(1, MAX_EPOCHS_P1 + 1):
        perm = torch.randperm(n, generator=g)
        for lo in range(0, n, BATCH_P1):
            sel = perm[lo:lo + BATCH_P1]
            z = phi[sel] @ W.t() + b
            loss = F.relu(1.0 - cos_margins(z, bank_n, tgt[sel]) / MARGIN_M).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        marg = margins_full(phi, W.detach(), b.detach(), bank_n, tgt)
        ok, n_bad, mmin = entry_satisfied(marg)
        if ep % 5 == 0 or ok:
            print(f"[func] seed {seed} P1 ep{ep}: n_bad={n_bad} "
                  f"min_margin={mmin:+.6f}", flush=True)
        if ok:
            entry_epoch = ep
            break
    if entry_epoch is None:
        run.update({"label": "FUNCTIONAL_PHASE1_FEASIBILITY_FAILURE",
                    "annotation": "FUNCTIONAL_PHASE1_FEASIBILITY_FAILURE"})
        append_tsv(os.path.join(out, "func_runs.tsv"), [run])
        return 0

    entry_W = W.detach().clone()
    entry_b = b.detach().clone()
    entry_marg_min = float(margins_full(phi, entry_W, entry_b, bank_n, tgt).min())
    entry_ldec = float(ldec_full(model, phi, entry_W, entry_b, din, dtg, w, pad_id))
    ep_path = save_head(out, seed, "ENTRY", entry_W, entry_b,
                        {"epoch": entry_epoch, "phase": 1,
                         "min_margin": entry_marg_min, "ldec": entry_ldec})
    # deployed-path official C at the entry head
    em = _isolated_model(tr, "p_last", {"2.weight": entry_W, "2.bias": entry_b})
    e_c1, e_c2, _ = verify_official_twice(tr, em, idx_c)
    if e_c1 != 0 or e_c2 != 0:
        run.update({"label": "NUMERICAL_FEASIBILITY_DISCREPANCY",
                    "annotation": "NUMERICAL_FEASIBILITY_DISCREPANCY",
                    "entry_epoch": entry_epoch, "entry_official_c": e_c1})
        append_tsv(os.path.join(out, "func_runs.tsv"), [run])
        print(f"[func] seed {seed} NUMERICAL_FEASIBILITY_DISCREPANCY at entry "
              f"(cached min margin {entry_marg_min:+.3e}, official C {e_c1})")
        return 0
    entry_ltm_err, entry_ltm_ex = ltm_only_errors(tr, em)   # LTM route ONLY
    del em
    append_tsv(os.path.join(out, "func_entry_heads.tsv"), [{
        "seed": seed, "entry_epoch": entry_epoch,
        "min_margin": entry_marg_min, "official_c_1": e_c1, "official_c_2": e_c2,
        "ldec_entry": entry_ldec, "ldec_source": ldec_source,
        "ltm_errors_entry": entry_ltm_err, "ltm_exact_entry": entry_ltm_ex,
        "ltm_errors_source": base["rep_canonical_ltm_errors"],
        "head_sha256": sha256_file(ep_path), "head_path": ep_path}])
    print(f"[func] seed {seed} ENTRY ep{entry_epoch}: minmarg={entry_marg_min:+.6f} "
          f"C={e_c1}/{e_c2} L_dec={entry_ldec:.6f} LTM_err={entry_ltm_err}",
          flush=True)

    # ==================== PHASE 2 : BACKTRACKED L_dec DESCENT ===============
    W = nn.Parameter(entry_W.clone())
    b = nn.Parameter(entry_b.clone())
    accepted = 0
    annotation = "BUDGET_EXHAUSTED"
    cur = float(ldec_full(model, phi, W.detach(), b.detach(), din, dtg, w, pad_id))
    rows = []
    while accepted < MAX_ACCEPTED:
        cur_g, gW, gb = ldec_grad(model, phi, W, b, din, dtg, w, pad_id)
        if not (torch.isfinite(gW).all() and torch.isfinite(gb).all()):
            annotation = "NUMERICAL_FAILURE"
            break
        gn2 = float((gW ** 2).sum() + (gb ** 2).sum())
        eta = ETA0
        took = False
        for k in range(MAX_HALVINGS + 1):
            Wt = (W.detach() - eta * gW)
            bt = (b.detach() - eta * gb)
            marg = margins_full(phi, Wt, bt, bank_n, tgt)
            mmin = float(marg.min())
            if mmin < EPSILON:                       # (1) C feasibility
                eta *= BACKTRACK
                continue
            cand = float(ldec_full(model, phi, Wt, bt, din, dtg, w, pad_id))
            if not (cand == cand):
                annotation = "NUMERICAL_FAILURE"
                took = False
                break
            if cand <= cur_g - ARMIJO_C * eta * gn2:  # (2) Armijo
                with torch.no_grad():
                    dth = float(((Wt - W.detach()) ** 2).sum()
                                + ((bt - b.detach()) ** 2).sum()) ** 0.5
                    W.copy_(Wt)
                    b.copy_(bt)
                accepted += 1
                cur = cand
                q = torch.quantile(marg.double(), torch.tensor([0.01, 0.05]).double())
                rows.append({
                    "seed": seed, "accepted_iter": accepted, "ldec": cand,
                    "grad_norm": gn2 ** 0.5, "eta": eta, "delta_theta": dth,
                    "halvings": k, "min_margin": mmin,
                    "p1_margin": float(q[0]), "p5_margin": float(q[1]),
                    "w2_norm": float(torch.linalg.matrix_norm(W.detach(), 2)),
                    "w2_norm_ratio": float(torch.linalg.matrix_norm(W.detach(), 2))
                                      / src_norm,
                    "official_c_check": "",      # filled at the 50-iter cadence
                    "head_sha256": sha256_tensor(W.detach())})
                took = True
                break
            eta *= BACKTRACK
        if annotation == "NUMERICAL_FAILURE":
            break
        if not took:
            annotation = "LINE_SEARCH_EXHAUSTED"
            break
        if accepted % 10 == 0:
            print(f"[func] seed {seed} P2 it{accepted}: L_dec={cur:.6f} "
                  f"minmarg={rows[-1]['min_margin']:+.3e} eta={rows[-1]['eta']:.2e} "
                  f"halv={rows[-1]['halvings']}", flush=True)
        if accepted % OFFICIAL_C_EVERY == 0:         # official C cadence
            cm = _isolated_model(tr, "p_last",
                                 {"2.weight": W.detach(), "2.bias": b.detach()})
            c1, c2, _ = verify_official_twice(tr, cm, idx_c)
            del cm
            rows[-1]["official_c_check"] = c1
            print(f"[func] seed {seed} official C @it{accepted}: {c1}/{c2}",
                  flush=True)
            if c1 != 0 or c2 != 0:
                annotation = "NUMERICAL_FEASIBILITY_DISCREPANCY"
                break
    if rows:
        append_tsv(traj_path, rows)

    if annotation == "NUMERICAL_FEASIBILITY_DISCREPANCY":
        run.update({"label": "NUMERICAL_FEASIBILITY_DISCREPANCY",
                    "annotation": annotation, "entry_epoch": entry_epoch,
                    "accepted_iterations": accepted})
        append_tsv(os.path.join(out, "func_runs.tsv"), [run])
        return 0
    if accepted == 0:
        annotation = "ZERO_ACCEPTED_STEPS"

    # -------- retained head = FINAL accepted iterate (never LTM/R/N) --------
    ret_W = W.detach().clone()
    ret_b = b.detach().clone()
    ret_ldec = cur
    rp = save_head(out, seed, "RETAINED", ret_W, ret_b,
                   {"phase": 2, "accepted_iterations": accepted,
                    "annotation": annotation, "ldec": ret_ldec})

    rm = _isolated_model(tr, "p_last", {"2.weight": ret_W, "2.bias": ret_b})
    r_c1, r_c2, det = verify_official_twice(tr, rm, idx_c)
    derived = full_battery(tr, rm, idx_c)
    del rm
    gd = guards(derived, base)
    fp_after = frozen_fingerprint(model)
    sha_after = sha256_file(ck)

    if not gd["naming_invariant"] or not gd["wm_invariant"]:
        label, ann = "IMPLEMENTATION_BUG_HARD_ABORT", "IMPLEMENTATION_BUG_HARD_ABORT"
    elif gd["full_functional_coexistence"]:
        label, ann = "FULL_FUNCTIONAL_COEXISTENCE", annotation
    elif gd["c_success"] and gd["whole_model_preservation"]:
        label = "STRICT_C_COEXISTS_WITH_WHOLE_MODEL_BASELINE_VENTRAL_DEGRADED"
        ann = annotation
    else:
        label, ann = "EMPIRICAL_FUNCTIONAL_COEXISTENCE_FAILURE", annotation
    assert label in LABELS_ALLOWED or label == "IMPLEMENTATION_BUG_HARD_ABORT"

    append_tsv(os.path.join(out, "func_retained_heads.tsv"), [{
        "seed": seed, "accepted_iterations": accepted, "annotation": ann,
        "ldec_source": ldec_source, "ldec_entry": entry_ldec,
        "ldec_retained": ret_ldec,
        "min_margin": float(margins_full(phi, ret_W, ret_b, bank_n, tgt).min()),
        "official_c_1": r_c1, "official_c_2": r_c2, "deterministic": int(det),
        "head_sha256": sha256_file(rp), "head_path": rp}])
    append_tsv(os.path.join(out, "func_compatibility.tsv"), [{
        "seed": seed, "candidate": "RETAINED", "label": label,
        "official_c_1": r_c1, "official_c_2": r_c2,
        **{k: derived[k] for k in sorted(derived)},
        **{f"base_{k}": base[k] for k in
           ("rep_canonical_full_errors", "rep_freear_full_errors",
            "naming_errors", "rep_canonical_ltm_errors",
            "rep_canonical_wm_errors")},
        "ltm_errors_entry": entry_ltm_err,
        **{k: int(v) if isinstance(v, bool) else v for k, v in gd.items()}}])
    run.update({"label": label, "annotation": ann, "entry_epoch": entry_epoch,
                "entry_min_margin": entry_marg_min,
                "accepted_iterations": accepted,
                "ldec_entry": entry_ldec, "ldec_retained": ret_ldec,
                "ltm_errors_entry": entry_ltm_err,
                "ltm_errors_retained": derived["rep_canonical_ltm_errors"],
                "sha_after": sha_after, "source_unchanged": int(sha_before == sha_after),
                "frozen_fingerprint_unchanged": int(fp_before == fp_after),
                "minutes": round((time.time() - t_start) / 60, 1)})
    append_tsv(os.path.join(out, "func_runs.tsv"), [run])
    print(f"[func] seed {seed} => {label} [{ann}] accepted={accepted} "
          f"L_dec {entry_ldec:.4f}->{ret_ldec:.4f} (src {ldec_source:.4f}) "
          f"LTM {entry_ltm_err}->{derived['rep_canonical_ltm_errors']} "
          f"(src {base['rep_canonical_ltm_errors']}) C={r_c1}/{r_c2}", flush=True)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seed", type=int, required=True)
    for p in (r,):
        p.add_argument("--out-dir", required=True)
        p.add_argument("--rep-dir", required=True)
        p.add_argument("--h0-dir", required=True)
        p.add_argument("--archive-runs", required=True)
        p.add_argument("--glove", required=True)
    a = ap.parse_args(argv)
    return {"run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
