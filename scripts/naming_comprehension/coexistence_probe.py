"""Semantic coexistence probe: can ONE to_semantic.2 on fixed phi be both
strict-C perfect AND preserve the learned ventral repetition route?

Everything except `ltm.to_semantic.2.{weight,bias}` stays frozen.  No encoder,
no to_semantic.0, no decoder, no gate, no WM.  Source checkpoints are opened
read-only and hashed before and after.

OBJECTIVES (frozen before execution; never re-tuned).

  margin_i = cos(z_i, ghat_target) - max_{j != target} cos(z_i, ghat_j)
  l_i      = max(0, 1 - margin_i / m),        m = 0.01
  L_C      = SUM_i l_i        (sum, not mean: any strict error contributes >=1)

  L_P      = (1/29571) * SUM_i w_i * ||h0_new(i) - h0_src(i)||^2 / (4*512)
             with h0 = tanh(sem_to_h0(s_hat)), h0 in [-1,1]^512 so L_P in [0,1].
             w_i are lexical multiplicities over the 27,981 UNIQUE phonological
             forms with sum(w) = 29,571, making this identical to the direct
             mean over the historical repetition population.

  COEX-1:  L_C
  COEX-2:  L_C + L_P            -- no lambda; the hierarchy is structural.
           Every strict C error costs >= 1 while the entire preservation term
           is bounded by 1, so constraints dominate until margins are met, and
           the optimisation then becomes pure minimal decoder-state
           displacement.

m = 0.01 is the project's historical DIAGNOSTIC near-tie constant
(MARGIN_EPS); its use as an optimisation margin is newly preregistered here,
not inherited.  m = 0 is rejected because an exact-tie strict error has
margin 0 and would receive no hinge gradient.

Failure is only ever empirical.  LINEAR/INTRINSIC impossibility is never
claimed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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
    SEEDS, RUN, STEP, BASELINE_C_ERRORS, N_COMP_POP, N_BANK, SEM_DIM,
    POOLED_DIM, CANON, sha256_file, sha256_tensor, write_tsv, append_tsv,
    ckpt_path, build_trainer, official_strict_errors, _isolated_model,
)

# ---------------------------------------------------------- frozen constants
MARGIN_M = 0.01
LR, BATCH, MAX_EPOCHS, EVAL_EVERY = 1e-3, 1024, 300, 5
DATA_SEED = 12345
N_REP_POP = 29_571
H0_DIM = 512
H0_SQ_MAX = 4.0                      # (tanh range [-1,1]) => max sq diff 4
TRAINABLE_NAMES = ("ltm.to_semantic.2.weight", "ltm.to_semantic.2.bias")
STAGES = ("coex1", "coex2")

LABELS_ALLOWED = {
    "COEX1_FULL_COEXISTENCE", "COEX1_C_ZERO_PRESERVATION_FAIL",
    "COEX1_EMPIRICAL_C_FAILURE", "COEX2_FULL_COEXISTENCE",
    "COEX2_C_ZERO_NO_PRESERVED_SOLUTION_FOUND", "COEX2_EMPIRICAL_C_FAILURE",
    "NUMERICAL_FAILURE",
}
LABELS_FORBIDDEN = ("NO_SHARED_HEAD_EXISTS", "INTRINSIC_C_LTM_CONFLICT",
                    "GENUINE_TENSION_PROVEN", "H512_INSUFFICIENT")


def feat_path(rep_dir: str, seed: int) -> str:
    return os.path.join(rep_dir, f"seed{seed}.pt")


def h0_path(out: str, seed: int) -> str:
    return os.path.join(out, "h0_src", f"seed{seed}.pt")


def head_path(out: str, seed: int, stage: str, tag: str) -> str:
    return os.path.join(out, "derived_heads", f"seed{seed}_{stage}_{tag}.pt")


# ------------------------------------------------------------- C objective
def cos_margins(z: torch.Tensor, bank_n: torch.Tensor,
                tgt: torch.Tensor) -> torch.Tensor:
    """cos(target) - max over NON-target competitors, full bank."""
    q = F.normalize(z, dim=-1)
    sc = q @ bank_n.t()
    ts = sc.gather(1, tgt.view(-1, 1)).squeeze(1)
    bw = sc.scatter(1, tgt.view(-1, 1), float("-inf")).max(dim=1).values
    return ts - bw


def hinge_terms(margin: torch.Tensor) -> torch.Tensor:
    """l_i = max(0, 1 - margin/m).  A strict error (margin <= 0) gives >= 1."""
    return F.relu(1.0 - margin / MARGIN_M)


def c_loss(z, bank_n, tgt) -> torch.Tensor:
    return hinge_terms(cos_margins(z, bank_n, tgt)).sum()


# --------------------------------------------------- preservation objective
def h0_of(z: torch.Tensor, Wh: torch.Tensor, bh: torch.Tensor) -> torch.Tensor:
    return torch.tanh(z @ Wh.t() + bh)


def p_loss(h0_new: torch.Tensor, h0_src: torch.Tensor,
           w: torch.Tensor) -> torch.Tensor:
    """Normalised, multiplicity-weighted decoder-state displacement in [0,1]."""
    sq = ((h0_new - h0_src) ** 2).sum(dim=-1) / (H0_SQ_MAX * H0_DIM)
    return (w * sq).sum() / N_REP_POP


# ================================================================ BASELINES
def full_battery(tr, model, idx_c: List[int]) -> dict:
    """The canonical compatibility battery.  Expensive: run only at the
    predeclared candidate points."""
    from scripts.naming_comprehension.train_tasks import (
        evaluate_comprehension_subset, evaluate_naming, repetition_snapshot)
    all_idx = list(range(len(tr.entries)))
    n = len(all_idx)
    out = {}
    c = evaluate_comprehension_subset(model, tr.vocab, tr.entries, tr.bank_raw,
                                      idx_c, "cpu", 512)
    out["c_top1"] = float(c["top1"])
    out["c_errors"] = int(round((1 - c["top1"]) * len(idx_c)))
    snap = repetition_snapshot(model, tr.vocab, tr.entries, all_idx,
                               tr.bank_raw, "cpu", include_teacher_forced=False)
    ex = snap["primary_readout"]["exact_match"]
    for r in ("full", "wm", "ltm"):
        out[f"rep_canonical_{r}"] = float(ex[r])
        out[f"rep_canonical_{r}_errors"] = int(round((1 - float(ex[r])) * n))
    orig = tr.model
    try:
        tr.model = model
        far = tr.free_ar_repetition(all_idx, routes=("full", "wm", "ltm"))
    finally:
        tr.model = orig
    for r in ("full", "wm", "ltm"):
        out[f"rep_freear_{r}"] = float(far[r])
        out[f"rep_freear_{r}_errors"] = int(round((1 - float(far[r])) * n))
    nm = evaluate_naming(model, tr.vocab, tr.entries, tr.bank_raw, all_idx,
                         "cpu", 256)
    out["naming_exact"] = float(nm.get("exact_match", 0.0))
    out["naming_errors"] = int(round((1 - out["naming_exact"]) * n))
    try:
        g = tr.gate_statistics(all_idx)
        out["gate_mean"] = float(g.get("gate_mean", float("nan")))
    except Exception:
        out["gate_mean"] = float("nan")
    return out


def cmd_baselines(a) -> int:
    os.makedirs(a.out_dir, exist_ok=True)
    rows, hashes = [], []
    for seed in SEEDS:
        ck = ckpt_path(a.archive_runs, seed)
        before = sha256_file(ck)
        tr, _ = build_trainer(ck, "cpu", a.glove)
        idx = list(tr.comp_idx)
        assert len(idx) == N_COMP_POP, f"C population {len(idx)}"
        assert int(tr.bank_raw.shape[0]) == N_BANK, "bank size"
        assert len(tr.entries) == N_REP_POP, f"rep population {len(tr.entries)}"
        b = full_battery(tr, tr.model, idx)
        if b["c_errors"] != BASELINE_C_ERRORS[seed]:
            print(f"[coex] HARD STOP seed {seed}: baseline C {b['c_errors']} "
                  f"!= {BASELINE_C_ERRORS[seed]}")
            return 2
        after = sha256_file(ck)
        if before != after:
            print(f"[coex] HARD STOP: source mutated {ck}")
            return 3
        rows.append({"seed": seed, "source": "SOURCE_HEAD", **b})
        hashes.append({"seed": seed, "checkpoint": ck, "sha256_before": before,
                       "sha256_after": after, "unchanged": int(before == after)})
        print(f"[coex] seed {seed} baseline: C={b['c_errors']} "
              f"R={b['rep_canonical_full_errors']} "
              f"RfreeAR={b['rep_freear_full_errors']} N={b['naming_errors']} "
              f"LTM_err={b['rep_canonical_ltm_errors']} "
              f"WM={b['rep_canonical_wm']:.6f}")
        del tr
    write_tsv(os.path.join(a.out_dir, "coex_baselines.tsv"), rows)
    write_tsv(os.path.join(a.out_dir, "source_checkpoint_hashes.tsv"), hashes)
    return 0


# ============================================ h0 SOURCE CACHE + MULTIPLICITY
def cmd_h0_src(a) -> int:
    """Cache h0_source and the lexical multiplicity weights.

    Homophones share a phoneme sequence, hence identical pooled, phi, s_hat and
    h0.  Weighting the 27,981 unique forms by their lexical multiplicity is
    therefore EXACTLY the mean over the historical 29,571-entry repetition
    population -- verified numerically here, not assumed.
    """
    os.makedirs(os.path.join(a.out_dir, "h0_src"), exist_ok=True)
    rows = []
    for seed in SEEDS:
        ck = ckpt_path(a.archive_runs, seed)
        before = sha256_file(ck)
        blob = torch.load(feat_path(a.rep_dir, seed), map_location="cpu",
                          weights_only=False)
        if blob["checkpoint_sha256"] != before:
            print(f"[coex] HARD STOP: cached features for seed {seed} were "
                  f"built from a different checkpoint")
            return 2
        tr, ckd = build_trainer(ck, "cpu", a.glove)
        idx = list(blob["target_idx"])
        # map every lexicon entry's phonology onto its canonical C target
        pos = {tuple(tr.entries[i].phonemes): k for k, i in enumerate(idx)}
        w = torch.zeros(len(idx), dtype=torch.float64)
        unmapped = 0
        for e in tr.entries:
            k = pos.get(tuple(e.phonemes))
            if k is None:
                unmapped += 1
            else:
                w[k] += 1.0
        if unmapped:
            print(f"[coex] HARD STOP: {unmapped} lexicon entries have no "
                  f"canonical C target of the same phonology")
            return 3
        if int(w.sum()) != N_REP_POP:
            print(f"[coex] HARD STOP: multiplicity sum {int(w.sum())} != {N_REP_POP}")
            return 4
        sd = ckd["model_state_dict"]
        Wh = sd["ltm.sem_to_h0.weight"].float(); bh = sd["ltm.sem_to_h0.bias"].float()
        phi = blob["phi"].float()
        W2 = blob["head_state"]["2.weight"].float(); b2 = blob["head_state"]["2.bias"].float()
        with torch.no_grad():
            z0 = phi @ W2.t() + b2
            h0 = h0_of(z0, Wh, bh)
        # L_P at the source head must be exactly 0
        lp0 = float(p_loss(h0, h0, w.float()))
        torch.save({"seed": seed, "h0_src": h0, "w": w.float(),
                    "sem_to_h0_weight": Wh, "sem_to_h0_bias": bh,
                    "checkpoint_sha256": before, "target_idx": idx},
                   h0_path(a.out_dir, seed))
        after = sha256_file(ck)
        rows.append({"seed": seed, "n_unique_forms": len(idx),
                     "multiplicity_sum": int(w.sum()),
                     "max_multiplicity": int(w.max()),
                     "n_forms_with_multiplicity_gt1": int((w > 1).sum()),
                     "L_P_source_head": lp0,
                     "h0_sha256": sha256_tensor(h0),
                     "source_unchanged": int(before == after)})
        print(f"[coex] seed {seed}: weights sum {int(w.sum())} over "
              f"{len(idx)} unique forms (max mult {int(w.max())}), "
              f"L_P(source) = {lp0:.3e}")
        del tr
    write_tsv(os.path.join(a.out_dir, "coex_h0_manifest.tsv"), rows)
    return 0


# ================================================ PRE-FIT ACTIVE-SET CHECK
def cmd_drift_check(a) -> int:
    """Recompute active-constraint counts with the EXACT COEX implementation,
    and evaluate L_P for the archived raw-hinge C-perfect heads."""
    rows = []
    for seed in SEEDS:
        blob = torch.load(feat_path(a.rep_dir, seed), map_location="cpu",
                          weights_only=False)
        hb = torch.load(h0_path(a.out_dir, seed), map_location="cpu",
                        weights_only=False)
        phi = blob["phi"].float()
        bank_n = F.normalize(blob["bank_raw"].float(), dim=-1)
        tgt = torch.tensor(list(blob["target_idx"]), dtype=torch.long)
        W2 = blob["head_state"]["2.weight"].float()
        b2 = blob["head_state"]["2.bias"].float()
        Wh, bh, w, h0s = (hb["sem_to_h0_weight"], hb["sem_to_h0_bias"],
                          hb["w"], hb["h0_src"])

        def stats(W, b, tag):
            with torch.no_grad():
                z = phi @ W.t() + b
                mg = torch.cat([cos_margins(z[lo:lo + 2048], bank_n,
                                            tgt[lo:lo + 2048])
                                for lo in range(0, z.shape[0], 2048)])
                lc = float(hinge_terms(mg).sum())
                lp = float(p_loss(h0_of(z, Wh, bh), h0s, w))
            return {"tag": tag, "c_errors": int((mg <= 0).sum()),
                    "active_items_m0.01": int((mg < MARGIN_M).sum()),
                    "min_margin": round(float(mg.min()), 6),
                    "L_C": round(lc, 4), "L_P": round(lp, 8),
                    "W2_norm": round(float(W.norm()), 4)}

        r = {"seed": seed, **{f"src_{k}": v for k, v in
                              stats(W2, b2, "source").items() if k != "tag"}}
        ap = os.path.join(a.prev_probe, "derived_heads",
                          f"seed{seed}_p_last_hinge.pt")
        if os.path.exists(ap):
            st = torch.load(ap, map_location="cpu", weights_only=False)["state"]
            s2 = stats(st["2.weight"].float(), st["2.bias"].float(), "rawhinge")
            r.update({f"rawhinge_{k}": v for k, v in s2.items() if k != "tag"})
            if not (0.0 <= s2["L_P"] <= 1.0):
                print(f"[coex] HARD STOP: L_P={s2['L_P']} outside [0,1]")
                return 2
        rows.append(r)
        print(f"[coex] seed {seed}: source C_err={r['src_c_errors']} "
              f"active(m=0.01)={r['src_active_items_m0.01']} "
              f"L_C={r['src_L_C']} | rawhinge C_err={r.get('rawhinge_c_errors')} "
              f"L_P={r.get('rawhinge_L_P')}")
    write_tsv(os.path.join(a.out_dir, "coex_drift.tsv"), rows)
    return 0


# ======================================================== GUARDS / VERDICTS
def guards(derived: dict, base: dict) -> dict:
    """All source-relative.  The ventral guard is MANDATORY primary."""
    g = {
        "R_canonical_ok": derived["rep_canonical_full_errors"]
                          <= base["rep_canonical_full_errors"],
        "R_freear_ok": derived["rep_freear_full_errors"]
                       <= base["rep_freear_full_errors"],
        "N_ok": derived["naming_errors"] <= base["naming_errors"],
        "ventral_ok": derived["rep_canonical_ltm_errors"]
                      <= base["rep_canonical_ltm_errors"],
    }
    g["whole_model_preservation"] = bool(g["R_canonical_ok"]
                                         and g["R_freear_ok"] and g["N_ok"])
    g["ventral_route_preservation"] = bool(g["ventral_ok"])
    g["c_success"] = derived["c_errors"] == 0
    g["full_coexistence"] = bool(g["c_success"] and g["whole_model_preservation"]
                                 and g["ventral_route_preservation"])
    # W2 does not enter the WM route: any change is an implementation bug.
    g["wm_invariant"] = (derived["rep_canonical_wm_errors"]
                         == base["rep_canonical_wm_errors"])
    return g


def verify_official_twice(tr, model, idx) -> Tuple[int, int, bool]:
    from scripts.naming_comprehension.frozen_probe import encode_all
    forms = [tr.entries[i].phonemes for i in idx]
    with torch.no_grad():
        s1 = encode_all(model, tr.vocab, forms, "cpu", 512)
        r1 = official_strict_errors(s1, tr.bank_raw, idx)["errors"]
        s2 = encode_all(model, tr.vocab, forms, "cpu", 512)
        r2 = official_strict_errors(s2, tr.bank_raw, idx)["errors"]
    return r1, r2, bool(torch.equal(s1, s2))


def save_head(out, seed, stage, tag, W, b, epoch, extra=None) -> str:
    p = head_path(out, seed, stage, tag)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    torch.save({"stage": stage, "tag": tag, "seed": seed, "epoch": epoch,
                "state": {"2.weight": W.detach().cpu().clone(),
                          "2.bias": b.detach().cpu().clone()},
                "trainable_names": TRAINABLE_NAMES, **(extra or {})}, p)
    return p


# ==================================================================== FIT ==
def cmd_fit(a) -> int:
    stage, seed = a.stage, a.seed
    out = a.out_dir
    blob = torch.load(feat_path(a.rep_dir, seed), map_location="cpu",
                      weights_only=False)
    hb = torch.load(h0_path(out, seed), map_location="cpu", weights_only=False)
    base = {r["seed"]: r for r in
            [{k: (int(v) if k.endswith("errors") or k == "seed"
                  else v) for k, v in row.items()}
             for row in csv.DictReader(
                 open(os.path.join(out, "coex_baselines.tsv")), delimiter="\t")]
            }[seed]

    phi = blob["phi"].float()
    bank_n = F.normalize(blob["bank_raw"].float(), dim=-1)
    idx = list(blob["target_idx"])
    tgt = torch.tensor(idx, dtype=torch.long)
    Wh, bh, w, h0s = (hb["sem_to_h0_weight"], hb["sem_to_h0_bias"],
                      hb["w"], hb["h0_src"])

    W = nn.Parameter(blob["head_state"]["2.weight"].float().clone())
    b = nn.Parameter(blob["head_state"]["2.bias"].float().clone())
    opt = torch.optim.Adam([W, b], lr=LR)          # fresh; never the joint state
    use_p = (stage == "coex2")

    ck = ckpt_path(a.archive_runs, seed)
    sha_before = sha256_file(ck)
    tr, _ = build_trainer(ck, "cpu", a.glove)      # for official verification
    print(f"[coex] {stage} seed {seed}: trainable {TRAINABLE_NAMES}, "
          f"objective L_C{' + L_P' if use_p else ''}")

    g = torch.Generator().manual_seed(DATA_SEED)
    n = phi.shape[0]
    traj, cands = [], []
    first_c0 = None
    label = "COEX1_EMPIRICAL_C_FAILURE" if stage == "coex1" \
        else "COEX2_EMPIRICAL_C_FAILURE"
    retained = None
    t0 = time.time()

    def cheap_eval(ep):
        with torch.no_grad():
            z = phi @ W.t() + b
            mg = torch.cat([cos_margins(z[lo:lo + 2048], bank_n, tgt[lo:lo + 2048])
                            for lo in range(0, n, 2048)])
            lc = float(hinge_terms(mg).sum())
            h0n = h0_of(z, Wh, bh)
            lp = float(p_loss(h0n, h0s, w))
            cerr = official_strict_errors(z, blob["bank_raw"], idx)["errors"]
            zs = blob["s_hat_reference"].float()
            row = {"stage": stage, "seed": seed, "epoch": ep,
                   "c_errors": cerr, "L_C": round(lc, 6), "L_P": round(lp, 9),
                   "min_cos_margin": round(float(mg.min()), 6),
                   "margin_q1": round(float(torch.quantile(mg.double(), .01)), 6),
                   "margin_q50": round(float(torch.quantile(mg.double(), .5)), 6),
                   "active_lt_m": int((mg < MARGIN_M).sum()),
                   "W2_norm": round(float(W.norm()), 4),
                   "s_hat_norm_ratio": round(float(
                       (z.norm(dim=-1) / zs.norm(dim=-1)).median()), 5),
                   "h0_cos_median": round(float(F.cosine_similarity(
                       h0n, h0s, dim=-1).median()), 6),
                   "elapsed_s": round(time.time() - t0, 1)}
        return row, cerr, lp

    for ep in range(1, MAX_EPOCHS + 1):
        perm = torch.randperm(n, generator=g)
        for lo in range(0, n, BATCH):
            sel = perm[lo:lo + BATCH]
            z = phi[sel] @ W.t() + b
            loss = c_loss(z, bank_n, tgt[sel])
            if use_p:
                loss = loss + p_loss(h0_of(z, Wh, bh), h0s[sel], w[sel])
            if not torch.isfinite(loss):
                label = "NUMERICAL_FAILURE"
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if label == "NUMERICAL_FAILURE":
            break

        if ep % EVAL_EVERY == 0 or ep == 1 or ep == MAX_EPOCHS:
            row, cerr, lp = cheap_eval(ep)
            traj.append(row)
            print(f"[coex]   ep{ep:>3} C={cerr:>3} L_C={row['L_C']:>10.3f} "
                  f"L_P={row['L_P']:.3e} minmarg={row['min_cos_margin']:+.5f} "
                  f"act={row['active_lt_m']:>4} h0cos={row['h0_cos_median']:.4f}")
            if cerr == 0:
                tag = "FIRST_C0" if first_c0 is None else f"c0_ep{ep}"
                p = save_head(out, seed, stage, tag, W, b, ep, {"L_P": lp})
                cands.append({"stage": stage, "seed": seed, "epoch": ep,
                              "tag": tag, "L_P": lp, "L_C": row["L_C"],
                              "min_cos_margin": row["min_cos_margin"],
                              "head_sha256": sha256_file(p), "path": p})
                if first_c0 is None:
                    first_c0 = cands[-1]
                    # --- predeclared: battery at FIRST_C0 -------------------
                    model = _isolated_model(tr, "p_last_hinge",
                                            {"2.weight": W.detach().clone(),
                                             "2.bias": b.detach().clone()})
                    r1, r2, det = verify_official_twice(tr, model, idx)
                    bat = full_battery(tr, model, idx)
                    gd = guards(bat, base)
                    if not gd["wm_invariant"]:
                        print("[coex] IMPLEMENTATION_BUG_HARD_ABORT: WM changed")
                        return 9
                    append_tsv(os.path.join(out, "coex_compatibility.tsv"),
                               [{"stage": stage, "seed": seed, "epoch": ep,
                                 "candidate": "FIRST_C0",
                                 "official_eval_1": r1, "official_eval_2": r2,
                                 "deterministic": int(det), "L_P": lp,
                                 **bat, **{k: int(v) if isinstance(v, bool) else v
                                           for k, v in gd.items()}}])
                    if gd["full_coexistence"] and r1 == 0 and r2 == 0:
                        label = ("COEX1_FULL_COEXISTENCE" if stage == "coex1"
                                 else "COEX2_FULL_COEXISTENCE")
                        retained = cands[-1]
                        print(f"[coex]   *** FULL COEXISTENCE at epoch {ep}")
                        break
                    label = ("COEX1_C_ZERO_PRESERVATION_FAIL" if stage == "coex1"
                             else "COEX2_C_ZERO_NO_PRESERVED_SOLUTION_FOUND")
                    retained = cands[-1]
                    if stage == "coex1":
                        print("[coex]   C=0 but preservation failed; "
                              "stopping COEX-1 for this seed")
                        break
                    print("[coex]   C=0 but preservation failed; continuing "
                          "to budget for the MIN_LP_C0 candidate")
    # ---- COEX-2 predeclared MIN_LP selection over post-FIRST_C0 candidates
    if stage == "coex2" and first_c0 is not None and \
            label == "COEX2_C_ZERO_NO_PRESERVED_SOLUTION_FOUND":
        later = [c for c in cands if c["epoch"] > first_c0["epoch"]]
        if later:
            best = min(later, key=lambda c: (c["L_P"], c["epoch"]))
            st = torch.load(best["path"], map_location="cpu",
                            weights_only=False)["state"]
            model = _isolated_model(tr, "p_last_hinge", st)
            r1, r2, det = verify_official_twice(tr, model, idx)
            bat = full_battery(tr, model, idx)
            gd = guards(bat, base)
            if not gd["wm_invariant"]:
                print("[coex] IMPLEMENTATION_BUG_HARD_ABORT: WM changed")
                return 9
            append_tsv(os.path.join(out, "coex_compatibility.tsv"),
                       [{"stage": stage, "seed": seed, "epoch": best["epoch"],
                         "candidate": "MIN_LP_C0", "official_eval_1": r1,
                         "official_eval_2": r2, "deterministic": int(det),
                         "L_P": best["L_P"], **bat,
                         **{k: int(v) if isinstance(v, bool) else v
                            for k, v in gd.items()}}])
            if gd["full_coexistence"] and r1 == 0 and r2 == 0:
                label = "COEX2_FULL_COEXISTENCE"
                retained = best
                print(f"[coex]   *** FULL COEXISTENCE at MIN_LP_C0 "
                      f"(epoch {best['epoch']}, L_P={best['L_P']:.3e})")

    sha_after = sha256_file(ck)
    if sha_before != sha_after:
        print("[coex] HARD STOP: source checkpoint mutated")
        return 8
    append_tsv(os.path.join(out, "coex_trajectory.tsv"), traj)
    append_tsv(os.path.join(out, "coex_candidates.tsv"), cands)
    append_tsv(os.path.join(out, "coex_runs.tsv"), [{
        "stage": stage, "seed": seed, "label": label,
        "objective": "L_C" if not use_p else "L_C + L_P",
        "margin_m": MARGIN_M, "lr": LR, "batch": BATCH,
        "max_epochs": MAX_EPOCHS, "init": "source_head",
        "first_c0_epoch": first_c0["epoch"] if first_c0 else None,
        "n_c0_candidates": len(cands),
        "retained_tag": retained["tag"] if retained else None,
        "retained_epoch": retained["epoch"] if retained else None,
        "retained_L_P": retained["L_P"] if retained else None,
        "final_c_errors": traj[-1]["c_errors"] if traj else None,
        "best_c_errors": min((t["c_errors"] for t in traj), default=None),
        "source_sha_unchanged": int(sha_before == sha_after),
        "elapsed_s": traj[-1]["elapsed_s"] if traj else None}])
    assert label in LABELS_ALLOWED, label
    print(f"[coex] {stage} seed {seed}: {label}")
    del tr
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("baselines", "h0-src", "drift-check", "fit"):
        p = sub.add_parser(name)
        p.add_argument("--out-dir", required=True)
        p.add_argument("--archive-runs", required=True)
        p.add_argument("--glove", required=True)
        p.add_argument("--rep-dir", default="")
        p.add_argument("--prev-probe", default="")
        if name == "fit":
            p.add_argument("--stage", required=True, choices=STAGES)
            p.add_argument("--seed", type=int, required=True, choices=SEEDS)
    a = ap.parse_args(argv)
    return {"baselines": cmd_baselines, "h0-src": cmd_h0_src,
            "drift-check": cmd_drift_check, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
