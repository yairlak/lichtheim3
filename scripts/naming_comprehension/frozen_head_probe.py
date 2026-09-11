"""Frozen semantic-head causal probe on the SETTLE u3600 control endpoints.

Bottleneck localization, NOT final certification.  Nothing here resumes joint
training, mutates a source checkpoint, or reuses the joint run's optimizer
state.  Derived heads live only under the probe output tree.

ARCHITECTURE (verified, not assumed).  The deployed head is an MLP:

    pooled(512) -> Linear(512,512) -> GELU -> Linear(512,300) -> cosine retrieval

so `to_semantic` is NOT a linear readout.  Three probes are defined on frozen
upstream parameters:

  P-last : train ltm.to_semantic.2 only, on frozen phi = GELU(W0 pooled + b0)
  P-lin  : train a FRESH diagnostic Linear(512,300) on raw frozen pooled
  P-head : train the whole existing head (to_semantic.0 and .2) on pooled

MATHEMATICAL STATUS.  For fixed features x and affine z = A x, strict top-1
is  z^T(ghat_i - ghat_j) > 0  for all j != i (bank rows are normalised and the
common 1/||z|| cancels; z = 0 fails every constraint).  Over this FINITE
system strict feasibility rescales to >= 1.  Hence:

  * strict feasibility for P-last / P-lin is a LINEAR (convex) system;
  * the unnormalised hardest-negative hinge is CONVEX in the parameters;
  * the historical cosine-CE is NONCONVEX even for P-last, because it passes
    through q = z/||z||.
  * P-head is nonconvex regardless (W0 and GELU move).

P-last is therefore never described as globally "exactly convex".

NO INFEASIBILITY CERTIFICATE is attempted: for <= 513 linearly independent
augmented features an affine map can hit arbitrary outputs, so small residual
subsets are not infeasibility witnesses.  The only labels used are
FEASIBLE_VERIFIED and EMPIRICAL_FAILURE_NO_CERTIFICATE.  A convex objective
does not make finite-budget Adam exhaustive, so a hinge failure is still
empirical.

WITNESS.  With margin target 1, a total hinge SUM < 1 implies every raw margin
is > 0.  That is reported as supplementary evidence only; the primary
existence criterion is 0/27,981 from the OFFICIAL evaluator run through a
reloaded isolated model.
"""
from __future__ import annotations

import argparse
import copy
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

# ------------------------------------------------------------- frozen config
SEEDS = (19, 20, 21, 22)
RUN = "final_settle_ctrl_h512_s{seed}"
STEP = 10_000_800
BASELINE_C_ERRORS = {19: 34, 20: 32, 21: 25, 22: 29}      # official, must match
N_COMP_POP, N_BANK, SEM_DIM, POOLED_DIM = 27_981, 29_571, 300, 512

TAU = 0.10
MARGIN_TARGET = 1.0
LR = 1e-3
BATCH = 1024
MAX_EPOCHS = 300
EVAL_EVERY = 5
DATA_SEED = 12345
INIT_SEED = 12345
# Central steering preference: no fragile no-improvement early stop.  A fit
# runs the full fixed budget unless a verified strict C = 0 occurs.
EARLY_STOP_RULE = "none: fixed 300-epoch budget unless verified strict C == 0"

CANON = {
    "comprehension_population_n": N_COMP_POP,
    "retrieval_bank_n": N_BANK,
    "comprehension_population_sha256":
        "10c2f06eda769bf620ca3dbb9889204e4431cac2bfe0d0f5dd37fa4df2bb9f50",
    "lexicon_sha256":
        "ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66",
    "glove_sha256":
        "91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed",
}

PROBES = ("p_last_hinge", "p_last_ce", "p_lin", "p_head_ce", "p_head_hinge")
TRAINABLE = {
    "p_last_hinge": ("ltm.to_semantic.2.weight", "ltm.to_semantic.2.bias"),
    "p_last_ce": ("ltm.to_semantic.2.weight", "ltm.to_semantic.2.bias"),
    "p_head_ce": ("ltm.to_semantic.0.weight", "ltm.to_semantic.0.bias",
                  "ltm.to_semantic.2.weight", "ltm.to_semantic.2.bias"),
    "p_head_hinge": ("ltm.to_semantic.0.weight", "ltm.to_semantic.0.bias",
                     "ltm.to_semantic.2.weight", "ltm.to_semantic.2.bias"),
    "p_lin": ("diag_linear.weight", "diag_linear.bias"),
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_tensor(t: torch.Tensor) -> str:
    a = t.detach().cpu().contiguous()
    return hashlib.sha256(a.numpy().tobytes()).hexdigest()


def write_tsv(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        open(path, "w").write("")
        return
    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[probe] wrote {os.path.basename(path)} ({len(rows)} rows)")


def append_tsv(path: str, rows: List[dict]) -> None:
    old = []
    if os.path.exists(path) and os.path.getsize(path):
        with open(path, encoding="utf-8") as f:
            old = list(csv.DictReader(f, delimiter="\t"))
    write_tsv(path, old + rows)


def ckpt_path(archive_runs: str, seed: int) -> str:
    return os.path.join(archive_runs, RUN.format(seed=seed), "checkpoints",
                        f"step_{STEP:08d}.pt")


def feat_path(out_dir: str, seed: int) -> str:
    return os.path.join(out_dir, "representations", f"seed{seed}.pt")


def head_path(out_dir: str, seed: int, probe: str) -> str:
    return os.path.join(out_dir, "derived_heads", f"seed{seed}_{probe}.pt")


# ========================================================= official evaluator
def official_strict_errors(s_hat: torch.Tensor, bank_raw: torch.Tensor,
                           idx: List[int], batch: int = 512) -> dict:
    """THE canonical evaluator -- `frozen_probe.comprehension_metrics`, the
    same function `train_tasks.evaluate_comprehension_subset` calls."""
    from scripts.naming_comprehension.frozen_probe import comprehension_metrics
    m = comprehension_metrics(s_hat.cpu().float(), bank_raw.cpu().float(),
                              idx, batch)
    n_err = int((m["top1"] == 0).sum())
    return {"errors": n_err, "n": len(idx), "metrics": m}


def build_trainer(ck_path: str, device: str, glove: str):
    from scripts.naming_comprehension.base123_error_audit import build
    return build(ck_path, device, glove_path=glove)


# =============================================================== STAGE 0 ====
def cmd_extract(a) -> int:
    """Frozen-representation extraction + baseline reconciliation.

    Captures `pooled` (the exact input to to_semantic) with a forward
    pre-hook during the canonical encode, recomputes phi = GELU(W0 p + b0),
    and asserts W2 phi + b2 reproduces the deployed s_hat.  Aborts unless the
    official strict C count equals the frozen baseline for every seed.
    """
    os.makedirs(a.out_dir, exist_ok=True)
    manifest, hashes = [], []
    for seed in SEEDS:
        ck = ckpt_path(a.archive_runs, seed)
        before = sha256_file(ck)
        tr, ckd = build_trainer(ck, "cpu", a.glove)
        model = tr.model
        model.eval()
        idx = list(tr.comp_idx)
        if len(idx) != N_COMP_POP:
            print(f"[probe] HARD STOP: C population {len(idx)} != {N_COMP_POP}")
            return 2
        if int(tr.bank_raw.shape[0]) != N_BANK:
            print(f"[probe] HARD STOP: bank {tr.bank_raw.shape[0]} != {N_BANK}")
            return 2
        if ckd.get("comprehension_population_sha256") != \
                CANON["comprehension_population_sha256"]:
            print("[probe] HARD STOP: C population hash mismatch")
            return 2

        from scripts.naming_comprehension.frozen_probe import encode_all
        forms = [tr.entries[i].phonemes for i in idx]
        buf: List[torch.Tensor] = []
        h = model.ltm.to_semantic.register_forward_pre_hook(
            lambda mod, inp: buf.append(inp[0].detach().clone()))
        with torch.no_grad():
            s_hat = encode_all(model, tr.vocab, forms, "cpu", 512)
        h.remove()
        pooled = torch.cat(buf, 0)
        if pooled.shape != (N_COMP_POP, POOLED_DIM):
            print(f"[probe] HARD STOP: pooled shape {tuple(pooled.shape)}")
            return 2

        ts = model.ltm.to_semantic
        with torch.no_grad():
            phi = F.gelu(ts[0](pooled))
            z = ts[2](phi)
        dev = float((z - s_hat).abs().max())
        if dev > 1e-4:
            print(f"[probe] HARD STOP: phi path deviates from s_hat by {dev:.3e}")
            return 2

        base = official_strict_errors(s_hat, tr.bank_raw, idx)
        if base["errors"] != BASELINE_C_ERRORS[seed]:
            print(f"[probe] HARD STOP: seed {seed} baseline "
                  f"{base['errors']} != {BASELINE_C_ERRORS[seed]}")
            return 3
        # determinism: a second independent encode must give identical s_hat
        with torch.no_grad():
            s2 = encode_all(model, tr.vocab, forms, "cpu", 512)
        if not torch.equal(s_hat, s2):
            print(f"[probe] HARD STOP: extraction not deterministic (seed {seed})")
            return 4

        os.makedirs(os.path.dirname(feat_path(a.out_dir, seed)), exist_ok=True)
        head0 = {k: v.detach().clone() for k, v in ts.state_dict().items()}
        torch.save({"seed": seed, "target_idx": idx, "pooled": pooled,
                    "phi": phi, "s_hat_reference": s_hat,
                    "bank_raw": tr.bank_raw.cpu(), "head_state": head0,
                    "checkpoint": ck, "checkpoint_sha256": before},
                   feat_path(a.out_dir, seed))
        after = sha256_file(ck)
        if before != after:
            print(f"[probe] HARD STOP: checkpoint mutated: {ck}")
            return 5
        manifest.append({
            "seed": seed, "checkpoint": ck, "checkpoint_sha256": before,
            "n_targets": len(idx), "bank_n": int(tr.bank_raw.shape[0]),
            "pooled_shape": f"{tuple(pooled.shape)}",
            "phi_shape": f"{tuple(phi.shape)}",
            "pooled_sha256": sha256_tensor(pooled),
            "phi_sha256": sha256_tensor(phi),
            "target_idx_sha256": hashlib.sha256(
                ",".join(map(str, idx)).encode()).hexdigest(),
            "baseline_c_errors": base["errors"],
            "baseline_expected": BASELINE_C_ERRORS[seed],
            "baseline_reconciled": 1,
            "phi_path_max_dev_vs_s_hat": f"{dev:.3e}",
            "deterministic_reencode": 1,
            "dtype": str(pooled.dtype), "device": "cpu"})
        hashes.append({"seed": seed, "checkpoint": ck,
                       "sha256_before": before, "sha256_after": after,
                       "unchanged": int(before == after)})
        print(f"[probe] seed {seed}: baseline C={base['errors']} OK, "
              f"pooled/phi extracted, checkpoint byte-identical")
        del tr, model
    write_tsv(os.path.join(a.out_dir, "representation_manifest.tsv"), manifest)
    write_tsv(os.path.join(a.out_dir, "source_checkpoint_hashes.tsv"), hashes)
    return 0


# ============================================================ probe modules
class LastLayer(nn.Module):
    """P-last: trains ltm.to_semantic.2 on frozen phi."""
    def __init__(self, head_state: dict):
        super().__init__()
        self.lin = nn.Linear(POOLED_DIM, SEM_DIM)
        with torch.no_grad():
            self.lin.weight.copy_(head_state["2.weight"])
            self.lin.bias.copy_(head_state["2.bias"])

    def forward(self, phi): return self.lin(phi)

    def export(self) -> dict:
        return {"2.weight": self.lin.weight.detach().cpu().clone(),
                "2.bias": self.lin.bias.detach().cpu().clone()}


class FullHead(nn.Module):
    """P-head: trains the whole existing Linear->GELU->Linear on pooled."""
    def __init__(self, head_state: dict):
        super().__init__()
        self.l0 = nn.Linear(POOLED_DIM, POOLED_DIM)
        self.l2 = nn.Linear(POOLED_DIM, SEM_DIM)
        with torch.no_grad():
            self.l0.weight.copy_(head_state["0.weight"])
            self.l0.bias.copy_(head_state["0.bias"])
            self.l2.weight.copy_(head_state["2.weight"])
            self.l2.bias.copy_(head_state["2.bias"])

    def forward(self, pooled): return self.l2(F.gelu(self.l0(pooled)))

    def export(self) -> dict:
        return {"0.weight": self.l0.weight.detach().cpu().clone(),
                "0.bias": self.l0.bias.detach().cpu().clone(),
                "2.weight": self.l2.weight.detach().cpu().clone(),
                "2.bias": self.l2.bias.detach().cpu().clone()}


class DiagLinear(nn.Module):
    """P-lin: a FRESH diagnostic Linear(512,300) on raw pooled.  Deterministic
    Xavier-uniform init at seed 12345.  This is NOT the deployed head."""
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(INIT_SEED)
        self.diag_linear = nn.Linear(POOLED_DIM, SEM_DIM)
        with torch.no_grad():
            w = torch.empty(SEM_DIM, POOLED_DIM)
            nn.init.xavier_uniform_(w, generator=g) if "generator" in \
                nn.init.xavier_uniform_.__code__.co_varnames else None
            if w.abs().sum() == 0:                       # torch w/o generator kw
                bound = (6.0 / (SEM_DIM + POOLED_DIM)) ** 0.5
                w = (torch.rand(SEM_DIM, POOLED_DIM, generator=g) * 2 - 1) * bound
            self.diag_linear.weight.copy_(w)
            self.diag_linear.bias.zero_()

    def forward(self, pooled): return self.diag_linear(pooled)

    def export(self) -> dict:
        return {"weight": self.diag_linear.weight.detach().cpu().clone(),
                "bias": self.diag_linear.bias.detach().cpu().clone()}


def make_module(probe: str, head_state: dict) -> nn.Module:
    if probe.startswith("p_last"):
        return LastLayer(head_state)
    if probe.startswith("p_head"):
        return FullHead(head_state)
    if probe == "p_lin":
        return DiagLinear()
    raise ValueError(probe)


def features_for(probe: str, blob: dict) -> torch.Tensor:
    return blob["phi"] if probe.startswith("p_last") else blob["pooled"]


# =============================================================== objectives
def hinge_terms(z: torch.Tensor, bank_n: torch.Tensor,
                tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unnormalised hardest-negative margin over the FULL bank.

    margin_i = z_i . ghat_target - max_{j != target} z_i . ghat_j
    The target column is excluded by masking it to -inf before the max.
    """
    scores = z @ bank_n.t()                                   # (B, N_BANK)
    tgt_score = scores.gather(1, tgt.view(-1, 1)).squeeze(1)
    masked = scores.scatter(1, tgt.view(-1, 1), float("-inf"))
    best_wrong = masked.max(dim=1).values
    margin = tgt_score - best_wrong
    return margin, F.relu(MARGIN_TARGET - margin)


def ce_loss(z: torch.Tensor, bank_n: torch.Tensor,
            tgt: torch.Tensor) -> torch.Tensor:
    """The historical normalised cosine cross-entropy (tau = 0.10).

    NONCONVEX in the parameters because of q = z/||z||.
    """
    q = F.normalize(z, dim=-1)
    return F.cross_entropy(q @ bank_n.t() / TAU, tgt)


def witness(z_all: torch.Tensor, bank_n: torch.Tensor,
            tgt_all: torch.Tensor, chunk: int = 2048) -> dict:
    """Full-population margin statistics; hinge SUM < 1 => all margins > 0."""
    mins, hs, margins = [], 0.0, []
    for lo in range(0, z_all.shape[0], chunk):
        m, h = hinge_terms(z_all[lo:lo + chunk], bank_n, tgt_all[lo:lo + chunk])
        margins.append(m.detach().cpu())
        mins.append(float(m.min()))
        hs += float(h.sum())
    mm = torch.cat(margins)
    q = torch.quantile(mm.double(), torch.tensor([0., .1, .25, .5, .75, .9, 1.],
                                                 dtype=torch.float64))
    return {"min_raw_margin": round(min(mins), 8),
            "n_margin_le_0": int((mm <= 0).sum()),
            "hinge_sum": round(hs, 6), "hinge_mean": round(hs / mm.numel(), 8),
            "margin_q0": round(float(q[0]), 8), "margin_q10": round(float(q[1]), 8),
            "margin_q25": round(float(q[2]), 8), "margin_q50": round(float(q[3]), 8),
            "margin_q75": round(float(q[4]), 8), "margin_q90": round(float(q[5]), 8),
            "margin_q100": round(float(q[6]), 8),
            "witness_all_margins_positive": int(hs < MARGIN_TARGET)}


# ==================================================================== FIT ===
def cmd_fit(a) -> int:
    """One fixed fitting procedure.  No LR ladder, no restarts, no schedule
    search, and the configuration is never altered after seeing results."""
    probe, seed = a.probe, a.seed
    blob = torch.load(feat_path(a.out_dir, seed), map_location="cpu",
                      weights_only=False)
    dev = torch.device(a.device)
    x = features_for(probe, blob).to(dev)
    bank_n = F.normalize(blob["bank_raw"].float(), dim=-1).to(dev)
    idx = list(blob["target_idx"])
    tgt = torch.tensor(idx, dtype=torch.long, device=dev)
    mod = make_module(probe, blob["head_state"]).to(dev)

    names = tuple(n for n, p in mod.named_parameters() if p.requires_grad)
    print(f"[probe] {probe} seed {seed}: trainable = {names}")
    opt = torch.optim.Adam(mod.parameters(), lr=LR)   # fresh; never the joint one
    use_ce = probe.endswith("_ce")

    g = torch.Generator().manual_seed(DATA_SEED)
    n = x.shape[0]
    traj, best, best_ep, zero_ep = [], None, None, None
    t0 = time.time()
    status = "BUDGET_EXHAUSTED"

    for ep in range(1, MAX_EPOCHS + 1):
        perm = torch.randperm(n, generator=g).to(dev)
        mod.train()
        tot, nb = 0.0, 0
        for lo in range(0, n, BATCH):
            b = perm[lo:lo + BATCH]
            z = mod(x[b])
            if use_ce:
                loss = ce_loss(z, bank_n, tgt[b])
            else:
                _, h = hinge_terms(z, bank_n, tgt[b])
                loss = h.mean()                       # MEAN is optimised
            if not torch.isfinite(loss):
                print(f"[probe] NUMERICAL_FAILURE at epoch {ep}")
                status = "NUMERICAL_FAILURE"
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()); nb += 1
        if status == "NUMERICAL_FAILURE":
            break

        if ep % EVAL_EVERY == 0 or ep == 1 or ep == MAX_EPOCHS:
            mod.eval()
            with torch.no_grad():
                z_all = torch.cat([mod(x[lo:lo + 4096])
                                   for lo in range(0, n, 4096)], 0)
                w = witness(z_all, bank_n, tgt)
            res = official_strict_errors(z_all, blob["bank_raw"], idx)
            pn = float(sum((p.detach() ** 2).sum() for p in mod.parameters()) ** 0.5)
            mx = float(max(p.detach().abs().max() for p in mod.parameters()))
            row = {"probe": probe, "seed": seed, "epoch": ep,
                   "train_loss": round(tot / max(nb, 1), 8),
                   "c_errors": res["errors"], "param_l2": round(pn, 4),
                   "param_absmax": round(mx, 6),
                   "elapsed_s": round(time.time() - t0, 1), **w}
            traj.append(row)
            print(f"[probe]   ep{ep:>3} loss={row['train_loss']:.6f} "
                  f"C_err={res['errors']:>4} min_margin={w['min_raw_margin']:+.5f} "
                  f"hinge_sum={w['hinge_sum']:.3f} |W|={pn:.1f}")
            if best is None or res["errors"] < best:
                best, best_ep = res["errors"], ep
            if res["errors"] == 0:
                zero_ep = ep
                status = "ZERO_REACHED"
                os.makedirs(os.path.dirname(head_path(a.out_dir, seed, probe)),
                            exist_ok=True)
                torch.save({"probe": probe, "seed": seed, "epoch": ep,
                            "state": mod.export(), "trainable_names": names},
                           head_path(a.out_dir, seed, probe))
                print(f"[probe]   *** strict C = 0 at epoch {ep}; head saved")
                break

    append_tsv(os.path.join(a.out_dir, "probe_trajectory.tsv"), traj)
    final = traj[-1] if traj else {}
    append_tsv(os.path.join(a.out_dir, "probe_runs.tsv"), [{
        "probe": probe, "seed": seed, "status": status,
        "objective": "historical_cosine_CE_tau0.10" if use_ce
                     else "unnormalised_hardest_negative_hinge_mean",
        "convexity": "NONCONVEX (q=z/||z||)" if use_ce else (
            "convex in params" if probe != "p_head_hinge"
            else "NONCONVEX (W0 and GELU move)"),
        "init": "checkpoint_head" if probe != "p_lin"
                else f"xavier_uniform_seed{INIT_SEED}",
        "trainable": ",".join(names), "lr": LR, "batch": BATCH,
        "max_epochs": MAX_EPOCHS, "early_stop_rule": EARLY_STOP_RULE,
        "initial_c_errors": traj[0]["c_errors"] if traj else None,
        "best_c_errors": best, "best_epoch": best_ep,
        "final_c_errors": final.get("c_errors"),
        "first_zero_epoch": zero_ep,
        "min_raw_margin": final.get("min_raw_margin"),
        "hinge_sum": final.get("hinge_sum"),
        "witness_all_margins_positive": final.get("witness_all_margins_positive"),
        "param_l2": final.get("param_l2"),
        "head_saved": int(zero_ep is not None),
        "elapsed_s": final.get("elapsed_s"), "device": a.device}])
    print(f"[probe] {probe} seed {seed}: {status} (best C={best})")
    return 0


# ================================================= VERIFY + COMPATIBILITY ===
def _isolated_model(tr, probe: str, state: dict):
    """An isolated COPY of the model carrying the derived head.  The source
    trainer's model is never modified, and nothing is written to disk."""
    model = copy.deepcopy(tr.model)
    if probe == "p_lin":
        lin = nn.Linear(POOLED_DIM, SEM_DIM)
        with torch.no_grad():
            lin.weight.copy_(state["weight"]); lin.bias.copy_(state["bias"])
        model.ltm.to_semantic = nn.Sequential(lin)     # DIAGNOSTIC ARCHITECTURE
    else:
        sd = model.ltm.to_semantic.state_dict()
        for k, v in state.items():
            sd[k] = v.clone()
        model.ltm.to_semantic.load_state_dict(sd)
    model.eval()
    return model


def cmd_verify(a) -> int:
    """Success verification through the FULL model path, twice."""
    from scripts.naming_comprehension.frozen_probe import encode_all
    rows = []
    for seed in SEEDS:
        for probe in PROBES:
            p = head_path(a.out_dir, seed, probe)
            if not os.path.exists(p):
                continue
            ck = ckpt_path(a.archive_runs, seed)
            before = sha256_file(ck)
            saved = torch.load(p, map_location="cpu", weights_only=False)
            tr, _ = build_trainer(ck, "cpu", a.glove)
            model = _isolated_model(tr, probe, saved["state"])
            idx = list(tr.comp_idx)
            forms = [tr.entries[i].phonemes for i in idx]
            with torch.no_grad():
                s1 = encode_all(model, tr.vocab, forms, "cpu", 512)
            r1 = official_strict_errors(s1, tr.bank_raw, idx)
            with torch.no_grad():
                s2 = encode_all(model, tr.vocab, forms, "cpu", 512)
            r2 = official_strict_errors(s2, tr.bank_raw, idx)
            after = sha256_file(ck)
            ok = (r1["errors"] == 0 and r2["errors"] == 0 and before == after)
            rows.append({
                "seed": seed, "probe": probe,
                "head_sha256": sha256_file(p),
                "official_eval_1_errors": r1["errors"],
                "official_eval_2_errors": r2["errors"],
                "deterministic_identical": int(torch.equal(s1, s2)),
                "source_sha_before": before, "source_sha_after": after,
                "source_unchanged": int(before == after),
                "verdict": "FEASIBLE_VERIFIED" if ok else "VERIFICATION_FAILED",
                "architecture": "P_LIN_DIAGNOSTIC_ARCHITECTURE"
                                if probe == "p_lin" else "deployed_architecture"})
            print(f"[probe] verify {probe} seed {seed}: "
                  f"{r1['errors']} / {r2['errors']} errors -> {rows[-1]['verdict']}")
            del tr, model
    append_tsv(os.path.join(a.out_dir, "probe_residuals.tsv"), rows)
    return 0


def cmd_compat(a) -> int:
    """Mandatory secondary check for every verified zero head: does the
    derived mapping preserve R / N / LTM?  Cannot revoke the primary result."""
    from scripts.naming_comprehension.train_tasks import (
        evaluate_comprehension_subset, evaluate_naming, repetition_snapshot)
    rows = []
    for seed in SEEDS:
        for probe in PROBES:
            p = head_path(a.out_dir, seed, probe)
            if not os.path.exists(p):
                continue
            ck = ckpt_path(a.archive_runs, seed)
            before = sha256_file(ck)
            saved = torch.load(p, map_location="cpu", weights_only=False)
            tr, _ = build_trainer(ck, "cpu", a.glove)
            derived = _isolated_model(tr, probe, saved["state"])
            idx = list(tr.comp_idx)
            rec = {"seed": seed, "probe": probe,
                   "label": "P_LIN_DIAGNOSTIC_ARCHITECTURE_COMPATIBILITY"
                            if probe == "p_lin" else "DEPLOYED_ARCHITECTURE_COMPATIBILITY"}
            c = evaluate_comprehension_subset(derived, tr.vocab, tr.entries,
                                              tr.bank_raw, idx, "cpu", 512)
            rec["c_top1"] = round(c["top1"], 8)
            rec["c_errors"] = int(round((1 - c["top1"]) * len(idx)))
            all_idx = list(range(len(tr.entries)))
            n_rep = len(all_idx)
            # canonical (forced-length) repetition, all three routes
            fr = repetition_snapshot(derived, tr.vocab, tr.entries, all_idx,
                                     tr.bank_raw, "cpu",
                                     include_teacher_forced=False)
            ex = fr["primary_readout"]["exact_match"]
            for route in ("full", "wm", "ltm"):
                rec[f"rep_canonical_{route}"] = round(float(ex[route]), 8)
            rec["rep_canonical_errors"] = int(round((1 - float(ex["full"])) * n_rep))
            # GENUINE free-AR repetition, via the trainer's own method on an
            # in-memory model swap (nothing is written; restored immediately)
            orig = tr.model
            try:
                tr.model = derived
                far = tr.free_ar_repetition(all_idx, routes=("full", "wm", "ltm"))
            finally:
                tr.model = orig
            for route in ("full", "wm", "ltm"):
                rec[f"rep_freear_{route}"] = round(float(far[route]), 8)
            rec["rep_freear_errors"] = int(round((1 - float(far["full"])) * n_rep))
            nm = evaluate_naming(derived, tr.vocab, tr.entries, tr.bank_raw,
                                 all_idx, "cpu", 256)
            rec["naming_exact"] = round(float(nm.get("exact_match", 0.0)), 8)
            rec["naming_errors"] = int(round(
                (1 - float(nm.get("exact_match", 0.0))) * n_rep))
            rec["source_unchanged"] = int(before == sha256_file(ck))
            rows.append(rec)
            print(f"[probe] compat {probe} seed {seed}: C_err={rec['c_errors']} "
                  f"naming={rec['naming_exact']:.6f}")
            del tr, derived
    append_tsv(os.path.join(a.out_dir, "compatibility_eval.tsv"), rows)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "fit", "verify", "compat"):
        p = sub.add_parser(name)
        p.add_argument("--out-dir", required=True)
        p.add_argument("--archive-runs", default="")
        p.add_argument("--glove", default="data/glove.6B.300d.txt")
        p.add_argument("--device", default="cpu")
        if name == "fit":
            p.add_argument("--probe", required=True, choices=PROBES)
            p.add_argument("--seed", type=int, required=True, choices=SEEDS)
    a = ap.parse_args(argv)
    return {"extract": cmd_extract, "fit": cmd_fit,
            "verify": cmd_verify, "compat": cmd_compat}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
