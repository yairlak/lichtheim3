#!/usr/bin/env python3
"""READ-ONLY C-align gradient diagnostic (contract: paper_programme/c_align_causal_pilot/
C_ALIGN_GRADIENT_DIAGNOSTIC_CONTRACT.md).

No optimizer.step(), no parameter or optimizer-state update, no .grad writes (torch.autograd.grad
only).  Hard-stops on any parameter / optimizer-state hash change.  Run from the worktree root:

    python3 scripts/c_align_design/c_align_gradient_diagnostic.py --out <new dir>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from losses import alignment_loss, total_loss  # noqa: E402
from scripts.naming_comprehension.train_joint_scratch import LAMBDA_C, TAU, build_batch  # noqa: E402
from scripts.naming_comprehension.train_tasks import comprehension_forward, retrieval_loss  # noqa: E402

REPO_PARENT = "/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3"
MANIFEST_TSV = "paper_programme/gating_route_diagnostics/checkpoint_manifest.tsv"
STATES = {
    "W3_SRC": {"source_sha256": "a5f21de9edb4b2090cb144c666ca4bd8e1f0572333000facb13f6c51cfaad76c",
               "composite": "6d7282854323b810727cd9c0c90cace6dd8a54d306408cafaa8c40c5455d10a0",
               "params_sha256": "4a8d4807f1a1928faebebf696d41f4c222b3997b52432d7015fc038881eac785"},
    "W4_SRC": {"source_sha256": "0657f41030e54eac11816116af6c76b66d597c393fc72579a3755401c9dc79f3",
               "composite": "32928707acc81af215faa829ad94bba566804982bc192bb5116ea32d43d56e29",
               "params_sha256": "004dda2d9a846ce194d4b0b3f7adc408d5c9d909aadbe413a35e96ceaa411974"},
}
LEXICON_SHA256 = "ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66"
GLOVE_SHA256 = "91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed"
C_POP_N = 27981
C_POP_SHA256 = "10c2f06eda769bf620ca3dbb9889204e4431cac2bfe0d0f5dd37fa4df2bb9f50"
BATCH = 64
W_REFERENCE = 0.1
LR = {"repetition": 3e-5, "naming": 3e-5, "comprehension": 1e-4}
BLOCK_PREFIXES = {
    "phon_embed": ("phon_embed.",),
    "ltm_encoder": ("ltm.encoder.",),
    "to_semantic_0": ("ltm.to_semantic.0.",),
    "to_semantic_2": ("ltm.to_semantic.2.",),
}
SEB = ("ltm_encoder", "to_semantic_0", "to_semantic_2")
COS_BLOCKS = ("ltm_encoder", "to_semantic_0", "to_semantic_2", "SEB", "phon_embed")
RHO_LOW, RHO_HIGH = 0.1, 10.0          # frozen "grossly disproportionate" band (contract §9)
EQUIV_REL_TOL = 1e-4                    # frozen R/C equivalence tolerance (contract §9)
QUANTILES = (0.05, 0.25, 0.75, 0.95)
NEAR_ORTHO = 0.1


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tensor_dict_sha256(items) -> str:
    h = hashlib.sha256()
    for name, t in items:
        h.update(str(name).encode())
        if torch.is_tensor(t):
            h.update(str(tuple(t.shape)).encode())
            h.update(t.detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(repr(t).encode())
    return h.hexdigest()


def optimizer_state_sha256(opt) -> str:
    items = []
    for pi, st in sorted(opt.state_dict()["state"].items()):
        for k in sorted(st):
            items.append((f"{pi}.{k}", st[k]))
    return tensor_dict_sha256(items)


# ------------------------------------------------------------------------- blocks
class Blocks:
    """Named, ordered parameter blocks (shared phon_embed counted once)."""

    def __init__(self, model):
        named = list(model.named_parameters())       # de-duplicated by identity
        self.params: Dict[str, List[torch.nn.Parameter]] = {}
        self.names: Dict[str, List[str]] = {}
        for b, prefixes in BLOCK_PREFIXES.items():
            sel = [(n, p) for n, p in named if n.startswith(prefixes)]
            if not sel:
                raise RuntimeError(f"HARD STOP: block {b} has no parameters")
            self.names[b] = [n for n, _ in sel]
            self.params[b] = [p for _, p in sel]
        self.all_params = [p for _, p in named]
        self.all_names = [n for n, _ in named]
        self.flat_order = [p for b in BLOCK_PREFIXES for p in self.params[b]]

    def split(self, grads) -> Dict[str, torch.Tensor]:
        """grads aligned with self.flat_order -> flattened vector per block (+ SEB)."""
        out, i = {}, 0
        for b in BLOCK_PREFIXES:
            parts = []
            for p in self.params[b]:
                g = grads[i]
                parts.append((torch.zeros_like(p) if g is None else g).detach().reshape(-1).double())
                i += 1
            out[b] = torch.cat(parts)
        out["SEB"] = torch.cat([out[b] for b in SEB])
        return out


def grad_blocks(loss: torch.Tensor, blocks: Blocks, retain: bool) -> Dict[str, torch.Tensor]:
    g = torch.autograd.grad(loss, blocks.flat_order, retain_graph=retain, allow_unused=True)
    return blocks.split(g)


def global_norm(loss: torch.Tensor, blocks: Blocks, retain: bool) -> float:
    g = torch.autograd.grad(loss, blocks.all_params, retain_graph=retain, allow_unused=True)
    sq = sum(float((x.detach().double() ** 2).sum()) for x in g if x is not None)
    return float(np.sqrt(sq))


def cos(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    na, nb = float(a.norm()), float(b.norm())
    if not (na > 0 and nb > 0 and np.isfinite(na) and np.isfinite(nb)):
        return None
    return float((a @ b) / (na * nb))


def clip_coef(norm: float, max_norm: float = 1.0) -> float:
    return float(min(1.0, max_norm / (norm + 1e-6)))


# ------------------------------------------------------------------------- per batch
def batch_metrics(model, tr, idx: List[int], blocks: Blocks, adam: Optional[Dict], device: str = "cpu") -> Dict:
    pad = tr.vocab.pad_id
    b = build_batch(tr.entries, tr.bank_raw, tr.vocab, idx, device)
    rec: Dict[str, object] = {"batch_n": len(idx), "first_item": int(idx[0]), "last_item": int(idx[-1])}
    # --- C step graph
    s_hat = comprehension_forward(model, b["enc_in"], b["enc_mask"])
    l_ret = retrieval_loss(s_hat, model.ltm.semantic_bank, b["bank_idx"], TAU)
    g_sem = b["semantic"]
    cos_term = 1.0 - F.cosine_similarity(s_hat, g_sem, dim=-1).mean()
    mse_term = 0.1 * F.mse_loss(s_hat, g_sem)
    l_align = alignment_loss(s_hat, g_sem)
    rec.update({"L_C_retrieval_raw": float(l_ret.detach()), "L_C_retrieval_effective": float(LAMBDA_C * l_ret.detach()),
                "L_C_align_cos_term": float(cos_term.detach()), "L_C_align_mse_term": float(mse_term.detach()),
                "L_C_align_unit": float(l_align.detach()),
                "L_C_align_components_sum_minus_total": float((cos_term + mse_term - l_align).detach())})
    g_ret_raw = grad_blocks(l_ret, blocks, retain=True)
    g_align = grad_blocks(l_align, blocks, retain=True)
    g_cos = grad_blocks(cos_term, blocks, retain=True)
    g_mse = grad_blocks(mse_term, blocks, retain=True)
    rec["global_norm_C_step_V6"] = global_norm(LAMBDA_C * l_ret, blocks, retain=True)
    rec["global_norm_C_step_ON_w0.1"] = global_norm(LAMBDA_C * l_ret + W_REFERENCE * l_align, blocks, retain=False)
    rec["clip_coef_C_step_V6"] = clip_coef(rec["global_norm_C_step_V6"])
    rec["clip_coef_C_step_ON_w0.1"] = clip_coef(rec["global_norm_C_step_ON_w0.1"])
    # --- R step graph on the SAME batch (full model forward, as in training)
    out = model(b["enc_in"], b["enc_mask"], b["dec_in"])
    parts = total_loss(out, b, tr.cfg.loss, pad, usage_prior=tr.cfg.gating.usage_prior)
    l_R_align = parts["align"]
    rec["L_R_align_unit"] = float(l_R_align.detach())
    rec["abs_diff_C_align_vs_R_align"] = abs(float(l_align.detach()) - float(l_R_align.detach()))
    g_R = grad_blocks(l_R_align, blocks, retain=True)
    rec["global_norm_R_step_total_excl_pool"] = global_norm(parts["total"], blocks, retain=False)
    rec["clip_coef_R_step_total_excl_pool"] = clip_coef(rec["global_norm_R_step_total_excl_pool"])
    # --- per block norms / flags
    for blk in list(BLOCK_PREFIXES) + ["SEB"]:
        gr, ga, gR = g_ret_raw[blk], g_align[blk], g_R[blk]
        n_ret = float(gr.norm())
        rec[f"{blk}__numel"] = int(gr.numel())
        for tag, g, nrm in (("ret_raw", gr, n_ret), ("align_unit", ga, float(ga.norm())), ("R_align_unit", gR, float(gR.norm()))):
            rec[f"{blk}__norm_{tag}"] = nrm
            rec[f"{blk}__finite_{tag}"] = int(bool(torch.isfinite(g).all()))
            rec[f"{blk}__allzero_{tag}"] = int(bool((g == 0).all()))
        rec[f"{blk}__norm_ret_eff"] = LAMBDA_C * n_ret
        rec[f"{blk}__norm_align_w0.1"] = W_REFERENCE * rec[f"{blk}__norm_align_unit"]
        eff = rec[f"{blk}__norm_ret_eff"]
        rec[f"{blk}__rho_ref_w0.1_over_ret_eff"] = rec[f"{blk}__norm_align_w0.1"] / eff if eff > 0 else float("nan")
        rec[f"{blk}__ratio_align_unit_over_ret_eff"] = rec[f"{blk}__norm_align_unit"] / eff if eff > 0 else float("nan")
        nR = rec[f"{blk}__norm_R_align_unit"]
        rec[f"{blk}__ratio_align_unit_over_R_align_unit"] = rec[f"{blk}__norm_align_unit"] / nR if nR > 0 else float("nan")
        rec[f"{blk}__equiv_max_abs_diff"] = float((ga - gR).abs().max())
        rec[f"{blk}__equiv_rel_l2_diff"] = float((ga - gR).norm() / ga.norm()) if float(ga.norm()) > 0 else float("nan")
        rec[f"{blk}__equiv_cos"] = cos(ga, gR)
        if blk in COS_BLOCKS:
            rec[f"{blk}__cos_ret_eff_vs_align_unit"] = cos(LAMBDA_C * gr, ga)
    rec["SEB__cos_ret_eff_vs_cos_term"] = cos(LAMBDA_C * g_ret_raw["SEB"], g_cos["SEB"])
    rec["SEB__cos_ret_eff_vs_mse_term"] = cos(LAMBDA_C * g_ret_raw["SEB"], g_mse["SEB"])
    rec["SEB__norm_cos_term"] = float(g_cos["SEB"].norm())
    rec["SEB__norm_mse_term"] = float(g_mse["SEB"].norm())
    # --- secondary Adam-preconditioned first-step proxy (SEB)
    if adam is not None:
        def pnorm(vec, lr):
            return float((lr * vec / adam["denom_SEB"]).norm())
        rec["adam_proxy_SEB__C_ret_eff"] = pnorm(LAMBDA_C * g_ret_raw["SEB"], LR["comprehension"])
        rec["adam_proxy_SEB__C_align_w0.1"] = pnorm(W_REFERENCE * g_align["SEB"], LR["comprehension"])
        rec["adam_proxy_SEB__R_align_unit"] = pnorm(g_R["SEB"], LR["repetition"])
        rec["adam_proxy_SEB__per_cycle_C_align_w0.1_x3"] = 3 * rec["adam_proxy_SEB__C_align_w0.1"]
        rec["adam_proxy_SEB__per_cycle_R_align_x1"] = rec["adam_proxy_SEB__R_align_unit"]
    return rec


def adam_denominator(tr, blocks: Blocks) -> Dict:
    """sqrt(v_hat)+eps per SEB parameter, mapped by optimizer order == model.parameters() order."""
    sd = tr.optim.state_dict()
    group = sd["param_groups"][0]
    beta2, eps = group["betas"][1], group["eps"]
    params = list(tr.model.parameters())
    if len(params) != len(group["params"]):
        raise RuntimeError("HARD STOP: optimizer/param count mismatch")
    pid = {id(p): i for i, p in enumerate(params)}
    parts = []
    for b in SEB:
        for p in blocks.params[b]:
            st = sd["state"][group["params"][pid[id(p)]]]
            if tuple(st["exp_avg_sq"].shape) != tuple(p.shape):
                raise RuntimeError("HARD STOP: optimizer state shape mismatch")
            step = float(st["step"])
            vhat = st["exp_avg_sq"].double() / (1.0 - beta2 ** step)
            parts.append((vhat.sqrt() + eps).reshape(-1))
    return {"denom_SEB": torch.cat(parts), "beta2": beta2, "eps": eps}


# ------------------------------------------------------------------------- summaries
def summarize(vals: List[Optional[float]]) -> Dict:
    x = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    out = {"n": int(x.size), "n_NA": int(len(vals) - x.size), "mean": float(x.mean()), "median": float(np.median(x)),
           "sd": float(x.std(ddof=1)) if x.size > 1 else None, "min": float(x.min()), "max": float(x.max())}
    for q in QUANTILES:
        out[f"q{int(round(q * 100)):02d}"] = float(np.quantile(x, q))
    return out


def cos_summary(vals) -> Dict:
    s = summarize(vals)
    x = np.array([v for v in vals if v is not None and np.isfinite(v)])
    if x.size:
        s["frac_lt_0"] = float((x < 0).mean())
        s["frac_abs_lt_0.1"] = float((np.abs(x) < NEAR_ORTHO).mean())
    return s


def state_summary(rows: List[Dict]) -> Dict:
    out: Dict[str, object] = {}
    for tag, sel in (("all_438", rows), ("full_size_437", [r for r in rows if r["batch_n"] == BATCH])):
        s: Dict[str, object] = {"n_batches": len(sel)}
        if not sel:
            out[tag] = s
            continue
        for k in ("L_C_retrieval_raw", "L_C_retrieval_effective", "L_C_align_cos_term", "L_C_align_mse_term",
                  "L_C_align_unit", "L_R_align_unit", "abs_diff_C_align_vs_R_align",
                  "global_norm_C_step_V6", "global_norm_C_step_ON_w0.1", "clip_coef_C_step_V6",
                  "clip_coef_C_step_ON_w0.1", "global_norm_R_step_total_excl_pool", "clip_coef_R_step_total_excl_pool",
                  "SEB__norm_cos_term", "SEB__norm_mse_term", "SEB__cos_ret_eff_vs_cos_term", "SEB__cos_ret_eff_vs_mse_term"):
            s[k] = summarize([r[k] for r in sel])
        for k in [k for k in sel[0] if k.startswith("adam_proxy")]:
            s[k] = summarize([r[k] for r in sel])
        for blk in list(BLOCK_PREFIXES) + ["SEB"]:
            bs = {}
            for m in ("norm_ret_raw", "norm_ret_eff", "norm_align_unit", "norm_align_w0.1", "norm_R_align_unit",
                      "rho_ref_w0.1_over_ret_eff", "ratio_align_unit_over_ret_eff",
                      "ratio_align_unit_over_R_align_unit", "equiv_max_abs_diff", "equiv_rel_l2_diff", "equiv_cos"):
                bs[m] = summarize([r[f"{blk}__{m}"] for r in sel])
            if blk in COS_BLOCKS:
                bs["cos_ret_eff_vs_align_unit"] = cos_summary([r[f"{blk}__cos_ret_eff_vs_align_unit"] for r in sel])
            bs["numel"] = sel[0][f"{blk}__numel"]
            bs["any_nonfinite"] = int(any(r[f"{blk}__finite_{t}"] == 0 for r in sel for t in ("ret_raw", "align_unit", "R_align_unit")))
            bs["n_batches_allzero_ret_raw"] = int(sum(r[f"{blk}__allzero_ret_raw"] for r in sel))
            bs["n_batches_allzero_align_unit"] = int(sum(r[f"{blk}__allzero_align_unit"] for r in sel))
            s[blk] = bs
        out[tag] = s
    return out


def weight_rule(summary: Dict, gates: Dict) -> Dict:
    med = {sid: summary[sid]["all_438"]["SEB"]["rho_ref_w0.1_over_ret_eff"]["median"] for sid in summary}
    equiv = {sid: summary[sid]["all_438"]["SEB"]["equiv_rel_l2_diff"]["median"] for sid in summary}
    pathology = []
    for sid, s in summary.items():
        a = s["all_438"]
        for blk in list(BLOCK_PREFIXES) + ["SEB"]:
            if a[blk]["any_nonfinite"]:
                pathology.append(f"{sid}:{blk}:nonfinite")
        if a["SEB"]["n_batches_allzero_ret_raw"] or a["SEB"]["n_batches_allzero_align_unit"]:
            pathology.append(f"{sid}:SEB:allzero")
        if not (equiv[sid] <= EQUIV_REL_TOL):
            pathology.append(f"{sid}:R_C_equivalence_failure")
    if not all(g.get("pass") for g in gates.values()):
        pathology.append("gate_failure")
    gross = {sid: (m < RHO_LOW or m > RHO_HIGH) for sid, m in med.items()}
    status = ("RECOMMEND_c_align_weight=0.1" if not any(gross.values()) and not pathology
              else "C_ALIGN_WEIGHT_STATUS=REQUIRES_CENTRAL_ARBITRATION")
    return {"rule": f"gross iff median_b rho_ref (SEB) < {RHO_LOW} or > {RHO_HIGH} in either state; "
                    f"pathology iff nonfinite/allzero/R-C equivalence median rel L2 > {EQUIV_REL_TOL}/gate failure",
            "median_rho_ref_SEB": med, "grossly_disproportionate": gross,
            "median_equiv_rel_l2_SEB": equiv, "pathology": pathology, "outcome": status}


# ------------------------------------------------------------------------- driver
def run(out_dir: str) -> int:
    if os.path.exists(out_dir):
        raise SystemExit(f"HARD STOP: output dir {out_dir} exists")
    if os.path.realpath(os.getcwd()) != os.path.realpath(ROOT):
        raise SystemExit("HARD STOP: run from the worktree root")
    from gate_x_lesion.identity import reconstructed_state_sha256
    from gate_x_lesion.hooks import state_dict_sha256
    from scripts.gating_diagnostics.run_gate_route_audit import build_state, load_manifest
    from scripts.naming_comprehension.train_tasks import subset_definition_hash, subset_records
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    rows_manifest = {r["state_id"]: r for r in load_manifest(os.path.join(ROOT, MANIFEST_TSV))}
    gates: Dict[str, Dict] = {}
    all_rows: List[Dict] = []
    summary: Dict[str, Dict] = {}
    inputs = {"lexicon_sha256": sha256_file(os.path.join(ROOT, "data/lexicon_en_glove_covered.tsv")),
              "glove_sha256": sha256_file(os.path.join(ROOT, "data/glove.6B.300d.txt"))}
    gates["G1_data"] = {"pass": inputs["lexicon_sha256"] == LEXICON_SHA256 and inputs["glove_sha256"] == GLOVE_SHA256, **inputs}
    for sid, exp in STATES.items():
        row = rows_manifest[sid]
        src = sha256_file(os.path.join(REPO_PARENT, row["artifact_path"]))
        g1 = {"source_sha256": src, "pass": src == exp["source_sha256"]
              and reconstructed_state_sha256(src, "") == exp["composite"]}
        tr, model, prov, _, _ = build_state(row, "cpu")
        blocks = Blocks(model)
        comp = list(tr.comp_idx)
        g1["c_pop_n"] = len(comp)
        g1["c_pop_sha256"] = subset_definition_hash(subset_records(tr.entries, comp, tr.vocab))
        g1["pass"] = g1["pass"] and len(comp) == C_POP_N and g1["c_pop_sha256"] == C_POP_SHA256
        gates[f"G1_identity_{sid}"] = g1
        p_before = state_dict_sha256(model)
        o_before = optimizer_state_sha256(tr.optim)
        grads_none_before = all(p.grad is None for p in model.parameters())
        gates[f"G5_noise_off_{sid}"] = {"pass": float(model.ltm.cfg.ventral_noise) == 0.0
                                        and float(model.wm.cfg.interference_noise) == 0.0}
        gates[f"G7_bank_{sid}"] = {"pass": bool(torch.equal(model.ltm.semantic_bank, F.normalize(tr.bank_raw, dim=-1)))}
        model.train(True)
        adam = adam_denominator(tr, blocks)
        batches = [comp[i:i + BATCH] for i in range(0, len(comp), BATCH)]
        rows = []
        for bi, idx in enumerate(batches):
            rec = batch_metrics(model, tr, idx, blocks, adam)
            rec = {"state_id": sid, "batch_index": bi, **rec}
            rows.append(rec)
            if bi % 50 == 0:
                print(f"{sid} batch {bi}/{len(batches)}", flush=True)
        r0 = batch_metrics(model, tr, batches[0], blocks, adam)
        det = all(r0[k] == rows[0][k] for k in ("L_C_retrieval_raw", "L_C_align_unit", "L_R_align_unit",
                                                 "SEB__norm_ret_raw", "SEB__norm_align_unit", "SEB__norm_R_align_unit"))
        gates[f"G6_determinism_{sid}"] = {"pass": bool(det)}
        p_after = state_dict_sha256(model)
        o_after = optimizer_state_sha256(tr.optim)
        gates[f"G2_params_{sid}"] = {"before": p_before, "after": p_after, "expected": exp["params_sha256"],
                                     "pass": p_before == p_after == exp["params_sha256"]}
        gates[f"G3_optimizer_state_{sid}"] = {"before": o_before, "after": o_after, "pass": o_before == o_after}
        gates[f"G4_no_grad_{sid}"] = {"pass": grads_none_before and all(p.grad is None for p in model.parameters())}
        summary[sid] = state_summary(rows)
        summary[sid]["adam_proxy_constants"] = {"beta2": adam["beta2"], "eps": adam["eps"], "lr": LR}
        all_rows.extend(rows)
        del tr, model
    rule = weight_rule(summary, gates)
    os.makedirs(out_dir)
    cols = list(all_rows[0].keys())
    with open(os.path.join(out_dir, "gradient_batch_metrics.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for r in all_rows:
            w.writerow(["NA" if r[c] is None else (repr(r[c]) if isinstance(r[c], float) else r[c]) for c in cols])
    json.dump({"gates": gates, "all_gates_pass": all(g["pass"] for g in gates.values()), "weight_rule": rule,
               "summary": summary, "constants": {"LAMBDA_C": LAMBDA_C, "TAU": TAU, "BATCH": BATCH, "W_REFERENCE": W_REFERENCE,
                                                 "RHO_BAND": [RHO_LOW, RHO_HIGH], "EQUIV_REL_TOL": EQUIV_REL_TOL,
                                                 "QUANTILES": QUANTILES, "NEAR_ORTHO_BIN": NEAR_ORTHO}},
              open(os.path.join(out_dir, "gradient_summary.json"), "w"), indent=1, sort_keys=True, default=str)
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    code = ["scripts/c_align_design/c_align_gradient_diagnostic.py", "losses.py",
            "scripts/naming_comprehension/train_joint_scratch.py", "scripts/naming_comprehension/train_tasks.py",
            "models/ltm_route.py", "models/dual_route.py", "scripts/gating_diagnostics/run_gate_route_audit.py",
            "paper_programme/c_align_causal_pilot/C_ALIGN_GRADIENT_DIAGNOSTIC_CONTRACT.md"]
    json.dump({"git_head": head, "states": STATES, "inputs": inputs,
               "code_sha256": {p: sha256_file(os.path.join(ROOT, p)) for p in code},
               "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu",
                               "deterministic_algorithms": True},
               "optimizer_step_calls": 0, "parameter_updates": 0},
              open(os.path.join(out_dir, "gradient_diagnostic_manifest.json"), "w"), indent=1, sort_keys=True)
    print("ALL_GATES_PASS" if all(g["pass"] for g in gates.values()) else "GATE_FAILURE", rule["outcome"])
    return 0 if all(g["pass"] for g in gates.values()) else 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return run(a.out)


if __name__ == "__main__":
    raise SystemExit(main())
