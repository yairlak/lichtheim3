"""Tests for the frozen semantic-head causal probe.

All 20 preregistered checks; no scientific fit may run with any of these
failing.
"""
from __future__ import annotations

import ast
import hashlib
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension import frozen_head_probe as m   # noqa: E402

SRC = os.path.join(ROOT, "scripts/naming_comprehension/frozen_head_probe.py")


def _head_state():
    g = torch.Generator().manual_seed(7)
    return {"0.weight": torch.randn(512, 512, generator=g) * 0.02,
            "0.bias": torch.randn(512, generator=g) * 0.02,
            "2.weight": torch.randn(300, 512, generator=g) * 0.02,
            "2.bias": torch.randn(300, generator=g) * 0.02}


# ---- 1/2/3: exact trainable parameter sets --------------------------------
def test_p_last_trains_exactly_the_final_layer():
    assert m.TRAINABLE["p_last_hinge"] == ("ltm.to_semantic.2.weight",
                                           "ltm.to_semantic.2.bias")
    assert m.TRAINABLE["p_last_ce"] == m.TRAINABLE["p_last_hinge"]
    mod = m.make_module("p_last_hinge", _head_state())
    names = {n for n, p in mod.named_parameters() if p.requires_grad}
    assert names == {"lin.weight", "lin.bias"}
    assert sum(p.numel() for p in mod.parameters()) == 300 * 512 + 300


def test_p_head_trains_exactly_both_head_layers():
    assert m.TRAINABLE["p_head_ce"] == ("ltm.to_semantic.0.weight",
                                        "ltm.to_semantic.0.bias",
                                        "ltm.to_semantic.2.weight",
                                        "ltm.to_semantic.2.bias")
    mod = m.make_module("p_head_ce", _head_state())
    names = {n for n, p in mod.named_parameters() if p.requires_grad}
    assert names == {"l0.weight", "l0.bias", "l2.weight", "l2.bias"}
    assert sum(p.numel() for p in mod.parameters()) == (512 * 512 + 512
                                                        + 300 * 512 + 300)


def test_p_lin_is_a_fresh_diagnostic_module_touching_no_source_param():
    mod = m.make_module("p_lin", _head_state())
    names = {n for n, p in mod.named_parameters() if p.requires_grad}
    assert names == {"diag_linear.weight", "diag_linear.bias"}
    # deterministic init, independent of the checkpoint head
    a = m.make_module("p_lin", _head_state()).diag_linear.weight
    b = m.make_module("p_lin", {k: v * 3 for k, v in _head_state().items()}
                      ).diag_linear.weight
    assert torch.equal(a, b)
    assert float(a.detach().abs().max()) > 0


# ---- 4/5/6/7: freezing, immutability, determinism -------------------------
def test_p_last_never_alters_the_frozen_first_layer():
    hs = _head_state()
    mod = m.make_module("p_last_hinge", hs)
    before = {k: v.clone() for k, v in hs.items()}
    x = torch.randn(8, 512)
    mod(x).sum().backward()
    torch.optim.Adam(mod.parameters(), lr=1e-3).step()
    for k in ("0.weight", "0.bias"):
        assert torch.equal(hs[k], before[k]), k
    out = mod.export()
    assert set(out) == {"2.weight", "2.bias"}


def test_module_forward_matches_the_deployed_head_arithmetic():
    hs = _head_state()
    ts = nn.Sequential(nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 300))
    with torch.no_grad():
        ts[0].weight.copy_(hs["0.weight"]); ts[0].bias.copy_(hs["0.bias"])
        ts[2].weight.copy_(hs["2.weight"]); ts[2].bias.copy_(hs["2.bias"])
    x = torch.randn(16, 512)
    with torch.no_grad():
        deployed = ts(x)
        phi = F.gelu(ts[0](x))
        via_last = m.make_module("p_last_hinge", hs)(phi)
        via_head = m.make_module("p_head_ce", hs)(x)
    assert torch.allclose(deployed, via_last, atol=1e-6)
    assert torch.allclose(deployed, via_head, atol=1e-6)


def test_sha_helpers_detect_any_change():
    t = torch.arange(10, dtype=torch.float32)
    h = m.sha256_tensor(t)
    t2 = t.clone(); t2[3] += 1e-6
    assert m.sha256_tensor(t2) != h
    assert m.sha256_tensor(t.clone()) == h


# ---- 8/9/10: population, bank, baseline ----------------------------------
def test_canonical_identity_constants():
    assert m.N_COMP_POP == 27_981 and m.N_BANK == 29_571
    assert m.BASELINE_C_ERRORS == {19: 34, 20: 32, 21: 25, 22: 29}
    assert m.CANON["comprehension_population_sha256"] == (
        "10c2f06eda769bf620ca3dbb9889204e4431cac2bfe0d0f5dd37fa4df2bb9f50")
    assert m.CANON["lexicon_sha256"].startswith("ae80918165e16b8c")
    assert m.CANON["glove_sha256"].startswith("91125602f730fea7")
    assert m.SEM_DIM == 300 and m.POOLED_DIM == 512


def test_extract_aborts_on_baseline_mismatch():
    """The reconciliation gate must be a hard abort, not a warning."""
    src = open(SRC, encoding="utf-8").read()
    assert 'if base["errors"] != BASELINE_C_ERRORS[seed]' in src
    assert "HARD STOP" in src
    assert "return 3" in src


# ---- 11/12: hardest-negative correctness ---------------------------------
def test_target_is_excluded_from_the_hardest_negative():
    bank = F.normalize(torch.eye(5), dim=-1)
    z = torch.tensor([[3.0, 1.0, 0.0, 0.0, 0.0]])      # target 0 is the max
    tgt = torch.tensor([0])
    margin, hinge = m.hinge_terms(z, bank, tgt)
    assert pytest.approx(float(margin), abs=1e-6) == 2.0   # 3 - 1, not 3 - 3
    assert pytest.approx(float(hinge), abs=1e-6) == 0.0


def test_margin_sign_convention_negative_when_wrong():
    bank = F.normalize(torch.eye(4), dim=-1)
    z = torch.tensor([[1.0, 5.0, 0.0, 0.0]])           # competitor 1 wins
    margin, hinge = m.hinge_terms(z, bank, torch.tensor([0]))
    assert float(margin) < 0
    assert float(hinge) == pytest.approx(1.0 - float(margin), abs=1e-6)


# ---- 13: rank 1 but strictly WRONG (tie against a lower index) -----------
def test_rank_one_can_still_be_strictly_wrong():
    from scripts.naming_comprehension.frozen_probe import comprehension_metrics
    bank = torch.zeros(3, 4)
    bank[0, 0] = 1.0
    bank[1, 0] = 1.0                     # index 1 is the TARGET, tied with 0
    bank[2, 1] = 1.0
    s_hat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    r = comprehension_metrics(s_hat, bank, [1], 8)
    assert int(r["target_rank"][0]) == 1          # strict '>' count + 1
    assert int(r["top1"][0]) == 0                 # argmax picks index 0 -> WRONG
    assert int(r["top1_idx"][0]) == 0


# ---- 14: the hinge witness ----------------------------------------------
def test_hinge_sum_below_one_implies_every_margin_positive():
    g = torch.Generator().manual_seed(3)
    bank = F.normalize(torch.randn(40, 6, generator=g), dim=-1)
    z = bank[:12] * 25.0                            # each item strongly on target
    tgt = torch.arange(12)
    w = m.witness(z, bank, tgt)
    assert w["hinge_sum"] < m.MARGIN_TARGET
    assert w["witness_all_margins_positive"] == 1
    assert w["min_raw_margin"] > 0 and w["n_margin_le_0"] == 0
    z2 = z.clone(); z2[0] = bank[5] * 25.0          # item 0 now retrieves item 5
    w2 = m.witness(z2, bank, tgt)
    assert w2["n_margin_le_0"] >= 1
    assert w2["hinge_sum"] >= m.MARGIN_TARGET
    assert w2["witness_all_margins_positive"] == 0


# ---- 15/16: labels and success criterion ---------------------------------
def test_empirical_failure_never_becomes_infeasibility():
    src = open(SRC, encoding="utf-8").read()
    assert "EMPIRICAL_FAILURE_NO_CERTIFICATE" in src
    assert "LINEAR_INFEASIBLE" not in src
    assert "certified_infeasible" not in src.lower()
    tree = ast.parse(src)
    strings = {n.value for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not any("INFEASIBL" in s.upper() and "NO_CERTIFICATE" not in s.upper()
                   and "infeasibility witnesses" not in s
                   for s in strings if len(s) < 80)


def test_success_requires_the_official_evaluator_through_the_full_model():
    src = open(SRC, encoding="utf-8").read()
    v = src.split("def cmd_verify", 1)[1].split("def cmd_compat", 1)[0]
    assert "encode_all" in v                    # full model path, not cached feats
    assert "official_strict_errors" in v
    assert v.count("official_strict_errors") >= 2      # evaluated twice
    assert "FEASIBLE_VERIFIED" in v
    assert 'r1["errors"] == 0 and r2["errors"] == 0' in v
    assert "before == after" in v               # source immutability in the verdict


# ---- 17/18/19/20: isolation, labelling, optimizer, writes ----------------
def test_derived_model_is_an_isolated_copy_and_never_written_to_source():
    src = open(SRC, encoding="utf-8").read()
    iso = src.split("def _isolated_model", 1)[1].split("def cmd_verify", 1)[0]
    assert "copy.deepcopy(tr.model)" in iso
    assert "torch.save" not in iso
    # every torch.save lives in an allowed function and writes under --out-dir
    tree = ast.parse(src)
    allowed = {"cmd_extract": "feat_path", "cmd_fit": "head_path"}
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        saves = [c for c in ast.walk(fn)
                 if isinstance(c, ast.Call)
                 and isinstance(c.func, ast.Attribute) and c.func.attr == "save"]
        if not saves:
            continue
        assert fn.name in allowed, f"torch.save in {fn.name}"
        seg = ast.get_source_segment(src, fn)
        assert allowed[fn.name] in seg and "a.out_dir" in seg, fn.name


def test_p_lin_compatibility_is_labelled_diagnostic_architecture():
    src = open(SRC, encoding="utf-8").read()
    assert "P_LIN_DIAGNOSTIC_ARCHITECTURE_COMPATIBILITY" in src
    assert "DEPLOYED_ARCHITECTURE_COMPATIBILITY" in src


def test_no_joint_optimizer_state_is_ever_reused():
    src = open(SRC, encoding="utf-8").read()
    ex = "\n".join(l for l in src.splitlines()
                   if l.strip() and not l.lstrip().startswith("#"))
    assert "optimizer_state_dict" not in ex
    assert "torch.optim.Adam(mod.parameters(), lr=LR)" in ex


def test_no_writes_to_any_source_run_or_archive_path():
    src = open(SRC, encoding="utf-8").read()
    tree = ast.parse(src)
    drop = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            drop.update(range(node.lineno - 1, node.end_lineno))
    ex = "\n".join(l for i, l in enumerate(src.splitlines())
                   if i not in drop and l.strip()
                   and not l.lstrip().startswith("#"))
    for forbidden in ("os.remove", "shutil.rmtree", "os.rename", "os.replace",
                      "unlink", "chmod", "archive_runs, \"w\"", "rmdir"):
        assert forbidden not in ex, forbidden


# ---- frozen protocol constants ------------------------------------------
def test_optimization_protocol_is_frozen_and_singular():
    assert m.LR == 1e-3 and m.BATCH == 1024
    assert m.MAX_EPOCHS == 300 and m.EVAL_EVERY == 5
    assert m.TAU == 0.10 and m.MARGIN_TARGET == 1.0
    assert m.DATA_SEED == 12345 and m.INIT_SEED == 12345
    assert m.EARLY_STOP_RULE.startswith("none")
    src = open(SRC, encoding="utf-8").read()
    tree = ast.parse(src)
    drop = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            drop.update(range(node.lineno - 1, node.end_lineno))
    ex = "\n".join(l for i, l in enumerate(src.splitlines())
                   if i not in drop and l.strip()
                   and not l.lstrip().startswith("#"))
    for forbidden in ("lr_ladder", "for lr in", "restart", "scheduler",
                      "weight_decay", "lr_scheduler"):
        assert forbidden not in ex, forbidden
