"""Mandatory tests for the semantic coexistence probe (24 preregistered
checks).  No scientific fit may run while any of these fails."""
from __future__ import annotations

import ast
import inspect
import os
import sys

import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension import coexistence_probe as m   # noqa: E402

SRC = os.path.join(ROOT, "scripts/naming_comprehension/coexistence_probe.py")


def _exec_lines(path):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    drop = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            drop.update(range(node.lineno - 1, node.end_lineno))
    return "\n".join(l for i, l in enumerate(src.splitlines())
                     if i not in drop and l.strip()
                     and not l.lstrip().startswith("#"))


# ---- 1/2: trainable set, everything else frozen --------------------------
def test_trainable_set_is_exactly_to_semantic_2():
    assert m.TRAINABLE_NAMES == ("ltm.to_semantic.2.weight",
                                 "ltm.to_semantic.2.bias")
    fit = inspect.getsource(m.cmd_fit)
    assert "torch.optim.Adam([W, b], lr=LR)" in fit
    for forbidden in ("to_semantic.0", "encoder", "decoder.", "gate.",
                      "sem_to_h0.weight.requires_grad"):
        assert forbidden not in fit, forbidden


def test_frozen_tensors_never_get_gradients():
    ex = _exec_lines(SRC)
    assert "requires_grad_(True)" not in ex
    assert ex.count("nn.Parameter") == 2          # exactly W and b


# ---- 3: source immutability ---------------------------------------------
def test_source_checkpoint_hashed_before_and_after():
    fit = inspect.getsource(m.cmd_fit)
    assert "sha_before = sha256_file(ck)" in fit
    assert "sha_after = sha256_file(ck)" in fit
    assert "HARD STOP: source checkpoint mutated" in fit


# ---- 4: identities -------------------------------------------------------
def test_population_and_bank_identities():
    assert m.N_COMP_POP == 27_981 and m.N_BANK == 29_571
    assert m.N_REP_POP == 29_571
    assert m.BASELINE_C_ERRORS == {19: 34, 20: 32, 21: 25, 22: 29}
    b = inspect.getsource(m.cmd_baselines)
    assert "len(idx) == N_COMP_POP" in b and "tr.bank_raw.shape[0]) == N_BANK" in b
    assert "len(tr.entries) == N_REP_POP" in b


# ---- 5/6/7: hardest negative, margin sign, m -----------------------------
def test_hardest_negative_excludes_the_target():
    bank = F.normalize(torch.eye(5), dim=-1)
    z = torch.tensor([[3.0, 1.0, 0.0, 0.0, 0.0]])
    mg = m.cos_margins(z, bank, torch.tensor([0]))
    q = F.normalize(z, dim=-1)
    assert pytest.approx(float(mg), abs=1e-6) == float(q[0, 0] - q[0, 1])
    assert float(mg) > 0


def test_margin_sign_is_negative_when_the_competitor_wins():
    bank = F.normalize(torch.eye(4), dim=-1)
    z = torch.tensor([[1.0, 5.0, 0.0, 0.0]])
    assert float(m.cos_margins(z, bank, torch.tensor([0]))) < 0


def test_margin_constant_is_exactly_0_01():
    assert m.MARGIN_M == 0.01
    assert "MARGIN_M = 0.01" in open(SRC, encoding="utf-8").read()


# ---- 9: normalized hinge, strict error => term >= 1 ----------------------
def test_strict_error_contributes_at_least_one():
    assert float(m.hinge_terms(torch.tensor([0.0]))) == pytest.approx(1.0)
    assert float(m.hinge_terms(torch.tensor([-0.005]))) >= 1.0
    assert float(m.hinge_terms(torch.tensor([-0.05]))) == pytest.approx(6.0)
    assert float(m.hinge_terms(torch.tensor([m.MARGIN_M]))) == 0.0
    assert float(m.hinge_terms(torch.tensor([0.5]))) == 0.0
    # L_C is a SUM, not a mean
    src = inspect.getsource(m.c_loss)
    assert ".sum()" in src and ".mean()" not in src


def test_L_C_lower_bound_equals_error_count():
    """n strict errors => L_C >= n, which is what makes the hierarchy work."""
    mg = torch.tensor([-0.02, -0.001, 0.0, 0.5, 0.2])   # 3 strict errors
    assert float(m.hinge_terms(mg).sum()) >= 3.0


# ---- 11/12/13: preservation term ----------------------------------------
def test_weighted_unique_form_L_P_equals_direct_population_mean():
    """Homophones share h0, so multiplicity weighting over unique forms is
    EXACTLY the mean over the 29,571-entry repetition population."""
    torch.manual_seed(0)
    U, D = 6, m.H0_DIM
    h0s = torch.tanh(torch.randn(U, D))
    h0n = torch.tanh(torch.randn(U, D))
    w = torch.tensor([3., 1., 5., 2., 1., 4.])          # sum 16
    expand = torch.repeat_interleave(torch.arange(U), w.long())
    direct = (((h0n[expand] - h0s[expand]) ** 2).sum(-1)
              / (m.H0_SQ_MAX * D)).sum() / float(w.sum())
    old = m.N_REP_POP
    try:
        m.N_REP_POP = int(w.sum())
        weighted = m.p_loss(h0n, h0s, w)
    finally:
        m.N_REP_POP = old
    assert torch.allclose(direct, weighted, atol=1e-12)


def test_L_P_is_zero_at_the_source_head():
    h = torch.tanh(torch.randn(10, m.H0_DIM))
    w = torch.ones(10) * (m.N_REP_POP / 10)
    assert float(m.p_loss(h, h, w)) == 0.0


def test_L_P_is_bounded_in_unit_interval():
    w = torch.ones(8) * (m.N_REP_POP / 8)
    worst = m.p_loss(torch.ones(8, m.H0_DIM), -torch.ones(8, m.H0_DIM), w)
    assert float(worst) == pytest.approx(1.0, abs=1e-9)   # the exact bound
    assert 0.0 <= float(m.p_loss(torch.tanh(torch.randn(8, m.H0_DIM)),
                                 torch.tanh(torch.randn(8, m.H0_DIM)), w)) <= 1.0
    assert m.H0_SQ_MAX == 4.0 and m.H0_DIM == 512


def test_hierarchy_one_error_outweighs_the_entire_preservation_term():
    """The structural justification for having no lambda."""
    assert float(m.hinge_terms(torch.tensor([0.0]))) >= 1.0
    w = torch.ones(4) * (m.N_REP_POP / 4)
    assert float(m.p_loss(torch.ones(4, m.H0_DIM),
                          -torch.ones(4, m.H0_DIM), w)) <= 1.0


# ---- 16/17/18: guards ----------------------------------------------------
def _bat(c=0, r=3, rf=3, n=0, ltm=3400, wm=0):
    return {"c_errors": c, "rep_canonical_full_errors": r,
            "rep_freear_full_errors": rf, "naming_errors": n,
            "rep_canonical_ltm_errors": ltm, "rep_canonical_wm_errors": wm}


def test_guards_are_source_relative_and_ventral_is_primary():
    base = _bat(c=34, r=4, rf=4, n=0, ltm=3500)
    ok = m.guards(_bat(c=0, r=4, rf=4, n=0, ltm=3500), base)
    assert ok["c_success"] and ok["whole_model_preservation"]
    assert ok["ventral_route_preservation"] and ok["full_coexistence"]
    # ventral degradation alone must veto full coexistence
    bad = m.guards(_bat(c=0, r=4, rf=4, n=0, ltm=3501), base)
    assert bad["whole_model_preservation"] is True
    assert bad["ventral_route_preservation"] is False
    assert bad["full_coexistence"] is False
    # a seed with a nonzero baseline is held to its OWN baseline, not zero
    assert m.guards(_bat(c=0, r=4, rf=4, n=0, ltm=3500),
                    base)["R_canonical_ok"] is True


def test_wm_invariance_is_flagged_as_an_implementation_bug():
    base = _bat(c=34, ltm=3500, wm=0)
    assert m.guards(_bat(c=0, ltm=3400, wm=0), base)["wm_invariant"] is True
    assert m.guards(_bat(c=0, ltm=3400, wm=7), base)["wm_invariant"] is False
    fit = inspect.getsource(m.cmd_fit)
    assert "IMPLEMENTATION_BUG_HARD_ABORT" in fit
    assert fit.count("IMPLEMENTATION_BUG_HARD_ABORT") >= 2   # both batteries


# ---- 19/20/21/22: selection rules ---------------------------------------
def test_first_c0_triggers_the_battery_in_both_stages():
    fit = inspect.getsource(m.cmd_fit)
    assert 'tag = "FIRST_C0" if first_c0 is None' in fit
    assert "full_battery(tr, model, idx)" in fit


def test_coex1_stops_at_first_c0_whatever_the_outcome():
    fit = inspect.getsource(m.cmd_fit)
    seg = fit.split('if stage == "coex1":')[1][:400]
    assert "break" in seg


def test_coex2_min_lp_selection_is_predeclared_and_uses_only_L_P():
    fit = inspect.getsource(m.cmd_fit)
    seg = fit.split("MIN_LP selection")[1] if "MIN_LP selection" in fit else fit
    sel = fit.split("later = [c for c in cands")[1].split("append_tsv")[0]
    assert 'min(later, key=lambda c: (c["L_P"], c["epoch"]))' in sel
    # the selection key must not mention any functional metric
    for forbidden in ("rep_", "ltm", "naming", "R_canonical", "ventral",
                      "full_coexistence"):
        assert forbidden not in sel, forbidden


def test_no_functional_metric_can_drive_candidate_selection():
    ex = _exec_lines(SRC)
    assert 'key=lambda c: (c["L_P"], c["epoch"])' in ex
    assert ex.count("min(later") == 1


# ---- 23: labels ----------------------------------------------------------
def test_failure_labels_cannot_express_impossibility():
    src = open(SRC, encoding="utf-8").read()
    for bad in m.LABELS_FORBIDDEN:
        assert src.count(bad) <= 1, bad        # only in the forbidden list
    assert m.LABELS_ALLOWED == {
        "COEX1_FULL_COEXISTENCE", "COEX1_C_ZERO_PRESERVATION_FAIL",
        "COEX1_EMPIRICAL_C_FAILURE", "COEX2_FULL_COEXISTENCE",
        "COEX2_C_ZERO_NO_PRESERVED_SOLUTION_FOUND",
        "COEX2_EMPIRICAL_C_FAILURE", "NUMERICAL_FAILURE"}
    assert "assert label in LABELS_ALLOWED" in src


# ---- 24: no writes outside the probe tree -------------------------------
def test_no_writes_to_source_or_archive_paths():
    ex = _exec_lines(SRC)
    for forbidden in ("os.remove", "shutil.rmtree", "os.rename", "os.replace",
                      "unlink", "chmod", "rmdir"):
        assert forbidden not in ex, forbidden
    tree = ast.parse(open(SRC, encoding="utf-8").read())
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        if any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
               and c.func.attr == "save" for c in ast.walk(fn)):
            seg = ast.get_source_segment(open(SRC, encoding="utf-8").read(), fn)
            assert ("h0_path(a.out_dir" in seg or "head_path(out" in seg), fn.name


# ---- protocol frozen -----------------------------------------------------
def test_optimization_protocol_is_frozen_and_singular():
    assert (m.LR, m.BATCH, m.MAX_EPOCHS, m.EVAL_EVERY) == (1e-3, 1024, 300, 5)
    assert m.DATA_SEED == 12345
    ex = _exec_lines(SRC)
    for forbidden in ("lr_ladder", "for lr in", "restart", "scheduler",
                      "weight_decay", "lambda_p", "lam_p"):
        assert forbidden not in ex, forbidden
