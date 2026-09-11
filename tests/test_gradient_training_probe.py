"""Mandatory validation for the Amendment-V4 gradient-training probe.

Synthetic fixtures only; no scientific population is trained on here.  Real-data
checks (test 2: W0 reproduces the official baseline; test 16: source hashes) are
run by the separate read-only V4 precheck before execution."""
import ast
import inspect
import json
import os
import sys

import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import scripts.naming_comprehension.gradient_training_probe as gtp  # noqa: E402
from scripts.naming_comprehension.gradient_training_probe import (  # noqa: E402
    GAMMA_RAW, TAU, R0, BETA, ARMIJO_C, MAX_BACKTRACKS, MAX_ITERS, WALL_CAP_S, TRAINABLE_NAMES,
    PREREG_COMMIT, augment, margins, gamma_loss, gamma_loss_and_grad, ce_loss, ce_loss_and_grad,
    direction, armijo, train_run, refuse_protected, assert_unchanged, sha256_tensor, _save_head,
    NumericalFailure, Metric,
)



@pytest.fixture(autouse=True)
def _float64_default_scoped():
    """float64 for THIS module's tests only; restored afterwards so it cannot
    leak into other suites (a module-level set_default_dtype runs at collection
    time and would change the default for every test collected after it)."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)
SRC = os.path.join(ROOT, "scripts/naming_comprehension/gradient_training_probe.py")


def _fix(seed=0, n=40, nb=25, m=6, p=6, h=9, scale=0.3):
    g = torch.Generator().manual_seed(seed)
    bn = F.normalize(torch.randn(nb, m, generator=g), dim=-1)
    tgt = torch.randint(0, nb, (n,), generator=g)
    X = augment(torch.randn(n, p, generator=g))
    A = torch.randn(h, m, generator=g) * 0.5
    c = torch.randn(h, generator=g) * 0.1
    th0 = torch.randn(m, p + 1, generator=g) * scale
    return X, bn, tgt, A, c, th0


def _fix_feasible(seed=1, n=12, nb=10, m=6, h=9, big=False):
    """One-hot features: every item is independently addressable, so C=0 is reachable."""
    g = torch.Generator().manual_seed(seed)
    bn = F.normalize(torch.randn(nb, m, generator=g), dim=-1)
    tgt = torch.randint(0, nb, (n,), generator=g)
    X = torch.cat([torch.eye(n), torch.ones(n, 1)], 1)
    A = torch.randn(h, m, generator=g) * 0.5
    c = torch.randn(h, generator=g) * 0.1
    if big:
        th0 = torch.zeros(m, n + 1)
        for i in range(n):
            th0[:, i] = 5.0 * bn[tgt[i]]
    else:
        th0 = torch.randn(m, n + 1, generator=g) * 0.05
    return X, bn, tgt, A, c, th0


def _ev(bn, tgt, calls=None, force=None):
    def ev(s):
        if calls is not None:
            calls.append(1)
        if force is not None:
            return force
        return int(((F.normalize(s.double(), dim=-1) @ bn.t()).argmax(1) != tgt).sum())
    return ev


def _metric(X, A, c, th0):
    z0 = (X @ th0.t()) @ A.t() + c
    D = 1 - torch.tanh(z0) ** 2
    return Metric("Jlinh0", A, (X.t() @ X) / X.shape[0], Dsq_mean=(D ** 2).mean(0), X_all=X, Dsq=D ** 2), D


def _code_only():
    src = open(SRC).read()
    doc = ast.get_docstring(ast.parse(src))
    src = src.replace(doc, "", 1) if doc else src
    return "\n".join(ln.split("#")[0] for ln in src.split("\n"))


# ---------------------------------------------------------------- constants
def test_constants_frozen_to_preregistration():
    assert PREREG_COMMIT == "054633cdf173c27296bdbc406775329bd0bc38a4"
    assert GAMMA_RAW == 1e-3 and TAU == 0.10
    assert (R0, BETA, ARMIJO_C, MAX_BACKTRACKS) == (0.1, 0.5, 1e-4, 20)
    assert (MAX_ITERS, WALL_CAP_S) == (150, 7200)


# 1 -- only the final Linear is trainable / saved
def test_only_final_layer_trainable_and_saved(tmp_path):
    assert TRAINABLE_NAMES == ("ltm.to_semantic.2.weight", "ltm.to_semantic.2.bias")
    th = torch.randn(300, 513)
    p = str(tmp_path / "h.pt")
    _save_head(p, th, {"seed": 19})
    st = torch.load(p, weights_only=False)["state"]
    assert set(st) == {"2.weight", "2.bias"}
    assert tuple(st["2.weight"].shape) == (300, 512) and tuple(st["2.bias"].shape) == (300,)


def test_battery_refuses_head_with_extra_parameters(tmp_path):
    bad = str(tmp_path / "bad.pt")
    torch.save({"state": {"2.weight": torch.zeros(300, 512), "2.bias": torch.zeros(300),
                          "0.weight": torch.zeros(512, 512)}}, bad)
    with pytest.raises(RuntimeError):
        gtp.main(["battery", "--seed", "19", "--head", bad, "--out-json", str(tmp_path / "o.json")])


# 3 -- gamma-hinge gradient equals autograd and finite differences
def test_gamma_grad_matches_autograd_and_fd():
    X, bn, tgt, A, c, th0 = _fix()
    L, g, m, j, s, V = gamma_loss_and_grad(th0, X, bn, tgt)
    assert V.numel() > 0
    th = th0.clone().requires_grad_(True)
    sc = (X @ th.t()) @ bn.t()
    ts = sc.gather(1, tgt[:, None]).squeeze(1)
    other = sc.scatter(1, tgt[:, None], float("-inf")).max(1).values
    Lr = F.relu(GAMMA_RAW - (ts - other)).sum()
    Lr.backward()
    assert abs(L - float(Lr)) < 1e-12
    assert torch.allclose(g, th.grad, atol=1e-12)
    u = torch.randn(th0.shape, generator=torch.Generator().manual_seed(7))
    eps = 1e-7
    fd = (gamma_loss(th0 + eps * u, X, bn, tgt) - gamma_loss(th0 - eps * u, X, bn, tgt)) / (2 * eps)
    assert abs(fd - float((g * u).sum())) < 1e-5 * max(1.0, abs(fd))


# 4 -- satisfied constraints produce zero loss and zero gradient
def test_satisfied_constraints_give_zero_gradient():
    X, bn, tgt, A, c, th0 = _fix_feasible(big=True)
    L, g, m, j, s, V = gamma_loss_and_grad(th0, X, bn, tgt)
    assert float(m.min()) > GAMMA_RAW
    assert L == 0.0 and V.numel() == 0 and float(g.abs().max()) == 0.0


# 5 -- preconditioner solves H p = g
def test_preconditioner_solves_Hp_eq_g():
    X, bn, tgt, A, c, th0 = _fix()
    M, _ = _metric(X, A, c, th0)
    _, g, *_ = gamma_loss_and_grad(th0, X, bn, tgt)
    p = M.Hinv(g)
    assert float((M.H(p) - g).norm() / g.norm()) < 1e-6


# 6 / 9 -- Arm A is exactly -H^{-1} g, with no extra compatibility term
def test_armA_direction_is_minus_Hinv_g_without_contamination():
    X, bn, tgt, A, c, th0 = _fix()
    M, _ = _metric(X, A, c, th0)
    _, g, *_ = gamma_loss_and_grad(th0, X, bn, tgt)
    pA = direction("A", g, M)
    assert torch.equal(pA, -M.Hinv(g))
    assert float((M.H(-pA) - g).norm() / g.norm()) < 1e-6
    body = inspect.getsource(direction)
    for bad in ("J_lin", "J_z", "J_true", "Dsq", "tanh", "lambda"):
        assert bad not in body, bad
    assert "L, g, m, j, s, V = gamma_loss_and_grad(theta, X, bn, tgt)" in inspect.getsource(train_run)


# 7 -- Arm B is exactly -g
def test_armB_direction_is_minus_g():
    X, bn, tgt, A, c, th0 = _fix()
    _, g, *_ = gamma_loss_and_grad(th0, X, bn, tgt)
    assert torch.equal(direction("B", g), -g)


# 8 -- Arm C is the deployed cosine-CE gradient
def test_armC_is_deployed_ce_gradient():
    X, bn, tgt, A, c, th0 = _fix()
    L, g = ce_loss_and_grad(th0, X, bn, tgt)
    th = th0.clone().requires_grad_(True)
    Lr = F.cross_entropy(F.normalize(X @ th.t(), dim=-1) @ bn.t() / TAU, tgt)
    Lr.backward()
    assert abs(L - float(Lr)) < 1e-12 and abs(ce_loss(th0, X, bn, tgt) - float(Lr)) < 1e-12
    assert torch.allclose(g, th.grad, atol=1e-12)
    assert torch.equal(direction("C", g), -g)


# 10 -- D_i / H frozen at W0 for the whole run
def test_D_frozen_from_W0():
    X, bn, tgt, A, c, th0 = _fix()
    res = train_run("A", th0, X, bn, tgt, A, c, _ev(bn, tgt), theta_v3=th0 + 0.01, max_iters=3)
    z0 = (X @ th0.t()) @ A.t() + c
    ref = sha256_tensor((1 - torch.tanh(z0) ** 2) ** 2)
    assert res["dsq_sha_start"] == res["dsq_sha_end"] == ref
    assert res["steps"] >= 1


# 11 -- deterministic, length-normalised line search
def test_line_search_deterministic_and_length_normalised():
    X, bn, tgt, A, c, th0 = _fix()
    L, g, *_ = gamma_loss_and_grad(th0, X, bn, tgt)
    p = -g
    f = lambda th: gamma_loss(th, X, bn, tgt)
    r1, r2 = armijo(f, th0, L, g, p), armijo(f, th0, L, g, p)
    assert r1 is not None and torch.equal(r1[0], r2[0]) and r1[1:] == r2[1:]
    q = lambda th: float((th ** 2).sum())
    th = torch.full((3, 4), 3.0)
    out = armijo(q, th, q(th), 2 * th, -2 * th)
    assert out[2] == 0 and abs(float((out[0] - th).norm()) - R0) < 1e-12


# 12 -- failed line search stops cleanly
def test_failed_line_search_returns_none_and_run_stops(monkeypatch):
    X, bn, tgt, A, c, th0 = _fix()
    L, g, *_ = gamma_loss_and_grad(th0, X, bn, tgt)
    assert armijo(lambda th: 1.0, th0, 1.0, g, -g) is None
    monkeypatch.setattr(gtp, "armijo", lambda *a, **k: None)
    res = gtp.train_run("B", th0, X, bn, tgt, A, c, _ev(bn, tgt), theta_v3=th0)
    assert res["status"] == "LINE_SEARCH_FAILED" and res["steps"] == 0


def test_non_descent_direction_is_numerical_failure():
    X, bn, tgt, A, c, th0 = _fix()
    L, g, *_ = gamma_loss_and_grad(th0, X, bn, tgt)
    with pytest.raises(NumericalFailure):
        armijo(lambda th: 0.0, th0, L, g, g)          # g.p > 0


# 13 / 14 -- first-C0 logic; official evaluator mandatory and authoritative
def test_first_c0_logic_and_evaluator_authority():
    X, bn, tgt, A, c, th0 = _fix_feasible()
    calls = []
    res = train_run("B", th0, X, bn, tgt, A, c, _ev(bn, tgt, calls), theta_v3=th0, max_iters=400)
    k0 = res["first_c0_iter"]
    assert k0 is not None and len(calls) >= 1
    tr = res["trajectory"]
    assert tr[k0]["internal_strict_errors"] == 0
    assert all(r["internal_strict_errors"] > 0 for r in tr[:k0])
    assert res["official_calls"][0] == (k0, 0)
    res2 = train_run("B", th0, X, bn, tgt, A, c, _ev(bn, tgt, force=1), theta_v3=th0, max_iters=400)
    assert res2["first_c0_iter"] is None and len(res2["official_calls"]) >= 1


def test_evaluator_is_required():
    X, bn, tgt, A, c, th0 = _fix()
    with pytest.raises(TypeError):
        train_run("B", th0, X, bn, tgt, A, c, theta_v3=th0)


def test_converged_at_start_takes_zero_steps():
    X, bn, tgt, A, c, th0 = _fix_feasible(big=True)
    res = train_run("A", th0, X, bn, tgt, A, c, _ev(bn, tgt), theta_v3=th0)
    assert res["status"] == "CONVERGED" and res["steps"] == 0 and res["first_c0_iter"] == 0


def test_max_iters_enforced():
    X, bn, tgt, A, c, th0 = _fix()
    res = train_run("B", th0, X, bn, tgt, A, c, _ev(bn, tgt), theta_v3=th0, max_iters=2)
    assert res["status"] == "MAX_ITERS" and len(res["trajectory"]) == 3


# 15 -- route battery unchanged (reused, not reimplemented)
def test_route_battery_is_reused_unchanged():
    from scripts.naming_comprehension.coexistence_probe import full_battery
    from scripts.naming_comprehension.frozen_head_probe import _isolated_model, official_strict_errors
    assert gtp.full_battery is full_battery and gtp._isolated_model is _isolated_model
    assert gtp.official_strict_errors is official_strict_errors
    body = inspect.getsource(gtp.cmd_battery)
    assert "full_battery(tr, model, list(tr.comp_idx))" in body
    assert '_isolated_model(tr, "p_last_hinge", h["state"])' in body
    code = _code_only()
    for bad in ("def full_battery", "def comprehension_metrics", "def repetition_snapshot"):
        assert bad not in code


# 16 -- source hash guard
def test_source_hash_guard(tmp_path):
    f = tmp_path / "ck.pt"
    f.write_bytes(b"abc")
    from scripts.naming_comprehension.frozen_head_probe import sha256_file
    h = sha256_file(str(f))
    assert_unchanged(str(f), h)
    f.write_bytes(b"abd")
    with pytest.raises(RuntimeError):
        assert_unchanged(str(f), h)


# 17 -- output isolation
@pytest.mark.parametrize("bad", [
    "/x/lichtheim3_runs/final_settle_ctrl_h512_s19",
    "/x/archives/settle_u3600_20260910",
    "/x/archives/constrained_coexistence_v31_20260911",
    "/x/archives/feasible_region_characterization_20260911",
    "/x/archives/brand_new_dir",
    "/x/lichtheim3-autoresearch/out",
    "/x/lichtheim3_autoresearch_runs/phase0",
])
def test_refuses_protected_output_roots(bad):
    with pytest.raises(RuntimeError):
        refuse_protected(bad)


def test_allows_fresh_diagnostic_root(tmp_path):
    refuse_protected(str(tmp_path / "v4_runs" / "seed19_A"))


# 18 / 19 -- no AutoResearch / scheduler dependency
def test_no_autoresearch_or_scheduler_dependency():
    tree = ast.parse(open(SRC).read())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module)
    for mname in mods:
        assert "autoresearch" not in mname.lower() and "slurm" not in mname.lower(), mname
    code = _code_only().lower()
    for bad in ("sbatch", "srun ", "scancel", "subprocess", "os.system", "arctl"):
        assert bad not in code, bad


# 20 -- deterministic rerun of a synthetic fixture is identical
@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_deterministic_rerun_identical(arm):
    def once():
        X, bn, tgt, A, c, th0 = _fix(seed=3)
        r = train_run(arm, th0, X, bn, tgt, A, c, _ev(bn, tgt), theta_v3=th0 + 0.02, max_iters=4)
        tr = [{k: v for k, v in x.items() if k != "wall_s"} for x in r["trajectory"]]
        return r["final_theta"], json.dumps(tr, sort_keys=True), r["status"]
    a, b = once(), once()
    assert torch.equal(a[0], b[0]) and a[1] == b[1] and a[2] == b[2]
