"""Mandatory validation for the Amendment-V3 constrained coexistence solver.

Synthetic fixtures only.  No scientific population is optimised here and no
checkpoint is written.  The trusted QP reference is exhaustive active-set
enumeration solved by direct linear algebra (no new dependency).
"""
import itertools
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.constrained_coexistence_v3 import (  # noqa: E402
    GAMMA_RAW, TOP_V, K_MAX, MAX_ROUNDS, CG_RTOL, CG_MAXIT, ARMS, SEEDS,
    PREREG_COMMIT, augment, constraint_value, Metric, solve_dual, kkt_report,
    margins_and_worst, refuse_protected,
)

torch.set_default_dtype(torch.float64)


def _src_code_only():
    """Module source with the leading docstring removed, so prose that names a
    frozen module is not mistaken for a code dependency."""
    import ast
    path = os.path.join(ROOT,
                        "scripts/naming_comprehension/constrained_coexistence_v3.py")
    src = open(path).read()
    doc = ast.get_docstring(ast.parse(src))
    return src.replace(doc, "", 1) if doc else src


def _sys(n=40, p=7, m=5, h=9, seed=0):
    g = torch.Generator().manual_seed(seed)
    phi = torch.randn(n, p - 1, generator=g)
    A = torch.randn(h, m, generator=g)
    X_all = augment(phi)
    Xc = (X_all.t() @ X_all) / n
    return phi, A, X_all, Xc


# 16 -- frozen constants
def test_frozen_constants_exact():
    assert GAMMA_RAW == 1e-3
    assert TOP_V == 3 and K_MAX == 20_000 and MAX_ROUNDS == 30
    assert CG_RTOL == 1e-8 and CG_MAXIT == 500
    assert ARMS == ("I", "Jz", "Jlinh0") and SEEDS == (19, 20, 21, 22)
    assert PREREG_COMMIT == "fb634bec7f1490be588a17ab3a12f0531b7ea749"


# 1 -- affine constraint vectorization, no silent row/col-major assumption
def test_constraint_vectorization_matches_inner_product_with_rank1():
    g = torch.Generator().manual_seed(1)
    m, p = 5, 7
    th = torch.randn(m, p, generator=g)
    d = torch.randn(m, generator=g)
    x = torch.randn(p, generator=g)
    a = d.outer(x)                       # a_k = d x^T
    assert a.shape == (m, p)
    lhs = constraint_value(th, d, x)     # d^T th x
    rhs = (a * th).sum()                 # <a, th> = tr(a^T th)
    assert torch.allclose(lhs, rhs, atol=1e-12)


def test_augment_appends_exactly_one():
    phi = torch.randn(6, 4)
    X = augment(phi)
    assert X.shape == (6, 5)
    assert torch.equal(X[:, :4], phi) and torch.all(X[:, 4] == 1.0)


# 13 -- source head gives exactly zero displacement objective
def test_zero_theta_is_zero_objective_for_every_arm():
    phi, A, X_all, Xc = _sys()
    z0 = (X_all @ torch.randn(5, X_all.shape[1],
          generator=torch.Generator().manual_seed(98)).t()) @ A.t()
    Dsq = (1 - torch.tanh(z0) ** 2) ** 2
    for arm in ARMS:
        mt = Metric(arm, A, Xc, Dsq_mean=Dsq.mean(0), X_all=X_all, Dsq=Dsq)
        th = torch.zeros(A.shape[1], X_all.shape[1])
        assert float((th * mt.H(th)).sum()) == 0.0


# 9/10 -- H identity and objective identity vs explicit dense form
@pytest.mark.parametrize("arm", ARMS)
def test_H_matches_explicit_mean_form(arm):
    phi, A, X_all, Xc = _sys(seed=2)
    th = torch.randn(A.shape[1], X_all.shape[1])
    z0 = torch.randn(X_all.shape[0], A.shape[0],
                     generator=torch.Generator().manual_seed(99))
    Dsq = (1 - torch.tanh(z0) ** 2) ** 2
    mt = Metric(arm, A, Xc, Dsq_mean=Dsq.mean(0), X_all=X_all, Dsq=Dsq)
    s = X_all @ th.t()
    if arm == "I":
        expect = (s ** 2).sum(1).mean()
    elif arm == "Jz":
        expect = ((s @ A.t()) ** 2).sum(1).mean()
    else:
        expect = ((Dsq.sqrt() * (s @ A.t())) ** 2).sum(1).mean()
    assert torch.allclose((th * mt.H(th)).sum(), expect, rtol=1e-10)


@pytest.mark.parametrize("arm", ARMS)
def test_Hinv_inverts_H(arm):
    phi, A, X_all, Xc = _sys(seed=3)
    z0 = torch.randn(X_all.shape[0], A.shape[0],
                     generator=torch.Generator().manual_seed(99))
    Dsq = (1 - torch.tanh(z0) ** 2) ** 2
    mt = Metric(arm, A, Xc, Dsq_mean=Dsq.mean(0), X_all=X_all, Dsq=Dsq)
    th = torch.randn(A.shape[1], X_all.shape[1])
    back = mt.Hinv(mt.H(th))
    assert float((back - th).abs().max()) < 1e-6


# 9 -- closed-form Gram equals direct evaluation (I and Jz)
@pytest.mark.parametrize("arm", ["I", "Jz"])
def test_closed_form_gram_matches_direct(arm):
    phi, A, X_all, Xc = _sys(seed=4)
    mt = Metric(arm, A, Xc)
    g = torch.Generator().manual_seed(5)
    K = 6
    ds = torch.randn(K, A.shape[1], generator=g)
    idx = torch.randint(0, X_all.shape[0], (K,), generator=g)
    xs = X_all[idx]
    G = mt.gram(ds, xs)
    direct = torch.zeros(K, K)
    for k in range(K):
        Hk = mt.Hinv(ds[k].outer(xs[k]))
        for l in range(K):
            direct[l, k] = (ds[l].outer(xs[l]) * Hk).sum()
    assert float((G - direct).abs().max()) < 1e-10


# 11 -- Jlinh0 CG gram agrees with the dense direct evaluation
def test_jlinh0_gram_cg_matches_direct():
    phi, A, X_all, Xc = _sys(seed=6)
    z0 = torch.randn(X_all.shape[0], A.shape[0],
                     generator=torch.Generator().manual_seed(99))
    Dsq = (1 - torch.tanh(z0) ** 2) ** 2
    mt = Metric("Jlinh0", A, Xc, Dsq_mean=Dsq.mean(0), X_all=X_all, Dsq=Dsq)
    g = torch.Generator().manual_seed(7)
    K = 4
    ds = torch.randn(K, A.shape[1], generator=g)
    xs = X_all[torch.randint(0, X_all.shape[0], (K,), generator=g)]
    G = mt.gram(ds, xs)
    direct = torch.zeros(K, K)
    for k in range(K):
        Hk = mt.Hinv(ds[k].outer(xs[k]))
        for l in range(K):
            direct[l, k] = (ds[l].outer(xs[l]) * Hk).sum()
    assert float((G - 0.5 * (direct + direct.t())).abs().max()) < 1e-6


# 8/12 -- optimum and KKT vs the exhaustive active-set reference
@pytest.mark.parametrize("arm", ARMS)
def test_dual_optimum_matches_exhaustive_reference_and_kkt(arm):
    phi, A, X_all, Xc = _sys(n=30, p=6, m=4, h=7, seed=8)
    z0 = torch.randn(X_all.shape[0], A.shape[0],
                     generator=torch.Generator().manual_seed(99))
    Dsq = (1 - torch.tanh(z0) ** 2) ** 2
    mt = Metric(arm, A, Xc, Dsq_mean=Dsq.mean(0), X_all=X_all, Dsq=Dsq)
    g = torch.Generator().manual_seed(9)
    K = 5
    ds = torch.randn(K, A.shape[1], generator=g)
    xs = X_all[torch.randint(0, X_all.shape[0], (K,), generator=g)]
    r = torch.rand(K, generator=g) * 0.5
    G = mt.gram(ds, xs)
    lam, info = solve_dual(G, r)
    th = 0.5 * mt.Hinv(sum((lam[k] * ds[k].outer(xs[k]) for k in range(K)),
                           torch.zeros(A.shape[1], X_all.shape[1])))
    obj = float((th * mt.H(th)).sum())
    # trusted reference: enumerate every active set, direct linear algebra
    best = None
    for S in itertools.chain.from_iterable(
            itertools.combinations(range(K), j) for j in range(K + 1)):
        if not S:
            lm = torch.zeros(K)
        else:
            try:
                ls = torch.linalg.solve(G[list(S)][:, list(S)], 2.0 * r[list(S)])
            except Exception:
                continue
            if bool((ls < -1e-10).any()):
                continue
            lm = torch.zeros(K)
            lm[list(S)] = ls
        t2 = 0.5 * mt.Hinv(sum((lm[k] * ds[k].outer(xs[k]) for k in range(K)),
                               torch.zeros(A.shape[1], X_all.shape[1])))
        sl = torch.stack([constraint_value(t2, ds[k], xs[k]) for k in range(K)]) - r
        if bool((sl < -1e-8).any()):
            continue
        o = float((t2 * mt.H(t2)).sum())
        if best is None or o < best:
            best = o
    assert best is not None
    assert obj <= best * (1 + 1e-6) + 1e-10, (arm, obj, best)
    k = kkt_report(mt, th, lam, ds, xs, r)
    assert k["primal_min_slack"] > -1e-7
    assert k["dual_min_lambda"] > -1e-12
    assert k["max_complementarity"] < 1e-6
    assert k["stationarity_rel"] < 1e-6


# 2/3/4 -- gamma sign, target exclusion, top-3 selection
def test_oracle_excludes_target_and_returns_topv():
    g = torch.Generator().manual_seed(10)
    n, nb, m = 12, 9, 4
    bn = torch.nn.functional.normalize(torch.randn(nb, m, generator=g), dim=-1)
    tgt = torch.randint(0, nb, (n,), generator=g)
    X_all = augment(torch.randn(n, 3, generator=g))
    s0 = bn[tgt] * 5.0                      # every item comfortably correct
    th = torch.zeros(m, X_all.shape[1])
    worst, cand, _ = margins_and_worst(th, X_all, s0, bn, tgt)
    assert bool((worst > 0).all())
    assert cand.shape == (n, TOP_V)
    for i in range(n):
        assert int(tgt[i]) not in cand[i].tolist()


def test_gamma_sign_convention_r_positive_for_violation():
    # a violated item has m < 0 so r = gamma - m > gamma > 0
    d = torch.tensor([1.0, 0.0]); s0 = torch.tensor([-0.5, 0.0])
    m = float(d @ s0)
    assert m < 0 and GAMMA_RAW - m > GAMMA_RAW > 0


# 5/6/7 -- active set bookkeeping
def test_active_set_dedup_exact():
    act = {}
    for pair in [(3, 7), (3, 7), (4, 7), (3, 8)]:
        act[pair] = None
    assert len(act) == 3 and (3, 7) in act


def test_kmax_and_maxrounds_are_enforced_bounds():
    assert K_MAX == 20_000 and MAX_ROUNDS == 30
    src = _src_code_only()
    assert "if len(act) > K_MAX:" in src
    assert "for rnd in range(1, MAX_ROUNDS + 1):" in src


# 15 -- only the final Linear(512->300) can differ
def test_only_final_layer_is_produced():
    src = _src_code_only()
    for forbidden in ("to_semantic.0", "ltm.encoder", "ltm.decoder",
                      "motor", "optimizer", "backward(", ".step()"):
        assert forbidden not in src, forbidden
    assert '"W": W, "b": b,' in src


# 17/18 -- official evaluators are reused, never reimplemented
def test_official_evaluators_not_reimplemented_here():
    src = _src_code_only()
    assert "def comprehension_metrics" not in src
    assert "def repetition_snapshot" not in src
    assert "def evaluate_naming" not in src


# 19 -- output isolation
@pytest.mark.parametrize("bad", [
    "/x/lichtheim3_runs/final_settle_ctrl_h512_s19",
    "/x/archives/settle_u3600_20260910",
    "/x/archives/frozen_semantic_head_probe_20260911",
    "/x/archives/head_interpolation_diag_20260911",
    "/x/lichtheim3-autoresearch/out",
    "/x/lichtheim3_autoresearch_runs/phase0",
    "/x/diag/final_thing",
])
def test_refuses_protected_output_roots(bad):
    with pytest.raises(RuntimeError):
        refuse_protected(bad)


def test_allows_fresh_diagnostic_root(tmp_path):
    refuse_protected(str(tmp_path / "constrained_v3_out"))


# 20 -- determinism
@pytest.mark.parametrize("arm", ARMS)
def test_deterministic_rerun_identical(arm):
    def once():
        phi, A, X_all, Xc = _sys(n=25, p=6, m=4, h=6, seed=11)
        z0 = torch.randn(X_all.shape[0], A.shape[0],
                     generator=torch.Generator().manual_seed(99))
        Dsq = (1 - torch.tanh(z0) ** 2) ** 2
        mt = Metric(arm, A, Xc, Dsq_mean=Dsq.mean(0), X_all=X_all, Dsq=Dsq)
        g = torch.Generator().manual_seed(12)
        K = 4
        ds = torch.randn(K, A.shape[1], generator=g)
        xs = X_all[torch.randint(0, X_all.shape[0], (K,), generator=g)]
        r = torch.rand(K, generator=g) * 0.3
        lam, _ = solve_dual(mt.gram(ds, xs), r)
        return lam
    a, b = once(), once()
    assert torch.equal(a, b)


# 21/22 -- no AutoResearch and no scheduler dependency
def test_no_autoresearch_or_scheduler_dependency():
    import ast
    path = os.path.join(ROOT,
                        "scripts/naming_comprehension/constrained_coexistence_v3.py")
    tree = ast.parse(open(path).read())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module)
    # no import may reach AutoResearch or a scheduler
    for m in mods:
        assert "autoresearch" not in m.lower(), m
        assert "slurm" not in m.lower(), m
    # no scheduler invocation anywhere in code (path blocklists are guards, not deps)
    src = _src_code_only().lower()
    for bad in ("sbatch", "srun ", "scancel", "subprocess", "os.system"):
        assert bad not in src, bad
    # arctl / registry / enablement must not be *used*; they may only appear
    # inside the refuse_protected path blocklist, which is a guard
    import re
    body = re.sub(r'forbidden = \([^)]*\)', '', src, flags=re.S)
    for bad in ("arctl", "registry", "enablement"):
        assert bad not in body, bad


# ===================================================== Amendment V3.1 tests
from scripts.naming_comprehension.constrained_coexistence_v3 import (  # noqa: E402
    FEAS_TOL_RAW, feasibility_threshold,
)


# V3.1-1 -- margin exactly gamma passes
def test_margin_exactly_gamma_is_feasible():
    assert GAMMA_RAW >= feasibility_threshold()


# V3.1-2 -- the observed roundoff deficit passes
def test_observed_roundoff_deficit_passes():
    observed = 0.00099998011995872105          # measured seed-19 witness
    assert GAMMA_RAW - observed == pytest.approx(1.988e-08, rel=0.05)
    assert observed >= feasibility_threshold()
    assert (GAMMA_RAW - 2e-8) >= feasibility_threshold()


# V3.1-3 -- a deficit an order of magnitude larger than the tolerance fails
def test_deficit_larger_than_tolerance_fails():
    assert (GAMMA_RAW - 2e-7) < feasibility_threshold()
    assert (GAMMA_RAW - 1e-5) < feasibility_threshold()


# V3.1-4 -- a genuine strict error can NEVER pass on the tolerance
@pytest.mark.parametrize("m", [0.0, -1e-12, -1e-6, -0.001, -0.5, -3.58])
def test_strict_errors_can_never_pass_on_tolerance(m):
    assert m < feasibility_threshold()
    # and the acceptance threshold is four orders of magnitude above zero
    assert feasibility_threshold() / GAMMA_RAW == pytest.approx(1 - 1e-4, rel=1e-9)


# V3.1-5 -- gamma itself is untouched
def test_gamma_raw_still_exactly_1e_3():
    assert GAMMA_RAW == 1e-3
    assert FEAS_TOL_RAW == 1e-7
    assert FEAS_TOL_RAW / GAMMA_RAW == pytest.approx(1e-4, rel=1e-12)


# V3.1-6 -- the tolerance is used ONLY for termination / feasibility, never in the QP
def test_tolerance_not_used_in_qp_construction():
    import re
    src = _src_code_only()
    # strip full-line and trailing comments so this checks CODE, not prose
    code = "\n".join(re.sub(r"#.*$", "", ln) for ln in src.split("\n"))
    # right-hand sides must still target the full gamma
    assert "GAMMA_RAW - ds[k] @ s0[keys[k][0]]" in code
    qp = code.split("def run_arm")[1].split("W = theta")[0]
    assert "FEAS_TOL_RAW" not in qp, "FEAS_TOL_RAW must not appear in QP assembly code"
    # the QP right-hand side must use the untouched gamma, not the threshold
    assert "feasibility_threshold()" not in qp.split("r = torch.stack")[1][:200]
    # exactly one definition of the threshold, used for the violation set
    assert src.count("def feasibility_threshold") == 1
    assert "viol = (worst < feasibility_threshold())" in src


# V3.1-7 -- official C verification remains mandatory and unmodified
def test_official_c_verification_still_required():
    src = _src_code_only()
    assert "def comprehension_metrics" not in src
    from scripts.naming_comprehension.frozen_head_probe import official_strict_errors
    assert callable(official_strict_errors)


# V3.1-8 -- objective and active-set construction unchanged by the amendment
@pytest.mark.parametrize("arm", ARMS)
def test_objective_and_gram_unchanged_by_amendment(arm):
    phi, A, X_all, Xc = _sys(seed=21)
    z0 = torch.randn(X_all.shape[0], A.shape[0],
                     generator=torch.Generator().manual_seed(99))
    Dsq = (1 - torch.tanh(z0) ** 2) ** 2
    mt = Metric(arm, A, Xc, Dsq_mean=Dsq.mean(0), X_all=X_all, Dsq=Dsq)
    th = torch.randn(A.shape[1], X_all.shape[1])
    s = X_all @ th.t()
    expect = {"I": (s ** 2).sum(1).mean(),
              "Jz": ((s @ A.t()) ** 2).sum(1).mean(),
              "Jlinh0": ((Dsq.sqrt() * (s @ A.t())) ** 2).sum(1).mean()}[arm]
    assert torch.allclose((th * mt.H(th)).sum(), expect, rtol=1e-10)


# V3.1-9 -- the seed-19 numerical state now terminates instead of spinning
def test_seed19_numerical_state_terminates():
    worst = torch.full((27981,), 0.5)
    worst[:19] = 0.00099998011995872105        # the measured active-constraint value
    viol = (worst < feasibility_threshold()).nonzero().flatten()
    assert viol.numel() == 0, "converged seed-19 state must terminate, not re-flag"
    # under the pre-fix predicate it would have re-flagged all 19 forever
    assert int((worst < GAMMA_RAW).sum()) == 19


# V3.1-10 -- a genuinely infeasible state still fails
def test_genuinely_infeasible_state_still_fails():
    worst = torch.full((100,), 0.5)
    worst[7] = -0.02                            # a real strict error
    viol = (worst < feasibility_threshold()).nonzero().flatten()
    assert viol.tolist() == [7]
