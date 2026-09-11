"""Tests for the Amendment-V2 weight-interpolation compatibility analysis.

Synthetic fixtures only; no scientific population is trained on and no
checkpoint is written.
"""
import os
import sys
import hashlib

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.head_interpolation_probe import (  # noqa: E402
    EPS, ALPHA_FRACTIONS, INHERITED, FP32_IDENTITY_TOL, lerp_state, s_hat_at,
    alpha_thresholds, refuse_protected,
)


def _heads(out=7, inp=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    h0 = {"2.weight": torch.randn(out, inp, generator=g),
          "2.bias": torch.randn(out, generator=g)}
    hs = {"2.weight": torch.randn(out, inp, generator=g),
          "2.bias": torch.randn(out, generator=g)}
    return h0, hs


# 1 / 2 -- endpoints reproduce the exact source layers
def test_alpha0_reproduces_deployed_layer_exactly():
    h0, hs = _heads()
    st = lerp_state(h0, hs, 0.0)
    for k in h0:
        assert torch.equal(st[k], h0[k]), k


def test_alpha1_reproduces_plast_layer_exactly():
    h0, hs = _heads()
    st = lerp_state(h0, hs, 1.0)
    for k in hs:
        assert torch.equal(st[k], hs[k]), k


# 3 -- weight interpolation == output interpolation
def test_weight_interpolation_equals_output_interpolation():
    h0, hs = _heads(out=11, inp=9, seed=3)
    phi = torch.randn(23, 9, generator=torch.Generator().manual_seed(4))
    s0, s1 = s_hat_at(phi, h0), s_hat_at(phi, hs)
    for a in (0.0, 0.137, 0.5, 0.9, 1.0):
        # fp64-retained weights: the identity is exact
        exact = s_hat_at(phi, lerp_state(h0, hs, a, dtype=torch.float64))
        mixed = (1.0 - a) * s0 + a * s1
        assert torch.allclose(exact, mixed, atol=1e-12, rtol=0), ("fp64", a)
        # fp32 weights (what a deployed module holds): within one ULP
        fp32 = s_hat_at(phi, lerp_state(h0, hs, a))
        assert (fp32 - mixed).abs().max().item() <= FP32_IDENTITY_TOL, ("fp32", a)
    # endpoints must be bitwise exact in fp32 too
    for a, h in ((0.0, h0), (1.0, hs)):
        assert torch.equal(s_hat_at(phi, lerp_state(h0, hs, a)), s_hat_at(phi, h))


# 4 -- analytic threshold agrees with brute-force argmax on a fixture
def test_analytic_threshold_matches_bruteforce_argmax():
    g = torch.Generator().manual_seed(11)
    n, N, d = 40, 25, 6
    bank = torch.nn.functional.normalize(torch.randn(N, d, generator=g), dim=-1)
    tgt = torch.randint(0, N, (n,), generator=g)
    u = torch.randn(n, d, generator=g) * 0.3
    # v deliberately pushed toward each target so alpha=1 is feasible
    v = bank[tgt] * 3.0 + torch.randn(n, d, generator=g) * 0.01
    L, U, argL, bad = alpha_thresholds(u, v, bank, tgt, block=7)
    assert not bool(bad.any())
    aC0 = float(L.max())
    assert aC0 < float(U.min())

    def errs(a):
        s = (1 - a) * u.double() + a * v.double()
        pred = (s @ bank.double().t()).argmax(dim=1)
        return int((pred != tgt).sum())

    assert errs(aC0 + 1e-4) == 0
    assert errs(aC0 - 1e-4) > 0
    assert errs(1.0) == 0


# 5 -- the target is excluded from its own competitor set
def test_target_excluded_from_constraints():
    g = torch.Generator().manual_seed(5)
    N, d = 9, 4
    bank = torch.nn.functional.normalize(torch.randn(N, d, generator=g), dim=-1)
    tgt = torch.tensor([2])
    u = (bank[tgt] * 5.0).clone()          # already comfortably correct
    v = (bank[tgt] * 6.0).clone()
    L, U, argL, bad = alpha_thresholds(u, v, bank, tgt)
    assert float(L[0]) == 0.0 and not bool(bad[0])
    assert int(argL[0]) != int(tgt[0])


# 6 -- strictness / zero-slope handling
def test_zero_slope_unsatisfied_is_infeasible_everywhere():
    d = 3
    bank = torch.eye(d)
    tgt = torch.tensor([0])
    # u gives an exact tie with competitor 1 and delta is orthogonal to (t0-t1)
    u = torch.tensor([[1.0, 1.0, 0.0]])
    v = torch.tensor([[1.0, 1.0, 2.0]])
    L, U, argL, bad = alpha_thresholds(u, v, bank, tgt)
    assert bool(bad[0]), "tie with zero slope must be flagged infeasible"


def test_eps_and_alpha_fractions_are_frozen_constants():
    assert EPS == 1e-6
    assert ALPHA_FRACTIONS == (0.25, 0.50, 0.75, 0.90)


# 7 -- source checkpoints are never written by this module
def test_module_contains_no_write_of_source_paths():
    src = open(os.path.join(
        ROOT, "scripts/naming_comprehension/head_interpolation_probe.py")).read()
    assert "torch.save" not in src
    assert "open(" not in src.replace("open(os.path.join", "")


# 8 -- official evaluator is reused, not reimplemented
def test_official_evaluator_is_imported_not_reimplemented():
    src = open(os.path.join(
        ROOT, "scripts/naming_comprehension/head_interpolation_probe.py")).read()
    assert "official_strict_errors" in src
    assert "evaluate_comprehension_subset" in src
    assert "repetition_snapshot" in src
    assert "evaluate_naming" in src
    assert "def comprehension_metrics" not in src     # no local copy


# 9 -- output-root refusal for protected locations
@pytest.mark.parametrize("bad", [
    "/x/lichtheim3_runs/final_settle_ctrl_h512_s19",
    "/x/archives/settle_u3600_20260910",
    "/x/archives/frozen_semantic_head_probe_20260911",
    "/x/lichtheim3-autoresearch/out",
    "/x/lichtheim3_autoresearch_runs/phase0",
    "/x/diag/final_something",
])
def test_refuses_protected_output_roots(bad):
    with pytest.raises(RuntimeError):
        refuse_protected(bad)


def test_allows_fresh_diagnostic_root(tmp_path):
    refuse_protected(str(tmp_path / "interp_diag_v2"))


def test_inherited_thresholds_are_labelled_unpreregistered():
    assert INHERITED["status"] == "UNPREREGISTERED_FOR_THIS_DIAGNOSTIC"
    assert INHERITED["relative_ltm_mean_guard"] == -0.02
    assert INHERITED["absolute_ltm_review_line"] == 0.8498
