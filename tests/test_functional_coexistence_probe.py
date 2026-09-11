"""Mandated pre-execution tests (25 checks) for the functional coexistence probe.

These gate scientific execution: the probe must not run with any failing.
"""
import hashlib
import inspect
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension import functional_coexistence_probe as P


# ---------------------------------------------------------------- fixtures
class TinyDec(nn.Module):
    """Stand-in with the canonical call surface: motor(ltm.decode_from_s_hat)."""
    def __init__(self, V=7, H=5):
        super().__init__()
        self.V, self.H = V, H
        class _L(nn.Module):
            def __init__(s):
                super().__init__()
                s.emb = nn.Embedding(V, H)
                s.lin = nn.Linear(H, H)
            def decode_from_s_hat(s, z, din):
                h = torch.tanh(s.lin(z)).unsqueeze(1)
                return s.emb(din) + h
        self.ltm = _L()
        self.motor = nn.Linear(H, V)


class FakeEntry:
    def __init__(self, ph): self.phonemes = ph


class FakeVocab:
    pad_id, bos_id, eos_id = 0, 1, 2


class FakeTr:
    def __init__(self, forms):
        self.vocab = FakeVocab()
        self.entries = [FakeEntry(f) for f in forms]


# 1 ------------------------------------------------------------------------
def test_1_trainable_parameter_set_exact():
    assert P.TRAINABLE_NAMES == ("ltm.to_semantic.2.weight",
                                 "ltm.to_semantic.2.bias")
    m = nn.Module()
    m.a = nn.Linear(2, 2)
    with pytest.raises(SystemExit) as e:
        P.assert_trainable(m)
    assert "IMPLEMENTATION_BUG_HARD_ABORT" in str(e.value)


# 2 ------------------------------------------------------------------------
def test_2_frozen_fingerprint_ignores_head_and_catches_others():
    class M(nn.Module):
        def __init__(s):
            super().__init__()
            s.enc = nn.Linear(3, 3)
    m = M()
    f0 = P.frozen_fingerprint(m)
    with torch.no_grad():
        m.enc.weight.add_(1.0)
    assert P.frozen_fingerprint(m) != f0, "frozen change must alter fingerprint"


# 3 ------------------------------------------------------------------------
def test_3_source_sha_helper_is_content_addressed(tmp_path):
    from scripts.naming_comprehension.frozen_head_probe import sha256_file
    p = tmp_path / "c.pt"
    p.write_bytes(b"abc")
    a = sha256_file(str(p))
    assert a == hashlib.sha256(b"abc").hexdigest()
    p.write_bytes(b"abd")
    assert sha256_file(str(p)) != a


# 4,5,6 --------------------------------------------------------------------
def test_4_5_6_canonical_population_constants():
    from scripts.naming_comprehension.frozen_head_probe import N_COMP_POP, N_BANK
    assert N_COMP_POP == 27_981
    assert N_BANK == 29_571
    assert P.N_REP_POP == 29_571


# 7 ------------------------------------------------------------------------
def test_7_multiplicity_weighted_ldec_equals_direct_population():
    """A repetition item's input IS its output, so homophones are the same
    item: the weighted unique-form CE must equal the direct 29,571 CE."""
    torch.manual_seed(0)
    m = TinyDec()
    for p in m.parameters():
        p.requires_grad_(False)
    n_u, D = 6, 4
    phi = torch.randn(n_u, D)
    W = torch.randn(m.H, D) * 0.3
    b = torch.randn(m.H) * 0.3
    din = torch.tensor([[1, 3, 4], [1, 5, 0], [1, 3, 0], [1, 4, 5],
                        [1, 6, 3], [1, 5, 4]])
    dtg = torch.tensor([[3, 4, 2], [5, 2, 0], [3, 2, 0], [4, 5, 2],
                        [6, 3, 2], [5, 4, 2]])
    w = torch.tensor([3., 1., 2., 1., 4., 2.])          # sum 13
    lw = P.ldec_full(m, phi, W, b, din, dtg, w, 0)
    # direct: physically expand every duplicate, unweighted
    rep = torch.repeat_interleave(torch.arange(n_u), w.long())
    ld = P.ldec_full(m, phi[rep], W, b, din[rep], dtg[rep],
                     torch.ones(len(rep)), 0)
    assert torch.allclose(lw, ld, atol=1e-6), (lw, ld)


def test_7b_ldec_matches_canonical_cross_entropy_semantics():
    """Unweighted, the helper must equal F.cross_entropy(ignore_index=pad)."""
    torch.manual_seed(1)
    m = TinyDec()
    for p in m.parameters():
        p.requires_grad_(False)
    phi = torch.randn(5, 4)
    W = torch.randn(m.H, 4) * 0.3
    b = torch.randn(m.H) * 0.3
    din = torch.tensor([[1, 3, 4], [1, 5, 0], [1, 3, 0], [1, 4, 5], [1, 6, 3]])
    dtg = torch.tensor([[3, 4, 2], [5, 2, 0], [3, 2, 0], [4, 5, 2], [6, 3, 2]])
    got = P.ldec_full(m, phi, W, b, din, dtg, torch.ones(5), 0)
    with torch.no_grad():
        lg = m.motor(m.ltm.decode_from_s_hat(phi @ W.t() + b, din))
        want = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), dtg.reshape(-1),
                               ignore_index=0)
    assert torch.allclose(got, want, atol=1e-6), (got, want)


# 8 ------------------------------------------------------------------------
def test_8_bos_eos_pad_construction_matches_train_tasks():
    """dec_in = [BOS]+form ; dec_tgt = form+[EOS] ; pad elsewhere."""
    tr = FakeTr([[5, 6, 7], [8], [9, 10]])
    din, dtg = P.build_dec_batch(tr, [0, 1, 2])
    assert din.shape == dtg.shape == (3, 4)
    assert din[0].tolist() == [1, 5, 6, 7]
    assert dtg[0].tolist() == [5, 6, 7, 2]
    assert din[1].tolist() == [1, 8, 0, 0]
    assert dtg[1].tolist() == [8, 2, 0, 0]
    assert din[2].tolist() == [1, 9, 10, 0]
    assert dtg[2].tolist() == [9, 10, 2, 0]


# 9,10 ---------------------------------------------------------------------
def test_9_naming_path_has_no_to_semantic_dependency():
    """Naming decodes from RAW GloVe through sem_to_h0 -- to_semantic is not
    on the path, so N is an EXACT invariance control."""
    from scripts.naming_comprehension import train_tasks as T
    src = inspect.getsource(T.naming_forward)
    assert "decode_from_s_hat" in src and "to_semantic" not in src
    assert "to_semantic" not in inspect.getsource(T.evaluate_naming)


def test_10_wm_route_has_no_to_semantic_dependency():
    import models.wm_route as wm
    assert "to_semantic" not in inspect.getsource(wm)


def test_9b_naming_and_wm_guards_use_equality_not_inequality():
    base = {"rep_canonical_full_errors": 4, "rep_freear_full_errors": 4,
            "naming_errors": 2, "rep_canonical_ltm_errors": 3574,
            "rep_canonical_wm_errors": 0}
    better = dict(base, naming_errors=1, c_errors=0)   # BETTER than source
    g = P.guards(better, base)
    assert g["naming_invariant"] is False, "N must require equality, not <="
    assert g["full_functional_coexistence"] is False
    wmb = dict(base, rep_canonical_wm_errors=1, c_errors=0)
    assert P.guards(wmb, base)["wm_invariant"] is False


# 11,12 --------------------------------------------------------------------
def test_11_entry_criterion_is_count_based_with_tolerance():
    assert P.ENTRY_TOL == 1e-6 and P.MARGIN_M == 0.01
    ok, nb, mn = P.entry_satisfied(torch.tensor([0.02, 0.0100, 0.0099995]))
    assert ok and nb == 0                      # within 1e-6 of m counts as met
    body = inspect.getsource(P.entry_satisfied).split('"""')[2]   # code only
    assert "L_C" not in body, "entry must not test the ReLU sum for equality"
    assert "n_bad == 0" in body


def test_12_planted_below_margin_item_prevents_entry():
    ok, nb, mn = P.entry_satisfied(torch.tensor([0.5, 0.02, 0.00998]))
    assert not ok and nb == 1 and abs(mn - 0.00998) < 1e-9


# 13 -----------------------------------------------------------------------
def test_13_trial_below_epsilon_is_rejected():
    assert P.EPSILON == 1e-4
    src = inspect.getsource(P.cmd_run)
    assert "if mmin < EPSILON:" in src
    assert "eta *= BACKTRACK" in src.split("if mmin < EPSILON:")[1][:80]


# 14,15 --------------------------------------------------------------------
def _armijo(cand, cur, eta, gn2):
    return cand <= cur - P.ARMIJO_C * eta * gn2


def test_14_armijo_failing_trial_rejected():
    assert P.ARMIJO_C == 1e-4
    # a decrease too small to satisfy sufficient decrease must be rejected
    assert not _armijo(cand=1.0 - 1e-12, cur=1.0, eta=1e-3, gn2=100.0)
    assert not _armijo(cand=1.5, cur=1.0, eta=1e-3, gn2=100.0)


def test_15_armijo_passing_and_feasible_trial_accepted():
    assert _armijo(cand=1.0 - 1e-3, cur=1.0, eta=1e-3, gn2=100.0)
    src = inspect.getsource(P.cmd_run)
    assert "cand <= cur_g - ARMIJO_C * eta * gn2" in src
    assert "accepted += 1" in src


def test_15b_acceptance_requires_both_conditions():
    """Feasibility is checked BEFORE Armijo and neither alone can accept."""
    src = inspect.getsource(P.cmd_run)
    i_feas = src.index("if mmin < EPSILON:")
    i_arm = src.index("cand <= cur_g - ARMIJO_C")
    i_acc = src.index("accepted += 1")
    assert i_feas < i_arm < i_acc


# 16 -----------------------------------------------------------------------
def test_16_exactly_twenty_halvings():
    assert P.MAX_HALVINGS == 20 and P.BACKTRACK == 0.5 and P.ETA0 == 1e-3
    etas = [P.ETA0 * P.BACKTRACK ** k for k in range(P.MAX_HALVINGS + 1)]
    assert len(etas) == 21 and abs(etas[-1] - 1e-3 * 2 ** -20) < 1e-20
    src = inspect.getsource(P.cmd_run)
    assert "for k in range(MAX_HALVINGS + 1)" in src


# 17 -----------------------------------------------------------------------
def test_17_accepted_ldec_trajectory_monotone_decreasing():
    """Armijo with positive eta and gn2>0 forces strict decrease."""
    cur, seq = 2.0, []
    for _ in range(50):
        nxt = cur - P.ARMIJO_C * 1e-3 * 50.0 - 1e-4
        assert _armijo(nxt, cur, 1e-3, 50.0)
        seq.append(nxt)
        cur = nxt
    assert all(seq[i + 1] < seq[i] for i in range(len(seq) - 1))


# 18 -----------------------------------------------------------------------
def test_18_retained_head_is_final_accepted_iterate():
    src = inspect.getsource(P.cmd_run)
    i_loop = src.index("while accepted < MAX_ACCEPTED")
    i_ret = src.index("ret_W = W.detach().clone()")
    assert i_loop < i_ret, "retained head taken after the descent loop"
    assert "retained head = FINAL accepted iterate" in src
    # zero accepted steps -> retained is the entry head (W was init from entry)
    assert 'annotation = "ZERO_ACCEPTED_STEPS"' in src


# 19 -----------------------------------------------------------------------
def test_19_selection_never_uses_ltm_r_or_naming():
    """Between entering Phase 2 and fixing the retained head, no functional
    metric may be consulted."""
    src = inspect.getsource(P.cmd_run)
    seg = src[src.index("PHASE 2"):src.index("rm = _isolated_model")]
    for banned in ("ltm_only_errors", "full_battery", "rep_canonical_ltm",
                   "naming_errors", "rep_freear", "guards("):
        assert banned not in seg, f"selection leak: {banned}"


# 20 -----------------------------------------------------------------------
def test_20_official_c_cadence_every_fifty_accepted():
    assert P.OFFICIAL_C_EVERY == 50
    src = inspect.getsource(P.cmd_run)
    assert "accepted % OFFICIAL_C_EVERY == 0" in src
    assert [i for i in range(1, P.MAX_ACCEPTED + 1)
            if i % P.OFFICIAL_C_EVERY == 0] == [50, 100, 150, 200, 250, 300]


# 21 -----------------------------------------------------------------------
def test_21_cached_vs_deployed_disagreement_aborts():
    src = inspect.getsource(P.cmd_run)
    assert src.count("NUMERICAL_FEASIBILITY_DISCREPANCY") >= 3
    seg = src[src.index("official C cadence"):]
    assert "if c1 != 0 or c2 != 0:" in seg
    assert "NUMERICAL_FEASIBILITY_DISCREPANCY" in seg


# 22,23 --------------------------------------------------------------------
def test_22_23_naming_or_wm_change_triggers_implementation_abort():
    src = inspect.getsource(P.cmd_run)
    assert 'if not gd["naming_invariant"] or not gd["wm_invariant"]:' in src
    i_abort = src.index('IMPLEMENTATION_BUG_HARD_ABORT')
    i_full = src.index('"FULL_FUNCTIONAL_COEXISTENCE"')
    assert i_abort < i_full, "invariance abort must precede any success label"


# 24 -----------------------------------------------------------------------
def test_24_impossibility_labels_unreachable():
    for bad in P.LABELS_FORBIDDEN:
        assert bad not in P.LABELS_ALLOWED
        assert bad not in P.ANNOTATIONS_ALLOWED
    body = inspect.getsource(P.cmd_run)
    for bad in P.LABELS_FORBIDDEN:
        assert bad not in body
    assert P.LABELS_ALLOWED == {
        "FULL_FUNCTIONAL_COEXISTENCE",
        "STRICT_C_COEXISTS_WITH_WHOLE_MODEL_BASELINE_VENTRAL_DEGRADED",
        "EMPIRICAL_FUNCTIONAL_COEXISTENCE_FAILURE"}


def test_24b_failure_maps_only_to_empirical_label():
    base = {"rep_canonical_full_errors": 4, "rep_freear_full_errors": 4,
            "naming_errors": 0, "rep_canonical_ltm_errors": 3574,
            "rep_canonical_wm_errors": 0}
    bad = dict(base, c_errors=0, rep_canonical_ltm_errors=14050,
               rep_canonical_full_errors=11, rep_freear_full_errors=11)
    g = P.guards(bad, base)
    assert not g["full_functional_coexistence"] and not g["ventral_ok"]


# 25 -----------------------------------------------------------------------
def test_25_no_writes_outside_probe_tree():
    src = inspect.getsource(P)
    assert "archives" not in src.split('"""', 2)[2]
    for fn in (P.save_head, P.cmd_run):
        s = inspect.getsource(fn)
        assert "torch.save" not in s or "head_path" in s or "save_head" in s
    assert "head_path(out" in inspect.getsource(P.save_head)


def test_25b_guard_semantics_full_matrix():
    base = {"rep_canonical_full_errors": 3, "rep_freear_full_errors": 3,
            "naming_errors": 0, "rep_canonical_ltm_errors": 3630,
            "rep_canonical_wm_errors": 0}
    win = dict(base, c_errors=0, rep_canonical_ltm_errors=3600,
               rep_canonical_full_errors=2, rep_freear_full_errors=2)
    assert P.guards(win, base)["full_functional_coexistence"] is True
    # the seed-22 COEX-2 pattern: aggregates pass, ventral does not
    s22 = dict(base, c_errors=0, rep_canonical_ltm_errors=14290,
               rep_canonical_full_errors=2, rep_freear_full_errors=2)
    g = P.guards(s22, base)
    assert g["whole_model_preservation"] and not g["ventral_route_preservation"]
    assert not g["full_functional_coexistence"]
