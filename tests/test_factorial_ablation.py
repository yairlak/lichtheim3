"""Acceptance tests for the Phase 4A3 2x2 objective factorial.

The design under test:

    regime   | retrieval | naming
    ---------+-----------+--------
    h0       |    OFF    |  OFF
    c_only   |    ON     |  OFF
    n_only   |    OFF    |  ON
    j0       |    ON     |  ON

Everything except objective presence is held identical, so the factorial
contrasts (c_only - h0), (n_only - h0) and the interaction
(j0 - c_only - n_only + h0) are interpretable. These tests pin exactly that:
shared initialization, shared repetition and dorsal-pool streams, per-cell
semantic-stream activation, the four loss formulas, gradient scope, exact
mid-epoch resume for the two new cells, and evaluation neutrality.

CPU is the reference device; equality assertions are bitwise (`torch.equal`).
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.train_joint_scratch import (               # noqa: E402
    LAMBDA_C, LAMBDA_N, NAMING_REGIMES, REGIMES, RETRIEVAL_REGIMES,
    STREAM_NAMES, TAU, JointScratchTrainer, objective_presence,
)

TINY = dict(device="cpu", max_words=400,
            lexicon_path="data/lexicon_en_glove_covered.tsv",
            dorsal_pool_size=32, batch_size=8, subset_mode="representative",
            subset_per_band=822, subset_size=32, lr_boundary_steps=6,
            allow_glove_fallback=True, require_subset_hash=False,
            glove_path="tests/_no_such_glove_file.txt")

CELLS = ("h0", "c_only", "n_only", "j0")


def make(regime="j0", seed=22, **over):
    kw = dict(TINY)
    kw.update(over)
    return JointScratchTrainer(regime=regime, seed=seed, **kw)


def params(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def assert_same_params(a, b, label):
    assert set(a) == set(b), f"{label}: different parameter names"
    bad = [k for k in a if not torch.equal(a[k], b[k])]
    assert not bad, f"{label}: {len(bad)} tensors differ, e.g. {bad[:4]}"


def optim_tensors(optim):
    out = {}
    for pid, st in optim.state_dict()["state"].items():
        for k, v in st.items():
            out[(pid, k)] = v.detach().clone() if torch.is_tensor(v) else v
    return out


def assert_same_optim(a, b, label):
    assert set(a) == set(b), f"{label}: different optimizer state keys"
    bad = [k for k in a if not (torch.equal(a[k], b[k]) if torch.is_tensor(a[k])
                                else a[k] == b[k])]
    assert not bad, f"{label}: optimizer state differs at {bad[:6]}"


def isnan(x):
    return x != x


# =============================================  regime bookkeeping  ====

def test_the_factorial_has_exactly_four_cells():
    assert REGIMES == ("h0", "c_only", "n_only", "j0")
    assert set(RETRIEVAL_REGIMES) == {"c_only", "j0"}
    assert set(NAMING_REGIMES) == {"n_only", "j0"}
    expected = {"h0": (False, False), "c_only": (True, False),
                "n_only": (False, True), "j0": (True, True)}
    for regime, (r, n) in expected.items():
        p = objective_presence(regime)
        assert (p["retrieval_enabled"], p["naming_enabled"]) == (r, n), regime
    with pytest.raises(ValueError, match="regime must be one of"):
        objective_presence("c0")


def test_trainer_exposes_objective_presence():
    for regime in CELLS:
        tr = make(regime)
        p = objective_presence(regime)
        assert tr.retrieval_enabled == p["retrieval_enabled"]
        assert tr.naming_enabled == p["naming_enabled"]
        assert tr.stream_active["repetition"] and tr.stream_active["pool"]
        assert tr.stream_active["comprehension"] == tr.retrieval_enabled
        assert tr.stream_active["naming"] == tr.naming_enabled


# ================================  A - four-way initialization pairing  ====

def test_A_all_four_cells_share_initialization():
    ref = params(make("h0", seed=22).model)
    for regime in CELLS:
        assert_same_params(ref, params(make(regime, seed=22).model),
                           f"{regime} vs h0 initialization at seed 22")


def test_A_a_different_seed_changes_initialization():
    a = params(make("c_only", seed=22).model)
    b = params(make("c_only", seed=23).model)
    assert any(not torch.equal(a[k], b[k]) for k in a)


# =========================  B/C - four-way R and pool stream pairing  ====

def test_B_repetition_stream_identical_across_all_four_cells():
    trs = {r: make(r) for r in CELLS}
    per_epoch = trs["h0"].streams["repetition"].per_epoch
    for k in range(3 * per_epoch + 5):          # several epochs, ends mid-epoch
        ref = trs["h0"].streams["repetition"].indices(k)
        for r in CELLS:
            assert trs[r].streams["repetition"].indices(k) == ref, \
                f"{r}: R batch {k} differs"


def test_C_pool_stream_identical_across_all_four_cells():
    trs = {r: make(r) for r in CELLS}
    ref_entries = [e.phonemes for e in trs["h0"].pool_entries]
    for r in CELLS:
        assert [e.phonemes for e in trs[r].pool_entries] == ref_entries
    for k in range(3 * trs["h0"].streams["pool"].per_epoch + 3):
        ref = trs["h0"].streams["pool"].indices(k)
        for r in CELLS:
            assert trs[r].streams["pool"].indices(k) == ref, \
                f"{r}: pool batch {k} differs"


def test_BC_streams_stay_paired_through_actual_training():
    """The strong form: R/pool sequences consumed during real optimizer steps
    are identical in all four cells, despite different objectives."""
    trs = {r: make(r) for r in CELLS}
    seen = {r: {"repetition": [], "pool": []} for r in CELLS}
    for _ in range(12):
        for r in CELLS:
            for s in ("repetition", "pool"):
                seen[r][s].append(trs[r].peek_indices(s)[0])
            trs[r].train_step()
    for s in ("repetition", "pool"):
        for r in CELLS:
            assert seen[r][s] == seen["h0"][s], f"{r}: {s} diverged during training"
    for r in CELLS:
        assert trs[r].cursors["repetition"] == trs[r].cursors["pool"] == 12


# =====================  D - semantic stream activation per cell  ====

@pytest.mark.parametrize("regime,c_moves,n_moves", [
    ("h0", False, False), ("c_only", True, False),
    ("n_only", False, True), ("j0", True, True)])
def test_D_semantic_cursor_activation(regime, c_moves, n_moves):
    tr = make(regime)
    for _ in range(9):
        tr.train_step()
    assert tr.cursors["comprehension"] == (9 if c_moves else 0), regime
    assert tr.cursors["naming"] == (9 if n_moves else 0), regime
    assert tr.cursors["repetition"] == tr.cursors["pool"] == 9


def test_D_inactive_streams_are_reported_as_inactive():
    expected = {
        "h0": ["pool", "repetition"],
        "c_only": ["comprehension", "pool", "repetition"],
        "n_only": ["naming", "pool", "repetition"],
        "j0": ["comprehension", "naming", "pool", "repetition"],
    }
    for regime in CELLS:
        s = make(regime).resolved_settings()
        assert s["active_training_streams"] == expected[regime], regime


def test_D_c_and_n_streams_remain_independent_in_j0():
    tr = make("j0")
    assert all(tr.streams["comprehension"].indices(k)
               != tr.streams["naming"].indices(k) for k in range(20))


# ==============================  E - the four objective compositions  ====

def test_E_h0_adds_neither_term():
    rec = make("h0").train_step()
    assert isnan(rec["retrieval_ce"]) and isnan(rec["naming_ce"])
    expected = rec["total"] + 0.5 * rec["pool_ce"]
    assert rec["joint_total"] == pytest.approx(expected, rel=1e-6)


def test_E_c_only_adds_retrieval_only():
    tr = make("c_only")
    rec = tr.train_step()
    assert not isnan(rec["retrieval_ce"]), "retrieval term missing in c_only"
    assert isnan(rec["naming_ce"]), "naming term present in c_only"
    expected = (rec["total"] + tr.cfg.loss.wm * rec["pool_ce"]
                + LAMBDA_C * rec["retrieval_ce"])
    assert rec["joint_total"] == pytest.approx(expected, rel=1e-6)


def test_E_n_only_adds_naming_only():
    tr = make("n_only")
    rec = tr.train_step()
    assert isnan(rec["retrieval_ce"]), "retrieval term present in n_only"
    assert not isnan(rec["naming_ce"]), "naming term missing in n_only"
    expected = (rec["total"] + tr.cfg.loss.wm * rec["pool_ce"]
                + LAMBDA_N * rec["naming_ce"])
    assert rec["joint_total"] == pytest.approx(expected, rel=1e-6)


def test_E_j0_adds_both():
    tr = make("j0")
    rec = tr.train_step()
    assert not isnan(rec["retrieval_ce"]) and not isnan(rec["naming_ce"])
    expected = (rec["total"] + tr.cfg.loss.wm * rec["pool_ce"]
                + LAMBDA_C * rec["retrieval_ce"] + LAMBDA_N * rec["naming_ce"])
    assert rec["joint_total"] == pytest.approx(expected, rel=1e-6)


def test_E_all_cells_produce_finite_losses():
    for regime in CELLS:
        rec = make(regime).train_step()
        for k, v in rec.items():
            if isinstance(v, float) and not isnan(v):
                assert abs(v) < float("inf"), f"{regime}: {k} not finite"


def test_E_effective_weights_reflect_presence():
    for regime in CELLS:
        tr = make(regime)
        s = tr.resolved_settings()
        assert s["lambda_C"] == (LAMBDA_C if tr.retrieval_enabled else 0.0)
        assert s["lambda_N"] == (LAMBDA_N if tr.naming_enabled else 0.0)
        assert s["tau"] == (TAU if tr.retrieval_enabled else None)
        # canonical constants are never rewritten by a regime
        assert s["lambda_C_canonical"] == LAMBDA_C
        assert s["lambda_N_canonical"] == LAMBDA_N
        assert s["tau_canonical"] == TAU


# ===============================  F - gradient scope per objective  ====

RETRIEVAL_SIDE = ("ltm.encoder.", "ltm.to_semantic.")
NAMING_SIDE = ("ltm.sem_to_h0.", "ltm.decoder.", "ltm.dec_to_premotor.")


def grads_from_extra_objectives(regime):
    """Gradient magnitudes per parameter from the ADDED objectives alone.

    The historical base loss is deliberately excluded, so what remains is
    exactly the graph the regime switch activates. Returns {} when the cell
    adds nothing. This is a scope check, not a repeat of Phase 4A0b.
    """
    tr = make(regime)
    tr.model.train(True)
    pad_id = tr.model.vocab.pad_id
    from scripts.naming_comprehension.train_joint_scratch import (
        comprehension_forward, naming_objective, retrieval_loss)

    loss = None
    if tr.retrieval_enabled:
        c = tr.batch("comprehension")
        s_hat = comprehension_forward(tr.model, c["enc_in"], c["enc_mask"])
        loss = LAMBDA_C * retrieval_loss(s_hat, tr.model.ltm.semantic_bank,
                                         c["bank_idx"], TAU)
    if tr.naming_enabled:
        n = tr.batch("naming")
        term = LAMBDA_N * naming_objective(tr.model, n, pad_id)["total"]
        loss = term if loss is None else loss + term
    if loss is None:
        return {}
    tr.optim.zero_grad(set_to_none=True)
    loss.backward()
    return {name: float(p.grad.abs().sum()) if p.grad is not None else 0.0
            for name, p in tr.model.named_parameters()}


def test_F_c_only_activates_the_retrieval_graph_only():
    g = grads_from_extra_objectives("c_only")
    enc = [n for n in g if n.startswith(RETRIEVAL_SIDE)]
    dec = [n for n in g if n.startswith(NAMING_SIDE)]
    assert enc and dec
    assert any(g[n] > 0 for n in enc), "retrieval reached no encoder parameter"
    assert all(g[n] == 0 for n in dec), \
        f"retrieval-only leaked into the naming decoder: " \
        f"{[n for n in dec if g[n] > 0]}"
    assert g["phon_embed.weight"] > 0, "retrieval must reach phon_embed"


def test_F_n_only_activates_the_naming_graph_only():
    g = grads_from_extra_objectives("n_only")
    enc = [n for n in g if n.startswith(RETRIEVAL_SIDE)]
    dec = [n for n in g if n.startswith(NAMING_SIDE)]
    assert any(g[n] > 0 for n in dec), "naming reached no decoder parameter"
    assert all(g[n] == 0 for n in enc), \
        f"naming-only leaked into the retrieval encoder: " \
        f"{[n for n in enc if g[n] > 0]}"
    assert g["motor.proj.weight"] > 0, "naming must reach motor.proj"


def test_F_h0_has_no_extra_objective_graph_at_all():
    g = grads_from_extra_objectives("h0")
    assert g == {}, "h0 built an extra-objective graph"


def test_F_every_parameter_still_trains_in_every_cell():
    """The historical base loss keeps the whole model trainable everywhere."""
    for regime in CELLS:
        tr = make(regime)
        tr.train_step()
        missing = [n for n, p in tr.model.named_parameters()
                   if p.requires_grad and (p.grad is None
                                           or float(p.grad.abs().sum()) == 0.0)]
        assert not missing, f"{regime}: no gradient reached {missing}"


# ======================  G/H - exact mid-epoch resume for the new cells  ====

def _run(tr, n):
    for _ in range(n):
        tr.train_step()
    return tr


def _resume_equivalence(tmp_path, regime, label):
    ref = make(regime)
    per_epoch = ref.streams["repetition"].per_epoch
    stop_at = per_epoch // 2 + 1
    assert stop_at % per_epoch != 0, "stop point must not be epoch-aligned"
    total = 2 * per_epoch + 3                     # crosses two epoch boundaries

    cont = _run(make(regime), total)

    part = _run(make(regime), stop_at)
    ck = tmp_path / f"{label}.pt"
    torch.save(part.state_dict(), ck)

    res = make(regime)
    res.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False),
                        source=str(ck))
    assert res.global_step == stop_at
    _run(res, total - stop_at)

    assert cont.global_step == res.global_step == total
    assert_same_params(params(cont.model), params(res.model), f"{label}: weights")
    assert_same_optim(optim_tensors(cont.optim), optim_tensors(res.optim),
                      f"{label}: optimizer")
    assert cont.cursors == res.cursors, f"{label}: cursors"
    for s in STREAM_NAMES:
        assert cont.peek_indices(s, 6) == res.peek_indices(s, 6), \
            f"{label}: next {s} batches differ"
    return cont


def test_G_exact_mid_epoch_resume_c_only(tmp_path):
    cont = _resume_equivalence(tmp_path, "c_only", "mid_epoch_c_only")
    assert cont.cursors["comprehension"] == cont.global_step
    assert cont.cursors["naming"] == 0


def test_H_exact_mid_epoch_resume_n_only(tmp_path):
    cont = _resume_equivalence(tmp_path, "n_only", "mid_epoch_n_only")
    assert cont.cursors["naming"] == cont.global_step
    assert cont.cursors["comprehension"] == 0


def test_GH_resume_rejects_a_different_factorial_cell(tmp_path):
    tr = _run(make("c_only"), 3)
    p = tmp_path / "c_only.pt"
    torch.save(tr.state_dict(), p)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    for other in ("h0", "n_only", "j0"):
        with pytest.raises(RuntimeError, match="regime"):
            make(other).load_state_dict(ck)


def test_GH_checkpoint_records_the_cell_explicitly():
    for regime in CELLS:
        tr = make(regime)
        tr.train_step()
        ck = tr.state_dict()
        assert ck["regime"] == regime
        s = ck["resolved_settings"]
        assert s["retrieval_enabled"] == tr.retrieval_enabled
        assert s["naming_enabled"] == tr.naming_enabled
        assert set(ck["cursors"]) == set(STREAM_NAMES)
        assert ck["cursors"]["comprehension"] == (1 if tr.retrieval_enabled else 0)
        assert ck["cursors"]["naming"] == (1 if tr.naming_enabled else 0)


# ==============================  I - evaluation neutrality per cell  ====

@pytest.mark.parametrize("regime", CELLS)
def test_I_evaluation_does_not_perturb_training(regime):
    plain = _run(make(regime), 6)
    with_eval = _run(make(regime), 3)
    with_eval.evaluate()
    _run(with_eval, 3)

    assert_same_params(params(plain.model), params(with_eval.model),
                       f"{regime}: weights with an evaluation inserted")
    assert_same_optim(optim_tensors(plain.optim), optim_tensors(with_eval.optim),
                      f"{regime}: optimizer with an evaluation inserted")
    assert plain.cursors == with_eval.cursors, f"{regime}: cursors moved"
    for s in STREAM_NAMES:
        assert plain.peek_indices(s, 5) == with_eval.peek_indices(s, 5), \
            f"{regime}: evaluation changed the {s} stream"


@pytest.mark.parametrize("regime", CELLS)
def test_I_both_semantic_behaviours_are_evaluated_in_every_cell(regime):
    """Cross-task emergence must be measurable even where the objective is off."""
    row = make(regime).evaluate()
    for key in ("comp_top1", "comp_top5", "comp_rank_median", "comp_cos_mean",
                "comp_margin_mean", "naming_exact", "naming_wer",
                "naming_mean_edit", "naming_eos_rate", "naming_pred_len_mean",
                "rep_wm", "rep_ltm", "rep_full"):
        assert key in row, f"{regime}: {key} missing from evaluation"
        assert not isnan(row[key]), f"{regime}: {key} was not computed"
