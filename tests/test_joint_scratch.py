"""Acceptance tests for the Phase 4 joint-from-scratch driver (H0 / J0).

These are the release criteria for the Phase 4A2 scientific run.  They cover:

  A  H0/J0 initialization pairing (and non-pairing across seeds)
  B  repetition-stream pairing across regimes
  C  dorsal-pool pairing across regimes
  D  comprehension/naming stream independence and non-interference
  E  the historical LR boundary: moments preserved, only LR changes
  F  exact resume at an epoch boundary
  G  exact resume MID-epoch, crossing an epoch boundary
  +  evaluation RNG neutrality

CPU is the reference device throughout, and every equality assertion is bitwise
(`torch.equal`), never a tolerance.  Tests build a small lexicon with GloVe
disabled so the 1GB vector file is never parsed; that affects only the semantic
vectors' provenance, not any mechanism under test.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.train_joint_scratch import (              # noqa: E402
    EPOCH_SEED_STRIDE, LAMBDA_C, LAMBDA_N, LR_BOUNDARY_STEPS, LR_STAGE1,
    LR_STAGE2, STREAM_NAMES, STREAM_SEED_OFFSET, STREAM_SEED_STRIDE, TAU,
    CounterStream, JointScratchTrainer, derive_stream_seeds, lr_for_step,
    lr_phase, preserved_rng,
)

# A lexicon small enough for fast tests; GloVe deliberately absent.
TINY = dict(device="cpu", max_words=400,
            lexicon_path="data/lexicon_en_glove_covered.tsv",
            dorsal_pool_size=32, batch_size=8, subset_mode="representative",
            subset_per_band=822, subset_size=32, lr_boundary_steps=6,
            allow_glove_fallback=True, require_subset_hash=False,
            glove_path="tests/_no_such_glove_file.txt")


def make_trainer(regime="j0", seed=22, **over):
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
    """Flat (param index, key) -> value view of every parameter's Adam state.

    Keyed by the positional index from `state_dict()`, not by the Parameter
    object, so states from two different trainer instances are comparable.
    """
    out = {}
    for pid, st in optim.state_dict()["state"].items():
        for k, v in st.items():
            out[(pid, k)] = v.detach().clone() if torch.is_tensor(v) else v
    return out


def assert_same_optim(a, b, label):
    assert set(a) == set(b), f"{label}: different optimizer state keys"
    bad = []
    for k in a:
        x, y = a[k], b[k]
        same = torch.equal(x, y) if torch.is_tensor(x) else x == y
        if not same:
            bad.append(k)
    assert not bad, f"{label}: optimizer state differs at {bad[:6]}"


# ===================================================  seed derivation  ====

def test_stream_seeds_are_derived_not_magic():
    for s in (0, 19, 22, 100):
        seeds = derive_stream_seeds(s)
        assert set(seeds) == set(STREAM_NAMES)
        for name, off in STREAM_SEED_OFFSET.items():
            assert seeds[name] == s * STREAM_SEED_STRIDE + off
        assert len(set(seeds.values())) == len(STREAM_NAMES)
    # distinct experimental seeds never share a stream seed
    all_seeds = [v for s in range(64) for v in derive_stream_seeds(s).values()]
    assert len(set(all_seeds)) == len(all_seeds)


def test_counter_stream_is_a_pure_function_of_the_cursor():
    pop = list(range(50))
    a = CounterStream("t", pop, 8, seed=123)
    b = CounterStream("t", pop, 8, seed=123)
    assert a.per_epoch == 7                      # ceil(50/8)
    for k in range(0, 30):
        assert a.indices(k) == b.indices(k)
    # a fresh stream reconstructs any cursor without replaying earlier ones
    c = CounterStream("t", pop, 8, seed=123)
    assert c.indices(23) == a.indices(23)
    # one epoch covers the population exactly once (no replacement)
    epoch0 = [i for k in range(a.per_epoch) for i in a.indices(k)]
    assert sorted(epoch0) == pop


def test_counter_stream_weighted_draws_with_replacement():
    pop = list(range(20))
    w = [1.0] * 20
    s = CounterStream("r", pop, 5, seed=7, weights=w)
    epoch0 = [i for k in range(s.per_epoch) for i in s.indices(k)]
    assert len(epoch0) == 20
    assert all(0 <= i < 20 for i in epoch0)
    # different epochs draw differently
    epoch1 = [i for k in range(s.per_epoch, 2 * s.per_epoch) for i in s.indices(k)]
    assert epoch0 != epoch1
    assert s.epoch_seed(3) == 7 * EPOCH_SEED_STRIDE + 3


# ==============================================  Test A — init pairing  ====

def test_A_initialization_pairing():
    h0 = make_trainer("h0", seed=22)
    j0 = make_trainer("j0", seed=22)
    assert_same_params(params(h0.model), params(j0.model),
                       "H0/J0 same seed initialization")

    other = make_trainer("j0", seed=23)
    a, b = params(j0.model), params(other.model)
    assert any(not torch.equal(a[k], b[k]) for k in a), \
        "different experimental seeds produced identical initialization"


# ==========================================  Test B — R-stream pairing  ====

def test_B_repetition_stream_pairing_across_regimes():
    h0 = make_trainer("h0", seed=22)
    j0 = make_trainer("j0", seed=22)
    per_epoch = h0.streams["repetition"].per_epoch
    n = 3 * per_epoch + 5                     # several epochs, ends mid-epoch
    for k in range(n):
        assert h0.streams["repetition"].indices(k) == \
               j0.streams["repetition"].indices(k), f"R batch {k} differs"


def test_B_repetition_cursor_advances_identically_during_training():
    h0 = make_trainer("h0", seed=22)
    j0 = make_trainer("j0", seed=22)
    seen_h, seen_j = [], []
    for _ in range(12):
        seen_h.append(h0.peek_indices("repetition")[0])
        h0.train_step()
        seen_j.append(j0.peek_indices("repetition")[0])
        j0.train_step()
    assert seen_h == seen_j
    assert h0.cursors["repetition"] == j0.cursors["repetition"] == 12


# =============================================  Test C — pool pairing  ====

def test_C_dorsal_pool_pairing_across_regimes():
    h0 = make_trainer("h0", seed=22)
    j0 = make_trainer("j0", seed=22)
    assert [e.phonemes for e in h0.pool_entries] == \
           [e.phonemes for e in j0.pool_entries], "pool contents differ"
    for k in range(3 * h0.streams["pool"].per_epoch + 3):
        assert h0.streams["pool"].indices(k) == j0.streams["pool"].indices(k)

    seen_h, seen_j = [], []
    for _ in range(10):
        seen_h.append(h0.peek_indices("pool")[0]); h0.train_step()
        seen_j.append(j0.peek_indices("pool")[0]); j0.train_step()
    assert seen_h == seen_j


# ========================================  Test D — C/N independence  ====

def test_D_comprehension_and_naming_streams_are_independent():
    a = make_trainer("j0", seed=22)
    b = make_trainer("j0", seed=22)
    # deterministic across identical constructions
    for k in range(20):
        assert a.streams["comprehension"].indices(k) == \
               b.streams["comprehension"].indices(k)
        assert a.streams["naming"].indices(k) == b.streams["naming"].indices(k)
    # C and N are different streams over the same population
    same = sum(a.streams["comprehension"].indices(k) ==
               a.streams["naming"].indices(k) for k in range(20))
    assert same == 0, "C and N produced identical batches; streams are not independent"
    assert set(a.streams["comprehension"].population) == set(a.subset_idx)
    assert set(a.streams["naming"].population) == set(a.subset_idx)


def test_D_cn_sampling_does_not_perturb_r_or_pool():
    """The J0 arm draws C and N batches every step; H0 draws none.  If any of
    that consumed shared randomness, the R/pool sequences would diverge."""
    h0 = make_trainer("h0", seed=22)
    j0 = make_trainer("j0", seed=22)
    r_h, r_j, p_h, p_j = [], [], [], []
    for _ in range(15):
        r_h.append(h0.peek_indices("repetition")[0])
        p_h.append(h0.peek_indices("pool")[0])
        h0.train_step()
        r_j.append(j0.peek_indices("repetition")[0])
        p_j.append(j0.peek_indices("pool")[0])
        j0.train_step()
    assert r_h == r_j and p_h == p_j
    # H0 never advances the C/N cursors at all
    assert h0.cursors["comprehension"] == 0 and h0.cursors["naming"] == 0
    assert j0.cursors["comprehension"] == 15 and j0.cursors["naming"] == 15


# ============================================  Test E — LR boundary  ====

def test_E_lr_schedule_is_a_pure_function_of_the_step():
    b = LR_BOUNDARY_STEPS
    assert b == 46_300                       # 100 epochs x 463 steps
    assert lr_for_step(0, b) == LR_STAGE1
    assert lr_for_step(b - 1, b) == LR_STAGE1     # the 46,300th update
    assert lr_for_step(b, b) == LR_STAGE2          # the 46,301st update
    assert lr_phase(b - 1, b) == "stage1_lr1e-3"
    assert lr_phase(b, b) == "stage2_lr1e-4"


def test_E_moments_preserved_across_the_boundary():
    """Crossing the boundary must change lr and nothing else about AdamW."""
    tr = make_trainer("j0", seed=22, lr_boundary_steps=4)
    while tr.global_step < 3:
        tr.train_step()
    assert tr.optim.param_groups[0]["lr"] == LR_STAGE1

    ids_before = {id(g) for g in tr.optim.param_groups}
    optim_obj = tr.optim
    before = optim_tensors(tr.optim)
    steps_before = {pid: float(st["step"]) for pid, st in tr.optim.state.items()}
    groups_before = {k: v for k, v in tr.optim.param_groups[0].items()
                     if k not in ("lr", "params")}

    tr.train_step()                            # the 4th update: last at 1e-3
    assert tr.global_step == 4
    tr.train_step()                            # the 5th update: first at 1e-4
    assert tr.optim.param_groups[0]["lr"] == LR_STAGE2

    # the optimizer object and its param groups were never rebuilt
    assert tr.optim is optim_obj
    assert {id(g) for g in tr.optim.param_groups} == ids_before
    groups_after = {k: v for k, v in tr.optim.param_groups[0].items()
                    if k not in ("lr", "params")}
    assert groups_after == groups_before, "betas/eps/weight_decay changed"

    # moments continued rather than reset
    after = optim_tensors(tr.optim)
    assert set(after) == set(before)
    moved = [k for k in before if torch.is_tensor(before[k])
             and not torch.equal(before[k], after[k])]
    assert moved, "no moment moved at all; the test is not exercising AdamW"
    for pid, st in tr.optim.state.items():
        assert float(st["step"]) == steps_before[pid] + 2, \
            "Adam step counter was reset at the boundary"
        assert float(st["exp_avg"].abs().sum()) > 0
        assert float(st["exp_avg_sq"].abs().sum()) > 0


def test_E_boundary_is_idempotent_under_resume(tmp_path):
    """A run resumed exactly at the boundary must land in stage 2 with its
    moments intact, without reconstructing the optimizer."""
    tr = make_trainer("j0", seed=22, lr_boundary_steps=4)
    for _ in range(4):
        tr.train_step()
    p = tmp_path / "at_boundary.pt"
    torch.save(tr.state_dict(), p)

    fresh = make_trainer("j0", seed=22, lr_boundary_steps=4)
    fresh.load_state_dict(torch.load(p, map_location="cpu", weights_only=False),
                          source=str(p))
    assert fresh.global_step == 4
    assert fresh.optim.param_groups[0]["lr"] == LR_STAGE2
    assert_same_optim(optim_tensors(tr.optim), optim_tensors(fresh.optim),
                      "optimizer state across a boundary resume")


# =================================  Tests F/G — exact resume equivalence  ====

def _run(trainer, n):
    for _ in range(n):
        trainer.train_step()
    return trainer


def _compare_continuous_vs_resumed(tmp_path, *, stop_at, total, label,
                                   regime="j0", boundary=6):
    """Shared body of the epoch-boundary and mid-epoch resume tests."""
    cont = make_trainer(regime, seed=22, lr_boundary_steps=boundary)
    _run(cont, total)

    part = make_trainer(regime, seed=22, lr_boundary_steps=boundary)
    _run(part, stop_at)
    ck = tmp_path / f"{label}.pt"
    torch.save(part.state_dict(), ck)

    resumed = make_trainer(regime, seed=22, lr_boundary_steps=boundary)
    resumed.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False),
                            source=str(ck))
    assert resumed.global_step == stop_at
    _run(resumed, total - stop_at)

    assert resumed.global_step == cont.global_step == total
    assert_same_params(params(cont.model), params(resumed.model),
                       f"{label}: model weights")
    assert_same_optim(optim_tensors(cont.optim), optim_tensors(resumed.optim),
                      f"{label}: optimizer state")
    assert cont.cursors == resumed.cursors, f"{label}: stream cursors"
    for stream in STREAM_NAMES:
        a = cont.peek_indices(stream, 6)
        b = resumed.peek_indices(stream, 6)
        assert a == b, f"{label}: next {stream} batches differ"
    return cont, resumed


def test_F_exact_resume_at_an_epoch_boundary(tmp_path):
    tr = make_trainer("j0", seed=22)
    per_epoch = tr.streams["repetition"].per_epoch
    assert per_epoch > 1
    _compare_continuous_vs_resumed(
        tmp_path, stop_at=per_epoch, total=per_epoch + 7, label="epoch_boundary")


def test_G_exact_resume_mid_epoch_crossing_a_boundary(tmp_path):
    """MANDATORY release criterion: stop at a non-epoch-aligned step, resume,
    and keep going far enough to cross at least one epoch boundary."""
    tr = make_trainer("j0", seed=22)
    per_epoch = tr.streams["repetition"].per_epoch
    stop_at = per_epoch // 2 + 1
    assert stop_at % per_epoch != 0, "stop point must not be epoch-aligned"
    total = 2 * per_epoch + 3                  # crosses two R epoch boundaries
    cont, resumed = _compare_continuous_vs_resumed(
        tmp_path, stop_at=stop_at, total=total, label="mid_epoch")
    assert cont.rep_epoch >= 2, "the test did not cross an epoch boundary"


def test_G_mid_epoch_resume_also_exact_for_h0(tmp_path):
    tr = make_trainer("h0", seed=22)
    per_epoch = tr.streams["repetition"].per_epoch
    _compare_continuous_vs_resumed(
        tmp_path, stop_at=per_epoch // 2 + 1, total=per_epoch + 4,
        label="mid_epoch_h0", regime="h0")


def test_G_resume_rejects_mismatched_provenance(tmp_path):
    tr = make_trainer("j0", seed=22)
    _run(tr, 3)
    p = tmp_path / "j0.pt"
    torch.save(tr.state_dict(), p)
    ck = torch.load(p, map_location="cpu", weights_only=False)

    with pytest.raises(RuntimeError, match="regime"):
        make_trainer("h0", seed=22).load_state_dict(ck)
    with pytest.raises(RuntimeError, match="seed"):
        make_trainer("j0", seed=23).load_state_dict(ck)


# ==========================  evaluation must not perturb the streams  ====

def test_evaluation_rng_neutrality(tmp_path):
    """Inserting an evaluation must not change any subsequent batch, nor the
    resulting weights.  Evaluation cadence must not be an experimental factor."""
    plain = make_trainer("j0", seed=22)
    _run(plain, 6)

    with_eval = make_trainer("j0", seed=22)
    _run(with_eval, 3)
    with_eval.evaluate()
    _run(with_eval, 3)

    assert_same_params(params(plain.model), params(with_eval.model),
                       "weights with an evaluation inserted")
    assert_same_optim(optim_tensors(plain.optim), optim_tensors(with_eval.optim),
                      "optimizer with an evaluation inserted")
    for stream in STREAM_NAMES:
        assert plain.peek_indices(stream, 5) == with_eval.peek_indices(stream, 5), \
            f"evaluation changed the {stream} stream"


def test_preserved_rng_restores_global_state():
    torch.manual_seed(0)
    before = torch.get_rng_state().clone()
    with preserved_rng():
        torch.randn(1000)
    assert torch.equal(before, torch.get_rng_state())


# =========================================  regime / objective wiring  ====

def test_h0_adds_no_retrieval_or_naming_gradient():
    h0 = make_trainer("h0", seed=22)
    rec = h0.train_step()
    assert rec["retrieval_ce"] != rec["retrieval_ce"]     # NaN => not computed
    assert rec["naming_ce"] != rec["naming_ce"]
    settings = h0.resolved_settings()
    assert settings["lambda_C"] == 0.0 and settings["lambda_N"] == 0.0


def test_j0_joint_total_is_the_declared_sum():
    j0 = make_trainer("j0", seed=22)
    rec = j0.train_step()
    expected = (rec["total"] + j0.cfg.loss.wm * rec["pool_ce"]
                + LAMBDA_C * rec["retrieval_ce"] + LAMBDA_N * rec["naming_ce"])
    assert rec["joint_total"] == pytest.approx(expected, rel=1e-6)
    for k in ("total", "rep", "align", "dec", "wm", "gate", "pool_ce",
              "retrieval_ce", "naming_ce", "joint_total", "grad_norm"):
        assert rec[k] == rec[k] and abs(rec[k]) < float("inf"), f"{k} not finite"


def test_all_parameters_receive_gradient_in_both_regimes():
    for regime in ("h0", "j0"):
        tr = make_trainer(regime, seed=22)
        tr.train_step()
        missing = [n for n, p in tr.model.named_parameters()
                   if p.requires_grad and (p.grad is None or
                                           float(p.grad.abs().sum()) == 0.0)]
        assert not missing, f"{regime}: no gradient reached {missing}"


def test_resolved_settings_names_every_critical_value():
    s = make_trainer("j0", seed=22).resolved_settings()
    for key in ("hidden_size", "ltm_encoder_mode", "batch_size",
                "repetition_population", "repetition_sampler",
                "teacher_forcing_ratio", "interference_noise", "ventral_noise",
                "gate_alpha", "gate_threshold", "loss_weights", "lambda_C",
                "lambda_N", "tau", "optimizer", "weight_decay", "grad_clip",
                "current_lr", "lr_boundary_steps", "seed", "stream_seeds"):
        assert key in s, f"{key} missing from the resolved configuration"
    assert s["ltm_encoder_mode"] == "unigru_last_hidden"
    assert s["hidden_size"] == 128
    assert s["gate_alpha"] == 2.0 and s["gate_threshold"] == 0.7
    assert s["interference_noise"] == 0.0 and s["ventral_noise"] == 0.0
    assert s["tau"] == TAU and s["lambda_C"] == LAMBDA_C and s["lambda_N"] == LAMBDA_N


def test_scientific_run_refuses_glove_fallback():
    with pytest.raises(RuntimeError, match="real GloVe"):
        make_trainer("j0", seed=22, allow_glove_fallback=False)


def test_checkpoint_schema_is_complete():
    tr = make_trainer("j0", seed=22)
    tr.train_step()
    ck = tr.state_dict()
    for key in ("format", "regime", "seed", "config", "resolved_settings",
                "model_state_dict", "optimizer_state_dict", "global_step",
                "rep_epoch", "batch_in_rep_epoch", "cursors", "stream_seeds",
                "lr", "lr_phase", "lr_boundary_steps",
                "subset_definition_sha256", "subset_indices", "probe_indices",
                "lexicon_path", "rng_states", "git", "resume_provenance",
                "historical_fidelity"):
        assert key in ck, f"checkpoint schema missing {key}"
    assert set(ck["cursors"]) == set(STREAM_NAMES)
    assert set(ck["rng_states"]) == {"torch", "numpy", "python", "cuda"}
    assert "not a bit-exact replay" in ck["historical_fidelity"].lower() or \
           "NOT a bit-exact replay" in ck["historical_fidelity"]
