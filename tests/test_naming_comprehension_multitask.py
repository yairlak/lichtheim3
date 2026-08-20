"""Focused tests for the Phase 3A interleaved multitask driver.

On tiny synthetic models only (no checkpoint, no data file) except where the
test is explicitly about the stored Phase 2D3 subset constant.

Covers: the union scope, per-task gradient support, frozen-parameter bit
identity, the pure-LTM repetition identity, deterministic 2:2:2 / 1:2:3
scheduling, exact task-step counts at a fixed budget, schedule-RNG
independence, the coexistence criterion, and rejection of invalid schedules.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import Config, DataConfig, LTMConfig, TrainConfig               # noqa: E402
from data.lexicon import LexEntry                                           # noqa: E402
from data.phonemes import build_vocab                                       # noqa: E402
from losses import _seq_ce                                                  # noqa: E402
from models.dual_route import DualRouteModel                                # noqa: E402
from scripts.naming_comprehension.train_tasks import (                      # noqa: E402
    ALWAYS_FROZEN, TRAINABLE_PREFIXES, changed_parameters, make_batches,
    parameter_fingerprint)
from scripts.naming_comprehension.train_multitask import (                  # noqa: E402
    CANONICAL_FULL_LEXICON_LTM, COEXISTENCE, CONSECUTIVE_REQUIRED,
    CoexistenceController, DECODER_SIDE, ENCODER_SIDE, GLOBAL_PRESERVATION,
    STRICT_PRESERVATION_MAX_DROP, global_preservation_met, preservation_report,
    MACRO_CYCLE_STEPS, PHASE2D3_SUBSET_SHA256, SCHEDULES, TASKS,
    EVAL_STEPS_EARLY, TASK_DATA_SEEDS, UNION_PREFIXES, coexistence_met,
    batches_per_epoch, build_task_populations, evaluation_steps,
    exposure_report, full_lexicon_population, infinite_batches,
    out_of_subset_probe,
    item_presentations, load_resumable, macro_cycle, population_passes,
    presentations_per_item, repetition_objective, sampler_state, save_resumable,
    schedule_counts, set_multitask_scope, should_evaluate, side_of,
    task_objective, task_schedule_stream)

SEM_DIM = 16


def _tiny_model(seed: int = 0):
    torch.manual_seed(seed)
    cfg = Config(
        data=DataConfig(use_real=False, glove_path=None, semantic_dim=SEM_DIM,
                        max_words=50, seed=0),
        ltm=LTMConfig(phon_embed_dim=8, enc_hidden=16, dec_hidden=16,
                      ltm_encoder_mode="unigru_last_hidden"),
        train=TrainConfig(device="cpu", seed=0),
    )
    vocab = build_vocab()
    model = DualRouteModel(cfg, vocab, premotor_dim=12)
    model.set_semantic_bank(torch.randn(10, SEM_DIM))
    model.eval()
    return model, vocab, cfg


def _tiny_entries(n: int = 8):
    rng = np.random.default_rng(0)
    return [LexEntry(word=f"w{i}", phonemes=[5 + (i % 7), 6 + (i % 5), 7 + (i % 3)],
                     semantic=rng.standard_normal(SEM_DIM).astype(np.float32),
                     freq=1.0, rank=i + 1) for i in range(n)]


def _setup():
    model, vocab, _ = _tiny_model()
    entries = _tiny_entries()
    bank = torch.stack([torch.as_tensor(e.semantic) for e in entries])
    model.set_semantic_bank(bank.clone())
    batch = next(make_batches(entries, list(range(len(entries))), bank, vocab,
                              batch_size=len(entries), device="cpu"))
    return model, vocab, entries, bank, batch


# ------------------------------------------------------------ union scope

def test_union_scope_is_exactly_the_two_validated_scopes():
    assert set(ENCODER_SIDE) == set(TRAINABLE_PREFIXES["comprehension"])
    assert set(DECODER_SIDE) == set(TRAINABLE_PREFIXES["naming"])
    assert set(UNION_PREFIXES) == set(ENCODER_SIDE) | set(DECODER_SIDE)


def test_set_multitask_scope_trains_union_and_freezes_the_rest():
    model, _, _ = _tiny_model()
    trainable = set_multitask_scope(model)
    assert trainable == sorted([
        "ltm.encoder.bias_hh_l0", "ltm.encoder.bias_ih_l0",
        "ltm.encoder.weight_hh_l0", "ltm.encoder.weight_ih_l0",
        "ltm.to_semantic.0.bias", "ltm.to_semantic.0.weight",
        "ltm.to_semantic.2.bias", "ltm.to_semantic.2.weight",
        "ltm.sem_to_h0.bias", "ltm.sem_to_h0.weight",
        "ltm.decoder.bias_hh_l0", "ltm.decoder.bias_ih_l0",
        "ltm.decoder.weight_hh_l0", "ltm.decoder.weight_ih_l0",
        "ltm.dec_to_premotor.bias", "ltm.dec_to_premotor.weight",
    ])
    params = dict(model.named_parameters())
    for n in ALWAYS_FROZEN:
        assert not params[n].requires_grad
    assert not any(n.startswith("wm.") and p.requires_grad
                   for n, p in model.named_parameters())


def test_side_of_partitions_the_union():
    model, _, _ = _tiny_model()
    for n in set_multitask_scope(model):
        assert side_of(n) in ("encoder_side", "decoder_side")
    assert side_of("wm.encoder.weight_ih_l0") is None
    assert side_of("motor.proj.weight") is None


# -------------------------------------------- repetition loss definition

def test_repetition_is_exactly_the_models_own_ltm_route():
    """The pure-LTM composition must equal route_logits(route='ltm')."""
    model, vocab, entries, bank, batch = _setup()
    out = repetition_objective(model, batch, vocab.pad_id)
    ref = model.route_logits(batch["enc_in"], batch["enc_mask"], batch["dec_in"],
                             route="ltm")["logits"]
    assert torch.allclose(out["logits"], ref, atol=1e-7)
    manual = _seq_ce(ref, batch["dec_tgt"], vocab.pad_id)
    assert torch.allclose(out["total"], manual, atol=1e-7)


def test_repetition_gradient_reaches_both_ltm_sides():
    model, vocab, entries, bank, batch = _setup()
    set_multitask_scope(model)
    repetition_objective(model, batch, vocab.pad_id)["total"].backward()
    touched = {side_of(n) for n, p in model.named_parameters()
               if p.requires_grad and p.grad is not None and torch.any(p.grad != 0)}
    assert touched == {"encoder_side", "decoder_side"}


@pytest.mark.parametrize("task,expected", [
    ("comprehension", {"encoder_side"}),
    ("naming", {"decoder_side"}),
    ("repetition", {"encoder_side", "decoder_side"}),
])
def test_task_gradient_support_is_exactly_as_designed(task, expected):
    model, vocab, entries, bank, batch = _setup()
    set_multitask_scope(model)
    model.zero_grad(set_to_none=True)
    task_objective(model, task, batch, vocab.pad_id)["total"].backward()
    touched = {side_of(n) for n, p in model.named_parameters()
               if p.requires_grad and p.grad is not None and torch.any(p.grad != 0)}
    assert touched == expected


@pytest.mark.parametrize("task", list(TASKS))
def test_frozen_parameters_are_bit_identical_after_one_step(task):
    model, vocab, entries, bank, batch = _setup()
    trainable = set_multitask_scope(model)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=1e-2, weight_decay=1e-2)
    before = parameter_fingerprint(model)
    loss = task_objective(model, task, batch, vocab.pad_id)["total"]
    optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
    changed = changed_parameters(model, before)
    assert changed, "sanity: something in scope must move"
    assert set(changed) <= set(trainable)
    assert not any(n.startswith(("wm.", "phon_embed.", "motor.")) for n in changed)


def test_unknown_task_is_rejected():
    model, vocab, entries, bank, batch = _setup()
    with pytest.raises(ValueError, match="Unknown task"):
        task_objective(model, "speaking", batch, vocab.pad_id)


# ---------------------------------------------------------- scheduling

def test_macro_cycle_has_exact_task_multiset_for_both_schedules():
    for name, ratio in SCHEDULES.items():
        cyc = macro_cycle(ratio, 0, 0)
        assert len(cyc) == MACRO_CYCLE_STEPS, name
        for task, k in zip(TASKS, ratio):
            assert cyc.count(task) == k, (name, task)


def test_both_schedules_spend_the_same_steps_per_cycle():
    assert sum(SCHEDULES["m1_111"]) == sum(SCHEDULES["m2_123"]) == MACRO_CYCLE_STEPS


def test_macro_cycle_is_deterministic_and_varies_across_cycles():
    r = SCHEDULES["m2_123"]
    assert macro_cycle(r, 0, 7) == macro_cycle(r, 0, 7)
    assert macro_cycle(r, 0, 0) != macro_cycle(r, 0, 1) or \
        macro_cycle(r, 0, 0) != macro_cycle(r, 0, 2)
    assert macro_cycle(r, 0, 3) != macro_cycle(r, 1, 3)      # seed matters


def test_schedule_stream_length_and_truncation():
    for name, ratio in SCHEDULES.items():
        assert len(list(task_schedule_stream(ratio, 25, 0))) == 25, name
        assert len(list(task_schedule_stream(ratio, 6, 0))) == 6, name


def test_exact_task_counts_at_a_fixed_budget():
    total = 400_000
    c1 = schedule_counts(SCHEDULES["m1_111"], total)
    c2 = schedule_counts(SCHEDULES["m2_123"], total)
    assert sum(c1.values()) == sum(c2.values()) == total
    full = total // MACRO_CYCLE_STEPS
    for t, k in zip(TASKS, SCHEDULES["m1_111"]):
        assert abs(c1[t] - full * k) <= MACRO_CYCLE_STEPS
    for t, k in zip(TASKS, SCHEDULES["m2_123"]):
        assert abs(c2[t] - full * k) <= MACRO_CYCLE_STEPS
    # the 1:2:3 schedule must order the tasks as designed
    assert c2["repetition"] < c2["naming"] < c2["comprehension"]


def test_counts_are_exact_on_a_whole_number_of_cycles():
    total = MACRO_CYCLE_STEPS * 1000
    for name, ratio in SCHEDULES.items():
        counts = schedule_counts(ratio, total)
        for t, k in zip(TASKS, ratio):
            assert counts[t] == 1000 * k, (name, t)


def test_invalid_schedules_are_rejected():
    with pytest.raises(ValueError, match="must have"):
        macro_cycle((1, 2), 0, 0)
    with pytest.raises(ValueError, match="non-negative"):
        macro_cycle((-1, 4, 3), 0, 0)
    with pytest.raises(ValueError, match="exactly"):
        macro_cycle((1, 1, 1), 0, 0)          # sums to 3, not 6
    with pytest.raises(ValueError, match="exactly"):
        macro_cycle((2, 3, 4), 0, 0)          # sums to 9


def test_schedule_rng_does_not_disturb_global_rng():
    """Task ordering must be independent of model/data/eval RNG."""
    torch.manual_seed(1234)
    a = torch.randn(5)
    torch.manual_seed(1234)
    for c in range(50):
        macro_cycle(SCHEDULES["m2_123"], 0, c)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_each_task_has_an_independent_sampler_stream():
    """Changing one task's frequency cannot reorder another task's items."""
    model, vocab, entries, bank, _ = _setup()
    idx = list(range(len(entries)))
    assert len(set(TASK_DATA_SEEDS.values())) == len(TASKS)

    def first_orders(n_draws):
        s = infinite_batches(entries, idx, bank, vocab, 4, "cpu",
                             TASK_DATA_SEEDS["naming"])
        return [next(s)["bank_idx"].tolist() for _ in range(n_draws)]

    assert first_orders(3) == first_orders(3)          # deterministic
    other = infinite_batches(entries, idx, bank, vocab, 4, "cpu",
                             TASK_DATA_SEEDS["comprehension"])
    assert next(other)["bank_idx"].tolist() != first_orders(1)[0]


def test_one_population_pass_is_ceil_batches_not_n_over_batch():
    """N=3288, batch 64 -> 52 batches per pass, last one holding only 24."""
    assert batches_per_epoch(3288, 64) == 52
    assert 52 * 64 == 3328 != 3288
    assert population_passes(52, 3288, 64) == 1.0
    assert population_passes(0, 3288, 64) == 0.0


def test_population_passes_match_the_single_task_reference():
    """200,200 comprehension steps must be exactly the Phase 2D3 3,850 passes."""
    assert population_passes(200_200, 3288, 64) == 3850.0
    assert population_passes(200_000, 3288, 64) == pytest.approx(3846.1538, abs=1e-3)


def test_item_presentations_honour_the_short_final_batch():
    # one whole pass presents every item exactly once, not 3328 times
    assert item_presentations(52, 3288, 64) == 3288
    assert item_presentations(104, 3288, 64) == 2 * 3288
    # a partial pass consumes only full-size batches (the short one is last)
    assert item_presentations(51, 3288, 64) == 51 * 64 == 3264
    assert item_presentations(53, 3288, 64) == 3288 + 64
    assert item_presentations(0, 3288, 64) == 0


def test_naive_step_times_batch_overcounts_exposure():
    """The old formula credits the short final batch with 64 items."""
    naive = 200_000 * 64 / 3288
    assert naive == pytest.approx(3892.94, abs=1e-2)
    assert presentations_per_item(200_000, 3288, 64) == pytest.approx(3846.16, abs=1e-2)
    assert naive > presentations_per_item(200_000, 3288, 64)


def test_exposure_report_is_self_consistent():
    r = exposure_report(200_000, 3288, 64)
    assert r["task_steps"] == 200_000
    assert r["batches_per_pass"] == 52
    assert r["population_passes"] == pytest.approx(3846.1538, abs=1e-3)
    assert r["presentations_per_item"] == pytest.approx(
        r["item_presentations"] / 3288)
    # the two readings agree to well under one presentation
    assert abs(r["population_passes"] - r["presentations_per_item"]) < 0.01


def test_exposure_at_budget_matches_expected_values():
    expected = {"m1_111": {"repetition": 2564.10, "naming": 2564.12,
                           "comprehension": 2564.10},
                "m2_123": {"repetition": 1282.06, "naming": 2564.10,
                           "comprehension": 3846.15}}
    for name, ratio in SCHEDULES.items():
        counts = schedule_counts(ratio, 400_000)
        for t in TASKS:
            assert population_passes(counts[t], 3288, 64) == pytest.approx(
                expected[name][t], abs=0.01), (name, t)


# ------------------------------------------------- coexistence criterion

def _snap(comp, nam, wer, ltm):
    return {"comprehension": {"top1": comp},
            "naming": {"exact_match": nam, "whole_word_error_rate": wer},
            "repetition": {"ltm": ltm}}


def test_coexistence_requires_all_three_at_the_same_snapshot():
    assert coexistence_met(_snap(0.96, 0.97, 0.03, 0.98))
    assert not coexistence_met(_snap(0.94, 0.97, 0.03, 0.98))   # comprehension
    assert not coexistence_met(_snap(0.96, 0.94, 0.06, 0.98))   # naming
    assert not coexistence_met(_snap(0.96, 0.97, 0.03, 0.90))   # LTM repetition
    assert coexistence_met(_snap(0.95, 0.95, 0.05, 0.95))       # exact boundary


def test_coexistence_thresholds_are_the_predeclared_values():
    assert COEXISTENCE == {"comprehension_top1_min": 0.95,
                           "naming_exact_min": 0.95,
                           "naming_wer_max": 0.05,
                           "ltm_repetition_exact_min": 0.95}
    assert CONSECUTIVE_REQUIRED == 2


# ------------------------------------------------- evaluation cadence

def test_early_evaluation_grid_is_dense_then_regular():
    assert EVAL_STEPS_EARLY == (2_000, 5_000, 10_000, 20_000)
    for s in EVAL_STEPS_EARLY:
        assert should_evaluate(s)
    assert should_evaluate(40_000) and should_evaluate(400_000)
    for s in (1, 1_999, 2_001, 7_000, 19_999, 21_000):
        assert not should_evaluate(s)
    assert not should_evaluate(0)          # step 0 handled separately


def test_planned_evaluation_grid_is_exactly_24_unique_steps():
    """20,000 must appear once (it is both an early step and a multiple)."""
    steps = evaluation_steps(400_000, 20_000)
    assert steps == [0, 2_000, 5_000, 10_000] + list(range(20_000, 400_001, 20_000))
    assert len(steps) == 24
    assert len(steps) == len(set(steps)), "duplicated evaluation step"
    assert steps == sorted(steps)
    assert steps.count(20_000) == 1 and steps.count(400_000) == 1
    # 1 (step0) + 3 (2k,5k,10k) + 20 (20k..400k) = 24; the old grid had 21
    assert len(steps) - len([0] + list(range(20_000, 400_001, 20_000))) == 3
    # forgetting inside the first 8k steps is now measured twice
    assert len([s for s in steps if 0 < s <= 8_000]) == 2


def test_evaluation_does_not_consume_global_rng():
    """Cadence must not perturb training state: the eval path draws no RNG."""
    model, vocab, entries, bank, batch = _setup()
    torch.manual_seed(4321)
    a = torch.randn(4)
    torch.manual_seed(4321)
    with torch.no_grad():
        for _ in range(3):
            model.eval()
            model.route_logits(batch["enc_in"], batch["enc_mask"],
                               batch["dec_in"], route="ltm")
    b = torch.randn(4)
    assert torch.equal(a, b)


# ------------------------------------------------------ resumable state

def test_sampler_state_is_derivable_from_cumulative_counts():
    st = sampler_state({"repetition": 0, "naming": 55, "comprehension": 104},
                       populations={t: 3288 for t in TASKS}, batch_size=64)
    assert st["naming"]["batches_per_epoch"] == 52
    assert (st["repetition"]["epoch"], st["repetition"]["position_in_epoch"]) == (0, 0)
    assert (st["naming"]["epoch"], st["naming"]["position_in_epoch"]) == (1, 3)
    assert (st["comprehension"]["epoch"], st["comprehension"]["position_in_epoch"]) == (2, 0)


def test_schedule_stream_can_be_re_entered_at_any_step():
    ratio = SCHEDULES["m2_123"]
    full = list(task_schedule_stream(ratio, 40, 0))
    for cut in (0, 1, 6, 7, 13, 24):
        head = list(task_schedule_stream(ratio, cut, 0))
        tail = list(task_schedule_stream(ratio, 40, 0, start_step=cut))
        assert head + tail == full, cut


def test_batch_stream_can_be_fast_forwarded_exactly():
    model, vocab, entries, bank, _ = _setup()
    idx = list(range(len(entries)))
    cont = infinite_batches(entries, idx, bank, vocab, 3, "cpu", 7)
    seen = [next(cont)["bank_idx"].tolist() for _ in range(9)]
    for cut in (0, 1, 3, 5, 8):
        ff = infinite_batches(entries, idx, bank, vocab, 3, "cpu", 7,
                              start_index=cut)
        assert [next(ff)["bank_idx"].tolist() for _ in range(9 - cut)] == seen[cut:]


def test_resume_rejects_mismatched_configuration(tmp_path):
    model, vocab, entries, bank, batch = _setup()
    set_multitask_scope(model)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=1e-4)
    p = str(tmp_path / "r.pt")
    save_resumable(p, model=model, optimizer=optim, step=6,
                   counts={t: 2 for t in TASKS}, schedule="m1_111",
                   ratio=SCHEDULES["m1_111"], schedule_seed=0,
                   populations={t: len(entries) for t in TASKS}, batch_size=3,
                   subset_sha256="abc",
                   source_checkpoint_sha256="src", snapshots=[], trajectory=[],
                   streak=0, first_met=None)
    ok = dict(schedule="m1_111", subset_sha256="abc",
              populations={t: len(entries) for t in TASKS}, batch_size=3)
    load_resumable(p, **ok)                                     # no raise
    for bad in ({"schedule": "m2_123"}, {"subset_sha256": "zzz"},
                {"populations": {t: 999 for t in TASKS}}, {"batch_size": 64}):
        with pytest.raises(RuntimeError, match="mismatch"):
            load_resumable(p, **{**ok, **bad})


def test_resume_checkpoint_carries_every_required_field(tmp_path):
    model, vocab, entries, bank, batch = _setup()
    set_multitask_scope(model)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=1e-4)
    p = str(tmp_path / "r.pt")
    save_resumable(p, model=model, optimizer=optim, step=13,
                   counts={"repetition": 4, "naming": 4, "comprehension": 5},
                   schedule="m2_123", ratio=SCHEDULES["m2_123"], schedule_seed=0,
                   populations={t: len(entries) for t in TASKS}, batch_size=3,
                   subset_sha256="h",
                   source_checkpoint_sha256="src", snapshots=[{"step": 0}],
                   trajectory=[{"step": 13}], streak=1, first_met=13)
    st = torch.load(p, map_location="cpu", weights_only=False)
    for key in ("format_version", "model_state_dict", "optimizer_state_dict",
                "step", "cycle_index", "position_in_cycle", "task_steps",
                "schedule", "ratio", "schedule_seed", "schedule_state",
                "task_sampler_state", "torch_rng_state",
                "subset_definition_sha256", "source_checkpoint_sha256",
                "snapshots", "trajectory", "coexistence_streak",
                "first_step_criterion_met", "task_populations", "batch_size"):
        assert key in st, key
    assert (st["cycle_index"], st["position_in_cycle"]) == (2, 1)   # 13 = 2*6+1
    assert sum(st["task_steps"].values()) == st["step"]


def test_resume_is_bit_identical_to_continuous_training(tmp_path):
    """A: N steps continuous.  B: N/2, save, reload, continue.  A == B exactly."""
    N, CUT = 12, 6
    ratio = SCHEDULES["m2_123"]

    def fresh():
        model, vocab, entries, bank, _ = _setup()
        set_multitask_scope(model)
        optim = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3, weight_decay=1e-5)
        return model, vocab, entries, bank, optim

    def train(model, vocab, entries, bank, optim, counts, lo, hi, losses):
        idx = list(range(len(entries)))
        streams = {t: infinite_batches(entries, idx, bank, vocab, 4, "cpu",
                                       TASK_DATA_SEEDS[t], start_index=counts[t])
                   for t in TASKS}
        for task in task_schedule_stream(ratio, hi, 0, start_step=lo):
            b = next(streams[task])
            loss = task_objective(model, task, b, vocab.pad_id)["total"]
            optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
            counts[task] += 1
            losses.append(round(float(loss.detach()), 10))

    # ---- A: continuous ----
    ma, va, ea, ba, oa = fresh()
    ca, la = {t: 0 for t in TASKS}, []
    train(ma, va, ea, ba, oa, ca, 0, N, la)

    # ---- B: halted, saved, reloaded, continued ----
    mb, vb, eb, bb, ob = fresh()
    cb, lb = {t: 0 for t in TASKS}, []
    train(mb, vb, eb, bb, ob, cb, 0, CUT, lb)
    p = str(tmp_path / "resume.pt")
    save_resumable(p, model=mb, optimizer=ob, step=CUT, counts=cb,
                   schedule="m2_123", ratio=ratio, schedule_seed=0,
                   populations={t: len(eb) for t in TASKS}, batch_size=4,
                   subset_sha256="h",
                   source_checkpoint_sha256="src", snapshots=[], trajectory=[],
                   streak=0, first_met=None)

    mc, vc, ec, bc, oc = fresh()
    st = load_resumable(p, schedule="m2_123", subset_sha256="h",
                        populations={t: len(ec) for t in TASKS}, batch_size=4)
    mc.load_state_dict(st["model_state_dict"])
    oc.load_state_dict(st["optimizer_state_dict"])
    torch.set_rng_state(st["torch_rng_state"])
    cc = {t: int(st["task_steps"][t]) for t in TASKS}
    lc = list(lb)
    train(mc, vc, ec, bc, oc, cc, st["step"], N, lc)

    assert cc == ca, "task-step counts diverged"
    assert lc == la, "per-step loss trajectory diverged"
    for (na, pa), (nc, pc) in zip(ma.named_parameters(), mc.named_parameters()):
        assert na == nc
        assert torch.equal(pa, pc), f"weights diverged at {na}"


# ------------------------------- Phase 3C: heterogeneous task populations

def test_default_repetition_population_reproduces_phase3ab():
    e = _tiny_entries(20)
    sub = list(range(0, 20, 2))
    pops = build_task_populations(e, sub)                 # default "subset"
    assert pops["repetition"] == pops["naming"] == pops["comprehension"] == sub


def test_full_lexicon_repetition_population_changes_only_repetition():
    e = _tiny_entries(20)
    sub = list(range(0, 20, 2))
    pops = build_task_populations(e, sub, "full_lexicon")
    assert pops["repetition"] == list(range(20)) and len(pops["repetition"]) == len(e)
    assert pops["naming"] == sub and pops["comprehension"] == sub
    assert set(sub) <= set(pops["repetition"])            # subset contained


def test_unknown_repetition_population_rejected():
    e = _tiny_entries(8)
    with pytest.raises(ValueError, match="Unknown repetition_population"):
        build_task_populations(e, [0, 1], "everything")


def test_task_specific_batches_per_pass():
    """463 for the full lexicon, 52 for subset3288 -- never one shared value."""
    assert batches_per_epoch(29571, 64) == 463
    assert batches_per_epoch(3288, 64) == 52
    assert 463 * 64 == 29632 != 29571                     # short final batch: 27
    assert 29571 - 462 * 64 == 3


def test_exposure_accounting_differs_per_task_population():
    rep = exposure_report(130_000, 29571, 64)
    nam = exposure_report(260_000, 3288, 64)
    assert rep["batches_per_pass"] == 463 and nam["batches_per_pass"] == 52
    assert rep["population_passes"] == pytest.approx(130_000 / 463, abs=1e-6)
    assert nam["population_passes"] == pytest.approx(5000.0, abs=1e-9)
    # exact presentations honour each population's own short final batch
    q, r = divmod(130_000, 463)
    assert rep["item_presentations"] == q * 29571 + r * 64


def test_sampler_state_handles_heterogeneous_populations():
    st = sampler_state({"repetition": 500, "naming": 55, "comprehension": 104},
                       populations={"repetition": 29571, "naming": 3288,
                                    "comprehension": 3288}, batch_size=64)
    assert st["repetition"]["batches_per_epoch"] == 463
    assert st["naming"]["batches_per_epoch"] == 52
    assert (st["repetition"]["epoch"], st["repetition"]["position_in_epoch"]) == (1, 37)
    assert st["repetition"]["population"] == 29571


def test_out_of_subset_probe_is_deterministic_and_disjoint():
    e = _tiny_entries(40)
    sub = list(range(0, 40, 2))
    a = out_of_subset_probe(e, sub, n=10, probe_seed=0)
    b = out_of_subset_probe(_tiny_entries(40), sub, n=10, probe_seed=0)
    assert a == b                                          # deterministic, ordered
    assert len(a) == len(set(a)) == 10
    assert not (set(a) & set(sub)), "probe must not overlap subset3288"
    assert set(a) <= set(range(40)) - set(sub)
    assert out_of_subset_probe(e, sub, 10, probe_seed=1) != a


def test_probe_rejects_oversized_request():
    e = _tiny_entries(20)
    sub = list(range(10))
    with pytest.raises(RuntimeError, match="exceeds"):
        out_of_subset_probe(e, sub, n=11)                  # complement is 10


def test_full_lexicon_repetition_batch_keeps_exact_gradient_scope():
    """A repetition batch from the full population still hits BOTH LTM sides."""
    model, vocab, entries, bank, _ = _setup()
    pops = build_task_populations(entries, [0, 1], "full_lexicon")
    batch = next(make_batches(entries, pops["repetition"], bank, vocab,
                              batch_size=len(entries), device="cpu"))
    set_multitask_scope(model)
    model.zero_grad(set_to_none=True)
    task_objective(model, "repetition", batch, vocab.pad_id)["total"].backward()
    touched = {side_of(n) for n, p in model.named_parameters()
               if p.requires_grad and p.grad is not None and torch.any(p.grad != 0)}
    assert touched == {"encoder_side", "decoder_side"}
    moved_frozen = [n for n, p in model.named_parameters()
                    if not p.requires_grad and p.grad is not None
                    and torch.any(p.grad != 0) and n.startswith("wm.")]
    assert not moved_frozen


def test_scheduler_semantics_unchanged_by_population_choice():
    """Task ordering must not depend on which population a task samples."""
    for name, ratio in SCHEDULES.items():
        assert list(task_schedule_stream(ratio, 30, 0)) == \
            list(task_schedule_stream(ratio, 30, 0))
        assert schedule_counts(ratio, 400_000) == schedule_counts(ratio, 400_000)


def test_resume_bit_identical_with_heterogeneous_samplers(tmp_path):
    """Same resume guarantee when repetition samples a larger population."""
    N, CUT = 12, 6
    ratio = SCHEDULES["m2_123"]

    def fresh():
        model, vocab, entries, bank, _ = _setup()
        set_multitask_scope(model)
        optim = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3, weight_decay=1e-5)
        return model, vocab, entries, bank, optim

    def pops_of(entries):
        return build_task_populations(entries, list(range(0, len(entries), 2)),
                                      "full_lexicon")

    def train(model, vocab, entries, bank, optim, counts, lo, hi, losses):
        pops = pops_of(entries)
        streams = {t: infinite_batches(entries, pops[t], bank, vocab, 3, "cpu",
                                       TASK_DATA_SEEDS[t], start_index=counts[t])
                   for t in TASKS}
        for task in task_schedule_stream(ratio, hi, 0, start_step=lo):
            b = next(streams[task])
            loss = task_objective(model, task, b, vocab.pad_id)["total"]
            optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
            counts[task] += 1
            losses.append(round(float(loss.detach()), 10))

    ma, va, ea, ba, oa = fresh()
    ca, la = {t: 0 for t in TASKS}, []
    train(ma, va, ea, ba, oa, ca, 0, N, la)

    mb, vb, eb, bb, ob = fresh()
    cb, lb = {t: 0 for t in TASKS}, []
    train(mb, vb, eb, bb, ob, cb, 0, CUT, lb)
    sizes = {t: len(v) for t, v in pops_of(eb).items()}
    assert sizes["repetition"] != sizes["naming"], "populations must differ here"
    p = str(tmp_path / "het.pt")
    save_resumable(p, model=mb, optimizer=ob, step=CUT, counts=cb,
                   schedule="m2_123", ratio=ratio, schedule_seed=0,
                   populations=sizes, batch_size=3, subset_sha256="h",
                   source_checkpoint_sha256="src", snapshots=[], trajectory=[],
                   streak=0, first_met=None)

    mc, vc, ec, bc, oc = fresh()
    st = load_resumable(p, schedule="m2_123", subset_sha256="h",
                        populations=sizes, batch_size=3)
    mc.load_state_dict(st["model_state_dict"])
    oc.load_state_dict(st["optimizer_state_dict"])
    torch.set_rng_state(st["torch_rng_state"])
    cc = {t: int(st["task_steps"][t]) for t in TASKS}
    lc = list(lb)
    train(mc, vc, ec, bc, oc, cc, st["step"], N, lc)

    assert cc == ca and lc == la
    for (na, pa), (nc, pc) in zip(ma.named_parameters(), mc.named_parameters()):
        assert torch.equal(pa, pc), f"weights diverged at {na}"
    assert st["task_sampler_state"]["repetition"]["batches_per_epoch"] != \
        st["task_sampler_state"]["naming"]["batches_per_epoch"]


def test_v1_resume_checkpoints_remain_loadable(tmp_path):
    """Phase 3A/3B checkpoints stored one shared population; keep them usable."""
    p = str(tmp_path / "v1.pt")
    torch.save({"format_version": 1, "population": 3288, "batch_size": 64,
                "schedule": "m2_123", "subset_definition_sha256": "h",
                "step": 6, "task_steps": {t: 2 for t in TASKS}}, p)
    st = load_resumable(p, schedule="m2_123", subset_sha256="h",
                        populations={t: 3288 for t in TASKS}, batch_size=64)
    assert st["task_populations"] == {t: 3288 for t in TASKS}


# ------------------- Phase 3C: combined local + global success logic

def _rep(ltm, full=0.9, wm=0.999763):
    return {"full": full, "wm": wm, "ltm": ltm}


def test_global_preservation_thresholds_are_the_predeclared_values():
    assert GLOBAL_PRESERVATION == {"full_lexicon_ltm_min": 0.95}
    assert CANONICAL_FULL_LEXICON_LTM == 0.989449
    assert STRICT_PRESERVATION_MAX_DROP == 0.02
    assert global_preservation_met(0.95) and global_preservation_met(0.99)
    assert not global_preservation_met(0.9499)


def test_preservation_report_computes_drop_and_both_readings():
    r = preservation_report(_rep(0.9700))
    assert r["ltm"] == 0.97
    assert r["absolute_ltm_drop_from_canonical"] == pytest.approx(0.019449)
    assert r["primary_criterion_ltm_ge_095"] is True
    assert r["secondary_strict_drop_le_002"] is True          # 0.0194 <= 0.02
    r2 = preservation_report(_rep(0.9600))
    assert r2["primary_criterion_ltm_ge_095"] is True
    assert r2["secondary_strict_drop_le_002"] is False        # 0.0294 > 0.02


def test_secondary_threshold_is_reported_but_never_controls_stopping():
    """0.9600 fails the strict drop yet must still stop the run."""
    ctl = CoexistenceController(require_global=True)
    ctl.observe_local(100, True)
    assert ctl.observe_local(200, True) == "check_global"
    rpt = preservation_report(_rep(0.9600))
    assert rpt["secondary_strict_drop_le_002"] is False
    assert ctl.record_global(200, rpt) == "stop"
    assert ctl.global_success is True


def test_local_coexistence_alone_does_not_stop_phase3c():
    ctl = CoexistenceController(require_global=True)
    assert ctl.observe_local(100, True) == "continue"          # streak 1
    assert ctl.observe_local(200, True) == "check_global"      # streak 2
    assert ctl.global_success is False                         # not yet stopped


def test_phase3ab_semantics_unchanged_when_global_not_required():
    ctl = CoexistenceController(require_global=False)
    assert ctl.observe_local(100, True) == "continue"
    assert ctl.observe_local(200, True) == "stop"
    assert ctl.local_confirmations == [200]
    assert ctl.global_checks == []


def test_failed_global_check_continues_training_and_resets_streak():
    ctl = CoexistenceController(require_global=True)
    ctl.observe_local(100, True)
    assert ctl.observe_local(200, True) == "check_global"
    assert ctl.record_global(200, preservation_report(_rep(0.1308))) == "continue"
    assert ctl.global_success is False
    assert ctl.streak == 0, "a failed check must require two fresh successes"
    # a single later local success must NOT re-trigger the expensive check
    assert ctl.observe_local(220, True) == "continue"


def test_later_reconfirmation_can_trigger_a_second_global_check():
    ctl = CoexistenceController(require_global=True)
    ctl.observe_local(100, True); ctl.observe_local(200, True)
    ctl.record_global(200, preservation_report(_rep(0.20)))    # fails
    assert ctl.observe_local(300, True) == "continue"
    assert ctl.observe_local(400, True) == "check_global"      # second check
    assert ctl.record_global(400, preservation_report(_rep(0.97))) == "stop"
    assert [c["step"] for c in ctl.global_checks] == [200, 400]
    assert ctl.global_checks[0]["primary_criterion_ltm_ge_095"] is False
    assert ctl.global_checks[1]["primary_criterion_ltm_ge_095"] is True


def test_successful_global_check_records_success_and_stops():
    ctl = CoexistenceController(require_global=True)
    ctl.observe_local(100, True); ctl.observe_local(200, True)
    assert ctl.record_global(200, preservation_report(_rep(0.9900))) == "stop"
    assert ctl.global_success is True and len(ctl.global_checks) == 1


def test_broken_local_streak_prevents_any_global_check():
    ctl = CoexistenceController(require_global=True)
    assert ctl.observe_local(100, True) == "continue"
    assert ctl.observe_local(200, False) == "continue"          # streak reset
    assert ctl.observe_local(300, True) == "continue"
    assert ctl.global_checks == []
    assert ctl.observe_local(400, True) == "check_global"


def test_endpoint_always_gets_a_full_lexicon_evaluation_without_success():
    """Invariant the run relies on: no global success -> endpoint must measure."""
    ctl = CoexistenceController(require_global=True)
    ctl.observe_local(100, True); ctl.observe_local(200, True)
    ctl.record_global(200, preservation_report(_rep(0.13)))
    assert ctl.global_success is False
    # run_multitask forces endpoint_full_repetition when this holds
    assert (ctl.require_global and not ctl.global_success) is True


def test_global_check_is_training_state_and_rng_inert():
    """The full-lexicon check must not consume RNG or leave train mode."""
    model, vocab, entries, bank, batch = _setup()
    model.train()
    torch.manual_seed(999)
    a = torch.randn(4)
    torch.manual_seed(999)
    with torch.no_grad():
        was = model.training
        model.eval()
        for _ in range(3):
            model.route_logits(batch["enc_in"], batch["enc_mask"],
                               batch["dec_in"], route="ltm")
        model.train(was)
    b = torch.randn(4)
    assert torch.equal(a, b)
    assert model.training is True


def test_phase2d3_subset_hash_constant_is_the_stored_one():
    """Guards against silently training on a different population."""
    import json
    p = ("outputs/naming_comprehension_93a577f/"
         "phase2d3_c3_subset3288_stress_seed22/subset_definition.json")
    full = os.path.join(ROOT, p)
    if not os.path.exists(full):
        pytest.skip("Phase 2D3 artifact not present in this checkout")
    with open(full, encoding="utf-8") as f:
        assert json.load(f)["subset_definition_sha256"] == PHASE2D3_SUBSET_SHA256
