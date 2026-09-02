"""Acceptance tests for the FINAL-3 interleaved 1:2:3 schedule.

FINAL-3P trains one task per optimizer step in deterministically shuffled
six-step macro-cycles holding exactly 1 R, 2 N and 3 C, from scratch, with all
parameters trainable and a shared AdamW state.  The summed FINAL-1 update
remains the default and must stay behaviourally unchanged.

Covered: macro-cycle composition and determinism, cursor advancement (pool
rides R; inactive streams frozen), exposure accounting against the canonical
populations, per-task parameter-touch maps, the repetition-cursor LR
convention (identical to the historical global-step rule under `summed`),
mid-cycle exact resume, resume across the LR boundary, and the schedule
mismatch guards.
"""
from __future__ import annotations

import collections
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.train_joint_scratch import (          # noqa: E402
    FINAL_FULL_MODE, INTERLEAVED_123, LR_STAGE1, LR_STAGE2,
    MACRO_CYCLE_STEPS, RATIO_123, SCHEDULE_SEED_OFFSET, STREAM_SEED_OFFSET,
    STREAM_SEED_STRIDE, SUMMED_SCHEDULE, JointScratchTrainer,
    derive_schedule_seed, derive_stream_seeds, lr_for_step, lr_phase,
)

TINY = dict(device="cpu", max_words=400,
            lexicon_path="data/lexicon_en_glove_covered.tsv",
            dorsal_pool_size=32, batch_size=8, subset_mode=FINAL_FULL_MODE,
            subset_per_band=822, subset_size=32, lr_boundary_steps=6,
            allow_glove_fallback=True, require_subset_hash=False,
            glove_path="tests/_no_such_glove_file.txt")


def make_trainer(regime="j0", seed=22, **over):
    kw = dict(TINY)
    kw.update(over)
    return JointScratchTrainer(regime=regime, seed=seed, **kw)


def params(model):
    """Deduplicated parameter snapshot.

    `state_dict()` is NOT used here: the shared phoneme embedding appears in it
    three times (`phon_embed.weight`, `wm.phon_embed.weight`,
    `ltm.phon_embed.weight`) because the one Embedding object is a submodule of
    both routes.  Those aliases make a `wm.`-prefixed key move whenever the
    SHARED embedding is trained, which would wrongly look like the dorsal route
    being updated.  `named_parameters()` yields each tensor once, under its
    canonical name.
    """
    return {k: v.detach().clone() for k, v in model.named_parameters()}


# ==================================================  macro-cycle structure ==

def test_every_macro_cycle_holds_exactly_one_R_two_N_three_C():
    tr = make_trainer(schedule=INTERLEAVED_123)
    for cycle in range(40):
        labels = [tr.task_for_step(cycle * MACRO_CYCLE_STEPS + p)
                  for p in range(MACRO_CYCLE_STEPS)]
        assert collections.Counter(labels) == {
            "repetition": 1, "naming": 2, "comprehension": 3}


def test_task_order_is_a_pure_function_of_seed_and_cycle():
    a = make_trainer(schedule=INTERLEAVED_123)
    b = make_trainer(schedule=INTERLEAVED_123)
    seq_a = [a.task_for_step(i) for i in range(60)]
    assert seq_a == [b.task_for_step(i) for i in range(60)]
    # recomputable in any order, from the step index alone
    assert [a.task_for_step(i) for i in reversed(range(60))][::-1] == seq_a
    # a different experimental seed gives a different order
    c = make_trainer(seed=23, schedule=INTERLEAVED_123)
    assert [c.task_for_step(i) for i in range(60)] != seq_a


def test_schedule_seed_namespace_is_disjoint_from_the_data_streams():
    for s in (0, 19, 22, 100):
        assert derive_schedule_seed(s) == s * STREAM_SEED_STRIDE + SCHEDULE_SEED_OFFSET
        assert derive_schedule_seed(s) not in set(derive_stream_seeds(s).values())
    assert SCHEDULE_SEED_OFFSET not in set(STREAM_SEED_OFFSET.values())


def test_task_order_draws_nothing_from_the_global_rng():
    tr = make_trainer(schedule=INTERLEAVED_123)
    torch.manual_seed(1234)
    before = torch.get_rng_state()
    _ = [tr.task_for_step(i) for i in range(120)]
    assert torch.equal(torch.get_rng_state(), before)


def test_summed_schedule_reports_a_single_task_label():
    tr = make_trainer()
    assert tr.schedule == SUMMED_SCHEDULE          # default
    assert tr.task_for_step(0) == SUMMED_SCHEDULE
    assert tr.train_step()["task"] == SUMMED_SCHEDULE


# ================================================  cursors and exposures  ===

def test_cursors_advance_only_for_the_scheduled_task_and_pool_rides_R():
    tr = make_trainer(schedule=INTERLEAVED_123)
    expected = {k: 0 for k in tr.cursors}
    for i in range(24):
        task = tr.task_for_step(tr.global_step)
        tr.train_step()
        if task == "repetition":
            expected["repetition"] += 1
            expected["pool"] += 1          # pool rides every R step
        else:
            expected[task] += 1
        assert tr.cursors == expected, f"after step {i} ({task})"
    # over 4 complete cycles: 4 R (+4 pool), 8 N, 12 C
    assert tr.cursors == {"repetition": 4, "pool": 4,
                          "naming": 8, "comprehension": 12}


def test_exposure_accounting_matches_the_canonical_populations():
    """Pin the arithmetic used for every FINAL-1 comparison."""
    per = {"repetition": -(-29_571 // 64), "naming": -(-29_571 // 64),
           "comprehension": -(-27_981 // 64)}
    assert (per["repetition"], per["naming"], per["comprehension"]) == (463, 463, 438)
    r, n, c = RATIO_123
    for cycles, exp_r in ((23_150, 50), (46_300, 100)):
        steps = cycles * MACRO_CYCLE_STEPS
        assert steps == {50: 138_900, 100: 277_800}[exp_r]
        assert cycles * r / per["repetition"] == pytest.approx(exp_r)
        assert cycles * n / per["naming"] == pytest.approx(2 * exp_r)
        assert cycles * c / per["comprehension"] == pytest.approx(
            {50: 158.5616, 100: 317.1233}[exp_r], abs=1e-3)


def test_trainer_exposure_helpers_agree_with_the_cursors():
    tr = make_trainer(schedule=INTERLEAVED_123)
    for _ in range(18):                      # 3 complete cycles
        tr.train_step()
    e = tr.exposures()
    per = {k: tr.streams[k].per_epoch for k in tr.cursors}
    for k in ("repetition", "naming", "comprehension"):
        assert e[k] == pytest.approx(tr.cursors[k] / per[k])
    acc = tr.exposure_accounting()
    assert acc["optimizer_steps_per_cycle"] == 6
    assert acc["batches_per_cycle"] == {"repetition": 1, "pool": 1,
                                        "naming": 2, "comprehension": 3}
    assert acc["ratio_R_N_C"] == [1, 2, 3]


def test_final3p_milestone_accounting():
    """M1 / M2 / M3, the declared FINAL-3P milestones.

    M2 sits exactly on the repetition-cursor LR boundary; M3 adds ten further
    R exposures under LR 1e-4 so the pilot cannot stop precisely at the
    transition and mistake missing consolidation for a negative result.
    """
    r_pass, c_pass = 463, 438
    expected = {                     # R exp -> (cycles, steps, N exp, C exp)
        50:  (23_150, 138_900, 100.0, 158.5616),
        100: (46_300, 277_800, 200.0, 317.1233),
        110: (50_930, 305_580, 220.0, 348.8356),
    }
    for r_exp, (cycles, steps, n_exp, c_exp) in expected.items():
        assert r_exp * r_pass == cycles
        assert cycles * MACRO_CYCLE_STEPS == steps
        r, n, c = RATIO_123
        assert cycles * r / r_pass == pytest.approx(r_exp)
        assert cycles * n / r_pass == pytest.approx(n_exp)
        assert cycles * c / c_pass == pytest.approx(c_exp, abs=1e-3)
    # the extension is 10 R exposures = 27,780 steps
    assert expected[110][1] - expected[100][1] == 27_780


def test_m3_lies_after_the_lr_boundary_and_m2_exactly_on_it():
    boundary = 46_300                       # canonical: 100 R exposures
    assert lr_for_step(50 * 463, boundary) == LR_STAGE1          # M1: stage 1
    assert lr_for_step(100 * 463 - 1, boundary) == LR_STAGE1     # just before M2
    assert lr_for_step(100 * 463, boundary) == LR_STAGE2         # M2: on the boundary
    assert lr_for_step(110 * 463, boundary) == LR_STAGE2         # M3: post-boundary
    # the final stretch is genuinely trained at 1e-4
    assert 110 * 463 - boundary == 4_630     # R batches after the transition


def test_epochs_are_repetition_epochs_in_both_schedules():
    s = make_trainer()
    i = make_trainer(schedule=INTERLEAVED_123)
    per = s.streams["repetition"].per_epoch
    assert s.steps_for_rep_epochs(100) == 100 * per
    assert i.steps_for_rep_epochs(100) == 100 * per * MACRO_CYCLE_STEPS
    # the FINAL-3P budget, on the real populations
    assert 110 * 463 * MACRO_CYCLE_STEPS == 305_580
    assert 100 * 463 * MACRO_CYCLE_STEPS == 277_800
    assert 50 * 463 * MACRO_CYCLE_STEPS == 138_900


def test_final3p_slurm_job_declares_the_pilot_budget():
    """Pin the submitted job against drift: it must stop at M3 and run full
    evaluations at all three milestones, with the recipe untouched."""
    path = os.path.join(ROOT, "scripts", "cluster", "jeanzay",
                        "final3p_run.slurm")
    text = open(path, encoding="utf-8").read()
    assert "EPOCHS=110" in text, "pilot must stop at 110 R exposures (M3)"
    assert "FULL_EVAL_AT=138900,277800,305580" in text
    for frozen in ("SCHEDULE=interleaved_123", "SEED=22",
                   "SUBSET_MODE=final_full", "--endpoint-eval"):
        assert frozen in text, f"{frozen} missing from the pilot job"
    # the frozen recipe must not be overridden on the command line
    for forbidden in ("--c-align-weight", "--lr-boundary-steps",
                      "--batch-size", "--dorsal-pool-size",
                      "--allow-glove-fallback", "--no-subset-hash-check"):
        assert forbidden not in text, f"{forbidden} must not appear in the pilot job"


# =============================================  per-task parameter touch  ===

ENCODER_SIDE = ("ltm.encoder.", "ltm.to_semantic.")
DECODER_SIDE = ("ltm.sem_to_h0.", "ltm.decoder.", "ltm.dec_to_premotor.")
# The dorsal route's OWN parameters.  `phon_embed` is deliberately excluded:
# it is one shared tensor trained by every task (see `params`).
WM_ONLY = ("wm.encoder.", "wm.decoder.", "wm.to_premotor.")


def _step_of_task(tr, task):
    """Run steps until one of `task` has been executed; return before/after."""
    while tr.task_for_step(tr.global_step) != task:
        tr.train_step()
    before = params(tr.model)
    rec = tr.train_step()
    return before, params(tr.model), rec


def test_naming_step_leaves_the_encoder_side_untouched():
    tr = make_trainer(schedule=INTERLEAVED_123)
    before, after, rec = _step_of_task(tr, "naming")
    assert rec["task"] == "naming"
    moved = {k for k in before if not torch.equal(before[k], after[k])}
    assert not any(k.startswith(ENCODER_SIDE) for k in moved), \
        "an N step must not update the comprehension encoder"
    assert any(k.startswith(DECODER_SIDE) for k in moved)
    assert not any(k.startswith(WM_ONLY) for k in moved), \
        "an N step must not update the dorsal route"
    # the SHARED embedding is legitimately trained by naming (decoder input)
    assert "phon_embed.weight" in moved


def test_comprehension_step_leaves_the_decoder_and_wm_untouched():
    tr = make_trainer(schedule=INTERLEAVED_123)
    before, after, rec = _step_of_task(tr, "comprehension")
    assert rec["task"] == "comprehension"
    moved = {k for k in before if not torch.equal(before[k], after[k])}
    assert any(k.startswith(ENCODER_SIDE) for k in moved)
    for prefix in DECODER_SIDE + WM_ONLY + ("motor.",):
        assert not any(k.startswith(prefix) for k in moved), \
            f"a C step must not update {prefix}"
    # the SHARED embedding is legitimately trained by comprehension
    assert "phon_embed.weight" in moved


def test_repetition_step_trains_both_routes_and_the_pool():
    tr = make_trainer(schedule=INTERLEAVED_123)
    before, after, rec = _step_of_task(tr, "repetition")
    assert rec["task"] == "repetition"
    assert "pool_ce" in rec and "rep" in rec and "align" in rec
    moved = {k for k in before if not torch.equal(before[k], after[k])}
    for prefix in WM_ONLY + ("motor.", "phon_embed.") + ENCODER_SIDE + DECODER_SIDE:
        assert any(k.startswith(prefix) for k in moved), f"{prefix} did not train"


def test_all_parameters_remain_trainable_under_interleaving():
    tr = make_trainer(schedule=INTERLEAVED_123)
    assert all(p.requires_grad for p in tr.model.parameters())
    before = params(tr.model)
    for _ in range(MACRO_CYCLE_STEPS * 2):
        tr.train_step()
    moved = {k for k in before if not torch.equal(before[k], params(tr.model)[k])}
    assert len(moved) == len(before), "some parameters never moved in two cycles"


def test_only_the_scheduled_task_loss_is_recorded():
    """Blank/absent means 'not computed on this step', never 'computed as 0'."""
    tr = make_trainer(schedule=INTERLEAVED_123)
    for _ in range(MACRO_CYCLE_STEPS):
        task = tr.task_for_step(tr.global_step)
        rec = tr.train_step()
        if task == "naming":
            assert "naming_ce" in rec
            assert "retrieval_ce" not in rec and "rep" not in rec
        elif task == "comprehension":
            assert "retrieval_ce" in rec and "retrieval_weighted" in rec
            assert "naming_ce" not in rec and "rep" not in rec
            assert "c_align" not in rec            # c_align_weight = 0
        else:
            assert "rep" in rec and "pool_ce" in rec
            assert "naming_ce" not in rec and "retrieval_ce" not in rec
        assert rec["grad_norm"] >= 0.0


# ==========================================================  LR semantics  ==

def test_lr_is_a_function_of_the_repetition_cursor():
    assert lr_for_step(46_299, 46_300) == LR_STAGE1
    assert lr_for_step(46_300, 46_300) == LR_STAGE2
    assert lr_phase(46_299, 46_300).startswith("stage1")
    assert lr_phase(46_300, 46_300).startswith("stage2")


def test_summed_lr_is_identical_to_the_historical_global_step_rule():
    """Under `summed` the repetition cursor equals global_step at every point,
    so the FINAL-1 LR trajectory is unchanged."""
    tr = make_trainer(lr_boundary_steps=4)
    seen = []
    for _ in range(8):
        seen.append(tr.train_step()["lr"])
        assert tr.rep_cursor == tr.global_step
    assert seen == [LR_STAGE1] * 4 + [LR_STAGE2] * 4


def test_interleaved_lr_boundary_falls_on_repetition_progress():
    """With a boundary of 2 R batches, the drop happens after the 2nd R step —
    not after 2 optimizer steps."""
    tr = make_trainer(schedule=INTERLEAVED_123, lr_boundary_steps=2)
    lrs, rcur = [], []
    for _ in range(MACRO_CYCLE_STEPS * 3):
        lrs.append(tr.train_step()["lr"])
        rcur.append(tr.rep_cursor)
    for lr, rc_before in zip(lrs, [0] + rcur[:-1]):
        assert lr == (LR_STAGE1 if rc_before < 2 else LR_STAGE2)
    assert LR_STAGE1 in lrs and LR_STAGE2 in lrs


# ==============================================================  resume  ====

def _drive(tr, n):
    for _ in range(n):
        tr.train_step()


@pytest.mark.parametrize("n_steps", [4, 7, 9])          # all mid-cycle
def test_exact_resume_mid_cycle(tmp_path, n_steps):
    a = make_trainer(schedule=INTERLEAVED_123)
    _drive(a, n_steps)
    assert a.global_step % MACRO_CYCLE_STEPS != 0, "test must resume mid-cycle"
    ck = tmp_path / f"mid{n_steps}.pt"
    torch.save(a.state_dict(), str(ck))
    _drive(a, 7)

    b = make_trainer(schedule=INTERLEAVED_123)
    b.load_state_dict(torch.load(str(ck), weights_only=False), source="t")
    assert b.global_step == n_steps
    _drive(b, 7)

    pa, pb = params(a.model), params(b.model)
    bad = [k for k in pa if not torch.equal(pa[k], pb[k])]
    assert not bad, f"resume diverged on {len(bad)} tensors, e.g. {bad[:3]}"
    assert a.cursors == b.cursors and a.global_step == b.global_step


def test_exact_resume_across_the_repetition_lr_boundary(tmp_path):
    a = make_trainer(schedule=INTERLEAVED_123, lr_boundary_steps=2)
    _drive(a, 5)
    ck = tmp_path / "pre_boundary.pt"
    torch.save(a.state_dict(), str(ck))
    _drive(a, 13)                      # crosses the 2nd R step

    b = make_trainer(schedule=INTERLEAVED_123, lr_boundary_steps=2)
    b.load_state_dict(torch.load(str(ck), weights_only=False), source="t")
    _drive(b, 13)

    pa, pb = params(a.model), params(b.model)
    assert not [k for k in pa if not torch.equal(pa[k], pb[k])]
    assert a.current_lr() == b.current_lr()


def test_resume_refuses_a_schedule_mismatch(tmp_path):
    a = make_trainer(schedule=INTERLEAVED_123)
    a.train_step()
    p = tmp_path / "i123.pt"
    torch.save(a.state_dict(), str(p))
    with pytest.raises(RuntimeError, match="schedule"):
        make_trainer().load_state_dict(
            torch.load(str(p), weights_only=False), source="t")
    # ... and a checkpoint with no schedule key counts as summed
    ck = torch.load(str(p), weights_only=False)
    ck.pop("schedule")
    with pytest.raises(RuntimeError, match="schedule"):
        make_trainer(schedule=INTERLEAVED_123).load_state_dict(ck, source="t")


def test_interleaved_requires_both_semantic_objectives():
    for regime in ("h0", "c_only", "n_only"):
        with pytest.raises(RuntimeError, match="requires a regime"):
            make_trainer(regime=regime, schedule=INTERLEAVED_123)
    with pytest.raises(ValueError):
        make_trainer(schedule="nonsense")


# =========================================================  provenance  =====

def test_settings_and_checkpoint_record_the_schedule():
    tr = make_trainer(schedule=INTERLEAVED_123)
    s = tr.resolved_settings()
    assert s["schedule"] == INTERLEAVED_123
    assert s["schedule_ratio_R_N_C"] == [1, 2, 3]
    assert s["schedule_seed"] == derive_schedule_seed(22)
    assert s["macro_cycle_steps"] == 6
    assert "shuffled macro-cycle" in s["task_order_policy"]
    assert "REPETITION cursor" in s["lr_convention"]
    assert s["exposure_accounting"]["batches_per_pass"]["comprehension"] > 0
    assert s["c_align_weight"] == 0.0
    ck = tr.state_dict()
    for k in ("schedule", "schedule_seed", "schedule_ratio", "exposures"):
        assert k in ck
    base = make_trainer().resolved_settings()
    assert base["schedule"] == SUMMED_SCHEDULE
    assert base["macro_cycle_steps"] == 1
