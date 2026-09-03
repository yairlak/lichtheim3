"""Acceptance tests for FINAL-9P: repetition frequency 1:2:3 -> 2:2:3.

FINAL-8P showed that raising one component inside an already saturated,
always-clipped repetition step is ineffective (LTM +.002 at dec 2.0, with
69/69 logged R steps clipped and a median pre-clip norm of 10.47).  FINAL-9P
therefore buys maintenance by the STEP instead of by the weight: twice as many
genuine repetition updates per macro-cycle, and nothing else.

The design is matched on ACQUISITION OPPORTUNITY, not on global step: the run
stops when naming and comprehension have had exactly the control's exposures
(N 320, C 507.3973), by which point repetition has had 190 rather than 160.
Because the streams are counter-addressed, each task at a given cursor has
seen exactly the same items in the same order as the control at that cursor --
only the interleaving and the number of R updates differ.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from collections import Counter

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.compare_checkpoints import (          # noqa: E402
    flat_optimizer,
)
from scripts.naming_comprehension.train_joint_scratch import (          # noqa: E402
    CANONICAL_LOSS_WEIGHTS, FINAL_FULL_MODE, INTERLEAVED_123, INTERLEAVED_223,
    LR_STAGE2, OPT_POLICY_GROUPED_RN_C, RATIO_123, RATIO_223, SUMMED_SCHEDULE,
    JointScratchTrainer, cycle_steps_for, lr_for_step, macro_cycle_n, main,
)
from scripts.naming_comprehension.train_multitask import (              # noqa: E402
    macro_cycle as historical_macro_cycle,
)

JOB = "scripts/cluster/jeanzay/final9p_ratio223_r130.slurm"

TINY = dict(device="cpu", max_words=400,
            lexicon_path="data/lexicon_en_glove_covered.tsv",
            dorsal_pool_size=32, batch_size=8, subset_mode=FINAL_FULL_MODE,
            subset_per_band=822, subset_size=32, lr_boundary_steps=6,
            allow_glove_fallback=True, require_subset_hash=False,
            glove_path="tests/_no_such_glove_file.txt")

SMOKE = ["--regime", "j0", "--seed", "22", "--subset-mode", "final_full",
         "--device", "cpu", "--max-words", "400", "--batch-size", "8",
         "--dorsal-pool-size", "32", "--lr-boundary-steps", "6",
         "--eval-every", "0", "--log-every", "0",
         "--glove-path", "tests/_no_such_glove_file.txt",
         "--allow-glove-fallback", "--no-subset-hash-check"]


def make(regime="j0", seed=22, **over):
    kw = dict(TINY)
    kw.update(over)
    return JointScratchTrainer(regime=regime, seed=seed, **kw)


def launch(out, run_id, steps, *, resume=None, schedule=INTERLEAVED_123,
           grouped=False, phase=False, save_every=None):
    argv = SMOKE + ["--out-dir", out, "--run-id", run_id,
                    "--schedule", schedule, "--max-steps", str(steps),
                    "--save-every", str(save_every or steps)]
    if grouped:
        argv += ["--optimizer-policy", "grouped_rn_c_adamw"]
    if phase:
        argv += ["--phase-transition"]
    if resume:
        argv += ["--resume", resume]
    assert main(argv) == 0


def ckpt(out, run_id, step):
    return os.path.join(out, run_id, "checkpoints", f"step_{step:08d}.pt")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def script():
    return open(os.path.join(ROOT, JOB), encoding="utf-8").read()


# ================================  1-3. schedules: old intact, new correct ==

def test_historical_123_cycle_is_bit_identical():
    """The generalised cycle must reproduce the historical function exactly,
    or every earlier interleaved run changes."""
    for seed in (22_000_070, 0, 999):
        for c in range(300):
            assert macro_cycle_n(RATIO_123, seed, c) == \
                historical_macro_cycle(RATIO_123, seed, c)
    assert cycle_steps_for(INTERLEAVED_123) == 6
    assert cycle_steps_for(SUMMED_SCHEDULE) == 1


def test_223_is_deterministic_and_has_the_right_composition():
    assert RATIO_223 == (2, 2, 3)
    assert cycle_steps_for(INTERLEAVED_223) == 7
    for c in range(200):
        cyc = macro_cycle_n(RATIO_223, 22_000_070, c)
        assert len(cyc) == 7
        assert Counter(cyc) == {"repetition": 2, "naming": 2,
                                "comprehension": 3}
        assert cyc == macro_cycle_n(RATIO_223, 22_000_070, c), "not pure"
    # a different seed gives a different order
    assert (macro_cycle_n(RATIO_223, 22_000_070, 0)
            != macro_cycle_n(RATIO_223, 23_000_070, 0))


def test_task_order_draws_nothing_from_the_global_rng():
    tr = make(schedule=INTERLEAVED_223)
    torch.manual_seed(1234)
    before = torch.get_rng_state()
    _ = [tr.task_for_step(i) for i in range(140)]
    assert torch.equal(torch.get_rng_state(), before)


def test_223_cursors_advance_two_r_two_n_three_c_per_cycle():
    tr = make(schedule=INTERLEAVED_223)
    for _ in range(7 * 4):                       # four complete cycles
        tr.train_step()
    assert tr.cursors == {"repetition": 8, "pool": 8, "naming": 8,
                          "comprehension": 12}


# =========================================  4, 13-14. the transition  ======

@pytest.fixture
def branched(tmp_path):
    """A grouped 1:2:3 run (the FINAL-7L analogue) branched into 2:2:3."""
    out = str(tmp_path / "runs")
    launch(out, "seven_l", 6, save_every=6)                            # shared
    launch(out, "seven_l", 18, resume=ckpt(out, "seven_l", 6), grouped=True,
           phase=True, save_every=6)                                   # RN|C
    src = ckpt(out, "seven_l", 18)
    launch(out, "nine_p", 32, resume=src, schedule=INTERLEAVED_223,
           grouped=True, phase=True, save_every=32)                    # 2:2:3
    return out, src


def test_schedule_transition_is_declared_once_and_anchored(branched):
    out, _ = branched
    end = torch.load(ckpt(out, "nine_p", 32), map_location="cpu",
                     weights_only=False)
    assert end["schedule"] == INTERLEAVED_223
    assert end["schedule_ratio"] == [2, 2, 3]
    assert end["schedule_anchor_step"] == 18, \
        "the new cycle count must start at the transition step"
    rec = end["phase_transitions"][-1]
    assert rec["changed"] == ["schedule"], rec["changed"]
    assert rec["old_schedule"] == INTERLEAVED_123
    assert rec["new_schedule"] == INTERLEAVED_223
    assert rec["old_schedule_ratio"] == [1, 2, 3]
    assert rec["new_schedule_ratio"] == [2, 2, 3]
    assert rec["schedule_anchor_step"] == 18
    assert rec["old_lr_policy"] == rec["new_lr_policy"]
    assert rec["old_optimizer_policy"] == rec["new_optimizer_policy"]
    assert rec["old_dec_weight"] == rec["new_dec_weight"] == 0.5
    assert rec["moment_initialization"] == "unchanged"


def test_silent_ratio_change_is_refused(branched):
    out, src = branched
    ck = torch.load(src, map_location="cpu", weights_only=False)
    with pytest.raises(RuntimeError, match="PHASE TRANSITION"):
        make(schedule=INTERLEAVED_223,
             optimizer_policy=OPT_POLICY_GROUPED_RN_C).load_state_dict(
                 copy.deepcopy(ck), source="t")
    # summed <-> interleaved is never a transition, declared or not
    for allow in (False, True):
        with pytest.raises(RuntimeError, match="never be spliced"):
            make(allow_phase_transition=allow).load_state_dict(
                copy.deepcopy(ck), source="t")


def test_requeue_keeps_the_anchor_and_declares_nothing(branched):
    out, _ = branched
    n_before = len(torch.load(ckpt(out, "nine_p", 32), map_location="cpu",
                              weights_only=False)["phase_transitions"])
    launch(out, "nine_p", 46, resume=ckpt(out, "nine_p", 32),
           schedule=INTERLEAVED_223, grouped=True, save_every=46)
    end = torch.load(ckpt(out, "nine_p", 46), map_location="cpu",
                     weights_only=False)
    assert end["schedule_anchor_step"] == 18, "the anchor moved on requeue"
    assert len(end["phase_transitions"]) == n_before


def test_interrupted_223_run_equals_uninterrupted(branched, tmp_path):
    """The anchor makes the post-transition task order exactly resumable."""
    out, src = branched
    launch(out, "whole", 39, resume=src, schedule=INTERLEAVED_223,
           grouped=True, phase=True, save_every=39)
    launch(out, "split", 25, resume=src, schedule=INTERLEAVED_223,
           grouped=True, phase=True, save_every=25)
    launch(out, "split", 39, resume=ckpt(out, "split", 25),
           schedule=INTERLEAVED_223, grouped=True, save_every=39)
    a = torch.load(ckpt(out, "whole", 39), map_location="cpu", weights_only=False)
    b = torch.load(ckpt(out, "split", 39), map_location="cpu", weights_only=False)
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    assert not [k for k in sa if not torch.equal(sa[k], sb[k])]
    assert a["cursors"] == b["cursors"]
    assert a["schedule_anchor_step"] == b["schedule_anchor_step"] == 18


# ============================  5-10. everything else must stay frozen  =====

def test_only_the_ratio_differs_in_the_effective_settings():
    a = make(schedule=INTERLEAVED_123,
             optimizer_policy=OPT_POLICY_GROUPED_RN_C).resolved_settings()
    b = make(schedule=INTERLEAVED_223,
             optimizer_policy=OPT_POLICY_GROUPED_RN_C).resolved_settings()
    diff = {k for k in a if a[k] != b[k]}
    assert diff == {"schedule", "schedule_ratio_R_N_C", "macro_cycle_steps",
                    "exposure_accounting", "task_order_policy"} - \
        {k for k in ("task_order_policy",) if a.get(k) == b.get(k)}, diff
    for k in ("lambda_C", "lambda_N", "tau", "grad_clip", "weight_decay",
              "dec_weight", "loss_weights", "c_align_weight", "lr_policy",
              "optimizer_policy", "optimizer_bank_layout", "seed",
              "stream_seeds", "comprehension_population_sha256",
              "naming_population_sha256"):
        assert a[k] == b[k], k
    assert b["dec_weight"] == CANONICAL_LOSS_WEIGHTS["dec"] == 0.5


def test_lr_and_dec_are_unchanged_by_the_intervention(branched):
    out, _ = branched
    end = torch.load(ckpt(out, "nine_p", 32), map_location="cpu",
                     weights_only=False)
    assert end["dec_weight"] == 0.5
    assert end["lr_policy"]["kind"] == "two_stage_rep_cursor"
    assert end["optimizer_policy"] == OPT_POLICY_GROUPED_RN_C
    # at the real source the boundary is long past, so the LR is 1e-4
    assert lr_for_step(130 * 463, 46_300) == LR_STAGE2


def test_optimizer_banks_survive_the_schedule_transition(branched):
    out, src = branched
    before = flat_optimizer(torch.load(src, map_location="cpu",
                                       weights_only=False))
    fresh = make(schedule=INTERLEAVED_223,
                 optimizer_policy=OPT_POLICY_GROUPED_RN_C,
                 allow_phase_transition=True)
    fresh.load_state_dict(torch.load(src, map_location="cpu",
                                     weights_only=False), source="t")
    got = {n: {f"{pid}.{k}": v for pid, e in o.state_dict()["state"].items()
               for k, v in e.items()} for n, o in fresh.banks.items()}
    assert set(got) == set(before)
    for name in before:
        for k, v in before[name].items():
            ok = (torch.equal(v, got[name][k]) if torch.is_tensor(v)
                  else v == got[name][k])
            assert ok, f"bank {name} entry {k} changed"
    ptrs = [next(iter(o.state.values()))["exp_avg"].data_ptr()
            for o in fresh.banks.values()]
    assert len(set(ptrs)) == 2, "banks were cloned or aliased"


def test_task_streams_see_identical_items_at_identical_cursors():
    """The matched design rests on this: a stream's k-th batch is a pure
    function of (stream seed, k), so at equal cursors the 2:2:3 run has seen
    exactly the control's items in the control's order."""
    a = make(schedule=INTERLEAVED_123)
    b = make(schedule=INTERLEAVED_223)
    for stream in ("naming", "comprehension", "repetition"):
        for cursor in (0, 1, 17, 128):
            assert a.streams[stream].indices(cursor) == \
                b.streams[stream].indices(cursor), (stream, cursor)


def test_source_run_is_untouched_by_the_branch(branched):
    out, src = branched
    sdir = os.path.join(out, "seven_l")
    fp = {n: sha(os.path.join(sdir, "checkpoints", n))
          for n in os.listdir(os.path.join(sdir, "checkpoints"))}
    fp.update({f: sha(os.path.join(sdir, f)) for f in os.listdir(sdir)
               if os.path.isfile(os.path.join(sdir, f))})
    launch(out, "nine_p2", 32, resume=src, schedule=INTERLEAVED_223,
           grouped=True, phase=True, save_every=32)
    now = {n: sha(os.path.join(sdir, "checkpoints", n))
           for n in os.listdir(os.path.join(sdir, "checkpoints"))}
    now.update({f: sha(os.path.join(sdir, f)) for f in os.listdir(sdir)
                if os.path.isfile(os.path.join(sdir, f))})
    assert now == fp, "the FINAL-7L source run was modified"


# ================================  11-12. matched N/C accounting  ==========

def test_matched_exposure_accounting_is_exact():
    """FINAL-9P stops at the control's N and C exposures, not its step count."""
    R, C = 463, 438
    src_cycles = 130 * R                       # 1:2:3 cycles at R130
    src = {"step": src_cycles * 6, "R": src_cycles, "N": 2 * src_cycles,
           "C": 3 * src_cycles}
    assert src["step"] == 361_140
    assert (src["R"] / R, src["N"] / R) == (130.0, 260.0)
    assert src["C"] / C == pytest.approx(412.2603, abs=1e-4)

    expected = {                # extra N exposures -> (step, R, N, C)
        20: (393_550, 150.0, 280.0, 443.9726),
        40: (425_960, 170.0, 300.0, 475.6849),
        60: (458_370, 190.0, 320.0, 507.3973),
    }
    for dN, (step, r_exp, n_exp, c_exp) in expected.items():
        cycles = (dN * R) // 2                 # 2 N batches per 7-step cycle
        assert (dN * R) % 2 == 0, "N batches must fall on whole cycles"
        assert src["step"] + cycles * 7 == step
        assert (src["R"] + 2 * cycles) / R == pytest.approx(r_exp)
        assert (src["N"] + 2 * cycles) / R == pytest.approx(n_exp)
        assert (src["C"] + 3 * cycles) / C == pytest.approx(c_exp, abs=1e-4)
    assert expected[60][0] - src["step"] == 97_230

    # the matched CONTROL states, reached under 1:2:3 at the same N and C
    for dN, ctrl_step, ctrl_R in ((20, 388_920, 140), (40, 416_700, 150),
                                  (60, 444_480, 160)):
        cyc = ctrl_step // 6
        assert cyc * 2 / R == pytest.approx(260.0 + dN)      # same N
        assert cyc * 3 / C == pytest.approx(expected[dN][3], abs=1e-4)  # same C
        assert cyc / R == pytest.approx(ctrl_R)              # fewer R
        assert ctrl_R < expected[dN][1], "the intervention must add R exposure"


def test_epochs_budgeting_refuses_a_partial_cycle():
    """463 R batches at 2 per 7-step cycle is not a whole number of cycles,
    so a 2:2:3 run must be budgeted in steps."""
    # The real population gives 463 R batches per epoch (odd), so at two per
    # seven-step cycle an ODD epoch count leaves half a cycle.  The tiny
    # fixture needs an odd batches-per-epoch to reproduce that: 400 words at
    # batch 9 is 45 per epoch.
    assert 463 % 2 == 1, "the real per-epoch count is odd"
    tr = make(schedule=INTERLEAVED_223, batch_size=9)
    assert tr.streams["repetition"].per_epoch % 2 == 1
    with pytest.raises(RuntimeError, match="--max-steps"):
        tr.steps_for_rep_epochs(161)
    # when it does divide it must be exact, never rounded
    assert tr.steps_for_rep_epochs(160) == 160 * 45 * 7 // 2
    # the 1:2:3 path is untouched
    assert make(schedule=INTERLEAVED_123).steps_for_rep_epochs(160) == 160 * 50 * 6
    assert 160 * 463 * 6 == 444_480


# ==================================================  15-17. job contract  ==

def test_final9p_job_contract():
    t = script()
    assert "SCHEDULE=interleaved_223" in t
    assert "--max-steps" in t and "MAX_STEPS=458370" in t
    assert "FULL_EVAL_AT=393550,425960,458370" in t
    assert "SOURCE_STEP=361140" in t
    assert 'RUN_ID="final9p_r223_r130_seed${SEED}_${SUBSET_MODE}"' in t
    assert 'PARENT_RUN_ID="final7l_fromscratch_seed${SEED}_${SUBSET_MODE}"' in t
    assert '[[ "$RUN_ID" != "$PARENT_RUN_ID" ]]' in t
    assert "FATAL: source checkpoint" in t
    assert "--phase-transition" in t
    assert "OPT_POLICY=grouped_rn_c_adamw" in t
    # dec must NOT be carried over from FINAL-8P, and no other knob appears
    for forbidden in ("--dec-weight", "--lr-repetition", "--lr-naming",
                      "--lr-comprehension", "--lr-boundary-steps",
                      "--c-align-weight", "--batch-size",
                      "--dorsal-pool-size", "--allow-glove-fallback",
                      "--no-subset-hash-check", "--epochs"):
        assert forbidden not in t, forbidden
    # the matched design is stated where it will be read
    assert "matched" in t.lower() and "507.3973" in t


def test_cadences_sit_at_a_constant_macro_cycle_phase():
    """Both cadences fire on ABSOLUTE step multiples.  Making them multiples
    of the seven-step cycle keeps every dev evaluation and safety checkpoint
    at one fixed phase, so the dev curve is not read at drifting R/N/C
    positions.  Landing them ON the milestones is arithmetically impossible
    -- gcd(361140, 32410) is 4630, which is not a multiple of seven -- and is
    also unnecessary, because --full-eval-at checkpoints unconditionally."""
    t = script()
    assert "EVAL_EVERY=16205" in t and "SAVE_EVERY=32410" in t
    cycle = cycle_steps_for(INTERLEAVED_223)
    anchor, end = 361_140, 458_370
    for cadence in (16_205, 32_410):
        assert cadence % cycle == 0, cadence
        hits = [s for s in range(cadence, end + 1, cadence) if s > anchor]
        assert hits, cadence
        phases = {(s - anchor) % cycle for s in hits}
        assert len(phases) == 1, (cadence, phases)
    assert [s for s in range(16_205, end + 1, 16_205) if s > anchor] == \
        [372_715, 388_920, 405_125, 421_330, 437_535, 453_740]
    assert [s for s in range(32_410, end + 1, 32_410) if s > anchor] == \
        [388_920, 421_330, 453_740]
    # one safety checkpoint shortly before each milestone
    for save, milestone in zip((388_920, 421_330, 453_740),
                               (393_550, 425_960, 458_370)):
        assert 0 < milestone - save <= 32_410


def test_milestones_are_checkpointed_even_off_the_save_cadence(tmp_path):
    """The milestones are not multiples of SAVE_EVERY, so the run depends on
    --full-eval-at saving a checkpoint on its own.  Prove it does."""
    out = str(tmp_path / "runs")
    argv = SMOKE + ["--out-dir", out, "--run-id", "m",
                    "--schedule", INTERLEAVED_223, "--max-steps", "21",
                    "--full-eval-at", "14", "--save-every", "21"]
    assert main(argv) == 0
    assert 14 % 21 != 0, "14 must not be on the save cadence"
    assert os.path.exists(ckpt(out, "m", 14)), \
        "a milestone must checkpoint itself"
    assert os.path.exists(ckpt(out, "m", 21))


def test_earlier_family_scripts_are_untouched():
    for name, must in (
            ("final7l_fromscratch_gate_r220.slurm", "GATE_EPOCHS=220"),
            ("final7l_continue_r220_to_r1000.slurm", "EPOCHS=1000"),
            ("final8p_dec_weight_r130_to_r160.slurm", "DEC_WEIGHT=2.0")):
        t = open(os.path.join(ROOT, "scripts/cluster/jeanzay", name),
                 encoding="utf-8").read()
        assert must in t
        assert "interleaved_223" not in t, f"{name} must keep its schedule"
