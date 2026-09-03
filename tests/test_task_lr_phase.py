"""Acceptance tests for FINAL-4: task-specific LR + explicit phase transition.

FINAL-4 changes ONLY the learning-rate policy.  From the R150 checkpoint
onward each task's optimizer step uses its own fixed rate
(eta_R=1e-4, eta_N=3e-4, eta_C=3e-4) on the SAME shared AdamW with its moments
carried over; lambda_C, the 1:2:3 ratio, the objectives, the populations and
the architecture are untouched.

Because that is a change of recipe rather than a continuation, resuming with a
different LR policy is refused unless a phase transition is declared, and the
declaration is recorded in the checkpoint and provenance.  Runs that never
mention the new flags keep the historical two-stage schedule bit-for-bit.
"""
from __future__ import annotations

import json
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.train_joint_scratch import (          # noqa: E402
    FINAL_FULL_MODE, INTERLEAVED_123, LR_POLICY_TASK, LR_POLICY_TWO_STAGE,
    LR_STAGE1, LR_STAGE2, MACRO_CYCLE_STEPS, SUMMED_SCHEDULE,
    JointScratchTrainer, build_parser, main, task_lr_policy,
    task_lrs_from_args, two_stage_lr_policy,
)

FINAL4_LRS = {"repetition": 1e-4, "naming": 3e-4, "comprehension": 3e-4}

TINY = dict(device="cpu", max_words=400,
            lexicon_path="data/lexicon_en_glove_covered.tsv",
            dorsal_pool_size=32, batch_size=8, subset_mode=FINAL_FULL_MODE,
            subset_per_band=822, subset_size=32, lr_boundary_steps=6,
            allow_glove_fallback=True, require_subset_hash=False,
            glove_path="tests/_no_such_glove_file.txt")


def make(regime="j0", seed=22, **over):
    kw = dict(TINY)
    kw.update(over)
    return JointScratchTrainer(regime=regime, seed=seed, **kw)


def params(model):
    # named_parameters(): the shared embedding is aliased in state_dict()
    return {k: v.detach().clone() for k, v in model.named_parameters()}


# =============================================  1-2. dispatch and sharing  ==

def test_each_task_step_uses_its_own_configured_lr():
    tr = make(schedule=INTERLEAVED_123, task_lrs=FINAL4_LRS)
    seen = {}
    for _ in range(MACRO_CYCLE_STEPS * 3):
        task = tr.task_for_step(tr.global_step)
        rec = tr.train_step()
        assert rec["task"] == task
        assert rec["lr"] == pytest.approx(FINAL4_LRS[task]), \
            f"{task} step ran at {rec['lr']}, expected {FINAL4_LRS[task]}"
        assert rec["lr_phase"] == f"task_specific_{task}"
        seen[task] = rec["lr"]
    assert set(seen) == set(FINAL4_LRS), "not every task was exercised"


def test_one_shared_adamw_instance_and_single_param_group():
    tr = make(schedule=INTERLEAVED_123, task_lrs=FINAL4_LRS)
    optim_id = id(tr.optim)
    assert len(tr.optim.param_groups) == 1
    for _ in range(MACRO_CYCLE_STEPS):
        tr.train_step()
        assert id(tr.optim) == optim_id, "the optimizer was replaced"
        assert len(tr.optim.param_groups) == 1
    # every parameter that has been touched shares that one optimizer's state
    assert tr.optim.state, "no optimizer state accumulated"


def test_lr_scales_the_applied_update_not_the_moments():
    """The audit's key mechanism: a task LR scales the applied delta only."""
    base = make(schedule=INTERLEAVED_123, task_lrs=FINAL4_LRS)
    task = base.task_for_step(0)
    hi = dict(FINAL4_LRS)
    hi[task] = FINAL4_LRS[task] * 3

    def one_step(lrs):
        tr = make(schedule=INTERLEAVED_123, task_lrs=lrs)
        before = params(tr.model)
        tr.train_step()
        after = params(tr.model)
        delta = torch.cat([(after[k] - before[k]).reshape(-1) for k in before])
        st = tr.optim.state_dict()["state"]
        return float(delta.norm()), st

    n1, s1 = one_step(FINAL4_LRS)
    n3, s3 = one_step(hi)
    assert n3 / n1 == pytest.approx(3.0, rel=1e-4), "delta not linear in lr"
    for pid in s1:
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(s1[pid][key], s3[pid][key]), \
                f"{key} changed with lr — moments must be lr-independent"


# ==========================================  3-4. moments and grad=None  ====

def test_moments_are_preserved_across_a_phase_transition(tmp_path):
    a = make(schedule=INTERLEAVED_123)               # two-stage policy
    for _ in range(MACRO_CYCLE_STEPS * 2):
        a.train_step()
    p = tmp_path / "pre.pt"
    torch.save(a.state_dict(), str(p))
    before = a.optim.state_dict()["state"]

    b = make(schedule=INTERLEAVED_123, task_lrs=FINAL4_LRS,
             allow_phase_transition=True)
    b.load_state_dict(torch.load(str(p), weights_only=False), source="t")
    after = b.optim.state_dict()["state"]

    assert set(before) == set(after) and before
    for pid in before:
        for key, va in before[pid].items():
            vb = after[pid][key]
            if torch.is_tensor(va):
                assert torch.equal(va, vb), f"moment {key} not preserved"
            else:
                assert va == vb
    pa, pb = params(a.model), params(b.model)
    assert not [k for k in pa if not torch.equal(pa[k], pb[k])]
    assert b.global_step == a.global_step and b.cursors == a.cursors


def test_grad_none_parameters_are_untouched_under_task_lr():
    tr = make(schedule=INTERLEAVED_123, task_lrs=FINAL4_LRS)
    while tr.task_for_step(tr.global_step) != "comprehension":
        tr.train_step()
    before = params(tr.model)
    n_state_before = len(tr.optim.state)
    tr.train_step()
    after = params(tr.model)
    moved = {k for k in before if not torch.equal(before[k], after[k])}
    # a C step reaches the encoder side only: no decoder/WM update, and no
    # weight decay leaks onto parameters without a gradient
    for prefix in ("ltm.sem_to_h0.", "ltm.decoder.", "ltm.dec_to_premotor.",
                   "wm.encoder.", "wm.decoder.", "motor."):
        assert not any(k.startswith(prefix) for k in moved), prefix
    assert len(tr.optim.state) >= n_state_before


# ==================================================  5. backward compat  ====

def test_absent_flags_keep_the_historical_two_stage_policy():
    for sched in (SUMMED_SCHEDULE, INTERLEAVED_123):
        tr = make(schedule=sched)
        assert tr.lr_policy == two_stage_lr_policy(tr.lr_boundary_steps)
        assert tr.lr_policy["kind"] == LR_POLICY_TWO_STAGE
    args = build_parser().parse_args(["--regime", "j0"])
    assert task_lrs_from_args(args) is None
    assert args.phase_transition is False


def test_two_stage_trajectory_is_unchanged_by_the_new_code_path():
    """A run that never mentions task LRs must be bit-identical to one built
    with the policy constructed explicitly, and must still switch at the
    repetition-cursor boundary."""
    a = make(schedule=INTERLEAVED_123, lr_boundary_steps=4)
    lrs = [a.train_step()["lr"] for _ in range(10)]
    rep_cursors = []
    b = make(schedule=INTERLEAVED_123, lr_boundary_steps=4)
    for _ in range(10):
        rep_cursors.append(b.rep_cursor)
        b.train_step()
    assert lrs == [(LR_STAGE1 if rc < 4 else LR_STAGE2) for rc in rep_cursors]
    pa, pb = params(a.model), params(b.model)
    assert not [k for k in pa if not torch.equal(pa[k], pb[k])]


def test_task_lr_requires_interleaved_and_all_three_rates():
    with pytest.raises(RuntimeError, match="interleaved"):
        make(schedule=SUMMED_SCHEDULE, task_lrs=FINAL4_LRS)
    with pytest.raises(ValueError, match="missing"):
        make(schedule=INTERLEAVED_123,
             task_lrs={"repetition": 1e-4, "naming": 3e-4})
    with pytest.raises(ValueError):
        make(schedule=INTERLEAVED_123,
             task_lrs=dict(FINAL4_LRS, comprehension=0.0))
    args = build_parser().parse_args(["--regime", "j0", "--lr-naming", "3e-4"])
    with pytest.raises(SystemExit, match="all-or-nothing"):
        task_lrs_from_args(args)


# ==============================================  6-8. phase transition  =====

def test_silent_lr_policy_change_is_refused(tmp_path):
    a = make(schedule=INTERLEAVED_123)
    a.train_step()
    p = tmp_path / "two_stage.pt"
    torch.save(a.state_dict(), str(p))
    ck = torch.load(str(p), weights_only=False)
    with pytest.raises(RuntimeError, match="PHASE TRANSITION"):
        make(schedule=INTERLEAVED_123, task_lrs=FINAL4_LRS).load_state_dict(
            ck, source="t")
    # ... and the reverse direction is equally refused
    b = make(schedule=INTERLEAVED_123, task_lrs=FINAL4_LRS,
             allow_phase_transition=True)
    b.load_state_dict(ck, source="t")
    q = tmp_path / "task.pt"
    torch.save(b.state_dict(), str(q))
    with pytest.raises(RuntimeError, match="PHASE TRANSITION"):
        make(schedule=INTERLEAVED_123).load_state_dict(
            torch.load(str(q), weights_only=False), source="t")


def test_declared_transition_is_accepted_and_recorded(tmp_path):
    a = make(schedule=INTERLEAVED_123)
    for _ in range(7):
        a.train_step()
    p = tmp_path / "src.pt"
    torch.save(a.state_dict(), str(p))

    b = make(schedule=INTERLEAVED_123, task_lrs=FINAL4_LRS,
             allow_phase_transition=True)
    b.load_state_dict(torch.load(str(p), weights_only=False), source="src.pt")
    assert len(b.phase_transitions) == 1
    rec = b.phase_transitions[0]
    assert rec["transition_step"] == 7
    assert rec["old_lr_policy"]["kind"] == LR_POLICY_TWO_STAGE
    assert rec["new_lr_policy"] == task_lr_policy(**FINAL4_LRS)
    assert rec["source_checkpoint"] == "src.pt"
    assert "source_commit" in rec and "new_commit" in rec and rec["declared_at"]
    assert rec["exposures_at_transition"]["repetition"] >= 0
    ck = b.state_dict()
    assert ck["lr_policy"]["kind"] == LR_POLICY_TASK
    assert len(ck["phase_transitions"]) == 1


def test_post_transition_checkpoint_resumes_without_a_second_declaration(tmp_path):
    a = make(schedule=INTERLEAVED_123)
    a.train_step()
    src = tmp_path / "src.pt"
    torch.save(a.state_dict(), str(src))

    b = make(schedule=INTERLEAVED_123, task_lrs=FINAL4_LRS,
             allow_phase_transition=True)
    b.load_state_dict(torch.load(str(src), weights_only=False), source="s")
    for _ in range(5):
        b.train_step()
    mid = tmp_path / "mid.pt"
    torch.save(b.state_dict(), str(mid))

    # requeue: SAME policy, NO flag -> accepted, and no new transition logged
    c = make(schedule=INTERLEAVED_123, task_lrs=FINAL4_LRS)
    c.load_state_dict(torch.load(str(mid), weights_only=False), source="m")
    assert len(c.phase_transitions) == 1, "history kept, nothing re-declared"
    assert c.global_step == b.global_step


def test_phase_transition_flag_relaxes_nothing_else(tmp_path):
    a = make(schedule=INTERLEAVED_123)
    a.train_step()
    p = tmp_path / "src.pt"
    torch.save(a.state_dict(), str(p))
    ck = torch.load(str(p), weights_only=False)
    for kw, match in ((dict(seed=23), "seed"),
                      (dict(c_align_weight=1.0), "c_align_weight"),
                      (dict(schedule=SUMMED_SCHEDULE), "schedule")):
        over = dict(kw)
        if over.get("schedule") == SUMMED_SCHEDULE:
            tr = make(allow_phase_transition=True, **over)
        else:
            tr = make(schedule=INTERLEAVED_123, task_lrs=FINAL4_LRS,
                      allow_phase_transition=True, **over)
        with pytest.raises(RuntimeError, match=match):
            tr.load_state_dict(ck, source="t")


# ============================================  9. exact mid-phase resume  ===

def test_exact_mid_phase_resume_is_bitwise(tmp_path):
    a = make(schedule=INTERLEAVED_123, task_lrs=FINAL4_LRS)
    for _ in range(9):                       # mid-cycle
        a.train_step()
    p = tmp_path / "mid.pt"
    torch.save(a.state_dict(), str(p))
    for _ in range(7):
        a.train_step()

    b = make(schedule=INTERLEAVED_123, task_lrs=FINAL4_LRS)
    b.load_state_dict(torch.load(str(p), weights_only=False), source="t")
    for _ in range(7):
        b.train_step()

    pa, pb = params(a.model), params(b.model)
    assert not [k for k in pa if not torch.equal(pa[k], pb[k])]
    assert a.cursors == b.cursors and a.global_step == b.global_step


# ======================================  10-12. budget and frozen recipe  ===

def test_m6_m7_m8_accounting():
    r_pass, c_pass = 463, 438
    expected = {                    # R exp -> (cycles, step, N exp, C exp)
        150: (69_450, 416_700, 300.0, 475.6849),     # source (R150)
        160: (74_080, 444_480, 320.0, 507.3973),     # M6
        170: (78_710, 472_260, 340.0, 539.1096),     # M7
        180: (83_340, 500_040, 360.0, 570.8219),     # M8
    }
    for r_exp, (cycles, step, n_exp, c_exp) in expected.items():
        assert r_exp * r_pass == cycles
        assert cycles * MACRO_CYCLE_STEPS == step
        assert cycles * 2 / r_pass == pytest.approx(n_exp)
        assert cycles * 3 / c_pass == pytest.approx(c_exp, abs=1e-3)
    assert expected[180][1] - expected[150][1] == 83_340


def test_final4_slurm_job_declares_the_experiment():
    path = os.path.join(ROOT, "scripts", "cluster", "jeanzay",
                        "final4_semantic_lr_r180.slurm")
    text = open(path, encoding="utf-8").read()
    assert "EPOCHS=180" in text
    assert "FULL_EVAL_AT=444480,472260,500040" in text
    assert "--lr-repetition" in text and "--lr-naming" in text \
        and "--lr-comprehension" in text
    assert "ETA_R=1e-4" in text and "ETA_N=3e-4" in text and "ETA_C=3e-4" in text
    assert "--phase-transition" in text and "SOURCE_STEP=416700" in text
    # the frozen recipe must not be overridden anywhere on the command line
    for forbidden in ("--c-align-weight", "--lr-boundary-steps",
                      "--batch-size", "--dorsal-pool-size",
                      "--allow-glove-fallback", "--no-subset-hash-check"):
        assert forbidden not in text, forbidden


def test_frozen_scientific_fields_are_reported_unchanged():
    tr = make(schedule=INTERLEAVED_123, task_lrs=FINAL4_LRS)
    s = tr.resolved_settings()
    assert s["lr_policy"] == task_lr_policy(**FINAL4_LRS)
    assert s["c_align_weight"] == 0.0
    assert s["lambda_C"] == 0.087 and s["lambda_N"] == 1.0
    assert s["schedule_ratio_R_N_C"] == [1, 2, 3]
    assert s["tau"] == 0.10
    assert s["grad_clip"] == 1.0
    assert s["schedule"] == INTERLEAVED_123


def test_end_to_end_launch_records_the_phase_transition(tmp_path):
    out = str(tmp_path / "runs")
    common = ["--regime", "j0", "--seed", "22", "--subset-mode", "final_full",
              "--schedule", "interleaved_123", "--device", "cpu",
              "--max-words", "400", "--batch-size", "8",
              "--dorsal-pool-size", "32", "--lr-boundary-steps", "6",
              "--eval-every", "0", "--log-every", "1",
              "--glove-path", "tests/_no_such_glove_file.txt",
              "--allow-glove-fallback", "--no-subset-hash-check",
              "--out-dir", out, "--run-id", "p"]
    assert main(common + ["--max-steps", "6", "--save-every", "6"]) == 0
    assert main(common + ["--max-steps", "12", "--save-every", "12",
                          "--lr-repetition", "1e-4", "--lr-naming", "3e-4",
                          "--lr-comprehension", "3e-4", "--phase-transition",
                          "--resume", os.path.join(out, "p", "checkpoints",
                                                   "step_00000006.pt")]) == 0
    prov = json.load(open(os.path.join(
        out, "p", "provenance_from_step_00000006.json")))
    assert prov["phase_transition_declared"] is True
    assert prov["lr_policy"]["kind"] == LR_POLICY_TASK
    assert prov["phase_transitions"][0]["transition_step"] == 6
    # the first launch's record is untouched
    first = json.load(open(os.path.join(out, "p", "provenance.json")))
    assert first["lr_policy"]["kind"] == LR_POLICY_TWO_STAGE
    # per-step LR in losses.tsv is the exact per-task rate
    import csv
    rows = list(csv.DictReader(open(os.path.join(out, "p", "logs",
                                                 "losses.tsv")), delimiter="\t"))
    post = [r for r in rows if int(r["step"]) > 6]
    assert post and all(
        float(r["lr"]) == pytest.approx(FINAL4_LRS[r["task"]]) for r in post)
