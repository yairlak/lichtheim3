"""Acceptance tests for the u750 -> u850 paired LR pilot.

The experiment's validity rests on three claims:
  1. the CONTROL arm is exactly "keep going" -- a flat task-LR of 1e-4 must be
     BITWISE IDENTICAL to continuing under the historical two-stage policy,
     whose stage 2 is 1e-4;
  2. the two arms differ ONLY in that numeric LR;
  3. the branch preserves model, shared AdamW moments, RNG, cursors, schedule
     anchor and streak, and cannot touch the canonical lineage.
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

from scripts.naming_comprehension.train_joint_scratch import (           # noqa: E402
    INTERLEAVED_123, LR_POLICY_TASK, LR_POLICY_TWO_STAGE, LR_STAGE2,
    OPT_POLICY_SHARED, JointScratchTrainer, main,
)

JOB = "scripts/cluster/jeanzay/final_lrpilot_u750_to_u850.slurm"
CONT = "scripts/cluster/jeanzay/final_base123_h512_continue750.slurm"
R_PASS, CYCLE = 463, 6
U750, U850 = 2_083_500, 2_361_300

# lr_boundary_steps 1 puts the tiny fixture past the boundary immediately, so
# the two-stage policy resolves to stage 2 (1e-4) exactly as it does at u750.
BASE = ["--regime", "j0", "--subset-mode", "final_full", "--device", "cpu",
        "--max-words", "400", "--batch-size", "8", "--dorsal-pool-size", "32",
        "--lr-boundary-steps", "1", "--eval-every", "0", "--log-every", "0",
        "--glove-path", "tests/_no_such_glove_file.txt",
        "--allow-glove-fallback", "--no-subset-hash-check",
        "--schedule", "interleaved_123",
        "--enc-hidden", "64", "--dec-hidden", "64"]


def script(path=JOB):
    return open(os.path.join(ROOT, path), encoding="utf-8").read()


def ck(out, run_id, step):
    return os.path.join(out, run_id, "checkpoints", f"step_{step:08d}.pt")


def load(p):
    return torch.load(p, map_location="cpu", weights_only=False)


def flat_lr(lr):
    return ["--lr-repetition", lr, "--lr-naming", lr, "--lr-comprehension", lr]


@pytest.fixture
def branched(tmp_path):
    """A source past the LR boundary, then both arms branched from it."""
    out = str(tmp_path / "runs")
    assert main(BASE + ["--seed", "19", "--out-dir", out, "--run-id", "src",
                        "--max-steps", "12", "--save-every", "12"]) == 0
    src = ck(out, "src", 12)
    assert load(src)["lr"] == LR_STAGE2, "the fixture must sit at stage 2"
    # plain continuation == what the canonical lineage would have done
    assert main(BASE + ["--seed", "19", "--out-dir", out, "--run-id", "plain",
                        "--max-steps", "24", "--save-every", "24",
                        "--resume", src]) == 0
    for rid, lr in (("control", "1e-4"), ("treat", "3e-5")):
        assert main(BASE + ["--seed", "19", "--out-dir", out, "--run-id", rid,
                            "--max-steps", "24", "--save-every", "24",
                            "--resume", src, "--phase-transition"]
                    + flat_lr(lr)) == 0
    return out, src


# =====================================  1. the control IS "keep going"  ====

def test_control_is_bitwise_identical_to_plain_continuation(branched):
    """A flat task-LR of 1e-4 must reproduce the two-stage policy exactly,
    otherwise the control is not a true control."""
    out, _ = branched
    a, b = load(ck(out, "plain", 24)), load(ck(out, "control", 24))
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    assert not [k for k in sa if not torch.equal(sa[k], sb[k])], "model differs"
    oa = a["optimizer_state_dict"]["state"]
    ob = b["optimizer_state_dict"]["state"]
    assert set(oa) == set(ob)
    for i in oa:
        for m in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(oa[i][m], ob[i][m]), f"{m} differs at {i}"
    assert a["cursors"] == b["cursors"]
    assert torch.equal(a["rng_states"]["torch"], b["rng_states"]["torch"])
    assert a["lr"] == b["lr"] == LR_STAGE2


def test_flat_task_lr_is_constant_across_tasks_and_steps():
    tr = JointScratchTrainer(
        regime="j0", seed=19, device="cpu", max_words=400, batch_size=8,
        lexicon_path="data/lexicon_en_glove_covered.tsv", dorsal_pool_size=32,
        subset_mode="final_full", subset_per_band=822, subset_size=32,
        lr_boundary_steps=1, allow_glove_fallback=True,
        require_subset_hash=False, schedule=INTERLEAVED_123,
        glove_path="tests/_no_such_glove_file.txt", enc_hidden=64,
        dec_hidden=64,
        task_lrs={"repetition": 3e-5, "naming": 3e-5, "comprehension": 3e-5})
    assert tr.lr_policy["kind"] == LR_POLICY_TASK
    for task in ("repetition", "naming", "comprehension"):
        assert tr.current_lr(task) == 3e-5
    tr.global_step = 10 ** 6                     # LR must not drift with time
    for task in ("repetition", "naming", "comprehension"):
        assert tr.current_lr(task) == 3e-5


# ==========================================  2. only the LR differs  =======

def test_the_two_arms_differ_and_only_in_the_learning_rate(branched):
    out, _ = branched
    a, b = load(ck(out, "control", 24)), load(ck(out, "treat", 24))
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    assert [k for k in sa if not torch.equal(sa[k], sb[k])], \
        "a different LR must produce different weights"
    # everything that defines the experiment is identical
    assert a["cursors"] == b["cursors"]
    assert a["global_step"] == b["global_step"]
    assert a["stream_seeds"] == b["stream_seeds"]
    assert a["schedule"] == b["schedule"] == INTERLEAVED_123
    assert a["schedule_ratio"] == b["schedule_ratio"] == [1, 2, 3]
    assert a["schedule_anchor_step"] == b["schedule_anchor_step"]
    assert a["widths"] == b["widths"]
    assert a["optimizer_policy"] == b["optimizer_policy"] == OPT_POLICY_SHARED
    assert a["dec_weight"] == b["dec_weight"] == 0.5
    assert a["c_align_weight"] == b["c_align_weight"] == 0.0
    assert a["subset_definition_sha256"] == b["subset_definition_sha256"]
    assert a["comprehension_population_sha256"] == \
        b["comprehension_population_sha256"]
    assert torch.equal(a["rng_states"]["torch"], b["rng_states"]["torch"])
    # and the only differing scientific field is the LR
    assert a["lr_policy"]["repetition"] == 1e-4
    assert b["lr_policy"]["repetition"] == 3e-5
    for t in ("repetition", "naming", "comprehension"):
        assert a["lr_policy"][t] == 1e-4 and b["lr_policy"][t] == 3e-5


def test_branch_preserves_moments_rng_and_cursors_from_the_source(branched):
    out, src = branched
    s = load(src)
    moments = {i: st["exp_avg"].clone()
               for i, st in s["optimizer_state_dict"]["state"].items()}
    tr = JointScratchTrainer(
        regime="j0", seed=19, device="cpu", max_words=400, batch_size=8,
        lexicon_path="data/lexicon_en_glove_covered.tsv", dorsal_pool_size=32,
        subset_mode="final_full", subset_per_band=822, subset_size=32,
        lr_boundary_steps=1, allow_glove_fallback=True,
        require_subset_hash=False, schedule=INTERLEAVED_123,
        glove_path="tests/_no_such_glove_file.txt", enc_hidden=64,
        dec_hidden=64, allow_phase_transition=True,
        task_lrs={"repetition": 3e-5, "naming": 3e-5, "comprehension": 3e-5})
    tr.load_state_dict(s)
    assert tr.global_step == s["global_step"]
    assert tr.cursors == {k: int(v) for k, v in s["cursors"].items()}
    assert tr.schedule_anchor_step == s["schedule_anchor_step"]
    assert tr.consecutive_ceiling == s["consecutive_ceiling"]
    assert tr.optimizer_policy == OPT_POLICY_SHARED and tr.task_optims is None
    for i, ref in moments.items():
        assert torch.equal(tr.optim.state_dict()["state"][i]["exp_avg"], ref), \
            "AdamW moments must survive the LR transition"
    assert torch.equal(torch.get_rng_state(), s["rng_states"]["torch"])


def test_the_lr_change_is_refused_unless_declared(branched):
    out, src = branched
    kw = dict(regime="j0", seed=19, device="cpu", max_words=400, batch_size=8,
              lexicon_path="data/lexicon_en_glove_covered.tsv",
              dorsal_pool_size=32, subset_mode="final_full",
              subset_per_band=822, subset_size=32, lr_boundary_steps=1,
              allow_glove_fallback=True, require_subset_hash=False,
              schedule=INTERLEAVED_123,
              glove_path="tests/_no_such_glove_file.txt",
              enc_hidden=64, dec_hidden=64,
              task_lrs={"repetition": 3e-5, "naming": 3e-5,
                        "comprehension": 3e-5})
    with pytest.raises(RuntimeError, match="PHASE TRANSITION"):
        JointScratchTrainer(**kw).load_state_dict(load(src))


def test_the_transition_is_recorded_with_old_and_new_policy(branched):
    out, _ = branched
    for rid, lr in (("control", 1e-4), ("treat", 3e-5)):
        end = load(ck(out, rid, 24))
        rec = end["phase_transitions"][-1]
        assert rec["old_lr_policy"]["kind"] == LR_POLICY_TWO_STAGE
        assert rec["new_lr_policy"]["kind"] == LR_POLICY_TASK
        assert rec["new_lr_policy"]["repetition"] == lr
        assert rec["moment_initialization"] == "unchanged"


def test_branch_records_its_ancestry(branched):
    """A branch must be reconstructible from its provenance alone."""
    out, src = branched
    for rid, lr in (("control", 1e-4), ("treat", 3e-5)):
        prov = json.load(open(os.path.join(out, rid, "provenance.json")))
        anc = prov["ancestry"]
        assert anc["is_branch"] is True, "must be recorded as a branch"
        assert anc["parent_run_id"] == "src"
        assert anc["parent_run_id"] != rid
        assert os.path.basename(anc["parent_checkpoint"]) == \
            os.path.basename(src)
        assert anc["parent_global_step"] == 12
        assert anc["parent_lr_policy"]["kind"] == LR_POLICY_TWO_STAGE
        for inherited in ("model", "optimizer moments", "task cursors",
                          "macro-cycle position", "RNG states"):
            assert inherited in anc["inherited"], inherited
        # the launch's own treatment is recorded too
        assert prov["resumed_at_step"] == 12
        assert json.dumps(prov).count(str(lr)) >= 1


# ==================================================  3. job contract  ======

def test_job_maps_eight_runs_two_arms_four_seeds():
    t = script()
    assert "#SBATCH --array=0-7" in t
    assert "ARMS=(control control control control 3e5 3e5 3e5 3e5)" in t
    assert "LRS=(1e-4 1e-4 1e-4 1e-4 3e-5 3e-5 3e-5 3e-5)" in t
    assert "SEEDS=(19 20 21 22 19 20 21 22)" in t
    assert 'RUN_ID="final_lrpilot_${ARM}_h512_s${SEED}"' in t
    arms = ["control"] * 4 + ["3e5"] * 4
    seeds = [19, 20, 21, 22] * 2
    assert len(set(zip(arms, seeds))) == 8


def test_job_pins_the_source_and_the_grid():
    t = script()
    assert "SOURCE_STEP=2083500" in t and "MAX_STEPS=2361300" in t
    assert "FULL_EVAL_AT=2152950,2222400,2291850,2361300" in t
    assert 'int(ck["cursors"]["repetition"]) == 347250' in t
    assert 'ck["lr_policy"]["kind"] == "two_stage_rep_cursor"' in t
    assert "sha256sum" in t and "L3_U750_SHA" in t
    for u, step in ((750, U750), (775, 2_152_950), (800, 2_222_400),
                    (825, 2_291_850), (850, U850)):
        assert u * R_PASS * CYCLE == step
        assert step % 69450 == 0, "every milestone must be on the save cadence"


def test_job_cannot_touch_the_canonical_lineage():
    t = script()
    assert '[[ "$RUN_ID" != final_base123_* ]]' in t
    assert "would write into the canonical lineage" in t
    assert 'PARENT_RUN_ID="final_base123_h512_s${SEED}"' in t
    assert '[[ "$RUN_ID" != "$PARENT_RUN_ID" ]]' in t
    assert "final_lrpilot_control_h512_s19" in t
    assert "final_lrpilot_3e5_h512_s22" in t
    assert "READ ONLY" in t


def test_job_changes_nothing_but_the_lr():
    t = script()
    ex = "\n".join(l for l in t.splitlines()
                   if l.strip() and not l.lstrip().startswith("#"))
    for forbidden in ("--dec-weight", "--c-align-weight", "--optimizer-policy",
                      "interleaved_223", "grouped_rn_c", "task_separated",
                      "--endpoint-eval", "--eval-at-start", "--batch-size",
                      "--lr-boundary-steps", "--allow-glove-fallback"):
        assert forbidden not in ex, forbidden
    # the three task LRs are all driven by the SAME variable
    assert '--lr-repetition "$LR" --lr-naming "$LR" --lr-comprehension "$LR"' in ex
    assert "EVAL_EVERY=13890" in t and "SAVE_EVERY=69450" in t
    assert "CEILING_REQUIRED=5" in t
    assert '--ceiling-consecutive-required "$CEILING_REQUIRED"' in ex
    assert 'REPO="$L3_REPO"' in ex and 'REPO=${L3_REPO:-' not in ex


def test_requeue_declares_no_second_transition():
    t = script()
    assert "nothing re-declared" in t
    assert "OWN_STEP >= MAX_STEPS" in t
    assert "sort | tail -1" in t
    assert "#SBATCH --requeue" in t


def test_the_continuation_script_is_untouched():
    t = script(CONT)
    assert "MAX_STEPS=2083500" in t and "--array=0-3" in t
    assert "2361300" not in t
