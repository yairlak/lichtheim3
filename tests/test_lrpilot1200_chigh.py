"""Acceptance tests for the u850 -> u1200 C-high LR pilot.

Two arms branch from the same 3e-5 u850 state.  Repetition and naming stay at
3e-5 in both; only the COMPREHENSION task LR moves (3e-5 -> 1e-4).  The tests
prove the control is a genuine continuation of the source policy, that the
arms differ in nothing but the C rate, and that the parent lineage cannot be
written to.
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
    INTERLEAVED_123, LR_POLICY_TASK, OPT_POLICY_SHARED, JointScratchTrainer,
    main,
)

JOB = "scripts/cluster/jeanzay/final_lrpilot_u850_to_u1200.slurm"
PREV = "scripts/cluster/jeanzay/final_lrpilot_u750_to_u850.slurm"
R_PASS, CYCLE = 463, 6
U850, U1200 = 2_361_300, 3_333_600
GRID_US = list(range(875, 1201, 25))

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


def lrs(r, n, c):
    return ["--lr-repetition", r, "--lr-naming", n, "--lr-comprehension", c]


@pytest.fixture
def branched(tmp_path):
    """A 3e-5 source (the u850 analogue), then both arms branched from it."""
    out = str(tmp_path / "runs")
    assert main(BASE + ["--seed", "19", "--out-dir", out, "--run-id", "pre",
                        "--max-steps", "12", "--save-every", "12"]) == 0
    assert main(BASE + ["--seed", "19", "--out-dir", out, "--run-id", "src",
                        "--max-steps", "24", "--save-every", "24",
                        "--resume", ck(out, "pre", 12), "--phase-transition"]
                + lrs("3e-5", "3e-5", "3e-5")) == 0
    src = ck(out, "src", 24)
    p = load(src)["lr_policy"]
    assert p["kind"] == LR_POLICY_TASK and p["comprehension"] == 3e-5
    for rid, c in (("all3e5", "3e-5"), ("chigh", "1e-4")):
        assert main(BASE + ["--seed", "19", "--out-dir", out, "--run-id", rid,
                            "--max-steps", "36", "--save-every", "36",
                            "--resume", src, "--phase-transition"]
                    + lrs("3e-5", "3e-5", c)) == 0
    return out, src


# ==========================  1. the control continues the source policy  ===

def test_resuming_a_task_lr_checkpoint_without_the_flags_is_refused(branched):
    """Safety property the design relies on: omitting the flags would revert
    to the two-stage policy, so both arms MUST declare all three."""
    out, src = branched
    with pytest.raises(RuntimeError, match="PHASE TRANSITION"):
        main(BASE + ["--seed", "19", "--out-dir", out, "--run-id", "bad",
                     "--max-steps", "36", "--save-every", "36",
                     "--resume", src])


def test_control_declaration_is_inert(tmp_path):
    """Declaring the source's own values must change nothing: the control arm
    with --phase-transition is bitwise identical to the same run without it,
    so the control is a true continuation, not a re-declared experiment."""
    out = str(tmp_path / "runs")
    assert main(BASE + ["--seed", "19", "--out-dir", out, "--run-id", "pre",
                        "--max-steps", "12", "--save-every", "12"]) == 0
    assert main(BASE + ["--seed", "19", "--out-dir", out, "--run-id", "src",
                        "--max-steps", "24", "--save-every", "24",
                        "--resume", ck(out, "pre", 12), "--phase-transition"]
                + lrs("3e-5", "3e-5", "3e-5")) == 0
    src = ck(out, "src", 24)
    for rid, extra in (("withflag", ["--phase-transition"]), ("noflag", [])):
        assert main(BASE + ["--seed", "19", "--out-dir", out, "--run-id", rid,
                            "--max-steps", "36", "--save-every", "36",
                            "--resume", src] + extra
                    + lrs("3e-5", "3e-5", "3e-5")) == 0
    a, b = load(ck(out, "noflag", 36)), load(ck(out, "withflag", 36))
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    assert not [k for k in sa if not torch.equal(sa[k], sb[k])]
    oa, ob = a["optimizer_state_dict"]["state"], b["optimizer_state_dict"]["state"]
    for i in oa:
        for m in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(oa[i][m], ob[i][m])
    assert a["cursors"] == b["cursors"]
    assert torch.equal(a["rng_states"]["torch"], b["rng_states"]["torch"])
    assert len(a["phase_transitions"]) == len(b["phase_transitions"]) == 1


# =====================================  2. only the C rate differs  ========

def test_arms_differ_and_only_in_the_comprehension_lr(branched):
    out, _ = branched
    a, b = load(ck(out, "all3e5", 36)), load(ck(out, "chigh", 36))
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    assert [k for k in sa if not torch.equal(sa[k], sb[k])], \
        "a different C LR must produce different weights"
    # R and N rates are identical; only C moves
    for t in ("repetition", "naming"):
        assert a["lr_policy"][t] == b["lr_policy"][t] == 3e-5, t
    assert a["lr_policy"]["comprehension"] == 3e-5
    assert b["lr_policy"]["comprehension"] == 1e-4
    # everything else that defines the experiment matches
    for key in ("cursors", "global_step", "stream_seeds", "schedule",
                "schedule_ratio", "schedule_seed", "schedule_anchor_step",
                "widths", "optimizer_policy", "dec_weight", "c_align_weight",
                "subset_definition_sha256", "comprehension_population_sha256",
                "naming_population_sha256", "lr_boundary_steps",
                "consecutive_ceiling"):
        assert a[key] == b[key], key
    assert torch.equal(a["rng_states"]["torch"], b["rng_states"]["torch"])
    assert a["optimizer_policy"] == OPT_POLICY_SHARED
    # the control records no new transition; the hybrid records exactly one
    assert len(a["phase_transitions"]) == 1
    assert len(b["phase_transitions"]) == 2
    rec = b["phase_transitions"][-1]
    assert rec["old_lr_policy"]["comprehension"] == 3e-5
    assert rec["new_lr_policy"]["comprehension"] == 1e-4
    assert rec["new_lr_policy"]["repetition"] == 3e-5
    assert rec["moment_initialization"] == "unchanged"


def test_per_task_lr_is_actually_asymmetric():
    tr = JointScratchTrainer(
        regime="j0", seed=19, device="cpu", max_words=400, batch_size=8,
        lexicon_path="data/lexicon_en_glove_covered.tsv", dorsal_pool_size=32,
        subset_mode="final_full", subset_per_band=822, subset_size=32,
        lr_boundary_steps=1, allow_glove_fallback=True,
        require_subset_hash=False, schedule=INTERLEAVED_123,
        glove_path="tests/_no_such_glove_file.txt", enc_hidden=64,
        dec_hidden=64,
        task_lrs={"repetition": 3e-5, "naming": 3e-5, "comprehension": 1e-4})
    assert tr.current_lr("repetition") == 3e-5
    assert tr.current_lr("naming") == 3e-5
    assert tr.current_lr("comprehension") == 1e-4
    tr.global_step = 10 ** 6
    assert tr.current_lr("comprehension") == 1e-4, "must not drift with time"


# ===========================================  3. state preservation  =======

def test_branch_preserves_moments_rng_cursors_and_streak(branched):
    out, src = branched
    s = load(src)
    moments = {i: (st["exp_avg"].clone(), st["exp_avg_sq"].clone())
               for i, st in s["optimizer_state_dict"]["state"].items()}
    tr = JointScratchTrainer(
        regime="j0", seed=19, device="cpu", max_words=400, batch_size=8,
        lexicon_path="data/lexicon_en_glove_covered.tsv", dorsal_pool_size=32,
        subset_mode="final_full", subset_per_band=822, subset_size=32,
        lr_boundary_steps=1, allow_glove_fallback=True,
        require_subset_hash=False, schedule=INTERLEAVED_123,
        glove_path="tests/_no_such_glove_file.txt", enc_hidden=64,
        dec_hidden=64, allow_phase_transition=True,
        task_lrs={"repetition": 3e-5, "naming": 3e-5, "comprehension": 1e-4})
    tr.load_state_dict(s)
    assert tr.global_step == s["global_step"]
    assert tr.cursors == {k: int(v) for k, v in s["cursors"].items()}
    assert set(tr.cursors) == {"repetition", "pool", "naming", "comprehension"}
    assert tr.schedule_anchor_step == s["schedule_anchor_step"]
    assert tr.consecutive_ceiling == s["consecutive_ceiling"]
    assert tr.last_ceiling_step == s["last_ceiling_step"]
    assert tr.task_optims is None, "one shared AdamW only"
    st = tr.optim.state_dict()["state"]
    for i, (avg, sq) in moments.items():
        assert torch.equal(st[i]["exp_avg"], avg), f"exp_avg moved at {i}"
        assert torch.equal(st[i]["exp_avg_sq"], sq), f"exp_avg_sq moved at {i}"
    assert torch.equal(torch.get_rng_state(), s["rng_states"]["torch"])


def test_branch_records_ancestry_to_the_3e5_parent(branched):
    out, src = branched
    for rid in ("all3e5", "chigh"):
        anc = json.load(open(os.path.join(out, rid, "provenance.json")))["ancestry"]
        assert anc["is_branch"] is True
        assert anc["parent_run_id"] == "src" != rid
        assert anc["parent_global_step"] == 24
        assert anc["parent_lr_policy"]["comprehension"] == 3e-5
        for k in ("model", "optimizer moments", "task cursors",
                  "macro-cycle position", "RNG states"):
            assert k in anc["inherited"]


# ==================================================  4. job contract  ======

def test_job_maps_two_arms_four_seeds():
    t = script()
    assert "#SBATCH --array=0-7" in t
    assert "ARMS=(all3e5 all3e5 all3e5 all3e5 chigh chigh chigh chigh)" in t
    assert "LR_C=(3e-5 3e-5 3e-5 3e-5 1e-4 1e-4 1e-4 1e-4)" in t
    assert "SEEDS=(19 20 21 22 19 20 21 22)" in t
    assert 'RUN_ID="final_lrpilot1200_${ARM}_h512_s${SEED}"' in t
    assert len(set(zip(["all3e5"] * 4 + ["chigh"] * 4,
                       [19, 20, 21, 22] * 2))) == 8


def test_job_holds_r_and_n_fixed_and_moves_only_c():
    t = script()
    ex = "\n".join(l for l in t.splitlines()
                   if l.strip() and not l.lstrip().startswith("#"))
    assert "LR_R=3e-5" in t and "LR_N=3e-5" in t
    assert "LR_COMP=${LR_C[$IDX]}" in t
    assert ('--lr-repetition "$LR_R" --lr-naming "$LR_N" '
            '--lr-comprehension "$LR_COMP"') in ex
    for forbidden in ("--dec-weight", "--c-align-weight", "--optimizer-policy",
                      "interleaved_223", "grouped_rn_c", "task_separated",
                      "--endpoint-eval", "--eval-at-start", "--batch-size",
                      "--lr-boundary-steps", "--allow-glove-fallback"):
        assert forbidden not in ex, forbidden
    assert "EVAL_EVERY=13890" in t and "SAVE_EVERY=69450" in t
    assert "CEILING_REQUIRED=5" in t
    assert 'REPO="$L3_REPO"' in ex and 'REPO=${L3_REPO:-' not in ex


def test_job_pins_source_and_the_full_u1200_grid():
    t = script()
    assert "SOURCE_STEP=2361300" in t and "MAX_STEPS=3333600" in t
    assert 'PARENT_RUN_ID="final_lrpilot_3e5_h512_s${SEED}"' in t
    assert 'int(ck["cursors"]["repetition"]) == 393550' in t
    assert 'abs(float(p[t]) - 3e-5) < 1e-12' in t, \
        "the source must be verified to be the 3e-5 arm"
    assert "L3_U850_SHA" in t and "sha256sum" in t
    # the grid is every 25u from u875 to u1200: 14 milestones, in order,
    # each an exact SAVE_EVERY multiple
    expected = [u * R_PASS * CYCLE for u in GRID_US]
    assert len(expected) == 14
    assert f"FULL_EVAL_AT={','.join(str(s) for s in expected)}" in t
    assert expected == sorted(expected)
    for s in expected:
        assert s % 69450 == 0
    assert expected[0] == 2_430_750 and expected[-1] == U1200
    assert U850 == 850 * R_PASS * CYCLE
    # the horizon is 350u past the source
    assert (U1200 - U850) // (R_PASS * CYCLE) == 350


def test_horizon_allows_the_five_milestone_ceiling_rule_to_fire():
    """Over four milestones the rule could not fire at all; over fourteen it
    can, so an early stop here is a genuine stable-ceiling result."""
    t = script()
    assert "CEILING_REQUIRED=5" in t
    assert len(GRID_US) >= 5
    assert "CAN fire" in t


def test_job_cannot_write_into_any_parent_lineage():
    t = script()
    assert '[[ "$RUN_ID" != final_base123_* ]]' in t
    assert '[[ "$RUN_ID" != "$PARENT_RUN_ID" ]]' in t
    assert "final_lrpilot1200_all3e5_h512_s19" in t
    assert "final_lrpilot1200_chigh_h512_s22" in t
    assert "READ ONLY" in t


def test_previous_pilot_script_is_untouched():
    t = script(PREV)
    assert "MAX_STEPS=2361300" in t
    assert "3333600" not in t and "2639100" not in t
