"""Acceptance tests for the paired 1:2:3 vs 2:2:3 repetition-rescue pilot.

The experiment's validity rests on one arithmetic claim and one state claim:

  * at EQUAL MACRO-CYCLE COUNT the two arms take an identical number of
    naming and comprehension updates and the rescue takes exactly twice the
    repetition updates -- so comparing them at a common global step would be
    the mistake this design exists to avoid;
  * the branch preserves model, the single shared AdamW and its moments, RNG,
    all four cursors and the schedule seed, changing only the ratio.
"""
from __future__ import annotations

import collections
import json
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.train_joint_scratch import (           # noqa: E402
    INTERLEAVED_123, INTERLEAVED_223, LR_POLICY_TASK, OPT_POLICY_SHARED,
    RATIO_123, RATIO_223, SCHEDULE_SEED_BASE, JointScratchTrainer,
    cycle_steps_for, macro_cycle_n, main,
)

JOB = "scripts/cluster/jeanzay/final_rep_rescue_u1200.slurm"
AUDIT = "scripts/cluster/jeanzay/base123_u1200_error_audit.slurm"
SRC_STEP = 3_333_600
DCYCLES = 92_600
MS_CYCLES = 11_575
SEEDS = [19, 20, 21, 22]

CTRL_MS = [3_403_050, 3_472_500, 3_541_950, 3_611_400,
           3_680_850, 3_750_300, 3_819_750, 3_889_200]
RESC_MS = [3_414_625, 3_495_650, 3_576_675, 3_657_700,
           3_738_725, 3_819_750, 3_900_775, 3_981_800]

BASE = ["--regime", "j0", "--subset-mode", "final_full", "--device", "cpu",
        "--max-words", "400", "--batch-size", "8", "--dorsal-pool-size", "32",
        "--lr-boundary-steps", "1", "--eval-every", "0", "--log-every", "0",
        "--glove-path", "tests/_no_such_glove_file.txt",
        "--allow-glove-fallback", "--no-subset-hash-check",
        "--enc-hidden", "64", "--dec-hidden", "64",
        "--lr-repetition", "3e-5", "--lr-naming", "3e-5",
        "--lr-comprehension", "1e-4"]


def script(path=JOB):
    return open(os.path.join(ROOT, path), encoding="utf-8").read()


def ck(out, run_id, step):
    return os.path.join(out, run_id, "checkpoints", f"step_{step:08d}.pt")


def load(p):
    return torch.load(p, map_location="cpu", weights_only=False)


# ==========================  1. matched-cycle arithmetic, from the code  ===

def counts(ratio, sseed, dcycles):
    n = collections.Counter()
    for c in range(dcycles):
        for t in macro_cycle_n(ratio, sseed, c):
            n[t] += 1
    return n


def test_equal_cycles_match_n_and_c_and_double_r():
    """Derived from the REAL cycle builder, not asserted."""
    sseed = 22 * SCHEDULE_SEED_BASE + 4
    for k in range(1, 9):
        dc = MS_CYCLES * k
        a, b = counts(RATIO_123, sseed, dc), counts(RATIO_223, sseed, dc)
        assert a["naming"] == b["naming"] == 2 * dc, k
        assert a["comprehension"] == b["comprehension"] == 3 * dc, k
        assert a["repetition"] == dc and b["repetition"] == 2 * dc, k
    a, b = counts(RATIO_123, sseed, DCYCLES), counts(RATIO_223, sseed, DCYCLES)
    assert (a["repetition"], a["naming"], a["comprehension"]) == \
        (92_600, 185_200, 277_800)
    assert (b["repetition"], b["naming"], b["comprehension"]) == \
        (185_200, 185_200, 277_800)


def test_milestone_step_mapping():
    assert cycle_steps_for(INTERLEAVED_123) == 6
    assert cycle_steps_for(INTERLEAVED_223) == 7
    for k in range(1, 9):
        assert SRC_STEP + 6 * MS_CYCLES * k == CTRL_MS[k - 1], k
        assert SRC_STEP + 7 * MS_CYCLES * k == RESC_MS[k - 1], k
    assert SRC_STEP + 6 * DCYCLES == 3_889_200
    assert SRC_STEP + 7 * DCYCLES == 3_981_800
    # the arms deliberately end at DIFFERENT global steps
    assert CTRL_MS[-1] != RESC_MS[-1]
    # the source sits exactly on a 1:2:3 cycle boundary
    assert SRC_STEP % 6 == 0


def test_cycle_composition_is_exactly_the_declared_ratio():
    sseed = 22 * SCHEDULE_SEED_BASE + 4
    for c in range(200):
        assert collections.Counter(macro_cycle_n(RATIO_123, sseed, c)) == \
            {"repetition": 1, "naming": 2, "comprehension": 3}
        assert collections.Counter(macro_cycle_n(RATIO_223, sseed, c)) == \
            {"repetition": 2, "naming": 2, "comprehension": 3}


# =====================================  2. behaviour of the two arms  ======

@pytest.fixture
def branched(tmp_path):
    """A 1:2:3 source, then both arms branched from it."""
    out = str(tmp_path / "runs")
    assert main(BASE + ["--seed", "19", "--out-dir", out, "--run-id", "pre",
                        "--schedule", "interleaved_123",
                        "--max-steps", "12", "--save-every", "12"]) == 0
    assert main(BASE + ["--seed", "19", "--out-dir", out, "--run-id", "src",
                        "--schedule", "interleaved_123",
                        "--max-steps", "24", "--save-every", "24",
                        "--resume", ck(out, "pre", 12),
                        "--phase-transition"]) == 0
    src = ck(out, "src", 24)
    assert load(src)["global_step"] % 6 == 0
    # control: same schedule, nothing declared.  rescue: 123 -> 223.
    assert main(BASE + ["--seed", "19", "--out-dir", out, "--run-id", "a123",
                        "--schedule", "interleaved_123",
                        "--max-steps", "24 ".strip() and str(24 + 6 * 5),
                        "--save-every", str(24 + 6 * 5),
                        "--resume", src]) == 0
    assert main(BASE + ["--seed", "19", "--out-dir", out, "--run-id", "b223",
                        "--schedule", "interleaved_223",
                        "--max-steps", str(24 + 7 * 5),
                        "--save-every", str(24 + 7 * 5),
                        "--resume", src, "--phase-transition"]) == 0
    return out, src


def test_arms_match_n_and_c_cursors_and_double_r_at_equal_cycles(branched):
    """The whole design, verified on real runs: 5 cycles in each arm."""
    out, src = branched
    s = load(src)["cursors"]
    a = load(ck(out, "a123", 24 + 6 * 5))["cursors"]
    b = load(ck(out, "b223", 24 + 7 * 5))["cursors"]
    assert a["naming"] - s["naming"] == b["naming"] - s["naming"] == 2 * 5
    assert (a["comprehension"] - s["comprehension"]
            == b["comprehension"] - s["comprehension"] == 3 * 5)
    assert a["repetition"] - s["repetition"] == 5
    assert b["repetition"] - s["repetition"] == 2 * 5
    # the pool rides every R step, so it doubles too -- a co-varying factor
    assert b["pool"] - s["pool"] == 2 * (a["pool"] - s["pool"])


def test_only_the_ratio_differs_between_the_arms(branched):
    out, _ = branched
    a = load(ck(out, "a123", 24 + 6 * 5))
    b = load(ck(out, "b223", 24 + 7 * 5))
    assert a["schedule"] == INTERLEAVED_123 and a["schedule_ratio"] == [1, 2, 3]
    assert b["schedule"] == INTERLEAVED_223 and b["schedule_ratio"] == [2, 2, 3]
    for key in ("seed", "stream_seeds", "schedule_seed", "widths",
                "optimizer_policy", "dec_weight", "c_align_weight",
                "lr_policy", "lr_boundary_steps",
                "subset_definition_sha256",
                "comprehension_population_sha256",
                "naming_population_sha256"):
        assert a[key] == b[key], key
    assert a["lr_policy"]["repetition"] == 3e-5
    assert a["lr_policy"]["naming"] == 3e-5
    assert a["lr_policy"]["comprehension"] == 1e-4
    assert a["optimizer_policy"] == OPT_POLICY_SHARED
    assert a["lr_policy"]["kind"] == LR_POLICY_TASK


def test_rescue_anchors_the_new_schedule_at_the_branch(branched):
    out, src = branched
    step = load(src)["global_step"]
    b = load(ck(out, "b223", 24 + 7 * 5))
    assert b["schedule_anchor_step"] == step, \
        "the 2:2:3 cycle index must restart at the branch"
    a = load(ck(out, "a123", 24 + 6 * 5))
    assert a["schedule_anchor_step"] == 0, "the control must not re-anchor"
    rec = b["phase_transitions"][-1]
    assert rec["changed"] == ["schedule"]
    assert rec["old_schedule"] == INTERLEAVED_123
    assert rec["new_schedule"] == INTERLEAVED_223
    assert rec["moment_initialization"] == "unchanged"
    assert len(a["phase_transitions"]) == len(b["phase_transitions"]) - 1


def test_branch_preserves_moments_rng_and_cursors(branched):
    out, src = branched
    s = load(src)
    moments = {i: (st["exp_avg"].clone(), st["exp_avg_sq"].clone())
               for i, st in s["optimizer_state_dict"]["state"].items()}
    tr = JointScratchTrainer(
        regime="j0", seed=19, device="cpu", max_words=400, batch_size=8,
        lexicon_path="data/lexicon_en_glove_covered.tsv", dorsal_pool_size=32,
        subset_mode="final_full", subset_per_band=822, subset_size=32,
        lr_boundary_steps=1, allow_glove_fallback=True,
        require_subset_hash=False, schedule=INTERLEAVED_223,
        glove_path="tests/_no_such_glove_file.txt", enc_hidden=64,
        dec_hidden=64, allow_phase_transition=True,
        task_lrs={"repetition": 3e-5, "naming": 3e-5, "comprehension": 1e-4})
    tr.load_state_dict(s)
    assert tr.cursors == {k: int(v) for k, v in s["cursors"].items()}
    assert tr.global_step == s["global_step"]
    assert tr.task_optims is None, "one shared AdamW only"
    stt = tr.optim.state_dict()["state"]
    for i, (avg, sq) in moments.items():
        assert torch.equal(stt[i]["exp_avg"], avg)
        assert torch.equal(stt[i]["exp_avg_sq"], sq)
    assert torch.equal(torch.get_rng_state(), s["rng_states"]["torch"])
    assert tr.schedule_seed == s["schedule_seed"]


def test_undeclared_ratio_change_is_refused(branched):
    out, src = branched
    kw = dict(regime="j0", seed=19, device="cpu", max_words=400, batch_size=8,
              lexicon_path="data/lexicon_en_glove_covered.tsv",
              dorsal_pool_size=32, subset_mode="final_full",
              subset_per_band=822, subset_size=32, lr_boundary_steps=1,
              allow_glove_fallback=True, require_subset_hash=False,
              schedule=INTERLEAVED_223,
              glove_path="tests/_no_such_glove_file.txt",
              enc_hidden=64, dec_hidden=64,
              task_lrs={"repetition": 3e-5, "naming": 3e-5,
                        "comprehension": 1e-4})
    with pytest.raises(RuntimeError, match="PHASE TRANSITION"):
        JointScratchTrainer(**kw).load_state_dict(load(src))


def test_interrupted_rescue_equals_uninterrupted(branched, tmp_path):
    """The anchor makes the 2:2:3 task order exactly resumable."""
    out, src = branched
    assert main(BASE + ["--seed", "19", "--out-dir", out, "--run-id", "split",
                        "--schedule", "interleaved_223",
                        "--max-steps", str(24 + 7 * 2),
                        "--save-every", str(24 + 7 * 2),
                        "--resume", src, "--phase-transition"]) == 0
    assert main(BASE + ["--seed", "19", "--out-dir", out, "--run-id", "split",
                        "--schedule", "interleaved_223",
                        "--max-steps", str(24 + 7 * 5),
                        "--save-every", str(24 + 7 * 5),
                        "--resume", ck(out, "split", 24 + 7 * 2)]) == 0
    a = load(ck(out, "b223", 24 + 7 * 5))
    b = load(ck(out, "split", 24 + 7 * 5))
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    assert not [k for k in sa if not torch.equal(sa[k], sb[k])]
    assert a["cursors"] == b["cursors"]
    assert a["schedule_anchor_step"] == b["schedule_anchor_step"]


# ==================================================  3. job contracts  =====

def test_training_job_maps_two_arms_four_seeds():
    t = script()
    assert "#SBATCH --array=0-7" in t
    assert "ARMS=(123 123 123 123 223 223 223 223)" in t
    assert "SEEDS=(19 20 21 22 19 20 21 22)" in t
    assert 'RUN_ID="final_rep_${RUN_ARM}_h512_s${SEED}"' in t
    assert "SOURCE_STEP=3333600" in t
    assert "DCYCLES=92600" in t and "MS_CYCLES=11575" in t


def test_training_job_derives_per_arm_steps_from_the_cycle_length():
    t = script()
    assert "SCHEDULE=interleaved_123; CS=6" in t
    assert "SCHEDULE=interleaved_223; CS=7" in t
    assert "MAX_STEPS=$(( SOURCE_STEP + CS * DCYCLES ))" in t
    assert "S=$(( SOURCE_STEP + CS * MS_CYCLES * K ))" in t
    assert "SAVE_EVERY=$(( CS * MS_CYCLES ))" in t
    # the control declares nothing; only the rescue declares the transition
    assert "PHASE=()" in t and "PHASE=(--phase-transition)" in t


def test_training_job_holds_the_lrs_identical_and_changes_nothing_else():
    t = script()
    ex = "\n".join(l for l in t.splitlines()
                   if l.strip() and not l.lstrip().startswith("#"))
    assert "LR_R=3e-5; LR_N=3e-5; LR_C=1e-4" in t
    assert ('--lr-repetition "$LR_R" --lr-naming "$LR_N" '
            '--lr-comprehension "$LR_C"') in ex
    for forbidden in ("--dec-weight", "--c-align-weight", "--optimizer-policy",
                      "--eval-at-start", "--endpoint-eval", "--batch-size",
                      "--lr-boundary-steps", "--allow-glove-fallback",
                      "interleaved_224", "grouped_rn_c"):
        assert forbidden not in ex, forbidden
    assert "CEILING_REQUIRED=5" in t, "the committed ceiling rule is unchanged"
    assert 'REPO="$L3_REPO"' in ex and 'REPO=${L3_REPO:-' not in ex


def test_training_job_protects_every_parent_lineage():
    t = script()
    assert '[[ "$RUN_ID" != final_base123_* && "$RUN_ID" != final_lrpilot* ]]' in t
    assert 'PARENT_RUN_ID="final_lrpilot1200_chigh_h512_s${SEED}"' in t
    assert "READ ONLY" in t
    assert "final_rep_rescue123_h512_s19" in t
    assert "final_rep_rescue223_h512_s22" in t


def test_training_job_preflight_pins_the_chigh_source():
    t = script()
    for probe in ('int(ck["global_step"]) == want',
                  "want % 6 == 0",
                  'ck.get("schedule") == "interleaved_123"',
                  'abs(float(p["comprehension"]) - 1e-4) < 1e-12',
                  'abs(float(p["repetition"]) - 3e-5) < 1e-12',
                  'int(c["naming"]) == 2 * int(c["repetition"])',
                  'int(c["comprehension"]) == 3 * int(c["repetition"])',
                  'int(c["pool"]) == int(c["repetition"])',
                  '10c2f06eda769bf620ca3dbb9889204e4431cac2bfe0d0f5dd37fa4df2bb9f50'):
        assert probe in t, probe


def test_audit_job_is_read_only_and_covers_u1200_and_u850():
    t = script(AUDIT)
    assert "#SBATCH --array=0-11" in t
    assert t.count("3333600") >= 8 and t.count("2361300") >= 4
    assert "final_lrpilot1200_chigh_h512_s19" in t
    assert "final_lrpilot1200_all3e5_h512_s19" in t
    assert "final_lrpilot_3e5_h512_s19" in t
    ex = "\n".join(l for l in t.splitlines()
                   if l.strip() and not l.lstrip().startswith("#"))
    assert "base123_error_audit.py" in ex
    assert "train_joint_scratch.py" not in ex, "the audit must not train"
    for forbidden in ("--resume", "--max-steps", "--stop-at-ceiling", "rm ",
                      "mv "):
        assert forbidden not in ex, forbidden
    assert "error_audit_u" in t


def test_paired_report_thresholds_are_the_preregistered_ones():
    from scripts.naming_comprehension import paired_rescue_report as m
    assert m.LTM_TARGET == 0.92
    assert m.LTM_MIN_SEEDS == 3
    assert m.C_NONINFERIORITY == 5.0
    assert m.LTM_NULL_BAND == 0.02
    assert m.MS_CYCLES == 11_575 and m.N_MILESTONES == 8
    assert m.CYCLE_STEPS == {"123": 6, "223": 7}
    assert m.SRC_STEP == SRC_STEP
    assert m.milestone_steps("123") == CTRL_MS
    assert m.milestone_steps("223") == RESC_MS
