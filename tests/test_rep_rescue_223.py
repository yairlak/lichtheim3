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
    # BOTH arms re-anchor at the branch: the control is told to explicitly,
    # the rescue does so because its schedule changes.
    assert main(BASE + ["--seed", "19", "--out-dir", out, "--run-id", "a123",
                        "--schedule", "interleaved_123",
                        "--max-steps", str(24 + 6 * 5),
                        "--save-every", str(24 + 6 * 5),
                        "--resume", src, "--reanchor-schedule",
                        "--phase-transition"]) == 0
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
    a = load(ck(out, "a123", 24 + 6 * 5))
    # THE ANCHOR CONFOUND FIX: both arms share the same schedule origin
    assert a["schedule_anchor_step"] == b["schedule_anchor_step"] == step, \
        "both arms must start post-branch macro-cycle index 0"
    assert a["schedule_seed"] == b["schedule_seed"]
    rec = b["phase_transitions"][-1]
    assert rec["changed"] == ["schedule"]
    assert rec["old_schedule"] == INTERLEAVED_123
    assert rec["new_schedule"] == INTERLEAVED_223
    assert rec["moment_initialization"] == "unchanged"
    arec = a["phase_transitions"][-1]
    assert arec["changed"] == ["schedule_anchor"]
    assert arec["old_schedule"] == arec["new_schedule"] == INTERLEAVED_123
    assert arec["old_schedule_ratio"] == arec["new_schedule_ratio"] == [1, 2, 3]
    assert arec["reanchored"] == 1 and rec["reanchored"] == 1
    assert arec["moment_initialization"] == "unchanged"


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
    # both arms declare; only the control needs the explicit re-anchor
    assert "PHASE=(--reanchor-schedule --phase-transition)" in t
    assert "PHASE=(--phase-transition)" in t


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


# ==========================  the anchor confound and its removal  ==========

def _src(out):
    assert main(BASE + ["--seed", "19", "--out-dir", out, "--run-id", "s",
                        "--schedule", "interleaved_123", "--max-steps", "24",
                        "--save-every", "24", "--phase-transition"]) == 0
    return ck(out, "s", 24)


def _cont(out, rid, src, extra):
    assert main(BASE + ["--seed", "19", "--out-dir", out, "--run-id", rid,
                        "--schedule", "interleaved_123", "--max-steps", "54",
                        "--save-every", "54", "--resume", src] + extra) == 0
    return load(ck(out, rid, 54))


def test_the_anchor_confound_was_real(tmp_path):
    """Without the fix the control keeps its historical cycle numbering while
    the rescue restarts at 0, which changes the deterministic task ORDER.
    Demonstrate that this is a real difference, not a bookkeeping detail."""
    out = str(tmp_path / "runs")
    src = _src(out)
    assert load(src)["schedule_anchor_step"] == 0
    assert load(src)["global_step"] % 6 == 0
    old = _cont(out, "old", src, [])
    new = _cont(out, "new", src, ["--reanchor-schedule", "--phase-transition"])
    assert old["schedule_anchor_step"] == 0
    assert new["schedule_anchor_step"] == 24
    so, sn = old["model_state_dict"], new["model_state_dict"]
    assert [k for k in so if not torch.equal(so[k], sn[k])], \
        "the cycle origin must genuinely affect the trajectory"


def test_nuisance_control_two_reanchored_123_arms_are_identical(tmp_path):
    """With both arms re-anchored and the SAME ratio, they must be bitwise
    identical from the branch -- so any later difference is attributable to
    the ratio alone."""
    out = str(tmp_path / "runs")
    src = _src(out)
    a = _cont(out, "c1", src, ["--reanchor-schedule", "--phase-transition"])
    b = _cont(out, "c2", src, ["--reanchor-schedule", "--phase-transition"])
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    assert not [k for k in sa if not torch.equal(sa[k], sb[k])]
    oa, ob = a["optimizer_state_dict"]["state"], b["optimizer_state_dict"]["state"]
    for i in oa:
        for m in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(oa[i][m], ob[i][m])
    assert a["cursors"] == b["cursors"]
    assert a["schedule_anchor_step"] == b["schedule_anchor_step"] == 24
    assert torch.equal(a["rng_states"]["torch"], b["rng_states"]["torch"])


def test_reanchor_requires_declaration_and_is_a_no_op_by_default(tmp_path):
    out = str(tmp_path / "runs")
    src = _src(out)
    with pytest.raises(RuntimeError, match="PHASE TRANSITION"):
        main(BASE + ["--seed", "19", "--out-dir", out, "--run-id", "bad",
                     "--schedule", "interleaved_123", "--max-steps", "54",
                     "--save-every", "54", "--resume", src,
                     "--reanchor-schedule"])
    # omitting the flag leaves every pre-existing run untouched
    plain = _cont(out, "plain", src, [])
    assert plain["schedule_anchor_step"] == 0
    assert plain["phase_transitions"] == []


def test_launcher_reanchors_both_arms():
    t = script()
    assert "PHASE=(--reanchor-schedule --phase-transition)" in t
    assert "PHASE=(--phase-transition)" in t
    assert "both arms start post-branch macro-cycle index 0" in t
    assert "both arms begin post-branch cycle index 0" in t
    assert "TRAIN_BLOB_EXPECTED=95295d63560ae4c235a6beee8dfb47166f4ed30d" in t
    assert "INCLUDING ITS ATTACHED DORSAL POOL AUXILIARY" in t


def test_persistence_report_accepts_lrpilot_run_ids():
    from scripts.naming_comprehension import base123_persistence_report as m
    assert m.DEFAULT_RUN_TEMPLATE == "final_base123_h{width}_s{seed}"
    d = m.audit_dir("/R", 19, 1200, 512,
                    "final_lrpilot1200_chigh_h512_s{seed}")
    assert d == "/R/final_lrpilot1200_chigh_h512_s19/error_audit_u1200"
    d = m.audit_dir("/R", 22, 850, 512, "final_lrpilot_3e5_h512_s{seed}")
    assert d == "/R/final_lrpilot_3e5_h512_s22/error_audit_u850"
    # default behaviour is unchanged
    assert m.audit_dir("/R", 19, 500, 512) == \
        "/R/final_base123_h512_s19/error_audit_u500"
    # and NO threshold moved
    assert m.MARGIN_EPS == 0.01
    import inspect
    src = inspect.getsource(m.switch_trigger)
    assert ">= 0.85" in src and ">= 0.80" in src
    assert "0.90" in src and "0.50" in src


# ================================  GloVe resolution (the 1857757/8 failure) ==

GLOVE_SHA = "91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed"


def test_glove_is_gitignored_so_a_worktree_never_has_it():
    """Root cause: data/glove.*.txt is not tracked, so a detached worktree
    contains no GloVe and the driver's cwd-relative default cannot resolve."""
    import subprocess
    tracked = subprocess.run(["git", "-C", ROOT, "ls-files", "data"],
                             capture_output=True, text=True).stdout
    assert "glove.6B.300d.txt" not in tracked
    ignore = open(os.path.join(ROOT, ".gitignore"), encoding="utf-8").read()
    assert "data/glove.*.txt" in ignore


@pytest.mark.parametrize("path", [JOB, AUDIT])
def test_both_launchers_verify_and_pass_the_real_glove(path):
    t = script(path)
    ex = "\n".join(l for l in t.splitlines()
                   if l.strip() and not l.lstrip().startswith("#"))
    # explicit, overridable, absolute canonical path
    assert "GLOVE=${L3_GLOVE:-/lustre/fswork/projects/rech/llg/uss35bp/" \
           "lichtheim3/lichtheim3/data/glove.6B.300d.txt}" in ex
    # existence AND sha verified before any Python
    assert f"GLOVE_SHA_EXPECTED={GLOVE_SHA}" in ex
    assert '[[ -f "$GLOVE" ]]' in ex
    assert 'sha256sum "$GLOVE"' in ex
    assert '[[ "$GLOVE_SHA" == "$GLOVE_SHA_EXPECTED" ]]' in ex
    # the path actually reaches the python invocation
    assert '--glove-path "$GLOVE"' in ex
    # fallback is never enabled
    assert "--allow-glove-fallback" not in ex
    # the old cwd-relative guard is gone
    assert "test -f data/glove.6B.300d.txt" not in ex


def test_guard_precedes_python_in_both_launchers():
    for path in (JOB, AUDIT):
        t = script(path)
        g = t.index("GLOVE_SHA_EXPECTED=")
        first_py = min([i for i in
                        (t.find("srun python"), t.find("python - <<"))
                        if i != -1])
        assert g < first_py, f"{path}: glove check must precede any Python"


def test_preflight_constants_match_the_launchers():
    from scripts.naming_comprehension import preflight_data as pf
    assert pf.GLOVE_SHA == GLOVE_SHA
    assert pf.C_N == 27_981
    assert pf.C_SHA == \
        "10c2f06eda769bf620ca3dbb9889204e4431cac2bfe0d0f5dd37fa4df2bb9f50"
    assert pf.FULL_N == 29_571
    assert pf.LEXICON_SHA == \
        "ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66"


def test_preflight_rejects_missing_and_wrong_sha(tmp_path):
    import subprocess
    script_path = os.path.join(
        ROOT, "scripts/naming_comprehension/preflight_data.py")
    r = subprocess.run([sys.executable, script_path, "--glove-path",
                        str(tmp_path / "absent.txt"), "--quick"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 1 and "not found" in r.stderr
    bad = tmp_path / "bad.txt"
    bad.write_text("not glove")
    r = subprocess.run([sys.executable, script_path, "--glove-path", str(bad),
                        "--quick"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 1 and "sha256" in r.stderr


# =============  audit reconstructs each checkpoint's own configuration  =====

def _audit(ckpt, out):
    from scripts.naming_comprehension.base123_error_audit import main as amain
    return amain(["--ckpt", ckpt, "--out-dir", out, "--max-words", "400",
                  "--dorsal-pool-size", "32", "--batch-size", "8",
                  "--glove-path", "tests/_no_such_glove_file.txt",
                  "--allow-glove-fallback", "--no-subset-hash-check"])


def _chain(out, rid, lrs):
    """A source, then a checkpoint under the given task LR policy."""
    assert main(BASE[:-6] + ["--seed", "19", "--out-dir", out,
                             "--run-id", "pre0", "--schedule",
                             "interleaved_123", "--max-steps", "12",
                             "--save-every", "12"]) == 0
    argv = BASE[:-6] + ["--seed", "19", "--out-dir", out, "--run-id", rid,
                        "--schedule", "interleaved_123", "--max-steps", "24",
                        "--save-every", "24",
                        "--resume", ck(out, "pre0", 12)]
    if lrs:
        argv += ["--lr-repetition", lrs[0], "--lr-naming", lrs[1],
                 "--lr-comprehension", lrs[2], "--phase-transition"]
    assert main(argv) == 0
    return ck(out, rid, 24)


@pytest.mark.parametrize("lrs,kind", [
    (("3e-5", "3e-5", "1e-4"), "task_specific"),     # C-HIGH family
    (("3e-5", "3e-5", "3e-5"), "task_specific"),     # ALL-3e-5 family
    (None, "two_stage_rep_cursor"),                  # historical family
])
def test_audit_reconstructs_the_checkpoint_lr_policy_exactly(tmp_path, lrs,
                                                             kind):
    """The failure of job 1859047: the audit rebuilt the trainer with the
    driver defaults, so a task-specific checkpoint was compared against the
    two-stage policy and correctly refused."""
    from scripts.naming_comprehension.base123_error_audit import build
    out = str(tmp_path / "runs")
    src = _chain(out, "r", lrs)
    stored = load(src)["lr_policy"]
    assert stored["kind"] == kind
    tr, ckd = build(src, "cpu", max_words=400, dorsal_pool_size=32,
                    batch_size=8,
                    glove_path="tests/_no_such_glove_file.txt",
                    allow_glove_fallback=True, require_subset_hash=False)
    assert dict(tr.lr_policy) == dict(stored), "policy must match exactly"
    # no artificial transition, and the guard was NOT relaxed
    assert tr.allow_phase_transition is False
    assert tr.phase_transitions == list(ckd.get("phase_transitions", []))


def test_audit_is_read_only_and_takes_no_optimizer_step(tmp_path):
    out = str(tmp_path / "runs")
    src = _chain(out, "r", ("3e-5", "3e-5", "1e-4"))
    before = load(src)["model_state_dict"]
    adir = str(tmp_path / "audit")
    assert _audit(src, adir) == 0
    after = load(src)["model_state_dict"]
    assert not [k for k in before if not torch.equal(before[k], after[k])], \
        "the source checkpoint file must be untouched"
    summary = json.load(open(os.path.join(adir, "error_summary.json")))
    assert summary["read_only_verified"] is True
    assert summary["lr_policy"]["comprehension"] == 1e-4
    # the audit module must never reach a training/update path
    import inspect
    from scripts.naming_comprehension import base123_error_audit as m
    # executable lines only: the module legitimately DISCUSSES these in its
    # comments, but must never execute them
    src_txt = "\n".join(l for l in inspect.getsource(m).splitlines()
                        if l.strip() and not l.lstrip().startswith("#"))
    for forbidden in ("train_step", "optim.step", ".backward(", "zero_grad",
                      "--phase-transition", "allow_phase_transition=True"):
        assert forbidden not in src_txt, forbidden


def test_audit_launcher_passes_no_phase_transition_and_verifies_glove():
    t = script(AUDIT)
    ex = "\n".join(l for l in t.splitlines()
                   if l.strip() and not l.lstrip().startswith("#"))
    assert "--phase-transition" not in ex
    assert "--allow-glove-fallback" not in ex
    assert '--glove-path "$GLOVE"' in ex
    assert "--lr-repetition" not in ex, \
        "the audit must read the policy from the checkpoint, not be told it"
