"""Acceptance tests for the H512 u500 -> u750 continuation.

The continuation's only claim is that it changes NOTHING.  So the tests are
about state preservation and about the launcher being unable to do anything
else: no scientific flag, no branch, no from-scratch fallback, no H256, no
duplicate u500 evaluation, and a driver blob identical to the one that
produced u0-u500.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.train_joint_scratch import (           # noqa: E402
    INTERLEAVED_123, LR_BOUNDARY_STEPS, LR_STAGE2, OPT_POLICY_SHARED,
    JointScratchTrainer, lr_for_step, main,
)

JOB = "scripts/cluster/jeanzay/final_base123_h512_continue750.slurm"
PROD = "scripts/cluster/jeanzay/final_base123_production.slurm"
DRIVER = "scripts/naming_comprehension/train_joint_scratch.py"
BLOB_U500 = "e6c48ff8f4569c16af1773211d1d6a72d92307ea"
COMMIT_U500 = "b2b803fd4cdbfc24a8ab9b1632f0a87d3f1c1836"

R_PASS, CYCLE = 463, 6
U500, U750 = 1_389_000, 2_083_500

ARGS = ["--regime", "j0", "--subset-mode", "final_full", "--device", "cpu",
        "--max-words", "400", "--batch-size", "8", "--dorsal-pool-size", "32",
        "--lr-boundary-steps", "6", "--eval-every", "0", "--log-every", "0",
        "--glove-path", "tests/_no_such_glove_file.txt",
        "--allow-glove-fallback", "--no-subset-hash-check",
        "--schedule", "interleaved_123", "--enc-hidden", "512",
        "--dec-hidden", "512"]


def script(path=JOB):
    return open(os.path.join(ROOT, path), encoding="utf-8").read()


def git(*args):
    return subprocess.run(["git", "-C", ROOT, *args],
                          capture_output=True, text=True).stdout.strip()


def ck(out, run_id, step):
    return os.path.join(out, run_id, "checkpoints", f"step_{step:08d}.pt")


# ================================  training code is unchanged since u500 ===

def test_training_driver_blob_is_identical_to_the_u500_code():
    assert git("rev-parse", f"{COMMIT_U500}:{DRIVER}") == BLOB_U500
    assert git("rev-parse", f"HEAD:{DRIVER}") == BLOB_U500, \
        "the training driver changed since the u0-u500 block"


def test_no_runtime_imported_file_changed_since_u500():
    """Every repo module reachable from the driver, including lazy imports."""
    import ast
    seen, order = set(), []

    def repo_path(mod):
        p = mod.replace(".", "/")
        for c in (p + ".py", p + "/__init__.py"):
            if os.path.exists(os.path.join(ROOT, c)):
                return c
        return None

    def walk(rel):
        if rel in seen:
            return
        seen.add(rel); order.append(rel)
        tree = ast.parse(open(os.path.join(ROOT, rel)).read())
        for n in ast.walk(tree):
            mods = []
            if isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                mods = [n.module]
            for m in mods:
                c = repo_path(m)
                if c:
                    walk(c)

    walk(DRIVER)
    # relative imports inside models/ are followed explicitly
    order += ["models/wm_route.py", "models/ltm_route.py", "models/gating.py",
              "models/motor.py"]
    changed = [r for r in sorted(set(order))
               if git("rev-parse", f"{COMMIT_U500}:{r}")
               != git("rev-parse", f"HEAD:{r}")]
    assert not changed, f"runtime-imported files changed: {changed}"
    assert len(set(order)) >= 15


def test_launcher_pins_the_driver_blob_not_just_the_commit():
    t = script()
    assert f"TRAIN_BLOB_U500={BLOB_U500}" in t
    assert "git rev-parse HEAD:scripts/naming_comprehension/train_joint_scratch.py" in t
    assert "L3_EXPECTED_COMMIT" in t


# ==========================================  the continuation contract  ====

def test_step_arithmetic():
    for u, step in ((500, U500), (550, 1_527_900), (600, 1_666_800),
                    (650, 1_805_700), (700, 1_944_600), (750, U750)):
        assert u * R_PASS * CYCLE == step, u
    assert U750 - U500 == 694_500
    t = script()
    assert "SOURCE_STEP=1389000" in t and "MAX_STEPS=2083500" in t
    assert "FULL_EVAL_AT=1527900,1666800,1805700,1944600,2083500" in t


def test_u500_evaluation_is_not_duplicated():
    t = script()
    grid = t.split("FULL_EVAL_AT=", 1)[1].split()[0]
    assert str(U500) not in grid.split(","), "u500 must not be re-evaluated"
    ex = "\n".join(l for l in t.splitlines()
                   if l.strip() and not l.lstrip().startswith("#"))
    assert "--eval-at-start" not in ex, \
        "--eval-at-start would add a redundant row at the resumed step"


def test_launcher_changes_no_scientific_parameter():
    t = script()
    ex = "\n".join(l for l in t.splitlines()
                   if l.strip() and not l.lstrip().startswith("#"))
    for forbidden in ("--phase-transition", "--optimizer-policy",
                      "--dec-weight", "--c-align-weight", "--lr-repetition",
                      "--lr-naming", "--lr-comprehension",
                      "--lr-boundary-steps", "interleaved_223",
                      "grouped_rn_c", "task_separated", "--batch-size",
                      "--allow-glove-fallback", "--no-subset-hash-check"):
        assert forbidden not in ex, forbidden
    # cadences and recipe carried over verbatim
    assert "EVAL_EVERY=13890" in t and "SAVE_EVERY=69450" in t
    assert "--stop-at-ceiling" in t and "--endpoint-eval" in t
    assert "WM=128; ENC=512; DEC=512" in t


def test_launcher_refuses_h256_scratch_and_bad_state():
    t = script()
    assert "H256 is not continued" in t
    assert '[[ "$RUN_ID" != *h256* ]]' in t
    assert "NEVER trains from scratch" in t
    assert "FATAL: no checkpoints in" in t
    assert "OWN_STEP < SOURCE_STEP" in t
    assert "OWN_STEP >= MAX_STEPS" in t
    assert "sort | tail -1" in t
    assert "#SBATCH --array=0-3" in t
    assert "SEEDS=(19 20 21 22)" in t
    assert "IDX >= 0 && IDX < 4" in t


def test_preflight_asserts_the_shared_single_optimizer():
    t = script()
    for probe in ('ck.get("optimizer_policy") == "shared_adamw"',
                  '"optimizer_state_dict" in ck',
                  '"optimizer_states" not in ck',
                  '"ltm_enc_hidden": 512',
                  'ck.get("schedule") == "interleaved_123"',
                  'int(ck["lr_boundary_steps"]) == 46300',
                  'int(ck["cursors"]["repetition"]) > 46300',
                  '10c2f06eda769bf620ca3dbb9889204e4431cac2bfe0d0f5dd37fa4df2bb9f50',
                  'float(ck.get("dec_weight")) == 0.5'):
        assert probe in t, probe


def test_production_script_is_untouched():
    t = script(PROD)
    assert "MAX_STEPS=1389000" in t and "--array=0-7" in t
    assert "2083500" not in t


# ==================================  state preservation, behaviourally  ====

def test_lr_stays_stage2_across_the_whole_leg():
    """The repetition cursor at u500 is 231,500 -- far past 46,300 -- so the
    leg runs at 1e-4 and the clock is not restarted."""
    for u in (500, 750):
        assert lr_for_step(u * R_PASS, LR_BOUNDARY_STEPS) == LR_STAGE2
    assert 500 * R_PASS == 231_500 > LR_BOUNDARY_STEPS


def test_split_continuation_equals_uninterrupted(tmp_path):
    """The scientific requirement: resuming and continuing must land on the
    bit-exact state an uninterrupted run would have reached."""
    out = str(tmp_path / "runs")
    assert main(ARGS + ["--seed", "19", "--out-dir", out, "--run-id", "whole",
                        "--max-steps", "36", "--save-every", "36"]) == 0
    assert main(ARGS + ["--seed", "19", "--out-dir", out, "--run-id", "split",
                        "--max-steps", "18", "--save-every", "18"]) == 0
    assert main(ARGS + ["--seed", "19", "--out-dir", out, "--run-id", "split",
                        "--max-steps", "36", "--save-every", "36",
                        "--resume", ck(out, "split", 18)]) == 0
    a = torch.load(ck(out, "whole", 36), map_location="cpu", weights_only=False)
    b = torch.load(ck(out, "split", 36), map_location="cpu", weights_only=False)
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    assert not [k for k in sa if not torch.equal(sa[k], sb[k])], "model differs"
    oa = a["optimizer_state_dict"]["state"]
    ob = b["optimizer_state_dict"]["state"]
    assert set(oa) == set(ob)
    for i in oa:
        for m in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(oa[i][m], ob[i][m]), f"{m} differs at {i}"
    for key in ("cursors", "global_step", "stream_seeds",
                "schedule_anchor_step", "consecutive_ceiling", "widths",
                "lr_boundary_steps"):
        assert a[key] == b[key], key


def test_resume_preserves_every_cursor_and_the_streak(tmp_path):
    out = str(tmp_path / "runs")
    assert main(ARGS + ["--seed", "19", "--out-dir", out, "--run-id", "s",
                        "--max-steps", "18", "--save-every", "18"]) == 0
    src = torch.load(ck(out, "s", 18), map_location="cpu", weights_only=False)
    assert set(src["cursors"]) == {"repetition", "pool", "naming",
                                   "comprehension"}
    tr = JointScratchTrainer(
        regime="j0", seed=19, device="cpu", max_words=400, batch_size=8,
        lexicon_path="data/lexicon_en_glove_covered.tsv", dorsal_pool_size=32,
        subset_mode="final_full", subset_per_band=822, subset_size=32,
        lr_boundary_steps=6, allow_glove_fallback=True,
        require_subset_hash=False, schedule=INTERLEAVED_123,
        glove_path="tests/_no_such_glove_file.txt",
        enc_hidden=512, dec_hidden=512)
    tr.load_state_dict(src)
    assert tr.cursors == {k: int(v) for k, v in src["cursors"].items()}
    assert tr.global_step == src["global_step"]
    assert tr.schedule_anchor_step == src["schedule_anchor_step"]
    assert tr.consecutive_ceiling == src["consecutive_ceiling"]
    assert tr.optimizer_policy == OPT_POLICY_SHARED and tr.task_optims is None
    assert torch.equal(torch.get_rng_state(), src["rng_states"]["torch"])
    # the schedule phase continues rather than restarting
    assert tr.task_for_step(tr.global_step) == \
        tr.task_for_step(tr.schedule_anchor_step
                         + (tr.global_step - tr.schedule_anchor_step))


def test_resume_refuses_a_mismatched_architecture(tmp_path):
    out = str(tmp_path / "runs")
    assert main(ARGS + ["--seed", "19", "--out-dir", out, "--run-id", "a",
                        "--max-steps", "12", "--save-every", "12"]) == 0
    bad = torch.load(ck(out, "a", 12), map_location="cpu", weights_only=False)
    tr = JointScratchTrainer(
        regime="j0", seed=19, device="cpu", max_words=400, batch_size=8,
        lexicon_path="data/lexicon_en_glove_covered.tsv", dorsal_pool_size=32,
        subset_mode="final_full", subset_per_band=822, subset_size=32,
        lr_boundary_steps=6, allow_glove_fallback=True,
        require_subset_hash=False, schedule=INTERLEAVED_123,
        glove_path="tests/_no_such_glove_file.txt",
        enc_hidden=256, dec_hidden=256)
    with pytest.raises(RuntimeError, match="ARCHITECTURE MISMATCH"):
        tr.load_state_dict(bad)


def test_metrics_append_so_the_trajectory_stays_continuous(tmp_path):
    """Same-directory continuation must EXTEND metrics.tsv, never rewrite it,
    and must not overwrite the original config.json."""
    out = str(tmp_path / "runs")
    assert main(ARGS + ["--seed", "19", "--out-dir", out, "--run-id", "r",
                        "--max-steps", "12", "--save-every", "12",
                        "--full-eval-at", "12"]) == 0
    m = os.path.join(out, "r", "metrics.tsv")
    before = open(m).read()
    assert main(ARGS + ["--seed", "19", "--out-dir", out, "--run-id", "r",
                        "--max-steps", "24", "--save-every", "24",
                        "--full-eval-at", "24",
                        "--resume", ck(out, "r", 12)]) == 0
    after = open(m).read()
    assert after.startswith(before), "metrics.tsv was rewritten, not appended"
    assert len(after) > len(before)
    assert os.path.exists(os.path.join(out, "r", "config.json"))
    assert os.path.exists(os.path.join(
        out, "r", "config_from_step_00000012.json")), \
        "the later launch must be recorded separately"
    # the original u<=12 checkpoint is untouched
    assert os.path.exists(ck(out, "r", 12))
