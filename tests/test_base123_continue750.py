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
STOPPING_ONLY_IDENTS = ("consecutive_ceiling", "last_ceiling_step",
                        "ceiling_consecutive_required", "at_ceiling",
                        "CEILING_CONSECUTIVE_REQUIRED", "required",
                        "ceiling_reached")
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

def test_driver_diff_since_u500_is_stopping_control_only():
    """The driver DOES differ from the u0-u500 blob -- the ceiling rule moved
    from a hard-coded 2 to a CLI value defaulting to 5, with distinct-step
    accounting.  Every changed line must belong to stopping/evaluation
    control; no optimizer, loss, sampler, schedule, LR or data line may move."""
    assert git("rev-parse", f"{COMMIT_U500}:{DRIVER}") == BLOB_U500
    # against the WORKING TREE, so the check covers what will actually run
    # whether or not the change is committed yet
    diff = git("diff", COMMIT_U500, "--", DRIVER)
    changed = [l[1:].strip() for l in diff.splitlines()
               if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    code = [l for l in changed
            if l and not l.startswith("#") and not l.startswith('"')]
    assert code, "expected a real change"
    forbidden = ("optim.step", "backward", "zero_grad", "clip_grad",
                 "lr =", "LAMBDA_C", "LAMBDA_N", "TAU", "weights",
                 "CounterStream", "logfreq", "cursors[", "macro_cycle",
                 "AdamW", "weight_decay", "total_loss", "retrieval_loss",
                 "naming_objective", "LR_STAGE", "LR_BOUNDARY")
    for line in code:
        if any(f in line for f in forbidden):
            assert any(i in line for i in STOPPING_ONLY_IDENTS), \
                f"training-relevant line changed: {line}"


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
    expected = git("hash-object", os.path.join(ROOT, DRIVER))
    assert f"TRAIN_BLOB_EXPECTED={expected}" in t, \
        "the launcher's blob pin is stale"
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


# =========================  the 5-consecutive-milestone ceiling rule  ======

class _Row(dict):
    """A metrics row at (or below) ceiling."""
    def __init__(self, ceiling=True):
        v = 1.0 if ceiling else 0.999999
        super().__init__(full_rep_full=v, full_rep_freear=v,
                         full_naming_exact=v, full_comp_top1=v)


def _streak(sequence, required=5, start=0, last_step=-1, steps=None):
    """Replay the driver's streak accounting over a sequence of milestone
    outcomes, mirroring main()'s in-loop branch exactly."""
    from scripts.naming_comprehension.train_joint_scratch import at_ceiling
    streak, last = start, last_step
    steps = steps or list(range(1, len(sequence) + 1))
    stopped_at = None
    for step, ok in zip(steps, sequence):
        if step <= last:
            continue                       # already counted; no increment, no reset
        last = step
        streak = streak + 1 if at_ceiling(_Row(ok)) else 0
        if streak >= required and stopped_at is None:
            stopped_at = step
    return streak, last, stopped_at


def test_requirement_is_five():
    from scripts.naming_comprehension.train_joint_scratch import (
        CEILING_CONSECUTIVE_REQUIRED)
    assert CEILING_CONSECUTIVE_REQUIRED == 5


def test_streak_of_one_to_four_does_not_stop():
    for n in (1, 2, 3, 4):
        streak, _, stopped = _streak([True] * n)
        assert streak == n and stopped is None, n


def test_fifth_consecutive_distinct_milestone_stops():
    streak, _, stopped = _streak([True] * 5, steps=[10, 20, 30, 40, 50])
    assert streak == 5 and stopped == 50


def test_a_single_shortfall_resets_four_to_zero():
    streak, _, stopped = _streak([True, True, True, True, False])
    assert streak == 0 and stopped is None
    # and the count restarts from scratch afterwards
    streak, _, stopped = _streak([True] * 4 + [False] + [True] * 5,
                                 steps=list(range(1, 11)))
    assert stopped == 10 and streak == 5


def test_resume_with_streak_three_continues_at_four_and_stops_at_five():
    streak, _, stopped = _streak([True, True], start=3, last_step=30,
                                 steps=[40, 50])
    assert streak == 5 and stopped == 50
    # one more evaluation would already have stopped at the 5th
    streak, _, stopped = _streak([True], start=3, last_step=30, steps=[40])
    assert streak == 4 and stopped is None


def test_the_same_global_step_cannot_be_counted_twice():
    # step 40 re-evaluated after a requeue: neither increments nor resets
    streak, last, stopped = _streak([True, True], start=3, last_step=40,
                                    steps=[40, 50])
    assert streak == 4 and last == 50 and stopped is None
    # and a repeated FAILING step cannot wipe an earned streak either
    streak, _, _ = _streak([False], start=4, last_step=40, steps=[40])
    assert streak == 4


def test_streak_state_round_trips_through_a_checkpoint(tmp_path):
    out = str(tmp_path / "runs")
    assert main(ARGS + ["--seed", "19", "--out-dir", out, "--run-id", "s",
                        "--max-steps", "12", "--save-every", "12"]) == 0
    src = torch.load(ck(out, "s", 12), map_location="cpu", weights_only=False)
    assert "last_ceiling_step" in src and "consecutive_ceiling" in src
    tr = JointScratchTrainer(
        regime="j0", seed=19, device="cpu", max_words=400, batch_size=8,
        lexicon_path="data/lexicon_en_glove_covered.tsv", dorsal_pool_size=32,
        subset_mode="final_full", subset_per_band=822, subset_size=32,
        lr_boundary_steps=6, allow_glove_fallback=True,
        require_subset_hash=False, schedule=INTERLEAVED_123,
        glove_path="tests/_no_such_glove_file.txt",
        enc_hidden=512, dec_hidden=512)
    tr.consecutive_ceiling, tr.last_ceiling_step = 3, 40
    sd = tr.state_dict()
    assert sd["consecutive_ceiling"] == 3 and sd["last_ceiling_step"] == 40
    tr2 = JointScratchTrainer(
        regime="j0", seed=19, device="cpu", max_words=400, batch_size=8,
        lexicon_path="data/lexicon_en_glove_covered.tsv", dorsal_pool_size=32,
        subset_mode="final_full", subset_per_band=822, subset_size=32,
        lr_boundary_steps=6, allow_glove_fallback=True,
        require_subset_hash=False, schedule=INTERLEAVED_123,
        glove_path="tests/_no_such_glove_file.txt",
        enc_hidden=512, dec_hidden=512)
    tr2.load_state_dict(sd)
    assert tr2.consecutive_ceiling == 3 and tr2.last_ceiling_step == 40
    # a pre-existing checkpoint without the field must still load
    old = dict(sd); old.pop("last_ceiling_step")
    tr2.load_state_dict(old)
    assert tr2.last_ceiling_step == -1


def test_endpoint_eval_cannot_increment_the_streak():
    """The endpoint block sits OUTSIDE the milestone branch, so the duplicate
    final evaluation can never move the streak."""
    import inspect
    from scripts.naming_comprehension import train_joint_scratch as m
    src = inspect.getsource(m.main)
    endpoint = src.split("if last_eval != (trainer.global_step, True)", 1)[1]
    for ident in ("consecutive_ceiling", "last_ceiling_step",
                  "ceiling_reached"):
        assert ident not in endpoint, \
            f"the endpoint block touches {ident}"


def test_milestone_yields_exactly_one_row_without_endpoint_eval(tmp_path):
    """u750 is in FULL_EVAL_AT, so no --endpoint-eval is needed and none must
    be used: the duplicate row is what produced two step-1,389,000 rows."""
    import csv
    out = str(tmp_path / "runs")
    assert main(ARGS + ["--seed", "19", "--out-dir", out, "--run-id", "one",
                        "--max-steps", "24", "--save-every", "24",
                        "--full-eval-at", "12,24"]) == 0
    with open(os.path.join(out, "one", "metrics.tsv")) as f:
        steps = [r["step"] for r in csv.DictReader(f, delimiter="\t")]
    assert steps == ["12", "24"], steps
    assert steps.count("24") == 1, "duplicate final row"

    assert main(ARGS + ["--seed", "19", "--out-dir", out, "--run-id", "dup",
                        "--max-steps", "24", "--save-every", "24",
                        "--full-eval-at", "12,24", "--endpoint-eval"]) == 0
    with open(os.path.join(out, "dup", "metrics.tsv")) as f:
        steps = [r["step"] for r in csv.DictReader(f, delimiter="\t")]
    assert steps.count("24") == 2, "the duplicate behaviour should be reproduced"


def test_launcher_drops_endpoint_eval_and_requests_five():
    t = script()
    ex = "\n".join(l for l in t.splitlines()
                   if l.strip() and not l.lstrip().startswith("#"))
    assert "--endpoint-eval" not in ex, "the duplicate final row must be gone"
    assert "CEILING_REQUIRED=5" in t
    assert '--ceiling-consecutive-required "$CEILING_REQUIRED"' in ex
    grid = t.split("FULL_EVAL_AT=", 1)[1].split()[0].split(",")
    assert grid == ["1527900", "1666800", "1805700", "1944600", "2083500"]
    assert len(grid) == 5, "exactly five milestones in this leg"


def test_launcher_requires_l3_repo_with_no_fallback():
    t = script()
    ex = "\n".join(l for l in t.splitlines()
                   if l.strip() and not l.lstrip().startswith("#"))
    assert 'REPO=${L3_REPO:-' not in ex, "the default fallback must be gone"
    assert '[[ -n "${L3_REPO:-}" ]]' in ex
    assert 'REPO="$L3_REPO"' in ex
    assert "FATAL: L3_REPO must be exported" in t
    # and the commit pin is still enforced after it
    assert '[[ "$HEAD" == "$L3_EXPECTED_COMMIT" ]]' in ex
    assert 'FATAL: HEAD != L3_EXPECTED_COMMIT' in t


def test_launcher_l3_repo_guards_execute(tmp_path):
    """Run the launcher's guard prologue for real: no L3_REPO must abort
    before anything else, and a non-worktree path must abort too."""
    prologue = script().split("# ---- explicit array mapping", 1)[0]
    prologue = prologue.replace("module purge", ":").replace(
        "module load pytorch-gpu/py3/2.6.0", ":").replace(
        "module load git", ":")
    sh = tmp_path / "prologue.sh"
    sh.write_text(prologue)
    env = {k: v for k, v in os.environ.items() if k != "L3_REPO"}
    r = subprocess.run(["bash", str(sh)], capture_output=True, text=True,
                       env=env)
    assert r.returncode == 1
    assert "L3_REPO must be exported" in r.stdout + r.stderr

    env["L3_REPO"] = str(tmp_path)          # exists but is not a worktree
    r = subprocess.run(["bash", str(sh)], capture_output=True, text=True,
                       env=env)
    assert r.returncode == 1
    assert "is not a git worktree" in r.stdout + r.stderr
