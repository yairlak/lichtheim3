"""Acceptance tests for FINAL-6F: continuing FINAL-6P from R160 to R220.

"6F" is an analysis label only -- scientifically this is the SAME FINAL-6P
regime carried further in the same run directory.  The three specialised
AdamW banks are restored bit-exactly from the run's own checkpoint: no phase
transition, no cloning from the shared R100 parent, no optimizer reset, no
change of run topology, no LR/loss/ratio/architecture change.

The deeper mechanism tests live in tests/test_optimizer_banks.py (bank
construction, non-aliasing, per-task isolation) and
tests/test_final6e_continuation.py (the R130 -> R160 continuation); this file
re-verifies the continuation contract at the R160 -> R220 budget and pins the
FINAL-6F accounting and job.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.train_joint_scratch import (          # noqa: E402
    LR_POLICY_TWO_STAGE, LR_TASKS, MACRO_CYCLE_STEPS, OPT_POLICY_SEPARATED,
    RATIO_123, main,
)

SMOKE = ["--regime", "j0", "--seed", "22", "--subset-mode", "final_full",
         "--schedule", "interleaved_123", "--device", "cpu",
         "--max-words", "400", "--batch-size", "8", "--dorsal-pool-size", "32",
         "--lr-boundary-steps", "6", "--eval-every", "0", "--log-every", "1",
         "--glove-path", "tests/_no_such_glove_file.txt",
         "--allow-glove-fallback", "--no-subset-hash-check"]


def launch(out, run_id, steps, *, resume=None, separated=False, phase=False,
           save_every=None, full_eval=None):
    argv = SMOKE + ["--out-dir", out, "--run-id", run_id,
                    "--max-steps", str(steps),
                    "--save-every", str(save_every or steps)]
    if separated:
        argv += ["--optimizer-policy", "task_separated_adamw"]
    if phase:
        argv += ["--phase-transition"]
    if full_eval:
        argv += ["--full-eval-at", full_eval]
    if resume:
        argv += ["--resume", resume]
    assert main(argv) == 0


def ckpt(out, run_id, step):
    return os.path.join(out, run_id, "checkpoints", f"step_{step:08d}.pt")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def bank_state(sd):
    """{task: {(param id, key): value}} from a saved separated checkpoint."""
    out = {}
    for t, st in sd["optimizer_states"].items():
        flat = {}
        for pid, entry in st["state"].items():
            for k, v in entry.items():
                flat[(pid, k)] = v
        out[t] = flat
    return out


def same_bank(a, b):
    if set(a) != set(b):
        return False
    for k, va in a.items():
        vb = b[k]
        ok = torch.equal(va, vb) if torch.is_tensor(va) else va == vb
        if not ok:
            return False
    return True


@pytest.fixture
def sixe(tmp_path):
    """A FINAL-6E analogue: shared -> separated, then already continued once,
    so the run has real history BEFORE the FINAL-6F extension starts."""
    out = str(tmp_path / "runs")
    launch(out, "parent", 6, save_every=6)                        # R100 analogue
    launch(out, "run", 18, resume=ckpt(out, "parent", 6), separated=True,
           phase=True, save_every=6, full_eval="18")              # -> R130
    launch(out, "run", 24, resume=ckpt(out, "run", 18), separated=True,
           save_every=6, full_eval="24")                          # -> R160
    return out


# ============================  1-4. exact three-bank resume, no transition  ==

def test_three_banks_restore_bitwise_from_the_runs_own_checkpoint(sixe):
    out = sixe
    src = ckpt(out, "run", 24)                       # the R160 analogue
    before = torch.load(src, map_location="cpu", weights_only=False)
    assert before["optimizer_policy"] == OPT_POLICY_SEPARATED
    assert set(before["optimizer_states"]) == set(LR_TASKS)

    # a zero-step relaunch writes the restored state straight back out, which
    # compares "restored" against "saved" without any training in between
    launch(out, "probe", 24, resume=src, separated=True, save_every=24)
    rt = torch.load(ckpt(out, "probe", 24), map_location="cpu",
                    weights_only=False)
    a, b = bank_state(before), bank_state(rt)
    for t in LR_TASKS:
        assert a[t] and same_bank(a[t], b[t]), f"bank {t} not restored bitwise"


def test_extension_declares_no_phase_transition(sixe):
    out = sixe
    src = ckpt(out, "run", 24)
    n_before = len(torch.load(src, map_location="cpu",
                              weights_only=False)["phase_transitions"])
    launch(out, "run", 30, resume=src, separated=True, save_every=30)
    after = torch.load(ckpt(out, "run", 30), map_location="cpu",
                       weights_only=False)
    assert len(after["phase_transitions"]) == n_before
    assert after["optimizer_policy"] == OPT_POLICY_SEPARATED
    assert after["lr_policy"]["kind"] == LR_POLICY_TWO_STAGE
    prov = json.load(open(os.path.join(
        out, "run", "provenance_from_step_00000024.json")))
    assert prov["phase_transition_declared"] is False
    assert prov["optimizer_policy"] == OPT_POLICY_SEPARATED
    assert prov["resumed_at_step"] == 24
    assert prov["budget_this_launch"]["total_steps"] == 30
    assert prov["ancestry"]["is_branch"] is False, "same run, not a branch"


def test_no_moment_cloning_banks_stay_specialised(sixe):
    """Re-cloning a shared state would make the banks identical again."""
    out = sixe
    src = ckpt(out, "run", 24)
    saved = bank_state(torch.load(src, map_location="cpu", weights_only=False))
    key = next(iter(saved["repetition"]))
    assert not torch.equal(saved["repetition"][key], saved["naming"][key]), \
        "fixture did not produce specialised banks"

    launch(out, "run", 30, resume=src, separated=True, save_every=30)
    after = bank_state(torch.load(ckpt(out, "run", 30), map_location="cpu",
                                  weights_only=False))
    assert not torch.equal(after["repetition"][key], after["naming"][key])
    assert not torch.equal(after["repetition"][key], after["comprehension"][key])
    assert not torch.equal(after["naming"][key], after["comprehension"][key])


def test_pointing_the_extension_at_the_shared_parent_still_refuses(sixe):
    """The R100 parent must never be resumable without an explicit
    declaration -- the extension must only ever use the run's own state."""
    out = sixe
    with pytest.raises(RuntimeError, match="PHASE TRANSITION"):
        main(SMOKE + ["--out-dir", out, "--run-id", "oops",
                      "--max-steps", "12", "--save-every", "12",
                      "--optimizer-policy", "task_separated_adamw",
                      "--resume", ckpt(out, "parent", 6)])


# =========================================  5-6. independence, exactness  ===

def test_bank_independence_retained_through_the_extension(sixe):
    out = sixe
    launch(out, "run", 36, resume=ckpt(out, "run", 24), separated=True,
           save_every=36)
    sd = torch.load(ckpt(out, "run", 36), map_location="cpu",
                    weights_only=False)["optimizer_states"]
    ptrs = [next(iter(sd[t]["state"].values()))["exp_avg"].data_ptr()
            for t in LR_TASKS]
    assert len(set(ptrs)) == 3, "banks alias after the extension"


def test_interrupted_extension_equals_uninterrupted(sixe):
    out = sixe
    src = ckpt(out, "run", 24)
    launch(out, "whole", 40, resume=src, separated=True, save_every=40)
    launch(out, "split", 31, resume=src, separated=True, save_every=31)
    launch(out, "split", 40, resume=ckpt(out, "split", 31), separated=True,
           save_every=40)

    a = torch.load(ckpt(out, "whole", 40), map_location="cpu", weights_only=False)
    b = torch.load(ckpt(out, "split", 40), map_location="cpu", weights_only=False)
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    assert not [k for k in sa if not torch.equal(sa[k], sb[k])]
    ba, bb = bank_state(a), bank_state(b)
    for t in LR_TASKS:
        assert same_bank(ba[t], bb[t]), f"bank {t} diverged after interruption"
    assert a["cursors"] == b["cursors"] and a["global_step"] == b["global_step"]


# =========================================  7-8. budget and job contract  ===

def test_final6f_milestone_accounting():
    r_pass, c_pass = 463, 438
    r, n, c = RATIO_123
    expected = {                    # R exp -> (cycles, step, N exp, C exp)
        160: (74_080, 444_480, 320.0, 507.3973),      # source (FINAL-6E end)
        180: (83_340, 500_040, 360.0, 570.8219),      # F1
        200: (92_600, 555_600, 400.0, 634.2466),      # F2
        220: (101_860, 611_160, 440.0, 697.6712),     # F3
    }
    for r_exp, (cycles, step, n_exp, c_exp) in expected.items():
        assert r_exp * r_pass == cycles
        assert cycles * MACRO_CYCLE_STEPS == step
        assert cycles * n / r_pass == pytest.approx(n_exp)
        assert cycles * c / c_pass == pytest.approx(c_exp, abs=1e-3)
    assert expected[220][1] - expected[160][1] == 166_680
    # the declared cadences land exactly on every milestone
    for step in (500_040, 555_600, 611_160):
        assert step % 13_890 == 0 and step % 27_780 == 0


def test_final6f_slurm_job_contract():
    path = os.path.join(ROOT, "scripts", "cluster", "jeanzay",
                        "final6f_r160_to_r220.slurm")
    text = open(path, encoding="utf-8").read()
    assert "EPOCHS=220" in text
    assert "FULL_EVAL_AT=500040,555600,611160" in text
    assert "R160_STEP=444480" in text and "FINAL_STEP=611160" in text
    # SAME run, same policy, resumes only its own newest checkpoint
    assert 'RUN_ID="final6p_r100_sepmoments_seed${SEED}_${SUBSET_MODE}"' in text
    assert "OPT_POLICY=task_separated_adamw" in text
    assert '--resume "$OWN_LATEST"' in text
    # never declares a transition, never references the shared parent run
    srun = text.split("srun python", 1)[1].split("\n\n", 1)[0]
    assert "--phase-transition" not in srun
    assert "PHASE_ARGS" not in text
    assert "final3p_i123_seed${SEED}_${SUBSET_MODE}" not in srun
    # no scientific override on the command line
    for forbidden in ("--lr-repetition", "--lr-naming", "--lr-comprehension",
                      "--c-align-weight", "--lr-boundary-steps",
                      "--batch-size", "--dorsal-pool-size",
                      "--allow-glove-fallback", "--no-subset-hash-check"):
        assert forbidden not in text, forbidden
    assert "--time=02:00:00" in text


# ==============================================  9. history preservation  ===

def test_earlier_history_and_checkpoints_are_retained(sixe):
    out = sixe
    run = os.path.join(out, "run")
    src = ckpt(out, "run", 24)
    src_sha, r130_sha = sha(src), sha(ckpt(out, "run", 18))
    before = open(os.path.join(run, "metrics.tsv")).read()

    launch(out, "run", 30, resume=src, separated=True, save_every=30,
           full_eval="30")

    # both earlier endpoints survive untouched
    assert sha(src) == src_sha and sha(ckpt(out, "run", 18)) == r130_sha
    # earlier rows preserved verbatim, new ones appended, exactly one header
    now = open(os.path.join(run, "metrics.tsv")).read()
    assert now.startswith(before)
    assert now.count("rep_epoch") == 1
    rows = list(csv.DictReader(open(os.path.join(run, "metrics.tsv")),
                               delimiter="\t"))
    steps = [int(r["step"]) for r in rows]
    assert 18 in steps and 24 in steps and 30 in steps
    # the run's first-launch records are never overwritten
    assert json.load(open(os.path.join(run, "config.json")))["total_steps"] == 18
    for s in ("00000018", "00000024"):
        assert os.path.exists(os.path.join(run, f"config_from_step_{s}.json"))


def test_metrics_schema_unchanged_so_appends_stay_aligned(sixe):
    out = sixe
    run = os.path.join(out, "run")
    header = open(os.path.join(run, "metrics.tsv")).readline().rstrip("\n").split("\t")
    launch(out, "run", 30, resume=ckpt(out, "run", 24), separated=True,
           save_every=30, full_eval="30")
    lines = open(os.path.join(run, "metrics.tsv")).read().strip().split("\n")
    assert lines[0].split("\t") == header
    for line in lines[1:]:
        assert len(line.split("\t")) == len(header), "row/header misalignment"
    for col in ("m_div_RN", "m_div_RC", "m_div_NC", "full_comp_top1",
                "full_comp_top5", "full_naming_exact", "full_rep_full",
                "rep_ltm", "comp_rank_median"):
        assert col in header, col
