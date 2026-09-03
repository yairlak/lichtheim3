"""Acceptance tests for FINAL-5P: a NEW run branching from the R100 parent.

FINAL-5P asks who should receive the historical R100 learning-rate drop.  From
step 277,800 onward repetition enters maintenance (1e-4) while naming and
comprehension KEEP the 1e-3 they already had, using the FINAL-4 task-LR and
phase-transition machinery unchanged.

Because the parent run already contains a different post-R100 history, the
branch must live in its OWN run directory: its own metrics/losses/checkpoints,
explicit ancestry, and the parent left byte-identical.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.train_joint_scratch import (          # noqa: E402
    LR_POLICY_TASK, LR_POLICY_TWO_STAGE, MACRO_CYCLE_STEPS, RATIO_123, main,
)

FINAL5P_LRS = {"repetition": 1e-4, "naming": 1e-3, "comprehension": 1e-3}

SMOKE = ["--regime", "j0", "--seed", "22", "--subset-mode", "final_full",
         "--schedule", "interleaved_123", "--device", "cpu",
         "--max-words", "400", "--batch-size", "8", "--dorsal-pool-size", "32",
         "--lr-boundary-steps", "6", "--eval-every", "0", "--log-every", "1",
         "--glove-path", "tests/_no_such_glove_file.txt",
         "--allow-glove-fallback", "--no-subset-hash-check"]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def launch(out, run_id, steps, *, resume=None, task_lrs=False,
           phase=False, full_eval=None, save_every=None):
    argv = SMOKE + ["--out-dir", out, "--run-id", run_id,
                    "--max-steps", str(steps),
                    "--save-every", str(save_every or steps)]
    if task_lrs:
        argv += ["--lr-repetition", "1e-4", "--lr-naming", "1e-3",
                 "--lr-comprehension", "1e-3"]
    if phase:
        argv += ["--phase-transition"]
    if full_eval:
        argv += ["--full-eval-at", full_eval]
    if resume:
        argv += ["--resume", resume]
    assert main(argv) == 0


def ckpt(out, run_id, step):
    return os.path.join(out, run_id, "checkpoints", f"step_{step:08d}.pt")


@pytest.fixture
def parent(tmp_path):
    """A parent run with history BEYOND the branch point, like FINAL-3."""
    out = str(tmp_path / "runs")
    launch(out, "parent", 6, save_every=6, full_eval="6")          # R100 analogue
    launch(out, "parent", 18, save_every=6, full_eval="18",        # its own later
           resume=ckpt(out, "parent", 6))                          # branch (1e-4)
    return out


# ======================================================  1, 8. accounting  ==

def test_final5p_milestone_accounting():
    r_pass, c_pass = 463, 438
    r, n, c = RATIO_123
    expected = {                    # R exp -> (cycles, step, N exp, C exp)
        100: (46_300, 277_800, 200.0, 317.1233),      # source
        110: (50_930, 305_580, 220.0, 348.8356),      # M5P-1
        120: (55_560, 333_360, 240.0, 380.5479),      # M5P-2
        130: (60_190, 361_140, 260.0, 412.2603),      # M5P-3
    }
    for r_exp, (cycles, step, n_exp, c_exp) in expected.items():
        assert r_exp * r_pass == cycles
        assert cycles * MACRO_CYCLE_STEPS == step
        assert cycles * n / r_pass == pytest.approx(n_exp)
        assert cycles * c / c_pass == pytest.approx(c_exp, abs=1e-3)
    assert expected[130][1] - expected[100][1] == 83_340


# ==============================  2-3. transition and optimizer preservation ==

def test_branch_declares_the_r100_transition_and_keeps_moments(parent, tmp_path):
    out = parent
    src = ckpt(out, "parent", 6)
    before = torch.load(src, map_location="cpu", weights_only=False)

    # Budget == the branch point: the transition is declared and the state is
    # written back with ZERO new optimizer steps, which is exactly the moment
    # at which the moments must be untouched.
    launch(out, "branch", 6, resume=src, task_lrs=True, phase=True,
           save_every=6)

    prov = json.load(open(os.path.join(out, "branch", "provenance.json")))
    assert prov["lr_policy"]["kind"] == LR_POLICY_TASK
    assert prov["lr_policy"]["repetition"] == 1e-4
    assert prov["lr_policy"]["naming"] == 1e-3
    assert prov["lr_policy"]["comprehension"] == 1e-3
    tr = prov["phase_transitions"][0]
    assert tr["transition_step"] == 6
    assert tr["old_lr_policy"]["kind"] == LR_POLICY_TWO_STAGE
    assert tr["new_lr_policy"]["kind"] == LR_POLICY_TASK

    # the first post-transition checkpoint carries the SAME moments as the
    # parent state it started from
    after = torch.load(ckpt(out, "branch", 6), map_location="cpu",
                       weights_only=False)
    sa = before["optimizer_state_dict"]["state"]
    sb = after["optimizer_state_dict"]["state"]
    assert set(sa) == set(sb) and sa
    for pid in sa:
        for k, va in sa[pid].items():
            vb = sb[pid][k]
            assert (torch.equal(va, vb) if torch.is_tensor(va) else va == vb), k
    assert after["cursors"] == before["cursors"]
    assert after["global_step"] == before["global_step"] == 6
    assert torch.equal(before["rng_states"]["torch"], after["rng_states"]["torch"])


# ==========================================  4-5. branch isolation  =========

def test_branch_does_not_touch_the_parent_run(parent, tmp_path):
    out = parent
    pdir = os.path.join(out, "parent")
    src = ckpt(out, "parent", 6)
    fingerprint = {f: sha(os.path.join(pdir, f))
                   for f in ("metrics.tsv", "config.json", "provenance.json")}
    fingerprint["losses"] = sha(os.path.join(pdir, "logs", "losses.tsv"))
    ckpt_fp = {os.path.basename(p): sha(p) for p in
               [os.path.join(pdir, "checkpoints", n)
                for n in os.listdir(os.path.join(pdir, "checkpoints"))]}

    launch(out, "branch", 12, resume=src, task_lrs=True, phase=True,
           save_every=6)

    assert {f: sha(os.path.join(pdir, f))
            for f in ("metrics.tsv", "config.json", "provenance.json")} \
        | {"losses": sha(os.path.join(pdir, "logs", "losses.tsv"))} == fingerprint, \
        "the parent run's logs were modified"
    assert {os.path.basename(p): sha(p) for p in
            [os.path.join(pdir, "checkpoints", n)
             for n in os.listdir(os.path.join(pdir, "checkpoints"))]} == ckpt_fp, \
        "a parent checkpoint was modified"

    # the branch owns an independent history starting at the branch point
    import csv
    brows = list(csv.DictReader(open(os.path.join(out, "branch", "metrics.tsv")),
                                delimiter="\t"))
    steps = [int(r["step"]) for r in brows]
    assert steps and min(steps) > 6, "branch logs must not replay parent history"
    prows = list(csv.DictReader(open(os.path.join(pdir, "metrics.tsv")),
                                delimiter="\t"))
    assert 18 in [int(r["step"]) for r in prows], "parent history still intact"


def test_branch_records_its_ancestry(parent):
    out = parent
    launch(out, "branch", 12, resume=ckpt(out, "parent", 6), task_lrs=True,
           phase=True, save_every=6)
    prov = json.load(open(os.path.join(out, "branch", "provenance.json")))
    anc = prov["ancestry"]
    assert anc["parent_run_id"] == "parent" and anc["is_branch"] is True
    assert anc["parent_global_step"] == 6
    assert anc["parent_checkpoint"].endswith("step_00000006.pt")
    assert anc["parent_lr_policy"]["kind"] == LR_POLICY_TWO_STAGE
    assert "optimizer moments" in anc["inherited"]
    # the parent's own metric row at the branch point travels with the branch
    assert anc.get("parent_metrics_row", {}).get("step") == "6"


def test_new_branch_writes_primary_metadata_not_only_suffixed(parent):
    """Analysis tools read config.json; a branch's first launch must have it."""
    out = parent
    launch(out, "branch", 12, resume=ckpt(out, "parent", 6), task_lrs=True,
           phase=True, save_every=6)
    bdir = os.path.join(out, "branch")
    assert os.path.exists(os.path.join(bdir, "config.json"))
    assert os.path.exists(os.path.join(bdir, "provenance.json"))
    assert not any(f.startswith("config_from_step_") for f in os.listdir(bdir))
    # a continuation of the SAME run still keeps the original and adds suffixed
    launch(out, "branch", 18, resume=ckpt(out, "branch", 12), task_lrs=True,
           save_every=6)
    assert os.path.exists(os.path.join(bdir, "config_from_step_00000012.json"))
    assert json.load(open(os.path.join(bdir, "config.json")))["total_steps"] == 12


# =============================================  6-7. resume and dispatch  ===

def test_mid_branch_resume_is_bitwise_and_needs_no_second_declaration(parent,
                                                                     tmp_path):
    out = parent
    src = ckpt(out, "parent", 6)
    launch(out, "whole", 18, resume=src, task_lrs=True, phase=True,
           save_every=18)
    launch(out, "split", 11, resume=src, task_lrs=True, phase=True,
           save_every=11)
    # requeue: no --phase-transition, policy already task_specific
    launch(out, "split", 18, resume=ckpt(out, "split", 11), task_lrs=True,
           save_every=18)

    a = torch.load(ckpt(out, "whole", 18), map_location="cpu", weights_only=False)
    b = torch.load(ckpt(out, "split", 18), map_location="cpu", weights_only=False)
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    assert not [k for k in sa if not torch.equal(sa[k], sb[k])]
    oa, ob = a["optimizer_state_dict"]["state"], b["optimizer_state_dict"]["state"]
    assert not [1 for k in oa for kk, v in oa[k].items()
                if torch.is_tensor(v) and not torch.equal(v, ob[k][kk])]
    assert a["cursors"] == b["cursors"]
    assert len(b["phase_transitions"]) == 1, "requeue re-declared the phase"


def test_losses_tsv_shows_the_per_task_rates(parent):
    import csv
    out = parent
    launch(out, "branch", 12, resume=ckpt(out, "parent", 6), task_lrs=True,
           phase=True, save_every=6)
    rows = list(csv.DictReader(
        open(os.path.join(out, "branch", "logs", "losses.tsv")), delimiter="\t"))
    assert rows
    for r in rows:
        assert float(r["lr"]) == pytest.approx(FINAL5P_LRS[r["task"]]), r["task"]
    assert {r["task"] for r in rows} <= {"repetition", "naming", "comprehension"}


# =====================================================  9-12. job contract ==

def test_final5p_slurm_job_contract():
    path = os.path.join(ROOT, "scripts", "cluster", "jeanzay",
                        "final5p_r100_semantic1e3.slurm")
    text = open(path, encoding="utf-8").read()
    assert "EPOCHS=130" in text
    assert "FULL_EVAL_AT=305580,333360,361140" in text
    assert "FINAL_STEP=361140" in text and "PARENT_STEP=277800" in text
    assert "ETA_R=1e-4" in text and "ETA_N=1e-3" in text and "ETA_C=1e-3" in text
    assert 'RUN_ID="final5p_r100_semantic1e3_seed${SEED}_${SUBSET_MODE}"' in text
    assert 'PARENT_RUN_ID="final3p_i123_seed${SEED}_${SUBSET_MODE}"' in text
    assert "--phase-transition" in text
    # refuses without the parent, never starts fresh, never collides
    assert 'FATAL: parent checkpoint' in text
    assert '[[ "$RUN_ID" != "$PARENT_RUN_ID" ]]' in text
    # first launch vs requeue: requeue must use the branch's OWN checkpoint
    assert 'OWN_LATEST=' in text and 'RESUME_FROM="$OWN_LATEST"' in text
    assert 'RESUME_FROM="$PARENT_CKPT"' in text
    for forbidden in ("--c-align-weight", "--lr-boundary-steps",
                      "--batch-size", "--allow-glove-fallback",
                      "--no-subset-hash-check"):
        assert forbidden not in text, forbidden


def test_old_runs_without_task_lr_are_unaffected(parent):
    """Backward compatibility: a plain continuation of the parent still uses
    the two-stage policy and needs no phase declaration."""
    out = parent
    launch(out, "parent", 24, resume=ckpt(out, "parent", 18), save_every=24)
    cfg = json.load(open(os.path.join(
        out, "parent", "config_from_step_00000018.json")))
    assert cfg["lr_policy"]["kind"] == LR_POLICY_TWO_STAGE
    assert "REPETITION cursor" in cfg["lr_convention"]
