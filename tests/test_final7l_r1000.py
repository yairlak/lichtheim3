"""Acceptance tests for the extended FINAL-7L overnight run, R220 -> R1000.

Scientifically this is the SAME predeclared from-scratch recipe carried
further in the same run directory: grouped RN|C banks restored bit-exactly
from the run's own R220 checkpoint, LR 1e-4, interleaved 1:2:3, every loss,
population, seed and architecture unchanged.  Relative to the R700 script the
only difference is the horizon and its two extra milestones; that script must
remain untouched and still valid.

Deliberately light on disk: the contract and accounting checks are static,
and the one behavioural check reuses a small driver fixture.
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

from scripts.naming_comprehension.compare_checkpoints import (          # noqa: E402
    flat_optimizer,
)
from scripts.naming_comprehension.train_joint_scratch import (          # noqa: E402
    LR_STAGE2, MACRO_CYCLE_STEPS, OPT_POLICY_GROUPED_RN_C, RATIO_123,
    lr_for_step, main,
)

R1000 = "scripts/cluster/jeanzay/final7l_continue_r220_to_r1000.slurm"
R700 = "scripts/cluster/jeanzay/final7l_continue_r220_to_r700.slurm"
GATE = "scripts/cluster/jeanzay/final7l_fromscratch_gate_r220.slurm"

SMOKE = ["--regime", "j0", "--seed", "22", "--subset-mode", "final_full",
         "--schedule", "interleaved_123", "--device", "cpu",
         "--max-words", "400", "--batch-size", "8", "--dorsal-pool-size", "32",
         "--lr-boundary-steps", "6", "--eval-every", "0", "--log-every", "0",
         "--glove-path", "tests/_no_such_glove_file.txt",
         "--allow-glove-fallback", "--no-subset-hash-check"]


def script(path):
    return open(os.path.join(ROOT, path), encoding="utf-8").read()


def srun_block(text):
    return "\n".join(chunk.split("\n\n", 1)[0]
                     for chunk in text.split("srun python")[1:])


def launch(out, run_id, steps, *, resume=None, grouped=False, phase=False,
           save_every=None):
    argv = SMOKE + ["--out-dir", out, "--run-id", run_id,
                    "--max-steps", str(steps),
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


# ============================================  accounting through R1000  ===

def test_exposure_accounting_through_r1000():
    r_pass, c_pass = 463, 438
    r, n, c = RATIO_123
    expected = {                     # R exp -> (cycles, step, N exp, C exp)
        220: (101_860, 611_160, 440.0, 697.6712),      # source
        300: (138_900, 833_400, 600.0, 951.3699),
        400: (185_200, 1_111_200, 800.0, 1268.4932),
        500: (231_500, 1_389_000, 1000.0, 1585.6164),
        600: (277_800, 1_666_800, 1200.0, 1902.7397),
        700: (324_100, 1_944_600, 1400.0, 2219.8630),
        800: (370_400, 2_222_400, 1600.0, 2536.9863),
        900: (416_700, 2_500_200, 1800.0, 2854.1096),
        1000: (463_000, 2_778_000, 2000.0, 3171.2329),
    }
    for r_exp, (cycles, step, n_exp, c_exp) in expected.items():
        assert r_exp * r_pass == cycles
        assert cycles * MACRO_CYCLE_STEPS == step
        assert cycles * n / r_pass == pytest.approx(n_exp)
        assert cycles * c / c_pass == pytest.approx(c_exp, abs=1e-3)
    assert expected[1000][1] - expected[220][1] == 2_166_840
    # every milestone lands on the unchanged cadences
    for step in (833_400, 1_111_200, 1_389_000, 1_666_800, 1_944_600,
                 2_222_400, 2_500_200, 2_778_000):
        assert step % 27_780 == 0


def test_lr_is_1e_4_throughout_the_extension():
    """R220 is far past the repetition-cursor boundary, so the untouched
    two-stage policy yields 1e-4 for every step of this leg."""
    for r_exp in (220, 500, 1000):
        assert lr_for_step(r_exp * 463, 46_300) == LR_STAGE2
    assert 220 * 463 > 46_300


# ==================================================  job contract  =========

def test_r1000_job_resumes_only_its_own_r220_endpoint():
    t = script(R1000)
    assert "GATE_STEP=611160" in t and "FINAL_STEP=2778000" in t
    assert 'GATE_CKPT="$CKPTS/step_$(printf ' in t
    assert "FATAL: gate endpoint" in t
    assert '--resume "$OWN_LATEST"' in t
    # its own run directory, never a branch, never the historical run
    assert 'RUN_ID="final7l_fromscratch_seed${SEED}_${SUBSET_MODE}"' in t
    assert "final3p" not in t
    # newest chosen by step number, not mtime
    assert 'sort | tail -1' in t


def test_r1000_job_declares_no_transition_and_no_recloning():
    t = script(R1000)
    assert "--phase-transition" not in srun_block(t)
    assert "PHASE_ARGS" not in t
    assert "clone" not in srun_block(t).lower()


def test_r1000_job_freezes_the_scientific_recipe():
    t = script(R1000)
    assert "EPOCHS=1000" in t
    assert ("FULL_EVAL_AT=833400,1111200,1389000,1666800,1944600,"
            "2222400,2500200,2778000") in t
    assert "GROUPED=grouped_rn_c_adamw" in t
    assert '--optimizer-policy "$GROUPED"' in t
    assert "SCHEDULE=interleaved_123" in t and "SEED=22" in t
    assert "SUBSET_MODE=final_full" in t
    # cadences identical to the gate and the R700 script
    assert "EVAL_EVERY=27780" in t and "SAVE_EVERY=27780" in t
    for forbidden in ("--lr-repetition", "--lr-naming", "--lr-comprehension",
                      "--lr-boundary-steps", "--c-align-weight",
                      "--batch-size", "--dorsal-pool-size",
                      "--allow-glove-fallback", "--no-subset-hash-check",
                      "--torch-deterministic"):
        assert forbidden not in t, forbidden


def test_r1000_job_is_requeue_safe_and_idempotent():
    t = script(R1000)
    assert "nothing to do: already reached the R1000 hard stop" in t
    assert "requeue: resuming the overnight run at step" in t
    assert "FATAL: newest checkpoint is step" in t
    assert "#SBATCH --requeue" in t
    assert "--time=20:00:00" in t


def test_r700_script_is_unchanged_and_still_valid():
    """The extension is a new script; the old horizon must still work."""
    t = script(R700)
    assert "EPOCHS=700" in t
    assert "FULL_EVAL_AT=833400,1111200,1389000,1666800,1944600" in t
    assert "FINAL_STEP=1944600" in t and "GATE_STEP=611160" in t
    assert 'RUN_ID="final7l_fromscratch_seed${SEED}_${SUBSET_MODE}"' in t
    assert "--phase-transition" not in srun_block(t)
    # the two scripts differ only in horizon-related settings
    assert "2222400" not in t and "2500200" not in t and "2778000" not in t


def test_gate_script_is_untouched_by_this_change():
    t = script(GATE)
    assert "GATE_STEP=611160" in t and "STAGE1_EPOCHS=100" in t
    assert "GATE_EPOCHS=220" in t and "--time=03:00:00" in t
    assert "STAGE2_FULL_EVAL=361140,444480,611160" in t


# ==========================================  behaviour on real artefacts  ==

def test_extension_style_resume_keeps_banks_and_adds_no_transition(tmp_path):
    """A grouped checkpoint resumed with a LARGER budget must restore both
    banks from their own saved states -- no transition, no re-clone."""
    out = str(tmp_path / "runs")
    launch(out, "run", 6, save_every=6)                                # shared
    launch(out, "run", 18, resume=ckpt(out, "run", 6), grouped=True,
           phase=True, save_every=6)                                   # grouped
    src = ckpt(out, "run", 18)
    before = flat_optimizer(torch.load(src, map_location="cpu",
                                       weights_only=False))
    diverged = [k for k, v in before["rn"].items()
                if torch.is_tensor(v)
                and not torch.equal(v, before["comprehension"][k])]
    assert diverged, "banks should have specialised before the extension"

    launch(out, "run", 30, resume=src, grouped=True, save_every=30)     # extend
    end = torch.load(ckpt(out, "run", 30), map_location="cpu",
                     weights_only=False)
    assert end["optimizer_policy"] == OPT_POLICY_GROUPED_RN_C
    assert len(end["phase_transitions"]) == 1, "a transition was re-declared"
    after = flat_optimizer(end)
    assert [k for k, v in after["rn"].items()
            if torch.is_tensor(v)
            and not torch.equal(v, after["comprehension"][k])], \
        "the banks were re-cloned from a shared state"
    prov = json.load(open(os.path.join(
        out, "run", "provenance_from_step_00000018.json")))
    assert prov["phase_transition_declared"] is False
    assert prov["ancestry"]["is_branch"] is False
