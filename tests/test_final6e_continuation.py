"""Acceptance tests for FINAL-6E: continuing FINAL-6P from R130 to R160.

"6E" is an analysis label only -- scientifically this is the SAME FINAL-6P
regime carried further in the same run directory.  The three specialised
AdamW banks are restored bit-exactly from FINAL-6P's own checkpoint: no phase
transition, no cloning from the shared R100 parent, no optimizer reset, no
change of run topology.

These tests use a locally built separated-policy checkpoint as the stand-in
for the real R130 endpoint (which lives on Jean Zay); the resume path is
state-independent.
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
    """{task: {(param id, key): tensor}} from a saved separated checkpoint."""
    out = {}
    for t, st in sd["optimizer_states"].items():
        flat = {}
        for pid, entry in st["state"].items():
            for k, v in entry.items():
                flat[(pid, k)] = v
        out[t] = flat
    return out


@pytest.fixture
def sixp(tmp_path):
    """A FINAL-6P analogue: shared run to step 6, then separated to step 18."""
    out = str(tmp_path / "runs")
    launch(out, "parent", 6, save_every=6)                     # R100 analogue
    launch(out, "sixp", 18, resume=ckpt(out, "parent", 6), separated=True,
           phase=True, save_every=6, full_eval="18")           # R100 -> R130
    return out


# ===========================  1-4. resume restores banks, nothing declared  ==

def test_continuation_restores_three_banks_bitwise(sixp):
    """The extension must reload FINAL-6P's OWN bank states exactly."""
    out = sixp
    src = ckpt(out, "sixp", 18)
    before = torch.load(src, map_location="cpu", weights_only=False)
    assert before["optimizer_policy"] == OPT_POLICY_SEPARATED
    assert set(before["optimizer_states"]) == set(LR_TASKS)

    # continue in the SAME run, larger budget, NO --phase-transition
    launch(out, "sixp", 24, resume=src, separated=True, save_every=24)

    # a zero-step relaunch writes the state straight back out, which is the
    # cleanest way to compare "restored" against "saved"
    launch(out, "probe", 18, resume=src, separated=True, save_every=18)
    round_trip = torch.load(ckpt(out, "probe", 18), map_location="cpu",
                            weights_only=False)
    a, b = bank_state(before), bank_state(round_trip)
    for t in LR_TASKS:
        assert set(a[t]) == set(b[t]) and a[t]
        for key, va in a[t].items():
            vb = b[t][key]
            ok = torch.equal(va, vb) if torch.is_tensor(va) else va == vb
            assert ok, f"bank {t} entry {key} not restored bitwise"


def test_no_phase_transition_and_no_cloning_on_continuation(sixp):
    out = sixp
    src = ckpt(out, "sixp", 18)
    n_before = len(torch.load(src, map_location="cpu",
                              weights_only=False)["phase_transitions"])
    launch(out, "sixp", 24, resume=src, separated=True, save_every=24)
    after = torch.load(ckpt(out, "sixp", 24), map_location="cpu",
                       weights_only=False)
    assert len(after["phase_transitions"]) == n_before, "a transition was declared"
    assert after["optimizer_policy"] == OPT_POLICY_SEPARATED
    assert after["lr_policy"]["kind"] == LR_POLICY_TWO_STAGE
    prov = json.load(open(os.path.join(
        out, "sixp", "provenance_from_step_00000018.json")))
    assert prov["phase_transition_declared"] is False
    assert prov["optimizer_policy"] == OPT_POLICY_SEPARATED
    assert prov["resumed_at_step"] == 18


def test_banks_stay_distinct_ie_nothing_was_recloned(sixp):
    """If the shared parent state had been re-cloned, the banks would be
    identical again; they must remain the specialised ones."""
    out = sixp
    src = ckpt(out, "sixp", 18)
    saved = bank_state(torch.load(src, map_location="cpu", weights_only=False))
    key = next(iter(saved["repetition"]))
    assert not torch.equal(saved["repetition"][key], saved["naming"][key]), \
        "fixture did not produce specialised banks"

    launch(out, "sixp", 24, resume=src, separated=True, save_every=24)
    after = bank_state(torch.load(ckpt(out, "sixp", 24), map_location="cpu",
                                  weights_only=False))
    assert not torch.equal(after["repetition"][key], after["naming"][key])
    assert not torch.equal(after["repetition"][key], after["comprehension"][key])


def test_resuming_the_shared_parent_would_still_require_a_declaration(sixp):
    """Guard against ever pointing the extension at the R100 parent."""
    out = sixp
    with pytest.raises(RuntimeError, match="PHASE TRANSITION"):
        argv = SMOKE + ["--out-dir", out, "--run-id", "oops",
                        "--max-steps", "12", "--save-every", "12",
                        "--optimizer-policy", "task_separated_adamw",
                        "--resume", ckpt(out, "parent", 6)]
        main(argv)


# =====================================  5-6. independence and exactness  ====

def test_bank_independence_survives_the_continuation(sixp):
    out = sixp
    launch(out, "sixp", 30, resume=ckpt(out, "sixp", 18), separated=True,
           save_every=30)
    sd = torch.load(ckpt(out, "sixp", 30), map_location="cpu",
                    weights_only=False)["optimizer_states"]
    ptrs = []
    for t in LR_TASKS:
        entry = next(iter(sd[t]["state"].values()))
        ptrs.append(entry["exp_avg"].data_ptr())
    assert len(set(ptrs)) == 3, "banks alias after the continuation"


def test_interrupted_continuation_equals_uninterrupted(sixp, tmp_path):
    out = sixp
    src = ckpt(out, "sixp", 18)
    launch(out, "whole", 30, resume=src, separated=True, save_every=30)
    launch(out, "split", 23, resume=src, separated=True, save_every=23)
    launch(out, "split", 30, resume=ckpt(out, "split", 23), separated=True,
           save_every=30)

    a = torch.load(ckpt(out, "whole", 30), map_location="cpu", weights_only=False)
    b = torch.load(ckpt(out, "split", 30), map_location="cpu", weights_only=False)
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    assert not [k for k in sa if not torch.equal(sa[k], sb[k])]
    ba, bb = bank_state(a), bank_state(b)
    for t in LR_TASKS:
        for key, va in ba[t].items():
            vb = bb[t][key]
            ok = torch.equal(va, vb) if torch.is_tensor(va) else va == vb
            assert ok, f"bank {t} diverged after an interrupted continuation"
    assert a["cursors"] == b["cursors"] and a["global_step"] == b["global_step"]


# ==========================================  7-8. budget and job contract  ==

def test_final6e_milestone_accounting():
    r_pass, c_pass = 463, 438
    r, n, c = RATIO_123
    expected = {                    # R exp -> (cycles, step, N exp, C exp)
        130: (60_190, 361_140, 260.0, 412.2603),      # source (FINAL-6P end)
        140: (64_820, 388_920, 280.0, 443.9726),      # E1
        150: (69_450, 416_700, 300.0, 475.6849),      # E2
        160: (74_080, 444_480, 320.0, 507.3973),      # E3
    }
    for r_exp, (cycles, step, n_exp, c_exp) in expected.items():
        assert r_exp * r_pass == cycles
        assert cycles * MACRO_CYCLE_STEPS == step
        assert cycles * n / r_pass == pytest.approx(n_exp)
        assert cycles * c / c_pass == pytest.approx(c_exp, abs=1e-3)
    assert expected[160][1] - expected[130][1] == 83_340


def test_final6e_slurm_job_contract():
    path = os.path.join(ROOT, "scripts", "cluster", "jeanzay",
                        "final6e_r130_to_r160.slurm")
    text = open(path, encoding="utf-8").read()
    assert "EPOCHS=160" in text
    assert "FULL_EVAL_AT=388920,416700,444480" in text
    assert "R130_STEP=361140" in text and "FINAL_STEP=444480" in text
    # SAME run, same policy, and explicitly NOT a new branch
    assert 'RUN_ID="final6p_r100_sepmoments_seed${SEED}_${SUBSET_MODE}"' in text
    assert "OPT_POLICY=task_separated_adamw" in text
    assert '--resume "$OWN_LATEST"' in text
    # The continuation must never declare a transition nor resume the parent.
    # Checked on the actual srun invocation, not on the prose around it: the
    # script legitimately explains in a comment WHY the flag is absent.
    srun = text.split("srun python", 1)[1].split("\n\n", 1)[0]
    assert "--phase-transition" not in srun
    assert "PHASE_ARGS" not in text, "no conditional transition flag may exist"
    assert '--resume "$PARENT' not in text
    assert 'PARENT_CKPT' not in srun
    # no scientific override on the command line
    for forbidden in ("--lr-repetition", "--lr-naming", "--lr-comprehension",
                      "--c-align-weight", "--lr-boundary-steps",
                      "--batch-size", "--allow-glove-fallback",
                      "--no-subset-hash-check"):
        assert forbidden not in text, forbidden


# ===============================================  9. history preservation  ==

def test_r130_history_and_checkpoint_are_retained(sixp):
    out = sixp
    run = os.path.join(out, "sixp")
    src = ckpt(out, "sixp", 18)
    src_sha = sha(src)
    before = open(os.path.join(run, "metrics.tsv")).read()
    n_header = before.count("rep_epoch")

    launch(out, "sixp", 24, resume=src, separated=True, save_every=24,
           full_eval="24")

    # the source checkpoint is untouched and still present
    assert os.path.exists(src) and sha(src) == src_sha
    # earlier rows are preserved verbatim, new ones appended, one header only
    now = open(os.path.join(run, "metrics.tsv")).read()
    assert now.startswith(before)
    assert now.count("rep_epoch") == n_header == 1
    rows = list(csv.DictReader(open(os.path.join(run, "metrics.tsv")),
                               delimiter="\t"))
    steps = [int(r["step"]) for r in rows]
    assert 18 in steps and 24 in steps
    assert len(steps) == len(set(steps)) or steps.count(18) == before.count("\n") - 1
    # the first launch's config/provenance are not overwritten
    assert json.load(open(os.path.join(run, "config.json")))["total_steps"] == 18
    assert os.path.exists(os.path.join(run, "config_from_step_00000018.json"))


def test_metrics_schema_is_unchanged_so_appends_stay_aligned(sixp):
    """The continuation appends to an existing metrics.tsv, so the column set
    must not have changed since FINAL-6P wrote its header."""
    out = sixp
    run = os.path.join(out, "sixp")
    header = open(os.path.join(run, "metrics.tsv")).readline().rstrip("\n").split("\t")
    launch(out, "sixp", 24, resume=ckpt(out, "sixp", 18), separated=True,
           save_every=24, full_eval="24")
    lines = open(os.path.join(run, "metrics.tsv")).read().strip().split("\n")
    assert lines[0].split("\t") == header
    for line in lines[1:]:
        assert len(line.split("\t")) == len(header), "row/header misalignment"
    for col in ("m_div_RN", "m_div_RC", "m_div_NC", "comp_rank_median",
                "full_comp_top1", "full_naming_exact", "full_rep_full"):
        assert col in header, col
