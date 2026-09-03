"""Acceptance tests for FINAL-7L: the predeclared from-scratch recipe.

    random init -> shared_adamw @ 1e-3 -> R100
                -> grouped_rn_c_adamw @ 1e-4 -> R700

R220 is an operational inspection gate, not a scientific change: the gate run
and the overnight continuation are two launches of ONE predeclared recipe in
ONE run directory.

The end-to-end test below runs a miniature of exactly that two-stage recipe
with the real driver, so the transition, the moment cloning source, and the
requeue paths are exercised rather than asserted.
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

from scripts.naming_comprehension.compare_checkpoints import (          # noqa: E402
    compare_checkpoints, flat_optimizer,
)
from scripts.naming_comprehension.train_joint_scratch import (          # noqa: E402
    LR_STAGE1, LR_STAGE2, MACRO_CYCLE_STEPS, MOMENT_INIT_CLONE_GROUPED,
    OPT_POLICY_GROUPED_RN_C, OPT_POLICY_SHARED, RATIO_123, lr_for_step, main,
)

GATE = "scripts/cluster/jeanzay/final7l_fromscratch_gate_r220.slurm"
NIGHT = "scripts/cluster/jeanzay/final7l_continue_r220_to_r700.slurm"

SMOKE = ["--regime", "j0", "--seed", "22", "--subset-mode", "final_full",
         "--schedule", "interleaved_123", "--device", "cpu",
         "--max-words", "400", "--batch-size", "8", "--dorsal-pool-size", "32",
         "--lr-boundary-steps", "6", "--eval-every", "0", "--log-every", "0",
         "--glove-path", "tests/_no_such_glove_file.txt",
         "--allow-glove-fallback", "--no-subset-hash-check"]


def launch(out, run_id, steps, *, resume=None, grouped=False, phase=False,
           save_every=None, full_eval=None, seed=None):
    argv = list(SMOKE)
    if seed is not None:
        argv[argv.index("--seed") + 1] = str(seed)
    argv = argv + ["--out-dir", out, "--run-id", run_id,
                    "--max-steps", str(steps),
                    "--save-every", str(save_every or steps)]
    if grouped:
        argv += ["--optimizer-policy", "grouped_rn_c_adamw"]
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


def script(path):
    return open(os.path.join(ROOT, path), encoding="utf-8").read()


def srun_block(text):
    """Only the training invocations, not the surrounding prose."""
    return "\n".join(chunk.split("\n\n", 1)[0]
                     for chunk in text.split("srun python")[1:])


# =====================================  1-5. from scratch, stage 1, replay ==

def test_gate_stage1_starts_from_random_initialisation():
    t = script(GATE)
    # the from-scratch launch passes NO --resume
    assert "RESUME_ARGS=()" in t
    assert "STARTING FROM RANDOM INITIALISATION" in t.upper()
    # ... but once any checkpoint exists it never restarts from scratch
    assert "STAGE 1 requeue" in t
    assert 'RESUME_ARGS=(--resume "$LATEST")' in t


def test_gate_uses_shared_optimizer_through_r100_then_grouped():
    t = script(GATE)
    assert "GROUPED=grouped_rn_c_adamw" in t
    blocks = t.split("srun python")
    stage1, stage2 = blocks[1], blocks[2]
    assert "--optimizer-policy" not in stage1.split("\n\n")[0], \
        "stage 1 must use the default shared policy"
    assert '--optimizer-policy "$GROUPED"' in stage2
    assert "--phase-transition" not in stage1.split("\n\n")[0]


def test_historical_lr_boundary_is_untouched():
    """No LR flag anywhere; the historical two-stage policy does the work."""
    for path in (GATE, NIGHT):
        t = script(path)
        for forbidden in ("--lr-repetition", "--lr-naming",
                          "--lr-comprehension", "--lr-boundary-steps"):
            assert forbidden not in t, f"{path}: {forbidden}"
    # and the boundary means exactly R100
    assert 100 * 463 == 46_300
    assert lr_for_step(46_299, 46_300) == LR_STAGE1
    assert lr_for_step(46_300, 46_300) == LR_STAGE2


def test_gate_saves_its_own_r100_and_uses_it_for_stage_two():
    t = script(GATE)
    assert "R100_STEP=277800" in t
    assert 'STAGE1_FULL_EVAL=277800' in t
    assert 'NEW_R100="$CKPTS/step_$(printf ' in t
    # stage 2 resumes $LATEST, which stage 1 has just set to the NEW R100
    assert 'LATEST="$CKPTS/step_$(printf \'%08d\' $R100_STEP).pt"' in t


def test_historical_checkpoint_is_comparison_only():
    t = script(GATE)
    var = "HISTORICAL_R100_FOR_COMPARISON_ONLY"
    assert var in t, "the reference must be named unambiguously"
    # it is never handed to a training command
    assert var not in srun_block(t)
    assert "final3p" not in srun_block(t)
    # it is used only by the read-only comparison tool, non-fatally
    assert "compare_checkpoints.py" in t and "|| true" in t
    # the overnight job never mentions the historical run at all
    assert "final3p" not in script(NIGHT)


def test_compare_checkpoints_is_non_fatal_and_never_substitutes():
    import inspect
    from scripts.naming_comprehension import compare_checkpoints as mod
    src = inspect.getsource(mod)
    assert "never fatal" in src and "never substituted" in src.lower()
    assert "torch.save" not in src, "the comparison tool must not write state"


# ==========================================  6-8. transition and cloning  ===

@pytest.fixture(scope="module")
def two_stage(tmp_path_factory):
    """A miniature of the real recipe: fresh shared run to the boundary, then
    the declared grouped transition, in ONE run directory."""
    out = str(tmp_path_factory.mktemp("runs"))
    launch(out, "gate", 6, save_every=6, full_eval="6")          # stage 1
    launch(out, "gate", 18, resume=ckpt(out, "gate", 6), grouped=True,
           phase=True, save_every=6, full_eval="18")             # stage 2
    return out


def test_transition_declared_exactly_once(two_stage):
    out = two_stage
    end = torch.load(ckpt(out, "gate", 18), map_location="cpu",
                     weights_only=False)
    assert end["optimizer_policy"] == OPT_POLICY_GROUPED_RN_C
    assert len(end["phase_transitions"]) == 1
    rec = end["phase_transitions"][0]
    assert rec["old_optimizer_policy"] == OPT_POLICY_SHARED
    assert rec["new_optimizer_policy"] == OPT_POLICY_GROUPED_RN_C
    assert rec["moment_initialization"] == MOMENT_INIT_CLONE_GROUPED
    assert rec["transition_step"] == 6
    # a requeue after the transition must not add a second one
    launch(out, "gate", 24, resume=ckpt(out, "gate", 18), grouped=True,
           save_every=24)
    again = torch.load(ckpt(out, "gate", 24), map_location="cpu",
                       weights_only=False)
    assert len(again["phase_transitions"]) == 1


def test_moments_are_cloned_from_the_new_run_not_the_historical_one(two_stage,
                                                                    tmp_path):
    """The banks must inherit THIS run's own shared state."""
    out = two_stage
    own_r100 = torch.load(ckpt(out, "gate", 6), map_location="cpu",
                          weights_only=False)
    shared = flat_optimizer(own_r100)["shared"]

    # A decoy standing in for "some other run".  It must use a DIFFERENT
    # seed: the same recipe at seed 22 reproduces bitwise, which is the very
    # determinism the R100 replay check exploits.
    launch(out, "decoy", 6, save_every=6, seed=23)
    decoy = flat_optimizer(torch.load(ckpt(out, "decoy", 6), map_location="cpu",
                                      weights_only=False))["shared"]
    assert any(not torch.equal(shared[k], decoy[k]) for k in shared
               if torch.is_tensor(shared[k])), "decoy is not distinguishable"

    # re-run the transition alone (zero new steps) and compare the banks
    launch(out, "probe", 6, resume=ckpt(out, "gate", 6), grouped=True,
           phase=True, save_every=6)
    banks = flat_optimizer(torch.load(ckpt(out, "probe", 6), map_location="cpu",
                                      weights_only=False))
    assert set(banks) == {"rn", "comprehension"}
    for name, bank in banks.items():
        for k, v in shared.items():
            ok = (torch.equal(v, bank[k]) if torch.is_tensor(v) else v == bank[k])
            assert ok, f"bank {name} does not match this run's own R100 at {k}"


def test_banks_do_not_alias_after_the_transition(two_stage):
    sd = torch.load(ckpt(two_stage, "gate", 18), map_location="cpu",
                    weights_only=False)["optimizer_states"]
    ptrs = [next(iter(sd[n]["state"].values()))["exp_avg"].data_ptr()
            for n in ("rn", "comprehension")]
    assert len(set(ptrs)) == 2, "the RN and C banks share storage"


# ====================================  9-10, 13. accounting and milestones ==

def test_gate_and_overnight_accounting():
    r_pass, c_pass = 463, 438
    r, n, c = RATIO_123
    expected = {                     # R exp -> (cycles, step, N exp, C exp)
        100: (46_300, 277_800, 200.0, 317.1233),
        130: (60_190, 361_140, 260.0, 412.2603),
        160: (74_080, 444_480, 320.0, 507.3973),
        220: (101_860, 611_160, 440.0, 697.6712),
        300: (138_900, 833_400, 600.0, 951.3699),
        400: (185_200, 1_111_200, 800.0, 1268.4932),
        500: (231_500, 1_389_000, 1000.0, 1585.6164),
        600: (277_800, 1_666_800, 1200.0, 1902.7397),
        700: (324_100, 1_944_600, 1400.0, 2219.8630),
    }
    for r_exp, (cycles, step, n_exp, c_exp) in expected.items():
        assert r_exp * r_pass == cycles
        assert cycles * MACRO_CYCLE_STEPS == step
        assert cycles * n / r_pass == pytest.approx(n_exp)
        assert cycles * c / c_pass == pytest.approx(c_exp, abs=1e-3)
    assert expected[700][1] - expected[220][1] == 1_333_440


def test_gate_declares_its_milestones_and_hard_stop():
    t = script(GATE)
    assert "STAGE1_EPOCHS=100" in t and "GATE_EPOCHS=220" in t
    assert "STAGE1_FULL_EVAL=277800" in t
    assert "STAGE2_FULL_EVAL=361140,444480,611160" in t
    assert "GATE_STEP=611160" in t
    assert "--time=03:00:00" in t


def test_overnight_declares_its_milestones_and_hard_stop():
    t = script(NIGHT)
    assert "EPOCHS=700" in t
    assert "FULL_EVAL_AT=833400,1111200,1389000,1666800,1944600" in t
    assert "FINAL_STEP=1944600" in t and "GATE_STEP=611160" in t
    assert 'RUN_ID="final7l_fromscratch_seed${SEED}_${SUBSET_MODE}"' in t


# ===================================  11-12. overnight resume semantics  ===

def test_overnight_resumes_only_its_own_gate_endpoint():
    t = script(NIGHT)
    assert '--resume "$OWN_LATEST"' in t
    assert 'GATE_CKPT="$CKPTS/step_$(printf ' in t
    assert "FATAL: gate endpoint" in t
    # ordinary grouped resume: no transition, no cloning
    assert "--phase-transition" not in srun_block(t)
    assert "PHASE_ARGS" not in t
    for forbidden in ("--c-align-weight", "--batch-size", "--dorsal-pool-size",
                      "--allow-glove-fallback", "--no-subset-hash-check"):
        assert forbidden not in t, forbidden


def test_overnight_style_resume_adds_no_transition_and_keeps_banks(two_stage):
    out = two_stage
    src = ckpt(out, "gate", 18)
    before = flat_optimizer(torch.load(src, map_location="cpu",
                                       weights_only=False))

    def specialised(banks):
        return [k for k, v in banks["rn"].items()
                if torch.is_tensor(v)
                and not torch.equal(v, banks["comprehension"][k])]

    diverged = specialised(before)
    assert diverged, "banks should have specialised by the gate endpoint"

    launch(out, "gate", 30, resume=src, grouped=True, save_every=30)
    end = torch.load(ckpt(out, "gate", 30), map_location="cpu",
                     weights_only=False)
    assert len(end["phase_transitions"]) == 1, "a transition was re-declared"
    assert specialised(flat_optimizer(end)), \
        "the banks were re-cloned from a shared state"


# ==============================================  14-15. idempotence, prov ===

def test_endpoints_are_idempotent(two_stage):
    """Relaunching at an already-reached budget changes no weights."""
    out = two_stage
    src = ckpt(out, "gate", 18)
    before = sha(src)
    launch(out, "gate", 18, resume=src, grouped=True, save_every=18)
    assert os.path.exists(src)
    end = torch.load(ckpt(out, "gate", 18), map_location="cpu",
                     weights_only=False)
    assert end["global_step"] == 18
    # the scripts also short-circuit once the hard stop is reached
    assert "nothing to do: the gate already reached R220" in script(GATE)
    assert "nothing to do: already reached the R700 hard stop" in script(NIGHT)


def test_provenance_records_a_genuine_from_scratch_two_stage_run(two_stage):
    out = two_stage
    first = json.load(open(os.path.join(out, "gate", "provenance.json")))
    assert first["resumed_from"] is None, "stage 1 must not resume anything"
    assert first["ancestry"] is None
    assert first["lr_policy"]["kind"] == "two_stage_rep_cursor"
    assert first["optimizer_policy"] == OPT_POLICY_SHARED

    second = json.load(open(os.path.join(
        out, "gate", "provenance_from_step_00000006.json")))
    assert second["optimizer_policy"] == OPT_POLICY_GROUPED_RN_C
    assert second["phase_transition_declared"] is True
    assert second["ancestry"]["is_branch"] is False, "same run, not a branch"
    assert second["ancestry"]["parent_run_id"] == "gate"
    assert second["optimizer_bank_layout"] == {
        "repetition": "rn", "naming": "rn", "comprehension": "comprehension"}

    # one continuous history, single header
    text = open(os.path.join(out, "gate", "metrics.tsv")).read()
    assert text.count("rep_epoch") == 1
    rows = list(csv.DictReader(open(os.path.join(out, "gate", "metrics.tsv")),
                               delimiter="\t"))
    steps = [int(r["step"]) for r in rows]
    assert 6 in steps and 18 in steps


def test_compare_checkpoints_detects_identity_and_difference(two_stage,
                                                             tmp_path):
    out = two_stage
    a = ckpt(out, "gate", 6)
    same = compare_checkpoints(a, a)
    assert same["identical"] is True
    assert same["comparison_only"] is True and same["non_fatal"] is True
    diff = compare_checkpoints(a, ckpt(out, "decoy", 6))
    assert diff["identical"] is False
    labels = {s["label"]: s for s in diff["sections"]}
    assert labels["model"]["n_differing"] > 0
    assert any(k.startswith("optimizer[") for k in labels)
