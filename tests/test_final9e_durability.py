"""Acceptance tests for FINAL-9E: the durability continuation of FINAL-9P.

FINAL-9P showed that 2:2:3 rescues the LTM route (+.189713 full LTM at
matched N320 / C507.397260) at a small acquisition cost, with a flat
trajectory across its three milestones.  FINAL-9E asks only whether that
holds over substantially more naming and comprehension acquisition.

There is therefore NO scientific change to test here.  What must be proven
is the opposite: that this leg changes NOTHING -- same ratio, same banks,
same anchor, same LR, same L_dec, no phase transition, no branch -- and that
its three milestones land at cursors EXACTLY equal to the 1:2:3 control's on
naming and comprehension, so the only difference remains repetition.

The matching is asserted on INTEGER CURSORS rather than on float exposures:
`n_exposures` and `c_exposures` are cursor/per_epoch quotients, and 507.3973
style decimals cannot express the equality exactly.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.train_joint_scratch import (          # noqa: E402
    FINAL_FULL_MODE, INTERLEAVED_123, INTERLEAVED_223, LR_STAGE2,
    OPT_POLICY_GROUPED_RN_C, RATIO_123, RATIO_223, cycle_steps_for,
    lr_for_step, main,
)

JOB = "scripts/cluster/jeanzay/final9e_durability_r190_to_r470.slurm"
NINE_P = "scripts/cluster/jeanzay/final9p_ratio223_r130.slurm"

R_PASS, C_PASS = 463, 438
ANCHOR = 361_140                      # step at which 1:2:3 -> 2:2:3 took effect
SOURCE = 458_370                      # FINAL-9P endpoint, FINAL-9E source
FINAL = 912_110                       # M6 hard stop
RUN_ID = "final9p_r223_r130_seed22_final_full"

# 9E step -> matched 1:2:3 control step
MILESTONES = {588_010: 555_600, 750_060: 694_500, 912_110: 833_400}

SMOKE = ["--regime", "j0", "--seed", "22", "--subset-mode", "final_full",
         "--device", "cpu", "--max-words", "400", "--batch-size", "8",
         "--dorsal-pool-size", "32", "--lr-boundary-steps", "6",
         "--eval-every", "0", "--log-every", "0",
         "--glove-path", "tests/_no_such_glove_file.txt",
         "--allow-glove-fallback", "--no-subset-hash-check"]


def script(path=JOB):
    return open(os.path.join(ROOT, path), encoding="utf-8").read()


def srun_block(text):
    """Only the driver invocation, so prose in the header cannot mask a flag."""
    return "\n".join(chunk.split("\n\n", 1)[0]
                     for chunk in text.split("srun python")[1:])


def executable(text):
    """The script with comment-only lines removed.  Flag names legitimately
    appear in the header prose explaining what is NOT passed; what matters is
    that no line that actually runs contains them."""
    return "\n".join(ln for ln in text.splitlines()
                     if ln.strip() and not ln.lstrip().startswith("#"))


def launch(out, run_id, steps, *, resume=None, schedule=INTERLEAVED_123,
           grouped=False, phase=False, save_every=None):
    argv = SMOKE + ["--out-dir", out, "--run-id", run_id,
                    "--schedule", schedule, "--max-steps", str(steps),
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


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


# =====================================  cursors on the two trajectories  ===

def cursors_9e(step):
    """R/N/C batch cursors on the 9P/9E trajectory: 1:2:3 to the anchor, then
    2:2:3.  Derived from the implementation's own ratios and cycle length."""
    pre = ANCHOR // cycle_steps_for(INTERLEAVED_123)
    post = (step - ANCHOR) // cycle_steps_for(INTERLEAVED_223)
    return tuple(pre * a + post * b for a, b in zip(RATIO_123, RATIO_223))


def cursors_control(step):
    cyc = step // cycle_steps_for(INTERLEAVED_123)
    return tuple(cyc * a for a in RATIO_123)


# ==========================================  1. milestone arithmetic  ======

def test_source_state_is_the_final9p_endpoint():
    r, n, c = cursors_9e(SOURCE)
    assert (r, n, c) == (87_970, 148_160, 222_240)
    assert (r / R_PASS, n / R_PASS) == (190.0, 320.0)
    assert c / C_PASS == pytest.approx(507.3972603, abs=1e-7)
    assert (SOURCE - ANCHOR) % cycle_steps_for(INTERLEAVED_223) == 0


def test_milestones_match_the_control_exactly_on_n_and_c():
    """The whole design rests on this: identical INTEGER naming and
    comprehension cursors, so both runs take the same number of N and C
    steps over the same counter-addressed items."""
    expect = {                       # 9E step -> (R, N, C exposures)
        588_010: (270, 400, 634.2465753),
        750_060: (370, 500, 792.8082192),
        912_110: (470, 600, 951.3698630),
    }
    ctl_r = {555_600: 200, 694_500: 250, 833_400: 300}
    for s9, sc in MILESTONES.items():
        a, b = cursors_9e(s9), cursors_control(sc)
        assert a[1] == b[1], (s9, sc, "naming cursors differ")
        assert a[2] == b[2], (s9, sc, "comprehension cursors differ")
        r, n, c = expect[s9]
        assert a[0] / R_PASS == r and a[1] / R_PASS == n
        assert a[2] / C_PASS == pytest.approx(c, abs=1e-7)
        assert b[0] / R_PASS == ctl_r[sc]
        # repetition is the ONLY thing that differs
        assert a[0] != b[0]
    assert cursors_9e(912_110) == (217_610, 277_800, 416_700)
    assert cursors_control(833_400) == (138_900, 277_800, 416_700)


def test_every_milestone_is_a_whole_number_of_cycles_past_the_source():
    cyc = cycle_steps_for(INTERLEAVED_223)
    for s9 in MILESTONES:
        assert (s9 - SOURCE) % cyc == 0, s9
        assert (s9 - ANCHOR) % cyc == 0, s9
    assert (FINAL - SOURCE) == 453_740 == 64_820 * cyc


def test_the_leg_doubles_only_repetition():
    cyc = (FINAL - SOURCE) // cycle_steps_for(INTERLEAVED_223)
    r, n, c = (cyc * k for k in RATIO_223)
    assert (r, n, c) == (129_640, 129_640, 194_460)
    # the control covers the same N and C over R160 -> R300
    ctl_cyc = (833_400 - 444_480) // cycle_steps_for(INTERLEAVED_123)
    cr, cn, cc = (ctl_cyc * k for k in RATIO_123)
    assert (cn, cc) == (n, c), "N/C step counts must be identical"
    assert r == 2 * cr, "repetition must be exactly doubled"


def test_lr_is_1e_4_across_the_whole_leg():
    """R190 is far past the repetition-cursor boundary, so the untouched
    two-stage policy yields 1e-4 everywhere in this leg."""
    for r_cursor in (cursors_9e(SOURCE)[0], cursors_9e(FINAL)[0]):
        assert lr_for_step(r_cursor, 46_300) == LR_STAGE2
    assert cursors_9e(SOURCE)[0] > 46_300


# =================================================  2. job contract  =======

def test_job_pins_the_exact_source_and_hard_stop():
    t = script()
    assert "SOURCE_STEP=458370" in t
    assert "FINAL_STEP=912110" in t and "MAX_STEPS=912110" in t
    assert "FULL_EVAL_AT=588010,750060,912110" in t
    assert "SCHEDULE=interleaved_223" in t
    assert "OPT_POLICY=grouped_rn_c_adamw" in t
    assert 'RUN_ID="final9p_r223_r130_seed${SEED}_${SUBSET_MODE}"' in t
    assert "SEED=22" in t and "SUBSET_MODE=final_full" in t


def test_job_introduces_no_scientific_change():
    t = script()
    body = srun_block(t)
    for forbidden in ("--phase-transition", "--dec-weight", "--lr-repetition",
                      "--lr-naming", "--lr-comprehension", "--lr-boundary-steps",
                      "--c-align-weight", "--batch-size", "--dorsal-pool-size",
                      "--allow-glove-fallback", "--no-subset-hash-check",
                      "--epochs", "--torch-deterministic"):
        assert forbidden not in body, forbidden
    # and on no executable line anywhere, not merely in the srun call
    ex = executable(t)
    for forbidden in ("--phase-transition", "--dec-weight", "--lr-repetition",
                      "--lr-naming", "--lr-comprehension", "--c-align-weight",
                      "--epochs"):
        assert forbidden not in ex, forbidden
    assert "PHASE_ARGS" not in t
    assert "clone" not in body.lower() and "reset" not in body.lower()


def test_job_continues_the_same_run_and_never_branches():
    t = script()
    assert "PARENT_RUN_ID" not in t, "a parent implies a branch"
    assert '--resume "$OWN_LATEST"' in t
    assert "sort | tail -1" in t, "newest must be by step number, not mtime"
    # refuses to run against an unexpected run directory
    assert "FATAL: $RUN_DIR not found" in t or "not found — FINAL-9E continues" in t
    assert "FATAL: no checkpoints in" in t


def test_job_refuses_a_stale_or_finished_state():
    t = script()
    assert "OWN_STEP >= FINAL_STEP" in t
    assert "nothing to do: FINAL-9E already reached the M6 hard stop" in t
    assert "OWN_STEP < SOURCE_STEP" in t
    assert "FATAL: newest checkpoint is step" in t
    assert "#SBATCH --requeue" in t


def test_job_is_relocatable_and_pins_no_commit():
    """FINAL-9P runs from an isolated checkout, so no SHA may be baked in and
    no other checkout's HEAD may be required."""
    t = script()
    assert "REPO=${L3_REPO:-" in t
    assert "FATAL: L3_REPO" in t
    # no 40-hex commit anywhere
    assert not re.search(r"\b[0-9a-f]{40}\b", t), "a commit SHA is pinned"
    # both pins are optional: each is guarded by a non-empty test
    for var in ("L3_EXPECTED_COMMIT", "L3_EXPECTED_BRANCH"):
        assert f'if [[ -n "${{{var}:-}}" ]]; then' in t, var
    # but a dirty tracked tree is always fatal
    assert 'git status --porcelain -uno' in t


def preflight_source():
    """The source-preflight block lifted verbatim out of the job script, so
    the test exercises the code that will actually run on the cluster."""
    t = script()
    body = t.split("# ---8<--- FINAL-9E SOURCE PREFLIGHT", 1)[1]
    body = body.split("# --->8--- end FINAL-9E SOURCE PREFLIGHT", 1)[0]
    return body.split("python - <<'PY'", 1)[1].rsplit("PY", 1)[0]


def run_preflight(ckpt_path, *, src, final, anchor):
    env = dict(os.environ, L3_CKPT=ckpt_path, L3_SRC_STEP=str(src),
               L3_FINAL=str(final), L3_ANCHOR=str(anchor))
    return subprocess.run([sys.executable, "-"], input=preflight_source(),
                          env=env, cwd=ROOT, capture_output=True, text=True)


def test_source_preflight_accepts_a_genuine_223_checkpoint(trajectory):
    """Happy path: every key the preflight reads must exist with the right
    value on a real grouped 2:2:3 checkpoint."""
    _, src = trajectory
    r = run_preflight(src, src=32, final=100, anchor=18)
    assert r.returncode == 0, r.stderr
    assert "source preflight OK" in r.stdout
    for expect in ("interleaved_223", "[2, 2, 3]", "grouped_rn_c_adamw",
                   "comprehension", "dec 0.5"):
        assert expect in r.stdout, (expect, r.stdout)


@pytest.mark.parametrize("kw,needle", [
    (dict(src=40, final=100, anchor=18), "outside"),
    (dict(src=32, final=32, anchor=18), "outside"),
    (dict(src=32, final=100, anchor=19), "anchor"),
])
def test_source_preflight_refuses_a_state_it_does_not_expect(trajectory, kw,
                                                             needle):
    _, src = trajectory
    r = run_preflight(src, **kw)
    assert r.returncode == 1
    assert "FATAL preflight" in r.stderr and needle in r.stderr


def test_source_preflight_refuses_a_123_checkpoint(trajectory, tmp_path):
    """The guard that matters most: never continue the wrong schedule."""
    out, _ = trajectory
    r = run_preflight(ckpt(out, "run", 18), src=0, final=100, anchor=18)
    assert r.returncode == 1
    assert "schedule" in r.stderr


def test_job_verifies_code_capability_and_the_source_artefact():
    """Instead of a SHA: prove the checkout can run 2:2:3, and prove the
    checkpoint really is the 2:2:3 state we mean to continue."""
    t = script()
    assert "FATAL: this checkout has no interleaved_223 support" in t
    assert "INTERLEAVED_223 in SCHEDULES" in t
    for probe in ('ck.get("schedule") == "interleaved_223"',
                  '[2, 2, 3]',
                  'schedule_anchor_step", -1)) == anchor',
                  'ANCHOR_STEP=361140',
                  'ck.get("optimizer_policy") == "grouped_rn_c_adamw"',
                  '{"rn", "comprehension"}',
                  'float(ck.get("dec_weight")) == 0.5'):
        assert probe in t, probe


def test_cadences_are_identical_to_final9p_and_phase_constant():
    t, p = script(), script(NINE_P)
    assert "EVAL_EVERY=16205" in t and "SAVE_EVERY=32410" in t
    assert "EVAL_EVERY=16205" in p and "SAVE_EVERY=32410" in p, \
        "the curve must stay evenly sampled across the whole trajectory"
    cyc = cycle_steps_for(INTERLEAVED_223)
    for cadence, n_expected in ((16_205, 28), (32_410, 14)):
        assert cadence % cyc == 0
        hits = [s for s in range(cadence, FINAL + 1, cadence) if s > SOURCE]
        assert len(hits) == n_expected, (cadence, len(hits))
        assert len({(s - ANCHOR) % cyc for s in hits}) == 1
    saves = [s for s in range(32_410, FINAL + 1, 32_410) if s > SOURCE]
    assert saves[0] == 486_150 and saves[-1] == 907_480
    for m in MILESTONES:                      # a safety save just before each
        assert m - max(s for s in saves if s < m) == 4_630
    # no cadence can land on a milestone, which is why --full-eval-at saves
    for m in MILESTONES:
        assert m % 32_410 != 0 and m % 16_205 != 0


# ============================================  3. behaviour on artefacts  ==

@pytest.fixture
def trajectory(tmp_path):
    """A miniature of the real history: shared -> grouped RN|C -> 2:2:3,
    ending at the analogue of the FINAL-9P endpoint."""
    out = str(tmp_path / "runs")
    launch(out, "run", 6, save_every=6)                                # shared
    launch(out, "run", 18, resume=ckpt(out, "run", 6), grouped=True,
           phase=True, save_every=6)                                   # RN|C
    launch(out, "run", 32, resume=ckpt(out, "run", 18),
           schedule=INTERLEAVED_223, grouped=True, phase=True,
           save_every=32)                                              # 2:2:3
    return out, ckpt(out, "run", 32)


def test_continuation_changes_nothing_and_declares_nothing(trajectory):
    out, src = trajectory
    before = load(src)
    n_tr = len(before["phase_transitions"])
    diverged = [k for k, v in before["optimizer_states"]["rn"]["state"].items()]
    assert diverged, "the banks must already hold state"

    # exactly what the job does: same run id, same flags, larger budget
    launch(out, "run", 60, resume=src, schedule=INTERLEAVED_223,
           grouped=True, save_every=60)
    end = load(ckpt(out, "run", 60))

    assert end["schedule"] == INTERLEAVED_223
    assert end["schedule_ratio"] == [2, 2, 3]
    assert end["schedule_anchor_step"] == before["schedule_anchor_step"] == 18
    assert end["optimizer_policy"] == OPT_POLICY_GROUPED_RN_C
    assert set(end["optimizer_states"]) == {"rn", "comprehension"}
    assert len(end["phase_transitions"]) == n_tr, "a transition was declared"
    assert end["dec_weight"] == before["dec_weight"] == 0.5
    assert end["lr_policy"] == before["lr_policy"]
    assert end["c_align_weight"] == before["c_align_weight"]
    assert end["subset_definition_sha256"] == before["subset_definition_sha256"]
    assert end["stream_seeds"] == before["stream_seeds"]


def test_continuation_does_not_reclone_or_reset_the_banks(trajectory):
    out, src = trajectory
    launch(out, "run", 60, resume=src, schedule=INTERLEAVED_223,
           grouped=True, save_every=60)
    end = load(ckpt(out, "run", 60))
    rn = end["optimizer_states"]["rn"]["state"]
    co = end["optimizer_states"]["comprehension"]["state"]
    shared_ids = set(rn) & set(co)
    assert shared_ids, "the banks cover the same parameters"
    assert any(not torch.equal(rn[i]["exp_avg"], co[i]["exp_avg"])
               for i in shared_ids), "the banks were re-cloned or reset"
    # and they are genuinely separate storage, not aliases
    assert all(rn[i]["exp_avg"].data_ptr() != co[i]["exp_avg"].data_ptr()
               for i in shared_ids)


def test_requeue_is_deterministic(trajectory):
    """A requeue mid-leg must land on exactly the uninterrupted state: the
    anchor makes task order a pure function of steps since the transition."""
    out, src = trajectory
    launch(out, "whole", 74, resume=src, schedule=INTERLEAVED_223,
           grouped=True, save_every=74)
    launch(out, "split", 53, resume=src, schedule=INTERLEAVED_223,
           grouped=True, save_every=53)
    launch(out, "split", 74, resume=ckpt(out, "split", 53),
           schedule=INTERLEAVED_223, grouped=True, save_every=74)
    a, b = load(ckpt(out, "whole", 74)), load(ckpt(out, "split", 74))
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    assert not [k for k in sa if not torch.equal(sa[k], sb[k])]
    assert a["cursors"] == b["cursors"]
    assert a["schedule_anchor_step"] == b["schedule_anchor_step"] == 18
    for bank in ("rn", "comprehension"):
        xa = a["optimizer_states"][bank]["state"]
        xb = b["optimizer_states"][bank]["state"]
        assert not [i for i in xa
                    if not torch.equal(xa[i]["exp_avg"], xb[i]["exp_avg"])]


def test_milestones_checkpoint_themselves_off_the_save_cadence(tmp_path):
    """No cadence divides the milestones, so --full-eval-at must save."""
    out = str(tmp_path / "runs")
    argv = SMOKE + ["--out-dir", out, "--run-id", "m",
                    "--schedule", INTERLEAVED_223, "--max-steps", "21",
                    "--full-eval-at", "14", "--save-every", "21"]
    assert main(argv) == 0
    assert 14 % 21 != 0
    assert os.path.exists(ckpt(out, "m", 14))


# ===============================================  4. nothing else moves  ===

def test_earlier_family_scripts_are_untouched():
    for name, must in (
            ("final7l_fromscratch_gate_r220.slurm", "GATE_EPOCHS=220"),
            ("final7l_continue_r220_to_r700.slurm", "EPOCHS=700"),
            ("final7l_continue_r220_to_r1000.slurm", "EPOCHS=1000"),
            ("final8p_dec_weight_r130_to_r160.slurm", "DEC_WEIGHT=2.0"),
            ("final9p_ratio223_r130.slurm", "MAX_STEPS=458370")):
        t = script(f"scripts/cluster/jeanzay/{name}")
        assert must in t, name
        assert "912110" not in t, f"{name} was touched by FINAL-9E"


def test_historical_123_schedule_is_unchanged():
    assert RATIO_123 == (1, 2, 3)
    assert cycle_steps_for(INTERLEAVED_123) == 6
    assert cursors_control(833_400) == (138_900, 277_800, 416_700)
