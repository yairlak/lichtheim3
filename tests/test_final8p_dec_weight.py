"""Acceptance tests for FINAL-8P: the ventral-decode weight intervention.

ONE scientific change: the historical repetition objective's L_dec weight
0.5 -> 2.0, branched from the genuine from-scratch FINAL-7L R130 checkpoint.
Optimizer topology, optimizer states, LR, ratio, every other loss weight,
populations, seeds, streams and architecture are untouched.

L_dec is the only term that couples the two ventral surfaces in the failing
direction -- it backpropagates through the decoder AND, via s_hat, through
the encoder -- so it is the natural single lever for the co-adaptation
hypothesis that the FINAL-7A transplants motivated.  Those transplants
localised ENDPOINT dependence; this is the training intervention that tests
the mechanism.
"""
from __future__ import annotations

import copy
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
    flat_optimizer,
)
from scripts.naming_comprehension.train_joint_scratch import (          # noqa: E402
    CANONICAL_LOSS_WEIGHTS, FINAL_FULL_MODE, INTERLEAVED_123,
    MACRO_CYCLE_STEPS, OPT_POLICY_GROUPED_RN_C, RATIO_123,
    JointScratchTrainer, build_parser, main,
)

JOB = "scripts/cluster/jeanzay/final8p_dec_weight_r130_to_r160.slurm"

TINY = dict(device="cpu", max_words=400,
            lexicon_path="data/lexicon_en_glove_covered.tsv",
            dorsal_pool_size=32, batch_size=8, subset_mode=FINAL_FULL_MODE,
            subset_per_band=822, subset_size=32, lr_boundary_steps=6,
            allow_glove_fallback=True, require_subset_hash=False,
            glove_path="tests/_no_such_glove_file.txt")

SMOKE = ["--regime", "j0", "--seed", "22", "--subset-mode", "final_full",
         "--schedule", "interleaved_123", "--device", "cpu",
         "--max-words", "400", "--batch-size", "8", "--dorsal-pool-size", "32",
         "--lr-boundary-steps", "6", "--eval-every", "0", "--log-every", "0",
         "--glove-path", "tests/_no_such_glove_file.txt",
         "--allow-glove-fallback", "--no-subset-hash-check"]


def make(regime="j0", seed=22, **over):
    kw = dict(TINY)
    kw.update(over)
    return JointScratchTrainer(regime=regime, seed=seed, **kw)


def launch(out, run_id, steps, *, resume=None, grouped=False, phase=False,
           dec=None, save_every=None):
    argv = SMOKE + ["--out-dir", out, "--run-id", run_id,
                    "--max-steps", str(steps),
                    "--save-every", str(save_every or steps)]
    if grouped:
        argv += ["--optimizer-policy", "grouped_rn_c_adamw"]
    if phase:
        argv += ["--phase-transition"]
    if dec is not None:
        argv += ["--dec-weight", str(dec)]
    if resume:
        argv += ["--resume", resume]
    assert main(argv) == 0


def ckpt(out, run_id, step):
    return os.path.join(out, run_id, "checkpoints", f"step_{step:08d}.pt")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def script():
    return open(os.path.join(ROOT, JOB), encoding="utf-8").read()


# ====================================================  default is 0.5  =====

def test_default_dec_weight_is_the_canonical_half():
    assert CANONICAL_LOSS_WEIGHTS["dec"] == 0.5
    tr = make(schedule=INTERLEAVED_123)
    assert tr.dec_weight == 0.5
    assert tr.cfg.loss.dec == 0.5
    assert build_parser().parse_args(["--regime", "j0"]).dec_weight is None
    s = tr.resolved_settings()
    assert s["dec_weight"] == 0.5
    assert s["loss_weights"]["dec"] == 0.5
    assert tr.state_dict()["dec_weight"] == 0.5


def test_absent_flag_is_bitwise_identical_to_explicit_half():
    a = make(schedule=INTERLEAVED_123)                     # flag absent
    b = make(schedule=INTERLEAVED_123, dec_weight=0.5)     # explicit
    for _ in range(MACRO_CYCLE_STEPS):
        ra, rb = a.train_step(), b.train_step()
        assert ra["joint_total"] == rb["joint_total"]
    pa = {k: v.detach().clone() for k, v in a.model.named_parameters()}
    pb = {k: v.detach().clone() for k, v in b.model.named_parameters()}
    assert not [k for k in pa if not torch.equal(pa[k], pb[k])]


def test_negative_weight_is_refused():
    with pytest.raises(ValueError, match="dec_weight"):
        make(schedule=INTERLEAVED_123, dec_weight=-1.0)


# ================================  the intervention changes only L_dec  ====

def test_dec_weight_changes_only_the_dec_term_of_the_objective():
    """joint(dec=2) - joint(dec=.5) == 1.5 * L_dec on the same R batch."""
    a = make(schedule=INTERLEAVED_123)
    b = make(schedule=INTERLEAVED_123, dec_weight=2.0)
    pa = {k: v.detach().clone() for k, v in a.model.named_parameters()}
    pb = {k: v.detach().clone() for k, v in b.model.named_parameters()}
    assert not [k for k in pa if not torch.equal(pa[k], pb[k])], "not paired"

    # advance both to the same repetition step
    while a.task_for_step(a.global_step) != "repetition":
        a.train_step(); b.train_step()
    ra, rb = a.train_step(), b.train_step()
    assert ra["task"] == rb["task"] == "repetition"
    # every recorded component matches except the weighted total
    for k in ("rep", "align", "dec", "wm", "gate", "pool_ce"):
        assert ra[k] == pytest.approx(rb[k], rel=1e-9), k
    assert rb["joint_total"] - ra["joint_total"] == pytest.approx(
        1.5 * ra["dec"], rel=1e-6)


def test_only_the_dec_weight_differs_in_the_effective_settings():
    a = make(schedule=INTERLEAVED_123).resolved_settings()
    b = make(schedule=INTERLEAVED_123, dec_weight=2.0).resolved_settings()
    diff = {k for k in a if a[k] != b[k]}
    assert diff == {"dec_weight", "loss_weights"}, diff
    assert b["loss_weights"] == dict(a["loss_weights"], dec=2.0)
    assert b["loss_weights_canonical"]["dec"] == 0.5, "canonical must be kept"
    # nothing else scientific moved
    for k in ("lambda_C", "lambda_N", "tau", "grad_clip", "weight_decay",
              "schedule_ratio_R_N_C", "optimizer_policy", "lr_policy",
              "c_align_weight"):
        assert a[k] == b[k], k


# =====================================  phase transition and resume  =======

@pytest.fixture
def branched(tmp_path):
    """A grouped RN|C run (the FINAL-7L analogue) branched into dec=2.0."""
    out = str(tmp_path / "runs")
    launch(out, "seven_l", 6, save_every=6)                            # shared
    launch(out, "seven_l", 18, resume=ckpt(out, "seven_l", 6), grouped=True,
           phase=True, save_every=6)                                   # RN|C
    src = ckpt(out, "seven_l", 18)
    launch(out, "eight_p", 24, resume=src, grouped=True, phase=True,
           dec=2.0, save_every=6)                                      # dec 2.0
    return out, src


def test_silent_dec_change_is_refused(branched):
    out, src = branched
    ck = torch.load(src, map_location="cpu", weights_only=False)
    with pytest.raises(RuntimeError, match="PHASE TRANSITION"):
        make(schedule=INTERLEAVED_123,
             optimizer_policy=OPT_POLICY_GROUPED_RN_C,
             dec_weight=2.0).load_state_dict(copy.deepcopy(ck), source="t")
    # ... and the reverse direction is equally refused
    end = torch.load(ckpt(out, "eight_p", 24), map_location="cpu",
                     weights_only=False)
    with pytest.raises(RuntimeError, match="PHASE TRANSITION"):
        make(schedule=INTERLEAVED_123,
             optimizer_policy=OPT_POLICY_GROUPED_RN_C).load_state_dict(
                 copy.deepcopy(end), source="t")


def test_transition_records_only_the_dec_change(branched):
    out, _ = branched
    end = torch.load(ckpt(out, "eight_p", 24), map_location="cpu",
                     weights_only=False)
    assert end["dec_weight"] == 2.0
    assert end["optimizer_policy"] == OPT_POLICY_GROUPED_RN_C
    rec = end["phase_transitions"][-1]
    assert rec["changed"] == ["dec_weight"], rec["changed"]
    assert rec["old_dec_weight"] == 0.5 and rec["new_dec_weight"] == 2.0
    assert rec["transition_step"] == 18
    assert rec["old_optimizer_policy"] == rec["new_optimizer_policy"]
    assert rec["old_lr_policy"] == rec["new_lr_policy"]
    assert rec["moment_initialization"] == "unchanged", \
        "no moment cloning may occur for a loss-weight change"
    prov = json.load(open(os.path.join(out, "eight_p", "provenance.json")))
    assert prov["dec_weight"] == 2.0
    assert prov["loss_weights"]["dec"] == 2.0
    assert prov["ancestry"]["is_branch"] is True
    assert prov["ancestry"]["parent_run_id"] == "seven_l"


def test_optimizer_states_rng_and_cursors_survive_the_transition(branched):
    """A loss-weight change must not touch the optimizer, RNG or cursors."""
    out, src = branched
    before = torch.load(src, map_location="cpu", weights_only=False)
    fresh = make(schedule=INTERLEAVED_123,
                 optimizer_policy=OPT_POLICY_GROUPED_RN_C, dec_weight=2.0,
                 allow_phase_transition=True)
    fresh.load_state_dict(copy.deepcopy(before), source="t")

    got = {n: {f"{pid}.{k}": v for pid, e in o.state_dict()["state"].items()
               for k, v in e.items()}
           for n, o in fresh.banks.items()}
    want = flat_optimizer(before)
    assert set(got) == set(want)
    for name in want:
        for k, v in want[name].items():
            ok = (torch.equal(v, got[name][k]) if torch.is_tensor(v)
                  else v == got[name][k])
            assert ok, f"bank {name} entry {k} changed"
    assert fresh.global_step == before["global_step"]
    assert fresh.cursors == {k: int(v) for k, v in before["cursors"].items()}
    sm = {k: v.detach().clone() for k, v in fresh.model.named_parameters()}
    for k, v in sm.items():
        assert torch.equal(v, before["model_state_dict"][k]), k
    # banks stay distinct: nothing was cloned or reset
    ptrs = [next(iter(o.state.values()))["exp_avg"].data_ptr()
            for o in fresh.banks.values()]
    assert len(set(ptrs)) == 2


def test_requeue_after_the_transition_declares_nothing(branched):
    out, _ = branched
    launch(out, "eight_p", 30, resume=ckpt(out, "eight_p", 24), grouped=True,
           dec=2.0, save_every=30)                       # NO --phase-transition
    end = torch.load(ckpt(out, "eight_p", 30), map_location="cpu",
                     weights_only=False)
    assert len(end["phase_transitions"]) == 2, "one grouped + one dec"
    assert end["dec_weight"] == 2.0


def test_source_run_is_untouched_by_the_branch(branched):
    out, src = branched
    sdir = os.path.join(out, "seven_l")
    fp = {f: sha(os.path.join(sdir, f)) for f in os.listdir(sdir)
          if os.path.isfile(os.path.join(sdir, f))}
    fp.update({n: sha(os.path.join(sdir, "checkpoints", n))
               for n in os.listdir(os.path.join(sdir, "checkpoints"))})
    launch(out, "eight_p2", 24, resume=src, grouped=True, phase=True,
           dec=2.0, save_every=24)
    now = {f: sha(os.path.join(sdir, f)) for f in os.listdir(sdir)
           if os.path.isfile(os.path.join(sdir, f))}
    now.update({n: sha(os.path.join(sdir, "checkpoints", n))
                for n in os.listdir(os.path.join(sdir, "checkpoints"))})
    assert now == fp, "the FINAL-7L source run was modified"


# ==============================================  budget and job contract  ==

def test_final8p_milestone_accounting():
    r_pass, c_pass = 463, 438
    r, n, c = RATIO_123
    expected = {                     # R exp -> (cycles, step, N exp, C exp)
        130: (60_190, 361_140, 260.0, 412.2603),      # source
        140: (64_820, 388_920, 280.0, 443.9726),
        150: (69_450, 416_700, 300.0, 475.6849),
        160: (74_080, 444_480, 320.0, 507.3973),
    }
    for r_exp, (cycles, step, n_exp, c_exp) in expected.items():
        assert r_exp * r_pass == cycles
        assert cycles * MACRO_CYCLE_STEPS == step
        assert cycles * n / r_pass == pytest.approx(n_exp)
        assert cycles * c / c_pass == pytest.approx(c_exp, abs=1e-3)
    assert expected[160][1] - expected[130][1] == 83_340


def test_final8p_job_contract():
    t = script()
    assert "DEC_WEIGHT=2.0" in t and '--dec-weight "$DEC_WEIGHT"' in t
    assert "EPOCHS=160" in t
    assert "FULL_EVAL_AT=388920,416700,444480" in t
    assert "SOURCE_STEP=361140" in t and "FINAL_STEP=444480" in t
    # a NEW run directory that branches from FINAL-7L and never writes to it
    assert 'RUN_ID="final8p_dec2_r130_seed${SEED}_${SUBSET_MODE}"' in t
    assert 'PARENT_RUN_ID="final7l_fromscratch_seed${SEED}_${SUBSET_MODE}"' in t
    assert '[[ "$RUN_ID" != "$PARENT_RUN_ID" ]]' in t
    assert "FATAL: source checkpoint" in t
    # first launch declares; requeue resumes its own without declaring
    assert 'RESUME_FROM="$PARENT_CKPT"' in t and 'RESUME_FROM="$OWN_LATEST"' in t
    assert "--phase-transition" in t
    # the optimizer topology and everything else stay frozen
    assert "OPT_POLICY=grouped_rn_c_adamw" in t
    assert "SCHEDULE=interleaved_123" in t and "SEED=22" in t
    for forbidden in ("--lr-repetition", "--lr-naming", "--lr-comprehension",
                      "--lr-boundary-steps", "--c-align-weight",
                      "--batch-size", "--dorsal-pool-size",
                      "--allow-glove-fallback", "--no-subset-hash-check"):
        assert forbidden not in t, forbidden


def test_final7l_scripts_are_untouched():
    """FINAL-8P must not disturb the running family."""
    for name, must in (
            ("final7l_fromscratch_gate_r220.slurm", "GATE_EPOCHS=220"),
            ("final7l_continue_r220_to_r700.slurm", "EPOCHS=700"),
            ("final7l_continue_r220_to_r1000.slurm", "EPOCHS=1000")):
        t = open(os.path.join(ROOT, "scripts/cluster/jeanzay", name),
                 encoding="utf-8").read()
        assert must in t
        assert "--dec-weight" not in t, f"{name} must not carry the FINAL-8 knob"
