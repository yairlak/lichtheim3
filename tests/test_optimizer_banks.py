"""Acceptance tests for FINAL-6P: task-separated AdamW moment banks.

ONE shared network, THREE AdamW instances over the SAME Parameter objects,
each owning its exp_avg / exp_avg_sq / per-parameter step counters.  Only the
current task's bank steps.  No model parameter is duplicated and no forward
path changes: the intervention is task-conditioned optimizer MEMORY.

At the R100 phase transition each bank inherits a bitwise clone of the shared
optimizer state, so the pilot asks what happens when optimizer memories become
task-specific FROM R100 onward after sharing one history up to it -- not what
training with separated banks from step 0 would do.
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

from scripts.naming_comprehension.train_joint_scratch import (          # noqa: E402
    FINAL_FULL_MODE, INTERLEAVED_123, LR_TASKS, MACRO_CYCLE_STEPS,
    MOMENT_INIT_CLONE, OPT_POLICY_SEPARATED, OPT_POLICY_SHARED, RATIO_123,
    SUMMED_SCHEDULE, JointScratchTrainer, build_parser, main,
)

TINY = dict(device="cpu", max_words=400,
            lexicon_path="data/lexicon_en_glove_covered.tsv",
            dorsal_pool_size=32, batch_size=8, subset_mode=FINAL_FULL_MODE,
            subset_per_band=822, subset_size=32, lr_boundary_steps=6,
            allow_glove_fallback=True, require_subset_hash=False,
            glove_path="tests/_no_such_glove_file.txt")

SMOKE = ["--regime", "j0", "--seed", "22", "--subset-mode", "final_full",
         "--schedule", "interleaved_123", "--device", "cpu",
         "--max-words", "400", "--batch-size", "8", "--dorsal-pool-size", "32",
         "--lr-boundary-steps", "6", "--eval-every", "0", "--log-every", "1",
         "--glove-path", "tests/_no_such_glove_file.txt",
         "--allow-glove-fallback", "--no-subset-hash-check"]


def make(regime="j0", seed=22, **over):
    kw = dict(TINY)
    kw.update(over)
    return JointScratchTrainer(regime=regime, seed=seed, **kw)


def params(model):
    return {k: v.detach().clone() for k, v in model.named_parameters()}


def opt_state(optimizer):
    """Flat {(param index, key): tensor} view of an optimizer's state."""
    out = {}
    for pid, st in optimizer.state_dict()["state"].items():
        for k, v in st.items():
            out[(pid, k)] = v.detach().clone() if torch.is_tensor(v) else v
    return out


def same_state(a, b):
    if set(a) != set(b):
        return False
    for k, va in a.items():
        vb = b[k]
        ok = torch.equal(va, vb) if torch.is_tensor(va) else va == vb
        if not ok:
            return False
    return True


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


# ==================================================  1-2. defaults, compat  ==

def test_default_optimizer_policy_is_shared():
    tr = make(schedule=INTERLEAVED_123)
    assert tr.optimizer_policy == OPT_POLICY_SHARED
    assert tr.task_optims is None
    assert tr.optimizer_for("naming") is tr.optim is tr.optimizer_for("repetition")
    assert build_parser().parse_args(
        ["--regime", "j0"]).optimizer_policy == OPT_POLICY_SHARED
    ck = tr.state_dict()
    assert "optimizer_state_dict" in ck and "optimizer_states" not in ck
    assert ck["optimizer_policy"] == OPT_POLICY_SHARED


def test_shared_runs_are_unchanged_by_the_new_code_path():
    a = make(schedule=INTERLEAVED_123)
    b = make(schedule=INTERLEAVED_123, optimizer_policy=OPT_POLICY_SHARED)
    for _ in range(MACRO_CYCLE_STEPS * 2):
        ra, rb = a.train_step(), b.train_step()
        assert ra["joint_total"] == rb["joint_total"]
    pa, pb = params(a.model), params(b.model)
    assert not [k for k in pa if not torch.equal(pa[k], pb[k])]
    assert same_state(opt_state(a.optim), opt_state(b.optim))


def test_separated_requires_an_interleaved_schedule():
    with pytest.raises(RuntimeError, match="interleaved"):
        make(schedule=SUMMED_SCHEDULE, optimizer_policy=OPT_POLICY_SEPARATED)
    with pytest.raises(ValueError):
        make(schedule=INTERLEAVED_123, optimizer_policy="nonsense")


# =====================================  3-7. transition, cloning, aliasing  ==

@pytest.fixture
def transitioned(tmp_path):
    """A shared-policy checkpoint plus a trainer that took it into separation."""
    src = make(schedule=INTERLEAVED_123)
    for _ in range(MACRO_CYCLE_STEPS * 2):       # build real moments
        src.train_step()
    p = tmp_path / "shared.pt"
    torch.save(src.state_dict(), str(p))
    ck = torch.load(str(p), weights_only=False)
    sep = make(schedule=INTERLEAVED_123, optimizer_policy=OPT_POLICY_SEPARATED,
               allow_phase_transition=True)
    sep.load_state_dict(copy.deepcopy(ck), source="shared.pt")
    return src, ck, sep


def test_explicit_phase_transition_is_required(tmp_path, transitioned):
    _, ck, _ = transitioned
    with pytest.raises(RuntimeError, match="PHASE TRANSITION"):
        make(schedule=INTERLEAVED_123,
             optimizer_policy=OPT_POLICY_SEPARATED).load_state_dict(
                 copy.deepcopy(ck), source="t")


def test_shared_state_is_cloned_bitwise_into_every_bank(transitioned):
    src, ck, sep = transitioned
    shared = opt_state(src.optim)
    assert shared, "the source had no optimizer state to clone"
    for t in LR_TASKS:
        assert same_state(opt_state(sep.task_optims[t]), shared), \
            f"bank {t} is not a bitwise copy of the shared state"
    rec = sep.phase_transitions[-1]
    assert rec["old_optimizer_policy"] == OPT_POLICY_SHARED
    assert rec["new_optimizer_policy"] == OPT_POLICY_SEPARATED
    assert rec["moment_initialization"] == MOMENT_INIT_CLONE
    assert "optimizer_policy" in rec["changed"]
    # step counters preserved, not reset
    for t in LR_TASKS:
        steps = [v for (pid, k), v in opt_state(sep.task_optims[t]).items()
                 if k == "step"]
        assert steps and all(float(x) > 0 for x in steps)


def test_banks_do_not_alias(transitioned):
    """The banks must own independent tensors: mutating one cannot touch
    another.  torch's load_state_dict does NOT copy already-matching tensors,
    so this is a real failure mode, not a theoretical one."""
    _, _, sep = transitioned
    r, n, c = (sep.task_optims[t] for t in ("repetition", "naming",
                                            "comprehension"))
    ptrs = []
    for opt in (r, n, c):
        st = next(iter(opt.state.values()))
        ptrs.append(st["exp_avg"].data_ptr())
    assert len(set(ptrs)) == 3, "banks share exp_avg storage"

    before_n = opt_state(n)
    before_c = opt_state(c)
    first = next(iter(r.state.values()))
    first["exp_avg"].add_(1.0)               # mutate the R bank in place
    assert same_state(opt_state(n), before_n), "mutating R changed N"
    assert same_state(opt_state(c), before_c), "mutating R changed C"


def test_transition_touches_nothing_but_the_optimizer(transitioned):
    src, ck, sep = transitioned
    ps, pn = params(src.model), params(sep.model)
    assert not [k for k in ps if not torch.equal(ps[k], pn[k])]
    assert sep.global_step == int(ck["global_step"])
    assert sep.cursors == {k: int(v) for k, v in ck["cursors"].items()}
    assert torch.equal(torch.get_rng_state(),
                       ck["rng_states"]["torch"].cpu().to(torch.uint8))


# ==========================================  8-11. only one bank per step  ==

@pytest.mark.parametrize("task", ["repetition", "naming", "comprehension"])
def test_only_the_scheduled_task_bank_changes(transitioned, task):
    _, _, sep = transitioned
    while sep.task_for_step(sep.global_step) != task:
        sep.train_step()
    before = {t: opt_state(sep.task_optims[t]) for t in LR_TASKS}
    rec = sep.train_step()
    assert rec["task"] == task
    after = {t: opt_state(sep.task_optims[t]) for t in LR_TASKS}
    assert not same_state(before[task], after[task]), \
        f"the {task} bank did not update"
    for other in LR_TASKS:
        if other != task:
            assert same_state(before[other], after[other]), \
                f"a {task} step modified the {other} bank"


def test_grad_none_semantics_unchanged_under_separation(transitioned):
    """A C step reaches only the encoder side: no decoder/WM moments are
    created in the C bank, and no weight decay leaks onto untouched params."""
    _, _, sep = transitioned
    while sep.task_for_step(sep.global_step) != "comprehension":
        sep.train_step()
    before = params(sep.model)
    sep.train_step()
    moved = {k for k in before if not torch.equal(before[k], params(sep.model)[k])}
    for prefix in ("ltm.sem_to_h0.", "ltm.decoder.", "ltm.dec_to_premotor.",
                   "wm.encoder.", "wm.decoder.", "motor."):
        assert not any(k.startswith(prefix) for k in moved), prefix


def test_banks_diverge_and_divergence_is_reported(transitioned):
    _, _, sep = transitioned
    assert all(v == 0.0 for v in sep.moment_divergence().values()), \
        "clones must start identical"
    for _ in range(MACRO_CYCLE_STEPS * 3):
        sep.train_step()
    div = sep.moment_divergence()
    assert set(div) == {"m_div_RN", "m_div_RC", "m_div_NC"}
    assert all(v > 0 for v in div.values()), "banks never diverged"
    assert make(schedule=INTERLEAVED_123).moment_divergence() == {}


# =============================================  12-16. checkpoints, resume  ==

def test_checkpoint_saves_and_restores_all_three_banks(transitioned, tmp_path):
    _, _, sep = transitioned
    for _ in range(MACRO_CYCLE_STEPS):
        sep.train_step()
    p = tmp_path / "sep.pt"
    torch.save(sep.state_dict(), str(p))
    ck = torch.load(str(p), weights_only=False)
    assert ck["optimizer_policy"] == OPT_POLICY_SEPARATED
    assert set(ck["optimizer_states"]) == set(LR_TASKS)
    assert "optimizer_state_dict" not in ck

    back = make(schedule=INTERLEAVED_123,
                optimizer_policy=OPT_POLICY_SEPARATED)
    back.load_state_dict(ck, source="sep.pt")      # no declaration needed
    for t in LR_TASKS:
        assert same_state(opt_state(back.task_optims[t]),
                          opt_state(sep.task_optims[t])), t
    assert back.global_step == sep.global_step and back.cursors == sep.cursors


def test_requeue_does_not_reclone_or_redeclare(transitioned, tmp_path):
    _, _, sep = transitioned
    n_decl = len(sep.phase_transitions)
    for _ in range(MACRO_CYCLE_STEPS * 2):
        sep.train_step()
    assert not same_state(opt_state(sep.task_optims["repetition"]),
                          opt_state(sep.task_optims["naming"]))
    p = tmp_path / "mid.pt"
    torch.save(sep.state_dict(), str(p))

    back = make(schedule=INTERLEAVED_123, optimizer_policy=OPT_POLICY_SEPARATED)
    back.load_state_dict(torch.load(str(p), weights_only=False), source="m")
    assert len(back.phase_transitions) == n_decl, "re-declared on requeue"
    # banks stayed distinct: nothing was re-cloned from a shared state
    assert not same_state(opt_state(back.task_optims["repetition"]),
                          opt_state(back.task_optims["naming"]))


def test_wrong_mode_is_refused_in_both_directions(transitioned, tmp_path):
    _, ck_shared, sep = transitioned
    p = tmp_path / "sep.pt"
    torch.save(sep.state_dict(), str(p))
    ck_sep = torch.load(str(p), weights_only=False)
    # separated checkpoint under the shared policy: never, even if declared
    for allow in (False, True):
        with pytest.raises(RuntimeError, match="merge or regroup"):
            make(schedule=INTERLEAVED_123,
                 allow_phase_transition=allow).load_state_dict(
                     copy.deepcopy(ck_sep), source="t")
    # legacy checkpoints (no field) read as shared
    legacy = copy.deepcopy(ck_shared)
    legacy.pop("optimizer_policy", None)
    make(schedule=INTERLEAVED_123).load_state_dict(legacy, source="t")


def test_mid_run_resume_is_bitwise(tmp_path):
    out = str(tmp_path / "runs")
    launch(out, "src", 6, save_every=6)
    src = ckpt(out, "src", 6)
    launch(out, "whole", 18, resume=src, separated=True, phase=True,
           save_every=18)
    launch(out, "split", 11, resume=src, separated=True, phase=True,
           save_every=11)
    launch(out, "split", 18, resume=ckpt(out, "split", 11), separated=True,
           save_every=18)

    a = torch.load(ckpt(out, "whole", 18), map_location="cpu", weights_only=False)
    b = torch.load(ckpt(out, "split", 18), map_location="cpu", weights_only=False)
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    assert not [k for k in sa if not torch.equal(sa[k], sb[k])]
    for t in LR_TASKS:
        oa = a["optimizer_states"][t]["state"]
        ob = b["optimizer_states"][t]["state"]
        assert set(oa) == set(ob)
        for pid in oa:
            for k, v in oa[pid].items():
                if torch.is_tensor(v):
                    assert torch.equal(v, ob[pid][k]), (t, pid, k)
    assert a["cursors"] == b["cursors"]


# =============================================  17-22. branch, job, report  ==

def test_parent_run_is_never_modified(tmp_path):
    out = str(tmp_path / "runs")
    launch(out, "parent", 6, save_every=6, full_eval="6")
    launch(out, "parent", 18, save_every=6, resume=ckpt(out, "parent", 6))
    pdir = os.path.join(out, "parent")
    fp = {f: sha(os.path.join(pdir, f)) for f in os.listdir(pdir)
          if os.path.isfile(os.path.join(pdir, f))}
    fp.update({n: sha(os.path.join(pdir, "checkpoints", n))
               for n in os.listdir(os.path.join(pdir, "checkpoints"))})

    launch(out, "final6p", 12, resume=ckpt(out, "parent", 6), separated=True,
           phase=True, save_every=6)

    now = {f: sha(os.path.join(pdir, f)) for f in os.listdir(pdir)
           if os.path.isfile(os.path.join(pdir, f))}
    now.update({n: sha(os.path.join(pdir, "checkpoints", n))
                for n in os.listdir(os.path.join(pdir, "checkpoints"))})
    assert now == fp, "the parent run was modified"

    import csv
    brows = list(csv.DictReader(
        open(os.path.join(out, "final6p", "metrics.tsv")), delimiter="\t"))
    assert brows and min(int(r["step"]) for r in brows) > 6


def test_final6p_milestone_accounting():
    r_pass, c_pass = 463, 438
    r, n, c = RATIO_123
    expected = {
        100: (46_300, 277_800, 200.0, 317.1233),      # source
        110: (50_930, 305_580, 220.0, 348.8356),      # M6P-1
        120: (55_560, 333_360, 240.0, 380.5479),      # M6P-2
        130: (60_190, 361_140, 260.0, 412.2603),      # M6P-3
    }
    for r_exp, (cycles, step, n_exp, c_exp) in expected.items():
        assert r_exp * r_pass == cycles
        assert cycles * MACRO_CYCLE_STEPS == step
        assert cycles * n / r_pass == pytest.approx(n_exp)
        assert cycles * c / c_pass == pytest.approx(c_exp, abs=1e-3)
    assert expected[130][1] - expected[100][1] == 83_340


def test_final6p_slurm_job_contract():
    path = os.path.join(ROOT, "scripts", "cluster", "jeanzay",
                        "final6p_r100_sepmoments.slurm")
    text = open(path, encoding="utf-8").read()
    assert "EPOCHS=130" in text
    assert "FULL_EVAL_AT=305580,333360,361140" in text
    assert "FINAL_STEP=361140" in text and "PARENT_STEP=277800" in text
    assert "OPT_POLICY=task_separated_adamw" in text
    assert '--optimizer-policy "$OPT_POLICY"' in text
    assert "--phase-transition" in text
    assert 'RUN_ID="final6p_r100_sepmoments_seed${SEED}_${SUBSET_MODE}"' in text
    assert 'PARENT_RUN_ID="final3p_i123_seed${SEED}_${SUBSET_MODE}"' in text
    assert "FATAL: parent checkpoint" in text
    assert 'RESUME_FROM="$OWN_LATEST"' in text and 'RESUME_FROM="$PARENT_CKPT"' in text
    # the LR policy must NOT be touched: after R100 the two-stage schedule
    # already yields 1e-4 for every task, so no LR flag may appear
    for forbidden in ("--lr-repetition", "--lr-naming", "--lr-comprehension",
                      "--c-align-weight", "--lr-boundary-steps",
                      "--batch-size", "--allow-glove-fallback"):
        assert forbidden not in text, forbidden


def test_task_specific_lr_still_composes_with_separated_banks():
    tr = make(schedule=INTERLEAVED_123, optimizer_policy=OPT_POLICY_SEPARATED,
              task_lrs={"repetition": 1e-4, "naming": 3e-4,
                        "comprehension": 3e-4})
    for _ in range(MACRO_CYCLE_STEPS):
        task = tr.task_for_step(tr.global_step)
        rec = tr.train_step()
        assert rec["lr"] == pytest.approx(
            {"repetition": 1e-4, "naming": 3e-4, "comprehension": 3e-4}[task])


def test_reporting_identifies_the_optimizer_policy(tmp_path):
    shared = make(schedule=INTERLEAVED_123).resolved_settings()
    assert shared["optimizer_policy"] == OPT_POLICY_SHARED
    assert shared["optimizer_state_banks"] == 1
    assert "ONE AdamW" in shared["optimizer_convention"]
    sep = make(schedule=INTERLEAVED_123,
               optimizer_policy=OPT_POLICY_SEPARATED).resolved_settings()
    assert sep["optimizer_policy"] == OPT_POLICY_SEPARATED
    assert sep["optimizer_state_banks"] == 3
    assert "3 AdamW bank(s)" in sep["optimizer_convention"]
    assert sep["optimizer_bank_names"] == ["comprehension", "naming",
                                           "repetition"]
    assert sep["optimizer_bank_layout"] == {t: t for t in LR_TASKS}

    out = str(tmp_path / "runs")
    launch(out, "src", 6, save_every=6)
    launch(out, "b", 12, resume=ckpt(out, "src", 6), separated=True,
           phase=True, save_every=6)
    prov = json.load(open(os.path.join(out, "b", "provenance.json")))
    assert prov["optimizer_policy"] == OPT_POLICY_SEPARATED
    tr = prov["phase_transitions"][-1]
    assert tr["moment_initialization"] == MOMENT_INIT_CLONE
    assert tr["transition_step"] == 6
