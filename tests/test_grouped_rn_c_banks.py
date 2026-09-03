"""Acceptance tests for FINAL-7P: grouped RN | C optimizer banks.

ONE shared network, TWO AdamW banks over the SAME Parameter objects:

    optimizer_RN  steps on repetition (with its attached pool loss) AND naming
    optimizer_C   steps on comprehension

Motivated by the asymmetric parameter overlap -- comprehension never trains
the LTM decoder core, so under the historical shared policy the decoder-side
statistics were an R+N history.  FINAL-6P broke that; FINAL-7P restores it
while keeping C's optimizer memory separate from R on the encoder side.

Branches from the same R100 checkpoint as the shared control and FINAL-6P, so
the three optimizer topologies form a matched family.
"""
from __future__ import annotations

import copy
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
    FINAL_FULL_MODE, INTERLEAVED_123, LR_TASKS, MACRO_CYCLE_STEPS,
    MOMENT_INIT_CLONE_GROUPED, OPT_BANK_MAP, OPT_POLICY_GROUPED_RN_C,
    OPT_POLICY_SEPARATED, OPT_POLICY_SHARED, RATIO_123, SUMMED_SCHEDULE,
    JointScratchTrainer, bank_names_for, main,
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


def launch(out, run_id, steps, *, resume=None, policy=None, phase=False,
           save_every=None, full_eval=None):
    argv = SMOKE + ["--out-dir", out, "--run-id", run_id,
                    "--max-steps", str(steps),
                    "--save-every", str(save_every or steps)]
    if policy:
        argv += ["--optimizer-policy", policy]
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


# ===========================================  1-4. policy and layout  ======

def test_grouped_policy_exists_with_two_banks():
    tr = make(schedule=INTERLEAVED_123,
              optimizer_policy=OPT_POLICY_GROUPED_RN_C)
    assert bank_names_for(OPT_POLICY_GROUPED_RN_C) == ["comprehension", "rn"]
    assert set(tr.banks) == {"rn", "comprehension"}
    assert tr.task_to_bank == {"repetition": "rn", "naming": "rn",
                               "comprehension": "comprehension"}
    # R and N share ONE optimizer object; C owns a different one
    assert tr.optimizer_for("repetition") is tr.optimizer_for("naming")
    assert tr.optimizer_for("comprehension") is not tr.optimizer_for("naming")


def test_default_and_separated_policies_are_untouched():
    shared = make(schedule=INTERLEAVED_123)
    assert shared.optimizer_policy == OPT_POLICY_SHARED
    assert shared.banks is None and shared.task_optims is None
    sep = make(schedule=INTERLEAVED_123,
               optimizer_policy=OPT_POLICY_SEPARATED)
    assert set(sep.banks) == set(LR_TASKS)
    assert sep.task_to_bank == OPT_BANK_MAP[OPT_POLICY_SEPARATED]
    assert len({id(sep.optimizer_for(t)) for t in LR_TASKS}) == 3


def test_grouped_requires_an_interleaved_schedule():
    with pytest.raises(RuntimeError, match="interleaved"):
        make(schedule=SUMMED_SCHEDULE,
             optimizer_policy=OPT_POLICY_GROUPED_RN_C)


def test_both_banks_reference_the_same_network_parameters():
    """No module or tensor is duplicated: the banks hold the very same
    Parameter objects as the model."""
    tr = make(schedule=INTERLEAVED_123,
              optimizer_policy=OPT_POLICY_GROUPED_RN_C)
    model_ids = {id(p) for p in tr.model.parameters()}
    for name, opt in tr.banks.items():
        got = {id(p) for g in opt.param_groups for p in g["params"]}
        assert got == model_ids, f"bank {name} does not own the model's params"
    # the two banks see identical parameter sets -- one network, two memories
    a, b = (tr.banks["rn"], tr.banks["comprehension"])
    assert [id(p) for g in a.param_groups for p in g["params"]] == \
           [id(p) for g in b.param_groups for p in g["params"]]


# ============================  5-8. transition, cloning, aliasing, purity  ==

@pytest.fixture
def grouped(tmp_path):
    """A shared checkpoint with real moments, taken into the grouped policy."""
    src = make(schedule=INTERLEAVED_123)
    for _ in range(MACRO_CYCLE_STEPS * 2):
        src.train_step()
    p = tmp_path / "shared.pt"
    torch.save(src.state_dict(), str(p))
    ck = torch.load(str(p), weights_only=False)
    tr = make(schedule=INTERLEAVED_123,
              optimizer_policy=OPT_POLICY_GROUPED_RN_C,
              allow_phase_transition=True)
    tr.load_state_dict(copy.deepcopy(ck), source="shared.pt")
    return src, ck, tr


def test_explicit_transition_required_from_the_shared_r100_checkpoint(grouped):
    _, ck, _ = grouped
    with pytest.raises(RuntimeError, match="PHASE TRANSITION"):
        make(schedule=INTERLEAVED_123,
             optimizer_policy=OPT_POLICY_GROUPED_RN_C).load_state_dict(
                 copy.deepcopy(ck), source="t")


def test_shared_state_cloned_bitwise_into_both_banks(grouped):
    src, _, tr = grouped
    shared = opt_state(src.optim)
    assert shared, "the source had no optimizer state to clone"
    for name in ("rn", "comprehension"):
        assert same_state(opt_state(tr.banks[name]), shared), \
            f"bank {name} is not a bitwise copy of the shared state"
    rec = tr.phase_transitions[-1]
    assert rec["changed"] == ["optimizer_policy"]
    assert rec["old_optimizer_policy"] == OPT_POLICY_SHARED
    assert rec["new_optimizer_policy"] == OPT_POLICY_GROUPED_RN_C
    assert rec["moment_initialization"] == MOMENT_INIT_CLONE_GROUPED
    assert rec["bank_layout"] == {"repetition": "rn", "naming": "rn",
                                  "comprehension": "comprehension"}
    assert rec["old_lr_policy"] == rec["new_lr_policy"], "LR must not change"
    # step counters preserved, not reset
    for name in ("rn", "comprehension"):
        steps = [v for (pid, k), v in opt_state(tr.banks[name]).items()
                 if k == "step"]
        assert steps and all(float(x) > 0 for x in steps)


def test_banks_do_not_alias(grouped):
    _, _, tr = grouped
    rn, c = tr.banks["rn"], tr.banks["comprehension"]
    p_rn = next(iter(rn.state.values()))["exp_avg"]
    p_c = next(iter(c.state.values()))["exp_avg"]
    assert p_rn.data_ptr() != p_c.data_ptr(), "banks share exp_avg storage"
    before_c = opt_state(c)
    p_rn.add_(1.0)
    assert same_state(opt_state(c), before_c), "mutating RN changed C"


def test_transition_touches_nothing_but_the_optimizer(grouped):
    src, ck, tr = grouped
    ps, pn = params(src.model), params(tr.model)
    assert not [k for k in ps if not torch.equal(ps[k], pn[k])]
    assert tr.global_step == int(ck["global_step"])
    assert tr.cursors == {k: int(v) for k, v in ck["cursors"].items()}
    assert torch.equal(torch.get_rng_state(),
                       ck["rng_states"]["torch"].cpu().to(torch.uint8))


# ==================================  9-11. bank ownership per task step  ===

@pytest.mark.parametrize("task,owner,other", [
    ("naming", "rn", "comprehension"),
    ("repetition", "rn", "comprehension"),
    ("comprehension", "comprehension", "rn"),
])
def test_only_the_owning_bank_changes(grouped, task, owner, other):
    _, _, tr = grouped
    while tr.task_for_step(tr.global_step) != task:
        tr.train_step()
    before = {n: opt_state(o) for n, o in tr.banks.items()}
    rec = tr.train_step()
    assert rec["task"] == task
    after = {n: opt_state(o) for n, o in tr.banks.items()}
    assert not same_state(before[owner], after[owner]), \
        f"the {owner} bank did not update on a {task} step"
    assert same_state(before[other], after[other]), \
        f"a {task} step modified the {other} bank"


def test_r_and_n_steps_share_one_moment_history(grouped):
    """The defining property of the grouped policy."""
    _, _, tr = grouped
    seen = set()
    while len(seen) < 2:
        task = tr.task_for_step(tr.global_step)
        before = opt_state(tr.banks["rn"])
        tr.train_step()
        if task in ("repetition", "naming"):
            assert not same_state(before, opt_state(tr.banks["rn"]))
            seen.add(task)
    assert seen == {"repetition", "naming"}


def test_grad_none_semantics_unchanged(grouped):
    _, _, tr = grouped
    while tr.task_for_step(tr.global_step) != "comprehension":
        tr.train_step()
    before = params(tr.model)
    tr.train_step()
    moved = {k for k in before if not torch.equal(before[k], params(tr.model)[k])}
    for prefix in ("ltm.sem_to_h0.", "ltm.decoder.", "ltm.dec_to_premotor.",
                   "wm.encoder.", "wm.decoder.", "motor."):
        assert not any(k.startswith(prefix) for k in moved), prefix


# ==========================================  12-16. checkpoints / resume  ==

def test_checkpoint_saves_two_named_banks_and_reloads_exactly(grouped, tmp_path):
    _, _, tr = grouped
    for _ in range(MACRO_CYCLE_STEPS):
        tr.train_step()
    p = tmp_path / "grouped.pt"
    torch.save(tr.state_dict(), str(p))
    ck = torch.load(str(p), weights_only=False)
    assert ck["optimizer_policy"] == OPT_POLICY_GROUPED_RN_C
    assert set(ck["optimizer_states"]) == {"rn", "comprehension"}
    assert ck["optimizer_bank_layout"]["naming"] == "rn"
    assert "optimizer_state_dict" not in ck

    back = make(schedule=INTERLEAVED_123,
                optimizer_policy=OPT_POLICY_GROUPED_RN_C)
    back.load_state_dict(ck, source="grouped.pt")     # no declaration needed
    for name in ("rn", "comprehension"):
        assert same_state(opt_state(back.banks[name]), opt_state(tr.banks[name]))
    assert back.global_step == tr.global_step and back.cursors == tr.cursors
    assert len(back.phase_transitions) == len(tr.phase_transitions)


def test_requeue_never_reclones(grouped, tmp_path):
    _, _, tr = grouped
    for _ in range(MACRO_CYCLE_STEPS * 2):
        tr.train_step()
    assert not same_state(opt_state(tr.banks["rn"]),
                          opt_state(tr.banks["comprehension"])), \
        "banks should have specialised before the requeue test"
    p = tmp_path / "mid.pt"
    torch.save(tr.state_dict(), str(p))
    back = make(schedule=INTERLEAVED_123,
                optimizer_policy=OPT_POLICY_GROUPED_RN_C)
    back.load_state_dict(torch.load(str(p), weights_only=False), source="m")
    assert not same_state(opt_state(back.banks["rn"]),
                          opt_state(back.banks["comprehension"]))


def test_cross_mode_translations_are_refused(grouped, tmp_path):
    """Only shared -> multi-bank is sanctioned; merging or regrouping
    specialised banks is refused with or without the flag."""
    _, ck_shared, tr = grouped
    p = tmp_path / "grouped.pt"
    torch.save(tr.state_dict(), str(p))
    ck_grouped = torch.load(str(p), weights_only=False)
    for policy in (OPT_POLICY_SHARED, OPT_POLICY_SEPARATED):
        for allow in (False, True):
            with pytest.raises(RuntimeError, match="merge or regroup"):
                make(schedule=INTERLEAVED_123, optimizer_policy=policy,
                     allow_phase_transition=allow).load_state_dict(
                         copy.deepcopy(ck_grouped), source="t")
    # legacy / shared checkpoints still read as shared
    legacy = copy.deepcopy(ck_shared)
    legacy.pop("optimizer_policy", None)
    make(schedule=INTERLEAVED_123).load_state_dict(legacy, source="t")


def test_interrupted_run_equals_uninterrupted(tmp_path):
    out = str(tmp_path / "runs")
    launch(out, "parent", 6, save_every=6)
    src = ckpt(out, "parent", 6)
    launch(out, "whole", 18, resume=src, policy=OPT_POLICY_GROUPED_RN_C,
           phase=True, save_every=18)
    launch(out, "split", 11, resume=src, policy=OPT_POLICY_GROUPED_RN_C,
           phase=True, save_every=11)
    launch(out, "split", 18, resume=ckpt(out, "split", 11),
           policy=OPT_POLICY_GROUPED_RN_C, save_every=18)

    a = torch.load(ckpt(out, "whole", 18), map_location="cpu", weights_only=False)
    b = torch.load(ckpt(out, "split", 18), map_location="cpu", weights_only=False)
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    assert not [k for k in sa if not torch.equal(sa[k], sb[k])]
    for name in ("rn", "comprehension"):
        oa = a["optimizer_states"][name]["state"]
        ob = b["optimizer_states"][name]["state"]
        assert set(oa) == set(ob)
        for pid in oa:
            for k, v in oa[pid].items():
                if torch.is_tensor(v):
                    assert torch.equal(v, ob[pid][k]), (name, pid, k)
    assert a["cursors"] == b["cursors"]


# =======================================  17-21. branch, budget, reporting ==

def test_parent_is_read_only_and_branch_has_no_parent_future(tmp_path):
    out = str(tmp_path / "runs")
    launch(out, "parent", 6, save_every=6, full_eval="6")
    launch(out, "parent", 18, save_every=6, resume=ckpt(out, "parent", 6))
    pdir = os.path.join(out, "parent")
    fp = {f: sha(os.path.join(pdir, f)) for f in os.listdir(pdir)
          if os.path.isfile(os.path.join(pdir, f))}
    fp.update({n: sha(os.path.join(pdir, "checkpoints", n))
               for n in os.listdir(os.path.join(pdir, "checkpoints"))})

    launch(out, "final7p", 12, resume=ckpt(out, "parent", 6),
           policy=OPT_POLICY_GROUPED_RN_C, phase=True, save_every=6)

    now = {f: sha(os.path.join(pdir, f)) for f in os.listdir(pdir)
           if os.path.isfile(os.path.join(pdir, f))}
    now.update({n: sha(os.path.join(pdir, "checkpoints", n))
                for n in os.listdir(os.path.join(pdir, "checkpoints"))})
    assert now == fp, "the parent run was modified"
    rows = list(csv.DictReader(
        open(os.path.join(out, "final7p", "metrics.tsv")), delimiter="\t"))
    assert rows and min(int(r["step"]) for r in rows) > 6


def test_final7p_milestone_accounting():
    r_pass, c_pass = 463, 438
    r, n, c = RATIO_123
    expected = {                    # R exp -> (cycles, step, N exp, C exp)
        100: (46_300, 277_800, 200.0, 317.1233),      # source
        110: (50_930, 305_580, 220.0, 348.8356),      # G1
        120: (55_560, 333_360, 240.0, 380.5479),      # G2
        130: (60_190, 361_140, 260.0, 412.2603),      # G3
    }
    for r_exp, (cycles, step, n_exp, c_exp) in expected.items():
        assert r_exp * r_pass == cycles
        assert cycles * MACRO_CYCLE_STEPS == step
        assert cycles * n / r_pass == pytest.approx(n_exp)
        assert cycles * c / c_pass == pytest.approx(c_exp, abs=1e-3)
    assert expected[130][1] - expected[100][1] == 83_340


def test_final7p_slurm_job_contract():
    path = os.path.join(ROOT, "scripts", "cluster", "jeanzay",
                        "final7p_r100_rn_c_banks.slurm")
    text = open(path, encoding="utf-8").read()
    assert "EPOCHS=130" in text
    assert "FULL_EVAL_AT=305580,333360,361140" in text
    assert "FINAL_STEP=361140" in text and "PARENT_STEP=277800" in text
    assert "OPT_POLICY=grouped_rn_c_adamw" in text
    assert '--optimizer-policy "$OPT_POLICY"' in text
    assert "--phase-transition" in text
    assert 'RUN_ID="final7p_r100_rn_c_banks_seed${SEED}_${SUBSET_MODE}"' in text
    assert 'PARENT_RUN_ID="final3p_i123_seed${SEED}_${SUBSET_MODE}"' in text
    assert "FATAL: parent checkpoint" in text
    assert 'RESUME_FROM="$OWN_LATEST"' in text
    assert 'RESUME_FROM="$PARENT_CKPT"' in text
    # LR must not be touched: after R100 the two-stage policy already gives 1e-4
    for forbidden in ("--lr-repetition", "--lr-naming", "--lr-comprehension",
                      "--c-align-weight", "--lr-boundary-steps",
                      "--batch-size", "--dorsal-pool-size",
                      "--allow-glove-fallback", "--no-subset-hash-check"):
        assert forbidden not in text, forbidden
    assert "--time=02:00:00" in text


def test_reporting_is_truthful_about_the_grouped_policy(tmp_path):
    s = make(schedule=INTERLEAVED_123,
             optimizer_policy=OPT_POLICY_GROUPED_RN_C).resolved_settings()
    assert s["optimizer_policy"] == OPT_POLICY_GROUPED_RN_C
    assert s["optimizer_state_banks"] == 2
    assert s["optimizer_bank_names"] == ["comprehension", "rn"]
    assert s["optimizer_bank_layout"]["naming"] == "rn"
    assert "2 AdamW bank(s)" in s["optimizer_convention"]

    out = str(tmp_path / "runs")
    launch(out, "parent", 6, save_every=6)
    launch(out, "g", 12, resume=ckpt(out, "parent", 6),
           policy=OPT_POLICY_GROUPED_RN_C, phase=True, save_every=6)
    prov = json.load(open(os.path.join(out, "g", "provenance.json")))
    assert prov["optimizer_policy"] == OPT_POLICY_GROUPED_RN_C
    assert prov["optimizer_bank_layout"] == {"repetition": "rn",
                                             "naming": "rn",
                                             "comprehension": "comprehension"}
    tr = prov["phase_transitions"][-1]
    assert tr["moment_initialization"] == MOMENT_INIT_CLONE_GROUPED
    assert tr["transition_step"] == 6
    # R and N genuinely share a bank, so their reported divergence is 0
    rows = list(csv.DictReader(open(os.path.join(out, "g", "metrics.tsv")),
                               delimiter="\t"))
    div = [r for r in rows if r.get("m_div_RN") not in (None, "")]
    if div:
        assert float(div[-1]["m_div_RN"]) == 0.0
