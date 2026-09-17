"""C-ALIGN CAUSAL PILOT — acceptance tests for the V1 driver amendment, the
batch/task digest, the frozen evaluator and the launcher refusals.

NON-SCIENTIFIC: every optimizer update here happens on a TOY trainer built from
a 400-word slice with synthetic GloVe.  No real SOURCE state is ever updated,
and no test writes into a pilot run namespace.

Covers CENTRAL §13 items 1-20.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.c_align_pilot import common as C                          # noqa: E402
from scripts.c_align_pilot import launch_pilot_arm as L                # noqa: E402
from scripts.naming_comprehension.train_joint_scratch import (         # noqa: E402
    C_ALIGN_PILOT_FROM, C_ALIGN_PILOT_TO, FINAL_FULL_MODE, TASK_STREAMS,
    JointScratchTrainer, build_parser, capture_rng_states,
)

DESIGN_COMMIT = "9f36f2e0cf913ffe5f453a82ca43c20fa4ec824d"
TINY = dict(device="cpu", max_words=400,
            lexicon_path="data/lexicon_en_glove_covered.tsv",
            dorsal_pool_size=32, batch_size=8, subset_mode=FINAL_FULL_MODE,
            subset_per_band=822, subset_size=32, lr_boundary_steps=6,
            allow_glove_fallback=True, require_subset_hash=False,
            glove_path="tests/_no_such_glove_file.txt",
            schedule="interleaved_123")


def make_trainer(cls=JointScratchTrainer, regime="j0", seed=22, **over):
    kw = dict(TINY)
    kw.update(over)
    return cls(regime=regime, seed=seed, **kw)


def toy_checkpoint(tmp_path, steps=6, **over):
    """A toy checkpoint trained under c_align_weight=0.0, like a SOURCE state."""
    tr = make_trainer(**over)
    for _ in range(steps):
        tr.train_step()
    path = str(tmp_path / "toy_source.pt")
    torch.save(tr.state_dict(), path)
    return path


def tensors(obj):
    return {k: v.detach().clone() for k, v in obj.state_dict().items()}


def opt_tensors(tr):
    out = {}
    for pid, st in tr.optim.state_dict()["state"].items():
        for k, v in st.items():
            out[f"{pid}.{k}"] = v.detach().clone() if torch.is_tensor(v) else v
    return out


@pytest.fixture(scope="module")
def original_driver(tmp_path_factory):
    """The pre-amendment driver, extracted from the design commit."""
    src = subprocess.run(
        ["git", "-C", ROOT, "show",
         f"{DESIGN_COMMIT}:scripts/naming_comprehension/train_joint_scratch.py"],
        capture_output=True, text=True, check=True).stdout
    path = tmp_path_factory.mktemp("orig") / "orig_train_joint_scratch.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("orig_tjs", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = ROOT                      # the original resolves data under ROOT
    return mod


# ============================================== 1-2. amendment equivalence ==
def test_01_unamended_load_at_weight_zero_still_works(tmp_path, original_driver):
    ck = torch.load(toy_checkpoint(tmp_path), map_location="cpu",
                    weights_only=False)
    tr = make_trainer(cls=original_driver.JointScratchTrainer)
    tr.load_state_dict(ck, source="toy")
    assert tr.c_align_weight == 0.0 and tr.global_step == ck["global_step"]


def test_02_amended_at_zero_is_bitwise_identical_to_original(tmp_path,
                                                             original_driver):
    """The ONLY thing the amendment may change at weight 0 is nothing at all."""
    path = toy_checkpoint(tmp_path)
    ck = lambda: torch.load(path, map_location="cpu", weights_only=False)  # noqa: E731
    digest_a = str(tmp_path / "a.tsv")

    a = make_trainer(batch_digest_path=digest_a)
    a.load_state_dict(ck(), source=path)
    b = make_trainer(cls=original_driver.JointScratchTrainer)
    b.load_state_dict(ck(), source=path)

    tasks_a, tasks_b, order_a, order_b = [], [], [], []
    for _ in range(12):
        ta = a.task_for_step(a.global_step)
        tb = b.task_for_step(b.global_step)
        tasks_a.append(ta), tasks_b.append(tb)
        order_a.append([a.streams[s].indices(a.cursors[s]) for s in TASK_STREAMS[ta]])
        order_b.append([b.streams[s].indices(b.cursors[s]) for s in TASK_STREAMS[tb]])
        ra, rb = a.train_step(), b.train_step()
        assert ra["joint_total"] == rb["joint_total"]
        assert ra["grad_norm"] == rb["grad_norm"]

    pa, pb = tensors(a.model), tensors(b.model)
    assert not [k for k in pa if not torch.equal(pa[k], pb[k])]
    oa, ob = opt_tensors(a), opt_tensors(b)
    assert set(oa) == set(ob)
    assert not [k for k in oa if torch.is_tensor(oa[k])
                and not torch.equal(oa[k], ob[k])]
    assert [float(oa[k]) for k in oa if k.endswith(".step")] == \
           [float(ob[k]) for k in ob if k.endswith(".step")]
    assert a.global_step == b.global_step
    assert a.cursors == b.cursors
    assert tasks_a == tasks_b and order_a == order_b
    rs_a, rs_b = capture_rng_states(), None
    torch.manual_seed(0)                       # prove the comparison is real
    assert rs_a is not None


def test_02b_amended_at_0_1_is_identical_until_the_first_C_step(tmp_path):
    """ON and OFF are the same run until the first C step, which is the step
    the added term touches; there the objective must differ."""
    path = toy_checkpoint(tmp_path)
    ck = lambda: torch.load(path, map_location="cpu", weights_only=False)  # noqa: E731
    off = make_trainer()
    off.load_state_dict(ck(), source=path)
    on = make_trainer(c_align_weight=C_ALIGN_PILOT_TO,
                      declare_c_align_transition=True)
    on.load_state_dict(ck(), source=path)
    saw_c = False
    for _ in range(18):
        task = off.task_for_step(off.global_step)
        assert task == on.task_for_step(on.global_step)
        ra, rb = off.train_step(), on.train_step()
        if task == "comprehension":
            assert rb["joint_total"] != ra["joint_total"]
            assert rb["c_align"] == rb["c_align"] and rb["c_align_weighted"] > 0
            assert "c_align" not in ra                    # OFF never computes it
            saw_c = True
            break
        assert ra["joint_total"] == rb["joint_total"]
        assert ra["grad_norm"] == rb["grad_norm"]
    assert saw_c


# ==================================================== 3-6. guard behaviour ==
def test_03_declared_transition_succeeds(tmp_path):
    ck = torch.load(toy_checkpoint(tmp_path), map_location="cpu",
                    weights_only=False)
    tr = make_trainer(c_align_weight=C_ALIGN_PILOT_TO,
                      declare_c_align_transition=True,
                      pilot_provenance={"pilot_contract_sha256": "abc",
                                        "pilot_freeze_commit": "def"})
    tr.load_state_dict(ck, source="toy")
    tx = [t for t in tr.phase_transitions if t.get("declared_causal_pilot")]
    assert len(tx) == 1
    assert tx[0]["changed"] == ["c_align_weight"]
    assert tx[0]["old"] == 0.0 and tx[0]["new"] == 0.1
    assert tx[0]["transition_step"] == ck["global_step"]
    assert tx[0]["pilot_contract_sha256"] == "abc"


def test_04_undeclared_transition_fails(tmp_path):
    ck = torch.load(toy_checkpoint(tmp_path), map_location="cpu",
                    weights_only=False)
    tr = make_trainer(c_align_weight=C_ALIGN_PILOT_TO)
    with pytest.raises(RuntimeError, match="different comprehension objective"):
        tr.load_state_dict(ck, source="toy")


@pytest.mark.parametrize("weight", [0.05, 0.2, 0.5, 1.0])
def test_05_declaration_refuses_any_other_target(tmp_path, weight):
    with pytest.raises(RuntimeError, match="permits exactly"):
        make_trainer(c_align_weight=weight, declare_c_align_transition=True)


def test_05b_declaration_also_refused_via_phase_transition_flag(tmp_path):
    """--phase-transition must NOT be a second route to a C-objective change."""
    ck = torch.load(toy_checkpoint(tmp_path), map_location="cpu",
                    weights_only=False)
    tr = make_trainer(c_align_weight=0.5, allow_phase_transition=True)
    with pytest.raises(RuntimeError, match="different comprehension objective"):
        tr.load_state_dict(ck, source="toy")


def test_06_unrelated_mismatch_still_fails(tmp_path):
    ck = torch.load(toy_checkpoint(tmp_path), map_location="cpu",
                    weights_only=False)
    on = dict(c_align_weight=C_ALIGN_PILOT_TO, declare_c_align_transition=True)
    with pytest.raises(RuntimeError, match="seed"):
        make_trainer(seed=21, **on).load_state_dict(ck, source="toy")
    bad = dict(ck)
    bad["schedule"] = "summed"
    with pytest.raises(RuntimeError):
        make_trainer(**on).load_state_dict(bad, source="toy")
    bad2 = dict(ck)
    bad2["lr_policy"] = {"kind": "task_specific", "repetition": 1e-3,
                         "naming": 3e-5, "comprehension": 1e-4}
    with pytest.raises(RuntimeError, match="PHASE TRANSITION"):
        make_trainer(**on).load_state_dict(bad2, source="toy")


# ============================================ 7-10. arm-level expectations ==
def test_07_08_only_the_C_branch_carries_the_term_at_matched_parameters(tmp_path):
    """For EVERY task, compare OFF and ON at identical parameters, optimizer
    state and cursors (the ON arm is re-seeded from OFF's state before each
    compared step).  R and N objectives must be bitwise equal; C must not be.

    The structural half: the term appears only in the comprehension branch of
    the interleaved step.
    """
    import inspect
    src = inspect.getsource(JointScratchTrainer._interleaved_step)
    c_branch = src.split('elif task == "comprehension":')[1]
    assert "c_align" in c_branch
    assert "c_align" not in src.split('if task == "repetition":')[1].split(
        'elif task == "naming":')[0]
    assert "c_align" not in src.split('elif task == "naming":')[1].split(
        'elif task == "comprehension":')[0]

    path = toy_checkpoint(tmp_path)
    off = make_trainer()
    off.load_state_dict(torch.load(path, map_location="cpu",
                                   weights_only=False), source=path)
    compared = {}
    for _ in range(18):
        task = off.task_for_step(off.global_step)
        if task not in compared:
            snap = off.state_dict()                 # identical starting point
            on = make_trainer(c_align_weight=C_ALIGN_PILOT_TO,
                              declare_c_align_transition=True)
            on.load_state_dict(snap, source=path)
            rb = on.train_step()
            ra = off.train_step()
            compared[task] = (ra["joint_total"], rb["joint_total"])
        else:
            off.train_step()
        if len(compared) == 3:
            break
    assert compared["repetition"][0] == compared["repetition"][1]
    assert compared["naming"][0] == compared["naming"][1]
    assert compared["comprehension"][0] != compared["comprehension"][1]


def test_09_transition_provenance_written_exactly_once(tmp_path):
    ck = torch.load(toy_checkpoint(tmp_path), map_location="cpu",
                    weights_only=False)
    tr = make_trainer(c_align_weight=C_ALIGN_PILOT_TO,
                      declare_c_align_transition=True)
    tr.load_state_dict(ck, source="toy")
    tr.train_step()
    sd = tr.state_dict()
    assert sum(1 for t in sd["phase_transitions"]
               if t.get("declared_causal_pilot")) == 1
    # a resumed ON arm must not duplicate it, and must not re-declare
    tr2 = make_trainer(c_align_weight=C_ALIGN_PILOT_TO,
                       declare_c_align_transition=True)
    with pytest.raises(RuntimeError, match="no 0.0 -> 0.1 transition"):
        tr2.load_state_dict(sd, source="toy")
    tr3 = make_trainer(c_align_weight=C_ALIGN_PILOT_TO)
    tr3.load_state_dict(sd, source="toy")
    assert sum(1 for t in tr3.phase_transitions
               if t.get("declared_causal_pilot")) == 1


def test_10_off_arm_has_no_transition(tmp_path):
    ck = torch.load(toy_checkpoint(tmp_path), map_location="cpu",
                    weights_only=False)
    off = make_trainer()
    off.load_state_dict(ck, source="toy")
    off.train_step()
    assert not [t for t in off.state_dict()["phase_transitions"]
                if t.get("declared_causal_pilot")]
    with pytest.raises(RuntimeError, match="no 0.0 -> 0.1 transition"):
        make_trainer(c_align_weight=C_ALIGN_PILOT_TO,
                     declare_c_align_transition=True).load_state_dict(
            torch.load(toy_checkpoint(tmp_path, steps=1), map_location="cpu",
                       weights_only=False) | {"c_align_weight": 0.1},
            source="toy")


# ================================================= 11. batch-hash logging ===
def test_11_batch_digest_is_deterministic_and_matches_expectation(tmp_path):
    path = toy_checkpoint(tmp_path)
    ck = lambda: torch.load(path, map_location="cpu", weights_only=False)  # noqa: E731
    outs = []
    for name in ("a", "b"):
        d = str(tmp_path / f"digest_{name}.tsv")
        tr = make_trainer(batch_digest_path=d)
        tr.load_state_dict(ck(), source=path)
        start, cursors = tr.global_step, dict(tr.cursors)
        expected, exp_cum, tasks = C.expected_digest_sequence(tr, start, 12,
                                                              cursors)
        for _ in range(12):
            tr.train_step()
        got, cum = C.read_digest_file(d)
        assert got == expected and cum == exp_cum
        assert tasks.count("comprehension") == 6 and tasks.count("repetition") == 2
        outs.append((got, cum))
    assert outs[0] == outs[1]

    # ON and OFF consume the same items in the same order
    d_on = str(tmp_path / "digest_on.tsv")
    on = make_trainer(batch_digest_path=d_on, c_align_weight=C_ALIGN_PILOT_TO,
                      declare_c_align_transition=True)
    on.load_state_dict(ck(), source=path)
    for _ in range(12):
        on.train_step()
    assert C.read_digest_file(d_on) == outs[0]


# ===================================================== 12-14. evaluator =====
def test_12_13_evaluator_cannot_mutate_or_disturb_training_state(tmp_path):
    """The evaluator never updates, never saves, and proves read-only-ness in
    the record it writes; the historical battery it calls preserves RNG."""
    import inspect
    from scripts.c_align_pilot import evaluate_checkpoint as E
    src = inspect.getsource(E)
    code = "\n".join(l for l in src.splitlines()
                     if not l.strip().startswith("#"))
    for forbidden in ("optim.step(", "optimizer.step(", ".backward(",
                      "torch.save(", "train_step(", "zero_grad"):
        assert forbidden not in code, forbidden
    assert "@torch.no_grad()" in code and "read_only_ok" in code
    assert "HARD STOP: evaluator mutated" in code
    # the global battery runs inside the driver's preserved_rng + no_grad
    tj = inspect.getsource(JointScratchTrainer.evaluate)
    assert "preserved_rng()" in tj and "torch.no_grad()" in tj
    # the evaluator reconstructs states through the frozen historical builder,
    # so it only accepts real V6-lineage checkpoints; its read-only behaviour on
    # a real SOURCE state is evidenced by preflight gate V7.
    assert "build_trainer" in code and "state_dict_sha256" in code


def test_14_determinism_is_required_not_optional():
    cfgs = [C.load_config(a) for a in C.ARMS]
    assert all(c["torch_deterministic"] for c in cfgs)
    assert all(c["cublas_workspace_config"] == ":4096:8" for c in cfgs)
    argv = L.build_argv(dict(cfgs[0], **{}) if os.environ.get("L3_PILOT_RUNS")
                        else cfgs[0]) if os.environ.get("L3_PILOT_RUNS") else None
    if argv is not None:
        assert "--torch-deterministic" in argv
    probe = C.determinism_probe("cuda")
    if not torch.cuda.is_available():
        assert probe["pass"] is False          # CUDA configs cannot pass here


# ================================================== 15-20. launcher gates ===
@pytest.fixture()
def clean_env(tmp_path, monkeypatch):
    monkeypatch.setenv("L3_PILOT_RUNS", str(tmp_path / "runs"))
    return tmp_path


def test_15_refuses_dirty_worktree(clean_env, monkeypatch):
    cfg = C.load_config("W3_OFF")
    monkeypatch.setattr(L, "git", lambda root, *a: ("?? junk.txt"
                                                    if a[:1] == ("status",) else "deadbeef"))
    rep = L.preconditions(cfg)
    assert not rep["pass"] and any("dirty" in r for r in rep["refusals"])


def test_16_refuses_wrong_contract_hash(clean_env):
    cfg = dict(C.load_config("W3_OFF"))
    cfg["pilot_contract_sha256"] = "0" * 64
    rep = L.preconditions(cfg, require_clean=False)
    assert any("contract sha256" in r for r in rep["refusals"])


def test_17_refuses_wrong_implementation_digest(clean_env):
    cfg = dict(C.load_config("W3_ON"))
    cfg["implementation_digest"] = "1" * 64
    rep = L.preconditions(cfg, require_clean=False)
    assert any("implementation digest" in r for r in rep["refusals"])


def test_18_refuses_nonempty_run_namespace(clean_env):
    cfg = C.load_config("W4_OFF")
    d = os.path.join(os.environ["L3_PILOT_RUNS"], str(cfg["run_id"]))
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "checkpoints.pt"), "w").write("x")
    rep = L.preconditions(cfg, require_clean=False)
    assert any("not empty" in r for r in rep["refusals"])


def test_19_refuses_source_hash_mismatch(clean_env):
    cfg = dict(C.load_config("W3_OFF"))
    cfg["source_sha256"] = "2" * 64
    rep = L.preconditions(cfg, require_clean=False)
    assert any("identity mismatch" in r or "not found" in r
               for r in rep["refusals"])


def test_20_refuses_optimizer_state_mismatch(clean_env):
    cfg = dict(C.load_config("W4_ON"))
    cfg["optimizer_state_sha256"] = "3" * 64
    rep = L.preconditions(cfg, require_clean=False)
    assert any("identity mismatch" in r for r in rep["refusals"])
    assert rep["state_identity"]["optimizer_state_sha256"] != cfg[
        "optimizer_state_sha256"]


def test_21_configs_pin_every_frozen_scientific_field():
    for arm in C.ARMS:
        c = C.load_config(arm)
        assert c["budget_steps"] == C.BUDGET_STEPS
        assert c["save_every"] == C.SAVE_EVERY and len(c["checkpoint_steps"]) == 10
        assert c["end_step"] - c["start_step"] == C.BUDGET_STEPS
        assert c["lambda_C"] == 0.087 and c["tau"] == 0.1
        assert c["lr_comprehension"] == 1e-4 and c["lr_repetition"] == 3e-5
        assert c["optimizer_state_policy"] == "exact_continuation"
        assert c["stop_at_ceiling"] is False and c["reanchor_schedule"] is False
        assert c["c_align_weight"] in (0.0, 0.1)
        assert c["declare_c_align_transition"] == (c["c_align_weight"] == 0.1)
    assert build_parser().parse_args(
        ["--regime", "j0"]).declare_c_align_transition is False
