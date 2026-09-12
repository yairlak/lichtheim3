"""V7 fresh step-0 orchestration: gates that must PASS before submission.

Nothing here is science.  The tests pin the orchestration to (a) the driver's
own arithmetic, (b) the archived checkpoint-level provenance of the seed-19/20
witness lineage, (c) the frozen V6 first-hit selector, and (d) the REAL driver
on a tiny fixture, where the leg/flag derivation is executed end to end and the
resulting `phase_transitions` ledger is checked entry by entry.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension import fresh_ceiling_v7 as v7          # noqa: E402
from scripts.naming_comprehension import first_hit_selector as fhs       # noqa: E402
from scripts.naming_comprehension.train_joint_scratch import (           # noqa: E402
    CounterStream, derive_schedule_seed, derive_stream_seeds, LR_BOUNDARY_STEPS,
    CANONICAL_N_WORDS, CANONICAL_BATCH_SIZE, cycle_steps_for, INTERLEAVED_123,
    two_stage_lr_policy, task_lr_policy, JointScratchTrainer, macro_cycle_n,
    RATIO_123, EXPECTED_CANONICAL_C_N, EXPECTED_CANONICAL_C_HASH)

DRIVER = os.path.join(ROOT, "scripts", "naming_comprehension", "train_joint_scratch.py")
JOB = os.path.join(ROOT, "scripts", "cluster", "jeanzay", "fresh_ceiling_v7.slurm")
ARCH = os.path.join(os.path.dirname(ROOT), "archives")


# ------------------------------------------------------------ arithmetic ---
def test_u_arithmetic_matches_the_driver():
    per_epoch = CounterStream("repetition", list(range(CANONICAL_N_WORDS)),
                              CANONICAL_BATCH_SIZE, seed=1).per_epoch
    assert per_epoch == 463
    assert per_epoch * cycle_steps_for(INTERLEAVED_123) == v7.STEPS_PER_U == 2778
    assert v7.DET_STEP == 13_890 and v7.END_STEP == 10_834_200 and v7.N_DET == 780
    assert v7.steps(3825) == 10_625_850 and v7.steps(3040) == 8_445_120   # V6 witnesses


def test_grid_is_exact():
    g = v7.grid()
    assert g[0] == 0 and g[-1] == v7.END_STEP and len(g) == 781
    assert all(b - a == v7.DET_STEP for a, b in zip(g, g[1:]))
    assert all(s % 6 == 0 for s in g)                     # every point on a cycle boundary
    assert 10_625_850 in g and 8_445_120 in g
    assert v7.END_STEP % v7.SAVE_EVERY == 0


def test_seed_derivations_match_the_driver():
    for s in (31, 32, 33, 34, 35):
        assert derive_schedule_seed(s) == s * v7.STREAM_SEED_STRIDE + v7.SCHEDULE_SEED_OFFSET
        assert derive_stream_seeds(s) == {n: s * v7.STREAM_SEED_STRIDE + o
                                          for n, o in v7.STREAM_SEED_OFFSET.items()}
    assert LR_BOUNDARY_STEPS == v7.LR_BOUNDARY_REP_BATCHES == 46_300
    assert two_stage_lr_policy(LR_BOUNDARY_STEPS) == v7.TWO_STAGE
    assert task_lr_policy(3e-5, 3e-5, 3e-5) == v7.TASK_3E5
    assert task_lr_policy(3e-5, 3e-5, 1e-4) == v7.TASK_CHIGH
    assert EXPECTED_CANONICAL_C_N == v7.EXPECTED["comprehension_population_n"]
    assert EXPECTED_CANONICAL_C_HASH == v7.EXPECTED["comprehension_population_sha256"]


# ---------------------------------------------------------------- ledger ---
def test_legs_partition_the_horizon_on_grid_and_save_boundaries():
    L = v7.legs()
    assert L[0]["start"] == 0 and L[-1]["end"] == v7.END_STEP
    for a, b in zip(L, L[1:]):
        assert a["end"] == b["start"]
        assert b["start"] % v7.DET_STEP == 0 and b["start"] % v7.SAVE_EVERY == 0
    assert [l["start"] for l in L] == [0, 2_083_500, 2_361_300, 3_333_600, 3_889_200, 5_556_000, 8_334_000]
    assert [l["reanchor"] for l in L] == [False, False, False, True, True, True, True]


def test_reanchor_changes_the_task_order_so_it_is_trajectory_relevant():
    sseed = derive_schedule_seed(31)
    step = 3_333_600
    order_old = macro_cycle_n(RATIO_123, sseed, (step - 0) // 6)
    order_new = macro_cycle_n(RATIO_123, sseed, (step - step) // 6)
    # not a tautology: with anchor 0 the cycle index is 555,600, with the
    # re-anchor it is 0; the two deterministic shuffles differ for this seed.
    assert order_old != order_new


@pytest.mark.parametrize("seed", [19, 20])
def test_ledger_equals_the_archived_witness_lineage(seed):
    """The V7 ledger must reproduce, entry for entry, the phase_transitions the
    seed-19/20 FIRST-HIT sources actually carried (archived SETTLE/CHIGH
    provenance, which the V6 sources inherited unchanged)."""
    cand = [os.path.join(ARCH, "settle_u3600_20260910", "runs", f"final_settle_ctrl_h512_s{seed}", "provenance.json"),
            os.path.join(ARCH, "prospective_rn_detector_v6_20260912", f"task{ {19: 0, 20: 2}[seed] }_seed{seed}", "provenance.json")]
    files = [c for c in cand if os.path.exists(c)]
    if not files:
        pytest.skip("archives not present on this machine")
    for f in files:
        got = json.load(open(f))["phase_transitions"]
        want = v7.expected_transitions_before(v7.END_STEP)
        assert [g["transition_step"] for g in got] == [w["transition_step"] for w in want]
        for g, w in zip(got, want):
            assert list(g["changed"]) == w["changed"]
            assert g["new_lr_policy"] == w["new_lr_policy"]
            assert g["old_lr_policy"] == w["old_lr_policy"]
            assert g["schedule_anchor_step"] == w["schedule_anchor_step"]
            assert g.get("old_schedule_anchor_step", 0) == w["old_schedule_anchor_step"]
            assert g["moment_initialization"] == "unchanged"
            assert g["old_dec_weight"] == g["new_dec_weight"] == 0.5
            assert g["old_optimizer_policy"] == g["new_optimizer_policy"] == "shared_adamw"
            assert g["old_schedule_ratio"] == g["new_schedule_ratio"] == [1, 2, 3]


def test_expected_state_at_every_boundary():
    assert v7.expected_anchor_at(3_333_600) == 0            # boundary ckpt carries the OLD anchor
    assert v7.expected_anchor_at(3_333_606) == 3_333_600
    assert v7.expected_anchor_at(10_625_850) == 8_334_000   # seed-19 witness
    assert v7.expected_anchor_at(8_445_120) == 8_334_000    # seed-20 witness
    assert v7.expected_lr_policy_at(2_083_500) == v7.TWO_STAGE
    assert v7.expected_lr_policy_at(2_083_506) == v7.TASK_3E5
    assert v7.expected_lr_policy_at(2_361_306) == v7.TASK_CHIGH
    assert len(v7.expected_transitions_before(2_083_500)) == 0
    assert len(v7.expected_transitions_before(2_083_506)) == 1
    assert len(v7.expected_transitions_before(v7.END_STEP)) == 6


# ------------------------------------------------------------------ plan ---
def test_plan_first_launch_boundaries_requeue_and_complete():
    p = v7.plan(None)
    assert p["status"] == "FIRST_LAUNCH" and p["resume"] is None and p["max_steps"] == 2_083_500
    assert p["flags"] == ["--eval-at-start"] and not p["declare"]
    p = v7.plan(2_083_500)
    assert p["status"] == "BOUNDARY_LAUNCH" and p["declare"] and not p["reanchor"]
    assert "--phase-transition" in p["flags"] and "--reanchor-schedule" not in p["flags"]
    assert p["flags"][:6] == ["--lr-repetition", "3e-05", "--lr-naming", "3e-05", "--lr-comprehension", "3e-05"]
    p = v7.plan(2_361_300)
    assert p["declare"] and p["flags"][5] == "0.0001" and "--reanchor-schedule" not in p["flags"]
    for b in (3_333_600, 3_889_200, 5_556_000, 8_334_000):
        p = v7.plan(b)
        assert p["status"] == "BOUNDARY_LAUNCH" and p["reanchor"] and p["declare"]
        assert p["flags"][-2:] == ["--reanchor-schedule", "--phase-transition"]
    p = v7.plan(3_347_490)                       # 5u inside L4: plain requeue
    assert p["status"] == "REQUEUE" and not p["declare"] and not p["reanchor"]
    assert "--phase-transition" not in p["flags"] and "--reanchor-schedule" not in p["flags"]
    assert "--eval-at-start" not in p["flags"] and p["max_steps"] == 3_889_200
    p = v7.plan(13_890)                          # 5u inside L1: no LR flags at all
    assert p["flags"] == [] and p["max_steps"] == 2_083_500
    assert v7.plan(v7.END_STEP)["status"] == "COMPLETE"
    with pytest.raises(ValueError):
        v7.plan(-6)


# ------------------------------------------------------------ selection ----
def _row(step, rcan, rfree, nexact, c=10):
    return {"step": str(step), "full_rep_errors": str(rcan), "full_rep_freear_errors": str(rfree),
            "full_naming_exact": repr(nexact), "full_comp_errors": str(c)}


def test_eligibility_agrees_with_the_frozen_selector(tmp_path):
    rows = [_row(13890, 3, 3, 0.999), _row(27780, 0, 0, 1.0, c=40), _row(41670, 0, 0, 1.0, c=5),
            _row(55560, 0, 1, 1.0), _row(69450, 0, 0, 0.9999)]
    hdr = list(rows[0].keys())
    m = tmp_path / "metrics.tsv"
    m.write_text("\t".join(hdr) + "\n" + "\n".join("\t".join(r[k] for k in hdr) for r in rows) + "\n")
    pts = fhs.detector_points(str(m))
    frozen = next(p for p in pts if fhs.eligible(p))
    ours = v7.first_hit(v7.read_rows(str(m)))
    assert int(ours["step"]) == frozen["step"] == 27780          # FIRST, not min-C (41670)
    assert [v7.eligible(r) for r in rows] == [fhs.eligible(p) for p in pts]


def test_dedupe_keeps_the_last_row_per_detector_step(tmp_path):
    hdr = ["step", "rep_ltm", "full_rep_errors", "full_rep_freear_errors", "full_naming_exact", "full_comp_errors"]
    rows = [["13890", "0.5", "0", "0", "1.0", "9"],      # orphan: eligible but its checkpoint was lost
            ["6945", "0.4", "", "", "", ""],             # dev row (not a detector point) stays
            ["13890", "0.5", "2", "2", "1.0", "9"],      # re-realised row: the one whose ckpt exists
            ["27780", "0.6", "0", "0", "1.0", "7"]]
    src = tmp_path / "metrics.tsv"
    src.write_text("\t".join(hdr) + "\n" + "\n".join("\t".join(r) for r in rows) + "\n")
    dst = tmp_path / "dedup.tsv"
    info = v7.dedupe_metrics(str(src), str(dst))
    out = v7.read_rows(str(dst))
    assert info["orphan_detector_rows_dropped"] == [13890] and info["rows_out"] == 3
    assert [r["step"] for r in out] == ["6945", "13890", "27780"]
    assert out[1]["full_rep_errors"] == "2"
    assert int(v7.first_hit(out)["step"]) == 27780
    assert src.read_text().count("\n") == 5                   # source untouched


def test_seed_audit_flags_only_real_tokens(tmp_path):
    (tmp_path / "final_settle_ctrl_h512_s19").mkdir()
    (tmp_path / "cap3_wm128_seed22_full").mkdir()
    (tmp_path / "s310_not_a_seed").mkdir()
    (tmp_path / "docs31").mkdir()
    r = v7.seed_audit([19, 22, 31, 32], [str(tmp_path)],
                      texts=["job l3_settle seed=22 ok", "the u3100 window", "seeds 31-34 are FUTURE candidates"])
    assert r["clean"] == [31, 32] and r["hits"][19] and r["hits"][22]
    r2 = v7.seed_audit([31], [], texts=["run final_x_h512_s31 done"])
    assert r2["clean"] == []


# ------------------------------------------------------- checkpoint gate ---
def _fake_ckpt(seed, step, lr, anchor, transitions, **over):
    ck = {"format": "lichtheim3.joint_scratch.v1", "seed": seed, "global_step": step,
          "widths": {"wm_hidden": 128, "ltm_enc_hidden": 512, "ltm_dec_hidden": 512},
          "schedule": "interleaved_123", "schedule_ratio": [1, 2, 3],
          "schedule_seed": derive_schedule_seed(seed), "stream_seeds": derive_stream_seeds(seed),
          "optimizer_policy": "shared_adamw", "optimizer_state_dict": {}, "dec_weight": 0.5,
          "c_align_weight": 0.0, "subset_mode": "final_full",
          "comprehension_population_n": 27981,
          "comprehension_population_sha256": v7.EXPECTED["comprehension_population_sha256"],
          "naming_population_n": 29571, "naming_population_sha256": v7.EXPECTED["naming_population_sha256"],
          "lexicon_file_sha256": v7.EXPECTED["lexicon_file_sha256"], "glove_fallback": 0,
          "lr_boundary_steps": 46300, "lr_policy": lr, "schedule_anchor_step": anchor,
          "phase_transitions": transitions,
          "cursors": {"repetition": step // 6, "pool": step // 6, "naming": step // 3, "comprehension": step // 2}}
    ck.update(over)
    return ck


def test_checkpoint_gate_accepts_the_ledger_and_refuses_deviations():
    T = v7.expected_transitions_before
    assert v7.verify_checkpoint(_fake_ckpt(31, 2_083_500, v7.TWO_STAGE, 0, []), 31, 2_083_500) == []
    assert v7.verify_checkpoint(_fake_ckpt(31, 2_083_500 + 13_890, v7.TASK_3E5, 0, T(2_083_506)), 31) == []
    ok = _fake_ckpt(32, 10_625_850, v7.TASK_CHIGH, 8_334_000, T(10_625_850))
    assert v7.verify_checkpoint(ok, 32) == []
    assert v7.verify_checkpoint(ok, 31)                                    # wrong seed
    assert v7.verify_checkpoint(_fake_ckpt(32, 10_625_850, v7.TASK_CHIGH, 5_556_000, T(10_625_850)), 32)   # missing re-anchor
    assert v7.verify_checkpoint(_fake_ckpt(32, 10_625_850, v7.TASK_3E5, 8_334_000, T(10_625_850)), 32)     # wrong LR
    assert v7.verify_checkpoint(_fake_ckpt(32, 3_333_600 + 13_890, v7.TASK_CHIGH, 0, T(3_333_600)), 32)   # re-anchor not declared
    bad_w = _fake_ckpt(32, 13_890, v7.TWO_STAGE, 0, [], widths={"wm_hidden": 128, "ltm_enc_hidden": 256, "ltm_dec_hidden": 256})
    assert any("widths" in b for b in v7.verify_checkpoint(bad_w, 32))
    assert any("glove" in b for b in v7.verify_checkpoint(_fake_ckpt(32, 13_890, v7.TWO_STAGE, 0, [], glove_fallback=3), 32))
    assert any("grid" in b for b in v7.verify_checkpoint(_fake_ckpt(32, 13_896, v7.TWO_STAGE, 0, []), 32))
    assert any("optimizer" in b or "AdamW" in b for b in v7.verify_checkpoint(
        _fake_ckpt(32, 13_890, v7.TWO_STAGE, 0, [], optimizer_policy="task_separated_adamw"), 32))


# ------------------------------------------------------------- job script --
def test_job_script_pins_the_contract():
    s = open(JOB, encoding="utf-8").read()
    assert "#SBATCH --array=0-3" in s and "#SBATCH --requeue" in s
    assert "SEEDS=(31 32 33 34)" in s and "SLOTS=(p1 p2 p3 p4)" in s
    assert "fresh_ceiling_v7_${SLOT}_s${SEED}" in s
    for sha in ("766244c1363d998f34fe3e4f725b57a9b3f95cf716876d71470d804cf962765c",   # train_joint_scratch.py
                "a867b6b5ec474fb79fcd18e769fa8d16d8bdc3de920f7704bfa9be2e7b907303",   # train_tasks.py
                "a7064708878365fc2875c3db6f903b3e370b16023bab12ce2be23c0f1ea5dcbf",   # gradient_training_probe.py
                "fc1a0865bba0810ba03a3a01a54feb234dd6cca7c341a94527c093f41f18088f",   # ceiling_source_completion.py (V6 fix)
                "2482adb2ab89ced1249303f8e898a00e2a6927e40d22e1cc5f57d55db50a6c5c",   # first_hit_selector.py
                "ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66",   # lexicon
                "91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed"):  # GloVe
        assert sha in s
    assert "MAX_U=3900" in s and "END_STEP=10834200" in s and "DET_STEP=13890" in s
    assert "--stop-at-ceiling" in s and "CEILING_REQUIRED=2" in s
    assert "lichtheim3_runs" in s and "l3_prospective_v6_runs" in s     # namespaces refused / audited
    assert "FIRST_HIT_FROZEN.json" in s
    for forbidden in ("--allow-glove-fallback", "--torch-deterministic", "--optimizer-policy",
                      "--dec-weight", "--c-align-weight", "interleaved_223", "--endpoint-eval"):
        assert forbidden not in s, forbidden


def test_frozen_scientific_files_are_byte_identical_to_the_v6_completion_commit():
    import hashlib
    want = {"scripts/naming_comprehension/train_joint_scratch.py": "766244c1363d998f34fe3e4f725b57a9b3f95cf716876d71470d804cf962765c",
            "scripts/naming_comprehension/train_tasks.py": "a867b6b5ec474fb79fcd18e769fa8d16d8bdc3de920f7704bfa9be2e7b907303",
            "scripts/naming_comprehension/frozen_probe.py": "d4c7bb9289cef56d09d05b1cbcbd9b47f0fa32dda549b333d29d8a4753689d4f",
            "scripts/naming_comprehension/base123_error_audit.py": "15008114d5e5ff3bde51b550193c07e62d5566347a29510ec9a75bb8ae910b37",
            "scripts/naming_comprehension/ceiling_source_completion.py": "fc1a0865bba0810ba03a3a01a54feb234dd6cca7c341a94527c093f41f18088f",
            "scripts/naming_comprehension/gradient_training_probe.py": "a7064708878365fc2875c3db6f903b3e370b16023bab12ce2be23c0f1ea5dcbf",
            "scripts/naming_comprehension/constrained_coexistence_v3.py": "1c69a59c525f11f54946fe7fd42412be601b4b451a55706925ed516b4e495419",
            "scripts/naming_comprehension/frozen_head_probe.py": "f2a30620aad79bba81bbd9e56ab0b1a38a74801c05e35b0f9e74f7bb63fb851b",
            "scripts/naming_comprehension/coexistence_probe.py": "de715e3c573c44d00c0fd8bb56fef1e916c79b4f1e58af3965cb3f48d828ecef",
            "scripts/naming_comprehension/first_hit_selector.py": "2482adb2ab89ced1249303f8e898a00e2a6927e40d22e1cc5f57d55db50a6c5c",
            "losses.py": "382eed2335ee2add7a00f6de9b73a788cdef5e94155ae1970e38bd5d7dcefd80",
            "data/lexicon_en_glove_covered.tsv": "ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66"}
    for rel, sha in want.items():
        assert hashlib.sha256(open(os.path.join(ROOT, rel), "rb").read()).hexdigest() == sha, rel


# ------------------------------------------------ REAL driver, tiny fixture --
TINY = ["--regime", "j0", "--subset-mode", "final_full", "--device", "cpu",
        "--schedule", "interleaved_123", "--wm-hidden", "128", "--enc-hidden", "512", "--dec-hidden", "512",
        "--max-words", "400", "--batch-size", "8", "--dorsal-pool-size", "32",
        "--allow-glove-fallback", "--no-subset-hash-check", "--lr-boundary-steps", "1",
        "--log-every", "0", "--probe-every", "0",
        "--stop-at-ceiling", "--ceiling-consecutive-required", "2"]


def _scaled_ledger():
    two = dict(v7.TWO_STAGE, boundary_rep_batches=1)
    return [{"name": "L1", "start": 0, "end": 24, "lr_policy": two, "reanchor": False, "det_step": 6},
            {"name": "L2", "start": 24, "end": 36, "lr_policy": v7.TASK_3E5, "reanchor": False},
            {"name": "L3", "start": 36, "end": 48, "lr_policy": v7.TASK_CHIGH, "reanchor": False},
            {"name": "L4", "start": 48, "end": 60, "lr_policy": v7.TASK_CHIGH, "reanchor": True},
            {"name": "L5", "start": 60, "end": 72, "lr_policy": v7.TASK_CHIGH, "reanchor": True}]


def _latest(ckdir):
    fs = sorted(f for f in os.listdir(ckdir) if f.startswith("step_"))
    return int(fs[-1][5:-3]) if fs else None


def _launch(out, run_id, seed, p, grid):
    cmd = [sys.executable, DRIVER, *TINY, "--seed", str(seed), "--out-dir", str(out), "--run-id", run_id,
           "--max-steps", str(p["max_steps"]), "--full-eval-at", ",".join(map(str, grid)),
           "--eval-every", "6", "--save-every", "6", *p["flags"]]
    if p["resume"] is not None:
        cmd += ["--resume", os.path.join(out, run_id, "checkpoints", f"step_{p['resume']:08d}.pt")]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
    return r


@pytest.mark.slow
def test_real_driver_executes_the_leg_plan_and_records_the_ledger(tmp_path):
    """Engineering smoke (bounded, NOT a scientific experiment): the exact
    flag sets `plan()` derives are fed to the UNCHANGED driver on a tiny
    fixture, leg after leg, including a plain mid-leg requeue.  The resulting
    checkpoint ledger must equal the frozen ledger's shape, every boundary
    checkpoint must pass the gate, and a mid-leg launch that wrongly carried
    --reanchor-schedule must be refused by the gate afterwards."""
    L = _scaled_ledger()
    grid = v7.grid(L)
    out, run_id, seed = tmp_path, "smoke_v7", 31
    ckdir = os.path.join(out, run_id, "checkpoints")
    launches = []
    latest = None
    while True:
        p = v7.plan(latest, L)
        if p["status"] == "COMPLETE":
            break
        if p["resume"] is not None:
            ck = torch.load(os.path.join(ckdir, f"step_{p['resume']:08d}.pt"), map_location="cpu", weights_only=False)
            assert v7.verify_checkpoint(ck, seed, p["resume"], L, tiny=True) == []
        # simulate a walltime kill in the middle of L4: stop 6 steps early once
        if p["leg"] == "L4" and p["status"] == "BOUNDARY_LAUNCH":
            p = dict(p, max_steps=54)
        r = _launch(out, run_id, seed, p, grid)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        launches.append((p["status"], p["leg"], p["resume"], p["max_steps"], p["flags"]))
        latest = _latest(ckdir)
        assert latest == p["max_steps"]
    statuses = [(s, leg) for s, leg, *_ in launches]
    assert statuses == [("FIRST_LAUNCH", "L1"), ("BOUNDARY_LAUNCH", "L2"), ("BOUNDARY_LAUNCH", "L3"),
                        ("BOUNDARY_LAUNCH", "L4"), ("REQUEUE", "L4"), ("BOUNDARY_LAUNCH", "L5")]
    final = torch.load(os.path.join(ckdir, "step_00000072.pt"), map_location="cpu", weights_only=False)
    assert v7.verify_checkpoint(final, seed, 72, L, tiny=True) == []
    T = final["phase_transitions"]
    assert [t["transition_step"] for t in T] == [24, 36, 48, 60]
    assert [t["changed"] for t in T] == [["lr_policy"], ["lr_policy"], ["schedule_anchor"], ["schedule_anchor"]]
    assert [t["schedule_anchor_step"] for t in T] == [0, 0, 48, 60]
    assert all(t["moment_initialization"] == "unchanged" for t in T)
    assert final["schedule_anchor_step"] == 60 and final["lr_policy"] == v7.TASK_CHIGH
    assert final["cursors"] == {"repetition": 12, "pool": 12, "naming": 24, "comprehension": 36}
    # every detector point produced exactly one full row and one checkpoint
    rows = v7.read_rows(os.path.join(out, run_id, "metrics.tsv"))
    det = [int(float(r["step"])) for r in rows if v7.is_detector_row(r)]
    assert det == grid
    assert sorted(_latest_all(ckdir)) == grid                  # step-0 start eval also checkpoints
    # the gate catches the ONE wrong thing a requeue could do: re-anchoring mid-leg
    wrong = dict(v7.plan(66, L)); wrong["flags"] = wrong["flags"] + ["--reanchor-schedule", "--phase-transition"]
    assert wrong["status"] == "REQUEUE"
    r = _launch(out, run_id, seed, dict(wrong, max_steps=72), grid)      # driver accepts (declared)...
    assert r.returncode == 0
    bad = v7.verify_checkpoint(torch.load(os.path.join(ckdir, "step_00000072.pt"), map_location="cpu",
                                          weights_only=False), seed, 72, L, tiny=True)
    assert bad and any("transitions" in b or "anchor" in b for b in bad)     # ...but the gate refuses it


def _latest_all(ckdir):
    return [int(f[5:-3]) for f in os.listdir(ckdir) if f.startswith("step_")]


@pytest.mark.slow
def test_watcher_freezes_the_first_hit_and_stops_the_driver(tmp_path):
    """The watcher must act on the FIRST eligible detector row, only once its
    checkpoint is fully written, and must ignore later/better rows."""
    import time, signal, threading
    run = tmp_path / "run"; (run / "checkpoints").mkdir(parents=True)
    marker = str(tmp_path / "FIRST_HIT_FROZEN.json")
    hdr = ["step", "full_rep_errors", "full_rep_freear_errors", "full_naming_exact", "full_comp_errors", "full_rep_ltm", "gate_mean"]
    m = run / "metrics.tsv"
    m.write_text("\t".join(hdr) + "\n" + "\t".join(["13890", "1", "1", "1.0", "50", "0.8", "0.5"]) + "\n")
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    stop = {}
    def run_watch():
        stop["rc"] = v7.watch(str(run), proc.pid, marker, poll_s=0.5)
    th = threading.Thread(target=run_watch); th.start()
    time.sleep(1.5)
    assert not os.path.exists(marker)
    with open(m, "a") as f:                                    # eligible row, checkpoint not yet written
        f.write("\t".join(["27780", "0", "0", "1.0", "40", "0.85", "0.5"]) + "\n")
    time.sleep(1.5)
    assert not os.path.exists(marker) and proc.poll() is None
    torch.save({"global_step": 27780, "x": torch.zeros(1000)}, run / "checkpoints" / "step_00027780.pt")
    with open(m, "a") as f:                                    # a LATER, lower-C eligible row must be ignored
        f.write("\t".join(["41670", "0", "0", "1.0", "5", "0.9", "0.5"]) + "\n")
    torch.save({"global_step": 41670, "x": torch.zeros(1000)}, run / "checkpoints" / "step_00041670.pt")
    th.join(timeout=120)
    assert stop.get("rc") == 0
    rec = json.load(open(marker))
    assert rec["first_hit_step"] == 27780 and rec["detector_row"]["full_comp_errors"] == "40"
    proc.wait(timeout=30)
    assert proc.returncode != 0                                # SIGTERM delivered
