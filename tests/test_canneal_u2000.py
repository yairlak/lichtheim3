"""Acceptance tests for the three-arm comprehension-LR anneal u1400 -> u2000.

All three arms keep schedule 1:2:3 and re-anchor at the same branch step, so
the design's whole claim is: identical everything, including cursors and
post-branch task order, except the comprehension learning rate.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.train_joint_scratch import (           # noqa: E402
    INTERLEAVED_123, LR_POLICY_TASK, OPT_POLICY_SHARED, JointScratchTrainer,
    main,
)

JOB = "scripts/cluster/jeanzay/final_canneal_u1400_to_u2000.slurm"
R_PASS, CYCLE = 463, 6
SRC_STEP, END_STEP, MS_STEP = 3_889_200, 5_556_000, 69_450
ARM_LR = {"ctrl": "1e-4", "5e5": "5e-5", "3e5": "3e-5"}

BASE = ["--regime", "j0", "--subset-mode", "final_full", "--device", "cpu",
        "--max-words", "400", "--batch-size", "8", "--dorsal-pool-size", "32",
        "--lr-boundary-steps", "1", "--eval-every", "0", "--log-every", "0",
        "--glove-path", "tests/_no_such_glove_file.txt",
        "--allow-glove-fallback", "--no-subset-hash-check",
        "--schedule", "interleaved_123",
        "--enc-hidden", "64", "--dec-hidden", "64"]


def script(path=JOB):
    return open(os.path.join(ROOT, path), encoding="utf-8").read()


def ck(out, run_id, step):
    return os.path.join(out, run_id, "checkpoints", f"step_{step:08d}.pt")


def load(p):
    return torch.load(p, map_location="cpu", weights_only=False)


def lrs(c):
    return ["--lr-repetition", "3e-5", "--lr-naming", "3e-5",
            "--lr-comprehension", c]


@pytest.fixture
def three_arms(tmp_path):
    """A C-HIGH-style source, then all three arms re-anchored from it."""
    out = str(tmp_path / "runs")
    assert main(BASE + ["--seed", "19", "--out-dir", out, "--run-id", "pre",
                        "--max-steps", "12", "--save-every", "12"]) == 0
    assert main(BASE + ["--seed", "19", "--out-dir", out, "--run-id", "src",
                        "--max-steps", "24", "--save-every", "24",
                        "--resume", ck(out, "pre", 12), "--phase-transition"]
                + lrs("1e-4")) == 0
    src = ck(out, "src", 24)
    assert load(src)["global_step"] % 6 == 0
    for arm, c in ARM_LR.items():
        assert main(BASE + ["--seed", "19", "--out-dir", out,
                            "--run-id", arm, "--max-steps", str(24 + 6 * 5),
                            "--save-every", str(24 + 6 * 5), "--resume", src,
                            "--reanchor-schedule", "--phase-transition"]
                    + lrs(c)) == 0
    return out, src


# ==========================  1. only the comprehension LR differs  =========

def test_all_arms_share_every_setting_except_the_c_lr(three_arms):
    out, _ = three_arms
    got = {arm: load(ck(out, arm, 24 + 6 * 5)) for arm in ARM_LR}
    for arm, want in (("ctrl", 1e-4), ("5e5", 5e-5), ("3e5", 3e-5)):
        p = got[arm]["lr_policy"]
        assert p["kind"] == LR_POLICY_TASK
        assert p["repetition"] == 3e-5 and p["naming"] == 3e-5
        assert p["comprehension"] == want, arm
    a, b, c = got["ctrl"], got["5e5"], got["3e5"]
    for key in ("seed", "stream_seeds", "schedule", "schedule_ratio",
                "schedule_seed", "schedule_anchor_step", "widths",
                "optimizer_policy", "dec_weight", "c_align_weight",
                "lr_boundary_steps", "cursors", "global_step",
                "subset_definition_sha256", "comprehension_population_sha256",
                "naming_population_sha256"):
        assert a[key] == b[key] == c[key], key
    assert a["schedule"] == INTERLEAVED_123
    assert a["schedule_ratio"] == [1, 2, 3]
    assert a["optimizer_policy"] == OPT_POLICY_SHARED


def test_cursors_are_identical_across_arms(three_arms):
    """All arms are 1:2:3, so R/N/C/pool must match exactly, not merely be
    'matched' by some accounting convention."""
    out, src = three_arms
    s = load(src)["cursors"]
    cur = [load(ck(out, arm, 24 + 6 * 5))["cursors"] for arm in ARM_LR]
    assert cur[0] == cur[1] == cur[2]
    for t, mult in (("repetition", 1), ("naming", 2), ("comprehension", 3),
                    ("pool", 1)):
        assert cur[0][t] - s[t] == mult * 5, t


def test_a_different_c_lr_actually_changes_the_trajectory(three_arms):
    out, _ = three_arms
    a = load(ck(out, "ctrl", 24 + 6 * 5))["model_state_dict"]
    c = load(ck(out, "3e5", 24 + 6 * 5))["model_state_dict"]
    assert [k for k in a if not torch.equal(a[k], c[k])]


# =====================================  2. common schedule origin  =========

def test_every_arm_reanchors_at_the_same_branch_step(three_arms):
    out, src = three_arms
    step = load(src)["global_step"]
    for arm in ARM_LR:
        e = load(ck(out, arm, 24 + 6 * 5))
        assert e["schedule_anchor_step"] == step, arm
    ctrl = load(ck(out, "ctrl", 24 + 6 * 5))
    rec = ctrl["phase_transitions"][-1]
    assert rec["changed"] == ["schedule_anchor"], \
        "the control's ONLY declared change is the re-anchor"
    assert rec["old_schedule"] == rec["new_schedule"] == INTERLEAVED_123
    assert rec["reanchored"] == 1
    assert rec["moment_initialization"] == "unchanged"
    low = load(ck(out, "3e5", 24 + 6 * 5))["phase_transitions"][-1]
    assert "schedule_anchor" in low["changed"] and "lr_policy" in low["changed"]
    assert low["new_lr_policy"]["comprehension"] == 3e-5
    assert low["moment_initialization"] == "unchanged"


def test_nuisance_control_two_identical_arms_are_bitwise_identical(tmp_path):
    """Two identically configured re-anchored 1:2:3 arms must be bitwise
    identical, so any later difference is attributable to the C LR alone."""
    out = str(tmp_path / "runs")
    assert main(BASE + ["--seed", "19", "--out-dir", out, "--run-id", "pre",
                        "--max-steps", "12", "--save-every", "12"]) == 0
    assert main(BASE + ["--seed", "19", "--out-dir", out, "--run-id", "src",
                        "--max-steps", "24", "--save-every", "24",
                        "--resume", ck(out, "pre", 12), "--phase-transition"]
                + lrs("1e-4")) == 0
    src = ck(out, "src", 24)
    for rid in ("a", "b"):
        assert main(BASE + ["--seed", "19", "--out-dir", out, "--run-id", rid,
                            "--max-steps", str(24 + 6 * 5), "--save-every",
                            str(24 + 6 * 5), "--resume", src,
                            "--reanchor-schedule", "--phase-transition"]
                    + lrs("1e-4")) == 0
    a, b = load(ck(out, "a", 24 + 6 * 5)), load(ck(out, "b", 24 + 6 * 5))
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    assert not [k for k in sa if not torch.equal(sa[k], sb[k])]
    oa, ob = a["optimizer_state_dict"]["state"], b["optimizer_state_dict"]["state"]
    for i in oa:
        for m in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(oa[i][m], ob[i][m])
    assert a["cursors"] == b["cursors"]
    assert a["schedule_anchor_step"] == b["schedule_anchor_step"] == 24
    assert torch.equal(a["rng_states"]["torch"], b["rng_states"]["torch"])


def test_moments_and_rng_survive_the_branch(three_arms):
    out, src = three_arms
    s = load(src)
    mom = {i: (st["exp_avg"].clone(), st["exp_avg_sq"].clone())
           for i, st in s["optimizer_state_dict"]["state"].items()}
    tr = JointScratchTrainer(
        regime="j0", seed=19, device="cpu", max_words=400, batch_size=8,
        lexicon_path="data/lexicon_en_glove_covered.tsv", dorsal_pool_size=32,
        subset_mode="final_full", subset_per_band=822, subset_size=32,
        lr_boundary_steps=1, allow_glove_fallback=True,
        require_subset_hash=False, schedule=INTERLEAVED_123,
        glove_path="tests/_no_such_glove_file.txt", enc_hidden=64,
        dec_hidden=64, allow_phase_transition=True, reanchor_schedule=True,
        task_lrs={"repetition": 3e-5, "naming": 3e-5, "comprehension": 5e-5})
    tr.load_state_dict(s)
    assert tr.cursors == {k: int(v) for k, v in s["cursors"].items()}
    assert tr.task_optims is None
    stt = tr.optim.state_dict()["state"]
    for i, (avg, sq) in mom.items():
        assert torch.equal(stt[i]["exp_avg"], avg)
        assert torch.equal(stt[i]["exp_avg_sq"], sq)
    assert torch.equal(torch.get_rng_state(), s["rng_states"]["torch"])
    assert tr.schedule_anchor_step == s["global_step"]


# ==================================================  3. job contract  =====

def test_job_maps_three_arms_four_seeds():
    t = script()
    assert "#SBATCH --array=0-11" in t
    assert "ARMS=(ctrl ctrl ctrl ctrl 5e5 5e5 5e5 5e5 3e5 3e5 3e5 3e5)" in t
    assert "LR_CS=(1e-4 1e-4 1e-4 1e-4 5e-5 5e-5 5e-5 5e-5 3e-5 3e-5 3e-5 3e-5)" in t
    assert "SEEDS=(19 20 21 22 19 20 21 22 19 20 21 22)" in t
    assert 'RUN_ID="final_canneal_${ARM}_h512_s${SEED}"' in t
    arms = ["ctrl"] * 4 + ["5e5"] * 4 + ["3e5"] * 4
    seeds = [19, 20, 21, 22] * 3
    assert len(set(zip(arms, seeds))) == 12


def test_job_holds_r_n_schedule_fixed_and_moves_only_c():
    t = script()
    ex = "\n".join(l for l in t.splitlines()
                   if l.strip() and not l.lstrip().startswith("#"))
    assert "LR_R=3e-5; LR_N=3e-5" in t
    assert "SCHEDULE=interleaved_123" in t
    assert "LR_C=${LR_CS[$IDX]}" in t
    assert ('--lr-repetition "$LR_R" --lr-naming "$LR_N" '
            '--lr-comprehension "$LR_C"') in ex
    for forbidden in ("interleaved_223", "interleaved_124", "--dec-weight",
                      "--c-align-weight", "--optimizer-policy", "grouped_rn_c",
                      "--endpoint-eval", "--eval-at-start", "--batch-size",
                      "--allow-glove-fallback", "margin", "hard_negative"):
        assert forbidden not in ex, forbidden
    assert "CEILING_REQUIRED=5" in t
    assert "EVAL_EVERY=13890" in t and "SAVE_EVERY=69450" in t


def test_job_reanchors_every_arm_including_the_control():
    t = script()
    assert "EXTRA=(--reanchor-schedule --phase-transition)" in t
    assert "re-anchoring the macro-cycle at $SOURCE_STEP in ALL arms" in t


def test_job_grid_and_source():
    t = script()
    assert "SOURCE_STEP=3889200" in t and "MAX_STEPS=5556000" in t
    assert "MS_STEP=69450" in t and "seq 1 24" in t
    assert 'PARENT_RUN_ID="final_rep_rescue123_h512_s${SEED}"' in t
    assert 'int(c["repetition"]) == 648200' in t
    assert 'abs(float(p["comprehension"]) - 1e-4) < 1e-12' in t
    assert "91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed" in t
    # grid arithmetic
    assert SRC_STEP == 1400 * R_PASS * CYCLE
    assert END_STEP == 2000 * R_PASS * CYCLE
    assert MS_STEP == 25 * R_PASS * CYCLE
    assert (END_STEP - SRC_STEP) // MS_STEP == 24
    for k in range(1, 25):
        assert (SRC_STEP + MS_STEP * k) % 69450 == 0


def test_job_protects_every_parent_lineage():
    t = script()
    assert 'final_base123_* && "$RUN_ID" != final_lrpilot*' in t
    assert 'final_rep_*' in t
    assert "READ ONLY" in t
    assert "final_canneal_ctrl_h512_s19" in t
    assert "final_canneal_3e5_h512_s22" in t


def test_report_constants_match_the_job():
    from scripts.naming_comprehension import canneal_report as m
    assert m.SRC_STEP == SRC_STEP and m.END_STEP == END_STEP
    assert m.MS_STEP == MS_STEP
    assert m.ARM_LR == {"ctrl": 1e-4, "5e5": 5e-5, "3e5": 3e-5}
    assert len(m.milestones()) == 24
    assert m.milestones()[0] == 3_958_650 and m.milestones()[-1] == END_STEP


def test_job_asks_for_a_walltime_the_measured_rate_supports():
    """u850 -> u1200 was 972,300 steps / 14 full evals in 8 h; this block is
    1.71x that, so 12 h would not have been enough."""
    t = script()
    assert "#SBATCH --time=20:00:00" in t
    assert "#SBATCH --requeue" in t
    assert "== requeue: resuming" in t, "an overrun must resume, not restart"
    assert 'RESUME_FROM="$OWN_LATEST"' in t
    assert "EXTRA=()" in t, "a requeue must NOT re-declare the transition"
    assert (1_666_800 / 972_300) > 1.5


# ==============================================  4. the report itself  =====

def _fixture(root, c_errors, cursor_bug=(), ceiling_from=None):
    """Synthetic metrics.tsv for all 12 runs, at the real milestone steps."""
    import csv as _csv
    cols = ["step", "r_exposures", "n_exposures", "c_exposures", "lr",
            "full_rep_ltm", "full_rep_full", "full_rep_wm", "full_comp_top1",
            "full_comp_top5", "full_naming_exact", "full_rep_freear",
            "full_rep_errors", "full_rep_freear_errors", "full_comp_errors",
            "full_naming_errors", "gate_mean"]
    for arm in ARM_LR:
        for seed in (19, 20, 21, 22):
            d = os.path.join(root, f"final_canneal_{arm}_h512_s{seed}")
            os.makedirs(d)
            with open(os.path.join(d, "metrics.tsv"), "w", newline="") as fh:
                w = _csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
                w.writeheader()
                for k in range(1, 25):
                    step = SRC_STEP + MS_STEP * k
                    u = step / (R_PASS * CYCLE)
                    ce = c_errors(arm, k)
                    hit = ceiling_from is not None and k >= ceiling_from \
                        and arm == "3e5"
                    ce = 0 if hit else ce
                    c_exp = 3 * u * 463 / 438
                    if (arm, seed, k) in cursor_bug:
                        c_exp -= 1.0
                    # a cheap non-full eval row that must be ignored
                    w.writerow({"step": step - 100, "lr": 3e-5})
                    w.writerow({
                        "step": step, "lr": 3e-5, "r_exposures": u,
                        "n_exposures": 2 * u, "c_exposures": c_exp,
                        "full_rep_ltm": 0.94, "full_rep_wm": 1.0,
                        "gate_mean": 0.51,
                        "full_rep_full": 1.0, "full_rep_errors": 0,
                        "full_rep_freear": 1.0, "full_rep_freear_errors": 0,
                        "full_naming_exact": 1.0, "full_naming_errors": 0,
                        "full_comp_top1": 1 - ce / 27981, "full_comp_errors": ce,
                        "full_comp_top5": 0.999})


def _run_report(runs, out):
    from scripts.naming_comprehension import canneal_report as m
    assert m.main(["--runs-root", runs, "--out-dir", out]) == 0
    got = {}
    for name in ("canneal_by_milestone", "cursor_identity_proof",
                 "canneal_summary", "canneal_endpoint_paired", "ceiling_hits"):
        p = os.path.join(out, name + ".tsv")
        txt = open(p, encoding="utf-8").read()
        got[name] = list(csv.DictReader(txt.splitlines(), delimiter="\t")) \
            if txt.strip() else []
    got["meta"] = json.load(open(os.path.join(out, "canneal_meta.json")))
    return got


def test_report_reads_only_full_evaluations_and_all_24_milestones(tmp_path):
    runs, out = str(tmp_path / "r"), str(tmp_path / "o")
    _fixture(runs, lambda arm, k: 100 - k)
    g = _run_report(runs, out)
    assert len(g["canneal_by_milestone"]) == 12 * 24
    assert g["meta"]["missing_rows"] == 0
    steps = {int(r["global_step"]) for r in g["canneal_by_milestone"]}
    assert steps == {SRC_STEP + MS_STEP * k for k in range(1, 25)}
    assert max(steps) == END_STEP


def test_report_flags_a_cursor_mismatch_instead_of_averaging_over_it(tmp_path):
    runs, out = str(tmp_path / "r"), str(tmp_path / "o")
    _fixture(runs, lambda arm, k: 90, cursor_bug={("ctrl", 20, 24)})
    g = _run_report(runs, out)
    bad = [r for r in g["cursor_identity_proof"] if r["MATCHED"] == "0"]
    assert len(bad) == 1
    assert bad[0]["seed"] == "20" and bad[0]["milestone"] == "24"
    assert bad[0]["C_identical"] == "0"
    assert bad[0]["R_identical"] == "1" and bad[0]["N_identical"] == "1"
    assert g["meta"]["cursor_identity_failures"] == 1


def test_report_tracks_the_five_milestone_ceiling_streak(tmp_path):
    runs, out = str(tmp_path / "r"), str(tmp_path / "o")
    _fixture(runs, lambda arm, k: 80, ceiling_from=20)   # k = 20..24 -> 5
    g = _run_report(runs, out)
    assert g["meta"]["ceiling_hits"] == 4 * 5
    low = [r for r in g["canneal_by_milestone"]
           if r["arm"] == "3e5" and r["seed"] == "19"]
    assert [r["ceiling_streak"] for r in low[-6:]] == \
        ["0", "1", "2", "3", "4", "5"]
    assert all(r["ceiling_streak"] == "0" for r in g["canneal_by_milestone"]
               if r["arm"] == "ctrl")
    end = {r["arm"]: r for r in g["canneal_summary"]
           if r["milestone"] == "24"}
    assert end["3e5"]["max_ceiling_streak"] == "5"


def test_report_pairs_each_arm_against_the_control_by_seed(tmp_path):
    runs, out = str(tmp_path / "r"), str(tmp_path / "o")
    _fixture(runs, lambda arm, k: {"ctrl": 100, "5e5": 70, "3e5": 40}[arm])
    g = _run_report(runs, out)
    assert len(g["canneal_endpoint_paired"]) == 8      # 4 seeds x 2 arms
    d = {(r["seed"], r["arm"]): int(r["dC_errors"])
         for r in g["canneal_endpoint_paired"]}
    for s in ("19", "20", "21", "22"):
        assert d[(s, "5e5")] == -30 and d[(s, "3e5")] == -60


def test_report_survives_a_run_that_never_started(tmp_path):
    runs, out = str(tmp_path / "r"), str(tmp_path / "o")
    _fixture(runs, lambda arm, k: 90)
    os.remove(os.path.join(runs, "final_canneal_5e5_h512_s21", "metrics.tsv"))
    g = _run_report(runs, out)
    assert g["meta"]["missing_rows"] == 24
    assert os.path.exists(os.path.join(out, "missing.txt"))
    # the surviving seeds are still summarised, at their true n
    s = [r for r in g["canneal_summary"]
         if r["arm"] == "5e5" and r["milestone"] == "24"]
    assert s and s[0]["n_seeds"] == "3"
    assert g["meta"]["cursor_identity_failures"] == 0
