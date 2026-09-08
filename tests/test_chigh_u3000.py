"""Acceptance tests for the three-arm comprehension-LR raise u2000 -> u3000.

All three arms keep schedule 1:2:3 and re-anchor at the same branch step, so
the design's whole claim is: identical everything, including cursors and
post-branch task order, except the comprehension learning rate.

The schedule-seed invariant is checked at the RIGHT scope: within a seed,
across arms.  Across different seeds it is expected to differ.
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
    derive_schedule_seed, main,
)

JOB = "scripts/cluster/jeanzay/final_chigh_u2000_to_u3000.slurm"
R_PASS, C_PASS, CYCLE = 463, 438, 6
SRC_STEP, END_STEP, MS_STEP = 5_556_000, 8_334_000, 69_450
ARM_LR = {"ctrl": "1e-4", "15e5": "1.5e-4", "2e4": "2e-4"}
ARM_LR_F = {"ctrl": 1e-4, "15e5": 1.5e-4, "2e4": 2e-4}
SEEDS = (19, 20, 21, 22)

BASE = ["--regime", "j0", "--subset-mode", "final_full", "--device", "cpu",
        "--max-words", "400", "--batch-size", "8", "--dorsal-pool-size", "32",
        "--lr-boundary-steps", "1", "--eval-every", "0", "--log-every", "0",
        "--glove-path", "tests/_no_such_glove_file.txt",
        "--allow-glove-fallback", "--no-subset-hash-check",
        "--schedule", "interleaved_123",
        "--enc-hidden", "64", "--dec-hidden", "64"]
END = 24 + 6 * 5


def script(path=JOB):
    return open(os.path.join(ROOT, path), encoding="utf-8").read()


def executable(path=JOB):
    return "\n".join(l for l in script(path).splitlines()
                     if l.strip() and not l.lstrip().startswith("#"))


def ck(out, run_id, step):
    return os.path.join(out, run_id, "checkpoints", f"step_{step:08d}.pt")


def load(p):
    return torch.load(p, map_location="cpu", weights_only=False)


def lrs(c):
    return ["--lr-repetition", "3e-5", "--lr-naming", "3e-5",
            "--lr-comprehension", c]


def _source(out, seed):
    """A C = 1e-4 source in the shape of the u2000 canneal control."""
    assert main(BASE + ["--seed", str(seed), "--out-dir", out,
                        "--run-id", f"pre{seed}", "--max-steps", "12",
                        "--save-every", "12"]) == 0
    assert main(BASE + ["--seed", str(seed), "--out-dir", out,
                        "--run-id", f"src{seed}", "--max-steps", "24",
                        "--save-every", "24", "--resume",
                        ck(out, f"pre{seed}", 12), "--phase-transition"]
                + lrs("1e-4")) == 0
    return ck(out, f"src{seed}", 24)


def _branch(out, seed, arm, run_id=None):
    rid = run_id or f"{arm}{seed}"
    assert main(BASE + ["--seed", str(seed), "--out-dir", out, "--run-id", rid,
                        "--max-steps", str(END), "--save-every", str(END),
                        "--resume", _SRC[seed],
                        "--reanchor-schedule", "--phase-transition"]
                + lrs(ARM_LR[arm])) == 0
    return ck(out, rid, END)


_SRC: dict = {}


@pytest.fixture
def three_arms(tmp_path):
    out = str(tmp_path / "runs")
    _SRC.clear()
    _SRC[19] = _source(out, 19)
    got = {arm: _branch(out, 19, arm) for arm in ARM_LR}
    return out, _SRC[19], got


# ==========================  1. only the comprehension LR differs  =========

def test_all_arms_share_every_setting_except_the_c_lr(three_arms):
    _, _, paths = three_arms
    got = {arm: load(p) for arm, p in paths.items()}
    for arm, want in ARM_LR_F.items():
        p = got[arm]["lr_policy"]
        assert p["kind"] == LR_POLICY_TASK
        assert p["repetition"] == 3e-5 and p["naming"] == 3e-5
        assert p["comprehension"] == want, arm
    a, b, c = got["ctrl"], got["15e5"], got["2e4"]
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
    _, src, paths = three_arms
    s = load(src)["cursors"]
    cur = [load(p)["cursors"] for p in paths.values()]
    assert cur[0] == cur[1] == cur[2]
    for t, mult in (("repetition", 1), ("naming", 2), ("comprehension", 3),
                    ("pool", 1)):
        assert cur[0][t] - s[t] == mult * 5, t


def test_raising_the_c_lr_actually_changes_the_trajectory(three_arms):
    _, _, paths = three_arms
    a = load(paths["ctrl"])["model_state_dict"]
    for arm in ("15e5", "2e4"):
        b = load(paths[arm])["model_state_dict"]
        assert [k for k in a if not torch.equal(a[k], b[k])], arm


# =====================================  2. common schedule origin  =========

def test_every_arm_reanchors_at_the_same_branch_step(three_arms):
    _, src, paths = three_arms
    step = load(src)["global_step"]
    for arm, p in paths.items():
        assert load(p)["schedule_anchor_step"] == step, arm
    rec = load(paths["ctrl"])["phase_transitions"][-1]
    assert rec["changed"] == ["schedule_anchor"], \
        "the control's ONLY declared change is the re-anchor"
    assert rec["old_schedule"] == rec["new_schedule"] == INTERLEAVED_123
    assert rec["reanchored"] == 1
    assert rec["moment_initialization"] == "unchanged"
    for arm in ("15e5", "2e4"):
        hi = load(paths[arm])["phase_transitions"][-1]
        assert "schedule_anchor" in hi["changed"] and "lr_policy" in hi["changed"]
        assert hi["new_lr_policy"]["comprehension"] == ARM_LR_F[arm]
        assert hi["new_lr_policy"]["repetition"] == 3e-5
        assert hi["new_lr_policy"]["naming"] == 3e-5
        assert hi["moment_initialization"] == "unchanged"


def test_nuisance_control_two_identical_arms_are_bitwise_identical(tmp_path):
    """Two identically configured re-anchored 1:2:3 arms must be bitwise
    identical in weights, BOTH AdamW moments, cursors and RNG, so any later
    difference is attributable to the C LR alone."""
    out = str(tmp_path / "runs")
    _SRC.clear()
    _SRC[19] = _source(out, 19)
    a = load(_branch(out, 19, "ctrl", run_id="a"))
    b = load(_branch(out, 19, "ctrl", run_id="b"))
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    assert set(sa) == set(sb)
    assert not [k for k in sa if not torch.equal(sa[k], sb[k])]
    oa, ob = a["optimizer_state_dict"]["state"], b["optimizer_state_dict"]["state"]
    assert set(oa) == set(ob) and oa
    for i in oa:
        for m in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(oa[i][m], ob[i][m])
        assert oa[i]["step"] == ob[i]["step"]
    assert a["cursors"] == b["cursors"]
    assert a["schedule_anchor_step"] == b["schedule_anchor_step"] == 24
    assert a["schedule_seed"] == b["schedule_seed"]
    assert torch.equal(a["rng_states"]["torch"], b["rng_states"]["torch"])


def test_moments_rng_and_sampler_cursors_survive_the_branch(three_arms):
    _, src, _ = three_arms
    s = load(src)
    mom = {i: (stt["exp_avg"].clone(), stt["exp_avg_sq"].clone())
           for i, stt in s["optimizer_state_dict"]["state"].items()}
    tr = JointScratchTrainer(
        regime="j0", seed=19, device="cpu", max_words=400, batch_size=8,
        lexicon_path="data/lexicon_en_glove_covered.tsv", dorsal_pool_size=32,
        subset_mode="final_full", subset_per_band=822, subset_size=32,
        lr_boundary_steps=1, allow_glove_fallback=True,
        require_subset_hash=False, schedule=INTERLEAVED_123,
        glove_path="tests/_no_such_glove_file.txt", enc_hidden=64,
        dec_hidden=64, allow_phase_transition=True, reanchor_schedule=True,
        task_lrs={"repetition": 3e-5, "naming": 3e-5, "comprehension": 2e-4})
    tr.load_state_dict(s)
    assert tr.cursors == {k: int(v) for k, v in s["cursors"].items()}
    assert tr.task_optims is None
    stt = tr.optim.state_dict()["state"]
    for i, (avg, sq) in mom.items():
        assert torch.equal(stt[i]["exp_avg"], avg)
        assert torch.equal(stt[i]["exp_avg_sq"], sq)
    assert torch.equal(torch.get_rng_state(), s["rng_states"]["torch"])
    assert tr.schedule_anchor_step == s["global_step"]


# ======================  3. the schedule-seed invariant, at the right scope

def test_schedule_seed_is_a_pure_function_of_the_seed():
    """seed*1000003 + 4 -- exactly the values the previous report mislabelled
    a confound."""
    assert {s: derive_schedule_seed(s) for s in SEEDS} == {
        19: 19000061, 20: 20000064, 21: 21000067, 22: 22000070}
    assert len({derive_schedule_seed(s) for s in SEEDS}) == 4, \
        "different seeds MUST get different schedule seeds"


def test_within_one_seed_all_three_arms_share_the_schedule_seed(three_arms):
    _, _, paths = three_arms
    got = {arm: load(p)["schedule_seed"] for arm, p in paths.items()}
    assert len(set(got.values())) == 1
    assert got["ctrl"] == derive_schedule_seed(19)


def test_two_different_seeds_are_supposed_to_differ(tmp_path):
    out = str(tmp_path / "runs")
    _SRC.clear()
    _SRC[19], _SRC[20] = _source(out, 19), _source(out, 20)
    a = load(_branch(out, 19, "ctrl"))["schedule_seed"]
    b = load(_branch(out, 20, "ctrl"))["schedule_seed"]
    assert a != b
    assert (a, b) == (derive_schedule_seed(19), derive_schedule_seed(20))


def test_report_checks_schedule_seed_within_seed_not_across(tmp_path):
    """The regression that motivated this: a global set-comparison over
    schedule_seed prints a confound that does not exist."""
    from scripts.naming_comprehension import chigh_report as m
    src = open(os.path.join(ROOT, "scripts/naming_comprehension/"
                                  "chigh_report.py"), encoding="utf-8").read()
    body = src.split('if policy:', 1)[1]
    glob_block = body.split("# (b)", 1)[0]
    assert "schedule_seed" not in glob_block, \
        "schedule_seed must NOT be in the across-all-runs identity loop"
    assert m.expected_schedule_seed(19) == 19000061
    assert m.expected_schedule_seed(22) == 22000070
    # and the same fix landed in the canneal report that produced the artefact
    from scripts.naming_comprehension import canneal_report as cm  # noqa: F401
    csrc = open(os.path.join(ROOT, "scripts/naming_comprehension/"
                                   "canneal_report.py"), encoding="utf-8").read()
    cglob = csrc.split('if policy:', 1)[1].split("# schedule_seed is", 1)[0]
    assert '"schedule_seed"' not in cglob


# ==================================================  4. job contract  =====

def test_job_maps_three_arms_four_seeds():
    t = script()
    assert "#SBATCH --array=0-11" in t
    assert "ARMS=(ctrl ctrl ctrl ctrl 15e5 15e5 15e5 15e5 2e4 2e4 2e4 2e4)" in t
    assert ("LR_CS=(1e-4 1e-4 1e-4 1e-4 1.5e-4 1.5e-4 1.5e-4 1.5e-4 "
            "2e-4 2e-4 2e-4 2e-4)") in t
    assert "SEEDS=(19 20 21 22 19 20 21 22 19 20 21 22)" in t
    assert 'RUN_ID="final_chigh_${ARM}_u3000_h512_s${SEED}"' in t
    arms = ["ctrl"] * 4 + ["15e5"] * 4 + ["2e4"] * 4
    assert len(set(zip(arms, list(SEEDS) * 3))) == 12


def test_job_raises_c_and_holds_everything_else():
    t, ex = script(), executable()
    assert "LR_R=3e-5; LR_N=3e-5" in t
    assert "SCHEDULE=interleaved_123" in t
    assert "LR_C=${LR_CS[$IDX]}" in t
    assert ('--lr-repetition "$LR_R" --lr-naming "$LR_N" '
            '--lr-comprehension "$LR_C"') in ex
    assert "WM=128; ENC=512; DEC=512" in t
    for forbidden in ("interleaved_223", "interleaved_124", "--dec-weight",
                      "--c-align-weight", "--optimizer-policy", "grouped_rn_c",
                      "--endpoint-eval", "--eval-at-start", "--batch-size",
                      "--allow-glove-fallback", "margin", "hard_negative",
                      "curriculum"):
        assert forbidden not in ex, forbidden
    assert "CEILING_REQUIRED=5" in t
    assert "EVAL_EVERY=13890" in t and "SAVE_EVERY=69450" in t
    assert "--ceiling-consecutive-required" in ex and "--stop-at-ceiling" in ex


def test_job_reanchors_every_arm_including_the_control():
    t = script()
    assert "EXTRA=(--reanchor-schedule --phase-transition)" in t
    assert "re-anchoring the macro-cycle at $SOURCE_STEP in ALL arms" in t


def test_job_grid_and_source():
    t = script()
    assert "SOURCE_STEP=5556000" in t and "MAX_STEPS=8334000" in t
    assert "MS_STEP=69450" in t and "seq 1 40" in t
    assert 'PARENT_RUN_ID="final_canneal_ctrl_h512_s${SEED}"' in t
    assert 'int(c["repetition"]) == 926000' in t
    assert 'abs(float(p["comprehension"]) - 1e-4) < 1e-12' in t
    assert 'int(ck.get("schedule_anchor_step")) == 3889200' in t
    assert 'seed * 1000003 + 4' in t
    assert "91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed" in t
    assert "95295d63560ae4c235a6beee8dfb47166f4ed30d" in t


def test_job_endpoint_arithmetic():
    assert SRC_STEP == 2000 * R_PASS * CYCLE
    assert END_STEP == 3000 * R_PASS * CYCLE
    assert MS_STEP == 25 * R_PASS * CYCLE
    assert (END_STEP - SRC_STEP) // MS_STEP == 40
    d = END_STEP - SRC_STEP
    assert d == 2_778_000
    cycles = d // CYCLE
    assert cycles == 463_000                       # = +1000 u
    assert cycles // R_PASS == 1000
    assert (926_000 + cycles, 1_852_000 + 2 * cycles, 2_778_000 + 3 * cycles) \
        == (1_389_000, 2_778_000, 4_167_000)
    assert abs(4_167_000 / C_PASS - 9513.6986) < 1e-3
    for k in range(1, 41):
        assert (SRC_STEP + MS_STEP * k) % 69450 == 0


def test_job_protects_every_parent_lineage():
    t = script()
    for pat in ("final_base123_*", "final_lrpilot*", "final_rep_*",
                "final_canneal_*"):
        assert pat in t, pat
    assert "READ ONLY" in t
    assert "final_chigh_ctrl_u3000_h512_s19" in t
    assert "final_chigh_2e4_u3000_h512_s22" in t


def test_job_is_honest_that_1000u_needs_two_allocations():
    """82.3 s/u measured on u850 -> u1200 puts 1000 u at ~23 h; --requeue does
    NOT cover a TIMEOUT, so the resume path must be the documented plan."""
    t = script()
    assert "#SBATCH --time=20:00:00" in t
    assert "does not fit one allocation" in t.lower()
    assert "NOT a TIMEOUT" in t
    assert "second allocation: resuming" in t
    assert 'RESUME_FROM="$OWN_LATEST"' in t
    assert "EXTRA=()" in t, "a second allocation must NOT re-declare anything"
    assert 1000 * 82.3 / 3600 > 20


def test_report_constants_match_the_job():
    from scripts.naming_comprehension import chigh_report as m
    assert m.SRC_STEP == SRC_STEP and m.END_STEP == END_STEP
    assert m.MS_STEP == MS_STEP and m.N_MILESTONES == 40
    assert m.ARM_LR == ARM_LR_F
    assert m.RUN == "final_chigh_{arm}_u3000_h512_s{seed}"
    assert len(m.milestones()) == 40
    assert m.milestones()[0] == 5_625_450 and m.milestones()[-1] == END_STEP


# ==============================================  5. the report itself  =====

def _fixture(root, c_errors, cursor_bug=(), ceiling_from=None,
             milestones=40):
    cols = ["step", "r_exposures", "n_exposures", "c_exposures", "lr",
            "full_rep_ltm", "full_rep_full", "full_rep_wm", "full_comp_top1",
            "full_comp_top5", "full_naming_exact", "full_rep_freear",
            "full_rep_errors", "full_rep_freear_errors", "full_comp_errors",
            "full_naming_errors", "gate_mean"]
    for arm in ARM_LR:
        for seed in SEEDS:
            d = os.path.join(root, f"final_chigh_{arm}_u3000_h512_s{seed}")
            os.makedirs(d)
            with open(os.path.join(d, "metrics.tsv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
                w.writeheader()
                for k in range(1, milestones + 1):
                    step = SRC_STEP + MS_STEP * k
                    u = step / (R_PASS * CYCLE)
                    hit = ceiling_from is not None and k >= ceiling_from \
                        and arm == "2e4"
                    ce = 0 if hit else c_errors(arm, k)
                    c_exp = 3 * u * R_PASS / C_PASS
                    if (arm, seed, k) in cursor_bug:
                        c_exp -= 1.0
                    w.writerow({"step": step - 100, "lr": 3e-5})   # cheap eval
                    w.writerow({
                        "step": step, "lr": 3e-5, "r_exposures": u,
                        "n_exposures": 2 * u, "c_exposures": c_exp,
                        "full_rep_ltm": 0.87, "full_rep_wm": 1.0,
                        "gate_mean": 0.51,
                        "full_rep_full": 1.0, "full_rep_errors": 0,
                        "full_rep_freear": 1.0, "full_rep_freear_errors": 0,
                        "full_naming_exact": 1.0, "full_naming_errors": 0,
                        "full_comp_top1": 1 - ce / 27981,
                        "full_comp_errors": ce, "full_comp_top5": 0.999})


def _run_report(runs, out):
    from scripts.naming_comprehension import chigh_report as m
    assert m.main(["--runs-root", runs, "--out-dir", out]) == 0
    got = {}
    for name in ("chigh_by_milestone", "cursor_identity_proof",
                 "chigh_summary", "chigh_paired", "ceiling_hits",
                 "c_zero_hits"):
        txt = open(os.path.join(out, name + ".tsv"), encoding="utf-8").read()
        got[name] = list(csv.DictReader(txt.splitlines(), delimiter="\t")) \
            if txt.strip() else []
    got["meta"] = json.load(open(os.path.join(out, "chigh_meta.json")))
    return got


def test_report_reads_only_full_evaluations_and_all_40_milestones(tmp_path):
    runs, out = str(tmp_path / "r"), str(tmp_path / "o")
    _fixture(runs, lambda arm, k: 55 - k // 4)
    g = _run_report(runs, out)
    assert len(g["chigh_by_milestone"]) == 12 * 40
    assert g["meta"]["missing_rows"] == 0
    steps = {int(r["global_step"]) for r in g["chigh_by_milestone"]}
    assert steps == {SRC_STEP + MS_STEP * k for k in range(1, 41)}
    assert max(steps) == END_STEP
    assert g["meta"]["paired_at_milestone"] == 40


def test_report_flags_a_cursor_mismatch_instead_of_averaging_over_it(tmp_path):
    runs, out = str(tmp_path / "r"), str(tmp_path / "o")
    _fixture(runs, lambda arm, k: 50, cursor_bug={("ctrl", 20, 40)})
    g = _run_report(runs, out)
    bad = [r for r in g["cursor_identity_proof"] if r["MATCHED"] == "0"]
    assert len(bad) == 1
    assert bad[0]["seed"] == "20" and bad[0]["milestone"] == "40"
    assert bad[0]["C_identical"] == "0"
    assert bad[0]["R_identical"] == "1" and bad[0]["N_identical"] == "1"
    assert g["meta"]["cursor_identity_failures"] == 1


def test_report_answers_the_primary_question_c_equals_zero(tmp_path):
    runs, out = str(tmp_path / "r"), str(tmp_path / "o")
    _fixture(runs, lambda arm, k: 50, ceiling_from=36)   # k 36..40 -> streak 5
    g = _run_report(runs, out)
    assert g["meta"]["c_zero_evaluations"] == 4 * 5
    assert g["meta"]["ceiling_hits"] == 4 * 5
    high = [r for r in g["chigh_by_milestone"]
            if r["arm"] == "2e4" and r["seed"] == "19"]
    assert [r["ceiling_streak"] for r in high[-6:]] == \
        ["0", "1", "2", "3", "4", "5"]
    assert all(r["ceiling_streak"] == "0" for r in g["chigh_by_milestone"]
               if r["arm"] == "ctrl")
    end = {r["arm"]: r for r in g["chigh_summary"] if r["milestone"] == "40"}
    assert end["2e4"]["max_ceiling_streak"] == "5"
    assert end["2e4"]["seeds_at_C_zero"] == "4"
    assert end["ctrl"]["seeds_at_C_zero"] == "0"


def test_report_pairs_each_arm_against_the_control_by_seed(tmp_path):
    runs, out = str(tmp_path / "r"), str(tmp_path / "o")
    _fixture(runs, lambda arm, k: {"ctrl": 55, "15e5": 40, "2e4": 30}[arm])
    g = _run_report(runs, out)
    assert len(g["chigh_paired"]) == 8          # 4 seeds x 2 arms
    d = {(r["seed"], r["arm"]): int(r["dC_errors"]) for r in g["chigh_paired"]}
    for s in ("19", "20", "21", "22"):
        assert d[(s, "15e5")] == -15 and d[(s, "2e4")] == -25
    assert all(r["milestone"] == "40" for r in g["chigh_paired"])


def test_report_pairs_at_the_deepest_common_milestone_when_arms_lag(tmp_path):
    """A timed-out second allocation leaves arms at different depths; the
    paired comparison must fall back to the deepest COMMON milestone rather
    than compare across different u."""
    runs, out = str(tmp_path / "r"), str(tmp_path / "o")
    _fixture(runs, lambda arm, k: 50 - k)
    p = os.path.join(runs, "final_chigh_2e4_u3000_h512_s21", "metrics.tsv")
    rows = list(csv.DictReader(open(p), delimiter="\t"))
    keep = [r for r in rows if int(r["step"]) <= SRC_STEP + MS_STEP * 30]
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader(); [w.writerow(r) for r in keep]
    g = _run_report(runs, out)
    assert g["meta"]["paired_at_milestone"] == 30
    assert all(r["milestone"] == "30" for r in g["chigh_paired"])
    assert len(g["chigh_paired"]) == 8
    assert g["meta"]["missing_rows"] == 10


def test_report_survives_a_run_that_never_started(tmp_path):
    runs, out = str(tmp_path / "r"), str(tmp_path / "o")
    _fixture(runs, lambda arm, k: 50)
    os.remove(os.path.join(runs, "final_chigh_15e5_u3000_h512_s21",
                           "metrics.tsv"))
    g = _run_report(runs, out)
    assert g["meta"]["missing_rows"] == 40
    assert g["meta"]["paired_at_milestone"] is None
    assert g["chigh_paired"] == []
    assert os.path.exists(os.path.join(out, "missing.txt"))
    s = [r for r in g["chigh_summary"]
         if r["arm"] == "15e5" and r["milestone"] == "40"]
    assert s and s[0]["n_seeds"] == "3"
    assert g["meta"]["cursor_identity_failures"] == 0
