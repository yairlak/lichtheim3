"""Acceptance tests for the late-state SETTLE test u3000 -> u3600.

Same causal contract as CANNEAL/CHIGH -- identical everything, including
cursors and post-branch task order, except the comprehension learning rate --
plus the two lessons this lineage has paid for: the primary readout is
SMOOTHED over the last four common milestones, and the schedule-seed
invariant is checked within a seed, never across seeds.
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
    INTERLEAVED_123, LR_POLICY_TASK, OPT_POLICY_SHARED,
    derive_schedule_seed, main,
)

JOB = "scripts/cluster/jeanzay/final_settle_u3000_to_u3600.slurm"
R_PASS, C_PASS, CYCLE = 463, 438, 6
SRC_STEP, END_STEP, MS_STEP = 8_334_000, 10_000_800, 69_450
ARM_LR = {"ctrl": "1e-4", "5e5": "5e-5", "3e5": "3e-5"}
ARM_LR_F = {"ctrl": 1e-4, "5e5": 5e-5, "3e5": 3e-5}
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
    """A C = 1e-4 source in the shape of the u3000 chigh control."""
    assert main(BASE + ["--seed", str(seed), "--out-dir", out,
                        "--run-id", f"pre{seed}", "--max-steps", "12",
                        "--save-every", "12"]) == 0
    assert main(BASE + ["--seed", str(seed), "--out-dir", out,
                        "--run-id", f"src{seed}", "--max-steps", "24",
                        "--save-every", "24", "--resume",
                        ck(out, f"pre{seed}", 12), "--phase-transition"]
                + lrs("1e-4")) == 0
    return ck(out, f"src{seed}", 24)


def _branch(out, seed, src, arm, run_id=None):
    rid = run_id or f"{arm}{seed}"
    assert main(BASE + ["--seed", str(seed), "--out-dir", out, "--run-id", rid,
                        "--max-steps", str(END), "--save-every", str(END),
                        "--resume", src,
                        "--reanchor-schedule", "--phase-transition"]
                + lrs(ARM_LR[arm])) == 0
    return ck(out, rid, END)


@pytest.fixture
def three_arms(tmp_path):
    out = str(tmp_path / "runs")
    src = _source(out, 19)
    return out, src, {arm: _branch(out, 19, src, arm) for arm in ARM_LR}


# ==========================  1. only the comprehension LR differs  =========

def test_all_arms_share_every_setting_except_the_c_lr(three_arms):
    _, _, paths = three_arms
    got = {arm: load(p) for arm, p in paths.items()}
    for arm, want in ARM_LR_F.items():
        p = got[arm]["lr_policy"]
        assert p["kind"] == LR_POLICY_TASK
        assert p["repetition"] == 3e-5 and p["naming"] == 3e-5
        assert p["comprehension"] == want, arm
    a, b, c = got["ctrl"], got["5e5"], got["3e5"]
    for key in ("seed", "stream_seeds", "schedule", "schedule_ratio",
                "schedule_seed", "schedule_anchor_step", "widths",
                "optimizer_policy", "dec_weight", "c_align_weight",
                "cursors", "global_step", "subset_definition_sha256",
                "comprehension_population_sha256", "naming_population_sha256"):
        assert a[key] == b[key] == c[key], key
    assert a["schedule"] == INTERLEAVED_123
    assert a["schedule_ratio"] == [1, 2, 3]
    assert a["optimizer_policy"] == OPT_POLICY_SHARED


def test_cursors_are_identical_across_arms_and_advance_123(three_arms):
    _, src, paths = three_arms
    s = load(src)["cursors"]
    cur = [load(p)["cursors"] for p in paths.values()]
    assert cur[0] == cur[1] == cur[2]
    for t, mult in (("repetition", 1), ("naming", 2), ("comprehension", 3),
                    ("pool", 1)):
        assert cur[0][t] - s[t] == mult * 5, t


def test_lowering_the_c_lr_actually_changes_the_trajectory(three_arms):
    _, _, paths = three_arms
    a = load(paths["ctrl"])["model_state_dict"]
    for arm in ("5e5", "3e5"):
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
    assert rec["reanchored"] == 1
    assert rec["moment_initialization"] == "unchanged"
    for arm in ("5e5", "3e5"):
        lo = load(paths[arm])["phase_transitions"][-1]
        assert "schedule_anchor" in lo["changed"] and "lr_policy" in lo["changed"]
        assert lo["new_lr_policy"]["comprehension"] == ARM_LR_F[arm]
        assert lo["new_lr_policy"]["repetition"] == 3e-5
        assert lo["moment_initialization"] == "unchanged"


def test_nuisance_control_two_identical_arms_are_bitwise_identical(tmp_path):
    out = str(tmp_path / "runs")
    src = _source(out, 19)
    a = load(_branch(out, 19, src, "ctrl", run_id="a"))
    b = load(_branch(out, 19, src, "ctrl", run_id="b"))
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    assert set(sa) == set(sb)
    assert not [k for k in sa if not torch.equal(sa[k], sb[k])]
    oa, ob = a["optimizer_state_dict"]["state"], b["optimizer_state_dict"]["state"]
    assert set(oa) == set(ob) and oa
    for i in oa:
        for m in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(oa[i][m], ob[i][m])
    assert a["cursors"] == b["cursors"]
    assert a["schedule_anchor_step"] == b["schedule_anchor_step"] == 24
    assert torch.equal(a["rng_states"]["torch"], b["rng_states"]["torch"])


def test_within_one_seed_all_three_arms_share_the_schedule_seed(three_arms):
    _, _, paths = three_arms
    got = {arm: load(p)["schedule_seed"] for arm, p in paths.items()}
    assert len(set(got.values())) == 1
    assert got["ctrl"] == derive_schedule_seed(19)


# ==================================================  3. job contract  =====

def test_job_maps_three_arms_four_seeds():
    t = script()
    assert "#SBATCH --array=0-11" in t
    assert "ARMS=(ctrl ctrl ctrl ctrl 5e5 5e5 5e5 5e5 3e5 3e5 3e5 3e5)" in t
    assert "LR_CS=(1e-4 1e-4 1e-4 1e-4 5e-5 5e-5 5e-5 5e-5 3e-5 3e-5 3e-5 3e-5)" in t
    assert "SEEDS=(19 20 21 22 19 20 21 22 19 20 21 22)" in t
    assert 'RUN_ID="final_settle_${ARM}_h512_s${SEED}"' in t


def test_job_lowers_c_and_holds_everything_else():
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
                      "curriculum", "1.5e-4", "2e-4", "1e-5,"):
        assert forbidden not in ex, forbidden
    assert "CEILING_REQUIRED=2" in t   # Amendment 2: confirmed = 2 consecutive
    assert "EVAL_EVERY=13890" in t and "SAVE_EVERY=69450" in t
    assert "--ceiling-consecutive-required" in ex and "--stop-at-ceiling" in ex


def test_job_reanchors_every_arm_including_the_control():
    t = script()
    assert "EXTRA=(--reanchor-schedule --phase-transition)" in t
    assert "re-anchoring the macro-cycle at $SOURCE_STEP in ALL arms" in t


def test_job_grid_and_source():
    t = script()
    assert "SOURCE_STEP=8334000" in t and "MAX_STEPS=10000800" in t
    assert "MS_STEP=69450" in t and "seq 1 24" in t
    assert 'PARENT_RUN_ID="final_chigh_ctrl_u3000_h512_s${SEED}"' in t
    assert 'int(c["repetition"]) == 1389000' in t
    assert 'abs(float(p["comprehension"]) - 1e-4) < 1e-12' in t
    assert 'int(ck.get("schedule_anchor_step")) == 5556000' in t
    assert 'seed * 1000003 + 4' in t
    assert "91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed" in t
    assert "95295d63560ae4c235a6beee8dfb47166f4ed30d" in t
    # grid arithmetic
    assert SRC_STEP == 3000 * R_PASS * CYCLE
    assert END_STEP == 3600 * R_PASS * CYCLE
    assert (END_STEP - SRC_STEP) // MS_STEP == 24
    d = END_STEP - SRC_STEP
    assert d // CYCLE == 277_800 and d // CYCLE // R_PASS == 600
    assert (1_389_000 + d // CYCLE, 2_778_000 + 2 * (d // CYCLE),
            4_167_000 + 3 * (d // CYCLE)) == (1_666_800, 3_333_600, 5_000_400)
    for k in range(1, 25):
        assert (SRC_STEP + MS_STEP * k) % 69450 == 0


def test_job_protects_every_parent_lineage():
    t = script()
    for pat in ("final_base123_*", "final_lrpilot*", "final_rep_*",
                "final_canneal_*", "final_chigh_*"):
        assert pat in t, pat
    assert "READ ONLY" in t
    assert "final_settle_ctrl_h512_s19" in t
    assert "final_settle_3e5_h512_s22" in t


def test_report_constants_match_the_job():
    from scripts.naming_comprehension import settle_report as m
    assert m.SRC_STEP == SRC_STEP and m.END_STEP == END_STEP
    assert m.MS_STEP == MS_STEP and m.N_MILESTONES == 24
    assert m.ARM_LR == ARM_LR_F
    assert m.RUN == "final_settle_{arm}_h512_s{seed}"
    assert m.milestones()[0] == 8_403_450 and m.milestones()[-1] == END_STEP
    assert m.CANNEAL_U1400_DC == {"5e5": 7.75, "3e5": 10.5}


# ==============================================  4. the report itself  =====

def _fixture(root, c_errors, sd_scale=None):
    """Synthetic metrics at the real milestones.  `c_errors(arm, seed, k)`
    gives the count; sd_scale lets an arm carry visibly larger jitter."""
    cols = ["step", "r_exposures", "n_exposures", "c_exposures", "lr",
            "full_rep_ltm", "full_rep_full", "full_rep_wm", "full_comp_top1",
            "full_comp_top5", "full_naming_exact", "full_rep_freear",
            "full_rep_errors", "full_rep_freear_errors", "full_comp_errors",
            "full_naming_errors", "gate_mean"]
    for arm in ARM_LR:
        for seed in SEEDS:
            d = os.path.join(root, f"final_settle_{arm}_h512_s{seed}")
            os.makedirs(d)
            with open(os.path.join(d, "metrics.tsv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
                w.writeheader()
                for k in range(1, 25):
                    step = SRC_STEP + MS_STEP * k
                    u = step / (R_PASS * CYCLE)
                    ce = c_errors(arm, seed, k)
                    w.writerow({"step": step - 100, "lr": 3e-5})
                    w.writerow({
                        "step": step, "lr": 3e-5, "r_exposures": u,
                        "n_exposures": 2 * u,
                        "c_exposures": 3 * u * R_PASS / C_PASS,
                        "full_rep_ltm": 0.88, "full_rep_wm": 1.0,
                        "gate_mean": 0.52,
                        "full_rep_full": 1.0, "full_rep_errors": 0,
                        "full_rep_freear": 1.0, "full_rep_freear_errors": 0,
                        "full_naming_exact": 1.0, "full_naming_errors": 0,
                        "full_comp_top1": 1 - ce / 27981,
                        "full_comp_errors": ce, "full_comp_top5": 0.9995})


def _run_report(runs, out):
    from scripts.naming_comprehension import settle_report as m
    assert m.main(["--runs-root", runs, "--out-dir", out]) == 0
    got = {}
    for name in ("settle_by_milestone", "settle_primary_smoothed",
                 "settle_variance", "settle_paired", "cursor_identity_proof"):
        p = os.path.join(out, name + ".tsv")
        txt = open(p, encoding="utf-8").read() if os.path.exists(p) else ""
        got[name] = list(csv.DictReader(txt.splitlines(), delimiter="\t")) \
            if txt.strip() else []
    got["meta"] = json.load(open(os.path.join(out, "settle_meta.json")))
    return got


def test_report_primary_is_smoothed_and_flags_the_sign_flip(tmp_path):
    """Settle arms end lower AND the same treatment lost at u1400 -> the
    report must call the sign flip out explicitly."""
    runs, out = str(tmp_path / "r"), str(tmp_path / "o")
    _fixture(runs, lambda arm, s, k:
             {"ctrl": 37, "5e5": 30, "3e5": 24}[arm] + (k % 3))
    g = _run_report(runs, out)
    sm = {r["arm"]: r for r in g["settle_primary_smoothed"]}
    assert float(sm["5e5"]["dC_last4_mean"]) < 0
    assert float(sm["3e5"]["dC_last4_mean"]) < 0
    assert sm["5e5"]["sign_flipped_vs_u1400"] == "1"
    assert sm["3e5"]["canneal_u1400_dC"] == "10.5"
    assert sm["5e5"]["better_seeds"] == "4"
    assert g["meta"]["primary_smoothed"][0]["milestones"] == "21,22,23,24"


def test_report_does_not_claim_a_flip_when_settle_loses(tmp_path):
    runs, out = str(tmp_path / "r"), str(tmp_path / "o")
    _fixture(runs, lambda arm, s, k:
             {"ctrl": 37, "5e5": 44, "3e5": 47}[arm] + (k % 3))
    g = _run_report(runs, out)
    sm = {r["arm"]: r for r in g["settle_primary_smoothed"]}
    assert float(sm["5e5"]["dC_last4_mean"]) > 0
    assert sm["5e5"]["sign_flipped_vs_u1400"] == "0"
    assert sm["3e5"]["sign_flipped_vs_u1400"] == "0"


def test_report_measures_the_variance_collapse(tmp_path):
    """Low arm: same mean, much smaller milestone-to-milestone jitter."""
    runs, out = str(tmp_path / "r"), str(tmp_path / "o")
    def ce(arm, s, k):
        base = 37
        jitter = {"ctrl": 6, "5e5": 3, "3e5": 0}[arm]
        return base + (jitter if k % 2 else -jitter)
    _fixture(runs, ce)
    g = _run_report(runs, out)
    v = {r["arm"]: r for r in g["settle_variance"]}
    assert float(v["ctrl"]["C_sd_mean"]) > float(v["5e5"]["C_sd_mean"]) \
        > float(v["3e5"]["C_sd_mean"])
    assert float(v["3e5"]["C_sd_ratio_vs_ctrl"]) < 0.1
    assert v["ctrl"]["C_sd_ratio_vs_ctrl"] == "1.0"
    # frozen-but-stable: means equal, so the smoothed primary shows ~0
    sm = {r["arm"]: r for r in g["settle_primary_smoothed"]}
    assert abs(float(sm["3e5"]["dC_last4_mean"])) < 3.1


def test_report_survives_a_run_that_never_started(tmp_path):
    runs, out = str(tmp_path / "r"), str(tmp_path / "o")
    _fixture(runs, lambda arm, s, k: 37)
    os.remove(os.path.join(runs, "final_settle_5e5_h512_s21", "metrics.tsv"))
    g = _run_report(runs, out)
    assert g["meta"]["missing_rows"] == 24
    sm = {r["arm"]: r for r in g["settle_primary_smoothed"]}
    assert sm["5e5"]["n_seeds"] == "3"
    assert sm["3e5"]["n_seeds"] == "4"
    # preregistered missing-run policy: 3/4 seeds is descriptive only
    assert sm["5e5"]["formal_decision_evaluable"] == "0"
    assert sm["3e5"]["formal_decision_evaluable"] == "1"


# ==================  5. the mechanical preregistered decision (Amend. 1)  ==

def _decision(runs, out):
    from scripts.naming_comprehension import settle_report as m
    assert m.main(["--runs-root", runs, "--out-dir", out]) == 0
    return json.load(open(os.path.join(out, "settle_decision.json")))


def test_branch_supported_when_a_settle_arm_wins(tmp_path):
    runs, out = str(tmp_path / "r"), str(tmp_path / "o")
    _fixture(runs, lambda arm, s, k:
             {"ctrl": 37, "5e5": 30, "3e5": 24}[arm] + (k % 3))
    d = _decision(runs, out)
    assert d["complete"] is True
    assert d["branch"] == "STATE_DEPENDENT_SETTLING_SUPPORTED"
    assert d["winner"] == "3e5"          # lower paired_mean, gap > 1.0
    assert d["arms"]["3e5"]["eligible_primary_winner"] is True
    assert d["arms"]["3e5"]["sign_flipped_vs_u1400"] == 1
    assert d["arms"]["3e5"]["guards"]["all_ok"] is True


def test_branch_structural_fixed_tolerance_not_paired_sd(tmp_path):
    """Same mean as control, collapsed variance: the FIXED |pm| <= 1.0 band
    plus variance_ratio <= 0.5, never a paired_sd-widened band."""
    runs, out = str(tmp_path / "r"), str(tmp_path / "o")
    def ce(arm, s, k):
        jitter = {"ctrl": 6, "5e5": 3, "3e5": 0}[arm]
        return 37 + (jitter if k % 2 else -jitter)
    _fixture(runs, ce)
    d = _decision(runs, out)
    assert d["branch"] == "STRUCTURAL_TAIL_SUPPORTED"
    assert d["structural_arm"] == "3e5"  # lower variance_ratio
    a = d["arms"]["3e5"]
    assert abs(a["paired_mean"]) <= 1.0
    assert a["variance_ratio"] <= 0.5
    assert a["eligible_primary_winner"] is False


def test_branch_rejected_when_both_arms_materially_worse(tmp_path):
    runs, out = str(tmp_path / "r"), str(tmp_path / "o")
    _fixture(runs, lambda arm, s, k:
             {"ctrl": 37, "5e5": 44, "3e5": 47}[arm] + (k % 3))
    d = _decision(runs, out)
    assert d["branch"] == "STATE_DEPENDENT_SETTLING_REJECTED"
    assert d["arms"]["5e5"]["paired_mean"] > 1.0
    assert d["arms"]["3e5"]["paired_mean"] > 1.0


def test_branch_mixed_is_reachable_not_shadowed(tmp_path):
    """One arm clearly better on the mean but only 2/4 seeds and no variance
    collapse; the other materially worse.  Branch 4 must NOT swallow it."""
    runs, out = str(tmp_path / "r"), str(tmp_path / "o")
    def ce(arm, s, k):
        base = 37 + (k % 3)
        if arm == "ctrl":
            return base
        if arm == "5e5":
            return base - 8 if s in (19, 20) else base + 2   # pm = -3, 2/4
        return base + 5                                       # pm = +5
    _fixture(runs, ce)
    d = _decision(runs, out)
    a = d["arms"]["5e5"]
    assert a["paired_mean"] == -3.0 and a["better_seeds"] == 2
    assert a["eligible_primary_winner"] is False
    assert a["structural_candidate"] is False    # |pm| > 1.0
    assert d["branch"] == "MIXED_SETTLE_THEN_AUDIT"


def test_branch_ceiling_confirmed_by_exactly_two_consecutive_zeros(tmp_path):
    """Amendment 2: TWO consecutive 0/0/0/0 full evaluations confirm and
    outrank everything, even an eligible settle winner."""
    runs, out = str(tmp_path / "r"), str(tmp_path / "o")
    def ce(arm, s, k):
        if arm == "3e5" and s == 19 and k in (23, 24):
            return 0                              # exactly two consecutive
        return {"ctrl": 37, "5e5": 30, "3e5": 24}[arm] + (k % 3)
    _fixture(runs, ce)
    d = _decision(runs, out)
    assert d["any_ceiling_hit"] is True
    assert d["max_ceiling_streak"] == 2
    assert d["ceiling_confirmed"] is True
    assert d["any_stability_5"] is False
    assert d["branch"] == "CEILING_CONFIRMED"
    assert d["confirmed_runs"] == [["3e5", 19]]
    assert d["first_confirmation"][3] == SRC_STEP + MS_STEP * 24


def test_single_unconfirmed_hit_does_not_shadow_the_tree(tmp_path):
    """One isolated perfect evaluation is a recorded CEILING_HIT, never a
    stop, never a branch: the normal scientific tree proceeds."""
    runs, out = str(tmp_path / "r"), str(tmp_path / "o")
    def ce(arm, s, k):
        if arm == "3e5" and s == 19 and k == 10:
            return 0                              # isolated hit, then resets
        return {"ctrl": 37, "5e5": 30, "3e5": 24}[arm] + (k % 3)
    _fixture(runs, ce)
    d = _decision(runs, out)
    assert d["any_ceiling_hit"] is True
    assert d["max_ceiling_streak"] == 1
    assert d["ceiling_confirmed"] is False
    assert d["branch"] == "STATE_DEPENDENT_SETTLING_SUPPORTED"
    assert d["winner"] == "3e5"


def test_confirmed_early_stop_beats_incomplete(tmp_path):
    """A run that stops on a confirmed ceiling leaves later milestones
    missing BY DESIGN; the confirmation must not be buried under
    INCOMPLETE_FOR_PREREGISTERED_DECISION."""
    runs, out = str(tmp_path / "r"), str(tmp_path / "o")
    def ce(arm, s, k):
        if arm == "3e5" and s == 19 and k in (1, 2):
            return 0                              # confirmed at k=2 ...
        return {"ctrl": 37, "5e5": 30, "3e5": 24}[arm] + (k % 3)
    _fixture(runs, ce)
    # ... then the run stopped after k=2: even the deepest-common fallback
    # cannot build a 4-milestone smoothed primary from 2 milestones
    q = os.path.join(runs, "final_settle_3e5_h512_s19", "metrics.tsv")
    kept = [r for r in csv.DictReader(open(q), delimiter="	")
            if int(r["step"]) <= SRC_STEP + MS_STEP * 2]
    with open(q, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(kept[0]), delimiter="	")
        w.writeheader(); [w.writerow(r) for r in kept]
    d = _decision(runs, out)
    assert d["complete"] is False
    assert d["branch"] == "CEILING_CONFIRMED"
    assert d["confirmed_runs"] == [["3e5", 19]]


def test_stability_5_is_reported_but_not_required(tmp_path):
    runs, out = str(tmp_path / "r"), str(tmp_path / "o")
    def ce(arm, s, k):
        if arm == "3e5" and s == 19 and k >= 19:
            return 0                              # streak reaches 6
        return {"ctrl": 37, "5e5": 30, "3e5": 24}[arm] + (k % 3)
    _fixture(runs, ce)
    d = _decision(runs, out)
    assert d["branch"] == "CEILING_CONFIRMED"
    assert d["any_stability_5"] is True
    assert d["max_ceiling_streak"] == 6


def test_branch_incomplete_precedes_all_branches(tmp_path):
    runs, out = str(tmp_path / "r"), str(tmp_path / "o")
    _fixture(runs, lambda arm, s, k:
             {"ctrl": 37, "5e5": 30, "3e5": 24}[arm] + (k % 3))
    os.remove(os.path.join(runs, "final_settle_5e5_h512_s21", "metrics.tsv"))
    d = _decision(runs, out)
    assert d["complete"] is False
    assert d["branch"] == "INCOMPLETE_FOR_PREREGISTERED_DECISION"
    assert "winner" not in d


def test_branches_partition_every_complete_outcome():
    """Exhaustiveness at the logic level: after CEILING/SUPPORTED/STRUCTURAL
    fail, 'both pm > 1.0' and 'min pm <= 1.0' are complements, so
    NO_CLEAR_DECISION is unreachable for complete numeric data."""
    for pm5 in (-5.0, -1.0, 0.0, 1.0, 1.01, 7.0):
        for pm3 in (-5.0, -1.0, 0.0, 1.0, 1.01, 7.0):
            rejected = pm5 > 1.0 and pm3 > 1.0
            mixed = min(pm5, pm3) <= 1.0
            assert rejected != mixed
