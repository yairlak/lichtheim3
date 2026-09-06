"""Tests for the base-123 persistence / switch-trigger report.

The report decides whether the C residual has become a small STABLE near-tie
core (which would justify a margin/hard-negative intervention) or is still a
moving frontier.  A wrong answer here would license the wrong experiment, so
the tests plant known structure and check the report recovers it, and check
that the trigger refuses to fire unless every preregistered condition holds.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.base123_persistence_report import (  # noqa: E402
    N_COMP, describe, main, overlap, switch_trigger,
)

JOB = "scripts/cluster/jeanzay/base123_error_audit.slurm"
SEEDS = [19, 20, 21, 22]
US = [500, 600, 750]
CORE = list(range(1000, 1027))                       # 27 planted items

COLS = ["task", "target_bank_index", "target_word", "target_rank", "in_top5",
        "pred_word", "margin_target_minus_top1", "pred_target_glove_cos",
        "target_cos", "is_homophone_of_target", "identical_glove_vectors",
        "mathematically_unavoidable", "relation"]


def script(path=JOB):
    return open(os.path.join(ROOT, path), encoding="utf-8").read()


def build(root, sizes, near_frac=1.0, core=CORE, seed0=0):
    """Plant `core` in every seed/milestone plus disjoint idiosyncratic items."""
    import random
    rng = random.Random(seed0)
    for k, s in enumerate(SEEDS):
        for j, u in enumerate(US):
            n_idio = sizes[u] - len(core)
            # Disjoint per (seed, milestone), so the ONLY overlap anywhere is
            # the planted core and the assertions can be exact.
            lo = 20000 + k * 200000 + j * 50000
            idio = rng.sample(range(lo, lo + 40000), n_idio)
            d = os.path.join(root, f"final_base123_h512_s{s}",
                             f"error_audit_u{u}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "comp_errors.tsv"), "w",
                      newline="") as f:
                w = csv.DictWriter(f, fieldnames=COLS, delimiter="\t")
                w.writeheader()
                for i in list(core) + idio:
                    near = rng.random() < near_frac
                    w.writerow({
                        "task": "comprehension", "target_bank_index": i,
                        "target_word": f"w{i}", "target_rank": 2 if near else 9,
                        "in_top5": 1 if near else 0, "pred_word": f"w{i+1}",
                        "margin_target_minus_top1": -0.005 if near else -0.09,
                        "pred_target_glove_cos": 0.8, "target_cos": 0.7,
                        "is_homophone_of_target": 0,
                        "identical_glove_vectors": 0,
                        "mathematically_unavoidable": 0,
                        "relation": "semantic_neighbour" if near else "other"})


def load_report(out):
    return json.load(open(os.path.join(out, "persistence_report.json")))


def rows(out, name):
    with open(os.path.join(out, name), encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


# ================================================  set arithmetic  =========

def test_overlap_reports_every_requested_quantity():
    a = {str(i): {} for i in range(100)}
    b = {str(i): {} for i in range(60, 200)}
    o = overlap(a, b, N_COMP)
    assert o["n_a"] == 100 and o["n_b"] == 140
    assert o["intersection"] == 40 and o["union"] == 200
    assert o["jaccard"] == pytest.approx(0.2)
    assert o["frac_a_retained"] == pytest.approx(0.4)
    assert o["resolved"] == 60 and o["new"] == 100
    # chance model: |A|*|B|/N
    assert o["expected_intersection_if_independent"] == pytest.approx(
        100 * 140 / N_COMP, abs=1e-3)
    assert o["enrichment_over_chance"] > 1


def test_disjoint_and_identical_sets_are_handled():
    a = {str(i): {} for i in range(10)}
    b = {str(i): {} for i in range(10, 20)}
    o = overlap(a, b, N_COMP)
    assert o["intersection"] == 0 and o["jaccard"] == 0.0
    assert o["frac_a_retained"] == 0.0 and o["enrichment_over_chance"] == 0.0
    same = overlap(a, dict(a), N_COMP)
    assert same["jaccard"] == 1.0 and same["frac_a_retained"] == 1.0
    assert same["new"] == 0


# ==========================================  planted-core recovery  ========

def test_report_recovers_the_planted_core_and_its_fate(tmp_path):
    root, out = str(tmp_path / "runs"), str(tmp_path / "rep")
    build(root, {500: 520, 600: 340, 750: 235})
    assert main(["--runs-root", root, "--out-dir", out,
                 "--last-window-ratio", "0.878"]) == 0
    r = load_report(out)
    assert r["sentinel_core_size"] == len(CORE) == 27, \
        "seed-disjoint idiosyncratic items must not inflate the core"
    assert sorted(r["sentinel_core_items"]) == [str(i) for i in CORE]
    for u in US:
        fate = r["sentinel_core_fate"][str(u)]
        assert fate["wrong_in_all_four"] == 27
        assert all(v == 27 for v in fate["per_seed_still_wrong"].values())
    # the four-way intersection at u750 IS the core here
    assert len(r["comprehension_four_way_intersection_750"]) == 27


def test_across_seed_chance_model_and_enrichment(tmp_path):
    root, out = str(tmp_path / "runs"), str(tmp_path / "rep")
    build(root, {500: 520, 600: 340, 750: 235})
    assert main(["--runs-root", root, "--out-dir", out]) == 0
    ac = [r for r in rows(out, "across_seed_overlap.tsv")
          if r["task"] == "comprehension"]
    pair = [r for r in ac if r["seed_a"] != "ALL"]
    assert len(pair) == 6, "all six pairwise comparisons"
    for r in pair:
        # seeds share exactly the planted core and nothing else
        assert int(r["intersection"]) == 27
        exp = float(r["expected_intersection_if_independent"])
        assert exp == pytest.approx(235 * 235 / N_COMP, abs=1e-2)
        assert float(r["enrichment_over_chance"]) > 5
    four = [r for r in ac if r["seed_a"] == "ALL"]
    assert len(four) == 1
    assert int(four[0]["intersection"]) == 27
    assert int(four[0]["union"]) == 27 + 4 * (235 - 27)


def test_within_seed_chain_covers_all_three_transitions(tmp_path):
    root, out = str(tmp_path / "runs"), str(tmp_path / "rep")
    build(root, {500: 520, 600: 340, 750: 235})
    assert main(["--runs-root", root, "--out-dir", out]) == 0
    ch = [r for r in rows(out, "within_seed_persistence.tsv")
          if r["task"] == "comprehension"]
    got = {(r["from_u"], r["to_u"]) for r in ch}
    assert got == {("500", "600"), ("500", "750"), ("600", "750")}
    for r in ch:
        assert int(r["intersection"]) == 27      # only the core persists
        assert int(r["resolved"]) == int(r["n_a"]) - 27
        assert int(r["new"]) == int(r["n_b"]) - 27


# ==============================================  the switch trigger  =======

def test_trigger_requires_every_condition():
    good = {"frac_in_top5": 0.97, "frac_margin_within_0.01": 0.72}
    t = switch_trigger(good, [0.85, 0.83, 0.86, 0.84], 0.90, 1.4)
    assert t["ALL_CONDITIONS_MET"] and "JUSTIFIED" in t["verdict"]
    # each condition alone can veto
    for kw in (dict(ratio=0.60), dict(ret=[0.4] * 4),
               dict(desc={"frac_in_top5": 0.5, "frac_margin_within_0.01": 0.2}),
               dict(growth=0.7)):
        t = switch_trigger(kw.get("desc", good), kw.get("ret", [0.85] * 4),
                           kw.get("ratio", 0.90), kw.get("growth", 1.4))
        assert not t["ALL_CONDITIONS_MET"], kw
        assert "NOT met" in t["verdict"]


def test_trigger_is_not_met_on_a_moving_frontier(tmp_path):
    """The planted fixture is a churning residual with a tiny fixed core:
    retention is low, so the trigger must refuse."""
    root, out = str(tmp_path / "runs"), str(tmp_path / "rep")
    build(root, {500: 520, 600: 340, 750: 235})
    assert main(["--runs-root", root, "--out-dir", out,
                 "--last-window-ratio", "0.878"]) == 0
    t = load_report(out)["c_switch_trigger"]
    assert t["conditions"]["last_window_ratio_ge_0.85"]["met"]
    assert not t["conditions"]["within_seed_retention_ge_0.80"]["met"]
    assert not t["ALL_CONDITIONS_MET"]


def test_trigger_fires_on_a_genuinely_stable_near_tie_core(tmp_path):
    """A residual that barely changes between milestones and is all near-tie
    must satisfy every condition."""
    root, out = str(tmp_path / "runs"), str(tmp_path / "rep")
    stable = list(range(1000, 1240))            # 240 items, shared and fixed
    build(root, {500: 250, 600: 245, 750: 242}, near_frac=1.0, core=stable)
    assert main(["--runs-root", root, "--out-dir", out,
                 "--sentinel-u", "500", "--last-window-ratio", "0.95"]) == 0
    t = load_report(out)["c_switch_trigger"]
    assert t["conditions"]["last_window_ratio_ge_0.85"]["met"]
    assert t["conditions"]["within_seed_retention_ge_0.80"]["met"]
    assert t["conditions"]["survivors_predominantly_near_tie_or_top5"]["met"]


def test_missing_last_window_ratio_cannot_silently_pass():
    t = switch_trigger({"frac_in_top5": 0.99, "frac_margin_within_0.01": 0.9},
                       [0.9] * 4, None, 2.0)
    assert not t["conditions"]["last_window_ratio_ge_0.85"]["met"]
    assert not t["ALL_CONDITIONS_MET"]


# ==================================================  descriptions  =========

def test_describe_reports_the_required_c_diagnostics():
    rows_ = [{"target_rank": "2", "in_top5": "1", "target_cos": "0.7",
              "margin_target_minus_top1": "-0.005", "relation": "morphological",
              "mathematically_unavoidable": "0"},
             {"target_rank": "40", "in_top5": "0", "target_cos": "0.4",
              "margin_target_minus_top1": "-0.2", "relation": "other",
              "mathematically_unavoidable": "0"}]
    d = describe(rows_, "comprehension")
    assert d["n"] == 2 and d["in_top5"] == 1 and d["outside_top5"] == 1
    assert d["rank_median"] == 21 and d["rank_max"] == 40
    assert d["margin_min"] == -0.2 and d["margin_max"] == -0.005
    assert d["n_margin_within_0.01"] == 1
    assert d["relations"] == {"morphological": 1, "other": 1}
    assert d["mathematically_unavoidable"] == 0


def test_describe_handles_naming_and_repetition():
    n = describe([{"edit_distance": "2", "no_eos": "1", "over_generation": "1"}],
                 "naming")
    assert n["edit_median"] == 2 and n["no_eos"] == 1 and n["over_generation"] == 1
    r = describe([{"canonical_exact": "1", "freear_exact_full": "0",
                   "freear_exact_wm": "1", "freear_exact_ltm": "0",
                   "convention_disagrees": "1"}], "repetition")
    assert r["convention_disagreements"] == 1
    assert r["freear_wrong_full"] == 1 and r["ltm_route_wrong"] == 1
    assert r["wm_route_wrong"] == 0


# ================================================  the audit job  ==========

def test_audit_array_targets_u600_and_u750():
    t = script()
    assert "STEPS=(1666800 1666800 1666800 1666800 " \
           "2083500 2083500 2083500 2083500)" in t
    assert "SEEDS=(19 20 21 22 19 20 21 22)" in t
    assert "#SBATCH --array=0-7" in t
    assert "L3_AUDIT_WIDTH:-512" in t


def test_audit_job_is_read_only():
    t = script()
    ex = "\n".join(l for l in t.splitlines()
                   if l.strip() and not l.lstrip().startswith("#"))
    assert "train_joint_scratch.py" not in ex, "the audit must not train"
    assert "base123_error_audit.py" in ex
    assert "error_audit_u" in t
    for forbidden in ("--stop-at-ceiling", "--resume", "--max-steps",
                      "--full-eval-at", "rm ", "mv "):
        assert forbidden not in ex, forbidden
