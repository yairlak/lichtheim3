"""Validation for the Phase 4A3d factorial synthesis.

Checks the derived analysis table against the original run artifacts. These are
analysis tests: they read metrics files and the derived TSVs, and never load a
model or run training. They are skipped when the synthesis has not been
generated yet.
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

from scripts.naming_comprehension.analyze_joint_factorial import (          # noqa: E402
    CONDITIONS, FINAL_EPOCH, FINAL_STEP, LEXICON_N, ORDER, errors,
    load_condition, ltm_error_decomposition, contrasts, endpoint,
)

REPORT = os.path.join(ROOT, "reports", "joint_scratch_factorial_seed22")
TABLE = os.path.join(REPORT, "data", "factorial_trajectories_seed22.tsv")

pytestmark = pytest.mark.skipif(
    not os.path.exists(TABLE),
    reason="run analyze_joint_factorial.py first to generate the synthesis")


@pytest.fixture(scope="module")
def derived():
    with open(TABLE, encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


@pytest.fixture(scope="module")
def loaded():
    return {c: load_condition(c, 22) for c in ORDER}


def test_four_conditions_present(derived):
    assert {r["condition"] for r in derived} == set(ORDER)
    assert len(ORDER) == 4


def test_seed_and_regimes(derived):
    assert {r["seed"] for r in derived} == {"22"}
    expected = {"h0": (False, False), "c_only": (True, False),
                "n_only": (False, True), "j0": (True, True)}
    for cond, spec in CONDITIONS.items():
        assert expected[spec["regime"]] == (spec["retrieval"], spec["naming"])


def test_every_condition_ends_at_e440(loaded):
    for c in ORDER:
        assert loaded[c][-1]["rep_epoch"] == FINAL_EPOCH
        assert loaded[c][-1]["step"] == FINAL_STEP


def test_no_duplicate_steps_and_no_e160_duplication(loaded):
    for c in ORDER:
        steps = [r["step"] for r in loaded[c]]
        assert steps == sorted(steps), f"{c}: steps not monotonic"
        assert len(steps) == len(set(steps)), f"{c}: duplicate step in trajectory"
    # the H0/J0 stitch point must appear exactly once
    for c in ("H0", "J0"):
        assert sum(1 for r in loaded[c] if r["step"] == 74080) == 1


def test_schedules_are_aligned(loaded):
    ref = [r["step"] for r in loaded["H0"]]
    for c in ORDER:
        assert [r["step"] for r in loaded[c]] == ref, f"{c}: schedule differs"


def test_endpoint_matches_original_metrics_files(loaded):
    """Derived endpoint values must equal the last row of the source run."""
    for c in ORDER:
        src = os.path.join(ROOT, CONDITIONS[c]["dirs"][-1], "metrics.tsv")
        with open(src, encoding="utf-8") as fh:
            last = list(csv.DictReader(fh, delimiter="\t"))[-1]
        got = loaded[c][-1]
        for src_col, dst_col in (("comp_top1", "comp_top1"),
                                 ("comp_top5", "comp_top5"),
                                 ("naming_exact", "naming_exact"),
                                 ("rep_ltm", "rep_ltm"),
                                 ("full_rep_ltm", "full_rep_ltm"),
                                 ("full_rep_wm", "full_rep_wm"),
                                 ("full_rep_full", "full_rep_full")):
            assert got[dst_col] == pytest.approx(float(last[src_col])), \
                f"{c}: {dst_col} disagrees with {src_col} in the source file"


def test_full_lexicon_error_counts(loaded):
    end = endpoint(loaded)
    expected = {"H0": (2, 2, 522), "C-only": (1, 2, 1217),
                "N-only": (1, 15, 932), "J0": (2, 1, 1635)}
    for c in ORDER:
        got = tuple(errors(end[c][k], LEXICON_N)
                    for k in ("full_rep_full", "full_rep_wm", "full_rep_ltm"))
        assert got == expected[c], f"{c}: endpoint errors {got} != {expected[c]}"


def test_interaction_formula_recomputes(loaded):
    d = ltm_error_decomposition(endpoint(loaded))
    assert d["retrieval_cost_naming_off"] == d["E_C_only"] - d["E_H0"]
    assert d["naming_cost_retrieval_off"] == d["E_N_only"] - d["E_H0"]
    assert d["additive_prediction"] == (d["retrieval_cost_naming_off"]
                                        + d["naming_cost_retrieval_off"])
    assert d["observed_J0_cost"] == d["E_J0"] - d["E_H0"]
    assert d["interaction"] == d["observed_J0_cost"] - d["additive_prediction"]
    assert d["interaction"] == (d["E_J0"] - d["E_C_only"]
                                - d["E_N_only"] + d["E_H0"])
    # the value pre-registered in Phase 4A3c before N-only was observed
    assert d["interaction"] == 940 - d["E_N_only"]
    assert d["interaction"] == 8


def test_accuracy_contrasts_are_self_consistent(loaded):
    end = endpoint(loaded)
    for metric in ("comp_top1", "naming_exact", "rep_ltm", "full_rep_ltm"):
        k = contrasts(end, metric)
        assert k["retrieval_effect_naming_off"] == pytest.approx(k["C_only"] - k["H0"])
        assert k["naming_effect_retrieval_off"] == pytest.approx(k["N_only"] - k["H0"])
        assert k["interaction"] == pytest.approx(
            k["J0"] - k["C_only"] - k["N_only"] + k["H0"])


def test_no_interpolation_of_missing_values(derived):
    """The expensive full-lexicon evaluation ran only where an endpoint was
    requested, and nowhere else may carry a value.

    H0 and J0 have two such points because their e0->e160 segment was itself
    finished with --endpoint-eval; C-only and N-only were single runs and so
    have one. Every other snapshot must stay blank -- never forward-filled.
    """
    expected = {"H0": {74080, FINAL_STEP}, "J0": {74080, FINAL_STEP},
                "C-only": {FINAL_STEP}, "N-only": {FINAL_STEP}}
    seen = {c: set() for c in ORDER}
    for r in derived:
        step, cond = int(r["step"]), r["condition"]
        if r["full_rep_ltm"] != "":
            seen[cond].add(step)
            assert step in expected[cond], \
                f"{cond} step {step}: unexpected full-lexicon value"
    assert seen == expected, f"endpoint coverage {seen} != {expected}"


@pytest.mark.skipif(not os.path.exists(os.path.join(REPORT, "data",
                                                    "wm_audit_summary.json")),
                    reason="WM audit not generated (--wm-audit)")
def test_wm_audit_reproduces_stored_endpoint_counts():
    with open(os.path.join(REPORT, "data", "wm_audit_summary.json"),
              encoding="utf-8") as fh:
        audit = json.load(fh)
    expected = {"H0": (2, 2, 522), "C-only": (1, 2, 1217),
                "N-only": (1, 15, 932), "J0": (2, 1, 1635)}
    for c in ORDER:
        e = audit[c]["errors"]
        assert (e["full"], e["wm"], e["ltm"]) == expected[c], \
            f"{c}: re-evaluated counts disagree with the stored endpoint"
    d = audit["N-only"]["wm_error_detail"]
    assert d["n_wm_errors"] == 15
    assert d["rescued_by_full"] == 15
    assert len(d["words"]) == 15
