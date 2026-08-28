"""Consistency tests for the two Yair meeting presentation figures.

These figures must not drift from the validated tables they report, so each
annotated number is re-derived here from its source table and compared.
No model is loaded and no scientific result is recomputed.
"""
from __future__ import annotations

import ast
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.behavioral_analysis.common import SEEDS                 # noqa: E402

YC = os.path.join(ROOT, "reports/behavioral_wfe_fulllexicon_93a577f/"
                        "yair_corrections")
RES = os.path.join(YC, "residual_trained_real", "tables")
SZ = os.path.join(YC, "stable_zero_audit")
FIG = os.path.join(YC, "meeting_figures")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(FIG), reason="meeting figures not built")


def _cap(stem: str) -> str:
    with open(os.path.join(FIG, f"{stem}_caption.md")) as f:
        return f.read()


def test_both_figures_and_captions_exist():
    for stem in ("mf1_trained_real_ltm_errors", "mf2_stable_zero_bottom_line"):
        for ext in ("png", "pdf", "svg"):
            assert os.path.exists(os.path.join(FIG, f"{stem}.{ext}")), stem
        assert os.path.exists(os.path.join(FIG, f"{stem}_caption.md")), stem


# ------------------------------------------------------------------ mf1

def test_mf1_annotation_numbers_match_the_source_tables():
    items = pd.read_csv(os.path.join(RES, "residual_trained_real_items.tsv"),
                        sep="\t")
    summ = pd.read_csv(os.path.join(RES, "residual_trained_real_summary.tsv"),
                       sep="\t").iloc[0]
    assert len(items) == 12
    assert int(summ["n_trained_real_items"]) == 671
    assert int(summ["n_error_events_seed_x_item"]) == 14
    assert int(summ["n_below_peer_median"]) == 11
    assert int((items["phoneme_length"] == 9).sum()) == 6
    assert round(float(summ["mean_within_length_zipf_percentile"]), 3) == 0.188
    assert float(summ["permutation_p_one_sided_lower"]) == 0.00015
    cap = _cap("mf1_trained_real_ltm_errors")
    for s in ("671", "12", "14", "11", "0.188", "0.00015"):
        assert s in cap, s


def test_mf1_labels_every_one_of_the_12_words():
    items = pd.read_csv(os.path.join(RES, "residual_trained_real_items.tsv"),
                        sep="\t")
    with open(os.path.join(ROOT, "scripts/make_meeting_figures.py")) as f:
        src = f.read()
    # words are drawn from the table, never hardcoded
    assert 'r["word"]' in src
    assert items["word"].notna().all() and items["word"].nunique() == 12


def test_mf1_population_is_the_671_trained_real_items():
    canon = pd.read_csv(os.path.join(
        ROOT, "outputs/behavioral_wfe_fulllexicon_93a577f/behavioral_analysis/"
              "tables/canonical_behavioral_item_table.tsv"), sep="\t")
    pop = canon[(canon["route"] == "ltm")
                & (canon["lichtheim_exposure_status"] == "TRAINED_REAL_EXACT")
                & (canon["seed"] == SEEDS[0])]
    assert len(pop) == 671


def test_mf1_marker_size_encodes_only_one_or_two_failing_seeds():
    items = pd.read_csv(os.path.join(RES, "residual_trained_real_items.tsv"),
                        sep="\t")
    assert set(items["n_failing_seeds"]) <= {1, 2}
    assert int(items["n_failing_seeds"].sum()) == 14


# ------------------------------------------------------------------ mf2

def test_mf2_streak_lengths_match_the_audit():
    streaks = pd.read_csv(os.path.join(SZ, "stable_zero_streaks.tsv"), sep="\t")
    longest = {int(s): int(g["length"].max())
               for s, g in streaks.groupby("seed")}
    for s in SEEDS:
        longest.setdefault(s, 0)
    assert longest == {19: 6, 20: 2, 21: 0, 22: 13}
    s19 = streaks[(streaks.seed == 19) & (streaks.length == 6)].iloc[0]
    assert (int(s19["first_checkpoint_of_streak"]),
            int(s19["last_checkpoint_of_streak"])) == (155, 180)
    s22 = streaks[(streaks.seed == 22) & (streaks.length == 13)].iloc[0]
    assert (int(s22["first_checkpoint_of_streak"]),
            int(s22["last_checkpoint_of_streak"])) == (140, 200)


def test_mf2_x_criterion_pass_counts():
    v = pd.read_csv(os.path.join(SZ, "stable_zero_verdicts.tsv"), sep="\t")
    assert v.groupby("X")["criterion_met"].sum().to_dict() == {2: 3, 3: 2, 5: 2}
    cap = _cap("mf2_stable_zero_bottom_line")
    for s in ("3/4", "2/4", "155-180", "130-135", "140-200"):
        assert s in cap, s


def test_mf2_selected_and_stop_epochs_are_distinct_where_x_is_large():
    v = pd.read_csv(os.path.join(SZ, "stable_zero_verdicts.tsv"), sep="\t")
    r = v[(v.seed == 22) & (v.X == 5)].iloc[0]
    assert int(r["selected_epoch"]) == 140
    assert int(r["stop_epoch_earliest_knowable"]) == 160
    # raising X must not move the selected checkpoint for seeds 19 and 22
    for s in (19, 22):
        sel = v[(v.seed == s) & v["criterion_met"]]["selected_epoch"].unique()
        assert len(sel) == 1, s


def test_mf2_trajectory_grid_is_complete():
    t = pd.read_csv(os.path.join(SZ, "stable_zero_trajectory.tsv"), sep="\t")
    assert sorted(t["seed"].unique()) == SEEDS
    for s in SEEDS:
        e = sorted(t[t["seed"] == s]["epoch"])
        assert len(e) == 20 and e[0] == 105 and e[-1] == 200
        assert set(np.diff(e).tolist()) == {5}


# ------------------------------------------------------------- provenance

def test_meeting_figures_load_no_model_and_recompute_nothing():
    path = os.path.join(ROOT, "scripts/make_meeting_figures.py")
    with open(path) as f:
        tree = ast.parse(f.read())
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module.split(".")[0])
    assert "torch" not in imported
    with open(os.path.join(FIG, "provenance.json")) as f:
        p = json.load(f)
    assert p["model_loaded"] is False
    assert p["inference_run"] is False
    assert p["scientific_values_recomputed"] is False
    assert p["annotations_asserted_against_source_tables"] is True
